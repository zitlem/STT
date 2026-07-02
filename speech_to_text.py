import argparse
import os
import sys
import warnings

# Determine application directory (works for both dev and PyInstaller bundle)
# APP_DIR    = user data dir: config, models, logs (script dir in dev, ~/.stt when frozen)
# BUNDLE_DIR = bundled read-only assets: templates, static (_MEIPASS when frozen)
if getattr(sys, 'frozen', False):
    APP_DIR    = os.path.join(os.path.expanduser("~"), ".stt")
    BUNDLE_DIR = sys._MEIPASS
else:
    APP_DIR    = os.path.dirname(os.path.abspath(__file__))
    BUNDLE_DIR = APP_DIR

os.makedirs(APP_DIR, exist_ok=True)
MODELS_DIR = os.path.join(APP_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Default base directory for database + audio backups (rooted in APP_DIR so compiled
# builds keep all data under ~/.stt instead of the launch directory).
BACKUP_DIR = os.path.join(APP_DIR, "_AUTOMATIC_BACKUP")


def _seed_from_bundle(filename):
    """Copy a bundled default file into APP_DIR on first run if missing (compiled builds)."""
    import shutil
    dst = os.path.join(APP_DIR, filename)
    src = os.path.join(BUNDLE_DIR, filename)
    if not os.path.exists(dst) and os.path.exists(src) and src != dst:
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"[INIT] Could not seed {filename} from bundle: {e}")
    return dst


def _chmod_quiet(path, mode):
    """Best-effort chmod; ignore failures (not owner, fs without unix perms, etc.)."""
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def make_db_world_readable(db_path):
    """Make a DB file and its WAL/SHM/journal sidecars world-readable (a+r, 0644),
    so databases written by the service (often as root) can be read by all users
    and downstream consumers."""
    if not db_path:
        return
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = db_path + suffix
        if os.path.exists(p):
            _chmod_quiet(p, 0o644)


def make_dirs_world_readable(leaf_dir, base_dir=None):
    """Make leaf_dir world-readable and traversable (a+rx, 0755). When base_dir is
    given and contains leaf_dir, every directory from base_dir down to leaf_dir is
    updated too, so DB files written inside are reachable by all users."""
    if not leaf_dir:
        return
    leaf = os.path.abspath(leaf_dir)
    if not base_dir:
        _chmod_quiet(leaf, 0o755)
        return
    base = os.path.abspath(base_dir)
    _chmod_quiet(base, 0o755)
    if leaf == base:
        return
    if leaf.startswith(base + os.sep):
        cur = base
        for part in os.path.relpath(leaf, base).split(os.sep):
            cur = os.path.join(cur, part)
            _chmod_quiet(cur, 0o755)
    else:
        _chmod_quiet(leaf, 0o755)


def make_tree_world_readable(root):
    """Recursively make every directory (a+rx, 0755) and file (a+r, 0644) under
    root readable by all users. Best-effort; skips entries it cannot chmod. Used to
    sweep the whole DB/backup folder, including files created during stop cleanup."""
    if not root or not os.path.isdir(root):
        return
    _chmod_quiet(root, 0o755)
    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            _chmod_quiet(os.path.join(dirpath, d), 0o755)
        for f in filenames:
            _chmod_quiet(os.path.join(dirpath, f), 0o644)

# Suppress NNPACK warnings from PyTorch (harmless but spammy)
# These are C++ warnings so we need to disable at the PyTorch level
os.environ['NNPACK_DISABLE'] = '1'
os.environ['PYTORCH_NNPACK_WARN'] = '0'
# macOS: allow fork() after Objective-C threads are initialized (Flask, PyTorch, etc.)
# Without this, multiprocessing with fork start method crashes on macOS
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
warnings.filterwarnings('ignore', message='.*NNPACK.*')

# Set HuggingFace cache to local models directory BEFORE any HF imports
# This prevents models from being downloaded to ~/.cache/huggingface/hub
_models_cache_dir = os.path.join(MODELS_DIR, ".hf_cache")
os.makedirs(_models_cache_dir, exist_ok=True)
os.environ["HF_HUB_CACHE"] = _models_cache_dir
os.environ["HF_HOME"] = _models_cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = _models_cache_dir

# TTS models directory (for piper models)
_tts_cache_dir = os.path.join(MODELS_DIR, "tts")
os.makedirs(_tts_cache_dir, exist_ok=True)

import sqlite3
import logging
import signal
import threading
import multiprocessing
from multiprocessing import Queue as MPQueue

import json
import re
import secrets
import shutil
import statistics
from pathlib import Path
from datetime import timedelta, datetime
from queue import Queue, Empty, Full
from time import sleep
import time
from sys import platform
from file_mover import execute_file_move_now, execute_file_move

# Tracks the most recent file mover run so the UI can show live activity/status.
# Updated by both the automatic (transcription-stop) path and the manual trigger.
_file_mover_runtime_lock = threading.Lock()
file_mover_runtime = {
    "state": "idle",      # idle | running | success | error
    "trigger": None,      # "auto" | "manual"
    "started_at": None,   # ISO timestamp
    "finished_at": None,  # ISO timestamp
    "moved": 0,
    "failed": 0,
    "message": "",
}


def set_file_mover_running(trigger):
    """Mark the file mover as actively running (so the UI shows a live indicator)."""
    with _file_mover_runtime_lock:
        file_mover_runtime.update({
            "state": "running",
            "trigger": trigger,
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "message": "File move in progress...",
        })


def set_file_mover_result(trigger, result):
    """Record the outcome of a completed file mover run."""
    with _file_mover_runtime_lock:
        file_mover_runtime.update({
            "state": "success" if result.get("success") else "error",
            "trigger": trigger,
            "finished_at": datetime.now().isoformat(),
            "moved": result.get("moved", 0),
            "failed": result.get("failed", 0),
            "message": result.get("message", ""),
        })


def get_file_mover_runtime():
    """Return a thread-safe copy of the current file mover runtime status."""
    with _file_mover_runtime_lock:
        return dict(file_mover_runtime)


from flask import Flask, render_template, jsonify, request, redirect, send_from_directory, make_response
from flask_socketio import SocketIO, emit
import speech_recognition as sr
import numpy as np

# Heavy ML imports will be loaded lazily when needed
torch = None
whisper = None
AutoModelForSpeechSeq2Seq = None
AutoProcessor = None
AutoModelForCTC = None
Wav2Vec2Processor = None
pipeline = None
HfApi = None
model_info = None


def _lazy_import_ml_libraries():
    """Import heavy ML libraries only when needed"""
    global \
        torch, \
        whisper, \
        AutoModelForSpeechSeq2Seq, \
        AutoProcessor, \
        AutoModelForCTC, \
        Wav2Vec2Processor, \
        pipeline, \
        HfApi, \
        model_info

    if torch is None:
        import torch as _torch

        torch = _torch
        print("[INFO] PyTorch loaded")

    if whisper is None:
        import whisper as _whisper

        whisper = _whisper
        print("[INFO] Whisper loaded")

    if AutoModelForSpeechSeq2Seq is None:
        from transformers import (
            AutoModelForSpeechSeq2Seq as _AutoModelForSpeechSeq2Seq,
            AutoProcessor as _AutoProcessor,
            AutoModelForCTC as _AutoModelForCTC,
            Wav2Vec2Processor as _Wav2Vec2Processor,
            pipeline as _pipeline,
        )
        from huggingface_hub import HfApi as _HfApi, model_info as _model_info

        AutoModelForSpeechSeq2Seq = _AutoModelForSpeechSeq2Seq
        AutoProcessor = _AutoProcessor
        AutoModelForCTC = _AutoModelForCTC
        Wav2Vec2Processor = _Wav2Vec2Processor
        pipeline = _pipeline
        HfApi = _HfApi
        model_info = _model_info
        print("[INFO] Transformers and HuggingFace Hub loaded")


# Suppress pydub ffmpeg warning - ffmpeg is only needed for specific audio formats
# WAV files (which we use) don't require ffmpeg
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")
# AudioSegment will be imported lazily when needed
AudioSegment = None


def _lazy_import_audio():
    """Import audio processing libraries only when needed"""
    global AudioSegment
    if AudioSegment is None:
        from pydub import AudioSegment as _AudioSegment

        AudioSegment = _AudioSegment
        print("[INFO] AudioSegment loaded")


import tempfile
import uuid
import random


# Whisper decoding parameters optimized for streaming (3s chunks)
LIVE_TRANSCRIPTION_PARAMS = {
    "beam_size": 3,  # Matches config.default.json — accuracy win over greedy, live loop tolerates it
    "best_of": 1,  # No sampling
    "temperature": 0.0,  # Deterministic
    "condition_on_previous_text": False,  # Chunks lack context
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
}

# Whisper decoding parameters optimized for file chunks (30s)
FILE_TRANSCRIPTION_PARAMS = {
    "beam_size": 5,  # Better quality for longer audio
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8),  # Fallback on failure
    "condition_on_previous_text": True,  # Chunks have context
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
}


# Configuration file management
# The canonical default/template config lives in config.default.json (bundled at
# BUNDLE_DIR when compiled). It is used only to seed a fresh config.json on first
# run, or to recover from a missing/corrupted config.json — see load_config().
CONFIG_FILE = os.path.join(APP_DIR, "config.json")
CONFIG_TEMPLATE_FILE = os.path.join(BUNDLE_DIR, "config.default.json")

# Serializes all writers to config.json so concurrent endpoint saves cannot
# interleave and corrupt the file.
_config_file_lock = threading.Lock()


def _atomic_write_json(path, data, *, ensure_ascii=True):
    """Write JSON to ``path`` atomically.

    Dumps to a temp file in the same directory, flushes+fsyncs it, then
    os.replace()s it into place. os.replace is atomic on POSIX and Windows, so
    a crash or concurrent reader never sees a truncated/half-written file.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=ensure_ascii)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_config(config_to_save):
    """Save configuration to config.json atomically with error handling."""
    try:
        with _config_file_lock:
            _atomic_write_json(CONFIG_FILE, config_to_save)
        print(f"[OK] Configuration saved to '{CONFIG_FILE}'")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save config: {e}")
        return False


# Word highlighting uses a separate config file
WORD_HIGHLIGHTING_FILE = _seed_from_bundle("word_highlighting.json")


def load_word_highlighting():
    """Load word highlighting configuration from separate file"""
    if os.path.exists(WORD_HIGHLIGHTING_FILE):
        try:
            with open(WORD_HIGHLIGHTING_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load word highlighting config: {e}")
    return {"enabled": True, "words": []}


def save_word_highlighting(data):
    """Save word highlighting configuration to separate file"""
    try:
        _atomic_write_json(WORD_HIGHLIGHTING_FILE, data, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save word highlighting config: {e}")
        return False


def _restore_config_from_template(reason=""):
    """Copy the bundled config.default.json over CONFIG_FILE. Returns True on success."""
    import shutil
    if not os.path.exists(CONFIG_TEMPLATE_FILE):
        print(f"[CONFIG] ERROR: template '{CONFIG_TEMPLATE_FILE}' is missing; cannot {reason or 'restore config'}.")
        return False
    shutil.copy2(CONFIG_TEMPLATE_FILE, CONFIG_FILE)
    return True


def load_config():
    """Load configuration from config.json.

    config.default.json (bundled, read from BUNDLE_DIR) is the canonical template.
    It seeds a fresh config.json on first run and is used to recover from a
    missing or corrupted config.json. No deep-merge is performed on normal loads —
    whatever seeds config.json is the complete config; missing keys are patched
    per-access via config.get(key, fallback)."""
    # First run: no config.json yet -> seed from the bundled template.
    if not os.path.exists(CONFIG_FILE):
        if _restore_config_from_template("create config.json"):
            print(f"[OK] Created '{CONFIG_FILE}' from config.default.json")
            print("[NOTE] Edit this file to configure your settings.")
        else:
            raise FileNotFoundError(
                f"Neither '{CONFIG_FILE}' nor template '{CONFIG_TEMPLATE_FILE}' exist; cannot start."
            )

    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        print(f"[OK] Loaded configuration from '{CONFIG_FILE}'")
    except Exception as e:
        # Corrupted / unreadable config.json: back up the bad file, then rewrite a
        # fresh one from the template so the app can still start.
        print(f"[CONFIG] ERROR: could not parse '{CONFIG_FILE}': {e}")
        from datetime import datetime
        import shutil
        try:
            corrupt_path = f"{CONFIG_FILE}.corrupt.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(CONFIG_FILE, corrupt_path)
            print(f"[CONFIG] Backed up corrupt config to '{corrupt_path}'")
        except Exception as move_err:
            print(f"[CONFIG] WARNING: could not back up corrupt config: {move_err}")
        if not _restore_config_from_template("recover config.json"):
            raise
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        print(f"[CONFIG] Recovered '{CONFIG_FILE}' from config.default.json")

    # Migrate old use_english_model format to new .en model names
    migrated = False
    for section in ["model", "file_transcription"]:
        if section in config:
            # Handle top-level model section
            if section == "model" and "whisper" in config[section]:
                whisper = config[section]["whisper"]
                if whisper.get("use_english_model", False):
                    model = whisper.get("model", "base")
                    if model in ["tiny", "base", "small", "medium"] and not model.endswith(".en"):
                        whisper["model"] = f"{model}.en"
                        print(f"[MIGRATION] Converted model '{model}' to '{model}.en'")
                        migrated = True
                # Remove the old flag
                if "use_english_model" in whisper:
                    whisper.pop("use_english_model")
                    migrated = True

            # Handle file_transcription model section
            elif section == "file_transcription" and "model" in config[section]:
                if "whisper" in config[section]["model"]:
                    whisper = config[section]["model"]["whisper"]
                    if whisper.get("use_english_model", False):
                        model = whisper.get("model", "base")
                        if model in ["tiny", "base", "small", "medium"] and not model.endswith(".en"):
                            whisper["model"] = f"{model}.en"
                            print(f"[MIGRATION] Converted file transcription model '{model}' to '{model}.en'")
                            migrated = True
                    # Remove the old flag
                    if "use_english_model" in whisper:
                        whisper.pop("use_english_model")
                        migrated = True

    # Save migrated config
    if migrated:
        try:
            with _config_file_lock:
                _atomic_write_json(CONFIG_FILE, config)
            print("[MIGRATION] Config file updated and saved")
        except Exception as e:
            print(f"[MIGRATION] Warning: Could not save migrated config: {e}")

    return config


# Load configuration
config = load_config()

# Create empty word highlighting file if it doesn't exist
if not os.path.exists(WORD_HIGHLIGHTING_FILE):
    print(f"[INIT] Creating empty {WORD_HIGHLIGHTING_FILE}")
    save_word_highlighting({"enabled": True, "words": []})

# Calibration mode state
calibration_mode = False
calibration_data = None

# Remote translation state
_pending_pair_requests = {}    # {ip: {"code": str, "expires": float}}
_translation_clients = {}      # {ip: last_seen_timestamp}
_translation_clients_lock = threading.Lock()
_trusted_translation_clients = set(
    config.get("live_translation", {}).get("trusted_clients", [])
)


def _is_trusted_translation_client(ip):
    return ip in _trusted_translation_clients


def _add_trusted_client(ip):
    _trusted_translation_clients.add(ip)
    if "live_translation" not in config:
        config["live_translation"] = {}
    trusted = config["live_translation"].setdefault("trusted_clients", [])
    if ip not in trusted:
        trusted.append(ip)
    save_config(config)


def _register_translation_client(ip):
    with _translation_clients_lock:
        _translation_clients[ip] = time.time()

# Generate random password if not configured
password_auth_config = config.get("web_server", {}).get("password_auth", {})
if password_auth_config.get("enabled", False) and not password_auth_config.get("password", ""):
    import secrets
    import string
    # Generate a random 12-character password
    alphabet = string.ascii_letters + string.digits
    random_password = ''.join(secrets.choice(alphabet) for i in range(12))

    # Update config with generated password
    if "web_server" not in config:
        config["web_server"] = {}
    if "password_auth" not in config["web_server"]:
        config["web_server"]["password_auth"] = {}
    config["web_server"]["password_auth"]["password"] = random_password

    # Save config
    save_config(config)

    print("=" * 80)
    print("[AUTH] Password authentication enabled with auto-generated password:")
    print(f"[AUTH] Password: {random_password}")
    print("[AUTH] Save this password to access settings from non-whitelisted IPs.")
    print("[AUTH] You can change it in config.json under web_server.password_auth.password")
    print("=" * 80)


# Timezone Helper Function
def get_configured_timezone():
    """Get system timezone."""
    return datetime.now().astimezone().tzinfo


# Load configured timezone
configured_timezone = get_configured_timezone()
print(f"[OK] Using timezone: {configured_timezone}")


# ====================================================================================
# NLLB Translation Support - Translate to 200+ languages
# ====================================================================================

# Map common ISO codes to NLLB language codes
NLLB_LANG_CODES = {
    "auto": "eng_Latn",  # Fallback to English if auto
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh": "zho_Hans",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "uk": "ukr_Cyrl",
    "cs": "ces_Latn",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "no": "nob_Latn",
    "el": "ell_Grek",
    "he": "heb_Hebr",
    "hu": "hun_Latn",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "ro": "ron_Latn",
    "sk": "slk_Latn",
    "bg": "bul_Cyrl",
    "hr": "hrv_Latn",
    "sr": "srp_Cyrl",
    "sl": "slv_Latn",
    "et": "est_Latn",
    "lv": "lvs_Latn",
    "lt": "lit_Latn",
    "fa": "pes_Arab",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
}

# Human-readable language names for UI
TRANSLATION_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese (Simplified)",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "uk": "Ukrainian",
    "cs": "Czech",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "el": "Greek",
    "he": "Hebrew",
    "hu": "Hungarian",
    "id": "Indonesian",
    "ms": "Malay",
    "ro": "Romanian",
    "sk": "Slovak",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sr": "Serbian",
    "sl": "Slovenian",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "fa": "Persian",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
}


def load_translation_model(use_gpu=True, model_id=None):
    """
    Load NLLB-200 translation model

    Args:
        use_gpu: Whether to use GPU acceleration
        model_id: HuggingFace model ID (e.g., "facebook/nllb-200-distilled-600M")
                  If None, defaults to facebook/nllb-200-distilled-600M

    Returns:
        Tuple of (model, tokenizer)
    """
    _lazy_import_ml_libraries()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # Use provided model_id or default
    if model_id is None:
        model_id = "facebook/nllb-200-distilled-600M"

    # Check ./models/ directory first for local copy
    local_dir_name = model_id.replace("/", "--")
    local_model_path = os.path.join(MODELS_DIR, local_dir_name)

    if os.path.exists(local_model_path):
        model_path = local_model_path
        print(f"[INFO] Loading translation model from local: {model_path}")
    else:
        model_path = model_id
        print(f"[INFO] Loading translation model from HuggingFace: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    bin_path = os.path.join(model_path, "pytorch_model.bin") if os.path.isdir(model_path) else None
    if bin_path and os.path.exists(bin_path) and not os.path.exists(os.path.join(model_path, "model.safetensors")):
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_config(cfg)
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    if use_gpu and torch.cuda.is_available():
        model = model.to("cuda")
        print("[INFO] Translation model loaded on GPU (CUDA)")
    elif use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model = model.to("mps")
        print("[INFO] Translation model loaded on GPU (MPS)")
    else:
        print("[INFO] Translation model loaded on CPU")

    return model, tokenizer


def _apply_glossary(text, source_lang, target_lang):
    """Apply NLLB glossary post-processing replacements from custom dictionary."""
    try:
        dict_config = config.get("custom_dictionary", {})
        if not dict_config.get("nllb_glossary_enabled", False):
            return text

        dict_file = dict_config.get("file", "custom_dictionary.json")
        if not os.path.isabs(dict_file):
            dict_file = os.path.join(APP_DIR, dict_file)

        if not os.path.exists(dict_file):
            return text

        import json as _json
        with open(dict_file, "r", encoding="utf-8") as f:
            dictionary = _json.load(f)

        glossary_key = f"{source_lang}_to_{target_lang}"
        glossary = dictionary.get("glossary", {}).get(glossary_key, {})
        if not glossary:
            return text

        import re
        # Longest terms first so a short term can't clobber a longer one containing it.
        # Lookarounds instead of \b: terms may start/end with punctuation, where \b
        # silently fails. Lambda replacement so backslashes in the target aren't
        # treated as regex templates. Note: \w matches CJK, so terms embedded in
        # unspaced CJK runs won't match.
        for source_term, target_term in sorted(glossary.items(), key=lambda kv: -len(kv[0])):
            pattern = r"(?<!\w)" + re.escape(source_term) + r"(?!\w)"
            text = re.sub(pattern, lambda m, t=target_term: t, text, flags=re.IGNORECASE)

        return text
    except Exception as e:
        print(f"[WARNING] Glossary application failed: {e}")
        return text


def translate_text(text, source_lang, target_lang, model, tokenizer, return_confidence=False, num_alternatives=0, generation_params=None):
    """
    Translate text using NLLB-200

    Args:
        text: Text to translate
        source_lang: Source language ISO code (e.g., 'en', 'fr')
        target_lang: Target language ISO code
        model: Loaded NLLB model
        tokenizer: Loaded NLLB tokenizer
        return_confidence: If True, return (text, confidence) tuple
        num_alternatives: Number of alternative translations to return (0 = none)
        generation_params: Dict of generation parameters (num_beams, length_penalty, etc.)

    Returns:
        Translated text string, or dict with text/confidence/alternatives if extras requested
    """
    if not text or not text.strip():
        if return_confidence or num_alternatives > 0:
            return {"text": text, "confidence": None, "alternatives": []}
        return text

    # Convert ISO codes to NLLB codes
    src_nllb = NLLB_LANG_CODES.get(source_lang, "eng_Latn")
    tgt_nllb = NLLB_LANG_CODES.get(target_lang, "eng_Latn")

    # Set source language
    tokenizer.src_lang = src_nllb

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)

    # Move inputs to same device as model (CUDA, MPS, or CPU)
    _device = next(model.parameters()).device
    if _device.type != "cpu":
        inputs = {k: v.to(_device) for k, v in inputs.items()}

    # Merge user generation params with defaults
    gp = generation_params or {}
    num_beams = gp.get("num_beams", 5)
    length_penalty = gp.get("length_penalty", 1.0)
    no_repeat_ngram_size = gp.get("no_repeat_ngram_size", 0)
    repetition_penalty = gp.get("repetition_penalty", 1.0)

    # Build generate kwargs — validate target token is known
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_nllb)
    if forced_bos_id == tokenizer.unk_token_id:
        print(f"[LIVE-TRANSLATION WARNING] Unknown target language token: {tgt_nllb} for lang={target_lang}, falling back to eng_Latn")
        forced_bos_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    generate_kwargs = {
        "forced_bos_token_id": forced_bos_id,
        "max_length": 1024,
        "num_beams": num_beams,
        "length_penalty": length_penalty,
        "early_stopping": True,
    }

    # Only add these if non-default to avoid warnings
    if no_repeat_ngram_size > 0:
        generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if repetition_penalty != 1.0:
        generate_kwargs["repetition_penalty"] = repetition_penalty

    # Enable confidence scoring and/or alternatives
    if return_confidence or num_alternatives > 0:
        generate_kwargs["return_dict_in_generate"] = True
        generate_kwargs["output_scores"] = True
        if num_alternatives > 0:
            generate_kwargs["num_return_sequences"] = min(num_alternatives + 1, 5)
            generate_kwargs["num_beams"] = max(5, num_alternatives + 1)

    # Generate translation
    translated = model.generate(**inputs, **generate_kwargs)

    if return_confidence or num_alternatives > 0:
        # Extract sequences and scores
        sequences = translated.sequences
        all_decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)

        # Compute confidence from sequence scores if available
        confidence = None
        if hasattr(translated, 'sequences_scores') and translated.sequences_scores is not None:
            import torch
            confidence = float(torch.exp(translated.sequences_scores[0]).item())

        best_text = _apply_glossary(all_decoded[0], source_lang, target_lang)
        alternatives = [_apply_glossary(t, source_lang, target_lang) for t in all_decoded[1:]] if len(all_decoded) > 1 else []

        return {"text": best_text, "confidence": confidence, "alternatives": alternatives}

    # Simple mode - decode and return
    result_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return _apply_glossary(result_text, source_lang, target_lang)


def translate_segments(segments, source_lang, target_lang, model, tokenizer, progress_callback=None, generation_params=None, context_window=1):
    """
    Translate a list of transcription segments

    Args:
        segments: List of segment dicts with 'text', 'start', 'end' keys
        source_lang: Source language ISO code
        target_lang: Target language ISO code
        model: Loaded NLLB model
        tokenizer: Loaded NLLB tokenizer
        progress_callback: Optional callback function(percent, status) for progress updates
        generation_params: Dict of generation parameters (num_beams, length_penalty, etc.)
        context_window: Number of segments to combine for context (1 = no batching)

    Returns:
        List of translated segment dicts with same structure
    """
    translated_segments = []
    total = len(segments)
    context_window = max(1, context_window)

    for i, seg in enumerate(segments):
        translated_text = None
        if context_window > 1 and i > 0:
            # Translate (context + target) in one call, then extract the target's
            # portion by sentence-count alignment
            start_idx = max(0, i - (context_window - 1))
            context_text = " ".join(segments[j]["text"] for j in range(start_idx, i)).strip()
            if context_text:
                num_ctx_sentences = count_sentence_units(context_text)
                combined_source = context_text + " " + seg["text"]
                ctx_char_ratio = (len(context_text) + 1) / max(1, len(combined_source))
                combined_translated = translate_text(combined_source, source_lang, target_lang, model, tokenizer, generation_params=generation_params)
                translated_text = extract_context_translation(combined_translated, num_ctx_sentences, ctx_char_ratio)
        if not translated_text:
            # No context, or alignment failed - translate the segment alone
            translated_text = translate_text(seg["text"], source_lang, target_lang, model, tokenizer, generation_params=generation_params)
        translated_segments.append({
            "text": translated_text,
            "start": seg["start"],
            "end": seg["end"]
        })

        # Report progress
        if progress_callback and total > 0:
            percent = int(70 + (i + 1) / total * 20)  # 70-90% range
            progress_callback(percent, f"Translating... ({i + 1}/{total} segments)")

    return translated_segments


def cleanup_translation_model(model, tokenizer):
    """Clean up translation model to free memory"""
    import gc

    del model
    del tokenizer

    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()
    print("[CLEANUP] Translation model unloaded")


# ====================================================================================
# Live Translation - Singleton Model Management and Caching
# ====================================================================================

# Global translation model state (persistent while live translation is enabled)
_live_translation_model = None
_live_translation_tokenizer = None
_live_translation_lock = threading.Lock()
_live_translation_model_loaded = False
_live_translation_model_loading = False  # Track when model is being loaded
_live_translation_model_id = None  # Track which model is loaded to detect config changes
_live_translation_target_lang = None


def is_live_translation_ready():
    """True when a live translation can actually be produced right now: a remote
    endpoint is configured, or the local NLLB model has finished loading. Used to
    avoid persisting a warmup echo (the source text returned unchanged while the
    model is still loading)."""
    remote_cfg = config.get("live_translation", {}).get("remote", {})
    if remote_cfg.get("enabled") and remote_cfg.get("endpoint") and not _trusted_translation_clients:
        return True
    return _live_translation_model_loaded


def get_live_translation_model(use_gpu=True, model_id=None):
    """Get or load the live translation model (singleton pattern).
    If model_id differs from the currently loaded model, unloads and reloads."""
    global _live_translation_model, _live_translation_tokenizer, _live_translation_model_loaded, _live_translation_model_loading, _live_translation_model_id

    with _live_translation_lock:
        # Don't load model if transcription is actively stopping (to prevent GPU memory leak)
        status = transcription_state.get("status", "")
        if _live_translation_model is None and status == "stopping":
            print("[LIVE-TRANSLATION] Skipping model load - transcription is stopping")
            return None, None

        # If model_id changed, unload the stale model so it reloads with the correct one
        if _live_translation_model is not None and model_id and _live_translation_model_id and model_id != _live_translation_model_id:
            print(f"[LIVE-TRANSLATION] Model changed: {_live_translation_model_id} -> {model_id}, reloading...")
            import gc
            del _live_translation_model
            del _live_translation_tokenizer
            _live_translation_model = None
            _live_translation_tokenizer = None
            _live_translation_model_loaded = False
            _live_translation_model_id = None
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        if _live_translation_model is None:
            _live_translation_model_loading = True
            try:
                print(f"[LIVE-TRANSLATION] Loading live translation model: {model_id or 'default'}...")
                _live_translation_model, _live_translation_tokenizer = load_translation_model(
                    use_gpu=use_gpu,
                    model_id=model_id
                )
                _live_translation_model_loaded = True
                _live_translation_model_id = model_id
                print(f"[LIVE-TRANSLATION] Live translation model loaded: {model_id or 'default'}")
            finally:
                _live_translation_model_loading = False
        return _live_translation_model, _live_translation_tokenizer


def unload_live_translation_model():
    """Unload the live translation model to free GPU memory"""
    global _live_translation_model, _live_translation_tokenizer, _live_translation_model_loaded, _live_translation_model_id
    import gc

    with _live_translation_lock:
        if _live_translation_model is not None:
            print("[LIVE-TRANSLATION] Unloading live translation model...")
            del _live_translation_model
            del _live_translation_tokenizer
            _live_translation_model = None
            _live_translation_tokenizer = None
            _live_translation_model_loaded = False
            _live_translation_model_id = None
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("[LIVE-TRANSLATION] Live translation model unloaded")


def is_live_translation_model_loaded():
    """Check if the live translation model is currently loaded"""
    return _live_translation_model_loaded


def is_live_translation_model_loading():
    """Check if the live translation model is currently being loaded"""
    return _live_translation_model_loading


# ====================================================================================
# TTS (Text-to-Speech) - Multi-backend: edge-tts (cloud) and piper (local)
# ====================================================================================

_tts_piper_model = None
_tts_lock = threading.Lock()
_tts_model_loaded = False
_tts_model_loading = False
_tts_sample_rate = 22050

# Edge-TTS voice cache (populated on first request)
_edge_tts_voices = None
_edge_tts_voices_lock = threading.Lock()


def _get_tts_backend():
    """Get the configured TTS backend ('edge' or 'piper')"""
    return config.get("live_translation", {}).get("tts", {}).get("backend", "edge")


def get_edge_tts_voices():
    """Get cached list of edge-tts voices. Returns list of dicts with Name, ShortName, Gender, Locale."""
    global _edge_tts_voices
    with _edge_tts_voices_lock:
        if _edge_tts_voices is not None:
            return _edge_tts_voices
    try:
        import edge_tts
    except ImportError:
        print("[TTS] edge-tts not installed. Install with: pip install edge-tts")
        return []
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        voices = loop.run_until_complete(edge_tts.list_voices())
        loop.close()
        with _edge_tts_voices_lock:
            _edge_tts_voices = voices
        print(f"[TTS] Loaded {len(voices)} edge-tts voices")
        return voices
    except Exception as e:
        print(f"[TTS] Failed to fetch edge-tts voices: {e}")
        return []


def _pick_default_edge_voice(lang_code):
    """Pick a default edge-tts voice for a language code (e.g., 'ru' -> 'ru-RU-DmitryNeural').
    Returns ShortName string or None if no match found."""
    voices = get_edge_tts_voices()
    if not voices:
        return None
    lang_lower = lang_code.lower()
    for v in voices:
        locale = v.get("Locale", "").lower()
        if locale.startswith(lang_lower):
            return v.get("ShortName")
    return None


def _pick_default_piper_model(lang_code):
    """Pick a downloaded piper model matching a language code.
    Returns model ID string or None if no match found."""
    lang_lower = lang_code.lower()
    for m in _PIPER_MODELS_CATALOG:
        if m["language"].lower() == lang_lower and _is_piper_model_downloaded(m["id"]):
            return m["id"]
    return None


def get_tts_model(use_gpu=False, model_name=None):
    """Load piper TTS model (singleton). For edge-tts, no model loading needed."""
    global _tts_piper_model, _tts_model_loaded, _tts_model_loading, _tts_sample_rate

    backend = _get_tts_backend()

    if backend == "edge":
        _tts_model_loaded = True
        return True  # edge-tts needs no model

    # Piper backend
    with _tts_lock:
        status = transcription_state.get("status", "")
        if _tts_piper_model is None and status == "stopping":
            print("[TTS] Skipping piper model load - transcription is stopping")
            return None

        if _tts_piper_model is None:
            _tts_model_loading = True
            try:
                from piper import PiperVoice
                tts_config = config.get("live_translation", {}).get("tts", {})
                if model_name is None:
                    model_name = tts_config.get("piper_model", "")

                if not model_name:
                    print("[TTS ERROR] No piper model configured")
                    return None

                model_path = os.path.join(_tts_cache_dir, "piper", model_name)
                onnx_files = [f for f in os.listdir(model_path) if f.endswith(".onnx")] if os.path.isdir(model_path) else []
                if not onnx_files:
                    print(f"[TTS ERROR] No .onnx model found in {model_path}")
                    return None

                onnx_path = os.path.join(model_path, onnx_files[0])
                json_path = onnx_path + ".json"

                print(f"[TTS] Loading piper model: {model_name}...")
                _tts_piper_model = PiperVoice.load(onnx_path, config_path=json_path if os.path.exists(json_path) else None)
                _tts_model_loaded = True
                _tts_sample_rate = _tts_piper_model.config.sample_rate if hasattr(_tts_piper_model, 'config') else 22050
                print(f"[TTS] Piper model loaded (sample_rate={_tts_sample_rate})")
            except Exception as e:
                print(f"[TTS ERROR] Failed to load piper model: {e}")
                _tts_piper_model = None
                _tts_model_loaded = False
            finally:
                _tts_model_loading = False
        return _tts_piper_model


def unload_tts_model():
    """Unload the piper TTS model to free memory"""
    global _tts_piper_model, _tts_model_loaded
    import gc

    with _tts_lock:
        if _tts_piper_model is not None:
            print("[TTS] Unloading piper model...")
            del _tts_piper_model
            _tts_piper_model = None
            _tts_model_loaded = False
            gc.collect()
            print("[TTS] Piper model unloaded")
        else:
            _tts_model_loaded = False


def is_tts_model_loaded():
    if _get_tts_backend() == "edge":
        return True  # edge-tts is always ready
    return _tts_model_loaded


def is_tts_model_loading():
    return _tts_model_loading


def _synthesize_edge_tts(text, voice=None, speed=1.0):
    """Synthesize speech using edge-tts (Microsoft Edge cloud TTS). Returns (mp3_bytes, sample_rate) or (None, None)."""
    try:
        import asyncio
        import edge_tts
        import io

        tts_config = config.get("live_translation", {}).get("tts", {})
        if voice is None:
            voice = tts_config.get("edge_voice", "en-US-AriaNeural")

        # edge-tts rate format: "+0%", "-50%", "+100%" etc
        rate_str = "+0%"
        if speed != 1.0:
            pct = int((speed - 1.0) * 100)
            rate_str = f"{pct:+d}%"

        async def _do_synth():
            communicate = edge_tts.Communicate(text, voice, rate=rate_str)
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            return buf.getvalue()

        loop = asyncio.new_event_loop()
        try:
            # Bound the network call: a hung connection would otherwise block
            # the TTS emit loop indefinitely
            mp3_bytes = loop.run_until_complete(asyncio.wait_for(_do_synth(), timeout=15))
        finally:
            loop.close()

        if not mp3_bytes:
            return None, None

        return mp3_bytes, 24000  # edge-tts outputs 24kHz mp3
    except Exception as e:
        print(f"[TTS ERROR] edge-tts synthesis failed: {e}")
        return None, None


def _synthesize_piper_tts(text, language="en"):
    """Synthesize speech using piper (local). Returns (wav_bytes, sample_rate) or (None, None)."""
    model = get_tts_model()
    if model is None:
        return None, None

    try:
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(_tts_sample_rate)
            model.synthesize(text, wav_file)

        return buf.getvalue(), _tts_sample_rate
    except Exception as e:
        print(f"[TTS ERROR] Piper synthesis failed: {e}")
        return None, None


def synthesize_tts(text, language="en"):
    """Synthesize speech from text. Returns (audio_bytes, sample_rate) or (None, None).
    Audio format: mp3 for edge-tts, wav for piper.
    """
    tts_config = config.get("live_translation", {}).get("tts", {})
    speed = tts_config.get("speed", 1.0)
    backend = _get_tts_backend()

    if backend == "edge":
        return _synthesize_edge_tts(text, speed=speed)
    elif backend == "piper":
        return _synthesize_piper_tts(text, language=language)
    else:
        print(f"[TTS ERROR] Unknown backend: {backend}")
        return None, None


class TranslationCache:
    """Cache translated segments to avoid re-translating"""

    def __init__(self, max_size=1000):
        self._cache = {}  # {segment_id: {original, translated, target_lang}}
        self._max_size = max_size
        self._lock = threading.Lock()
        self._target_lang = None

    def get(self, segment_id, original_text, target_lang, accept_stale_lang=False):
        """Get cached translation or None.
        If accept_stale_lang=True, return cached translation even if target_lang differs
        (used to avoid retranslating old segments after a hot language switch)."""
        with self._lock:
            entry = self._cache.get(segment_id)
            if entry and entry['original'] == original_text and entry['target_lang'] == target_lang:
                return entry['translated']
            if accept_stale_lang and entry and entry['original'] == original_text:
                return entry['translated']
            return None

    def _evict_if_full(self):
        """Remove oldest entries when full (insertion order). Caller must hold _lock."""
        if len(self._cache) >= self._max_size:
            oldest = list(self._cache.keys())[:100]
            for key in oldest:
                del self._cache[key]

    def set(self, segment_id, original_text, translated_text, target_lang):
        """Cache a translation"""
        with self._lock:
            self._evict_if_full()
            self._cache[segment_id] = {
                'original': original_text,
                'translated': translated_text,
                'target_lang': target_lang
            }

    def invalidate(self, segment_id):
        """Invalidate a specific cached translation (e.g., after correction)"""
        with self._lock:
            if segment_id in self._cache:
                del self._cache[segment_id]

    def set_with_extras(self, segment_id, original_text, translated_text, target_lang, confidence=None, alternatives=None):
        """Cache a translation with confidence and alternatives"""
        with self._lock:
            self._evict_if_full()
            self._cache[segment_id] = {
                'original': original_text,
                'translated': translated_text,
                'target_lang': target_lang,
                'confidence': confidence,
                'alternatives': alternatives or [],
            }

    def get_extras(self, segment_id):
        """Get cached confidence and alternatives for a segment"""
        with self._lock:
            entry = self._cache.get(segment_id)
            if entry:
                return {
                    'confidence': entry.get('confidence'),
                    'alternatives': entry.get('alternatives', []),
                }
            return None

    def clear(self):
        """Clear all cached translations"""
        with self._lock:
            self._cache.clear()
            print("[LIVE-TRANSLATION] Translation cache cleared")

    def get_size(self):
        """Get current cache size"""
        with self._lock:
            return len(self._cache)

    def max_segment_id(self):
        """Highest integer segment id currently cached, or 0 if none"""
        with self._lock:
            return max((sid for sid in self._cache if isinstance(sid, int)), default=0)


# Global translation cache instance
_translation_cache = TranslationCache()


def get_translation_cache():
    """Get the global translation cache"""
    return _translation_cache


# ====================================================================================
# Model Factory - Supports multiple STT model types
# ====================================================================================


class ModelFactory:
    """Factory class for loading different types of speech-to-text models"""

    _model_cache = {}
    _cache_lock = threading.Lock()

    @staticmethod
    def load_model(model_config, use_gpu=True):
        """
        Load a speech-to-text model based on configuration with caching

        Args:
            model_config: Dictionary with model configuration
            use_gpu: Whether to use GPU acceleration

        Returns:
            Tuple of (model, processor/tokenizer, model_type)
        """
        # Import ML libraries before using them
        _lazy_import_ml_libraries()

        model_type = model_config.get("type", "whisper")
        if use_gpu and torch.cuda.is_available():
            device = "cuda"
        elif use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # Create cache key
        cache_key = f"{model_type}_{str(model_config)}_{device}"

        # Check cache first
        with ModelFactory._cache_lock:
            if cache_key in ModelFactory._model_cache:
                print(f"Using cached {model_type} model on {device}")
                return ModelFactory._model_cache[cache_key]

        print(f"Loading {model_type} model on {device}...")

        try:
            if model_type == "whisper":
                # Check for backend preference (faster-whisper or standard whisper)
                backend = model_config.get("backend")
                if backend == "faster-whisper":
                    print("Using faster-whisper backend (4-10x faster)")
                    model, processor, model_type_return = ModelFactory._load_faster_whisper(
                        model_config["whisper"], use_gpu
                    )
                else:
                    print("Using standard OpenAI Whisper backend")
                    model, processor, model_type_return = ModelFactory._load_whisper(
                        model_config["whisper"], use_gpu
                    )
            elif model_type == "huggingface":
                model, processor, model_type_return = ModelFactory._load_huggingface(
                    model_config["huggingface"], device
                )
            elif model_type == "custom":
                model, processor, model_type_return = ModelFactory._load_custom(
                    model_config["custom"], device
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Cache the loaded model
            with ModelFactory._cache_lock:
                ModelFactory._model_cache[cache_key] = (
                    model,
                    processor,
                    model_type_return,
                )

            return model, processor, model_type_return
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise

    @staticmethod
    def cleanup_models():
        """Clean up all cached models to free memory"""
        import gc

        with ModelFactory._cache_lock:
            # First, copy items and clear cache to remove references
            cache_items = list(ModelFactory._model_cache.items())
            ModelFactory._model_cache.clear()

            # Now delete model objects from the copied list
            for cache_key, (model, processor, _) in cache_items:
                try:
                    # Move to CPU first if possible (frees GPU memory faster)
                    if hasattr(model, "cpu"):
                        model.cpu()
                    # Delete the actual model object
                    del model
                    if processor:
                        del processor
                except Exception as e:
                    print(f"[WARNING] Error cleaning up model {cache_key}: {e}")

            # Delete the list too
            del cache_items

        # Force garbage collection OUTSIDE the lock
        # This is CRITICAL for ctranslate2/faster-whisper to release GPU memory
        gc.collect()

        # Now clear CUDA cache
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("[OK] All models cleaned up from memory, GPU cache cleared")
        except Exception as e:
            print(f"[OK] All models cleaned up from memory (GPU cleanup: {e})")

    @staticmethod
    def _load_whisper(whisper_config, device):
        """Load OpenAI Whisper model"""
        _lazy_import_ml_libraries()

        model_name = whisper_config.get("model", "base")

        print(f"Loading Whisper model: {model_name}")

        # Check if model exists in ./models directory first
        models_dir = MODELS_DIR
        whisper_model_dir = os.path.join(models_dir, f"whisper-{model_name}")

        # Determine download_root based on where model is located
        download_root = None
        if os.path.exists(whisper_model_dir):
            # Model exists in new ./models location
            download_root = whisper_model_dir
            print(f"Using Whisper model from: {whisper_model_dir}")
        else:
            # Fall back to checking old cache location
            whisper_cache_old = os.path.expanduser("~/.cache/whisper")
            model_file = f"{model_name}.pt"
            old_model_path = os.path.join(whisper_cache_old, model_file)
            if os.path.exists(old_model_path):
                print(f"Using Whisper model from cache: {whisper_cache_old}")
            else:
                raise FileNotFoundError(
                    f"Whisper model '{model_name}' is not downloaded. "
                    f"Please download it first from the Model Manager (Settings → Model Manager)."
                )

        # Load the model (model_name can include .en suffix for English-only variants)
        model = whisper.load_model(model_name, download_root=download_root)

        if device == "cuda":
            model = model.cuda()

        return model, None, "whisper"

    @staticmethod
    def _load_faster_whisper(whisper_config, use_gpu=True):
        """Load faster-whisper model (CTranslate2-based, 4-10x faster)"""
        import ctypes

        # Preload cuDNN libraries from nvidia-cudnn-cu12 pip package if available
        # This must happen BEFORE importing faster_whisper/ctranslate2
        # On Windows, cuDNN DLLs are auto-discovered via PATH — skip manual preloading
        cudnn_lib_path = os.path.expanduser("~/.local/lib/python3.12/site-packages/nvidia/cudnn/lib")
        if not platform.startswith('win') and os.path.exists(cudnn_lib_path) and use_gpu:
            cudnn_libs = [
                "libcudnn_ops.so.9",
                "libcudnn_cnn.so.9",
                "libcudnn_adv.so.9",
                "libcudnn_graph.so.9",
                "libcudnn_engines_precompiled.so.9",
                "libcudnn_engines_runtime_compiled.so.9",
                "libcudnn_heuristic.so.9",
                "libcudnn.so.9",
            ]
            for lib in cudnn_libs:
                lib_full_path = os.path.join(cudnn_lib_path, lib)
                if os.path.exists(lib_full_path):
                    try:
                        ctypes.CDLL(lib_full_path, mode=ctypes.RTLD_GLOBAL)
                    except OSError as e:
                        print(f"Warning: Could not preload {lib}: {e}")

        from faster_whisper import WhisperModel

        model_name = whisper_config.get("model", "small")
        compute_type = whisper_config.get("compute_type", "auto")

        print(f"Loading faster-whisper model: {model_name}")

        # Auto-detect best compute type based on hardware
        if compute_type == "auto":
            if use_gpu and torch.cuda.is_available():
                # Check GPU compute capability for float16 support
                gpu_props = torch.cuda.get_device_properties(0)
                if gpu_props.major >= 7:
                    compute_type = "float16"
                else:
                    compute_type = "float32"
            else:
                compute_type = "int8"  # CPU optimized

        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        # Check for local model first
        models_dir = MODELS_DIR
        local_model_path = os.path.join(models_dir, f"faster-whisper-{model_name}")

        if os.path.exists(local_model_path):
            model_path = local_model_path
            print(f"Using faster-whisper model from: {local_model_path}")
        else:
            raise FileNotFoundError(
                f"Faster-whisper model '{model_name}' is not downloaded. "
                f"Please download it first from the Model Manager (Settings → Model Manager)."
            )

        print(f"Device: {device}, Compute type: {compute_type}")

        model = WhisperModel(model_path, device=device, compute_type=compute_type)

        return model, None, "faster_whisper"

    @staticmethod
    def _load_huggingface(hf_config, device):
        """Load Hugging Face transformers model"""
        _lazy_import_ml_libraries()

        model_id = hf_config.get("model_id", "openai/whisper-tiny")
        use_flash_attention = hf_config.get("use_flash_attention", False)

        # Check if model exists locally in ./models directory
        models_dir = MODELS_DIR
        model_dir_name = model_id.replace("/", "--")
        local_model_path = os.path.join(models_dir, model_dir_name)

        # Use local path if it exists, otherwise tell user to download first
        if os.path.exists(local_model_path):
            model_path = local_model_path
            print(f"Loading Hugging Face model from local path: {local_model_path}")
        else:
            raise FileNotFoundError(
                f"HuggingFace model '{model_id}' is not downloaded. "
                f"Please download it first from the Model Manager (Settings → Model Manager)."
            )

        try:
            # Determine model architecture from model card
            info = model_info(model_id)
            pipeline_tag = info.pipeline_tag

            # Load based on architecture
            if (
                "whisper" in model_id.lower()
                or pipeline_tag == "automatic-speech-recognition"
            ):
                # Whisper-based models (including Distil-Whisper)
                torch_dtype = torch.float16 if device == "cuda" else torch.float32

                model_kwargs = {
                    "torch_dtype": torch_dtype,
                    "low_cpu_mem_usage": True,
                }

                if use_flash_attention and device == "cuda":
                    model_kwargs["attn_implementation"] = "flash_attention_2"

                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_path, **model_kwargs
                )
                model.to(device)

                processor = AutoProcessor.from_pretrained(model_path)

                # Create pipeline for easier inference
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    max_new_tokens=128,
                    chunk_length_s=30,
                    batch_size=16,
                    torch_dtype=torch_dtype,
                    device=device,
                )

                return pipe, processor, "huggingface_whisper"

            elif "wav2vec2" in model_id.lower():
                # Wav2Vec2 models
                model = AutoModelForCTC.from_pretrained(model_path)
                model.to(device)
                processor = Wav2Vec2Processor.from_pretrained(model_path)

                return model, processor, "huggingface_wav2vec2"

            else:
                # Generic ASR model
                pipe = pipeline(
                    "automatic-speech-recognition", model=model_path, device=device
                )
                return pipe, None, "huggingface_generic"

        except Exception as e:
            print(f"Error loading Hugging Face model: {e}")
            raise

    @staticmethod
    def _load_custom(custom_config, device):
        """Load custom model from local path"""
        _lazy_import_ml_libraries()

        model_path = custom_config.get("model_path", "")
        model_type = custom_config.get("model_type", "whisper")

        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Custom model path not found: {model_path}")

        print(f"Loading custom {model_type} model from: {model_path}")

        if model_type == "whisper":
            # If model_path is a directory (e.g., ./models/whisper-base),
            # check if it contains a .pt file
            if os.path.isdir(model_path):
                pt_files = [f for f in os.listdir(model_path) if f.endswith(".pt")]
                if pt_files:
                    # Use the directory as download_root and extract model name
                    model_file = pt_files[0]
                    model_name = model_file.replace(".pt", "").replace(".en", "")
                    model = whisper.load_model(model_name, download_root=model_path)
                else:
                    raise ValueError(f"No .pt files found in directory: {model_path}")
            else:
                # model_path points directly to a .pt file
                model = whisper.load_model(model_path)

            if device == "cuda":
                model = model.cuda()
            return model, None, "whisper"
        else:
            # Try loading as Hugging Face model
            pipe = pipeline(
                "automatic-speech-recognition", model=model_path, device=device
            )
            return pipe, None, "huggingface_generic"

    @staticmethod
    def transcribe(model, processor, model_type, audio_data, language="auto", whisper_params=None, return_segments=False):
        """
        Transcribe audio using the loaded model

        Args:
            model: The loaded model
            processor: The processor/tokenizer (if applicable)
            model_type: Type of model ('whisper', 'huggingface_whisper', etc.)
            audio_data: Audio data as numpy array
            language: Language code (default: 'en', 'auto' for auto-detection)
            whisper_params: Dict of Whisper decoding parameters (optional)
                          For Whisper models: beam_size, temperature, condition_on_previous_text, etc.
                          See LIVE_TRANSCRIPTION_PARAMS and FILE_TRANSCRIPTION_PARAMS constants
            return_segments: If True, return list of segments with timestamps instead of just text

        Returns:
            If return_segments=False: Transcription text (str)
            If return_segments=True: List of segment dicts with 'text', 'start', 'end' keys
        """
        try:
            if model_type == "whisper":
                # Original Whisper model (OpenAI whisper)
                # Build params dict with language and whisper_params
                params = {}
                if language != "auto":
                    params["language"] = language

                # OpenAI whisper supported parameters (transcribe-level + DecodingOptions)
                whisper_transcribe_params = {
                    "verbose", "temperature", "compression_ratio_threshold",
                    "logprob_threshold", "no_speech_threshold",
                    "condition_on_previous_text", "initial_prompt",
                    "word_timestamps", "prepend_punctuations", "append_punctuations",
                    "clip_timestamps", "hallucination_silence_threshold",
                    "carry_initial_prompt",
                    # DecodingOptions params
                    "task", "language", "sample_len", "best_of", "beam_size",
                    "patience", "length_penalty", "prefix", "suppress_tokens",
                    "suppress_blank", "without_timestamps", "max_initial_timestamp",
                    "fp16",
                }

                if whisper_params:
                    for k, v in whisper_params.items():
                        if k.startswith("_"):
                            continue
                        if k in whisper_transcribe_params:
                            params[k] = v

                result = model.transcribe(audio_data, **params)

                if return_segments:
                    # Return Whisper's native segments with timestamps
                    segments = result.get("segments", [])
                    return [{"text": seg["text"].strip(), "start": seg["start"], "end": seg["end"]} for seg in segments if seg["text"].strip()]
                return result["text"].strip()

            elif model_type == "faster_whisper":
                # faster-whisper model (CTranslate2-based)
                # Build params dict with language and whisper_params
                params = {
                    "vad_filter": False,  # External VAD already screens; internal VAD over-chunks long buffers
                }
                if language != "auto":
                    params["language"] = language

                # faster-whisper supported parameters (different from standard whisper)
                # NOTE: initial_prompt and hotwords are intentionally excluded.
                # Both are tokenized into the decoder prefix and consume decoder positions
                # (max 448 total). The full Bible-book hotwords list is ~227 tokens alone,
                # leaving only ~220 positions for actual transcription — not enough for
                # dense speech. Whisper's language setting is sufficient for recognition.
                faster_whisper_params = {
                    "beam_size", "best_of", "patience", "length_penalty",
                    "repetition_penalty", "no_repeat_ngram_size",
                    "temperature", "compression_ratio_threshold",
                    "log_prob_threshold", "no_speech_threshold",
                    "condition_on_previous_text",
                    "prefix", "suppress_blank", "suppress_tokens",
                    "without_timestamps", "max_initial_timestamp",
                    "word_timestamps", "prepend_punctuations",
                    "append_punctuations", "vad_filter", "vad_parameters",
                    "task",
                }

                # Parameter name mapping (whisper -> faster-whisper)
                param_mapping = {
                    "logprob_threshold": "log_prob_threshold",  # Different naming
                }

                if whisper_params:
                    for k, v in whisper_params.items():
                        if k.startswith("_"):
                            continue
                        # Map parameter name if needed
                        mapped_key = param_mapping.get(k, k)
                        # Only include if faster-whisper supports it
                        if mapped_key in faster_whisper_params:
                            params[mapped_key] = v

                # faster-whisper returns (segments_iterator, info)
                segments_iter, info = model.transcribe(audio_data, **params)

                # Convert iterator to list with standard format
                segments = []
                full_text = []
                for seg in segments_iter:
                    text = seg.text.strip()
                    if text:
                        seg_dict = {
                            "text": text,
                            "start": seg.start,
                            "end": seg.end,
                            "no_speech_prob": getattr(seg, "no_speech_prob", 0),
                            "avg_logprob": getattr(seg, "avg_logprob", 0),
                            # Detected source language (ISO code) so rows can record source_language
                            # even when audio.language is 'auto'. Same for every seg in the chunk.
                            "language": getattr(info, "language", None),
                        }
                        # Extract word-level confidence if available
                        if hasattr(seg, 'words') and seg.words:
                            seg_dict["words"] = [
                                {
                                    "word": w.word,
                                    "probability": getattr(w, "probability", None),
                                    "start": w.start,
                                    "end": w.end,
                                }
                                for w in seg.words
                            ]
                        segments.append(seg_dict)
                        full_text.append(text)

                if return_segments:
                    return segments
                return " ".join(full_text)

            elif model_type == "huggingface_whisper":
                # Hugging Face Whisper pipeline
                generate_kwargs = {}
                if language != "auto":
                    generate_kwargs["language"] = language

                # Map Whisper params to HuggingFace generate_kwargs
                if whisper_params:
                    if "beam_size" in whisper_params:
                        generate_kwargs["num_beams"] = whisper_params["beam_size"]
                    if "temperature" in whisper_params:
                        _hf_temp = whisper_params["temperature"]
                        if isinstance(_hf_temp, (list, tuple)):
                            # HF supports a fallback tuple in long-form generation
                            generate_kwargs["temperature"] = tuple(float(t) for t in _hf_temp)
                        elif isinstance(_hf_temp, (int, float)):
                            generate_kwargs["temperature"] = _hf_temp
                    # Quality thresholds: honored by HF Whisper long-form generation;
                    # stripped via the retry below when the installed transformers
                    # version doesn't accept them.
                    if "compression_ratio_threshold" in whisper_params:
                        generate_kwargs["compression_ratio_threshold"] = whisper_params["compression_ratio_threshold"]
                    if "logprob_threshold" in whisper_params:
                        generate_kwargs["logprob_threshold"] = whisper_params["logprob_threshold"]
                    if "no_speech_threshold" in whisper_params:
                        generate_kwargs["no_speech_threshold"] = whisper_params["no_speech_threshold"]
                    if "condition_on_previous_text" in whisper_params:
                        # HF names this condition_on_prev_tokens
                        generate_kwargs["condition_on_prev_tokens"] = whisper_params["condition_on_previous_text"]

                _hf_optional_keys = (
                    "compression_ratio_threshold", "logprob_threshold",
                    "no_speech_threshold", "condition_on_prev_tokens",
                )

                def _hf_pipeline_call(**call_kwargs):
                    try:
                        return model(audio_data, generate_kwargs=generate_kwargs, **call_kwargs) if generate_kwargs else model(audio_data, **call_kwargs)
                    except (TypeError, ValueError) as hf_err:
                        stripped = {k: v for k, v in generate_kwargs.items() if k not in _hf_optional_keys}
                        if len(stripped) == len(generate_kwargs):
                            raise
                        print(f"[WARNING] HF pipeline rejected quality thresholds ({hf_err}); retrying without them")
                        return model(audio_data, generate_kwargs=stripped, **call_kwargs) if stripped else model(audio_data, **call_kwargs)

                if return_segments:
                    # Request timestamps from HuggingFace pipeline
                    result = _hf_pipeline_call(return_timestamps=True)
                    chunks = result.get("chunks", [])
                    segments = []
                    for chunk in chunks:
                        text = chunk.get("text", "").strip()
                        timestamp = chunk.get("timestamp", (0, 0))
                        if text and timestamp:
                            start = timestamp[0] if timestamp[0] is not None else 0
                            end = timestamp[1] if timestamp[1] is not None else start
                            segments.append({"text": text, "start": start, "end": end})
                    return segments

                result = _hf_pipeline_call()
                return result["text"].strip()

            elif model_type == "huggingface_wav2vec2":
                # Wav2Vec2 model (doesn't support language parameter or timestamps)
                import torch

                inputs = processor(
                    audio_data, sampling_rate=16000, return_tensors="pt", padding=True
                )

                with torch.no_grad():
                    logits = model(inputs.input_values.to(model.device)).logits

                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)
                text = transcription[0].strip()

                if return_segments:
                    # Wav2Vec2 doesn't provide timestamps, return as single segment
                    return [{"text": text, "start": 0, "end": 0}] if text else []
                return text

            elif model_type == "huggingface_generic":
                # Generic Hugging Face pipeline
                # Try to pass language if supported, otherwise just use audio
                try:
                    if return_segments:
                        if language == "auto":
                            result = model(audio_data, return_timestamps=True)
                        else:
                            result = model(audio_data, return_timestamps=True, generate_kwargs={"language": language})
                        chunks = result.get("chunks", [])
                        segments = []
                        for chunk in chunks:
                            text = chunk.get("text", "").strip()
                            timestamp = chunk.get("timestamp", (0, 0))
                            if text and timestamp:
                                start = timestamp[0] if timestamp[0] is not None else 0
                                end = timestamp[1] if timestamp[1] is not None else start
                                segments.append({"text": text, "start": start, "end": end})
                        return segments

                    if language == "auto":
                        result = model(audio_data)
                    else:
                        result = model(
                            audio_data, generate_kwargs={"language": language}
                        )
                except (TypeError, ValueError, RuntimeError) as e:
                    # Fallback if language parameter not supported
                    print(f"[WARNING] Model language parameter failed ({e}), falling back to auto-detect")
                    result = model(audio_data)
                    if return_segments:
                        text = result["text"].strip()
                        return [{"text": text, "start": 0, "end": 0}] if text else []
                return result["text"].strip()

            else:
                raise ValueError(f"Unknown model type: {model_type}")

        except Exception as e:
            print(f"Transcription error: {e}")
            return [] if return_segments else ""


class WhisperLiveTranscriber:
    """
    Streaming transcription using whisper-live approach.

    Uses a rolling numpy buffer instead of dual confirmed/active buffers.
    Segments are finalized when the same output is repeated N times (same_output_threshold).

    Ported from: Whisper-Live-main/whisper_live/backend/base.py
    """

    RATE = 16000  # Sample rate in Hz

    def __init__(
        self,
        sample_rate=16000,
        same_output_threshold=7,
        no_speech_thresh=0.45,
        send_last_n_segments=10,
    ):
        """
        Initialize the transcriber.

        Args:
            sample_rate: Audio sample rate (default 16000)
            same_output_threshold: Number of repeated outputs before finalizing (default 7)
            no_speech_thresh: Threshold for filtering no-speech segments (default 0.45)
            send_last_n_segments: Number of recent segments to keep (default 10)
        """
        self.RATE = sample_rate
        self.same_output_threshold = same_output_threshold
        self.no_speech_thresh = no_speech_thresh
        self.send_last_n_segments = send_last_n_segments

        # Frame buffer (numpy array of float32 audio samples)
        self.frames_np = None
        self.frames_offset = 0.0  # Time offset when buffer was clipped
        self.timestamp_offset = 0.0  # Current transcription position

        # Segment tracking
        self.transcript = []  # List of completed segments
        self.current_out = ""  # Current incomplete output
        self.prev_out = ""  # Previous output for comparison
        self.same_output_count = 0
        self.end_time_for_same_output = None
        self._last_seg_confidence = {}  # Word-level confidence from last segment

        # Threading lock for buffer access
        self.lock = threading.Lock()

    def _is_similar_output(self, text1, text2, threshold=0.85):
        """
        Check if two outputs are similar enough to count as 'same'.

        Uses fuzzy matching because Whisper often returns slightly different text
        each iteration (e.g., "I'm going to" vs "I'm gonna").

        Args:
            text1: First text to compare
            text2: Second text to compare
            threshold: Similarity threshold (0.0-1.0), default 0.85

        Returns:
            bool: True if texts are similar enough
        """
        if not text1 or not text2:
            return False
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio()
        return ratio >= threshold

    def add_frames(self, audio_bytes, sample_width=2):
        """
        Add audio frames to the buffer.

        Converts bytes to float32 numpy array and appends to rolling buffer.
        If buffer exceeds 45 seconds, clips oldest 30 seconds.

        Args:
            audio_bytes: Raw audio bytes (int16 PCM)
            sample_width: Bytes per sample (default 2 for int16)
        """
        # Convert bytes to float32 numpy array normalized to [-1, 1]
        frame_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        with self.lock:
            # Clip buffer if it exceeds 45 seconds
            if self.frames_np is not None and self.frames_np.shape[0] > 45 * self.RATE:
                self.frames_offset += 30.0
                self.frames_np = self.frames_np[int(30 * self.RATE):]
                # Ensure timestamp_offset doesn't fall behind frames_offset
                if self.timestamp_offset < self.frames_offset:
                    self.timestamp_offset = self.frames_offset

            # Append frames
            if self.frames_np is None:
                self.frames_np = frame_np.copy()
            else:
                self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)

    def get_audio_chunk_for_processing(self):
        """
        Get the next audio chunk to transcribe.

        Returns audio from timestamp_offset to end of buffer.

        Returns:
            tuple: (audio_np, duration) - audio as float32 numpy array and duration in seconds
        """
        with self.lock:
            if self.frames_np is None:
                return np.array([], dtype=np.float32), 0.0

            samples_to_skip = max(0, int((self.timestamp_offset - self.frames_offset) * self.RATE))
            audio_chunk = self.frames_np[samples_to_skip:].copy()

        duration = audio_chunk.shape[0] / self.RATE if audio_chunk.shape[0] > 0 else 0.0
        return audio_chunk, duration

    def get_buffer_duration(self):
        """Get total duration of audio in buffer in seconds."""
        with self.lock:
            if self.frames_np is None:
                return 0.0
            return self.frames_np.shape[0] / self.RATE

    def update_segments(self, segments, duration):
        """
        Process segments from Whisper transcription using Whisper-Live's approach.

        Key insight: Whisper returns segments with timestamps. We finalize all
        segments except the last one immediately. The last segment stays as
        "in-progress" and only finalizes when it repeats (same_output_threshold).

        This prevents overlapping text because we use Whisper's segment boundaries
        instead of guessing where phrases end.

        Args:
            segments: List of dicts with 'text', 'start', 'end' keys
            duration: Duration of the audio chunk that was transcribed

        Returns:
            dict: {
                'completed_segments': list of newly completed segments,
                'current_text': current incomplete text (last segment),
                'is_finalized': whether last segment was just finalized
            }
        """
        result = {
            'completed_segments': [],
            'current_text': '',
            'is_finalized': False
        }

        if not segments:
            return result

        # FIX: Detect garbage output from Whisper (overwhelmed by too much audio)
        # When Whisper gets 30+ seconds, it often returns just '...' or empty text
        all_text = ' '.join(seg.get('text', '').strip() for seg in segments)
        if len(all_text) < 5 or all_text == '...' or all_text.strip() == '':
            # Whisper is overwhelmed - force buffer trim and skip garbage
            with self.lock:
                if self.frames_np is not None:
                    buffer_duration = self.frames_np.shape[0] / self.RATE
                    current_pos = self.timestamp_offset - self.frames_offset
                    chunk_to_process = buffer_duration - current_pos
                    if chunk_to_process > 10:
                        # Force advance to keep only 10 seconds
                        extra_advance = chunk_to_process - 10
                        self.timestamp_offset += extra_advance
            return result  # Skip processing garbage segments

        offset = None

        # Process all segments except the last one (finalize them immediately)
        # This is the key difference from our previous approach
        if len(segments) > 1:
            for seg in segments[:-1]:
                text = seg.get('text', '').strip()
                if not text:
                    continue

                start = self.timestamp_offset + seg.get('start', 0)
                end = self.timestamp_offset + min(duration, seg.get('end', duration))

                if start >= end:
                    continue

                completed = {
                    'start': start,
                    'end': end,
                    'text': text,
                    'completed': True,
                }
                # Pass through word-level confidence data if available
                if 'words' in seg:
                    completed['words'] = seg['words']
                if 'avg_logprob' in seg:
                    completed['avg_logprob'] = seg['avg_logprob']
                if 'no_speech_prob' in seg:
                    completed['no_speech_prob'] = seg['no_speech_prob']
                if seg.get('language'):
                    completed['language'] = seg['language']
                self.transcript.append(completed)
                result['completed_segments'].append(completed)
                # print(f"[SEGMENT] Finalized: '{text[:50]}...' ({seg.get('start', 0):.1f}s-{seg.get('end', 0):.1f}s)" if len(text) > 50 else f"[SEGMENT] Finalized: '{text}'", flush=True)
                offset = min(duration, seg.get('end', duration))

        # Handle the last segment (in-progress until repeated)
        last_seg = segments[-1]
        self.current_out = last_seg.get('text', '').strip()
        # Store last segment's confidence data for finalization
        self._last_seg_confidence = {
            k: last_seg[k] for k in ('words', 'avg_logprob', 'no_speech_prob', 'language') if k in last_seg
        }
        result['current_text'] = self.current_out

        # Check if last segment is repeating (same_output_threshold logic)
        if self._is_similar_output(self.current_out, self.prev_out) and self.current_out:
            self.same_output_count += 1
            if self.end_time_for_same_output is None:
                self.end_time_for_same_output = last_seg.get('end', duration)

            # Debug logging for same_output tracking
        else:
            self.same_output_count = 0
            self.end_time_for_same_output = None

        # Finalize last segment if repeated enough times
        if self.same_output_count >= self.same_output_threshold:
            if self.current_out:
                completed = {
                    'start': self.timestamp_offset,
                    'end': self.timestamp_offset + min(duration, self.end_time_for_same_output or duration),
                    'text': self.current_out,
                    'completed': True,
                }
                # Pass through word-level confidence from last segment
                if 'words' in last_seg:
                    completed['words'] = last_seg['words']
                if 'avg_logprob' in last_seg:
                    completed['avg_logprob'] = last_seg['avg_logprob']
                if 'no_speech_prob' in last_seg:
                    completed['no_speech_prob'] = last_seg['no_speech_prob']
                if last_seg.get('language'):
                    completed['language'] = last_seg['language']
                self.transcript.append(completed)
                result['completed_segments'].append(completed)
                result['is_finalized'] = True
                # FIX: Save the text that was just finalized so phrase_complete knows not to re-process
                result['just_finalized_text'] = self.current_out
                print(f"[SAME_OUTPUT] Finalized last segment: '{self.current_out[:50]}...'" if len(self.current_out) > 50 else f"[SAME_OUTPUT] Finalized: '{self.current_out}'", flush=True)
                offset = min(duration, self.end_time_for_same_output or duration)

            # Reset
            self.current_out = ''
            self.prev_out = ''
            self.same_output_count = 0
            self.end_time_for_same_output = None
            result['current_text'] = ''
        else:
            self.prev_out = self.current_out

        # Advance timestamp_offset by the end of finalized segments
        if offset is not None:
            with self.lock:
                self.timestamp_offset += offset

        # PROACTIVE FIX: ALWAYS check buffer size and limit it, not just when offset is None
        # This prevents the buffer from slowly growing over time even when segments are finalizing
        with self.lock:
            if self.frames_np is not None:
                buffer_duration = self.frames_np.shape[0] / self.RATE
                current_pos = self.timestamp_offset - self.frames_offset
                chunk_to_process = buffer_duration - current_pos
                # If chunk exceeds 20 seconds, force advance to keep ~15 seconds
                # This is more aggressive than before to prevent Whisper from being overwhelmed
                if chunk_to_process > 20:
                    extra_advance = chunk_to_process - 15  # Keep 15 seconds
                    self.timestamp_offset += extra_advance

        return result

    def force_finalize(self):
        """
        Force finalize current text (e.g., on phrase timeout / silence detection).

        Returns:
            dict: Segment if there was text to finalize, None otherwise
        """
        if not self.current_out:
            return None

        # Get current duration from buffer
        _, duration = self.get_audio_chunk_for_processing()

        segment = {
            'start': self.timestamp_offset,
            'end': self.timestamp_offset + duration,
            'text': self.current_out,
            'completed': True,
        }
        # Include stored confidence data from last segment
        if hasattr(self, '_last_seg_confidence') and self._last_seg_confidence:
            segment.update(self._last_seg_confidence)
        self.transcript.append(segment)

        # Update timestamp offset
        with self.lock:
            self.timestamp_offset += duration

        # Reset
        self.current_out = ""
        self.prev_out = ""
        self.same_output_count = 0
        self.end_time_for_same_output = None

        return segment

    def get_recent_segments(self):
        """Get the most recent completed segments."""
        return self.transcript[-self.send_last_n_segments:] if self.transcript else []

    def get_all_text(self):
        """Get all transcribed text concatenated."""
        return " ".join(seg['text'] for seg in self.transcript if seg.get('text'))

    def reset(self):
        """Reset transcriber state for new session."""
        with self.lock:
            self.frames_np = None
            self.frames_offset = 0.0
            self.timestamp_offset = 0.0
        self.transcript = []
        self.current_out = ""
        self.prev_out = ""
        self.same_output_count = 0
        self.end_time_for_same_output = None


# Create shared state only in the main process.
# On macOS, 'spawn' is the default start method (safe — avoids ObjC/fork crashes after
# PyTorch/Whisper initialize the Objective-C runtime with background threads).
# On Linux, 'fork' is the default; forked children inherit these objects in memory.
# With spawn (macOS), the child re-imports this module and must NOT recreate the Manager
# (it would fail before bootstrap completes). Instead, the child receives these objects
# as pickled arguments to thread1_function and assigns them to module globals there.
if multiprocessing.current_process().name == 'MainProcess':
    mp_manager = multiprocessing.Manager()

    # Create multiprocessing Queue for config updates (hot-reload)
    config_queue = MPQueue()

    # Create multiprocessing Queue for control commands (start/stop)
    control_queue = MPQueue()

    # Create multiprocessing Queue for streaming audio to web clients
    audio_stream_queue = MPQueue(maxsize=10)

    # Global transcription state - use Manager.dict() for cross-process sharing
    transcription_state = mp_manager.dict(
        {
            "running": False,
            "status": "stopped",
            "message": "Transcription not started",
            "error": None,  # Error message if status == "error"
            "db_name": None,  # Shared database name for cross-process access
            "session_id": None,  # Stable per-session id (.db filename stem); cross-process
            "audio_level": 0,  # Audio level for histogram (0-100)
            "audio_db": -60,  # Audio level in decibels
            "audio_energy": 0,  # Raw audio energy (RMS)
            "start_time": 0,  # epoch seconds when transcription became active; 0 = not running
            "live_text": "",  # Live preview text (not yet saved to DB)
            "live_start": 0,  # Start time of the live preview within the session
            "live_end": 0,  # End time of the live preview within the session
            "live_word_confidences": [],  # Word-level confidence for the live preview
            "loaded_model": "",  # Name of the actual model that was loaded
            "audio_stream_enabled": False,  # Whether to stream audio to web clients
            "audio_type": None,  # "Speaking", "Music", or "Quiet" — PANNs detection (no_speech_prob fallback)
            "detection_mode": None,  # "panns" (tagger live) or "energy" (fallback) — which detector is actually running
        }
    )

    # Shared calibration state for cross-process communication
    calibration_state = mp_manager.dict(
        {
            "active": False,
            "step": 1,  # 1 = noise floor calibration, 2 = speech calibration
            "step1_complete": False,
            "start_time": 0,
            "duration": 15,  # 15 seconds per step (30 total)
            "speech_samples": 0,
            "noise_samples": 0,
            "silence_samples": 0,
        }
    )

    # Shared calibration data storage (Manager lists for cross-process)
    calibration_data_shared = mp_manager.dict(
        {
            "speech_samples": mp_manager.list(),
            "noise_samples": mp_manager.list(),
            "silence_durations": mp_manager.list(),
            "energy_levels": mp_manager.list(),
            "vad_probabilities": mp_manager.list(),
        }
    )

    # Step 1 calibration data (noise floor only)
    calibration_step1_data = mp_manager.dict(
        {
            "noise_energies": mp_manager.list(),
            "avg_noise": 0.0,
            "max_noise": 0.0,
        }
    )
else:
    # Spawned worker process: shared objects will be received as function arguments
    # and assigned to these globals at the top of thread1_function.
    mp_manager = None
    config_queue = None
    control_queue = None
    transcription_state = None
    calibration_state = None
    calibration_data_shared = None
    calibration_step1_data = None

# Global reference to transcription process for restart functionality
transcription_process = None

# Database cache for performance
_db_cache = {
    "last_entries": [],
    "last_fetch_time": 0,
    "cache_duration": 1.0,  # Cache for 1 second
}

# Thread locks for synchronization
_db_lock = threading.Lock()
_cache_lock = threading.Lock()
_transcription_state_lock = threading.Lock()
_transcription_start_lock = threading.Lock()  # Guards worker-process creation in the start route
_audio_queue_lock = threading.Lock()
# Generate the current date and time as a string
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M_%A")
current_year = datetime.now().strftime("%Y")
current_month = datetime.now().strftime("%Y-%m")

# Database will be created lazily when transcription starts
db_name = None  # Will be set when database is initialized
db_initialized = False
live_session_id = None  # Stable per-session id (the .db filename stem); set in initialize_database


# ============== PANNs audio tagger (music / speech detection) ==============
# Default checkpoint location (kept under APP_DIR so compiled builds stay self-contained)
PANNS_CHECKPOINT = os.path.join(APP_DIR, "panns_data", "Cnn14_mAP=0.431.pth")
PANNS_CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
# AudioSet class labels. panns_inference loads these at IMPORT TIME from a hardcoded
# ~/panns_data/class_labels_indices.csv and (over plain http) tries to wget them if
# absent. When that fetch is blocked/offline it leaves an empty file, which is never
# retried -> labels=[] -> classes_num=0 -> the 527-class CNN14 checkpoint fails to load
# -> music detection silently falls back to energy-only ("Speaking"/"Quiet", never
# "Music"). We ship the CSV under APP_DIR and place a valid copy before importing the
# library (with an https fallback) so detection works self-contained / offline.
PANNS_LABELS_FILENAME = "class_labels_indices.csv"
PANNS_LABELS_BUNDLED = os.path.join(APP_DIR, "panns_data", PANNS_LABELS_FILENAME)
PANNS_LABELS_URL = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
_PANNS_LABELS_MIN_BYTES = 1024  # a valid 527-row CSV is ~14 KB; smaller == missing/poisoned
_panns_labels_ready = False  # only need to repair the on-disk CSV once per process
_audio_tagger = None
_audio_tagger_failed_key = None  # (device, ckpt) that hit a real load error — don't retry it
_audio_tagger_key = None  # (device, ckpt) currently loaded
_panns_label_idx = None  # (music_idx_list, speech_idx_list)
_panns_missing_logged = False  # avoid spamming the "checkpoint missing" log


def panns_checkpoint_path(cfg=None):
    """Resolve the CNN14 checkpoint path (config override or the default location)."""
    cfg = cfg if cfg is not None else config.get("speech_type_detection", {})
    custom = (cfg.get("checkpoint_path", "") or "").strip()
    return custom or PANNS_CHECKPOINT


def panns_labels_home_path():
    """The path panns_inference.config hardcodes for the AudioSet label CSV."""
    return os.path.join(os.path.expanduser("~"), "panns_data", PANNS_LABELS_FILENAME)


def ensure_panns_labels_csv():
    """Guarantee a valid class_labels_indices.csv exists where panns_inference expects
    it, BEFORE the library is imported (it hardcodes ~/panns_data at import time).
    All PANNs data lives in APP_DIR/panns_data; ~/panns_data is kept as a symlink
    into it so nothing real is stored outside the app folder. Best-effort: never
    raises."""
    global _panns_labels_ready
    if _panns_labels_ready:
        return
    try:
        home_csv = panns_labels_home_path()
        home_dir = os.path.dirname(home_csv)
        app_dir = os.path.dirname(PANNS_LABELS_BUNDLED)

        # The app-folder copy is the real storage — make sure it's valid first
        # (https download; the library itself only tries plain http, often blocked).
        if not (os.path.exists(PANNS_LABELS_BUNDLED) and os.path.getsize(PANNS_LABELS_BUNDLED) >= _PANNS_LABELS_MIN_BYTES):
            os.makedirs(app_dir, exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(PANNS_LABELS_URL, PANNS_LABELS_BUNDLED)
            print(f"[PANNS] Downloaded AudioSet labels -> {PANNS_LABELS_BUNDLED}")
        if not (os.path.exists(PANNS_LABELS_BUNDLED) and os.path.getsize(PANNS_LABELS_BUNDLED) >= _PANNS_LABELS_MIN_BYTES):
            print("[PANNS] AudioSet labels unavailable; music detection will be unavailable")
            return

        # Point ~/panns_data at the app folder. Migrate a stale real directory
        # only when it holds nothing but the label CSV (never delete unknown data).
        if os.path.islink(home_dir):
            if os.path.realpath(home_dir) != os.path.realpath(app_dir):
                os.unlink(home_dir)
        elif os.path.isdir(home_dir):
            if all(name == PANNS_LABELS_FILENAME for name in os.listdir(home_dir)):
                shutil.rmtree(home_dir)
        if not os.path.lexists(home_dir):
            try:
                os.symlink(app_dir, home_dir)
                print(f"[PANNS] Linked {home_dir} -> {app_dir}")
            except OSError:
                pass

        # Fallback when the symlink couldn't be made (or a real dir with other
        # content remains): copy the CSV like before.
        if not (os.path.exists(home_csv) and os.path.getsize(home_csv) >= _PANNS_LABELS_MIN_BYTES):
            os.makedirs(home_dir, exist_ok=True)
            shutil.copyfile(PANNS_LABELS_BUNDLED, home_csv)
            print(f"[PANNS] Installed AudioSet labels from bundled copy -> {home_csv}")

        if os.path.exists(home_csv) and os.path.getsize(home_csv) >= _PANNS_LABELS_MIN_BYTES:
            _panns_labels_ready = True
        else:
            print("[PANNS] AudioSet labels still missing after repair attempt; music detection will be unavailable")
    except Exception as e:
        print(f"[PANNS] Could not install AudioSet labels: {e}")


def panns_package_installed():
    try:
        import importlib.util
        return importlib.util.find_spec("panns_inference") is not None
    except Exception:
        return False


def get_audio_tagger(cfg):
    """Lazy-load the PANNs CNN14 tagger for the given speech_type_detection cfg.
    Reloads if device/checkpoint changed. Returns None if unavailable (missing
    package or checkpoint). MUST be called off the audio-drain path (it can block
    for seconds on first load)."""
    global _audio_tagger, _audio_tagger_failed_key, _audio_tagger_key, _panns_label_idx, _panns_missing_logged
    # Make sure the AudioSet label CSV is valid before panns_inference is imported
    # below (it loads labels once at import time). Without this the checkpoint fails
    # to load and detection silently degrades to energy-only.
    ensure_panns_labels_csv()
    ckpt = panns_checkpoint_path(cfg)
    device = cfg.get("device", "cpu") or "cpu"
    key = (device, ckpt)
    if _audio_tagger is not None and _audio_tagger_key == key:
        return _audio_tagger
    if _audio_tagger_failed_key == key:
        return None
    # Missing checkpoint is *transient* (a download in the main process may produce
    # it later) — recheck cheaply each call instead of failing permanently.
    if not os.path.exists(ckpt):
        if not _panns_missing_logged:
            print(f"[PANNS] Checkpoint not found at {ckpt}; music detection falls back to energy-based until it's downloaded")
            _panns_missing_logged = True
        return None
    # Config changed (device/checkpoint) or first load: drop any stale model.
    _audio_tagger = None
    _audio_tagger_key = None
    try:
        from panns_inference import AudioTagging, labels
        tagger = AudioTagging(checkpoint_path=ckpt, device=device)
        music_idx = [i for i, l in enumerate(labels) if l in ("Music", "Singing")]
        _panns_label_idx = (music_idx, None)
        _audio_tagger = tagger
        _audio_tagger_key = key
        _audio_tagger_failed_key = None
        _panns_missing_logged = False
        print(f"[PANNS] Audio tagger loaded on {device} from {ckpt}")
        return _audio_tagger
    except Exception as e:
        _audio_tagger_failed_key = key
        print(f"[PANNS] Could not load audio tagger: {e}; falling back to energy-based")
        return None


def unload_audio_tagger():
    """Release the tagger (called when transcription stops/unloads to free VRAM).
    The detector reloads it lazily if detection is used again."""
    global _audio_tagger, _audio_tagger_key, _audio_tagger_failed_key, _panns_missing_logged
    _audio_tagger = None
    _audio_tagger_key = None
    _audio_tagger_failed_key = None
    _panns_missing_logged = False


def compute_music_prob(audio_np, sr, cfg):
    """Return (music_prob, dominant_tag) for an audio buffer using PANNs, or
    (None, None) when the tagger is unavailable. audio_np: float32 mono in [-1, 1]."""
    tagger = get_audio_tagger(cfg)
    if tagger is None or audio_np is None or len(audio_np) == 0:
        return None, None
    try:
        from panns_inference import labels
        wav = np.asarray(audio_np, dtype=np.float32)
        if sr != 32000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=32000)
        clipwise, _ = tagger.inference(wav[None, :])
        clip = clipwise[0]
        music_idx, _ = _panns_label_idx
        music_prob = float(max((clip[i] for i in music_idx), default=0.0))
        return music_prob, labels[int(np.argmax(clip))]
    except Exception as e:
        print(f"[PANNS] inference error: {e}")
        return None, None


def panns_label_from_prob(smoothed_prob, audio_db, cfg):
    """Map a (smoothed) music probability to a Speaking/Music/Quiet label."""
    quiet_db_threshold = cfg.get("quiet_db_threshold", -40)
    if smoothed_prob > cfg.get("music_prob_threshold", 0.5):
        return "Music"
    return "Quiet" if (audio_db or -60) <= quiet_db_threshold else "Speaking"


def classify_audio_type(audio_db, cfg=None):
    """Energy-based fallback label (no PANNs): audible => Speaking, else Quiet.
    We never claim Music without the PANNs detector."""
    cfg = cfg if cfg is not None else config.get("speech_type_detection", {})
    quiet_db_threshold = cfg.get("quiet_db_threshold", -40)
    return "Speaking" if (audio_db or -60) > quiet_db_threshold else "Quiet"


class MusicDetector:
    """Runs PANNs inference on a dedicated daemon thread so the audio-drain loop
    never blocks. Latest-buffer / drop-stale: submit() just hands off the newest
    buffer; the thread loads the model (off the hot path), throttles, smooths, and
    writes music_prob / audio_tag / audio_type into the shared transcription state.
    Reads speech_type_detection live from process_config each iteration -> hot-reload."""

    def __init__(self):
        self.cfg_root = {}      # live process_config (refreshed on each submit)
        self.state = None       # shared transcription_state
        self._buf = None
        self._sr = 16000
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._history = []      # instance-local smoothing buffer (no cross-thread race)
        self._last_ts = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True, name="panns-detector")
        self._thread.start()

    def submit(self, process_config, state, audio_np, sr):
        """Non-blocking hand-off of the latest raw (pre-VAD) buffer."""
        self.cfg_root = process_config
        self.state = state
        with self._lock:
            self._buf = audio_np
            self._sr = sr
        self._event.set()

    def _run(self):
        while True:
            self._event.wait(timeout=1.0)
            self._event.clear()
            cfg = (self.cfg_root or {}).get("speech_type_detection", {})
            if not cfg.get("enabled", True) or cfg.get("method", "panns") != "panns":
                self._history.clear()
                # Clear stale PANNs state so the live monitor doesn't keep showing
                # the last music label after detection is disabled mid-session.
                st = self.state
                if st is not None:
                    st["detection_mode"] = "energy"
                    if st.get("music_prob") is not None:
                        st["music_prob"] = None
                        st["audio_tag"] = None
                        st["audio_type"] = None
                continue
            now = time.time()
            if now - self._last_ts < 0.4:  # throttle to ~2-3 runs/sec
                continue
            with self._lock:
                buf = self._buf
                sr = self._sr
                self._buf = None  # consume: don't re-run on a stale buffer once audio stops
            if buf is None:
                continue
            self._last_ts = now
            try:
                music_prob, tag = compute_music_prob(buf, sr, cfg)
            except Exception as e:
                print(f"[PANNS] detector error: {e}")
                continue
            if music_prob is None:
                # PANNs enabled but the tagger is unavailable (missing/failed load):
                # finalized_audio_type falls back to the energy-based label.
                st = self.state
                if st is not None:
                    st["detection_mode"] = "energy"
                continue
            window = max(1, int(cfg.get("smoothing_window", 4) or 1))
            self._history.append(float(music_prob))
            del self._history[:-window]
            smoothed = sum(self._history) / len(self._history)
            st = self.state
            if st is not None:
                st["detection_mode"] = "panns"
                st["music_prob"] = music_prob
                st["audio_tag"] = tag
                # Live TYPE for the monitor, even when this audio isn't transcribed.
                st["audio_type"] = panns_label_from_prob(smoothed, st.get("audio_db"), cfg)


_music_detector = None


def submit_music_detection(process_config, state, audio_np, sr):
    """Hand the latest pre-VAD buffer to the background detector (non-blocking).
    Creates/starts the detector thread on first use."""
    global _music_detector
    if _music_detector is None:
        _music_detector = MusicDetector()
    _music_detector.submit(process_config, state, audio_np, sr)


def finalized_audio_type(process_config, state):
    """Label for a finalized (transcribed) segment: the detector's live PANNs
    label when active, else the energy-based fallback."""
    cfg = process_config.get("speech_type_detection", {})
    if (cfg.get("enabled", True) and cfg.get("method", "panns") == "panns"
            and state.get("music_prob") is not None):
        return state.get("audio_type") or "Speaking"
    return classify_audio_type(state.get("audio_db"), cfg)


def words_to_session_ms(completed_segments):
    """Flatten faster-whisper word lists from a batch of completed segments into an
    ordered stream of {w, s_ms, e_ms, c} dicts on a session-relative millisecond
    timeline (so words from every row share one timeline for offline replay).

    Word timings are chunk-relative seconds while each segment's `start` is
    session-relative; the segments in one batch share a chunk base, so a single
    offset (the first word-bearing segment's start minus its first word's start)
    maps the whole stream onto the session timeline. `w` is the verbatim
    faster-whisper token (never normalized). Returns [] when no segment carries
    word data (e.g. the standard openai-whisper backend, which emits no words[]).
    """
    segs = [s for s in (completed_segments or []) if s.get('words')]
    if not segs:
        return []
    fw = segs[0]['words']
    try:
        off = (segs[0].get('start') or 0) - (fw[0].get('start') or 0)
    except Exception:
        off = 0
    stream = []
    for seg in segs:
        for w in seg.get('words', []):
            ws, we = w.get('start'), w.get('end')
            ws = ws if ws is not None else 0
            we = we if we is not None else ws
            stream.append({
                "w": w.get('word', ''),
                "s_ms": round((ws + off) * 1000),
                "e_ms": round((we + off) * 1000),
                "c": w.get('probability'),
            })
    return stream


def attribute_words_to_sentences(stream, num_sentences):
    """Assign each word in an ordered session-ms `stream` to one of `num_sentences`
    re-split rows by MAX TEMPORAL OVERLAP, so no boundary word is ever dropped
    (re-splits fall on pauses, exactly where book tokens sit).

    Provisional per-sentence spans are found by cutting the stream at its
    `num_sentences-1` largest inter-word gaps (the pauses the sentence splitter
    used) — robust without relying on token text. Each word is then assigned to the
    span it overlaps most; ties or gap-words go to the earlier sentence. Returns a
    list of `num_sentences` word-lists; every word lands in exactly one list.
    """
    n = max(1, num_sentences)
    groups = [[] for _ in range(n)]
    if not stream:
        return groups
    if n == 1:
        groups[0] = list(stream)
        return groups
    # Cut at the largest inter-word gaps -> provisional contiguous groups.
    gaps = [(stream[i]["s_ms"] - stream[i - 1]["e_ms"], i) for i in range(1, len(stream))]
    cut_count = min(n - 1, len(gaps))
    cut_idx = sorted(i for _, i in sorted(gaps, key=lambda g: g[0], reverse=True)[:cut_count])
    prov, start = [], 0
    for ci in cut_idx:
        prov.append(stream[start:ci])
        start = ci
    prov.append(stream[start:])
    while len(prov) < n:
        prov.append([])
    prov = prov[:n]
    spans = [((g[0]["s_ms"], g[-1]["e_ms"]) if g else None) for g in prov]

    def _overlap(w, span):
        if span is None:
            return None
        return min(w["e_ms"], span[1]) - max(w["s_ms"], span[0])

    for w in stream:
        best_j, best_ov = 0, None
        for j, span in enumerate(spans):
            ov = _overlap(w, span)
            if ov is None:
                continue
            if best_ov is None or ov > best_ov:
                best_ov, best_j = ov, j
        groups[best_j].append(w)
    return groups


def words_json_or_none(word_objs):
    """Serialize a list of {w, s_ms, e_ms, c} word dicts to a JSON array string for
    the `words_json` column, or None when empty (NULL = no per-word data — see
    `words_source` to tell 'backend has no words' from an alignment miss)."""
    if not word_objs:
        return None
    try:
        return json.dumps(word_objs, ensure_ascii=False)
    except Exception:
        return None


def initialize_database():
    """Initialize database only when transcription starts (lazy loading)"""
    global db_name, db_initialized, live_session_id

    if db_initialized:
        return db_name

    # Get custom database path from config or use default
    custom_db_path = config.get("database", {}).get("path", "").strip()
    path_format = config.get("database", {}).get("path_format", "").strip() or "%Y/%m"
    now = datetime.now()
    formatted_path = now.strftime(path_format)

    if custom_db_path:
        # Use custom base path + path_format subdirectory
        folder_name = os.path.join(custom_db_path, formatted_path)
        print(f"[OK] Using custom database path: {folder_name}")
    else:
        # Use default base path (under APP_DIR) + path_format subdirectory so compiled
        # builds keep the DB in ~/.stt instead of the launch directory.
        folder_name = os.path.join(BACKUP_DIR, formatted_path)
        print(f"[OK] Using default database path: {folder_name}")

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    # Make the DB directory tree readable/traversable by all users (consumers read these)
    make_dirs_world_readable(folder_name, custom_db_path or BACKUP_DIR)

    # Create database file path with configurable format (using Python strftime format)
    filename_format = config.get("database", {}).get(
        "filename_format", ""
    ).strip() or "%Y-%m-%d_%H%M%S"

    # Validate that format includes time component for unique per-session databases
    if not any(time_fmt in filename_format for time_fmt in ["%H", "%M", "%S"]):
        print(f"[WARNING] Database filename_format '{filename_format}' does not include time component.")
        print("[WARNING] This may cause sessions on the same day to share a database.")
        print("[WARNING] Using default format: %Y-%m-%d_%H%M%S for unique per-session databases.")
        filename_format = "%Y-%m-%d_%H%M%S"

    now = datetime.now()

    # Use strftime directly with user's format
    formatted_filename = now.strftime(filename_format)

    # Get custom filename prefix or use default
    filename_prefix = config.get("database", {}).get("filename_prefix", "").strip()
    if filename_prefix:
        db_name = os.path.join(
            folder_name, f"{formatted_filename}_{filename_prefix}.db"
        )
        print(
            f"[OK] Using custom database filename: {formatted_filename}_{filename_prefix}.db"
        )
    else:
        db_name = os.path.join(folder_name, f"{formatted_filename}.db")
        print(
            f"[OK] Using default database filename: {formatted_filename}.db"
        )

    print(f"[OK] Initializing database: {db_name}")

    try:
        # Use context manager for database connection
        with sqlite3.connect(db_name) as db_connection:
            db_cursor = db_connection.cursor()

            # Enable WAL mode for better concurrent read/write performance
            # WAL (Write-Ahead Logging) allows simultaneous reads while writing
            db_cursor.execute("PRAGMA journal_mode=WAL")
            db_cursor.execute("PRAGMA synchronous=NORMAL")  # Faster writes, still safe

            # Create the table if it doesn't exist (with start_time, end_time for temporal ordering)
            db_cursor.execute(
                """CREATE TABLE IF NOT EXISTS transcriptions (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, text TEXT, start_time REAL DEFAULT 0, end_time REAL DEFAULT 0)"""
            )

            # Create index on timestamp for faster ORDER BY queries
            db_cursor.execute(
                """CREATE INDEX IF NOT EXISTS idx_timestamp ON transcriptions(timestamp DESC)"""
            )

            # Migration: Check if id column exists, add it if missing
            db_cursor.execute("PRAGMA table_info(transcriptions)")
            columns = [row[1] for row in db_cursor.fetchall()]
            if "id" not in columns:
                print("[DB] Migrating database: adding id column...")
                # SQLite doesn't support ALTER TABLE ADD COLUMN with PRIMARY KEY
                # So we need to recreate the table (wrapped in transaction for safety)
                try:
                    db_cursor.execute("BEGIN")
                    db_cursor.execute(
                        """CREATE TABLE transcriptions_new (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, text TEXT)"""
                    )
                    db_cursor.execute(
                        """INSERT INTO transcriptions_new (timestamp, text) SELECT timestamp, text FROM transcriptions"""
                    )
                    db_cursor.execute("""DROP TABLE transcriptions""")
                    db_cursor.execute(
                        """ALTER TABLE transcriptions_new RENAME TO transcriptions"""
                    )
                    # Recreate index
                    db_cursor.execute(
                        """CREATE INDEX IF NOT EXISTS idx_timestamp ON transcriptions(timestamp DESC)"""
                    )
                    db_connection.commit()
                    print("[DB] OK: Migration complete")
                except Exception:
                    db_connection.rollback()
                    raise

            # Migration: Check if start_time/end_time columns exist, add them if missing
            db_cursor.execute("PRAGMA table_info(transcriptions)")
            columns = [row[1] for row in db_cursor.fetchall()]
            if "start_time" not in columns:
                print("[DB] Migrating database: adding start_time and end_time columns...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN start_time REAL DEFAULT 0")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN end_time REAL DEFAULT 0")
                db_connection.commit()
                print("[DB] OK: Migration complete (added temporal columns)")

            # Migration: Add corrections-related columns
            db_cursor.execute("PRAGMA table_info(transcriptions)")
            columns = [row[1] for row in db_cursor.fetchall()]
            if "confidence" not in columns:
                print("[DB] Migrating database: adding corrections columns (confidence, original_text, corrected_by, needs_review)...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN confidence REAL DEFAULT NULL")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN original_text TEXT DEFAULT NULL")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN corrected_by TEXT DEFAULT NULL")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN needs_review INTEGER DEFAULT 0")
                db_cursor.execute("CREATE INDEX IF NOT EXISTS idx_needs_review ON transcriptions(needs_review) WHERE needs_review = 1")
                db_connection.commit()
                print("[DB] OK: Migration complete (added corrections columns)")

            if "translated_text" not in columns:
                print("[DB] Migrating database: adding translation columns (translated_text, translation_language)...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN translated_text TEXT DEFAULT NULL")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN translation_language TEXT DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added translation columns)")

            if "speech_type" not in columns:
                print("[DB] Migrating database: adding speech_type column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN speech_type TEXT DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added speech_type column)")

            if "audio_tag" not in columns:
                print("[DB] Migrating database: adding audio_tag column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN audio_tag TEXT DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added audio_tag column)")

            if "music_prob" not in columns:
                print("[DB] Migrating database: adding music_prob column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN music_prob REAL DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added music_prob column)")

            if "denied" not in columns:
                # `denied` is a UI visibility/hide flag (0 = visible, 1 = hidden from
                # the transcript view); toggled by handle_set_segment_denied.
                print("[DB] Migrating database: adding denied column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN denied INTEGER DEFAULT 0")
                db_connection.commit()
                print("[DB] OK: Migration complete (added denied column)")

            # Schema v2: additive columns for the downstream consumer (epoch-ms ordering,
            # per-word timing/confidence, partial/final flag, source language, segment pairing).
            # All nullable / defaulted so old .db files and readers keep working unchanged.
            db_cursor.execute("PRAGMA table_info(transcriptions)")
            columns = [row[1] for row in db_cursor.fetchall()]
            if "ts_ms" not in columns:
                print("[DB] Migrating database: adding ts_ms column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN ts_ms INTEGER DEFAULT NULL")
                db_cursor.execute("CREATE INDEX IF NOT EXISTS idx_ts_ms ON transcriptions(ts_ms)")
                db_connection.commit()
                print("[DB] OK: Migration complete (added ts_ms column)")
            if "words_json" not in columns:
                print("[DB] Migrating database: adding words_json column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN words_json TEXT DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added words_json column)")
            if "is_final" not in columns:
                print("[DB] Migrating database: adding is_final column...")
                # 1 = finalized (all existing rows are finals); 0 = partial hypothesis
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN is_final INTEGER DEFAULT 1")
                db_connection.commit()
                print("[DB] OK: Migration complete (added is_final column)")
            if "partial_seq" not in columns:
                print("[DB] Migrating database: adding partial_seq column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN partial_seq INTEGER DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added partial_seq column)")
            if "source_language" not in columns:
                print("[DB] Migrating database: adding source_language column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN source_language TEXT DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added source_language column)")
            if "segment_id" not in columns:
                print("[DB] Migrating database: adding segment_id column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN segment_id TEXT DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added segment_id column)")
            if "words_source" not in columns:
                # ASR backend that produced the row (faster_whisper/whisper/...). Makes a
                # NULL words_json interpretable: 'whisper' emits no per-word data, so NULL
                # is expected there, vs 'faster_whisper' where NULL would be unexpected.
                print("[DB] Migrating database: adding words_source column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN words_source TEXT DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added words_source column)")
            if "session_id" not in columns:
                # Stable per-session id (the .db filename stem) on every row, so the
                # consumer can anchor socket<->db and group rows by session.
                print("[DB] Migrating database: adding session_id column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN session_id TEXT DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added session_id column)")
            if "denied_reason" not in columns:
                # Why the row was denied: 'hallucination', 'cjk', 'cjk_shadow', 'short', 'dup'.
                # NULL means the row is a normal visible segment.
                print("[DB] Migrating database: adding denied_reason column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN denied_reason TEXT DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added denied_reason column)")

            # Insert a blank first entry with default values
            default_timestamp = " "
            default_text = " "
            db_cursor.execute(
                "INSERT INTO transcriptions (timestamp, text) VALUES (?, ?)",
                (default_timestamp, default_text),
            )
            db_connection.commit()

        db_initialized = True
        # Make the DB file (and any WAL/SHM sidecars) readable by all users
        make_db_world_readable(db_name)
        # Stable per-session id = the .db filename stem (e.g. 2026-06-22_183007).
        # Stored on every row and emitted top-level on every socket payload so the
        # consumer can anchor socket<->db by exact match; re-derived each session.
        live_session_id = os.path.splitext(os.path.basename(db_name))[0]
        # Store database name + session id in shared state for web server access
        transcription_state["db_name"] = db_name
        transcription_state["session_id"] = live_session_id
        print("[OK] Database initialized successfully")

        return db_name
    except Exception as e:
        print(f"[ERROR] Failed to initialize database: {e}")
        # Clean up database file if initialization failed
        if db_name and os.path.exists(db_name):
            try:
                os.unlink(db_name)
            except OSError:
                pass
        raise


# File Transcription Helper Functions

SUPPORTED_AUDIO_FORMATS = ["mp3", "wav", "flac", "ogg", "m4a", "aac", "wma", "opus"]
SUPPORTED_VIDEO_FORMATS = [
    "mp4",
    "mkv",
    "avi",
    "mov",
    "wmv",
    "flv",
    "webm",
    "mpg",
    "mpeg",
    "m4v",
    "3gp",
]


def extract_audio_from_file(file_path):
    """
    Extract audio from video/audio file using pydub.
    Converts to WAV 16kHz mono for transcription.

    Args:
        file_path: Path to audio/video file

    Returns:
        Path to converted WAV file
    """
    temp_wav = None
    try:
        # Get file extension
        ext = os.path.splitext(file_path)[1].lower().replace(".", "")

        # Load file
        _lazy_import_audio()
        if ext in SUPPORTED_VIDEO_FORMATS + SUPPORTED_AUDIO_FORMATS:
            audio = AudioSegment.from_file(file_path, format=ext)
        else:
            audio = AudioSegment.from_file(file_path)

        # Convert to WAV 16kHz mono
        audio = audio.set_frame_rate(16000)
        audio = audio.set_channels(1)

        # Save as WAV with proper cleanup
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.close()  # Close the file handle before pydub writes to it
        audio.export(temp_wav.name, format="wav")

        return temp_wav.name

    except Exception as e:
        # Clean up temp file if it exists
        if temp_wav and os.path.exists(temp_wav.name):
            try:
                os.unlink(temp_wav.name)
            except OSError:
                pass
        raise Exception(f"Failed to extract audio: {str(e)}") from e


def format_transcription(segments, format_type):
    """
    Format transcription segments into requested format.

    Args:
        segments: List of (text, start_time, end_time) tuples
        format_type: One of 'txt', 'srt', 'vtt', 'json'

    Returns:
        Formatted transcription string
    """
    if format_type == "txt":
        # Plain text - just concatenate all text
        return "\n".join([seg["text"] for seg in segments])

    elif format_type == "srt":
        # SubRip format
        output = []
        for i, seg in enumerate(segments, 1):
            start = format_timestamp_srt(seg["start"])
            end = format_timestamp_srt(seg["end"])
            output.append(f"{i}\n{start} --> {end}\n{seg['text']}\n")
        return "\n".join(output)

    elif format_type == "vtt":
        # WebVTT format
        output = ["WEBVTT\n"]
        for seg in segments:
            start = format_timestamp_vtt(seg["start"])
            end = format_timestamp_vtt(seg["end"])
            output.append(f"{start} --> {end}\n{seg['text']}\n")
        return "\n".join(output)

    elif format_type == "json":
        # JSON format with metadata
        return json.dumps(
            {
                "segments": segments,
                "total_segments": len(segments),
                "duration": segments[-1]["end"] if segments else 0,
            },
            indent=2,
        )

    else:
        return "\n".join([seg["text"] for seg in segments])


def format_timestamp_srt(seconds):
    """Format timestamp for SRT format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds):
    """Format timestamp for VTT format: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def convert_db_to_srt(db_path):
    """
    Convert a Transcriptions.db file to SRT subtitle format.

    Args:
        db_path: Path to the .db file

    Returns:
        Path to the created .srt file, or None on failure
    """
    from datetime import datetime

    try:
        if not db_path or not os.path.exists(db_path):
            print(f"[SRT] Database not found: {db_path}")
            return None

        # Read entries from database
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT timestamp, text FROM transcriptions
                WHERE timestamp IS NOT NULL AND timestamp != ''
                AND text IS NOT NULL AND TRIM(text) != '' AND TRIM(text) != ' '
                AND COALESCE(denied, 0) = 0
                ORDER BY id ASC
            """
            )
            entries = cursor.fetchall()

        if not entries:
            print("[SRT] No valid entries found in database")
            return None

        # Parse timestamps and calculate durations
        segments = []
        first_time = None

        for i, (timestamp_str, text) in enumerate(entries):
            try:
                # Parse ISO timestamp (handles both "2025-01-02 12:34:56" and "2025-01-02T12:34:56")
                ts_normalized = timestamp_str.replace("T", " ").strip()
                dt = datetime.strptime(ts_normalized, "%Y-%m-%d %H:%M:%S")

                if first_time is None:
                    first_time = dt

                # Calculate seconds from start
                start_seconds = (dt - first_time).total_seconds()

                # Estimate end time (use next entry's start or add 3 seconds)
                if i + 1 < len(entries):
                    next_ts = entries[i + 1][0].replace("T", " ").strip()
                    next_dt = datetime.strptime(next_ts, "%Y-%m-%d %H:%M:%S")
                    end_seconds = (next_dt - first_time).total_seconds()
                    # Ensure minimum 1 second duration
                    if end_seconds <= start_seconds:
                        end_seconds = start_seconds + 1.0
                else:
                    end_seconds = start_seconds + 3.0

                segments.append(
                    {"text": text.strip(), "start": start_seconds, "end": end_seconds}
                )
            except Exception as e:
                print(f"[SRT] Error parsing entry {i}: {e}")
                continue

        if not segments:
            print("[SRT] No valid segments after parsing")
            return None

        # Format as SRT using existing function
        srt_content = format_transcription(segments, "srt")

        # Write SRT file alongside the database
        srt_path = db_path.replace(".db", ".srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        print(f"[SRT] Created: {srt_path} ({len(segments)} entries)")

        # Also generate HTML with word highlighting (if enabled)
        # Reload config to get fresh settings (global config may be stale)
        html_enabled = load_config().get("database", {}).get("html_enabled", True)
        if html_enabled:
            try:
                convert_db_to_html(db_path)
            except Exception as e:
                print(f"[SRT] Failed to generate HTML: {e}")
        else:
            print("[HTML] HTML generation disabled in settings")

        return srt_path
    except Exception as e:
        print(f"[SRT] Error converting database to SRT: {e}")
        import traceback

        traceback.print_exc()
        return None


def convert_db_to_translation_srt(db_path):
    """Convert translated text from a Transcriptions.db to SRT subtitle format."""
    from datetime import datetime

    try:
        if not db_path or not os.path.exists(db_path):
            return None

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT timestamp, translated_text FROM transcriptions
                WHERE timestamp IS NOT NULL AND timestamp != ''
                AND translated_text IS NOT NULL AND TRIM(translated_text) != ''
                AND COALESCE(denied, 0) = 0
                ORDER BY id ASC
            """
            )
            entries = cursor.fetchall()

        if not entries:
            print("[SRT-TRANSLATION] No translated entries found in database")
            return None

        segments = []
        first_time = None

        for i, (timestamp_str, text) in enumerate(entries):
            try:
                ts_normalized = timestamp_str.replace("T", " ").strip()
                dt = datetime.strptime(ts_normalized, "%Y-%m-%d %H:%M:%S")

                if first_time is None:
                    first_time = dt

                start_seconds = (dt - first_time).total_seconds()

                if i + 1 < len(entries):
                    next_ts = entries[i + 1][0].replace("T", " ").strip()
                    next_dt = datetime.strptime(next_ts, "%Y-%m-%d %H:%M:%S")
                    end_seconds = (next_dt - first_time).total_seconds()
                    if end_seconds <= start_seconds:
                        end_seconds = start_seconds + 1.0
                else:
                    end_seconds = start_seconds + 3.0

                segments.append(
                    {"text": text.strip(), "start": start_seconds, "end": end_seconds}
                )
            except Exception:
                continue

        if not segments:
            return None

        srt_content = format_transcription(segments, "srt")
        srt_path = db_path.replace(".db", ".translated.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        print(f"[SRT-TRANSLATION] Created: {srt_path} ({len(segments)} entries)")
        return srt_path
    except Exception as e:
        print(f"[SRT-TRANSLATION] Error: {e}")
        return None


def apply_word_highlighting_server(text, config):
    """
    Apply word highlighting to text using the highlighting configuration.
    Server-side equivalent of the JavaScript applyWordHighlighting() function.

    Args:
        text: The text to apply highlighting to
        config: The word highlighting configuration dict

    Returns:
        Text with HTML span tags for highlighting
    """
    import html
    import re

    if not config or not config.get("enabled") or not config.get("words"):
        return html.escape(text)

    # First escape HTML entities in the original text
    result = html.escape(text)

    # Get list of disabled color groups
    disabled_colors = config.get("disabled_colors", [])

    for word_entry in config["words"]:
        word = word_entry.get("word", "")
        color = word_entry.get("color", "#ffff00")
        is_regex = word_entry.get("is_regex", False)
        case_sensitive = word_entry.get("case_sensitive", False)

        if not word:
            continue

        # Skip if this color group is disabled
        if color in disabled_colors:
            continue

        try:
            flags = 0 if case_sensitive else re.IGNORECASE

            if is_regex:
                # Use the pattern as-is for regex mode
                # Use word boundaries that work with Unicode
                pattern = r"(?<![A-Za-z\u0400-\u04FF0-9])(" + word + r")(?![A-Za-z\u0400-\u04FF0-9])"
            else:
                # Escape special regex characters for plain words
                escaped_word = re.escape(word)
                pattern = r"(?<![A-Za-z\u0400-\u04FF0-9])(" + escaped_word + r")(?![A-Za-z\u0400-\u04FF0-9])"

            # Replace matches with highlighted spans
            result = re.sub(
                pattern,
                f'<span style="color: {color};">\\1</span>',
                result,
                flags=flags,
            )
        except re.error as e:
            print(f"[HTML] Invalid regex pattern '{word}': {e}")
            continue

    return result


def convert_db_to_html(db_path):
    """
    Convert a Transcriptions.db file to HTML format with word highlighting.

    Args:
        db_path: Path to the .db file

    Returns:
        Path to the created .html file, or None on failure
    """
    from datetime import datetime

    try:
        if not db_path or not os.path.exists(db_path):
            print(f"[HTML] Database not found: {db_path}")
            return None

        # Load word highlighting configuration
        highlight_config = None
        highlight_config_path = os.path.join(
            APP_DIR, "word_highlighting.json"
        )
        if os.path.exists(highlight_config_path):
            try:
                with open(highlight_config_path, "r", encoding="utf-8") as f:
                    highlight_config = json.load(f)
            except Exception as e:
                print(f"[HTML] Error loading word highlighting config: {e}")

        # Read entries from database
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT timestamp, text FROM transcriptions
                WHERE timestamp IS NOT NULL AND timestamp != ''
                AND text IS NOT NULL AND TRIM(text) != '' AND TRIM(text) != ' '
                AND COALESCE(denied, 0) = 0
                ORDER BY id ASC
            """
            )
            entries = cursor.fetchall()

        if not entries:
            print("[HTML] No valid entries found in database")
            return None

        # Parse timestamps and build segments
        segments_html = []
        first_time = None

        for i, (timestamp_str, text) in enumerate(entries):
            try:
                # Parse ISO timestamp
                ts_normalized = timestamp_str.replace("T", " ").strip()
                dt = datetime.strptime(ts_normalized, "%Y-%m-%d %H:%M:%S")

                if first_time is None:
                    first_time = dt

                # Calculate both clock time and elapsed time
                clock_time = dt.strftime("%H:%M:%S")
                elapsed = dt - first_time
                elapsed_hours = int(elapsed.total_seconds() // 3600)
                elapsed_minutes = int((elapsed.total_seconds() % 3600) // 60)
                elapsed_seconds = int(elapsed.total_seconds() % 60)
                elapsed_time = f"{elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}"

                # Apply word highlighting
                highlighted_text = apply_word_highlighting_server(text.strip(), highlight_config)

                segments_html.append(
                    f'<div class="segment"><span class="timestamp" data-clock="{clock_time}" data-elapsed="{elapsed_time}">[{clock_time}]</span><span class="text">{highlighted_text}</span></div>'
                )
            except Exception as e:
                print(f"[HTML] Error parsing entry {i}: {e}")
                continue

        if not segments_html:
            print("[HTML] No valid segments after parsing")
            return None

        # Get the date for the title
        title_date = first_time.strftime("%Y-%m-%d %H:%M") if first_time else "Unknown"

        # Build the HTML document
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transcription - {title_date}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: #252525;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }}
        h1 {{
            color: #bb86fc;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #bb86fc;
        }}
        .meta {{
            color: #888;
            font-size: 0.9em;
            margin-bottom: 25px;
        }}
        .segment {{
            margin: 12px 0;
            padding: 10px 15px;
            background: #2a2a2a;
            border-radius: 6px;
            border-left: 3px solid #3a3a3a;
        }}
        .segment:hover {{
            border-left-color: #bb86fc;
        }}
        .timestamp {{
            color: #888;
            font-size: 0.8em;
            margin-right: 12px;
            font-family: monospace;
        }}
        /* Controls panel */
        .controls {{
            position: sticky;
            top: 0;
            z-index: 100;
            background: #2a2a2a;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: center;
        }}
        .controls-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: center;
            width: 100%;
        }}
        .controls label {{
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            font-size: 0.9em;
        }}
        .controls input[type="checkbox"] {{
            width: 18px;
            height: 18px;
            cursor: pointer;
        }}
        .controls input[type="text"] {{
            background: #1a1a1a;
            border: 1px solid #3a3a3a;
            border-radius: 6px;
            padding: 8px 12px;
            color: #e0e0e0;
            font-size: 0.9em;
            width: 200px;
        }}
        .controls input[type="text"]:focus {{
            outline: none;
            border-color: #bb86fc;
        }}
        .controls button {{
            background: #bb86fc;
            color: #1a1a1a;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
        }}
        .controls button:hover {{
            background: #9a67ea;
        }}
        .controls button.secondary {{
            background: #3a3a3a;
            color: #e0e0e0;
        }}
        .controls button.secondary:hover {{
            background: #4a4a4a;
        }}
        .font-controls {{
            display: flex;
            gap: 5px;
            align-items: center;
        }}
        .font-controls button {{
            width: 32px;
            height: 32px;
            padding: 0;
            font-size: 1.1em;
        }}
        .font-controls span {{
            font-size: 0.85em;
            color: #888;
            min-width: 40px;
            text-align: center;
        }}
        .search-count {{
            font-size: 0.85em;
            color: #888;
            margin-left: 8px;
        }}
        .search-highlight {{
            background: #ffeb3b !important;
            color: #000 !important;
            padding: 1px 2px;
            border-radius: 2px;
        }}
        .search-highlight-active {{
            background: #ff6d00 !important;
            color: #fff !important;
            padding: 1px 2px;
            border-radius: 2px;
        }}
        .controls-spacer {{
            flex: 1;
        }}
        /* Printer mode - white background */
        body.printer-mode {{
            background: #fff;
            color: #000;
        }}
        body.printer-mode .container {{
            background: #fff;
            box-shadow: none;
        }}
        body.printer-mode .segment {{
            background: #f5f5f5;
            border-left-color: #ccc;
        }}
        body.printer-mode h1 {{
            color: #333;
            border-bottom-color: #333;
        }}
        body.printer-mode .meta {{
            color: #666;
        }}
        body.printer-mode .timestamp {{
            color: #666;
        }}
        body.printer-mode .controls {{
            background: #f0f0f0;
        }}
        body.printer-mode .controls input[type="text"] {{
            background: #fff;
            border-color: #ccc;
            color: #000;
        }}
        body.printer-mode .controls button.secondary {{
            background: #e0e0e0;
            color: #333;
        }}
        /* Hide timestamps */
        body.hide-timestamps .timestamp {{
            display: none;
        }}
        /* No rows - continuous text */
        body.no-rows .segment {{
            display: inline;
            background: none;
            border: none;
            padding: 0;
            margin: 0;
            border-radius: 0;
        }}
        body.no-rows .segment::after {{
            content: ' ';
        }}
        /* Hide word highlighting */
        body.no-highlighting span[style*="color:"] {{
            color: inherit !important;
        }}
        /* Print styles */
        @media print {{
            .no-print {{
                display: none !important;
            }}
            body {{
                background: #fff !important;
                color: #000 !important;
                padding: 0 !important;
            }}
            .container {{
                background: #fff !important;
                box-shadow: none !important;
                padding: 0 !important;
            }}
            .segment {{
                background: #fff !important;
                border: none !important;
                padding: 5px 0 !important;
                margin: 5px 0 !important;
            }}
            h1 {{
                color: #000 !important;
                border-bottom-color: #000 !important;
            }}
            .meta {{
                color: #333 !important;
            }}
            .timestamp {{
                color: #333 !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Transcription</h1>
        <div class="controls no-print">
            <div class="controls-row">
                <label><input type="checkbox" id="showTimestamps" checked> Timestamps</label>
                <label><input type="checkbox" id="useElapsed"> Elapsed Time</label>
                <label><input type="checkbox" id="showRows" checked> Row Separators</label>
                <label><input type="checkbox" id="printerMode"> Printer Mode</label>
                <label><input type="checkbox" id="showHighlighting" checked> Highlighting</label>
                <div class="font-controls">
                    <button class="secondary" onclick="changeFontSize(-1)">A-</button>
                    <span id="fontSizeDisplay">100%</span>
                    <button class="secondary" onclick="changeFontSize(1)">A+</button>
                </div>
                <div class="controls-spacer"></div>
                <button class="secondary" onclick="copyToClipboard()">Copy Text</button>
                <button onclick="window.print()">Print</button>
            </div>
            <div class="controls-row">
                <input type="text" id="searchInput" placeholder="Search text..." onkeyup="searchText(event)">
                <button class="secondary" onclick="doSearch()">Search</button>
                <button class="secondary" onclick="prevMatch()">&uarr; Prev</button>
                <button class="secondary" onclick="nextMatch()">&darr; Next</button>
                <span id="searchCount" class="search-count"></span>
                <button class="secondary" onclick="clearSearch()">Clear</button>
            </div>
        </div>
        <div class="meta">
            <p>Date: {title_date}</p>
            <p>Segments: {len(segments_html)}</p>
        </div>
        {"".join(segments_html)}
    </div>
    <script>
        // Toggle controls
        document.getElementById('showTimestamps').addEventListener('change', function() {{
            document.body.classList.toggle('hide-timestamps', !this.checked);
        }});
        document.getElementById('useElapsed').addEventListener('change', function() {{
            const useElapsed = this.checked;
            document.querySelectorAll('.timestamp').forEach(ts => {{
                const value = useElapsed ? ts.dataset.elapsed : ts.dataset.clock;
                ts.textContent = '[' + value + ']';
            }});
        }});
        document.getElementById('showRows').addEventListener('change', function() {{
            document.body.classList.toggle('no-rows', !this.checked);
        }});
        document.getElementById('printerMode').addEventListener('change', function() {{
            document.body.classList.toggle('printer-mode', this.checked);
        }});
        document.getElementById('showHighlighting').addEventListener('change', function() {{
            document.body.classList.toggle('no-highlighting', !this.checked);
        }});

        // Font size controls
        let currentFontSize = 100;
        function changeFontSize(delta) {{
            currentFontSize = Math.max(50, Math.min(200, currentFontSize + delta * 10));
            document.querySelector('.container').style.fontSize = currentFontSize + '%';
            document.getElementById('fontSizeDisplay').textContent = currentFontSize + '%';
        }}

        // Copy to clipboard
        function copyToClipboard() {{
            const segments = document.querySelectorAll('.segment');
            const showTimestamps = document.getElementById('showTimestamps').checked;
            let text = '';
            segments.forEach(seg => {{
                if (showTimestamps) {{
                    const ts = seg.querySelector('.timestamp');
                    if (ts) text += ts.textContent + ' ';
                }}
                // Get text content without HTML tags
                const clone = seg.cloneNode(true);
                const tsClone = clone.querySelector('.timestamp');
                if (tsClone) tsClone.remove();
                text += clone.textContent.trim() + '\\n';
            }});
            navigator.clipboard.writeText(text.trim()).then(() => {{
                const btn = event.target;
                const originalText = btn.textContent;
                btn.textContent = 'Copied!';
                setTimeout(() => btn.textContent = originalText, 1500);
            }});
        }}

        // Search functionality
        let originalContent = {{}};
        let searchMatches = [];
        let currentMatchIndex = -1;

        function searchText(event) {{
            if (event.key === 'Enter') doSearch();
            else if (event.key === 'ArrowDown') {{ event.preventDefault(); nextMatch(); }}
            else if (event.key === 'ArrowUp') {{ event.preventDefault(); prevMatch(); }}
        }}
        function doSearch() {{
            const query = document.getElementById('searchInput').value.trim().toLowerCase();
            clearSearch();
            if (!query) return;

            const segments = document.querySelectorAll('.segment');
            const regex = new RegExp('(' + query.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&') + ')', 'gi');
            segments.forEach((seg, idx) => {{
                if (!originalContent[idx]) {{
                    originalContent[idx] = seg.innerHTML;
                }}
                if (seg.textContent.toLowerCase().includes(query)) {{
                    seg.innerHTML = seg.innerHTML.replace(/>([^<]+)</g, (match, content) => {{
                        const highlighted = content.replace(regex, '<span class="search-highlight">$1</span>');
                        return '>' + highlighted + '<';
                    }});
                }}
            }});
            searchMatches = Array.from(document.querySelectorAll('.search-highlight'));
            currentMatchIndex = searchMatches.length > 0 ? 0 : -1;
            updateActiveMatch();
        }}
        function updateActiveMatch() {{
            searchMatches.forEach((el, i) => {{
                el.className = i === currentMatchIndex ? 'search-highlight-active' : 'search-highlight';
            }});
            const counter = document.getElementById('searchCount');
            if (searchMatches.length === 0) {{
                counter.textContent = 'No matches';
            }} else {{
                counter.textContent = (currentMatchIndex + 1) + ' / ' + searchMatches.length;
                searchMatches[currentMatchIndex].scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}
        function nextMatch() {{
            if (searchMatches.length === 0) return;
            currentMatchIndex = (currentMatchIndex + 1) % searchMatches.length;
            updateActiveMatch();
        }}
        function prevMatch() {{
            if (searchMatches.length === 0) return;
            currentMatchIndex = (currentMatchIndex - 1 + searchMatches.length) % searchMatches.length;
            updateActiveMatch();
        }}
        function clearSearch() {{
            const segments = document.querySelectorAll('.segment');
            segments.forEach((seg, idx) => {{
                if (originalContent[idx]) {{
                    seg.innerHTML = originalContent[idx];
                }}
            }});
            originalContent = {{}};
            searchMatches = [];
            currentMatchIndex = -1;
            document.getElementById('searchCount').textContent = '';
        }}
    </script>
</body>
</html>"""

        # Write HTML file alongside the database
        html_path = db_path.replace(".db", ".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"[HTML] Created: {html_path} ({len(segments_html)} entries)")
        return html_path
    except Exception as e:
        print(f"[HTML] Error converting database to HTML: {e}")
        import traceback

        traceback.print_exc()
        return None


def validate_file(file):
    """
    Validate uploaded file.

    Args:
        file: Flask file object

    Returns:
        (is_valid, error_message) tuple
    """
    if not file or file.filename == "":
        return False, "No file selected"

    # Check file extension
    ext = os.path.splitext(file.filename)[1].lower().replace(".", "")
    if ext not in SUPPORTED_AUDIO_FORMATS + SUPPORTED_VIDEO_FORMATS:
        return False, f"Unsupported file format: {ext}"

    return True, None


app = Flask(__name__,
            template_folder=os.path.join(BUNDLE_DIR, "templates"),
            static_folder=os.path.join(BUNDLE_DIR, "static"))
app.config["SECRET_KEY"] = os.environ.get("STT_SECRET_KEY") or secrets.token_urlsafe(32)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4GB cap on uploads (media files are large)
app.config["TEMPLATES_AUTO_RELOAD"] = True  # Auto-reload templates when they change
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # Disable caching for static files
socketio = SocketIO(app, async_mode="threading", static_url_path="/static", static_folder=os.path.join(BUNDLE_DIR, "static"), ping_timeout=120, ping_interval=25)

app_logger = logging.getLogger(__name__)  # Use your module name here
socket_io_logger = logging.getLogger("socketio")

# Set log levels as needed
app_logger.setLevel(logging.DEBUG)
socket_io_logger.setLevel(logging.WARNING)

# Disable Flask's built-in logging
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

# Password-based authentication sessions
# Format: {session_token: {"ip": client_ip, "expires": datetime}}
auth_sessions = {}
auth_sessions_lock = threading.Lock()

def generate_session_token():
    """Generate a secure random session token"""
    import secrets
    return secrets.token_urlsafe(32)

def cleanup_expired_sessions():
    """Remove expired sessions from the auth_sessions dict"""
    now = datetime.now()
    with auth_sessions_lock:
        expired = [token for token, data in auth_sessions.items() if data["expires"] < now]
        for token in expired:
            del auth_sessions[token]
        if expired:
            print(f"[AUTH] Cleaned up {len(expired)} expired sessions")


@app.route("/")
def index():
    # Check if URL has any parameters
    if not request.args:
        # No parameters provided: redirect to the active URL-builder profile (if any)
        profiles, active = get_url_builder_profiles()
        params = next((p["params"] for p in profiles if p["name"] == active), None)
        if params:
            from flask import redirect, url_for
            return redirect(url_for('index', **params))

    response = make_response(render_template("index.html"))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response


@app.route("/profile/<name>")
def profile_view(name):
    """Display the page using a named profile's settings, e.g. /profile/lower3rd.
    Case-insensitive; unknown names fall back to the root view."""
    from flask import redirect, url_for
    profiles, _ = get_url_builder_profiles()
    target = (name or "").strip().lower()
    params = next((p["params"] for p in profiles if p["name"].strip().lower() == target), None)
    if params is None:
        return redirect("/")
    return redirect(url_for("index", **params))


@app.route("/favicon.ico")
def favicon():
    # Return a custom response or an empty response
    return "", 204


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors by redirecting to appropriate page"""
    if check_ip_whitelist():
        return redirect("/live-settings")
    else:
        return redirect("/")


def check_ip_whitelist():
    """Check if the client IP is in the whitelist or has a valid password session"""
    import ipaddress

    # First check if password authentication is enabled
    password_auth_config = config.get("web_server", {}).get("password_auth", {})
    password_auth_enabled = password_auth_config.get("enabled", False)

    # Check for valid session token (from cookie or header)
    if password_auth_enabled:
        # Check cookie first
        session_token = request.cookies.get("auth_session")

        # Fallback to Authorization header
        if not session_token:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                session_token = auth_header[7:]  # Remove "Bearer " prefix

        # Validate session token
        if session_token:
            cleanup_expired_sessions()  # Clean up expired sessions
            with auth_sessions_lock:
                session_data = auth_sessions.get(session_token)
                if session_data:
                    # Check if session is still valid
                    if session_data["expires"] > datetime.now():
                        # Check if IP matches (prevent session hijacking)
                        if session_data["ip"] == request.remote_addr:
                            return True
                        else:
                            print(f"[AUTH WARNING] Session token used from different IP: {request.remote_addr} != {session_data['ip']}")
                    else:
                        # Session expired, remove it
                        del auth_sessions[session_token]

    whitelist = config.get("web_server", {}).get("settings_ip_whitelist", [])

    # If whitelist is empty, allow all
    if not whitelist:
        return True

    client_ip = request.remote_addr

    # Always allow localhost variations
    localhost_ips = ["127.0.0.1", "::1", "localhost"]
    if client_ip in localhost_ips:
        return True

    # Check whitelist (supports both exact IPs and CIDR ranges)
    try:
        client_ip_obj = ipaddress.ip_address(client_ip)

        for entry in whitelist:
            entry = entry.strip()
            if not entry or entry.startswith("#"):  # Skip comments
                continue

            try:
                # Check if entry is a CIDR range (contains /)
                if "/" in entry:
                    network = ipaddress.ip_network(entry, strict=False)
                    if client_ip_obj in network:
                        return True
                else:
                    # Exact IP match
                    if client_ip == entry:
                        return True
            except ValueError:
                # Invalid entry, skip it
                print(f"[WARNING] Invalid IP whitelist entry: {entry}")
                continue

        return False
    except ValueError:
        # Invalid client IP format
        print(f"[WARNING] Invalid client IP format: {client_ip}")
        return False


@app.route("/api/auth/login", methods=["POST"])
def password_login():
    """Authenticate with password and create a temporary session"""
    try:
        password_auth_config = config.get("web_server", {}).get("password_auth", {})

        # Check if password auth is enabled
        if not password_auth_config.get("enabled", False):
            return jsonify({"success": False, "error": "Password authentication is disabled"}), 403

        # Get password from request
        data = request.get_json()
        provided_password = data.get("password", "")

        if not provided_password:
            return jsonify({"success": False, "error": "Password required"}), 400

        # Get configured password
        configured_password = password_auth_config.get("password", "")

        # If no password configured, reject
        if not configured_password:
            return jsonify({"success": False, "error": "No password configured on server"}), 500

        # Verify password
        if provided_password != configured_password:
            print(f"[AUTH] Failed login attempt from {request.remote_addr}")
            return jsonify({"success": False, "error": "Invalid password"}), 401

        # Password correct - create session
        session_token = generate_session_token()
        timeout_minutes = password_auth_config.get("session_timeout_minutes", 60)
        expires = datetime.now() + timedelta(minutes=timeout_minutes)

        with auth_sessions_lock:
            auth_sessions[session_token] = {
                "ip": request.remote_addr,
                "expires": expires,
                "created": datetime.now()
            }

        print(f"[AUTH] Successful login from {request.remote_addr}, session expires in {timeout_minutes} minutes")

        # Return session token
        response = jsonify({
            "success": True,
            "session_token": session_token,
            "expires": expires.isoformat(),
            "timeout_minutes": timeout_minutes
        })

        # Set cookie for browser-based access
        response.set_cookie(
            "auth_session",
            session_token,
            max_age=timeout_minutes * 60,  # in seconds
            httponly=True,  # Prevent JavaScript access
            samesite="Strict"  # CSRF protection
        )

        return response

    except Exception as e:
        print(f"[AUTH ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/auth/logout", methods=["POST"])
def password_logout():
    """Logout and invalidate the current session"""
    try:
        # Get session token
        session_token = request.cookies.get("auth_session")
        if not session_token:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                session_token = auth_header[7:]

        if session_token:
            with auth_sessions_lock:
                if session_token in auth_sessions:
                    del auth_sessions[session_token]
                    print(f"[AUTH] Logged out session from {request.remote_addr}")

        response = jsonify({"success": True, "message": "Logged out successfully"})

        # Clear cookie
        response.set_cookie("auth_session", "", max_age=0)

        return response

    except Exception as e:
        print(f"[AUTH ERROR] {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/auth/status", methods=["GET"])
def auth_status():
    """Check if the current session is authenticated"""
    try:
        is_authenticated = check_ip_whitelist()

        # Get session info if authenticated via password
        session_info = None
        if is_authenticated:
            session_token = request.cookies.get("auth_session")
            if not session_token:
                auth_header = request.headers.get("Authorization")
                if auth_header and auth_header.startswith("Bearer "):
                    session_token = auth_header[7:]

            if session_token:
                with auth_sessions_lock:
                    session_data = auth_sessions.get(session_token)
                    if session_data:
                        session_info = {
                            "authenticated_via": "password",
                            "expires": session_data["expires"].isoformat(),
                            "created": session_data["created"].isoformat()
                        }

        if not session_info and is_authenticated:
            session_info = {
                "authenticated_via": "ip_whitelist"
            }

        # Determine redirect URL based on authentication
        redirect_url = "/live-settings" if is_authenticated else "/url-builder"

        return jsonify({
            "success": True,
            "authenticated": is_authenticated,
            "session": session_info,
            "ip": request.remote_addr,
            "redirect_url": redirect_url
        })

    except Exception as e:
        print(f"[AUTH ERROR] {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/live-settings")
def settings_page():
    """Render the live settings page"""
    if not check_ip_whitelist():
        return render_template("auth-required.html"), 403

    return render_template("live-settings.html")


@app.route("/corrections")
def corrections_page():
    """Render the corrections page for editing transcriptions in real-time"""
    if not check_ip_whitelist():
        return render_template("auth-required.html"), 403

    return render_template("corrections.html")


@app.route("/url-builder")
def url_builder_page():
    """Render the URL builder page"""
    return render_template("url-builder.html")


@app.route("/server-settings")
def server_settings_page():
    """Render the server settings page"""
    if not check_ip_whitelist():
        return render_template("auth-required.html"), 403
    return render_template("server-settings.html")


@app.route("/translation")
def translation_settings_page():
    """Render the live translation settings page"""
    if not check_ip_whitelist():
        return render_template("auth-required.html"), 403
    return render_template("translation.html")


@app.route("/word-highlighting")
def word_highlighting_page():
    """Render the word highlighting page"""
    if not check_ip_whitelist():
        return render_template("auth-required.html"), 403
    return render_template("word-highlighting.html")


@app.route("/api/config", methods=["GET"])
def get_config():
    """API endpoint to get current configuration"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        return jsonify({"success": True, "config": config})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/config", methods=["POST"])
def update_config():
    """API endpoint to update configuration"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    try:
        new_config = request.get_json()

        if not new_config:
            return jsonify(
                {"success": False, "error": "No configuration data provided"}
            ), 400

        # Deep merge to preserve fields not sent from frontend (like audio.backend)
        def deep_merge(base, updates):
            """Recursively merge updates into base, preserving existing fields"""
            for key, value in updates.items():
                if (
                    key in base
                    and isinstance(base[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
            return base

        # Merge new config into existing config (preserves backend and other fields)
        config = deep_merge(config, new_config)

        # Live transcription requires temperature 0: nonzero output varies between
        # re-transcription passes, so same_output_threshold finalization never
        # triggers and no rows are ever saved.
        live_temp_clamped = False
        _live_decode = config.get("whisper_decoding", {}).get("live_transcription")
        if isinstance(_live_decode, dict) and "temperature" in _live_decode:
            _temp = _live_decode["temperature"]
            if isinstance(_temp, (list, tuple)) or (_temp or 0) != 0:
                _live_decode["temperature"] = 0.0
                live_temp_clamped = True

        # If the audio device selection changed, also persist a stable name-based
        # identifier so the correct card can be re-found after ALSA index reshuffles
        # across reboots (USB vs onboard/GPU HDA enumeration order is not stable).
        try:
            new_mic = new_config.get("audio", {}).get("default_microphone")
            if new_mic and os.path.isfile(new_mic):
                # A "Test Audio File" selection is a file path, not hardware. Clear the
                # stale stable-name so it doesn't keep resolving to a real device.
                config.setdefault("audio", {})["default_microphone_name"] = ""
            elif new_mic:
                from audio_capture import list_audio_devices
                markers = config.get("audio", {}).get("deprioritize_device_markers", [])
                devices = list_audio_devices(deprioritize_markers=markers)
                matched = next((d for d in devices if d.get("name") == new_mic), None)
                if matched and matched.get("card_id"):
                    config.setdefault("audio", {})["default_microphone_name"] = matched["card_id"]
        except Exception as e:
            app_logger.warning(f"Could not derive stable microphone name: {e}")

        # Write to config file
        with _config_file_lock:
            _atomic_write_json(CONFIG_FILE, config)

        # Send config update through queue for hot-reload
        try:
            config_queue.put({"type": "config_update", "config": config})
        except (OSError, ValueError):
            pass  # Queue might be full or process not ready

        # Determine which settings need restart
        needs_restart = False
        restart_reason = []

        # Check for settings that require full restart
        old_config = load_config()
        if old_config.get("whisper", {}).get("model") != config.get("whisper", {}).get(
            "model"
        ):
            needs_restart = True
            restart_reason.append("Whisper model changed")

        if old_config.get("vad", {}).get("enabled") != config.get("vad", {}).get(
            "enabled"
        ):
            needs_restart = True
            restart_reason.append("VAD enabled/disabled")

        message = "Configuration updated successfully!"
        if live_temp_clamped:
            message += " Live temperature forced to 0 — nonzero values prevent segment finalization."
        if needs_restart:
            message += ' Some changes require restarting the transcription process. Use the "Restart Transcription" button.'
        else:
            message += " Changes will be applied automatically within a few seconds."

        return jsonify(
            {
                "success": True,
                "message": message,
                "config": config,
                "needs_restart": needs_restart,
                "restart_reason": restart_reason,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/config/reset", methods=["POST"])
def reset_config():
    """API endpoint to reset configuration to defaults with optional backup"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    try:
        # Check if backup is requested
        request_data = request.get_json() or {}
        create_backup = request_data.get("create_backup", False)
        backup_path = None

        # Create backup if requested
        if create_backup and os.path.exists(CONFIG_FILE):
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{CONFIG_FILE}.backup.{timestamp}"

            try:
                import shutil
                shutil.copy2(CONFIG_FILE, backup_path)
                print(f"[OK] Config backup created: {backup_path}")
            except Exception as backup_error:
                print(f"[WARNING] Failed to create backup: {backup_error}")
                # Continue with reset even if backup fails
                backup_path = None

        # Reset to defaults: reseed config.json from the canonical template, then
        # reload it (same path as first-run init).
        if not _restore_config_from_template("reset config to defaults"):
            return jsonify({"success": False, "error": "Default config template is missing; cannot reset."}), 500
        config = load_config()

        # Send config update through queue
        try:
            config_queue.put({"type": "config_update", "config": config})
        except (OSError, ValueError):
            pass

        response_data = {
            "success": True,
            "message": 'Configuration reset to defaults. Use "Restart Transcription" button to apply changes.',
            "config": config,
            "needs_restart": True,
        }

        if backup_path:
            response_data["backup_path"] = backup_path

        return jsonify(response_data)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def analyze_calibration_data(data):
    """Analyze calibration data and suggest optimal settings"""
    import statistics

    results = {
        "suggestions": {},
        "analysis": {},
        "confidence": "medium"
    }

    # 1. ANALYZE ENERGY THRESHOLD
    # With two-step calibration, we ALWAYS have noise data from step 1
    if data["noise_samples"]:
        noise_energies = [s["energy"] for s in data["noise_samples"]]
        speech_energies = [s["energy"] for s in data["speech_samples"]] if data["speech_samples"] else []

        avg_noise = statistics.mean(noise_energies)
        max_noise = max(noise_energies)
        # Use 75th percentile to ignore outlier noise spikes
        noise_75th = np.percentile(noise_energies, 75) if len(noise_energies) > 10 else max_noise

        if speech_energies:
            # Have both noise and speech - optimal case (two-step calibration success)
            min_speech = min(speech_energies)
            avg_speech = statistics.mean(speech_energies)

            # Set threshold between noise ceiling and speech floor
            # Use 75th percentile of noise to ignore spikes
            suggested_threshold = int((noise_75th + min_speech) / 2)

            # Ensure it's above noise but below speech
            suggested_threshold = max(int(noise_75th * 1.2), suggested_threshold)
            suggested_threshold = min(int(min_speech * 0.9), suggested_threshold)

            results["analysis"]["speech_level"] = {
                "minimum": round(min_speech, 1),
                "average": round(avg_speech, 1),
                "samples": len(speech_energies)
            }
            results["analysis"]["threshold_confidence"] = "high"
        else:
            # Only have noise from step 1 (user didn't speak in step 2)
            # Use conservative threshold based on average noise
            suggested_threshold = int(avg_noise * 2.0)
            suggested_threshold = max(300, suggested_threshold)

            # Add warning about missing speech samples
            if "warnings" not in results:
                results["warnings"] = []
            results["warnings"].append(
                "No speech detected in Step 2. "
                "Using conservative threshold based on noise floor only."
            )

            # Mark threshold confidence as low
            results["analysis"]["threshold_confidence"] = "low"

        # Clamp to reasonable range
        suggested_threshold = max(100, min(10000, suggested_threshold))

        results["suggestions"]["energy_threshold"] = suggested_threshold
        results["analysis"]["noise_level"] = {
            "average": round(avg_noise, 1),
            "maximum": round(max_noise, 1),
            "percentile_75": round(noise_75th, 1),
            "environment": "quiet" if avg_noise < 500 else "normal" if avg_noise < 2000 else "noisy"
        }

    # 2. ANALYZE PHRASE TIMEOUT
    if data["silence_durations"]:
        silence_durations = data["silence_durations"]

        # DEBUG: Log raw silence durations data
        print(f"[CALIBRATION-ANALYSIS] Analyzing {len(silence_durations)} silence durations: {silence_durations[:20] if len(silence_durations) > 20 else silence_durations}", flush=True)

        # Find typical pause length (median of shorter pauses)
        short_pauses = [d for d in silence_durations if d < 5.0]  # Ignore very long pauses

        print(f"[CALIBRATION-ANALYSIS] After filtering (< 5.0s): {len(short_pauses)} short pauses", flush=True)

        if short_pauses:
            median_pause = statistics.median(short_pauses)

            print(f"[CALIBRATION-ANALYSIS] Median pause: {median_pause:.2f}s", flush=True)

            # Suggest phrase_timeout slightly above median pause
            # This splits on longer pauses but not normal speech pauses
            suggested_timeout = round(median_pause * 1.3, 1)
            suggested_timeout = max(1.0, min(5.0, suggested_timeout))  # Clamp to 1-5 seconds

            print(f"[CALIBRATION-ANALYSIS] Suggested phrase_timeout: {suggested_timeout}s", flush=True)

            results["suggestions"]["phrase_timeout"] = suggested_timeout
            results["analysis"]["pause_pattern"] = {
                "median_pause": round(median_pause, 2),
                "min_pause": round(min(short_pauses), 2),
                "max_pause": round(max(short_pauses), 2),
                "total_pauses": len(silence_durations)
            }

    # 3. ANALYZE VAD SETTINGS
    if data["vad_probabilities"]:
        vad_probs = data["vad_probabilities"]
        avg_vad = statistics.mean(vad_probs)
        min_vad = min(vad_probs)

        # If average VAD confidence is high, can use stricter threshold
        if avg_vad > 0.8:
            suggested_vad_threshold = 0.6  # Stricter
        elif avg_vad > 0.6:
            suggested_vad_threshold = 0.5  # Normal
        else:
            suggested_vad_threshold = 0.4  # More lenient

        results["suggestions"]["vad_threshold"] = suggested_vad_threshold
        results["suggestions"]["vad_enabled"] = True
        results["analysis"]["vad_performance"] = {
            "average_confidence": round(avg_vad, 2),
            "minimum_confidence": round(min_vad, 2),
            "recommendation": "strict" if avg_vad > 0.8 else "normal" if avg_vad > 0.6 else "lenient"
        }

    # 4. DETERMINE CONFIDENCE LEVEL
    speech_samples = len(data.get("speech_samples", []))
    noise_samples = len(data.get("noise_samples", []))

    if speech_samples >= 10 and noise_samples >= 10:
        results["confidence"] = "high"
    elif speech_samples >= 5 or noise_samples >= 5:
        results["confidence"] = "medium"
    else:
        results["confidence"] = "low"

    return results


@app.route("/api/calibration/start", methods=["POST"])
def start_calibration():
    """Start calibration mode to analyze environment and suggest optimal settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global calibration_state, transcription_state

    try:
        # Get duration and skip_step1 from request
        request_data = request.get_json() or {}
        duration = request_data.get("duration", 15)
        skip_step1 = request_data.get("skip_step1", False)

        # Auto-start transcription if not running (calibration needs audio data)
        if not transcription_state["running"]:
            print("[CALIBRATION] Auto-starting transcription for calibration...", flush=True)
            control_queue.put({"command": "start"})
            transcription_state["status"] = "starting"
            transcription_state["message"] = "Starting for calibration..."

            # Wait for model to be fully loaded (status changes to "running")
            max_wait = 60  # Maximum 60 seconds for model loading
            waited = 0
            while transcription_state["status"] != "running" and waited < max_wait:
                time.sleep(0.5)
                waited += 0.5
                if transcription_state["status"] == "error":
                    return jsonify({"success": False, "error": transcription_state.get("error", "Model loading failed")}), 500

            if transcription_state["status"] != "running":
                return jsonify({"success": False, "error": "Timeout waiting for model to load"}), 500

            print(f"[CALIBRATION] Model ready after {waited}s", flush=True)

        # Clear shared calibration data
        calibration_data_shared["speech_samples"][:] = []
        calibration_data_shared["noise_samples"][:] = []
        calibration_data_shared["silence_durations"][:] = []
        calibration_data_shared["energy_levels"][:] = []
        calibration_data_shared["vad_probabilities"][:] = []

        # Clear step 1 data
        calibration_step1_data["noise_energies"][:] = []
        calibration_step1_data["avg_noise"] = 0.0
        calibration_step1_data["max_noise"] = 0.0

        # Initialize shared calibration state for two-step process
        calibration_state["active"] = True
        calibration_state["start_time"] = time.time()
        calibration_state["duration"] = duration
        calibration_state["speech_samples"] = 0
        calibration_state["noise_samples"] = 0
        calibration_state["silence_samples"] = 0

        if skip_step1:
            # Skip directly to Step 2
            calibration_state["step"] = 2
            calibration_state["step1_complete"] = True
            # Use current energy threshold from config as noise baseline
            # This preserves user's existing tuning
            current_threshold = config.get("audio", {}).get("energy_threshold", 300)
            # Use inverse of suggestion formula (suggested = avg_noise * 2.0)
            # So avg_noise = current_threshold / 2.0 to maintain round-trip consistency
            calibration_step1_data["avg_noise"] = float(current_threshold / 2.0)
            calibration_step1_data["max_noise"] = float(current_threshold)
            print(f"[CALIBRATION] Skipping Step 1, starting at Step 2 (speech only) - using current threshold {current_threshold} as baseline", flush=True)
        else:
            # Normal two-step process
            calibration_state["step"] = 1
            calibration_state["step1_complete"] = False
            print(f"[CALIBRATION] Started two-step calibration - {duration}s per step ({duration * 2}s total)", flush=True)

        # Send calibration command to transcription process via control queue
        control_queue.put({
            "command": "start_calibration",
            "duration": duration
        })

        return jsonify({
            "success": True,
            "message": f"Two-step calibration started ({duration}s per step)",
            "duration": duration,
            "total_duration": duration * 2
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/calibration/status", methods=["GET"])
def calibration_status():
    """Get current calibration progress (two-step process)"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global calibration_state

    if not calibration_state["active"]:
        return jsonify({"calibrating": False, "active": False})

    elapsed = time.time() - calibration_state["start_time"]
    duration = calibration_state.get("duration", 15)
    current_step = calibration_state.get("step", 1)
    progress = min(100, int((elapsed / duration) * 100))

    return jsonify({
        "calibrating": calibration_state["active"],
        "active": calibration_state["active"],
        "step": current_step,
        "step1_complete": calibration_state.get("step1_complete", False),
        "progress": progress,
        "elapsed": round(elapsed, 1),
        "duration": duration,
        "samples_collected": {
            "noise": calibration_state["noise_samples"],
            "speech": calibration_state["speech_samples"],
            "silence": calibration_state["silence_samples"],
        }
    })


@app.route("/api/calibration/continue", methods=["POST"])
def continue_calibration():
    """Continue calibration from Step 1 to Step 2"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global calibration_state

    if not calibration_state.get("step1_complete", False):
        return jsonify({"success": False, "error": "Step 1 not complete yet"}), 400

    if calibration_state.get("step", 1) != 1:
        return jsonify({"success": False, "error": "Already on Step 2 or not calibrating"}), 400

    # Transition to step 2
    calibration_state["step"] = 2
    calibration_state["start_time"] = time.time()
    calibration_state["reset_timer"] = True  # Signal transcription process to reset local timer

    print("[CALIBRATION] Transitioning to Step 2 - user clicked Start Step 2", flush=True)

    return jsonify({"success": True, "message": "Starting Step 2"})


@app.route("/api/calibration/results", methods=["GET"])
def calibration_results():
    """Get calibration results and suggested settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global calibration_state, calibration_data_shared

    if calibration_state["active"]:
        return jsonify({"success": False, "error": "Calibration still in progress"}), 400

    # Convert shared data to regular dict for analysis
    calibration_data = {
        "speech_samples": list(calibration_data_shared["speech_samples"]),
        "noise_samples": list(calibration_data_shared["noise_samples"]),
        "silence_durations": list(calibration_data_shared["silence_durations"]),
        "energy_levels": list(calibration_data_shared["energy_levels"]),
        "vad_probabilities": list(calibration_data_shared["vad_probabilities"]),
    }

    if not calibration_data.get("energy_levels"):
        return jsonify({"success": False, "error": "No calibration data available"}), 400

    try:
        # Analyze collected data
        results = analyze_calibration_data(calibration_data)

        return jsonify({
            "success": True,
            "current_settings": {
                "energy_threshold": config.get("audio", {}).get("energy_threshold", 3500),
                "phrase_timeout": config.get("audio", {}).get("phrase_timeout", 2),
                "active_window_duration": config.get("audio", {}).get("active_window_duration", 5.0),
                "confirmation_delay": config.get("audio", {}).get("confirmation_delay", 1.5),
                "stride_length": config.get("audio", {}).get("stride_length", 2.0),
                "vad_enabled": config.get("vad", {}).get("enabled", True),
                "vad_threshold": config.get("vad", {}).get("threshold", 0.5),
            },
            "suggested_settings": results["suggestions"],
            "analysis": results["analysis"],
            "confidence": results["confidence"],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/word-highlighting/words", methods=["GET"])
def get_highlighted_words():
    """API endpoint to get list of highlighted words (accessible to all for index page)"""
    # No IP whitelist check - this needs to be accessible to all users viewing the index page
    data = load_word_highlighting()
    return jsonify({"success": True, "enabled": data.get("enabled", True), "words": data.get("words", []), "disabled_colors": data.get("disabled_colors", [])})


@app.route("/api/word-highlighting/words", methods=["POST"])
def add_highlighted_word():
    """Add a new highlighted word
    Example: POST /api/word-highlighting/words {"word": "hello", "color": "#ff0000", "case_sensitive": false}"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    req_data = request.json
    word = req_data.get("word", "").strip()
    color = req_data.get("color", "#ffff00")  # Default yellow
    case_sensitive = req_data.get("case_sensitive", False)
    is_regex = req_data.get("is_regex", False)

    if not word:
        return jsonify({"success": False, "error": "Word is required"}), 400

    # Load from separate file
    wh_data = load_word_highlighting()
    words = wh_data.get("words", [])

    # Check if word already exists
    for existing_word in words:
        if existing_word.get("word") == word:
            return jsonify({"success": False, "error": "Word already exists"}), 400

    # Add new word
    new_word = {"word": word, "color": color, "case_sensitive": case_sensitive, "is_regex": is_regex}
    words.append(new_word)
    wh_data["words"] = words

    # Save to separate file
    save_word_highlighting(wh_data)

    # Broadcast update to all connected clients
    socketio.emit("word_highlighting_update", {
        "enabled": wh_data.get("enabled", True),
        "words": wh_data.get("words", []),
        "disabled_colors": wh_data.get("disabled_colors", [])
    })

    return jsonify({"success": True, "word": new_word})


@app.route("/api/word-highlighting/words/<int:index>", methods=["DELETE"])
def delete_highlighted_word(index):
    """Delete a highlighted word by index
    Example: DELETE /api/word-highlighting/words/0"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    wh_data = load_word_highlighting()
    words = wh_data.get("words", [])

    if index < 0 or index >= len(words):
        return jsonify({"success": False, "error": "Invalid index"}), 400

    deleted_word = words.pop(index)
    wh_data["words"] = words

    # Save to separate file
    save_word_highlighting(wh_data)

    # Broadcast update to all connected clients
    socketio.emit("word_highlighting_update", {
        "enabled": wh_data.get("enabled", True),
        "words": wh_data.get("words", []),
        "disabled_colors": wh_data.get("disabled_colors", [])
    })

    return jsonify({"success": True, "deleted": deleted_word})


@app.route("/api/word-highlighting/words/<int:index>", methods=["PUT"])
def update_highlighted_word(index):
    """Update a highlighted word by index
    Example: PUT /api/word-highlighting/words/0 {"word": "test", "color": "#ff0000"}"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    req_data = request.json
    wh_data = load_word_highlighting()
    words = wh_data.get("words", [])

    if index < 0 or index >= len(words):
        return jsonify({"success": False, "error": "Invalid index"}), 400

    # Update word properties
    if "word" in req_data:
        words[index]["word"] = req_data["word"].strip()
    if "color" in req_data:
        words[index]["color"] = req_data["color"]
    if "case_sensitive" in req_data:
        words[index]["case_sensitive"] = req_data["case_sensitive"]
    if "is_regex" in req_data:
        words[index]["is_regex"] = req_data["is_regex"]

    wh_data["words"] = words

    # Save to separate file
    save_word_highlighting(wh_data)

    # Broadcast update to all connected clients
    socketio.emit("word_highlighting_update", {
        "enabled": wh_data.get("enabled", True),
        "words": wh_data.get("words", []),
        "disabled_colors": wh_data.get("disabled_colors", [])
    })

    return jsonify({"success": True, "word": words[index]})


@app.route("/api/word-highlighting/toggle", methods=["POST"])
def toggle_word_highlighting():
    """API endpoint to toggle word highlighting on/off"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    wh_data = load_word_highlighting()
    current = wh_data.get("enabled", True)
    wh_data["enabled"] = not current

    # Save to separate file
    save_word_highlighting(wh_data)

    # Broadcast update to all connected clients
    socketio.emit("word_highlighting_update", {
        "enabled": wh_data.get("enabled", True),
        "words": wh_data.get("words", []),
        "disabled_colors": wh_data.get("disabled_colors", [])
    })

    return jsonify({"success": True, "enabled": wh_data["enabled"]})


@app.route("/api/word-highlighting/toggle-color", methods=["POST"])
def toggle_color_group():
    """Toggle a color group on/off (disable highlighting without deleting)
    Example: POST /api/word-highlighting/toggle-color {"color": "#ff0000"}"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    req_data = request.json
    color = req_data.get("color", "").strip().lower()

    if not color:
        return jsonify({"success": False, "error": "Color is required"}), 400

    wh_data = load_word_highlighting()
    disabled_colors = wh_data.get("disabled_colors", [])

    # Toggle the color in disabled list
    if color in disabled_colors:
        disabled_colors.remove(color)
        is_disabled = False
    else:
        disabled_colors.append(color)
        is_disabled = True

    wh_data["disabled_colors"] = disabled_colors
    save_word_highlighting(wh_data)

    # Broadcast update to all connected clients
    socketio.emit("word_highlighting_update", {
        "enabled": wh_data.get("enabled", True),
        "words": wh_data.get("words", []),
        "disabled_colors": disabled_colors
    })

    return jsonify({"success": True, "color": color, "disabled": is_disabled})


# Hallucination Filter API Endpoints


@app.route("/api/hallucination-filter/toggle", methods=["POST"])
def toggle_hallucination_filter():
    """API endpoint to toggle hallucination filter on/off"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    try:
        if "hallucination_filter" not in config:
            config["hallucination_filter"] = {"enabled": True, "phrases": []}

        current = config["hallucination_filter"].get("enabled", True)
        config["hallucination_filter"]["enabled"] = not current
        save_config(config)

        return jsonify({"success": True, "enabled": config["hallucination_filter"]["enabled"]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/hallucination-filter/cjk-toggle", methods=["POST"])
def toggle_cjk_filter():
    """API endpoint to toggle CJK character hallucination filter on/off"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    try:
        if "hallucination_filter" not in config:
            config["hallucination_filter"] = {"enabled": True, "phrases": [], "cjk_filter_enabled": True}

        current = config["hallucination_filter"].get("cjk_filter_enabled", True)
        config["hallucination_filter"]["cjk_filter_enabled"] = not current
        save_config(config)

        return jsonify({"success": True, "cjk_filter_enabled": config["hallucination_filter"]["cjk_filter_enabled"]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/hallucination-filter/phrases", methods=["POST"])
def update_hallucination_phrases():
    """API endpoint to update hallucination filter phrases"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    try:
        data = request.get_json()
        phrases = data.get("phrases", [])

        if "hallucination_filter" not in config:
            config["hallucination_filter"] = {"enabled": True, "phrases": []}

        config["hallucination_filter"]["phrases"] = phrases
        save_config(config)

        return jsonify({"success": True, "phrases": phrases})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/profanity-filter/toggle", methods=["POST"])
def toggle_profanity_filter():
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    global config
    try:
        if "profanity_filter" not in config:
            config["profanity_filter"] = {"enabled": False, "words": []}
        current = config["profanity_filter"].get("enabled", False)
        config["profanity_filter"]["enabled"] = not current
        save_config(config)
        return jsonify({"success": True, "enabled": config["profanity_filter"]["enabled"]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/profanity-filter/words", methods=["POST"])
def update_profanity_words():
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    global config
    try:
        data = request.get_json()
        words = data.get("words", [])
        if "profanity_filter" not in config:
            config["profanity_filter"] = {"enabled": False, "words": []}
        config["profanity_filter"]["words"] = words
        save_config(config)
        return jsonify({"success": True, "words": words})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def get_url_builder_profiles():
    """Return (profiles, active) for the URL builder, migrating the legacy
    single-blob `url_builder_defaults` into a named-profile list on first access.

    profiles: list of {"name": str, "params": {url param dict}}
    active:   name of the profile that drives the root "/" redirect ("" = none)
    """
    profiles = config.get("url_builder_profiles")
    if profiles is None:
        legacy = config.get("url_builder_defaults")
        if legacy:
            profiles = [{"name": "Default", "params": legacy}]
            config["url_builder_active"] = "Default"
        else:
            profiles = []
            config.setdefault("url_builder_active", "")
        config["url_builder_profiles"] = profiles
    return profiles, config.get("url_builder_active", "")


@app.route("/api/url-builder/profiles", methods=["GET"])
def get_url_builder_profiles_endpoint():
    """List all saved URL builder profiles and which one is active (root)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    profiles, active = get_url_builder_profiles()
    return jsonify({"success": True, "profiles": profiles, "active": active})


@app.route("/api/url-builder/profiles", methods=["POST"])
def save_url_builder_profile():
    """Create or update (upsert by name) a URL builder profile."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    try:
        data = request.get_json() or {}
        name = (data.get("name") or "").strip()
        params = data.get("params") or {}
        if not name:
            return jsonify({"success": False, "error": "Profile name is required"}), 400
        if not all(c.isalnum() or c in "-_" for c in name):
            return jsonify({"success": False, "error": "Profile names can only use letters, numbers, - and _ (no spaces) for clean /profile URLs"}), 400

        profiles, _ = get_url_builder_profiles()
        for p in profiles:
            if p["name"] == name:
                p["params"] = params
                break
        else:
            profiles.append({"name": name, "params": params})

        config["url_builder_profiles"] = profiles
        save_config(config)
        return jsonify({"success": True, "profiles": profiles, "active": config.get("url_builder_active", "")})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/url-builder/profiles/activate", methods=["POST"])
def activate_url_builder_profile():
    """Set which profile drives the root "/" redirect."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    try:
        data = request.get_json() or {}
        name = (data.get("name") or "").strip()
        profiles, _ = get_url_builder_profiles()
        if not any(p["name"] == name for p in profiles):
            return jsonify({"success": False, "error": "Profile not found"}), 404

        config["url_builder_active"] = name
        save_config(config)
        return jsonify({"success": True, "active": name})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/url-builder/profiles/<name>", methods=["DELETE"])
def delete_url_builder_profile(name):
    """Delete a profile by name; clears active if it was the root profile."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    try:
        profiles, active = get_url_builder_profiles()
        new_profiles = [p for p in profiles if p["name"] != name]
        if len(new_profiles) == len(profiles):
            return jsonify({"success": False, "error": "Profile not found"}), 404

        config["url_builder_profiles"] = new_profiles
        if active == name:
            config["url_builder_active"] = ""
        save_config(config)
        return jsonify({"success": True, "profiles": new_profiles, "active": config.get("url_builder_active", "")})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Backward-compat shims: map the old single-blob "defaults" API onto the active profile.
@app.route("/api/url-builder/defaults", methods=["POST"])
def save_url_builder_defaults():
    """Legacy endpoint: upsert a 'Default' profile from the posted params and activate it."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    try:
        data = request.get_json() or {}
        profiles, _ = get_url_builder_profiles()
        for p in profiles:
            if p["name"] == "Default":
                p["params"] = data
                break
        else:
            profiles.append({"name": "Default", "params": data})
        config["url_builder_profiles"] = profiles
        config["url_builder_active"] = "Default"
        save_config(config)
        return jsonify({"success": True, "message": "Default settings saved"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/url-builder/defaults", methods=["GET"])
def get_url_builder_defaults():
    """Legacy endpoint: return the active profile's params as the saved defaults."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    profiles, active = get_url_builder_profiles()
    params = next((p["params"] for p in profiles if p["name"] == active), {})
    return jsonify({"success": True, "defaults": params})


# File Transcription Settings Endpoints


@app.route("/api/file-transcription/settings", methods=["GET"])
def get_file_transcription_settings():
    """API endpoint to get file transcription settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    ft_config = config.get("file_transcription", {})

    return jsonify(
        {
            "success": True,
            "settings": {
                "use_gpu": ft_config.get("use_gpu", True),
                "language": ft_config.get("language", "auto"),
                "translate_enabled": ft_config.get("translate_enabled", False),
                "translate_to": ft_config.get("translate_to", "en"),
                "translation_model": ft_config.get("translation_model", "facebook/nllb-200-distilled-600M"),
                "translation_method": config.get("live_translation", {}).get("translation_method", "nllb"),
                "model": {
                    "type": ft_config.get("model", {}).get("type", "whisper"),
                    "whisper": {
                        "model": ft_config.get("model", {})
                        .get("whisper", {})
                        .get("model", "base"),
                    },
                    "huggingface": {
                        "model_id": ft_config.get("model", {})
                        .get("huggingface", {})
                        .get("model_id", "openai/whisper-base"),
                        "use_flash_attention": ft_config.get("model", {})
                        .get("huggingface", {})
                        .get("use_flash_attention", False),
                    },
                    "custom": {
                        "model_path": ft_config.get("model", {})
                        .get("custom", {})
                        .get("model_path", ""),
                        "model_type": ft_config.get("model", {})
                        .get("custom", {})
                        .get("model_type", "whisper"),
                    },
                },
            },
        }
    )


@app.route("/api/file-transcription/settings", methods=["POST"])
def save_file_transcription_settings():
    """API endpoint to save file transcription settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    data = request.json

    if "file_transcription" not in config:
        config["file_transcription"] = {}

    # Update settings
    if "use_gpu" in data:
        config["file_transcription"]["use_gpu"] = data["use_gpu"]
    if "language" in data:
        config["file_transcription"]["language"] = data["language"]
    if "translate_enabled" in data:
        config["file_transcription"]["translate_enabled"] = data["translate_enabled"]
    if "translate_to" in data:
        config["file_transcription"]["translate_to"] = data["translate_to"]
    if "translation_model" in data:
        config["file_transcription"]["translation_model"] = data["translation_model"]

    # Update model settings
    if "model" in data:
        if "model" not in config["file_transcription"]:
            config["file_transcription"]["model"] = {}

        if "type" in data["model"]:
            config["file_transcription"]["model"]["type"] = data["model"]["type"]

        if "whisper" in data["model"]:
            if "whisper" not in config["file_transcription"]["model"]:
                config["file_transcription"]["model"]["whisper"] = {}
            config["file_transcription"]["model"]["whisper"].update(
                data["model"]["whisper"]
            )

        if "huggingface" in data["model"]:
            if "huggingface" not in config["file_transcription"]["model"]:
                config["file_transcription"]["model"]["huggingface"] = {}
            config["file_transcription"]["model"]["huggingface"].update(
                data["model"]["huggingface"]
            )

        if "custom" in data["model"]:
            if "custom" not in config["file_transcription"]["model"]:
                config["file_transcription"]["model"]["custom"] = {}
            config["file_transcription"]["model"]["custom"].update(
                data["model"]["custom"]
            )

    # Save to file
    save_config(config)

    return jsonify(
        {"success": True, "message": "File transcription settings saved successfully"}
    )


# Timezone Settings Endpoints


@app.route("/api/timezone/settings", methods=["GET"])
def get_timezone_settings():
    """API endpoint to get timezone settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    tz_config = config.get("timezone", {"mode": "auto", "value": ""})

    return jsonify(
        {
            "success": True,
            "settings": {
                "mode": tz_config.get("mode", "auto"),
                "value": tz_config.get("value", ""),
            },
        }
    )


@app.route("/api/timezone/settings", methods=["POST"])
def save_timezone_settings():
    """API endpoint to save timezone settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    data = request.json

    if "timezone" not in config:
        config["timezone"] = {}

    if "mode" in data:
        config["timezone"]["mode"] = data["mode"]
    if "value" in data:
        config["timezone"]["value"] = data["value"]

    # Save to file
    save_config(config)

    return jsonify(
        {
            "success": True,
            "message": "Timezone settings saved successfully. Restart application to apply.",
        }
    )


@app.route("/api/server/time", methods=["GET"])
def get_server_time():
    """API endpoint to get server's current time for timezone comparison"""
    now = datetime.now()
    return jsonify({
        "success": True,
        "timestamp": now.timestamp(),
        "formatted": now.strftime("%A, %B %d, %Y at %I:%M:%S %p"),
        "timezone": str(now.astimezone().tzinfo),
        "iso": now.isoformat(),
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "hour": now.hour,
        "minute": now.minute
    })


# Live Translation API Endpoints


@app.route("/api/translation/settings", methods=["GET"])
def get_translation_settings():
    """API endpoint to get live translation settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    trans_config = config.get("live_translation", {
        "enabled": False,
        "target_language": "en",
        "source_language": "auto",
        "translate_in_progress": False,
        "display_mode": "translated_only",
        "translation_model": "facebook/nllb-200-distilled-600M"
    })

    translation_count = config.get("corrections", {}).get("n_best_alternatives", {}).get("translation_count", 3)

    return jsonify({
        "success": True,
        "settings": trans_config,
        "translation_count": translation_count,
        "available_languages": TRANSLATION_LANGUAGES,
        "model_loaded": is_live_translation_model_loaded(),
        "cache_size": get_translation_cache().get_size()
    })


@app.route("/api/translation/settings", methods=["POST"])
def save_translation_settings():
    """API endpoint to save live translation settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    data = request.json

    if "live_translation" not in config:
        config["live_translation"] = {}

    # Track if we need to handle model loading/unloading
    was_enabled = config.get("live_translation", {}).get("enabled", False)
    old_target_lang = config.get("live_translation", {}).get("target_language", "en")
    old_model = config.get("live_translation", {}).get("translation_model", "")
    old_use_gpu = config.get("live_translation", {}).get("use_gpu", True)
    old_method = config.get("live_translation", {}).get("translation_method", "nllb")

    # Update settings
    for key in ["enabled", "target_language", "source_language", "translate_in_progress",
                "display_mode", "translation_model", "use_gpu", "translation_method"]:
        if key in data:
            config["live_translation"][key] = data[key]

    # Clamp to match the UI slider (1-5); larger windows approach NLLB's 1024-token truncation
    if "context_window" in data:
        try:
            config["live_translation"]["context_window"] = max(1, min(5, int(data["context_window"])))
        except (TypeError, ValueError):
            pass

    # Save generation parameters
    if "generation_params" in data:
        gp = data["generation_params"]
        config["live_translation"]["generation_params"] = {
            "num_beams": max(1, min(20, int(gp.get("num_beams", 5)))),
            "length_penalty": max(0.1, min(3.0, float(gp.get("length_penalty", 1.0)))),
            "no_repeat_ngram_size": max(0, min(10, int(gp.get("no_repeat_ngram_size", 0)))),
            "repetition_penalty": max(0.5, min(3.0, float(gp.get("repetition_penalty", 1.0)))),
        }

    # Save remote translation endpoint config
    if "remote" in data:
        config["live_translation"]["remote"] = {
            "enabled": bool(data["remote"].get("enabled", False)),
            "endpoint": str(data["remote"].get("endpoint", "")),
        }

    # Save translation alternatives count to corrections config
    if "translation_count" in data:
        config.setdefault("corrections", {}).setdefault("n_best_alternatives", {})["translation_count"] = int(data["translation_count"])

    save_config(config)

    # Push to config queue for hot-reload (so transcription subprocess picks up translation_method changes)
    if config_queue:
        try:
            config_queue.put({"type": "config_update", "config": config.copy()})
        except (OSError, ValueError):
            pass

    # Handle model loading/unloading based on enabled state
    now_enabled = config["live_translation"].get("enabled", False)
    new_target_lang = config["live_translation"].get("target_language", "en")
    new_model = config["live_translation"].get("translation_model", "")
    new_use_gpu = config["live_translation"].get("use_gpu", True)
    new_method = config["live_translation"].get("translation_method", "nllb")

    model_changed = old_model != new_model or old_use_gpu != new_use_gpu
    method_changed = old_method != new_method
    using_whisper = new_method in ("whisper_translate", "whisper_forced_lang")

    if not now_enabled and was_enabled:
        # Translation just disabled - unload model
        threading.Thread(target=unload_live_translation_model, daemon=True).start()
    elif using_whisper and not (old_method in ("whisper_translate", "whisper_forced_lang")):
        # Switched to Whisper method — unload NLLB model (not needed)
        threading.Thread(target=unload_live_translation_model, daemon=True).start()
    elif now_enabled and not using_whisper and (not was_enabled or model_changed or method_changed):
        # Using NLLB: translation just enabled, model/GPU/method changed - reload model
        # Skip eager loading if this machine serves remote clients (Machine B) —
        # model will be loaded when Machine A starts transcription via /api/translate/preload
        if _trusted_translation_clients:
            if was_enabled and (model_changed or method_changed):
                # Model changed — unload old one, new one loads on next request/preload
                threading.Thread(target=unload_live_translation_model, daemon=True).start()
        else:
            def reload_translation_model():
                if was_enabled:
                    unload_live_translation_model()
                use_gpu = config.get("live_translation", {}).get("use_gpu", config.get("performance", {}).get("use_gpu", True))
                model_id = config["live_translation"].get("translation_model")
                get_live_translation_model(use_gpu, model_id)
            threading.Thread(target=reload_translation_model, daemon=True).start()

    # Clear cache on model or method change (stale tokenizer/model).
    # Don't clear on language change — stale-lang fallback keeps old translations.
    if model_changed or method_changed:
        get_translation_cache().clear()
    elif new_target_lang != old_target_lang:
        # Notify clients to reset display for clean language transition
        socketio.emit("language_switched", {
            "old_language": old_target_lang,
            "new_language": new_target_lang,
            "language_name": TRANSLATION_LANGUAGES.get(new_target_lang, new_target_lang),
        })

    return jsonify({
        "success": True,
        "message": "Translation settings saved. Changes take effect immediately."
    })


@app.route("/api/translation/language", methods=["POST"])
def hot_switch_translation_language():
    """Hot-switch target language without restart - clears cache and re-translates"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    data = request.json
    new_language = data.get("target_language")

    if not new_language:
        return jsonify({"success": False, "error": "target_language required"}), 400

    if new_language not in NLLB_LANG_CODES:
        return jsonify({"success": False, "error": f"Invalid language: {new_language}"}), 400

    old_language = config.get("live_translation", {}).get("target_language", "en")

    if "live_translation" not in config:
        config["live_translation"] = {}

    config["live_translation"]["target_language"] = new_language

    # Auto-switch TTS voice/model to match the new language
    new_tts_voice = None
    tts_section = config["live_translation"].setdefault("tts", {})
    backend = tts_section.get("backend", "edge")

    if old_language != new_language:
        if backend == "edge":
            prefs = tts_section.setdefault("edge_voice_preferences", {})
            current_voice = tts_section.get("edge_voice", "")
            if current_voice:
                prefs[old_language] = current_voice
            new_tts_voice = prefs.get(new_language) or _pick_default_edge_voice(new_language)
            if new_tts_voice:
                tts_section["edge_voice"] = new_tts_voice
                print(f"[TTS] Auto-switched edge voice: {current_voice} -> {new_tts_voice}")

        elif backend == "piper":
            prefs = tts_section.setdefault("piper_model_preferences", {})
            current_model = tts_section.get("piper_model", "")
            if current_model:
                prefs[old_language] = current_model
            new_tts_voice = prefs.get(new_language) or _pick_default_piper_model(new_language)
            if new_tts_voice:
                tts_section["piper_model"] = new_tts_voice
                print(f"[TTS] Auto-switched piper model: {current_model} -> {new_tts_voice}")
                # Reload piper model in background
                def _reload():
                    unload_tts_model()
                    get_tts_model(model_name=new_tts_voice)
                threading.Thread(target=_reload, daemon=True).start()

    save_config(config)

    # Push to config queue so transcription subprocess picks up the new target language
    if config_queue:
        try:
            config_queue.put({"type": "config_update", "config": config.copy()})
        except (OSError, ValueError):
            pass

    # Propagate language change to remote Machine B so its TTS/display/config also updates
    if old_language != new_language:
        _remote_ep = _get_remote_endpoint_safe()
        if _remote_ep:
            try:
                import requests as _req
                _req.post(
                    _remote_ep.rstrip("/") + "/api/translate/language",
                    json={"target_language": new_language},
                    timeout=5,
                )
            except Exception as _e:
                print(f"[HOT-SWITCH] Could not notify remote server of language change: {_e}")

    # Don't clear cache — old segments keep their cached translations (stale-lang fallback).
    # Only new segments will be translated to the new language.
    language_name = TRANSLATION_LANGUAGES.get(new_language, new_language)
    print(f"[LIVE-TRANSLATION] Hot-switched language: {old_language} -> {new_language} ({language_name})")

    # Notify clients so they can cleanly reset their display
    socketio.emit("language_switched", {
        "old_language": old_language,
        "new_language": new_language,
        "language_name": language_name,
    })

    result = {
        "success": True,
        "message": f"Switched to {language_name}. Translations will update shortly.",
        "old_language": old_language,
        "new_language": new_language,
        "language_name": language_name,
    }
    if new_tts_voice:
        result["tts_voice"] = new_tts_voice
        result["tts_backend"] = backend
    return jsonify(result)


@app.route("/api/translation/status", methods=["GET"])
def get_translation_status():
    """Check if translation is active and model loaded"""
    trans_config = config.get("live_translation", {})
    caller_ip = request.remote_addr
    is_local = check_ip_whitelist()
    is_paired = _is_trusted_translation_client(caller_ip)

    # Collect active remote clients (last seen within 60s) and prune stale ones
    with _translation_clients_lock:
        now = time.time()
        active = {ip: ts for ip, ts in _translation_clients.items() if now - ts < 60}
        _translation_clients.clear()
        _translation_clients.update(active)

    remote_cfg = trans_config.get("remote", {})
    remote_active = bool(remote_cfg.get("enabled") and remote_cfg.get("endpoint"))

    # Show Whisper model when using Whisper translation methods
    _method = trans_config.get("translation_method", "nllb")
    _using_whisper = _method in ("whisper_translate", "whisper_forced_lang")
    if _using_whisper:
        _status_model = "whisper/" + config.get("model", {}).get("whisper", {}).get("model", "whisper")
    else:
        _status_model = trans_config.get("translation_model", "facebook/nllb-200-distilled-600M")

    result = {
        "success": True,
        "enabled": trans_config.get("enabled", False),
        "target_language": trans_config.get("target_language", "en"),
        "target_language_name": TRANSLATION_LANGUAGES.get(
            trans_config.get("target_language", "en"), "English"
        ),
        "model_loaded": True if (remote_active or _using_whisper) else is_live_translation_model_loaded(),
        "model_loading": False if (remote_active or _using_whisper) else is_live_translation_model_loading(),
        "translation_model": _status_model,
        "translation_method": _method,
        "remote_active": remote_active,
        "remote_endpoint": remote_cfg.get("endpoint", "") if remote_active else "",
        "cache_size": get_translation_cache().get_size(),
        "is_transcription_running": transcription_state.get("running", False),
    }

    # Only expose sensitive info (clients, pairs) to local/whitelisted or paired callers
    if is_local or is_paired:
        pending = [
            {"ip": ip, "code": v["code"]}
            for ip, v in list(_pending_pair_requests.items())
            if time.time() < v["expires"]
        ]
        result["remote_clients"] = list(active.keys())
        result["trusted_clients"] = list(_trusted_translation_clients)
        result["pending_pairs"] = pending
    else:
        result["remote_clients"] = []
        result["trusted_clients"] = []
        result["pending_pairs"] = []

    return jsonify(result)


@app.route("/api/translation/clear-cache", methods=["POST"])
def clear_translation_cache():
    """Clear the translation cache"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    get_translation_cache().clear()
    return jsonify({
        "success": True,
        "message": "Translation cache cleared"
    })


# Remote translation endpoints

@app.route("/api/translate", methods=["POST"])
def translate_remote():
    """Remote translation endpoint — called by a paired machine (Machine A).
    Body JSON: {text, source_lang, target_lang, return_extras, num_alternatives}
    Returns: {translated_text, confidence?, alternatives?}
    """
    client_ip = request.remote_addr
    if not _is_trusted_translation_client(client_ip):
        return jsonify({"error": "Not paired. Use the pairing flow in the Translations tab."}), 403

    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"translated_text": "", "confidence": None, "alternatives": []}), 200

    _register_translation_client(client_ip)

    cfg = config.get("live_translation", {})
    source_lang = data.get("source_lang", cfg.get("source_language", "auto"))
    target_lang = data.get("target_lang", cfg.get("target_language", "en"))
    return_extras = bool(data.get("return_extras", False))
    num_alternatives = int(data.get("num_alternatives", 0))
    # Use generation_params from request (Machine A's settings) if provided
    generation_params = data.get("generation_params")

    result = translate_live_text(text, source_lang, target_lang,
                                 return_extras=return_extras,
                                 num_alternatives=num_alternatives,
                                 generation_params=generation_params)

    if return_extras and isinstance(result, dict):
        return jsonify({
            "translated_text": result.get("text", text),
            "confidence": result.get("confidence"),
            "alternatives": result.get("alternatives", []),
        })
    return jsonify({"translated_text": result if isinstance(result, str) else text,
                    "confidence": None, "alternatives": []})


@app.route("/api/translate/unload", methods=["POST"])
def translate_unload():
    """Remote unload — called by a paired Machine A to ask this machine to unload its translation model.
    Only unloads if no other trusted clients have been active in the last 60 seconds."""
    client_ip = request.remote_addr
    if not _is_trusted_translation_client(client_ip):
        return jsonify({"error": "Not paired"}), 403

    # Only unload if no other trusted clients are actively translating
    active_others = []
    now = time.time()
    with _translation_clients_lock:
        for ip, last_seen in _translation_clients.items():
            if ip != client_ip and (now - last_seen) < 60:
                active_others.append(ip)

    if active_others:
        return jsonify({"success": False, "reason": "Other clients still active", "active": len(active_others)})

    if is_live_translation_model_loaded():
        import threading as _threading
        _threading.Thread(target=unload_live_translation_model, daemon=True).start()
        return jsonify({"success": True, "message": "Unloading translation model"})

    return jsonify({"success": True, "message": "Model not loaded"})


@app.route("/api/translate/preload", methods=["POST"])
def translate_preload():
    """Remote preload — called by a paired Machine A when it starts transcription.
    Loads the translation model in the background so it's ready for requests."""
    client_ip = request.remote_addr
    if not _is_trusted_translation_client(client_ip):
        return jsonify({"error": "Not paired"}), 403

    if is_live_translation_model_loaded():
        return jsonify({"success": True, "message": "Model already loaded"})

    if is_live_translation_model_loading():
        return jsonify({"success": True, "message": "Model already loading"})

    cfg = config.get("live_translation", {})
    if not cfg.get("enabled", False):
        return jsonify({"success": False, "message": "Translation not enabled on this machine"})

    use_gpu = cfg.get("use_gpu", True)
    model_id = cfg.get("translation_model")

    def _preload():
        print(f"[PRELOAD] Loading translation model for remote client {client_ip}...")
        get_live_translation_model(use_gpu, model_id)
        print("[PRELOAD] Translation model loaded and ready")

    import threading
    threading.Thread(target=_preload, daemon=True).start()
    return jsonify({"success": True, "message": "Loading translation model"})


@app.route("/api/translate/pair/request", methods=["POST"])
def translation_pair_request():
    """Machine A calls this to initiate pairing. Machine B shows the code."""
    client_ip = request.remote_addr
    code = str(random.randint(100000, 999999))
    _pending_pair_requests[client_ip] = {"code": code, "expires": time.time() + 300}
    socketio.emit("translation_pair_request", {"ip": client_ip, "code": code})
    return jsonify({"status": "pending", "message": "Check the Translations tab on Machine B for the 6-digit code"})


@app.route("/api/translate/pair/confirm", methods=["POST"])
def translation_pair_confirm():
    """Machine A calls this with the code displayed on Machine B."""
    client_ip = request.remote_addr
    data = request.get_json() or {}
    code = str(data.get("code", "")).strip()
    pending = _pending_pair_requests.get(client_ip)
    if not pending or time.time() > pending["expires"] or pending["code"] != code:
        return jsonify({"error": "Invalid or expired code"}), 400
    _add_trusted_client(client_ip)
    del _pending_pair_requests[client_ip]
    socketio.emit("translation_pair_confirmed", {"ip": client_ip})
    return jsonify({"status": "paired"})


@app.route("/api/translate/pair/respond", methods=["POST"])
def translation_pair_respond():
    """Machine B's Allow/Deny buttons call this."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    data = request.get_json() or {}
    client_ip = data.get("ip", "")
    allow = bool(data.get("allow", False))
    if allow and client_ip:
        _add_trusted_client(client_ip)
        _pending_pair_requests.pop(client_ip, None)
        socketio.emit("translation_pair_confirmed", {"ip": client_ip})
    else:
        _pending_pair_requests.pop(client_ip, None)
        socketio.emit("translation_pair_denied", {"ip": client_ip})
    return jsonify({"success": True})


@app.route("/api/translate/pair/status", methods=["GET"])
def translation_pair_status():
    """Machine A polls this to check if it is paired."""
    client_ip = request.remote_addr
    return jsonify({"paired": _is_trusted_translation_client(client_ip)})


@app.route("/api/translate/pair/unpair", methods=["POST"])
def translation_unpair():
    """Remove a trusted client IP from the paired list."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    data = request.get_json() or {}
    ip = data.get("ip", "").strip()
    if not ip:
        return jsonify({"success": False, "error": "ip required"}), 400
    _trusted_translation_clients.discard(ip)
    trusted = config.get("live_translation", {}).get("trusted_clients", [])
    if ip in trusted:
        trusted.remove(ip)
    save_config(config)
    socketio.emit("translation_pair_denied", {"ip": ip})
    return jsonify({"success": True})


@app.route("/api/translate/language", methods=["POST"])
def translate_remote_language():
    """Paired Machine A calls this to switch Machine B's target translation language."""
    client_ip = request.remote_addr
    if not _is_trusted_translation_client(client_ip):
        return jsonify({"error": "Not paired"}), 403

    global config
    data = request.get_json() or {}
    new_language = data.get("target_language")
    if not new_language or new_language not in NLLB_LANG_CODES:
        return jsonify({"error": "Invalid language"}), 400

    old_language = config.get("live_translation", {}).get("target_language", "en")
    if "live_translation" not in config:
        config["live_translation"] = {}
    config["live_translation"]["target_language"] = new_language

    # Auto-switch TTS voice/model on Machine B to match the new language
    new_tts_voice = None
    tts_section = config["live_translation"].setdefault("tts", {})
    backend = tts_section.get("backend", "edge")
    if old_language != new_language:
        if backend == "edge":
            prefs = tts_section.setdefault("edge_voice_preferences", {})
            current_voice = tts_section.get("edge_voice", "")
            if current_voice:
                prefs[old_language] = current_voice
            new_tts_voice = prefs.get(new_language) or _pick_default_edge_voice(new_language)
            if new_tts_voice:
                tts_section["edge_voice"] = new_tts_voice
        elif backend == "piper":
            prefs = tts_section.setdefault("piper_model_preferences", {})
            current_model = tts_section.get("piper_model", "")
            if current_model:
                prefs[old_language] = current_model
            new_tts_voice = prefs.get(new_language) or _pick_default_piper_model(new_language)
            if new_tts_voice:
                tts_section["piper_model"] = new_tts_voice
                def _reload():
                    unload_tts_model()
                    get_tts_model(model_name=new_tts_voice)
                threading.Thread(target=_reload, daemon=True).start()

    save_config(config)
    language_name = TRANSLATION_LANGUAGES.get(new_language, new_language)
    print(f"[LIVE-TRANSLATION] Remote hot-switch: {old_language} -> {new_language} ({language_name})")
    socketio.emit("language_switched", {
        "old_language": old_language,
        "new_language": new_language,
        "language_name": language_name,
    })
    return jsonify({"success": True, "language_name": language_name})


@app.route("/api/translate/pair/unpair-me", methods=["POST"])
def translation_unpair_me():
    """A paired Machine A calls this to remove itself from Machine B's trusted list."""
    client_ip = request.remote_addr
    if not _is_trusted_translation_client(client_ip):
        return jsonify({"success": False, "error": "Not paired"}), 403
    _trusted_translation_clients.discard(client_ip)
    trusted = config.get("live_translation", {}).get("trusted_clients", [])
    if client_ip in trusted:
        trusted.remove(client_ip)
    save_config(config)
    # Unload model if no other clients remain
    if not _trusted_translation_clients and is_live_translation_model_loaded():
        import threading
        threading.Thread(target=unload_live_translation_model, daemon=True).start()
    socketio.emit("translation_pair_denied", {"ip": client_ip})
    return jsonify({"success": True, "message": f"Unpaired {client_ip}"})


# =============================================================================
# TTS (Text-to-Speech) API Endpoints
# =============================================================================

@app.route("/api/tts/status", methods=["GET"])
def get_tts_status():
    """Get TTS status for both edge-tts and piper backends"""
    tts_config = config.get("live_translation", {}).get("tts", {})
    backend = _get_tts_backend()

    result = {
        "success": True,
        "enabled": tts_config.get("enabled", False),
        "backend": backend,
        "model_loaded": is_tts_model_loaded(),
        "model_loading": is_tts_model_loading(),
        "downloading": _tts_download_status.get("status") == "downloading",
        "speed": tts_config.get("speed", 1.0),
    }

    if backend == "edge":
        result["edge_voice"] = tts_config.get("edge_voice", "en-US-AriaNeural")
        result["edge_available"] = True
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            result["edge_available"] = False
    elif backend == "piper":
        result["piper_model"] = tts_config.get("piper_model", "")
        result["piper_available"] = True
        try:
            import piper  # noqa: F401
        except ImportError:
            result["piper_available"] = False

    return jsonify(result)


@app.route("/api/tts/voices", methods=["GET"])
def get_tts_voices():
    """Get available edge-tts voices, optionally filtered by language"""
    lang_filter = request.args.get("language", "").strip().lower()
    try:
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            return jsonify({"success": False, "error": "edge-tts not installed. Run: pip install edge-tts"}), 500

        voices = get_edge_tts_voices()
        if not voices:
            return jsonify({"success": True, "voices": [], "error": "Could not fetch voices (network issue?)"})

        result = []
        for v in voices:
            locale = v.get("Locale", "").lower()
            if lang_filter and not locale.startswith(lang_filter):
                continue
            result.append({
                "id": v.get("ShortName", ""),
                "name": v.get("FriendlyName", v.get("ShortName", "")),
                "gender": v.get("Gender", ""),
                "locale": v.get("Locale", ""),
            })
        return jsonify({"success": True, "voices": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/tts/settings", methods=["POST"])
def save_tts_settings():
    """Save TTS settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    data = request.json

    if "live_translation" not in config:
        config["live_translation"] = {}
    if "tts" not in config["live_translation"]:
        config["live_translation"]["tts"] = {}

    old_enabled = config["live_translation"]["tts"].get("enabled", False)
    old_backend = config["live_translation"]["tts"].get("backend", "edge")
    old_piper_model = config["live_translation"]["tts"].get("piper_model", "")

    allowed_keys = ["enabled", "backend", "edge_voice", "piper_model", "speed"]
    for key in allowed_keys:
        if key in data:
            config["live_translation"]["tts"][key] = data[key]

    # Save manual voice/model selection as per-language preference
    target_lang = config.get("live_translation", {}).get("target_language", "en")
    if "edge_voice" in data and data["edge_voice"]:
        prefs = config["live_translation"]["tts"].setdefault("edge_voice_preferences", {})
        prefs[target_lang] = data["edge_voice"]
    if "piper_model" in data and data["piper_model"]:
        prefs = config["live_translation"]["tts"].setdefault("piper_model_preferences", {})
        prefs[target_lang] = data["piper_model"]

    save_config(config)

    now_enabled = config["live_translation"]["tts"].get("enabled", False)
    new_backend = config["live_translation"]["tts"].get("backend", "edge")
    new_piper_model = config["live_translation"]["tts"].get("piper_model", "")

    # Handle piper model loading/unloading
    if new_backend == "piper":
        if not now_enabled and old_enabled:
            threading.Thread(target=unload_tts_model, daemon=True).start()
        elif now_enabled and (old_backend != "piper" or old_piper_model != new_piper_model):
            def reload_piper():
                unload_tts_model()
                get_tts_model(model_name=new_piper_model)
            threading.Thread(target=reload_piper, daemon=True).start()
    elif old_backend == "piper":
        # Switched away from piper, unload
        threading.Thread(target=unload_tts_model, daemon=True).start()

    return jsonify({
        "success": True,
        "message": "TTS settings saved."
    })


# ─── Piper TTS model management ─────────────────────────────────────────────

_PIPER_MODELS_CATALOG = [
    {"id": "en_US-lessac-medium", "name": "Lessac (US English)", "language": "en", "quality": "medium", "size": "75MB"},
    {"id": "en_US-amy-medium", "name": "Amy (US English)", "language": "en", "quality": "medium", "size": "75MB"},
    {"id": "en_US-ryan-medium", "name": "Ryan (US English)", "language": "en", "quality": "medium", "size": "75MB"},
    {"id": "en_GB-alba-medium", "name": "Alba (British English)", "language": "en", "quality": "medium", "size": "75MB"},
    {"id": "de_DE-thorsten-medium", "name": "Thorsten (German)", "language": "de", "quality": "medium", "size": "75MB"},
    {"id": "es_ES-davefx-medium", "name": "Davefx (Spanish)", "language": "es", "quality": "medium", "size": "75MB"},
    {"id": "fr_FR-siwis-medium", "name": "Siwis (French)", "language": "fr", "quality": "medium", "size": "75MB"},
    {"id": "ru_RU-ruslan-medium", "name": "Ruslan (Russian)", "language": "ru", "quality": "medium", "size": "75MB"},
    {"id": "uk_UA-ukrainian_tts-medium", "name": "Ukrainian TTS", "language": "uk", "quality": "medium", "size": "75MB"},
    {"id": "zh_CN-huayan-medium", "name": "Huayan (Chinese)", "language": "zh", "quality": "medium", "size": "75MB"},
]

_tts_download_status = {"status": "idle", "model": "", "error": ""}


def _get_piper_model_dir(model_id):
    """Get the directory for a piper model"""
    return os.path.join(_tts_cache_dir, "piper", model_id)


def _is_piper_model_downloaded(model_id):
    """Check if a piper model is downloaded"""
    model_dir = _get_piper_model_dir(model_id)
    if os.path.isdir(model_dir):
        return any(f.endswith(".onnx") for f in os.listdir(model_dir))
    return False


@app.route("/api/models/tts-list", methods=["GET"])
def list_tts_models_catalog():
    """List TTS models (piper) with download status"""
    try:
        models = []
        for m in _PIPER_MODELS_CATALOG:
            entry = dict(m)
            entry["downloaded"] = _is_piper_model_downloaded(m["id"])
            models.append(entry)
        return jsonify({"success": True, "models": models})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/tts/download", methods=["POST"])
def download_tts_model():
    """Download a piper TTS model from HuggingFace"""
    global _tts_download_status
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = request.get_json() or {}
    model_name = data.get("model_name", "").strip()
    if not model_name:
        return jsonify({"success": False, "error": "model_name required"}), 400

    # Parse model ID: e.g., "en_US-lessac-medium" -> lang "en_US", name "lessac", quality "medium"
    parts = model_name.split("-")
    if len(parts) < 3:
        return jsonify({"success": False, "error": f"Invalid piper model ID format: {model_name}"}), 400

    lang_code, voice_name, quality = parts[0], parts[1], parts[2]

    # HuggingFace piper voices URL
    lang_family = lang_code.split("_")[0]  # "en_US" -> "en"
    base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang_family}/{lang_code}/{voice_name}/{quality}"
    onnx_url = f"{base_url}/{model_name}.onnx"
    json_url = f"{base_url}/{model_name}.onnx.json"

    # Best-effort total size for a real progress percentage (json config is negligible)
    total_size = None
    try:
        import urllib.request
        req = urllib.request.Request(onnx_url, method="HEAD")
        with urllib.request.urlopen(req, timeout=15) as resp:
            total_size = int(resp.headers.get("Content-Length") or 0) or None
    except Exception as e:
        print(f"[TTS] Could not get size of {model_name}: {e}")

    download_key = f"tts-{model_name}"
    if not try_register_download(download_key, total=total_size):
        return jsonify({"success": False, "error": "Download already in progress"}), 409

    def _do_download():
        global _tts_download_status
        _tts_download_status = {"status": "downloading", "model": model_name, "error": ""}
        model_dir = _get_piper_model_dir(model_name)
        try:
            os.makedirs(model_dir, exist_ok=True)
            start_download_monitor(download_key, model_dir, total=total_size)

            print(f"[TTS] Downloading piper model: {model_name}")
            for url, filename in ((onnx_url, f"{model_name}.onnx"),
                                  (json_url, f"{model_name}.onnx.json")):
                print(f"[TTS]   {url}")
                outcome = download_url_to_file(
                    url, os.path.join(model_dir, filename),
                    cancel_check=lambda: download_key in cancelled_downloads,
                )
                if outcome == "cancelled":
                    print(f"[TTS] Download cancelled: {model_name}")
                    # Piper models live under models/tts/piper/, which the generic
                    # cancel-route cleanup doesn't know about — clean up here
                    shutil.rmtree(model_dir, ignore_errors=True)
                    _tts_download_status = {"status": "failed", "model": model_name, "error": "Cancelled"}
                    finish_download(download_key, cancelled=True)
                    return

            print(f"[TTS] Piper model downloaded: {model_name}")
            _tts_download_status = {"status": "completed", "model": model_name, "error": ""}
            finish_download(download_key)
        except Exception as e:
            print(f"[TTS ERROR] Download failed: {e}")
            _tts_download_status = {"status": "failed", "model": model_name, "error": str(e)}
            finish_download(download_key, error=e)

    threading.Thread(target=_do_download, daemon=True).start()
    return jsonify({"success": True, "message": f"Downloading {model_name}..."})


@app.route("/api/models/tts/download-progress", methods=["GET"])
def tts_download_progress():
    """Get TTS model download progress"""
    return jsonify(_tts_download_status)


@app.route("/api/models/tts/remove", methods=["POST"])
def remove_tts_model():
    """Remove a downloaded piper TTS model"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = request.get_json() or {}
    model_name = data.get("model_name", "").strip()
    if not model_name:
        return jsonify({"success": False, "error": "model_name required"}), 400

    # Unload if this is the currently loaded model
    tts_config = config.get("live_translation", {}).get("tts", {})
    if tts_config.get("piper_model") == model_name and is_tts_model_loaded():
        unload_tts_model()

    try:
        model_dir = _get_piper_model_dir(model_name)
        if os.path.isdir(model_dir):
            import shutil
            shutil.rmtree(model_dir, ignore_errors=True)
            return jsonify({"success": True, "message": f"Removed {model_name}"})
        return jsonify({"success": False, "error": "Model not found on disk"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/tts/models", methods=["GET"])
def list_tts_models():
    """List available TTS voices/models for the active backend"""
    backend = _get_tts_backend()
    if backend == "edge":
        # Return edge-tts voices
        lang_filter = request.args.get("language", "").strip().lower()
        voices = get_edge_tts_voices()
        result = []
        for v in voices:
            locale = v.get("Locale", "").lower()
            if lang_filter and not locale.startswith(lang_filter):
                continue
            result.append(v.get("ShortName", ""))
        return jsonify({"success": True, "models": result})
    else:
        # Return piper models
        models = [m["id"] for m in _PIPER_MODELS_CATALOG if _is_piper_model_downloaded(m["id"])]
        return jsonify({"success": True, "models": models})


# Proxy endpoints — forward browser requests to Machine B server-side (avoids CORS)

class _RemoteEndpointError(Exception):
    pass


def _probe_remote_port(base_url):
    """Try common ports to find which one the STT app is listening on."""
    from urllib.parse import urlparse
    import requests as _req
    parsed = urlparse(base_url)
    hostname = parsed.hostname
    for port in [80, 8080, 443, 5000, 8000]:
        scheme = "https" if port == 443 else "http"
        try:
            r = _req.get(f"{scheme}://{hostname}:{port}/api/translation/status", timeout=2)
            if r.status_code == 200:
                return f"{scheme}://{hostname}:{port}"
        except Exception:
            continue
    return None


def _get_remote_endpoint():
    from urllib.parse import urlparse
    remote_cfg = config.get("live_translation", {}).get("remote", {})
    if not (remote_cfg.get("enabled") and remote_cfg.get("endpoint")):
        return None
    ep = remote_cfg["endpoint"].strip().rstrip("/")
    if not ep:
        return None
    if "://" not in ep:
        ep = "http://" + ep
    # If no port specified, probe for it and save the result
    if not urlparse(ep).port:
        found = _probe_remote_port(ep)
        if found:
            remote_cfg["endpoint"] = found
            save_config(config)
            ep = found
        else:
            host = urlparse(ep).hostname
            raise _RemoteEndpointError(
                f"Could not find STT server on {host} — try specifying the port manually (e.g. {host}:8080)"
            )
    return ep


def _get_remote_endpoint_safe():
    """Return the remote endpoint URL, or None if not configured or unreachable."""
    try:
        return _get_remote_endpoint()
    except _RemoteEndpointError:
        return None


@app.route("/api/remote-translation/status", methods=["GET"])
def proxy_remote_translation_status():
    """Proxy: fetch Machine B's translation status for display on Machine A's UI."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    try:
        endpoint = _get_remote_endpoint()
    except _RemoteEndpointError as e:
        return jsonify({"success": False, "error": str(e)}), 502
    if not endpoint:
        return jsonify({"success": False, "error": "No remote endpoint configured"}), 400
    import requests as _req
    try:
        r = _req.get(endpoint + "/api/translation/status", timeout=5)
        try:
            data = r.json()
        except ValueError:
            data = {"success": False, "error": "Invalid JSON from remote"}
        return jsonify(data), r.status_code
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 502


@app.route("/api/remote-translation/pair/request", methods=["POST"])
def proxy_pair_request():
    """Proxy: send pairing request from Machine A's server to Machine B (avoids CORS)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    try:
        endpoint = _get_remote_endpoint()
    except _RemoteEndpointError as e:
        return jsonify({"error": str(e)}), 502
    if not endpoint:
        return jsonify({"success": False, "error": "No remote endpoint configured"}), 400
    import requests as _req
    try:
        r = _req.post(endpoint + "/api/translate/pair/request", timeout=10)
        try:
            data = r.json()
        except ValueError:
            data = {"error": "Invalid JSON from remote"}
        return jsonify(data), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/remote-translation/pair/confirm", methods=["POST"])
def proxy_pair_confirm():
    """Proxy: send pairing confirmation from Machine A's server to Machine B (avoids CORS)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    try:
        endpoint = _get_remote_endpoint()
    except _RemoteEndpointError as e:
        return jsonify({"error": str(e)}), 502
    if not endpoint:
        return jsonify({"success": False, "error": "No remote endpoint configured"}), 400
    import requests as _req
    try:
        r = _req.post(endpoint + "/api/translate/pair/confirm",
                      json=request.get_json() or {}, timeout=10)
        try:
            data = r.json()
        except ValueError:
            data = {"error": "Invalid JSON from remote"}
        return jsonify(data), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/remote-translation/pair/status", methods=["GET"])
def proxy_pair_status():
    """Proxy: check if Machine A is paired with Machine B (avoids CORS)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    try:
        endpoint = _get_remote_endpoint()
    except _RemoteEndpointError as e:
        return jsonify({"paired": False, "error": str(e)}), 502
    if not endpoint:
        return jsonify({"paired": False}), 200
    import requests as _req
    try:
        r = _req.get(endpoint + "/api/translate/pair/status", timeout=5)
        try:
            data = r.json()
        except ValueError:
            data = {"paired": False, "error": "Invalid JSON from remote"}
        return jsonify(data), r.status_code
    except Exception as e:
        return jsonify({"paired": False, "error": str(e)}), 200


@app.route("/api/remote-translation/unload", methods=["POST"])
def proxy_translate_unload():
    """Proxy: tell Machine B to unload its translation model (avoids CORS)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    try:
        endpoint = _get_remote_endpoint()
    except _RemoteEndpointError as e:
        return jsonify({"success": False, "error": str(e)}), 502
    if not endpoint:
        return jsonify({"success": False, "error": "No remote endpoint configured"}), 400
    import requests as _req
    try:
        r = _req.post(endpoint + "/api/translate/unload", timeout=10)
        try:
            data = r.json()
        except ValueError:
            data = {"success": False, "error": "Invalid JSON from remote"}
        return jsonify(data), r.status_code
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 502


@app.route("/api/remote-translation/preload", methods=["POST"])
def proxy_translate_preload():
    """Proxy: tell Machine B to preload its translation model (avoids CORS)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    try:
        endpoint = _get_remote_endpoint()
    except _RemoteEndpointError as e:
        return jsonify({"success": False, "error": str(e)}), 502
    if not endpoint:
        return jsonify({"success": False, "error": "No remote endpoint configured"}), 400
    import requests as _req
    try:
        r = _req.post(endpoint + "/api/translate/preload", timeout=10)
        try:
            data = r.json()
        except ValueError:
            data = {"success": False, "error": "Invalid JSON from remote"}
        return jsonify(data), r.status_code
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 502


@app.route("/api/remote-translation/unpair", methods=["POST"])
def proxy_translate_unpair():
    """Proxy: tell Machine B to remove this machine from its trusted list."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    try:
        endpoint = _get_remote_endpoint()
    except _RemoteEndpointError as e:
        return jsonify({"success": False, "error": str(e)}), 502
    if not endpoint:
        return jsonify({"success": False, "error": "No remote endpoint configured"}), 400
    import requests as _req
    try:
        r = _req.post(endpoint + "/api/translate/pair/unpair-me", timeout=10)
        try:
            data = r.json()
        except ValueError:
            data = {"success": False, "error": "Invalid JSON from remote"}
        return jsonify(data), r.status_code
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 502


# Transcription Language Endpoints


@app.route("/api/transcription/language", methods=["GET"])
def get_transcription_language():
    """Get current transcription language
    Example: GET /api/transcription/language"""
    language = config.get("audio", {}).get("language", "auto")
    return jsonify({
        "success": True,
        "language": language
    })


@app.route("/api/transcription/language", methods=["POST"])
def hot_switch_transcription_language():
    """Hot-switch transcription language without restart
    Example: POST /api/transcription/language {"language": "en"}
    Example: POST /api/transcription/language {"language": "auto"}"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    data = request.json
    new_language = data.get("language")

    if not new_language:
        return jsonify({"success": False, "error": "language required"}), 400

    old_language = config.get("audio", {}).get("language", "auto")

    if "audio" not in config:
        config["audio"] = {}

    config["audio"]["language"] = new_language
    save_config(config)

    # Push to config queue for hot-reload
    if config_queue:
        config_queue.put({"type": "config_update", "config": config.copy()})

    print(f"[TRANSCRIPTION] Hot-switched language: {old_language} -> {new_language}")

    return jsonify({
        "success": True,
        "message": f"Transcription language switched to {new_language}. Takes effect on next audio chunk.",
        "old_language": old_language,
        "new_language": new_language
    })


@app.route("/api/language", methods=["GET"])
def get_all_languages():
    """Get current transcription and translation languages
    Example: GET /api/language"""
    transcription_lang = config.get("audio", {}).get("language", "auto")
    translation_lang = config.get("live_translation", {}).get("target_language", "en")
    translation_name = TRANSLATION_LANGUAGES.get(translation_lang, translation_lang)

    return jsonify({
        "success": True,
        "transcription": transcription_lang,
        "translation": translation_lang,
        "translation_name": translation_name
    })


@app.route("/api/language", methods=["POST"])
def hot_switch_all_languages():
    """Hot-switch both transcription and translation languages
    Example: POST /api/language {"transcription": "en", "translation": "es"}
    Example: POST /api/language {"transcription": "auto", "translation": "fr"}"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    data = request.json

    transcription_lang = data.get("transcription")
    translation_lang = data.get("translation")

    if not transcription_lang and not translation_lang:
        return jsonify({"success": False, "error": "At least one of 'transcription' or 'translation' required"}), 400

    results = {}

    # Update transcription language
    if transcription_lang:
        old_trans = config.get("audio", {}).get("language", "auto")
        if "audio" not in config:
            config["audio"] = {}
        config["audio"]["language"] = transcription_lang
        results["transcription"] = {
            "old": old_trans,
            "new": transcription_lang
        }
        print(f"[TRANSCRIPTION] Hot-switched language: {old_trans} -> {transcription_lang}")

    # Update translation language
    if translation_lang:
        if translation_lang not in NLLB_LANG_CODES:
            return jsonify({"success": False, "error": f"Invalid translation language: {translation_lang}"}), 400

        old_target = config.get("live_translation", {}).get("target_language", "en")
        if "live_translation" not in config:
            config["live_translation"] = {}
        config["live_translation"]["target_language"] = translation_lang

        language_name = TRANSLATION_LANGUAGES.get(translation_lang, translation_lang)
        results["translation"] = {
            "old": old_target,
            "new": translation_lang,
            "language_name": language_name
        }
        print(f"[LIVE-TRANSLATION] Hot-switched language: {old_target} -> {translation_lang}")

    save_config(config)

    # Push to config queue for transcription hot-reload
    if config_queue and transcription_lang:
        config_queue.put({"type": "config_update", "config": config.copy()})

    return jsonify({
        "success": True,
        "message": "Language settings updated. Changes take effect immediately.",
        "changes": results
    })


# File Transcription Endpoints


@app.route("/file")
def upload_page():
    """Render file transcription page"""
    if not check_ip_whitelist():
        return render_template("auth-required.html"), 403
    return render_template("file.html")


@app.route("/model-manager")
def model_manager_page():
    """Render model manager page"""
    if not check_ip_whitelist():
        return render_template("auth-required.html"), 403
    return render_template("model-manager.html")


@app.route("/file-manager")
def file_manager_page():
    """Render file manager page"""
    if not check_ip_whitelist():
        return render_template("auth-required.html"), 403
    return render_template("file-manager.html")


@app.route("/api/file-transcription-settings", methods=["GET", "POST"])
def file_transcription_settings_endpoint():
    """Get or update file transcription settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config

    if request.method == "GET":
        # Return current file transcription settings
        ft_config = config.get("file_transcription", {})
        model_config = ft_config.get("model", {})

        settings = {
            "model_type": model_config.get("type", "whisper"),
            "whisper_model": model_config.get("whisper", {}).get("model", "base"),
            "hf_model": model_config.get("huggingface", {}).get(
                "model_id", "openai/whisper-base"
            ),
            "language": ft_config.get("language", "auto"),
            "use_gpu": ft_config.get("use_gpu", True),
            "use_flash_attention": model_config.get("huggingface", {}).get(
                "use_flash_attention", False
            ),
        }

        return jsonify({"success": True, "settings": settings})

    elif request.method == "POST":
        # Update file transcription settings
        try:
            new_settings = request.json

            # Update config
            if "file_transcription" not in config:
                config["file_transcription"] = {}

            if "model" not in config["file_transcription"]:
                config["file_transcription"]["model"] = {}

            # Update model type
            if "model_type" in new_settings:
                config["file_transcription"]["model"]["type"] = new_settings[
                    "model_type"
                ]

            # Update Whisper settings
            if "whisper" not in config["file_transcription"]["model"]:
                config["file_transcription"]["model"]["whisper"] = {}

            if "whisper_model" in new_settings:
                config["file_transcription"]["model"]["whisper"]["model"] = (
                    new_settings["whisper_model"]
                )

            # Update HuggingFace settings
            if "huggingface" not in config["file_transcription"]["model"]:
                config["file_transcription"]["model"]["huggingface"] = {}

            if "hf_model" in new_settings:
                config["file_transcription"]["model"]["huggingface"]["model_id"] = (
                    new_settings["hf_model"]
                )

            if "use_flash_attention" in new_settings:
                config["file_transcription"]["model"]["huggingface"][
                    "use_flash_attention"
                ] = new_settings["use_flash_attention"]

            # Update other settings
            if "language" in new_settings:
                config["file_transcription"]["language"] = new_settings["language"]

            if "use_gpu" in new_settings:
                config["file_transcription"]["use_gpu"] = new_settings["use_gpu"]

            # Save config to file
            try:
                with _config_file_lock:
                    _atomic_write_json(CONFIG_FILE, config)
                print(
                    "[OK] File transcription settings updated and saved to config.json"
                )
            except Exception as e:
                print(f"[WARNING] Failed to save config to file: {e}")

            return jsonify(
                {"success": True, "message": "Settings updated successfully"}
            )

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/transcribe-file", methods=["POST"])
def transcribe_file_endpoint():
    """Handle file upload and start transcription in background thread"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    temp_upload = None
    try:
        # Get uploaded file
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files["file"]
        output_format = request.form.get("format", "txt")
        if output_format not in ("txt", "srt", "vtt", "json"):
            return jsonify({"success": False, "error": f"Invalid format: {output_format}"}), 400
        language = request.form.get("language", "auto")  # Get language from upload form
        translate_to = request.form.get("translate_to", "")  # Optional translation target

        # Validate file
        is_valid, error_msg = validate_file(file)
        if not is_valid:
            return jsonify({"success": False, "error": error_msg}), 400

        # Save uploaded file temporarily with proper cleanup
        temp_upload = tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename)[1]
        )
        file.save(temp_upload.name)
        temp_upload.close()

        # Generate session ID
        session_id = str(uuid.uuid4())

        # Start transcription in background thread with proper error handling
        def safe_transcription():
            try:
                process_file_transcription(
                    temp_upload.name, output_format, session_id, file.filename, language, translate_to
                )
            except Exception as e:
                print(f"[ERROR] Background transcription failed: {e}")
                socketio.emit(
                    "file_error",
                    {
                        "session_id": session_id,
                        "error": f"Transcription failed: {str(e)}",
                    },
                )

        thread = threading.Thread(target=safe_transcription)
        thread.daemon = True
        thread.start()

        return jsonify(
            {
                "success": True,
                "session_id": session_id,
                "message": "Transcription started",
            }
        )

    except Exception as e:
        # Clean up temp file if it exists
        if temp_upload and os.path.exists(temp_upload.name):
            try:
                os.unlink(temp_upload.name)
            except OSError:
                pass
        return jsonify({"success": False, "error": str(e)}), 500


def process_file_transcription(file_path, output_format, session_id, filename, language=None, translate_to=None):
    """Process file transcription in background thread with proper resource cleanup

    Args:
        file_path: Path to the uploaded file
        output_format: Output format (txt, srt, vtt, json)
        session_id: Unique session ID for progress tracking
        filename: Original filename
        language: Source language code (or 'auto')
        translate_to: Target language code for translation (optional)
    """
    import gc
    wav_path = None
    model = None
    processor = None
    translation_model = None
    translation_tokenizer = None

    try:
        # Send initial progress
        socketio.emit(
            "file_progress",
            {"session_id": session_id, "percent": 5, "status": "Extracting audio..."},
        )

        # Extract/convert audio to WAV
        wav_path = extract_audio_from_file(file_path)

        socketio.emit(
            "file_progress",
            {"session_id": session_id, "percent": 10, "status": "Loading audio..."},
        )

        # Load entire audio file for transcription (no chunking - let Whisper handle segmentation)
        import librosa
        audio_data, sr = librosa.load(wav_path, sr=16000)
        audio_duration = len(audio_data) / sr

        socketio.emit(
            "file_progress",
            {
                "session_id": session_id,
                "percent": 20,
                "status": f"Loading model... (audio: {int(audio_duration // 60)}m {int(audio_duration % 60)}s)",
            },
        )

        # Load model using file transcription settings (or fall back to main config)
        ft_config = config.get("file_transcription", {})
        ft_model_config = ft_config.get("model", config["model"])
        ft_use_gpu = ft_config.get(
            "use_gpu", config.get("performance", {}).get("use_gpu", True)
        )
        # Use language from upload request, or fall back to config, or finally default to auto
        ft_language = language if language is not None else ft_config.get("language", "auto")

        model, processor, model_type = ModelFactory.load_model(
            ft_model_config, ft_use_gpu
        )

        socketio.emit(
            "file_progress",
            {
                "session_id": session_id,
                "percent": 30,
                "status": "Transcribing audio (this may take a while)...",
            },
        )

        # Transcribe entire audio file at once - Whisper handles segmentation naturally
        whisper_params = config.get("whisper_decoding", {}).get(
            "file_transcription", FILE_TRANSCRIPTION_PARAMS
        )

        segments = ModelFactory.transcribe(
            model, processor, model_type, audio_data,
            language=ft_language, whisper_params=whisper_params,
            return_segments=True
        )
        segments = [dict(s, text=apply_profanity_filter(s.get("text", ""))) for s in segments]

        socketio.emit(
            "file_progress",
            {"session_id": session_id, "percent": 55, "status": f"Found {len(segments)} segments..."},
        )

        # Initialize translated_segments as None (will be populated if translation is requested)
        translated_segments = None
        target_language_name = None

        # Handle translation if requested
        if translate_to and translate_to.strip() and translate_to in NLLB_LANG_CODES:
            source_lang = ft_language if ft_language != "auto" else "en"

            # Check translation method
            ft_translation_method = config.get("live_translation", {}).get("translation_method", "nllb")
            remote_cfg = config.get("live_translation", {}).get("remote", {})

            if ft_translation_method in ("whisper_translate", "whisper_forced_lang") and model is not None:
                # Whisper-based translation: run a second pass on the same audio with translation params
                socketio.emit(
                    "file_progress",
                    {"session_id": session_id, "percent": 60, "status": "Translating with Whisper (pass 2)..."},
                )

                pass2_params = dict(whisper_params)
                pass2_language = ft_language
                if ft_translation_method == "whisper_translate" and translate_to == "en":
                    pass2_params["task"] = "translate"
                elif ft_translation_method == "whisper_forced_lang":
                    pass2_language = translate_to

                pass2_segments = ModelFactory.transcribe(
                    model, processor, model_type, audio_data,
                    language=pass2_language, whisper_params=pass2_params,
                    return_segments=True
                )

                socketio.emit(
                    "file_progress",
                    {"session_id": session_id, "percent": 85, "status": f"Whisper translation: {len(pass2_segments)} segments..."},
                )

                # Build translated_segments by matching pass 1 and pass 2 results
                translated_segments = []
                for i, seg in enumerate(segments):
                    translated_seg = dict(seg)
                    if i < len(pass2_segments):
                        translated_seg["translated_text"] = pass2_segments[i].get("text", "").strip()
                    else:
                        # More segments in pass 1 than pass 2 — use last pass 2 text or original
                        translated_seg["translated_text"] = pass2_segments[-1].get("text", "").strip() if pass2_segments else seg.get("text", "")
                    translated_segments.append(translated_seg)

                # If pass 2 had more segments, append remaining translated text to last segment
                if len(pass2_segments) > len(segments) and translated_segments:
                    extra_text = " ".join(s.get("text", "").strip() for s in pass2_segments[len(segments):] if s.get("text", "").strip())
                    if extra_text:
                        translated_segments[-1]["translated_text"] += " " + extra_text

            elif remote_cfg.get("enabled") and remote_cfg.get("endpoint"):
                # Remote path: send each segment to Machine B, no local model load needed
                try:
                    _file_remote_ep = _get_remote_endpoint()
                except _RemoteEndpointError as e:
                    print(f"[FILE_TRANSLATE] Endpoint error: {e}")
                    _file_remote_ep = None
                socketio.emit(
                    "file_progress",
                    {"session_id": session_id, "percent": 65, "status": "Translating via remote server..."},
                )
                translated_segments = []
                total = len(segments)
                for i, seg in enumerate(segments):
                    text = seg.get("text", "").strip()
                    translated_text = _translate_via_remote(text, source_lang, translate_to, _file_remote_ep) if (text and _file_remote_ep) else text
                    translated_seg = dict(seg)
                    translated_seg["translated_text"] = translated_text
                    translated_segments.append(translated_seg)
                    if total > 0:
                        pct = 65 + int(30 * (i + 1) / total)
                        socketio.emit("file_progress", {"session_id": session_id, "percent": pct, "status": f"Translating... {i+1}/{total}"})
            else:
                # Local path: unload transcription model to free VRAM, load NLLB locally
                socketio.emit(
                    "file_progress",
                    {"session_id": session_id, "percent": 60, "status": "Unloading transcription model..."},
                )

                # CRITICAL: Unload transcription model BEFORE loading translation model
                # This frees GPU memory for the translation model
                if model:
                    del model
                    model = None
                if processor:
                    del processor
                    processor = None

                ModelFactory.cleanup_models()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                print("[CLEANUP] Transcription model unloaded before translation")

                socketio.emit(
                    "file_progress",
                    {"session_id": session_id, "percent": 65, "status": "Loading translation model..."},
                )

                # Load translation model (use configured model or default)
                translation_model_id = ft_config.get("translation_model", "facebook/nllb-200-distilled-600M")
                translation_model, translation_tokenizer = load_translation_model(ft_use_gpu, model_id=translation_model_id)

                # Create progress callback for translation
                def translation_progress(percent, status):
                    socketio.emit(
                        "file_progress",
                        {"session_id": session_id, "percent": percent, "status": status},
                    )

                # Get generation params from live_translation config (shared settings)
                ft_gen_params = config.get("live_translation", {}).get("generation_params", {})
                ft_context_window = config.get("live_translation", {}).get("context_window", 1)

                translated_segments = translate_segments(
                    segments, source_lang, translate_to,
                    translation_model, translation_tokenizer,
                    progress_callback=translation_progress,
                    generation_params=ft_gen_params,
                    context_window=ft_context_window
                )

                # Cleanup translation model
                if translation_model:
                    del translation_model
                    translation_model = None
                if translation_tokenizer:
                    del translation_tokenizer
                    translation_tokenizer = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            target_language_name = TRANSLATION_LANGUAGES.get(translate_to, translate_to)
            print(f"[INFO] Translation complete: {len(translated_segments)} segments translated to {target_language_name}")
            print("[CLEANUP] Translation model unloaded")

        # Send segments for client-side formatting
        socketio.emit(
            "file_progress",
            {"session_id": session_id, "percent": 95, "status": "Preparing results..."},
        )

        # Build completion data
        completion_data = {
            "session_id": session_id,
            "segments": segments,  # Original transcription segments
            "format": output_format,
            "duration": segments[-1]["end"] if segments else 0,
            "total_segments": len(segments),
            "filename": filename,
            "source_language": ft_language,
        }

        # Add translation data if available
        if translated_segments:
            completion_data["translated_segments"] = translated_segments
            completion_data["target_language"] = translate_to
            completion_data["target_language_name"] = target_language_name

        # Send completion with segments array for client-side format switching
        socketio.emit("file_complete", completion_data)

    except Exception as e:
        socketio.emit("file_error", {"session_id": session_id, "error": str(e)})
    finally:
        # Cleanup resources
        try:
            # Clean up models
            if model:
                del model
            if processor:
                del processor
            if translation_model:
                del translation_model
            if translation_tokenizer:
                del translation_tokenizer

            # Clear model cache to prevent blocking live transcription
            ModelFactory.cleanup_models()
            print("[CLEANUP] Model cache cleared after file transcription")

            # GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("[CLEANUP] GPU cache cleared")

            gc.collect()

            # Clean up temp files
            if os.path.exists(file_path):
                os.unlink(file_path)
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
        except Exception as cleanup_error:
            print(f"[WARNING] Error during cleanup: {cleanup_error}")


@app.route("/api/restart", methods=["POST"])
def restart_transcription():
    """API endpoint to restart the transcription process only (not the whole server).

    Use this for:
    - Restarting transcription after model changes
    - Resetting audio capture without losing web server connection

    For full server restart (config changes, port changes), use /api/server/restart instead.
    """
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global transcription_process
    try:
        if transcription_process is not None and transcription_process.is_alive():
            # Send stop command
            control_queue.put({"command": "stop"})
            sleep(2)  # Wait for graceful shutdown

            # Terminate if still alive
            if transcription_process.is_alive():
                transcription_process.terminate()
                transcription_process.join(timeout=5)

                # Force kill if still alive
                if transcription_process.is_alive():
                    transcription_process.kill()
                    transcription_process.join(timeout=2)

            # Start new process
            transcription_process = multiprocessing.Process(
                target=thread1_function,
                args=(transcription_state, control_queue, config_queue,
                      calibration_state, calibration_data_shared, calibration_step1_data,
                      audio_stream_queue)
            )
            transcription_process.start()

            # CRITICAL: Update global reference for signal handler
            globals()["thread1"] = transcription_process

            control_queue.put({"command": "start"})

            return jsonify(
                {
                    "success": True,
                    "message": "Transcription process restarted successfully!",
                }
            )
        else:
            return jsonify(
                {"success": False, "error": "Transcription process is not running"}
            ), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/server/restart", methods=["POST"])
def restart_server():
    """API endpoint to restart the entire server (full application restart).

    Use this for:
    - Applying config changes that require full restart (port, host, etc.)
    - Loading new dependencies or major updates

    For just restarting transcription (model changes), use /api/restart instead.
    """
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        import sys
        import subprocess

        # Return success response before restarting
        response = jsonify(
            {
                "success": True,
                "message": "Server is restarting... Please wait 10-15 seconds and refresh the page.",
            }
        )

        # Schedule restart after response is sent
        def do_restart():
            sleep(1)  # Wait for response to be sent
            print("[RESTART] Server restart requested via API")

            if sys.platform.startswith('win'):
                # Windows: use restart_server.bat to cleanly stop and restart
                script_dir = APP_DIR
                restart_bat = os.path.join(script_dir, "restart_server.bat")
                if os.path.exists(restart_bat):
                    print("[RESTART] Calling restart_server.bat...")
                    subprocess.Popen(
                        ["cmd.exe", "/c", restart_bat],
                        cwd=script_dir,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    )
                else:
                    # Fallback: spawn new process directly
                    print("[RESTART] restart_server.bat not found, spawning directly...")
                    python = sys.executable
                    subprocess.Popen(
                        [python] + sys.argv,
                        cwd=script_dir,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    )
                sleep(1)
                os._exit(0)
                return

            # Use systemctl restart if running as a systemd service
            # This is atomic - systemd handles stop+start without race conditions
            for service_name in ["stt-watchdog", "stt-server", "stt"]:
                result = subprocess.run(
                    ["systemctl", "is-active", "--quiet", service_name],
                    capture_output=True,
                )
                if result.returncode == 0:
                    print(f"[RESTART] Restarting via systemctl ({service_name})...")
                    subprocess.Popen(
                        ["systemctl", "restart", service_name],
                        start_new_session=True,
                    )
                    return

            # Fallback: not running under systemd, use restart_server.sh or execv
            script_dir = APP_DIR
            restart_script = os.path.join(script_dir, "restart_server.sh")

            if os.path.exists(restart_script):
                print("[RESTART] Calling restart_server.sh...")
                subprocess.Popen(
                    ["bash", restart_script],
                    cwd=script_dir,
                    start_new_session=True,
                )
                sleep(2)
                os._exit(0)
            else:
                print("[RESTART] Falling back to execv...")
                python = sys.executable
                os.execv(python, [python] + sys.argv)

        # Start restart in background thread
        import threading

        restart_thread = threading.Thread(target=do_restart, daemon=True)
        restart_thread.start()

        return response

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/disk-space", methods=["GET"])
def get_disk_space():
    """API endpoint to get disk space information (cross-platform)"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        # Get the current working directory path
        current_path = APP_DIR

        # Get disk usage statistics using shutil (works on both Windows and Linux)
        disk_usage = shutil.disk_usage(current_path)

        # Calculate values in bytes
        total_bytes = disk_usage.total
        used_bytes = disk_usage.used
        free_bytes = disk_usage.free

        # Calculate percentage used
        percent_used = (used_bytes / total_bytes * 100) if total_bytes > 0 else 0

        # Helper function to format bytes to human-readable format
        def format_bytes(bytes_value):
            """Convert bytes to human-readable format"""
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if bytes_value < 1024.0:
                    return f"{bytes_value:.2f} {unit}"
                bytes_value /= 1024.0
            return f"{bytes_value:.2f} PB"

        return jsonify(
            {
                "success": True,
                "disk_space": {
                    "total": total_bytes,
                    "used": used_bytes,
                    "free": free_bytes,
                    "percent_used": round(percent_used, 2),
                    "total_formatted": format_bytes(total_bytes),
                    "used_formatted": format_bytes(used_bytes),
                    "free_formatted": format_bytes(free_bytes),
                    "path": current_path,
                },
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/endpoints", methods=["GET"])
def get_api_endpoints():
    """API endpoint to list all available API endpoints (auto-generated from Flask routes)"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    try:
        endpoints = []
        for rule in app.url_map.iter_rules():
            # Skip static files and internal endpoints
            if rule.endpoint == "static" or rule.rule.startswith("/static"):
                continue
            # Get methods, excluding HEAD and OPTIONS
            methods = list(rule.methods - {"HEAD", "OPTIONS"})
            if methods:  # Only include if there are actual methods
                # Get description and examples from docstring
                description = ""
                examples = []
                view_func = app.view_functions.get(rule.endpoint)
                auth_required = False
                if view_func and view_func.__doc__:
                    doc_lines = view_func.__doc__.strip().split('\n')
                    # First line is description
                    description = doc_lines[0].strip()
                    # Lines starting with "Example:" are examples
                    for line in doc_lines[1:]:
                        line = line.strip()
                        if line.startswith("Example:"):
                            examples.append(line[8:].strip())
                if view_func:
                    try:
                        import inspect
                        src = inspect.getsource(view_func)
                        auth_required = "check_ip_whitelist()" in src
                    except Exception:
                        pass

                endpoints.append({
                    "path": rule.rule,
                    "methods": sorted(methods),
                    "description": description,
                    "examples": examples,
                    "auth_required": auth_required,
                })

        # Sort by path for consistent ordering
        endpoints.sort(key=lambda x: x["path"])

        return jsonify({"success": True, "endpoints": endpoints})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-manager/browse", methods=["GET"])
def browse_files():
    """API endpoint to browse files and directories"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    try:
        # Get query parameters
        path = request.args.get("path", APP_DIR)
        show_hidden = request.args.get("show_hidden", "false").lower() == "true"

        # Normalize and validate path
        path = os.path.abspath(path)

        # Security check: ensure path is not trying to escape
        if not os.path.exists(path):
            return jsonify({"success": False, "error": "Path does not exist"}), 404

        if not os.path.isdir(path):
            return jsonify({"success": False, "error": "Path is not a directory"}), 400

        # Get parent directory
        parent_dir = os.path.dirname(path) if path != os.path.dirname(path) else None

        # Get hidden items from config
        hidden_items = config.get("file_manager", {}).get("hidden_items", [])
        working_dir = APP_DIR

        # List directory contents
        items = []
        try:
            for item_name in os.listdir(path):
                item_path = os.path.join(path, item_name)

                # Skip dotfiles unless show_hidden is True
                if not show_hidden and item_name.startswith("."):
                    continue

                # Skip __pycache__ directories unless show_hidden is True
                if not show_hidden and item_name == "__pycache__":
                    continue

                # Skip items in hidden list unless show_hidden is True
                if not show_hidden:
                    try:
                        rel_path = os.path.relpath(item_path, working_dir)
                    except ValueError:
                        rel_path = item_path

                    if rel_path in hidden_items:
                        continue

                try:
                    stat_info = os.stat(item_path)
                    is_dir = os.path.isdir(item_path)

                    # Format file size
                    size = stat_info.st_size
                    size_formatted = format_file_size(size) if not is_dir else "-"

                    # Format modification time
                    modified = datetime.fromtimestamp(stat_info.st_mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    # Check if item is in hidden list
                    try:
                        rel_path = os.path.relpath(item_path, working_dir)
                    except ValueError:
                        rel_path = item_path
                    is_hidden = rel_path in hidden_items

                    items.append(
                        {
                            "name": item_name,
                            "path": item_path,
                            "type": "directory" if is_dir else "file",
                            "size": size if not is_dir else 0,
                            "size_formatted": size_formatted,
                            "modified": modified,
                            "extension": os.path.splitext(item_name)[1]
                            if not is_dir
                            else "",
                            "is_hidden": is_hidden,
                        }
                    )
                except (PermissionError, OSError):
                    # Skip items we can't access
                    continue
        except PermissionError:
            return jsonify({"success": False, "error": "Permission denied"}), 403

        # Sort: directories first, then files, alphabetically
        items.sort(key=lambda x: (x["type"] != "directory", x["name"].lower()))

        return jsonify(
            {
                "success": True,
                "current_path": path,
                "parent_path": parent_dir,
                "items": items,
                "show_hidden": show_hidden,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def format_file_size(bytes_value):
    """Convert bytes to human-readable format"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


@app.route("/api/file-manager/delete", methods=["POST"])
def delete_file():
    """API endpoint to delete a file or directory"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    try:
        data = request.get_json()
        path = data.get("path")

        if not path:
            return jsonify({"success": False, "error": "Path is required"}), 400

        # Normalize path
        path = os.path.abspath(path)

        # Security check: prevent deletion of critical files
        if path == APP_DIR or path in [
            os.path.abspath("speech_to_text.py"),
            CONFIG_FILE,
        ]:
            return jsonify(
                {
                    "success": False,
                    "error": "Cannot delete critical files or current directory",
                }
            ), 403

        if not os.path.exists(path):
            return jsonify({"success": False, "error": "Path does not exist"}), 404

        # Delete file or directory
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

        return jsonify({"success": True, "message": "Deleted successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-manager/rename", methods=["POST"])
def rename_file():
    """API endpoint to rename a file or directory"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    try:
        data = request.get_json()
        old_path = data.get("old_path")
        new_name = data.get("new_name")

        if not old_path or not new_name:
            return jsonify(
                {"success": False, "error": "Old path and new name are required"}
            ), 400

        # Normalize old path
        old_path = os.path.abspath(old_path)

        if not os.path.exists(old_path):
            return jsonify({"success": False, "error": "Path does not exist"}), 404

        # Create new path
        parent_dir = os.path.dirname(old_path)
        new_path = os.path.join(parent_dir, new_name)

        if os.path.exists(new_path):
            return jsonify(
                {
                    "success": False,
                    "error": "A file or directory with that name already exists",
                }
            ), 400

        # Rename
        os.rename(old_path, new_path)

        return jsonify(
            {"success": True, "message": "Renamed successfully", "new_path": new_path}
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-manager/create-folder", methods=["POST"])
def create_folder():
    """API endpoint to create a new folder"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    try:
        data = request.get_json()
        parent_path = data.get("parent_path")
        folder_name = data.get("folder_name")

        if not parent_path or not folder_name:
            return jsonify(
                {"success": False, "error": "Parent path and folder name are required"}
            ), 400

        # Normalize parent path
        parent_path = os.path.abspath(parent_path)

        if not os.path.exists(parent_path):
            return jsonify(
                {"success": False, "error": "Parent directory does not exist"}
            ), 404

        if not os.path.isdir(parent_path):
            return jsonify(
                {"success": False, "error": "Parent path is not a directory"}
            ), 400

        # Create new folder path
        new_folder_path = os.path.join(parent_path, folder_name)

        if os.path.exists(new_folder_path):
            return jsonify(
                {
                    "success": False,
                    "error": "A file or directory with that name already exists",
                }
            ), 400

        # Create folder
        os.makedirs(new_folder_path)

        return jsonify(
            {
                "success": True,
                "message": "Folder created successfully",
                "path": new_folder_path,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-manager/hidden-items", methods=["GET"])
def get_hidden_items():
    """API endpoint to get list of hidden files/folders"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    try:
        hidden_items = config.get("file_manager", {}).get("hidden_items", [])
        return jsonify({"success": True, "hidden_items": hidden_items})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-manager/hide", methods=["POST"])
def hide_item():
    """API endpoint to hide a specific file or folder"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    try:
        global config
        data = request.get_json()
        path = data.get("path")

        if not path:
            return jsonify({"success": False, "error": "Path is required"}), 400

        # Normalize path to be relative to working directory
        abs_path = os.path.abspath(path)
        working_dir = APP_DIR

        # Get relative path
        try:
            rel_path = os.path.relpath(abs_path, working_dir)
        except ValueError:
            # On Windows, relpath fails if paths are on different drives
            rel_path = abs_path

        # Get current hidden items
        if "file_manager" not in config:
            config["file_manager"] = {}

        hidden_items = config["file_manager"].get("hidden_items", [])

        # Add if not already hidden
        if rel_path not in hidden_items:
            hidden_items.append(rel_path)
            config["file_manager"]["hidden_items"] = hidden_items
            save_config(config)

        return jsonify(
            {
                "success": True,
                "message": "Item hidden successfully",
                "hidden_items": hidden_items,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-manager/unhide", methods=["POST"])
def unhide_item():
    """API endpoint to unhide a specific file or folder"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    try:
        global config
        data = request.get_json()
        path = data.get("path")

        if not path:
            return jsonify({"success": False, "error": "Path is required"}), 400

        # Normalize path to be relative to working directory
        abs_path = os.path.abspath(path)
        working_dir = APP_DIR

        # Get relative path
        try:
            rel_path = os.path.relpath(abs_path, working_dir)
        except ValueError:
            rel_path = abs_path

        # Get current hidden items
        if "file_manager" not in config:
            config["file_manager"] = {}

        hidden_items = config["file_manager"].get("hidden_items", [])

        # Remove if hidden
        if rel_path in hidden_items:
            hidden_items.remove(rel_path)
            config["file_manager"]["hidden_items"] = hidden_items
            save_config(config)

        return jsonify(
            {
                "success": True,
                "message": "Item unhidden successfully",
                "hidden_items": hidden_items,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-manager/download", methods=["GET"])
def download_file():
    """API endpoint to download a file"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    try:
        path = request.args.get("path")

        if not path:
            return jsonify({"success": False, "error": "Path is required"}), 400

        # Security: Ensure the path is within the working directory
        abs_path = os.path.abspath(path)
        working_dir = APP_DIR

        if not abs_path.startswith(working_dir):
            return jsonify({"success": False, "error": "Access denied"}), 403

        # Check if file exists
        if not os.path.exists(abs_path):
            return jsonify({"success": False, "error": "File not found"}), 404

        # Check if it's a file (not a directory)
        if not os.path.isfile(abs_path):
            return jsonify({"success": False, "error": "Not a file"}), 400

        # Send the file
        directory = os.path.dirname(abs_path)
        filename = os.path.basename(abs_path)
        return send_from_directory(directory, filename, as_attachment=True)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# File Mover Endpoints
@app.route("/api/file-mover/status", methods=["GET"])
def get_file_mover_status():
    """Get file mover status and configuration"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        mover_config = config.get("file_manager", {}).get("file_mover", {})
        return jsonify({
            "success": True,
            "config": mover_config,
            "runtime": get_file_mover_runtime(),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-mover/configure", methods=["POST"])
def configure_file_mover():
    """Update file mover configuration"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        data = request.get_json()

        if "file_manager" not in config:
            config["file_manager"] = {}
        if "file_mover" not in config["file_manager"]:
            config["file_manager"]["file_mover"] = {}

        # Update configuration
        mover_config = config["file_manager"]["file_mover"]

        if "enabled" in data:
            mover_config["enabled"] = bool(data["enabled"])
        if "move_on_transcription_stop" in data:
            mover_config["move_on_transcription_stop"] = bool(
                data["move_on_transcription_stop"]
            )
        if "destination_path" in data:
            mover_config["destination_path"] = data["destination_path"]
        if "smb_username" in data:
            mover_config["smb_username"] = data["smb_username"]
        if "smb_password" in data:
            mover_config["smb_password"] = data["smb_password"]
        if "smb_domain" in data:
            mover_config["smb_domain"] = data["smb_domain"]
        if "source_patterns" in data:
            mover_config["source_patterns"] = data["source_patterns"]
        if "delete_source" in data:
            mover_config["delete_source"] = bool(data["delete_source"])
        if "preserve_structure" in data:
            mover_config["preserve_structure"] = bool(data["preserve_structure"])
        if "retry_on_failure" in data:
            mover_config["retry_on_failure"] = bool(data["retry_on_failure"])

        save_config(config)

        return jsonify({"success": True, "message": "Configuration updated"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-mover/test", methods=["POST"])
def test_file_mover_connection():
    """Test connection to destination"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        from file_mover import test_destination_accessible

        data = request.get_json()
        dest_path = data.get("destination_path", "")
        username = data.get("smb_username", "")
        password = data.get("smb_password", "")
        domain = data.get("smb_domain", "")

        if not dest_path:
            return jsonify(
                {"success": False, "error": "No destination path provided"}
            ), 400

        accessible = test_destination_accessible(dest_path, username, password, domain)

        if accessible:
            return jsonify({"success": True, "message": "Destination is accessible"})
        else:
            return jsonify({"success": False, "error": "Destination is not accessible"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-mover/trigger", methods=["POST"])
def trigger_file_mover_endpoint():
    """API endpoint to manually trigger file mover check"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        # Use the core file move function for consistency
        set_file_mover_running("manual")
        result = execute_file_move(lambda: config)
        set_file_mover_result("manual", result)

        if not result['success']:
            return jsonify({
                "success": False,
                "error": result.get('message', 'Unknown error'),
                "errors": result.get('errors', [])
            }), 400

        return jsonify({
            "success": True,
            "moved": result['moved'],
            "failed": result['failed'],
            "errors": result.get('errors', []),
            "message": result['message'],
            "delete_source": result.get('delete_source', True)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-mover/browse-remote", methods=["POST"])
def browse_remote_destination():
    """Browse files in the remote SMB destination"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        from file_mover import test_destination_accessible

        data = request.get_json()
        dest_path = data.get("destination_path", "").strip()
        username = data.get("smb_username", "").strip()
        password = data.get("smb_password", "")
        domain = data.get("smb_domain", "").strip()
        subpath = data.get("subpath", "").strip()

        if not dest_path:
            return jsonify(
                {"success": False, "error": "No destination path specified"}
            ), 400

        # Build full path with subpath
        if subpath:
            full_path = os.path.join(dest_path, subpath)
        else:
            full_path = dest_path

        # Test/mount the destination
        if not test_destination_accessible(dest_path, username, password, domain):
            return jsonify(
                {"success": False, "error": "Cannot access remote destination"}
            ), 400

        # On Linux, we need to use the mount point
        if platform == "linux":
            from file_mover import is_smb_path

            if is_smb_path(dest_path):
                # Extract server and share to build mount point path
                unc_path = dest_path.replace("\\", "/")
                if not unc_path.startswith("//"):
                    unc_path = "//" + unc_path
                parts = unc_path.replace("//", "").split("/")
                if len(parts) >= 2:
                    server = parts[0]
                    share = parts[1]
                    if "darwin" in platform:
                        mount_point = f"/Volumes/{share}"
                    else:
                        mount_point = f"/mnt/{server}_{share}"

                    # Replace dest_path with mount point
                    remaining_path = "/".join(parts[2:]) if len(parts) > 2 else ""
                    if subpath:
                        full_path = os.path.join(mount_point, remaining_path, subpath)
                    elif remaining_path:
                        full_path = os.path.join(mount_point, remaining_path)
                    else:
                        full_path = mount_point

        # List files in the remote path
        items = []
        try:
            for item_name in os.listdir(full_path):
                item_path = os.path.join(full_path, item_name)

                try:
                    stat_info = os.stat(item_path)
                    is_dir = os.path.isdir(item_path)

                    size = stat_info.st_size
                    size_formatted = format_file_size(size) if not is_dir else "-"
                    modified = datetime.fromtimestamp(stat_info.st_mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    items.append(
                        {
                            "name": item_name,
                            "path": os.path.join(subpath, item_name)
                            if subpath
                            else item_name,
                            "type": "directory" if is_dir else "file",
                            "size": size if not is_dir else 0,
                            "size_formatted": size_formatted,
                            "modified": modified,
                            "extension": os.path.splitext(item_name)[1]
                            if not is_dir
                            else "",
                        }
                    )
                except (PermissionError, OSError):
                    continue
        except PermissionError:
            return jsonify({"success": False, "error": "Permission denied"}), 403

        # Sort: directories first, then files
        items.sort(key=lambda x: (x["type"] != "directory", x["name"].lower()))

        # Calculate parent path
        parent_path = os.path.dirname(subpath) if subpath else ""

        return jsonify(
            {
                "success": True,
                "current_path": subpath,
                "parent_path": parent_path,
                "items": items,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/transcription/start", methods=["POST"])
def start_transcription():
    """Start the transcription process
    Example: POST /api/transcription/start"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global transcription_process, transcription_state
    try:
        with _transcription_start_lock:
            worker_dead = transcription_process is None or not transcription_process.is_alive()

            if transcription_state["running"]:
                if worker_dead:
                    # Worker crashed mid-run and never reset the state — recover
                    # instead of telling the user "already running" forever
                    print("[START] State says running but worker is dead — resetting state")
                    transcription_state["running"] = False
                    transcription_state["status"] = "stopped"
                else:
                    return jsonify(
                        {"success": False, "error": "Transcription is already running"}
                    ), 400

            # Don't start if still stopping (unless the worker is gone)
            if transcription_state["status"] == "stopping" and not worker_dead:
                return jsonify(
                    {"success": False, "error": "Transcription is still stopping, please wait"}
                ), 400

            # Ensure we have a valid worker process
            # Worker stays alive between Start/Stop cycles, so we usually just reuse it
            if worker_dead:
                # Worker doesn't exist or crashed - create a new one
                # This should only happen on first start or if worker unexpectedly died
                print("[START] Worker process not running, creating new worker...")
                transcription_process = multiprocessing.Process(
                    target=thread1_function,
                    args=(transcription_state, control_queue, config_queue,
                          calibration_state, calibration_data_shared, calibration_step1_data,
                          audio_stream_queue)
                )
                transcription_process.start()
                globals()["thread1"] = transcription_process
            else:
                # Worker is alive, just reuse it (it's waiting in idle loop)
                print(f"[START] Reusing existing worker process PID={transcription_process.pid}")

        # Send start command through queue
        control_queue.put({"command": "start"})

        # Update state - don't set running=True yet, worker will do that after initialization
        transcription_state["status"] = "starting"
        transcription_state["message"] = (
            "Initializing audio interface and loading model..."
        )
        transcription_state["error"] = None  # Clear any previous errors

        # Warm up the translation model so the first translated segment doesn't
        # pay the load cost. Remote setups preload on Machine B; otherwise, for a
        # local seq2seq model (not Whisper-based translation), preload here.
        trans_cfg = config.get("live_translation", {})
        remote_cfg = trans_cfg.get("remote", {})
        remote_active = bool(remote_cfg.get("enabled") and remote_cfg.get("endpoint"))
        if remote_active:
            def _notify_remote_preload():
                try:
                    ep = _get_remote_endpoint()
                    if not ep:
                        return
                    import requests as _req
                    r = _req.post(ep + "/api/translate/preload", timeout=10)
                    print(f"[START] Remote translation preload: {r.json()}")
                except Exception as e:
                    print(f"[START] Remote translation preload failed: {e}")
            import threading
            threading.Thread(target=_notify_remote_preload, daemon=True).start()
        elif trans_cfg.get("enabled") and trans_cfg.get("translation_method", "nllb") not in (
            "whisper_translate", "whisper_forced_lang"
        ):
            def _preload_local_translation():
                try:
                    use_gpu = trans_cfg.get("use_gpu", True)
                    model_id = trans_cfg.get("translation_model")
                    get_live_translation_model(use_gpu, model_id=model_id)
                    print("[START] Local translation model preloaded")
                except Exception as e:
                    print(f"[START] Local translation preload failed: {e}")
            import threading
            threading.Thread(target=_preload_local_translation, daemon=True).start()

        return jsonify(
            {
                "success": True,
                "message": "Transcription starting...",
                "state": dict(transcription_state),  # Convert DictProxy to dict
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/transcription/stop", methods=["POST"])
def stop_transcription():
    """Stop the transcription process
    Example: POST /api/transcription/stop"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global transcription_state, transcription_process
    try:
        if not transcription_state["running"] and transcription_state["status"] != "starting":
            return jsonify(
                {"success": False, "error": "Transcription is not running"}
            ), 400

        # Send stop command through queue
        control_queue.put({"command": "stop"})

        # Update state
        transcription_state["running"] = False
        transcription_state["status"] = "stopping"
        transcription_state["message"] = (
            "Stopping transcription, unloading model, closing connections..."
        )
        transcription_state["loaded_model"] = ""  # Clear loaded model name
        transcription_state["live_text"] = ""  # Clear in-progress text
        transcription_state["live_start"] = 0
        transcription_state["live_end"] = 0

        # Clear database cache immediately so clients get empty data
        with _cache_lock:
            _db_cache["last_entries"] = []
            _db_cache["last_fetch_time"] = 0

        # Unload Live Translation model synchronously to free GPU memory
        # Do this BEFORE starting cleanup thread to avoid CUDA conflicts
        if is_live_translation_model_loaded():
            print("[STOP] Unloading Live Translation model...")
            unload_live_translation_model()
            print("[STOP] Live Translation model unloaded")

        # Tell remote Machine B to unload its translation model too
        remote_cfg = config.get("live_translation", {}).get("remote", {})
        if remote_cfg.get("enabled") and remote_cfg.get("endpoint"):
            def _notify_remote_unload():
                try:
                    ep = _get_remote_endpoint()
                    if not ep:
                        return
                    import requests as _req
                    r = _req.post(ep + "/api/translate/unload", timeout=10)
                    print(f"[STOP] Remote translation unload: {r.json()}")
                except Exception as e:
                    print(f"[STOP] Remote translation unload failed: {e}")
            import threading
            threading.Thread(target=_notify_remote_unload, daemon=True).start()

        if is_tts_model_loaded():
            print("[STOP] Unloading TTS model...")
            unload_tts_model()
            print("[STOP] TTS model unloaded")

        # Background cleanup to send unload command and update status
        # NOTE: We keep the worker process alive! This avoids CUDA fork issues on subsequent starts.
        # The worker will unload its models (Whisper, VAD) when it receives the unload command,
        # which releases GPU memory. The worker stays in its idle loop, ready for the next Start.
        def cleanup_process():
            """Background cleanup to send unload command and update status"""
            import time
            _log_path = os.path.join(APP_DIR, "server.log")
            def log(msg):
                with open(_log_path, "a") as f:
                    f.write(msg + "\n")
                    f.flush()

            log("[STOP-CLEANUP] Thread started")

            time.sleep(2)  # Wait for graceful shutdown of transcription loop

            # Send unload command to worker to release GPU memory
            # Worker stays alive to avoid CUDA fork issues on subsequent starts
            log("[STOP-CLEANUP] Sending unload command to worker...")
            try:
                control_queue.put({"command": "unload"})
                log("[STOP-CLEANUP] Unload command sent to worker")
            except Exception as e:
                log(f"[STOP-CLEANUP] Error sending unload command: {e}")

            with _transcription_state_lock:
                if transcription_state["status"] == "stopping":
                    transcription_state["status"] = "stopped"
                    transcription_state["message"] = "Transcription stopped"

        # Run cleanup in background thread
        import threading
        _server_log = os.path.join(APP_DIR, "server.log")
        # Write directly to log file since stdout might be buffered
        with open(_server_log, "a") as logf:
            logf.write(f"[STOP] Creating cleanup thread, transcription_process={transcription_process}\n")
            logf.flush()
        cleanup_thread = threading.Thread(target=cleanup_process, daemon=True)
        cleanup_thread.start()
        with open(_server_log, "a") as logf:
            logf.write("[STOP] Cleanup thread started\n")
            logf.flush()

        return jsonify(
            {
                "success": True,
                "message": "Transcription stopping...",
                "state": dict(transcription_state),  # Convert DictProxy to dict
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/transcription/status", methods=["GET"])
def get_transcription_status():
    """API endpoint to get transcription status"""
    return jsonify(
        {
            "success": True,
            "state": dict(
                transcription_state
            ),  # Convert DictProxy to regular dict for JSON serialization
        }
    )


@app.route("/api/transcription/force-reset", methods=["POST"])
def force_reset_transcription():
    """API endpoint to force reset transcription state (emergency use)"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global transcription_state, transcription_process

    try:
        print("[FORCE RESET] Forcing transcription state reset...")

        # Force kill the transcription process if it exists
        if transcription_process is not None and transcription_process.is_alive():
            print("[FORCE RESET] Terminating transcription process...")
            try:
                transcription_process.terminate()
                transcription_process.join(timeout=3)
            except (OSError, ProcessLookupError):
                pass

            # If still alive, force kill
            if transcription_process.is_alive():
                print("[FORCE RESET] Force killing transcription process...")
                try:
                    transcription_process.kill()
                    transcription_process.join(timeout=2)
                except (OSError, ProcessLookupError):
                    pass

        # Clear the control queue
        import queue as _queue_mod
        while not control_queue.empty():
            try:
                control_queue.get_nowait()
            except _queue_mod.Empty:
                break

        # Reset transcription state
        transcription_state["running"] = False
        transcription_state["status"] = "stopped"
        transcription_state["message"] = "Transcription forcefully reset"
        transcription_state["loaded_model"] = ""
        transcription_state["live_text"] = ""
        transcription_state["live_start"] = 0
        transcription_state["live_end"] = 0
        transcription_state["db_name"] = None
        transcription_state["session_id"] = None

        # Clear database cache immediately so clients get empty data
        with _cache_lock:
            _db_cache["last_entries"] = []
            _db_cache["last_fetch_time"] = 0

        # Restart the transcription process
        transcription_process = multiprocessing.Process(
            target=thread1_function,
            args=(transcription_state, control_queue, config_queue,
                  calibration_state, calibration_data_shared, calibration_step1_data,
                  audio_stream_queue)
        )
        transcription_process.start()

        # Update global reference for signal handler
        globals()["thread1"] = transcription_process

        print("[FORCE RESET] Transcription process reset complete")

        return jsonify(
            {
                "success": True,
                "message": "Transcription state forcefully reset. You can now start transcription again.",
                "state": dict(transcription_state),  # Convert DictProxy to dict for JSON serialization
            }
        )
    except Exception as e:
        print(f"[FORCE RESET ERROR] {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Real-Time Corrections API Endpoints
# =============================================================================


@app.route("/api/transcription/correct", methods=["POST"])
def correct_transcription():
    """Correct a transcription segment's text"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    segment_id = data.get("segment_id")
    new_text = data.get("new_text", "").strip()
    correction_type = data.get("correction_type", "manual")

    if segment_id is None or not new_text:
        return jsonify({"success": False, "error": "segment_id and new_text are required"}), 400

    current_db_name = transcription_state.get("db_name")
    if not current_db_name or not os.path.exists(current_db_name):
        return jsonify({"success": False, "error": "No active database"}), 400

    try:
        with sqlite3.connect(current_db_name) as conn:
            cursor = conn.cursor()
            # Store original text before correction (only on first correction)
            cursor.execute(
                """UPDATE transcriptions
                   SET original_text = COALESCE(original_text, text),
                       text = ?,
                       corrected_by = ?
                   WHERE id = ?""",
                (new_text, correction_type, segment_id),
            )
            conn.commit()

            if cursor.rowcount == 0:
                return jsonify({"success": False, "error": "Segment not found"}), 404

        # Invalidate DB cache so next emit picks up the change
        with _cache_lock:
            _db_cache["last_entries"] = []
            _db_cache["last_fetch_time"] = 0

        # Invalidate translation cache for this segment
        cache = get_translation_cache()
        cache.invalidate(segment_id)

        # Re-translate if translation is active
        translated_text = None
        trans_config = config.get("live_translation", {})
        if trans_config.get("enabled", False):
            target_lang = trans_config.get("target_language", "en")
            source_lang = trans_config.get("source_language", "auto")
            if source_lang == "auto":
                source_lang = config.get("audio", {}).get("language", "en")
                if source_lang == "auto":
                    source_lang = "en"
            translated_text = translate_live_text(new_text, source_lang, target_lang)
            if translated_text:
                cache.set(segment_id, new_text, translated_text, target_lang)

        # Emit correction event to all clients
        socketio.emit("correction_applied", {
            "segment_id": segment_id,
            "new_text": new_text,
            "corrected_by": correction_type,
            "translated_text": translated_text,
        })

        return jsonify({"success": True, "segment_id": segment_id, "new_text": new_text, "translated_text": translated_text})

    except Exception as e:
        print(f"[CORRECTION ERROR] {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/transcription/review-queue", methods=["GET"])
def get_review_queue():
    """Get segments that need review (low confidence)"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    current_db_name = transcription_state.get("db_name")
    if not current_db_name or not os.path.exists(current_db_name):
        return jsonify({"success": True, "segments": []})

    try:
        with sqlite3.connect(current_db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT id, timestamp, text, COALESCE(start_time, 0), COALESCE(end_time, 0), confidence
                   FROM transcriptions
                   WHERE needs_review = 1 AND COALESCE(denied, 0) = 0
                   ORDER BY id DESC
                   LIMIT 50"""
            )
            rows = cursor.fetchall()

        segments = [
            {"id": r[0], "timestamp": r[1], "text": r[2], "start": r[3], "end": r[4], "confidence": r[5]}
            for r in rows
        ]
        return jsonify({"success": True, "segments": segments})

    except Exception as e:
        print(f"[REVIEW QUEUE ERROR] {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/transcription/mark-reviewed", methods=["POST"])
def mark_reviewed():
    """Mark segments as reviewed (remove from review queue)"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    segment_ids = data.get("segment_ids", [])
    if not segment_ids:
        return jsonify({"success": False, "error": "segment_ids required"}), 400

    current_db_name = transcription_state.get("db_name")
    if not current_db_name or not os.path.exists(current_db_name):
        return jsonify({"success": False, "error": "No active database"}), 400

    try:
        with sqlite3.connect(current_db_name) as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in segment_ids)
            cursor.execute(
                f"UPDATE transcriptions SET needs_review = 0 WHERE id IN ({placeholders})",
                segment_ids,
            )
            conn.commit()

        # Invalidate DB cache
        with _cache_lock:
            _db_cache["last_entries"] = []
            _db_cache["last_fetch_time"] = 0

        return jsonify({"success": True, "updated": len(segment_ids)})

    except Exception as e:
        print(f"[MARK REVIEWED ERROR] {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Custom Dictionary API Endpoints
# =============================================================================

_dictionary_cache = None
_dictionary_mtime = 0


def load_custom_dictionary():
    """Load custom dictionary from JSON file, with caching"""
    global _dictionary_cache, _dictionary_mtime

    dict_file = config.get("custom_dictionary", {}).get("file", "custom_dictionary.json")
    if not os.path.isabs(dict_file):
        dict_file = os.path.join(APP_DIR, dict_file)

    try:
        if os.path.exists(dict_file):
            mtime = os.path.getmtime(dict_file)
            if _dictionary_cache is not None and mtime == _dictionary_mtime:
                return _dictionary_cache
            with open(dict_file, "r", encoding="utf-8") as f:
                import json as _json
                _dictionary_cache = _json.load(f)
                _dictionary_mtime = mtime
                return _dictionary_cache
        else:
            # Create default dictionary file if it doesn't exist
            default_dict = {"glossary": {}}
            import json as _json
            with open(dict_file, "w", encoding="utf-8") as f:
                _json.dump(default_dict, f, indent=2, ensure_ascii=False)
            _dictionary_cache = default_dict
            _dictionary_mtime = os.path.getmtime(dict_file)
            print(f"[DICTIONARY] Created default dictionary: {dict_file}")
            return default_dict
    except Exception as e:
        print(f"[DICTIONARY] Error loading dictionary: {e}")

    return {"glossary": {}}


def save_custom_dictionary(data):
    """Save custom dictionary to JSON file"""
    global _dictionary_cache, _dictionary_mtime

    dict_file = config.get("custom_dictionary", {}).get("file", "custom_dictionary.json")
    if not os.path.isabs(dict_file):
        dict_file = os.path.join(APP_DIR, dict_file)

    try:
        import json as _json
        with open(dict_file, "w", encoding="utf-8") as f:
            _json.dump(data, f, indent=2, ensure_ascii=False)
        _dictionary_cache = data
        _dictionary_mtime = os.path.getmtime(dict_file)
        return True
    except Exception as e:
        print(f"[DICTIONARY] Error saving dictionary: {e}")
        return False


@app.route("/api/dictionary", methods=["GET"])
def get_dictionary():
    """Get the custom dictionary"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    return jsonify({"success": True, "dictionary": load_custom_dictionary()})


@app.route("/api/dictionary", methods=["POST"])
def update_dictionary():
    """Update the entire custom dictionary"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    dictionary = data.get("dictionary", {})
    if save_custom_dictionary(dictionary):
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Failed to save dictionary"}), 500


@app.route("/api/dictionary/glossary", methods=["POST"])
def update_glossary():
    """Add or update a glossary mapping"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    lang_pair = data.get("lang_pair", "").strip()  # e.g., "en_to_es"
    source_term = data.get("source_term", "").strip()
    target_term = data.get("target_term", "").strip()

    if not lang_pair or not source_term or not target_term:
        return jsonify({"success": False, "error": "lang_pair, source_term, and target_term are required"}), 400

    dictionary = load_custom_dictionary()
    dictionary.setdefault("glossary", {}).setdefault(lang_pair, {})[source_term] = target_term
    save_custom_dictionary(dictionary)

    return jsonify({"success": True, "glossary": dictionary["glossary"]})


@app.route("/api/dictionary/glossary", methods=["DELETE"])
def remove_glossary_entry():
    """Remove a glossary mapping"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    lang_pair = data.get("lang_pair", "").strip()
    source_term = data.get("source_term", "").strip()

    if not lang_pair or not source_term:
        return jsonify({"success": False, "error": "lang_pair and source_term are required"}), 400

    dictionary = load_custom_dictionary()
    glossary = dictionary.get("glossary", {}).get(lang_pair, {})
    if source_term in glossary:
        del glossary[source_term]
        save_custom_dictionary(dictionary)

    return jsonify({"success": True, "glossary": dictionary.get("glossary", {})})


@app.route("/api/corrections/settings", methods=["GET"])
def get_corrections_settings():
    """Get corrections configuration"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    return jsonify({"success": True, "corrections": config.get("corrections", {})})


@app.route("/api/corrections/settings", methods=["POST"])
def update_corrections_settings():
    """Update corrections configuration"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    corrections = data.get("corrections", {})
    config["corrections"] = {**config.get("corrections", {}), **corrections}
    save_config(config)
    return jsonify({"success": True, "corrections": config["corrections"]})


# =============================================================================
# Remote Mic API Endpoints
# =============================================================================

@app.route("/api/audio-devices", methods=["GET"])
def get_audio_devices():
    """API endpoint to get list of available audio input devices"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        from audio_capture import list_audio_devices

        # Get backend from config
        audio_config = config.get("audio", {})
        backend = audio_config.get("backend", "ffmpeg")

        devices = []

        # Get devices using ffmpeg
        try:
            markers = audio_config.get("deprioritize_device_markers", [])
            devices = list_audio_devices(deprioritize_markers=markers)
            app_logger.info(f"Listed {len(devices)} devices using ffmpeg")

            # Normalize device format for UI compatibility
            normalized_devices = []
            for dev in devices:
                normalized_devices.append(
                    {
                        "index": dev.get("index", 0),
                        "name": dev.get(
                            "display_name", dev.get("name", "Unknown Device")
                        ),
                        "device_id": dev.get(
                            "name"
                        ),  # Actual device identifier for ffmpeg
                        "is_default": dev.get("is_default", False),
                    }
                )
            devices = normalized_devices

            # Prepend any .wav files sitting directly in APP_DIR as selectable test
            # sources (non-recursive — backups live in _AUTOMATIC_BACKUP/ subdirs and
            # are excluded so they don't flood the dropdown).
            import glob
            test_wavs = sorted(glob.glob(os.path.join(APP_DIR, "*.wav")))
            for i, wav_path in enumerate(test_wavs):
                devices.insert(0, {
                    "index": -1 - i,
                    "name": f"{os.path.basename(wav_path)} — Test Audio File",
                    "device_id": wav_path,
                    "is_default": False,
                })

        except Exception as e:
            app_logger.error(f"Error listing devices: {e}")
            devices = []

        # Ensure we have at least one device
        if not devices:
            devices = [
                {
                    "index": 0,
                    "name": "Default Microphone",
                    "device_id": "default",
                    "is_default": True,
                }
            ]

        return jsonify(
            {
                "success": True,
                "devices": devices,
                "default_index": 0,
                "backend": backend,
            }
        )

    except Exception:
        # Return fallback device
        return jsonify(
            {
                "success": True,
                "devices": [
                    {
                        "index": 0,
                        "name": "Default Microphone",
                        "is_default": True,
                    }
                ],
                "default_index": 0,
            }
        )


@app.route("/api/models/search", methods=["GET"])
def search_models():
    """Search for ASR models on Hugging Face"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        query = request.args.get("query", "whisper")
        limit = int(request.args.get("limit", 20))

        # Search Hugging Face for ASR models
        api = HfApi()
        models = api.list_models(
            filter="automatic-speech-recognition",
            search=query,
            sort="downloads",
            direction=-1,
            limit=limit,
        )

        model_list = []
        for model in models:
            try:
                info = model_info(model.modelId)

                # Calculate approximate model size from safetensors files
                size_bytes = 0
                size_str = "Unknown"
                try:
                    if hasattr(info, "siblings") and info.siblings:
                        for sibling in info.siblings:
                            if hasattr(sibling, "rfilename") and hasattr(
                                sibling, "size"
                            ):
                                # Sum up sizes of model files
                                if any(
                                    ext in sibling.rfilename
                                    for ext in [".safetensors", ".bin", ".pt", ".onnx"]
                                ):
                                    size_bytes += sibling.size

                        # Convert to readable format
                        if size_bytes > 0:
                            size_mb = size_bytes / (1024 * 1024)
                            if size_mb > 1024:
                                size_str = f"{size_mb / 1024:.2f} GB"
                            else:
                                size_str = f"{size_mb:.0f} MB"
                except OSError:
                    pass

                model_list.append(
                    {
                        "id": model.modelId,
                        "downloads": model.downloads
                        if hasattr(model, "downloads")
                        else 0,
                        "likes": model.likes if hasattr(model, "likes") else 0,
                        "tags": model.tags if hasattr(model, "tags") else [],
                        "library": info.library_name
                        if hasattr(info, "library_name")
                        else "unknown",
                        "size": size_str,
                        "size_bytes": size_bytes,
                    }
                )
            except (KeyError, ValueError, OSError, AttributeError):
                model_list.append(
                    {
                        "id": model.modelId,
                        "downloads": model.downloads
                        if hasattr(model, "downloads")
                        else 0,
                        "likes": model.likes if hasattr(model, "likes") else 0,
                        "tags": model.tags if hasattr(model, "tags") else [],
                        "library": "unknown",
                        "size": "Unknown",
                        "size_bytes": 0,
                    }
                )

        return jsonify(
            {"success": True, "models": model_list, "count": len(model_list)}
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e), "models": []}), 500


@app.route("/api/models/popular", methods=["GET"])
def get_popular_models():
    """Get a curated list of popular ASR models"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    popular_models = {
        "Whisper (OpenAI)": [
            {
                "id": "openai/whisper-tiny",
                "description": "Fastest, smallest Whisper model (39M params)",
                "size": "39M",
            },
            {
                "id": "openai/whisper-base",
                "description": "Base Whisper model (74M params)",
                "size": "74M",
            },
            {
                "id": "openai/whisper-small",
                "description": "Small Whisper model (244M params)",
                "size": "244M",
            },
            {
                "id": "openai/whisper-medium",
                "description": "Medium Whisper model (769M params)",
                "size": "769M",
            },
            {
                "id": "openai/whisper-large-v3",
                "description": "Latest large Whisper model (1.5B params)",
                "size": "1.5B",
            },
        ],
        "Distil-Whisper (Faster Whisper)": [
            {
                "id": "distil-whisper/distil-small.en",
                "description": "6x faster than Whisper, English only (166M params)",
                "size": "166M",
            },
            {
                "id": "distil-whisper/distil-medium.en",
                "description": "6x faster than Whisper, English only (394M params)",
                "size": "394M",
            },
            {
                "id": "distil-whisper/distil-large-v2",
                "description": "6x faster than large-v2, multilingual (756M params)",
                "size": "756M",
            },
            {
                "id": "distil-whisper/distil-large-v3",
                "description": "Latest distilled large model, 6x faster (756M params)",
                "size": "756M",
            },
        ],
        "Wav2Vec2 (Meta)": [
            {
                "id": "facebook/wav2vec2-base-960h",
                "description": "Base Wav2Vec2 trained on LibriSpeech (95M params)",
                "size": "95M",
            },
            {
                "id": "facebook/wav2vec2-large-960h",
                "description": "Large Wav2Vec2 trained on LibriSpeech (317M params)",
                "size": "317M",
            },
            {
                "id": "facebook/wav2vec2-large-960h-lv60-self",
                "description": "Large Wav2Vec2 with self-training (317M params)",
                "size": "317M",
            },
        ],
        "Other Popular Models": [
            {
                "id": "facebook/s2t-small-librispeech-asr",
                "description": "Speech2Text small model",
                "size": "77M",
            },
            {
                "id": "nvidia/stt_en_conformer_ctc_large",
                "description": "NVIDIA Conformer-CTC model",
                "size": "120M",
            },
        ],
    }

    return jsonify({"success": True, "models": popular_models})


@app.route("/api/models/info", methods=["GET"])
def get_model_info():
    """Get detailed information about a specific model"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        model_id = request.args.get("model_id")
        if not model_id:
            return jsonify({"success": False, "error": "model_id required"}), 400

        info = model_info(model_id)

        # Calculate model size from file siblings
        size_bytes = 0
        size_str = "Unknown"
        try:
            if hasattr(info, "siblings") and info.siblings:
                for sibling in info.siblings:
                    if hasattr(sibling, "rfilename") and hasattr(sibling, "size"):
                        # Sum up sizes of model files
                        if any(
                            ext in sibling.rfilename
                            for ext in [".safetensors", ".bin", ".pt", ".onnx"]
                        ):
                            size_bytes += sibling.size

                # Convert to readable format
                if size_bytes > 0:
                    size_mb = size_bytes / (1024 * 1024)
                    if size_mb > 1024:
                        size_str = f"{size_mb / 1024:.2f} GB"
                    else:
                        size_str = f"{size_mb:.0f} MB"
        except OSError:
            pass

        model_details = {
            "id": model_id,
            "author": info.author if hasattr(info, "author") else "unknown",
            "downloads": info.downloads if hasattr(info, "downloads") else 0,
            "likes": info.likes if hasattr(info, "likes") else 0,
            "tags": info.tags if hasattr(info, "tags") else [],
            "pipeline_tag": info.pipeline_tag
            if hasattr(info, "pipeline_tag")
            else "unknown",
            "library": info.library_name
            if hasattr(info, "library_name")
            else "unknown",
            "languages": [],
            "size": size_str,
            "size_bytes": size_bytes,
        }

        # Extract language tags
        if hasattr(info, "tags"):
            for tag in info.tags:
                if tag.startswith("language:"):
                    model_details["languages"].append(tag.replace("language:", ""))

        return jsonify({"success": True, "model": model_details})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Download progress tracking
DOWNLOAD_PROGRESS_FILE = os.path.join(APP_DIR, "download_progress.json")


def load_download_progress():
    """Load download progress from file"""
    try:
        if os.path.exists(DOWNLOAD_PROGRESS_FILE):
            with open(DOWNLOAD_PROGRESS_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load download progress: {e}")
    return {}


def save_download_progress():
    """Save download progress to file"""
    try:
        with active_downloads_lock:
            with open(DOWNLOAD_PROGRESS_FILE, "w") as f:
                json.dump(active_downloads, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save download progress: {e}")


def cleanup_stale_downloads():
    """Remove downloads based on status and age"""
    import time

    current_time = time.time()

    # Different retention periods by status
    DOWNLOADING_STALE_THRESHOLD = 86400  # 24 hours for stuck downloads
    COMPLETED_GRACE_PERIOD = 7200  # 2 hours for completed downloads
    FAILED_GRACE_PERIOD = 3600  # 1 hour for failed downloads

    with active_downloads_lock:
        stale_keys = []
        for model_id, info in active_downloads.items():
            last_update = info.get("last_update", 0)
            status = info.get("status", "downloading")
            age = current_time - last_update

            # Determine if should be removed based on status
            should_remove = False

            if status == "downloading" and age > DOWNLOADING_STALE_THRESHOLD:
                # Stuck download, likely stale
                should_remove = True
                print(f"[CLEANUP] Removing stale downloading: {model_id} (age: {age/3600:.1f}h)")
            elif status == "completed" and age > COMPLETED_GRACE_PERIOD:
                # Completed downloads after grace period
                should_remove = True
                print(f"[CLEANUP] Removing old completed download: {model_id} (age: {age/3600:.1f}h)")
            elif status == "failed" and age > FAILED_GRACE_PERIOD:
                # Failed downloads after shorter grace period
                should_remove = True
                print(f"[CLEANUP] Removing old failed download: {model_id} (age: {age/3600:.1f}h)")

            if should_remove:
                stale_keys.append(model_id)

        for key in stale_keys:
            del active_downloads[key]

        if stale_keys:
            # Save while still holding the lock - don't call save_download_progress() which would deadlock
            try:
                with open(DOWNLOAD_PROGRESS_FILE, "w") as f:
                    json.dump(active_downloads, f, indent=2)
            except Exception as e:
                print(f"[ERROR] Failed to save download progress: {e}")
            print(f"[CLEANUP] Removed {len(stale_keys)} stale download record(s)")


# Whisper model sizes in bytes (for progress tracking)
WHISPER_MODEL_SIZES = {
    "tiny.en": 75572083,
    "tiny": 75572083,
    "base.en": 145262807,
    "base": 145262807,
    "small.en": 483617219,
    "small": 483617219,
    "medium.en": 1528008539,
    "medium": 1528008539,
    "large-v1": 3087371615,
    "large-v2": 3087371615,
    "large-v3": 3087371615,
    "large": 3087371615,
    "large-v3-turbo": 1550580107,
    "turbo": 1550580107,
}

# Faster-Whisper models (CTranslate2 format, 4-10x faster)
FASTER_WHISPER_MODELS = {
    "tiny": {"repo": "Systran/faster-whisper-tiny", "size": "~75MB", "params": "39M", "lang": "Multilingual"},
    "tiny.en": {"repo": "Systran/faster-whisper-tiny.en", "size": "~75MB", "params": "39M", "lang": "English-only"},
    "base": {"repo": "Systran/faster-whisper-base", "size": "~145MB", "params": "74M", "lang": "Multilingual"},
    "base.en": {"repo": "Systran/faster-whisper-base.en", "size": "~145MB", "params": "74M", "lang": "English-only"},
    "small": {"repo": "Systran/faster-whisper-small", "size": "~465MB", "params": "244M", "lang": "Multilingual"},
    "small.en": {"repo": "Systran/faster-whisper-small.en", "size": "~465MB", "params": "244M", "lang": "English-only"},
    "medium": {"repo": "Systran/faster-whisper-medium", "size": "~1.5GB", "params": "769M", "lang": "Multilingual"},
    "medium.en": {"repo": "Systran/faster-whisper-medium.en", "size": "~1.5GB", "params": "769M", "lang": "English-only"},
    "large-v1": {"repo": "Systran/faster-whisper-large-v1", "size": "~3GB", "params": "1550M", "lang": "Multilingual"},
    "large-v2": {"repo": "Systran/faster-whisper-large-v2", "size": "~3GB", "params": "1550M", "lang": "Multilingual"},
    "large-v3": {"repo": "Systran/faster-whisper-large-v3", "size": "~3GB", "params": "1550M", "lang": "Multilingual"},
    "large-v3-turbo": {"repo": "Systran/faster-whisper-large-v3-turbo", "size": "~1.6GB", "params": "809M", "lang": "Multilingual"},
    "distil-large-v3": {"repo": "Systran/faster-distil-whisper-large-v3", "size": "~1.5GB", "params": "756M", "lang": "Multilingual"},
}

# Global dictionary to track active downloads
active_downloads = load_download_progress()
active_downloads_lock = threading.Lock()
cancelled_downloads = set()  # Track cancelled download IDs to prevent re-adding

# A "downloading" entry loaded from disk means the server died mid-download:
# the download thread is gone, so mark it failed instead of showing it for 24h
for _key, _info in active_downloads.items():
    if _info.get("status") == "downloading":
        _info["status"] = "failed"
        _info["error"] = "Interrupted by server restart"
        _info["last_update"] = time.time()
        print(f"[DOWNLOAD] Marked interrupted download as failed: {_key}")
save_download_progress()

# Clean up stale downloads on startup
cleanup_stale_downloads()


def try_register_download(key, total=None):
    """Atomically register a download in active_downloads.

    Returns False if a download for this key is already in progress."""
    with active_downloads_lock:
        existing = active_downloads.get(key)
        if existing and existing.get("status") == "downloading":
            return False
        cancelled_downloads.discard(key)
        active_downloads[key] = {
            "downloaded": 0,
            "total": total,
            "percentage": 0 if total else None,
            "start_time": time.time(),
            "last_update": time.time(),
            "status": "downloading",
        }
    save_download_progress()
    return True


def finish_download(key, error=None, cancelled=False):
    """Mark a download completed/failed and drop it from the cancelled set."""
    with active_downloads_lock:
        cancelled_downloads.discard(key)
        if not cancelled and key in active_downloads:
            entry = active_downloads[key]
            entry["last_update"] = time.time()
            if error is not None:
                entry["status"] = "failed"
                entry["error"] = str(error)
            else:
                entry["status"] = "completed"
                entry["percentage"] = 100
                entry["completion_time"] = time.time()
                if entry.get("total"):
                    entry["downloaded"] = entry["total"]
    save_download_progress()


def _path_size(path):
    """Size in bytes of a file, or recursive size of a directory."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def monitor_download_progress(key, path, total=None, interval=2):
    """Poll the size of `path` (file or directory) and update active_downloads[key].

    Runs until the entry leaves "downloading" state, disappears, or is cancelled.
    Percentage is capped at 99 — the download code sets 100 on completion."""
    import time as _time

    while True:
        with active_downloads_lock:
            entry = active_downloads.get(key)
            if entry is None or entry.get("status") != "downloading" or key in cancelled_downloads:
                return
        if os.path.exists(path):
            size = _path_size(path)
            with active_downloads_lock:
                entry = active_downloads.get(key)
                if entry is None or entry.get("status") != "downloading":
                    return
                entry["downloaded"] = size
                entry["last_update"] = _time.time()
                entry_total = entry.get("total") or total
                if entry_total and entry_total > 0:
                    entry["percentage"] = min(int((size / entry_total) * 100), 99)
            save_download_progress()
        _time.sleep(interval)


def start_download_monitor(key, path, total=None, interval=2):
    """Spawn the directory-size progress monitor as a daemon thread."""
    threading.Thread(
        target=monitor_download_progress,
        args=(key, path, total, interval),
        daemon=True,
        name=f"dl-monitor-{key}",
    ).start()


def download_url_to_file(url, dest_path, cancel_check=None, max_attempts=5, log=print):
    """Download a URL to a file with resume + retry, preferring wget/curl.

    Falls back to a pure-Python streaming download when neither tool exists
    (e.g. minimal Windows installs). `cancel_check` is polled during the
    download; returning True aborts it. Returns "ok" or "cancelled"; raises
    after all attempts fail."""
    import subprocess
    import tempfile as _tempfile
    import time as _time
    import urllib.request

    if shutil.which("wget"):
        dl_cmd = ['wget', '-c', '-t', '3', '-T', '120', '--retry-connrefused',
                  '--waitretry', '5', '-O', dest_path, url]
    elif shutil.which("curl"):
        dl_cmd = ['curl', '-L', '-C', '-', '--retry', '3', '--retry-delay', '5',
                  '--retry-connrefused', '--connect-timeout', '30',
                  '--max-time', '600', '-o', dest_path, url]
    else:
        dl_cmd = None  # pure-Python fallback below

    last_error = ""
    for attempt in range(1, max_attempts + 1):
        if dl_cmd:
            # Output goes to a temp file: a PIPE would fill up with progress
            # noise and block the process, since nothing drains it while we poll
            with _tempfile.TemporaryFile(mode="w+", errors="replace") as outf:
                proc = subprocess.Popen(dl_cmd, stdout=outf, stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    if cancel_check and cancel_check():
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        return "cancelled"
                    _time.sleep(0.5)
                if proc.returncode == 0:
                    return "ok"
                outf.seek(0)
                last_error = outf.read()[-500:]
            returncode = proc.returncode
        else:
            try:
                with urllib.request.urlopen(url, timeout=120) as src, open(dest_path, "wb") as out:
                    while True:
                        if cancel_check and cancel_check():
                            return "cancelled"
                        chunk = src.read(65536)
                        if not chunk:
                            break
                        out.write(chunk)
                return "ok"
            except Exception as e:
                last_error = str(e)
                returncode = 1

        log(f"[WARNING] Download attempt {attempt}/{max_attempts} failed for "
            f"{os.path.basename(dest_path)} (exit code {returncode})")
        if attempt < max_attempts:
            if os.path.exists(dest_path):
                partial_size = os.path.getsize(dest_path)
                log(f"[INFO] Partial file exists ({partial_size / (1024*1024):.1f} MB), will resume")
            _time.sleep(5 * attempt)

    raise Exception(
        f"Failed to download {os.path.basename(dest_path)} after {max_attempts} attempts: {last_error[:300]}"
    )


def download_hf_repo_files(repo_id, local_dir, download_key, log=print):
    """Download every file of a HuggingFace repo with resume + cancellation.

    Returns "ok" or "cancelled"; raises on failure after retries."""
    from huggingface_hub import list_repo_files, hf_hub_url

    os.makedirs(local_dir, exist_ok=True)
    local_root = os.path.abspath(local_dir)
    files = list_repo_files(repo_id=repo_id)
    log(f"[DOWNLOAD] Found {len(files)} files to download for {repo_id}")

    for idx, filename in enumerate(files):
        if download_key in cancelled_downloads:
            return "cancelled"

        dest_path = os.path.abspath(os.path.join(local_root, filename))
        if not dest_path.startswith(local_root + os.sep):
            raise ValueError(f"Unsafe filename in repo {repo_id}: {filename}")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Skip if already downloaded and has content
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            log(f"[DOWNLOAD] Already exists: {filename}")
            continue

        log(f"[DOWNLOAD] Downloading file {idx + 1}/{len(files)}: {filename}")

        # File-count progress only when no byte total is known (a directory
        # size monitor provides byte-accurate progress otherwise)
        with active_downloads_lock:
            entry = active_downloads.get(download_key)
            if entry and entry.get("status") == "downloading" and not entry.get("total"):
                entry["percentage"] = int((idx / len(files)) * 100)
                entry["last_update"] = time.time()
        save_download_progress()

        url = hf_hub_url(repo_id=repo_id, filename=filename)
        outcome = download_url_to_file(
            url, dest_path,
            cancel_check=lambda: download_key in cancelled_downloads,
            log=log,
        )
        if outcome == "cancelled":
            return "cancelled"
        log(f"[OK] Downloaded: {filename}")

    return "ok"

# Start periodic cleanup thread
def periodic_cleanup():
    """Run cleanup every hour"""
    import time
    while True:
        time.sleep(3600)  # Every hour
        cleanup_stale_downloads()

cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()


@app.route("/api/models/download", methods=["POST"])
def download_model():
    """Download/cache a model (Hugging Face or Whisper)
    Example: POST /api/models/download {"model_type": "whisper", "model_name": "small"}"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        data = request.get_json()
        model_type = data.get("model_type")

        if model_type == "whisper":
            # Handle Whisper model download
            model_name = data.get("model_name")
            if not model_name:
                return jsonify(
                    {
                        "success": False,
                        "error": "model_name required for Whisper models",
                    }
                ), 400

            print(f"[DOWNLOAD] Downloading Whisper model: {model_name}")

            _lazy_import_ml_libraries()

            # Create custom download directory in ./models
            models_dir = MODELS_DIR
            os.makedirs(models_dir, exist_ok=True)

            whisper_dir = os.path.join(models_dir, f"whisper-{model_name}")
            os.makedirs(whisper_dir, exist_ok=True)

            # Set environment variable to use custom download directory
            os.environ["WHISPER_CACHE"] = whisper_dir


            model_key = f"whisper-{model_name}"
            if not try_register_download(model_key, total=WHISPER_MODEL_SIZES.get(model_name)):
                return jsonify({"success": False, "error": "Download already in progress"}), 409

            try:
                start_download_monitor(
                    model_key, os.path.join(whisper_dir, f"{model_name}.pt")
                )

                # Download the model file without loading it into GPU memory
                # Using custom download function that computes SHA256 during download
                # This avoids the blocking post-download verification that reads the whole file
                from whisper import _MODELS
                import urllib.request
                import hashlib
                from tqdm import tqdm

                if model_name not in _MODELS:
                    raise ValueError(f"Unknown Whisper model: {model_name}. Available: {list(_MODELS.keys())}")

                url = _MODELS[model_name]
                expected_sha256 = url.split("/")[-2]
                download_target = os.path.join(whisper_dir, os.path.basename(url))

                # Check if already downloaded with correct checksum
                if os.path.isfile(download_target):
                    print(f"[INFO] File exists, verifying checksum: {download_target}")
                    sha256_hash = hashlib.sha256()
                    with open(download_target, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            sha256_hash.update(chunk)
                    if sha256_hash.hexdigest() == expected_sha256:
                        print("[OK] Existing file checksum matches")
                        model_path = download_target
                    else:
                        print("[WARN] Existing file checksum mismatch, re-downloading")
                        os.remove(download_target)
                        model_path = None
                else:
                    model_path = None

                if model_path is None:
                    # Download with streaming SHA256 computation
                    print(f"[INFO] Downloading Whisper model to {download_target}")
                    sha256_hash = hashlib.sha256()

                    download_cancelled = False
                    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
                        total_size = int(source.info().get("Content-Length", 0))
                        with tqdm(total=total_size, ncols=80, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
                            while True:
                                if model_key in cancelled_downloads:
                                    download_cancelled = True
                                    break
                                buffer = source.read(8192)
                                if not buffer:
                                    break
                                output.write(buffer)
                                sha256_hash.update(buffer)  # Compute SHA256 during download
                                pbar.update(len(buffer))

                    if download_cancelled:
                        print(f"[CANCELLED] Whisper download cancelled: {model_name}")
                        try:
                            os.remove(download_target)
                        except OSError:
                            pass
                        finish_download(model_key, cancelled=True)
                        return jsonify({"success": False, "message": "Download cancelled"})

                    # Verify checksum computed during download
                    computed_sha256 = sha256_hash.hexdigest()
                    if computed_sha256 != expected_sha256:
                        os.remove(download_target)
                        raise RuntimeError(
                            f"Model download failed: SHA256 mismatch. Expected {expected_sha256}, got {computed_sha256}"
                        )

                    model_path = download_target
                    print("[OK] Download complete, checksum verified")

                message = f"Whisper {model_name} model downloaded to {model_path}"

                print(f"[OK] {message}")

                finish_download(model_key)
            except Exception as e:
                print(f"[ERROR] Whisper model download failed: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                finish_download(model_key, error=e)
                raise

            return jsonify(
                {
                    "success": True,
                    "message": f"downloaded to {whisper_dir}",
                    "model_name": model_name,
                    "path": whisper_dir,
                }
            )

        else:
            # Handle HuggingFace model download (original logic)
            model_id = data.get("model_id")
            local_dir = data.get("local_dir")

            if not model_id:
                return jsonify({"success": False, "error": "model_id required"}), 400

            print(f"[DOWNLOAD] Downloading HuggingFace model: {model_id}")

            _lazy_import_ml_libraries()

            # If no local_dir specified, download to ./models directory
            if not local_dir:
                models_dir = MODELS_DIR
                os.makedirs(models_dir, exist_ok=True)

                # Use model name as directory (replace / with --)
                model_dir_name = model_id.replace("/", "--")
                local_dir = os.path.join(models_dir, model_dir_name)


            if not try_register_download(model_id):
                return jsonify({"success": False, "error": "Download already in progress"}), 409

            try:
                # Per-file download with resume + cancellation
                # (huggingface_hub's snapshot_download hangs on large files)
                outcome = download_hf_repo_files(model_id, local_dir, model_id)
                if outcome == "cancelled":
                    print(f"[CANCELLED] Download cancelled for {model_id}")
                    finish_download(model_id, cancelled=True)
                    return jsonify({"success": False, "message": "Download cancelled"})

                path = local_dir
                message = f"Model {model_id} downloaded to: {path}"

                print(f"[OK] {message}")

                finish_download(model_id)
            except Exception as e:
                finish_download(model_id, error=e)
                raise

            return jsonify(
                {
                    "success": True,
                    "message": message,
                    "model_id": model_id,
                    "path": path,
                }
            )

    except Exception as e:
        print(f"[ERROR] Error downloading model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/download-status", methods=["GET"])
def download_status():
    """Get download status with real-time progress tracking"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        with active_downloads_lock:
            download_list = [
                {
                    "model": name,
                    "downloaded": info.get("downloaded", 0),
                    "total": info.get("total"),
                    "percentage": info.get("percentage"),
                    "status": info.get("status", "downloading"),
                    "error": info.get("error"),
                    "completion_time": info.get("completion_time"),
                }
                for name, info in active_downloads.items()
            ]

        # Separate by status
        downloading = [d for d in download_list if d["status"] == "downloading"]
        completed = [d for d in download_list if d["status"] == "completed"]
        failed = [d for d in download_list if d["status"] == "failed"]

        return jsonify(
            {
                "status": "active" if downloading else "idle",
                "active_downloads": download_list,
                "downloading": downloading,
                "completed": completed,
                "failed": failed,
                "message": f"{len(downloading)} active, {len(completed)} completed, {len(failed)} failed"
                if download_list
                else "No active downloads",
            }
        )
    except Exception as e:
        print(f"[ERROR] Error getting download status: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/models/cancel-download", methods=["POST"])
def cancel_download():
    """Cancel an active download and clean up"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = request.get_json()
    model_id = data.get("model_id") if data else None

    if not model_id:
        return jsonify({"success": False, "error": "model_id required"}), 400

    try:
        with active_downloads_lock:
            entry = active_downloads.get(model_id)
            was_downloading = entry is not None and entry.get("status") == "downloading"
            if was_downloading:
                # Signal the download thread to stop
                cancelled_downloads.add(model_id)
            if entry is not None:
                del active_downloads[model_id]

        # Save outside the lock to avoid deadlock (save_download_progress acquires the same lock)
        save_download_progress()

        if not was_downloading:
            # Completed/failed entry (or already gone): this is a dismiss, not a
            # cancel — never delete files for a download that isn't in flight
            return jsonify({"success": True, "message": f"Dismissed {model_id}"})

        # Clean up partial download directories/files so model shows as not downloaded
        models_dir = MODELS_DIR

        # model_id already includes prefix (e.g., "whisper-small.en" or "faster-whisper-base.en")
        # For HuggingFace models like "facebook/nllb-200-distilled-600M", slashes become double dashes
        dir_name = model_id.replace("/", "--")
        model_path = os.path.normpath(os.path.join(models_dir, dir_name))
        # Containment check: model_id is user input and must resolve to a
        # subdirectory strictly inside MODELS_DIR (rmtree happens below)
        if (model_path == models_dir
                or os.path.commonpath([model_path, models_dir]) != models_dir):
            return jsonify({"success": False, "error": "Invalid model id"}), 400
        if os.path.exists(model_path):
            try:
                shutil.rmtree(model_path)
                print(f"[INFO] Cleaned up partial download: {model_path}")
            except Exception as e:
                print(f"[WARNING] Failed to clean up {model_path}: {e}")

        # For whisper .pt files in cache, extract base name (e.g., "whisper-small.en" -> "small.en")
        if model_id.startswith("whisper-") and not model_id.startswith("whisper-faster"):
            whisper_cache = os.path.expanduser("~/.cache/whisper")
            base_name = model_id[8:]  # Remove "whisper-" prefix to get "small.en"
            pt_file = os.path.join(whisper_cache, f"{base_name}.pt")
            if os.path.exists(pt_file):
                try:
                    os.remove(pt_file)
                    print(f"[INFO] Cleaned up partial download: {pt_file}")
                except Exception as e:
                    print(f"[WARNING] Failed to clean up {pt_file}: {e}")

        return jsonify({"success": True, "message": f"Cancelled {model_id}"})
    except Exception as e:
        print(f"[ERROR] Error cancelling download: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/upload", methods=["POST"])
def upload_model():
    """Upload a local model to Hugging Face"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        data = request.get_json()
        token = data.get("token")
        model_path = data.get("model_path")
        repo_id = data.get("repo_id")
        commit_message = data.get("commit_message", "Upload model")
        private = data.get("private", False)

        if not all([token, model_path, repo_id]):
            return jsonify(
                {"success": False, "error": "token, model_path, and repo_id required"}
            ), 400

        _lazy_import_ml_libraries()
        from huggingface_hub import HfApi
        from pathlib import Path

        model_path = Path(model_path)
        if not model_path.exists():
            return jsonify(
                {"success": False, "error": f"Model path not found: {model_path}"}
            ), 404

        api = HfApi(token=token)

        # Create repository if it doesn't exist
        try:
            api.create_repo(
                repo_id=repo_id,
                token=token,
                private=private,
                repo_type="model",
                exist_ok=True,
            )
            print(f"[OK] Repository ready: {repo_id}")
        except Exception as e:
            print(f"[WARNING] Repository creation: {e}")

        # Upload files
        if model_path.is_file():
            # Single file upload
            api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=model_path.name,
                repo_id=repo_id,
                token=token,
                commit_message=commit_message,
            )
        else:
            # Directory upload
            api.upload_folder(
                folder_path=str(model_path),
                repo_id=repo_id,
                token=token,
                commit_message=commit_message,
            )

        print(f"[OK] Model uploaded successfully to: https://huggingface.co/{repo_id}")

        return jsonify(
            {
                "success": True,
                "message": f"Model uploaded successfully to: https://huggingface.co/{repo_id}",
                "repo_id": repo_id,
                "url": f"https://huggingface.co/{repo_id}",
            }
        )

    except Exception as e:
        print(f"[ERROR] Error uploading model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/local", methods=["GET"])
def get_local_models():
    """Get list of local models in directory"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        directory = request.args.get("directory", "./models")
        models_path = Path(directory)

        if not models_path.exists():
            return jsonify(
                {
                    "success": True,
                    "models": [],
                    "message": f"Directory not found: {directory}",
                }
            )

        models = []
        for item in models_path.iterdir():
            if item.is_dir():
                # Check for model files
                model_files = (
                    list(item.glob("*.bin"))
                    + list(item.glob("*.safetensors"))
                    + list(item.glob("*.pt"))
                )
                if model_files:
                    size_mb = sum(f.stat().st_size for f in model_files) / (1024 * 1024)
                    models.append(
                        {
                            "name": item.name,
                            "path": str(item),
                            "files": [f.name for f in model_files],
                            "size_mb": size_mb,
                        }
                    )

        return jsonify({"success": True, "models": models, "count": len(models)})

    except Exception as e:
        return jsonify({"success": False, "error": str(e), "models": []}), 500


@app.route("/api/models/cached", methods=["GET"])
def get_cached_models():
    """Get list of cached/downloaded models"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        import os

        # Get Hugging Face cache directory
        cache_dir = os.getenv(
            "HF_HOME",
            os.getenv("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface")),
        )
        models_dir = os.path.join(cache_dir, "hub")

        cached_models = []

        if os.path.exists(models_dir):
            # List all model directories
            for item in os.listdir(models_dir):
                if item.startswith("models--"):
                    # Extract model name from directory
                    model_name = item.replace("models--", "").replace("--", "/")
                    model_path = os.path.join(models_dir, item)

                    # Get directory size
                    total_size = 0
                    for dirpath, dirnames, filenames in os.walk(model_path):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            if os.path.exists(fp):
                                total_size += os.path.getsize(fp)

                    # Convert to readable format
                    size_mb = total_size / (1024 * 1024)
                    if size_mb > 1024:
                        size_str = f"{size_mb / 1024:.2f} GB"
                    else:
                        size_str = f"{size_mb:.2f} MB"

                    cached_models.append(
                        {
                            "id": model_name,
                            "size": size_str,
                            "size_bytes": total_size,
                            "path": model_path,
                        }
                    )

        # Also check for Whisper models
        whisper_cache = os.path.expanduser("~/.cache/whisper")
        if os.path.exists(whisper_cache):
            for item in os.listdir(whisper_cache):
                if item.endswith(".pt"):
                    model_path = os.path.join(whisper_cache, item)
                    size_bytes = os.path.getsize(model_path)
                    size_mb = size_bytes / (1024 * 1024)
                    if size_mb > 1024:
                        size_str = f"{size_mb / 1024:.2f} GB"
                    else:
                        size_str = f"{size_mb:.2f} MB"

                    cached_models.append(
                        {
                            "id": f"whisper/{item.replace('.pt', '')}",
                            "size": size_str,
                            "size_bytes": size_bytes,
                            "path": model_path,
                        }
                    )

        return jsonify(
            {
                "success": True,
                "models": cached_models,
                "count": len(cached_models),
                "cache_dir": cache_dir,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e), "models": []}), 500


# Model Manager Endpoints
@app.route("/api/models/manager", methods=["GET"])
def get_model_manager():
    """Get centralized model configuration for both live and file transcription"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        global config

        # Get live transcription model config
        live_model_config = config.get("model", {})

        # Get file transcription model config
        ft_config = config.get("file_transcription", {})
        ft_model_config = ft_config.get("model", {})

        # If file transcription doesn't have model config, use live config as fallback
        if not ft_model_config:
            ft_model_config = live_model_config.copy()

        return jsonify(
            {
                "success": True,
                "live_model": {
                    "type": live_model_config.get("type", "whisper"),
                    "backend": live_model_config.get("backend", ""),
                    "whisper": {
                        "model": live_model_config.get("whisper", {}).get(
                            "model", "tiny"
                        ),
                    },
                    "huggingface": {
                        "model_id": live_model_config.get("huggingface", {}).get(
                            "model_id", "openai/whisper-base"
                        ),
                        "use_flash_attention": live_model_config.get(
                            "huggingface", {}
                        ).get("use_flash_attention", False),
                    },
                    "custom": {
                        "model_path": live_model_config.get("custom", {}).get(
                            "model_path", ""
                        ),
                        "model_type": live_model_config.get("custom", {}).get(
                            "model_type", "whisper"
                        ),
                    },
                },
                "file_transcription_model": {
                    "type": ft_model_config.get("type", "whisper"),
                    "backend": ft_model_config.get("backend", ""),
                    "whisper": {
                        "model": ft_model_config.get("whisper", {}).get(
                            "model", "base"
                        ),
                    },
                    "huggingface": {
                        "model_id": ft_model_config.get("huggingface", {}).get(
                            "model_id", "openai/whisper-base"
                        ),
                        "use_flash_attention": ft_model_config.get(
                            "huggingface", {}
                        ).get("use_flash_attention", False),
                    },
                    "custom": {
                        "model_path": ft_model_config.get("custom", {}).get(
                            "model_path", ""
                        ),
                        "model_type": ft_model_config.get("custom", {}).get(
                            "model_type", "whisper"
                        ),
                    },
                },
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/models/manager", methods=["POST"])
def update_model_manager():
    """Update centralized model configuration"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        data = request.get_json()
        global config

        # Update live transcription model config
        if "live_model" in data:
            live_model = data["live_model"]
            config["model"] = {"type": live_model.get("type", "whisper")}

            if live_model.get("type") == "whisper":
                config["model"]["whisper"] = {
                    "model": live_model.get("whisper", {}).get("model", "tiny"),
                }
                # Always save backend setting (faster-whisper vs standard whisper)
                config["model"]["backend"] = live_model.get("backend")
            elif live_model.get("type") == "huggingface":
                config["model"]["huggingface"] = {
                    "model_id": live_model.get("huggingface", {}).get(
                        "model_id", "openai/whisper-base"
                    ),
                    "use_flash_attention": live_model.get("huggingface", {}).get(
                        "use_flash_attention", False
                    ),
                }
            elif live_model.get("type") == "custom":
                config["model"]["custom"] = {
                    "model_path": live_model.get("custom", {}).get("model_path", ""),
                    "model_type": live_model.get("custom", {}).get(
                        "model_type", "whisper"
                    ),
                }

        # Update file transcription model config
        if "file_transcription_model" in data:
            ft_model = data["file_transcription_model"]

            if "file_transcription" not in config:
                config["file_transcription"] = {}

            config["file_transcription"]["model"] = {
                "type": ft_model.get("type", "whisper")
            }

            if ft_model.get("type") == "whisper":
                config["file_transcription"]["model"]["whisper"] = {
                    "model": ft_model.get("whisper", {}).get("model", "base"),
                }
                # Always save backend setting (faster-whisper vs standard whisper)
                config["file_transcription"]["model"]["backend"] = ft_model.get("backend")
            elif ft_model.get("type") == "huggingface":
                config["file_transcription"]["model"]["huggingface"] = {
                    "model_id": ft_model.get("huggingface", {}).get(
                        "model_id", "openai/whisper-base"
                    ),
                    "use_flash_attention": ft_model.get("huggingface", {}).get(
                        "use_flash_attention", False
                    ),
                }
            elif ft_model.get("type") == "custom":
                config["file_transcription"]["model"]["custom"] = {
                    "model_path": ft_model.get("custom", {}).get("model_path", ""),
                    "model_type": ft_model.get("custom", {}).get(
                        "model_type", "whisper"
                    ),
                }

        # Save configuration
        save_config(config)

        return jsonify(
            {"success": True, "message": "Model configurations updated successfully"}
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/models/sync", methods=["POST"])
def sync_model_configs():
    """Sync model configuration between live and file transcription"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        data = request.get_json()
        direction = data.get("direction", "live_to_file")  # or 'file_to_live'

        global config

        if direction == "live_to_file":
            # Copy live model config to file transcription
            live_model_config = config.get("model", {}).copy()

            if "file_transcription" not in config:
                config["file_transcription"] = {}

            config["file_transcription"]["model"] = live_model_config

            message = "Live model configuration copied to file transcription"

        elif direction == "file_to_live":
            # Copy file transcription model config to live
            ft_model_config = (
                config.get("file_transcription", {}).get("model", {}).copy()
            )

            if ft_model_config:
                config["model"] = ft_model_config
                message = "File transcription model configuration copied to live"
            else:
                return jsonify(
                    {
                        "success": False,
                        "error": "No file transcription model configuration found",
                    }
                )

        # Save configuration
        save_config(config)

        return jsonify({"success": True, "message": message})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# Cache for discovered Whisper models (to avoid frequent checks)
_whisper_models_cache = None
_whisper_models_cache_time = None
WHISPER_CACHE_DURATION = 86400  # Cache for 24 hours (1 day)
WHISPER_MODELS_FILE = _seed_from_bundle("whisper_models.json")  # Persistent storage file


def load_whisper_models_from_file():
    """Load discovered Whisper models from local file"""
    try:
        if os.path.exists(WHISPER_MODELS_FILE):
            with open(WHISPER_MODELS_FILE, "r") as f:
                data = json.load(f)
                return data.get("models", {}), data.get("timestamp", 0)
    except Exception as e:
        print(f"[WARNING] Could not load Whisper models from file: {e}")
    return None, None


def save_whisper_models_to_file(models):
    """Save discovered Whisper models to local file"""
    try:
        import datetime

        data = {
            "models": models,
            "timestamp": time.time(),
            "last_updated": datetime.datetime.now().isoformat(),
        }
        with open(WHISPER_MODELS_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[OK] Saved {len(models)} Whisper models to {WHISPER_MODELS_FILE}")
    except Exception as e:
        print(f"[ERROR] Could not save Whisper models to file: {e}")


def get_whisper_models_list():
    """Get list of available Whisper models with fallback to defaults"""
    global _whisper_models_cache, _whisper_models_cache_time

    # Check if memory cache is still valid
    if _whisper_models_cache and _whisper_models_cache_time:
        if (time.time() - _whisper_models_cache_time) < WHISPER_CACHE_DURATION:
            return _whisper_models_cache

    # Try to load from file
    file_models, file_timestamp = load_whisper_models_from_file()
    if file_models and file_timestamp:
        # Use file cache (even if expired, better than defaults)
        _whisper_models_cache = file_models
        _whisper_models_cache_time = file_timestamp
        print(
            f"[OK] Loaded {len(file_models)} Whisper models from {WHISPER_MODELS_FILE}"
        )
        return file_models

    # Default models (fallback)
    default_models = {
        "tiny": {
            "params": "39M",
            "size": "~75MB",
            "desc": "Fastest",
            "lang": "Multilingual",
        },
        "tiny.en": {
            "params": "39M",
            "size": "~75MB",
            "desc": "Fastest",
            "lang": "English-only",
        },
        "base": {
            "params": "74M",
            "size": "~142MB",
            "desc": "Balanced",
            "lang": "Multilingual",
        },
        "base.en": {
            "params": "74M",
            "size": "~142MB",
            "desc": "Balanced",
            "lang": "English-only",
        },
        "small": {
            "params": "244M",
            "size": "~466MB",
            "desc": "Good accuracy",
            "lang": "Multilingual",
        },
        "small.en": {
            "params": "244M",
            "size": "~466MB",
            "desc": "Good accuracy",
            "lang": "English-only",
        },
        "medium": {
            "params": "769M",
            "size": "~1.5GB",
            "desc": "Better accuracy",
            "lang": "Multilingual",
        },
        "medium.en": {
            "params": "769M",
            "size": "~1.5GB",
            "desc": "Better accuracy",
            "lang": "English-only",
        },
        "large": {
            "params": "1550M",
            "size": "~3GB",
            "desc": "Best accuracy",
            "lang": "Multilingual",
        },
        "large-v2": {
            "params": "1550M",
            "size": "~3GB",
            "desc": "Best accuracy v2",
            "lang": "Multilingual",
        },
        "large-v3": {
            "params": "1550M",
            "size": "~3GB",
            "desc": "Best accuracy v3",
            "lang": "Multilingual",
        },
    }

    # Update memory cache
    _whisper_models_cache = default_models
    _whisper_models_cache_time = time.time()

    # Save to file for persistence
    save_whisper_models_to_file(default_models)

    return default_models


# Cache for discovered Faster-Whisper models
_faster_whisper_models_cache = None
_faster_whisper_models_cache_time = None
FASTER_WHISPER_MODELS_FILE = _seed_from_bundle("faster_whisper_models.json")


def load_faster_whisper_models_from_file():
    """Load discovered Faster-Whisper models from local file"""
    try:
        if os.path.exists(FASTER_WHISPER_MODELS_FILE):
            with open(FASTER_WHISPER_MODELS_FILE, "r") as f:
                data = json.load(f)
                return data.get("models", {}), data.get("timestamp", 0)
    except Exception as e:
        print(f"[WARNING] Could not load Faster-Whisper models from file: {e}")
    return None, None


def save_faster_whisper_models_to_file(models):
    """Save discovered Faster-Whisper models to local file"""
    try:
        import datetime

        data = {
            "models": models,
            "timestamp": time.time(),
            "last_updated": datetime.datetime.now().isoformat(),
        }
        with open(FASTER_WHISPER_MODELS_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[OK] Saved {len(models)} Faster-Whisper models to {FASTER_WHISPER_MODELS_FILE}")
    except Exception as e:
        print(f"[ERROR] Could not save Faster-Whisper models to file: {e}")


def get_faster_whisper_models_list():
    """Get list of available Faster-Whisper models with fallback to defaults"""
    global _faster_whisper_models_cache, _faster_whisper_models_cache_time

    # Check if memory cache is still valid
    if _faster_whisper_models_cache and _faster_whisper_models_cache_time:
        if (time.time() - _faster_whisper_models_cache_time) < WHISPER_CACHE_DURATION:
            return _faster_whisper_models_cache

    # Try to load from file
    file_models, file_timestamp = load_faster_whisper_models_from_file()
    if file_models and file_timestamp:
        _faster_whisper_models_cache = file_models
        _faster_whisper_models_cache_time = file_timestamp
        print(f"[OK] Loaded {len(file_models)} Faster-Whisper models from {FASTER_WHISPER_MODELS_FILE}")
        return file_models

    # Use hardcoded defaults as fallback
    _faster_whisper_models_cache = FASTER_WHISPER_MODELS
    _faster_whisper_models_cache_time = time.time()

    # Save to file for persistence
    save_faster_whisper_models_to_file(FASTER_WHISPER_MODELS)

    return FASTER_WHISPER_MODELS


@app.route("/api/models/refresh-faster-whisper", methods=["POST"])
def refresh_faster_whisper_models():
    """Discover available Faster-Whisper models from HuggingFace"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global _faster_whisper_models_cache, _faster_whisper_models_cache_time

    try:
        from huggingface_hub import HfApi

        api = HfApi()

        # Search for Systran's faster-whisper models
        models = list(api.list_models(author="Systran", search="faster-whisper"))

        # Known model specifications for size/params estimation
        model_specs = {
            "tiny": {"size": "~75MB", "params": "39M"},
            "base": {"size": "~145MB", "params": "74M"},
            "small": {"size": "~465MB", "params": "244M"},
            "medium": {"size": "~1.5GB", "params": "769M"},
            "large": {"size": "~3GB", "params": "1550M"},
            "distil": {"size": "~1.5GB", "params": "756M"},
            "turbo": {"size": "~1.6GB", "params": "809M"},
        }

        discovered_models = {}
        old_models = _faster_whisper_models_cache or FASTER_WHISPER_MODELS

        for model in models:
            repo_id = model.id  # e.g., "Systran/faster-whisper-large-v3"
            # Accept both faster-whisper and faster-distil-whisper repos
            if not (repo_id.startswith("Systran/faster-whisper") or repo_id.startswith("Systran/faster-distil-whisper")):
                continue

            # Extract model name from repo_id
            if repo_id.startswith("Systran/faster-distil-whisper-"):
                model_name = "distil-" + repo_id.replace("Systran/faster-distil-whisper-", "")
            else:
                model_name = repo_id.replace("Systran/faster-whisper-", "")

            # Determine specs based on model name
            size = "~3GB"
            params = "1550M"
            lang = "Multilingual"

            for key, specs in model_specs.items():
                if key in model_name.lower():
                    size = specs["size"]
                    params = specs["params"]
                    break

            if ".en" in model_name:
                lang = "English-only"

            discovered_models[model_name] = {
                "repo": repo_id,
                "size": size,
                "params": params,
                "lang": lang,
            }

        # Find new and removed models
        old_names = set(old_models.keys())
        new_names = set(discovered_models.keys())
        added = new_names - old_names
        removed = old_names - new_names

        # Update cache
        _faster_whisper_models_cache = discovered_models
        _faster_whisper_models_cache_time = time.time()

        # Save to file
        save_faster_whisper_models_to_file(discovered_models)

        return jsonify({
            "success": True,
            "message": f"Found {len(discovered_models)} Faster-Whisper models",
            "count": len(discovered_models),
            "new_models": list(added),
            "removed_models": list(removed),
        })

    except Exception as e:
        print(f"[ERROR] Error refreshing Faster-Whisper models: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/refresh-whisper", methods=["POST"])
def refresh_whisper_models():
    """Discover available Whisper models from the whisper package"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global _whisper_models_cache, _whisper_models_cache_time

    try:
        # Check whisper package for available models
        try:
            import whisper

            available_models = whisper.available_models()

            # Known model specifications
            model_specs = {
                "tiny": {
                    "params": "39M",
                    "size": "~75MB",
                    "desc": "Fastest",
                    "lang": "Multilingual",
                },
                "tiny.en": {
                    "params": "39M",
                    "size": "~75MB",
                    "desc": "Fastest",
                    "lang": "English-only",
                },
                "base": {
                    "params": "74M",
                    "size": "~142MB",
                    "desc": "Balanced",
                    "lang": "Multilingual",
                },
                "base.en": {
                    "params": "74M",
                    "size": "~142MB",
                    "desc": "Balanced",
                    "lang": "English-only",
                },
                "small": {
                    "params": "244M",
                    "size": "~466MB",
                    "desc": "Good accuracy",
                    "lang": "Multilingual",
                },
                "small.en": {
                    "params": "244M",
                    "size": "~466MB",
                    "desc": "Good accuracy",
                    "lang": "English-only",
                },
                "medium": {
                    "params": "769M",
                    "size": "~1.5GB",
                    "desc": "Better accuracy",
                    "lang": "Multilingual",
                },
                "medium.en": {
                    "params": "769M",
                    "size": "~1.5GB",
                    "desc": "Better accuracy",
                    "lang": "English-only",
                },
                "large": {
                    "params": "1550M",
                    "size": "~3GB",
                    "desc": "Best accuracy",
                    "lang": "Multilingual",
                },
                "large-v1": {
                    "params": "1550M",
                    "size": "~3GB",
                    "desc": "Best accuracy v1",
                    "lang": "Multilingual",
                },
                "large-v2": {
                    "params": "1550M",
                    "size": "~3GB",
                    "desc": "Best accuracy v2",
                    "lang": "Multilingual",
                },
                "large-v3": {
                    "params": "1550M",
                    "size": "~3GB",
                    "desc": "Best accuracy v3",
                    "lang": "Multilingual",
                },
                "large-v3-turbo": {
                    "params": "809M",
                    "size": "~1.6GB",
                    "desc": "Fast large model",
                    "lang": "Multilingual",
                },
                "turbo": {
                    "params": "809M",
                    "size": "~1.6GB",
                    "desc": "Fastest large model",
                    "lang": "Multilingual",
                },
            }

            # Get previous models to detect removals
            old_models = (
                set(_whisper_models_cache.keys()) if _whisper_models_cache else set()
            )

            # Build models dict from discovered models
            discovered_models = {}
            new_models_found = []

            for model_name in available_models:
                # Skip if already added
                if model_name in discovered_models:
                    continue

                # Use known specs or create generic entry for new models
                if model_name in model_specs:
                    discovered_models[model_name] = model_specs[model_name]
                else:
                    # New model found - determine if it's English-only based on .en suffix
                    is_english_only = model_name.endswith(".en")
                    discovered_models[model_name] = {
                        "params": "Unknown",
                        "size": "Unknown",
                        "desc": "New model",
                        "lang": "English-only" if is_english_only else "Multilingual",
                    }
                    # Only count as new if not in old cache
                    if model_name not in old_models:
                        new_models_found.append(model_name)

            # Detect removed models
            current_models = set(discovered_models.keys())
            removed_models = list(old_models - current_models)

            # Update memory cache
            _whisper_models_cache = discovered_models
            _whisper_models_cache_time = time.time()

            # Save to file for persistence across restarts
            save_whisper_models_to_file(discovered_models)

            print(
                f"[OK] Whisper models refreshed: {len(discovered_models)} total, {len(new_models_found)} new, {len(removed_models)} removed"
            )

            return jsonify(
                {
                    "success": True,
                    "message": f"Discovered {len(discovered_models)} Whisper models",
                    "total_models": len(discovered_models),
                    "models": list(discovered_models.keys()),
                    "new_models": new_models_found,
                    "removed_models": removed_models,
                }
            )

        except ImportError:
            return jsonify(
                {"success": False, "error": "Whisper package not installed"}
            ), 500

    except Exception as e:
        print(f"[ERROR] Error refreshing Whisper models: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/remove-whisper", methods=["POST"])
def remove_whisper_model():
    """Remove a Whisper model from both old cache and new ./models directory"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        data = request.get_json()
        model_name = data.get("model_name")

        if not model_name:
            return jsonify({"success": False, "error": "Model name is required"}), 400

        files_removed = []
        dirs_removed = []

        # Check old Whisper cache directory
        whisper_cache_old = os.path.expanduser("~/.cache/whisper")
        if os.path.exists(whisper_cache_old):
            for filename in os.listdir(whisper_cache_old):
                if filename.endswith(".pt"):
                    # Check if this file matches the model name
                    base_name = filename.replace(".pt", "").replace(".en", "")
                    if base_name == model_name:
                        file_path = os.path.join(whisper_cache_old, filename)
                        try:
                            os.remove(file_path)
                            files_removed.append(filename)
                            print(
                                f"[OK] Removed Whisper model file from cache: {filename}"
                            )
                        except Exception as e:
                            print(f"[ERROR] Failed to remove {filename}: {e}")

        # Check new ./models/whisper-* directory
        models_dir = MODELS_DIR
        whisper_model_dir = os.path.join(models_dir, f"whisper-{model_name}")

        if os.path.exists(whisper_model_dir):
            try:
                import shutil

                shutil.rmtree(whisper_model_dir)
                dirs_removed.append(f"whisper-{model_name}")
                print(f"[OK] Removed Whisper model directory: {whisper_model_dir}")
            except Exception as e:
                print(f"[ERROR] Failed to remove directory {whisper_model_dir}: {e}")

        if files_removed or dirs_removed:
            message_parts = []
            if files_removed:
                message_parts.append(f"{len(files_removed)} file(s) from cache")
            if dirs_removed:
                message_parts.append(f"{len(dirs_removed)} directory from models")

            return jsonify(
                {
                    "success": True,
                    "message": f'Successfully removed Whisper model "{model_name}" ({", ".join(message_parts)})',
                    "files_removed": files_removed,
                    "dirs_removed": dirs_removed,
                }
            )
        else:
            return jsonify(
                {
                    "success": False,
                    "error": f'No files or directories found for model "{model_name}"',
                }
            ), 404

    except Exception as e:
        print(f"[ERROR] Error removing Whisper model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/faster-whisper/list", methods=["GET"])
def list_faster_whisper_models():
    """List available faster-whisper models with download status"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        models_dir = MODELS_DIR
        models_list = []

        # Use dynamic model list instead of hardcoded
        available_models = get_faster_whisper_models_list()

        for model_name, details in available_models.items():
            model_path = os.path.join(models_dir, f"faster-whisper-{model_name}")
            # Check directory exists AND contains model weight files
            downloaded = False
            if os.path.exists(model_path):
                model_files = os.listdir(model_path)
                downloaded = any(
                    f in ["model.safetensors", "pytorch_model.bin", "model.bin"] or
                    (f.startswith("pytorch_model-") and f.endswith(".bin")) or
                    (f.startswith("model-") and f.endswith(".safetensors"))
                    for f in model_files
                )

            models_list.append({
                "name": model_name,
                "repo": details["repo"],
                "size": details["size"],
                "params": details["params"],
                "lang": details["lang"],
                "downloaded": downloaded,
                "path": model_path if downloaded else None,
            })

        # Reverse order to match Whisper models (smallest first)
        models_list.reverse()

        return jsonify({
            "success": True,
            "models": models_list,
            "count": len(models_list),
        })

    except Exception as e:
        print(f"[ERROR] Error listing faster-whisper models: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/faster-whisper/download", methods=["POST"])
def download_faster_whisper_model():
    """Download a faster-whisper model from HuggingFace"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        data = request.get_json()
        model_name = data.get("model_name")

        if not model_name:
            return jsonify({"success": False, "error": "model_name required"}), 400

        # Use dynamic model list instead of hardcoded
        available_models = get_faster_whisper_models_list()

        if model_name not in available_models:
            return jsonify({"success": False, "error": f"Unknown model: {model_name}. Try refreshing the model list."}), 400

        model_info = available_models[model_name]
        repo_id = model_info["repo"]

        print(f"[DOWNLOAD] Downloading faster-whisper model: {model_name} from {repo_id}")

        models_dir = MODELS_DIR
        os.makedirs(models_dir, exist_ok=True)
        local_dir = os.path.join(models_dir, f"faster-whisper-{model_name}")

        # Best-effort total size so the UI can show a real percentage
        total_size = None
        try:
            from huggingface_hub import HfApi
            repo_info = HfApi().model_info(repo_id, files_metadata=True)
            total_size = sum(f.size or 0 for f in repo_info.siblings) or None
        except Exception as e:
            print(f"[WARNING] Could not get size of {repo_id}: {e}")

        download_key = f"faster-whisper-{model_name}"
        if not try_register_download(download_key, total=total_size):
            return jsonify({"success": False, "error": "Download already in progress"}), 409

        try:
            start_download_monitor(download_key, local_dir, total=total_size)

            # Per-file download with resume + cancellation (snapshot_download
            # can't be interrupted once started)
            outcome = download_hf_repo_files(repo_id, local_dir, download_key)
            if outcome == "cancelled":
                print(f"[CANCELLED] Download cancelled for {download_key}")
                finish_download(download_key, cancelled=True)
                return jsonify({"success": False, "message": "Download cancelled"})

            message = f"faster-whisper {model_name} downloaded to: {local_dir}"
            print(f"[OK] {message}")

            finish_download(download_key)

        except Exception as e:
            finish_download(download_key, error=e)
            raise

        return jsonify({
            "success": True,
            "message": message,
            "model_name": model_name,
            "path": local_dir,
        })

    except Exception as e:
        print(f"[ERROR] Error downloading faster-whisper model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/faster-whisper/remove", methods=["POST"])
def remove_faster_whisper_model():
    """Remove a downloaded faster-whisper model"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        data = request.get_json()
        model_name = data.get("model_name")

        if not model_name:
            return jsonify({"success": False, "error": "model_name required"}), 400

        models_dir = MODELS_DIR
        model_path = os.path.join(models_dir, f"faster-whisper-{model_name}")

        if not os.path.exists(model_path):
            return jsonify({
                "success": False,
                "error": f"Model not found: faster-whisper-{model_name}",
            }), 404

        import shutil
        shutil.rmtree(model_path)
        print(f"[OK] Removed faster-whisper model: {model_path}")

        return jsonify({
            "success": True,
            "message": f"Successfully removed faster-whisper-{model_name}",
            "model_name": model_name,
        })

    except Exception as e:
        print(f"[ERROR] Error removing faster-whisper model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/list", methods=["GET"])
def list_models():
    """List available and downloaded models"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        # Get available Whisper models (from cache or defaults)
        all_whisper_models = get_whisper_models_list()

        downloaded_whisper = []

        # Check both old Whisper cache directory and new ./models location
        whisper_cache_old = os.path.expanduser("~/.cache/whisper")
        models_dir = MODELS_DIR

        # Check old cache location for backward compatibility
        if os.path.exists(whisper_cache_old):
            for item in os.listdir(whisper_cache_old):
                if item.endswith(".pt"):
                    # Extract model name from filename (e.g., 'base.pt' -> 'base')
                    model_name = item.replace(".pt", "")
                    # Handle .en variants
                    if model_name.endswith(".en"):
                        base_name = model_name.replace(".en", "")
                        if base_name not in downloaded_whisper:
                            downloaded_whisper.append(base_name)
                    else:
                        downloaded_whisper.append(model_name)

        # Check new ./models/whisper-* directories
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                if item.startswith("whisper-"):
                    whisper_model_dir = os.path.join(models_dir, item)
                    if os.path.isdir(whisper_model_dir):
                        # Check if directory contains .pt files
                        for file in os.listdir(whisper_model_dir):
                            if file.endswith(".pt"):
                                # Extract model name from directory (e.g., 'whisper-base' -> 'base')
                                model_name = item.replace("whisper-", "")
                                if model_name not in downloaded_whisper:
                                    downloaded_whisper.append(model_name)
                                break

        # Create whisper models list with download status and details
        whisper_models = []
        for model_name, details in all_whisper_models.items():
            whisper_models.append(
                {
                    "name": model_name,
                    "downloaded": model_name in downloaded_whisper,
                    "params": details["params"],
                    "size": details["size"],
                    "desc": details["desc"],
                    "lang": details["lang"],
                }
            )

        # Get downloaded/custom models (this would scan a models directory)
        downloaded_models = []
        models_dir = MODELS_DIR
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                if os.path.isdir(os.path.join(models_dir, item)):
                    # Skip internal cache/data directories
                    if item.startswith(".") or item in ("tts", "piper"):
                        continue

                    # Detect if it's a HuggingFace model (contains --)
                    if "--" in item:
                        # HuggingFace model - convert back to original ID
                        model_id = item.replace("--", "/")
                        downloaded_models.append(
                            {
                                "name": model_id,
                                "type": "huggingface",
                                "path": os.path.join(models_dir, item),
                                "directory": item,
                            }
                        )
                    else:
                        # Local/uploaded model
                        downloaded_models.append(
                            {
                                "name": item,
                                "type": "local",
                                "path": os.path.join(models_dir, item),
                                "directory": item,
                            }
                        )

        # Add downloaded Piper TTS models
        for m in _PIPER_MODELS_CATALOG:
            if _is_piper_model_downloaded(m["id"]):
                downloaded_models.append(
                    {
                        "name": m["name"],
                        "type": "piper",
                        "path": _get_piper_model_dir(m["id"]),
                        "directory": m["id"],
                    }
                )

        return jsonify(
            {
                "success": True,
                "whisper_models": whisper_models,
                "downloaded_models": downloaded_models,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/models/remove", methods=["POST"])
def remove_model():
    """Remove a downloaded model"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        data = request.get_json()
        model_name = data.get("model_name")
        model_type = data.get("model_type")  # 'whisper', 'huggingface', 'local'

        if not model_name:
            return jsonify({"success": False, "error": "Model name is required"})

        if model_type == "whisper":
            # Can't remove built-in Whisper models
            return jsonify(
                {"success": False, "error": "Cannot remove built-in Whisper models"}
            )

        elif model_type == "huggingface":
            # Remove HuggingFace model directory
            models_dir = MODELS_DIR
            # Convert model ID (org/name) to directory name (org--name)
            model_dir_name = model_name.replace("/", "--")
            model_path = os.path.join(models_dir, model_dir_name)

            if os.path.exists(model_path):
                import shutil

                shutil.rmtree(model_path)
                return jsonify(
                    {"success": True, "message": f"Successfully removed {model_name}"}
                )
            else:
                return jsonify(
                    {"success": False, "error": f"Model {model_name} not found"}
                )

        elif model_type == "local":
            # Remove local model directory
            models_dir = MODELS_DIR
            model_path = os.path.join(models_dir, model_name)

            if os.path.exists(model_path):
                import shutil

                shutil.rmtree(model_path)
                return jsonify(
                    {"success": True, "message": f"Successfully removed {model_name}"}
                )
            else:
                return jsonify(
                    {"success": False, "error": f"Model {model_name} not found"}
                )

        else:
            return jsonify({"success": False, "error": "Invalid model type"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/models/nllb-status", methods=["GET"])
def nllb_status():
    """Check if NLLB translation model is downloaded"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        models_dir = MODELS_DIR

        # Check ALL NLLB model directories, not just 600M
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                if item.startswith("facebook--nllb-") and os.path.isdir(os.path.join(models_dir, item)):
                    nllb_path = os.path.join(models_dir, item)
                    has_model = False
                    total_size = 0
                    for root, dirs, files in os.walk(nllb_path):
                        for f in files:
                            file_path = os.path.join(root, f)
                            total_size += os.path.getsize(file_path)
                            if (f in ["model.safetensors", "pytorch_model.bin"] or
                                (f.startswith("pytorch_model-") and f.endswith(".bin")) or
                                (f.startswith("model-") and f.endswith(".safetensors"))):
                                has_model = True

                    if has_model:
                        size_gb = total_size / (1024 * 1024 * 1024)
                        model_id = item.replace("--", "/")
                        return jsonify({
                            "success": True,
                            "downloaded": True,
                            "path": nllb_path,
                            "model_id": model_id,
                            "size": f"{size_gb:.2f} GB"
                        })

        # Also check HuggingFace cache as fallback
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M")
        if os.path.exists(hf_cache):
            # Check if download is complete by looking for model files in snapshots
            snapshots_dir = os.path.join(hf_cache, "snapshots")
            if os.path.exists(snapshots_dir):
                for snapshot in os.listdir(snapshots_dir):
                    snapshot_path = os.path.join(snapshots_dir, snapshot)
                    if os.path.isdir(snapshot_path):
                        # Check for model file (single or sharded)
                        for f in os.listdir(snapshot_path):
                            if (f in ["model.safetensors", "pytorch_model.bin"] or
                                (f.startswith("pytorch_model-") and f.endswith(".bin")) or
                                (f.startswith("model-") and f.endswith(".safetensors"))):
                                # Model exists in cache, offer to move it
                                return jsonify({
                                    "success": True,
                                    "downloaded": False,
                                    "in_cache": True,
                                    "cache_path": hf_cache,
                                    "message": "Model found in HuggingFace cache. Click download to move it to ./models/"
                                })

            # Partial download exists
            return jsonify({
                "success": True,
                "downloaded": False,
                "partial": True,
                "message": "Partial download found in cache. Click download to complete."
            })

        return jsonify({
            "success": True,
            "downloaded": False,
            "message": "NLLB model not downloaded"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# Track NLLB download progress globally
nllb_download_progress = {"status": "idle", "progress": 0, "message": ""}


@app.route("/api/models/nllb-download-progress", methods=["GET"])
def nllb_download_progress_endpoint():
    """Get NLLB download progress"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global nllb_download_progress
    return jsonify({"success": True, **nllb_download_progress})


# Cache for NLLB models list
_nllb_models_cache = {"models": [], "last_updated": 0}


@app.route("/api/models/nllb-list", methods=["GET"])
def list_nllb_models():
    """List available NLLB translation models from HuggingFace"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global _nllb_models_cache
    import time

    # Check if we should use cache (valid for 1 hour)
    refresh = request.args.get("refresh", "false").lower() == "true"
    cache_valid = (time.time() - _nllb_models_cache["last_updated"]) < 3600

    if cache_valid and _nllb_models_cache["models"] and not refresh:
        models = _nllb_models_cache["models"]
    else:
        # Fetch from HuggingFace API
        try:
            import requests

            # Search for NLLB models from Facebook
            response = requests.get(
                "https://huggingface.co/api/models",
                params={
                    "search": "nllb",
                    "author": "facebook",
                    "filter": "translation",
                    "limit": 50
                },
                timeout=10
            )

            if response.status_code == 200:
                hf_models = response.json()
                models = []

                for m in hf_models:
                    model_id = m.get("modelId", "")
                    # Only include NLLB models
                    if "nllb" in model_id.lower():
                        # Determine size from model name
                        size = "Unknown"
                        if "distilled-600M" in model_id:
                            size = "~1.2 GB"
                            size_order = 1
                        elif "distilled-1.3B" in model_id:
                            size = "~2.6 GB"
                            size_order = 2
                        elif "1.3B" in model_id:
                            size = "~5.2 GB"
                            size_order = 3
                        elif "3.3B" in model_id:
                            size = "~13 GB"
                            size_order = 4
                        elif "moe" in model_id.lower():
                            size = "~17 GB"
                            size_order = 5
                        else:
                            size_order = 10

                        models.append({
                            "model_id": model_id,
                            "name": model_id.split("/")[-1],
                            "size": size,
                            "size_order": size_order,
                            "downloads": m.get("downloads", 0),
                            "likes": m.get("likes", 0),
                            "description": get_nllb_model_description(model_id)
                        })

                # Sort by size order
                models.sort(key=lambda x: x["size_order"])

                # Update cache
                _nllb_models_cache = {"models": models, "last_updated": time.time()}
            else:
                # Fallback to known models if API fails
                models = get_default_nllb_models()

        except Exception as e:
            print(f"[WARN] Failed to fetch NLLB models from HuggingFace: {e}")
            models = get_default_nllb_models()

    # Check which models are downloaded
    models_dir = MODELS_DIR
    for model in models:
        dir_name = model["model_id"].replace("/", "--")
        model_path = os.path.join(models_dir, dir_name)
        if os.path.exists(model_path):
            model_files = os.listdir(model_path)
            # Check for single model file or sharded weights (e.g. pytorch_model-00001-of-00003.bin)
            model["downloaded"] = any(
                f in ["model.safetensors", "pytorch_model.bin"] or
                (f.startswith("pytorch_model-") and f.endswith(".bin")) or
                (f.startswith("model-") and f.endswith(".safetensors"))
                for f in model_files
            )
        else:
            model["downloaded"] = False

    return jsonify({"success": True, "models": models})


def get_nllb_model_description(model_id):
    """Get description for NLLB model"""
    descriptions = {
        "facebook/nllb-200-distilled-600M": "Distilled 600M - Fast, good quality. Recommended for most users.",
        "facebook/nllb-200-distilled-1.3B": "Distilled 1.3B - Better quality, moderate speed.",
        "facebook/nllb-200-1.3B": "Full 1.3B - High quality, slower.",
        "facebook/nllb-200-3.3B": "Full 3.3B - Best quality, requires significant VRAM.",
        "facebook/nllb-moe-54b": "MoE 54B - Mixture of Experts, requires 80GB+ VRAM.",
    }
    return descriptions.get(model_id, "NLLB translation model - 200+ languages supported")


def get_default_nllb_models():
    """Return default list of known NLLB models"""
    return [
        {
            "model_id": "facebook/nllb-200-distilled-600M",
            "name": "nllb-200-distilled-600M",
            "size": "~1.2 GB",
            "size_order": 1,
            "downloads": 0,
            "likes": 0,
            "description": "Distilled 600M - Fast, good quality. Recommended for most users."
        },
        {
            "model_id": "facebook/nllb-200-distilled-1.3B",
            "name": "nllb-200-distilled-1.3B",
            "size": "~2.6 GB",
            "size_order": 2,
            "downloads": 0,
            "likes": 0,
            "description": "Distilled 1.3B - Better quality, moderate speed."
        },
        {
            "model_id": "facebook/nllb-200-1.3B",
            "name": "nllb-200-1.3B",
            "size": "~5.2 GB",
            "size_order": 3,
            "downloads": 0,
            "likes": 0,
            "description": "Full 1.3B - High quality, slower."
        },
        {
            "model_id": "facebook/nllb-200-3.3B",
            "name": "nllb-200-3.3B",
            "size": "~13 GB",
            "size_order": 4,
            "downloads": 0,
            "likes": 0,
            "description": "Full 3.3B - Best quality, requires significant VRAM."
        }
    ]


# ============== Silero VAD Status ==============
# Note: Silero VAD is now handled via pip package (silero-vad>=4.0.0)
# No separate download needed - model is bundled with the package


@app.route("/api/models/silero-vad-status", methods=["GET"])
def silero_vad_status():
    """Check if Silero VAD is available (via pip package)"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        # Check if silero-vad pip package is installed
        import importlib.util
        if importlib.util.find_spec("silero_vad") is None:
            raise ImportError("silero_vad not installed")
        return jsonify({
            "success": True,
            "downloaded": True,
            "source": "pip package (silero-vad)",
            "message": "Silero VAD available via pip package"
        })
    except ImportError:
        return jsonify({
            "success": True,
            "downloaded": False,
            "message": "Install with: pip install silero-vad"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ============== PANNs music/speech detector status + download ==============
PANNS_CKPT_SIZE = 327_000_000  # ~312 MB CNN14 checkpoint (approx, for progress %)


@app.route("/api/models/panns-status", methods=["GET"])
def panns_status():
    """Report whether the PANNs package is installed and the checkpoint downloaded."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    installed = panns_package_installed()
    ckpt = panns_checkpoint_path()
    downloaded = os.path.exists(ckpt)
    # Self-heal the AudioSet label CSV so a missing/0-byte file doesn't silently
    # break detection; then report whether labels are present.
    ensure_panns_labels_csv()
    labels_csv = panns_labels_home_path()
    labels_ok = os.path.exists(labels_csv) and os.path.getsize(labels_csv) >= _PANNS_LABELS_MIN_BYTES
    if not installed:
        msg = "panns-inference not installed. Install with: pip install panns-inference"
    elif not downloaded:
        msg = "PANNs CNN14 checkpoint not downloaded."
    elif not labels_ok:
        msg = "PANNs AudioSet labels missing; detection will not classify music."
    else:
        msg = "PANNs music/speech detector ready."
    return jsonify({
        "success": True,
        "package_installed": installed,
        "downloaded": bool(installed and downloaded and labels_ok),
        "checkpoint_path": ckpt,
        "labels_present": bool(labels_ok),
        "message": msg,
    })


@app.route("/api/models/panns/download", methods=["POST"])
def download_panns_model():
    """Download the PANNs CNN14 checkpoint in the background (progress via the
    shared download tracker, same as other model downloads)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    if not panns_package_installed():
        return jsonify({"success": False, "error": "panns-inference is not installed. Install it first: pip install panns-inference"}), 400

    dest = panns_checkpoint_path()
    if os.path.exists(dest):
        ensure_panns_labels_csv()  # checkpoint present but labels may still be missing
        return jsonify({"success": True, "message": "Checkpoint already downloaded"})

    key = "panns_cnn14"
    if not try_register_download(key, total=PANNS_CKPT_SIZE):
        return jsonify({"success": False, "error": "Download already in progress"}), 409

    def worker():
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            # The checkpoint is useless without the AudioSet label CSV, so make sure
            # it's in place too (the detector loads labels at import time).
            ensure_panns_labels_csv()
            start_download_monitor(key, dest, total=PANNS_CKPT_SIZE)
            result = download_url_to_file(
                PANNS_CHECKPOINT_URL, dest,
                cancel_check=lambda: key in cancelled_downloads,
            )
            if result == "cancelled":
                finish_download(key, cancelled=True)
                return
            finish_download(key)
            # No global reset needed: the detector runs in the worker process and
            # re-checks the checkpoint each tick, so it picks this up without a restart.
            print(f"[PANNS] Checkpoint downloaded to {dest}")
        except Exception as e:
            finish_download(key, error=e)
            print(f"[PANNS] Checkpoint download failed: {e}")

    threading.Thread(target=worker, daemon=True, name="dl-panns").start()
    return jsonify({"success": True, "message": "Download started"})


@app.route("/api/models/translation/download", methods=["POST"])
def download_translation_model():
    """Download any translation model to ./models/ directory"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global nllb_download_progress

    try:
        import threading

        data = request.get_json()
        model_id = data.get("model_id", "facebook/nllb-200-distilled-600M")

        # The status shim below is a single global, so only one translation
        # download can run at a time
        if nllb_download_progress.get("status") in ["downloading", "starting"]:
            return jsonify({"success": False, "error": "Download already in progress"})

        # Best-effort total size so progress can be a real percentage
        expected_total = None
        try:
            from huggingface_hub import HfApi
            repo_info = HfApi().model_info(model_id, files_metadata=True)
            expected_total = sum(f.size or 0 for f in repo_info.siblings) or None
        except Exception as e:
            print(f"[WARNING] Could not get size of {model_id}: {e}")

        # Atomic per-model registration in the shared download tracker
        if not try_register_download(model_id, total=expected_total):
            return jsonify({"success": False, "error": "Download already in progress"}), 409

        def download_model():
            global nllb_download_progress
            import time
            import logging

            # Set up detailed logging
            log_file = os.path.join(APP_DIR, "logs", "translation_download.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Configure file handler for this download
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

            dl_logger = logging.getLogger('translation_download')
            dl_logger.setLevel(logging.DEBUG)
            dl_logger.addHandler(file_handler)

            start_time = time.time()
            dl_logger.info("=" * 60)
            dl_logger.info(f"Starting download of {model_id}")
            dl_logger.info("=" * 60)

            try:
                models_dir = MODELS_DIR
                os.makedirs(models_dir, exist_ok=True)
                model_dir_name = model_id.replace("/", "--")
                model_path = os.path.join(models_dir, model_dir_name)

                dl_logger.info(f"Target directory: {model_path}")
                dl_logger.info(f"Checking if directory exists: {os.path.exists(model_path)}")

                nllb_download_progress = {"status": "downloading", "progress": 10, "message": f"Downloading {model_id}..."}

                # Start a background thread to monitor progress
                stop_monitor = threading.Event()

                def monitor_progress():
                    """Monitor download progress by checking file sizes"""
                    # Without this, the assignment below creates a local var and
                    # the UI never sees these updates (global doesn't inherit
                    # from the enclosing function's declaration)
                    global nllb_download_progress
                    last_size = 0
                    stall_count = 0
                    while not stop_monitor.is_set():
                        try:
                            # Calculate total size of model directory
                            total_size = 0
                            incomplete_files = []
                            for root, dirs, files in os.walk(model_path):
                                for f in files:
                                    fp = os.path.join(root, f)
                                    try:
                                        size = os.path.getsize(fp)
                                        total_size += size
                                        if f.endswith('.incomplete'):
                                            incomplete_files.append((f, size))
                                    except OSError:
                                        pass

                            size_mb = total_size / (1024 * 1024)
                            speed = (total_size - last_size) / (1024 * 1024)  # MB in last second

                            # Log progress
                            if incomplete_files:
                                for fname, fsize in incomplete_files:
                                    dl_logger.debug(f"Incomplete file: {fname[:30]}... = {fsize / (1024*1024):.1f} MB")

                            dl_logger.info(f"Progress: {size_mb:.1f} MB downloaded, speed: {speed:.2f} MB/s")

                            # Update progress for UI
                            if expected_total:
                                progress = min(99, int((total_size / expected_total) * 100))
                            else:
                                # No known total: estimate against ~2.5GB (typical NLLB)
                                progress = min(85, int(10 + (size_mb / 2500) * 75))
                            nllb_download_progress = {
                                "status": "downloading",
                                "progress": progress,
                                "message": f"Downloading: {size_mb:.0f} MB ({speed:.1f} MB/s)"
                            }

                            # Mirror into the shared download tracker (main status endpoint)
                            with active_downloads_lock:
                                entry = active_downloads.get(model_id)
                                if entry and entry.get("status") == "downloading":
                                    entry["downloaded"] = total_size
                                    entry["last_update"] = time.time()
                                    if expected_total:
                                        entry["percentage"] = min(int((total_size / expected_total) * 100), 99)

                            # Detect stalls
                            if total_size == last_size and total_size > 0:
                                stall_count += 1
                                if stall_count >= 30:  # 30 seconds of no progress
                                    dl_logger.warning(f"Download appears stalled for {stall_count} seconds!")
                            else:
                                stall_count = 0

                            last_size = total_size
                        except Exception as e:
                            dl_logger.error(f"Monitor error: {e}")

                        time.sleep(1)

                monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
                monitor_thread.start()
                dl_logger.info("Started progress monitor thread")

                # Download files using wget for reliability (huggingface_hub hangs on large files)
                dl_logger.info("Fetching file list from HuggingFace...")
                try:
                    from huggingface_hub import list_repo_files, hf_hub_url

                    # Get list of files in the repo
                    files = list_repo_files(repo_id=model_id)
                    dl_logger.info(f"Found {len(files)} files to download: {files}")

                    # Download each file using wget
                    for idx, filename in enumerate(files):
                        dest_path = os.path.join(model_path, filename)
                        os.makedirs(os.path.dirname(dest_path) if os.path.dirname(dest_path) else model_path, exist_ok=True)

                        # Skip if already downloaded and has content
                        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                            dl_logger.info(f"Already exists: {filename}")
                            continue

                        dl_logger.info(f"Downloading file {idx+1}/{len(files)}: {filename}")
                        nllb_download_progress = {
                            "status": "downloading",
                            "progress": int(10 + (idx / len(files)) * 70),
                            "message": f"Downloading {filename}..."
                        }

                        # Get download URL from HuggingFace
                        url = hf_hub_url(repo_id=model_id, filename=filename)
                        dl_logger.info(f"URL: {url}")

                        # Download with resume + retry, checking cancellation mid-file
                        outcome = download_url_to_file(
                            url, dest_path,
                            cancel_check=lambda: model_id in cancelled_downloads,
                            log=dl_logger.info,
                        )
                        if outcome == "cancelled":
                            dl_logger.info(f"Download cancelled for {model_id}")
                            nllb_download_progress = {"status": "error", "progress": 0, "message": "Download cancelled"}
                            finish_download(model_id, cancelled=True)
                            return

                        dl_logger.info(f"Successfully downloaded: {filename}")

                    dl_logger.info("All files downloaded successfully")
                except Exception as download_error:
                    dl_logger.error(f"Download failed: {type(download_error).__name__}: {download_error}")
                    raise
                finally:
                    stop_monitor.set()
                    dl_logger.info("Stopped progress monitor")

                elapsed = time.time() - start_time
                dl_logger.info(f"Download phase completed in {elapsed:.1f} seconds")

                # Post-download: check for incomplete files and finalize them
                nllb_download_progress = {"status": "downloading", "progress": 90, "message": "Finalizing download..."}
                dl_logger.info("Checking for incomplete files...")

                cache_dir = os.path.join(model_path, ".cache")
                if os.path.exists(cache_dir):
                    dl_logger.info(f"Found cache directory: {cache_dir}")
                    # Look for incomplete files that are actually complete
                    for root, dirs, files in os.walk(cache_dir):
                        for f in files:
                            if f.endswith(".incomplete"):
                                incomplete_path = os.path.join(root, f)
                                file_size = os.path.getsize(incomplete_path)
                                dl_logger.info(f"Found incomplete file: {f[:40]}... size={file_size / (1024*1024):.1f} MB")

                                # If file is large (>100MB), it's likely the model weights
                                if file_size > 100_000_000:
                                    # Check if pytorch_model.bin or model.safetensors exists
                                    model_bin = os.path.join(model_path, "pytorch_model.bin")
                                    model_safetensors = os.path.join(model_path, "model.safetensors")
                                    if not os.path.exists(model_bin) and not os.path.exists(model_safetensors):
                                        dl_logger.info("Copying incomplete file to pytorch_model.bin")
                                        import shutil
                                        shutil.copy2(incomplete_path, model_bin)
                                        dl_logger.info("Copy completed")
                    # Clean up cache
                    dl_logger.info("Cleaning up cache directory")
                    import shutil
                    shutil.rmtree(cache_dir, ignore_errors=True)
                else:
                    dl_logger.info("No cache directory found (download completed normally)")

                # Verify final state
                final_files = os.listdir(model_path)
                dl_logger.info(f"Final files in model directory: {final_files}")

                total_elapsed = time.time() - start_time
                dl_logger.info(f"Download complete! Total time: {total_elapsed:.1f} seconds")
                dl_logger.info("=" * 60)

                nllb_download_progress = {"status": "complete", "progress": 100, "message": "Download complete!"}
                finish_download(model_id)

            except Exception as e:
                import traceback
                dl_logger.error(f"Download failed: {type(e).__name__}: {e}")
                dl_logger.error(traceback.format_exc())
                nllb_download_progress = {"status": "error", "progress": 0, "message": str(e)}
                finish_download(model_id, error=e)
            finally:
                dl_logger.removeHandler(file_handler)
                file_handler.close()

        nllb_download_progress = {"status": "starting", "progress": 0, "message": "Starting download..."}
        thread = threading.Thread(target=download_model)
        thread.daemon = True
        thread.start()

        return jsonify({"success": True, "message": f"Download started for {model_id}"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/models/translation/remove", methods=["POST"])
def remove_translation_model():
    """Remove a downloaded translation model"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        import shutil
        data = request.get_json()
        model_id = data.get("model_id")

        if not model_id:
            return jsonify({"success": False, "error": "model_id is required"})

        models_dir = MODELS_DIR
        model_dir_name = model_id.replace("/", "--")
        model_path = os.path.join(models_dir, model_dir_name)

        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            return jsonify({"success": True, "message": f"Removed {model_id}"})
        else:
            return jsonify({"success": False, "error": f"Model {model_id} not found"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/models/upload-local", methods=["POST"])
def upload_local_model():
    """Upload a local model file or folder to the server"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        # Check if the post request has the file part
        if "files[]" not in request.files:
            return jsonify({"success": False, "error": "No files uploaded"}), 400

        files = request.files.getlist("files[]")
        model_name = request.form.get("model_name")

        if not model_name:
            return jsonify({"success": False, "error": "Model name is required"}), 400

        # Reject path separators / traversal so the model dir stays inside MODELS_DIR
        if not re.fullmatch(r"[\w.\- ]+", model_name) or model_name.strip(". ") == "":
            return jsonify({"success": False, "error": "Invalid model name"}), 400

        if not files or len(files) == 0:
            return jsonify({"success": False, "error": "No files selected"}), 400

        # Create models directory if it doesn't exist
        models_dir = MODELS_DIR
        os.makedirs(models_dir, exist_ok=True)

        # Create model directory
        model_path = os.path.join(models_dir, model_name)

        if os.path.exists(model_path):
            return jsonify(
                {
                    "success": False,
                    "error": f'Model "{model_name}" already exists. Please choose a different name or remove the existing model first.',
                }
            ), 400

        os.makedirs(model_path, exist_ok=True)

        # Save uploaded files
        saved_files = []
        for file in files:
            if file and file.filename:
                # Sanitize filename
                filename = os.path.basename(file.filename)
                file_path = os.path.join(model_path, filename)

                # Create subdirectories if needed
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                file.save(file_path)
                saved_files.append(filename)

        print(f"[OK] Uploaded {len(saved_files)} files for model: {model_name}")

        return jsonify(
            {
                "success": True,
                "message": f'Successfully uploaded model "{model_name}" with {len(saved_files)} file(s)',
                "model_name": model_name,
                "files_count": len(saved_files),
            }
        )

    except Exception as e:
        print(f"[ERROR] Error uploading local model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@socketio.on("connect")
def handle_connect():
    # print('Client connected')
    emit("connected", {"data": "Connected to Alexs server"})


@socketio.on("disconnect")
def handle_disconnect():
    emit("connected", {"data": "Disconnected from Alexs server"})


@socketio.on("request_all_entries")
def handle_request_all_entries():
    """Send all historical transcription entries to the requesting client only"""
    entries = get_new_entries(limit_override=0)  # 0 = no limit
    segments = [
        {
            "id": e[0], "timestamp": e[1], "text": e[2], "start": e[3], "end": e[4],
            "completed": True,
            "needs_review": bool(e[6]) if len(e) > 6 and e[6] is not None else False,
            "speech_type": e[9] if len(e) > 9 else None,
            "denied": bool(e[10]) if len(e) > 10 and e[10] is not None else False,
            "denied_reason": e[11] if len(e) > 11 else None,
        }
        for e in entries
    ]
    _attach_segment_ids(segments)
    emit("transcription_update", {
        "segments": segments,
        "in_progress_segment": None,
        "entries": [(e[1], e[2]) for e in entries],
        "in_progress": "",
        "is_running": transcription_state.get("running", False),
        "session_id": transcription_state.get("session_id"),
    })


@socketio.on("request_all_translation_entries")
def handle_request_all_translation_entries():
    """Send all historical translation entries to the requesting client only"""
    trans_config = config.get("live_translation", {})
    if not trans_config.get("enabled", False):
        # Translation is off — tell the client so the translate view can say so
        emit("translation_update", {
            "segments": [],
            "in_progress": None,
            "target_language": trans_config.get("target_language", "en"),
            "source_language": trans_config.get("source_language", "auto"),
            "enabled": False,
            "is_running": transcription_state.get("running", False),
            "session_id": transcription_state.get("session_id"),
        })
        return

    target_lang = trans_config.get("target_language", "en")
    source_lang = trans_config.get("source_language", "auto")
    if source_lang == "auto":
        source_lang = config.get("audio", {}).get("language", "en")
        if source_lang == "auto":
            source_lang = "en"

    entries = get_new_entries(limit_override=0)  # 0 = no limit
    cache = get_translation_cache()
    translated_segments = []

    for entry in entries:
        seg_id = entry[0]
        original_text = entry[2]
        cached = cache.get(seg_id, original_text, target_lang)
        if cached:
            translated_text = cached
        else:
            translated_text = translate_live_text(original_text, source_lang, target_lang)
            cache.set(seg_id, original_text, translated_text, target_lang)

        if is_whisper_hallucination(translated_text):
            continue

        translated_segments.append({
            "id": seg_id,
            "timestamp": entry[1],
            "original_text": original_text,
            "translated_text": translated_text,
            "start": entry[3],
            "end": entry[4],
            "completed": True,
            "denied": bool(entry[10]) if len(entry) > 10 and entry[10] is not None else False,
        })

    _attach_segment_ids(translated_segments)
    emit("translation_update", {
        "segments": translated_segments,
        "in_progress": None,
        "target_language": target_lang,
        "target_language_name": TRANSLATION_LANGUAGES.get(target_lang, target_lang),
        "source_language": source_lang,
        "enabled": True,
        "is_running": transcription_state.get("running", False),
        "session_id": transcription_state.get("session_id"),
    })


# =============================================================================
# Real-Time Corrections Socket.IO Handlers
# =============================================================================


@socketio.on("submit_correction")
def handle_submit_correction(data):
    """Handle correction submitted via Socket.IO"""
    if not data:
        return

    segment_id = data.get("segment_id")
    new_text = data.get("new_text", "").strip()
    correction_type = data.get("correction_type", "manual")

    if segment_id is None or not new_text:
        emit("correction_error", {"error": "segment_id and new_text are required"})
        return

    current_db_name = transcription_state.get("db_name")
    if not current_db_name or not os.path.exists(current_db_name):
        emit("correction_error", {"error": "No active database"})
        return

    try:
        with sqlite3.connect(current_db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE transcriptions
                   SET original_text = COALESCE(original_text, text),
                       text = ?,
                       corrected_by = ?
                   WHERE id = ?""",
                (new_text, correction_type, segment_id),
            )
            conn.commit()

        # Invalidate caches
        with _cache_lock:
            _db_cache["last_entries"] = []
            _db_cache["last_fetch_time"] = 0

        cache = get_translation_cache()
        cache.invalidate(segment_id)

        # Re-translate if needed
        translated_text = None
        trans_config = config.get("live_translation", {})
        if trans_config.get("enabled", False):
            target_lang = trans_config.get("target_language", "en")
            source_lang = trans_config.get("source_language", "auto")
            if source_lang == "auto":
                source_lang = config.get("audio", {}).get("language", "en")
                if source_lang == "auto":
                    source_lang = "en"
            translated_text = translate_live_text(new_text, source_lang, target_lang)
            if translated_text:
                cache.set(segment_id, new_text, translated_text, target_lang)

        # Broadcast to all clients
        socketio.emit("correction_applied", {
            "segment_id": segment_id,
            "new_text": new_text,
            "corrected_by": correction_type,
            "translated_text": translated_text,
        })

    except Exception as e:
        print(f"[CORRECTION SOCKET ERROR] {e}")
        emit("correction_error", {"error": str(e)})


@socketio.on("mark_reviewed")
def handle_mark_reviewed(data):
    """Mark segments as reviewed via Socket.IO"""
    if not data:
        return

    segment_ids = data.get("segment_ids", [])
    if not segment_ids:
        return

    try:
        with _db_lock:
            conn = _open_db_writer()
            if conn is None:
                return
            try:
                placeholders = ",".join("?" for _ in segment_ids)
                conn.execute(
                    f"UPDATE transcriptions SET needs_review = 0 WHERE id IN ({placeholders})",
                    segment_ids,
                )
                conn.commit()
            finally:
                conn.close()

        _invalidate_entries_cache()

    except Exception as e:
        print(f"[MARK REVIEWED SOCKET ERROR] {e}")


@socketio.on("submit_translation_correction")
def handle_translation_correction(data):
    """Handle correction of translated text — updates TranslationCache only"""
    if not data:
        return

    segment_id = data.get("segment_id")
    new_translated_text = data.get("new_translated_text", "").strip()

    if segment_id is None or not new_translated_text:
        emit("correction_error", {"error": "segment_id and new_translated_text are required"})
        return

    try:
        cache = get_translation_cache()
        # Get the current cache entry to preserve original text and target lang
        with cache._lock:
            entry = cache._cache.get(segment_id)
            if entry:
                entry['translated'] = new_translated_text
            else:
                # No cache entry — create one with the corrected text
                trans_config = config.get("live_translation", {})
                target_lang = trans_config.get("target_language", "en")
                cache._cache[segment_id] = {
                    'original': '',
                    'translated': new_translated_text,
                    'target_lang': target_lang,
                }

        # Broadcast to all clients
        socketio.emit("translation_correction_applied", {
            "segment_id": segment_id,
            "new_translated_text": new_translated_text,
        })

    except Exception as e:
        print(f"[TRANSLATION CORRECTION ERROR] {e}")
        emit("correction_error", {"error": str(e)})


@socketio.on("select_translation_alternative")
def handle_select_translation_alternative(data):
    """Handle selection of a translation alternative"""
    if not data:
        return

    segment_id = data.get("segment_id")
    alternative_text = data.get("alternative_text", "").strip()

    if segment_id is None or not alternative_text:
        return

    try:
        cache = get_translation_cache()
        with cache._lock:
            entry = cache._cache.get(segment_id)
            if entry:
                entry['translated'] = alternative_text

        socketio.emit("translation_correction_applied", {
            "segment_id": segment_id,
            "new_translated_text": alternative_text,
        })

    except Exception as e:
        print(f"[TRANSLATION ALT SELECT ERROR] {e}")


# =============================================================================
# Audio Streaming Socket.IO Handlers
# =============================================================================

@socketio.on("join_audio_stream")
def handle_join_audio_stream():
    """Client wants to receive live microphone audio"""
    from flask_socketio import join_room
    join_room("audio_stream")
    transcription_state["audio_stream_enabled"] = True
    emit("audio_stream_info", {
        "sample_rate": 16000,
        "channels": 1,
        "bit_depth": 16
    })


@socketio.on("leave_audio_stream")
def handle_leave_audio_stream():
    """Client no longer wants live microphone audio"""
    from flask_socketio import leave_room
    leave_room("audio_stream")


@socketio.on("join_tts_audio")
def handle_join_tts_audio():
    """Client wants to receive TTS audio for translated text"""
    from flask_socketio import join_room
    join_room("tts_audio")


@socketio.on("leave_tts_audio")
def handle_leave_tts_audio():
    """Client no longer wants TTS audio"""
    from flask_socketio import leave_room
    leave_room("tts_audio")


# =============================================================================
# Staging Buffer for Output Delay
# =============================================================================


# "Staged" segments are ordinary DB rows whose timestamp is younger than the
# configured delay — emit_new_entries tags them and withholds them from the
# live view. Approve/discard therefore operate on the DB rows directly.


def _invalidate_entries_cache():
    """Force the next emit cycle to re-read the DB."""
    with _cache_lock:
        _db_cache["last_entries"] = []
        _db_cache["last_fetch_time"] = 0


def _open_db_writer():
    """Open a short-lived writer connection to the active session database.

    The caller MUST hold ``_db_lock`` so these UI-triggered writes serialize
    against the transcription thread's persistent-connection writes instead of
    racing them. A long ``busy_timeout`` is also set as defense-in-depth so a
    concurrent writer waits rather than failing immediately with
    "database is locked". Returns ``None`` if there is no active DB.
    """
    current_db_name = transcription_state.get("db_name")
    if not current_db_name or not os.path.exists(current_db_name):
        return None
    conn = sqlite3.connect(current_db_name, timeout=30.0)
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


@socketio.on("toggle_delay")
def handle_toggle_delay(data):
    """Toggle the output delay on/off"""
    if not data:
        return
    enabled = data.get("enabled", False)
    config.setdefault("corrections", {}).setdefault("output_delay", {})["enabled"] = enabled
    save_config(config)

    # Disabling needs no flush: rows are already in the DB, and the next
    # emit cycle publishes everything once the gate is off
    socketio.emit("delay_status", {"enabled": enabled})


@socketio.on("set_delay_seconds")
def handle_set_delay_seconds(data):
    """Update the output delay duration"""
    if not data:
        return
    seconds = max(2, min(30, int(data.get("delay_seconds", 7))))
    config.setdefault("corrections", {}).setdefault("output_delay", {})["delay_seconds"] = seconds
    save_config(config)


def _backdate_staged_rows(seg_id=None):
    """Publish staged row(s) immediately by backdating their timestamp past
    the delay window. seg_id None = all rows still inside the window."""
    delay_seconds = config.get("corrections", {}).get("output_delay", {}).get("delay_seconds", 7)
    # Match the emit-time age check, which compares against datetime.now()
    backdated = (datetime.now() - timedelta(seconds=delay_seconds + 1)).strftime("%Y-%m-%d %H:%M:%S")
    try:
        with _db_lock:
            conn = _open_db_writer()
            if conn is None:
                return
            try:
                if seg_id is not None:
                    conn.execute("UPDATE transcriptions SET timestamp = ? WHERE id = ?", (backdated, int(seg_id)))
                else:
                    cutoff = (datetime.now() - timedelta(seconds=delay_seconds)).strftime("%Y-%m-%d %H:%M:%S")
                    conn.execute("UPDATE transcriptions SET timestamp = ? WHERE timestamp > ?", (backdated, cutoff))
                conn.commit()
            finally:
                conn.close()
        _invalidate_entries_cache()
    except Exception as e:
        print(f"[STAGING] Approve failed: {e}")


@socketio.on("approve_staged")
def handle_approve_staged(data):
    """Approve one or all staged segments for immediate publishing"""
    if not data:
        return
    if data.get("all", False):
        _backdate_staged_rows()
    else:
        staging_id = data.get("staging_id")
        if staging_id is not None:
            _backdate_staged_rows(staging_id)


@socketio.on("discard_staged")
def handle_discard_staged(data):
    """Discard a staged segment (delete the row before it publishes)"""
    if not data:
        return
    staging_id = data.get("staging_id")
    if staging_id is None:
        return
    try:
        with _db_lock:
            conn = _open_db_writer()
            if conn is None:
                return
            try:
                conn.execute("DELETE FROM transcriptions WHERE id = ?", (int(staging_id),))
                conn.commit()
            finally:
                conn.close()
        _invalidate_entries_cache()
    except Exception as e:
        print(f"[STAGING] Discard failed: {e}")


@socketio.on("set_segment_denied")
def handle_set_segment_denied(data):
    """Toggle the 'denied' flag on one or more segments.

    Denied segments are hidden from the output display (index.html filters them
    client-side) but kept in the DB and still shown — struck-through — on the
    corrections page so they can be restored. Broadcasts the new state to every
    connected client so open displays update live without a reload.
    """
    if not data:
        return

    segment_ids = data.get("segment_ids", [])
    if not segment_ids:
        return
    denied_val = 1 if data.get("denied", True) else 0

    try:
        with _db_lock:
            conn = _open_db_writer()
            if conn is None:
                return
            try:
                placeholders = ",".join("?" for _ in segment_ids)
                conn.execute(
                    f"UPDATE transcriptions SET denied = ? WHERE id IN ({placeholders})",
                    [denied_val, *segment_ids],
                )
                conn.commit()
            finally:
                conn.close()

        _invalidate_entries_cache()

        # Broadcast to all clients (no room) so open index/corrections pages react live
        socketio.emit("segment_denied", {"segment_ids": segment_ids, "denied": bool(denied_val)})
    except Exception as e:
        print(f"[DENY] set_segment_denied failed: {e}")


def get_new_entries(limit_override=None):
    """Get recent transcriptions with caching and efficient querying

    Args:
        limit_override: Optional limit to override database.max_entries_to_send config

    Returns all rows including denied ones — callers that should hide denied entries
    (live preview, translation) must filter on entry[10] (denied flag) themselves.
    """
    global _db_cache

    # Get database name from shared transcription state
    current_db_name = transcription_state.get("db_name")

    # Debug logging (uncomment to trace issues)
    # import sys; print(f"[GET_ENTRIES] db_name={current_db_name}, exists={os.path.exists(current_db_name) if current_db_name else 'N/A'}", file=sys.stderr, flush=True)

    # If database not initialized yet, return empty list
    if current_db_name is None:
        return []

    # Check if database file exists
    if not os.path.exists(current_db_name):
        return []

    current_time = time.time()

    # Check cache first (only use cache if no limit_override, since cache is shared)
    if limit_override is None:
        with _cache_lock:
            if (
                current_time - _db_cache["last_fetch_time"] < _db_cache["cache_duration"]
                and _db_cache["last_entries"]
            ):
                return _db_cache["last_entries"]

    try:
        limit = limit_override if limit_override is not None else config.get("database", {}).get("max_entries_to_send", 100)

        # Use context manager for database connection
        with sqlite3.connect(current_db_name) as conn:
            cursor = conn.cursor()
            if limit <= 0:
                # 0 or negative means no limit — return all entries
                cursor.execute(
                    """
                    SELECT id, timestamp, text, COALESCE(start_time, 0) as start_time, COALESCE(end_time, 0) as end_time,
                           confidence, needs_review, translated_text, translation_language, speech_type,
                           COALESCE(denied, 0), denied_reason, music_prob
                    FROM transcriptions
                    WHERE timestamp != '' AND TRIM(text) != ''
                    ORDER BY id ASC
                """
                )
            else:
                # FIX: Get the LATEST N entries, not the oldest N
                # Use subquery to get last N entries by id DESC, then re-order ASC for display
                cursor.execute(
                    """
                    SELECT id, timestamp, text, start_time, end_time, confidence, needs_review, translated_text, translation_language, speech_type, denied, denied_reason, music_prob FROM (
                        SELECT id, timestamp, text, COALESCE(start_time, 0) as start_time, COALESCE(end_time, 0) as end_time,
                               confidence, needs_review, translated_text, translation_language, speech_type,
                               COALESCE(denied, 0) as denied, denied_reason, music_prob
                        FROM transcriptions
                        WHERE timestamp != '' AND TRIM(text) != ''
                        ORDER BY id DESC
                        LIMIT ?
                    ) ORDER BY id ASC
                """,
                    (limit,),
                )
            transcriptions = cursor.fetchall()

        # Update cache only if using default limit (not override)
        if limit_override is None:
            with _cache_lock:
                _db_cache["last_entries"] = transcriptions
                _db_cache["last_fetch_time"] = current_time

        return transcriptions
    except Exception as e:
        print(f"[ERROR] Failed to fetch transcriptions: {e}")
        return []


def _attach_segment_ids(segments):
    """Add segment_id (string form of the db id) to each emitted segment so the
    socket key matches the db.segment_id TEXT column exactly. None when there is no
    id (e.g. a not-yet-persisted in_progress segment)."""
    for s in segments:
        if isinstance(s, dict):
            sid = s.get("id")
            s["segment_id"] = str(sid) if sid is not None else None
    return segments


def emit_new_entries():
    """Emit combined transcription updates and audio levels to web clients"""
    update_interval = config.get("web_server", {}).get("update_interval", 0.5)
    while True:
        # Check if transcription is running - if not, send empty data to clear display
        is_running = transcription_state.get("running", False)

        if not is_running:
            # Send empty data when stopped so frontend clears the display
            entries = []
            in_progress = ""
            in_progress_start = 0
            in_progress_end = 0
        else:
            # Get finalized entries from database (now includes id, timestamp, text, start_time, end_time)
            entries = get_new_entries()
            # Get in-progress text (not yet saved to DB)
            in_progress = transcription_state.get("live_text", "")
            in_progress_start = transcription_state.get("live_start", 0)
            in_progress_end = transcription_state.get("live_end", 0)


        # Convert entries to segment format with temporal data
        # entries format: (id, timestamp, text, start_time, end_time, confidence, needs_review, translated_text, translation_language, speech_type, denied, denied_reason, music_prob)
        segments = []
        for entry in entries:
            seg = {
                "id": entry[0],
                "timestamp": entry[1],
                "text": entry[2],
                "start": entry[3],
                "end": entry[4],
                "completed": True,
            }
            # Include confidence data if available (new columns may be None for old DBs)
            if len(entry) > 5:
                seg["confidence"] = entry[5]
                seg["needs_review"] = bool(entry[6]) if entry[6] is not None else False
            if len(entry) > 9:
                seg["speech_type"] = entry[9]
            if len(entry) > 10:
                seg["denied"] = bool(entry[10]) if entry[10] is not None else False
            if len(entry) > 11:
                seg["denied_reason"] = entry[11]
            if len(entry) > 12:
                seg["music_prob"] = entry[12]
            segments.append(seg)

        # Stable key matching db.segment_id (TEXT). Done before the output-delay split
        # below so live + staged segments (same dict refs) both carry it.
        _attach_segment_ids(segments)

        # Build in-progress segment if there's text
        in_progress_segment = None
        if in_progress and in_progress.strip():
            in_progress_segment = {
                "text": in_progress,
                "start": in_progress_start,
                "end": in_progress_end,
                "completed": False,
                "segment_id": None,  # not yet persisted — no db row/segment_id
            }
            # Include word-level confidence for in-progress text
            live_words = transcription_state.get("live_word_confidences")
            if live_words:
                in_progress_segment["word_confidences"] = list(live_words) if hasattr(live_words, '__iter__') else []

        # Split segments into live and staged based on output delay setting
        delay_config = config.get("corrections", {}).get("output_delay", {})
        delay_enabled = delay_config.get("enabled", False)
        delay_seconds = delay_config.get("delay_seconds", 7)
        staged_segments = []

        if delay_enabled and segments:
            now_ts = datetime.now()
            live_segments = []
            for seg in segments:
                # Parse segment timestamp to check age
                try:
                    seg_ts = datetime.strptime(seg["timestamp"], "%Y-%m-%d %H:%M:%S")
                    age = (now_ts - seg_ts).total_seconds()
                    if age < delay_seconds:
                        # Still in delay window — staged
                        seg["staged"] = True
                        seg["delay_remaining"] = max(0, delay_seconds - age)
                        staged_segments.append(seg)
                    else:
                        live_segments.append(seg)
                except (ValueError, KeyError):
                    live_segments.append(seg)
            segments = live_segments

        # Emit single unified transcription update with new format
        emit_data = {
            "segments": segments,  # [{id, timestamp, text, start, end, completed}, ...]
            "in_progress_segment": in_progress_segment,  # {text, start, end, completed} or null
            # Keep backward compatibility with old format
            "entries": [(e[1], e[2]) for e in entries],  # [(timestamp, text), ...]
            "in_progress": in_progress,  # Current incomplete text or ""
            "is_running": is_running,
            "audio_type": transcription_state.get("audio_type"),  # "Speaking", "Music", or "Quiet"
            "detection_mode": transcription_state.get("detection_mode"),  # "panns" or "energy"
            "session_id": transcription_state.get("session_id"),  # stable per-session anchor
        }
        if staged_segments:
            emit_data["staged_segments"] = staged_segments
        emit_data["delay_seconds"] = delay_seconds
        if delay_enabled:
            emit_data["delay_enabled"] = True
        socketio.emit("transcription_update", emit_data)

        # Emit audio level only if transcription is running
        is_running = transcription_state.get("running", False)
        if is_running:
            audio_level = transcription_state.get("audio_level")
            audio_db = transcription_state.get("audio_db")
            audio_energy = transcription_state.get("audio_energy")

            if audio_level is not None and audio_db is not None:
                try:
                    socketio.emit(
                        "audio_level",
                        {
                            "level": audio_level,
                            "db": audio_db,
                            "energy": audio_energy if audio_energy is not None else 0,
                            "audio_type": transcription_state.get("audio_type"),
                            "detection_mode": transcription_state.get("detection_mode"),
                            "audio_tag": transcription_state.get("audio_tag"),
                            "music_prob": transcription_state.get("music_prob"),
                        },
                    )
                except Exception as emit_error:
                    print(f"[AUDIO-DEBUG] {time.strftime('%H:%M:%S')} - EMIT FAILED: {emit_error}", flush=True)

        try:
            socketio.sleep(update_interval)  # Emit updates based on config
        except Exception as sleep_error:
            print(f"[AUDIO-DEBUG] {time.strftime('%H:%M:%S')} - SLEEP FAILED: {sleep_error}", flush=True)


def _translate_via_remote(text, source_lang, target_lang, endpoint,
                          return_extras=False, num_alternatives=0, generation_params=None):
    """Send text to a remote machine's /api/translate endpoint."""
    import requests as _requests
    try:
        payload = {
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "return_extras": return_extras,
            "num_alternatives": num_alternatives,
        }
        if generation_params:
            payload["generation_params"] = generation_params
        resp = _requests.post(
            endpoint.rstrip("/") + "/api/translate",
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if return_extras:
            return {
                "text": data.get("translated_text", text),
                "confidence": data.get("confidence"),
                "alternatives": data.get("alternatives", []),
            }
        return data.get("translated_text", text)
    except Exception as e:
        print(f"[REMOTE_TRANSLATE] Failed: {e}")
        if return_extras:
            return {"text": text, "confidence": None, "alternatives": []}
        return text


def translate_live_text(text, source_lang, target_lang, return_extras=False, num_alternatives=0, generation_params=None):
    """Translate text for live display using the singleton model"""
    # Get generation params: explicit param > config fallback
    gen_params = generation_params or config.get("live_translation", {}).get("generation_params", {})

    # Route to remote translation server if configured
    # Skip remote offload if this machine is itself serving remote clients (prevents chaining)
    remote_cfg = config.get("live_translation", {}).get("remote", {})
    if remote_cfg.get("enabled") and remote_cfg.get("endpoint") and not _trusted_translation_clients:
        try:
            _remote_ep = _get_remote_endpoint()
        except _RemoteEndpointError as e:
            print(f"[REMOTE_TRANSLATE] Endpoint error: {e}")
            _remote_ep = None
        if _remote_ep:
            return _translate_via_remote(text, source_lang, target_lang, _remote_ep,
                                         return_extras=return_extras, num_alternatives=num_alternatives,
                                         generation_params=gen_params)

    if not text or not text.strip():
        if return_extras:
            return {"text": "", "confidence": None, "alternatives": []}
        return ""

    try:
        trans_use_gpu = config.get("live_translation", {}).get("use_gpu", True)
        trans_model_id = config.get("live_translation", {}).get("translation_model")
        model, tokenizer = get_live_translation_model(trans_use_gpu, model_id=trans_model_id)
        if model is None:
            if return_extras:
                return {"text": text, "confidence": None, "alternatives": []}
            return text

        result = translate_text(
            text, source_lang, target_lang, model, tokenizer,
            return_confidence=return_extras,
            num_alternatives=num_alternatives if return_extras else 0,
            generation_params=gen_params,
        )
        return result
    except Exception as e:
        print(f"[LIVE-TRANSLATION ERROR] {e}")
        if return_extras:
            return {"text": text, "confidence": None, "alternatives": []}
        return text  # Return original on error


def emit_translated_entries():
    """Background task that emits translated transcription updates"""
    update_interval = config.get("web_server", {}).get("update_interval", 0.5)

    while True:
        trans_config = config.get("live_translation", {})
        if not trans_config.get("enabled", False):
            # Translation is off — emit a disabled marker so the translate view
            # can show "Translation disabled" instead of being stuck on "Waiting..."
            socketio.emit("translation_update", {
                "segments": [],
                "in_progress": None,
                "target_language": trans_config.get("target_language", "en"),
                "source_language": trans_config.get("source_language", "auto"),
                "enabled": False,
                "is_running": transcription_state.get("running", False),
                "model_loaded": is_live_translation_ready(),
                "model_loading": _live_translation_model_loading,
                "session_id": transcription_state.get("session_id"),
            })
            # Sleep longer when translation is disabled
            socketio.sleep(update_interval * 2)
            continue

        try:
            is_running = transcription_state.get("running", False)
            if not is_running:
                # Send empty data when stopped
                socketio.emit("translation_update", {
                    "segments": [],
                    "in_progress": None,
                    "target_language": trans_config.get("target_language", "en"),
                    "source_language": trans_config.get("source_language", "auto"),
                    "enabled": True,
                    "is_running": False,
                    "model_loaded": is_live_translation_ready(),
                    "model_loading": _live_translation_model_loading,
                    "session_id": transcription_state.get("session_id"),
                })
                socketio.sleep(update_interval)
                continue

            target_lang = trans_config.get("target_language", "en")
            source_lang = trans_config.get("source_language", "auto")

            # Resolve "auto" source language to actual language
            if source_lang == "auto":
                source_lang = config.get("audio", {}).get("language", "en")
                if source_lang == "auto":
                    source_lang = "en"  # Default fallback

            # Get finalized entries from database (use translation-specific limit)
            translation_limit = trans_config.get("max_entries_to_send")
            entries = get_new_entries(limit_override=translation_limit)
            cache = get_translation_cache()
            translated_segments = []

            # Check if Whisper-based translation is active (translations already cached by transcription loop)
            _translation_method = trans_config.get("translation_method", "nllb")
            _whisper_translation_active = _translation_method in ("whisper_translate", "whisper_forced_lang")

            # Check if corrections features are enabled for translation confidence
            corrections_cfg = config.get("corrections", {})
            want_confidence = corrections_cfg.get("enabled", True) and corrections_cfg.get("confidence_highlighting", True)
            n_alternatives = corrections_cfg.get("n_best_alternatives", {}).get("translation_count", 3) if corrections_cfg.get("enabled", True) else 0

            # Max 5: beyond that the combined NLLB input approaches the 1024-token truncation
            context_window = max(1, min(5, int(trans_config.get("context_window", 1) or 1)))

            max_translations_per_cycle = 3  # Limit new translations per cycle so cached segments emit fast
            translations_this_cycle = 0
            for idx, entry in enumerate(e for e in entries if not e[10]):
                seg_id = entry[0]
                original_text = entry[2]

                # Whisper-based translation: translations saved to DB by subprocess
                # (subprocess cache is in separate memory, so read from DB instead)
                if _whisper_translation_active:
                    cached = cache.get(seg_id, "", target_lang)
                    if not cached:
                        cached = cache.get(seg_id, "", target_lang, accept_stale_lang=True)
                    if not cached and len(entry) > 7 and entry[7]:
                        # Read translation from DB (written by subprocess)
                        cached = entry[7]
                        cache.set(seg_id, "", cached, entry[8] or target_lang)
                    translated_text = cached if cached else original_text
                    extras = None
                    seg_data = {
                        "id": seg_id,
                        "timestamp": entry[1],
                        "original_text": original_text,
                        "translated_text": translated_text,
                        "start": entry[3],
                        "end": entry[4],
                        "completed": True,
                    }
                    if not is_whisper_hallucination(translated_text):
                        translated_segments.append(seg_data)
                    continue

                # Check cache first (exact language match)
                cached = cache.get(seg_id, original_text, target_lang)
                if cached:
                    translated_text = cached
                    # Get cached extras (confidence, alternatives)
                    extras = cache.get_extras(seg_id) if want_confidence else None
                else:
                    # After a hot language switch, keep old translations for already-translated segments
                    # instead of retranslating everything — only new segments get the new language
                    stale_cached = cache.get(seg_id, original_text, target_lang, accept_stale_lang=True)
                    if stale_cached:
                        translated_text = stale_cached
                        extras = cache.get_extras(seg_id) if want_confidence else None
                        seg_data = {
                            "id": seg_id,
                            "timestamp": entry[1],
                            "original_text": original_text,
                            "translated_text": translated_text,
                            "start": entry[3],
                            "end": entry[4],
                            "completed": True,
                        }
                        if extras:
                            seg_data["confidence"] = extras.get("confidence")
                            seg_data["alternatives"] = extras.get("alternatives", [])
                        if not is_whisper_hallucination(translated_text):
                            translated_segments.append(seg_data)
                        continue
                    # Cache cold (e.g. server restart): seed from DB if it has any translation
                    # and skip live retranslation, same as stale-lang cache hit.
                    if len(entry) > 7 and entry[7]:
                        db_translation = entry[7]
                        db_lang = entry[8] if len(entry) > 8 and entry[8] else target_lang
                        cache.set(seg_id, original_text, db_translation, db_lang)
                        if not is_whisper_hallucination(db_translation):
                            translated_segments.append({
                                "id": seg_id,
                                "timestamp": entry[1],
                                "original_text": original_text,
                                "translated_text": db_translation,
                                "start": entry[3],
                                "end": entry[4],
                                "completed": True,
                            })
                        continue
                    # Build context from preceding segments if context_window > 1.
                    # The combined (context + target) text is translated in one call, then the
                    # target's portion is extracted by sentence-count alignment. If alignment
                    # fails (translator merged sentences), fall back to translating the target
                    # alone - never emit the combined translation.
                    text_to_translate = original_text
                    num_ctx_sentences = 0
                    ctx_char_ratio = None
                    if context_window > 1 and idx > 0:
                        ctx_start = max(0, idx - (context_window - 1))
                        context_texts = [entries[j][2] for j in range(ctx_start, idx)]
                        if context_texts:
                            context_prefix = " ".join(context_texts)
                            num_ctx_sentences = count_sentence_units(context_prefix)
                            text_to_translate = context_prefix + " " + original_text
                            # Context share of the source — guides the proportional
                            # split when the translator merges sentences
                            ctx_char_ratio = (len(context_prefix) + 1) / max(1, len(text_to_translate))

                    # Translate with confidence/alternatives if corrections enabled
                    if want_confidence or n_alternatives > 0:
                        result = translate_live_text(
                            text_to_translate, source_lang, target_lang,
                            return_extras=True, num_alternatives=n_alternatives,
                        )
                        if num_ctx_sentences:
                            extracted = extract_context_translation(result.get("text", ""), num_ctx_sentences, ctx_char_ratio)
                            if extracted:
                                result["text"] = extracted
                                result["alternatives"] = [
                                    alt_extracted for alt_extracted in (
                                        extract_context_translation(a, num_ctx_sentences, ctx_char_ratio)
                                        for a in result.get("alternatives", [])
                                    ) if alt_extracted
                                ]
                            else:
                                # Alignment failed - retranslate without context
                                result = translate_live_text(
                                    original_text, source_lang, target_lang,
                                    return_extras=True, num_alternatives=n_alternatives,
                                )
                        translated_text = result["text"]
                        extras = {"confidence": result.get("confidence"), "alternatives": result.get("alternatives", [])}
                    else:
                        translated_text = translate_live_text(text_to_translate, source_lang, target_lang)
                        if num_ctx_sentences and isinstance(translated_text, str):
                            extracted = extract_context_translation(translated_text, num_ctx_sentences, ctx_char_ratio)
                            translated_text = extracted if extracted else translate_live_text(original_text, source_lang, target_lang)
                        extras = None

                    # Warmup guard: while the local model is still loading, translate_live_text
                    # returns the source unchanged. Don't cache/persist that echo — leave the row
                    # NULL so it retries next cycle and translates correctly once the model is up.
                    if not is_live_translation_ready():
                        continue
                    if extras is not None:
                        cache.set_with_extras(seg_id, original_text, translated_text, target_lang,
                                              confidence=extras.get("confidence"), alternatives=extras.get("alternatives", []))
                    else:
                        cache.set(seg_id, original_text, translated_text, target_lang)

                    # Save translation to database
                    try:
                        current_db = transcription_state.get("db_name")
                        if current_db and os.path.exists(current_db):
                            with sqlite3.connect(current_db) as _tconn:
                                _tconn.execute(
                                    "UPDATE transcriptions SET translated_text = ?, translation_language = ? WHERE id = ?",
                                    (translated_text, target_lang, seg_id),
                                )
                                _tconn.commit()
                    except Exception as e:
                        # Translation still shows from cache, but won't survive a
                        # page reload — surface the reason instead of hiding it
                        print(f"[TRANSLATION] DB save failed for segment {seg_id}: {e}", flush=True)

                    # Limit new translations per cycle so cached segments emit quickly on page load
                    translations_this_cycle += 1
                    if translations_this_cycle >= max_translations_per_cycle:
                        break

                # Skip known Whisper hallucinations in translated text
                if is_whisper_hallucination(translated_text):
                    print(f"[SKIP HALLUCINATION-TRANSLATION] '{translated_text[:40]}'", flush=True)
                    continue

                seg_data = {
                    "id": seg_id,
                    "timestamp": entry[1],
                    "original_text": original_text,
                    "translated_text": translated_text,
                    "start": entry[3],
                    "end": entry[4],
                    "completed": True,
                }
                if extras:
                    seg_data["confidence"] = extras.get("confidence")
                    seg_data["alternatives"] = extras.get("alternatives", [])
                translated_segments.append(seg_data)

            # Always send in-progress text; is_translated tells the frontend whether
            # it's in the target language (so it can suppress source-language flash).
            in_progress_translation = None
            in_progress = transcription_state.get("live_text", "")
            if in_progress and in_progress.strip():
                should_translate_ip = trans_config.get("translate_in_progress", False) and not _whisper_translation_active
                if should_translate_ip:
                    translated_in_progress = translate_live_text(in_progress, source_lang, target_lang)
                else:
                    translated_in_progress = in_progress  # untranslated; frontend filters by is_translated
                if not is_whisper_hallucination(translated_in_progress):
                    in_progress_translation = {
                        "original_text": in_progress,
                        "translated_text": translated_in_progress,
                        "is_translated": should_translate_ip,
                        "start": transcription_state.get("live_start", 0),
                        "end": transcription_state.get("live_end", 0),
                        "completed": False,
                        "segment_id": None,  # not yet persisted — no db row/segment_id
                    }

            # Tag denied state per segment (so the output can hide denied rows and the
            # corrections page can show them struck-through). Built from the DB rows above.
            _denied_by_id = {
                e[0]: (bool(e[10]) if len(e) > 10 and e[10] is not None else False)
                for e in entries
            }
            for _seg in translated_segments:
                _seg["denied"] = _denied_by_id.get(_seg["id"], False)
            _attach_segment_ids(translated_segments)

            # Emit translation update
            socketio.emit("translation_update", {
                "segments": translated_segments,
                "in_progress": in_progress_translation,
                "target_language": target_lang,
                "target_language_name": TRANSLATION_LANGUAGES.get(target_lang, target_lang),
                "source_language": source_lang,
                "enabled": True,
                "is_running": is_running,
                "model_loaded": is_live_translation_ready(),
                "model_loading": _live_translation_model_loading,
                "session_id": transcription_state.get("session_id"),
            })

        except Exception as e:
            print(f"[LIVE-TRANSLATION EMIT ERROR] {e}")
            import traceback
            traceback.print_exc()

        socketio.sleep(update_interval)


# =============================================================================
# Audio Streaming Background Tasks
# =============================================================================

def emit_audio_stream():
    """Background task that streams live audio chunks to web clients"""
    while True:
        try:
            data = audio_stream_queue.get(timeout=0.5)
            socketio.emit("audio_chunk", data, room="audio_stream")
        except Empty:
            pass  # No audio queued within the timeout window
        except Exception as e:
            print(f"[AUDIO-STREAM] emit error: {e}", flush=True)


_tts_last_spoken_id = 0

def emit_tts_audio():
    """Background task that synthesizes speech from translated text and emits audio.
    Buffers segments until a sentence-ending punctuation is found so TTS speaks
    complete phrases rather than tiny fragments."""
    global _tts_last_spoken_id
    import base64

    _tts_buffer = []  # Buffered segments waiting for sentence end
    _tts_buffer_last_update = 0  # Timestamp of last buffer addition
    _tts_was_off = True  # Start as off so first enable skips existing segments

    while True:
        tts_config = config.get("live_translation", {}).get("tts", {})
        trans_config = config.get("live_translation", {})

        if not tts_config.get("enabled", False) or not trans_config.get("enabled", False):
            _tts_buffer.clear()
            # Mark that TTS is off so we can skip existing segments when re-enabled
            _tts_was_off = True
            socketio.sleep(1)
            continue

        # When TTS is first enabled mid-session, skip to the latest segment
        # so we don't replay everything from the beginning
        if _tts_was_off:
            _tts_was_off = False
            max_id = get_translation_cache().max_segment_id()
            if max_id > _tts_last_spoken_id:
                _tts_last_spoken_id = max_id

        if not transcription_state.get("running", False):
            _tts_last_spoken_id = 0
            _tts_buffer.clear()
            socketio.sleep(1)
            continue

        try:
            target_lang = trans_config.get("target_language", "en")
            cache = get_translation_cache()

            # Get segments that have been translated but not yet spoken
            new_segments = []
            with cache._lock:
                for seg_id, entry in cache._cache.items():
                    if isinstance(seg_id, int) and seg_id > _tts_last_spoken_id:
                        translated = entry.get("translated", "")
                        if translated and translated.strip():
                            new_segments.append({
                                "id": seg_id,
                                "translated_text": translated,
                            })

            # Sort by ID to speak in order
            new_segments.sort(key=lambda s: s["id"])

            # Add new segments to buffer
            for segment in new_segments:
                _tts_buffer.append(segment)
                _tts_last_spoken_id = segment["id"]
                _tts_buffer_last_update = time.time()

            # Check if buffer has a complete phrase to speak
            # Speak when: buffer ends with sentence punctuation, or buffer has been waiting too long (flush timeout)
            if _tts_buffer:
                combined_text = " ".join(s["translated_text"] for s in _tts_buffer).strip()
                last_char = combined_text.rstrip()[-1] if combined_text.rstrip() else ""
                sentence_complete = last_char in ".!?;:。！？"
                flush_timeout = time.time() - _tts_buffer_last_update > 4.0  # Flush after 4s of no new segments

                if sentence_complete or flush_timeout:
                    # Check if TTS is still enabled
                    if not config.get("live_translation", {}).get("tts", {}).get("enabled", False):
                        _tts_buffer.clear()
                        continue

                    try:
                        audio_bytes, sample_rate = synthesize_tts(combined_text, language=target_lang)
                    except Exception as synth_err:
                        # Drop the buffered text: retrying the same input every
                        # cycle would loop forever on a persistent failure
                        print(f"[TTS] Synthesis failed, dropping buffered text: {synth_err}")
                        _tts_buffer.clear()
                        continue
                    if audio_bytes:
                        backend = _get_tts_backend()
                        audio_format = "mp3" if backend == "edge" else "wav"
                        audio_b64 = base64.b64encode(audio_bytes).decode('ascii')
                        socketio.emit("tts_audio", {
                            "segment_id": _tts_buffer[-1]["id"],
                            "audio": audio_b64,
                            "format": audio_format,
                            "sample_rate": sample_rate,
                            "text": combined_text,
                        }, room="tts_audio")

                    _tts_buffer.clear()

        except Exception as e:
            print(f"[TTS EMIT ERROR] {e}")

        socketio.sleep(0.5)


def filter_hallucinated_text(text, language="auto"):
    """
    Filter out hallucinated foreign characters when transcribing specific languages.
    Whisper sometimes hallucinates random Chinese/Japanese/Korean characters
    in the middle of English transcriptions.

    Args:
        text: The transcribed text
        language: The target language code (e.g., "en", "auto")

    Returns:
        Filtered text with foreign characters removed if language is specific
    """
    import re
    if not text:
        return text

    # Remove CJK (Chinese, Japanese, Korean) characters
    # Unicode ranges:
    # - CJK Unified Ideographs: \u4e00-\u9fff
    # - Hiragana: \u3040-\u309f
    # - Katakana: \u30a0-\u30ff
    # - Hangul: \uac00-\ud7af
    cjk_pattern = r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+'

    filtered = re.sub(cjk_pattern, '', text)

    # Clean up any double spaces left behind
    filtered = re.sub(r'\s+', ' ', filtered).strip()

    if filtered != text:
        print(f"[FILTER] Removed hallucinated characters: '{text}' -> '{filtered}'")

    return filtered


# Default Whisper hallucinations - phantom subtitle credits from training data
# These are common phrases Whisper hallucinates from YouTube videos in its training set
# Matching is substring-based (phrase anywhere in the sentence, case/punctuation insensitive),
# so one entry catches all surface variants (e.g. "DimaTorzok" catches all 3 Russian credit lines).
# User can override/extend this list via config.json hallucination_filter.phrases
DEFAULT_WHISPER_HALLUCINATIONS = [
    "DimaTorzok",           # catches all Субтитры * DimaTorzok variants
    "Продолжение следует",
    "for watching",         # catches "thank you for watching", "thanks for watching", etc.
    "Please subscribe",
    "Like and subscribe",
    "Don't forget to subscribe",
]


def get_hallucination_phrases():
    """Get hallucination phrases from config, falling back to defaults."""
    hallucination_config = config.get("hallucination_filter", {})
    if not hallucination_config.get("enabled", True):
        return []  # Filter disabled
    return hallucination_config.get("phrases", DEFAULT_WHISPER_HALLUCINATIONS)


def normalize_for_hallucination_check(text):
    """Normalize text for hallucination comparison: lowercase, remove apostrophes and punctuation."""
    if not text:
        return ""
    import re
    # Lowercase and strip
    normalized = text.strip().lower()
    # Remove apostrophes (both straight and curly)
    normalized = normalized.replace("'", "").replace("'", "").replace("'", "")
    # Remove punctuation but keep spaces and alphanumeric
    normalized = re.sub(r'[^\w\s]', '', normalized)
    # Normalize whitespace
    normalized = ' '.join(normalized.split())
    return normalized


def apply_profanity_filter(text):
    """Replace broadcast-forbidden words with **** (or configured replacement)."""
    import re
    if not text:
        return text
    cfg = config.get("profanity_filter", {})
    if not cfg.get("enabled", False):
        return text
    words = cfg.get("words", [])
    if not words:
        return text
    replacement = cfg.get("replacement", "****")
    pattern = r'\b(' + '|'.join(re.escape(w) for w in words) + r')\b'
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)


def is_whisper_hallucination(text):
    """Check if text is a known Whisper hallucination (exact phrase match, case/punctuation insensitive)."""
    if not text:
        return False

    phrases = get_hallucination_phrases()
    if not phrases:
        return False  # Filter disabled or empty

    text_normalized = normalize_for_hallucination_check(text)
    for hallucination in phrases:
        if normalize_for_hallucination_check(hallucination) in text_normalized:
            return True
    return False


def split_into_sentences(text):
    """
    Split text into complete sentences and remainder.

    Returns:
        tuple: (list of complete sentences, remaining incomplete text)
    """
    import re
    if not text:
        return [], ""

    sentences = []
    remainder = text.strip()

    # Pattern: sentence ending with . ! ? followed by space or end
    # Also handles multiple punctuation like "..." or "!?"
    pattern = r'^(.*?[.!?]+)(?:\s+|$)'

    while remainder:
        match = re.match(pattern, remainder)
        if match:
            sentence = match.group(1).strip()
            if sentence:  # Only add non-empty sentences
                sentences.append(sentence)
            remainder = remainder[match.end():].strip()
        else:
            break  # No more complete sentences

    return sentences, remainder


def count_sentence_units(text):
    """Count sentence units in text (a trailing incomplete fragment counts as one)."""
    sentences, remainder = split_into_sentences(text)
    return len(sentences) + (1 if remainder else 0)


_ctx_align_stats = {"exact": 0, "proportional": 0, "failed": 0}


def extract_context_translation(combined_translated, num_context_sentences, source_char_ratio=None):
    """
    Extract the target portion from a translation of (context + target) text.

    Splits the combined translation into sentences and drops the first
    num_context_sentences. When the translator merged sentences (fewer output
    units than context units) and source_char_ratio is given (context chars /
    total source chars), falls back to splitting at the sentence boundary whose
    cumulative character share is closest to that ratio — sentence boundaries
    stay intact, and the context benefit isn't discarded. Returns None only
    when no sensible split exists, so the caller can retranslate the target
    in isolation.
    """
    if num_context_sentences <= 0:
        return combined_translated
    if not combined_translated:
        return None
    out_sentences, out_remainder = split_into_sentences(combined_translated)
    if out_remainder:
        out_sentences = out_sentences + [out_remainder]
    if len(out_sentences) > num_context_sentences:
        _ctx_align_stats["exact"] += 1
        _log_ctx_align_stats()
        return " ".join(out_sentences[num_context_sentences:]).strip() or None
    if source_char_ratio is not None and len(out_sentences) >= 2:
        total_len = max(1, len(combined_translated))
        best_k, best_diff = 1, None
        cum = 0
        for k in range(1, len(out_sentences)):
            cum += len(out_sentences[k - 1]) + 1
            diff = abs(cum / total_len - source_char_ratio)
            if best_diff is None or diff < best_diff:
                best_k, best_diff = k, diff
        _ctx_align_stats["proportional"] += 1
        _log_ctx_align_stats()
        return " ".join(out_sentences[best_k:]).strip() or None
    _ctx_align_stats["failed"] += 1
    _log_ctx_align_stats()
    return None


def _log_ctx_align_stats():
    """Log alignment mix every 50 extractions so the mismatch rate is measurable."""
    total = sum(_ctx_align_stats.values())
    if total and total % 50 == 0:
        print(f"[CTX-ALIGN] exact={_ctx_align_stats['exact']} proportional={_ctx_align_stats['proportional']} failed={_ctx_align_stats['failed']}", flush=True)


def distribute_whisper_translation(translated_text, row_texts):
    """
    Split a whole-chunk Whisper translation across the rows saved from that chunk.

    Pass 2 of whisper_translate translates the entire audio chunk in one call,
    but that chunk may have produced several DB rows. Sentence-count alignment
    is used when it matches; otherwise the translation is split into word spans
    proportional to each source row's length. Callers first narrow the text to
    the batch's time span via scope_whisper_translation; the split here is
    still approximate when sentence counts don't match.

    Returns a list of len(row_texts) strings.
    """
    n = len(row_texts)
    if n == 0:
        return []
    if n == 1:
        return [translated_text]

    units, rem = split_into_sentences(translated_text)
    if rem.strip():
        units.append(rem.strip())
    if len(units) == n:
        return units

    words = translated_text.split()
    if len(words) < n:
        # Too little text to split meaningfully; keep it on the first row
        return [translated_text] + [""] * (n - 1)

    src_counts = [max(1, len(t.split())) for t in row_texts]
    total = sum(src_counts)
    parts = []
    start = 0
    for i, count in enumerate(src_counts):
        if i == n - 1:
            end = len(words)
        else:
            end = start + max(1, round(len(words) * count / total))
            # Leave at least one word for every remaining row
            end = min(end, len(words) - (n - 1 - i))
            end = max(end, start + 1)
        parts.append(" ".join(words[start:end]))
        start = end
    return parts


def scope_whisper_translation(pass2_timed, span_end, margin=0.5):
    """
    Keep only pass-2 translation segments that fall within the finalized span.

    Pass 2 of whisper_translate covers the whole unfinalized buffer, including
    the still-in-progress tail; that tail's translation belongs to future rows,
    not this batch. Dropping segments whose midpoint lies past span_end keeps
    the distributed translation aligned with the rows actually being saved.

    pass2_timed: list of (session_start, session_end, text).
    Returns the joined in-span text, or None if nothing/nothing usable remains
    (caller falls back to the full pass-2 text).
    """
    if not pass2_timed or span_end is None:
        return None
    parts = [t for (s, e, t) in pass2_timed if (s + e) / 2 <= span_end + margin]
    return " ".join(parts).strip() or None


def is_fuzzy_duplicate(new_sentence, saved_sentences, threshold=0.85):
    """
    Check if new_sentence is a fuzzy duplicate of any saved sentence.

    Uses difflib.SequenceMatcher for similarity comparison.

    Args:
        new_sentence: The sentence to check
        saved_sentences: List of already-saved sentences
        threshold: Minimum similarity ratio (0.0-1.0) to consider duplicate

    Returns:
        True if new_sentence is a fuzzy duplicate of any saved sentence
    """
    from difflib import SequenceMatcher

    if not new_sentence or not saved_sentences:
        return False

    new_lower = new_sentence.lower().strip()

    for saved in saved_sentences:
        saved_lower = saved.lower().strip()
        ratio = SequenceMatcher(None, new_lower, saved_lower).ratio()
        if ratio >= threshold:
            return True

    return False


def remove_overlapping_prefix(new_text, previous_text, min_overlap_words=3):
    """
    Remove overlapping prefix from new_text if it matches the end of previous_text.

    This handles the case where Whisper's rolling buffer transcription produces
    text that starts with the end of the previous finalized text.

    Example:
        previous: "I hope you are doing well from your food comas"
        new: "from your food comas that you experienced"
        result: "that you experienced"

    Args:
        new_text: The new text to check
        previous_text: The previously saved text
        min_overlap_words: Minimum words to consider as overlap (default 3)

    Returns:
        new_text with overlapping prefix removed, or original if no overlap
    """
    if not new_text or not previous_text:
        return new_text

    new_words = new_text.split()
    prev_words = previous_text.split()

    if len(new_words) < min_overlap_words or len(prev_words) < min_overlap_words:
        return new_text

    # Check if new_text starts with the end of previous_text
    # Start with longest possible overlap and work down
    max_overlap = min(len(new_words), len(prev_words))
    for overlap_len in range(max_overlap, min_overlap_words - 1, -1):
        # Get last N words of previous and first N words of new
        prev_suffix = ' '.join(prev_words[-overlap_len:]).lower()
        new_prefix = ' '.join(new_words[:overlap_len]).lower()

        # Use fuzzy match to handle minor transcription differences
        from difflib import SequenceMatcher
        if SequenceMatcher(None, prev_suffix, new_prefix).ratio() > 0.85:
            # Found overlap - return text after the overlap
            remaining = ' '.join(new_words[overlap_len:])
            if remaining.strip():
                print(f"[OVERLAP] Removed {overlap_len} words: '{new_prefix[:40]}...'", flush=True)
                return remaining
            else:
                # Entire new text was overlap - return empty
                print("[OVERLAP] Entire text was overlap, skipping", flush=True)
                return ""

    return new_text


class _TimestampedStream:
    """Wraps a text stream so each output line is prefixed with a local time.
    Turns the app's bare '[TAG] msg' prints into greppable-by-time log lines
    without touching the ~500 print() call sites. Buffers across writes so a
    prefix only lands at a true line start."""

    def __init__(self, wrapped):
        self._w = wrapped
        self._at_line_start = True
        self.stt_timestamped = True  # marker so we never double-wrap

    def write(self, s):
        if not s:
            return 0
        ts = time.strftime("[%H:%M:%S] ")
        prefix = ts if self._at_line_start else ""
        self._at_line_start = s.endswith("\n")
        body = s.replace("\n", "\n" + ts)
        if self._at_line_start and body.endswith(ts):
            body = body[: -len(ts)]  # don't prefix the not-yet-written next line
        try:
            self._w.write(prefix + body)
        except Exception:
            pass
        return len(s)

    def flush(self):
        try:
            self._w.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._w, name)


_diag_installed = False
_faulthandler_fh = None  # kept open for the whole process so native crashes dump


def install_crash_diagnostics(role="main"):
    """Per-process observability: native-crash capture, uncaught-exception
    logging, and timestamped stdout/stderr. Idempotent and best-effort; safe to
    call in every process (fork children inherit an already-installed setup)."""
    global _diag_installed, _faulthandler_fh
    if _diag_installed:
        return
    _diag_installed = True

    logs_dir = os.path.join(APP_DIR, "logs")
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except OSError:
        pass

    # Native crashes (CUDA/torch/native segfaults, aborts) -> C stack trace.
    # A dedicated kept-open handle guarantees capture even if stdout/stderr are
    # redirected or None (GUI builds).
    try:
        import faulthandler
        _faulthandler_fh = open(os.path.join(logs_dir, "faulthandler.log"),
                                "a", buffering=1, encoding="utf-8", errors="replace")
        faulthandler.enable(file=_faulthandler_fh, all_threads=True)
    except Exception:
        try:
            import faulthandler
            faulthandler.enable()  # fall back to stderr
        except Exception:
            pass

    # Timestamp stdout/stderr (covers every [TAG] print).
    for _name in ("stdout", "stderr"):
        _s = getattr(sys, _name, None)
        if _s is not None and not getattr(_s, "stt_timestamped", False):
            try:
                setattr(sys, _name, _TimestampedStream(_s))
            except Exception:
                pass

    # Uncaught exceptions (main + worker threads) -> full traceback before exit.
    import traceback as _tb

    def _excepthook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        try:
            print(f"[FATAL] Uncaught exception in {role} process:", flush=True)
            _tb.print_exception(exc_type, exc, tb)
            sys.stderr.flush()
        except Exception:
            pass

    sys.excepthook = _excepthook

    try:
        def _thread_hook(args):
            if issubclass(args.exc_type, KeyboardInterrupt):
                return
            try:
                print(f"[FATAL] Uncaught exception in thread {args.thread.name}:", flush=True)
                _tb.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
                sys.stderr.flush()
            except Exception:
                pass
        threading.excepthook = _thread_hook
    except Exception:
        pass


def thread1_function(ts, cq, cfq, cal_state, cal_data, cal_step1, asq):
    """Main transcription process with start/stop support"""
    install_crash_diagnostics("worker")
    # On macOS/Windows (spawn), shared state objects are passed as arguments because the spawned
    # child re-imports this module and cannot recreate them. Assign to module globals so
    # all existing code in this function works unchanged. On Linux (fork), the globals are
    # already set via fork, and these reassignments are a no-op.
    global transcription_state, control_queue, config_queue, audio_stream_queue
    global calibration_state, calibration_data_shared, calibration_step1_data
    transcription_state = ts
    control_queue = cq
    config_queue = cfq
    calibration_state = cal_state
    calibration_data_shared = cal_data
    calibration_step1_data = cal_step1
    audio_stream_queue = asq

    # Make every file/dir this subprocess creates readable by all users (a+r files,
    # a+rx dirs): the session DB, its WAL/SHM sidecars, SRT/HTML exports, audio
    # backups, and file-mover output — including everything written during stop
    # cleanup. This is a separate process from the web server, so config writes
    # (which may hold secrets) are unaffected.
    try:
        os.umask(0o022)
    except Exception:
        pass

    # Initialize state variables
    is_running = False
    audio_model = None
    processor = None
    model_type = None
    vad_model = None
    source = None
    recorder = None
    persistent_db_conn = None
    stop_listening = None
    session_audio_file = None
    session_audio_written = False
    calibration_mode = False
    calibration_data = None

    try:
        while True:
            # Check for control commands
            try:
                command = None
                try:
                    command = control_queue.get_nowait()
                except Empty:
                    pass  # No pending control command
                if command is not None:
                    if command["command"] == "start" and not is_running:
                        is_running = True
                        print("[WORKER] Starting transcription...", flush=True)
                    elif command["command"] == "start_calibration":
                        # Handle calibration start command - use local state
                        calibration_mode = True
                        calibration_data = {
                            "start_time": time.time(),
                            "duration": command.get("duration", 30),
                            "noise_samples": [],
                            "speech_samples": [],
                            "silence_durations": [],
                            "energy_levels": [],
                            "vad_probabilities": [],
                        }
                        print(f"[CALIBRATION-PROCESS] Calibration mode enabled in transcription process - duration: {calibration_data['duration']}s", flush=True)
                    elif command["command"] == "stop" and is_running:
                        print("[STOP] Stop command received, signaling main loop to exit...")
                        is_running = False  # Signal the main loop to stop
                        transcription_state["live_text"] = ""  # Clear live preview
                        # Abort any in-flight calibration; the worker survives
                        # stop/start, so stale flags would report "calibrating" forever
                        if calibration_mode or calibration_state.get("active"):
                            print("[STOP] Aborting in-flight calibration")
                            calibration_mode = False
                            calibration_state["active"] = False
                        print("[STOP] is_running set to False - main() will exit and cleanup")

                        # Stop audio source to unblock the main loop if it's waiting for audio
                        if source:
                            try:
                                print("[STOP] Stopping audio source to unblock main loop...")
                                if hasattr(source, "stop") and callable(source.stop):
                                    source.stop()
                                    print("[STOP] OK: Audio source stopped")
                                # Fallback: kill any lingering ffmpeg processes for this device
                                if hasattr(source, "device_name") and source.device_name:
                                    import subprocess as sp
                                    try:
                                        if platform.startswith('win'):
                                            sp.run(["taskkill", "/F", "/IM", "ffmpeg.exe"],
                                                   capture_output=True, timeout=2)
                                        else:
                                            sp.run(["pkill", "-9", "-f", f"ffmpeg.*{re.escape(source.device_name)}"],
                                                   capture_output=True, timeout=2)
                                        print(f"[STOP] OK: Sent kill for ffmpeg using {source.device_name}")
                                    except Exception as pkill_err:
                                        print(f"[STOP] kill fallback failed: {pkill_err}")
                            except Exception as e:
                                print(f"[STOP] WARNING: Error stopping audio source: {e}")

                        # Let main() handle all the cleanup when it exits
                        print("[STOP] Waiting for main() to exit and cleanup...")
                        continue
                    elif command["command"] == "unload":
                        # Explicit unload command to release GPU memory without killing the process
                        print("[UNLOAD] Unload command received, cleaning up models...")
                        import gc
                        # Unload models to release GPU memory
                        if audio_model is not None:
                            try:
                                del audio_model
                            except (NameError, AttributeError):
                                pass
                            audio_model = None
                        if processor is not None:
                            try:
                                del processor
                            except (NameError, AttributeError):
                                pass
                            processor = None
                        if vad_model is not None:
                            try:
                                del vad_model
                            except (NameError, AttributeError):
                                pass
                            vad_model = None
                        model_type = None
                        # Release the PANNs audio tagger too
                        try:
                            unload_audio_tagger()
                        except Exception:
                            pass
                        # Clean up model cache
                        try:
                            ModelFactory.cleanup_models()
                        except Exception as e:
                            print(f"[UNLOAD] Warning: ModelFactory cleanup error: {e}")
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print("[UNLOAD] Models unloaded, GPU memory released")
            except Exception as e:
                print(f"[WARNING] Error processing control command: {e}", flush=True)
                # Continue processing even if queue operations fail

            # If not running, just wait
            if not is_running:
                sleep(0.3)
                continue

            # If running but not initialized, initialize now
            if is_running and audio_model is None:
                def main():
                    # Declare nonlocal variables to update outer scope
                    nonlocal \
                        is_running, \
                        audio_model, \
                        processor, \
                        model_type, \
                        vad_model, \
                        source, \
                        recorder, \
                        persistent_db_conn, \
                        stop_listening, \
                        session_audio_file, \
                        session_audio_written, \
                        calibration_mode, \
                        calibration_data

                    # Load fresh config for this process
                    process_config = load_config()
                    # Get defaults from config file
                    # Note: model config is at model.whisper, not whisper
                    whisper_config = process_config.get("model", {}).get("whisper", {})
                    audio_config = process_config.get("audio", {})
                    vad_config = process_config.get("vad", {})

                    parser = argparse.ArgumentParser(
                        description="Real-time Speech-to-Text with Whisper",
                        epilog="Note: Command-line arguments override config.json settings",
                    )
                    parser.add_argument(
                        "--model",
                        default=whisper_config.get("model", "tiny"),
                        help=f"Model to use (default from config: {whisper_config.get('model', 'tiny')})",
                        choices=[
                            "tiny",
                            "base",
                            "small",
                            "medium",
                            "large",
                            "large-v1",
                            "large-v2",
                        ],
                    )
                    parser.add_argument(
                        "--energy_threshold",
                        default=audio_config.get("energy_threshold", 3500),
                        help=f"Energy level for mic to detect (default from config: {audio_config.get('energy_threshold', 3500)})",
                        type=int,
                    )
                    parser.add_argument(
                        "--record_timeout",
                        default=audio_config.get("record_timeout", 3),
                        help=f"How real time the recording is in seconds (default from config: {audio_config.get('record_timeout', 3)})",
                        type=float,
                    )
                    parser.add_argument(
                        "--phrase_timeout",
                        default=audio_config.get("phrase_timeout", 2),
                        help=f"Silence duration before new line (default from config: {audio_config.get('phrase_timeout', 2)}s)",
                        type=float,
                    )
                    parser.add_argument(
                        "--use_vad",
                        default=vad_config.get("enabled", True),
                        action="store_true",
                        help=f"Enable Voice Activity Detection (default from config: {vad_config.get('enabled', True)})",
                    )
                    parser.add_argument(
                        "--disable_vad",
                        dest="use_vad",
                        action="store_false",
                        help="Disable Voice Activity Detection (use only energy-based detection)",
                    )
                    parser.add_argument(
                        "--vad_threshold",
                        default=vad_config.get("threshold", 0.5),
                        help=f"VAD confidence threshold 0.0-1.0 (default from config: {vad_config.get('threshold', 0.5)}). "
                        "Examples: 0.3=sensitive, 0.5=balanced, 0.7=strict, 0.9=very strict",
                        type=float,
                    )
                    parser.add_argument(
                        "--config",
                        default=CONFIG_FILE,
                        help=f"Path to config file (default: {CONFIG_FILE})",
                    )
                    parser.add_argument(
                        "--default_microphone",
                        default=audio_config.get("default_microphone", "default"),
                        help=f"Default microphone for ffmpeg (default from config: {audio_config.get('default_microphone', 'default')}). "
                        "Linux: 'default', 'plughw:0,0', etc. macOS: ':0', ':1', or device name. Run with 'list' to view available devices",
                        type=str,
                    )
                    # Use empty args list to prevent inheriting CLI args from parent process
                    args = parser.parse_args([])

                    # The last time a recording was retreived from the queue.
                    phrase_time = None

                    # Initialize WhisperLive-style transcriber (replaces dual-buffer approach)
                    same_output_threshold = audio_config.get("same_output_threshold", 7)
                    live_transcriber = WhisperLiveTranscriber(
                        sample_rate=16000,
                        same_output_threshold=same_output_threshold,
                    )
                    print(f"[INIT] WhisperLiveTranscriber initialized (same_output_threshold={same_output_threshold})")

                    # Preserved from old implementation
                    saved_sentences = []        # Sentences already saved to DB (for fuzzy duplicate detection)
                    fuzzy_threshold = audio_config.get("fuzzy_duplicate_threshold", 0.85)  # Similarity threshold for dedup
                    min_words_threshold = audio_config.get("min_words", 0)  # Minimum word count to save segment

                    # Pending-remainder buffer: hold incomplete sentence fragments across captures
                    # so only complete sentences become DB rows (= translation units)
                    pending_buffer_cfg = audio_config.get("pending_buffer", {})
                    pending_buffer_enabled = pending_buffer_cfg.get("enabled", True)
                    pending_max_words = pending_buffer_cfg.get("max_words", 30)
                    pending_max_age = pending_buffer_cfg.get("max_age_seconds", 10)
                    pending_remainder = ""       # Incomplete sentence fragment held back from DB
                    pending_remainder_since = None  # When the current fragment was first buffered
                    pending_remainder_meta = None   # (start_time, end_time, confidence) of fragment

                    # Full session audio file path (append mode)
                    session_audio_file = None
                    session_audio_written = False
                    # Thread safe Queue for passing data from the threaded recording callback.
                    data_queue = Queue()
                    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
                    recorder = sr.Recognizer()

                    # Use energy threshold from config
                    recorder.energy_threshold = args.energy_threshold
                    print(f"[AUDIO] Energy threshold: {args.energy_threshold}")

                    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
                    recorder.dynamic_energy_threshold = False

                    # Initialize ffmpeg audio backend with multi-level fallback
                    source = None
                    from audio_capture import create_compatible_audio_source

                    # A configured file path wins: when default_microphone points at an
                    # existing file (a "Test Audio File" selection), drive the pipeline from
                    # that file only. No mic fallback, so a bad/missing file errors visibly
                    # instead of silently reverting to hardware capture.
                    if args.default_microphone and os.path.isfile(args.default_microphone):
                        print(f"[AUDIO] File playback mode: {args.default_microphone}")
                        audio_devices_to_try = [args.default_microphone]
                    else:
                        # Resolve saved stable device name (e.g. "UR22mkII") against the CURRENT
                        # ALSA enumeration first, since plughw:N,0 indices are not stable across
                        # reboots (USB vs onboard/GPU HDA cards can enumerate in either order).
                        resolved_device = None
                        saved_mic_name = audio_config.get("default_microphone_name", "")
                        if saved_mic_name:
                            try:
                                from audio_capture import list_audio_devices, FFmpegAudioCapture
                                markers = audio_config.get("deprioritize_device_markers", [])
                                current_devices = list_audio_devices(deprioritize_markers=markers)
                                matched = FFmpegAudioCapture.resolve_device_by_name(saved_mic_name, current_devices)
                                if matched:
                                    resolved_device = matched["name"]
                                    print(f"[AUDIO] Resolved saved device name '{saved_mic_name}' -> '{resolved_device}'")
                                else:
                                    print(f"[AUDIO] WARNING: Saved device name '{saved_mic_name}' not found in current devices; falling back")
                            except Exception as e:
                                print(f"[AUDIO] WARNING: Device name resolution failed: {e}")

                        # Try multiple audio devices in order of preference
                        audio_devices_to_try = [
                            resolved_device,           # Name-resolved device (correct card, current index)
                            args.default_microphone,  # Configured device (e.g., plughw:1,0)
                            "default",                 # System default
                            "plughw:0,0"              # First hardware device
                        ]

                    last_error = None
                    for device in audio_devices_to_try:
                        # Skip invalid entries
                        if not device or device == "list":
                            continue

                        try:
                            print(f"[INIT] Step 2/5: Trying audio device: {device}")
                            # Get backup settings from config for MPEG-TS backup
                            backup_cfg = process_config.get("audio_backup", {})
                            # Check if .ts backup is enabled (default True for backward compatibility)
                            ts_enabled = backup_cfg.get("ts_enabled", True)
                            ts_filename_format = backup_cfg.get("filename_format", "").strip() or "%Y-%m-%d_%H%M%S"
                            ts_filename_prefix = backup_cfg.get("filename_prefix", "")
                            # Build full backup directory path (same as .wav uses)
                            ts_base_dir = backup_cfg.get("base_directory", "").strip() or BACKUP_DIR
                            ts_path_format = backup_cfg.get("path_format", "").strip() or "%Y/%m"
                            ts_formatted_path = datetime.now().strftime(ts_path_format)
                            ts_backup_dir = os.path.join(ts_base_dir, ts_formatted_path) if ts_enabled else None
                            source = create_compatible_audio_source(
                                device_name=device,
                                sample_rate=16000,
                                backup_dir=ts_backup_dir,
                                filename_format=ts_filename_format,
                                filename_prefix=ts_filename_prefix,
                                ts_enabled=ts_enabled,
                            )
                            # Start the ffmpeg capture (it will populate the data_queue)
                            source.start()
                            print(f"[OK] Audio initialized successfully with device: {device}")
                            break  # Success! Exit the loop
                        except Exception as e:
                            print(f"[WARN] Audio device '{device}' failed: {e}")
                            last_error = e
                            # Clean up failed source
                            if source:
                                try:
                                    source.stop()
                                except Exception:
                                    pass
                            source = None
                            # Continue to next device

                    # If all devices failed, check if ANY audio devices exist
                    if not source:
                        error_msg = None
                        try:
                            import subprocess
                            print("[CHECK] Checking for available audio devices...")
                            if platform.startswith('win'):
                                # Windows: use PowerShell to check for audio devices
                                result = subprocess.run(
                                    ["powershell", "-Command", "Get-WmiObject Win32_SoundDevice | Select-Object Name"],
                                    capture_output=True,
                                    text=True,
                                    timeout=5
                                )
                                if not result.stdout.strip() or "name" not in result.stdout.lower():
                                    error_msg = "No audio devices found on system. Please connect a microphone."
                                else:
                                    error_msg = f"All audio devices failed to initialize. Last error: {last_error}"
                            else:
                                result = subprocess.run(
                                    ["arecord", "-l"],
                                    capture_output=True,
                                    text=True,
                                    timeout=5
                                )
                                if "no soundcards found" in result.stderr.lower() or "no soundcards found" in result.stdout.lower():
                                    error_msg = "No audio devices found on system. Please connect a microphone."
                                else:
                                    error_msg = f"All audio devices failed to initialize. Last error: {last_error}"
                        except FileNotFoundError:
                            error_msg = f"Audio initialization failed: {last_error}. Unable to check for devices."
                        except Exception:
                            error_msg = f"Audio initialization failed: {last_error}"

                        print(f"[ERROR] {error_msg}")
                        import traceback
                        traceback.print_exc()

                        # Update state to notify UI
                        with _transcription_state_lock:
                            transcription_state["running"] = False
                            transcription_state["status"] = "error"
                            transcription_state["error"] = error_msg
                            transcription_state["message"] = "Audio initialization failed"
                        return


                    # Check if stop was requested during audio initialization
                    if not is_running:
                        print(
                            "[INFO] Stop requested during audio initialization, cleaning up..."
                        )
                        if source:
                            try:
                                if hasattr(source, "__del__"):
                                    source.__del__()
                                del source
                            except Exception:
                                pass
                        return  # Exit main() function early

                    # Load / Download model using ModelFactory
                    model_config = process_config.get("model", {})
                    use_gpu = process_config.get("performance", {}).get("use_gpu", True)

                    # For backward compatibility with old CLI args
                    if model_config.get("type") == "whisper":
                        # Update whisper config with CLI args if provided
                        model_config["whisper"]["model"] = args.model

                    print(
                        f"[INIT] Step 3/5: Loading model ({model_config.get('type', 'whisper')})..."
                    )

                    try:
                        audio_model, processor, model_type = ModelFactory.load_model(
                            model_config, use_gpu
                        )
                        print(f"[OK] Model loaded successfully: {model_config.get('type', 'whisper')}")

                        # Determine the actual loaded model name for display
                        loaded_model_name = ""
                        if model_config.get("type") == "whisper":
                            # Model name now includes .en suffix directly (e.g., "small.en")
                            model_name = model_config.get("whisper", {}).get("model", "base")
                            backend = model_config.get("backend")
                            prefix = "Faster Whisper" if backend == "faster-whisper" else "Whisper"
                            loaded_model_name = f"{prefix} {model_name}"
                        elif model_config.get("type") == "huggingface":
                            # Extract model name from HuggingFace model_id (e.g., "openai/whisper-large-v3" -> "whisper-large-v3")
                            model_id = model_config.get("huggingface", {}).get("model_id", "")
                            loaded_model_name = model_id.split("/")[-1] if model_id else "huggingface"
                        elif model_config.get("type") == "custom":
                            # Extract basename from custom model path
                            model_path = model_config.get("custom", {}).get("model_path", "")
                            loaded_model_name = os.path.basename(model_path) if model_path else "custom"

                        # Update transcription state with loaded model name
                        with _transcription_state_lock:
                            transcription_state["loaded_model"] = loaded_model_name

                    except Exception as e:
                        error_msg = f"Model loading failed: {str(e)}"
                        print(f"[ERROR] {error_msg}")
                        import traceback
                        traceback.print_exc()

                        # Update state to notify UI
                        with _transcription_state_lock:
                            transcription_state["running"] = False
                            transcription_state["status"] = "error"
                            transcription_state["error"] = error_msg
                            transcription_state["message"] = "Model loading failed"

                        # Cleanup audio source if it was initialized
                        if source:
                            try:
                                source.stop()
                            except Exception:
                                pass

                        # Clear orphaned GPU memory from failed model load
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                print("[CLEANUP] GPU cache cleared after failed model load")
                        except (RuntimeError, AttributeError):
                            pass

                        return

                    # Check if stop was requested during model loading
                    if not is_running:
                        print(
                            "[INFO] Stop requested during model loading, cleaning up..."
                        )
                        # Clear references BEFORE cleanup to allow garbage collection
                        audio_model = None
                        processor = None
                        model_type = None
                        ModelFactory.cleanup_models()
                        return  # Exit main() function early

                    # Load Silero VAD model (if enabled)
                    vad_model = None
                    vad_threshold = args.vad_threshold
                    if args.use_vad:
                        print("Loading VAD model...")
                        try:
                            from silero_vad import load_silero_vad
                            vad_model = load_silero_vad()
                            print(f"VAD enabled (silero-vad) with threshold: {vad_threshold}")
                        except ImportError:
                            print("[VAD] silero-vad package not installed. Install with: pip install silero-vad")
                            print("[VAD] Continuing without VAD (using energy-based detection only)")
                            vad_model = None
                        except Exception as e:
                            print(f"[VAD] Error loading silero-vad: {e}")
                            print("[VAD] Continuing without VAD (using energy-based detection only)")
                            vad_model = None
                    else:
                        print("VAD disabled - using energy-based detection only")

                    # Check if stop was requested during VAD loading
                    if not is_running:
                        print(
                            "[INFO] Stop requested during VAD loading, cleaning up..."
                        )
                        # Clear references BEFORE cleanup to allow garbage collection
                        if vad_model:
                            del vad_model
                        audio_model = None
                        processor = None
                        model_type = None
                        vad_model = None
                        ModelFactory.cleanup_models()
                        return  # Exit main() function early

                    record_timeout = args.record_timeout
                    phrase_timeout = args.phrase_timeout

                    # Initialize database (lazy loading - only when transcription starts)
                    print("[INIT] Step 4/5: Initializing database...")

                    try:
                        db_path = initialize_database()

                        # Create persistent database connection for this process
                        # This avoids overhead of opening/closing connection on every transcription
                        persistent_db_conn = sqlite3.connect(
                            db_path, check_same_thread=False, timeout=30.0
                        )
                        persistent_db_cursor = persistent_db_conn.cursor()

                        # Enable WAL mode for this connection too
                        persistent_db_cursor.execute("PRAGMA journal_mode=WAL")
                        persistent_db_cursor.execute("PRAGMA synchronous=NORMAL")
                        persistent_db_cursor.execute(
                            "PRAGMA busy_timeout=30000"
                        )  # 30 second timeout
                        # WAL/SHM sidecars are (re)created here by this connection —
                        # keep them readable by all users alongside the .db file.
                        make_db_world_readable(db_path)

                        print(f"[OK] Database initialized: {db_path}")
                    except Exception as e:
                        error_msg = f"Database initialization failed: {str(e)}"
                        print(f"[ERROR] {error_msg}")
                        import traceback
                        traceback.print_exc()

                        # Update state to notify UI
                        with _transcription_state_lock:
                            transcription_state["running"] = False
                            transcription_state["status"] = "error"
                            transcription_state["error"] = error_msg
                            transcription_state["message"] = "Database initialization failed"

                        # Cleanup resources
                        if source:
                            try:
                                source.stop()
                            except Exception:
                                pass
                        # Clear references BEFORE cleanup to allow garbage collection
                        audio_model = None
                        processor = None
                        model_type = None
                        vad_model = None
                        try:
                            ModelFactory.cleanup_models()
                        except Exception:
                            pass
                        return

                    # Initialize full session audio file (if .wav backup is enabled)
                    backup_config = process_config.get("audio_backup", {})
                    # Support both old "enabled" key and new "wav_enabled" key for backward compatibility
                    wav_backup_enabled = backup_config.get("wav_enabled", backup_config.get("enabled", False))
                    if wav_backup_enabled:
                        try:
                            now = datetime.now(configured_timezone)
                            base_dir = backup_config.get(
                                "base_directory", ""
                            ).strip() or BACKUP_DIR
                            path_format = backup_config.get("path_format", "").strip() or "%Y/%m"
                            formatted_path = now.strftime(path_format)
                            full_dir_path = os.path.join(base_dir, formatted_path)
                            os.makedirs(full_dir_path, exist_ok=True)

                            filename_format = backup_config.get(
                                "filename_format", ""
                            ).strip() or "%Y-%m-%d_%H%M%S"
                            filename_prefix = backup_config.get("filename_prefix", "")
                            # Build filename: {timestamp}_{prefix}.wav or {timestamp}.wav
                            if filename_prefix:
                                session_filename = f"{now.strftime(filename_format)}_{filename_prefix}.wav"
                            else:
                                session_filename = f"{now.strftime(filename_format)}.wav"
                            session_audio_file = os.path.join(
                                full_dir_path, session_filename
                            )
                            print(
                                f"[BACKUP] Full session file initialized: {session_audio_file}"
                            )
                        except Exception as e:
                            print(
                                f"[WARNING] Failed to initialize session audio file: {e}"
                            )
                            session_audio_file = None

                    def has_speech(audio_bytes, sample_rate=16000):
                        """
                        Check if audio contains speech using hybrid two-stage filtering:
                        1. Energy threshold (always applied) - fast rejection of quiet audio
                        2. VAD (if enabled) - accurate speech detection for loud audio

                        Returns True if speech is detected, False otherwise.
                        """
                        # Stage 1: Energy Threshold Filter (ALWAYS applied)
                        try:
                            # Calculate raw energy using EXACT same method as visualization
                            # Normalize to -1.0 to 1.0 range BEFORE calculating RMS
                            audio_np = (
                                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                                / 32768.0
                            )
                            rms = np.sqrt(np.mean(audio_np ** 2))
                            raw_energy = float(rms * 32768.0)  # Convert back to raw energy

                            # Get energy threshold from config (default 3500)
                            energy_threshold = process_config.get("audio", {}).get("energy_threshold", 3500)

                            # Fast rejection: if audio is too quiet, skip transcription
                            if raw_energy < energy_threshold:
                                # print(f"[FILTER] Energy rejected: {raw_energy:.0f} < {energy_threshold}")
                                return False
                            # else:
                            #     print(f"[FILTER] Energy passed: {raw_energy:.0f} >= {energy_threshold}")
                        except Exception as e:
                            # If energy calculation fails, continue to VAD check
                            print(f"[WARNING] Energy threshold check failed: {e}, continuing to VAD")

                        # Stage 2: VAD Filter (only if enabled and audio passed energy check)
                        if vad_model is not None:
                            try:
                                # Convert raw audio bytes to numpy array
                                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                                # Silero-VAD requires fixed chunk sizes: 512 samples for 16kHz, 256 for 8kHz
                                chunk_size = 512 if sample_rate == 16000 else 256

                                # Process audio in correctly-sized chunks
                                for i in range(0, len(audio_np) - chunk_size + 1, chunk_size):
                                    chunk = audio_np[i:i + chunk_size]
                                    audio_tensor = torch.from_numpy(chunk)
                                    prob = vad_model(audio_tensor, sample_rate).item()
                                    if prob >= vad_threshold:
                                        return True

                                # No chunk exceeded the VAD threshold
                                return False
                            except Exception as e:
                                # If VAD fails, default to processing the audio (passed energy check)
                                print(f"[WARNING] VAD check failed: {e}, defaulting to process audio")
                                return True

                        # VAD is disabled, audio passed energy check, so process it
                        return True

                    def save_audio_backup(wav_data_bytes, backup_config):
                        """
                        Save raw unprocessed audio input to backup directory with configurable format.
                        This saves ALL audio input (speech, music, noise, silence) before VAD filtering.
                        Uses configurable path and filename formats like database.

                        Args:
                            wav_data_bytes: WAV audio data as bytes (raw unprocessed input)
                            backup_config: Configuration dict with 'wav_enabled', 'base_directory', 'path_format', 'filename_format', 'filename_prefix', and 'format' keys

                        Returns:
                            str: Path to saved file if successful, None otherwise
                        """
                        # Support both old "enabled" key and new "wav_enabled" key for backward compatibility
                        if not backup_config.get("wav_enabled", backup_config.get("enabled", False)):
                            return None

                        try:
                            # Get current time in configured timezone
                            now = datetime.now(configured_timezone)

                            # Use configurable base directory or default
                            base_dir = backup_config.get("base_directory", "").strip() or BACKUP_DIR

                            # Use configurable path format or default (using Python strftime format)
                            path_format = backup_config.get("path_format", "").strip() or "%Y/%m"

                            # Use strftime directly with user's format
                            formatted_path = now.strftime(path_format)

                            # Create full directory path
                            full_dir_path = os.path.join(base_dir, formatted_path)
                            os.makedirs(full_dir_path, exist_ok=True)

                            # Use configurable filename format (using Python strftime format)
                            filename_format = backup_config.get(
                                "filename_format", ""
                            ).strip() or "%Y-%m-%d_%H%M%S"
                            audio_format = backup_config.get("format", "wav")
                            filename_prefix = backup_config.get(
                                "filename_prefix", ""
                            ).strip()

                            # Use strftime directly with user's format
                            formatted_filename = now.strftime(filename_format)

                            # Add prefix and extension
                            if filename_prefix:
                                filename = f"{formatted_filename}_{filename_prefix}.{audio_format}"
                            else:
                                filename = f"{formatted_filename}.{audio_format}"

                            filepath = os.path.join(full_dir_path, filename)

                            # Save audio file
                            with open(filepath, "wb") as f:
                                f.write(wav_data_bytes)

                            return filepath
                        except Exception as e:
                            print(f"[WARNING] Failed to save audio backup: {e}")
                            return None

                        try:
                            # Get current time in configured timezone
                            now = datetime.now(configured_timezone)

                            # Create directory structure: base/YYYY/YYYY-MM/
                            # Use same default as database if not specified
                            base_dir = backup_config.get("base_directory", "").strip()
                            if not base_dir:
                                base_dir = BACKUP_DIR
                            year_dir = os.path.join(base_dir, now.strftime("%Y"))
                            month_dir = os.path.join(year_dir, now.strftime("%Y-%m"))

                            # Create directories if they don't exist
                            os.makedirs(month_dir, exist_ok=True)

                            # Create filename: YYYY-MM-DD-HHmmss.wav or with custom prefix
                            audio_format = backup_config.get("format", "wav")
                            filename_prefix = backup_config.get(
                                "filename_prefix", ""
                            ).strip()

                            if filename_prefix:
                                filename = now.strftime(
                                    f"%Y-%m-%d-%H%M%S_{filename_prefix}.{audio_format}"
                                )
                            else:
                                filename = now.strftime(
                                    f"%Y-%m-%d-%H%M%S.{audio_format}"
                                )

                            filepath = os.path.join(month_dir, filename)

                            # Save the audio file
                            with open(filepath, "wb") as f:
                                f.write(wav_data_bytes)

                            return filepath
                        except Exception as e:
                            print(f"[WARNING] Failed to save audio backup: {e}")
                            return None

                    # Scratch path for per-chunk audio. Use mkstemp + close so we
                    # don't leak the open file descriptor that NamedTemporaryFile
                    # would keep alive.
                    _temp_fd, temp_file = tempfile.mkstemp(suffix=".wav")
                    os.close(_temp_fd)

                    # ffmpeg backend doesn't need ambient noise adjustment
                    print(
                        "[AUDIO] Skipping ambient noise adjustment (ffmpeg handles this)"
                    )

                    # Check if stop was requested during audio setup
                    if not is_running:
                        print(
                            "[INFO] Stop requested during audio setup, cleaning up..."
                        )
                        if persistent_db_conn:
                            try:
                                persistent_db_conn.close()
                            except (sqlite3.Error, OSError):
                                pass
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                        # Clear references BEFORE cleanup to allow garbage collection
                        if vad_model:
                            try:
                                del vad_model
                            except (NameError, AttributeError):
                                pass
                        audio_model = None
                        processor = None
                        model_type = None
                        vad_model = None
                        ModelFactory.cleanup_models()
                        if source:
                            try:
                                if hasattr(source, "__del__"):
                                    source.__del__()
                                del source
                            except Exception:
                                pass
                        return  # Exit main() function early

                    audio_callback_count = [
                        0
                    ]  # Use list to allow modification in nested function

                    def record_callback(_, audio: sr.AudioData) -> None:
                        """
                        Threaded callback function to recieve audio data when recordings finish.
                        audio: An AudioData containing the recorded bytes.
                        """
                        # Grab the raw bytes and push it into the thread safe queue.
                        data = audio.get_raw_data()
                        data_queue.put(data)
                        audio_callback_count[0] += 1
                        if (
                            audio_callback_count[0] <= 5
                            or audio_callback_count[0] % 100 == 0
                        ):
                            print(
                                f"[AUDIO_CALLBACK] Called {audio_callback_count[0]} times, queue_size={data_queue.qsize()}, data_size={len(data)}"
                            )

                        # Calculate and update energy level for every callback
                        try:
                            import audioop

                            # Calculate RMS energy (raw value, not normalized)
                            energy = audioop.rms(data, source.SAMPLE_WIDTH)

                            # Store raw energy and dB in shared state
                            if energy > 0:
                                db = 20 * np.log10(energy / 32768.0)
                            else:
                                db = -60

                            level = max(0, min(100, (db + 60) * (100 / 60)))

                            transcription_state["audio_energy"] = (
                                energy  # Raw energy value
                            )
                            transcription_state["audio_level"] = level
                            transcription_state["audio_db"] = db
                        except Exception:
                            pass  # Don't break callback on error

                    # For ffmpeg, the capture is already running and filling source.data_queue
                    # We need to use that queue as data_queue
                    data_queue = source.data_queue
                    print(f"[AUDIO] OK: ffmpeg backend already capturing audio to queue (queue id={id(data_queue)})", flush=True)

                    # Cue the user that we're ready to go.
                    print("Model loaded.\n")

                    # Update transcription state to running - ALL initialization complete
                    print("[INIT] Step 5/5: Starting transcription loop...")
                    with _transcription_state_lock:
                        transcription_state["running"] = True
                        transcription_state["status"] = "running"
                        transcription_state["message"] = "Transcription is active and ready"
                        transcription_state["error"] = None
                        transcription_state["start_time"] = time.time()
                    print("[READY] Transcription system initialized successfully!")

                    while True:
                        try:
                            # Check if we should exit the loop
                            if not is_running:
                                print("[LOOP] is_running is False, exiting main loop")
                                break

                            # Check for stop commands and calibration commands with non-blocking operations
                            try:
                                while not control_queue.empty():
                                    command = control_queue.get_nowait()
                                    if command["command"] == "stop":
                                        print("[LOOP] Stop command received, exiting main loop")
                                        is_running = False
                                        break
                                    elif command["command"] == "start_calibration":
                                        # Handle calibration start command in inner loop - use local state
                                        calibration_mode = True
                                        calibration_data = {
                                            "start_time": time.time(),
                                            "duration": command.get("duration", 30),
                                            "noise_samples": [],
                                            "speech_samples": [],
                                            "silence_durations": [],
                                            "energy_levels": [],
                                            "vad_probabilities": [],
                                        }
                                        print(f"[CALIBRATION-PROCESS] Calibration mode enabled in transcription process - duration: {calibration_data['duration']}s", flush=True)
                            except Exception as e:
                                print(f"[WARNING] Error checking control queue: {e}")

                            # Check again after processing control queue
                            if not is_running:
                                break

                            # Check for config updates (hot-reload) with non-blocking operations
                            try:
                                while not config_queue.empty():
                                    config_update = config_queue.get_nowait()
                                    if config_update["type"] == "config_update":
                                        new_config = config_update["config"]

                                        # Update process_config to use new values throughout
                                        process_config.update(new_config)

                                        # Update hot-reloadable settings
                                        if "audio" in new_config:
                                            recorder.energy_threshold = new_config[
                                                "audio"
                                            ].get(
                                                "energy_threshold",
                                                recorder.energy_threshold,
                                            )
                                            record_timeout = new_config["audio"].get(
                                                "record_timeout", record_timeout
                                            )
                                            phrase_timeout = new_config["audio"].get(
                                                "phrase_timeout", phrase_timeout
                                            )
                                            if "pending_buffer" in new_config["audio"]:
                                                _pb_cfg = new_config["audio"]["pending_buffer"]
                                                pending_buffer_enabled = _pb_cfg.get("enabled", pending_buffer_enabled)
                                                pending_max_words = _pb_cfg.get("max_words", pending_max_words)
                                                pending_max_age = _pb_cfg.get("max_age_seconds", pending_max_age)

                                        if "vad" in new_config:
                                            vad_threshold = new_config["vad"].get(
                                                "threshold", vad_threshold
                                            )

                                        print(
                                            f"[OK] Config hot-reloaded: energy={recorder.energy_threshold}, vad_threshold={vad_threshold}, process_config updated"
                                        )
                            except Exception as e:
                                print(f"[WARNING] Error processing config update: {e}")
                                # Continue processing even if queue operations fail

                            # Get the current time in configured timezone
                            now = datetime.now(configured_timezone)

                            # Pull raw recorded audio from the queue.
                            if not data_queue.empty():
                                # Track accumulated new data for session file
                                accumulated_new_data = bytes()
                                # Use timeout to prevent deadlock
                                if _audio_queue_lock.acquire(timeout=5.0):
                                    try:
                                        while not data_queue.empty():
                                            data = data_queue.get()
                                            skip_transcription = False

                                            # Calibration mode: collect environmental data (two-step process)
                                            if calibration_mode and calibration_data:
                                                skip_transcription = True
                                                current_step = calibration_state.get("step", 1)

                                                # Check if we need to reset the local timer for Step 2
                                                if calibration_state.get("reset_timer", False):
                                                    calibration_data["start_time"] = time.time()
                                                    calibration_state["reset_timer"] = False
                                                    print(f"[CALIBRATION] Timer reset for Step 2 - new start_time: {calibration_data['start_time']}", flush=True)

                                                # CRITICAL FIX: If starting at Step 2 from the beginning (skip_step1),
                                                # reset the timer on the FIRST audio frame to avoid premature completion
                                                if current_step == 2 and not calibration_data.get("step2_timer_initialized", False):
                                                    calibration_data["start_time"] = time.time()
                                                    calibration_data["step2_timer_initialized"] = True
                                                    print(f"[CALIBRATION] Step 2 timer initialized on first frame - start_time: {calibration_data['start_time']}", flush=True)

                                                # print(f"[CALIBRATION-DEBUG] Processing audio data - Step {current_step}")
                                                try:
                                                    # Calculate energy level
                                                    audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                                                    raw_energy = np.sqrt(np.mean(audio_np**2)) * 32768
                                                    calibration_data["energy_levels"].append(raw_energy)
                                                    calibration_data_shared["energy_levels"].append(raw_energy)

                                                    if current_step == 1:
                                                        # STEP 1: Collect noise floor (NO speech detection)

                                                        # Check if step 1 is already complete (waiting for user to click "Start Step 2")
                                                        if not calibration_state.get("step1_complete", False):
                                                            sample_dict = {
                                                                "energy": raw_energy,
                                                                "timestamp": time.time(),
                                                            }

                                                            # Store in step 1 data and shared data
                                                            calibration_step1_data["noise_energies"].append(raw_energy)
                                                            calibration_data_shared["noise_samples"].append(sample_dict)
                                                            calibration_data["noise_samples"].append(sample_dict)
                                                            calibration_state["noise_samples"] = len(calibration_data["noise_samples"])

                                                            # Check if step 1 is complete
                                                            elapsed = time.time() - calibration_data["start_time"]
                                                            if elapsed >= calibration_data["duration"]:
                                                                # Calculate noise statistics for step 2
                                                                noise_list = list(calibration_step1_data["noise_energies"])
                                                                print(f"[CALIBRATION] Step 1 time elapsed: {elapsed:.1f}s >= {calibration_data['duration']}s, noise samples collected: {len(noise_list)}", flush=True)

                                                                if noise_list:
                                                                    calibration_step1_data["avg_noise"] = statistics.mean(noise_list)
                                                                    calibration_step1_data["max_noise"] = max(noise_list)
                                                                    print(f"[CALIBRATION] Step 1 complete - avg_noise: {calibration_step1_data['avg_noise']:.1f}, max_noise: {calibration_step1_data['max_noise']:.1f}", flush=True)
                                                                else:
                                                                    # No noise samples collected - use conservative defaults
                                                                    calibration_step1_data["avg_noise"] = 300.0
                                                                    calibration_step1_data["max_noise"] = 500.0
                                                                    print("[CALIBRATION] Step 1 complete - WARNING: No noise samples collected, using defaults", flush=True)

                                                                # Mark step 1 as complete but DON'T auto-transition
                                                                # Wait for user to click "Start Step 2" button
                                                                calibration_state["step1_complete"] = True
                                                                print("[CALIBRATION] Set step1_complete = True", flush=True)
                                                                # DON'T set step = 2 yet - user must manually continue

                                                    elif current_step == 2:
                                                        # STEP 2: Collect speech with temporary low threshold
                                                        # Use noise floor + 100 as temporary threshold
                                                        temp_threshold = int(calibration_step1_data.get("avg_noise", 300) + 100)

                                                        # Temporarily override energy threshold for this check
                                                        old_threshold = process_config["audio"]["energy_threshold"]
                                                        process_config["audio"]["energy_threshold"] = temp_threshold

                                                        is_speech = has_speech(data, source.SAMPLE_RATE)

                                                        # Restore original threshold
                                                        process_config["audio"]["energy_threshold"] = old_threshold

                                                        sample_dict = {
                                                            "energy": raw_energy,
                                                            "timestamp": time.time(),
                                                        }

                                                        if is_speech:
                                                            calibration_data["speech_samples"].append(sample_dict)
                                                            calibration_data_shared["speech_samples"].append(sample_dict)
                                                            calibration_state["speech_samples"] = len(calibration_data["speech_samples"])

                                                            # Track when speech ends for pause detection
                                                            calibration_data["last_speech_time"] = time.time()

                                                            # Measure VAD probability if VAD enabled
                                                            if vad_model is not None:
                                                                try:
                                                                    # silero-vad pip package - simple call
                                                                    audio_tensor = torch.from_numpy(audio_np).float()
                                                                    speech_prob = vad_model(audio_tensor, source.SAMPLE_RATE).item()
                                                                    calibration_data["vad_probabilities"].append(speech_prob)
                                                                    calibration_data_shared["vad_probabilities"].append(speech_prob)
                                                                except (RuntimeError, TypeError, ValueError):
                                                                    pass
                                                        else:
                                                            # Still collecting some noise during speech phase
                                                            calibration_data["noise_samples"].append(sample_dict)
                                                            calibration_data_shared["noise_samples"].append(sample_dict)
                                                            calibration_state["noise_samples"] = len(calibration_data["noise_samples"])

                                                            # Track silence durations between speech segments
                                                            # Only record if we've had speech before and this is a new silence period
                                                            if "last_speech_time" in calibration_data and "silence_start_time" not in calibration_data:
                                                                # Just transitioned from speech to silence
                                                                calibration_data["silence_start_time"] = time.time()
                                                            elif "last_speech_time" in calibration_data and "silence_start_time" in calibration_data:
                                                                # Already in silence, check if enough time passed to record it
                                                                current_silence = time.time() - calibration_data["silence_start_time"]
                                                                if current_silence > 0.3:  # Only record meaningful silences
                                                                    # Mark that we've recorded this silence period
                                                                    calibration_data["silence_start_time"] = time.time()

                                                        # If we just detected speech after silence, record the pause duration
                                                        if is_speech and "silence_start_time" in calibration_data:
                                                            silence_duration = time.time() - calibration_data["silence_start_time"]
                                                            if silence_duration > 0.3:  # Only record pauses > 0.3s
                                                                calibration_data["silence_durations"].append(silence_duration)
                                                                calibration_data_shared["silence_durations"].append(silence_duration)
                                                                calibration_state["silence_samples"] = len(calibration_data["silence_durations"])
                                                                print(f"[CALIBRATION] Detected pause: {silence_duration:.2f}s", flush=True)
                                                            # Clear silence tracking
                                                            del calibration_data["silence_start_time"]

                                                        # Check if step 2 is complete
                                                        elapsed = time.time() - calibration_data["start_time"]

                                                        # Debug logging every 5 seconds
                                                        if not hasattr(calibration_data, '_last_log_time'):
                                                            calibration_data['_last_log_time'] = 0
                                                        if elapsed - calibration_data.get('_last_log_time', 0) >= 5:
                                                            print(f"[CALIBRATION-TIMER] Step 2 - elapsed: {elapsed:.1f}s / {calibration_data['duration']}s", flush=True)
                                                            calibration_data['_last_log_time'] = elapsed

                                                        # Force completion if elapsed significantly exceeds duration (safety mechanism)
                                                        if elapsed >= calibration_data["duration"] or elapsed > (calibration_data["duration"] * 2):
                                                            if elapsed > (calibration_data["duration"] * 2):
                                                                print(f"[CALIBRATION] WARNING: Forced completion - elapsed {elapsed:.1f}s exceeds 2x duration {calibration_data['duration']}s", flush=True)
                                                            # Both steps complete - end calibration
                                                            calibration_mode = False
                                                            calibration_state["active"] = False
                                                            print(f"[CALIBRATION] Complete - {len(calibration_data['speech_samples'])} speech samples, {len(calibration_data['noise_samples'])} noise samples, elapsed: {elapsed:.1f}s", flush=True)

                                                except Exception as calib_error:
                                                    print(f"[CALIBRATION] Error: {calib_error}")

                                            # Always stream audio to web clients regardless of VAD/transcription state
                                            if transcription_state.get("audio_stream_enabled", False):
                                                try:
                                                    audio_stream_queue.put_nowait(data)
                                                except Full:
                                                    pass  # Queue full, drop chunk to prevent lag

                                            # Add audio to WhisperLive transcriber buffer (replaces old dual-buffer approach)
                                            if not skip_transcription:
                                                live_transcriber.add_frames(data)
                                                accumulated_new_data += data
                                    finally:
                                        _audio_queue_lock.release()
                                else:
                                    print("[WARN] Failed to acquire audio queue lock, skipping this iteration")

                                # Write accumulated audio data to session file IMMEDIATELY (before phrase logic)
                                if session_audio_file and accumulated_new_data:
                                    try:
                                        # For continuous append, we need to handle WAV format properly
                                        # First write: write full WAV header + data
                                        # Subsequent writes: append only PCM data, update header
                                        if not session_audio_written:
                                            # First write - create WAV file with header
                                            temp_audio = sr.AudioData(
                                                accumulated_new_data,
                                                source.SAMPLE_RATE,
                                                source.SAMPLE_WIDTH,
                                            )
                                            temp_wav = temp_audio.get_wav_data()
                                            with open(session_audio_file, "wb") as f:
                                                f.write(temp_wav)
                                            session_audio_written = True
                                            print(
                                                "[BACKUP] Started session audio file"
                                            )
                                        else:
                                            # Append PCM data only (skip WAV header)
                                            with open(session_audio_file, "ab") as f:
                                                f.write(accumulated_new_data)
                                    except Exception as e:
                                        print(
                                            f"[WARNING] Failed to append to session file: {e}"
                                        )

                                # Calculate and store audio level for volume meter
                                try:
                                    # Convert audio bytes to numpy array for level calculation
                                    audio_np = (
                                        np.frombuffer(
                                            accumulated_new_data, dtype=np.int16
                                        ).astype(np.float32)
                                        / 32768.0
                                    )

                                    # Calculate RMS (Root Mean Square) for audio level
                                    if len(audio_np) > 0:
                                        rms = np.sqrt(np.mean(audio_np**2))
                                    else:
                                        rms = 0

                                    # Convert to dB
                                    if rms > 0:
                                        db = 20 * np.log10(rms)
                                    else:
                                        db = -60  # Silence

                                    # Normalize to 0-100% for display (assuming -60dB to 0dB range)
                                    level = max(0, min(100, (db + 60) * (100 / 60)))

                                    # Convert RMS back to raw energy value
                                    raw_energy = float(rms * 32768.0)

                                    # Store in shared state (parent process will emit via Socket.IO)
                                    transcription_state["audio_level"] = level
                                    transcription_state["audio_db"] = db
                                    transcription_state["audio_energy"] = raw_energy

                                    # Hand the raw pre-VAD buffer to the background
                                    # PANNs detector (non-blocking; never stalls draining).
                                    submit_music_detection(process_config, transcription_state, audio_np, source.SAMPLE_RATE)
                                except Exception as e:
                                    print(f"[WARNING] Audio level calculation failed: {e}")

                                # Check if audio contains speech using VAD
                                speech_detected = has_speech(accumulated_new_data, source.SAMPLE_RATE)

                                # Let confidently-detected music through to transcription even
                                # when VAD would drop it. Music is always transcribed; the
                                # transcribe_detected_music toggle only controls whether its
                                # rows are visible or auto-denied ('music') at insert time.
                                _std_cfg = process_config.get("speech_type_detection", {})
                                if (not speech_detected
                                        and (transcription_state.get("music_prob") or 0.0)
                                            > _std_cfg.get("music_prob_threshold", 0.5)):
                                    speech_detected = True

                                # Check if phrase is complete (silence after speech)
                                phrase_complete = False
                                if not speech_detected:
                                    # No speech - check if we had speech recently and it's been quiet for phrase_timeout
                                    if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                                        phrase_complete = True
                                        print(f"[PHRASE_TIMEOUT] Silence for {phrase_timeout}s detected", flush=True)
                                        # Flush any partial audio buffer to ensure last words are transcribed immediately
                                        if hasattr(source, 'flush_buffer'):
                                            source.flush_buffer()
                                    # Skip transcription if no speech detected (but still check phrase_complete below)
                                    if not phrase_complete:
                                        sleep(0.25)
                                        continue
                                else:
                                    # Speech detected - update phrase_time
                                    phrase_time = now

                                # === WHISPER-LIVE TRANSCRIPTION APPROACH ===
                                # Get audio chunk from the rolling buffer
                                audio_chunk, chunk_duration = live_transcriber.get_audio_chunk_for_processing()

                                # Need at least 1 second of audio before transcribing
                                if chunk_duration < 1.0:
                                    sleep(0.1)
                                    continue

                                # Optional pre-ASR loudness normalization: one gain per pass,
                                # boost-only, applied to the transcription copy. The rolling
                                # buffer, energy gate, and calibration always see raw levels.
                                _norm_cfg = process_config.get("audio", {}).get("loudness_normalization", {})
                                if _norm_cfg.get("enabled", False) and audio_chunk is not None and len(audio_chunk) > 0:
                                    try:
                                        _rms = float(np.sqrt(np.mean(np.square(audio_chunk.astype(np.float64)))))
                                        if _rms > 1e-6:
                                            _target_rms = 10.0 ** (float(_norm_cfg.get("target_rms_dbfs", -20)) / 20.0)
                                            _gain = min(float(_norm_cfg.get("max_gain", 10.0)), _target_rms / _rms)
                                            if _gain > 1.01:
                                                _peak = float(np.max(np.abs(audio_chunk)))
                                                if _peak > 0:
                                                    _gain = min(_gain, 0.99 / _peak)  # never clip
                                                if _gain > 1.01:
                                                    audio_chunk = (audio_chunk * np.float32(_gain)).astype(np.float32)
                                    except Exception as _norm_err:
                                        print(f"[WARNING] Loudness normalization failed: {_norm_err}")

                                try:
                                    # Get language from config
                                    live_language = process_config.get("audio", {}).get("language", "auto")
                                    # Get whisper params from config or use defaults
                                    whisper_params = process_config.get("whisper_decoding", {}).get(
                                        "live_transcription", LIVE_TRANSCRIPTION_PARAMS
                                    )

                                    # Live requires temperature 0: nonzero output varies between
                                    # passes, same_output_threshold never triggers, and no rows
                                    # save. Covers hand-edited configs the save endpoint missed.
                                    _temp_val = whisper_params.get("temperature", 0)
                                    if isinstance(_temp_val, (list, tuple)) or (_temp_val or 0) != 0:
                                        whisper_params = dict(whisper_params)
                                        whisper_params["temperature"] = 0.0
                                        if not globals().get("_live_temp_clamp_warned"):
                                            globals()["_live_temp_clamp_warned"] = True
                                            print(f"[LIVE] temperature {_temp_val!r} forced to 0.0 — nonzero temperature prevents segment finalization (no rows would save)")

                                    # Enable word_timestamps for confidence highlighting if configured
                                    corrections_config = process_config.get("corrections", {})
                                    if corrections_config.get("confidence_highlighting", True) and corrections_config.get("enabled", True):
                                        whisper_params = dict(whisper_params)  # Copy to avoid mutating config
                                        whisper_params["word_timestamps"] = True

                                    # Cross-capture context: feed the tail of the finalized transcript
                                    # as initial_prompt so each capture knows what came before.
                                    # Never include pending_remainder — that audio is still being
                                    # re-transcribed from the rolling buffer and would get echoed.
                                    _ctx_prompt_added = False
                                    _prompt_before_ctx = None
                                    ctx_prompt_cfg = process_config.get("audio", {}).get("context_prompt", {})
                                    if ctx_prompt_cfg.get("enabled", True) and saved_sentences:
                                        ctx_max_chars = ctx_prompt_cfg.get("max_chars", 200)
                                        if ctx_max_chars > 0:
                                            prompt_tail = " ".join(saved_sentences[-5:])
                                            if len(prompt_tail) > ctx_max_chars:
                                                prompt_tail = prompt_tail[-ctx_max_chars:]
                                                _cut = prompt_tail.find(" ")
                                                # Drop the leading partial word; if the tail is one
                                                # unbroken run with no space, discard it entirely
                                                if _cut > 0:
                                                    prompt_tail = prompt_tail[_cut + 1:]
                                                elif _cut < 0:
                                                    prompt_tail = ""
                                            if prompt_tail:
                                                whisper_params = dict(whisper_params)
                                                existing = whisper_params.get("initial_prompt")
                                                _prompt_before_ctx = existing
                                                whisper_params["initial_prompt"] = (existing + " " + prompt_tail) if existing else prompt_tail
                                                _ctx_prompt_added = True


                                    # Transcribe the audio chunk and get segments with timestamps
                                    # This is key to avoiding overlaps - Whisper knows segment boundaries
                                    segments = ModelFactory.transcribe(
                                        audio_model,
                                        processor,
                                        model_type,
                                        audio_chunk,
                                        language=live_language,
                                        whisper_params=whisper_params,
                                        return_segments=True
                                    )

                                    # === Whisper Translation Pass (dual-pass) ===
                                    # If Whisper-based translation is active, run a second pass on the same audio
                                    _whisper_translated_text = None
                                    _pass2_timed = []  # (session_start, session_end, text) per pass-2 segment
                                    _trans_cfg = process_config.get("live_translation", {})
                                    _trans_method = _trans_cfg.get("translation_method", "nllb")
                                    _trans_enabled = _trans_cfg.get("enabled", False)
                                    if _trans_enabled and _trans_method in ("whisper_translate", "whisper_forced_lang") and segments:
                                        try:
                                            _target_lang = _trans_cfg.get("target_language", "en")
                                            _pass2_params = dict(whisper_params)  # Copy pass 1 params
                                            # Drop the source-language context tail for the translation
                                            # pass - it would bias output toward the source language
                                            if _ctx_prompt_added:
                                                if _prompt_before_ctx:
                                                    _pass2_params["initial_prompt"] = _prompt_before_ctx
                                                else:
                                                    _pass2_params.pop("initial_prompt", None)
                                            _pass2_language = live_language

                                            if _trans_method == "whisper_translate" and _target_lang == "en":
                                                _pass2_params["task"] = "translate"
                                            elif _trans_method == "whisper_forced_lang":
                                                _pass2_language = _target_lang

                                            _pass2_segments = ModelFactory.transcribe(
                                                audio_model, processor, model_type,
                                                audio_chunk,
                                                language=_pass2_language,
                                                whisper_params=_pass2_params,
                                                return_segments=True
                                            )
                                            if _pass2_segments:
                                                _whisper_translated_text = " ".join(
                                                    s.get("text", "").strip() for s in _pass2_segments if s.get("text", "").strip()
                                                )
                                                # Session-time pass-2 segments (timestamp_offset hasn't
                                                # advanced yet — update_segments runs after this) so the
                                                # translation can later be scoped to the batch's time span
                                                _pass2_offset = live_transcriber.timestamp_offset
                                                _pass2_timed = [
                                                    (
                                                        _pass2_offset + (s.get("start") or 0),
                                                        _pass2_offset + (s.get("end") or 0),
                                                        s.get("text", "").strip(),
                                                    )
                                                    for s in _pass2_segments if s.get("text", "").strip()
                                                ]
                                                if _whisper_translated_text:
                                                    print(f"[WHISPER-TRANSLATE] Pass 2: '{_whisper_translated_text[:80]}'", flush=True)
                                        except Exception as _wt_err:
                                            print(f"[WHISPER-TRANSLATE] Pass 2 error: {_wt_err}", flush=True)

                                    # Update segments using Whisper-Live's approach
                                    # This finalizes all segments except the last one immediately
                                    result = live_transcriber.update_segments(segments, chunk_duration)

                                    _cjk_filter_enabled = config.get("hallucination_filter", {}).get("cjk_filter_enabled", True)
                                    # Music rows are transcribed but auto-denied when the user
                                    # hasn't opted in to seeing lyrics (restorable in /corrections).
                                    # The threshold in effect is recorded in the deny reason
                                    # ('music:<thr>') so Music Sensitivity can be tuned against
                                    # each denied row's stored music_prob.
                                    _transcribe_music_enabled = process_config.get("speech_type_detection", {}).get("transcribe_detected_music", False)
                                    _music_deny_reason = f"music:{process_config.get('speech_type_detection', {}).get('music_prob_threshold', 0.5):g}"

                                    # Handle completed segments (save to DB)
                                    if result['completed_segments']:
                                        confidence_threshold = process_config.get("corrections", {}).get("confidence_threshold", 0.7)
                                        # Collect the batch of completed segments into one text so
                                        # sentences spanning Whisper's segment boundaries stay intact
                                        batch_parts = []
                                        batch_start = None
                                        batch_end = 0
                                        batch_confidences = []
                                        for segment in result['completed_segments']:
                                            segment_text = segment.get('text', '').strip()
                                            # Compute average word confidence for this segment
                                            word_confidences = segment.get('words', [])
                                            if word_confidences:
                                                probs = [w.get('probability') for w in word_confidences if w.get('probability') is not None]
                                                if probs:
                                                    batch_confidences.append(sum(probs) / len(probs))
                                            if not segment_text:
                                                continue
                                            # Remove overlapping prefix from previous saved text
                                            if saved_sentences:
                                                segment_text = remove_overlapping_prefix(segment_text, saved_sentences[-1])
                                            # The rolling buffer can re-transcribe words already held in
                                            # the pending fragment - strip those too
                                            if segment_text and pending_remainder:
                                                segment_text = remove_overlapping_prefix(segment_text, pending_remainder)
                                            if not segment_text:
                                                continue  # Entire segment was overlap
                                            batch_parts.append(segment_text)
                                            if batch_start is None:
                                                batch_start = segment.get('start', 0)
                                            batch_end = segment.get('end', 0)

                                        if batch_parts:
                                            batch_text = " ".join(batch_parts)
                                            segment_start = batch_start if batch_start is not None else 0
                                            segment_end = batch_end
                                            segment_confidence = sum(batch_confidences) / len(batch_confidences) if batch_confidences else None
                                            segment_speech_type = finalized_audio_type(process_config, transcription_state)
                                            transcription_state['audio_type'] = segment_speech_type
                                            segment_audio_tag = transcription_state.get("audio_tag")
                                            segment_music_prob = transcription_state.get("music_prob")
                                            # Source language ISO code: configured value, or Whisper's
                                            # detected language when audio.language is 'auto'.
                                            _detected_lang = next((s.get('language') for s in result['completed_segments'] if s.get('language')), None)
                                            # Never NULL on a non-blank row: configured -> detected -> 'und' (ISO 639 undetermined)
                                            src_lang = (live_language if (live_language and live_language != "auto") else _detected_lang) or "und"
                                            # Prepend the fragment held from the previous capture so it
                                            # can complete its sentence
                                            if pending_remainder:
                                                batch_text = pending_remainder + " " + batch_text
                                                if pending_remainder_meta:
                                                    segment_start = pending_remainder_meta[0]
                                                    if segment_confidence is None:
                                                        segment_confidence = pending_remainder_meta[2]

                                            # Split into sentences
                                            sentences, remainder = split_into_sentences(batch_text)

                                            if pending_buffer_enabled:
                                                # Hold the incomplete remainder for the next capture
                                                # instead of saving a fragment row
                                                if remainder:
                                                    if sentences or pending_remainder_since is None:
                                                        # Fresh fragment (old one consumed or none existed)
                                                        pending_remainder_since = now
                                                        pending_remainder_meta = (batch_end if sentences else segment_start, batch_end, segment_confidence)
                                                    elif pending_remainder_meta:
                                                        # Fragment still growing - keep its start, extend its end
                                                        pending_remainder_meta = (pending_remainder_meta[0], batch_end, pending_remainder_meta[2])
                                                else:
                                                    pending_remainder_since = None
                                                    pending_remainder_meta = None
                                                pending_remainder = remainder
                                                remainder = ""  # Insert block below must not save the held fragment

                                            # Per-word timing+confidence for words_json, attributed to
                                            # each re-split sentence by max temporal overlap. Built from
                                            # this batch's words (already in hand); a sentence stitched
                                            # across chunks keeps only its in-chunk words, never wrong ones.
                                            _word_stream = words_to_session_ms(result['completed_segments'])
                                            _sentence_word_groups = attribute_words_to_sentences(_word_stream, len(sentences))
                                            _words_source = model_type

                                            # Save substantial sentences to DB
                                            MIN_WORDS = min_words_threshold
                                            _newly_inserted_ids = []  # Track IDs for Whisper translation caching
                                            _accepted_rows = []  # (row_id, text) of non-denied rows — whisper translation targets
                                            with _db_lock:
                                                try:
                                                    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                                                    ts_ms = int(now.timestamp() * 1000)
                                                    for _sidx, sentence in enumerate(sentences):
                                                        # CJK filter: applied here so we have both versions for shadow row
                                                        _cjk_deny = False
                                                        _cjk_shadow = None
                                                        if _cjk_filter_enabled:
                                                            _cjk_stripped = filter_hallucinated_text(sentence, live_language)
                                                            if not _cjk_stripped.strip():
                                                                _cjk_deny = True
                                                                print(f"[CJK→DENIED] '{sentence[:40]}'", flush=True)
                                                            elif _cjk_stripped != sentence:
                                                                _cjk_shadow = sentence  # original with CJK → shadow row
                                                                sentence = _cjk_stripped

                                                        _is_hallucination = is_whisper_hallucination(sentence)
                                                        if _is_hallucination:
                                                            print(f"[HALLUCINATION→DENIED] '{sentence[:40]}'", flush=True)

                                                        _music_deny = (not _transcribe_music_enabled) and segment_speech_type == "Music"
                                                        if _music_deny and not (_is_hallucination or _cjk_deny):
                                                            print(f"[MUSIC→DENIED] '{sentence[:40]}'", flush=True)
                                                        _denied = 1 if (_is_hallucination or _cjk_deny or _music_deny) else 0
                                                        _denied_reason = ('hallucination' if _is_hallucination else 'cjk' if _cjk_deny else _music_deny_reason) if _denied else None

                                                        # original_text = verbatim ASR before profanity normalization
                                                        # (words_json `w` tokens are the fullest-raw form, never filtered)
                                                        _verbatim = sentence
                                                        sentence = apply_profanity_filter(sentence)
                                                        word_count = len(sentence.split())
                                                        is_dup = is_fuzzy_duplicate(sentence, saved_sentences, fuzzy_threshold)
                                                        needs_review = 1 if (segment_confidence is not None and segment_confidence < confidence_threshold) else 0
                                                        _words_json = words_json_or_none(_sentence_word_groups[_sidx] if _sidx < len(_sentence_word_groups) else None)
                                                        if _denied or (word_count >= MIN_WORDS and not is_dup):
                                                            persistent_db_cursor.execute(
                                                                "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, needs_review, speech_type, audio_tag, music_prob, ts_ms, source_language, original_text, words_json, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                                                                (timestamp, sentence, segment_start, segment_end, segment_confidence, needs_review, segment_speech_type, segment_audio_tag, segment_music_prob, ts_ms, src_lang, _verbatim, _words_json, _words_source, live_session_id, _denied, _denied_reason),
                                                            )
                                                            _newly_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                            if not _denied:
                                                                _accepted_rows.append((_newly_inserted_ids[-1], sentence))
                                                                saved_sentences.append(sentence)
                                                                conf_str = f", conf={segment_confidence:.2f}" if segment_confidence is not None else ""
                                                                print(f"[DB INSERT] '{sentence[:50]}...'{conf_str}" if len(sentence) > 50 else f"[DB INSERT] '{sentence}'{conf_str}", flush=True)
                                                            # Shadow row: full original sentence with CJK preserved, denied
                                                            if _cjk_shadow:
                                                                persistent_db_cursor.execute(
                                                                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, needs_review, speech_type, audio_tag, music_prob, ts_ms, source_language, original_text, words_json, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'cjk_shadow')",
                                                                    (timestamp, _cjk_shadow, segment_start, segment_end, segment_confidence, needs_review, segment_speech_type, segment_audio_tag, segment_music_prob, ts_ms, src_lang, _cjk_shadow, _words_json, _words_source, live_session_id),
                                                                )
                                                                _newly_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                                print(f"[CJK SHADOW→DENIED] '{_cjk_shadow[:40]}'", flush=True)
                                                        elif word_count < MIN_WORDS:
                                                            persistent_db_cursor.execute(
                                                                "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, needs_review, speech_type, audio_tag, music_prob, ts_ms, source_language, original_text, words_json, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'short')",
                                                                (timestamp, sentence, segment_start, segment_end, segment_confidence, needs_review, segment_speech_type, segment_audio_tag, segment_music_prob, ts_ms, src_lang, _verbatim, _words_json, _words_source, live_session_id),
                                                            )
                                                            _newly_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                            print(f"[SHORT→DENIED] '{sentence}' ({word_count} words)", flush=True)
                                                        elif is_dup:
                                                            persistent_db_cursor.execute(
                                                                "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, needs_review, speech_type, audio_tag, music_prob, ts_ms, source_language, original_text, words_json, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'dup')",
                                                                (timestamp, sentence, segment_start, segment_end, segment_confidence, needs_review, segment_speech_type, segment_audio_tag, segment_music_prob, ts_ms, src_lang, _verbatim, _words_json, _words_source, live_session_id),
                                                            )
                                                            _newly_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                            print(f"[DUP→DENIED] '{sentence[:40]}'", flush=True)
                                                    # Run-on safety valve: flush a held fragment that never
                                                    # completes a sentence (word cap or age cap exceeded)
                                                    if pending_buffer_enabled and pending_remainder and (
                                                        len(pending_remainder.split()) > pending_max_words
                                                        or (pending_remainder_since is not None and (now - pending_remainder_since).total_seconds() > pending_max_age)
                                                    ):
                                                        remainder = pending_remainder
                                                        if pending_remainder_meta:
                                                            segment_start = pending_remainder_meta[0]
                                                            segment_end = pending_remainder_meta[1]
                                                        pending_remainder = ""
                                                        pending_remainder_since = None
                                                        pending_remainder_meta = None
                                                        print(f"[PENDING FLUSH] Run-on fragment hit cap: '{remainder[:50]}'", flush=True)
                                                    # Save substantial remainder (run-on flush, or pending buffer disabled)
                                                    if remainder:
                                                        # CJK filter on remainder
                                                        _rem_cjk_deny = False
                                                        _rem_cjk_shadow = None
                                                        if _cjk_filter_enabled:
                                                            _rem_stripped = filter_hallucinated_text(remainder, live_language)
                                                            if not _rem_stripped.strip():
                                                                _rem_cjk_deny = True
                                                                print(f"[CJK REMAINDER→DENIED] '{remainder[:40]}'", flush=True)
                                                            elif _rem_stripped != remainder:
                                                                _rem_cjk_shadow = remainder
                                                                remainder = _rem_stripped
                                                        _rem_is_hallucination = is_whisper_hallucination(remainder)
                                                        if _rem_is_hallucination:
                                                            print(f"[HALLUCINATION REMAINDER→DENIED] '{remainder[:40]}'", flush=True)
                                                        _rem_music_deny = (not _transcribe_music_enabled) and segment_speech_type == "Music"
                                                        if _rem_music_deny and not (_rem_is_hallucination or _rem_cjk_deny):
                                                            print(f"[MUSIC REMAINDER→DENIED] '{remainder[:40]}'", flush=True)
                                                        _rem_denied = 1 if (_rem_is_hallucination or _rem_cjk_deny or _rem_music_deny) else 0
                                                        _rem_denied_reason = ('hallucination' if _rem_is_hallucination else 'cjk' if _rem_cjk_deny else _music_deny_reason) if _rem_denied else None
                                                        _verbatim_rem = remainder
                                                        remainder = apply_profanity_filter(remainder)
                                                        rem_word_count = len(remainder.split())
                                                        rem_is_dup = is_fuzzy_duplicate(remainder, saved_sentences, fuzzy_threshold)
                                                        rem_needs_review = 1 if (segment_confidence is not None and segment_confidence < confidence_threshold) else 0
                                                        if _rem_denied or (rem_word_count >= MIN_WORDS and not rem_is_dup):
                                                            # words_json NULL: the remainder is the trailing fragment, not one of
                                                            # the attributed `sentences` (and may be carried text with no words).
                                                            persistent_db_cursor.execute(
                                                                "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, needs_review, speech_type, audio_tag, music_prob, ts_ms, source_language, original_text, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                                                                (timestamp, remainder, segment_start, segment_end, segment_confidence, rem_needs_review, segment_speech_type, segment_audio_tag, segment_music_prob, ts_ms, src_lang, _verbatim_rem, _words_source, live_session_id, _rem_denied, _rem_denied_reason),
                                                            )
                                                            _newly_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                            if not _rem_denied:
                                                                _accepted_rows.append((_newly_inserted_ids[-1], remainder))
                                                                saved_sentences.append(remainder)
                                                                print(f"[DB INSERT REMAINDER] '{remainder[:50]}...'" if len(remainder) > 50 else f"[DB INSERT REMAINDER] '{remainder}'", flush=True)
                                                            if _rem_cjk_shadow:
                                                                persistent_db_cursor.execute(
                                                                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, needs_review, speech_type, audio_tag, music_prob, ts_ms, source_language, original_text, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'cjk_shadow')",
                                                                    (timestamp, _rem_cjk_shadow, segment_start, segment_end, segment_confidence, rem_needs_review, segment_speech_type, segment_audio_tag, segment_music_prob, ts_ms, src_lang, _rem_cjk_shadow, _words_source, live_session_id),
                                                                )
                                                                _newly_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                                print(f"[CJK SHADOW REMAINDER→DENIED] '{_rem_cjk_shadow[:40]}'", flush=True)
                                                        elif rem_word_count < MIN_WORDS:
                                                            persistent_db_cursor.execute(
                                                                "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, needs_review, speech_type, audio_tag, music_prob, ts_ms, source_language, original_text, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'short')",
                                                                (timestamp, remainder, segment_start, segment_end, segment_confidence, rem_needs_review, segment_speech_type, segment_audio_tag, segment_music_prob, ts_ms, src_lang, _verbatim_rem, _words_source, live_session_id),
                                                            )
                                                            _newly_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                            print(f"[SHORT REMAINDER→DENIED] '{remainder}' ({rem_word_count} words)", flush=True)
                                                        elif rem_is_dup:
                                                            persistent_db_cursor.execute(
                                                                "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, needs_review, speech_type, audio_tag, music_prob, ts_ms, source_language, original_text, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'dup')",
                                                                (timestamp, remainder, segment_start, segment_end, segment_confidence, rem_needs_review, segment_speech_type, segment_audio_tag, segment_music_prob, ts_ms, src_lang, _verbatim_rem, _words_source, live_session_id),
                                                            )
                                                            _newly_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                            print(f"[DUP REMAINDER→DENIED] '{remainder[:50]}...'" if len(remainder) > 50 else f"[DUP REMAINDER→DENIED] '{remainder}'", flush=True)
                                                    # segment_id links a transcript row to its translation
                                                    # (same row here); populate it equal to the row id.
                                                    if _newly_inserted_ids:
                                                        persistent_db_cursor.executemany(
                                                            "UPDATE transcriptions SET segment_id = ? WHERE id = ?",
                                                            [(str(_rid), _rid) for _rid in _newly_inserted_ids],
                                                        )
                                                    persistent_db_conn.commit()
                                                    # Cache Whisper translation for accepted rows only, distributing
                                                    # the whole-chunk pass-2 translation across them instead of
                                                    # duplicating it on every row. Denied rows are never emitted
                                                    # as translations, so they get none.
                                                    if _whisper_translated_text and _accepted_rows:
                                                        _target_lang = process_config.get("live_translation", {}).get("target_language", "en")
                                                        _tcache = get_translation_cache()
                                                        # Scope pass-2 text to this batch's time span (drops the
                                                        # in-progress tail's translation); fall back to full text
                                                        _wt_text = scope_whisper_translation(_pass2_timed, batch_end) or _whisper_translated_text
                                                        _parts = distribute_whisper_translation(
                                                            _wt_text, [t for _, t in _accepted_rows]
                                                        )
                                                        for (_row_id, _), _part in zip(_accepted_rows, _parts):
                                                            if not _part:
                                                                continue
                                                            _tcache.set(_row_id, "", _part, _target_lang)
                                                            # Also save to DB translated_text column
                                                            persistent_db_cursor.execute(
                                                                "UPDATE transcriptions SET translated_text = ?, translation_language = ? WHERE id = ?",
                                                                (_part, _target_lang, _row_id),
                                                            )
                                                        persistent_db_conn.commit()
                                                    # Track saved_sentences and database row count
                                                    # Periodically verify database row count matches
                                                    if len(saved_sentences) % 10 == 0:
                                                        db_count = persistent_db_cursor.execute("SELECT COUNT(*) FROM transcriptions").fetchone()[0]
                                                        if db_count != len(saved_sentences) + 1:  # +1 for default first entry
                                                            pass  # Row count mismatch — non-critical
                                                    # print(f"[LOOP-DEBUG] {time.strftime('%H:%M:%S')} - DB commit done", flush=True)
                                                except Exception as db_error:
                                                    print(f"[ERROR] DB save failed: {db_error}")

                                            # print(f"[FINALIZED] '{batch_text[:60]}...'" if len(batch_text) > 60 else f"[FINALIZED] '{batch_text}'")

                                    # Handle phrase completion (silence timeout)
                                    if phrase_complete:
                                        # FIX: Check if update_segments already finalized via same_output
                                        # If so, don't double-process (it's already in completed_segments)
                                        just_finalized = result.get('just_finalized_text', '')
                                        if just_finalized:
                                            finalized_segment = None  # Already handled
                                        else:
                                            # FIX: Capture pending text BEFORE force_finalize (which clears current_out)
                                            # This handles the case where same_output finalization already cleared current_out
                                            pending_text = result.get('current_text', '').strip()

                                            # Force finalize any remaining text
                                            finalized_segment = live_transcriber.force_finalize()

                                            # FIX: If force_finalize returned nothing but we had pending text, create segment from it
                                            if finalized_segment is None and pending_text:
                                                finalized_segment = {
                                                    'text': pending_text,
                                                    'start': live_transcriber.timestamp_offset,
                                                    'end': live_transcriber.timestamp_offset + chunk_duration,
                                                    'completed': True
                                                }

                                        if finalized_segment or pending_remainder:
                                            segment_text = (finalized_segment or {}).get('text', '').strip()
                                            segment_start = (finalized_segment or {}).get('start', 0)
                                            segment_end = (finalized_segment or {}).get('end', 0)
                                            _phrase_words = (finalized_segment or {}).get('words') or []
                                            # Remove overlapping prefix from previous saved text
                                            if segment_text and saved_sentences:
                                                segment_text = remove_overlapping_prefix(segment_text, saved_sentences[-1])
                                            if segment_text and pending_remainder:
                                                segment_text = remove_overlapping_prefix(segment_text, pending_remainder)
                                            # Silence boundary: no more words are coming, so flush the
                                            # held fragment together with (or instead of) the new text
                                            if pending_remainder:
                                                segment_text = (pending_remainder + " " + segment_text).strip() if segment_text else pending_remainder
                                                if pending_remainder_meta:
                                                    segment_start = pending_remainder_meta[0]
                                                    if not segment_end:
                                                        segment_end = pending_remainder_meta[1]
                                                pending_remainder = ""
                                                pending_remainder_since = None
                                                pending_remainder_meta = None

                                            if segment_text:  # Check again after overlap removal
                                                sentences, remainder = split_into_sentences(segment_text)
                                                MIN_WORDS = min_words_threshold
                                                _phrase_speech_type = finalized_audio_type(process_config, transcription_state)
                                                transcription_state['audio_type'] = _phrase_speech_type
                                                _phrase_audio_tag = transcription_state.get("audio_tag")
                                                _phrase_music_prob = transcription_state.get("music_prob")
                                                # Segment-level confidence from per-word probabilities
                                                _phrase_threshold = process_config.get("corrections", {}).get("confidence_threshold", 0.7)
                                                _phrase_probs = [w.get('probability') for w in _phrase_words if w.get('probability') is not None]
                                                _phrase_conf = (sum(_phrase_probs) / len(_phrase_probs)) if _phrase_probs else None
                                                _phrase_needs_review = 1 if (_phrase_conf is not None and _phrase_conf < _phrase_threshold) else 0
                                                # Source language (configured, or Whisper-detected when 'auto')
                                                _phrase_src = (live_language if (live_language and live_language != "auto") else (finalized_segment or {}).get('language')) or "und"
                                                # Per-word words_json attributed to each sentence by max
                                                # temporal overlap (same approach as the batch path).
                                                _phrase_stream = words_to_session_ms([finalized_segment] if finalized_segment else [])
                                                _phrase_word_groups = attribute_words_to_sentences(_phrase_stream, len(sentences))
                                                _phrase_words_source = model_type
                                                _phrase_inserted_ids = []
                                                _phrase_accepted_rows = []  # (row_id, text) of non-denied rows — whisper translation targets
                                                with _db_lock:
                                                    try:
                                                        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                                                        ts_ms = int(now.timestamp() * 1000)
                                                        for _sidx, sentence in enumerate(sentences):
                                                            # CJK filter: applied here so we have both versions for shadow row
                                                            _cjk_deny = False
                                                            _cjk_shadow = None
                                                            if _cjk_filter_enabled:
                                                                _cjk_stripped = filter_hallucinated_text(sentence, live_language)
                                                                if not _cjk_stripped.strip():
                                                                    _cjk_deny = True
                                                                    print(f"[CJK→DENIED] '{sentence[:40]}'", flush=True)
                                                                elif _cjk_stripped != sentence:
                                                                    _cjk_shadow = sentence
                                                                    sentence = _cjk_stripped

                                                            _is_hallucination = is_whisper_hallucination(sentence)
                                                            if _is_hallucination:
                                                                print(f"[HALLUCINATION→DENIED] '{sentence[:40]}'", flush=True)

                                                            _music_deny = (not _transcribe_music_enabled) and _phrase_speech_type == "Music"
                                                            if _music_deny and not (_is_hallucination or _cjk_deny):
                                                                print(f"[MUSIC→DENIED] '{sentence[:40]}'", flush=True)
                                                            _denied = 1 if (_is_hallucination or _cjk_deny or _music_deny) else 0
                                                            _denied_reason = ('hallucination' if _is_hallucination else 'cjk' if _cjk_deny else _music_deny_reason) if _denied else None

                                                            _verbatim = sentence
                                                            sentence = apply_profanity_filter(sentence)
                                                            word_count = len(sentence.split())
                                                            is_dup = is_fuzzy_duplicate(sentence, saved_sentences, fuzzy_threshold)
                                                            _phrase_words_json = words_json_or_none(_phrase_word_groups[_sidx] if _sidx < len(_phrase_word_groups) else None)
                                                            if _denied or (word_count >= MIN_WORDS and not is_dup):
                                                                persistent_db_cursor.execute(
                                                                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, speech_type, audio_tag, music_prob, confidence, needs_review, ts_ms, source_language, original_text, words_json, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                                                                    (timestamp, sentence, segment_start, segment_end, _phrase_speech_type, _phrase_audio_tag, _phrase_music_prob, _phrase_conf, _phrase_needs_review, ts_ms, _phrase_src, _verbatim, _phrase_words_json, _phrase_words_source, live_session_id, _denied, _denied_reason),
                                                                )
                                                                _phrase_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                                if not _denied:
                                                                    _phrase_accepted_rows.append((_phrase_inserted_ids[-1], sentence))
                                                                    saved_sentences.append(sentence)
                                                                    print(f"[DB INSERT PHRASE] '{sentence[:50]}...'" if len(sentence) > 50 else f"[DB INSERT PHRASE] '{sentence}'", flush=True)
                                                                if _cjk_shadow:
                                                                    persistent_db_cursor.execute(
                                                                        "INSERT INTO transcriptions (timestamp, text, start_time, end_time, speech_type, audio_tag, music_prob, confidence, needs_review, ts_ms, source_language, original_text, words_json, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'cjk_shadow')",
                                                                        (timestamp, _cjk_shadow, segment_start, segment_end, _phrase_speech_type, _phrase_audio_tag, _phrase_music_prob, _phrase_conf, _phrase_needs_review, ts_ms, _phrase_src, _cjk_shadow, _phrase_words_json, _phrase_words_source, live_session_id),
                                                                    )
                                                                    _phrase_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                                    print(f"[CJK SHADOW→DENIED] '{_cjk_shadow[:40]}'", flush=True)
                                                            elif word_count < MIN_WORDS:
                                                                persistent_db_cursor.execute(
                                                                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, speech_type, audio_tag, music_prob, confidence, needs_review, ts_ms, source_language, original_text, words_json, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'short')",
                                                                    (timestamp, sentence, segment_start, segment_end, _phrase_speech_type, _phrase_audio_tag, _phrase_music_prob, _phrase_conf, _phrase_needs_review, ts_ms, _phrase_src, _verbatim, _phrase_words_json, _phrase_words_source, live_session_id),
                                                                )
                                                                _phrase_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                                print(f"[SHORT→DENIED] '{sentence}' ({word_count} words)", flush=True)
                                                            elif is_dup:
                                                                persistent_db_cursor.execute(
                                                                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, speech_type, audio_tag, music_prob, confidence, needs_review, ts_ms, source_language, original_text, words_json, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'dup')",
                                                                    (timestamp, sentence, segment_start, segment_end, _phrase_speech_type, _phrase_audio_tag, _phrase_music_prob, _phrase_conf, _phrase_needs_review, ts_ms, _phrase_src, _verbatim, _phrase_words_json, _phrase_words_source, live_session_id),
                                                                )
                                                                _phrase_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                                print(f"[DUP→DENIED] '{sentence[:40]}'", flush=True)
                                                        # Also save substantial remainder from phrase_complete
                                                        if remainder:
                                                            _rem_cjk_deny = False
                                                            _rem_cjk_shadow = None
                                                            if _cjk_filter_enabled:
                                                                _rem_stripped = filter_hallucinated_text(remainder, live_language)
                                                                if not _rem_stripped.strip():
                                                                    _rem_cjk_deny = True
                                                                    print(f"[CJK REMAINDER→DENIED] '{remainder[:40]}'", flush=True)
                                                                elif _rem_stripped != remainder:
                                                                    _rem_cjk_shadow = remainder
                                                                    remainder = _rem_stripped
                                                            _rem_is_hallucination = is_whisper_hallucination(remainder)
                                                            if _rem_is_hallucination:
                                                                print(f"[HALLUCINATION REMAINDER→DENIED] '{remainder[:40]}'", flush=True)
                                                            _rem_music_deny = (not _transcribe_music_enabled) and _phrase_speech_type == "Music"
                                                            if _rem_music_deny and not (_rem_is_hallucination or _rem_cjk_deny):
                                                                print(f"[MUSIC REMAINDER→DENIED] '{remainder[:40]}'", flush=True)
                                                            _rem_denied = 1 if (_rem_is_hallucination or _rem_cjk_deny or _rem_music_deny) else 0
                                                            _rem_denied_reason = ('hallucination' if _rem_is_hallucination else 'cjk' if _rem_cjk_deny else _music_deny_reason) if _rem_denied else None
                                                            _verbatim_rem = remainder
                                                            remainder = apply_profanity_filter(remainder)
                                                            rem_word_count = len(remainder.split())
                                                            rem_is_dup = is_fuzzy_duplicate(remainder, saved_sentences, fuzzy_threshold)
                                                            if _rem_denied or (rem_word_count >= MIN_WORDS and not rem_is_dup):
                                                                persistent_db_cursor.execute(
                                                                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, speech_type, audio_tag, music_prob, confidence, needs_review, ts_ms, source_language, original_text, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                                                                    (timestamp, remainder, segment_start, segment_end, _phrase_speech_type, _phrase_audio_tag, _phrase_music_prob, _phrase_conf, _phrase_needs_review, ts_ms, _phrase_src, _verbatim_rem, _phrase_words_source, live_session_id, _rem_denied, _rem_denied_reason),
                                                                )
                                                                _phrase_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                                if not _rem_denied:
                                                                    _phrase_accepted_rows.append((_phrase_inserted_ids[-1], remainder))
                                                                    saved_sentences.append(remainder)
                                                                if _rem_cjk_shadow:
                                                                    persistent_db_cursor.execute(
                                                                        "INSERT INTO transcriptions (timestamp, text, start_time, end_time, speech_type, audio_tag, music_prob, confidence, needs_review, ts_ms, source_language, original_text, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'cjk_shadow')",
                                                                        (timestamp, _rem_cjk_shadow, segment_start, segment_end, _phrase_speech_type, _phrase_audio_tag, _phrase_music_prob, _phrase_conf, _phrase_needs_review, ts_ms, _phrase_src, _rem_cjk_shadow, _phrase_words_source, live_session_id),
                                                                    )
                                                                    _phrase_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                            elif rem_word_count < MIN_WORDS:
                                                                persistent_db_cursor.execute(
                                                                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, speech_type, audio_tag, music_prob, confidence, needs_review, ts_ms, source_language, original_text, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'short')",
                                                                    (timestamp, remainder, segment_start, segment_end, _phrase_speech_type, _phrase_audio_tag, _phrase_music_prob, _phrase_conf, _phrase_needs_review, ts_ms, _phrase_src, _verbatim_rem, _phrase_words_source, live_session_id),
                                                                )
                                                                _phrase_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                                print(f"[SHORT REMAINDER→DENIED] '{remainder}' ({rem_word_count} words)", flush=True)
                                                            elif rem_is_dup:
                                                                persistent_db_cursor.execute(
                                                                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, speech_type, audio_tag, music_prob, confidence, needs_review, ts_ms, source_language, original_text, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'dup')",
                                                                    (timestamp, remainder, segment_start, segment_end, _phrase_speech_type, _phrase_audio_tag, _phrase_music_prob, _phrase_conf, _phrase_needs_review, ts_ms, _phrase_src, _verbatim_rem, _phrase_words_source, live_session_id),
                                                                )
                                                                _phrase_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                                print(f"[DUP REMAINDER→DENIED] '{remainder[:50]}...'" if len(remainder) > 50 else f"[DUP REMAINDER→DENIED] '{remainder}'", flush=True)
                                                        if _phrase_inserted_ids:
                                                            persistent_db_cursor.executemany(
                                                                "UPDATE transcriptions SET segment_id = ? WHERE id = ?",
                                                                [(str(_rid), _rid) for _rid in _phrase_inserted_ids],
                                                            )
                                                        persistent_db_conn.commit()
                                                        # Cache Whisper translation for accepted phrase rows only,
                                                        # distributing the whole-chunk pass-2 translation across
                                                        # them instead of duplicating it on every row.
                                                        if _whisper_translated_text and _phrase_accepted_rows:
                                                            _target_lang = process_config.get("live_translation", {}).get("target_language", "en")
                                                            _tcache = get_translation_cache()
                                                            # Scope pass-2 text to the finalized span; fall back to full text
                                                            _wt_text = scope_whisper_translation(_pass2_timed, segment_end) or _whisper_translated_text
                                                            _parts = distribute_whisper_translation(
                                                                _wt_text, [t for _, t in _phrase_accepted_rows]
                                                            )
                                                            for (_row_id, _), _part in zip(_phrase_accepted_rows, _parts):
                                                                if not _part:
                                                                    continue
                                                                _tcache.set(_row_id, "", _part, _target_lang)
                                                                persistent_db_cursor.execute(
                                                                    "UPDATE transcriptions SET translated_text = ?, translation_language = ? WHERE id = ?",
                                                                    (_part, _target_lang, _row_id),
                                                                )
                                                            persistent_db_conn.commit()
                                                    except Exception as db_error:
                                                        print(f"[ERROR] phrase_complete DB save failed: {db_error}")

                                                # print(f"[PHRASE_COMPLETE] Finalized: '{segment_text[:40]}...'")

                                        # Clear live preview (single update: readers in the
                                        # Flask process never see a half-written generation)
                                        transcription_state.update({
                                            "live_text": "",
                                            "live_start": 0,
                                            "live_end": 0,
                                        })
                                        # Reset phrase_time so next silence doesn't immediately re-trigger
                                        phrase_time = None

                                    # Update live preview with current incomplete text only
                                    # (finalized segments are already shown separately from the database)
                                    # Prepend any held fragment so it stays visible until its sentence completes
                                    current_text = result.get('current_text', '')
                                    if pending_remainder:
                                        current_text = (pending_remainder + " " + current_text).strip()
                                    if current_text:
                                        # Single update so text/timing/confidence stay consistent
                                        # for readers in the Flask process
                                        _live_update = {
                                            "live_text": current_text,
                                            "live_start": live_transcriber.timestamp_offset,
                                            "live_end": live_transcriber.timestamp_offset + chunk_duration,
                                        }
                                        if hasattr(live_transcriber, '_last_seg_confidence'):
                                            _live_update["live_word_confidences"] = live_transcriber._last_seg_confidence.get('words', [])
                                        transcription_state.update(_live_update)

                                except Exception as transcribe_error:
                                    print(f"[ERROR] Transcription failed: {transcribe_error}")
                                    import traceback
                                    traceback.print_exc()

                                # Infinite loops are bad for processors, must sleep.
                                sleep(0.25)
                        except KeyboardInterrupt:
                            break

                    # Clean up resources before exiting
                    print("\nCleaning up resources...")

                    # Flush any held sentence fragment so it isn't lost on stop
                    if pending_remainder:
                        try:
                            # CJK filter on flush remainder
                            _flush_cjk_deny = False
                            _flush_cjk_shadow = None
                            try:
                                _flush_cjk_enabled = config.get("hallucination_filter", {}).get("cjk_filter_enabled", True)
                                _flush_ll = live_language
                            except NameError:
                                _flush_cjk_enabled = True
                                _flush_ll = None
                            if _flush_cjk_enabled:
                                _flush_stripped = filter_hallucinated_text(pending_remainder, _flush_ll)
                                if not _flush_stripped.strip():
                                    _flush_cjk_deny = True
                                    print(f"[CJK STOP-FLUSH→DENIED] '{pending_remainder[:40]}'", flush=True)
                                elif _flush_stripped != pending_remainder:
                                    _flush_cjk_shadow = pending_remainder
                                    pending_remainder = _flush_stripped
                            _flush_is_hallucination = is_whisper_hallucination(pending_remainder)
                            if _flush_is_hallucination:
                                print(f"[HALLUCINATION STOP-FLUSH→DENIED] '{pending_remainder[:40]}'", flush=True)
                            _flush_denied = 1 if (_flush_is_hallucination or _flush_cjk_deny) else 0
                            _flush_denied_reason = ('hallucination' if _flush_is_hallucination else 'cjk') if _flush_denied else None
                            _verbatim_flush = pending_remainder
                            _flush_text = apply_profanity_filter(pending_remainder)
                            _flush_word_count = len(_flush_text.split())
                            _flush_is_dup = is_fuzzy_duplicate(_flush_text, saved_sentences, fuzzy_threshold)
                            _flush_word_ok = _flush_word_count >= min_words_threshold and not _flush_is_dup
                            _flush_start, _flush_end, _flush_conf = pending_remainder_meta if pending_remainder_meta else (0, 0, None)
                            _flush_now = datetime.now(configured_timezone)
                            _flush_ts_ms = int(_flush_now.timestamp() * 1000)
                            _flush_speech_type = finalized_audio_type(process_config, transcription_state)
                            _flush_audio_tag = transcription_state.get("audio_tag")
                            _flush_music_prob = transcription_state.get("music_prob")
                            _flush_threshold = process_config.get("corrections", {}).get("confidence_threshold", 0.7)
                            _flush_needs_review = 1 if (_flush_conf is not None and _flush_conf < _flush_threshold) else 0
                            try:
                                _flush_ll = live_language
                            except NameError:
                                _flush_ll = None
                            _flush_src = _flush_ll if (_flush_ll and _flush_ll != "auto") else "und"
                            try:
                                _flush_words_source = model_type
                            except NameError:
                                _flush_words_source = None
                            _flush_std_cfg = process_config.get("speech_type_detection", {})
                            if not _flush_denied and _flush_speech_type == "Music" and not _flush_std_cfg.get("transcribe_detected_music", False):
                                _flush_denied = 1
                                _flush_denied_reason = f"music:{_flush_std_cfg.get('music_prob_threshold', 0.5):g}"
                                print(f"[MUSIC STOP-FLUSH→DENIED] '{_flush_text[:40]}'", flush=True)
                            if not _flush_denied and not _flush_word_ok:
                                _flush_denied = 1
                                _flush_denied_reason = 'short' if _flush_word_count < min_words_threshold else 'dup'
                            with _db_lock:
                                # words_json NULL: carried fragment, no per-word data retained.
                                persistent_db_cursor.execute(
                                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, speech_type, audio_tag, music_prob, needs_review, ts_ms, source_language, original_text, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                                    (_flush_now.strftime("%Y-%m-%d %H:%M:%S"), _flush_text, _flush_start, _flush_end, _flush_conf, _flush_speech_type, _flush_audio_tag, _flush_music_prob, _flush_needs_review, _flush_ts_ms, _flush_src, _verbatim_flush, _flush_words_source, live_session_id, _flush_denied, _flush_denied_reason),
                                )
                                _flush_row_id = persistent_db_cursor.lastrowid
                                persistent_db_cursor.execute(
                                    "UPDATE transcriptions SET segment_id = ? WHERE id = ?",
                                    (str(_flush_row_id), _flush_row_id),
                                )
                                if _flush_cjk_shadow:
                                    persistent_db_cursor.execute(
                                        "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, speech_type, audio_tag, music_prob, needs_review, ts_ms, source_language, original_text, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'cjk_shadow')",
                                        (_flush_now.strftime("%Y-%m-%d %H:%M:%S"), _flush_cjk_shadow, _flush_start, _flush_end, _flush_conf, _flush_speech_type, _flush_audio_tag, _flush_music_prob, _flush_needs_review, _flush_ts_ms, _flush_src, _flush_cjk_shadow, _flush_words_source, live_session_id),
                                    )
                                    _shadow_id = persistent_db_cursor.lastrowid
                                    persistent_db_cursor.execute(
                                        "UPDATE transcriptions SET segment_id = ? WHERE id = ?",
                                        (str(_shadow_id), _shadow_id),
                                    )
                                    print(f"[CJK SHADOW STOP-FLUSH→DENIED] '{_flush_cjk_shadow[:40]}'", flush=True)
                                persistent_db_conn.commit()
                            if not _flush_denied:
                                saved_sentences.append(_flush_text)
                                print(f"[DB INSERT STOP-FLUSH] '{_flush_text[:50]}'", flush=True)
                            else:
                                print(f"[STOP-FLUSH {_flush_denied_reason.upper()}→DENIED] '{_flush_text[:50]}'", flush=True)
                        except Exception as _flush_err:
                            print(f"[WARNING] Failed to flush pending fragment on stop: {_flush_err}")
                        pending_remainder = ""
                        pending_remainder_since = None
                        pending_remainder_meta = None

                    # Stop audio source FIRST to release audio device
                    if source:
                        try:
                            print("[CLEANUP] Stopping audio source...")
                            source.stop()
                            print("[CLEANUP] OK: Audio source stopped and ffmpeg terminated")
                        except Exception as e:
                            print(f"[CLEANUP] WARNING: Error stopping audio source: {e}")

                    # Fix WAV header for session audio file (update file size in header)
                    if session_audio_file and session_audio_written:
                        try:
                            # Read all data
                            with open(session_audio_file, "rb") as f:
                                data = f.read()

                            # Recreate proper WAV file with correct header
                            audio_data = sr.AudioData(
                                data[44:], source.SAMPLE_RATE, source.SAMPLE_WIDTH
                            )  # Skip old header
                            correct_wav = audio_data.get_wav_data()

                            with open(session_audio_file, "wb") as f:
                                f.write(correct_wav)

                            print(
                                f"[BACKUP] Full session audio finalized: {session_audio_file}"
                            )
                        except Exception as e:
                            print(
                                f"[WARNING] Failed to finalize session audio header: {e}"
                            )

                    print("[DB-CLEANUP] Starting database cleanup...", flush=True)
                    try:
                        if persistent_db_conn:
                            # Save db_path for SRT conversion and WAL/SHM cleanup later
                            saved_db_path = db_path
                            print(f"[DB-CLEANUP] saved_db_path = {saved_db_path}", flush=True)
                            print(f"[DB-CLEANUP] global db_name = {db_name}", flush=True)

                            # CRITICAL: Clear db_name from state BEFORE cleanup to prevent
                            # web server thread (emit_new_entries) from opening new connections
                            with _transcription_state_lock:
                                transcription_state["db_name"] = None
                                transcription_state["session_id"] = None
                            print("[DB-CLEANUP] Cleared db_name from state (prevents new connections)", flush=True)

                            # Wait for any in-flight emit_new_entries() iterations to complete
                            # emit_new_entries runs every 0.5 seconds, so 1 second should be safe
                            sleep(1)
                            print("[DB-CLEANUP] Waited for web server thread to release connections", flush=True)

                            # Checkpoint WAL to flush all changes to main database file
                            # TRUNCATE mode removes WAL file after checkpoint
                            try:
                                result = persistent_db_cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                                checkpoint_result = result.fetchone()
                                print(f"[DB-CLEANUP] WAL checkpoint completed: {checkpoint_result}", flush=True)
                            except Exception as checkpoint_error:
                                print(f"[DB-CLEANUP] WAL checkpoint failed: {checkpoint_error}", flush=True)

                            persistent_db_conn.close()
                            print("[DB-CLEANUP] Database connection closed", flush=True)

                            # NOTE: WAL/SHM file deletion is deferred until AFTER SRT conversion
                            # because SRT conversion opens a new connection which recreates these files
                            # The actual deletion happens after SRT conversion outside of main()
                        else:
                            print("[DB-CLEANUP] persistent_db_conn is None, skipping cleanup", flush=True)
                    except Exception as e:
                        print(f"[DB-CLEANUP] Error closing DB connection: {e}", flush=True)
                        import traceback
                        traceback.print_exc()

                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            print("[OK] Temp file removed")
                    except Exception as e:
                        print(f"[WARNING] Error removing temp file: {e}")

                    # Clear model references BEFORE cleanup to allow garbage collection
                    audio_model = None
                    processor = None
                    model_type = None
                    vad_model = None

                    # Clean up models
                    ModelFactory.cleanup_models()

            main()

            # Ensure audio source is cleaned up after main() exits
            if source:
                try:
                    print("[CLEANUP] Ensuring audio source cleanup after main() exit...")
                    source.stop()
                    print("[CLEANUP] OK: Audio source cleanup complete")
                except Exception as e:
                    print(f"[CLEANUP] Error in post-main cleanup: {e}")
                finally:
                    source = None

            # Reset model references so next Start can reinitialize
            audio_model = None
            processor = None
            model_type = None
            vad_model = None

            # After main() returns, check if we're not running anymore and update status
            if not is_running:
                # Reset database initialization flag for next session
                global db_initialized
                db_initialized = False

                # Use global db_name for SRT conversion (transcription_state["db_name"] was cleared earlier)
                session_db_name = db_name

                with _transcription_state_lock:
                    transcription_state["running"] = False
                    transcription_state["status"] = "stopped"
                    transcription_state["message"] = "Transcription stopped"
                    transcription_state["error"] = None
                    # db_name already cleared in cleanup code above
                print("[INFO] Transcription stopped successfully")

                # Convert database to SRT before file mover runs
                if session_db_name:
                    # Check if SRT generation is enabled (reload config for fresh settings)
                    fresh_config = load_config()
                    srt_enabled = fresh_config.get("database", {}).get("srt_enabled", True)
                    html_enabled = fresh_config.get("database", {}).get("html_enabled", True)
                    if srt_enabled:
                        try:
                            print(f"[SRT] Converting session database to SRT: {session_db_name}")
                            srt_result = convert_db_to_srt(session_db_name)
                            if srt_result:
                                print("[SRT] Successfully created SRT file")
                            else:
                                print("[SRT] No SRT file created (no valid entries or error)")
                        except Exception as e:
                            print(f"[SRT] Error during SRT conversion: {e}")
                    else:
                        print("[SRT] SRT generation disabled in settings")
                        # Generate HTML separately if SRT is disabled but HTML is enabled
                        if html_enabled:
                            try:
                                print(f"[HTML] Generating HTML file: {session_db_name}")
                                convert_db_to_html(session_db_name)
                            except Exception as e:
                                print(f"[HTML] Error during HTML generation: {e}")

                    # Generate translation SRT if enabled
                    trans_srt_enabled = fresh_config.get("live_translation", {}).get("srt_enabled", True)
                    if trans_srt_enabled and fresh_config.get("live_translation", {}).get("enabled", False):
                        try:
                            print(f"[SRT-TRANSLATION] Converting translations to SRT: {session_db_name}")
                            trans_srt_result = convert_db_to_translation_srt(session_db_name)
                            if trans_srt_result:
                                print("[SRT-TRANSLATION] Successfully created translation SRT file")
                            else:
                                print("[SRT-TRANSLATION] No translation SRT created (no translated entries)")
                        except Exception as e:
                            print(f"[SRT-TRANSLATION] Error: {e}")

                    # NOW delete WAL and SHM files after SRT conversion is complete
                    # SRT conversion opens a new DB connection which recreates these files
                    print("[WAL-CLEANUP] Deleting WAL/SHM files after SRT conversion...", flush=True)
                    try:
                        wal_file = session_db_name + "-wal"
                        shm_file = session_db_name + "-shm"

                        wal_exists = os.path.exists(wal_file)
                        shm_exists = os.path.exists(shm_file)
                        print(f"[WAL-CLEANUP] WAL exists: {wal_exists}, SHM exists: {shm_exists}", flush=True)

                        if wal_exists:
                            os.remove(wal_file)
                            print("[WAL-CLEANUP] WAL file deleted", flush=True)

                        if shm_exists:
                            os.remove(shm_file)
                            print("[WAL-CLEANUP] SHM file deleted", flush=True)

                        if not wal_exists and not shm_exists:
                            print("[WAL-CLEANUP] No WAL/SHM files to delete", flush=True)
                    except Exception as e:
                        print(f"[WAL-CLEANUP] Error deleting WAL/SHM files: {e}", flush=True)

                # File mover is THE VERY LAST operation after everything is fully stopped
                # Wait 10 seconds to ensure all file handles are released
                try:
                    # Reload config to get latest settings (supports hot-reload)
                    current_config = load_config()
                    mover_config = current_config.get("file_manager", {}).get("file_mover", {})
                    if mover_config.get("move_on_transcription_stop", True):
                        print("[FILE MOVER] Waiting 10 seconds for all file handles to close...")
                        set_file_mover_running("auto")
                        sleep(10)
                        print("[FILE MOVER] Executing file move after final cleanup...")
                        result = execute_file_move_now(lambda cfg=current_config: cfg)
                        set_file_mover_result("auto", result)
                        if result['success']:
                            print(f"[FILE MOVER] OK: Moved {result['moved']} files")
                            if result['failed'] > 0:
                                print(f"[FILE MOVER] ! {result['failed']} files failed")
                        else:
                            print(f"[FILE MOVER] FAIL: Error: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    print(f"[FILE MOVER] Error executing file mover: {e}")

                # Final safety net: make the whole DB/backup folder readable by all
                # users — covering every file produced during stop cleanup (SRT/HTML
                # exports, the checkpointed DB, audio) and any pre-existing files.
                try:
                    make_tree_world_readable(BACKUP_DIR)
                    _custom_db_base = (load_config().get("database", {}).get("path", "") or "").strip()
                    if _custom_db_base and os.path.abspath(_custom_db_base) != os.path.abspath(BACKUP_DIR):
                        make_tree_world_readable(_custom_db_base)
                    print("[PERMS] DB/backup folder made world-readable", flush=True)
                except Exception as _perm_err:
                    print(f"[PERMS] Failed to update DB folder permissions: {_perm_err}", flush=True)
    except KeyboardInterrupt:
        print("Thread 1 received KeyboardInterrupt")
        os._exit(0)


def thread2_function():
    try:
        # Get web server config
        web_config = config.get("web_server", {})
        host = web_config.get("host", "0.0.0.0")
        port = web_config.get("port", 80)

        print(f"Starting web server on {host}:{port}")

        # Start the background task for emitting transcriptions
        socketio.start_background_task(emit_new_entries)

        # Start the background task for emitting translations
        socketio.start_background_task(emit_translated_entries)

        # Start audio streaming background tasks
        socketio.start_background_task(emit_audio_stream)
        socketio.start_background_task(emit_tts_audio)

        # Use socketio.run() instead of app.run() for proper Socket.IO support
        socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("Thread 2 received KeyboardInterrupt")
        os._exit(0)


def signal_handler(signum, frame):
    print("\n[SHUTDOWN] Interrupt signal received, stopping threads...")

    # Terminate the transcription process (multiprocessing.Process)
    try:
        if "thread1" in globals() and globals()["thread1"].is_alive():
            print("[SHUTDOWN] Terminating transcription process...")
            globals()["thread1"].terminate()
            globals()["thread1"].join(timeout=3)
            if globals()["thread1"].is_alive():
                print("[SHUTDOWN] Force killing transcription process...")
                globals()["thread1"].kill()
                globals()["thread1"].join(timeout=1)
            print("[SHUTDOWN] Transcription process terminated")
        else:
            print("[SHUTDOWN] Transcription process already stopped")
    except Exception as e:
        print(f"[SHUTDOWN] Error terminating transcription process: {e}")

    # Stop the web server thread
    try:
        if "thread2" in globals() and globals()["thread2"].is_alive():
            print("[SHUTDOWN] Stopping web server thread...")
            # Thread will stop when main exits
        else:
            print("[SHUTDOWN] Web server thread already stopped")
    except Exception as e:
        print(f"[SHUTDOWN] Error checking web server thread: {e}")

    print("[SHUTDOWN] Cleanup complete, exiting...")
    os._exit(0)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    install_crash_diagnostics("main")
    # Bound server.log at startup (small breadcrumb log; rotated across launches)
    try:
        _srv_log = os.path.join(APP_DIR, "server.log")
        if os.path.exists(_srv_log) and os.path.getsize(_srv_log) > 5_000_000:
            os.replace(_srv_log, _srv_log + ".1")
    except OSError:
        pass
    signal.signal(signal.SIGINT, signal_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, signal_handler)

    transcription_process = multiprocessing.Process(
        target=thread1_function,
        args=(transcription_state, control_queue, config_queue,
              calibration_state, calibration_data_shared, calibration_step1_data,
              audio_stream_queue)
    )
    # thread2 = multiprocessing.Process(target=thread2_function)
    # thread1 = threading.Thread(target=thread1_function)
    thread2 = threading.Thread(target=thread2_function)

    transcription_process.start()
    thread2.start()

    # Store references in module for restart endpoint and signal handler
    globals()["thread1"] = transcription_process
    globals()["thread2"] = thread2

    # Use a loop with timeout instead of blocking join
    # This makes the main process responsive to signals
    try:
        while transcription_process.is_alive() or thread2.is_alive():
            transcription_process.join(timeout=1.0)
            thread2.join(timeout=1.0)
    except KeyboardInterrupt:
        print("\nMain process received KeyboardInterrupt, cleaning up...")
        signal_handler(signal.SIGINT, None)
