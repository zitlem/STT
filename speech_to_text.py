import argparse
import io
import os
import warnings

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
_models_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", ".hf_cache")
os.makedirs(_models_cache_dir, exist_ok=True)
os.environ["HF_HUB_CACHE"] = _models_cache_dir
os.environ["HF_HOME"] = _models_cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = _models_cache_dir

# TTS models directory (for piper models)
_tts_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "tts")
os.makedirs(_tts_cache_dir, exist_ok=True)

import sqlite3
import logging
import signal
import threading
import multiprocessing
import platform
from multiprocessing import Queue as MPQueue

import json
import shutil
import statistics
from datetime import timedelta, datetime
import pytz
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
import time
from sys import platform
from file_mover import execute_file_move_now, execute_file_move

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
import mimetypes
import random


# Whisper decoding parameters optimized for streaming (3s chunks)
LIVE_TRANSCRIPTION_PARAMS = {
    "beam_size": 1,  # Greedy decoding - fast, reduces hallucinations
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
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "model": {
        "type": "whisper",
        "whisper": {"model": "small"},
        "backend": "whisper",
        "huggingface": {
            "model_id": "openai/whisper-tiny",
            "use_flash_attention": False,
        },
        "custom": {"model_path": "", "model_type": "whisper"},
    },
    "audio": {
        "backend": "ffmpeg",  # ffmpeg is more reliable than pyaudio
        "energy_threshold": 100,
        "phrase_timeout": 2,
        "default_microphone": "plughw:1,0",
        "device_index": None,
        "language": "auto",
        "same_output_threshold": 7,  # Number of repeated outputs before finalizing text
        "fuzzy_duplicate_threshold": 0.85,
        "min_words": 0,
    },
    "vad": {"enabled": True, "threshold": 0.5},
    "database": {
        "path": "",
        "filename_prefix": "",
        "path_format": "%Y/%m",
        "filename_format": "%Y-%m-%d_%H%M%S",
        "max_entries_to_send": 100,
        "srt_enabled": True,
        "html_enabled": True,
    },
    "web_server": {
        "host": "0.0.0.0",
        "port": 80,
        "update_interval": 0.5,
        "settings_ip_whitelist": ["127.0.0.1", "::1", "10.1.10.0/24"],
        "password_auth": {
            "enabled": True,
            "password": "admin",
            "session_timeout_minutes": 60,
        },
    },
    "performance": {"use_gpu": True},
    "audio_backup": {
        "wav_enabled": True,
        "ts_enabled": True,
        "base_directory": "",
        "filename_prefix": "Recording",
        "format": "wav",
        "path_format": "%Y/%m",
        "filename_format": "%Y-%m-%d_%H%M%S",
    },
    "file_transcription": {
        "model": {
            "type": "whisper",
            "whisper": {"model": "tiny.en"},
            "backend": "",
        },
        "use_gpu": True,
        "language": "auto",
        "translate_enabled": False,
        "translate_to": "en",
        "translation_model": "facebook/nllb-200-distilled-600M",
    },
    "whisper_decoding": {
        "live_transcription": {
            "beam_size": 3,
            "best_of": 1,
            "temperature": 0,
            "condition_on_previous_text": False,
            "compression_ratio_threshold": 1.8,
            "logprob_threshold": -0.5,
            "no_speech_threshold": 0.6,
        },
        "file_transcription": {
            "beam_size": 5,
            "temperature": [0.0, 0.2, 0.4, 0.6, 0.8],
            "condition_on_previous_text": True,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
        },
    },
    "file_manager": {
        "hidden_items": [
            "static",
            "__pycache__",
            ".claude",
            "templates",
            "config.json",
            "models",
            "audio_capture.py",
            "faster_whisper_models.json",
            "file_mover.py",
            "huggingface_manager.py",
            "INSTALL.md",
            "install.sh",
            "install.bat",
            "install.ps1",
            "README.md",
            "requirements.txt",
            "word_highlighting.json",
            "whisper_models.json",
            "speech_to_text.py",
            "start_server.sh",
            "start_server.bat",
            "stop_server.sh",
            "stop_server.bat",
            "restart_server.sh",
            "restart_server.bat",
            "download_progress.json",
        ],
        "file_mover": {
            "move_on_transcription_stop": False,
            "destination_path": "",
            "smb_username": "",
            "smb_password": "",
            "smb_domain": "",
            "source_patterns": ["_AUTOMATIC_BACKUP/**/*"],
            "delete_source": True,
            "preserve_structure": True,
        },
    },
    "url_builder_defaults": {
        "fontSize": "30",
        "bgColor": "000000",
        "textColor": "ffffff",
        "alignment": "center",
        "verticalAlign": "bottom",
        "fontFamily": "Arial",
        "fontWeight": "400",
        "lineHeight": "1",
        "paddingTop": "10",
        "paddingBottom": "20",
        "paddingLeft": "20",
        "paddingRight": "20",
        "maxWidth": "100",
        "inProgressOpacity": "0.9",
        "layout": "side_by_side",
        "inverse": "true",
    },
    "hallucination_filter": {
        "enabled": True,
        "phrases": [
            "Субтитры сделал DimaTorzok",
            "Субтитры делал DimaTorzok",
            "Субтитры подготовил DimaTorzok",
            "Продолжение следует...",
            "Thank you for watching",
            "Thanks for watching",
            "Please subscribe",
            "Like and subscribe",
            "Don't forget to subscribe",
        ],
        "cjk_filter_enabled": True,
    },
    "live_translation": {
        "enabled": True,
        "target_language": "en",
        "source_language": "auto",
        "translate_in_progress": True,
        "display_mode": "translated_only",
        "translation_model": "facebook/nllb-200-distilled-600M",
        "use_gpu": True,
        "max_entries_to_send": 100,
        "srt_enabled": True,
        "html_enabled": True,
        "remote": {
            "enabled": False,
            "endpoint": "",
        },
        "trusted_clients": [],
        "tts": {
            "enabled": False,
            "backend": "edge",
            "edge_voice": "en-US-AriaNeural",
            "piper_model": "",
            "speed": 1.0,
        },
    },
    "corrections": {
        "enabled": True,
        "confidence_threshold": 0.7,
        "confidence_highlighting": True,
        "show_review_queue": True,
        "n_best_alternatives": {
            "translation_count": 3,
        },
        "output_delay": {
            "enabled": False,
            "delay_seconds": 7,
            "auto_publish": True,
        },
    },
    "custom_dictionary": {
        "file": "custom_dictionary.json",
        "whisper_hotwords_enabled": True,
        "whisper_initial_prompt_enabled": True,
        "nllb_glossary_enabled": True,
    },
}


def save_config(config_to_save):
    """Save configuration to config.json with error handling"""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config_to_save, f, indent=2)
        print(f"[OK] Configuration saved to '{CONFIG_FILE}'")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save config: {e}")
        return False


# Word highlighting uses a separate config file
WORD_HIGHLIGHTING_FILE = "word_highlighting.json"


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
        with open(WORD_HIGHLIGHTING_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save word highlighting config: {e}")
        return False


def load_config():
    """Load configuration from config.json, create if doesn't exist"""
    if not os.path.exists(CONFIG_FILE):
        print(
            f"Config file '{CONFIG_FILE}' not found. Creating default config with documentation..."
        )
        try:
            # Create comprehensive config with comments
            full_config = {
                "_comment": "Speech-to-Text Configuration File",
                "_usage": "Edit these values to configure your transcription settings. Command-line arguments will override these defaults.",
                "model": {
                    "type": "whisper",
                    "_type_comment": "Model type: 'whisper' (OpenAI), 'huggingface' (Transformers), or 'custom' (local path)",
                    "_type_options": ["whisper", "huggingface", "custom"],
                    "whisper": {
                        "model": "small",
                        "_model_options": [
                            "tiny",
                            "base",
                            "small",
                            "medium",
                            "large",
                            "large-v1",
                            "large-v2",
                            "large-v3",
                        ],
                        "_model_comment": "Whisper model size. Use .en suffix for English-only (e.g., small.en). tiny/base for real-time, medium/large for accuracy",
                    },
                    "backend": "whisper",
                    "_backend_comment": "Whisper backend: 'whisper' (OpenAI) or 'faster-whisper' (CTranslate2, faster)",
                    "huggingface": {
                        "model_id": "openai/whisper-tiny",
                        "_model_id_comment": "Hugging Face model ID. Popular options below",
                        "_popular_models": {
                            "whisper": [
                                "openai/whisper-tiny",
                                "openai/whisper-base",
                                "openai/whisper-small",
                                "openai/whisper-medium",
                                "openai/whisper-large-v3",
                            ],
                            "distil_whisper": [
                                "distil-whisper/distil-small.en",
                                "distil-whisper/distil-medium.en",
                                "distil-whisper/distil-large-v2",
                                "distil-whisper/distil-large-v3",
                            ],
                            "wav2vec2": [
                                "facebook/wav2vec2-base-960h",
                                "facebook/wav2vec2-large-960h-lv60-self",
                            ],
                            "other": [
                                "facebook/s2t-small-librispeech-asr",
                                "nvidia/stt_en_conformer_ctc_large",
                            ],
                        },
                        "use_flash_attention": False,
                        "_use_flash_attention_comment": "Enable Flash Attention 2 for faster inference (requires compatible GPU)",
                    },
                    "custom": {
                        "model_path": "",
                        "_model_path_comment": "Path to custom model directory or file",
                        "model_type": "whisper",
                        "_model_type_comment": "Architecture type: 'whisper', 'wav2vec2', 'speech2text', etc.",
                    },
                },
                "audio": {
                    "backend": "ffmpeg",
                    "_backend_comment": "Audio capture backend: 'ffmpeg' (recommended, more reliable) or 'pyaudio' (legacy, may hang on some systems)",
                    "_backend_options": ["ffmpeg", "pyaudio"],
                    "default_microphone": "plughw:1,0",
                    "_default_microphone_comment": "Audio device name. For ffmpeg: 'default', 'plughw:0,0', 'plughw:1,0', etc. Use plughw for better format compatibility.",
                    "energy_threshold": 100,
                    "_energy_threshold_comment": "Energy level for mic to detect speech. Lower = more sensitive, Higher = less sensitive",
                    "phrase_timeout": 2,
                    "_phrase_timeout_comment": "Seconds of silence before considering it a new phrase. Higher = fewer false phrase breaks, Lower = more responsive to pauses",
                    "language": "auto",
                    "_language_comment": "Language code for transcription (auto, en, es, fr, de, etc.)",
                    "same_output_threshold": 7,
                    "_same_output_threshold_comment": "Number of repeated outputs before text is finalized. Lower = faster finalization (3-5), Higher = more stable but slower (10-25). Default: 7",
                    "fuzzy_duplicate_threshold": 0.85,
                    "_fuzzy_duplicate_threshold_comment": "Similarity threshold (0.0-1.0) for detecting duplicate sentences in live transcription. Higher = stricter matching. Default: 0.85",
                    "min_words": 0,
                    "_min_words_comment": "Minimum word count to save a segment. 0 = save all segments, 5 = skip short fragments. Default: 0",
                },
                "vad": {
                    "enabled": True,
                    "_enabled_comment": "Enable Voice Activity Detection to filter out music and background noise",
                    "threshold": 0.5,
                    "_threshold_comment": "VAD confidence threshold (0.0-1.0). Examples: 0.3=sensitive, 0.5=balanced, 0.7=strict, 0.9=very strict",
                    "_threshold_recommendations": {
                        "quiet_environments": 0.3,
                        "normal_speech": 0.5,
                        "noisy_with_music": 0.7,
                        "very_noisy": 0.9,
                    },
                },
                "database": {
                    "path": "",
                    "_path_comment": "Custom database base path. Empty = use default (_AUTOMATIC_BACKUP). Can be absolute or relative path.",
                    "path_format": "%Y/%m",
                    "_path_format_comment": "Python strftime format for database subdirectories. Default: %Y/%m (2025/01)",
                    "filename_format": "%Y-%m-%d_%H%M%S",
                    "_filename_format_comment": "Python strftime format for database filenames. MUST include time (%H%M%S) for unique databases per session. Default: %Y-%m-%d_%H%M%S (2025-01-08_143527). Use %Y-%m-%d_%H for hourly databases.",
                    "filename_prefix": "",
                    "_filename_prefix_comment": "Optional prefix for session files. Empty = date only. Example: 'Church' creates '{format}_Church.db'",
                    "max_entries_to_send": 100,
                    "_max_entries_comment": "Number of recent transcriptions to send to web UI. Lower = faster, Higher = more history shown",
                    "srt_enabled": True,
                    "_srt_enabled_comment": "Automatically generate SRT subtitle files alongside database",
                    "html_enabled": True,
                    "_html_enabled_comment": "Automatically generate HTML transcript files alongside database",
                },
                "web_server": {
                    "host": "0.0.0.0",
                    "_host_comment": "Server host. 0.0.0.0 = accessible from network, 127.0.0.1 = localhost only",
                    "port": 8080,
                    "_port_comment": "Server port. Port 80 requires admin/root privileges. Use 8080 for non-privileged access",
                    "update_interval": 0.5,
                    "_update_interval_comment": "Seconds between web UI updates. Lower = more responsive, Higher = less network traffic",
                    "settings_ip_whitelist": ["127.0.0.1", "::1", "10.1.10.0/24"],
                    "_settings_ip_whitelist_comment": "IP addresses or CIDR ranges allowed to access settings. Empty array = allow all. Example: ['127.0.0.1', '192.168.1.0/24']",
                    "password_auth": {
                        "enabled": True,
                        "_enabled_comment": "Enable password authentication as fallback for non-whitelisted IPs",
                        "password": "admin",
                        "_password_comment": "Password for temporary access. Empty = generate random password on startup and display in console",
                        "session_timeout_minutes": 60,
                        "_session_timeout_comment": "How long password-authenticated sessions remain valid (in minutes)",
                    },
                },
                "performance": {
                    "use_gpu": True,
                    "_use_gpu_comment": "Automatically use GPU (CUDA) if available for faster transcription",
                },
                "audio_backup": {
                    "wav_enabled": True,
                    "_wav_enabled_comment": "Save final session audio as .wav file (created when transcription stops)",
                    "ts_enabled": True,
                    "_ts_enabled_comment": "Save continuous MPEG-TS backup (power-fail safe, created during recording)",
                    "base_directory": "",
                    "_base_directory_comment": "Base directory for audio backups. Empty = use default (_AUTOMATIC_BACKUP). Can be absolute or relative path.",
                    "path_format": "%Y/%m",
                    "_path_format_comment": "Python strftime format for audio backup subdirectories. Default: %Y/%m (2025/01)",
                    "filename_format": "%Y-%m-%d_%H%M%S",
                    "_filename_format_comment": "Python strftime format for audio filenames. Default: %Y-%m-%d_%H%M%S (2025-01-08_143527)",
                    "filename_prefix": "",
                    "_filename_prefix_comment": "Optional custom prefix for audio files. Empty = no prefix. Example: 'Recording' creates '{format}_Recording.wav'",
                    "format": "wav",
                    "_format_comment": "Audio format for backups. WAV recommended to prevent file corruption",
                },
                "file_transcription": {
                    "model": {
                        "type": "whisper",
                        "whisper": {
                            "model": "tiny.en",
                        },
                        "backend": "",
                        "_backend_comment": "Backend for file transcription. Empty = use same as main model.",
                    },
                    "use_gpu": True,
                    "_use_gpu_comment": "Use GPU for file transcription if available",
                    "language": "auto",
                    "_language_comment": "Language code for transcription (auto, en, es, fr, de, etc.)",
                    "translate_enabled": True,
                    "translate_to": "en",
                    "translation_model": "facebook/nllb-200-distilled-600M",
                },
                "whisper_decoding": {
                    "_comment": "Advanced Whisper decoding parameters. Only modify if you understand Whisper internals. Defaults are optimized for each use case.",
                    "live_transcription": {
                        "beam_size": 3,
                        "_beam_size_comment": "Search breadth (1=greedy/fast, 5=thorough/slow). Default 1 for live reduces hallucinations.",
                        "best_of": 1,
                        "_best_of_comment": "Number of sampling runs (1=single pass, higher=more options to choose from).",
                        "temperature": 0,
                        "_temperature_comment": "Randomness (0.0=deterministic, higher=creative). Can be array like [0.0, 0.2, 0.4] for fallback.",
                        "condition_on_previous_text": False,
                        "_condition_comment": "Use previous text as context. False for live because short chunks lack context.",
                        "compression_ratio_threshold": 1.8,
                        "_compression_ratio_threshold_comment": "Reject transcriptions with too much repetition. Lower = stricter (rejects more), higher = permissive.",
                        "logprob_threshold": -0.5,
                        "_logprob_threshold_comment": "Reject transcriptions with low confidence scores. Higher = stricter (rejects more).",
                        "no_speech_threshold": 0.6,
                        "_no_speech_threshold_comment": "Probability threshold for detecting silence (0.0-1.0).",
                    },
                    "file_transcription": {
                        "beam_size": 5,
                        "_beam_size_comment": "Search breadth (1=greedy/fast, 5=thorough/slow). Default 5 for file prioritizes quality.",
                        "temperature": [0.0, 0.2, 0.4, 0.6, 0.8],
                        "_temperature_comment": "Temperature fallback sequence. Tries deterministic first, then increases creativity if fails.",
                        "condition_on_previous_text": True,
                        "_condition_comment": "Use previous text as context. True for file because 30s chunks have context.",
                        "compression_ratio_threshold": 2.4,
                        "logprob_threshold": -1.0,
                        "no_speech_threshold": 0.6,
                    },
                },
                "file_manager": {
                    "hidden_items": [
                        "static",
                        "__pycache__",
                        ".claude",
                        "templates",
                        "config.json",
                        "models",
                        "audio_capture.py",
                        "faster_whisper_models.json",
                        "file_mover.py",
                        "huggingface_manager.py",
                        "INSTALL.md",
                        "install.sh",
                        "install.bat",
                        "install.ps1",
                        "README.md",
                        "requirements.txt",
                        "word_highlighting.json",
                        "whisper_models.json",
                        "speech_to_text.py",
                        "start_server.sh",
                        "start_server.bat",
                        "stop_server.sh",
                        "stop_server.bat",
                        "restart_server.sh",
                        "restart_server.bat",
                        "download_progress.json",
                    ],
                    "_hidden_items_comment": "List of specific file/folder paths to hide in file manager. Paths are relative to the working directory. Example: ['templates', 'config.json', 'models/cached']",
                    "file_mover": {
                        "move_on_transcription_stop": False,
                        "_move_on_transcription_stop_comment": "Automatically move files when transcription stops",
                        "destination_path": "",
                        "_destination_path_comment": "Destination path for files (can be SMB path like //server/share/folder or local path)",
                        "smb_username": "",
                        "_smb_username_comment": "Username for SMB/network share authentication (leave empty for local paths)",
                        "smb_password": "",
                        "_smb_password_comment": "Password for SMB/network share authentication (stored in plain text - use with caution)",
                        "smb_domain": "",
                        "_smb_domain_comment": "Domain for SMB authentication (optional, can be empty)",
                        "source_patterns": ["_AUTOMATIC_BACKUP/**/*"],
                        "_source_patterns_comment": "File patterns to move (glob patterns relative to working directory)",
                        "delete_source": True,
                        "_delete_source_comment": "Delete source file after successful move (false = copy instead)",
                        "preserve_structure": True,
                        "_preserve_structure_comment": "Preserve directory structure when moving files",
                    },
                },
                "url_builder_defaults": {
                    "fontSize": "30",
                    "bgColor": "000000",
                    "textColor": "ffffff",
                    "alignment": "center",
                    "verticalAlign": "bottom",
                    "fontFamily": "Arial",
                    "fontWeight": "400",
                    "lineHeight": "1",
                    "paddingTop": "10",
                    "paddingBottom": "20",
                    "paddingLeft": "20",
                    "paddingRight": "20",
                    "maxWidth": "100",
                    "inProgressOpacity": "0.9",
                    "layout": "side_by_side",
                    "inverse": "true",
                },
                "hallucination_filter": {
                    "enabled": True,
                    "_enabled_comment": "Filter out known Whisper hallucinations (phantom subtitles from training data)",
                    "phrases": [
                        "Субтитры сделал DimaTorzok",
                        "Субтитры делал DimaTorzok",
                        "Субтитры подготовил DimaTorzok",
                        "Продолжение следует...",
                        "Thank you for watching",
                        "Thanks for watching",
                        "Please subscribe",
                        "Like and subscribe",
                        "Don't forget to subscribe",
                    ],
                    "_phrases_comment": "List of phrases to filter. Partial matches are detected (phrase in text or text in phrase)",
                    "cjk_filter_enabled": True,
                    "_cjk_filter_enabled_comment": "Filter out hallucinated CJK characters when transcribing Latin-based languages",
                },
                "live_translation": {
                    "enabled": True,
                    "_enabled_comment": "Enable real-time translation of live transcription",
                    "target_language": "en",
                    "_target_language_comment": "Target language code (e.g., 'en', 'es', 'fr', 'de', 'ru')",
                    "source_language": "auto",
                    "_source_language_comment": "Source language (auto uses configured audio.language)",
                    "translate_in_progress": True,
                    "_translate_in_progress_comment": "Also translate in-progress text (may cause flicker)",
                    "display_mode": "translated_only",
                    "_display_mode_comment": "Display mode: translated_only, side_by_side, stacked",
                    "_display_mode_options": ["translated_only", "side_by_side", "stacked"],
                    "translation_model": "facebook/nllb-200-distilled-600M",
                    "_translation_model_comment": "HuggingFace model ID for translation",
                    "use_gpu": True,
                    "_use_gpu_comment": "Use GPU for translation model. Disable to run NLLB on CPU and free VRAM for Whisper",
                    "max_entries_to_send": 100,
                    "_max_entries_to_send_comment": "Number of recent translations to send to web UI. Lower = faster, Higher = more history shown",
                    "srt_enabled": True,
                    "_srt_enabled_comment": "Generate SRT subtitle files for translations",
                    "html_enabled": True,
                    "_html_enabled_comment": "Generate HTML transcript files for translations",
                    "remote": {
                        "enabled": False,
                        "_enabled_comment": "Offload translation to another machine running this app",
                        "endpoint": "",
                        "_endpoint_comment": "URL of remote machine (e.g. http://192.168.1.200:80). Leave empty to use local translation.",
                    },
                    "trusted_clients": [],
                    "_trusted_clients_comment": "IPs allowed to use this machine for remote translation (managed automatically via pairing)",
                    "tts": {
                        "enabled": False,
                        "_enabled_comment": "Enable text-to-speech for translated output",
                        "backend": "edge",
                        "_backend_comment": "TTS backend: 'edge' (Microsoft Edge cloud, 400+ voices, requires internet) or 'piper' (local/offline, download models from Model Manager)",
                        "edge_voice": "en-US-AriaNeural",
                        "_edge_voice_comment": "Edge TTS voice name. Browse voices in Translation Settings page.",
                        "piper_model": "",
                        "_piper_model_comment": "Piper model ID (e.g. 'en_US-lessac-medium'). Download from Model Manager.",
                        "speed": 1.0,
                        "_speed_comment": "Speech speed multiplier (0.5 = half speed, 2.0 = double speed)",
                    },
                },
                "corrections": {
                    "enabled": True,
                    "_enabled_comment": "Enable correction and confidence features",
                    "confidence_threshold": 0.7,
                    "_confidence_threshold_comment": "Minimum confidence score (0.0-1.0) to accept transcription without review",
                    "confidence_highlighting": True,
                    "_confidence_highlighting_comment": "Highlight low-confidence words in the UI",
                    "show_review_queue": True,
                    "n_best_alternatives": {
                        "translation_count": 3,
                        "_translation_count_comment": "Number of alternative translations to generate",
                    },
                    "output_delay": {
                        "enabled": False,
                        "_enabled_comment": "Delay output to allow corrections before publishing",
                        "delay_seconds": 7,
                        "auto_publish": True,
                    },
                },
                "custom_dictionary": {
                    "file": "custom_dictionary.json",
                    "_file_comment": "Path to custom dictionary JSON file for domain-specific terms",
                    "whisper_hotwords_enabled": True,
                    "_whisper_hotwords_enabled_comment": "Pass dictionary terms as hotwords to Whisper for better recognition",
                    "whisper_initial_prompt_enabled": True,
                    "_whisper_initial_prompt_enabled_comment": "Include dictionary terms in Whisper initial prompt for context",
                    "nllb_glossary_enabled": True,
                    "_nllb_glossary_enabled_comment": "Use dictionary for NLLB translation glossary (forced term translations)",
                },
            }

            with open(CONFIG_FILE, "w") as f:
                json.dump(full_config, f, indent=2)
            print(f"[OK] Created '{CONFIG_FILE}' with full documentation.")
            print(f"[NOTE] You can edit this file to change default settings.")

            # Return the config without comments for runtime use
            return DEFAULT_CONFIG
        except Exception as e:
            print(f"WARNING:️ Warning: Could not create config file: {e}")
            return DEFAULT_CONFIG

    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        print(f"[OK] Loaded configuration from '{CONFIG_FILE}'")

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
                with open(CONFIG_FILE, "w") as f:
                    json.dump(config, f, indent=2)
                print("[MIGRATION] Config file updated and saved")
            except Exception as e:
                print(f"[MIGRATION] Warning: Could not save migrated config: {e}")

        return config
    except Exception as e:
        print(f"[ERROR] Error loading config file: {e}")
        print("Using default configuration.")
        return DEFAULT_CONFIG


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
    local_model_path = os.path.join(os.getcwd(), "models", local_dir_name)

    if os.path.exists(local_model_path):
        model_path = local_model_path
        print(f"[INFO] Loading translation model from local: {model_path}")
    else:
        model_path = model_id
        print(f"[INFO] Loading translation model from HuggingFace: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
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
            dict_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), dict_file)

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
        for source_term, target_term in glossary.items():
            # Case-insensitive replacement
            text = re.sub(re.escape(source_term), target_term, text, flags=re.IGNORECASE)

        return text
    except Exception:
        return text


def translate_text(text, source_lang, target_lang, model, tokenizer, return_confidence=False, num_alternatives=0):
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

    # Build generate kwargs
    generate_kwargs = {
        "forced_bos_token_id": tokenizer.convert_tokens_to_ids(tgt_nllb),
        "max_length": 1024,
        "num_beams": 5,
        "early_stopping": True,
    }

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


def translate_segments(segments, source_lang, target_lang, model, tokenizer, progress_callback=None):
    """
    Translate a list of transcription segments

    Args:
        segments: List of segment dicts with 'text', 'start', 'end' keys
        source_lang: Source language ISO code
        target_lang: Target language ISO code
        model: Loaded NLLB model
        tokenizer: Loaded NLLB tokenizer
        progress_callback: Optional callback function(percent, status) for progress updates

    Returns:
        List of translated segment dicts with same structure
    """
    translated_segments = []
    total = len(segments)

    for i, seg in enumerate(segments):
        translated_text = translate_text(seg["text"], source_lang, target_lang, model, tokenizer)
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
_live_translation_target_lang = None


def get_live_translation_model(use_gpu=True, model_id=None):
    """Get or load the live translation model (singleton pattern)"""
    global _live_translation_model, _live_translation_tokenizer, _live_translation_model_loaded, _live_translation_model_loading

    with _live_translation_lock:
        # Don't load model if transcription is actively stopping (to prevent GPU memory leak)
        # Only block during "stopping" - allow loading when "stopped" (user may start again)
        status = transcription_state.get("status", "")
        if _live_translation_model is None and status == "stopping":
            print(f"[LIVE-TRANSLATION] Skipping model load - transcription is stopping")
            return None, None

        if _live_translation_model is None:
            _live_translation_model_loading = True
            try:
                # Use GPU for translation - worker stays alive so no CUDA fork issues
                print("[LIVE-TRANSLATION] Loading live translation model on GPU...")
                _live_translation_model, _live_translation_tokenizer = load_translation_model(
                    use_gpu=use_gpu,  # Use GPU for speed (worker stays alive, no fork issues)
                    model_id=model_id
                )
                _live_translation_model_loaded = True
                print("[LIVE-TRANSLATION] Live translation model loaded on GPU")
            finally:
                _live_translation_model_loading = False
        return _live_translation_model, _live_translation_tokenizer


def unload_live_translation_model():
    """Unload the live translation model to free GPU memory"""
    global _live_translation_model, _live_translation_tokenizer, _live_translation_model_loaded
    import gc

    with _live_translation_lock:
        if _live_translation_model is not None:
            print("[LIVE-TRANSLATION] Unloading live translation model...")
            del _live_translation_model
            del _live_translation_tokenizer
            _live_translation_model = None
            _live_translation_tokenizer = None
            _live_translation_model_loaded = False
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
_tts_downloading = False
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
        mp3_bytes = loop.run_until_complete(_do_synth())
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

    def get(self, segment_id, original_text, target_lang):
        """Get cached translation or None"""
        with self._lock:
            entry = self._cache.get(segment_id)
            if entry and entry['original'] == original_text and entry['target_lang'] == target_lang:
                return entry['translated']
            return None

    def set(self, segment_id, original_text, translated_text, target_lang):
        """Cache a translation"""
        with self._lock:
            if len(self._cache) >= self._max_size:
                # Remove oldest entries (simple LRU - remove first 100 entries)
                oldest = list(self._cache.keys())[:100]
                for key in oldest:
                    del self._cache[key]
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
            if len(self._cache) >= self._max_size:
                oldest = list(self._cache.keys())[:100]
                for key in oldest:
                    del self._cache[key]
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
        models_dir = os.path.join(os.getcwd(), "models")
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
        models_dir = os.path.join(os.getcwd(), "models")
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
        models_dir = os.path.join(os.getcwd(), "models")
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
                # Original Whisper model
                # Build params dict with language and whisper_params
                params = {}
                if language != "auto":
                    params["language"] = language
                if whisper_params:
                    # Filter out comment fields (keys starting with "_")
                    filtered_params = {k: v for k, v in whisper_params.items() if not k.startswith("_")}
                    params.update(filtered_params)

                result = model.transcribe(audio_data, **params)

                if return_segments:
                    # Return Whisper's native segments with timestamps
                    segments = result.get("segments", [])
                    return [{"text": seg["text"].strip(), "start": seg["start"], "end": seg["end"]} for seg in segments if seg["text"].strip()]
                return result["text"].strip()

            elif model_type == "faster_whisper":
                # faster-whisper model (CTranslate2-based)
                # Build params dict with language and whisper_params
                params = {"vad_filter": True}  # Enable VAD by default for live transcription
                if language != "auto":
                    params["language"] = language

                # faster-whisper supported parameters (different from standard whisper)
                faster_whisper_params = {
                    "beam_size", "best_of", "patience", "length_penalty",
                    "repetition_penalty", "no_repeat_ngram_size",
                    "temperature", "compression_ratio_threshold",
                    "log_prob_threshold", "no_speech_threshold",
                    "condition_on_previous_text", "initial_prompt",
                    "prefix", "suppress_blank", "suppress_tokens",
                    "without_timestamps", "max_initial_timestamp",
                    "word_timestamps", "prepend_punctuations",
                    "append_punctuations", "vad_filter", "vad_parameters",
                    "hotwords",
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
                    if "temperature" in whisper_params and isinstance(whisper_params["temperature"], (int, float)):
                        generate_kwargs["temperature"] = whisper_params["temperature"]

                if return_segments:
                    # Request timestamps from HuggingFace pipeline
                    result = model(audio_data, return_timestamps=True, generate_kwargs=generate_kwargs) if generate_kwargs else model(audio_data, return_timestamps=True)
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

                result = model(audio_data, generate_kwargs=generate_kwargs) if generate_kwargs else model(audio_data)
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
                except:
                    # Fallback if language parameter not supported
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
            # print(f"[DEBUG-BUFFER] Garbage detected: '{all_text[:30]}', forcing buffer trim", flush=True)
            with self.lock:
                if self.frames_np is not None:
                    buffer_duration = self.frames_np.shape[0] / self.RATE
                    current_pos = self.timestamp_offset - self.frames_offset
                    chunk_to_process = buffer_duration - current_pos
                    if chunk_to_process > 10:
                        # Force advance to keep only 10 seconds
                        extra_advance = chunk_to_process - 10
                        self.timestamp_offset += extra_advance
                        # print(f"[DEBUG-BUFFER] Garbage trim: advanced by {extra_advance:.1f}s (was {chunk_to_process:.1f}s)", flush=True)
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
                    # print(f"[DEBUG-TRANSCRIBER] Skipping segment (start >= end): start={start:.2f}, end={end:.2f}, text='{text[:50]}'", flush=True)
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
                self.transcript.append(completed)
                result['completed_segments'].append(completed)
                # print(f"[SEGMENT] Finalized: '{text[:50]}...' ({seg.get('start', 0):.1f}s-{seg.get('end', 0):.1f}s)" if len(text) > 50 else f"[SEGMENT] Finalized: '{text}'", flush=True)
                offset = min(duration, seg.get('end', duration))

        # Handle the last segment (in-progress until repeated)
        last_seg = segments[-1]
        self.current_out = last_seg.get('text', '').strip()
        # Store last segment's confidence data for finalization
        self._last_seg_confidence = {
            k: last_seg[k] for k in ('words', 'avg_logprob', 'no_speech_prob') if k in last_seg
        }
        result['current_text'] = self.current_out

        # Check if last segment is repeating (same_output_threshold logic)
        if self._is_similar_output(self.current_out, self.prev_out) and self.current_out:
            self.same_output_count += 1
            if self.end_time_for_same_output is None:
                self.end_time_for_same_output = last_seg.get('end', duration)

            # Debug logging for same_output tracking
            # print(f"[DEBUG-SAME-OUTPUT] count={self.same_output_count}, threshold={self.same_output_threshold}, current='{self.current_out[:40]}', prev='{self.prev_out[:40]}'", flush=True)
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
                    # print(f"[DEBUG-BUFFER] PROACTIVE: Advanced by {extra_advance:.1f}s (chunk was {chunk_to_process:.1f}s)", flush=True)

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
            "audio_level": 0,  # Audio level for histogram (0-100)
            "audio_db": -60,  # Audio level in decibels
            "audio_energy": 0,  # Raw audio energy (RMS)
            "live_text": "",  # Live preview text (not yet saved to DB)
            "loaded_model": "",  # Name of the actual model that was loaded
            "audio_stream_enabled": False,  # Whether to stream audio to web clients
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
_audio_queue_lock = threading.Lock()
# Generate the current date and time as a string
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M_%A")
current_year = datetime.now().strftime("%Y")
current_month = datetime.now().strftime("%Y-%m")

# Database will be created lazily when transcription starts
db_name = None  # Will be set when database is initialized
db_initialized = False


def initialize_database():
    """Initialize database only when transcription starts (lazy loading)"""
    global db_name, db_initialized

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
        # Use default base path + path_format subdirectory
        folder_name = os.path.join("_AUTOMATIC_BACKUP", formatted_path)
        print(f"[OK] Using default database path: {folder_name}")

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)

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
                # So we need to recreate the table
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

            # Insert a blank first entry with default values
            default_timestamp = " "
            default_text = " "
            db_cursor.execute(
                "INSERT INTO transcriptions (timestamp, text) VALUES (?, ?)",
                (default_timestamp, default_text),
            )
            db_connection.commit()

        db_initialized = True
        # Store database name in shared state for web server access
        transcription_state["db_name"] = db_name
        print("[OK] Database initialized successfully")

        return db_name
    except Exception as e:
        print(f"[ERROR] Failed to initialize database: {e}")
        # Clean up database file if initialization failed
        if db_name and os.path.exists(db_name):
            try:
                os.unlink(db_name)
            except:
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
            except:
                pass
        raise Exception(f"Failed to extract audio: {str(e)}")


def chunk_audio_file(audio_path, chunk_duration=30):
    """
    Split audio file into chunks for processing.

    Args:
        audio_path: Path to WAV audio file
        chunk_duration: Duration of each chunk in seconds

    Returns:
        List of (audio_chunk_array, start_time, end_time) tuples
    """
    try:
        import librosa

        # Load audio
        audio_data, sr = librosa.load(audio_path, sr=16000)

        # Calculate chunk size in samples
        chunk_size = chunk_duration * sr
        total_duration = len(audio_data) / sr

        chunks = []
        for start_sample in range(0, len(audio_data), chunk_size):
            end_sample = min(start_sample + chunk_size, len(audio_data))
            chunk_data = audio_data[start_sample:end_sample]

            start_time = start_sample / sr
            end_time = end_sample / sr

            chunks.append((chunk_data, start_time, end_time))

        return chunks

    except Exception as e:
        raise Exception(f"Failed to chunk audio: {str(e)}")


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
                ORDER BY id ASC
            """
            )
            entries = cursor.fetchall()

        if not entries:
            print(f"[SRT] No valid entries found in database")
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
            print(f"[SRT] No valid segments after parsing")
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
            os.path.dirname(os.path.abspath(__file__)), "word_highlighting.json"
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
                ORDER BY id ASC
            """
            )
            entries = cursor.fetchall()

        if not entries:
            print(f"[HTML] No valid entries found in database")
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
            print(f"[HTML] No valid segments after parsing")
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


app = Flask(__name__)
app.config["SECRET_KEY"] = "your_secret_keyss"
app.config["TEMPLATES_AUTO_RELOAD"] = True  # Auto-reload templates when they change
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # Disable caching for static files
socketio = SocketIO(app, static_url_path="/static", static_folder="static", ping_timeout=120, ping_interval=25)

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
        # No parameters provided, check for saved defaults
        defaults = config.get("url_builder_defaults", {})
        if defaults:
            # Redirect to root with default parameters
            from flask import redirect, url_for
            return redirect(url_for('index', **defaults))

    response = make_response(render_template("index.html"))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response


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

        # Write to config file
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)

        # Send config update through queue for hot-reload
        try:
            config_queue.put({"type": "config_update", "config": config})
        except:
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

        # Reset to defaults
        config = DEFAULT_CONFIG.copy()

        # Write default config to file
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)

        # Send config update through queue
        try:
            config_queue.put({"type": "config_update", "config": config})
        except:
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
                "phrase_timeout": config.get("audio", {}).get("phrase_timeout", 2.5),
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


@app.route("/api/url-builder/defaults", methods=["POST"])
def save_url_builder_defaults():
    """API endpoint to save URL builder default settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    try:
        data = request.get_json()
        if "url_builder_defaults" not in config:
            config["url_builder_defaults"] = {}

        config["url_builder_defaults"] = data
        save_config(config)

        return jsonify({"success": True, "message": "Default settings saved"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/url-builder/defaults", methods=["GET"])
def get_url_builder_defaults():
    """API endpoint to get URL builder default settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    defaults = config.get("url_builder_defaults", {})
    return jsonify({"success": True, "defaults": defaults})


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

    # Update settings
    for key in ["enabled", "target_language", "source_language", "translate_in_progress",
                "display_mode", "translation_model", "use_gpu"]:
        if key in data:
            config["live_translation"][key] = data[key]

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

    # Handle model loading/unloading based on enabled state
    now_enabled = config["live_translation"].get("enabled", False)
    new_target_lang = config["live_translation"].get("target_language", "en")
    new_model = config["live_translation"].get("translation_model", "")
    new_use_gpu = config["live_translation"].get("use_gpu", True)

    model_changed = old_model != new_model or old_use_gpu != new_use_gpu

    if not now_enabled and was_enabled:
        # Translation just disabled - unload model
        threading.Thread(target=unload_live_translation_model, daemon=True).start()
    elif now_enabled and (not was_enabled or model_changed):
        # Translation just enabled, or model/GPU setting changed - reload model
        # Skip eager loading if this machine serves remote clients (Machine B) —
        # model will be loaded when Machine A starts transcription via /api/translate/preload
        if _trusted_translation_clients:
            if was_enabled and model_changed:
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

    # Clear cache if target language or model changed
    if new_target_lang != old_target_lang or model_changed:
        get_translation_cache().clear()

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

    # Don't clear cache — old segments keep their translations
    # Only new segments will be translated to the new language

    language_name = TRANSLATION_LANGUAGES.get(new_language, new_language)
    print(f"[LIVE-TRANSLATION] Hot-switched language: {old_language} -> {new_language} ({language_name})")

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

    result = {
        "success": True,
        "enabled": trans_config.get("enabled", False),
        "target_language": trans_config.get("target_language", "en"),
        "target_language_name": TRANSLATION_LANGUAGES.get(
            trans_config.get("target_language", "en"), "English"
        ),
        "model_loaded": True if remote_active else is_live_translation_model_loaded(),
        "model_loading": False if remote_active else is_live_translation_model_loading(),
        "translation_model": trans_config.get("translation_model", "facebook/nllb-200-distilled-600M"),
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

    result = translate_live_text(text, source_lang, target_lang,
                                 return_extras=return_extras,
                                 num_alternatives=num_alternatives)

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
        print(f"[PRELOAD] Translation model loaded and ready")

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
        "downloading": _tts_downloading,
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

    if _tts_downloading or _tts_download_status.get("status") == "downloading":
        return jsonify({"success": False, "error": "A TTS download is already in progress"}), 409

    def _do_download():
        global _tts_downloading, _tts_download_status
        _tts_downloading = True
        _tts_download_status = {"status": "downloading", "model": model_name, "error": ""}
        try:
            import urllib.request

            # Parse model ID: e.g., "en_US-lessac-medium" -> lang "en_US", name "lessac", quality "medium"
            parts = model_name.split("-")
            if len(parts) < 3:
                raise ValueError(f"Invalid piper model ID format: {model_name}")

            lang_code = parts[0]  # e.g., "en_US"
            voice_name = parts[1]  # e.g., "lessac"
            quality = parts[2]     # e.g., "medium"

            # HuggingFace piper voices URL
            lang_family = lang_code.split("_")[0]  # "en_US" -> "en"
            base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang_family}/{lang_code}/{voice_name}/{quality}"
            onnx_filename = f"{model_name}.onnx"
            json_filename = f"{model_name}.onnx.json"

            model_dir = _get_piper_model_dir(model_name)
            os.makedirs(model_dir, exist_ok=True)

            onnx_url = f"{base_url}/{onnx_filename}"
            json_url = f"{base_url}/{json_filename}"

            print(f"[TTS] Downloading piper model: {model_name}")
            print(f"[TTS]   ONNX: {onnx_url}")

            urllib.request.urlretrieve(onnx_url, os.path.join(model_dir, onnx_filename))
            print(f"[TTS]   JSON config: {json_url}")
            urllib.request.urlretrieve(json_url, os.path.join(model_dir, json_filename))

            print(f"[TTS] Piper model downloaded: {model_name}")
            _tts_download_status = {"status": "completed", "model": model_name, "error": ""}
        except Exception as e:
            print(f"[TTS ERROR] Download failed: {e}")
            _tts_download_status = {"status": "failed", "model": model_name, "error": str(e)}
        finally:
            _tts_downloading = False

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

def _get_remote_endpoint():
    remote_cfg = config.get("live_translation", {}).get("remote", {})
    if remote_cfg.get("enabled") and remote_cfg.get("endpoint"):
        return remote_cfg["endpoint"].rstrip("/")
    return None


@app.route("/api/remote-translation/status", methods=["GET"])
def proxy_remote_translation_status():
    """Proxy: fetch Machine B's translation status for display on Machine A's UI."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    endpoint = _get_remote_endpoint()
    if not endpoint:
        return jsonify({"success": False, "error": "No remote endpoint configured"}), 400
    import requests as _req
    try:
        r = _req.get(endpoint + "/api/translation/status", timeout=5)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 502


@app.route("/api/remote-translation/pair/request", methods=["POST"])
def proxy_pair_request():
    """Proxy: send pairing request from Machine A's server to Machine B (avoids CORS)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    endpoint = _get_remote_endpoint()
    if not endpoint:
        return jsonify({"success": False, "error": "No remote endpoint configured"}), 400
    import requests as _req
    try:
        r = _req.post(endpoint + "/api/translate/pair/request", timeout=10)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/remote-translation/pair/confirm", methods=["POST"])
def proxy_pair_confirm():
    """Proxy: send pairing confirmation from Machine A's server to Machine B (avoids CORS)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    endpoint = _get_remote_endpoint()
    if not endpoint:
        return jsonify({"success": False, "error": "No remote endpoint configured"}), 400
    import requests as _req
    try:
        r = _req.post(endpoint + "/api/translate/pair/confirm",
                      json=request.get_json() or {}, timeout=10)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/remote-translation/pair/status", methods=["GET"])
def proxy_pair_status():
    """Proxy: check if Machine A is paired with Machine B (avoids CORS)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    endpoint = _get_remote_endpoint()
    if not endpoint:
        return jsonify({"paired": False}), 200
    import requests as _req
    try:
        r = _req.get(endpoint + "/api/translate/pair/status", timeout=5)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"paired": False, "error": str(e)}), 200


@app.route("/api/remote-translation/unload", methods=["POST"])
def proxy_translate_unload():
    """Proxy: tell Machine B to unload its translation model (avoids CORS)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    endpoint = _get_remote_endpoint()
    if not endpoint:
        return jsonify({"success": False, "error": "No remote endpoint configured"}), 400
    import requests as _req
    try:
        r = _req.post(endpoint + "/api/translate/unload", timeout=10)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 502


@app.route("/api/remote-translation/preload", methods=["POST"])
def proxy_translate_preload():
    """Proxy: tell Machine B to preload its translation model (avoids CORS)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    endpoint = _get_remote_endpoint()
    if not endpoint:
        return jsonify({"success": False, "error": "No remote endpoint configured"}), 400
    import requests as _req
    try:
        r = _req.post(endpoint + "/api/translate/preload", timeout=10)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 502


@app.route("/api/remote-translation/unpair", methods=["POST"])
def proxy_translate_unpair():
    """Proxy: tell Machine B to remove this machine from its trusted list."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    endpoint = _get_remote_endpoint()
    if not endpoint:
        return jsonify({"success": False, "error": "No remote endpoint configured"}), 400
    import requests as _req
    try:
        r = _req.post(endpoint + "/api/translate/pair/unpair-me", timeout=10)
        return jsonify(r.json()), r.status_code
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

        # Clear translation cache
        get_translation_cache().clear()

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
                with open("config.json", "w") as f:
                    json.dump(config, f, indent=2)
                print(
                    f"[OK] File transcription settings updated and saved to config.json"
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
            except:
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

            remote_cfg = config.get("live_translation", {}).get("remote", {})
            if remote_cfg.get("enabled") and remote_cfg.get("endpoint"):
                # Remote path: send each segment to Machine B, no local model load needed
                socketio.emit(
                    "file_progress",
                    {"session_id": session_id, "percent": 65, "status": "Translating via remote server..."},
                )
                translated_segments = []
                total = len(segments)
                for i, seg in enumerate(segments):
                    text = seg.get("text", "").strip()
                    translated_text = _translate_via_remote(text, source_lang, translate_to, remote_cfg["endpoint"]) if text else ""
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

                translated_segments = translate_segments(
                    segments, source_lang, translate_to,
                    translation_model, translation_tokenizer,
                    progress_callback=translation_progress
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
                script_dir = os.path.dirname(os.path.abspath(__file__))
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
            for service_name in ["stt-server", "stt"]:
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
            script_dir = os.path.dirname(os.path.abspath(__file__))
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
        current_path = os.getcwd()

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
                if view_func and view_func.__doc__:
                    doc_lines = view_func.__doc__.strip().split('\n')
                    # First line is description
                    description = doc_lines[0].strip()
                    # Lines starting with "Example:" are examples
                    for line in doc_lines[1:]:
                        line = line.strip()
                        if line.startswith("Example:"):
                            examples.append(line[8:].strip())

                endpoints.append({
                    "path": rule.rule,
                    "methods": sorted(methods),
                    "description": description,
                    "examples": examples
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
        path = request.args.get("path", os.getcwd())
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
        working_dir = os.getcwd()

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
        if path == os.getcwd() or path in [
            os.path.abspath("speech_to_text.py"),
            os.path.abspath("config.json"),
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
        working_dir = os.getcwd()

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
        working_dir = os.getcwd()

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
        working_dir = os.getcwd()

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
        return jsonify({"success": True, "config": mover_config})
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
        result = execute_file_move(lambda: config)

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
        if transcription_state["running"]:
            return jsonify(
                {"success": False, "error": "Transcription is already running"}
            ), 400

        # Don't start if still stopping
        if transcription_state["status"] == "stopping":
            return jsonify(
                {"success": False, "error": "Transcription is still stopping, please wait"}
            ), 400

        # Ensure we have a valid worker process
        # Worker stays alive between Start/Stop cycles, so we usually just reuse it
        if transcription_process is None or not transcription_process.is_alive():
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

        # Tell remote Machine B to preload its translation model
        remote_cfg = config.get("live_translation", {}).get("remote", {})
        if remote_cfg.get("enabled") and remote_cfg.get("endpoint"):
            def _notify_remote_preload():
                try:
                    import requests as _req
                    ep = remote_cfg["endpoint"].rstrip("/")
                    r = _req.post(ep + "/api/translate/preload", timeout=10)
                    print(f"[START] Remote translation preload: {r.json()}")
                except Exception as e:
                    print(f"[START] Remote translation preload failed: {e}")
            import threading
            threading.Thread(target=_notify_remote_preload, daemon=True).start()

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
                    import requests as _req
                    ep = remote_cfg["endpoint"].rstrip("/")
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
            _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.log")
            def log(msg):
                with open(_log_path, "a") as f:
                    f.write(msg + "\n")
                    f.flush()

            log(f"[STOP-CLEANUP] Thread started")

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
        _server_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.log")
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
            except:
                pass

            # If still alive, force kill
            if transcription_process.is_alive():
                print("[FORCE RESET] Force killing transcription process...")
                try:
                    transcription_process.kill()
                    transcription_process.join(timeout=2)
                except:
                    pass

        # Clear the control queue
        while not control_queue.empty():
            try:
                control_queue.get_nowait()
            except:
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
                   WHERE needs_review = 1
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
        dict_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), dict_file)

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
            default_dict = {"hotwords": [], "glossary": {}}
            import json as _json
            with open(dict_file, "w", encoding="utf-8") as f:
                _json.dump(default_dict, f, indent=2, ensure_ascii=False)
            _dictionary_cache = default_dict
            _dictionary_mtime = os.path.getmtime(dict_file)
            print(f"[DICTIONARY] Created default dictionary: {dict_file}")
            return default_dict
    except Exception as e:
        print(f"[DICTIONARY] Error loading dictionary: {e}")

    return {"hotwords": [], "glossary": {}}


def save_custom_dictionary(data):
    """Save custom dictionary to JSON file"""
    global _dictionary_cache, _dictionary_mtime

    dict_file = config.get("custom_dictionary", {}).get("file", "custom_dictionary.json")
    if not os.path.isabs(dict_file):
        dict_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), dict_file)

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


@app.route("/api/dictionary/hotword", methods=["POST"])
def add_hotword():
    """Add a hotword to the dictionary"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = request.get_json()
    word = data.get("word", "").strip() if data else ""
    if not word:
        return jsonify({"success": False, "error": "word is required"}), 400

    dictionary = load_custom_dictionary()
    if word not in dictionary.get("hotwords", []):
        dictionary.setdefault("hotwords", []).append(word)
        save_custom_dictionary(dictionary)

    return jsonify({"success": True, "hotwords": dictionary["hotwords"]})


@app.route("/api/dictionary/hotword", methods=["DELETE"])
def remove_hotword():
    """Remove a hotword from the dictionary"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = request.get_json()
    word = data.get("word", "").strip() if data else ""
    if not word:
        return jsonify({"success": False, "error": "word is required"}), 400

    dictionary = load_custom_dictionary()
    hotwords = dictionary.get("hotwords", [])
    if word in hotwords:
        hotwords.remove(word)
        dictionary["hotwords"] = hotwords
        save_custom_dictionary(dictionary)

    return jsonify({"success": True, "hotwords": dictionary.get("hotwords", [])})


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
            devices = list_audio_devices()
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

    except Exception as e:
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
                except:
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
            except:
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
        except:
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
DOWNLOAD_PROGRESS_FILE = os.path.join(os.getcwd(), "download_progress.json")


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

# Clean up stale downloads on startup
cleanup_stale_downloads()

# Start periodic cleanup thread
def periodic_cleanup():
    """Run cleanup every hour"""
    import time
    while True:
        time.sleep(3600)  # Every hour
        cleanup_stale_downloads()

cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()


class DownloadProgressTracker:
    """Custom progress tracker for HuggingFace downloads"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.total = 0
        self.n = 0
        self.downloaded = 0
        self.iterable = None

    def __call__(self, iterable=None, **kwargs):
        """Make the class callable to work with tqdm_class parameter"""
        self.iterable = iterable
        self.total = kwargs.get("total", 0)
        return self

    def __iter__(self):
        """Make the tracker iterable - wrap the underlying iterable"""
        if self.iterable is not None:
            for item in self.iterable:
                yield item
                self.update(1)
        return self

    def __len__(self):
        """Return length if available"""
        if hasattr(self.iterable, '__len__'):
            return len(self.iterable)
        return 0

    def update(self, n=1):
        """Update progress"""
        import time

        # Check if download was cancelled
        if self.model_name in cancelled_downloads:
            return  # Skip update if cancelled

        self.n += n
        self.downloaded = self.n

        # Update global progress tracking
        with active_downloads_lock:
            # Double-check cancelled status inside lock
            if self.model_name in cancelled_downloads:
                return

            if self.model_name in active_downloads:
                active_downloads[self.model_name]["downloaded"] = self.downloaded
                active_downloads[self.model_name]["total"] = self.total
                active_downloads[self.model_name]["last_update"] = time.time()
                if self.total > 0:
                    active_downloads[self.model_name]["percentage"] = int(
                        (self.downloaded / self.total) * 100
                    )

        # Save to file periodically (every 10 updates to reduce I/O)
        if self.n % 10 == 0:
            save_download_progress()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def close(self):
        """Close the progress tracker"""
        pass

    @staticmethod
    def get_lock():
        """Return a lock for thread-safe progress updates (required by huggingface_hub)"""
        import threading
        return threading.Lock()

    @staticmethod
    def set_lock(lock):
        """Set lock for thread-safe progress updates (required by huggingface_hub)"""
        pass


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
            import whisper

            # Create custom download directory in ./models
            models_dir = os.path.join(os.getcwd(), "models")
            os.makedirs(models_dir, exist_ok=True)

            whisper_dir = os.path.join(models_dir, f"whisper-{model_name}")
            os.makedirs(whisper_dir, exist_ok=True)

            # Set environment variable to use custom download directory
            os.environ["WHISPER_CACHE"] = whisper_dir

            # Register download in active_downloads
            import time

            model_key = f"whisper-{model_name}"

            with active_downloads_lock:
                # Clear from cancelled set so this download can proceed
                cancelled_downloads.discard(model_key)

                model_size = WHISPER_MODEL_SIZES.get(model_name)
                active_downloads[model_key] = {
                    "downloaded": 0,
                    "total": model_size,
                    "percentage": 0 if model_size else None,
                    "start_time": time.time(),
                    "last_update": time.time(),
                    "status": "downloading",
                }

            save_download_progress()

            def update_whisper_progress(model_name, whisper_dir):
                """Periodically update file size for Whisper downloads"""
                import time
                import os

                # Expected download file path
                download_file = os.path.join(whisper_dir, f"{model_name}.pt")
                model_key = f"whisper-{model_name}"

                while model_key in active_downloads:
                    # Check if cancelled
                    if model_key in cancelled_downloads:
                        return  # Stop the progress update thread

                    time.sleep(2)  # Poll every 2 seconds

                    # Check if file exists and get its size
                    if os.path.exists(download_file):
                        file_size = os.path.getsize(download_file)

                        with active_downloads_lock:
                            # Check cancelled again inside lock
                            if model_key in cancelled_downloads:
                                return

                            if model_key in active_downloads:
                                active_downloads[model_key]["downloaded"] = file_size
                                active_downloads[model_key]["last_update"] = time.time()

                                # Calculate percentage if we have total size
                                total = active_downloads[model_key].get("total")
                                if total and total > 0:
                                    percentage = int((file_size / total) * 100)
                                    # Cap at 99% until completion (download may include verification)
                                    active_downloads[model_key]["percentage"] = min(percentage, 99)

                        save_download_progress()

                    time.sleep(0.5)  # Brief pause before next iteration

            try:
                # Start background thread to update timestamps
                import threading

                update_thread = threading.Thread(
                    target=update_whisper_progress,
                    args=(model_name, whisper_dir),
                    daemon=True,
                )
                update_thread.start()

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
                        print(f"[OK] Existing file checksum matches")
                        model_path = download_target
                    else:
                        print(f"[WARN] Existing file checksum mismatch, re-downloading")
                        os.remove(download_target)
                        model_path = None
                else:
                    model_path = None

                if model_path is None:
                    # Download with streaming SHA256 computation
                    print(f"[INFO] Downloading Whisper model to {download_target}")
                    sha256_hash = hashlib.sha256()

                    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
                        total_size = int(source.info().get("Content-Length", 0))
                        with tqdm(total=total_size, ncols=80, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
                            while True:
                                buffer = source.read(8192)
                                if not buffer:
                                    break
                                output.write(buffer)
                                sha256_hash.update(buffer)  # Compute SHA256 during download
                                pbar.update(len(buffer))

                    # Verify checksum computed during download
                    computed_sha256 = sha256_hash.hexdigest()
                    if computed_sha256 != expected_sha256:
                        os.remove(download_target)
                        raise RuntimeError(
                            f"Model download failed: SHA256 mismatch. Expected {expected_sha256}, got {computed_sha256}"
                        )

                    model_path = download_target
                    print(f"[OK] Download complete, checksum verified")

                message = f"Whisper {model_name} model downloaded to {model_path}"

                print(f"[OK] {message}")

                # Mark as completed instead of deleting
                with active_downloads_lock:
                    if f"whisper-{model_name}" in active_downloads:
                        active_downloads[f"whisper-{model_name}"]["status"] = "completed"
                        active_downloads[f"whisper-{model_name}"]["percentage"] = 100
                        active_downloads[f"whisper-{model_name}"]["last_update"] = time.time()
                        active_downloads[f"whisper-{model_name}"]["completion_time"] = time.time()
                save_download_progress()
            except Exception as e:
                # Mark as failed on error
                print(f"[ERROR] Whisper model download failed: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                with active_downloads_lock:
                    if f"whisper-{model_name}" in active_downloads:
                        active_downloads[f"whisper-{model_name}"]["status"] = "failed"
                        active_downloads[f"whisper-{model_name}"]["error"] = str(e)
                        active_downloads[f"whisper-{model_name}"]["last_update"] = time.time()
                save_download_progress()
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
            from huggingface_hub import snapshot_download

            # If no local_dir specified, download to ./models directory
            if not local_dir:
                models_dir = os.path.join(os.getcwd(), "models")
                os.makedirs(models_dir, exist_ok=True)

                # Use model name as directory (replace / with --)
                model_dir_name = model_id.replace("/", "--")
                local_dir = os.path.join(models_dir, model_dir_name)

            # Register download in active_downloads
            import time

            with active_downloads_lock:
                # Clear from cancelled set so this download can proceed
                cancelled_downloads.discard(model_id)

                active_downloads[model_id] = {
                    "downloaded": 0,
                    "total": None,
                    "percentage": None,
                    "start_time": time.time(),
                    "last_update": time.time(),
                    "status": "downloading",
                }

            save_download_progress()

            try:
                # Use wget for reliable download (huggingface_hub hangs on large files)
                from huggingface_hub import list_repo_files, hf_hub_url
                import subprocess

                os.makedirs(local_dir, exist_ok=True)

                # Get list of files in the repo
                files = list_repo_files(repo_id=model_id)
                print(f"[DOWNLOAD] Found {len(files)} files to download")

                # Download each file using wget
                for idx, filename in enumerate(files):
                    # Check if cancelled before each file
                    if model_id in cancelled_downloads:
                        print(f"[CANCELLED] Download cancelled for {model_id}")
                        return jsonify({"success": False, "message": "Download cancelled"})

                    dest_path = os.path.join(local_dir, filename)
                    os.makedirs(os.path.dirname(dest_path) if os.path.dirname(dest_path) else local_dir, exist_ok=True)

                    # Skip if already downloaded and has content
                    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                        print(f"[DOWNLOAD] Already exists: {filename}")
                        continue

                    print(f"[DOWNLOAD] Downloading file {idx+1}/{len(files)}: {filename}")

                    # Update progress
                    with active_downloads_lock:
                        # Check cancelled inside lock
                        if model_id in cancelled_downloads:
                            print(f"[CANCELLED] Download cancelled for {model_id}")
                            return jsonify({"success": False, "message": "Download cancelled"})

                        if model_id in active_downloads:
                            active_downloads[model_id]["percentage"] = int((idx / len(files)) * 100)
                            active_downloads[model_id]["last_update"] = time.time()
                    save_download_progress()

                    # Get download URL from HuggingFace
                    url = hf_hub_url(repo_id=model_id, filename=filename)

                    # Use wget or curl for reliable download with resume capability
                    # Retry loop: re-run with resume (-c / -C -) on transient failures
                    import shutil as _shutil
                    max_attempts = 5
                    for attempt in range(1, max_attempts + 1):
                        if _shutil.which('wget'):
                            dl_cmd = ['wget', '-c', '-t', '3', '-T', '120', '--retry-connrefused',
                                      '--waitretry', '5', '-O', dest_path, url]
                        elif _shutil.which('curl'):
                            dl_cmd = ['curl', '-L', '-C', '-', '--retry', '3', '--retry-delay', '5',
                                      '--retry-connrefused', '--connect-timeout', '30',
                                      '--max-time', '600', '-o', dest_path, url]
                        else:
                            raise Exception("Neither wget nor curl found — install one to download models")

                        result = subprocess.run(dl_cmd, capture_output=True, text=True)

                        if result.returncode == 0:
                            break  # Success

                        print(f"[WARNING] Download attempt {attempt}/{max_attempts} failed for {filename} "
                              f"(exit code {result.returncode})")

                        if attempt < max_attempts:
                            if os.path.exists(dest_path):
                                partial_size = os.path.getsize(dest_path)
                                print(f"[INFO] Partial file exists ({partial_size / (1024*1024):.1f} MB), will resume")
                            import time as _time
                            _time.sleep(5 * attempt)
                        else:
                            print(f"[ERROR] All {max_attempts} attempts failed for {filename}: {result.stderr}")
                            raise Exception(f"Failed to download {filename} after {max_attempts} attempts (last exit code: {result.returncode})")

                    print(f"[OK] Downloaded: {filename}")

                path = local_dir
                message = f"Model {model_id} downloaded to: {path}"

                print(f"[OK] {message}")

                # Mark as completed instead of deleting
                with active_downloads_lock:
                    if model_id in active_downloads:
                        active_downloads[model_id]["status"] = "completed"
                        active_downloads[model_id]["percentage"] = 100
                        active_downloads[model_id]["last_update"] = time.time()
                        active_downloads[model_id]["completion_time"] = time.time()
                save_download_progress()
            except Exception as e:
                # Mark as failed on error
                with active_downloads_lock:
                    if model_id in active_downloads:
                        active_downloads[model_id]["status"] = "failed"
                        active_downloads[model_id]["error"] = str(e)
                        active_downloads[model_id]["last_update"] = time.time()
                save_download_progress()
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
            # Add to cancelled set to prevent download thread from re-adding
            cancelled_downloads.add(model_id)

            if model_id in active_downloads:
                del active_downloads[model_id]

        # Save outside the lock to avoid deadlock (save_download_progress acquires the same lock)
        save_download_progress()

        # Also clean up the persisted file directly
        try:
            if os.path.exists(DOWNLOAD_PROGRESS_FILE):
                with open(DOWNLOAD_PROGRESS_FILE, 'r') as f:
                    persisted = json.load(f)
                if model_id in persisted:
                    del persisted[model_id]
                    with open(DOWNLOAD_PROGRESS_FILE, 'w') as f:
                        json.dump(persisted, f)
        except Exception as e:
            print(f"[WARNING] Error cleaning persisted download entry: {e}")

        # Clean up partial download directories/files so model shows as not downloaded
        import shutil
        models_dir = os.path.join(os.getcwd(), "models")

        # model_id already includes prefix (e.g., "whisper-small.en" or "faster-whisper-base.en")
        # For HuggingFace models like "facebook/nllb-200-distilled-600M", slashes become double dashes
        dir_name = model_id.replace("/", "--")
        model_path = os.path.join(models_dir, dir_name)
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
        from huggingface_hub import HfApi, Repository
        import os
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
        from pathlib import Path

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
WHISPER_MODELS_FILE = "whisper_models.json"  # Persistent storage file


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
FASTER_WHISPER_MODELS_FILE = "faster_whisper_models.json"


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
        models_dir = os.path.join(os.getcwd(), "models")
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
        models_dir = os.path.join(os.getcwd(), "models")
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

        from huggingface_hub import snapshot_download

        models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(models_dir, exist_ok=True)
        local_dir = os.path.join(models_dir, f"faster-whisper-{model_name}")

        # Register download in active_downloads
        download_key = f"faster-whisper-{model_name}"
        with active_downloads_lock:
            active_downloads[download_key] = {
                "downloaded": 0,
                "total": None,
                "percentage": None,
                "start_time": time.time(),
                "last_update": time.time(),
                "status": "downloading",
            }
        save_download_progress()

        try:
            # Download model (progress shown in server console via tqdm)
            path = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            message = f"faster-whisper {model_name} downloaded to: {path}"
            print(f"[OK] {message}")

            # Mark as completed
            with active_downloads_lock:
                if download_key in active_downloads:
                    active_downloads[download_key]["status"] = "completed"
                    active_downloads[download_key]["percentage"] = 100
                    active_downloads[download_key]["last_update"] = time.time()
                    active_downloads[download_key]["completion_time"] = time.time()
            save_download_progress()

        except Exception as e:
            # Mark as failed
            with active_downloads_lock:
                if download_key in active_downloads:
                    active_downloads[download_key]["status"] = "failed"
                    active_downloads[download_key]["error"] = str(e)
                    active_downloads[download_key]["last_update"] = time.time()
            save_download_progress()
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

        models_dir = os.path.join(os.getcwd(), "models")
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
        models_dir = os.path.join(os.getcwd(), "models")

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
        models_dir = os.path.join(os.getcwd(), "models")
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                if os.path.isdir(os.path.join(models_dir, item)):
                    # Skip Whisper models (they have their own section)
                    if item.startswith("whisper-"):
                        continue

                    # Skip NLLB translation models (shown in Translation tab)
                    if item.startswith("facebook--nllb-"):
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
            models_dir = os.path.join(os.getcwd(), "models")
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
            models_dir = os.path.join(os.getcwd(), "models")
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
        models_dir = os.path.join(os.getcwd(), "models")

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


@app.route("/api/models/download-nllb", methods=["POST"])
def download_nllb():
    """Download NLLB translation model to ./models/ directory"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global nllb_download_progress

    try:
        import threading
        from huggingface_hub import snapshot_download

        def download_nllb_model():
            global nllb_download_progress
            try:
                models_dir = os.path.join(os.getcwd(), "models")
                os.makedirs(models_dir, exist_ok=True)
                nllb_dir_name = "facebook--nllb-200-distilled-600M"
                nllb_path = os.path.join(models_dir, nllb_dir_name)

                model_id = "facebook/nllb-200-distilled-600M"

                nllb_download_progress = {"status": "downloading", "progress": 20, "message": "Downloading NLLB model (~2.5GB)..."}

                # Download directly using snapshot_download (same as faster-whisper)
                snapshot_download(
                    repo_id=model_id,
                    local_dir=nllb_path,
                    local_dir_use_symlinks=False
                )

                nllb_download_progress = {"status": "complete", "progress": 100, "message": "Download complete!"}

            except Exception as e:
                nllb_download_progress = {"status": "error", "progress": 0, "message": str(e)}

        # Check if already downloading
        if nllb_download_progress.get("status") in ["downloading", "starting"]:
            return jsonify({"success": False, "error": "Download already in progress"})

        # Start download in background thread
        nllb_download_progress = {"status": "starting", "progress": 0, "message": "Starting download..."}
        thread = threading.Thread(target=download_nllb_model)
        thread.daemon = True
        thread.start()

        return jsonify({"success": True, "message": "Download started"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


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
    models_dir = os.path.join(os.getcwd(), "models")
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
        from silero_vad import load_silero_vad
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


@app.route("/api/models/translation/download", methods=["POST"])
def download_translation_model():
    """Download any translation model to ./models/ directory"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global nllb_download_progress

    try:
        import threading
        from huggingface_hub import snapshot_download

        data = request.get_json()
        model_id = data.get("model_id", "facebook/nllb-200-distilled-600M")

        def download_model():
            global nllb_download_progress
            import time
            import logging

            # Set up detailed logging
            log_file = os.path.join(os.getcwd(), "logs", "translation_download.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Configure file handler for this download
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

            dl_logger = logging.getLogger('translation_download')
            dl_logger.setLevel(logging.DEBUG)
            dl_logger.addHandler(file_handler)

            start_time = time.time()
            dl_logger.info(f"=" * 60)
            dl_logger.info(f"Starting download of {model_id}")
            dl_logger.info(f"=" * 60)

            try:
                models_dir = os.path.join(os.getcwd(), "models")
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
                                    except:
                                        pass

                            size_mb = total_size / (1024 * 1024)
                            speed = (total_size - last_size) / (1024 * 1024)  # MB in last second

                            # Log progress
                            if incomplete_files:
                                for fname, fsize in incomplete_files:
                                    dl_logger.debug(f"Incomplete file: {fname[:30]}... = {fsize / (1024*1024):.1f} MB")

                            dl_logger.info(f"Progress: {size_mb:.1f} MB downloaded, speed: {speed:.2f} MB/s")

                            # Update progress for UI (estimate based on expected ~2.5GB for NLLB)
                            expected_size = 2500  # MB
                            progress = min(85, int(10 + (size_mb / expected_size) * 75))
                            nllb_download_progress = {
                                "status": "downloading",
                                "progress": progress,
                                "message": f"Downloading: {size_mb:.0f} MB ({speed:.1f} MB/s)"
                            }

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
                    import subprocess

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

                        # Use wget or curl for reliable download with resume capability
                        # Retry loop: re-run with resume (-c / -C -) on transient failures
                        import shutil as _shutil
                        max_attempts = 5
                        for attempt in range(1, max_attempts + 1):
                            if _shutil.which('wget'):
                                dl_cmd = ['wget', '-c', '-t', '3', '-T', '120', '--retry-connrefused',
                                          '--waitretry', '5', '-O', dest_path, url]
                            elif _shutil.which('curl'):
                                dl_cmd = ['curl', '-L', '-C', '-', '--retry', '3', '--retry-delay', '5',
                                          '--retry-connrefused', '--connect-timeout', '30',
                                          '--max-time', '600', '-o', dest_path, url]
                            else:
                                raise Exception("Neither wget nor curl found — install one to download models")

                            result = subprocess.run(dl_cmd, capture_output=True, text=True)

                            if result.returncode == 0:
                                break  # Success

                            dl_logger.warning(f"Download attempt {attempt}/{max_attempts} failed for {filename} "
                                            f"(exit code {result.returncode}): {result.stderr[:200] if result.stderr else 'no stderr'}")

                            if attempt < max_attempts:
                                # Check if partial file exists (resume will pick up from there)
                                if os.path.exists(dest_path):
                                    partial_size = os.path.getsize(dest_path)
                                    dl_logger.info(f"Partial file exists ({partial_size / (1024*1024):.1f} MB), will resume on next attempt")
                                import time as _time
                                _time.sleep(5 * attempt)  # Increasing backoff: 5s, 10s, 15s, 20s
                            else:
                                dl_logger.error(f"All {max_attempts} download attempts failed for {filename}")
                                if result.stdout:
                                    dl_logger.error(f"Downloader stdout: {result.stdout}")
                                raise Exception(f"Failed to download {filename} after {max_attempts} attempts (last exit code: {result.returncode})")

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
                                        dl_logger.info(f"Copying incomplete file to pytorch_model.bin")
                                        import shutil
                                        shutil.copy2(incomplete_path, model_bin)
                                        dl_logger.info(f"Copy completed")
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
                dl_logger.info(f"=" * 60)

                nllb_download_progress = {"status": "complete", "progress": 100, "message": "Download complete!"}

            except Exception as e:
                import traceback
                dl_logger.error(f"Download failed: {type(e).__name__}: {e}")
                dl_logger.error(traceback.format_exc())
                nllb_download_progress = {"status": "error", "progress": 0, "message": str(e)}
            finally:
                dl_logger.removeHandler(file_handler)
                file_handler.close()

        if nllb_download_progress.get("status") in ["downloading", "starting"]:
            return jsonify({"success": False, "error": "Download already in progress"})

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

        models_dir = os.path.join(os.getcwd(), "models")
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

        if not files or len(files) == 0:
            return jsonify({"success": False, "error": "No files selected"}), 400

        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.getcwd(), "models")
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
        {"id": e[0], "timestamp": e[1], "text": e[2], "start": e[3], "end": e[4], "completed": True}
        for e in entries
    ]
    emit("transcription_update", {
        "segments": segments,
        "in_progress_segment": None,
        "entries": [(e[1], e[2]) for e in entries],
        "in_progress": ""
    })


@socketio.on("request_all_translation_entries")
def handle_request_all_translation_entries():
    """Send all historical translation entries to the requesting client only"""
    trans_config = config.get("live_translation", {})
    if not trans_config.get("enabled", False):
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
            "completed": True
        })

    emit("translation_update", {
        "segments": translated_segments,
        "in_progress": None,
        "target_language": target_lang,
        "target_language_name": TRANSLATION_LANGUAGES.get(target_lang, target_lang),
        "source_language": source_lang,
        "enabled": True
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

    current_db_name = transcription_state.get("db_name")
    if not current_db_name or not os.path.exists(current_db_name):
        return

    try:
        with sqlite3.connect(current_db_name) as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in segment_ids)
            cursor.execute(
                f"UPDATE transcriptions SET needs_review = 0 WHERE id IN ({placeholders})",
                segment_ids,
            )
            conn.commit()

        with _cache_lock:
            _db_cache["last_entries"] = []
            _db_cache["last_fetch_time"] = 0

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
    emit("tts_audio_info", {"status": "joined"})


@socketio.on("leave_tts_audio")
def handle_leave_tts_audio():
    """Client no longer wants TTS audio"""
    from flask_socketio import leave_room
    leave_room("tts_audio")


# =============================================================================
# Staging Buffer for Output Delay
# =============================================================================


class StagingBuffer:
    """Buffer that holds finalized segments before publishing to DB.

    When output delay is enabled, segments are held here for a configurable
    duration before being committed to the database, giving the user time
    to review and correct them.
    """

    def __init__(self):
        self._buffer = []  # List of {segment_data, finalized_at, id}
        self._lock = threading.Lock()
        self._next_id = 1

    def add(self, segment_data):
        """Add a segment to the staging buffer. Returns the staging ID."""
        with self._lock:
            staging_id = self._next_id
            self._next_id += 1
            self._buffer.append({
                "segment": segment_data,
                "finalized_at": time.time(),
                "staging_id": staging_id,
            })
            return staging_id

    def get_ready(self, delay_seconds):
        """Get segments that have been in the buffer longer than delay_seconds."""
        now = time.time()
        with self._lock:
            ready = [item for item in self._buffer if now - item["finalized_at"] >= delay_seconds]
            self._buffer = [item for item in self._buffer if now - item["finalized_at"] < delay_seconds]
            return ready

    def get_staged(self):
        """Get all currently staged segments with their remaining time."""
        now = time.time()
        with self._lock:
            return [
                {
                    "staging_id": item["staging_id"],
                    "segment": item["segment"],
                    "elapsed": now - item["finalized_at"],
                    "finalized_at": item["finalized_at"],
                }
                for item in self._buffer
            ]

    def approve(self, staging_id):
        """Immediately approve a staged segment (remove from buffer and return it)."""
        with self._lock:
            for i, item in enumerate(self._buffer):
                if item["staging_id"] == staging_id:
                    return self._buffer.pop(i)
            return None

    def approve_all(self):
        """Approve all staged segments."""
        with self._lock:
            items = list(self._buffer)
            self._buffer.clear()
            return items

    def edit(self, staging_id, new_text):
        """Edit a staged segment's text before it's published."""
        with self._lock:
            for item in self._buffer:
                if item["staging_id"] == staging_id:
                    item["segment"]["text"] = new_text
                    return True
            return False

    def discard(self, staging_id):
        """Discard a staged segment (don't publish it)."""
        with self._lock:
            self._buffer = [item for item in self._buffer if item["staging_id"] != staging_id]

    def is_empty(self):
        """Check if buffer has any staged segments."""
        with self._lock:
            return len(self._buffer) == 0


# Global staging buffer instance
_staging_buffer = StagingBuffer()


@socketio.on("toggle_delay")
def handle_toggle_delay(data):
    """Toggle the output delay buffer on/off"""
    if not data:
        return
    enabled = data.get("enabled", False)
    config.setdefault("corrections", {}).setdefault("output_delay", {})["enabled"] = enabled
    save_config(config)

    # If disabling, flush all staged segments
    if not enabled:
        items = _staging_buffer.approve_all()
        if items:
            _flush_staged_to_db(items)

    socketio.emit("delay_status", {"enabled": enabled})


@socketio.on("set_delay_seconds")
def handle_set_delay_seconds(data):
    """Update the output delay duration"""
    if not data:
        return
    seconds = max(2, min(30, int(data.get("delay_seconds", 7))))
    config.setdefault("corrections", {}).setdefault("output_delay", {})["delay_seconds"] = seconds
    save_config(config)


@socketio.on("approve_staged")
def handle_approve_staged(data):
    """Approve one or all staged segments for immediate publishing"""
    if not data:
        return

    if data.get("all", False):
        items = _staging_buffer.approve_all()
        if items:
            _flush_staged_to_db(items)
    else:
        staging_id = data.get("staging_id")
        if staging_id is not None:
            item = _staging_buffer.approve(staging_id)
            if item:
                _flush_staged_to_db([item])


@socketio.on("edit_staged")
def handle_edit_staged(data):
    """Edit a staged segment's text before publishing"""
    if not data:
        return
    staging_id = data.get("staging_id")
    new_text = data.get("new_text", "").strip()
    if staging_id is not None and new_text:
        _staging_buffer.edit(staging_id, new_text)


@socketio.on("discard_staged")
def handle_discard_staged(data):
    """Discard a staged segment"""
    if not data:
        return
    staging_id = data.get("staging_id")
    if staging_id is not None:
        _staging_buffer.discard(staging_id)


def _flush_staged_to_db(items):
    """Write approved/expired staged segments to the database"""
    current_db_name = transcription_state.get("db_name")
    if not current_db_name or not os.path.exists(current_db_name):
        return

    try:
        with sqlite3.connect(current_db_name) as conn:
            cursor = conn.cursor()
            for item in items:
                seg = item["segment"]
                timestamp = seg.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
                text = seg.get("text", "").strip()
                if not text:
                    continue
                confidence = seg.get("confidence")
                confidence_threshold = config.get("corrections", {}).get("confidence_threshold", 0.7)
                needs_review = 1 if (confidence is not None and confidence < confidence_threshold) else 0
                cursor.execute(
                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, needs_review) VALUES (?, ?, ?, ?, ?, ?)",
                    (timestamp, text, seg.get("start", 0), seg.get("end", 0), confidence, needs_review),
                )
            conn.commit()

        # Invalidate cache
        with _cache_lock:
            _db_cache["last_entries"] = []
            _db_cache["last_fetch_time"] = 0

    except Exception as e:
        print(f"[STAGING FLUSH ERROR] {e}")


def get_new_entries(limit_override=None):
    """Get recent transcriptions with caching and efficient querying

    Args:
        limit_override: Optional limit to override database.max_entries_to_send config
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
                           confidence, needs_review
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
                    SELECT id, timestamp, text, start_time, end_time, confidence, needs_review FROM (
                        SELECT id, timestamp, text, COALESCE(start_time, 0) as start_time, COALESCE(end_time, 0) as end_time,
                               confidence, needs_review
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
        # entries format: (id, timestamp, text, start_time, end_time, confidence, needs_review)
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
            segments.append(seg)

        # Build in-progress segment if there's text
        in_progress_segment = None
        if in_progress and in_progress.strip():
            in_progress_segment = {
                "text": in_progress,
                "start": in_progress_start,
                "end": in_progress_end,
                "completed": False,
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
                        },
                    )
                except Exception as emit_error:
                    print(f"[AUDIO-DEBUG] {time.strftime('%H:%M:%S')} - EMIT FAILED: {emit_error}", flush=True)

        try:
            socketio.sleep(update_interval)  # Emit updates based on config
        except Exception as sleep_error:
            print(f"[AUDIO-DEBUG] {time.strftime('%H:%M:%S')} - SLEEP FAILED: {sleep_error}", flush=True)


def _translate_via_remote(text, source_lang, target_lang, endpoint,
                          return_extras=False, num_alternatives=0):
    """Send text to a remote machine's /api/translate endpoint."""
    import requests as _requests
    try:
        resp = _requests.post(
            endpoint.rstrip("/") + "/api/translate",
            json={
                "text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "return_extras": return_extras,
                "num_alternatives": num_alternatives,
            },
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


def translate_live_text(text, source_lang, target_lang, return_extras=False, num_alternatives=0):
    """Translate text for live display using the singleton model"""
    # Route to remote translation server if configured
    # Skip remote offload if this machine is itself serving remote clients (prevents chaining)
    remote_cfg = config.get("live_translation", {}).get("remote", {})
    if remote_cfg.get("enabled") and remote_cfg.get("endpoint") and not _trusted_translation_clients:
        return _translate_via_remote(text, source_lang, target_lang, remote_cfg["endpoint"],
                                     return_extras=return_extras, num_alternatives=num_alternatives)

    if not text or not text.strip():
        if return_extras:
            return {"text": "", "confidence": None, "alternatives": []}
        return ""

    try:
        trans_use_gpu = config.get("live_translation", {}).get("use_gpu", True)
        model, tokenizer = get_live_translation_model(trans_use_gpu)
        if model is None:
            if return_extras:
                return {"text": text, "confidence": None, "alternatives": []}
            return text

        result = translate_text(
            text, source_lang, target_lang, model, tokenizer,
            return_confidence=return_extras,
            num_alternatives=num_alternatives if return_extras else 0,
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
                    "enabled": True
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

            # Check if corrections features are enabled for translation confidence
            corrections_cfg = config.get("corrections", {})
            want_confidence = corrections_cfg.get("enabled", True) and corrections_cfg.get("confidence_highlighting", True)
            n_alternatives = corrections_cfg.get("n_best_alternatives", {}).get("translation_count", 0) if corrections_cfg.get("enabled", True) else 0

            translations_this_cycle = 0
            for entry in entries:
                seg_id = entry[0]
                original_text = entry[2]

                # Check cache first
                cached = cache.get(seg_id, original_text, target_lang)
                if cached:
                    translated_text = cached
                    # Get cached extras (confidence, alternatives)
                    extras = cache.get_extras(seg_id) if want_confidence else None
                else:
                    # Translate with confidence/alternatives if corrections enabled
                    if want_confidence or n_alternatives > 0:
                        result = translate_live_text(
                            original_text, source_lang, target_lang,
                            return_extras=True, num_alternatives=n_alternatives,
                        )
                        translated_text = result["text"]
                        extras = {"confidence": result.get("confidence"), "alternatives": result.get("alternatives", [])}
                        cache.set_with_extras(seg_id, original_text, translated_text, target_lang,
                                              confidence=extras["confidence"], alternatives=extras["alternatives"])
                    else:
                        translated_text = translate_live_text(original_text, source_lang, target_lang)
                        extras = None
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
                    except Exception:
                        pass

                    # Yield control after each translation so socketio can process events
                    translations_this_cycle += 1
                    if translations_this_cycle % 3 == 0:
                        socketio.sleep(0)

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

            # Translate in-progress text if enabled
            in_progress_translation = None
            if trans_config.get("translate_in_progress", False):
                in_progress = transcription_state.get("live_text", "")
                if in_progress and in_progress.strip():
                    translated_in_progress = translate_live_text(in_progress, source_lang, target_lang)
                    if not is_whisper_hallucination(translated_in_progress):
                        in_progress_translation = {
                            "original_text": in_progress,
                            "translated_text": translated_in_progress,
                            "start": transcription_state.get("live_start", 0),
                            "end": transcription_state.get("live_end", 0),
                            "completed": False
                        }

            # Emit translation update
            socketio.emit("translation_update", {
                "segments": translated_segments,
                "in_progress": in_progress_translation,
                "target_language": target_lang,
                "target_language_name": TRANSLATION_LANGUAGES.get(target_lang, target_lang),
                "source_language": source_lang,
                "enabled": True
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
        except Exception:
            pass  # Queue.Empty on timeout or other errors


_tts_last_spoken_id = 0

def emit_tts_audio():
    """Background task that synthesizes speech from translated text and emits audio"""
    global _tts_last_spoken_id
    import base64

    while True:
        tts_config = config.get("live_translation", {}).get("tts", {})
        trans_config = config.get("live_translation", {})

        if not tts_config.get("enabled", False) or not trans_config.get("enabled", False):
            socketio.sleep(1)
            continue

        if not transcription_state.get("running", False):
            _tts_last_spoken_id = 0
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

            for segment in new_segments:
                # Check if TTS is still enabled (could be toggled mid-loop)
                if not config.get("live_translation", {}).get("tts", {}).get("enabled", False):
                    break

                audio_bytes, sample_rate = synthesize_tts(segment["translated_text"], language=target_lang)
                if audio_bytes is None:
                    continue

                backend = _get_tts_backend()
                audio_format = "mp3" if backend == "edge" else "wav"
                audio_b64 = base64.b64encode(audio_bytes).decode('ascii')
                socketio.emit("tts_audio", {
                    "segment_id": segment["id"],
                    "audio": audio_b64,
                    "format": audio_format,
                    "sample_rate": sample_rate,
                    "text": segment["translated_text"],
                }, room="tts_audio")

                _tts_last_spoken_id = segment["id"]
                socketio.sleep(0.1)  # Small delay between segments

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
# User can override/extend this list via config.json hallucination_filter.phrases
DEFAULT_WHISPER_HALLUCINATIONS = [
    "Субтитры сделал DimaTorzok",
    "Субтитры делал DimaTorzok",
    "Субтитры подготовил DimaTorzok",
    "Продолжение следует...",
    "Thank you for watching",
    "Thanks for watching",
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


def is_whisper_hallucination(text):
    """Check if text is a known Whisper hallucination (exact phrase match, case/punctuation insensitive)."""
    if not text:
        return False

    phrases = get_hallucination_phrases()
    if not phrases:
        return False  # Filter disabled or empty

    text_normalized = normalize_for_hallucination_check(text)
    for hallucination in phrases:
        if normalize_for_hallucination_check(hallucination) == text_normalized:
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
                print(f"[OVERLAP] Entire text was overlap, skipping", flush=True)
                return ""

    return new_text


def thread1_function(ts, cq, cfq, cal_state, cal_data, cal_step1, asq):
    """Main transcription process with start/stop support"""
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
                except Exception:
                    pass
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
                                            sp.run(["pkill", "-9", "-f", f"ffmpeg.*{source.device_name}"],
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
                            except:
                                pass
                            audio_model = None
                        if processor is not None:
                            try:
                                del processor
                            except:
                                pass
                            processor = None
                        if vad_model is not None:
                            try:
                                del vad_model
                            except:
                                pass
                            vad_model = None
                        model_type = None
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

                    # Try multiple audio devices in order of preference
                    audio_devices_to_try = [
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
                            ts_base_dir = backup_cfg.get("base_directory", "").strip() or "_AUTOMATIC_BACKUP"
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
                                except:
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
                        except Exception as check_error:
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

                    # Remove old PyAudio initialization code
                    if False:  # Disabled - PyAudio code removed
                        try:
                            if "linux" in platform:
                                mic_name = args.default_microphone
                                if not mic_name or mic_name == "list":
                                    print("Available microphone devices are: ")
                                    for index, name in enumerate(
                                        sr.Microphone.list_microphone_names()
                                    ):
                                        print(f'Microphone with name "{name}" found')
                                    return
                                else:
                                    # Try to parse as device index first
                                    device_index = None
                                    try:
                                        device_index = int(mic_name)
                                    except ValueError:
                                        # If not a number, try name matching
                                        for index, name in enumerate(
                                            sr.Microphone.list_microphone_names()
                                        ):
                                            if mic_name in name:
                                                device_index = index
                                                break

                                    if device_index is None:
                                        print(
                                            f"[ERROR] Microphone '{mic_name}' not found. Available devices:"
                                        )
                                        for index, name in enumerate(
                                            sr.Microphone.list_microphone_names()
                                        ):
                                            print(f"  {index}: {name}")
                                        return

                                    # Try different sample rates until one works
                                    last_error = None
                                    for sample_rate in sample_rates_to_try:
                                        try:
                                            source = sr.Microphone(
                                                sample_rate=sample_rate,
                                                device_index=device_index,
                                            )
                                            print(
                                                f"[OK] Audio device initialized at {sample_rate} Hz"
                                            )
                                            break
                                        except OSError as e:
                                            last_error = e
                                            continue

                                    if source is None:
                                        print(
                                            f"[ERROR] Could not initialize audio device with any supported sample rate"
                                        )
                                        print(f"Last error: {last_error}")
                                        return
                            else:
                                # Windows - try sample rates
                                last_error = None
                                for sample_rate in sample_rates_to_try:
                                    try:
                                        source = sr.Microphone(sample_rate=sample_rate)
                                        print(
                                            f"[OK] Audio device initialized at {sample_rate} Hz"
                                        )
                                        break
                                    except OSError as e:
                                        last_error = e
                                        continue

                                if source is None:
                                    print(
                                        f"[ERROR] Could not initialize audio device with any supported sample rate"
                                    )
                                    print(f"Last error: {last_error}")
                                    return
                        except Exception as e:
                            print(f"[ERROR] Failed to initialize microphone: {e}")
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
                            except:
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
                            except:
                                pass

                        # Clear orphaned GPU memory from failed model load
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                print("[CLEANUP] GPU cache cleared after failed model load")
                        except:
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
                            except:
                                pass
                        # Clear references BEFORE cleanup to allow garbage collection
                        audio_model = None
                        processor = None
                        model_type = None
                        vad_model = None
                        try:
                            ModelFactory.cleanup_models()
                        except:
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
                            ).strip() or "_AUTOMATIC_BACKUP"
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
                            base_dir = backup_config.get("base_directory", "").strip() or "_AUTOMATIC_BACKUP"

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
                                base_dir = "_AUTOMATIC_BACKUP"
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

                    # Use in-memory temp file instead of disk I/O for better performance
                    temp_file = NamedTemporaryFile(delete=False).name
                    transcription = [""]
                    current_phrase_db_id = (
                        None  # Track current phrase's database row ID
                    )

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
                            except:
                                pass
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                        # Clear references BEFORE cleanup to allow garbage collection
                        if vad_model:
                            try:
                                del vad_model
                            except:
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
                            except:
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
                        except Exception as e:
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
                    print("[READY] Transcription system initialized successfully!")

                    while True:
                        try:
                            # Check if we should exit the loop
                            if not is_running:
                                print(f"[LOOP] is_running is False, exiting main loop")
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
                                                                    print(f"[CALIBRATION] Step 1 complete - WARNING: No noise samples collected, using defaults", flush=True)

                                                                # Mark step 1 as complete but DON'T auto-transition
                                                                # Wait for user to click "Start Step 2" button
                                                                calibration_state["step1_complete"] = True
                                                                print(f"[CALIBRATION] Set step1_complete = True", flush=True)
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
                                                                except:
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
                                                except Exception:
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
                                                f"[BACKUP] Started session audio file"
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
                                except Exception as e:
                                    print(f"[WARNING] Audio level calculation failed: {e}")

                                # Check if audio contains speech using VAD
                                speech_detected = has_speech(accumulated_new_data, source.SAMPLE_RATE)

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
                                # print(f"[DEBUG-AUDIO] Chunk duration: {chunk_duration:.2f}s, audio size: {len(audio_chunk) if audio_chunk is not None else 'None'}", flush=True)

                                # Need at least 1 second of audio before transcribing
                                if chunk_duration < 1.0:
                                    sleep(0.1)
                                    continue

                                try:
                                    # Get language from config
                                    live_language = process_config.get("audio", {}).get("language", "auto")
                                    # Get whisper params from config or use defaults
                                    whisper_params = process_config.get("whisper_decoding", {}).get(
                                        "live_transcription", LIVE_TRANSCRIPTION_PARAMS
                                    )

                                    # Enable word_timestamps for confidence highlighting if configured
                                    corrections_config = process_config.get("corrections", {})
                                    if corrections_config.get("confidence_highlighting", True) and corrections_config.get("enabled", True):
                                        whisper_params = dict(whisper_params)  # Copy to avoid mutating config
                                        whisper_params["word_timestamps"] = True

                                    # Inject custom dictionary hotwords if configured
                                    dict_config = process_config.get("custom_dictionary", {})
                                    if dict_config.get("whisper_hotwords_enabled", False) or dict_config.get("whisper_initial_prompt_enabled", False):
                                        dict_file = dict_config.get("file", "custom_dictionary.json")
                                        if not os.path.isabs(dict_file):
                                            dict_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), dict_file)
                                        if os.path.exists(dict_file):
                                            try:
                                                import json as _json
                                                with open(dict_file, "r", encoding="utf-8") as _f:
                                                    _dict_data = _json.load(_f)
                                                hotwords_list = _dict_data.get("hotwords", [])
                                                if hotwords_list:
                                                    if isinstance(whisper_params, dict) is False:
                                                        whisper_params = dict(whisper_params)
                                                    if dict_config.get("whisper_hotwords_enabled", False):
                                                        whisper_params["hotwords"] = " ".join(hotwords_list)
                                                    if dict_config.get("whisper_initial_prompt_enabled", False) and "initial_prompt" not in whisper_params:
                                                        whisper_params["initial_prompt"] = ", ".join(hotwords_list)
                                            except Exception as dict_err:
                                                print(f"[DICTIONARY] Error loading hotwords: {dict_err}", flush=True)

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

                                    # Filter out hallucinated foreign characters from each segment
                                    if segments and config.get("hallucination_filter", {}).get("cjk_filter_enabled", True):
                                        for seg in segments:
                                            if seg.get('text'):
                                                seg['text'] = filter_hallucinated_text(seg['text'], live_language)

                                    # Update segments using Whisper-Live's approach
                                    # This finalizes all segments except the last one immediately
                                    result = live_transcriber.update_segments(segments, chunk_duration)
                                    # print(f"[DEBUG-SEGMENTS] Completed: {len(result.get('completed_segments', []))}, Current text: '{result.get('current_text', '')[:50]}...'", flush=True)

                                    # Handle completed segments (save to DB)
                                    if result['completed_segments']:
                                        confidence_threshold = process_config.get("corrections", {}).get("confidence_threshold", 0.7)
                                        for segment in result['completed_segments']:
                                            segment_text = segment.get('text', '').strip()
                                            segment_start = segment.get('start', 0)
                                            segment_end = segment.get('end', 0)
                                            # Compute average word confidence for this segment
                                            segment_confidence = None
                                            word_confidences = segment.get('words', [])
                                            if word_confidences:
                                                probs = [w.get('probability') for w in word_confidences if w.get('probability') is not None]
                                                if probs:
                                                    segment_confidence = sum(probs) / len(probs)
                                            if segment_text:
                                                # Remove overlapping prefix from previous saved text
                                                original_segment_text = segment_text
                                                if saved_sentences:
                                                    segment_text = remove_overlapping_prefix(segment_text, saved_sentences[-1])
                                                    if segment_text != original_segment_text:
                                                        pass  # Overlap was removed
                                                    if not segment_text:
                                                        continue  # Skip if entire segment was overlap

                                                # Split into sentences
                                                sentences, remainder = split_into_sentences(segment_text)
                                                # print(f"[DEBUG-SPLIT] Input: '{segment_text[:60]}' -> Sentences: {len(sentences)}, Remainder: '{remainder[:30] if remainder else 'None'}'", flush=True)
                                                # Debug logging commented out for production
                                                # if not sentences:
                                                #     print(f"[DEBUG] No sentences from: '{segment_text[:60]}...' (remainder: '{remainder}')")

                                                # Save substantial sentences to DB
                                                MIN_WORDS = min_words_threshold
                                                with _db_lock:
                                                    try:
                                                        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                                                        for sentence in sentences:
                                                            # Skip known Whisper hallucinations
                                                            if is_whisper_hallucination(sentence):
                                                                print(f"[SKIP HALLUCINATION] '{sentence[:40]}'", flush=True)
                                                                continue
                                                            word_count = len(sentence.split())
                                                            is_dup = is_fuzzy_duplicate(sentence, saved_sentences, fuzzy_threshold)
                                                            if word_count >= MIN_WORDS and not is_dup:
                                                                needs_review = 1 if (segment_confidence is not None and segment_confidence < confidence_threshold) else 0
                                                                persistent_db_cursor.execute(
                                                                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, needs_review) VALUES (?, ?, ?, ?, ?, ?)",
                                                                    (timestamp, sentence, segment_start, segment_end, segment_confidence, needs_review),
                                                                )
                                                                saved_sentences.append(sentence)
                                                                conf_str = f", conf={segment_confidence:.2f}" if segment_confidence is not None else ""
                                                                print(f"[DB INSERT] '{sentence[:50]}...'{conf_str}" if len(sentence) > 50 else f"[DB INSERT] '{sentence}'{conf_str}", flush=True)
                                                            elif word_count < MIN_WORDS:
                                                                print(f"[SKIP SHORT] '{sentence}' ({word_count} words, min={MIN_WORDS})", flush=True)
                                                            elif is_dup:
                                                                # Find which saved sentence it matched
                                                                from difflib import SequenceMatcher
                                                                for saved in saved_sentences[-5:]:  # Check last 5
                                                                    ratio = SequenceMatcher(None, sentence.lower(), saved.lower()).ratio()
                                                                    if ratio >= fuzzy_threshold:
                                                                        print(f"[SKIP DUP] '{sentence[:40]}' matched '{saved[:40]}' (ratio={ratio:.2f})", flush=True)
                                                                        break
                                                        # Also save substantial remainder from finalized segments
                                                        if remainder and not is_whisper_hallucination(remainder):
                                                            rem_word_count = len(remainder.split())
                                                            rem_is_dup = is_fuzzy_duplicate(remainder, saved_sentences, fuzzy_threshold)
                                                            if rem_word_count >= MIN_WORDS and not rem_is_dup:
                                                                needs_review = 1 if (segment_confidence is not None and segment_confidence < confidence_threshold) else 0
                                                                persistent_db_cursor.execute(
                                                                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, needs_review) VALUES (?, ?, ?, ?, ?, ?)",
                                                                    (timestamp, remainder, segment_start, segment_end, segment_confidence, needs_review),
                                                                )
                                                                saved_sentences.append(remainder)
                                                                print(f"[DB INSERT REMAINDER] '{remainder[:50]}...'" if len(remainder) > 50 else f"[DB INSERT REMAINDER] '{remainder}'", flush=True)
                                                            elif rem_word_count < MIN_WORDS:
                                                                print(f"[SKIP SHORT REMAINDER] '{remainder}' ({rem_word_count} words)", flush=True)
                                                            elif rem_is_dup:
                                                                print(f"[SKIP DUP REMAINDER] '{remainder[:50]}...'" if len(remainder) > 50 else f"[SKIP DUP REMAINDER] '{remainder}'", flush=True)
                                                        # print(f"[LOOP-DEBUG] {time.strftime('%H:%M:%S')} - DB commit start", flush=True)
                                                        persistent_db_conn.commit()
                                                        # Track saved_sentences and database row count
                                                        # print(f"[DEBUG-SAVED] saved_sentences count: {len(saved_sentences)}, last 3: {[s[:30] for s in saved_sentences[-3:]] if len(saved_sentences) >= 3 else saved_sentences}", flush=True)
                                                        # Periodically verify database row count matches
                                                        if len(saved_sentences) % 10 == 0:
                                                            db_count = persistent_db_cursor.execute("SELECT COUNT(*) FROM transcriptions").fetchone()[0]
                                                            # print(f"[DEBUG-DB-COUNT] Database has {db_count} rows, saved_sentences has {len(saved_sentences)} entries", flush=True)
                                                            if db_count != len(saved_sentences) + 1:  # +1 for default first entry
                                                                pass  # Row count mismatch — non-critical
                                                        # print(f"[LOOP-DEBUG] {time.strftime('%H:%M:%S')} - DB commit done", flush=True)
                                                    except Exception as db_error:
                                                        print(f"[ERROR] DB save failed: {db_error}")

                                                # print(f"[FINALIZED] '{segment_text[:60]}...'" if len(segment_text) > 60 else f"[FINALIZED] '{segment_text}'")

                                    # Handle phrase completion (silence timeout)
                                    if phrase_complete:
                                        # FIX: Check if update_segments already finalized via same_output
                                        # If so, don't double-process (it's already in completed_segments)
                                        just_finalized = result.get('just_finalized_text', '')
                                        if just_finalized:
                                            # print(f"[DEBUG-PHRASE] Text already finalized via same_output: '{just_finalized[:50]}'", flush=True)
                                            finalized_segment = None  # Already handled
                                        else:
                                            # FIX: Capture pending text BEFORE force_finalize (which clears current_out)
                                            # This handles the case where same_output finalization already cleared current_out
                                            pending_text = result.get('current_text', '').strip()

                                            # Force finalize any remaining text
                                            finalized_segment = live_transcriber.force_finalize()
                                            # print(f"[DEBUG-PHRASE] Force finalize returned: {finalized_segment is not None}, text: '{finalized_segment.get('text', '')[:50] if finalized_segment else 'None'}'", flush=True)

                                            # FIX: If force_finalize returned nothing but we had pending text, create segment from it
                                            if finalized_segment is None and pending_text:
                                                # print(f"[DEBUG-PHRASE] Using pending text (force_finalize was empty): '{pending_text[:50]}'", flush=True)
                                                finalized_segment = {
                                                    'text': pending_text,
                                                    'start': live_transcriber.timestamp_offset,
                                                    'end': live_transcriber.timestamp_offset + chunk_duration,
                                                    'completed': True
                                                }

                                        if finalized_segment:
                                            segment_text = finalized_segment.get('text', '').strip()
                                            segment_start = finalized_segment.get('start', 0)
                                            segment_end = finalized_segment.get('end', 0)
                                            if segment_text:
                                                # Remove overlapping prefix from previous saved text
                                                original_segment_text = segment_text
                                                if saved_sentences:
                                                    segment_text = remove_overlapping_prefix(segment_text, saved_sentences[-1])
                                                    if segment_text != original_segment_text:
                                                        pass  # Overlap was removed
                                                    if not segment_text:
                                                        pass  # Entire segment was overlap — will be skipped below

                                                if segment_text:  # Check again after overlap removal
                                                    sentences, remainder = split_into_sentences(segment_text)
                                                    # print(f"[DEBUG-PHRASE-SPLIT] Input: '{segment_text[:60]}' -> Sentences: {len(sentences)}, Remainder: '{remainder[:30] if remainder else 'None'}'", flush=True)
                                                    MIN_WORDS = min_words_threshold
                                                    with _db_lock:
                                                        try:
                                                            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                                                            for sentence in sentences:
                                                                # Skip known Whisper hallucinations
                                                                if is_whisper_hallucination(sentence):
                                                                    print(f"[SKIP HALLUCINATION PHRASE] '{sentence[:40]}'", flush=True)
                                                                    continue
                                                                word_count = len(sentence.split())
                                                                is_dup = is_fuzzy_duplicate(sentence, saved_sentences, fuzzy_threshold)
                                                                if word_count >= MIN_WORDS and not is_dup:
                                                                    persistent_db_cursor.execute(
                                                                        "INSERT INTO transcriptions (timestamp, text, start_time, end_time) VALUES (?, ?, ?, ?)",
                                                                        (timestamp, sentence, segment_start, segment_end),
                                                                    )
                                                                    saved_sentences.append(sentence)
                                                                    print(f"[DB INSERT PHRASE] '{sentence[:50]}...'" if len(sentence) > 50 else f"[DB INSERT PHRASE] '{sentence}'", flush=True)
                                                                elif word_count < MIN_WORDS:
                                                                    print(f"[SKIP SHORT PHRASE] '{sentence}' ({word_count} words, min={MIN_WORDS})", flush=True)
                                                                elif is_dup:
                                                                    print(f"[SKIP DUP PHRASE] '{sentence[:40]}'", flush=True)
                                                            # Also save substantial remainder from phrase_complete
                                                            if remainder and not is_whisper_hallucination(remainder):
                                                                rem_word_count = len(remainder.split())
                                                                rem_is_dup = is_fuzzy_duplicate(remainder, saved_sentences, fuzzy_threshold)
                                                                if rem_word_count >= MIN_WORDS and not rem_is_dup:
                                                                    persistent_db_cursor.execute(
                                                                        "INSERT INTO transcriptions (timestamp, text, start_time, end_time) VALUES (?, ?, ?, ?)",
                                                                        (timestamp, remainder, segment_start, segment_end),
                                                                    )
                                                                    saved_sentences.append(remainder)
                                                                    # print(f"[DB INSERT REMAINDER] '{remainder[:50]}...'" if len(remainder) > 50 else f"[DB INSERT REMAINDER] '{remainder}'")
                                                                # elif rem_word_count < MIN_WORDS:
                                                                #     print(f"[SKIP SHORT REMAINDER] '{remainder}' ({rem_word_count} words)")
                                                            persistent_db_conn.commit()
                                                        except Exception as db_error:
                                                            print(f"[ERROR] phrase_complete DB save failed: {db_error}")

                                                    # print(f"[PHRASE_COMPLETE] Finalized: '{segment_text[:40]}...'")

                                        # Clear live preview
                                        transcription_state["live_text"] = ""
                                        transcription_state["live_start"] = 0
                                        transcription_state["live_end"] = 0

                                    # Update live preview with current incomplete text only
                                    # (finalized segments are already shown separately from the database)
                                    current_text = result.get('current_text', '')
                                    if current_text:
                                        old_live_text = transcription_state.get("live_text", "")
                                        transcription_state["live_text"] = current_text
                                        # Track timing for in-progress segment
                                        transcription_state["live_start"] = live_transcriber.timestamp_offset
                                        transcription_state["live_end"] = live_transcriber.timestamp_offset + chunk_duration
                                        # Store word-level confidence for in-progress display
                                        if hasattr(live_transcriber, '_last_seg_confidence'):
                                            transcription_state["live_word_confidences"] = live_transcriber._last_seg_confidence.get('words', [])
                                        # print(f"[DEBUG-LIVE] Setting live_text: '{current_text[:50]}...' (was: '{old_live_text[:30]}...')", flush=True)

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
                            import wave

                            # Get current file size
                            file_size = os.path.getsize(session_audio_file)

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
                                print(f"[SRT] Successfully created SRT file")
                            else:
                                print(f"[SRT] No SRT file created (no valid entries or error)")
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
                                print(f"[SRT-TRANSLATION] Successfully created translation SRT file")
                            else:
                                print(f"[SRT-TRANSLATION] No translation SRT created (no translated entries)")
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
                        sleep(10)
                        print("[FILE MOVER] Executing file move after final cleanup...")
                        result = execute_file_move_now(lambda: current_config)
                        if result['success']:
                            print(f"[FILE MOVER] OK: Moved {result['moved']} files")
                            if result['failed'] > 0:
                                print(f"[FILE MOVER] ! {result['failed']} files failed")
                        else:
                            print(f"[FILE MOVER] FAIL: Error: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    print(f"[FILE MOVER] Error executing file mover: {e}")
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
    signal.signal(signal.SIGINT, signal_handler)

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
