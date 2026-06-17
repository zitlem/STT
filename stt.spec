"""
PyInstaller spec for the STT main application.

Build:
    pyinstaller stt.spec

Or via build.py:
    python build.py
"""

import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None
IS_WINDOWS = sys.platform == "win32"
IS_MACOS   = sys.platform == "darwin"

# Heavy packages that use dynamic loading — must be fully collected
_torch_d, _torch_b, _torch_h         = collect_all("torch")
_torchaudio_d, _torchaudio_b, _torchaudio_h = collect_all("torchaudio")
_transformers_d, _transformers_b, _transformers_h = collect_all("transformers")
_fw_d, _fw_b, _fw_h                  = collect_all("faster_whisper")
_hf_d, _hf_b, _hf_h                  = collect_all("huggingface_hub")

a = Analysis(
    ["speech_to_text.py"],
    pathex=["."],
    binaries=_torch_b + _torchaudio_b + _fw_b + _transformers_b,
    datas=[
        # Bundled read-only assets (available via BUNDLE_DIR / sys._MEIPASS)
        ("templates",                "templates"),
        ("static",                   "static"),
        ("faster_whisper_models.json", "."),
        ("whisper_models.json",      "."),
        ("word_highlighting.json",   "."),
        ("custom_dictionary.json",   "."),
        ("config.default.json",      "."),
        ("VERSION",                  "."),
        # Local modules
        ("file_mover.py",            "."),
        ("audio_capture.py",         "."),
        ("huggingface_manager.py",   "."),
    ] + _torch_d + _torchaudio_d + _fw_d + _transformers_d + _hf_d,
    hiddenimports=[
        "engineio.async_drivers.threading",
        "flask.templating",
        "flask_socketio",
        "flask_cors",
        "speech_recognition",
        "pyaudio",
        "sounddevice",
        "soundfile",
        "librosa",
        "numpy",
        "torch",
        "torchaudio",
        "whisper",
        "faster_whisper",
        "transformers",
        "huggingface_hub",
        "accelerate",
        "datasets",
        "silero_vad",
        "edge_tts",
        "pydub",
        "pytz",
        "smbprotocol",
        "sqlite3",
        "certifi",
    ] + _torch_h + _torchaudio_h + _fw_h + _transformers_h + _hf_h,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "pytest",
        "IPython",
        "jupyter",
        "notebook",
        "scipy.tests",
        "torch.testing",
        "tkinter",
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="STT",
    debug=False,
    strip=False,
    upx=True,
    # Windows: no console (web UI is the interface); Linux/macOS: keep console for logs
    console=not IS_WINDOWS,
    disable_windowed_traceback=False,
    argv_emulation=IS_MACOS,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="STT",
)
