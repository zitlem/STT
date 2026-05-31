"""
PyInstaller spec for the STT Watchdog — single binary that also runs the STT server.

The watchdog self-relaunches with --run-stt to start speech_to_text.py from within
the same bundle, so users only need one executable.

Build:
    pyinstaller watchdog.spec

Or via build.py:
    python build.py
"""

import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None
IS_WINDOWS = sys.platform == "win32"
IS_MACOS   = sys.platform == "darwin"

_icon_file = "icon.icns" if IS_MACOS else "icon.ico"
_icon = _icon_file if os.path.exists(_icon_file) else None

# Heavy packages needed by speech_to_text.py
_torch_d, _torch_b, _torch_h                     = collect_all("torch")
_torchaudio_d, _torchaudio_b, _torchaudio_h      = collect_all("torchaudio")
_transformers_d, _transformers_b, _transformers_h = collect_all("transformers")
_fw_d, _fw_b, _fw_h                               = collect_all("faster_whisper")
_hf_d, _hf_b, _hf_h                               = collect_all("huggingface_hub")

a = Analysis(
    ["watchdog.py"],
    pathex=["."],
    binaries=_torch_b + _torchaudio_b + _fw_b + _transformers_b,
    datas=[
        # Watchdog assets
        ("VERSION", "."),
] + ([(_icon_file, ".")] if os.path.exists(_icon_file) else []) + [
        # STT server script and local modules (launched via --run-stt self-relaunch)
        ("speech_to_text.py",          "."),
        ("file_mover.py",              "."),
        ("audio_capture.py",           "."),
        ("huggingface_manager.py",     "."),
        # STT web assets and model lists
        ("templates",                  "templates"),
        ("static",                     "static"),
        ("faster_whisper_models.json", "."),
        ("whisper_models.json",        "."),
        ("word_highlighting.json",     "."),
        ("custom_dictionary.json",     "."),
    ] + _torch_d + _torchaudio_d + _fw_d + _transformers_d + _hf_d,
    hiddenimports=[
        # Watchdog GUI
        "tkinter", "_tkinter", "tkinter.ttk",
        # STT server deps
        "engineio.async_drivers.threading",
        "flask.templating",
        "flask_socketio",
        "flask_cors",
        "speech_recognition",
        "soundfile",
        "librosa",
        "numpy",
        "torch",
        "torchaudio",
        "whisper",
        "faster_whisper",
        "transformers",
        "huggingface_hub",
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
    console=not IS_WINDOWS,
    disable_windowed_traceback=False,
    argv_emulation=IS_MACOS,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=_icon,
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

if IS_MACOS:
    app = BUNDLE(
        coll,
        name="STT.app",
        icon=_icon,
        bundle_identifier="com.stt.watchdog",
        info_plist={
            "CFBundleName": "STT",
            "CFBundleDisplayName": "STT",
            "CFBundleExecutable": "STT",
            "NSHighResolutionCapable": True,
            "NSMicrophoneUsageDescription":
                "STT needs microphone access for speech recognition.",
            "LSUIElement": False,
        },
    )
