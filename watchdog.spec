"""
PyInstaller spec for the STT thin bootstrapper.

This builds a tiny (~10-20 MB) launcher from watchdog.py only. No ML libraries
are bundled: on first run the bootstrapper provisions a local venv (via uv),
clones the app source, and installs dependencies + ffmpeg on the user machine.
The STT server then runs as a real script from that venv.

Build:
    pyinstaller watchdog.spec
Or via build.py:
    python build.py
"""

import sys
import os

block_cipher = None
IS_WINDOWS = sys.platform == "win32"
IS_MACOS   = sys.platform == "darwin"

_icon_file = "icon.icns" if IS_MACOS else "icon.ico"
_icon = _icon_file if os.path.exists(_icon_file) else None

a = Analysis(
    ["watchdog.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("VERSION", "."),
    ] + ([(_icon_file, ".")] if os.path.exists(_icon_file) else []),
    # Bootstrapper needs only stdlib + tkinter (setup GUI) + certifi (TLS).
    # Everything else arrives via `git clone` + `uv pip install` on first run.
    hiddenimports=[
        "tkinter", "_tkinter", "tkinter.ttk", "tkinter.scrolledtext",
        "certifi",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "pytest",
        "IPython",
        "jupyter",
        "notebook",
        # Guard against accidentally sweeping a dev venv's ML libs into the exe.
        "torch",
        "torchaudio",
        "transformers",
        "faster_whisper",
        "whisper",
        "huggingface_hub",
        "numpy",
        "scipy",
        "librosa",
        "pandas",
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
