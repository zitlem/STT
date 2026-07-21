"""
PyInstaller spec for the STT thin bootstrapper.

This builds a tiny (~10-20 MB) launcher from stt/watchdog.py only. No ML libraries
are bundled: on first run the bootstrapper provisions a local venv (via uv),
clones the app source, and installs dependencies + ffmpeg on the user machine.
The STT server then runs as a real script from that venv.

Build (from the repo root):
    pyinstaller packaging/watchdog.spec
Or via build.py:
    python packaging/build.py
"""

import sys
import os

block_cipher = None
IS_WINDOWS = sys.platform == "win32"
IS_MACOS   = sys.platform == "darwin"

# Anchor all inputs to the repo root (parent of this spec's dir) so the build
# works regardless of the invoking cwd. Icons are generated into the root by
# make_icon.py; dist/ and the build workdir land in the invoking cwd as before.
ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821 (SPECPATH is a PyInstaller global)

_icon_file = os.path.join(ROOT, "icon.icns" if IS_MACOS else "icon.ico")
_icon = _icon_file if os.path.exists(_icon_file) else None

# Bundle version for the .app's Info.plist so macOS ("About STT", Finder Get Info)
# shows the real version instead of PyInstaller's default 0.0.0. Read from the
# git-managed VERSION file that's also bundled as runtime data.
try:
    with open(os.path.join(ROOT, "VERSION")) as _vf:
        _bundle_version = _vf.read().strip() or "0.0.0"
except OSError:
    _bundle_version = "0.0.0"

a = Analysis(
    [os.path.join(ROOT, "stt", "watchdog.py")],
    pathex=[ROOT],
    binaries=[],
    datas=[
        (os.path.join(ROOT, "VERSION"), "."),
    ] + ([(_icon_file, ".")] if os.path.exists(_icon_file) else []),
    # Bootstrapper needs stdlib + tkinter (setup GUI) + certifi (TLS) +
    # sentry_sdk (so first-run provisioning failures are reported — the venv
    # that normally provides the SDK doesn't exist yet during setup).
    # Everything else arrives via `git clone` + `uv pip install` on first run.
    hiddenimports=[
        "tkinter", "_tkinter", "tkinter.ttk", "tkinter.scrolledtext",
        "certifi",
        # sentry_sdk loads its default integrations dynamically; list them so
        # PyInstaller's static analysis picks them up.
        "sentry_sdk",
        "sentry_sdk.integrations.stdlib",
        "sentry_sdk.integrations.excepthook",
        "sentry_sdk.integrations.dedupe",
        "sentry_sdk.integrations.atexit",
        "sentry_sdk.integrations.modules",
        "sentry_sdk.integrations.logging",
        "sentry_sdk.integrations.threading",
        "sentry_sdk.integrations.argv",
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
            "CFBundleShortVersionString": _bundle_version,
            "CFBundleVersion": _bundle_version,
            "NSHighResolutionCapable": True,
            "NSMicrophoneUsageDescription":
                "STT needs microphone access for speech recognition.",
            "LSUIElement": False,
        },
    )
