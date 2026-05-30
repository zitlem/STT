"""
PyInstaller spec for the STT Watchdog (stdlib-only, one-file).

Build:
    pyinstaller watchdog.spec

Or via build.py:
    python build.py
"""

import sys

block_cipher = None
IS_WINDOWS = sys.platform == "win32"
IS_MACOS   = sys.platform == "darwin"

a = Analysis(
    ["watchdog.py"],
    pathex=["."],
    binaries=[],
    datas=[("VERSION", ".")],
    hiddenimports=["tkinter", "_tkinter", "tkinter.ttk"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# One-file: watchdog is tiny (stdlib only), so extraction overhead is minimal
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="STT-Watchdog",
    debug=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=IS_MACOS,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
