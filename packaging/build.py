"""
Local build script — wraps the PyInstaller spec for convenience.

Builds the thin STT bootstrapper (watchdog.py only, ~10-20 MB). It provisions a
local venv + downloads dependencies + models on first run; nothing heavy is
bundled here.

Usage (from anywhere):
    python packaging/build.py [--platform NAME]

Output (in the repo root):
    dist/STT/            — tiny bootstrapper application directory (one-dir)
"""

import os
import sys
import shutil
import platform
import subprocess
import argparse

# Everything (icons, PyInstaller build/dist dirs) is produced in the repo root,
# regardless of where this script is invoked from.
PACKAGING_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PACKAGING_DIR)


def run(cmd):
    print(f"\n> {' '.join(cmd)}\n")
    ret = subprocess.call(cmd, cwd=ROOT)
    if ret != 0:
        print(f"ERROR: command exited with code {ret}")
        sys.exit(ret)


def get_platform_name():
    s = platform.system().lower()
    return "windows" if s == "windows" else "macos" if s == "darwin" else "linux"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", default=get_platform_name(),
                        help="Platform name for output directory label")
    args = parser.parse_args()

    # Clean previous artefacts
    for d in ("build", "dist"):
        d = os.path.join(ROOT, d)
        if os.path.exists(d):
            shutil.rmtree(d)

    # Generate application icon (requires Pillow)
    run([sys.executable, os.path.join(PACKAGING_DIR, "make_icon.py")])

    # Single build: the thin bootstrapper (no ML libs; deps installed on first run)
    run([sys.executable, "-m", "PyInstaller",
         os.path.join(PACKAGING_DIR, "watchdog.spec"), "--noconfirm"])

    # Clean intermediate build dir
    shutil.rmtree(os.path.join(ROOT, "build"), ignore_errors=True)

    out = os.path.join(ROOT, "dist", "STT")
    print(f"\nBuild complete: {out}")
    exe = "STT.exe" if sys.platform == "win32" else "STT"
    print(f"Run: {os.path.join(out, exe)}")


if __name__ == "__main__":
    main()
