"""
Local build script — wraps PyInstaller spec files for convenience.

Usage:
    python build.py [--platform NAME]

Output:
    dist/STT/            — single application directory (one-dir)
"""

import os
import sys
import shutil
import platform
import subprocess
import argparse


def run(cmd):
    print(f"\n> {' '.join(cmd)}\n")
    ret = subprocess.call(cmd)
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
        if os.path.exists(d):
            shutil.rmtree(d)

    # Generate application icon (requires Pillow)
    run([sys.executable, "make_icon.py"])

    # Single build: watchdog bundles speech_to_text.py and all STT deps
    run([sys.executable, "-m", "PyInstaller", "watchdog.spec", "--noconfirm"])

    # Clean intermediate build dir
    shutil.rmtree("build", ignore_errors=True)

    out = os.path.abspath(os.path.join("dist", "STT"))
    print(f"\nBuild complete: {out}")
    exe = "STT.exe" if sys.platform == "win32" else "STT"
    print(f"Run: {os.path.join(out, exe)}")


if __name__ == "__main__":
    main()
