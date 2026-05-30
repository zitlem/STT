"""
Local build script — wraps PyInstaller spec files for convenience.

Usage:
    python build.py [--platform NAME]

Output:
    dist/STT/            — main application (one-dir)
    dist/STT/STT-Watchdog[.exe]  — watchdog (one-file, copied in)
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

    # Build main app
    run([sys.executable, "-m", "PyInstaller", "stt.spec", "--noconfirm"])

    # Build watchdog
    run([sys.executable, "-m", "PyInstaller", "watchdog.spec", "--noconfirm"])

    # Copy watchdog into the STT dist dir so it ships together
    watchdog_src = os.path.join("dist", "STT-Watchdog.exe" if sys.platform == "win32" else "STT-Watchdog")
    watchdog_dst = os.path.join("dist", "STT", os.path.basename(watchdog_src))
    shutil.copy2(watchdog_src, watchdog_dst)
    print(f"Copied {watchdog_src} → {watchdog_dst}")

    # Clean intermediate build dir
    shutil.rmtree("build", ignore_errors=True)

    out = os.path.abspath(os.path.join("dist", "STT"))
    print(f"\nBuild complete: {out}")
    exe = "STT.exe" if sys.platform == "win32" else "STT"
    print(f"Run: {os.path.join(out, exe)}")


if __name__ == "__main__":
    main()
