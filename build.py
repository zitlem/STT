"""
Build script for STT application.
Creates a distributable package using PyInstaller.

Usage:
    python build.py
"""

import os
import sys
import shutil
import platform

APP_NAME = "stt"
ENTRY_POINT = "speech_to_text.py"

# Directories/files to bundle inside the exe
DATA_FILES = [
    ("templates", "templates"),
    ("static", "static"),
]

# JSON files to copy next to the exe (user-editable)
EXTERNAL_FILES = [
    "config.json",
    "whisper_models.json",
    "faster_whisper_models.json",
    "custom_dictionary.json",
    "word_highlighting.json",
]

# Additional Python files needed
EXTRA_MODULES = [
    "file_mover.py",
    "huggingface_manager.py",
    "audio_capture.py",
]

# Hidden imports that PyInstaller can't detect automatically
HIDDEN_IMPORTS = [
    "engineio.async_drivers.threading",
    "flask.templating",
    "flask_socketio",
    "flask_cors",
    "speech_recognition",
    "pyaudio",
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
]

# Packages to exclude (reduce size)
EXCLUDES = [
    "tkinter",
    "matplotlib",
    "pytest",
    "IPython",
    "jupyter",
    "notebook",
    "scipy.tests",
    "torch.testing",
]


def get_platform_name():
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    elif system == "darwin":
        return "macos"
    else:
        return "linux"


def build():
    platform_name = get_platform_name()
    dist_dir = os.path.join("dist", f"{APP_NAME}-{platform_name}")

    print(f"Building {APP_NAME} for {platform_name}...")
    print(f"Output directory: {dist_dir}")

    # Clean previous build
    if os.path.exists(dist_dir):
        shutil.rmtree(dist_dir)
    if os.path.exists("build"):
        shutil.rmtree("build")

    # Build PyInstaller command
    separator = ";" if platform.system() == "Windows" else ":"

    cmd_parts = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--console",
        f"--name={APP_NAME}",
        f"--distpath={dist_dir}",
    ]

    # Add data files (bundled inside exe)
    for src, dst in DATA_FILES:
        if os.path.exists(src):
            cmd_parts.append(f"--add-data={src}{separator}{dst}")

    # Add extra module files
    for module in EXTRA_MODULES:
        if os.path.exists(module):
            cmd_parts.append(f"--add-data={module}{separator}.")

    # Add hidden imports
    for imp in HIDDEN_IMPORTS:
        cmd_parts.append(f"--hidden-import={imp}")

    # Add excludes
    for exc in EXCLUDES:
        cmd_parts.append(f"--exclude-module={exc}")

    # Entry point
    cmd_parts.append(ENTRY_POINT)

    # Run PyInstaller
    cmd = " ".join(f'"{p}"' if " " in p else p for p in cmd_parts)
    print(f"\nRunning: {cmd}\n")
    ret = os.system(cmd)

    if ret != 0:
        print(f"\nERROR: PyInstaller failed with exit code {ret}")
        sys.exit(1)

    # Copy external files next to the exe
    exe_dir = os.path.join(dist_dir, APP_NAME)
    print(f"\nCopying external files to {exe_dir}...")

    for f in EXTERNAL_FILES:
        if os.path.exists(f):
            shutil.copy2(f, exe_dir)
            print(f"  Copied {f}")

    # Create runtime directories
    for dirname in ["models", "_AUTOMATIC_BACKUP", "logs"]:
        dirpath = os.path.join(exe_dir, dirname)
        os.makedirs(dirpath, exist_ok=True)
        print(f"  Created {dirname}/")

    # Clean build artifacts
    if os.path.exists("build"):
        shutil.rmtree("build")
    spec_file = f"{APP_NAME}.spec"
    if os.path.exists(spec_file):
        os.remove(spec_file)

    print(f"\nBuild complete!")
    print(f"Output: {os.path.abspath(exe_dir)}")
    print(f"\nTo run:")
    if platform_name == "windows":
        print(f"  cd {exe_dir} && {APP_NAME}.exe")
    else:
        print(f"  cd {exe_dir} && ./{APP_NAME}")


if __name__ == "__main__":
    build()
