#!/usr/bin/env python3
"""
STT Watchdog — crash recovery and auto-updater for speech_to_text.py

Responsibilities:
  - Starts speech_to_text.py as a managed child process
  - Restarts it on crash with exponential backoff
  - Checks GitHub releases at 1am daily; applies updates while preserving config.json
  - Supports headless daemon mode and desktop GUI (tkinter)
  - Cross-platform: Linux, macOS, Windows

Usage:
  python3 watchdog.py             # auto-detect GUI vs headless
  python3 watchdog.py --gui       # force GUI window
  python3 watchdog.py --headless  # force headless daemon
  python3 watchdog.py --check-update         # one-shot update check then exit
  python3 watchdog.py --channel beta         # set update channel and run
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import ssl
import urllib.error
import urllib.request
import webbrowser
import zipfile

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception as e:
    logging.warning(f"[SSL] certifi unavailable ({e}); using system default certificates")
    _SSL_CTX = None

IS_WINDOWS = sys.platform == "win32"

# ---------------------------------------------------------------------------
# Crash reporting is fully local: crashes are written to logs/crashes/ and
# nothing is ever sent off the machine.
# ---------------------------------------------------------------------------

_FROZEN = getattr(sys, 'frozen', False)

# Thin-bootstrapper layout separates DATA (config/models/logs — never touched by
# updates) from SOURCE (the git checkout + its venv). When frozen, the tiny
# bootstrapper provisions SOURCE under DATA_DIR/app and runs the app from that
# venv. In dev-from-repo, both collapse to the repo directory.
if _FROZEN:
    DATA_DIR   = os.path.join(os.path.expanduser("~"), ".stt")
    SOURCE_DIR = os.path.join(DATA_DIR, "app")
else:
    DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
    SOURCE_DIR = DATA_DIR
APP_DIR = DATA_DIR  # config/logs/crash dumps live in DATA
os.makedirs(APP_DIR, exist_ok=True)

GITHUB_REPO = "zitlem/STT"
GITHUB_REPO_URL = f"https://github.com/{GITHUB_REPO}"
GITHUB_API_BASE = f"https://api.github.com/repos/{GITHUB_REPO}"
VERSION_FILE = os.path.join(SOURCE_DIR, "VERSION")  # git-managed
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")
LOG_DIR = os.path.join(DATA_DIR, "logs")

# The worker always runs as a real script from the venv (frozen or dev).
STT_SCRIPT = os.path.join(SOURCE_DIR, "speech_to_text.py")

# uv-provisioned Python for the venv (see requirements.txt / CI).
UV_PYTHON_VERSION = "3.11"

# Port used only for single-instance lock (never serves traffic)
_LOCK_PORT = 57337

BACKOFF = [5, 10, 30, 60]       # seconds between crash restarts; capped at last entry
STABLE_RUN_THRESHOLD = 30        # seconds of uptime before resetting crash counter
UPDATE_HOUR = 1                  # hour (24h) at which daily update check fires

# Files/dirs inside SOURCE never discarded by the zipball-fallback update path
# (DATA is a separate tree, so config/models/logs are already safe).
_UPDATE_PRESERVE = frozenset({".venv"})


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging():
    from logging.handlers import RotatingFileHandler
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        handlers.insert(0, RotatingFileHandler(
            os.path.join(LOG_DIR, "watchdog.log"),
            maxBytes=5_000_000, backupCount=3, encoding="utf-8",
        ))
    except OSError:
        pass
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def _rotate_if_large(path, max_bytes, backups):
    """Roll path -> path.1 -> path.2 ... when it exceeds max_bytes. Called while
    no process holds the file open (e.g. before (re)launching the STT worker), so
    a plain rename is safe. Best-effort: never raises."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) < max_bytes:
            return
        for i in range(backups - 1, 0, -1):
            src, dst = f"{path}.{i}", f"{path}.{i + 1}"
            if os.path.exists(src):
                os.replace(src, dst)
        os.replace(path, f"{path}.1")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def venv_python(venv_dir=None):
    """Path the venv's Python would live at (whether or not it exists yet)."""
    venv_dir = venv_dir or os.path.join(SOURCE_DIR, ".venv")
    if IS_WINDOWS:
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python3")


def get_python_bin():
    """Return the SOURCE venv Python if present, else the current interpreter."""
    venv_dir = os.path.join(SOURCE_DIR, ".venv")
    if IS_WINDOWS:
        candidates = [
            os.path.join(venv_dir, "Scripts", "python.exe"),
            os.path.join(venv_dir, "Scripts", "python3.exe"),
        ]
    else:
        candidates = [
            os.path.join(venv_dir, "bin", "python3"),
            os.path.join(venv_dir, "bin", "python"),
        ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return sys.executable


def load_config():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(cfg):
    tmp = CONFIG_FILE + ".watchdog.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    os.replace(tmp, CONFIG_FILE)


def read_version():
    # SOURCE/VERSION is git-managed (updated by git reset). Fall back to the
    # bundled VERSION inside the frozen bootstrapper before source is cloned.
    candidates = [VERSION_FILE]
    if _FROZEN:
        candidates.append(os.path.join(sys._MEIPASS, "VERSION"))
    for path in candidates:
        try:
            with open(path) as f:
                v = f.read().strip()
            if v:
                return v
        except (FileNotFoundError, OSError):
            pass
    return "0.0.0"


def write_version(version):
    tmp = VERSION_FILE + ".tmp"
    with open(tmp, "w") as f:
        f.write(version + "\n")
    os.replace(tmp, VERSION_FILE)


def parse_version(v):
    """'1.2.3' → (1, 2, 3); non-numeric parts become 0."""
    parts = v.lstrip("v").split(".")
    result = []
    for p in parts:
        try:
            result.append(int(p))
        except ValueError:
            result.append(0)
    return tuple(result)


def acquire_lock(open_browser_if_taken=False):
    """Single-instance guard: bind a local socket. Released automatically on exit."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    try:
        sock.bind(("127.0.0.1", _LOCK_PORT))
    except OSError:
        if open_browser_if_taken:
            return None  # caller will open monitoring GUI
        print("[ERROR] Another watchdog instance is already running.", file=sys.stderr)
        sys.exit(1)
    return sock  # keep reference alive; OS releases on process exit


# ---------------------------------------------------------------------------
# First-run provisioning (thin bootstrapper)
# ---------------------------------------------------------------------------

PROVISION_MARKER = os.path.join(DATA_DIR, ".provisioned")
_FFMPEG_BIN_DIR = os.path.join(DATA_DIR, "bin")


class ProvisionError(Exception):
    pass


def is_provisioned():
    """True when SOURCE is checked out with a venv the worker can run from."""
    have_source = os.path.isdir(os.path.join(SOURCE_DIR, ".git")) or os.path.exists(PROVISION_MARKER)
    return (
        have_source
        and os.path.isfile(STT_SCRIPT)
        and get_python_bin() != sys.executable
    )


def _augmented_path():
    """PATH with the common uv/local install locations prepended."""
    home = os.path.expanduser("~")
    extra = [
        os.path.join(home, ".local", "bin"),
        os.path.join(home, ".cargo", "bin"),
        _FFMPEG_BIN_DIR,
    ]
    if IS_WINDOWS:
        extra.append(os.path.join(home, ".local", "bin"))
    return os.pathsep.join(extra) + os.pathsep + os.environ.get("PATH", "")


def _which(name):
    return shutil.which(name, path=_augmented_path())


class Provisioner:
    """Builds the local runtime on first launch: uv, Python, git, ffmpeg, source
    checkout, venv, and dependencies. Each step is idempotent and retriable; the
    marker is only written once every step has succeeded. `log` is a callback
    (message) -> None used by both the GUI pane and headless logging."""

    # Ordered (label, method-name) — drives the "Step k/N" display.
    STEPS = [
        ("Checking disk space",        "_step_disk"),
        ("Installing uv",              "_step_uv"),
        ("Installing Python",          "_step_python"),
        ("Installing git",             "_step_git"),
        ("Installing ffmpeg",          "_step_ffmpeg"),
        ("Downloading application",     "_step_source"),
        ("Creating environment",       "_step_venv"),
        ("Installing dependencies",    "_step_deps"),
    ]

    def __init__(self, log=None):
        self.log = log or (lambda m: logging.info(m))
        self._uv = None  # resolved uv path

    # -- orchestration -------------------------------------------------------

    def run(self):
        """Run every step in order. Raises ProvisionError on the first failure."""
        total = len(self.STEPS)
        for i, (label, meth) in enumerate(self.STEPS, 1):
            self.log(f"[{i}/{total}] {label}...")
            getattr(self, meth)()
        self._write_marker()
        self.log("Setup complete.")

    def _write_marker(self):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(PROVISION_MARKER, "w", encoding="utf-8") as f:
                json.dump({
                    "version": read_version(),
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                    "gpu": self._has_nvidia(),
                    "source_dir": SOURCE_DIR,
                }, f, indent=2)
        except OSError as e:
            self.log(f"[WARN] Could not write provision marker: {e}")

    # -- process helper ------------------------------------------------------

    def _run(self, cmd, desc=None, check=True, timeout=3600, extra_env=None):
        """Run a subprocess, streaming stdout to the log callback."""
        if desc:
            self.log(f"  $ {desc}")
        env = dict(os.environ)
        env["PATH"] = _augmented_path()
        if extra_env:
            env.update(extra_env)
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0,
            )
        except FileNotFoundError as e:
            if check:
                raise ProvisionError(f"{cmd[0]} not found: {e}")
            return 1
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                self.log("    " + line)
        code = proc.wait(timeout=timeout)
        if check and code != 0:
            raise ProvisionError(f"command failed ({code}): {' '.join(str(c) for c in cmd)}")
        return code

    # -- environment detection ----------------------------------------------

    def _has_nvidia(self):
        return _which("nvidia-smi") is not None

    def _is_mac_arm(self):
        return sys.platform == "darwin" and platform.machine() == "arm64"

    # -- steps ---------------------------------------------------------------

    def _step_disk(self):
        try:
            free_gb = shutil.disk_usage(DATA_DIR).free / (1024 ** 3)
        except OSError:
            return
        self.log(f"  {free_gb:.1f} GB free")
        if free_gb < 8:
            raise ProvisionError(
                f"Only {free_gb:.1f} GB free; ~8 GB needed for PyTorch and models."
            )

    def _step_uv(self):
        uv = _which("uv")
        if uv:
            self._uv = uv
            self.log(f"  uv present: {uv}")
            return
        self.log("  downloading uv installer...")
        if IS_WINDOWS:
            ps = ("$ErrorActionPreference='Stop';"
                  "irm https://astral.sh/uv/install.ps1 | iex")
            self._run(["powershell", "-NoProfile", "-Command", ps],
                      desc="install uv (astral.sh)")
        else:
            script = self._download_text("https://astral.sh/uv/install.sh")
            tmp = os.path.join(tempfile.gettempdir(), "uv-install.sh")
            with open(tmp, "w") as f:
                f.write(script)
            self._run(["sh", tmp], desc="install uv (astral.sh)")
        self._uv = _which("uv")
        if not self._uv:
            raise ProvisionError("uv installation did not put 'uv' on PATH")
        self.log(f"  uv installed: {self._uv}")

    def _step_python(self):
        # uv-managed Python removes any dependency on a system Python.
        self._run([self._uv, "python", "install", UV_PYTHON_VERSION],
                  desc=f"uv python install {UV_PYTHON_VERSION}")

    def _step_git(self):
        if _which("git"):
            self.log("  git present")
            return
        self.log("  git missing; attempting to install...")
        try:
            if IS_WINDOWS and _which("winget"):
                self._run(["winget", "install", "--id", "Git.Git", "-e",
                           "--accept-source-agreements", "--accept-package-agreements"],
                          desc="winget install Git.Git")
            elif sys.platform == "darwin" and _which("brew"):
                self._run(["brew", "install", "git"], desc="brew install git")
            elif _which("apt-get"):
                self._run(["sudo", "-n", "apt-get", "install", "-y", "git"],
                          desc="apt-get install git", check=False)
        except ProvisionError as e:
            self.log(f"  [WARN] git install failed: {e}")
        # Not fatal: source can still be fetched via zipball fallback.
        if not _which("git"):
            self.log("  [WARN] git unavailable; will fetch source as an archive "
                     "(auto-updates will use archive fallback).")

    def _step_ffmpeg(self):
        if _which("ffmpeg"):
            self.log("  ffmpeg present")
            return
        self.log("  ffmpeg missing; attempting to install...")
        try:
            if IS_WINDOWS and _which("winget"):
                self._run(["winget", "install", "--id", "Gyan.FFmpeg", "-e",
                           "--accept-source-agreements", "--accept-package-agreements"],
                          desc="winget install Gyan.FFmpeg", check=False)
            elif sys.platform == "darwin" and _which("brew"):
                self._run(["brew", "install", "ffmpeg"], desc="brew install ffmpeg", check=False)
            elif _which("apt-get"):
                self._run(["sudo", "-n", "apt-get", "install", "-y", "ffmpeg"],
                          desc="apt-get install ffmpeg", check=False)
        except ProvisionError as e:
            self.log(f"  [WARN] package-manager ffmpeg install failed: {e}")
        if _which("ffmpeg"):
            self.log("  ffmpeg installed")
            return
        # Static-binary fallback into DATA_DIR/bin (prepended to worker PATH).
        self._install_static_ffmpeg()

    def _install_static_ffmpeg(self):
        os.makedirs(_FFMPEG_BIN_DIR, exist_ok=True)
        if sys.platform.startswith("linux"):
            url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
        elif sys.platform == "darwin":
            url = "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip"
        elif IS_WINDOWS:
            url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        else:
            raise ProvisionError("no static ffmpeg source for this platform")
        self.log(f"  downloading static ffmpeg from {url}")
        archive = os.path.join(tempfile.gettempdir(), os.path.basename(url.split("?")[0]) or "ffmpeg.pkg")
        self._download_file(url, archive)
        self._extract_ffmpeg(archive, _FFMPEG_BIN_DIR)
        if not _which("ffmpeg"):
            raise ProvisionError(
                "ffmpeg could not be provisioned; install it manually and re-run setup."
            )
        self.log(f"  static ffmpeg installed to {_FFMPEG_BIN_DIR}")

    def _extract_ffmpeg(self, archive, dest):
        """Extract just the ffmpeg (+ffprobe) binaries from a static archive into dest."""
        import tarfile
        wanted = ("ffmpeg", "ffprobe", "ffmpeg.exe", "ffprobe.exe")
        if archive.endswith((".tar.xz", ".tar.gz", ".txz", ".tgz")):
            with tarfile.open(archive) as tf:
                for m in tf.getmembers():
                    base = os.path.basename(m.name)
                    if base in wanted and m.isfile():
                        m.name = base
                        tf.extract(m, dest)
                        os.chmod(os.path.join(dest, base), 0o755)
        else:  # zip
            with zipfile.ZipFile(archive) as zf:
                for n in zf.namelist():
                    base = os.path.basename(n)
                    if base in wanted:
                        with zf.open(n) as src, open(os.path.join(dest, base), "wb") as out:
                            shutil.copyfileobj(src, out)
                        if not IS_WINDOWS:
                            os.chmod(os.path.join(dest, base), 0o755)

    def _step_source(self):
        if os.path.isfile(STT_SCRIPT) and os.path.isdir(os.path.join(SOURCE_DIR, ".git")):
            if _which("git"):
                self.log("  updating existing checkout...")
                self._run(["git", "-C", SOURCE_DIR, "fetch", "--depth", "1", "origin"],
                          desc="git fetch", check=False)
                self._run(["git", "-C", SOURCE_DIR, "reset", "--hard", "origin/HEAD"],
                          desc="git reset --hard", check=False)
            return
        os.makedirs(os.path.dirname(SOURCE_DIR) or DATA_DIR, exist_ok=True)
        if _which("git"):
            if os.path.isdir(SOURCE_DIR) and os.listdir(SOURCE_DIR):
                # Non-git leftovers — clear so clone can proceed (DATA is separate).
                shutil.rmtree(SOURCE_DIR, ignore_errors=True)
            self._run(["git", "clone", "--depth", "1", GITHUB_REPO_URL, SOURCE_DIR],
                      desc=f"git clone {GITHUB_REPO_URL}")
        else:
            self._fetch_source_zipball()
        if not os.path.isfile(STT_SCRIPT):
            raise ProvisionError("source checkout did not produce speech_to_text.py")

    def _fetch_source_zipball(self):
        """git-less fallback: download the default-branch source archive."""
        url = f"{GITHUB_REPO_URL}/archive/refs/heads/main.zip"
        self.log(f"  downloading source archive {url}")
        archive = os.path.join(tempfile.gettempdir(), "stt-source.zip")
        self._download_file(url, archive)
        os.makedirs(SOURCE_DIR, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            root = zf.namelist()[0].split("/")[0]
            for member in zf.infolist():
                rel = member.filename[len(root) + 1:]
                if not rel:
                    continue
                target = os.path.join(SOURCE_DIR, rel)
                # zip-slip guard
                if not os.path.abspath(target).startswith(os.path.abspath(SOURCE_DIR)):
                    continue
                if member.is_dir():
                    os.makedirs(target, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zf.open(member) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out)

    def _step_venv(self):
        venv_dir = os.path.join(SOURCE_DIR, ".venv")
        if os.path.isfile(get_python_bin()) and get_python_bin() != sys.executable:
            self.log("  venv present")
            return
        self._run([self._uv, "venv", venv_dir, "--python", UV_PYTHON_VERSION],
                  desc="uv venv")

    def install_deps_only(self, log=None):
        """Resolve uv and (re)install dependencies — reused by the auto-updater."""
        if log:
            self.log = log
        self._uv = _which("uv") or self._uv
        if not self._uv:
            self._step_uv()
        self._step_deps()

    def _step_deps(self):
        req = os.path.join(SOURCE_DIR, "requirements.txt")
        py = venv_python()
        cmd = [self._uv, "pip", "install", "--python", py, "-r", req]
        gpu = self._has_nvidia()
        if gpu and sys.platform.startswith(("linux", "win")):
            cmd += ["--extra-index-url", "https://download.pytorch.org/whl/cu121"]
            self.log("  NVIDIA GPU detected — installing CUDA wheels")
        elif self._is_mac_arm():
            self.log("  Apple Silicon — installing MPS/CPU wheels")
        else:
            self.log("  no NVIDIA GPU — installing CPU wheels")
        self._run(cmd, desc="uv pip install -r requirements.txt", timeout=7200)

    # -- download helpers ----------------------------------------------------

    def _download_text(self, url):
        req = urllib.request.Request(url, headers={"User-Agent": "STT-Bootstrapper"})
        with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
            return resp.read().decode("utf-8", "replace")

    def _download_file(self, url, dest):
        req = urllib.request.Request(url, headers={"User-Agent": "STT-Bootstrapper"})
        with urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as resp, \
                open(dest, "wb") as out:
            shutil.copyfileobj(resp, out)


# ---------------------------------------------------------------------------
# Shared state (thread-safe)
# ---------------------------------------------------------------------------

class WatchdogState:
    def __init__(self):
        self._lock = threading.Lock()
        self.status = "stopped"          # "starting"|"running"|"stopped"|"crashed"|"updating"
        self.process = None              # subprocess.Popen | None
        self.consecutive_crashes = 0
        self.last_update_check = ""
        self.last_update_result = ""
        self.stop_requested = False      # set True on watchdog shutdown

    def get(self, attr):
        with self._lock:
            return getattr(self, attr)

    def set(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)


# ---------------------------------------------------------------------------
# Process manager
# ---------------------------------------------------------------------------

class ProcessManager:
    def __init__(self, state, no_restart_event):
        self.state = state
        self._no_restart = no_restart_event   # set = don't auto-restart on exit
        self._python = get_python_bin()
        self._log_fh = None

    def start(self):
        with self.state._lock:
            if self.state.process and self.state.process.poll() is None:
                logging.warning("[PM] STT is already running")
                return False

        self._no_restart.clear()   # allow crash recovery once running
        log_path = os.path.join(LOG_DIR, "stt.log")
        try:
            if self._log_fh:
                try:
                    self._log_fh.close()
                except Exception:
                    pass
            # Bound stt.log across restarts (the worker holds it open via Popen
            # while running, so rotate now that the old handle is closed).
            _rotate_if_large(log_path, max_bytes=10_000_000, backups=5)
            self._log_fh = open(log_path, "a", encoding="utf-8")
            # Resolve the venv python fresh (provisioning may have created it after
            # this manager was constructed). Always run the real script from SOURCE.
            self._python = get_python_bin()
            cmd = [self._python, STT_SCRIPT]
            # Keep DATA and SOURCE separate for the worker, and expose any
            # provisioned ffmpeg (DATA_DIR/bin) so audio_capture's bare 'ffmpeg' resolves.
            env = dict(os.environ)
            env["STT_DATA_DIR"] = DATA_DIR
            env["PYTHONUNBUFFERED"] = "1"
            _ffbin = os.path.join(DATA_DIR, "bin")
            if os.path.isdir(_ffbin):
                env["PATH"] = _ffbin + os.pathsep + env.get("PATH", "")
            proc = subprocess.Popen(
                cmd,
                cwd=SOURCE_DIR,
                env=env,
                stdout=self._log_fh,
                stderr=self._log_fh,
                close_fds=not IS_WINDOWS,
                creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0,
            )
            self.state.set(process=proc, status="running", consecutive_crashes=0)
            logging.info(f"[PM] STT started (PID {proc.pid})")
            return True
        except Exception as e:
            logging.error(f"[PM] Failed to start STT: {e}")
            self.state.set(status="stopped")
            return False

    def stop(self, timeout=15):
        with self.state._lock:
            proc = self.state.process

        if proc is None or proc.poll() is not None:
            self.state.set(status="stopped", process=None)
            return True

        self._no_restart.set()   # tell crash thread this stop is intentional
        logging.info(f"[PM] Stopping STT (PID {proc.pid})...")
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logging.warning("[PM] STT did not stop in time, killing...")
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass

        # Best-effort: clean up orphaned ffmpeg subprocesses
        try:
            if IS_WINDOWS:
                subprocess.run(
                    ["taskkill", "/F", "/IM", "ffmpeg.exe"],
                    capture_output=True, timeout=5,
                )
            else:
                subprocess.run(
                    ["pkill", "-f", r"ffmpeg.*pipe:1"],
                    capture_output=True, timeout=5,
                )
        except Exception:
            pass

        self.state.set(process=None, status="stopped")
        logging.info("[PM] STT stopped")
        return True

    def is_alive(self):
        with self.state._lock:
            p = self.state.process
        return p is not None and p.poll() is None


# ---------------------------------------------------------------------------
# Crash recovery thread
# ---------------------------------------------------------------------------

class CrashRecoveryThread(threading.Thread):
    def __init__(self, state, pm, no_restart_event, crash_reporter=None):
        super().__init__(daemon=True, name="CrashRecovery")
        self.state = state
        self.pm = pm
        self._no_restart = no_restart_event
        self._crash_reporter = crash_reporter

    def run(self):
        last_proc = None
        while not self.state.get("stop_requested"):
            with self.state._lock:
                proc = self.state.process

            if proc is None or proc is last_proc:
                time.sleep(0.5)
                continue

            last_proc = proc
            start_time = time.monotonic()
            returncode = proc.wait()     # blocks until the process exits
            elapsed = time.monotonic() - start_time

            # Intentional stop (pm.stop() sets _no_restart) or shutdown
            if self._no_restart.is_set() or self.state.get("stop_requested"):
                continue

            # Clean exit — don't restart
            if returncode == 0:
                self.state.set(status="stopped")
                logging.info("[CR] STT exited cleanly (code 0)")
                continue

            # Crash — apply backoff then restart
            if elapsed >= STABLE_RUN_THRESHOLD:
                self.state.set(consecutive_crashes=0)

            with self.state._lock:
                n = self.state.consecutive_crashes + 1
                self.state.consecutive_crashes = n

            delay = BACKOFF[min(n - 1, len(BACKOFF) - 1)]
            logging.warning(
                f"[CR] STT crashed (code {returncode}), "
                f"restarting in {delay}s (crash #{n})"
            )
            self.state.set(status="crashed")
            if self._crash_reporter:
                self._crash_reporter.report(returncode, n)

            deadline = time.monotonic() + delay
            while time.monotonic() < deadline:
                if self._no_restart.is_set() or self.state.get("stop_requested"):
                    break
                time.sleep(0.1)
            else:
                self.pm.start()


# ---------------------------------------------------------------------------
# Auto-updater
# ---------------------------------------------------------------------------

class AutoUpdater:
    def __init__(self, state, pm):
        self.state = state
        self.pm = pm
        self._pending_update = None  # (remote_version, assets) when update detected but not yet applied

    # -- GitHub API ----------------------------------------------------------

    def _api_get(self, url):
        headers = {
            "User-Agent": f"STT-Watchdog/{read_version()}",
            "Accept": "application/vnd.github.v3+json",
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
            return json.loads(resp.read().decode())

    def get_latest_release(self, channel):
        """Return (tag_name, zipball_url, assets) or (None, None, {}) if no releases exist."""
        try:
            if channel == "beta":
                data = self._api_get(f"{GITHUB_API_BASE}/releases")
                if not data:
                    return None, None, {}
                release = data[0]
            else:
                release = self._api_get(f"{GITHUB_API_BASE}/releases/latest")
            assets = {a["name"]: a["browser_download_url"]
                      for a in release.get("assets", [])}
            return release.get("tag_name"), release.get("zipball_url"), assets
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None, None, {}   # no releases yet
            raise

    # -- Check & update ------------------------------------------------------

    def check_for_update(self):
        """Fetch latest release and store it as pending if newer. Does not download or apply."""
        cfg = load_config()
        channel = cfg.get("watchdog", {}).get("update_channel", "stable")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.state.set(last_update_check=now)
        logging.info(f"[AU] Checking for updates (channel: {channel})...")

        try:
            tag, zipball_url, assets = self.get_latest_release(channel)
        except Exception as e:
            result = f"Check failed: {e}"
            logging.warning(f"[AU] {result}")
            self.state.set(last_update_result=result)
            return

        if tag is None:
            self.state.set(last_update_result="No releases yet")
            return

        current = read_version()
        remote = tag.lstrip("v")

        if parse_version(current) >= parse_version(remote):
            self._pending_update = None
            result = f"Up to date ({current})"
            logging.info(f"[AU] {result}")
            self.state.set(last_update_result=result)
            return

        logging.info(f"[AU] Update available: {current} → {remote}")
        self._pending_update = (remote, zipball_url, assets)
        self.state.set(last_update_result=f"Update available: {remote}")

    def apply_pending_update(self):
        """Apply the pending update (if any) via git; fall back to source archive."""
        if not self._pending_update:
            return
        remote, zipball_url, assets = self._pending_update
        self._pending_update = None
        self._apply_git_update(remote, zipball_url)

    def check_and_update(self):
        """Check for update and immediately apply if one is found (manual update / 1am)."""
        self.check_for_update()
        self.apply_pending_update()

    def _git(self, *args, check=True):
        return subprocess.run(["git", "-C", SOURCE_DIR, *args],
                              capture_output=True, text=True,
                              check=check)

    def _apply_git_update(self, remote, zipball_url):
        """Update the managed checkout: git fetch + reset to the release, reinstall
        dependencies (deps may have changed), then restart. Data dir is untouched."""
        have_git = bool(shutil.which("git")) and os.path.isdir(os.path.join(SOURCE_DIR, ".git"))
        if not have_git:
            logging.info("[AU] git unavailable; using source-archive update")
            self._apply_update(remote, zipball_url)
            return

        self.state.set(status="updating")
        self.pm.stop(timeout=20)
        try:
            logging.info(f"[AU] Fetching {remote}...")
            self._git("fetch", "--tags", "--force", "origin", check=False)
            # Reset to the release tag (with or without a leading 'v'); else default branch.
            target = None
            for ref in (f"v{remote}", remote):
                if self._git("rev-parse", "--verify", "--quiet", ref, check=False).returncode == 0:
                    target = ref
                    break
            if target is None:
                # Fall back to the fetched default branch head
                self._git("fetch", "--depth", "1", "origin", check=False)
                target = "origin/HEAD"
            logging.info(f"[AU] Resetting to {target}")
            self._git("reset", "--hard", target)
            self._git("clean", "-fd", check=False)

            logging.info("[AU] Reinstalling dependencies...")
            Provisioner(log=lambda m: logging.info(f"[AU] {m}")).install_deps_only()

            result = (f"Updated to {read_version()} "
                      f"({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})")
            logging.info(f"[AU] {result}")
            self.state.set(last_update_result=result)
        except Exception as e:
            result = f"Update failed: {e}"
            logging.error(f"[AU] {result}")
            self.state.set(last_update_result=result)
        finally:
            self.pm.start()

    def _apply_update(self, remote, zipball_url):
        # git-less fallback: refresh the SOURCE checkout from a release/source archive.
        # Download BEFORE stopping the app so it keeps running during the transfer
        tmpdir = tempfile.mkdtemp(prefix="stt-update-", dir=DATA_DIR)
        zip_path = os.path.join(tmpdir, "update.zip")
        extract_dir = os.path.join(tmpdir, "extracted")

        try:
            logging.info(f"[AU] Downloading {remote}...")
            headers = {"User-Agent": f"STT-Watchdog/{read_version()}"}
            req = urllib.request.Request(zipball_url, headers=headers)
            with urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as resp:
                with open(zip_path, "wb") as f:
                    shutil.copyfileobj(resp, f)

            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                # Guard against zip-slip: no absolute paths or traversal outside extract_dir
                extract_root = os.path.abspath(extract_dir)
                for member in zf.infolist():
                    target = os.path.abspath(os.path.join(extract_root, member.filename))
                    if not target.startswith(extract_root + os.sep):
                        raise ValueError(f"Unsafe path in update zip: {member.filename}")
                zf.extractall(extract_dir)

            # GitHub source zips contain one top-level dir, e.g. "zitlem-STT-<sha>/"
            dirs = [e for e in os.listdir(extract_dir)
                    if os.path.isdir(os.path.join(extract_dir, e))]
            src_root = os.path.join(extract_dir, dirs[0]) if dirs else extract_dir

        except Exception as e:
            logging.error(f"[AU] Download/extract failed: {e}")
            shutil.rmtree(tmpdir, ignore_errors=True)
            self.state.set(last_update_result=f"Download failed: {e}")
            return

        # Stop the app (pm.stop() sets _no_restart so crash thread stays quiet)
        self.state.set(status="updating")
        self.pm.stop(timeout=20)

        # Stop the app, refresh SOURCE (preserving .venv), reinstall deps, restart.
        self.state.set(status="updating")
        self.pm.stop(timeout=20)
        try:
            failed_items = []
            for item in os.listdir(src_root):
                if item in _UPDATE_PRESERVE:
                    continue
                src = os.path.join(src_root, item)
                dst = os.path.join(SOURCE_DIR, item)
                try:
                    if os.path.isdir(src):
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
                except Exception as e:
                    failed_items.append(item)
                    logging.warning(f"[AU] Failed to copy {item}: {e}")

            if failed_items:
                # Don't record the new version: the install is partial, and writing
                # it would stop the next update cycle from retrying.
                result = f"Update to {remote} incomplete; failed: {', '.join(failed_items)}"
                logging.error(f"[AU] {result}")
                self.state.set(last_update_result=result)
            else:
                logging.info("[AU] Reinstalling dependencies...")
                Provisioner(log=lambda m: logging.info(f"[AU] {m}")).install_deps_only()
                result = (
                    f"Updated to {read_version()} "
                    f"({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})"
                )
                logging.info(f"[AU] {result}")
                self.state.set(last_update_result=result)

        except Exception as e:
            result = f"Apply failed: {e}"
            logging.error(f"[AU] {result}")
            self.state.set(last_update_result=result)

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            self.pm.start()   # clears _no_restart, starts fresh process

    # -- Scheduler -----------------------------------------------------------

    def run_scheduler(self):
        t = threading.Thread(
            target=self._scheduler_loop, daemon=True, name="UpdateScheduler"
        )
        t.start()

    def _scheduler_loop(self):
        while not self.state.get("stop_requested"):
            self.check_for_update()
            if datetime.datetime.now().hour == UPDATE_HOUR and self._pending_update:
                logging.info("[AU] 1am auto-apply triggered")
                self.apply_pending_update()
            # Sleep one hour in 60s increments so stop_requested is checked promptly
            for _ in range(60):
                if self.state.get("stop_requested"):
                    return
                time.sleep(60)


# ---------------------------------------------------------------------------
# Desktop GUI (tkinter — optional; falls back to headless if unavailable)
# ---------------------------------------------------------------------------

class GuiWindow:
    def __init__(self, state, pm, updater, monitoring=False):
        import tkinter as tk
        from tkinter import messagebox
        self._tk = tk
        self._mb = messagebox
        self.state = state
        self.pm = pm
        self.updater = updater
        self._monitoring = monitoring
        self.root = tk.Tk()
        self._transcription_running = False
        self._set_icon()
        self._build_ui()
        self.root.after(500, self._poll)

    def _set_icon(self):
        try:
            icon_path = os.path.join(sys._MEIPASS if _FROZEN else
                                     os.path.dirname(os.path.abspath(__file__)),
                                     "icon.ico")
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except Exception:
            pass

    def _build_ui(self):
        tk = self._tk
        root = self.root
        root.title(f"STT v{read_version()}")
        root.resizable(False, False)
        root.geometry("330x390")

        pad = {"padx": 12, "pady": 4}

        # ── Web UI row ───────────────────────────────────────────────────────
        sf = tk.Frame(root)
        sf.pack(fill="x", **pad)
        tk.Label(sf, text="Web UI:", width=13, anchor="w").pack(side="left")
        self._status_lbl = tk.Label(
            sf, text="● Stopped", fg="red", font=("", 10, "bold")
        )
        self._status_lbl.pack(side="left", expand=True, anchor="w")
        if self._monitoring:
            tk.Label(sf, text="Daemon managed", fg="gray", font=("", 8)).pack(side="right")
            self._toggle_btn = None
        else:
            self._toggle_btn = tk.Button(
                sf, text="Start", width=10, command=self._on_toggle
            )
            self._toggle_btn.pack(side="right")

        # ── Transcription row ────────────────────────────────────────────────
        tf = tk.Frame(root)
        tf.pack(fill="x", **pad)
        tk.Label(tf, text="Transcription:", width=13, anchor="w").pack(side="left")
        self._transcription_lbl = tk.Label(
            tf, text="● Stopped", fg="red", font=("", 10, "bold")
        )
        self._transcription_lbl.pack(side="left", expand=True, anchor="w")
        self._transcription_btn = tk.Button(
            tf, text="Start", width=10, command=self._on_toggle_transcription,
            state="disabled"
        )
        self._transcription_btn.pack(side="right")

        tk.Frame(root, height=1, bg="#cccccc").pack(fill="x", padx=12, pady=4)

        # ── Configuration ───────────────────────────────────────────────────
        tk.Label(root, text="── Configuration ──").pack()
        cf = tk.Frame(root)
        cf.pack(fill="x", **pad)

        tk.Label(cf, text="Port:", width=11, anchor="w").grid(
            row=0, column=0, sticky="w"
        )
        self._port_var = tk.StringVar()
        tk.Entry(cf, textvariable=self._port_var, width=14).grid(
            row=0, column=1, sticky="w"
        )

        tk.Label(cf, text="Password:", width=11, anchor="w").grid(
            row=1, column=0, sticky="w", pady=3
        )
        self._pass_var = tk.StringVar()
        tk.Entry(cf, textvariable=self._pass_var, width=14).grid(
            row=1, column=1, sticky="w"
        )

        tk.Label(cf, text="Channel:", width=11, anchor="w").grid(
            row=2, column=0, sticky="w"
        )
        self._chan_var = tk.StringVar(value="Stable")
        tk.OptionMenu(cf, self._chan_var, "Stable", "Beta").grid(
            row=2, column=1, sticky="w"
        )

        self._reload_config()

        tk.Button(root, text="Save Config", command=self._on_save).pack(pady=2)
        tk.Button(
            root, text="Open Web Interface", command=self._on_open_browser
        ).pack(pady=2)
        self._update_btn = tk.Button(
            root, text="Check for Updates", command=self._on_update
        )
        self._update_btn.pack(pady=2)

        tk.Frame(root, height=1, bg="#cccccc").pack(fill="x", padx=12, pady=4)

        self._check_lbl = tk.Label(
            root, text="Last check: —", fg="gray", font=("", 8)
        )
        self._check_lbl.pack()
        self._result_lbl = tk.Label(root, text="", fg="gray", font=("", 8))
        self._result_lbl.pack()

    def _reload_config(self):
        cfg = load_config()
        self._port_var.set(str(cfg.get("web_server", {}).get("port", 80)))
        self._pass_var.set(
            cfg.get("web_server", {})
               .get("password_auth", {})
               .get("password", "")
        )
        ch = cfg.get("watchdog", {}).get("update_channel", "stable")
        self._chan_var.set("Beta" if ch == "beta" else "Stable")

    def _poll(self):
        if self._monitoring:
            import urllib.request as _ur
            cfg = load_config()
            port = cfg.get("web_server", {}).get("port", 8080)
            try:
                _ur.urlopen(f"http://127.0.0.1:{port}/api/transcription/status", timeout=1)
                status = "running"
            except Exception:
                status = "stopped"
        else:
            status = self.state.get("status")
        colors = {
            "running": "green",
            "stopped": "red",
            "crashed": "orange",
            "starting": "#aaaa00",
            "updating": "#0066cc",
        }
        self._status_lbl.config(
            text=f"● {status.capitalize()}", fg=colors.get(status, "gray")
        )
        if self._toggle_btn:
            self._toggle_btn.config(
                text="Stop" if status == "running" else "Start"
            )

        # ── Transcription button ─────────────────────────────────────────────
        if status == "running":
            self._transcription_btn.config(state="normal")
            try:
                import urllib.request, json as _json
                cfg = load_config()
                port = cfg.get("web_server", {}).get("port", 8080)
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/transcription/status", timeout=1
                ) as r:
                    data = _json.loads(r.read())
                running = data.get("state", {}).get("running", False)
                self._transcription_lbl.config(
                    text="● Active" if running else "● Stopped",
                    fg="green" if running else "red",
                )
                self._transcription_btn.config(
                    text="Stop" if running else "Start"
                )
                self._transcription_running = running
            except Exception:
                self._transcription_lbl.config(text="● Unknown", fg="gray")
        else:
            self._transcription_btn.config(state="disabled", text="Start")
            self._transcription_lbl.config(text="● Stopped", fg="red")
            self._transcription_running = False

        if not self._monitoring:
            chk = self.state.get("last_update_check")
            res = self.state.get("last_update_result")
            if chk:
                self._check_lbl.config(text=f"Last check: {chk}")
            if res:
                self._result_lbl.config(text=res)
            pending = self.updater._pending_update
            if pending:
                self._update_btn.config(
                    text=f"Update to {pending[0]}", state="normal"
                )
            elif self._update_btn.cget("text") not in ("Checking…", "Updating…"):
                self._update_btn.config(text="Check for Updates", state="normal")

        self.root.after(1000, self._poll)

    def _on_toggle(self):
        if self.pm.is_alive():
            threading.Thread(target=self.pm.stop, daemon=True).start()
        else:
            threading.Thread(target=self.pm.start, daemon=True).start()

    def _on_toggle_transcription(self):
        import urllib.request, urllib.error
        cfg = load_config()
        port = cfg.get("web_server", {}).get("port", 8080)
        action = "stop" if self._transcription_running else "start"
        def _call():
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/api/transcription/{action}",
                    data=b"", method="POST"
                )
                urllib.request.urlopen(req, timeout=5)
            except Exception as e:
                logging.warning(f"[GUI] Transcription {action} failed: {e}")
        threading.Thread(target=_call, daemon=True).start()

    def _on_update(self):
        self._update_btn.config(state="disabled", text="Checking…")
        def _run():
            self.updater.check_and_update()
        threading.Thread(target=_run, daemon=True).start()

    def _on_save(self):
        try:
            port = int(self._port_var.get())
        except ValueError:
            self._mb.showerror("Error", "Port must be a number")
            return
        cfg = load_config()
        if "web_server" not in cfg:
            cfg["web_server"] = {}
        cfg["web_server"]["port"] = port
        if "password_auth" not in cfg["web_server"]:
            cfg["web_server"]["password_auth"] = {}
        cfg["web_server"]["password_auth"]["password"] = self._pass_var.get()
        if "watchdog" not in cfg:
            cfg["watchdog"] = {}
        cfg["watchdog"]["update_channel"] = (
            "beta" if self._chan_var.get() == "Beta" else "stable"
        )
        save_config(cfg)
        self._mb.showinfo(
            "Saved",
            "Configuration saved.\nRestart STT for port changes to take effect.",
        )

    def _on_open_browser(self):
        cfg = load_config()
        port = cfg.get("web_server", {}).get("port", 8080)
        webbrowser.open(f"http://127.0.0.1:{port}")

    def mainloop(self):
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Crash reporting (fully local — nothing is sent off the machine)
# ---------------------------------------------------------------------------

def _crash_fingerprint(log_tail: str) -> str:
    """Stable 12-char hash of the crash signature (traceback, minus line numbers)."""
    lines = log_tail.splitlines()
    sig_lines = []
    for line in lines[-60:]:
        s = line.strip()
        if any(kw in s for kw in ('Error', 'Exception', 'Traceback', 'CRITICAL', 'File "',
                                   'dyld', 'library load', 'system policy')):
            # Strip line numbers so the same bug hashes identically across versions
            sig_lines.append(re.sub(r',\s*line\s+\d+', '', s))
    signature = '\n'.join(sig_lines) if sig_lines else log_tail[-300:]
    return hashlib.sha256(signature.encode('utf-8', errors='replace')).hexdigest()[:12]


class CrashReporter:
    """Writes crash reports to logs/crashes/ as self-contained files. Nothing is
    ever sent off the machine. Reports are deduped by fingerprint (a hash of the
    traceback) with a cooldown so a crash-loop writes one file per distinct bug,
    and the directory is pruned to the most recent _MAX_FILES dumps.
    """

    _STATE_FILE    = os.path.join(LOG_DIR, 'crash_reporter.json')
    _CRASH_DIR     = os.path.join(LOG_DIR, 'crashes')
    _COOLDOWN      = 600    # cooldown per fingerprint (seconds) — collapses crash-loops
    _LOG_LINES     = 400    # lines of stt.log tail to embed (full detail, local only)
    _MAX_FILES     = 50     # keep at most this many crash dumps

    def __init__(self):
        self._state = self._load_state()

    # -- Persistence ---------------------------------------------------------

    def _load_state(self):
        try:
            with open(self._STATE_FILE) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_state(self):
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(self._STATE_FILE, 'w') as f:
                json.dump(self._state, f)
        except Exception:
            pass

    # -- Rate limiting -------------------------------------------------------

    def _is_rate_limited(self, fingerprint: str) -> bool:
        return (time.time() - self._state.get(fingerprint, 0)) < self._COOLDOWN

    def _mark_written(self, fingerprint: str):
        self._state[fingerprint] = time.time()
        cutoff = time.time() - 86400
        self._state = {k: v for k, v in self._state.items() if v > cutoff}
        self._save_state()

    # -- Public API ----------------------------------------------------------

    def report(self, exit_code: int, consecutive_crashes: int):
        """Non-blocking: fire-and-forget daemon thread."""
        cfg = load_config()
        if not cfg.get('crash_reporting', {}).get('enabled', True):
            return
        threading.Thread(
            target=self._write_local,
            args=(exit_code, consecutive_crashes, cfg),
            daemon=True,
            name='CrashReport',
        ).start()

    # -- Internal ------------------------------------------------------------

    def _collect_log(self) -> str:
        # Full detail, unsanitized — the file never leaves this machine.
        try:
            with open(os.path.join(LOG_DIR, 'stt.log'), 'r',
                      encoding='utf-8', errors='replace') as f:
                return ''.join(f.readlines()[-self._LOG_LINES:])
        except Exception:
            return '(stt.log unavailable)'

    def _prune(self):
        """Keep only the most recent _MAX_FILES crash dumps."""
        try:
            files = sorted(
                (os.path.join(self._CRASH_DIR, n) for n in os.listdir(self._CRASH_DIR)
                 if n.startswith('crash_') and n.endswith('.txt')),
                key=os.path.getmtime,
            )
            for old in files[:-self._MAX_FILES]:
                try:
                    os.remove(old)
                except OSError:
                    pass
        except OSError:
            pass

    def _write_local(self, exit_code: int, consecutive_crashes: int, cfg: dict):
        log_tail    = self._collect_log()
        fingerprint = _crash_fingerprint(log_tail)

        if self._is_rate_limited(fingerprint):
            logging.debug(f'[CrashReport] Skipped — rate limited ({fingerprint})')
            return

        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        header = (
            f"STT crash report\n"
            f"timestamp           : {datetime.datetime.now().isoformat(timespec='seconds')}\n"
            f"version             : {read_version()}\n"
            f"platform            : {platform.system()} {platform.release()} {platform.machine()}\n"
            f"python_version      : {platform.python_version()}\n"
            f"exit_code           : {exit_code}\n"
            f"consecutive_crashes : {consecutive_crashes}\n"
            f"fingerprint         : {fingerprint}\n"
            f"gpu_enabled         : {cfg.get('performance', {}).get('use_gpu', False)}\n"
            f"whisper_model       : {cfg.get('model', {}).get('whisper', {}).get('model', 'unknown')}\n"
            f"audio_backend       : {cfg.get('audio', {}).get('backend', 'unknown')}\n"
            f"{'-' * 60}\n"
        )

        try:
            os.makedirs(self._CRASH_DIR, exist_ok=True)
            path = os.path.join(self._CRASH_DIR, f'crash_{ts}_{fingerprint}.txt')
            with open(path, 'w', encoding='utf-8', errors='replace') as f:
                f.write(header)
                f.write(log_tail)
            logging.info(f'[CrashReport] Wrote {path}')
            self._mark_written(fingerprint)
            self._prune()
        except Exception as e:
            logging.warning(f'[CrashReport] Could not write crash dump: {e}')


# ---------------------------------------------------------------------------
# GUI availability detection
# ---------------------------------------------------------------------------

def detect_gui():
    """Return True if a functional tkinter display is available."""
    # Linux requires an X11 or Wayland display; macOS and Windows use native APIs.
    if sys.platform.startswith("linux"):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return False
    try:
        import tkinter as tk
        r = tk.Tk()
        r.destroy()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _run_crash_report_test():
    setup_logging()
    reporter = CrashReporter()
    # Synthesize a crash and write it locally (bypass the cooldown for the test).
    reporter._state = {}
    reporter._write_local(exit_code=1, consecutive_crashes=1, cfg=load_config())
    latest = None
    try:
        files = [os.path.join(reporter._CRASH_DIR, n) for n in os.listdir(reporter._CRASH_DIR)]
        latest = max(files, key=os.path.getmtime) if files else None
    except OSError:
        pass
    if latest:
        print(f"[test] OK — wrote local crash dump: {latest}")
    else:
        print("[test] FAIL — no crash dump was written (check logs/ permissions)")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# First-run setup UI
# ---------------------------------------------------------------------------

def _run_provisioning_headless():
    try:
        Provisioner(log=lambda m: logging.info(f"[SETUP] {m}")).run()
        return True
    except Exception as e:
        logging.error(f"[SETUP] Provisioning failed: {e}")
        return False


class ProvisionWindow:
    """First-run setup window: runs the Provisioner on a worker thread, streams
    its log into a text pane with a step label + progress bar, and offers
    Retry/Cancel on failure. Returns success via mainloop()."""

    def __init__(self):
        import tkinter as tk
        from tkinter import ttk, scrolledtext
        import queue
        self._tk = tk
        self.root = tk.Tk()
        self.root.title(f"STT Setup v{read_version()}")
        self.root.geometry("580x440")
        self._q = queue.Queue()
        self.success = False
        self._set_icon()

        self._step = tk.Label(self.root, text="Preparing setup…", anchor="w",
                              font=("", 11, "bold"))
        self._step.pack(fill="x", padx=12, pady=(12, 4))
        self._bar = ttk.Progressbar(self.root, mode="indeterminate")
        self._bar.pack(fill="x", padx=12)
        self._logw = scrolledtext.ScrolledText(self.root, height=18, state="disabled",
                                                font=("monospace", 9))
        self._logw.pack(fill="both", expand=True, padx=12, pady=8)
        row = tk.Frame(self.root)
        row.pack(fill="x", padx=12, pady=(0, 12))
        self._retry = tk.Button(row, text="Retry", command=self._start, state="disabled")
        self._retry.pack(side="right", padx=4)
        tk.Button(row, text="Cancel", command=self._on_cancel).pack(side="right")

        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.root.after(120, self._pump)
        self._start()

    def _set_icon(self):
        try:
            icon = os.path.join(sys._MEIPASS if _FROZEN else
                                os.path.dirname(os.path.abspath(__file__)), "icon.ico")
            if os.path.exists(icon):
                self.root.iconbitmap(icon)
        except Exception:
            pass

    def _emit(self, m):
        self._q.put(("log", m))

    def _start(self):
        self._retry.config(state="disabled")
        self._bar.start(12)

        def work():
            try:
                Provisioner(log=self._emit).run()
                self._q.put(("done", True))
            except Exception as e:
                self._q.put(("log", f"[ERROR] {e}"))
                self._q.put(("done", False))

        threading.Thread(target=work, daemon=True).start()

    def _append(self, m):
        self._logw.config(state="normal")
        self._logw.insert("end", m + "\n")
        self._logw.see("end")
        self._logw.config(state="disabled")

    def _pump(self):
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "log":
                    self._append(payload)
                    if payload.startswith("[") and "/" in payload[:6]:
                        self._step.config(text=payload.lstrip("[0123456789/] "))
                else:  # done
                    self.success = bool(payload)
                    self._bar.stop()
                    if self.success:
                        self._step.config(text="Setup complete — starting STT…")
                        self.root.after(700, self.root.destroy)
                    else:
                        self._step.config(text="Setup failed — see log below. Retry or Cancel.")
                        self._retry.config(state="normal")
        except Exception:
            pass
        self.root.after(150, self._pump)

    def _on_cancel(self):
        self.success = False
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()
        return self.success


def run_provisioning(use_gui):
    """Provision the runtime. GUI shows a progress window; headless logs to stdout."""
    if use_gui:
        try:
            return ProvisionWindow().mainloop()
        except Exception as e:
            logging.warning(f"[SETUP] GUI unavailable ({e}); running headless setup")
    return _run_provisioning_headless()


def main():
    # freeze_support() is harmless here; the STT worker runs as a real venv
    # script (not via a frozen re-exec), so multiprocessing spawns re-import
    # speech_to_text.py as __main__ natively — no --run-stt shim needed.
    import multiprocessing
    multiprocessing.freeze_support()

    setup_logging()

    parser = argparse.ArgumentParser(description="STT Watchdog")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--gui", action="store_true", help="Force GUI mode")
    grp.add_argument("--headless", action="store_true", help="Force headless mode")
    parser.add_argument(
        "--check-update",
        action="store_true",
        help="Run one update check then exit",
    )
    parser.add_argument(
        "--reprovision",
        action="store_true",
        help="Re-run first-time setup (rebuild venv, refresh source, reinstall deps)",
    )
    parser.add_argument(
        "--channel",
        choices=["stable", "beta"],
        help="Set update channel (saved to config.json)",
    )
    parser.add_argument(
        "--test-crash-report",
        action="store_true",
        help="Write a synthetic local crash dump to logs/crashes/ and exit",
    )
    args = parser.parse_args()

    if args.test_crash_report:
        _run_crash_report_test()
        return

    _will_be_gui = args.gui or (not args.headless and detect_gui())
    _lock = acquire_lock(open_browser_if_taken=_will_be_gui)  # noqa: F841 — keep socket alive

    if _lock is None:
        # Headless daemon is already running — open a monitoring-only GUI.
        try:
            gui = GuiWindow(state=None, pm=None, updater=None, monitoring=True)
            gui.mainloop()
        except Exception:
            cfg = load_config()
            port = cfg.get("web_server", {}).get("port", 8080)
            webbrowser.open(f"http://127.0.0.1:{port}")
        return

    if args.channel:
        cfg = load_config()
        if "watchdog" not in cfg:
            cfg["watchdog"] = {}
        cfg["watchdog"]["update_channel"] = args.channel
        save_config(cfg)
        logging.info(f"[WATCHDOG] Update channel set to: {args.channel}")

    no_restart = threading.Event()
    state = WatchdogState()
    pm = ProcessManager(state, no_restart)
    crash_reporter = CrashReporter()
    crash_thread = CrashRecoveryThread(state, pm, no_restart, crash_reporter)
    updater = AutoUpdater(state, pm)

    if args.check_update:
        updater.check_and_update()
        sys.exit(0)

    def shutdown(signum=None, frame=None):
        logging.info("[WATCHDOG] Shutting down...")
        state.set(stop_requested=True)
        no_restart.set()
        pm.stop(timeout=10)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    if not IS_WINDOWS:
        signal.signal(signal.SIGTERM, shutdown)

    use_gui = args.gui or (not args.headless and detect_gui())

    logging.info(
        f"[WATCHDOG] Starting STT Watchdog v{read_version()} "
        f"({'GUI' if use_gui else 'headless'} mode)"
    )

    # First-run (or repair): provision the local runtime before starting the app.
    if args.reprovision or not is_provisioned():
        logging.info("[WATCHDOG] Runtime not provisioned; running first-time setup...")
        if not run_provisioning(use_gui):
            logging.error("[WATCHDOG] Setup did not complete; exiting.")
            sys.exit(1)

    crash_thread.start()
    updater.run_scheduler()
    pm.start()

    if use_gui:
        try:
            gui = GuiWindow(state, pm, updater)
            gui.mainloop()
            shutdown()   # window closed by user
        except Exception as e:
            logging.error(f"[GUI] Failed to start GUI: {e}. Falling back to headless.")
            _block_until_stopped(state)
    else:
        _block_until_stopped(state)


def _block_until_stopped(state):
    try:
        while not state.get("stop_requested"):
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
