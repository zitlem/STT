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
  python3 watchdog.py --channel stable       # set update channel and run
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
from typing import ClassVar, Optional

try:
    import certifi
    _SSL_CTX: Optional[ssl.SSLContext] = ssl.create_default_context(cafile=certifi.where())
except Exception as e:
    logging.warning(f"[SSL] certifi unavailable ({e}); using system default certificates")
    _SSL_CTX = None

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"

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
    # This file lives in <repo>/stt/ — the repo root is one level up
    DATA_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SOURCE_DIR = DATA_DIR
APP_DIR = DATA_DIR  # config/logs/crash dumps live in DATA
os.makedirs(APP_DIR, exist_ok=True)

GITHUB_REPO = "ChurchPresenter/STT"
GITHUB_REPO_URL = f"https://github.com/{GITHUB_REPO}"
GITHUB_API_BASE = f"https://api.github.com/repos/{GITHUB_REPO}"
VERSION_FILE = os.path.join(SOURCE_DIR, "VERSION")  # git-managed
CONFIG_DIR = os.path.join(DATA_DIR, "config")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
LOG_DIR = os.path.join(DATA_DIR, "logs")
MODELS_DIR = os.path.join(DATA_DIR, "models")  # mirrors speech_to_text.py MODELS_DIR
def migrate_config_layout():
    """One-time layout migration: live config files used to sit in DATA_DIR root.
    speech_to_text.py performs the same move-if-absent migration; whichever
    process starts first wins, the other finds nothing left to move. Called
    from main() (not at import) so merely importing this module has no side
    effects — tests import it."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    for name in ("config.json", "custom_dictionary.json", "word_highlighting.json",
                 "whisper_models.json", "faster_whisper_models.json"):
        old = os.path.join(DATA_DIR, name)
        new = os.path.join(CONFIG_DIR, name)
        if os.path.exists(old) and not os.path.exists(new):
            try:
                shutil.move(old, new)
                print(f"[MIGRATE] {name} -> config/{name}")
            except OSError as e:
                print(f"[MIGRATE] Could not move {name}: {e}")

# The worker always runs as a real script from the venv (frozen or dev).
STT_SCRIPT = os.path.join(SOURCE_DIR, "speech_to_text.py")
# The watchdog's own source (in the git-managed checkout). A git pull advances
# this; a frozen binary hands off to it so watchdog updates take effect too.
WATCHDOG_SCRIPT = os.path.join(SOURCE_DIR, "stt", "watchdog.py")

# uv-provisioned Python for the venv (see requirements.txt / CI).
UV_PYTHON_VERSION = "3.11"
# Fallback store for uv-managed Pythons. uv's default (%APPDATA%\uv\python on
# Windows) can sit behind a user-created junction — OneDrive folder redirection,
# a relocated profile — that Windows 11 24H2 refuses to let an elevated process
# traverse ("untrusted mount point", os error 448; seen in the field). When
# `uv python install` fails that way, we retry with the store under DATA_DIR.
UV_PYTHON_FALLBACK_DIR = os.path.join(DATA_DIR, "uv-python")

# Port used only for single-instance lock (never serves traffic)
_LOCK_PORT = 57337
# Separate lock so only one desktop control window opens at a time (57337 guards
# the daemon; monitors don't take it, so they need their own guard).
_GUI_LOCK_PORT = 57339

# Control channel so a client (monitor) window can drive the headless daemon's
# updater: the client drops a one-word command; the daemon publishes status.
WD_CMD_FILE = os.path.join(DATA_DIR, ".wd-cmd")
WD_STATUS_FILE = os.path.join(DATA_DIR, ".wd-status.json")

# SOURCE git commit this process launched with; set in main(). A later divergence
# from the current source HEAD means a restart would apply a pulled update.
_LAUNCH_HEAD = None

BACKOFF = [5, 10, 30, 60]       # seconds between crash restarts; capped at last entry
STABLE_RUN_THRESHOLD = 30        # seconds of uptime before resetting crash counter
UPDATE_HOUR = 1                  # hour (24h) at which daily update check fires

# Files/dirs inside SOURCE never discarded by the zipball-fallback update path.
# "config" holds gitignored live user settings next to tracked *.default.json
# templates — replacing it with the zipball's copy (defaults only) would drop
# the user's settings. Zip-updated installs keep stale templates as the price;
# git-based updates refresh templates while leaving the gitignored files alone.
_UPDATE_PRESERVE = frozenset({".venv", "config"})


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging():
    from logging.handlers import RotatingFileHandler
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers: "list[logging.Handler]" = [logging.StreamHandler(sys.stdout)]
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


def _source_head():
    """Short git HEAD of the SOURCE checkout, or '' if unavailable.

    Used to tell whether an auto-update has advanced the on-disk watchdog code
    past what the running process launched with (i.e. a restart would apply it).
    """
    if not (shutil.which("git") and os.path.isdir(os.path.join(SOURCE_DIR, ".git"))):
        return ""
    try:
        r = subprocess.run(["git", "-C", SOURCE_DIR, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _relaunch_watchdog(pm=None):
    """Restart the watchdog by re-running the current launcher. Never returns.

    Re-execs the *same* entry point: a frozen .app re-runs the app (so macOS
    keeps its identity/icon), whose main() then loads the latest source
    in-process; a source install re-runs its script. Clearing STT_WD_FROMSOURCE
    makes the fresh run pick up any newer pulled source. Stops the managed
    server first so it isn't orphaned; the fresh watchdog starts its own.
    """
    if pm is not None:
        try:
            pm.stop(timeout=20)  # also sets no_restart so the crash thread stays quiet
        except Exception as e:
            logging.warning(f"[WATCHDOG] Could not stop STT before relaunch: {e}")
    os.environ.pop("STT_WD_FROMSOURCE", None)  # let the fresh run re-load the latest source
    if _FROZEN:
        exe, argv = sys.argv[0], list(sys.argv)      # re-run the .app itself
    else:
        exe, argv = sys.executable, [sys.executable, *sys.argv]  # python + script + flags
    logging.info(f"[WATCHDOG] Relaunching: {' '.join(argv)}")
    if IS_WINDOWS:
        # os.execv is unreliable under a frozen build on Windows; spawn + exit.
        subprocess.Popen(argv, close_fds=False)
        os._exit(0)
    else:
        os.execv(exe, argv)


def _maybe_handoff_to_source(args):
    """Frozen binary: run the pulled SOURCE watchdog *in-process* so updates apply.

    Once provisioned, the frozen bootstrapper loads SOURCE_DIR/stt/watchdog.py
    (the git-updated code) and runs its main() within this very process, instead
    of exec-ing into the venv python. Staying inside the .app keeps its identity
    and Dock icon (and its bundled Tcl/Tk), while still running the latest
    watchdog code — for the daemon and the GUI window alike. The watchdog only
    needs stdlib + tkinter + certifi + sentry, all bundled, so no venv is needed
    (only the server subprocess uses the venv). Source installs run directly.

    Guarded against loops (STT_WD_FROMSOURCE) and broken pulls (py_compile); any
    load/run failure falls back to the bundled code so a bad pull can't brick it.
    """
    if not _FROZEN or os.environ.get("STT_WD_FROMSOURCE"):
        return
    if not is_provisioned():
        return  # first run: no source yet — provision as the frozen binary first
    if not os.path.isfile(WATCHDOG_SCRIPT):
        return
    # Broken-pull guard: don't run source that won't even compile.
    try:
        import py_compile
        py_compile.compile(WATCHDOG_SCRIPT, doraise=True)
    except Exception as e:
        logging.warning(f"[WATCHDOG] Source watchdog won't compile ({e}); staying on bundled code")
        return
    os.environ["STT_WD_FROMSOURCE"] = "1"  # so the source's own hand-off check no-ops (no re-entry)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("stt_watchdog_source", WATCHDOG_SCRIPT)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {WATCHDOG_SCRIPT}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        logging.info("[WATCHDOG] Running source watchdog in-process (keeps app identity/icon)")
        mod.main()
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        logging.error(f"[WATCHDOG] Source watchdog failed in-process ({e}); using bundled code")
        os.environ.pop("STT_WD_FROMSOURCE", None)
        return  # fall through: frozen main() continues with the bundled code


def _sync_source_to_bundle():
    """Bring the source checkout up to the installed app's version.

    A newer .app installed over an older source checkout leaves the running
    ('run') version behind the app ('app') version until the scheduled/manual
    auto-update catches up. When the frozen bundle is newer than the source,
    fast-forward the source to the bundle's release tag (and reinstall deps,
    which may have changed) so a fresh install is immediately in sync. Frozen +
    provisioned only; run it before handing off so the newer code is loaded.
    """
    if not _FROZEN or os.environ.get("STT_WD_FROMSOURCE"):
        return
    if not (shutil.which("git") and os.path.isdir(os.path.join(SOURCE_DIR, ".git"))):
        return
    try:
        bundle = read_bundle_version()
        source = read_version()
        if parse_version(source) >= parse_version(bundle):
            return  # source already at least as new as the app
        logging.info(f"[WATCHDOG] Source ({source}) is behind the app ({bundle}); syncing…")
        subprocess.run(["git", "-C", SOURCE_DIR, "fetch", "--depth", "1", "--force",
                        "origin", "+refs/tags/*:refs/tags/*"],
                       capture_output=True, timeout=120)
        target = None
        for ref in (f"v{bundle}", bundle):
            if subprocess.run(["git", "-C", SOURCE_DIR, "rev-parse", "--verify", "--quiet", ref],
                              capture_output=True).returncode == 0:
                target = ref
                break
        if not target:
            logging.warning(f"[WATCHDOG] No tag for {bundle}; leaving auto-update to catch up")
            return
        subprocess.run(["git", "-C", SOURCE_DIR, "reset", "--hard", target],
                       capture_output=True, timeout=60)
        subprocess.run(["git", "-C", SOURCE_DIR, "clean", "-fd"], capture_output=True, timeout=60)
        try:
            Provisioner(log=lambda m: logging.info(f"[SYNC] {m}")).install_deps_only()
        except Exception as e:
            logging.warning(f"[WATCHDOG] Dep sync after source bump failed: {e}")
        logging.info(f"[WATCHDOG] Source synced to {read_version()}")
    except Exception as e:
        logging.warning(f"[WATCHDOG] Source sync skipped: {e}")


def _quit_watchdog(pm=None):
    """Fully stop STT: stop the managed server, and on macOS boot the LaunchAgents
    out so KeepAlive doesn't relaunch the daemon. Never returns.

    macOS: the daemon is a per-user LaunchAgent (KeepAlive=true) — a bare exit
    would be relaunched, so `launchctl bootout` the daemon and the control-window
    agent (which also closes any monitor window). Linux/Windows: the supervisor
    doesn't restart a clean exit, so stopping + exiting is enough."""
    try:
        if pm is not None:
            pm.stop(timeout=15)
    except Exception as e:
        logging.warning(f"[WATCHDOG] Error stopping STT during quit: {e}")
    if IS_MACOS:
        uid = str(os.getuid())
        agents = os.path.join(os.path.expanduser("~"), "Library", "LaunchAgents")
        for label in ("com.stt.gui", "com.stt.watchdog"):
            plist = os.path.join(agents, f"{label}.plist")
            try:
                subprocess.run(["launchctl", "bootout", f"gui/{uid}", plist],
                               capture_output=True, timeout=10)
            except Exception:
                pass
    logging.info("[WATCHDOG] Quit requested — STT stopped.")
    os._exit(0)


def _write_wd_status(state, updater):
    """Publish the daemon's update status so a client (monitor) window — which
    has no access to the daemon's in-memory state — can display it."""
    try:
        # A restart is needed when the daemon's running code is behind the pulled
        # source (its launch commit != the current source HEAD).
        cur = _source_head()
        restart_needed = bool(_LAUNCH_HEAD) and bool(cur) and cur != _LAUNCH_HEAD
        data = {
            "last_update_check": state.get("last_update_check"),
            "last_update_result": state.get("last_update_result"),
            "pending": bool(getattr(updater, "_pending_update", None)),
            "status": state.get("status"),  # process-level: so a client can show Starting/Running/…
            "restart_needed": restart_needed,
        }
        tmp = WD_STATUS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, WD_STATUS_FILE)
    except Exception:
        pass


def _watchdog_control_loop(state, pm, updater):
    """Let a client (monitor) window drive this daemon's updater.

    The client can't call the updater directly (separate process), so it drops a
    one-word command file; here we execute it and publish status back. Runs only
    in the owning daemon (GUI or headless), never in a monitor window."""
    busy = threading.Event()

    def _run_check():
        busy.set()
        try:
            updater.check_for_update()
        finally:
            busy.clear()

    def _run_update():
        busy.set()
        try:
            updater.check_for_update()
            updater.apply_pending_update()  # stops the server, applies, restarts
        finally:
            busy.clear()

    try:
        if os.path.exists(WD_CMD_FILE):
            os.remove(WD_CMD_FILE)  # discard a stale command from a previous run
    except OSError:
        pass

    while not state.get("stop_requested"):
        _write_wd_status(state, updater)
        try:
            if os.path.exists(WD_CMD_FILE):
                with open(WD_CMD_FILE, encoding="utf-8") as f:
                    cmd = f.read().strip()
                os.remove(WD_CMD_FILE)
                if cmd == "restart":
                    _relaunch_watchdog(pm)  # never returns
                elif cmd == "quit":
                    _quit_watchdog(pm)  # never returns
                elif not busy.is_set():
                    if cmd == "check":
                        threading.Thread(target=_run_check, daemon=True).start()
                    elif cmd == "update":
                        threading.Thread(target=_run_update, daemon=True).start()
        except Exception:
            pass
        time.sleep(2)


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
    # The release tag the checkout is pinned to is the truth; the VERSION file
    # in historic tags can lag behind the tag itself. Fall back to
    # SOURCE/VERSION, then the bundled VERSION inside the frozen bootstrapper
    # before source is cloned.
    if shutil.which("git") and os.path.isdir(os.path.join(SOURCE_DIR, ".git")):
        try:
            r = subprocess.run(
                ["git", "-C", SOURCE_DIR, "describe", "--tags", "--exact-match", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
            tag = r.stdout.strip().lstrip("v") if r.returncode == 0 else ""
            if tag:
                return tag
        except (OSError, subprocess.SubprocessError):
            pass
    candidates = [VERSION_FILE]
    if _FROZEN:
        candidates.append(os.path.join(sys._MEIPASS, "VERSION"))  # type: ignore[attr-defined]
    for path in candidates:
        try:
            with open(path) as f:
                v = f.read().strip()
            if v:
                return v
        except (FileNotFoundError, OSError):
            pass
    return "0.0.0"


def read_display_version():
    """Human version string for UI display, matching the server's format.

    Folds git describe's commits-since-tag count into the patch number
    ('26.1.2-17-g398f75e' -> '26.1.19-398f75e'); an exact tag shows as-is.
    The 'g' git describe prefixes onto the hash is stripped for display.
    Falls back to read_version() when the checkout has no git metadata.
    Display only — update comparisons must keep using read_version().
    """
    if shutil.which("git") and os.path.isdir(os.path.join(SOURCE_DIR, ".git")):
        try:
            r = subprocess.run(
                ["git", "-C", SOURCE_DIR, "describe", "--tags", "--always"],
                capture_output=True, text=True, timeout=10,
            )
            desc = r.stdout.strip().lstrip("v") if r.returncode == 0 else ""
            if desc:
                m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)-(\d+)-g([0-9a-f]+)", desc)
                if m:
                    return f"{m.group(1)}.{m.group(2)}.{int(m.group(3)) + int(m.group(4))}-{m.group(5)}"
                return desc
        except (OSError, subprocess.SubprocessError):
            pass
    return read_version()


def read_bundle_version():
    """Version of the installed .app/.exe itself (frozen build), matching the
    macOS 'About STT' / Info.plist CFBundleShortVersionString. Distinct from
    read_version(), which tracks the auto-updated SOURCE checkout that git pull
    advances. In dev-from-repo the two coincide.
    """
    if _FROZEN:
        try:
            with open(os.path.join(sys._MEIPASS, "VERSION")) as f:  # type: ignore[attr-defined] # noqa: SLF001
                v = f.read().strip()
            if v:
                return v
        except (OSError, AttributeError):
            pass
    return read_version()


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


def acquire_gui_lock():
    """Single-control-window guard.

    Only one desktop control window should exist at a time. On macOS several
    launches can race in (the com.stt.gui LaunchAgent, the post-install open,
    a user re-opening the app) — each would otherwise pop its own window.
    Binding a dedicated loopback port lets the first control window win and
    later ones detect it and exit. Returns the socket (keep alive) or None if a
    control window is already up. Separate from the daemon lock (57337), which
    monitors deliberately don't take."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    try:
        sock.bind(("127.0.0.1", _GUI_LOCK_PORT))
    except OSError:
        return None
    return sock


# ---------------------------------------------------------------------------
# First-run provisioning (thin bootstrapper)
# ---------------------------------------------------------------------------

PROVISION_MARKER = os.path.join(DATA_DIR, ".provisioned")
_FFMPEG_BIN_DIR = os.path.join(DATA_DIR, "bin")


class ProvisionError(Exception):
    pass


def is_provisioned():
    """True when SOURCE is checked out with a venv the worker can run from.

    Checks the venv interpreter *file* exists rather than 'get_python_bin() !=
    sys.executable' — that comparison is False whenever this process already is
    the venv python (i.e. after the hand-off), which made a handed-off watchdog
    re-run first-time setup on every launch.
    """
    have_source = os.path.isdir(os.path.join(SOURCE_DIR, ".git")) or os.path.exists(PROVISION_MARKER)
    return (
        have_source
        and os.path.isfile(STT_SCRIPT)
        and os.path.isfile(venv_python())
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


def _git_usable():
    """True when a real git is on PATH. On macOS without the Command Line
    Tools, /usr/bin/git is an xcode-select shim that pops Apple's GUI install
    dialog and exits 1 on first use — treat that shim as no git so provisioning
    falls back to the archive download instead of failing."""
    git = _which("git")
    if not git:
        return False
    if sys.platform == "darwin" and os.path.realpath(git).startswith("/usr/bin"):
        try:
            return subprocess.run(["xcode-select", "-p"], capture_output=True).returncode == 0
        except OSError:
            return False
    return True


class Provisioner:
    """Builds the local runtime on first launch: uv, Python, git, ffmpeg, source
    checkout, venv, and dependencies. Each step is idempotent and retriable; the
    marker is only written once every step has succeeded. `log` is a callback
    (message) -> None used by both the GUI pane and headless logging."""

    # Ordered (label, method-name) — drives the "Step k/N" display.
    STEPS: ClassVar[list] = [
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
        # Extra env for uv python/venv steps; set when the managed-Python store
        # is relocated to UV_PYTHON_FALLBACK_DIR (see _step_python).
        self._uv_env = None

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
                creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0,  # type: ignore[attr-defined]
            )
        except FileNotFoundError as e:
            if check:
                raise ProvisionError(f"{cmd[0]} not found: {e}") from e
            return 1
        tail = []
        assert proc.stdout is not None  # stdout=PIPE above
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                self.log("    " + line)
                tail.append(line)
                if len(tail) > 5:
                    tail.pop(0)
        code = proc.wait(timeout=timeout)
        if check and code != 0:
            # Include the last output lines so remote crash reports carry the
            # actual error, not just the exit code.
            detail = f" — last output: {' | '.join(tail)}" if tail else ""
            raise ProvisionError(f"command failed ({code}): {' '.join(str(c) for c in cmd)}{detail}")
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
            # Force TLS 1.2 — PowerShell on stock Windows may still default to
            # TLS 1.0, which astral.sh rejects (observed in the field).
            ps = ("$ErrorActionPreference='Stop';"
                  "[Net.ServicePointManager]::SecurityProtocol="
                  "[Net.ServicePointManager]::SecurityProtocol -bor 3072;"
                  "irm https://astral.sh/uv/install.ps1 | iex")
            code = self._run(["powershell", "-NoProfile", "-Command", ps],
                             desc="install uv (astral.sh)", check=False)
            if code != 0 or not _which("uv"):
                # Installer script blocked (proxy/AV/TLS)? Fetch the official
                # zip directly over our certifi-backed urllib instead.
                self.log("  installer script failed; downloading uv zip from GitHub...")
                arch = "aarch64" if platform.machine().lower() in ("arm64", "aarch64") else "x86_64"
                url = f"https://github.com/astral-sh/uv/releases/latest/download/uv-{arch}-pc-windows-msvc.zip"
                dest_dir = os.path.join(os.path.expanduser("~"), ".local", "bin")
                os.makedirs(dest_dir, exist_ok=True)
                tmp = os.path.join(tempfile.gettempdir(), "uv-windows.zip")
                self._download_file(url, tmp)
                with zipfile.ZipFile(tmp) as zf:
                    zf.extractall(dest_dir)
                os.remove(tmp)
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
        try:
            self._run([self._uv, "python", "install", UV_PYTHON_VERSION],
                      desc=f"uv python install {UV_PYTHON_VERSION}")
        except ProvisionError as e:
            msg = str(e)
            if "untrusted mount point" not in msg and "os error 448" not in msg:
                raise
            self.log("  default uv Python dir is behind an untrusted mount point; "
                     f"retrying with the store under {UV_PYTHON_FALLBACK_DIR}")
            self._uv_env = {"UV_PYTHON_INSTALL_DIR": UV_PYTHON_FALLBACK_DIR}
            self._run([self._uv, "python", "install", UV_PYTHON_VERSION],
                      desc=f"uv python install {UV_PYTHON_VERSION} (fallback dir)",
                      extra_env=self._uv_env)

    def _step_git(self):
        if _git_usable():
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
        if not _git_usable():
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
            # Valid checkout — leave it on whatever ref the auto-updater pinned;
            # resetting to origin/HEAD here would drag releases back to main.
            self.log("  source checkout present")
            return
        os.makedirs(os.path.dirname(SOURCE_DIR) or DATA_DIR, exist_ok=True)
        if _git_usable():
            if os.path.isdir(SOURCE_DIR) and os.listdir(SOURCE_DIR):
                # Non-git leftovers — clear so clone can proceed (DATA is separate).
                shutil.rmtree(SOURCE_DIR, ignore_errors=True)
            self._run(["git", "clone", "--depth", "1", GITHUB_REPO_URL, SOURCE_DIR],
                      desc=f"git clone {GITHUB_REPO_URL}")
            self._pin_latest_release()
        else:
            self._fetch_source_zipball()
        if not os.path.isfile(STT_SCRIPT):
            raise ProvisionError("source checkout did not produce speech_to_text.py")

    def _pin_latest_release(self):
        """Move a fresh clone from the default branch to the newest release tag so
        installs and auto-updates track the same refs. No tags → stay on main."""
        self._run(["git", "-C", SOURCE_DIR, "fetch", "--depth", "1", "--force",
                   "origin", "+refs/tags/*:refs/tags/*"],
                  desc="git fetch --tags", check=False)
        r = subprocess.run(["git", "-C", SOURCE_DIR, "tag", "--list"],
                           capture_output=True, text=True)
        tags = [t.strip() for t in r.stdout.splitlines() if t.strip()] if r.returncode == 0 else []
        if not tags:
            self.log("  no release tags — staying on default branch")
            return
        latest = max(tags, key=parse_version)
        self.log(f"  pinning to release {latest}")
        self._run(["git", "-C", SOURCE_DIR, "reset", "--hard", latest],
                  desc=f"git reset --hard {latest}", check=False)

    def _latest_release_tag(self):
        """Latest release tag from the GitHub API, or None (offline / no releases)."""
        try:
            data = json.loads(self._download_text(f"{GITHUB_API_BASE}/releases/latest"))
            return data.get("tag_name") or None
        except Exception:
            return None

    def _fetch_source_zipball(self):
        """git-less fallback: download the latest-release source archive (default
        branch when no releases exist)."""
        tag = self._latest_release_tag()
        if tag:
            url = f"{GITHUB_REPO_URL}/archive/refs/tags/{tag}.zip"
        else:
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
        # Skip when the venv interpreter already exists (keying off its file, not
        # '!= sys.executable', which is False when we're running the venv python).
        if os.path.isfile(venv_python(venv_dir)):
            self.log("  venv present")
            return
        # --clear: we only get here when no *valid* venv exists, but a broken or
        # partial .venv dir may still be present (e.g. an interrupted first run).
        # Without --clear, `uv venv` fails with "already exists" (exit 2) and
        # provisioning aborts forever. Matches install.sh's `uv venv --clear`.
        # _uv_env carries UV_PYTHON_INSTALL_DIR when _step_python fell back to
        # the relocated store — uv must look there to find the interpreter.
        self._run([self._uv, "venv", venv_dir, "--clear", "--python", UV_PYTHON_VERSION],
                  desc="uv venv", extra_env=self._uv_env)

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
            cmd += ["--extra-index-url", "https://download.pytorch.org/whl/cu128"]
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
            env["STT_MANAGED"] = "1"  # tells the server it's watchdog-managed (skip its own git self-update)
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
                creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0,  # type: ignore[attr-defined]
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

    def _transcription_active(self):
        """True if a transcription is currently running on the local server.

        Used to defer updates until idle. Fails open: if the server is
        unreachable (down means nothing is transcribing) we return False so
        updates are never blocked forever."""
        try:
            cfg = load_config()
            port = cfg.get("web_server", {}).get("port", 8080)
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/transcription/status", timeout=2
            ) as r:
                data = json.loads(r.read())
            return bool(data.get("state", {}).get("running", False))
        except Exception:
            return False

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
        """Return (tag_name, zipball_url, assets) or (None, None, {}) if no releases exist.

        Any non-'main' channel (including a legacy 'beta' left in old configs)
        resolves to the latest full release."""
        try:
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
        """Detect an available update and store it as pending. Does not download or apply.

        Channel 'main' tracks the repo's main branch via git (the default);
        'stable' remains release-pinned via the GitHub Releases API."""
        cfg = load_config()
        channel = cfg.get("watchdog", {}).get("update_channel", "main")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.state.set(last_update_check=now)
        logging.info(f"[AU] Checking for updates (channel: {channel})...")

        if channel == "main":
            self._check_for_branch_update()
            return

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

    _BRANCH_TARGET = "origin/main"  # sentinel remote for branch-tracking updates

    def _check_for_branch_update(self):
        """'main' channel: update available iff local HEAD != origin/main.

        Pure git — no GitHub REST call, so no API quota. Requires the managed
        source to be a git checkout (the normal provisioned state)."""
        if not (shutil.which("git") and os.path.isdir(os.path.join(SOURCE_DIR, ".git"))):
            result = "Main channel needs a git checkout; switch update_channel to 'stable' or reprovision"
            logging.warning(f"[AU] {result}")
            self.state.set(last_update_result=result)
            return
        try:
            # --tags keeps `git describe` (version display) accurate on main.
            self._git("fetch", "--tags", "--force", "origin", "main", check=False)
            local = self._git("rev-parse", "HEAD", check=False).stdout.strip()
            remote = self._git("rev-parse", self._BRANCH_TARGET, check=False).stdout.strip()
        except Exception as e:
            result = f"Check failed: {e}"
            logging.warning(f"[AU] {result}")
            self.state.set(last_update_result=result)
            return
        if not local or not remote:
            result = "Check failed: could not resolve HEAD/origin/main"
            logging.warning(f"[AU] {result}")
            self.state.set(last_update_result=result)
            return
        if local == remote:
            self._pending_update = None
            result = f"Up to date (main @ {local[:9]})"
            logging.info(f"[AU] {result}")
            self.state.set(last_update_result=result)
            return
        logging.info(f"[AU] Update available: main {local[:9]} → {remote[:9]}")
        self._pending_update = (self._BRANCH_TARGET, None, {})
        self.state.set(last_update_result=f"Update available: main @ {remote[:9]}")

    def apply_pending_update(self):
        """Apply the pending update (if any) via git; fall back to source archive."""
        if not self._pending_update:
            return
        remote, zipball_url, _assets = self._pending_update
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
            if remote == self._BRANCH_TARGET:
                # Branch tracking has no archive equivalent (no zipball URL).
                result = "Main-channel update needs a git checkout; skipped"
                logging.warning(f"[AU] {result}")
                self.state.set(last_update_result=result)
                return
            logging.info("[AU] git unavailable; using source-archive update")
            self._apply_update(remote, zipball_url)
            return

        self.state.set(status="updating")
        self.pm.stop(timeout=20)
        try:
            logging.info(f"[AU] Fetching {remote}...")
            self._git("fetch", "--tags", "--force", "origin", check=False)
            if remote == self._BRANCH_TARGET:
                # 'main' channel: track the branch head directly.
                target = self._BRANCH_TARGET
            else:
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
            prev = self._git("rev-parse", "HEAD", check=False).stdout.strip() or None
            logging.info(f"[AU] Resetting to {target}")
            self._git("reset", "--hard", target)
            self._git("clean", "-fd", check=False)

            try:
                logging.info("[AU] Reinstalling dependencies...")
                Provisioner(log=lambda m: logging.info(f"[AU] {m}")).install_deps_only()
            except Exception as e:
                # New source with old/broken deps won't run — put the old source
                # back so the restarted app matches the venv it has.
                if prev:
                    logging.error(f"[AU] Dependency install failed; rolling back to {prev[:12]}")
                    self._git("reset", "--hard", prev, check=False)
                    self._git("clean", "-fd", check=False)
                    try:
                        Provisioner(log=lambda m: logging.info(f"[AU] {m}")).install_deps_only()
                    except Exception as e2:
                        logging.error(f"[AU] Dep reinstall after rollback also failed: {e2}")
                    result = f"Update to {remote} failed (deps); rolled back: {e}"
                else:
                    result = f"Update to {remote} failed (deps): {e}"
                logging.error(f"[AU] {result}")
                self.state.set(last_update_result=result)
                return

            if remote != self._BRANCH_TARGET:
                write_version(remote)
            # Branch mode: leave the VERSION file at the last release so
            # parse_version comparisons stay sane if the install switches back
            # to 'stable'; read_display_version reports the true main state.
            result = (f"Updated to {read_display_version()} "
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

            # GitHub source zips contain one top-level dir, e.g. "ChurchPresenter-STT-<sha>/"
            dirs = [e for e in os.listdir(extract_dir)
                    if os.path.isdir(os.path.join(extract_dir, e))]
            src_root = os.path.join(extract_dir, dirs[0]) if dirs else extract_dir

        except Exception as e:
            logging.error(f"[AU] Download/extract failed: {e}")
            shutil.rmtree(tmpdir, ignore_errors=True)
            self.state.set(last_update_result=f"Download failed: {e}")
            return

        # Stop the app (pm.stop() sets _no_restart so crash thread stays quiet),
        # swap SOURCE items with the staged tree (preserving .venv), reinstall
        # deps, restart. Old items are parked in a backup dir so any failure
        # restores the previous version instead of leaving a mixed tree.
        self.state.set(status="updating")
        self.pm.stop(timeout=20)
        backup_dir = os.path.join(tmpdir, "backup")
        os.makedirs(backup_dir, exist_ok=True)
        moved: "list[tuple[str, bool]]" = []  # (item, had_backup) in swap order

        def _restore():
            for item, had_backup in reversed(moved):
                dst = os.path.join(SOURCE_DIR, item)
                bak = os.path.join(backup_dir, item)
                try:
                    if os.path.lexists(dst):
                        if os.path.isdir(dst) and not os.path.islink(dst):
                            shutil.rmtree(dst, ignore_errors=True)
                        else:
                            os.remove(dst)
                    if had_backup:
                        shutil.move(bak, dst)
                except OSError as re:
                    logging.error(f"[AU] Rollback of {item} failed: {re}")

        try:
            try:
                for item in os.listdir(src_root):
                    if item in _UPDATE_PRESERVE:
                        continue
                    src = os.path.join(src_root, item)
                    dst = os.path.join(SOURCE_DIR, item)
                    had_backup = os.path.lexists(dst)
                    if had_backup:
                        shutil.move(dst, os.path.join(backup_dir, item))
                    moved.append((item, had_backup))
                    shutil.move(src, dst)

                logging.info("[AU] Reinstalling dependencies...")
                Provisioner(log=lambda m: logging.info(f"[AU] {m}")).install_deps_only()
            except Exception:
                _restore()
                raise

            write_version(remote)
            result = (
                f"Updated to {read_version()} "
                f"({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})"
            )
            logging.info(f"[AU] {result}")
            self.state.set(last_update_result=result)

        except Exception as e:
            result = f"Update to {remote} failed; previous version restored: {e}"
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
                if self._transcription_active():
                    logging.info("[AU] Update pending but transcription active — deferring")
                else:
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
        # SOURCE commit at launch; if the checkout later advances (auto-update),
        # a watchdog restart would apply the newer code — surfaced on the button.
        self._launch_head = _source_head()
        self._restart_needed = False  # published by the daemon (monitor mode)
        self._set_icon()
        self._build_ui()
        self.root.after(500, self._poll)

    def _set_icon(self):
        try:
            icon_path = os.path.join(sys._MEIPASS if _FROZEN else  # type: ignore[attr-defined]
                                     os.path.dirname(os.path.abspath(__file__)),
                                     "icon.ico")
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except Exception:
            pass

    def _build_ui(self):
        tk = self._tk
        root = self.root
        root.title(f"STT v{read_display_version()}")
        # Suppress Tk's auto-populated macOS menubar (File/Edit/…): this app
        # uses none of it. An empty menubar clears the stock entries; macOS
        # still provides the standard app menu (and its own Window/Help).
        root.config(menu=tk.Menu(root))
        root.resizable(False, False)
        # Fixed width, but let the height track the content (Tk sizes the window
        # to its packed widgets when no explicit height is forced) — otherwise a
        # hard-coded height clips the lower buttons as sections are added/shown.
        root.minsize(340, 1)

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

        # ── Getting started ─────────────────────────────────────────────────
        # A live checklist of the prerequisites for transcription to work:
        # server running → a model downloaded → a microphone selected. Each row
        # flips to green as its step is done; once all three pass the rows
        # collapse to a single "Ready" line so the checklist gets out of the way.
        tk.Label(root, text="── Getting started ──").pack()
        # Container holds this slot in the root's pack order; _refresh_checklist
        # swaps between the rows and the "Ready" line inside it so collapsing
        # never reorders the surrounding widgets.
        self._checklist_container = tk.Frame(root)
        self._checklist_container.pack(fill="x", padx=12, pady=(0, 2))
        self._checklist_frame = tk.Frame(self._checklist_container)
        self._checklist_frame.pack(fill="x")
        self._ready_lbl = tk.Label(
            self._checklist_container,
            text="✓ Ready — press Start under Transcription",
            fg="green", font=("", 9, "bold"),
        )  # packed/unpacked by _refresh_checklist when all steps pass

        def _step_row(text, action=None):
            row = tk.Frame(self._checklist_frame)
            row.pack(fill="x", pady=1)
            mark = tk.Label(row, text="☐", fg="red", width=2, font=("", 11))
            mark.pack(side="left")
            tk.Label(row, text=text, anchor="w").pack(side="left")
            if action:
                label, path = action
                tk.Button(
                    row, text=label, font=("", 8),
                    command=lambda p=path: self._on_open_browser(p),
                ).pack(side="right")
            return mark

        self._chk_server = _step_row("Start the web server")
        self._chk_model = _step_row("Download & select a model (faster-whisper recommended)", ("Model Manager", "/model-manager"))
        self._chk_mic = _step_row("Select a microphone", ("Settings", "/live-settings"))
        # Cache the disk/config-derived checks so _poll doesn't re-listdir every
        # second; recomputed on a counter tick or when config.json changes.
        self._checklist_tick = 0
        self._checklist_cfg_mtime = None
        self._model_ready = False
        self._mic_ready = False

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
        tk.Entry(cf, textvariable=self._pass_var, width=14, show="•").grid(
            row=1, column=1, sticky="w"
        )

        tk.Label(cf, text="Channel:", width=11, anchor="w").grid(
            row=2, column=0, sticky="w"
        )
        self._chan_var = tk.StringVar(value="Main")
        tk.OptionMenu(cf, self._chan_var, "Main", "Stable").grid(
            row=2, column=1, sticky="w"
        )

        # Microphone: populated from the running server's /api/audio-devices;
        # saving on change goes through /api/config (sets the flag + hot-reloads).
        tk.Label(cf, text="Microphone:", width=11, anchor="w").grid(
            row=3, column=0, sticky="w", pady=3
        )
        self._mic_var = tk.StringVar(value="— start server —")
        self._mic_map = {}  # display label -> device_id
        self._mic_menu = tk.OptionMenu(cf, self._mic_var, self._mic_var.get())
        self._mic_menu.config(width=13, font=("", 9))
        self._mic_menu.grid(row=3, column=1, sticky="w")

        # Load current values without triggering the auto-save trace below.
        self._suppress_autosave = True
        self._reload_config()
        self._suppress_autosave = False

        # Auto-save whenever any field changes — no Save button needed.
        for _var in (self._port_var, self._pass_var, self._chan_var):
            _var.trace_add("write", self._on_config_change)

        tk.Label(
            root,
            text="Changes save automatically · port applies after restart",
            fg="gray",
            font=("", 8),
        ).pack(pady=(2, 0))
        bf = tk.Frame(root)
        bf.pack(pady=2)
        tk.Button(
            bf, text="Main Page", command=lambda: self._on_open_browser("/")
        ).pack(side="left", padx=2)
        tk.Button(
            bf, text="Settings", command=lambda: self._on_open_browser("/live-settings")
        ).pack(side="left", padx=2)
        tk.Button(
            bf, text="URL Builder", command=lambda: self._on_open_browser("/url-builder")
        ).pack(side="left", padx=2)
        # Two manual update actions: "Check for Updates" reports availability
        # without touching anything; "Update Now" applies it (idle-gated). A
        # client (monitor) window has no updater of its own, so its buttons
        # command the background daemon over the control-file channel instead.
        uf = tk.Frame(root)
        uf.pack(pady=2)
        self._check_btn = tk.Button(
            uf, text="Check for Updates", command=self._on_check_update
        )
        self._check_btn.pack(side="left", padx=2)
        self._update_now_btn = tk.Button(
            uf, text="Update Now", command=self._on_update_now
        )
        self._update_now_btn.pack(side="left", padx=2)

        # "Restart" applies a pulled update (reloads the watchdog + server, and
        # this window, from source). Hidden until a restart is actually needed;
        # the container holds its slot so showing it doesn't shift other widgets.
        self._restart_container = tk.Frame(root)
        self._restart_container.pack(fill="x")
        self._restart_wd_btn = tk.Button(
            self._restart_container, text="Restart", command=self._on_restart_watchdog
        )  # packed on demand by _refresh_checklist

        # Fully stop STT (server + background daemon). Closing the window only
        # hides it; this is how a user quits when it runs as a daemon.
        tk.Button(
            root, text="Quit STT", fg="#b00020", command=self._on_quit_stt
        ).pack(pady=(0, 2))

        tk.Frame(root, height=1, bg="#cccccc").pack(fill="x", padx=12, pady=4)

        _footer = tk.Frame(root)
        _footer.pack(fill="x", padx=12)
        self._check_lbl = tk.Label(
            _footer, text="Last check: —", fg="gray", font=("", 8)
        )
        self._check_lbl.pack(side="left")
        # Show both the installed app version and the running (auto-updated) code
        # version. They diverge once git-pull updates advance the source checkout
        # ahead of the installed .app/.exe; collapse to one when they match.
        _installed = read_bundle_version()
        _running = read_display_version()
        _ver_text = f"v{_running}" if _running == _installed else f"app v{_installed} · run v{_running}"
        tk.Label(
            _footer, text=_ver_text, fg="gray", font=("", 8)
        ).pack(side="right")
        self._result_lbl = tk.Label(root, text="", fg="gray", font=("", 9))
        self._result_lbl.pack()

    @staticmethod
    def _selected_model_downloaded(cfg):
        """True when the *currently selected* transcription model is on disk.

        Mirrors _selected_model_downloaded / ModelFactory._load_* in
        speech_to_text.py (the watchdog can't import the server), so this is
        'downloaded AND set' — a different downloaded model doesn't satisfy it.
        """
        model_cfg = cfg.get("model", {})
        mtype = model_cfg.get("type", "whisper")
        if mtype == "whisper":
            name = model_cfg.get("whisper", {}).get("model", "small")
            if model_cfg.get("backend", "whisper") == "faster-whisper":
                return os.path.isdir(os.path.join(MODELS_DIR, f"faster-whisper-{name}"))
            if os.path.isdir(os.path.join(MODELS_DIR, f"whisper-{name}")):
                return True
            return os.path.exists(os.path.expanduser(f"~/.cache/whisper/{name}.pt"))
        if mtype == "huggingface":
            model_id = model_cfg.get("huggingface", {}).get("model_id", "openai/whisper-tiny")
            return os.path.isdir(os.path.join(MODELS_DIR, model_id.replace("/", "--")))
        if mtype == "custom":
            return os.path.exists(model_cfg.get("custom", {}).get("model_path", ""))
        return False

    @staticmethod
    def _mic_configured(cfg):
        """True when an audio input is configured. The system default ('default')
        counts — on a single-mic machine it's the only option, so requiring an
        explicit non-default pick was a dead-end. Only a truly empty value reads
        as 'no mic configured'. Mirrors _mic_explicitly_selected in the server.
        """
        audio = cfg.get("audio", {})
        return bool(audio.get("default_microphone")) or \
            bool(audio.get("default_microphone_name")) or \
            bool(audio.get("microphone_selected"))

    def _refresh_checklist(self, status):
        """Update the getting-started rows. Cheap config/HTTP-derived row 1 is
        refreshed every call; the disk/config-derived model + mic checks are
        recomputed only every 5th tick or when config.json changes."""
        self._checklist_tick += 1
        try:
            mtime = os.path.getmtime(CONFIG_FILE)
        except OSError:
            mtime = None
        if (self._checklist_tick % 5 == 1) or mtime != self._checklist_cfg_mtime:
            self._checklist_cfg_mtime = mtime
            cfg = load_config()
            self._model_ready = self._selected_model_downloaded(cfg)
            self._mic_ready = self._mic_configured(cfg)
            # Show the Restart button only when a restart would apply a pulled
            # update, hidden otherwise. Owner: the source advanced past what this
            # process launched with. Client: the daemon says it's behind, or this
            # frozen window is behind the pulled source (a window update waits).
            if self._restart_wd_btn.cget("text") != "Restarting…":
                if self._monitoring:
                    needs_restart = self._restart_needed
                    try:
                        if read_version() != read_bundle_version():
                            needs_restart = True
                    except Exception:
                        pass
                else:
                    cur = _source_head()
                    needs_restart = bool(self._launch_head) and bool(cur) and cur != self._launch_head
                if needs_restart and not self._restart_wd_btn.winfo_manager():
                    self._restart_wd_btn.pack(pady=(0, 2))
                elif not needs_restart and self._restart_wd_btn.winfo_manager():
                    self._restart_wd_btn.pack_forget()

        def _set(mark, ok):
            mark.config(text="☑" if ok else "☐", fg="green" if ok else "red")

        server_ok = status == "running"
        _set(self._chk_server, server_ok)
        _set(self._chk_model, self._model_ready)
        _set(self._chk_mic, self._mic_ready)

        # Grey out the transcription Start button until setup is complete. When
        # transcription is already active, always leave Stop clickable.
        if server_ok and not self._transcription_running:
            self._transcription_btn.config(
                state="normal" if (self._model_ready and self._mic_ready) else "disabled"
            )

        all_ok = server_ok and self._model_ready and self._mic_ready
        if all_ok and self._checklist_frame.winfo_manager():
            self._checklist_frame.pack_forget()
            self._ready_lbl.pack(fill="x")
        elif not all_ok and self._ready_lbl.winfo_manager():
            self._ready_lbl.pack_forget()
            self._checklist_frame.pack(fill="x")

    def _reload_config(self):
        cfg = load_config()
        self._port_var.set(str(cfg.get("web_server", {}).get("port", 80)))
        self._pass_var.set(
            cfg.get("web_server", {})
               .get("password_auth", {})
               .get("password", "")
        )
        ch = cfg.get("watchdog", {}).get("update_channel", "main")
        # Legacy 'beta' configs display (and save back) as Stable.
        self._chan_var.set({"beta": "Stable", "stable": "Stable"}.get(ch, "Main"))

    def _poll(self):
        cfg = load_config()
        port = cfg.get("web_server", {}).get("port", 8080)
        # Is the web server actually serving yet?
        web_ok = False
        try:
            import urllib.request as _ur
            _ur.urlopen(f"http://127.0.0.1:{port}/api/transcription/status", timeout=1)
            web_ok = True
        except Exception:
            web_ok = False
        # Process-level status: the owner reads its own state; a client reads the
        # daemon's published status.
        if self._monitoring:
            proc = None
            try:
                with open(WD_STATUS_FILE, encoding="utf-8") as f:
                    proc = json.load(f).get("status")
            except (OSError, ValueError):
                pass
        else:
            proc = self.state.get("status")
        # Derive the Web UI status, surfacing "starting" — the process is up but
        # the web server hasn't bound yet — between stopped and running.
        if web_ok:
            status = "running"
        elif proc in ("running", "starting"):
            status = "starting"
        elif proc in ("crashed", "updating", "stopped"):
            status = proc
        else:
            status = "stopped"
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

        # Update status: the owner reads its own state; a client (monitor) reads
        # the status file the daemon publishes over the control channel.
        if self._monitoring:
            chk = res = None
            pending = False
            try:
                with open(WD_STATUS_FILE, encoding="utf-8") as f:
                    _st = json.load(f)
                chk = _st.get("last_update_check")
                res = _st.get("last_update_result")
                pending = bool(_st.get("pending"))
                self._restart_needed = bool(_st.get("restart_needed"))
            except (OSError, ValueError):
                pass
        else:
            chk = self.state.get("last_update_check")
            res = self.state.get("last_update_result")
            pending = bool(self.updater._pending_update)
        if chk:
            self._check_lbl.config(text=f"Last check: {chk}")
        if res:
            # Colour the outcome so "up to date" reads as a clear success rather
            # than looking like nothing happened after a check.
            _rl = res.lower()
            if "up to date" in _rl:
                _fg = "green"
            elif "available" in _rl or "updated to" in _rl:
                _fg = "#0066cc"
            elif "fail" in _rl or "error" in _rl:
                _fg = "red"
            else:
                _fg = "gray"
            self._result_lbl.config(text=res, fg=_fg)
        # Update Now is clickable only when an update is actually available
        # (pending, from a manual check or the hourly scheduler); otherwise it's
        # disabled. Left alone while a click is in flight.
        if self._update_now_btn.cget("text") not in ("Updating…",):
            self._update_now_btn.config(
                text="Update Now ●" if pending else "Update Now",
                state="normal" if pending else "disabled",
            )

        self._refresh_checklist(status)

        # Populate the microphone dropdown once the server is up (it enumerates
        # devices via ffmpeg, so fetch only until we have the list).
        if status == "running" and not self._mic_map:
            self._refresh_mic_options()

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

    def _refresh_mic_options(self):
        """Fetch audio devices from the running server and fill the dropdown.
        Runs off the Tk thread (ffmpeg enumeration is slow); the menu is rebuilt
        back on the main thread via root.after."""
        if getattr(self, "_mic_fetching", False):
            return
        self._mic_fetching = True
        port = load_config().get("web_server", {}).get("port", 8080)
        def _work():
            labels, mapping, current = [], {}, "default"
            try:
                import urllib.request, json as _json
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/audio-devices", timeout=3
                ) as r:
                    data = _json.loads(r.read())
                for d in (data.get("devices") or []):
                    name = d.get("name") or d.get("device_id") or "?"
                    label = f"{name} (Default)" if d.get("is_default") else name
                    mapping[label] = d.get("device_id") or "default"
                    labels.append(label)
                current = load_config().get("audio", {}).get("default_microphone", "default")
            except Exception:
                pass
            finally:
                self._mic_fetching = False
            if labels:
                self.root.after(0, lambda: self._apply_mic_options(labels, mapping, current))
        threading.Thread(target=_work, daemon=True).start()

    def _apply_mic_options(self, labels, mapping, current):
        self._mic_map = mapping
        menu = self._mic_menu["menu"]
        menu.delete(0, "end")
        for label in labels:
            menu.add_command(label=label, command=lambda l=label: self._on_mic_change(l))
        sel = next((l for l, did in mapping.items() if did == current), None) or labels[0]
        self._mic_var.set(sel)  # programmatic set doesn't fire _on_mic_change

    def _on_mic_change(self, label):
        """Save the picked mic via /api/config (sets the flag + hot-reloads)."""
        self._mic_var.set(label)
        device_id = self._mic_map.get(label, "default")
        port = load_config().get("web_server", {}).get("port", 8080)
        def _work():
            try:
                import urllib.request, json as _json
                body = _json.dumps({"audio": {"default_microphone": device_id}}).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/api/config", data=body, method="POST",
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=5)
                logging.info(f"[GUI] Microphone set to {device_id}")
            except Exception as e:
                logging.warning(f"[GUI] Saving microphone failed: {e}")
        threading.Thread(target=_work, daemon=True).start()

    def _send_wd_command(self, cmd):
        """Client (monitor) window: ask the background daemon to run an update
        action, since we have no updater/pm of our own."""
        try:
            with open(WD_CMD_FILE, "w", encoding="utf-8") as f:
                f.write(cmd)
        except OSError as e:
            logging.warning(f"[GUI] Could not send '{cmd}' to daemon: {e}")

    def _on_check_update(self):
        """Check for an available update and report it — never applies."""
        self._check_btn.config(state="disabled", text="Checking…")
        # Immediate acknowledgment; _poll replaces it with the colour-coded
        # outcome (e.g. green "Up to date …") once the check finishes.
        self._result_lbl.config(text="Checking for updates…", fg="gray")
        if self._monitoring:
            self._send_wd_command("check")
            self.root.after(4000, lambda: self._check_btn.config(
                state="normal", text="Check for Updates"))
            return
        def _run():
            try:
                self.updater.check_for_update()  # sets last_update_result + _pending_update
            finally:
                self.root.after(0, lambda: self._check_btn.config(
                    state="normal", text="Check for Updates"))
        threading.Thread(target=_run, daemon=True).start()

    def _on_update_now(self):
        """Check and apply immediately (idle-gated so it never interrupts a run)."""
        # Idle-gate on the client side too: don't ask the daemon to update while
        # a transcription is running.
        if self._transcription_running:
            self._result_lbl.config(text="Transcription running — stop it to update")
            return
        self._update_now_btn.config(state="disabled", text="Updating…")
        if self._monitoring:
            self._send_wd_command("update")
            self.root.after(6000, lambda: self._update_now_btn.config(
                state="normal", text="Update Now"))
            return
        def _run():
            try:
                self.updater.check_for_update()
                self.updater.apply_pending_update()  # may restart the process
            finally:
                self.root.after(0, lambda: self._update_now_btn.config(
                    state="normal", text="Update Now"))
        threading.Thread(target=_run, daemon=True).start()

    def _on_restart_watchdog(self):
        """Restart to apply a pulled update — reloads the watchdog, server, and
        this window from source (keeping the app icon via the in-process load)."""
        if self._transcription_running and not self._mb.askyesno(
            "Restart",
            "This stops transcription briefly and restarts STT to apply the "
            "update. Continue?",
        ):
            return
        self._restart_wd_btn.config(state="disabled", text="Restarting…")
        if self._monitoring:
            # Ask the daemon to restart itself (reloads watchdog + restarts the
            # server from updated source), then re-run this window so its code
            # updates too. Re-exec ends the Tk loop, which is expected.
            self._send_wd_command("restart")
            threading.Thread(target=lambda: _relaunch_watchdog(None), daemon=True).start()
            return
        # Owner window: re-exec restarts the watchdog, server, and this window.
        threading.Thread(target=lambda: _relaunch_watchdog(self.pm), daemon=True).start()

    def _on_quit_stt(self):
        """Fully stop STT (server + background daemon), not just close the window."""
        if not self._mb.askyesno(
            "Quit STT",
            "Stop STT completely?\n\nThis stops transcription and the background "
            "service. Re-open STT from Applications to start it again.",
        ):
            return
        if self._monitoring:
            # The daemon owns the process/LaunchAgents; ask it to quit, then close
            # this client window (the daemon also boots out the control-window agent).
            self._send_wd_command("quit")
            self.root.destroy()
            return
        # Owner window: stop everything from here.
        threading.Thread(target=lambda: _quit_watchdog(self.pm), daemon=True).start()

    def _on_config_change(self, *_args):
        """Trace callback: persist config whenever a field changes."""
        if getattr(self, "_suppress_autosave", False):
            return
        self._persist_config()

    def _persist_config(self):
        try:
            port = int(self._port_var.get())
        except ValueError:
            # Half-typed / empty port entry: skip silently and save once it's a
            # valid number, so auto-save never writes a partial value or nags.
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
            "stable" if self._chan_var.get() == "Stable" else "main"
        )
        save_config(cfg)

    def _on_open_browser(self, path="/"):
        cfg = load_config()
        port = cfg.get("web_server", {}).get("port", 8080)
        webbrowser.open(f"http://127.0.0.1:{port}{path}")

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
        _sentry_capture(e)
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
        self._q: "queue.Queue[tuple]" = queue.Queue()
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
            icon = os.path.join(sys._MEIPASS if _FROZEN else  # type: ignore[attr-defined]
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
                _sentry_capture(e)

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
    grp.add_argument(
        "--monitor",
        action="store_true",
        help="Open the control window only (client of a running daemon); never manages the server",
    )
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
        choices=["main", "stable"],
        help="Set update channel (saved to config.json): main = track latest code, stable = tagged releases",
    )
    parser.add_argument(
        "--test-crash-report",
        action="store_true",
        help="Write a synthetic local crash dump to logs/crashes/ and exit",
    )
    args = parser.parse_args()

    # Headless daemon only (clients would race on git): if a newer .app was
    # installed over an older source checkout, fast-forward the source to the
    # app's version before handing off, so 'run' matches 'app' immediately.
    if not (args.monitor or args.gui or (not args.headless and detect_gui())):
        _sync_source_to_bundle()

    # Frozen binary: run the git-updated source watchdog in-process so watchdog.py
    # updates take effect while keeping the .app identity/icon. No-op for source
    # installs and after a hand-off.
    _maybe_handoff_to_source(args)

    # Record the source commit this (now-final) code launched with, so a later
    # auto-update advancing the checkout surfaces the "Restart" affordance.
    global _LAUNCH_HEAD
    _LAUNCH_HEAD = _source_head()

    if args.test_crash_report:
        _run_crash_report_test()
        return

    if args.monitor:
        # Control-window-only mode: a client of the (separately running) daemon.
        # Never acquires the single-instance lock and never manages the server —
        # closing the window leaves the daemon untouched.
        cfg = load_config()
        if not cfg.get("watchdog", {}).get("show_control_window", True):
            logging.info("[MONITOR] show_control_window is false; not opening window.")
            return
        _gui_lock = acquire_gui_lock()  # noqa: F841 — keep socket alive for the window's lifetime
        if _gui_lock is None:
            logging.info("[MONITOR] A control window is already open; not opening another.")
            return
        try:
            gui = GuiWindow(state=None, pm=None, updater=None, monitoring=True)
            gui.mainloop()
        except Exception as e:
            logging.warning(f"[MONITOR] GUI unavailable ({e}); opening browser instead.")
            port = cfg.get("web_server", {}).get("port", 8080)
            webbrowser.open(f"http://127.0.0.1:{port}")
        return

    _will_be_gui = args.gui or (not args.headless and detect_gui())
    _lock = acquire_lock(open_browser_if_taken=_will_be_gui)  # noqa: F841 — keep socket alive

    if _lock is None:
        # Headless daemon is already running — open a monitoring-only GUI, unless
        # a control window is already up (avoid stacking duplicate windows).
        _gui_lock = acquire_gui_lock()  # noqa: F841 — keep socket alive
        if _gui_lock is None:
            logging.info("[MONITOR] A control window is already open; exiting.")
            return
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
    # Control channel so a client (monitor) window can drive this daemon's
    # updater (Check / Update Now / Restart / Quit) even when we run headless.
    threading.Thread(target=_watchdog_control_loop, args=(state, pm, updater),
                     daemon=True).start()
    pm.start()

    _gui_lock = acquire_gui_lock() if use_gui else None  # noqa: F841 — keep socket alive
    if use_gui and _gui_lock is None:
        # A control window is already open (e.g. a --monitor client); don't open a
        # second one — keep managing the server headlessly.
        logging.info("[WATCHDOG] A control window is already open; running headless.")
        use_gui = False
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


# Project default DSN (ingest-only key), same as speech_to_text.py. Overridable
# via crash_reporting.sentry_dsn; disable with crash_reporting.sentry_enabled=false.
_SENTRY_DEFAULT_DSN = "https://eff01fdec5e9330b80ffd96093038588@o4511050918723584.ingest.us.sentry.io/4511714251702272"


def _sentry_capture(exc):
    """Best-effort capture+flush of a watchdog-side exception (e.g. a
    provisioning failure). Silent no-op when the SDK is absent or Sentry is
    disabled; the flush matters because the headless path exits right after."""
    try:
        import sentry_sdk
        sentry_sdk.capture_exception(exc)
        sentry_sdk.flush(timeout=5)
    except Exception:
        pass


def _init_sentry():
    """Sentry error reporting + logs, on by default. Silent no-op when disabled
    in config or the SDK is absent. The frozen bootstrapper bundles the SDK
    (see packaging/watchdog.spec) so even first-run provisioning failures are
    reported; source runs report once the venv provides sentry-sdk."""
    try:
        cr = load_config().get("crash_reporting", {})
    except Exception:
        cr = {}
    if not cr.get("sentry_enabled", True):
        return
    dsn = (cr.get("sentry_dsn", "") or "").strip() or _SENTRY_DEFAULT_DSN
    try:
        import sentry_sdk
        release = None
        try:
            release = "stt@" + read_version()
        except Exception:
            pass
        sentry_sdk.init(
            dsn=dsn,
            release=release,
            send_default_pii=bool(cr.get("sentry_send_pii", True)),
            # The watchdog logs via the logging module — these become Sentry Logs
            enable_logs=bool(cr.get("sentry_enable_logs", True)),
        )
        sentry_sdk.set_tag("process", "watchdog")
        print("[SENTRY] Error reporting enabled")
    except ImportError:
        pass
    except Exception as e:
        print(f"[SENTRY] Init failed (continuing without): {e}")


if __name__ == "__main__":
    migrate_config_layout()
    _init_sentry()
    main()
