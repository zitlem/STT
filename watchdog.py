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
except Exception:
    _SSL_CTX = None

IS_WINDOWS = sys.platform == "win32"

# ---------------------------------------------------------------------------
# Crash reporting endpoint — set after deploying cloudflare_worker.js
# ---------------------------------------------------------------------------
# The Worker URL and key are baked into every release so all users can
# report without any per-user setup. The GitHub token never leaves Cloudflare.
_CRASH_WORKER_URL = "https://stt-crash-reporter.zitlem-a.workers.dev"
_CRASH_API_KEY    = "stt-crash-v1"  # matched in the Worker; rate limiting is server-side

_FROZEN = getattr(sys, 'frozen', False)
APP_DIR = (
    os.path.join(os.path.expanduser("~"), ".stt") if _FROZEN
    else os.path.dirname(os.path.abspath(__file__))
)
os.makedirs(APP_DIR, exist_ok=True)

GITHUB_REPO = "zitlem/STT"
GITHUB_API_BASE = f"https://api.github.com/repos/{GITHUB_REPO}"
VERSION_FILE = os.path.join(APP_DIR, "VERSION")
CONFIG_FILE = os.path.join(APP_DIR, "config.json")
LOG_DIR = os.path.join(APP_DIR, "logs")

# When frozen the watchdog launches the bundled STT exe (same install dir).
# In dev mode it launches the Python script via the venv.
if _FROZEN:
    _install_dir = os.path.dirname(sys.executable)
    STT_SCRIPT = os.path.join(_install_dir, "STT.exe" if IS_WINDOWS else "STT")
else:
    STT_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speech_to_text.py")

# Port used only for single-instance lock (never serves traffic)
_LOCK_PORT = 57337

BACKOFF = [5, 10, 30, 60]       # seconds between crash restarts; capped at last entry
STABLE_RUN_THRESHOLD = 30        # seconds of uptime before resetting crash counter
UPDATE_HOUR = 1                  # hour (24h) at which daily update check fires

# Files/dirs never overwritten during an update
_UPDATE_PRESERVE = frozenset({
    "config.json",
    "config.json.backup",
    ".venv",
    "logs",
    "VERSION",  # written explicitly via write_version() after copy
})


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging():
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        handlers.insert(0, logging.FileHandler(
            os.path.join(LOG_DIR, "watchdog.log"), encoding="utf-8"
        ))
    except OSError:
        pass
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_python_bin():
    """Return the venv Python binary appropriate for the current OS."""
    if IS_WINDOWS:
        candidates = [
            os.path.join(APP_DIR, ".venv", "Scripts", "python.exe"),
            os.path.join(APP_DIR, ".venv", "Scripts", "python3.exe"),
        ]
    else:
        candidates = [
            os.path.join(APP_DIR, ".venv", "bin", "python3"),
            os.path.join(APP_DIR, ".venv", "bin", "python"),
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
    try:
        with open(VERSION_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
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


def acquire_lock():
    """Single-instance guard: bind a local socket. Released automatically on exit."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    try:
        sock.bind(("127.0.0.1", _LOCK_PORT))
    except OSError:
        print("[ERROR] Another watchdog instance is already running.", file=sys.stderr)
        sys.exit(1)
    return sock  # keep reference alive; OS releases on process exit


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
            self._log_fh = open(log_path, "a", encoding="utf-8")
            cmd = [STT_SCRIPT] if _FROZEN else [self._python, STT_SCRIPT]
            proc = subprocess.Popen(
                cmd,
                cwd=APP_DIR,
                stdout=self._log_fh,
                stderr=self._log_fh,
                close_fds=not IS_WINDOWS,
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
        """Return (tag_name, zipball_url) or (None, None) if no releases exist."""
        try:
            if channel == "beta":
                data = self._api_get(f"{GITHUB_API_BASE}/releases")
                if not data:
                    return None, None
                release = data[0]
            else:
                release = self._api_get(f"{GITHUB_API_BASE}/releases/latest")
            return release.get("tag_name"), release.get("zipball_url")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None, None   # no releases yet
            raise

    # -- Check & update ------------------------------------------------------

    def check_and_update(self):
        cfg = load_config()
        channel = cfg.get("watchdog", {}).get("update_channel", "stable")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.state.set(last_update_check=now)
        logging.info(f"[AU] Checking for updates (channel: {channel})...")

        try:
            tag, zipball_url = self.get_latest_release(channel)
        except Exception as e:
            result = f"Check failed: {e}"
            logging.warning(f"[AU] {result}")
            self.state.set(last_update_result=result)
            return

        if tag is None:
            result = "No releases yet"
            logging.info(f"[AU] {result}")
            self.state.set(last_update_result=result)
            return

        current = read_version()
        remote = tag.lstrip("v")

        if parse_version(current) >= parse_version(remote):
            result = f"Up to date ({current})"
            logging.info(f"[AU] {result}")
            self.state.set(last_update_result=result)
            return

        logging.info(f"[AU] Update available: {current} → {remote}")
        self._apply_update(remote, zipball_url)

    def _apply_update(self, remote, zipball_url):
        # Download BEFORE stopping the app so it keeps running during the transfer
        tmpdir = tempfile.mkdtemp(prefix="stt-update-", dir=APP_DIR)
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

        try:
            for item in os.listdir(src_root):
                if item in _UPDATE_PRESERVE:
                    continue
                src = os.path.join(src_root, item)
                dst = os.path.join(APP_DIR, item)
                try:
                    if os.path.isdir(src):
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
                except Exception as e:
                    logging.warning(f"[AU] Skipped {item}: {e}")

            write_version(remote)
            result = (
                f"Updated to {remote} "
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
            wait = self._seconds_until_next_update()
            logging.info(f"[AU] Next update check in {wait / 3600:.1f}h")
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                if self.state.get("stop_requested"):
                    return
                time.sleep(60)
            if not self.state.get("stop_requested"):
                self.check_and_update()

    @staticmethod
    def _seconds_until_next_update():
        now = datetime.datetime.now()
        target = now.replace(
            hour=UPDATE_HOUR, minute=0, second=0, microsecond=0
        )
        if now >= target:
            target += datetime.timedelta(days=1)
        return max((target - now).total_seconds(), 1)


# ---------------------------------------------------------------------------
# Desktop GUI (tkinter — optional; falls back to headless if unavailable)
# ---------------------------------------------------------------------------

class GuiWindow:
    def __init__(self, state, pm, updater):
        import tkinter as tk
        from tkinter import messagebox
        self._tk = tk
        self._mb = messagebox
        self.state = state
        self.pm = pm
        self.updater = updater
        self.root = tk.Tk()
        self._build_ui()
        self.root.after(500, self._poll)

    def _build_ui(self):
        tk = self._tk
        root = self.root
        root.title(f"STT Watchdog v{read_version()}")
        root.resizable(False, False)
        root.geometry("330x320")

        pad = {"padx": 12, "pady": 4}

        # ── Status row ──────────────────────────────────────────────────────
        sf = tk.Frame(root)
        sf.pack(fill="x", **pad)
        tk.Label(sf, text="Status:", width=9, anchor="w").pack(side="left")
        self._status_lbl = tk.Label(
            sf, text="● Stopped", fg="red", font=("", 10, "bold")
        )
        self._status_lbl.pack(side="left")

        # ── Start / Stop ────────────────────────────────────────────────────
        bf = tk.Frame(root)
        bf.pack(**pad)
        self._toggle_btn = tk.Button(
            bf, text="Start", width=12, command=self._on_toggle
        )
        self._toggle_btn.pack()

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
        self._toggle_btn.config(
            text="Stop" if status == "running" else "Start"
        )

        chk = self.state.get("last_update_check")
        res = self.state.get("last_update_result")
        if chk:
            self._check_lbl.config(text=f"Last check: {chk}")
        if res:
            self._result_lbl.config(text=res)

        self.root.after(1000, self._poll)

    def _on_toggle(self):
        if self.pm.is_alive():
            threading.Thread(target=self.pm.stop, daemon=True).start()
        else:
            threading.Thread(target=self.pm.start, daemon=True).start()

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
        port = cfg.get("web_server", {}).get("port", 80)
        webbrowser.open(f"http://localhost:{port}")

    def mainloop(self):
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Crash reporting
# ---------------------------------------------------------------------------

# Ordered list of (compiled-regex, replacement) sanitization rules.
# Applied in sequence — later rules can operate on already-substituted text.
_SANITIZE_RULES = [
    # Auto-generated password printed at startup: [AUTH] Password: <secret>
    (re.compile(r'(?i)(\[AUTH\]\s*Password\s*:)\s*\S+'), r'\1 [REDACTED]'),
    # Generic key=value credential patterns (config dumps, debug prints)
    (re.compile(r'(?i)(password|passwd|secret|api[_-]?key|token)\s*[=:\'\"]\s*\S+'),
     r'\1=[REDACTED]'),
    # IPv4 addresses
    (re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'), '[IP]'),
    # SMB/UNC paths: //host/share or \\host\share
    (re.compile(r'(?://|\\\\)[^\s/\\\'\"]+[/\\][^\s/\\\'\"]+'), '//[HOST]/[SHARE]'),
    # Unix home dirs: /home/username/ and /Users/username/
    (re.compile(r'(?<=/home/)[^/\s\'"\\]+'), '[USER]'),
    (re.compile(r'(?<=/Users/)[^/\s\'"\\]+'), '[USER]'),
    # Windows user dirs: C:\Users\username
    (re.compile(r'(?<=\\Users\\)[^\\]+'), '[USER]'),
    # Email addresses
    (re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}'), '[EMAIL]'),
]


def sanitize_log(text: str) -> str:
    """Strip PII and credentials from log text before sending in a crash report."""
    for pattern, replacement in _SANITIZE_RULES:
        text = pattern.sub(replacement, text)
    return text


def _crash_fingerprint(log_tail: str) -> str:
    """Stable 12-char hash of the crash signature (traceback, minus line numbers)."""
    lines = log_tail.splitlines()
    sig_lines = []
    for line in lines[-60:]:
        s = line.strip()
        if any(kw in s for kw in ('Error', 'Exception', 'Traceback', 'CRITICAL', 'File "')):
            # Strip line numbers so the same bug hashes identically across versions
            sig_lines.append(re.sub(r',\s*line\s+\d+', '', s))
    signature = '\n'.join(sig_lines) if sig_lines else log_tail[-300:]
    return hashlib.sha256(signature.encode('utf-8', errors='replace')).hexdigest()[:12]


class CrashReporter:
    """Collects, sanitizes, and POSTs crash reports to a Cloudflare Worker.

    The Worker holds the GitHub token in encrypted CF secrets and files
    the issue on behalf of any user. The token never touches user machines.
    All users get crash reporting just by having enabled=true in config.json.
    """

    _STATE_FILE    = os.path.join(LOG_DIR, 'crash_reporter.json')
    _COOLDOWN      = 600    # client-side cooldown per fingerprint (seconds)
    _MAX_LOG_BYTES = 8_000
    _LOG_LINES     = 120

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

    # -- Rate limiting (client-side) -----------------------------------------

    def _is_rate_limited(self, fingerprint: str) -> bool:
        return (time.time() - self._state.get(fingerprint, 0)) < self._COOLDOWN

    def _mark_sent(self, fingerprint: str):
        self._state[fingerprint] = time.time()
        cutoff = time.time() - 86400
        self._state = {k: v for k, v in self._state.items() if v > cutoff}
        self._save_state()

    # -- Public API ----------------------------------------------------------

    def report(self, exit_code: int, consecutive_crashes: int):
        """Non-blocking: fire-and-forget daemon thread."""
        if not _CRASH_WORKER_URL:
            return
        cfg = load_config()
        if not cfg.get('crash_reporting', {}).get('enabled', True):
            return
        threading.Thread(
            target=self._send,
            args=(exit_code, consecutive_crashes, cfg),
            daemon=True,
            name='CrashReport',
        ).start()

    # -- Internal ------------------------------------------------------------

    def _collect_log(self) -> str:
        try:
            with open(os.path.join(LOG_DIR, 'stt.log'), 'r',
                      encoding='utf-8', errors='replace') as f:
                raw = ''.join(f.readlines()[-self._LOG_LINES:])
        except Exception:
            raw = '(stt.log unavailable)'
        return sanitize_log(raw)

    def _send(self, exit_code: int, consecutive_crashes: int, cfg: dict):
        log_tail    = self._collect_log()
        fingerprint = _crash_fingerprint(log_tail)

        if self._is_rate_limited(fingerprint):
            logging.debug(f'[CrashReport] Skipped — rate limited ({fingerprint})')
            return

        payload = {
            'version':             read_version(),
            'platform':            f'{platform.system()} {platform.release()} {platform.machine()}',
            'python_version':      platform.python_version(),
            'exit_code':           exit_code,
            'consecutive_crashes': consecutive_crashes,
            'timestamp':           datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'log_tail':            log_tail[-self._MAX_LOG_BYTES:],
            'fingerprint':         fingerprint,
            'gpu_enabled':         cfg.get('performance', {}).get('use_gpu', False),
            'whisper_model':       cfg.get('model', {}).get('whisper', {}).get('model', 'unknown'),
            'audio_backend':       cfg.get('audio', {}).get('backend', 'unknown'),
        }

        try:
            data = json.dumps(payload).encode('utf-8')
            req  = urllib.request.Request(
                _CRASH_WORKER_URL,
                data=data,
                headers={
                    'Content-Type': 'application/json',
                    'X-Api-Key':    _CRASH_API_KEY,
                    'User-Agent':   f'STT-Watchdog/{read_version()}',
                },
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=20, context=_SSL_CTX) as resp:
                result = json.loads(resp.read().decode())
            logging.info(f'[CrashReport] Filed: {result.get("issue_url", "OK")}')
            self._mark_sent(fingerprint)
        except urllib.error.HTTPError as e:
            logging.warning(f'[CrashReport] Worker {e.code}: {e.read().decode()[:120]}')
        except Exception as e:
            logging.warning(f'[CrashReport] Failed: {e}')


# ---------------------------------------------------------------------------
# GUI availability detection
# ---------------------------------------------------------------------------

def detect_gui():
    """Return True if a functional tkinter display is available."""
    if not IS_WINDOWS:
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
    import datetime, platform
    setup_logging()
    print(f"[test] Worker URL : {_CRASH_WORKER_URL or '(not set)'}")
    if not _CRASH_WORKER_URL:
        print("[test] FAIL — _CRASH_WORKER_URL is empty. Deploy the Cloudflare Worker first.")
        raise SystemExit(1)

    fingerprint = "test-" + __import__("hashlib").sha256(b"test-crash-report").hexdigest()[:8]
    payload = {
        "version":             read_version(),
        "platform":            f"{platform.system()} {platform.release()} {platform.machine()}",
        "python_version":      platform.python_version(),
        "timestamp":           datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exit_code":           1,
        "consecutive_crashes": 1,
        "gpu_enabled":         False,
        "whisper_model":       "test",
        "audio_backend":       "test",
        "fingerprint":         fingerprint,
        "log_tail":            "RuntimeError: synthetic test crash — please ignore and close this issue",
    }
    data = __import__("json").dumps(payload).encode()
    req  = urllib.request.Request(
        _CRASH_WORKER_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-Api-Key":    _CRASH_API_KEY,
            "User-Agent":   f"STT-Watchdog/{read_version()}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20, context=_SSL_CTX) as resp:
            result = __import__("json").loads(resp.read().decode())
        print(f"[test] OK — issue filed: {result.get('issue_url', result)}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        print(f"[test] FAIL — HTTP {e.code}: {body}")
        raise SystemExit(1)
    except Exception as e:
        print(f"[test] FAIL — {e}")
        raise SystemExit(1)


def main():
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
        "--channel",
        choices=["stable", "beta"],
        help="Set update channel (saved to config.json)",
    )
    parser.add_argument(
        "--test-crash-report",
        action="store_true",
        help="Send a synthetic crash report and exit (tests the full pipeline)",
    )
    args = parser.parse_args()

    if args.test_crash_report:
        _run_crash_report_test()
        return

    _lock = acquire_lock()  # noqa: F841 — keep socket alive

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
