"""Model-download tracking: registration state machine, progress persistence,
size monitoring, and the resumable URL downloader.

Extracted from speech_to_text.py so it can be imported (and unit-tested)
without the monolith's import-time side effects. This module OWNS the shared
state (active_downloads / lock / cancelled_downloads); the monolith re-imports
these names, so both sides mutate the same objects. Call configure() with the
progress-file path before persistence matters, then load_state() to restore
the previous run's entries in place.
"""

import json
import os
import shutil
import threading
import time
from typing import Any, Callable, Dict, Optional, Set

# Path of the JSON progress file; set via configure(). While None, persistence
# is skipped (state-machine behavior is unaffected).
_progress_file: Optional[str] = None

# Global dictionary to track active downloads
active_downloads: Dict[str, dict] = {}
active_downloads_lock = threading.Lock()
cancelled_downloads: Set[str] = set()  # Track cancelled download IDs to prevent re-adding


def configure(progress_file: Optional[str]) -> None:
    """Set the on-disk location for download-progress persistence."""
    global _progress_file
    _progress_file = progress_file


def load_state() -> None:
    """Restore active_downloads from disk IN PLACE (the dict object is shared
    with importers, so it must never be rebound)."""
    data = load_download_progress()
    with active_downloads_lock:
        active_downloads.clear()
        active_downloads.update(data)


def load_download_progress() -> dict:
    """Load download progress from file"""
    try:
        if _progress_file and os.path.exists(_progress_file):
            with open(_progress_file, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load download progress: {e}")
    return {}


def save_download_progress() -> None:
    """Save download progress to file"""
    if _progress_file is None:
        return  # not configured (e.g. tests exercising only the state machine)
    try:
        with active_downloads_lock:
            with open(_progress_file, "w") as f:
                json.dump(active_downloads, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save download progress: {e}")


def cleanup_stale_downloads() -> None:
    """Remove downloads based on status and age"""
    import time

    current_time = time.time()

    # Different retention periods by status
    DOWNLOADING_STALE_THRESHOLD = 86400  # 24 hours for stuck downloads
    COMPLETED_GRACE_PERIOD = 7200  # 2 hours for completed downloads
    FAILED_GRACE_PERIOD = 3600  # 1 hour for failed downloads

    with active_downloads_lock:
        stale_keys = []
        for model_id, info in active_downloads.items():
            last_update = info.get("last_update", 0)
            status = info.get("status", "downloading")
            age = current_time - last_update

            # Determine if should be removed based on status
            should_remove = False

            if status == "downloading" and age > DOWNLOADING_STALE_THRESHOLD:
                # Stuck download, likely stale
                should_remove = True
                print(f"[CLEANUP] Removing stale downloading: {model_id} (age: {age/3600:.1f}h)")
            elif status == "completed" and age > COMPLETED_GRACE_PERIOD:
                # Completed downloads after grace period
                should_remove = True
                print(f"[CLEANUP] Removing old completed download: {model_id} (age: {age/3600:.1f}h)")
            elif status == "failed" and age > FAILED_GRACE_PERIOD:
                # Failed downloads after shorter grace period
                should_remove = True
                print(f"[CLEANUP] Removing old failed download: {model_id} (age: {age/3600:.1f}h)")

            if should_remove:
                stale_keys.append(model_id)

        for key in stale_keys:
            del active_downloads[key]

        if stale_keys:
            # Save while still holding the lock - don't call save_download_progress() which would deadlock
            if _progress_file is not None:
                try:
                    with open(_progress_file, "w") as f:
                        json.dump(active_downloads, f, indent=2)
                except Exception as e:
                    print(f"[ERROR] Failed to save download progress: {e}")
            print(f"[CLEANUP] Removed {len(stale_keys)} stale download record(s)")


def try_register_download(key: str, total: Optional[int] = None) -> bool:
    """Atomically register a download in active_downloads.

    Returns False if a download for this key is already in progress."""
    with active_downloads_lock:
        existing = active_downloads.get(key)
        if existing and existing.get("status") == "downloading":
            return False
        cancelled_downloads.discard(key)
        active_downloads[key] = {
            "downloaded": 0,
            "total": total,
            "percentage": 0 if total else None,
            "start_time": time.time(),
            "last_update": time.time(),
            "status": "downloading",
        }
    save_download_progress()
    return True


def finish_download(key: str, error: Optional[Any] = None, cancelled: bool = False) -> None:
    """Mark a download completed/failed and drop it from the cancelled set."""
    with active_downloads_lock:
        cancelled_downloads.discard(key)
        if not cancelled and key in active_downloads:
            entry = active_downloads[key]
            entry["last_update"] = time.time()
            if error is not None:
                entry["status"] = "failed"
                entry["error"] = str(error)
            else:
                entry["status"] = "completed"
                entry["percentage"] = 100
                entry["completion_time"] = time.time()
                if entry.get("total"):
                    entry["downloaded"] = entry["total"]
    save_download_progress()


def _path_size(path: str) -> int:
    """Size in bytes of a file, or recursive size of a directory."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def monitor_download_progress(key: str, path: str, total: Optional[int] = None, interval: float = 2) -> None:
    """Poll the size of `path` (file or directory) and update active_downloads[key].

    Runs until the entry leaves "downloading" state, disappears, or is cancelled.
    Percentage is capped at 99 — the download code sets 100 on completion."""
    import time as _time

    while True:
        with active_downloads_lock:
            entry = active_downloads.get(key)
            if entry is None or entry.get("status") != "downloading" or key in cancelled_downloads:
                return
        if os.path.exists(path):
            size = _path_size(path)
            with active_downloads_lock:
                entry = active_downloads.get(key)
                if entry is None or entry.get("status") != "downloading":
                    return
                entry["downloaded"] = size
                entry["last_update"] = _time.time()
                entry_total = entry.get("total") or total
                if entry_total and entry_total > 0:
                    entry["percentage"] = min(int((size / entry_total) * 100), 99)
            save_download_progress()
        _time.sleep(interval)


def start_download_monitor(key: str, path: str, total: Optional[int] = None, interval: float = 2) -> None:
    """Spawn the directory-size progress monitor as a daemon thread."""
    threading.Thread(
        target=monitor_download_progress,
        args=(key, path, total, interval),
        daemon=True,
        name=f"dl-monitor-{key}",
    ).start()


def download_url_to_file(url: str, dest_path: str, cancel_check: Optional[Callable[[], bool]] = None, max_attempts: int = 5, log: Callable[[str], Any] = print) -> str:
    """Download a URL to a file with resume + retry, preferring wget/curl.

    Falls back to a pure-Python streaming download when neither tool exists
    (e.g. minimal Windows installs). `cancel_check` is polled during the
    download; returning True aborts it. Returns "ok" or "cancelled"; raises
    after all attempts fail."""
    import subprocess
    import tempfile as _tempfile
    import time as _time
    import urllib.request

    if shutil.which("wget"):
        dl_cmd = ['wget', '-c', '-t', '3', '-T', '120', '--retry-connrefused',
                  '--waitretry', '5', '-O', dest_path, url]
    elif shutil.which("curl"):
        dl_cmd = ['curl', '-L', '-C', '-', '--retry', '3', '--retry-delay', '5',
                  '--retry-connrefused', '--connect-timeout', '30',
                  '--max-time', '600', '-o', dest_path, url]
    else:
        dl_cmd = None  # pure-Python fallback below

    last_error = ""
    for attempt in range(1, max_attempts + 1):
        if dl_cmd:
            # Output goes to a temp file: a PIPE would fill up with progress
            # noise and block the process, since nothing drains it while we poll
            with _tempfile.TemporaryFile(mode="w+", errors="replace") as outf:
                # creationflags: windowless server — a console child would flash
                # a window on Windows (0 elsewhere).
                proc = subprocess.Popen(dl_cmd, stdout=outf, stderr=subprocess.STDOUT,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                while proc.poll() is None:
                    if cancel_check and cancel_check():
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        return "cancelled"
                    _time.sleep(0.5)
                if proc.returncode == 0:
                    return "ok"
                outf.seek(0)
                last_error = outf.read()[-500:]
            returncode = proc.returncode
        else:
            try:
                with urllib.request.urlopen(url, timeout=120) as src, open(dest_path, "wb") as out:
                    while True:
                        if cancel_check and cancel_check():
                            return "cancelled"
                        chunk = src.read(65536)
                        if not chunk:
                            break
                        out.write(chunk)
                return "ok"
            except Exception as e:
                last_error = str(e)
                returncode = 1

        log(f"[WARNING] Download attempt {attempt}/{max_attempts} failed for "
            f"{os.path.basename(dest_path)} (exit code {returncode})")
        if attempt < max_attempts:
            if os.path.exists(dest_path):
                partial_size = os.path.getsize(dest_path)
                log(f"[INFO] Partial file exists ({partial_size / (1024*1024):.1f} MB), will resume")
            _time.sleep(5 * attempt)

    raise Exception(
        f"Failed to download {os.path.basename(dest_path)} after {max_attempts} attempts: {last_error[:300]}"
    )
