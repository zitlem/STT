"""Download registration / completion state machine (stt/downloads.py)."""

import json
import threading
import time

import pytest

from stt import downloads


@pytest.fixture(autouse=True)
def clean_state():
    """The module owns shared state; isolate every test."""
    downloads.configure(None)
    with downloads.active_downloads_lock:
        downloads.active_downloads.clear()
    downloads.cancelled_downloads.clear()
    yield
    downloads.configure(None)
    with downloads.active_downloads_lock:
        downloads.active_downloads.clear()
    downloads.cancelled_downloads.clear()


def test_register_and_duplicate_guard():
    assert downloads.try_register_download("m1") is True
    assert downloads.try_register_download("m1") is False, "duplicate while downloading must be rejected"


def test_reregister_after_failure_and_completion():
    assert downloads.try_register_download("m1")
    downloads.finish_download("m1", error="boom")
    assert downloads.active_downloads["m1"]["status"] == "failed"
    assert downloads.try_register_download("m1"), "re-register after failure must succeed"
    downloads.finish_download("m1")
    entry = downloads.active_downloads["m1"]
    assert entry["status"] == "completed"
    assert entry["percentage"] == 100
    assert downloads.try_register_download("m1"), "re-register after completion must succeed"


def test_completion_sets_downloaded_to_total():
    downloads.try_register_download("m2", total=500)
    downloads.active_downloads["m2"]["downloaded"] = 123
    downloads.finish_download("m2")
    assert downloads.active_downloads["m2"]["downloaded"] == 500


def test_cancelled_finish_clears_set_and_keeps_entry_untouched():
    downloads.try_register_download("m3")
    downloads.cancelled_downloads.add("m3")
    downloads.finish_download("m3", cancelled=True)
    assert "m3" not in downloads.cancelled_downloads, "finish must clear the cancelled set"
    assert downloads.active_downloads["m3"]["status"] == "downloading", (
        "cancelled finish must not overwrite the entry (cancel route removes it)"
    )


def test_registration_with_total_starts_at_zero_percent():
    downloads.try_register_download("m4", total=1000)
    assert downloads.active_downloads["m4"]["percentage"] == 0
    downloads.try_register_download("m5")
    assert downloads.active_downloads["m5"]["percentage"] is None


# --- persistence (configure / load_state) ------------------------------------


def test_progress_persisted_and_restored(tmp_path):
    progress = tmp_path / "download_progress.json"
    downloads.configure(str(progress))
    downloads.try_register_download("m1", total=100)
    assert json.loads(progress.read_text())["m1"]["status"] == "downloading"

    # Simulate a restart: state cleared, then restored from disk in place
    dict_id = id(downloads.active_downloads)
    with downloads.active_downloads_lock:
        downloads.active_downloads.clear()
    downloads.load_state()
    assert downloads.active_downloads["m1"]["total"] == 100
    assert id(downloads.active_downloads) == dict_id, "load_state must never rebind the shared dict"


def test_unconfigured_persistence_is_a_noop():
    downloads.try_register_download("m1")  # must not raise with no file configured
    assert downloads.load_download_progress() == {}


def test_corrupt_progress_file_loads_empty(tmp_path):
    progress = tmp_path / "download_progress.json"
    progress.write_text("{not json")
    downloads.configure(str(progress))
    assert downloads.load_download_progress() == {}


# --- cleanup_stale_downloads --------------------------------------------------


def test_cleanup_by_status_and_age(tmp_path, monkeypatch):
    downloads.configure(str(tmp_path / "p.json"))
    now = 1_000_000.0
    monkeypatch.setattr(downloads.time, "time", lambda: now)
    with downloads.active_downloads_lock:
        downloads.active_downloads.update({
            "stuck":        {"status": "downloading", "last_update": now - 90000},  # >24h
            "fresh-dl":     {"status": "downloading", "last_update": now - 3600},
            "old-done":     {"status": "completed",   "last_update": now - 8000},   # >2h
            "recent-done":  {"status": "completed",   "last_update": now - 600},
            "old-failed":   {"status": "failed",      "last_update": now - 4000},   # >1h
            "recent-fail":  {"status": "failed",      "last_update": now - 600},
        })
    downloads.cleanup_stale_downloads()
    assert sorted(downloads.active_downloads) == ["fresh-dl", "recent-done", "recent-fail"]


def test_cleanup_noop_keeps_file_untouched(tmp_path):
    progress = tmp_path / "p.json"
    downloads.configure(str(progress))
    downloads.cleanup_stale_downloads()
    assert not progress.exists()  # nothing removed -> nothing written


# --- monitor_download_progress ------------------------------------------------


def _wait_for(cond, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_monitor_tracks_size_and_exits_on_finish(tmp_path):
    target = tmp_path / "model.bin"
    target.write_bytes(b"x" * 50)
    downloads.try_register_download("mon", total=100)
    t = threading.Thread(
        target=downloads.monitor_download_progress,
        args=("mon", str(target), None, 0.01), daemon=True,
    )
    t.start()
    assert _wait_for(lambda: downloads.active_downloads["mon"].get("downloaded") == 50)
    assert downloads.active_downloads["mon"]["percentage"] == 50
    downloads.finish_download("mon")  # leaves "downloading" -> monitor must exit
    t.join(timeout=5)
    assert not t.is_alive()
    assert downloads.active_downloads["mon"]["percentage"] == 100


def test_monitor_caps_percentage_at_99(tmp_path):
    target = tmp_path / "model.bin"
    target.write_bytes(b"x" * 300)  # larger than the claimed total
    downloads.try_register_download("mon2", total=100)
    t = threading.Thread(
        target=downloads.monitor_download_progress,
        args=("mon2", str(target), None, 0.01), daemon=True,
    )
    t.start()
    assert _wait_for(lambda: downloads.active_downloads["mon2"].get("downloaded") == 300)
    assert downloads.active_downloads["mon2"]["percentage"] == 99, "100 is reserved for completion"
    downloads.finish_download("mon2")
    t.join(timeout=5)


def test_monitor_returns_immediately_when_cancelled(tmp_path):
    downloads.try_register_download("mon3")
    downloads.cancelled_downloads.add("mon3")
    # Runs in-thread: must return on the first loop iteration without sleeping
    downloads.monitor_download_progress("mon3", str(tmp_path / "nope.bin"), interval=60)


def test_monitor_returns_when_entry_missing(tmp_path):
    downloads.monitor_download_progress("ghost", str(tmp_path / "nope.bin"), interval=60)


# --- _path_size ---------------------------------------------------------------


def test_path_size_file_and_directory(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"x" * 10)
    assert downloads._path_size(str(f)) == 10
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 5)
    assert downloads._path_size(str(tmp_path)) == 15
