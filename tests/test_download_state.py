"""Download registration / completion state machine (speech_to_text.py)."""

import threading
import time

from conftest import extract_definitions


def make_ns():
    return extract_definitions(
        "speech_to_text.py",
        ["try_register_download", "finish_download"],
        extra_globals={
            "active_downloads": {},
            "active_downloads_lock": threading.Lock(),
            "cancelled_downloads": set(),
            "save_download_progress": lambda: None,
            "time": time,
        },
    )


def test_register_and_duplicate_guard():
    ns = make_ns()
    reg = ns["try_register_download"]
    assert reg("m1") is True
    assert reg("m1") is False, "duplicate while downloading must be rejected"


def test_reregister_after_failure_and_completion():
    ns = make_ns()
    reg, fin = ns["try_register_download"], ns["finish_download"]
    assert reg("m1")
    fin("m1", error="boom")
    assert ns["active_downloads"]["m1"]["status"] == "failed"
    assert reg("m1"), "re-register after failure must succeed"
    fin("m1")
    entry = ns["active_downloads"]["m1"]
    assert entry["status"] == "completed"
    assert entry["percentage"] == 100
    assert reg("m1"), "re-register after completion must succeed"


def test_completion_sets_downloaded_to_total():
    ns = make_ns()
    ns["try_register_download"]("m2", total=500)
    ns["active_downloads"]["m2"]["downloaded"] = 123
    ns["finish_download"]("m2")
    assert ns["active_downloads"]["m2"]["downloaded"] == 500


def test_cancelled_finish_clears_set_and_keeps_entry_untouched():
    ns = make_ns()
    ns["try_register_download"]("m3")
    ns["cancelled_downloads"].add("m3")
    ns["finish_download"]("m3", cancelled=True)
    assert "m3" not in ns["cancelled_downloads"], "finish must clear the cancelled set"
    assert ns["active_downloads"]["m3"]["status"] == "downloading", (
        "cancelled finish must not overwrite the entry (cancel route removes it)"
    )


def test_registration_with_total_starts_at_zero_percent():
    ns = make_ns()
    ns["try_register_download"]("m4", total=1000)
    assert ns["active_downloads"]["m4"]["percentage"] == 0
    ns2 = make_ns()
    ns2["try_register_download"]("m5")
    assert ns2["active_downloads"]["m5"]["percentage"] is None
