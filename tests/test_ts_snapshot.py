"""Tests for the Manager-proxy-tolerant transcription_state accessors."""

import threading

from conftest import extract_definitions

SOURCE = "speech_to_text.py"
PROXY_ERRORS = (BrokenPipeError, EOFError, ConnectionError, FileNotFoundError, AttributeError, TypeError)


class FakeProxy:
    """Mimics a DictProxy (mapping protocol, NOT a dict subclass — dict(proxy)
    must go through keys()); raises `error` on access when set."""

    def __init__(self, data=None, error=None):
        self._data = dict(data or {})
        self.error = error

    def _raise_if_broken(self):
        if self.error is not None:
            raise self.error

    def get(self, key, default=None):
        self._raise_if_broken()
        return self._data.get(key, default)

    def keys(self):
        self._raise_if_broken()
        return self._data.keys()

    def __getitem__(self, key):
        self._raise_if_broken()
        return self._data[key]


def _ns(proxy):
    return extract_definitions(
        SOURCE,
        ["_ts_get", "_ts_snapshot"],
        extra_globals={
            "transcription_state": proxy,
            "_server_shutting_down": threading.Event(),
            "_TS_PROXY_ERRORS": PROXY_ERRORS,
        },
    )


class TestTsGet:
    def test_healthy_proxy_passes_through(self):
        ns = _ns(FakeProxy({"running": True}))
        assert ns["_ts_get"]("running") is True
        assert ns["_ts_get"]("missing", "d") == "d"
        assert not ns["_server_shutting_down"].is_set()

    def test_dead_proxy_returns_default_and_flags_shutdown(self):
        for err in (ConnectionRefusedError(111, "Connection refused"), BrokenPipeError(),
                    EOFError(), FileNotFoundError()):
            ns = _ns(FakeProxy({"running": True}, error=err))
            assert ns["_ts_get"]("running", False) is False
            assert ns["_server_shutting_down"].is_set()

    def test_none_proxy_returns_default(self):
        ns = _ns(None)  # spawn-mode fallback: transcription_state is None
        assert ns["_ts_get"]("running", False) is False
        assert ns["_server_shutting_down"].is_set()


class TestTsSnapshot:
    def test_healthy_proxy_converts_to_dict(self):
        ns = _ns(FakeProxy({"running": True, "status": "running"}))
        snap = ns["_ts_snapshot"]()
        assert snap == {"running": True, "status": "running"}
        assert isinstance(snap, dict)
        assert not ns["_server_shutting_down"].is_set()

    def test_dead_proxy_returns_restarting_state(self):
        ns = _ns(FakeProxy({"running": True}, error=ConnectionRefusedError(111, "Connection refused")))
        snap = ns["_ts_snapshot"]()
        assert snap["running"] is False
        assert snap["status"] == "restarting"
        assert ns["_server_shutting_down"].is_set()

    def test_none_proxy_returns_restarting_state(self):
        ns = _ns(None)
        assert ns["_ts_snapshot"]()["status"] == "restarting"
