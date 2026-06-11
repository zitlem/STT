"""download_url_to_file: retry/resume/cancel downloader (speech_to_text.py)."""

import http.server
import shutil
import socketserver
import threading
import time

import pytest

from conftest import extract_definitions

CONTENT = b"x" * 300_000


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        slow = self.path == "/slow"
        self.send_response(200)
        self.send_header("Content-Length", str(len(CONTENT)))
        self.end_headers()
        try:
            if slow:
                for i in range(0, len(CONTENT), 10_000):
                    self.wfile.write(CONTENT[i:i + 10_000])
                    time.sleep(0.2)
            else:
                self.wfile.write(CONTENT)
        except BrokenPipeError:
            pass  # client (or terminated wget) went away — expected in cancel tests

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def http_server():
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


class _NoToolShutil:
    """Force the pure-Python fallback path."""

    @staticmethod
    def which(name):
        return None


def _downloader(shutil_impl):
    ns = extract_definitions(
        "speech_to_text.py", ["download_url_to_file"],
        extra_globals={"shutil": shutil_impl},
    )
    return ns["download_url_to_file"]


def test_python_fallback_ok(http_server, tmp_path):
    dl = _downloader(_NoToolShutil)
    dest = tmp_path / "f.bin"
    assert dl(f"{http_server}/fast", str(dest)) == "ok"
    assert dest.stat().st_size == len(CONTENT)


def test_python_fallback_cancel(http_server, tmp_path):
    dl = _downloader(_NoToolShutil)
    assert dl(f"{http_server}/fast", str(tmp_path / "c.bin"), cancel_check=lambda: True) == "cancelled"


def test_python_fallback_raises_after_retries(http_server, tmp_path):
    dl = _downloader(_NoToolShutil)
    bad_url = http_server.rsplit(":", 1)[0] + ":1/nope"  # nothing listens on port 1
    with pytest.raises(Exception, match="Failed to download"):
        dl(bad_url, str(tmp_path / "f.bin"), max_attempts=1)


needs_tool = pytest.mark.skipif(
    not (shutil.which("wget") or shutil.which("curl")),
    reason="neither wget nor curl installed",
)


@needs_tool
def test_subprocess_path_ok(http_server, tmp_path):
    dl = _downloader(shutil)
    dest = tmp_path / "a.bin"
    assert dl(f"{http_server}/fast", str(dest)) == "ok"
    assert dest.stat().st_size == len(CONTENT)


@needs_tool
def test_subprocess_path_cancel_mid_download(http_server, tmp_path):
    dl = _downloader(shutil)
    cancel = {"flag": False}
    threading.Timer(1.0, lambda: cancel.update(flag=True)).start()
    t0 = time.time()
    outcome = dl(f"{http_server}/slow", str(tmp_path / "b.bin"), cancel_check=lambda: cancel["flag"])
    assert outcome == "cancelled"
    assert time.time() - t0 < 4, "cancel must terminate the subprocess promptly"
