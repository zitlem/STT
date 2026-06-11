"""AutoUpdater._apply_update: zip-slip rejection, success path, partial failure.

Runs the real update code against a file:// zipball in an isolated temp
APP_DIR, with the process manager and state stubbed out — the closest thing
to an end-to-end updater test that doesn't need GitHub or a service restart.
"""

import os
import stat
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import watchdog  # noqa: E402  (imports cleanly: stdlib + optional certifi only)


class StubPM:
    def __init__(self):
        self.calls = []

    def stop(self, timeout=None):
        self.calls.append("stop")

    def start(self):
        self.calls.append("start")


class StubState:
    def __init__(self):
        self.values = {}

    def set(self, **kwargs):
        self.values.update(kwargs)


@pytest.fixture
def updater(tmp_path, monkeypatch):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "old.txt").write_text("old content")
    monkeypatch.setattr(watchdog, "APP_DIR", str(app_dir))
    monkeypatch.setattr(watchdog, "VERSION_FILE", str(app_dir / "VERSION"))

    upd = watchdog.AutoUpdater.__new__(watchdog.AutoUpdater)
    upd.pm = StubPM()
    upd.state = StubState()
    return upd, app_dir


def make_zipball(path, files):
    """GitHub-style source zip: one top-level directory."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(f"repo-abc123/{name}", content)
    return path.as_uri()


def test_successful_update_applies_files_and_version(updater, tmp_path):
    upd, app_dir = updater
    url = make_zipball(tmp_path / "u.zip", {"new.txt": "new content", "old.txt": "updated"})

    upd._apply_update("9.9.9", url)

    assert (app_dir / "new.txt").read_text() == "new content"
    assert (app_dir / "old.txt").read_text() == "updated"
    assert watchdog.read_version() == "9.9.9"
    assert "Updated to 9.9.9" in upd.state.values["last_update_result"]
    assert upd.pm.calls == ["stop", "start"], "app must be stopped for copy and restarted after"


def test_zip_slip_is_rejected_before_stopping_the_app(updater, tmp_path):
    upd, app_dir = updater
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("repo-abc/ok.txt", "fine")
        zf.writestr("../escape.txt", "evil")

    upd._apply_update("9.9.9", zip_path.as_uri())

    assert not (tmp_path / "escape.txt").exists()
    assert (app_dir / "old.txt").read_text() == "old content", "no files may change"
    assert watchdog.read_version() != "9.9.9", "version must not be recorded"
    assert "failed" in upd.state.values["last_update_result"].lower()
    assert upd.pm.calls == [], "rejected before the app is ever stopped"


def test_partial_copy_failure_does_not_record_version(updater, tmp_path):
    upd, app_dir = updater
    url = make_zipball(tmp_path / "u.zip", {"blocked/file.txt": "x", "fine.txt": "y"})

    # Make the existing "blocked" dir undeletable so copytree replacement fails
    blocked = app_dir / "blocked"
    blocked.mkdir()
    (blocked / "keep.txt").write_text("z")
    os.chmod(blocked, stat.S_IRUSR | stat.S_IXUSR)  # no write: rmtree fails
    try:
        upd._apply_update("9.9.9", url)
    finally:
        os.chmod(blocked, stat.S_IRWXU)

    assert watchdog.read_version() != "9.9.9", "partial update must not record the new version"
    assert "incomplete" in upd.state.values["last_update_result"].lower()
    assert upd.pm.calls == ["stop", "start"], "app must still be restarted after a failed copy"


def test_preserved_files_are_not_overwritten(updater, tmp_path):
    upd, app_dir = updater
    (app_dir / "config.json").write_text('{"user": "settings"}')
    url = make_zipball(tmp_path / "u.zip", {"config.json": '{"shipped": "default"}', "a.txt": "a"})

    upd._apply_update("9.9.9", url)

    assert (app_dir / "config.json").read_text() == '{"user": "settings"}'
    assert (app_dir / "a.txt").exists()
