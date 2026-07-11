"""AutoUpdater._apply_update: zip-slip rejection, success path, swap rollback.

Runs the real update code against a file:// zipball in an isolated temp
SOURCE_DIR/DATA_DIR, with the process manager, state, and dependency
installer stubbed out — the closest thing to an end-to-end updater test
that doesn't need GitHub or a service restart.
"""

import os
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stt.watchdog as watchdog  # noqa: E402  (imports cleanly: stdlib + optional certifi only)


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
    source_dir = tmp_path / "app"
    data_dir = tmp_path / "data"
    source_dir.mkdir()
    data_dir.mkdir()
    (source_dir / "old.txt").write_text("old content")
    monkeypatch.setattr(watchdog, "SOURCE_DIR", str(source_dir))
    monkeypatch.setattr(watchdog, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(watchdog, "VERSION_FILE", str(source_dir / "VERSION"))
    monkeypatch.setattr(watchdog.Provisioner, "install_deps_only",
                        lambda self, log=None: None)

    upd = watchdog.AutoUpdater.__new__(watchdog.AutoUpdater)
    upd.pm = StubPM()
    upd.state = StubState()
    return upd, source_dir


def make_zipball(path, files):
    """GitHub-style source zip: one top-level directory."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(f"repo-abc123/{name}", content)
    return path.as_uri()


def test_successful_update_applies_files_and_version(updater, tmp_path):
    upd, source_dir = updater
    url = make_zipball(tmp_path / "u.zip", {"new.txt": "new content", "old.txt": "updated"})

    upd._apply_update("9.9.9", url)

    assert (source_dir / "new.txt").read_text() == "new content"
    assert (source_dir / "old.txt").read_text() == "updated"
    assert watchdog.read_version() == "9.9.9"
    assert "Updated to 9.9.9" in upd.state.values["last_update_result"]
    assert upd.pm.calls == ["stop", "start"], "app must be stopped for the swap and restarted after"


def test_zip_slip_is_rejected_before_stopping_the_app(updater, tmp_path):
    upd, source_dir = updater
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("repo-abc/ok.txt", "fine")
        zf.writestr("../escape.txt", "evil")

    upd._apply_update("9.9.9", zip_path.as_uri())

    assert not (tmp_path / "escape.txt").exists()
    assert (source_dir / "old.txt").read_text() == "old content", "no files may change"
    assert watchdog.read_version() != "9.9.9", "version must not be recorded"
    assert "failed" in upd.state.values["last_update_result"].lower()
    assert upd.pm.calls == [], "rejected before the app is ever stopped"


def test_swap_failure_restores_previous_version(updater, tmp_path, monkeypatch):
    upd, source_dir = updater
    url = make_zipball(tmp_path / "u.zip",
                       {"a.txt": "a", "boom.txt": "b", "old.txt": "updated"})

    real_move = shutil.move

    def failing_move(src, dst, *a, **kw):
        # Fail only when placing the staged "boom.txt" into SOURCE (not on
        # backup parking or rollback restores, which move other paths).
        if os.path.basename(str(src)) == "boom.txt" and "extracted" in str(src):
            raise OSError("disk full")
        return real_move(src, dst, *a, **kw)

    monkeypatch.setattr(watchdog.shutil, "move", failing_move)

    upd._apply_update("9.9.9", url)

    assert (source_dir / "old.txt").read_text() == "old content", "old version must be restored"
    assert not (source_dir / "a.txt").exists(), "partially applied items must be rolled back"
    assert not (source_dir / "boom.txt").exists()
    assert watchdog.read_version() != "9.9.9", "failed update must not record the new version"
    assert "restored" in upd.state.values["last_update_result"].lower()
    assert upd.pm.calls == ["stop", "start"], "app must still be restarted after a failed swap"


def test_dep_install_failure_restores_previous_version(updater, tmp_path, monkeypatch):
    upd, source_dir = updater
    url = make_zipball(tmp_path / "u.zip", {"old.txt": "updated", "new.txt": "n"})

    def boom(self, log=None):
        raise watchdog.ProvisionError("uv exploded")

    monkeypatch.setattr(watchdog.Provisioner, "install_deps_only", boom)

    upd._apply_update("9.9.9", url)

    assert (source_dir / "old.txt").read_text() == "old content", "old version must be restored"
    assert not (source_dir / "new.txt").exists()
    assert watchdog.read_version() != "9.9.9"
    assert upd.pm.calls == ["stop", "start"]


def test_preserved_venv_is_not_touched(updater, tmp_path):
    upd, source_dir = updater
    venv = source_dir / ".venv"
    venv.mkdir()
    (venv / "marker.txt").write_text("keep me")
    url = make_zipball(tmp_path / "u.zip", {".venv/evil.txt": "nope", "a.txt": "a"})

    upd._apply_update("9.9.9", url)

    assert (venv / "marker.txt").read_text() == "keep me"
    assert not (venv / "evil.txt").exists(), ".venv content from the archive must be skipped"
    assert (source_dir / "a.txt").exists()
    assert watchdog.read_version() == "9.9.9"
