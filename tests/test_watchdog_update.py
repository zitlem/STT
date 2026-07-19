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


# --- 'main' channel: branch-tracking detection and apply ---------------------
# Real bare origin + clone (pattern from test_self_update.py) so the actual
# git fetch/reset/rollback code runs; PM, state, and deps stay stubbed.

import subprocess  # noqa: E402

needs_git = pytest.mark.skipif(not shutil.which("git"), reason="git not available")


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True, text=True)


def _head(repo):
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def git_updater(tmp_path, monkeypatch):
    """(updater, seed, clone): clone is the managed SOURCE_DIR checkout."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)],
                   check=True, capture_output=True, text=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(origin), str(seed)],
                   check=True, capture_output=True, text=True)
    _git(seed, "config", "user.email", "test@example.com")
    _git(seed, "config", "user.name", "Test")
    (seed / "file.txt").write_text("v1\n")
    _git(seed, "add", "file.txt")
    _git(seed, "commit", "-m", "initial")
    _git(seed, "push", "-u", "origin", "main")

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", str(origin), str(clone)],
                   check=True, capture_output=True, text=True)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(watchdog, "SOURCE_DIR", str(clone))
    monkeypatch.setattr(watchdog, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(watchdog, "VERSION_FILE", str(clone / "VERSION"))
    monkeypatch.setattr(watchdog.Provisioner, "install_deps_only",
                        lambda self, log=None: None)

    upd = watchdog.AutoUpdater.__new__(watchdog.AutoUpdater)
    upd.pm = StubPM()
    upd.state = StubState()
    upd._pending_update = None
    return upd, seed, clone


def _advance_origin(seed, content="v2\n"):
    (seed / "file.txt").write_text(content)
    _git(seed, "commit", "-am", "advance")
    _git(seed, "push", "origin", "main")


@needs_git
def test_main_channel_up_to_date(git_updater):
    upd, seed, clone = git_updater

    upd._check_for_branch_update()

    assert upd._pending_update is None
    assert upd.state.values["last_update_result"].startswith("Up to date (main @")


@needs_git
def test_main_channel_detects_and_applies_update(git_updater):
    upd, seed, clone = git_updater
    _advance_origin(seed)

    upd._check_for_branch_update()
    assert upd._pending_update == (watchdog.AutoUpdater._BRANCH_TARGET, None, {})
    assert "Update available: main @" in upd.state.values["last_update_result"]

    upd.apply_pending_update()

    assert (clone / "file.txt").read_text() == "v2\n"
    assert _head(clone) == _head(seed)
    assert upd.pm.calls == ["stop", "start"]
    assert not (clone / "VERSION").exists(), "branch mode must not write the VERSION file"
    assert "Updated to" in upd.state.values["last_update_result"]


@needs_git
def test_main_channel_rollback_on_dep_failure(git_updater, monkeypatch):
    upd, seed, clone = git_updater
    prev = _head(clone)
    _advance_origin(seed)
    upd._check_for_branch_update()

    calls = {"n": 0}

    def flaky_deps(self, log=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("dep fail")

    monkeypatch.setattr(watchdog.Provisioner, "install_deps_only", flaky_deps)
    upd.apply_pending_update()

    assert _head(clone) == prev, "failed dep install must roll the checkout back"
    assert "rolled back" in upd.state.values["last_update_result"]
    assert upd.pm.calls == ["stop", "start"]


@needs_git
def test_check_for_update_defaults_to_main_channel(git_updater, monkeypatch):
    upd, seed, clone = git_updater
    monkeypatch.setattr(watchdog, "load_config", lambda: {})  # no channel configured
    _advance_origin(seed)

    upd.check_for_update()  # must route to branch flow, no Releases API call

    assert upd._pending_update == (watchdog.AutoUpdater._BRANCH_TARGET, None, {})


@needs_git
def test_stable_channel_still_uses_releases(git_updater, monkeypatch):
    upd, seed, clone = git_updater
    _advance_origin(seed)
    monkeypatch.setattr(watchdog, "load_config",
                        lambda: {"watchdog": {"update_channel": "stable"}})
    monkeypatch.setattr(upd, "get_latest_release", lambda channel: (None, None, {}))

    upd.check_for_update()

    assert upd._pending_update is None
    assert upd.state.values["last_update_result"] == "No releases yet"
    assert _head(clone) != _head(seed), "stable channel must not track main"
