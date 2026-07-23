"""stt.self_update.git_self_update: dev-safe fast-forward behavior.

Builds throwaway git repos (bare origin + working clones) in a temp dir and
exercises the real update function — no network, no monolith import. The key
guarantees under test are the developer-safety ones: a dirty tree or unpushed
local commits must never be touched.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stt.self_update as self_update  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("git"), reason="git not available")


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True, text=True)


def _configure(repo):
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")


def _head(repo):
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repos(tmp_path):
    """Return (origin_bare, seed, clone). `clone` is the checkout under test."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)],
                   check=True, capture_output=True, text=True)

    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(origin), str(seed)],
                   check=True, capture_output=True, text=True)
    _configure(seed)
    (seed / "file.txt").write_text("v1\n")
    _git(seed, "add", "file.txt")
    _git(seed, "commit", "-m", "initial")
    _git(seed, "push", "-u", "origin", "main")

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", str(origin), str(clone)],
                   check=True, capture_output=True, text=True)
    _configure(clone)
    return origin, seed, clone


def _advance_origin(seed, content="v2\n"):
    (seed / "file.txt").write_text(content)
    _git(seed, "commit", "-am", "advance")
    _git(seed, "push", "origin", "main")


def test_fast_forward(repos):
    _, seed, clone = repos
    _advance_origin(seed)
    before = _head(clone)

    updated, reason = self_update.git_self_update(str(clone))

    assert updated is True
    assert reason == "fast-forwarded"
    assert _head(clone) != before
    assert (clone / "file.txt").read_text() == "v2\n"


def test_up_to_date(repos):
    _, _, clone = repos
    before = _head(clone)

    updated, reason = self_update.git_self_update(str(clone))

    assert updated is False
    assert reason == "up-to-date"
    assert _head(clone) == before


def test_dirty_worktree_is_left_untouched(repos):
    """Uncommitted developer changes must block the update and survive intact."""
    _, seed, clone = repos
    _advance_origin(seed)  # an update IS available...
    (clone / "file.txt").write_text("my local edits\n")  # ...but the tree is dirty
    before = _head(clone)

    updated, reason = self_update.git_self_update(str(clone))

    assert updated is False
    assert reason == "dirty-worktree"
    assert _head(clone) == before
    assert (clone / "file.txt").read_text() == "my local edits\n"  # not discarded


def test_unpushed_commits_are_not_clobbered(repos):
    """A branch ahead of origin (unpushed commits) must not be reset/lost."""
    _, seed, clone = repos
    _advance_origin(seed)  # origin advances (diverging)
    # local, unpushed commit on the clone
    (clone / "local.txt").write_text("wip\n")
    _git(clone, "add", "local.txt")
    _git(clone, "commit", "-m", "local wip")
    local_head = _head(clone)

    updated, reason = self_update.git_self_update(str(clone))

    assert updated is False
    assert reason == "not-fast-forwardable"
    assert _head(clone) == local_head           # local commit still HEAD
    assert (clone / "local.txt").exists()       # local work intact


def test_not_a_git_checkout(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()

    updated, reason = self_update.git_self_update(str(plain))

    assert updated is False
    assert reason == "not-a-git-checkout"


def test_git_commit_returns_short_sha(repos):
    _, _, clone = repos
    sha = self_update.git_commit(str(clone))
    assert sha  # non-empty
    assert re.fullmatch(r"[0-9a-f]{7,40}", sha)
    # matches git's own short SHA for HEAD
    assert _head(str(clone)).startswith(sha)


def test_git_commit_empty_for_non_git_dir(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert self_update.git_commit(str(plain)) == ""


# --- _sync_deps_if_needed: hash-marker dependency healing -------------------


@pytest.fixture
def synced_calls(monkeypatch):
    """Stub out the real uv install; record invocations."""
    calls = []
    monkeypatch.setattr(self_update, "_sync_deps", lambda repo: calls.append(repo) or True)
    return calls


def test_no_requirements_no_sync(tmp_path, synced_calls):
    self_update._sync_deps_if_needed(str(tmp_path))
    assert synced_calls == []


def test_missing_marker_triggers_sync_and_writes_marker(tmp_path, synced_calls):
    (tmp_path / "requirements.txt").write_text("flask\n")
    (tmp_path / ".venv").mkdir()

    self_update._sync_deps_if_needed(str(tmp_path))

    assert len(synced_calls) == 1
    marker = tmp_path / ".venv" / ".requirements-synced"
    assert marker.read_text().strip() == self_update._requirements_hash(str(tmp_path))


def test_matching_marker_skips_sync(tmp_path, synced_calls):
    (tmp_path / "requirements.txt").write_text("flask\n")
    (tmp_path / ".venv").mkdir()
    self_update._sync_deps_if_needed(str(tmp_path))  # writes marker

    self_update._sync_deps_if_needed(str(tmp_path))

    assert len(synced_calls) == 1  # no second sync


def test_changed_requirements_resyncs(tmp_path, synced_calls):
    (tmp_path / "requirements.txt").write_text("flask\n")
    (tmp_path / ".venv").mkdir()
    self_update._sync_deps_if_needed(str(tmp_path))

    (tmp_path / "requirements.txt").write_text("flask\nsentry-sdk\n")
    self_update._sync_deps_if_needed(str(tmp_path))

    assert len(synced_calls) == 2


def test_failed_sync_leaves_no_marker_so_it_retries(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(self_update, "_sync_deps", lambda repo: calls.append(repo) and False)
    (tmp_path / "requirements.txt").write_text("flask\n")
    (tmp_path / ".venv").mkdir()

    self_update._sync_deps_if_needed(str(tmp_path))
    self_update._sync_deps_if_needed(str(tmp_path))

    assert len(calls) == 2  # retried because no marker was written
    assert not (tmp_path / ".venv" / ".requirements-synced").exists()


# --- git_describe ------------------------------------------------------------


def test_git_describe_exact_tag(repos):
    _, _, clone = repos
    _git(clone, "tag", "v26.1.2")
    assert self_update.git_describe(str(clone)) == "v26.1.2"


def test_git_describe_falls_back_to_hash_without_tags(repos):
    _, _, clone = repos
    desc = self_update.git_describe(str(clone))  # --always: bare short hash
    assert desc
    assert _head(str(clone)).startswith(desc)


def test_git_describe_empty_for_non_git_dir(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert self_update.git_describe(str(plain)) == ""


# --- _sync_deps: the real uv invocation (subprocess stubbed) -----------------


class FakeCompleted:
    def __init__(self, returncode, stderr="", stdout=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout


def test_sync_deps_success(tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("flask\n")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["cwd"] = kw.get("cwd")
        return FakeCompleted(0)

    monkeypatch.setattr(self_update.subprocess, "run", fake_run)
    assert self_update._sync_deps(str(tmp_path)) is True
    assert seen["cmd"][:3] == ["uv", "pip", "install"]
    assert seen["cwd"] == str(tmp_path)


def test_sync_deps_nonzero_exit_returns_false(tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("flask\n")
    monkeypatch.setattr(self_update.subprocess, "run", lambda *a, **k: FakeCompleted(1, stderr="resolver error"))
    assert self_update._sync_deps(str(tmp_path)) is False


def test_sync_deps_exception_swallowed(tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("flask\n")

    def boom(*a, **k):
        raise OSError("uv not installed")

    monkeypatch.setattr(self_update.subprocess, "run", boom)
    assert self_update._sync_deps(str(tmp_path)) is False


def test_sync_deps_without_requirements_is_false(tmp_path):
    assert self_update._sync_deps(str(tmp_path)) is False


# --- git_self_update failure paths -------------------------------------------


def test_git_timeout_reports_error(repos, monkeypatch):
    _, _, clone = repos

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="git", timeout=1)

    monkeypatch.setattr(self_update, "_git", timeout)
    assert self_update.git_self_update(str(clone)) == (False, "error")


def test_git_status_failure_reports_error(repos, monkeypatch):
    _, _, clone = repos
    monkeypatch.setattr(self_update, "_git", lambda *a, **k: FakeCompleted(128, stderr="fatal: broken"))
    assert self_update.git_self_update(str(clone)) == (False, "error")


def test_fast_forward_triggers_dep_sync(repos, monkeypatch):
    _, seed, clone = repos
    _advance_origin(seed)
    synced = []
    monkeypatch.setattr(self_update, "_sync_deps_if_needed", lambda repo: synced.append(repo))

    updated, _ = self_update.git_self_update(str(clone))

    assert updated is True
    assert synced == [str(clone)]


def test_up_to_date_also_heals_dep_drift(repos, monkeypatch):
    _, _, clone = repos
    synced = []
    monkeypatch.setattr(self_update, "_sync_deps_if_needed", lambda repo: synced.append(repo))

    updated, reason = self_update.git_self_update(str(clone))

    assert (updated, reason) == (False, "up-to-date")
    assert synced == [str(clone)]
