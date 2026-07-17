"""stt.self_update.git_self_update: dev-safe fast-forward behavior.

Builds throwaway git repos (bare origin + working clones) in a temp dir and
exercises the real update function — no network, no monolith import. The key
guarantees under test are the developer-safety ones: a dirty tree or unpushed
local commits must never be touched.
"""

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
