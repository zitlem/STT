"""Dev-safe git self-update for direct (non-watchdog) runs of the STT server.

When the server is launched directly (start_server.sh, a custom systemd unit, or
plain ``python speech_to_text.py``) the watchdog's updater is not in play, so this
module fast-forwards the checkout to its upstream branch instead.

Unlike the watchdog (which does a destructive ``git reset --hard`` + ``git clean``),
this path is deliberately non-destructive so developers who run from their own
checkout never lose work:

  * it refuses to touch a **dirty** working tree, and
  * it uses ``git pull --ff-only``, which refuses to run when the branch has
    diverged or is **ahead** (unpushed local commits).

In both cases it simply skips — nothing is discarded, and the developer can still
commit and push normally.
"""

import logging
import os
import shutil
import subprocess
import sys

log = logging.getLogger(__name__)

# Timeouts (seconds) for the individual git invocations.
_GIT_TIMEOUT = 60


def _git(repo_dir, *args, timeout=_GIT_TIMEOUT):
    """Run ``git -C repo_dir <args>`` and return the CompletedProcess.

    Never raises on a non-zero exit (``check=False``); callers inspect
    ``returncode``/``stdout`` themselves.
    """
    return subprocess.run(
        ["git", "-C", repo_dir, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def git_commit(repo_dir):
    """Short git commit SHA of repo_dir, or '' if unavailable (frozen build / no git)."""
    try:
        if not shutil.which("git") or not os.path.isdir(os.path.join(repo_dir, ".git")):
            return ""
        r = _git(repo_dir, "rev-parse", "--short", "HEAD")
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _requirements_touched(repo_dir, before, after):
    """True if the pulled range changed a requirements file or install script."""
    diff = _git(repo_dir, "diff", "--name-only", f"{before}..{after}")
    if diff.returncode != 0:
        return False
    for line in diff.stdout.splitlines():
        name = line.strip()
        if name.startswith("requirements") or name in ("install.sh", "pyproject.toml"):
            return True
    return False


def _sync_deps(repo_dir):
    """Best-effort dependency sync after a pull that changed requirements.

    The venv is uv-managed and has no pip (see AGENTS.md), so use ``uv pip``.
    Failures are logged and swallowed — a dep mismatch should not wedge startup.
    """
    req = os.path.join(repo_dir, "requirements.txt")
    if not os.path.isfile(req):
        return
    try:
        r = subprocess.run(
            ["uv", "pip", "install", "-r", req],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if r.returncode != 0:
            log.warning("[self-update] dependency sync failed: %s", (r.stderr or r.stdout).strip())
        else:
            log.info("[self-update] dependencies synced after update")
    except Exception as e:  # noqa: BLE001 - never let dep sync crash the caller
        log.warning("[self-update] dependency sync error: %s", e)


def git_self_update(repo_dir):
    """Fast-forward ``repo_dir`` to its upstream branch, non-destructively.

    Returns ``(updated, reason)``:
      * ``(True,  "fast-forwarded")``      HEAD advanced; caller should restart.
      * ``(False, "up-to-date")``          already current.
      * ``(False, "not-a-git-checkout")``  git missing or no ``.git``.
      * ``(False, "dirty-worktree")``      uncommitted/untracked changes present.
      * ``(False, "not-fast-forwardable")``diverged or ahead (unpushed commits).
      * ``(False, "error")``               a git command failed unexpectedly.
    """
    try:
        if not shutil.which("git") or not os.path.isdir(os.path.join(repo_dir, ".git")):
            return False, "not-a-git-checkout"

        # Protect developer work: never update a dirty tree (tracked or untracked).
        status = _git(repo_dir, "status", "--porcelain")
        if status.returncode != 0:
            log.warning("[self-update] git status failed: %s", status.stderr.strip())
            return False, "error"
        if status.stdout.strip():
            return False, "dirty-worktree"

        before = _git(repo_dir, "rev-parse", "HEAD")
        if before.returncode != 0:
            return False, "error"
        before_sha = before.stdout.strip()

        # --ff-only refuses to pull a diverged/ahead branch, leaving HEAD untouched.
        pull = _git(repo_dir, "pull", "--ff-only", "--quiet")

        after = _git(repo_dir, "rev-parse", "HEAD")
        if after.returncode != 0:
            return False, "error"
        after_sha = after.stdout.strip()

        if after_sha != before_sha:
            if _requirements_touched(repo_dir, before_sha, after_sha):
                _sync_deps(repo_dir)
            log.info("[self-update] fast-forwarded %s -> %s", before_sha[:9], after_sha[:9])
            return True, "fast-forwarded"

        # HEAD unchanged: either already current, or the pull was rejected
        # (diverged/ahead). Distinguish the two by the pull's exit status.
        if pull.returncode != 0:
            return False, "not-fast-forwardable"
        return False, "up-to-date"
    except subprocess.TimeoutExpired:
        log.warning("[self-update] git command timed out")
        return False, "error"
    except Exception as e:  # noqa: BLE001 - self-update must never crash the server
        log.warning("[self-update] unexpected error: %s", e)
        return False, "error"


def restart_via_execv():
    """Replace the current process image with a fresh interpreter run.

    Python does not hot-reload source, so after a successful update the process
    must re-exec to pick up new code. Mirrors the primitive already used by the
    ``/api/restart`` route in speech_to_text.py. Any multiprocessing children
    must be terminated by the caller first — ``execv`` would orphan them.
    """
    log.info("[self-update] restarting to load updated code")
    os.execv(sys.executable, [sys.executable] + sys.argv)
