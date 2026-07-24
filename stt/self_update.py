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

import hashlib
import logging
import os
import shutil
import subprocess
import sys
from typing import Tuple

log = logging.getLogger(__name__)

# Timeouts (seconds) for the individual git invocations.
_GIT_TIMEOUT = 60

# The server runs windowless on Windows; without this every git/uv child
# would flash a console window. 0 off-Windows: safe to pass unconditionally.
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _git(repo_dir: str, *args: str, timeout: int = _GIT_TIMEOUT) -> "subprocess.CompletedProcess[str]":
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
        creationflags=_CREATE_NO_WINDOW,
    )


def git_commit(repo_dir: str) -> str:
    """Short git commit SHA of repo_dir, or '' if unavailable (frozen build / no git)."""
    try:
        if not shutil.which("git") or not os.path.isdir(os.path.join(repo_dir, ".git")):
            return ""
        r = _git(repo_dir, "rev-parse", "--short", "HEAD")
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def git_describe(repo_dir: str) -> str:
    """`git describe --tags --always` of repo_dir (e.g. '26.1.2-9-gc588d29'),
    or '' if unavailable (frozen build / no git)."""
    try:
        if not shutil.which("git") or not os.path.isdir(os.path.join(repo_dir, ".git")):
            return ""
        r = _git(repo_dir, "describe", "--tags", "--always")
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _requirements_hash(repo_dir: str) -> str:
    """sha256 of requirements.txt, or '' if it doesn't exist / can't be read."""
    req = os.path.join(repo_dir, "requirements.txt")
    try:
        with open(req, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return ""


def _sync_marker_path(repo_dir: str) -> str:
    # Lives inside the gitignored .venv so it never dirties the worktree
    # (a dirty tree blocks future updates) and dies with the venv it describes.
    return os.path.join(repo_dir, ".venv", ".requirements-synced")


def _sync_deps(repo_dir: str) -> bool:
    """Best-effort dependency sync; returns True on success.

    The venv is uv-managed and has no pip (see AGENTS.md), so use ``uv pip``.
    Failures are logged and swallowed — a dep mismatch should not wedge startup.
    """
    req = os.path.join(repo_dir, "requirements.txt")
    if not os.path.isfile(req):
        return False
    try:
        r = subprocess.run(
            ["uv", "pip", "install", "-r", req],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            creationflags=_CREATE_NO_WINDOW,
        )
        if r.returncode != 0:
            log.warning("[self-update] dependency sync failed: %s", (r.stderr or r.stdout).strip())
            return False
        log.info("[self-update] dependencies synced with requirements.txt")
        return True
    except Exception as e:  # noqa: BLE001 - never let dep sync crash the caller
        log.warning("[self-update] dependency sync error: %s", e)
        return False


def _sync_deps_if_needed(repo_dir: str) -> None:
    """Sync the venv whenever requirements.txt differs from the last synced state.

    Compares against a hash marker instead of the pulled diff, so a box that
    missed a sync (crash, uv failure, venv rebuilt, requirements added before
    this feature existed) heals on the next update check rather than waiting
    for another commit that happens to touch the requirements files. Newly
    installed packages take effect on the next restart.
    """
    current = _requirements_hash(repo_dir)
    if not current:
        return  # no requirements.txt (e.g. tests, stripped deploys)
    marker = _sync_marker_path(repo_dir)
    try:
        with open(marker, encoding="utf-8") as f:
            if f.read().strip() == current:
                return
    except OSError:
        pass  # no marker yet -> treat as out of sync
    if _sync_deps(repo_dir) and os.path.isdir(os.path.dirname(marker)):
        try:
            with open(marker, "w", encoding="utf-8") as f:
                f.write(current + "\n")
        except OSError as e:
            log.warning("[self-update] could not write sync marker: %s", e)


def git_self_update(repo_dir: str) -> Tuple[bool, str]:
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
            _sync_deps_if_needed(repo_dir)
            log.info("[self-update] fast-forwarded %s -> %s", before_sha[:9], after_sha[:9])
            return True, "fast-forwarded"

        # HEAD unchanged: either already current, or the pull was rejected
        # (diverged/ahead). Distinguish the two by the pull's exit status.
        if pull.returncode != 0:
            return False, "not-fast-forwardable"
        # Heal dependency drift even with nothing to pull, so a box that
        # missed a sync doesn't stay broken until the next requirements commit.
        _sync_deps_if_needed(repo_dir)
        return False, "up-to-date"
    except subprocess.TimeoutExpired:
        log.warning("[self-update] git command timed out")
        return False, "error"
    except Exception as e:  # noqa: BLE001 - self-update must never crash the server
        log.warning("[self-update] unexpected error: %s", e)
        return False, "error"


def restart_via_execv() -> None:
    """Replace the current process image with a fresh interpreter run.

    Python does not hot-reload source, so after a successful update the process
    must re-exec to pick up new code. Any multiprocessing children must be
    terminated by the caller first — ``execv`` would orphan them, and a child
    forked after the web server bound its port keeps that socket alive across
    the exec, so the re-exec'd server dies with EADDRINUSE. Only safe before
    the web server and workers exist (the startup update path); once they're
    running, use ``perform_server_restart()`` in speech_to_text.py instead.
    """
    log.info("[self-update] restarting to load updated code")
    os.execv(sys.executable, [sys.executable, *sys.argv])
