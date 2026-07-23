"""Config persistence, upload validation, and version-string helpers.

Extracted from speech_to_text.py so they can be imported (and unit-tested)
without the monolith's import-time side effects. Stdlib-only. Paths and
version strings are passed in explicitly; thin wrappers in speech_to_text.py
supply the live values.
"""

import json
import os
import re
import shutil
import tempfile
from typing import Any, Optional, Tuple

SUPPORTED_AUDIO_FORMATS = ["mp3", "wav", "flac", "ogg", "m4a", "aac", "wma", "opus"]
SUPPORTED_VIDEO_FORMATS = [
    "mp4",
    "mkv",
    "avi",
    "mov",
    "wmv",
    "flv",
    "webm",
    "mpg",
    "mpeg",
    "m4v",
    "3gp",
]


def _atomic_write_json(path: str, data: Any, *, ensure_ascii: bool = True) -> None:
    """Write JSON to ``path`` atomically.

    Dumps to a temp file in the same directory, flushes+fsyncs it, then
    os.replace()s it into place. os.replace is atomic on POSIX and Windows, so
    a crash or concurrent reader never sees a truncated/half-written file.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=ensure_ascii)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _merge_missing_keys(dst: dict, src: dict) -> bool:
    """Recursively add keys present in src but absent in dst. Returns True if dst changed.

    Existing values are never overwritten — a key the user has set (even to a
    different type than the template) is left exactly as-is; only genuinely
    missing keys are filled in."""
    changed = False
    for key, value in src.items():
        if key not in dst:
            dst[key] = json.loads(json.dumps(value))  # deep copy via round-trip
            changed = True
        elif isinstance(dst[key], dict) and isinstance(value, dict):
            if _merge_missing_keys(dst[key], value):
                changed = True
    return changed


def restore_config_from_template(template_file: str, config_file: str, reason: str = "") -> bool:
    """Copy the bundled config.default.json over config_file. Returns True on success."""
    if not os.path.exists(template_file):
        print(f"[CONFIG] ERROR: template '{template_file}' is missing; cannot {reason or 'restore config'}.")
        return False
    shutil.copy2(template_file, config_file)
    return True


def validate_file(file: Any) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded file.

    Args:
        file: Flask file object

    Returns:
        (is_valid, error_message) tuple
    """
    if not file or file.filename == "":
        return False, "No file selected"

    # Check file extension
    ext = os.path.splitext(file.filename)[1].lower().replace(".", "")
    if ext not in SUPPORTED_AUDIO_FORMATS + SUPPORTED_VIDEO_FORMATS:
        return False, f"Unsupported file format: {ext}"

    return True, None


def compute_display_version(describe: str, commit: str, version: str) -> str:
    """Single monotonic version string for scripts and the UI.

    Folds git describe's commits-since-tag count into the patch number:
    '26.1.2-17-g398f75e' -> '26.1.19-398f75e'. The number only ever moves
    forward (a later 26.1.3 tag shows 26.1.3, then 26.1.4-... one commit on),
    and the -<hash> suffix distinguishes it from any real future tag. The 'g'
    that git describe prefixes onto the hash is stripped for display.

    describe/commit/version: git describe output, exact commit hash, and the
    VERSION-file string for the running checkout (any may be empty/unknown).
    """
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)-(\d+)-g([0-9a-f]+)", describe or "")
    if m:
        return f"{m.group(1)}.{m.group(2)}.{int(m.group(3)) + int(m.group(4))}-{m.group(5)}"
    if describe:
        return describe  # exact tag, non-semver tag, or bare hash
    if commit:
        return f"{version}-{commit}"  # frozen build with known commit
    return version
