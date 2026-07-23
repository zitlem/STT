"""Path-containment guards for user-supplied names and file-manager paths.

Extracted from speech_to_text.py so they can be imported (and unit-tested)
without the monolith's import-time side effects. Stdlib-only; base
directories are passed in explicitly (a thin wrapper in speech_to_text.py
supplies APP_DIR as the file-manager default).
"""

import os
from typing import Optional


def safe_model_path(base_dir: str, name: Optional[str]) -> Optional[str]:
    """Resolve a user-supplied model name against base_dir and return the
    absolute path only if it stays strictly inside base_dir. Returns None for
    traversal payloads ('..', '../x', absolute paths, backslash variants).

    Used by every route that rmtree's a model directory built from request
    input — see cancel_download / remove_* handlers.
    """
    if not name or not isinstance(name, str):
        return None
    # Normalize slash variants so a Windows-style '..\\' can't slip past.
    candidate = name.replace("\\", "/")
    base = os.path.normpath(base_dir)
    model_path = os.path.normpath(os.path.join(base, candidate))
    if model_path == base:
        return None
    try:
        if os.path.commonpath([model_path, base]) != base:
            return None
    except ValueError:
        # Different drives (Windows) or mixed absolute/relative — reject.
        return None
    return model_path


def safe_managed_path(path: Optional[str], base_dir: str) -> Optional[str]:
    """Resolve a file-manager path and return its realpath only if it stays
    inside base_dir. Resolves symlinks so a symlink inside the tree can't
    escape it. Returns None on any escape.
    """
    base = os.path.realpath(base_dir)
    if not path:
        return base
    target = os.path.realpath(path if os.path.isabs(path) else os.path.join(base, path))
    if target == base:
        return target
    try:
        if os.path.commonpath([target, base]) != base:
            return None
    except ValueError:
        return None
    return target
