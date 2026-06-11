"""Shared helpers for the STT test suite.

speech_to_text.py loads ML libraries, creates the Flask app, and starts
background threads at import time, so it cannot be imported in tests.
Instead, tests extract individual top-level functions/classes from its AST
and exec them against a stub global namespace. watchdog.py imports cleanly
and is tested as a normal module.
"""

import ast
import os
import shutil
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def extract_definitions(source_file, names, extra_globals=None):
    """Exec selected top-level def/class blocks from `source_file`.

    Returns the namespace dict containing the executed definitions.
    Extracted functions keep their source line behavior (local imports etc.)
    but resolve module globals from the stub namespace, which tests control.
    """
    src = (REPO / source_file).read_text()
    tree = ast.parse(src)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            found[node.name] = ast.get_source_segment(src, node)

    missing = set(names) - set(found)
    assert not missing, f"definitions not found in {source_file}: {missing}"

    ns = {
        "os": os,
        "shutil": shutil,
        "time": time,
        "threading": threading,
        "print": print,
    }
    if extra_globals:
        ns.update(extra_globals)
    exec("\n\n".join(found[n] for n in names), ns)
    return ns
