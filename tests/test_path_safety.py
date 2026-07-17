"""Path containment / input validation logic used by the model routes.

These replicate the exact expressions used in speech_to_text.py
(cancel_download containment, upload_local_model name validation) so a
regression in the logic shape is caught even though the route functions
themselves can't be imported.
"""

import os
import re


MODELS_DIR = "/srv/stt/models"


def safe_model_path(base_dir, name):
    # mirrors safe_model_path (speech_to_text.py) — the shared guard used by
    # cancel_download and every remove_* route
    if not name or not isinstance(name, str):
        return None
    candidate = name.replace("\\", "/")
    base = os.path.normpath(base_dir)
    model_path = os.path.normpath(os.path.join(base, candidate))
    if model_path == base:
        return None
    try:
        if os.path.commonpath([model_path, base]) != base:
            return None
    except ValueError:
        return None
    return model_path


def cancel_path_allowed(model_id, models_dir=MODELS_DIR):
    # cancel_download replaces "/" with "--" before calling the guard
    dir_name = model_id.replace("/", "--")
    return safe_model_path(models_dir, dir_name) is not None


def model_name_allowed(name):
    # mirrors upload_local_model (speech_to_text.py)
    return bool(re.fullmatch(r"[\w.\- ]+", name)) and name.strip(". ") != ""


def repo_filename_allowed(filename, local_dir="/srv/stt/models/x"):
    # mirrors download_hf_repo_files containment check
    local_root = os.path.abspath(local_dir)
    dest_path = os.path.abspath(os.path.join(local_root, filename))
    return dest_path.startswith(local_root + os.sep)


def test_cancel_normal_ids_allowed():
    assert cancel_path_allowed("whisper-small.en")
    assert cancel_path_allowed("facebook/nllb-200-distilled-600M")
    assert cancel_path_allowed("tts-en_US-amy-medium")


def test_cancel_traversal_blocked():
    # "/" is replaced before joining, so slash payloads become literal names;
    # bare dot segments are the dangerous remainder
    assert not cancel_path_allowed("..")
    assert not cancel_path_allowed(".")


def test_model_name_validation():
    assert model_name_allowed("my-model_v1.0")
    assert model_name_allowed("nllb 600M")
    assert not model_name_allowed("../evil")
    assert not model_name_allowed("a/b")
    assert not model_name_allowed("..")
    assert not model_name_allowed("...")
    assert not model_name_allowed("")


def test_repo_filename_containment():
    assert repo_filename_allowed("model.bin")
    assert repo_filename_allowed("sub/dir/config.json")
    assert not repo_filename_allowed("../outside.bin")
    assert not repo_filename_allowed("/etc/passwd")


def test_safe_model_path_normal_names():
    # the remove_* routes build names like these
    assert safe_model_path(MODELS_DIR, "whisper-small.en")
    assert safe_model_path(MODELS_DIR, "faster-whisper-base.en")
    assert safe_model_path(MODELS_DIR, "facebook--nllb-200-distilled-600M")


def safe_managed_path(path, base_dir):
    # mirrors safe_managed_path (speech_to_text.py) — file-manager confinement
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


def test_safe_managed_path_confinement(tmp_path):
    base = str(tmp_path)
    inside = tmp_path / "_AUTOMATIC_BACKUP"
    inside.mkdir()
    sibling = tmp_path.parent / (tmp_path.name + "_secrets")  # shares the prefix
    sibling.mkdir()

    assert safe_managed_path(str(inside), base) is not None
    assert safe_managed_path("", base) == os.path.realpath(base)
    # escapes must be rejected
    assert safe_managed_path(os.path.join(base, ".."), base) is None
    assert safe_managed_path("/etc/passwd", base) is None
    assert safe_managed_path("../../etc", base) is None
    # a sibling dir sharing the name prefix must NOT pass (the old startswith bug)
    assert safe_managed_path(str(sibling), base) is None


def test_safe_model_path_traversal_blocked():
    for payload in (
        "..",
        "../evil",
        "../../../../tmp/victim",
        "faster-whisper-../../../../tmp/x",
        "whisper-../../../etc",
        "/etc/passwd",
        "..\\..\\windows",  # backslash variant (Windows-style)
        "",
        None,
    ):
        assert safe_model_path(MODELS_DIR, payload) is None, payload
