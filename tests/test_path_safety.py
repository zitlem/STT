"""Path containment / input validation logic used by the model routes.

These replicate the exact expressions used in speech_to_text.py
(cancel_download containment, upload_local_model name validation) so a
regression in the logic shape is caught even though the route functions
themselves can't be imported.
"""

import os
import re


MODELS_DIR = "/srv/stt/models"


def cancel_path_allowed(model_id, models_dir=MODELS_DIR):
    # mirrors cancel_download (speech_to_text.py)
    dir_name = model_id.replace("/", "--")
    model_path = os.path.normpath(os.path.join(models_dir, dir_name))
    return not (
        model_path == models_dir
        or os.path.commonpath([model_path, models_dir]) != models_dir
    )


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
