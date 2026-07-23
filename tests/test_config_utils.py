"""Config persistence, upload validation, and version helpers (stt/config_utils.py)."""

import json
import os

import pytest

from stt.config_utils import (
    _atomic_write_json,
    _merge_missing_keys,
    compute_display_version,
    restore_config_from_template,
    validate_file,
)


class TestAtomicWriteJson:
    def test_writes_valid_json(self, tmp_path):
        path = tmp_path / "cfg.json"
        _atomic_write_json(str(path), {"a": 1, "b": {"c": 2}})
        assert json.loads(path.read_text()) == {"a": 1, "b": {"c": 2}}

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / "cfg.json"
        path.write_text('{"old": true}')
        _atomic_write_json(str(path), {"new": True})
        assert json.loads(path.read_text()) == {"new": True}

    def test_no_temp_litter_on_success(self, tmp_path):
        _atomic_write_json(str(tmp_path / "cfg.json"), {"a": 1})
        assert sorted(p.name for p in tmp_path.iterdir()) == ["cfg.json"]

    def test_failure_leaves_original_and_no_litter(self, tmp_path):
        path = tmp_path / "cfg.json"
        path.write_text('{"old": true}')
        with pytest.raises(TypeError):
            _atomic_write_json(str(path), {"bad": object()})
        assert json.loads(path.read_text()) == {"old": True}
        assert sorted(p.name for p in tmp_path.iterdir()) == ["cfg.json"]

    def test_ensure_ascii_toggle(self, tmp_path):
        path = tmp_path / "cfg.json"
        _atomic_write_json(str(path), {"w": "señor"}, ensure_ascii=False)
        assert "señor" in path.read_text()
        _atomic_write_json(str(path), {"w": "señor"})
        assert "se\\u00f1or" in path.read_text()


class TestMergeMissingKeys:
    def test_adds_missing_keys(self):
        dst = {"a": 1}
        assert _merge_missing_keys(dst, {"a": 9, "b": 2}) is True
        assert dst == {"a": 1, "b": 2}

    def test_never_overwrites_existing(self):
        dst = {"a": "user-set", "nested": {"x": False}}
        _merge_missing_keys(dst, {"a": "template", "nested": {"x": True, "y": 1}})
        assert dst["a"] == "user-set"
        assert dst["nested"] == {"x": False, "y": 1}

    def test_type_mismatch_left_alone(self):
        # User set a scalar where the template has a dict: keep the user's value
        dst = {"a": 5}
        assert _merge_missing_keys(dst, {"a": {"sub": 1}}) is False
        assert dst == {"a": 5}

    def test_no_change_returns_false(self):
        dst = {"a": 1, "b": {"c": 2}}
        assert _merge_missing_keys(dst, {"a": 0, "b": {"c": 9}}) is False

    def test_added_values_are_deep_copies(self):
        src = {"b": {"list": [1, 2]}}
        dst = {}
        _merge_missing_keys(dst, src)
        dst["b"]["list"].append(3)
        assert src["b"]["list"] == [1, 2]


class TestRestoreConfigFromTemplate:
    def test_copies_template(self, tmp_path):
        template = tmp_path / "config.default.json"
        template.write_text('{"fresh": true}')
        target = tmp_path / "config.json"
        assert restore_config_from_template(str(template), str(target)) is True
        assert json.loads(target.read_text()) == {"fresh": True}

    def test_missing_template_returns_false(self, tmp_path):
        target = tmp_path / "config.json"
        assert restore_config_from_template(str(tmp_path / "nope.json"), str(target)) is False
        assert not target.exists()


class FakeUpload:
    def __init__(self, filename):
        self.filename = filename


class TestValidateFile:
    def test_no_file(self):
        assert validate_file(None) == (False, "No file selected")
        assert validate_file(FakeUpload("")) == (False, "No file selected")

    def test_supported_audio_and_video(self):
        assert validate_file(FakeUpload("sermon.mp3")) == (True, None)
        assert validate_file(FakeUpload("Service.MP4")) == (True, None)

    def test_unsupported_extension(self):
        ok, err = validate_file(FakeUpload("notes.txt"))
        assert ok is False
        assert "txt" in err

    def test_no_extension(self):
        ok, _err = validate_file(FakeUpload("plainfile"))
        assert ok is False


class TestComputeDisplayVersion:
    def test_commits_since_tag_folded_into_patch(self):
        assert compute_display_version("26.1.2-17-g398f75e", "398f75e", "26.1.2") == "26.1.19-398f75e"

    def test_exact_tag_passthrough(self):
        assert compute_display_version("26.1.3", "abc1234", "26.1.3") == "26.1.3"

    def test_non_semver_describe_passthrough(self):
        assert compute_display_version("v-weird-tag", "abc", "1.0") == "v-weird-tag"

    def test_frozen_build_with_commit(self):
        assert compute_display_version("", "abc1234", "26.1.2") == "26.1.2-abc1234"

    def test_fallback_to_version_file(self):
        assert compute_display_version("", "", "26.1.2") == "26.1.2"

    def test_monotonic_across_a_release(self):
        # one commit after the 26.1.3 tag must sort above the tag itself
        assert compute_display_version("26.1.3-1-gaaaa111", "", "x") == "26.1.4-aaaa111"
