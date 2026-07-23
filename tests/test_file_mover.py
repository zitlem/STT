"""Local-path logic of stt/file_mover.py (SMB paths are exercised only up to
the is_smb_path branch — network operations are out of unit-test scope)."""

import os
import sys
import types

import pytest

from stt.file_mover import (
    cleanup_empty_directories,
    execute_file_move,
    execute_file_move_now,
    find_files_to_move,
    get_base_directories_from_patterns,
    get_base_directory_for_file,
    is_smb_path,
    move_file_with_structure,
)


class TestIsSmbPath:
    def test_unc_forward_and_back(self):
        assert is_smb_path("//server/share")
        assert is_smb_path("\\\\server\\share")

    def test_local_paths(self):
        assert not is_smb_path("/mnt/backup")
        assert not is_smb_path("C:/backup")
        assert not is_smb_path("relative/path")


class TestGetBaseDirectories:
    def test_stops_at_first_glob(self, tmp_path):
        dirs = get_base_directories_from_patterns(["_AUTOMATIC_BACKUP/**/*.wav"], str(tmp_path))
        assert dirs == {os.path.abspath(str(tmp_path / "_AUTOMATIC_BACKUP"))}

    def test_multiple_patterns(self, tmp_path):
        dirs = get_base_directories_from_patterns(["backup/**/*.db", "logs/*.txt"], str(tmp_path))
        assert dirs == {str(tmp_path / "backup"), str(tmp_path / "logs")}

    def test_glob_char_classes_stop_the_base(self, tmp_path):
        dirs = get_base_directories_from_patterns(["a/[bc]/d.txt", "e/f?.txt"], str(tmp_path))
        assert dirs == {str(tmp_path / "a"), str(tmp_path / "e")}

    def test_no_base_falls_back_to_working_dir(self, tmp_path):
        dirs = get_base_directories_from_patterns(["*.wav"], str(tmp_path))
        assert dirs == {os.path.abspath(str(tmp_path))}

    def test_empty_patterns(self, tmp_path):
        assert get_base_directories_from_patterns([], str(tmp_path)) == {os.path.abspath(str(tmp_path))}


class TestGetBaseDirectoryForFile:
    def test_finds_owning_base(self, tmp_path):
        bases = {str(tmp_path / "backup"), str(tmp_path / "logs")}
        f = str(tmp_path / "backup" / "2026" / "a.db")
        assert get_base_directory_for_file(f, bases) == str(tmp_path / "backup")

    def test_no_match_returns_none(self, tmp_path):
        assert get_base_directory_for_file(str(tmp_path / "other" / "a.db"), {str(tmp_path / "backup")}) is None

    def test_base_itself_is_not_under_base(self, tmp_path):
        # A path equal to the base dir (no trailing separator content) doesn't match
        assert get_base_directory_for_file(str(tmp_path / "backup"), {str(tmp_path / "backup")}) is None


class TestCleanupEmptyDirectories:
    def test_removes_empty_chain_up_to_base(self, tmp_path):
        base = tmp_path / "backup"
        deep = base / "2026" / "07" / "23"
        deep.mkdir(parents=True)
        cleanup_empty_directories(str(deep / "moved.db"), str(base))
        assert base.exists()
        assert not (base / "2026").exists()

    def test_stops_at_non_empty_dir(self, tmp_path):
        base = tmp_path / "backup"
        deep = base / "2026" / "07"
        deep.mkdir(parents=True)
        (base / "2026" / "keep.txt").write_text("x")
        cleanup_empty_directories(str(deep / "moved.db"), str(base))
        assert not deep.exists()
        assert (base / "2026").exists()

    def test_never_removes_base_dir(self, tmp_path):
        base = tmp_path / "backup"
        base.mkdir()
        cleanup_empty_directories(str(base / "moved.db"), str(base))
        assert base.exists()

    def test_file_outside_base_untouched(self, tmp_path):
        other = tmp_path / "other" / "sub"
        other.mkdir(parents=True)
        cleanup_empty_directories(str(other / "f.db"), str(tmp_path / "backup"))
        assert other.exists()


class TestFindFilesToMove:
    def test_recursive_glob_matches_files_only(self, tmp_path):
        (tmp_path / "backup" / "2026").mkdir(parents=True)
        (tmp_path / "backup" / "2026" / "a.db").write_text("x")
        (tmp_path / "backup" / "top.db").write_text("x")
        (tmp_path / "backup" / "skip.txt").write_text("x")
        found = find_files_to_move(["backup/**/*.db"], str(tmp_path))
        assert sorted(os.path.basename(f) for f in found) == ["a.db", "top.db"]

    def test_no_matches(self, tmp_path):
        assert find_files_to_move(["backup/**/*.db"], str(tmp_path)) == []


class TestMoveFileWithStructure:
    def test_move_preserving_structure(self, tmp_path):
        work = tmp_path / "work"
        (work / "backup" / "2026").mkdir(parents=True)
        src = work / "backup" / "2026" / "a.db"
        src.write_text("data")
        dest = tmp_path / "dest"
        ok, err = move_file_with_structure(str(src), str(dest), str(work))
        assert (ok, err) == (True, None)
        assert (dest / "backup" / "2026" / "a.db").read_text() == "data"
        assert not src.exists()

    def test_flat_copy_keeps_source(self, tmp_path):
        work = tmp_path / "work"
        (work / "backup").mkdir(parents=True)
        src = work / "backup" / "a.db"
        src.write_text("data")
        dest = tmp_path / "dest"
        ok, _ = move_file_with_structure(str(src), str(dest), str(work), preserve_structure=False, delete_source=False)
        assert ok is True
        assert (dest / "a.db").exists()
        assert src.exists()

    def test_missing_source_reports_error(self, tmp_path):
        ok, err = move_file_with_structure(str(tmp_path / "nope.db"), str(tmp_path / "dest"), str(tmp_path))
        assert ok is False
        assert err


@pytest.fixture
def app_dir(tmp_path, monkeypatch):
    """Point execute_file_move's working dir (speech_to_text.APP_DIR) at tmp_path."""
    monkeypatch.setitem(sys.modules, "speech_to_text", types.SimpleNamespace(APP_DIR=str(tmp_path)))
    return tmp_path


def mover_config(dest, patterns, **overrides):
    cfg = {"destination_path": dest, "source_patterns": patterns}
    cfg.update(overrides)
    return {"file_manager": {"file_mover": cfg}}


class TestExecuteFileMove:
    def test_no_destination_configured(self, app_dir):
        result = execute_file_move(lambda: mover_config("  ", []))
        assert result["success"] is False
        assert result["errors"] == ["No destination path configured"]

    def test_end_to_end_local_move_with_cleanup(self, app_dir):
        (app_dir / "backup" / "2026" / "07").mkdir(parents=True)
        (app_dir / "backup" / "2026" / "07" / "a.db").write_text("data")
        (app_dir / "backup" / "b.db").write_text("data2")
        dest = app_dir / "dest"
        result = execute_file_move(lambda: mover_config(str(dest), ["backup/**/*.db"]))
        assert result["success"] is True
        assert (result["moved"], result["failed"]) == (2, 0)
        assert (dest / "backup" / "2026" / "07" / "a.db").exists()
        assert (dest / "backup" / "b.db").exists()
        # Source deleted and its emptied subdirs pruned back to the base dir
        assert (app_dir / "backup").exists()
        assert not (app_dir / "backup" / "2026").exists()

    def test_copy_mode_keeps_sources(self, app_dir):
        (app_dir / "backup").mkdir()
        (app_dir / "backup" / "a.db").write_text("data")
        dest = app_dir / "dest"
        result = execute_file_move(lambda: mover_config(str(dest), ["backup/*.db"], delete_source=False))
        assert result["success"] is True and result["moved"] == 1
        assert (app_dir / "backup" / "a.db").exists()

    def test_no_matching_files_succeeds_with_zero(self, app_dir):
        result = execute_file_move(lambda: mover_config(str(app_dir / "dest"), ["backup/**/*.db"]))
        assert result["success"] is True
        assert result["moved"] == 0

    def test_config_getter_failure_fails_closed(self, app_dir):
        def boom():
            raise RuntimeError("config unavailable")
        result = execute_file_move(boom)
        assert result["success"] is False
        assert "config unavailable" in result["errors"][0]

    def test_execute_now_passthrough(self, app_dir):
        result = execute_file_move_now(lambda: mover_config("", []))
        assert result["success"] is False
