"""Pure/local helpers of stt/watchdog.py: version parsing, crash fingerprints,
log rotation, and the one-time config layout migration."""

import os

from stt import watchdog


class TestParseVersion:
    def test_plain(self):
        assert watchdog.parse_version("1.2.3") == (1, 2, 3)

    def test_v_prefix_stripped(self):
        assert watchdog.parse_version("v26.1.64") == (26, 1, 64)

    def test_non_numeric_parts_become_zero(self):
        assert watchdog.parse_version("1.2.3-rc1") == (1, 2, 0)

    def test_ordering_across_releases(self):
        assert watchdog.parse_version("26.1.9") < watchdog.parse_version("26.1.10")


class TestCrashFingerprint:
    TRACE = (
        'Traceback (most recent call last):\n'
        '  File "speech_to_text.py", line 123, in <module>\n'
        '    main()\n'
        'ValueError: bad audio device\n'
    )

    def test_stable_hash(self):
        assert watchdog._crash_fingerprint(self.TRACE) == watchdog._crash_fingerprint(self.TRACE)
        assert len(watchdog._crash_fingerprint(self.TRACE)) == 12

    def test_line_numbers_do_not_change_fingerprint(self):
        moved = self.TRACE.replace("line 123", "line 456")
        assert watchdog._crash_fingerprint(self.TRACE) == watchdog._crash_fingerprint(moved)

    def test_different_errors_differ(self):
        other = self.TRACE.replace("ValueError: bad audio device", "KeyError: 'model'")
        assert watchdog._crash_fingerprint(self.TRACE) != watchdog._crash_fingerprint(other)

    def test_no_signature_lines_falls_back_to_tail(self):
        assert len(watchdog._crash_fingerprint("plain log output\nno traceback here")) == 12


class TestRotateIfLarge:
    def test_below_threshold_untouched(self, tmp_path):
        log = tmp_path / "stt.log"
        log.write_text("small")
        watchdog._rotate_if_large(str(log), max_bytes=1000, backups=3)
        assert log.read_text() == "small"
        assert not (tmp_path / "stt.log.1").exists()

    def test_rotation_chain(self, tmp_path):
        log = tmp_path / "stt.log"
        log.write_text("current-big-content")
        (tmp_path / "stt.log.1").write_text("older")
        (tmp_path / "stt.log.2").write_text("oldest")
        watchdog._rotate_if_large(str(log), max_bytes=5, backups=3)
        assert not log.exists()
        assert (tmp_path / "stt.log.1").read_text() == "current-big-content"
        assert (tmp_path / "stt.log.2").read_text() == "older"
        assert (tmp_path / "stt.log.3").read_text() == "oldest"

    def test_missing_file_is_noop(self, tmp_path):
        watchdog._rotate_if_large(str(tmp_path / "nope.log"), max_bytes=5, backups=3)


class TestVenvPython:
    def test_explicit_venv_dir(self):
        p = watchdog.venv_python("/some/venv")
        if watchdog.IS_WINDOWS:
            assert p == os.path.join("/some/venv", "Scripts", "python.exe")
        else:
            assert p == os.path.join("/some/venv", "bin", "python3")

    def test_defaults_to_source_venv(self):
        assert watchdog.venv_python().startswith(os.path.join(watchdog.SOURCE_DIR, ".venv"))


class TestMigrateConfigLayout:
    def test_moves_known_files_into_config_dir(self, tmp_path, monkeypatch):
        data = tmp_path
        cfg = tmp_path / "config"
        monkeypatch.setattr(watchdog, "DATA_DIR", str(data))
        monkeypatch.setattr(watchdog, "CONFIG_DIR", str(cfg))
        (data / "config.json").write_text('{"a": 1}')
        (data / "word_highlighting.json").write_text("{}")
        (data / "unrelated.json").write_text("{}")
        watchdog.migrate_config_layout()
        assert (cfg / "config.json").read_text() == '{"a": 1}'
        assert (cfg / "word_highlighting.json").exists()
        assert not (data / "config.json").exists()
        assert (data / "unrelated.json").exists()  # only known names migrate

    def test_existing_destination_never_overwritten(self, tmp_path, monkeypatch):
        data = tmp_path
        cfg = tmp_path / "config"
        cfg.mkdir()
        monkeypatch.setattr(watchdog, "DATA_DIR", str(data))
        monkeypatch.setattr(watchdog, "CONFIG_DIR", str(cfg))
        (data / "config.json").write_text('{"old": true}')
        (cfg / "config.json").write_text('{"live": true}')
        watchdog.migrate_config_layout()
        assert (cfg / "config.json").read_text() == '{"live": true}'
        assert (data / "config.json").exists()  # left in place, nothing to do

    def test_idempotent_on_empty_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(watchdog, "DATA_DIR", str(tmp_path))
        monkeypatch.setattr(watchdog, "CONFIG_DIR", str(tmp_path / "config"))
        watchdog.migrate_config_layout()
        watchdog.migrate_config_layout()
        assert (tmp_path / "config").is_dir()


class TestWriteVersion:
    def test_atomic_write_with_newline(self, tmp_path, monkeypatch):
        vf = tmp_path / "VERSION"
        monkeypatch.setattr(watchdog, "VERSION_FILE", str(vf))
        watchdog.write_version("26.1.64")
        assert vf.read_text() == "26.1.64\n"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["VERSION"]


class TestStepPythonUntrustedMountFallback:
    """Windows 24H2 blocks elevated processes from traversing user-created
    junctions (os error 448) — uv's default Python store can be behind one
    (OneDrive-redirected %APPDATA%). _step_python must retry with the store
    relocated to UV_PYTHON_FALLBACK_DIR, and _step_venv must reuse that env."""

    ERR_448 = watchdog.ProvisionError(
        "command failed (2): uv python install 3.11 — last output: "
        "error: Failed to create Python minor version link directory | "
        "Caused by: The path cannot be traversed because it contains an "
        "untrusted mount point. (os error 448)"
    )

    def _provisioner(self, run_impl):
        p = watchdog.Provisioner(log=lambda m: None)
        p._uv = "uv"
        p._run = run_impl
        return p

    def test_retries_with_relocated_store(self):
        calls = []

        def fake_run(cmd, desc=None, check=True, timeout=3600, extra_env=None):
            calls.append((list(cmd), extra_env))
            if len(calls) == 1:
                raise self.ERR_448
            return 0

        p = self._provisioner(fake_run)
        p._step_python()

        assert len(calls) == 2
        assert calls[0][1] is None
        assert calls[1][1] == {"UV_PYTHON_INSTALL_DIR": watchdog.UV_PYTHON_FALLBACK_DIR}
        assert p._uv_env == {"UV_PYTHON_INSTALL_DIR": watchdog.UV_PYTHON_FALLBACK_DIR}

    def test_unrelated_failure_is_not_retried(self):
        calls = []

        def fake_run(cmd, desc=None, check=True, timeout=3600, extra_env=None):
            calls.append(list(cmd))
            raise watchdog.ProvisionError("command failed (1): uv python install 3.11 — last output: network unreachable")

        p = self._provisioner(fake_run)
        try:
            p._step_python()
            raise AssertionError("expected ProvisionError")
        except watchdog.ProvisionError:
            pass
        assert len(calls) == 1
        assert p._uv_env is None

    def test_venv_reuses_fallback_env(self, tmp_path, monkeypatch):
        monkeypatch.setattr(watchdog, "SOURCE_DIR", str(tmp_path))
        seen = {}

        def fake_run(cmd, desc=None, check=True, timeout=3600, extra_env=None):
            seen["cmd"], seen["extra_env"] = list(cmd), extra_env
            return 0

        p = self._provisioner(fake_run)
        p._uv_env = {"UV_PYTHON_INSTALL_DIR": watchdog.UV_PYTHON_FALLBACK_DIR}
        p._step_venv()

        assert seen["extra_env"] == {"UV_PYTHON_INSTALL_DIR": watchdog.UV_PYTHON_FALLBACK_DIR}
        assert "venv" in seen["cmd"]
