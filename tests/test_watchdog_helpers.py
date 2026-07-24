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


def _exe_rel():
    """Interpreter path inside a store dir, per platform."""
    if watchdog.IS_WINDOWS:
        return "python.exe"
    return os.path.join("bin", "python3")


def _make_store(root, versions, minor_link_decoy=False):
    """Build a fake uv Python store: one patch-versioned dir per version, each
    holding a dummy interpreter file. Optionally a dir named like the
    minor-version *link* (no patch component) that must never be selected."""
    root.mkdir(parents=True, exist_ok=True)
    for v in versions:
        exe = root / f"cpython-{v}-windows-x86_64-none" / _exe_rel()
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_text(f"fake python {v}")
    if minor_link_decoy:
        exe = root / f"cpython-{watchdog.UV_PYTHON_VERSION}-windows-x86_64-none" / _exe_rel()
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_text("minor-version link decoy")
    return root


class TestStepPythonUntrustedMountFallback:
    """'Untrusted mount point' (os error 448): OneDrive's minifilter / Win11
    24H2 hardening refuses the junction uv creates as the store's minor-version
    link (astral-sh/uv#19616) — extraction usually succeeded, only the link
    step died, so relocating the store doesn't recover (seen in the field).
    _step_python's ladder: install → salvage the extracted interpreter out of
    the default store → install into a relocated store → salvage from that.
    _step_venv pins the salvaged exe (or reuses the relocated-store env)."""

    ERR_448 = watchdog.ProvisionError(
        "command failed (2): uv python install 3.11 — last output: "
        "error: Failed to create Python minor version link directory | "
        "Caused by: The path cannot be traversed because it contains an "
        "untrusted mount point. (os error 448)"
    )

    def _provisioner(self, run_impl, monkeypatch, tmp_path,
                     default_store=None, fallback_store=None):
        """Provisioner with stubbed _run and store locations isolated to
        tmp_path (the dev machine may have a real uv store)."""
        monkeypatch.setattr(watchdog, "_uv_default_python_store",
                            lambda: str(default_store or tmp_path / "no-default-store"))
        monkeypatch.setattr(watchdog, "UV_PYTHON_FALLBACK_DIR",
                            str(fallback_store or tmp_path / "no-fallback-store"))
        monkeypatch.setattr(watchdog, "UV_SALVAGED_PYTHON_DIR",
                            str(tmp_path / "salvaged"))
        p = watchdog.Provisioner(log=lambda m: None)
        p._uv = "uv"
        p._run = run_impl
        return p

    @staticmethod
    def _fake_run(calls, install_results):
        """`uv python install` pops the next result (exception → raised,
        else returned); anything else (the salvage `--version` probe) → 0."""
        def run(cmd, desc=None, check=True, timeout=3600, extra_env=None):
            if "install" in cmd:
                calls.append((list(cmd), extra_env))
                result = install_results.pop(0)
                if isinstance(result, Exception):
                    raise result
                return result
            return 0
        return run

    def test_retries_with_relocated_store_when_nothing_extracted(self, tmp_path, monkeypatch):
        calls = []
        p = self._provisioner(self._fake_run(calls, [self.ERR_448, 0]),
                              monkeypatch, tmp_path)
        p._step_python()

        assert len(calls) == 2
        assert calls[0][1] is None
        assert calls[1][1] == {"UV_PYTHON_INSTALL_DIR": watchdog.UV_PYTHON_FALLBACK_DIR}
        assert p._uv_env == {"UV_PYTHON_INSTALL_DIR": watchdog.UV_PYTHON_FALLBACK_DIR}
        assert p._uv_python is None

    def test_unrelated_failure_is_not_retried(self, tmp_path, monkeypatch):
        calls = []
        err = watchdog.ProvisionError(
            "command failed (1): uv python install 3.11 — last output: network unreachable")
        p = self._provisioner(self._fake_run(calls, [err]), monkeypatch, tmp_path)
        try:
            p._step_python()
            raise AssertionError("expected ProvisionError")
        except watchdog.ProvisionError:
            pass
        assert len(calls) == 1
        assert p._uv_env is None

    def test_salvages_extracted_interpreter_from_default_store(self, tmp_path, monkeypatch):
        # The field case: extraction succeeded, only the link step failed.
        # No retry should happen — the interpreter is copied out and used.
        store = _make_store(tmp_path / "store", ["3.11.9", "3.11.15"],
                            minor_link_decoy=True)
        calls = []
        p = self._provisioner(self._fake_run(calls, [self.ERR_448]),
                              monkeypatch, tmp_path, default_store=store)
        p._step_python()

        assert len(calls) == 1  # no pointless re-download into a second store
        assert p._uv_env is None
        # Newest patch wins numerically (3.11.15 > 3.11.9 despite lexicographic
        # order), and the copy lives outside the store.
        expected = os.path.join(str(tmp_path / "salvaged"),
                                "cpython-3.11.15-windows-x86_64-none", _exe_rel())
        assert p._uv_python == expected
        assert os.path.isfile(expected)

    def test_minor_link_name_is_never_selected(self, tmp_path, monkeypatch):
        # Only the link-named dir exists (no patch-versioned dir): nothing to
        # salvage — must fall through to the relocated-store retry.
        store = _make_store(tmp_path / "store", [], minor_link_decoy=True)
        calls = []
        p = self._provisioner(self._fake_run(calls, [self.ERR_448, 0]),
                              monkeypatch, tmp_path, default_store=store)
        p._step_python()

        assert len(calls) == 2
        assert p._uv_python is None

    def test_salvages_from_relocated_store_when_both_installs_fail(self, tmp_path, monkeypatch):
        fallback = _make_store(tmp_path / "fallback", ["3.11.15"])
        calls = []
        p = self._provisioner(self._fake_run(calls, [self.ERR_448, self.ERR_448]),
                              monkeypatch, tmp_path, fallback_store=fallback)
        p._step_python()

        assert len(calls) == 2
        assert p._uv_python is not None
        assert os.path.isfile(p._uv_python)

    def test_broken_salvaged_interpreter_falls_through(self, tmp_path, monkeypatch):
        # --version probe fails → salvage rejected → relocated retry runs.
        store = _make_store(tmp_path / "store", ["3.11.15"])
        calls = []

        def run(cmd, desc=None, check=True, timeout=3600, extra_env=None):
            if "install" in cmd:
                calls.append(list(cmd))
                if len(calls) == 1:
                    raise self.ERR_448
                return 0
            return 1  # the probe

        p = self._provisioner(run, monkeypatch, tmp_path, default_store=store)
        p._step_python()

        assert len(calls) == 2
        assert p._uv_python is None
        assert p._uv_env == {"UV_PYTHON_INSTALL_DIR": watchdog.UV_PYTHON_FALLBACK_DIR}

    def test_venv_reuses_fallback_env(self, tmp_path, monkeypatch):
        monkeypatch.setattr(watchdog, "SOURCE_DIR", str(tmp_path))
        seen = {}

        def fake_run(cmd, desc=None, check=True, timeout=3600, extra_env=None):
            seen["cmd"], seen["extra_env"] = list(cmd), extra_env
            return 0

        p = watchdog.Provisioner(log=lambda m: None)
        p._uv = "uv"
        p._run = fake_run
        p._uv_env = {"UV_PYTHON_INSTALL_DIR": watchdog.UV_PYTHON_FALLBACK_DIR}
        p._step_venv()

        assert seen["extra_env"] == {"UV_PYTHON_INSTALL_DIR": watchdog.UV_PYTHON_FALLBACK_DIR}
        assert "venv" in seen["cmd"]
        # No salvage happened → version spec, not a pinned path.
        assert seen["cmd"][-2:] == ["--python", watchdog.UV_PYTHON_VERSION]

    def test_venv_pins_salvaged_interpreter(self, tmp_path, monkeypatch):
        monkeypatch.setattr(watchdog, "SOURCE_DIR", str(tmp_path))
        seen = {}

        def fake_run(cmd, desc=None, check=True, timeout=3600, extra_env=None):
            seen["cmd"] = list(cmd)
            return 0

        p = watchdog.Provisioner(log=lambda m: None)
        p._uv = "uv"
        p._run = fake_run
        p._uv_python = os.path.join("salvaged", "cpython-3.11.15", _exe_rel())
        p._step_venv()

        assert seen["cmd"][-2:] == ["--python", p._uv_python]
