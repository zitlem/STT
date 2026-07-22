"""Tests for the dynamic system-requirements estimate/warning logic."""

import ast
import os
import shutil
import sys
import tempfile

from conftest import REPO, extract_definitions

SOURCE = "speech_to_text.py"

# Fake models dir so the estimate's "translation model downloaded?" check has
# somewhere to look; tests create/remove the NLLB subdir to simulate set-up.
_MODELS_DIR = tempfile.mkdtemp(prefix="stt-test-models-")


def _nllb_dir(nllb="facebook/nllb-200-distilled-600M"):
    return os.path.join(_MODELS_DIR, nllb.replace("/", "--"))


def _extract_constants(names):
    """Literal top-level assignments (constants) from the monolith's AST."""
    src = (REPO / SOURCE).read_text()
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    missing = set(names) - set(out)
    assert not missing, f"constants not found: {missing}"
    return out


def _ns():
    consts = _extract_constants(["BASELINE_RAM_GB", "GPU_HOST_RAM_GB", "BASE_DISK_GB", "MODEL_MEMORY_ESTIMATES"])
    consts["sys"] = sys
    consts["os"] = os
    consts["MODELS_DIR"] = _MODELS_DIR
    return extract_definitions(
        SOURCE,
        ["_normalize_whisper_size", "_estimate_memory_requirements",
         "_format_parts", "_check_system_requirements"],
        extra_globals=consts,
    )


NS = _ns()
BASELINE = NS["BASELINE_RAM_GB"]
GPU_HOST = NS["GPU_HOST_RAM_GB"]
EST = NS["MODEL_MEMORY_ESTIMATES"]

GB = 1024**3

AMPLE_HW = {"ram_bytes": 64 * GB, "cpu_cores": 16, "disk_free_bytes": 500 * GB,
            "vram_bytes": 24 * GB, "has_cuda": True, "apple_silicon": False}


def _cfg(model="small", backend="whisper", use_gpu=False, ft_model=None, ft_gpu=False,
         translation=False, nllb="facebook/nllb-200-distilled-600M", nllb_gpu=False, remote=False):
    cfg = {
        "model": {"type": "whisper", "whisper": {"model": model}, "backend": backend},
        "performance": {"use_gpu": use_gpu},
        "file_transcription": {"model": {"type": "whisper", "whisper": {"model": ft_model or model},
                                         "backend": ""}, "use_gpu": ft_gpu},
        "live_translation": {"enabled": translation, "translation_model": nllb, "use_gpu": nllb_gpu,
                             "remote": {"enabled": remote, "endpoint": "http://x" if remote else ""}},
    }
    return cfg


class TestNormalizeWhisperSize:
    def test_plain_sizes(self):
        for s in ("tiny", "base", "small", "medium"):
            assert NS["_normalize_whisper_size"](s) == s

    def test_en_suffix(self):
        assert NS["_normalize_whisper_size"]("tiny.en") == "tiny"
        assert NS["_normalize_whisper_size"]("small.en") == "small"

    def test_large_variants(self):
        for s in ("large", "large-v1", "large-v2", "large-v3"):
            assert NS["_normalize_whisper_size"](s) == "large"

    def test_turbo_variants(self):
        for s in ("turbo", "large-v3-turbo", "distil-large-v3"):
            assert NS["_normalize_whisper_size"](s) == "turbo"

    def test_unknown(self):
        assert NS["_normalize_whisper_size"]("gigantic") is None
        assert NS["_normalize_whisper_size"]("") is None
        assert NS["_normalize_whisper_size"](None) is None


class TestEstimate:
    def est(self, cfg, gpu=False):
        return NS["_estimate_memory_requirements"](cfg, gpu_available=gpu)

    def test_cpu_whisper(self):
        need = self.est(_cfg(model="small", backend="whisper"))
        assert need["ram_gb"] == BASELINE + EST["whisper"]["small"]["ram"]
        assert need["vram_gb"] == 0
        assert need["tier"] == "transcription"
        assert need["min_cores"] == 6

    def test_faster_whisper_cheaper_than_openai(self):
        fw = self.est(_cfg(model="large-v3", backend="faster-whisper"))
        ow = self.est(_cfg(model="large-v3", backend="whisper"))
        assert fw["ram_gb"] < ow["ram_gb"]

    def test_gpu_moves_cost_to_vram(self):
        need = self.est(_cfg(model="medium", backend="whisper", use_gpu=True, ft_gpu=True), gpu=True)
        assert need["vram_gb"] == EST["whisper"]["medium"]["vram"]
        assert need["ram_gb"] == BASELINE + GPU_HOST

    def test_use_gpu_without_hardware_stays_on_cpu(self):
        need = self.est(_cfg(model="medium", use_gpu=True), gpu=False)
        assert need["vram_gb"] == 0
        assert need["ram_gb"] == BASELINE + EST["whisper"]["medium"]["ram"]

    def test_local_nllb_adds(self):
        # Counted only when the translation model is actually downloaded.
        os.makedirs(_nllb_dir(), exist_ok=True)
        try:
            base = self.est(_cfg())
            with_t = self.est(_cfg(translation=True))
        finally:
            shutil.rmtree(_nllb_dir(), ignore_errors=True)
        assert with_t["ram_gb"] == base["ram_gb"] + EST["nllb"]["nllb-200-distilled-600M"]["ram"]
        assert with_t["tier"] == "transcription + translation"
        assert with_t["min_cores"] == 8

    def test_local_nllb_not_counted_without_model(self):
        # Enabled but no model downloaded → not set up → transcription-only.
        shutil.rmtree(_nllb_dir(), ignore_errors=True)
        need = self.est(_cfg(translation=True))
        assert need["tier"] == "transcription"
        assert need["min_cores"] == 6

    def test_remote_nllb_adds_nothing(self):
        base = self.est(_cfg())
        remote = self.est(_cfg(translation=True, remote=True))
        assert remote["ram_gb"] == base["ram_gb"]
        assert remote["tier"] == "transcription"

    def test_same_file_model_not_double_counted(self):
        one = self.est(_cfg(model="small", ft_model="small"))
        assert one["ram_gb"] == BASELINE + EST["whisper"]["small"]["ram"]

    def test_different_file_model_sums(self):
        need = self.est(_cfg(model="small", ft_model="tiny.en"))
        assert need["ram_gb"] == BASELINE + EST["whisper"]["small"]["ram"] + EST["whisper"]["tiny"]["ram"]

    def test_unknown_model_contributes_nothing(self):
        need = self.est(_cfg(model="gigantic", ft_model="gigantic"))
        assert need["ram_gb"] == BASELINE

    def test_malformed_config_fails_open(self):
        need = self.est({})
        assert need["ram_gb"] == BASELINE
        assert need["vram_gb"] == 0


class TestWarnings:
    def check(self, cfg, hw):
        return NS["_check_system_requirements"](cfg=cfg, hw=hw)

    def test_ample_hardware_no_warnings(self):
        assert self.check(_cfg(), AMPLE_HW) == []

    def test_low_ram_warns_with_model_name(self):
        hw = dict(AMPLE_HW, ram_bytes=4 * GB, has_cuda=False, vram_bytes=None)
        warns = self.check(_cfg(model="large-v3", backend="faster-whisper"), hw)
        assert any("RAM" in w and "large-v3" in w for w in warns)

    def test_ram_probe_failure_fails_open(self):
        hw = dict(AMPLE_HW, ram_bytes=0)
        assert self.check(_cfg(model="large-v3"), hw) == []

    def test_unknown_vram_no_vram_warning(self):
        hw = dict(AMPLE_HW, vram_bytes=None, has_cuda=False)
        warns = self.check(_cfg(model="large-v3", use_gpu=True), hw)
        assert not any("VRAM" in w for w in warns)

    def test_low_cuda_vram_warns(self):
        hw = dict(AMPLE_HW, vram_bytes=2 * GB)
        warns = self.check(_cfg(model="large-v3", use_gpu=True), hw)
        assert any("VRAM" in w and "GPU has 2.0 GB" in w for w in warns)

    def test_apple_silicon_pools_unified_memory(self):
        hw = {"ram_bytes": 8 * GB, "cpu_cores": 10, "disk_free_bytes": 500 * GB,
              "vram_bytes": None, "has_cuda": False, "apple_silicon": True}
        warns = self.check(_cfg(model="large-v3", use_gpu=True, translation=True, nllb_gpu=True), hw)
        assert any("unified memory" in w for w in warns)
        assert not any("VRAM" in w for w in warns)

    def test_low_cores_warns(self):
        hw = dict(AMPLE_HW, cpu_cores=4)
        warns = self.check(_cfg(), hw)
        assert any("CPU cores" in w and "minimum 6" in w for w in warns)

    def test_low_disk_warns(self):
        hw = dict(AMPLE_HW, disk_free_bytes=2 * GB)
        warns = self.check(_cfg(), hw)
        assert any("free disk" in w for w in warns)

    def test_messages_are_self_contained(self):
        hw = {"ram_bytes": 4 * GB, "cpu_cores": 2, "disk_free_bytes": 1 * GB,
              "vram_bytes": 1 * GB, "has_cuda": True, "apple_silicon": False}
        warns = self.check(_cfg(model="large-v3", use_gpu=True, translation=True), hw)
        assert len(warns) == 4  # cores, RAM, VRAM, disk
        for w in warns:
            if "CPU cores" not in w:
                assert "GB" in w
