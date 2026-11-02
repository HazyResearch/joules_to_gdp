"""Tests for execution/preflight.py — startup hardware baseline check."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ipw.execution.preflight import run_preflight


def test_allowed_gpu_pids_expands_server_workers(monkeypatch) -> None:
    from types import SimpleNamespace

    from ipw.execution.preflight import _allowed_gpu_pids

    monkeypatch.setenv("IPW_PREFLIGHT_ALLOWED_GPU_PIDS", "1234, 5678")

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def children(self, recursive=False):
            assert recursive
            return [SimpleNamespace(pid=4321)] if self.pid == 1234 else []

    monkeypatch.setattr("psutil.Process", FakeProcess)
    assert {1234, 5678, 4321} <= _allowed_gpu_pids()


def test_allowed_gpu_pids_rejects_bad_input(monkeypatch) -> None:
    from ipw.execution.preflight import _allowed_gpu_pids

    monkeypatch.setenv("IPW_PREFLIGHT_ALLOWED_GPU_PIDS", "12,not-a-pid")
    with pytest.raises(ValueError, match="comma-separated positive PIDs"):
        _allowed_gpu_pids()
    with pytest.raises(ValueError, match="comma-separated positive PIDs"):
        run_preflight(sample_seconds=0.0, strict=True)


class TestPreflight:
    def test_clean_hardware_returns_dirty_false(self) -> None:
        with patch("ipw.execution.preflight._sample_gpu_util", return_value=(0.5, [])), \
             patch("ipw.execution.preflight._sample_cpu_util", return_value=2.0):
            r = run_preflight(sample_seconds=0.0)
        assert r.shared_device_baseline_dirty is False
        assert r.gpu_util_pct_avg == pytest.approx(0.5)
        assert r.cpu_util_pct_avg == pytest.approx(2.0)
        assert r.foreign_gpu_pids == []

    def test_gpu_util_above_threshold_sets_dirty(self) -> None:
        with patch("ipw.execution.preflight._sample_gpu_util", return_value=(25.0, [])), \
             patch("ipw.execution.preflight._sample_cpu_util", return_value=1.0):
            r = run_preflight(sample_seconds=0.0, gpu_util_threshold_pct=5.0)
        assert r.shared_device_baseline_dirty is True

    def test_cpu_util_above_threshold_sets_dirty(self) -> None:
        with patch("ipw.execution.preflight._sample_gpu_util", return_value=(0.1, [])), \
             patch("ipw.execution.preflight._sample_cpu_util", return_value=50.0):
            r = run_preflight(sample_seconds=0.0, cpu_util_threshold_pct=10.0)
        assert r.shared_device_baseline_dirty is True

    def test_foreign_pids_set_dirty(self) -> None:
        with patch("ipw.execution.preflight._sample_gpu_util", return_value=(0.1, [12345])), \
             patch("ipw.execution.preflight._sample_cpu_util", return_value=1.0):
            r = run_preflight(sample_seconds=0.0)
        assert r.shared_device_baseline_dirty is True
        assert r.foreign_gpu_pids == [12345]

    def test_strict_mode_raises_when_dirty(self) -> None:
        with patch("ipw.execution.preflight._sample_gpu_util", return_value=(50.0, [])), \
             patch("ipw.execution.preflight._sample_cpu_util", return_value=1.0):
            with pytest.raises(RuntimeError, match="Shared-device contamination"):
                run_preflight(sample_seconds=0.0, strict=True)

    def test_nvml_unavailable_degrades_gracefully(self) -> None:
        with patch("ipw.execution.preflight._sample_gpu_util", side_effect=RuntimeError("no nvml")), \
             patch("ipw.execution.preflight._sample_cpu_util", return_value=1.0):
            r = run_preflight(sample_seconds=0.0)
        assert r.gpu_util_pct_avg is None
        assert r.shared_device_baseline_dirty is False
        assert any("NVML" in w for w in r.warnings)

    def test_strict_mode_rejects_nvml_sampling_failure(self) -> None:
        with patch("ipw.execution.preflight._sample_gpu_util", side_effect=RuntimeError("no nvml")), \
             patch("ipw.execution.preflight._sample_cpu_util", return_value=1.0):
            with pytest.raises(RuntimeError, match="could not sample NVIDIA GPUs: no nvml"):
                run_preflight(sample_seconds=0.0, strict=True)

    def test_psutil_unavailable_degrades_gracefully(self) -> None:
        with patch("ipw.execution.preflight._sample_gpu_util", return_value=(0.1, [])), \
             patch("ipw.execution.preflight._sample_cpu_util", side_effect=RuntimeError("no psutil")):
            r = run_preflight(sample_seconds=0.0)
        assert r.cpu_util_pct_avg is None
        assert r.shared_device_baseline_dirty is False
        assert any("psutil" in w for w in r.warnings)

    def test_strict_mode_rejects_cpu_sampling_failure(self) -> None:
        with patch("ipw.execution.preflight._sample_gpu_util", return_value=(0.1, [])), \
             patch("ipw.execution.preflight._sample_cpu_util", side_effect=RuntimeError("no psutil")):
            with pytest.raises(RuntimeError, match="could not sample CPU utilization: no psutil"):
                run_preflight(sample_seconds=0.0, strict=True)

    def test_gpu_process_listing_failure_is_visible_to_strict_mode(self, monkeypatch) -> None:
        def fail_process_listing(_handle):
            raise RuntimeError("process inventory unavailable")

        fake_nvml = SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlShutdown=lambda: None,
            nvmlDeviceGetCount=lambda: 1,
            nvmlDeviceGetHandleByIndex=lambda _index: object(),
            nvmlDeviceGetUtilizationRates=lambda _handle: SimpleNamespace(gpu=0),
            nvmlDeviceGetComputeRunningProcesses=fail_process_listing,
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)
        with patch("ipw.execution.preflight._sample_cpu_util", return_value=1.0):
            with pytest.raises(RuntimeError, match="process inventory unavailable"):
                run_preflight(sample_seconds=0.0, strict=True)

    def test_strict_does_not_raise_when_clean(self) -> None:
        with patch("ipw.execution.preflight._sample_gpu_util", return_value=(0.1, [])), \
             patch("ipw.execution.preflight._sample_cpu_util", return_value=1.0):
            # Should not raise
            r = run_preflight(sample_seconds=0.0, strict=True)
        assert r.shared_device_baseline_dirty is False
