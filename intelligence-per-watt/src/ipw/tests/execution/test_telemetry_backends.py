from __future__ import annotations

import pytest

from ipw.execution import telemetry_backends as mod


class _FakeContext:
    def __init__(self, kind: str, ids=None, **kwargs) -> None:
        self.kind = kind
        self.ids = tuple(ids or ())
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_legacy_telemetry_gpu_id_uses_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod,
        "NvidiaSmiTelemetrySession",
        lambda ids, **kwargs: _FakeContext("nvidia", ids, **kwargs),
    )

    ctx = mod.build_telemetry_context(
        backend="energy-monitor",
        telemetry_gpu_id=3,
        telemetry_device_ids=(),
        interval_seconds=0.5,
        buffer_seconds=60.0,
        max_samples=10,
    )

    assert ctx.kind == "nvidia"
    assert ctx.ids == (3,)
    assert ctx.kwargs["interval_seconds"] == pytest.approx(0.5)


def test_legacy_gpu_id_rejects_amd_backend() -> None:
    with pytest.raises(ValueError, match="NVIDIA-only"):
        mod.build_telemetry_context(
            backend="amd-smi",
            telemetry_gpu_id=0,
            telemetry_device_ids=(),
            interval_seconds=0.2,
            buffer_seconds=60.0,
            max_samples=10,
        )


def test_explicit_amd_backend_uses_amd_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod,
        "AmdSmiTelemetrySession",
        lambda ids, **kwargs: _FakeContext("amd", ids, **kwargs),
    )

    ctx = mod.build_telemetry_context(
        backend="rocm-smi",
        telemetry_gpu_id=None,
        telemetry_device_ids=(1, 2),
        interval_seconds=0.2,
        buffer_seconds=60.0,
        max_samples=10,
    )

    assert ctx.kind == "amd"
    assert ctx.ids == (1, 2)
    assert ctx.kwargs["backend"] == "rocm-smi"


def test_auto_amd_profile_uses_visible_rocm_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IPW_ACCELERATOR_PROFILE", "amd-mi300")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5")
    monkeypatch.setattr(
        mod,
        "AmdSmiTelemetrySession",
        lambda ids, **kwargs: _FakeContext("amd", ids, **kwargs),
    )

    ctx = mod.build_telemetry_context(
        backend="auto",
        telemetry_gpu_id=None,
        telemetry_device_ids=(),
        interval_seconds=0.2,
        buffer_seconds=60.0,
        max_samples=10,
    )

    assert ctx.kind == "amd"
    assert ctx.ids == (4, 5)


def test_energy_monitor_backend_preserves_previous_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod,
        "EnergyMonitorCollector",
        lambda timeout=30.0, interval_seconds=None: _FakeContext(
            "collector", (), interval_seconds=interval_seconds
        ),
    )
    monkeypatch.setattr(
        mod,
        "TelemetrySession",
        lambda collector, **kwargs: _FakeContext("energy-monitor", (), collector=collector, **kwargs),
    )

    ctx = mod.build_telemetry_context(
        backend="energy-monitor",
        telemetry_gpu_id=None,
        telemetry_device_ids=(),
        interval_seconds=0.2,
        buffer_seconds=60.0,
        max_samples=10,
    )

    assert ctx.kind == "energy-monitor"
    assert ctx.kwargs["buffer_seconds"] == pytest.approx(60.0)
    assert ctx.kwargs["collector"].kwargs["interval_seconds"] == pytest.approx(0.2)


def test_resolve_visible_device_ids_prefers_rocm_when_cuda_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IPW_GPU_DEVICE_ID", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "7")

    assert mod.resolve_visible_device_ids() == (7,)
