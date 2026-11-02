"""Telemetry backend selection helpers for CLI runs."""

from __future__ import annotations

import os
import shutil
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Sequence

from ..telemetry import EnergyMonitorCollector
from .amd_smi_telemetry import AmdSmiTelemetrySession
from .nvidia_smi_telemetry import NvidiaSmiTelemetrySession
from .telemetry_session import TelemetrySession

TelemetryBackend = str

SUPPORTED_TELEMETRY_BACKENDS = (
    "energy-monitor",
    "auto",
    "nvidia-smi",
    "amd-smi",
    "rocm-smi",
)


def resolve_visible_device_ids() -> tuple[int, ...]:
    """Return the visible GPU ids from CUDA or ROCm visibility env vars."""

    for env_name in (
        "IPW_GPU_DEVICE_ID",
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        value = os.getenv(env_name, "").strip()
        if not value:
            continue
        ids = _parse_device_ids(value)
        if ids:
            return ids
    return ()


def build_telemetry_context(
    *,
    backend: TelemetryBackend,
    telemetry_gpu_id: int | None,
    telemetry_device_ids: Sequence[int],
    interval_seconds: float,
    buffer_seconds: float,
    max_samples: int,
    sample_output_path: Path | None = None,
) -> AbstractContextManager:
    """Build the telemetry context requested by ``ipw run``.

    ``telemetry_gpu_id`` is the legacy NVIDIA-only option and intentionally
    keeps its old behavior unless the caller explicitly asks for an incompatible
    backend.
    """

    if backend not in SUPPORTED_TELEMETRY_BACKENDS:
        raise ValueError(f"Unsupported telemetry backend: {backend}")

    device_ids = tuple(int(device_id) for device_id in telemetry_device_ids)
    if telemetry_gpu_id is not None:
        if device_ids:
            raise ValueError("Use either --telemetry-gpu-id or --telemetry-device-id, not both")
        if backend not in {"energy-monitor", "auto", "nvidia-smi"}:
            raise ValueError("--telemetry-gpu-id is NVIDIA-only; use --telemetry-device-id for AMD")
        return NvidiaSmiTelemetrySession(
            [telemetry_gpu_id],
            interval_seconds=interval_seconds,
            buffer_seconds=buffer_seconds,
            max_samples=max_samples,
            sample_output_path=sample_output_path,
        )

    if backend == "energy-monitor":
        return TelemetrySession(
            EnergyMonitorCollector(timeout=30.0, interval_seconds=interval_seconds),
            buffer_seconds=buffer_seconds,
            max_samples=max_samples,
            sample_output_path=sample_output_path,
        )

    if backend == "nvidia-smi":
        return NvidiaSmiTelemetrySession(
            _resolve_device_ids(device_ids),
            interval_seconds=interval_seconds,
            buffer_seconds=buffer_seconds,
            max_samples=max_samples,
            sample_output_path=sample_output_path,
        )

    if backend in {"amd-smi", "rocm-smi"}:
        return AmdSmiTelemetrySession(
            _resolve_device_ids(device_ids),
            backend=backend,
            interval_seconds=interval_seconds,
            buffer_seconds=buffer_seconds,
            max_samples=max_samples,
            sample_output_path=sample_output_path,
        )

    return _build_auto_context(
        device_ids=device_ids,
        interval_seconds=interval_seconds,
        buffer_seconds=buffer_seconds,
        max_samples=max_samples,
        sample_output_path=sample_output_path,
    )


def _build_auto_context(
    *,
    device_ids: Sequence[int],
    interval_seconds: float,
    buffer_seconds: float,
    max_samples: int,
    sample_output_path: Path | None = None,
) -> AbstractContextManager:
    profile = os.getenv("IPW_ACCELERATOR_PROFILE", "").strip().lower()
    resolved_ids = _resolve_device_ids(device_ids)
    if profile in {"amd-mi300", "amd300", "mi300", "mi300x"}:
        return AmdSmiTelemetrySession(
            resolved_ids,
            backend="auto",
            interval_seconds=interval_seconds,
            buffer_seconds=buffer_seconds,
            max_samples=max_samples,
            sample_output_path=sample_output_path,
        )
    if profile in {"dgx-spark", "dgxspark", "nvidia", "cuda"}:
        return NvidiaSmiTelemetrySession(
            resolved_ids,
            interval_seconds=interval_seconds,
            buffer_seconds=buffer_seconds,
            max_samples=max_samples,
            sample_output_path=sample_output_path,
        )
    if shutil.which("amd-smi") or shutil.which("rocm-smi"):
        return AmdSmiTelemetrySession(
            resolved_ids,
            backend="auto",
            interval_seconds=interval_seconds,
            buffer_seconds=buffer_seconds,
            max_samples=max_samples,
            sample_output_path=sample_output_path,
        )
    if shutil.which("nvidia-smi"):
        return NvidiaSmiTelemetrySession(
            resolved_ids,
            interval_seconds=interval_seconds,
            buffer_seconds=buffer_seconds,
            max_samples=max_samples,
            sample_output_path=sample_output_path,
        )
    return TelemetrySession(
        EnergyMonitorCollector(timeout=30.0, interval_seconds=interval_seconds),
        buffer_seconds=buffer_seconds,
        max_samples=max_samples,
        sample_output_path=sample_output_path,
    )


def _resolve_device_ids(device_ids: Sequence[int]) -> tuple[int, ...]:
    if device_ids:
        return tuple(int(device_id) for device_id in device_ids)
    visible = resolve_visible_device_ids()
    return visible if visible else (0,)


def _parse_device_ids(value: str) -> tuple[int, ...]:
    ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            return ()
    return tuple(ids)


__all__ = [
    "SUPPORTED_TELEMETRY_BACKENDS",
    "build_telemetry_context",
    "resolve_visible_device_ids",
]
