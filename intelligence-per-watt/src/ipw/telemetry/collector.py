"""Hardware collector implementation backed by the energy monitor."""

from __future__ import annotations

import math
import os
import threading
from contextlib import contextmanager
from typing import Iterable, Optional, Tuple

import grpc

from ipw.core.types import GpuInfo, SystemInfo, TelemetryReading

from .launcher import DEFAULT_TARGET, ensure_monitor, normalize_target, wait_for_ready
from .proto import get_stub_bundle


def _default_target_for_visible_gpu() -> str:
    """Use a stable per-H100 monitor port when a run is GPU-pinned."""

    visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if "," in visible:
        return DEFAULT_TARGET
    try:
        gpu_index = int(visible)
    except ValueError:
        return DEFAULT_TARGET
    return f"127.0.0.1:{50053 + gpu_index}"


class EnergyMonitorCollector:
    collector_name = "Energy Monitor"

    def __init__(
        self,
        target: str | None = None,
        *,
        channel_options: Optional[Tuple[Tuple[str, str], ...]] = None,
        timeout: float = 5.0,
        interval_seconds: float | None = None,
    ) -> None:
        target = target or os.getenv("IPW_ENERGY_MONITOR_TARGET") or _default_target_for_visible_gpu()
        self._target = normalize_target(target or DEFAULT_TARGET)
        self._channel_options = channel_options or ()
        self._timeout = timeout
        self._interval_seconds = interval_seconds
        self._bundle = get_stub_bundle()
        self._stream_lock = threading.Lock()
        self._active_stream = None
        self._stopped = threading.Event()

    @contextmanager
    def start(self):
        self._stopped.clear()
        interval_override = os.getenv("IPW_ENERGY_MONITOR_INTERVAL_MS")
        if interval_override is not None:
            try:
                interval_ms = int(interval_override)
            except ValueError as exc:
                raise ValueError("IPW_ENERGY_MONITOR_INTERVAL_MS must be a positive integer") from exc
        elif self._interval_seconds is not None:
            interval_ms = round(self._interval_seconds * 1000)
        else:
            interval_ms = None
        if interval_ms is not None and interval_ms <= 0:
            raise ValueError("Energy monitor collection interval must be positive")
        launch_args = (
            ["--collection-interval-ms", str(interval_ms)]
            if interval_ms is not None else None
        )
        with ensure_monitor(
            self._target, timeout=self._timeout, launch=True, launch_args=launch_args
        ):
            if os.getenv("IPW_REQUIRE_TOTAL_ENERGY") == "1" and not wait_for_ready(
                self._target, timeout=self._timeout, require_energy_components=True
            ):
                raise RuntimeError(
                    "Full-energy preflight failed: GPU, CPU package, and DRAM meter "
                    "fields must all be available on this host"
                )
            yield

    def stop(self) -> None:
        """Cancel a blocking telemetry RPC when its session closes."""
        with self._stream_lock:
            self._stopped.set()
            stream = self._active_stream
        if stream is not None:
            stream.cancel()

    @classmethod
    def is_available(cls) -> bool:
        target = os.getenv("IPW_ENERGY_MONITOR_TARGET") or _default_target_for_visible_gpu()
        return wait_for_ready(target, timeout=1.0)

    def stream_readings(self) -> Iterable[TelemetryReading]:
        channel = grpc.insecure_channel(self._target, options=self._channel_options)
        stub = self._bundle.stub_factory(channel)
        stream = None
        try:
            if self._stopped.is_set():
                return
            stream = stub.StreamTelemetry(self._bundle.StreamRequestCls())
            with self._stream_lock:
                if self._stopped.is_set():
                    stream.cancel()
                    return
                self._active_stream = stream
            for raw in stream:
                yield self._convert(raw)
            if not self._stopped.is_set():
                raise RuntimeError("Energy monitor stream ended before the telemetry session closed")
        except grpc.RpcError as exc:
            if self._stopped.is_set():
                return
            status = exc.code() if isinstance(exc, grpc.Call) else None
            details = exc.details() if isinstance(exc, grpc.Call) else ""
            code_name = getattr(status, "name", str(status)) if status else "UNKNOWN"
            message = f"Energy monitor stream failed: {code_name}"
            if details:
                message = f"{message} {details}"
            raise RuntimeError(message) from exc
        finally:
            with self._stream_lock:
                if self._active_stream is stream:
                    self._active_stream = None
            channel.close()

    def _convert(self, message) -> TelemetryReading:
        system_info = getattr(message, "system_info", None)
        gpu_info = getattr(message, "gpu_info", None)

        system = None
        if system_info is not None:
            system = SystemInfo(
                os_name=getattr(system_info, "os_name", ""),
                os_version=getattr(system_info, "os_version", ""),
                kernel_version=getattr(system_info, "kernel_version", ""),
                host_name=getattr(system_info, "host_name", ""),
                cpu_count=getattr(system_info, "cpu_count", 0),
                cpu_brand=getattr(system_info, "cpu_brand", ""),
            )

        gpu = None
        if gpu_info is not None:
            gpu = GpuInfo(
                name=getattr(gpu_info, "name", ""),
                vendor=getattr(gpu_info, "vendor", ""),
                device_id=getattr(gpu_info, "device_id", 0),
                device_type=getattr(gpu_info, "device_type", ""),
                backend=getattr(gpu_info, "backend", ""),
            )

        return TelemetryReading(
            power_watts=_safe_float(getattr(message, "power_watts", None)),
            energy_joules=_safe_float(getattr(message, "energy_joules", None)),
            temperature_celsius=_safe_float(
                getattr(message, "temperature_celsius", None)
            ),
            gpu_memory_usage_mb=_safe_float(
                getattr(message, "gpu_memory_usage_mb", None)
            ),
            gpu_memory_total_mb=_safe_float(
                getattr(message, "gpu_memory_total_mb", None)
            ),
            cpu_memory_usage_mb=_safe_float(
                getattr(message, "cpu_memory_usage_mb", None)
            ),
            cpu_power_watts=_safe_float(
                getattr(message, "cpu_power_watts", None)
            ),
            cpu_energy_joules=_safe_float(
                getattr(message, "cpu_energy_joules", None)
            ),
            dram_power_watts=_safe_float(
                getattr(message, "dram_power_watts", None)
            ),
            dram_energy_joules=_safe_float(
                getattr(message, "dram_energy_joules", None)
            ),
            ane_power_watts=_safe_float(getattr(message, "ane_power_watts", None)),
            ane_energy_joules=_safe_float(getattr(message, "ane_energy_joules", None)),
            gpu_compute_utilization_pct=_safe_float(
                getattr(message, "gpu_compute_utilization_pct", None)
            ),
            gpu_memory_bandwidth_utilization_pct=_safe_float(
                getattr(message, "gpu_memory_bandwidth_utilization_pct", None)
            ),
            gpu_tensor_core_utilization_pct=_safe_float(
                getattr(message, "gpu_tensor_core_utilization_pct", None)
            ),
            platform=getattr(message, "platform", None),
            timestamp_nanos=getattr(message, "timestamp_nanos", None),
            system_info=system,
            gpu_info=gpu,
        )


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


__all__ = ["EnergyMonitorCollector"]
