"""Telemetry session helpers for profiling runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Iterable, Iterator, Optional

from ..core.types import TelemetryReading
from ..telemetry import EnergyMonitorCollector


@dataclass
class TelemetrySample:
    timestamp: float
    reading: TelemetryReading


class TelemetrySampleRecorder:
    """Write every accepted meter sample before the rolling buffer trims it."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.sample_count = 0
        self._digest = hashlib.sha256()
        self._stream = None
        self.error: OSError | None = None
        self._lock = threading.Lock()
        self._latest_sample_time: float | None = None
        self._window_scan_offset = 0
        self.failure_check: Callable[[], None] | None = None

    def _check_failure(self) -> None:
        if self.failure_check is not None:
            self.failure_check()

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("wb")

    def write(self, sample: TelemetrySample) -> None:
        reading = sample.reading
        gpu_info = reading.gpu_info
        row = {
            "schema_version": 1,
            "sample_time_s": sample.timestamp,
            "timestamp_nanos": reading.timestamp_nanos,
            "platform": reading.platform,
            "gpu_device_id": gpu_info.device_id if gpu_info is not None else None,
            "gpu_vendor": gpu_info.vendor if gpu_info is not None else None,
            "gpu_backend": gpu_info.backend if gpu_info is not None else None,
            "gpu_power_watts": reading.power_watts,
            "gpu_energy_joules": reading.energy_joules,
            "cpu_power_watts": reading.cpu_power_watts,
            "cpu_energy_joules": reading.cpu_energy_joules,
            "dram_power_watts": reading.dram_power_watts,
            "dram_energy_joules": reading.dram_energy_joules,
            "ane_power_watts": reading.ane_power_watts,
            "ane_energy_joules": reading.ane_energy_joules,
            "gpu_compute_utilization_pct": reading.gpu_compute_utilization_pct,
            "gpu_memory_bandwidth_utilization_pct": (
                reading.gpu_memory_bandwidth_utilization_pct
            ),
            "gpu_tensor_core_utilization_pct": reading.gpu_tensor_core_utilization_pct,
            "gpu_memory_usage_mb": reading.gpu_memory_usage_mb,
            "gpu_memory_total_mb": reading.gpu_memory_total_mb,
            "cpu_memory_usage_mb": reading.cpu_memory_usage_mb,
            "temperature_celsius": reading.temperature_celsius,
        }
        row = {
            key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in row.items()
        }
        encoded = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        with self._lock:
            if self._stream is None:
                return
            try:
                self._stream.write(encoded)
                self._stream.flush()
            except OSError as exc:
                self.error = exc
                raise
            self._digest.update(encoded)
            self.sample_count += 1
            self._latest_sample_time = sample.timestamp

    def close(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None

    def wait_until_after(self, timestamp: float, *, timeout_s: float = 2.0) -> None:
        """Give the sampler time to write the first post-window reading."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._check_failure()
            with self._lock:
                if self._latest_sample_time is not None and self._latest_sample_time >= timestamp:
                    return
            time.sleep(0.05)
        self._check_failure()
        raise RuntimeError("Telemetry sampler did not record a post-window reading in time")

    def read_window(self, start_time: float, end_time: float) -> list[TelemetrySample]:
        """Read an entire batch window from disk, including boundary samples."""
        self._check_failure()
        with self._lock:
            size = self._stream.tell() if self._stream is not None else self.path.stat().st_size
            offset = self._window_scan_offset
        before: TelemetrySample | None = None
        inside: list[TelemetrySample] = []
        after: TelemetrySample | None = None
        previous_offset = offset
        with self.path.open("rb") as stream:
            stream.seek(offset)
            while stream.tell() < size:
                line_offset = stream.tell()
                line = stream.readline()
                row = json.loads(line)
                sample_time = float(row["sample_time_s"])
                reading = TelemetryReading(
                    power_watts=row.get("gpu_power_watts"),
                    energy_joules=row.get("gpu_energy_joules"),
                    cpu_power_watts=row.get("cpu_power_watts"),
                    cpu_energy_joules=row.get("cpu_energy_joules"),
                    dram_power_watts=row.get("dram_power_watts"),
                    dram_energy_joules=row.get("dram_energy_joules"),
                    gpu_memory_bandwidth_utilization_pct=row.get(
                        "gpu_memory_bandwidth_utilization_pct"
                    ),
                    timestamp_nanos=row.get("timestamp_nanos"),
                )
                sample = TelemetrySample(timestamp=sample_time, reading=reading)
                if sample_time < start_time:
                    before = sample
                elif sample_time <= end_time:
                    inside.append(sample)
                else:
                    after = sample
                    break
                previous_offset = line_offset
        # The next batch starts after this one. Retain a nearby earlier sample
        # as its left boundary without scanning all prior batches again.
        self._window_scan_offset = previous_offset
        return [sample for sample in (before, *inside, after) if sample is not None]

    def manifest(self) -> dict[str, object]:
        self._check_failure()
        if self.error is not None:
            raise RuntimeError("Raw telemetry sample export failed") from self.error
        return {
            "schema_version": 1,
            "path": self.path.name,
            "sample_count": self.sample_count,
            "sha256": self._digest.hexdigest(),
        }


class TelemetrySession(AbstractContextManager["TelemetrySession"]):
    """Capture telemetry readings in a background thread."""

    def __init__(
        self,
        collector: EnergyMonitorCollector,
        *,
        buffer_seconds: Optional[float] = 30.0,
        max_samples: Optional[int] = 10_000,
        sample_output_path: Path | None = None,
    ) -> None:
        self._collector = collector
        self._buffer_seconds = buffer_seconds
        self._max_samples = max_samples
        self._samples: Deque[TelemetrySample] = deque(maxlen=max_samples)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._first_sample = threading.Event()
        self._collector_ctx = None
        self._failure: Exception | None = None
        self._gpu_device_id = self._resolve_gpu_device_id()
        self._single_visible_gpu = self._has_single_visible_gpu()
        self.sample_recorder = (
            TelemetrySampleRecorder(sample_output_path)
            if sample_output_path is not None else None
        )
        if self.sample_recorder is not None:
            self.sample_recorder.failure_check = self.raise_if_failed

    def __enter__(self) -> "TelemetrySession":
        self._failure = None
        self._first_sample.clear()
        if self.sample_recorder is not None:
            self.sample_recorder.open()
        self._collector_ctx = self._collector.start()
        try:
            self._collector_ctx.__enter__()
        except BaseException:
            if self.sample_recorder is not None:
                self.sample_recorder.close()
            raise
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if self.sample_recorder is not None:
            try:
                deadline = time.monotonic() + 10.0
                while not self._first_sample.wait(0.05):
                    self.raise_if_failed()
                    if self._thread is not None and not self._thread.is_alive():
                        raise RuntimeError("Telemetry sampler ended before its first reading")
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Telemetry sampler produced no first reading within 10 seconds")
            except BaseException:
                self._shutdown()
                raise
        return self

    def _shutdown(self) -> None:
        self._stop_event.set()
        stop = getattr(self._collector, "stop", None)
        if callable(stop):
            stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._collector_ctx is not None:
            self._collector_ctx.__exit__(None, None, None)
        if self.sample_recorder is not None:
            self.sample_recorder.close()

    def __exit__(self, exc_type, exc, tb) -> None:
        self._shutdown()
        if self._thread is not None and self._thread.is_alive():
            if exc_type is None:
                raise RuntimeError("Telemetry sampler did not stop cleanly")
        if exc_type is None:
            self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError("Telemetry sampler stopped unexpectedly") from self._failure

    def _run(self) -> None:
        try:
            for reading in self._collector.stream_readings():
                if not self._include_reading(reading):
                    continue
                timestamp = (
                    float(reading.timestamp_nanos) / 1_000_000_000.0
                    if reading.timestamp_nanos is not None
                    else time.time()
                )
                sample = TelemetrySample(timestamp=timestamp, reading=reading)
                self._samples.append(sample)
                if self.sample_recorder is not None:
                    self.sample_recorder.write(sample)
                self._first_sample.set()
                self._trim(timestamp)
                if self._stop_event.is_set():
                    break
        except Exception as exc:
            self._failure = exc
            self._stop_event.set()

    def _visible_gpu_parts(self) -> list[str]:
        visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
        return [part.strip() for part in visible.split(",") if part.strip()]

    def _has_single_visible_gpu(self) -> bool:
        return len(self._visible_gpu_parts()) == 1

    def _resolve_gpu_device_id(self) -> Optional[int]:
        explicit = os.getenv("IPW_GPU_DEVICE_ID")
        if explicit:
            try:
                return int(explicit)
            except ValueError:
                return None

        parts = self._visible_gpu_parts()
        if len(parts) != 1:
            return None
        try:
            return int(parts[0])
        except ValueError:
            return None

    def _include_reading(self, reading: TelemetryReading) -> bool:
        if self._gpu_device_id is None:
            return True
        gpu_info = reading.gpu_info
        if gpu_info is None:
            return True
        device_id = int(gpu_info.device_id)
        if device_id == self._gpu_device_id:
            return True
        # NVML reports device 0 when CUDA_VISIBLE_DEVICES exposes exactly one
        # physical GPU. In that mode the monitor is already isolated to one H100,
        # so keep the visible device-0 samples instead of dropping all telemetry.
        return self._single_visible_gpu and device_id == 0

    def _trim(self, current_time: float) -> None:
        if self._buffer_seconds is None:
            return
        cutoff = current_time - self._buffer_seconds
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    def readings(self) -> Iterable[TelemetrySample]:
        self.raise_if_failed()
        return list(self._samples)

    def window(self, start_time: float, end_time: float) -> Iterator[TelemetrySample]:
        self.raise_if_failed()
        for sample in list(self._samples):
            if start_time <= sample.timestamp <= end_time:
                yield sample
