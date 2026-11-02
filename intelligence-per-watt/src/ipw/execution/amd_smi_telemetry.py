"""Per-GPU telemetry sampling backed by AMD SMI tools."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Iterable, Iterator, Literal, Sequence

from ..core.types import GpuInfo, TelemetryReading
from .telemetry_session import TelemetrySample, TelemetrySampleRecorder

AmdSmiBackend = Literal["auto", "amd-smi", "rocm-smi"]


@dataclass(frozen=True)
class _AmdGpuSample:
    gpu_id: int
    power_watts: float
    memory_used_mb: float | None
    memory_total_mb: float | None
    utilization_pct: float | None
    temperature_celsius: float | None
    name: str
    backend: str


class AmdSmiTelemetrySession(AbstractContextManager["AmdSmiTelemetrySession"]):
    """Capture cumulative energy for selected AMD GPU ids.

    The bundled energy monitor can cover AMD hosts, but GDPval shard runs need
    explicit device selection similar to ``--telemetry-gpu-id`` on NVIDIA. This
    session polls ``amd-smi`` when available and falls back to ``rocm-smi``.
    """

    def __init__(
        self,
        gpu_ids: Sequence[int],
        *,
        backend: AmdSmiBackend = "auto",
        interval_seconds: float = 0.2,
        buffer_seconds: float = 30.0,
        max_samples: int = 10_000,
        sample_output_path: Path | None = None,
    ) -> None:
        if not gpu_ids:
            raise ValueError("At least one AMD GPU id is required")
        if backend not in {"auto", "amd-smi", "rocm-smi"}:
            raise ValueError(f"Unsupported AMD telemetry backend: {backend}")
        self._gpu_ids = tuple(int(gpu_id) for gpu_id in gpu_ids)
        self._backend: AmdSmiBackend = backend
        self._interval_seconds = max(0.05, float(interval_seconds))
        self._buffer_seconds = float(buffer_seconds)
        self._samples: Deque[TelemetrySample] = deque(maxlen=max_samples)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_timestamp: float | None = None
        self._last_power_watts: dict[int, float] = {}
        self._energy_joules: dict[int, float] = {gpu_id: 0.0 for gpu_id in self._gpu_ids}
        self.sample_recorder = (
            TelemetrySampleRecorder(sample_output_path)
            if sample_output_path is not None else None
        )

    def __enter__(self) -> "AmdSmiTelemetrySession":
        if self.sample_recorder is not None:
            self.sample_recorder.open()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.sample_recorder is not None:
            self.sample_recorder.close()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._sample_once()
            except Exception:
                pass
            self._stop_event.wait(self._interval_seconds)

    def _sample_once(self) -> None:
        now = time.time()
        gpu_samples = _query_amd_smi(self._gpu_ids, backend=self._backend)
        if not gpu_samples:
            return

        current_power = {sample.gpu_id: sample.power_watts for sample in gpu_samples}
        if self._last_timestamp is not None:
            dt = max(0.0, now - self._last_timestamp)
            for gpu_id, power in current_power.items():
                previous = self._last_power_watts.get(gpu_id, power)
                self._energy_joules[gpu_id] += ((previous + power) / 2.0) * dt

        self._last_timestamp = now
        self._last_power_watts = current_power

        total_power = sum(current_power.values())
        total_energy = sum(self._energy_joules.get(gpu_id, 0.0) for gpu_id in self._gpu_ids)
        memory_used = _sum_optional(sample.memory_used_mb for sample in gpu_samples)
        memory_total = _sum_optional(sample.memory_total_mb for sample in gpu_samples)
        utilization = _mean_optional(sample.utilization_pct for sample in gpu_samples)
        temperature = _mean_optional(sample.temperature_celsius for sample in gpu_samples)

        device_id = self._gpu_ids[0] if len(self._gpu_ids) == 1 else -1
        first = gpu_samples[0]
        reading = TelemetryReading(
            power_watts=total_power,
            energy_joules=total_energy,
            temperature_celsius=temperature,
            gpu_memory_usage_mb=memory_used,
            gpu_memory_total_mb=memory_total,
            gpu_compute_utilization_pct=utilization,
            timestamp_nanos=int(now * 1_000_000_000),
            platform="amd",
            gpu_info=GpuInfo(
                name=first.name if device_id >= 0 else "AMD GPU group",
                vendor="AMD",
                device_id=device_id,
                device_type="GPU",
                backend=first.backend,
            ),
        )
        sample = TelemetrySample(timestamp=now, reading=reading)
        with self._lock:
            self._samples.append(sample)
            if self.sample_recorder is not None:
                self.sample_recorder.write(sample)
            self._trim(now)

    def _trim(self, current_time: float) -> None:
        cutoff = current_time - self._buffer_seconds
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    def readings(self) -> Iterable[TelemetrySample]:
        with self._lock:
            return list(self._samples)

    def window(self, start_time: float, end_time: float) -> Iterator[TelemetrySample]:
        with self._lock:
            samples = list(self._samples)
        for sample in samples:
            if start_time <= sample.timestamp <= end_time:
                yield sample


def _query_amd_smi(
    gpu_ids: Sequence[int],
    *,
    backend: AmdSmiBackend = "auto",
) -> list[_AmdGpuSample]:
    if backend in {"auto", "amd-smi"} and shutil.which("amd-smi"):
        samples = _query_amd_smi_metric(gpu_ids)
        if samples or backend == "amd-smi":
            return samples
    if backend in {"auto", "rocm-smi"} and shutil.which("rocm-smi"):
        return _query_rocm_smi(gpu_ids)
    if backend == "amd-smi":
        raise FileNotFoundError("amd-smi not found")
    if backend == "rocm-smi":
        raise FileNotFoundError("rocm-smi not found")
    raise FileNotFoundError("Neither amd-smi nor rocm-smi is available")


def _query_amd_smi_metric(gpu_ids: Sequence[int]) -> list[_AmdGpuSample]:
    proc = subprocess.run(
        [
            "amd-smi",
            "metric",
            "--json",
            "-g",
            ",".join(str(gpu_id) for gpu_id in gpu_ids),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5.0,
    )
    data = json.loads(proc.stdout or "[]")
    return _parse_amd_smi_json(data, gpu_ids)


def _query_rocm_smi(gpu_ids: Sequence[int]) -> list[_AmdGpuSample]:
    proc = subprocess.run(
        [
            "rocm-smi",
            "-d",
            ",".join(str(gpu_id) for gpu_id in gpu_ids),
            "--showproductname",
            "--showpower",
            "--showuse",
            "--showmeminfo",
            "vram",
            "--showtemp",
            "--json",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5.0,
    )
    data = json.loads(proc.stdout or "{}")
    return _parse_rocm_smi_json(data, gpu_ids)


def _parse_amd_smi_json(data: Any, gpu_ids: Sequence[int]) -> list[_AmdGpuSample]:
    rows = data if isinstance(data, list) else _dict_values(data)
    samples: list[_AmdGpuSample] = []
    wanted = {int(gpu_id) for gpu_id in gpu_ids}
    for row in rows:
        if not isinstance(row, dict):
            continue
        gpu_id = _extract_gpu_id(row)
        if gpu_id is None or gpu_id not in wanted:
            continue
        flat = dict(_flatten(row))
        power = _first_number(flat, include=("power",), exclude=("cap", "limit"))
        if power is None:
            continue
        samples.append(
            _AmdGpuSample(
                gpu_id=gpu_id,
                power_watts=power,
                memory_used_mb=_first_memory_mb(flat, total=False),
                memory_total_mb=_first_memory_mb(flat, total=True),
                utilization_pct=_first_number(
                    flat,
                    include=("activity",),
                    exclude=("memory", "mem"),
                )
                or _first_number(flat, include=("gpu", "use"), exclude=("memory", "mem")),
                temperature_celsius=_first_number(flat, include=("temp",), exclude=()),
                name=str(_first_text(flat, include=("name", "market"), exclude=()) or f"AMD GPU {gpu_id}"),
                backend="amd-smi",
            )
        )
    return samples


def _parse_rocm_smi_json(data: Any, gpu_ids: Sequence[int]) -> list[_AmdGpuSample]:
    if not isinstance(data, dict):
        return []
    samples: list[_AmdGpuSample] = []
    wanted = {int(gpu_id) for gpu_id in gpu_ids}
    for key, value in data.items():
        gpu_id = _id_from_text(str(key))
        if gpu_id is None or gpu_id not in wanted or not isinstance(value, dict):
            continue
        flat = dict(_flatten(value))
        power = _first_number(flat, include=("power",), exclude=("cap", "limit"))
        if power is None:
            continue
        samples.append(
            _AmdGpuSample(
                gpu_id=gpu_id,
                power_watts=power,
                memory_used_mb=_first_memory_mb(flat, total=False),
                memory_total_mb=_first_memory_mb(flat, total=True),
                utilization_pct=_first_number(flat, include=("gpu", "use"), exclude=("memory", "mem")),
                temperature_celsius=_first_number(flat, include=("temp",), exclude=()),
                name=str(
                    _first_text(flat, include=("card", "series"), exclude=())
                    or _first_text(flat, include=("product", "name"), exclude=())
                    or f"AMD GPU {gpu_id}"
                ),
                backend="rocm-smi",
            )
        )
    return samples


def _flatten(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(child, next_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            next_prefix = f"{prefix}.{index}" if prefix else str(index)
            yield from _flatten(child, next_prefix)
    else:
        yield _norm_key(prefix), value


def _dict_values(data: Any) -> list[Any]:
    return list(data.values()) if isinstance(data, dict) else []


def _extract_gpu_id(row: dict[str, Any]) -> int | None:
    for key in ("gpu", "gpu_id", "gpu_index", "id", "device_id"):
        value = row.get(key)
        gpu_id = _to_int(value)
        if gpu_id is not None:
            return gpu_id
    return _id_from_text(str(row.get("name") or row.get("device") or ""))


def _id_from_text(value: str) -> int | None:
    match = re.search(r"(?:card|gpu|\[)?\s*(\d+)\]?", value, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _first_number(
    flat: dict[str, Any],
    *,
    include: Sequence[str],
    exclude: Sequence[str],
) -> float | None:
    for key, value in flat.items():
        if all(part in key for part in include) and not any(part in key for part in exclude):
            parsed = _to_float(value)
            if parsed is not None:
                return parsed
    return None


def _first_text(
    flat: dict[str, Any],
    *,
    include: Sequence[str],
    exclude: Sequence[str],
) -> str | None:
    for key, value in flat.items():
        if all(part in key for part in include) and not any(part in key for part in exclude):
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _first_memory_mb(flat: dict[str, Any], *, total: bool) -> float | None:
    alternatives = ("usage", "used", "allocated") if not total else ("total",)
    for key, value in flat.items():
        if "vram" not in key and "memory" not in key and "mem" not in key:
            continue
        if total and "total" not in key:
            continue
        if total and any(part in key for part in ("used", "allocated")):
            continue
        if not total and not any(part in key for part in alternatives):
            continue
        parsed = _to_float(value)
        if parsed is None:
            continue
        unit = _memory_unit_for_key(flat, key)
        if "(b)" in key or key.endswith("_b") or key.endswith(".b"):
            return parsed / (1024.0 * 1024.0)
        if "(kb)" in key or key.endswith("_kb") or key.endswith(".kb"):
            return parsed / 1024.0
        if "(gb)" in key or key.endswith("_gb") or key.endswith(".gb"):
            return parsed * 1024.0
        if unit == "b":
            return parsed / (1024.0 * 1024.0)
        if unit == "kb":
            return parsed / 1024.0
        if unit == "gb":
            return parsed * 1024.0
        if "bytes" in key:
            return parsed / (1024.0 * 1024.0)
        if "gb" in key:
            return parsed * 1024.0
        if "kb" in key:
            return parsed / 1024.0
        return parsed
    return None


def _memory_unit_for_key(flat: dict[str, Any], key: str) -> str | None:
    base = key.rsplit("_", 1)[0] if key.endswith("_value") else key
    candidates = (f"{base}_unit", key.replace("_value", "_unit"))
    for candidate in candidates:
        unit = flat.get(candidate)
        if isinstance(unit, str):
            norm = unit.strip().lower()
            if norm in {"b", "bytes", "byte"}:
                return "b"
            if norm in {"kb", "kib"}:
                return "kb"
            if norm in {"mb", "mib"}:
                return "mb"
            if norm in {"gb", "gib"}:
                return "gb"
    return None


def _norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _to_float(value: Any) -> float | None:
    if isinstance(value, dict):
        for key in ("value", "val", "current"):
            if key in value:
                return _to_float(value[key])
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    if not match:
        return None
    try:
        parsed = float(match.group(0))
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _to_int(value: Any) -> int | None:
    parsed = _to_float(value)
    return int(parsed) if parsed is not None else None


def _sum_optional(values) -> float | None:
    filtered = [value for value in values if value is not None]
    return sum(filtered) if filtered else None


def _mean_optional(values) -> float | None:
    filtered = [value for value in values if value is not None]
    return sum(filtered) / len(filtered) if filtered else None


__all__ = ["AmdSmiTelemetrySession"]
