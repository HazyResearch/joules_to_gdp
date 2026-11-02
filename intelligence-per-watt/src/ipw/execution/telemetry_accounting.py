"""Shared telemetry accounting helpers for query, batch, and benchmark runs."""

from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .telemetry_session import TelemetrySample, TelemetrySession

TELEMETRY_ACCOUNTING_MODES = ("query", "batch", "benchmark", "per-action")


@dataclass
class BatchTelemetryRecord:
    """Aggregate telemetry for one concurrently executed query batch."""

    batch_id: str
    batch_index: int
    telemetry_accounting: str = "batch"
    query_ids: list[str] = field(default_factory=list)
    batch_size: int = 0
    batch_prompt_start_time_s: float | None = None
    batch_prompt_end_time_s: float | None = None
    batch_prompt_wall_clock_s: float | None = None
    batch_total_start_time_s: float | None = None
    batch_total_end_time_s: float | None = None
    batch_total_wall_clock_s: float | None = None
    prompt_finish_offsets_s: dict[str, float | None] = field(default_factory=dict)
    batch_gpu_energy_joules: float | None = None
    batch_cpu_energy_joules: float | None = None
    batch_dram_energy_joules: float | None = None
    batch_gpu_power_avg_watts: float | None = None
    batch_cpu_power_avg_watts: float | None = None
    batch_dram_power_avg_watts: float | None = None
    energy_window_coverage: dict[str, bool] = field(default_factory=dict)
    telemetry_sample_count: int = 0
    telemetry_device_ids: list[int] = field(default_factory=list)
    batch_mbu_avg_pct: float | None = None
    batch_mbu_max_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "batch_index": self.batch_index,
            "telemetry_accounting": self.telemetry_accounting,
            "query_ids": list(self.query_ids),
            "batch_size": self.batch_size,
            "batch_prompt_start_time_s": self.batch_prompt_start_time_s,
            "batch_prompt_end_time_s": self.batch_prompt_end_time_s,
            "batch_prompt_wall_clock_s": self.batch_prompt_wall_clock_s,
            "batch_total_start_time_s": self.batch_total_start_time_s,
            "batch_total_end_time_s": self.batch_total_end_time_s,
            "batch_total_wall_clock_s": self.batch_total_wall_clock_s,
            "prompt_finish_offsets_s": dict(self.prompt_finish_offsets_s),
            "batch_gpu_energy_joules": self.batch_gpu_energy_joules,
            "batch_cpu_energy_joules": self.batch_cpu_energy_joules,
            "batch_dram_energy_joules": self.batch_dram_energy_joules,
            "batch_gpu_power_avg_watts": self.batch_gpu_power_avg_watts,
            "batch_cpu_power_avg_watts": self.batch_cpu_power_avg_watts,
            "batch_dram_power_avg_watts": self.batch_dram_power_avg_watts,
            "energy_window_coverage": dict(self.energy_window_coverage),
            "telemetry_sample_count": self.telemetry_sample_count,
            "telemetry_device_ids": list(self.telemetry_device_ids),
            "batch_mbu_avg_pct": self.batch_mbu_avg_pct,
            "batch_mbu_max_pct": self.batch_mbu_max_pct,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchTelemetryRecord":
        return cls(
            batch_id=str(data.get("batch_id") or ""),
            batch_index=int(data.get("batch_index") or 0),
            telemetry_accounting=str(data.get("telemetry_accounting") or "batch"),
            query_ids=[str(value) for value in data.get("query_ids") or []],
            batch_size=int(data.get("batch_size") or 0),
            batch_prompt_start_time_s=data.get("batch_prompt_start_time_s"),
            batch_prompt_end_time_s=data.get("batch_prompt_end_time_s"),
            batch_prompt_wall_clock_s=data.get("batch_prompt_wall_clock_s"),
            batch_total_start_time_s=data.get("batch_total_start_time_s"),
            batch_total_end_time_s=data.get("batch_total_end_time_s"),
            batch_total_wall_clock_s=data.get("batch_total_wall_clock_s"),
            prompt_finish_offsets_s=dict(data.get("prompt_finish_offsets_s") or {}),
            batch_gpu_energy_joules=data.get("batch_gpu_energy_joules"),
            batch_cpu_energy_joules=data.get("batch_cpu_energy_joules"),
            batch_dram_energy_joules=data.get("batch_dram_energy_joules"),
            batch_gpu_power_avg_watts=data.get("batch_gpu_power_avg_watts"),
            batch_cpu_power_avg_watts=data.get("batch_cpu_power_avg_watts"),
            batch_dram_power_avg_watts=data.get("batch_dram_power_avg_watts"),
            energy_window_coverage=dict(data.get("energy_window_coverage") or {}),
            telemetry_sample_count=int(data.get("telemetry_sample_count") or 0),
            telemetry_device_ids=[
                int(value) for value in data.get("telemetry_device_ids") or []
            ],
            batch_mbu_avg_pct=data.get("batch_mbu_avg_pct"),
            batch_mbu_max_pct=data.get("batch_mbu_max_pct"),
        )


def normalize_telemetry_accounting(value: str | None, *, default: str = "query") -> str:
    """Return a supported accounting mode or raise ``ValueError``."""
    mode = (value or default).strip().lower()
    if mode not in TELEMETRY_ACCOUNTING_MODES:
        allowed = ", ".join(TELEMETRY_ACCOUNTING_MODES)
        raise ValueError(f"Unsupported telemetry_accounting={value!r}; expected one of {allowed}")
    return mode


def collect_telemetry_window(
    telemetry: TelemetrySession,
    start_time: float,
    end_time: float,
) -> list[TelemetrySample]:
    """Collect telemetry samples plus nearest boundary samples for interpolation."""
    samples = sorted(list(telemetry.readings()), key=lambda sample: sample.timestamp)
    if not samples:
        return list(telemetry.window(start_time, end_time))

    before = None
    inside: list[TelemetrySample] = []
    after = None
    for sample in samples:
        if sample.timestamp < start_time:
            before = sample
        elif sample.timestamp <= end_time:
            inside.append(sample)
        else:
            after = sample
            break

    selected: list[TelemetrySample] = []
    for sample in [before, *inside, after]:
        if sample is None:
            continue
        if selected and selected[-1].timestamp == sample.timestamp:
            continue
        selected.append(sample)
    return selected


def compute_energy_delta(
    readings: Sequence[TelemetrySample],
    field: str,
    start_time: float | None = None,
    end_time: float | None = None,
) -> float | None:
    """Compute a non-negative cumulative energy delta for ``field``."""
    points = sorted(
        (
            (sample.timestamp, getattr(sample.reading, field))
            for sample in readings
            if getattr(sample.reading, field, None) is not None
            and math.isfinite(getattr(sample.reading, field))
        ),
        key=lambda item: item[0],
    )
    if len(points) < 2:
        return None

    if start_time is not None and end_time is not None and end_time >= start_time:
        if start_time > points[-1][0] or end_time < points[0][0]:
            delta = points[-1][1] - points[0][1]
            return delta if delta >= 0 else None
        start_value = _interpolate_cumulative_value(points, start_time)
        end_value = _interpolate_cumulative_value(points, end_time)
        if start_value is None or end_value is None:
            return None
        delta = end_value - start_value
        return delta if delta >= 0 else None

    delta = points[-1][1] - points[0][1]
    return delta if delta >= 0 else None


def compute_power_avg(readings: Sequence[TelemetrySample], field: str) -> float | None:
    values = [
        getattr(sample.reading, field)
        for sample in readings
        if getattr(sample.reading, field, None) is not None
        and math.isfinite(getattr(sample.reading, field))
    ]
    return statistics.mean(values) if values else None


def estimate_energy_from_power(
    readings: Sequence[TelemetrySample],
    power_field: str,
    duration_s: float,
) -> float | None:
    if duration_s <= 0:
        return None
    avg_power = compute_power_avg(readings, power_field)
    if avg_power is not None and avg_power > 0:
        return avg_power * duration_s
    return None


def build_batch_record(
    *,
    batch_id: str,
    batch_index: int,
    query_ids: Sequence[str],
    prompt_start_time: float,
    prompt_end_time: float,
    total_start_time: float | None = None,
    total_end_time: float | None = None,
    prompt_finish_offsets_s: dict[str, float | None] | None = None,
    readings: Sequence[TelemetrySample] = (),
    telemetry_device_ids: Sequence[int] = (),
) -> BatchTelemetryRecord:
    """Build a ``BatchTelemetryRecord`` from one prompt-batch window."""
    prompt_wall = max(prompt_end_time - prompt_start_time, 0.0)
    total_start = total_start_time if total_start_time is not None else prompt_start_time
    total_end = total_end_time if total_end_time is not None else prompt_end_time
    max_sample_gap_s = _required_sample_gap_limit()
    counter_fields = {
        "gpu": "energy_joules", "cpu": "cpu_energy_joules", "dram": "dram_energy_joules",
    }
    power_fields = {
        "gpu": "power_watts", "cpu": "cpu_power_watts", "dram": "dram_power_watts",
    }
    counter_coverage = {
        component: _covers_interval(
            readings, field, prompt_start_time, prompt_end_time, max_sample_gap_s,
        )
        for component, field in counter_fields.items()
    }
    power_coverage = {
        component: _covers_interval(
            readings, field, prompt_start_time, prompt_end_time, max_sample_gap_s,
        )
        for component, field in power_fields.items()
    }
    coverage = {component: counter_coverage[component] or power_coverage[component] for component in counter_coverage}
    gpu_energy = (
        compute_energy_delta(readings, "energy_joules", prompt_start_time, prompt_end_time)
        if counter_coverage["gpu"] else None
    )
    cpu_energy = (
        compute_energy_delta(readings, "cpu_energy_joules", prompt_start_time, prompt_end_time)
        if counter_coverage["cpu"] else None
    )
    dram_energy = (
        compute_energy_delta(readings, "dram_energy_joules", prompt_start_time, prompt_end_time)
        if counter_coverage["dram"] else None
    )
    if gpu_energy is None and power_coverage["gpu"]:
        gpu_energy = estimate_energy_from_power(readings, "power_watts", prompt_wall)
    if cpu_energy is None and power_coverage["cpu"]:
        cpu_energy = estimate_energy_from_power(readings, "cpu_power_watts", prompt_wall)
    if dram_energy is None and power_coverage["dram"]:
        dram_energy = estimate_energy_from_power(readings, "dram_power_watts", prompt_wall)

    mbu_values = [
        sample.reading.gpu_memory_bandwidth_utilization_pct
        for sample in readings
        if getattr(sample.reading, "gpu_memory_bandwidth_utilization_pct", None) is not None
        and sample.reading.gpu_memory_bandwidth_utilization_pct >= 0
    ]
    return BatchTelemetryRecord(
        batch_id=batch_id,
        batch_index=batch_index,
        query_ids=[str(query_id) for query_id in query_ids],
        batch_size=len(query_ids),
        batch_prompt_start_time_s=prompt_start_time,
        batch_prompt_end_time_s=prompt_end_time,
        batch_prompt_wall_clock_s=prompt_wall,
        batch_total_start_time_s=total_start,
        batch_total_end_time_s=total_end,
        batch_total_wall_clock_s=max(total_end - total_start, 0.0),
        prompt_finish_offsets_s=dict(prompt_finish_offsets_s or {}),
        batch_gpu_energy_joules=gpu_energy,
        batch_cpu_energy_joules=cpu_energy,
        batch_dram_energy_joules=dram_energy,
        batch_gpu_power_avg_watts=compute_power_avg(readings, "power_watts"),
        batch_cpu_power_avg_watts=compute_power_avg(readings, "cpu_power_watts"),
        batch_dram_power_avg_watts=compute_power_avg(readings, "dram_power_watts"),
        energy_window_coverage=coverage,
        telemetry_sample_count=len(readings),
        telemetry_device_ids=[int(value) for value in telemetry_device_ids],
        batch_mbu_avg_pct=statistics.mean(mbu_values) if mbu_values else None,
        batch_mbu_max_pct=max(mbu_values) if mbu_values else None,
    )


def _required_sample_gap_limit() -> float | None:
    """Bound missing meter ticks for measured, full-energy runs."""
    if os.getenv("IPW_REQUIRE_TOTAL_ENERGY") != "1":
        return None
    raw_interval_ms = os.getenv("IPW_ENERGY_MONITOR_INTERVAL_MS")
    try:
        interval_ms = int(raw_interval_ms) if raw_interval_ms is not None else 0
    except ValueError as exc:
        raise ValueError("Full-energy accounting requires a positive meter interval") from exc
    if interval_ms <= 0:
        raise ValueError("Full-energy accounting requires a positive meter interval")
    # Leave room for scheduling jitter while rejecting a paused stream. RAPL
    # counter deltas cannot establish how many wraps occurred across long gaps.
    return 5 * interval_ms / 1000.0


def _covers_interval(
    readings: Sequence[TelemetrySample], field: str, start_time: float, end_time: float,
    max_sample_gap_s: float | None = None,
) -> bool:
    if max_sample_gap_s is not None:
        if end_time <= start_time or any(
            not math.isfinite(sample.timestamp) for sample in readings
        ):
            return False
        ordered = sorted(readings, key=lambda sample: sample.timestamp)
        before = None
        after = None
        for index, sample in enumerate(ordered):
            if sample.timestamp <= start_time:
                before = index
            if after is None and sample.timestamp >= end_time:
                after = index
        if before is None or after is None or after <= before:
            return False
        window = ordered[before : after + 1]
        previous_time: float | None = None
        previous_value: float | None = None
        for sample in window:
            timestamp = sample.timestamp
            value = getattr(sample.reading, field, None)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                return False
            if previous_time is not None:
                if (
                    timestamp <= previous_time
                    or timestamp - previous_time > max_sample_gap_s + 1e-9
                ):
                    return False
                if (
                    field.endswith("energy_joules")
                    and previous_value is not None
                    and value < previous_value
                ):
                    return False
            previous_time, previous_value = timestamp, value
        return True

    points = [
        sample.timestamp
        for sample in readings
        if getattr(sample.reading, field, None) is not None
        and math.isfinite(getattr(sample.reading, field))
    ]
    return len(points) >= 2 and min(points) <= start_time and max(points) >= end_time


def export_batches_jsonl(
    batches: Sequence[BatchTelemetryRecord],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for batch in batches:
            handle.write(json.dumps(batch.to_dict()) + "\n")
    return path


def load_batches_jsonl(path: Path) -> list[BatchTelemetryRecord]:
    batches: list[BatchTelemetryRecord] = []
    if not path.exists():
        return batches
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        value = json.loads(stripped)
        if isinstance(value, dict):
            batches.append(BatchTelemetryRecord.from_dict(value))
    return batches


def summarize_batch_records(
    batches: Sequence[BatchTelemetryRecord],
    *,
    query_count: int,
) -> dict[str, Any] | None:
    if not batches:
        return None
    gpu_values = [b.batch_gpu_energy_joules for b in batches if b.batch_gpu_energy_joules is not None]
    cpu_values = [b.batch_cpu_energy_joules for b in batches if b.batch_cpu_energy_joules is not None]
    dram_values = [b.batch_dram_energy_joules for b in batches if b.batch_dram_energy_joules is not None]
    power_values = [b.batch_gpu_power_avg_watts for b in batches if b.batch_gpu_power_avg_watts is not None]
    cpu_power_values = [b.batch_cpu_power_avg_watts for b in batches if b.batch_cpu_power_avg_watts is not None]
    dram_power_values = [b.batch_dram_power_avg_watts for b in batches if b.batch_dram_power_avg_watts is not None]
    wall_values = [b.batch_prompt_wall_clock_s for b in batches if b.batch_prompt_wall_clock_s is not None]
    total_gpu = sum(gpu_values) if gpu_values else None
    total_cpu = sum(cpu_values) if cpu_values else None
    total_dram = sum(dram_values) if dram_values else None
    return {
        "telemetry_accounting": "batch",
        "batch_count": len(batches),
        "query_count": query_count,
        "total_batch_gpu_energy_joules": total_gpu,
        "total_batch_cpu_energy_joules": total_cpu,
        "total_batch_dram_energy_joules": total_dram,
        "avg_batch_gpu_power_watts": statistics.mean(power_values) if power_values else None,
        "avg_batch_cpu_power_watts": statistics.mean(cpu_power_values) if cpu_power_values else None,
        "avg_batch_dram_power_watts": statistics.mean(dram_power_values) if dram_power_values else None,
        "batch_prompt_wall_clock_s": _stat_summary(wall_values),
        "amortized_gpu_energy_per_query_joules": (
            total_gpu / query_count if total_gpu is not None and query_count > 0 else None
        ),
        "amortized_cpu_energy_per_query_joules": (
            total_cpu / query_count if total_cpu is not None and query_count > 0 else None
        ),
        "amortized_dram_energy_per_query_joules": (
            total_dram / query_count if total_dram is not None and query_count > 0 else None
        ),
        "batches": [batch.to_dict() for batch in batches],
    }


def _interpolate_cumulative_value(
    points: list[tuple[float, float]],
    timestamp: float,
) -> float | None:
    if not points:
        return None
    if timestamp <= points[0][0]:
        return points[0][1]
    if timestamp >= points[-1][0]:
        return points[-1][1]

    previous_time, previous_value = points[0]
    for current_time, current_value in points[1:]:
        if current_time < timestamp:
            previous_time, previous_value = current_time, current_value
            continue
        span = current_time - previous_time
        if span <= 0:
            return current_value
        fraction = (timestamp - previous_time) / span
        return previous_value + ((current_value - previous_value) * fraction)
    return points[-1][1]


def _stat_summary(values: Iterable[float]) -> dict[str, float | None]:
    clean = [float(value) for value in values]
    if not clean:
        return {"avg": None, "median": None, "min": None, "max": None, "std": None}
    return {
        "avg": statistics.mean(clean),
        "median": statistics.median(clean),
        "min": min(clean),
        "max": max(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
    }


__all__ = [
    "BatchTelemetryRecord",
    "TELEMETRY_ACCOUNTING_MODES",
    "build_batch_record",
    "collect_telemetry_window",
    "compute_energy_delta",
    "compute_power_avg",
    "estimate_energy_from_power",
    "export_batches_jsonl",
    "load_batches_jsonl",
    "normalize_telemetry_accounting",
    "summarize_batch_records",
]
