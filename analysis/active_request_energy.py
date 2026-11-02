"""Conservative, overlap-aware attribution of one physical batch's energy.

Allocate physical attempts before selecting repeats, successful outputs, matched
hardware tasks, or economic weights. Retries are separate physical attempts and
their allocations must subsequently be summed into their logical task/repeat.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RequestInterval:
    request_id: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class BatchAttribution:
    request_energy_j: dict[str, float]
    equal_split_energy_j: dict[str, float]
    method: str
    no_active_energy_j: float
    no_active_seconds: float
    total_energy_j: float
    reconciliation_error_j: float


def attribute_batch(
    requests: Sequence[RequestInterval],
    *,
    batch_start_s: float,
    batch_end_s: float,
    batch_energy_j: float,
    counter_times_s: Sequence[float] | None = None,
    cumulative_energy_j: Sequence[float] | None = None,
) -> BatchAttribution:
    """Integrate dE(t)/N(t), sharing N=0 overhead equally across the batch.

    If cumulative meter samples are supplied, linearly interpolate the meter at
    every request boundary. Samples must cover the entire measurement window,
    be monotone, and reconcile with the authoritative batch total. Otherwise
    explicitly use constant average batch power. Missing timestamps are errors;
    neither simultaneous starts nor equal allocation are silently substituted.
    """
    left, right, total = map(float, (batch_start_s, batch_end_s, batch_energy_j))
    if not all(map(math.isfinite, (left, right, total))) or right <= left or total <= 0:
        raise ValueError("Finite positive batch energy and measurement duration required")
    if not requests or len({r.request_id for r in requests}) != len(requests):
        raise ValueError("Nonempty, unique physical request IDs required")
    starts = np.array([r.start_s for r in requests], dtype=float)
    ends = np.array([r.end_s for r in requests], dtype=float)
    if not np.isfinite(starts).all() or not np.isfinite(ends).all() or np.any(ends <= starts):
        raise ValueError("Each request needs finite, ordered start/end timestamps")
    if starts.min() < left - 1e-5 or ends.max() > right + 1e-5:
        raise ValueError("Request interval extends outside the measured batch")
    starts, ends = np.clip(starts, left, right), np.clip(ends, left, right)
    if (counter_times_s is None) != (cumulative_energy_j is None):
        raise ValueError("Both cumulative energy and sample times are required")
    if counter_times_s is None:
        times, energy = np.array([left, right]), np.array([0., total])
        method = "active_request_constant_batch_power_estimate"
    else:
        times, energy = np.asarray(counter_times_s, dtype=float), np.asarray(cumulative_energy_j, dtype=float)
        if times.ndim != 1 or energy.shape != times.shape or len(times) < 2:
            raise ValueError("Meter sample arrays must have matching one-dimensional shapes")
        if not np.isfinite(times).all() or not np.isfinite(energy).all() or np.any(np.diff(times) <= 0) or np.any(np.diff(energy) < 0):
            raise ValueError("Meter timestamps must increase and cumulative energy must not decrease")
        if times[0] > left or times[-1] < right:
            raise ValueError("Meter does not cover the entire measurement window")
        metered = float(np.interp(right, times, energy) - np.interp(left, times, energy))
        if not math.isclose(metered, total, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError(f"Meter total {metered} J differs from batch total {total} J")
        method = "active_request_metered_intervals"
    cuts = np.unique(np.r_[left, right, starts, ends, times[(times > left) & (times < right)]])
    meter = np.interp(cuts, times, energy)
    allocated = np.zeros(len(requests))
    no_active_energy = no_active_seconds = 0.
    for a, b, delta in zip(cuts[:-1], cuts[1:], np.diff(meter)):
        mid = (a + b) / 2
        active = (starts <= mid) & (ends > mid)
        count = int(active.sum())
        if count:
            allocated[active] += delta / count
        else:
            no_active_energy += delta
            no_active_seconds += b - a
            allocated += delta / len(requests)
    error = float(allocated.sum() - total)
    if abs(error) > max(1e-6, total * 1e-12):
        raise AssertionError("Energy allocation failed batch conservation")
    return BatchAttribution(
        {r.request_id: float(e) for r, e in zip(requests, allocated)},
        {r.request_id: total / len(requests) for r in requests},
        method, float(no_active_energy), float(no_active_seconds), total, error,
    )
