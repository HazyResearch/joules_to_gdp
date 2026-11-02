"""Hardware preflight: sample GPU/CPU utilization to detect contamination.

Run before the first benchmark query. Whole-device energy measurement inflates
per-query attribution proportionally to any other workload on the same GPU/CPU,
so we sample baseline utilization and flag (or abort) when contamination is
detected.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_log = logging.getLogger(__name__)


def _allowed_gpu_pids() -> set[int]:
    """Return the harness and configured inference server process trees.

    A server started by the deployment launcher is a sibling of this CLI, so
    its PID must be provided explicitly. psutil expands children because vLLM
    often forks GPU workers after startup.
    """
    roots = {os.getpid()}
    raw = os.getenv("IPW_PREFLIGHT_ALLOWED_GPU_PIDS", "")
    for value in raw.split(","):
        value = value.strip()
        if not value:
            continue
        try:
            pid = int(value)
        except ValueError as exc:
            raise ValueError(
                "IPW_PREFLIGHT_ALLOWED_GPU_PIDS must be comma-separated positive PIDs"
            ) from exc
        if pid <= 0:
            raise ValueError(
                "IPW_PREFLIGHT_ALLOWED_GPU_PIDS must be comma-separated positive PIDs"
            )
        roots.add(pid)

    try:
        import psutil
    except ImportError:
        return roots
    allowed = set(roots)
    for pid in roots:
        try:
            allowed.update(child.pid for child in psutil.Process(pid).children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return allowed


@dataclass
class PreflightResult:
    gpu_util_pct_avg: Optional[float]
    cpu_util_pct_avg: Optional[float]
    foreign_gpu_pids: List[int]
    shared_device_baseline_dirty: bool
    warnings: List[str] = field(default_factory=list)


def run_preflight(
    *,
    sample_seconds: float = 3.0,
    gpu_util_threshold_pct: float = 5.0,
    cpu_util_threshold_pct: float = 10.0,
    strict: bool = False,
) -> PreflightResult:
    """Run preflight check. Returns a PreflightResult.

    Raises RuntimeError in strict mode if contamination is detected or either
    hardware sampler cannot establish a clean baseline.
    """
    warnings: List[str] = []

    # Reject a malformed allowlist before the broad sampling exception handler
    # can turn the configuration error into a non-blocking NVML warning.
    if os.getenv("IPW_PREFLIGHT_ALLOWED_GPU_PIDS"):
        _allowed_gpu_pids()

    try:
        gpu_util, foreign_pids = _sample_gpu_util(sample_seconds)
    except Exception as exc:
        if strict:
            raise RuntimeError(f"Strict hardware preflight could not sample NVIDIA GPUs: {exc}") from exc
        _log.warning("GPU utilization sampling failed: %s", exc)
        warnings.append(f"NVML unavailable: {exc}")
        gpu_util = None
        foreign_pids = []

    try:
        cpu_util = _sample_cpu_util(sample_seconds)
    except Exception as exc:
        if strict:
            raise RuntimeError(f"Strict hardware preflight could not sample CPU utilization: {exc}") from exc
        _log.warning("CPU utilization sampling failed: %s", exc)
        warnings.append(f"psutil unavailable: {exc}")
        cpu_util = None

    dirty = False
    if gpu_util is not None and gpu_util > gpu_util_threshold_pct:
        dirty = True
        warnings.append(f"GPU baseline {gpu_util:.1f}% > {gpu_util_threshold_pct}%")
    if cpu_util is not None and cpu_util > cpu_util_threshold_pct:
        dirty = True
        warnings.append(f"CPU baseline {cpu_util:.1f}% > {cpu_util_threshold_pct}%")
    if foreign_pids:
        dirty = True
        warnings.append(f"Foreign GPU processes detected: {foreign_pids}")

    result = PreflightResult(
        gpu_util_pct_avg=gpu_util,
        cpu_util_pct_avg=cpu_util,
        foreign_gpu_pids=foreign_pids,
        shared_device_baseline_dirty=dirty,
        warnings=warnings,
    )

    if strict and dirty:
        raise RuntimeError(
            "Shared-device contamination detected in strict preflight mode: "
            + "; ".join(warnings)
        )

    return result


def _sample_gpu_util(sample_seconds: float) -> Tuple[float, List[int]]:
    """Sample average GPU utilization and list foreign PIDs via NVML.

    Returns (avg_util_pct, foreign_pids). Raises if NVML is unavailable.
    """
    try:
        import pynvml  # type: ignore
    except ImportError as e:
        raise RuntimeError(f"pynvml not installed: {e}") from e

    pynvml.nvmlInit()
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            return 0.0, []

        allowed_pids = _allowed_gpu_pids()
        foreign_pids: List[int] = []
        utils: List[float] = []

        end_time = time.monotonic() + sample_seconds
        # At least one sweep regardless of sample_seconds
        first_sweep = True
        while first_sweep or time.monotonic() < end_time:
            first_sweep = False
            for i in range(device_count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                u = pynvml.nvmlDeviceGetUtilizationRates(h)
                utils.append(float(u.gpu))
            if sample_seconds > 0.0:
                time.sleep(0.1)
            else:
                break

        for i in range(device_count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            # Failure here leaves foreign GPU users unknown. Let strict callers
            # abort; lenient callers can keep the warning-only behavior.
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            for p in procs:
                if int(p.pid) not in allowed_pids:
                    foreign_pids.append(int(p.pid))

        avg = sum(utils) / len(utils) if utils else 0.0
        return avg, sorted(set(foreign_pids))
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception as exc:  # noqa: BLE001
            _log.debug("nvmlShutdown error (ignored): %s", exc)


def _sample_cpu_util(sample_seconds: float) -> float:
    """Sample average CPU utilization percent via psutil."""
    try:
        import psutil
    except ImportError as e:
        raise RuntimeError(f"psutil not installed: {e}") from e
    if sample_seconds <= 0.0:
        return psutil.cpu_percent(interval=None)
    return psutil.cpu_percent(interval=sample_seconds)


__all__ = ["PreflightResult", "run_preflight"]
