"""Agentic runner for multi-turn agent benchmarking with energy telemetry."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import inspect
import json
import logging
import math
import os
import re
import signal
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from tqdm.auto import tqdm

from ..agents.base import BaseAgent, ToolUsingAgent
from ..core.types import AgentRunResult, DatasetRecord
from ..datasets.base import DatasetProvider
from ..execution.telemetry_accounting import (
    BatchTelemetryRecord,
    build_batch_record,
    normalize_telemetry_accounting,
)
from ..execution.telemetry_session import TelemetrySample, TelemetrySession
from ..execution.trace import QueryTrace, TurnTrace
from ..execution.types import (
    ComputeMetrics,
    CostMetrics,
    DerivedEfficiencyMetrics,
    EnergyMetrics,
    LatencyMetrics,
    MemoryMetrics,
    MetricStats,
    ModelMetrics,
    PowerComponentMetrics,
    PowerMetrics,
    ProfilingRecord,
    TokenMetrics,
)
from ..telemetry.energy_attribution import EnergyAttribution
from ..telemetry.events import EventRecorder, EventType
from .executor import Executor
from .preflight import run_preflight

LOGGER = logging.getLogger(__name__)


def _env_truthy(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


class _QueryTimeout(BaseException):
    """Internal timeout sentinel that bypasses per-query Exception handling."""


def _raise_query_timeout(_signum: int, _frame: Any) -> None:
    raise _QueryTimeout()


def _classify_agent_exception(exc: BaseException) -> tuple[str, str]:
    """Return ``(failure_type, reason)`` for trace metadata."""
    explicit_failure_type = getattr(exc, "ipw_failure_type", None)
    if isinstance(explicit_failure_type, str) and explicit_failure_type:
        return explicit_failure_type, explicit_failure_type
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if (
        isinstance(exc, (asyncio.TimeoutError, _QueryTimeout))
        or "timeout" in lowered
        or "apitimeouterror" in lowered
        or "request timed out" in lowered
        or "litellm.timeout" in lowered
    ):
        return "infra_timeout", "agent_or_llm_timeout"
    if "connection" in lowered or "server disconnected" in lowered:
        return "infra_connection_error", "agent_or_llm_connection_error"
    if (
        "codex_cli" in lowered
        or "codex cli" in lowered
        or "provider config" in lowered
        or "model_provider" in lowered
        or "missing final message" in lowered
    ):
        return "infra_harness_error", "agent_harness_error"
    return "agent_error", "agent_error"


def _classify_agent_result_failure(result: AgentRunResult) -> tuple[str | None, str | None]:
    """Detect agents that returned a failure payload instead of raising."""
    metadata = dict(result.metadata or {})
    explicit_failure_type = metadata.get("failure_type")
    if explicit_failure_type:
        return (
            str(explicit_failure_type),
            str(
                metadata.get("unscorable_reason")
                or metadata.get("reason")
                or metadata.get("run_status")
                or "agent_run_failed"
            ),
        )
    raw_status = str(metadata.get("status") or metadata.get("run_status") or "").lower()
    if raw_status in {"failed", "failure", "error", "timeout", "timed_out"}:
        text = " ".join(
            str(part)
            for part in [
                metadata.get("error"),
                metadata.get("reason"),
                result.content,
            ]
            if part
        )
        failure_type, reason = _classify_agent_exception(RuntimeError(text or raw_status))
        if raw_status in {"timeout", "timed_out"}:
            return "infra_timeout", "agent_or_llm_timeout"
        if failure_type == "agent_error":
            return failure_type, "agent_run_failed"
        return failure_type, reason

    text = " ".join(
        str(part)
        for part in [
            metadata.get("error"),
            metadata.get("exception"),
            result.content,
        ]
        if part
    )
    lowered = text.lower()
    failure_markers = (
        "conversation run failed",
        "agent run failed",
        "openhands tb agent failed",
    )
    if not any(marker in lowered for marker in failure_markers):
        return None, None
    failure_type, reason = _classify_agent_exception(RuntimeError(text))
    if failure_type == "agent_error":
        reason = "agent_run_failed"
    return failure_type, reason


def _uses_stirrup_single_submission_contract(
    record: DatasetRecord,
    result: AgentRunResult,
) -> bool:
    """Detect AA-style Stirrup GDPval runs.

    Stirrup owns the full task-submission loop. Running IPW's generic GDPval
    repair/checklist prompts after Stirrup returns adds extra agent turns and
    can merge multiple finish calls into one query, so those prompts are
    intentionally skipped for this contract.
    """
    metadata = dict(record.dataset_metadata or {})
    result_metadata = dict(result.metadata or {})
    workload_type = str(metadata.get("workload_type") or "").lower()
    dataset_name = str(metadata.get("dataset_name") or "").lower()
    if workload_type == "gdpval-stirrup" or "stirrup" in dataset_name:
        return True
    return any(
        key in result_metadata
        for key in (
            "stirrup_finish_params",
            "stirrup_finished",
            "stirrup_abandoned",
        )
    )


_GDPVAL_FINAL_NUDGE_TEXT = (
    "Submit your final answer now with the finish tool. "
    "Do not call terminal or file_editor. If output files already exist, "
    "finish with a concise list of their paths."
)
_GDPVAL_CONTROL_LINE_RE = re.compile(
    r"^(?:USER_CONTEXT|TASK_TRACKING|COMPLETED|PENDING|CURRENT_STATE|"
    r"CODE_STATE|TESTS|CHANGES|DEPS|VERSION_CONTROL_STATUS):.*$",
    re.MULTILINE,
)
_GDPVAL_SECTION_RE = re.compile(
    r"^\s*---\s*(GDPval (?:deliverable repair|final deliverable checklist) "
    r"attempt \d+)\s*---\s*$",
    re.MULTILINE,
)


def _sanitize_gdpval_response_text(content: str) -> str:
    """Remove harness-control chatter from GDPval final answers."""
    text = str(content or "")
    if not text:
        return ""
    text = text.replace(_GDPVAL_FINAL_NUDGE_TEXT, "")
    text = re.sub(
        r"Submit your final answer now with the finish tool\.\s*"
        r"Do not call terminal or file_editor\.\s*"
        r"If output files already exist, finish with a concise list of their paths\.",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = _GDPVAL_CONTROL_LINE_RE.sub("", text)
    text = _GDPVAL_SECTION_RE.sub(r"--- \1 ---", text)

    compact: list[str] = []
    blank = False
    for line in (line.rstrip() for line in text.splitlines()):
        if not line.strip():
            if compact and not blank:
                compact.append("")
            blank = True
            continue
        compact.append(line)
        blank = False
    return "\n".join(compact).strip()


def _gdpval_finish_summary_from_validation(validation: dict[str, Any] | None) -> str:
    if not validation or validation.get("status") == "invalid":
        return ""
    paths = validation.get("primary_files") or validation.get("files") or []
    if not paths:
        return ""
    lines = ["Submitted deliverable file(s):"]
    for path in paths:
        lines.append(f"- {Path(str(path)).name}")
    return "\n".join(lines)


def _replace_agent_result_content(
    result: AgentRunResult,
    content: str,
) -> AgentRunResult:
    return AgentRunResult(
        content=content,
        tool_calls_attempted=result.tool_calls_attempted,
        tool_calls_succeeded=result.tool_calls_succeeded,
        tool_names_used=list(result.tool_names_used or []),
        num_turns=result.num_turns,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        trace=result.trace,
        metadata=dict(result.metadata or {}),
    )


def _normalize_gdpval_result_content(
    record: DatasetRecord,
    result: AgentRunResult,
) -> AgentRunResult:
    dataset_name = str(record.dataset_metadata.get("dataset_name") or "")
    if "gdpval" not in dataset_name.lower():
        return result
    cleaned = _sanitize_gdpval_response_text(result.content or "")
    validation = record.dataset_metadata.get("deliverable_validation")
    if not cleaned:
        cleaned = _gdpval_finish_summary_from_validation(
            validation if isinstance(validation, dict) else None
        )
    if cleaned and cleaned != (result.content or ""):
        return _replace_agent_result_content(result, cleaned)
    return result


def _is_gdpval_record(record: DatasetRecord) -> bool:
    dataset_name = str(record.dataset_metadata.get("dataset_name") or "")
    return "gdpval" in dataset_name.lower()


def _gdpval_workspace_signature(record: DatasetRecord) -> tuple[int, int, int]:
    outputs_dir = record.dataset_metadata.get("gdpval_outputs_dir")
    if not outputs_dir:
        return (0, 0, 0)
    root = Path(str(outputs_dir))
    if not root.exists() or not root.is_dir():
        return (0, 0, 0)
    count = 0
    total_size = 0
    latest_mtime_ns = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        count += 1
        total_size += int(stat.st_size)
        latest_mtime_ns = max(latest_mtime_ns, int(stat.st_mtime_ns))
    return (count, total_size, latest_mtime_ns)


# ---------------------------------------------------------------------------
# Energy computation helpers
# ---------------------------------------------------------------------------


def _compute_energy_delta(
    readings: list[TelemetrySample],
    field: str,
    start_time: float | None = None,
    end_time: float | None = None,
) -> float | None:
    """Compute energy delta from first to last reading for *field*."""
    points = sorted(
        (
            (s.timestamp, getattr(s.reading, field))
            for s in readings
            if getattr(s.reading, field, None) is not None
            and math.isfinite(getattr(s.reading, field))
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
    if delta >= 0:
        return delta
    return None


def _interpolate_cumulative_value(
    points: list[tuple[float, float]],
    timestamp: float,
) -> float | None:
    """Interpolate a cumulative telemetry counter at an exact timestamp."""
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


def _collect_telemetry_window(
    telemetry: TelemetrySession,
    start_time: float,
    end_time: float,
) -> list[TelemetrySample]:
    """Collect samples inside a window plus nearest boundary samples.

    The extra boundary samples let cumulative-energy deltas be interpolated at
    the exact prompt/judge start and end times without resetting the underlying
    telemetry counter.
    """
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


def _compute_power_avg(
    readings: list[TelemetrySample],
    field: str,
) -> float | None:
    """Compute average power across readings for *field*."""
    values = [
        getattr(s.reading, field)
        for s in readings
        if getattr(s.reading, field, None) is not None
        and math.isfinite(getattr(s.reading, field))
    ]
    return statistics.mean(values) if values else None


def _estimate_energy_from_power(
    readings: list[TelemetrySample],
    power_field: str,
    duration_s: float,
) -> float | None:
    """Fallback: energy ≈ avg_power × duration when cumulative counters unavailable."""
    if duration_s <= 0:
        return None
    avg_power = _compute_power_avg(readings, power_field)
    if avg_power is not None and avg_power > 0:
        return avg_power * duration_s
    return None


# ---------------------------------------------------------------------------
# Patch extraction helpers
# ---------------------------------------------------------------------------

_FENCED_DIFF_RE = re.compile(
    r"```(?:diff|patch)\s*\n(.*?)```", re.DOTALL
)
_UNIFIED_DIFF_MARKERS = ("diff --git", "--- a/", "+++ b/", "@@ ")


def _extract_patch(text: str) -> Optional[str]:
    """Extract a unified-diff patch from agent response text.

    Looks for fenced ``diff`` code blocks first, then falls back to raw
    unified-diff markers.  Returns ``None`` when no patch is detected.
    """
    # 1. Fenced ```diff blocks
    fenced = _FENCED_DIFF_RE.findall(text)
    if fenced:
        return "\n\n".join(block.strip() for block in fenced)

    # 2. Raw unified diff markers
    lines = text.splitlines()
    patch_lines: list[str] = []
    in_diff = False
    for line in lines:
        if any(line.startswith(m) for m in _UNIFIED_DIFF_MARKERS):
            in_diff = True
        if in_diff:
            patch_lines.append(line)

    if patch_lines:
        return "\n".join(patch_lines)
    return None


class AgenticRunner:
    """Orchestrate multi-turn agent runs with energy telemetry correlation.

    Similar to ProfilerRunner but designed for agentic workloads where a single
    query may involve multiple LLM turns and tool calls.
    """

    _FLUSH_INTERVAL = 50

    def __init__(
        self,
        agent: BaseAgent,
        dataset: DatasetProvider,
        telemetry_session: Optional[TelemetrySession] = None,
        config: Optional[dict[str, Any]] = None,
        event_recorder: Optional[EventRecorder] = None,
        run_dir: Optional[Path] = None,
        concurrency: int = 1,
        agent_factory: Optional[Callable[..., BaseAgent]] = None,
        query_timeout: Optional[float] = None,
        max_attempts: int = 3,
        max_turns: int = 10,
        require_dedicated_hardware: bool = False,
        telemetry_accounting: str = "query",
    ) -> None:
        """Initialise the agentic runner.

        ``max_attempts`` is the **attempt count** passed directly to
        ``Executor(max_attempts_per_turn=max_attempts)``.  A value of 1 means
        no retry; 3 (the default) allows two retries after the first attempt.
        A caller using *retries* semantics (extra attempts beyond the first)
        should convert ``retries + 1`` to the attempt count before passing it
        here.
        """
        self._agent = agent
        self._dataset = dataset
        self._telemetry = telemetry_session
        self._config = config or {}
        self._event_recorder = event_recorder if event_recorder is not None else EventRecorder()
        self._run_dir = run_dir
        self._traces: list[QueryTrace] = []
        self._records: list[ProfilingRecord] = []
        self._batch_records: list[BatchTelemetryRecord] = []
        self._concurrency = max(1, concurrency)
        self._telemetry_accounting = normalize_telemetry_accounting(
            telemetry_accounting,
            default=str(self._config.get("telemetry_accounting") or "query"),
        )
        self._config["telemetry_accounting"] = self._telemetry_accounting
        self._agent_factory = agent_factory
        self._query_timeout = query_timeout
        self._max_attempts = max_attempts
        self._max_turns = max_turns
        self._gdpval_repair_attempts = int(
            self._config.get("gdpval_repair_attempts", 1)
        )
        self._gdpval_final_check_attempts = int(
            self._config.get("gdpval_final_check_attempts", 0)
        )
        self._gdpval_file_settle_seconds = float(
            self._config.get(
                "gdpval_file_settle_seconds",
                os.getenv("IPW_GDPVAL_FILE_SETTLE_SECONDS", "2"),
            )
        )
        self._gdpval_file_settle_poll_seconds = max(
            0.05,
            float(
                self._config.get(
                    "gdpval_file_settle_poll_seconds",
                    os.getenv("IPW_GDPVAL_FILE_SETTLE_POLL_SECONDS", "1"),
                )
            ),
        )
        self._gdpval_autorun_repair_scripts = bool(
            self._config.get(
                "gdpval_autorun_repair_scripts",
                _env_truthy("IPW_GDPVAL_AUTORUN_REPAIR_SCRIPTS", default=False),
            )
        )
        self._gdpval_repair_script_timeout_seconds = max(
            1.0,
            float(
                self._config.get(
                    "gdpval_repair_script_timeout_seconds",
                    os.getenv("IPW_GDPVAL_REPAIR_SCRIPT_TIMEOUT_SECONDS", "300"),
                )
            ),
        )
        self._require_dedicated_hardware = require_dedicated_hardware
        self._results_lock = threading.Lock()
        self._preflight_done: bool = False
        self._preflight_baseline_dirty: bool = False
        self._energy_attribution: Optional[EnergyAttribution] = None
        self._attach_energy_attribution()

    @property
    def batch_records(self) -> list[BatchTelemetryRecord]:
        return list(self._batch_records)

    def _telemetry_device_ids(self) -> list[int]:
        """Return the configured GPU ids represented by query-level telemetry."""
        raw_ids = self._config.get("telemetry_device_ids") or []
        ids: list[int] = []
        for raw_id in raw_ids:
            try:
                ids.append(int(raw_id))
            except (TypeError, ValueError):
                continue
        if ids:
            return ids
        legacy_id = self._config.get("telemetry_gpu_id")
        if legacy_id is not None:
            try:
                return [int(legacy_id)]
            except (TypeError, ValueError):
                return []
        return []

    def _annotate_trace_window(
        self,
        trace: QueryTrace,
        *,
        prompt_start_time: float,
        prompt_end_time: float,
        prompt_readings: list[TelemetrySample],
        judge_start_time: float | None = None,
        judge_end_time: float | None = None,
    ) -> QueryTrace:
        """Attach prompt/judge timing metadata used to audit energy isolation."""
        trace.prompt_start_time_s = prompt_start_time
        trace.prompt_end_time_s = prompt_end_time
        trace.judge_start_time_s = judge_start_time
        trace.judge_end_time_s = judge_end_time
        trace.telemetry_device_ids = self._telemetry_device_ids()
        trace.telemetry_sample_count = len(prompt_readings)
        return trace

    def _attach_energy_attribution(self) -> None:
        """Attach the shadow EnergyAttribution subscriber to the recorder bus.

        The subscriber observes TOOL_CALL_*/LM_INFERENCE_* events as they flow
        through the bus and emits ENERGY_ATTRIBUTED events for downstream
        consumers to read.
        """
        if self._telemetry is None:
            return
        if self._energy_attribution is not None:
            return
        self._energy_attribution = EnergyAttribution(
            bus=self._event_recorder.bus,
            session=self._telemetry,
            is_cloud_fn=lambda evt: bool(evt.payload.get("is_cloud", False)),
            shared_device_warning_fn=lambda: self._preflight_baseline_dirty,
        )

    def _run_preflight_if_needed(self, *, strict: bool = False) -> None:
        """Run hardware preflight once per runner lifetime.

        Whole-device energy measurement inflates per-query attribution proportionally
        to other GPU/CPU workloads. Preflight samples baseline utilization once;
        the resulting `_preflight_baseline_dirty` flag propagates to subsequent
        TurnTrace/QueryTrace records as `shared_device_warning`.
        """
        if self._preflight_done:
            return
        result = run_preflight(strict=strict)
        self._preflight_done = True
        self._preflight_baseline_dirty = result.shared_device_baseline_dirty
        if self._preflight_baseline_dirty:
            for msg in result.warnings:
                LOGGER.warning("Preflight: %s", msg)

    async def run(
        self,
        max_queries: Optional[int] = None,
        *,
        start_index: int = 0,
    ) -> list[QueryTrace]:
        """Run the agent over the dataset, collecting traces and telemetry.

        Args:
            max_queries: Maximum number of records to process from start_index.
                None means all.
            start_index: First dataset record index to process.

        Returns:
            List of QueryTrace objects with energy-correlated telemetry.
        """
        start_index = max(0, int(start_index))
        end_index = (
            start_index + max(0, int(max_queries))
            if max_queries is not None
            else self._dataset.size()
        )
        self._run_preflight_if_needed(strict=self._require_dedicated_hardware)
        model = self._config.get("model", "unknown")

        # Collect the records we'll process
        work_items: list[tuple[int, DatasetRecord]] = []
        for index, record in enumerate(self._dataset):
            if index < start_index:
                continue
            if end_index is not None and index >= end_index:
                break
            work_items.append((index, record))

        if self._run_dir:
            self._write_subset_manifest(work_items, start_index, end_index)
            self._initialize_incremental_outputs()

        if self._telemetry_accounting == "batch":
            return await self._run_batched(work_items, model)
        if self._concurrency <= 1:
            return await self._run_sequential(work_items, model)
        return await self._run_concurrent(work_items, model)

    async def _run_sequential(
        self,
        work_items: list[tuple[int, DatasetRecord]],
        model: str,
    ) -> list[QueryTrace]:
        """Original sequential execution path."""
        with tqdm(total=len(work_items), desc="Agent run", unit="query") as progress:
            for index, record in work_items:
                query_id = f"q{index:04d}"
                start_time = time.time()
                try:
                    if (
                        self._query_timeout
                        and threading.current_thread() is threading.main_thread()
                        and hasattr(signal, "setitimer")
                    ):
                        old_handler = signal.getsignal(signal.SIGALRM)
                        signal.signal(signal.SIGALRM, _raise_query_timeout)
                        signal.setitimer(signal.ITIMER_REAL, float(self._query_timeout))
                        try:
                            trace = await self._run_single_query(
                                index, record, model, self._agent, self._event_recorder
                            )
                        finally:
                            signal.setitimer(signal.ITIMER_REAL, 0)
                            signal.signal(signal.SIGALRM, old_handler)
                    else:
                        fut = self._run_single_query(
                            index, record, model, self._agent, self._event_recorder
                        )
                        if self._query_timeout:
                            trace = await asyncio.wait_for(fut, timeout=self._query_timeout)
                        else:
                            trace = await fut
                except (asyncio.TimeoutError, _QueryTimeout):
                    elapsed = time.time() - start_time
                    LOGGER.warning(
                        "Query %s timed out after %.0fs (limit=%ss)",
                        query_id, elapsed, self._query_timeout,
                    )
                    trace = self._build_timeout_trace(
                        query_id=query_id,
                        record=record,
                        elapsed=elapsed,
                        start_time=start_time,
                        event_recorder=self._event_recorder,
                    )
                self._traces.append(trace)

                # Log per-task latency
                status = "TIMEOUT" if trace.timed_out else ("OK" if trace.completed else "FAIL")
                LOGGER.info(
                    "Task %s: %s in %.1fs",
                    query_id, status, trace.total_wall_clock_s,
                )

                profiling_record = self._build_profiling_record(
                    record, trace, model
                )
                self._records.append(profiling_record)

                if self._run_dir:
                    self._save_query_artifacts(index, record, trace)
                    self._flush_incremental_outputs(trace, self._traces)

                if len(self._traces) % self._FLUSH_INTERVAL == 0:
                    LOGGER.debug(
                        "Processed %d/%d queries",
                        len(self._traces),
                        len(work_items),
                    )

                progress.update(1)

        return self._traces

    async def _run_concurrent(
        self,
        work_items: list[tuple[int, DatasetRecord]],
        model: str,
    ) -> list[QueryTrace]:
        """Run tasks concurrently using a thread pool.

        Each task gets its own agent instance (from agent_factory) to avoid
        shared state conflicts.  Results are collected in index order.
        """
        total = len(work_items)
        LOGGER.info(
            "Running %d queries with concurrency=%d",
            total,
            self._concurrency,
        )

        # Pre-allocate result slots so we preserve index ordering
        result_slots: list[Optional[tuple[QueryTrace, ProfilingRecord]]] = [
            None
        ] * total
        progress = tqdm(total=total, desc="Agent run", unit="query")
        semaphore = asyncio.Semaphore(self._concurrency)
        loop = asyncio.get_event_loop()

        async def _process(slot: int, index: int, record: DatasetRecord) -> None:
            async with semaphore:
                recorder = EventRecorder()
                agent, recorder = self._create_concurrent_agent(recorder)

                query_id = f"q{index:04d}"
                start_time = time.time()

                try:
                    # Run the blocking work in a thread, with optional timeout
                    fut = loop.run_in_executor(
                        None,
                        self._run_single_query_sync,
                        index,
                        record,
                        model,
                        agent,
                        recorder,
                    )
                    if self._query_timeout:
                        trace = await asyncio.wait_for(fut, timeout=self._query_timeout)
                    else:
                        trace = await fut
                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    LOGGER.warning(
                        "Query %s timed out after %.0fs (limit=%ss)",
                        query_id, elapsed, self._query_timeout,
                    )
                    trace = self._build_timeout_trace(
                        query_id=query_id,
                        record=record,
                        elapsed=elapsed,
                        start_time=start_time,
                        event_recorder=recorder,
                    )

                # Log per-task latency
                status = "TIMEOUT" if trace.timed_out else ("OK" if trace.completed else "FAIL")
                LOGGER.info(
                    "Task %s: %s in %.1fs",
                    query_id, status, trace.total_wall_clock_s,
                )

                profiling_record = self._build_profiling_record(
                    record, trace, model
                )

                if self._run_dir:
                    self._save_query_artifacts(index, record, trace)

                with self._results_lock:
                    result_slots[slot] = (trace, profiling_record)
                    if self._run_dir:
                        partial_traces = [
                            item[0] for item in result_slots if item is not None
                        ]
                        self._flush_incremental_outputs(trace, partial_traces)
                    progress.update(1)

        tasks = [
            _process(slot, index, record)
            for slot, (index, record) in enumerate(work_items)
        ]
        await asyncio.gather(*tasks)
        progress.close()

        # Collect results in original order
        for slot_result in result_slots:
            if slot_result is not None:
                trace, profiling_record = slot_result
                self._traces.append(trace)
                self._records.append(profiling_record)

        return self._traces

    async def _run_batched(
        self,
        work_items: list[tuple[int, DatasetRecord]],
        model: str,
    ) -> list[QueryTrace]:
        """Run fixed-size concurrent batches with one telemetry window each."""
        total = len(work_items)
        LOGGER.info(
            "Running %d queries with batch telemetry, batch_size=%d",
            total,
            self._concurrency,
        )
        result_slots: list[Optional[tuple[QueryTrace, ProfilingRecord]]] = [
            None
        ] * total
        loop = asyncio.get_event_loop()

        with tqdm(total=total, desc="Agent run", unit="query") as progress:
            for batch_index, start_slot in enumerate(range(0, total, self._concurrency)):
                batch_items = work_items[start_slot:start_slot + self._concurrency]
                batch_id = f"batch_{batch_index:04d}"
                batch_start_time = time.time()

                async def _process(
                    slot: int,
                    index: int,
                    record: DatasetRecord,
                ) -> tuple[int, QueryTrace, ProfilingRecord]:
                    recorder = EventRecorder()
                    agent, recorder = self._create_concurrent_agent(recorder)
                    query_id = f"q{index:04d}"
                    start_time = time.time()
                    try:
                        fut = loop.run_in_executor(
                            None,
                            self._run_single_query_sync,
                            index,
                            record,
                            model,
                            agent,
                            recorder,
                        )
                        if self._query_timeout:
                            trace = await asyncio.wait_for(fut, timeout=self._query_timeout)
                        else:
                            trace = await fut
                    except asyncio.TimeoutError:
                        elapsed = time.time() - start_time
                        LOGGER.warning(
                            "Query %s timed out after %.0fs (limit=%ss)",
                            query_id,
                            elapsed,
                            self._query_timeout,
                        )
                        trace = self._build_timeout_trace(
                            query_id=query_id,
                            record=record,
                            elapsed=elapsed,
                            start_time=start_time,
                            event_recorder=recorder,
                        )

                    status = "TIMEOUT" if trace.timed_out else ("OK" if trace.completed else "FAIL")
                    LOGGER.info("Task %s: %s in %.1fs", query_id, status, trace.total_wall_clock_s)
                    return slot, trace, self._build_profiling_record(record, trace, model)

                batch_results = await asyncio.gather(
                    *[
                        _process(start_slot + offset, index, record)
                        for offset, (index, record) in enumerate(batch_items)
                    ]
                )
                batch_end_time = time.time()
                traces_by_slot = sorted(batch_results, key=lambda item: item[0])
                traces = [item[1] for item in traces_by_slot]
                query_ids = [trace.query_id for trace in traces]
                prompt_ends = [
                    trace.prompt_end_time_s
                    for trace in traces
                    if trace.prompt_end_time_s is not None
                ]
                prompt_end_time = max(prompt_ends) if prompt_ends else batch_end_time
                prompt_finish_offsets = {
                    trace.query_id: (
                        max((trace.prompt_end_time_s or prompt_end_time) - batch_start_time, 0.0)
                        if trace.prompt_end_time_s is not None
                        else None
                    )
                    for trace in traces
                }
                recorder = (
                    getattr(self._telemetry, "sample_recorder", None)
                    if self._telemetry is not None else None
                )
                if recorder is not None:
                    recorder.wait_until_after(prompt_end_time)
                    readings = recorder.read_window(batch_start_time, prompt_end_time)
                elif self._telemetry is not None:
                    readings = _collect_telemetry_window(
                        self._telemetry, batch_start_time, prompt_end_time
                    )
                else:
                    readings = []
                batch_record = build_batch_record(
                    batch_id=batch_id,
                    batch_index=batch_index,
                    query_ids=query_ids,
                    prompt_start_time=batch_start_time,
                    prompt_end_time=prompt_end_time,
                    total_start_time=batch_start_time,
                    total_end_time=batch_end_time,
                    prompt_finish_offsets_s=prompt_finish_offsets,
                    readings=readings,
                    telemetry_device_ids=self._telemetry_device_ids(),
                )
                self._batch_records.append(batch_record)

                for batch_position, (slot, trace, profiling_record) in enumerate(traces_by_slot):
                    self._annotate_trace_batch(
                        trace,
                        batch_record=batch_record,
                        batch_position=batch_position,
                    )
                    profiling_record = self._build_profiling_record(
                        batch_items[slot - start_slot][1],
                        trace,
                        model,
                    )
                    result_slots[slot] = (trace, profiling_record)
                    if self._run_dir:
                        index, record = batch_items[slot - start_slot]
                        self._save_query_artifacts(index, record, trace)
                    progress.update(1)

                if self._run_dir:
                    completed_traces = [
                        item[0] for item in result_slots if item is not None
                    ]
                    self._flush_batch_outputs(traces, completed_traces, batch_record)

        for slot_result in result_slots:
            if slot_result is not None:
                trace, profiling_record = slot_result
                self._traces.append(trace)
                self._records.append(profiling_record)

        return self._traces

    def _create_concurrent_agent(
        self,
        recorder: EventRecorder,
    ) -> tuple[BaseAgent, EventRecorder]:
        """Create a per-task agent and return the recorder it will write to."""
        if self._agent_factory is not None:
            agent = self._call_agent_factory(recorder)
        else:
            agent = copy.deepcopy(self._agent)
            if hasattr(agent, "event_recorder"):
                agent.event_recorder = recorder
            return agent, recorder

        agent_recorder = getattr(agent, "event_recorder", None)
        if isinstance(agent_recorder, EventRecorder):
            return agent, agent_recorder
        if hasattr(agent, "event_recorder"):
            agent.event_recorder = recorder
        return agent, recorder

    def _call_agent_factory(self, recorder: EventRecorder) -> BaseAgent:
        assert self._agent_factory is not None
        try:
            signature = inspect.signature(self._agent_factory)
        except (TypeError, ValueError):
            return self._agent_factory(recorder)

        accepts_positional = any(
            param.kind
            in (
                param.POSITIONAL_ONLY,
                param.POSITIONAL_OR_KEYWORD,
                param.VAR_POSITIONAL,
            )
            for param in signature.parameters.values()
        )
        accepts_event_recorder = (
            "event_recorder" in signature.parameters
            or any(
                param.kind == param.VAR_KEYWORD
                for param in signature.parameters.values()
            )
        )
        if accepts_event_recorder and not accepts_positional:
            return self._agent_factory(event_recorder=recorder)
        if accepts_positional:
            return self._agent_factory(recorder)
        return self._agent_factory()

    def _build_timeout_trace(
        self,
        *,
        query_id: str,
        record: DatasetRecord,
        elapsed: float,
        start_time: float,
        event_recorder: EventRecorder,
    ) -> QueryTrace:
        """Create a metric-complete trace for a timed-out query."""
        end_time = time.time()
        readings = (
            _collect_telemetry_window(self._telemetry, start_time, end_time)
            if self._telemetry
            else []
        )
        turns = self._build_turn_traces(event_recorder.get_events(), readings)
        if not turns:
            turns = [
                TurnTrace(
                    turn_index=0,
                    input_tokens=None,
                    output_tokens=None,
                    wall_clock_s=elapsed,
                )
            ]
        elif sum((t.input_tokens or 0) + (t.output_tokens or 0) for t in turns) == 0:
            turns[0].input_tokens = None
            turns[0].output_tokens = None
            turns[0].wall_clock_s = turns[0].wall_clock_s or elapsed

        query_gpu_energy = _compute_energy_delta(
            readings, "energy_joules", start_time, end_time
        )
        query_cpu_energy = _compute_energy_delta(
            readings, "cpu_energy_joules", start_time, end_time
        )
        query_dram_energy = _compute_energy_delta(
            readings, "dram_energy_joules", start_time, end_time
        )
        query_gpu_power_avg = _compute_power_avg(readings, "power_watts")
        query_cpu_power_avg = _compute_power_avg(readings, "cpu_power_watts")
        query_dram_power_avg = _compute_power_avg(readings, "dram_power_watts")
        if query_gpu_energy is None and readings:
            query_gpu_energy = _estimate_energy_from_power(readings, "power_watts", elapsed)
        if query_cpu_energy is None and readings:
            query_cpu_energy = _estimate_energy_from_power(readings, "cpu_power_watts", elapsed)
        if query_dram_energy is None and readings:
            query_dram_energy = _estimate_energy_from_power(readings, "dram_power_watts", elapsed)

        query_mbu_avg = None
        query_mbu_max = None
        if readings:
            mbu_values = [
                s.reading.gpu_memory_bandwidth_utilization_pct
                for s in readings
                if getattr(s.reading, "gpu_memory_bandwidth_utilization_pct", None) is not None
                and s.reading.gpu_memory_bandwidth_utilization_pct >= 0
            ]
            if mbu_values:
                query_mbu_avg = statistics.mean(mbu_values)
                query_mbu_max = max(mbu_values)

        workload_type = record.dataset_metadata.get("workload_type", "agentic")
        record.dataset_metadata["failure_type"] = "infra_timeout"
        record.dataset_metadata["unscorable_reason"] = "timeout"
        record.dataset_metadata["score_metadata"] = {
            "reason": "timeout",
            "failure_type": "infra_timeout",
            "timeout_s": self._query_timeout,
        }
        trace = QueryTrace(
            query_id=query_id,
            workload_type=str(workload_type),
            query_text=record.problem,
            response_text=f"Query timed out after {elapsed:.0f}s",
            turns=turns,
            total_wall_clock_s=elapsed,
            completed=False,
            timed_out=True,
            query_gpu_energy_joules=query_gpu_energy,
            query_cpu_energy_joules=query_cpu_energy,
            query_dram_energy_joules=query_dram_energy,
            query_gpu_power_avg_watts=query_gpu_power_avg,
            query_cpu_power_avg_watts=query_cpu_power_avg,
            query_dram_power_avg_watts=query_dram_power_avg,
            query_mbu_avg_pct=query_mbu_avg,
            query_mbu_max_pct=query_mbu_max,
            is_resolved=record.dataset_metadata.get("is_resolved"),
            unscorable_reason="timeout",
            score_metadata=dict(record.dataset_metadata["score_metadata"]),
        )
        self._annotate_trace_window(
            trace,
            prompt_start_time=start_time,
            prompt_end_time=end_time,
            prompt_readings=readings,
        )
        return self._correlate_energy(trace, readings)

    def _initialize_incremental_outputs(self) -> None:
        """Prepare per-query output files that are updated during the run."""
        assert self._run_dir is not None
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "traces.jsonl").write_text("", encoding="utf-8")
        (self._run_dir / "pre_judge_traces.jsonl").write_text("", encoding="utf-8")
        if self._telemetry_accounting == "batch":
            (self._run_dir / "batches.jsonl").write_text("", encoding="utf-8")

    def _annotate_trace_batch(
        self,
        trace: QueryTrace,
        *,
        batch_record: BatchTelemetryRecord,
        batch_position: int,
    ) -> None:
        """Attach batch timing metadata and remove per-query prompt energy."""
        trace.telemetry_accounting = "batch"
        trace.batch_id = batch_record.batch_id
        trace.batch_index = batch_record.batch_index
        trace.batch_position = batch_position
        trace.batch_size = batch_record.batch_size
        trace.batch_prompt_start_time_s = batch_record.batch_prompt_start_time_s
        trace.batch_prompt_end_time_s = batch_record.batch_prompt_end_time_s
        trace.batch_prompt_wall_clock_s = batch_record.batch_prompt_wall_clock_s
        trace.batch_total_wall_clock_s = batch_record.batch_total_wall_clock_s
        trace.batch_prompt_finish_offset_s = batch_record.prompt_finish_offsets_s.get(
            trace.query_id
        )
        trace.query_gpu_energy_joules = None
        trace.query_cpu_energy_joules = None
        trace.query_dram_energy_joules = None
        trace.query_gpu_power_avg_watts = None
        trace.query_cpu_power_avg_watts = None
        trace.query_dram_power_avg_watts = None
        trace.query_mbu_avg_pct = None
        trace.query_mbu_max_pct = None
        trace.telemetry_sample_count = 0
        trace.telemetry_device_ids = list(batch_record.telemetry_device_ids)
        for turn in trace.turns:
            turn.gpu_energy_joules = None
            turn.cpu_energy_joules = None
            turn.dram_energy_joules = None
            turn.gpu_power_avg_watts = None
            turn.cpu_power_avg_watts = None
            turn.dram_power_avg_watts = None

    def _write_subset_manifest(
        self,
        work_items: list[tuple[int, DatasetRecord]],
        start_index: int,
        end_index: int | None,
    ) -> None:
        """Write a stable manifest for the exact dataset records in this run."""
        assert self._run_dir is not None
        self._run_dir.mkdir(parents=True, exist_ok=True)
        records = [
            self._subset_record_entry(index, record)
            for index, record in work_items
        ]
        subset_hash = hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        manifest = {
            "schema_version": 1,
            "dataset": self._config.get("dataset"),
            "model": self._config.get("model"),
            "agent": self._config.get("agent"),
            "start_index": start_index,
            "end_index": end_index,
            "subset_size": len(records),
            "subset_hash": subset_hash,
            "records": records,
        }
        self._config["subset"] = {
            "subset_hash": subset_hash,
            "subset_size": len(records),
            "start_index": start_index,
            "end_index": end_index,
        }
        (self._run_dir / "subset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _subset_record_entry(index: int, record: DatasetRecord) -> dict[str, Any]:
        metadata = record.dataset_metadata or {}
        id_keys = (
            "task_id",
            "instance_id",
            "question_id",
            "uid",
            "original_index",
            "id",
        )
        stable_ids = {
            key: str(metadata[key])
            for key in id_keys
            if metadata.get(key) is not None
        }
        query_hash = hashlib.sha256(record.problem.encode("utf-8")).hexdigest()
        return {
            "index": index,
            "query_id": f"q{index:04d}",
            "query_hash": query_hash,
            "subject": record.subject,
            "stable_ids": stable_ids,
        }

    def _flush_incremental_outputs(
        self,
        trace: QueryTrace,
        traces_for_summary: list[QueryTrace],
    ) -> None:
        """Persist completed query trace data before the full run finishes."""
        assert self._run_dir is not None
        try:
            with (self._run_dir / "traces.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(trace.to_dict()) + "\n")

            from ..execution.exporters import export_summary_json

            export_summary_json(
                traces_for_summary,
                self._config,
                self._run_dir / "summary.json",
            )
        except Exception as exc:
            LOGGER.warning("Failed to flush incremental trace output: %s", exc)

    def _flush_batch_outputs(
        self,
        traces: list[QueryTrace],
        traces_for_summary: list[QueryTrace],
        batch_record: BatchTelemetryRecord,
    ) -> None:
        """Persist one completed telemetry batch before the full run finishes."""
        assert self._run_dir is not None
        try:
            with (self._run_dir / "traces.jsonl").open("a", encoding="utf-8") as f:
                for trace in traces:
                    f.write(json.dumps(trace.to_dict()) + "\n")
            with (self._run_dir / "batches.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(batch_record.to_dict()) + "\n")

            from ..execution.exporters import export_summary_json

            export_summary_json(
                traces_for_summary,
                self._config,
                self._run_dir / "summary.json",
                batch_records=self._batch_records,
            )
        except Exception as exc:
            LOGGER.warning("Failed to flush incremental batch output: %s", exc)

    def _build_pre_judge_trace(
        self,
        *,
        query_id: str,
        record: DatasetRecord,
        workload_type: str,
        result: AgentRunResult,
        start_time: float,
        prompt_end_time: float,
        event_recorder: EventRecorder,
        agent_result_failure_type: str | None,
    ) -> QueryTrace:
        """Build a prompt-only trace before any judge/scoring work starts."""
        readings = (
            _collect_telemetry_window(self._telemetry, start_time, prompt_end_time)
            if self._telemetry
            else []
        )
        events = event_recorder.get_events()
        if result.trace is not None and result.trace.turns:
            turns = result.trace.turns
            for turn_index, turn in enumerate(turns):
                turn.turn_index = turn_index
        else:
            turns = self._build_turn_traces(events, readings)

        result_has_token_counts = (
            result.input_tokens is not None
            and result.output_tokens is not None
        )
        if (
            not turns
            and result_has_token_counts
            and (result.input_tokens > 0 or result.output_tokens > 0)
        ):
            turns = [TurnTrace(
                turn_index=0,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                wall_clock_s=prompt_end_time - start_time,
                cost_usd=result.cost_usd if result.cost_usd is not None else None,
            )]
        elif (
            turns
            and result_has_token_counts
            and result.input_tokens > 0
            and result.output_tokens > 0
        ):
            total_turn_in = sum(t.input_tokens or 0 for t in turns)
            total_turn_out = sum(t.output_tokens or 0 for t in turns)
            if total_turn_in == 0 and total_turn_out == 0:
                turns[0].input_tokens = result.input_tokens
                turns[0].output_tokens = result.output_tokens
                turns[0].wall_clock_s = turns[0].wall_clock_s or (prompt_end_time - start_time)
                if result.cost_usd is not None and turns[0].cost_usd is None:
                    turns[0].cost_usd = result.cost_usd

        query_gpu_energy = _compute_energy_delta(
            readings,
            "energy_joules",
            start_time,
            prompt_end_time,
        )
        query_cpu_energy = _compute_energy_delta(
            readings,
            "cpu_energy_joules",
            start_time,
            prompt_end_time,
        )
        query_dram_energy = _compute_energy_delta(
            readings,
            "dram_energy_joules",
            start_time,
            prompt_end_time,
        )
        duration = prompt_end_time - start_time
        if query_gpu_energy is None and readings:
            query_gpu_energy = _estimate_energy_from_power(readings, "power_watts", duration)
        if query_cpu_energy is None and readings:
            query_cpu_energy = _estimate_energy_from_power(readings, "cpu_power_watts", duration)
        if query_dram_energy is None and readings:
            query_dram_energy = _estimate_energy_from_power(readings, "dram_power_watts", duration)

        query_mbu_avg = None
        query_mbu_max = None
        if readings:
            mbu_values = [
                s.reading.gpu_memory_bandwidth_utilization_pct
                for s in readings
                if getattr(s.reading, "gpu_memory_bandwidth_utilization_pct", None) is not None
                and s.reading.gpu_memory_bandwidth_utilization_pct >= 0
            ]
            if mbu_values:
                query_mbu_avg = statistics.mean(mbu_values)
                query_mbu_max = max(mbu_values)

        score_metadata: dict[str, Any] = {}
        if agent_result_failure_type is not None:
            score_metadata = dict(record.dataset_metadata.get("score_metadata") or {})
        agent_metadata = dict(record.dataset_metadata.get("agent_metadata") or {})
        if agent_metadata:
            score_metadata.setdefault("agent_metadata", agent_metadata)
            if agent_metadata.get("token_source") is not None:
                score_metadata.setdefault("token_source", agent_metadata["token_source"])

        trace = QueryTrace(
            query_id=query_id,
            workload_type=str(workload_type),
            query_text=record.problem,
            response_text=result.content,
            turns=turns,
            total_wall_clock_s=prompt_end_time - start_time,
            completed=agent_result_failure_type is None,
            timed_out=agent_result_failure_type == "infra_timeout",
            query_gpu_energy_joules=query_gpu_energy,
            query_cpu_energy_joules=query_cpu_energy,
            query_dram_energy_joules=query_dram_energy,
            query_gpu_power_avg_watts=_compute_power_avg(readings, "power_watts"),
            query_cpu_power_avg_watts=_compute_power_avg(readings, "cpu_power_watts"),
            query_dram_power_avg_watts=_compute_power_avg(readings, "dram_power_watts"),
            query_mbu_avg_pct=query_mbu_avg,
            query_mbu_max_pct=query_mbu_max,
            is_resolved=None,
            unscorable_reason=(
                str(record.dataset_metadata["unscorable_reason"])
                if record.dataset_metadata.get("unscorable_reason") is not None
                and agent_result_failure_type is not None
                else None
            ),
            score_metadata=score_metadata,
        )
        self._annotate_trace_window(
            trace,
            prompt_start_time=start_time,
            prompt_end_time=prompt_end_time,
            prompt_readings=readings,
        )
        trace.telemetry_accounting = self._telemetry_accounting
        return self._correlate_energy(trace, readings)

    def _save_pre_judge_snapshot(
        self,
        index: int,
        record: DatasetRecord,
        trace: QueryTrace,
    ) -> None:
        """Persist prompt-only response, metadata, and telemetry before judging."""
        assert self._run_dir is not None
        instance_id = record.dataset_metadata.get("instance_id", "")
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", str(instance_id))[:80]
        query_dir = self._run_dir / "artifacts" / f"q{index:04d}_{slug}"
        query_dir.mkdir(parents=True, exist_ok=True)

        metadata_without_judge = {
            key: value
            for key, value in record.dataset_metadata.items()
            if key not in {
                "is_resolved",
                "score_metadata",
                "evaluation_metadata",
                "test_results",
                "unscorable_reason",
            }
        }
        meta: dict[str, object] = {
            "schema_version": 1,
            "snapshot_stage": "pre_judge",
            "judge_status": "not_started",
            "query_id": trace.query_id,
            "instance_id": str(instance_id),
            "completed": trace.completed,
            "timed_out": trace.timed_out,
            "wall_clock_s": trace.total_wall_clock_s,
            "num_turns": trace.num_turns,
            "total_input_tokens": trace.total_input_tokens,
            "total_output_tokens": trace.total_output_tokens,
            "total_tokens": trace.total_tokens,
            "tool_call_count": trace.total_tool_calls,
            "query_gpu_energy_joules": trace.query_gpu_energy_joules,
            "query_cpu_energy_joules": trace.query_cpu_energy_joules,
            "query_dram_energy_joules": trace.query_dram_energy_joules,
            "query_gpu_power_avg_watts": trace.query_gpu_power_avg_watts,
            "query_cpu_power_avg_watts": trace.query_cpu_power_avg_watts,
            "query_dram_power_avg_watts": trace.query_dram_power_avg_watts,
            "prompt_start_time_s": trace.prompt_start_time_s,
            "prompt_end_time_s": trace.prompt_end_time_s,
            "telemetry_sample_count": trace.telemetry_sample_count,
            "telemetry_device_ids": trace.telemetry_device_ids,
            "telemetry_accounting": trace.telemetry_accounting,
            "query_mbu_avg_pct": trace.query_mbu_avg_pct,
            "query_mbu_max_pct": trace.query_mbu_max_pct,
            "response_path": "pre_judge_response.txt",
            "dataset_metadata": metadata_without_judge,
        }
        if trace.score_metadata:
            meta["agent_failure_metadata"] = trace.score_metadata

        (query_dir / "pre_judge_response.txt").write_text(
            trace.response_text or "", encoding="utf-8"
        )
        (query_dir / "pre_judge_metadata.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )

    def _flush_pre_judge_snapshot(
        self,
        index: int,
        record: DatasetRecord,
        trace: QueryTrace,
    ) -> None:
        """Best-effort durable prompt snapshot before judge/scoring starts."""
        if self._run_dir is None:
            return
        try:
            with self._results_lock:
                with (self._run_dir / "pre_judge_traces.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(trace.to_dict()) + "\n")
                self._save_pre_judge_snapshot(index, record, trace)
        except Exception as exc:
            LOGGER.warning("Failed to flush pre-judge snapshot: %s", exc)
    def _run_single_query_sync(
        self,
        index: int,
        record: DatasetRecord,
        model: str,
        agent: BaseAgent,
        event_recorder: EventRecorder,
    ) -> QueryTrace:
        """Synchronous wrapper for _run_single_query (used by thread pool)."""
        return asyncio.run(
            self._run_single_query(index, record, model, agent, event_recorder)
        )

    async def _run_with_executor(
        self,
        index: int,
        record: "DatasetRecord",
        model: str,
        agent: "ToolUsingAgent",
        event_recorder: "EventRecorder",
    ) -> "QueryTrace":
        """Native-agent dispatch path — delegates the turn loop to Executor.

        Seeds the task via agent.set_task() (subclasses must define it),
        runs Executor.execute(), and constructs a QueryTrace from the result.
        Per-turn telemetry population is left to EventBus subscribers
        (EnergyAttribution runs here; richer per-turn TurnTrace population can be
        layered on later).

        A TemporaryDirectory is created per query and set on every agent tool via
        set_workspace() before executor.execute() runs. This keeps agent-driven
        side effects isolated from the project root and from other concurrent
        queries. The tempdir is cleaned up automatically on context exit; tools
        are cleared (set_workspace(None)) in a finally block to prevent stale
        path leakage across query reuse.
        """
        import tempfile

        assert isinstance(agent, ToolUsingAgent)

        query_id = f"q{index:04d}"
        workload_type = record.dataset_metadata.get("workload_type", "agentic")

        event_recorder.clear()
        task_text = record.problem
        if hasattr(agent, "set_task"):
            agent.set_task(task_text)

        # Per-query temp workspace: every agent-driven tool invocation defaults
        # its cwd here, keeping side effects out of the project root and the
        # runner's worktree. Cleaned up automatically on context exit.
        with tempfile.TemporaryDirectory(prefix=f"ipw_q{index:04d}_") as workspace:
            for tool in getattr(agent, "tools", []):
                if hasattr(tool, "set_workspace"):
                    tool.set_workspace(workspace)

            try:
                executor = Executor(
                    bus=event_recorder.bus,
                    max_attempts_per_turn=self._max_attempts,
                )
                result = await executor.execute(
                    agent, task_id=query_id,
                    max_turns=self._max_turns,
                    agent_name=agent.__class__.__name__,
                )
            finally:
                # Clear workspace so tool reuse across queries doesn't leak
                # the now-deleted tempdir path.
                for tool in getattr(agent, "tools", []):
                    if hasattr(tool, "set_workspace"):
                        tool.set_workspace(None)

        return QueryTrace(
            query_id=query_id, workload_type=workload_type,
            query_text=task_text, response_text=result.final_answer or "",
            completed=result.status == "success",
            timed_out=False, n_retries=result.n_retries,
        )

    async def _run_single_query(
        self,
        index: int,
        record: DatasetRecord,
        model: str,
        agent: Optional[BaseAgent] = None,
        event_recorder: Optional[EventRecorder] = None,
    ) -> QueryTrace:
        """Run a single query through the agent with telemetry capture."""
        agent = agent or self._agent
        event_recorder = (
            event_recorder if event_recorder is not None else self._event_recorder
        )

        if isinstance(agent, ToolUsingAgent):
            return await self._run_with_executor(index, record, model, agent, event_recorder)

        query_id = f"q{index:04d}"
        workload_type = record.dataset_metadata.get("workload_type", "agentic")

        # Capture prompt and judge/evaluation telemetry in separate windows.
        # The telemetry source can remain cumulative; each field below is a
        # delta over its own wall-clock interval.
        start_time = time.time()
        prompt_end_time: float | None = None
        judge_start_time: float | None = None
        judge_end_time: float | None = None
        _telemetry_samples_before = (  # noqa: F841
            list(self._telemetry.readings()) if self._telemetry else []
        )

        event_recorder.clear()

        # Set up per-query workspace for agents that support it
        if self._run_dir and hasattr(agent, "set_workspace"):
            instance_id = record.dataset_metadata.get("instance_id", "")
            slug = re.sub(r"[^a-zA-Z0-9_-]", "_", str(instance_id))[:80]
            workspace = (
                self._run_dir / "artifacts" / f"q{index:04d}_{slug}" / "workspace"
            )
            workspace.mkdir(parents=True, exist_ok=True)
            dataset_name = str(record.dataset_metadata.get("dataset_name") or "")
            if "gdpval" in dataset_name.lower():
                record.dataset_metadata.setdefault("gdpval_outputs_dir", str(workspace))
            if hasattr(self._dataset, "prepare_workspace"):
                try:
                    self._dataset.prepare_workspace(record, workspace)
                except Exception as exc:
                    LOGGER.warning("Workspace setup failed for %s: %s", query_id, exc)
                    record.dataset_metadata["workspace_prepared"] = False
                    record.dataset_metadata["workspace_error"] = str(exc)
            agent.set_workspace(str(workspace))

        # Enter the dataset's task context.
        from contextlib import nullcontext

        # Make the selected model available to the task context.
        record.dataset_metadata.setdefault("model", model)

        task_env = self._dataset.create_task_env(record)
        ctx = task_env if task_env is not None else nullcontext()
        agent_result_failure_type: str | None = None
        agent_result_failure_reason: str | None = None

        try:
            with ctx:
                # Let the agent use any metadata prepared by the task context.
                agent.set_task_metadata(record.dataset_metadata)

                run_async = getattr(agent, "run_async", None)
                if inspect.iscoroutinefunction(run_async):
                    result = await run_async(record.problem)
                else:
                    result = agent.run(record.problem)
                agent_metadata = dict(result.metadata or {})
                if agent_metadata:
                    record.dataset_metadata["agent_metadata"] = agent_metadata
                    record.dataset_metadata.setdefault("agent_result_metadata", agent_metadata)
                    token_source = agent_metadata.get("token_source")
                    if token_source is not None:
                        record.dataset_metadata["token_source"] = token_source
                    for key in (
                        "gdpval_outputs_dir",
                        "gdpval_submitted_files",
                    ):
                        if key in agent_metadata:
                            record.dataset_metadata[key] = agent_metadata[key]

                agent_result_failure_type, agent_result_failure_reason = _classify_agent_result_failure(result)
                if agent_result_failure_type is not None:
                    record.dataset_metadata["failure_type"] = agent_result_failure_type
                    record.dataset_metadata["unscorable_reason"] = agent_result_failure_reason
                    record.dataset_metadata["score_metadata"] = {
                        "reason": agent_result_failure_reason,
                        "failure_type": agent_result_failure_type,
                        "error": result.content,
                    }
                else:
                    result = await self._maybe_repair_gdpval_deliverables(
                        agent=agent,
                        record=record,
                        result=result,
                    )
                    repair_metadata = dict(result.metadata or {})
                    if repair_metadata:
                        record.dataset_metadata["agent_metadata"] = repair_metadata
                        record.dataset_metadata["agent_result_metadata"] = repair_metadata
                        if repair_metadata.get("token_source") is not None:
                            record.dataset_metadata["token_source"] = repair_metadata["token_source"]
                        for key in (
                            "gdpval_outputs_dir",
                            "gdpval_submitted_files",
                        ):
                            if key in repair_metadata:
                                record.dataset_metadata[key] = repair_metadata[key]
                    agent_result_failure_type, agent_result_failure_reason = _classify_agent_result_failure(result)
                    if agent_result_failure_type is not None:
                        record.dataset_metadata["failure_type"] = agent_result_failure_type
                        record.dataset_metadata["unscorable_reason"] = agent_result_failure_reason
                        record.dataset_metadata["score_metadata"] = {
                            "reason": agent_result_failure_reason,
                            "failure_type": agent_result_failure_type,
                            "error": result.content,
                        }

                await self._settle_gdpval_deliverables(record)
                if (
                    agent_result_failure_type is not None
                    and self._gdpval_deliverables_valid(record)
                ):
                    LOGGER.info(
                        "GDPval task %s produced valid deliverables after an "
                        "agent failure marker; clearing failure for artifact judging.",
                        query_id,
                    )
                    agent_result_failure_type = None
                    agent_result_failure_reason = None
                    record.dataset_metadata.pop("failure_type", None)
                    record.dataset_metadata.pop("unscorable_reason", None)
                    failure_score = record.dataset_metadata.get("score_metadata")
                    if isinstance(failure_score, dict) and failure_score.get("failure_type"):
                        record.dataset_metadata.pop("score_metadata", None)
                result = _normalize_gdpval_result_content(record, result)

                prompt_end_time = time.time()

                if self._run_dir:
                    pre_judge_trace = self._build_pre_judge_trace(
                        query_id=query_id,
                        record=record,
                        workload_type=str(workload_type),
                        result=result,
                        start_time=start_time,
                        prompt_end_time=prompt_end_time,
                        event_recorder=event_recorder,
                        agent_result_failure_type=agent_result_failure_type,
                    )
                    self._flush_pre_judge_snapshot(index, record, pre_judge_trace)

                if agent_result_failure_type is not None:
                    pass
                elif task_env is not None:
                    judge_start_time = time.time()
                    try:
                        task_env.run_tests()
                        if (
                            "is_resolved" in record.dataset_metadata
                            and hasattr(self._dataset, "score")
                        ):
                            try:
                                is_correct, score_metadata = self._dataset.score(
                                    record, result.content
                                )
                                score_metadata = score_metadata or {}
                                record.dataset_metadata["is_resolved"] = is_correct
                                if score_metadata:
                                    record.dataset_metadata["score_metadata"] = score_metadata
                                if is_correct is None:
                                    reason = str(score_metadata.get("reason", "unscorable"))
                                    record.dataset_metadata["unscorable_reason"] = reason
                                await self._maybe_rescore_late_gdpval_deliverables(
                                    record=record,
                                    result=result,
                                )
                            except Exception as score_exc:
                                LOGGER.warning("Scoring failed for %s: %s", query_id, score_exc)
                                record.dataset_metadata["unscorable_reason"] = "scoring_error"
                                record.dataset_metadata["score_metadata"] = {
                                    "reason": "scoring_error",
                                    "error": str(score_exc),
                                }
                    finally:
                        judge_end_time = time.time()
                elif hasattr(self._dataset, "score") and record.answer:
                    judge_start_time = time.time()
                    try:
                        is_correct, score_metadata = self._dataset.score(record, result.content)
                        score_metadata = score_metadata or {}
                        record.dataset_metadata["is_resolved"] = is_correct
                        if score_metadata:
                            record.dataset_metadata["score_metadata"] = score_metadata
                            record.dataset_metadata["evaluation_metadata"] = score_metadata
                        if is_correct is None:
                            reason = str(score_metadata.get("reason", "unscorable"))
                            record.dataset_metadata["unscorable_reason"] = reason
                        await self._maybe_rescore_late_gdpval_deliverables(
                            record=record,
                            result=result,
                        )
                    except Exception as score_exc:
                        LOGGER.warning("Scoring failed for %s: %s", query_id, score_exc)
                        record.dataset_metadata["unscorable_reason"] = "scoring_error"
                        record.dataset_metadata["score_metadata"] = {
                            "reason": "scoring_error",
                            "error": str(score_exc),
                        }
                    finally:
                        judge_end_time = time.time()
        except Exception as exc:
            LOGGER.warning("Agent failed on query %s: %s", query_id, exc)
            failure_type, failure_reason = _classify_agent_exception(exc)
            record.dataset_metadata["failure_type"] = failure_type
            record.dataset_metadata["unscorable_reason"] = failure_reason
            record.dataset_metadata["score_metadata"] = {
                "reason": failure_reason,
                "failure_type": failure_type,
                "error": str(exc),
            }
            failure_time = time.time()
            prompt_end_time = prompt_end_time or failure_time
            prompt_readings = []
            judge_readings = []
            if self._telemetry:
                prompt_readings = _collect_telemetry_window(
                    self._telemetry,
                    start_time,
                    prompt_end_time,
                )
                if judge_start_time is not None:
                    judge_readings = _collect_telemetry_window(
                        self._telemetry,
                        judge_start_time,
                        judge_end_time or failure_time,
                    )
            events = event_recorder.get_events() if event_recorder is not None else []
            turns = self._build_turn_traces(events, prompt_readings)
            trace = QueryTrace(
                query_id=query_id,
                workload_type=str(workload_type),
                query_text=record.problem,
                response_text=str(exc),
                turns=turns,
                total_wall_clock_s=prompt_end_time - start_time,
                completed=False,
                query_gpu_energy_joules=_compute_energy_delta(
                    prompt_readings,
                    "energy_joules",
                    start_time,
                    prompt_end_time,
                ),
                query_cpu_energy_joules=_compute_energy_delta(
                    prompt_readings,
                    "cpu_energy_joules",
                    start_time,
                    prompt_end_time,
                ),
                query_dram_energy_joules=_compute_energy_delta(
                    prompt_readings,
                    "dram_energy_joules",
                    start_time,
                    prompt_end_time,
                ),
                query_gpu_power_avg_watts=_compute_power_avg(prompt_readings, "power_watts"),
                query_cpu_power_avg_watts=_compute_power_avg(prompt_readings, "cpu_power_watts"),
                query_dram_power_avg_watts=_compute_power_avg(prompt_readings, "dram_power_watts"),
                judge_gpu_energy_joules=_compute_energy_delta(
                    judge_readings,
                    "energy_joules",
                    judge_start_time,
                    judge_end_time or failure_time,
                ) if judge_start_time is not None else None,
                judge_cpu_energy_joules=_compute_energy_delta(
                    judge_readings,
                    "cpu_energy_joules",
                    judge_start_time,
                    judge_end_time or failure_time,
                ) if judge_start_time is not None else None,
                judge_dram_energy_joules=_compute_energy_delta(
                    judge_readings,
                    "dram_energy_joules",
                    judge_start_time,
                    judge_end_time or failure_time,
                ) if judge_start_time is not None else None,
                judge_gpu_power_avg_watts=_compute_power_avg(judge_readings, "power_watts"),
                judge_cpu_power_avg_watts=_compute_power_avg(judge_readings, "cpu_power_watts"),
                judge_dram_power_avg_watts=_compute_power_avg(judge_readings, "dram_power_watts"),
                judge_wall_clock_s=(
                    (judge_end_time or failure_time) - judge_start_time
                    if judge_start_time is not None
                    else None
                ),
                is_resolved=record.dataset_metadata.get("is_resolved"),
                unscorable_reason=str(
                    record.dataset_metadata.get("unscorable_reason", failure_reason)
                ),
                score_metadata=dict(record.dataset_metadata["score_metadata"]),
            )
            self._annotate_trace_window(
                trace,
                prompt_start_time=start_time,
                prompt_end_time=prompt_end_time,
                prompt_readings=prompt_readings,
                judge_start_time=judge_start_time,
                judge_end_time=(
                    judge_end_time or failure_time
                    if judge_start_time is not None
                    else None
                ),
            )
            return trace

        prompt_end_time = prompt_end_time or time.time()

        # Collect telemetry samples for the prompt-only query window. Judge
        # telemetry is kept separate below.
        readings: list[TelemetrySample] = []
        judge_readings: list[TelemetrySample] = []
        if self._telemetry:
            readings = _collect_telemetry_window(
                self._telemetry,
                start_time,
                prompt_end_time,
            )
            if judge_start_time is not None and judge_end_time is not None:
                judge_readings = _collect_telemetry_window(
                    self._telemetry,
                    judge_start_time,
                    judge_end_time,
                )

        # Build turn traces from event recorder
        events = event_recorder.get_events()
        if result.trace is not None and result.trace.turns:
            turns = result.trace.turns
            for turn_index, turn in enumerate(turns):
                turn.turn_index = turn_index
        else:
            turns = self._build_turn_traces(events, readings)

        # When EventRecorder captured nothing, create a synthetic turn from
        # AgentRunResult so token counts and wall clock are preserved.
        result_has_token_counts = (
            result.input_tokens is not None
            and result.output_tokens is not None
        )
        if (
            not turns
            and result_has_token_counts
            and (result.input_tokens > 0 or result.output_tokens > 0)
        ):
            turns = [TurnTrace(
                turn_index=0,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                wall_clock_s=prompt_end_time - start_time,
                cost_usd=result.cost_usd if result.cost_usd is not None else None,
            )]

        # Backfill tokens from AgentRunResult when turns have zero tokens
        # (e.g. OpenHands fires lm_inference events without token metadata)
        if (
            turns
            and result_has_token_counts
            and result.input_tokens > 0
            and result.output_tokens > 0
        ):
            total_turn_in = sum(t.input_tokens or 0 for t in turns)
            total_turn_out = sum(t.output_tokens or 0 for t in turns)
            if total_turn_in == 0 and total_turn_out == 0:
                turns[0].input_tokens = result.input_tokens
                turns[0].output_tokens = result.output_tokens
                turns[0].wall_clock_s = turns[0].wall_clock_s or (prompt_end_time - start_time)
                if result.cost_usd is not None and turns[0].cost_usd is None:
                    turns[0].cost_usd = result.cost_usd
            else:
                missing_token_turns = [
                    t
                    for t in turns
                    if t.input_tokens is None
                    and t.output_tokens is None
                    and not t.tools_called
                    and t.error is None
                ]
                if missing_token_turns:
                    known_in = sum(
                        t.input_tokens or 0
                        for t in turns
                        if t.input_tokens is not None
                    )
                    known_out = sum(
                        t.output_tokens or 0
                        for t in turns
                        if t.output_tokens is not None
                    )
                    remaining_in = result.input_tokens - known_in
                    remaining_out = result.output_tokens - known_out
                    if remaining_in >= 0 and remaining_out >= 0:
                        missing_token_turns[0].input_tokens = remaining_in
                        missing_token_turns[0].output_tokens = remaining_out

        # Always compute prompt-only query energy from the prompt window.
        query_gpu_energy = _compute_energy_delta(
            readings,
            "energy_joules",
            start_time,
            prompt_end_time,
        )
        query_cpu_energy = _compute_energy_delta(
            readings,
            "cpu_energy_joules",
            start_time,
            prompt_end_time,
        )
        query_dram_energy = _compute_energy_delta(
            readings,
            "dram_energy_joules",
            start_time,
            prompt_end_time,
        )
        query_gpu_power_avg = _compute_power_avg(readings, "power_watts")
        query_cpu_power_avg = _compute_power_avg(readings, "cpu_power_watts")
        query_dram_power_avg = _compute_power_avg(readings, "dram_power_watts")
        judge_gpu_energy = _compute_energy_delta(
            judge_readings,
            "energy_joules",
            judge_start_time,
            judge_end_time,
        ) if judge_start_time is not None and judge_end_time is not None else None
        judge_cpu_energy = _compute_energy_delta(
            judge_readings,
            "cpu_energy_joules",
            judge_start_time,
            judge_end_time,
        ) if judge_start_time is not None and judge_end_time is not None else None
        judge_dram_energy = _compute_energy_delta(
            judge_readings,
            "dram_energy_joules",
            judge_start_time,
            judge_end_time,
        ) if judge_start_time is not None and judge_end_time is not None else None
        judge_gpu_power_avg = _compute_power_avg(judge_readings, "power_watts")
        judge_cpu_power_avg = _compute_power_avg(judge_readings, "cpu_power_watts")
        judge_dram_power_avg = _compute_power_avg(judge_readings, "dram_power_watts")

        # Fallback: estimate energy from average power when cumulative counters
        # have fewer than 2 samples (no delta possible).
        duration = prompt_end_time - start_time
        if query_gpu_energy is None and readings:
            query_gpu_energy = _estimate_energy_from_power(readings, "power_watts", duration)
        if query_cpu_energy is None and readings:
            query_cpu_energy = _estimate_energy_from_power(readings, "cpu_power_watts", duration)
        if query_dram_energy is None and readings:
            query_dram_energy = _estimate_energy_from_power(readings, "dram_power_watts", duration)

        # Extract MBU from telemetry samples
        query_mbu_avg = None
        query_mbu_max = None
        if readings:
            mbu_values = [
                s.reading.gpu_memory_bandwidth_utilization_pct
                for s in readings
                if getattr(s.reading, 'gpu_memory_bandwidth_utilization_pct', None) is not None
                and s.reading.gpu_memory_bandwidth_utilization_pct >= 0
            ]
            if mbu_values:
                query_mbu_avg = statistics.mean(mbu_values)
                query_mbu_max = max(mbu_values)

        score_metadata = dict(record.dataset_metadata.get("score_metadata") or {})
        agent_metadata = dict(record.dataset_metadata.get("agent_metadata") or {})
        if agent_metadata:
            score_metadata.setdefault("agent_metadata", agent_metadata)
            if agent_metadata.get("token_source") is not None:
                score_metadata.setdefault("token_source", agent_metadata["token_source"])

        trace = QueryTrace(
            query_id=query_id,
            workload_type=str(workload_type),
            query_text=record.problem,
            response_text=result.content,
            turns=turns,
            total_wall_clock_s=prompt_end_time - start_time,
            completed=agent_result_failure_type is None,
            timed_out=agent_result_failure_type == "infra_timeout",
            query_gpu_energy_joules=query_gpu_energy,
            query_cpu_energy_joules=query_cpu_energy,
            query_dram_energy_joules=query_dram_energy,
            query_gpu_power_avg_watts=query_gpu_power_avg,
            query_cpu_power_avg_watts=query_cpu_power_avg,
            query_dram_power_avg_watts=query_dram_power_avg,
            judge_gpu_energy_joules=judge_gpu_energy,
            judge_cpu_energy_joules=judge_cpu_energy,
            judge_dram_energy_joules=judge_dram_energy,
            judge_gpu_power_avg_watts=judge_gpu_power_avg,
            judge_cpu_power_avg_watts=judge_cpu_power_avg,
            judge_dram_power_avg_watts=judge_dram_power_avg,
            judge_wall_clock_s=(
                judge_end_time - judge_start_time
                if judge_start_time is not None and judge_end_time is not None
                else None
            ),
            query_mbu_avg_pct=query_mbu_avg,
            query_mbu_max_pct=query_mbu_max,
            is_resolved=record.dataset_metadata.get("is_resolved"),
            unscorable_reason=(
                str(record.dataset_metadata["unscorable_reason"])
                if record.dataset_metadata.get("unscorable_reason") is not None
                else None
            ),
            score_metadata=score_metadata,
        )
        self._annotate_trace_window(
            trace,
            prompt_start_time=start_time,
            prompt_end_time=prompt_end_time,
            prompt_readings=readings,
            judge_start_time=judge_start_time,
            judge_end_time=judge_end_time,
        )

        # Correlate energy data with trace
        trace = self._correlate_energy(trace, readings)

        return trace

    def _merge_agent_run_results(
        self,
        *,
        original: AgentRunResult,
        current: AgentRunResult,
        section_title: str,
    ) -> AgentRunResult:
        original_content = _sanitize_gdpval_response_text(original.content or "")
        current_content = _sanitize_gdpval_response_text(current.content or "")
        combined_content = ""
        if current_content:
            combined_content = f"\n\n--- {section_title} ---\n\n{current_content}"
        return AgentRunResult(
            content=original_content + combined_content,
            tool_calls_attempted=(
                (original.tool_calls_attempted or 0)
                + (current.tool_calls_attempted or 0)
            ),
            tool_calls_succeeded=(
                (original.tool_calls_succeeded or 0)
                + (current.tool_calls_succeeded or 0)
            ),
            tool_names_used=[
                *(original.tool_names_used or []),
                *(current.tool_names_used or []),
            ],
            num_turns=(original.num_turns or 0) + (current.num_turns or 0),
            input_tokens=(
                None
                if original.input_tokens is None and current.input_tokens is None
                else (original.input_tokens or 0) + (current.input_tokens or 0)
            ),
            output_tokens=(
                None
                if original.output_tokens is None and current.output_tokens is None
                else (original.output_tokens or 0) + (current.output_tokens or 0)
            ),
            cost_usd=(original.cost_usd or 0.0) + (current.cost_usd or 0.0),
            trace=current.trace or original.trace,
            metadata=dict(current.metadata or original.metadata or {}),
        )

    def _update_gdpval_metadata_from_result(
        self,
        *,
        record: DatasetRecord,
        result: AgentRunResult,
    ) -> None:
        repair_metadata = dict(result.metadata or {})
        for key in ("gdpval_outputs_dir", "gdpval_submitted_files"):
            if key in repair_metadata:
                record.dataset_metadata[key] = repair_metadata[key]

    async def _settle_gdpval_deliverables(self, record: DatasetRecord) -> None:
        """Refresh GDPval deliverable validation after late file writes settle."""
        if not _is_gdpval_record(record):
            return
        if self._gdpval_file_settle_seconds <= 0:
            return
        outputs_dir = record.dataset_metadata.get("gdpval_outputs_dir")
        if not outputs_dir:
            return
        try:
            from ..evaluation.deliverables import (
                deliverable_paths_from_metadata,
                validate_deliverable_contract,
            )
        except Exception as exc:  # pragma: no cover - defensive import guard.
            LOGGER.warning("GDPval deliverable settle unavailable: %s", exc)
            return

        existing_validation = record.dataset_metadata.get("deliverable_validation")
        if (
            isinstance(existing_validation, dict)
            and existing_validation.get("status") != "invalid"
        ):
            return

        def validate_now() -> dict[str, Any]:
            paths, _ = deliverable_paths_from_metadata(dict(record.dataset_metadata))
            validation = validate_deliverable_contract(
                paths=paths,
                problem=record.problem,
                metadata=dict(record.dataset_metadata),
            )
            record.dataset_metadata["deliverable_validation"] = validation
            return validation

        deadline = time.time() + self._gdpval_file_settle_seconds
        while True:
            validation = validate_now()
            if validation.get("status") != "invalid":
                return
            if time.time() >= deadline:
                return
            await asyncio.sleep(self._gdpval_file_settle_poll_seconds)

    async def _maybe_rescore_late_gdpval_deliverables(
        self,
        *,
        record: DatasetRecord,
        result: AgentRunResult,
    ) -> None:
        """Retry GDPval scoring if a late file write fixes a validation zero."""
        if not _is_gdpval_record(record):
            return
        if not hasattr(self._dataset, "score"):
            return
        score_metadata = record.dataset_metadata.get("score_metadata")
        if not isinstance(score_metadata, dict):
            return
        if (
            score_metadata.get("failure_type") != "harness_validation"
            or score_metadata.get("reason") != "deliverable_contract_invalid"
        ):
            return

        before_validation = record.dataset_metadata.get("deliverable_validation")
        await self._settle_gdpval_deliverables(record)
        if not self._gdpval_deliverables_valid(record):
            return

        LOGGER.info(
            "GDPval deliverables became valid after initial validation failure; "
            "rescoring task."
        )
        is_correct, rescored_metadata = self._dataset.score(record, result.content)
        rescored_metadata = rescored_metadata or {}
        record.dataset_metadata["is_resolved"] = is_correct
        record.dataset_metadata.pop("failure_type", None)
        if is_correct is None:
            reason = str(rescored_metadata.get("reason", "unscorable"))
            record.dataset_metadata["unscorable_reason"] = reason
        else:
            record.dataset_metadata.pop("unscorable_reason", None)
        if rescored_metadata:
            record.dataset_metadata["score_metadata"] = rescored_metadata
            record.dataset_metadata["evaluation_metadata"] = rescored_metadata
        elif before_validation is not record.dataset_metadata.get("deliverable_validation"):
            record.dataset_metadata.pop("score_metadata", None)

    @staticmethod
    def _gdpval_deliverables_valid(record: DatasetRecord) -> bool:
        validation = record.dataset_metadata.get("deliverable_validation")
        return isinstance(validation, dict) and validation.get("status") != "invalid"

    @staticmethod
    def _gdpval_repair_script_candidates(outputs_dir: Path) -> list[Path]:
        if not outputs_dir.exists() or not outputs_dir.is_dir():
            return []

        preferred = {
            "ipw_repair_deliverables.py": 0,
            "build_deliverable.py": 1,
            "generate_deliverables.py": 2,
            "create_deliverable.py": 3,
            "build.py": 4,
        }
        prefixes = ("ipw_repair", "build", "create", "generate", "make", "write")
        candidates: list[Path] = []
        for path in outputs_dir.glob("*.py"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if name in preferred or name.startswith(prefixes):
                candidates.append(path)

        def sort_key(path: Path) -> tuple[int, float, str]:
            rank = preferred.get(path.name.lower(), 50)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            return (rank, -mtime, path.name.lower())

        return sorted(candidates, key=sort_key)

    @staticmethod
    def _gdpval_repair_script_log_summary(record: DatasetRecord) -> str:
        logs = record.dataset_metadata.get("gdpval_autorun_repair_script_logs")
        if not isinstance(logs, list) or not logs:
            return ""
        lines = [
            "Recent model-authored repair scripts were autorun by the harness "
            "and did not yet produce a valid deliverable. Fix the script error "
            "or create the deliverable directly; do not repeat the same failing "
            "script unchanged."
        ]
        for item in logs[-3:]:
            if not isinstance(item, dict):
                continue
            script = str(item.get("script") or "unknown script")
            returncode = item.get("returncode")
            wall_clock = item.get("wall_clock_s")
            tail = str(item.get("output_tail") or "").strip()
            if len(tail) > 1600:
                tail = tail[-1600:]
            lines.append(
                f"- {script} exited with {returncode} after {wall_clock}s."
            )
            if tail:
                lines.append(f"  Output tail:\n{tail}")
        return "\n".join(lines)

    @staticmethod
    def _gdpval_repair_type_playbook(validation: dict[str, Any]) -> str:
        extensions = set(validation.get("expected_extensions") or [])
        parts = [
            "Generic repair playbook by requested output type:",
        ]
        if extensions & {".xlsx", ".xlsm", ".xltx", ".xltm"}:
            parts.append(
                "- Spreadsheet: use openpyxl/xlsxwriter to create a real workbook, "
                "save it under the requested .xlsx/.xlsm-compatible name, reopen "
                "it with openpyxl, and verify the required sheets/cells/formulas."
            )
        if ".docx" in extensions:
            parts.append(
                "- Word document: use python-docx or a valid DOCX template. Reopen "
                "with python-docx after saving. Do not edit raw document.xml unless "
                "you also validate the zip/XML package."
            )
        if ".pptx" in extensions:
            parts.append(
                "- Slide deck: use python-pptx to create a real .pptx, add the "
                "requested slides/content, save, then reopen with Presentation()."
            )
        if ".pdf" in extensions:
            parts.append(
                "- PDF: use reportlab, fpdf2, PyMuPDF, or a verified conversion "
                "pipeline. Check that the final .pdf opens with pdfinfo or PyMuPDF."
            )
        if extensions & {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
            parts.append(
                "- Video: create the final media file with ffmpeg/moviepy, then "
                "verify duration and codec with ffprobe. A script transcript or "
                "image sequence is not a video deliverable."
            )
        if extensions & {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}:
            parts.append(
                "- Audio: write a real audio file and verify duration/codec with "
                "ffprobe or wave/pydub. Probe scripts and analysis notes do not count."
            )
        if ".zip" in extensions:
            parts.append(
                "- Archive: use zipfile to package the requested finished files, "
                "then reopen the archive and list its contents."
            )
        if len(parts) == 1:
            parts.append(
                "- Create at least one real final artifact with a requested "
                "extension, reopen or inspect it, and finish with that path."
            )
        return "\n".join(parts)

    async def _maybe_autorun_gdpval_repair_scripts(
        self,
        *,
        record: DatasetRecord,
        deliverable_paths_from_metadata: Callable[..., tuple[list[Path], list[Path]]],
        validate_deliverable_contract: Callable[..., dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Run model-authored repair scripts once when GLM stops before executing them."""
        if not self._gdpval_autorun_repair_scripts:
            return None
        outputs_dir_raw = record.dataset_metadata.get("gdpval_outputs_dir")
        if not outputs_dir_raw:
            return None
        outputs_dir = Path(str(outputs_dir_raw)).resolve()
        scripts = self._gdpval_repair_script_candidates(outputs_dir)
        if not scripts:
            return None

        already_run = set(record.dataset_metadata.get("gdpval_autorun_repair_scripts") or [])
        run_logs = list(record.dataset_metadata.get("gdpval_autorun_repair_script_logs") or [])
        for script in scripts[:3]:
            try:
                relative_script = str(script.relative_to(outputs_dir))
            except ValueError:
                continue
            if relative_script in already_run:
                continue
            LOGGER.info("GDPval autorunning model-authored repair script: %s", script)
            started = time.time()
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(script),
                    cwd=str(outputs_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._gdpval_repair_script_timeout_seconds,
                )
                returncode = proc.returncode
                output = stdout.decode("utf-8", errors="replace") if stdout else ""
            except asyncio.TimeoutError:
                with contextlib.suppress(Exception):
                    proc.kill()  # type: ignore[name-defined]
                    await proc.wait()  # type: ignore[name-defined]
                returncode = None
                output = "Timed out while autorunning repair script."
            except Exception as exc:  # pragma: no cover - defensive subprocess guard.
                returncode = None
                output = f"{type(exc).__name__}: {exc}"

            already_run.add(relative_script)
            run_logs.append(
                {
                    "script": relative_script,
                    "returncode": returncode,
                    "wall_clock_s": round(time.time() - started, 3),
                    "output_tail": output[-4000:],
                }
            )
            record.dataset_metadata["gdpval_autorun_repair_scripts"] = sorted(already_run)
            record.dataset_metadata["gdpval_autorun_repair_script_logs"] = run_logs[-10:]

            paths, _ = deliverable_paths_from_metadata(dict(record.dataset_metadata))
            validation = validate_deliverable_contract(
                paths=paths,
                problem=record.problem,
                metadata=dict(record.dataset_metadata),
            )
            record.dataset_metadata["deliverable_validation"] = validation
            if validation.get("status") != "invalid":
                return validation
        return record.dataset_metadata.get("deliverable_validation")

    async def _run_agent_repair_prompt_async(
        self,
        agent: BaseAgent,
        prompt: str,
        *,
        max_turns: int | None = None,
    ) -> AgentRunResult:
        old_max_turns = getattr(agent, "_max_turns", None)
        should_restore = max_turns is not None and old_max_turns is not None
        if should_restore:
            setattr(agent, "_max_turns", max(1, int(max_turns)))
        run_async = getattr(agent, "run_async", None)
        try:
            if inspect.iscoroutinefunction(run_async):
                return await run_async(prompt)
            return agent.run(prompt)
        finally:
            if should_restore:
                setattr(agent, "_max_turns", old_max_turns)

    async def _maybe_repair_gdpval_deliverables(
        self,
        *,
        agent: BaseAgent,
        record: DatasetRecord,
        result: AgentRunResult,
    ) -> AgentRunResult:
        """Give GDPval agents a chance to fix missing/invalid deliverables."""
        if self._gdpval_repair_attempts <= 0 and self._gdpval_final_check_attempts <= 0:
            return result
        dataset_name = str(record.dataset_metadata.get("dataset_name") or "")
        if "gdpval" not in dataset_name.lower():
            return result
        if _uses_stirrup_single_submission_contract(record, result):
            return result

        try:
            from ..evaluation.deliverables import (
                deliverable_paths_from_metadata,
                validate_deliverable_contract,
            )
        except Exception as exc:  # pragma: no cover - defensive import guard.
            LOGGER.warning("GDPval deliverable repair unavailable: %s", exc)
            return result

        current = result
        for attempt in range(1, self._gdpval_repair_attempts + 1):
            paths, _ = deliverable_paths_from_metadata(dict(record.dataset_metadata))
            validation = validate_deliverable_contract(
                paths=paths,
                problem=record.problem,
                metadata=dict(record.dataset_metadata),
            )
            record.dataset_metadata["deliverable_validation"] = validation
            if validation["status"] != "invalid":
                break
            script_log_summary = self._gdpval_repair_script_log_summary(record)
            type_playbook = self._gdpval_repair_type_playbook(validation)

            repair_prompt = (
                "The pre-judge deliverable validator found that this GDPval task "
                "does not yet have a valid submitted deliverable. Do not continue "
                "broad research. In the current task working directory, create or "
                "repair the requested final deliverable file(s), verify them with "
                "`ls -la`, and call the finish tool with the relative file "
                "path(s). Saved scripts, scraped research, screenshots, or notes "
                "do not count as primary deliverables unless the task explicitly "
                "asked for them. Files created outside the current task working "
                "directory are not graded. If you write a script, write it in the "
                "current task working directory, run it, and verify the resulting "
                "deliverable file bytes are valid before finishing.\n\n"
                "Strict repair procedure:\n"
                "1. If expected final filename(s) are listed below, your next "
                "terminal action must create or overwrite each requested "
                "deliverable as a valid file. Prefer the listed names, but "
                "case/spacing variants or clearly equivalent descriptive names "
                "with the correct extension are acceptable when the content is "
                "right. Use one compact script or command that creates all "
                "missing PDF/DOCX/XLSX/PPTX/media outputs before any more "
                "inspection.\n"
                "2. Prefer a script named `ipw_repair_deliverables.py` in the "
                "current directory, then run it immediately and verify the "
                "deliverable file(s) with `ls -lh`. Do not print or "
                "read back a long script after writing it unless execution fails "
                "and you need the failing line.\n"
                "3. Use the installed file libraries directly: `openpyxl` or "
                "`xlsxwriter` for `.xlsx`, `python-docx` for `.docx`, and "
                "`reportlab`, `fpdf2`, or `PyMuPDF` for `.pdf`.\n"
                "4. Do not run broad searches, long greps, or open-ended commands "
                "during repair. Inspect only the needed input files and workbook "
                "metadata.\n"
                "5. If any command soft-times out, interrupt it once at most. Do "
                "not loop on interrupts, empty commands, or process checks; create "
                "or submit the best valid deliverable file immediately.\n"
                "6. A partial but valid deliverable is better than no submitted "
                "file. Finish as soon as the expected file exists and opens.\n\n"
                f"{type_playbook}\n\n"
                f"Validation issues:\n{json.dumps(validation.get('issues', []), indent=2)}\n\n"
                f"Expected extensions inferred by the harness: "
                f"{validation.get('expected_extensions', [])}\n"
                f"Target filename hint(s), when specified by the task: "
                f"{validation.get('expected_filenames', [])}\n"
                "Use the listed names when convenient, but do not stall on exact "
                "case or spacing. Avoid generic placeholders such as "
                "deliverable.pdf or deliverable.zip unless the task asked for "
                "that name.\n"
                "Do not submit copied reference/input files as final deliverables. "
                "If the task asks for a workbook, document, PDF, slide deck, media "
                "file, or archive, create the requested finished artifact with the "
                "requested content, not just a renamed source file.\n"
                f"Files currently visible to the harness: {validation.get('files', [])}\n"
                + (
                    f"\n{script_log_summary}\n"
                    if script_log_summary
                    else ""
                )
                +
                f"This is repair attempt {attempt} of {self._gdpval_repair_attempts}.\n\n"
                "Original GDPval task prompt, including reference file locations, "
                "follows. These are the instructions you must satisfy during repair; "
                "do not claim the task instructions are unavailable unless this "
                "section is empty.\n\n"
                f"{record.problem}"
            )
            LOGGER.info("GDPval deliverable repair attempt %d/%d", attempt, self._gdpval_repair_attempts)
            repair_result = await self._run_agent_repair_prompt_async(agent, repair_prompt)
            current = self._merge_agent_run_results(
                original=result,
                current=repair_result,
                section_title=f"GDPval deliverable repair attempt {attempt}",
            )
            result = current
            self._update_gdpval_metadata_from_result(record=record, result=current)
            if self._gdpval_autorun_repair_scripts:
                autorun_validation = await self._maybe_autorun_gdpval_repair_scripts(
                    record=record,
                    deliverable_paths_from_metadata=deliverable_paths_from_metadata,
                    validate_deliverable_contract=validate_deliverable_contract,
                )
                if (
                    isinstance(autorun_validation, dict)
                    and autorun_validation.get("status") != "invalid"
                ):
                    break

        paths, _ = deliverable_paths_from_metadata(dict(record.dataset_metadata))
        final_validation = validate_deliverable_contract(
            paths=paths,
            problem=record.problem,
            metadata=dict(record.dataset_metadata),
        )
        record.dataset_metadata["deliverable_validation"] = final_validation
        if final_validation["status"] == "invalid":
            return current

        final_check_max_turns = max(
            1,
            int(os.getenv("IPW_GDPVAL_FINAL_CHECK_MAX_TURNS", "24")),
        )
        total_turn_limit = max(
            1,
            int(getattr(agent, "_max_turns", None) or self._max_turns),
        )
        for attempt in range(1, self._gdpval_final_check_attempts + 1):
            turns_used = int(current.num_turns or 0)
            remaining_turns = total_turn_limit - turns_used
            if remaining_turns <= 0:
                LOGGER.info(
                    "Skipping GDPval final checklist attempt %d/%d; "
                    "task turn budget exhausted (%d/%d turns used).",
                    attempt,
                    self._gdpval_final_check_attempts,
                    turns_used,
                    total_turn_limit,
                )
                break
            attempt_max_turns = min(final_check_max_turns, remaining_turns)
            checklist_prompt = (
                "Before GDPval judging, perform a final deliverable checklist "
                "against the original task and any rubric included in that task. "
                "Inspect the actual files in the current task working directory. "
                f"This is a bounded final check with at most {attempt_max_turns} "
                f"turns remaining in the overall {total_turn_limit}-turn task "
                "budget; it is not a second full attempt. Do not redo broad "
                "exploration. Run only compact checks needed to confirm or repair "
                "the existing final artifacts.\n\n"
                "Use the task prompt and reference files as the source of truth; "
                "do not invent facts or add irrelevant content just to satisfy a "
                "rubric phrase. If anything is missing or wrong, repair it now "
                "and then call finish with the relative file path(s). If all "
                "deliverables are already correct, call finish with the relative "
                "file path(s) and a concise confirmation.\n\n"
                "Checklist:\n"
                "1. Every requested deliverable below exists in this working "
                "directory with valid bytes for its file type. Prefer listed "
                "target names, but case/spacing variants or similarly "
                "descriptive filenames with the correct extension are acceptable "
                "when the content is right.\n"
                "2. Reference/input files are not submitted as final deliverables.\n"
                "3. Each final file satisfies the task-specific content, not only "
                "the extension or filename. For spreadsheets, verify required "
                "sheets, formulas/calculations, added columns, selected rows, and "
                "saved workbook names. For document/PDF packages, verify every "
                "requested PDF/DOCX/XLSX deliverable exists under the expected "
                "or clearly equivalent name and contains the requested updated "
                "content.\n"
                "4. Convert visible acceptance criteria into artifact checks "
                "where possible, using structured tools such as openpyxl, "
                "python-docx, PyMuPDF, pandas, or file metadata commands. Repair "
                "the artifact if a check fails.\n"
                "5. Do not finish with unresolved caveats, pending rebuild notes, "
                "or claims that scripts/notebooks are the deliverable unless the "
                "task explicitly asked for them.\n"
                "6. Once the expected final artifact exists, opens successfully, "
                "and one structured verification pass confirms the main "
                "task/rubric criteria, finish with the file path. Do not re-read "
                "the full source, rewrite the artifact, or continue open-ended "
                "optimization unless a concrete verification check failed; if "
                "one failed, repair only that failure and re-check it.\n"
                "7. If visible task/rubric criteria name compatible filename "
                "variants for the same artifact type, create both variants with "
                "finished content unless the task or rubric explicitly requires "
                "a single final file or otherwise forbids extra files, and "
                "include both relative paths in the finish message.\n"
                "8. If visible criteria conflict only on packaging, filename, "
                "tab order, or presentation rather than facts, add a clearly "
                "labeled compatibility artifact, worksheet, or view so both "
                "interpretations are available without falsifying source data. "
                "For spreadsheet sampling tasks, keep selected sample rows and "
                "full population/reference rows clearly separated, usually with "
                "the selected sample in the primary sample sheet and the full "
                "population copy in a separate reference sheet.\n"
                "9. Run `ls -la` before finish.\n\n"
                f"Target filename hint(s): {final_validation.get('expected_filenames', [])}\n"
                f"Expected extension(s): {final_validation.get('expected_extensions', [])}\n"
                f"Current validation status: {final_validation.get('status')}\n"
                f"Current validation issues: {json.dumps(final_validation.get('issues', []), indent=2)}\n"
                f"Files visible to the harness: {final_validation.get('files', [])}\n"
                f"This is final checklist attempt {attempt} of {self._gdpval_final_check_attempts}.\n\n"
                "Original GDPval task prompt, including reference file locations "
                "and any rubric/acceptance criteria, follows. Use it for the "
                "checklist; do not claim the task instructions are unavailable "
                "unless this section is empty.\n\n"
                f"{record.problem}"
            )
            LOGGER.info(
                "GDPval final deliverable checklist attempt %d/%d",
                attempt,
                self._gdpval_final_check_attempts,
            )
            checklist_result = await self._run_agent_repair_prompt_async(
                agent,
                checklist_prompt,
                max_turns=attempt_max_turns,
            )
            current = self._merge_agent_run_results(
                original=result,
                current=checklist_result,
                section_title=f"GDPval final deliverable checklist attempt {attempt}",
            )
            result = current
            self._update_gdpval_metadata_from_result(record=record, result=current)
            paths, _ = deliverable_paths_from_metadata(dict(record.dataset_metadata))
            final_validation = validate_deliverable_contract(
                paths=paths,
                problem=record.problem,
                metadata=dict(record.dataset_metadata),
            )
            record.dataset_metadata["deliverable_validation"] = final_validation
            if final_validation["status"] == "invalid":
                LOGGER.info("GDPval final checklist produced invalid deliverables; retrying if budget remains.")
            else:
                LOGGER.info(
                    "GDPval final checklist accepted deliverables with status %s.",
                    final_validation["status"],
                )
                break
        return current

    def _build_turn_traces(
        self,
        events: list,
        readings: list[TelemetrySample],
    ) -> list[TurnTrace]:
        """Build TurnTrace objects from recorded events.

        A turn is modeled as one LLM call plus the tool calls produced by that
        model response. Tool calls usually happen after the ``lm_inference_end``
        event, so they are attached to the most recent completed LLM turn and
        extend that turn's telemetry window.
        """
        turn_records: list[dict[str, Any]] = []
        current_turn: Optional[dict[str, Any]] = None
        tool_start_times: dict[str, list[tuple[float, Optional[dict[str, Any]]]]] = {}

        def _new_turn(start_ts: float) -> dict[str, Any]:
            return {
                "start": start_ts,
                "end": start_ts,
                "input_tokens": None,
                "output_tokens": None,
                "cost_usd": None,
                "tools_called": [],
                "tool_latencies_s": {},
            }

        def _latency_key(latencies: dict[str, float], tool_name: str) -> str:
            if tool_name not in latencies:
                return tool_name
            suffix = 2
            while f"{tool_name}#{suffix}" in latencies:
                suffix += 1
            return f"{tool_name}#{suffix}"

        def _attach_tool(
            record: Optional[dict[str, Any]],
            tool_name: str,
            latency: Optional[float],
            end_ts: float,
        ) -> None:
            nonlocal current_turn
            if record is None:
                record = current_turn or (turn_records[-1] if turn_records else None)
            if record is None:
                record = _new_turn(end_ts)
                turn_records.append(record)
            record["tools_called"].append(tool_name)
            if latency is not None:
                record["tool_latencies_s"][
                    _latency_key(record["tool_latencies_s"], tool_name)
                ] = latency
            record["end"] = max(record["end"], end_ts)

        for event in events:
            etype = event.event_type

            if etype == EventType.LM_INFERENCE_START:
                current_turn = _new_turn(event.timestamp)

            elif etype == EventType.LM_INFERENCE_END:
                if current_turn is None:
                    current_turn = _new_turn(event.timestamp)
                current_turn["end"] = max(current_turn["end"], event.timestamp)
                current_turn["input_tokens"] = event.metadata.get("prompt_tokens")
                current_turn["output_tokens"] = event.metadata.get("completion_tokens")
                if event.metadata.get("cost_usd") is not None:
                    current_turn["cost_usd"] = event.metadata.get("cost_usd")
                turn_records.append(current_turn)
                current_turn = None

            elif etype == EventType.TOOL_CALL_START:
                tool_name = event.metadata.get("tool", "unknown")
                owner = current_turn or (turn_records[-1] if turn_records else None)
                tool_start_times.setdefault(tool_name, []).append(
                    (event.timestamp, owner)
                )

            elif etype == EventType.TOOL_CALL_END:
                tool_name = event.metadata.get("tool", "unknown")
                starts = tool_start_times.get(tool_name, [])
                start_ts = None
                owner = None
                if starts:
                    start_ts, owner = starts.pop()
                    if not starts:
                        tool_start_times.pop(tool_name, None)
                latency = (
                    event.timestamp - start_ts
                    if start_ts is not None
                    else None
                )
                _attach_tool(owner, tool_name, latency, event.timestamp)

        if current_turn is not None:
            turn_records.append(current_turn)

        turns: list[TurnTrace] = []
        for turn_index, record in enumerate(turn_records):
            start = record["start"]
            end = record["end"]
            wall_clock = max(0.0, end - start)
            turn_readings = [
                s for s in readings
                if start <= s.timestamp <= end
            ]
            turn_gpu_energy = _compute_energy_delta(turn_readings, "energy_joules")
            turn_cpu_energy = _compute_energy_delta(turn_readings, "cpu_energy_joules")
            turn_dram_energy = _compute_energy_delta(turn_readings, "dram_energy_joules")
            turn_gpu_power_avg = _compute_power_avg(turn_readings, "power_watts")
            turn_cpu_power_avg = _compute_power_avg(turn_readings, "cpu_power_watts")
            turn_dram_power_avg = _compute_power_avg(turn_readings, "dram_power_watts")

            if turn_gpu_energy is None and turn_readings:
                turn_gpu_energy = _estimate_energy_from_power(
                    turn_readings, "power_watts", wall_clock
                )
            if turn_cpu_energy is None and turn_readings:
                turn_cpu_energy = _estimate_energy_from_power(
                    turn_readings, "cpu_power_watts", wall_clock
                )
            if turn_dram_energy is None and turn_readings:
                turn_dram_energy = _estimate_energy_from_power(
                    turn_readings, "dram_power_watts", wall_clock
                )

            turns.append(
                TurnTrace(
                    turn_index=turn_index,
                    input_tokens=record["input_tokens"],
                    output_tokens=record["output_tokens"],
                    tools_called=list(record["tools_called"]),
                    tool_latencies_s=dict(record["tool_latencies_s"]),
                    wall_clock_s=wall_clock,
                    gpu_energy_joules=turn_gpu_energy,
                    cpu_energy_joules=turn_cpu_energy,
                    dram_energy_joules=turn_dram_energy,
                    gpu_power_avg_watts=turn_gpu_power_avg,
                    cpu_power_avg_watts=turn_cpu_power_avg,
                    dram_power_avg_watts=turn_dram_power_avg,
                    cost_usd=record.get("cost_usd"),
                )
            )

        return turns

    def _correlate_energy(
        self,
        trace: QueryTrace,
        readings: list[TelemetrySample],
    ) -> QueryTrace:
        """Correlate energy readings with the trace at the query level.

        If per-turn energy was not populated from events (e.g., no event
        recorder), distribute energy evenly across turns based on wall clock.
        """
        if not readings or not trace.turns:
            return trace

        total_wall = sum(t.wall_clock_s for t in trace.turns)
        if total_wall <= 0:
            return trace

        # Each component may have different hardware support. Do not let GPU
        # turn samples suppress CPU or DRAM allocation when those are missing.
        for energy_field, power_field, turn_field in (
            ("energy_joules", "power_watts", "gpu_energy_joules"),
            ("cpu_energy_joules", "cpu_power_watts", "cpu_energy_joules"),
            ("dram_energy_joules", "dram_power_watts", "dram_energy_joules"),
        ):
            if any(getattr(turn, turn_field) is not None for turn in trace.turns):
                continue
            energy = _compute_energy_delta(readings, energy_field)
            if energy is None:
                energy = _estimate_energy_from_power(readings, power_field, total_wall)
            if energy is None:
                continue
            for turn in trace.turns:
                setattr(turn, turn_field, energy * turn.wall_clock_s / total_wall)

        return trace

    def _save_query_artifacts(
        self,
        index: int,
        record: DatasetRecord,
        trace: QueryTrace,
    ) -> None:
        """Save per-query artifacts to structured subdirectories."""
        assert self._run_dir is not None
        instance_id = record.dataset_metadata.get("instance_id", "")
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", str(instance_id))[:80]
        query_dir = self._run_dir / "artifacts" / f"q{index:04d}_{slug}"
        query_dir.mkdir(parents=True, exist_ok=True)

        # response.txt — full agent response
        (query_dir / "response.txt").write_text(
            trace.response_text or "", encoding="utf-8"
        )

        # metadata.json — query-level metadata
        meta: dict[str, object] = {
            "query_id": trace.query_id,
            "instance_id": str(instance_id),
            "completed": trace.completed,
            "timed_out": trace.timed_out,
            "wall_clock_s": trace.total_wall_clock_s,
            "num_turns": trace.num_turns,
            "query_gpu_energy_joules": trace.query_gpu_energy_joules,
            "judge_gpu_energy_joules": trace.judge_gpu_energy_joules,
            "judge_wall_clock_s": trace.judge_wall_clock_s,
            "total_task_gpu_energy_joules": trace.total_task_gpu_energy_joules,
        }
        # Include select dataset metadata
        for key in (
            "repo",
            "base_commit",
            "dataset_name",
            "task_id",
            "occupation",
            "sector",
            "repeat_index",
            "repeat_count",
            "repeated_instance",
            "occupation_batch_index",
            "occupation_batch_id",
            "occupation_task_position",
            "source_task_index",
            "task_instance_position",
            "include_rubric_in_prompt",
            "deliverable_files",
            "is_resolved",
            "unscorable_reason",
            "score_metadata",
            "test_results",
            "agent_metadata",
            "token_source",
            "failure_type",
            "gdpval_outputs_dir",
            "gdpval_submitted_files",
            "deliverable_validation",
            "evaluation_metadata",
        ):
            val = record.dataset_metadata.get(key)
            if val is not None:
                meta[key] = val
        (query_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )

        # patch.diff — extracted patch (if present)
        patch = _extract_patch(trace.response_text or "")
        if patch:
            (query_dir / "patch.diff").write_text(patch, encoding="utf-8")

    def _build_profiling_record(
        self,
        record: DatasetRecord,
        trace: QueryTrace,
        model: str,
    ) -> ProfilingRecord:
        """Build a ProfilingRecord from a completed query trace."""
        total_input_tokens = trace.total_input_tokens
        total_output_tokens = trace.total_output_tokens
        total_seconds = trace.total_wall_clock_s

        # Energy metrics from trace (per-turn sums, falling back to query-level)
        gpu_energy = trace.total_gpu_energy_joules
        cpu_energy = trace.total_cpu_energy_joules

        # Per-token energy normalization
        energy_per_output_token = None
        energy_per_total_token = None
        total_tokens = (
            total_input_tokens + total_output_tokens
            if total_input_tokens is not None and total_output_tokens is not None
            else None
        )
        if gpu_energy is not None and gpu_energy > 0:
            if total_output_tokens is not None and total_output_tokens > 0:
                energy_per_output_token = gpu_energy / total_output_tokens
            if total_tokens is not None and total_tokens > 0:
                energy_per_total_token = gpu_energy / total_tokens

        energy_metrics = EnergyMetrics(
            per_query_joules=gpu_energy,
            total_joules=gpu_energy,
            cpu_per_query_joules=cpu_energy,
            cpu_total_joules=cpu_energy,
            judge_gpu_joules=trace.judge_gpu_energy_joules,
            judge_cpu_joules=trace.judge_cpu_energy_joules,
            total_task_gpu_joules=trace.total_task_gpu_energy_joules,
            total_task_cpu_joules=trace.total_task_cpu_energy_joules,
            energy_per_output_token_joules=energy_per_output_token,
            energy_per_total_token_joules=energy_per_total_token,
        )

        # Latency
        per_token_ms = None
        throughput = None
        if total_output_tokens is not None and total_output_tokens > 0 and total_seconds > 0:
            per_token_ms = (total_seconds * 1000.0) / total_output_tokens
            throughput = total_output_tokens / total_seconds

        latency_metrics = LatencyMetrics(
            per_token_ms=per_token_ms,
            throughput_tokens_per_sec=throughput,
            total_query_seconds=total_seconds,
        )

        # Cost — use trace cost if available, otherwise compute from pricing.
        # Note: AgentRunResult.cost_usd defaults to 0.0 (not Optional), so
        # treat 0.0 as "not provided" and try pricing tables. The localhost
        # fallback below ensures local models still get cost=0.0.
        cost = trace.total_cost_usd
        if (
            (cost is None or cost == 0.0)
            and total_input_tokens is not None
            and total_output_tokens is not None
            and total_input_tokens > 0
        ):
            from ..cost.pricing import calculate_cost

            provider = self._config.get("provider", "")
            cost = calculate_cost(provider, model, total_input_tokens, total_output_tokens)
            if cost == 0.0:
                cost = None

        # Local models (localhost inference) have zero dollar cost
        base_url = self._config.get("client_base_url", "")
        if cost is None and ("localhost" in base_url or "127.0.0.1" in base_url):
            cost = 0.0

        cost_metrics = CostMetrics(total_cost_usd=cost)

        # Power metrics from trace
        power_metrics = PowerMetrics(
            gpu=PowerComponentMetrics(
                per_query_watts=MetricStats(avg=trace.avg_gpu_power_watts),
            ),
            cpu=PowerComponentMetrics(
                per_query_watts=MetricStats(avg=trace.avg_cpu_power_watts),
            ),
        )

        # Derived efficiency
        throughput_per_watt = None
        avg_gpu_power = trace.avg_gpu_power_watts
        if throughput is not None and avg_gpu_power is not None and avg_gpu_power > 0:
            throughput_per_watt = throughput / avg_gpu_power

        model_metrics = ModelMetrics(
            compute_metrics=ComputeMetrics(),
            energy_metrics=energy_metrics,
            latency_metrics=latency_metrics,
            memory_metrics=MemoryMetrics(),
            power_metrics=power_metrics,
            temperature_metrics=MetricStats(),
            token_metrics=TokenMetrics(
                input=total_input_tokens,
                output=total_output_tokens,
                total=total_tokens,
            ),
            efficiency=DerivedEfficiencyMetrics(
                throughput_per_watt=throughput_per_watt,
            ),
            cost=cost_metrics,
            lm_response=trace.response_text,
        )

        return ProfilingRecord(
            problem=record.problem,
            answer=record.answer,
            dataset_metadata=dict(record.dataset_metadata),
            subject=record.subject,
            model_answers={model: trace.response_text},
            model_metrics={model: model_metrics},
        )

    @property
    def traces(self) -> list[QueryTrace]:
        """Return collected traces."""
        return list(self._traces)

    @property
    def records(self) -> list[ProfilingRecord]:
        """Return collected profiling records."""
        return list(self._records)


__all__ = ["AgenticRunner"]
