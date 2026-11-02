"""Trace data model for eval benchmark runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


def _json_safe(value: Any) -> Any:
    """Convert metadata from agents/datasets into JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _json_safe(value.model_dump())
    if hasattr(value, "dict") and callable(value.dict):
        return _json_safe(value.dict())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


@dataclass
class TurnTrace:
    """Per-turn telemetry data."""

    turn_index: int
    input_tokens: Optional[int] = 0
    output_tokens: Optional[int] = 0
    tool_result_tokens: Optional[int] = 0
    tools_called: List[str] = field(default_factory=list)
    tool_latencies_s: Dict[str, float] = field(default_factory=dict)
    wall_clock_s: float = 0.0
    error: Optional[str] = None
    # Energy and cost fields
    gpu_energy_joules: Optional[float] = None
    cpu_energy_joules: Optional[float] = None
    gpu_power_avg_watts: Optional[float] = None
    cpu_power_avg_watts: Optional[float] = None
    dram_power_avg_watts: Optional[float] = None
    cost_usd: Optional[float] = None
    dram_energy_joules: Optional[float] = None
    peak_watts: Optional[float] = None
    reasoning_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    has_parallel_tools: bool = False
    shared_device_warning: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool_result_tokens": self.tool_result_tokens,
            "tools_called": list(self.tools_called),
            "tool_latencies_s": dict(self.tool_latencies_s),
            "wall_clock_s": self.wall_clock_s,
            "error": self.error,
            "gpu_energy_joules": self.gpu_energy_joules,
            "cpu_energy_joules": self.cpu_energy_joules,
            "gpu_power_avg_watts": self.gpu_power_avg_watts,
            "cpu_power_avg_watts": self.cpu_power_avg_watts,
            "cost_usd": self.cost_usd,
            "dram_energy_joules": self.dram_energy_joules,
            "dram_power_avg_watts": self.dram_power_avg_watts,
            "peak_watts": self.peak_watts,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "has_parallel_tools": self.has_parallel_tools,
            "shared_device_warning": self.shared_device_warning,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TurnTrace:
        return cls(
            turn_index=d["turn_index"],
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            tool_result_tokens=d.get("tool_result_tokens", 0),
            tools_called=d.get("tools_called", []),
            tool_latencies_s=d.get("tool_latencies_s", {}),
            wall_clock_s=d.get("wall_clock_s", 0.0),
            error=d.get("error"),
            gpu_energy_joules=d.get("gpu_energy_joules"),
            cpu_energy_joules=d.get("cpu_energy_joules"),
            gpu_power_avg_watts=d.get("gpu_power_avg_watts"),
            cpu_power_avg_watts=d.get("cpu_power_avg_watts"),
            cost_usd=d.get("cost_usd"),
            dram_energy_joules=d.get("dram_energy_joules"),
            dram_power_avg_watts=d.get("dram_power_avg_watts"),
            peak_watts=d.get("peak_watts"),
            reasoning_tokens=d.get("reasoning_tokens"),
            cached_tokens=d.get("cached_tokens"),
            has_parallel_tools=d.get("has_parallel_tools", False),
            shared_device_warning=d.get("shared_device_warning", False),
        )


@dataclass
class QueryTrace:
    """Per-query aggregate telemetry."""

    query_id: str
    workload_type: str
    query_text: str = ""
    response_text: str = ""
    turns: List[TurnTrace] = field(default_factory=list)
    total_wall_clock_s: float = 0.0
    completed: bool = False
    timed_out: bool = False
    # Query-level energy fields (populated even when turns are empty)
    query_gpu_energy_joules: Optional[float] = None
    query_cpu_energy_joules: Optional[float] = None
    query_dram_energy_joules: Optional[float] = None
    query_gpu_power_avg_watts: Optional[float] = None
    query_cpu_power_avg_watts: Optional[float] = None
    query_dram_power_avg_watts: Optional[float] = None
    # Judge/evaluation energy is recorded separately so query energy remains
    # attributable to the agent prompt window.
    judge_gpu_energy_joules: Optional[float] = None
    judge_cpu_energy_joules: Optional[float] = None
    judge_dram_energy_joules: Optional[float] = None
    judge_gpu_power_avg_watts: Optional[float] = None
    judge_cpu_power_avg_watts: Optional[float] = None
    judge_dram_power_avg_watts: Optional[float] = None
    judge_wall_clock_s: Optional[float] = None
    prompt_start_time_s: Optional[float] = None
    prompt_end_time_s: Optional[float] = None
    judge_start_time_s: Optional[float] = None
    judge_end_time_s: Optional[float] = None
    telemetry_device_ids: List[int] = field(default_factory=list)
    telemetry_sample_count: int = 0
    telemetry_accounting: str = "query"
    batch_id: Optional[str] = None
    batch_index: Optional[int] = None
    batch_position: Optional[int] = None
    batch_size: Optional[int] = None
    batch_prompt_start_time_s: Optional[float] = None
    batch_prompt_end_time_s: Optional[float] = None
    batch_prompt_wall_clock_s: Optional[float] = None
    batch_total_wall_clock_s: Optional[float] = None
    batch_prompt_finish_offset_s: Optional[float] = None
    query_mbu_avg_pct: Optional[float] = None
    query_mbu_max_pct: Optional[float] = None
    is_resolved: Optional[bool] = None
    unscorable_reason: Optional[str] = None
    score_metadata: Dict[str, Any] = field(default_factory=dict)
    # True by default; runner sets False for cloud LMs where GPU telemetry is unavailable
    lm_energy_measurable: bool = True
    shared_device_warning: bool = False
    # evaluators may return float, bool, or a dict; float is the common case — cast explicitly when needed
    accuracy_score: Optional[float] = None
    accuracy_metadata: Dict[str, Any] = field(default_factory=dict)
    n_retries: int = 0

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def total_input_tokens(self) -> Optional[int]:
        if any(t.input_tokens is None for t in self.turns):
            return None
        return sum(t.input_tokens or 0 for t in self.turns)

    @property
    def total_output_tokens(self) -> Optional[int]:
        if any(t.output_tokens is None for t in self.turns):
            return None
        return sum(t.output_tokens or 0 for t in self.turns)

    @property
    def tool_call_count(self) -> int:
        return sum(len(t.tools_called) for t in self.turns)

    @property
    def total_tool_calls(self) -> int:
        return self.tool_call_count

    @property
    def total_gpu_energy_joules(self) -> Optional[float]:
        if self.telemetry_accounting == "batch":
            return None
        if self.query_gpu_energy_joules is not None:
            return self.query_gpu_energy_joules
        values = [t.gpu_energy_joules for t in self.turns if t.gpu_energy_joules is not None]
        if values:
            return sum(values)
        return None

    @property
    def total_cpu_energy_joules(self) -> Optional[float]:
        if self.telemetry_accounting == "batch":
            return None
        if self.query_cpu_energy_joules is not None:
            return self.query_cpu_energy_joules
        values = [t.cpu_energy_joules for t in self.turns if t.cpu_energy_joules is not None]
        if values:
            return sum(values)
        return None

    @property
    def total_dram_energy_joules(self) -> Optional[float]:
        if self.telemetry_accounting == "batch":
            return None
        if self.query_dram_energy_joules is not None:
            return self.query_dram_energy_joules
        values = [t.dram_energy_joules for t in self.turns if t.dram_energy_joules is not None]
        return sum(values) if values else None

    @property
    def total_task_gpu_energy_joules(self) -> Optional[float]:
        values = [
            value
            for value in (self.total_gpu_energy_joules, self.judge_gpu_energy_joules)
            if value is not None
        ]
        return sum(values) if values else None

    @property
    def total_task_cpu_energy_joules(self) -> Optional[float]:
        values = [
            value
            for value in (self.total_cpu_energy_joules, self.judge_cpu_energy_joules)
            if value is not None
        ]
        return sum(values) if values else None

    @property
    def total_task_dram_energy_joules(self) -> Optional[float]:
        values = [
            value
            for value in (self.total_dram_energy_joules, self.judge_dram_energy_joules)
            if value is not None
        ]
        return sum(values) if values else None

    @property
    def total_cost_usd(self) -> Optional[float]:
        values = [t.cost_usd for t in self.turns if t.cost_usd is not None]
        return sum(values) if values else None

    @property
    def total_tokens(self) -> Optional[int]:
        """Total tokens (input + output) across all turns."""
        if self.total_input_tokens is None or self.total_output_tokens is None:
            return None
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_gpu_power_watts(self) -> Optional[float]:
        """Mean GPU power across turns; falls back to query-level power."""
        values = [t.gpu_power_avg_watts for t in self.turns if t.gpu_power_avg_watts is not None]
        if values:
            return sum(values) / len(values)
        return self.query_gpu_power_avg_watts

    @property
    def avg_cpu_power_watts(self) -> Optional[float]:
        """Mean CPU power across turns; falls back to query-level power."""
        values = [t.cpu_power_avg_watts for t in self.turns if t.cpu_power_avg_watts is not None]
        if values:
            return sum(values) / len(values)
        return self.query_cpu_power_avg_watts

    @property
    def avg_dram_power_watts(self) -> Optional[float]:
        values = [t.dram_power_avg_watts for t in self.turns if t.dram_power_avg_watts is not None]
        return sum(values) / len(values) if values else self.query_dram_power_avg_watts

    @property
    def throughput_tokens_per_sec(self) -> Optional[float]:
        """Output tokens per second; None if zero tokens or zero time."""
        output_tokens = self.total_output_tokens
        if output_tokens is not None and output_tokens > 0 and self.total_wall_clock_s > 0:
            return output_tokens / self.total_wall_clock_s
        return None

    @property
    def energy_per_token_joules(self) -> Optional[float]:
        """GPU energy per output token; None if no energy data or zero tokens."""
        gpu_energy = self.total_gpu_energy_joules
        output_tokens = self.total_output_tokens
        if gpu_energy is not None and gpu_energy > 0 and output_tokens is not None and output_tokens > 0:
            return gpu_energy / output_tokens
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "workload_type": self.workload_type,
            "query_text": self.query_text,
            "response_text": self.response_text,
            "turns": [t.to_dict() for t in self.turns],
            "num_turns": self.num_turns,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "tool_call_count": self.tool_call_count,
            "total_tool_calls": self.total_tool_calls,
            "total_wall_clock_s": self.total_wall_clock_s,
            "completed": self.completed,
            "timed_out": self.timed_out,
            "query_gpu_energy_joules": self.query_gpu_energy_joules,
            "query_cpu_energy_joules": self.query_cpu_energy_joules,
            "query_dram_energy_joules": self.query_dram_energy_joules,
            "query_gpu_power_avg_watts": self.query_gpu_power_avg_watts,
            "query_cpu_power_avg_watts": self.query_cpu_power_avg_watts,
            "query_dram_power_avg_watts": self.query_dram_power_avg_watts,
            "judge_gpu_energy_joules": self.judge_gpu_energy_joules,
            "judge_cpu_energy_joules": self.judge_cpu_energy_joules,
            "judge_dram_energy_joules": self.judge_dram_energy_joules,
            "judge_gpu_power_avg_watts": self.judge_gpu_power_avg_watts,
            "judge_cpu_power_avg_watts": self.judge_cpu_power_avg_watts,
            "judge_dram_power_avg_watts": self.judge_dram_power_avg_watts,
            "judge_wall_clock_s": self.judge_wall_clock_s,
            "prompt_start_time_s": self.prompt_start_time_s,
            "prompt_end_time_s": self.prompt_end_time_s,
            "judge_start_time_s": self.judge_start_time_s,
            "judge_end_time_s": self.judge_end_time_s,
            "telemetry_device_ids": list(self.telemetry_device_ids),
            "telemetry_sample_count": self.telemetry_sample_count,
            "telemetry_accounting": self.telemetry_accounting,
            "batch_id": self.batch_id,
            "batch_index": self.batch_index,
            "batch_position": self.batch_position,
            "batch_size": self.batch_size,
            "batch_prompt_start_time_s": self.batch_prompt_start_time_s,
            "batch_prompt_end_time_s": self.batch_prompt_end_time_s,
            "batch_prompt_wall_clock_s": self.batch_prompt_wall_clock_s,
            "batch_total_wall_clock_s": self.batch_total_wall_clock_s,
            "batch_prompt_finish_offset_s": self.batch_prompt_finish_offset_s,
            "total_task_gpu_energy_joules": self.total_task_gpu_energy_joules,
            "total_task_cpu_energy_joules": self.total_task_cpu_energy_joules,
            "total_task_dram_energy_joules": self.total_task_dram_energy_joules,
            "query_mbu_avg_pct": self.query_mbu_avg_pct,
            "query_mbu_max_pct": self.query_mbu_max_pct,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "energy_per_output_token_joules": self.energy_per_token_joules,
            "total_cost_usd": self.total_cost_usd,
            "is_resolved": self.is_resolved,
            "unscorable_reason": self.unscorable_reason,
            "score_metadata": _json_safe(self.score_metadata),
            "lm_energy_measurable": self.lm_energy_measurable,
            "shared_device_warning": self.shared_device_warning,
            "accuracy_score": self.accuracy_score,
            "accuracy_metadata": _json_safe(self.accuracy_metadata),
            "n_retries": self.n_retries,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> QueryTrace:
        return cls(
            query_id=d["query_id"],
            workload_type=d["workload_type"],
            query_text=d.get("query_text", ""),
            response_text=d.get("response_text", ""),
            turns=[TurnTrace.from_dict(t) for t in d.get("turns", [])],
            total_wall_clock_s=d.get("total_wall_clock_s", 0.0),
            completed=d.get("completed", False),
            timed_out=d.get("timed_out", False),
            query_gpu_energy_joules=d.get("query_gpu_energy_joules"),
            query_cpu_energy_joules=d.get("query_cpu_energy_joules"),
            query_dram_energy_joules=d.get("query_dram_energy_joules"),
            query_gpu_power_avg_watts=d.get("query_gpu_power_avg_watts"),
            query_cpu_power_avg_watts=d.get("query_cpu_power_avg_watts"),
            query_dram_power_avg_watts=d.get("query_dram_power_avg_watts"),
            judge_gpu_energy_joules=d.get("judge_gpu_energy_joules"),
            judge_cpu_energy_joules=d.get("judge_cpu_energy_joules"),
            judge_dram_energy_joules=d.get("judge_dram_energy_joules"),
            judge_gpu_power_avg_watts=d.get("judge_gpu_power_avg_watts"),
            judge_cpu_power_avg_watts=d.get("judge_cpu_power_avg_watts"),
            judge_dram_power_avg_watts=d.get("judge_dram_power_avg_watts"),
            judge_wall_clock_s=d.get("judge_wall_clock_s"),
            prompt_start_time_s=d.get("prompt_start_time_s"),
            prompt_end_time_s=d.get("prompt_end_time_s"),
            judge_start_time_s=d.get("judge_start_time_s"),
            judge_end_time_s=d.get("judge_end_time_s"),
            telemetry_device_ids=list(d.get("telemetry_device_ids") or []),
            telemetry_sample_count=int(d.get("telemetry_sample_count") or 0),
            telemetry_accounting=d.get("telemetry_accounting", "query"),
            batch_id=d.get("batch_id"),
            batch_index=d.get("batch_index"),
            batch_position=d.get("batch_position"),
            batch_size=d.get("batch_size"),
            batch_prompt_start_time_s=d.get("batch_prompt_start_time_s"),
            batch_prompt_end_time_s=d.get("batch_prompt_end_time_s"),
            batch_prompt_wall_clock_s=d.get("batch_prompt_wall_clock_s"),
            batch_total_wall_clock_s=d.get("batch_total_wall_clock_s"),
            batch_prompt_finish_offset_s=d.get("batch_prompt_finish_offset_s"),
            query_mbu_avg_pct=d.get("query_mbu_avg_pct"),
            query_mbu_max_pct=d.get("query_mbu_max_pct"),
            is_resolved=d.get("is_resolved"),
            unscorable_reason=d.get("unscorable_reason"),
            score_metadata=d.get("score_metadata") or {},
            lm_energy_measurable=d.get("lm_energy_measurable", True),
            shared_device_warning=d.get("shared_device_warning", False),
            accuracy_score=d.get("accuracy_score"),
            accuracy_metadata=d.get("accuracy_metadata", {}),
            n_retries=d.get("n_retries", 0),
        )

    def save_jsonl(self, path: Path) -> None:
        """Append this trace as a JSONL line."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(self.to_dict()) + "\n")

    @classmethod
    def load_jsonl(cls, path: Path) -> List[QueryTrace]:
        """Load traces from a JSONL file."""
        traces = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(cls.from_dict(json.loads(line)))
        return traces

    @staticmethod
    def to_hf_dataset(traces: List[QueryTrace]) -> Any:
        """Convert a list of QueryTrace objects to a HuggingFace Dataset.

        Returns:
            A datasets.Dataset with one row per trace.
        """
        from datasets import Dataset

        rows = []
        for trace in traces:
            rows.append({
                "query_id": trace.query_id,
                "workload_type": trace.workload_type,
                "query_text": trace.query_text,
                "response_text": trace.response_text,
                "num_turns": trace.num_turns,
                "total_input_tokens": trace.total_input_tokens,
                "total_output_tokens": trace.total_output_tokens,
                "total_tool_calls": trace.total_tool_calls,
                "total_wall_clock_s": trace.total_wall_clock_s,
                "total_gpu_energy_joules": trace.total_gpu_energy_joules,
                "total_cpu_energy_joules": trace.total_cpu_energy_joules,
                "total_dram_energy_joules": trace.total_dram_energy_joules,
                "judge_gpu_energy_joules": trace.judge_gpu_energy_joules,
                "judge_cpu_energy_joules": trace.judge_cpu_energy_joules,
                "judge_dram_energy_joules": trace.judge_dram_energy_joules,
                "judge_wall_clock_s": trace.judge_wall_clock_s,
                "prompt_start_time_s": trace.prompt_start_time_s,
                "prompt_end_time_s": trace.prompt_end_time_s,
                "judge_start_time_s": trace.judge_start_time_s,
                "judge_end_time_s": trace.judge_end_time_s,
                "telemetry_device_ids": trace.telemetry_device_ids,
                "telemetry_sample_count": trace.telemetry_sample_count,
                "telemetry_accounting": trace.telemetry_accounting,
                "batch_id": trace.batch_id,
                "batch_index": trace.batch_index,
                "batch_position": trace.batch_position,
                "batch_size": trace.batch_size,
                "batch_prompt_start_time_s": trace.batch_prompt_start_time_s,
                "batch_prompt_end_time_s": trace.batch_prompt_end_time_s,
                "batch_prompt_wall_clock_s": trace.batch_prompt_wall_clock_s,
                "batch_total_wall_clock_s": trace.batch_total_wall_clock_s,
                "batch_prompt_finish_offset_s": trace.batch_prompt_finish_offset_s,
                "total_task_gpu_energy_joules": trace.total_task_gpu_energy_joules,
                "total_task_cpu_energy_joules": trace.total_task_cpu_energy_joules,
                "total_task_dram_energy_joules": trace.total_task_dram_energy_joules,
                "total_tokens": trace.total_tokens,
                "total_cost_usd": trace.total_cost_usd,
                "avg_gpu_power_watts": trace.avg_gpu_power_watts,
                "avg_cpu_power_watts": trace.avg_cpu_power_watts,
                "avg_dram_power_watts": trace.avg_dram_power_watts,
                "throughput_tokens_per_sec": trace.throughput_tokens_per_sec,
                "energy_per_token_joules": trace.energy_per_token_joules,
                "query_mbu_avg_pct": trace.query_mbu_avg_pct,
                "query_mbu_max_pct": trace.query_mbu_max_pct,
                "completed": trace.completed,
                "timed_out": trace.timed_out,
                "is_resolved": trace.is_resolved,
                "unscorable_reason": trace.unscorable_reason,
                "score_metadata_json": json.dumps(_json_safe(trace.score_metadata)),
                "trace_json": json.dumps(trace.to_dict()),
            })
        return Dataset.from_list(rows)


__all__ = ["TurnTrace", "QueryTrace"]
