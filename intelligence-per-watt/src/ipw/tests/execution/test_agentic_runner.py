"""Tests for execution/agentic_runner.py — AgenticRunner."""

from __future__ import annotations

import asyncio
import json
import zipfile
from unittest.mock import MagicMock

import pytest

from ipw.agents.base import BaseAgent
from ipw.core.types import AgentRunResult, DatasetRecord, TelemetryReading
from ipw.execution.agentic_runner import (
    AgenticRunner,
    _classify_agent_result_failure,
    _compute_energy_delta,
    _compute_power_avg,
    _gdpval_finish_summary_from_validation,
    _sanitize_gdpval_response_text,
)
from ipw.execution.telemetry_session import TelemetrySample
from ipw.execution.trace import QueryTrace, TurnTrace
from ipw.telemetry.events import EventRecorder, EventType


def test_turn_dram_energy_is_correlated_even_with_gpu_turn_energy() -> None:
    trace = QueryTrace(
        query_id="q1",
        workload_type="agentic",
        turns=[
            TurnTrace(turn_index=0, wall_clock_s=1.0, gpu_energy_joules=5.0),
            TurnTrace(turn_index=1, wall_clock_s=1.0, gpu_energy_joules=5.0),
        ],
    )
    samples = [
        TelemetrySample(
            timestamp=10.0,
            reading=TelemetryReading(cpu_energy_joules=100.0, dram_energy_joules=20.0),
        ),
        TelemetrySample(
            timestamp=12.0,
            reading=TelemetryReading(cpu_energy_joules=140.0, dram_energy_joules=26.0),
        ),
    ]

    result = AgenticRunner.__new__(AgenticRunner)._correlate_energy(trace, samples)

    assert [turn.cpu_energy_joules for turn in result.turns] == pytest.approx([20.0, 20.0])
    assert [turn.dram_energy_joules for turn in result.turns] == pytest.approx([3.0, 3.0])
    assert result.total_dram_energy_joules == pytest.approx(6.0)
    assert QueryTrace.from_dict(result.to_dict()).total_dram_energy_joules == pytest.approx(6.0)


def test_gdpval_response_sanitizer_removes_final_nudge_and_state_dump() -> None:
    raw = (
        "USER_CONTEXT: Awaiting task initialization. No events provided.\n"
        "TASK_TRACKING: No active tasks.\n"
        "Submit your final answer now with the finish tool. Do not call terminal "
        "or file_editor. If output files already exist, finish with a concise "
        "list of their paths.\n\n"
        "--- GDPval final deliverable checklist attempt 1 ---\n\n"
        "Final deliverable checklist complete.\n"
        "- report.xlsx"
    )

    cleaned = _sanitize_gdpval_response_text(raw)

    assert "Submit your final answer" not in cleaned
    assert "USER_CONTEXT" not in cleaned
    assert "Final deliverable checklist complete" in cleaned
    assert "report.xlsx" in cleaned


def test_gdpval_finish_summary_uses_primary_files() -> None:
    summary = _gdpval_finish_summary_from_validation(
        {
            "status": "ok",
            "files": ["/workspace/scratch.py", "/workspace/report.xlsx"],
            "primary_files": ["/workspace/report.xlsx"],
        }
    )

    assert summary == "Submitted deliverable file(s):\n- report.xlsx"


def test_model_script_traceback_is_not_agent_run_failure() -> None:
    result = AgentRunResult(
        content=(
            "Traceback (most recent call last):\n"
            "  File \"build.py\", line 3, in <module>\n"
            "ValueError: model-generated script failed\n"
        ),
        metadata={},
    )

    assert _classify_agent_result_failure(result) == (None, None)


def test_gdpval_repair_script_log_summary_surfaces_autorun_failure() -> None:
    record = DatasetRecord(
        problem="Create report.xlsx.",
        answer="rubric",
        subject="audit",
        dataset_metadata={
            "gdpval_autorun_repair_script_logs": [
                {
                    "script": "build_workbook.py",
                    "returncode": 1,
                    "wall_clock_s": 0.5,
                    "output_tail": "TypeError: bad workbook mutation",
                }
            ]
        },
    )

    summary = AgenticRunner._gdpval_repair_script_log_summary(record)

    assert "build_workbook.py exited with 1" in summary
    assert "TypeError: bad workbook mutation" in summary
    assert "do not repeat the same failing script unchanged" in summary


def test_gdpval_repair_type_playbook_matches_expected_extensions() -> None:
    playbook = AgenticRunner._gdpval_repair_type_playbook(
        {"expected_extensions": [".xlsx", ".docx", ".mp4"]}
    )

    assert "Spreadsheet: use openpyxl" in playbook
    assert "Word document: use python-docx" in playbook
    assert "Video: create the final media file" in playbook
    assert "ffprobe" in playbook


class TestAgenticRunner:
    """Test AgenticRunner with mocks."""

    def _make_runner(
        self,
        agent: MagicMock | None = None,
        dataset: MagicMock | None = None,
        event_recorder: EventRecorder | None = None,
        config: dict | None = None,
    ) -> AgenticRunner:
        if agent is None:
            agent = MagicMock()
            agent.run.return_value = AgentRunResult(
                content="mock answer",
                tool_calls_attempted=1,
                tool_calls_succeeded=1,
                num_turns=1,
                input_tokens=50,
                output_tokens=25,
            )
        if dataset is None:
            dataset = MagicMock()
            records = [
                DatasetRecord(
                    problem="Q1", answer="A1", subject="math",
                    dataset_metadata={"dataset_name": "test"},
                )
            ]
            dataset.__iter__ = MagicMock(return_value=iter(records))
            dataset.size.return_value = 1
        return AgenticRunner(
            agent=agent,
            dataset=dataset,
            telemetry_session=None,
            config=config or {"model": "test-model"},
            event_recorder=(
                event_recorder if event_recorder is not None else EventRecorder()
            ),
        )

    def test_run_returns_traces(self) -> None:
        runner = self._make_runner()
        traces = asyncio.run(runner.run())
        assert len(traces) == 1
        assert isinstance(traces[0], QueryTrace)
        assert traces[0].completed is True

    def test_agent_invocation_returns_content(self) -> None:
        runner = self._make_runner()
        traces = asyncio.run(runner.run())
        assert traces[0].response_text == "mock answer"

    def test_traces_property(self) -> None:
        runner = self._make_runner()
        asyncio.run(runner.run())
        traces = runner.traces
        assert len(traces) == 1
        assert traces[0].query_id == "q0000"

    def test_records_property(self) -> None:
        runner = self._make_runner()
        asyncio.run(runner.run())
        records = runner.records
        assert len(records) == 1
        assert records[0].problem == "Q1"
        assert "test-model" in records[0].model_metrics

    def test_profiling_record_constructed(self) -> None:
        runner = self._make_runner()
        asyncio.run(runner.run())
        record = runner.records[0]
        assert "test-model" in record.model_metrics
        # Token metrics come from trace turns, not AgentRunResult directly.
        # With no events recorded, there are no turns, so tokens are 0.
        metrics = record.model_metrics["test-model"]
        assert metrics.lm_response == "mock answer"

    def test_agent_failure_creates_incomplete_trace(self) -> None:
        agent = MagicMock()
        agent.run.side_effect = RuntimeError("agent error")
        runner = self._make_runner(agent=agent)
        traces = asyncio.run(runner.run())
        assert len(traces) == 1
        assert traces[0].completed is False
        assert "agent error" in traces[0].response_text

    def test_returned_litellm_timeout_is_infra_timeout(self) -> None:
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(
            content=(
                "Conversation run failed for id=abc: litellm.Timeout: "
                "APITimeoutError - Request timed out."
            )
        )

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q1",
                answer="",
                subject="s",
                dataset_metadata={"dataset_name": "gdpval"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            telemetry_session=None,
            config={"model": "test-model"},
        )
        traces = asyncio.run(runner.run())

        assert traces[0].completed is False
        assert traces[0].timed_out is True
        assert traces[0].unscorable_reason == "agent_or_llm_timeout"
        assert traces[0].score_metadata["failure_type"] == "infra_timeout"

    def test_max_queries_limits_processing(self) -> None:
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(content="ok")

        dataset = MagicMock()
        records = [
            DatasetRecord(problem=f"Q{i}", answer=f"A{i}", subject="s")
            for i in range(10)
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 10

        runner = self._make_runner(agent=agent, dataset=dataset)
        traces = asyncio.run(runner.run(max_queries=3))
        assert len(traces) == 3

    def test_start_index_keeps_max_queries_as_count(self) -> None:
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(content="ok")

        dataset = MagicMock()
        records = [
            DatasetRecord(problem=f"Q{i}", answer=f"A{i}", subject="s")
            for i in range(10)
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 10

        runner = self._make_runner(agent=agent, dataset=dataset)
        traces = asyncio.run(runner.run(max_queries=3, start_index=4))

        assert [trace.query_id for trace in traces] == ["q0004", "q0005", "q0006"]
        assert [trace.query_text for trace in traces] == ["Q4", "Q5", "Q6"]

    def test_subset_manifest_records_query_hashes(self, tmp_path) -> None:
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(content="ok")

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem=f"Q{i}",
                answer=f"A{i}",
                subject="s",
                dataset_metadata={"task_id": f"task-{i}"},
            )
            for i in range(3)
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 3

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            telemetry_session=None,
            config={"model": "test-model", "dataset": "test-dataset", "agent": "test-agent"},
            event_recorder=EventRecorder(),
            run_dir=tmp_path,
        )

        traces = asyncio.run(runner.run(max_queries=2))
        manifest = json.loads((tmp_path / "subset_manifest.json").read_text())

        assert len(traces) == 2
        assert manifest["subset_size"] == 2
        assert len(manifest["records"]) == 2
        assert manifest["records"][0]["query_id"] == "q0000"
        assert manifest["records"][0]["stable_ids"] == {"task_id": "task-0"}
        assert manifest["records"][0]["query_hash"]
        assert runner._config["subset"]["subset_hash"] == manifest["subset_hash"]

    def test_multi_turn_trace_building(self) -> None:
        """Verify _build_turn_traces correctly parses events into turns."""
        from ipw.telemetry.events import AgentEvent, EventType

        runner = self._make_runner()
        now = 1000.0

        events = [
            AgentEvent(event_type=EventType.LM_INFERENCE_START, timestamp=now),
            AgentEvent(
                event_type=EventType.TOOL_CALL_START,
                timestamp=now + 0.1,
                metadata={"tool": "calc"},
            ),
            AgentEvent(
                event_type=EventType.TOOL_CALL_END,
                timestamp=now + 0.5,
                metadata={"tool": "calc"},
            ),
            AgentEvent(
                event_type=EventType.LM_INFERENCE_END,
                timestamp=now + 1.0,
                metadata={"prompt_tokens": 50, "completion_tokens": 20},
            ),
            AgentEvent(
                event_type=EventType.LM_INFERENCE_START,
                timestamp=now + 1.1,
            ),
            AgentEvent(
                event_type=EventType.LM_INFERENCE_END,
                timestamp=now + 1.5,
                metadata={"prompt_tokens": 30, "completion_tokens": 10},
            ),
        ]

        turns = runner._build_turn_traces(events, readings=[])

        assert len(turns) == 2
        assert turns[0].tools_called == ["calc"]
        assert turns[0].input_tokens == 50
        assert turns[0].output_tokens == 20
        assert turns[1].input_tokens == 30
        assert turns[1].output_tokens == 10

    def test_tool_after_lm_end_attaches_to_previous_turn(self) -> None:
        """Terminal-style tools run after the LLM response that requested them."""
        from ipw.telemetry.events import AgentEvent, EventType

        runner = self._make_runner()
        now = 1000.0

        events = [
            AgentEvent(event_type=EventType.LM_INFERENCE_START, timestamp=now),
            AgentEvent(
                event_type=EventType.LM_INFERENCE_END,
                timestamp=now + 1.0,
                metadata={"prompt_tokens": 50, "completion_tokens": 20},
            ),
            AgentEvent(
                event_type=EventType.TOOL_CALL_START,
                timestamp=now + 1.1,
                metadata={"tool": "terminal"},
            ),
            AgentEvent(
                event_type=EventType.TOOL_CALL_END,
                timestamp=now + 2.0,
                metadata={"tool": "terminal"},
            ),
            AgentEvent(
                event_type=EventType.LM_INFERENCE_START,
                timestamp=now + 2.1,
            ),
            AgentEvent(
                event_type=EventType.LM_INFERENCE_END,
                timestamp=now + 3.0,
                metadata={"prompt_tokens": 30, "completion_tokens": 10},
            ),
        ]

        turns = runner._build_turn_traces(events, readings=[])

        assert len(turns) == 2
        assert turns[0].tools_called == ["terminal"]
        assert turns[0].tool_latencies_s["terminal"] == pytest.approx(0.9)
        assert turns[0].wall_clock_s == pytest.approx(2.0)
        assert turns[1].tools_called == []

    def test_event_recorder_integration_with_runner(self) -> None:
        """Verify events recorded during agent.run() flow into traces.

        Note: AgenticRunner.__init__ uses ``event_recorder or EventRecorder()``
        which evaluates an empty recorder as falsy (because __len__ == 0).
        We work around this by assigning the recorder directly after init.
        """
        from ipw.telemetry.events import EventType

        recorder = EventRecorder()
        agent = MagicMock()

        def run_with_events(prompt: str, **kwargs) -> AgentRunResult:
            recorder.record(EventType.LM_INFERENCE_START)
            recorder.record(EventType.TOOL_CALL_START, tool="calc")
            recorder.record(EventType.TOOL_CALL_END, tool="calc")
            recorder.record(
                EventType.LM_INFERENCE_END,
                prompt_tokens=50,
                completion_tokens=20,
            )
            recorder.record(EventType.LM_INFERENCE_START)
            recorder.record(
                EventType.LM_INFERENCE_END,
                prompt_tokens=30,
                completion_tokens=10,
            )
            return AgentRunResult(content="final answer")

        agent.run.side_effect = run_with_events

        runner = self._make_runner(agent=agent)
        # Directly inject the recorder to bypass the falsy-empty-container
        # issue in AgenticRunner.__init__ (event_recorder or EventRecorder())
        runner._event_recorder = recorder

        traces = asyncio.run(runner.run())

        assert len(traces) == 1
        trace = traces[0]
        assert trace.num_turns == 2
        assert trace.turns[0].tools_called == ["calc"]
        assert trace.turns[0].input_tokens == 50
        assert trace.turns[0].output_tokens == 20
        assert trace.turns[1].input_tokens == 30
        assert trace.turns[1].output_tokens == 10

    def test_task_env_integration(self) -> None:
        """Verify create_task_env context wraps agent.run() and run_tests() is called."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(content="done")

        mock_env = MagicMock()
        mock_env.__enter__ = MagicMock(return_value=mock_env)
        mock_env.__exit__ = MagicMock(return_value=False)

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Task 1", answer="", subject="test",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = mock_env
        mock_env.run_tests.side_effect = lambda: records[0].dataset_metadata.update(
            {
                "is_resolved": True,
                "test_results": {"parser_results": {"test_one": "passed"}},
            }
        )
        dataset.score.return_value = (
            True,
            {
                "match_type": "test_script",
                "test_results": {"parser_results": {"test_one": "passed"}},
            },
        )

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            config={"model": "test-model"},
        )
        traces = asyncio.run(runner.run())

        assert len(traces) == 1
        assert traces[0].completed is True
        dataset.create_task_env.assert_called_once()
        mock_env.__enter__.assert_called_once()
        mock_env.__exit__.assert_called_once()
        mock_env.run_tests.assert_called_once()
        dataset.score.assert_called_once()
        assert traces[0].is_resolved is True
        assert traces[0].score_metadata["match_type"] == "test_script"

    def test_task_env_none_uses_nullcontext(self) -> None:
        """When create_task_env returns None, agent.run() still works."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(content="ok")

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q", answer="A", subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            config={"model": "test-model"},
        )
        traces = asyncio.run(runner.run())

        assert len(traces) == 1
        assert traces[0].completed is True
        agent.run.assert_called_once()

    def test_concurrent_execution(self) -> None:
        """Test concurrency > 1 processes all queries and returns correct count."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(content="concurrent ok")

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem=f"Q{i}", answer=f"A{i}", subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
            for i in range(4)
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 4
        dataset.create_task_env.return_value = None

        def make_agent():
            a = MagicMock()
            a.run.return_value = AgentRunResult(content="concurrent ok")
            return a

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            config={"model": "test-model"},
            concurrency=2,
            agent_factory=make_agent,
        )
        traces = asyncio.run(runner.run())

        assert len(traces) == 4
        assert all(t.completed for t in traces)

    def test_concurrent_factory_agent_events_flow_into_trace(self) -> None:
        """Concurrent agents and runner must share the same EventRecorder."""

        class RecordingAgent(BaseAgent):
            def run(self, input: str, **kwargs) -> AgentRunResult:
                self._record_event(EventType.LM_INFERENCE_START)
                self._record_event(
                    EventType.LM_INFERENCE_END,
                    prompt_tokens=11,
                    completion_tokens=7,
                )
                return AgentRunResult(content=f"ok: {input}")

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem=f"Q{i}",
                answer=f"A{i}",
                subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
            for i in range(2)
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 2
        dataset.create_task_env.return_value = None

        def make_agent(recorder: EventRecorder) -> BaseAgent:
            return RecordingAgent(event_recorder=recorder)

        runner = AgenticRunner(
            agent=RecordingAgent(event_recorder=EventRecorder()),
            dataset=dataset,
            config={"model": "test-model"},
            concurrency=2,
            agent_factory=make_agent,
        )

        traces = asyncio.run(runner.run())

        assert len(traces) == 2
        assert all(trace.num_turns == 1 for trace in traces)
        assert all(trace.turns[0].input_tokens == 11 for trace in traces)
        assert all(trace.turns[0].output_tokens == 7 for trace in traces)

    def test_concurrent_uses_agent_factory(self) -> None:
        """Verify agent_factory is called for each concurrent task."""
        factory_calls = []

        def tracked_factory():
            a = MagicMock()
            a.run.return_value = AgentRunResult(content="factory agent")
            factory_calls.append(a)
            return a

        agent = MagicMock()
        agent.run.return_value = AgentRunResult(content="main agent")

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem=f"Q{i}", answer=f"A{i}", subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
            for i in range(3)
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 3
        dataset.create_task_env.return_value = None

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            config={"model": "test-model"},
            concurrency=3,
            agent_factory=tracked_factory,
        )
        traces = asyncio.run(runner.run())

        assert len(traces) == 3
        assert len(factory_calls) == 3

    def test_batch_telemetry_chunks_queries_and_suppresses_per_query_energy(
        self,
        tmp_path,
    ) -> None:
        """Batch mode records one energy row per chunk and no per-query joules."""

        class FakeTelemetry:
            def __init__(self) -> None:
                self._samples = [
                    TelemetrySample(
                        timestamp=0.0,
                        reading=TelemetryReading(
                            energy_joules=100.0,
                            power_watts=200.0,
                            cpu_energy_joules=50.0,
                            cpu_power_watts=75.0,
                        ),
                    ),
                    TelemetrySample(
                        timestamp=9_999_999_999.0,
                        reading=TelemetryReading(
                            energy_joules=160.0,
                            power_watts=220.0,
                            cpu_energy_joules=80.0,
                            cpu_power_watts=85.0,
                        ),
                    ),
                ]

            def readings(self):
                return []

            def window(self, start, end):
                return iter(
                    [
                        TelemetrySample(
                            timestamp=start,
                            reading=self._samples[0].reading,
                        ),
                        TelemetrySample(
                            timestamp=end,
                            reading=self._samples[1].reading,
                        ),
                    ]
                )

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem=f"Q{i}",
                answer=f"A{i}",
                subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
            for i in range(3)
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 3
        dataset.create_task_env.return_value = None

        def make_agent(_recorder: EventRecorder):
            agent = MagicMock()
            agent.run.return_value = AgentRunResult(
                content="ok",
                input_tokens=10,
                output_tokens=5,
            )
            return agent

        runner = AgenticRunner(
            agent=MagicMock(),
            dataset=dataset,
            telemetry_session=FakeTelemetry(),
            config={
                "model": "test-model",
                "telemetry_device_ids": [0, 1],
            },
            run_dir=tmp_path,
            concurrency=2,
            agent_factory=make_agent,
            telemetry_accounting="batch",
        )

        traces = asyncio.run(runner.run())

        assert len(traces) == 3
        assert len(runner.batch_records) == 2
        assert (tmp_path / "batches.jsonl").exists()
        assert all(trace.telemetry_accounting == "batch" for trace in traces)
        assert all(trace.batch_id for trace in traces)
        assert all(trace.query_gpu_energy_joules is None for trace in traces)
        assert all(trace.total_gpu_energy_joules is None for trace in traces)
        assert all(
            trace.batch_prompt_finish_offset_s is not None
            and trace.batch_prompt_finish_offset_s >= 0
            for trace in traces
        )
        assert runner.batch_records[0].query_ids == ["q0000", "q0001"]
        assert runner.batch_records[1].query_ids == ["q0002"]
        assert runner.batch_records[0].batch_gpu_energy_joules is not None
        assert runner.batch_records[0].batch_gpu_energy_joules == 60.0
        assert runner.batch_records[1].batch_gpu_energy_joules == 60.0

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["totals"]["gpu_energy_joules"] == 120.0
        assert summary["averages"]["gpu_energy_per_query_joules"] == 40.0
        assert summary["batch_telemetry"]["batch_count"] == 2
        assert summary["batch_telemetry"]["total_batch_gpu_energy_joules"] == 120.0

    def test_set_task_metadata_called_inside_context(self) -> None:
        """Verify set_task_metadata is called after context enters."""
        call_order = []

        class TrackingEnv:
            def __enter__(self):
                call_order.append("enter")
                return self

            def __exit__(self, *args):
                call_order.append("exit")
                return False

            def run_tests(self):
                call_order.append("run_tests")

        agent = MagicMock()

        def tracked_set_metadata(metadata):
            call_order.append("set_metadata")

        agent.set_task_metadata = tracked_set_metadata
        agent.run.side_effect = lambda *a, **kw: (
            call_order.append("agent_run"),
            AgentRunResult(content="ok"),
        )[1]

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q", answer="A", subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = TrackingEnv()

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            config={"model": "test-model"},
        )
        asyncio.run(runner.run())

        assert call_order == ["enter", "set_metadata", "agent_run", "run_tests", "exit"]

    def test_synthetic_turn_from_agent_result(self) -> None:
        """When EventRecorder has no events but AgentRunResult has tokens,
        a synthetic turn is created to preserve token counts."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(
            content="result from docker",
            input_tokens=500,
            output_tokens=200,
            num_turns=3,
            cost_usd=0.05,
        )

        runner = self._make_runner(agent=agent)
        traces = asyncio.run(runner.run())

        assert len(traces) == 1
        trace = traces[0]
        # Synthetic turn should be created
        assert trace.num_turns == 1
        assert trace.total_input_tokens == 500
        assert trace.total_output_tokens == 200
        assert trace.turns[0].cost_usd == 0.05

    def test_no_synthetic_turn_when_zero_tokens(self) -> None:
        """No synthetic turn is created when AgentRunResult has 0 tokens."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(
            content="no tokens",
            input_tokens=0,
            output_tokens=0,
        )

        runner = self._make_runner(agent=agent)
        traces = asyncio.run(runner.run())

        assert len(traces) == 1
        assert traces[0].num_turns == 0

    def test_query_level_energy_without_turns(self) -> None:
        """Query-level energy is populated from telemetry even with 0 turns."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(content="ok")

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q1", answer="A1", subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        # Create a mock telemetry session with GPU energy readings
        telemetry = MagicMock()
        telemetry.readings.return_value = []
        telemetry.window.return_value = iter([
            TelemetrySample(
                timestamp=1000.0,
                reading=TelemetryReading(
                    energy_joules=100.0,
                    power_watts=200.0,
                    cpu_energy_joules=50.0,
                    cpu_power_watts=80.0,
                ),
            ),
            TelemetrySample(
                timestamp=1001.0,
                reading=TelemetryReading(
                    energy_joules=110.0,
                    power_watts=220.0,
                    cpu_energy_joules=55.0,
                    cpu_power_watts=90.0,
                ),
            ),
        ])

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            telemetry_session=telemetry,
            config={"model": "test-model"},
        )
        traces = asyncio.run(runner.run())

        assert len(traces) == 1
        trace = traces[0]
        assert trace.query_gpu_energy_joules == pytest.approx(10.0)
        assert trace.query_cpu_energy_joules == pytest.approx(5.0)
        assert trace.query_gpu_power_avg_watts == pytest.approx(210.0)
        assert trace.query_cpu_power_avg_watts == pytest.approx(85.0)
        # total_gpu_energy_joules should fall back to query-level
        assert trace.total_gpu_energy_joules == pytest.approx(10.0)

    def test_judge_energy_recorded_separately_from_prompt_energy(self) -> None:
        """Scoring/judge telemetry is not folded into query_gpu_energy_joules."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(
            content="ok",
            input_tokens=10,
            output_tokens=5,
        )

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q1", answer="A1", subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None
        dataset.score.return_value = (True, {"score": 1.0})

        prompt_samples = [
            TelemetrySample(
                timestamp=1000.0,
                reading=TelemetryReading(energy_joules=100.0, power_watts=200.0),
            ),
            TelemetrySample(
                timestamp=1001.0,
                reading=TelemetryReading(energy_joules=112.0, power_watts=220.0),
            ),
        ]
        judge_samples = [
            TelemetrySample(
                timestamp=1002.0,
                reading=TelemetryReading(energy_joules=112.0, power_watts=210.0),
            ),
            TelemetrySample(
                timestamp=1003.0,
                reading=TelemetryReading(energy_joules=115.0, power_watts=230.0),
            ),
        ]

        telemetry = MagicMock()
        telemetry.readings.return_value = []
        telemetry.window.side_effect = [iter(prompt_samples), iter(judge_samples)]

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            telemetry_session=telemetry,
            config={"model": "test-model"},
        )
        traces = asyncio.run(runner.run())

        trace = traces[0]
        assert trace.query_gpu_energy_joules == pytest.approx(12.0)
        assert trace.total_gpu_energy_joules == pytest.approx(12.0)
        assert trace.judge_gpu_energy_joules == pytest.approx(3.0)
        assert trace.total_task_gpu_energy_joules == pytest.approx(15.0)

        metrics = runner.records[0].model_metrics["test-model"].energy_metrics
        assert metrics.per_query_joules == pytest.approx(12.0)
        assert metrics.judge_gpu_joules == pytest.approx(3.0)
        assert metrics.total_task_gpu_joules == pytest.approx(15.0)

    def test_pre_judge_snapshot_survives_scoring_failure(self, tmp_path) -> None:
        """Prompt energy/metadata are saved before judge output is required."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(
            content="candidate answer",
            input_tokens=11,
            output_tokens=7,
            metadata={
                "gdpval_outputs_dir": str(tmp_path / "outputs"),
                "gdpval_submitted_files": ["answer.xlsx"],
            },
        )

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q1",
                answer="A1",
                subject="s",
                dataset_metadata={
                    "dataset_name": "gdpval",
                    "instance_id": "case-1",
                    "rubric_json": "[{\"criterion\":\"ok\",\"score\":1}]",
                },
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None
        dataset.score.side_effect = RuntimeError("judge unavailable")

        class FakeTelemetry:
            def readings(self):
                return []

            def window(self, start, end):
                return iter([
                    TelemetrySample(
                        timestamp=start,
                        reading=TelemetryReading(
                            energy_joules=100.0,
                            power_watts=200.0,
                            cpu_energy_joules=40.0,
                            cpu_power_watts=80.0,
                        ),
                    ),
                    TelemetrySample(
                        timestamp=end,
                        reading=TelemetryReading(
                            energy_joules=112.0,
                            power_watts=220.0,
                            cpu_energy_joules=45.0,
                            cpu_power_watts=90.0,
                        ),
                    ),
                ])

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            telemetry_session=FakeTelemetry(),
            config={"model": "test-model", "telemetry_device_ids": [0]},
            run_dir=tmp_path,
        )

        traces = asyncio.run(runner.run())

        assert traces[0].unscorable_reason == "scoring_error"
        pre_judge_rows = [
            json.loads(line)
            for line in (tmp_path / "pre_judge_traces.jsonl").read_text().splitlines()
        ]
        assert len(pre_judge_rows) == 1
        pre_judge_trace = pre_judge_rows[0]
        assert "candidate answer" in pre_judge_trace["response_text"]
        assert pre_judge_trace["query_gpu_energy_joules"] == pytest.approx(12.0)
        assert pre_judge_trace["query_cpu_energy_joules"] == pytest.approx(5.0)
        assert pre_judge_trace["judge_start_time_s"] is None
        assert pre_judge_trace["score_metadata"]["agent_metadata"]["gdpval_submitted_files"] == ["answer.xlsx"]

        artifact_dir = next((tmp_path / "artifacts").glob("q0000_case-1"))
        pre_meta = json.loads((artifact_dir / "pre_judge_metadata.json").read_text())
        assert pre_meta["snapshot_stage"] == "pre_judge"
        assert pre_meta["judge_status"] == "not_started"
        assert pre_meta["query_gpu_energy_joules"] == pytest.approx(12.0)
        assert pre_meta["dataset_metadata"]["rubric_json"]
        assert "score_metadata" not in pre_meta["dataset_metadata"]
        assert "candidate answer" in (artifact_dir / "pre_judge_response.txt").read_text()

    def test_cost_computation_wired(self) -> None:
        """Cost is computed from pricing tables when trace has no cost but has tokens."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(
            content="result",
            input_tokens=1000,
            output_tokens=500,
        )

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q1", answer="A1", subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            config={"model": "gpt-4o", "provider": "openai"},
        )
        asyncio.run(runner.run())

        record = runner.records[0]
        metrics = record.model_metrics["gpt-4o"]
        # gpt-4o: $2.50/1M input, $10.00/1M output
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert metrics.cost.total_cost_usd == pytest.approx(expected)

    def test_cost_not_computed_for_unknown_provider(self) -> None:
        """Cost remains None when provider/model isn't in pricing tables."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(
            content="result",
            input_tokens=1000,
            output_tokens=500,
        )

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q1", answer="A1", subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            config={"model": "unknown-model", "provider": "unknown"},
        )
        asyncio.run(runner.run())

        record = runner.records[0]
        metrics = record.model_metrics["unknown-model"]
        assert metrics.cost.total_cost_usd is None

    def test_is_resolved_persisted_in_artifacts(self, tmp_path) -> None:
        """is_resolved and test_results from dataset_metadata are saved to metadata.json."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(content="ok")

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q1", answer="A1", subject="s",
                dataset_metadata={
                    "dataset_name": "test",
                    "instance_id": "inst_001",
                    "task_id": "task-001",
                    "occupation": "Analyst",
                    "sector": "Finance",
                    "include_rubric_in_prompt": True,
                    "deliverable_files": ["answer.xlsx"],
                    "is_resolved": True,
                    "test_results": {"passed": 5, "failed": 0},
                },
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            config={"model": "test-model"},
            run_dir=tmp_path,
        )
        asyncio.run(runner.run())

        import json

        meta_path = tmp_path / "artifacts" / "q0000_inst_001" / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["is_resolved"] is True
        assert meta["test_results"] == {"passed": 5, "failed": 0}
        assert meta["task_id"] == "task-001"
        assert meta["occupation"] == "Analyst"
        assert meta["sector"] == "Finance"
        assert meta["include_rubric_in_prompt"] is True
        assert meta["deliverable_files"] == ["answer.xlsx"]


    def test_local_model_cost_is_zero(self) -> None:
        """Cost = 0.0 when client_base_url is localhost (local inference)."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(
            content="result",
            input_tokens=1000,
            output_tokens=500,
        )

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q1", answer="A1", subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            config={
                "model": "glm-4-flash",
                "provider": "",
                "client_base_url": "http://localhost:8000/v1",
            },
        )
        asyncio.run(runner.run())

        record = runner.records[0]
        metrics = record.model_metrics["glm-4-flash"]
        assert metrics.cost.total_cost_usd == 0.0

    def test_tokens_backfilled_from_result_when_turns_have_zero_tokens(self) -> None:
        """When event-based turns exist but have 0 tokens, backfill from AgentRunResult."""
        from ipw.telemetry.events import EventType

        recorder = EventRecorder()
        agent = MagicMock()

        def run_with_zero_token_events(prompt: str, **kwargs) -> AgentRunResult:
            # Simulate OpenHands: fires lm_inference events without token metadata
            recorder.record(EventType.LM_INFERENCE_START)
            recorder.record(EventType.LM_INFERENCE_END)  # no prompt_tokens/completion_tokens
            return AgentRunResult(
                content="answer",
                input_tokens=1200,
                output_tokens=400,
                cost_usd=0.0,
            )

        agent.run.side_effect = run_with_zero_token_events

        runner = self._make_runner(agent=agent)
        runner._event_recorder = recorder

        traces = asyncio.run(runner.run())
        trace = traces[0]

        assert trace.num_turns == 1
        assert trace.total_input_tokens == 1200
        assert trace.total_output_tokens == 400
        assert trace.turns[0].cost_usd == 0.0

    def test_is_resolved_flows_into_query_trace(self) -> None:
        """is_resolved from dataset_metadata appears in the QueryTrace."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(content="done")

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q1", answer="A1", subject="s",
                dataset_metadata={"dataset_name": "test", "is_resolved": True},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            config={"model": "test-model"},
        )
        traces = asyncio.run(runner.run())

        assert traces[0].is_resolved is True

    def test_zero_cost_preserved_for_local_models(self) -> None:
        """cost_usd=0.0 from AgentRunResult is preserved (not treated as falsy)."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(
            content="result",
            input_tokens=500,
            output_tokens=200,
            cost_usd=0.0,
        )

        runner = self._make_runner(agent=agent)
        traces = asyncio.run(runner.run())

        trace = traces[0]
        assert trace.num_turns == 1
        assert trace.turns[0].cost_usd == 0.0

    def test_gdpval_final_check_runs_for_structurally_valid_deliverable(self, tmp_path) -> None:
        """GDPval gets a final checklist pass even when extension/name gates pass."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        deliverable = workspace / "Sample.zip"
        with zipfile.ZipFile(deliverable, "w") as zf:
            zf.writestr("sample.txt", "sample artifact\n")

        agent = MagicMock()
        agent.run.side_effect = [
            AgentRunResult(content="initial", metadata={}),
            AgentRunResult(
                content="checked",
                metadata={"gdpval_submitted_files": [str(deliverable)]},
            ),
        ]

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem=(
                    "Create Sample.zip with finished deliverables.\n\n"
                    "## Evaluation rubric\n\n"
                    "Rubric: check the final archive contents against source facts."
                ),
                answer="",
                subject="audit",
                dataset_metadata={
                    "dataset_name": "GDPval",
                    "gdpval_outputs_dir": str(workspace),
                    "deliverable_files": ["Sample.zip"],
                },
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        runner = self._make_runner(
            agent=agent,
            dataset=dataset,
            config={
                "model": "test-model",
                "gdpval_repair_attempts": 0,
                "gdpval_final_check_attempts": 1,
            },
        )
        traces = asyncio.run(runner.run())

        assert agent.run.call_count == 2
        checklist_prompt = agent.run.call_args_list[1].args[0]
        assert "final deliverable checklist" in checklist_prompt
        assert "Sample.zip" in checklist_prompt
        assert "Reference/input files are not submitted" in checklist_prompt
        assert "one structured verification pass" in checklist_prompt
        assert "Do not re-read the full source" in checklist_prompt
        assert "compatible filename variants" in checklist_prompt
        assert "compatibility artifact" in checklist_prompt
        assert "selected sample rows and full population/reference rows clearly separated" in checklist_prompt
        assert "Original GDPval task prompt" in checklist_prompt
        assert "Rubric: check the final archive contents against source facts" in checklist_prompt
        assert "checked" in traces[0].response_text

    def test_gdpval_final_check_temporarily_caps_agent_turns(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        """Final checklist is bounded and restores the main agent turn budget."""
        monkeypatch.setenv("IPW_GDPVAL_FINAL_CHECK_MAX_TURNS", "7")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        deliverable = workspace / "Sample.zip"
        with zipfile.ZipFile(deliverable, "w") as zf:
            zf.writestr("sample.txt", "sample artifact\n")

        agent = MagicMock()
        agent._max_turns = 180
        seen_turn_budgets: list[int] = []
        seen_prompts: list[str] = []

        def run_agent(prompt: str) -> AgentRunResult:
            seen_prompts.append(prompt)
            seen_turn_budgets.append(agent._max_turns)
            if len(seen_turn_budgets) == 1:
                return AgentRunResult(content="initial", metadata={}, num_turns=176)
            return AgentRunResult(
                content="checked",
                metadata={"gdpval_submitted_files": [str(deliverable)]},
                num_turns=4,
            )

        agent.run.side_effect = run_agent

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Create Sample.zip with finished deliverables.",
                answer="rubric",
                subject="audit",
                dataset_metadata={
                    "dataset_name": "GDPval",
                    "gdpval_outputs_dir": str(workspace),
                    "deliverable_files": ["Sample.zip"],
                },
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        runner = self._make_runner(
            agent=agent,
            dataset=dataset,
            config={
                "model": "test-model",
                "gdpval_repair_attempts": 0,
                "gdpval_final_check_attempts": 1,
            },
        )
        asyncio.run(runner.run())

        assert seen_turn_budgets == [180, 4]
        assert "at most 4 turns remaining in the overall 180-turn task budget" in seen_prompts[1]

    def test_gdpval_autoruns_model_written_repair_script(self, tmp_path) -> None:
        """A repair script written but not run by the model is executed once."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        deliverable = workspace / "Sample.zip"
        script = workspace / "ipw_repair_deliverables.py"

        agent = MagicMock()

        def run_agent(prompt: str) -> AgentRunResult:
            if "Strict repair procedure" in prompt:
                script.write_text(
                    "import zipfile\n"
                    "with zipfile.ZipFile('Sample.zip', 'w') as zf:\n"
                    "    zf.writestr('sample.txt', 'sample artifact\\n')\n",
                    encoding="utf-8",
                )
                return AgentRunResult(
                    content="I wrote ipw_repair_deliverables.py.",
                    metadata={"gdpval_outputs_dir": str(workspace)},
                    num_turns=1,
                )
            return AgentRunResult(
                content="initial answer without artifact",
                metadata={"gdpval_outputs_dir": str(workspace)},
                num_turns=1,
            )

        agent.run.side_effect = run_agent

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Create Sample.zip with finished deliverables.",
                answer="rubric",
                subject="audit",
                dataset_metadata={
                    "dataset_name": "GDPval",
                    "gdpval_outputs_dir": str(workspace),
                    "deliverable_files": ["Sample.zip"],
                },
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        def score(record: DatasetRecord, response: str):
            validation = dict(record.dataset_metadata.get("deliverable_validation") or {})
            return True, {
                "score": 1.0,
                "achieved_points": 1.0,
                "max_points": 1.0,
                "deliverable_validation": validation,
            }

        dataset.score.side_effect = score

        runner = self._make_runner(
            agent=agent,
            dataset=dataset,
            config={
                "model": "test-model",
                "gdpval_repair_attempts": 1,
                "gdpval_final_check_attempts": 0,
                "gdpval_autorun_repair_scripts": True,
                "gdpval_repair_script_timeout_seconds": 5,
            },
        )
        traces = asyncio.run(runner.run())

        assert deliverable.exists()
        assert agent.run.call_count == 2
        assert dataset.score.call_count == 1
        assert traces[0].score_metadata["score"] == 1.0
        validation = traces[0].score_metadata["deliverable_validation"]
        assert validation["status"] == "ok"
        assert validation["primary_files"] == [str(deliverable)]
        assert records[0].dataset_metadata["gdpval_autorun_repair_scripts"] == [
            "ipw_repair_deliverables.py"
        ]

    def test_gdpval_settle_waits_for_late_deliverable_before_scoring(self, tmp_path) -> None:
        """Late workspace files are included in GDPval scoring validation."""
        import threading
        import time as _time

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        deliverable = workspace / "Sample.zip"

        records = [
            DatasetRecord(
                problem="Create Sample.zip with finished deliverables.",
                answer="rubric",
                subject="audit",
                dataset_metadata={
                    "dataset_name": "GDPval",
                    "gdpval_outputs_dir": str(workspace),
                    "deliverable_files": ["Sample.zip"],
                },
            )
        ]

        agent = MagicMock()

        def run_agent(prompt: str) -> AgentRunResult:
            def write_late_file() -> None:
                _time.sleep(0.15)
                with zipfile.ZipFile(deliverable, "w") as zf:
                    zf.writestr("sample.txt", "sample artifact\n")

            threading.Thread(target=write_late_file, daemon=True).start()
            return AgentRunResult(
                content="finished",
                metadata={"gdpval_outputs_dir": str(workspace)},
            )

        agent.run.side_effect = run_agent

        dataset = MagicMock()
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        def score(record: DatasetRecord, response: str):
            validation = dict(record.dataset_metadata.get("deliverable_validation") or {})
            return True, {
                "score": 1.0,
                "achieved_points": 1.0,
                "max_points": 1.0,
                "deliverable_validation": validation,
            }

        dataset.score.side_effect = score

        runner = self._make_runner(
            agent=agent,
            dataset=dataset,
            config={
                "model": "test-model",
                "gdpval_repair_attempts": 0,
                "gdpval_final_check_attempts": 0,
                "gdpval_file_settle_seconds": 2,
                "gdpval_file_settle_poll_seconds": 0.05,
            },
        )
        traces = asyncio.run(runner.run())

        validation = traces[0].score_metadata["deliverable_validation"]
        assert validation["status"] == "ok"
        assert validation["primary_files"] == [str(deliverable)]
        assert traces[0].completed is True

    def test_gdpval_stirrup_skips_ipw_repair_and_checklist(self, tmp_path) -> None:
        """AA-style Stirrup runs keep a single model-controlled finish."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        deliverable = workspace / "Sample.xlsx"
        deliverable.write_bytes(b"placeholder workbook bytes")

        agent = MagicMock()
        agent.run.return_value = AgentRunResult(
            content="submitted once",
            metadata={
                "gdpval_outputs_dir": str(workspace),
                "gdpval_submitted_files": [str(deliverable)],
                "stirrup_finished": True,
                "stirrup_finish_params": {
                    "reason": "done",
                    "paths": ["/home/user/Sample.xlsx"],
                },
            },
        )

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Create Sample.xlsx with finished deliverables.",
                answer="",
                subject="audit",
                dataset_metadata={
                    "dataset_name": "GDPval Stirrup Single-Submission",
                    "workload_type": "gdpval-stirrup",
                    "gdpval_outputs_dir": str(workspace),
                    "deliverable_files": ["Sample.xlsx"],
                },
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        runner = self._make_runner(
            agent=agent,
            dataset=dataset,
            config={
                "model": "test-model",
                "gdpval_repair_attempts": 1,
                "gdpval_final_check_attempts": 1,
            },
        )
        traces = asyncio.run(runner.run())

        assert agent.run.call_count == 1
        assert traces[0].response_text == "submitted once"
        assert traces[0].score_metadata["agent_metadata"]["stirrup_finished"] is True

    def test_gdpval_repair_prompt_includes_original_task_context(self, tmp_path) -> None:
        """Repair runs are fresh agent calls, so they must restate the task."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        original_prompt = (
            "## Task\n\n"
            "Create Sample.xlsx from /home/user/inputs/source.xlsx with a summary tab."
        )

        agent = MagicMock()
        agent.run.side_effect = [
            AgentRunResult(
                content="submitted placeholder",
                metadata={
                    "gdpval_outputs_dir": str(workspace),
                    "gdpval_submitted_files": [],
                },
            ),
            AgentRunResult(
                content="cannot repair",
                metadata={
                    "status": "failed",
                    "run_status": "abandoned",
                    "failure_type": "agent_abandoned",
                    "reason": "agent_abandoned",
                },
            ),
        ]

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem=original_prompt,
                answer="",
                subject="audit",
                dataset_metadata={
                    "dataset_name": "GDPval",
                    "gdpval_outputs_dir": str(workspace),
                    "deliverable_files": ["Sample.xlsx"],
                },
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        runner = self._make_runner(
            agent=agent,
            dataset=dataset,
            config={
                "model": "test-model",
                "gdpval_repair_attempts": 1,
                "gdpval_final_check_attempts": 0,
            },
        )
        traces = asyncio.run(runner.run())

        assert agent.run.call_count == 2
        repair_prompt = agent.run.call_args_list[1].args[0]
        assert "Original GDPval task prompt" in repair_prompt
        assert "ipw_repair_deliverables.py" in repair_prompt
        assert "case/spacing variants or clearly equivalent descriptive names" in repair_prompt
        assert "Do not print or read back a long script" in repair_prompt
        assert "Create Sample.xlsx" in repair_prompt
        assert "/home/user/inputs/source.xlsx" in repair_prompt
        assert "Sample.xlsx" in repair_prompt
        assert traces[0].unscorable_reason == "agent_abandoned"
        assert traces[0].score_metadata["failure_type"] == "agent_abandoned"

    def test_power_metrics_populated_from_telemetry(self) -> None:
        """Power metrics in ProfilingRecord are populated from trace data."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(
            content="result",
            input_tokens=500,
            output_tokens=200,
        )

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q1", answer="A1", subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        telemetry = MagicMock()
        telemetry.readings.return_value = []
        telemetry.window.return_value = iter([
            TelemetrySample(
                timestamp=1000.0,
                reading=TelemetryReading(
                    energy_joules=100.0,
                    power_watts=200.0,
                    cpu_energy_joules=50.0,
                    cpu_power_watts=80.0,
                ),
            ),
            TelemetrySample(
                timestamp=1001.0,
                reading=TelemetryReading(
                    energy_joules=110.0,
                    power_watts=220.0,
                    cpu_energy_joules=55.0,
                    cpu_power_watts=90.0,
                ),
            ),
        ])

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            telemetry_session=telemetry,
            config={"model": "test-model"},
        )
        asyncio.run(runner.run())

        record = runner.records[0]
        metrics = record.model_metrics["test-model"]
        # Power metrics should be populated
        assert metrics.power_metrics.gpu.per_query_watts.avg == pytest.approx(210.0)
        assert metrics.power_metrics.cpu.per_query_watts.avg == pytest.approx(85.0)

    def test_energy_per_token_populated(self) -> None:
        """Energy-per-token fields in EnergyMetrics are populated."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(
            content="result",
            input_tokens=500,
            output_tokens=200,
        )

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q1", answer="A1", subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        telemetry = MagicMock()
        telemetry.readings.return_value = []
        telemetry.window.return_value = iter([
            TelemetrySample(
                timestamp=1000.0,
                reading=TelemetryReading(energy_joules=100.0, power_watts=200.0),
            ),
            TelemetrySample(
                timestamp=1001.0,
                reading=TelemetryReading(energy_joules=120.0, power_watts=220.0),
            ),
        ])

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            telemetry_session=telemetry,
            config={"model": "test-model"},
        )
        asyncio.run(runner.run())

        record = runner.records[0]
        metrics = record.model_metrics["test-model"]
        # GPU energy = 20 J, output = 200, total = 700
        assert metrics.energy_metrics.energy_per_output_token_joules == pytest.approx(20.0 / 200.0)
        assert metrics.energy_metrics.energy_per_total_token_joules == pytest.approx(20.0 / 700.0)

    def test_efficiency_metrics_populated(self) -> None:
        """DerivedEfficiencyMetrics.throughput_per_watt is populated."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(
            content="result",
            input_tokens=500,
            output_tokens=200,
        )

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q1", answer="A1", subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        telemetry = MagicMock()
        telemetry.readings.return_value = []
        telemetry.window.return_value = iter([
            TelemetrySample(
                timestamp=1000.0,
                reading=TelemetryReading(power_watts=200.0),
            ),
            TelemetrySample(
                timestamp=1001.0,
                reading=TelemetryReading(power_watts=200.0),
            ),
        ])

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            telemetry_session=telemetry,
            config={"model": "test-model"},
        )
        asyncio.run(runner.run())

        record = runner.records[0]
        metrics = record.model_metrics["test-model"]
        assert metrics.efficiency.throughput_per_watt is not None
        assert metrics.efficiency.throughput_per_watt > 0

    def test_query_timeout_creates_timed_out_trace(self) -> None:
        """When query_timeout fires, trace has timed_out=True and completed=False.

        Uses concurrency > 1 so the blocking sleep runs in a thread pool,
        allowing asyncio.wait_for to actually cancel it.
        """
        import time as _time

        def make_slow_agent():
            a = MagicMock()

            def slow_run(prompt: str, **kwargs) -> AgentRunResult:
                _time.sleep(5.0)
                return AgentRunResult(content="too slow")

            a.run.side_effect = slow_run
            return a

        agent = make_slow_agent()

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q1", answer="A1", subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            config={"model": "test-model"},
            query_timeout=0.2,
            concurrency=2,
            agent_factory=make_slow_agent,
        )
        traces = asyncio.run(runner.run())

        assert len(traces) == 1
        assert traces[0].completed is False
        assert traces[0].timed_out is True
        assert "timed out" in traces[0].response_text.lower()

    def test_local_model_cost_127(self) -> None:
        """Cost = 0.0 when client_base_url is 127.0.0.1."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(
            content="result",
            input_tokens=500,
            output_tokens=200,
        )

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q1", answer="A1", subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            config={
                "model": "local-model",
                "provider": "",
                "client_base_url": "http://127.0.0.1:8000/v1",
            },
        )
        asyncio.run(runner.run())

        record = runner.records[0]
        metrics = record.model_metrics["local-model"]
        assert metrics.cost.total_cost_usd == 0.0

    def test_mbu_extracted_from_telemetry(self) -> None:
        """MBU avg and max are extracted from telemetry readings."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(content="ok")

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q1", answer="A1", subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        telemetry = MagicMock()
        telemetry.readings.return_value = []
        telemetry.window.return_value = iter([
            TelemetrySample(
                timestamp=1000.0,
                reading=TelemetryReading(
                    power_watts=200.0,
                    gpu_memory_bandwidth_utilization_pct=40.0,
                ),
            ),
            TelemetrySample(
                timestamp=1000.5,
                reading=TelemetryReading(
                    power_watts=210.0,
                    gpu_memory_bandwidth_utilization_pct=60.0,
                ),
            ),
            TelemetrySample(
                timestamp=1001.0,
                reading=TelemetryReading(
                    power_watts=220.0,
                    gpu_memory_bandwidth_utilization_pct=80.0,
                ),
            ),
        ])

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            telemetry_session=telemetry,
            config={"model": "test-model"},
        )
        traces = asyncio.run(runner.run())

        assert len(traces) == 1
        trace = traces[0]
        assert trace.query_mbu_avg_pct == pytest.approx(60.0)
        assert trace.query_mbu_max_pct == pytest.approx(80.0)

    def test_mbu_filters_negative_values(self) -> None:
        """MBU extraction filters out -1 (unavailable) values."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(content="ok")

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q1", answer="A1", subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        telemetry = MagicMock()
        telemetry.readings.return_value = []
        telemetry.window.return_value = iter([
            TelemetrySample(
                timestamp=1000.0,
                reading=TelemetryReading(
                    power_watts=200.0,
                    gpu_memory_bandwidth_utilization_pct=-1.0,
                ),
            ),
            TelemetrySample(
                timestamp=1000.5,
                reading=TelemetryReading(
                    power_watts=210.0,
                    gpu_memory_bandwidth_utilization_pct=50.0,
                ),
            ),
            TelemetrySample(
                timestamp=1001.0,
                reading=TelemetryReading(
                    power_watts=220.0,
                    gpu_memory_bandwidth_utilization_pct=70.0,
                ),
            ),
        ])

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            telemetry_session=telemetry,
            config={"model": "test-model"},
        )
        traces = asyncio.run(runner.run())

        trace = traces[0]
        # -1 should be filtered out; avg of (50, 70) = 60, max = 70
        assert trace.query_mbu_avg_pct == pytest.approx(60.0)
        assert trace.query_mbu_max_pct == pytest.approx(70.0)

    def test_mbu_none_when_no_telemetry(self) -> None:
        """MBU fields are None when there is no telemetry session."""
        runner = self._make_runner()
        traces = asyncio.run(runner.run())

        assert len(traces) == 1
        assert traces[0].query_mbu_avg_pct is None
        assert traces[0].query_mbu_max_pct is None

    def test_mbu_none_when_all_values_unavailable(self) -> None:
        """MBU fields are None when all readings have -1 (unavailable)."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(content="ok")

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q1", answer="A1", subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        telemetry = MagicMock()
        telemetry.readings.return_value = []
        telemetry.window.return_value = iter([
            TelemetrySample(
                timestamp=1000.0,
                reading=TelemetryReading(
                    power_watts=200.0,
                    gpu_memory_bandwidth_utilization_pct=-1.0,
                ),
            ),
            TelemetrySample(
                timestamp=1001.0,
                reading=TelemetryReading(
                    power_watts=220.0,
                    gpu_memory_bandwidth_utilization_pct=-1.0,
                ),
            ),
        ])

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            telemetry_session=telemetry,
            config={"model": "test-model"},
        )
        traces = asyncio.run(runner.run())

        trace = traces[0]
        assert trace.query_mbu_avg_pct is None
        assert trace.query_mbu_max_pct is None

    def test_mbu_none_when_field_missing_from_readings(self) -> None:
        """MBU fields are None when readings don't have the MBU field at all."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(content="ok")

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q1", answer="A1", subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        telemetry = MagicMock()
        telemetry.readings.return_value = []
        telemetry.window.return_value = iter([
            TelemetrySample(
                timestamp=1000.0,
                reading=TelemetryReading(power_watts=200.0),
            ),
            TelemetrySample(
                timestamp=1001.0,
                reading=TelemetryReading(power_watts=220.0),
            ),
        ])

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            telemetry_session=telemetry,
            config={"model": "test-model"},
        )
        traces = asyncio.run(runner.run())

        trace = traces[0]
        assert trace.query_mbu_avg_pct is None
        assert trace.query_mbu_max_pct is None

    def test_is_resolved_populated_when_dataset_has_score(self) -> None:
        """When a dataset provides a score() method, AgenticRunner calls it
        and the resulting is_resolved value appears on the QueryTrace."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(content="42")

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="What is 6*7?",
                answer="42",
                subject="math",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        # create_task_env returns None so the elif-score branch is taken
        dataset.create_task_env.return_value = None
        # score() returns (True, {}) indicating a correct answer
        dataset.score.return_value = (True, {})

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            config={"model": "test-model"},
        )
        traces = asyncio.run(runner.run())

        assert len(traces) == 1
        assert traces[0].is_resolved is True
        dataset.score.assert_called_once()

    def test_is_resolved_false_when_score_returns_false(self) -> None:
        """When dataset.score() returns False, is_resolved is False on the trace."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(content="wrong answer")

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="What is 6*7?",
                answer="42",
                subject="math",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None
        dataset.score.return_value = (False, {})

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            config={"model": "test-model"},
        )
        traces = asyncio.run(runner.run())

        assert len(traces) == 1
        assert traces[0].is_resolved is False

    def test_is_resolved_none_when_no_score_method(self) -> None:
        """When dataset has no score() method, is_resolved remains None."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(content="answer")

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q", answer="A", subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None
        # Remove the auto-generated score attribute so hasattr returns False
        del dataset.score

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            config={"model": "test-model"},
        )
        traces = asyncio.run(runner.run())

        assert len(traces) == 1
        assert traces[0].is_resolved is None

    def test_mbu_zero_value_included(self) -> None:
        """MBU value of 0.0 is valid and should be included (not filtered)."""
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(content="ok")

        dataset = MagicMock()
        records = [
            DatasetRecord(
                problem="Q1", answer="A1", subject="s",
                dataset_metadata={"dataset_name": "test"},
            )
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1
        dataset.create_task_env.return_value = None

        telemetry = MagicMock()
        telemetry.readings.return_value = []
        telemetry.window.return_value = iter([
            TelemetrySample(
                timestamp=1000.0,
                reading=TelemetryReading(
                    power_watts=200.0,
                    gpu_memory_bandwidth_utilization_pct=0.0,
                ),
            ),
            TelemetrySample(
                timestamp=1001.0,
                reading=TelemetryReading(
                    power_watts=220.0,
                    gpu_memory_bandwidth_utilization_pct=100.0,
                ),
            ),
        ])

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            telemetry_session=telemetry,
            config={"model": "test-model"},
        )
        traces = asyncio.run(runner.run())

        trace = traces[0]
        # 0.0 is valid (>= 0); avg of (0, 100) = 50, max = 100
        assert trace.query_mbu_avg_pct == pytest.approx(50.0)
        assert trace.query_mbu_max_pct == pytest.approx(100.0)


class TestEnergyHelpers:
    """Test _compute_energy_delta and _compute_power_avg helpers."""

    def test_compute_energy_delta(self) -> None:
        readings = [
            TelemetrySample(
                timestamp=1.0,
                reading=TelemetryReading(energy_joules=100.0),
            ),
            TelemetrySample(
                timestamp=2.0,
                reading=TelemetryReading(energy_joules=115.0),
            ),
        ]
        assert _compute_energy_delta(readings, "energy_joules") == pytest.approx(15.0)

    def test_compute_energy_delta_interpolates_window_boundaries(self) -> None:
        readings = [
            TelemetrySample(
                timestamp=1.0,
                reading=TelemetryReading(energy_joules=100.0),
            ),
            TelemetrySample(
                timestamp=2.0,
                reading=TelemetryReading(energy_joules=120.0),
            ),
            TelemetrySample(
                timestamp=3.0,
                reading=TelemetryReading(energy_joules=150.0),
            ),
        ]
        assert _compute_energy_delta(
            readings,
            "energy_joules",
            start_time=1.5,
            end_time=2.5,
        ) == pytest.approx(25.0)

    def test_compute_energy_delta_none_with_single_reading(self) -> None:
        readings = [
            TelemetrySample(
                timestamp=1.0,
                reading=TelemetryReading(energy_joules=100.0),
            ),
        ]
        assert _compute_energy_delta(readings, "energy_joules") is None

    def test_compute_energy_delta_none_for_missing_field(self) -> None:
        readings = [
            TelemetrySample(
                timestamp=1.0,
                reading=TelemetryReading(),
            ),
            TelemetrySample(
                timestamp=2.0,
                reading=TelemetryReading(),
            ),
        ]
        assert _compute_energy_delta(readings, "energy_joules") is None

    def test_compute_power_avg(self) -> None:
        readings = [
            TelemetrySample(
                timestamp=1.0,
                reading=TelemetryReading(power_watts=200.0),
            ),
            TelemetrySample(
                timestamp=2.0,
                reading=TelemetryReading(power_watts=300.0),
            ),
        ]
        assert _compute_power_avg(readings, "power_watts") == pytest.approx(250.0)

    def test_compute_power_avg_none_for_empty(self) -> None:
        assert _compute_power_avg([], "power_watts") is None
