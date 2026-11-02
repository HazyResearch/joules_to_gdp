"""Tests for telemetry session management."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from ipw.core.types import TelemetryReading
from ipw.execution.telemetry_accounting import build_batch_record
from ipw.execution.telemetry_session import TelemetrySample, TelemetrySession


class TestTelemetrySession:
    """Test TelemetrySession context manager."""

    def test_sampler_failure_aborts_session(self) -> None:
        collector = Mock()
        collector.start.return_value = nullcontext()
        collector.stream_readings.side_effect = RuntimeError("meter disconnected")
        with pytest.raises(RuntimeError, match="Telemetry sampler stopped unexpectedly"):
            with TelemetrySession(collector) as session:
                assert session._stop_event.wait(1.0)

    def test_missing_post_window_sample_is_an_error(self, tmp_path) -> None:
        session = TelemetrySession(
            Mock(), sample_output_path=tmp_path / "telemetry_samples.jsonl"
        )
        assert session.sample_recorder is not None
        with pytest.raises(RuntimeError, match="post-window reading"):
            session.sample_recorder.wait_until_after(10.0, timeout_s=0.01)

    def test_run_waits_for_first_meter_sample(self, tmp_path) -> None:
        collector = Mock()
        collector.start.return_value = nullcontext()

        def delayed_readings():
            time.sleep(0.08)
            yield TelemetryReading(timestamp_nanos=int(time.time() * 1_000_000_000))

        collector.stream_readings.return_value = delayed_readings()
        started = time.monotonic()
        with TelemetrySession(
            collector, sample_output_path=tmp_path / "telemetry_samples.jsonl"
        ) as session:
            assert session.sample_recorder is not None
            assert session.sample_recorder.sample_count == 1
        assert time.monotonic() - started >= 0.06

    def test_streams_every_sample_before_buffer_trimming(self, tmp_path) -> None:
        collector = Mock()
        collector.stream_readings.return_value = iter(
            [
                TelemetryReading(
                    timestamp_nanos=10_000_000_000,
                    cpu_energy_joules=100.0,
                    dram_energy_joules=20.0,
                ),
                TelemetryReading(
                    timestamp_nanos=12_000_000_000,
                    cpu_energy_joules=140.0,
                    dram_energy_joules=26.0,
                ),
            ]
        )
        path = tmp_path / "telemetry_samples.jsonl"
        session = TelemetrySession(
            collector, buffer_seconds=1.0, max_samples=1, sample_output_path=path
        )
        assert session.sample_recorder is not None
        session.sample_recorder.open()
        session._run()
        session.sample_recorder.close()

        assert len(list(session.readings())) == 1
        lines = path.read_bytes().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["cpu_energy_joules"] == 100.0
        assert json.loads(lines[1])["dram_energy_joules"] == 26.0
        manifest = session.sample_recorder.manifest()
        assert manifest["sample_count"] == 2
        assert manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

        # The bounded live buffer has lost the start sample, but the raw meter
        # stream still supplies the complete interval for batch accounting.
        partial = build_batch_record(
            batch_id="partial", batch_index=0, query_ids=["q1"],
            prompt_start_time=10.0, prompt_end_time=12.0,
            readings=list(session.readings()),
        )
        assert partial.batch_cpu_energy_joules is None
        assert partial.batch_dram_energy_joules is None
        complete = build_batch_record(
            batch_id="complete", batch_index=0, query_ids=["q1"],
            prompt_start_time=10.0, prompt_end_time=12.0,
            readings=session.sample_recorder.read_window(10.0, 12.0),
        )
        assert complete.batch_cpu_energy_joules == pytest.approx(40.0)
        assert complete.batch_dram_energy_joules == pytest.approx(6.0)
        assert complete.energy_window_coverage == {"gpu": False, "cpu": True, "dram": True}

    def test_full_energy_batch_rejects_paused_or_incomplete_meter(self, monkeypatch) -> None:
        monkeypatch.setenv("IPW_REQUIRE_TOTAL_ENERGY", "1")
        monkeypatch.setenv("IPW_ENERGY_MONITOR_INTERVAL_MS", "200")

        def sample(index: int, *, missing_dram: bool = False) -> TelemetrySample:
            return TelemetrySample(
                timestamp=10.0 + index * 0.2,
                reading=TelemetryReading(
                    energy_joules=float(index * 10),
                    power_watts=50.0,
                    cpu_energy_joules=float(index * 4),
                    cpu_power_watts=20.0,
                    dram_energy_joules=None if missing_dram else float(index),
                    dram_power_watts=None if missing_dram else 5.0,
                ),
            )

        def batch(readings: list[TelemetrySample]):
            return build_batch_record(
                batch_id="measured", batch_index=0, query_ids=["q1"],
                prompt_start_time=10.1, prompt_end_time=11.5, readings=readings,
            )

        complete = batch([sample(index) for index in range(9)])
        assert complete.energy_window_coverage == {"gpu": True, "cpu": True, "dram": True}
        assert complete.batch_dram_energy_joules == pytest.approx(7.0)

        paused = batch([sample(index) for index in (0, 1, 7, 8)])
        assert paused.energy_window_coverage == {"gpu": False, "cpu": False, "dram": False}
        assert paused.batch_gpu_energy_joules is None
        assert paused.batch_cpu_energy_joules is None
        assert paused.batch_dram_energy_joules is None

        missing_component = batch([
            sample(index, missing_dram=index == 4) for index in range(9)
        ])
        assert missing_component.energy_window_coverage == {
            "gpu": True, "cpu": True, "dram": False,
        }
        assert missing_component.batch_dram_energy_joules is None

    def test_full_energy_batch_requires_meter_interval(self, monkeypatch) -> None:
        monkeypatch.setenv("IPW_REQUIRE_TOTAL_ENERGY", "1")
        monkeypatch.delenv("IPW_ENERGY_MONITOR_INTERVAL_MS", raising=False)
        with pytest.raises(ValueError, match="positive meter interval"):
            build_batch_record(
                batch_id="measured", batch_index=0, query_ids=["q1"],
                prompt_start_time=10.0, prompt_end_time=11.0,
            )

    def test_raw_window_reader_advances_between_batches(self, tmp_path) -> None:
        collector = Mock()
        collector.stream_readings.return_value = iter(
            TelemetryReading(timestamp_nanos=int(second * 1e9), cpu_energy_joules=joules)
            for second, joules in ((10, 10), (12, 12), (14, 14), (16, 16))
        )
        session = TelemetrySession(
            collector, buffer_seconds=1.0, max_samples=1,
            sample_output_path=tmp_path / "telemetry_samples.jsonl",
        )
        assert session.sample_recorder is not None
        session.sample_recorder.open()
        session._run()
        session.sample_recorder.close()

        first = session.sample_recorder.read_window(10.0, 12.0)
        second = session.sample_recorder.read_window(14.0, 16.0)
        assert [sample.timestamp for sample in first] == [10.0, 12.0, 14.0]
        assert [sample.timestamp for sample in second] == [12.0, 14.0, 16.0]

    def test_initializes_with_collector(self) -> None:
        collector = Mock()
        session = TelemetrySession(collector)
        assert session._collector is collector

    def test_context_manager_starts_collector(self) -> None:
        collector = Mock()
        collector_ctx = Mock()
        collector.start.return_value = collector_ctx
        collector_ctx.__enter__ = Mock(return_value=collector_ctx)
        collector_ctx.__exit__ = Mock(return_value=None)

        # Create an empty iterator for stream_readings
        collector.stream_readings.return_value = iter([])

        with TelemetrySession(collector) as session:
            collector.start.assert_called_once()
            assert session is not None

    def test_context_manager_stops_thread(self) -> None:
        collector = Mock()
        collector_ctx = Mock()
        collector.start.return_value = collector_ctx
        collector_ctx.__enter__ = Mock(return_value=collector_ctx)
        collector_ctx.__exit__ = Mock(return_value=None)
        collector.stream_readings.return_value = iter([])

        session = TelemetrySession(collector)
        with session:
            pass

        # Thread should be stopped
        assert session._stop_event.is_set()

    def test_window_filters_by_time(self) -> None:
        collector = Mock()
        session = TelemetrySession(collector)

        # Manually add samples
        session._samples.append(
            TelemetrySample(timestamp=1.0, reading=TelemetryReading())
        )
        session._samples.append(
            TelemetrySample(timestamp=2.0, reading=TelemetryReading())
        )
        session._samples.append(
            TelemetrySample(timestamp=3.0, reading=TelemetryReading())
        )
        session._samples.append(
            TelemetrySample(timestamp=4.0, reading=TelemetryReading())
        )

        windowed = list(session.window(1.5, 3.5))
        assert len(windowed) == 2
        assert windowed[0].timestamp == 2.0
        assert windowed[1].timestamp == 3.0

    def test_window_includes_boundaries(self) -> None:
        collector = Mock()
        session = TelemetrySession(collector)

        session._samples.append(
            TelemetrySample(timestamp=1.0, reading=TelemetryReading())
        )
        session._samples.append(
            TelemetrySample(timestamp=2.0, reading=TelemetryReading())
        )
        session._samples.append(
            TelemetrySample(timestamp=3.0, reading=TelemetryReading())
        )

        windowed = list(session.window(1.0, 3.0))
        assert len(windowed) == 3

    def test_window_returns_empty_when_no_overlap(self) -> None:
        collector = Mock()
        session = TelemetrySession(collector)

        session._samples.append(
            TelemetrySample(timestamp=1.0, reading=TelemetryReading())
        )
        session._samples.append(
            TelemetrySample(timestamp=2.0, reading=TelemetryReading())
        )

        windowed = list(session.window(5.0, 10.0))
        assert len(windowed) == 0

    def test_readings_returns_all_samples(self) -> None:
        collector = Mock()
        session = TelemetrySession(collector)

        session._samples.append(
            TelemetrySample(timestamp=1.0, reading=TelemetryReading())
        )
        session._samples.append(
            TelemetrySample(timestamp=2.0, reading=TelemetryReading())
        )

        readings = list(session.readings())
        assert len(readings) == 2

    def test_trim_removes_old_samples(self) -> None:
        collector = Mock()
        session = TelemetrySession(collector, buffer_seconds=5.0)

        session._samples.append(
            TelemetrySample(timestamp=1.0, reading=TelemetryReading())
        )
        session._samples.append(
            TelemetrySample(timestamp=2.0, reading=TelemetryReading())
        )
        session._samples.append(
            TelemetrySample(timestamp=10.0, reading=TelemetryReading())
        )

        session._trim(10.0)

        # Only samples within buffer_seconds (5.0) should remain
        # cutoff = 10.0 - 5.0 = 5.0
        # So samples at timestamp >= 5.0 remain
        # That means only the sample at 10.0 should remain
        assert len(session._samples) == 1
        assert session._samples[0].timestamp == 10.0

    def test_respects_max_samples(self) -> None:
        collector = Mock()
        session = TelemetrySession(collector, max_samples=3)

        for i in range(5):
            session._samples.append(
                TelemetrySample(timestamp=float(i), reading=TelemetryReading())
            )

        # deque with maxlen should keep only last 3
        assert len(session._samples) == 3
        assert session._samples[0].timestamp == 2.0
        assert session._samples[-1].timestamp == 4.0

    def test_integration_with_real_collector(self) -> None:
        """Integration test with actual streaming (if available)."""
        collector = Mock()
        collector_ctx = Mock()
        collector.start.return_value = collector_ctx
        collector_ctx.__enter__ = Mock(return_value=collector_ctx)
        collector_ctx.__exit__ = Mock(return_value=None)

        # Simulate a few readings
        readings = [
            TelemetryReading(energy_joules=100.0),
            TelemetryReading(energy_joules=150.0),
            TelemetryReading(energy_joules=200.0),
        ]

        def reading_generator():
            for r in readings:
                yield r
                time.sleep(0.01)  # Small delay

        collector.stream_readings.return_value = reading_generator()

        with TelemetrySession(collector, buffer_seconds=10.0) as session:
            # Give thread time to collect samples
            time.sleep(0.1)

        # Should have collected some samples
        assert len(session._samples) > 0
