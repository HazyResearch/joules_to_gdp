"""Tests for telemetry/collector.py — EnergyMonitorCollector._convert."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ipw.telemetry.collector import EnergyMonitorCollector


def test_proto_cpu_dram_fields_reach_batch_summary(tmp_path):
    """Rust wire fields survive protobuf, collector, accounting, and export."""
    from ipw.execution.exporters import export_summary_json
    from ipw.execution.telemetry_accounting import build_batch_record
    from ipw.execution.telemetry_session import TelemetrySample
    from ipw.execution.trace import QueryTrace
    from ipw.telemetry.proto import get_stub_bundle

    message_cls = get_stub_bundle().TelemetryReadingCls
    collector = EnergyMonitorCollector.__new__(EnergyMonitorCollector)
    samples = []
    for timestamp, cpu_joules, dram_joules in ((10.0, 100.0, 20.0), (12.0, 140.0, 26.0)):
        encoded = message_cls(
            energy_joules=200.0 + timestamp,
            cpu_power_watts=20.0,
            cpu_energy_joules=cpu_joules,
            dram_power_watts=3.0,
            dram_energy_joules=dram_joules,
        ).SerializeToString()
        reading = collector._convert(message_cls.FromString(encoded))
        samples.append(TelemetrySample(timestamp=timestamp, reading=reading))

    batch = build_batch_record(
        batch_id="batch_0000",
        batch_index=0,
        query_ids=["q1", "q2"],
        prompt_start_time=10.0,
        prompt_end_time=12.0,
        readings=samples,
    )
    assert batch.batch_cpu_energy_joules == pytest.approx(40.0)
    assert batch.batch_dram_energy_joules == pytest.approx(6.0)
    assert batch.batch_dram_power_avg_watts == pytest.approx(3.0)

    traces = [
        QueryTrace(
            query_id=query_id,
            workload_type="agentic",
            completed=True,
            telemetry_accounting="batch",
        )
        for query_id in ("q1", "q2")
    ]
    output = tmp_path / "summary.json"
    export_summary_json(traces, {"telemetry_accounting": "batch"}, output, batch_records=[batch])
    summary = json.loads(output.read_text())
    assert summary["totals"]["cpu_energy_joules"] == pytest.approx(40.0)
    assert summary["totals"]["dram_energy_joules"] == pytest.approx(6.0)
    assert summary["batch_telemetry"]["amortized_dram_energy_per_query_joules"] == pytest.approx(3.0)


def test_collection_interval_is_forwarded_to_monitor(monkeypatch):
    calls = []

    @contextmanager
    def fake_ensure_monitor(*args, **kwargs):
        calls.append((args, kwargs))
        yield

    monkeypatch.setattr("ipw.telemetry.collector.ensure_monitor", fake_ensure_monitor)
    monkeypatch.delenv("IPW_ENERGY_MONITOR_INTERVAL_MS", raising=False)
    collector = EnergyMonitorCollector(interval_seconds=0.2)
    with collector.start():
        pass
    assert calls[0][1]["launch_args"] == ["--collection-interval-ms", "200"]


def test_collection_interval_env_override_and_validation(monkeypatch):
    calls = []

    @contextmanager
    def fake_ensure_monitor(*args, **kwargs):
        calls.append(kwargs)
        yield

    monkeypatch.setattr("ipw.telemetry.collector.ensure_monitor", fake_ensure_monitor)
    collector = EnergyMonitorCollector(interval_seconds=0.2)
    monkeypatch.setenv("IPW_ENERGY_MONITOR_INTERVAL_MS", "350")
    with collector.start():
        pass
    assert calls[0]["launch_args"] == ["--collection-interval-ms", "350"]

    monkeypatch.setenv("IPW_ENERGY_MONITOR_INTERVAL_MS", "bad")
    with pytest.raises(ValueError, match="positive integer"):
        with collector.start():
            pass


def test_full_energy_preflight_rejects_missing_dram(monkeypatch):
    @contextmanager
    def fake_ensure_monitor(*args, **kwargs):
        yield

    monkeypatch.setattr("ipw.telemetry.collector.ensure_monitor", fake_ensure_monitor)
    monkeypatch.setattr("ipw.telemetry.collector.wait_for_ready", lambda *args, **kwargs: False)
    monkeypatch.setenv("IPW_REQUIRE_TOTAL_ENERGY", "1")
    with pytest.raises(RuntimeError, match="GPU, CPU package, and DRAM"):
        with EnergyMonitorCollector(interval_seconds=0.2).start():
            pass


class TestConvertMethod:
    """Test that _convert extracts all fields from protobuf messages."""

    def _make_message(self, **kwargs) -> SimpleNamespace:
        """Create a fake protobuf message with the given fields."""
        return SimpleNamespace(**kwargs)

    def test_cpu_fields_extracted(self) -> None:
        """cpu_power_watts and cpu_energy_joules are extracted from protobuf."""
        msg = self._make_message(
            power_watts=200.0,
            energy_joules=100.0,
            temperature_celsius=65.0,
            gpu_memory_usage_mb=4096.0,
            cpu_memory_usage_mb=8192.0,
            cpu_power_watts=85.0,
            cpu_energy_joules=42.5,
            platform="linux",
            timestamp_nanos=1000000000,
        )

        collector = EnergyMonitorCollector.__new__(EnergyMonitorCollector)
        reading = collector._convert(msg)

        assert reading.cpu_power_watts == 85.0
        assert reading.cpu_energy_joules == 42.5

    def test_cpu_fields_none_when_absent(self) -> None:
        """cpu_power_watts and cpu_energy_joules are None when not in message."""
        msg = self._make_message(
            power_watts=200.0,
            energy_joules=100.0,
        )

        collector = EnergyMonitorCollector.__new__(EnergyMonitorCollector)
        reading = collector._convert(msg)

        assert reading.cpu_power_watts is None
        assert reading.cpu_energy_joules is None

    def test_negative_cpu_values_become_none(self) -> None:
        """Negative values (sentinel for unavailable) become None via _safe_float."""
        msg = self._make_message(
            cpu_power_watts=-1.0,
            cpu_energy_joules=-1.0,
        )

        collector = EnergyMonitorCollector.__new__(EnergyMonitorCollector)
        reading = collector._convert(msg)

        assert reading.cpu_power_watts is None
        assert reading.cpu_energy_joules is None

    def test_all_standard_fields_extracted(self) -> None:
        """All standard telemetry fields are properly extracted."""
        msg = self._make_message(
            power_watts=250.0,
            energy_joules=500.0,
            temperature_celsius=72.0,
            gpu_memory_usage_mb=6144.0,
            cpu_memory_usage_mb=16384.0,
            cpu_power_watts=95.0,
            cpu_energy_joules=200.0,
            platform="linux",
            timestamp_nanos=2000000000,
        )

        collector = EnergyMonitorCollector.__new__(EnergyMonitorCollector)
        reading = collector._convert(msg)

        assert reading.power_watts == 250.0
        assert reading.energy_joules == 500.0
        assert reading.temperature_celsius == 72.0
        assert reading.gpu_memory_usage_mb == 6144.0
        assert reading.cpu_memory_usage_mb == 16384.0
        assert reading.cpu_power_watts == 95.0
        assert reading.cpu_energy_joules == 200.0
        assert reading.platform == "linux"
        assert reading.timestamp_nanos == 2000000000
