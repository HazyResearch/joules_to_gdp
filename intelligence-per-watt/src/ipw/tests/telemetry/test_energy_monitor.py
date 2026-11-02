from __future__ import annotations

import itertools
import socket
import time
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ipw.telemetry import EnergyMonitorCollector, ensure_monitor, launcher, wait_for_ready


@pytest.fixture(scope="module")
def monitor_target() -> Iterator[str]:
    try:
        with ensure_monitor(timeout=15.0) as target:
            # Give the monitor a moment to begin streaming metrics
            time.sleep(0.5)
            yield target
    except FileNotFoundError as exc:
        pytest.skip(f"Energy monitor binary missing: {exc}")
    except RuntimeError as exc:
        pytest.skip(f"Unable to launch energy monitor: {exc}")


def test_wait_for_ready_returns_true(monitor_target: str) -> None:
    assert wait_for_ready(monitor_target, timeout=5.0)


def test_health_must_report_usable_collector_and_required_energy(monkeypatch: pytest.MonkeyPatch) -> None:
    channel = Mock()
    future = Mock()
    monkeypatch.setattr(launcher.grpc, "insecure_channel", lambda target: channel)
    monkeypatch.setattr(launcher.grpc, "channel_ready_future", lambda ch: future)
    response = SimpleNamespace(
        healthy=False,
        gpu_energy_available=True,
        cpu_energy_available=True,
        dram_energy_available=True,
    )
    health = Mock(return_value=response)
    monkeypatch.setattr(
        launcher,
        "get_stub_bundle",
        lambda: SimpleNamespace(
            stub_factory=lambda ch: SimpleNamespace(Health=health),
            HealthRequestCls=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )
    assert not launcher.wait_for_ready("localhost")
    response.healthy = True
    response.dram_energy_available = False
    assert launcher.wait_for_ready("localhost")
    assert not launcher.wait_for_ready("localhost", require_energy_components=True)
    response.dram_energy_available = True
    assert launcher.wait_for_ready("localhost", require_energy_components=True)
    assert health.call_args.args[0].require_energy_components is True
    assert channel.close.call_count == 4
    assert launcher.normalize_target("localhost") == "localhost:50053"


def test_owned_monitor_rejects_occupied_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher, "_launch_monitor_process", lambda *args, **kwargs: pytest.fail("launched on occupied port"))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        with pytest.raises(RuntimeError, match="already in use"):
            with launcher.ensure_monitor(
                f"127.0.0.1:{port}", env={"IPW_REQUIRE_OWNED_MONITOR": "1"}
            ):
                pass


def test_owned_monitor_is_reaped_after_use(monkeypatch: pytest.MonkeyPatch) -> None:
    process = Mock()
    monkeypatch.setattr(launcher, "_require_free_loopback_target", lambda target: None)
    monkeypatch.setattr(launcher, "_launch_monitor_process", lambda *args, **kwargs: (process, "127.0.0.1:50123"))
    monkeypatch.setattr(launcher, "wait_for_ready", lambda *args, **kwargs: pytest.fail("reused monitor"))
    with launcher.ensure_monitor("127.0.0.1:50123", env={"IPW_REQUIRE_OWNED_MONITOR": "1"}):
        pass
    process.terminate.assert_called_once()
    process.wait.assert_called_once()


def test_monitor_launch_rejects_target_port_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher._binaries, "launch", lambda *args, **kwargs: pytest.fail("spawned"))
    with pytest.raises(ValueError, match="differs from target"):
        launcher.launch_monitor(["--port", "50054"], target="127.0.0.1:50053")


def test_collector_uses_visible_gpu_specific_default_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IPW_ENERGY_MONITOR_TARGET", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")

    collector = EnergyMonitorCollector()

    assert collector._target == "127.0.0.1:50057"


def test_stream_readings_produces_samples(monitor_target: str) -> None:
    collector = EnergyMonitorCollector(target=monitor_target)
    assert collector.is_available()

    readings = collector.stream_readings()
    samples = []
    for reading in itertools.islice(readings, 5):
        samples.append(reading)
        if reading.timestamp_nanos:
            break

    assert samples, "collector produced no telemetry samples"

    sample = samples[0]
    assert sample.timestamp_nanos is None or isinstance(sample.timestamp_nanos, int)
    assert sample.platform is None or isinstance(sample.platform, str)
