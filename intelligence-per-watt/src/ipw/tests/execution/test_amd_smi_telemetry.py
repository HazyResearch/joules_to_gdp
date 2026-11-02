from __future__ import annotations

import subprocess

import pytest

from ipw.execution import amd_smi_telemetry as mod
from ipw.execution.amd_smi_telemetry import AmdSmiTelemetrySession


def test_parse_rocm_smi_json() -> None:
    data = {
        "card0": {
            "Card series": "AMD Instinct MI300X",
            "Average Graphics Package Power (W)": "212.5",
            "GPU use (%)": "91",
            "Temperature (Sensor edge) (C)": "64.0",
            "VRAM Total Memory (B)": "206158430208",
            "VRAM Total Used Memory (B)": "1073741824",
        }
    }

    rows = mod._parse_rocm_smi_json(data, [0])

    assert len(rows) == 1
    assert rows[0].gpu_id == 0
    assert rows[0].name == "AMD Instinct MI300X"
    assert rows[0].power_watts == pytest.approx(212.5)
    assert rows[0].memory_used_mb == pytest.approx(1024.0)
    assert rows[0].memory_total_mb == pytest.approx(196608.0)
    assert rows[0].utilization_pct == pytest.approx(91.0)
    assert rows[0].backend == "rocm-smi"


def test_parse_amd_smi_metric_json() -> None:
    data = [
        {
            "gpu": 2,
            "asic": {"market_name": "AMD Instinct MI300X"},
            "power": {"average_socket_power": {"value": 301.25, "unit": "W"}},
            "usage": {"gfx_activity": {"value": 87, "unit": "%"}},
            "mem_usage": {
                "vram_used": {"value": 4096, "unit": "MB"},
                "vram_total": {"value": 192, "unit": "GB"},
            },
        }
    ]

    rows = mod._parse_amd_smi_json(data, [2])

    assert len(rows) == 1
    assert rows[0].gpu_id == 2
    assert rows[0].name == "AMD Instinct MI300X"
    assert rows[0].power_watts == pytest.approx(301.25)
    assert rows[0].memory_used_mb == pytest.approx(4096.0)
    assert rows[0].memory_total_mb == pytest.approx(196608.0)
    assert rows[0].utilization_pct == pytest.approx(87.0)
    assert rows[0].backend == "amd-smi"


def test_query_amd_smi_falls_back_to_rocm_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/rocm-smi" if name == "rocm-smi" else None)

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout='{"card0":{"Average Graphics Package Power (W)":"200","GPU use (%)":"50"}}',
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    rows = mod._query_amd_smi([0], backend="auto")

    assert len(rows) == 1
    assert rows[0].backend == "rocm-smi"
    assert rows[0].power_watts == pytest.approx(200.0)


def test_session_integrates_energy_for_selected_amd_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([10.0, 11.0, 12.0])
    samples = iter([
        [mod._AmdGpuSample(1, 100.0, 10.0, 100.0, 40.0, 50.0, "MI300X", "amd-smi")],
        [mod._AmdGpuSample(1, 200.0, 20.0, 100.0, 50.0, 51.0, "MI300X", "amd-smi")],
        [mod._AmdGpuSample(1, 200.0, 30.0, 100.0, 60.0, 52.0, "MI300X", "amd-smi")],
    ])
    monkeypatch.setattr(mod.time, "time", lambda: next(times))
    monkeypatch.setattr(mod, "_query_amd_smi", lambda gpu_ids, backend="auto": next(samples))

    session = AmdSmiTelemetrySession([1], backend="amd-smi")
    session._sample_once()
    session._sample_once()
    session._sample_once()

    readings = list(session.readings())
    assert len(readings) == 3
    assert readings[-1].reading.gpu_info is not None
    assert readings[-1].reading.gpu_info.vendor == "AMD"
    assert readings[-1].reading.gpu_info.device_id == 1
    assert readings[-1].reading.energy_joules == pytest.approx(350.0)
    assert readings[-1].reading.power_watts == pytest.approx(200.0)
