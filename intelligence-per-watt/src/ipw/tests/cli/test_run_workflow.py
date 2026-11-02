"""Check that the deployment launcher reaches the IPW run implementation."""

from __future__ import annotations

import json
from contextlib import nullcontext
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from ipw.cli import cli

ROOT = Path(__file__).resolve().parents[5]


def _launcher_command(profile: str, output_dir: Path) -> list[str]:
    spec = spec_from_file_location("j2g_test_deployment", ROOT / "reproduction/run_deployment.py")
    assert spec is not None and spec.loader is not None
    deployment = module_from_spec(spec)
    spec.loader.exec_module(deployment)

    tasks = json.loads(
        (ROOT / "reproduction/fixtures/fabricated_tasks.json").read_text(encoding="utf-8")
    )
    batches = deployment.occupation_batches(tasks)
    assert len(batches) == 44
    server, runs = deployment.commands(
        profile,
        output_dir,
        batches,
        port=8000,
        revision="0" * 40,
        vllm_args=[],
        existing_server=True,
    )
    assert server is None
    assert len(runs) == 44
    return runs[0]


def _invoke_paper_run(
    monkeypatch: pytest.MonkeyPatch,
    command: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run Click with fabricated providers and capture the runner's inputs."""
    import ipw.clients
    import ipw.datasets
    from ipw.core.registry import AgentRegistry, DatasetRegistry
    from ipw.execution import agentic_runner, telemetry_backends

    captured: dict[str, Any] = {}
    telemetry_args: dict[str, Any] = {}

    class FakeDataset:
        requires_serial_telemetry = False

        def __init__(self, **kwargs: Any) -> None:
            captured["dataset_kwargs"] = kwargs
            captured["dataset"] = self

        def verify_requirements(self) -> list[str]:
            return []

    class FakeAgent:
        def __init__(self, **kwargs: Any) -> None:
            captured["agent_kwargs"] = kwargs

    class FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            captured["runner_kwargs"] = kwargs
            self.batch_records: list[Any] = []

        async def run(self, **kwargs: Any) -> list[Any]:
            captured["run_kwargs"] = kwargs
            return []

    def fake_telemetry_context(**kwargs: Any) -> nullcontext[object]:
        telemetry_args.update(kwargs)
        return nullcontext(object())

    monkeypatch.setattr(ipw.clients, "ensure_registered", lambda: None)
    monkeypatch.setattr(ipw.datasets, "ensure_registered", lambda: None)
    monkeypatch.setattr(AgentRegistry, "get", lambda _id: FakeAgent)
    monkeypatch.setattr(DatasetRegistry, "get", lambda _id: FakeDataset)
    monkeypatch.setattr(agentic_runner, "AgenticRunner", FakeRunner)
    monkeypatch.setattr(telemetry_backends, "build_telemetry_context", fake_telemetry_context)
    monkeypatch.setattr("ipw.cli.run._create_model_for_agent", lambda *_args: object())

    result = CliRunner().invoke(cli, command[1:], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "runner_kwargs" in captured
    return captured, captured["runner_kwargs"], telemetry_args


def test_glm_launcher_flags_reach_agent_runner_and_meter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    command = _launcher_command("glm52_h200_8", tmp_path)
    captured, runner, meter = _invoke_paper_run(monkeypatch, command)

    assert runner["concurrency"] == 20
    assert runner["telemetry_accounting"] == "batch"
    assert runner["require_dedicated_hardware"] is True
    assert runner["max_attempts"] == 4
    assert runner["agent_factory"] is not None
    assert captured["run_kwargs"] == {"max_queries": 20}
    assert captured["dataset_kwargs"]["include_rubric_in_prompt"] is False
    assert captured["dataset"].eval_client == "openai"
    assert captured["dataset"].eval_model == "gpt-5.4-mini"
    assert captured["agent_kwargs"]["max_turns"] == 160
    assert runner["config"]["agent"] == "openhands"
    assert runner["config"]["dataset"] == "gdpval"
    assert runner["config"]["judge_model_preset"] is None
    assert runner["config"]["max_queries"] == 20
    assert runner["query_timeout"] == 14400
    assert meter["backend"] == "energy-monitor"
    assert meter["interval_seconds"] == 0.2
    assert meter["buffer_seconds"] == 14700
    assert meter["sample_output_path"].name == "telemetry_samples.jsonl"


def test_retry_override_and_amd_preflight_reach_agent_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    command = _launcher_command("qwen36_mi350x_1", tmp_path)
    captured, runner, _meter = _invoke_paper_run(
        monkeypatch, [*command, "--max-retries", "0"],
    )

    assert runner["concurrency"] == 20
    assert runner["telemetry_accounting"] == "batch"
    assert runner["require_dedicated_hardware"] is False
    assert runner["max_attempts"] == 1
    assert captured["dataset"].eval_client == "gemini"
