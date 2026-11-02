"""Tests for the public GDPVal/OpenHands CLI surface."""

from __future__ import annotations

from click.testing import CliRunner

from ipw.cli import cli
from ipw.cli.run import _add_litellm_prefix, _is_cloud_model


def test_only_run_command_is_exposed() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "profile" not in result.output
    assert "plot" not in result.output


def test_run_help_includes_reproduction_options() -> None:
    result = CliRunner().invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    for option in (
        "--agent", "--model", "--dataset", "--dataset-kwargs",
        "--agent-kwargs", "--client-base-url", "--max-queries",
        "--concurrency", "--telemetry-accounting", "--telemetry-backend",
        "--telemetry-interval", "--eval-client", "--eval-model",
        "--judge-model", "--output-dir",
    ):
        assert option in result.output


def test_rejects_omitted_model() -> None:
    result = CliRunner().invoke(cli, ["run", "--agent", "openhands", "--dataset", "gdpval"])
    assert result.exit_code != 0
    assert "--model is required" in result.output


def test_rejects_other_agents_before_loading_dataset() -> None:
    result = CliRunner().invoke(cli, [
        "run", "--agent", "react", "--model", "example", "--dataset", "gdpval",
    ])
    assert result.exit_code != 0
    assert "Only the OpenHands agent" in result.output


def test_rejects_other_datasets_before_loading_agent() -> None:
    result = CliRunner().invoke(cli, [
        "run", "--agent", "openhands", "--model", "example", "--dataset", "gaia",
    ])
    assert result.exit_code != 0
    assert "Only the GDPval dataset" in result.output


def test_openai_prefix_is_local_with_local_endpoint() -> None:
    assert not _is_cloud_model("openai/Qwen3-32B", "http://127.0.0.1:8000")
    assert _add_litellm_prefix("Qwen3-32B", "http://127.0.0.1:8000") == "openai/Qwen3-32B"
