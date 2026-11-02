"""Run agentic benchmarks against a dataset."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import click

from ._console import error, info, success, warning
from ._display import (
    compute_trace_metrics,
    print_banner,
    print_config_summary,
    print_efficiency_panel,
    print_metrics_table,
    print_output_path,
)
from .judge_models import available_judge_presets, resolve_judge_options

_LITELLM_PREFIXES = (
    "openai/", "ollama/", "anthropic/", "gemini/", "google/",
    "azure/", "bedrock/", "vertex_ai/",
)

_ALWAYS_CLOUD_PROVIDER_PREFIXES = (
    "anthropic/", "gemini/", "azure/", "bedrock/", "vertex_ai/",
)

_BARE_CLOUD_PREFIXES = ("claude-", "gemini-", "gpt-5", "gpt-4", "o1", "o3", "o4")
_DEFAULT_LLM_TIMEOUT_SECONDS = 120
def _is_commandline_option(ctx: click.Context, name: str) -> bool:
    return ctx.get_parameter_source(name) is click.core.ParameterSource.COMMANDLINE


def _is_cloud_model(model: str, base_url: str) -> bool:
    """Return True when the model targets a cloud API provider.

    Cloud models are detected by their litellm provider prefix.  ``openai/``
    is treated as local when the base_url points to a local server
    (localhost / 127.0.0.1 / 0.0.0.0), and cloud otherwise.
    """
    _local_hosts = ("localhost", "127.0.0.1", "0.0.0.0")
    local_base_url = any(h in base_url for h in _local_hosts)
    if model.startswith(_BARE_CLOUD_PREFIXES) and not model.startswith("gpt-oss"):
        return True
    if model.startswith(_ALWAYS_CLOUD_PROVIDER_PREFIXES):
        return True
    if model.startswith("google/"):
        return not local_base_url
    if model.startswith("openai/"):
        return not local_base_url
    return False


def _add_litellm_prefix(model: str, base_url: str) -> str:
    """Add LiteLLM provider prefix if not already present."""
    if model.startswith(_LITELLM_PREFIXES):
        if (
            not _is_cloud_model(model, base_url)
            and not model.startswith(("openai/", "ollama/"))
        ):
            return f"openai/{model}"
        return model
    if model.startswith("claude-"):
        return f"anthropic/{model}"
    if model.startswith("gemini-"):
        return f"gemini/{model}"
    if model.startswith(("gpt-", "o1", "o3", "o4")) and not model.startswith("gpt-oss"):
        return f"openai/{model}"
    is_ollama = "11434" in base_url or "ollama" in base_url.lower()
    prefix = "ollama" if is_ollama else "openai"
    return f"{prefix}/{model}"


def _create_model_for_agent(agent_id: str, model: str, base_url: str, api_key: str):
    """Create the framework-specific model object for the given agent type."""
    import os

    cloud = _is_cloud_model(model, base_url)

    # Normalize api_key: treat "EMPTY" as None so litellm reads env vars
    effective_key = None if api_key == "EMPTY" else api_key

    if agent_id == "openhands":
        from openhands.sdk import LLM

        litellm_model = _add_litellm_prefix(model, base_url)
        llm_kwargs = {
            "timeout": int(
                os.getenv("IPW_OPENHANDS_LLM_TIMEOUT")
                or os.getenv("IPW_OPENHANDS_LLM_TIMEOUT_SECONDS")
                or str(_DEFAULT_LLM_TIMEOUT_SECONDS)
            )
        }
        if os.getenv("IPW_OPENHANDS_MAX_OUTPUT_TOKENS"):
            llm_kwargs["max_output_tokens"] = int(
                os.environ["IPW_OPENHANDS_MAX_OUTPUT_TOKENS"]
            )
        if os.getenv("IPW_OPENHANDS_MAX_MESSAGE_CHARS"):
            llm_kwargs["max_message_chars"] = int(
                os.environ["IPW_OPENHANDS_MAX_MESSAGE_CHARS"]
            )
        if os.getenv("IPW_OPENHANDS_TEMPERATURE"):
            llm_kwargs["temperature"] = float(os.environ["IPW_OPENHANDS_TEMPERATURE"])
        if os.getenv("IPW_OPENHANDS_TOP_P"):
            llm_kwargs["top_p"] = float(os.environ["IPW_OPENHANDS_TOP_P"])
        if os.getenv("IPW_OPENHANDS_SEED"):
            llm_kwargs["seed"] = int(os.environ["IPW_OPENHANDS_SEED"])
        if os.getenv("IPW_OPENHANDS_REASONING_EFFORT"):
            llm_kwargs["reasoning_effort"] = os.environ[
                "IPW_OPENHANDS_REASONING_EFFORT"
            ]
        if os.getenv("IPW_OPENHANDS_EXTENDED_THINKING_BUDGET"):
            value = os.environ["IPW_OPENHANDS_EXTENDED_THINKING_BUDGET"]
            llm_kwargs["extended_thinking_budget"] = (
                None if value.lower() == "none" else int(value)
            )
        if os.getenv("IPW_OPENHANDS_LITELLM_EXTRA_BODY_JSON"):
            try:
                llm_kwargs["litellm_extra_body"] = json.loads(
                    os.environ["IPW_OPENHANDS_LITELLM_EXTRA_BODY_JSON"]
                )
            except json.JSONDecodeError as exc:
                raise click.ClickException(
                    f"Invalid IPW_OPENHANDS_LITELLM_EXTRA_BODY_JSON: {exc}"
                ) from exc
        if os.getenv("IPW_OPENHANDS_NUM_RETRIES"):
            llm_kwargs["num_retries"] = int(
                os.environ["IPW_OPENHANDS_NUM_RETRIES"]
            )
        if os.getenv("IPW_OPENHANDS_CONNECTION_CLOSE"):
            llm_kwargs["extra_headers"] = {"Connection": "close"}
        if cloud:
            return LLM(model=litellm_model, api_key=effective_key, **llm_kwargs)
        # Ollama native API doesn't use /v1; OpenAI-compatible servers do
        is_ollama = "11434" in base_url or "ollama" in base_url.lower()
        if is_ollama:
            llm_base_url = base_url
        elif base_url.rstrip("/").endswith("/v1"):
            llm_base_url = base_url
        else:
            llm_base_url = f"{base_url}/v1"
        return LLM(
            model=litellm_model,
            api_key=api_key,
            base_url=llm_base_url,
            **llm_kwargs,
        )
    raise click.ClickException(f"Unsupported agent: {agent_id}; use openhands")


@click.command(help="Run an agentic benchmark against a dataset.")
@click.option("--agent", "agent_id", required=True, help="Agent identifier")
@click.option("--model", default=None, help="Model name for the OpenHands agent")
@click.option("--dataset", "dataset_id", required=True, help="Dataset identifier")
@click.option(
    "--client-base-url",
    default="http://localhost:8000",
    show_default=True,
    help="Inference server base URL",
)
@click.option(
    "--api-key",
    default="EMPTY",
    show_default=True,
    help="API key for the inference server",
)
@click.option(
    "--max-queries",
    type=int,
    default=None,
    help="Max queries to run (default: all)",
)
@click.option(
    "--start-index",
    type=int,
    default=0,
    show_default=True,
    help="Dataset record index to start from; useful for continuing a partial run",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./runs/",
    show_default=True,
    help="Output directory",
)
@click.option(
    "--export-format",
    default="jsonl,hf",
    show_default=True,
    help="Comma-separated export formats (jsonl, hf)",
)
@click.option(
    "--agent-kwargs",
    default=None,
    help="JSON string of extra agent keyword arguments",
)
@click.option(
    "--dataset-kwargs",
    default=None,
    help="JSON string of extra dataset keyword arguments",
)
@click.option(
    "--concurrency",
    type=int,
    default=1,
    show_default=True,
    help="Number of tasks to run in parallel",
)
@click.option(
    "--telemetry-accounting",
    type=click.Choice(["query", "batch"]),
    default="query",
    show_default=True,
    help="Energy accounting scope. query preserves per-prompt telemetry; batch accounts overlapping concurrent prompts once per batch.",
)
@click.option(
    "--query-timeout",
    type=float,
    default=None,
    help="Wall-clock timeout in seconds per query (default: no limit)",
)
@click.option(
    "--telemetry-gpu-id",
    type=int,
    default=None,
    help="Sample only this NVIDIA GPU id via nvidia-smi instead of aggregate energy-monitor telemetry",
)
@click.option(
    "--telemetry-backend",
    type=click.Choice(["energy-monitor", "auto", "nvidia-smi", "amd-smi", "rocm-smi"]),
    default="energy-monitor",
    show_default=True,
    help="Telemetry backend. energy-monitor preserves the previous default; chip profiles may use auto or explicit SMI backends.",
)
@click.option(
    "--telemetry-device-id",
    type=int,
    multiple=True,
    help="GPU device id for --telemetry-backend nvidia-smi/amd-smi/rocm-smi/auto. Repeat for tensor-parallel groups.",
)
@click.option(
    "--telemetry-interval",
    type=float,
    default=0.2,
    show_default=True,
    help="Telemetry sampling interval in seconds",
)
@click.option(
    "--telemetry-buffer-seconds",
    type=float,
    default=None,
    help="Telemetry retention window in seconds (default: query timeout + 300s, or 7200s)",
)
@click.option(
    "--max-retries",
    type=int,
    default=3,
    show_default=True,
    help="Per-turn retry budget on transient errors (0 disables retry).",
)
@click.option(
    "--require-dedicated-hardware",
    is_flag=True,
    default=False,
    help="Abort if preflight detects competing GPU/CPU workloads (precision mode).",
)
@click.option(
    "--judge-model",
    default=None,
    help=(
        "Judge model preset. Available: "
        + ", ".join(available_judge_presets())
        + " (default: gpt-mini)"
    ),
)
@click.option(
    "--eval-client",
    default=None,
    help="Raw evaluation client override (default: resolved from --judge-model)",
)
@click.option(
    "--eval-base-url",
    default=None,
    help="Base URL for evaluation client (default: dataset/client default)",
)
@click.option(
    "--eval-model",
    default=None,
    help="Raw evaluation model override (default: gpt-mini preset)",
)
def run_cmd(
    agent_id: str,
    model: str | None,
    dataset_id: str,
    client_base_url: str,
    api_key: str,
    max_queries: int | None,
    start_index: int,
    output_dir: str,
    export_format: str,
    agent_kwargs: str | None,
    dataset_kwargs: str | None,
    concurrency: int,
    telemetry_accounting: str,
    query_timeout: float | None,
    telemetry_gpu_id: int | None,
    telemetry_backend: str,
    telemetry_device_id: tuple[int, ...],
    telemetry_interval: float,
    telemetry_buffer_seconds: float | None,
    max_retries: int,
    require_dedicated_hardware: bool,
    judge_model: str | None,
    eval_client: str | None,
    eval_base_url: str | None,
    eval_model: str | None,
) -> None:
    """Execute an agentic benchmark run."""
    if not model:
        raise click.ClickException("--model is required")
    if agent_id != "openhands":
        raise click.ClickException("Only the OpenHands agent is included in this release")
    if dataset_id != "gdpval":
        raise click.ClickException("Only the GDPval dataset is included in this release")
    if concurrency < 1:
        raise click.ClickException("--concurrency must be positive")

    ctx = click.get_current_context()
    try:
        judge_config = resolve_judge_options(
            judge_model=judge_model,
            eval_client=eval_client,
            eval_base_url=eval_base_url,
            eval_model=eval_model,
            explicit_eval_client=_is_commandline_option(ctx, "eval_client"),
            explicit_eval_base_url=_is_commandline_option(ctx, "eval_base_url"),
            explicit_eval_model=_is_commandline_option(ctx, "eval_model"),
        )
    except (KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    eval_client = judge_config.client
    eval_base_url = judge_config.base_url
    eval_model = judge_config.model

    import ipw.clients
    import ipw.datasets

    ipw.clients.ensure_registered()
    ipw.datasets.ensure_registered()

    from ipw.agents import openhands as _openhands  # noqa: F401
    from ipw.core.registry import AgentRegistry, DatasetRegistry
    from ipw.execution.agentic_runner import AgenticRunner
    from ipw.execution.exporters import (
        export_artifacts_manifest,
        export_batches_jsonl,
        export_hf_dataset,
        export_jsonl,
        export_summary_json,
    )
    from ipw.execution.telemetry_backends import build_telemetry_context
    from ipw.telemetry.events import EventRecorder

    # Parse agent kwargs
    extra_kwargs: dict = {}
    if agent_kwargs:
        try:
            extra_kwargs = json.loads(agent_kwargs)
        except json.JSONDecodeError as exc:
            raise click.ClickException(
                f"Invalid JSON for --agent-kwargs: {exc}"
            ) from exc

    # Parse dataset kwargs
    extra_dataset_kwargs: dict = {}
    if dataset_kwargs:
        try:
            extra_dataset_kwargs = json.loads(dataset_kwargs)
        except json.JSONDecodeError as exc:
            raise click.ClickException(
                f"Invalid JSON for --dataset-kwargs: {exc}"
            ) from exc

    # Resolve agent
    try:
        agent_cls = AgentRegistry.get(agent_id)
    except KeyError:
        available = [k for k, _ in AgentRegistry.items()]
        raise click.ClickException(
            f"Unknown agent '{agent_id}'. Available: {', '.join(available) or 'none'}"
        )

    # Resolve dataset
    try:
        dataset_cls = DatasetRegistry.get(dataset_id)
    except KeyError:
        available = [k for k, _ in DatasetRegistry.items()]
        raise click.ClickException(
            f"Unknown dataset '{dataset_id}'. Available: {', '.join(available) or 'none'}"
        )

    # Preflight: dataset requirements
    try:
        dataset_instance = dataset_cls(**extra_dataset_kwargs)
        dataset_instance.eval_client = eval_client
        if eval_base_url:
            dataset_instance.eval_base_url = eval_base_url
        if eval_model:
            dataset_instance.eval_model = eval_model
        issues = dataset_instance.verify_requirements()
        if issues:
            raise click.ClickException(
                "Dataset requirements not satisfied:\n- " + "\n- ".join(issues)
            )
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f"Failed to initialize dataset: {exc}") from exc

    # Create event recorder and agent
    event_recorder = EventRecorder()
    resolved_model = _create_model_for_agent(agent_id, model, client_base_url, api_key)

    def _build_agent(a_cls, a_resolved_model, a_event_recorder, a_extra_kwargs):
        return a_cls(
            model=a_resolved_model,
            event_recorder=a_event_recorder,
            **a_extra_kwargs,
        )

    try:
        agent_instance = _build_agent(agent_cls, resolved_model, event_recorder, extra_kwargs)
    except TypeError as exc:
        raise click.ClickException(
            f"Failed to initialize agent '{agent_id}': {exc}"
        ) from exc

    # Prepare output directory
    model_slug = "".join(c if c.isalnum() else "_" for c in model).strip("_") or "model"
    run_dir = Path(output_dir) / f"run_{agent_id}_{model_slug}_{dataset_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Convert CLI retries count to runner attempt count (retries + 1 = attempts).
    max_attempts = max_retries + 1


    # Build config for summary
    run_config = {
        "agent": agent_id,
        "model": model,
        "dataset": dataset_id,
        "client_base_url": client_base_url,
        "max_queries": max_queries,
        "start_index": start_index,
        "concurrency": concurrency,
        "telemetry_accounting": telemetry_accounting,
        "query_timeout": query_timeout,
        "telemetry_gpu_id": telemetry_gpu_id,
        "telemetry_backend": telemetry_backend,
        "telemetry_device_ids": list(telemetry_device_id),
        "telemetry_interval": telemetry_interval,
        "telemetry_buffer_seconds": telemetry_buffer_seconds,
        "export_format": export_format,
        "eval_client": eval_client,
        "eval_base_url": eval_base_url,
        "eval_model": eval_model,
        "judge_model_preset": judge_config.preset,
        "max_retries": max_retries,
        "require_dedicated_hardware": require_dedicated_hardware,
        "gdpval_repair_attempts": int(
            os.getenv("IPW_GDPVAL_REPAIR_ATTEMPTS", "2")
        ),
        "gdpval_final_check_attempts": int(
            os.getenv("IPW_GDPVAL_FINAL_CHECK_ATTEMPTS", "1")
        ),
    }

    print_banner()
    run_display_config: dict[str, object] = {
        "Agent": agent_id,
        "Model": model,
        "Dataset": dataset_id,
        "Base URL": client_base_url,
        "Max Queries": max_queries or "(all)",
    }
    if start_index:
        run_display_config["Start Index"] = start_index
    if concurrency > 1:
        run_display_config["Concurrency"] = concurrency
    if telemetry_accounting != "query":
        run_display_config["Telemetry Accounting"] = telemetry_accounting
    if query_timeout:
        run_display_config["Query Timeout"] = f"{query_timeout:.0f}s"
    if telemetry_gpu_id is not None:
        run_display_config["Telemetry GPU"] = telemetry_gpu_id
    if telemetry_backend != "energy-monitor":
        run_display_config["Telemetry Backend"] = telemetry_backend
    if telemetry_device_id:
        run_display_config["Telemetry Device IDs"] = ",".join(str(v) for v in telemetry_device_id)
    if telemetry_buffer_seconds is not None:
        run_display_config["Telemetry Buffer"] = f"{telemetry_buffer_seconds:.0f}s"
    if judge_config.preset:
        run_display_config["Judge"] = f"{judge_config.preset} ({eval_model})"
    else:
        run_display_config["Judge"] = f"{eval_client}/{eval_model}"
    if getattr(dataset_instance, "requires_serial_telemetry", False) and telemetry_accounting == "batch":
        raise click.ClickException(
            f"Dataset '{dataset_id}' requires clean per-prompt telemetry; batch telemetry is not allowed."
        )
    if getattr(dataset_instance, "requires_serial_telemetry", False) and concurrency != 1:
        warning(
            f"Dataset '{dataset_id}' requires clean per-prompt telemetry; forcing concurrency=1."
        )
        concurrency = 1
        run_config["concurrency"] = 1
        run_display_config["Concurrency"] = 1
    elif telemetry_accounting == "query" and concurrency > 1:
        warning(
            "concurrency > 1 with query telemetry accounting can overlap prompt windows; "
            "aggregate energy may double count. Use --telemetry-accounting batch for shared-server batch accounting."
        )
    run_display_config["Output"] = str(run_dir)
    print_config_summary(config=run_display_config)

    # Build an agent factory for batch/concurrent execution so each worker gets
    # its own agent instance with independent state.
    _model_ref = resolved_model
    _agent_cls_ref = agent_cls
    _extra_kwargs_ref = dict(extra_kwargs)

    def _make_agent(recorder: EventRecorder) -> "BaseAgent":  # noqa: F821
        return _build_agent(_agent_cls_ref, _model_ref, recorder, _extra_kwargs_ref)

    # Run the agentic benchmark.
    # Cloud models don't use the local GPU — skip energy telemetry so traces
    # report None instead of misleading idle-GPU power draw.
    cloud = _is_cloud_model(model, client_base_url)

    batch_records = []

    def _run_with_telemetry(telemetry):
        nonlocal batch_records
        runner = AgenticRunner(
            agent=agent_instance,
            dataset=dataset_instance,
            telemetry_session=telemetry,
            config=run_config,
            event_recorder=event_recorder,
            run_dir=run_dir,
            concurrency=concurrency,
            agent_factory=(
                _make_agent if concurrency > 1 or telemetry_accounting == "batch" else None
            ),
            query_timeout=query_timeout,
            max_attempts=max_attempts,
            require_dedicated_hardware=require_dedicated_hardware,
            telemetry_accounting=telemetry_accounting,
        )
        try:
            run_kwargs = {"max_queries": max_queries}
            if start_index:
                run_kwargs["start_index"] = start_index
            traces_result = asyncio.run(runner.run(**run_kwargs))
            batch_records = runner.batch_records
            return traces_result
        except Exception as exc:
            error(f"Run failed: {exc}")
            sys.exit(1)

    if cloud:
        traces = _run_with_telemetry(None)
    else:
        resolved_telemetry_buffer_seconds = telemetry_buffer_seconds
        if resolved_telemetry_buffer_seconds is None:
            resolved_telemetry_buffer_seconds = (
                max(300.0, query_timeout + 300.0)
                if query_timeout is not None
                else 7200.0
            )
        if resolved_telemetry_buffer_seconds <= 0:
            raise click.ClickException("--telemetry-buffer-seconds must be positive")
        telemetry_max_samples = int(
            resolved_telemetry_buffer_seconds / max(telemetry_interval, 0.001)
        ) + 100
        try:
            telemetry_context = build_telemetry_context(
                backend=telemetry_backend,
                telemetry_gpu_id=telemetry_gpu_id,
                telemetry_device_ids=telemetry_device_id,
                interval_seconds=telemetry_interval,
                buffer_seconds=resolved_telemetry_buffer_seconds,
                max_samples=telemetry_max_samples,
                sample_output_path=run_dir / "telemetry_samples.jsonl",
            )
        except Exception as exc:
            raise click.ClickException(f"Telemetry setup failed: {exc}") from exc
        with telemetry_context as telemetry:
            traces = _run_with_telemetry(telemetry)
        sample_recorder = getattr(telemetry, "sample_recorder", None)
        if sample_recorder is not None:
            sample_manifest = sample_recorder.manifest()
            (run_dir / "telemetry_samples_manifest.json").write_text(
                json.dumps(sample_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            run_config["telemetry_samples"] = sample_manifest

    if not traces:
        warning("No traces collected.")
        return

    # Export results
    formats = [f.strip().lower() for f in export_format.split(",") if f.strip()]

    if "jsonl" in formats:
        jsonl_path = export_jsonl(traces, run_dir / "traces.jsonl")
        info(f"Exported JSONL: {jsonl_path}")

    if batch_records:
        batches_path = export_batches_jsonl(batch_records, run_dir / "batches.jsonl")
        info(f"Exported batch telemetry JSONL: {batches_path}")

    if "hf" in formats:
        hf_path = export_hf_dataset(traces, run_dir / "hf_dataset")
        info(f"Exported HF dataset: {hf_path}")

    summary_path = export_summary_json(
        traces,
        run_config,
        run_dir / "summary.json",
        batch_records=batch_records,
    )
    info(f"Exported summary: {summary_path}")

    manifest_path = export_artifacts_manifest(run_dir)
    if manifest_path:
        info(f"Exported artifacts manifest: {manifest_path}")

    # Rich summary display
    total_completed = sum(1 for t in traces if t.completed)
    timed_out = [t for t in traces if t.timed_out]
    success(f"Run complete: {total_completed}/{len(traces)} queries completed")

    if timed_out:
        warning(f"{len(timed_out)} queries timed out:")
        for t in timed_out:
            warning(f"  {t.query_id}: {t.total_wall_clock_s:.0f}s")

    metric_rows = compute_trace_metrics(traces)
    print_metrics_table(rows=metric_rows, title="Run Metrics")

    # Compute accuracy from is_resolved traces
    resolved_traces = [t for t in traces if t.is_resolved is not None]
    accuracy = None
    if resolved_traces:
        accuracy = sum(1 for t in resolved_traces if t.is_resolved) / len(resolved_traces)
    print_efficiency_panel(traces=traces, accuracy=accuracy)
    print_output_path(path=run_dir)


__all__ = ["run_cmd"]
