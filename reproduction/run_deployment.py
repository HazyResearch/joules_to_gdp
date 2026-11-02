#!/usr/bin/env python3
"""Run occupation-pure GDPVal/OpenHands batches into an ignored directory.

The task inventory is an external input. Only task IDs and occupation names are
copied into the generated schedule; prompts, reference files, and results stay
outside the repository.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "reproduction" / "deployments.json"
INPUT_MANIFEST = ROOT / "reproduction" / "input_manifest.json"
RUNS = ROOT / "runs"
LOCKED_VLLM_VERSION = "0.11.0"
AGENT_INSTANCE_SOURCE = "cluster/gdpval_occupation_repeat4_instances.json"
LOCKED_VLLM_AUTOSTART_MODELS = frozenset({"openai/gpt-oss-120b", "openai/gpt-oss-20b"})
PINNED_VLLM_FLAGS = frozenset({
    "--model", "--served-model-name", "--revision", "--tensor-parallel-size",
    "--max-model-len", "--max-num-seqs", "--port", "--host", "--api-key",
    "--ssl-keyfile", "--ssl-certfile", "--root-path", "--config",
    "--generation-config", "--override-generation-config", "--chat-template", "--seed",
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_task_inventory(path: Path) -> list[dict[str, str]]:
    """Require the recorded 220 tasks: 44 occupations with five tasks each."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != 220:
        raise ValueError("task inventory must be a JSON list of exactly 220 tasks")
    groups: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"task {index} must be a JSON object")
        task_id = item.get("task_id")
        occupation = item.get("occupation")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"task {index} has no task_id")
        if not isinstance(occupation, str) or not occupation.strip():
            raise ValueError(f"task {index} has no occupation")
        task_id, occupation = task_id.strip(), occupation.strip()
        if task_id in seen:
            raise ValueError(f"duplicate task_id: {task_id}")
        seen.add(task_id)
        groups[occupation].append(task_id)
    if len(groups) != 44 or any(len(ids) != 5 for ids in groups.values()):
        raise ValueError("task inventory must have 44 occupations of five tasks each")
    return [{"task_id": item["task_id"].strip(), "occupation": item["occupation"].strip()}
            for item in raw]


def validate_recorded_task_hash(path: Path) -> None:
    manifest = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    matches = [item for item in manifest["inputs"]
               if item["path"] == AGENT_INSTANCE_SOURCE]
    if len(matches) != 1 or not matches[0].get("sha256"):
        raise ValueError("input manifest lacks the recorded GDPVal agent-instance checksum")
    if sha256_file(path) != matches[0]["sha256"]:
        raise ValueError("agent-instance roster hash does not match the recorded external input")


def load_task_instances(path: Path) -> list[list[dict[str, Any]]]:
    """Validate the external task roster without reordering its 880 instances."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != 880:
        raise ValueError("agent-instance roster must contain exactly 880 requests")
    batches: list[list[dict[str, Any]]] = []
    task_metadata: dict[str, tuple[str, int, int, int]] = {}
    instance_ids: set[str] = set()
    occupations: set[str] = set()
    for batch_index in range(44):
        batch = raw[batch_index * 20:(batch_index + 1) * 20]
        batch_occupations: set[str] = set()
        for position, item in enumerate(batch):
            if not isinstance(item, dict):
                raise ValueError("agent-instance roster contains a non-object")
            task_id, occupation, instance_id = (
                item.get("task_id"), item.get("occupation"), item.get("instance_id")
            )
            if any(not isinstance(value, str) or not value.strip()
                   for value in (task_id, occupation, instance_id)):
                raise ValueError("agent-instance roster has an empty task, occupation, or instance ID")
            repeat, source_index = item.get("repeat_index"), item.get("source_task_index")
            expected_position, expected_repeat = divmod(position, 4)
            expected_id = f"occ_b{batch_index:02d}"
            if (type(repeat) is not int or repeat != expected_repeat
                    or item.get("repeat_count") != 4
                    or item.get("occupation_batch_index") != batch_index
                    or item.get("occupation_batch_id") != expected_id
                    or item.get("occupation_task_position") != expected_position
                    or type(source_index) is not int
                    or not 0 <= source_index < 220
                    or instance_id != f"{task_id}::rep{repeat:02d}"):
                raise ValueError("agent-instance roster order or repeat metadata differs from the configured protocol")
            if instance_id in instance_ids:
                raise ValueError("agent-instance roster contains a duplicate instance ID")
            instance_ids.add(instance_id)
            metadata = (occupation, batch_index, expected_position, source_index)
            if task_id in task_metadata and task_metadata[task_id] != metadata:
                raise ValueError("agent task appears in multiple occupations or positions")
            task_metadata[task_id] = metadata
            batch_occupations.add(occupation)
        if len(batch_occupations) != 1:
            raise ValueError("agent-instance batch mixes occupations")
        occupations.update(batch_occupations)
        batches.append(batch)
    if (len(task_metadata) != 220 or len(occupations) != 44
            or {item[3] for item in task_metadata.values()} != set(range(220))):
        raise ValueError("agent-instance roster must have 220 tasks in 44 occupations")
    return batches


def occupation_batches(tasks: list[dict[str, str]], attempts: int = 4) -> list[list[dict[str, Any]]]:
    """Group each occupation's five tasks into one concurrent 20-request batch."""
    if not 1 <= attempts <= 4:
        raise ValueError("attempts must be between one and four")
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for source_index, item in enumerate(tasks):
        groups[item["occupation"]].append((item["task_id"], source_index))
    if len(groups) != 44 or any(len(ids) != 5 for ids in groups.values()):
        raise ValueError("expected 44 occupations with five tasks each")
    batches = []
    for batch_number, occupation in enumerate(sorted(groups)):
        instances: list[dict[str, Any]] = []
        for task_position, (task_id, source_index) in enumerate(groups[occupation]):
            for attempt in range(attempts):
                instances.append({
                    "task_id": task_id,
                    "instance_id": f"{task_id}::rep{attempt:02d}",
                    "repeat_index": attempt,
                    "repeat_count": attempts,
                    "occupation": occupation,
                    "occupation_batch_index": batch_number,
                    "occupation_batch_id": f"occ_b{batch_number:02d}",
                    "occupation_task_position": task_position,
                    "source_task_index": source_index,
                })
        batches.append(instances)
    return batches


def validate_vllm_args(args: list[str]) -> None:
    """Additional server tuning must not replace the recorded protocol or carry secrets."""
    for value in args:
        flag = value.split("=", 1)[0]
        if flag in PINNED_VLLM_FLAGS:
            raise ValueError(f"--vllm-arg cannot override {flag}")
        if (re.search(r"(?:key|secret|password|auth-token|access-token|hf-token|token)(?:=|$)", flag)
                and not flag.startswith("--tokenizer")):
            raise ValueError(f"--vllm-arg cannot contain credential flag {flag}; use the environment")
        if "\n" in value or "\r" in value or "\x00" in value:
            raise ValueError("--vllm-arg contains a control character")


def ipw_run_dir(model: str, run_dir: Path, batch_id: str) -> Path:
    slug = "".join(c if c.isalnum() else "_" for c in model).strip("_") or "model"
    return run_dir / "batches" / batch_id / f"run_openhands_{slug}_gdpval"


def operational_judge(deployment: dict[str, Any], protocol: dict[str, Any]) -> dict[str, str]:
    return deployment.get("operational_judge", {
        "client": "gemini", "model": protocol["operational_judge_model"],
    })


def server_max_num_seqs(deployment: dict[str, Any], protocol: dict[str, Any]) -> int:
    return int(deployment.get("server_max_num_seqs", protocol["concurrency"]))


def commands(profile: str, run_dir: Path, batches: list[list[dict[str, Any]]], *,
             port: int, revision: str, vllm_args: list[str],
             existing_server: bool) -> tuple[list[str] | None, list[list[str]]]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    deployment = config["deployments"][profile]
    model = deployment["model"]
    server = None
    if not existing_server:
        if model not in LOCKED_VLLM_AUTOSTART_MODELS:
            raise ValueError(f"For {model}, start vLLM and pass --existing-server and --server-manifest")
        server = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                  "--model", model, "--served-model-name", model,
                  "--revision", revision,
                  "--tensor-parallel-size", str(deployment["tensor_parallel"]),
                  "--max-model-len", str(protocol["context_tokens"]),
                  "--max-num-seqs", str(server_max_num_seqs(deployment, protocol)),
                  "--host", "127.0.0.1", "--port", str(port), *vllm_args]
    agent_kwargs = json.dumps({
        "max_turns": deployment.get("agent_max_turns", 150),
        "openhands_tool_names": ["terminal", "file_editor"],
    })
    judge = operational_judge(deployment, protocol)
    runner = str(Path(sys.executable).parent / "ipw")
    hardware_guard = (["--require-dedicated-hardware"]
                      if deployment["accelerator"] != "MI350X" else [])
    run_commands = []
    for batch in batches:
        batch_id = batch[0]["occupation_batch_id"]
        dataset_kwargs = json.dumps({
            "revision": protocol["dataset_revision"],
            "task_instances_file": str(run_dir / "schedule" / f"{batch_id}.json"),
            "include_rubric_in_prompt": False,
        })
        run_commands.append([
            runner, "run", "--agent", protocol["agent"], "--model", model,
            "--dataset", protocol["dataset"], "--dataset-kwargs", dataset_kwargs,
            "--agent-kwargs", agent_kwargs, "--client-base-url", f"http://127.0.0.1:{port}",
            "--max-queries", str(len(batch)), "--concurrency", str(protocol["concurrency"]),
            "--telemetry-accounting", "batch", "--telemetry-backend", "energy-monitor",
            "--telemetry-interval", str(protocol["telemetry_interval_seconds"]),
            *hardware_guard, "--eval-client", judge["client"],
            "--eval-model", judge["model"],
            *(["--query-timeout", str(deployment["query_timeout_seconds"])]
              if "query_timeout_seconds" in deployment else []),
            *(["--telemetry-buffer-seconds", str(deployment["telemetry_buffer_seconds"])]
              if "telemetry_buffer_seconds" in deployment else []),
            "--output-dir", str(run_dir / "batches" / batch_id),
        ])
    return server, run_commands


def energy_source_config(profile: str, run_dir: Path, batches: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Point numerical analysis at this run's IPW batch, trace, and subset files."""
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    deployment = config["deployments"][profile]
    sources = []
    for batch in batches:
        batch_id = batch[0]["occupation_batch_id"]
        ipw_dir = ipw_run_dir(deployment["model"], run_dir, batch_id)
        relative = ipw_dir.relative_to(ROOT).as_posix()
        sources.append({
            "batches": f"{relative}/batches.jsonl",
            "traces": f"{relative}/traces.jsonl",
            "subset_manifest": f"{relative}/subset_manifest.json",
            "raw_meter": f"{relative}/telemetry_samples.jsonl",
            "batch_key_prefix": f"{run_dir.name}:{batch_id}:",
            "expected_batch_size": len(batch),
        })
    provenance_path = (run_dir / "provenance.json").relative_to(ROOT).as_posix()
    return {"profiles": [{
        "profile": deployment["analysis_profile"],
        "source_mode": "ipw-run",
        "run_provenance": provenance_path,
        "sources": sources,
    }]}


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def validate_server_manifest(path: Path, *, deployment: dict[str, Any],
                             revision: str, protocol: dict[str, Any],
                             port: int) -> dict[str, Any]:
    """Validate local vLLM manifest fields against the selected deployment."""
    raw = _json_object(path)
    fields = ("model", "model_revision", "server_pid", "runtime", "vllm_version",
              "runtime_artifact_sha256", "host", "port", "tensor_parallel_size",
              "max_model_len", "max_num_seqs")
    claim = {key: raw.get(key) for key in fields}
    if claim["model"] != deployment["model"] or claim["model_revision"] != revision:
        raise ValueError("server manifest model or revision disagrees with deployment")
    if claim["runtime"] != "vllm":
        raise ValueError("server manifest must attest a vLLM runtime")
    version = claim["vllm_version"]
    version_match = (re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[.+-][\w.-]+)?", version)
                     if isinstance(version, str) else None)
    if version_match is None:
        raise ValueError("server manifest needs a vLLM runtime version")
    if (claim["model"] not in LOCKED_VLLM_AUTOSTART_MODELS
            and tuple(map(int, version_match.groups())) <= (0, 11, 0)):
        raise ValueError("Select a vLLM server runtime newer than 0.11.0 for this deployment")
    digest = claim["runtime_artifact_sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise ValueError("server manifest needs a SHA-256 of its server image or wheel")
    claim["runtime_artifact_sha256"] = digest.lower()
    expected = {
        "host": "127.0.0.1", "port": port,
        "tensor_parallel_size": deployment["tensor_parallel"],
        "max_model_len": protocol["context_tokens"],
        "max_num_seqs": server_max_num_seqs(deployment, protocol),
    }
    for key, value in expected.items():
        if claim[key] != value or (isinstance(value, int) and type(claim[key]) is not int):
            raise ValueError(f"server manifest {key} disagrees with deployment")
    pid = claim["server_pid"]
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError("server manifest needs a positive local server_pid")
    return claim


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected nonempty JSONL objects")
    return rows


def write_private_json(path: Path, value: dict[str, Any]) -> None:
    """Replace run metadata atomically so interruption leaves a valid prior record."""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verify_batch_artifacts(ipw_dir: Path, scheduled: list[dict[str, Any]], *,
                           model: str, protocol: dict[str, Any],
                           deployment: dict[str, Any] | None = None) -> dict[str, Any]:
    """Attest that one IPW invocation recorded every scheduled request and meter component.

    Task-level failures remain observations. Infrastructure, input, and judge
    failures invalidate the batch even when IPW wrote a trace for every request.
    """
    names = ("summary.json", "subset_manifest.json", "traces.jsonl", "batches.jsonl",
             "telemetry_samples.jsonl", "telemetry_samples_manifest.json")
    paths = {name: ipw_dir / name for name in names}
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError(f"{path}: required nonempty IPW artifact is missing")
    summary = _json_object(paths["summary.json"])
    subset = _json_object(paths["subset_manifest.json"])
    raw_manifest = _json_object(paths["telemetry_samples_manifest.json"])
    config = summary.get("config") or {}
    judge = operational_judge(deployment or {}, protocol)
    expected_config = {
        "agent": protocol["agent"], "model": model, "dataset": protocol["dataset"],
        "max_queries": len(scheduled), "concurrency": protocol["concurrency"],
        "telemetry_accounting": "batch", "telemetry_backend": "energy-monitor",
        "telemetry_interval": protocol["telemetry_interval_seconds"],
        "eval_client": judge["client"], "eval_model": judge["model"],
    }
    if not isinstance(config, dict) or any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError(f"{paths['summary.json']}: IPW configuration differs from deployment")
    records = subset.get("records")
    if (subset.get("schema_version") != 1 or subset.get("dataset") != protocol["dataset"]
            or subset.get("model") != model or subset.get("agent") != protocol["agent"]
            or not isinstance(records, list) or len(records) != len(scheduled)
            or subset.get("subset_size") != len(records)):
        raise ValueError(f"{paths['subset_manifest.json']}: invalid scheduled subset")
    subset_hash = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if subset.get("subset_hash") != subset_hash:
        raise ValueError(f"{paths['subset_manifest.json']}: subset hash mismatch")
    expected_instances = {(item["task_id"], item["instance_id"]) for item in scheduled}
    actual_instances = {(str((row.get("stable_ids") or {}).get("task_id", "")),
                         str((row.get("stable_ids") or {}).get("instance_id", "")))
                        for row in records}
    query_ids = [str(row.get("query_id", "")) for row in records]
    if (actual_instances != expected_instances or len(actual_instances) != len(records)
            or len(set(query_ids)) != len(records) or any(not value for value in query_ids)):
        raise ValueError(f"{paths['subset_manifest.json']}: task instances or query IDs differ from schedule")
    traces = _jsonl_objects(paths["traces.jsonl"])
    batches = _jsonl_objects(paths["batches.jsonl"])
    if len(batches) != 1:
        raise ValueError(f"{paths['batches.jsonl']}: expected exactly one physical batch")
    batch = batches[0]
    batch_id = batch.get("batch_id")
    if (batch.get("telemetry_accounting") != "batch" or batch.get("batch_size") != len(scheduled)
            or len(batch.get("query_ids") or []) != len(scheduled)
            or set(batch["query_ids"]) != set(query_ids)):
        raise ValueError(f"{paths['batches.jsonl']}: batch query roster mismatch")
    if (len(traces) != len(scheduled) or {row.get("query_id") for row in traces} != set(query_ids)
            or any(row.get("batch_id") != batch_id for row in traces)):
        raise ValueError(f"{paths['traces.jsonl']}: trace roster or batch identity mismatch")
    start, end = batch.get("batch_prompt_start_time_s"), batch.get("batch_prompt_end_time_s")
    if (not isinstance(start, (int, float)) or not isinstance(end, (int, float))
            or not math.isfinite(start) or not math.isfinite(end) or end <= start):
        raise ValueError(f"{paths['batches.jsonl']}: invalid measurement interval")
    sample_count = batch.get("telemetry_sample_count")
    if type(sample_count) is not int or sample_count < 2:
        raise ValueError(f"{paths['batches.jsonl']}: fewer than two batch telemetry samples")
    finish_offsets = batch.get("prompt_finish_offsets_s")
    if not isinstance(finish_offsets, dict) or set(finish_offsets) != set(query_ids):
        raise ValueError(f"{paths['batches.jsonl']}: finish-offset roster differs from batch queries")
    for trace in traces:
        trace_start, trace_end = trace.get("prompt_start_time_s"), trace.get("prompt_end_time_s")
        if (not isinstance(trace.get("completed"), bool)
                or not isinstance(trace_start, (int, float))
                or not isinstance(trace_end, (int, float))
                or not math.isfinite(trace_start) or not math.isfinite(trace_end)
                or not start <= trace_start < trace_end <= end):
            raise ValueError(f"{paths['traces.jsonl']}: incomplete request interval or status")
        tool_calls = trace.get("tool_call_count")
        if type(tool_calls) is not int or tool_calls <= 0:
            raise ValueError(f"{paths['traces.jsonl']}: request recorded no OpenHands tool calls")
        finish_offset = finish_offsets[trace["query_id"]]
        trace_offset = trace.get("batch_prompt_finish_offset_s")
        if (not isinstance(finish_offset, (int, float)) or isinstance(finish_offset, bool)
                or not math.isfinite(finish_offset) or not 0 <= finish_offset <= end - start
                or not isinstance(trace_offset, (int, float)) or isinstance(trace_offset, bool)
                or not math.isfinite(trace_offset)
                or not math.isclose(finish_offset, trace_end - start, abs_tol=1e-6)
                or not math.isclose(trace_offset, finish_offset, abs_tol=1e-6)):
            raise ValueError(f"{paths['batches.jsonl']}: request finish offset differs from trace")
        score = trace.get("score_metadata")
        if not isinstance(score, dict):
            raise ValueError(f"{paths['traces.jsonl']}: missing GDPVal scoring metadata")
        failure_type = score.get("failure_type")
        reasons = (trace.get("unscorable_reason"), score.get("reason"))
        infrastructure_reasons = {
            "scoring_error", "no_rubric", "no_llm_client_for_judging",
            "visual_evidence_unavailable", "infra_reference_error",
            "agent_or_llm_timeout", "agent_or_llm_connection_error",
            "agent_harness_error",
        }
        invalid_failure_type = (
            failure_type is not None
            and failure_type not in {"agent_error", "harness_validation"}
        )
        invalid_harness_result = (
            failure_type == "harness_validation"
            and score.get("reason") != "deliverable_contract_invalid"
        )
        invalid_completion = (
            trace["completed"] is False and failure_type != "agent_error"
            or trace["completed"] is True and failure_type == "agent_error"
        )
        infrastructure_reason = any(
            isinstance(reason, str)
            and (reason in infrastructure_reasons or reason.startswith("infra_"))
            for reason in reasons
        )
        if (invalid_failure_type or invalid_harness_result or invalid_completion
                or infrastructure_reason):
            raise ValueError(f"{paths['traces.jsonl']}: GDPVal input or judging infrastructure failed")
        criteria = score.get("per_criterion")
        if isinstance(criteria, list) and any(
            not isinstance(item, dict) or item.get("error") for item in criteria
        ):
            raise ValueError(f"{paths['traces.jsonl']}: GDPVal deliverable evaluation failed")
        if trace["completed"] is True and type(trace.get("is_resolved")) is not bool:
            raise ValueError(f"{paths['traces.jsonl']}: completed GDPVal request lacks a judgment")
    for component in ("gpu", "cpu", "dram"):
        value = batch.get(f"batch_{component}_energy_joules")
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value <= 0
                or (batch.get("energy_window_coverage") or {}).get(component) is not True):
            raise ValueError(f"{paths['batches.jsonl']}: incomplete measured {component} energy")
    totals = summary.get("totals") or {}
    batch_totals = summary.get("batch_telemetry") or {}
    completed = sum(row.get("completed") is True for row in traces)
    if (totals.get("queries") != len(scheduled) or totals.get("completed") != completed
            or batch_totals.get("batch_count") != 1
            or batch_totals.get("query_count") != len(scheduled)
            or batch_totals.get("batches") != [batch]
            or not isinstance(batch_id, str) or not batch_id):
        raise ValueError(f"{paths['summary.json']}: incomplete or inconsistent IPW summary")
    for component in ("gpu", "cpu", "dram"):
        aggregate = batch_totals.get(f"total_batch_{component}_energy_joules")
        measured = batch[f"batch_{component}_energy_joules"]
        if (not isinstance(aggregate, (int, float)) or not math.isfinite(aggregate)
                or not math.isclose(aggregate, measured, rel_tol=1e-9)):
            raise ValueError(f"{paths['summary.json']}: {component} total differs from measured batch")
    raw_path = paths["telemetry_samples.jsonl"]
    digest = hashlib.sha256()
    count = 0
    last_sample_time = -math.inf
    with raw_path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            sample_time = row.get("sample_time_s") if isinstance(row, dict) else None
            if (not isinstance(row, dict) or row.get("schema_version") != 1
                    or not isinstance(sample_time, (int, float))
                    or not math.isfinite(sample_time) or sample_time < last_sample_time):
                raise ValueError(f"{raw_path}: invalid raw meter sample")
            last_sample_time = sample_time
            count += 1
    if (raw_manifest.get("schema_version") != 1 or raw_manifest.get("path") != raw_path.name
            or raw_manifest.get("sample_count") != count or count < sample_count
            or raw_manifest.get("sha256") != digest.hexdigest()
            or config.get("telemetry_samples") != raw_manifest):
        raise ValueError(f"{raw_path}: raw meter manifest or summary linkage mismatch")
    return {
        "query_count": len(scheduled), "completed_queries": completed,
        "artifact_sha256": {name: sha256_file(path) for name, path in paths.items()},
    }


def installed_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def source_revision() -> str:
    """Require a committed source tree for an executable scientific run."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
        check=True, capture_output=True, text=True,
    ).stdout
    if status:
        raise RuntimeError("commit or remove source-tree changes before executing a model run")
    return commit


def gpu_provenance(deployment: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for accelerator validation") from exc
    visible = torch.cuda.device_count()
    required = deployment["gpu_count"]
    if visible != required:
        raise RuntimeError(f"expected {required} visible GPUs, found {visible}")
    devices = []
    expected = deployment["accelerator"].casefold().replace("x", "")
    for index in range(visible):
        properties = torch.cuda.get_device_properties(index)
        name = properties.name
        if expected not in name.casefold().replace("x", ""):
            raise RuntimeError(f"GPU {index} is {name!r}, expected {deployment['accelerator']}")
        devices.append({"index": index, "name": name, "memory_bytes": properties.total_memory})
    return devices


def nvidia_selection_provenance(deployment: dict[str, Any]) -> dict[str, Any]:
    """Require unambiguous physical UUIDs whenever NVIDIA devices are a subset."""
    selector = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    evidence: dict[str, Any] = {"cuda_visible_devices": selector or None}
    if deployment["accelerator"] == "MI350X":
        return evidence
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            uuids = []
            for index in range(count):
                value = pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(index))
                uuids.append(value.decode() if isinstance(value, bytes) else value)
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        raise RuntimeError("cannot verify the NVIDIA physical GPU UUIDs") from exc
    if count < 1 or any(not isinstance(value, str) or not value.startswith("GPU-") for value in uuids):
        raise RuntimeError("NVML did not report full physical NVIDIA GPU UUIDs")
    required = deployment["gpu_count"]
    if count < required:
        raise RuntimeError(f"NVML reports {count} physical GPUs, expected at least {required}")
    selectors = [part.strip() for part in selector.split(",") if part.strip()]
    if count > required and (
        len(selectors) != required or len(set(selectors)) != required
        or any(value not in uuids for value in selectors)
    ):
        raise RuntimeError("a NVIDIA GPU subset requires exactly the selected number of "
                           "distinct full GPU-* UUIDs in CUDA_VISIBLE_DEVICES")
    evidence["nvml_physical_gpu_count"] = count
    evidence["selected_gpu_uuids"] = selectors if count > required else uuids
    evidence["physical_gpu_subset"] = count > required
    return evidence


def verify_existing_server_gpu_selection(pid: int, expected_selectors: str) -> None:
    """Confirm every external server process uses the measured GPU UUIDs."""
    import psutil
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
        for process in processes:
            selected = process.environ().get("CUDA_VISIBLE_DEVICES", "").strip()
            if selected != expected_selectors:
                raise RuntimeError(
                    f"server process {process.pid} CUDA_VISIBLE_DEVICES differs "
                    "from the measured GPU UUIDs"
                )
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as exc:
        raise RuntimeError(f"cannot verify server PID {pid} GPU selection: {exc}") from exc


def wait_for_server(port: int, process: subprocess.Popen[bytes] | None, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {process.returncode}; inspect server.log")
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError("vLLM did not become healthy before the timeout")


def validate_served_model(port: int, model: str) -> None:
    with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as response:
        payload = json.load(response)
    names = [item.get("id") for item in payload.get("data", [])]
    if model not in names:
        raise RuntimeError(f"served model {model!r} absent from /v1/models: {names}")


def verify_server_port_owner(pid: int, port: int) -> None:
    """Require the declared existing server process tree to own the local listener."""
    import psutil
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
        for process in processes:
            for connection in process.net_connections(kind="tcp"):
                if (connection.status == psutil.CONN_LISTEN and connection.laddr
                        and connection.laddr.port == port
                        and connection.laddr.ip == "127.0.0.1"):
                    return
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as exc:
        raise RuntimeError(f"cannot verify declared server PID {pid}: {exc}") from exc
    raise RuntimeError(f"server PID {pid} and its children do not own loopback port {port}")


def require_free_loopback_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"loopback port {port} is already in use") from exc


def choose_monitor_port(*, exclude: int) -> int:
    """Give the owned meter a private loopback endpoint for this run."""
    for _ in range(10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = int(probe.getsockname()[1])
        if port != exclude:
            return port
    raise RuntimeError("could not allocate a distinct energy-monitor port")


def controlled_run_environment(inherited: dict[str, str], *, protocol: dict[str, Any],
                               monitor_port: int, gpu_count: int,
                               accelerator: str,
                               deployment: dict[str, Any] | None = None) -> dict[str, str]:
    """Discard inherited IPW experiment knobs while preserving the judge credential."""
    env = {key: value for key, value in inherited.items()
           if not key.startswith("IPW_") or key == "IPW_EVAL_API_KEY"}
    env.update({
        "IPW_OPENHANDS_MAX_OUTPUT_TOKENS": str(protocol["output_tokens"]),
        "IPW_OPENHANDS_TEMPERATURE": "0.0",
        "IPW_OPENHANDS_TOP_P": "1.0",
        "IPW_ENERGY_MONITOR_INTERVAL_MS": str(round(protocol["telemetry_interval_seconds"] * 1000)),
        "IPW_REQUIRE_TOTAL_ENERGY": "1",
        "IPW_REQUIRE_OWNED_MONITOR": "1",
        "IPW_ENERGY_MONITOR_TARGET": f"127.0.0.1:{monitor_port}",
        "IPW_EXPECTED_GPU_COUNT": str(gpu_count),
        "IPW_EXPECTED_METER_PLATFORM": "amd" if accelerator == "MI350X" else "nvidia",
        "IPW_REQUIRE_GDPVAL_REFERENCES": "1",
        "IPW_REQUIRE_GDPVAL_JUDGE": "1",
    })
    env.update((deployment or {}).get("ipw_environment", {}))
    return env


def require_runtime(server: list[str] | None, run_commands: list[list[str]]) -> Path:
    runner = Path(run_commands[0][0])
    if not runner.is_file() or not os.access(runner, os.X_OK):
        raise RuntimeError(f"IPW launcher is unavailable: {runner}; install the locked project first")
    from ipw.telemetry._binaries import binary_path
    meter = binary_path("energy-monitor")
    if not os.access(meter, os.X_OK):
        raise RuntimeError(f"energy monitor is not executable: {meter}")
    if server and installed_version("vllm") != LOCKED_VLLM_VERSION:
        raise RuntimeError(f"Install vllm=={LOCKED_VLLM_VERSION} for launcher startup, "
                           "or connect a running server with --existing-server")
    return meter


def stop_process_group(process: subprocess.Popen[bytes]) -> None:
    """Stop vLLM or IPW and their child workers after failure or interruption."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_batch_command(command: list[str], *, env: dict[str, str]) -> None:
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    try:
        result = process.wait()
        if result != 0:
            raise subprocess.CalledProcessError(result, command)
    except BaseException:
        stop_process_group(process)
        raise
    stop_process_group(process)


def _terminate_on_signal(signum: int, _frame: Any) -> None:
    raise SystemExit(128 + signum)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=config["deployments"])
    parser.add_argument("--tasks-file", type=Path, required=True,
                        help="External 880-instance agent roster; 220-task fixture is dry-run only")
    parser.add_argument("--output-dir", type=Path, default=RUNS)
    parser.add_argument("--attempts", type=int, default=4, help="Attempts per task (1-4)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-revision", help="Pinned checkpoint revision")
    parser.add_argument("--vllm-arg", action="append", default=[], help="Repeat for additional vLLM args")
    parser.add_argument("--existing-server", action="store_true")
    parser.add_argument("--server-manifest", type=Path,
                        help="Required with --existing-server; JSON with model and model_revision")
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--execute", action="store_true", help="Run; otherwise print a dry run")
    args = parser.parse_args()
    if not 1 <= args.attempts <= 4:
        parser.error("--attempts must be between 1 and 4")
    if args.execute and args.attempts != 4:
        parser.error("deployment execution requires four attempts per task")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.startup_timeout <= 0:
        parser.error("--startup-timeout must be positive")
    if args.existing_server != (args.server_manifest is not None):
        parser.error("--existing-server and --server-manifest must be supplied together")
    if args.existing_server and args.vllm_arg:
        parser.error("--vllm-arg applies only when this launcher starts vLLM")
    try:
        validate_vllm_args(args.vllm_arg)
    except ValueError as exc:
        parser.error(str(exc))
    deployment = config["deployments"][args.profile]
    if not args.existing_server and deployment["model"] not in LOCKED_VLLM_AUTOSTART_MODELS:
        parser.error("start vLLM, then pass --existing-server and --server-manifest")
    revision = args.model_revision or deployment.get("model_revision")
    if not revision or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        parser.error("a pinned 40-character commit SHA is required as --model-revision")
    if deployment.get("model_revision") and revision != deployment["model_revision"]:
        parser.error("--model-revision differs from the pinned deployment revision")
    tasks_file = args.tasks_file.resolve(strict=True)
    raw_inventory = json.loads(tasks_file.read_text(encoding="utf-8"))
    if isinstance(raw_inventory, list) and len(raw_inventory) == 880:
        batches = load_task_instances(tasks_file)
        task_count = 220
    else:
        if args.execute:
            parser.error("execution requires the recorded 880-instance agent roster")
        tasks = load_task_inventory(tasks_file)
        batches = occupation_batches(tasks, args.attempts)
        task_count = len(tasks)
    if args.execute:
        validate_recorded_task_hash(tasks_file)
    if args.attempts == 4 and any(len(batch) != 20 for batch in batches):
        parser.error("expected 20 requests per occupation batch")
    judge = operational_judge(deployment, config["protocol"])
    judge_keys = ("OPENAI_API_KEY",) if judge["client"] == "openai" else ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if args.execute and not (os.getenv("IPW_EVAL_API_KEY") or any(os.getenv(key) for key in judge_keys)):
        parser.error(f"{judge['client']} operational judge key is required")
    # Keep the logical repository path in provenance even if runs/ is mounted
    # or symlinked to durable external storage.
    output_root = Path(os.path.abspath(args.output_dir))
    if (not output_root.is_relative_to(RUNS)
            or not output_root.resolve().is_relative_to(RUNS.resolve())):
        parser.error("outputs must be inside ignored runs/")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = output_root / args.profile / run_id
    server, run_commands = commands(args.profile, run_dir, batches,
                                    port=args.port, revision=revision,
                                    vllm_args=args.vllm_arg,
                                    existing_server=args.existing_server)
    print(f"{len(batches)} occupation batches, {sum(map(len, batches))} requests")
    print(f"output: {run_dir}")
    if not args.execute:
        for command in ([server] if server else []) + run_commands:
            print(shlex.join(command))
        return

    server_claim = None
    server_manifest_sha256 = None
    if args.server_manifest is not None:
        try:
            server_claim = validate_server_manifest(
                args.server_manifest, deployment=deployment, revision=revision,
                protocol=config["protocol"], port=args.port,
            )
        except ValueError as exc:
            parser.error(str(exc))
        server_manifest_sha256 = sha256_file(args.server_manifest)
    git_commit = source_revision()
    meter_binary = require_runtime(server, run_commands)
    monitor_port = choose_monitor_port(exclude=args.port)
    if server:
        require_free_loopback_port(args.port)
    else:
        verify_server_port_owner(server_claim["server_pid"], args.port)
    hardware = gpu_provenance(deployment)
    gpu_selection = nvidia_selection_provenance(deployment)
    if args.existing_server and gpu_selection.get("physical_gpu_subset"):
        verify_existing_server_gpu_selection(server_claim["server_pid"],
                                             gpu_selection["cuda_visible_devices"])
    run_env = controlled_run_environment(
        os.environ.copy(), protocol=config["protocol"], monitor_port=monitor_port,
        gpu_count=deployment["gpu_count"], accelerator=deployment["accelerator"],
        deployment=deployment,
    )
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    schedule_dir = run_dir / "schedule"
    schedule_dir.mkdir(mode=0o700)
    schedule_hashes: dict[str, str] = {}
    for batch in batches:
        batch_id = batch[0]["occupation_batch_id"]
        schedule_path = schedule_dir / f"{batch_id}.json"
        schedule_path.write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8")
        schedule_hashes[batch_id] = sha256_file(schedule_path)
    provenance = {
        "profile": args.profile,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(CONFIG),
        "runner_sha256": sha256_file(Path(__file__)),
        "python_lock_sha256": sha256_file(ROOT / "intelligence-per-watt" / "uv.lock"),
        "energy_monitor_binary_sha256": sha256_file(meter_binary),
        "energy_monitor_target": f"127.0.0.1:{monitor_port}",
        "controlled_ipw_environment": {
            key: value for key, value in run_env.items()
            if key.startswith("IPW_") and key != "IPW_EVAL_API_KEY"
        },
        "git_commit": git_commit,
        "task_inventory_sha256": sha256_file(tasks_file),
        "schedule_sha256": schedule_hashes,
        "task_count": task_count,
        "occupation_batches": len(batches),
        "attempts_per_task": args.attempts,
        "model": deployment["model"],
        "model_revision": revision,
        "dataset": "openai/gdpval",
        "dataset_revision": config["protocol"]["dataset_revision"],
        "rubric_judge": judge,
        "rubric_visible_to_agent": False,
        "telemetry_interval_seconds": config["protocol"]["telemetry_interval_seconds"],
        "hardware": hardware,
        "gpu_selection": gpu_selection,
        "strict_dedicated_hardware_preflight": deployment["accelerator"] != "MI350X",
        "versions": {name: installed_version(name) for name in ("ipw", "vllm", "torch", "datasets")},
        "python_version": sys.version,
        "platform": platform.platform(),
        "server_command_sha256": (hashlib.sha256(json.dumps(server).encode()).hexdigest()
                                  if server else None),
        "batch_command_sha256": [hashlib.sha256(json.dumps(command).encode()).hexdigest()
                                 for command in run_commands],
        "server_argv": server,
        "batch_argv": run_commands,
        "server_claim": server_claim,
        "server_manifest_sha256": server_manifest_sha256,
        "verified_batches": [],
        "status": "running",
    }
    provenance_path = run_dir / "provenance.json"
    write_private_json(provenance_path, provenance)
    process = None
    log = None
    previous_sigterm = signal.signal(signal.SIGTERM, _terminate_on_signal)
    try:
        if server:
            log = (run_dir / "server.log").open("xb")
            process = subprocess.Popen(server, cwd=ROOT, env=run_env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            run_env["IPW_PREFLIGHT_ALLOWED_GPU_PIDS"] = str(process.pid)
        else:
            run_env["IPW_PREFLIGHT_ALLOWED_GPU_PIDS"] = str(server_claim["server_pid"])
        provenance["server_pid"] = int(run_env["IPW_PREFLIGHT_ALLOWED_GPU_PIDS"])
        write_private_json(provenance_path, provenance)
        wait_for_server(args.port, process, args.startup_timeout)
        if args.existing_server:
            verify_server_port_owner(server_claim["server_pid"], args.port)
            if gpu_selection.get("physical_gpu_subset"):
                verify_existing_server_gpu_selection(
                    server_claim["server_pid"], gpu_selection["cuda_visible_devices"]
                )
        validate_served_model(args.port, deployment["model"])
        for batch, command in zip(batches, run_commands, strict=True):
            batch_id = batch[0]["occupation_batch_id"]
            if args.existing_server and gpu_selection.get("physical_gpu_subset"):
                verify_server_port_owner(server_claim["server_pid"], args.port)
                verify_existing_server_gpu_selection(
                    server_claim["server_pid"], gpu_selection["cuda_visible_devices"]
                )
            schedule_path = schedule_dir / f"{batch_id}.json"
            if sha256_file(schedule_path) != schedule_hashes[batch_id]:
                raise RuntimeError(f"{batch_id}: schedule changed before execution")
            provenance["active_batch"] = batch_id
            write_private_json(provenance_path, provenance)
            run_batch_command(command, env=run_env)
            if sha256_file(schedule_path) != schedule_hashes[batch_id]:
                raise RuntimeError(f"{batch_id}: schedule changed during execution")
            verified = verify_batch_artifacts(
                ipw_run_dir(deployment["model"], run_dir, batch_id), batch,
                model=deployment["model"], protocol=config["protocol"], deployment=deployment,
            )
            provenance["verified_batches"].append({"batch_id": batch_id, **verified})
            provenance.pop("active_batch", None)
            write_private_json(provenance_path, provenance)
        source_config_path = run_dir / "energy_sources.json"
        write_private_json(source_config_path, energy_source_config(args.profile, run_dir, batches))
        provenance["energy_sources_sha256"] = sha256_file(source_config_path)
        provenance["status"] = "completed"
    except BaseException:
        provenance["status"] = "failed"
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        cleanup_error = None
        try:
            if process is not None:
                stop_process_group(process)
        except Exception as exc:
            cleanup_error = exc
            provenance["status"] = "failed"
        finally:
            try:
                if log is not None:
                    log.close()
                provenance["finished_utc"] = datetime.now(timezone.utc).isoformat()
                write_private_json(provenance_path, provenance)
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)
        if cleanup_error is not None:
            raise cleanup_error


if __name__ == "__main__":
    main()
