#!/usr/bin/env python3
"""Stage measured energy components from IPW batch and trace JSONL files.

The external config is JSON: {"profiles": [{"profile": "glm52_h200",
"source_mode": "ipw-run", "run_provenance": "runs/.../provenance.json",
"sources": [{"batches": "relative/path/batches.jsonl",
"traces": "relative/path/traces.jsonl", "subset_manifest": "relative/path/subset_manifest.json",
"query_map": "optional/relative/map.csv",
"batch_key_prefix": "GDPVal:", "raw_meter": "relative/raw/samples.jsonl"}]}]}.
A query map, when needed, contains
query_id, task_id, repeat. All paths must be relative to the repository root.
The output CSVs are consumed by energy_boundary.py. Measurement validation
requires GPU, CPU package, and DRAM energy for every batch.
For launched runs, the raw samples must match IPW's adjacent sample manifest,
and that manifest must match the IPW summary configuration. This establishes
sample provenance; it does not independently recompute batch joules from the
samples. Batch joules are checked against IPW's batches.jsonl by energy_boundary.
Set source_mode to "ipw-run" or "historical-archive" for the selected input format.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from energy_boundary import BATCH_COLUMNS, REQUEST_COLUMNS, ROOT, sha256, source_file


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def required_energy(row: dict, field: str, source: Path) -> float:
    value = row.get(field)
    if value is None:
        raise ValueError(f"{source}: {field} is absent or null; total energy cannot be derived")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{source}: {field} must be positive measured joules")
    return number


def source_info(relative: str) -> tuple[Path, str]:
    path = source_file(relative)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, sha256(path)


def raw_meter_provenance(raw_path: Path, raw_hash: str, *, require_summary: bool) -> tuple[Path | None, Path | None]:
    """Verify IPW's recorder checksum/count and its summary linkage."""
    manifest_path = raw_path.with_name("telemetry_samples_manifest.json")
    summary_path = raw_path.with_name("summary.json")
    if not manifest_path.is_file():
        if require_summary:
            raise FileNotFoundError(f"IPW raw telemetry manifest is required: {manifest_path}")
        return None, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("path") != raw_path.name
            or manifest.get("sha256") != raw_hash):
        raise ValueError(f"{manifest_path}: raw telemetry manifest checksum or path mismatch")
    count = 0
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                raise ValueError(f"{raw_path}: blank raw telemetry record")
            sample = json.loads(line)
            sample_time = sample.get("sample_time_s")
            if (sample.get("schema_version") != 1 or isinstance(sample_time, bool)
                    or not isinstance(sample_time, (int, float)) or not math.isfinite(sample_time)):
                raise ValueError(f"{raw_path}: invalid raw telemetry sample")
            count += 1
    if count <= 0 or manifest.get("sample_count") != count:
        raise ValueError(f"{manifest_path}: raw telemetry sample count mismatch")
    if not summary_path.is_file():
        if require_summary:
            raise FileNotFoundError(f"IPW summary is required: {summary_path}")
        return manifest_path, None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("config") or {}).get("telemetry_samples") != manifest:
        raise ValueError(f"{summary_path}: raw telemetry manifest differs from IPW summary config")
    return manifest_path, summary_path


def subset_mapping(path: Path) -> dict[str, tuple[str, int]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list) or not records or manifest.get("subset_size") != len(records):
        raise ValueError(f"{path}: invalid subset manifest record count")
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if digest != manifest.get("subset_hash"):
        raise ValueError(f"{path}: subset record hash mismatch")
    parsed = []
    for record in records:
        stable = record.get("stable_ids") or {}
        match = re.search(r"::rep(\d{2})$", str(stable.get("instance_id", "")))
        if not match or not stable.get("task_id"):
            raise ValueError(f"{path}: missing task ID or repeat-bearing instance ID")
        parsed.append((str(record["query_id"]), str(stable["task_id"]), int(match.group(1))))
    labels = {repeat for _, _, repeat in parsed}
    if labels == {0, 1, 2, 3}:
        offset = 0
    elif labels == {1, 2, 3, 4}:
        offset = 1
    else:
        raise ValueError(f"{path}: cannot determine four-repeat numbering from {sorted(labels)}")
    mapping = {query_id: (task_id, repeat - offset) for query_id, task_id, repeat in parsed}
    if len(mapping) != len(records):
        raise ValueError(f"{path}: duplicate query ID")
    by_task: dict[str, set[int]] = {}
    for task_id, repeat in mapping.values():
        by_task.setdefault(task_id, set()).add(repeat)
    if any(repeats != {0, 1, 2, 3} for repeats in by_task.values()):
        raise ValueError(f"{path}: incomplete four-repeat task mapping")
    return mapping


def run_provenance(profile_spec: dict) -> tuple[str, str, str, str, Path | None]:
    """Validate the selected source mode and its run metadata."""
    mode = profile_spec.get("source_mode")
    if mode == "historical-archive":
        if profile_spec.get("run_provenance"):
            raise ValueError("Use source_mode=ipw-run when specifying run_provenance")
        return "", "", "", "", None
    if mode != "ipw-run":
        raise ValueError("Each profile needs source_mode=ipw-run or historical-archive")
    relative = profile_spec.get("run_provenance")
    if not relative:
        raise ValueError("IPW run requires run_provenance")
    path, digest = source_info(str(relative))
    record = json.loads(path.read_text(encoding="utf-8"))
    original_relative = (path.parent / "energy_sources.json").relative_to(ROOT).as_posix()
    original_path, original_hash = source_info(original_relative)
    if record.get("status") != "completed" or record.get("energy_sources_sha256") != original_hash:
        raise ValueError(f"{path}: incomplete run or energy source config hash mismatch")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    if original.get("profiles") != [profile_spec]:
        raise ValueError(f"{original_path}: profile sources differ from run config")
    deployments_path = ROOT / "reproduction/deployments.json"
    deployments = json.loads(deployments_path.read_text(encoding="utf-8"))
    deployment_id = record.get("profile")
    deployment = deployments["deployments"].get(deployment_id)
    if not deployment or deployment.get("analysis_profile") != profile_spec.get("profile"):
        raise ValueError(f"{path}: deployment profile does not match analysis profile")
    protocol = deployments["protocol"]
    expected = {
        "config_sha256": sha256(deployments_path),
        "runner_sha256": sha256(ROOT / "reproduction/run_deployment.py"),
        "python_lock_sha256": sha256(ROOT / "intelligence-per-watt/uv.lock"),
        "task_count": 220,
        "occupation_batches": 44,
        "attempts_per_task": 4,
        "model": deployment["model"],
        "dataset": "openai/gdpval",
        "dataset_revision": protocol["dataset_revision"],
        "rubric_visible_to_agent": False,
        "telemetry_interval_seconds": protocol["telemetry_interval_seconds"],
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"{path}: run provenance {field} mismatch")
    revision = str(record.get("model_revision") or "")
    if not revision or (deployment.get("model_revision") and revision != deployment["model_revision"]):
        raise ValueError(f"{path}: missing or wrong model revision")
    operational_judge = deployment.get(
        "operational_judge", {"client": "gemini", "model": protocol["operational_judge_model"]},
    )
    if record.get("rubric_judge") != operational_judge:
        raise ValueError(f"{path}: judge protocol mismatch")
    task_manifest = json.loads((ROOT / "reproduction/input_manifest.json").read_text(encoding="utf-8"))
    task_hashes = [entry.get("sha256") for entry in task_manifest["inputs"]
                   if entry["path"] == "cluster/gdpval_occupation_repeat4_instances.json"]
    if len(task_hashes) != 1 or record.get("task_inventory_sha256") != task_hashes[0]:
        raise ValueError(f"{path}: recorded task inventory SHA-256 mismatch")
    hardware = record.get("hardware")
    if not isinstance(hardware, list) or len(hardware) != deployment["gpu_count"]:
        raise ValueError(f"{path}: accelerator count mismatch")
    expected_gpu = deployment["accelerator"].casefold().replace("x", "")
    if any(expected_gpu not in str(device.get("name", "")).casefold().replace("x", "")
           or int(device.get("memory_bytes") or 0) <= 0 for device in hardware):
        raise ValueError(f"{path}: accelerator identity mismatch")
    return str(relative), digest, original_relative, original_hash, path.parent


def build_from_ipw(config_path: Path, output_dir: Path) -> tuple[int, int]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    profiles = config.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("Config needs a nonempty profiles list")
    batches_out: list[dict] = []
    requests_out: list[dict] = []
    for profile_spec in profiles:
        profile = str(profile_spec["profile"])
        provenance_relative, provenance_hash, source_config_relative, source_config_hash, run_dir = run_provenance(profile_spec)
        sources = profile_spec.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"{profile}: missing IPW sources")
        for shard_index, shard in enumerate(sources):
            batch_relative = str(shard["batches"])
            trace_relative = str(shard["traces"])
            batch_path, batch_hash = source_info(batch_relative)
            trace_path, trace_hash = source_info(trace_relative)
            if run_dir is not None and (not batch_path.is_relative_to(run_dir) or not trace_path.is_relative_to(run_dir)):
                raise ValueError(f"{profile}: IPW sources are outside the validated run directory")
            raw_meter_relative = str(shard.get("raw_meter") or "")
            if run_dir is not None and (not raw_meter_relative or not source_file(raw_meter_relative).is_relative_to(run_dir)):
                raise ValueError(f"{profile}: launched IPW run requires raw meter samples inside its run directory")
            raw_meter_hash = source_info(raw_meter_relative)[1] if raw_meter_relative else ""
            raw_manifest_relative = summary_relative = ""
            raw_manifest_hash = summary_hash = ""
            if raw_meter_relative:
                raw_path = source_file(raw_meter_relative)
                raw_manifest_path, summary_path = raw_meter_provenance(
                    raw_path, raw_meter_hash, require_summary=run_dir is not None,
                )
                if raw_manifest_path is not None:
                    raw_manifest_relative = raw_manifest_path.relative_to(ROOT).as_posix()
                    raw_manifest_hash = sha256(raw_manifest_path)
                if summary_path is not None:
                    summary_relative = summary_path.relative_to(ROOT).as_posix()
                    summary_hash = sha256(summary_path)
            mapping: dict[str, tuple[str, int]] = {}
            subset_relative = shard.get("subset_manifest")
            map_relative = subset_relative or shard.get("query_map")
            map_hash = ""
            if map_relative:
                map_path, map_hash = source_info(str(map_relative))
                if subset_relative:
                    mapping = subset_mapping(map_path)
                else:
                    with map_path.open(newline="", encoding="utf-8") as handle:
                        for row in csv.DictReader(handle):
                            query_id = str(row["query_id"])
                            if query_id in mapping:
                                raise ValueError(f"{map_path}: duplicate query_id {query_id}")
                            mapping[query_id] = (str(row["task_id"]), int(row["repeat"]))
            batch_records = jsonl(batch_path)
            trace_records = jsonl(trace_path)
            if not batch_records or not trace_records:
                raise ValueError(f"{profile} shard {shard_index}: empty batch or trace file")
            trace_by_query: dict[str, dict] = {}
            for trace in trace_records:
                query_id = str(trace["query_id"])
                if query_id in trace_by_query:
                    raise ValueError(f"{trace_path}: duplicate query_id {query_id}")
                trace_by_query[query_id] = trace
            seen_queries: set[str] = set()
            for batch in batch_records:
                source_batch_id = str(batch["batch_id"])
                batch_key = str(shard.get("batch_key_prefix", f"{profile}:{shard_index}:")) + source_batch_id
                ids = [str(query_id) for query_id in batch["query_ids"]]
                if not ids or len(ids) != len(set(ids)):
                    raise ValueError(f"{batch_path}: missing or duplicate batch query IDs")
                expected_batch_size = int(shard.get("expected_batch_size", profile_spec.get("expected_batch_size", 20)))
                if len(ids) != expected_batch_size or int(batch.get("batch_size", len(ids))) != expected_batch_size:
                    raise ValueError(f"{batch_path}: expected {expected_batch_size} requests in {batch_key}")
                start = float(batch["batch_prompt_start_time_s"])
                end = float(batch["batch_prompt_end_time_s"])
                if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                    raise ValueError(f"{batch_path}: invalid batch window")
                batches_out.append({
                    "profile": profile, "batch_key": batch_key,
                    "batch_role": str(shard.get("batch_role", "primary")),
                    "source_batch_id": source_batch_id,
                    "batch_start_s": start, "batch_end_s": end,
                    "gpu_energy_j": required_energy(batch, "batch_gpu_energy_joules", batch_path),
                    "cpu_energy_j": required_energy(batch, "batch_cpu_energy_joules", batch_path),
                    "dram_energy_j": required_energy(batch, "batch_dram_energy_joules", batch_path),
                    "source_path": batch_relative, "source_sha256": batch_hash,
                    "raw_meter_source_path": raw_meter_relative,
                    "raw_meter_source_sha256": raw_meter_hash,
                    "raw_meter_manifest_source_path": raw_manifest_relative,
                    "raw_meter_manifest_source_sha256": raw_manifest_hash,
                    "summary_source_path": summary_relative,
                    "summary_source_sha256": summary_hash,
                    "run_provenance_source_path": provenance_relative,
                    "run_provenance_source_sha256": provenance_hash,
                    "energy_sources_source_path": source_config_relative,
                    "energy_sources_source_sha256": source_config_hash,
                })
                for query_id in ids:
                    if query_id in seen_queries or query_id not in trace_by_query:
                        raise ValueError(f"{profile} shard {shard_index}: duplicate or missing trace {query_id}")
                    seen_queries.add(query_id)
                    trace = trace_by_query[query_id]
                    if str(trace.get("batch_id")) != source_batch_id:
                        raise ValueError(f"{trace_path}: query {query_id} has wrong batch")
                    if query_id in mapping:
                        task_id, repeat = mapping[query_id]
                    else:
                        if trace.get("task_id") is None or trace.get("repeat_index", trace.get("repeat")) is None:
                            raise ValueError(f"{trace_path}: query {query_id} needs task/repeat or a query map")
                        task_id = str(trace["task_id"])
                        repeat = int(trace.get("repeat_index", trace.get("repeat")))
                    requests_out.append({
                        "profile": profile, "batch_key": batch_key,
                        "physical_request_id": f"{profile}:{shard_index}:{query_id}",
                        "task_id": task_id, "repeat": repeat,
                        "start_s": float(trace["prompt_start_time_s"]),
                        "end_s": float(trace["prompt_end_time_s"]),
                        "trace_source_path": trace_relative, "trace_source_sha256": trace_hash,
                        "mapping_source_path": str(map_relative or ""),
                        "mapping_source_sha256": map_hash,
                    })
            if seen_queries != set(trace_by_query):
                raise ValueError(f"{trace_path}: unassigned trace records")
            if mapping and seen_queries != set(mapping):
                raise ValueError(f"{map_relative}: query map does not match traces")
    if len({(row["profile"], row["batch_key"]) for row in batches_out}) != len(batches_out):
        raise ValueError("Duplicate profile/batch identity across IPW shards")
    if len({(row["profile"], row["physical_request_id"]) for row in requests_out}) != len(requests_out):
        raise ValueError("Duplicate physical request identity across IPW shards")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "energy_component_batches.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(BATCH_COLUMNS | {
            "source_batch_id", "raw_meter_source_path", "raw_meter_source_sha256",
            "raw_meter_manifest_source_path", "raw_meter_manifest_source_sha256",
            "summary_source_path", "summary_source_sha256",
            "run_provenance_source_path", "run_provenance_source_sha256",
            "energy_sources_source_path", "energy_sources_source_sha256",
        }))
        writer.writeheader()
        writer.writerows(batches_out)
    fields = sorted(REQUEST_COLUMNS | {
        "trace_source_path", "trace_source_sha256", "mapping_source_path", "mapping_source_sha256",
    })
    with (output_dir / "energy_component_requests.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(requests_out)
    return len(batches_out), len(requests_out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduction/inputs")
    args = parser.parse_args()
    batches, requests = build_from_ipw(args.config, args.output_dir)
    print(f"Staged {batches} IPW energy batches and {requests} physical requests")


if __name__ == "__main__":
    main()
