#!/usr/bin/env python3
"""Prepare and validate pinned GPT-OSS GDPVal pairwise studies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

import gdpval_pairwise_eval as base
import qwen_pairwise_eval as qwen


HF_REPO_ID = (
    "ScalingIntelligence/"
    "gdpval-gpt-oss-occupation-repeat-results-azeinali-20260717"
)
HF_REVISION = "4a277392cddacbef090be029d33f4f69372d487e"
HF_REPO_TYPE = "dataset"
CONDITION = "without_rubric"
GOOGLE_MEDIA_PARSER_MODEL = "models/gemini-3.1-flash-lite"
DEFAULT_BASELINES = Path("GDPVal_220_Tasks")
DEFAULT_GLM_STATE = Path("reports/gdpval_pairwise_glm52_20260715")

EXPECTED_BASELINE_SHA256 = qwen.EXPECTED_BASELINE_SHA256
EXPECTED_GLM_CROSSWALK_SHA256 = qwen.EXPECTED_GLM_CROSSWALK_SHA256
EXPECTED_GLM_MANIFEST_SHA256 = qwen.EXPECTED_GLM_MANIFEST_SHA256


@dataclass(frozen=True)
class ModelConfig:
    key: str
    model_name: str
    source_root: str
    report_root: str
    study_id: str
    default_state: Path

    @property
    def artifact_root(self) -> str:
        return f"{self.source_root}/run/artifacts"


MODEL_CONFIGS = {
    "120b": ModelConfig(
        key="120b",
        model_name="openai/gpt-oss-120b",
        source_root="gpt-oss-120b-high-no-rubric",
        report_root="gpt-oss-120b-high-no-rubric/reports/final",
        study_id="gpt_oss_120b_norubric_hf_20260717",
        default_state=Path("reports/gpt_oss_120b_norubric_source_20260724"),
    ),
    "20b": ModelConfig(
        key="20b",
        model_name="openai/gpt-oss-20b",
        source_root="gpt-oss-20b-high-no-rubric",
        report_root="gpt-oss-20b-high-no-rubric/reports/final_corrected",
        study_id="gpt_oss_20b_norubric_hf_20260717",
        default_state=Path("reports/gpt_oss_20b_norubric_source_20260724"),
    ),
}


def require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or ""
    if not token:
        raise base.EvaluationError(
            "HF_TOKEN is not set in the process environment; do not pass "
            "Hugging Face tokens on the command line"
        )
    return token


def query_index_from_name(name: str) -> int | None:
    match = re.match(r"^q(\d{4})(?:\D|$)", name)
    return int(match.group(1)) if match else None


def selected_artifacts(
    entries: Iterable[Any],
    *,
    artifact_root: str,
) -> tuple[dict[int, str], dict[str, int]]:
    prefix = artifact_root.rstrip("/") + "/"
    selected: dict[int, str] = {}
    files: dict[str, int] = {}
    for entry in entries:
        path = str(getattr(entry, "path", "") or "")
        if not path.startswith(prefix):
            continue
        relative = path[len(prefix) :]
        artifact_name = relative.split("/", 1)[0]
        query_index = query_index_from_name(artifact_name)
        if query_index is None:
            continue
        artifact_path = prefix + artifact_name
        if hasattr(entry, "size"):
            files[path] = int(getattr(entry, "size", 0) or 0)
        if path != artifact_path + "/metadata.json":
            continue
        prior = selected.setdefault(query_index, artifact_path)
        if prior != artifact_path:
            raise base.EvaluationError(
                f"multiple metadata-bearing artifact directories for "
                f"q{query_index:04d}: {prior!r}, {artifact_path!r}"
            )
    expected = set(range(880))
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        extra = sorted(set(selected) - expected)
        raise base.EvaluationError(
            "canonical artifact indices are not exactly q0000-q0879; "
            f"missing={missing[:20]}, extra={extra[:20]}"
        )
    return selected, files


def copy_hf_file(
    *,
    repo_path: str,
    state: Path,
    token: str,
    collision_safe: bool,
) -> Path:
    from huggingface_hub import hf_hub_download

    if collision_safe:
        destination = qwen.deliverable_local_path(state, repo_path)
        transfer_root = (
            state
            / ".hf_transfers"
            / hashlib.sha256(repo_path.encode("utf-8")).hexdigest()
        )
        if destination.is_file():
            return destination
        downloaded = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=repo_path,
                repo_type=HF_REPO_TYPE,
                revision=HF_REVISION,
                token=token,
                local_dir=transfer_root,
            )
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(downloaded), str(destination))
        shutil.rmtree(transfer_root, ignore_errors=True)
        return destination

    source_root = state / "source"
    destination = source_root / PurePosixPath(repo_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination
    downloaded = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=repo_path,
            repo_type=HF_REPO_TYPE,
            revision=HF_REVISION,
            token=token,
            local_dir=source_root,
        )
    )
    if downloaded != destination and not destination.is_file():
        shutil.copyfile(downloaded, destination)
    return destination


def validate_metadata(
    metadata: dict[str, Any],
    *,
    query_index: int,
    crosswalk_task_ids: set[str],
) -> dict[str, Any]:
    task_id = str(metadata.get("task_id") or "")
    repeat = int(metadata.get("repeat_index", -1))
    source_task_index = int(metadata.get("source_task_index", -1))
    completed = metadata.get("completed")
    timed_out = metadata.get("timed_out")
    failures: list[str] = []
    if metadata.get("include_rubric_in_prompt") is not False:
        failures.append("include_rubric_in_prompt is not the JSON boolean false")
    if not isinstance(completed, bool):
        failures.append("completed is not a JSON boolean")
    if not isinstance(timed_out, bool):
        failures.append("timed_out is not a JSON boolean")
    if task_id not in crosswalk_task_ids:
        failures.append(f"task_id {task_id!r} is absent from the frozen GLM crosswalk")
    if source_task_index not in range(220):
        failures.append(f"source_task_index={source_task_index}, expected 0-219")
    if repeat not in range(4):
        failures.append(f"repeat_index={repeat}, expected 0-3")
    if failures:
        raise base.EvaluationError(
            f"q{query_index:04d} failed canonical metadata validation: "
            + "; ".join(failures)
        )
    return {
        "task_id": task_id,
        "repeat": repeat,
        "source_task_index": source_task_index,
        "completed": completed,
        "timed_out": timed_out,
        "include_rubric_in_prompt": False,
    }


def resolve_attempt_deliverables(
    metadata: dict[str, Any],
    *,
    artifact_path: str,
    workspace_files: Sequence[str],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    resolved: list[str] = []
    decisions: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for declared in qwen.declared_deliverable_values(metadata):
        try:
            files, item_decisions = qwen.resolve_declared_remote_files(
                {"deliverable_files": [declared]},
                artifact_path=artifact_path,
                workspace_files=workspace_files,
            )
            resolved.extend(files)
            decisions.extend(item_decisions)
        except base.EvaluationError as exc:
            missing.append(
                {
                    "declared_path": declared,
                    "failure": str(exc),
                    "classification": "genuine_missing_or_invalid_submission",
                }
            )
    if missing and metadata.get("completed") is True:
        raise base.EvaluationError(
            "archive-integrity failure: a completed attempt declares absent "
            f"deliverables under {artifact_path}: "
            f"{[item['declared_path'] for item in missing]}"
        )
    return sorted(set(resolved)), decisions, missing


def candidate_quality_status(
    metadata: dict[str, Any],
    *,
    resolved_paths: Sequence[str],
    missing: Sequence[dict[str, Any]],
    remote_sizes: dict[str, int],
) -> str:
    validation = metadata.get("deliverable_validation")
    validation_status = (
        str(validation.get("status") or "")
        if isinstance(validation, dict)
        else ""
    )
    if not resolved_paths:
        return "missing_or_invalid"
    if any(int(remote_sizes[path]) == 0 for path in resolved_paths):
        return "empty_or_invalid"
    if metadata.get("timed_out") is True:
        return "timed_out_with_artifacts"
    if metadata.get("completed") is not True or missing:
        return "incomplete_with_artifacts"
    if validation_status == "invalid":
        return "invalid_with_artifacts"
    return "present"


def stage_source(
    state: Path,
    *,
    glm_state: Path,
    config: ModelConfig,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    token = require_hf_token()
    crosswalk = qwen.frozen_crosswalk(glm_state)
    crosswalk_ids = {str(row["agent_task_id"]) for row in crosswalk}
    api = HfApi(token=token)
    info = api.repo_info(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        revision=HF_REVISION,
    )
    resolved_revision = str(getattr(info, "sha", "") or "")
    if resolved_revision != HF_REVISION:
        raise base.EvaluationError(
            f"HF revision mismatch: requested {HF_REVISION}, "
            f"resolved {resolved_revision}"
        )

    entries = list(
        api.list_repo_tree(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            revision=HF_REVISION,
            path_in_repo=config.artifact_root,
            recursive=True,
            expand=False,
        )
    )
    artifacts, remote_files = selected_artifacts(
        entries,
        artifact_root=config.artifact_root,
    )

    selected_records: list[dict[str, Any]] = []
    inventory_files: list[dict[str, Any]] = []
    seen_task_repeats: set[tuple[str, int]] = set()
    source_indices: dict[int, set[int]] = defaultdict(set)
    for index, (query_index, artifact_path) in enumerate(
        sorted(artifacts.items()),
        start=1,
    ):
        metadata_repo_path = f"{artifact_path}/metadata.json"
        expected_metadata_size = remote_files.get(metadata_repo_path)
        if expected_metadata_size is None:
            raise base.EvaluationError(
                f"archive-integrity failure: missing {metadata_repo_path}"
            )
        metadata_path = copy_hf_file(
            repo_path=metadata_repo_path,
            state=state,
            token=token,
            collision_safe=False,
        )
        if metadata_path.stat().st_size != expected_metadata_size:
            raise base.EvaluationError(
                f"metadata size mismatch for {metadata_repo_path}"
            )
        metadata = base.read_json(metadata_path)
        if not isinstance(metadata, dict):
            raise base.EvaluationError(
                f"metadata is not an object: {metadata_repo_path}"
            )
        canonical = validate_metadata(
            metadata,
            query_index=query_index,
            crosswalk_task_ids=crosswalk_ids,
        )
        workspace_prefix = artifact_path + "/workspace/"
        workspace_files = sorted(
            path
            for path in remote_files
            if path.startswith(workspace_prefix)
            and not qwen.ignored_remote_path(
                path,
                artifact_path + "/workspace",
            )
        )
        deliverable_paths, decisions, missing = resolve_attempt_deliverables(
            metadata,
            artifact_path=artifact_path,
            workspace_files=workspace_files,
        )
        key = (canonical["task_id"], canonical["repeat"])
        if key in seen_task_repeats:
            raise base.EvaluationError(
                f"duplicate selected GPT-OSS task/repeat: {key}"
            )
        seen_task_repeats.add(key)
        source_indices[canonical["source_task_index"]].add(
            canonical["repeat"]
        )
        metadata_record = base.source_file_record(
            metadata_path,
            roots=[state / "source"],
        )
        metadata_record["hf_path"] = metadata_repo_path
        metadata_record["role"] = "canonical_metadata"
        inventory_files.append(metadata_record)
        selected_records.append(
            {
                **canonical,
                "run_name": config.source_root,
                "query_index": query_index,
                "query_id": f"q{query_index:04d}",
                "artifact_hf_path": artifact_path,
                "metadata_hf_path": metadata_repo_path,
                "metadata_path": str(metadata_path),
                "declared_deliverables": qwen.declared_deliverable_values(
                    metadata
                ),
                "deliverable_resolution": decisions,
                "missing_declared_deliverables": missing,
                "deliverable_hf_paths": deliverable_paths,
                "deliverable_remote_sizes": {
                    path: remote_files[path] for path in deliverable_paths
                },
                "deliverable_files": [],
                "deliverable_validation_status": (
                    (metadata.get("deliverable_validation") or {}).get(
                        "status"
                    )
                    if isinstance(metadata.get("deliverable_validation"), dict)
                    else None
                ),
                "candidate_quality_status": candidate_quality_status(
                    metadata,
                    resolved_paths=deliverable_paths,
                    missing=missing,
                    remote_sizes=remote_files,
                ),
            }
        )
        if index % 100 == 0:
            print(f"validated metadata {index}/880", flush=True)

    if len(selected_records) != 880 or len(seen_task_repeats) != 880:
        raise base.EvaluationError(
            "expected 880 unique selected outputs, got "
            f"rows={len(selected_records)}, "
            f"task_repeats={len(seen_task_repeats)}"
        )
    if set(source_indices) != set(range(220)):
        raise base.EvaluationError(
            "source task indices are not exactly 0-219"
        )
    bad_repeats = {
        task_index: sorted(repeats)
        for task_index, repeats in source_indices.items()
        if repeats != {0, 1, 2, 3}
    }
    if bad_repeats:
        raise base.EvaluationError(
            "source task indices without repeats 0-3: "
            f"{list(bad_repeats.items())[:10]}"
        )
    if len({row["task_id"] for row in selected_records}) != 220:
        raise base.EvaluationError(
            "selected source does not contain exactly 220 task IDs"
        )

    planned_files: dict[str, int] = {}
    for selected in selected_records:
        for repo_path, size in selected["deliverable_remote_sizes"].items():
            prior = planned_files.setdefault(str(repo_path), int(size))
            if prior != int(size):
                raise base.EvaluationError(
                    f"HF reports conflicting sizes for {repo_path}: "
                    f"{prior}, {size}"
                )
    missing_bytes = 0
    existing_bytes = 0
    for repo_path, expected_size in planned_files.items():
        local_path = qwen.deliverable_local_path(state, repo_path)
        if local_path.is_file():
            if local_path.stat().st_size != expected_size:
                raise base.EvaluationError(
                    f"existing staged file has wrong size: {local_path}"
                )
            existing_bytes += expected_size
        else:
            missing_bytes += expected_size
    free_bytes = shutil.disk_usage(state).free
    reserve_bytes = 5 * 1024**3
    download_plan = {
        "created_at": base.utc_now(),
        "hf_repo_id": HF_REPO_ID,
        "hf_revision": HF_REVISION,
        "model": config.model_name,
        "selected_outputs": 880,
        "metadata_files_staged": 880,
        "unique_deliverable_files": len(planned_files),
        "total_deliverable_bytes": sum(planned_files.values()),
        "already_staged_deliverable_bytes": existing_bytes,
        "remaining_download_bytes": missing_bytes,
        "free_bytes_before_deliverables": free_bytes,
        "required_free_space_reserve_bytes": reserve_bytes,
        "fits_with_reserve": missing_bytes + reserve_bytes <= free_bytes,
        "files": [
            {"hf_path": path, "size_bytes": size}
            for path, size in sorted(planned_files.items())
        ],
    }
    base.write_json(state / "source_download_plan.json", download_plan)
    if not download_plan["fits_with_reserve"]:
        raise base.EvaluationError(
            "canonical HF deliverables do not fit with the 5 GiB safety "
            "reserve; see source_download_plan.json"
        )

    downloaded_records: dict[str, dict[str, Any]] = {}
    for index, (repo_path, expected_size) in enumerate(
        sorted(planned_files.items()),
        start=1,
    ):
        local_path = copy_hf_file(
            repo_path=repo_path,
            state=state,
            token=token,
            collision_safe=True,
        )
        if local_path.stat().st_size != expected_size:
            raise base.EvaluationError(
                f"downloaded file size mismatch for {repo_path}: "
                f"expected {expected_size}, got {local_path.stat().st_size}"
            )
        record = base.source_file_record(
            local_path,
            roots=[state / "source"],
        )
        record["hf_path"] = repo_path
        record["role"] = "declared_deliverable"
        inventory_files.append(record)
        downloaded_records[repo_path] = record
        if index % 50 == 0 or index == len(planned_files):
            print(
                f"staged declared deliverables "
                f"{index}/{len(planned_files)}",
                flush=True,
            )
    for selected in selected_records:
        selected["deliverable_files"] = [
            downloaded_records[path]
            for path in selected.pop("deliverable_hf_paths")
        ]
        selected.pop("deliverable_remote_sizes")

    selected_records.sort(
        key=lambda row: (
            int(row["source_task_index"]),
            int(row["repeat"]),
            str(row["task_id"]),
        )
    )
    base.write_json(state / "gptoss_source_outputs.json", selected_records)
    missing_attempts = [
        row
        for row in selected_records
        if row["missing_declared_deliverables"]
    ]
    inventory = {
        "schema_version": "gpt-oss-hf-source-v1",
        "created_at": base.utc_now(),
        "model": config.model_name,
        "hf_repo_id": HF_REPO_ID,
        "hf_repo_type": HF_REPO_TYPE,
        "requested_revision": HF_REVISION,
        "resolved_revision": resolved_revision,
        "selection_rule": (
            f"all q0000-q0879 records from {config.artifact_root}"
        ),
        "canonical_report_root": config.report_root,
        "selected_outputs": 880,
        "unique_tasks": 220,
        "unique_task_repeats": 880,
        "metadata_files": sum(
            row["role"] == "canonical_metadata"
            for row in inventory_files
        ),
        "deliverable_files": sum(
            row["role"] == "declared_deliverable"
            for row in inventory_files
        ),
        "files": sorted(
            inventory_files,
            key=lambda row: str(row["hf_path"]),
        ),
        "excluded_content_classes": sorted(
            qwen.IGNORED_DELIVERABLE_PARTS
        ),
        "incomplete_attempt_policy": (
            "All four attempts are retained. Existing declared artifacts are "
            "evaluated. A declared path absent from an attempt marked "
            "completed=false is frozen as a genuinely missing/invalid "
            "submission; files from other repeats are never substituted. "
            "A declared path absent from completed=true is an "
            "archive-integrity failure."
        ),
        "missing_or_invalid_attempts": [
            {
                "query_id": row["query_id"],
                "task_id": row["task_id"],
                "repeat": row["repeat"],
                "completed": row["completed"],
                "timed_out": row["timed_out"],
                "missing_declared_deliverables": (
                    row["missing_declared_deliverables"]
                ),
            }
            for row in missing_attempts
        ],
    }
    base.write_json(state / "frozen_source_inventory.json", inventory)
    summary = {
        "created_at": base.utc_now(),
        "model": config.model_name,
        "hf_repo_id": HF_REPO_ID,
        "hf_revision": HF_REVISION,
        "outputs": 880,
        "tasks": 220,
        "repeats": 4,
        "condition": CONDITION,
        "completed": sum(row["completed"] is True for row in selected_records),
        "incomplete": sum(row["completed"] is False for row in selected_records),
        "timed_out": sum(row["timed_out"] is True for row in selected_records),
        "missing_declared_attempts": len(missing_attempts),
        "quality_status": dict(
            Counter(
                row["candidate_quality_status"]
                for row in selected_records
            )
        ),
        "frozen_source_inventory_sha256": base.sha256_file(
            state / "frozen_source_inventory.json"
        ),
    }
    base.write_json(state / "source_stage_summary.json", summary)
    return summary


def build_manifest(
    state: Path,
    *,
    baselines: Path,
    glm_state: Path,
    config: ModelConfig,
) -> dict[str, Any]:
    output_path = state / "gptoss_source_outputs.json"
    if not output_path.is_file():
        raise base.EvaluationError(
            "run `stage-source` before building the GPT-OSS manifest"
        )
    outputs = base.read_json(output_path)
    if not isinstance(outputs, list) or len(outputs) != 880:
        raise base.EvaluationError(
            f"expected 880 staged GPT-OSS outputs in {output_path}"
        )
    crosswalk = qwen.frozen_crosswalk(glm_state)
    crosswalk_by_agent = {
        str(row["agent_task_id"]): row for row in crosswalk
    }
    templates = qwen.glm_templates(glm_state)
    baseline_json = baselines / "final_1-220_samples.json"
    if not baseline_json.is_file():
        raise base.EvaluationError(
            f"missing baseline snapshot: {baseline_json}"
        )
    if base.sha256_file(baseline_json) != EXPECTED_BASELINE_SHA256:
        raise base.EvaluationError(
            f"baseline snapshot SHA-256 differs from GLM: {baseline_json}"
        )
    baseline_rows = base.read_json(baseline_json)
    baseline_by_id = {
        str(row["task_id"]): row for row in baseline_rows
    }
    if len(baseline_by_id) != 220:
        raise base.EvaluationError(
            "baseline snapshot must contain 220 unique tasks"
        )

    roots = [state / "source", baselines]
    file_hash_cache: dict[str, dict[str, Any]] = {}

    def record(path: Path) -> dict[str, Any]:
        key = str(path.resolve())
        if key not in file_hash_cache:
            file_hash_cache[key] = base.source_file_record(
                path,
                roots=roots,
            )
        return file_hash_cache[key]

    for output in outputs:
        record(Path(str(output["metadata_path"])))
    record(baseline_json)
    record(glm_state / "crosswalk.json")

    comparisons: list[dict[str, Any]] = []
    for output in outputs:
        task_id = str(output["task_id"])
        cross = crosswalk_by_agent.get(task_id)
        if not cross:
            raise base.EvaluationError(
                f"staged GPT-OSS task absent from crosswalk: {task_id}"
            )
        baseline_task_id = str(cross["baseline_task_id"])
        baseline_task = baseline_by_id.get(baseline_task_id)
        if not baseline_task:
            raise base.EvaluationError(
                "crosswalk baseline task absent from snapshot: "
                f"{baseline_task_id}"
            )
        agent_records = [
            record(Path(str(item["path"])))
            for item in output.get("deliverable_files") or []
        ]
        for baseline in base.BASELINES:
            template = templates[(task_id, baseline)]
            baseline_dir = (
                baselines
                / "deliverables"
                / baseline_task_id
                / ("human" if baseline == "human" else "human+ai")
            )
            baseline_records = [
                record(path) for path in base.files_under(baseline_dir)
            ]
            if not baseline_records:
                raise base.EvaluationError(
                    f"missing {baseline} baseline deliverable: "
                    f"{baseline_dir}"
                )
            if base.bundle_source_hash(
                baseline_records
            ) != base.bundle_source_hash(template["baseline_files"]):
                raise base.EvaluationError(
                    f"{task_id}/{baseline}: local baseline bundle differs "
                    "from frozen GLM content"
                )
            repeat = int(output["repeat"])
            position = base.agent_position(
                task_id,
                CONDITION,
                baseline,
                repeat,
            )
            comparison_id = (
                f"{config.study_id}__{CONDITION}__{baseline}__"
                f"{task_id}__rep{repeat:02d}"
            )
            comparisons.append(
                {
                    "schema_version": base.SCHEMA_VERSION,
                    "study_id": config.study_id,
                    "comparison_id": comparison_id,
                    "task_id": task_id,
                    "baseline_task_id": baseline_task_id,
                    "condition": CONDITION,
                    "baseline": baseline,
                    "repeat": repeat,
                    "occupation": cross["occupation"],
                    "sector": cross["sector"],
                    "title": cross["title"],
                    "task_prompt": template["task_prompt"],
                    "rubric": template["rubric"],
                    "rubric_pretty": template.get("rubric_pretty") or "",
                    "expected_reference_names": (
                        template["expected_reference_names"]
                    ),
                    "reference_files": template["reference_files"],
                    "reference_linkage": template["reference_linkage"],
                    "agent_position": position,
                    "candidate_A_source": (
                        "agent" if position == "A" else baseline
                    ),
                    "candidate_B_source": (
                        baseline if position == "A" else "agent"
                    ),
                    "agent_files": agent_records,
                    "baseline_files": baseline_records,
                    "agent_quality_status": (
                        output["candidate_quality_status"]
                    ),
                    "agent_completed": output["completed"],
                    "agent_timed_out": output["timed_out"],
                    "agent_missing_declared_count": len(
                        output["missing_declared_deliverables"]
                    ),
                    "agent_run_name": output["run_name"],
                    "agent_artifact_dir": output["artifact_hf_path"],
                    "baseline_dir": str(baseline_dir),
                    "source_task_index": int(output["source_task_index"]),
                    "source_query_id": output["query_id"],
                }
            )

    comparisons.sort(key=lambda row: str(row["comparison_id"]))
    base.validate_manifest(
        comparisons,
        expected_tasks=220,
        repeat_count=4,
        conditions=(CONDITION,),
    )
    base.write_jsonl(
        state / "comparison_manifest.unblinded.jsonl",
        comparisons,
    )
    base.write_json(state / "crosswalk.json", crosswalk)
    base.write_json(
        state / "source_files.json",
        sorted(file_hash_cache.values(), key=lambda row: str(row["path"])),
    )
    shared_hashes = {
        base.bundle_source_hash(files)
        for row in comparisons
        for files in (row["reference_files"], row["baseline_files"])
    }
    shared_evidence = qwen.verified_shared_evidence(
        glm_state=glm_state,
        shared_hashes=shared_hashes,
    )
    base.write_json(state / "evidence_index.json", shared_evidence)
    all_specs = base.unique_bundle_specs(comparisons)
    base.write_json(
        state / "evidence_summary.json",
        {
            "created_at": base.utc_now(),
            "unique_bundles_expected": len(all_specs),
            "unique_bundles_prepared": len(shared_evidence),
            "shared_bundles_inherited_by_content_hash": (
                len(shared_evidence)
            ),
            "parser_error_bundles": 0,
            "media_pending_items": 0,
        },
    )
    summary = {
        "schema_version": base.SCHEMA_VERSION,
        "dataset": f"GDPVal {config.model_name}",
        "study_id": config.study_id,
        "created_at": base.utc_now(),
        "comparisons": len(comparisons),
        "tasks": 220,
        "repeat_count": 4,
        "active_conditions": [CONDITION],
        "conditions": {CONDITION: len(comparisons)},
        "baselines": {
            baseline: sum(
                row["baseline"] == baseline for row in comparisons
            )
            for baseline in base.BASELINES
        },
        "position_counts": dict(
            Counter(row["agent_position"] for row in comparisons)
        ),
        "agent_outputs": len(
            {
                (row["task_id"], int(row["repeat"]))
                for row in comparisons
            }
        ),
        "agent_quality_status": dict(
            Counter(row["agent_quality_status"] for row in comparisons)
        ),
        "hf_repo_id": HF_REPO_ID,
        "hf_revision": HF_REVISION,
        "canonical_source_root": config.source_root,
        "canonical_report_root": config.report_root,
        "baseline_snapshot_sha256": base.sha256_file(baseline_json),
        "glm_crosswalk_sha256": base.sha256_file(
            glm_state / "crosswalk.json"
        ),
        "shared_evidence_source_state": str(glm_state.resolve()),
        "shared_evidence_bundles": len(shared_evidence),
        "manifest_sha256": base.sha256_file(
            state / "comparison_manifest.unblinded.jsonl"
        ),
        "provenance_markers": {
            "GPT-OSS model family": "GPT-OSS",
            "GPT-OSS model name": config.model_name,
            "GPT-OSS HF repository": HF_REPO_ID,
            "GPT-OSS HF revision": HF_REVISION,
            "GPT-OSS source root": config.source_root,
            "GPT-OSS canonical report root": config.report_root,
            "GPT-OSS local source staging path": str(
                (state / "source").resolve()
            ),
            "GPT-OSS local state path": str(state.resolve()),
        },
    }
    base.write_json(state / "manifest_summary.json", summary)
    return summary


def agent_bundle_hashes(state: Path) -> set[str]:
    return {
        base.bundle_source_hash(row["agent_files"])
        for row in base.iter_jsonl(
            state / "comparison_manifest.unblinded.jsonl"
        )
    }


def prepare_candidate_evidence(
    state: Path,
    *,
    only_hashes: set[str] | None,
    workers: int,
) -> dict[str, Any]:
    selected = agent_bundle_hashes(state)
    if only_hashes is not None:
        selected &= only_hashes
    result = base.prepare_evidence(
        state,
        only_hashes=selected,
        workers=workers,
    )
    result["gptoss_candidate_hashes_selected"] = len(selected)
    result["shared_bundles_inherited_by_content_hash"] = sum(
        bool(item.get("inherited_by_content_hash"))
        for item in base.read_json(state / "evidence_index.json")
    )
    base.write_json(state / "evidence_summary.json", result)
    return result


def prepare_candidate_evidence_pilot(
    state: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    selection_path = state / "evidence_pilot_selection.json"
    selection = (
        base.read_json(selection_path)
        if selection_path.is_file()
        else base.select_evidence_pilot(state)
    )
    hashes = {
        str(item["source_hash"]) for item in selection["bundles"]
    } & agent_bundle_hashes(state)
    return prepare_candidate_evidence(
        state,
        only_hashes=hashes,
        workers=workers,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_CONFIGS),
        required=True,
    )
    parser.add_argument("--state", type=Path)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--glm-state", type=Path, default=DEFAULT_GLM_STATE)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stage-source")
    sub.add_parser("manifest")
    sub.add_parser("select-evidence-pilot")
    sub.add_parser("select-judge-pilot")
    pilot = sub.add_parser("prepare-evidence-pilot")
    pilot.add_argument("--workers", type=int, default=4)
    prepare = sub.add_parser("prepare-evidence")
    prepare.add_argument("--hash", action="append", default=[])
    prepare.add_argument("--workers", type=int, default=4)
    cap = sub.add_parser("cap-visuals")
    cap.add_argument(
        "--max-pages",
        type=int,
        default=base.MAX_VISUAL_PAGES_PER_BUNDLE,
    )
    media = sub.add_parser("parse-media-google")
    media.add_argument(
        "--parser-model",
        dest="media_model",
        default=GOOGLE_MEDIA_PARSER_MODEL,
    )
    media.add_argument("--limit", type=int)
    audit = sub.add_parser("audit-inputs")
    audit.add_argument("--skip-source-rehash", action="store_true")
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--bootstrap-samples", type=int, default=10_000)
    aggregate.add_argument("--seed", type=int, default=20260724)
    sub.add_parser("validate")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = MODEL_CONFIGS[args.model]
    state = (args.state or config.default_state).resolve()
    baselines = args.baselines.resolve()
    glm_state = args.glm_state.resolve()
    try:
        if args.command == "stage-source":
            result = stage_source(
                state,
                glm_state=glm_state,
                config=config,
            )
        elif args.command == "manifest":
            result = build_manifest(
                state,
                baselines=baselines,
                glm_state=glm_state,
                config=config,
            )
        elif args.command == "select-evidence-pilot":
            result = base.select_evidence_pilot(state)
        elif args.command == "select-judge-pilot":
            rows = list(
                base.iter_jsonl(
                    state / "comparison_manifest.unblinded.jsonl"
                )
            )
            selected = base.select_pilot_rows(
                rows,
                minimum=24,
                state=state,
            )
            result = {
                "comparison_ids": selected,
                "count": len(selected),
                "created_at": base.utc_now(),
            }
            base.write_json(state / "pilot_selection.json", result)
        elif args.command == "prepare-evidence-pilot":
            result = prepare_candidate_evidence_pilot(
                state,
                workers=args.workers,
            )
        elif args.command == "prepare-evidence":
            result = prepare_candidate_evidence(
                state,
                only_hashes=set(args.hash) or None,
                workers=args.workers,
            )
        elif args.command == "cap-visuals":
            result = base.cap_existing_visuals(
                state,
                max_pages=args.max_pages,
            )
        elif args.command == "parse-media-google":
            result = qwen.parse_pending_media_google(
                state,
                model=args.media_model,
                limit=args.limit,
            )
        elif args.command == "audit-inputs":
            result = base.frozen_input_audit(
                state,
                verify_source_hashes=not args.skip_source_rehash,
            )
        elif args.command == "aggregate":
            result = base.aggregate_results(
                state,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
        elif args.command == "validate":
            result = base.validation_report(state)
        else:
            raise AssertionError(args.command)
        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except base.EvaluationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
