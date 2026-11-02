#!/usr/bin/env python3
"""Prepare the repaired AMD MI350X Qwen3.6-27B GDPval source.

This is a thin source adapter around the frozen Qwen pairwise-evaluation
pipeline.  It changes only how the 880 agent outputs are selected and staged;
the crosswalk, expert-only and expert-with-AI references, evidence builders,
blinding, balancing, pilot
selection, judge prompts, and provider runner remain the frozen Qwen versions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


AMD_HF_REPO_ID = (
    "ScalingIntelligence/"
    "azeinali-qwen36-27b-gdpval-occ-repeat4-norubric-amd-mi350x"
)
AMD_HF_REVISION = "ad092b6fce8fe7a63a0ffb05563dcb9977b5dc4b"
AMD_HF_ARCHIVE_PREFIX = (
    "archives/amd-mi350x-complete-all-attempts-20260807T230305Z"
)
AMD_STUDY_ID = "qwen36_27b_norubric_amd_mi350x_20260807"

# qwen_pairwise_eval reads these at import time.  They are provenance markers;
# this adapter never downloads from or writes to Hugging Face.
os.environ.setdefault("QWEN_PAIRWISE_HF_REPO_ID", AMD_HF_REPO_ID)
os.environ.setdefault("QWEN_PAIRWISE_HF_REVISION", AMD_HF_REVISION)
os.environ.setdefault("QWEN_PAIRWISE_HF_RUN_REL", AMD_HF_ARCHIVE_PREFIX)
os.environ.setdefault("QWEN_PAIRWISE_STUDY_ID", AMD_STUDY_ID)

import gdpval_pairwise_eval as base  # noqa: E402
import qwen_pairwise_eval as qwen  # noqa: E402


DEFAULT_STATE = Path("reports/qwen36_27b_norubric_amd_mi350x_source_20260815")
DEFAULT_BASELINES = Path("GDPVal_220_Tasks")
DEFAULT_GLM_STATE = Path("reports/gdpval_pairwise_glm52_20260715")
DEFAULT_CANONICAL = Path(
    "/matx/u/azeinali/gdp_ipw/qwen36_occ_runs/"
    "qwen36_gdpval_occ4_norubric_amd_cont_0020_20260727T063657Z/"
    "reports/canonical_fully_repaired_20260807T112814Z/"
    "qwen36_occ_repeat4_norubric_amd_mi350x_fully_repaired_"
    "canonical_traces.jsonl"
)
CLUSTER_WORKSPACE_PREFIX = "/workspace/matx_gdp_ipw"
CLUSTER_HOST_PREFIX = "/matx/u/azeinali/gdp_ipw"
AUTHORIZED_SOURCE_ROOT = Path(CLUSTER_HOST_PREFIX) / "qwen36_occ_runs"


def host_path(raw: str) -> Path:
    value = str(raw)
    if value.startswith(CLUSTER_WORKSPACE_PREFIX + "/"):
        value = CLUSTER_HOST_PREFIX + value[len(CLUSTER_WORKSPACE_PREFIX) :]
    path = Path(value)
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(AUTHORIZED_SOURCE_ROOT.resolve())
    except ValueError as exc:
        raise base.EvaluationError(
            f"declared AMD deliverable escapes the authorized source root: {raw!r}"
        ) from exc
    return resolved


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", value).strip()
    return cleaned or "deliverable"


def load_manifest_entry(
    path: Path,
    instance_id: str,
    cache: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    key = str(path.resolve())
    if key not in cache:
        raw = base.read_json(path)
        if not isinstance(raw, list):
            raise base.EvaluationError(f"source manifest is not a list: {path}")
        entries = {
            str(item.get("instance_id") or ""): item
            for item in raw
            if isinstance(item, dict)
        }
        if len(entries) != len(raw):
            raise base.EvaluationError(
                f"source manifest has missing or duplicate instance IDs: {path}"
            )
        cache[key] = entries
    try:
        return cache[key][instance_id]
    except KeyError as exc:
        raise base.EvaluationError(
            f"canonical instance {instance_id!r} is absent from source manifest {path}"
        ) from exc


def artifact_metadata_path(declared: list[str]) -> Path:
    if not declared:
        raise base.EvaluationError("canonical row has no declared deliverable path")
    workspace_file = host_path(declared[0])
    workspace = next(
        (parent for parent in workspace_file.parents if parent.name == "workspace"),
        None,
    )
    if workspace is None:
        raise base.EvaluationError(
            f"declared AMD deliverable is not inside a workspace: {workspace_file}"
        )
    metadata = workspace.parent / "metadata.json"
    if not metadata.is_file():
        raise base.EvaluationError(f"missing AMD artifact metadata: {metadata}")
    return metadata


def declared_deliverables(row: dict[str, Any]) -> tuple[list[str], str]:
    score_metadata = row.get("score_metadata") or {}
    validation = score_metadata.get("deliverable_validation") or {}
    raw = [str(value) for value in validation.get("primary_files") or []]
    if raw:
        return list(dict.fromkeys(raw)), "validated_primary_files"
    raw = [
        str(item.get("path"))
        for item in score_metadata.get("deliverables") or []
        if isinstance(item, dict) and item.get("path")
    ]
    return list(dict.fromkeys(raw)), "validated_deliverables_fallback"


def stage_file(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file():
            raise base.EvaluationError(f"staged target is not a regular file: {target}")
        if (
            target.stat().st_size != source.stat().st_size
            or base.sha256_file(target) != base.sha256_file(source)
        ):
            raise base.EvaluationError(f"conflicting existing staged file: {target}")
        return "already_present"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def validate_row(
    row: dict[str, Any],
    *,
    canonical_row_index: int,
    crosswalk_ids: set[str],
    manifest_entry: dict[str, Any],
    artifact_metadata: dict[str, Any],
) -> tuple[str, int, int]:
    task_id = str(row.get("task_id") or "")
    repeat = int(row.get("repeat_index", -1))
    source_task_index = int(manifest_entry.get("source_task_index", -1))
    instance_id = str(row.get("instance_id") or "")
    score = (row.get("score_metadata") or {}).get("score")
    failures: list[str] = []
    if task_id not in crosswalk_ids:
        failures.append("task ID absent from frozen crosswalk")
    if instance_id != f"{task_id}::rep{repeat:02d}":
        failures.append("instance ID does not match task/repeat")
    if repeat not in range(4):
        failures.append(f"repeat_index={repeat}")
    if source_task_index not in range(220):
        failures.append(f"source_task_index={source_task_index}, expected 0-219")
    if str(manifest_entry.get("task_id") or "") != task_id:
        failures.append("source manifest task ID mismatch")
    if int(manifest_entry.get("repeat_index", -1)) != repeat:
        failures.append("source manifest repeat mismatch")
    if int(manifest_entry.get("repeat_count", -1)) != 4:
        failures.append("source manifest repeat_count is not 4")
    if row.get("completed") is not True or artifact_metadata.get("completed") is not True:
        failures.append("trace or artifact metadata is not complete")
    if row.get("timed_out") is not False or artifact_metadata.get("timed_out") is not False:
        failures.append("trace or artifact metadata is timed out")
    if artifact_metadata.get("include_rubric_in_prompt") is not False:
        failures.append("artifact metadata does not match the default prompt configuration")
    if str(artifact_metadata.get("task_id") or "") != task_id:
        failures.append("artifact metadata task ID mismatch")
    if int(artifact_metadata.get("repeat_index", -1)) != repeat:
        failures.append("artifact metadata repeat mismatch")
    if int(artifact_metadata.get("source_task_index", -1)) != source_task_index:
        failures.append("artifact metadata source task index mismatch")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
    ):
        failures.append("canonical score is not finite numeric")
    if failures:
        raise base.EvaluationError(
            f"canonical AMD row {canonical_row_index}/{instance_id} failed validation: "
            + "; ".join(failures)
        )
    return task_id, repeat, source_task_index


def stage_source(state: Path, *, glm_state: Path, canonical: Path) -> dict[str, Any]:
    if not canonical.is_file():
        raise base.EvaluationError(f"missing repaired AMD canonical trace: {canonical}")
    crosswalk = qwen.frozen_crosswalk(glm_state)
    crosswalk_ids = {str(row["agent_task_id"]) for row in crosswalk}
    rows = list(base.iter_jsonl(canonical))
    if len(rows) != 880:
        raise base.EvaluationError(
            f"expected 880 repaired AMD canonical rows, found {len(rows)}"
        )

    state.mkdir(parents=True, exist_ok=True)
    source_root = state / "source"
    metadata_root = source_root / "metadata"
    deliverable_root = source_root / "deliverables"
    metadata_root.mkdir(parents=True, exist_ok=True)
    deliverable_root.mkdir(parents=True, exist_ok=True)

    manifest_cache: dict[str, dict[str, dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    seen_global_positions: set[int] = set()
    repeats_by_task: dict[str, set[int]] = defaultdict(set)
    selected: list[dict[str, Any]] = []
    inventory_files: list[dict[str, Any]] = []
    missing_declared: list[dict[str, Any]] = []
    stage_modes: Counter[str] = Counter()

    for canonical_row_index, row in enumerate(rows):
        instance_id = str(row.get("instance_id") or "")
        source_manifest = Path(str(row.get("source_manifest") or ""))
        if not source_manifest.is_file():
            raise base.EvaluationError(
                f"missing source manifest for canonical row {canonical_row_index}: {source_manifest}"
            )
        manifest_entry = load_manifest_entry(
            source_manifest, instance_id, manifest_cache
        )
        declared, selection_rule = declared_deliverables(row)
        if not declared:
            raise base.EvaluationError(
                f"canonical row {canonical_row_index}/{instance_id} has no validated deliverables"
            )
        metadata_path = artifact_metadata_path(declared)
        artifact_metadata = base.read_json(metadata_path)
        if not isinstance(artifact_metadata, dict):
            raise base.EvaluationError(f"artifact metadata is not an object: {metadata_path}")
        task_id, repeat, source_task_index = validate_row(
            row,
            canonical_row_index=canonical_row_index,
            crosswalk_ids=crosswalk_ids,
            manifest_entry=manifest_entry,
            artifact_metadata=artifact_metadata,
        )
        position = source_task_index * 4 + repeat
        key = (task_id, repeat)
        if key in seen:
            raise base.EvaluationError(f"duplicate canonical AMD task/repeat: {key}")
        seen.add(key)
        if position in seen_global_positions:
            raise base.EvaluationError(
                f"duplicate canonical AMD global position: {position}"
            )
        seen_global_positions.add(position)
        repeats_by_task[task_id].add(repeat)

        metadata_snapshot = metadata_root / f"g{position:04d}_{instance_id}.json"
        snapshot = {
            "schema_version": "amd-mi350x-pairwise-source-v1",
            "canonical_row_index": canonical_row_index,
            "global_position": position,
            "canonical_trace_path": str(canonical.resolve()),
            "canonical_trace_sha256": base.sha256_file(canonical),
            "canonical_row": row,
            "source_manifest_entry": manifest_entry,
            "artifact_metadata": artifact_metadata,
            "deliverable_selection_rule": selection_rule,
            "declared_deliverables": declared,
        }
        base.write_json(metadata_snapshot, snapshot)
        metadata_record = base.source_file_record(
            metadata_snapshot, roots=[source_root]
        )
        metadata_record.update(
            {
                "role": "canonical_metadata_snapshot",
                "source_path": str(metadata_path),
            }
        )
        inventory_files.append(metadata_record)

        staged_records: list[dict[str, Any]] = []
        resolutions: list[dict[str, Any]] = []
        for file_index, raw in enumerate(declared, start=1):
            source = host_path(raw)
            if not source.is_file() or source.is_symlink() or source.stat().st_size <= 0:
                missing = {
                    "canonical_position": position,
                    "canonical_row_index": canonical_row_index,
                    "instance_id": instance_id,
                    "declared_path": raw,
                    "resolved_host_path": str(source),
                    "reason": "absent, symlink, or empty at source freeze time",
                }
                missing_declared.append(missing)
                resolutions.append({**missing, "status": "missing"})
                continue
            target = (
                deliverable_root
                / f"g{position:04d}_{instance_id}"
                / f"{file_index:03d}_{safe_name(source.name)}"
            )
            mode = stage_file(source, target)
            stage_modes[mode] += 1
            record = base.source_file_record(target, roots=[source_root])
            record.update(
                {
                    "role": "declared_deliverable",
                    "source_path": str(source),
                    "declared_path": raw,
                    "selection_rule": selection_rule,
                    "stage_mode": mode,
                }
            )
            staged_records.append(record)
            inventory_files.append(record)
            resolutions.append(
                {
                    "declared_path": raw,
                    "resolved_host_path": str(source),
                    "staged_path": str(target),
                    "status": "present",
                    "size_bytes": record["size_bytes"],
                    "sha256": record["sha256"],
                }
            )

        quality = (
            "missing"
            if not staged_records
            else "partial_missing"
            if len(staged_records) != len(declared)
            else "present"
        )
        selected.append(
            {
                "task_id": task_id,
                "repeat": repeat,
                "source_task_index": source_task_index,
                "completed": True,
                "timed_out": False,
                "include_rubric_in_prompt": False,
                "run_name": f"amd_mi350x_{row.get('source_name') or 'canonical'}",
                "query_index": position,
                "query_id": f"g{position:04d}",
                "artifact_hf_path": str(metadata_path.parent),
                "metadata_path": str(metadata_snapshot),
                "metadata_stable_path": metadata_record["stable_path"],
                "declared_deliverables": declared,
                "deliverable_selection_rule": selection_rule,
                "deliverable_resolution": resolutions,
                "deliverable_files": staged_records,
                "candidate_quality_status": quality,
                "canonical_score": float((row.get("score_metadata") or {})["score"]),
                "canonical_source_name": row.get("source_name"),
                "canonical_source_manifest": str(source_manifest),
            }
        )
        if (canonical_row_index + 1) % 50 == 0 or canonical_row_index == len(rows) - 1:
            print(
                f"staged AMD outputs {canonical_row_index + 1}/{len(rows)}",
                flush=True,
            )

    if (
        len(seen) != 880
        or len(seen_global_positions) != 880
        or seen_global_positions != set(range(880))
        or len(repeats_by_task) != 220
    ):
        raise base.EvaluationError(
            "AMD source coverage mismatch: "
            f"task/repeats={len(seen)}, global_positions={len(seen_global_positions)}, "
            f"tasks={len(repeats_by_task)}"
        )
    bad = {
        task_id: sorted(repeats)
        for task_id, repeats in repeats_by_task.items()
        if repeats != {0, 1, 2, 3}
    }
    if bad:
        raise base.EvaluationError(
            f"AMD tasks without repeats 0-3: {list(bad.items())[:10]}"
        )

    selected.sort(
        key=lambda item: (
            int(item["source_task_index"]),
            int(item["repeat"]),
            str(item["task_id"]),
        )
    )
    base.write_json(state / "qwen_source_outputs.json", selected)
    base.write_json(state / "missing_declared_deliverables.json", missing_declared)
    inventory = {
        "schema_version": "amd-mi350x-pairwise-source-v1",
        "created_at": base.utc_now(),
        "source_kind": "fully repaired AMD MI350X canonical trace",
        "canonical_trace_path": str(canonical.resolve()),
        "canonical_trace_sha256": base.sha256_file(canonical),
        "hf_repo_id": AMD_HF_REPO_ID,
        "hf_repo_type": "dataset",
        "requested_revision": AMD_HF_REVISION,
        "resolved_revision": AMD_HF_REVISION,
        "hf_archive_repo_id": AMD_HF_REPO_ID,
        "hf_archive_revision": AMD_HF_REVISION,
        "hf_archive_prefix": AMD_HF_ARCHIVE_PREFIX,
        "selected_outputs": len(selected),
        "unique_tasks": len(repeats_by_task),
        "unique_task_repeats": len(seen),
        "metadata_files": len(selected),
        "deliverable_files": sum(len(row["deliverable_files"]) for row in selected),
        "deliverable_bytes": sum(
            int(item["size_bytes"])
            for item in inventory_files
            if item["role"] == "declared_deliverable"
        ),
        "missing_declared_deliverables": missing_declared,
        "files": sorted(inventory_files, key=lambda item: str(item["stable_path"])),
        "selection_policy": (
            "Use score_metadata.deliverable_validation.primary_files when present; "
            "otherwise use the validator's score_metadata.deliverables records. "
            "Declared paths absent at source-freeze time are preserved as missing "
            "candidate evidence rather than silently substituted."
        ),
    }
    base.write_json(state / "frozen_source_inventory.json", inventory)
    summary = {
        "created_at": base.utc_now(),
        "outputs": len(selected),
        "tasks": len(repeats_by_task),
        "repeats": 4,
        "condition": qwen.CONDITION,
        "hardware": "AMD Instinct MI350X",
        "model": "Qwen/Qwen3.6-27B",
        "canonical_trace_path": str(canonical.resolve()),
        "canonical_trace_sha256": base.sha256_file(canonical),
        "quality_status": dict(Counter(row["candidate_quality_status"] for row in selected)),
        "deliverable_selection_rules": dict(
            Counter(row["deliverable_selection_rule"] for row in selected)
        ),
        "missing_declared_deliverables": len(missing_declared),
        "stage_modes": dict(stage_modes),
        "frozen_source_inventory_sha256": base.sha256_file(
            state / "frozen_source_inventory.json"
        ),
    }
    base.write_json(state / "source_stage_summary.json", summary)
    return summary


def relocate_source(state: Path) -> dict[str, Any]:
    outputs_path = state / "qwen_source_outputs.json"
    if not outputs_path.is_file():
        raise base.EvaluationError(f"missing staged AMD source outputs: {outputs_path}")
    outputs = base.read_json(outputs_path)
    if not isinstance(outputs, list) or len(outputs) != 880:
        raise base.EvaluationError("relocation requires exactly 880 source outputs")
    source_root = state / "source"
    checked = 0
    for output in outputs:
        metadata = source_root / str(output["metadata_stable_path"])
        if not metadata.is_file():
            raise base.EvaluationError(f"missing relocated metadata: {metadata}")
        output["metadata_path"] = str(metadata)
        for item in output.get("deliverable_files") or []:
            path = source_root / str(item["stable_path"])
            if not path.is_file():
                raise base.EvaluationError(f"missing relocated deliverable: {path}")
            if (
                path.stat().st_size != int(item["size_bytes"])
                or base.sha256_file(path) != str(item["sha256"])
            ):
                raise base.EvaluationError(f"relocated deliverable differs: {path}")
            item["path"] = str(path)
            checked += 1
    base.write_json(outputs_path, outputs)

    inventory_path = state / "frozen_source_inventory.json"
    inventory = base.read_json(inventory_path)
    for item in inventory.get("files") or []:
        path = source_root / str(item["stable_path"])
        if not path.is_file():
            raise base.EvaluationError(f"missing relocated inventory file: {path}")
        item["path"] = str(path)
    inventory["relocated_at"] = base.utc_now()
    inventory["relocated_state"] = str(state)
    # The AMD source is staged from the immutable archive commit recorded in
    # the manifest.  Keep the generic provider validator's revision fields in
    # sync with the AMD-specific archive provenance fields.
    inventory["hf_repo_id"] = AMD_HF_REPO_ID
    inventory["hf_repo_type"] = "dataset"
    inventory["requested_revision"] = AMD_HF_REVISION
    inventory["resolved_revision"] = AMD_HF_REVISION
    base.write_json(inventory_path, inventory)
    source_summary_path = state / "source_stage_summary.json"
    if source_summary_path.is_file():
        source_summary = base.read_json(source_summary_path)
        source_summary["frozen_source_inventory_sha256"] = base.sha256_file(
            inventory_path
        )
        base.write_json(source_summary_path, source_summary)
    result = {
        "created_at": base.utc_now(),
        "state": str(state),
        "outputs": len(outputs),
        "deliverable_files_checked": checked,
        "inventory_files_checked": len(inventory.get("files") or []),
        "complete": True,
    }
    base.write_json(state / "source_relocation_report.json", result)
    return result


def build_manifest(state: Path, *, baselines: Path, glm_state: Path) -> dict[str, Any]:
    result = qwen.build_manifest(state, baselines=baselines, glm_state=glm_state)
    source_summary = base.read_json(state / "source_stage_summary.json")
    result.update(
        {
            "dataset": "GDPval Qwen3.6-27B AMD MI350X",
            "hardware": "AMD Instinct MI350X",
            "canonical_trace_path": source_summary["canonical_trace_path"],
            "canonical_trace_sha256": source_summary["canonical_trace_sha256"],
            "source_quality_status": source_summary["quality_status"],
            "source_missing_declared_deliverables": source_summary[
                "missing_declared_deliverables"
            ],
        }
    )
    result["provenance_markers"].update(
        {
            "Qwen accelerator": "AMD Instinct MI350X",
            "Qwen canonical trace": source_summary["canonical_trace_path"],
            "Qwen canonical trace SHA-256": source_summary[
                "canonical_trace_sha256"
            ],
            "Qwen archive prefix": AMD_HF_ARCHIVE_PREFIX,
        }
    )
    base.write_json(state / "manifest_summary.json", result)
    return result


def sanitize_provenance(state: Path) -> dict[str, Any]:
    """Redact source roots embedded inside submitted text or stack traces.

    This changes derived evidence only.  Original deliverables, source hashes,
    semantic content after the final workspace separator, and visual evidence
    remain unchanged.
    """

    index_path = state / "evidence_index.json"
    if not index_path.is_file():
        raise base.EvaluationError(f"missing evidence index: {index_path}")
    index = base.read_json(index_path)
    patterns = (
        re.compile(
            r"(?m)(?:/workspace/matx_gdp_ipw|/matx/u/azeinali/gdp_ipw)/"
            r"[^\n]*?/workspace/"
        ),
        re.compile(
            r"(?m)/Users/[^/\n]+/Desktop/joules_to_gdp/reports/"
            r"qwen36_27b_norubric_amd_mi350x_source_20260815/source/"
            r"deliverables/[^\n]*?/"
        ),
    )
    changed: list[dict[str, Any]] = []
    for item in index:
        text_path = Path(str(item["text_path"]))
        audit_path = Path(str(item["audit_path"]))
        if not text_path.is_file() or not audit_path.is_file():
            raise base.EvaluationError(
                f"missing evidence files during provenance sanitization: {text_path}"
            )
        original = text_path.read_text(encoding="utf-8", errors="replace")
        sanitized = original
        replacements = 0
        for pattern in patterns:
            sanitized, count = pattern.subn("[source-blind workspace]/", sanitized)
            replacements += count
        if sanitized == original:
            continue
        text_path.write_text(sanitized, encoding="utf-8")
        audit = base.read_json(audit_path)
        before_sha = str(audit.get("text_sha256") or "")
        after_sha = base.sha256_file(text_path)
        audit["text_sha256"] = after_sha
        audit["provenance_sanitization"] = {
            "rule": (
                "Replace embedded AMD source/staging roots through the final "
                "workspace separator with [source-blind workspace]/."
            ),
            "replacements": replacements,
            "before_text_sha256": before_sha,
            "after_text_sha256": after_sha,
            "source_deliverables_unchanged": True,
        }
        base.write_json(audit_path, audit)
        changed.append(
            {
                "source_hash": item["source_hash"],
                "replacements": replacements,
                "before_text_sha256": before_sha,
                "after_text_sha256": after_sha,
            }
        )
    result = {
        "created_at": base.utc_now(),
        "bundles_scanned": len(index),
        "bundles_changed": len(changed),
        "replacements": sum(int(item["replacements"]) for item in changed),
        "changes": changed,
        "complete": True,
    }
    base.write_json(state / "provenance_sanitization_report.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--glm-state", type=Path, default=DEFAULT_GLM_STATE)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stage-source")
    sub.add_parser("relocate-source")
    sub.add_parser("manifest")
    sub.add_parser("sanitize-provenance")
    sub.add_parser("select-evidence-pilot")
    sub.add_parser("select-judge-pilot")
    pilot = sub.add_parser("prepare-evidence-pilot")
    pilot.add_argument("--workers", type=int, default=4)
    prepare = sub.add_parser("prepare-evidence")
    prepare.add_argument("--hash", action="append", default=[])
    prepare.add_argument("--workers", type=int, default=4)
    cap = sub.add_parser("cap-visuals")
    cap.add_argument("--max-pages", type=int, default=base.MAX_VISUAL_PAGES_PER_BUNDLE)
    media = sub.add_parser("parse-media-google")
    media.add_argument("--model", default=qwen.GOOGLE_MEDIA_PARSER_MODEL)
    media.add_argument("--limit", type=int)
    audit = sub.add_parser("audit-inputs")
    audit.add_argument("--skip-source-rehash", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    state = args.state.resolve()
    baselines = args.baselines.resolve()
    glm_state = args.glm_state.resolve()
    canonical = args.canonical.resolve()
    try:
        if args.command == "stage-source":
            result = stage_source(state, glm_state=glm_state, canonical=canonical)
        elif args.command == "relocate-source":
            result = relocate_source(state)
        elif args.command == "manifest":
            result = build_manifest(state, baselines=baselines, glm_state=glm_state)
        elif args.command == "sanitize-provenance":
            result = sanitize_provenance(state)
        elif args.command == "select-evidence-pilot":
            result = base.select_evidence_pilot(state)
        elif args.command == "select-judge-pilot":
            rows = list(
                base.iter_jsonl(state / "comparison_manifest.unblinded.jsonl")
            )
            selected = base.select_pilot_rows(rows, minimum=24, state=state)
            result = {
                "comparison_ids": selected,
                "count": len(selected),
                "created_at": base.utc_now(),
            }
            base.write_json(state / "pilot_selection.json", result)
        elif args.command == "prepare-evidence-pilot":
            result = qwen.prepare_qwen_evidence_pilot(state, workers=args.workers)
        elif args.command == "prepare-evidence":
            result = qwen.prepare_qwen_evidence(
                state,
                only_hashes=set(args.hash) or None,
                workers=args.workers,
            )
        elif args.command == "cap-visuals":
            result = base.cap_existing_visuals(state, max_pages=args.max_pages)
        elif args.command == "parse-media-google":
            result = qwen.parse_pending_media_google(
                state, model=args.model, limit=args.limit
            )
        elif args.command == "audit-inputs":
            result = base.frozen_input_audit(
                state, verify_source_hashes=not args.skip_source_rehash
            )
        else:
            raise AssertionError(args.command)
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except base.EvaluationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
