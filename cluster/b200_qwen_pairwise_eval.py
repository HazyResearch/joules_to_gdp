#!/usr/bin/env python3
"""Adapt the completed Modal B200 Qwen run to the frozen pairwise workflow.

Only source staging differs from ``qwen_pairwise_eval.py``.  All evidence,
blinding, pilot selection, provider requests, aggregation, and validation reuse
the previously frozen Qwen implementation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import gdpval_pairwise_eval as base
import qwen_pairwise_eval as qwen


RUN_ID = (
    "20260807T230415Z_qwen36_27b_gdpval_occ_repeat4_full_norubric_"
    "modal_b200_1g_128k_vllm019"
)
STUDY_ID = "qwen36_27b_norubric_modal_b200_20260807"
EXPECTED_HF_REPO_ID = (
    "ScalingIntelligence/"
    "azeinali-qwen36-27b-gdpval-occ-repeat4-norubric-modal-b200"
)
EXPECTED_ARCHIVE_PREFIX = "archives/modal-b200-complete-all-volumes-20260815"
ARTIFACT_NAME = re.compile(
    r"^q(?P<local_query>[0-9]{4})_"
    r"(?P<task_id>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
    r"__rep(?P<repeat>[0-9]{2})$"
)


def _read_archive_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise base.EvaluationError(f"missing verified HF archive receipt: {path}")
    receipt = base.read_json(path)
    if not isinstance(receipt, dict) or receipt.get("status") != "ok":
        raise base.EvaluationError("HF archive receipt is not successful")
    if receipt.get("private") is not True:
        raise base.EvaluationError("HF archive receipt does not certify a private repo")
    if receipt.get("run_id") != RUN_ID:
        raise base.EvaluationError("HF archive receipt is for a different run")
    if receipt.get("repo_id") != EXPECTED_HF_REPO_ID:
        raise base.EvaluationError("HF archive receipt is for a different repository")
    if receipt.get("archive_prefix") != EXPECTED_ARCHIVE_PREFIX:
        raise base.EvaluationError("HF archive receipt is for a different archive prefix")
    if not str(receipt.get("revision") or ""):
        raise base.EvaluationError("HF archive receipt has no immutable revision")
    if not str(receipt.get("archive_sha256") or ""):
        raise base.EvaluationError("HF archive receipt has no archive SHA-256")
    return receipt


def _completion_payload(run_root: Path) -> dict[str, Any]:
    marker = run_root / "run.complete.ok"
    final_manifest = run_root / "final" / "final_manifest.json"
    if not marker.is_file() or not final_manifest.is_file():
        raise base.EvaluationError("completed B200 run markers are missing")
    completion = base.read_json(marker)
    final = base.read_json(final_manifest)
    completion_core = {
        key: value
        for key, value in completion.items()
        if key not in {"private_final_manifest_sha256", "private_run_complete_sha256"}
    }
    if completion_core != final:
        raise base.EvaluationError(
            "run.complete.ok differs from final_manifest.json beyond its private-copy hashes"
        )
    if completion.get("private_final_manifest_sha256") != base.sha256_file(final_manifest):
        raise base.EvaluationError("run.complete.ok has the wrong final-manifest hash")
    if completion.get("status") != "ok" or completion.get("strict_postprocess_status") != "ok":
        raise base.EvaluationError("B200 completion marker is not strict-clean")
    expected = {
        "rows": 880,
        "completed": 880,
        "zero_scores": 0,
        "missing_scores": 0,
        "timed_out": 0,
        "invalid_deliverables": 0,
        "judge_errors": 0,
    }
    anomalies = completion.get("anomaly_counts") or {}
    if any(int(anomalies.get(key, -1)) != value for key, value in expected.items()):
        raise base.EvaluationError("B200 completion anomalies are not all final-clean")
    if int(completion.get("private_backup_marker_count") or -1) != 44:
        raise base.EvaluationError("B200 completion marker does not certify 44 private backups")
    contract = completion.get("contract") or {}
    if contract.get("include_rubric_in_prompt") is not False:
        raise base.EvaluationError("B200 completion contract does not match the default prompt configuration")
    if contract.get("gpu") != "B200" or int(contract.get("gpu_count") or -1) != 1:
        raise base.EvaluationError("B200 completion contract has the wrong GPU assignment")
    return completion


def _relocated_glm_evidence_path(raw_path: Any, *, glm_state: Path) -> Path:
    """Resolve a frozen GLM evidence path after relocating its state directory.

    The frozen evidence index records absolute paths from the machine that
    produced it.  The complete state directory is copied unchanged to Modal,
    so resolve missing absolute paths relative to the copied state root while
    preserving the original within-state suffix.
    """

    original = Path(str(raw_path))
    if original.is_file():
        return original
    anchors = [index for index, part in enumerate(original.parts) if part == glm_state.name]
    if anchors:
        candidate = glm_state.joinpath(*original.parts[anchors[-1] + 1 :])
        if candidate.is_file():
            return candidate
    return original


def _verified_relocated_shared_evidence(
    *,
    glm_state: Path,
    shared_hashes: set[str],
) -> list[dict[str, Any]]:
    """Apply the frozen evidence checks using relocated, content-identical files."""

    frozen_path = glm_state / "frozen_input_audit.json"
    frozen = base.read_json(frozen_path) if frozen_path.is_file() else {}
    if (
        not frozen_path.is_file()
        or not bool(frozen.get("complete"))
        or frozen.get("manifest_sha256") != qwen.EXPECTED_GLM_MANIFEST_SHA256
    ):
        raise base.EvaluationError(
            f"GLM shared evidence is not backed by a complete frozen input audit: {frozen_path}"
        )
    index_path = glm_state / "evidence_index.json"
    by_hash = {
        str(item["source_hash"]): item
        for item in base.read_json(index_path)
        if isinstance(item, dict)
    }
    missing = sorted(shared_hashes - set(by_hash))
    if missing:
        raise base.EvaluationError(
            f"GLM evidence index lacks {len(missing)} shared content hashes: {missing[:10]}"
        )

    selected: list[dict[str, Any]] = []
    for source_hash in sorted(shared_hashes):
        item = dict(by_hash[source_hash])
        audit_path = _relocated_glm_evidence_path(
            item["audit_path"], glm_state=glm_state
        )
        text_path = _relocated_glm_evidence_path(
            item["text_path"], glm_state=glm_state
        )
        if not audit_path.is_file() or not text_path.is_file():
            raise base.EvaluationError(
                f"missing verified shared evidence for {source_hash}"
            )
        audit = base.read_json(audit_path)
        if (
            base.sha256_file(text_path) != audit.get("text_sha256")
            or item.get("parser_errors")
            or item.get("media_pending")
        ):
            raise base.EvaluationError(
                f"invalid verified shared evidence for {source_hash}"
            )
        item["audit_path"] = str(audit_path)
        item["text_path"] = str(text_path)
        visual_raw = item.get("visual_path")
        if visual_raw:
            visual = _relocated_glm_evidence_path(visual_raw, glm_state=glm_state)
            if (
                not visual.is_file()
                or base.sha256_file(visual) != audit.get("visual_sha256")
            ):
                raise base.EvaluationError(
                    f"invalid shared visual evidence for {source_hash}"
                )
            item["visual_path"] = str(visual)
        item["inherited_from_state"] = str(glm_state.resolve())
        item["inherited_by_content_hash"] = True
        selected.append(item)
    return selected


def _available_workspace_files(workspace: Path) -> list[Path]:
    return sorted(
        path
        for path in workspace.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and not any(part.lower() in qwen.IGNORED_DELIVERABLE_PARTS for part in path.relative_to(workspace).parts)
    )


def _resolve_local_deliverables(
    metadata: dict[str, Any], *, workspace: Path
) -> tuple[list[Path], list[dict[str, Any]]]:
    available = _available_workspace_files(workspace)
    resolved: list[Path] = []
    decisions: list[dict[str, Any]] = []
    for declared in qwen.declared_deliverable_values(metadata):
        relative = qwen.declared_workspace_relpath(declared)
        exact = workspace / relative.as_posix()
        matches = (
            [exact]
            if exact.is_file() and not exact.is_symlink()
            else base.files_under(exact)
            if exact.is_dir()
            else []
        )
        match_type = "exact_path_or_directory"
        if not matches:
            basename_matches = [path for path in available if path.name == relative.name]
            if len(basename_matches) == 1:
                matches = basename_matches
                match_type = "unique_basename_fallback"
        if not matches:
            raise base.EvaluationError(
                f"archive-integrity failure: declared deliverable {declared!r} "
                f"is missing under {workspace}"
            )
        matches = [
            path
            for path in matches
            if path.is_file()
            and not path.is_symlink()
            and not any(
                part.lower() in qwen.IGNORED_DELIVERABLE_PARTS
                for part in path.relative_to(workspace).parts
            )
        ]
        if not matches:
            raise base.EvaluationError(
                f"declared deliverable {declared!r} resolves only to excluded content"
            )
        decisions.append(
            {
                "declared_path": declared,
                "normalized_workspace_path": relative.as_posix(),
                "match_type": match_type,
                "files": [str(path.relative_to(workspace)) for path in matches],
            }
        )
        resolved.extend(matches)
    return base.dedupe_paths(sorted(resolved)), decisions


def stage_b200_source(
    state: Path,
    *,
    run_root: Path,
    glm_state: Path,
    archive_receipt_path: Path,
) -> dict[str, Any]:
    completion = _completion_payload(run_root)
    archive = _read_archive_receipt(archive_receipt_path)
    crosswalk = qwen.frozen_crosswalk(glm_state)
    crosswalk_ids = {str(row["agent_task_id"]) for row in crosswalk}
    artifact_root = run_root / "final" / "artifacts"
    artifact_dirs = sorted(path for path in artifact_root.iterdir() if path.is_dir())
    if len(artifact_dirs) != 880:
        raise base.EvaluationError(
            f"expected exactly 880 canonical artifact directories, found {len(artifact_dirs)}"
        )

    selected: list[dict[str, Any]] = []
    inventory_files: list[dict[str, Any]] = []
    seen_task_repeats: set[tuple[str, int]] = set()
    source_indices: dict[int, set[int]] = defaultdict(set)
    roots = [run_root]

    for index, artifact in enumerate(artifact_dirs, start=1):
        match = ARTIFACT_NAME.fullmatch(artifact.name)
        if not match:
            raise base.EvaluationError(f"noncanonical B200 artifact name: {artifact.name}")
        metadata_path = artifact / "metadata.json"
        workspace = artifact / "workspace"
        if not metadata_path.is_file() or not workspace.is_dir():
            raise base.EvaluationError(f"incomplete canonical artifact: {artifact}")
        metadata = base.read_json(metadata_path)
        if not isinstance(metadata, dict):
            raise base.EvaluationError(f"metadata is not an object: {metadata_path}")

        task_id = str(metadata.get("task_id") or "")
        repeat = int(metadata.get("repeat_index", -1))
        source_task_index = int(metadata.get("source_task_index", -1))
        failures: list[str] = []
        if metadata.get("include_rubric_in_prompt") is not False:
            failures.append("include_rubric_in_prompt is not false")
        if metadata.get("completed") is not True:
            failures.append("completed is not true")
        if metadata.get("timed_out") is not False:
            failures.append("timed_out is not false")
        if task_id not in crosswalk_ids:
            failures.append("task_id is absent from the frozen crosswalk")
        if repeat not in range(4):
            failures.append(f"repeat_index={repeat}")
        if source_task_index not in range(220):
            failures.append(f"source_task_index={source_task_index}")
        if task_id != match.group("task_id"):
            failures.append("artifact-name task UUID differs from metadata")
        if repeat != int(match.group("repeat")):
            failures.append("artifact-name repeat differs from metadata")
        if str(metadata.get("query_id") or "") != f"q{int(match.group('local_query')):04d}":
            failures.append("artifact-name local query differs from metadata")
        if failures:
            raise base.EvaluationError(
                f"{artifact.name} failed canonical metadata validation: " + "; ".join(failures)
            )

        key = (task_id, repeat)
        if key in seen_task_repeats:
            raise base.EvaluationError(f"duplicate B200 task/repeat: {key}")
        seen_task_repeats.add(key)
        source_indices[source_task_index].add(repeat)

        deliverables, resolution = _resolve_local_deliverables(
            metadata, workspace=workspace
        )
        metadata_record = base.source_file_record(metadata_path, roots=roots)
        metadata_record.update(
            {
                "role": "canonical_metadata",
                "archive_member": f"primary_run/{metadata_path.relative_to(run_root)}",
            }
        )
        inventory_files.append(metadata_record)
        deliverable_records: list[dict[str, Any]] = []
        for path in deliverables:
            record = base.source_file_record(path, roots=roots)
            record.update(
                {
                    "role": "declared_deliverable",
                    "archive_member": f"primary_run/{path.relative_to(run_root)}",
                }
            )
            inventory_files.append(record)
            deliverable_records.append(record)

        global_query = source_task_index * 4 + repeat
        archive_artifact = f"primary_run/{artifact.relative_to(run_root)}"
        selected.append(
            {
                "task_id": task_id,
                "repeat": repeat,
                "source_task_index": source_task_index,
                "completed": True,
                "timed_out": False,
                "include_rubric_in_prompt": False,
                "run_name": RUN_ID,
                "query_index": global_query,
                "query_id": f"q{global_query:04d}",
                "source_local_query_id": str(metadata.get("query_id") or ""),
                "artifact_hf_path": (
                    f"hf://datasets/{archive['repo_id']}@{archive['revision']}/"
                    f"{archive['archive_prefix']}#{archive_artifact}"
                ),
                "artifact_archive_member": archive_artifact,
                "metadata_hf_path": (
                    f"{archive['archive_prefix']}#"
                    f"primary_run/{metadata_path.relative_to(run_root)}"
                ),
                "metadata_path": str(metadata_path),
                "declared_deliverables": qwen.declared_deliverable_values(metadata),
                "deliverable_resolution": resolution,
                "deliverable_files": deliverable_records,
                "candidate_quality_status": (
                    "missing"
                    if not deliverables
                    else "empty_or_invalid"
                    if any(path.stat().st_size == 0 for path in deliverables)
                    else "present"
                ),
            }
        )
        if index % 100 == 0 or index == len(artifact_dirs):
            print(f"validated B200 canonical artifacts {index}/{len(artifact_dirs)}", flush=True)

    if len(selected) != 880 or len(seen_task_repeats) != 880:
        raise base.EvaluationError("B200 staging did not produce 880 unique task/repeats")
    if set(source_indices) != set(range(220)):
        raise base.EvaluationError("B200 source indices are not exactly 0-219")
    bad_repeats = {
        source_index: sorted(repeats)
        for source_index, repeats in source_indices.items()
        if repeats != {0, 1, 2, 3}
    }
    if bad_repeats:
        raise base.EvaluationError(
            f"B200 source indices without repeats 0-3: {list(bad_repeats.items())[:10]}"
        )
    if len({row["task_id"] for row in selected}) != 220:
        raise base.EvaluationError("B200 source does not contain exactly 220 unique tasks")

    selected.sort(
        key=lambda row: (int(row["source_task_index"]), int(row["repeat"]), str(row["task_id"]))
    )
    state.mkdir(parents=True, exist_ok=True)
    base.write_json(state / "qwen_source_outputs.json", selected)
    inventory = {
        "schema_version": "qwen36-modal-b200-source-v1",
        "created_at": base.utc_now(),
        "run_id": RUN_ID,
        "selection_rule": "all and only strict-clean canonical final B200 artifacts",
        "selected_outputs": len(selected),
        "unique_tasks": len({row["task_id"] for row in selected}),
        "unique_task_repeats": len(seen_task_repeats),
        "metadata_files": sum(row["role"] == "canonical_metadata" for row in inventory_files),
        "deliverable_files": sum(row["role"] == "declared_deliverable" for row in inventory_files),
        "files": sorted(inventory_files, key=lambda row: str(row["archive_member"])),
        "excluded_content_classes": sorted(qwen.IGNORED_DELIVERABLE_PARTS),
        "hf_repo_id": archive["repo_id"],
        "hf_revision": archive["revision"],
        "requested_revision": archive["revision"],
        "resolved_revision": archive["revision"],
        "hf_archive_prefix": archive["archive_prefix"],
        "hf_archive_sha256": archive["archive_sha256"],
        "completion_traces_sha256": completion["traces_sha256"],
        "completion_batches_sha256": completion["batches_sha256"],
        "approved_artifact_salvage_query_ids": completion.get(
            "approved_artifact_salvage_query_ids"
        )
        or [],
        "integrity_policy": (
            "Every declared deliverable must resolve inside its strict-clean canonical "
            "workspace; ignored dependency/cache trees and symlinks are excluded."
        ),
    }
    base.write_json(state / "frozen_source_inventory.json", inventory)
    summary = {
        "created_at": base.utc_now(),
        "run_id": RUN_ID,
        "hf_repo_id": archive["repo_id"],
        "hf_revision": archive["revision"],
        "hf_archive_sha256": archive["archive_sha256"],
        "outputs": 880,
        "tasks": 220,
        "repeats": 4,
        "condition": qwen.CONDITION,
        "selection_by_run": {RUN_ID: 880},
        "quality_status": dict(Counter(row["candidate_quality_status"] for row in selected)),
        "frozen_source_inventory_sha256": base.sha256_file(
            state / "frozen_source_inventory.json"
        ),
    }
    base.write_json(state / "source_stage_summary.json", summary)
    return summary


def _configure_qwen_provenance(archive: dict[str, Any]) -> None:
    qwen.STUDY_ID = STUDY_ID
    qwen.HF_REPO_ID = str(archive["repo_id"])
    qwen.HF_REVISION = str(archive["revision"])
    qwen.HF_RUN_REL = PurePosixPath(EXPECTED_ARCHIVE_PREFIX)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--archive-receipt", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--glm-state", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stage-source")
    sub.add_parser("manifest")
    sub.add_parser("select-evidence-pilot")
    sub.add_parser("select-judge-pilot")
    prepare_pilot = sub.add_parser("prepare-evidence-pilot")
    prepare_pilot.add_argument("--workers", type=int, default=4)
    prepare = sub.add_parser("prepare-evidence")
    prepare.add_argument("--hash", action="append", default=[])
    prepare.add_argument("--workers", type=int, default=4)
    cap = sub.add_parser("cap-visuals")
    cap.add_argument("--max-pages", type=int, default=base.MAX_VISUAL_PAGES_PER_BUNDLE)
    audit = sub.add_parser("audit-inputs")
    audit.add_argument("--skip-source-rehash", action="store_true")
    media = sub.add_parser("parse-media-google")
    media.add_argument("--model", default=qwen.GOOGLE_MEDIA_PARSER_MODEL)
    media.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    state = args.state.resolve()
    run_root = args.run_root.resolve()
    archive_path = args.archive_receipt.resolve()
    baselines = args.baselines.resolve()
    glm_state = args.glm_state.resolve()
    try:
        archive = _read_archive_receipt(archive_path)
        _configure_qwen_provenance(archive)
        if args.command == "stage-source":
            result = stage_b200_source(
                state,
                run_root=run_root,
                glm_state=glm_state,
                archive_receipt_path=archive_path,
            )
        elif args.command == "manifest":
            qwen.verified_shared_evidence = _verified_relocated_shared_evidence
            result = qwen.build_manifest(state, baselines=baselines, glm_state=glm_state)
        elif args.command == "select-evidence-pilot":
            result = base.select_evidence_pilot(state)
        elif args.command == "select-judge-pilot":
            rows = list(base.iter_jsonl(state / "comparison_manifest.unblinded.jsonl"))
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
        elif args.command == "audit-inputs":
            result = base.frozen_input_audit(
                state, verify_source_hashes=not args.skip_source_rehash
            )
        elif args.command == "parse-media-google":
            result = qwen.parse_pending_media_google(
                state,
                model=args.model,
                limit=args.limit,
            )
        else:
            raise base.EvaluationError(f"unsupported command: {args.command}")
    except base.EvaluationError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
