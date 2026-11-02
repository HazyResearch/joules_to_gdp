#!/usr/bin/env python3
"""Prepare and validate the pinned Qwen3.6-27B GDPval experiment."""

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


HF_REPO_ID = os.environ.get(
    "QWEN_PAIRWISE_HF_REPO_ID",
    "ScalingIntelligence/qwen36-27b-gdpval-occ-repeat4-results-20260717",
)
HF_REVISION = os.environ.get(
    "QWEN_PAIRWISE_HF_REVISION",
    "46f8f4acba0d6bec24beec5d726a0ade81d181c2",
)
HF_REPO_TYPE = "dataset"
STUDY_ID = os.environ.get(
    "QWEN_PAIRWISE_STUDY_ID",
    "qwen36_27b_norubric_hf_20260717",
)
CONDITION = "without_rubric"
GOOGLE_MEDIA_PARSER_MODEL = "models/gemini-3.1-flash-lite"
HF_RUN_REL = PurePosixPath(
    os.environ.get(
        "QWEN_PAIRWISE_HF_RUN_REL",
        "runs/gdpval_smoke/run_openhands_Qwen_Qwen3_6_27B_gdpval",
    )
)
EXPECTED_BASELINE_SHA256 = "050c869501a125fe33e58cb5d2f3cc53ccde0d19f827e78043888d1519c7360e"
EXPECTED_GLM_CROSSWALK_SHA256 = "b04e200aab9c0c57df64999732befb2434e4593c087519655a5812142cd6bfa2"
EXPECTED_GLM_MANIFEST_SHA256 = "532e98f9f65ede097f468369690cf219461a38ec7961c315882270afab89f7ca"

DEFAULT_STATE = Path("reports/qwen36_27b_norubric_source_20260723")
DEFAULT_BASELINES = Path("GDPVal_220_Tasks")
DEFAULT_GLM_STATE = Path("reports/gdpval_pairwise_glm52_20260715")

IGNORED_DELIVERABLE_PARTS = {
    ".cache",
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "bower_components",
    "node_modules",
    "site-packages",
    "venv",
}


@dataclass(frozen=True)
class SourceShard:
    run_name: str
    first_query: int
    count: int
    source_task_start: int

    @property
    def artifact_root(self) -> str:
        return f"runs/{self.run_name}/{HF_RUN_REL.as_posix()}/artifacts"


SOURCE_SHARDS = (
    SourceShard("norubric_full_first840", 0, 840, 0),
    SourceShard("norubric_resume_from840_count40_partial840_859", 0, 20, 210),
    SourceShard("norubric_resume_from860_count20", 0, 20, 215),
)

EXCLUSIONS = (
    {
        "run_name": "norubric_full_first840",
        "query_range": "q0840-q0859",
        "reason": "incomplete artifact directories lack metadata",
    },
    {
        "run_name": "norubric_resume_from840_count40_partial840_859",
        "query_range": "q0020-q0039",
        "reason": "timed out, incomplete, or missing metadata",
    },
)

GOOGLE_MEDIA_PARSER_SCHEMA = {
    "type": "object",
    "properties": {
        "transcript": {"type": "string"},
        "nonverbal_description": {"type": "string"},
        "language": {"type": "string"},
        "processing_notes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "transcript",
        "nonverbal_description",
        "language",
        "processing_notes",
    ],
    "additionalProperties": False,
}


def require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or ""
    if not token:
        raise base.EvaluationError(
            "HF_TOKEN is not set in the process environment; do not pass Hugging Face tokens on the command line"
        )
    return token


def parse_google_audio_chunk(
    client: Any,
    path: Path,
    *,
    model: str,
) -> dict[str, Any]:
    from google.genai import types as google_types

    prompt = (
        "Analyze this audio as neutral evidence for a blinded professional-deliverable "
        "evaluation. Do not infer who created it. Return one JSON object with keys: "
        "transcript (verbatim speech, or an empty string), nonverbal_description "
        "(music, sound effects, ambience, silence, timing, delivery, and audible "
        "production defects), language, and processing_notes (array of strings describing extraction methods). "
        "Be factual and source-blind."
    )
    content = google_types.Content(
        role="user",
        parts=[
            google_types.Part.from_bytes(
                data=path.read_bytes(),
                mime_type="audio/wav",
            ),
            google_types.Part.from_text(text=prompt),
        ],
    )
    failures: list[dict[str, Any]] = []
    last_text = ""
    last_data: dict[str, Any] = {}
    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[content],
                config=google_types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=8_000,
                    response_mime_type="application/json",
                    response_json_schema=GOOGLE_MEDIA_PARSER_SCHEMA,
                ),
            )
            last_data = response.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
            )
            last_text = str(response.text or "")
            parsed = base.extract_json_from_text(last_text)
            parsed["api_metadata"] = {
                "provider": "google",
                "model_requested": model,
                "model_returned": (
                    last_data.get("modelVersion")
                    or last_data.get("model_version")
                    or model
                ),
                "response_id": (
                    last_data.get("responseId")
                    or last_data.get("response_id")
                ),
                "usage": (
                    last_data.get("usageMetadata")
                    or last_data.get("usage_metadata")
                    or {}
                ),
                "attempts": attempt,
                "failed_attempts": failures,
            }
            return parsed
        except Exception as exc:
            failures.append(
                {
                    "attempt": attempt,
                    "failure": f"{type(exc).__name__}: {exc}",
                    "response_id": (
                        last_data.get("responseId")
                        or last_data.get("response_id")
                    ),
                    "usage": (
                        last_data.get("usageMetadata")
                        or last_data.get("usage_metadata")
                        or {}
                    ),
                    "output_sha256": (
                        base.sha256_text(last_text) if last_text else None
                    ),
                }
            )
    if last_text.strip():
        return {
            "transcript": "",
            "nonverbal_description": last_text.strip(),
            "language": "undetermined",
            "processing_notes": [
                "Extraction format: model prose, preserved in nonverbal_description.",
            ],
            "api_metadata": {
                "provider": "google",
                "model_requested": model,
                "model_returned": (
                    last_data.get("modelVersion")
                    or last_data.get("model_version")
                    or model
                ),
                "response_id": (
                    last_data.get("responseId")
                    or last_data.get("response_id")
                ),
                "usage": (
                    last_data.get("usageMetadata")
                    or last_data.get("usage_metadata")
                    or {}
                ),
                "attempts": 3,
                "failed_attempts": failures,
                "unstructured_fallback": True,
            },
        }
    raise base.EvaluationError(
        "Gemini audio parser returned no usable content after three attempts: "
        f"{failures}"
    )


def parse_pending_media_google(
    state: Path,
    *,
    model: str = GOOGLE_MEDIA_PARSER_MODEL,
    limit: int | None = None,
) -> dict[str, Any]:
    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY") or ""
    if not api_key:
        raise base.EvaluationError(
            "GOOGLE_API_KEY is not set in the process environment"
        )
    evidence = base.read_json(state / "evidence_index.json")
    pending: dict[str, dict[str, Any]] = {}
    for bundle in evidence:
        for item in bundle.get("media_pending") or []:
            pending.setdefault(str(item["cache_key"]), item)
    cache_path = state / "media_parser_cache.json"
    cache = base.read_json(cache_path) if cache_path.is_file() else {}
    missing_all = {
        key: item for key, item in sorted(pending.items()) if key not in cache
    }
    missing = dict(list(missing_all.items())[:limit]) if limit else missing_all
    if not missing:
        return {
            "provider": "google",
            "model": model,
            "pending": len(pending),
            "newly_parsed": 0,
            "cached": len(cache),
            "remaining": len(missing_all),
            "evidence_rebuild_required": False,
        }
    client = genai.Client(api_key=api_key)
    for index, (cache_key, item) in enumerate(missing.items(), start=1):
        audio_path = Path(str(item["audio_path"]))
        if not audio_path.is_file():
            raise base.EvaluationError(
                f"sanitized pending audio is absent: {audio_path}"
            )
        if audio_path.stat().st_size >= 20 * 1024**2:
            raise base.EvaluationError(
                f"sanitized audio exceeds Gemini's 20 MB inline limit: {audio_path}"
            )
        parsed = parse_google_audio_chunk(
            client,
            audio_path,
            model=model,
        )
        cache[cache_key] = {
            "schema_version": base.SCHEMA_VERSION,
            "parser_provider": "google",
            "parser_model": model,
            "parsed_at": base.utc_now(),
            "chunk_count": 1,
            "chunks": [parsed],
        }
        base.write_json(cache_path, cache)
        print(f"parsed Gemini media {index}/{len(missing)}", flush=True)
    return {
        "provider": "google",
        "model": model,
        "pending": len(pending),
        "newly_parsed": len(missing),
        "cached": len(cache),
        "remaining": len(missing_all) - len(missing),
        "evidence_rebuild_required": True,
    }


def query_index_from_name(name: str) -> int | None:
    match = re.match(r"^q(\d{4})(?:\D|$)", name)
    return int(match.group(1)) if match else None


def selected_artifact_paths(entries: Iterable[Any], shard: SourceShard) -> dict[int, str]:
    selected: dict[int, str] = {}
    lower = shard.first_query
    upper = shard.first_query + shard.count
    prefix = shard.artifact_root.rstrip("/") + "/"
    for entry in entries:
        path = str(getattr(entry, "path", "") or "")
        if not path.startswith(prefix):
            continue
        relative = path[len(prefix) :]
        artifact_name = relative.split("/", 1)[0]
        query_index = query_index_from_name(artifact_name)
        if query_index is None or not lower <= query_index < upper:
            continue
        artifact_path = prefix + artifact_name
        prior = selected.setdefault(query_index, artifact_path)
        if prior != artifact_path:
            raise base.EvaluationError(
                f"{shard.run_name}: multiple artifact directories for q{query_index:04d}: "
                f"{prior!r}, {artifact_path!r}"
            )
    expected = set(range(lower, upper))
    missing = sorted(expected - set(selected))
    if missing:
        raise base.EvaluationError(
            f"{shard.run_name}: missing selected artifact directories: "
            f"{[f'q{value:04d}' for value in missing[:20]]}"
        )
    return selected


def declared_deliverable_values(metadata: dict[str, Any]) -> list[str]:
    raw: list[str] = []
    validation = metadata.get("deliverable_validation")
    if isinstance(validation, dict):
        raw.extend(str(value) for value in validation.get("primary_files") or [])
    if not raw:
        raw.extend(str(value) for value in metadata.get("deliverable_files") or [])
    return list(dict.fromkeys(value for value in raw if value.strip()))


def declared_workspace_relpath(value: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    parts = list(PurePosixPath(normalized).parts)
    if "workspace" in parts:
        reverse_index = parts[::-1].index("workspace")
        parts = parts[len(parts) - reverse_index :]
    else:
        parts = [part for part in parts if part not in {"/", ""}]
    if not parts or any(part in {".", ".."} for part in parts):
        raise base.EvaluationError(f"unsafe or empty declared deliverable path: {value!r}")
    return PurePosixPath(*parts)


def ignored_remote_path(path: str, workspace_root: str) -> bool:
    relative = PurePosixPath(path).relative_to(PurePosixPath(workspace_root))
    return any(part.lower() in IGNORED_DELIVERABLE_PARTS for part in relative.parts)


def resolve_declared_remote_files(
    metadata: dict[str, Any],
    *,
    artifact_path: str,
    workspace_files: Sequence[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    workspace_root = f"{artifact_path}/workspace"
    available = sorted(
        {
            str(path)
            for path in workspace_files
            if str(path).startswith(workspace_root + "/")
            and not ignored_remote_path(str(path), workspace_root)
        }
    )
    resolved: list[str] = []
    decisions: list[dict[str, Any]] = []
    for declared in declared_deliverable_values(metadata):
        rel = declared_workspace_relpath(declared)
        exact = f"{workspace_root}/{rel.as_posix()}"
        matches = [path for path in available if path == exact or path.startswith(exact.rstrip("/") + "/")]
        match_type = "exact_path_or_directory"
        if not matches:
            basename_matches = [path for path in available if PurePosixPath(path).name == rel.name]
            if len(basename_matches) == 1:
                matches = basename_matches
                match_type = "unique_basename_fallback"
        if not matches:
            raise base.EvaluationError(
                f"archive-integrity failure: metadata declares {declared!r}, "
                f"but no matching file exists under {workspace_root}"
            )
        decisions.append(
            {
                "declared_path": declared,
                "normalized_workspace_path": rel.as_posix(),
                "match_type": match_type,
                "remote_files": matches,
            }
        )
        resolved.extend(matches)
    return sorted(set(resolved)), decisions


def copy_hf_file(
    *,
    repo_path: str,
    state: Path,
    token: str,
) -> Path:
    from huggingface_hub import hf_hub_download

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


def deliverable_local_path(state: Path, repo_path: str) -> Path:
    """Map a case-sensitive HF path to a collision-safe local object path."""
    path_digest = hashlib.sha256(repo_path.encode("utf-8")).hexdigest()
    return (
        state
        / "source"
        / "_objects"
        / path_digest[:2]
        / path_digest[2:]
        / PurePosixPath(repo_path).name
    )


def copy_hf_deliverable(
    *,
    repo_path: str,
    state: Path,
    token: str,
) -> Path:
    """Download one HF deliverable without aliasing case-distinct repository paths."""
    from huggingface_hub import hf_hub_download

    destination = deliverable_local_path(state, repo_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination
    path_digest = hashlib.sha256(repo_path.encode("utf-8")).hexdigest()
    transfer_root = state / ".hf_transfers" / path_digest
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


def remote_file_inventory(api: Any, *, artifact_path: str) -> dict[str, int]:
    from huggingface_hub.errors import HfHubHTTPError

    workspace_root = f"{artifact_path}/workspace"
    try:
        entries = api.list_repo_tree(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            revision=HF_REVISION,
            path_in_repo=workspace_root,
            recursive=True,
            expand=False,
        )
        return {
            str(getattr(entry, "path", "") or ""): int(getattr(entry, "size", 0) or 0)
            for entry in entries
            if getattr(entry, "path", None) and hasattr(entry, "size")
        }
    except HfHubHTTPError as exc:
        raise base.EvaluationError(
            f"archive-integrity failure: missing workspace tree for {artifact_path}: {exc}"
        ) from exc


def validate_selected_metadata(
    metadata: dict[str, Any],
    *,
    shard: SourceShard,
    query_index: int,
    crosswalk_task_ids: set[str],
) -> dict[str, Any]:
    task_id = str(metadata.get("task_id") or "")
    repeat = int(metadata.get("repeat_index", -1))
    source_task_index = int(metadata.get("source_task_index", -1))
    failures: list[str] = []
    if metadata.get("include_rubric_in_prompt") is not False:
        failures.append("include_rubric_in_prompt is not the JSON boolean false")
    if metadata.get("completed") is not True:
        failures.append("completed is not the JSON boolean true")
    if metadata.get("timed_out") is not False:
        failures.append("timed_out is not the JSON boolean false")
    if task_id not in crosswalk_task_ids:
        failures.append(f"task_id {task_id!r} is absent from the frozen GLM crosswalk")
    if repeat not in range(4):
        failures.append(f"repeat_index={repeat}, expected one of 0, 1, 2, 3")
    source_task_stop = shard.source_task_start + shard.count // 4
    if source_task_index not in range(shard.source_task_start, source_task_stop):
        failures.append(
            f"source_task_index={source_task_index}, "
            f"expected {shard.source_task_start}-{source_task_stop - 1} for this shard"
        )
    if failures:
        raise base.EvaluationError(
            f"{shard.run_name}/q{query_index:04d} failed canonical metadata validation: "
            + "; ".join(failures)
        )
    return {
        "task_id": task_id,
        "repeat": repeat,
        "source_task_index": source_task_index,
        "completed": True,
        "timed_out": False,
        "include_rubric_in_prompt": False,
    }


def frozen_crosswalk(glm_state: Path) -> list[dict[str, Any]]:
    path = glm_state / "crosswalk.json"
    if not path.is_file():
        raise base.EvaluationError(f"missing frozen GLM crosswalk: {path}")
    if base.sha256_file(path) != EXPECTED_GLM_CROSSWALK_SHA256:
        raise base.EvaluationError(f"frozen GLM crosswalk SHA-256 mismatch: {path}")
    rows = base.read_json(path)
    if not isinstance(rows, list) or len(rows) != 220:
        raise base.EvaluationError(f"frozen GLM crosswalk must contain 220 rows: {path}")
    if len({str(row["agent_task_id"]) for row in rows}) != 220:
        raise base.EvaluationError("frozen GLM crosswalk agent task IDs are not unique")
    return rows


def stage_source(state: Path, *, glm_state: Path) -> dict[str, Any]:
    from huggingface_hub import HfApi

    token = require_hf_token()
    crosswalk = frozen_crosswalk(glm_state)
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
            f"HF revision mismatch: requested {HF_REVISION}, resolved {resolved_revision}"
        )

    inventory_files: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    seen_task_repeats: set[tuple[str, int]] = set()
    source_indices: dict[int, set[int]] = defaultdict(set)
    for shard in SOURCE_SHARDS:
        entries = list(
            api.list_repo_tree(
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                revision=HF_REVISION,
                path_in_repo=shard.artifact_root,
                recursive=False,
                expand=False,
            )
        )
        artifacts = selected_artifact_paths(entries, shard)
        for query_index, artifact_path in sorted(artifacts.items()):
            metadata_repo_path = f"{artifact_path}/metadata.json"
            metadata_path = copy_hf_file(
                repo_path=metadata_repo_path,
                state=state,
                token=token,
            )
            metadata = base.read_json(metadata_path)
            if not isinstance(metadata, dict):
                raise base.EvaluationError(f"metadata is not an object: {metadata_repo_path}")
            canonical = validate_selected_metadata(
                metadata,
                shard=shard,
                query_index=query_index,
                crosswalk_task_ids=crosswalk_ids,
            )
            declared_values = declared_deliverable_values(metadata)
            workspace_inventory = (
                remote_file_inventory(api, artifact_path=artifact_path)
                if declared_values
                else {}
            )
            deliverable_repo_paths, decisions = resolve_declared_remote_files(
                metadata,
                artifact_path=artifact_path,
                workspace_files=list(workspace_inventory),
            )
            metadata_record = base.source_file_record(
                metadata_path, roots=[state / "source"]
            )
            metadata_record["hf_path"] = metadata_repo_path
            metadata_record["role"] = "canonical_metadata"
            inventory_files.append(metadata_record)
            key = (canonical["task_id"], canonical["repeat"])
            if key in seen_task_repeats:
                raise base.EvaluationError(f"duplicate selected Qwen task/repeat: {key}")
            seen_task_repeats.add(key)
            source_indices[canonical["source_task_index"]].add(canonical["repeat"])
            selected_records.append(
                {
                    **canonical,
                    "run_name": shard.run_name,
                    "query_index": query_index,
                    "query_id": f"q{query_index:04d}",
                    "artifact_hf_path": artifact_path,
                    "metadata_hf_path": metadata_repo_path,
                    "metadata_path": str(metadata_path),
                    "declared_deliverables": declared_values,
                    "deliverable_resolution": decisions,
                    "deliverable_hf_paths": deliverable_repo_paths,
                    "deliverable_remote_sizes": {
                        repo_path: workspace_inventory[repo_path]
                        for repo_path in deliverable_repo_paths
                    },
                    "deliverable_files": [],
                    "candidate_quality_status": (
                        "missing"
                        if not deliverable_repo_paths
                        else "empty_or_invalid"
                        if any(
                            int(workspace_inventory[repo_path]) == 0
                            for repo_path in deliverable_repo_paths
                        )
                        else "present"
                    ),
                }
            )

    if len(selected_records) != 880 or len(seen_task_repeats) != 880:
        raise base.EvaluationError(
            f"expected 880 unique selected outputs, got rows={len(selected_records)}, "
            f"task_repeats={len(seen_task_repeats)}"
        )
    if set(source_indices) != set(range(220)):
        raise base.EvaluationError(
            f"source task indices are not exactly 0-219: {sorted(source_indices)}"
        )
    bad_repeats = {
        index: sorted(repeats)
        for index, repeats in source_indices.items()
        if repeats != {0, 1, 2, 3}
    }
    if bad_repeats:
        raise base.EvaluationError(
            f"source task indices without repeats 0-3: {list(bad_repeats.items())[:10]}"
        )
    if len({record["task_id"] for record in selected_records}) != 220:
        raise base.EvaluationError("selected source does not contain exactly 220 task IDs")

    planned_files: dict[str, int] = {}
    for selected in selected_records:
        for repo_path, size in selected["deliverable_remote_sizes"].items():
            prior = planned_files.setdefault(str(repo_path), int(size))
            if prior != int(size):
                raise base.EvaluationError(
                    f"HF reports conflicting sizes for {repo_path}: {prior}, {size}"
                )
    source_root = state / "source"
    missing_bytes = 0
    existing_bytes = 0
    for repo_path, expected_size in planned_files.items():
        local_path = deliverable_local_path(state, repo_path)
        if local_path.is_file():
            if local_path.stat().st_size != expected_size:
                raise base.EvaluationError(
                    f"existing staged file has wrong size: {local_path}; "
                    f"expected {expected_size}, got {local_path.stat().st_size}"
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
        "selected_outputs": len(selected_records),
        "metadata_files_staged": sum(
            record["role"] == "canonical_metadata" for record in inventory_files
        ),
        "unique_deliverable_files": len(planned_files),
        "total_deliverable_bytes": sum(planned_files.values()),
        "already_staged_deliverable_bytes": existing_bytes,
        "remaining_download_bytes": missing_bytes,
        "free_bytes_before_deliverables": free_bytes,
        "required_free_space_reserve_bytes": reserve_bytes,
        "fits_with_reserve": missing_bytes + reserve_bytes <= free_bytes,
        "files": [
            {"hf_path": repo_path, "size_bytes": size}
            for repo_path, size in sorted(planned_files.items())
        ],
    }
    base.write_json(state / "source_download_plan.json", download_plan)
    if not download_plan["fits_with_reserve"]:
        raise base.EvaluationError(
            "canonical HF deliverables do not fit with the 5 GiB safety reserve; "
            f"remaining={missing_bytes:,} bytes, free={free_bytes:,} bytes. "
            "See source_download_plan.json."
        )

    downloaded_records: dict[str, dict[str, Any]] = {}
    for index, (repo_path, expected_size) in enumerate(sorted(planned_files.items()), start=1):
        local_path = copy_hf_deliverable(
            repo_path=repo_path,
            state=state,
            token=token,
        )
        if local_path.stat().st_size != expected_size:
            raise base.EvaluationError(
                f"downloaded file size mismatch for {repo_path}: "
                f"expected {expected_size}, got {local_path.stat().st_size}"
            )
        record = base.source_file_record(local_path, roots=[source_root])
        record["hf_path"] = repo_path
        record["role"] = "declared_deliverable"
        inventory_files.append(record)
        downloaded_records[repo_path] = record
        if index % 50 == 0 or index == len(planned_files):
            print(f"staged declared deliverables {index}/{len(planned_files)}", flush=True)
    for selected in selected_records:
        selected["deliverable_files"] = [
            downloaded_records[repo_path]
            for repo_path in selected.pop("deliverable_hf_paths")
        ]
        selected.pop("deliverable_remote_sizes")

    selected_records.sort(
        key=lambda record: (
            int(record["source_task_index"]),
            int(record["repeat"]),
            str(record["task_id"]),
        )
    )
    state.mkdir(parents=True, exist_ok=True)
    base.write_json(state / "qwen_source_outputs.json", selected_records)
    inventory = {
        "schema_version": "qwen36-hf-source-v1",
        "created_at": base.utc_now(),
        "hf_repo_id": HF_REPO_ID,
        "hf_repo_type": HF_REPO_TYPE,
        "requested_revision": HF_REVISION,
        "resolved_revision": resolved_revision,
        "selection_rule": "exact canonical shards and query ranges only",
        "selected_ranges": [
            {
                "run_name": shard.run_name,
                "artifact_root": shard.artifact_root,
                "first_query": f"q{shard.first_query:04d}",
                "last_query": f"q{shard.first_query + shard.count - 1:04d}",
                "count": shard.count,
                "source_task_start": shard.source_task_start,
            }
            for shard in SOURCE_SHARDS
        ],
        "exclusions": list(EXCLUSIONS),
        "selected_outputs": len(selected_records),
        "unique_tasks": len({record["task_id"] for record in selected_records}),
        "unique_task_repeats": len(seen_task_repeats),
        "metadata_files": sum(record["role"] == "canonical_metadata" for record in inventory_files),
        "deliverable_files": sum(record["role"] == "declared_deliverable" for record in inventory_files),
        "files": sorted(inventory_files, key=lambda record: str(record["hf_path"])),
        "excluded_content_classes": sorted(IGNORED_DELIVERABLE_PARTS),
        "integrity_policy": (
            "Every declared deliverable must resolve to an HF file. Empty or invalid submissions are "
            "preserved only when metadata declares no deliverable or a present zero-byte deliverable."
        ),
    }
    base.write_json(state / "frozen_source_inventory.json", inventory)
    summary = {
        "created_at": base.utc_now(),
        "hf_repo_id": HF_REPO_ID,
        "hf_revision": HF_REVISION,
        "outputs": len(selected_records),
        "tasks": 220,
        "repeats": 4,
        "condition": CONDITION,
        "selection_by_run": dict(Counter(record["run_name"] for record in selected_records)),
        "quality_status": dict(
            Counter(record["candidate_quality_status"] for record in selected_records)
        ),
        "frozen_source_inventory_sha256": base.sha256_file(
            state / "frozen_source_inventory.json"
        ),
    }
    base.write_json(state / "source_stage_summary.json", summary)
    return summary


def glm_templates(glm_state: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = glm_state / "comparison_manifest.unblinded.jsonl"
    if not path.is_file():
        raise base.EvaluationError(f"missing frozen GLM comparison manifest: {path}")
    if base.sha256_file(path) != EXPECTED_GLM_MANIFEST_SHA256:
        raise base.EvaluationError(f"frozen GLM manifest SHA-256 mismatch: {path}")
    templates: dict[tuple[str, str], dict[str, Any]] = {}
    for row in base.iter_jsonl(path):
        if row["condition"] != CONDITION:
            continue
        key = (str(row["task_id"]), str(row["baseline"]))
        templates.setdefault(key, row)
    if len(templates) != 220 * len(base.BASELINES):
        raise base.EvaluationError(
            f"expected 440 GLM task/baseline templates, found {len(templates)}"
        )
    return templates


def verified_shared_evidence(
    *,
    glm_state: Path,
    shared_hashes: set[str],
) -> list[dict[str, Any]]:
    frozen_path = glm_state / "frozen_input_audit.json"
    frozen = base.read_json(frozen_path) if frozen_path.is_file() else {}
    if (
        not frozen_path.is_file()
        or not bool(frozen.get("complete"))
        or frozen.get("manifest_sha256") != EXPECTED_GLM_MANIFEST_SHA256
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
        audit_path = Path(str(item["audit_path"]))
        text_path = Path(str(item["text_path"]))
        if not audit_path.is_file() or not text_path.is_file():
            raise base.EvaluationError(f"missing verified shared evidence for {source_hash}")
        audit = base.read_json(audit_path)
        if (
            base.sha256_file(text_path) != audit.get("text_sha256")
            or item.get("parser_errors")
            or item.get("media_pending")
        ):
            raise base.EvaluationError(f"invalid verified shared evidence for {source_hash}")
        visual_raw = item.get("visual_path")
        if visual_raw:
            visual = Path(str(visual_raw))
            if not visual.is_file() or base.sha256_file(visual) != audit.get("visual_sha256"):
                raise base.EvaluationError(f"invalid shared visual evidence for {source_hash}")
        item["inherited_from_state"] = str(glm_state.resolve())
        item["inherited_by_content_hash"] = True
        selected.append(item)
    return selected


def build_manifest(
    state: Path,
    *,
    baselines: Path,
    glm_state: Path,
) -> dict[str, Any]:
    output_path = state / "qwen_source_outputs.json"
    if not output_path.is_file():
        raise base.EvaluationError("run `stage-source` before building the Qwen manifest")
    outputs = base.read_json(output_path)
    if not isinstance(outputs, list) or len(outputs) != 880:
        raise base.EvaluationError(f"expected 880 staged Qwen outputs in {output_path}")
    crosswalk = frozen_crosswalk(glm_state)
    crosswalk_by_agent = {str(row["agent_task_id"]): row for row in crosswalk}
    templates = glm_templates(glm_state)
    baseline_json = baselines / "final_1-220_samples.json"
    if not baseline_json.is_file():
        raise base.EvaluationError(f"missing baseline snapshot: {baseline_json}")
    if base.sha256_file(baseline_json) != EXPECTED_BASELINE_SHA256:
        raise base.EvaluationError(
            f"baseline snapshot SHA-256 differs from GLM: {baseline_json}"
        )
    baseline_rows = base.read_json(baseline_json)
    baseline_by_id = {str(row["task_id"]): row for row in baseline_rows}
    if len(baseline_by_id) != 220:
        raise base.EvaluationError("baseline snapshot must contain 220 unique tasks")

    roots = [state / "source", baselines]
    file_hash_cache: dict[str, dict[str, Any]] = {}

    def record(path: Path) -> dict[str, Any]:
        key = str(path.resolve())
        if key not in file_hash_cache:
            file_hash_cache[key] = base.source_file_record(path, roots=roots)
        return file_hash_cache[key]

    # Metadata is part of the frozen source inventory even though it is not
    # evidence shown to the judge.
    for output in outputs:
        record(Path(str(output["metadata_path"])))
    record(baseline_json)
    record(glm_state / "crosswalk.json")

    comparisons: list[dict[str, Any]] = []
    for output in outputs:
        task_id = str(output["task_id"])
        cross = crosswalk_by_agent.get(task_id)
        if not cross:
            raise base.EvaluationError(f"staged Qwen task absent from crosswalk: {task_id}")
        baseline_task_id = str(cross["baseline_task_id"])
        baseline_task = baseline_by_id.get(baseline_task_id)
        if not baseline_task:
            raise base.EvaluationError(
                f"crosswalk baseline task absent from snapshot: {baseline_task_id}"
            )
        agent_records = [
            record(Path(str(item["path"])))
            for item in output.get("deliverable_files") or []
        ]
        for baseline in base.BASELINES:
            template = templates[(task_id, baseline)]
            baseline_dir = baselines / "deliverables" / baseline_task_id / (
                "human" if baseline == "human" else "human+ai"
            )
            baseline_records = [record(path) for path in base.files_under(baseline_dir)]
            if not baseline_records:
                raise base.EvaluationError(f"missing {baseline} baseline deliverable: {baseline_dir}")
            if base.bundle_source_hash(baseline_records) != base.bundle_source_hash(
                template["baseline_files"]
            ):
                raise base.EvaluationError(
                    f"{task_id}/{baseline}: local baseline bundle differs from frozen GLM content"
                )
            repeat = int(output["repeat"])
            position = base.agent_position(task_id, CONDITION, baseline, repeat)
            comparison_id = (
                f"{STUDY_ID}__{CONDITION}__{baseline}__{task_id}__rep{repeat:02d}"
            )
            comparisons.append(
                {
                    "schema_version": base.SCHEMA_VERSION,
                    "study_id": STUDY_ID,
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
                    "expected_reference_names": template["expected_reference_names"],
                    "reference_files": template["reference_files"],
                    "reference_linkage": template["reference_linkage"],
                    "agent_position": position,
                    "candidate_A_source": "agent" if position == "A" else baseline,
                    "candidate_B_source": baseline if position == "A" else "agent",
                    "agent_files": agent_records,
                    "baseline_files": baseline_records,
                    "agent_quality_status": output["candidate_quality_status"],
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
    state.mkdir(parents=True, exist_ok=True)
    base.write_jsonl(state / "comparison_manifest.unblinded.jsonl", comparisons)
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
    shared_evidence = verified_shared_evidence(
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
            "shared_bundles_inherited_by_content_hash": len(shared_evidence),
            "parser_error_bundles": 0,
            "media_pending_items": 0,
        },
    )
    summary = {
        "schema_version": base.SCHEMA_VERSION,
        "dataset": "GDPval Qwen3.6-27B",
        "study_id": STUDY_ID,
        "created_at": base.utc_now(),
        "comparisons": len(comparisons),
        "tasks": 220,
        "repeat_count": 4,
        "active_conditions": [CONDITION],
        "conditions": {CONDITION: len(comparisons)},
        "baselines": {
            baseline: sum(row["baseline"] == baseline for row in comparisons)
            for baseline in base.BASELINES
        },
        "position_counts": dict(Counter(row["agent_position"] for row in comparisons)),
        "agent_outputs": len(
            {(row["task_id"], int(row["repeat"])) for row in comparisons}
        ),
        "agent_quality_status": dict(
            Counter(row["agent_quality_status"] for row in comparisons)
        ),
        "hf_repo_id": HF_REPO_ID,
        "hf_revision": HF_REVISION,
        "baseline_snapshot_sha256": base.sha256_file(baseline_json),
        "glm_crosswalk_sha256": base.sha256_file(glm_state / "crosswalk.json"),
        "shared_evidence_source_state": str(glm_state.resolve()),
        "shared_evidence_bundles": len(shared_evidence),
        "manifest_sha256": base.sha256_file(
            state / "comparison_manifest.unblinded.jsonl"
        ),
        "provenance_markers": {
            "Qwen model family": "Qwen",
            "Qwen model name": "Qwen3.6-27B",
            "Qwen HF repository": HF_REPO_ID,
            "Qwen HF revision": HF_REVISION,
            "Qwen harness run path": HF_RUN_REL.as_posix(),
            **{
                f"Qwen run {index:02d}": run_name
                for index, run_name in enumerate(
                    sorted({str(output["run_name"]) for output in outputs}),
                    start=1,
                )
            },
            "Qwen local source staging path": str((state / "source").resolve()),
            "Qwen local state path": str(state.resolve()),
        },
    }
    base.write_json(state / "manifest_summary.json", summary)
    return summary


def agent_bundle_hashes(state: Path) -> set[str]:
    return {
        base.bundle_source_hash(row["agent_files"])
        for row in base.iter_jsonl(state / "comparison_manifest.unblinded.jsonl")
    }


def prepare_qwen_evidence(
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
    result["qwen_candidate_hashes_selected"] = len(selected)
    result["shared_bundles_inherited_by_content_hash"] = sum(
        bool(item.get("inherited_by_content_hash"))
        for item in base.read_json(state / "evidence_index.json")
    )
    base.write_json(state / "evidence_summary.json", result)
    return result


def prepare_qwen_evidence_pilot(state: Path, *, workers: int) -> dict[str, Any]:
    selection_path = state / "evidence_pilot_selection.json"
    selection = (
        base.read_json(selection_path)
        if selection_path.is_file()
        else base.select_evidence_pilot(state)
    )
    hashes = {
        str(item["source_hash"])
        for item in selection["bundles"]
    } & agent_bundle_hashes(state)
    return prepare_qwen_evidence(
        state,
        only_hashes=hashes,
        workers=workers,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
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
    cap.add_argument("--max-pages", type=int, default=base.MAX_VISUAL_PAGES_PER_BUNDLE)
    media = sub.add_parser("parse-media")
    media.add_argument("--model", default=base.DEFAULT_AUDIO_MODEL)
    media.add_argument("--limit", type=int)
    google_media = sub.add_parser("parse-media-google")
    google_media.add_argument("--model", default=GOOGLE_MEDIA_PARSER_MODEL)
    google_media.add_argument("--limit", type=int)
    audit = sub.add_parser("audit-inputs")
    audit.add_argument("--skip-source-rehash", action="store_true")
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--bootstrap-samples", type=int, default=10_000)
    aggregate.add_argument("--seed", type=int, default=20260723)
    sub.add_parser("validate")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    state = args.state.resolve()
    baselines = args.baselines.resolve()
    glm_state = args.glm_state.resolve()
    try:
        if args.command == "stage-source":
            result = stage_source(state, glm_state=glm_state)
        elif args.command == "manifest":
            result = build_manifest(
                state,
                baselines=baselines,
                glm_state=glm_state,
            )
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
            result = prepare_qwen_evidence_pilot(state, workers=args.workers)
        elif args.command == "prepare-evidence":
            result = prepare_qwen_evidence(
                state,
                only_hashes=set(args.hash) or None,
                workers=args.workers,
            )
        elif args.command == "cap-visuals":
            result = base.cap_existing_visuals(state, max_pages=args.max_pages)
        elif args.command == "parse-media":
            result = base.parse_pending_media(
                state,
                model=args.model,
                limit=args.limit,
            )
        elif args.command == "parse-media-google":
            result = parse_pending_media_google(
                state,
                model=args.model,
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
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except base.EvaluationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
