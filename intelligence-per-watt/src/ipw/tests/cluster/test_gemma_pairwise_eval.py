"""Unit tests for the pinned Gemma Hugging Face source adapter."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module():
    cluster = Path(__file__).resolve().parents[5] / "cluster"
    sys.path.insert(0, str(cluster))
    try:
        sys.modules.pop("gemma_pairwise_eval", None)
        return importlib.import_module("gemma_pairwise_eval")
    finally:
        sys.path.remove(str(cluster))


def test_pinned_source_shards_cover_exactly_880_outputs() -> None:
    module = _load_module()

    assert module.HF_REPO_ID == (
        "ScalingIntelligence/gemma4-31b-gdpval-occ-repeat4-202607"
    )
    assert module.HF_REVISION == (
        "716ea26d5ecc5991e42e82d10424e6ddc286ccfc"
    )
    assert sum(shard.count for shard in module.SOURCE_SHARDS) == 880
    assert {
        shard.source_task_start for shard in module.SOURCE_SHARDS
    } == {0, 115}
    assert [
        shard.count // 4 for shard in module.SOURCE_SHARDS
    ] == [115, 105]
    assert [shard.first_query for shard in module.SOURCE_SHARDS] == [0, 0]
    assert [shard.chunk_path for shard in module.SOURCE_SHARDS] == [
        "no-rubric/chunks/rows-000-459",
        "no-rubric/chunks/rows-460-879",
    ]


def test_artifact_selection_uses_only_the_declared_query_range() -> None:
    module = _load_module()
    shard = module.SourceShard("run", "chunk", 0, 4, 10)
    entries = [
        SimpleNamespace(path=f"{shard.artifact_root}/q{index:04d}_task")
        for index in range(6)
    ]

    selected = module.selected_artifact_paths(entries, shard)

    assert sorted(selected) == [0, 1, 2, 3]
    assert all("q0004" not in path and "q0005" not in path for path in selected.values())


def test_declared_deliverable_resolution_excludes_dependency_trees() -> None:
    module = _load_module()
    artifact = "runs/run/artifacts/q0000_task"
    metadata = {
        "deliverable_validation": {
            "primary_files": ["/workspace/submission"]
        }
    }
    files = [
        f"{artifact}/workspace/submission/report.pdf",
        f"{artifact}/workspace/submission/src/app.py",
        f"{artifact}/workspace/submission/node_modules/pkg/index.js",
    ]

    resolved, decisions = module.resolve_declared_remote_files(
        metadata,
        artifact_path=artifact,
        workspace_files=files,
    )

    assert resolved == [
        f"{artifact}/workspace/submission/report.pdf",
        f"{artifact}/workspace/submission/src/app.py",
    ]
    assert decisions[0]["match_type"] == "exact_path_or_directory"


def test_missing_declared_deliverable_is_archive_integrity_failure() -> None:
    module = _load_module()
    with pytest.raises(module.base.EvaluationError, match="archive-integrity failure"):
        module.resolve_declared_remote_files(
            {"deliverable_files": ["/workspace/missing.pdf"]},
            artifact_path="runs/run/artifacts/q0000_task",
            workspace_files=[],
        )


def test_explicit_empty_primary_files_preserves_missing_submission() -> None:
    module = _load_module()
    metadata = {
        "deliverable_validation": {
            "status": "invalid",
            "primary_files": [],
        },
        "deliverable_files": ["/workspace/expected_but_never_created.pdf"],
    }

    assert module.declared_deliverable_values(metadata) == []
    resolved, decisions = module.resolve_declared_remote_files(
        metadata,
        artifact_path="runs/run/artifacts/q0000_task",
        workspace_files=[],
    )
    assert resolved == []
    assert decisions == []
    assert module.candidate_quality_status(
        metadata,
        resolved_paths=[],
        remote_sizes={},
    ) == "missing"


def test_selected_metadata_requires_no_rubric_completed_and_crosswalked() -> None:
    module = _load_module()
    shard = module.SourceShard("run", "chunk", 0, 4, 12)
    metadata = {
        "task_id": "task-1",
        "repeat_index": 2,
        "source_task_index": 12,
        "include_rubric_in_prompt": False,
        "completed": True,
        "timed_out": False,
    }

    canonical = module.validate_selected_metadata(
        metadata,
        shard=shard,
        query_index=2,
        crosswalk_task_ids={"task-1"},
    )

    assert canonical == {
        "task_id": "task-1",
        "repeat": 2,
        "source_task_index": 12,
        "completed": True,
        "timed_out": False,
        "include_rubric_in_prompt": False,
    }
    metadata.pop("timed_out")
    with pytest.raises(module.base.EvaluationError, match="timed_out"):
        module.validate_selected_metadata(
            metadata,
            shard=shard,
            query_index=2,
            crosswalk_task_ids={"task-1"},
        )


def test_overlapping_query_names_stay_separate_by_shard() -> None:
    module = _load_module()
    first, second = module.SOURCE_SHARDS
    assert first.first_query == second.first_query == 0
    assert first.artifact_root != second.artifact_root
    assert first.artifact_root.startswith(first.chunk_path)
    assert second.artifact_root.startswith(second.chunk_path)


def test_invalid_deliverable_status_is_preserved_as_candidate_quality() -> None:
    module = _load_module()
    metadata = {
        "deliverable_validation": {
            "status": "invalid",
            "primary_files": ["/workspace/report.pdf"],
        }
    }
    artifact = "chunk/run/artifacts/q0000_task"
    resolved, _ = module.resolve_declared_remote_files(
        metadata,
        artifact_path=artifact,
        workspace_files=[f"{artifact}/workspace/report.pdf"],
    )
    assert resolved == [f"{artifact}/workspace/report.pdf"]
    assert module.candidate_quality_status(
        metadata,
        resolved_paths=resolved,
        remote_sizes={resolved[0]: 100},
    ) == "invalid_with_artifacts"


def test_collision_safe_local_paths_include_full_hf_path_hash(
    tmp_path: Path,
) -> None:
    module = _load_module()
    upper = module.deliverable_local_path(
        tmp_path,
        "chunk/artifacts/q0000/workspace/Report.PDF",
    )
    lower = module.deliverable_local_path(
        tmp_path,
        "chunk/artifacts/q0000/workspace/report.pdf",
    )
    assert upper != lower
    assert upper.name == "Report.PDF"
    assert lower.name == "report.pdf"
