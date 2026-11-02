"""Unit tests for the pinned Qwen Hugging Face source adapter."""

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
        sys.modules.pop("qwen_pairwise_eval", None)
        return importlib.import_module("qwen_pairwise_eval")
    finally:
        sys.path.remove(str(cluster))


def test_pinned_source_shards_cover_exactly_880_outputs() -> None:
    module = _load_module()

    assert module.HF_REVISION == "46f8f4acba0d6bec24beec5d726a0ade81d181c2"
    assert sum(shard.count for shard in module.SOURCE_SHARDS) == 880
    assert {shard.source_task_start for shard in module.SOURCE_SHARDS} == {0, 210, 215}
    assert [shard.count // 4 for shard in module.SOURCE_SHARDS] == [210, 5, 5]


def test_artifact_selection_uses_only_the_declared_query_range() -> None:
    module = _load_module()
    shard = module.SourceShard("run", 0, 4, 10)
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


def test_selected_metadata_requires_no_rubric_completed_and_crosswalked() -> None:
    module = _load_module()
    shard = module.SourceShard("run", 0, 4, 12)
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
