"""Unit tests for the pinned GPT-OSS Hugging Face source adapter."""

from __future__ import annotations

import importlib
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module():
    cluster = Path(__file__).resolve().parents[5] / "cluster"
    sys.path.insert(0, str(cluster))
    try:
        sys.modules.pop("gptoss_pairwise_eval", None)
        return importlib.import_module("gptoss_pairwise_eval")
    finally:
        sys.path.remove(str(cluster))


def test_pinned_models_use_separate_canonical_roots() -> None:
    module = _load_module()

    assert module.HF_REVISION == "4a277392cddacbef090be029d33f4f69372d487e"
    assert module.MODEL_CONFIGS["120b"].report_root.endswith("/reports/final")
    assert module.MODEL_CONFIGS["20b"].report_root.endswith(
        "/reports/final_corrected"
    )
    assert (
        module.MODEL_CONFIGS["120b"].study_id
        != module.MODEL_CONFIGS["20b"].study_id
    )


def test_artifact_selection_requires_q0000_through_q0879() -> None:
    module = _load_module()
    root = "run/artifacts"
    entries = [
        SimpleNamespace(path=f"{root}/q{index:04d}_task/metadata.json", size=1)
        for index in range(880)
    ]

    artifacts, files = module.selected_artifacts(
        entries,
        artifact_root=root,
    )

    assert len(artifacts) == 880
    assert sorted(artifacts) == list(range(880))
    assert len(files) == 880


def test_metadata_allows_auditable_failures_but_requires_four_run_layout() -> None:
    module = _load_module()
    metadata = {
        "task_id": "task-1",
        "repeat_index": 2,
        "source_task_index": 12,
        "include_rubric_in_prompt": False,
        "completed": False,
        "timed_out": True,
    }

    canonical = module.validate_metadata(
        metadata,
        query_index=50,
        crosswalk_task_ids={"task-1"},
    )

    assert canonical["completed"] is False
    assert canonical["timed_out"] is True
    metadata["repeat_index"] = 4
    with pytest.raises(module.base.EvaluationError, match="expected 0-3"):
        module.validate_metadata(
            metadata,
            query_index=51,
            crosswalk_task_ids={"task-1"},
        )


def test_missing_file_is_only_preserved_for_incomplete_attempt() -> None:
    module = _load_module()
    incomplete = {
        "deliverable_files": ["missing.pdf"],
        "completed": False,
    }

    files, decisions, missing = module.resolve_attempt_deliverables(
        incomplete,
        artifact_path="run/artifacts/q0000_task",
        workspace_files=[],
    )

    assert files == []
    assert decisions == []
    assert missing[0]["classification"] == (
        "genuine_missing_or_invalid_submission"
    )
    with pytest.raises(module.base.EvaluationError, match="archive-integrity"):
        module.resolve_attempt_deliverables(
            {**incomplete, "completed": True},
            artifact_path="run/artifacts/q0000_task",
            workspace_files=[],
        )


def test_invalid_document_classifier_is_narrow_and_structural(
    tmp_path: Path,
) -> None:
    module = _load_module()
    mislabeled_pdf = tmp_path / "submission.pdf"
    mislabeled_pdf.write_text("<!DOCTYPE html>", encoding="utf-8")
    invalid_docx = tmp_path / "submission.docx"
    invalid_docx.write_text("not an OOXML package", encoding="utf-8")
    valid_docx = tmp_path / "valid.docx"
    with zipfile.ZipFile(valid_docx, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", "<document/>")
    ordinary_text = tmp_path / "notes.txt"
    ordinary_text.write_text("plain text", encoding="utf-8")

    assert "PDF header" in module.base.invalid_submitted_document_reason(
        mislabeled_pdf
    )
    assert "valid Office package" in (
        module.base.invalid_submitted_document_reason(invalid_docx) or ""
    )
    assert module.base.invalid_submitted_document_reason(valid_docx) is None
    assert module.base.invalid_submitted_document_reason(ordinary_text) is None
    assert module.base.is_corrupt_media_error(
        "could not find codec parameters; Error opening input files: End of file"
    )
