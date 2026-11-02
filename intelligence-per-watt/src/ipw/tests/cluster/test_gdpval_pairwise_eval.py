"""Unit tests for the frozen GDPval pairwise evaluation protocol."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[5] / "cluster" / "gdpval_pairwise_eval.py"
    spec = importlib.util.spec_from_file_location("gdpval_pairwise_eval", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _file(name: str, digest: str) -> dict[str, object]:
    return {"name": name, "sha256": digest * 64, "size_bytes": 1, "path": f"/source/{name}", "suffix": Path(name).suffix}


def test_order_is_exactly_two_a_two_b_in_every_stratum() -> None:
    module = _load_module()
    for task in ("task-1", "task-2", "task-3"):
        for condition in module.CONDITIONS:
            for baseline in module.BASELINES:
                positions = [module.agent_position(task, condition, baseline, repeat) for repeat in range(4)]
                assert positions.count("A") == 2
                assert positions.count("B") == 2


def test_reference_resolver_excludes_run_created_helpers_and_handles_alias(tmp_path: Path) -> None:
    module = _load_module()
    canonical = tmp_path / "Case.pdf"
    helper = tmp_path / "Case.pdf.txt"
    alias = tmp_path / "Field Investigator's notes.docx"
    for path in (canonical, helper, alias):
        path.write_text("fixture", encoding="utf-8")

    selected, linkage = module.resolve_reference_paths(
        [canonical, helper, alias], ["Case.pdf", "Field Investigator"]
    )

    assert selected == [canonical, alias]
    assert [item["match_type"] for item in linkage] == ["exact_filename", "verified_alias_candidate"]


def test_manifest_validator_enforces_full_balanced_design() -> None:
    module = _load_module()
    rows = []
    for task_index in range(220):
        task = f"task-{task_index:03d}"
        reference = [_file("reference.pdf", f"{task_index % 10:x}")]
        for condition in module.CONDITIONS:
            for baseline in module.BASELINES:
                baseline_file = [_file(f"{baseline}.pdf", f"{(task_index + (baseline == 'human_ai')) % 10:x}")]
                for repeat in range(4):
                    rows.append(
                        {
                            "comparison_id": f"{condition}-{baseline}-{task}-{repeat}",
                            "task_id": task,
                            "condition": condition,
                            "baseline": baseline,
                            "repeat": repeat,
                            "agent_position": module.agent_position(task, condition, baseline, repeat),
                            "rubric": [{"index": 1, "criterion": "Correct", "points": 1}],
                            "expected_reference_names": ["reference.pdf"],
                            "reference_linkage": [{"expected_name": "reference.pdf", "actual_name": "reference.pdf"}],
                            "reference_files": reference,
                            "baseline_files": baseline_file,
                        }
                    )

    module.validate_manifest(rows, conditions=module.CONDITIONS)


def test_manifest_validator_supports_frozen_single_condition_design(tmp_path: Path) -> None:
    module = _load_module()
    rows = []
    for task_index in range(6):
        task = f"task-{task_index:03d}"
        reference = [_file("reference.pdf", f"{task_index % 10:x}")]
        for baseline in module.BASELINES:
            baseline_file = [
                _file(
                    f"{baseline}.pdf",
                    f"{(task_index + (baseline == 'human_ai')) % 10:x}",
                )
            ]
            for repeat in range(4):
                rows.append(
                    {
                        "comparison_id": f"qwen-{baseline}-{task}-{repeat}",
                        "task_id": task,
                        "condition": "without_rubric",
                        "baseline": baseline,
                        "repeat": repeat,
                        "agent_position": module.agent_position(
                            task, "without_rubric", baseline, repeat
                        ),
                        "rubric": [{"index": 1, "criterion": "Correct", "points": 1}],
                        "expected_reference_names": ["reference.pdf"],
                        "reference_linkage": [
                            {
                                "expected_name": "reference.pdf",
                                "actual_name": "reference.pdf",
                            }
                        ],
                        "reference_files": reference,
                        "baseline_files": baseline_file,
                    }
                )

    module.validate_manifest(
        rows,
        expected_tasks=6,
        repeat_count=4,
    )
    (tmp_path / "manifest_summary.json").write_text(
        json.dumps(
            {
                "comparisons": 48,
                "tasks": 6,
                "repeat_count": 4,
                "active_conditions": ["without_rubric"],
            }
        ),
        encoding="utf-8",
    )
    assert module.active_conditions_for_state(tmp_path) == ("without_rubric",)
    assert module.expected_comparison_count(tmp_path) == 48


def test_default_state_and_cli_require_explicit_prompt_opt_in(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    assert module.active_conditions_for_state(tmp_path) == ("without_rubric",)
    assert module.expected_comparison_count(tmp_path) == 1760
    module.write_json(tmp_path / "manifest_summary.json", {
        "conditions": {"with_rubric": 1760, "without_rubric": 1760},
    })
    assert module.expected_comparison_count(tmp_path) == 3520
    selected = []

    def capture_manifest(volume, baselines, state, *, include_rubric=False):
        selected.append(include_rubric)
        return {"prepared": True}

    monkeypatch.setattr(module, "build_manifest", capture_manifest)
    assert module.main(["--state", str(tmp_path), "manifest"]) == 0
    assert module.main(["--state", str(tmp_path), "manifest", "--include-rubric"]) == 0
    assert selected == [False, True]


def test_reference_alias_validation_covers_selected_repeats(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "Budget Notes.txt"
    source.write_text("Budget Notes and task reference details. " * 30, encoding="utf-8")
    rows = [{
        "task_id": "fabricated-task", "condition": "without_rubric", "repeat": repeat,
        "reference_linkage": [{"match_type": "verified_alias_candidate", "expected_name": "Budget Notes"}],
        "reference_files": [{"name": source.name, "path": str(source), "sha256": module.sha256_file(source)}],
    } for repeat in range(4)]
    crosswalk = [{"agent_task_id": "fabricated-task", "baseline_task_id": "fabricated-reference",
                  "reference_names_exact": False, "prompt_word_jaccard": 1.0}]
    audit = module.verify_reference_alias_content(rows, crosswalk)
    assert audit[0]["verified_across_task_condition_repeats"] == 4
    rows[-1]["reference_linkage"][0]["match_type"] = "exact_filename"
    with pytest.raises(module.EvaluationError, match="every selected repeat"):
        module.verify_reference_alias_content(rows, crosswalk)


def test_standard_aggregation_reports_metrics_and_evaluation_method(tmp_path: Path) -> None:
    module = _load_module()
    module.write_json(tmp_path / "manifest_summary.json", {
        "tasks": 1, "comparisons": 8, "repeat_count": 4,
        "active_conditions": ["without_rubric"],
    })
    rows = [{
        "comparison_id": f"fabricated-{baseline}-{repeat}", "task_id": "fabricated-task",
        "condition": "without_rubric", "baseline": baseline, "repeat": repeat,
        "occupation": "Fabricated occupation", "sector": "Fabricated sector",
        "agent_position": module.agent_position("fabricated-task", "without_rubric", baseline, repeat),
        "agent_outcome": "win", "agent_score": 1.0,
    } for baseline in module.BASELINES for repeat in range(4)]
    module.write_jsonl(tmp_path / "judgments.raw.unblinded.jsonl", rows)
    result = module.aggregate_results(tmp_path, bootstrap_samples=10)
    assert result["valid_verdicts"] == 8
    assert len(result["headline"]) == 2
    assert all(values["wins"] == 4 for values in result["headline"].values())
    assert result["condition_differences"] == []
    assert result["evaluation_method"]["judge_type"] == "model"
    assert "limitations" not in result


def test_corrupt_media_decoder_errors_are_narrowly_classified() -> None:
    module = _load_module()

    assert module.is_corrupt_media_error("EOFError: ")
    assert module.is_corrupt_media_error(
        "Failed to find two consecutive MPEG audio frames"
    )
    assert module.is_corrupt_media_error(
        "Error opening input: Invalid data found when processing input"
    )
    assert module.is_corrupt_media_error(
        "Output file does not contain any stream"
    )
    assert not module.is_corrupt_media_error("ffmpeg executable is unavailable")


def test_audio_prose_preserves_content_and_attempt_provenance(tmp_path: Path) -> None:
    module = _load_module()
    audio = tmp_path / "fabricated.wav"
    audio.write_bytes(b"fabricated audio fixture")

    class AudioAPI:
        calls = 0

        def json(self, method, endpoint, payload):
            self.calls += 1
            return {
                "id": f"fabricated-response-{self.calls}", "model": "fabricated-model",
                "choices": [{"message": {"content": "A fabricated spoken greeting."}}],
                "usage": {"total_tokens": 10},
            }

    api = AudioAPI()
    result = module.parse_audio_chunk(api, audio, model="fabricated-model")
    assert api.calls == 3
    assert result["transcript"] == ""
    assert result["nonverbal_description"] == "A fabricated spoken greeting."
    assert result["processing_notes"]
    assert result["api_metadata"]["unstructured_fallback"] is True
    assert len(result["api_metadata"]["failed_attempts"]) == 3
    assert result["api_metadata"]["usage"] == {"total_tokens": 10}


def test_audio_transcription_preserves_method_and_original_error(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    def parser_error(*args, **kwargs):
        raise module.EvaluationError("fabricated parser error")

    class AudioAPI:
        def transcribe(self, path):
            return {"text": "Fabricated greeting.", "language": "en", "duration": 1}

    monkeypatch.setattr(module, "parse_audio_chunk", parser_error)
    result, = module.parse_audio_chunk_adaptive(
        AudioAPI(), tmp_path / "fabricated.wav", model="fabricated-model",
        work_dir=tmp_path, fallback_seconds=(),
    )
    assert result["transcript"] == "Fabricated greeting."
    assert result["nonverbal_description"] == ""
    assert result["language"] == "en"
    assert result["processing_notes"]
    assert result["api_metadata"]["primary_parser_error"] == "fabricated parser error"
    assert result["api_metadata"]["transcription_fallback"] is True
    assert result["api_metadata"]["fallback_response_metadata"] == {"language": "en", "duration": 1}


def test_ties_are_half_credit_and_reported_separately() -> None:
    module = _load_module()
    rows = [
        {"agent_outcome": "win"},
        {"agent_outcome": "tie"},
        {"agent_outcome": "loss"},
        {"agent_outcome": "tie"},
    ]

    metrics = module.outcome_metrics(rows)

    assert metrics["tie_adjusted_preference"] == 0.5
    assert metrics["strict_win_rate"] == 0.25
    assert metrics["tie_rate"] == 0.5
    assert metrics["loss_rate"] == 0.25
    assert metrics["non_tie_win_rate"] == 0.5
    assert metrics["win_or_tie_rate"] == 0.75


def test_leak_scan_catches_provenance_but_not_anonymous_labels() -> None:
    module = _load_module()
    clean = {"body": {"input": "Candidate A and Candidate B"}}
    leaked = {"body": {"input": "/Volumes/GDPvalGLMFullRuns/g52full_secret"}}

    assert module.leak_scan(clean) == []
    assert module.leak_scan(leaked)


def test_judge_model_and_batch_pricing_are_pinned() -> None:
    module = _load_module()

    assert module.DEFAULT_MODEL == "gpt-5-mini-2025-08-07"
    assert module.DEFAULT_MAX_OUTPUT_TOKENS == 12_000
    usage = module.usage_accounting(
        [
            {
                "api_metadata": {
                    "usage": {
                        "input_tokens": 1_000_000,
                        "input_tokens_details": {"cached_tokens": 200_000},
                        "output_tokens": 100_000,
                    }
                }
            }
        ]
    )
    assert usage["pricing_per_million_usd"] == {
        "input": 0.25,
        "cached_input": 0.025,
        "output": 2.0,
    }
    assert usage["estimated_batch_cost_usd"] == 0.405
