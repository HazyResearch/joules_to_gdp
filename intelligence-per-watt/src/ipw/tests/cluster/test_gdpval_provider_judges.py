"""Unit tests for Gemini pilot gating and usage accounting."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


def _load_module():
    cluster = Path(__file__).resolve().parents[5] / "cluster"
    sys.path.insert(0, str(cluster))
    try:
        sys.modules.pop("gdpval_provider_judges", None)
        return importlib.import_module("gdpval_provider_judges")
    finally:
        sys.path.remove(str(cluster))


def test_full_submission_gate_requires_24_valid_pilot_verdicts(tmp_path: Path) -> None:
    module = _load_module()
    comparison_ids = [f"qwen-comparison-{index:02d}" for index in range(24)]
    module.write_json(
        tmp_path / "pilot_selection.json",
        {"comparison_ids": comparison_ids, "count": 24},
    )
    judgments = [
        {
            "comparison_id": comparison_id,
            "blind_id": module.base.blind_id(comparison_id),
        }
        for comparison_id in comparison_ids
    ]
    module.write_jsonl(
        tmp_path / "judgments.raw.unblinded.jsonl",
        judgments[:-1],
    )
    module.write_jsonl(tmp_path / "judgment_failures.jsonl", [])

    with pytest.raises(module.base.EvaluationError, match="24/24"):
        module.require_valid_pilot(tmp_path)

    module.write_jsonl(
        tmp_path / "judgments.raw.unblinded.jsonl",
        judgments,
    )
    gate = module.require_valid_pilot(tmp_path)
    assert gate["passed"]
    assert gate["valid_pilot_verdicts"] == 24


def test_google_usage_accounting_preserves_provider_token_fields() -> None:
    module = _load_module()
    rows = [
        {
            "api_metadata": {
                "usage": {
                    "promptTokenCount": 100,
                    "cachedContentTokenCount": 20,
                    "candidatesTokenCount": 30,
                    "thoughtsTokenCount": 7,
                    "totalTokenCount": 137,
                    "promptTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 80},
                        {"modality": "DOCUMENT", "tokenCount": 20},
                    ],
                }
            }
        },
        {
            "api_metadata": {
                "usage": {
                    "prompt_token_count": 50,
                    "candidates_token_count": 10,
                    "total_token_count": 60,
                }
            }
        },
    ]

    usage = module.google_usage_accounting(rows)

    assert usage["provider"] == "google"
    assert usage["model"] == "models/gemini-3.1-flash-lite"
    assert usage["requests_with_usage_metadata"] == 2
    assert usage["prompt_tokens"] == 150
    assert usage["cached_content_tokens"] == 20
    assert usage["candidate_tokens"] == 40
    assert usage["thought_tokens"] == 7
    assert usage["total_tokens"] == 197
    assert usage["modality_token_counts"]["prompt"] == {
        "DOCUMENT": 20,
        "TEXT": 80,
    }
    assert "estimated_batch_cost_usd" not in usage


def test_gemini_settings_are_frozen() -> None:
    module = _load_module()

    assert module.GOOGLE_MODEL == "models/gemini-3.1-flash-lite"
    assert module.MAX_OUTPUT_TOKENS == 24_000


def test_provider_state_copies_exact_secret_audit(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "source"
    state = tmp_path / "provider"
    source.mkdir()
    module.write_json(
        source / "secret_audit.json",
        {"passed": True, "hits": []},
    )

    module.initialize_state("google", source=source, state=state)

    assert module.read_json(state / "secret_audit.json")["passed"] is True
