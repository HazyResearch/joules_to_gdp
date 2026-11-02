from __future__ import annotations

import pytest

from ipw.cli.judge_models import resolve_judge_options


@pytest.mark.parametrize(
    ("preset", "client", "model"),
    [
        ("gpt-mini", "openai", "gpt-5.4-mini"),
        ("gpt_mini", "openai", "gpt-5.4-mini"),
        ("gpt-nano", "openai", "gpt-5.4-nano"),
        ("gpt_nano", "openai", "gpt-5.4-nano"),
        ("gemini-flash", "gemini", "gemini-3.5-flash"),
        ("gemini_flash", "gemini", "gemini-3.5-flash"),
        ("haiku", "anthropic", "claude-haiku-4-5"),
        ("sonnet", "anthropic", "claude-sonnet-4-6"),
    ],
)
def test_judge_presets_resolve_latest_models(preset: str, client: str, model: str) -> None:
    config = resolve_judge_options(
        judge_model=preset,
        eval_client=None,
        eval_base_url=None,
        eval_model=None,
    )

    assert config.client == client
    assert config.model == model
    assert config.preset is not None


def test_no_flags_default_to_gpt_mini() -> None:
    config = resolve_judge_options(
        judge_model=None,
        eval_client=None,
        eval_base_url=None,
        eval_model=None,
    )

    assert config.preset == "gpt-mini"
    assert config.client == "openai"
    assert config.model == "gpt-5.4-mini"


def test_raw_legacy_eval_model_is_preserved() -> None:
    config = resolve_judge_options(
        judge_model=None,
        eval_client=None,
        eval_base_url=None,
        eval_model="gpt-5-nano-2025-08-07",
        explicit_eval_model=True,
    )

    assert config.preset is None
    assert config.client == "openai"
    assert config.model == "gpt-5-nano-2025-08-07"


def test_conflicting_preset_and_raw_eval_model_fails() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        resolve_judge_options(
            judge_model="sonnet",
            eval_client=None,
            eval_base_url=None,
            eval_model="gpt-5.4-mini",
            explicit_eval_model=True,
        )

