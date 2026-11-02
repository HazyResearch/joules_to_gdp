"""Shared judge model preset resolution for CLI commands."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_JUDGE_PRESET = "gpt-mini"
DEFAULT_JUDGE_CLIENT = "openai"
DEFAULT_JUDGE_BASE_URL = "https://api.openai.com/v1"
DEFAULT_JUDGE_MODEL = "gpt-5.4-mini"
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


@dataclass(frozen=True)
class JudgePreset:
    name: str
    client: str
    model: str
    base_url: str | None = None


@dataclass(frozen=True)
class JudgeConfig:
    client: str
    base_url: str | None
    model: str
    preset: str | None = None


_PRESETS: dict[str, JudgePreset] = {
    "gpt-mini": JudgePreset(
        name="gpt-mini",
        client="openai",
        base_url=DEFAULT_JUDGE_BASE_URL,
        model="gpt-5.4-mini",
    ),
    "gpt-nano": JudgePreset(
        name="gpt-nano",
        client="openai",
        base_url=DEFAULT_JUDGE_BASE_URL,
        model="gpt-5.4-nano",
    ),
    "gemini-flash": JudgePreset(
        name="gemini-flash",
        client="gemini",
        base_url=GEMINI_BASE_URL,
        model="gemini-3.5-flash",
    ),
    "haiku": JudgePreset(
        name="haiku",
        client="anthropic",
        base_url=ANTHROPIC_BASE_URL,
        model="claude-haiku-4-5",
    ),
    "sonnet": JudgePreset(
        name="sonnet",
        client="anthropic",
        base_url=ANTHROPIC_BASE_URL,
        model="claude-sonnet-4-6",
    ),
}

_ALIASES = {
    "gpt_mini": "gpt-mini",
    "gptmini": "gpt-mini",
    "gpt-nano": "gpt-nano",
    "gpt_nano": "gpt-nano",
    "gptnano": "gpt-nano",
    "gemini_flash": "gemini-flash",
    "geminiflash": "gemini-flash",
}


def available_judge_presets() -> tuple[str, ...]:
    return tuple(_PRESETS)


def normalize_judge_preset(name: str) -> str:
    key = name.strip().lower()
    canonical = _ALIASES.get(key, key.replace("_", "-"))
    if canonical not in _PRESETS:
        available = ", ".join(available_judge_presets())
        raise KeyError(f"Unknown judge preset '{name}'. Available: {available}")
    return canonical


def resolve_judge_options(
    *,
    judge_model: str | None,
    eval_client: str | None,
    eval_base_url: str | None,
    eval_model: str | None,
    explicit_eval_client: bool = False,
    explicit_eval_base_url: bool = False,
    explicit_eval_model: bool = False,
) -> JudgeConfig:
    """Resolve preset and raw eval flags into concrete judge settings."""
    preset_name = judge_model.strip() if judge_model else None

    if preset_name:
        canonical = normalize_judge_preset(preset_name)
        preset = _PRESETS[canonical]
        _validate_matching_raw_flags(
            preset=preset,
            eval_client=eval_client,
            eval_base_url=eval_base_url,
            eval_model=eval_model,
            explicit_eval_client=explicit_eval_client,
            explicit_eval_base_url=explicit_eval_base_url,
            explicit_eval_model=explicit_eval_model,
        )
        return JudgeConfig(
            client=preset.client,
            base_url=preset.base_url,
            model=preset.model,
            preset=preset.name,
        )

    if explicit_eval_client or explicit_eval_base_url or explicit_eval_model:
        client = (eval_client or DEFAULT_JUDGE_CLIENT).strip() or DEFAULT_JUDGE_CLIENT
        return JudgeConfig(
            client=client,
            base_url=eval_base_url or _default_base_url_for_client(client),
            model=(eval_model or DEFAULT_JUDGE_MODEL).strip() or DEFAULT_JUDGE_MODEL,
            preset=None,
        )

    default_preset = _PRESETS[DEFAULT_JUDGE_PRESET]
    return JudgeConfig(
        client=default_preset.client,
        base_url=default_preset.base_url,
        model=default_preset.model,
        preset=default_preset.name,
    )


def _validate_matching_raw_flags(
    *,
    preset: JudgePreset,
    eval_client: str | None,
    eval_base_url: str | None,
    eval_model: str | None,
    explicit_eval_client: bool,
    explicit_eval_base_url: bool,
    explicit_eval_model: bool,
) -> None:
    conflicts: list[str] = []
    if explicit_eval_client and (eval_client or "").strip() != preset.client:
        conflicts.append(f"--eval-client must be '{preset.client}'")
    if explicit_eval_model and (eval_model or "").strip() != preset.model:
        conflicts.append(f"--eval-model must be '{preset.model}'")
    if explicit_eval_base_url:
        provided_base = (eval_base_url or "").rstrip("/")
        expected_base = (preset.base_url or "").rstrip("/")
        if provided_base != expected_base:
            conflicts.append(
                "--eval-base-url must be "
                + (f"'{preset.base_url}'" if preset.base_url else "omitted")
            )
    if conflicts:
        detail = "; ".join(conflicts)
        raise ValueError(
            f"--judge-model {preset.name} conflicts with raw eval flags: {detail}."
        )


def _default_base_url_for_client(client: str) -> str | None:
    if client == "openai":
        return DEFAULT_JUDGE_BASE_URL
    if client == "anthropic":
        return ANTHROPIC_BASE_URL
    if client == "gemini":
        return GEMINI_BASE_URL
    return None
