from __future__ import annotations

import base64
import json
import os
from typing import Any, Optional, Sequence

import requests
from dotenv import load_dotenv
from requests import exceptions as req_exc

from ..core.registry import ClientRegistry
from ..core.types import ChatUsage, Response
from .base import InferenceClient

DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

load_dotenv()


def _usage_int(usage: dict[str, Any], key: str) -> int | None:
    value = usage.get(key)
    return int(value) if value is not None else None


def _inline_data_part(url: str) -> dict[str, Any] | None:
    prefix, _, encoded = url.partition(",")
    if not encoded or ";base64" not in prefix:
        return None
    mime_type = prefix.removeprefix("data:").split(";", 1)[0] or "image/png"
    try:
        base64.b64decode(encoded, validate=True)
    except Exception:
        return None
    return {"inline_data": {"mime_type": mime_type, "data": encoded}}


def _content_to_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return [{"text": str(content)}]

    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            parts.append({"text": str(part)})
            continue
        part_type = part.get("type")
        if part_type == "text":
            parts.append({"text": str(part.get("text", ""))})
        elif part_type == "image_url":
            image_url = part.get("image_url") or {}
            inline_part = _inline_data_part(str(image_url.get("url", "")))
            if inline_part is not None:
                parts.append(inline_part)
    return parts or [{"text": ""}]


def _messages_to_gemini(messages: Sequence[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            else:
                system_parts.append(str(content))
            continue
        contents.append(
            {
                "role": "model" if role == "assistant" else "user",
                "parts": _content_to_parts(content),
            }
        )
    return "\n".join(part for part in system_parts if part), contents


@ClientRegistry.register("gemini")
class GeminiClient(InferenceClient):
    """Google Gemini client for LLM-as-judge evaluation."""

    client_id, client_name = "gemini", "Gemini"

    def __init__(self, base_url: str | None = None, **config: Any) -> None:
        host = str(base_url or config.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        super().__init__(host, **config)
        self.model = str(config.get("model") or DEFAULT_MODEL).strip()
        self.timeout_seconds = float(config.get("timeout_seconds", 60.0))

    def stream_chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        raise RuntimeError(
            "GeminiClient is evaluation-only; do not use it for profiling."
        )

    def list_models(self) -> Sequence[str]:
        raise RuntimeError("GeminiClient is evaluation-only and cannot list models.")

    def health(self) -> bool:
        raise RuntimeError("GeminiClient is evaluation-only and does not expose health checks.")

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        response = self._chat_completion(
            self.model,
            "",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return response.content

    def _chat_completion(self, model: str, prompt: str, **params: Any) -> Response:
        api_key = os.getenv("IPW_EVAL_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "EMPTY":
            raise RuntimeError("IPW_EVAL_API_KEY, GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini judge models")

        messages = params.get("messages") or [{"role": "user", "content": prompt}]
        system_text, contents = _messages_to_gemini(messages)
        generation_config: dict[str, Any] = {}
        if params.get("temperature") is not None:
            generation_config["temperature"] = params["temperature"]
        max_tokens = params.get("max_output_tokens") or params.get("max_tokens")
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = int(max_tokens)

        payload: dict[str, Any] = {"contents": contents}
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}
        if generation_config:
            payload["generationConfig"] = generation_config

        request_model = model or self.model
        try:
            response = requests.post(
                f"{self.base_url}/models/{request_model}:generateContent",
                params={"key": api_key},
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except req_exc.HTTPError as exc:
            detail = ""
            if exc.response is not None:
                detail = f" | body: {exc.response.text[:500]}"
            raise RuntimeError(f"GeminiClient request failed: {exc}{detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"GeminiClient request failed: {exc}") from exc

        data = response.json()
        candidates = data.get("candidates") or []
        content = ""
        if candidates:
            parts = ((candidates[0].get("content") or {}).get("parts") or [])
            content = "\n".join(
                str(part.get("text", ""))
                for part in parts
                if isinstance(part, dict) and part.get("text") is not None
            )

        usage = data.get("usageMetadata") or {}
        prompt_tokens = _usage_int(usage, "promptTokenCount")
        completion_tokens = _usage_int(usage, "candidatesTokenCount")
        total_tokens = _usage_int(usage, "totalTokenCount")
        if (
            total_tokens is None
            and prompt_tokens is not None
            and completion_tokens is not None
        ):
            total_tokens = prompt_tokens + completion_tokens
        return Response(
            content=content,
            usage=ChatUsage(prompt_tokens, completion_tokens, total_tokens),
            time_to_first_token_ms=0.0,
        )


__all__ = ["GeminiClient", "DEFAULT_MODEL"]

