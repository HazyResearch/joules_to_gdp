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

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_BASE_URL = "https://api.anthropic.com/v1"

load_dotenv()


def _usage_int(usage: dict[str, Any], key: str) -> int | None:
    value = usage.get(key)
    return int(value) if value is not None else None


def _data_url_source(url: str) -> dict[str, Any] | None:
    prefix, _, encoded = url.partition(",")
    if not encoded or ";base64" not in prefix:
        return None
    media_type = prefix.removeprefix("data:").split(";", 1)[0] or "image/png"
    try:
        base64.b64decode(encoded, validate=True)
    except Exception:
        return None
    return {
        "type": "base64",
        "media_type": media_type,
        "data": encoded,
    }


def _content_to_anthropic(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            parts.append({"type": "text", "text": str(part)})
            continue
        part_type = part.get("type")
        if part_type == "text":
            parts.append({"type": "text", "text": str(part.get("text", ""))})
        elif part_type == "image_url":
            image_url = part.get("image_url") or {}
            source = _data_url_source(str(image_url.get("url", "")))
            if source is not None:
                parts.append({"type": "image", "source": source})
    return parts or ""


def _messages_to_anthropic(messages: Sequence[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            else:
                system_parts.append(str(content))
            continue
        converted.append(
            {
                "role": "assistant" if role == "assistant" else "user",
                "content": _content_to_anthropic(content),
            }
        )
    return "\n".join(part for part in system_parts if part), converted


@ClientRegistry.register("anthropic")
class AnthropicClient(InferenceClient):
    """Anthropic Claude client for LLM-as-judge evaluation."""

    client_id, client_name = "anthropic", "Anthropic"

    def __init__(self, base_url: str | None = None, **config: Any) -> None:
        host = str(base_url or config.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        super().__init__(host, **config)
        self.model = str(config.get("model") or DEFAULT_MODEL).strip()
        self.timeout_seconds = float(config.get("timeout_seconds", 60.0))

    def stream_chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        raise RuntimeError(
            "AnthropicClient is evaluation-only; do not use it for profiling."
        )

    def list_models(self) -> Sequence[str]:
        raise RuntimeError("AnthropicClient is evaluation-only and cannot list models.")

    def health(self) -> bool:
        raise RuntimeError("AnthropicClient is evaluation-only and does not expose health checks.")

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
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "EMPTY":
            raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic judge models")

        messages = params.get("messages") or [{"role": "user", "content": prompt}]
        system, anthropic_messages = _messages_to_anthropic(messages)
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": anthropic_messages,
            "max_tokens": int(params.get("max_output_tokens") or params.get("max_tokens") or 4096),
        }
        if system:
            payload["system"] = system
        if params.get("temperature") is not None:
            payload["temperature"] = params["temperature"]

        try:
            response = requests.post(
                f"{self.base_url}/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                data=json.dumps(payload),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except req_exc.HTTPError as exc:
            detail = ""
            if exc.response is not None:
                detail = f" | body: {exc.response.text[:500]}"
            raise RuntimeError(f"AnthropicClient request failed: {exc}{detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"AnthropicClient request failed: {exc}") from exc

        data = response.json()
        content_parts = data.get("content") or []
        content = "\n".join(
            str(part.get("text", ""))
            for part in content_parts
            if isinstance(part, dict) and part.get("type") == "text"
        )
        usage = data.get("usage") or {}
        prompt_tokens = _usage_int(usage, "input_tokens")
        completion_tokens = _usage_int(usage, "output_tokens")
        total_tokens = (
            prompt_tokens + completion_tokens
            if prompt_tokens is not None and completion_tokens is not None
            else None
        )
        return Response(
            content=content,
            usage=ChatUsage(prompt_tokens, completion_tokens, total_tokens),
            time_to_first_token_ms=0.0,
        )


__all__ = ["AnthropicClient", "DEFAULT_MODEL"]

