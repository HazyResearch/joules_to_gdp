from __future__ import annotations

import base64
from typing import Any
from unittest.mock import Mock


class _FakeResponse:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self.text = str(data)

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._data


def test_anthropic_chat_uses_messages_api(monkeypatch) -> None:
    from ipw.clients.anthropic import AnthropicClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    post = Mock(
        return_value=_FakeResponse(
            {
                "content": [{"type": "text", "text": "judge verdict"}],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            }
        )
    )
    monkeypatch.setattr("ipw.clients.anthropic.requests.post", post)

    client = AnthropicClient(model="claude-sonnet-4-6")

    assert client.chat("system", "user") == "judge verdict"
    payload = post.call_args.kwargs["data"]
    assert "claude-sonnet-4-6" in payload
    assert post.call_args.kwargs["headers"]["x-api-key"] == "test-key"


def test_gemini_chat_uses_generate_content(monkeypatch) -> None:
    from ipw.clients.gemini import GeminiClient

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    post = Mock(
        return_value=_FakeResponse(
            {
                "candidates": [{"content": {"parts": [{"text": "judge verdict"}]}}],
                "usageMetadata": {
                    "promptTokenCount": 3,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 5,
                },
            }
        )
    )
    monkeypatch.setattr("ipw.clients.gemini.requests.post", post)

    client = GeminiClient(model="gemini-3.5-flash")

    assert client.chat("system", "user") == "judge verdict"
    assert post.call_args.kwargs["params"] == {"key": "test-key"}
    assert "gemini-3.5-flash:generateContent" in post.call_args.args[0]


def test_gemini_accepts_openai_style_image_messages(monkeypatch) -> None:
    from ipw.clients.gemini import GeminiClient

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    post = Mock(return_value=_FakeResponse({"candidates": [], "usageMetadata": {}}))
    monkeypatch.setattr("ipw.clients.gemini.requests.post", post)
    encoded = base64.b64encode(b"image").decode("ascii")

    client = GeminiClient(model="gemini-3.5-flash")
    client._chat_completion(
        "gemini-3.5-flash",
        "",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            }
        ],
    )

    payload = post.call_args.kwargs["data"]
    assert "inline_data" in payload
    assert encoded in payload

