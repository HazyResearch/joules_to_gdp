"""Integration tests for the OpenHands agent harness."""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from ipw.core.registry import AgentRegistry
from ipw.core.types import AgentRunResult
from ipw.telemetry.events import EventRecorder, EventType


@pytest.fixture(autouse=True)
def _clean_openhands_registration():
    """Ensure the AgentRegistry 'openhands' entry is cleared between tests.

    Each test re-imports ipw.agents.openhands (because the sys.modules mock
    removes it on teardown), which re-triggers @AgentRegistry.register.
    """
    yield
    entries = AgentRegistry._entries()
    entries.pop("openhands", None)
    # Remove cached module so next test can re-import cleanly
    sys.modules.pop("ipw.agents.openhands", None)


@pytest.fixture()
def _mock_openhands():
    """Patch openhands SDK imports used by the OpenHands agent."""
    mock_agent_cls = MagicMock()
    mock_conversation_cls = MagicMock()
    mock_condenser_cls = MagicMock()
    mock_event = MagicMock()
    mock_action_event = type("ActionEvent", (), {})
    mock_observation_event = type("ObservationEvent", (), {})
    mock_llm_convertible = type("LLMConvertibleEvent", (), {})

    modules = {
        "openhands": MagicMock(),
        "openhands.sdk": MagicMock(
            Agent=mock_agent_cls,
            Event=mock_event,
            LLMConvertibleEvent=mock_llm_convertible,
            LLMSummarizingCondenser=mock_condenser_cls,
            LocalConversation=mock_conversation_cls,
        ),
        "openhands.sdk.event": MagicMock(),
        "openhands.sdk.event.llm_convertible": MagicMock(),
        "openhands.sdk.event.llm_convertible.action": MagicMock(
            ActionEvent=mock_action_event,
        ),
        "openhands.sdk.event.llm_convertible.observation": MagicMock(
            ObservationEvent=mock_observation_event,
        ),
        "openhands.sdk.conversation": MagicMock(),
        "openhands.sdk.conversation.response_utils": MagicMock(
            get_agent_final_response=MagicMock(return_value="The final answer"),
        ),
    }

    with patch.dict("sys.modules", modules):
        yield {
            "Agent": mock_agent_cls,
            "LocalConversation": mock_conversation_cls,
            "Condenser": mock_condenser_cls,
            "ActionEvent": mock_action_event,
            "ObservationEvent": mock_observation_event,
            "get_agent_final_response": modules["openhands.sdk.conversation.response_utils"].get_agent_final_response,
        }


class TestOpenHandsIntegration:
    """Integration tests for the OpenHands agent with mocked SDK."""

    def test_gemma_text_tool_call_patch_is_opt_in(self, monkeypatch) -> None:
        from ipw.agents.openhands import _patch_text_tool_calls_if_enabled

        response = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '<|tool_call>call:terminal{command:<|"|>ls -la<|"|>,'
                            'security_risk:<|"|>LOW<|"|>,'
                            'summary:<|"|>List files.<|"|>}<tool_call|>'
                        )
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        assert _patch_text_tool_calls_if_enabled(response) is False
        monkeypatch.setenv("IPW_OPENHANDS_TEXT_TOOL_CALL_PARSER", "gemma")
        assert _patch_text_tool_calls_if_enabled(response) is True

        message = response["choices"][0]["message"]
        assert message["content"] is None
        assert response["choices"][0]["finish_reason"] == "tool_calls"
        assert message["tool_calls"] == [
            {
                "id": "ipw_gemma_text_tool_0",
                "type": "function",
                "function": {
                    "name": "terminal",
                    "arguments": (
                        '{"command":"ls -la","security_risk":"LOW",'
                        '"summary":"List files."}'
                    ),
                },
            }
        ]

    def test_litellm_text_tool_call_patch_is_opt_in(self, monkeypatch) -> None:
        from ipw.agents.openhands import _litellm_text_tool_call_patch

        def completion(**_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '<|tool_call>call:terminal{command:<|"|>pwd<|"|>,'
                                'security_risk:<|"|>LOW<|"|>,'
                                'summary:<|"|>Print workspace.<|"|>}<tool_call|>'
                            )
                        },
                        "finish_reason": "stop",
                    }
                ]
            }

        fake_litellm = types.SimpleNamespace(completion=completion)
        monkeypatch.setenv("IPW_OPENHANDS_TEXT_TOOL_CALL_PARSER", "gemma")
        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            with _litellm_text_tool_call_patch():
                response = fake_litellm.completion(model="gemma4-31b-it")

        message = response["choices"][0]["message"]
        assert response["choices"][0]["finish_reason"] == "tool_calls"
        assert message["content"] is None
        assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
            "command": "pwd",
            "security_risk": "LOW",
            "summary": "Print workspace.",
        }

    def test_glm_finish_message_text_is_patched_to_finish_tool(self, monkeypatch) -> None:
        from ipw.agents.openhands import _patch_text_tool_calls_if_enabled

        response = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "The file is already written.\n"
                            "Finish with message:\n"
                            "Created Sample v2.xlsx in the workspace."
                        )
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        assert _patch_text_tool_calls_if_enabled(response) is False
        monkeypatch.setenv("IPW_OPENHANDS_TEXT_TOOL_CALL_PARSER", "glm")
        assert _patch_text_tool_calls_if_enabled(response) is True

        choice = response["choices"][0]
        message = choice["message"]
        assert choice["finish_reason"] == "tool_calls"
        assert message["content"] is None
        assert message["tool_calls"][0]["function"]["name"] == "finish"
        assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
            "message": "Created Sample v2.xlsx in the workspace."
        }

    def test_glm_raw_xml_file_editor_call_is_patched(self, monkeypatch) -> None:
        from ipw.agents.openhands import _patch_text_tool_calls_if_enabled

        response = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "Let me write the script.\n"
                            "<tool_call>file_editor"
                            "<arg_key>command</arg_key><arg_value>create</arg_value>"
                            "<arg_key>path</arg_key><arg_value>/tmp/build.py</arg_value>"
                            "<arg_key>file_text</arg_key><arg_value>"
                            "print('hello')\n"
                            "</arg_value>"
                        )
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        assert _patch_text_tool_calls_if_enabled(response) is False
        monkeypatch.setenv("IPW_OPENHANDS_TEXT_TOOL_CALL_PARSER", "glm")
        assert _patch_text_tool_calls_if_enabled(response) is True

        choice = response["choices"][0]
        message = choice["message"]
        assert choice["finish_reason"] == "tool_calls"
        assert message["content"] is None
        function = message["tool_calls"][0]["function"]
        assert function["name"] == "file_editor"
        assert json.loads(function["arguments"]) == {
            "command": "create",
            "path": "/tmp/build.py",
            "file_text": "print('hello')\n",
        }

    def test_gpt_oss_channel_suffix_tool_names_are_normalized(self) -> None:
        from ipw.agents.openhands import _patch_text_tool_calls_if_enabled

        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "terminal<|channel|>commentary",
                                    "arguments": '{"command":"ls -la"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        assert _patch_text_tool_calls_if_enabled(response) is True
        assert (
            response["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
            == "terminal"
        )

    def test_malformed_multiline_terminal_arguments_are_repaired(self) -> None:
        from ipw.agents.openhands import _patch_text_tool_calls_if_enabled

        arguments = (
            '{"command":"cat <<\\\'PY\\\' > generate.py\n'
            'print("hello")\n'
            'PY"],"timeout":10000}'
        )
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "terminal<|channel|>commentary",
                                    "arguments": arguments,
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        assert _patch_text_tool_calls_if_enabled(response) is True
        function = response["choices"][0]["message"]["tool_calls"][0]["function"]
        assert function["name"] == "terminal"
        assert json.loads(function["arguments"]) == {
            "command": "cat <<\\'PY\\' > generate.py\nprint(\"hello\")\nPY",
            "timeout": 300,
        }

    def test_incomplete_heredoc_terminal_arguments_are_blocked(self) -> None:
        from ipw.agents.openhands import _patch_text_tool_calls_if_enabled

        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "terminal",
                                    "arguments": json.dumps(
                                        {
                                            "command": (
                                                "cat > build_theatre_cba.py <<'PY'\n"
                                                "from pathlib import Path\n"
                                                "Path('out.txt').write_text('ok')\n"
                                            ),
                                            "is_input": True,
                                            "reset": True,
                                            "timeout": 300,
                                        }
                                    ),
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        assert _patch_text_tool_calls_if_enabled(response) is True
        function = response["choices"][0]["message"]["tool_calls"][0]["function"]
        repaired = json.loads(function["arguments"])
        assert "blocked an incomplete terminal heredoc" in repaired["command"]
        assert repaired["timeout"] == 5
        assert repaired["summary"] == "Blocked incomplete heredoc terminal command"
        assert "is_input" not in repaired
        assert "reset" not in repaired

    def test_kimi_text_terminal_call_with_complete_heredoc_is_patched(self, monkeypatch) -> None:
        from ipw.agents.openhands import _patch_text_tool_calls_if_enabled

        response = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "function_call terminal: "
                            '{"command":"cat > build.py <<\'PY\'\n'
                            'from pathlib import Path\n'
                            'Path("sample v2.xlsx").write_text("ok")\n'
                            "PY"
                        )
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        monkeypatch.setenv("IPW_OPENHANDS_TEXT_TOOL_CALL_PARSER", "kimi")
        assert _patch_text_tool_calls_if_enabled(response) is True

        choice = response["choices"][0]
        message = choice["message"]
        assert choice["finish_reason"] == "tool_calls"
        assert message["content"] is None
        function = message["tool_calls"][0]["function"]
        assert function["name"] == "terminal"
        assert json.loads(function["arguments"]) == {
            "command": (
                "cat > build.py <<'PY'\n"
                "from pathlib import Path\n"
                "Path(\"sample v2.xlsx\").write_text(\"ok\")\n"
                "PY"
            )
        }

    def test_kimi_text_terminal_call_with_escaped_newlines_is_patched(self, monkeypatch) -> None:
        from ipw.agents.openhands import _patch_text_tool_calls_if_enabled

        response = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "function_call terminal: "
                            '{"command":"cat > build.py <<\'PY\'\\n'
                            'from pathlib import Path\\n'
                            'Path(\\"sample v2.xlsx\\").write_text(\\"ok\\")\\n'
                            "PY"
                        )
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        monkeypatch.setenv("IPW_OPENHANDS_TEXT_TOOL_CALL_PARSER", "kimi")
        assert _patch_text_tool_calls_if_enabled(response) is True

        function = response["choices"][0]["message"]["tool_calls"][0]["function"]
        assert function["name"] == "terminal"
        assert json.loads(function["arguments"]) == {
            "command": (
                "cat > build.py <<'PY'\n"
                "from pathlib import Path\n"
                'Path("sample v2.xlsx").write_text("ok")\n'
                "PY"
            )
        }

    def test_incomplete_kimi_heredoc_is_not_executed(self, monkeypatch) -> None:
        from ipw.agents.openhands import _patch_text_tool_calls_if_enabled

        response = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "function_call terminal: "
                            '{"command":"cat > build.py <<\\\'PY\\\'\n'
                            'from pathlib import Path\n'
                            'Path("sample v2.xlsx").write_text("ok")\n'
                        )
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        monkeypatch.setenv("IPW_OPENHANDS_TEXT_TOOL_CALL_PARSER", "kimi")
        assert _patch_text_tool_calls_if_enabled(response) is False
        assert "tool_calls" not in response["choices"][0]["message"]

    def test_initializes_with_model(self, _mock_openhands: dict) -> None:
        from ipw.agents.openhands import OpenHands

        model = MagicMock()
        agent = OpenHands(model=model)
        assert agent.model is model
        _mock_openhands["Agent"].assert_not_called()

    def test_run_returns_agent_run_result(self, _mock_openhands: dict) -> None:
        from ipw.agents.openhands import OpenHands

        _mock_openhands["get_agent_final_response"].return_value = "42"
        conv_mock = MagicMock()
        _mock_openhands["LocalConversation"].return_value = conv_mock

        model = MagicMock()
        agent = OpenHands(model=model)
        result = agent.run("What is 6 * 7?")

        assert isinstance(result, AgentRunResult)
        assert "42" in result.content
        conv_mock.send_message.assert_called_once_with("What is 6 * 7?")
        conv_mock.run.assert_called()

    def test_max_turns_respected(self, _mock_openhands: dict) -> None:
        from ipw.agents.openhands import OpenHands

        model = MagicMock()
        agent = OpenHands(model=model, max_turns=5)
        assert agent._max_turns == 5

    def test_agent_kwargs_forwarded_to_sdk_agent(self, _mock_openhands: dict) -> None:
        from ipw.agents.openhands import OpenHands

        conv_mock = MagicMock()
        _mock_openhands["LocalConversation"].return_value = conv_mock

        model = MagicMock()
        agent = OpenHands(model=model, include_default_tools=["finish"])
        agent.run("Create the deliverable.")

        assert (
            _mock_openhands["Agent"].call_args.kwargs["include_default_tools"]
            == ["finish"]
        )

    def test_agent_kwargs_cannot_override_managed_args(self, _mock_openhands: dict) -> None:
        from ipw.agents.openhands import OpenHands

        model = MagicMock()
        with pytest.raises(ValueError, match="cannot override managed arguments"):
            OpenHands(model=model, llm=MagicMock())

    def test_callback_records_tool_events(self, _mock_openhands: dict) -> None:
        from ipw.agents.openhands import OpenHands

        recorder = EventRecorder()
        model = MagicMock()
        agent = OpenHands(model=model, event_recorder=recorder)

        # Simulate an ActionEvent callback
        action_event = _mock_openhands["ActionEvent"]()
        action_event.tool_name = "bash"
        agent._instrumented_callback(action_event)

        # Simulate an ObservationEvent callback
        obs_event = _mock_openhands["ObservationEvent"]()
        obs_event.tool_name = "bash"
        agent._instrumented_callback(obs_event)

        events = recorder.get_events()
        assert len(events) == 2
        assert events[0].event_type == EventType.TOOL_CALL_START
        assert events[0].metadata["tool"] == "bash"
        assert events[1].event_type == EventType.TOOL_CALL_END

    def test_lm_events_recorded_on_run(self, _mock_openhands: dict) -> None:
        from ipw.agents.openhands import OpenHands

        _mock_openhands["get_agent_final_response"].return_value = "done"
        conv_mock = MagicMock()
        _mock_openhands["LocalConversation"].return_value = conv_mock

        recorder = EventRecorder()
        model = MagicMock()
        agent = OpenHands(model=model, event_recorder=recorder)

        def _run_once() -> None:
            agent._instrumented_model.completion(messages=[])

        conv_mock.run.side_effect = _run_once
        agent.run("test")

        events = recorder.get_events()
        event_types = [e.event_type for e in events]
        assert EventType.LM_INFERENCE_START in event_types
        assert EventType.LM_INFERENCE_END in event_types

    def test_lm_events_recorded_per_openhands_model_call(
        self, _mock_openhands: dict
    ) -> None:
        from ipw.agents.openhands import OpenHands

        _mock_openhands["get_agent_final_response"].return_value = "done"
        conv_mock = MagicMock()
        _mock_openhands["LocalConversation"].return_value = conv_mock

        recorder = EventRecorder()
        model = MagicMock()
        usage = MagicMock()
        usage.prompt_tokens = 0
        usage.completion_tokens = 0
        model.metrics.accumulated_token_usage = usage
        model.metrics.accumulated_cost = 0.0

        def _completion(*args, **kwargs):
            usage.prompt_tokens += 10
            usage.completion_tokens += 4
            return "ok"

        model.completion.side_effect = _completion
        agent = OpenHands(model=model, event_recorder=recorder)

        def _run_twice() -> None:
            agent._instrumented_model.completion(messages=[])
            agent._instrumented_model.completion(messages=[])

        conv_mock.run.side_effect = _run_twice
        agent.run("test")

        events = recorder.get_events()
        lm_starts = [
            event for event in events
            if event.event_type == EventType.LM_INFERENCE_START
        ]
        lm_ends = [
            event for event in events
            if event.event_type == EventType.LM_INFERENCE_END
        ]

        assert len(lm_starts) == 2
        assert len(lm_ends) == 2
        assert [event.metadata["prompt_tokens"] for event in lm_ends] == [10, 10]
        assert [event.metadata["completion_tokens"] for event in lm_ends] == [4, 4]

    def test_conversation_closed_after_run(self, _mock_openhands: dict) -> None:
        from ipw.agents.openhands import OpenHands

        _mock_openhands["get_agent_final_response"].return_value = "done"
        conv_mock = MagicMock()
        _mock_openhands["LocalConversation"].return_value = conv_mock

        model = MagicMock()
        agent = OpenHands(model=model)
        agent.run("test")

        conv_mock.close.assert_called_once()

    def test_set_workspace(self, _mock_openhands: dict) -> None:
        from ipw.agents.openhands import OpenHands

        model = MagicMock()
        agent = OpenHands(model=model)
        agent.set_workspace("/tmp/workspace")
        assert agent._workspace == "/tmp/workspace"
