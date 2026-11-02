"""Tests for OpenHands request-history sanitization."""

from __future__ import annotations

import hashlib

import pytest

from ipw.agents.openhands import (
    GDPValReferenceStagingError,
    OpenHands,
    _drop_malformed_openhands_state_events,
    _drop_malformed_openhands_tail_events,
    _is_recoverable_tool_arg_error,
    _sanitize_litellm_messages,
    _stage_gdpval_inputs_for_workspace,
)
from ipw.execution.agentic_runner import _classify_agent_exception


class FakeMessage:
    def __init__(self, **data):
        self.data = data

    def model_dump(self, exclude_none=False):
        if not exclude_none:
            return dict(self.data)
        return {key: value for key, value in self.data.items() if value is not None}

    def model_copy(self, update=None):
        copied = dict(self.data)
        copied.update(update or {})
        return FakeMessage(**copied)


class UserMessageEvent:
    pass


class FileEditorActionEvent:
    pass


class ConversationErrorEvent:
    pass


class CondensationEvent:
    pass


class AgentStateChangedEvent:
    pass


class FakeState:
    def __init__(self, events):
        self.events = events


def test_valid_tool_call_history_is_unchanged() -> None:
    messages = [
        {"role": "user", "content": "make a file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command": "python build.py"}',
                    },
                }
            ],
        },
    ]

    sanitized, changed = _sanitize_litellm_messages(messages)

    assert changed is False
    assert sanitized is messages


def test_malformed_tool_call_is_omitted_but_preserved_as_text_context() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command": "python3 <',
                    },
                }
            ],
        }
    ]

    sanitized, changed = _sanitize_litellm_messages(messages)

    assert changed is True
    assert "tool_calls" not in sanitized[0]
    assert "malformed JSON arguments" in sanitized[0]["content"]
    assert "terminal" in sanitized[0]["content"]
    assert sanitized[0]["role"] == "assistant"


def test_malformed_json_content_is_converted_to_plain_text_note() -> None:
    messages = [{"role": "assistant", "content": '{"command": "python3 <'}]

    sanitized, changed = _sanitize_litellm_messages(messages)

    assert changed is True
    assert sanitized[0]["content"].startswith("[IPW harness note:")
    assert "malformed content" in sanitized[0]["content"]


def test_pydantic_style_message_is_preserved_when_sanitized() -> None:
    messages = [
        FakeMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "file_editor",
                        "arguments": '{"path": "report.py", "content": "unterminated',
                    },
                }
            ],
        )
    ]

    sanitized, changed = _sanitize_litellm_messages(messages)

    assert changed is True
    assert isinstance(sanitized[0], FakeMessage)
    assert "tool_calls" not in sanitized[0].model_dump(exclude_none=True)
    assert "file_editor" in sanitized[0].data["content"]


def test_recoverable_tool_arg_error_matches_openhands_badrequest_shape() -> None:
    exc = RuntimeError(
        "Conversation run failed: litellm.BadRequestError: OpenAIException - "
        "Unterminated string starting at: line 1 column 13 (char 12)"
    )

    assert _is_recoverable_tool_arg_error(exc) is True


def test_recoverable_tool_arg_error_does_not_match_arbitrary_badrequest() -> None:
    exc = RuntimeError("litellm.BadRequestError: context length exceeded")

    assert _is_recoverable_tool_arg_error(exc) is False


def test_drop_malformed_openhands_tail_events_removes_action_error_suffix() -> None:
    user_event = UserMessageEvent()
    events = [
        user_event,
        FileEditorActionEvent(),
        ConversationErrorEvent(),
        CondensationEvent(),
    ]

    removed = _drop_malformed_openhands_tail_events(events)

    assert removed == 2
    assert events[0] is user_event
    assert len(events) == 2
    assert isinstance(events[1], CondensationEvent)


def test_drop_malformed_openhands_tail_events_leaves_history_without_error() -> None:
    events = [UserMessageEvent(), FileEditorActionEvent()]

    removed = _drop_malformed_openhands_tail_events(events)

    assert removed == 0
    assert len(events) == 2


def test_drop_malformed_openhands_tail_events_scans_recent_bookkeeping() -> None:
    user_event = UserMessageEvent()
    bookkeeping_event = AgentStateChangedEvent()
    events = [
        user_event,
        FileEditorActionEvent(),
        bookkeeping_event,
        ConversationErrorEvent(),
        CondensationEvent(),
    ]

    removed = _drop_malformed_openhands_tail_events(events)

    assert removed == 2
    assert events[:2] == [user_event, bookkeeping_event]
    assert len(events) == 3
    assert isinstance(events[2], CondensationEvent)


def test_drop_malformed_openhands_state_events_writes_back_tuple_events() -> None:
    user_event = UserMessageEvent()
    bookkeeping_event = AgentStateChangedEvent()
    state = FakeState(
        (
            user_event,
            FileEditorActionEvent(),
            bookkeeping_event,
            ConversationErrorEvent(),
            CondensationEvent(),
        )
    )

    removed = _drop_malformed_openhands_state_events(state)

    assert removed == 2
    assert state.events[:2] == [user_event, bookkeeping_event]
    assert len(state.events) == 3
    assert isinstance(state.events[2], CondensationEvent)


def test_stage_gdpval_inputs_adds_inputs_and_root_aliases(tmp_path) -> None:
    cache_inputs = tmp_path / "cache" / "task" / "inputs"
    cache_inputs.mkdir(parents=True)
    source = cache_inputs / "clip.mp4"
    source.write_bytes(b"video")
    workspace = tmp_path / "workspace"
    metadata = {"gdpval_inputs_dir": str(cache_inputs)}

    staged = _stage_gdpval_inputs_for_workspace(metadata, workspace)

    input_copy = workspace / "inputs" / "clip.mp4"
    root_alias = workspace / "clip.mp4"
    assert input_copy.read_bytes() == b"video"
    assert root_alias.exists()
    assert root_alias.resolve() == input_copy.resolve()
    assert str(input_copy) in metadata["gdpval_reference_files"]
    assert str(root_alias) in metadata["gdpval_reference_files"]
    assert staged == [input_copy, root_alias]


def test_stage_gdpval_inputs_can_skip_root_aliases(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPW_OPENHANDS_GDPVAL_ROOT_INPUT_ALIASES", "0")
    cache_inputs = tmp_path / "cache" / "task" / "inputs"
    cache_inputs.mkdir(parents=True)
    source = cache_inputs / "clip.mp4"
    source.write_bytes(b"video")
    workspace = tmp_path / "workspace"
    metadata = {"gdpval_inputs_dir": str(cache_inputs)}

    staged = _stage_gdpval_inputs_for_workspace(metadata, workspace)

    input_copy = workspace / "inputs" / "clip.mp4"
    root_alias = workspace / "clip.mp4"
    assert input_copy.read_bytes() == b"video"
    assert not root_alias.exists()
    assert str(input_copy) in metadata["gdpval_reference_files"]
    assert str(root_alias) not in metadata["gdpval_reference_files"]
    assert staged == [input_copy]


def test_strict_gdpval_reference_staging_fails_before_agent_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IPW_REQUIRE_GDPVAL_REFERENCES", "1")
    cache_inputs = tmp_path / "cache" / "task" / "inputs"
    cache_inputs.mkdir(parents=True)
    source = cache_inputs / "fixture.pdf"
    source.write_bytes(b"expected")
    metadata = {
        "reference_file_hf_uris": ["hf://datasets/openai/gdpval/files/fixture.pdf"],
        "gdpval_inputs_dir": str(cache_inputs),
        "gdpval_reference_sha256": {"fixture.pdf": hashlib.sha256(b"expected").hexdigest()},
    }
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _stage_gdpval_inputs_for_workspace(metadata, workspace)
    assert (workspace / "inputs" / "fixture.pdf").read_bytes() == b"expected"

    source.write_bytes(b"changed")
    agent = OpenHands.__new__(OpenHands)
    agent._workspace = str(tmp_path / "second-workspace")
    with pytest.raises(GDPValReferenceStagingError):
        agent.set_task_metadata(metadata)
    assert _classify_agent_exception(GDPValReferenceStagingError("missing")) == (
        "infra_reference_error", "infra_reference_error"
    )
