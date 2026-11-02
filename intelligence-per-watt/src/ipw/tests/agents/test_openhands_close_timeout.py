"""Tests for bounded OpenHands conversation cleanup."""

from __future__ import annotations

import sys
import threading
import time
import types

import pytest

from ipw.agents.openhands import (
    OpenHands,
    _close_conversation_with_timeout,
    _conversation_state_has_finish,
    _conversation_state_has_max_iteration_exhausted,
    _finish_log_contains_finish,
    _finish_log_contains_max_iteration,
    _finish_log_message,
    _finish_log_watch_start,
    _finish_text_fallback,
    _gdpval_valid_deliverable_snapshot,
    _looks_like_openhands_finish,
)


def test_close_conversation_with_timeout_returns_on_hung_close() -> None:
    started = threading.Event()
    release = threading.Event()

    class HangingConversation:
        def close(self):
            started.set()
            release.wait(timeout=5)

    t0 = time.monotonic()
    try:
        _close_conversation_with_timeout(HangingConversation(), timeout_s=0.01)
    finally:
        release.set()

    assert started.wait(timeout=1)
    assert time.monotonic() - t0 < 1


def test_close_conversation_with_timeout_reraises_close_error() -> None:
    class ExplodingConversation:
        def close(self):
            raise RuntimeError("close failed")

    with pytest.raises(RuntimeError, match="close failed"):
        _close_conversation_with_timeout(ExplodingConversation(), timeout_s=1)


def test_gdpval_early_return_requires_all_expected_deliverables(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "summary.txt").write_text("valid partial deliverable", encoding="utf-8")
    metadata = {
        "dataset_name": "gdpval",
        "gdpval_outputs_dir": str(workspace),
        "deliverable_files": ["summary.txt", "action_items.txt"],
    }

    assert _gdpval_valid_deliverable_snapshot(metadata, "Create two text files.") is None

    (workspace / "action_items.txt").write_text("second deliverable", encoding="utf-8")

    assert _gdpval_valid_deliverable_snapshot(metadata, "Create two text files.") is not None


def test_finish_action_marks_finish_observed() -> None:
    class FinishAction:
        tool_name = "finish"

        def to_llm_message(self):
            return "done"

    agent = OpenHands.__new__(OpenHands)
    agent.event_recorder = None
    agent._ActionEvent = FinishAction
    agent._ObservationEvent = type("Observation", (), {})
    agent._LLMConvertibleEvent = FinishAction
    agent._pending_tool = None
    agent._tool_names_used = []
    agent._num_turns = 0
    agent.current_result = ""
    agent._finish_observed = threading.Event()

    agent._instrumented_callback(FinishAction())

    assert agent.current_result == "done"
    assert agent._tool_names_used == ["finish"]
    assert agent._num_turns == 1
    assert agent._finish_observed.is_set()


def test_finish_detection_handles_sdk_variants() -> None:
    class FinishToolAction:
        tool_name = "FinishTool"

    class FinishMessageEvent:
        def __str__(self) -> str:
            return "Finish with message:\ndone"

    class NestedFinishEvent:
        action = type("FinishAction", (), {})()

    class FinishSummaryEvent:
        summary = "Created the file\n\nFinish with message:\nCreated report.docx"

    assert _looks_like_openhands_finish(tool_name="finish")
    assert _looks_like_openhands_finish(FinishToolAction())
    assert _looks_like_openhands_finish(FinishMessageEvent())
    assert _looks_like_openhands_finish(NestedFinishEvent())
    assert _looks_like_openhands_finish(FinishSummaryEvent())
    assert _looks_like_openhands_finish(llm_message="Finish with message:\ndone")
    assert not _looks_like_openhands_finish(tool_name="unfinished")


def test_finish_text_fallback_extracts_plain_finish_message() -> None:
    assert (
        _finish_text_fallback(
            "The file is verified.\nFinish with message:\nCreated Sample v2.xlsx"
        )
        == "Created Sample v2.xlsx"
    )


def test_run_uses_observed_finish_text_without_final_nudge(monkeypatch) -> None:
    import ipw.agents.openhands as openhands_mod

    response_utils = types.ModuleType("openhands.sdk.conversation.response_utils")
    response_utils.get_agent_final_response = lambda events: ""
    monkeypatch.setitem(
        sys.modules,
        "openhands.sdk.conversation.response_utils",
        response_utils,
    )

    class Conversation:
        def __init__(self) -> None:
            self.messages: list[str] = []
            self.state = type("State", (), {"events": []})()

        def send_message(self, message: str) -> None:
            self.messages.append(message)

        def close(self) -> None:
            pass

    conversation = Conversation()
    agent = openhands_mod.OpenHands.__new__(openhands_mod.OpenHands)
    agent.model = object()
    agent.current_result = ""
    agent._task_metadata = None
    agent._create_conversation = lambda: conversation

    def mark_finished() -> None:
        agent.current_result = (
            "Everything is verified.\n"
            "Finish with message:\n"
            "Created Sample v2.xlsx in the workspace."
        )
        agent._finish_observed.set()

    agent._run_conversation_with_tool_arg_repair = mark_finished

    result = agent.run("do the task")

    assert conversation.messages == ["do the task"]
    assert result.content == "Created Sample v2.xlsx in the workspace."


def test_run_uses_unobserved_finish_text_without_final_nudge(monkeypatch) -> None:
    import ipw.agents.openhands as openhands_mod

    response_utils = types.ModuleType("openhands.sdk.conversation.response_utils")
    response_utils.get_agent_final_response = lambda events: ""
    monkeypatch.setitem(
        sys.modules,
        "openhands.sdk.conversation.response_utils",
        response_utils,
    )
    monkeypatch.setenv("IPW_OPENHANDS_FINAL_NUDGE_MODE", "skip")

    class Conversation:
        def __init__(self) -> None:
            self.messages: list[str] = []
            self.state = type("State", (), {"events": []})()

        def send_message(self, message: str) -> None:
            self.messages.append(message)

        def close(self) -> None:
            pass

    conversation = Conversation()
    agent = openhands_mod.OpenHands.__new__(openhands_mod.OpenHands)
    agent.model = object()
    agent.current_result = ""
    agent._task_metadata = None
    agent._create_conversation = lambda: conversation

    def write_finish_text() -> None:
        agent.current_result = (
            "All checks passed.\n"
            "Finish with message:\n"
            "Created Sample v3.xlsx in the workspace."
        )

    agent._run_conversation_with_tool_arg_repair = write_finish_text

    result = agent.run("do the task")

    assert conversation.messages == ["do the task"]
    assert result.content == "Created Sample v3.xlsx in the workspace."


def test_final_nudge_finish_during_send_message_skips_second_run(
    monkeypatch,
) -> None:
    import ipw.agents.openhands as openhands_mod

    response_utils = types.ModuleType("openhands.sdk.conversation.response_utils")
    response_utils.get_agent_final_response = lambda events: ""
    monkeypatch.setitem(
        sys.modules,
        "openhands.sdk.conversation.response_utils",
        response_utils,
    )

    nudge_offset = 123
    monkeypatch.setattr(
        openhands_mod,
        "_finish_log_watch_start",
        lambda: (None, nudge_offset),
    )
    monkeypatch.setattr(
        openhands_mod,
        "_finish_log_message",
        lambda path, offset: "Created report.xlsx",
    )

    class Conversation:
        def __init__(self) -> None:
            self.messages: list[str] = []
            self.max_iteration_per_run = 120
            self.state = type("State", (), {"events": []})()

        def send_message(self, message: str) -> None:
            self.messages.append(message)

        def close(self) -> None:
            pass

    conversation = Conversation()
    agent = openhands_mod.OpenHands.__new__(openhands_mod.OpenHands)
    agent.model = object()
    agent.current_result = ""
    agent._task_metadata = None
    agent._create_conversation = lambda: conversation
    agent._run_conversation_with_tool_arg_repair = lambda: None

    result = agent.run("do the task")

    assert len(conversation.messages) == 2
    assert conversation.messages[0] == "do the task"
    assert "Submit your final answer now" in conversation.messages[1]
    assert result.content == "Created report.xlsx"


def test_run_conversation_returns_when_finish_observed_after_sdk_hang() -> None:
    started = threading.Event()
    release = threading.Event()

    class HangingConversation:
        def run(self):
            started.set()
            release.wait(timeout=5)

    agent = OpenHands.__new__(OpenHands)
    agent.conversation = HangingConversation()
    agent._finish_observed = threading.Event()

    def mark_finished() -> None:
        assert started.wait(timeout=1)
        agent._finish_observed.set()

    marker = threading.Thread(target=mark_finished, daemon=True)
    marker.start()

    t0 = time.monotonic()
    try:
        agent._run_conversation_with_tool_arg_repair()
    finally:
        release.set()

    assert time.monotonic() - t0 < 1


def test_run_conversation_waits_for_sdk_thread_when_early_return_disabled(monkeypatch) -> None:
    monkeypatch.setenv("IPW_OPENHANDS_EARLY_RETURN_ON_FINISH", "0")
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class HangingConversation:
        def run(self):
            started.set()
            release.wait(timeout=5)
            finished.set()

    agent = OpenHands.__new__(OpenHands)
    agent.conversation = HangingConversation()
    agent.current_result = ""
    agent._finish_observed = threading.Event()

    def mark_finished_then_release() -> None:
        assert started.wait(timeout=1)
        agent._finish_observed.set()
        time.sleep(0.1)
        release.set()

    marker = threading.Thread(target=mark_finished_then_release, daemon=True)
    marker.start()

    t0 = time.monotonic()
    agent._run_conversation_with_tool_arg_repair()

    assert finished.is_set()
    assert time.monotonic() - t0 >= 0.1


def test_run_conversation_returns_when_current_result_has_finish_text() -> None:
    started = threading.Event()
    release = threading.Event()

    class HangingConversation:
        def __init__(self):
            self.state = type("State", (), {"events": []})()

        def run(self):
            started.set()
            release.wait(timeout=5)

    conversation = HangingConversation()
    agent = OpenHands.__new__(OpenHands)
    agent.conversation = conversation
    agent.current_result = ""
    agent._finish_observed = threading.Event()

    def mark_finished() -> None:
        assert started.wait(timeout=1)
        agent.current_result = (
            "Created the deliverable.\n"
            "Finish with message:\n"
            "Created final memo.docx"
        )

    marker = threading.Thread(target=mark_finished, daemon=True)
    marker.start()

    t0 = time.monotonic()
    try:
        agent._run_conversation_with_tool_arg_repair()
    finally:
        release.set()

    assert agent._finish_observed.is_set()
    assert time.monotonic() - t0 < 1


def test_run_conversation_returns_when_finish_appears_in_state_events() -> None:
    started = threading.Event()
    release = threading.Event()

    class FinishAction:
        pass

    class FinishEvent:
        action = FinishAction()

    class HangingConversation:
        def __init__(self):
            self.state = type("State", (), {"events": []})()

        def run(self):
            started.set()
            release.wait(timeout=5)

    conversation = HangingConversation()
    agent = OpenHands.__new__(OpenHands)
    agent.conversation = conversation
    agent._finish_observed = threading.Event()

    def mark_finished() -> None:
        assert started.wait(timeout=1)
        conversation.state.events.append(FinishEvent())

    marker = threading.Thread(target=mark_finished, daemon=True)
    marker.start()

    t0 = time.monotonic()
    try:
        agent._run_conversation_with_tool_arg_repair()
    finally:
        release.set()

    assert _conversation_state_has_finish(conversation)
    assert agent._finish_observed.is_set()
    assert time.monotonic() - t0 < 1


def test_run_conversation_returns_when_max_iteration_event_appears() -> None:
    started = threading.Event()
    release = threading.Event()

    class ConversationErrorEvent:
        message = "Agent reached maximum iterations limit (120)."

    class HangingConversation:
        def __init__(self):
            self.state = type("State", (), {"events": []})()

        def run(self):
            started.set()
            release.wait(timeout=5)

    conversation = HangingConversation()
    agent = OpenHands.__new__(OpenHands)
    agent.conversation = conversation
    agent._finish_observed = threading.Event()

    def mark_exhausted() -> None:
        assert started.wait(timeout=1)
        conversation.state.events.append(ConversationErrorEvent())

    marker = threading.Thread(target=mark_exhausted, daemon=True)
    marker.start()

    t0 = time.monotonic()
    try:
        agent._run_conversation_with_tool_arg_repair()
    finally:
        release.set()

    assert _conversation_state_has_max_iteration_exhausted(conversation)
    assert time.monotonic() - t0 < 1


def test_finish_log_detection_only_reads_after_offset(tmp_path) -> None:
    log_path = tmp_path / "run.log"
    log_path.write_text("Finish with message:\nold\n", encoding="utf-8")
    offset = log_path.stat().st_size

    assert not _finish_log_contains_finish(log_path, offset)

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("Agent Action\nFinish with message:\nnew\n")

    assert _finish_log_contains_finish(log_path, offset)
    assert _finish_log_message(log_path, offset) == "new"


def test_max_iteration_log_detection_only_reads_after_offset(tmp_path) -> None:
    log_path = tmp_path / "run.log"
    log_path.write_text("Agent reached maximum iterations limit (old).\n", encoding="utf-8")
    offset = log_path.stat().st_size

    assert not _finish_log_contains_max_iteration(log_path, offset)

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("Agent reached maximum iterations limit (120).\n")

    assert _finish_log_contains_max_iteration(log_path, offset)


def test_finish_log_watch_start_uses_explicit_log_path(monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "outer.log"
    log_path.write_text("before\n", encoding="utf-8")
    monkeypatch.setenv("IPW_OPENHANDS_FINISH_LOG_WATCH", "1")
    monkeypatch.setenv("IPW_OPENHANDS_FINISH_LOG_PATH", str(log_path))

    path, offset = _finish_log_watch_start()

    assert path == log_path
    assert offset == log_path.stat().st_size
