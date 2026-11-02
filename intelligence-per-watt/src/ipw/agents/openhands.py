"""OpenHands agent implementation with per-tool energy tracking."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, MutableMapping, Optional, Sequence

from ipw.agents.base import BaseAgent
from ipw.core.registry import AgentRegistry
from ipw.core.types import AgentRunResult

if TYPE_CHECKING:
    from ipw.telemetry.events import EventRecorder

logger = logging.getLogger(__name__)

_OPENHANDS_CONVERSATION_INIT_LOCK = threading.Lock()


def _usage_counts(model: Any) -> tuple[int | None, int | None, float | None]:
    """Return accumulated prompt/output tokens and cost from an OpenHands LLM."""
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    try:
        metrics = model.metrics
        if metrics.accumulated_token_usage is not None:
            prompt_tokens = metrics.accumulated_token_usage.prompt_tokens
            completion_tokens = metrics.accumulated_token_usage.completion_tokens
            if isinstance(prompt_tokens, (int, float)):
                input_tokens = int(prompt_tokens)
            if isinstance(completion_tokens, (int, float)):
                output_tokens = int(completion_tokens)
        accumulated_cost = metrics.accumulated_cost
        if isinstance(accumulated_cost, (int, float)):
            cost_usd = float(accumulated_cost)
    except Exception:
        pass
    return input_tokens, output_tokens, cost_usd


def _looks_like_openhands_finish(
    event: Any = None,
    *,
    tool_name: Any = None,
    llm_message: Any = None,
) -> bool:
    """Return true for the finish markers emitted by different SDK versions."""
    candidates = [tool_name, llm_message]
    if event is not None:
        action = getattr(event, "action", None)
        observation = getattr(event, "observation", None)
        event_text_attrs = (
            "message",
            "error",
            "reason",
            "content",
            "summary",
            "thought",
            "final_message",
        )
        candidates.extend(
            [
                event,
                event.__class__.__name__,
                getattr(event, "tool_name", None),
                getattr(event, "name", None),
                action,
                getattr(action, "__class__", type(None)).__name__,
                getattr(action, "tool_name", None),
                getattr(action, "name", None),
                observation,
                getattr(observation, "__class__", type(None)).__name__,
                getattr(observation, "tool_name", None),
                getattr(observation, "name", None),
            ]
        )
        for source in (event, action, observation):
            if source is None:
                continue
            for attr in event_text_attrs:
                candidates.append(getattr(source, attr, None))
            model_dump = getattr(source, "model_dump", None)
            if callable(model_dump):
                try:
                    candidates.append(json.dumps(model_dump(), default=str))
                except Exception:
                    pass

    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip().lower()
        normalized = re.sub(r"[^a-z0-9]+", "", text)
        if normalized in {"finish", "finishtool", "finishaction"}:
            return True
        if "finishwithmessage" in normalized:
            return True
        if text.startswith("finish with message:"):
            return True
    return False


def _close_conversation_with_timeout(conversation: Any, timeout_s: float) -> None:
    """Close an OpenHands conversation without letting SDK cleanup hang a run."""
    error: list[BaseException] = []

    def _close() -> None:
        try:
            conversation.close()
        except BaseException as exc:  # pragma: no cover - re-raised below
            error.append(exc)

    thread = threading.Thread(target=_close, name="openhands-conversation-close", daemon=True)
    thread.start()
    thread.join(timeout=max(0.0, timeout_s))
    if thread.is_alive():
        logger.warning(
            "Timed out closing OpenHands conversation after %.1fs; continuing",
            timeout_s,
        )
        return
    if error:
        raise error[0]


def _request_conversation_run_stop(conversation: Any, reason: str) -> bool:
    """Ask the OpenHands run loop to stop before cleanup closes executors."""
    state = getattr(conversation, "_state", None) or getattr(conversation, "state", None)
    if state is None:
        logger.debug("OpenHands conversation has no state to stop (%s)", reason)
        return False
    try:
        current_status = getattr(state, "execution_status", None)
        status_type = type(current_status)
        paused_status = getattr(status_type, "PAUSED", None)
        if paused_status is None:
            logger.debug(
                "OpenHands execution status has no PAUSED value; cannot pause run loop (%s)",
                reason,
            )
            return False
        with state:
            if getattr(state, "execution_status", None) != paused_status:
                state.execution_status = paused_status
        logger.debug("Requested OpenHands run-loop stop after early return (%s)", reason)
        return True
    except Exception as exc:
        logger.warning("Failed to request OpenHands run-loop stop (%s): %s", reason, exc)
        return False


def _conversation_state_has_finish(conversation: Any) -> bool:
    state = getattr(conversation, "state", None)
    events = getattr(state, "events", None)
    if not events:
        return False
    try:
        recent_events = list(events)[-50:]
    except TypeError:
        return False
    return any(_looks_like_openhands_finish(event) for event in recent_events)


_OPENHANDS_MAX_ITERATION_MARKERS = (
    "agent reached maximum iterations limit",
    "maximum iterations limit",
    "max_iteration_per_run",
    "max iterations exhausted",
)


def _text_has_openhands_max_iteration(text: str) -> bool:
    lowered = text.strip().lower()
    return any(marker in lowered for marker in _OPENHANDS_MAX_ITERATION_MARKERS)


def _event_text_candidates(event: Any) -> list[str]:
    candidates = [type(event).__name__]
    for value in (
        event,
        getattr(event, "message", None),
        getattr(event, "error", None),
        getattr(event, "reason", None),
        getattr(event, "content", None),
    ):
        if value is not None:
            candidates.append(str(value))
    model_dump = getattr(event, "model_dump", None)
    if callable(model_dump):
        try:
            candidates.append(json.dumps(model_dump(), default=str))
        except Exception:
            pass
    return candidates


def _looks_like_openhands_max_iteration(event: Any) -> bool:
    name = type(event).__name__.lower()
    if "conversationerror" not in name and "errorevent" not in name:
        return False
    return any(_text_has_openhands_max_iteration(text) for text in _event_text_candidates(event))


def _conversation_state_has_max_iteration_exhausted(conversation: Any) -> bool:
    state = getattr(conversation, "state", None)
    events = getattr(state, "events", None)
    if not events:
        return False
    try:
        recent_events = list(events)[-50:]
    except TypeError:
        return False
    return any(_looks_like_openhands_max_iteration(event) for event in recent_events)


def _gdpval_valid_deliverable_snapshot(
    metadata: MutableMapping[str, Any] | None,
    problem: str,
) -> tuple[tuple[tuple[str, int, int], ...], str, float] | None:
    """Return a stable-file signature when GDPval has valid primary artifacts."""
    if not metadata:
        return None
    dataset_name = str(metadata.get("dataset_name") or "")
    if "gdpval" not in dataset_name.lower():
        return None
    if not metadata.get("gdpval_outputs_dir"):
        return None

    try:
        from ..evaluation.deliverables import (
            deliverable_paths_from_metadata,
            validate_deliverable_contract,
        )

        metadata_copy = dict(metadata)
        paths, _ = deliverable_paths_from_metadata(metadata_copy)
        validation = validate_deliverable_contract(
            paths=paths,
            problem=problem,
            metadata=metadata_copy,
        )
    except Exception as exc:
        logger.debug("GDPval deliverable early-return validation unavailable: %s", exc)
        return None

    if validation.get("status") == "invalid":
        return None
    primary_files = validation.get("primary_files") or validation.get("files") or []
    expected_filenames = validation.get("expected_filenames") or []
    if (
        expected_filenames
        and not validation.get("explicit_submission")
        and len(primary_files) < len(expected_filenames)
    ):
        return None
    signature: list[tuple[str, int, int]] = []
    names: list[str] = []
    for raw_path in primary_files:
        path = Path(str(raw_path))
        try:
            stat = path.stat()
        except OSError:
            return None
        if not path.is_file() or stat.st_size <= 0:
            return None
        signature.append((str(path), int(stat.st_size), int(stat.st_mtime_ns)))
        names.append(path.name)
    if not signature:
        return None
    newest_mtime_s = max(item[2] / 1_000_000_000 for item in signature)
    return tuple(sorted(signature)), ", ".join(sorted(names)), newest_mtime_s


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _finish_log_watch_start() -> tuple[Path | None, int]:
    if not _env_truthy("IPW_OPENHANDS_FINISH_LOG_WATCH"):
        return None, 0
    explicit_path = os.getenv("IPW_OPENHANDS_FINISH_LOG_PATH", "").strip()
    if explicit_path:
        path = Path(explicit_path)
        if path.is_absolute():
            try:
                return path, path.stat().st_size
            except OSError:
                return path, 0
    try:
        stdout_target = os.readlink("/proc/self/fd/1")
    except OSError:
        return None, 0
    if not stdout_target.startswith("/"):
        return None, 0
    path = Path(stdout_target)
    try:
        return path, path.stat().st_size
    except OSError:
        return None, 0


def _finish_log_contains_finish(path: Path | None, offset: int) -> bool:
    return _finish_log_message(path, offset) is not None


def _finish_log_contains_max_iteration(path: Path | None, offset: int) -> bool:
    if path is None:
        return False
    try:
        size = path.stat().st_size
        if size <= offset:
            return False
        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
    except OSError:
        return False
    text = data.decode("utf-8", errors="replace")
    return _text_has_openhands_max_iteration(text)


def _finish_log_message(path: Path | None, offset: int) -> str | None:
    if path is None:
        return None
    try:
        size = path.stat().st_size
        if size <= offset:
            return None
        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
    except OSError:
        return None
    marker_pos = data.lower().rfind(b"finish with message:")
    if marker_pos < 0:
        return None
    text = data[marker_pos:marker_pos + 8192].decode("utf-8", errors="replace")
    return _plain_finish_message(text) or "Done."


_CALL_METHODS = {
    "_transport_call",
    "call",
    "chat",
    "completion",
    "acompletion",
    "complete",
    "acomplete",
    "invoke",
    "ainvoke",
}


_MALFORMED_TOOL_HISTORY_NOTE = (
    "[IPW harness note: a previous assistant tool call had malformed JSON "
    "arguments and was omitted from tool_call history so the model server can "
    "continue the conversation. Treat it as an invalid attempted tool call.]"
)

_MALFORMED_TOOL_RETRY_PROMPT = (
    "Your previous tool call could not be executed because its JSON arguments "
    "were malformed or truncated. Continue the task, but reissue that tool call "
    "only if you can provide complete valid JSON. For terminal commands, use a "
    "single valid JSON object with a complete string value for `command`; avoid "
    "unfinished heredocs or bare redirection markers. If the failed call was "
    "`file_editor`, do not retry a large `file_text` or `new_str` payload; use "
    "existing files, smaller edits, or a short terminal command instead."
)

_TOOL_CHANNEL_SUFFIX_RE = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)<\|channel\|>[A-Za-z_][A-Za-z0-9_.-]*$"
)

_GEMMA_TEXT_TOOL_CALL_RE = re.compile(
    r"<\|tool_call\>call:(?P<name>[A-Za-z_][\w.-]*)\{(?P<body>.*?)\}<tool_call\|>",
    re.DOTALL,
)
_GEMMA_TEXT_TOOL_ARG_RE = re.compile(
    r"(?P<key>[A-Za-z_][\w.-]*)\s*:\s*<\|\"\|>(?P<value>.*?)<\|\"\|>\s*(?:,|$)",
    re.DOTALL,
)
_KIMI_TEXT_TOOL_PARSER_NAMES = {"kimi", "kimi-k2", "kimi-k2.7", "kimi-k2.7-code"}
_GLM_TEXT_TOOL_PARSER_NAMES = {"glm", "glm45", "glm47", "glm-5", "glm-5.2"}
_GLM_XML_TOOL_START_RE = re.compile(
    r"<tool_call>\s*(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)",
    re.DOTALL,
)
_GLM_XML_TOOL_ARG_RE = re.compile(
    r"<arg_key>(?P<key>.*?)</arg_key>\s*"
    r"<arg_value>(?P<value>.*?)</arg_value>",
    re.DOTALL,
)
_KIMI_TEXT_TOOL_PREFIX_RE = re.compile(
    r"(?:^|\n)(?:tool[_ -]?call|function[_ -]?call)\s*(?:for\s+)?terminal\s*:\s*",
    re.IGNORECASE,
)
_FINISH_WITH_MESSAGE_RE = re.compile(r"finish\s+with\s+message\s*:", re.IGNORECASE)
_TERMINAL_COMMAND_ARG_RE = re.compile(r'"command"\s*:\s*"')
_TERMINAL_COMMAND_END_RE = re.compile(
    r'"\]?\s*(?:,\s*"(?:timeout|security_risk|summary|is_input|working_dir)"\s*:|}\s*$)',
    re.DOTALL,
)
_TERMINAL_TIMEOUT_ARG_RE = re.compile(r'"timeout"\s*:\s*(?P<value>\d+)')
_TERMINAL_BOOL_ARG_TEMPLATE = r'"{key}"\s*:\s*(?P<value>true|false)'
_TERMINAL_STRING_ARG_TEMPLATE = r'"{key}"\s*:\s*"(?P<value>.*?)"'
_HEREDOC_START_RE = re.compile(
    r"<<-?\s*(?:'(?P<single>[^'\n]+)'|\"(?P<double>[^\"\n]+)\"|(?P<bare>[A-Za-z_][A-Za-z0-9_]*))"
)
_DEFAULT_TERMINAL_TIMEOUT_MAX_SECONDS = 300
_FINAL_NUDGE_SKIP_VALUES = {"0", "false", "no", "off", "skip", "none", "disabled"}
_INCOMPLETE_HEREDOC_BLOCK_COMMAND = (
    "printf '%s\\n' 'IPW harness blocked an incomplete terminal heredoc. "
    "Reissue a complete command or use file_editor for large files.' >&2; exit 64"
)


def _env_enabled(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FINAL_NUDGE_SKIP_VALUES


def _trace_openhands_return(message: str, *args: Any) -> None:
    if _env_enabled("IPW_OPENHANDS_TRACE_RETURN", default=False):
        logger.warning(message, *args)


def _append_metadata_paths(metadata: MutableMapping[str, Any], key: str, paths: list[Path]) -> None:
    existing = metadata.get(key)
    if isinstance(existing, (list, tuple, set)):
        values = [str(item) for item in existing]
    elif existing:
        values = [str(existing)]
    else:
        values = []
    seen = set(values)
    for path in paths:
        value = str(path)
        if value not in seen:
            values.append(value)
            seen.add(value)
    if values:
        metadata[key] = values


def _copy_or_link_input_alias(src: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return
    if src.is_dir():
        shutil.copytree(src, target, symlinks=True)
        return
    try:
        target.symlink_to(src)
    except OSError:
        shutil.copy2(src, target)


class GDPValReferenceStagingError(RuntimeError):
    ipw_failure_type = "infra_reference_error"


def _reference_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_gdpval_inputs_for_workspace(metadata: MutableMapping[str, Any], workspace: str | Path) -> list[Path]:
    inputs_dir = metadata.get("gdpval_inputs_dir") if metadata else None
    if not inputs_dir:
        return []

    workspace_path = Path(workspace)
    dest = workspace_path / "inputs"
    dest.mkdir(parents=True, exist_ok=True)
    staged_reference_paths: list[Path] = []
    create_root_aliases = _env_enabled("IPW_OPENHANDS_GDPVAL_ROOT_INPUT_ALIASES", default=True)

    strict = os.getenv("IPW_REQUIRE_GDPVAL_REFERENCES") == "1"
    expected = metadata.get("gdpval_reference_sha256")
    if strict:
        if not isinstance(expected, dict) or len(expected) != len(metadata.get("reference_file_hf_uris") or []):
            raise GDPValReferenceStagingError("GDPVal reference manifest is incomplete")
        names = list(expected)
        if any(not isinstance(name, str) or Path(name).name != name for name in names):
            raise GDPValReferenceStagingError("GDPVal reference manifest has an unsafe filename")
        sources = [Path(inputs_dir) / name for name in names]
    else:
        sources = list(Path(inputs_dir).iterdir())

    for src in sources:
        if strict:
            digest = expected[src.name]
            if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    or not src.is_file() or src.is_symlink()
                    or _reference_digest(src) != digest):
                raise GDPValReferenceStagingError(f"GDPVal reference cache is incomplete: {src.name}")
        input_target = dest / src.name
        if strict and input_target.is_symlink():
            raise GDPValReferenceStagingError(f"GDPVal workspace input is a symlink: {src.name}")
        if not input_target.exists():
            if src.is_dir():
                shutil.copytree(src, input_target, symlinks=True)
            else:
                shutil.copy2(src, input_target)
        if strict and (not input_target.is_file() or _reference_digest(input_target) != digest):
            raise GDPValReferenceStagingError(f"GDPVal workspace input differs from cache: {src.name}")
        staged_reference_paths.append(input_target)

        # GDPval prompts often mention files by basename. Keep inputs/ as the
        # canonical location, but provide a root alias so generated scripts that
        # use bare filenames do not fail because of harness path layout.
        root_alias = workspace_path / src.name
        if create_root_aliases and root_alias != input_target:
            _copy_or_link_input_alias(input_target, root_alias)
            if strict and (not root_alias.is_file() or _reference_digest(root_alias) != digest):
                raise GDPValReferenceStagingError(f"GDPVal reference alias differs from input: {src.name}")
            staged_reference_paths.append(root_alias)

    _append_metadata_paths(metadata, "gdpval_reference_files", staged_reference_paths)
    return staged_reference_paths


def _is_recoverable_tool_arg_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    if not (
        "unterminated string" in text
        or "jsondecodeerror" in text
        or "expecting property name" in text
        or "expecting value" in text
        or "error validating args" in text
    ):
        return False
    return (
        "tool" in text
        or "badrequesterror" in text
        or "openaiexception" in text
        or "validating args" in text
    )


def _event_name(event: Any) -> str:
    return type(event).__name__


def _is_openhands_action_event(event: Any) -> bool:
    return _event_name(event).endswith("ActionEvent")


def _is_openhands_error_event(event: Any) -> bool:
    name = _event_name(event).lower()
    return "errorevent" in name or "conversationerror" in name


def _is_openhands_condensation_event(event: Any) -> bool:
    return "condensation" in _event_name(event).lower()


def _drop_malformed_openhands_tail_events(events: Any) -> int:
    """Remove the malformed action/error event pair that blocks retries.

    OpenHands appends the generated action before validating its JSON tool
    arguments.  If validation fails, retrying the same conversation can replay
    that invalid action and fail before the model gets a chance to repair it.
    The error pair is usually near the tail, but OpenHands may append/visualize
    additional bookkeeping before the exception reaches us.  This removes only
    the latest error event and its immediately preceding generated action within
    a small recent window after a recoverable malformed tool-argument exception;
    normal tool failures and task history stay intact.
    """
    if not isinstance(events, list) or not events:
        return 0

    error_index = None
    for index in range(len(events) - 1, max(-1, len(events) - 12), -1):
        if _is_openhands_error_event(events[index]):
            error_index = index
            break
    if error_index is None:
        return 0

    action_index = None
    for index in range(error_index - 1, max(-1, error_index - 8), -1):
        if _is_openhands_action_event(events[index]):
            action_index = index
            break

    remove_indexes = [error_index]
    if action_index is not None:
        remove_indexes.append(action_index)

    for index in sorted(remove_indexes, reverse=True):
        del events[index]
    return len(remove_indexes)


def _drop_malformed_openhands_state_events(state: Any) -> int:
    """Drop malformed OpenHands events from a state object when possible."""
    events = getattr(state, "events", None)
    if isinstance(events, list):
        return _drop_malformed_openhands_tail_events(events)

    try:
        mutable_events = list(events)
    except Exception:
        return 0

    removed = _drop_malformed_openhands_tail_events(mutable_events)
    if not removed:
        return 0

    try:
        setattr(state, "events", mutable_events)
        return removed
    except Exception:
        logger.warning(
            "Dropped malformed OpenHands events in a copy, but could not "
            "write the repaired event list back to state type %s",
            type(state).__name__,
            exc_info=True,
        )
        return 0


def _recent_openhands_event_names(state: Any, limit: int = 12) -> list[str]:
    events = getattr(state, "events", None)
    try:
        event_list = list(events)
    except Exception:
        return [f"<unreadable events: {type(events).__name__}>"]
    return [_event_name(event) for event in event_list[-limit:]]


def _is_bad_json_like(value: str) -> bool:
    text = value.strip()
    if not text or text[0] not in "{[":
        return False
    try:
        json.loads(text)
        return False
    except json.JSONDecodeError:
        return True


def _shorten(value: str, limit: int = 500) -> str:
    value = value.replace("\x00", "\\0")
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _append_malformed_tool_note(content: Any, notes: list[str]) -> str:
    suffix = _MALFORMED_TOOL_HISTORY_NOTE
    if notes:
        suffix = f"{suffix}\n" + "\n".join(notes)
    if isinstance(content, str) and content.strip():
        return f"{content}\n\n{suffix}"
    return suffix


def _object_to_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(exclude_none=True)
            return dumped if isinstance(dumped, dict) else None
        except TypeError:
            dumped = model_dump()
            return dumped if isinstance(dumped, dict) else None
    return None


def _with_updates(original: Any, updates: dict[str, Any]) -> Any:
    if isinstance(original, dict):
        return updates
    original_dict = _object_to_dict(original) or {}
    object_updates = dict(updates)
    for key in original_dict:
        if key not in object_updates:
            object_updates[key] = None
    model_copy = getattr(original, "model_copy", None)
    if callable(model_copy):
        try:
            return model_copy(update=object_updates)
        except Exception:
            pass
    copy = getattr(original, "copy", None)
    if callable(copy):
        try:
            return copy(update=object_updates)
        except Exception:
            pass
    return updates


def _sanitize_message_dict(message: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Remove only malformed JSON tool-call fragments from OpenHands history.

    Some small/local models occasionally emit truncated tool-call arguments.
    OpenHands records the invalid attempt, then sends that history back to the
    OpenAI-compatible server.  vLLM may try to JSON-decode those malformed
    fragments during chat-template postprocessing and reject the next request
    with HTTP 400.  Valid tool calls are left untouched; malformed ones become
    plain text context so the agent can continue instead of losing the run.
    """
    changed = False
    sanitized = dict(message)
    notes: list[str] = []

    content = sanitized.get("content")
    if isinstance(content, str) and _is_bad_json_like(content):
        notes.append(f"malformed content: {_shorten(content)}")
        sanitized["content"] = _append_malformed_tool_note(None, notes)
        changed = True
        content = sanitized["content"]

    tool_calls = sanitized.get("tool_calls")
    if isinstance(tool_calls, list):
        kept_tool_calls: list[Any] = []
        removed_count = 0
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                kept_tool_calls.append(tool_call)
                continue

            function = tool_call.get("function")
            arguments = None
            tool_name = None
            if isinstance(function, dict):
                arguments = function.get("arguments")
                tool_name = function.get("name")

            if isinstance(arguments, str) and _is_bad_json_like(arguments):
                removed_count += 1
                label = f" for {tool_name}" if tool_name else ""
                notes.append(f"malformed tool_call{label}: {_shorten(arguments)}")
                changed = True
                continue

            kept_tool_calls.append(tool_call)

        if removed_count:
            if kept_tool_calls:
                sanitized["tool_calls"] = kept_tool_calls
            else:
                sanitized.pop("tool_calls", None)
            sanitized["content"] = _append_malformed_tool_note(content, notes)

    return sanitized, changed


def _get_attr_or_item(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _set_attr_or_item(value: Any, key: str, new_value: Any) -> bool:
    if isinstance(value, dict):
        value[key] = new_value
        return True
    try:
        setattr(value, key, new_value)
        return True
    except Exception:
        pass
    model_copy = getattr(value, "model_copy", None)
    if callable(model_copy):
        try:
            patched = model_copy(update={key: new_value})
            if isinstance(value, dict):
                value.update(patched)
            return False
        except Exception:
            return False
    return False


def _replace_attr_or_item(value: Any, key: str, new_value: Any) -> Any:
    """Return ``value`` with one field replaced, copying when possible."""
    if isinstance(value, dict):
        patched = dict(value)
        patched[key] = new_value
        return patched

    original_dict = _object_to_dict(value)
    if original_dict is not None:
        patched_dict = dict(original_dict)
        patched_dict[key] = new_value
        model_validate = getattr(value.__class__, "model_validate", None)
        if callable(model_validate):
            try:
                return model_validate(patched_dict)
            except Exception:
                pass

    model_copy = getattr(value, "model_copy", None)
    if callable(model_copy):
        try:
            return model_copy(update={key: new_value})
        except Exception:
            pass

    copy = getattr(value, "copy", None)
    if callable(copy):
        try:
            return copy(update={key: new_value})
        except Exception:
            pass

    try:
        setattr(value, key, new_value)
        return value
    except Exception:
        pass

    return value


def _first_choice_message(response: Any) -> Any | None:
    choices = _get_attr_or_item(response, "choices")
    if not choices:
        return None
    try:
        choice = choices[0]
    except Exception:
        return None
    return _get_attr_or_item(choice, "message")


def _parse_gemma_text_tool_args(body: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for match in _GEMMA_TEXT_TOOL_ARG_RE.finditer(body):
        value = match.group("value")
        value = value.replace("<|n|>", "\n")
        args[match.group("key")] = value
    return args


def _gemma_text_tool_calls(content: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for index, match in enumerate(_GEMMA_TEXT_TOOL_CALL_RE.finditer(content)):
        name = match.group("name")
        arguments = _parse_gemma_text_tool_args(match.group("body"))
        if not name or not arguments:
            continue
        calls.append(
            {
                "id": f"ipw_gemma_text_tool_{index}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, separators=(",", ":")),
                },
            }
        )
    return calls


def _with_gemma_text_tool_calls(response: Any) -> tuple[Any, bool]:
    """Convert Gemma/OpenHands text tool syntax into OpenAI tool_calls.

    vLLM's ``functiongemma`` parser can accept the model, but in this setup
    Gemma emits OpenHands-looking calls as plain content, for example
    ``<|tool_call>call:terminal{command:<|"|>ls<|"|>}<tool_call|>``.
    OpenHands treats that as a final answer unless LiteLLM returns structured
    ``tool_calls``.  This opt-in bridge preserves other models' normal path and
    only patches responses with no existing structured tool calls.
    """
    message = _first_choice_message(response)
    if message is None:
        return response, False

    existing_tool_calls = _get_attr_or_item(message, "tool_calls")
    if existing_tool_calls:
        return response, False
    content = _get_attr_or_item(message, "content")
    if not isinstance(content, str) or "<|tool_call>call:" not in content:
        return response, False

    tool_calls = _gemma_text_tool_calls(content)
    if not tool_calls:
        return response, False

    patched_content = _GEMMA_TEXT_TOOL_CALL_RE.sub("", content).strip() or None

    choices = _get_attr_or_item(response, "choices")
    try:
        choice = choices[0]
    except Exception:
        patched_message = _replace_attr_or_item(message, "tool_calls", tool_calls)
        patched_message = _replace_attr_or_item(patched_message, "content", patched_content)
        return _replace_attr_or_item(response, "message", patched_message), True

    patched_message = _replace_attr_or_item(message, "tool_calls", tool_calls)
    patched_message = _replace_attr_or_item(patched_message, "content", patched_content)
    patched_choice = _replace_attr_or_item(choice, "message", patched_message)
    patched_choice = _replace_attr_or_item(patched_choice, "finish_reason", "tool_calls")

    patched_choices = list(choices)
    patched_choices[0] = patched_choice
    patched_response = _replace_attr_or_item(response, "choices", patched_choices)
    return patched_response, True


def _complete_heredoc_command_end(command: str) -> int | None:
    """Return the end offset when every heredoc in a shell command is closed."""
    matches = list(_HEREDOC_START_RE.finditer(command))
    if not matches:
        return None

    last_end: int | None = None
    for match in matches:
        marker = match.group("single") or match.group("double") or match.group("bare")
        if not marker:
            return None
        close_match = re.search(rf"(?m)^[ \t]*{re.escape(marker)}[ \t]*$", command[match.end():])
        if close_match is None:
            return None
        last_end = match.end() + close_match.end()
    return last_end


def _terminal_arguments_from_jsonish(raw: str) -> tuple[str | None, str]:
    command_match = _TERMINAL_COMMAND_ARG_RE.search(raw)
    if command_match is None:
        return None, ""

    command_tail = raw[command_match.end():]
    end_match = None
    for match in _TERMINAL_COMMAND_END_RE.finditer(command_tail):
        end_match = match
    if end_match is not None:
        return command_tail[: end_match.start()], command_tail[end_match.start():]

    heredoc_end = _complete_heredoc_command_end(command_tail)
    if heredoc_end is not None:
        remaining = command_tail[heredoc_end:]
        # Only repair an unterminated JSON wrapper if the shell heredoc itself
        # is complete. Never execute a visibly truncated command fragment.
        if not remaining.strip() or remaining.strip() in {'"', '"}', "'"}:
            return command_tail[:heredoc_end], remaining

    # Kimi often renders the terminal call as text containing JSON string
    # escapes (``\n``/``\"``) but omits the final JSON wrapper. Decode only
    # enough to prove the shell heredoc is complete before turning it into a
    # structured tool call.
    for candidate in (command_tail.strip(), command_tail.strip().rstrip('"')):
        if not candidate:
            continue
        try:
            decoded = json.loads(f'"{candidate}"')
        except json.JSONDecodeError:
            continue
        if not isinstance(decoded, str):
            continue
        heredoc_end = _complete_heredoc_command_end(decoded)
        if heredoc_end is None:
            continue
        remaining = decoded[heredoc_end:]
        if not remaining.strip() or remaining.strip() in {'"', '"}', "'"}:
            return decoded[:heredoc_end], remaining

    return None, ""


def _terminal_tool_call_from_arguments(
    raw: str,
    *,
    call_id: str,
) -> dict[str, Any] | None:
    arguments = raw.strip()
    if not arguments:
        return None

    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        repaired, changed = _normalized_terminal_arguments(arguments)
        if not changed:
            return None
        arguments = repaired if isinstance(repaired, str) else json.dumps(repaired, separators=(",", ":"))
    else:
        if not isinstance(parsed, dict) or not isinstance(parsed.get("command"), str):
            return None
        arguments = json.dumps(parsed, separators=(",", ":"))

    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "terminal",
            "arguments": arguments,
        },
    }


def _kimi_text_tool_calls(content: str) -> list[dict[str, Any]]:
    """Convert Kimi textual terminal calls into OpenAI tool_calls.

    This intentionally ignores the harness-generated malformed-history note:
    that note may contain a shortened/truncated command and must not be
    executed. The parser is for fresh model responses that expose a terminal
    call as text instead of a structured tool call.
    """
    if _MALFORMED_TOOL_HISTORY_NOTE in content:
        return []

    candidates: list[str] = []
    stripped = content.strip()
    if stripped.startswith("{") and '"command"' in stripped:
        candidates.append(stripped)
    for match in _KIMI_TEXT_TOOL_PREFIX_RE.finditer(content):
        tail = content[match.end():].strip()
        if tail.startswith("{") and '"command"' in tail:
            candidates.append(tail)

    calls: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        call = _terminal_tool_call_from_arguments(candidate, call_id=f"ipw_kimi_text_tool_{index}")
        if call is not None:
            calls.append(call)
    return calls


def _with_kimi_text_tool_calls(response: Any) -> tuple[Any, bool]:
    message = _first_choice_message(response)
    if message is None:
        return response, False

    existing_tool_calls = _get_attr_or_item(message, "tool_calls")
    if existing_tool_calls:
        return response, False
    content = _get_attr_or_item(message, "content")
    if not isinstance(content, str):
        return response, False

    tool_calls = _kimi_text_tool_calls(content)
    if not tool_calls:
        return response, False

    choices = _get_attr_or_item(response, "choices")
    try:
        choice = choices[0]
    except Exception:
        patched_message = _replace_attr_or_item(message, "tool_calls", tool_calls)
        patched_message = _replace_attr_or_item(patched_message, "content", None)
        return _replace_attr_or_item(response, "message", patched_message), True

    patched_message = _replace_attr_or_item(message, "tool_calls", tool_calls)
    patched_message = _replace_attr_or_item(patched_message, "content", None)
    patched_choice = _replace_attr_or_item(choice, "message", patched_message)
    patched_choice = _replace_attr_or_item(patched_choice, "finish_reason", "tool_calls")

    patched_choices = list(choices)
    patched_choices[0] = patched_choice
    patched_response = _replace_attr_or_item(response, "choices", patched_choices)
    return patched_response, True


def _plain_finish_message(content: str) -> str | None:
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    text = re.sub(r".*</think>", "", text, flags=re.DOTALL).strip()
    matches = list(_FINISH_WITH_MESSAGE_RE.finditer(text))
    match = matches[-1] if matches else None
    if match is None:
        return None
    message = text[match.end():].strip()
    for separator in (
        "\nTokens:",
        "\nObservation ",
        "\nAgent Action ",
        "\nTool:",
        "\nResult:",
        "\n[",
    ):
        index = message.find(separator)
        if index >= 0:
            message = message[:index].strip()
    return message or "Done."


def _finish_text_fallback(content: str) -> str | None:
    finish_message = _plain_finish_message(content)
    if finish_message is not None:
        return finish_message
    if _looks_like_openhands_finish(llm_message=content):
        text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        text = re.sub(r".*</think>", "", text, flags=re.DOTALL).strip()
        return text or "Done."
    return None


def _coerce_glm_tool_arg(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if re.fullmatch(r"-?\d+", stripped):
        try:
            return int(stripped)
        except ValueError:
            return stripped
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", stripped):
        try:
            return float(stripped)
        except ValueError:
            return stripped
    return value


def _glm_xml_text_tool_calls(content: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    starts = list(_GLM_XML_TOOL_START_RE.finditer(content))
    for index, match in enumerate(starts):
        body_start = match.end()
        next_start = starts[index + 1].start() if index + 1 < len(starts) else len(content)
        close_start = content.find("</tool_call>", body_start, next_start)
        body_end = close_start if close_start != -1 else next_start
        body = content[body_start:body_end]
        arguments: dict[str, Any] = {}
        for arg_match in _GLM_XML_TOOL_ARG_RE.finditer(body):
            key = arg_match.group("key").strip()
            if not key:
                continue
            arguments[key] = _coerce_glm_tool_arg(arg_match.group("value"))
        if not arguments:
            continue
        calls.append(
            {
                "id": f"ipw_glm_text_tool_{index}",
                "type": "function",
                "function": {
                    "name": match.group("name"),
                    "arguments": json.dumps(arguments, separators=(",", ":")),
                },
            }
        )
    return calls


def _with_glm_text_tool_calls(response: Any) -> tuple[Any, bool]:
    message = _first_choice_message(response)
    if message is None:
        return response, False

    existing_tool_calls = _get_attr_or_item(message, "tool_calls")
    if existing_tool_calls:
        return response, False
    content = _get_attr_or_item(message, "content")
    if not isinstance(content, str):
        return response, False

    tool_calls = _glm_xml_text_tool_calls(content)
    if tool_calls:
        choices = _get_attr_or_item(response, "choices")
        try:
            choice = choices[0]
        except Exception:
            patched_message = _replace_attr_or_item(message, "tool_calls", tool_calls)
            patched_message = _replace_attr_or_item(patched_message, "content", None)
            return _replace_attr_or_item(response, "message", patched_message), True

        patched_message = _replace_attr_or_item(message, "tool_calls", tool_calls)
        patched_message = _replace_attr_or_item(patched_message, "content", None)
        patched_choice = _replace_attr_or_item(choice, "message", patched_message)
        patched_choice = _replace_attr_or_item(patched_choice, "finish_reason", "tool_calls")
        patched_choices = list(choices)
        patched_choices[0] = patched_choice
        patched_response = _replace_attr_or_item(response, "choices", patched_choices)
        return patched_response, True

    finish_message = _plain_finish_message(content)
    if finish_message is None:
        return response, False

    tool_calls = [
        {
            "id": "ipw_glm_text_finish_0",
            "type": "function",
            "function": {
                "name": "finish",
                "arguments": json.dumps(
                    {"message": finish_message},
                    separators=(",", ":"),
                ),
            },
        }
    ]

    choices = _get_attr_or_item(response, "choices")
    try:
        choice = choices[0]
    except Exception:
        patched_message = _replace_attr_or_item(message, "tool_calls", tool_calls)
        patched_message = _replace_attr_or_item(patched_message, "content", None)
        return _replace_attr_or_item(response, "message", patched_message), True

    patched_message = _replace_attr_or_item(message, "tool_calls", tool_calls)
    patched_message = _replace_attr_or_item(patched_message, "content", None)
    patched_choice = _replace_attr_or_item(choice, "message", patched_message)
    patched_choice = _replace_attr_or_item(patched_choice, "finish_reason", "tool_calls")

    patched_choices = list(choices)
    patched_choices[0] = patched_choice
    patched_response = _replace_attr_or_item(response, "choices", patched_choices)
    return patched_response, True


def _patch_gemma_text_tool_calls(response: Any) -> bool:
    patched_response, changed = _with_gemma_text_tool_calls(response)
    if changed and patched_response is not response:
        _set_attr_or_item(response, "choices", _get_attr_or_item(patched_response, "choices"))
    return changed


def _patch_text_tool_calls_if_enabled(response: Any) -> bool:
    patched_response, changed = _with_openhands_response_compat(response)
    if changed and patched_response is not response:
        _set_attr_or_item(response, "choices", _get_attr_or_item(patched_response, "choices"))
    return changed


def _with_text_tool_calls_if_enabled(response: Any) -> tuple[Any, bool]:
    parser = os.getenv("IPW_OPENHANDS_TEXT_TOOL_CALL_PARSER", "").strip().lower()
    if parser in {"gemma", "functiongemma"}:
        return _with_gemma_text_tool_calls(response)
    if parser in _KIMI_TEXT_TOOL_PARSER_NAMES:
        return _with_kimi_text_tool_calls(response)
    if parser in _GLM_TEXT_TOOL_PARSER_NAMES:
        return _with_glm_text_tool_calls(response)
    return response, False


def _normalize_openhands_tool_name(name: Any) -> tuple[Any, bool]:
    if not isinstance(name, str):
        return name, False
    match = _TOOL_CHANNEL_SUFFIX_RE.match(name)
    if not match:
        return name, False
    return match.group("name"), True


def _repaired_terminal_arguments_json(arguments: Any) -> tuple[Any, bool]:
    """Repair GPT-OSS terminal calls with raw multiline command JSON strings."""
    if not isinstance(arguments, str) or not _is_bad_json_like(arguments):
        return arguments, False
    command, remaining = _terminal_arguments_from_jsonish(arguments)
    if command is None:
        return arguments, False

    repaired: dict[str, Any] = {"command": command}

    timeout_match = _TERMINAL_TIMEOUT_ARG_RE.search(remaining)
    if timeout_match is not None:
        try:
            repaired["timeout"] = int(timeout_match.group("value"))
        except ValueError:
            pass

    for key in ("security_risk", "summary", "working_dir"):
        match = re.search(
            _TERMINAL_STRING_ARG_TEMPLATE.format(key=re.escape(key)),
            remaining,
            re.DOTALL,
        )
        if match is not None:
            repaired[key] = match.group("value")

    for key in ("is_input",):
        match = re.search(_TERMINAL_BOOL_ARG_TEMPLATE.format(key=re.escape(key)), remaining)
        if match is not None:
            repaired[key] = match.group("value") == "true"

    return json.dumps(repaired, separators=(",", ":")), True


def _terminal_timeout_max_seconds() -> int:
    raw_value = os.getenv("IPW_OPENHANDS_TERMINAL_TIMEOUT_MAX_SECONDS", "").strip()
    if not raw_value:
        return _DEFAULT_TERMINAL_TIMEOUT_MAX_SECONDS
    try:
        return max(1, int(float(raw_value)))
    except ValueError:
        return _DEFAULT_TERMINAL_TIMEOUT_MAX_SECONDS


def _coerce_timeout_seconds(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def _with_clamped_terminal_timeout(arguments: Any) -> tuple[Any, bool]:
    """Clamp model-supplied terminal timeouts that look like millisecond values."""
    if isinstance(arguments, dict):
        payload = dict(arguments)
        return_string = False
    elif isinstance(arguments, str):
        try:
            loaded = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments, False
        if not isinstance(loaded, dict):
            return arguments, False
        payload = dict(loaded)
        return_string = True
    else:
        return arguments, False

    timeout = _coerce_timeout_seconds(payload.get("timeout"))
    max_timeout = _terminal_timeout_max_seconds()
    if timeout is None or timeout <= max_timeout:
        return arguments, False

    payload["timeout"] = max_timeout
    if return_string:
        return json.dumps(payload, separators=(",", ":")), True
    return payload, True


def _with_guarded_terminal_command(arguments: Any) -> tuple[Any, bool]:
    """Fail fast on terminal heredocs that would leave the shell waiting."""
    if isinstance(arguments, dict):
        payload = dict(arguments)
        return_string = False
    elif isinstance(arguments, str):
        try:
            loaded = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments, False
        if not isinstance(loaded, dict):
            return arguments, False
        payload = dict(loaded)
        return_string = True
    else:
        return arguments, False

    command = payload.get("command")
    if not isinstance(command, str):
        return arguments, False
    if _HEREDOC_START_RE.search(command) is None:
        return arguments, False
    if _complete_heredoc_command_end(command) is not None:
        return arguments, False

    payload["command"] = _INCOMPLETE_HEREDOC_BLOCK_COMMAND
    payload.pop("is_input", None)
    payload.pop("reset", None)
    payload["timeout"] = min(_coerce_timeout_seconds(payload.get("timeout")) or 5, 5)
    payload["summary"] = "Blocked incomplete heredoc terminal command"
    if return_string:
        return json.dumps(payload, separators=(",", ":")), True
    return payload, True


def _normalized_terminal_arguments(arguments: Any) -> tuple[Any, bool]:
    repaired, repaired_changed = _repaired_terminal_arguments_json(arguments)
    clamped, clamped_changed = _with_clamped_terminal_timeout(repaired)
    guarded, guarded_changed = _with_guarded_terminal_command(clamped)
    return guarded, repaired_changed or clamped_changed or guarded_changed


def _with_repaired_tool_call_arguments(response: Any) -> tuple[Any, bool]:
    choices = _get_attr_or_item(response, "choices")
    if not choices:
        return response, False

    changed = False
    patched_choices = list(choices)
    for choice_idx, choice in enumerate(list(choices)):
        message = _get_attr_or_item(choice, "message")
        if message is None:
            continue
        tool_calls = _get_attr_or_item(message, "tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue

        patched_tool_calls = list(tool_calls)
        message_changed = False
        for call_idx, tool_call in enumerate(list(tool_calls)):
            function = _get_attr_or_item(tool_call, "function")
            if function is None:
                continue
            name, _ = _normalize_openhands_tool_name(_get_attr_or_item(function, "name"))
            if name != "terminal":
                continue
            repaired, repaired_changed = _normalized_terminal_arguments(
                _get_attr_or_item(function, "arguments")
            )
            if not repaired_changed:
                continue
            patched_function = _replace_attr_or_item(function, "arguments", repaired)
            patched_tool_calls[call_idx] = _replace_attr_or_item(
                tool_call,
                "function",
                patched_function,
            )
            message_changed = True

        if not message_changed:
            continue
        patched_message = _replace_attr_or_item(message, "tool_calls", patched_tool_calls)
        patched_choices[choice_idx] = _replace_attr_or_item(choice, "message", patched_message)
        changed = True

    if not changed:
        return response, False
    return _replace_attr_or_item(response, "choices", patched_choices), True


def _with_normalized_tool_call_names(response: Any) -> tuple[Any, bool]:
    """Strip GPT-OSS channel suffixes from structured OpenHands tool names.

    Some OpenAI-compatible GPT-OSS tool parsers can return names such as
    ``terminal<|channel|>commentary``.  OpenHands only registers ``terminal``;
    the suffix is transport syntax, not part of the tool identity.
    """
    choices = _get_attr_or_item(response, "choices")
    if not choices:
        return response, False

    changed = False
    patched_choices = list(choices)
    for choice_idx, choice in enumerate(list(choices)):
        message = _get_attr_or_item(choice, "message")
        if message is None:
            continue
        tool_calls = _get_attr_or_item(message, "tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue

        patched_tool_calls = list(tool_calls)
        message_changed = False
        for call_idx, tool_call in enumerate(list(tool_calls)):
            function = _get_attr_or_item(tool_call, "function")
            if function is None:
                continue
            name = _get_attr_or_item(function, "name")
            normalized, name_changed = _normalize_openhands_tool_name(name)
            if not name_changed:
                continue
            patched_function = _replace_attr_or_item(function, "name", normalized)
            patched_tool_calls[call_idx] = _replace_attr_or_item(
                tool_call,
                "function",
                patched_function,
            )
            message_changed = True

        if not message_changed:
            continue
        patched_message = _replace_attr_or_item(message, "tool_calls", patched_tool_calls)
        patched_choices[choice_idx] = _replace_attr_or_item(choice, "message", patched_message)
        changed = True

    if not changed:
        return response, False
    return _replace_attr_or_item(response, "choices", patched_choices), True


def _with_openhands_response_compat(response: Any) -> tuple[Any, bool]:
    response, text_patched = _with_text_tool_calls_if_enabled(response)
    response, args_patched = _with_repaired_tool_call_arguments(response)
    response, name_patched = _with_normalized_tool_call_names(response)
    return response, text_patched or args_patched or name_patched


@contextmanager
def _litellm_text_tool_call_patch() -> Iterator[None]:
    """Patch LiteLLM responses for models that emit tool calls as text.

    OpenHands calls LiteLLM internally, so wrapping only the OpenHands LLM
    object's public methods is not enough for every SDK version.  This opt-in
    context manager patches the LiteLLM boundary that OpenHands uses and leaves
    all normal structured tool-call responses unchanged.
    """
    parser = os.getenv("IPW_OPENHANDS_TEXT_TOOL_CALL_PARSER", "").strip().lower()
    if parser not in {
        "gemma",
        "functiongemma",
        *_KIMI_TEXT_TOOL_PARSER_NAMES,
        *_GLM_TEXT_TOOL_PARSER_NAMES,
    }:
        yield
        return

    try:
        import litellm
    except Exception:
        yield
        return

    originals: list[tuple[Any, str, Callable[..., Any]]] = []

    def _patch_attr(name: str) -> None:
        original = getattr(litellm, name, None)
        if not callable(original):
            return

        if inspect.iscoroutinefunction(original):
            async def _async_wrapped(*args: Any, **kwargs: Any) -> Any:
                response = await original(*args, **kwargs)
                response, patched = _with_openhands_response_compat(response)
                if patched:
                    logger.warning("Patched OpenHands tool-call compatibility from LiteLLM response")
                return response

            replacement: Callable[..., Any] = _async_wrapped
        else:
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                response = original(*args, **kwargs)
                response, patched = _with_openhands_response_compat(response)
                if patched:
                    logger.warning("Patched OpenHands tool-call compatibility from LiteLLM response")
                return response

            replacement = _wrapped

        originals.append((litellm, name, original))
        setattr(litellm, name, replacement)

    _patch_attr("completion")
    _patch_attr("acompletion")

    try:
        yield
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)


def _sanitize_message(message: Any) -> tuple[Any, bool]:
    message_dict = _object_to_dict(message)
    if message_dict is None:
        return message, False
    sanitized, changed = _sanitize_message_dict(message_dict)
    if not changed:
        return message, False
    return _with_updates(message, sanitized), True


def _sanitize_litellm_messages(value: Any) -> tuple[Any, bool]:
    if not isinstance(value, list):
        return value, False

    changed = False
    sanitized_messages: list[Any] = []
    for message in value:
        sanitized, message_changed = _sanitize_message(message)
        sanitized_messages.append(sanitized)
        changed = changed or message_changed
    return (sanitized_messages if changed else value), changed


def _sanitize_llm_call_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any], bool]:
    changed = False
    new_args = args
    new_kwargs = kwargs

    if "messages" in kwargs:
        sanitized, changed = _sanitize_litellm_messages(kwargs["messages"])
        if changed:
            new_kwargs = dict(kwargs)
            new_kwargs["messages"] = sanitized
        return new_args, new_kwargs, changed

    for index, arg in enumerate(args):
        sanitized, arg_changed = _sanitize_litellm_messages(arg)
        if arg_changed:
            if not changed:
                new_args = tuple(args)
            mutable_args = list(new_args)
            mutable_args[index] = sanitized
            new_args = tuple(mutable_args)
            changed = True

    return new_args, new_kwargs, changed


_MCP_TOOL_TYPES: tuple[type, type, type, type] | None = None


def _get_mcp_tool_types() -> tuple[type, type, type, type]:
    """Create stable OpenHands MCP tool classes without requiring OpenHands at import."""
    global _MCP_TOOL_TYPES
    if _MCP_TOOL_TYPES is not None:
        return _MCP_TOOL_TYPES

    from openhands.sdk.tool.schema import Action, Observation
    from openhands.sdk.tool.tool import ToolDefinition, ToolExecutor

    class IPWMCPAction(Action):
        query: Optional[str] = None
        command: Optional[str] = None
        path: Optional[str] = None
        content: Optional[str] = None
        mode: Optional[str] = None
        old_str: Optional[str] = None
        new_str: Optional[str] = None
        timeout: Optional[int] = None
        working_dir: Optional[str] = None

    class IPWMCPObservation(Observation):
        pass

    class MCPToolExecutor(ToolExecutor):
        def __init__(self, mcp_server: Any) -> None:
            self._server = mcp_server

        def __call__(self, action: IPWMCPAction, conversation: Any = None) -> IPWMCPObservation:
            payload = action.model_dump(exclude_none=True)
            prompt = (
                payload.pop("query", None)
                or payload.pop("command", None)
                or payload.pop("path", None)
                or payload.pop("content", None)
                or ""
            )
            result = self._server.execute(str(prompt), **payload)
            content = result.content if hasattr(result, "content") else str(result)
            return IPWMCPObservation.from_text(text=content)

    class MCPToolDefinition(ToolDefinition[IPWMCPAction, IPWMCPObservation]):
        @classmethod
        def create(cls, *args: Any, **kwargs: Any) -> Sequence["MCPToolDefinition"]:
            return []

    for cls in (IPWMCPAction, IPWMCPObservation, MCPToolExecutor, MCPToolDefinition):
        cls.__module__ = __name__
        cls.__qualname__ = cls.__name__
        globals()[cls.__name__] = cls

    _MCP_TOOL_TYPES = (IPWMCPAction, IPWMCPObservation, MCPToolExecutor, MCPToolDefinition)
    return _MCP_TOOL_TYPES


def _instrument_openhands_llm(
    model: Any,
    recorder: Optional["EventRecorder"],
) -> Any:
    """Record every OpenHands LLM call while preserving the LLM instance type."""
    if recorder is None or getattr(model, "_ipw_instrumented", False) is True:
        return model

    def _record_event(event_type: str, **metadata: Any) -> None:
        recorder.record(event_type, **metadata)

    def _record_lm_end(
        pre_input: int | None,
        pre_output: int | None,
        pre_cost: float | None,
    ) -> None:
        post_input, post_output, post_cost = _usage_counts(model)
        metadata: dict[str, Any] = {"model": str(model)}
        input_delta = (
            post_input - pre_input
            if post_input is not None and pre_input is not None
            else None
        )
        output_delta = (
            post_output - pre_output
            if post_output is not None and pre_output is not None
            else None
        )
        cost_delta = (
            post_cost - pre_cost
            if post_cost is not None and pre_cost is not None
            else None
        )
        if input_delta is not None and input_delta > 0:
            metadata["prompt_tokens"] = input_delta
        if output_delta is not None and output_delta > 0:
            metadata["completion_tokens"] = output_delta
        if cost_delta:
            metadata["cost_usd"] = cost_delta
        _record_event("lm_inference_end", **metadata)

    def _wrap_call(func: Any, *, patch_only: bool = False) -> Any:
        if inspect.iscoroutinefunction(func):
            async def _async_wrapped(*args: Any, **kwargs: Any) -> Any:
                args, kwargs, sanitized = _sanitize_llm_call_args(args, kwargs)
                if sanitized:
                    logger.warning("Sanitized malformed OpenHands tool-call history")
                pre_input, pre_output, pre_cost = _usage_counts(model)
                if not patch_only:
                    _record_event("lm_inference_start", model=str(model))
                try:
                    response = await func(*args, **kwargs)
                    response, patched = _with_openhands_response_compat(response)
                    if patched:
                        logger.warning("Patched OpenHands tool-call compatibility")
                    return response
                finally:
                    if not patch_only:
                        _record_lm_end(pre_input, pre_output, pre_cost)

            return _async_wrapped

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            args, kwargs, sanitized = _sanitize_llm_call_args(args, kwargs)
            if sanitized:
                logger.warning("Sanitized malformed OpenHands tool-call history")
            pre_input, pre_output, pre_cost = _usage_counts(model)
            if not patch_only:
                _record_event("lm_inference_start", model=str(model))
            try:
                response = func(*args, **kwargs)
                response, patched = _with_openhands_response_compat(response)
                if patched:
                    logger.warning("Patched OpenHands tool-call compatibility")
                return response
            finally:
                if not patch_only:
                    _record_lm_end(pre_input, pre_output, pre_cost)

        return _wrapped

    for name in _CALL_METHODS:
        attr = getattr(model, name, None)
        if callable(attr):
            object.__setattr__(model, name, _wrap_call(attr, patch_only=name.startswith("_")))

    object.__setattr__(model, "_ipw_instrumented", True)
    return model


def _register_mcp_tools(mcp_tools: Dict[str, Any]) -> list:
    """Register MCP servers as OpenHands tools and return Tool specs.

    Each MCP server is wrapped as a ToolDefinition and registered in the
    OpenHands tool registry so the Agent can resolve it by name.

    Args:
        mcp_tools: Mapping of tool names to custom MCP server instances.

    Returns:
        List of Tool specs that can be passed to Agent(tools=...).
    """
    from openhands.sdk import Tool, register_tool
    from openhands.sdk.tool.tool import ToolAnnotations, ToolDefinition

    (
        IPWMCPAction,
        IPWMCPObservation,
        MCPToolExecutor,
        MCPToolDefinition,
    ) = _get_mcp_tool_types()

    tool_specs: list = []

    for name, server in mcp_tools.items():
        oh_name = getattr(server, "openhands_name", None) or (
            name if name in {"bash", "str_replace_editor"} else f"mcp_{name}"
        )

        spec = getattr(server, "_spec", None)
        description = (
            spec.description
            if spec and hasattr(spec, "description")
            else f"Execute the {name} tool"
        )

        executor = MCPToolExecutor(server)

        def _make_factory(_oh_name: str, _desc: str, _executor: MCPToolExecutor):
            def factory(conv_state: Any = None, **kwargs: Any) -> Sequence[ToolDefinition]:
                tool_def = MCPToolDefinition(
                    description=_desc,
                    action_type=IPWMCPAction,
                    observation_type=IPWMCPObservation,
                    executor=_executor,
                    annotations=ToolAnnotations(title=_oh_name),
                )
                object.__setattr__(tool_def, "name", _oh_name)
                return [tool_def]

            return factory

        register_tool(oh_name, _make_factory(oh_name, description, executor))
        tool_specs.append(Tool(name=oh_name))
        logger.info(f"Registered OpenHands MCP tool: {oh_name}")

    return tool_specs


@AgentRegistry.register("openhands")
class OpenHands(BaseAgent):
    """OpenHands agent using the OpenHands SDK with energy telemetry."""

    DEFAULT_MAX_TURNS = 20

    DEFAULT_INSTRUCTIONS = (
        "You operate autonomously with two tools: `terminal` (bash) and "
        "`file_editor`. NEVER ask the user questions — make every decision "
        "yourself. To produce a deliverable, write a script with "
        "`file_editor`, RUN it via `terminal` (e.g. `python script.py`), "
        "and verify the output files exist with `ls`. Saving a script "
        "without running it accomplishes nothing. Install any missing "
        "libraries via `terminal` (e.g. `pip install openpyxl pdfplumber`). "
        "Call `finish` exactly once, only after the deliverable files are "
        "on disk."
    )

    def __init__(
        self,
        model: Any,
        tools: list | None = None,
        mcp_tools: Optional[Dict[str, Any]] = None,
        event_recorder: Optional["EventRecorder"] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        openhands_tool_names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenHands agent.

        Args:
            model: The LLM model instance to use.
            tools: List of OpenHands Tool specs.
            mcp_tools: Optional mapping of tool names to custom MCP server instances.
            event_recorder: Optional EventRecorder for per-action energy telemetry.
            max_turns: Maximum iterations per run.
            openhands_tool_names: Optional subset of OpenHands built-in tools to expose.
            **kwargs: Additional keyword arguments passed to the Agent constructor.
        """
        super().__init__(event_recorder=event_recorder)

        # Lazy imports: openhands-sdk is optional
        try:
            from openhands.sdk import (  # noqa: F401
                Agent,
                Event,
                LLMConvertibleEvent,
                LLMSummarizingCondenser,
                LocalConversation,
                Tool,
            )
            from openhands.sdk.event.llm_convertible.action import ActionEvent
            from openhands.sdk.event.llm_convertible.observation import ObservationEvent
        except ImportError:
            raise ImportError(
                "openhands-sdk package is required for OpenHands agent. "
                "Install with: pip install openhands-sdk"
            )

        # Importing openhands.tools registers the standard tools (terminal,
        # file_editor, task_tracker, etc.) in the SDK's tool registry. This
        # is optional — if it isn't installed the agent will fall back to
        # the SDK builtins (finish/think only).
        _have_openhands_tools = False
        try:
            import openhands.tools  # noqa: F401
            _have_openhands_tools = True
        except ImportError:
            logger.warning(
                "openhands-tools not installed; agent will only have finish/"
                "think available. Install with: pip install openhands-tools"
            )

        self.model = model
        self._instrumented_model = _instrument_openhands_llm(model, event_recorder)
        self.tools = tools
        self._mcp_tools = mcp_tools or {}
        self._pending_tool: Optional[str] = None
        self._tool_names_used: List[str] = []
        self._num_turns: int = 0
        self._workspace: str = os.getcwd()
        self._max_turns = max_turns

        # Store references for use in callbacks
        self._ActionEvent = ActionEvent
        self._ObservationEvent = ObservationEvent
        self._LLMConvertibleEvent = LLMConvertibleEvent

        # Store SDK classes for lazy conversation creation
        self._LocalConversation = LocalConversation
        self._LLMSummarizingCondenser = LLMSummarizingCondenser

        # Context condenser
        condenser = LLMSummarizingCondenser(
            llm=self._instrumented_model,
            max_tokens=24000,
            keep_first=2,
        )

        # Build agent_kwargs up front but DEFER Agent construction to per-run.
        #
        # Why: openhands.sdk.AgentBase._initialize() builds the tools map
        # exactly once and refuses to re-init. The first conversation's
        # workspace becomes baked into the TerminalExecutor and reused for
        # every subsequent conversation, so task N+1's `python build.py`
        # writes outputs into task N's workspace. Building a fresh Agent
        # per run() call forces _initialize to re-run with the current
        # conversation's state (and the correct per-task workspace).
        agent_kwargs: dict[str, Any] = {
            "llm": self._instrumented_model,
            "condenser": condenser,
        }

        if tools:
            agent_kwargs["tools"] = tools
        elif mcp_tools:
            extra_tool_specs = _register_mcp_tools(mcp_tools)
            agent_kwargs["tools"] = extra_tool_specs
        elif _have_openhands_tools:
            # Default tools: terminal + file_editor only. Skipping
            # task_tracker so small models don't spend turns building todo
            # lists. terminal_type="subprocess" avoids the tmux pane pool
            # (also a source of cross-conversation state leakage).
            requested_tool_names = list(openhands_tool_names or ["terminal", "file_editor"])
            allowed_tool_names = {"terminal", "file_editor"}
            unknown_tool_names = sorted(set(requested_tool_names) - allowed_tool_names)
            if unknown_tool_names:
                raise ValueError(f"Unsupported OpenHands tool names: {unknown_tool_names}")
            builtins = {
                "terminal": Tool(name="terminal", params={"terminal_type": "subprocess"}),
                "file_editor": Tool(name="file_editor"),
            }
            agent_kwargs["tools"] = [builtins[name] for name in requested_tool_names]

        reserved_agent_kwargs = set(agent_kwargs)
        conflicting_agent_kwargs = sorted(reserved_agent_kwargs.intersection(kwargs))
        if conflicting_agent_kwargs:
            raise ValueError(
                "OpenHands agent kwargs cannot override managed arguments: "
                + ", ".join(conflicting_agent_kwargs)
            )
        agent_kwargs.update(kwargs)

        self._Agent = Agent
        self._agent_kwargs = agent_kwargs
        self.agent: Optional[Any] = None  # rebuilt in each run()
        self.conversation: Optional[Any] = None
        self.current_result = ""
        self._finish_observed = threading.Event()
        self._task_metadata: Optional[MutableMapping[str, Any]] = None

    def _instrumented_callback(self, event: Any) -> None:
        """Instrumented callback that emits telemetry events for tool calls."""
        if _looks_like_openhands_finish(event):
            self._finish_observed.set()

        if isinstance(event, self._LLMConvertibleEvent):
            self.current_result = event.to_llm_message()
            if _looks_like_openhands_finish(event, llm_message=self.current_result):
                self._finish_observed.set()

        if isinstance(event, self._ActionEvent):
            tool_name = event.tool_name
            self._pending_tool = tool_name
            self._tool_names_used.append(tool_name)
            self._num_turns += 1
            self._record_event("tool_call_start", tool=tool_name)
            if _looks_like_openhands_finish(event, tool_name=tool_name):
                self._finish_observed.set()
        elif isinstance(event, self._ObservationEvent):
            tool_name = self._pending_tool or event.tool_name
            self._record_event("tool_call_end", tool=tool_name)
            self._pending_tool = None

    def set_task_metadata(self, metadata: MutableMapping[str, Any]) -> None:
        self._task_metadata = metadata
        # GDPval: tell the rubric judge where the agent's deliverables live.
        # OpenHands writes outputs directly into the workspace root (not a
        # subdirectory), so point the judge at the workspace itself.
        if metadata is not None and self._workspace:
            metadata.setdefault("gdpval_outputs_dir", self._workspace)
        # GDPval: materialize reference files into the workspace so the agent
        # can read them via its file tools.
        inputs_dir = metadata.get("gdpval_inputs_dir") if metadata else None
        if (os.getenv("IPW_REQUIRE_GDPVAL_REFERENCES") == "1"
                and metadata and metadata.get("reference_file_hf_uris") and not inputs_dir):
            raise GDPValReferenceStagingError("GDPVal reference inputs directory is missing")
        if inputs_dir:
            try:
                _stage_gdpval_inputs_for_workspace(metadata, self._workspace)
            except Exception as exc:
                if os.getenv("IPW_REQUIRE_GDPVAL_REFERENCES") == "1":
                    raise GDPValReferenceStagingError("Failed to stage GDPVal references") from exc
                logger.warning("Failed to stage gdpval inputs", exc_info=True)

    def set_workspace(self, workspace_path: str) -> None:
        """Set the workspace directory for the next agent run."""
        self._workspace = workspace_path
        for server in self._mcp_tools.values():
            if hasattr(server, "working_dir"):
                server.working_dir = workspace_path
            if getattr(server, "_ipw_dynamic_allowed_dirs", False) and hasattr(
                server, "allowed_dirs"
            ):
                server.allowed_dirs = [workspace_path]

    def _create_conversation(self) -> Any:
        """Create a fresh Agent + LocalConversation for the next run.

        Building a fresh Agent forces openhands.sdk.AgentBase._initialize()
        to re-run with the current conversation's state, which in turn
        creates a fresh TerminalExecutor bound to this task's workspace.
        Without this, the executor's working_dir is frozen at the first
        task's workspace and every subsequent task's outputs land there.
        """
        self.agent = self._Agent(**self._agent_kwargs)
        return self._LocalConversation(
            agent=self.agent,
            callbacks=[self._instrumented_callback],
            workspace=self._workspace,
            max_iteration_per_run=self._max_turns,
        )

    @staticmethod
    def _extract_text(message: Any) -> str:
        """Extract plain text from an OpenHands Message or fallback to str()."""
        if hasattr(message, "content") and isinstance(message.content, (list, tuple)):
            try:
                from openhands.sdk.llm.message import TextContent
                parts = [
                    item.text for item in message.content if isinstance(item, TextContent)
                ]
                if parts:
                    text = "\n".join(parts)
                else:
                    text = str(message)
            except ImportError:
                text = str(message)
        elif isinstance(message, str):
            text = message
        else:
            text = str(message)

        # Strip <think>...</think> blocks (extended thinking output)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r".*</think>", "", text, flags=re.DOTALL)
        return text.strip()

    def _run_conversation_with_tool_arg_repair(self) -> None:
        """Run OpenHands, recovering from malformed generated tool arguments."""
        attempts = int(os.environ.get("IPW_OPENHANDS_TOOL_ARG_REPAIR_ATTEMPTS", "2"))
        for attempt in range(attempts + 1):
            try:
                self._run_conversation_until_return_or_finish()
                return
            except Exception as exc:
                if attempt >= attempts or not _is_recoverable_tool_arg_error(exc):
                    raise
                logger.warning(
                    "Recovering from malformed OpenHands tool arguments "
                    "(attempt %d/%d): %s",
                    attempt + 1,
                    attempts,
                    exc,
                )
                state = getattr(self.conversation, "state", None)
                removed_events = _drop_malformed_openhands_state_events(state)
                if removed_events:
                    logger.warning(
                        "Dropped %d malformed OpenHands state event(s) before retry",
                        removed_events,
                    )
                else:
                    logger.warning(
                        "No malformed OpenHands state events were dropped before "
                        "retry; recent event types: %s",
                        _recent_openhands_event_names(state),
                    )
                self._record_event(
                    "tool_call_end",
                    tool="malformed_tool_args",
                    error=str(exc),
                )
                self.conversation.send_message(_MALFORMED_TOOL_RETRY_PROMPT)

    def _run_conversation_until_return_or_finish(self) -> None:
        """Run the SDK conversation, returning once a FinishTool action is seen.

        Some OpenHands SDK versions can keep ``LocalConversation.run()`` alive
        after the finish action is emitted. Raising from a callback turns that
        valid finish into an SDK failure, so run the SDK in a daemon thread and
        let bounded conversation cleanup stop any tail work after we return.
        """
        assert self.conversation is not None
        errors: list[BaseException] = []
        completed = threading.Event()

        def _run() -> None:
            try:
                self.conversation.run()
            except BaseException as exc:  # pragma: no cover - re-raised below
                errors.append(exc)
            finally:
                completed.set()

        thread = threading.Thread(
            target=_run,
            name="openhands-conversation-run",
            daemon=True,
        )
        finish_log_path, finish_log_offset = _finish_log_watch_start()
        thread.start()
        early_return = _env_enabled("IPW_OPENHANDS_EARLY_RETURN_ON_FINISH", default=True)
        valid_deliverable_early_return = _env_enabled(
            "IPW_OPENHANDS_EARLY_RETURN_ON_VALID_GDPVAL_DELIVERABLE",
            default=False,
        )
        try:
            valid_deliverable_poll_s = max(
                0.25,
                float(os.environ.get("IPW_OPENHANDS_VALID_DELIVERABLE_POLL_SECONDS", "5")),
            )
        except ValueError:
            valid_deliverable_poll_s = 5.0
        try:
            valid_deliverable_settle_s = max(
                0.0,
                float(
                    os.environ.get(
                        "IPW_OPENHANDS_VALID_DELIVERABLE_SETTLE_SECONDS",
                        "300",
                    )
                ),
            )
        except ValueError:
            valid_deliverable_settle_s = 300.0
        try:
            valid_deliverable_min_turns = max(
                0,
                int(os.environ.get("IPW_OPENHANDS_VALID_DELIVERABLE_MIN_TURNS", "0")),
            )
        except ValueError:
            valid_deliverable_min_turns = 0
        next_valid_deliverable_check = 0.0
        valid_deliverable_signature: tuple[tuple[str, int, int], ...] | None = None
        valid_deliverable_names = ""
        finish_seen = False
        max_iteration_seen = False

        def _early_return_close_timeout() -> float:
            try:
                return float(
                    os.environ.get(
                        "IPW_OPENHANDS_EARLY_RETURN_CLOSE_TIMEOUT_SECONDS",
                        os.environ.get("IPW_OPENHANDS_CLOSE_TIMEOUT_SECONDS", "10"),
                    )
                )
            except ValueError:
                return 10.0

        def _return_after_requesting_stop(reason: str) -> None:
            stop_requested = _request_conversation_run_stop(self.conversation, reason)
            if not stop_requested:
                if hasattr(self.conversation, "close"):
                    logger.warning(
                        "OpenHands run-loop pause unavailable after early return "
                        "(%s); closing conversation executors now",
                        reason,
                    )
                    try:
                        _close_conversation_with_timeout(
                            self.conversation,
                            _early_return_close_timeout(),
                        )
                    except Exception as exc:
                        logger.warning(
                            "Error closing OpenHands conversation after early "
                            "return (%s): %s",
                            reason,
                            exc,
                        )
                    completed.wait(timeout=0.5)
                return
            try:
                join_timeout_s = float(
                    os.environ.get(
                        "IPW_OPENHANDS_EARLY_RETURN_JOIN_TIMEOUT_SECONDS",
                        os.environ.get("IPW_OPENHANDS_CLOSE_TIMEOUT_SECONDS", "10"),
                    )
                )
            except ValueError:
                join_timeout_s = 10.0
            if not completed.wait(timeout=max(0.0, join_timeout_s)):
                logger.warning(
                    "OpenHands run thread still alive %.1fs after early return (%s); "
                    "closing conversation executors now",
                    join_timeout_s,
                    reason,
                )
                try:
                    _close_conversation_with_timeout(
                        self.conversation,
                        _early_return_close_timeout(),
                    )
                except Exception as exc:
                    logger.warning(
                        "Error closing OpenHands conversation after early "
                        "return (%s): %s",
                        reason,
                        exc,
                    )
                completed.wait(timeout=0.5)
            elif errors:
                logger.warning(
                    "OpenHands run thread exited with error after early return (%s): %s",
                    reason,
                    errors[0],
                )

        while True:
            if self._finish_observed.is_set():
                if early_return:
                    return _return_after_requesting_stop("finish_observed")
                if not finish_seen:
                    logger.info(
                        "OpenHands finish observed; waiting for conversation.run() "
                        "to drain because IPW_OPENHANDS_EARLY_RETURN_ON_FINISH=0"
                    )
                    finish_seen = True
            if _conversation_state_has_finish(self.conversation):
                self._finish_observed.set()
                if early_return:
                    return _return_after_requesting_stop("finish_event")
                if not finish_seen:
                    logger.info(
                        "OpenHands finish event observed; waiting for "
                        "conversation.run() to drain because "
                        "IPW_OPENHANDS_EARLY_RETURN_ON_FINISH=0"
                    )
                    finish_seen = True
            current_finish = _finish_text_fallback(
                self._extract_text(getattr(self, "current_result", ""))
            )
            if current_finish:
                self._finish_observed.set()
                if early_return:
                    return _return_after_requesting_stop("finish_text")
                if not finish_seen:
                    logger.info(
                        "OpenHands finish text observed; waiting for "
                        "conversation.run() to drain because "
                        "IPW_OPENHANDS_EARLY_RETURN_ON_FINISH=0"
                    )
                    finish_seen = True
            if _finish_log_contains_finish(finish_log_path, finish_log_offset):
                self._finish_observed.set()
                if early_return:
                    return _return_after_requesting_stop("finish_log")
                if not finish_seen:
                    logger.info(
                        "OpenHands finish log observed; waiting for "
                        "conversation.run() to drain because "
                        "IPW_OPENHANDS_EARLY_RETURN_ON_FINISH=0"
                    )
                    finish_seen = True
            if _conversation_state_has_max_iteration_exhausted(self.conversation):
                if early_return:
                    logger.warning(
                        "OpenHands reached max_iteration_per_run; returning current "
                        "result so GDPval validation/repair can continue"
                    )
                    return _return_after_requesting_stop("max_iteration_state")
                if not max_iteration_seen:
                    logger.warning(
                        "OpenHands reached max_iteration_per_run; waiting for "
                        "conversation.run() to drain because "
                        "IPW_OPENHANDS_EARLY_RETURN_ON_FINISH=0"
                    )
                    max_iteration_seen = True
            if _finish_log_contains_max_iteration(finish_log_path, finish_log_offset):
                if early_return:
                    logger.warning(
                        "OpenHands log reported max_iteration_per_run; returning "
                        "current result so GDPval validation/repair can continue"
                    )
                    return _return_after_requesting_stop("max_iteration_log")
                if not max_iteration_seen:
                    logger.warning(
                        "OpenHands log reported max_iteration_per_run; waiting for "
                        "conversation.run() to drain because "
                        "IPW_OPENHANDS_EARLY_RETURN_ON_FINISH=0"
                    )
                    max_iteration_seen = True
            if (
                valid_deliverable_early_return
                and self._num_turns >= valid_deliverable_min_turns
            ):
                now = time.time()
                if now >= next_valid_deliverable_check:
                    next_valid_deliverable_check = now + valid_deliverable_poll_s
                    snapshot = _gdpval_valid_deliverable_snapshot(
                        self._task_metadata,
                        getattr(self, "_current_problem", ""),
                    )
                    if snapshot is None:
                        valid_deliverable_signature = None
                        valid_deliverable_names = ""
                    else:
                        signature, names, newest_mtime_s = snapshot
                        if signature != valid_deliverable_signature:
                            valid_deliverable_signature = signature
                            valid_deliverable_names = names
                            logger.warning(
                                "Valid GDPval deliverable observed; waiting %.1fs "
                                "of file mtime stability before early return: %s",
                                valid_deliverable_settle_s,
                                names,
                            )
                        file_age_s = now - newest_mtime_s
                        if file_age_s >= valid_deliverable_settle_s:
                            self.current_result = (
                                "Valid GDPval deliverable(s) produced: "
                                f"{valid_deliverable_names or names}"
                            )
                            self._early_return_result = self.current_result
                            self._finish_observed.set()
                            logger.warning(
                                "Early-returning OpenHands run after valid GDPval "
                                "deliverable stayed unchanged for %.1fs: %s",
                                file_age_s,
                                valid_deliverable_names or names,
                            )
                            # A valid GDPval artifact is already on disk. Avoid
                            # the pause request here because it can block behind
                            # an in-flight model/tool request and prevent the
                            # runner from reaching validation/scoring.
                            return
            if completed.wait(timeout=0.05):
                if errors:
                    raise errors[0]
                return

    def run(self, input: str, **kwargs: Any) -> AgentRunResult:
        """Run the OpenHands agent with telemetry.

        Runs locally via ``LocalConversation``.

        Args:
            input: The input message or prompt for the agent.
            **kwargs: Additional keyword arguments.

        Returns:
            AgentRunResult with content, tool_names_used, and num_turns.
        """
        from openhands.sdk.conversation.response_utils import get_agent_final_response

        # Reset per-run tracking
        self._tool_names_used = []
        self._num_turns = 0
        self._finish_observed = threading.Event()
        self._current_problem = input
        self._early_return_result = ""

        # Snapshot LLM token metrics before this run to compute per-query delta
        _pre_input, _pre_output, _pre_cost = _usage_counts(self.model)
        finish_log_fallback: str | None = None

        try:
            with _litellm_text_tool_call_patch():
                # OpenHands/tools dynamically create Pydantic action classes
                # while initializing tool definitions. That registry is process
                # global in the SDK, so concurrent Agent/LocalConversation
                # setup can trip duplicate-class validation before the first
                # model turn. Keep only this schema-construction phase serial;
                # the actual agent loop below still runs concurrently.
                with _OPENHANDS_CONVERSATION_INIT_LOCK:
                    self.conversation = self._create_conversation()
                    self.conversation.send_message(input)
                self._run_conversation_with_tool_arg_repair()
                _trace_openhands_return("OpenHands conversation loop returned")

            result = str(getattr(self, "_early_return_result", "") or "").strip()
            if result:
                _trace_openhands_return("Using explicit OpenHands early-return result")
            if not result:
                _trace_openhands_return("Calling get_agent_final_response()")
                result = get_agent_final_response(self.conversation.state.events)
                _trace_openhands_return("get_agent_final_response() returned")
            if not result:
                fallback = _finish_text_fallback(self._extract_text(self.current_result))
                if fallback:
                    result = fallback
                    logger.info(
                        "Using finish-text fallback after OpenHands "
                        "returned no structured final response"
                    )
            if not result:
                # If the SDK did not emit a finish event, ask for a final answer
                # without lowering the iteration budget. Query/client timeouts are
                # the operational guardrail for long-running model behavior.
                final_nudge_mode = os.environ.get(
                    "IPW_OPENHANDS_FINAL_NUDGE_MODE", "run"
                ).strip().lower()
                if final_nudge_mode in _FINAL_NUDGE_SKIP_VALUES:
                    logger.info(
                        "No FinishTool call detected; final-answer nudge disabled "
                        "by IPW_OPENHANDS_FINAL_NUDGE_MODE=%s",
                        final_nudge_mode,
                    )
                else:
                    logger.info("No FinishTool call detected, sending final-answer nudge")
                    saved_limit = self.conversation.max_iteration_per_run
                    self.conversation.max_iteration_per_run = saved_limit + 1
                    nudge_log_path, nudge_log_offset = _finish_log_watch_start()
                    self.conversation.send_message(
                        "Submit your final answer now with the finish tool. "
                        "Do not call terminal or file_editor. If output files "
                        "already exist, finish with a concise list of their paths."
                    )
                    finish_log_fallback = _finish_log_message(
                        nudge_log_path, nudge_log_offset
                    )
                    if finish_log_fallback:
                        self._finish_observed.set()
                    else:
                        self._run_conversation_with_tool_arg_repair()
                    self.conversation.max_iteration_per_run = saved_limit
                    result = get_agent_final_response(self.conversation.state.events)
                    if not result and self._finish_observed.is_set():
                        fallback = _finish_text_fallback(
                            self._extract_text(self.current_result)
                        )
                        result = fallback or finish_log_fallback

            if not result:
                result = self._extract_text(self.current_result)
                logger.warning("get_agent_final_response() returned empty after nudge, using callback fallback")

            result = self._extract_text(result)
            self.current_result = ""
            _trace_openhands_return("OpenHands result text normalized")

            # Extract per-query token usage as delta from pre-run snapshot
            post_input, post_output, post_cost = _usage_counts(self.model)
            input_tokens = (
                post_input - _pre_input
                if post_input is not None and _pre_input is not None
                else None
            )
            output_tokens = (
                post_output - _pre_output
                if post_output is not None and _pre_output is not None
                else None
            )
            cost_usd = (
                post_cost - _pre_cost
                if post_cost is not None and _pre_cost is not None
                else None
            )
            token_source = (
                "openhands_litellm_metrics"
                if (input_tokens is not None and input_tokens > 0)
                or (output_tokens is not None and output_tokens > 0)
                else "missing"
            )

            _trace_openhands_return("Returning AgentRunResult from OpenHands")
            return AgentRunResult(
                content=result,
                tool_calls_attempted=len(self._tool_names_used),
                tool_calls_succeeded=len(self._tool_names_used),
                tool_names_used=list(self._tool_names_used),
                num_turns=self._num_turns,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                metadata={"token_source": token_source},
            )
        finally:
            try:
                close_timeout_s = float(
                    os.environ.get("IPW_OPENHANDS_CLOSE_TIMEOUT_SECONDS", "10")
                )
                _close_conversation_with_timeout(self.conversation, close_timeout_s)
                _trace_openhands_return("OpenHands conversation close returned")
            except Exception as e:
                logger.warning(f"Error closing conversation: {e}")
