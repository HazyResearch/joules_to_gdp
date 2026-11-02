#!/usr/bin/env python3
"""Blinded pairwise evaluation for the 220-task GDPval slice.

The program is deliberately stateful and idempotent.  It separates the
unblinded source manifest from the blinded request ledger, hashes every source
artifact, builds neutral multimodal evidence, uploads only sanitized derived
files, submits resumable OpenAI Batch jobs, and computes task-clustered
statistics after all verdicts are present.

Nothing in this module executes a submitted workbook, macro, notebook, web
application, or native binary.  Office rendering is performed by LibreOffice
in a temporary profile.  Archives are extracted with path and expansion
limits.  Audio/video are decoded only with ffmpeg/imageio-ffmpeg and are never
executed.
"""

from __future__ import annotations

import argparse
import base64
import csv
import concurrent.futures
import datetime as dt
import email
import hashlib
import html
import json
import math
import mimetypes
import os
import random
import re
import shutil
import ssl
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import wave
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from email import policy
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Sequence


SCHEMA_VERSION = "gdpval-pairwise-v1"
ORDER_SALT = "gdpval-order-v1"
DEFAULT_MODEL = "gpt-5-mini-2025-08-07"
DEFAULT_AUDIO_MODEL = "gpt-audio-1.5"
DEFAULT_MAX_OUTPUT_TOKENS = 12_000
BASELINES = ("human", "human_ai")
CONDITIONS = ("with_rubric", "without_rubric")
DEFAULT_CONDITIONS = ("without_rubric",)
VALID_WINNERS = {"A", "B", "tie"}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOLUME = Path(
    os.getenv("GDPVAL_RUN_VOLUME", "/Volumes/GDPvalGLMFullRuns")
)
DEFAULT_BASELINES = Path(
    os.getenv("GDPVAL_BASELINES", str(REPO_ROOT / "GDPVal_220_Tasks"))
)
DEFAULT_STATE = Path(
    os.getenv("GDPVAL_PAIRWISE_STATE", "reports/gdpval_pairwise_glm52_20260715")
)
RUN_REL = Path("runs/gdpval_smoke/run_openhands_glm_5_2_fp8_gdpval")
WITH_RUNS = (
    "g52full_20260709T072707Z",
    "g52resume_20260709T2048Z_from_q0220_gmem086",
    "g52rerunzeros_20260710T0700Z",
)
WITHOUT_RUN = "g52norubric_ws_20260711T213231Z"

TEXT_EXTS = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".html",
    ".htm", ".svg", ".py", ".js", ".jsx", ".ts", ".tsx", ".sol",
    ".tf", ".tfvars", ".hcl", ".css", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".conf", ".sh", ".log", ".step", ".stp", ".example",
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
OFFICE_EXTS = {".docx", ".pptx", ".xlsx", ".xlsm", ".xls", ".doc", ".ppt"}
ARCHIVE_EXTS = {".zip"}

MAX_ARCHIVE_FILES = 500
MAX_ARCHIVE_BYTES = 1_000_000_000
MAX_TEXT_CHARS_PER_FILE = 120_000
MAX_BUNDLE_TEXT_CHARS = 700_000
COMPACT_RETRY_TEXT_CHARS_PER_BUNDLE = 50_000
MAX_UPLOAD_BYTES = 48 * 1024 * 1024
MAX_VISUAL_PAGES_PER_BUNDLE = 48

# These are pipeline provenance markers, not ordinary task vocabulary.  A
# match in uploaded evidence or a blinded request is therefore always a leak.
PROVENANCE_MARKERS = {
    "absolute run volume": str(DEFAULT_VOLUME),
    "baseline root": str(DEFAULT_BASELINES),
    "optional prompt configuration": "with_rubric",
    "default prompt configuration": "without_rubric",
    "full-run identifier": "g52full_",
    "resume-run identifier": "g52resume_",
    "zero-rerun identifier": "g52rerunzeros_",
    "source run identifier": "g52norubric_",
    "agent harness run path": "run_openhands_glm_5_2_fp8_gdpval",
}


class EvaluationError(RuntimeError):
    """A reproducibility, input, parser, API, or validation failure."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    tmp.replace(path)
    return count


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise EvaluationError(f"expected object at {path}:{line_no}")
            yield row


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_rel(path: Path, roots: Sequence[Path]) -> str:
    resolved = path.resolve()
    for root in roots:
        try:
            return str(resolved.relative_to(root.resolve()))
        except ValueError:
            pass
    return path.name


def dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and not p.is_symlink())
    return []


def source_file_record(path: Path, *, roots: Sequence[Path]) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "stable_path": stable_rel(path, roots),
        "name": path.name,
        "suffix": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def manifest_summary_for_state(state: Path) -> dict[str, Any]:
    path = state / "manifest_summary.json"
    if not path.is_file():
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def expected_comparison_count(state: Path) -> int:
    recorded = manifest_summary_for_state(state).get("comparisons")
    if recorded:
        return int(recorded)
    return (expected_task_count(state) * len(active_conditions_for_state(state))
            * len(BASELINES) * repeat_count_for_state(state))


def expected_task_count(state: Path) -> int:
    return int(manifest_summary_for_state(state).get("tasks") or 220)


def active_conditions_for_state(state: Path) -> tuple[str, ...]:
    summary = manifest_summary_for_state(state)
    raw = summary.get("active_conditions")
    if raw is None and isinstance(summary.get("conditions"), dict):
        raw = [name for name, count in summary["conditions"].items() if count]
    if raw is None:
        return DEFAULT_CONDITIONS
    if not isinstance(raw, (list, tuple)) or not raw:
        raise EvaluationError("manifest active_conditions must be a nonempty list")
    values = tuple(str(value) for value in raw)
    if len(set(values)) != len(values) or any(value not in CONDITIONS for value in values):
        raise EvaluationError(f"invalid manifest active_conditions: {values}")
    return tuple(condition for condition in CONDITIONS if condition in values)


def repeat_count_for_state(state: Path) -> int:
    value = int(manifest_summary_for_state(state).get("repeat_count") or 4)
    if value <= 0 or value % 2:
        raise EvaluationError(f"repeat_count must be a positive even integer, got {value}")
    return value


def provenance_markers_for_state(state: Path | None = None) -> dict[str, str]:
    markers = dict(PROVENANCE_MARKERS)
    if state is None:
        return markers
    extra = manifest_summary_for_state(state).get("provenance_markers")
    if isinstance(extra, dict):
        markers.update(
            {
                str(label): str(marker)
                for label, marker in extra.items()
                if str(marker)
            }
        )
    elif isinstance(extra, list):
        markers.update(
            {
                f"state provenance marker {index:02d}": str(marker)
                for index, marker in enumerate(extra, start=1)
                if str(marker)
            }
        )
    return markers


def artifact_dirs(run_dir: Path) -> list[Path]:
    root = run_dir / "artifacts"
    if not root.is_dir():
        raise EvaluationError(f"missing artifacts directory: {root}")
    return sorted(p for p in root.iterdir() if p.is_dir())


def load_artifact(path: Path, *, run_name: str, precedence: int) -> dict[str, Any]:
    meta_path = path / "metadata.json"
    if not meta_path.is_file():
        raise EvaluationError(f"missing metadata: {meta_path}")
    meta = read_json(meta_path)
    if not isinstance(meta, dict):
        raise EvaluationError(f"metadata is not an object: {meta_path}")
    task_id = str(meta.get("task_id") or "")
    repeat = int(meta.get("repeat_index", -1))
    if not task_id or repeat not in range(4):
        raise EvaluationError(f"bad task/repeat metadata: {meta_path}")
    workspace = path / "workspace"
    declared = _declared_deliverables(meta, workspace)
    completed = bool(meta.get("completed")) and not bool(meta.get("timed_out"))
    infra_reason = str(meta.get("unscorable_reason") or meta.get("failure_type") or "")
    infra_failure = bool(infra_reason) and any(
        token in infra_reason.lower()
        for token in ("infra", "connection", "timeout", "agent_or_llm", "server_error")
    )
    usable_result = completed and not infra_failure and workspace.is_dir()
    quality_status = "present"
    if not declared:
        quality_status = "missing"
    elif any(not p.is_file() or p.stat().st_size == 0 for p in declared):
        quality_status = "empty_or_missing"
    return {
        "task_id": task_id,
        "repeat": repeat,
        "instance_id": str(meta.get("instance_id") or f"{task_id}::rep{repeat:02d}"),
        "query_id": str(meta.get("query_id") or ""),
        "occupation": str(meta.get("occupation") or ""),
        "sector": str(meta.get("sector") or ""),
        "source_task_index": int(meta.get("source_task_index", -1)),
        "condition_in_prompt": bool(meta.get("include_rubric_in_prompt")),
        "artifact_dir": str(path),
        "workspace": str(workspace),
        "run_name": run_name,
        "precedence": precedence,
        "completed": completed,
        "infra_failure": infra_failure,
        "usable_result": usable_result,
        "candidate_quality_status": quality_status,
        "declared_names": [p.name for p in declared],
        "deliverable_paths": [str(p) for p in declared],
        "reference_paths": [str(p) for p in files_under(workspace / "inputs")],
        "metadata_path": str(meta_path),
    }


def _declared_deliverables(meta: dict[str, Any], workspace: Path) -> list[Path]:
    raw: list[str] = []
    validation = meta.get("deliverable_validation")
    if isinstance(validation, dict):
        for value in validation.get("primary_files") or []:
            raw.append(str(value))
    if not raw:
        for value in meta.get("deliverable_files") or []:
            raw.append(str(value))
    resolved: list[Path] = []
    for value in raw:
        old = Path(value)
        candidates: list[Path] = []
        if old.is_absolute():
            parts = list(old.parts)
            if "workspace" in parts:
                idx = len(parts) - 1 - parts[::-1].index("workspace")
                candidates.append(workspace.joinpath(*parts[idx + 1 :]))
            candidates.append(workspace / old.name)
        else:
            candidates.append(workspace / old)
            candidates.append(workspace / old.name)
        chosen = next((p for p in candidates if p.exists()), candidates[0])
        if chosen.is_dir():
            resolved.extend(files_under(chosen))
        else:
            resolved.append(chosen)
    return dedupe_paths(resolved)


def select_condition_artifacts(volume: Path, condition: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if condition == "without_rubric":
        run_names = (WITHOUT_RUN,)
    elif condition == "with_rubric":
        run_names = WITH_RUNS
    else:
        raise EvaluationError(f"unknown condition: {condition}")
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    audit: dict[str, Any] = {"condition": condition, "runs": {}, "selection_rule": "later usable result wins"}
    for precedence, run_name in enumerate(run_names):
        run_dir = volume / run_name / RUN_REL
        rows = [load_artifact(p, run_name=run_name, precedence=precedence) for p in artifact_dirs(run_dir)]
        audit["runs"][run_name] = {
            "rows": len(rows),
            "usable": sum(bool(r["usable_result"]) for r in rows),
            "candidate_missing_or_empty": sum(r["candidate_quality_status"] != "present" for r in rows),
        }
        for row in rows:
            by_key[(row["task_id"], row["repeat"])].append(row)
    selected: list[dict[str, Any]] = []
    replacements = Counter()
    for key, candidates in by_key.items():
        ordered = sorted(candidates, key=lambda r: int(r["precedence"]))
        usable = [r for r in ordered if r["usable_result"]]
        chosen = usable[-1] if usable else ordered[-1]
        chosen = dict(chosen)
        chosen["condition"] = condition
        chosen["selection_candidates"] = [
            {k: r[k] for k in ("run_name", "query_id", "usable_result", "candidate_quality_status")}
            for r in ordered
        ]
        selected.append(chosen)
        replacements[chosen["run_name"]] += 1
    selected.sort(key=lambda r: (int(r["source_task_index"]), int(r["repeat"]), str(r["task_id"])))
    audit["selected_by_run"] = dict(replacements)
    audit["unique_task_repeats"] = len(selected)
    _validate_880(selected, condition)
    return selected, audit


def _validate_880(rows: Sequence[dict[str, Any]], condition: str) -> None:
    keys = {(r["task_id"], int(r["repeat"])) for r in rows}
    tasks = {r["task_id"] for r in rows}
    if len(rows) != 880 or len(keys) != 880 or len(tasks) != 220:
        raise EvaluationError(
            f"{condition}: expected 880 unique task-repeat rows over 220 tasks; "
            f"got rows={len(rows)}, keys={len(keys)}, tasks={len(tasks)}"
        )
    by_task: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        by_task[str(row["task_id"])].add(int(row["repeat"]))
    bad = {task: sorted(reps) for task, reps in by_task.items() if reps != {0, 1, 2, 3}}
    if bad:
        raise EvaluationError(f"{condition}: tasks without repeats 0..3: {bad}")


def extract_task_prompt(wrapped: str) -> str:
    match = re.search(r"(?im)^##\s+Task\s*$", wrapped)
    if match:
        tail = wrapped[match.end() :]
        next_heading = re.search(r"(?im)^##\s+(?:Tools and workflow|Rubric)\s*$", tail)
        return (tail[: next_heading.start()] if next_heading else tail).strip()
    return wrapped.strip()


def baseline_prompt(task: dict[str, Any]) -> str:
    solved = task.get("estimation_tasks", {}).get("solved_with_ai", {})
    turns = solved.get("turns") if isinstance(solved, dict) else None
    if not isinstance(turns, list) or not turns:
        return ""
    prompt = str(turns[0].get("user_message") or "")
    prompt = re.split(r"(?i)please\s+make\s+sure\s+the\s+output\s+meets\s+these\s+rubric\s+requirements", prompt)[0]
    return prompt.strip()


def normalized_words(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 2 and token not in {"the", "and", "for", "with", "you", "your", "this", "that"}
    }


def jaccard(a: str, b: str) -> float:
    aa, bb = normalized_words(a), normalized_words(b)
    return len(aa & bb) / len(aa | bb) if aa or bb else 1.0


def title_position(title: str) -> int:
    match = re.search(r"_(\d+)\s*$", title)
    if not match:
        raise EvaluationError(f"baseline title lacks numeric suffix: {title!r}")
    value = int(match.group(1))
    if value not in range(1, 6):
        raise EvaluationError(f"baseline title suffix outside 1..5: {title!r}")
    return value


def normalize_filename(value: str) -> str:
    name = Path(value).name.lower()
    name = re.sub(r"\s+", " ", name).strip()
    return name


def filename_tokens(value: str) -> set[str]:
    stem = Path(value).stem.lower()
    return {token for token in re.findall(r"[a-z0-9]+", stem) if len(token) > 1}


def resolve_reference_paths(
    available: Sequence[str | Path], expected_names: Sequence[str]
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Resolve the canonical task references and reject run-created input helpers.

    Exact basename matches are required except for the single known GDPval
    filename alias.  The alias resolver is intentionally conservative: the
    expected stem tokens must be contained in a unique best candidate, and a
    later content audit verifies identical bytes across all selected repeats.
    """
    candidates = [
        Path(value)
        for value in available
        if Path(value).is_file()
        and not Path(value).name.startswith(("~$", "._"))
        and Path(value).name not in {".DS_Store", "Thumbs.db"}
    ]
    selected: list[Path] = []
    linkage: list[dict[str, Any]] = []
    used: set[str] = set()
    for expected in expected_names:
        exact = [
            path for path in candidates
            if str(path.resolve()) not in used
            and normalize_filename(path.name) == normalize_filename(expected)
        ]
        if len(exact) == 1:
            chosen, match_type, similarity = exact[0], "exact_filename", 1.0
        elif len(exact) > 1:
            raise EvaluationError(f"ambiguous exact reference match for {expected!r}: {exact}")
        else:
            expected_tokens = filename_tokens(expected)
            scored: list[tuple[float, Path]] = []
            for path in candidates:
                if str(path.resolve()) in used:
                    continue
                actual_tokens = filename_tokens(path.name)
                if expected_tokens and expected_tokens.issubset(actual_tokens):
                    scored.append((len(expected_tokens) / len(actual_tokens), path))
            scored.sort(key=lambda item: (-item[0], normalize_filename(item[1].name)))
            if not scored or scored[0][0] < 0.5 or (len(scored) > 1 and scored[0][0] == scored[1][0]):
                raise EvaluationError(
                    f"cannot uniquely resolve canonical reference {expected!r}; "
                    f"available={[path.name for path in candidates]}"
                )
            similarity, chosen = scored[0]
            match_type = "verified_alias_candidate"
        used.add(str(chosen.resolve()))
        selected.append(chosen)
        linkage.append(
            {
                "expected_name": expected,
                "actual_name": chosen.name,
                "match_type": match_type,
                "filename_token_similarity": round(float(similarity), 6),
            }
        )
    if len(selected) != len(expected_names):
        raise EvaluationError("canonical reference resolution is incomplete")
    return selected, linkage


def load_agent_prompts(run_dir: Path) -> dict[str, str]:
    path = run_dir / "pre_judge_traces.jsonl"
    prompts: dict[str, str] = {}
    for row in iter_jsonl(path):
        query_id = str(row.get("query_id") or "")
        if query_id:
            prompts[query_id] = extract_task_prompt(str(row.get("query_text") or row.get("problem") or ""))
    return prompts


def build_crosswalk(
    without_rows: Sequence[dict[str, Any]],
    baseline_tasks: Sequence[dict[str, Any]],
    prompts: dict[str, str],
) -> list[dict[str, Any]]:
    first_by_task: dict[str, dict[str, Any]] = {}
    for row in without_rows:
        if int(row["repeat"]) == 0:
            first_by_task[str(row["task_id"])] = row
    by_occ_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in first_by_task.values():
        by_occ_agent[str(row["occupation"])].append(row)
    by_occ_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in baseline_tasks:
        by_occ_base[str(task["occupation"])].append(task)
    if set(by_occ_agent) != set(by_occ_base):
        raise EvaluationError(
            f"occupation mismatch: only agent={sorted(set(by_occ_agent)-set(by_occ_base))}; "
            f"only baseline={sorted(set(by_occ_base)-set(by_occ_agent))}"
        )
    crosswalk: list[dict[str, Any]] = []
    for occupation in sorted(by_occ_agent):
        agents = sorted(by_occ_agent[occupation], key=lambda r: int(r["source_task_index"]))
        bases = sorted(by_occ_base[occupation], key=lambda t: title_position(str(t["title"])))
        if len(agents) != 5 or len(bases) != 5:
            raise EvaluationError(f"{occupation}: expected five tasks, got agent={len(agents)} baseline={len(bases)}")
        for position, (agent, base) in enumerate(zip(agents, bases), start=1):
            aprompt = prompts.get(str(agent["query_id"]), "")
            bprompt = baseline_prompt(base)
            agent_refs = [Path(p).name for p in agent.get("reference_paths", [])]
            base_refs = [str(v) for v in base.get("reference_files") or []]
            exact_refs = sorted(map(normalize_filename, agent_refs)) == sorted(map(normalize_filename, base_refs))
            ref_similarity = jaccard(" ".join(agent_refs), " ".join(base_refs))
            prompt_similarity = jaccard(aprompt, bprompt)
            crosswalk.append(
                {
                    "agent_task_id": agent["task_id"],
                    "baseline_task_id": base["task_id"],
                    "occupation": occupation,
                    "sector": base.get("sector") or agent.get("sector"),
                    "title": base["title"],
                    "position_within_occupation": position,
                    "agent_prompt": aprompt,
                    "baseline_prompt": bprompt,
                    "prompt_word_jaccard": round(prompt_similarity, 6),
                    "reference_word_jaccard": round(ref_similarity, 6),
                    "agent_reference_names": agent_refs,
                    "baseline_reference_names": base_refs,
                    "reference_names_exact": exact_refs,
                }
            )
    if len(crosswalk) != 220 or len({x["agent_task_id"] for x in crosswalk}) != 220 or len({x["baseline_task_id"] for x in crosswalk}) != 220:
        raise EvaluationError("crosswalk is not one-to-one over 220 tasks")
    exact_count = sum(bool(x["reference_names_exact"]) for x in crosswalk)
    if exact_count != 219:
        raise EvaluationError(f"crosswalk reference verification expected 219/220 exact sets; got {exact_count}/220")
    return crosswalk


def canonical_rubric(task: dict[str, Any]) -> list[dict[str, Any]]:
    raw = task.get("rubric_json")
    if not isinstance(raw, list) or not raw:
        raise EvaluationError(f"missing rubric_json for {task.get('task_id')}")
    parsed: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if isinstance(item, dict):
            text = str(item.get("criterion") or item.get("text") or item)
            points = item.get("points")
        else:
            text = str(item)
            match = re.match(r"\s*\[([+-]?\d+(?:\.\d+)?)\]\s*(.*)", text, re.S)
            points = float(match.group(1)) if match else None
            text = match.group(2).strip() if match else text.strip()
        parsed.append({"index": index, "points": points, "criterion": text})
    return parsed


def agent_position(task_id: str, condition: str, baseline: str, repeat: int) -> str:
    values = []
    for rep in range(4):
        token = f"{ORDER_SALT}|{task_id}|{condition}|{baseline}|{rep}"
        values.append((sha256_text(token), rep))
    ranked = [rep for _, rep in sorted(values)]
    return "A" if repeat in set(ranked[:2]) else "B"


def verify_reference_alias_content(
    comparisons: Sequence[dict[str, Any]], crosswalk: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    alias_rows = [row for row in crosswalk if not row["reference_names_exact"]]
    if len(alias_rows) != 1:
        raise EvaluationError(f"expected exactly one reference-filename alias task, got {len(alias_rows)}")
    audit_rows: list[dict[str, Any]] = []
    expected_slots = {(str(row["condition"]), int(row["repeat"])) for row in comparisons}
    for cross in alias_rows:
        task_id = str(cross["agent_task_id"])
        occurrences: dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]] = {}
        for row in comparisons:
            if str(row["task_id"]) != task_id:
                continue
            for link, source in zip(row["reference_linkage"], row["reference_files"], strict=True):
                if link["match_type"] == "verified_alias_candidate":
                    occurrences[(str(row["condition"]), int(row["repeat"]))] = (link, source)
        if set(occurrences) != expected_slots:
            raise EvaluationError(f"reference alias {task_id} was not resolved in every selected repeat")
        links = [value[0] for value in occurrences.values()]
        sources = [value[1] for value in occurrences.values()]
        hashes = {str(source["sha256"]) for source in sources}
        names = {str(source["name"]) for source in sources}
        if len(hashes) != 1 or len(names) != 1:
            raise EvaluationError(
                f"reference alias {task_id} does not resolve to byte-identical content across runs: "
                f"names={names}, hashes={hashes}"
            )
        source_path = Path(str(sources[0]["path"]))
        semantic_text, semantic_meta, semantic_errors = inspect_file_semantics(source_path)
        expected_tokens = filename_tokens(str(links[0]["expected_name"]))
        content_overlap = sorted(expected_tokens & normalized_words(semantic_text))
        if semantic_errors or len(semantic_text.strip()) < 500 or not content_overlap:
            raise EvaluationError(
                f"reference alias {task_id} failed semantic content verification: "
                f"errors={semantic_errors}, chars={len(semantic_text)}, overlap={content_overlap}"
            )
        audit_rows.append(
            {
                "agent_task_id": task_id,
                "baseline_task_id": cross["baseline_task_id"],
                "expected_name": links[0]["expected_name"],
                "resolved_actual_name": next(iter(names)),
                "source_sha256": next(iter(hashes)),
                "verified_across_task_condition_repeats": len(occurrences),
                "semantic_content_sha256": sha256_text(semantic_text),
                "semantic_characters": len(semantic_text),
                "expected_name_tokens_found_in_content": content_overlap,
                "semantic_metadata": semantic_meta,
                "agent_baseline_prompt_word_jaccard": cross["prompt_word_jaccard"],
                "verification": "stable bytes plus nonempty semantic extraction and task-name/content anchor",
            }
        )
    return audit_rows


def build_manifest(
    volume: Path, baseline_root: Path, state: Path, *, include_rubric: bool = False
) -> dict[str, Any]:
    if not volume.is_dir():
        raise EvaluationError(f"GDPval run volume is not mounted: {volume}")
    baseline_json = baseline_root / "final_1-220_samples.json"
    baseline_tasks = read_json(baseline_json)
    if not isinstance(baseline_tasks, list) or len(baseline_tasks) != 220:
        raise EvaluationError(f"expected 220 baseline tasks in {baseline_json}")

    active_conditions = CONDITIONS if include_rubric else DEFAULT_CONDITIONS
    selected = {condition: select_condition_artifacts(volume, condition)
                for condition in active_conditions}
    without_rows, _ = selected["without_rubric"]
    no_run_dir = volume / WITHOUT_RUN / RUN_REL
    prompts = load_agent_prompts(no_run_dir)
    crosswalk = build_crosswalk(without_rows, baseline_tasks, prompts)
    x_by_agent = {x["agent_task_id"]: x for x in crosswalk}
    b_by_id = {str(t["task_id"]): t for t in baseline_tasks}

    roots = [volume, baseline_root]
    file_hash_cache: dict[str, dict[str, Any]] = {}

    def record(path: Path) -> dict[str, Any]:
        key = str(path.resolve())
        if key not in file_hash_cache:
            file_hash_cache[key] = source_file_record(path, roots=roots)
        return file_hash_cache[key]

    comparisons: list[dict[str, Any]] = []
    for condition in active_conditions:
        for output in selected[condition][0]:
            task_id = str(output["task_id"])
            cross = x_by_agent[task_id]
            baseline_task = b_by_id[str(cross["baseline_task_id"])]
            agent_files = [Path(p) for p in output["deliverable_paths"]]
            agent_records = [record(p) for p in agent_files if p.is_file()]
            expected_refs = list(baseline_task.get("reference_files") or [])
            reference_files, reference_linkage = resolve_reference_paths(
                output["reference_paths"], expected_refs
            )
            reference_records = [record(p) for p in reference_files]
            for baseline in BASELINES:
                baseline_dir = baseline_root / "deliverables" / str(baseline_task["task_id"]) / (
                    "human" if baseline == "human" else "human+ai"
                )
                baseline_files = files_under(baseline_dir)
                baseline_records = [record(p) for p in baseline_files]
                if not baseline_records:
                    raise EvaluationError(f"missing {baseline} deliverable: {baseline_dir}")
                pos = agent_position(task_id, condition, baseline, int(output["repeat"]))
                comparison_id = (
                    f"{condition}__{baseline}__{task_id}__rep{int(output['repeat']):02d}"
                )
                comparisons.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "comparison_id": comparison_id,
                        "task_id": task_id,
                        "baseline_task_id": baseline_task["task_id"],
                        "condition": condition,
                        "baseline": baseline,
                        "repeat": int(output["repeat"]),
                        "occupation": cross["occupation"],
                        "sector": cross["sector"],
                        "title": cross["title"],
                        "task_prompt": cross["agent_prompt"] or cross["baseline_prompt"],
                        "rubric": canonical_rubric(baseline_task),
                        "rubric_pretty": str(baseline_task.get("rubric_pretty") or ""),
                        "expected_reference_names": expected_refs,
                        "reference_files": reference_records,
                        "reference_linkage": reference_linkage,
                        "agent_position": pos,
                        "candidate_A_source": "agent" if pos == "A" else baseline,
                        "candidate_B_source": baseline if pos == "A" else "agent",
                        "agent_files": agent_records,
                        "baseline_files": baseline_records,
                        "agent_quality_status": output["candidate_quality_status"],
                        "agent_run_name": output["run_name"],
                        "agent_artifact_dir": output["artifact_dir"],
                        "baseline_dir": str(baseline_dir),
                    }
                )

    comparisons.sort(key=lambda r: r["comparison_id"])
    validate_manifest(comparisons, conditions=active_conditions)
    alias_audit = verify_reference_alias_content(comparisons, crosswalk)
    state.mkdir(parents=True, exist_ok=True)
    write_jsonl(state / "comparison_manifest.unblinded.jsonl", comparisons)
    write_json(state / "crosswalk.json", crosswalk)
    write_json(state / "source_files.json", sorted(file_hash_cache.values(), key=lambda r: r["path"]))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "volume": str(volume),
        "baseline_root": str(baseline_root),
        "baseline_task_json": source_file_record(baseline_json, roots=roots),
        "comparisons": len(comparisons),
        "tasks": len({r["task_id"] for r in comparisons}),
        "repeat_count": 4,
        "active_conditions": list(active_conditions),
        "source_files_hashed": len(file_hash_cache),
        "position_counts": dict(Counter(r["agent_position"] for r in comparisons)),
        "conditions": {c: sum(r["condition"] == c for r in comparisons) for c in active_conditions},
        "baselines": {b: sum(r["baseline"] == b for r in comparisons) for b in BASELINES},
        "source_reconstruction": {condition: selected[condition][1] for condition in active_conditions},
        "crosswalk_reference_name_exact": sum(bool(x["reference_names_exact"]) for x in crosswalk),
        "crosswalk_reference_alias_rows": [x for x in crosswalk if not x["reference_names_exact"]],
        "reference_alias_content_audit": alias_audit,
        "minimum_prompt_word_jaccard": min(x["prompt_word_jaccard"] for x in crosswalk),
        "agent_quality_status": dict(Counter(r["agent_quality_status"] for r in comparisons)),
        "manifest_sha256": sha256_file(state / "comparison_manifest.unblinded.jsonl"),
    }
    write_json(state / "manifest_summary.json", summary)
    return summary


def validate_manifest(
    rows: Sequence[dict[str, Any]],
    *,
    expected_tasks: int = 220,
    repeat_count: int = 4,
    conditions: Sequence[str] = DEFAULT_CONDITIONS,
) -> None:
    active_conditions = tuple(str(condition) for condition in conditions)
    if (
        not active_conditions
        or len(set(active_conditions)) != len(active_conditions)
        or any(condition not in CONDITIONS for condition in active_conditions)
    ):
        raise EvaluationError(f"invalid active conditions: {active_conditions}")
    if repeat_count <= 0 or repeat_count % 2:
        raise EvaluationError(f"repeat_count must be a positive even integer, got {repeat_count}")
    expected_comparisons = expected_tasks * len(active_conditions) * len(BASELINES) * repeat_count
    expected_strata = expected_tasks * len(active_conditions) * len(BASELINES)
    if len(rows) != expected_comparisons:
        raise EvaluationError(f"expected {expected_comparisons:,} comparisons, got {len(rows)}")
    if len({r["comparison_id"] for r in rows}) != expected_comparisons:
        raise EvaluationError("comparison IDs are not unique")
    counts = Counter(r["agent_position"] for r in rows)
    expected_position_count = expected_comparisons // 2
    if counts != {"A": expected_position_count, "B": expected_position_count}:
        raise EvaluationError(f"global position imbalance: {dict(counts)}")
    strata: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    repeats: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for row in rows:
        if str(row["condition"]) not in active_conditions:
            raise EvaluationError(
                f"manifest contains condition outside active design: {row['condition']}"
            )
        key = (row["task_id"], row["condition"], row["baseline"])
        strata[key][row["agent_position"]] += 1
        repeats[key].add(int(row["repeat"]))
    expected_repeats = set(range(repeat_count))
    expected_positions = {"A": repeat_count // 2, "B": repeat_count // 2}
    bad = {
        key: {"positions": dict(strata[key]), "repeats": sorted(repeats[key])}
        for key in strata
        if strata[key] != expected_positions or repeats[key] != expected_repeats
    }
    if bad:
        raise EvaluationError(f"within-task position/repeat validation failed: {list(bad.items())[:5]}")
    if len(strata) != expected_strata:
        raise EvaluationError(f"expected {expected_strata} task-condition-baseline strata, got {len(strata)}")
    if any(not row.get("rubric") for row in rows):
        raise EvaluationError("one or more comparisons lack the canonical rubric")
    linkage_failures = [
        row["comparison_id"]
        for row in rows
        if len(row.get("reference_files") or []) != len(row.get("expected_reference_names") or [])
        or len(row.get("reference_linkage") or []) != len(row.get("expected_reference_names") or [])
    ]
    if linkage_failures:
        raise EvaluationError(f"incomplete canonical reference linkage: {linkage_failures[:10]}")
    reference_hashes: dict[str, set[str]] = defaultdict(set)
    baseline_hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        reference_hashes[str(row["task_id"])].add(bundle_source_hash(row["reference_files"]))
        baseline_hashes[(str(row["task_id"]), str(row["baseline"]))].add(
            bundle_source_hash(row["baseline_files"])
        )
    bad_references = {task: sorted(values) for task, values in reference_hashes.items() if len(values) != 1}
    bad_baselines = {str(key): sorted(values) for key, values in baseline_hashes.items() if len(values) != 1}
    if bad_references:
        raise EvaluationError(f"reference bundle differs across repeats/conditions: {list(bad_references.items())[:5]}")
    if bad_baselines:
        raise EvaluationError(f"baseline bundle differs across comparisons: {list(bad_baselines.items())[:5]}")
    expected_baseline_bundles = expected_tasks * len(BASELINES)
    if len(reference_hashes) != expected_tasks or len(baseline_hashes) != expected_baseline_bundles:
        raise EvaluationError(
            f"expected reference linkage for {expected_tasks} tasks and baseline linkage for {expected_baseline_bundles} task-baselines; "
            f"got {len(reference_hashes)} and {len(baseline_hashes)}"
        )


def ffmpeg_executable() -> str | None:
    configured = os.environ.get("IPW_FFMPEG_EXECUTABLE")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def is_corrupt_media_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in (
            "eoferror",
            "failed to find two consecutive mpeg audio frames",
            "invalid data found when processing input",
            "moov atom not found",
            "could not find codec parameters",
            "output file does not contain any stream",
            "error opening input files: end of file",
        )
    )


def invalid_submitted_document_reason(path: Path) -> str | None:
    """Return a source-content defect only when a document is demonstrably invalid.

    This check is intentionally narrower than catching parser exceptions.  A
    renderer/library failure on a structurally valid document must remain a
    fatal parser error.  Only malformed PDFs and OOXML packages can be
    classified as submitted-file quality defects.
    """

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if not path.is_file() or path.stat().st_size == 0:
            return "the submitted PDF is empty"
        if path.read_bytes()[:5] != b"%PDF-":
            return "the submitted file does not contain a PDF header"
        try:
            from pypdf import PdfReader

            reader = PdfReader(path, strict=False)
            # Materialize the page tree so truncated streams/xrefs are checked
            # rather than deferred until visual sanitization.
            _ = len(reader.pages)
        except Exception:
            return "the submitted PDF package is truncated or structurally invalid"
        return None

    required_members = {
        ".docx": {"[Content_Types].xml", "word/document.xml"},
        ".xlsx": {"[Content_Types].xml", "xl/workbook.xml"},
        ".xlsm": {"[Content_Types].xml", "xl/workbook.xml"},
        ".pptx": {"[Content_Types].xml", "ppt/presentation.xml"},
    }.get(suffix)
    if not required_members:
        return None
    if not path.is_file() or path.stat().st_size == 0:
        return "the submitted Office document is empty"
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            missing = sorted(required_members - names)
            if missing:
                return (
                    "the submitted Office package is structurally invalid "
                    f"(missing {', '.join(missing)})"
                )
            bad_member = archive.testzip()
            if bad_member:
                return "the submitted Office package contains a corrupt member"
    except (OSError, zipfile.BadZipFile):
        return "the submitted file is not a valid Office package"
    return None


def run_checked(
    command: list[str],
    *,
    timeout: int = 300,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
        env=env,
    )
    if result.returncode:
        raise EvaluationError(
            f"command failed ({result.returncode}): {' '.join(command[:4])}: "
            f"{(result.stderr or result.stdout)[-1200:]}"
        )
    return result


def bounded(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    head = limit * 3 // 4
    tail = limit - head
    return value[:head] + f"\n... [{len(value) - limit:,} characters omitted] ...\n" + value[-tail:]


def xml_text_from_zip(path: Path, members: Sequence[str]) -> str:
    parts: list[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            for member in members:
                try:
                    raw = zf.read(member).decode("utf-8", errors="replace")
                except KeyError:
                    continue
                text = re.sub(r"<[^>]+>", " ", raw)
                text = html.unescape(re.sub(r"\s+", " ", text)).strip()
                if text:
                    parts.append(f"[{member}]\n{text}")
    except Exception:
        return ""
    return "\n\n".join(parts)


def inspect_spreadsheet(path: Path) -> tuple[str, dict[str, Any]]:
    from openpyxl import load_workbook

    formulas = load_workbook(path, data_only=False, read_only=False, keep_vba=False)
    # Random access on an openpyxl read-only worksheet reparses XML for every
    # ``cell()`` lookup and becomes quadratic. Materialize both views once.
    values = load_workbook(path, data_only=True, read_only=False, keep_vba=False)
    lines: list[str] = []
    sheets: list[dict[str, Any]] = []
    for ws, wsv in zip(formulas.worksheets, values.worksheets):
        sheet_meta: dict[str, Any] = {
            "name": ws.title,
            "state": ws.sheet_state,
            "max_row": ws.max_row,
            "max_column": ws.max_column,
            "freeze_panes": str(ws.freeze_panes or ""),
            "auto_filter": str(ws.auto_filter.ref or ""),
            "merged_ranges": [str(x) for x in list(ws.merged_cells.ranges)[:200]],
            "tables": [
                {"name": name, "ref": ws.tables[name].ref}
                for name in list(ws.tables)[:100]
            ],
            "charts": len(getattr(ws, "_charts", [])),
            "images": len(getattr(ws, "_images", [])),
            "hidden_rows": sum(bool(dim.hidden) for dim in ws.row_dimensions.values()),
            "hidden_columns": sum(bool(dim.hidden) for dim in ws.column_dimensions.values()),
        }
        sheets.append(sheet_meta)
        lines.append(f"## Sheet {ws.title!r} ({ws.sheet_state})")
        lines.append(json.dumps(sheet_meta, ensure_ascii=False))
        row_cap = min(ws.max_row or 0, 1200)
        col_cap = min(ws.max_column or 0, 80)
        for row_idx in range(1, row_cap + 1):
            cells: list[str] = []
            nonempty = False
            for col_idx in range(1, col_cap + 1):
                formula = ws.cell(row_idx, col_idx).value
                value = wsv.cell(row_idx, col_idx).value
                if formula is None and value is None:
                    cells.append("")
                    continue
                nonempty = True
                coord = ws.cell(row_idx, col_idx).coordinate
                if isinstance(formula, str) and formula.startswith("="):
                    cells.append(f"{coord}: formula={formula!r}; displayed={value!r}")
                else:
                    cells.append(f"{coord}: {value!r}")
            if nonempty:
                lines.append(" | ".join(cells))
        if (ws.max_row or 0) > row_cap:
            lines.append(f"[Rows {row_cap + 1}..{ws.max_row} not expanded; dimensions retained above]")
        if (ws.max_column or 0) > col_cap:
            lines.append(f"[Columns {col_cap + 1}..{ws.max_column} not expanded; dimensions retained above]")
    metadata = {
        "sheet_count": len(formulas.sheetnames),
        "sheet_names": formulas.sheetnames,
        "defined_names": sorted(str(name) for name in formulas.defined_names)[:500],
        "sheets": sheets,
        "calculation": str(getattr(formulas, "calculation", "")),
    }
    formulas.close()
    values.close()
    return bounded("\n".join(lines), MAX_TEXT_CHARS_PER_FILE), metadata


def recover_spreadsheet_xml(path: Path) -> tuple[str, dict[str, Any]]:
    """Recover neutral spreadsheet text when openpyxl rejects valid package extensions."""
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if name == "xl/workbook.xml"
            or name == "xl/sharedStrings.xml"
            or name.startswith("xl/worksheets/")
            or name.startswith("xl/tables/")
            or name.startswith("xl/charts/")
        )
    text = xml_text_from_zip(path, names)
    if not text.strip():
        raise EvaluationError(f"recovered spreadsheet XML contained no text: {path}")
    return bounded(text, MAX_TEXT_CHARS_PER_FILE), {
        "recovery": "source-blind OOXML package text fallback",
        "package_members_read": len(names),
    }


def inspect_docx(path: Path) -> tuple[str, dict[str, Any]]:
    import docx

    document = docx.Document(path)
    lines: list[str] = []
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            lines.append(paragraph.text)
    for table_idx, table in enumerate(document.tables, start=1):
        lines.append(f"## Table {table_idx}")
        for row in table.rows:
            lines.append(" | ".join(cell.text for cell in row.cells))
    for section_idx, section in enumerate(document.sections, start=1):
        for label, part in (("header", section.header), ("footer", section.footer)):
            text = "\n".join(p.text for p in part.paragraphs if p.text.strip())
            if text:
                lines.append(f"## Section {section_idx} {label}\n{text}")
    comments = ""
    try:
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n in {"word/comments.xml", "word/footnotes.xml", "word/endnotes.xml"}]
        comments = xml_text_from_zip(path, names)
    except zipfile.BadZipFile:
        pass
    if comments:
        lines.append("## Comments, footnotes, and endnotes\n" + comments)
    return bounded("\n\n".join(lines), MAX_TEXT_CHARS_PER_FILE), {
        "paragraphs": len(document.paragraphs),
        "tables": len(document.tables),
        "sections": len(document.sections),
        "inline_shapes": len(document.inline_shapes),
    }


def recover_docx_xml(path: Path) -> tuple[str, dict[str, Any]]:
    from lxml import etree

    with zipfile.ZipFile(path) as zf:
        raw = zf.read("word/document.xml")
    root = etree.fromstring(raw, parser=etree.XMLParser(recover=True, huge_tree=True))
    paragraphs: list[str] = []
    for paragraph in root.xpath("//*[local-name()='p']"):
        text = "".join(paragraph.xpath(".//*[local-name()='t']/text()"))
        if text.strip():
            paragraphs.append(text)
    if not paragraphs:
        raise EvaluationError(f"recovered DOCX XML contained no text: {path}")
    return bounded("\n\n".join(paragraphs), MAX_TEXT_CHARS_PER_FILE), {
        "recovery": "lxml recover=True over word/document.xml",
        "recovered_paragraphs": len(paragraphs),
    }


def inspect_pptx(path: Path) -> tuple[str, dict[str, Any]]:
    from pptx import Presentation

    presentation = Presentation(path)
    lines: list[str] = []
    slides: list[dict[str, Any]] = []
    for slide_idx, slide in enumerate(presentation.slides, start=1):
        lines.append(f"## Slide {slide_idx}")
        slide_meta = {"slide": slide_idx, "shapes": len(slide.shapes), "notes": False, "tables": 0, "charts": 0}
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False) and shape.text.strip():
                lines.append(shape.text)
            if getattr(shape, "has_table", False):
                slide_meta["tables"] += 1
                for row in shape.table.rows:
                    lines.append(" | ".join(cell.text for cell in row.cells))
            if getattr(shape, "has_chart", False):
                slide_meta["charts"] += 1
                chart = shape.chart
                lines.append(f"[Chart: type={chart.chart_type}; series={len(chart.series)}]")
                for series in chart.series:
                    lines.append(f"Series {getattr(series, 'name', '')!r}")
        try:
            notes = "\n".join(
                shape.text for shape in slide.notes_slide.shapes
                if getattr(shape, "has_text_frame", False) and shape.text.strip()
            )
            if notes:
                slide_meta["notes"] = True
                lines.append("### Speaker notes\n" + notes)
        except Exception:
            pass
        slides.append(slide_meta)
    with zipfile.ZipFile(path) as zf:
        media = [n for n in zf.namelist() if n.startswith("ppt/media/") and not n.endswith("/")]
    return bounded("\n\n".join(lines), MAX_TEXT_CHARS_PER_FILE), {
        "slides": len(presentation.slides),
        "slide_size_inches": [presentation.slide_width / 914400, presentation.slide_height / 914400],
        "slide_details": slides,
        "embedded_media_inventory": [Path(name).name for name in media],
    }


def inspect_pdf(path: Path) -> tuple[str, dict[str, Any]]:
    from pypdf import PdfReader

    reader = PdfReader(path, strict=False)
    lines: list[str] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            lines.append(f"## Page {idx}\n{text}")
    return bounded("\n\n".join(lines), MAX_TEXT_CHARS_PER_FILE), {
        "pages": len(reader.pages),
        "encrypted": bool(reader.is_encrypted),
    }


def inspect_image(path: Path) -> tuple[str, dict[str, Any]]:
    from PIL import Image

    with Image.open(path) as image:
        meta = {
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "frames": int(getattr(image, "n_frames", 1)),
        }
    return "Visual content is provided in the sanitized visual evidence PDF.", meta


def inspect_audio(path: Path) -> tuple[str, dict[str, Any]]:
    meta: dict[str, Any] = {}
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wav:
                meta = {
                    "duration_seconds": wav.getnframes() / wav.getframerate(),
                    "sample_rate": wav.getframerate(),
                    "channels": wav.getnchannels(),
                    "sample_width_bytes": wav.getsampwidth(),
                    "wav_encoding": "PCM",
                }
        except wave.Error as exc:
            # Valid WAV can use IEEE float or extensible encodings unsupported
            # by the stdlib reader. ffmpeg sanitization validates/normalizes it
            # before the source-blind audio parser.
            meta = {"wav_encoding": "non-PCM or extensible", "stdlib_wave_note": str(exc)}
    return "Audio content is described in the source-blind audio-parser record.", meta


def inspect_email(path: Path) -> tuple[str, dict[str, Any]]:
    message = email.message_from_bytes(path.read_bytes(), policy=policy.default)
    lines = []
    for key in ("Date", "From", "To", "Cc", "Subject"):
        if message.get(key):
            lines.append(f"{key}: {message.get(key)}")
    attachments: list[str] = []
    for part in message.walk():
        filename = part.get_filename()
        if filename:
            attachments.append(filename)
        if part.get_content_type() == "text/plain" and not filename:
            try:
                lines.append(part.get_content())
            except Exception:
                payload = part.get_payload(decode=True) or b""
                lines.append(payload.decode(errors="replace"))
    return bounded("\n\n".join(lines), MAX_TEXT_CHARS_PER_FILE), {"attachments": attachments}


def inspect_notebook(path: Path) -> tuple[str, dict[str, Any]]:
    data = read_json(path)
    cells = data.get("cells") if isinstance(data, dict) else []
    lines: list[str] = []
    for index, cell in enumerate(cells or [], start=1):
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else str(source)
        lines.append(f"## Cell {index} ({cell.get('cell_type')})\n{source}")
        for output in cell.get("outputs") or []:
            if isinstance(output.get("text"), list):
                lines.append("".join(output["text"]))
            elif output.get("text"):
                lines.append(str(output["text"]))
            plain = (output.get("data") or {}).get("text/plain")
            if isinstance(plain, list):
                lines.append("".join(plain))
            elif plain:
                lines.append(str(plain))
    return bounded("\n\n".join(lines), MAX_TEXT_CHARS_PER_FILE), {
        "nbformat": data.get("nbformat"),
        "cell_count": len(cells or []),
        "executed_cell_count": sum(bool(c.get("outputs")) for c in cells or []),
    }


def inspect_text(path: Path) -> tuple[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return bounded(text, MAX_TEXT_CHARS_PER_FILE), {"characters": len(text), "lines": text.count("\n") + 1}


def inspect_file_semantics(path: Path) -> tuple[str, dict[str, Any], list[str]]:
    suffix = path.suffix.lower()
    try:
        if suffix in {".xlsx", ".xlsm"}:
            try:
                text, meta = inspect_spreadsheet(path)
            except Exception as exc:
                text, meta = recover_spreadsheet_xml(path)
                meta["primary_parser_error"] = f"{type(exc).__name__}: {exc}"
        elif suffix == ".docx":
            try:
                text, meta = inspect_docx(path)
            except Exception as exc:
                text, meta = recover_docx_xml(path)
                meta["primary_parser_error"] = f"{type(exc).__name__}: {exc}"
        elif suffix == ".pptx":
            text, meta = inspect_pptx(path)
        elif suffix == ".pdf":
            text, meta = inspect_pdf(path)
        elif suffix in IMAGE_EXTS:
            text, meta = inspect_image(path)
        elif suffix in AUDIO_EXTS or suffix in VIDEO_EXTS:
            text, meta = inspect_audio(path) if suffix in AUDIO_EXTS else ("Video frames and parsed audio are provided separately.", {})
        elif suffix == ".eml":
            text, meta = inspect_email(path)
        elif suffix == ".ipynb":
            text, meta = inspect_notebook(path)
        elif suffix in TEXT_EXTS or not suffix:
            text, meta = inspect_text(path)
        elif suffix == ".psd":
            text, meta = "Layered image; flattened visual evidence is provided when decoding succeeds.", {}
        elif suffix == ".dwg":
            text, meta = "DWG binary inventory only; no safe installed static renderer was available.", {}
        else:
            text, meta = "Binary file inventory only; no safe specialized parser was available.", {}
        return text, meta, []
    except Exception as exc:
        return "", {}, [f"{type(exc).__name__}: {exc}"]


def safe_extract_zip(path: Path, target: Path) -> tuple[list[Path], list[str]]:
    extracted: list[Path] = []
    errors: list[str] = []
    total = 0
    with zipfile.ZipFile(path) as zf:
        members = zf.infolist()
        if len(members) > MAX_ARCHIVE_FILES:
            raise EvaluationError(f"archive exceeds {MAX_ARCHIVE_FILES} entries: {path}")
        for member in members:
            pure = PurePosixPath(member.filename)
            if pure.is_absolute() or ".." in pure.parts:
                errors.append(f"unsafe archive member skipped: {member.filename}")
                continue
            if member.is_dir():
                continue
            total += member.file_size
            if total > MAX_ARCHIVE_BYTES:
                raise EvaluationError(f"archive exceeds {MAX_ARCHIVE_BYTES} expanded bytes: {path}")
            destination = target.joinpath(*pure.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(destination)
    return extracted, errors


def sanitized_pdf_copy(source: Path, target: Path) -> None:
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(source, strict=False)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata({"/Title": "", "/Author": "", "/Subject": "", "/Creator": "", "/Producer": ""})
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        writer.write(f)


def image_to_pdf(source: Path, target: Path) -> None:
    from PIL import Image, ImageSequence

    pages = []
    with Image.open(source) as image:
        for frame in ImageSequence.Iterator(image):
            page = frame.convert("RGB")
            page.info.clear()
            pages.append(page.copy())
    if not pages:
        raise EvaluationError(f"no decodable frames in {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(target, "PDF", save_all=True, append_images=pages[1:], resolution=150.0)


def render_office_to_pdf(source: Path, target: Path) -> None:
    soffice = shutil.which("soffice")
    if not soffice:
        raise EvaluationError("LibreOffice/soffice is unavailable")
    with tempfile.TemporaryDirectory(prefix="gdpval_lo_") as temp:
        root = Path(temp)
        profile = root / "profile"
        out = root / "out"
        cache = root / "cache"
        home = root / "home"
        temp_dir = root / "tmp"
        for directory in (out, cache, home, temp_dir):
            directory.mkdir()
        env = os.environ.copy()
        env.update({"XDG_CACHE_HOME": str(cache), "HOME": str(home), "TMPDIR": str(temp_dir)})
        local_source = root / ("input" + source.suffix.lower())
        shutil.copyfile(source, local_source)
        command = [
            soffice,
            "--headless",
            f"-env:UserInstallation={profile.as_uri()}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(out),
            str(local_source),
        ]
        run_checked(command, timeout=600, env=env)
        products = sorted(out.glob("*.pdf"))
        if len(products) != 1:
            raise EvaluationError(f"LibreOffice did not produce one PDF for {source}: {products}")
        sanitized_pdf_copy(products[0], target)


def render_html_to_pdf(source: Path, target: Path) -> bool:
    try:
        render_office_to_pdf(source, target)
        return True
    except Exception:
        pass
    try:
        from weasyprint import HTML

        HTML(filename=str(source), base_url=str(source.parent)).write_pdf(str(target))
        sanitized_pdf_copy(target, target.with_suffix(".clean.pdf"))
        target.with_suffix(".clean.pdf").replace(target)
        return True
    except Exception:
        return False


def render_docx_reconstruction(source: Path, target: Path) -> None:
    """Render a structurally reconstructed DOCX when LibreOffice rejects a valid package.

    The original semantic extraction remains authoritative. This fallback
    preserves paragraph/table/header/footer order in a neutral document while
    explicitly avoiding a false claim that original styling was recovered.
    """
    import docx

    original = docx.Document(source)
    reconstructed = docx.Document()
    for paragraph in original.paragraphs:
        reconstructed.add_paragraph(paragraph.text)
    for table in original.tables:
        rows = max(1, len(table.rows))
        cols = max(1, len(table.columns))
        new_table = reconstructed.add_table(rows=rows, cols=cols)
        for row_idx, row in enumerate(table.rows):
            for col_idx, cell in enumerate(row.cells[:cols]):
                new_table.cell(row_idx, col_idx).text = cell.text
    for section_idx, section in enumerate(original.sections):
        if section_idx >= len(reconstructed.sections):
            reconstructed.add_section()
        target_section = reconstructed.sections[min(section_idx, len(reconstructed.sections) - 1)]
        target_section.header.paragraphs[0].text = "\n".join(p.text for p in section.header.paragraphs)
        target_section.footer.paragraphs[0].text = "\n".join(p.text for p in section.footer.paragraphs)
    with tempfile.TemporaryDirectory(prefix="gdpval_docx_rebuild_") as temp:
        rebuilt = Path(temp) / "reconstructed.docx"
        reconstructed.save(rebuilt)
        render_office_to_pdf(rebuilt, target)


def render_docx_recovered_xml(source: Path, target: Path) -> None:
    import docx

    text, _ = recover_docx_xml(source)
    reconstructed = docx.Document()
    for paragraph in text.split("\n\n"):
        reconstructed.add_paragraph(paragraph)
    with tempfile.TemporaryDirectory(prefix="gdpval_docx_xml_") as temp:
        rebuilt = Path(temp) / "recovered.docx"
        reconstructed.save(rebuilt)
        render_office_to_pdf(rebuilt, target)


def render_pptx_reconstruction(source: Path, target: Path) -> None:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    from pptx.util import Inches, Pt

    original = Presentation(source)
    rebuilt = Presentation()
    rebuilt.slide_width = original.slide_width
    rebuilt.slide_height = original.slide_height
    while rebuilt.slides:
        rel = rebuilt.slides._sldIdLst[0].rId
        rebuilt.part.drop_rel(rel)
        del rebuilt.slides._sldIdLst[0]
    blank_layout = rebuilt.slide_layouts[6]
    with tempfile.TemporaryDirectory(prefix="gdpval_pptx_rebuild_") as temp:
        temp_root = Path(temp)
        for slide_index, slide in enumerate(original.slides, start=1):
            new_slide = rebuilt.slides.add_slide(blank_layout)
            fallback_lines: list[str] = []
            for shape_index, shape in enumerate(slide.shapes, start=1):
                try:
                    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                        image = shape.image
                        image_path = temp_root / f"s{slide_index:03d}_{shape_index:03d}.{image.ext}"
                        image_path.write_bytes(image.blob)
                        new_slide.shapes.add_picture(
                            str(image_path), shape.left, shape.top, width=shape.width, height=shape.height
                        )
                        continue
                except Exception:
                    pass
                if getattr(shape, "has_text_frame", False) and shape.text.strip():
                    try:
                        box = new_slide.shapes.add_textbox(shape.left, shape.top, shape.width, shape.height)
                        box.text_frame.text = shape.text
                        for paragraph in box.text_frame.paragraphs:
                            for run in paragraph.runs:
                                run.font.size = Pt(14)
                    except Exception:
                        fallback_lines.append(shape.text)
                elif getattr(shape, "has_table", False):
                    fallback_lines.extend(" | ".join(cell.text for cell in row.cells) for row in shape.table.rows)
            if fallback_lines:
                box = new_slide.shapes.add_textbox(Inches(0.5), Inches(0.5), rebuilt.slide_width - Inches(1), rebuilt.slide_height - Inches(1))
                box.text_frame.text = "\n".join(fallback_lines)
        rebuilt_path = temp_root / "reconstructed.pptx"
        rebuilt.save(rebuilt_path)
        render_office_to_pdf(rebuilt_path, target)


def extract_video(source: Path, target_dir: Path) -> tuple[list[Path], Path | None, dict[str, Any]]:
    ffmpeg = ffmpeg_executable()
    if not ffmpeg:
        raise EvaluationError("ffmpeg is required for video evidence")
    target_dir.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(source)], capture_output=True, text=True, check=False)
    banner = probe.stderr or ""
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", banner)
    duration = None
    if duration_match:
        h, m, s = duration_match.groups()
        duration = int(h) * 3600 + int(m) * 60 + float(s)
    uniform_count = 12
    interval = max(1.0, (duration or 60.0) / uniform_count)
    uniform_pattern = target_dir / "uniform_%03d.png"
    run_checked(
        [ffmpeg, "-y", "-i", str(source), "-vf", f"fps=1/{interval},scale=1280:-2", "-frames:v", str(uniform_count), str(uniform_pattern)],
        timeout=900,
    )
    scene_pattern = target_dir / "scene_%03d.png"
    scene = subprocess.run(
        [ffmpeg, "-y", "-i", str(source), "-vf", "select='gt(scene,0.35)',scale=1280:-2", "-vsync", "vfr", "-frames:v", "12", str(scene_pattern)],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    frames = sorted(target_dir.glob("uniform_*.png")) + sorted(target_dir.glob("scene_*.png"))
    unique: list[Path] = []
    hashes: set[str] = set()
    for frame in frames:
        digest = sha256_file(frame)
        if digest not in hashes:
            hashes.add(digest)
            unique.append(frame)
    audio = target_dir / "audio.wav"
    audio_run = subprocess.run(
        [ffmpeg, "-y", "-i", str(source), "-vn", "-ar", "16000", "-ac", "1", str(audio)],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if audio_run.returncode or not audio.is_file() or audio.stat().st_size <= 44:
        audio = None
    return unique[:24], audio, {"duration_seconds": duration, "uniform_frames": uniform_count, "scene_frames": len(list(target_dir.glob('scene_*.png'))), "scene_command_returncode": scene.returncode}


def merge_pdfs(sources: Sequence[Path], target: Path) -> None:
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for source in sources:
        reader = PdfReader(source, strict=False)
        for page in reader.pages:
            writer.add_page(page)
    writer.add_metadata({"/Title": "", "/Author": "", "/Subject": "", "/Creator": "", "/Producer": ""})
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        writer.write(f)


def representative_page_indices(page_count: int, limit: int = MAX_VISUAL_PAGES_PER_BUNDLE) -> list[int]:
    if page_count <= limit:
        return list(range(page_count))
    anchors = {0, 1, 2, 3, page_count - 4, page_count - 3, page_count - 2, page_count - 1}
    selected = {index for index in anchors if 0 <= index < page_count}
    # Greedily add the page furthest from the current selection. This retains
    # the beginning/end and spreads the rest over the complete document.
    while len(selected) < limit:
        candidate = max(
            (index for index in range(page_count) if index not in selected),
            key=lambda index: (min(abs(index - chosen) for chosen in selected), -index),
        )
        selected.add(candidate)
    return sorted(selected)


def sample_pdf_pages(source: Path, target: Path, *, limit: int = MAX_VISUAL_PAGES_PER_BUNDLE) -> dict[str, Any]:
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(source, strict=False)
    indices = representative_page_indices(len(reader.pages), limit)
    writer = PdfWriter()
    for index in indices:
        writer.add_page(reader.pages[index])
    writer.add_metadata({"/Title": "", "/Author": "", "/Subject": "", "/Creator": "", "/Producer": ""})
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        writer.write(handle)
    return {
        "original_pages": len(reader.pages),
        "retained_pages": len(indices),
        "retained_page_numbers_1_based": [index + 1 for index in indices],
        "rule": "first/last anchors plus deterministic farthest-point coverage over the full PDF",
        "semantic_extraction_retained_under_frozen_text_caps": True,
    }


def file_label(index: int, path: Path) -> str:
    # A basename can be task-substantive (the rubric may require it).  Paths,
    # directory roots, and author metadata are never exposed.
    name = path.name
    if re.search(r"(?i)(?:^|[\s_.+()-])(?:human|human[-_+ ]?ai|agent|chatgpt|claude|gpt)(?:$|[\s_.+()-])", name):
        name = f"anonymous_file_{index}{path.suffix.lower()}"
    return f"File {index}: {name}"


@dataclass
class BundleBuild:
    bundle_id: str
    source_hash: str
    text_path: Path
    visual_path: Path | None
    audit_path: Path
    parser_errors: list[dict[str, Any]]
    media_pending: list[dict[str, Any]]


def bundle_source_hash(files: Sequence[dict[str, Any]]) -> str:
    canonical = [(str(item["sha256"]), str(item["name"]), int(item["size_bytes"])) for item in files]
    return sha256_text(json.dumps(canonical, sort_keys=True, separators=(",", ":")))


def build_evidence_bundle(
    bundle_id: str,
    files: Sequence[dict[str, Any]],
    evidence_root: Path,
    media_cache: dict[str, Any],
) -> BundleBuild:
    source_hash = bundle_source_hash(files)
    bundle_dir = evidence_root / source_hash
    audit_path = bundle_dir / "audit.json"
    if audit_path.is_file():
        audit = read_json(audit_path)
        text_path = bundle_dir / "evidence.txt"
        visual = bundle_dir / "visual.pdf"
        pending_keys = {str(item.get("cache_key") or "") for item in audit.get("media_pending") or []}
        pending_now_resolved = bool(pending_keys) and pending_keys.issubset(media_cache)
        pending_paths_missing = any(
            not Path(str(item.get("audio_path") or "")).is_file()
            for item in audit.get("media_pending") or []
            if str(item.get("cache_key") or "") not in media_cache
        )
        if (
            not pending_now_resolved
            and not pending_paths_missing
            and not audit.get("parser_errors")
            and text_path.is_file()
            and (not audit.get("visual_expected") or visual.is_file())
        ):
            return BundleBuild(
                bundle_id,
                source_hash,
                text_path,
                visual if visual.is_file() else None,
                audit_path,
                list(audit.get("parser_errors") or []),
                list(audit.get("media_pending") or []),
            )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    derived = bundle_dir / "derived"
    if derived.exists():
        shutil.rmtree(derived)
    derived.mkdir()
    sections: list[str] = [
        "# Neutral deliverable evidence",
        "The following is a source-blind semantic and structural extraction. Original directory roots and author metadata were removed.",
    ]
    visual_parts: list[Path] = []
    parser_errors: list[dict[str, Any]] = []
    quality_defects: list[dict[str, Any]] = []
    media_pending: list[dict[str, Any]] = []
    file_audits: list[dict[str, Any]] = []
    expanded: list[tuple[Path, str, str]] = []
    temp_archives: list[tempfile.TemporaryDirectory[str]] = []
    agent_bundle = bundle_id.startswith("agent:")

    def record_quality_defect(
        *,
        label: str,
        kind: str,
        message: str,
    ) -> None:
        if not any(
            item["label"] == label and item["kind"] == kind
            for item in quality_defects
        ):
            quality_defects.append(
                {
                    "label": label,
                    "kind": kind,
                    "message": message,
                }
            )
            sections.append(
                f"Submitted-file quality defect for {label}: {message}"
            )

    try:
        for file_index, item in enumerate(files, start=1):
            source = Path(str(item["path"]))
            if not source.is_file():
                parser_errors.append({"file_index": file_index, "kind": "source_missing", "message": "declared source file is absent"})
                continue
            if source.name.startswith("~$") or source.name in {".DS_Store", "Thumbs.db"} or source.name.startswith("._"):
                file_audits.append(
                    {
                        "file_index": file_index,
                        "kind": "excluded_provenance_or_lockfile",
                        "source_sha256": item["sha256"],
                        "reason": "Office lockfile or operating-system metadata is not substantive deliverable evidence",
                    }
                )
                continue
            if source.suffix.lower() in ARCHIVE_EXTS:
                temp = tempfile.TemporaryDirectory(prefix="gdpval_zip_")
                temp_archives.append(temp)
                try:
                    children, errors = safe_extract_zip(source, Path(temp.name))
                    sections.append(f"\n# {file_label(file_index, source)}\nArchive entries: {len(children)}")
                    sections.extend(errors)
                    for child in children:
                        child_label = f"{file_label(file_index, source)} / {file_label(file_index, child)}"
                        expanded.append((child, child_label, sha256_file(child)))
                    file_audits.append({"file_index": file_index, "kind": "archive", "entries": len(children), "errors": errors})
                except Exception as exc:
                    parser_errors.append({"file_index": file_index, "kind": "archive", "message": str(exc)})
            else:
                expanded.append((source, file_label(file_index, source), item["sha256"]))

        for expanded_index, (source, label, parent_sha) in enumerate(expanded, start=1):
            suffix = source.suffix.lower()
            text, metadata, errors = inspect_file_semantics(source)
            sections.append(f"\n# {label}\nType: {suffix or '[no extension]'}\nMetadata: {json.dumps(metadata, ensure_ascii=False)}")
            if text:
                sections.append(text)
            if errors:
                semantic_error = "; ".join(errors)
                if suffix in VIDEO_EXTS | AUDIO_EXTS and is_corrupt_media_error(
                    semantic_error
                ):
                    record_quality_defect(
                        label=label,
                        kind="corrupt_submitted_media",
                        message=(
                            "The submitted media file is corrupt/unreadable "
                            "(the decoder could not parse a valid media stream)."
                        ),
                    )
                elif (
                    agent_bundle
                    and (
                        invalid_reason := invalid_submitted_document_reason(
                            source
                        )
                    )
                ):
                    record_quality_defect(
                        label=label,
                        kind="invalid_submitted_document",
                        message=(
                            "The submitted document is corrupt, truncated, or "
                            f"mislabeled: {invalid_reason}."
                        ),
                    )
                else:
                    parser_errors.append(
                        {
                            "label": label,
                            "kind": "semantic",
                            "message": semantic_error,
                        }
                    )
            derived_pdf = derived / f"{expanded_index:04d}.pdf"
            visual_fallback: str | None = None
            try:
                if suffix == ".pdf":
                    sanitized_pdf_copy(source, derived_pdf)
                    visual_parts.append(derived_pdf)
                elif suffix in OFFICE_EXTS:
                    try:
                        render_office_to_pdf(source, derived_pdf)
                    except Exception:
                        if suffix == ".docx":
                            try:
                                render_docx_reconstruction(source, derived_pdf)
                                visual_fallback = "DOCX structure rendered with the reconstruction adapter"
                            except Exception:
                                render_docx_recovered_xml(source, derived_pdf)
                                visual_fallback = "DOCX XML rendered with the recovery adapter"
                            sections.append(f"Rendering method for {label}: {visual_fallback}.")
                        elif suffix == ".pptx":
                            render_pptx_reconstruction(source, derived_pdf)
                            visual_fallback = "PPTX reconstruction preserving slide order, text, tables, and decodable pictures"
                            sections.append(f"Rendering method for {label}: {visual_fallback}.")
                        else:
                            raise
                    visual_parts.append(derived_pdf)
                elif suffix in IMAGE_EXTS:
                    image_to_pdf(source, derived_pdf)
                    visual_parts.append(derived_pdf)
                elif suffix in {".html", ".htm"} and render_html_to_pdf(source, derived_pdf):
                    visual_parts.append(derived_pdf)
                elif suffix == ".psd":
                    try:
                        from psd_tools import PSDImage

                        image = PSDImage.open(source).composite()
                        preview = derived / f"{expanded_index:04d}.png"
                        image.save(preview)
                        image_to_pdf(preview, derived_pdf)
                        visual_parts.append(derived_pdf)
                    except Exception as exc:
                        parser_errors.append({"label": label, "kind": "psd_render", "message": str(exc)})
                elif suffix in VIDEO_EXTS:
                    media_dir = derived / f"video_{expanded_index:04d}"
                    frames, audio, video_meta = extract_video(source, media_dir)
                    sections.append("Video sampling: " + json.dumps(video_meta, ensure_ascii=False))
                    if frames:
                        frame_pdfs: list[Path] = []
                        for frame_index, frame in enumerate(frames, start=1):
                            frame_pdf = media_dir / f"frame_{frame_index:03d}.pdf"
                            image_to_pdf(frame, frame_pdf)
                            frame_pdfs.append(frame_pdf)
                        merge_pdfs(frame_pdfs, derived_pdf)
                        visual_parts.append(derived_pdf)
                    if audio:
                        audio_sha = sha256_file(audio)
                        parsed = media_cache.get(parent_sha) or media_cache.get(audio_sha)
                        if parsed:
                            sections.append("## Source-blind parsed audio\n" + json.dumps(parsed, ensure_ascii=False, indent=2))
                        else:
                            media_pending.append({"source_path": str(source), "audio_path": str(audio), "cache_key": parent_sha, "kind": "video_audio"})
                elif suffix in AUDIO_EXTS:
                    parsed = media_cache.get(parent_sha)
                    if parsed:
                        sections.append("## Source-blind parsed audio\n" + json.dumps(parsed, ensure_ascii=False, indent=2))
                    else:
                        persistent_audio = derived / f"audio_{expanded_index:04d}.wav"
                        sanitize_audio(source, persistent_audio)
                        media_pending.append(
                            {
                                "source_path": str(source),
                                "audio_path": str(persistent_audio),
                                "cache_key": parent_sha,
                                "kind": "audio",
                            }
                        )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                if (
                    suffix in VIDEO_EXTS | AUDIO_EXTS
                    and is_corrupt_media_error(message)
                ):
                    record_quality_defect(
                        label=label,
                        kind="corrupt_submitted_media",
                        message=(
                            "The submitted media file is corrupt/unreadable "
                            "(the decoder could not parse a valid media stream)."
                        ),
                    )
                elif (
                    agent_bundle
                    and (
                        invalid_reason := invalid_submitted_document_reason(
                            source
                        )
                    )
                ):
                    record_quality_defect(
                        label=label,
                        kind="invalid_submitted_document",
                        message=(
                            "The submitted document is corrupt, truncated, or "
                            f"mislabeled: {invalid_reason}."
                        ),
                    )
                else:
                    parser_errors.append({"label": label, "kind": "visual", "message": message})
            file_audits.append(
                {
                    "label": label,
                    "suffix": suffix,
                    "source_sha256": parent_sha,
                    "semantic_chars": len(text),
                    "metadata": metadata,
                    "semantic_errors": errors,
                    "visual_rendered": derived_pdf.is_file(),
                    "visual_fallback": visual_fallback,
                }
            )
    finally:
        for temp in temp_archives:
            temp.cleanup()

    visual_path = bundle_dir / "visual.pdf"
    visual_page_sampling: dict[str, Any] | None = None
    if visual_parts:
        full_visual = derived / "complete_visual.pdf"
        merge_pdfs(visual_parts, full_visual)
        from pypdf import PdfReader

        full_page_count = len(PdfReader(full_visual, strict=False).pages)
        if full_page_count > MAX_VISUAL_PAGES_PER_BUNDLE:
            visual_page_sampling = sample_pdf_pages(full_visual, visual_path)
            sections.append(
                "## Document processing\n"
                f"Selected {MAX_VISUAL_PAGES_PER_BUNDLE} of {full_page_count} rendered pages using "
                "deterministic beginning/end anchors and coverage across the full PDF. "
                "Text extraction uses the configured per-file and per-bundle character caps."
            )
        else:
            shutil.copyfile(full_visual, visual_path)
        if visual_path.stat().st_size > MAX_UPLOAD_BYTES:
            parser_errors.append(
                {
                    "kind": "visual_size",
                    "message": f"merged visual PDF is {visual_path.stat().st_size} bytes; exceeds upload limit",
                }
            )
    evidence_text = bounded("\n\n".join(sections), MAX_BUNDLE_TEXT_CHARS)
    # Some PDF text maps contain lone UTF-16 surrogate code points. They are
    # not valid Unicode scalar values and cannot be emitted as UTF-8. Preserve
    # the surrounding extracted evidence while marking only those invalid code
    # points with the standard replacement character.
    evidence_text = "".join(
        "\ufffd" if 0xD800 <= ord(character) <= 0xDFFF else character
        for character in evidence_text
    )
    text_path = bundle_dir / "evidence.txt"
    text_path.write_text(evidence_text, encoding="utf-8")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "source_hash": source_hash,
        "created_at": utc_now(),
        "source_files": list(files),
        "file_audits": file_audits,
        "parser_errors": parser_errors,
        "quality_defects": quality_defects,
        "media_pending": media_pending,
        "visual_expected": bool(visual_parts),
        "visual_pages_source_count": len(visual_parts),
        "visual_page_sampling": visual_page_sampling,
        "text_sha256": sha256_file(text_path),
        "visual_sha256": sha256_file(visual_path) if visual_path.is_file() else None,
    }
    write_json(audit_path, audit)
    return BundleBuild(bundle_id, source_hash, text_path, visual_path if visual_path.is_file() else None, audit_path, parser_errors, media_pending)


def unique_bundle_specs(manifest: Sequence[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    specs: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    for row in manifest:
        for role, files in (
            ("agent", row["agent_files"]),
            (row["baseline"], row["baseline_files"]),
            ("references", row["reference_files"]),
        ):
            source_hash = bundle_source_hash(files)
            specs.setdefault(source_hash, (f"{role}:{row['task_id']}:{source_hash[:12]}", files))
    return [specs[key] for key in sorted(specs)]


def _bundle_worker(args: tuple[str, list[dict[str, Any]], str, dict[str, Any]]) -> dict[str, Any]:
    bundle_id, files, evidence_root_raw, media_cache = args
    built = build_evidence_bundle(bundle_id, files, Path(evidence_root_raw), media_cache)
    return {
        "bundle_id": built.bundle_id,
        "source_hash": built.source_hash,
        "text_path": str(built.text_path),
        "visual_path": str(built.visual_path) if built.visual_path else None,
        "audit_path": str(built.audit_path),
        "parser_errors": built.parser_errors,
        "media_pending": built.media_pending,
    }


def prepare_evidence(
    state: Path,
    *,
    only_hashes: set[str] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    manifest_path = state / "comparison_manifest.unblinded.jsonl"
    manifest = list(iter_jsonl(manifest_path))
    validate_manifest(
        manifest,
        expected_tasks=expected_task_count(state),
        repeat_count=repeat_count_for_state(state),
        conditions=active_conditions_for_state(state),
    )
    evidence_root = state / "evidence"
    media_cache_path = state / "media_parser_cache.json"
    media_cache = read_json(media_cache_path) if media_cache_path.is_file() else {}
    specs = unique_bundle_specs(manifest)
    expected_hashes = {bundle_source_hash(files) for _, files in specs}
    selected = [
        (bundle_id, files)
        for bundle_id, files in specs
        if only_hashes is None or bundle_source_hash(files) in only_hashes
    ]
    jobs = [(bundle_id, files, str(evidence_root), media_cache) for bundle_id, files in selected]
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for index, job in enumerate(jobs, start=1):
            results.append(_bundle_worker(job))
            if index % 25 == 0 or index == len(jobs):
                print(f"prepared evidence {index}/{len(jobs)}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_bundle_worker, job): job[0] for job in jobs}
            for index, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
                try:
                    results.append(future.result())
                except Exception as exc:
                    raise EvaluationError(f"evidence worker failed for {future_map[future]}: {exc}") from exc
                if index % 25 == 0 or index == len(jobs):
                    print(f"prepared evidence {index}/{len(jobs)} with {workers} workers", flush=True)
    index_path = state / "evidence_index.json"
    previous = read_json(index_path) if index_path.is_file() else []
    # Drop bundles orphaned by a frozen-manifest correction. Content-addressed
    # files may remain on disk for auditability, but they are not uploaded or
    # counted as evidence for the final evaluation.
    merged = {
        r["source_hash"]: r
        for r in previous
        if isinstance(r, dict) and str(r.get("source_hash")) in expected_hashes
    }
    merged.update({r["source_hash"]: r for r in results})
    final = [merged[key] for key in sorted(merged)]
    write_json(index_path, final)
    summary = {
        "created_at": utc_now(),
        "unique_bundles_expected": len(specs),
        "unique_bundles_prepared": len(final),
        "parser_error_bundles": sum(bool(r.get("parser_errors")) for r in final),
        "media_pending_items": sum(len(r.get("media_pending") or []) for r in final),
    }
    write_json(state / "evidence_summary.json", summary)
    return summary


def cap_existing_visuals(
    state: Path, *, max_pages: int = MAX_VISUAL_PAGES_PER_BUNDLE
) -> dict[str, Any]:
    """Apply the frozen context-budget rule to already prepared bundles."""
    from pypdf import PdfReader

    evidence = read_json(state / "evidence_index.json")
    sampled: list[dict[str, Any]] = []
    for item in evidence:
        visual_raw = item.get("visual_path")
        if not visual_raw:
            continue
        visual = Path(str(visual_raw))
        audit_path = Path(str(item["audit_path"]))
        audit = read_json(audit_path)
        page_count = len(PdfReader(visual, strict=False).pages)
        if page_count <= max_pages:
            continue
        temporary = visual.with_suffix(".sampled.tmp.pdf")
        sampling = sample_pdf_pages(visual, temporary, limit=max_pages)
        temporary.replace(visual)
        text_path = Path(str(item["text_path"]))
        marker = "## Document processing"
        processing_note = (
            f"{marker}\nSelected {max_pages} of {page_count} rendered pages using deterministic "
            "beginning/end anchors and coverage across the full PDF. "
            "Text extraction uses the configured per-file and per-bundle character caps.\n\n"
        )
        existing_text = text_path.read_text(encoding="utf-8", errors="replace")
        if marker not in existing_text:
            text_path.write_text(bounded(processing_note + existing_text, MAX_BUNDLE_TEXT_CHARS), encoding="utf-8")
        audit["visual_page_sampling"] = sampling
        audit["text_sha256"] = sha256_file(text_path)
        audit["visual_sha256"] = sha256_file(visual)
        write_json(audit_path, audit)
        sampled.append({"source_hash": item["source_hash"], **sampling})
    result = {
        "created_at": utc_now(),
        "max_visual_pages_per_bundle": max_pages,
        "bundles_sampled": len(sampled),
        "sampling_records": sampled,
    }
    write_json(state / "visual_sampling_audit.json", result)
    return result


def select_evidence_pilot(state: Path, *, minimum: int = 30) -> dict[str, Any]:
    manifest = list(iter_jsonl(state / "comparison_manifest.unblinded.jsonl"))
    specs = unique_bundle_specs(manifest)
    records: dict[str, dict[str, Any]] = {}
    for bundle_id, files in specs:
        source_hash = bundle_source_hash(files)
        suffixes = {str(item.get("suffix") or "[none]") for item in files}
        features = {"ext:" + suffix for suffix in suffixes}
        if any(s in AUDIO_EXTS for s in suffixes):
            features.add("modality:audio")
        if any(s in VIDEO_EXTS for s in suffixes):
            features.add("modality:video")
        if any(s in ARCHIVE_EXTS for s in suffixes):
            features.add("modality:archive")
        if any(s in OFFICE_EXTS for s in suffixes):
            features.add("modality:office")
        if any(s == ".psd" for s in suffixes):
            features.add("modality:layered_image")
        if not files:
            features.add("edge:missing_submission")
        records[source_hash] = {
            "bundle_id": bundle_id,
            "source_hash": source_hash,
            "features": sorted(features),
            "suffixes": sorted(suffixes),
            "source_bytes": sum(int(item.get("size_bytes") or 0) for item in files),
            "file_count": len(files),
        }
    uncovered = set().union(*(set(r["features"]) for r in records.values()))
    remaining = set(records)
    selected: list[str] = []
    while uncovered and remaining:
        best = max(
            remaining,
            key=lambda key: (len(set(records[key]["features"]) & uncovered), records[key]["source_bytes"], key),
        )
        covered = set(records[best]["features"]) & uncovered
        if not covered:
            break
        selected.append(best)
        uncovered -= covered
        remaining.remove(best)
    # Explicitly retain the largest bundle for each observed extension and the
    # largest bundles overall, even if another bundle already covered its type.
    extensions = sorted({suffix for r in records.values() for suffix in r["suffixes"]})
    for suffix in extensions:
        selected.append(max((key for key, r in records.items() if suffix in r["suffixes"]), key=lambda key: records[key]["source_bytes"]))
    for key in sorted(records, key=lambda value: (-records[value]["source_bytes"], value))[:10]:
        selected.append(key)
    for key in sorted(remaining, key=lambda value: (-records[value]["source_bytes"], value)):
        if len(set(selected)) >= minimum:
            break
        selected.append(key)
    selected = sorted(set(selected))
    result = {
        "created_at": utc_now(),
        "selected_count": len(selected),
        "unique_bundles": len(records),
        "uncovered_features": sorted(uncovered),
        "bundles": [records[key] for key in selected],
    }
    write_json(state / "evidence_pilot_selection.json", result)
    return result


def prepare_evidence_pilot(state: Path, *, workers: int = 1) -> dict[str, Any]:
    selection_path = state / "evidence_pilot_selection.json"
    selection = read_json(selection_path) if selection_path.is_file() else select_evidence_pilot(state)
    hashes = {str(item["source_hash"]) for item in selection["bundles"]}
    return prepare_evidence(state, only_hashes=hashes, workers=workers)


class OpenAIAPI:
    """Small stdlib-only OpenAI client; never logs authorization headers."""

    def __init__(self, *, api_key: str | None = None, timeout: int = 900) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        if not self.api_key:
            raise EvaluationError("OPENAI_API_KEY is not set in the process environment")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout
        self.ssl_context = ssl.create_default_context()

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        body: bytes | None = None,
        content_type: str | None = "application/json",
        retries: int = 6,
    ) -> tuple[bytes, dict[str, str]]:
        url = endpoint if endpoint.startswith("http") else self.base_url + endpoint
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if content_type:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout, context=self.ssl_context) as response:
                    return response.read(), dict(response.headers.items())
            except urllib.error.HTTPError as exc:
                payload = exc.read().decode("utf-8", errors="replace")
                transient = exc.code in {408, 409, 429, 500, 502, 503, 504}
                if exc.code == 500 and "model produced invalid content" in payload.lower():
                    transient = False
                if not transient or attempt == retries:
                    raise EvaluationError(f"OpenAI HTTP {exc.code} for {endpoint}: {payload[:2000]}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == retries:
                    raise EvaluationError(f"OpenAI request failed for {endpoint}: {exc}") from exc
            time.sleep(min(60.0, (2**attempt) + random.random()))
        raise AssertionError("unreachable")

    def json(self, method: str, endpoint: str, value: Any | None = None) -> dict[str, Any]:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8") if value is not None else None
        raw, _ = self._request(method, endpoint, body=body)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"OpenAI returned non-JSON for {endpoint}: {raw[:1000]!r}") from exc
        if not isinstance(parsed, dict):
            raise EvaluationError(f"OpenAI returned non-object for {endpoint}")
        return parsed

    def upload_file(self, path: Path, *, purpose: str) -> dict[str, Any]:
        boundary = "----gdpval" + uuid.uuid4().hex
        name = path.name
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        chunks = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\n{purpose}\r\n".encode(),
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                f"filename=\"{name}\"\r\nContent-Type: {mime}\r\n\r\n"
            ).encode(),
            path.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
        raw, _ = self._request(
            "POST",
            "/files",
            body=b"".join(chunks),
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or not parsed.get("id"):
            raise EvaluationError(f"file upload returned unexpected payload: {parsed}")
        return parsed

    def transcribe(self, path: Path, *, model: str = "gpt-4o-mini-transcribe") -> dict[str, Any]:
        boundary = "----gdpval" + uuid.uuid4().hex
        mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
        chunks = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\njson\r\n".encode(),
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                f"filename=\"audio.wav\"\r\nContent-Type: {mime}\r\n\r\n"
            ).encode(),
            path.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
        raw, _ = self._request(
            "POST",
            "/audio/transcriptions",
            body=b"".join(chunks),
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise EvaluationError("audio transcription returned a non-object")
        return parsed

    def download_file(self, file_id: str) -> bytes:
        raw, _ = self._request("GET", f"/files/{file_id}/content", content_type=None)
        return raw


def extract_json_from_text(value: str) -> dict[str, Any]:
    stripped = value.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        try:
            parsed, _ = decoder.raw_decode(stripped[match.start() :])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    raise EvaluationError("model output did not contain a JSON object")


def response_output_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return str(response["output_text"])
    chunks: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            if isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "\n".join(chunks)


def sanitize_audio(source: Path, target: Path) -> None:
    ffmpeg = ffmpeg_executable()
    if not ffmpeg:
        raise EvaluationError("ffmpeg is required to sanitize audio")
    target.parent.mkdir(parents=True, exist_ok=True)
    run_checked([ffmpeg, "-y", "-i", str(source), "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(target)], timeout=1800)


def split_audio(source: Path, target_dir: Path, *, seconds: int = 600) -> list[Path]:
    ffmpeg = ffmpeg_executable()
    if not ffmpeg:
        raise EvaluationError("ffmpeg is required to split audio")
    target_dir.mkdir(parents=True, exist_ok=True)
    pattern = target_dir / "chunk_%03d.wav"
    run_checked(
        [ffmpeg, "-y", "-i", str(source), "-f", "segment", "-segment_time", str(seconds), "-c", "copy", str(pattern)],
        timeout=1800,
    )
    chunks = sorted(target_dir.glob("chunk_*.wav"))
    if not chunks:
        raise EvaluationError(f"audio splitting produced no chunks for {source}")
    return chunks


def parse_audio_chunk(api: OpenAIAPI, path: Path, *, model: str) -> dict[str, Any]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    prompt = (
        "Analyze this audio as neutral evidence for a blinded professional-deliverable evaluation. "
        "Do not infer who created it. Return one JSON object with keys: transcript (verbatim speech, "
        "or an empty string), nonverbal_description (music, sound effects, ambience, silence, timing, "
        "delivery, and audible production defects), language, and processing_notes (array of strings describing extraction methods). "
        "Be factual and source-blind."
    )
    payload = {
        "model": model,
        "modalities": ["text"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}},
                ],
            }
        ],
    }
    failed_attempts: list[dict[str, Any]] = []
    last_response: dict[str, Any] = {}
    last_content = ""
    for attempt in range(1, 4):
        response = api.json("POST", "/chat/completions", payload)
        last_response = response
        choices = response.get("choices") or []
        if not choices:
            failed_attempts.append(
                {"attempt": attempt, "response_id": response.get("id"), "usage": response.get("usage"), "failure": "no choices"}
            )
            continue
        content = str(choices[0].get("message", {}).get("content") or "")
        last_content = content
        try:
            parsed = extract_json_from_text(content)
        except EvaluationError as exc:
            failed_attempts.append(
                {
                    "attempt": attempt,
                    "response_id": response.get("id"),
                    "model": response.get("model"),
                    "usage": response.get("usage"),
                    "failure": str(exc),
                    "output_sha256": sha256_text(content),
                }
            )
            continue
        parsed["api_metadata"] = {
            "model_requested": model,
            "model_returned": response.get("model"),
            "response_id": response.get("id"),
            "usage": response.get("usage"),
            "attempts": attempt,
            "failed_attempts": failed_attempts,
        }
        return parsed
    if last_content.strip():
        return {
            "transcript": "",
            "nonverbal_description": last_content.strip(),
            "language": "undetermined",
            "processing_notes": [
                "Extraction format: model prose, preserved in nonverbal_description.",
            ],
            "api_metadata": {
                "model_requested": model,
                "model_returned": last_response.get("model"),
                "response_id": last_response.get("id"),
                "usage": last_response.get("usage"),
                "attempts": 3,
                "failed_attempts": failed_attempts,
                "unstructured_fallback": True,
            },
        }
    raise EvaluationError(f"audio parser returned no usable content after three identical-prompt attempts: {failed_attempts}")


def parse_audio_chunk_adaptive(
    api: OpenAIAPI,
    path: Path,
    *,
    model: str,
    work_dir: Path,
    fallback_seconds: Sequence[int] = (120, 30),
) -> list[dict[str, Any]]:
    try:
        return [parse_audio_chunk(api, path, model=model)]
    except EvaluationError as exc:
        if not fallback_seconds:
            transcription = api.transcribe(path)
            return [
                {
                    "transcript": str(transcription.get("text") or ""),
                    "nonverbal_description": "",
                    "language": str(transcription.get("language") or "undetermined"),
                    "processing_notes": [
                        "Extraction method: speech-to-text transcription.",
                        "Evidence fields: transcript and language.",
                    ],
                    "api_metadata": {
                        "model_requested": model,
                        "primary_parser_error": str(exc),
                        "fallback_model": "gpt-4o-mini-transcribe",
                        "fallback_response_metadata": {
                            key: value for key, value in transcription.items() if key != "text"
                        },
                        "transcription_fallback": True,
                    },
                }
            ]
        seconds = int(fallback_seconds[0])
        pieces = split_audio(path, work_dir / f"split_{seconds}", seconds=seconds)
        parsed: list[dict[str, Any]] = []
        for piece_index, piece in enumerate(pieces):
            parsed.extend(
                parse_audio_chunk_adaptive(
                    api,
                    piece,
                    model=model,
                    work_dir=work_dir / f"piece_{piece_index:03d}",
                    fallback_seconds=fallback_seconds[1:],
                )
            )
        return parsed


def parse_pending_media(
    state: Path,
    *,
    model: str = DEFAULT_AUDIO_MODEL,
    limit: int | None = None,
) -> dict[str, Any]:
    index = read_json(state / "evidence_index.json")
    pending: dict[str, dict[str, Any]] = {}
    affected_hashes: set[str] = set()
    for bundle in index:
        for item in bundle.get("media_pending") or []:
            pending.setdefault(str(item["cache_key"]), item)
            affected_hashes.add(str(bundle["source_hash"]))
    cache_path = state / "media_parser_cache.json"
    cache = read_json(cache_path) if cache_path.is_file() else {}
    missing_all = {key: value for key, value in pending.items() if key not in cache}
    missing = dict(list(sorted(missing_all.items()))[:limit]) if limit else missing_all
    if not missing:
        return {"pending": len(pending), "newly_parsed": 0, "cached": len(cache), "remaining": len(missing_all)}
    api = OpenAIAPI()
    media_work = state / "media_work"
    for index_num, (cache_key, item) in enumerate(sorted(missing.items()), start=1):
        source = Path(str(item["audio_path"]))
        if not source.is_file():
            original = Path(str(item["source_path"]))
            if original.suffix.lower() in VIDEO_EXTS:
                _, extracted, _ = extract_video(original, media_work / cache_key / "video")
                if not extracted:
                    raise EvaluationError(f"no audio stream could be extracted from {original}")
                source = extracted
            else:
                source = original
        sanitized = media_work / cache_key / "audio.wav"
        sanitize_audio(source, sanitized)
        chunks = split_audio(sanitized, media_work / cache_key / "chunks")
        parsed_chunks: list[dict[str, Any]] = []
        for chunk_index, chunk in enumerate(chunks):
            parsed_chunks.extend(
                parse_audio_chunk_adaptive(
                    api,
                    chunk,
                    model=model,
                    work_dir=media_work / cache_key / "adaptive" / f"chunk_{chunk_index:03d}",
                )
            )
        cache[cache_key] = {
            "schema_version": SCHEMA_VERSION,
            "parser_model": model,
            "parsed_at": utc_now(),
            "chunk_count": len(parsed_chunks),
            "chunks": parsed_chunks,
        }
        write_json(cache_path, cache)
        print(f"parsed media {index_num}/{len(missing)}", flush=True)
    # Rebuild affected evidence so parser text becomes part of the frozen bundle.
    # Media parsing can resolve many independent evidence bundles at once.  Use
    # the same bounded local parallelism as the full evidence build; all writes
    # remain content-addressed and therefore idempotent.
    prepare_evidence(state, only_hashes=affected_hashes, workers=4)
    return {
        "pending": len(pending),
        "newly_parsed": len(missing),
        "cached": len(cache),
        "remaining": max(0, len(missing_all) - len(missing)),
    }


def neutral_upload_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def evidence_provenance_audit(state: Path) -> dict[str, Any]:
    from pypdf import PdfReader

    evidence = read_json(state / "evidence_index.json")
    text_hits: list[dict[str, str]] = []
    metadata_hits: list[dict[str, Any]] = []
    missing: list[str] = []
    for item in evidence:
        source_hash = str(item["source_hash"])
        text_path = Path(str(item["text_path"]))
        if not text_path.is_file():
            missing.append(str(text_path))
            continue
        evidence_text = text_path.read_text(encoding="utf-8", errors="replace")
        for label, marker in provenance_markers_for_state(state).items():
            if marker in evidence_text:
                text_hits.append({"source_hash": source_hash, "marker": label})
        visual_raw = item.get("visual_path")
        if visual_raw:
            visual = Path(str(visual_raw))
            if not visual.is_file():
                missing.append(str(visual))
                continue
            metadata = PdfReader(visual, strict=False).metadata or {}
            nonempty = {
                key: str(metadata.get(key))
                for key in ("/Title", "/Author", "/Subject", "/Creator", "/Producer")
                if str(metadata.get(key) or "").strip()
            }
            if nonempty:
                metadata_hits.append({"source_hash": source_hash, "metadata": nonempty})
    result = {
        "created_at": utc_now(),
        "bundles_scanned": len(evidence),
        "text_provenance_hits": text_hits,
        "visual_metadata_hits": metadata_hits,
        "missing_derived_files": missing,
        "passed": not text_hits and not metadata_hits and not missing,
    }
    write_json(state / "evidence_provenance_audit.json", result)
    return result


def media_parser_accounting(state: Path) -> dict[str, Any]:
    expected_count = expected_comparison_count(state)
    cache_path = state / "media_parser_cache.json"
    cache = read_json(cache_path) if cache_path.is_file() else {}
    prompt_tokens = completion_tokens = prompt_audio_tokens = completion_audio_tokens = 0
    recorded_responses = transcription_fallbacks = unstructured_fallbacks = 0
    chunk_records = 0
    parser_providers: Counter[str] = Counter()
    google_prompt_tokens = google_candidate_tokens = google_total_tokens = 0

    def add_usage(usage: dict[str, Any] | None) -> None:
        nonlocal prompt_tokens, completion_tokens, prompt_audio_tokens, completion_audio_tokens
        nonlocal google_prompt_tokens, google_candidate_tokens, google_total_tokens
        if not isinstance(usage, dict):
            return
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        prompt_audio_tokens += int((usage.get("prompt_tokens_details") or {}).get("audio_tokens") or 0)
        completion_audio_tokens += int((usage.get("completion_tokens_details") or {}).get("audio_tokens") or 0)
        google_prompt_tokens += int(
            usage.get("promptTokenCount")
            or usage.get("prompt_token_count")
            or 0
        )
        google_candidate_tokens += int(
            usage.get("candidatesTokenCount")
            or usage.get("candidates_token_count")
            or 0
        )
        google_total_tokens += int(
            usage.get("totalTokenCount")
            or usage.get("total_token_count")
            or 0
        )

    for record in cache.values():
        parser_providers[
            str(record.get("parser_provider") or "openai")
        ] += 1
        for chunk in record.get("chunks") or []:
            chunk_records += 1
            metadata = chunk.get("api_metadata") or {}
            if metadata.get("response_id"):
                recorded_responses += 1
                add_usage(metadata.get("usage"))
            for failure in metadata.get("failed_attempts") or []:
                if failure.get("response_id"):
                    recorded_responses += 1
                add_usage(failure.get("usage"))
            transcription_fallbacks += int(bool(metadata.get("transcription_fallback")))
            unstructured_fallbacks += int(bool(metadata.get("unstructured_fallback")))
    result = {
        "created_at": utc_now(),
        "media_source_records": len(cache),
        "parsed_chunk_records": chunk_records,
        "recorded_audio_model_responses": recorded_responses,
        "transcription_fallbacks": transcription_fallbacks,
        "unstructured_prose_fallbacks": unstructured_fallbacks,
        "recorded_usage": {
            "openai": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "prompt_audio_tokens": prompt_audio_tokens,
                "completion_audio_tokens": completion_audio_tokens,
            },
            "google": {
                "prompt_tokens": google_prompt_tokens,
                "candidate_tokens": google_candidate_tokens,
                "total_tokens": google_total_tokens,
            },
        },
        "parser_providers": dict(parser_providers),
        "cost_note": f"Media parsing is accounted separately from the {expected_count:,} final judge calls. Dollar cost is not estimated here because the frozen parser records span provider-specific audio-token and transcription billing units.",
    }
    write_json(state / "media_parser_accounting.json", result)
    return result


def frozen_input_audit(state: Path, *, verify_source_hashes: bool = True) -> dict[str, Any]:
    manifest_path = state / "comparison_manifest.unblinded.jsonl"
    manifest = list(iter_jsonl(manifest_path))
    manifest_summary = read_json(state / "manifest_summary.json")
    expected_tasks = int(manifest_summary.get("tasks") or 220)
    expected_comparisons = expected_comparison_count(state)
    active_conditions = active_conditions_for_state(state)
    repeat_count = repeat_count_for_state(state)
    validate_manifest(
        manifest,
        expected_tasks=expected_tasks,
        repeat_count=repeat_count,
        conditions=active_conditions,
    )
    crosswalk = read_json(state / "crosswalk.json")
    source_files = read_json(state / "source_files.json")
    evidence = read_json(state / "evidence_index.json")
    evidence_summary = read_json(state / "evidence_summary.json")
    provenance = evidence_provenance_audit(state)
    media = media_parser_accounting(state)
    prior_audit = read_json(state / "frozen_input_audit.json") if (state / "frozen_input_audit.json").is_file() else {}
    current_manifest_sha = sha256_file(manifest_path)
    source_verification_inherited = bool(
        not verify_source_hashes
        and prior_audit.get("complete")
        and prior_audit.get("source_hashes_reverified")
        and prior_audit.get("manifest_sha256") == current_manifest_sha
    )
    source_hash_failures: list[str] = []
    if verify_source_hashes:
        for item in source_files:
            path = Path(str(item["path"]))
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                source_hash_failures.append(str(path))
    evidence_hash_failures: list[str] = []
    quality_defects: list[dict[str, Any]] = []
    visual_fallbacks: Counter[str] = Counter()
    visual_page_counts: list[int] = []
    for item in evidence:
        audit = read_json(Path(str(item["audit_path"])))
        text_path = Path(str(item["text_path"]))
        if not text_path.is_file() or sha256_file(text_path) != audit.get("text_sha256"):
            evidence_hash_failures.append(str(text_path))
        visual_raw = item.get("visual_path")
        if visual_raw:
            visual_path = Path(str(visual_raw))
            if not visual_path.is_file() or sha256_file(visual_path) != audit.get("visual_sha256"):
                evidence_hash_failures.append(str(visual_path))
            else:
                from pypdf import PdfReader

                visual_page_counts.append(len(PdfReader(visual_path, strict=False).pages))
        for defect in audit.get("quality_defects") or []:
            quality_defects.append({"source_hash": item["source_hash"], **defect})
        for file_audit in audit.get("file_audits") or []:
            if file_audit.get("visual_fallback"):
                visual_fallbacks[str(file_audit["visual_fallback"])] += 1
    task_condition_outputs = Counter((row["task_id"], row["condition"]) for row in manifest if row["baseline"] == "human")
    baseline_task_bundles = {
        (str(row["task_id"]), str(row["baseline"]), bundle_source_hash(row["baseline_files"]))
        for row in manifest
    }
    parser_error_count = sum(len(item.get("parser_errors") or []) for item in evidence)
    media_pending_count = sum(len(item.get("media_pending") or []) for item in evidence)
    complete = (
        len(manifest) == expected_comparisons
        and len(crosswalk) == expected_tasks
        and len(task_condition_outputs) == expected_tasks * len(active_conditions)
        and all(count == repeat_count for count in task_condition_outputs.values())
        and len(baseline_task_bundles) == expected_tasks * len(BASELINES)
        and len(evidence) == int(evidence_summary["unique_bundles_expected"])
        and (verify_source_hashes or source_verification_inherited)
        and not source_hash_failures
        and not evidence_hash_failures
        and not parser_error_count
        and not media_pending_count
        and (not visual_page_counts or max(visual_page_counts) <= MAX_VISUAL_PAGES_PER_BUNDLE)
        and provenance["passed"]
        and manifest_summary.get("manifest_sha256") == sha256_file(manifest_path)
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "complete": complete,
        "source_hashes_reverified": verify_source_hashes or source_verification_inherited,
        "source_hash_verification_inherited_from": prior_audit.get("created_at") if source_verification_inherited else None,
        "manifest_sha256": current_manifest_sha,
        "manifest_rows": len(manifest),
        "mapped_tasks": len(crosswalk),
        "task_condition_output_groups": len(task_condition_outputs),
        "task_condition_groups_with_expected_outputs": sum(
            count == repeat_count for count in task_condition_outputs.values()
        ),
        "task_condition_groups_with_four_outputs": sum(
            count == 4 for count in task_condition_outputs.values()
        ),
        "baseline_task_bundles": len(baseline_task_bundles),
        "rubric_linked_comparisons": sum(bool(row.get("rubric")) for row in manifest),
        "reference_linked_comparisons": sum(
            len(row.get("reference_files") or []) == len(row.get("expected_reference_names") or [])
            for row in manifest
        ),
        "source_files_hashed": len(source_files),
        "source_hash_failures": source_hash_failures,
        "evidence_bundles": len(evidence),
        "evidence_hash_failures": evidence_hash_failures,
        "parser_errors": parser_error_count,
        "media_pending": media_pending_count,
        "position_counts": dict(Counter(row["agent_position"] for row in manifest)),
        "within_stratum_two_A_two_B": True,
        "reference_alias_content_audit": manifest_summary.get("reference_alias_content_audit"),
        "provenance_audit": provenance,
        "candidate_quality_defects": quality_defects,
        "visual_fallback_counts": dict(visual_fallbacks),
        "visual_context_budget": {
            "max_pages_per_bundle": MAX_VISUAL_PAGES_PER_BUNDLE,
            "observed_max_pages": max(visual_page_counts) if visual_page_counts else 0,
            "bundles_sampled": int((read_json(state / "visual_sampling_audit.json") if (state / "visual_sampling_audit.json").is_file() else {}).get("bundles_sampled") or 0),
        },
        "media_parser_accounting": media,
    }
    write_json(state / "frozen_input_audit.json", result)
    if not complete:
        raise EvaluationError("frozen input audit failed; see frozen_input_audit.json")
    return result


def upload_evidence(state: Path, *, workers: int = 8) -> dict[str, Any]:
    audit_path = state / "frozen_input_audit.json"
    if not audit_path.is_file():
        raise EvaluationError("run audit-inputs before uploading evidence")
    frozen = read_json(audit_path)
    current_manifest_sha = sha256_file(state / "comparison_manifest.unblinded.jsonl")
    if not frozen.get("complete") or frozen.get("manifest_sha256") != current_manifest_sha:
        raise EvaluationError("frozen input audit is incomplete or stale; rerun audit-inputs")
    evidence = read_json(state / "evidence_index.json")
    summary = read_json(state / "evidence_summary.json")
    if summary.get("media_pending_items"):
        # The summary is refreshed after parse-pending-media; stale pending is
        # also checked from the per-bundle index below.
        pending = sum(len(item.get("media_pending") or []) for item in evidence)
        if pending:
            raise EvaluationError(f"{pending} media evidence items remain unparsed")
    parser_errors = [
        (item["source_hash"], error)
        for item in evidence
        for error in item.get("parser_errors") or []
        if error.get("kind") not in {"psd_render"}
    ]
    if parser_errors:
        raise EvaluationError(f"evidence has {len(parser_errors)} parser errors; repair before upload: {parser_errors[:5]}")
    uploads_path = state / "uploads.json"
    uploads = read_json(uploads_path) if uploads_path.is_file() else {}
    neutral_dir = state / "upload_files"
    jobs: list[dict[str, Any]] = []
    for item in evidence:
        source_hash = str(item["source_hash"])
        entry = uploads.setdefault(source_hash, {})
        for kind, raw_path, suffix in (
            ("text", item.get("text_path"), ".txt"),
            ("visual", item.get("visual_path"), ".pdf"),
        ):
            if not raw_path:
                continue
            source = Path(str(raw_path))
            digest = sha256_file(source)
            existing = entry.get(kind)
            if isinstance(existing, dict) and existing.get("sha256") == digest and existing.get("file_id"):
                continue
            neutral = neutral_dir / f"{kind}_{source_hash}{suffix}"
            neutral_upload_copy(source, neutral)
            if neutral.stat().st_size > MAX_UPLOAD_BYTES:
                raise EvaluationError(f"upload file exceeds 48 MiB: {neutral}")
            jobs.append(
                {
                    "source_hash": source_hash,
                    "kind": kind,
                    "path": neutral,
                    "sha256": digest,
                    "bytes": neutral.stat().st_size,
                }
            )

    def upload_one(job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        uploaded = OpenAIAPI().upload_file(Path(job["path"]), purpose="user_data")
        return job, uploaded

    errors: list[str] = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_map = {executor.submit(upload_one, job): job for job in jobs}
        for future in concurrent.futures.as_completed(future_map):
            job = future_map[future]
            try:
                finished, uploaded = future.result()
            except Exception as exc:
                errors.append(f"{job['source_hash']}:{job['kind']}: {exc}")
                continue
            entry = uploads.setdefault(str(finished["source_hash"]), {})
            entry[str(finished["kind"])] = {
                "file_id": uploaded["id"],
                "sha256": finished["sha256"],
                "bytes": finished["bytes"],
                "created_at": utc_now(),
                "api_filename": Path(finished["path"]).name,
            }
            write_json(uploads_path, uploads)
            completed += 1
            if completed % 50 == 0 or completed == len(jobs):
                print(f"uploaded files {completed}/{len(jobs)} with {max(1, workers)} workers", flush=True)
    if errors:
        raise EvaluationError(f"{len(errors)} evidence uploads failed; rerun is resumable: {errors[:10]}")
    return {
        "bundles": len(evidence),
        "new_files_uploaded": completed,
        "uploaded_text": sum("text" in item for item in uploads.values()),
        "uploaded_visual": sum("visual" in item for item in uploads.values()),
    }


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "rationale": {"type": "string"},
        "candidate_A_strengths": {"type": "array", "items": {"type": "string"}},
        "candidate_A_defects": {"type": "array", "items": {"type": "string"}},
        "candidate_B_strengths": {"type": "array", "items": {"type": "string"}},
        "candidate_B_defects": {"type": "array", "items": {"type": "string"}},
        "processing_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "winner",
        "rationale",
        "candidate_A_strengths",
        "candidate_A_defects",
        "candidate_B_strengths",
        "candidate_B_defects",
        "processing_notes",
    ],
    "additionalProperties": False,
}


def rubric_text(row: dict[str, Any]) -> str:
    lines = []
    for item in row["rubric"]:
        points = item.get("points")
        prefix = f"[{points:+g}] " if isinstance(points, (int, float)) else ""
        lines.append(f"{item['index']}. {prefix}{item['criterion']}")
    return "\n".join(lines)


def judge_instructions(row: dict[str, Any]) -> str:
    return f"""You are judging two anonymous deliverable bundles for the same professional task.

Choose Candidate A, Candidate B, or tie. Judge the complete submitted deliverable holistically against the task, reference materials, and canonical rubric below. The rubric guides the overall preference; do not invent a new point system and do not mechanically total rubric points as the sole decision rule. Prefer the deliverable that is meaningfully better overall in correctness, completeness, usability, fidelity to references, and professional quality.

Return tie when neither candidate is meaningfully better overall, including when material advantages balance. Do not use tie merely because the decision is difficult. Record preprocessing details under processing_notes; do not turn an infrastructure failure into a tie. Do not infer or discuss who or what created either candidate. Candidate order is randomized and carries no meaning.

## Task
{row['task_prompt']}

## Canonical rubric
{rubric_text(row)}
"""


def add_bundle_content(content: list[dict[str, Any]], label: str, source_hash: str, uploads: dict[str, Any]) -> None:
    entry = uploads.get(source_hash)
    if not isinstance(entry, dict) or not entry.get("text"):
        raise EvaluationError(f"missing uploaded evidence for bundle {source_hash}")
    content.append({"type": "input_text", "text": f"\n## {label}\n"})
    content.append({"type": "input_file", "file_id": entry["text"]["file_id"]})
    if entry.get("visual"):
        # `detail` is valid for input_image, not input_file.  PDFs are parsed as
        # both extracted text and page images by the Responses API.
        content.append({"type": "input_file", "file_id": entry["visual"]["file_id"]})


def compact_retry_uploads_for_targets(
    state: Path,
    target_ids: set[str],
    original_requests: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence_by_hash = {
        str(item["source_hash"]): item
        for item in read_json(state / "evidence_index.json")
    }
    full_uploads = read_json(state / "uploads.json")
    needed_file_ids: set[str] = set()
    for opaque in target_ids:
        request = original_requests[opaque]
        content = (((request.get("body") or {}).get("input") or [{}])[0]).get("content") or []
        needed_file_ids.update(
            str(item.get("file_id") or "")
            for item in content
            if item.get("type") == "input_file" and item.get("file_id")
        )
    needed_hashes = {
        source_hash
        for source_hash, entry in full_uploads.items()
        for kind in ("text", "visual")
        if isinstance(entry, dict)
        and isinstance(entry.get(kind), dict)
        and str(entry[kind].get("file_id") or "") in needed_file_ids
    }
    if not needed_hashes:
        raise EvaluationError("compact retry could not resolve any source bundles from failed requests")

    compact_uploads_path = state / "compact_retry_uploads.json"
    compact_uploads = read_json(compact_uploads_path) if compact_uploads_path.is_file() else {}
    compact_dir = state / "compact_retry_upload_files"
    compact_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for source_hash in sorted(needed_hashes):
        evidence = evidence_by_hash.get(source_hash)
        if not evidence:
            raise EvaluationError(f"compact retry target lacks evidence index entry: {source_hash}")
        text_path = Path(str(evidence.get("text_path") or ""))
        if not text_path.is_file():
            raise EvaluationError(f"compact retry target lacks evidence text: {source_hash}")
        original = text_path.read_text(encoding="utf-8", errors="replace")
        compact_text = (
            "# Neutral deliverable evidence (compact retry)\n"
            "Evidence format: compact text extraction with deterministic head/tail selection. "
            "Attached evidence: text. Evaluate the submitted content separately from the extraction method.\n\n"
            + bounded(original, COMPACT_RETRY_TEXT_CHARS_PER_BUNDLE)
        )
        compact_path = compact_dir / f"compact_text_{source_hash}.txt"
        if not compact_path.is_file() or compact_path.read_text(encoding="utf-8", errors="replace") != compact_text:
            compact_path.write_text(compact_text, encoding="utf-8")
        digest = sha256_file(compact_path)
        entry = compact_uploads.setdefault(source_hash, {})
        existing = entry.get("text")
        if isinstance(existing, dict) and existing.get("sha256") == digest and existing.get("file_id"):
            continue
        jobs.append(
            {
                "source_hash": source_hash,
                "kind": "text",
                "path": compact_path,
                "sha256": digest,
                "bytes": compact_path.stat().st_size,
            }
        )

    for job in jobs:
        if int(job["bytes"]) > MAX_UPLOAD_BYTES:
            raise EvaluationError(f"compact retry upload exceeds 48 MiB: {job['path']}")
        uploaded = OpenAIAPI().upload_file(Path(job["path"]), purpose="user_data")
        entry = compact_uploads.setdefault(str(job["source_hash"]), {})
        entry["text"] = {
            "file_id": uploaded["id"],
            "sha256": job["sha256"],
            "bytes": job["bytes"],
            "created_at": utc_now(),
            "api_filename": Path(job["path"]).name,
            "compact_retry": True,
            "source_text_chars": COMPACT_RETRY_TEXT_CHARS_PER_BUNDLE,
            "visual_omitted": True,
        }
        write_json(compact_uploads_path, compact_uploads)

    plan = {
        "needed_hashes": sorted(needed_hashes),
        "new_files_uploaded": len(jobs),
        "compact_text_chars_per_bundle": COMPACT_RETRY_TEXT_CHARS_PER_BUNDLE,
        "visual_omitted": True,
    }
    return compact_uploads, plan


def compact_retry_request(request: dict[str, Any], compact_uploads: dict[str, Any], full_uploads: dict[str, Any]) -> dict[str, Any]:
    file_id_to_hash = {
        str(entry[kind]["file_id"]): source_hash
        for source_hash, entry in full_uploads.items()
        if isinstance(entry, dict)
        for kind in ("text", "visual")
        if isinstance(entry.get(kind), dict) and entry[kind].get("file_id")
    }
    compact = json.loads(json.dumps(request))
    body = compact["body"]
    body["max_output_tokens"] = max(int(body.get("max_output_tokens") or 0), DEFAULT_MAX_OUTPUT_TOKENS)
    original_content = body["input"][0]["content"]
    new_content: list[dict[str, Any]] = []
    last_label: str | None = None
    emitted_for_label: set[str] = set()
    for item in original_content:
        if item.get("type") == "input_text":
            text = str(item.get("text") or "")
            new_content.append(item)
            stripped = text.strip()
            if stripped.startswith("## "):
                last_label = stripped
                emitted_for_label.discard(last_label)
            continue
        if item.get("type") != "input_file":
            new_content.append(item)
            continue
        source_hash = file_id_to_hash.get(str(item.get("file_id") or ""))
        if not source_hash:
            continue
        if last_label and last_label in emitted_for_label:
            continue
        compact_entry = compact_uploads.get(source_hash)
        if not isinstance(compact_entry, dict) or not compact_entry.get("text"):
            raise EvaluationError(f"missing compact retry upload for bundle {source_hash}")
        new_content.append({"type": "input_file", "file_id": compact_entry["text"]["file_id"]})
        if last_label:
            emitted_for_label.add(last_label)
    body["input"][0]["content"] = new_content
    return compact


def blind_id(comparison_id: str) -> str:
    return "cmp_" + sha256_text(SCHEMA_VERSION + "|" + comparison_id)[:28]


def request_for_row(row: dict[str, Any], uploads: dict[str, Any], *, model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    agent_hash = bundle_source_hash(row["agent_files"])
    baseline_hash = bundle_source_hash(row["baseline_files"])
    reference_hash = bundle_source_hash(row["reference_files"])
    content: list[dict[str, Any]] = [{"type": "input_text", "text": judge_instructions(row)}]
    if row["reference_files"]:
        add_bundle_content(content, "Reference materials (common to both candidates)", reference_hash, uploads)
    else:
        content.append({"type": "input_text", "text": "\n## Reference materials\nNo reference files were supplied for this task.\n"})
    if row["agent_position"] == "A":
        a_hash, b_hash = agent_hash, baseline_hash
    else:
        a_hash, b_hash = baseline_hash, agent_hash
    add_bundle_content(content, "Candidate A", a_hash, uploads)
    add_bundle_content(content, "Candidate B", b_hash, uploads)
    opaque = blind_id(str(row["comparison_id"]))
    request = {
        "custom_id": opaque,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "reasoning": {"effort": "medium"},
            "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "gdpval_pairwise_verdict",
                    "schema": JUDGE_SCHEMA,
                    "strict": True,
                }
            },
        },
    }
    mapping = {
        "blind_id": opaque,
        "comparison_id": row["comparison_id"],
        "task_id": row["task_id"],
        "condition": row["condition"],
        "baseline": row["baseline"],
        "repeat": row["repeat"],
        "agent_position": row["agent_position"],
        "occupation": row["occupation"],
        "sector": row["sector"],
    }
    return request, mapping


def leak_scan(request: dict[str, Any], *, state: Path | None = None) -> list[str]:
    serialized = json.dumps(request, ensure_ascii=False)
    patterns = {
        **provenance_markers_for_state(state),
        "human-ai label": "human_ai",
        "human+ai label": "human+ai",
        "artifact directory marker": "artifact_dir",
    }
    return [label for label, needle in patterns.items() if needle in serialized]


def audit_pilot_payloads(
    state: Path,
    pilot_requests: Sequence[dict[str, Any]],
    rows_by_blind_id: dict[str, dict[str, Any]],
    uploads: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for request in pilot_requests:
        opaque = str(request["custom_id"])
        row = rows_by_blind_id[opaque]
        body = request.get("body") or {}
        content = (((body.get("input") or [{}])[0]).get("content") or [])
        input_files = [item for item in content if item.get("type") == "input_file"]
        input_text = "\n".join(
            str(item.get("text") or "") for item in content if item.get("type") == "input_text"
        )
        allowed_content_keys = {
            "input_text": {"type", "text"},
            "input_file": {"type", "file_id"},
        }
        unexpected_keys = [
            sorted(set(item) - allowed_content_keys.get(str(item.get("type")), set()))
            for item in content
            if set(item) - allowed_content_keys.get(str(item.get("type")), set())
        ]
        expected_file_count = 0
        for files in (row["reference_files"], row["agent_files"], row["baseline_files"]):
            if not files and files is row["reference_files"]:
                continue
            bundle_hash = bundle_source_hash(files)
            entry = uploads.get(bundle_hash) or {}
            expected_file_count += int(bool(entry.get("text"))) + int(bool(entry.get("visual")))
        serialized = json.dumps(request, ensure_ascii=False)
        checks = {
            "pinned_model": body.get("model") == model == DEFAULT_MODEL,
            "responses_endpoint": request.get("url") == "/v1/responses",
            "structured_schema": ((body.get("text") or {}).get("format") or {}).get("schema") == JUDGE_SCHEMA,
            "only_supported_content_keys": not unexpected_keys,
            "file_attachment_count": len(input_files) == expected_file_count,
            "all_file_ids_opaque": all(
                str(item.get("file_id") or "").startswith("file-") and "/" not in str(item.get("file_id") or "")
                for item in input_files
            ),
            "task_present": str(row["task_prompt"]) in input_text,
            "rubric_present": all(str(item["criterion"]) in input_text for item in row["rubric"]),
            "anonymous_candidate_labels": "## Candidate A" in input_text and "## Candidate B" in input_text,
            "common_reference_label": (
                "## Reference materials (common to both candidates)" in input_text
                if row["reference_files"]
                else "No reference files were supplied" in input_text
            ),
            "no_provenance_leak": not leak_scan(request, state=state),
            "no_input_file_detail_field": '"detail"' not in serialized,
            "winner_contract": set(JUDGE_SCHEMA["properties"]["winner"]["enum"]) == {"A", "B", "tie"},
        }
        if not all(checks.values()):
            failures.append(
                {
                    "blind_id": opaque,
                    "failed_checks": [name for name, passed in checks.items() if not passed],
                    "unexpected_content_keys": unexpected_keys,
                    "expected_file_count": expected_file_count,
                    "actual_file_count": len(input_files),
                }
            )
        records.append(
            {
                "blind_id": opaque,
                "condition": row["condition"],
                "baseline": row["baseline"],
                "repeat": row["repeat"],
                "agent_position": row["agent_position"],
                "agent_quality_status": row["agent_quality_status"],
                "request_bytes": len(serialized.encode("utf-8")),
                "input_files": len(input_files),
                "checks": checks,
            }
        )
    report = {
        "created_at": utc_now(),
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "pilot_requests": len(pilot_requests),
        "passed": not failures,
        "failures": failures,
        "condition_counts": dict(Counter(record["condition"] for record in records)),
        "baseline_counts": dict(Counter(record["baseline"] for record in records)),
        "agent_position_counts": dict(Counter(record["agent_position"] for record in records)),
        "agent_quality_status_counts": dict(Counter(record["agent_quality_status"] for record in records)),
        "records": records,
    }
    write_json(state / "pilot_payload_audit.json", report)
    previews = []
    for request in pilot_requests[:5]:
        preview = json.loads(json.dumps(request))
        for item in preview["body"]["input"][0]["content"]:
            if item.get("type") == "input_file":
                item["file_id"] = "<uploaded-file-id>"
        previews.append(preview)
    write_json(state / "pilot_payload_redacted_preview.json", previews)
    if failures:
        raise EvaluationError(f"pilot payload audit failed: {failures[:5]}")
    return report


def select_pilot_rows(
    rows: Sequence[dict[str, Any]], *, minimum: int = 24, state: Path | None = None
) -> list[str]:
    if minimum <= 0 or minimum > len(rows):
        raise EvaluationError(f"invalid pilot size {minimum} for {len(rows)} comparisons")
    active_conditions = (
        active_conditions_for_state(state)
        if state is not None
        else tuple(condition for condition in CONDITIONS if any(r["condition"] == condition for r in rows))
    )
    evidence_features: dict[str, set[str]] = defaultdict(set)
    if state is not None and (state / "evidence_index.json").is_file():
        for item in read_json(state / "evidence_index.json"):
            audit = read_json(Path(str(item["audit_path"])))
            source_hash = str(item["source_hash"])
            if audit.get("quality_defects"):
                evidence_features[source_hash].add("edge:submitted_corrupt_or_unreadable")
            if audit.get("visual_page_sampling"):
                evidence_features[source_hash].add("edge:visual_context_sampling")
            for file_audit in audit.get("file_audits") or []:
                if file_audit.get("visual_fallback"):
                    evidence_features[source_hash].add("edge:visual_reconstruction_fallback")
    features: dict[str, set[str]] = {}
    sizes: dict[str, int] = {}
    for row in rows:
        present: set[str] = set()
        total = 0
        for key in ("agent_files", "baseline_files", "reference_files"):
            present |= evidence_features.get(bundle_source_hash(row[key]), set())
            for item in row[key]:
                suffix = str(item.get("suffix") or "[none]")
                present.add("ext:" + suffix)
                total += int(item.get("size_bytes") or 0)
                if suffix in AUDIO_EXTS:
                    present.add("modality:audio")
                if suffix in VIDEO_EXTS:
                    present.add("modality:video")
                if suffix in ARCHIVE_EXTS:
                    present.add("modality:archive")
                if suffix in OFFICE_EXTS:
                    present.add("modality:office")
        present.add("agent_quality:" + str(row.get("agent_quality_status") or "unknown"))
        present.add("agent_position:" + str(row["agent_position"]))
        present.add(f"stratum:{row['condition']}:{row['baseline']}")
        features[str(row["comparison_id"])] = present
        sizes[str(row["comparison_id"])] = total
    largest_comparison = max(sizes, key=lambda cid: (sizes[cid], cid))
    features[largest_comparison].add("edge:largest_payload")
    uncovered = set().union(*features.values())
    selected: list[str] = []
    remaining = set(features)
    while uncovered and remaining:
        best = max(remaining, key=lambda cid: (len(features[cid] & uncovered), sizes[cid], cid))
        if not (features[best] & uncovered):
            break
        selected.append(best)
        uncovered -= features[best]
        remaining.remove(best)
    for cid in sorted(remaining, key=lambda value: (-sizes[value], value)):
        if len(selected) >= minimum:
            break
        selected.append(cid)
    by_id = {str(r["comparison_id"]): r for r in rows}
    selected = sorted(set(selected))
    if len(selected) != minimum:
        raise EvaluationError(
            f"deterministic pilot must contain exactly {minimum} comparisons; selected {len(selected)}"
        )
    if state is not None:
        all_features = set().union(*features.values())
        selected_features = set().union(*(features[cid] for cid in selected))
        selected_rows = [by_id[cid] for cid in selected]
        combinations = {
            (str(row["condition"]), str(row["baseline"])) for row in selected_rows
        }
        coverage = {
            "created_at": utc_now(),
            "selected_count": len(selected),
            "selected_comparison_ids": selected,
            "covered_features": sorted(selected_features),
            "uncovered_features": sorted(all_features - selected_features),
            "largest_comparison_selected": largest_comparison in selected,
            "condition_baseline_combinations": [list(value) for value in sorted(combinations)],
            "both_agent_positions": {str(row["agent_position"]) for row in selected_rows} == {"A", "B"},
        }
        coverage["passed"] = (
            not coverage["uncovered_features"]
            and coverage["largest_comparison_selected"]
            and combinations == set((condition, baseline) for condition in active_conditions for baseline in BASELINES)
            and coverage["both_agent_positions"]
            and len(selected) == minimum
        )
        write_json(state / "pilot_coverage_audit.json", coverage)
        if not coverage["passed"]:
            raise EvaluationError(f"pilot coverage audit failed: {coverage}")
    return selected


def build_batch_requests(state: Path, *, model: str = DEFAULT_MODEL, shard_size: int = 880) -> dict[str, Any]:
    rows = list(iter_jsonl(state / "comparison_manifest.unblinded.jsonl"))
    expected_count = expected_comparison_count(state)
    validate_manifest(
        rows,
        expected_tasks=expected_task_count(state),
        repeat_count=repeat_count_for_state(state),
        conditions=active_conditions_for_state(state),
    )
    uploads = read_json(state / "uploads.json")
    requests: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    leak_failures: list[dict[str, Any]] = []
    for row in rows:
        request, mapping = request_for_row(row, uploads, model=model)
        leaks = leak_scan(request, state=state)
        if leaks:
            leak_failures.append({"comparison_id": row["comparison_id"], "leaks": leaks})
        requests.append(request)
        mappings.append(mapping)
    if leak_failures:
        raise EvaluationError(f"identity/provenance leakage detected: {leak_failures[:10]}")
    if len({r["custom_id"] for r in requests}) != expected_count:
        raise EvaluationError("blinded custom IDs are not unique")
    batch_dir = state / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[str] = []
    for shard_index in range(math.ceil(len(requests) / shard_size)):
        chunk = requests[shard_index * shard_size : (shard_index + 1) * shard_size]
        path = batch_dir / f"final_requests_{shard_index:02d}.jsonl"
        write_jsonl(path, chunk)
        if path.stat().st_size > 190 * 1024 * 1024:
            raise EvaluationError(f"Batch shard exceeds 190 MiB: {path}")
        shard_paths.append(str(path))
    write_jsonl(state / "blind_map.unblinded.jsonl", mappings)
    pilot = select_pilot_rows(rows, state=state)
    write_json(state / "pilot_selection.json", {"comparison_ids": pilot, "count": len(pilot), "created_at": utc_now()})
    pilot_blind = {blind_id(cid) for cid in pilot}
    pilot_requests = [r for r in requests if r["custom_id"] in pilot_blind]
    rows_by_blind_id = {blind_id(str(row["comparison_id"])): row for row in rows}
    pilot_audit = audit_pilot_payloads(
        state, pilot_requests, rows_by_blind_id, uploads, model=model
    )
    write_jsonl(batch_dir / "pilot_requests.jsonl", pilot_requests)
    remaining = [r for r in requests if r["custom_id"] not in pilot_blind]
    remaining_paths: list[str] = []
    for shard_index in range(math.ceil(len(remaining) / shard_size)):
        chunk = remaining[shard_index * shard_size : (shard_index + 1) * shard_size]
        path = batch_dir / f"remaining_requests_{shard_index:02d}.jsonl"
        write_jsonl(path, chunk)
        remaining_paths.append(str(path))
    summary = {
        "created_at": utc_now(),
        "model": model,
        "request_count": len(requests),
        "shards": shard_paths,
        "remaining_shards": remaining_paths,
        "pilot_count": len(pilot),
        "pilot_payload_audit_passed": pilot_audit["passed"],
        "pilot_request_sha256": sha256_file(batch_dir / "pilot_requests.jsonl"),
        "remaining_request_sha256": {
            Path(path).name: sha256_file(Path(path)) for path in remaining_paths
        },
        "judge_schema_sha256": sha256_text(json.dumps(JUDGE_SCHEMA, sort_keys=True, separators=(",", ":"))),
        "request_protocol": {
            "endpoint": "/v1/responses",
            "reasoning_effort": "medium",
            "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
            "rubric_visible": True,
            "explicit_tie": True,
            "candidate_identity_blinded": True,
        },
        "leak_failures": 0,
        "request_sha256": {str(Path(p).name): sha256_file(Path(p)) for p in shard_paths},
    }
    write_json(state / "batch_request_summary.json", summary)
    return summary


def submit_batches(state: Path, *, mode: str) -> dict[str, Any]:
    if mode not in {"pilot", "remaining", "retry", "all"}:
        raise EvaluationError(f"unknown submission mode: {mode}")
    summary = read_json(state / "batch_request_summary.json")
    batch_dir = state / "batches"
    if mode == "pilot":
        paths = [batch_dir / "pilot_requests.jsonl"]
    elif mode == "remaining":
        pilot_jobs = read_json(state / "batch_jobs.json") if (state / "batch_jobs.json").is_file() else []
        pilot_complete = any(job.get("kind") == "pilot" and job.get("status") == "completed" for job in pilot_jobs)
        if not pilot_complete:
            raise EvaluationError("pilot Batch must complete successfully before remaining requests are submitted")
        pilot_ids = {
            blind_id(str(comparison_id))
            for comparison_id in read_json(state / "pilot_selection.json")["comparison_ids"]
        }
        valid_pilot_ids = {
            str(row["blind_id"])
            for row in iter_jsonl(state / "judgments.raw.unblinded.jsonl")
        } if (state / "judgments.raw.unblinded.jsonl").is_file() else set()
        if not pilot_ids.issubset(valid_pilot_ids):
            raise EvaluationError("all pilot verdicts must parse and validate before remaining requests are submitted")
        paths = [Path(p) for p in summary["remaining_shards"]]
    elif mode == "retry":
        paths = sorted(batch_dir.glob("retry_requests_*.jsonl"))
        if not paths:
            raise EvaluationError("no retry request file exists; run build-retry first")
        paths = paths[-1:]
    else:
        paths = [Path(p) for p in summary["shards"]]
    jobs_path = state / "batch_jobs.json"
    jobs = read_json(jobs_path) if jobs_path.is_file() else []
    by_digest = {str(job.get("input_sha256")): job for job in jobs}
    api = OpenAIAPI()
    new_jobs = 0
    for path in paths:
        digest = sha256_file(path)
        existing = by_digest.get(digest)
        if existing and existing.get("batch_id"):
            continue
        uploaded = api.upload_file(path, purpose="batch")
        batch = api.json(
            "POST",
            "/batches",
            {
                "input_file_id": uploaded["id"],
                "endpoint": "/v1/responses",
                "completion_window": "24h",
                "metadata": {
                    "evaluation": SCHEMA_VERSION,
                    "kind": mode,
                    "input_sha256": digest[:32],
                },
            },
        )
        job = {
            "kind": mode,
            "path": str(path),
            "request_count": sum(1 for _ in iter_jsonl(path)),
            "input_sha256": digest,
            "input_file_id": uploaded["id"],
            "batch_id": batch["id"],
            "status": batch.get("status"),
            "submitted_at": utc_now(),
            "api": batch,
        }
        jobs.append(job)
        by_digest[digest] = job
        write_json(jobs_path, jobs)
        new_jobs += 1
        print(f"submitted {path.name}: {batch['id']}", flush=True)
    return {"mode": mode, "jobs_total": len(jobs), "jobs_new": new_jobs}


TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


def poll_batches(state: Path, *, wait: bool = False, interval: int = 60) -> dict[str, Any]:
    jobs_path = state / "batch_jobs.json"
    jobs = read_json(jobs_path)
    api = OpenAIAPI()
    output_dir = state / "batch_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    while True:
        changed = False
        for job in jobs:
            batch = api.json("GET", f"/batches/{job['batch_id']}")
            old = job.get("status")
            job["status"] = batch.get("status")
            job["api"] = batch
            job["polled_at"] = utc_now()
            if old != job["status"]:
                changed = True
                print(f"{job['batch_id']}: {old} -> {job['status']}", flush=True)
            output_file_id = batch.get("output_file_id")
            if output_file_id and not job.get("output_path"):
                target = output_dir / f"{job['batch_id']}.output.jsonl"
                target.write_bytes(api.download_file(str(output_file_id)))
                job["output_path"] = str(target)
                job["output_sha256"] = sha256_file(target)
                changed = True
            error_file_id = batch.get("error_file_id")
            if error_file_id and not job.get("error_path"):
                target = output_dir / f"{job['batch_id']}.errors.jsonl"
                target.write_bytes(api.download_file(str(error_file_id)))
                job["error_path"] = str(target)
                job["error_sha256"] = sha256_file(target)
                changed = True
        if changed:
            write_json(jobs_path, jobs)
        statuses = Counter(str(job.get("status")) for job in jobs)
        if not wait or all(str(job.get("status")) in TERMINAL_BATCH_STATUSES for job in jobs):
            write_json(jobs_path, jobs)
            return {"jobs": len(jobs), "statuses": dict(statuses)}
        time.sleep(max(10, interval))


def parse_batch_outputs(state: Path) -> dict[str, Any]:
    expected_count = expected_comparison_count(state)
    mappings = {row["blind_id"]: row for row in iter_jsonl(state / "blind_map.unblinded.jsonl")}
    jobs = read_json(state / "batch_jobs.json")
    valid_by_id: dict[str, dict[str, Any]] = {}
    failure_by_id: dict[str, dict[str, Any]] = {}
    submitted_ids: set[str] = set()
    attempt_counts: Counter[str] = Counter()
    for job in jobs:
        input_path = Path(str(job.get("path") or ""))
        if input_path.is_file():
            submitted_ids.update(str(request["custom_id"]) for request in iter_jsonl(input_path))
        for result_path_key in ("output_path", "error_path"):
            result_path = job.get(result_path_key)
            if not result_path:
                continue
            for result in iter_jsonl(Path(result_path)):
                opaque = str(result.get("custom_id") or "")
                attempt_counts[opaque] += 1
                mapping = mappings.get(opaque)
                if not mapping:
                    failure_by_id[opaque] = {"blind_id": opaque, "failure": "missing blind mapping", "raw": result}
                    continue
                response = result.get("response") or {}
                body = response.get("body") if isinstance(response, dict) else None
                if result.get("error") or not isinstance(body, dict) or int(response.get("status_code") or 0) != 200:
                    if opaque not in valid_by_id:
                        failure_by_id[opaque] = {**mapping, "failure": result.get("error") or response, "raw": result}
                    continue
                try:
                    verdict = extract_json_from_text(response_output_text(body))
                    winner = str(verdict.get("winner") or "")
                    if winner not in VALID_WINNERS:
                        raise EvaluationError(f"invalid winner: {winner!r}")
                    for key in JUDGE_SCHEMA["required"]:
                        if key not in verdict:
                            raise EvaluationError(f"missing structured-output key: {key}")
                except Exception as exc:
                    if opaque not in valid_by_id:
                        failure_by_id[opaque] = {**mapping, "failure": f"schema/parse failure: {exc}", "raw": result}
                    continue
                agent_position_value = str(mapping["agent_position"])
                outcome = "tie" if winner == "tie" else ("win" if winner == agent_position_value else "loss")
                score = 0.5 if outcome == "tie" else (1.0 if outcome == "win" else 0.0)
                valid_by_id[opaque] = {
                    **mapping,
                    "winner": winner,
                    "agent_outcome": outcome,
                    "agent_score": score,
                    "judgment": verdict,
                    "api_metadata": {
                        "batch_id": job["batch_id"],
                        "request_id": response.get("request_id"),
                        "response_id": body.get("id"),
                        "model": body.get("model"),
                        "created_at": body.get("created_at"),
                        "usage": body.get("usage"),
                        "attempt_count": int(attempt_counts[opaque]),
                    },
                }
                failure_by_id.pop(opaque, None)
    raw_rows = list(valid_by_id.values())
    failures = [failure for opaque, failure in failure_by_id.items() if opaque not in valid_by_id]
    missing_submitted = sorted(submitted_ids - set(valid_by_id) - set(failure_by_id))
    raw_rows.sort(key=lambda r: r["comparison_id"])
    failures.sort(key=lambda r: str(r.get("comparison_id") or r.get("blind_id") or ""))
    write_jsonl(state / "judgments.raw.unblinded.jsonl", raw_rows)
    write_jsonl(state / "judgment_failures.jsonl", failures)
    result = {
        "valid_verdicts": len(raw_rows),
        "failures": len(failures),
        "submitted_unique": len(submitted_ids),
        "missing_submitted_results": len(missing_submitted),
        "missing_submitted_blind_ids": missing_submitted,
        "retried_result_attempts": sum(count > 1 for count in attempt_counts.values()),
        "expected": expected_count,
        "complete": len(raw_rows) == expected_count and not failures and not missing_submitted,
    }
    write_json(state / "result_parse_summary.json", result)
    return result


def build_retry_requests(state: Path) -> dict[str, Any]:
    jobs = read_json(state / "batch_jobs.json")
    active = [job["batch_id"] for job in jobs if str(job.get("status")) not in TERMINAL_BATCH_STATUSES]
    if active:
        raise EvaluationError(f"cannot build retries while Batch jobs are still active: {active}")
    failures = list(iter_jsonl(state / "judgment_failures.jsonl")) if (state / "judgment_failures.jsonl").is_file() else []
    parse_summary = read_json(state / "result_parse_summary.json") if (state / "result_parse_summary.json").is_file() else {}
    target_ids = {str(row.get("blind_id") or "") for row in failures if row.get("blind_id")}
    target_ids.update(str(value) for value in parse_summary.get("missing_submitted_blind_ids") or [])
    valid_ids = {
        str(row["blind_id"])
        for row in iter_jsonl(state / "judgments.raw.unblinded.jsonl")
    } if (state / "judgments.raw.unblinded.jsonl").is_file() else set()
    target_ids -= valid_ids
    if not target_ids:
        raise EvaluationError("there are no unresolved submitted judgments to retry")
    original_requests: dict[str, dict[str, Any]] = {}
    for path in sorted((state / "batches").glob("final_requests_*.jsonl")):
        for request in iter_jsonl(path):
            original_requests[str(request["custom_id"])] = request
    missing_original = sorted(target_ids - set(original_requests))
    if missing_original:
        raise EvaluationError(f"retry targets lack frozen original requests: {missing_original[:10]}")
    existing = sorted((state / "batches").glob("retry_requests_*.jsonl"))
    retry_number = len(existing) + 1
    path = state / "batches" / f"retry_requests_{retry_number:02d}.jsonl"
    compact_uploads, compact_plan = compact_retry_uploads_for_targets(state, target_ids, original_requests)
    full_uploads = read_json(state / "uploads.json")
    requests = [
        compact_retry_request(original_requests[opaque], compact_uploads, full_uploads)
        for opaque in sorted(target_ids)
    ]
    leak_failures = [
        {"blind_id": str(request["custom_id"]), "leaks": leak_scan(request)}
        for request in requests
        if leak_scan(request)
    ]
    if leak_failures:
        raise EvaluationError(f"compact retry request leakage detected: {leak_failures[:10]}")
    write_jsonl(path, requests)
    plan = {
        "created_at": utc_now(),
        "retry_number": retry_number,
        "request_count": len(requests),
        "blind_ids": sorted(target_ids),
        "path": str(path),
        "sha256": sha256_file(path),
        "requests_are_byte_equivalent_to_frozen_originals": False,
        "retry_transform": "compact_text_only_evidence",
        "compact_plan": compact_plan,
    }
    plans_path = state / "retry_plans.json"
    plans = read_json(plans_path) if plans_path.is_file() else []
    plans.append(plan)
    write_json(plans_path, plans)
    return plan


def outcome_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float | int | None]:
    n = len(rows)
    wins = sum(r["agent_outcome"] == "win" for r in rows)
    ties = sum(r["agent_outcome"] == "tie" for r in rows)
    losses = sum(r["agent_outcome"] == "loss" for r in rows)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "tie_adjusted_preference": (wins + 0.5 * ties) / n,
        "strict_win_rate": wins / n,
        "tie_rate": ties / n,
        "loss_rate": losses / n,
        "non_tie_win_rate": wins / (wins + losses) if wins + losses else None,
        "win_or_tie_rate": (wins + ties) / n,
    }


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise EvaluationError("cannot take percentile of empty values")
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def bootstrap_headline(
    rows: Sequence[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> tuple[dict[str, list[float]], list[dict[str, float]]]:
    task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_occ: dict[str, str] = {}
    for row in rows:
        task_rows[str(row["task_id"])].append(row)
        task_occ[str(row["task_id"])] = str(row["occupation"])
    if not task_rows or any(len(values) != 4 for values in task_rows.values()):
        raise EvaluationError(
            f"headline bootstrap requires nonempty tasks with four repeats each; "
            f"got tasks={len(task_rows)} repeat_counts={sorted({len(v) for v in task_rows.values()})}"
        )
    by_occ: dict[str, list[str]] = defaultdict(list)
    for task_id, occupation in task_occ.items():
        by_occ[occupation].append(task_id)
    if not by_occ:
        raise EvaluationError("headline bootstrap requires at least one occupation stratum")
    rng = random.Random(seed)
    distributions: dict[str, list[float]] = defaultdict(list)
    sample_rows: list[dict[str, float]] = []
    for sample_idx in range(samples):
        chosen: list[dict[str, Any]] = []
        for occupation in sorted(by_occ):
            ids = sorted(by_occ[occupation])
            for _ in ids:
                chosen.extend(task_rows[rng.choice(ids)])
        metrics = outcome_metrics(chosen)
        row_out: dict[str, float] = {"sample": float(sample_idx)}
        for name in ("tie_adjusted_preference", "strict_win_rate", "tie_rate", "loss_rate", "win_or_tie_rate"):
            value = float(metrics[name])
            distributions[name].append(value)
            row_out[name] = value
        non_tie = metrics["non_tie_win_rate"]
        if non_tie is not None:
            distributions["non_tie_win_rate"].append(float(non_tie))
            row_out["non_tie_win_rate"] = float(non_tie)
        sample_rows.append(row_out)
    cis = {
        metric: [percentile(values, 0.025), percentile(values, 0.975)]
        for metric, values in distributions.items()
    }
    return cis, sample_rows


def usage_accounting(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = output_tokens = cached_tokens = reasoning_tokens = 0
    for row in rows:
        usage = row.get("api_metadata", {}).get("usage") or {}
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
        cached_tokens += int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
        reasoning_tokens += int((usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)
    uncached_tokens = max(0, input_tokens - cached_tokens)
    estimated = uncached_tokens / 1_000_000 * 0.25 + cached_tokens / 1_000_000 * 0.025 + output_tokens / 1_000_000 * 2.00
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "uncached_input_tokens": uncached_tokens,
        "output_tokens_including_reasoning": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "estimated_batch_cost_usd": estimated,
        "pricing_as_of": "2026-07-15",
        "pricing_per_million_usd": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
        "pricing_source": "https://developers.openai.com/api/docs/models/gpt-5-mini",
        "note": "Estimate excludes separate media-parser calls and any regional-processing uplift.",
    }


def paired_condition_bootstrap(
    rows: Sequence[dict[str, Any]], baseline: str, *, samples: int, seed: int
) -> dict[str, Any]:
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    occupations: dict[str, str] = {}
    for row in rows:
        if row["baseline"] != baseline:
            continue
        scores[(row["task_id"], row["condition"])].append(float(row["agent_score"]))
        occupations[row["task_id"]] = row["occupation"]
    differences: dict[str, float] = {}
    for task_id in occupations:
        with_values = scores[(task_id, "with_rubric")]
        without_values = scores[(task_id, "without_rubric")]
        if len(with_values) != 4 or len(without_values) != 4:
            raise EvaluationError(f"condition difference lacks four repeats for {task_id}/{baseline}")
        differences[task_id] = statistics.mean(with_values) - statistics.mean(without_values)
    by_occ: dict[str, list[str]] = defaultdict(list)
    for task_id, occupation in occupations.items():
        by_occ[occupation].append(task_id)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        selected: list[float] = []
        for occupation in sorted(by_occ):
            ids = sorted(by_occ[occupation])
            selected.extend(differences[rng.choice(ids)] for _ in ids)
        draws.append(statistics.mean(selected))
    return {
        "baseline": baseline,
        "estimate_with_minus_without": statistics.mean(differences.values()),
        "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
    }


def aggregate_results(state: Path, *, bootstrap_samples: int = 10_000, seed: int = 20260715) -> dict[str, Any]:
    expected_count = expected_comparison_count(state)
    active_conditions = active_conditions_for_state(state)
    rows = list(iter_jsonl(state / "judgments.raw.unblinded.jsonl"))
    if len(rows) != expected_count:
        raise EvaluationError(f"aggregation requires {expected_count:,} valid verdicts, found {len(rows)}")
    if len({r["comparison_id"] for r in rows}) != expected_count:
        raise EvaluationError("aggregation input contains duplicate comparison IDs")
    headline: dict[str, Any] = {}
    bootstrap_records: list[dict[str, Any]] = []
    for condition in active_conditions:
        for baseline in BASELINES:
            name = f"{condition}_vs_{baseline}"
            subset = [r for r in rows if r["condition"] == condition and r["baseline"] == baseline]
            metrics = outcome_metrics(subset)
            cis, draws = bootstrap_headline(subset, samples=bootstrap_samples, seed=seed + len(headline))
            metrics["ci95_task_clustered_stratified"] = cis
            headline[name] = metrics
            for draw in draws:
                bootstrap_records.append({"comparison": name, **draw})
    per_occupation: list[dict[str, Any]] = []
    for condition in active_conditions:
        for baseline in BASELINES:
            occupations = sorted({r["occupation"] for r in rows})
            for occupation in occupations:
                subset = [
                    r for r in rows
                    if r["condition"] == condition and r["baseline"] == baseline and r["occupation"] == occupation
                ]
                per_occupation.append({"condition": condition, "baseline": baseline, "occupation": occupation, **outcome_metrics(subset)})
    position_bias: list[dict[str, Any]] = []
    for condition in active_conditions:
        for baseline in BASELINES:
            for position in ("A", "B"):
                subset = [
                    r for r in rows
                    if r["condition"] == condition and r["baseline"] == baseline and r["agent_position"] == position
                ]
                position_bias.append({"condition": condition, "baseline": baseline, "agent_position": position, **outcome_metrics(subset)})
    condition_differences = (
        [
            paired_condition_bootstrap(rows, baseline, samples=bootstrap_samples, seed=seed + 100 + index)
            for index, baseline in enumerate(BASELINES)
        ]
        if set(CONDITIONS).issubset(active_conditions)
        else []
    )
    write_jsonl(state / "results.output_level.jsonl", rows)
    task_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["task_id"], row["condition"], row["baseline"])].append(row)
    for (task_id, condition, baseline), values in sorted(grouped.items()):
        task_rows.append(
            {
                "task_id": task_id,
                "condition": condition,
                "baseline": baseline,
                "occupation": values[0]["occupation"],
                "sector": values[0]["sector"],
                "repeat_count": len(values),
                "task_tie_adjusted_score": statistics.mean(float(v["agent_score"]) for v in values),
                "wins": sum(v["agent_outcome"] == "win" for v in values),
                "ties": sum(v["agent_outcome"] == "tie" for v in values),
                "losses": sum(v["agent_outcome"] == "loss" for v in values),
            }
        )
    write_jsonl(state / "results.task_level.jsonl", task_rows)
    write_jsonl(state / "bootstrap_samples.jsonl", bootstrap_records)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "valid_verdicts": len(rows),
        "headline": headline,
        "condition_differences": condition_differences,
        "per_occupation": per_occupation,
        "position_bias": position_bias,
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "unit": "task",
            "stratified_within": f"{len({r['occupation'] for r in rows})} occupations",
            "repeats_retained_together": True,
        },
        "usage": usage_accounting(rows),
        "evaluation_method": {"judge_type": "model", "candidate_assignment": "balanced randomized A/B"},
    }
    write_json(state / "results_summary.json", result)
    write_numeric_summary(state, result)
    return result


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_numeric_summary(state: Path, result: dict[str, Any]) -> None:
    flat_headline = [
        {"comparison": name, **{k: v for k, v in values.items() if not isinstance(v, dict)}}
        for name, values in result["headline"].items()
    ]
    write_csv(
        state / "headline_results.csv",
        flat_headline,
        [
            "comparison", "n", "wins", "ties", "losses", "tie_adjusted_preference",
            "strict_win_rate", "tie_rate", "loss_rate", "non_tie_win_rate", "win_or_tie_rate",
        ],
    )
    write_csv(
        state / "per_occupation_results.csv",
        result["per_occupation"],
        [
            "condition", "baseline", "occupation", "n", "wins", "ties", "losses",
            "tie_adjusted_preference", "strict_win_rate", "tie_rate", "loss_rate",
            "non_tie_win_rate", "win_or_tie_rate",
        ],
    )
    write_csv(
        state / "position_bias_diagnostic.csv",
        result["position_bias"],
        [
            "condition", "baseline", "agent_position", "n", "wins", "ties", "losses",
            "tie_adjusted_preference", "strict_win_rate", "tie_rate", "loss_rate",
            "non_tie_win_rate", "win_or_tie_rate",
        ],
    )


def validation_report(state: Path) -> dict[str, Any]:
    manifest = list(iter_jsonl(state / "comparison_manifest.unblinded.jsonl"))
    expected_count = expected_comparison_count(state)
    validate_manifest(
        manifest,
        expected_tasks=expected_task_count(state),
        repeat_count=repeat_count_for_state(state),
        conditions=active_conditions_for_state(state),
    )
    source_files = read_json(state / "source_files.json")
    bad_hashes = [item["path"] for item in source_files if not Path(item["path"]).is_file() or sha256_file(Path(item["path"])) != item["sha256"]]
    evidence = read_json(state / "evidence_index.json") if (state / "evidence_index.json").is_file() else []
    parser_errors = [(r["source_hash"], e) for r in evidence for e in r.get("parser_errors") or []]
    media_pending = [(r["source_hash"], e) for r in evidence for e in r.get("media_pending") or []]
    requests = []
    for path in sorted((state / "batches").glob("final_requests_*.jsonl")) if (state / "batches").is_dir() else []:
        requests.extend(iter_jsonl(path))
    leaks = [(r["custom_id"], leak_scan(r, state=state)) for r in requests if leak_scan(r, state=state)]
    judgments = list(iter_jsonl(state / "judgments.raw.unblinded.jsonl")) if (state / "judgments.raw.unblinded.jsonl").is_file() else []
    report = {
        "created_at": utc_now(),
        "manifest_rows": len(manifest),
        "source_files": len(source_files),
        "source_hash_failures": bad_hashes,
        "evidence_bundles": len(evidence),
        "parser_errors": parser_errors,
        "media_pending": media_pending,
        "blinded_requests": len(requests),
        "request_leaks": leaks,
        "valid_judgments": len(judgments),
        "position_counts": dict(Counter(r["agent_position"] for r in manifest)),
        "complete": (
            len(manifest) == expected_count
            and not bad_hashes
            and bool(evidence)
            and not parser_errors
            and not media_pending
            and len(requests) == expected_count
            and not leaks
            and len(judgments) == expected_count
        ),
    }
    write_json(state / "validation_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--volume", type=Path, default=DEFAULT_VOLUME)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--include-rubric", action="store_true", help="Also select the optional prompt inputs")
    sub.add_parser("select-evidence-pilot")
    sub.add_parser("prepare-evidence-pilot")
    prepare = sub.add_parser("prepare-evidence")
    prepare.add_argument("--hash", action="append", default=[])
    prepare.add_argument("--workers", type=int, default=4)
    cap_visuals = sub.add_parser("cap-visuals")
    cap_visuals.add_argument("--max-pages", type=int, default=MAX_VISUAL_PAGES_PER_BUNDLE)
    pilot_prepare = sub.choices["prepare-evidence-pilot"]
    pilot_prepare.add_argument("--workers", type=int, default=4)
    media = sub.add_parser("parse-media")
    media.add_argument("--model", default=DEFAULT_AUDIO_MODEL)
    media.add_argument("--limit", type=int)
    audit = sub.add_parser("audit-inputs")
    audit.add_argument("--skip-source-rehash", action="store_true")
    upload = sub.add_parser("upload-evidence")
    upload.add_argument("--workers", type=int, default=8)
    build = sub.add_parser("build-batches")
    build.add_argument("--model", default=DEFAULT_MODEL)
    build.add_argument("--shard-size", type=int, default=880)
    submit = sub.add_parser("submit")
    submit.add_argument("--mode", choices=("pilot", "remaining", "retry", "all"), required=True)
    poll = sub.add_parser("poll")
    poll.add_argument("--wait", action="store_true")
    poll.add_argument("--interval", type=int, default=60)
    sub.add_parser("parse-results")
    sub.add_parser("build-retry")
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--bootstrap-samples", type=int, default=10_000)
    aggregate.add_argument("--seed", type=int, default=20260715)
    sub.add_parser("validate")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    state = args.state.resolve()
    try:
        if args.command == "manifest":
            result = build_manifest(
                args.volume.resolve(), args.baselines.resolve(), state,
                include_rubric=args.include_rubric,
            )
        elif args.command == "select-evidence-pilot":
            result = select_evidence_pilot(state)
        elif args.command == "prepare-evidence-pilot":
            result = prepare_evidence_pilot(state, workers=args.workers)
        elif args.command == "prepare-evidence":
            result = prepare_evidence(state, only_hashes=set(args.hash) or None, workers=args.workers)
        elif args.command == "cap-visuals":
            result = cap_existing_visuals(state, max_pages=args.max_pages)
        elif args.command == "parse-media":
            result = parse_pending_media(state, model=args.model, limit=args.limit)
        elif args.command == "audit-inputs":
            result = frozen_input_audit(state, verify_source_hashes=not args.skip_source_rehash)
        elif args.command == "upload-evidence":
            result = upload_evidence(state, workers=args.workers)
        elif args.command == "build-batches":
            result = build_batch_requests(state, model=args.model, shard_size=args.shard_size)
        elif args.command == "submit":
            result = submit_batches(state, mode=args.mode)
        elif args.command == "poll":
            result = poll_batches(state, wait=args.wait, interval=args.interval)
        elif args.command == "parse-results":
            result = parse_batch_outputs(state)
        elif args.command == "build-retry":
            result = build_retry_requests(state)
        elif args.command == "aggregate":
            result = aggregate_results(state, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
        elif args.command == "validate":
            result = validation_report(state)
        else:
            raise AssertionError(args.command)
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except EvaluationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
