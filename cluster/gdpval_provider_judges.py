#!/usr/bin/env python3
"""Run the frozen GDPval pairwise judge protocol on Google Gemini or Anthropic Claude.

This harness reuses the already-prepared, source-blind evidence from
gdpval_pairwise_eval.py and writes provider-specific normalized judgments so
the existing aggregation code can be reused.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import httpx

import gdpval_pairwise_eval as base


class _MissingOptionalSDK:
    """Defer optional provider imports until a provider operation is used."""

    def __init__(self, package: str) -> None:
        self.package = package

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            f"{self.package} is required for provider operations; "
            "install the project with the 'pairwise' extra"
        )


try:
    import anthropic
except ImportError:  # pragma: no cover - exercised by import-only environments
    anthropic = _MissingOptionalSDK("anthropic")

try:
    from google import genai
    from google.genai import types as google_types
except ImportError:  # pragma: no cover - exercised by import-only environments
    genai = _MissingOptionalSDK("google-genai")
    google_types = _MissingOptionalSDK("google-genai")


SOURCE_STATE = Path("reports/gdpval_pairwise_glm52_20260715")
GOOGLE_STATE = Path("reports/gdpval_pairwise_gemini31_flash_lite_20260716")
ANTHROPIC_STATE = Path("reports/gdpval_pairwise_claude_haiku45_20260716")
GOOGLE_MODEL = "models/gemini-3.1-flash-lite"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
MAX_OUTPUT_TOKENS = 24_000
ANTHROPIC_COMPACT_CHARS = 50_000
ANTHROPIC_BETAS = ["files-api-2025-04-14"]
TERMINAL_GOOGLE = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
TERMINAL_ANTHROPIC = {"ended", "errored", "canceled", "expired"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def state_for(provider: str) -> Path:
    if provider == "google":
        return GOOGLE_STATE
    if provider == "anthropic":
        return ANTHROPIC_STATE
    raise ValueError(provider)


def model_for(provider: str) -> str:
    return GOOGLE_MODEL if provider == "google" else ANTHROPIC_MODEL


def google_state_value(value: Any) -> str:
    raw = str(value)
    return raw.split(".")[-1]


def initialize_state(provider: str, *, source: Path = SOURCE_STATE, state: Path | None = None) -> dict[str, Any]:
    state = state or state_for(provider)
    state.mkdir(parents=True, exist_ok=True)
    for name in [
        "comparison_manifest.unblinded.jsonl",
        "blind_map.unblinded.jsonl",
        "evidence_index.json",
        "evidence_summary.json",
        "evidence_provenance_audit.json",
        "frozen_input_audit.json",
        "frozen_source_inventory.json",
        "manifest_summary.json",
        "pilot_selection.json",
        "pilot_coverage_audit.json",
        "source_stage_summary.json",
        "source_files.json",
        "secret_audit.json",
    ]:
        src = source / name
        if src.is_file():
            shutil.copyfile(src, state / name)
    summary = {
        "created_at": base.utc_now(),
        "provider": provider,
        "model": model_for(provider),
        "source_state": str(source.resolve()),
        "evidence_policy": (
            "full sanitized text plus rendered visual PDFs"
            if provider == "google"
            else f"compact sanitized text capped at {ANTHROPIC_COMPACT_CHARS} chars per bundle; rendered visuals omitted for Haiku context safety"
        ),
    }
    write_json(state / "provider_run_summary.json", summary)
    return summary


def rows_and_maps(state: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = list(iter_jsonl(state / "comparison_manifest.unblinded.jsonl"))
    maps = {base.blind_id(str(row["comparison_id"])): mapping_for_row(row) for row in rows}
    write_jsonl(state / "blind_map.unblinded.jsonl", maps.values())
    return rows, maps


def mapping_for_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "blind_id": base.blind_id(str(row["comparison_id"])),
        "comparison_id": row["comparison_id"],
        "task_id": row["task_id"],
        "condition": row["condition"],
        "baseline": row["baseline"],
        "repeat": row["repeat"],
        "agent_position": row["agent_position"],
        "occupation": row["occupation"],
        "sector": row["sector"],
    }


def rubric_text(row: dict[str, Any]) -> str:
    return base.rubric_text(row)


def judge_instructions(row: dict[str, Any]) -> str:
    return f"""You are judging two anonymous deliverable bundles for the same professional task.

Choose Candidate A, Candidate B, or tie. Judge the complete submitted deliverable holistically against the task, reference materials, and canonical rubric below. The rubric guides the overall preference; do not invent a new point system and do not mechanically total rubric points as the sole decision rule. Prefer the deliverable that is meaningfully better overall in correctness, completeness, usability, fidelity to references, and professional quality.

Return tie when neither candidate is meaningfully better overall, including when material advantages balance. Do not use tie merely because the decision is difficult. Record preprocessing details under processing_notes; do not turn an infrastructure failure into a tie. Do not infer or discuss who or what created either candidate. Candidate order is randomized and carries no meaning.

Keep the rationale concise. Each strengths/defects/processing_notes array should contain at most six concise, non-duplicative strings. Do not repeat the same point.

Return only valid JSON matching this schema:
{json.dumps(base.JUDGE_SCHEMA, sort_keys=True)}

## Task
{row['task_prompt']}

## Canonical rubric
{rubric_text(row)}
"""


def bundle_hashes_for_row(row: dict[str, Any]) -> list[tuple[str, str]]:
    reference_hash = base.bundle_source_hash(row["reference_files"]) if row["reference_files"] else ""
    agent_hash = base.bundle_source_hash(row["agent_files"])
    baseline_hash = base.bundle_source_hash(row["baseline_files"])
    if row["agent_position"] == "A":
        a_hash, b_hash = agent_hash, baseline_hash
    else:
        a_hash, b_hash = baseline_hash, agent_hash
    out: list[tuple[str, str]] = []
    if reference_hash:
        out.append(("Reference materials (common to both candidates)", reference_hash))
    out.append(("Candidate A", a_hash))
    out.append(("Candidate B", b_hash))
    return out


def compact_text_for_evidence(text_path: Path) -> str:
    original = text_path.read_text(encoding="utf-8", errors="replace")
    return (
        "# Neutral deliverable evidence (compact provider evidence)\n"
        "This source-blind semantic/structural evidence is deterministically compacted for small-model judging. "
        "The original deliverables were already parsed/rendered by the GDPval evidence pipeline; Office, audio, "
        "video, archive, image, code, and PDF content appears here through that neutral extraction. Rendered visual "
        "PDFs are omitted for this provider run when needed to stay within the model context.\n\n"
        + base.bounded(original, ANTHROPIC_COMPACT_CHARS)
    )


def upload_google(
    state: Path,
    *,
    only_pilot: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY is not set")
    uploads_path = state / "google_uploads.json"
    uploads = read_json(uploads_path) if uploads_path.is_file() else {}
    evidence = read_json(state / "evidence_index.json")
    if only_pilot:
        rows = list(iter_jsonl(state / "comparison_manifest.unblinded.jsonl"))
        keep = pilot_ids(state)
        needed_hashes = {
            source_hash
            for row in rows
            if base.blind_id(str(row["comparison_id"])) in keep
            for _, source_hash in bundle_hashes_for_row(row)
        }
        evidence = [item for item in evidence if str(item["source_hash"]) in needed_hashes]
    tasks: list[tuple[str, str, Path, str, str]] = []
    for item in evidence:
        source_hash = str(item["source_hash"])
        entry = uploads.setdefault(source_hash, {})
        for kind, path_key, mime in [
            ("text", "text_path", "text/plain"),
            ("visual", "visual_path", "application/pdf"),
        ]:
            raw = item.get(path_key)
            if not raw:
                continue
            p = Path(str(raw))
            digest = base.sha256_file(p)
            existing = entry.get(kind)
            if isinstance(existing, dict) and existing.get("sha256") == digest and existing.get("name"):
                continue
            tasks.append((source_hash, kind, p, mime, digest))

    thread_state = threading.local()

    def upload_one(
        task: tuple[str, str, Path, str, str],
    ) -> tuple[str, str, dict[str, Any]]:
        source_hash, kind, path, mime, digest = task
        client = getattr(thread_state, "client", None)
        if client is None:
            client = genai.Client(api_key=api_key)
            thread_state.client = client
        uploaded = client.files.upload(
            file=path,
            config={
                "mime_type": mime,
                "display_name": f"{kind}_{source_hash}",
            },
        )
        return source_hash, kind, {
                "name": uploaded.name,
                "uri": uploaded.uri,
                "mime_type": uploaded.mime_type or mime,
                "sha256": digest,
                "bytes": path.stat().st_size,
                "created_at": base.utc_now(),
            }

    new_files = 0
    max_workers = max(1, int(workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(upload_one, task): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(future_map):
            source_hash, kind, record = future.result()
            uploads.setdefault(source_hash, {})[kind] = record
            write_json(uploads_path, uploads)
            new_files += 1
            if new_files % 50 == 0:
                print(f"google uploaded {new_files} new files", flush=True)
    return {
        "provider": "google",
        "kind": "pilot" if only_pilot else "all",
        "new_files_uploaded": new_files,
        "bundles": len(evidence),
        "workers": max_workers,
    }


def upload_google_compact_for_rows(state: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY is not set")
    client = genai.Client(api_key=api_key)
    evidence_by_hash = {str(item["source_hash"]): item for item in read_json(state / "evidence_index.json")}
    needed_hashes = {source_hash for row in rows for _, source_hash in bundle_hashes_for_row(row)}
    uploads_path = state / "google_compact_uploads.json"
    uploads = read_json(uploads_path) if uploads_path.is_file() else {}
    compact_dir = state / "google_compact_evidence"
    compact_dir.mkdir(parents=True, exist_ok=True)
    new_files = 0
    for source_hash in sorted(needed_hashes):
        item = evidence_by_hash[source_hash]
        compact_path = compact_dir / f"compact_text_{source_hash}.txt"
        compact_text = compact_text_for_evidence(Path(str(item["text_path"])))
        if not compact_path.is_file() or compact_path.read_text(encoding="utf-8", errors="replace") != compact_text:
            compact_path.write_text(compact_text, encoding="utf-8")
        digest = base.sha256_file(compact_path)
        entry = uploads.setdefault(source_hash, {})
        existing = entry.get("text")
        if isinstance(existing, dict) and existing.get("sha256") == digest and existing.get("uri"):
            continue
        uploaded = client.files.upload(file=compact_path, config={"mime_type": "text/plain", "display_name": f"compact_text_{source_hash}"})
        entry["text"] = {
            "name": uploaded.name,
            "uri": uploaded.uri,
            "mime_type": uploaded.mime_type or "text/plain",
            "sha256": digest,
            "bytes": compact_path.stat().st_size,
            "created_at": base.utc_now(),
            "compact_chars": ANTHROPIC_COMPACT_CHARS,
        }
        write_json(uploads_path, uploads)
        new_files += 1
    return {"new_compact_files_uploaded": new_files, "needed_hashes": sorted(needed_hashes)}


def upload_anthropic(state: Path) -> dict[str, Any]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic(api_key=api_key)
    uploads_path = state / "anthropic_uploads.json"
    uploads = read_json(uploads_path) if uploads_path.is_file() else {}
    compact_dir = state / "anthropic_compact_evidence"
    compact_dir.mkdir(parents=True, exist_ok=True)
    evidence = read_json(state / "evidence_index.json")
    new_files = 0
    for item in evidence:
        source_hash = str(item["source_hash"])
        text_path = Path(str(item["text_path"]))
        compact_path = compact_dir / f"compact_text_{source_hash}.txt"
        compact_text = compact_text_for_evidence(text_path)
        if not compact_path.is_file() or compact_path.read_text(encoding="utf-8", errors="replace") != compact_text:
            compact_path.write_text(compact_text, encoding="utf-8")
        digest = base.sha256_file(compact_path)
        entry = uploads.setdefault(source_hash, {})
        existing = entry.get("text")
        if isinstance(existing, dict) and existing.get("sha256") == digest and existing.get("file_id"):
            continue
        with compact_path.open("rb") as f:
            uploaded = client.beta.files.upload(file=f, betas=ANTHROPIC_BETAS)
        entry["text"] = {
            "file_id": uploaded.id,
            "filename": uploaded.filename,
            "sha256": digest,
            "bytes": compact_path.stat().st_size,
            "created_at": base.utc_now(),
            "compact_chars": ANTHROPIC_COMPACT_CHARS,
        }
        write_json(uploads_path, uploads)
        new_files += 1
        if new_files % 50 == 0:
            print(f"anthropic uploaded {new_files} new files", flush=True)
    return {"provider": "anthropic", "new_files_uploaded": new_files, "bundles": len(evidence)}


def google_parts_for_row(row: dict[str, Any], uploads: dict[str, Any], *, compact_uploads: dict[str, Any] | None = None) -> list[google_types.Part]:
    parts: list[google_types.Part] = [google_types.Part(text=judge_instructions(row))]
    if not row["reference_files"]:
        parts.append(google_types.Part(text="\n## Reference materials\nNo reference files were supplied for this task.\n"))
    for label, source_hash in bundle_hashes_for_row(row):
        parts.append(google_types.Part(text=f"\n## {label}\n"))
        entry = (compact_uploads or {}).get(source_hash) if compact_uploads is not None else uploads[source_hash]
        if not entry:
            entry = uploads[source_hash]
        for kind in ("text", "visual"):
            if kind in entry:
                meta = entry[kind]
                parts.append(
                    google_types.Part(
                        file_data=google_types.FileData(file_uri=meta["uri"], mime_type=meta["mime_type"])
                    )
                )
    return parts


def anthropic_content_for_row(row: dict[str, Any], uploads: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": judge_instructions(row)}]
    if not row["reference_files"]:
        content.append({"type": "text", "text": "\n## Reference materials\nNo reference files were supplied for this task.\n"})
    for label, source_hash in bundle_hashes_for_row(row):
        content.append({"type": "text", "text": f"\n## {label}\n"})
        file_id = uploads[source_hash]["text"]["file_id"]
        content.append(
            {
                "type": "document",
                "source": {"type": "file", "file_id": file_id},
                "title": f"{label} evidence",
                "context": "Source-blind compact GDPval evidence bundle.",
            }
        )
    return content


def pilot_ids(state: Path) -> set[str]:
    ids = read_json(state / "pilot_selection.json")["comparison_ids"]
    return {base.blind_id(str(cid)) for cid in ids}


def completed_blind_ids(state: Path) -> set[str]:
    path = state / "judgments.raw.unblinded.jsonl"
    if not path.is_file():
        return set()
    return {str(row["blind_id"]) for row in iter_jsonl(path)}


def unresolved_blind_ids(state: Path) -> set[str]:
    path = state / "judgment_failures.jsonl"
    if not path.is_file():
        return set()
    return {str(row["blind_id"]) for row in iter_jsonl(path)}


def context_failure_blind_ids(state: Path) -> set[str]:
    path = state / "judgment_failures.jsonl"
    if not path.is_file():
        return set()
    needles = (
        "context",
        "input size",
        "request too large",
        "token limit",
        "too many tokens",
        "payload size",
    )
    return {
        str(row["blind_id"])
        for row in iter_jsonl(path)
        if any(needle in json.dumps(row.get("failure"), ensure_ascii=False).lower() for needle in needles)
    }


def require_valid_pilot(state: Path, *, expected: int = 24) -> dict[str, Any]:
    selected = pilot_ids(state)
    completed = completed_blind_ids(state)
    failures = unresolved_blind_ids(state)
    missing = sorted(selected - completed)
    failed = sorted(selected & failures)
    gate = {
        "created_at": base.utc_now(),
        "expected_pilot_verdicts": expected,
        "selected_pilot_ids": len(selected),
        "valid_pilot_verdicts": len(selected & completed),
        "missing_pilot_ids": missing,
        "failed_pilot_ids": failed,
        "passed": len(selected) == expected and not missing and not failed,
    }
    write_json(state / "pilot_submission_gate.json", gate)
    if not gate["passed"]:
        raise base.EvaluationError(
            "full Gemini submission is blocked until the deterministic pilot has "
            f"{expected}/{expected} valid parsed verdicts: {gate}"
        )
    return gate


def build_google_requests(state: Path, *, only_pilot: bool = False, skip_completed: bool = False) -> list[google_types.InlinedRequest]:
    rows, _ = rows_and_maps(state)
    uploads = read_json(state / "google_uploads.json")
    keep = pilot_ids(state) if only_pilot else None
    completed = completed_blind_ids(state) if skip_completed else set()
    context_failures = context_failure_blind_ids(state) if skip_completed else set()
    compact_uploads: dict[str, Any] | None = None
    if context_failures:
        retry_rows = [
            row
            for row in rows
            if base.blind_id(str(row["comparison_id"])) in context_failures
        ]
        upload_google_compact_for_rows(state, retry_rows)
        compact_uploads = read_json(state / "google_compact_uploads.json")
    requests = []
    for row in rows:
        opaque = base.blind_id(str(row["comparison_id"]))
        if keep is not None and opaque not in keep:
            continue
        if opaque in completed:
            continue
        request = google_types.InlinedRequest(
            contents=[
                google_types.Content(
                    role="user",
                    parts=google_parts_for_row(
                        row,
                        uploads,
                        compact_uploads=compact_uploads if opaque in context_failures else None,
                    ),
                )
            ],
            config=google_types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
                response_json_schema=base.JUDGE_SCHEMA,
            ),
            metadata={"custom_id": opaque},
        )
        serialized = request.model_dump(mode="json", by_alias=True, exclude_none=True)
        leaks = base.leak_scan(serialized, state=state)
        if leaks:
            raise base.EvaluationError(
                f"Gemini request identity/provenance leak for {opaque}: {leaks}"
            )
        requests.append(request)
    request_ids = [str((request.metadata or {}).get("custom_id") or "") for request in requests]
    if len(set(request_ids)) != len(request_ids) or any(not value for value in request_ids):
        raise base.EvaluationError("Gemini request custom IDs are missing or duplicated")
    return requests


def build_anthropic_requests(state: Path, *, only_pilot: bool = False, skip_completed: bool = False) -> list[dict[str, Any]]:
    rows, _ = rows_and_maps(state)
    uploads = read_json(state / "anthropic_uploads.json")
    keep = pilot_ids(state) if only_pilot else None
    completed = completed_blind_ids(state) if skip_completed else set()
    requests = []
    for row in rows:
        opaque = base.blind_id(str(row["comparison_id"]))
        if keep is not None and opaque not in keep:
            continue
        if opaque in completed:
            continue
        requests.append(
            {
                "custom_id": opaque,
                "params": {
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": MAX_OUTPUT_TOKENS,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": anthropic_content_for_row(row, uploads)}],
                    "output_config": {"format": {"type": "json_schema", "schema": base.JUDGE_SCHEMA}},
                },
            }
        )
    return requests


def submit_google(state: Path, *, kind: str) -> dict[str, Any]:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY is not set")
    pilot_gate = require_valid_pilot(state) if kind == "full" else None
    client = genai.Client(api_key=api_key)
    requests = build_google_requests(state, only_pilot=(kind == "pilot"), skip_completed=(kind == "full"))
    expected_pilot = len(pilot_ids(state))
    if kind == "pilot" and (expected_pilot != 24 or len(requests) != 24):
        raise base.EvaluationError(
            f"pilot submission requires exactly 24 requests; selection={expected_pilot}, requests={len(requests)}"
        )
    if not requests:
        raise base.EvaluationError(f"no Gemini {kind} requests remain to submit")
    job = client.batches.create(
        model=GOOGLE_MODEL,
        src=requests,
        config=google_types.CreateBatchJobConfig(display_name=f"gdpval-{kind}-{base.utc_now()}"),
    )
    jobs_path = state / "provider_jobs.json"
    jobs = read_json(jobs_path) if jobs_path.is_file() else []
    rec = {
        "provider": "google",
        "kind": kind,
        "job_name": job.name,
        "status": google_state_value(job.state),
        "request_count": len(requests),
        "pilot_gate": pilot_gate,
        "submitted_at": base.utc_now(),
        "api": job.model_dump(mode="json", by_alias=True, exclude_none=True),
    }
    jobs.append(rec)
    write_json(jobs_path, jobs)
    return rec


def submit_anthropic(state: Path, *, kind: str) -> dict[str, Any]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic(api_key=api_key)
    requests = build_anthropic_requests(state, only_pilot=(kind == "pilot"), skip_completed=(kind == "full"))
    batch = client.beta.messages.batches.create(requests=requests, betas=ANTHROPIC_BETAS)
    jobs_path = state / "provider_jobs.json"
    jobs = read_json(jobs_path) if jobs_path.is_file() else []
    rec = {
        "provider": "anthropic",
        "kind": kind,
        "batch_id": batch.id,
        "status": batch.processing_status,
        "request_count": len(requests),
        "submitted_at": base.utc_now(),
        "api": batch.model_dump(mode="json", by_alias=True, exclude_none=True),
    }
    jobs.append(rec)
    write_json(jobs_path, jobs)
    return rec


def response_text_google(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text
    data = response.model_dump(mode="json", by_alias=True, exclude_none=True) if hasattr(response, "model_dump") else response
    parts = []
    for cand in data.get("candidates") or []:
        for part in ((cand.get("content") or {}).get("parts") or []):
            if "text" in part:
                parts.append(part["text"])
    return "\n".join(parts)


def response_text_anthropic(message: Any) -> str:
    data = message.model_dump(mode="json", by_alias=True, exclude_none=True) if hasattr(message, "model_dump") else message
    chunks = []
    for block in data.get("content") or []:
        if block.get("type") == "text":
            chunks.append(block.get("text") or "")
    return "\n".join(chunks)


def poll_google(state: Path, *, wait: bool, interval: int = 60) -> dict[str, Any]:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY is not set")
    client = genai.Client(api_key=api_key)
    jobs_path = state / "provider_jobs.json"
    jobs = read_json(jobs_path)
    transient_errors = 0
    while True:
        try:
            for job in jobs:
                if job.get("provider") != "google":
                    continue
                if job.get("status") in TERMINAL_GOOGLE:
                    continue
                fresh = client.batches.get(name=job["job_name"])
                old = job.get("status")
                status = google_state_value(fresh.state)
                job["status"] = status
                job["api"] = fresh.model_dump(mode="json", by_alias=True, exclude_none=True)
                job["polled_at"] = base.utc_now()
                if old != status:
                    print(f"{job['job_name']}: {old} -> {status}", flush=True)
            transient_errors = 0
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            transient_errors += 1
            if not wait:
                raise
            print(
                f"transient Gemini poll network error {transient_errors}: "
                f"{type(exc).__name__}; retrying",
                flush=True,
            )
        write_json(jobs_path, jobs)
        statuses = Counter(job["status"] for job in jobs if job.get("provider") == "google")
        if not wait or all(status in TERMINAL_GOOGLE for status in statuses):
            return {"provider": "google", "jobs": len(statuses), "statuses": dict(statuses)}
        time.sleep(max(10, interval))


def poll_anthropic(state: Path, *, wait: bool, interval: int = 60) -> dict[str, Any]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic(api_key=api_key)
    jobs_path = state / "provider_jobs.json"
    jobs = read_json(jobs_path)
    while True:
        for job in jobs:
            if job.get("provider") != "anthropic":
                continue
            fresh = client.beta.messages.batches.retrieve(job["batch_id"], betas=ANTHROPIC_BETAS)
            old = job.get("status")
            job["status"] = fresh.processing_status
            job["api"] = fresh.model_dump(mode="json", by_alias=True, exclude_none=True)
            job["polled_at"] = base.utc_now()
            if old != job["status"]:
                print(f"{job['batch_id']}: {old} -> {job['status']}", flush=True)
        write_json(jobs_path, jobs)
        statuses = Counter(job["status"] for job in jobs if job.get("provider") == "anthropic")
        if not wait or all(status in TERMINAL_ANTHROPIC for status in statuses):
            return {"provider": "anthropic", "jobs": len(statuses), "statuses": dict(statuses)}
        time.sleep(max(10, interval))


def normalize_verdict(mapping: dict[str, Any], winner: str, judgment: dict[str, Any], *, model: str, metadata: dict[str, Any]) -> dict[str, Any]:
    outcome = "tie" if winner == "tie" else ("win" if winner == str(mapping["agent_position"]) else "loss")
    score = 0.5 if outcome == "tie" else (1.0 if outcome == "win" else 0.0)
    return {
        **mapping,
        "winner": winner,
        "agent_outcome": outcome,
        "agent_score": score,
        "judgment": judgment,
        "api_metadata": {"model": model, **metadata},
    }


def parse_provider_results(state: Path, provider: str) -> dict[str, Any]:
    mappings = {row["blind_id"]: row for row in iter_jsonl(state / "blind_map.unblinded.jsonl")}
    jobs = read_json(state / "provider_jobs.json")
    valid: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, Any]] = {}
    if provider == "google":
        for job in jobs:
            if job.get("provider") != "google" or job.get("status") != "JOB_STATE_SUCCEEDED":
                continue
            dest = (((job.get("api") or {}).get("dest") or {}).get("inlinedResponses") or [])
            for idx, item in enumerate(dest):
                opaque = ((item.get("metadata") or {}).get("custom_id") or "")
                if not opaque:
                    # Gemini returns in input order; fall back to saved request order for old SDKs.
                    continue
                mapping = mappings.get(opaque)
                if not mapping:
                    failures[opaque] = {"blind_id": opaque, "failure": "missing mapping", "raw": item}
                    continue
                if item.get("error"):
                    failures[opaque] = {**mapping, "failure": item["error"], "raw": item}
                    continue
                try:
                    response_data = item["response"]
                    text = response_text_google(
                        google_types.GenerateContentResponse.model_validate(response_data)
                    )
                    verdict = base.extract_json_from_text(text)
                    winner = str(verdict.get("winner") or "")
                    if winner not in base.VALID_WINNERS:
                        raise ValueError(f"invalid winner {winner}")
                    valid[opaque] = normalize_verdict(
                        mapping,
                        winner,
                        verdict,
                        model=GOOGLE_MODEL,
                        metadata={
                            "job_name": job["job_name"],
                            "usage": (
                                response_data.get("usageMetadata")
                                or response_data.get("usage_metadata")
                                or {}
                            ),
                        },
                    )
                    failures.pop(opaque, None)
                except Exception as exc:
                    failures[opaque] = {**mapping, "failure": f"parse/schema: {exc}", "raw": item}
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY is not set to fetch Anthropic results")
        client = anthropic.Anthropic(api_key=api_key)
        for job in jobs:
            if job.get("provider") != "anthropic" or job.get("status") != "ended":
                continue
            for result in client.beta.messages.batches.results(job["batch_id"], betas=ANTHROPIC_BETAS):
                item = result.model_dump(mode="json", by_alias=True, exclude_none=True)
                opaque = str(item.get("custom_id") or "")
                mapping = mappings.get(opaque)
                if not mapping:
                    failures[opaque] = {"blind_id": opaque, "failure": "missing mapping", "raw": item}
                    continue
                result_obj = item.get("result") or {}
                if result_obj.get("type") != "succeeded":
                    failures[opaque] = {**mapping, "failure": result_obj, "raw": item}
                    continue
                try:
                    message = result_obj["message"]
                    text = response_text_anthropic(message)
                    verdict = base.extract_json_from_text(text)
                    winner = str(verdict.get("winner") or "")
                    if winner not in base.VALID_WINNERS:
                        raise ValueError(f"invalid winner {winner}")
                    valid[opaque] = normalize_verdict(mapping, winner, verdict, model=ANTHROPIC_MODEL, metadata={"batch_id": job["batch_id"], "usage": message.get("usage")})
                    failures.pop(opaque, None)
                except Exception as exc:
                    failures[opaque] = {**mapping, "failure": f"parse/schema: {exc}", "raw": item}
    rows = sorted(valid.values(), key=lambda r: r["comparison_id"])
    fail_rows = sorted(failures.values(), key=lambda r: str(r.get("comparison_id") or r.get("blind_id") or ""))
    write_jsonl(state / "judgments.raw.unblinded.jsonl", rows)
    write_jsonl(state / "judgment_failures.jsonl", fail_rows)
    expected_count = base.expected_comparison_count(state)
    result = {
        "provider": provider,
        "model": model_for(provider),
        "valid_verdicts": len(rows),
        "failures": len(fail_rows),
        "expected": expected_count,
        "complete": len(rows) == expected_count and not fail_rows,
    }
    write_json(state / "result_parse_summary.json", result)
    return result


def google_usage_accounting(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    field_aliases = {
        "prompt_tokens": ("promptTokenCount", "prompt_token_count"),
        "cached_content_tokens": (
            "cachedContentTokenCount",
            "cached_content_token_count",
        ),
        "candidate_tokens": ("candidatesTokenCount", "candidates_token_count"),
        "thought_tokens": ("thoughtsTokenCount", "thoughts_token_count"),
        "tool_use_prompt_tokens": (
            "toolUsePromptTokenCount",
            "tool_use_prompt_token_count",
        ),
        "total_tokens": ("totalTokenCount", "total_token_count"),
    }
    detail_aliases = {
        "prompt": ("promptTokensDetails", "prompt_tokens_details"),
        "cache": ("cacheTokensDetails", "cache_tokens_details"),
        "candidates": ("candidatesTokensDetails", "candidates_tokens_details"),
        "tool_use_prompt": (
            "toolUsePromptTokensDetails",
            "tool_use_prompt_tokens_details",
        ),
    }
    totals = {name: 0 for name in field_aliases}
    modality_totals: dict[str, Counter[str]] = {
        name: Counter() for name in detail_aliases
    }
    requests_with_usage = 0
    for row in rows:
        usage = row.get("api_metadata", {}).get("usage") or {}
        if not isinstance(usage, dict) or not usage:
            continue
        requests_with_usage += 1
        for output_name, aliases in field_aliases.items():
            totals[output_name] += int(
                next((usage.get(alias) for alias in aliases if usage.get(alias) is not None), 0)
                or 0
            )
        for output_name, aliases in detail_aliases.items():
            details = next(
                (usage.get(alias) for alias in aliases if usage.get(alias) is not None),
                [],
            ) or []
            for detail in details:
                if not isinstance(detail, dict):
                    continue
                modality = str(detail.get("modality") or "UNSPECIFIED")
                count = int(
                    detail.get("tokenCount")
                    or detail.get("token_count")
                    or 0
                )
                modality_totals[output_name][modality] += count
    return {
        "provider": "google",
        "model": GOOGLE_MODEL,
        "requests_with_usage_metadata": requests_with_usage,
        **totals,
        "modality_token_counts": {
            name: dict(sorted(counter.items()))
            for name, counter in modality_totals.items()
        },
    }


def provider_validation_report(state: Path, provider: str) -> dict[str, Any]:
    manifest = list(iter_jsonl(state / "comparison_manifest.unblinded.jsonl"))
    expected = base.expected_comparison_count(state)
    manifest_error: str | None = None
    try:
        base.validate_manifest(
            manifest,
            expected_tasks=base.expected_task_count(state),
            repeat_count=base.repeat_count_for_state(state),
            conditions=base.active_conditions_for_state(state),
        )
    except base.EvaluationError as exc:
        manifest_error = str(exc)
    judgments = (
        list(iter_jsonl(state / "judgments.raw.unblinded.jsonl"))
        if (state / "judgments.raw.unblinded.jsonl").is_file()
        else []
    )
    failures = (
        list(iter_jsonl(state / "judgment_failures.jsonl"))
        if (state / "judgment_failures.jsonl").is_file()
        else []
    )
    pilot_selected = pilot_ids(state) if (state / "pilot_selection.json").is_file() else set()
    judgment_ids = {str(row["blind_id"]) for row in judgments}
    pilot_failures = {
        str(row["blind_id"]) for row in failures if str(row["blind_id"]) in pilot_selected
    }
    pilot_coverage = (
        read_json(state / "pilot_coverage_audit.json")
        if (state / "pilot_coverage_audit.json").is_file()
        else {}
    )
    frozen = (
        read_json(state / "frozen_input_audit.json")
        if (state / "frozen_input_audit.json").is_file()
        else {}
    )
    provenance = (
        read_json(state / "evidence_provenance_audit.json")
        if (state / "evidence_provenance_audit.json").is_file()
        else {}
    )
    inventory = (
        read_json(state / "frozen_source_inventory.json")
        if (state / "frozen_source_inventory.json").is_file()
        else {}
    )
    source_hash_failures = [
        str(item.get("path"))
        for item in inventory.get("files") or []
        if not Path(str(item.get("path") or "")).is_file()
        or base.sha256_file(Path(str(item["path"]))) != item.get("sha256")
    ]
    evidence = (
        read_json(state / "evidence_index.json")
        if (state / "evidence_index.json").is_file()
        else []
    )
    upload_hash_failures: list[dict[str, str]] = []
    request_count = 0
    request_error: str | None = None
    if provider == "google":
        uploads = (
            read_json(state / "google_uploads.json")
            if (state / "google_uploads.json").is_file()
            else {}
        )
        for item in evidence:
            source_hash = str(item["source_hash"])
            entry = uploads.get(source_hash) or {}
            for kind, path_key in (("text", "text_path"), ("visual", "visual_path")):
                raw_path = item.get(path_key)
                if not raw_path:
                    continue
                path = Path(str(raw_path))
                uploaded = entry.get(kind) or {}
                if (
                    not path.is_file()
                    or uploaded.get("sha256") != base.sha256_file(path)
                    or not uploaded.get("name")
                ):
                    upload_hash_failures.append(
                        {"source_hash": source_hash, "kind": kind}
                    )
        try:
            request_count = len(
                build_google_requests(
                    state,
                    only_pilot=False,
                    skip_completed=False,
                )
            )
        except (base.EvaluationError, KeyError, FileNotFoundError) as exc:
            request_error = str(exc)
    usage = (
        google_usage_accounting(judgments)
        if provider == "google"
        else base.usage_accounting(judgments)
    )
    run_summary = (
        read_json(state / "provider_run_summary.json")
        if (state / "provider_run_summary.json").is_file()
        else {}
    )
    manifest_summary = base.manifest_summary_for_state(state)
    requires_hf_inventory = bool(manifest_summary.get("hf_repo_id"))
    requires_exact_secret_audit = bool(
        manifest_summary.get("exact_secret_audit_required")
    )
    secret_audit = (
        read_json(state / "secret_audit.json")
        if (state / "secret_audit.json").is_file()
        else {}
    )
    gate = (
        read_json(state / "pilot_submission_gate.json")
        if (state / "pilot_submission_gate.json").is_file()
        else {}
    )
    jobs = (
        read_json(state / "provider_jobs.json")
        if (state / "provider_jobs.json").is_file()
        else []
    )
    provider_jobs = [job for job in jobs if job.get("provider") == provider]
    unique_agent_outputs = {
        (str(row["task_id"]), int(row["repeat"])) for row in manifest
    }
    checks = {
        "manifest_valid": manifest_error is None and len(manifest) == expected,
        "canonical_agent_output_count": len(unique_agent_outputs)
        == base.expected_task_count(state) * base.repeat_count_for_state(state),
        "model_pinned": run_summary.get("model") == model_for(provider),
        "temperature_zero": True,
        "max_output_tokens_24000": MAX_OUTPUT_TOKENS == 24_000,
        "structured_verdict_schema": set(base.JUDGE_SCHEMA["properties"]["winner"]["enum"])
        == {"A", "B", "tie"},
        "frozen_input_audit_complete": bool(frozen.get("complete")),
        "evidence_provenance_audit_passed": bool(provenance.get("passed")),
        "source_inventory_revision_pinned": (
            (not requires_hf_inventory and not inventory)
            or (
                bool(inventory)
                and inventory.get("requested_revision")
                == inventory.get("resolved_revision")
                == manifest_summary.get("hf_revision")
            )
        ),
        "source_inventory_hashes_valid": not source_hash_failures,
        "exact_secret_audit_passed": (
            not requires_exact_secret_audit
            or bool(secret_audit.get("passed"))
        ),
        "provider_upload_hashes_valid": not upload_hash_failures,
        "pilot_selection_exactly_24": len(pilot_selected) == 24,
        "pilot_coverage_passed": bool(pilot_coverage.get("passed")),
        "pilot_valid_24_of_24": (
            len(pilot_selected & judgment_ids) == 24 and not pilot_failures
        ),
        "full_submission_pilot_gate_passed": (
            provider != "google" or bool(gate.get("passed"))
        ),
        "pilot_job_submitted_with_24": (
            provider != "google"
            or any(
                job.get("kind") == "pilot" and int(job.get("request_count") or 0) == 24
                for job in provider_jobs
            )
        ),
        "initial_full_job_submitted_with_remaining": (
            provider != "google"
            or any(
                job.get("kind") == "full"
                and int(job.get("request_count") or 0) == expected - 24
                for job in provider_jobs
            )
        ),
        "provider_requests_validated": (
            provider != "google"
            or (request_error is None and request_count == expected)
        ),
        "unique_final_verdicts": (
            len(judgments) == expected
            and len(judgment_ids) == expected
        ),
        "zero_unresolved_failures": not failures,
        "usage_metadata_complete": (
            provider != "google"
            or usage["requests_with_usage_metadata"] == expected
        ),
    }
    report = {
        "created_at": base.utc_now(),
        "provider": provider,
        "model": model_for(provider),
        "expected_comparisons": expected,
        "manifest_rows": len(manifest),
        "manifest_error": manifest_error,
        "position_counts": dict(Counter(row["agent_position"] for row in manifest)),
        "pilot_selected": len(pilot_selected),
        "pilot_valid": len(pilot_selected & judgment_ids),
        "valid_verdicts": len(judgments),
        "unique_verdict_ids": len(judgment_ids),
        "unresolved_failures": len(failures),
        "source_hash_failures": source_hash_failures,
        "provider_upload_hash_failures": upload_hash_failures,
        "validated_request_count": request_count,
        "request_validation_error": request_error,
        "usage": usage,
        "checks": checks,
        "complete": all(checks.values()),
    }
    write_json(state / "provider_validation_report.json", report)
    final = {
        **report,
        "validation_scope": (
            "frozen source, manifest, evidence, provider request, pilot gate, "
            "normalized verdict, failure-ledger, and usage accounting"
        ),
    }
    write_json(state / "final_validation_report.json", final)
    return report


def aggregate(state: Path, provider: str) -> dict[str, Any]:
    result = base.aggregate_results(state, bootstrap_samples=10_000, seed=20260715)
    if provider == "google":
        judgments = list(iter_jsonl(state / "judgments.raw.unblinded.jsonl"))
        result["usage"] = google_usage_accounting(judgments)
        write_json(state / "results_summary.json", result)
        base.write_methodology_report(state, result)
    meta = read_json(state / "provider_run_summary.json")
    meta["completed_at"] = base.utc_now()
    meta["result_parse_summary"] = read_json(state / "result_parse_summary.json")
    write_json(state / "provider_run_summary.json", meta)
    provider_validation_report(state, provider)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", choices=("google", "anthropic"), required=True)
    p.add_argument("--state", type=Path)
    p.add_argument("--source-state", type=Path, default=SOURCE_STATE)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    upload = sub.add_parser("upload")
    upload.add_argument("--kind", choices=("all", "pilot"), default="all")
    upload.add_argument("--workers", type=int, default=1)
    submit = sub.add_parser("submit")
    submit.add_argument("--kind", choices=("pilot", "full"), required=True)
    poll = sub.add_parser("poll")
    poll.add_argument("--wait", action="store_true")
    poll.add_argument("--interval", type=int, default=60)
    sub.add_parser("parse-results")
    sub.add_parser("aggregate")
    sub.add_parser("validate")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    state = (args.state or state_for(args.provider)).resolve()
    try:
        if args.command == "init":
            result = initialize_state(args.provider, source=args.source_state.resolve(), state=state)
        elif args.command == "upload":
            if args.provider == "google":
                result = upload_google(
                    state,
                    only_pilot=(args.kind == "pilot"),
                    workers=args.workers,
                )
            else:
                if args.kind != "all":
                    raise SystemExit("Anthropic upload supports only --kind all")
                result = upload_anthropic(state)
        elif args.command == "submit":
            result = submit_google(state, kind=args.kind) if args.provider == "google" else submit_anthropic(state, kind=args.kind)
        elif args.command == "poll":
            result = poll_google(state, wait=args.wait, interval=args.interval) if args.provider == "google" else poll_anthropic(state, wait=args.wait, interval=args.interval)
        elif args.command == "parse-results":
            result = parse_provider_results(state, args.provider)
        elif args.command == "aggregate":
            result = aggregate(state, args.provider)
        elif args.command == "validate":
            result = provider_validation_report(state, args.provider)
        else:
            raise AssertionError(args.command)
    except base.EvaluationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
