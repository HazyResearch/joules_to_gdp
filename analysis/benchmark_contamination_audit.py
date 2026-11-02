#!/usr/bin/env python3
"""Audit GDPval model artifacts for exact overlap with expert deliverables.

The audit is intentionally conservative.  It reports whole-file and semantic
OOXML/ZIP-member identity, plus long exact word sequences that do not already
appear in the task prompt or shared reference files.  Crosswalk rows with weak
prompt alignment are retained in the machine-readable output but excluded from
the headline contamination screen.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_reports" / "benchmark_contamination_audit_20260920"
NGRAM = 12
HIGH_CONFIDENCE_JACCARD = 0.80
PHRASE_FLAG_WORDS = 30

CONFIGS = {
    "GLM-5.2": {
        "report": ROOT / "reports/gdpval_pairwise_gemini31_flash_lite_20260716",
        "crosswalk": ROOT / "reports/gdpval_pairwise_glm52_20260715/crosswalk.json",
    },
    "Qwen3.6-27B": {
        "report": ROOT / "reports/qwen36_27b_norubric_gemini31_flash_lite_20260723",
        "crosswalk": ROOT / "reports/qwen36_27b_norubric_source_20260723/crosswalk.json",
    },
    "Gemma 4 31B": {
        "report": ROOT / "reports/gemma4_31b_norubric_gemini31_flash_lite_20260725",
        "crosswalk": ROOT / "reports/gemma4_31b_norubric_source_20260725/crosswalk.json",
    },
    "GPT-OSS 120B": {
        "report": ROOT / "reports/gpt_oss_120b_norubric_gemini31_flash_lite_20260724",
        "crosswalk": ROOT / "reports/gpt_oss_120b_norubric_source_20260724/crosswalk.json",
    },
    "GPT-OSS 20B": {
        "report": ROOT / "reports/gpt_oss_20b_norubric_gemini31_flash_lite_20260724",
        "crosswalk": ROOT / "reports/gpt_oss_20b_norubric_source_20260724/crosswalk.json",
    },
}


def recorded_path(value: str) -> Path:
    """Rebase archived local paths onto the checkout containing staged inputs."""
    path = Path(value)
    if not path.is_absolute():
        return ROOT / path
    if path.is_relative_to(ROOT):
        return path
    for directory in ("reports", "outputs", "GDPVal_220_Tasks"):
        marker = f"/{directory}/"
        if marker in value:
            return ROOT / directory / value.split(marker, 1)[1]
    raise ValueError(f"Archived evidence path cannot be rebased: {value}")

WORD_RE = re.compile(r"[\w]+(?:['’-][\w]+)*", re.UNICODE)
SEMANTIC_MEMBER_RE = re.compile(
    r"^(?:word/(?:document|header\d+|footer\d+|footnotes|endnotes)\.xml|"
    r"xl/(?:worksheets/[^/]+|sharedStrings)\.xml|"
    r"ppt/(?:slides/slide\d+|notesSlides/notesSlide\d+)\.xml|"
    r"[^/]+\.(?:csv|tsv|txt|md|html?))$",
    re.I,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_key(files: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted({str(item.get("sha256", "")) for item in files if item.get("sha256")}))


def load_evidence(report: Path) -> dict[tuple[str, ...], dict[str, Any]]:
    index = json.loads((report / "evidence_index.json").read_text())
    result: dict[tuple[str, ...], dict[str, Any]] = {}
    for item in index:
        audit_path = recorded_path(item["audit_path"])
        try:
            audit = json.loads(audit_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        key = file_key(audit.get("source_files", []))
        if not key:
            continue
        text_path = recorded_path(item["text_path"])
        text = text_path.read_text(errors="replace") if text_path.exists() else ""
        result.setdefault(
            key,
            {
                "text": text,
                "bundle_id": item.get("bundle_id"),
                "audit_path": str(audit_path),
                "source_files": audit.get("source_files", []),
            },
        )
    return result


def tokens(text: str) -> list[str]:
    return [m.group(0).lower().replace("’", "'") for m in WORD_RE.finditer(text)]


def semantic_text(text: str) -> str:
    """Remove extractor boilerplate while retaining document/cell contents."""
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith((
            "# Neutral deliverable evidence",
            "The following is a source-blind semantic and structural extraction.",
            "# File ",
            "Type: ",
            "Metadata: ",
            "## Sheet ",
        )):
            continue
        if stripped.startswith("{") and any(
            marker in stripped for marker in ('"max_row"', '"sheet_count"', '"pages"')
        ):
            continue
        kept.append(stripped)
    return "\n".join(kept)


def shingles(items: list[str], n: int = NGRAM) -> dict[tuple[str, ...], list[int]]:
    found: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx in range(len(items) - n + 1):
        key = tuple(items[idx : idx + n])
        if len(found[key]) < 4:
            found[key].append(idx)
    return found


def longest_unexplained_match(
    agent_text: str, baseline_text: str, control_text: str
) -> dict[str, Any]:
    a = tokens(semantic_text(agent_text))
    b = tokens(semantic_text(baseline_text))
    c = tokens(semantic_text(control_text))
    if len(a) < NGRAM or len(b) < NGRAM:
        return {"words": 0, "phrase": "", "agent_words": len(a), "baseline_words": len(b)}

    b_idx = shingles(b)
    c_set = set(shingles(c)) if len(c) >= NGRAM else set()
    a_counts = Counter(tuple(a[i : i + NGRAM]) for i in range(len(a) - NGRAM + 1))
    best_start = 0
    best_len = 0
    for ai in range(len(a) - NGRAM + 1):
        key = tuple(a[ai : ai + NGRAM])
        positions = b_idx.get(key)
        if not positions or key in c_set or a_counts[key] > 3 or len(positions) > 3:
            continue
        for bi in positions:
            left = 0
            while ai - left - 1 >= 0 and bi - left - 1 >= 0 and a[ai - left - 1] == b[bi - left - 1]:
                left += 1
            right = NGRAM
            while ai + right < len(a) and bi + right < len(b) and a[ai + right] == b[bi + right]:
                right += 1
            run_start = ai - left
            run_end = ai + right
            # Split the exact run wherever its N-word window also occurs in
            # prompt/reference evidence. This avoids treating unchanged parts
            # of a supplied workbook or form as suspicious output overlap.
            unique_windows = [
                tuple(a[pos : pos + NGRAM]) not in c_set
                for pos in range(run_start, run_end - NGRAM + 1)
            ]
            segment_start = 0
            for offset in range(len(unique_windows) + 1):
                if offset < len(unique_windows) and unique_windows[offset]:
                    continue
                segment_windows = offset - segment_start
                if segment_windows:
                    length = segment_windows + NGRAM - 1
                    if length > best_len:
                        best_start = run_start + segment_start
                        best_len = length
                segment_start = offset + 1
    phrase = " ".join(a[best_start : best_start + min(best_len, 90)])
    return {
        "words": best_len,
        "phrase": phrase,
        "agent_words": len(a),
        "baseline_words": len(b),
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def package_members(path: Path) -> dict[str, str]:
    if not path.exists() or not zipfile.is_zipfile(path):
        return {}
    result: dict[str, str] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.endswith("/") or not SEMANTIC_MEMBER_RE.match(name):
                    continue
                try:
                    result[name] = sha256_bytes(archive.read(name))
                except (KeyError, OSError, RuntimeError):
                    pass
    except (OSError, zipfile.BadZipFile):
        return {}
    return result


def normalized_name(name: str) -> str:
    stem = Path(name).stem.lower()
    return re.sub(r"[^a-z0-9]+", "", stem)


def package_overlap(
    agent_files: list[dict[str, Any]],
    baseline_files: list[dict[str, Any]],
    reference_files: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    reference_hashes: set[str] = set()
    for item in reference_files:
        path = item.get("path")
        if path:
            reference_hashes.update(package_members(recorded_path(path)).values())
    reference_stems = {
        normalized_name(item.get("name", ""))
        for item in reference_files
        if len(normalized_name(item.get("name", ""))) >= 8
    }
    matches: list[dict[str, str]] = []
    explained: list[dict[str, str]] = []
    for agent in agent_files:
        a_parts = package_members(recorded_path(agent["path"]))
        for baseline in baseline_files:
            b_parts = package_members(recorded_path(baseline["path"]))
            a_stem = normalized_name(agent["name"])
            b_stem = normalized_name(baseline["name"])
            is_named_template_output = any(
                stem in a_stem or stem in b_stem for stem in reference_stems
            )
            by_hash: dict[str, list[str]] = defaultdict(list)
            for name, digest in b_parts.items():
                by_hash[digest].append(name)
            for a_name, digest in a_parts.items():
                for b_name in by_hash.get(digest, []):
                    item = {
                        "agent_file": agent["name"],
                        "baseline_file": baseline["name"],
                        "agent_member": a_name,
                        "baseline_member": b_name,
                        "sha256": digest,
                    }
                    if digest in reference_hashes or is_named_template_output:
                        explained.append(item)
                    else:
                        matches.append(item)
    return matches, explained


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    findings: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    linkage_rows: list[dict[str, Any]] = []

    for model, config in CONFIGS.items():
        report = config["report"]
        evidence = load_evidence(report)
        crosswalk = json.loads(config["crosswalk"].read_text())
        crosswalk_by_task = {row["agent_task_id"]: row for row in crosswalk}
        for row in crosswalk:
            linkage_rows.append(
                {
                    "model": model,
                    "agent_task_id": row["agent_task_id"],
                    "baseline_task_id": row["baseline_task_id"],
                    "occupation": row["occupation"],
                    "position_within_occupation": row["position_within_occupation"],
                    "prompt_word_jaccard": row["prompt_word_jaccard"],
                    "reference_word_jaccard": row["reference_word_jaccard"],
                    "high_confidence": row["prompt_word_jaccard"] >= HIGH_CONFIDENCE_JACCARD,
                }
            )

        rows = [
            row
            for row in load_jsonl(report / "comparison_manifest.unblinded.jsonl")
            if row["condition"] == "without_rubric" and row["baseline"] == "human"
        ]
        counts = Counter()
        counts["attempts"] = len(rows)
        for row in rows:
            cross = crosswalk_by_task[row["task_id"]]
            high_confidence = cross["prompt_word_jaccard"] >= HIGH_CONFIDENCE_JACCARD
            counts["high_confidence_attempts" if high_confidence else "weak_linkage_attempts"] += 1
            agent_files = row.get("agent_files", [])
            baseline_files = row.get("baseline_files", [])
            reference_files = row.get("reference_files", [])
            ref_hashes = {item.get("sha256") for item in reference_files}

            exact_hashes = sorted(
                ({item.get("sha256") for item in agent_files} & {item.get("sha256") for item in baseline_files})
                - ref_hashes
                - {None}
            )
            exact_names = sorted(
                {normalized_name(item["name"]) for item in agent_files if normalized_name(item["name"])}
                & {normalized_name(item["name"]) for item in baseline_files if normalized_name(item["name"])}
            )
            agent_bundle = evidence.get(file_key(agent_files), {})
            baseline_bundle = evidence.get(file_key(baseline_files), {})
            reference_bundle = evidence.get(file_key(reference_files), {}) if reference_files else {}
            # Archived manifests can point at detached volumes. Evidence audits
            # retain equivalent local object-store paths for content checks.
            local_reference_files = reference_bundle.get("source_files", reference_files)
            part_matches, explained_part_matches = package_overlap(
                agent_files, baseline_files, local_reference_files
            )
            phrase = longest_unexplained_match(
                agent_bundle.get("text", ""),
                baseline_bundle.get("text", ""),
                row.get("task_prompt", "") + "\n" + reference_bundle.get("text", ""),
            )
            if not agent_bundle:
                counts["missing_agent_evidence"] += 1
                if high_confidence:
                    counts["high_confidence_missing_agent_evidence"] += 1
            if not baseline_bundle:
                counts["missing_baseline_evidence"] += 1
                if high_confidence:
                    counts["high_confidence_missing_baseline_evidence"] += 1
            if exact_hashes:
                counts["exact_file_attempts"] += 1
                if high_confidence:
                    counts["high_confidence_exact_file_attempts"] += 1
            if exact_names:
                counts["same_filename_attempts"] += 1
            if part_matches:
                counts["exact_semantic_part_attempts"] += 1
                if high_confidence:
                    counts["high_confidence_exact_semantic_part_attempts"] += 1
            if explained_part_matches:
                counts["explained_template_part_attempts"] += 1
                if high_confidence:
                    counts["high_confidence_explained_template_part_attempts"] += 1
            if phrase["words"] >= PHRASE_FLAG_WORDS:
                counts["long_phrase_attempts"] += 1
                if high_confidence:
                    counts["high_confidence_long_phrase_attempts"] += 1

            if exact_hashes or part_matches or explained_part_matches or phrase["words"] >= PHRASE_FLAG_WORDS:
                findings.append(
                    {
                        "model": model,
                        "task_id": row["task_id"],
                        "baseline_task_id": row["baseline_task_id"],
                        "repeat": row["repeat"],
                        "occupation": row["occupation"],
                        "title": row["title"],
                        "prompt_word_jaccard": cross["prompt_word_jaccard"],
                        "high_confidence": high_confidence,
                        "agent_files": [item["name"] for item in agent_files],
                        "baseline_files": [item["name"] for item in baseline_files],
                        "reference_files": [item.get("name", "") for item in reference_files],
                        "exact_file_hashes": exact_hashes,
                        "same_normalized_filenames": exact_names,
                        "exact_semantic_package_parts": part_matches,
                        "explained_template_package_parts": explained_part_matches,
                        "longest_unexplained_exact_phrase_words": phrase["words"],
                        "longest_unexplained_exact_phrase": phrase["phrase"],
                        "agent_evidence_words": phrase["agent_words"],
                        "baseline_evidence_words": phrase["baseline_words"],
                        "agent_evidence_audit": agent_bundle.get("audit_path"),
                        "baseline_evidence_audit": baseline_bundle.get("audit_path"),
                    }
                )
        summaries[model] = dict(counts)

    findings.sort(
        key=lambda item: (
            not item["high_confidence"],
            -len(item["exact_file_hashes"]),
            -len(item["exact_semantic_package_parts"]),
            -item["longest_unexplained_exact_phrase_words"],
            item["model"],
            item["task_id"],
            item["repeat"],
        )
    )
    (OUT / "findings.json").write_text(json.dumps(findings, indent=2) + "\n")
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "method": {
                    "condition": "without_rubric",
                    "baseline": "human",
                    "prompt_alignment_threshold": HIGH_CONFIDENCE_JACCARD,
                    "exact_phrase_threshold_words": PHRASE_FLAG_WORDS,
                    "exact_phrase_ngram_anchor_words": NGRAM,
                },
                "models": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    with (OUT / "crosswalk_quality.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(linkage_rows[0]))
        writer.writeheader()
        writer.writerows(linkage_rows)

    print(OUT / "summary.json")


if __name__ == "__main__":
    main()
