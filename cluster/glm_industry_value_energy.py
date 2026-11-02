#!/usr/bin/env python3
"""Aggregate the final GLM GDPVal results by source industry.

Industry assignment comes from the frozen GDPVal pairwise manifest. Eight labels
are exact 2022 NAICS sector titles. The source label "Government" is retained
and displayed as "Government / Public Administration" with an explicit caveat
that it is broader than a strict NAICS 92 establishment classification.

The value-energy methodology matches the Qwen/GPT-OSS follow-on analyses:

* final Gemini win or tie against the expert-only reference = successful attempt;
* threshold k qualifies a distinct prompt with at least k successes out of 4;
* qualifying output dollar value is counted once;
* prompt time saving is recorded expert completion time minus mean AI-system
  completion time among its
  successful attempts, without clamping negative values;
* industry energy contains all four attempts for every prompt, including
  failed, incomplete, or timed-out attempts;
* authoritative batch energy is allocated equally across its 20 attempts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any


getcontext().prec = 50

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "outputs/019f86a0-e5e6-7e90-8578-b08ddb4385fb"
PAIRWISE_DIR = ROOT / "reports/gdpval_pairwise_glm52_20260715"

QUERY_LEDGER = SOURCE_DIR / "glm52_no_rubric_query_ledger.csv"
BATCH_LEDGER = SOURCE_DIR / "glm52_gdpval_batch_ledger.csv"
RESULTS = PAIRWISE_DIR / "results.output_level.jsonl"
MANIFEST = PAIRWISE_DIR / "comparison_manifest.unblinded.jsonl"
VALUATION_WORKBOOK = SOURCE_DIR / "gdpval_snorkel_task_wage_values.xlsx"

NAICS_SOURCES = {
    "census_naics": (
        "https://www.census.gov/programs-surveys/economic-census/"
        "year/2022/guidance/understanding-naics.html"
    ),
    "bls_oews": "https://www.bls.gov/oes/oes_emp.htm",
    "bls_matrix": (
        "https://www.bls.gov/emp/tables/industry-occupation-matrix-occupation.htm"
    ),
}

INDUSTRIES = {
    "Finance and Insurance": {
        "display": "Finance and Insurance",
        "naics_code": "52",
        "mapping_status": "Exact 2022 NAICS sector title",
    },
    "Government": {
        "display": "Government / Public Administration",
        "naics_code": "92*",
        "mapping_status": (
            "GDPVal owner-sector label; broader than strict NAICS 92"
        ),
    },
    "Health Care and Social Assistance": {
        "display": "Health Care and Social Assistance",
        "naics_code": "62",
        "mapping_status": "Exact 2022 NAICS sector title",
    },
    "Information": {
        "display": "Information",
        "naics_code": "51",
        "mapping_status": "Exact 2022 NAICS sector title",
    },
    "Manufacturing": {
        "display": "Manufacturing",
        "naics_code": "31-33",
        "mapping_status": "Exact 2022 NAICS sector title",
    },
    "Professional, Scientific, and Technical Services": {
        "display": "Professional, Scientific, and Technical Services",
        "naics_code": "54",
        "mapping_status": "Exact 2022 NAICS sector title",
    },
    "Real Estate and Rental and Leasing": {
        "display": "Real Estate and Rental and Leasing",
        "naics_code": "53",
        "mapping_status": "Exact 2022 NAICS sector title",
    },
    "Retail Trade": {
        "display": "Retail Trade",
        "naics_code": "44-45",
        "mapping_status": "Exact 2022 NAICS sector title",
    },
    "Wholesale Trade": {
        "display": "Wholesale Trade",
        "naics_code": "42",
        "mapping_status": "Exact 2022 NAICS sector title",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    # Iterate physical LF-delimited records. str.splitlines() also treats
    # Unicode NEL characters inside task text as record separators.
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rank_rows(rows: list[dict[str, Any]]) -> None:
    by_value = sorted(
        rows, key=lambda row: (-row["value_per_mj"], row["industry"])
    )
    by_hours = sorted(
        rows, key=lambda row: (-row["hours_saved_per_mj"], row["industry"])
    )
    for rank, row in enumerate(by_value, 1):
        row["value_rank"] = rank
    for rank, row in enumerate(by_hours, 1):
        row["hours_saved_rank"] = rank
    rows.sort(key=lambda row: row["value_rank"])


def calculate() -> dict[str, Any]:
    manifest_rows = [
        row
        for row in read_jsonl(MANIFEST)
        if row["condition"] == "without_rubric" and row["baseline"] == "human"
    ]
    if len(manifest_rows) != 880:
        raise AssertionError(f"expected 880 manifest rows, got {len(manifest_rows)}")

    manifest_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    task_metadata: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        key = (str(row["task_id"]), int(row["repeat"]))
        if key in manifest_by_key:
            raise AssertionError(f"duplicate manifest key {key}")
        manifest_by_key[key] = row
        task_id = str(row["task_id"])
        metadata = {
            "task_id": task_id,
            "baseline_task_id": str(row["baseline_task_id"]),
            "occupation": str(row["occupation"]),
            "source_sector": str(row["sector"]),
        }
        prior = task_metadata.setdefault(task_id, metadata)
        if prior != metadata:
            raise AssertionError(f"inconsistent task metadata for {task_id}")
    if len(task_metadata) != 220:
        raise AssertionError(f"expected 220 tasks, got {len(task_metadata)}")

    results_rows = [
        row
        for row in read_jsonl(RESULTS)
        if row["condition"] == "without_rubric" and row["baseline"] == "human"
    ]
    if len(results_rows) != 880:
        raise AssertionError(f"expected 880 results, got {len(results_rows)}")
    results_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in results_rows:
        key = (str(row["task_id"]), int(row["repeat"]))
        if key in results_by_key:
            raise AssertionError(f"duplicate result key {key}")
        if row["agent_outcome"] not in {"win", "tie", "loss"}:
            raise AssertionError(f"unexpected outcome {row['agent_outcome']}")
        results_by_key[key] = row

    query_rows = [
        row for row in read_csv(QUERY_LEDGER) if row["dataset"] == "GDPVal"
    ]
    if len(query_rows) != 880:
        raise AssertionError(f"expected 880 GDPVal queries, got {len(query_rows)}")
    query_by_key: dict[tuple[str, int], dict[str, str]] = {}
    for row in query_rows:
        key = (str(row["judge_task_id"]), int(row["repeat_index"]))
        if key in query_by_key:
            raise AssertionError(f"duplicate query key {key}")
        query_by_key[key] = row

    batch_rows = read_csv(BATCH_LEDGER)
    if len(batch_rows) != 44:
        raise AssertionError(f"expected 44 batches, got {len(batch_rows)}")
    batches_by_key = {row["batch_key"]: row for row in batch_rows}
    if len(batches_by_key) != 44:
        raise AssertionError("duplicate batch keys")

    attempts: list[dict[str, Any]] = []
    for key, result in results_by_key.items():
        query = query_by_key.get(key)
        manifest = manifest_by_key.get(key)
        if query is None or manifest is None:
            raise AssertionError(f"missing joined record for {key}")
        task_id, repeat = key
        source_sector = str(manifest["sector"])
        if source_sector not in INDUSTRIES:
            raise AssertionError(f"unmapped source sector {source_sector}")
        industry = INDUSTRIES[source_sector]
        latency_hours = float(query["agent_latency_hours"])
        latency_seconds = float(query["query_latency_s"])
        if not math.isclose(
            latency_hours, latency_seconds / 3600, rel_tol=1e-10, abs_tol=1e-10
        ):
            raise AssertionError(f"latency mismatch for {key}")
        energy_j = float(query["allocated_gpu_energy_joules"])
        if not math.isfinite(energy_j) or energy_j <= 0:
            raise AssertionError(f"invalid energy for {key}")
        human_hours = float(query["human_only_hours"])
        hourly_wage = float(query["hourly_wage_usd"])
        human_value = float(query["human_only_value_usd"])
        if not math.isclose(
            human_value, human_hours * hourly_wage, rel_tol=1e-10, abs_tol=1e-8
        ):
            raise AssertionError(f"valuation mismatch for {key}")
        if str(query["occupation"]) != str(manifest["occupation"]):
            raise AssertionError(f"occupation mismatch for {key}")
        outcome = str(result["agent_outcome"])
        attempts.append(
            {
                "task_id": task_id,
                "baseline_task_id": str(query["prompt_id"]),
                "repeat": repeat,
                "occupation": str(manifest["occupation"]),
                "source_sector": source_sector,
                "industry": industry["display"],
                "naics_code": industry["naics_code"],
                "mapping_status": industry["mapping_status"],
                "outcome": outcome,
                "success": outcome in {"win", "tie"},
                "latency_seconds": latency_seconds,
                "latency_hours": latency_hours,
                "batch_key": str(query["batch_key"]),
                "query_id": str(query["query_id"]),
                "allocated_energy_j": energy_j,
                "allocated_energy_mj": energy_j / 1_000_000,
                "hourly_wage": hourly_wage,
                "human_only_hours": human_hours,
                "human_only_value_usd": human_value,
                "completed": query["completed"] == "True",
                "timed_out": query["timed_out"] == "True",
            }
        )
    if len(attempts) != 880:
        raise AssertionError("joined attempt count mismatch")

    attempt_counts = Counter(row["task_id"] for row in attempts)
    if len(attempt_counts) != 220 or set(attempt_counts.values()) != {4}:
        raise AssertionError("tasks do not have exactly four attempts")
    repeats_by_task: dict[str, set[int]] = defaultdict(set)
    for row in attempts:
        repeats_by_task[row["task_id"]].add(row["repeat"])
    if any(repeats != {0, 1, 2, 3} for repeats in repeats_by_task.values()):
        raise AssertionError("repeat coverage is not exactly 0-3")

    occupations_to_industries: dict[str, set[str]] = defaultdict(set)
    for row in attempts:
        occupations_to_industries[row["occupation"]].add(row["source_sector"])
    if len(occupations_to_industries) != 44:
        raise AssertionError("expected 44 occupations")
    if any(len(sectors) != 1 for sectors in occupations_to_industries.values()):
        raise AssertionError("an occupation maps to multiple source sectors")

    attempts_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        attempts_by_task[row["task_id"]].append(row)
    prompts: list[dict[str, Any]] = []
    for task_id, task_attempts in attempts_by_task.items():
        first = task_attempts[0]
        for row in task_attempts[1:]:
            for field in (
                "baseline_task_id",
                "occupation",
                "source_sector",
                "industry",
                "naics_code",
                "hourly_wage",
                "human_only_hours",
                "human_only_value_usd",
            ):
                if row[field] != first[field]:
                    raise AssertionError(f"inconsistent {field} for {task_id}")
        successes = [row for row in task_attempts if row["success"]]
        prompts.append(
            {
                "task_id": task_id,
                "baseline_task_id": first["baseline_task_id"],
                "occupation": first["occupation"],
                "source_sector": first["source_sector"],
                "industry": first["industry"],
                "naics_code": first["naics_code"],
                "hourly_wage": first["hourly_wage"],
                "human_only_hours": first["human_only_hours"],
                "human_only_value_usd": first["human_only_value_usd"],
                "success_count": len(successes),
                "mean_success_latency_hours": (
                    sum(row["latency_hours"] for row in successes) / len(successes)
                    if successes
                    else None
                ),
                "hours_saved_if_qualified": (
                    first["human_only_hours"]
                    - sum(row["latency_hours"] for row in successes) / len(successes)
                    if successes
                    else None
                ),
            }
        )
    if len(prompts) != 220:
        raise AssertionError("expected 220 prompts")

    industry_names = [value["display"] for value in INDUSTRIES.values()]
    thresholds: dict[str, Any] = {}
    for threshold in range(1, 5):
        rows: list[dict[str, Any]] = []
        for industry in industry_names:
            industry_prompts = [
                row for row in prompts if row["industry"] == industry
            ]
            industry_attempts = [
                row for row in attempts if row["industry"] == industry
            ]
            qualifying = [
                row
                for row in industry_prompts
                if row["success_count"] >= threshold
            ]
            energy_mj = sum(row["allocated_energy_mj"] for row in industry_attempts)
            generated_value = sum(
                row["human_only_value_usd"] for row in qualifying
            )
            hours_saved = sum(
                float(row["hours_saved_if_qualified"]) for row in qualifying
            )
            successful_attempts = sum(row["success"] for row in industry_attempts)
            source_sector = industry_prompts[0]["source_sector"]
            rows.append(
                {
                    "industry": industry,
                    "source_sector": source_sector,
                    "naics_code": INDUSTRIES[source_sector]["naics_code"],
                    "threshold": threshold,
                    "qualifying_prompts": len(qualifying),
                    "total_prompts": len(industry_prompts),
                    "successful_attempts": successful_attempts,
                    "total_attempts": len(industry_attempts),
                    "attempt_win_or_tie_rate": successful_attempts
                    / len(industry_attempts),
                    "energy_mj_all_four_runs": energy_mj,
                    "generated_human_only_value_usd": generated_value,
                    "value_per_mj": generated_value / energy_mj,
                    "total_human_only_hours_saved": hours_saved,
                    "hours_saved_per_mj": hours_saved / energy_mj,
                }
            )
        rank_rows(rows)
        qualifying_all = [
            row for row in prompts if row["success_count"] >= threshold
        ]
        total_energy_mj = sum(row["allocated_energy_mj"] for row in attempts)
        total_value = sum(row["human_only_value_usd"] for row in qualifying_all)
        total_hours = sum(
            float(row["hours_saved_if_qualified"]) for row in qualifying_all
        )
        successful_attempts = sum(row["success"] for row in attempts)
        thresholds[str(threshold)] = {
            "industry_rows": rows,
            "overall": {
                "threshold": threshold,
                "qualifying_prompts": len(qualifying_all),
                "total_prompts": 220,
                "successful_attempts": successful_attempts,
                "total_attempts": 880,
                "attempt_win_or_tie_rate": successful_attempts / 880,
                "energy_mj_all_four_runs": total_energy_mj,
                "generated_human_only_value_usd": total_value,
                "value_per_mj": total_value / total_energy_mj,
                "total_human_only_hours_saved": total_hours,
                "hours_saved_per_mj": total_hours / total_energy_mj,
            },
        }

    batch_qa: list[dict[str, Any]] = []
    allocated_total = Decimal("0")
    source_total = Decimal("0")
    query_by_batch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        query_by_batch[row["batch_key"]].append(row)
    for batch_key, batch in batches_by_key.items():
        source_energy = Decimal(batch["gpu_energy_joules"])
        source_total += source_energy
        batch_attempts = query_by_batch[batch_key]
        if len(batch_attempts) != int(batch["attempts"]) or len(batch_attempts) != 20:
            raise AssertionError(f"batch attempt mismatch for {batch_key}")
        allocated = sum(
            (Decimal(str(row["allocated_energy_j"])) for row in batch_attempts),
            Decimal("0"),
        )
        allocated_total += allocated
        batch_qa.append(
            {
                "batch_key": batch_key,
                "occupation": batch["occupation"],
                "attempts": len(batch_attempts),
                "gpu_count": int(batch["gpu_device_count"]),
                "source_energy_j": float(source_energy),
                "allocated_energy_j": float(allocated),
                "reconciliation_error_j": float(allocated - source_energy),
            }
        )
    reconciliation_error = allocated_total - source_total

    final_outcomes = Counter(row["outcome"] for row in attempts)
    old_ledger_matches = sum(
        query_by_key[(row["task_id"], row["repeat"])]["outcome"] == row["outcome"]
        for row in attempts
    )
    industry_prompt_counts = Counter(row["industry"] for row in prompts)
    qa = {
        "prompt_count": len(prompts),
        "attempt_count": len(attempts),
        "attempts_per_prompt_values": sorted(set(attempt_counts.values())),
        "repeat_sets_all_0_to_3": all(
            repeats == {0, 1, 2, 3} for repeats in repeats_by_task.values()
        ),
        "occupation_count": len(occupations_to_industries),
        "occupations_with_multiple_industries": sum(
            len(sectors) > 1 for sectors in occupations_to_industries.values()
        ),
        "industry_count": len(industry_prompt_counts),
        "industry_prompt_counts": dict(sorted(industry_prompt_counts.items())),
        "batch_count": len(batch_rows),
        "batch_attempt_count_values": sorted(
            {len(value) for value in query_by_batch.values()}
        ),
        "gpu_count_values": sorted(
            {int(row["gpu_device_count"]) for row in batch_rows}
        ),
        "source_energy_j": float(source_total),
        "allocated_attempt_energy_j": float(allocated_total),
        "energy_reconciliation_error_j": float(reconciliation_error),
        "max_batch_reconciliation_error_j": max(
            abs(row["reconciliation_error_j"]) for row in batch_qa
        ),
        "missing_wage_values": sum(
            row["hourly_wage"] is None for row in prompts
        ),
        "missing_human_time_values": sum(
            row["human_only_hours"] is None for row in prompts
        ),
        "missing_outcome_values": sum(not row["outcome"] for row in attempts),
        "missing_latency_values": sum(
            row["latency_hours"] is None for row in attempts
        ),
        "missing_energy_values": sum(
            row["allocated_energy_j"] is None for row in attempts
        ),
        "final_outcomes": dict(sorted(final_outcomes.items())),
        "successful_attempt_count": sum(row["success"] for row in attempts),
        "completed_source_attempt_count": sum(
            row["completed"] for row in attempts
        ),
        "timed_out_source_attempt_count": sum(
            row["timed_out"] for row in attempts
        ),
        "old_query_ledger_outcomes_matching_final_gemini": old_ledger_matches,
        "old_query_ledger_outcome_match_rate": old_ledger_matches / 880,
        "success_source": "Final results.output_level.jsonl; not old query-ledger outcome",
        "passed": all(
            [
                len(prompts) == 220,
                len(attempts) == 880,
                set(attempt_counts.values()) == {4},
                all(
                    repeats == {0, 1, 2, 3}
                    for repeats in repeats_by_task.values()
                ),
                len(occupations_to_industries) == 44,
                all(
                    len(sectors) == 1
                    for sectors in occupations_to_industries.values()
                ),
                len(industry_prompt_counts) == 9,
                len(batch_rows) == 44,
                set(len(value) for value in query_by_batch.values()) == {20},
                set(int(row["gpu_device_count"]) for row in batch_rows) == {8},
                abs(float(reconciliation_error)) < 1e-6,
            ]
        ),
    }
    if not qa["passed"]:
        raise AssertionError(f"QA failed: {qa}")

    occupation_mapping = []
    for occupation, sectors in sorted(occupations_to_industries.items()):
        source_sector = next(iter(sectors))
        tasks = {
            row["task_id"]
            for row in attempts
            if row["occupation"] == occupation
        }
        occupation_mapping.append(
            {
                "occupation": occupation,
                "source_sector": source_sector,
                "industry": INDUSTRIES[source_sector]["display"],
                "naics_code": INDUSTRIES[source_sector]["naics_code"],
                "mapping_status": INDUSTRIES[source_sector]["mapping_status"],
                "prompt_count": len(tasks),
            }
        )

    return {
        "methodology": {
            "study": "GLM-5.2 GDPVal industry value-energy analysis",
            "primary_baseline": "human",
            "judge_model": "models/gemini-3.1-flash-lite",
            "success_definition": "Final Gemini agent_outcome in {win, tie}",
            "thresholds": [1, 2, 3, 4],
            "industry_rule": (
                "Use frozen GDPVal source sector labels. Eight are exact 2022 "
                "NAICS sector titles; Government is retained as a broader source "
                "owner-sector label and displayed with a NAICS 92 caveat."
            ),
            "energy_rule": (
                "All four attempts per prompt, including failed/incomplete runs; "
                "source batch energy sums all eight GPUs and is allocated equally "
                "across the batch's 20 attempts."
            ),
            "time_saved_rule": (
                "Recorded expert completion time minus mean AI-system completion time across successful "
                "attempts; negative values are not clamped."
            ),
            "source_urls": NAICS_SOURCES,
        },
        "thresholds": thresholds,
        "occupation_mapping": occupation_mapping,
        "qa": qa,
        "batch_qa": batch_qa,
        "prompt_ledger": sorted(
            prompts, key=lambda row: (row["industry"], row["occupation"], row["task_id"])
        ),
        "attempt_ledger": sorted(
            attempts,
            key=lambda row: (
                row["industry"],
                row["occupation"],
                row["task_id"],
                row["repeat"],
            ),
        ),
        "source_hashes": {
            "query_ledger_sha256": sha256(QUERY_LEDGER),
            "batch_ledger_sha256": sha256(BATCH_LEDGER),
            "results_sha256": sha256(RESULTS),
            "manifest_sha256": sha256(MANIFEST),
            "valuation_workbook_sha256": sha256(VALUATION_WORKBOOK),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = calculate()
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
