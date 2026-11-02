#!/usr/bin/env python3
"""GLM GDPVal value-energy experiment using the expert-with-AI reference.

This is intentionally separate from the expert-only analysis:

* final Gemini GLM-vs-expert-with-AI win or tie = successful attempt;
* output dollar value = recorded expert-with-AI completion time × hourly wage;
* a prompt qualifies at threshold k with at least k successes among four runs;
* qualifying prompt value is counted once;
* time saving = recorded expert-with-AI completion time minus mean AI-system
  completion time across successful attempts;
* three paired time-saving views are also reported:
  expert-only to GLM, expert-only to expert-with-AI, and expert-with-AI to GLM;
* negatives are not clamped;
* category energy includes all four GLM attempts, including failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any


getcontext().prec = 50

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from glm_industry_value_energy import (  # noqa: E402
    BATCH_LEDGER,
    QUERY_LEDGER,
    RESULTS,
    VALUATION_WORKBOOK,
    read_csv,
    read_jsonl,
)


CATEGORIES = {
    "11": "Management",
    "13": "Business & Financial Operations",
    "15": "Computer & Mathematical",
    "17": "Architecture & Engineering",
    "21": "Community & Social Service",
    "23": "Legal",
    "27": "Arts, Design, Entertainment & Media",
    "29": "Healthcare Practitioners & Technical",
    "33": "Protective Service",
    "39": "Personal Care & Service",
    "41": "Sales & Related",
    "43": "Office & Administrative Support",
    "51": "Production",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_rows(rows: list[dict[str, Any]]) -> None:
    by_value = sorted(
        rows, key=lambda row: (-row["value_per_mj"], row["category"])
    )
    by_hours = sorted(
        rows, key=lambda row: (-row["hours_saved_per_mj"], row["category"])
    )
    for rank, row in enumerate(by_value, 1):
        row["value_rank"] = rank
    for rank, row in enumerate(by_hours, 1):
        row["hours_saved_rank"] = rank
    rows.sort(key=lambda row: row["value_rank"])


def calculate() -> dict[str, Any]:
    results = [
        row
        for row in read_jsonl(RESULTS)
        if row["condition"] == "without_rubric" and row["baseline"] == "human_ai"
    ]
    if len(results) != 880:
        raise AssertionError(f"expected 880 final expert-with-AI results, got {len(results)}")
    results_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in results:
        key = (str(row["task_id"]), int(row["repeat"]))
        if key in results_by_key:
            raise AssertionError(f"duplicate result {key}")
        if row["agent_outcome"] not in {"win", "tie", "loss"}:
            raise AssertionError(f"unexpected outcome {row['agent_outcome']}")
        results_by_key[key] = row

    queries = [
        row for row in read_csv(QUERY_LEDGER) if row["dataset"] == "GDPVal"
    ]
    if len(queries) != 880:
        raise AssertionError(f"expected 880 GDPVal query rows, got {len(queries)}")
    queries_by_key: dict[tuple[str, int], dict[str, str]] = {}
    for row in queries:
        key = (str(row["judge_task_id"]), int(row["repeat_index"]))
        if key in queries_by_key:
            raise AssertionError(f"duplicate query {key}")
        queries_by_key[key] = row

    batch_rows = read_csv(BATCH_LEDGER)
    if len(batch_rows) != 44:
        raise AssertionError(f"expected 44 batches, got {len(batch_rows)}")
    batch_by_key = {row["batch_key"]: row for row in batch_rows}
    if len(batch_by_key) != 44:
        raise AssertionError("duplicate batch keys")

    attempts: list[dict[str, Any]] = []
    for key, result in results_by_key.items():
        query = queries_by_key.get(key)
        if query is None:
            raise AssertionError(f"missing query row for {key}")
        onet_code = str(query["onet_code"])
        major_code = onet_code[:2]
        if major_code not in CATEGORIES:
            raise AssertionError(f"unmapped O*NET code {onet_code}")
        latency_seconds = float(query["query_latency_s"])
        latency_hours = float(query["agent_latency_hours"])
        if not math.isclose(
            latency_hours, latency_seconds / 3600, rel_tol=1e-10, abs_tol=1e-10
        ):
            raise AssertionError(f"latency mismatch for {key}")
        hourly_wage = float(query["hourly_wage_usd"])
        human_only_hours = float(query["human_only_hours"])
        human_ai_hours = float(query["human_ai_hours"])
        human_ai_value = float(query["human_ai_value_usd"])
        if not math.isclose(
            human_ai_value,
            hourly_wage * human_ai_hours,
            rel_tol=1e-10,
            abs_tol=1e-8,
        ):
            raise AssertionError(f"expert-with-AI value mismatch for {key}")
        energy_j = float(query["allocated_gpu_energy_joules"])
        if not math.isfinite(energy_j) or energy_j <= 0:
            raise AssertionError(f"invalid energy for {key}")
        outcome = str(result["agent_outcome"])
        attempts.append(
            {
                "task_id": str(result["task_id"]),
                "baseline_task_id": str(query["prompt_id"]),
                "repeat": int(result["repeat"]),
                "occupation": str(query["occupation"]),
                "onet_code": onet_code,
                "category": CATEGORIES[major_code],
                "outcome": outcome,
                "success": outcome in {"win", "tie"},
                "latency_seconds": latency_seconds,
                "latency_hours": latency_hours,
                "batch_key": str(query["batch_key"]),
                "query_id": str(query["query_id"]),
                "allocated_energy_j": energy_j,
                "allocated_energy_mj": energy_j / 1_000_000,
                "hourly_wage": hourly_wage,
                "human_only_hours": human_only_hours,
                "human_ai_hours": human_ai_hours,
                "human_ai_value_usd": human_ai_value,
                "completed": query["completed"] == "True",
                "timed_out": query["timed_out"] == "True",
            }
        )
    if len(attempts) != 880:
        raise AssertionError("attempt count mismatch")

    attempt_counts = Counter(row["task_id"] for row in attempts)
    if len(attempt_counts) != 220 or set(attempt_counts.values()) != {4}:
        raise AssertionError("not exactly four attempts per prompt")
    repeats_by_task: dict[str, set[int]] = defaultdict(set)
    for row in attempts:
        repeats_by_task[row["task_id"]].add(row["repeat"])
    if any(repeats != {0, 1, 2, 3} for repeats in repeats_by_task.values()):
        raise AssertionError("repeat coverage is not exactly 0-3")

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
                "onet_code",
                "category",
                "hourly_wage",
                "human_only_hours",
                "human_ai_hours",
                "human_ai_value_usd",
            ):
                if row[field] != first[field]:
                    raise AssertionError(f"inconsistent {field} for {task_id}")
        successes = [row for row in task_attempts if row["success"]]
        mean_success_latency = (
            sum(row["latency_hours"] for row in successes) / len(successes)
            if successes
            else None
        )
        prompts.append(
            {
                "task_id": task_id,
                "baseline_task_id": first["baseline_task_id"],
                "occupation": first["occupation"],
                "onet_code": first["onet_code"],
                "category": first["category"],
                "hourly_wage": first["hourly_wage"],
                "human_only_hours": first["human_only_hours"],
                "human_ai_hours": first["human_ai_hours"],
                "human_ai_value_usd": first["human_ai_value_usd"],
                "success_count": len(successes),
                "mean_success_latency_hours": mean_success_latency,
                "hours_saved_if_qualified": (
                    first["human_ai_hours"] - mean_success_latency
                    if mean_success_latency is not None
                    else None
                ),
                "human_to_ai_hours_saved_if_qualified": (
                    first["human_only_hours"] - mean_success_latency
                    if mean_success_latency is not None
                    else None
                ),
                "human_to_human_ai_hours_saved": (
                    first["human_only_hours"] - first["human_ai_hours"]
                ),
                "human_ai_to_ai_hours_saved_if_qualified": (
                    first["human_ai_hours"] - mean_success_latency
                    if mean_success_latency is not None
                    else None
                ),
            }
        )
    if len(prompts) != 220:
        raise AssertionError("prompt count mismatch")

    categories = list(CATEGORIES.values())
    if {row["category"] for row in prompts} != set(categories):
        raise AssertionError("13-category coverage mismatch")

    thresholds: dict[str, Any] = {}
    for threshold in range(1, 5):
        category_rows: list[dict[str, Any]] = []
        for category in categories:
            category_prompts = [
                row for row in prompts if row["category"] == category
            ]
            category_attempts = [
                row for row in attempts if row["category"] == category
            ]
            qualifying = [
                row
                for row in category_prompts
                if row["success_count"] >= threshold
            ]
            energy_mj = sum(
                row["allocated_energy_mj"] for row in category_attempts
            )
            generated_value = sum(
                row["human_ai_value_usd"] for row in qualifying
            )
            hours_saved = sum(
                float(row["hours_saved_if_qualified"]) for row in qualifying
            )
            human_hours = sum(
                float(row["human_only_hours"]) for row in qualifying
            )
            human_ai_hours = sum(
                float(row["human_ai_hours"]) for row in qualifying
            )
            ai_hours = sum(
                float(row["mean_success_latency_hours"]) for row in qualifying
            )
            human_to_ai_hours_saved = human_hours - ai_hours
            human_to_human_ai_hours_saved = human_hours - human_ai_hours
            human_ai_to_ai_hours_saved = human_ai_hours - ai_hours
            successful_attempts = sum(row["success"] for row in category_attempts)
            category_rows.append(
                {
                    "category": category,
                    "threshold": threshold,
                    "qualifying_prompts": len(qualifying),
                    "total_prompts": len(category_prompts),
                    "successful_attempts": successful_attempts,
                    "total_attempts": len(category_attempts),
                    "attempt_win_or_tie_rate": successful_attempts
                    / len(category_attempts),
                    "energy_mj_all_four_runs": energy_mj,
                    "generated_human_ai_value_usd": generated_value,
                    "value_per_mj": generated_value / energy_mj,
                    "total_human_ai_hours_saved": hours_saved,
                    "hours_saved_per_mj": hours_saved / energy_mj,
                    "time_savings": {
                        "human_hours": human_hours,
                        "human_ai_hours": human_ai_hours,
                        "ai_hours": ai_hours,
                        "human_to_ai": {
                            "hours_saved": human_to_ai_hours_saved,
                            "percent_saved": (
                                human_to_ai_hours_saved / human_hours
                                if human_hours
                                else None
                            ),
                            "hours_saved_per_mj": (
                                human_to_ai_hours_saved / energy_mj
                            ),
                        },
                        "human_to_human_ai": {
                            "hours_saved": human_to_human_ai_hours_saved,
                            "percent_saved": (
                                human_to_human_ai_hours_saved / human_hours
                                if human_hours
                                else None
                            ),
                            "hours_saved_per_mj": (
                                human_to_human_ai_hours_saved / energy_mj
                            ),
                        },
                        "human_ai_to_ai": {
                            "hours_saved": human_ai_to_ai_hours_saved,
                            "percent_saved": (
                                human_ai_to_ai_hours_saved / human_ai_hours
                                if human_ai_hours
                                else None
                            ),
                            "hours_saved_per_mj": (
                                human_ai_to_ai_hours_saved / energy_mj
                            ),
                        },
                    },
                }
            )
        rank_rows(category_rows)
        qualifying_all = [
            row for row in prompts if row["success_count"] >= threshold
        ]
        total_energy_mj = sum(row["allocated_energy_mj"] for row in attempts)
        total_value = sum(row["human_ai_value_usd"] for row in qualifying_all)
        total_hours = sum(
            float(row["hours_saved_if_qualified"]) for row in qualifying_all
        )
        total_human_hours = sum(
            float(row["human_only_hours"]) for row in qualifying_all
        )
        total_human_ai_hours = sum(
            float(row["human_ai_hours"]) for row in qualifying_all
        )
        total_ai_hours = sum(
            float(row["mean_success_latency_hours"]) for row in qualifying_all
        )
        total_human_to_ai_hours_saved = total_human_hours - total_ai_hours
        total_human_to_human_ai_hours_saved = (
            total_human_hours - total_human_ai_hours
        )
        total_human_ai_to_ai_hours_saved = (
            total_human_ai_hours - total_ai_hours
        )
        successful_attempts = sum(row["success"] for row in attempts)
        thresholds[str(threshold)] = {
            "category_rows": category_rows,
            "overall": {
                "threshold": threshold,
                "qualifying_prompts": len(qualifying_all),
                "total_prompts": 220,
                "successful_attempts": successful_attempts,
                "total_attempts": 880,
                "attempt_win_or_tie_rate": successful_attempts / 880,
                "energy_mj_all_four_runs": total_energy_mj,
                "generated_human_ai_value_usd": total_value,
                "value_per_mj": total_value / total_energy_mj,
                "total_human_ai_hours_saved": total_hours,
                "hours_saved_per_mj": total_hours / total_energy_mj,
                "time_savings": {
                    "human_hours": total_human_hours,
                    "human_ai_hours": total_human_ai_hours,
                    "ai_hours": total_ai_hours,
                    "human_to_ai": {
                        "hours_saved": total_human_to_ai_hours_saved,
                        "percent_saved": (
                            total_human_to_ai_hours_saved / total_human_hours
                            if total_human_hours
                            else None
                        ),
                        "hours_saved_per_mj": (
                            total_human_to_ai_hours_saved / total_energy_mj
                        ),
                    },
                    "human_to_human_ai": {
                        "hours_saved": total_human_to_human_ai_hours_saved,
                        "percent_saved": (
                            total_human_to_human_ai_hours_saved
                            / total_human_hours
                            if total_human_hours
                            else None
                        ),
                        "hours_saved_per_mj": (
                            total_human_to_human_ai_hours_saved
                            / total_energy_mj
                        ),
                    },
                    "human_ai_to_ai": {
                        "hours_saved": total_human_ai_to_ai_hours_saved,
                        "percent_saved": (
                            total_human_ai_to_ai_hours_saved
                            / total_human_ai_hours
                            if total_human_ai_hours
                            else None
                        ),
                        "hours_saved_per_mj": (
                            total_human_ai_to_ai_hours_saved / total_energy_mj
                        ),
                    },
                },
            },
        }

    queries_by_batch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        queries_by_batch[row["batch_key"]].append(row)
    source_energy = Decimal("0")
    allocated_energy = Decimal("0")
    batch_qa: list[dict[str, Any]] = []
    for batch_key, batch in batch_by_key.items():
        batch_source = Decimal(batch["gpu_energy_joules"])
        batch_allocated = sum(
            (
                Decimal(str(row["allocated_energy_j"]))
                for row in queries_by_batch[batch_key]
            ),
            Decimal("0"),
        )
        source_energy += batch_source
        allocated_energy += batch_allocated
        if len(queries_by_batch[batch_key]) != int(batch["attempts"]):
            raise AssertionError(f"batch size mismatch for {batch_key}")
        batch_qa.append(
            {
                "batch_key": batch_key,
                "attempts": len(queries_by_batch[batch_key]),
                "gpu_count": int(batch["gpu_device_count"]),
                "source_energy_j": float(batch_source),
                "allocated_energy_j": float(batch_allocated),
                "reconciliation_error_j": float(
                    batch_allocated - batch_source
                ),
            }
        )
    reconciliation_error = allocated_energy - source_energy
    outcomes = Counter(row["outcome"] for row in attempts)
    prompt_counts = Counter(row["category"] for row in prompts)

    qa = {
        "prompt_count": len(prompts),
        "attempt_count": len(attempts),
        "attempts_per_prompt_values": sorted(set(attempt_counts.values())),
        "repeat_sets_all_0_to_3": all(
            repeats == {0, 1, 2, 3} for repeats in repeats_by_task.values()
        ),
        "category_count": len(prompt_counts),
        "category_prompt_counts": dict(sorted(prompt_counts.items())),
        "batch_count": len(batch_rows),
        "batch_attempt_count_values": sorted(
            {len(rows) for rows in queries_by_batch.values()}
        ),
        "gpu_count_values": sorted(
            {int(row["gpu_device_count"]) for row in batch_rows}
        ),
        "source_energy_j": float(source_energy),
        "allocated_attempt_energy_j": float(allocated_energy),
        "energy_reconciliation_error_j": float(reconciliation_error),
        "max_batch_reconciliation_error_j": max(
            abs(row["reconciliation_error_j"]) for row in batch_qa
        ),
        "missing_wage_values": sum(
            row["hourly_wage"] is None for row in prompts
        ),
        "missing_human_ai_time_values": sum(
            row["human_ai_hours"] is None for row in prompts
        ),
        "missing_human_only_time_values": sum(
            row["human_only_hours"] is None for row in prompts
        ),
        "missing_human_ai_value_values": sum(
            row["human_ai_value_usd"] is None for row in prompts
        ),
        "missing_outcome_values": sum(not row["outcome"] for row in attempts),
        "missing_latency_values": sum(
            row["latency_hours"] is None for row in attempts
        ),
        "missing_energy_values": sum(
            row["allocated_energy_j"] is None for row in attempts
        ),
        "final_human_ai_outcomes": dict(sorted(outcomes.items())),
        "successful_attempt_count": sum(row["success"] for row in attempts),
        "completed_source_attempt_count": sum(
            row["completed"] for row in attempts
        ),
        "timed_out_source_attempt_count": sum(
            row["timed_out"] for row in attempts
        ),
        "success_source": (
            "Final results.output_level.jsonl, baseline=human_ai"
        ),
        "passed": all(
            [
                len(prompts) == 220,
                len(attempts) == 880,
                set(attempt_counts.values()) == {4},
                all(
                    repeats == {0, 1, 2, 3}
                    for repeats in repeats_by_task.values()
                ),
                len(prompt_counts) == 13,
                len(batch_rows) == 44,
                set(len(rows) for rows in queries_by_batch.values()) == {20},
                set(int(row["gpu_device_count"]) for row in batch_rows) == {8},
                abs(float(reconciliation_error)) < 1e-6,
            ]
        ),
    }
    if not qa["passed"]:
        raise AssertionError(f"QA failed: {qa}")

    return {
        "methodology": {
            "study": (
                "GLM-5.2 GDPVal expert-with-AI-reference category "
                "value-energy experiment"
            ),
            "primary_baseline": "human_ai",
            "judge_model": "models/gemini-3.1-flash-lite",
            "success_definition": "Final GLM-vs-expert-with-AI outcome in {win, tie}",
            "generated_value_rule": (
                "Recorded expert-with-AI completion time × hourly wage, counted once for each qualifying "
                "prompt"
            ),
            "time_saved_rule": (
                "Recorded expert-with-AI completion time minus mean AI-system completion time across successful "
                "attempts; negative values are not clamped"
            ),
            "three_way_time_savings_rule": (
                "For the same qualifying prompts: expert-only to GLM, "
                "expert-only to expert-with-AI, and expert-with-AI to GLM. Percentage saved "
                "uses the starting-point hours for each comparison. Hours/MJ "
                "uses the category's complete all-four-run GLM energy."
            ),
            "energy_rule": (
                "All four GLM attempts per prompt, including failed/incomplete "
                "runs; batch energy sums all eight GPUs"
            ),
            "thresholds": [1, 2, 3, 4],
        },
        "thresholds": thresholds,
        "qa": qa,
        "batch_qa": batch_qa,
        "prompt_ledger": sorted(
            prompts,
            key=lambda row: (
                row["category"],
                row["occupation"],
                row["task_id"],
            ),
        ),
        "attempt_ledger": sorted(
            attempts,
            key=lambda row: (
                row["category"],
                row["occupation"],
                row["task_id"],
                row["repeat"],
            ),
        ),
        "source_hashes": {
            "query_ledger_sha256": sha256(QUERY_LEDGER),
            "batch_ledger_sha256": sha256(BATCH_LEDGER),
            "results_sha256": sha256(RESULTS),
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
