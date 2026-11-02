#!/usr/bin/env python3
"""Recompute GDP-structured task value for the Qwen B200 run.

The economic task values and 44 occupation weights are frozen from the GLM
analysis. Outcomes, latency, and energy come only from the independently
validated Qwen B200 run and its source-blind Gemini pairwise evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import glm_gdp_structured_value_capture as glm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    ROOT
    / "outputs/019fde63-1f71-7461-a23f-2f017c2c95cd"
    / "qwen36_b200_gdp_structured_value_energy_20260815/inputs"
)
FROZEN_GLM_REPORT = (
    ROOT / "reports/glm52_norubric_gdp_structured_value_capture_20260724.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def frozen_economic_inputs() -> tuple[
    dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]
]:
    report = json.loads(FROZEN_GLM_REPORT.read_text())
    weights = [dict(row) for row in report["occupation_weights"]]
    by_occupation = {row["occupation"]: row for row in weights}
    if len(weights) != 44 or len(by_occupation) != 44:
        raise AssertionError("frozen GLM report does not contain 44 unique weights")
    if not math.isclose(
        sum(row["gdp_structure_weight"] for row in weights), 1.0, abs_tol=1e-12
    ):
        raise AssertionError("frozen GDP-structure weights do not sum to one")
    return report, weights, by_occupation


def load_b200(
    input_dir: Path,
    frozen_report: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[tuple[str, int], dict[str, Any]]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    manifest_all = read_jsonl(input_dir / "comparison_manifest.unblinded.jsonl")
    results_all = read_jsonl(input_dir / "results.output_level.jsonl")
    traces = read_jsonl(input_dir / "traces.jsonl")
    batches = read_jsonl(input_dir / "batches.jsonl")
    final_manifest = json.loads((input_dir / "final_manifest.json").read_text())
    pairwise_validation = json.loads(
        (input_dir / "pairwise_final_validation_report.json").read_text()
    )

    if final_manifest.get("status") != "ok" or final_manifest.get(
        "strict_postprocess_status"
    ) != "ok":
        raise AssertionError("B200 final manifest is not strict-clean")
    if not pairwise_validation.get("complete"):
        raise AssertionError("pairwise final validation is incomplete")
    if pairwise_validation.get("unresolved_failures") != 0:
        raise AssertionError("pairwise validation has unresolved failures")
    if pairwise_validation.get("unique_verdict_ids") != 1760:
        raise AssertionError("pairwise validation does not contain 1,760 verdicts")

    human_manifest = [
        row
        for row in manifest_all
        if row["condition"] == "without_rubric" and row["baseline"] == "human"
    ]
    if len(human_manifest) != 880:
        raise AssertionError(f"expected 880 expert-only reference manifest rows, got {len(human_manifest)}")
    manifest_by_key = {
        (str(row["task_id"]), int(row["repeat"])): row for row in human_manifest
    }
    if len(manifest_by_key) != 880:
        raise AssertionError("duplicate B200 expert-only reference manifest keys")

    results_by_baseline: dict[
        str, dict[tuple[str, int], dict[str, Any]]
    ] = {}
    for baseline in ("human", "human_ai"):
        rows = [
            row
            for row in results_all
            if row["condition"] == "without_rubric" and row["baseline"] == baseline
        ]
        if len(rows) != 880:
            raise AssertionError(f"expected 880 {baseline} results, got {len(rows)}")
        by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            key = (str(row["task_id"]), int(row["repeat"]))
            if key in by_key:
                raise AssertionError(f"duplicate {baseline} result {key}")
            if row["agent_outcome"] not in {"win", "tie", "loss"}:
                raise AssertionError(f"invalid {baseline} outcome {key}")
            by_key[key] = row
        results_by_baseline[baseline] = by_key

    trace_by_slot = {
        (int(row["batch_index"]), int(row["batch_position"])): row
        for row in traces
    }
    batch_by_index = {int(row["batch_index"]): row for row in batches}
    if len(trace_by_slot) != 880 or len(batch_by_index) != 44:
        raise AssertionError("B200 trace or batch cardinality mismatch")

    frozen_human = {
        str(row["baseline_task_id"]): row
        for row in frozen_report["prompt_ledgers"]["human"]
    }
    if len(frozen_human) != 220:
        raise AssertionError("frozen economic task metadata is incomplete")
    current_baselines = {str(row["baseline_task_id"]) for row in human_manifest}
    if current_baselines != set(frozen_human):
        raise AssertionError("B200 task set differs from the frozen GLM task set")

    attempts: list[dict[str, Any]] = []
    batch_occupations: dict[int, set[str]] = defaultdict(set)
    for key, manifest in manifest_by_key.items():
        task_id, repeat = key
        source_query_id = str(manifest["source_query_id"])
        if not source_query_id.startswith("q"):
            raise AssertionError(f"unexpected source query id {source_query_id}")
        global_index = int(source_query_id[1:])
        batch_index, batch_position = divmod(global_index, 20)
        trace = trace_by_slot.get((batch_index, batch_position))
        batch = batch_by_index.get(batch_index)
        if trace is None or batch is None:
            raise AssertionError(f"missing B200 telemetry for {source_query_id}")
        if trace["batch_id"] != batch["batch_id"]:
            raise AssertionError(f"batch mismatch for {source_query_id}")
        if str(trace["query_id"]) != f"q{batch_position:04d}":
            raise AssertionError(f"local query-position mismatch for {source_query_id}")
        if int(batch["batch_size"]) != 20:
            raise AssertionError("B200 batch size changed from 20")

        baseline_task_id = str(manifest["baseline_task_id"])
        meta = frozen_human[baseline_task_id]
        if meta["occupation"] != manifest["occupation"]:
            raise AssertionError(f"occupation mismatch for {source_query_id}")
        if meta["sector"] != manifest["sector"]:
            raise AssertionError(f"sector mismatch for {source_query_id}")
        human_value = float(meta["human_only_value_usd"])
        human_ai_value = float(meta["human_ai_value_usd"])
        hourly_wage = float(meta["hourly_wage_usd"])
        if not math.isclose(
            human_value,
            float(meta["human_only_hours"]) * hourly_wage,
            abs_tol=1e-8,
        ):
            raise AssertionError(f"frozen expert-only valuation mismatch for {task_id}")
        if not math.isclose(
            human_ai_value,
            float(meta["human_ai_hours"]) * hourly_wage,
            abs_tol=1e-8,
        ):
            raise AssertionError(f"frozen expert-with-AI valuation mismatch for {task_id}")
        energy_j = float(batch["batch_gpu_energy_joules"]) / 20
        latency_seconds = float(trace["total_wall_clock_s"])
        if not math.isfinite(energy_j) or energy_j <= 0:
            raise AssertionError(f"invalid B200 energy for {source_query_id}")
        if not math.isfinite(latency_seconds) or latency_seconds <= 0:
            raise AssertionError(f"invalid B200 latency for {source_query_id}")

        batch_occupations[batch_index].add(str(meta["occupation"]))
        attempts.append(
            {
                "task_id": task_id,
                "baseline_task_id": baseline_task_id,
                "repeat": repeat,
                "occupation": str(meta["occupation"]),
                "onet_code": str(meta["onet_code"]),
                "category": str(meta["category"]),
                "sector": str(meta["sector"]),
                "hourly_wage_usd": hourly_wage,
                "human_only_hours": float(meta["human_only_hours"]),
                "human_ai_hours": float(meta["human_ai_hours"]),
                "human_only_value_usd": human_value,
                "human_ai_value_usd": human_ai_value,
                "latency_hours": latency_seconds / 3600,
                "latency_seconds": latency_seconds,
                "energy_j": energy_j,
                "energy_mj": energy_j / 1_000_000,
                "batch_key": str(batch["batch_id"]),
                "batch_index": batch_index,
                "batch_position": batch_position,
                "source_query_id": source_query_id,
                "local_query_id": str(trace["query_id"]),
                "completed": bool(trace["completed"]),
                "timed_out": bool(trace["timed_out"]),
                "agent_quality_status": str(manifest["agent_quality_status"]),
            }
        )

    if any(len(value) != 1 for value in batch_occupations.values()):
        raise AssertionError("a B200 batch mixes occupations")
    counts = Counter(row["task_id"] for row in attempts)
    if len(counts) != 220 or set(counts.values()) != {4}:
        raise AssertionError("B200 task/repeat coverage is not 220 x 4")
    repeats: dict[str, set[int]] = defaultdict(set)
    for row in attempts:
        repeats[row["task_id"]].add(row["repeat"])
    if any(value != {0, 1, 2, 3} for value in repeats.values()):
        raise AssertionError("B200 repeat coverage is not exactly 0-3")

    task_meta: dict[str, dict[str, Any]] = {}
    for row in attempts:
        meta = {
            key: row[key]
            for key in (
                "task_id",
                "baseline_task_id",
                "occupation",
                "onet_code",
                "category",
                "sector",
                "hourly_wage_usd",
                "human_only_hours",
                "human_ai_hours",
                "human_only_value_usd",
                "human_ai_value_usd",
            )
        }
        prior = task_meta.setdefault(row["task_id"], meta)
        if prior != meta:
            raise AssertionError(f"inconsistent B200 task metadata {row['task_id']}")

    batch_rows: list[dict[str, Any]] = []
    for batch_index in sorted(batch_by_index):
        batch = batch_by_index[batch_index]
        batch_rows.append(
            {
                "batch_key": str(batch["batch_id"]),
                "batch_index": batch_index,
                "occupation": next(iter(batch_occupations[batch_index])),
                "attempts": int(batch["batch_size"]),
                "gpu_name": "NVIDIA B200",
                "gpu_device_count": len(batch["telemetry_device_ids"]),
                "gpu_energy_joules": float(batch["batch_gpu_energy_joules"]),
                "gpu_energy_mj": float(batch["batch_gpu_energy_joules"]) / 1_000_000,
                "wall_clock_s": float(batch["batch_total_wall_clock_s"]),
            }
        )
    return attempts, results_by_baseline, task_meta, batch_rows, {
        "final_manifest": final_manifest,
        "pairwise_validation": pairwise_validation,
        "manifest_quality": dict(Counter(row["agent_quality_status"] for row in human_manifest)),
    }


def flatten_csvs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with (output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write("occupation_weights.csv", report["occupation_weights"])
    write("category_weights.csv", report["category_weights"])
    write("batch_reconciliation.csv", report["batch_reconciliation"])

    categories: list[dict[str, Any]] = []
    occupations: list[dict[str, Any]] = []
    globals_: list[dict[str, Any]] = []
    bootstrap: list[dict[str, Any]] = []
    for baseline, threshold_map in report["analyses"].items():
        for threshold, result in threshold_map.items():
            categories.extend({"baseline": baseline, **row} for row in result["categories"])
            occupations.extend({"baseline": baseline, **row} for row in result["occupations"])
            global_row: dict[str, Any] = {
                "baseline": baseline,
                "threshold": int(threshold),
                **result["global_indices"],
            }
            for weighting, values in result["global_supporting_metrics"].items():
                for key, value in values.items():
                    global_row[f"{weighting}__{key}"] = value
            globals_.append(global_row)
            for metric, bounds in result["bootstrap_95_ci"].items():
                bootstrap.append(
                    {
                        "baseline": baseline,
                        "threshold": int(threshold),
                        "metric": metric,
                        **bounds,
                    }
                )
    write("category_results.csv", categories)
    write("occupation_results.csv", occupations)
    write("global_metrics.csv", globals_)
    write("bootstrap_cis.csv", bootstrap)
    prompts = [
        row
        for baseline in ("human", "human_ai")
        for row in report["prompt_ledgers"][baseline]
    ]
    write("prompt_ledger.csv", prompts)


def calculate(input_dir: Path) -> dict[str, Any]:
    frozen_report, weight_rows, weight_by_occ = frozen_economic_inputs()
    attempts, results_by_baseline, task_meta, batch_rows, validation = load_b200(
        input_dir, frozen_report
    )

    rng = np.random.default_rng(glm.BOOTSTRAP_SEED)
    bootstrap_indices = rng.integers(
        0, 5, size=(glm.BOOTSTRAP_SAMPLES, 44, 5), dtype=np.int16
    )

    prompts_by_baseline: dict[str, list[dict[str, Any]]] = {}
    analyses: dict[str, Any] = {}
    for baseline in ("human", "human_ai"):
        prompts = glm.build_prompts(
            attempts, results_by_baseline[baseline], baseline, weight_by_occ
        )
        prompts_by_baseline[baseline] = prompts
        analyses[baseline] = {
            str(threshold): glm.calculate_threshold(
                prompts,
                weight_rows,
                weight_by_occ,
                threshold,
                bootstrap_indices,
            )
            for threshold in range(1, 5)
        }

    attempts_by_batch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        attempts_by_batch[row["batch_key"]].append(row)
    batch_reconciliation: list[dict[str, Any]] = []
    for batch in batch_rows:
        allocated = sum(row["energy_j"] for row in attempts_by_batch[batch["batch_key"]])
        source = float(batch["gpu_energy_joules"])
        batch_reconciliation.append(
            {
                **batch,
                "allocated_attempt_energy_j": allocated,
                "reconciliation_error_j": allocated - source,
            }
        )

    category_weights: list[dict[str, Any]] = []
    for category in glm.CATEGORIES.values():
        rows = [row for row in weight_rows if row["category"] == category]
        category_weights.append(
            {
                "category": category,
                "occupation_count": len(rows),
                "prompt_count": 5 * len(rows),
                "gdp_structure_weight": sum(row["gdp_structure_weight"] for row in rows),
                "wage_bill_only_weight": sum(row["wage_bill_only_weight"] for row in rows),
            }
        )

    source_energy_j = sum(row["gpu_energy_joules"] for row in batch_rows)
    allocated_energy_j = sum(row["energy_j"] for row in attempts)
    outcomes = {
        baseline: dict(Counter(row["agent_outcome"] for row in values.values()))
        for baseline, values in results_by_baseline.items()
    }
    final_anomalies = validation["final_manifest"].get("anomaly_counts", {})
    checks: dict[str, Any] = {
        "prompt_count": len(task_meta),
        "attempt_count": len(attempts),
        "attempts_per_prompt": sorted(set(Counter(row["task_id"] for row in attempts).values())),
        "occupation_count": len(weight_rows),
        "prompts_per_occupation": sorted(set(Counter(row["occupation"] for row in prompts_by_baseline["human"]).values())),
        "category_count": len(category_weights),
        "sector_count": len(glm.SECTOR_GDP_SHARES),
        "published_sector_share_sum": sum(glm.SECTOR_GDP_SHARES.values()),
        "gdp_structure_weight_sum": sum(row["gdp_structure_weight"] for row in weight_rows),
        "wage_bill_only_weight_sum": sum(row["wage_bill_only_weight"] for row in weight_rows),
        "batch_count": len(batch_rows),
        "attempts_per_batch": sorted(set(len(value) for value in attempts_by_batch.values())),
        "gpu_names": sorted(set(row["gpu_name"] for row in batch_rows)),
        "gpu_devices_per_batch": sorted(set(row["gpu_device_count"] for row in batch_rows)),
        "source_batch_energy_mj": source_energy_j / 1_000_000,
        "source_batch_energy_kwh": source_energy_j / 3_600_000,
        "allocated_attempt_energy_mj": allocated_energy_j / 1_000_000,
        "energy_reconciliation_error_j": allocated_energy_j - source_energy_j,
        "completed_attempts": sum(row["completed"] for row in attempts),
        "timed_out_attempts": sum(row["timed_out"] for row in attempts),
        "quality_states": validation["manifest_quality"],
        "final_anomaly_counts": final_anomalies,
        "outcomes": outcomes,
        "pairwise_valid_verdicts": validation["pairwise_validation"]["unique_verdict_ids"],
        "pairwise_unresolved_failures": validation["pairwise_validation"]["unresolved_failures"],
        "bootstrap_samples": glm.BOOTSTRAP_SAMPLES,
        "bootstrap_seed": glm.BOOTSTRAP_SEED,
    }
    checks["passed"] = all(
        [
            checks["prompt_count"] == 220,
            checks["attempt_count"] == 880,
            checks["attempts_per_prompt"] == [4],
            checks["occupation_count"] == 44,
            checks["prompts_per_occupation"] == [5],
            checks["category_count"] == 13,
            checks["sector_count"] == 9,
            math.isclose(checks["gdp_structure_weight_sum"], 1.0, abs_tol=1e-12),
            math.isclose(checks["wage_bill_only_weight_sum"], 1.0, abs_tol=1e-12),
            checks["batch_count"] == 44,
            checks["attempts_per_batch"] == [20],
            checks["gpu_names"] == ["NVIDIA B200"],
            checks["gpu_devices_per_batch"] == [1],
            abs(checks["energy_reconciliation_error_j"]) < 1e-6,
            checks["completed_attempts"] == 880,
            checks["timed_out_attempts"] == 0,
            checks["pairwise_valid_verdicts"] == 1760,
            checks["pairwise_unresolved_failures"] == 0,
            final_anomalies.get("rows") == 880,
            final_anomalies.get("missing_scores") == 0,
            final_anomalies.get("timed_out") == 0,
            final_anomalies.get("invalid_deliverables") == 0,
            final_anomalies.get("judge_errors") == 0,
        ]
    )
    if not checks["passed"]:
        raise AssertionError(f"B200 QA checks failed: {checks}")

    primary = analyses["human"]["1"]
    return {
        "methodology": {
            "study": "Qwen3.6-27B GDPVal on one B200",
            "index_name": "GDPVal GDP-structured value-capture index (GVCI)",
            "interpretation": (
                "Percentage of economically weighted benchmark output dollar value "
                "captured by Qwen3.6-27B; not a percentage of U.S. GDP automated."
            ),
            "success_definition": "Source-blind Gemini 3.1 Flash-Lite win or tie",
            "primary_variant": "Qwen vs the expert-only reference, at least 1 success in 4 runs",
            "thresholds": [1, 2, 3, 4],
            "task_value_rule": frozen_report["methodology"]["task_value_rule"],
            "weight_rule": frozen_report["methodology"]["weight_rule"],
            "energy_rule": (
                "All four Qwen B200 runs for every prompt, including quality-defect "
                "evidence; one B200 summed; equal within-batch allocation exactly "
                "reconciled to the 44 source batch energy measurements."
            ),
            "bootstrap_rule": frozen_report["methodology"]["bootstrap_rule"],
            "literature_status": frozen_report["methodology"]["literature_status"],
            "source_urls": frozen_report["methodology"]["source_urls"],
            "frozen_economic_input_report": str(FROZEN_GLM_REPORT.relative_to(ROOT)),
        },
        "headline": {
            "baseline": "human",
            "threshold": 1,
            "gvci_percent": primary["global_indices"]["gdp_structure_gvci_percent"],
            "gvci_95_ci": primary["bootstrap_95_ci"]["gdp_structure_gvci_percent"],
            "wage_bill_only_percent": primary["global_indices"]["wage_bill_only_value_capture_percent"],
            "unweighted_occupation_percent": primary["global_indices"]["unweighted_occupation_value_capture_percent"],
        },
        "analyses": analyses,
        "three_way_time": glm.calculate_three_way_time(
            prompts_by_baseline["human_ai"], weight_rows, bootstrap_indices
        ),
        "occupation_weights": weight_rows,
        "category_weights": category_weights,
        "prompt_ledgers": prompts_by_baseline,
        "batch_reconciliation": batch_reconciliation,
        "qa": checks,
        "source_inventory": {
            "frozen_glm_economic_report": {
                "path": str(FROZEN_GLM_REPORT.relative_to(ROOT)),
                "sha256": sha256(FROZEN_GLM_REPORT),
            },
            **{
                name: {
                    "path": str((input_dir / filename).relative_to(ROOT)),
                    "sha256": sha256(input_dir / filename),
                }
                for name, filename in {
                    "b200_traces": "traces.jsonl",
                    "b200_batches": "batches.jsonl",
                    "b200_final_manifest": "final_manifest.json",
                    "pairwise_results": "results.output_level.jsonl",
                    "pairwise_manifest": "comparison_manifest.unblinded.jsonl",
                    "pairwise_final_validation": "pairwise_final_validation_report.json",
                }.items()
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-dir", type=Path, required=True)
    args = parser.parse_args()
    report = calculate(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    flatten_csvs(report, args.csv_dir)
    print(json.dumps({"status": "ok", "output": str(args.output), "headline": report["headline"], "qa": report["qa"]}, indent=2))


if __name__ == "__main__":
    main()
