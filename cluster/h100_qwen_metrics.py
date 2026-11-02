#!/usr/bin/env python3
"""Calculate five efficiency metrics for Qwen3.6-27B on one H100.

The formulas and fixed bootstrap implementation are imported from the frozen
analysis module. Only the exact completed one-H100 run's outcomes, end-to-end
latencies, and measured batch GPU-energy ledger are substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = (
    "20260728T200027Z_qwen36_27b_gdpval_occ_repeat4_full_"
    "norubric_h100_1g_128k_vllm019"
)
RESTORED_RUN = Path(os.environ.get("J2G_H100_RUN_ROOT", ROOT / ".codex_tmp_qwen_pairwise/restored_runs" / RUN_ID))
WORKFLOW = Path(os.environ.get("J2G_H100_WORKFLOW_ROOT", ROOT / ".codex_tmp_qwen_pairwise/workflow_20260815T073235Z"))
PAIRWISE_STATE = (
    WORKFLOW
    / "reports/qwen36_27b_norubric_h100_gemini31_flash_lite"
)
METRICS_MODULE_PATH = ROOT / "analysis/metrics.py"
DEFAULT_OUTPUT_DIR = (
    ROOT / "reports/qwen36_27b_one_h100_paper_five_metrics_20260815"
)
TRACE_PATH = (
    RESTORED_RUN
    / "reports/canonical/qwen36_occ_repeat4_norubric_h100_canonical_traces.jsonl"
)
BATCH_PATHS = (
    RESTORED_RUN
    / "attempts/attempt_000_start_0000/runs/gdpval_smoke/"
    "run_openhands_Qwen_Qwen3_6_27B_gdpval/batches.jsonl",
    RESTORED_RUN
    / "attempts/attempt_005_start_0860/runs/gdpval_smoke/"
    "run_openhands_Qwen_Qwen3_6_27B_gdpval/batches.jsonl",
)
RESULT_PATH = PAIRWISE_STATE / "results.output_level.jsonl"
MANIFEST_PATH = PAIRWISE_STATE / "comparison_manifest.unblinded.jsonl"
PAIRWISE_VALIDATION_PATH = PAIRWISE_STATE / "final_validation_report.json"
SOURCE_OUTPUTS_PATH = (
    WORKFLOW
    / "reports/qwen36_27b_norubric_h100_source/qwen_source_outputs.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="strict") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_metrics_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "joules_to_gdp_paper_analysis_h100", METRICS_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load analysis module: {METRICS_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_bootstrap_inputs(
    metrics_module: Any,
    task_meta: pd.DataFrame,
    weights: pd.DataFrame,
) -> Any:
    occupations = sorted(weights["occupation"].tolist())
    category_lookup = weights.set_index("occupation")["category"].to_dict()
    weight_lookup = weights.set_index("occupation")[
        "gdp_structure_weight"
    ].to_dict()
    rng = np.random.default_rng(metrics_module.BOOTSTRAP_SEED)
    return metrics_module.AnalysisInputs(
        task_meta=task_meta,
        weights=weights,
        models={},
        attempts={},
        occupation_order=occupations,
        category_index_by_occupation=np.array(
            [metrics_module.CATEGORIES.index(category_lookup[item]) for item in occupations],
            dtype=int,
        ),
        weight_vector=np.array([weight_lookup[item] for item in occupations]),
        bootstrap_draws=rng.integers(
            0,
            5,
            size=(metrics_module.BOOTSTRAP_SAMPLES, len(occupations), 5),
            dtype=np.int16,
        ),
        source_inventory=[],
        human_ai_model_counts=Counter(),
    )


def metric_payload(
    global_point: dict[str, float],
    global_uncertainty: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    fields = (
        "generated_value_usd",
        "hours_saved",
        "energy_mj",
        "value_per_mj",
        "hours_per_mj",
    )
    return {
        field: {
            "estimate": float(global_point[field]),
            "bootstrap_mean": float(global_uncertainty[field]["bootstrap_mean"]),
            "bootstrap_sd": float(global_uncertainty[field]["bootstrap_sd"]),
            "ci95_low": float(global_uncertainty[field]["ci95_low"]),
            "ci95_high": float(global_uncertainty[field]["ci95_high"]),
        }
        for field in fields
    }


def load_h100_inputs(
    task_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    traces = read_jsonl(TRACE_PATH)
    results = read_jsonl(RESULT_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    validation = json.loads(PAIRWISE_VALIDATION_PATH.read_text())
    source_outputs = json.loads(SOURCE_OUTPUTS_PATH.read_text())

    batches: list[dict[str, Any]] = []
    for path in BATCH_PATHS:
        source_name = path.parents[3].name
        for row in read_jsonl(path):
            batches.append({"source_name": source_name, **row})

    if len(traces) != 880:
        raise AssertionError(f"expected 880 canonical traces, got {len(traces)}")
    if len(batches) != 44:
        raise AssertionError(f"expected 44 measured batches, got {len(batches)}")
    if len(source_outputs) != 880:
        raise AssertionError("source-output cardinality changed")
    if not validation.get("complete"):
        raise AssertionError("pairwise strict validation is incomplete")
    if int(validation.get("unique_verdict_ids") or 0) != 1760:
        raise AssertionError("pairwise strict validation is not 1,760 verdicts")
    if int(validation.get("unresolved_failures", -1)) != 0:
        raise AssertionError("pairwise strict validation has unresolved failures")
    if not all(validation.get("checks", {}).values()):
        raise AssertionError("one or more pairwise validation checks failed")

    results_by_baseline: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for baseline in ("human", "human_ai"):
        rows = [
            row
            for row in results
            if row["condition"] == "without_rubric"
            and row["baseline"] == baseline
        ]
        by_key = {(str(row["task_id"]), int(row["repeat"])): row for row in rows}
        if len(rows) != 880 or len(by_key) != 880:
            raise AssertionError(f"{baseline}: result coverage is not 880 unique rows")
        if any(row["agent_outcome"] not in {"win", "tie", "loss"} for row in rows):
            raise AssertionError(f"{baseline}: invalid agent outcome")
        results_by_baseline[baseline] = by_key

    trace_by_key = {
        (str(row["task_id"]), int(row["repeat_index"])): row for row in traces
    }
    if len(trace_by_key) != 880:
        raise AssertionError("canonical trace task/repeat keys are not unique")
    expected_task_ids = set(task_meta["task_id"])
    if {key[0] for key in trace_by_key} != expected_task_ids:
        raise AssertionError("H100 trace task set differs from the configured task set")
    if set(trace_by_key) != set(results_by_baseline["human"]):
        raise AssertionError("H100 traces and expert-only reference verdicts do not align")

    batch_by_key = {
        (str(row["source_name"]), str(row["batch_id"])): row for row in batches
    }
    if len(batch_by_key) != 44:
        raise AssertionError("source-attempt/batch keys are not unique")
    traces_by_batch: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        key = (str(trace["source_name"]), str(trace["batch_id"]))
        traces_by_batch[key].append(trace)
    if set(traces_by_batch) != set(batch_by_key):
        raise AssertionError("trace and batch ledgers do not cover identical batches")
    for key, rows in traces_by_batch.items():
        batch = batch_by_key[key]
        if len(rows) != 20 or int(batch["batch_size"]) != 20:
            raise AssertionError(f"{key}: expected exactly 20 attempts")
        trace_queries = {str(row["query_id"]) for row in rows}
        if trace_queries != set(map(str, batch["query_ids"])):
            raise AssertionError(f"{key}: trace/batch query IDs differ")
        if list(batch["telemetry_device_ids"]) != [3]:
            raise AssertionError(f"{key}: telemetry was not isolated to GPU 3")
        energy_j = float(batch["batch_gpu_energy_joules"])
        if not math.isfinite(energy_j) or energy_j <= 0:
            raise AssertionError(f"{key}: invalid measured GPU energy")

    human_manifest = [
        row
        for row in manifest
        if row["condition"] == "without_rubric" and row["baseline"] == "human"
    ]
    manifest_by_key = {
        (str(row["task_id"]), int(row["repeat"])): row
        for row in human_manifest
    }
    if len(human_manifest) != 880 or len(manifest_by_key) != 880:
        raise AssertionError("expert-only reference manifest coverage is not 880 unique rows")

    attempt_rows: list[dict[str, Any]] = []
    for key, trace in trace_by_key.items():
        task_id, repeat = key
        batch_key = (str(trace["source_name"]), str(trace["batch_id"]))
        batch = batch_by_key[batch_key]
        result = results_by_baseline["human"][key]
        latency_seconds = float(trace["total_wall_clock_s"])
        if not math.isfinite(latency_seconds) or latency_seconds <= 0:
            raise AssertionError(f"{key}: invalid end-to-end latency")
        attempt_rows.append(
            {
                "task_id": task_id,
                "repeat": repeat,
                "agent_outcome": str(result["agent_outcome"]),
                "success": result["agent_outcome"] in {"win", "tie"},
                "latency_hours": latency_seconds / 3600,
                "latency_seconds": latency_seconds,
                "energy_mj": float(batch["batch_gpu_energy_joules"])
                / 20
                / 1_000_000,
                "batch_key": f"{batch_key[0]}::{batch_key[1]}",
                "completed": bool(trace["completed"]),
                "timed_out": bool(trace["timed_out"]),
                "unscorable_reason": trace.get("unscorable_reason"),
                "agent_quality_status": str(
                    manifest_by_key[key]["agent_quality_status"]
                ),
            }
        )

    attempts = pd.DataFrame(attempt_rows)
    if not attempts.groupby("task_id").size().eq(4).all():
        raise AssertionError("H100 task coverage is not exactly four attempts each")
    if any(set(group["repeat"]) != {0, 1, 2, 3} for _, group in attempts.groupby("task_id")):
        raise AssertionError("H100 repeat coverage is not exactly 0-3")

    prompt_rows: list[dict[str, Any]] = []
    for task_id, rows in attempts.groupby("task_id"):
        successful = rows[rows["success"]]
        prompt_rows.append(
            {
                "task_id": task_id,
                "success_count": int(successful.shape[0]),
                "mean_success_latency_hours": (
                    float(successful["latency_hours"].mean())
                    if not successful.empty
                    else math.nan
                ),
                "energy_all_four_runs_mj": float(rows["energy_mj"].sum()),
            }
        )
    prompt = task_meta.merge(
        pd.DataFrame(prompt_rows), on="task_id", how="left", validate="one_to_one"
    )
    if len(prompt) != 220 or prompt["success_count"].isna().any():
        raise AssertionError("H100 prompt metadata did not join to all 220 tasks")
    if not prompt.groupby("occupation").size().eq(5).all():
        raise AssertionError("H100 prompt panel is not five tasks per occupation")

    raw_energy_mj = sum(float(row["batch_gpu_energy_joules"]) for row in batches) / 1e6
    allocated_energy_mj = float(attempts["energy_mj"].sum())
    if not math.isclose(raw_energy_mj, allocated_energy_mj, abs_tol=1e-12):
        raise AssertionError("H100 measured and allocated GPU energy do not reconcile")

    source_names = Counter(str(row["source_name"]) for row in traces)
    if sum(source_names.values()) != len(traces):
        raise AssertionError("H100 source attribution does not cover traces")

    quality_counts = Counter(str(row["agent_quality_status"]) for row in human_manifest)
    unscorable_counts = Counter(
        str(row["unscorable_reason"])
        for row in attempt_rows
        if row["unscorable_reason"] is not None
    )
    qa = {
        "passed": True,
        "run_id": RUN_ID,
        "canonical_rows": len(traces),
        "unique_prompts": 220,
        "attempts": len(attempts),
        "batches": len(batches),
        "gpu_name": "NVIDIA H100",
        "gpu_device_ids": [3],
        "source_selection": dict(source_names),
        "completed_attempts": int(attempts["completed"].sum()),
        "timed_out_attempts": int(attempts["timed_out"].sum()),
        "unscorable_reasons": dict(unscorable_counts),
        "quality_states": dict(quality_counts),
        "human_outcomes": dict(
            Counter(row["agent_outcome"] for row in results_by_baseline["human"].values())
        ),
        "human_ai_outcomes": dict(
            Counter(
                row["agent_outcome"]
                for row in results_by_baseline["human_ai"].values()
            )
        ),
        "pairwise_valid_verdicts": int(validation["unique_verdict_ids"]),
        "pairwise_unresolved_failures": int(validation["unresolved_failures"]),
        "pairwise_all_validation_checks_true": all(validation["checks"].values()),
        "measured_gpu_energy_mj": raw_energy_mj,
        "allocated_gpu_energy_mj": allocated_energy_mj,
        "energy_reconciliation_error_mj": allocated_energy_mj - raw_energy_mj,
    }
    return prompt, batches, qa


def calculate() -> dict[str, Any]:
    metrics_module = load_metrics_module()
    task_meta, weights = metrics_module.load_task_metadata()
    prompt, batches, qa = load_h100_inputs(task_meta)
    bootstrap_inputs = make_bootstrap_inputs(metrics_module, task_meta, weights)
    category_point, global_point, occupation_point = metrics_module.point_metrics(
        prompt,
        weights,
        threshold=1,
        baseline_hours_column="human_only_hours",
        wage_column="median_hourly_wage_usd",
    )
    category_uncertainty, global_uncertainty = metrics_module.bootstrap_metrics(
        bootstrap_inputs,
        prompt,
        threshold=1,
        baseline_hours_column="human_only_hours",
        wage_column="median_hourly_wage_usd",
        point_categories=category_point,
        point_global=global_point,
    )

    raw_energy_mj = sum(float(row["batch_gpu_energy_joules"]) for row in batches) / 1e6
    if not math.isclose(raw_energy_mj, global_point["raw_energy_mj"], abs_tol=1e-12):
        raise AssertionError("raw H100 energy does not match the metric input")
    if not math.isclose(
        global_point["generated_value_usd"] / global_point["energy_mj"],
        global_point["value_per_mj"],
        abs_tol=1e-12,
    ):
        raise AssertionError("Value/MJ arithmetic failed")
    if not math.isclose(
        global_point["hours_saved"] / global_point["energy_mj"],
        global_point["hours_per_mj"],
        abs_tol=1e-12,
    ):
        raise AssertionError("Hours/MJ arithmetic failed")

    category_ci = category_uncertainty.set_index("occupation_category")
    category_rows: list[dict[str, Any]] = []
    for row in category_point.to_dict(orient="records"):
        uncertainty_row = category_ci.loc[row["occupation_category"]]
        for field in (
            "generated_value_usd",
            "energy_mj",
            "value_per_mj",
            "hours_saved",
            "hours_per_mj",
        ):
            row[f"{field}_ci95_low"] = float(
                uncertainty_row[f"{field}_ci95_low"]
            )
            row[f"{field}_ci95_high"] = float(
                uncertainty_row[f"{field}_ci95_high"]
            )
        category_rows.append(row)

    metrics = metric_payload(global_point, global_uncertainty)
    return {
        "model": "Qwen3.6-27B",
        "hardware": "1×NVIDIA H100 (GPU 3)",
        "energy_boundary": "GPU",
        "energy_components": ["gpu"],
        "run_id": RUN_ID,
        "benchmark": "GDPVal gold subset",
        "condition": "standard",
        "comparison_baseline": "expert-only reference deliverable",
        "success_definition": "Gemini source-blind judgment is win or tie",
        "credit_threshold": "at least 1 success in 4 attempts; value counted once",
        "valuation_basis": "recorded expert completion time (hours) × May 2024 OEWS median hourly wage",
        "weighting": (
            "GDPVal sector shares normalized across nine sectors, then within-sector "
            "OEWS occupation wage-bill weights; the five tasks in each occupation are "
            "averaged, occupation weights sum to one for GDP-level aggregation, and "
            "occupation weights are renormalized to one within each occupation category "
            "for category-level comparisons"
        ),
        "bootstrap": {
            "unit": "distinct prompt within each of 44 occupations",
            "samples": metrics_module.BOOTSTRAP_SAMPLES,
            "seed": metrics_module.BOOTSTRAP_SEED,
            "interval": "percentile 95%",
            "four_attempts_retained_together": True,
        },
        "metrics": metrics,
        "supporting": {
            "n_unique_prompts": 220,
            "attempts": 880,
            "qualifying_prompts": int((prompt["success_count"] >= 1).sum()),
            "successful_attempts": int(prompt["success_count"].sum()),
            "attempt_win_or_tie_rate": float(global_point["attempt_win_or_tie_rate"]),
            "raw_measured_gpu_energy_mj": raw_energy_mj,
            "raw_measured_gpu_energy_kwh": raw_energy_mj / 3.6,
            "gdp_weighted_gpu_energy_mj": float(global_point["energy_mj"]),
            "raw_unweighted_generated_value_usd": float(
                global_point["raw_generated_value_usd"]
            ),
            "raw_unweighted_hours_saved": float(global_point["raw_hours_saved"]),
        },
        "category_breakdown": category_rows,
        "occupation_breakdown": occupation_point.to_dict(orient="records"),
        "qa": qa,
        "sources": {
            "paper_analysis": {
                "path": str(METRICS_MODULE_PATH.relative_to(ROOT)),
                "sha256": sha256(METRICS_MODULE_PATH),
            },
            "canonical_traces": {
                "path": str(TRACE_PATH.relative_to(ROOT)),
                "sha256": sha256(TRACE_PATH),
            },
            "measured_batches": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                for path in BATCH_PATHS
            ],
            "pairwise_results": {
                "path": str(RESULT_PATH.relative_to(ROOT)),
                "sha256": sha256(RESULT_PATH),
            },
            "pairwise_manifest": {
                "path": str(MANIFEST_PATH.relative_to(ROOT)),
                "sha256": sha256(MANIFEST_PATH),
            },
            "pairwise_validation": {
                "path": str(PAIRWISE_VALIDATION_PATH.relative_to(ROOT)),
                "sha256": sha256(PAIRWISE_VALIDATION_PATH),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = calculate()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "qwen36_one_h100_paper_five_metrics_20260815.json"
    headline_path = output_dir / "qwen36_one_h100_paper_five_metrics_20260815.csv"
    category_path = output_dir / "qwen36_one_h100_paper_category_breakdown_20260815.csv"
    occupation_path = output_dir / "qwen36_one_h100_paper_occupation_breakdown_20260815.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    headline_rows = [
        {"metric": metric, **values} for metric, values in report["metrics"].items()
    ]
    pd.DataFrame(headline_rows).to_csv(headline_path, index=False)
    pd.DataFrame(report["category_breakdown"]).to_csv(category_path, index=False)
    pd.DataFrame(report["occupation_breakdown"]).to_csv(occupation_path, index=False)
    print(
        json.dumps(
            {
                "output": str(json_path),
                "energy_boundary": report["energy_boundary"],
                "headline_csv": str(headline_path),
                "metrics": report["metrics"],
                "supporting": report["supporting"],
                "qa": report["qa"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
