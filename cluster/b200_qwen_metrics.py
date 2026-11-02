#!/usr/bin/env python3
"""Calculate five efficiency metrics for Qwen3.6-27B on one B200.

This adapter imports the shared analysis implementation
instead of re-implementing its formulas. It swaps in only the completed B200
run's outcomes, successful-attempt latency, and measured GPU energy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import b200_qwen_gdp_structured_value_capture as b200


ROOT = Path(__file__).resolve().parents[1]
METRICS_MODULE_PATH = ROOT / "analysis/metrics.py"
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs/019fde63-1f71-7461-a23f-2f017c2c95cd"
    / "qwen36_b200_gdp_structured_value_energy_20260815"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metrics_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "joules_to_gdp_paper_analysis", METRICS_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load analysis module: {METRICS_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_b200_prompt_frame(
    task_meta: pd.DataFrame,
    attempts: list[dict[str, Any]],
    results: dict[tuple[str, int], dict[str, Any]],
) -> pd.DataFrame:
    attempts_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        attempts_by_task[str(attempt["task_id"])].append(attempt)

    rows: list[dict[str, Any]] = []
    for task_id, task_attempts in attempts_by_task.items():
        if len(task_attempts) != 4:
            raise AssertionError(f"{task_id}: expected four B200 attempts")
        successful = [
            attempt
            for attempt in task_attempts
            if results[(task_id, int(attempt["repeat"]))]["agent_outcome"]
            in {"win", "tie"}
        ]
        rows.append(
            {
                "task_id": task_id,
                "success_count": len(successful),
                "mean_success_latency_hours": (
                    float(np.mean([row["latency_hours"] for row in successful]))
                    if successful
                    else math.nan
                ),
                "energy_all_four_runs_mj": float(
                    sum(row["energy_mj"] for row in task_attempts)
                ),
            }
        )

    prompt = task_meta.merge(
        pd.DataFrame(rows), on="task_id", how="left", validate="one_to_one"
    )
    if len(prompt) != 220 or prompt["success_count"].isna().any():
        raise AssertionError("B200 prompt metadata did not join to all 220 tasks")
    if not prompt.groupby("occupation").size().eq(5).all():
        raise AssertionError("B200 prompt panel is not five tasks per occupation")
    return prompt


def make_bootstrap_inputs(
    metrics_module: Any,
    task_meta: pd.DataFrame,
    weights: pd.DataFrame,
) -> Any:
    occupations = sorted(weights["occupation"].tolist())
    category_lookup = weights.set_index("occupation")["category"].to_dict()
    weight_lookup = weights.set_index("occupation")["gdp_structure_weight"].to_dict()
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
    return {
        field: {
            "estimate": float(global_point[field]),
            "bootstrap_mean": float(global_uncertainty[field]["bootstrap_mean"]),
            "bootstrap_sd": float(global_uncertainty[field]["bootstrap_sd"]),
            "ci95_low": float(global_uncertainty[field]["ci95_low"]),
            "ci95_high": float(global_uncertainty[field]["ci95_high"]),
        }
        for field in (
            "generated_value_usd",
            "hours_saved",
            "energy_mj",
            "value_per_mj",
            "hours_per_mj",
        )
    }


def calculate(input_dir: Path) -> dict[str, Any]:
    metrics_module = load_metrics_module()
    task_meta, weights = metrics_module.load_task_metadata()
    frozen_report, _, _ = b200.frozen_economic_inputs()
    attempts, results_by_baseline, _, batch_rows, validation = b200.load_b200(
        input_dir, frozen_report
    )
    prompt = build_b200_prompt_frame(
        task_meta, attempts, results_by_baseline["human"]
    )
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

    category_ci = category_uncertainty.set_index("occupation_category")
    category_rows = []
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

    raw_energy_mj = sum(float(row["gpu_energy_mj"]) for row in batch_rows)
    if not math.isclose(raw_energy_mj, global_point["raw_energy_mj"], abs_tol=1e-12):
        raise AssertionError("raw B200 energy did not reconcile")
    if validation["pairwise_validation"]["unique_verdict_ids"] != 1760:
        raise AssertionError("pairwise verdict count changed")

    return {
        "model": "Qwen3.6-27B",
        "hardware": "1×NVIDIA B200",
        "energy_boundary": "GPU",
        "energy_components": ["gpu"],
        "benchmark": "GDPVal gold subset",
        "condition": "standard",
        "comparison_baseline": "expert-only reference deliverable",
        "success_definition": "Gemini source-blind judgment is win or tie",
        "credit_threshold": "at least 1 success in 4 attempts; value counted once",
        "valuation_basis": "recorded expert completion time (hours) × May 2024 OEWS median hourly wage",
        "weighting": (
            "GDPVal sector shares normalized across nine sectors, then within-sector "
            "May 2024 OEWS occupation wage-bill weights; occupation panels scaled by 44"
        ),
        "bootstrap": {
            "unit": "distinct prompt within each of 44 occupations",
            "samples": metrics_module.BOOTSTRAP_SAMPLES,
            "seed": metrics_module.BOOTSTRAP_SEED,
            "interval": "percentile 95%",
        },
        "metrics": metric_payload(global_point, global_uncertainty),
        "supporting": {
            "n_unique_prompts": 220,
            "attempts": 880,
            "qualifying_prompts": int((prompt["success_count"] >= 1).sum()),
            "successful_attempts": int(prompt["success_count"].sum()),
            "attempt_win_or_tie_rate": float(
                global_point["attempt_win_or_tie_rate"]
            ),
            "raw_measured_gpu_energy_mj": raw_energy_mj,
            "raw_measured_gpu_energy_kwh": raw_energy_mj / 3.6,
            "raw_unweighted_generated_value_usd": float(
                global_point["raw_generated_value_usd"]
            ),
            "raw_unweighted_hours_saved": float(
                global_point["raw_hours_saved"]
            ),
        },
        "category_breakdown": category_rows,
        "occupation_breakdown": occupation_point.to_dict(orient="records"),
        "qa": {
            "passed": True,
            "pairwise_valid_verdicts": validation["pairwise_validation"][
                "unique_verdict_ids"
            ],
            "pairwise_unresolved_failures": validation["pairwise_validation"][
                "unresolved_failures"
            ],
            "source_energy_reconciled": True,
        },
        "sources": {
            "paper_analysis": {
                "path": str(METRICS_MODULE_PATH.relative_to(ROOT)),
                "sha256": sha256(METRICS_MODULE_PATH),
            },
            "b200_inputs": {
                name: {
                    "path": str((input_dir / name).relative_to(ROOT)),
                    "sha256": sha256(input_dir / name),
                }
                for name in (
                    "traces.jsonl",
                    "batches.jsonl",
                    "results.output_level.jsonl",
                    "comparison_manifest.unblinded.jsonl",
                    "final_manifest.json",
                    "pairwise_final_validation_report.json",
                )
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=b200.DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = calculate(args.input_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "qwen36_b200_paper_five_metrics_20260815.json"
    category_path = args.output_dir / "qwen36_b200_paper_category_breakdown_20260815.csv"
    occupation_path = (
        args.output_dir / "qwen36_b200_paper_occupation_breakdown_20260815.csv"
    )
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(report["category_breakdown"]).to_csv(category_path, index=False)
    pd.DataFrame(report["occupation_breakdown"]).to_csv(occupation_path, index=False)
    print(json.dumps({"output": str(json_path), "energy_boundary": report["energy_boundary"], **report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
