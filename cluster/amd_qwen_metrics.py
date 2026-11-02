#!/usr/bin/env python3
"""Calculate five efficiency metrics for Qwen3.6-27B on AMD MI350X.

The economic definitions and 10,000-draw occupation-stratified bootstrap are
imported from the shared analysis module. This adapter swaps in the completed
AMD canonical outputs, the source-blind Gemini expert-only reference verdicts, exact
selected-output wall-clock latency, and every measured production/repair batch
that was required to construct the final 880-output canonical run.
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


ROOT = Path(__file__).resolve().parents[1]
METRICS_MODULE_PATH = ROOT / "analysis/metrics.py"
SOURCE_ROOT = ROOT / "reports/qwen36_27b_norubric_amd_mi350x_source_20260815"
JUDGE_ROOT = (
    ROOT
    / "reports/qwen36_27b_norubric_amd_mi350x_gemini31_flash_lite_20260815"
)
DEFAULT_INPUT_ROOT = (
    ROOT / "outputs/amd_qwen_mi350x_paper_five_metrics_20260815/inputs"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/amd_qwen_mi350x_paper_five_metrics_20260815"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metrics_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "joules_to_gdp_paper_analysis_amd", METRICS_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load analysis module: {METRICS_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_name(path: Path) -> str:
    for suffix in (".batches.jsonl", ".gdpval_smoke_ids.json"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    raise AssertionError(f"unexpected input filename: {path.name}")


def query_number(query_id: str) -> int:
    if len(query_id) < 2 or query_id[0] != "q" or not query_id[1:].isdigit():
        raise AssertionError(f"unexpected query id: {query_id}")
    return int(query_id[1:])


def manifest_global_position(row: dict[str, Any]) -> int:
    return 4 * int(row["source_task_index"]) + int(row["repeat_index"])


def load_energy_ledger(
    input_root: Path,
) -> tuple[
    dict[int, float],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Allocate every measured production and repair batch to target slots.

    Production query IDs retain their global qNNNN positions.  Isolated repair
    jobs use local q0000... IDs, so their manifests provide the global target
    slots.  A repaired slot therefore accumulates the failed/original energy
    and every repair attempt, accounting for the fact that failures consume
    energy even when they do not earn value.
    """

    batch_paths = sorted((input_root / "batches").glob("*.batches.jsonl"))
    manifest_paths = sorted(
        (input_root / "manifests").glob("*.gdpval_smoke_ids.json")
    )
    if len(batch_paths) != 8 or len(manifest_paths) != 8:
        raise AssertionError("expected eight AMD source batch and manifest files")

    manifests = {
        source_name(path): json.loads(path.read_text()) for path in manifest_paths
    }
    if set(manifests) != {source_name(path) for path in batch_paths}:
        raise AssertionError("batch/manifest source names differ")

    energy_by_global_j: dict[int, float] = defaultdict(float)
    contribution_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    physical_attempts = 0

    for batch_path in batch_paths:
        name = source_name(batch_path)
        source_manifest = manifests[name]
        manifest_globals = {
            manifest_global_position(row): row for row in source_manifest
        }
        if len(manifest_globals) != len(source_manifest):
            raise AssertionError(f"{name}: duplicate global manifest positions")
        rows = read_jsonl(batch_path)
        if not rows:
            raise AssertionError(f"{name}: empty batch ledger")
        source_query_numbers = {
            query_number(query_id)
            for row in rows
            for query_id in row["query_ids"]
        }
        uses_global_query_ids = source_query_numbers <= set(manifest_globals)
        if not uses_global_query_ids and max(source_query_numbers) >= len(source_manifest):
            raise AssertionError(f"{name}: query IDs map to neither global nor local manifest")

        for row in rows:
            batch_size = int(row["batch_size"])
            query_ids = [str(item) for item in row["query_ids"]]
            energy_j = float(row["batch_gpu_energy_joules"])
            if batch_size != len(query_ids) or batch_size <= 0:
                raise AssertionError(f"{name}/{row['batch_id']}: batch-size mismatch")
            if not math.isfinite(energy_j) or energy_j <= 0:
                raise AssertionError(f"{name}/{row['batch_id']}: invalid GPU energy")
            if len(row.get("telemetry_device_ids", [])) != 1:
                raise AssertionError(f"{name}/{row['batch_id']}: not a one-GPU ledger")
            if int(row.get("telemetry_sample_count", 0)) <= 0:
                raise AssertionError(f"{name}/{row['batch_id']}: missing telemetry samples")

            allocated_j = energy_j / batch_size
            physical_attempts += batch_size
            batch_rows.append(
                {
                    "source_name": name,
                    "batch_id": str(row["batch_id"]),
                    "batch_index": int(row["batch_index"]),
                    "batch_size": batch_size,
                    "gpu_energy_joules": energy_j,
                    "gpu_energy_mj": energy_j / 1e6,
                    "gpu_power_avg_watts": float(
                        row.get("batch_gpu_power_avg_watts")
                        or energy_j / float(row["batch_total_wall_clock_s"])
                    ),
                    "wall_clock_s": float(row["batch_total_wall_clock_s"]),
                    "telemetry_sample_count": int(row["telemetry_sample_count"]),
                    "telemetry_device_ids": json.dumps(row["telemetry_device_ids"]),
                    "query_id_mode": "global" if uses_global_query_ids else "local",
                }
            )
            for query_id in query_ids:
                number = query_number(query_id)
                if uses_global_query_ids:
                    global_position = number
                    manifest_row = manifest_globals[number]
                else:
                    manifest_row = source_manifest[number]
                    global_position = manifest_global_position(manifest_row)
                energy_by_global_j[global_position] += allocated_j
                contribution_rows.append(
                    {
                        "source_name": name,
                        "batch_id": str(row["batch_id"]),
                        "query_id": query_id,
                        "global_position": global_position,
                        "task_id": str(manifest_row["task_id"]),
                        "repeat": int(manifest_row["repeat_index"]),
                        "occupation": str(manifest_row["occupation"]),
                        "allocated_gpu_energy_joules": allocated_j,
                        "allocated_gpu_energy_mj": allocated_j / 1e6,
                    }
                )

    if set(energy_by_global_j) != set(range(880)):
        raise AssertionError("energy ledger does not cover exactly global positions 0-879")
    if len(batch_rows) != 48 or physical_attempts != 894:
        raise AssertionError("expected 48 measured batches and 894 physical attempts")
    raw_batch_j = sum(row["gpu_energy_joules"] for row in batch_rows)
    allocated_j = sum(energy_by_global_j.values())
    if not math.isclose(raw_batch_j, allocated_j, abs_tol=1e-6):
        raise AssertionError("AMD batch energy did not reconcile to slot allocations")

    production_j = sum(
        row["gpu_energy_joules"]
        for row in batch_rows
        if row["source_name"].startswith("attempt_")
    )
    repair_j = raw_batch_j - production_j
    return (
        dict(energy_by_global_j),
        contribution_rows,
        batch_rows,
        {
            "measured_batch_count": len(batch_rows),
            "production_batch_count": sum(
                row["source_name"].startswith("attempt_") for row in batch_rows
            ),
            "repair_batch_count": sum(
                not row["source_name"].startswith("attempt_") for row in batch_rows
            ),
            "physical_attempts_including_retries": physical_attempts,
            "canonical_output_slots": 880,
            "retry_or_repair_attempts": physical_attempts - 880,
            "raw_production_gpu_energy_mj": production_j / 1e6,
            "raw_repair_gpu_energy_mj": repair_j / 1e6,
            "raw_total_gpu_energy_mj": raw_batch_j / 1e6,
            "raw_total_gpu_energy_kwh": raw_batch_j / 3.6e6,
        },
    )


def load_amd_attempts(
    input_root: Path,
    energy_by_global_j: dict[int, float],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    source_outputs_path = SOURCE_ROOT / "qwen_source_outputs.json"
    outputs = json.loads(source_outputs_path.read_text())
    if len(outputs) != 880:
        raise AssertionError("expected 880 canonical AMD source outputs")
    outputs_by_query = {str(row["query_id"]): row for row in outputs}
    if len(outputs_by_query) != 880:
        raise AssertionError("duplicate canonical AMD source query IDs")

    manifest_rows = read_jsonl(SOURCE_ROOT / "comparison_manifest.unblinded.jsonl")
    human_manifest = [
        row
        for row in manifest_rows
        if row["condition"] == "without_rubric" and row["baseline"] == "human"
    ]
    if len(human_manifest) != 880:
        raise AssertionError("expected 880 expert-only reference comparison rows")
    manifest_by_query = {str(row["source_query_id"]): row for row in human_manifest}
    if len(manifest_by_query) != 880 or set(manifest_by_query) != set(outputs_by_query):
        raise AssertionError("expert-only reference manifest and canonical outputs do not align")

    results_rows = read_jsonl(JUDGE_ROOT / "results.output_level.jsonl")
    human_results = [
        row
        for row in results_rows
        if row["condition"] == "without_rubric" and row["baseline"] == "human"
    ]
    results_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in human_results:
        key = (str(row["task_id"]), int(row["repeat"]))
        if key in results_by_key or row["agent_outcome"] not in {"win", "tie", "loss"}:
            raise AssertionError(f"invalid or duplicate expert-only reference verdict: {key}")
        results_by_key[key] = row
    if len(results_by_key) != 880:
        raise AssertionError("expected 880 unique expert-only reference verdicts")

    validation = json.loads((JUDGE_ROOT / "final_validation_report.json").read_text())
    if not validation.get("complete") or validation.get("unresolved_failures") != 0:
        raise AssertionError("pairwise judging is not strict-clean")
    if validation.get("unique_verdict_ids") != 1760:
        raise AssertionError("pairwise judging does not contain 1,760 verdicts")

    attempts: list[dict[str, Any]] = []
    for query_id, source in outputs_by_query.items():
        manifest = manifest_by_query[query_id]
        global_position = int(source["query_index"])
        metadata_path = Path(source["metadata_path"])
        metadata = json.loads(metadata_path.read_text())
        canonical = metadata["canonical_row"]
        artifact = metadata["artifact_metadata"]
        key = (str(manifest["task_id"]), int(manifest["repeat"]))
        if global_position != int(metadata["global_position"]):
            raise AssertionError(f"{query_id}: metadata global-position mismatch")
        if global_position not in energy_by_global_j:
            raise AssertionError(f"{query_id}: missing allocated energy")
        if str(source["task_id"]) != key[0] or int(source["repeat"]) != key[1]:
            raise AssertionError(f"{query_id}: task/repeat mismatch")
        if not source["completed"] or source["timed_out"]:
            raise AssertionError(f"{query_id}: canonical output is incomplete")
        if not canonical["completed"] or canonical["timed_out"]:
            raise AssertionError(f"{query_id}: canonical trace is incomplete")
        latency_s = float(canonical["total_wall_clock_s"])
        if not math.isclose(latency_s, float(artifact["wall_clock_s"]), abs_tol=1e-9):
            raise AssertionError(f"{query_id}: latency metadata differs")
        if not math.isfinite(latency_s) or latency_s <= 0:
            raise AssertionError(f"{query_id}: invalid latency")
        attempts.append(
            {
                "global_position": global_position,
                "source_query_id": query_id,
                "task_id": key[0],
                "repeat": key[1],
                "occupation": str(manifest["occupation"]),
                "agent_outcome": str(results_by_key[key]["agent_outcome"]),
                "success": results_by_key[key]["agent_outcome"] in {"win", "tie"},
                "latency_seconds": latency_s,
                "latency_hours": latency_s / 3600,
                "energy_j": energy_by_global_j[global_position],
                "energy_mj": energy_by_global_j[global_position] / 1e6,
                "canonical_source_name": str(source["canonical_source_name"]),
                "candidate_quality_status": str(source["candidate_quality_status"]),
            }
        )

    if {row["global_position"] for row in attempts} != set(range(880)):
        raise AssertionError("canonical attempts do not cover global positions 0-879")
    counts = Counter(row["task_id"] for row in attempts)
    if len(counts) != 220 or set(counts.values()) != {4}:
        raise AssertionError("AMD attempt coverage is not 220 x 4")
    if any(row["agent_outcome"] not in {"win", "tie", "loss"} for row in attempts):
        raise AssertionError("AMD judgment contains an invalid outcome")
    return attempts, results_by_key, validation


def build_prompt_frame(
    task_meta: pd.DataFrame,
    attempts: list[dict[str, Any]],
) -> pd.DataFrame:
    attempts_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        attempts_by_task[row["task_id"]].append(row)
    rows = []
    for task_id, task_attempts in attempts_by_task.items():
        successful = [row for row in task_attempts if row["success"]]
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
        raise AssertionError("AMD prompt frame did not join all 220 tasks")
    if not prompt.groupby("occupation").size().eq(5).all():
        raise AssertionError("AMD prompt panel is not five tasks per occupation")
    return prompt


def bootstrap_inputs(metrics_module: Any, task_meta: pd.DataFrame, weights: pd.DataFrame) -> Any:
    occupations = sorted(weights["occupation"].tolist())
    category = weights.set_index("occupation")["category"].to_dict()
    weight = weights.set_index("occupation")["gdp_structure_weight"].to_dict()
    rng = np.random.default_rng(metrics_module.BOOTSTRAP_SEED)
    return metrics_module.AnalysisInputs(
        task_meta=task_meta,
        weights=weights,
        models={},
        attempts={},
        occupation_order=occupations,
        category_index_by_occupation=np.array(
            [metrics_module.CATEGORIES.index(category[item]) for item in occupations], dtype=int
        ),
        weight_vector=np.array([weight[item] for item in occupations]),
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
    point: dict[str, float], uncertainty: dict[str, dict[str, float]]
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
            "estimate": float(point[field]),
            "bootstrap_mean": float(uncertainty[field]["bootstrap_mean"]),
            "bootstrap_sd": float(uncertainty[field]["bootstrap_sd"]),
            "ci95_low": float(uncertainty[field]["ci95_low"]),
            "ci95_high": float(uncertainty[field]["ci95_high"]),
        }
        for field in fields
    }


def calculate(input_root: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    metrics_module = load_metrics_module()
    task_meta, weights = metrics_module.load_task_metadata()
    energy, contributions, batches, energy_qa = load_energy_ledger(input_root)
    attempts, _, validation = load_amd_attempts(input_root, energy)
    prompt = build_prompt_frame(task_meta, attempts)
    inputs = bootstrap_inputs(metrics_module, task_meta, weights)

    final_source_by_global = {
        row["global_position"]: row["canonical_source_name"] for row in attempts
    }
    final_only_energy_j: dict[int, float] = defaultdict(float)
    for row in contributions:
        if row["source_name"] == final_source_by_global[row["global_position"]]:
            final_only_energy_j[row["global_position"]] += float(
                row["allocated_gpu_energy_joules"]
            )
    if set(final_only_energy_j) != set(range(880)):
        raise AssertionError("final-canonical-only sensitivity lacks output slots")
    final_only_attempts = [
        {
            **row,
            "energy_j": final_only_energy_j[row["global_position"]],
            "energy_mj": final_only_energy_j[row["global_position"]] / 1e6,
        }
        for row in attempts
    ]
    final_only_prompt = build_prompt_frame(task_meta, final_only_attempts)

    category_point, global_point, occupation_point = metrics_module.point_metrics(
        prompt,
        weights,
        threshold=1,
        baseline_hours_column="human_only_hours",
        wage_column="median_hourly_wage_usd",
    )
    category_uncertainty, global_uncertainty = metrics_module.bootstrap_metrics(
        inputs,
        prompt,
        threshold=1,
        baseline_hours_column="human_only_hours",
        wage_column="median_hourly_wage_usd",
        point_categories=category_point,
        point_global=global_point,
    )
    _, final_only_point, _ = metrics_module.point_metrics(
        final_only_prompt,
        weights,
        threshold=1,
        baseline_hours_column="human_only_hours",
        wage_column="median_hourly_wage_usd",
    )
    if not math.isclose(
        global_point["raw_energy_mj"], energy_qa["raw_total_gpu_energy_mj"], abs_tol=1e-9
    ):
        raise AssertionError("prompt energy does not reconcile to AMD batch measurements")

    category_ci = category_uncertainty.set_index("occupation_category")
    category_rows = []
    for row in category_point.to_dict(orient="records"):
        ci = category_ci.loc[row["occupation_category"]]
        for field in (
            "generated_value_usd",
            "hours_saved",
            "energy_mj",
            "value_per_mj",
            "hours_per_mj",
        ):
            row[f"{field}_ci95_low"] = float(ci[f"{field}_ci95_low"])
            row[f"{field}_ci95_high"] = float(ci[f"{field}_ci95_high"])
        category_rows.append(row)

    source_paths = [
        METRICS_MODULE_PATH,
        SOURCE_ROOT / "qwen_source_outputs.json",
        SOURCE_ROOT / "comparison_manifest.unblinded.jsonl",
        SOURCE_ROOT / "frozen_input_audit.json",
        JUDGE_ROOT / "results.output_level.jsonl",
        JUDGE_ROOT / "final_validation_report.json",
        *sorted((input_root / "batches").glob("*.jsonl")),
        *sorted((input_root / "manifests").glob("*.json")),
    ]
    report = {
        "energy_boundary": "GPU",
        "energy_components": ["gpu"],
        "model": "Qwen/Qwen3.6-27B",
        "hardware": "1×AMD Instinct MI350X",
        "benchmark": "GDPVal gold subset",
        "condition": "standard",
        "comparison_baseline": "expert-only reference deliverable",
        "judge": "models/gemini-3.1-flash-lite, source-blind, temperature 0",
        "success_definition": "agent win or tie",
        "credit_threshold": "at least 1 success in 4 attempts; value counted once",
        "valuation_basis": "recorded expert completion time (hours) × May 2024 OEWS median hourly wage",
        "weighting": (
            "GDPVal sector shares normalized across nine sectors, then within-sector "
            "May 2024 OEWS occupation wage-bill weights; occupation panels scaled by 44"
        ),
        "energy_accounting": (
            "Exact measured one-GPU batch energy for all 44 production batches and four "
            "targeted repair batches. Batch energy is allocated equally to physical "
            "attempts; failed/replaced attempts remain charged to their task slot."
        ),
        "bootstrap": {
            "unit": "distinct prompt within each of 44 occupations",
            "samples": metrics_module.BOOTSTRAP_SAMPLES,
            "seed": metrics_module.BOOTSTRAP_SEED,
            "interval": "percentile 95%",
        },
        "metrics": metric_payload(global_point, global_uncertainty),
        "sensitivity": {
            "final_canonical_outputs_only_excluding_discarded_attempts": {
                "interpretation": (
                    "Not the primary estimate; excludes energy from failed "
                    "or superseded physical attempts while retaining final output latency."
                ),
                "raw_gpu_energy_mj": float(final_only_point["raw_energy_mj"]),
                "gdp_weighted_gpu_energy_mj": float(final_only_point["energy_mj"]),
                "value_per_mj": float(final_only_point["value_per_mj"]),
                "hours_per_mj": float(final_only_point["hours_per_mj"]),
            },
            "failed_or_superseded_overhead_raw_gpu_energy_mj": float(
                global_point["raw_energy_mj"] - final_only_point["raw_energy_mj"]
            ),
        },
        "supporting": {
            "n_unique_prompts": 220,
            "canonical_output_slots": 880,
            "qualifying_prompts": int((prompt["success_count"] >= 1).sum()),
            "successful_attempts": int(prompt["success_count"].sum()),
            "attempt_win_or_tie_rate": float(global_point["attempt_win_or_tie_rate"]),
            "human_baseline_outcomes": dict(Counter(row["agent_outcome"] for row in attempts)),
            "raw_unweighted_generated_value_usd": float(
                global_point["raw_generated_value_usd"]
            ),
            "raw_unweighted_hours_saved": float(global_point["raw_hours_saved"]),
            **energy_qa,
        },
        "category_breakdown": category_rows,
        "occupation_breakdown": occupation_point.to_dict(orient="records"),
        "qa": {
            "passed": True,
            "paper_formula_reused": True,
            "pairwise_valid_verdicts": int(validation["unique_verdict_ids"]),
            "pairwise_unresolved_failures": int(validation["unresolved_failures"]),
            "canonical_rows": len(attempts),
            "numeric_scores": len(attempts),
            "complete_non_timeout_traces": sum(
                math.isfinite(row["latency_seconds"]) and row["latency_seconds"] > 0
                for row in attempts
            ),
            "energy_batch_to_slot_reconciled": True,
            "one_gpu_per_batch": True,
        },
        "sources": [
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in source_paths
        ],
    }
    frames = {
        "attempts": pd.DataFrame(attempts).sort_values("global_position"),
        "prompts": prompt.sort_values(["occupation", "task_id"]),
        "energy_contributions": pd.DataFrame(contributions).sort_values(
            ["global_position", "source_name", "batch_id", "query_id"]
        ),
        "batches": pd.DataFrame(batches).sort_values(
            ["source_name", "batch_index"]
        ),
        "categories": pd.DataFrame(category_rows),
        "occupations": occupation_point,
    }
    return report, frames




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    report, frames = calculate(args.input_root.absolute())
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "qwen36_27b_amd_mi350x_paper_five_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for name, frame in frames.items():
        frame.to_csv(args.output_root / f"{name}.csv", index=False)
    print(json.dumps({"output": str(report_path), "energy_boundary": report["energy_boundary"], "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
