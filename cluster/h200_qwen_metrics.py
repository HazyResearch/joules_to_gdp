#!/usr/bin/env python3
"""Calculate five efficiency metrics for Qwen3.6-27B on one H200.

The shared implementation supplies the economic weighting, valuation,
hours-saved, and bootstrap definitions. This adapter supplies only the one-H200
run's canonical attempts, measured batch energy, and the independent blinded
Gemini pairwise verdicts recovered from the two successful provider jobs.
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
from typing import Any, Iterable

import numpy as np
import pandas as pd

import gdpval_pairwise_eval as pairwise


ROOT = Path(__file__).resolve().parents[1]
METRICS_MODULE_PATH = ROOT / "analysis/metrics.py"
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs/019fde63-1f71-7461-a23f-2f017c2c95cd"
    / "qwen36_h200_paper_five_metrics_20260815"
)
DEFAULT_INPUT_DIR = DEFAULT_OUTPUT_DIR / "inputs"

SCHEMA_VERSION = "gdpval-pairwise-v1"
STUDY_ID = "qwen36_27b_norubric_h200_20260726T045906Z"
CONDITION = "without_rubric"
BASELINES = ("human", "human_ai")
INPUT_FILES = (
    "canonical_traces.jsonl",
    "attempt_000_start_0000.batches.jsonl",
    "attempt_003_start_0860.batches.jsonl",
    "google_pilot.operation.json",
    "google_full.operation.json",
    "pairwise_final_validation_report.json",
    "pairwise_results_summary.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc


def load_metrics_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "joules_to_gdp_paper_analysis_h200", METRICS_MODULE_PATH
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


def comparison_map(task_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    mappings: dict[str, dict[str, Any]] = {}
    for task_id in sorted(str(value) for value in task_ids):
        for baseline in BASELINES:
            for repeat in range(4):
                comparison_id = (
                    f"{STUDY_ID}__{CONDITION}__{baseline}__"
                    f"{task_id}__rep{repeat:02d}"
                )
                blind_id = pairwise.blind_id(comparison_id)
                if blind_id in mappings:
                    raise AssertionError(f"duplicate blind ID: {blind_id}")
                mappings[blind_id] = {
                    "blind_id": blind_id,
                    "comparison_id": comparison_id,
                    "task_id": task_id,
                    "baseline": baseline,
                    "repeat": repeat,
                    "agent_position": pairwise.agent_position(
                        task_id, CONDITION, baseline, repeat
                    ),
                }
    if len(mappings) != 1760:
        raise AssertionError(f"expected 1760 comparison mappings, got {len(mappings)}")
    return mappings


def provider_items(operation_path: Path) -> list[dict[str, Any]]:
    operation = json.loads(operation_path.read_text(encoding="utf-8"))
    if not operation.get("done"):
        raise AssertionError(f"provider operation is not done: {operation_path}")
    state = str(operation.get("metadata", {}).get("state") or "")
    if state != "BATCH_STATE_SUCCEEDED":
        raise AssertionError(f"provider operation did not succeed: {state}")
    return list(
        operation["response"]["inlinedResponses"]["inlinedResponses"]
    )


def load_pairwise_verdicts(
    input_dir: Path,
    task_ids: Iterable[str],
) -> tuple[dict[str, dict[tuple[str, int], dict[str, Any]]], dict[str, Any]]:
    mappings = comparison_map(task_ids)
    items: list[dict[str, Any]] = []
    job_counts: dict[str, int] = {}
    for name in ("google_pilot.operation.json", "google_full.operation.json"):
        job_items = provider_items(input_dir / name)
        job_counts[name] = len(job_items)
        items.extend(job_items)
    if job_counts != {
        "google_pilot.operation.json": 24,
        "google_full.operation.json": 1736,
    }:
        raise AssertionError(f"unexpected provider job counts: {job_counts}")

    actual_ids = [str(row.get("metadata", {}).get("custom_id") or "") for row in items]
    if len(actual_ids) != 1760 or len(set(actual_ids)) != 1760:
        raise AssertionError("provider responses are not 1760 unique custom IDs")
    if set(actual_ids) != set(mappings):
        missing = sorted(set(mappings) - set(actual_ids))[:5]
        extra = sorted(set(actual_ids) - set(mappings))[:5]
        raise AssertionError(f"provider/mapping mismatch; missing={missing}, extra={extra}")

    results: dict[str, dict[tuple[str, int], dict[str, Any]]] = {
        baseline: {} for baseline in BASELINES
    }
    outcome_counts = {baseline: Counter() for baseline in BASELINES}
    model_versions: Counter[str] = Counter()
    response_ids: set[str] = set()
    for item in items:
        blind_id = str(item["metadata"]["custom_id"])
        mapping = mappings[blind_id]
        response = item.get("response") or {}
        candidates = response.get("candidates") or []
        if len(candidates) != 1:
            raise AssertionError(f"{blind_id}: expected one candidate response")
        parts = candidates[0].get("content", {}).get("parts") or []
        text = "\n".join(str(part.get("text") or "") for part in parts)
        verdict = pairwise.extract_json_from_text(text)
        winner = str(verdict.get("winner") or "").strip()
        if winner.lower() == "tie":
            winner = "tie"
        elif winner.upper() in {"A", "B"}:
            winner = winner.upper()
        else:
            raise AssertionError(f"{blind_id}: invalid winner {winner!r}")
        agent_position = str(mapping["agent_position"])
        outcome = (
            "tie"
            if winner == "tie"
            else ("win" if winner == agent_position else "loss")
        )
        baseline = str(mapping["baseline"])
        key = (str(mapping["task_id"]), int(mapping["repeat"]))
        if key in results[baseline]:
            raise AssertionError(f"duplicate result: {baseline}/{key}")
        results[baseline][key] = {
            **mapping,
            "winner": winner,
            "agent_outcome": outcome,
            "model_version": response.get("modelVersion"),
            "response_id": response.get("responseId"),
            "usage_metadata": response.get("usageMetadata") or {},
        }
        outcome_counts[baseline][outcome] += 1
        model_versions[str(response.get("modelVersion") or "")] += 1
        response_id = str(response.get("responseId") or "")
        if not response_id or response_id in response_ids:
            raise AssertionError(f"missing or duplicate provider response ID: {blind_id}")
        response_ids.add(response_id)

    for baseline in BASELINES:
        if len(results[baseline]) != 880:
            raise AssertionError(f"{baseline}: expected 880 verdicts")
        if sum(outcome_counts[baseline].values()) != len(results[baseline]):
            raise AssertionError(f"{baseline}: incomplete outcome accounting")

    validation = json.loads(
        (input_dir / "pairwise_final_validation_report.json").read_text()
    )
    if not validation.get("complete"):
        raise AssertionError("frozen pairwise validation is not complete")
    if validation.get("unique_verdict_ids") != 1760:
        raise AssertionError("frozen pairwise verdict count changed")
    if validation.get("unresolved_failures") != 0:
        raise AssertionError("frozen pairwise report has unresolved failures")

    summary = json.loads((input_dir / "pairwise_results_summary.json").read_text())
    for baseline in BASELINES:
        headline = summary["headline"][f"without_rubric_vs_{baseline}"]
        expected = outcome_counts[baseline]
        if (
            int(headline["wins"]) != expected["win"]
            or int(headline["ties"]) != expected["tie"]
            or int(headline["losses"]) != expected["loss"]
        ):
            raise AssertionError(f"{baseline}: provider verdicts differ from report")

    return results, {
        "job_counts": job_counts,
        "outcome_counts": {
            baseline: dict(outcome_counts[baseline]) for baseline in BASELINES
        },
        "model_versions": dict(model_versions),
        "unique_response_ids": len(response_ids),
        "validation": validation,
    }


def load_attempts(
    input_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    batch_sources = {
        "attempt_000_start_0000": input_dir / "attempt_000_start_0000.batches.jsonl",
        "attempt_003_start_0860": input_dir / "attempt_003_start_0860.batches.jsonl",
    }
    batches: list[dict[str, Any]] = []
    batch_map: dict[tuple[str, int], dict[str, Any]] = {}
    for source_name, path in batch_sources.items():
        source_rows = list(iter_jsonl(path))
        for row in source_rows:
            key = (source_name, int(row["batch_index"]))
            if key in batch_map:
                raise AssertionError(f"duplicate batch key: {key}")
            if int(row["batch_size"]) != 20 or len(row["query_ids"]) != 20:
                raise AssertionError(f"{key}: batch is not exactly 20 queries")
            device_ids = [int(value) for value in row.get("telemetry_device_ids") or []]
            if device_ids != [0]:
                raise AssertionError(f"{key}: expected one H200 device, got {device_ids}")
            row = dict(row)
            row["source_name"] = source_name
            batch_map[key] = row
            batches.append(row)
    if len(batches) != 44:
        raise AssertionError(f"expected 44 production batches, got {len(batches)}")

    traces = list(iter_jsonl(input_dir / "canonical_traces.jsonl"))
    if len(traces) != 880:
        raise AssertionError(f"expected 880 canonical traces, got {len(traces)}")
    attempts: list[dict[str, Any]] = []
    trace_keys: set[tuple[str, int]] = set()
    source_counts: Counter[str] = Counter()
    completion_counts: Counter[str] = Counter()
    unscorable_counts: Counter[str] = Counter()
    for trace in traces:
        task_id = str(trace["task_id"])
        repeat = int(trace["repeat_index"])
        trace_key = (task_id, repeat)
        if trace_key in trace_keys:
            raise AssertionError(f"duplicate canonical attempt: {trace_key}")
        trace_keys.add(trace_key)
        completion_counts["completed" if trace.get("completed") else "not_completed"] += 1
        completion_counts["timed_out" if trace.get("timed_out") else "not_timed_out"] += 1
        if trace.get("unscorable_reason"):
            unscorable_counts[str(trace["unscorable_reason"])] += 1
        source_name = str(trace["source_name"])
        batch_key = (source_name, int(trace["batch_index"]))
        if batch_key not in batch_map:
            raise AssertionError(f"canonical trace missing source batch: {trace_key}")
        batch = batch_map[batch_key]
        source_query_id = str(trace["source_query_id"])
        if source_query_id not in {str(value) for value in batch["query_ids"]}:
            raise AssertionError(f"{trace_key}: source query absent from batch")
        attempts.append(
            {
                "task_id": task_id,
                "repeat": repeat,
                "latency_hours": float(trace["total_wall_clock_s"]) / 3600.0,
                "energy_mj": float(batch["batch_gpu_energy_joules"])
                / float(batch["batch_size"])
                / 1_000_000.0,
                "source_name": source_name,
                "source_query_id": source_query_id,
                "batch_index": int(trace["batch_index"]),
            }
        )
        source_counts[source_name] += 1
    if sum(source_counts.values()) != len(attempts):
        raise AssertionError("H200 source attribution does not cover attempts")
    if completion_counts["completed"] + completion_counts["not_completed"] != len(attempts):
        raise AssertionError("H200 completion accounting does not cover attempts")
    if completion_counts["timed_out"] + completion_counts["not_timed_out"] != len(attempts):
        raise AssertionError("H200 timeout accounting does not cover attempts")
    if len({row["task_id"] for row in attempts}) != 220:
        raise AssertionError("canonical panel does not contain 220 tasks")
    if Counter(row["task_id"] for row in attempts) != Counter(
        {task_id: 4 for task_id in {row["task_id"] for row in attempts}}
    ):
        raise AssertionError("canonical panel is not four repeats per task")

    raw_energy_mj = sum(float(row["batch_gpu_energy_joules"]) for row in batches) / 1e6
    allocated_energy_mj = sum(row["energy_mj"] for row in attempts)
    if not math.isclose(raw_energy_mj, allocated_energy_mj, abs_tol=1e-12):
        raise AssertionError("allocated attempt energy did not reconcile to batch energy")

    return attempts, batches, {
        "raw_measured_gpu_energy_mj": raw_energy_mj,
        "raw_measured_gpu_energy_kwh": raw_energy_mj / 3.6,
        "source_counts": dict(source_counts),
        "completion_counts": dict(completion_counts),
        "unscorable_counts": dict(unscorable_counts),
    }


def build_prompt_frame(
    task_meta: pd.DataFrame,
    attempts: list[dict[str, Any]],
    human_results: dict[tuple[str, int], dict[str, Any]],
) -> pd.DataFrame:
    attempts_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        attempts_by_task[str(attempt["task_id"])].append(attempt)

    rows: list[dict[str, Any]] = []
    for task_id, task_attempts in attempts_by_task.items():
        if len(task_attempts) != 4:
            raise AssertionError(f"{task_id}: expected four H200 attempts")
        successful = [
            attempt
            for attempt in task_attempts
            if human_results[(task_id, int(attempt["repeat"]))]["agent_outcome"]
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
        raise AssertionError("H200 prompt metadata did not join to all 220 tasks")
    if not prompt.groupby("occupation").size().eq(5).all():
        raise AssertionError("H200 prompt panel is not five tasks per occupation")
    return prompt


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
    missing = [name for name in INPUT_FILES if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing H200 inputs: {missing}")
    metrics_module = load_metrics_module()
    task_meta, weights = metrics_module.load_task_metadata()
    results_by_baseline, pairwise_qa = load_pairwise_verdicts(
        input_dir, task_meta["task_id"].astype(str)
    )
    attempts, batches, attempt_qa = load_attempts(input_dir)
    occupation_by_task = task_meta.set_index("task_id")["occupation"].astype(str).to_dict()
    batch_occupations: dict[tuple[str, int], set[str]] = defaultdict(set)
    batch_attempt_counts: Counter[tuple[str, int]] = Counter()
    for attempt in attempts:
        batch_key = (str(attempt["source_name"]), int(attempt["batch_index"]))
        batch_occupations[batch_key].add(occupation_by_task[str(attempt["task_id"])])
        batch_attempt_counts[batch_key] += 1
    if len(batch_occupations) != 44 or any(
        len(occupations) != 1 for occupations in batch_occupations.values()
    ):
        raise AssertionError("H200 production batches are not 44 occupation-pure batches")
    if set(batch_attempt_counts.values()) != {20}:
        raise AssertionError("H200 occupation-pure batches do not each contain 20 attempts")
    represented_occupations = Counter(
        next(iter(occupations)) for occupations in batch_occupations.values()
    )
    if len(represented_occupations) != 44 or set(represented_occupations.values()) != {1}:
        raise AssertionError("H200 batches do not map one-to-one to 44 occupations")
    prompt = build_prompt_frame(task_meta, attempts, results_by_baseline["human"])
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
            row[f"{field}_ci95_low"] = float(uncertainty_row[f"{field}_ci95_low"])
            row[f"{field}_ci95_high"] = float(uncertainty_row[f"{field}_ci95_high"])
        category_rows.append(row)

    raw_energy_mj = float(attempt_qa["raw_measured_gpu_energy_mj"])
    if not math.isclose(raw_energy_mj, global_point["raw_energy_mj"], abs_tol=1e-12):
        raise AssertionError("raw H200 energy did not reconcile through the analysis module")
    if not math.isclose(
        global_point["value_per_mj"],
        global_point["generated_value_usd"] / global_point["energy_mj"],
        abs_tol=1e-12,
    ):
        raise AssertionError("value/MJ ratio did not reconcile")
    if not math.isclose(
        global_point["hours_per_mj"],
        global_point["hours_saved"] / global_point["energy_mj"],
        abs_tol=1e-12,
    ):
        raise AssertionError("hours/MJ ratio did not reconcile")

    return {
        "model": "Qwen3.6-27B",
        "hardware": "1×NVIDIA H200",
        "energy_boundary": "GPU",
        "energy_components": ["gpu"],
        "run_id": (
            "20260726T045906Z_qwen36_27b_gdpval_occ_repeat4_full_norubric_"
            "h200_1g_128k_vllm019"
        ),
        "benchmark": "GDPVal gold subset",
        "condition": "standard",
        "comparison_baseline": "expert-only reference deliverable",
        "judge": "models/gemini-3.1-flash-lite",
        "success_definition": "source-blind judgment is win or tie",
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
            "repeats_retained_together": True,
        },
        "metrics": metric_payload(global_point, global_uncertainty),
        "supporting": {
            "n_unique_prompts": 220,
            "attempts": 880,
            "production_batches": len(batches),
            "qualifying_prompts": int((prompt["success_count"] >= 1).sum()),
            "successful_attempts": int(prompt["success_count"].sum()),
            "attempt_win_or_tie_rate": float(global_point["attempt_win_or_tie_rate"]),
            "raw_measured_gpu_energy_mj": raw_energy_mj,
            "raw_measured_gpu_energy_kwh": raw_energy_mj / 3.6,
            "raw_unweighted_generated_value_usd": float(
                global_point["raw_generated_value_usd"]
            ),
            "raw_unweighted_hours_saved": float(global_point["raw_hours_saved"]),
        },
        "category_breakdown": category_rows,
        "occupation_breakdown": occupation_point.to_dict(orient="records"),
        "qa": {
            "passed": True,
            "canonical_attempts": len(attempts),
            "completion_counts": attempt_qa["completion_counts"],
            "unscorable_counts": attempt_qa["unscorable_counts"],
            "production_batches": len(batches),
            "occupation_pure_batches": len(batch_occupations),
            "single_gpu_device_id": 0,
            "pairwise_valid_verdicts": pairwise_qa["validation"][
                "unique_verdict_ids"
            ],
            "pairwise_unresolved_failures": pairwise_qa["validation"][
                "unresolved_failures"
            ],
            "pairwise_outcomes": pairwise_qa["outcome_counts"],
            "provider_job_counts": pairwise_qa["job_counts"],
            "provider_unique_response_ids": pairwise_qa["unique_response_ids"],
            "provider_model_versions": pairwise_qa["model_versions"],
            "source_energy_reconciled": True,
            "ratios_reconciled": True,
        },
        "sources": {
            "paper_analysis": {
                "path": str(METRICS_MODULE_PATH.relative_to(ROOT)),
                "sha256": sha256(METRICS_MODULE_PATH),
            },
            "h200_inputs": {
                name: {
                    "path": str((input_dir / name).relative_to(ROOT)),
                    "sha256": sha256(input_dir / name),
                }
                for name in INPUT_FILES
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = calculate(args.input_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "qwen36_h200_paper_five_metrics_20260815.json"
    category_path = args.output_dir / "qwen36_h200_paper_category_breakdown_20260815.csv"
    occupation_path = (
        args.output_dir / "qwen36_h200_paper_occupation_breakdown_20260815.csv"
    )
    metrics_path = args.output_dir / "qwen36_h200_paper_five_metrics_20260815.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    metric_units = {
        "generated_value_usd": "USD",
        "hours_saved": "hours",
        "energy_mj": "MJ",
        "value_per_mj": "USD/MJ",
        "hours_per_mj": "hours/MJ",
    }
    pd.DataFrame(
        [
            {"metric": name, "unit": metric_units[name], **values}
            for name, values in report["metrics"].items()
        ]
    ).to_csv(metrics_path, index=False)
    pd.DataFrame(report["category_breakdown"]).to_csv(category_path, index=False)
    pd.DataFrame(report["occupation_breakdown"]).to_csv(occupation_path, index=False)
    print(json.dumps({"output": str(json_path), "energy_boundary": report["energy_boundary"], **report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
