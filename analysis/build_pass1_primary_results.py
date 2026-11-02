#!/usr/bin/env python3
"""Build Joules to GDP pass@1 numerical metrics.

Pass@1--pass@4 is a sensitivity analysis. Every other
comparison selects recorded repeat 0, credits a win or tie once, and charges
that run's allocated end-to-end energy.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from energy_boundary import attempt_energy


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs/metrics"
METRICS_MODULE = ROOT / "analysis/metrics.py"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260726


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_frame(name: str, frame: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLE_DIR / f"{name}.csv", index=False, float_format="%.12g")
    records = frame.replace({np.nan: None}).to_dict(orient="records")
    (TABLE_DIR / f"{name}.json").write_text(
        json.dumps(records, indent=2, sort_keys=True, default=float) + "\n",
        encoding="utf-8",
    )


def pass1_prompt(inputs: Any, model_id: str, baseline: str) -> pd.DataFrame:
    attempts = inputs.attempts[(model_id, baseline)].copy()
    attempts["repeat"] = attempts["repeat"].astype(int)
    attempts = attempts[attempts["repeat"].eq(0)].copy()
    if len(attempts) != 220 or attempts["task_id"].nunique() != 220:
        raise AssertionError(f"{model_id}/{baseline}: repeat-0 panel is incomplete")
    if "energy_mj" not in attempts:
        prompt_energy = inputs.models[model_id][
            ["task_id", "energy_all_four_runs_mj"]
        ].copy()
        prompt_energy["energy_mj"] = prompt_energy["energy_all_four_runs_mj"] / 4
        attempts = attempts.merge(
            prompt_energy[["task_id", "energy_mj"]],
            on="task_id",
            how="left",
            validate="one_to_one",
        )
    prompt = inputs.task_meta.merge(
        attempts[["task_id", "success", "latency_hours", "energy_mj"]],
        on="task_id",
        how="left",
        validate="one_to_one",
    )
    if prompt[["success", "latency_hours", "energy_mj"]].isna().any().any():
        raise AssertionError(f"{model_id}/{baseline}: pass@1 join is incomplete")
    prompt["success_count"] = prompt["success"].astype(int)
    prompt["mean_success_latency_hours"] = np.where(
        prompt["success"], prompt["latency_hours"], np.nan
    )
    prompt["energy_all_four_runs_mj"] = prompt["energy_mj"]
    return prompt


def success_interval(metrics_module: Any, inputs: Any, prompt: pd.DataFrame) -> dict[str, float]:
    matrix = metrics_module.prompt_matrix(prompt, inputs.occupation_order, "success_count")
    expanded = np.broadcast_to(matrix, (inputs.bootstrap_draws.shape[0],) + matrix.shape)
    sampled = np.take_along_axis(expanded, inputs.bootstrap_draws, axis=2).sum(axis=2)
    values = sampled.sum(axis=1) / 220
    return {
        "estimate": float(prompt["success_count"].mean()),
        **metrics_module.uncertainty(values),
    }


def analyze_prompt(
    metrics_module: Any,
    inputs: Any,
    prompt: pd.DataFrame,
    baseline_hours: str,
    wage: str = "median_hourly_wage_usd",
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame, dict[str, dict[str, float]]]:
    category, global_point, occupation = metrics_module.point_metrics(
        prompt, inputs.weights, 1, baseline_hours, wage
    )
    category_uncertainty, global_uncertainty = metrics_module.bootstrap_metrics(
        inputs,
        prompt,
        1,
        baseline_hours,
        wage,
        category,
        global_point,
    )
    global_point["attempts"] = 220
    global_point["attempt_win_or_tie_rate"] = float(prompt["success_count"].mean())
    global_uncertainty["attempt_win_or_tie_rate"] = success_interval(
        metrics_module, inputs, prompt
    )
    weight_lookup = inputs.weights.set_index("occupation")["gdp_structure_weight"]
    for category_name in category["occupation_category"]:
        mask = prompt["category"].eq(category_name)
        category_weight = float(
            inputs.weights.loc[
                inputs.weights["category"].eq(category_name), "gdp_structure_weight"
            ].sum()
        )
        probabilities = prompt.loc[mask, "occupation"].map(weight_lookup) / (
            5 * category_weight
        )
        index = category.index[category["occupation_category"].eq(category_name)][0]
        category.loc[index, "attempts"] = int(mask.sum())
        category.loc[index, "attempt_win_or_tie_rate"] = float(
            (probabilities.to_numpy() * prompt.loc[mask, "success_count"].to_numpy()).sum()
        )
    return category, global_point, occupation, global_uncertainty


def model_and_reference_tables(metrics_module: Any, inputs: Any) -> dict[str, pd.DataFrame]:
    model_ids = (
        "glm52_h200",
        "qwen36_27b",
        "gemma4_31b",
        "gpt_oss_120b",
        "gpt_oss_20b",
    )
    rows: list[dict[str, Any]] = []
    categories: dict[str, pd.DataFrame] = {}
    hardware_override = {
        "glm52_h200": "8×NVIDIA H200",
        "qwen36_27b": "8×NVIDIA H100",
        "gemma4_31b": "8×NVIDIA H100",
        "gpt_oss_120b": "8×NVIDIA H100",
        "gpt_oss_20b": "8×NVIDIA H100",
    }
    for model_id in model_ids:
        prompt = pass1_prompt(inputs, model_id, "human")
        category, _, _, uncertainty = analyze_prompt(
            metrics_module, inputs, prompt, "human_only_hours"
        )
        categories[model_id] = category
        row: dict[str, Any] = {
            "model": metrics_module.MODEL_CONFIGS[model_id]["display"],
            "system_label": metrics_module.MODEL_CONFIGS[model_id]["system"],
            "hardware": hardware_override[model_id],
            "n_unique_prompts": 220,
            "attempts": 220,
        }
        for field in (
            "attempt_win_or_tie_rate",
            "generated_value_usd",
            "energy_mj",
            "value_per_mj",
            "hours_saved",
            "hours_per_mj",
        ):
            for key, value in uncertainty[field].items():
                row[f"{field}_{key}"] = value
        rows.append(row)
    model_frame = pd.DataFrame(rows)
    for field, descending in (
        ("generated_value_usd_estimate", True),
        ("energy_mj_estimate", False),
        ("value_per_mj_estimate", True),
        ("hours_saved_estimate", True),
        ("hours_per_mj_estimate", True),
    ):
        model_frame[f"{field}_rank"] = metrics_module.ranks(
            model_frame[field].to_numpy(), descending
        ).astype(int)
    write_frame("five_model_global_system_comparison", model_frame)

    correlation_rows: list[dict[str, Any]] = []
    for first_index, first in enumerate(model_ids):
        for second in model_ids[first_index + 1 :]:
            for field in (
                "generated_value_usd",
                "hours_saved",
                "value_per_mj",
                "hours_per_mj",
            ):
                left = metrics_module.ranks(categories[first][field].to_numpy(), True)
                right = metrics_module.ranks(categories[second][field].to_numpy(), True)
                correlation_rows.append(
                    {
                        "model_a": metrics_module.MODEL_CONFIGS[first]["display"],
                        "model_b": metrics_module.MODEL_CONFIGS[second]["display"],
                        "metric": field,
                        "spearman_rank_correlation": float(np.corrcoef(left, right)[0, 1]),
                    }
                )
    write_frame("five_model_category_rank_correlations", pd.DataFrame(correlation_rows))

    reference_rows: list[dict[str, Any]] = []
    for baseline, hours_column in (
        ("expert-only", "human_only_hours"),
        ("expert-with-AI", "human_ai_hours"),
    ):
        key = "human" if baseline == "expert-only" else "human_ai"
        prompt = pass1_prompt(inputs, "glm52_h200", key)
        category, _, _, uncertainty = analyze_prompt(
            metrics_module, inputs, prompt, hours_column
        )
        if baseline == "expert-with-AI":
            category_uncertainty, _ = metrics_module.bootstrap_metrics(
                inputs,
                prompt,
                1,
                hours_column,
                "median_hourly_wage_usd",
                category,
                metrics_module.point_metrics(
                    prompt,
                    inputs.weights,
                    1,
                    hours_column,
                    "median_hourly_wage_usd",
                )[1],
            )
            combined = metrics_module.combine_point_uncertainty(category, category_uncertainty)
            combined["attempts"] = combined["n_unique_prompts"]
            write_frame("glm_human_ai_category_metrics_pass1", combined)
        row = {"condition": baseline, "n_unique_prompts": 220, "attempts": 220}
        for field in (
            "attempt_win_or_tie_rate",
            "generated_value_usd",
            "energy_mj",
            "value_per_mj",
            "hours_saved",
            "hours_per_mj",
        ):
            for label, value in uncertainty[field].items():
                row[f"{field}_{label}"] = value
        reference_rows.append(row)
    reference_frame = pd.DataFrame(reference_rows)
    write_frame("glm_reference_condition_pass1", reference_frame)
    return {"models": model_frame, "references": reference_frame, **categories}


def h100_attempts(task_meta: pd.DataFrame) -> pd.DataFrame:
    module = load_module("pass1_h100", ROOT / "cluster/h100_qwen_metrics.py")
    if not module.TRACE_PATH.is_file():
        # A frozen task-level ledger can replay pass@1 when the raw H100 archive
        # is unavailable. The archive is still required for fresh attribution.
        fallback = ROOT / "reports/energy_vs_cost_latency_20260922/one_h100_all_attempts.csv"
        frame = pd.read_csv(fallback)
        required = {"task_id", "repeat", "success", "latency_hours", "energy_mj"}
        if set(frame.columns) != required or len(frame) != 880:
            raise ValueError("H100 fallback ledger has the wrong schema or row count")
        if frame.task_id.nunique() != 220 or not frame.groupby("task_id").repeat.apply(lambda x: set(x) == {0, 1, 2, 3}).all():
            raise ValueError("H100 fallback ledger does not cover four attempts per task")
        return frame
    traces = module.read_jsonl(module.TRACE_PATH)
    results = module.read_jsonl(module.RESULT_PATH)
    batches: list[dict[str, Any]] = []
    for path in module.BATCH_PATHS:
        source_name = path.parents[3].name
        for row in module.read_jsonl(path):
            batches.append({"source_name": source_name, **row})
    batch_map = {
        (str(row["source_name"]), str(row["batch_id"])): row for row in batches
    }
    result_map = {
        (str(row["task_id"]), int(row["repeat"])): row
        for row in results
        if row["condition"] == "without_rubric" and row["baseline"] == "human"
    }
    rows = []
    for trace in traces:
        key = (str(trace["task_id"]), int(trace["repeat_index"]))
        batch = batch_map[(str(trace["source_name"]), str(trace["batch_id"]))]
        outcome = str(result_map[key]["agent_outcome"])
        rows.append(
            {
                "task_id": key[0],
                "repeat": key[1],
                "success": outcome in {"win", "tie"},
                "latency_hours": float(trace["total_wall_clock_s"]) / 3600,
                "energy_mj": float(batch["batch_gpu_energy_joules"]) / 20 / 1e6,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 880 or frame["task_id"].nunique() != 220:
        raise AssertionError("H100 attempt panel is incomplete")
    return frame


def hardware_attempts(task_meta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cluster = str(ROOT / "cluster")
    if cluster not in sys.path:
        sys.path.insert(0, cluster)

    b200 = load_module(
        "pass1_b200", ROOT / "cluster/b200_qwen_gdp_structured_value_capture.py"
    )
    frozen, _, _ = b200.frozen_economic_inputs()
    attempts, results, _, _, _ = b200.load_b200(
        ROOT
        / "outputs/019fde63-1f71-7461-a23f-2f017c2c95cd"
        / "qwen36_b200_gdp_structured_value_energy_20260815/inputs",
        frozen,
    )
    b200_frame = pd.DataFrame(attempts)
    b200_frame["success"] = [
        results["human"][(str(task), int(repeat))]["agent_outcome"] in {"win", "tie"}
        for task, repeat in zip(b200_frame["task_id"], b200_frame["repeat"])
    ]

    h200 = load_module(
        "pass1_h200", ROOT / "cluster/h200_qwen_metrics.py"
    )
    h200_input = (
        ROOT
        / "outputs/019fde63-1f71-7461-a23f-2f017c2c95cd"
        / "qwen36_h200_paper_five_metrics_20260815/inputs"
    )
    verdicts, _ = h200.load_pairwise_verdicts(
        h200_input, task_meta["task_id"].astype(str)
    )
    h200_rows, _, _ = h200.load_attempts(h200_input)
    h200_frame = pd.DataFrame(h200_rows)
    h200_frame["success"] = [
        verdicts["human"][(str(task), int(repeat))]["agent_outcome"] in {"win", "tie"}
        for task, repeat in zip(h200_frame["task_id"], h200_frame["repeat"])
    ]

    amd_frame = pd.read_csv(
        ROOT / "outputs/amd_qwen_mi350x_paper_five_metrics_20260815/attempts.csv"
    )
    frames = {
        "NVIDIA B200": b200_frame,
        "NVIDIA H200": h200_frame,
        "NVIDIA H100": h100_attempts(task_meta),
        "AMD MI350X": amd_frame,
    }
    for frame in frames.values():
        frame["task_id"] = frame["task_id"].astype(str)
        frame["repeat"] = frame["repeat"].astype(int)
        frame["success"] = frame["success"].astype(bool)
    boundary = os.environ.get("J2G_ENERGY_BOUNDARY", "total")
    if boundary == "total":
        energy = attempt_energy()
        profiles = {
            "NVIDIA B200": "qwen_b200",
            "NVIDIA H200": "qwen_h200",
            "NVIDIA H100": "qwen_h100",
            "AMD MI350X": "qwen_mi350x",
        }
        for hardware, profile in profiles.items():
            frame = frames[hardware]
            recorded = energy[energy.profile.eq(profile)][["task_id", "repeat", "total_energy_j"]]
            if len(frame) != len(recorded) or frame.duplicated(["task_id", "repeat"]).any():
                raise ValueError(f"{hardware}: incomplete hardware attempt coverage")
            frame = frame.drop(columns="energy_mj", errors="ignore").merge(
                recorded, on=["task_id", "repeat"], how="left", validate="one_to_one"
            )
            if frame.total_energy_j.isna().any():
                raise ValueError(f"{hardware}: missing measured GPU+CPU+DRAM energy")
            frame["energy_mj"] = frame.pop("total_energy_j") / 1e6
            frames[hardware] = frame
    elif boundary != "accelerator":
        raise ValueError(f"Unknown energy boundary: {boundary}")
    return frames


def hardware_tables(metrics_module: Any, inputs: Any, model_frame: pd.DataFrame) -> None:
    frames = hardware_attempts(inputs.task_meta)
    rows = []
    for hardware, frame in frames.items():
        selected = frame[frame["repeat"].eq(0)][
            ["task_id", "success", "latency_hours", "energy_mj"]
        ]
        prompt = inputs.task_meta.merge(
            selected, on="task_id", how="left", validate="one_to_one"
        )
        if len(prompt) != 220 or prompt[["success", "latency_hours", "energy_mj"]].isna().any().any():
            raise AssertionError(f"{hardware}: incomplete pass@1 panel")
        prompt["success_count"] = prompt["success"].astype(int)
        prompt["mean_success_latency_hours"] = np.where(
            prompt["success"], prompt["latency_hours"], np.nan
        )
        prompt["energy_all_four_runs_mj"] = prompt["energy_mj"]
        _, _, _, uncertainty = analyze_prompt(metrics_module, inputs, prompt, "human_only_hours")
        row = {"hardware": hardware, "n_unique_prompts": 220, "attempts": 220,
               "occupations_represented": 44, "categories_represented": 13}
        for field, stats in uncertainty.items():
            row[field] = stats["estimate"]
            for key, value in stats.items():
                row[f"{field}_{key}"] = value
        rows.append(row)
    write_frame("qwen36_hardware_full_benchmark", pd.DataFrame(rows))
    validation_path = TABLE_DIR / "validation_report.csv"
    if validation_path.exists():
        validation = pd.read_csv(validation_path)
        validation = validation[~validation["check"].str.startswith("hardware_")]
        hardware_checks = pd.DataFrame([
            {"check": "hardware_tasks_220_each", "passed": all(r["n_unique_prompts"] == 220 for r in rows)},
            {"check": "hardware_occupations_44_each", "passed": all(r["occupations_represented"] == 44 for r in rows)},
        ])
        write_frame("validation_report", pd.concat([validation, hardware_checks], ignore_index=True))

    h100 = frames["NVIDIA H100"].copy()
    h100_prompt = inputs.task_meta.merge(
        h100[h100["repeat"].eq(0)][
            ["task_id", "success", "latency_hours", "energy_mj"]
        ],
        on="task_id",
        how="left",
        validate="one_to_one",
    )
    h100_prompt["success_count"] = h100_prompt["success"].astype(int)
    h100_prompt["mean_success_latency_hours"] = np.where(
        h100_prompt["success"], h100_prompt["latency_hours"], np.nan
    )
    h100_prompt["energy_all_four_runs_mj"] = h100_prompt["energy_mj"]
    _, _, _, uncertainty = analyze_prompt(
        metrics_module, inputs, h100_prompt, "human_only_hours"
    )
    one_rows = []
    for field in (
        "generated_value_usd",
        "hours_saved",
        "energy_mj",
        "value_per_mj",
        "hours_per_mj",
    ):
        one_rows.append({"metric": field, **uncertainty[field]})
    write_frame("qwen36_one_h100_pass1", pd.DataFrame(one_rows))


def wage_and_three_way_tables(metrics_module: Any, inputs: Any) -> None:
    human_prompt = pass1_prompt(inputs, "glm52_h200", "human")
    category, global_frame = metrics_module.wage_robustness(inputs, human_prompt)
    category["attempts"] = category["n_unique_prompts"]
    write_frame("wage_basis_category_robustness", category)
    write_frame("wage_basis_global_robustness", global_frame)

    prompt = pass1_prompt(inputs, "glm52_h200", "human_ai")
    draws = inputs.bootstrap_draws
    weight = inputs.weight_vector
    success = metrics_module.prompt_matrix(prompt, inputs.occupation_order, "success_count")
    latency = np.nan_to_num(
        metrics_module.prompt_matrix(
            prompt, inputs.occupation_order, "mean_success_latency_hours"
        ),
        nan=0.0,
    )
    human = metrics_module.prompt_matrix(prompt, inputs.occupation_order, "human_only_hours")
    human_ai = metrics_module.prompt_matrix(prompt, inputs.occupation_order, "human_ai_hours")
    energy = metrics_module.prompt_matrix(
        prompt, inputs.occupation_order, "energy_all_four_runs_mj"
    )

    def sampled(array: np.ndarray) -> np.ndarray:
        expanded = np.broadcast_to(array, (draws.shape[0],) + array.shape)
        return np.take_along_axis(expanded, draws, axis=2).sum(axis=2)

    credit = success >= 1
    transitions = {
        "Expert-only → expert-with-AI": (credit * (human - human_ai), credit * human, False),
        "Expert-with-AI → AI system": (credit * (human_ai - latency), credit * human_ai, True),
        "Expert-only → AI system": (credit * (human - latency), credit * human, True),
    }
    energy_point = float(np.sum(weight[:, None] * energy) / 5)
    energy_boot = (sampled(energy) * weight[None, :]).sum(axis=1) / 5
    rows = []
    for transition, (saved, denominator, has_energy) in transitions.items():
        point = float(np.sum(weight[:, None] * saved) / 5)
        denominator_point = float(np.sum(weight[:, None] * denominator) / 5)
        saved_boot = (sampled(saved) * weight[None, :]).sum(axis=1) / 5
        denominator_boot = (sampled(denominator) * weight[None, :]).sum(axis=1) / 5
        row = {
            "policy": "pass@1",
            "transition": transition,
            "n_unique_prompts": 220,
            "attempts": 220,
            "hours_saved_estimate": point,
            **{
                f"hours_saved_{key}": value
                for key, value in metrics_module.uncertainty(saved_boot).items()
            },
            "percent_saved_estimate": 100 * point / denominator_point,
            **{
                f"percent_saved_{key}": value
                for key, value in metrics_module.uncertainty(
                    100 * saved_boot / denominator_boot
                ).items()
            },
            "energy_mj_exact": energy_point if has_energy else math.nan,
            "hours_per_mj_estimate": point / energy_point if has_energy else math.nan,
        }
        if has_energy:
            for key, value in metrics_module.uncertainty(saved_boot / energy_boot).items():
                row[f"hours_per_mj_{key}"] = value
        else:
            for key in ("bootstrap_mean", "bootstrap_sd", "ci95_low", "ci95_high"):
                row[f"hours_per_mj_{key}"] = math.nan
        rows.append(row)
    write_frame("glm_three_way_time_savings_pass1", pd.DataFrame(rows))


def main() -> None:
    metrics_module = load_module("pass1_metrics", METRICS_MODULE)
    inputs = metrics_module.load_inputs(BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
    built = model_and_reference_tables(metrics_module, inputs)
    hardware_tables(metrics_module, inputs, built["models"])
    wage_and_three_way_tables(metrics_module, inputs)
    # Active-request GLM energy is merged in build_net_metrics.py. The
    # older energy-only patcher enforces byte identity for every other table,
    # which is intentionally incompatible with this GDP-employment reweighting.
    print("wrote pass@1 primary result tables")


if __name__ == "__main__":
    main()
