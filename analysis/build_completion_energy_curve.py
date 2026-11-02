#!/usr/bin/env python3
"""Estimate cumulative batch energy at request-completion milestones.

Each batch's measured total is allocated over time using constant average
power. Intermediate points are estimates; the 100% endpoint is measured.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from build_pass1_primary_results import ROOT, METRICS_MODULE, load_module
from energy_boundary import INPUT_DIR, attempt_energy

OUT = ROOT / "reports/glm_energy_allocation_sensitivity_20260919"
PERCENTILES = (20, 30, 40, 50, 60, 70, 80, 90, 95, 100)


def main() -> None:
    metrics_module = load_module("completion_energy_metrics", METRICS_MODULE)
    _, weights = metrics_module.load_task_metadata()
    query = pd.read_csv(metrics_module.SOURCE_PATHS["glm_query"]).query("dataset == 'GDPVal'")
    batches = pd.read_csv(metrics_module.SOURCE_PATHS["glm_batch"]).query("dataset == 'GDPVal'")
    if len(query) != 880 or len(batches) != 44:
        raise ValueError("Expected 880 requests in 44 measured batches")
    boundary = os.environ.get("J2G_ENERGY_BOUNDARY", "total")
    if boundary == "total":
        attempt_energy()  # validates the complete ledger and source hashes
        components = pd.read_csv(INPUT_DIR / "energy_component_batches.csv")
        components = components[components.profile.eq("glm52_h200")]
        if components.batch_key.duplicated().any() or set(components.batch_key) != set(batches.batch_key):
            raise ValueError("GLM component batches do not match recorded completion batches")
        batches = batches.merge(
            components[["batch_key", "batch_start_s", "batch_end_s", "gpu_energy_j", "cpu_energy_j", "dram_energy_j"]],
            on="batch_key", validate="one_to_one",
        )
        if not np.allclose(batches.prompt_start_time_s, batches.batch_start_s, rtol=0, atol=1e-4) or not np.allclose(
            batches.prompt_end_time_s, batches.batch_end_s, rtol=0, atol=1e-4
        ):
            raise ValueError("GLM energy component windows differ from recorded prompt windows")
        batches["measured_energy_j"] = batches[["gpu_energy_j", "cpu_energy_j", "dram_energy_j"]].sum(axis=1)
    elif boundary == "accelerator":
        batches["measured_energy_j"] = batches.gpu_energy_joules
    else:
        raise ValueError(f"Unknown energy boundary: {boundary}")
    records = []
    for batch in batches.itertuples():
        requests = query[query.batch_key == batch.batch_key]
        if len(requests) != 20:
            raise ValueError(f"Incomplete batch: {batch.batch_key}")
        end_times = np.sort(requests.prompt_end_time_s.to_numpy(dtype=float))
        duration = float(batch.prompt_end_time_s - batch.prompt_start_time_s)
        if duration <= 0 or end_times[-1] > batch.prompt_end_time_s + 1e-4:
            raise ValueError(f"Invalid timing boundary: {batch.batch_key}")
        for percentile in PERCENTILES:
            count = percentile * 20 // 100
            elapsed = duration if percentile == 100 else max(0.0, min(duration, float(end_times[count - 1] - batch.prompt_start_time_s)))
            records.append({
                "batch_key": batch.batch_key,
                "occupation": batch.occupation,
                "percent_queries_terminated": percentile,
                "estimated_energy_consumed_j": float(batch.measured_energy_j) * elapsed / duration,
                "measured_batch_energy_j": float(batch.measured_energy_j),
                "elapsed_seconds": elapsed,
            })
    frame = pd.DataFrame(records).merge(
        weights[["occupation", "category", "gdp_structure_weight"]],
        on="occupation", validate="many_to_one"
    )
    summaries = []
    groups = [("ALL_UNWEIGHTED", frame), ("ALL_GDP_WEIGHTED", frame)] + list(frame.groupby("category"))
    for category, group in groups:
        for percentile, part in group.groupby("percent_queries_terminated"):
            w = np.ones(len(part)) if category == "ALL_UNWEIGHTED" else part.gdp_structure_weight.to_numpy()
            consumed = float(w @ part.estimated_energy_consumed_j.to_numpy())
            total = float(w @ part.measured_batch_energy_j.to_numpy())
            summaries.append({
                "category": category,
                "percent_queries_terminated": int(percentile),
                "estimated_cumulative_energy_mj": consumed / 1e6,
                "reference_full_energy_mj": total / 1e6,
                "estimated_cumulative_energy_pct": 100 * consumed / total,
                "estimated_energy_remaining_pct": 100 * (1 - consumed / total),
                "median_elapsed_minutes": float(part.elapsed_seconds.median() / 60),
                "measurement_status": "constant-power estimate from measured batch total",
                "energy_boundary": "GPU+CPU+DRAM" if boundary == "total" else "GPU",
            })
    result = pd.DataFrame(summaries).sort_values(["category", "percent_queries_terminated"])
    result["estimated_interval_energy_pct"] = result.groupby("category").estimated_cumulative_energy_pct.diff().fillna(result.estimated_cumulative_energy_pct)
    endpoint = result[result.percent_queries_terminated.eq(100)]
    if not np.allclose(endpoint.estimated_cumulative_energy_pct, 100, rtol=0, atol=1e-3):
        raise ValueError("Completion curves do not reconcile with batch energy")
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "completion_percentile_summary_estimated.csv", index=False, float_format="%.15g")
    print(f"Built completion curves for {len(batches)} measured batches")


if __name__ == "__main__":
    main()
