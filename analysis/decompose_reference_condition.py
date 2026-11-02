"""Separate reference judgments from reference-time valuation.

Read-only with respect to original inputs. This calculates
pass@1 endpoints and the paired, occupation-stratified
task-composition bootstrap, then evaluates the two missing factorial cells.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/paper_net_metrics_20260919"
OUT = ROOT / "reports/reference_condition_decomposition_20260920"
WEIGHTS = ROOT / "outputs/paper_gdp_employment_weights/occupation_weights.csv"
SAMPLES, SEED = 10_000, 20260726
NAMES = {"E": "Expert-only", "A": "Expert-with-AI"}


def interval(point, samples):
    lo, hi = np.quantile(samples, [.025, .975])
    return {"estimate": float(point), "ci_low": float(lo), "ci_high": float(hi)}


def main():
    paths = [SOURCE / f"glm52_h200_{b}_tasks.csv" for b in ("human", "human_ai")]
    frames = [pd.read_csv(p).sort_values(["occupation", "task_id"]).reset_index(drop=True)
              for p in paths]
    h, a = frames
    assert len(h) == len(a) == 220
    assert h.task_id.nunique() == a.task_id.nunique() == 220
    assert h.groupby("occupation").size().eq(5).all()
    shared = ["task_id", "baseline_task_id", "occupation", "category", "human_only_hours",
              "human_ai_hours", "median_hourly_wage_usd", "energy_mj", "rental_cost_usd"]
    pd.testing.assert_frame_equal(h[shared], a[shared])
    both = h.credit & a.credit
    np.testing.assert_allclose(h.loc[both, "mean_latency"], a.loc[both, "mean_latency"])
    for f in frames:
        assert f.credit.dtype == bool
        assert (f.success_count == f.credit.astype(int)).all()
        assert f.loc[f.credit, "mean_latency"].notna().all()

    verdict_path = ROOT / "reports/gdpval_pairwise_gemini31_flash_lite_20260716/results.output_level.jsonl"
    verdicts = pd.read_json(verdict_path, lines=True)
    query_path = ROOT / "outputs/019f86a0-e5e6-7e90-8578-b08ddb4385fb/glm52_no_rubric_query_ledger.csv"
    query = pd.read_csv(query_path)
    query = query[query.dataset.eq("GDPVal") & query.repeat_index.eq(0)].set_index("judge_task_id")
    assert len(query) == 220 and query.index.is_unique
    for f, baseline in zip(frames, ["human", "human_ai"]):
        selected = verdicts[verdicts.condition.eq("without_rubric") &
                            verdicts.baseline.eq(baseline) & verdicts.repeat.eq(0)]
        assert len(selected) == 220 and selected.task_id.is_unique
        expected = selected.set_index("task_id").loc[f.task_id, "agent_outcome"].isin(["win", "tie"])
        np.testing.assert_array_equal(f.credit.to_numpy(), expected.to_numpy())
        actual_latency = query.loc[f.task_id, "agent_latency_hours"].to_numpy()
        np.testing.assert_allclose(f.loc[f.credit, "mean_latency"], actual_latency[f.credit])

    weights = pd.read_csv(WEIGHTS)
    occ = sorted(h.occupation.unique())
    w = weights.set_index("occupation").loc[occ, "gdp_structure_weight"].to_numpy()
    np.testing.assert_allclose(w.sum(), 1., atol=1e-12)
    task_w = np.repeat(w / 5, 5)
    draws = np.random.default_rng(SEED).integers(0, 5, size=(SAMPLES, 44, 5), dtype=np.int16)

    def aggregate(x):
        x = np.asarray(x, dtype=float)
        matrix = x.reshape(44, 5)
        sampled = np.take_along_axis(np.broadcast_to(matrix, (SAMPLES, 44, 5)), draws, axis=2)
        return float(task_w @ x), sampled.mean(axis=2) @ w

    def unweighted(x):
        x = np.asarray(x, dtype=float).reshape(44, 5)
        samples = np.take_along_axis(np.broadcast_to(x, (SAMPLES, 44, 5)), draws, axis=2)
        return float(x.mean()), samples.mean(axis=(1, 2))

    energy, energy_boot = aggregate(h.energy_mj)
    cost, cost_boot = aggregate(h.rental_cost_usd)
    hours = {"E": h.human_only_hours, "A": h.human_ai_hours}
    point, boot, cells = {}, {}, {}
    for decision, f in zip(["E", "A"], frames):
        for value, ref_hours in hours.items():
            key = decision + value
            value_usd = f.credit * ref_hours * h.median_hourly_wage_usd
            saved_hours = np.where(f.credit, ref_hours - f.mean_latency, 0.)
            gv, gvb = aggregate(value_usd)
            sh, shb = aggregate(saved_hours)
            sr, srb = aggregate(f.credit)
            ur, urb = unweighted(f.credit)
            point[key] = dict(generated_value_usd=gv, net_value_usd=gv-cost,
                              hours_saved=sh, dpj=gv/energy, nvj=(gv-cost)/energy,
                              tsj=sh/energy, weighted_success_rate=sr,
                              unweighted_success_rate=ur)
            boot[key] = dict(generated_value_usd=gvb, net_value_usd=gvb-cost_boot,
                             hours_saved=shb, dpj=gvb/energy_boot, nvj=(gvb-cost_boot)/energy_boot,
                             tsj=shb/energy_boot, weighted_success_rate=srb,
                             unweighted_success_rate=urb)
            cells[key] = dict(decision_reference=NAMES[decision], time_reference=NAMES[value],
                              successes=int(f.credit.sum()),
                              **{k: interval(v, boot[key][k]) for k, v in point[key].items()})

    # Check point estimates and bootstrap endpoints against the frozen source metrics.
    source = pd.read_csv(SOURCE / "global_net_metrics.csv")
    source = source[source.model.eq("glm52_h200")].set_index("baseline")
    checks = []
    for key, baseline in [("EE", "human"), ("AA", "human_ai")]:
        for field, old in [("nvj", "nvj"), ("tsj", "tsj"),
                           ("generated_value_usd", "generated_value_usd"),
                           ("net_value_usd", "net_value_usd"), ("hours_saved", "hours_saved")]:
            expected = source.loc[baseline]
            got = cells[key][field]
            for suffix in ["estimate", "ci_low", "ci_high"]:
                old_column = old if suffix == "estimate" else old + "_" + suffix
                np.testing.assert_allclose(got[suffix], expected[old_column], rtol=1e-11, atol=1e-11)
                checks.append(f"{key}:{field}:{suffix}")
    np.testing.assert_allclose(energy, source.energy_mj.to_numpy())
    np.testing.assert_allclose(cost, source.rental_cost_usd.to_numpy())

    comparisons = {}
    edges = {
        "deliverables_at_expert_only_times": ("EE", "AE"),
        "times_at_expert_only_decisions": ("EE", "EA"),
        "times_at_expert_ai_decisions": ("AE", "AA"),
        "deliverables_at_expert_ai_times": ("EA", "AA"),
        "combined": ("EE", "AA"),
    }
    for label, (start, end) in edges.items():
        metrics = {}
        for metric in ["nvj", "dpj", "tsj", "generated_value_usd", "hours_saved"]:
            reduction = point[start][metric] - point[end][metric]
            reduction_boot = boot[start][metric] - boot[end][metric]
            metrics[metric] = {
                "absolute_reduction": interval(reduction, reduction_boot),
                "percent_reduction": interval(100*reduction/point[start][metric],
                                              100*reduction_boot/boot[start][metric]),
                "percentage_points_of_EE": interval(100*reduction/point["EE"][metric],
                                                    100*reduction_boot/boot["EE"][metric]),
                "share_of_combined_reduction_percent": interval(
                    100*reduction/(point["EE"][metric]-point["AA"][metric]),
                    100*reduction_boot/(boot["EE"][metric]-boot["AA"][metric])),
            }
        comparisons[label] = dict(start=start, end=end, metrics=metrics)

    # Symmetric attribution averages the two switching orders. It is a convention,
    # not a unique causal allocation of the observed loss.
    symmetric = {}
    for metric in ["nvj", "dpj", "tsj"]:
        d = .5*((point["EE"][metric]-point["AE"][metric]) + (point["EA"][metric]-point["AA"][metric]))
        db = .5*((boot["EE"][metric]-boot["AE"][metric]) + (boot["EA"][metric]-boot["AA"][metric]))
        t = .5*((point["EE"][metric]-point["EA"][metric]) + (point["AE"][metric]-point["AA"][metric]))
        tb = .5*((boot["EE"][metric]-boot["EA"][metric]) + (boot["AE"][metric]-boot["AA"][metric]))
        total = point["EE"][metric]-point["AA"][metric]
        totalb = boot["EE"][metric]-boot["AA"][metric]
        np.testing.assert_allclose(d+t, total)
        np.testing.assert_allclose(db+tb, totalb, atol=1e-12)
        symmetric[metric] = {
            "deliverable_absolute_reduction": interval(d, db),
            "time_absolute_reduction": interval(t, tb),
            "deliverable_share_percent": interval(100*d/total, 100*db/totalb),
            "time_share_percent": interval(100*t/total, 100*tb/totalb),
            "interaction": interval(point["AA"][metric]-point["AE"][metric]-point["EA"][metric]+point["EE"][metric],
                                    boot["AA"][metric]-boot["AE"][metric]-boot["EA"][metric]+boot["EE"][metric]),
        }

    success = {
        "both_pass": int((h.credit & a.credit).sum()),
        "expert_only_pass_only": int((h.credit & ~a.credit).sum()),
        "expert_ai_pass_only": int((~h.credit & a.credit).sum()),
        "neither_pass": int((~h.credit & ~a.credit).sum()),
    }
    for metric in ["weighted_success_rate", "unweighted_success_rate"]:
        success[metric+"_drop_pp"] = interval(100*(point["EE"][metric]-point["AA"][metric]),
                                             100*(boot["EE"][metric]-boot["AA"][metric]))

    result = {
        "scope": "GLM-5.2 reference analysis; recorded repeat 0; 220 tasks; 44 occupations",
        "method": "NVJ(s,t) = [sum_i p_i Y_i(s) W_i H_i(t) - sum_i p_i C_i] / sum_i p_i E_i; TSJ(s,t) = sum_i p_i Y_i(s) [H_i(t)-L_i] / sum_i p_i E_i. p_i is occupation GDP weight / 5. All selected failures retain energy and cost. Negative time saving is retained.",
        "bootstrap": {"samples": SAMPLES, "seed": SEED, "method": "paired draws of five tasks with replacement within each occupation; percentile 95% task-composition intervals; same draws for every cell and difference; conditional on fixed judges, crosswalk, rates and energy allocation"},
        "energy_mj": energy, "rental_cost_usd": cost,
        "cells": cells, "comparisons": comparisons, "symmetric_attribution": symmetric,
        "success_transitions": success,
        "expert_ai_time_longer_count": int((h.human_ai_hours > h.human_only_hours).sum()),
        "validation": {"endpoint_checks_passed": len(checks), "checked_fields": checks,
                       "identical_energy_and_cost": True, "identical_task_metadata": True,
                       "identical_latency_on_joint_successes": True,
                       "success_and_latency_match_raw_repeat0_records": True},
        "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in paths+[WEIGHTS, SOURCE/"global_net_metrics.csv", verdict_path, query_path]},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    for key in ["EE", "AE", "EA", "AA"]:
        print(key, {m: cells[key][m] for m in ["nvj", "dpj", "tsj"]})
    print("SUCCESS", success)
    for label, data in comparisons.items():
        print(label, {m: data["metrics"][m]["percent_reduction"] for m in ["nvj", "tsj"]})
    print("SYMMETRIC", symmetric)
    print("VALIDATED", len(checks), "endpoint point/interval values")




if __name__ == "__main__":
    main()
