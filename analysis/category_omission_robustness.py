"""Uniform one-task-per-category sensitivity, with no outcome-based selection.

Primary data are never modified. Each sensitivity portfolio omits 13 tasks,
one uniformly chosen from each category, then averages retained tasks within
occupation while preserving the original occupation weights. Recorded outcomes,
costs, and attributed energy are held fixed. This is not a confidence interval.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/paper_net_metrics_20260919"
WEIGHTS = ROOT / "outputs/paper_gdp_employment_weights/occupation_weights.csv"
OUT = ROOT / "reports/category_omission_robustness_20260920"
SEED = 20260920
TRIALS = 10000
FIELDS = ("generated_value_usd", "rental_cost_usd", "energy_mj", "hours_saved", "credit")


def aggregate(tasks, occupation_weights):
    counts = tasks.groupby("occupation").task_id.transform("count")
    mass = tasks.occupation.map(occupation_weights) / counts
    mass = mass / mass.sum()
    values = {field: float(mass @ tasks[field]) for field in FIELDS}
    values["nvj"] = (values["generated_value_usd"] - values["rental_cost_usd"]) / values["energy_mj"]
    values["tsj"] = values["hours_saved"] / values["energy_mj"]
    values["win_tie_rate"] = values.pop("credit")
    return values


def categories(tasks, weights):
    return pd.DataFrame([
        {"category": name, "n_tasks": len(group), **aggregate(group, weights)}
        for name, group in tasks.groupby("category")
    ]).set_index("category")


def descending_ranks(values):
    return np.argsort(np.argsort(-values, axis=-1), axis=-1) + 1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = SOURCE / "glm52_h200_human_tasks.csv"
    tasks = pd.read_csv(path).sort_values(["category", "task_id"])
    weights = pd.read_csv(WEIGHTS).set_index("occupation").gdp_structure_weight
    base = categories(tasks, weights).sort_index()
    assert len(tasks) == 220 and len(base) == 13
    effects = []
    indices = []
    scores = {m: [] for m in ["nvj", "tsj", "win_tie_rate"]}
    rng = np.random.default_rng(SEED)
    for category, group in tasks.groupby("category", sort=True):
        options = []
        for option, row in enumerate(group.itertuples()):
            r = aggregate(group[group.task_id.ne(row.task_id)], weights)
            options.append(r)
            effects.append({"category": category, "option": option, "omitted_task_id": row.task_id, **r})
        chosen = rng.integers(0, len(options), size=TRIALS)
        indices.append(chosen)
        for m in scores:
            scores[m].append(np.array([r[m] for r in options])[chosen])
    # Category aggregation is independent across disjoint categories; combining
    # these independently sampled columns equals deleting 13 tasks at once.
    scores = {m: np.array(v).T for m, v in scores.items()}
    effects = pd.DataFrame(effects)
    indices = np.array(indices).T
    for trial in [0, TRIALS // 2, TRIALS - 1]:
        removed = [effects[(effects.category.eq(c)) & (effects.option.eq(indices[trial, j]))].omitted_task_id.item()
                   for j, c in enumerate(base.index)]
        reduced = tasks[~tasks.task_id.isin(removed)]
        assert len(reduced) == 207 and len(set(removed)) == 13
        assert tasks.groupby("category").size().sub(reduced.groupby("category").size()).eq(1).all()
        direct = categories(reduced, weights).loc[base.index]
        for m in scores:
            np.testing.assert_allclose(scores[m][trial], direct[m], rtol=1e-12)
    summaries, category_rows, trials = {}, [], pd.DataFrame({"trial": np.arange(TRIALS)})
    stored = {"category_order": base.index.to_numpy(dtype=str), "omission_option": indices}
    n = len(base)
    pair_i, pair_j = np.triu_indices(n, 1)
    for metric, values in scores.items():
        rank = descending_ranks(values)
        baseline_rank = descending_ranks(base[metric].to_numpy())
        squared = ((rank - baseline_rank) ** 2).sum(axis=1)
        rho = 1 - 6 * squared / (n * (n * n - 1))
        pair_sign = np.sign(rank[:, pair_i] - rank[:, pair_j])
        base_sign = np.sign(baseline_rank[pair_i] - baseline_rank[pair_j])
        tau = (pair_sign * base_sign).mean(axis=1)
        winner = np.argmin(baseline_rank)
        same_order = (rank == baseline_rank).all(axis=1)
        top_same = rank[:, winner] == 1
        summaries[metric] = {
            "baseline_leader": str(base.index[winner]),
            "spearman_median": float(np.median(rho)),
            "spearman_5th_95th_percentiles": np.quantile(rho, [.05, .95]).tolist(),
            "kendall_median": float(np.median(tau)),
            "same_leader_percent": float(top_same.mean() * 100),
            "exact_same_order_percent": float(same_order.mean() * 100),
            "median_mean_absolute_rank_change": float(np.median(np.abs(rank - baseline_rank).mean(axis=1))),
            "leader_counts": pd.Series(base.index[np.argmin(rank, axis=1)]).value_counts().to_dict(),
        }
        trials[metric + "_spearman"] = rho
        trials[metric + "_kendall"] = tau
        trials[metric + "_same_leader"] = top_same
        trials[metric + "_same_order"] = same_order
        stored[metric] = values
        stored[metric + "_rank"] = rank
        for j, category in enumerate(base.index):
            category_rows.append({"metric": metric, "category": category,
                "baseline_rank": int(baseline_rank[j]), "estimate": float(base.loc[category, metric]),
                "median_rank": float(np.median(rank[:, j])), "min_rank": int(rank[:, j].min()),
                "max_rank": int(rank[:, j].max()), "unchanged_rank_percent": float((rank[:, j] == baseline_rank[j]).mean() * 100),
                "within_one_rank_percent": float((np.abs(rank[:, j] - baseline_rank[j]) <= 1).mean() * 100)})
    effects.to_csv(OUT / "all_category_omission_options.csv", index=False)
    pd.DataFrame(category_rows).to_csv(OUT / "category_rank_sensitivity.csv", index=False)
    trials.to_csv(OUT / "trial_summary.csv", index=False)
    np.savez_compressed(OUT / "sensitivity_trials.npz", **stored)
    result = {"seed": SEED, "trials": TRIALS, "primary_task_count": 220, "tasks_per_sensitivity_portfolio": 207,
        "method": "Independently and uniformly omit one task per category; preserve occupation mass and renormalize retained tasks within each occupation. No influential task is targeted. These are finite-portfolio perturbations, not new observations or confidence intervals.",
        "metrics": summaries, "direct_207_task_recomputations_verified": True,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [path, WEIGHTS]}}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
