"""Allocate recorded serving-window rental cost across concurrent requests.

The reference tariff values the recorded deployment's occupied prompt windows;
it is not an invoice or an estimate of startup and other unobserved overhead.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GLM_TARIFF = ROOT / "reports/glm52_reference_rental_20260919/sources/aws_2026-07.json"
OTHER_MODEL_TARIFF = ROOT / "reports/paper_net_metrics_20260919/aws_july_2026_hardware_prices.json"


def occupied_shares(starts, ends, left, right):
    """Allocate each instant equally to active requests and return unoccupied time."""
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    if not len(starts) or len(starts) != len(ends):
        raise ValueError("A nonempty set of matched request intervals is required")
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
        raise ValueError("Request intervals must be finite")
    if not np.all(ends > starts):
        raise ValueError("Request intervals must have positive duration")
    if starts.min() < left - 1e-4 or ends.max() > right + 1e-4:
        raise ValueError("Request intervals exceed batch boundaries")
    starts = np.clip(starts, left, right)
    ends = np.clip(ends, left, right)
    cuts = np.unique(np.r_[left, starts, ends, right])
    allocated = np.zeros(len(starts))
    idle = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        active = (starts <= (a + b) / 2) & (ends > (a + b) / 2)
        if active.any():
            allocated[active] += (b - a) / active.sum()
        else:
            idle += b - a
    np.testing.assert_allclose(allocated.sum() + idle, right - left, rtol=1e-12)
    return allocated, idle


def tariff_rates(path: Path = GLM_TARIFF) -> dict[str, float]:
    source = json.loads(path.read_text())
    rows = source["rows"]

    def unique(usage_type: str) -> dict:
        matches = [row for row in rows if row.get("usageType") == usage_type]
        if len(matches) != 1:
            raise ValueError(f"Expected one {usage_type} tariff row in {path}")
        return matches[0]

    node = unique("BoxUsage:p5en.48xlarge")
    disk = unique("EBS:VolumeUsage.gp3")
    if node.get("MarketOption") != "OnDemand" or node.get("CapacityStatus") != "Used":
        raise ValueError("Reference compute tariff is not ordinary on-demand usage")
    return {
        "node_usd_per_hour": float(node["PricePerUnit"]),
        "gp3_usd_per_gib_month": float(disk["PricePerUnit"]),
        "source_price_period": source["price_period"],
    }


def glm_prompt_window_costs(query_path: Path, batch_path: Path, tariff_path: Path = GLM_TARIFF):
    """Return one cost per recorded GDPVal task attempt and a batch reconciliation."""
    queries = pd.read_csv(query_path).query("dataset == 'GDPVal'").copy()
    batches = pd.read_csv(batch_path).query("dataset == 'GDPVal'").copy()
    if queries.empty or batches.empty or queries["query_id"].duplicated().any():
        raise ValueError("Incomplete or duplicate GLM query/batch ledgers")
    if batches["batch_key"].duplicated().any() or set(queries.batch_key) != set(batches.batch_key):
        raise ValueError("GLM query and batch keys do not agree")
    queries = queries.rename(columns={"judge_task_id": "task_id", "repeat_index": "repeat"})
    prices = tariff_rates(tariff_path)
    # The reference deployment uses a 32 GiB boot disk and one public IPv4.
    hourly = prices["node_usd_per_hour"] + 32 * prices["gp3_usd_per_gib_month"] / 744 + 0.005
    queries["allocated_seconds"] = np.nan
    audit = []
    for batch in batches.itertuples():
        group = queries[queries.batch_key == batch.batch_key]
        if len(group) != int(batch.attempts):
            raise ValueError(f"{batch.batch_key}: missing physical requests")
        shares, idle = occupied_shares(
            group.prompt_start_time_s, group.prompt_end_time_s,
            float(batch.prompt_start_time_s), float(batch.prompt_end_time_s),
        )
        if shares.sum() <= 0:
            raise ValueError(f"{batch.batch_key}: no occupied prompt time")
        allocated = shares + idle * shares / shares.sum()
        queries.loc[group.index, "allocated_seconds"] = allocated
        duration = float(batch.prompt_end_time_s - batch.prompt_start_time_s)
        np.testing.assert_allclose(allocated.sum(), duration, rtol=1e-12)
        audit.append({
            "batch_key": batch.batch_key,
            "request_count": len(group),
            "prompt_window_seconds": duration,
            "allocated_seconds": float(allocated.sum()),
            "unoccupied_seconds": idle,
            "prompt_window_cost_usd": duration * hourly / 3600,
        })
    if queries.allocated_seconds.isna().any():
        raise ValueError("Some GLM requests received no cost allocation")
    queries["rental_cost_usd"] = queries.allocated_seconds * hourly / 3600
    queries["rate_usd_per_hour"] = hourly
    queries["cost_scope"] = "recorded_prompt_window_only"
    ledger = queries[["task_id", "repeat", "batch_key", "query_id", "allocated_seconds",
                      "rate_usd_per_hour", "rental_cost_usd", "cost_scope"]].copy()
    if ledger.duplicated(["task_id", "repeat"]).any():
        raise ValueError("More than one GLM cost per logical attempt")
    np.testing.assert_allclose(
        ledger.rental_cost_usd.sum(),
        sum(row["prompt_window_cost_usd"] for row in audit), rtol=1e-12,
    )
    return ledger, pd.DataFrame(audit)
