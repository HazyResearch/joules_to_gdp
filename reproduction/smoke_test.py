#!/usr/bin/env python3
"""Check Joules to GDP calculations offline with fabricated inputs.

No model, judge service, or energy meter is contacted. All intermediate files
are created in a temporary directory and removed when the check finishes.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

FIXTURE = ROOT / "analysis/fixtures/energy_boundary"


def check_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError(f"{label}: expected {expected}, received {actual}")


def check_metrics(tasks: pd.DataFrame, *, value: float, time_saving: float) -> None:
    """Check aggregate ratios from the production calculation's components."""
    energy_mj = float(tasks.energy_mj.sum())
    execution_cost = float(tasks.rental_cost_usd.sum())
    check_close(energy_mj, 125 / 1_000_000, "All-attempt energy (MJ)")
    check_close(execution_cost, 0.30, "All-attempt execution cost (USD)")
    check_close(tasks.generated_value_usd.sum(), value, "Credited output dollar value")
    check_close(tasks.hours_saved.sum(), time_saving, "Credited time saving (hours)")
    net_value = float((tasks.generated_value_usd - tasks.rental_cost_usd).sum())
    check_close(net_value, value - 0.30, "Net dollar value")
    check_close(net_value / energy_mj, (value - 0.30) / 0.000125, "NVJ (USD/MJ)")
    check_close(tasks.hours_saved.sum() / energy_mj,
                time_saving / 0.000125, "TSJ (hours/MJ)")


def main() -> None:
    import pandas as pd

    from build_energy_components_from_ipw import build_from_ipw
    from build_net_metrics import prompt_from_attempts
    from energy_boundary import build
    from rental_cost import occupied_shares

    with tempfile.TemporaryDirectory(prefix="joules-to-gdp-smoke-") as directory:
        output = Path(directory)
        counts = build_from_ipw(FIXTURE / "ipw_config.json", output)
        if counts != (1, 2):
            raise RuntimeError(f"Expected one fabricated batch and two tasks: {counts}")
        validation = build(
            output / "energy_component_batches.csv",
            output / "energy_component_requests.csv", output,
            expected_profiles=("fixture",), expected_tasks=2,
            expected_repeats=(0,), expected_batches=1, expected_batch_size=2,
            expected_occupations=1, task_inventory_path=FIXTURE / "tasks.json",
        )
        for component, expected in (("gpu", 100), ("cpu", 20), ("dram", 5)):
            check_close(validation["profiles"]["fixture"][f"{component}_j"],
                        expected, f"Batch {component.upper()} energy (J)")

        energy = pd.read_csv(output / "attempt_energy.csv").sort_values("task_id")
        for row, expected in zip(energy.itertuples(), (93.75, 31.25), strict=True):
            check_close(row.total_energy_j, expected, "Task energy attribution (J)")
        check_close(energy.total_energy_j.sum(), 125, "Complete batch energy (J)")

        # Allocate a fabricated $108/hour system over the same ten-second batch.
        requests = pd.read_csv(output / "energy_component_requests.csv").sort_values("task_id")
        seconds, idle = occupied_shares(requests.start_s, requests.end_s, 0, 10)
        check_close(idle, 0, "Unoccupied batch time (seconds)")
        check_close(seconds.sum(), 10, "Allocated execution time (seconds)")
        costs = requests[["task_id", "repeat"]].copy()
        costs["rental_cost_usd"] = seconds * 108 / 3600

        metadata = pd.DataFrame({
            "task_id": energy.task_id,
            "human_only_hours": [2.0, 1.0],
            "median_hourly_wage_usd": [30.0, 40.0],
        })
        inputs = SimpleNamespace(task_meta=metadata)
        attempts = energy[["task_id", "repeat"]].copy()
        attempts["energy_mj"] = energy.total_energy_j / 1_000_000
        attempts["latency_hours"] = (requests.end_s - requests.start_s).to_numpy() / 3600

        for outcome in ("win", "tie"):
            # The second task loses, but its measured energy and cost still count.
            attempts["success"] = pd.Series([outcome, "loss"]).isin(["win", "tie"])
            tasks = prompt_from_attempts(inputs, attempts, costs, baseline="human")
            check_metrics(tasks, value=60, time_saving=2 - 10 / 3600)
            loss = tasks.loc[tasks.task_id.eq("fabricated-task-2")].iloc[0]
            check_close(loss.generated_value_usd, 0, "Loss output value")
            check_close(loss.hours_saved, 0, "Loss time saving")
            check_close(loss.rental_cost_usd, 0.075, "Loss execution cost retained")
            check_close(loss.energy_mj, 31.25 / 1_000_000, "Loss energy retained")

        # A qualifying ten-second run saves negative time against a one-second reference.
        inputs.task_meta = metadata.assign(human_only_hours=[1 / 3600, 1.0])
        tasks = prompt_from_attempts(inputs, attempts, costs, baseline="human")
        check_metrics(tasks, value=30 / 3600, time_saving=-9 / 3600)

    print("Joules to GDP smoke test passed (fabricated data)")
    print("Checked: batch GPU/CPU/DRAM energy, task energy attribution, execution cost,")
    print("expert-time and wage valuation, win/tie/loss credit, negative time saving, NVJ, and TSJ.")


if __name__ == "__main__":
    main()
