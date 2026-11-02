#!/usr/bin/env python3
"""Export GDP-sector × employment occupation weights for Joules to GDP."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis/metrics.py"
OUT = ROOT / "outputs/paper_gdp_employment_weights"


def load_analysis():
    spec = importlib.util.spec_from_file_location("gdp_employment_paper", ANALYSIS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {ANALYSIS}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    metrics_module = load_analysis()
    _, weights = metrics_module.load_task_metadata()
    category = metrics_module.category_weight_table(
        type("Inputs", (), {"weights": weights, "task_meta": metrics_module.load_task_metadata()[0]})
    )

    OUT.mkdir(parents=True, exist_ok=True)

    columns = [
        "occupation",
        "onet_code",
        "category",
        "sector",
        "oews_soc_code",
        "national_employment",
        "sector_gdp_share_published",
        "sector_gdp_share_normalized",
        "within_sector_employment_share",
        "gdp_structure_weight",
        "legacy_wage_bill_weight",
    ]
    occupation = weights[columns].sort_values(["sector", "occupation"])
    occupation.to_csv(OUT / "occupation_weights.csv", index=False, float_format="%.15g")

    sector = weights.groupby("sector", as_index=False).agg(
        target_sector_gdp_share=("sector_gdp_share_normalized", "first"),
        assigned_weight=("gdp_structure_weight", "sum"),
        sampled_occupations=("occupation", "size"),
        sampled_employment=("national_employment", "sum"),
    )
    sector["reconciliation_error"] = (
        sector.assigned_weight - sector.target_sector_gdp_share
    )
    sector.to_csv(OUT / "sector_reconciliation.csv", index=False, float_format="%.15g")
    category.to_csv(OUT / "category_weights.csv", index=False, float_format="%.15g")

    validation = {
        "method": "normalized sampled-sector GDP share multiplied by within-sector sampled-occupation employment share",
        "wages_used_in_weights": False,
        "wages_used_in_task_valuation": True,
        "occupation_weight_sum": float(weights.gdp_structure_weight.sum()),
        "maximum_sector_reconciliation_error": float(
            np.abs(sector.reconciliation_error).max()
        ),
        "occupation_count": int(len(weights)),
        "sector_count": int(weights.sector.nunique()),
        "category_count": int(weights.category.nunique()),
    }
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
