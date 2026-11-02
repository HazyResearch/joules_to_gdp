#!/usr/bin/env python3
"""Compute the GLM GDPVal GDP-structured value-capture index.

The headline index is a weighted average of occupation-level task-value capture
rates. GDPVal sector shares determine the weight assigned to each of the nine
included sectors; May 2024 national OEWS wage bills allocate each sector's
weight across GDPVal's selected occupations.

This report intentionally does not describe the result as a share of U.S. GDP.
It is a newly constructed benchmark index using GDPVal's published economic
structure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "outputs/019f86a0-e5e6-7e90-8578-b08ddb4385fb"
THREAD_DIR = ROOT / "outputs/019f9187-6eaf-7643-b1d4-931e16f4d844"
PAIRWISE_DIR = ROOT / "reports/gdpval_pairwise_glm52_20260715"

QUERY_LEDGER = SOURCE_DIR / "glm52_no_rubric_query_ledger.csv"
BATCH_LEDGER = SOURCE_DIR / "glm52_gdpval_batch_ledger.csv"
RESULTS = PAIRWISE_DIR / "results.output_level.jsonl"
MANIFEST = PAIRWISE_DIR / "comparison_manifest.unblinded.jsonl"
VALUATION_WORKBOOK = SOURCE_DIR / "gdpval_snorkel_task_wage_values.xlsx"
OEWS_ZIP = THREAD_DIR / "sources/oews_may2024/oesm24nat.zip"
OEWS_WORKBOOK = (
    THREAD_DIR
    / "sources/oews_may2024/oesm24nat/national_M2024_dl.xlsx"
)

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_724

SECTOR_GDP_SHARES = {
    "Real Estate and Rental and Leasing": 0.138,
    "Manufacturing": 0.100,
    "Professional, Scientific, and Technical Services": 0.081,
    "Government": 0.113,
    "Health Care and Social Assistance": 0.076,
    "Finance and Insurance": 0.074,
    "Retail Trade": 0.063,
    "Wholesale Trade": 0.058,
    "Information": 0.054,
}

CATEGORIES = {
    "11": "Management",
    "13": "Business & Financial Operations",
    "15": "Computer & Mathematical",
    "17": "Architecture & Engineering",
    "21": "Community & Social Service",
    "23": "Legal",
    "27": "Arts, Design, Entertainment & Media",
    "29": "Healthcare Practitioners & Technical",
    "33": "Protective Service",
    "39": "Personal Care & Service",
    "41": "Sales & Related",
    "43": "Office & Administrative Support",
    "51": "Production",
}

# May 2024 OEWS publishes this occupation at the aggregate 13-1020 code.
OEWS_SOC_OVERRIDES = {"13-1023": "13-1020"}

# GDPVal Table 1, billions of dollars, rounded to two decimal places.
PUBLISHED_COMPENSATION_BILLIONS = {
    "Property, Real Estate, and Community Association Managers": 24.54,
    "Counter and Rental Clerks": 17.42,
    "Real Estate Sales Agents": 13.53,
    "Real Estate Brokers": 4.55,
    "Concierges": 1.80,
    "First-Line Supervisors of Production and Operating Workers": 51.07,
    "Buyers and Purchasing Agents": 39.79,
    "Shipping, Receiving, and Inventory Clerks": 38.50,
    "Industrial Engineers": 37.79,
    "Mechanical Engineers": 31.57,
    "Software Developers": 239.18,
    "Lawyers": 136.66,
    "Accountants and Auditors": 135.44,
    "Computer and Information Systems Managers": 121.44,
    "Project Management Specialists": 108.77,
    "Compliance Officers": 33.80,
    "Administrative Services Managers": 32.03,
    "Child, Family, and School Social Workers": 24.10,
    "First-Line Supervisors of Police and Detectives": 17.00,
    "Recreation Workers": 11.51,
    "Registered Nurses": 323.05,
    "First-Line Supervisors of Office and Administrative Support Workers": 107.02,
    "Medical and Health Services Managers": 77.93,
    "Nurse Practitioners": 40.58,
    "Medical Secretaries and Administrative Assistants": 37.87,
    "Financial Managers": 147.74,
    "Customer Service Representatives": 123.70,
    "Securities, Commodities, and Financial Services Sales Agents": 52.14,
    "Personal Financial Advisors": 43.33,
    "Financial and Investment Analysts": 39.67,
    "General and Operations Managers": 477.16,
    "First-Line Supervisors of Retail Sales Workers": 58.27,
    "Pharmacists": 45.12,
    "Private Detectives and Investigators": 2.39,
    (
        "Sales Representatives, Wholesale and Manufacturing, Except Technical "
        "and Scientific Products"
    ): 103.21,
    "Sales Managers": 97.16,
    (
        "Sales Representatives, Wholesale and Manufacturing, Technical and "
        "Scientific Products"
    ): 33.66,
    "First-Line Supervisors of Non-Retail Sales Workers": 21.43,
    "Order Clerks": 3.86,
    "Producers and Directors": 16.60,
    "Editors": 8.18,
    "News Analysts, Reporters, and Journalists": 4.41,
    "Audio and Video Technicians": 4.30,
    "Film and Video Editors": 2.41,
}

SOURCE_URLS = {
    "oews_may_2024": (
        "https://www.bls.gov/oes/special-requests/oesm24nat.zip"
    ),
    "gdpval": "https://openai.com/index/gdpval/",
    "gdpval_paper": (
        "https://cdn.openai.com/pdf/"
        "d5eb7428-c4e9-4a33-bd86-86dd4bcf12ce/GDPval.pdf"
    ),
    "anthropic_productivity": (
        "https://www.anthropic.com/research/estimating-productivity-gains"
    ),
    "task_based_hulten": (
        "https://economics.mit.edu/sites/default/files/2024-04/"
        "The%20Simple%20Macroeconomics%20of%20AI.pdf"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ci(values: np.ndarray) -> dict[str, float]:
    return {
        "lower_95": float(np.percentile(values, 2.5)),
        "upper_95": float(np.percentile(values, 97.5)),
    }


def weighted_support(
    occupation_rows: list[dict[str, Any]], weight_field: str
) -> dict[str, float]:
    weighted_generated_panel = sum(
        row[weight_field] * row["generated_value_panel_usd"]
        for row in occupation_rows
    )
    weighted_hours_panel = sum(
        row[weight_field] * row["hours_saved_panel"]
        for row in occupation_rows
    )
    weighted_energy_panel = sum(
        row[weight_field] * row["energy_panel_mj"]
        for row in occupation_rows
    )
    return {
        "generated_value_per_representative_task_usd": (
            weighted_generated_panel / 5
        ),
        "generated_value_220_task_equivalent_usd": (
            weighted_generated_panel * 44
        ),
        "energy_per_representative_task_all_four_runs_mj": (
            weighted_energy_panel / 5
        ),
        "energy_220_task_equivalent_all_four_runs_mj": (
            weighted_energy_panel * 44
        ),
        "value_per_mj_ratio_of_weighted_sums": (
            weighted_generated_panel / weighted_energy_panel
        ),
        "hours_saved_per_representative_task": weighted_hours_panel / 5,
        "hours_saved_220_task_equivalent": weighted_hours_panel * 44,
        "hours_saved_per_mj_ratio_of_weighted_sums": (
            weighted_hours_panel / weighted_energy_panel
        ),
    }


def load_base() -> tuple[
    list[dict[str, Any]],
    dict[str, dict[tuple[str, int], dict[str, Any]]],
    dict[str, dict[str, Any]],
    list[dict[str, str]],
]:
    manifest_rows = [
        row
        for row in read_jsonl(MANIFEST)
        if row["condition"] == "without_rubric" and row["baseline"] == "human"
    ]
    if len(manifest_rows) != 880:
        raise AssertionError(f"expected 880 manifest rows, got {len(manifest_rows)}")
    manifest_by_key = {
        (str(row["task_id"]), int(row["repeat"])): row
        for row in manifest_rows
    }
    if len(manifest_by_key) != 880:
        raise AssertionError("duplicate expert-only reference manifest keys")

    results_by_baseline: dict[
        str, dict[tuple[str, int], dict[str, Any]]
    ] = {}
    all_results = read_jsonl(RESULTS)
    for baseline in ("human", "human_ai"):
        rows = [
            row
            for row in all_results
            if row["condition"] == "without_rubric"
            and row["baseline"] == baseline
        ]
        if len(rows) != 880:
            raise AssertionError(
                f"expected 880 {baseline} results, got {len(rows)}"
            )
        by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            key = (str(row["task_id"]), int(row["repeat"]))
            if key in by_key:
                raise AssertionError(f"duplicate {baseline} result {key}")
            if row["agent_outcome"] not in {"win", "tie", "loss"}:
                raise AssertionError(f"invalid outcome in {key}")
            by_key[key] = row
        results_by_baseline[baseline] = by_key

    queries = [
        row for row in read_csv(QUERY_LEDGER) if row["dataset"] == "GDPVal"
    ]
    if len(queries) != 880:
        raise AssertionError(f"expected 880 query rows, got {len(queries)}")
    query_by_key = {
        (str(row["judge_task_id"]), int(row["repeat_index"])): row
        for row in queries
    }
    if len(query_by_key) != 880:
        raise AssertionError("duplicate query keys")

    attempts: list[dict[str, Any]] = []
    for key, query in query_by_key.items():
        manifest = manifest_by_key.get(key)
        if manifest is None:
            raise AssertionError(f"missing manifest record for {key}")
        task_id, repeat = key
        onet_code = str(query["onet_code"])
        major = onet_code[:2]
        if major not in CATEGORIES:
            raise AssertionError(f"unmapped O*NET code {onet_code}")
        sector = str(manifest["sector"])
        if sector not in SECTOR_GDP_SHARES:
            raise AssertionError(f"unmapped sector {sector}")
        if str(query["occupation"]) != str(manifest["occupation"]):
            raise AssertionError(f"occupation mismatch {key}")
        human_hours = float(query["human_only_hours"])
        human_ai_hours = float(query["human_ai_hours"])
        hourly_wage = float(query["hourly_wage_usd"])
        human_value = float(query["human_only_value_usd"])
        human_ai_value = float(query["human_ai_value_usd"])
        if not math.isclose(
            human_value, human_hours * hourly_wage, abs_tol=1e-8
        ):
            raise AssertionError(f"expert-only valuation mismatch {key}")
        if not math.isclose(
            human_ai_value, human_ai_hours * hourly_wage, abs_tol=1e-8
        ):
            raise AssertionError(f"expert-with-AI valuation mismatch {key}")
        latency_seconds = float(query["query_latency_s"])
        latency_hours = float(query["agent_latency_hours"])
        if not math.isclose(
            latency_seconds / 3600,
            latency_hours,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise AssertionError(f"latency mismatch {key}")
        energy_j = float(query["allocated_gpu_energy_joules"])
        if not math.isfinite(energy_j) or energy_j <= 0:
            raise AssertionError(f"invalid energy {key}")
        attempts.append(
            {
                "task_id": task_id,
                "baseline_task_id": str(query["prompt_id"]),
                "repeat": repeat,
                "occupation": str(query["occupation"]),
                "onet_code": onet_code,
                "category": CATEGORIES[major],
                "sector": sector,
                "hourly_wage_usd": hourly_wage,
                "human_only_hours": human_hours,
                "human_ai_hours": human_ai_hours,
                "human_only_value_usd": human_value,
                "human_ai_value_usd": human_ai_value,
                "latency_hours": latency_hours,
                "latency_seconds": latency_seconds,
                "energy_j": energy_j,
                "energy_mj": energy_j / 1_000_000,
                "batch_key": str(query["batch_key"]),
                "query_id": str(query["query_id"]),
                "completed": query["completed"] == "True",
                "timed_out": query["timed_out"] == "True",
            }
        )

    counts = Counter(row["task_id"] for row in attempts)
    if len(counts) != 220 or set(counts.values()) != {4}:
        raise AssertionError("expected 220 prompts with four attempts")
    repeats: dict[str, set[int]] = defaultdict(set)
    for row in attempts:
        repeats[row["task_id"]].add(row["repeat"])
    if any(value != {0, 1, 2, 3} for value in repeats.values()):
        raise AssertionError("repeat coverage is not exactly 0-3")

    task_meta: dict[str, dict[str, Any]] = {}
    for row in attempts:
        meta = {
            key: row[key]
            for key in (
                "task_id",
                "baseline_task_id",
                "occupation",
                "onet_code",
                "category",
                "sector",
                "hourly_wage_usd",
                "human_only_hours",
                "human_ai_hours",
                "human_only_value_usd",
                "human_ai_value_usd",
            )
        }
        prior = task_meta.setdefault(row["task_id"], meta)
        if prior != meta:
            raise AssertionError(f"inconsistent task metadata {row['task_id']}")

    batch_rows = read_csv(BATCH_LEDGER)
    if len(batch_rows) != 44:
        raise AssertionError(f"expected 44 batch rows, got {len(batch_rows)}")
    return attempts, results_by_baseline, task_meta, batch_rows


def load_weights(
    attempts: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    occupation_meta: dict[str, dict[str, Any]] = {}
    for row in attempts:
        meta = {
            "occupation": row["occupation"],
            "onet_code": row["onet_code"],
            "soc_code": row["onet_code"].removesuffix(".00"),
            "category": row["category"],
            "sector": row["sector"],
        }
        prior = occupation_meta.setdefault(row["occupation"], meta)
        if prior != meta:
            raise AssertionError(f"inconsistent occupation metadata {row['occupation']}")
    if len(occupation_meta) != 44:
        raise AssertionError(f"expected 44 occupations, got {len(occupation_meta)}")
    if set(occupation_meta) != set(PUBLISHED_COMPENSATION_BILLIONS):
        raise AssertionError("published compensation occupation coverage mismatch")

    frame = pd.read_excel(OEWS_WORKBOOK, dtype={"OCC_CODE": str})
    frame = frame[frame["O_GROUP"] == "detailed"].copy()
    frame_by_soc = {
        str(row["OCC_CODE"]): row for _, row in frame.iterrows()
    }

    weight_rows: list[dict[str, Any]] = []
    for occupation, meta in occupation_meta.items():
        requested_soc = meta["soc_code"]
        oews_soc = OEWS_SOC_OVERRIDES.get(requested_soc, requested_soc)
        source = frame_by_soc.get(oews_soc)
        if source is None:
            raise AssertionError(
                f"missing May 2024 OEWS match for {occupation} ({oews_soc})"
            )
        employment = pd.to_numeric(source["TOT_EMP"], errors="coerce")
        annual_mean = pd.to_numeric(source["A_MEAN"], errors="coerce")
        hourly_mean = pd.to_numeric(source["H_MEAN"], errors="coerce")
        annual_wage_source = "A_MEAN"
        if pd.isna(annual_mean):
            if pd.isna(hourly_mean):
                raise AssertionError(f"missing mean wage for {occupation}")
            annual_mean = hourly_mean * 2080
            annual_wage_source = "H_MEAN × 2,080"
        if pd.isna(employment):
            raise AssertionError(f"missing employment for {occupation}")
        wage_bill = float(employment) * float(annual_mean)
        published = PUBLISHED_COMPENSATION_BILLIONS[occupation]
        if round(wage_bill / 1_000_000_000, 2) != published:
            raise AssertionError(
                f"GDPVal compensation mismatch for {occupation}: "
                f"{wage_bill / 1e9:.6f} vs {published:.2f}"
            )
        weight_rows.append(
            {
                **meta,
                "oews_soc_code": oews_soc,
                "oews_occupation_title": str(source["OCC_TITLE"]),
                "soc_match_type": (
                    "published aggregate crosswalk"
                    if oews_soc != requested_soc
                    else "exact SOC"
                ),
                "national_employment": float(employment),
                "mean_annual_wage_usd": float(annual_mean),
                "mean_hourly_wage_usd": float(hourly_mean),
                "annual_wage_source": annual_wage_source,
                "wage_bill_usd": wage_bill,
                "wage_bill_billions_usd": wage_bill / 1_000_000_000,
                "gdpval_published_compensation_billions_usd": published,
                "gdpval_compensation_rounding_error_billions_usd": (
                    wage_bill / 1_000_000_000 - published
                ),
                "sector_gdp_share_published": SECTOR_GDP_SHARES[meta["sector"]],
            }
        )

    gdp_share_total = sum(SECTOR_GDP_SHARES.values())
    total_wage_bill = sum(row["wage_bill_usd"] for row in weight_rows)
    sector_wage_bills: dict[str, float] = defaultdict(float)
    for row in weight_rows:
        sector_wage_bills[row["sector"]] += row["wage_bill_usd"]
    for row in weight_rows:
        row["sector_gdp_share_normalized"] = (
            row["sector_gdp_share_published"] / gdp_share_total
        )
        row["within_sector_wage_bill_share"] = (
            row["wage_bill_usd"] / sector_wage_bills[row["sector"]]
        )
        row["gdp_structure_weight"] = (
            row["sector_gdp_share_normalized"]
            * row["within_sector_wage_bill_share"]
        )
        row["wage_bill_only_weight"] = row["wage_bill_usd"] / total_wage_bill
        row["unweighted_occupation_weight"] = 1 / 44

    weight_rows.sort(key=lambda row: (row["sector"], row["occupation"]))
    by_occupation = {row["occupation"]: row for row in weight_rows}
    return weight_rows, by_occupation


def build_prompts(
    attempts: list[dict[str, Any]],
    results_by_key: dict[tuple[str, int], dict[str, Any]],
    baseline: str,
    weights: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    attempts_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        enriched = dict(row)
        result = results_by_key[(row["task_id"], row["repeat"])]
        enriched["outcome"] = str(result["agent_outcome"])
        enriched["success"] = enriched["outcome"] in {"win", "tie"}
        attempts_by_task[row["task_id"]].append(enriched)

    prompts: list[dict[str, Any]] = []
    for task_id, rows in attempts_by_task.items():
        rows.sort(key=lambda row: row["repeat"])
        first = rows[0]
        successes = [row for row in rows if row["success"]]
        mean_latency = (
            sum(row["latency_hours"] for row in successes) / len(successes)
            if successes
            else None
        )
        baseline_hours_field = (
            "human_only_hours" if baseline == "human" else "human_ai_hours"
        )
        baseline_value_field = (
            "human_only_value_usd"
            if baseline == "human"
            else "human_ai_value_usd"
        )
        prompts.append(
            {
                "task_id": task_id,
                "baseline": baseline,
                "baseline_task_id": first["baseline_task_id"],
                "occupation": first["occupation"],
                "onet_code": first["onet_code"],
                "category": first["category"],
                "sector": first["sector"],
                "hourly_wage_usd": first["hourly_wage_usd"],
                "human_only_hours": first["human_only_hours"],
                "human_ai_hours": first["human_ai_hours"],
                "human_only_value_usd": first["human_only_value_usd"],
                "human_ai_value_usd": first["human_ai_value_usd"],
                "baseline_hours": first[baseline_hours_field],
                "baseline_value_usd": first[baseline_value_field],
                "success_count": len(successes),
                "mean_success_latency_hours": mean_latency,
                "hours_saved_if_qualified": (
                    first[baseline_hours_field] - mean_latency
                    if mean_latency is not None
                    else None
                ),
                "human_to_ai_hours_saved_if_qualified": (
                    first["human_only_hours"] - mean_latency
                    if mean_latency is not None
                    else None
                ),
                "human_to_human_ai_hours_saved": (
                    first["human_only_hours"] - first["human_ai_hours"]
                ),
                "human_ai_to_ai_hours_saved_if_qualified": (
                    first["human_ai_hours"] - mean_latency
                    if mean_latency is not None
                    else None
                ),
                "energy_all_four_runs_mj": sum(row["energy_mj"] for row in rows),
                "successful_attempts": len(successes),
                "attempt_count": len(rows),
                "gdp_structure_weight": weights[first["occupation"]][
                    "gdp_structure_weight"
                ],
                "wage_bill_only_weight": weights[first["occupation"]][
                    "wage_bill_only_weight"
                ],
                "unweighted_occupation_weight": 1 / 44,
            }
        )
    prompts.sort(key=lambda row: (row["occupation"], row["task_id"]))
    if len(prompts) != 220:
        raise AssertionError("expected 220 prompt records")
    occ_counts = Counter(row["occupation"] for row in prompts)
    if len(occ_counts) != 44 or set(occ_counts.values()) != {5}:
        raise AssertionError("expected five prompts per occupation")
    return prompts


def bootstrap_global(
    prompts: list[dict[str, Any]],
    occupations: list[str],
    weights: dict[str, dict[str, Any]],
    indices: np.ndarray,
    threshold: int,
) -> dict[str, Any]:
    by_occ: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prompts:
        by_occ[row["occupation"]].append(row)
    baseline_values = np.array(
        [[row["baseline_value_usd"] for row in by_occ[occ]] for occ in occupations]
    )
    qualified_values = np.array(
        [
            [
                row["baseline_value_usd"]
                if row["success_count"] >= threshold
                else 0.0
                for row in by_occ[occ]
            ]
            for occ in occupations
        ]
    )
    hours_saved = np.array(
        [
            [
                float(row["hours_saved_if_qualified"])
                if row["success_count"] >= threshold
                else 0.0
                for row in by_occ[occ]
            ]
            for occ in occupations
        ]
    )
    energy = np.array(
        [
            [row["energy_all_four_runs_mj"] for row in by_occ[occ]]
            for occ in occupations
        ]
    )
    occ_axis = np.arange(len(occupations))[None, :, None]
    den = baseline_values[occ_axis, indices].sum(axis=2)
    num = qualified_values[occ_axis, indices].sum(axis=2)
    saved = hours_saved[occ_axis, indices].sum(axis=2)
    energy_panel = energy[occ_axis, indices].sum(axis=2)
    capture_rates = num / den
    gdp_w = np.array([weights[occ]["gdp_structure_weight"] for occ in occupations])
    wage_w = np.array([weights[occ]["wage_bill_only_weight"] for occ in occupations])
    unweighted_w = np.repeat(1 / 44, 44)

    result: dict[str, Any] = {
        "gdp_structure_gvci_percent": ci(100 * (capture_rates @ gdp_w)),
        "wage_bill_only_value_capture_percent": ci(
            100 * (capture_rates @ wage_w)
        ),
        "unweighted_occupation_value_capture_percent": ci(
            100 * (capture_rates @ unweighted_w)
        ),
    }
    weighted_generated_panel = num @ wage_w
    weighted_saved_panel = saved @ wage_w
    weighted_energy_panel = energy_panel @ wage_w
    result["wage_bill_weighted_generated_value_per_representative_task_usd"] = ci(
        weighted_generated_panel / 5
    )
    result["wage_bill_weighted_generated_value_220_task_equivalent_usd"] = ci(
        weighted_generated_panel * 44
    )
    result["wage_bill_weighted_value_per_mj"] = ci(
        weighted_generated_panel / weighted_energy_panel
    )
    result["wage_bill_weighted_hours_saved_per_representative_task"] = ci(
        weighted_saved_panel / 5
    )
    result["wage_bill_weighted_hours_saved_220_task_equivalent"] = ci(
        weighted_saved_panel * 44
    )
    result["wage_bill_weighted_hours_saved_per_mj"] = ci(
        weighted_saved_panel / weighted_energy_panel
    )
    result["unweighted_total_generated_value_usd"] = ci(num.sum(axis=1))
    result["unweighted_total_hours_saved"] = ci(saved.sum(axis=1))
    result["unweighted_value_per_mj"] = ci(
        num.sum(axis=1) / energy_panel.sum(axis=1)
    )
    result["unweighted_hours_saved_per_mj"] = ci(
        saved.sum(axis=1) / energy_panel.sum(axis=1)
    )
    return result


def calculate_threshold(
    prompts: list[dict[str, Any]],
    weight_rows: list[dict[str, Any]],
    weight_by_occ: dict[str, dict[str, Any]],
    threshold: int,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    occupation_rows: list[dict[str, Any]] = []
    for weight in weight_rows:
        occ_prompts = [
            row for row in prompts if row["occupation"] == weight["occupation"]
        ]
        qualifying = [
            row for row in occ_prompts if row["success_count"] >= threshold
        ]
        available_value = sum(row["baseline_value_usd"] for row in occ_prompts)
        generated_value = sum(row["baseline_value_usd"] for row in qualifying)
        hours_saved = sum(
            float(row["hours_saved_if_qualified"]) for row in qualifying
        )
        energy_mj = sum(row["energy_all_four_runs_mj"] for row in occ_prompts)
        occupation_rows.append(
            {
                "occupation": weight["occupation"],
                "onet_code": weight["onet_code"],
                "category": weight["category"],
                "sector": weight["sector"],
                "threshold": threshold,
                "qualifying_prompts": len(qualifying),
                "total_prompts": len(occ_prompts),
                "successful_attempts": sum(
                    row["successful_attempts"] for row in occ_prompts
                ),
                "total_attempts": 20,
                "available_baseline_value_panel_usd": available_value,
                "generated_value_panel_usd": generated_value,
                "value_capture_rate": generated_value / available_value,
                "hours_saved_panel": hours_saved,
                "energy_panel_mj": energy_mj,
                "gdp_structure_weight": weight["gdp_structure_weight"],
                "wage_bill_only_weight": weight["wage_bill_only_weight"],
                "unweighted_occupation_weight": 1 / 44,
            }
        )

    indices = {
        "gdp_structure_gvci_percent": 100
        * sum(
            row["gdp_structure_weight"] * row["value_capture_rate"]
            for row in occupation_rows
        ),
        "wage_bill_only_value_capture_percent": 100
        * sum(
            row["wage_bill_only_weight"] * row["value_capture_rate"]
            for row in occupation_rows
        ),
        "unweighted_occupation_value_capture_percent": 100
        * sum(row["value_capture_rate"] for row in occupation_rows)
        / 44,
    }

    categories: list[dict[str, Any]] = []
    for category in CATEGORIES.values():
        rows = [row for row in occupation_rows if row["category"] == category]
        category_weight = sum(row["gdp_structure_weight"] for row in rows)
        contribution = 100 * sum(
            row["gdp_structure_weight"] * row["value_capture_rate"]
            for row in rows
        )
        weighted_generated_panel = sum(
            row["gdp_structure_weight"] * row["generated_value_panel_usd"]
            for row in rows
        )
        weighted_hours_panel = sum(
            row["gdp_structure_weight"] * row["hours_saved_panel"]
            for row in rows
        )
        weighted_energy_panel = sum(
            row["gdp_structure_weight"] * row["energy_panel_mj"]
            for row in rows
        )
        categories.append(
            {
                "category": category,
                "threshold": threshold,
                "occupation_count": len(rows),
                "prompt_count": 5 * len(rows),
                "gdp_structure_weight": category_weight,
                "value_capture_rate": (
                    contribution / 100 / category_weight
                ),
                "gvci_contribution_percentage_points": contribution,
                "weighted_generated_value_per_representative_task_usd": (
                    weighted_generated_panel / 5
                ),
                "weighted_generated_value_220_task_equivalent_contribution_usd": (
                    weighted_generated_panel * 44
                ),
                "weighted_energy_per_representative_task_all_four_runs_mj": (
                    weighted_energy_panel / 5
                ),
                "value_per_mj_ratio_of_weighted_sums": (
                    weighted_generated_panel / weighted_energy_panel
                ),
                "weighted_hours_saved_per_representative_task": (
                    weighted_hours_panel / 5
                ),
                "weighted_hours_saved_220_task_equivalent_contribution": (
                    weighted_hours_panel * 44
                ),
                "hours_saved_per_mj_ratio_of_weighted_sums": (
                    weighted_hours_panel / weighted_energy_panel
                ),
            }
        )
    categories.sort(
        key=lambda row: (-row["gvci_contribution_percentage_points"], row["category"])
    )
    if not math.isclose(
        sum(row["gvci_contribution_percentage_points"] for row in categories),
        indices["gdp_structure_gvci_percent"],
        abs_tol=1e-12,
    ):
        raise AssertionError("category GVCI contributions do not reconcile")

    raw_generated = sum(row["generated_value_panel_usd"] for row in occupation_rows)
    raw_hours = sum(row["hours_saved_panel"] for row in occupation_rows)
    raw_energy = sum(row["energy_panel_mj"] for row in occupation_rows)
    supporting = {
        "wage_bill_weighted": weighted_support(
            occupation_rows, "wage_bill_only_weight"
        ),
        "gdp_structure_weighted": weighted_support(
            occupation_rows, "gdp_structure_weight"
        ),
        "original_unweighted": {
            "generated_value_usd": raw_generated,
            "hours_saved": raw_hours,
            "energy_all_four_runs_mj": raw_energy,
            "value_per_mj": raw_generated / raw_energy,
            "hours_saved_per_mj": raw_hours / raw_energy,
        },
    }

    occupations = [row["occupation"] for row in weight_rows]
    bootstrap = bootstrap_global(
        prompts,
        occupations,
        weight_by_occ,
        bootstrap_indices,
        threshold,
    )
    return {
        "threshold": threshold,
        "global_indices": indices,
        "global_supporting_metrics": supporting,
        "bootstrap_95_ci": bootstrap,
        "categories": categories,
        "occupations": occupation_rows,
    }


def calculate_three_way_time(
    prompts: list[dict[str, Any]],
    weight_rows: list[dict[str, Any]],
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    pathways = {
        "human_to_ai": {
            "hours_field": "human_to_ai_hours_saved_if_qualified",
            "denominator_field": "human_only_hours",
        },
        "human_to_human_ai": {
            "hours_field": "human_to_human_ai_hours_saved",
            "denominator_field": "human_only_hours",
        },
        "human_ai_to_ai": {
            "hours_field": "human_ai_to_ai_hours_saved_if_qualified",
            "denominator_field": "human_ai_hours",
        },
    }
    output: dict[str, Any] = {}
    for threshold in range(1, 5):
        qualifying = [
            row for row in prompts if row["success_count"] >= threshold
        ]
        occupations = [row["occupation"] for row in weight_rows]
        prompts_by_occ: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in prompts:
            prompts_by_occ[row["occupation"]].append(row)
        for rows in prompts_by_occ.values():
            rows.sort(key=lambda row: row["task_id"])
        occ_axis = np.arange(len(occupations))[None, :, None]
        wage_weights = np.array(
            [row["wage_bill_only_weight"] for row in weight_rows]
        )
        energy_matrix = np.array(
            [
                [
                    row["energy_all_four_runs_mj"]
                    for row in prompts_by_occ[occupation]
                ]
                for occupation in occupations
            ]
        )
        boot_energy = energy_matrix[occ_axis, bootstrap_indices].sum(axis=2)
        threshold_result: dict[str, Any] = {}
        for pathway, spec in pathways.items():
            occ_rows = []
            for weight in weight_rows:
                rows = [
                    row
                    for row in qualifying
                    if row["occupation"] == weight["occupation"]
                ]
                hours = sum(float(row[spec["hours_field"]]) for row in rows)
                source_hours = sum(
                    float(row[spec["denominator_field"]]) for row in rows
                )
                energy = sum(
                    row["energy_all_four_runs_mj"]
                    for row in prompts
                    if row["occupation"] == weight["occupation"]
                )
                occ_rows.append(
                    {
                        "occupation": weight["occupation"],
                        "category": weight["category"],
                        "qualifying_prompts": len(rows),
                        "hours_saved_panel": hours,
                        "source_hours_panel": source_hours,
                        "energy_panel_mj": energy,
                        "gdp_structure_weight": weight["gdp_structure_weight"],
                        "wage_bill_only_weight": weight["wage_bill_only_weight"],
                    }
                )
            weighted_hours = sum(
                row["wage_bill_only_weight"] * row["hours_saved_panel"]
                for row in occ_rows
            )
            weighted_source = sum(
                row["wage_bill_only_weight"] * row["source_hours_panel"]
                for row in occ_rows
            )
            weighted_energy = sum(
                row["wage_bill_only_weight"] * row["energy_panel_mj"]
                for row in occ_rows
            )
            category_rows = []
            for category in CATEGORIES.values():
                rows = [row for row in occ_rows if row["category"] == category]
                cat_hours = sum(
                    row["gdp_structure_weight"] * row["hours_saved_panel"]
                    for row in rows
                )
                cat_source = sum(
                    row["gdp_structure_weight"] * row["source_hours_panel"]
                    for row in rows
                )
                cat_energy = sum(
                    row["gdp_structure_weight"] * row["energy_panel_mj"]
                    for row in rows
                )
                category_rows.append(
                    {
                        "category": category,
                        "gdp_structure_weight": sum(
                            row["gdp_structure_weight"] for row in rows
                        ),
                        "qualifying_prompts": sum(
                            row["qualifying_prompts"] for row in rows
                        ),
                        "weighted_hours_saved_per_representative_task": (
                            cat_hours / 5
                        ),
                        "weighted_hours_saved_220_task_equivalent_contribution": (
                            cat_hours * 44
                        ),
                        "weighted_percent_saved": (
                            cat_hours / cat_source if cat_source else None
                        ),
                        "hours_saved_per_mj_ratio_of_weighted_sums": (
                            cat_hours / cat_energy
                        ),
                    }
                )
            hours_matrix = np.array(
                [
                    [
                        (
                            float(row[spec["hours_field"]])
                            if row["success_count"] >= threshold
                            else 0.0
                        )
                        for row in prompts_by_occ[occupation]
                    ]
                    for occupation in occupations
                ]
            )
            source_matrix = np.array(
                [
                    [
                        (
                            float(row[spec["denominator_field"]])
                            if row["success_count"] >= threshold
                            else 0.0
                        )
                        for row in prompts_by_occ[occupation]
                    ]
                    for occupation in occupations
                ]
            )
            boot_hours = hours_matrix[occ_axis, bootstrap_indices].sum(axis=2)
            boot_source = source_matrix[occ_axis, bootstrap_indices].sum(axis=2)
            weighted_boot_hours = boot_hours @ wage_weights
            weighted_boot_source = boot_source @ wage_weights
            weighted_boot_energy = boot_energy @ wage_weights
            unweighted_boot_hours = boot_hours.sum(axis=1)
            unweighted_boot_source = boot_source.sum(axis=1)
            unweighted_boot_energy = boot_energy.sum(axis=1)
            threshold_result[pathway] = {
                "qualifying_prompts": len(qualifying),
                "wage_bill_weighted_hours_saved_per_representative_task": (
                    weighted_hours / 5
                ),
                "wage_bill_weighted_hours_saved_220_task_equivalent": (
                    weighted_hours * 44
                ),
                "wage_bill_weighted_percent_saved": (
                    weighted_hours / weighted_source
                    if weighted_source
                    else None
                ),
                "wage_bill_weighted_hours_saved_per_mj": (
                    weighted_hours / weighted_energy
                ),
                "unweighted_total_hours_saved": sum(
                    row["hours_saved_panel"] for row in occ_rows
                ),
                "unweighted_percent_saved": (
                    sum(row["hours_saved_panel"] for row in occ_rows)
                    / sum(row["source_hours_panel"] for row in occ_rows)
                    if sum(row["source_hours_panel"] for row in occ_rows)
                    else None
                ),
                "unweighted_hours_saved_per_mj": (
                    sum(row["hours_saved_panel"] for row in occ_rows)
                    / sum(row["energy_panel_mj"] for row in occ_rows)
                ),
                "bootstrap_95_ci": {
                    "wage_bill_weighted_hours_saved_per_representative_task": ci(
                        weighted_boot_hours / 5
                    ),
                    "wage_bill_weighted_hours_saved_220_task_equivalent": ci(
                        weighted_boot_hours * 44
                    ),
                    "wage_bill_weighted_percent_saved": ci(
                        weighted_boot_hours / weighted_boot_source
                    ),
                    "wage_bill_weighted_hours_saved_per_mj": ci(
                        weighted_boot_hours / weighted_boot_energy
                    ),
                    "unweighted_total_hours_saved": ci(
                        unweighted_boot_hours
                    ),
                    "unweighted_percent_saved": ci(
                        unweighted_boot_hours / unweighted_boot_source
                    ),
                    "unweighted_hours_saved_per_mj": ci(
                        unweighted_boot_hours / unweighted_boot_energy
                    ),
                },
                "categories": category_rows,
            }
        output[str(threshold)] = threshold_result
    return output


def calculate() -> dict[str, Any]:
    attempts, results_by_baseline, task_meta, batch_rows = load_base()
    weight_rows, weight_by_occ = load_weights(attempts)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = rng.integers(
        0, 5, size=(BOOTSTRAP_SAMPLES, 44, 5), dtype=np.int16
    )

    prompts_by_baseline: dict[str, list[dict[str, Any]]] = {}
    analyses: dict[str, Any] = {}
    for baseline in ("human", "human_ai"):
        prompts = build_prompts(
            attempts,
            results_by_baseline[baseline],
            baseline,
            weight_by_occ,
        )
        prompts_by_baseline[baseline] = prompts
        analyses[baseline] = {
            str(threshold): calculate_threshold(
                prompts,
                weight_rows,
                weight_by_occ,
                threshold,
                bootstrap_indices,
            )
            for threshold in range(1, 5)
        }

    allocated_energy_j = sum(row["energy_j"] for row in attempts)
    source_batch_energy_j = sum(
        float(row["gpu_energy_joules"]) for row in batch_rows
    )
    query_by_batch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        query_by_batch[row["batch_key"]].append(row)
    batch_reconciliation = []
    for batch in batch_rows:
        allocated = sum(
            row["energy_j"] for row in query_by_batch[batch["batch_key"]]
        )
        source = float(batch["gpu_energy_joules"])
        batch_reconciliation.append(
            {
                "batch_key": batch["batch_key"],
                "occupation": batch["occupation"],
                "attempts": len(query_by_batch[batch["batch_key"]]),
                "gpu_device_count": int(batch["gpu_device_count"]),
                "source_gpu_energy_j": source,
                "allocated_attempt_energy_j": allocated,
                "reconciliation_error_j": allocated - source,
            }
        )

    category_weights = []
    for category in CATEGORIES.values():
        rows = [row for row in weight_rows if row["category"] == category]
        category_weights.append(
            {
                "category": category,
                "occupation_count": len(rows),
                "prompt_count": len(rows) * 5,
                "gdp_structure_weight": sum(
                    row["gdp_structure_weight"] for row in rows
                ),
                "wage_bill_only_weight": sum(
                    row["wage_bill_only_weight"] for row in rows
                ),
            }
        )

    outcomes = {
        baseline: dict(
            Counter(
                row["agent_outcome"]
                for row in results_by_baseline[baseline].values()
            )
        )
        for baseline in ("human", "human_ai")
    }
    checks = {
        "prompt_count": len(task_meta),
        "attempt_count": len(attempts),
        "attempts_per_prompt": sorted(
            set(Counter(row["task_id"] for row in attempts).values())
        ),
        "occupation_count": len(weight_rows),
        "prompts_per_occupation": sorted(
            set(
                Counter(
                    row["occupation"]
                    for row in prompts_by_baseline["human"]
                ).values()
            )
        ),
        "category_count": len(category_weights),
        "sector_count": len(SECTOR_GDP_SHARES),
        "published_sector_share_sum": sum(SECTOR_GDP_SHARES.values()),
        "normalized_sector_share_sum": sum(
            value / sum(SECTOR_GDP_SHARES.values())
            for value in SECTOR_GDP_SHARES.values()
        ),
        "gdp_structure_weight_sum": sum(
            row["gdp_structure_weight"] for row in weight_rows
        ),
        "wage_bill_only_weight_sum": sum(
            row["wage_bill_only_weight"] for row in weight_rows
        ),
        "unweighted_occupation_weight_sum": 44 * (1 / 44),
        "exact_soc_matches": sum(
            row["soc_match_type"] == "exact SOC" for row in weight_rows
        ),
        "published_aggregate_crosswalks": sum(
            row["soc_match_type"] == "published aggregate crosswalk"
            for row in weight_rows
        ),
        "compensation_rounding_matches": sum(
            round(row["wage_bill_billions_usd"], 2)
            == row["gdpval_published_compensation_billions_usd"]
            for row in weight_rows
        ),
        "batch_count": len(batch_rows),
        "attempts_per_batch": sorted(
            set(len(value) for value in query_by_batch.values())
        ),
        "gpu_devices_per_batch": sorted(
            set(int(row["gpu_device_count"]) for row in batch_rows)
        ),
        "source_batch_energy_mj": source_batch_energy_j / 1_000_000,
        "allocated_attempt_energy_mj": allocated_energy_j / 1_000_000,
        "energy_reconciliation_error_j": (
            allocated_energy_j - source_batch_energy_j
        ),
        "missing_task_wage_time_energy_values": sum(
            not all(
                math.isfinite(float(row[field]))
                for field in (
                    "hourly_wage_usd",
                    "human_only_hours",
                    "human_ai_hours",
                    "human_only_value_usd",
                    "human_ai_value_usd",
                    "energy_j",
                )
            )
            for row in attempts
        ),
        "outcomes": outcomes,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    checks["passed"] = all(
        [
            checks["prompt_count"] == 220,
            checks["attempt_count"] == 880,
            checks["attempts_per_prompt"] == [4],
            checks["occupation_count"] == 44,
            checks["prompts_per_occupation"] == [5],
            checks["category_count"] == 13,
            checks["sector_count"] == 9,
            math.isclose(checks["gdp_structure_weight_sum"], 1, abs_tol=1e-12),
            math.isclose(checks["wage_bill_only_weight_sum"], 1, abs_tol=1e-12),
            checks["exact_soc_matches"] == 43,
            checks["published_aggregate_crosswalks"] == 1,
            checks["compensation_rounding_matches"] == 44,
            checks["batch_count"] == 44,
            checks["attempts_per_batch"] == [20],
            checks["gpu_devices_per_batch"] == [8],
            abs(checks["energy_reconciliation_error_j"]) < 1e-6,
            checks["missing_task_wage_time_energy_values"] == 0,
        ]
    )
    if not checks["passed"]:
        raise AssertionError(f"QA checks failed: {checks}")

    primary = analyses["human"]["1"]
    return {
        "energy_boundary": "GPU",
        "energy_components": ["gpu"],
        "methodology": {
            "study": "GLM-5.2 GDPVal GDP-structured value capture",
            "index_name": "GDPVal GDP-structured value-capture index (GVCI)",
            "interpretation": (
                "Percentage of economically weighted benchmark output dollar value "
                "captured by GLM; not a percentage of U.S. GDP automated."
            ),
            "success_definition": "Final Gemini win or tie",
            "primary_variant": "GLM vs the expert-only reference, at least 1 success in 4 runs",
            "thresholds": [1, 2, 3, 4],
            "task_value_rule": (
                "Recorded expert completion time multiplied by GDPVal's frozen occupation median "
                "hourly wage; each qualifying prompt is counted once."
            ),
            "weight_rule": (
                "Normalize GDPVal's nine published Q2 2024 sector shares, then "
                "allocate each sector share by May 2024 national OEWS occupation "
                "wage bill (TOT_EMP × A_MEAN; H_MEAN × 2,080 fallback)."
            ),
            "energy_rule": (
                "All four GLM runs for every prompt, including failed attempts; "
                "all GPUs summed; equal within-batch attempt allocation exactly "
                "reconciled to source batch energy."
            ),
            "bootstrap_rule": (
                "10,000 task-clustered samples, resampling five prompts with "
                "replacement within each of 44 occupations; fixed economic weights."
            ),
            "literature_status": (
                "Newly constructed scalar. GDPVal supplies the sector and "
                "compensation structure; Anthropic and task-based Hulten work "
                "motivate economic-share-weighted task aggregation."
            ),
            "source_urls": SOURCE_URLS,
        },
        "headline": {
            "baseline": "human",
            "threshold": 1,
            "gvci_percent": primary["global_indices"][
                "gdp_structure_gvci_percent"
            ],
            "gvci_95_ci": primary["bootstrap_95_ci"][
                "gdp_structure_gvci_percent"
            ],
            "wage_bill_only_percent": primary["global_indices"][
                "wage_bill_only_value_capture_percent"
            ],
            "unweighted_occupation_percent": primary["global_indices"][
                "unweighted_occupation_value_capture_percent"
            ],
        },
        "analyses": analyses,
        "three_way_time": calculate_three_way_time(
            prompts_by_baseline["human_ai"], weight_rows, bootstrap_indices
        ),
        "occupation_weights": weight_rows,
        "category_weights": category_weights,
        "prompt_ledgers": prompts_by_baseline,
        "batch_reconciliation": batch_reconciliation,
        "qa": checks,
        "source_inventory": {
            "query_ledger": {
                "path": str(QUERY_LEDGER.relative_to(ROOT)),
                "sha256": sha256(QUERY_LEDGER),
            },
            "batch_ledger": {
                "path": str(BATCH_LEDGER.relative_to(ROOT)),
                "sha256": sha256(BATCH_LEDGER),
            },
            "results": {
                "path": str(RESULTS.relative_to(ROOT)),
                "sha256": sha256(RESULTS),
            },
            "manifest": {
                "path": str(MANIFEST.relative_to(ROOT)),
                "sha256": sha256(MANIFEST),
            },
            "valuation_workbook": {
                "path": str(VALUATION_WORKBOOK.relative_to(ROOT)),
                "sha256": sha256(VALUATION_WORKBOOK),
            },
            "oews_zip": {
                "path": str(OEWS_ZIP.relative_to(ROOT)),
                "sha256": sha256(OEWS_ZIP),
            },
            "oews_workbook": {
                "path": str(OEWS_WORKBOOK.relative_to(ROOT)),
                "sha256": sha256(OEWS_WORKBOOK),
            },
        },
    }


def write_csvs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "occupation_weights.csv"
    with weights_path.open("w", newline="") as handle:
        rows = report["occupation_weights"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    category_path = output_dir / "category_results.csv"
    rows = []
    for baseline, thresholds in report["analyses"].items():
        for threshold, result in thresholds.items():
            for row in result["categories"]:
                rows.append({"baseline": baseline, **row})
    with category_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-dir", type=Path)
    args = parser.parse_args()
    report = calculate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.csv_dir:
        write_csvs(report, args.csv_dir)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "headline": report["headline"],
                "qa": report["qa"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
