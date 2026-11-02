#!/usr/bin/env python3
"""Numerical GDPVal aggregation and bootstrap calculations for Joules-to-GDP.

Inputs are the separately obtained, recorded task, outcome, energy, and wage
ledgers. This module emits machine-readable metrics only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260726
THRESHOLDS = (1, 2, 3, 4)

CATEGORIES = [
    "Management",
    "Business & Financial Operations",
    "Computer & Mathematical",
    "Architecture & Engineering",
    "Community & Social Service",
    "Legal",
    "Arts, Design, Entertainment & Media",
    "Healthcare Practitioners & Technical",
    "Protective Service",
    "Personal Care & Service",
    "Sales & Related",
    "Office & Administrative Support",
    "Production",
]

MODEL_CONFIGS = {
    "glm52_h200": {
        "display": "GLM-5.2",
        "system": "GLM-5.2 + OpenHands + 8×H200",
        "hardware": "8×NVIDIA H200",
        "prompt_source": "glm",
        "results": ROOT
        / "reports/gdpval_pairwise_gemini31_flash_lite_20260716/results.output_level.jsonl",
    },
    "qwen36_27b": {
        "display": "Qwen3.6-27B",
        "system": "Qwen3.6-27B source run",
        "hardware": "8 GPUs per source batch",
        "prompt_source": "qwen_xlsx",
        "results": ROOT
        / "reports/qwen36_27b_norubric_gemini31_flash_lite_20260723/results.output_level.jsonl",
        "source_outputs": ROOT
        / "reports/qwen36_27b_norubric_source_20260723/qwen_source_outputs.json",
    },
    "gemma4_31b": {
        "display": "Gemma 4 31B",
        "system": "Gemma 4 31B + OpenHands source run",
        "hardware": "8×NVIDIA H100",
        "prompt_source": "gemma_json",
        "results": ROOT
        / "reports/gemma4_31b_norubric_gemini31_flash_lite_20260725/results.output_level.jsonl",
        "source_outputs": ROOT
        / "reports/gemma4_31b_norubric_source_20260725/gemma_source_outputs.json",
    },
    "gpt_oss_120b": {
        "display": "GPT-OSS 120B",
        "system": "GPT-OSS 120B source run",
        "hardware": "8 GPUs per source batch",
        "prompt_source": "gpt_json",
        "model_key": "GPT-OSS 120B",
        "results": ROOT
        / "reports/gpt_oss_120b_norubric_gemini31_flash_lite_20260724/results.output_level.jsonl",
        "source_outputs": ROOT
        / "reports/gpt_oss_120b_norubric_source_20260724/gptoss_source_outputs.json",
    },
    "gpt_oss_20b": {
        "display": "GPT-OSS 20B",
        "system": "GPT-OSS 20B source run",
        "hardware": "8 GPUs per source batch",
        "prompt_source": "gpt_json",
        "model_key": "GPT-OSS 20B",
        "results": ROOT
        / "reports/gpt_oss_20b_norubric_gemini31_flash_lite_20260724/results.output_level.jsonl",
        "source_outputs": ROOT
        / "reports/gpt_oss_20b_norubric_source_20260724/gptoss_source_outputs.json",
    },
}

SOURCE_PATHS = {
    "weight_report": ROOT
    / "reports/glm52_norubric_gdp_structured_value_capture_20260724.json",
    "oews": ROOT / "occupation_data/data/input/OEWS.csv",
    "gdpval_tasks": ROOT / "GDPVal_220_Tasks/final_1-220_samples.json",
    "glm_query": ROOT
    / "outputs/019f86a0-e5e6-7e90-8578-b08ddb4385fb/glm52_no_rubric_query_ledger.csv",
    "glm_batch": ROOT
    / "outputs/019f86a0-e5e6-7e90-8578-b08ddb4385fb/glm52_gdpval_batch_ledger.csv",
    "qwen_workbook": ROOT
    / "outputs/019f9187-6eaf-7643-b1d4-931e16f4d844/qwen36_27b_gdpval_major_category_value_energy.xlsx",
    "gemma_report": ROOT
    / "reports/gemma4_31b_norubric_major_category_value_energy_20260725/gemma_value_energy_report.json",
    "gpt_report": ROOT
    / "reports/gpt_oss_120b_20b_major_category_value_energy_20260724.json",
    "gemini_summary": ROOT
    / "reports/gdpval_pairwise_gemini31_flash_lite_20260716/provider_run_summary.json",
    "gemini_parse": ROOT
    / "reports/gdpval_pairwise_gemini31_flash_lite_20260716/result_parse_summary.json",
    "gemini_frozen_audit": ROOT
    / "reports/gdpval_pairwise_gemini31_flash_lite_20260716/frozen_input_audit.json",
}

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

def finite(value: Any, label: str) -> float:
    result = float(value)
    require(math.isfinite(result), f"{label} is not finite: {value!r}")
    return result

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]

def write_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Cannot serialize {type(item).__name__}")

    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=convert) + "\n"
    )

def parse_numeric(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text in {"", "*", "**", "#", "NA", "N/A"}:
        return math.nan
    return float(text)

def excel_column_index(reference: str) -> int:
    letters = re.match(r"([A-Z]+)", reference)
    require(letters is not None, f"Invalid Excel cell reference: {reference}")
    value = 0
    for character in letters.group(1):
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1

def read_xlsx_sheet(path: Path, sheet_name: str) -> list[list[Any]]:
    """Read cached cell values from a simple XLSX worksheet without openpyxl."""

    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = {
        "r": "http://schemas.openxmlformats.org/package/2006/relationships"
    }
    office_rel = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    )
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relationship_map = {
            item.attrib["Id"]: item.attrib["Target"].lstrip("/")
            for item in rels.findall("r:Relationship", rel_ns)
        }
        target = None
        for sheet in workbook.findall("x:sheets/x:sheet", ns):
            if sheet.attrib["name"] == sheet_name:
                target = relationship_map[
                    sheet.attrib[f"{{{office_rel}}}id"]
                ]
                break
        require(target is not None, f"Worksheet not found: {sheet_name}")
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("x:si", ns):
                shared.append(
                    "".join(node.text or "" for node in item.iterfind(".//x:t", ns))
                )
        sheet_root = ET.fromstring(archive.read(target))
        output: list[list[Any]] = []
        for row in sheet_root.findall("x:sheetData/x:row", ns):
            row_number = int(row.attrib["r"])
            while len(output) < row_number - 1:
                output.append([])
            cells: dict[int, Any] = {}
            max_index = -1
            for cell in row.findall("x:c", ns):
                index = excel_column_index(cell.attrib["r"])
                max_index = max(max_index, index)
                cell_type = cell.attrib.get("t")
                value_node = cell.find("x:v", ns)
                inline_node = cell.find("x:is", ns)
                if inline_node is not None:
                    value: Any = "".join(
                        node.text or "" for node in inline_node.iterfind(".//x:t", ns)
                    )
                elif value_node is None or value_node.text is None:
                    value = None
                elif cell_type in {"str", "inlineStr"}:
                    value = value_node.text
                elif cell_type == "s":
                    value = shared[int(value_node.text)]
                elif cell_type == "b":
                    value = value_node.text == "1"
                else:
                    value = parse_numeric(value_node.text)
                cells[index] = value
            if max_index >= 0:
                output.append([cells.get(index) for index in range(max_index + 1)])
        return output

def xlsx_table(path: Path, sheet_name: str, header_row: int) -> pd.DataFrame:
    rows = read_xlsx_sheet(path, sheet_name)
    require(len(rows) >= header_row, f"{sheet_name}: missing header row")
    headers = [str(value) if value is not None else "" for value in rows[header_row - 1]]
    data: list[list[Any]] = []
    for row in rows[header_row:]:
        padded = row + [None] * (len(headers) - len(row))
        if any(value is not None for value in padded):
            data.append(padded[: len(headers)])
    return pd.DataFrame(data, columns=headers)

def uncertainty(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    require(np.isfinite(values).all(), "Non-finite bootstrap sample")
    return {
        "bootstrap_mean": float(values.mean()),
        "bootstrap_sd": float(values.std(ddof=1)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }

def ranks(values: np.ndarray, descending: bool = True) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(-values if descending else values, axis=-1, kind="mergesort")
    result = np.empty_like(order, dtype=float)
    if values.ndim == 1:
        result[order] = np.arange(1, len(values) + 1)
    else:
        rows = np.arange(values.shape[0])[:, None]
        result[rows, order] = np.arange(1, values.shape[1] + 1)[None, :]
    return result

@dataclass
class AnalysisInputs:
    task_meta: pd.DataFrame
    weights: pd.DataFrame
    models: dict[str, pd.DataFrame]
    attempts: dict[tuple[str, str], pd.DataFrame]
    occupation_order: list[str]
    category_index_by_occupation: np.ndarray
    weight_vector: np.ndarray
    bootstrap_draws: np.ndarray
    source_inventory: list[dict[str, Any]]
    human_ai_model_counts: Counter

def load_task_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    report = read_json(SOURCE_PATHS["weight_report"])
    weights = pd.DataFrame(report["occupation_weights"]).copy()
    require(len(weights) == 44, "Expected 44 occupation weights")
    # Preserve each sampled sector's GDP share, but distribute that mass among
    # its sampled occupations by employment. Wages enter task valuation only.
    weights["legacy_wage_bill_weight"] = weights["gdp_structure_weight"]
    weights["within_sector_employment_share"] = (
        weights["national_employment"]
        / weights.groupby("sector")["national_employment"].transform("sum")
    )
    weights["gdp_structure_weight"] = (
        weights["sector_gdp_share_normalized"]
        * weights["within_sector_employment_share"]
    )
    require(abs(weights["gdp_structure_weight"].sum() - 1.0) < 1e-12, "Weight sum")
    sector_check = weights.groupby("sector").agg(
        assigned=("gdp_structure_weight", "sum"),
        target=("sector_gdp_share_normalized", "first"),
    )
    require(
        np.allclose(sector_check["assigned"], sector_check["target"], rtol=0, atol=1e-12),
        "Sector GDP shares were not preserved",
    )
    require(set(weights["category"]) == set(CATEGORIES), "Category coverage in weights")

    prompts = pd.DataFrame(report["prompt_ledgers"]["human"]).copy()
    require(len(prompts) == 220, "Expected 220 expert-only reference prompt records")
    prompts = prompts[
        [
            "task_id",
            "baseline_task_id",
            "occupation",
            "onet_code",
            "category",
            "sector",
            "human_only_hours",
            "human_ai_hours",
            "hourly_wage_usd",
        ]
    ].copy()
    prompts = prompts.rename(columns={"hourly_wage_usd": "mean_hourly_wage_usd"})

    oews = pd.read_csv(SOURCE_PATHS["oews"], dtype=str)
    oews = oews[(oews["AREA"] == "99") & (oews["NAICS"] == "000000")].copy()
    oews["median_hourly_wage_usd"] = oews["H_MEDIAN"].map(parse_numeric)
    oews["mean_hourly_wage_check"] = oews["H_MEAN"].map(parse_numeric)
    median_by_soc = (
        oews.dropna(subset=["median_hourly_wage_usd"])
        .drop_duplicates("OCC_CODE")
        .set_index("OCC_CODE")["median_hourly_wage_usd"]
        .to_dict()
    )
    weight_by_occupation = weights.set_index("occupation").to_dict(orient="index")
    median_wages = []
    oews_codes = []
    for row in prompts.itertuples():
        mapping = weight_by_occupation[row.occupation]
        code = mapping["oews_soc_code"]
        require(code in median_by_soc, f"Missing OEWS median wage for {row.occupation}")
        median_wages.append(median_by_soc[code])
        oews_codes.append(code)
    prompts["oews_soc_code"] = oews_codes
    prompts["median_hourly_wage_usd"] = median_wages
    require(prompts["median_hourly_wage_usd"].notna().all(), "Missing median wages")
    require(prompts["human_only_hours"].notna().all(), "Missing recorded expert completion time")
    require(prompts["human_ai_hours"].notna().all(), "Missing recorded expert-with-AI completion time")
    require(prompts.groupby("occupation").size().eq(5).all(), "Five prompts/occupation")
    require(prompts["task_id"].nunique() == 220, "Unique task IDs")
    return prompts, weights

def filtered_results(path: Path, baseline: str) -> pd.DataFrame:
    rows = pd.DataFrame(read_jsonl(path))
    rows = rows[
        (rows["condition"] == "without_rubric") & (rows["baseline"] == baseline)
    ].copy()
    require(len(rows) == 880, f"{path.name}: expected 880 {baseline} judgments")
    require(rows[["task_id", "repeat"]].drop_duplicates().shape[0] == 880, "Unique verdicts")
    require(set(rows["agent_outcome"]) <= {"win", "tie", "loss"}, "Outcome vocabulary")
    rows["repeat"] = rows["repeat"].astype(int)
    rows["success"] = rows["agent_outcome"].isin(["win", "tie"])
    return rows

def load_glm(
    task_meta: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[tuple[str, str], pd.DataFrame]]:
    query = pd.read_csv(SOURCE_PATHS["glm_query"])
    query = query[query["dataset"] == "GDPVal"].copy()
    require(len(query) == 880, "GLM query count")
    query["repeat"] = query["repeat_index"].astype(int)
    query["task_id"] = query["judge_task_id"].astype(str)
    query["latency_hours"] = query["agent_latency_hours"].astype(float)
    query["energy_mj"] = query["allocated_gpu_energy_joules"].astype(float) / 1e6
    require(query[["task_id", "repeat"]].drop_duplicates().shape[0] == 880, "GLM attempts")
    require(query[["latency_hours", "energy_mj"]].notna().all().all(), "GLM telemetry")

    batch = pd.read_csv(SOURCE_PATHS["glm_batch"])
    batch = batch[batch["dataset"] == "GDPVal"].copy()
    source_energy_mj = batch["gpu_energy_joules"].astype(float).sum() / 1e6
    require(abs(query["energy_mj"].sum() - source_energy_mj) < 1e-9, "GLM energy reconciliation")

    prompts: dict[str, pd.DataFrame] = {}
    attempts: dict[tuple[str, str], pd.DataFrame] = {}
    for baseline in ("human", "human_ai"):
        verdict = filtered_results(MODEL_CONFIGS["glm52_h200"]["results"], baseline)
        merged = verdict.merge(
            query[["task_id", "repeat", "latency_hours", "energy_mj"]],
            on=["task_id", "repeat"],
            how="left",
            validate="one_to_one",
        )
        require(merged[["latency_hours", "energy_mj"]].notna().all().all(), "GLM join")
        grouped_rows = []
        for task_id, task_attempts in merged.groupby("task_id"):
            successful = task_attempts[task_attempts["success"]]
            grouped_rows.append(
                {
                    "task_id": task_id,
                    "success_count": int(successful.shape[0]),
                    "mean_success_latency_hours": (
                        float(successful["latency_hours"].mean())
                        if len(successful)
                        else math.nan
                    ),
                    "energy_all_four_runs_mj": float(task_attempts["energy_mj"].sum()),
                }
            )
        prompt = task_meta.merge(
            pd.DataFrame(grouped_rows), on="task_id", how="left", validate="one_to_one"
        )
        require(len(prompt) == 220, f"GLM {baseline} prompts")
        prompts[baseline] = prompt
        attempts[("glm52_h200", baseline)] = merged
    require(
        np.allclose(
            prompts["human"]["energy_all_four_runs_mj"],
            prompts["human_ai"]["energy_all_four_runs_mj"],
        ),
        "GLM baseline energy mismatch",
    )
    return prompts, attempts

def standardize_model_prompt(
    raw: pd.DataFrame,
    task_meta: pd.DataFrame,
    task_column: str,
    success_column: str,
    latency_column: str,
    energy_by_task: dict[str, float] | None = None,
    energy_column: str | None = None,
) -> pd.DataFrame:
    fields = pd.DataFrame(
        {
            "task_id": raw[task_column].astype(str),
            "reported_success_count": raw[success_column].astype(int),
            "mean_success_latency_hours": pd.to_numeric(raw[latency_column], errors="coerce"),
        }
    )
    if energy_by_task is not None:
        fields["energy_all_four_runs_mj"] = fields["task_id"].map(energy_by_task)
    else:
        require(energy_column is not None, "Energy source required")
        fields["energy_all_four_runs_mj"] = pd.to_numeric(
            raw[energy_column], errors="raise"
        )
    result = task_meta.merge(fields, on="task_id", how="left", validate="one_to_one")
    require(len(result) == 220, "Model prompt count")
    require(result["reported_success_count"].notna().all(), "Prompt join completeness")
    require(result["energy_all_four_runs_mj"].notna().all(), "Prompt energy completeness")
    return result

def attempt_latency_from_source_outputs(
    source_path: Path,
    verdicts: pd.DataFrame,
) -> pd.DataFrame:
    source_rows = read_json(source_path)
    latency_rows = []
    for source in source_rows:
        metadata = read_json(Path(source["metadata_path"]))
        latency_rows.append(
            {
                "task_id": str(source["task_id"]),
                "repeat": int(source["repeat"]),
                "latency_hours": finite(metadata["wall_clock_s"], "wall clock") / 3600,
            }
        )
    latency = pd.DataFrame(latency_rows)
    require(len(latency) == 880, f"{source_path.name}: latency count")
    merged = verdicts.merge(
        latency, on=["task_id", "repeat"], how="left", validate="one_to_one"
    )
    require(merged["latency_hours"].notna().all(), "Model latency join")
    return merged

def load_other_models(
    task_meta: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[tuple[str, str], pd.DataFrame]]:
    models: dict[str, pd.DataFrame] = {}
    attempts: dict[tuple[str, str], pd.DataFrame] = {}

    qwen_prompt = xlsx_table(SOURCE_PATHS["qwen_workbook"], "Prompt Ledger", 4)
    qwen = standardize_model_prompt(
        qwen_prompt,
        task_meta,
        "Qwen task ID",
        "W/T success count",
        "Mean successful Qwen latency (hours)",
        energy_column="Energy, all 4 attempts (MJ)",
    )
    qwen_verdict = filtered_results(MODEL_CONFIGS["qwen36_27b"]["results"], "human")
    qwen_counts = qwen_verdict.groupby("task_id")["success"].sum().astype(int)
    qwen_reported = qwen.set_index("task_id")["reported_success_count"].sort_index()
    qwen_counts = qwen_counts.reindex(qwen_reported.index)
    require(
        qwen_counts.notna().all() and (qwen_reported == qwen_counts).all(),
        "Qwen verdict/workbook success mismatch",
    )
    qwen["success_count"] = qwen["reported_success_count"].astype(int)
    models["qwen36_27b"] = qwen
    attempts[("qwen36_27b", "human")] = attempt_latency_from_source_outputs(
        MODEL_CONFIGS["qwen36_27b"]["source_outputs"], qwen_verdict
    )

    gemma_report = read_json(SOURCE_PATHS["gemma_report"])
    gemma_attempt = pd.DataFrame(gemma_report["attempt_ledger"])
    gemma_energy = (
        gemma_attempt.groupby(gemma_attempt["agent_task_id"].astype(str))[
            "allocated_attempt_energy_mj"
        ]
        .sum()
        .to_dict()
    )
    gemma_prompt = pd.DataFrame(gemma_report["prompt_ledger"])
    gemma = standardize_model_prompt(
        gemma_prompt,
        task_meta,
        "agent_task_id",
        "success_count",
        "mean_success_latency_hours",
        energy_by_task=gemma_energy,
    )
    gemma_verdict = filtered_results(MODEL_CONFIGS["gemma4_31b"]["results"], "human")
    gemma_counts = gemma_verdict.groupby("task_id")["success"].sum().astype(int)
    gemma_reported = gemma.set_index("task_id")["reported_success_count"].sort_index()
    gemma_counts = gemma_counts.reindex(gemma_reported.index)
    require(
        gemma_counts.notna().all() and (gemma_reported == gemma_counts).all(),
        "Gemma verdict/report success mismatch",
    )
    gemma["success_count"] = gemma["reported_success_count"].astype(int)
    models["gemma4_31b"] = gemma
    attempts[("gemma4_31b", "human")] = attempt_latency_from_source_outputs(
        MODEL_CONFIGS["gemma4_31b"]["source_outputs"], gemma_verdict
    )

    gpt_report = read_json(SOURCE_PATHS["gpt_report"])
    for model_id in ("gpt_oss_120b", "gpt_oss_20b"):
        config = MODEL_CONFIGS[model_id]
        source = gpt_report["models"][config["model_key"]]
        source_attempt = pd.DataFrame(source["attempt_ledger"])
        energy = (
            source_attempt.groupby(source_attempt["agent_task_id"].astype(str))[
                "allocated_attempt_energy_mj"
            ]
            .sum()
            .to_dict()
        )
        prompt = standardize_model_prompt(
            pd.DataFrame(source["prompt_ledger"]),
            task_meta,
            "agent_task_id",
            "success_count",
            "mean_success_latency_hours",
            energy_by_task=energy,
        )
        verdict = filtered_results(config["results"], "human")
        verdict_counts = verdict.groupby("task_id")["success"].sum().astype(int)
        reported_counts = (
            prompt.set_index("task_id")["reported_success_count"].sort_index()
        )
        verdict_counts = verdict_counts.reindex(reported_counts.index)
        require(
            verdict_counts.notna().all()
            and (reported_counts == verdict_counts).all(),
            f"{model_id} verdict/report success mismatch",
        )
        prompt["success_count"] = prompt["reported_success_count"].astype(int)
        models[model_id] = prompt
        attempts[(model_id, "human")] = attempt_latency_from_source_outputs(
            config["source_outputs"], verdict
        )
    return models, attempts

def source_inventory() -> list[dict[str, Any]]:
    paths = dict(SOURCE_PATHS)
    for model_id, config in MODEL_CONFIGS.items():
        paths[f"{model_id}_results"] = config["results"]
        if "source_outputs" in config:
            paths[f"{model_id}_source_outputs"] = config["source_outputs"]
    inventory = []
    for label, path in sorted(paths.items()):
        require(Path(path).is_file(), f"Missing source: {path}")
        inventory.append(
            {
                "source": label,
                "path": str(Path(path).relative_to(ROOT)),
                "size_bytes": Path(path).stat().st_size,
                "sha256": sha256_file(Path(path)),
            }
        )
    return inventory

def load_inputs(samples: int, seed: int) -> AnalysisInputs:
    task_meta, weights = load_task_metadata()
    glm_prompts, glm_attempts = load_glm(task_meta)
    other_models, other_attempts = load_other_models(task_meta)
    models = {"glm52_h200": glm_prompts["human"], **other_models}
    attempts = {**glm_attempts, **other_attempts}
    models["glm52_h200_human_ai"] = glm_prompts["human_ai"]
    component_energy = recorded_attempt_energy()
    if component_energy is not None:
        for (model_id, baseline), frame in attempts.items():
            source = component_energy[component_energy.profile.eq(model_id)][
                ["task_id", "repeat", "total_energy_j"]
            ]
            if len(source) != len(frame):
                raise ValueError(f"{model_id}/{baseline}: incomplete total-energy attempt ledger")
            merged = frame.drop(columns=["energy_mj"], errors="ignore").merge(
                source, on=["task_id", "repeat"], how="left", validate="one_to_one"
            )
            if merged.total_energy_j.isna().any():
                raise ValueError(f"{model_id}/{baseline}: missing GPU+CPU+DRAM energy")
            merged["energy_mj"] = merged.pop("total_energy_j") / 1e6
            attempts[(model_id, baseline)] = merged
        for model_id, frame in models.items():
            profile = "glm52_h200" if model_id == "glm52_h200_human_ai" else model_id
            totals = component_energy[component_energy.profile.eq(profile)].groupby(
                "task_id"
            ).total_energy_j.sum() / 1e6
            frame["energy_all_four_runs_mj"] = frame.task_id.map(totals)
            if frame.energy_all_four_runs_mj.isna().any():
                raise ValueError(f"{model_id}: missing total-energy task values")

    occupation_order = sorted(weights["occupation"].tolist())
    category_lookup = weights.set_index("occupation")["category"].to_dict()
    category_index_by_occupation = np.array(
        [CATEGORIES.index(category_lookup[occupation]) for occupation in occupation_order],
        dtype=int,
    )
    weight_lookup = weights.set_index("occupation")["gdp_structure_weight"].to_dict()
    weight_vector = np.array(
        [weight_lookup[occupation] for occupation in occupation_order], dtype=float
    )
    rng = np.random.default_rng(seed)
    bootstrap_draws = rng.integers(
        0, 5, size=(samples, len(occupation_order), 5), dtype=np.int16
    )

    gdpval_tasks = read_json(SOURCE_PATHS["gdpval_tasks"])
    model_counts = Counter(
        row["estimation_tasks"]["solved_with_ai"]["ai_environment"]["ai_model_version"]
        for row in gdpval_tasks
    )
    require(sum(model_counts.values()) == len(gdpval_tasks), "Expert-with-AI model mix coverage")
    return AnalysisInputs(
        task_meta=task_meta,
        weights=weights,
        models=models,
        attempts=attempts,
        occupation_order=occupation_order,
        category_index_by_occupation=category_index_by_occupation,
        weight_vector=weight_vector,
        bootstrap_draws=bootstrap_draws,
        source_inventory=source_inventory(),
        human_ai_model_counts=model_counts,
    )


def recorded_attempt_energy() -> pd.DataFrame | None:
    mode = os.environ.get("J2G_ENERGY_BOUNDARY", "total")
    if mode == "accelerator":
        return None
    if mode != "total":
        raise ValueError(f"Unknown J2G_ENERGY_BOUNDARY={mode!r}")
    directory = str(Path(__file__).resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    from energy_boundary import attempt_energy

    return attempt_energy()

def prompt_matrix(frame: pd.DataFrame, occupation_order: list[str], column: str) -> np.ndarray:
    rows = []
    for occupation in occupation_order:
        occupation_rows = frame[frame["occupation"] == occupation].sort_values("task_id")
        require(len(occupation_rows) == 5, f"{occupation}: expected five prompts")
        rows.append(pd.to_numeric(occupation_rows[column], errors="coerce").to_numpy(float))
    result = np.vstack(rows)
    return result

def point_metrics(
    prompt: pd.DataFrame,
    weights: pd.DataFrame,
    threshold: int,
    baseline_hours_column: str,
    wage_column: str,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    frame = prompt.copy()
    frame["credit_mask"] = frame["success_count"] >= threshold
    frame["baseline_hours"] = frame[baseline_hours_column].astype(float)
    frame["task_value"] = frame["baseline_hours"] * frame[wage_column].astype(float)
    frame["generated_value"] = frame["credit_mask"] * frame["task_value"]
    frame["hours_saved"] = np.where(
        frame["credit_mask"],
        frame["baseline_hours"] - frame["mean_success_latency_hours"],
        0.0,
    )
    weight_lookup = weights.set_index("occupation")["gdp_structure_weight"]
    frame["gdp_structure_weight"] = frame["occupation"].map(weight_lookup)
    occupation_rows = []
    for occupation, rows in frame.groupby("occupation"):
        weight = finite(rows["gdp_structure_weight"].iloc[0], "occupation weight")
        captured = float(rows["generated_value"].sum())
        occupation_rows.append(
            {
                "occupation": occupation,
                "category": rows["category"].iloc[0],
                "gdp_structure_weight": weight,
                "n_unique_prompts": len(rows),
                "attempts": 4 * len(rows),
                "attempt_win_or_tie_rate": float(rows["success_count"].sum() / (4 * len(rows))),
                "generated_value_panel_usd": captured,
                "hours_saved_panel": float(rows["hours_saved"].sum()),
                "energy_panel_mj": float(rows["energy_all_four_runs_mj"].sum()),
            }
        )
    occupations = pd.DataFrame(occupation_rows)
    category_rows = []
    for category in CATEGORIES:
        rows = occupations[occupations["category"] == category]
        category_weight = float(rows["gdp_structure_weight"].sum())
        conditional_weight = rows["gdp_structure_weight"] / category_weight
        # Each occupation panel contains five tasks. Category outcomes describe
        # one representative task under the category's occupation mix, with the
        # conditional occupation weights normalized to one inside the category.
        generated = float(
            (conditional_weight * rows["generated_value_panel_usd"] / 5).sum()
        )
        energy = float(
            (conditional_weight * rows["energy_panel_mj"] / 5).sum()
        )
        hours = float(
            (conditional_weight * rows["hours_saved_panel"] / 5).sum()
        )
        category_rows.append(
            {
                "occupation_category": category,
                "n_unique_prompts": int(rows["n_unique_prompts"].sum()),
                "attempts": int(rows["attempts"].sum()),
                "attempt_win_or_tie_rate": float(
                    (conditional_weight * rows["attempt_win_or_tie_rate"]).sum()
                ),
                "gdp_structure_weight": category_weight,
                "generated_value_usd": generated,
                "energy_mj": energy,
                "value_per_mj": generated / energy,
                "hours_saved": hours,
                "hours_per_mj": hours / energy,
                "raw_energy_mj": float(
                    frame[frame["category"] == category]["energy_all_four_runs_mj"].sum()
                ),
            }
        )
    categories = pd.DataFrame(category_rows)
    for field, descending in (
        ("generated_value_usd", True),
        ("energy_mj", False),
        ("value_per_mj", True),
        ("hours_saved", True),
        ("hours_per_mj", True),
    ):
        categories[f"{field}_rank"] = ranks(categories[field].to_numpy(), descending).astype(int)

    credited = frame[frame["credit_mask"]]
    global_metrics = {
        "threshold": threshold,
        "n_unique_prompts": 220,
        "attempts": 880,
        "attempt_win_or_tie_rate": float(frame["success_count"].sum() / 880),
        # Each occupation panel contains five tasks. Dividing by five first
        # makes the global outcome a GDP-weighted representative task: the
        # occupation weights sum to one globally, exactly as the category
        # weights do in the two-level aggregation.
        "generated_value_usd": float(
            (
                occupations["gdp_structure_weight"]
                * occupations["generated_value_panel_usd"]
            ).sum()
            / 5
        ),
        "energy_mj": float(
            (
                occupations["gdp_structure_weight"]
                * occupations["energy_panel_mj"]
            ).sum()
            / 5
        ),
        "hours_saved": float(
            (
                occupations["gdp_structure_weight"]
                * occupations["hours_saved_panel"]
            ).sum()
            / 5
        ),
        "raw_generated_value_usd": float(credited["task_value"].sum()),
        "raw_energy_mj": float(frame["energy_all_four_runs_mj"].sum()),
        "raw_hours_saved": float(credited["hours_saved"].sum()),
    }
    global_metrics["value_per_mj"] = (
        global_metrics["generated_value_usd"] / global_metrics["energy_mj"]
    )
    global_metrics["hours_per_mj"] = (
        global_metrics["hours_saved"] / global_metrics["energy_mj"]
    )
    global_metrics["raw_value_per_mj"] = (
        global_metrics["raw_generated_value_usd"] / global_metrics["raw_energy_mj"]
    )
    global_metrics["raw_hours_per_mj"] = (
        global_metrics["raw_hours_saved"] / global_metrics["raw_energy_mj"]
    )
    return categories, global_metrics, occupations

def bootstrap_metrics(
    inputs: AnalysisInputs,
    prompt: pd.DataFrame,
    threshold: int,
    baseline_hours_column: str,
    wage_column: str,
    point_categories: pd.DataFrame,
    point_global: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    draws = inputs.bootstrap_draws
    success_count = prompt_matrix(prompt, inputs.occupation_order, "success_count")
    latency = prompt_matrix(prompt, inputs.occupation_order, "mean_success_latency_hours")
    baseline_hours = prompt_matrix(prompt, inputs.occupation_order, baseline_hours_column)
    wage = prompt_matrix(prompt, inputs.occupation_order, wage_column)
    energy = prompt_matrix(prompt, inputs.occupation_order, "energy_all_four_runs_mj")
    credit_mask = success_count >= threshold
    safe_latency = np.nan_to_num(latency, nan=0.0)
    task_value = baseline_hours * wage
    generated = credit_mask * task_value
    saved = credit_mask * (baseline_hours - safe_latency)

    def sampled_panel(array: np.ndarray) -> np.ndarray:
        expanded = np.broadcast_to(array, (draws.shape[0],) + array.shape)
        return np.take_along_axis(expanded, draws, axis=2).sum(axis=2)

    generated_panel = sampled_panel(generated)
    saved_panel = sampled_panel(saved)
    energy_panel = sampled_panel(energy)
    success_panel = sampled_panel(success_count)

    weighted_generated = generated_panel * inputs.weight_vector[None, :]
    weighted_saved = saved_panel * inputs.weight_vector[None, :]
    weighted_energy = energy_panel * inputs.weight_vector[None, :]

    indicator = np.eye(len(CATEGORIES))[inputs.category_index_by_occupation]
    category_weight = inputs.weight_vector @ indicator
    category_scale = 5 * category_weight[None, :]
    category_generated = (weighted_generated @ indicator) / category_scale
    category_saved = (weighted_saved @ indicator) / category_scale
    category_energy = (weighted_energy @ indicator) / category_scale
    category_value_per_mj = category_generated / category_energy
    category_hours_per_mj = category_saved / category_energy
    category_attempt_rate = (
        (success_panel * inputs.weight_vector[None, :]) @ indicator
    ) / (
        20 * category_weight[None, :]
    )

    arrays = {
        "attempt_win_or_tie_rate": category_attempt_rate,
        "generated_value_usd": category_generated,
        "energy_mj": category_energy,
        "value_per_mj": category_value_per_mj,
        "hours_saved": category_saved,
        "hours_per_mj": category_hours_per_mj,
    }
    rank_arrays = {
        "generated_value_usd_rank": ranks(category_generated, True),
        "energy_mj_rank": ranks(category_energy, False),
        "value_per_mj_rank": ranks(category_value_per_mj, True),
        "hours_saved_rank": ranks(category_saved, True),
        "hours_per_mj_rank": ranks(category_hours_per_mj, True),
    }

    uncertainty_rows = []
    for category_index, category in enumerate(CATEGORIES):
        row: dict[str, Any] = {
            "occupation_category": category,
            "n_unique_prompts": int(
                point_categories.loc[
                    point_categories["occupation_category"] == category,
                    "n_unique_prompts",
                ].iloc[0]
            ),
        }
        for field, values in arrays.items():
            observed = float(
                point_categories.loc[
                    point_categories["occupation_category"] == category, field
                ].iloc[0]
            )
            row[f"{field}_estimate"] = observed
            for label, value in uncertainty(values[:, category_index]).items():
                row[f"{field}_{label}"] = value
        for field, values in rank_arrays.items():
            observed = int(
                point_categories.loc[
                    point_categories["occupation_category"] == category, field
                ].iloc[0]
            )
            row[f"{field}_observed"] = observed
            row[f"{field}_bootstrap_mean"] = float(values[:, category_index].mean())
            row[f"{field}_bootstrap_sd"] = float(
                values[:, category_index].std(ddof=1)
            )
        uncertainty_rows.append(row)
    category_uncertainty = pd.DataFrame(uncertainty_rows)

    global_arrays = {
        "attempt_win_or_tie_rate": success_panel.sum(axis=1) / 880,
        "generated_value_usd": weighted_generated.sum(axis=1) / 5,
        "energy_mj": weighted_energy.sum(axis=1) / 5,
        "value_per_mj": weighted_generated.sum(axis=1)
        / weighted_energy.sum(axis=1),
        "hours_saved": weighted_saved.sum(axis=1) / 5,
        "hours_per_mj": weighted_saved.sum(axis=1)
        / weighted_energy.sum(axis=1),
    }
    global_uncertainty: dict[str, dict[str, float]] = {}
    for field, values in global_arrays.items():
        global_uncertainty[field] = {
            "estimate": float(point_global[field]),
            **uncertainty(values),
        }
    return category_uncertainty, global_uncertainty

def combine_point_uncertainty(
    point: pd.DataFrame, uncertainty_frame: pd.DataFrame
) -> pd.DataFrame:
    result = point.merge(
        uncertainty_frame, on=["occupation_category", "n_unique_prompts"], how="left"
    )
    ordered = [
        "occupation_category",
        "n_unique_prompts",
        "attempts",
        "attempt_win_or_tie_rate",
        "gdp_structure_weight",
        "generated_value_usd",
        "energy_mj",
        "value_per_mj",
        "hours_saved",
        "hours_per_mj",
    ]
    rank_fields = [
        "generated_value_usd_rank",
        "energy_mj_rank",
        "value_per_mj_rank",
        "hours_saved_rank",
        "hours_per_mj_rank",
    ]
    extra = [
        column
        for column in result.columns
        if column not in ordered + rank_fields + ["raw_energy_mj"]
    ]
    return result[ordered + rank_fields + extra]

def global_threshold_analyses(
    inputs: AnalysisInputs,
    prompt: pd.DataFrame,
    baseline: str,
    baseline_hours_column: str,
    wage_column: str,
) -> tuple[pd.DataFrame, dict[int, tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]]]:
    rows = []
    details = {}
    for threshold in THRESHOLDS:
        point_category, point_global, point_occupation = point_metrics(
            prompt, inputs.weights, threshold, baseline_hours_column, wage_column
        )
        category_uncertainty, global_uncertainty = bootstrap_metrics(
            inputs,
            prompt,
            threshold,
            baseline_hours_column,
            wage_column,
            point_category,
            point_global,
        )
        row: dict[str, Any] = {
            "baseline": baseline,
            "threshold": f"≥{threshold}/4" if threshold < 4 else "4/4",
            "n_unique_prompts": 220,
            "attempts": 880,
        }
        for field in (
            "attempt_win_or_tie_rate",
            "generated_value_usd",
            "energy_mj",
            "value_per_mj",
            "hours_saved",
            "hours_per_mj",
        ):
            stats = global_uncertainty[field]
            for key, value in stats.items():
                row[f"{field}_{key}"] = value
        for field in (
            "raw_generated_value_usd",
            "raw_energy_mj",
            "raw_value_per_mj",
            "raw_hours_saved",
            "raw_hours_per_mj",
        ):
            row[field] = point_global[field]
        rows.append(row)
        details[threshold] = (point_category, category_uncertainty, point_global)
    return pd.DataFrame(rows), details

def passk_global_analysis(
    inputs: AnalysisInputs,
    baseline: str,
    baseline_hours_column: str,
    wage_column: str,
) -> pd.DataFrame:
    """Compute GDP-level pass@k outcomes from the first k recorded runs.

    Unlike the four-run consistency analysis, both the success set and the
    energy denominator change with k. A task receives value once if any of its
    first k runs succeeds, while energy includes every one of those k runs.
    """

    attempts = inputs.attempts[("glm52_h200", baseline)].copy()
    attempts["repeat"] = attempts["repeat"].astype(int)
    require(set(attempts["repeat"]) == {0, 1, 2, 3}, "Expected GLM repeats 0--3")
    weight_lookup = inputs.weights.set_index("occupation")["gdp_structure_weight"]
    draws = inputs.bootstrap_draws
    rows: list[dict[str, Any]] = []

    def sampled_panel(array: np.ndarray) -> np.ndarray:
        expanded = np.broadcast_to(array, (draws.shape[0],) + array.shape)
        return np.take_along_axis(expanded, draws, axis=2).sum(axis=2)

    for k in THRESHOLDS:
        selected = attempts[attempts["repeat"] < k].copy()
        require(len(selected) == 220 * k, f"pass@{k}: attempt count")
        grouped = selected.groupby("task_id").agg(
            success_count=("success", "sum"),
            energy_policy_mj=("energy_mj", "sum"),
        )
        grouped["mean_success_latency_hours"] = (
            selected[selected["success"]]
            .groupby("task_id")["latency_hours"]
            .mean()
        )
        grouped = grouped.reset_index()
        prompt = inputs.task_meta.merge(
            grouped, on="task_id", how="left", validate="one_to_one"
        )
        require(len(prompt) == 220, f"pass@{k}: prompt count")
        require(prompt[["success_count", "energy_policy_mj"]].notna().all().all(), f"pass@{k}: prompt join")

        prompt["credit_mask"] = prompt["success_count"] >= 1
        prompt["task_value"] = (
            prompt[baseline_hours_column].astype(float)
            * prompt[wage_column].astype(float)
        )
        prompt["generated_value"] = prompt["credit_mask"] * prompt["task_value"]
        prompt["hours_saved"] = np.where(
            prompt["credit_mask"],
            prompt[baseline_hours_column].astype(float)
            - prompt["mean_success_latency_hours"].fillna(0.0),
            0.0,
        )
        prompt["gdp_structure_weight"] = prompt["occupation"].map(weight_lookup)

        occupation = prompt.groupby("occupation").agg(
            generated_panel=("generated_value", "sum"),
            saved_panel=("hours_saved", "sum"),
            energy_panel=("energy_policy_mj", "sum"),
        )
        occupation["weight"] = occupation.index.map(weight_lookup)
        point_generated = float(
            (occupation["weight"] * occupation["generated_panel"]).sum()
        ) / 5
        point_saved = float(
            (occupation["weight"] * occupation["saved_panel"]).sum()
        ) / 5
        point_energy = float(
            (occupation["weight"] * occupation["energy_panel"]).sum()
        ) / 5
        point = {
            "task_success_rate": float(prompt["credit_mask"].mean()),
            "generated_value_usd": point_generated,
            "energy_mj": point_energy,
            "value_per_mj": point_generated / point_energy,
            "hours_saved": point_saved,
            "hours_per_mj": point_saved / point_energy,
        }

        generated_panel = sampled_panel(
            prompt_matrix(prompt, inputs.occupation_order, "generated_value")
        )
        saved_panel = sampled_panel(
            prompt_matrix(prompt, inputs.occupation_order, "hours_saved")
        )
        energy_panel = sampled_panel(
            prompt_matrix(prompt, inputs.occupation_order, "energy_policy_mj")
        )
        success_panel = sampled_panel(
            prompt_matrix(prompt, inputs.occupation_order, "credit_mask")
        )
        weighted_generated = (
            generated_panel * inputs.weight_vector[None, :]
        ).sum(axis=1) / 5
        weighted_saved = (
            saved_panel * inputs.weight_vector[None, :]
        ).sum(axis=1) / 5
        weighted_energy = (
            energy_panel * inputs.weight_vector[None, :]
        ).sum(axis=1) / 5
        arrays = {
            "task_success_rate": success_panel.sum(axis=1) / 220,
            "generated_value_usd": weighted_generated,
            "energy_mj": weighted_energy,
            "value_per_mj": weighted_generated / weighted_energy,
            "hours_saved": weighted_saved,
            "hours_per_mj": weighted_saved / weighted_energy,
        }
        row: dict[str, Any] = {
            "baseline": baseline.replace("_", "-"),
            "policy": f"pass@{k}",
            "k": k,
            "n_unique_prompts": 220,
            "attempts": 220 * k,
        }
        for field, values in arrays.items():
            row[f"{field}_estimate"] = point[field]
            for key, value in uncertainty(values).items():
                row[f"{field}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)

def latency_stats(inputs: AnalysisInputs) -> pd.DataFrame:
    rows = []
    for (model_id, baseline), attempts in inputs.attempts.items():
        for subset, selected in (
            ("all attempts", attempts),
            ("successful attempts", attempts[attempts["success"]]),
        ):
            values = selected["latency_hours"].astype(float)
            rows.append(
                {
                    "model": MODEL_CONFIGS[model_id]["display"],
                    "baseline": baseline.replace("_", "-"),
                    "subset": subset,
                    "n_attempts": len(values),
                    "mean_latency_hours": float(values.mean()),
                    "sd_latency_hours": float(values.std(ddof=1)),
                    "median_latency_hours": float(values.median()),
                    "min_latency_hours": float(values.min()),
                    "max_latency_hours": float(values.max()),
                }
            )
    return pd.DataFrame(rows)

def three_way_time_analysis(
    inputs: AnalysisInputs, prompt: pd.DataFrame
) -> pd.DataFrame:
    draws = inputs.bootstrap_draws
    success_count = prompt_matrix(prompt, inputs.occupation_order, "success_count")
    latency = np.nan_to_num(
        prompt_matrix(prompt, inputs.occupation_order, "mean_success_latency_hours"),
        nan=0.0,
    )
    human = prompt_matrix(prompt, inputs.occupation_order, "human_only_hours")
    human_ai = prompt_matrix(prompt, inputs.occupation_order, "human_ai_hours")
    energy = prompt_matrix(prompt, inputs.occupation_order, "energy_all_four_runs_mj")

    def sampled_panel(array: np.ndarray) -> np.ndarray:
        expanded = np.broadcast_to(array, (draws.shape[0],) + array.shape)
        return np.take_along_axis(expanded, draws, axis=2).sum(axis=2)

    rows = []
    for threshold in THRESHOLDS:
        credit_mask = success_count >= threshold
        transitions = {
            "Expert-only → expert-with-AI": {
                "saved": credit_mask * (human - human_ai),
                "denominator": credit_mask * human,
                "energy": False,
            },
            "Expert-with-AI → AI system": {
                "saved": credit_mask * (human_ai - latency),
                "denominator": credit_mask * human_ai,
                "energy": True,
            },
            "Expert-only → AI system": {
                "saved": credit_mask * (human - latency),
                "denominator": credit_mask * human,
                "energy": True,
            },
        }
        energy_point = np.sum(
            inputs.weight_vector[:, None] * energy
        ) / 5
        energy_boot = (
            sampled_panel(energy) * inputs.weight_vector[None, :]
        ).sum(axis=1) / 5
        for transition, spec in transitions.items():
            saved_point = np.sum(
                inputs.weight_vector[:, None] * spec["saved"]
            ) / 5
            denominator_point = np.sum(
                inputs.weight_vector[:, None] * spec["denominator"]
            ) / 5
            saved_boot = (
                sampled_panel(spec["saved"]) * inputs.weight_vector[None, :]
            ).sum(axis=1) / 5
            denominator_boot = (
                sampled_panel(spec["denominator"]) * inputs.weight_vector[None, :]
            ).sum(axis=1) / 5
            percent_point = 100 * saved_point / denominator_point
            percent_boot = 100 * saved_boot / denominator_boot
            row: dict[str, Any] = {
                "threshold": f"≥{threshold}/4" if threshold < 4 else "4/4",
                "transition": transition,
                "n_unique_prompts": 220,
                "attempts": 880,
                "hours_saved_estimate": saved_point,
                **{
                    f"hours_saved_{key}": value
                    for key, value in uncertainty(saved_boot).items()
                },
                "percent_saved_estimate": percent_point,
                **{
                    f"percent_saved_{key}": value
                    for key, value in uncertainty(percent_boot).items()
                },
            }
            if spec["energy"]:
                hours_per_mj_point = saved_point / energy_point
                hours_per_mj_boot = saved_boot / energy_boot
                row.update(
                    {
                        "energy_mj_exact": energy_point,
                        "hours_per_mj_estimate": hours_per_mj_point,
                        **{
                            f"hours_per_mj_{key}": value
                            for key, value in uncertainty(hours_per_mj_boot).items()
                        },
                    }
                )
            else:
                row.update(
                    {
                        "energy_mj_exact": math.nan,
                        "hours_per_mj_estimate": math.nan,
                        "hours_per_mj_bootstrap_mean": math.nan,
                        "hours_per_mj_bootstrap_sd": math.nan,
                        "hours_per_mj_ci95_low": math.nan,
                        "hours_per_mj_ci95_high": math.nan,
                    }
                )
            rows.append(row)
    result = pd.DataFrame(rows)
    for threshold in result["threshold"].unique():
        selected = result[result["threshold"] == threshold].set_index("transition")
        require(
            abs(
                selected.loc["Expert-only → AI system", "hours_saved_estimate"]
                - selected.loc["Expert-only → expert-with-AI", "hours_saved_estimate"]
                - selected.loc["Expert-with-AI → AI system", "hours_saved_estimate"]
            )
            < 1e-9,
            f"Three-way time identity at {threshold}",
        )
    return result

def model_swap_analysis(
    inputs: AnalysisInputs,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    category_points: dict[str, pd.DataFrame] = {}
    for model_id in (
        "glm52_h200",
        "qwen36_27b",
        "gemma4_31b",
        "gpt_oss_120b",
        "gpt_oss_20b",
    ):
        prompt = inputs.models[model_id]
        point_category, point_global, _ = point_metrics(
            prompt,
            inputs.weights,
            1,
            "human_only_hours",
            "median_hourly_wage_usd",
        )
        _, global_uncertainty = bootstrap_metrics(
            inputs,
            prompt,
            1,
            "human_only_hours",
            "median_hourly_wage_usd",
            point_category,
            point_global,
        )
        category_points[model_id] = point_category
        row: dict[str, Any] = {
            "model": MODEL_CONFIGS[model_id]["display"],
            "system_label": MODEL_CONFIGS[model_id]["system"],
            "hardware": MODEL_CONFIGS[model_id]["hardware"],
            "n_unique_prompts": 220,
            "attempts": 880,
        }
        for field in (
            "attempt_win_or_tie_rate",
            "generated_value_usd",
            "energy_mj",
            "value_per_mj",
            "hours_saved",
            "hours_per_mj",
        ):
            for key, value in global_uncertainty[field].items():
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
        model_frame[f"{field}_rank"] = ranks(
            model_frame[field].to_numpy(), descending
        ).astype(int)

    correlation_rows = []
    model_ids = list(category_points)
    for first_index, first in enumerate(model_ids):
        for second in model_ids[first_index + 1 :]:
            for field in (
                "generated_value_usd",
                "hours_saved",
                "value_per_mj",
                "hours_per_mj",
            ):
                left = ranks(category_points[first][field].to_numpy(), True)
                right = ranks(category_points[second][field].to_numpy(), True)
                correlation = float(np.corrcoef(left, right)[0, 1])
                correlation_rows.append(
                    {
                        "model_a": MODEL_CONFIGS[first]["display"],
                        "model_b": MODEL_CONFIGS[second]["display"],
                        "metric": field,
                        "spearman_rank_correlation": correlation,
                    }
                )
    return model_frame, pd.DataFrame(correlation_rows), category_points

def wage_robustness(
    inputs: AnalysisInputs, prompt: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    median_category, median_global, _ = point_metrics(
        prompt,
        inputs.weights,
        1,
        "human_only_hours",
        "median_hourly_wage_usd",
    )
    mean_category, mean_global, _ = point_metrics(
        prompt,
        inputs.weights,
        1,
        "human_only_hours",
        "mean_hourly_wage_usd",
    )
    rows = []
    for category in CATEGORIES:
        median = median_category[
            median_category["occupation_category"] == category
        ].iloc[0]
        mean = mean_category[mean_category["occupation_category"] == category].iloc[0]
        rows.append(
            {
                "occupation_category": category,
                "n_unique_prompts": int(median["n_unique_prompts"]),
                "mean_wage_generated_value_usd": mean["generated_value_usd"],
                "median_wage_generated_value_usd": median["generated_value_usd"],
                "generated_value_percent_change": 100
                * (
                    median["generated_value_usd"]
                    / mean["generated_value_usd"]
                    - 1
                ),
                "mean_wage_generated_value_rank": int(
                    mean["generated_value_usd_rank"]
                ),
                "median_wage_generated_value_rank": int(
                    median["generated_value_usd_rank"]
                ),
                "mean_wage_value_per_mj": mean["value_per_mj"],
                "median_wage_value_per_mj": median["value_per_mj"],
                "mean_wage_value_per_mj_rank": int(mean["value_per_mj_rank"]),
                "median_wage_value_per_mj_rank": int(median["value_per_mj_rank"]),
            }
        )
    global_rows = []
    for label, value in (("OEWS mean hourly wage", mean_global), ("OEWS median hourly wage", median_global)):
        global_rows.append(
            {
                "wage_basis": label,
                "n_unique_prompts": 220,
                "generated_value_usd": value["generated_value_usd"],
                "energy_mj": value["energy_mj"],
                "value_per_mj": value["value_per_mj"],
                "hours_saved": value["hours_saved"],
                "hours_per_mj": value["hours_per_mj"],
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(global_rows)

def category_weight_table(inputs: AnalysisInputs) -> pd.DataFrame:
    category = (
        inputs.weights.groupby("category")
        .agg(
            occupation_count=("occupation", "size"),
            gdp_structure_weight=("gdp_structure_weight", "sum"),
            national_employment=("national_employment", "sum"),
        )
        .reindex(CATEGORIES)
        .reset_index()
        .rename(columns={"category": "occupation_category"})
    )
    counts = inputs.task_meta.groupby("category").size().reindex(CATEGORIES)
    category["n_unique_prompts"] = counts.to_numpy(int)
    category["attempts"] = 4 * category["n_unique_prompts"]
    return category[
        [
            "occupation_category",
            "occupation_count",
            "n_unique_prompts",
            "attempts",
            "gdp_structure_weight",
            "national_employment",
        ]
    ]

def build_qa(
    inputs: AnalysisInputs,
    human_thresholds: pd.DataFrame,
    human_ai_thresholds: pd.DataFrame,
) -> dict[str, Any]:
    gemini = read_json(SOURCE_PATHS["gemini_summary"])
    parse = read_json(SOURCE_PATHS["gemini_parse"])
    frozen = read_json(SOURCE_PATHS["gemini_frozen_audit"])
    glm_prompt = inputs.models["glm52_h200"]
    raw_energy = float(glm_prompt["energy_all_four_runs_mj"].sum())
    allocated_energy = float(inputs.attempts[("glm52_h200", "human")]["energy_mj"].sum())
    query = pd.read_csv(SOURCE_PATHS["glm_query"])
    query = query[query["dataset"] == "GDPVal"]
    source_gpu_energy = float(query["allocated_gpu_energy_joules"].sum() / 1e6)
    checks = {
        "distinct_prompts_220": len(inputs.task_meta) == 220,
        "attempts_880": all(
            len(frame) == 880 for frame in inputs.attempts.values()
        ),
        "four_attempts_per_prompt": all(
            frame.groupby("task_id").size().eq(4).all()
            for frame in inputs.attempts.values()
        ),
        "occupations_44": inputs.task_meta["occupation"].nunique() == 44,
        "five_prompts_per_occupation": inputs.task_meta.groupby("occupation").size().eq(5).all(),
        "categories_13": inputs.task_meta["category"].nunique() == 13,
        "category_prompt_sum_220": int(
            inputs.task_meta.groupby("category").size().sum()
        )
        == 220,
        "gdp_weight_sum_one": abs(inputs.weights["gdp_structure_weight"].sum() - 1) < 1e-12,
        "median_wages_complete": inputs.task_meta["median_hourly_wage_usd"].notna().all(),
        "human_time_complete": inputs.task_meta["human_only_hours"].notna().all(),
        "human_ai_time_complete": inputs.task_meta["human_ai_hours"].notna().all(),
        "outcomes_complete": all(
            frame["agent_outcome"].notna().all() for frame in inputs.attempts.values()
        ),
        "energy_complete": all(
            frame["energy_all_four_runs_mj"].notna().all()
            for frame in inputs.models.values()
        ),
        "glm_energy_reconciles": abs(raw_energy - allocated_energy) < 1e-9,
        "glm_source_gpu_energy_positive": source_gpu_energy > 0,
        "gemini_model_pinned": gemini["model"] == "models/gemini-3.1-flash-lite",
        "gemini_valid_verdicts": parse["valid_verdicts"] == 3520 and parse["failures"] == 0,
        "evidence_audit_complete": frozen["complete"] is True,
        "human_threshold_energy_constant": human_thresholds["energy_mj_estimate"].nunique() == 1,
        "human_ai_threshold_energy_constant": human_ai_thresholds["energy_mj_estimate"].nunique() == 1,
    }
    require(all(checks.values()), f"QA failure: {[k for k, v in checks.items() if not v]}")
    return {
        "overall_validation": "PASS",
        "checks": checks,
        "observed": {
            "distinct_prompts": 220,
            "attempts_per_baseline_or_model": 880,
            "occupations": 44,
            "categories": 13,
            "bootstrap_samples": inputs.bootstrap_draws.shape[0],
            "bootstrap_seed": BOOTSTRAP_SEED,
            "glm_raw_all_four_run_energy_mj": raw_energy,
            "glm_allocated_energy_mj": allocated_energy,
            "glm_source_gpu_energy_mj": source_gpu_energy,
            "energy_reconciliation_error_mj": raw_energy - allocated_energy,
            "human_ai_model_counts": dict(inputs.human_ai_model_counts),
            "gemini_judge": gemini["model"],
            "gemini_valid_verdicts": parse["valid_verdicts"],
        },
    }


def save_frame(output_dir: Path, name: str, frame: pd.DataFrame) -> None:
    """Persist a numeric ledger without publication formatting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{name}.csv", index=False, float_format="%.15g")


def build_numeric_metrics(
    output_dir: Path, bootstrap_samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    """Recompute Joules to GDP source-level estimates from recorded inputs."""
    inputs = load_inputs(bootstrap_samples, seed)
    save_frame(output_dir, "category_weights", category_weight_table(inputs))

    human_thresholds, human_details = global_threshold_analyses(
        inputs, inputs.models["glm52_h200"], "expert-only", "human_only_hours",
        "median_hourly_wage_usd",
    )
    save_frame(output_dir, "glm_human_global_thresholds", human_thresholds)
    save_frame(
        output_dir, "glm_human_global_passk",
        passk_global_analysis(inputs, "human", "human_only_hours", "median_hourly_wage_usd"),
    )
    human_category, human_uncertainty, _ = human_details[1]
    save_frame(
        output_dir, "glm_human_category_metrics_k1",
        combine_point_uncertainty(human_category, human_uncertainty),
    )

    human_ai_thresholds, human_ai_details = global_threshold_analyses(
        inputs, inputs.models["glm52_h200_human_ai"], "expert-with-AI", "human_ai_hours",
        "median_hourly_wage_usd",
    )
    save_frame(output_dir, "glm_human_ai_global_thresholds", human_ai_thresholds)
    human_ai_category, human_ai_uncertainty, _ = human_ai_details[1]
    save_frame(
        output_dir, "glm_human_ai_category_metrics_k1",
        combine_point_uncertainty(human_ai_category, human_ai_uncertainty),
    )
    save_frame(
        output_dir, "glm_three_way_time_savings",
        three_way_time_analysis(inputs, inputs.models["glm52_h200_human_ai"]),
    )
    model_comparison, model_correlations, _ = model_swap_analysis(inputs)
    save_frame(output_dir, "five_model_global_system_comparison", model_comparison)
    save_frame(output_dir, "five_model_category_rank_correlations", model_correlations)
    wage_category, wage_global = wage_robustness(inputs, inputs.models["glm52_h200"])
    save_frame(output_dir, "wage_basis_category_robustness", wage_category)
    save_frame(output_dir, "wage_basis_global_robustness", wage_global)
    save_frame(output_dir, "attempt_latency_descriptive_statistics", latency_stats(inputs))
    save_frame(output_dir, "source_inventory", pd.DataFrame(inputs.source_inventory))

    qa = build_qa(inputs, human_thresholds, human_ai_thresholds)
    energy_mode = os.environ.get("J2G_ENERGY_BOUNDARY", "total")
    qa["energy_boundary"] = (
        "measured GPU+CPU+DRAM" if energy_mode == "total" else "GPU"
    )
    if energy_mode == "total":
        qa["energy_component_ledger_sha256"] = sha256_file(
            ROOT / "outputs/energy_boundary/attempt_energy.csv"
        )
    write_json(output_dir / "validation.json", qa)
    return qa


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/metrics")
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    build_numeric_metrics(args.output_dir, args.bootstrap_samples, args.seed)
    print((args.output_dir / "validation.json").read_text(), end="")
