#!/usr/bin/env python3
"""Compute GDPVal value/energy metrics for the GPT-OSS studies.

The calculation follows the frozen Qwen/GLM methodology:

* Gemini win or tie against the expert-only reference is a successful attempt.
* A prompt qualifies at threshold k when at least k of its four attempts succeed.
* A qualifying prompt contributes its output dollar value exactly once.
* Time saving is recorded expert completion time less mean AI-system completion
  time across that
  prompt's successful attempts. Negative savings are retained.
* Every category's energy denominator contains all four attempts for every
  prompt in the category, including unsuccessful and incomplete attempts.
* Source batch energy (already summed across all GPUs) is allocated equally
  across the batch's declared attempts and reconciled back to the batch total.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Iterable


getcontext().prec = 50

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = (
    ROOT
    / "outputs/019f86a0-e5e6-7e90-8578-b08ddb4385fb"
    / "gdpval_snorkel_task_wage_values.xlsx"
)
HF_REVISION = "4a277392cddacbef090be029d33f4f69372d487e"
DEFAULT_BATCH_INPUT_ROOT = ROOT / "reproduction/inputs/gptoss/batches"
DEFAULT_TELEMETRY_INPUT_ROOT = ROOT / "reproduction/inputs/gptoss/telemetry"

XML_NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
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

MODELS = {
    "GPT-OSS 120B": {
        "slug": "gpt_oss_120b",
        "crosswalk": ROOT
        / "reports/gpt_oss_120b_norubric_source_20260724/crosswalk.json",
        "source_outputs": ROOT
        / "reports/gpt_oss_120b_norubric_source_20260724/gptoss_source_outputs.json",
        "results": ROOT
        / "reports/gpt_oss_120b_norubric_gemini31_flash_lite_20260724/results.output_level.jsonl",
        "batches": Path(
            "gpt-oss-120b-high-no-rubric/run/batches.jsonl"
        ),
        "telemetry_audit": Path(
            "gpt-oss-120b-high-no-rubric/reports/final/telemetry_audit.json"
        ),
    },
    "GPT-OSS 20B": {
        "slug": "gpt_oss_20b",
        "crosswalk": ROOT
        / "reports/gpt_oss_20b_norubric_source_20260724/crosswalk.json",
        "source_outputs": ROOT
        / "reports/gpt_oss_20b_norubric_source_20260724/gptoss_source_outputs.json",
        "results": ROOT
        / "reports/gpt_oss_20b_norubric_gemini31_flash_lite_20260724/results.output_level.jsonl",
        "batches": Path(
            "gpt-oss-20b-high-no-rubric/run/batches.jsonl"
        ),
        "telemetry_audit": Path(
            "gpt-oss-20b-high-no-rubric/reports/final_corrected/"
            "telemetry_audit.json"
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_lines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _sheet_rows(path: Path) -> Iterable[dict[str, str | None]]:
    """Read the first XLSX worksheet without a third-party Excel dependency."""
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in item.findall(".//m:t", XML_NS))
                for item in strings.findall("m:si", XML_NS)
            ]
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        for row in root.findall(".//m:sheetData/m:row", XML_NS):
            values: dict[str, str | None] = {}
            for cell in row.findall("m:c", XML_NS):
                reference = cell.attrib["r"]
                column = "".join(ch for ch in reference if ch.isalpha())
                kind = cell.attrib.get("t")
                raw = cell.find("m:v", XML_NS)
                inline = cell.find("m:is", XML_NS)
                if kind == "s" and raw is not None:
                    value = shared[int(raw.text)]
                elif kind == "inlineStr" and inline is not None:
                    value = "".join(
                        node.text or "" for node in inline.findall(".//m:t", XML_NS)
                    )
                else:
                    value = raw.text if raw is not None else None
                values[column] = value
            yield values


def valuation_rows() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in _sheet_rows(WORKBOOK):
        if row.get("A") != "GDPVal":
            continue
        task_id = str(row["B"])
        onet_code = str(row["D"])
        major_code = onet_code[:2]
        if major_code not in CATEGORIES:
            raise AssertionError(f"unexpected O*NET major code {onet_code} for {task_id}")
        hourly_wage = float(row["K"])
        human_hours = float(row["L"])
        human_value = float(row["O"])
        if not math.isclose(
            human_value, hourly_wage * human_hours, rel_tol=1e-10, abs_tol=1e-8
        ):
            raise AssertionError(f"value mismatch for {task_id}")
        output[task_id] = {
            "baseline_task_id": task_id,
            "occupation": str(row["C"]),
            "onet_code": onet_code,
            "category": CATEGORIES[major_code],
            "hourly_wage": hourly_wage,
            "human_only_hours": human_hours,
            "human_only_value_usd": human_value,
        }
    if len(output) != 220:
        raise AssertionError(f"expected 220 GDPVal tasks, got {len(output)}")
    return output


def _rank_rows(rows: list[dict[str, Any]]) -> None:
    value_order = sorted(
        rows, key=lambda row: (-row["value_per_mj"], row["category"])
    )
    hours_order = sorted(
        rows, key=lambda row: (-row["hours_saved_per_mj"], row["category"])
    )
    for rank, row in enumerate(value_order, 1):
        row["value_rank"] = rank
    for rank, row in enumerate(hours_order, 1):
        row["hours_saved_rank"] = rank
    rows.sort(key=lambda row: row["value_rank"])


def calculate_model(
    model: str,
    config: dict[str, Any],
    valuations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    crosswalk = json.loads(Path(config["crosswalk"]).read_text())
    if len(crosswalk) != 220:
        raise AssertionError(f"{model}: crosswalk has {len(crosswalk)} rows")

    task_info: dict[str, dict[str, Any]] = {}
    for row in crosswalk:
        agent_task_id = str(row["agent_task_id"])
        baseline_task_id = str(row["baseline_task_id"])
        info = dict(valuations[baseline_task_id])
        info["agent_task_id"] = agent_task_id
        info["agent_occupation"] = str(row["occupation"])
        task_info[agent_task_id] = info
    if len(task_info) != 220:
        raise AssertionError(f"{model}: expected 220 unique agent tasks")

    all_results = json_lines(Path(config["results"]))
    human_results = [row for row in all_results if row["baseline"] == "human"]
    if len(human_results) != 880:
        raise AssertionError(f"{model}: expected 880 expert-only reference comparisons")
    results_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in human_results:
        key = (str(row["task_id"]), int(row["repeat"]))
        if key in results_by_key:
            raise AssertionError(f"{model}: duplicate result {key}")
        if row["agent_outcome"] not in {"win", "tie", "loss"}:
            raise AssertionError(f"{model}: unexpected outcome {row['agent_outcome']}")
        results_by_key[key] = row

    source_outputs = json.loads(Path(config["source_outputs"]).read_text())
    if len(source_outputs) != 880:
        raise AssertionError(f"{model}: expected 880 source outputs")
    source_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in source_outputs:
        key = (str(row["task_id"]), int(row["repeat"]))
        if key in source_by_key:
            raise AssertionError(f"{model}: duplicate source output {key}")
        source_by_key[key] = row

    batches = json_lines(Path(config["batches"]))
    if len(batches) != 44:
        raise AssertionError(f"{model}: expected 44 source batches")
    batch_by_query: dict[str, dict[str, Any]] = {}
    batch_qa: list[dict[str, Any]] = []
    source_energy_j = Decimal("0")
    for batch in batches:
        query_ids = [str(query_id) for query_id in batch["query_ids"]]
        batch_size = int(batch["batch_size"])
        energy_j = Decimal(str(batch["batch_gpu_energy_joules"]))
        devices = [int(device) for device in batch["telemetry_device_ids"]]
        if batch_size != len(query_ids) or batch_size != 20:
            raise AssertionError(f"{model}: bad batch size in {batch['batch_id']}")
        if devices != list(range(8)):
            raise AssertionError(f"{model}: unexpected devices in {batch['batch_id']}")
        for query_id in query_ids:
            if query_id in batch_by_query:
                raise AssertionError(f"{model}: duplicate batch query {query_id}")
            batch_by_query[query_id] = batch
        source_energy_j += energy_j
        allocation = energy_j / Decimal(batch_size)
        allocated_back = sum((allocation for _ in query_ids), Decimal("0"))
        batch_qa.append(
            {
                "batch_id": str(batch["batch_id"]),
                "batch_size": batch_size,
                "gpu_count": len(devices),
                "source_energy_j": float(energy_j),
                "allocated_energy_j": float(allocated_back),
                "reconciliation_error_j": float(allocated_back - energy_j),
            }
        )
    if len(batch_by_query) != 880:
        raise AssertionError(f"{model}: batches cover {len(batch_by_query)} queries")

    attempts: list[dict[str, Any]] = []
    missing: dict[str, list[str]] = defaultdict(list)
    for key, result in results_by_key.items():
        task_id, repeat = key
        if task_id not in task_info:
            missing["crosswalk"].append(f"{task_id}/rep{repeat}")
            continue
        source = source_by_key.get(key)
        if source is None:
            missing["source_output"].append(f"{task_id}/rep{repeat}")
            continue
        query_id = str(source["query_id"])
        batch = batch_by_query.get(query_id)
        if batch is None:
            missing["batch"].append(query_id)
            continue
        metadata_path = Path(source["metadata_path"])
        if not metadata_path.is_file():
            missing["metadata"].append(str(metadata_path))
            continue
        metadata = json.loads(metadata_path.read_text())
        latency_s = metadata.get("wall_clock_s")
        if latency_s is None:
            missing["latency"].append(query_id)
            continue
        latency_s = float(latency_s)
        if not math.isfinite(latency_s):
            missing["latency"].append(query_id)
            continue
        energy_j = Decimal(str(batch["batch_gpu_energy_joules"])) / Decimal(
            int(batch["batch_size"])
        )
        info = task_info[task_id]
        outcome = str(result["agent_outcome"])
        attempt = {
            "model": model,
            "comparison_id": str(result["comparison_id"]),
            "agent_task_id": task_id,
            "baseline_task_id": info["baseline_task_id"],
            "category": info["category"],
            "onet_code": info["onet_code"],
            "occupation": info["occupation"],
            "repeat": repeat,
            "query_id": query_id,
            "outcome": outcome,
            "success": outcome in {"win", "tie"},
            "latency_seconds": latency_s,
            "latency_hours": latency_s / 3600.0,
            "batch_id": str(batch["batch_id"]),
            "batch_size": int(batch["batch_size"]),
            "source_batch_energy_j": float(batch["batch_gpu_energy_joules"]),
            "allocated_attempt_energy_j": float(energy_j),
            "allocated_attempt_energy_mj": float(energy_j / Decimal("1000000")),
            "gpu_count": len(batch["telemetry_device_ids"]),
            "completed": bool(source["completed"]),
            "timed_out": bool(source["timed_out"]),
            "metadata_path": str(metadata_path),
        }
        attempts.append(attempt)
    if missing:
        raise AssertionError(f"{model}: missing joined values {dict(missing)}")
    if len(attempts) != 880:
        raise AssertionError(f"{model}: expected 880 joined attempts")

    attempt_counts = Counter(attempt["agent_task_id"] for attempt in attempts)
    if set(attempt_counts.values()) != {4} or len(attempt_counts) != 220:
        raise AssertionError(f"{model}: attempts are not exactly four per prompt")
    repeat_sets: dict[str, set[int]] = defaultdict(set)
    for attempt in attempts:
        repeat_sets[attempt["agent_task_id"]].add(attempt["repeat"])
    if any(repeats != {0, 1, 2, 3} for repeats in repeat_sets.values()):
        raise AssertionError(f"{model}: repeat coverage is not 0-3 for every prompt")

    attempts_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        attempts_by_task[attempt["agent_task_id"]].append(attempt)
    prompts: list[dict[str, Any]] = []
    for task_id, task_attempts in attempts_by_task.items():
        successes = [attempt for attempt in task_attempts if attempt["success"]]
        info = task_info[task_id]
        prompts.append(
            {
                "model": model,
                "agent_task_id": task_id,
                "baseline_task_id": info["baseline_task_id"],
                "category": info["category"],
                "onet_code": info["onet_code"],
                "occupation": info["occupation"],
                "hourly_wage": info["hourly_wage"],
                "human_only_hours": info["human_only_hours"],
                "human_only_value_usd": info["human_only_value_usd"],
                "success_count": len(successes),
                "mean_success_latency_hours": (
                    sum(attempt["latency_hours"] for attempt in successes)
                    / len(successes)
                    if successes
                    else None
                ),
            }
        )
    if len(prompts) != 220:
        raise AssertionError(f"{model}: expected 220 prompt rows")

    categories = list(CATEGORIES.values())
    thresholds: dict[str, Any] = {}
    for threshold in range(1, 5):
        category_rows: list[dict[str, Any]] = []
        for category in categories:
            category_prompts = [
                prompt for prompt in prompts if prompt["category"] == category
            ]
            category_attempts = [
                attempt for attempt in attempts if attempt["category"] == category
            ]
            qualifying = [
                prompt
                for prompt in category_prompts
                if prompt["success_count"] >= threshold
            ]
            successes = sum(attempt["success"] for attempt in category_attempts)
            energy_mj = sum(
                attempt["allocated_attempt_energy_mj"]
                for attempt in category_attempts
            )
            generated_value = sum(
                prompt["human_only_value_usd"] for prompt in qualifying
            )
            hours_saved = sum(
                prompt["human_only_hours"]
                - float(prompt["mean_success_latency_hours"])
                for prompt in qualifying
            )
            category_rows.append(
                {
                    "category": category,
                    "threshold": threshold,
                    "qualifying_prompts": len(qualifying),
                    "total_prompts": len(category_prompts),
                    "successful_attempts": successes,
                    "total_attempts": len(category_attempts),
                    "attempt_win_or_tie_rate": successes
                    / len(category_attempts),
                    "energy_mj_all_four_runs": energy_mj,
                    "generated_human_only_value_usd": generated_value,
                    "value_per_mj": generated_value / energy_mj,
                    "total_human_only_hours_saved": hours_saved,
                    "hours_saved_per_mj": hours_saved / energy_mj,
                }
            )
        _rank_rows(category_rows)
        overall_qualifying = [
            prompt for prompt in prompts if prompt["success_count"] >= threshold
        ]
        total_energy_mj = sum(
            attempt["allocated_attempt_energy_mj"] for attempt in attempts
        )
        total_generated_value = sum(
            prompt["human_only_value_usd"] for prompt in overall_qualifying
        )
        total_hours_saved = sum(
            prompt["human_only_hours"]
            - float(prompt["mean_success_latency_hours"])
            for prompt in overall_qualifying
        )
        thresholds[str(threshold)] = {
            "category_rows": category_rows,
            "overall": {
                "threshold": threshold,
                "qualifying_prompts": len(overall_qualifying),
                "total_prompts": 220,
                "successful_attempts": sum(
                    attempt["success"] for attempt in attempts
                ),
                "total_attempts": 880,
                "attempt_win_or_tie_rate": sum(
                    attempt["success"] for attempt in attempts
                )
                / 880,
                "energy_mj_all_four_runs": total_energy_mj,
                "generated_human_only_value_usd": total_generated_value,
                "value_per_mj": total_generated_value / total_energy_mj,
                "total_human_only_hours_saved": total_hours_saved,
                "hours_saved_per_mj": total_hours_saved / total_energy_mj,
            },
        }

    allocated_energy_j = sum(
        (Decimal(str(attempt["allocated_attempt_energy_j"])) for attempt in attempts),
        Decimal("0"),
    )
    # Float serialization of per-attempt allocations can introduce a tiny error.
    reconciliation_error_j = allocated_energy_j - source_energy_j
    outcomes = Counter(attempt["outcome"] for attempt in attempts)
    telemetry_audit = json.loads(Path(config["telemetry_audit"]).read_text())
    qa = {
        "valuation_prompt_count": len(valuations),
        "crosswalk_prompt_count": len(task_info),
        "prompt_count": len(prompts),
        "attempt_count": len(attempts),
        "attempts_per_prompt_values": sorted(set(attempt_counts.values())),
        "repeat_sets_all_0_to_3": all(
            repeats == {0, 1, 2, 3} for repeats in repeat_sets.values()
        ),
        "category_count": len({prompt["category"] for prompt in prompts}),
        "category_names_complete": set(prompt["category"] for prompt in prompts)
        == set(CATEGORIES.values()),
        "batch_count": len(batches),
        "batch_size_values": sorted({int(batch["batch_size"]) for batch in batches}),
        "gpu_count_values": sorted(
            {len(batch["telemetry_device_ids"]) for batch in batches}
        ),
        "source_energy_j": float(source_energy_j),
        "allocated_attempt_energy_j": float(allocated_energy_j),
        "category_energy_j": float(
            sum(
                Decimal(str(attempt["allocated_attempt_energy_j"]))
                for attempt in attempts
            )
        ),
        "energy_reconciliation_error_j": float(reconciliation_error_j),
        "max_batch_reconciliation_error_j": max(
            abs(batch["reconciliation_error_j"]) for batch in batch_qa
        ),
        "missing_wage_values": sum(
            prompt["hourly_wage"] is None for prompt in prompts
        ),
        "missing_human_time_values": sum(
            prompt["human_only_hours"] is None for prompt in prompts
        ),
        "missing_human_value_values": sum(
            prompt["human_only_value_usd"] is None for prompt in prompts
        ),
        "missing_outcome_values": sum(not attempt["outcome"] for attempt in attempts),
        "missing_latency_values": sum(
            attempt["latency_seconds"] is None for attempt in attempts
        ),
        "missing_energy_values": sum(
            attempt["allocated_attempt_energy_j"] is None for attempt in attempts
        ),
        "outcomes": dict(sorted(outcomes.items())),
        "successful_attempt_count": sum(
            attempt["success"] for attempt in attempts
        ),
        "completed_source_attempt_count": sum(
            attempt["completed"] for attempt in attempts
        ),
        "timed_out_source_attempt_count": sum(
            attempt["timed_out"] for attempt in attempts
        ),
        "telemetry_audit_accounting": telemetry_audit.get(
            "telemetry_accounting"
        ),
        "passed": all(
            [
                len(valuations) == 220,
                len(task_info) == 220,
                len(prompts) == 220,
                len(attempts) == 880,
                set(attempt_counts.values()) == {4},
                all(repeats == {0, 1, 2, 3} for repeats in repeat_sets.values()),
                len({prompt["category"] for prompt in prompts}) == 13,
                len(batches) == 44,
                len(batch_by_query) == 880,
                all(batch["batch_size"] == 20 for batch in batches),
                all(
                    len(batch["telemetry_device_ids"]) == 8 for batch in batches
                ),
                abs(float(reconciliation_error_j)) < 1e-6,
            ]
        ),
    }
    if not qa["passed"]:
        raise AssertionError(f"{model}: QA failed: {qa}")

    return {
        "model": model,
        "hf_revision": HF_REVISION,
        "thresholds": thresholds,
        "qa": qa,
        "batch_qa": batch_qa,
        "prompt_ledger": sorted(
            prompts, key=lambda row: (row["category"], row["baseline_task_id"])
        ),
        "attempt_ledger": sorted(
            attempts,
            key=lambda row: (
                row["category"],
                row["baseline_task_id"],
                row["repeat"],
            ),
        ),
        "source_hashes": {
            "workbook_sha256": sha256(WORKBOOK),
            "crosswalk_sha256": sha256(Path(config["crosswalk"])),
            "source_outputs_sha256": sha256(Path(config["source_outputs"])),
            "results_sha256": sha256(Path(config["results"])),
            "batches_sha256": sha256(Path(config["batches"])),
            "telemetry_audit_sha256": sha256(Path(config["telemetry_audit"])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. If omitted, the JSON is printed.",
    )
    parser.add_argument(
        "--batch-input-root",
        type=Path,
        default=DEFAULT_BATCH_INPUT_ROOT,
        help="Root containing each GPT-OSS source run's run/batches.jsonl.",
    )
    parser.add_argument(
        "--telemetry-input-root",
        type=Path,
        default=DEFAULT_TELEMETRY_INPUT_ROOT,
        help="Root containing each GPT-OSS source run's telemetry audit.",
    )
    args = parser.parse_args()

    valuations = valuation_rows()
    report = {
        "methodology": {
            "study": "GDPVal GPT-OSS value/energy analysis",
            "primary_baseline": "human",
            "success_definition": "Gemini agent_outcome in {win, tie}",
            "thresholds": [1, 2, 3, 4],
            "generated_value_rule": (
                "Count each qualifying prompt's output dollar value once."
            ),
            "time_saved_rule": (
                "Recorded expert completion time minus mean AI-system completion time across "
                "successful attempts; negative values are not clamped."
            ),
            "energy_rule": (
                "All four attempts for every category prompt, including failed "
                "runs; source batch energy already sums all eight GPUs and is "
                "allocated equally across the declared batch size."
            ),
            "hf_revision": HF_REVISION,
        },
        "models": {
            model: calculate_model(
                model,
                {
                    **config,
                    "batches": args.batch_input_root / config["batches"],
                    "telemetry_audit": args.telemetry_input_root / config["telemetry_audit"],
                },
                valuations,
            )
            for model, config in MODELS.items()
        },
    }
    text = json.dumps(report, indent=2, sort_keys=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
