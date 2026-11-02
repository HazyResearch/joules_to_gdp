#!/usr/bin/env python3
"""Compute GDPVal value/energy metrics for the full Gemma study.

This reproduces the frozen Qwen/GLM/GPT-OSS accounting contract:

* Gemini win or tie against the expert-only reference is a successful attempt.
* A prompt qualifies at threshold k when at least k of four attempts succeeds.
* A qualifying prompt contributes its output dollar value exactly once.
* Time saving is recorded expert completion time less mean AI-system completion
  time over successful attempts.
* Energy includes all four attempts for every prompt, regardless of outcome.
* Eight-GPU source batch energy is allocated equally over each 20-row batch.
"""

from __future__ import annotations

import argparse
import csv
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
SOURCE_STATE = ROOT / "reports/gemma4_31b_norubric_source_20260725"
JUDGE_STATE = (
    ROOT / "reports/gemma4_31b_norubric_gemini31_flash_lite_20260725"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/gemma4_31b_norubric_major_category_value_energy_20260725"
)
VALUATION_WORKBOOK = (
    ROOT
    / "outputs/019f86a0-e5e6-7e90-8578-b08ddb4385fb"
    / "gdpval_snorkel_task_wage_values.xlsx"
)
HF_REPO = "ScalingIntelligence/gemma4-31b-gdpval-occ-repeat4-202607"
HF_REVISION = "716ea26d5ecc5991e42e82d10424e6ddc286ccfc"

SHARD_SPECS = {
    "rows-000-459": {
        "source_index_min": 0,
        "source_index_max": 114,
        "attempt_count": 460,
    },
    "rows-460-879": {
        "source_index_min": 115,
        "source_index_max": 219,
        "attempt_count": 420,
    },
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

XML_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


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
    """Read the first XLSX worksheet without requiring an Excel library."""
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
    for row in _sheet_rows(VALUATION_WORKBOOK):
        if row.get("A") != "GDPVal":
            continue
        task_id = str(row["B"])
        onet_code = str(row["D"])
        major_code = onet_code[:2]
        if major_code not in CATEGORIES:
            raise AssertionError(f"unexpected O*NET major code {onet_code}")
        hourly_wage = float(row["K"])
        human_hours = float(row["L"])
        human_value = float(row["O"])
        if not math.isclose(
            human_value,
            hourly_wage * human_hours,
            rel_tol=1e-10,
            abs_tol=1e-8,
        ):
            raise AssertionError(f"valuation mismatch for {task_id}")
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
        raise AssertionError(f"expected 220 valuation rows, got {len(output)}")
    return output


def source_shard(row: dict[str, Any]) -> str:
    hf_path = str(row["metadata_hf_path"])
    matches = [name for name in SHARD_SPECS if f"/{name}/" in hf_path]
    if len(matches) != 1:
        raise AssertionError(f"cannot resolve source shard from {hf_path}")
    shard = matches[0]
    source_index = int(row["source_task_index"])
    spec = SHARD_SPECS[shard]
    if not spec["source_index_min"] <= source_index <= spec["source_index_max"]:
        raise AssertionError(
            f"source index {source_index} does not belong to {shard}"
        )
    return shard


def rank_rows(rows: list[dict[str, Any]]) -> None:
    for rank, row in enumerate(
        sorted(rows, key=lambda item: (-item["value_per_mj"], item["category"])),
        1,
    ):
        row["value_rank"] = rank
    for rank, row in enumerate(
        sorted(
            rows,
            key=lambda item: (-item["hours_saved_per_mj"], item["category"]),
        ),
        1,
    ):
        row["hours_saved_rank"] = rank
    rows.sort(key=lambda item: item["value_rank"])


def build_report(output_dir: Path) -> dict[str, Any]:
    valuations = valuation_rows()
    crosswalk = json.loads((SOURCE_STATE / "crosswalk.json").read_text())
    if len(crosswalk) != 220:
        raise AssertionError(f"expected 220 crosswalk rows, got {len(crosswalk)}")

    task_info: dict[str, dict[str, Any]] = {}
    for row in crosswalk:
        agent_task_id = str(row["agent_task_id"])
        baseline_task_id = str(row["baseline_task_id"])
        if baseline_task_id not in valuations:
            raise AssertionError(f"missing valuation for {baseline_task_id}")
        if agent_task_id in task_info:
            raise AssertionError(f"duplicate agent task {agent_task_id}")
        info = dict(valuations[baseline_task_id])
        info["agent_task_id"] = agent_task_id
        info["agent_occupation"] = str(row["occupation"])
        task_info[agent_task_id] = info

    result_path = JUDGE_STATE / "results.output_level.jsonl"
    human_results = [
        row for row in json_lines(result_path) if row["baseline"] == "human"
    ]
    if len(human_results) != 880:
        raise AssertionError(f"expected 880 expert-only reference verdicts, got {len(human_results)}")
    results_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in human_results:
        key = (str(row["task_id"]), int(row["repeat"]))
        if key in results_by_key:
            raise AssertionError(f"duplicate verdict {key}")
        if row["agent_outcome"] not in {"win", "tie", "loss"}:
            raise AssertionError(f"unexpected verdict {row['agent_outcome']}")
        results_by_key[key] = row

    source_path = SOURCE_STATE / "gemma_source_outputs.json"
    source_outputs = json.loads(source_path.read_text())
    if len(source_outputs) != 880:
        raise AssertionError(f"expected 880 source outputs, got {len(source_outputs)}")
    source_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    shard_counts: Counter[str] = Counter()
    for row in source_outputs:
        if row["include_rubric_in_prompt"] is not False:
            raise AssertionError("Gemma source prompt configuration does not match the default")
        key = (str(row["task_id"]), int(row["repeat"]))
        if key in source_by_key:
            raise AssertionError(f"duplicate source output {key}")
        shard = source_shard(row)
        row = dict(row)
        row["source_shard"] = shard
        source_by_key[key] = row
        shard_counts[shard] += 1
    for shard, spec in SHARD_SPECS.items():
        if shard_counts[shard] != spec["attempt_count"]:
            raise AssertionError(
                f"{shard} has {shard_counts[shard]} attempts, "
                f"expected {spec['attempt_count']}"
            )

    batch_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    batches: list[dict[str, Any]] = []
    batch_qa: list[dict[str, Any]] = []
    source_energy_j = Decimal("0")
    batch_hashes: dict[str, str] = {}
    for shard in SHARD_SPECS:
        batch_path = output_dir / "source_telemetry" / shard / "batches.jsonl"
        shard_batches = json_lines(batch_path)
        expected_batches = SHARD_SPECS[shard]["attempt_count"] // 20
        if len(shard_batches) != expected_batches:
            raise AssertionError(
                f"{shard} has {len(shard_batches)} batches, "
                f"expected {expected_batches}"
            )
        batch_hashes[shard] = sha256(batch_path)
        for source_batch in shard_batches:
            batch = dict(source_batch)
            batch["source_shard"] = shard
            query_ids = [str(query_id) for query_id in batch["query_ids"]]
            batch_size = int(batch["batch_size"])
            energy_j = Decimal(str(batch["batch_gpu_energy_joules"]))
            devices = [int(device) for device in batch["telemetry_device_ids"]]
            if batch_size != len(query_ids) or batch_size != 20:
                raise AssertionError(f"bad batch size in {shard}/{batch['batch_id']}")
            if devices != list(range(8)):
                raise AssertionError(
                    f"unexpected GPU set in {shard}/{batch['batch_id']}"
                )
            for query_id in query_ids:
                key = (shard, query_id)
                if key in batch_by_key:
                    raise AssertionError(f"duplicate batch query {key}")
                batch_by_key[key] = batch
            source_energy_j += energy_j
            allocation_j = energy_j / Decimal(batch_size)
            allocated_back = allocation_j * Decimal(batch_size)
            batch_qa.append(
                {
                    "source_shard": shard,
                    "batch_id": str(batch["batch_id"]),
                    "batch_size": batch_size,
                    "gpu_count": len(devices),
                    "source_energy_j": float(energy_j),
                    "allocated_energy_j": float(allocated_back),
                    "reconciliation_error_j": float(allocated_back - energy_j),
                }
            )
            batches.append(batch)
    if len(batches) != 44 or len(batch_by_key) != 880:
        raise AssertionError(
            f"telemetry covers {len(batches)} batches/{len(batch_by_key)} attempts"
        )

    attempts: list[dict[str, Any]] = []
    for key, result in results_by_key.items():
        task_id, repeat = key
        source = source_by_key.get(key)
        if source is None:
            raise AssertionError(f"missing source output {key}")
        shard = str(source["source_shard"])
        query_id = str(source["query_id"])
        batch = batch_by_key.get((shard, query_id))
        if batch is None:
            raise AssertionError(f"missing telemetry for {(shard, query_id)}")
        metadata_path = Path(source["metadata_path"])
        if not metadata_path.is_file():
            raise AssertionError(f"missing metadata {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        latency_s = metadata.get("wall_clock_s")
        if latency_s is None or not math.isfinite(float(latency_s)):
            raise AssertionError(f"invalid latency for {(shard, query_id)}")
        energy_j = Decimal(str(batch["batch_gpu_energy_joules"])) / Decimal(
            int(batch["batch_size"])
        )
        info = task_info[task_id]
        outcome = str(result["agent_outcome"])
        attempts.append(
            {
                "comparison_id": str(result["comparison_id"]),
                "agent_task_id": task_id,
                "baseline_task_id": info["baseline_task_id"],
                "category": info["category"],
                "onet_code": info["onet_code"],
                "occupation": info["occupation"],
                "repeat": repeat,
                "source_shard": shard,
                "query_id": query_id,
                "outcome": outcome,
                "success": outcome in {"win", "tie"},
                "latency_seconds": float(latency_s),
                "latency_hours": float(latency_s) / 3600.0,
                "batch_id": str(batch["batch_id"]),
                "batch_size": int(batch["batch_size"]),
                "source_batch_energy_j": float(
                    batch["batch_gpu_energy_joules"]
                ),
                "allocated_attempt_energy_j": float(energy_j),
                "allocated_attempt_energy_mj": float(
                    energy_j / Decimal("1000000")
                ),
                "gpu_count": len(batch["telemetry_device_ids"]),
                "completed": bool(source["completed"]),
                "timed_out": bool(source["timed_out"]),
            }
        )
    if len(attempts) != 880:
        raise AssertionError(f"expected 880 joined attempts, got {len(attempts)}")

    attempt_counts = Counter(row["agent_task_id"] for row in attempts)
    repeat_sets: dict[str, set[int]] = defaultdict(set)
    attempts_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        repeat_sets[attempt["agent_task_id"]].add(attempt["repeat"])
        attempts_by_task[attempt["agent_task_id"]].append(attempt)
    if (
        len(attempt_counts) != 220
        or set(attempt_counts.values()) != {4}
        or any(repeats != {0, 1, 2, 3} for repeats in repeat_sets.values())
    ):
        raise AssertionError("attempts are not exactly repeats 0-3 per prompt")

    prompts: list[dict[str, Any]] = []
    for task_id, task_attempts in attempts_by_task.items():
        successes = [row for row in task_attempts if row["success"]]
        info = task_info[task_id]
        prompts.append(
            {
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
                    sum(row["latency_hours"] for row in successes) / len(successes)
                    if successes
                    else None
                ),
            }
        )

    thresholds: dict[str, Any] = {}
    total_energy_mj = sum(row["allocated_attempt_energy_mj"] for row in attempts)
    for threshold in range(1, 5):
        category_rows: list[dict[str, Any]] = []
        for category in CATEGORIES.values():
            category_prompts = [
                row for row in prompts if row["category"] == category
            ]
            category_attempts = [
                row for row in attempts if row["category"] == category
            ]
            qualifying = [
                row for row in category_prompts if row["success_count"] >= threshold
            ]
            successful_attempts = sum(row["success"] for row in category_attempts)
            energy_mj = sum(
                row["allocated_attempt_energy_mj"] for row in category_attempts
            )
            value = sum(row["human_only_value_usd"] for row in qualifying)
            hours_saved = sum(
                row["human_only_hours"] - float(row["mean_success_latency_hours"])
                for row in qualifying
            )
            category_rows.append(
                {
                    "category": category,
                    "threshold": threshold,
                    "qualifying_prompts": len(qualifying),
                    "total_prompts": len(category_prompts),
                    "successful_attempts": successful_attempts,
                    "total_attempts": len(category_attempts),
                    "attempt_win_or_tie_rate": (
                        successful_attempts / len(category_attempts)
                    ),
                    "energy_mj_all_four_runs": energy_mj,
                    "generated_human_only_value_usd": value,
                    "value_per_mj": value / energy_mj,
                    "total_human_only_hours_saved": hours_saved,
                    "hours_saved_per_mj": hours_saved / energy_mj,
                }
            )
        rank_rows(category_rows)
        qualifying = [
            row for row in prompts if row["success_count"] >= threshold
        ]
        total_value = sum(row["human_only_value_usd"] for row in qualifying)
        total_hours_saved = sum(
            row["human_only_hours"] - float(row["mean_success_latency_hours"])
            for row in qualifying
        )
        successful_attempts = sum(row["success"] for row in attempts)
        thresholds[str(threshold)] = {
            "category_rows": category_rows,
            "overall": {
                "threshold": threshold,
                "qualifying_prompts": len(qualifying),
                "total_prompts": 220,
                "successful_attempts": successful_attempts,
                "total_attempts": 880,
                "attempt_win_or_tie_rate": successful_attempts / 880,
                "energy_mj_all_four_runs": total_energy_mj,
                "generated_human_only_value_usd": total_value,
                "value_per_mj": total_value / total_energy_mj,
                "total_human_only_hours_saved": total_hours_saved,
                "hours_saved_per_mj": total_hours_saved / total_energy_mj,
            },
        }

    outcomes = Counter(row["outcome"] for row in attempts)
    qa = {
        "passed": True,
        "hf_repo": HF_REPO,
        "hf_revision": HF_REVISION,
        "valuation_prompt_count": len(valuations),
        "crosswalk_prompt_count": len(task_info),
        "prompt_count": len(prompts),
        "attempt_count": len(attempts),
        "attempts_per_prompt_values": sorted(set(attempt_counts.values())),
        "repeat_sets_all_0_to_3": all(
            repeats == {0, 1, 2, 3} for repeats in repeat_sets.values()
        ),
        "category_count": len({row["category"] for row in prompts}),
        "batch_count": len(batches),
        "batch_size_values": sorted({int(row["batch_size"]) for row in batches}),
        "gpu_count_values": sorted(
            {len(row["telemetry_device_ids"]) for row in batches}
        ),
        "source_shard_attempt_counts": dict(sorted(shard_counts.items())),
        "source_energy_j": float(source_energy_j),
        "allocated_attempt_energy_j": total_energy_mj * 1_000_000,
        "energy_reconciliation_error_j": (
            total_energy_mj * 1_000_000 - float(source_energy_j)
        ),
        "missing_wage_values": sum(row["hourly_wage"] is None for row in prompts),
        "missing_human_time_values": sum(
            row["human_only_hours"] is None for row in prompts
        ),
        "missing_human_value_values": sum(
            row["human_only_value_usd"] is None for row in prompts
        ),
        "missing_latency_values": sum(
            row["latency_seconds"] is None for row in attempts
        ),
        "missing_energy_values": sum(
            row["allocated_attempt_energy_j"] is None for row in attempts
        ),
        "outcomes": dict(sorted(outcomes.items())),
        "successful_attempt_count": sum(row["success"] for row in attempts),
        "completed_source_attempt_count": sum(row["completed"] for row in attempts),
        "timed_out_source_attempt_count": sum(row["timed_out"] for row in attempts),
    }
    required = [
        len(valuations) == 220,
        len(task_info) == 220,
        len(prompts) == 220,
        len(attempts) == 880,
        set(attempt_counts.values()) == {4},
        all(repeats == {0, 1, 2, 3} for repeats in repeat_sets.values()),
        len({row["category"] for row in prompts}) == 13,
        len(batches) == 44,
        len(batch_by_key) == 880,
        all(row["batch_size"] == 20 for row in batches),
        all(len(row["telemetry_device_ids"]) == 8 for row in batches),
        abs(qa["energy_reconciliation_error_j"]) < 1e-6,
        sum(outcomes.values()) == len(attempts) and set(outcomes) <= {"win", "tie", "loss"},
    ]
    qa["passed"] = all(required)
    if not qa["passed"]:
        raise AssertionError(f"QA failed: {qa}")

    return {
        "methodology": {
            "study": "Gemma 4 31B GDPVal value/energy analysis",
            "judge": "Gemini 3.1 Flash-Lite pairwise model judge",
            "primary_baseline": "human",
            "success_definition": "agent_outcome in {win, tie}",
            "thresholds": [1, 2, 3, 4],
            "generated_value_rule": (
                "Count each qualifying prompt's output dollar value once."
            ),
            "time_saved_rule": (
                "Recorded expert completion time minus mean AI-system completion time across "
                "successful attempts; negative values are retained."
            ),
            "energy_rule": (
                "All four Gemma attempts for every prompt, including losses; "
                "eight-GPU source batch energy is divided equally across each "
                "20-attempt batch. Judge energy is excluded."
            ),
            "hf_repo": HF_REPO,
            "hf_revision": HF_REVISION,
        },
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
            "valuation_workbook_sha256": sha256(VALUATION_WORKBOOK),
            "crosswalk_sha256": sha256(SOURCE_STATE / "crosswalk.json"),
            "source_outputs_sha256": sha256(source_path),
            "results_sha256": sha256(result_path),
            "batch_files_sha256": batch_hashes,
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gemma_value_energy_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n"
    )
    write_csv(
        output_dir / "primary_ranking_at_least_1_of_4.csv",
        report["thresholds"]["1"]["category_rows"],
    )
    write_csv(output_dir / "prompt_ledger.csv", report["prompt_ledger"])
    write_csv(output_dir / "attempt_ledger.csv", report["attempt_ledger"])
    write_csv(output_dir / "batch_energy_qa.csv", report["batch_qa"])
    for threshold in range(1, 5):
        write_csv(
            output_dir / f"category_ranking_at_least_{threshold}_of_4.csv",
            report["thresholds"][str(threshold)]["category_rows"],
        )



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.output_dir)
    write_outputs(args.output_dir, report)
    print(json.dumps({"output_dir": str(args.output_dir), "qa": report["qa"]}))


if __name__ == "__main__":
    main()
