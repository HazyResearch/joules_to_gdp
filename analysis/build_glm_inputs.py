#!/usr/bin/env python3
"""Build GDPVal GLM query and batch inputs from raw run records.

These inputs contain GPU telemetry. The total-energy denominator requires
separate measured CPU and DRAM energy via analysis/energy_boundary.py.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def valuation_rows(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    values = payload["values"]
    header = values[3]
    rows = [dict(zip(header, row)) for row in values[4:] if row and row[1]]
    if len(rows) != 270:
        raise ValueError(f"Expected 270 valuation rows; found {len(rows)}")
    out = {}
    for row in rows:
        prompt_id = str(row["Prompt ID"])
        out[prompt_id] = {
            "dataset": row["Dataset"],
            "prompt_id": prompt_id,
            "occupation": row["Source Occupation"],
            "source_onet_code": row["Source O*NET Code"],
            "onet_occupation": row["Matched O*NET Occupation"],
            "onet_code": row["Matched O*NET Code"],
            "hourly_wage_usd": float(row["Hourly Wage (USD/hr)"]),
            "human_only_hours": float(row["Human-Only Total (hours)"]),
            "human_ai_hours": float(row["Human-AI Total (hours)"]),
            "human_only_value_usd": float(row["Human-Only Value (USD)"]),
            "human_ai_value_usd": float(row["Human-AI Value (USD)"]),
            "wage_source": row["Wage Source File"],
        }
    if len(out) != 270:
        raise ValueError("Valuation prompt IDs are not unique")
    return out

def filtered_judgments(path: Path) -> dict[tuple[str, int], dict]:
    rows = [
        row
        for row in read_jsonl(path)
        if row.get("condition") == "without_rubric" and row.get("baseline") == "human"
    ]
    out = {(str(row["task_id"]), int(row["repeat"])): row for row in rows}
    if len(out) != len(rows):
        raise ValueError(f"Duplicate judgments in {path}")
    for row in rows:
        if row.get("agent_outcome") not in {"win", "tie", "loss"}:
            raise ValueError(f"Invalid outcome in {path}: {row.get('agent_outcome')}")
        model = str((row.get("api_metadata") or {}).get("model") or "")
        if "gemini-3.1-flash-lite" not in model:
            raise ValueError(f"Unexpected judge model in {path}: {model}")
    return out

def gdp_crosswalk(path: Path) -> dict[str, dict]:
    out = {}
    seen = set()
    for row in read_jsonl(path):
        if row.get("condition") != "without_rubric" or row.get("baseline") != "human":
            continue
        key = (str(row["task_id"]), int(row["repeat"]))
        if key in seen:
            continue
        seen.add(key)
        match = re.search(r"/(q\d+)_", str(row["agent_artifact_dir"]))
        if not match:
            raise ValueError(f"Cannot parse query ID: {row['agent_artifact_dir']}")
        out[match.group(1)] = {
            "judge_task_id": key[0],
            "repeat_index": key[1],
            "prompt_id": str(row["baseline_task_id"]),
            "occupation": row["occupation"],
        }
    if len(out) != 880:
        raise ValueError(f"Expected 880 GDPVal query mappings; found {len(out)}")
    return out

def build_gdp(
    values: dict[str, dict], batches_path: Path, traces_path: Path, judgments_path: Path, manifest_path: Path
) -> tuple[list[dict], list[dict]]:
    batches = read_jsonl(batches_path)
    traces = read_jsonl(traces_path)
    judgments = filtered_judgments(judgments_path)
    mapping = gdp_crosswalk(manifest_path)
    if len(batches) != 44 or len(traces) != 880 or len(judgments) != 880:
        raise ValueError("GDPVal must contain 44 batches, 880 traces, and 880 filtered judgments")
    trace_by_id = {row["query_id"]: row for row in traces}
    queries, batch_rows = [], []
    for batch in batches:
        ids = batch["query_ids"]
        batch_maps = [mapping[qid] for qid in ids]
        prompt_ids = sorted({row["prompt_id"] for row in batch_maps})
        occupations = {values[prompt_id]["occupation"] for prompt_id in prompt_ids}
        if len(occupations) != 1 or len(prompt_ids) != 5:
            raise ValueError(f"GDPVal batch {batch['batch_index']} is not one occupation / five tasks")
        batch_key = f"GDPVal:batch_{int(batch['batch_index']):04d}"
        batch_rows.append(make_batch_row("GDPVal", batch_key, batch, occupations.pop(), prompt_ids))
        for qid in ids:
            link = mapping[qid]
            value = values[link["prompt_id"]]
            judgment = judgments[(link["judge_task_id"], link["repeat_index"])]
            trace = trace_by_id[qid]
            queries.append(
                make_query_row("GDPVal", batch_key, batch, trace, link, value, judgment)
            )
    return queries, batch_rows

def make_batch_row(dataset: str, key: str, batch: dict, occupation: str, prompt_ids: list[str]) -> dict:
    energy_j = float(batch["batch_gpu_energy_joules"])
    return {
        "dataset": dataset,
        "batch_key": key,
        "source_batch_id": batch["batch_id"],
        "source_batch_index": int(batch["batch_index"]),
        "occupation": occupation,
        "unique_tasks": len(prompt_ids),
        "prompt_ids": " | ".join(prompt_ids),
        "attempts": int(batch["batch_size"]),
        "prompt_start_time_s": float(batch["batch_prompt_start_time_s"]),
        "prompt_end_time_s": float(batch["batch_prompt_end_time_s"]),
        "batch_prompt_wall_clock_s": float(batch["batch_prompt_wall_clock_s"]),
        "batch_total_wall_clock_s": float(batch["batch_total_wall_clock_s"]),
        "gpu_energy_joules": energy_j,
        "gpu_energy_kwh": energy_j / 3_600_000,
        "gpu_power_avg_watts": float(batch["batch_gpu_power_avg_watts"]),
        "telemetry_sample_count": int(batch["telemetry_sample_count"]),
        "gpu_device_count": len(batch.get("telemetry_device_ids") or []),
        "telemetry_accounting": batch.get("telemetry_accounting"),
    }

def make_query_row(
    dataset: str, batch_key: str, batch: dict, trace: dict, link: dict, value: dict, judgment: dict
) -> dict:
    size = int(batch["batch_size"])
    outcome = judgment["agent_outcome"]
    success = 1 if outcome in {"win", "tie"} else 0
    energy_j = float(batch["batch_gpu_energy_joules"]) / size
    prompt_runtime_s = float(batch["batch_prompt_wall_clock_s"]) / size
    agent_latency_hours = float(trace["total_wall_clock_s"]) / 3600
    return {
        "dataset": dataset,
        "prompt_id": link["prompt_id"],
        "judge_task_id": link["judge_task_id"],
        "repeat_index": int(link["repeat_index"]),
        "occupation": value["occupation"],
        "onet_occupation": value["onet_occupation"],
        "onet_code": value["onet_code"],
        "hourly_wage_usd": value["hourly_wage_usd"],
        "human_only_hours": value["human_only_hours"],
        "human_only_value_usd": value["human_only_value_usd"],
        "human_ai_hours": value["human_ai_hours"],
        "human_ai_value_usd": value["human_ai_value_usd"],
        "outcome": outcome,
        "success_credit": success,
        "earned_human_only_value_usd": value["human_only_value_usd"] * success,
        "earned_human_ai_value_usd": value["human_ai_value_usd"] * success,
        "earned_value_usd": value["human_only_value_usd"] * success,
        "batch_key": batch_key,
        "query_id": trace["query_id"],
        "batch_position": int(trace["batch_position"]),
        "prompt_start_time_s": float(trace["prompt_start_time_s"]),
        "prompt_end_time_s": float(trace["prompt_end_time_s"]),
        "query_latency_s": float(trace["total_wall_clock_s"]),
        "agent_latency_hours": agent_latency_hours,
        "successful_human_only_vs_ai_saved_hours": value["human_only_hours"] - agent_latency_hours if success else None,
        "successful_human_ai_vs_ai_saved_hours": value["human_ai_hours"] - agent_latency_hours if success else None,
        "successful_human_only_vs_human_ai_saved_hours": value["human_only_hours"] - value["human_ai_hours"] if success else None,
        "time_saved_inclusion": "win/tie included" if success else "loss excluded",
        "allocated_batch_prompt_runtime_s": prompt_runtime_s,
        "allocated_gpu_energy_joules": energy_j,
        "allocated_gpu_energy_kwh": energy_j / 3_600_000,
        "total_tokens": int(trace.get("total_tokens") or 0),
        "completed": bool(trace.get("completed")),
        "timed_out": bool(trace.get("timed_out")),
        "resolved": bool(trace.get("is_resolved")),
        "judge_model": (judgment.get("api_metadata") or {}).get("model"),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valuation-json", type=Path, required=True)
    parser.add_argument("--gdp-batches", type=Path, required=True)
    parser.add_argument("--gdp-traces", type=Path, required=True)
    parser.add_argument("--gdp-judgments", type=Path, required=True)
    parser.add_argument("--gdp-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    values = valuation_rows(args.valuation_json)
    queries, batches = build_gdp(values, args.gdp_batches, args.gdp_traces, args.gdp_judgments, args.gdp_manifest)
    if len(queries) != 880 or len(batches) != 44:
        raise ValueError("GDPVal source panel must have 880 queries and 44 batches")
    attempts = {}
    for row in queries:
        key = (row["prompt_id"], row["repeat_index"])
        if key in attempts:
            raise ValueError(f"Duplicate prompt attempt: {key}")
        attempts[key] = row
    if len({key[0] for key in attempts}) != 220 or any({r for t, r in attempts if t == task} != {0, 1, 2, 3} for task in {t for t, _ in attempts}):
        raise ValueError("Incomplete four-repeat GDPVal panel")
    write_csv(args.output_dir / "glm52_no_rubric_query_ledger.csv", queries)
    write_csv(args.output_dir / "glm52_gdpval_batch_ledger.csv", batches)
    print(f"Staged {len(queries)} GDPVal GLM requests in {len(batches)} batches")

if __name__ == "__main__":
    main()
