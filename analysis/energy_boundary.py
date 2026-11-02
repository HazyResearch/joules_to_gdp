#!/usr/bin/env python3
"""Validate and attribute GPU, CPU, and DRAM energy for recorded deployments.

The two external CSVs must cover every physical request, including retries.
All timestamps are seconds on the same clock within a batch; all energies are
joules measured over that batch's execution window. Each batch cites a staged
measurement artifact by repository-relative path and SHA-256 digest. The IPW
adapter cites batches.jsonl, a summary of meter readings, and can additionally
cite raw meter samples if available. The three components are attributed
independently with the recorded request intervals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from active_request_energy import RequestInterval, attribute_batch

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "reproduction/inputs"
OUTPUT_DIR = ROOT / "outputs/energy_boundary"
AGENT_INSTANCE_INVENTORY = INPUT_DIR / "gdpval_agent_occupation_repeat4_instances.json"
AGENT_INSTANCE_MANIFEST_PATH = "cluster/gdpval_occupation_repeat4_instances.json"
PROFILES = (
    "glm52_h200", "qwen36_27b", "gemma4_31b", "gpt_oss_120b", "gpt_oss_20b",
    "qwen_h100", "qwen_h200", "qwen_b200", "qwen_mi350x",
)
COMPONENTS = ("gpu", "cpu", "dram")
BATCH_COLUMNS = {
    "profile", "batch_key", "batch_role", "batch_start_s", "batch_end_s",
    "gpu_energy_j", "cpu_energy_j", "dram_energy_j", "source_path", "source_sha256",
}
REQUEST_COLUMNS = {
    "profile", "batch_key", "physical_request_id", "task_id", "repeat", "start_s", "end_s",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_frame(path: Path, columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Total energy requires {path}: staged GPU, CPU, and DRAM evidence is missing"
        )
    frame = pd.read_csv(path, dtype={
        "profile": str, "task_id": str, "batch_key": str,
        "physical_request_id": str, "source_path": str, "source_sha256": str,
        "source_batch_id": str, "batch_role": str,
    })
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame.empty or frame[list(columns)].isna().any().any():
        raise ValueError(f"{path}: empty or missing energy-boundary fields")
    return frame


def source_file(relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"source_path must be repository-relative: {relative_path}")
    return ROOT / path


def task_occupations(
    inventory: object,
    *,
    expected_tasks: int,
    expected_repeats: tuple[int, ...],
    expected_batches: int,
    expected_batch_size: int,
    expected_occupations: int,
    require_instances: bool = False,
) -> dict[str, str]:
    """Validate an agent-instance roster, or an explicit fabricated task fixture."""
    if not isinstance(inventory, list) or not all(isinstance(item, dict) for item in inventory):
        raise ValueError("Task inventory must be a JSON list of objects")
    instances = bool(inventory and "instance_id" in inventory[0])
    if require_instances and not instances:
        raise ValueError("Recorded agent inventory must contain repeated task instances")
    expected_rows = expected_tasks * len(expected_repeats) if instances else expected_tasks
    if len(inventory) != expected_rows:
        raise ValueError("Task inventory has the wrong task or instance count")

    occupation_by_task: dict[str, str] = {}
    repeat_by_task: dict[str, set[int]] = {}
    batch_by_task: dict[str, int] = {}
    source_by_task: dict[str, int] = {}
    position_by_task: dict[str, int] = {}
    instance_ids: set[str] = set()
    batch_rows: dict[int, list[dict]] = {}
    for item in inventory:
        task_id, occupation = item.get("task_id"), item.get("occupation")
        if not isinstance(task_id, str) or not task_id.strip() or not isinstance(occupation, str) or not occupation.strip():
            raise ValueError("Task inventory has a missing task ID or occupation")
        task_id, occupation = task_id.strip(), occupation.strip()
        if task_id in occupation_by_task and occupation_by_task[task_id] != occupation:
            raise ValueError("Task inventory assigns one task to multiple occupations")
        occupation_by_task[task_id] = occupation
        if not instances:
            continue
        repeat, batch, position, source = (
            item.get("repeat_index"), item.get("occupation_batch_index"),
            item.get("occupation_task_position"), item.get("source_task_index"),
        )
        if (type(repeat) is not int or repeat not in expected_repeats
                or type(batch) is not int or not 0 <= batch < expected_batches
                or type(position) is not int or not 0 <= position < expected_batch_size // len(expected_repeats)
                or type(source) is not int or not 0 <= source < expected_tasks
                or item.get("repeat_count") != len(expected_repeats)
                or item.get("occupation_batch_id") != f"occ_b{batch:02d}"
                or item.get("instance_id") != f"{task_id}::rep{repeat:02d}"):
            raise ValueError("Agent task instance metadata differs from the recorded batch protocol")
        instance_id = item["instance_id"]
        if instance_id in instance_ids:
            raise ValueError("Duplicate agent task instance ID")
        instance_ids.add(instance_id)
        for mapping, value in ((batch_by_task, batch), (source_by_task, source),
                               (position_by_task, position)):
            if task_id in mapping and mapping[task_id] != value:
                raise ValueError("Agent task metadata changes between repeats")
            mapping[task_id] = value
        repeat_by_task.setdefault(task_id, set()).add(repeat)
        batch_rows.setdefault(batch, []).append(item)

    if len(occupation_by_task) != expected_tasks:
        raise ValueError("Task inventory has duplicate or missing task IDs")
    occupation_counts = pd.Series(list(occupation_by_task.values())).value_counts()
    if (len(occupation_counts) != expected_occupations
            or not (occupation_counts == expected_tasks // expected_occupations).all()):
        raise ValueError("Task inventory does not have the expected occupations and tasks per occupation")
    if instances:
        if (set(batch_rows) != set(range(expected_batches))
                or set(source_by_task.values()) != set(range(expected_tasks))
                or any(repeats != set(expected_repeats) for repeats in repeat_by_task.values())):
            raise ValueError("Agent inventory has incomplete batch, source-task, or repeat coverage")
        for batch, rows in batch_rows.items():
            tasks = {item["task_id"] for item in rows}
            occupations = {item["occupation"] for item in rows}
            positions = {item["occupation_task_position"] for item in rows}
            if (len(rows) != expected_batch_size or len(occupations) != 1
                    or len(tasks) != expected_batch_size // len(expected_repeats)
                    or positions != set(range(len(tasks)))
                    or any(batch_by_task[task] != batch for task in tasks)):
                raise ValueError("Agent inventory batch is not occupation-pure with complete task repeats")
    return occupation_by_task


def build(
    batch_path: Path,
    request_path: Path,
    output_dir: Path,
    *,
    expected_profiles: tuple[str, ...] = PROFILES,
    expected_tasks: int = 220,
    expected_repeats: tuple[int, ...] = (0, 1, 2, 3),
    expected_batches: int = 44,
    expected_batch_size: int = 20,
    expected_occupations: int = 44,
    task_inventory_path: Path = AGENT_INSTANCE_INVENTORY,
) -> dict:
    batches = checked_frame(batch_path, BATCH_COLUMNS)
    requests = checked_frame(request_path, REQUEST_COLUMNS)
    if not task_inventory_path.is_file():
        raise FileNotFoundError(f"Task inventory is required to validate occupation-pure batches: {task_inventory_path}")
    inventory_digest = sha256(task_inventory_path)
    if task_inventory_path == AGENT_INSTANCE_INVENTORY:
        manifest = json.loads((ROOT / "reproduction/input_manifest.json").read_text())
        recorded = [item.get("sha256") for item in manifest["inputs"]
                    if item["path"] == AGENT_INSTANCE_MANIFEST_PATH]
        if recorded != [inventory_digest]:
            raise ValueError("Agent task inventory differs from recorded SHA-256")
    inventory = json.loads(task_inventory_path.read_text(encoding="utf-8"))
    occupation_by_task = task_occupations(
        inventory, expected_tasks=expected_tasks, expected_repeats=expected_repeats,
        expected_batches=expected_batches, expected_batch_size=expected_batch_size,
        expected_occupations=expected_occupations,
        require_instances=task_inventory_path == AGENT_INSTANCE_INVENTORY,
    )
    if set(batches.profile) != set(expected_profiles) or set(requests.profile) != set(expected_profiles):
        raise ValueError("Energy ledgers must contain exactly the expected model and hardware profiles")
    if batches.duplicated(["profile", "batch_key"]).any():
        raise ValueError("Duplicate physical batch identity")
    if requests.duplicated(["profile", "physical_request_id"]).any():
        raise ValueError("Duplicate physical request identity")
    if set(map(tuple, requests[["profile", "batch_key"]].to_numpy())) != set(
        map(tuple, batches[["profile", "batch_key"]].to_numpy())
    ):
        raise ValueError("Batch coverage is incomplete or has no requests")

    for column in ("batch_start_s", "batch_end_s", *(f"{part}_energy_j" for part in COMPONENTS)):
        batches[column] = pd.to_numeric(batches[column], errors="raise")
        if not np.isfinite(batches[column]).all():
            raise ValueError(f"Non-finite batch {column}")
    for column in ("start_s", "end_s"):
        requests[column] = pd.to_numeric(requests[column], errors="raise")
        if not np.isfinite(requests[column]).all():
            raise ValueError(f"Non-finite request {column}")
    repeat_values = pd.to_numeric(requests["repeat"], errors="raise")
    if not np.isfinite(repeat_values).all() or not (repeat_values == np.floor(repeat_values)).all():
        raise ValueError("Repeat IDs must be finite integers")
    requests["repeat"] = repeat_values.astype(int)
    if not batches.batch_role.isin({"primary", "repair"}).all():
        raise ValueError("Batch role must be primary or repair")
    if set(requests.task_id) != set(occupation_by_task):
        raise ValueError("Energy request task IDs differ from the task inventory")
    request_batches = requests.merge(batches[["profile", "batch_key", "batch_role"]],
                                     on=["profile", "batch_key"], validate="many_to_one")
    for profile in expected_profiles:
        primary = request_batches[request_batches.profile.eq(profile) & request_batches.batch_role.eq("primary")]
        if primary.batch_key.nunique() != expected_batches or len(primary) != expected_batches * expected_batch_size:
            raise ValueError(f"{profile}: wrong primary occupation-batch count or size")
        if primary.duplicated(["task_id", "repeat"]).any():
            raise ValueError(f"{profile}: duplicate primary task/repeat")
        if set(primary.task_id) != set(occupation_by_task) or len(primary) != expected_tasks * len(expected_repeats):
            raise ValueError(f"{profile}: incomplete primary task/repeat panel")
        if not primary.groupby("task_id").repeat.apply(lambda values: set(values) == set(expected_repeats)).all():
            raise ValueError(f"{profile}: primary batches have wrong repeat IDs")
        seen_occupations = set()
        for _, group in primary.groupby("batch_key"):
            occupations = {occupation_by_task[task_id] for task_id in group.task_id}
            if (len(group) != expected_batch_size or len(occupations) != 1
                    or group.task_id.nunique() != expected_batch_size // len(expected_repeats)):
                raise ValueError(f"{profile}: primary batch is not occupation-pure with complete task repeats")
            occupation = next(iter(occupations))
            if occupation in seen_occupations:
                raise ValueError(f"{profile}: occupation spans multiple primary batches")
            seen_occupations.add(occupation)
        if seen_occupations != set(occupation_by_task.values()):
            raise ValueError(f"{profile}: primary occupation coverage is incomplete")
    if not (batches.batch_end_s > batches.batch_start_s).all():
        raise ValueError("Batch windows must have positive duration")
    if not (requests.end_s > requests.start_s).all():
        raise ValueError("Request windows must have positive duration")
    for part in COMPONENTS:
        if not (batches[f"{part}_energy_j"] > 0).all():
            raise ValueError(f"Every batch requires measured, positive {part.upper()} joules")

    verified_sources: dict[str, str] = {}
    source_pairs = [(batches, "source_path", "source_sha256")]
    for prefix in ("raw_meter", "raw_meter_manifest", "summary", "run_provenance", "energy_sources"):
        pair = (f"{prefix}_source_path", f"{prefix}_source_sha256")
        if len(set(pair) & set(batches)) == 1:
            raise ValueError(f"{prefix} provenance requires both path and SHA-256 columns")
        if all(column in batches for column in pair):
            source_pairs.append((batches, *pair))
    for prefix in ("trace", "mapping"):
        pair = (f"{prefix}_source_path", f"{prefix}_source_sha256")
        if len(set(pair) & set(requests)) == 1:
            raise ValueError(f"{prefix} provenance requires both path and SHA-256 columns")
        if all(column in requests for column in pair):
            source_pairs.append((requests, *pair))
    for frame, path_col, hash_col in source_pairs:
        for source_path, source_hash in frame[[path_col, hash_col]].drop_duplicates().itertuples(index=False, name=None):
            if pd.isna(source_path) or not source_path:
                if pd.isna(source_hash) or not source_hash:
                    continue
                raise ValueError(f"Source SHA-256 has no path in {path_col}")
            if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
                raise ValueError(f"Invalid source SHA-256 for {source_path}")
            if source_path in verified_sources and verified_sources[source_path] != source_hash:
                raise ValueError(f"Conflicting source hashes for {source_path}")
            actual = sha256(source_file(source_path))
            if actual != source_hash:
                raise ValueError(f"Source hash mismatch for {source_path}")
            verified_sources[source_path] = actual
    if "source_batch_id" in batches:
        if batches.source_batch_id.isna().any():
            raise ValueError("IPW-derived batch rows require source_batch_id")
        for source_path, group in batches.groupby("source_path"):
            records = {}
            with source_file(source_path).open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        batch_id = str(record["batch_id"])
                        if batch_id in records:
                            raise ValueError(f"Duplicate raw IPW batch {batch_id} in {source_path}")
                        records[batch_id] = record
            for row in group.itertuples():
                raw = records.get(row.source_batch_id)
                if raw is None:
                    raise ValueError(f"Missing raw IPW batch {row.source_batch_id} in {source_path}")
                for column, raw_key in (
                    ("batch_start_s", "batch_prompt_start_time_s"),
                    ("batch_end_s", "batch_prompt_end_time_s"),
                    ("gpu_energy_j", "batch_gpu_energy_joules"),
                    ("cpu_energy_j", "batch_cpu_energy_joules"),
                    ("dram_energy_j", "batch_dram_energy_joules"),
                ):
                    raw_value = raw.get(raw_key)
                    if raw_value is None or not np.isclose(getattr(row, column), float(raw_value), rtol=1e-12, atol=1e-6):
                        raise ValueError(f"{source_path}/{row.source_batch_id}: {column} differs from raw IPW measurement")

    allocations = []
    checks = []
    for batch in batches.itertuples():
        group = requests[
            requests.profile.eq(batch.profile) & requests.batch_key.eq(batch.batch_key)
        ]
        intervals = [
            RequestInterval(row.physical_request_id, row.start_s, row.end_s)
            for row in group.itertuples()
        ]
        attributed = {
            part: attribute_batch(
                intervals,
                batch_start_s=batch.batch_start_s,
                batch_end_s=batch.batch_end_s,
                batch_energy_j=getattr(batch, f"{part}_energy_j"),
            )
            for part in COMPONENTS
        }
        for row in group.itertuples():
            record = {
                "profile": row.profile,
                "batch_key": row.batch_key,
                "physical_request_id": row.physical_request_id,
                "task_id": row.task_id,
                "repeat": int(row.repeat),
            }
            for part, result in attributed.items():
                record[f"{part}_energy_j"] = result.request_energy_j[row.physical_request_id]
                record[f"{part}_equal_energy_j"] = result.equal_split_energy_j[row.physical_request_id]
            allocations.append(record)
        for part, result in attributed.items():
            checks.append({
                "profile": batch.profile, "batch_key": batch.batch_key, "component": part,
                "measured_energy_j": getattr(batch, f"{part}_energy_j"),
                "allocated_energy_j": sum(result.request_energy_j.values()),
                "reconciliation_error_j": result.reconciliation_error_j,
                "method": result.method,
            })
    physical = pd.DataFrame(allocations)
    logical = physical.groupby(["profile", "task_id", "repeat"], as_index=False)[
        [f"{part}{'_' + kind if kind else ''}_energy_j"
         for part in COMPONENTS for kind in ("", "equal")]
    ].sum()
    logical["total_energy_j"] = logical[[f"{part}_energy_j" for part in COMPONENTS]].sum(axis=1)
    logical["equal_total_energy_j"] = logical[
        [f"{part}_equal_energy_j" for part in COMPONENTS]
    ].sum(axis=1)
    profile_validation = {}
    for profile in expected_profiles:
        rows = logical[logical.profile.eq(profile)]
        if len(rows) != expected_tasks * len(expected_repeats) or rows.task_id.nunique() != expected_tasks:
            raise ValueError(f"{profile}: incomplete task/repeat energy coverage")
        if not rows.groupby("task_id").repeat.apply(lambda x: set(x) == set(expected_repeats)).all():
            raise ValueError(f"{profile}: wrong repeat IDs")
        profile_validation[profile] = {
            "logical_attempts": len(rows),
            "physical_requests": int(physical.profile.eq(profile).sum()),
            "batches": int(batches.profile.eq(profile).sum()),
            "gpu_j": float(rows.gpu_energy_j.sum()),
            "cpu_j": float(rows.cpu_energy_j.sum()),
            "dram_j": float(rows.dram_energy_j.sum()),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    physical.to_csv(output_dir / "physical_request_energy.csv", index=False, float_format="%.17g")
    logical.to_csv(output_dir / "attempt_energy.csv", index=False, float_format="%.17g")
    pd.DataFrame(checks).to_csv(output_dir / "batch_reconciliation.csv", index=False, float_format="%.17g")
    validation = {
        "energy_boundary": "measured GPU+CPU+DRAM, attributed over complete physical batch windows",
        "batch_input_sha256": sha256(batch_path),
        "request_input_sha256": sha256(request_path),
        "task_inventory_path": task_inventory_path.relative_to(ROOT).as_posix() if task_inventory_path.is_relative_to(ROOT) else str(task_inventory_path),
        "task_inventory_sha256": inventory_digest,
        "source_sha256": verified_sources,
        "attempt_output_sha256": sha256(output_dir / "attempt_energy.csv"),
        "profiles": profile_validation,
    }
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    return validation


def attempt_energy(path: Path = OUTPUT_DIR) -> pd.DataFrame:
    ledger = path / "attempt_energy.csv"
    validation = path / "validation.json"
    if not ledger.is_file() or not validation.is_file():
        raise FileNotFoundError(
            "Total-energy metrics require GPU+CPU+DRAM attribution. Run "
            "analysis/energy_boundary.py with the external batches and requests CSVs first."
        )
    metadata = json.loads(validation.read_text())
    if metadata.get("energy_boundary") != "measured GPU+CPU+DRAM, attributed over complete physical batch windows":
        raise ValueError("Energy boundary must include measured GPU+CPU+DRAM")
    if sha256(ledger) != metadata.get("attempt_output_sha256"):
        raise ValueError("Total-energy attempt ledger changed after validation")
    for name, digest in (
        ("energy_component_batches.csv", metadata.get("batch_input_sha256")),
        ("energy_component_requests.csv", metadata.get("request_input_sha256")),
    ):
        source = INPUT_DIR / name
        if not source.is_file() or sha256(source) != digest:
            raise ValueError(f"Staged energy input changed or is missing: {source}")
    for source, digest in metadata.get("source_sha256", {}).items():
        if sha256(source_file(source)) != digest:
            raise ValueError(f"Energy evidence source changed: {source}")
    inventory = Path(metadata["task_inventory_path"])
    inventory = inventory if inventory.is_absolute() else source_file(str(inventory))
    if not inventory.is_file() or sha256(inventory) != metadata.get("task_inventory_sha256"):
        raise ValueError("Validated task inventory changed or is missing")
    frame = pd.read_csv(ledger, dtype={"task_id": str})
    needed = {"profile", "task_id", "repeat", "gpu_energy_j", "cpu_energy_j", "dram_energy_j", "total_energy_j", "equal_total_energy_j"}
    if not needed <= set(frame.columns) or frame[list(needed)].isna().any().any():
        raise ValueError("Incomplete total-energy attempt ledger")
    np.testing.assert_allclose(
        frame.total_energy_j,
        frame[["gpu_energy_j", "cpu_energy_j", "dram_energy_j"]].sum(axis=1), rtol=1e-12,
    )
    if not (frame[["gpu_energy_j", "cpu_energy_j", "dram_energy_j"]] > 0).all().all():
        raise ValueError("Missing positive measured energy component")
    if set(frame.profile) != set(PROFILES):
        raise ValueError("Total-energy input does not cover all configured deployments")
    for profile in PROFILES:
        rows = frame[frame.profile.eq(profile)]
        if len(rows) != 880 or rows.task_id.nunique() != 220:
            raise ValueError(f"{profile}: incomplete total-energy task panel")
        if not rows.groupby("task_id").repeat.apply(lambda values: set(values) == {0, 1, 2, 3}).all():
            raise ValueError(f"{profile}: incomplete four-repeat task panel")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=Path, default=INPUT_DIR / "energy_component_batches.csv")
    parser.add_argument("--requests", type=Path, default=INPUT_DIR / "energy_component_requests.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--fixture", action="store_true", help="Validate the fabricated two-task fixture")
    args = parser.parse_args()
    if args.fixture:
        fixture = ROOT / "analysis/fixtures/energy_boundary"
        args.batches = fixture / "batches.csv"
        args.requests = fixture / "requests.csv"
        result = build(args.batches, args.requests, args.output_dir,
                       expected_profiles=("fixture",), expected_tasks=2, expected_repeats=(0,),
                       expected_batches=1, expected_batch_size=2, expected_occupations=1,
                       task_inventory_path=fixture / "tasks.json")
    else:
        result = build(args.batches, args.requests, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
