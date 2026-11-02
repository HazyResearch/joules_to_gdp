"""Checks for the public, fabricated GPU+CPU+DRAM input contract."""
import hashlib
import json
from pathlib import Path
import tempfile

import pandas as pd
import pytest

from energy_boundary import ROOT, build, sha256, task_occupations
from build_energy_components_from_ipw import build_from_ipw, raw_meter_provenance, subset_mapping

FIXTURE = ROOT / "analysis/fixtures/energy_boundary"
FIXTURE_PROTOCOL = dict(
    expected_profiles=("fixture",), expected_tasks=2, expected_repeats=(0,),
    expected_batches=1, expected_batch_size=2, expected_occupations=1,
    task_inventory_path=FIXTURE / "tasks.json",
)


def test_complete_components_reconcile_and_follow_request_overlap():
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        result = build(
            FIXTURE / "batches.csv", FIXTURE / "requests.csv", output,
            **FIXTURE_PROTOCOL,
        )
        attempts = pd.read_csv(output / "attempt_energy.csv").set_index("task_id")
        assert result["profiles"]["fixture"]["dram_j"] == 5
        assert attempts.loc["fabricated-task-1", "total_energy_j"] == pytest.approx(93.75)
        assert attempts.loc["fabricated-task-2", "total_energy_j"] == pytest.approx(31.25)
        assert attempts.total_energy_j.sum() == pytest.approx(125)


def test_missing_dram_and_invalid_source_hash_fail_closed():
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        batches = pd.read_csv(FIXTURE / "batches.csv")
        batches.drop(columns="dram_energy_j").to_csv(output / "missing_dram.csv", index=False)
        with pytest.raises(ValueError, match="dram_energy_j"):
            build(output / "missing_dram.csv", FIXTURE / "requests.csv", output,
                  **FIXTURE_PROTOCOL)
        batches.loc[0, "source_sha256"] = "0" * 64
        batches.to_csv(output / "wrong_hash.csv", index=False)
        with pytest.raises(ValueError, match="Source hash mismatch"):
            build(output / "wrong_hash.csv", FIXTURE / "requests.csv", output,
                  **FIXTURE_PROTOCOL)


def test_ipw_adapter_derives_component_and_request_csvs():
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        assert build_from_ipw(FIXTURE / "ipw_config.json", output) == (1, 2)
        requests = pd.read_csv(output / "energy_component_requests.csv")
        assert requests["mapping_source_path"].isna().all()
        assert requests["trace_source_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
        result = build(
            output / "energy_component_batches.csv", output / "energy_component_requests.csv", output,
            **FIXTURE_PROTOCOL,
        )
        assert len(result["source_sha256"]) == 2


def test_primary_batch_must_be_occupation_pure():
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        inventory = json.loads((FIXTURE / "tasks.json").read_text())
        inventory[1]["occupation"] = "Different Fabricated Occupation"
        task_path = output / "mixed_tasks.json"
        task_path.write_text(json.dumps(inventory))
        with pytest.raises(ValueError, match="occupation-pure"):
            build(FIXTURE / "batches.csv", FIXTURE / "requests.csv", output,
                  **{**FIXTURE_PROTOCOL, "task_inventory_path": task_path, "expected_occupations": 2})


def test_agent_instance_inventory_requires_complete_zero_based_repeats():
    rows = [
        {
            "task_id": f"fabricated-task-{task}",
            "instance_id": f"fabricated-task-{task}::rep{repeat:02d}",
            "repeat_index": repeat,
            "repeat_count": 4,
            "occupation": "Fabricated Occupation",
            "occupation_batch_index": 0,
            "occupation_batch_id": "occ_b00",
            "occupation_task_position": task,
            "source_task_index": task,
        }
        for task in range(2) for repeat in range(4)
    ]
    expected = {
        "fabricated-task-0": "Fabricated Occupation",
        "fabricated-task-1": "Fabricated Occupation",
    }
    options = dict(expected_tasks=2, expected_repeats=(0, 1, 2, 3),
                   expected_batches=1, expected_batch_size=8, expected_occupations=1,
                   require_instances=True)
    assert task_occupations(rows, **options) == expected
    incomplete = rows[:-1]
    with pytest.raises(ValueError, match="wrong task or instance count"):
        task_occupations(incomplete, **options)
    wrong_batch = [dict(item) for item in rows]
    wrong_batch[-1]["occupation_batch_id"] = "occ_b01"
    with pytest.raises(ValueError, match="metadata differs"):
        task_occupations(wrong_batch, **options)
    with pytest.raises(ValueError, match="must contain repeated"):
        task_occupations([{"task_id": task, "occupation": occupation}
                          for task, occupation in expected.items()], **options)


def test_raw_sample_manifest_and_ipw_summary_must_agree():
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        raw = output / "telemetry_samples.jsonl"
        raw.write_text(json.dumps({"schema_version": 1, "sample_time_s": 1.0}) + "\n")
        manifest = {"schema_version": 1, "path": raw.name, "sample_count": 1, "sha256": sha256(raw)}
        (output / "telemetry_samples_manifest.json").write_text(json.dumps(manifest))
        (output / "summary.json").write_text(json.dumps({"config": {"telemetry_samples": manifest}}))
        assert raw_meter_provenance(raw, sha256(raw), require_summary=True) == (
            output / "telemetry_samples_manifest.json", output / "summary.json")
        raw.write_text(raw.read_text() + json.dumps({"schema_version": 1, "sample_time_s": 2.0}) + "\n")
        with pytest.raises(ValueError, match="checksum"):
            raw_meter_provenance(raw, sha256(raw), require_summary=True)


def test_subset_manifest_rep01_to_rep04_is_zero_based_and_hashed():
    records = [
        {"query_id": f"q{i}", "stable_ids": {"task_id": "fabricated-task", "instance_id": f"fabricated-task::rep{i:02d}"}}
        for i in range(1, 5)
    ]
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "subset_manifest.json"
        path.write_text(json.dumps({"records": records, "subset_size": 4, "subset_hash": digest}))
        assert subset_mapping(path) == {f"q{i}": ("fabricated-task", i - 1) for i in range(1, 5)}
        payload = json.loads(path.read_text())
        payload["subset_hash"] = "0" * 64
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="hash mismatch"):
            subset_mapping(path)
