"""Offline contract checks for the public reproduction entry points."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import run_deployment as deployment_runner
import build_metrics
import stage_inputs
import verify_inputs

from run_deployment import (CONFIG, LOCKED_VLLM_AUTOSTART_MODELS, LOCKED_VLLM_VERSION,
                            commands, controlled_run_environment, energy_source_config,
                            load_task_inventory, load_task_instances, nvidia_selection_provenance,
                            occupation_batches,
                            validate_recorded_task_hash, validate_vllm_args,
                            validate_server_manifest, verify_batch_artifacts,
                            verify_existing_server_gpu_selection, verify_server_port_owner)
from verify_inputs import directory_digest, verify


def fabricated_tasks() -> list[dict[str, str]]:
    return [
        {"task_id": f"fabricated-task-{occupation:02d}-{task:02d}",
         "occupation": f"Fabricated occupation {occupation:02d}"}
        for occupation in range(44) for task in range(5)
    ]


class PublicReleaseGuardTests(unittest.TestCase):
    """Check what would be committed even when working files conceal it."""

    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts/check_public_release.py"
        spec = importlib.util.spec_from_file_location("public_release_guard", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import release guard: {path}")
        cls.guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.guard)

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="j2g-release-index-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        subprocess.run(["git", "init", "--quiet", str(self.root)], check=True,
                       capture_output=True)
        patch = mock.patch.object(self.guard, "ROOT", self.root)
        patch.start()
        self.addCleanup(patch.stop)

    def stage(self, relative: str, content: bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        subprocess.run(["git", "add", "--force", "--", relative], cwd=self.root,
                       check=True, capture_output=True)
        return path

    def test_staged_credential_remains_visible_after_working_copy_is_cleaned(self) -> None:
        # Deliberately fabricated; construct the token so this test contains no secret.
        token = "ghp_" + "A" * 36
        staged = f'API_TOKEN = "{token}"\n'.encode()
        for working_copy in ("overwritten", "deleted"):
            with self.subTest(working_copy=working_copy):
                path = self.stage("settings.py", staged)
                if working_copy == "overwritten":
                    path.write_text('API_TOKEN = ""\n', encoding="utf-8")
                else:
                    path.unlink()
                files, modes = self.guard.staged_snapshot()
                self.assertEqual(files[Path("settings.py")], staged)
                issues = self.guard.check_paths(files, modes)
                self.assertTrue(any("possible secret" in issue and "settings.py" in issue
                                    for issue in issues), issues)

    def test_staged_symlink_is_rejected_even_when_target_is_safe(self) -> None:
        self.stage("module.py", b"VALUE = 1\n")
        (self.root / "linked.py").symlink_to("module.py")
        subprocess.run(["git", "add", "--", "linked.py"], cwd=self.root,
                       check=True, capture_output=True)
        files, modes = self.guard.staged_snapshot()
        self.assertEqual(modes[Path("linked.py")], "120000")
        self.assertIn("nonregular staged file: linked.py", self.guard.check_paths(files, modes))

    def test_forced_stage_of_local_outputs_and_credentials_is_rejected(self) -> None:
        excluded = (
            ".cache/state.json", ".ruff_cache/state", ".coverage", ".coverage.host",
            "htmlcov/index.html", "coverage.xml", ".env.production", "module.pyc",
            "module.so", "energy-monitor/target/release/energy-monitor",
            "intelligence-per-watt/src/ipw/telemetry/bin/windows-x86_64/energy-monitor.exe",
        )
        self.stage("src/module.py", b"VALUE = 1\n")
        for name in excluded:
            self.stage(name, b"fabricated local artifact\n")
        files, modes = self.guard.staged_snapshot()
        issues = self.guard.check_paths(files, modes)
        self.assertEqual(len(issues), len(excluded), issues)
        for name in excluded:
            self.assertTrue(any(issue.endswith(f": {name}") for issue in issues), name)
        self.assertFalse(any(issue.endswith(": src/module.py") for issue in issues))

    def test_staged_bundled_source_tamper_cannot_be_hidden_by_restoring_local_file(self) -> None:
        original = b"source,value\nfabricated,1\n"
        sources = (
            "occupation_data/data/input/OEWS.csv",
            "occupation_data/data/input/ONET.csv",
            "occupation_data/data/input/task_statements.csv",
        )
        for name in sources:
            self.stage(name, original)
        manifest = {"inputs": [
            {"path": name, "sha256": hashlib.sha256(original).hexdigest(), "bundled": True}
            for name in sources
        ]}
        self.stage("reproduction/input_manifest.json", json.dumps(manifest).encode())
        files, _ = self.guard.staged_snapshot()
        self.assertEqual(self.guard.check_bundled_sources(files), [])
        path = self.stage(sources[0], b"source,value\nfabricated,2\n")
        path.write_bytes(original)
        files, _ = self.guard.staged_snapshot()
        self.assertEqual(self.guard.check_bundled_sources(files),
                         [f"bundled source checksum mismatch: {sources[0]}"])


class ReleaseContractTests(unittest.TestCase):
    def test_metrics_use_total_components_by_default_and_explicit_accelerator_selection(self) -> None:
        for selection in (None, "accelerator"):
            args = ["build_metrics.py", "--through-step", "source_metrics"]
            if selection:
                args.extend(["--energy-boundary", selection])
            with self.subTest(selection=selection), mock.patch.object(sys, "argv", args), \
                    mock.patch.object(build_metrics.subprocess, "run") as run:
                build_metrics.main()
                calls = run.call_args_list
                self.assertEqual(len(calls), 1 if selection else 2)
                self.assertEqual(calls[-1].args[0][-1], str(build_metrics.ROOT / "analysis/metrics.py"))
                self.assertEqual(calls[-1].kwargs["env"]["J2G_ENERGY_BOUNDARY"], selection or "total")
                if not selection:
                    self.assertEqual(calls[0].args[0][-1], str(build_metrics.ROOT / "analysis/energy_boundary.py"))

    def test_nvidia_gpu_subset_requires_full_uuid_for_server_and_meter(self) -> None:
        config = json.loads(CONFIG.read_text())
        deployment = config["deployments"]["qwen36_h100_1"]
        nvml = mock.Mock()
        nvml.nvmlDeviceGetCount.return_value = 8
        nvml.nvmlDeviceGetHandleByIndex.side_effect = lambda index: index
        nvml.nvmlDeviceGetUUID.side_effect = lambda index: f"GPU-{index:032x}"
        with mock.patch.dict(sys.modules, {"pynvml": nvml}):
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}):
                with self.assertRaisesRegex(RuntimeError, "distinct full GPU-\\* UUIDs"):
                    nvidia_selection_provenance(deployment)
            selected = "GPU-" + f"{3:032x}"
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": selected}):
                evidence = nvidia_selection_provenance(deployment)
        self.assertEqual(evidence["selected_gpu_uuids"], [selected])
        self.assertEqual(evidence["nvml_physical_gpu_count"], 8)
        self.assertTrue(evidence["physical_gpu_subset"])
        with mock.patch("psutil.Process") as process:
            process.return_value.children.return_value = []
            process.return_value.environ.return_value = {"CUDA_VISIBLE_DEVICES": selected}
            verify_existing_server_gpu_selection(1234, selected)
            process.return_value.environ.return_value = {"CUDA_VISIBLE_DEVICES": "GPU-other"}
            with self.assertRaisesRegex(RuntimeError, "differs from the measured GPU UUIDs"):
                verify_existing_server_gpu_selection(1234, selected)
            process.return_value.environ.return_value = {"CUDA_VISIBLE_DEVICES": selected}
            child = mock.Mock(pid=4321)
            child.environ.return_value = {"CUDA_VISIBLE_DEVICES": "GPU-other"}
            process.return_value.children.return_value = [child]
            with self.assertRaisesRegex(RuntimeError, "server process 4321.*differs"):
                verify_existing_server_gpu_selection(1234, selected)

    def test_eight_gpu_subset_rejects_numeric_duplicate_and_unknown_selectors(self) -> None:
        deployment = json.loads(CONFIG.read_text())["deployments"]["qwen36_h100_8"]
        nvml = mock.Mock()
        nvml.nvmlDeviceGetCount.return_value = 16
        nvml.nvmlDeviceGetHandleByIndex.side_effect = lambda index: index
        nvml.nvmlDeviceGetUUID.side_effect = lambda index: f"GPU-{index:032x}"
        selected = ",".join(f"GPU-{index:032x}" for index in range(8))
        with mock.patch.dict(sys.modules, {"pynvml": nvml}):
            for value in ("0,1,2,3,4,5,6,7", selected.rsplit(",", 1)[0] + ",GPU-" + f"{0:032x}",
                          selected.rsplit(",", 1)[0] + ",GPU-" + "f" * 32):
                with self.subTest(value=value), mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": value}):
                    with self.assertRaisesRegex(RuntimeError, "distinct full GPU-\\* UUIDs"):
                        nvidia_selection_provenance(deployment)
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": selected}):
                evidence = nvidia_selection_provenance(deployment)
        self.assertEqual(evidence["selected_gpu_uuids"], selected.split(","))
        self.assertEqual(evidence["nvml_physical_gpu_count"], 16)
        self.assertTrue(evidence["physical_gpu_subset"])

    def test_existing_server_pid_must_own_requested_port(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            verify_server_port_owner(os.getpid(), port)
            with self.assertRaisesRegex(RuntimeError, "do not own loopback port"):
                verify_server_port_owner(os.getpid(), port + 1 if port < 65535 else port - 1)

    def test_existing_server_rejects_wildcard_and_ipv6_only_listeners(self) -> None:
        import psutil

        for host in ("0.0.0.0", "::", "::1"):
            with self.subTest(host=host):
                process = mock.Mock()
                process.children.return_value = []
                process.net_connections.return_value = [mock.Mock(
                    status=psutil.CONN_LISTEN,
                    laddr=mock.Mock(ip=host, port=8000),
                )]
                with mock.patch("psutil.Process", return_value=process):
                    with self.assertRaisesRegex(RuntimeError, "do not own loopback port"):
                        verify_server_port_owner(1234, 8000)

    def test_inherited_ipw_experiment_knobs_are_removed(self) -> None:
        protocol = json.loads(CONFIG.read_text())["protocol"]
        env = controlled_run_environment({
            "IPW_EVAL_API_KEY": "fabricated-private-judge-key",
            "IPW_OPENHANDS_SEED": "999", "IPW_OPENHANDS_REASONING_EFFORT": "high",
            "IPW_OPENHANDS_TEMPERATURE": "0.9", "IPW_GDPVAL_REPAIR_ATTEMPTS": "0",
            "IPW_GPU_DEVICE_ID": "5", "IPW_ENABLE_MEDIA_TRANSCRIPTION": "1",
            "IPW_EXPECTED_METER_PLATFORM": "amd",
            "HF_TOKEN": "fabricated-private-model-key",
        }, protocol=protocol, monitor_port=50111, gpu_count=8, accelerator="H100")
        self.assertEqual(env["IPW_EVAL_API_KEY"], "fabricated-private-judge-key")
        self.assertEqual(env["HF_TOKEN"], "fabricated-private-model-key")
        for name in ("IPW_OPENHANDS_SEED", "IPW_OPENHANDS_REASONING_EFFORT",
                     "IPW_GDPVAL_REPAIR_ATTEMPTS", "IPW_GPU_DEVICE_ID",
                     "IPW_ENABLE_MEDIA_TRANSCRIPTION"):
            self.assertNotIn(name, env)
        self.assertEqual(env["IPW_OPENHANDS_TEMPERATURE"], "0.0")
        self.assertEqual(env["IPW_OPENHANDS_MAX_OUTPUT_TOKENS"], "32768")
        self.assertEqual(env["IPW_EXPECTED_GPU_COUNT"], "8")
        self.assertEqual(env["IPW_EXPECTED_METER_PLATFORM"], "nvidia")
        self.assertEqual(env["IPW_ENERGY_MONITOR_TARGET"], "127.0.0.1:50111")
        self.assertEqual(env["IPW_REQUIRE_GDPVAL_REFERENCES"], "1")
        self.assertEqual(env["IPW_REQUIRE_GDPVAL_JUDGE"], "1")
        amd_env = controlled_run_environment({}, protocol=protocol, monitor_port=50112,
                                             gpu_count=1, accelerator="MI350X")
        self.assertEqual(amd_env["IPW_EXPECTED_METER_PLATFORM"], "amd")

    def test_failed_batch_keeps_provenance_failed_and_no_energy_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = root / "fabricated_instances.json"
            fixture.write_text(json.dumps([
                item for batch in occupation_batches(fabricated_tasks()) for item in batch
            ]))
            lock = root / "intelligence-per-watt" / "uv.lock"
            lock.parent.mkdir()
            lock.write_text("fabricated lock")
            meter = root / "meter"
            meter.write_text("fabricated meter")
            manifest = root / "server.json"
            manifest.write_text(json.dumps({
                "model": "Qwen/Qwen3.6-27B",
                "model_revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
                "server_pid": 1234,
                "runtime": "vllm", "vllm_version": "0.28.0",
                "runtime_artifact_sha256": "a" * 64,
                "host": "127.0.0.1", "port": 8000,
                "tensor_parallel_size": 8, "max_model_len": 131072, "max_num_seqs": 20,
                "api_key": "do-not-copy-to-provenance",
            }))
            argv = ["run_deployment.py", "qwen36_h100_8", "--tasks-file", str(fixture),
                    "--output-dir", str(root / "runs"), "--existing-server",
                    "--server-manifest", str(manifest), "--execute"]
            with (mock.patch.object(deployment_runner, "ROOT", root),
                  mock.patch.object(deployment_runner, "RUNS", root / "runs"),
                  mock.patch.object(deployment_runner, "source_revision", return_value="fabricated-commit"),
                  mock.patch.object(deployment_runner, "validate_recorded_task_hash"),
                  mock.patch.object(deployment_runner, "require_runtime", return_value=meter),
                  mock.patch.object(deployment_runner, "choose_monitor_port", return_value=50111),
                  mock.patch.object(deployment_runner, "gpu_provenance", return_value=[]),
                  mock.patch.object(deployment_runner, "nvidia_selection_provenance", return_value={}),
                  mock.patch.object(deployment_runner, "verify_server_port_owner"),
                  mock.patch.object(deployment_runner, "wait_for_server"),
                  mock.patch.object(deployment_runner, "validate_served_model"),
                  mock.patch.object(deployment_runner, "run_batch_command", side_effect=RuntimeError("fabricated batch failure")),
                  mock.patch.object(sys, "argv", argv),
                  mock.patch.dict(os.environ, {"GEMINI_API_KEY": "fabricated"})):
                with self.assertRaisesRegex(RuntimeError, "fabricated batch failure"):
                    deployment_runner.main()
            run_dir = next((root / "runs" / "qwen36_h100_8").iterdir())
            provenance = json.loads((run_dir / "provenance.json").read_text())
            self.assertEqual(provenance["status"], "failed")
            self.assertEqual(provenance["active_batch"], "occ_b00")
            self.assertEqual(provenance["verified_batches"], [])
            self.assertNotIn("api_key", provenance["server_claim"])
            self.assertNotIn("do-not-copy-to-provenance", (run_dir / "provenance.json").read_text())
            self.assertNotIn("fabricated", json.dumps(provenance["controlled_ipw_environment"]))
            self.assertFalse((run_dir / "energy_sources.json").exists())

    def test_vllm_protocol_flags_cannot_be_overridden(self) -> None:
        validate_vllm_args(["--gpu-memory-utilization", "0.9"])
        for value in ("--model", "--revision=mutable", "--port", "--host=0.0.0.0",
                      "--max-model-len", "--max-num-seqs", "--api-key"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "cannot override"):
                validate_vllm_args([value])

    def test_external_server_manifest_attests_runtime_and_protocol(self) -> None:
        config = json.loads(CONFIG.read_text())
        deployment = config["deployments"]["qwen36_h100_8"]
        revision = deployment["model_revision"]
        claim = {
            "model": deployment["model"], "model_revision": revision,
            "server_pid": 1234, "runtime": "vllm", "vllm_version": "0.28.0",
            "runtime_artifact_sha256": "A" * 64, "host": "127.0.0.1", "port": 8000,
            "tensor_parallel_size": 8, "max_model_len": 131072, "max_num_seqs": 20,
            "api_key": "fabricated-secret-not-for-provenance",
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "server.json"
            path.write_text(json.dumps(claim))
            safe = validate_server_manifest(path, deployment=deployment, revision=revision,
                                            protocol=config["protocol"], port=8000)
            self.assertNotIn("api_key", safe)
            self.assertEqual(safe["runtime_artifact_sha256"], "a" * 64)
            for key, value, message in (
                ("vllm_version", LOCKED_VLLM_VERSION, "newer than 0.11.0"),
                ("vllm_version", "0.11.0+cuda", "newer than 0.11.0"),
                ("vllm_version", "0.10.2", "newer than 0.11.0"),
                ("runtime_artifact_sha256", "mutable", "SHA-256"),
                ("max_model_len", 4096, "max_model_len"),
                ("tensor_parallel_size", 1, "tensor_parallel_size"),
                ("port", 8001, "port"),
            ):
                with self.subTest(key=key, value=value):
                    path.write_text(json.dumps({**claim, key: value}))
                    with self.assertRaisesRegex(ValueError, message):
                        validate_server_manifest(path, deployment=deployment, revision=revision,
                                                 protocol=config["protocol"], port=8000)

    def test_batch_verification_rejects_infra_and_missing_energy_after_ipw_success(self) -> None:
        scheduled = occupation_batches(fabricated_tasks())[0]
        protocol = json.loads(CONFIG.read_text())["protocol"]
        model = "fabricated/model"
        with tempfile.TemporaryDirectory() as temp:
            ipw_dir = Path(temp)
            records = [
                {"query_id": f"q{index:04d}", "stable_ids": {
                    "task_id": item["task_id"], "instance_id": item["instance_id"]}}
                for index, item in enumerate(scheduled)
            ]
            subset_hash = hashlib.sha256(
                json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            (ipw_dir / "subset_manifest.json").write_text(json.dumps({
                "schema_version": 1, "dataset": protocol["dataset"], "agent": protocol["agent"],
                "model": model, "subset_size": len(records), "subset_hash": subset_hash,
                "records": records,
            }))
            ids = [record["query_id"] for record in records]
            batch = {"batch_id": "batch_0000", "telemetry_accounting": "batch",
                     "batch_size": 20, "query_ids": ids, "batch_prompt_start_time_s": 1.0,
                     "batch_prompt_end_time_s": 3.0, "telemetry_sample_count": 2,
                     "prompt_finish_offsets_s": {query_id: 1.9 for query_id in ids},
                     "energy_window_coverage": {part: True for part in ("gpu", "cpu", "dram")},
                     **{f"batch_{part}_energy_joules": 2.0 for part in ("gpu", "cpu", "dram")}}
            (ipw_dir / "batches.jsonl").write_text(json.dumps(batch) + "\n")
            traces = [{"query_id": query_id, "batch_id": "batch_0000", "completed": True,
                       "prompt_start_time_s": 1.1, "prompt_end_time_s": 2.9,
                       "batch_prompt_finish_offset_s": 1.9,
                       "tool_call_count": 1,
                       "is_resolved": False, "score_metadata": {"reason": "empty_response"}}
                      for query_id in ids]
            (ipw_dir / "traces.jsonl").write_text("".join(json.dumps(row) + "\n" for row in traces))
            raw = ipw_dir / "telemetry_samples.jsonl"
            raw.write_text("".join(json.dumps({"schema_version": 1, "sample_time_s": t}) + "\n"
                                   for t in (0.9, 3.1)))
            manifest = {"schema_version": 1, "path": raw.name, "sample_count": 2,
                        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}
            (ipw_dir / "telemetry_samples_manifest.json").write_text(json.dumps(manifest))
            config = {"agent": protocol["agent"], "model": model, "dataset": protocol["dataset"],
                      "max_queries": 20, "concurrency": 20, "telemetry_accounting": "batch",
                      "telemetry_backend": "energy-monitor", "telemetry_interval": 0.2,
                      "eval_client": "gemini", "eval_model": protocol["operational_judge_model"],
                      "telemetry_samples": manifest}
            summary = {"config": config, "totals": {"queries": 20, "completed": 20},
                       "batch_telemetry": {"batch_count": 1, "query_count": 20, "batches": [batch],
                                           **{f"total_batch_{part}_energy_joules": 2.0
                                              for part in ("gpu", "cpu", "dram")}}}
            (ipw_dir / "summary.json").write_text(json.dumps(summary))
            evidence = verify_batch_artifacts(ipw_dir, scheduled, model=model, protocol=protocol)
            self.assertEqual(evidence["query_count"], 20)
            self.assertEqual(len(evidence["artifact_sha256"]), 6)
            traces[0]["tool_call_count"] = 0
            (ipw_dir / "traces.jsonl").write_text("".join(json.dumps(row) + "\n" for row in traces))
            with self.assertRaisesRegex(ValueError, "no OpenHands tool calls"):
                verify_batch_artifacts(ipw_dir, scheduled, model=model, protocol=protocol)
            traces[0]["tool_call_count"] = 1
            (ipw_dir / "traces.jsonl").write_text("".join(json.dumps(row) + "\n" for row in traces))
            batch["telemetry_sample_count"] = 1
            (ipw_dir / "batches.jsonl").write_text(json.dumps(batch) + "\n")
            with self.assertRaisesRegex(ValueError, "fewer than two batch telemetry samples"):
                verify_batch_artifacts(ipw_dir, scheduled, model=model, protocol=protocol)
            batch["telemetry_sample_count"] = 2
            batch["prompt_finish_offsets_s"].pop(ids[0])
            (ipw_dir / "batches.jsonl").write_text(json.dumps(batch) + "\n")
            with self.assertRaisesRegex(ValueError, "finish-offset roster"):
                verify_batch_artifacts(ipw_dir, scheduled, model=model, protocol=protocol)
            batch["prompt_finish_offsets_s"][ids[0]] = 1.9
            (ipw_dir / "batches.jsonl").write_text(json.dumps(batch) + "\n")
            traces[0]["batch_prompt_finish_offset_s"] = 0.5
            (ipw_dir / "traces.jsonl").write_text("".join(json.dumps(row) + "\n" for row in traces))
            with self.assertRaisesRegex(ValueError, "request finish offset differs from trace"):
                verify_batch_artifacts(ipw_dir, scheduled, model=model, protocol=protocol)
            traces[0]["batch_prompt_finish_offset_s"] = 1.9
            traces[0]["score_metadata"] = {"match_type": "rubric_llm_judge", "per_criterion": [
                {"satisfied": False, "error": "fabricated judge timeout"}
            ]}
            (ipw_dir / "traces.jsonl").write_text("".join(json.dumps(row) + "\n" for row in traces))
            with self.assertRaisesRegex(ValueError, "deliverable evaluation failed"):
                verify_batch_artifacts(ipw_dir, scheduled, model=model, protocol=protocol)
            traces[0]["score_metadata"] = {"reason": "empty_response"}
            traces[0]["is_resolved"] = None
            (ipw_dir / "traces.jsonl").write_text("".join(json.dumps(row) + "\n" for row in traces))
            with self.assertRaisesRegex(ValueError, "lacks a judgment"):
                verify_batch_artifacts(ipw_dir, scheduled, model=model, protocol=protocol)
            traces[0]["is_resolved"] = False
            traces[0]["completed"] = False
            for failure_type, reason in (
                ("infra_reference_error", "infra_reference_error"),
                ("infra_connection_error", "agent_run_failed"),
                ("infra_harness_error", "agent_harness_error"),
                ("infra_timeout", "agent_or_llm_timeout"),
            ):
                with self.subTest(failure_type=failure_type):
                    traces[0]["score_metadata"] = {
                        "failure_type": failure_type, "reason": reason,
                    }
                    traces[0]["unscorable_reason"] = reason
                    (ipw_dir / "traces.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in traces)
                    )
                    with self.assertRaisesRegex(ValueError, "input or judging infrastructure failed"):
                        verify_batch_artifacts(ipw_dir, scheduled, model=model, protocol=protocol)
            traces[0]["score_metadata"] = {"failure_type": "agent_error", "reason": "agent_error"}
            traces[0]["unscorable_reason"] = "agent_error"
            (ipw_dir / "traces.jsonl").write_text("".join(json.dumps(row) + "\n" for row in traces))
            summary["totals"]["completed"] = 19
            (ipw_dir / "summary.json").write_text(json.dumps(summary))
            self.assertEqual(
                verify_batch_artifacts(ipw_dir, scheduled, model=model, protocol=protocol)["completed_queries"],
                19,
            )
            summary["totals"]["completed"] = 20
            (ipw_dir / "summary.json").write_text(json.dumps(summary))
            traces[0]["completed"] = True
            traces[0].pop("unscorable_reason", None)
            traces[0]["score_metadata"] = {
                "failure_type": "harness_validation", "reason": "deliverable_contract_invalid",
            }
            (ipw_dir / "traces.jsonl").write_text("".join(json.dumps(row) + "\n" for row in traces))
            verify_batch_artifacts(ipw_dir, scheduled, model=model, protocol=protocol)
            traces[0]["score_metadata"] = {
                "failure_type": "harness_validation", "reason": "unexpected_harness_error",
            }
            (ipw_dir / "traces.jsonl").write_text("".join(json.dumps(row) + "\n" for row in traces))
            with self.assertRaisesRegex(ValueError, "input or judging infrastructure failed"):
                verify_batch_artifacts(ipw_dir, scheduled, model=model, protocol=protocol)
            traces[0]["score_metadata"] = {"reason": "empty_response"}
            (ipw_dir / "traces.jsonl").write_text("".join(json.dumps(row) + "\n" for row in traces))
            batch["batch_dram_energy_joules"] = None
            (ipw_dir / "batches.jsonl").write_text(json.dumps(batch) + "\n")
            with self.assertRaisesRegex(ValueError, "incomplete measured dram energy"):
                verify_batch_artifacts(ipw_dir, scheduled, model=model, protocol=protocol)

    def test_fabricated_launcher_smoke_does_not_write_outputs(self) -> None:
        fixture = Path(__file__).resolve().parent / "fixtures" / "fabricated_tasks.json"
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "run_deployment.py"),
             "gptoss20_h100_8", "--tasks-file", str(fixture),
             "--model-revision", "0" * 40],
            text=True, capture_output=True, check=True,
        )
        self.assertIn("44 occupation batches, 880 requests", process.stdout)
        self.assertEqual(process.stdout.count("--dataset-kwargs"), 44)

    def test_existing_server_command_preview(self) -> None:
        fixture = Path(__file__).resolve().parent / "fixtures" / "fabricated_tasks.json"
        command = [sys.executable, str(Path(__file__).resolve().parent / "run_deployment.py"),
                   "qwen36_h100_8", "--tasks-file", str(fixture)]
        rejected = subprocess.run(command, text=True, capture_output=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("--existing-server", rejected.stderr)
        accepted = subprocess.run(command + ["--existing-server", "--server-manifest",
                                             "/private/server.json"],
                                  text=True, capture_output=True, check=True)
        self.assertIn("44 occupation batches, 880 requests", accepted.stdout)
        self.assertNotIn("vllm.entrypoints", accepted.stdout)

    def test_all_paper_deployments_share_the_recorded_protocol(self) -> None:
        data = json.loads(CONFIG.read_text())
        self.assertEqual(len(data["deployments"]), 9)
        batches = occupation_batches(fabricated_tasks())
        self.assertEqual(len(batches), 44)
        self.assertTrue(all(len(batch) == 20 for batch in batches))
        self.assertEqual(LOCKED_VLLM_AUTOSTART_MODELS,
                         {"openai/gpt-oss-20b", "openai/gpt-oss-120b"})
        for profile, deployment in data["deployments"].items():
            existing_server = deployment["model"] not in LOCKED_VLLM_AUTOSTART_MODELS
            server, runs = commands(profile, Path("runs/test"), batches, port=8000,
                                    revision="fabricated-revision", vllm_args=[],
                                    existing_server=existing_server)
            self.assertEqual(len(runs), 44)
            if existing_server:
                self.assertIsNone(server)
                with self.assertRaisesRegex(ValueError, "pass --existing-server and --server-manifest"):
                    commands(profile, Path("runs/test"), batches, port=8000,
                             revision="fabricated-revision", vllm_args=[], existing_server=False)
            else:
                self.assertEqual(server[server.index("--tensor-parallel-size") + 1],
                                 str(deployment["gpu_count"]))
                self.assertEqual(server[server.index("--host") + 1], "127.0.0.1")
                self.assertIn("131072", server)
                self.assertEqual(server[server.index("--revision") + 1], "fabricated-revision")
            for run in runs:
                self.assertEqual(run[run.index("--concurrency") + 1], "20")
                self.assertEqual(run[run.index("--max-queries") + 1], "20")
                self.assertEqual(run[run.index("--telemetry-accounting") + 1], "batch")
                self.assertEqual(run[run.index("--telemetry-interval") + 1], "0.2")
                judge = deployment.get("operational_judge", {
                    "client": "gemini", "model": data["protocol"]["operational_judge_model"],
                })
                self.assertEqual(run[run.index("--eval-client") + 1], judge["client"])
                self.assertEqual(run[run.index("--eval-model") + 1], judge["model"])
                agent_kwargs = json.loads(run[run.index("--agent-kwargs") + 1])
                self.assertEqual(agent_kwargs["max_turns"], deployment.get("agent_max_turns", 150))
                if profile == "glm52_h200_8":
                    self.assertEqual(run[run.index("--query-timeout") + 1], "14400")
                    self.assertEqual(run[run.index("--telemetry-buffer-seconds") + 1], "14700")
                    environment = controlled_run_environment(
                        {}, protocol=data["protocol"], monitor_port=50111,
                        gpu_count=8, accelerator="H200", deployment=deployment,
                    )
                    self.assertEqual(environment["IPW_OPENHANDS_FINAL_NUDGE_MODE"], "skip")
                    self.assertEqual(environment["IPW_GDPVAL_REPAIR_ATTEMPTS"], "12")
                    self.assertEqual(environment["IPW_GDPVAL_FINAL_CHECK_ATTEMPTS"], "5")
                    self.assertEqual(environment["IPW_GDPVAL_FINAL_CHECK_MAX_TURNS"], "80")
                    self.assertEqual(environment["IPW_GDPVAL_FILE_SETTLE_SECONDS"], "120")
                    self.assertEqual(environment["IPW_OPENHANDS_TEXT_TOOL_CALL_PARSER"], "glm")
                self.assertEqual("--require-dedicated-hardware" in run,
                                 deployment["accelerator"] != "MI350X")
                dataset_kwargs = json.loads(run[run.index("--dataset-kwargs") + 1])
                self.assertFalse(dataset_kwargs["include_rubric_in_prompt"])
                self.assertEqual(dataset_kwargs["revision"], data["protocol"]["dataset_revision"])
                self.assertTrue(dataset_kwargs["task_instances_file"].endswith(".json"))
            self.assertEqual(len({run[run.index("--output-dir") + 1] for run in runs}), 44)
            source_config = energy_source_config(
                profile, Path(__file__).resolve().parents[1] / "runs" / profile / "fabricated-run", batches
            )
            self.assertEqual(len(source_config["profiles"][0]["sources"]), 44)
            self.assertEqual(source_config["profiles"][0]["profile"], deployment["analysis_profile"])
            self.assertEqual(source_config["profiles"][0]["source_mode"], "ipw-run")
            self.assertTrue(source_config["profiles"][0]["run_provenance"].endswith("provenance.json"))
            self.assertTrue(source_config["profiles"][0]["sources"][0]["subset_manifest"].endswith(".json"))
            self.assertTrue(source_config["profiles"][0]["sources"][0]["raw_meter"].endswith("telemetry_samples.jsonl"))

    def test_glm_external_server_matches_archived_sequence_limit(self) -> None:
        data = json.loads(CONFIG.read_text())
        deployment = data["deployments"]["glm52_h200_8"]
        claim = {
            "model": deployment["model"], "model_revision": "a" * 40,
            "server_pid": 1234, "runtime": "vllm", "vllm_version": "0.24.0",
            "runtime_artifact_sha256": "b" * 64, "host": "127.0.0.1", "port": 8000,
            "tensor_parallel_size": 8, "max_model_len": 131072, "max_num_seqs": 2,
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "server.json"
            path.write_text(json.dumps(claim))
            validate_server_manifest(path, deployment=deployment, revision="a" * 40,
                                     protocol=data["protocol"], port=8000)
            claim["max_num_seqs"] = 20
            path.write_text(json.dumps(claim))
            with self.assertRaisesRegex(ValueError, "max_num_seqs"):
                validate_server_manifest(path, deployment=deployment, revision="a" * 40,
                                         protocol=data["protocol"], port=8000)

    def test_inventory_and_schedule_reject_duplicates_and_wrong_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fabricated_tasks.json"
            tasks = fabricated_tasks()
            path.write_text(json.dumps(tasks))
            self.assertEqual(load_task_inventory(path), tasks)
            with self.assertRaisesRegex(ValueError, "agent-instance roster hash"):
                validate_recorded_task_hash(path)
            schedule = occupation_batches(tasks)
            self.assertEqual(len({item["instance_id"] for batch in schedule for item in batch}), 880)
            roster_path = Path(temp) / "fabricated_instances.json"
            roster_path.write_text(json.dumps([item for batch in schedule for item in batch]))
            self.assertEqual(load_task_instances(roster_path), schedule)
            invalid = [item.copy() for batch in schedule for item in batch]
            invalid[0]["occupation_batch_id"] = "occ_b01"
            roster_path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "order or repeat metadata"):
                load_task_instances(roster_path)
            for batch in schedule:
                self.assertEqual(len({item["occupation"] for item in batch}), 1)
                self.assertEqual(len({item["task_id"] for item in batch}), 5)
            tasks[-1]["task_id"] = tasks[0]["task_id"]
            path.write_text(json.dumps(tasks))
            with self.assertRaisesRegex(ValueError, "duplicate task_id"):
                load_task_inventory(path)

    def test_manifest_verifies_only_external_synthetic_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "input.csv").write_text("task_id,value\nsynthetic,1\n")
            sha = hashlib.sha256((root / "input.csv").read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"inputs": [{"path": "input.csv", "sha256": sha}]}))
            self.assertEqual(verify(root, manifest), [])
            (root / "input.csv").write_text("task_id,value\nsynthetic,2\n")
            self.assertEqual(verify(root, manifest), ["checksum mismatch: input.csv"])
            manifest.write_text(json.dumps({"inputs": [{"path": "../outside", "sha256": None}]}))
            self.assertEqual(verify(root, manifest), ["unsafe manifest path: ../outside"])

    def test_bundled_public_input_is_verified_without_external_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "checkout"
            external = Path(temp) / "recorded"
            root.mkdir()
            external.mkdir()
            bundled = root / "public.csv"
            private = external / "private.csv"
            bundled.write_text("kind,value\nsource,fabricated\n")
            private.write_text("kind,value\nexternal,fabricated\n")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"inputs": [
                {"path": "public.csv", "sha256": hashlib.sha256(bundled.read_bytes()).hexdigest(),
                 "bundled": True},
                {"path": "private.csv", "target_path": "staged/private.csv",
                 "sha256": hashlib.sha256(private.read_bytes()).hexdigest()},
            ]}))
            with (mock.patch.object(verify_inputs, "ROOT", root),
                  mock.patch.object(stage_inputs, "ROOT", root),
                  mock.patch.object(sys, "argv", ["stage_inputs.py", "--data-root", str(external),
                                                 "--manifest", str(manifest)])):
                self.assertEqual(verify(external, manifest), [])
                stage_inputs.main()
                self.assertFalse(bundled.is_symlink())
                self.assertTrue((root / "staged/private.csv").is_symlink())
                bundled.write_text("kind,value\nsource,tampered\n")
                self.assertEqual(verify(external, manifest), ["checksum mismatch: public.csv"])

    def test_manifest_hashes_fabricated_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "fabricated"
            source.mkdir()
            (source / "one.txt").write_text("one")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"inputs": [{"path": "fabricated", "kind": "directory",
                                                        "sha256": directory_digest(source)}]}))
            self.assertEqual(verify(root, manifest), [])
            (source / "one.txt").write_text("two")
            self.assertEqual(verify(root, manifest), ["checksum mismatch: fabricated"])
            (source / "one.txt").unlink()
            (source / "one.txt").symlink_to(root / "outside.txt")
            self.assertIn("symlink in directory", verify(root, manifest)[0])
            source.rename(root / "real")
            (root / "fabricated").symlink_to(root / "real", target_is_directory=True)
            self.assertEqual(verify(root, manifest), ["symlink in input path: fabricated"])


if __name__ == "__main__":
    unittest.main()
