# Run guide

Set up a measured run, check its inputs, and calculate NVJ and TSJ. The supplied [deployment profiles](deployments.json) configure a GDPVal example; use the [task execution guide](../intelligence-per-watt/README.md#run-a-task-selection) for a smaller task selection.

## Install

Use a dedicated Linux accelerator host with Python 3.13, [uv](https://docs.astral.sh/uv/), Rust/Cargo, and the matching GPU driver. Install LibreOffice, Poppler, FFmpeg, and Tesseract for task files and judging. OpenHands executes locally: use a dedicated account and an isolated working directory.

Run all commands from the repository root:

```bash
uv sync --project intelligence-per-watt --locked --extra openhands --extra deliverables --extra pairwise --extra analysis
python3 scripts/build_energy_monitor.py
```

Build the meter on the measurement host. For MI350X, install ROCm SMI libraries and headers, then build with `python3 scripts/build_energy_monitor.py --features amd`. Arrange exclusive host access with the operator; NVIDIA profiles also run an automatic dedicated-hardware check.

## Preview a schedule

This offline preview uses fabricated task IDs. It prints commands without starting a model, collecting energy, or writing run files:

```bash
uv run --project intelligence-per-watt --no-sync python reproduction/run_deployment.py \
  gptoss20_h100_8 --tasks-file reproduction/fixtures/fabricated_tasks.json \
  --model-revision 0000000000000000000000000000000000000000
```

The zero revision is a preview placeholder. The supplied example schedules 44 occupation batches, each with five tasks repeated four times, for 880 requests. Measured execution uses the separately supplied agent-instance roster listed in the [input manifest](input_manifest.json).

## Run with measurement

1. Choose a profile from [deployments.json](deployments.json).
2. Configure a local model server using the separate [vLLM guide](vllm/README.md). Pin its checkpoint to a 40-character commit SHA.
3. Provide model access through Hugging Face authentication. Set judge credentials in the host environment or secret store: `OPENAI_API_KEY` for GLM, or `GEMINI_API_KEY` / `GOOGLE_API_KEY` for the other profiles. `IPW_EVAL_API_KEY` overrides the judge key.
4. Make the task roster and reference files available, and [check the energy components](../energy-monitor/README.md#check-component-availability).
5. Preview the command below, then add `--execute` to collect measurements.

```bash
uv run --project intelligence-per-watt --no-sync python reproduction/run_deployment.py \
  qwen36_h100_8 --tasks-file /private/inputs/cluster/gdpval_occupation_repeat4_instances.json \
  --existing-server --server-manifest /private/server.json
```

Use a committed, clean checkout and dedicated accelerators. The launcher verifies the roster checksum, model/server identity, GPU selection, task references, judge responses, and GPU + CPU + DRAM measurements. It applies the profile's settings after clearing inherited `IPW_*` settings, except `IPW_EVAL_API_KEY`.

Runs are saved under ignored `runs/<profile>/<run_id>/`:

| File or directory | Contents |
| --- | --- |
| `provenance.json` | Configuration, source revision, input hashes, and run status |
| `energy_sources.json` | Inputs for energy analysis, written after a complete run |
| `schedule/` | Task selections for each batch |
| `batches/` | Task outputs, scores, traces, raw measurements, and checksum manifests |

After a failed run, fix the cause and launch a new run. An existing server remains operator-managed; a server started by the launcher stops on exit.

Task scoring during execution and [blinded A/B judging](../cluster/README.md) are separate steps. Keep judge settings and timing with the run records; scoring that overlaps a measured batch contributes to host energy.

## Verify external inputs

The [input manifest](input_manifest.json) lists bundled BLS and O*NET® sources and separately supplied task inputs, time estimates, measurements, judgments, and price snapshots.

Keep your completed manifest outside the checkout. Fill its SHA-256 values, include the raw measurement files, and verify before staging:

```bash
uv run --project intelligence-per-watt --no-sync python reproduction/verify_inputs.py \
  --data-root /path/to/inputs --manifest /path/to/local-manifest.json
uv run --project intelligence-per-watt --no-sync python reproduction/stage_inputs.py \
  --data-root /path/to/inputs --manifest /path/to/local-manifest.json
```

Verification checks bundled files in the checkout and external files under `--data-root`. Directory hashes cover sorted relative paths, file sizes, and file hashes. Input symlinks are rejected. Staging checks the hashes again and links external inputs into ignored paths.

## Calculate metrics

Prepare the inputs in the [analysis guide](../analysis/README.md), then inspect and run the calculation steps:

```bash
uv run --project intelligence-per-watt --no-sync python reproduction/build_metrics.py --list
uv run --project intelligence-per-watt --no-sync python reproduction/build_metrics.py --dry-run
uv run --project intelligence-per-watt --no-sync python reproduction/build_metrics.py
```

Calculations use measured GPU + CPU + DRAM energy. Generated metrics go to ignored `outputs/metrics/` and related output directories. Use `--from-step` / `--through-step` to select steps, or `--include-audits` to add evaluation audits and contamination checks.

## Offline checks

The [Joules to GDP smoke test](smoke_test.py) checks batch GPU, CPU, and DRAM energy accounting, task value, execution cost, NVJ, and TSJ with fabricated inputs. It needs only the `analysis` extra and does not start a model or call a judge.

```bash
uv run --project intelligence-per-watt --no-sync python reproduction/smoke_test.py
```

Install test dependencies for the additional release checks:

```bash
uv sync --project intelligence-per-watt --locked --extra dev --extra openhands --extra deliverables --extra pairwise --extra analysis
uv run --project intelligence-per-watt --no-sync python reproduction/test_release.py
uv run --project intelligence-per-watt --no-sync python scripts/check_public_release.py
```
