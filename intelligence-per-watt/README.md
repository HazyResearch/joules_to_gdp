# Task execution

The [Intelligence Per Watt (IPW)](https://github.com/HazyResearch/intelligence-per-watt) harness runs tasks with OpenHands, scores the resulting work, and saves GPU, CPU, and memory measurements. Start with the [run guide](../reproduction/README.md) for the supplied GDPVal example.

## Install

Use Python 3.13 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --project intelligence-per-watt --locked --extra openhands --extra deliverables --extra pairwise --extra analysis
python3 scripts/build_energy_monitor.py
```

Install and build on the measurement host. See the [run prerequisites](../reproduction/README.md#install) and separate [vLLM guide](../reproduction/vllm/README.md) before collecting measurements.

| Extra | Enables |
| --- | --- |
| `openhands` | Agent SDK and tools |
| `deliverables` | Reading and scoring task files |
| `pairwise` | Judge provider SDKs and file readers |
| `analysis` | Numerical calculations |
| `dev` | Tests and code checks |

## Run a task selection

For the complete supplied schedule and automatic input checks, use the [run guide](../reproduction/README.md#run-with-measurement). To choose tasks directly, use `ipw run` with an external task-selection file:

```bash
uv run --project intelligence-per-watt --no-sync ipw run \
  --agent openhands --model Qwen/Qwen3.6-27B --dataset gdpval \
  --client-base-url http://127.0.0.1:8000 \
  --dataset-kwargs '{"revision":"11e7900cdcac61bc4daf59e65feb238acda98fbf","task_instances_file":"/private/inputs/task-selection.json"}' \
  --telemetry-backend energy-monitor --telemetry-interval 0.2 \
  --eval-client gemini --eval-model gemini-3.5-flash \
  --output-dir runs/task-selection
```

Replace the model, endpoint, and task-selection path with your inputs. The selection file is a JSON list using IDs from the task dataset:

```json
[{"task_id": "YOUR_TASK_ID", "instance_id": "YOUR_TASK_ID::attempt01"}]
```

Use a unique `instance_id` for each repeated attempt. Provide task reference files and the judge's API key through the host environment. See [GDPVal's input options](src/ipw/datasets/gdpval.py) for more options.

Useful options:

| Option | Purpose |
| --- | --- |
| `--max-queries` | Limit the number of selected tasks |
| `--concurrency` | Choose how many tasks run at once |
| `--telemetry-accounting batch` | Account for energy once across concurrent tasks |
| `--agent-kwargs` | Set agent options as JSON |
| `--require-dedicated-hardware` | Check exclusive NVIDIA GPU use before starting |

Use batch accounting when running concurrent tasks. `--telemetry-interval` is in seconds: `0.2` requests a 200 ms interval. `IPW_ENERGY_MONITOR_INTERVAL_MS` overrides the meter's launch interval; an existing meter uses its configured interval.

Adding a rubric to the agent prompt is optional and disabled by default. To enable it, add `"include_rubric_in_prompt":true` to `--dataset-kwargs`.

Deliverable judges read `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY` / `GOOGLE_API_KEY`, according to the chosen provider. `IPW_EVAL_API_KEY` overrides the judge key. [Blinded A/B comparison](../cluster/README.md) is a separate step.

For all options:

```bash
uv run --project intelligence-per-watt --no-sync ipw run --help
```

## Outputs

Each run creates a directory under `--output-dir` with:

- `traces.jsonl` — task outcomes, timings, and scores.
- `batches.jsonl` — batch boundaries and energy calculations.
- `telemetry_samples.jsonl` — timestamped GPU, CPU, and DRAM power, cumulative energy, and memory use.
- `telemetry_samples_manifest.json` — raw measurement checksum and schema.
- `summary.json` and `subset_manifest.json` — run settings and selected tasks.

Keep generated files under ignored `runs/`. Continue with [judging](../cluster/README.md) and [NVJ / TSJ calculations](../analysis/README.md).

## Tests

Install the `dev` extra, then run the offline harness tests:

```bash
uv run --project intelligence-per-watt --no-sync pytest intelligence-per-watt/src/ipw/tests -q -m "not integration and not stress"
```

Hardware integration tests run on the measurement host.
