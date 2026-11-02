# Serve a model with vLLM

Run the model server on the same host as the agent and energy meter. Use the [run guide](../README.md) to select tasks, provide judge credentials, and collect measurements.

## Choose the server runtime

| Profiles | Setup |
| --- | --- |
| GPT-OSS | The launcher can start the locked vLLM `0.11.0` runtime. Install the `vllm` extra below. |
| Qwen, Gemma, GLM | Start a compatible vLLM version newer than `0.11.0` in its own environment or container, then provide a server manifest. |

To install the locked runtime for launcher-managed GPT-OSS serving:

```bash
uv sync --project intelligence-per-watt --locked --extra openhands --extra deliverables --extra pairwise --extra analysis --extra vllm
```

Record the runtime wheel or image SHA-256, model and tokenizer revisions, reasoning/tool parsers, quantization, GPU architecture, and driver configuration. Use a pinned 40-character checkpoint commit. Profiles with a revision in [deployments.json](../deployments.json) require that exact revision.

## Configure an existing server

Start the server with the model ID and GPU count from the selected profile. Bind it to `127.0.0.1` and use the settings below. Save its identity in a private JSON file such as `/private/server.json`:

| Field | Value |
| --- | --- |
| `model` | Exact model ID from the profile |
| `model_revision` | Pinned 40-character checkpoint commit |
| `server_pid` | PID of the local server process |
| `runtime` | `vllm` |
| `vllm_version` | Installed server version |
| `runtime_artifact_sha256` | 64-character SHA-256 of its wheel or image |
| `host`, `port` | `127.0.0.1` and the selected port, normally `8000` |
| `tensor_parallel_size` | Profile's tensor-parallel GPU count |
| `max_model_len` | `131072` |
| `max_num_seqs` | `2` for GLM; `20` for the other supplied profiles |

Preview the run using both server flags:

```bash
uv run --project intelligence-per-watt --no-sync python reproduction/run_deployment.py \
  qwen36_h100_8 --tasks-file /private/inputs/cluster/gdpval_occupation_repeat4_instances.json \
  --existing-server --server-manifest /private/server.json
```

For profiles without a configured model revision, also pass `--model-revision` with the checkpoint SHA. Add `--execute` after completing the [run checks](../README.md#run-with-measurement).

At execution, the launcher validates every manifest field, checks that the server PID or its child owns the listening port, and checks the model ID at `/v1/models`. Keep the server command, checkpoint revision, artifact digest, and settings with the run records. Store credentials in the host environment or secret store.

## Choose GPUs

Keep the measurement host dedicated. If a profile uses a subset of NVIDIA GPUs, get their full physical UUIDs with:

```bash
nvidia-smi -L
```

Set `CUDA_VISIBLE_DEVICES` to the comma-separated full UUIDs before starting **both** server and launcher. The launcher checks that both use the same selection. Use full UUIDs for subsets rather than numeric device indices.

## Server lifecycle

An existing server stays under your control. A server started by the launcher stops when the launcher exits.

Extra `--vllm-arg` flags apply when the launcher starts the server. Core model, protocol, context, concurrency, and credential settings are protected against overrides. The supplied schedule sends 20 requests per batch; the server queues work according to `max_num_seqs`.

Start with the [offline preview](../README.md#preview-a-schedule), then [check the energy components](../../energy-monitor/README.md#check-component-availability) before measuring.
