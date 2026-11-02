# Blinded comparison

Compare AI-system deliverables with the selected expert reference as anonymous **A/B** pairs. The module prepares the files, hides which side produced them, asks a model judge to compare them, and counts wins, ties, and losses.

## Before you start

1. Install the `pairwise` and `deliverables` extras in the [setup guide](../reproduction/README.md#install).
2. Supply task instructions, expert-reference deliverables, AI-system outputs, and their task mappings. Use the adapter that matches their source layout.
3. Set the selected provider's key in your environment: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, or `ANTHROPIC_API_KEY`. Keep prepared evidence, hidden A/B assignments, responses, and summaries under ignored `outputs/` or `reports/`.

## Entry points

Run commands from the repository root. Start by viewing the available steps:

```bash
uv run --project intelligence-per-watt --no-sync python cluster/gdpval_pairwise_eval.py --help
uv run --project intelligence-per-watt --no-sync python cluster/gdpval_provider_judges.py --provider google --help
```

For the GLM source layout, set the run-volume, reference, and output paths before preparing evidence:

```bash
export GDPVAL_RUN_VOLUME=/path/to/run-volume
export GDPVAL_BASELINES=/path/to/reference-tasks
export GDPVAL_PAIRWISE_STATE=outputs/pairwise

uv run --project intelligence-per-watt --no-sync python cluster/gdpval_pairwise_eval.py manifest
uv run --project intelligence-per-watt --no-sync python cluster/gdpval_pairwise_eval.py prepare-evidence
uv run --project intelligence-per-watt --no-sync python cluster/gdpval_pairwise_eval.py audit-inputs
```

Review the preparation and input-audit outputs before submitting requests. If the evidence summary lists pending audio or video, run `parse-media`, then repeat `audit-inputs`. Use the same state directory for the remaining commands:

| Step | Subcommand for `gdpval_pairwise_eval.py` |
| --- | --- |
| Upload prepared evidence | `upload-evidence` |
| Build judge requests | `build-batches --model <judge-model>` |
| Submit a pilot | `submit --mode pilot` |
| Wait and read responses | `poll --wait`, then `parse-results` |
| Submit remaining comparisons | `submit --mode remaining` |
| Collect the remaining responses | `poll --wait`, then `parse-results` |
| Check and summarize judgments | `validate`, then `aggregate` |

For Google or Anthropic, use [gdpval_provider_judges.py](gdpval_provider_judges.py) with `--provider google` or `--provider anthropic`. Pass the prepared directory with `--source-state` and a separate output directory with `--state`; start with `init`. Its `--help` lists upload, pilot/full submission, polling, parsing, and aggregation commands.

A win or tie against the expert reference receives value and time-saving credit. The win–tie rate is the share of AI-system outputs judged a win or tie. Preserve the judge model, temperature, prompts, source hashes, and hidden A/B assignment with each comparison. The [task harness's quality scores](../intelligence-per-watt/README.md) and these pairwise judgments are separate evaluations.

## Code map

| Code | Purpose |
| --- | --- |
| [gdpval_pairwise_eval.py](gdpval_pairwise_eval.py) | Prepare and audit evidence; run OpenAI comparisons; summarize judgments |
| [gdpval_provider_judges.py](gdpval_provider_judges.py) | Run Google and Anthropic comparisons |
| Model-specific `*_pairwise_eval.py` adapters | Read Qwen, Gemma, GPT-OSS, and hardware-specific source layouts |
| `*_value_*` and `*_metrics.py` adapters | Join judgments with task values, energy, and hardware information |

Adapters expect their documented source inventories and mappings. The [input manifest](../reproduction/input_manifest.json) lists separately supplied inputs.

### GPT-OSS numerical inputs

[gptoss_category_value_energy.py](gptoss_category_value_energy.py) reads separately supplied batch records and telemetry audits from `reproduction/inputs/gptoss/batches/` and `reproduction/inputs/gptoss/telemetry/`. Keep the source layout below each root:

| Model | Below the batch root | Below the telemetry root |
| --- | --- | --- |
| GPT-OSS 120B | `gpt-oss-120b-high-no-rubric/run/batches.jsonl` | `gpt-oss-120b-high-no-rubric/reports/final/telemetry_audit.json` |
| GPT-OSS 20B | `gpt-oss-20b-high-no-rubric/run/batches.jsonl` | `gpt-oss-20b-high-no-rubric/reports/final_corrected/telemetry_audit.json` |

Use `--batch-input-root` and `--telemetry-input-root` when those inputs are stored elsewhere. For example:

```bash
uv run --project intelligence-per-watt --no-sync python cluster/gptoss_category_value_energy.py \
  --batch-input-root /path/to/batch-records \
  --telemetry-input-root /path/to/telemetry-audits \
  --output outputs/metrics/gptoss_value_energy.json
```
