# Calculate output dollar value, NVJ, and TSJ

Combine expert completion times, occupation median hourly wages, judged outcomes, measured energy, and AI-system execution costs. Run the commands below from the repository root after [installing the dependencies](../reproduction/README.md#install).

## NVJ and TSJ

| Term | Definition |
| --- | --- |
| **Expert-only reference** | An expert deliverable produced without AI assistance, paired with its recorded expert completion time. |
| **Expert-with-AI reference** | An expert deliverable produced using AI, paired with its recorded expert completion time. |
| **Output dollar value** | Recorded expert completion time × occupation median hourly wage. |
| **Net dollar value** | Credited output dollar value − AI-system execution cost. |
| **Time saving** | Recorded expert completion time − AI-system completion time, credited to a win or tie. |
| **Net Value per Joule (NVJ)** | Net dollar value ÷ measured energy. |
| **Time Saving per Joule (TSJ)** | Time saving ÷ measured energy. |

For each selected run, let `H` be recorded expert completion time in hours, `L` be AI-system completion time in hours, `W` be the occupation median hourly wage, `C` be AI-system execution cost in USD, and `E` be measured energy in MJ. Let `Y = 1` for a win or tie against the selected expert reference and `Y = 0` for a loss.

```text
Output dollar value G = H × W
Credited output dollar value = Y × G
Net dollar value = Y × G − C
Time saving = Y × (H − L)
```

Every selected run contributes cost and energy, including losses and timeouts. Retain negative time saving when a qualifying run takes longer than the expert reference. AI-system completion time covers the run's elapsed execution, including inference, tool use, and waits. AI-system execution cost values the recorded resource allocation at the selected GPU, CPU, and RAM rates, plus applicable storage and transfer charges.

For an aggregate, apply the same task weights `p` to each component before forming either ratio:

```text
NVJ = sum(p × (Y × G − C)) / sum(p × E)
TSJ = sum(p × Y × (H − L)) / sum(p × E)
```

GDP-based occupation weights combine sector GDP shares with within-sector employment shares. Occupation means use the selected task count; category weights are normalized within each category. Wages enter output dollar value. The commands report NVJ in **USD/MJ** and TSJ in **hours/MJ**; one megajoule is one million joules.

For repeated-attempt policies, credit output dollar value once per qualifying task and use mean AI-system completion time among its wins or ties. Cost and energy include every selected attempt.

## Prepare inputs

1. Use the bundled [BLS wages and O*NET® metadata](../occupation_data/README.md).
2. Supply recorded expert completion times, task/reference mappings, judgments, measurements, and price snapshots listed in the [input manifest](../reproduction/input_manifest.json).
3. [Verify and stage the external inputs](../reproduction/README.md#verify-external-inputs) with their SHA-256 hashes.

## Complete energy inputs

Total energy is **GPU + CPU package + DRAM** over each complete batch window. Energy is assigned to tasks using every request's start and finish times, including retries. Validation checks task IDs, repeats, batch membership, source hashes, and all three components.

Provide the following CSVs under ignored `reproduction/inputs/`:

| Input | Required fields |
| --- | --- |
| `energy_component_batches.csv` | `profile`, `batch_key`, `batch_role` (`primary` or `repair`), `batch_start_s`, `batch_end_s`, positive `gpu_energy_j`, `cpu_energy_j`, `dram_energy_j`, `source_path`, `source_sha256` |
| `energy_component_requests.csv` | `profile`, `batch_key`, `physical_request_id`, `task_id`, `repeat`, `start_s`, `end_s` |

For the supplied experiment, the 880-instance roster validates 44 primary occupation batches, each with five tasks and four repeats. Repair batches may add physical requests. Referenced measurement paths must be repository-relative staged files with matching SHA-256 hashes. Add these files and their measurement sources to a completed external manifest before [verification and staging](../reproduction/README.md#verify-external-inputs).

For completed IPW runs, combine the profile entries from each run's `energy_sources.json` into a private config, then convert the measurements:

```bash
uv run --project intelligence-per-watt --no-sync python analysis/build_energy_components_from_ipw.py \
  --config /path/to/ipw-energy-config.json \
  --output-dir /path/to/inputs/reproduction/inputs
```

The `ipw-run` mode checks the completed run record, model and deployment identity, input and lockfile hashes, batch size, and raw sample manifests. It uses IPW's batch energy totals and verifies their source records. See [the adapter's configuration format](build_energy_components_from_ipw.py).

## Run calculations

List the steps and preview their commands:

```bash
uv run --project intelligence-per-watt --no-sync python reproduction/build_metrics.py --list
uv run --project intelligence-per-watt --no-sync python reproduction/build_metrics.py --dry-run
```

Run the pipeline after staging the required inputs:

```bash
uv run --project intelligence-per-watt --no-sync python reproduction/build_metrics.py
```

Use `--from-step` and `--through-step` to run a range of steps. Generated CSV and JSON files go into ignored `outputs/metrics/` and related output directories. The default energy boundary is GPU + CPU + DRAM. Each source adapter expects the layouts, mappings, and price sources listed in the input manifest.

Run the Joules to GDP calculation smoke test with fabricated data. It checks batch energy accounting, task value, execution cost, NVJ, and TSJ:

```bash
uv run --project intelligence-per-watt --no-sync python reproduction/smoke_test.py
```

## Code map

| Calculation | Code |
| --- | --- |
| Output dollar values and weighted summaries | [metrics.py](metrics.py) |
| First-attempt selection and repeat analysis | [build_pass1_primary_results.py](build_pass1_primary_results.py) |
| Employment and occupation weights | [build_gdp_employment_weights.py](build_gdp_employment_weights.py) |
| Energy per request | [active_request_energy.py](active_request_energy.py), [energy_boundary.py](energy_boundary.py) |
| IPW measurement conversion | [build_energy_components_from_ipw.py](build_energy_components_from_ipw.py) |
| AI-system execution cost, NVJ, and TSJ | [rental_cost.py](rental_cost.py), [build_net_metrics.py](build_net_metrics.py) |
| Sensitivity and judge checks | [category_omission_robustness.py](category_omission_robustness.py), [decompose_reference_condition.py](decompose_reference_condition.py), [compare_glm_judges.py](compare_glm_judges.py) |
