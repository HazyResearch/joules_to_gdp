# Energy profiling

Build this Rust service to measure energy while the [IPW harness](../intelligence-per-watt/README.md) runs tasks. It sends readings to IPW over a local gRPC connection.

## What it measures

| Component | Collection path |
| --- | --- |
| GPU power and energy | NVIDIA NVML or optional AMD ROCm SMI |
| CPU package power and energy | Linux RAPL under `/sys/class/powercap/intel-rapl` |
| DRAM power and energy | Separate `dram` RAPL subdomains |
| GPU and system memory usage | Device and host telemetry, recorded in MB |

GPU, CPU, and DRAM energy are recorded separately. Memory usage is recorded in MB.

## Batch energy

Measure energy once over the full execution window of each batch, including concurrently running tasks. The [analysis module](../analysis/README.md#complete-energy-inputs) checks the component totals and assigns batch energy to tasks using their start and finish times. Task allocations add up to the measured batch energy.

## Build

Install Rust/Cargo and the [Python dependencies](../reproduction/README.md#install), then run from the repository root:

```bash
python3 scripts/build_energy_monitor.py
```

This runs `cargo build --locked --release` and copies the executable to `intelligence-per-watt/src/ipw/telemetry/bin/<platform>/`. The locked build includes `protoc`; compiled binaries stay ignored.

On a ROCm host, install the ROCm SMI library and headers, then enable the AMD collector:

```bash
python3 scripts/build_energy_monitor.py --features amd
```

The service samples every 50 ms by default. Use `--collection-interval-ms 200` for 200 ms sampling when starting it directly. When IPW starts the service, `--telemetry-interval 0.2` selects the same interval. An already-running service keeps its own interval.

## Check component availability

On x86_64 Linux, start the built meter in one terminal:

```bash
intelligence-per-watt/src/ipw/telemetry/bin/linux-x86_64/energy-monitor --collection-interval-ms 200
```

In another terminal, check that GPU, CPU, and DRAM energy are available:

```bash
uv run --project intelligence-per-watt --no-sync python -c 'from ipw.telemetry.launcher import wait_for_ready; raise SystemExit(0 if wait_for_ready(require_energy_components=True) else 1)'
```

The command exits successfully when all three components are available. Stop the manual meter with `Ctrl+C`, then follow [Run with measurement](../reproduction/README.md#run-with-measurement). The run launcher starts its own meter and checks every batch. Use a dedicated measurement host.

## Code map

| Code | Purpose |
| --- | --- |
| [linux_rapl.rs](src/collectors/linux_rapl.rs) | CPU package and DRAM energy collection |
| [collectors/](src/collectors/) | NVIDIA, optional AMD, and macOS collectors |
| [energy.proto](proto/energy.proto) | Measurement fields sent to Python |
| [collector.py](../intelligence-per-watt/src/ipw/telemetry/collector.py) | IPW's Python measurement reader |
| [telemetry_accounting.py](../intelligence-per-watt/src/ipw/execution/telemetry_accounting.py) | Energy accounting for batches and tasks |

Run Rust checks from the repository root:

```bash
cargo test --locked --manifest-path energy-monitor/Cargo.toml
```
