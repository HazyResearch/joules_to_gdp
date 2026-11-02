#!/usr/bin/env python3
"""Calculate machine-readable Joules to GDP metrics from supplied inputs.

The required task outputs, judgments, telemetry, and price snapshots are
described in ``reproduction/input_manifest.json`` and obtained separately.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE_STEPS = (
    ("energy_boundary", "analysis/energy_boundary.py"),
    ("source_metrics", "analysis/metrics.py"),
    ("amd_metrics", "cluster/amd_qwen_metrics.py"),
    ("pass1_metrics", "analysis/build_pass1_primary_results.py"),
    ("gdp_weights", "analysis/build_gdp_employment_weights.py"),
    ("active_energy", "analysis/glm_primary_energy_update.py"),
    ("completion_energy", "analysis/build_completion_energy_curve.py"),
    ("net_metrics", "analysis/build_net_metrics.py"),
    ("category_omission", "analysis/category_omission_robustness.py"),
    ("reference_decomposition", "analysis/decompose_reference_condition.py"),
)
AUDIT_STEPS = (
    ("judge_comparison", "analysis/compare_glm_judges.py"),
    ("judge_repeat_concordance", "analysis/verify_glm_judge_repeat_concordance_20260924.py"),
    ("contamination_audit", "analysis/benchmark_contamination_audit.py"),
)
STEPS = CORE_STEPS + AUDIT_STEPS


def main() -> None:
    names = [name for name, _ in STEPS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-step", choices=names)
    parser.add_argument("--through-step", choices=names)
    parser.add_argument("--list", action="store_true", help="List the numerical steps and exit")
    parser.add_argument("--include-audits", action="store_true", help="Run optional archived-judge and contamination audits")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--energy-boundary", choices=("total", "accelerator"), default="total",
        help="Energy components: total (GPU + CPU + DRAM, default) or accelerator (GPU)",
    )
    args = parser.parse_args()
    if args.list:
        for name, script in STEPS:
            suffix = " (additional archived inputs)" if (name, script) in AUDIT_STEPS else ""
            print(f"{name}: {script}{suffix}")
        return
    first = names.index(args.from_step) if args.from_step else 0
    last = names.index(args.through_step) + 1 if args.through_step else (
        len(STEPS) if args.include_audits or first >= len(CORE_STEPS) else len(CORE_STEPS)
    )
    if first >= last:
        parser.error("--from-step must precede --through-step")
    env = os.environ.copy()
    env["J2G_ENERGY_BOUNDARY"] = args.energy_boundary
    for name, script in STEPS[first:last]:
        if name == "energy_boundary" and args.energy_boundary == "accelerator":
            continue
        path = ROOT / script
        if not path.is_file():
            raise FileNotFoundError(path)
        command = [sys.executable, str(path)]
        print(f"{name}: {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
