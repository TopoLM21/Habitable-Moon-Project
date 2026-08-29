#!/usr/bin/env python3
"""Run the 2x2x2 v0.28 uplift/magmatism/mechanical-rifting factorial."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from itertools import product
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SEED_CONFIGS = {
    "20260806": ROOT / "configs" / "validation_v125_sub3.yaml",
    "20260807": ROOT / "configs" / "validation_v125_sub3_seed_20260807.yaml",
    "20260808": ROOT / "configs" / "validation_v125_sub3_seed_20260808.yaml",
    "20260809": ROOT / "configs" / "validation_v125_sub3_seed_20260809.yaml",
}
MODES = {
    f"d{dynamic}m{magmatism}r{rifting}": (bool(dynamic), bool(magmatism), bool(rifting))
    for dynamic, magmatism, rifting in product((0, 1), repeat=3)
}


def _prepare_config(seed: str, mode: str) -> Path:
    dynamic, magmatism, rifting = MODES[mode]
    config = deepcopy(
        yaml.safe_load(SEED_CONFIGS[seed].read_text(encoding="utf-8"))
    )
    config["plume_dynamic_topography"]["enabled"] = dynamic
    config["plume_magmatism"]["enabled"] = magmatism
    # Keep the productivity diagnostic alive in every case.  This switch only
    # controls whether the forcing is coupled into the mechanical rift solver.
    config["plume_rifting"]["enabled"] = True
    config["plume_rifting"]["couple_to_lithosphere"] = rifting
    generated = RESULTS / "v128_factorial_configs"
    generated.mkdir(parents=True, exist_ok=True)
    path = generated / f"validation_{seed}_{mode}.yaml"
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _output(seed: str, mode: str, end_time: float) -> Path:
    label = f"{end_time:g}".replace(".", "p")
    return RESULTS / f"v128_factorial_{mode}_sub3_{label}_seed_{seed}"


def _run(seed: str, mode: str, end_time: float, dt: float, force: bool) -> str:
    config = _prepare_config(seed, mode)
    output = _output(seed, mode, end_time)
    summary = output / "summary_v128.json"
    if summary.exists() and not force:
        return f"SKIP seed={seed} mode={mode}"
    output.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_long_evolution_v128.py"),
            "--config", str(config),
            "--end-time", f"{end_time:g}",
            "--dt", f"{dt:g}",
            "--output", str(output),
            "--finalize",
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"seed={seed} mode={mode} failed with exit code "
            f"{completed.returncode}; see {output / 'run.log'}"
        )
    if not summary.exists():
        raise RuntimeError(f"missing {summary}")
    return f"DONE seed={seed} mode={mode}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-time", type=float, default=500.0)
    parser.add_argument("--dt", type=float, default=4.0)
    parser.add_argument("--seeds", nargs="+", choices=SEED_CONFIGS,
                        default=["20260806"])
    parser.add_argument("--modes", nargs="+", choices=MODES,
                        default=list(MODES))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    tasks = [(seed, mode) for seed in args.seeds for mode in args.modes]
    jobs = max(1, min(int(args.jobs), len(tasks)))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(_run, seed, mode, args.end_time, args.dt, args.force):
            (seed, mode)
            for seed, mode in tasks
        }
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
