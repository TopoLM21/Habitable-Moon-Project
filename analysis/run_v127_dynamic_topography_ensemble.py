#!/usr/bin/env python3
"""Run paired v0.27 dynamic-topography validation worlds locally."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
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
MODES = {"control": False, "dynamic": True}


def _prepare_config(seed: str, mode: str) -> Path:
    config = yaml.safe_load(SEED_CONFIGS[seed].read_text(encoding="utf-8"))
    config = deepcopy(config)
    config["plume_dynamic_topography"]["enabled"] = MODES[mode]
    generated = RESULTS / "v127_dynamic_topography_configs"
    generated.mkdir(parents=True, exist_ok=True)
    path = generated / f"validation_{seed}_{mode}.yaml"
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _output_dir(seed: str, mode: str, end_time: float) -> Path:
    label = f"{end_time:g}".replace(".", "p")
    return RESULTS / f"v127_{mode}_sub3_{label}_seed_{seed}"


def _run_one(seed: str, mode: str, end_time: float, dt: float, force: bool) -> str:
    config = _prepare_config(seed, mode)
    output = _output_dir(seed, mode, end_time)
    summary = output / "summary_v127.json"
    if summary.exists() and not force:
        return f"SKIP seed={seed} mode={mode} (summary exists)"
    output.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_long_evolution_v127.py"),
            "--config",
            str(config),
            "--end-time",
            f"{end_time:g}",
            "--dt",
            f"{dt:g}",
            "--output",
            str(output),
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
        raise RuntimeError(f"seed={seed} mode={mode} did not create {summary}")
    return f"DONE seed={seed} mode={mode}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-time", type=float, default=500.0)
    parser.add_argument("--dt", type=float, default=4.0)
    parser.add_argument("--seeds", nargs="+", choices=SEED_CONFIGS, default=list(SEED_CONFIGS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    tasks = [(seed, mode) for seed in args.seeds for mode in args.modes]
    jobs = max(1, min(int(args.jobs), len(tasks)))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(
                _run_one, seed, mode, args.end_time, args.dt, args.force
            ): (seed, mode)
            for seed, mode in tasks
        }
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
