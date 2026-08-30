#!/usr/bin/env python3
"""Run paired fixed/mobile-source v0.30 validation worlds."""

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
MODES = ("fixed", "mobile")


def _prepare_config(seed: str, mode: str, subdivision: int) -> Path:
    config = deepcopy(yaml.safe_load(SEED_CONFIGS[seed].read_text(encoding="utf-8")))
    config["mesh"]["subdivisions"] = int(subdivision)
    config["hotspot_tracks"] = {
        "enabled": True,
        "head_tail_separation_enabled": True,
        "magmatic_thermal_weakening_enabled": True,
        "couple_thermal_to_lithosphere": True,
        "dike_localization_enabled": True,
        "underplate_evolution_enabled": True,
        "area_normalize_component_flux": True,
    }
    config["plume_drift"] = {
        "enabled": mode == "mobile",
        "minimum_speed_km_per_myr": 8.0,
        "maximum_speed_km_per_myr": 30.0,
        "direction_persistence_myr": 80.0,
        "direction_memory": 0.65,
        "path_sample_interval_myr": 4.0,
        "area_normalization_exponent": 1.4,
    }
    generated = RESULTS / "v130_validation_configs"
    generated.mkdir(parents=True, exist_ok=True)
    path = generated / f"validation_{seed}_{mode}_sub{subdivision}.yaml"
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _output(seed: str, mode: str, subdivision: int, end_time: float) -> Path:
    label = f"{end_time:g}".replace(".", "p")
    return RESULTS / f"v130_validation_{mode}_sub{subdivision}_{label}_seed_{seed}"


def _run(seed: str, mode: str, subdivision: int, end_time: float,
         dt: float, force: bool) -> str:
    config = _prepare_config(seed, mode, subdivision)
    output = _output(seed, mode, subdivision, end_time)
    summary = output / "summary_v130.json"
    if summary.exists() and not force:
        return f"SKIP seed={seed} mode={mode} sub={subdivision}"
    output.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_long_evolution_v130.py"),
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
            f"seed={seed} mode={mode} sub={subdivision} failed; "
            f"see {output / 'run.log'}"
        )
    if not summary.exists():
        raise RuntimeError(f"missing {summary}")
    return f"DONE seed={seed} mode={mode} sub={subdivision}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-time", type=float, default=500.0)
    parser.add_argument("--dt", type=float, default=4.0)
    parser.add_argument("--seeds", nargs="+", choices=SEED_CONFIGS,
                        default=list(SEED_CONFIGS))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-sub4", action="store_true")
    args = parser.parse_args()
    tasks = [(seed, mode, 3) for seed in args.seeds for mode in MODES]
    if not args.skip_sub4:
        tasks.extend(("20260806", mode, 4) for mode in MODES)
    jobs = max(1, min(int(args.jobs), len(tasks)))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(
                _run, seed, mode, subdivision, args.end_time, args.dt, args.force
            ): (seed, mode, subdivision)
            for seed, mode, subdivision in tasks
        }
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
