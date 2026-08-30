#!/usr/bin/env python3
"""Run the final flow-coupled plume validation worlds for v0.31."""

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


def _prepare_config(seed: str, subdivision: int) -> Path:
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
        "enabled": True,
        "minimum_speed_km_per_myr": 8.0,
        "maximum_speed_km_per_myr": 30.0,
        "direction_persistence_myr": 80.0,
        "direction_memory": 0.65,
        "path_sample_interval_myr": 4.0,
        "area_normalization_exponent": 1.4,
    }
    config["plume_flow_coupling"] = {
        "enabled": True,
        "mantle_flow_velocity_fraction": 0.35,
        "residual_drift_fraction": 0.30,
        "mantle_flow_sampling_radius_km": 550.0,
    }
    generated = RESULTS / "v131_validation_configs"
    generated.mkdir(parents=True, exist_ok=True)
    path = generated / f"validation_{seed}_flow_sub{subdivision}.yaml"
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _output(seed: str, subdivision: int, end_time: float) -> Path:
    label = f"{end_time:g}".replace(".", "p")
    return RESULTS / f"v131_validation_flow_sub{subdivision}_{label}_seed_{seed}"


def _run(seed: str, subdivision: int, end_time: float, dt: float, force: bool) -> str:
    config = _prepare_config(seed, subdivision)
    output = _output(seed, subdivision, end_time)
    summary = output / "summary_v131.json"
    if summary.exists() and not force:
        return f"SKIP seed={seed} flow sub={subdivision}"
    output.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_long_evolution_v131.py"),
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
            f"seed={seed} flow sub={subdivision} failed; see {output / 'run.log'}"
        )
    if not summary.exists():
        raise RuntimeError(f"missing {summary}")
    return f"DONE seed={seed} flow sub={subdivision}"


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
    tasks = [(seed, 3) for seed in args.seeds]
    if not args.skip_sub4:
        tasks.append(("20260806", 4))
    with ThreadPoolExecutor(max_workers=max(1, min(args.jobs, len(tasks)))) as pool:
        futures = {
            pool.submit(_run, seed, sub, args.end_time, args.dt, args.force): (seed, sub)
            for seed, sub in tasks
        }
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
