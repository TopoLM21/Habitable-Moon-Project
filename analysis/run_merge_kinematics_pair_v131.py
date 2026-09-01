#!/usr/bin/env python3
"""Replay a checkpoint with area- and inertia-weighted plate mergers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moon_gui.backend import checkpoint_name, segment_targets


RULES = ("area_weighted", "inertia_tensor")


def checkpoint_time(path: Path) -> float:
    metadata = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    return float(metadata["time_myr"])


def checkpoint_cells(path: Path) -> int:
    with np.load(path / "state.npz", allow_pickle=False) as state:
        return int(state["system_cell_plate"].shape[0])


def write_case_config(
    source: Path,
    destination: Path,
    subdivisions: int,
    rule: str,
) -> Path:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config.setdefault("mesh", {})["subdivisions"] = int(subdivisions)
    config.setdefault("plate_topology", {})["merge_kinematics_rule"] = rule
    config.setdefault("diagnostic_experiment", {}).update(
        {
            "name": "v131_merge_kinematics_pair",
            "merge_kinematics_rule": rule,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    path = destination / "case_config.yaml"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return path


def run_case(
    *,
    rule: str,
    source_config: Path,
    resume: Path,
    output: Path,
    subdivisions: int,
    end_time_myr: float,
    dt_myr: float,
    checkpoint_interval_myr: float,
    frame_interval_myr: float,
) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite non-empty case output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = write_case_config(source_config, output, subdivisions, rule)
    previous = resume
    targets = segment_targets(
        checkpoint_time(resume), end_time_myr, checkpoint_interval_myr, dt_myr
    )
    for index, target in enumerate(targets):
        checkpoint = output / checkpoint_name(target)
        command = [
            sys.executable,
            str(ROOT / "run_long_evolution_v131.py"),
            "--config",
            str(config),
            "--output",
            str(output),
            "--resume",
            str(previous),
            "--end-time",
            f"{target:g}",
            "--dt",
            f"{dt_myr:g}",
            "--checkpoint",
            str(checkpoint),
            "--frame-interval",
            f"{frame_interval_myr:g}",
            "--surface-only-frames",
        ]
        if index == len(targets) - 1:
            command.append("--finalize")
        print(f"[{rule}] segment {index + 1}/{len(targets)} -> {target:g} Myr")
        subprocess.run(command, cwd=ROOT, check=True)
        previous = checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, default=ROOT / "configs" / "canonical_moon.yaml")
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--end-time", type=float, default=400.0)
    parser.add_argument("--subdivisions", type=int, default=5)
    parser.add_argument("--dt", type=float, default=4.0)
    parser.add_argument("--checkpoint-interval", type=float, default=20.0)
    parser.add_argument("--frame-interval", type=float, default=20.0)
    args = parser.parse_args()

    source_config = args.source_config.resolve()
    resume = args.resume.resolve()
    output_root = args.output_root.resolve()
    if checkpoint_cells(resume) != 20 * 4 ** int(args.subdivisions):
        raise ValueError("Resume checkpoint cell count does not match subdivisions")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "v131_merge_kinematics_pair",
        "source_config": str(source_config),
        "resume_checkpoint": str(resume),
        "start_time_myr": checkpoint_time(resume),
        "end_time_myr": float(args.end_time),
        "subdivisions": int(args.subdivisions),
        "dt_myr": float(args.dt),
        "checkpoint_interval_myr": float(args.checkpoint_interval),
        "frame_interval_myr": float(args.frame_interval),
        "rules": list(RULES),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (output_root / "pair_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    for rule in RULES:
        run_case(
            rule=rule,
            source_config=source_config,
            resume=resume,
            output=output_root / rule,
            subdivisions=args.subdivisions,
            end_time_myr=args.end_time,
            dt_myr=args.dt,
            checkpoint_interval_myr=args.checkpoint_interval,
            frame_interval_myr=args.frame_interval,
        )
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
