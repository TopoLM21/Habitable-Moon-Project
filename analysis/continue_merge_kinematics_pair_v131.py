#!/usr/bin/env python3
"""Continue both v0.31 merger-kinematics branches in place."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moon_gui.backend import checkpoint_name, read_checkpoint_time, segment_targets


RULES = ("area_weighted", "inertia_tensor")


def complete_checkpoints(case: Path) -> list[Path]:
    checkpoints = []
    for path in case.glob("gui_checkpoint_*_Myr"):
        if (path / "meta.json").is_file() and (path / "state.npz").is_file():
            checkpoints.append(path)
    return sorted(checkpoints, key=read_checkpoint_time)


def configured_rule(config_path: Path) -> str:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return str(config.get("plate_topology", {}).get("merge_kinematics_rule", ""))


def run_case(
    *,
    pair_root: Path,
    rule: str,
    end_time_myr: float,
    dt_myr: float,
    checkpoint_interval_myr: float,
    frame_interval_myr: float,
) -> tuple[float, float]:
    case = pair_root / rule
    config = case / "case_config.yaml"
    if not config.is_file():
        raise ValueError(f"Missing case configuration: {config}")
    if configured_rule(config) != rule:
        raise ValueError(f"Case {case} is not configured for {rule}")

    checkpoints = complete_checkpoints(case)
    if not checkpoints:
        raise ValueError(f"No complete checkpoints in {case}")
    previous = checkpoints[-1]
    start_time = read_checkpoint_time(previous)
    targets = segment_targets(start_time, end_time_myr, checkpoint_interval_myr, dt_myr)
    if not targets:
        print(f"[{rule}] already reaches {start_time:g} Myr")
        return start_time, start_time

    for index, target in enumerate(targets):
        checkpoint = case / checkpoint_name(target)
        if checkpoint.exists():
            raise ValueError(f"Refusing to overwrite checkpoint: {checkpoint}")
        command = [
            sys.executable,
            str(ROOT / "run_long_evolution_v131.py"),
            "--config",
            str(config),
            "--output",
            str(case),
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
    return start_time, targets[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", required=True, type=Path)
    parser.add_argument("--end-time", type=float, default=500.0)
    parser.add_argument("--dt", type=float, default=4.0)
    parser.add_argument("--checkpoint-interval", type=float, default=20.0)
    parser.add_argument("--frame-interval", type=float, default=20.0)
    args = parser.parse_args()

    pair_root = args.pair_root.resolve()
    if not pair_root.is_dir():
        raise ValueError(f"Pair root does not exist: {pair_root}")
    started = datetime.now(timezone.utc).isoformat()
    ranges = {}
    for rule in RULES:
        start, end = run_case(
            pair_root=pair_root,
            rule=rule,
            end_time_myr=float(args.end_time),
            dt_myr=float(args.dt),
            checkpoint_interval_myr=float(args.checkpoint_interval),
            frame_interval_myr=float(args.frame_interval),
        )
        ranges[rule] = {"start_time_myr": start, "end_time_myr": end}

    record = {
        "format": "v131_merge_kinematics_pair_continuation",
        "pair_root": str(pair_root),
        "requested_end_time_myr": float(args.end_time),
        "dt_myr": float(args.dt),
        "checkpoint_interval_myr": float(args.checkpoint_interval),
        "frame_interval_myr": float(args.frame_interval),
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "ranges": ranges,
    }
    time_token = f"{args.end_time:g}".replace(".", "p")
    name = f"pair_continuation_to_{time_token}_Myr.json"
    (pair_root / name).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(pair_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
