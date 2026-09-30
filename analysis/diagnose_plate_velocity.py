"""Audit one or more saved Genesis continuation directories without advancing them.

Example: python analysis/diagnose_plate_velocity.py CHECKPOINT --output analysis/velocity_budget
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tectonics.plate_velocity_diagnostics import diagnose_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-myr", type=float, default=1.)
    parser.add_argument("--projection", choices=("legacy_area_mean", "velocity_least_squares"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for source in args.checkpoints:
        row = diagnose_checkpoint(source, step_myr=args.step_myr, projection=args.projection)
        name = f"{row['time_myr']:.8f}_{source.name}.json"
        (args.output/name).write_text(json.dumps(row, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        compact = {key: row[key] for key in ("source", "time_myr", "plate_count", "mantle",
            "stages_speed_km_per_myr", "contributions_speed_km_per_myr", "boundaries", "verification")}
        summary.append(compact)
        print(f"t={row['time_myr']:.8f} plates={row['plate_count']} "
              f"actual_mean={row['stages_speed_km_per_myr']['current']['mean']:.12g} km/Myr")
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
