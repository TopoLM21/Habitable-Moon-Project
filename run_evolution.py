#!/usr/bin/env python3
"""Run v0.2: rigid Lagrangian plate patches through time, without crust."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tectonics.evolution import snapshot_at_time, snapshot_times
from tectonics.simulation import build_prototype, load_config
from visualization.evolution import (
    build_gif,
    save_boundary_separation_history,
    save_coverage_history,
    save_evolution_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "v0.2 rigid plate-patch evolution. Gaps/overlaps are left unresolved on purpose; "
            "spreading, subduction and crust age are deferred to v0.3."
        )
    )
    parser.add_argument("--config", default="configs/canonical_moon.yaml")
    parser.add_argument("--output", default=None, help="Override output directory")
    parser.add_argument("--duration", type=float, default=None, help="Override duration in Myr")
    parser.add_argument("--interval", type=float, default=None, help="Override frame interval in Myr")
    parser.add_argument("--no-gif", action="store_true", help="Do not assemble history.gif")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    initial = build_prototype(config)

    evolution_cfg = config["evolution"]
    duration = float(args.duration if args.duration is not None else evolution_cfg["duration_myr"])
    interval = float(args.interval if args.interval is not None else evolution_cfg["frame_interval_myr"])
    times = snapshot_times(duration, interval)

    output_dir = Path(args.output or config["output"].get("evolution_directory", "outputs_v02"))
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(config["output"].get("evolution_dpi", 130))

    rows: list[dict[str, object]] = []
    frame_paths: list[Path] = []
    mean_sep: list[float] = []
    p95_sep: list[float] = []
    max_sep: list[float] = []
    uncovered: list[float] = []
    multiply: list[float] = []

    for index, time_myr in enumerate(times):
        snap = snapshot_at_time(
            mesh=initial.mesh,
            initial_system=initial.plates,
            initial_boundaries=initial.boundaries,
            radius_km=float(config["moon"]["radius_km"]),
            time_myr=float(time_myr),
        )
        frame = frames_dir / f"frame_{index:04d}_{time_myr:08.3f}_Myr.png"
        save_evolution_frame(initial, snap, frame, dpi=dpi)
        frame_paths.append(frame)

        sep = snap.boundary_separation_km
        mean_value = float(np.mean(sep)) if len(sep) else 0.0
        p95_value = float(np.percentile(sep, 95)) if len(sep) else 0.0
        max_value = float(np.max(sep)) if len(sep) else 0.0
        mean_sep.append(mean_value)
        p95_sep.append(p95_value)
        max_sep.append(max_value)
        uncovered.append(snap.coverage.uncovered_cell_fraction)
        multiply.append(snap.coverage.multiply_covered_cell_fraction)

        rows.append(
            {
                "time_myr": float(time_myr),
                "mean_boundary_pair_separation_km": mean_value,
                "p95_boundary_pair_separation_km": p95_value,
                "max_boundary_pair_separation_km": max_value,
                "uncovered_reference_cell_fraction": snap.coverage.uncovered_cell_fraction,
                "multiply_covered_reference_cell_fraction": snap.coverage.multiply_covered_cell_fraction,
                "nearest_marker_angle_deg_mean": snap.coverage.nearest_marker_angle_deg_mean,
                "nearest_marker_angle_deg_max": snap.coverage.nearest_marker_angle_deg_max,
            }
        )
        print(
            f"t={time_myr:7.2f} Myr | mean edge split={mean_value:8.1f} km | "
            f"max={max_value:8.1f} km | gaps={100*snap.coverage.uncovered_cell_fraction:5.1f}% | "
            f"overlaps={100*snap.coverage.multiply_covered_cell_fraction:5.1f}%"
        )

    save_boundary_separation_history(
        times,
        np.asarray(mean_sep),
        np.asarray(p95_sep),
        np.asarray(max_sep),
        output_dir / "boundary_separation_history.png",
    )
    save_coverage_history(
        times,
        np.asarray(uncovered),
        np.asarray(multiply),
        output_dir / "coverage_history.png",
    )

    with (output_dir / "kinematic_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if not args.no_gif:
        build_gif(
            frame_paths,
            output_dir / "history.gif",
            frame_duration_ms=int(evolution_cfg.get("gif_frame_duration_ms", 350)),
        )

    summary = {
        "version": "0.2",
        "model": "rigid Lagrangian plate patches under fixed Euler rotations",
        "crust_physics": False,
        "duration_myr": duration,
        "frame_interval_myr": interval,
        "frames": len(times),
        "mesh_cells": initial.mesh.cell_count,
        "plate_count": len(initial.plates.plates),
        "initial_boundary_edges": len(initial.boundaries),
        "initial_rigid_motion_residual": initial.rigid_residual,
        "final": rows[-1],
        "warning": (
            "v0.2 intentionally leaves divergent gaps and convergent overlaps unresolved. "
            "No oceanic crust, crust age, spreading or subduction is present; those are v0.3."
        ),
    }
    with (output_dir / "summary_v02.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved v0.2 outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
