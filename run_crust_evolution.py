#!/usr/bin/env python3
"""Run v0.3 oceanic crust evolution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tectonics.crust import (
    OceanicSnapshot,
    advance_oceanic_crust,
    boundary_records_for_state,
    initialize_oceanic_crust,
)
from tectonics.evolution import snapshot_times
from tectonics.simulation import build_prototype, load_config
from visualization.crust import (
    build_gif,
    save_boundary_map,
    save_budget_history,
    save_crust_age_map,
    save_crust_frame,
    save_gap_overlap_history,
    save_plate_map,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v0.3 oceanic crust evolution with spreading and subduction bookkeeping")
    parser.add_argument("--config", default="configs/canonical_moon.yaml")
    parser.add_argument("--output", default=None, help="Override output directory")
    parser.add_argument("--duration", type=float, default=None, help="Override duration in Myr")
    parser.add_argument("--interval", type=float, default=None, help="Override frame interval in Myr")
    parser.add_argument("--dt", type=float, default=None, help="Override internal time step in Myr")
    parser.add_argument("--no-gif", action="store_true", help="Do not assemble history.gif")
    return parser.parse_args()


def _substeps(start: float, end: float, dt: float) -> list[float]:
    if end < start:
        raise ValueError("end must be >= start")
    out: list[float] = []
    t = start
    while t + dt < end - 1e-12:
        out.append(dt)
        t += dt
    if end > t + 1e-12:
        out.append(end - t)
    return out


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    initial = build_prototype(config)

    evo = config.get("crust_evolution", {})
    duration = float(args.duration if args.duration is not None else evo.get("duration_myr", 50.0))
    interval = float(args.interval if args.interval is not None else evo.get("frame_interval_myr", 5.0))
    dt = float(args.dt if args.dt is not None else evo.get("time_step_myr", 0.5))
    times = snapshot_times(duration, interval)

    output_dir = Path(args.output or config["output"].get("crust_directory", "outputs_v03"))
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(config["output"].get("crust_dpi", 125))
    radius_km = float(config["moon"]["radius_km"])
    normal_threshold = float(config["classification"]["normal_threshold_km_per_myr"])
    inactive_speed = float(config["classification"]["inactive_speed_km_per_myr"])

    state = initialize_oceanic_crust(initial.plates)
    rows: list[dict[str, float]] = []
    frame_paths: list[Path] = []

    # initial frame
    boundaries = boundary_records_for_state(initial.mesh, state, initial.plates, radius_km, normal_threshold, inactive_speed)
    snapshot = OceanicSnapshot(state=state, boundaries=boundaries, diagnostics=None)
    frame0 = frames_dir / f"frame_{0:04d}_{0.0:08.3f}_Myr.png"
    save_crust_frame(initial.mesh, snapshot, frame0, dpi=dpi)
    frame_paths.append(frame0)
    rows.append({
        "time_myr": 0.0,
        "dt_myr": 0.0,
        "pre_resolution_gap_fraction": 0.0,
        "pre_resolution_overlap_fraction": 0.0,
        "created_area_km2": 0.0,
        "subducted_area_km2": 0.0,
        "mean_age_myr": float(np.mean(state.crust_age_myr)),
        "max_age_myr": float(np.max(state.crust_age_myr)),
    })

    for frame_index, target_time in enumerate(times[1:], start=1):
        for dti in _substeps(state.time_myr, float(target_time), dt):
            state, diag = advance_oceanic_crust(
                mesh=initial.mesh,
                initial_system=initial.plates,
                state=state,
                dt_myr=float(dti),
                radius_km=radius_km,
            )
            rows.append({
                "time_myr": diag.time_myr,
                "dt_myr": diag.dt_myr,
                "pre_resolution_gap_fraction": diag.pre_resolution_gap_fraction,
                "pre_resolution_overlap_fraction": diag.pre_resolution_overlap_fraction,
                "created_area_km2": diag.created_area_km2,
                "subducted_area_km2": diag.subducted_area_km2,
                "mean_age_myr": diag.mean_age_myr,
                "max_age_myr": diag.max_age_myr,
            })

        boundaries = boundary_records_for_state(initial.mesh, state, initial.plates, radius_km, normal_threshold, inactive_speed)
        frame = frames_dir / f"frame_{frame_index:04d}_{target_time:08.3f}_Myr.png"
        last_diag = None if not rows else None
        # Snapshot annotation uses last time-step diagnostics for context.
        if len(rows) >= 2:
            lr = rows[-1]
            from tectonics.crust import StepDiagnostics
            last_diag = StepDiagnostics(
                time_myr=float(lr["time_myr"]),
                dt_myr=float(lr["dt_myr"]),
                pre_resolution_gap_fraction=float(lr["pre_resolution_gap_fraction"]),
                pre_resolution_overlap_fraction=float(lr["pre_resolution_overlap_fraction"]),
                created_area_km2=float(lr["created_area_km2"]),
                subducted_area_km2=float(lr["subducted_area_km2"]),
                mean_age_myr=float(lr["mean_age_myr"]),
                max_age_myr=float(lr["max_age_myr"]),
            )
        snapshot = OceanicSnapshot(state=state, boundaries=boundaries, diagnostics=last_diag)
        save_crust_frame(initial.mesh, snapshot, frame, dpi=dpi)
        frame_paths.append(frame)
        print(
            f"t={state.time_myr:7.2f} Myr | mean age={np.mean(state.crust_age_myr):6.2f} | "
            f"max age={np.max(state.crust_age_myr):6.2f} | gaps={100*rows[-1]['pre_resolution_gap_fraction']:5.1f}% | "
            f"overlaps={100*rows[-1]['pre_resolution_overlap_fraction']:5.1f}%"
        )

    times_hist = np.asarray([r["time_myr"] for r in rows[1:]], dtype=np.float64)
    created_hist = np.asarray([r["created_area_km2"] for r in rows[1:]], dtype=np.float64)
    subducted_hist = np.asarray([r["subducted_area_km2"] for r in rows[1:]], dtype=np.float64)
    mean_age_hist = np.asarray([r["mean_age_myr"] for r in rows[1:]], dtype=np.float64)
    gap_hist = np.asarray([r["pre_resolution_gap_fraction"] for r in rows[1:]], dtype=np.float64)
    overlap_hist = np.asarray([r["pre_resolution_overlap_fraction"] for r in rows[1:]], dtype=np.float64)

    save_crust_age_map(initial.mesh, state, output_dir / "crust_age_final.png", dpi=int(config["output"].get("dpi", 180)))
    save_plate_map(initial.mesh, state, output_dir / "plate_map_final.png", dpi=int(config["output"].get("dpi", 180)))
    final_boundaries = boundary_records_for_state(initial.mesh, state, initial.plates, radius_km, normal_threshold, inactive_speed)
    save_boundary_map(initial.mesh, state, final_boundaries, output_dir / "boundary_map_final.png", dpi=int(config["output"].get("dpi", 180)))
    if len(times_hist):
        save_budget_history(times_hist, created_hist, subducted_hist, mean_age_hist, output_dir / "crust_budget_history.png")
        save_gap_overlap_history(times_hist, gap_hist, overlap_hist, output_dir / "gap_overlap_history.png")

    with (output_dir / "crust_budget.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if not args.no_gif:
        build_gif(frame_paths, output_dir / "history.gif", frame_duration_ms=int(evo.get("gif_frame_duration_ms", 350)))

    total_surface = float(np.sum(initial.mesh.physical_cell_areas_km2(radius_km)))
    summary = {
        "version": "0.3",
        "model": "single-layer oceanic crust on a fixed diagnostic icosphere",
        "duration_myr": duration,
        "frame_interval_myr": interval,
        "time_step_myr": dt,
        "frames": len(frame_paths),
        "mesh_cells": initial.mesh.cell_count,
        "surface_area_km2": total_surface,
        "plate_count": len(initial.plates.plates),
        "final_mean_age_myr": float(np.mean(state.crust_age_myr)),
        "final_max_age_myr": float(np.max(state.crust_age_myr)),
        "mean_created_area_per_step_km2": float(np.mean(created_hist)) if len(created_hist) else 0.0,
        "mean_subducted_area_per_step_km2": float(np.mean(subducted_hist)) if len(subducted_hist) else 0.0,
        "mean_gap_fraction_before_resolution": float(np.mean(gap_hist)) if len(gap_hist) else 0.0,
        "mean_overlap_fraction_before_resolution": float(np.mean(overlap_hist)) if len(overlap_hist) else 0.0,
        "final_boundary_count": len(final_boundaries),
        "notes": [
            "Every surface cell is occupied by exactly one top-surface oceanic parcel after each step.",
            "Divergent openings are filled with newborn crust age 0.",
            "Overlaps are resolved by consuming older oceanic crust preferentially.",
            "Plate motions remain fixed Euler rotations from the initial prototype.",
            "No continents or topography yet; those are deferred to later versions.",
        ],
    }
    with (output_dir / "summary_v03.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved v0.3 outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
