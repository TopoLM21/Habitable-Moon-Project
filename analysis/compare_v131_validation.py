#!/usr/bin/env python3
"""Compare fixed, independent-drift and flow-coupled plume sources."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SEEDS = ("20260806", "20260807", "20260808", "20260809")


def _csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _base_rows() -> list[dict]:
    with (ROOT / "V130_MOBILE_PLUME_VALIDATION_500MYR.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            if key == "mode":
                converted[key] = "independent" if value == "mobile" else value
            elif key in {"seed", "subdivision", "plate_count", "total_clip_cells", "active_plumes"}:
                converted[key] = int(value)
            else:
                converted[key] = float(value)
        converted.update(
            mean_resolved_flow_speed_km_per_myr=0.0,
            mean_residual_speed_km_per_myr=0.0,
            mean_effective_flow_alignment=0.0,
        )
        out.append(converted)
    return out


def _flow_row(seed: str, subdivision: int) -> dict:
    output = RESULTS / f"v131_validation_flow_sub{subdivision}_500_seed_{seed}"
    summary = json.loads((output / "summary_v131.json").read_text(encoding="utf-8"))
    magmatic = _csv_rows(output / "plume_magmatism_history.csv")[-1]
    hotspot_rows = _csv_rows(output / "hotspot_track_history.csv")
    hotspot = hotspot_rows[-1]
    drift = _csv_rows(output / "plume_drift_history.csv")[-1]
    coupling = _csv_rows(output / "plume_flow_coupling_history.csv")[-1]
    plume = _csv_rows(output / "mantle_plume_history.csv")[-1]
    return {
        "seed": int(seed),
        "mode": "flow",
        "subdivision": int(subdivision),
        "generated_igneous_volume_km3": float(magmatic["cumulative_generated_total_volume_km3"]),
        "surface_igneous_volume_km3": sum(float(magmatic[k]) for k in ("surface_extrusive_volume_km3", "surface_dyke_volume_km3", "surface_underplate_volume_km3")),
        "igneous_ledger_error_km3": float(magmatic["global_igneous_ledger_error_km3"]),
        "continental_ledger_error_km3": float(summary["final_global_continental_ledger_error_km3"]),
        "max_rift_extension": float(summary["final_max_rift_extension"]),
        "maximum_elevation_m": float(summary["max_elevation_m"]),
        "land_fraction": float(summary["final_land_area_fraction"]),
        "plate_count": int(summary["final_plate_count"]),
        "age_distance_correlation": float(hotspot["hotspot_track_age_distance_correlation"]),
        "head_productivity_time_integral_myr": sum(float(r["mean_head_productivity"]) * float(r["dt_myr"]) for r in hotspot_rows),
        "tail_productivity_time_integral_myr": sum(float(r["mean_tail_productivity"]) * float(r["dt_myr"]) for r in hotspot_rows),
        "population_source_path_length_km": float(drift["population_source_path_length_km"]),
        "population_source_bend_angle_deg": float(drift["population_source_bend_angle_deg"]),
        "mean_source_speed_km_per_myr": float(drift["mean_source_speed_km_per_myr"]),
        "mean_plate_speed_km_per_myr": float(drift["mean_overlying_plate_speed_km_per_myr"]),
        "mean_relative_track_speed_km_per_myr": float(drift["mean_relative_track_speed_km_per_myr"]),
        "mean_source_motion_deflection_deg": float(drift["mean_source_motion_deflection_deg"]),
        "total_clip_cells": int(summary["total_numerical_min_clip_cells"]) + int(summary["total_numerical_max_clip_cells"]),
        "active_plumes": int(float(plume["active_plume_count"])),
        "mean_resolved_flow_speed_km_per_myr": float(coupling["mean_resolved_flow_speed_km_per_myr"]),
        "mean_residual_speed_km_per_myr": float(coupling["mean_residual_speed_km_per_myr"]),
        "mean_effective_flow_alignment": float(coupling["mean_effective_flow_alignment"]),
    }


def main() -> None:
    rows = _base_rows()
    rows.extend(_flow_row(seed, 3) for seed in SEEDS)
    rows.append(_flow_row("20260806", 4))
    path = ROOT / "V131_FLOW_COUPLED_PLUME_VALIDATION_500MYR.csv"
    fieldnames = list(rows[-1].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    sub3 = [row for row in rows if row["subdivision"] == 3]
    modes = ("fixed", "independent", "flow")
    metrics = (
        ("population_source_path_length_km", "Source path, km"),
        ("population_source_bend_angle_deg", "Cumulative bend, deg"),
        ("age_distance_correlation", "Age-distance correlation"),
        ("max_rift_extension", "Maximum rift extension"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.8))
    for ax, (metric, title) in zip(axes, metrics):
        values = [np.mean([row[metric] for row in sub3 if row["mode"] == mode]) for mode in modes]
        ax.bar(range(3), values, color=["#777777", "#d95f02", "#1b9e77"])
        ax.set_xticks(range(3), modes, rotation=20)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("v0.31 final 500-Myr plume-source comparison (subdivision 3 means)")
    fig.tight_layout()
    fig.savefig(ROOT / "V131_FLOW_COUPLED_PLUME_VALIDATION_500MYR.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(path)
    print("metric,fixed,independent,flow")
    for metric in (
        "generated_igneous_volume_km3", "surface_igneous_volume_km3",
        "max_rift_extension", "maximum_elevation_m", "land_fraction",
        "plate_count", "age_distance_correlation",
        "population_source_path_length_km", "population_source_bend_angle_deg",
        "mean_source_speed_km_per_myr", "mean_relative_track_speed_km_per_myr",
    ):
        values = [np.mean([row[metric] for row in sub3 if row["mode"] == mode]) for mode in modes]
        print(metric + "," + ",".join(f"{value:.10g}" for value in values))
    print(
        "maximum absolute ledgers:",
        max(abs(row["igneous_ledger_error_km3"]) for row in rows),
        max(abs(row["continental_ledger_error_km3"]) for row in rows),
        "maximum clips:", max(row["total_clip_cells"] for row in rows),
    )


if __name__ == "__main__":
    main()
