#!/usr/bin/env python3
"""Summarize fixed/mobile plume-source validation results."""

from __future__ import annotations

import argparse
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
MODES = ("fixed", "mobile")


def _read_last_csv(path: Path) -> dict:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row(seed: str, mode: str, subdivision: int, end_time: float) -> dict:
    label = f"{end_time:g}".replace(".", "p")
    output = RESULTS / f"v130_validation_{mode}_sub{subdivision}_{label}_seed_{seed}"
    with (output / "summary_v130.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    drift = _read_last_csv(output / "plume_drift_history.csv")
    plume = _read_last_csv(output / "mantle_plume_history.csv")
    hotspot_rows = _read_csv(output / "hotspot_track_history.csv")
    hotspot = hotspot_rows[-1]
    magmatic = _read_last_csv(output / "plume_magmatism_history.csv")
    return {
        "seed": int(seed),
        "mode": mode,
        "subdivision": int(subdivision),
        "generated_igneous_volume_km3": float(magmatic["cumulative_generated_total_volume_km3"]),
        "surface_igneous_volume_km3": sum(
            float(magmatic[name])
            for name in (
                "surface_extrusive_volume_km3",
                "surface_dyke_volume_km3",
                "surface_underplate_volume_km3",
            )
        ),
        "igneous_ledger_error_km3": float(magmatic["global_igneous_ledger_error_km3"]),
        "continental_ledger_error_km3": float(summary["final_global_continental_ledger_error_km3"]),
        "max_rift_extension": float(summary["final_max_rift_extension"]),
        "maximum_elevation_m": float(summary["max_elevation_m"]),
        "land_fraction": float(summary["final_land_area_fraction"]),
        "plate_count": int(summary["final_plate_count"]),
        "age_distance_correlation": float(hotspot["hotspot_track_age_distance_correlation"]),
        "head_productivity_time_integral_myr": sum(
            float(item["mean_head_productivity"]) * float(item["dt_myr"])
            for item in hotspot_rows
        ),
        "tail_productivity_time_integral_myr": sum(
            float(item["mean_tail_productivity"]) * float(item["dt_myr"])
            for item in hotspot_rows
        ),
        "population_source_path_length_km": float(drift["population_source_path_length_km"]),
        "population_source_bend_angle_deg": float(drift["population_source_bend_angle_deg"]),
        "mean_source_speed_km_per_myr": float(drift["mean_source_speed_km_per_myr"]),
        "mean_plate_speed_km_per_myr": float(drift["mean_overlying_plate_speed_km_per_myr"]),
        "mean_relative_track_speed_km_per_myr": float(drift["mean_relative_track_speed_km_per_myr"]),
        "mean_source_motion_deflection_deg": float(drift["mean_source_motion_deflection_deg"]),
        "total_clip_cells": int(summary["total_numerical_min_clip_cells"]) + int(summary["total_numerical_max_clip_cells"]),
        "active_plumes": int(float(plume["active_plume_count"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-time", type=float, default=500.0)
    args = parser.parse_args()
    rows = [_row(seed, mode, 3, args.end_time) for seed in SEEDS for mode in MODES]
    rows.extend(_row("20260806", mode, 4, args.end_time) for mode in MODES)
    csv_path = ROOT / "V130_MOBILE_PLUME_VALIDATION_500MYR.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    sub3 = [row for row in rows if row["subdivision"] == 3]
    pairs = {
        seed: {row["mode"]: row for row in sub3 if row["seed"] == int(seed)}
        for seed in SEEDS
    }
    labels = ["Source path\n(1000 km)", "Source bend\n(deg)", "Track deflection\n(deg)", "Age-distance\nr"]
    fixed = np.asarray([
        np.mean([pair["fixed"]["population_source_path_length_km"] for pair in pairs.values()]) / 1000.0,
        np.mean([pair["fixed"]["population_source_bend_angle_deg"] for pair in pairs.values()]),
        np.mean([pair["fixed"]["mean_source_motion_deflection_deg"] for pair in pairs.values()]),
        np.mean([pair["fixed"]["age_distance_correlation"] for pair in pairs.values()]),
    ])
    mobile = np.asarray([
        np.mean([pair["mobile"]["population_source_path_length_km"] for pair in pairs.values()]) / 1000.0,
        np.mean([pair["mobile"]["population_source_bend_angle_deg"] for pair in pairs.values()]),
        np.mean([pair["mobile"]["mean_source_motion_deflection_deg"] for pair in pairs.values()]),
        np.mean([pair["mobile"]["age_distance_correlation"] for pair in pairs.values()]),
    ])
    # Separate axes keep the differently scaled physical diagnostics legible.
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.8))
    for i, ax in enumerate(axes):
        ax.bar([0, 1], [fixed[i], mobile[i]], color=["#777777", "#d95f02"])
        ax.set_xticks([0, 1], ["fixed", "mobile"])
        ax.set_title(labels[i])
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("v0.30 paired 500-Myr source-mobility validation (subdivision 3 means)")
    fig.tight_layout()
    fig.savefig(ROOT / "V130_MOBILE_PLUME_VALIDATION_500MYR.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(csv_path)
    metrics = (
        "generated_igneous_volume_km3",
        "surface_igneous_volume_km3",
        "max_rift_extension",
        "maximum_elevation_m",
        "land_fraction",
        "plate_count",
        "age_distance_correlation",
        "population_source_path_length_km",
        "population_source_bend_angle_deg",
        "mean_source_speed_km_per_myr",
        "mean_plate_speed_km_per_myr",
        "mean_relative_track_speed_km_per_myr",
        "mean_source_motion_deflection_deg",
    )
    print("metric,fixed_mean,mobile_mean,paired_mobile_minus_fixed")
    for metric in metrics:
        fixed_values = [pair["fixed"][metric] for pair in pairs.values()]
        mobile_values = [pair["mobile"][metric] for pair in pairs.values()]
        deltas = [mobile - fixed for fixed, mobile in zip(fixed_values, mobile_values)]
        print(
            f"{metric},{np.mean(fixed_values):.10g},"
            f"{np.mean(mobile_values):.10g},{np.mean(deltas):.10g}"
        )
    print(
        "maximum absolute ledgers:",
        max(abs(row["igneous_ledger_error_km3"]) for row in rows),
        max(abs(row["continental_ledger_error_km3"]) for row in rows),
        "maximum clips:",
        max(row["total_clip_cells"] for row in rows),
    )


if __name__ == "__main__":
    main()
