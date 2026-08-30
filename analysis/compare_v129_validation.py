#!/usr/bin/env python3
"""Summarize v0.29 paired multi-seed and mesh-convergence integrations."""

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
MODES = ("legacy", "full")


def _output(seed: str, mode: str, subdivision: int, end_time: float) -> Path:
    label = f"{end_time:g}".replace(".", "p")
    return RESULTS / f"v129_validation_{mode}_sub{subdivision}_{label}_seed_{seed}"


def _row(seed: str, mode: str, subdivision: int, end_time: float) -> dict:
    output = _output(seed, mode, subdivision, end_time)
    summary = json.loads(
        (output / "summary_v129.json").read_text(
            encoding="utf-8"
        )
    )
    with (output / "hotspot_track_history.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        track_rows = list(csv.DictReader(handle))
    with (output / "plume_magmatism_history.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        magmatic_rows = list(csv.DictReader(handle))
    track_final = track_rows[-1]
    magmatic_final = magmatic_rows[-1]
    return {
        "seed": seed,
        "mode": mode,
        "subdivision": subdivision,
        "cell_count": 20 * 4 ** subdivision,
        "final_plate_count": int(summary["final_plate_count"]),
        "final_max_rift_extension": float(summary["final_max_rift_extension"]),
        "maximum_elevation_m": float(summary["max_elevation_m"]),
        "land_area_fraction": float(summary["final_land_area_fraction"]),
        "sea_level_m": float(summary["final_sea_level_m"]),
        "surface_igneous_volume_km3": float(summary.get("final_surface_plume_igneous_volume_km3", 0.0)),
        "generated_igneous_volume_km3": float(summary.get("cumulative_generated_plume_igneous_volume_km3", 0.0)),
        "deep_recycled_igneous_volume_km3": float(summary.get("deep_recycled_plume_igneous_volume_km3", 0.0)),
        "maximum_igneous_thickness_km": float(summary.get("final_maximum_plume_igneous_thickness_km", 0.0)),
        "maximum_tail_productivity": float(summary.get("final_maximum_tail_productivity", 0.0)),
        "maximum_thermal_anomaly": float(summary.get("final_maximum_magmatic_thermal_anomaly", 0.0)),
        "maximum_thermal_anomaly_over_run": max(float(row["maximum_thermal_anomaly"]) for row in track_rows),
        "maximum_dike_localization_over_run": max(float(row["maximum_dike_localization"]) for row in track_rows),
        "eclogitized_underplate_volume_km3": float(summary.get("final_eclogitized_underplate_volume_km3", 0.0)),
        "delaminated_underplate_volume_km3": float(summary.get("cumulative_delaminated_underplate_volume_km3", 0.0)),
        "head_generated_volume_km3": float(track_final["cumulative_head_generated_volume_km3"]),
        "tail_generated_volume_km3": float(track_final["cumulative_tail_generated_volume_km3"]),
        "surface_dyke_volume_km3": float(magmatic_final["surface_dyke_volume_km3"]),
        "surface_underplate_volume_km3": float(magmatic_final["surface_underplate_volume_km3"]),
        "age_distance_correlation": float(summary.get("final_hotspot_track_age_distance_correlation", 0.0)),
        "igneous_ledger_error_km3": float(summary.get("final_igneous_ledger_error_km3", 0.0)),
        "continental_ledger_error_km3": float(summary["final_global_continental_ledger_error_km3"]),
        "minimum_clip_cells": int(summary["total_numerical_min_clip_cells"]),
        "maximum_clip_cells": int(summary["total_numerical_max_clip_cells"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-time", type=float, default=500.0)
    parser.add_argument("--seeds", nargs="+", default=["20260806", "20260807", "20260808", "20260809"])
    parser.add_argument("--output-csv", type=Path,
                        default=ROOT / "V129_HOTSPOT_TRACK_VALIDATION_500MYR.csv")
    parser.add_argument("--output-plot", type=Path,
                        default=ROOT / "V129_HOTSPOT_TRACK_VALIDATION_500MYR.png")
    args = parser.parse_args()
    rows = [
        _row(seed, mode, 3, args.end_time)
        for seed in args.seeds for mode in MODES
    ]
    rows.extend(
        _row("20260806", mode, 4, args.end_time) for mode in MODES
    )
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sub3 = [row for row in rows if row["subdivision"] == 3]
    seeds = list(args.seeds)
    legacy = {row["seed"]: row for row in sub3 if row["mode"] == "legacy"}
    full = {row["seed"]: row for row in sub3 if row["mode"] == "full"}
    x = np.arange(len(seeds))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    width = 0.36
    metrics = (
        ("surface_igneous_volume_km3", 1e6, "Surface igneous volume, million km³"),
        ("final_max_rift_extension", 1.0, "Maximum rift extension"),
        ("maximum_elevation_m", 1.0, "Maximum elevation, m"),
        ("land_area_fraction", 0.01, "Land area, % surface"),
    )
    for ax, (metric, scale, ylabel) in zip(axes.flat, metrics):
        ax.bar(x - width / 2, [legacy[s][metric] / scale for s in seeds], width, label="v0.28-compatible")
        ax.bar(x + width / 2, [full[s][metric] / scale for s in seeds], width, label="v0.29 full")
        ax.set_xticks(x, seeds, rotation=30)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
    fig.suptitle("Moon Tectonics v0.29 — paired 500-Myr hotspot-track validation")
    fig.tight_layout()
    fig.savefig(args.output_plot, dpi=180, bbox_inches="tight")
    plt.close(fig)

    for metric in (
        "surface_igneous_volume_km3",
        "final_max_rift_extension",
        "maximum_elevation_m",
        "land_area_fraction",
        "final_plate_count",
    ):
        differences = [float(full[s][metric]) - float(legacy[s][metric]) for s in seeds]
        print(metric, f"mean paired effect={np.mean(differences):+.6g}",
              f"range=[{np.min(differences):+.6g}, {np.max(differences):+.6g}]")


if __name__ == "__main__":
    main()
