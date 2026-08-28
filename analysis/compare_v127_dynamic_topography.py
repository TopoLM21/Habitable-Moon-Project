#!/usr/bin/env python3
"""Summarize four paired 500-Myr v0.27 dynamic-topography runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SEEDS = ("20260806", "20260807", "20260808", "20260809")
METRICS = (
    "max_elevation_m",
    "min_elevation_m",
    "final_sea_level_m",
    "final_land_area_fraction",
    "final_shallow_sea_area_fraction",
    "final_exposed_continental_material_area_fraction",
    "final_submerged_continental_material_area_fraction",
    "final_mean_ocean_depth_m",
    "surface_sediment_volume_km3",
    "cumulative_eroded_bedrock_volume_km3",
    "cumulative_reworked_sediment_volume_km3",
    "final_plate_count",
    "topology_event_count",
    "final_max_rift_extension",
    "cumulative_breakup_area_km2",
    "final_continental_volume_km3",
)


def _summary(seed: str, mode: str) -> dict:
    path = (
        RESULTS
        / f"v127_{mode}_sub3_500_seed_{seed}"
        / "summary_v127.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _load(seed: str) -> dict[str, float | int | str]:
    control = _summary(seed, "control")
    dynamic = _summary(seed, "dynamic")
    row: dict[str, float | int | str] = {"seed": seed}
    for key in METRICS:
        row[f"control_{key}"] = float(control[key])
        row[f"dynamic_{key}"] = float(dynamic[key])
        row[f"delta_{key}"] = float(dynamic[key]) - float(control[key])
    row.update(
        {
            "maximum_dynamic_uplift_over_run_m": float(
                dynamic["maximum_plume_dynamic_uplift_over_run_m"]
            ),
            "final_rms_dynamic_topography_m": float(
                dynamic["final_rms_plume_dynamic_topography_m"]
            ),
            "final_dynamic_uplift_area_fraction": float(
                dynamic["final_plume_dynamic_uplift_area_fraction"]
            ),
            "maximum_abs_dynamic_displacement_volume_km3": float(
                dynamic["maximum_abs_dynamic_displacement_volume_km3"]
            ),
            "dynamic_numerical_clip_cells": int(
                dynamic["total_numerical_min_clip_cells"]
            )
            + int(dynamic["total_numerical_max_clip_cells"]),
            "dynamic_global_continental_ledger_error_km3": float(
                dynamic["final_global_continental_ledger_error_km3"]
            ),
        }
    )
    return row


def _write_csv(rows: list[dict], path: Path) -> None:
    numeric = [key for key in rows[0] if key != "seed"]
    mean_row = {"seed": "ensemble_mean"}
    std_row = {"seed": "ensemble_std"}
    for key in numeric:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        mean_row[key] = float(np.mean(values))
        std_row[key] = float(np.std(values))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(mean_row)
        writer.writerow(std_row)


def _paired_panel(ax, rows, key, title, ylabel, scale=1.0):
    x = np.arange(len(rows), dtype=float)
    width = 0.36
    control = scale * np.asarray(
        [float(row[f"control_{key}"]) for row in rows]
    )
    dynamic = scale * np.asarray(
        [float(row[f"dynamic_{key}"]) for row in rows]
    )
    ax.bar(x - width / 2, control, width, label="No dynamic topography", color="#607d8b")
    ax.bar(x + width / 2, dynamic, width, label="Transient plume support", color="#c0392b")
    ax.set_xticks(x, [str(row["seed"])[-2:] for row in rows])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def _write_figure(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    _paired_panel(
        axes[0, 0], rows, "max_elevation_m",
        "Maximum surface elevation", "m",
    )
    _paired_panel(
        axes[0, 1], rows, "final_land_area_fraction",
        "Final land area", "% of surface", 100.0,
    )
    _paired_panel(
        axes[0, 2], rows, "final_sea_level_m",
        "Final solved sea level", "m",
    )
    _paired_panel(
        axes[1, 0], rows, "cumulative_eroded_bedrock_volume_km3",
        "Cumulative bedrock erosion", "million km³", 1.0e-6,
    )
    _paired_panel(
        axes[1, 1], rows, "surface_sediment_volume_km3",
        "Final surface sediment", "million km³", 1.0e-6,
    )
    _paired_panel(
        axes[1, 2], rows, "final_plate_count",
        "Final plate count after coupled evolution", "plates",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Moon Tectonics v0.27 — paired 500-Myr transient dynamic topography",
        y=0.992,
    )
    fig.text(
        0.5,
        0.012,
        "Seed labels show the final two digits; each pair shares identical plume chronology and initial plate geometry.",
        ha="center",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.91))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = [_load(seed) for seed in SEEDS]
    _write_csv(rows, ROOT / "V127_DYNAMIC_TOPOGRAPHY_PAIRED_500MYR.csv")
    _write_figure(rows, ROOT / "V127_DYNAMIC_TOPOGRAPHY_PAIRED_500MYR.png")


if __name__ == "__main__":
    main()
