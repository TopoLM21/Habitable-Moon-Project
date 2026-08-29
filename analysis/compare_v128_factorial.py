#!/usr/bin/env python3
"""Summarize completed v0.28 2x2x2 factorial integrations."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODES = [f"d{d}m{m}r{r}" for d, m, r in product((0, 1), repeat=3)]


def _output(seed: str, mode: str, end_time: float) -> Path:
    label = f"{end_time:g}".replace(".", "p")
    return RESULTS / f"v128_factorial_{mode}_sub3_{label}_seed_{seed}"


def _row(seed: str, mode: str, end_time: float) -> dict:
    summary = json.loads(
        (_output(seed, mode, end_time) / "summary_v128.json").read_text(
            encoding="utf-8"
        )
    )
    dynamic, magmatism, rifting = (bool(int(mode[i])) for i in (1, 3, 5))
    return {
        "seed": seed,
        "mode": mode,
        "dynamic_topography": dynamic,
        "magmatism": magmatism,
        "mechanical_rifting": rifting,
        "final_plate_count": int(summary["final_plate_count"]),
        "final_max_rift_extension": float(summary["final_max_rift_extension"]),
        "maximum_elevation_m": float(summary["max_elevation_m"]),
        "land_area_fraction": float(summary["final_land_area_fraction"]),
        "sea_level_m": float(summary["final_sea_level_m"]),
        "surface_igneous_volume_km3": float(summary.get("final_surface_plume_igneous_volume_km3", 0.0)),
        "generated_igneous_volume_km3": float(summary.get("cumulative_generated_plume_igneous_volume_km3", 0.0)),
        "deep_recycled_igneous_volume_km3": float(summary.get("deep_recycled_plume_igneous_volume_km3", 0.0)),
        "maximum_igneous_thickness_km": float(summary.get("final_maximum_plume_igneous_thickness_km", 0.0)),
        "maximum_magmatic_support_m": float(summary.get("final_maximum_magmatic_isostatic_support_m", 0.0)),
        "igneous_ledger_error_km3": float(summary.get("final_igneous_ledger_error_km3", 0.0)),
        "cumulative_bedrock_erosion_km3": float(summary["cumulative_eroded_bedrock_volume_km3"]),
        "surface_sediment_volume_km3": float(summary["surface_sediment_volume_km3"]),
        "minimum_clip_cells": int(summary["total_numerical_min_clip_cells"]),
        "maximum_clip_cells": int(summary["total_numerical_max_clip_cells"]),
    }


def _main_effect(rows, factor: str, metric: str) -> float:
    on = [float(row[metric]) for row in rows if row[factor]]
    off = [float(row[metric]) for row in rows if not row[factor]]
    return float(np.mean(on) - np.mean(off))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-time", type=float, default=500.0)
    parser.add_argument("--seeds", nargs="+", default=["20260806"])
    parser.add_argument("--output-csv", type=Path,
                        default=ROOT / "V128_PLUME_MAGMATISM_FACTORIAL_500MYR.csv")
    parser.add_argument("--output-plot", type=Path,
                        default=ROOT / "V128_PLUME_MAGMATISM_FACTORIAL_500MYR.png")
    args = parser.parse_args()
    rows = [_row(seed, mode, args.end_time) for seed in args.seeds for mode in MODES]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["mode"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].bar(x, [row["surface_igneous_volume_km3"] / 1e6 for row in rows])
    axes[0, 0].set_ylabel("Surface igneous volume, million km³")
    axes[0, 1].bar(x, [row["maximum_igneous_thickness_km"] for row in rows])
    axes[0, 1].set_ylabel("Maximum igneous thickness, km")
    axes[1, 0].bar(x, [row["maximum_elevation_m"] for row in rows])
    axes[1, 0].set_ylabel("Maximum elevation, m")
    axes[1, 1].bar(x, [100.0 * row["land_area_fraction"] for row in rows])
    axes[1, 1].set_ylabel("Land area, % surface")
    for ax in axes.flat:
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Moon Tectonics v0.28 — 2×2×2 plume-process factorial")
    fig.tight_layout()
    fig.savefig(args.output_plot, dpi=180, bbox_inches="tight")
    plt.close(fig)

    for metric in (
        "maximum_elevation_m", "land_area_fraction", "final_plate_count",
        "surface_igneous_volume_km3", "maximum_igneous_thickness_km",
    ):
        effects = {
            factor: _main_effect(rows, factor, metric)
            for factor in ("dynamic_topography", "magmatism", "mechanical_rifting")
        }
        print(metric, json.dumps(effects, sort_keys=True))


if __name__ == "__main__":
    main()
