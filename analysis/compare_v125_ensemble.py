#!/usr/bin/env python3
"""Summarize the four paired v0.24/v0.25 500-Myr validation runs."""

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


def _directories(seed: str) -> tuple[Path, Path]:
    suffix = "" if seed == "20260806" else f"_seed_{seed}"
    return (
        RESULTS / f"paired_v124_control_sub3_500{suffix}",
        RESULTS / f"paired_v125_plumes_sub3_500{suffix}",
    )


def _load(seed: str) -> dict[str, float | int | str]:
    control_dir, plume_dir = _directories(seed)
    control = json.loads((control_dir / "summary_v124.json").read_text(encoding="utf-8"))
    plume = json.loads((plume_dir / "summary_v125.json").read_text(encoding="utf-8"))
    with (plume_dir / "mantle_plume_history.csv").open(encoding="utf-8", newline="") as handle:
        plume_history = list(csv.DictReader(handle))
    with np.load(
        plume_dir / "checkpoints" / "checkpoint_0500_Myr" / "state.npz",
        allow_pickle=False,
    ) as state:
        max_root_erosion = float(np.max(state["plume_cumulative_root_erosion_km"]))

    row: dict[str, float | int | str] = {
        "seed": seed,
        "control_mean_craton_strength": float(control["final_mean_craton_strength"]),
        "plume_mean_craton_strength": float(plume["final_mean_craton_strength"]),
        "control_cratonic_continent_fraction": float(
            control["final_cratonic_fraction_of_continental_material"]
        ),
        "plume_cratonic_continent_fraction": float(
            plume["final_cratonic_fraction_of_continental_material"]
        ),
        "control_mean_continental_root_km": float(
            control["final_mean_continental_mantle_lithosphere_thickness_km"]
        ),
        "plume_mean_continental_root_km": float(
            plume["final_mean_continental_mantle_lithosphere_thickness_km"]
        ),
        "control_max_rift_extension": float(control["final_max_rift_extension"]),
        "plume_max_rift_extension": float(plume["final_max_rift_extension"]),
        "control_final_plate_count": int(control["final_plate_count"]),
        "plume_final_plate_count": int(plume["final_plate_count"]),
        "mean_surface_plume_exposure_myr": float(
            plume["cumulative_mean_plume_exposure_myr"]
        ),
        "max_surface_plume_exposure_myr": float(
            plume["cumulative_max_plume_exposure_myr"]
        ),
        "max_local_cumulative_root_erosion_km": max_root_erosion,
        "summed_global_mean_continental_age_loss_myr": float(
            sum(float(item["mean_continental_age_loss_myr"]) for item in plume_history)
        ),
        "summed_global_mean_root_erosion_km": float(
            sum(float(item["mean_root_erosion_this_step_km"]) for item in plume_history)
        ),
        "plume_global_ledger_error_km3": float(
            plume["final_global_continental_ledger_error_km3"]
        ),
        "plume_numerical_clip_cells": int(plume["total_numerical_min_clip_cells"])
        + int(plume["total_numerical_max_clip_cells"]),
    }
    for name in (
        "mean_craton_strength",
        "cratonic_continent_fraction",
        "mean_continental_root_km",
        "max_rift_extension",
        "final_plate_count",
    ):
        row[f"delta_{name}"] = float(row[f"plume_{name}"]) - float(
            row[f"control_{name}"]
        )
    return row


def _write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    numeric_keys = [key for key in rows[0] if key != "seed"]
    mean_row: dict[str, float | str] = {"seed": "ensemble_mean"}
    for key in numeric_keys:
        mean_row[key] = float(np.mean([float(row[key]) for row in rows]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(mean_row)


def _paired_panel(ax, rows, control_key, plume_key, title, ylabel, scale=1.0):
    x = np.arange(len(rows), dtype=float)
    width = 0.36
    control = scale * np.asarray([float(row[control_key]) for row in rows])
    plume = scale * np.asarray([float(row[plume_key]) for row in rows])
    ax.bar(x - width / 2, control, width, label="v0.24 control", color="#607d8b")
    ax.bar(x + width / 2, plume, width, label="v0.25 plumes", color="#d35400")
    ax.set_xticks(x, [str(row["seed"])[-2:] for row in rows])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def _write_figure(rows: list[dict[str, float | int | str]], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    _paired_panel(
        axes[0, 0], rows, "control_mean_craton_strength", "plume_mean_craton_strength",
        "Mean craton strength", "Dimensionless",
    )
    _paired_panel(
        axes[0, 1], rows, "control_cratonic_continent_fraction", "plume_cratonic_continent_fraction",
        "Cratonic share of continent", "%", 100.0,
    )
    _paired_panel(
        axes[0, 2], rows, "control_mean_continental_root_km", "plume_mean_continental_root_km",
        "Mean continental root", "km",
    )
    _paired_panel(
        axes[1, 0], rows, "control_max_rift_extension", "plume_max_rift_extension",
        "Maximum final rift extension", "Dimensionless",
    )
    _paired_panel(
        axes[1, 1], rows, "control_final_plate_count", "plume_final_plate_count",
        "Final plate count", "Plates",
    )
    x = np.arange(len(rows), dtype=float)
    axes[1, 2].bar(
        x - 0.18,
        [float(row["max_local_cumulative_root_erosion_km"]) for row in rows],
        0.36,
        label="max local root erosion",
        color="#8e44ad",
    )
    axes[1, 2].bar(
        x + 0.18,
        [float(row["max_surface_plume_exposure_myr"]) / 5.0 for row in rows],
        0.36,
        label="max exposure / 5",
        color="#f1c40f",
    )
    axes[1, 2].set_xticks(x, [str(row["seed"])[-2:] for row in rows])
    axes[1, 2].set_title("Local imposed plume effect")
    axes[1, 2].set_ylabel("km or scaled Myr")
    axes[1, 2].grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=2,
        frameon=False,
    )
    axes[1, 2].legend(fontsize=8)
    fig.suptitle("Moon Tectonics v0.25 — four paired 500-Myr sub3 runs", y=0.992)
    fig.text(0.5, 0.01, "Seed labels show the final two digits; paired runs share identical initial plate geometry.", ha="center")
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.90))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = [_load(seed) for seed in SEEDS]
    _write_csv(rows, ROOT / "V125_PLUME_PAIRED_500MYR.csv")
    _write_figure(rows, ROOT / "V125_PLUME_PAIRED_500MYR.png")


if __name__ == "__main__":
    main()
