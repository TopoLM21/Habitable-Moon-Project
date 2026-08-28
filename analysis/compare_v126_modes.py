#!/usr/bin/env python3
"""Summarize the four-mode, four-seed v0.26 500-Myr validation ensemble."""

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
MODES = ("control", "weakening_only", "forcing_only", "combined")
LABELS = {
    "control": "Control",
    "weakening_only": "Weakening only",
    "forcing_only": "Forcing only",
    "combined": "Combined",
}
COLORS = {
    "control": "#607d8b",
    "weakening_only": "#8e44ad",
    "forcing_only": "#e67e22",
    "combined": "#c0392b",
}


def _directory(seed: str, mode: str) -> Path:
    return RESULTS / f"v126_{mode}_sub3_500_seed_{seed}"


def _load(seed: str, mode: str) -> dict[str, float | int | str | bool]:
    root = _directory(seed, mode)
    summary = json.loads((root / "summary_v126.json").read_text(encoding="utf-8"))
    with (root / "plume_rifting_history.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rift_history = list(csv.DictReader(handle))
    checkpoint = root / "checkpoints" / "checkpoint_0500_Myr" / "state.npz"
    with np.load(checkpoint, allow_pickle=False) as state:
        rift_extension = np.asarray(state["rift_extension"], dtype=np.float64)
        root_erosion = (
            np.asarray(state["plume_cumulative_root_erosion_km"], dtype=np.float64)
            if "plume_cumulative_root_erosion_km" in state.files
            else np.zeros_like(rift_extension)
        )
    weakening = mode in {"weakening_only", "combined"}
    forcing = mode in {"forcing_only", "combined"}
    return {
        "seed": seed,
        "mode": mode,
        "weakening_enabled": weakening,
        "mechanical_forcing_enabled": forcing,
        "final_plate_count": int(summary["final_plate_count"]),
        "topology_event_count": int(summary["topology_event_count"]),
        "final_max_rift_extension": float(summary["final_max_rift_extension"]),
        "final_mean_rift_extension": float(np.mean(rift_extension)),
        "final_rifted_cell_fraction_ge_0p5": float(np.mean(rift_extension >= 0.5)),
        "cumulative_breakup_area_km2": float(summary["cumulative_breakup_area_km2"]),
        "cumulative_continental_thinning_volume_km3": float(
            summary["cumulative_continental_thinning_volume_km3"]
        ),
        "final_continental_area_fraction": float(
            summary["final_continental_area_fraction"]
        ),
        "final_mean_continental_root_km": float(
            summary["final_mean_continental_mantle_lithosphere_thickness_km"]
        ),
        "final_mean_craton_strength": float(summary["final_mean_craton_strength"]),
        "final_cratonic_continent_fraction": float(
            summary["final_cratonic_fraction_of_continental_material"]
        ),
        "maximum_plume_extension_forcing_over_run": max(
            float(item["max_surface_extension_forcing"]) for item in rift_history
        ),
        "cumulative_mean_plume_extension_impulse_myr": float(
            summary["cumulative_mean_plume_extension_impulse_myr"]
        ),
        "cumulative_max_plume_extension_impulse_myr": float(
            summary["cumulative_max_plume_extension_impulse_myr"]
        ),
        "maximum_local_cumulative_root_erosion_km": float(np.max(root_erosion)),
        "numerical_clip_cells": int(summary["total_numerical_min_clip_cells"])
        + int(summary["total_numerical_max_clip_cells"]),
        "global_continental_ledger_error_km3": float(
            summary["final_global_continental_ledger_error_km3"]
        ),
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    numeric = [
        key
        for key, value in rows[0].items()
        if key not in {"seed", "mode", "weakening_enabled", "mechanical_forcing_enabled"}
        and isinstance(value, (int, float))
    ]
    aggregates = []
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        for statistic, function in (("ensemble_mean", np.mean), ("ensemble_std", np.std)):
            row = {
                "seed": statistic,
                "mode": mode,
                "weakening_enabled": mode in {"weakening_only", "combined"},
                "mechanical_forcing_enabled": mode in {"forcing_only", "combined"},
            }
            for key in numeric:
                row[key] = float(function([float(item[key]) for item in selected]))
            aggregates.append(row)
    return aggregates


def _write_csv(rows: list[dict], path: Path) -> None:
    all_rows = rows + _aggregate(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)


def _panel(ax, rows, key, title, ylabel, scale=1.0):
    values = []
    errors = []
    for mode in MODES:
        selected = np.asarray(
            [float(row[key]) for row in rows if row["mode"] == mode], dtype=float
        )
        values.append(scale * float(np.mean(selected)))
        errors.append(scale * float(np.std(selected)))
    x = np.arange(len(MODES), dtype=float)
    ax.bar(
        x,
        values,
        yerr=np.vstack((np.minimum(errors, values), errors)),
        capsize=4,
        color=[COLORS[mode] for mode in MODES],
    )
    ax.set_xticks(x, [LABELS[mode].replace(" ", "\n") for mode in MODES])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def _write_figure(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))
    _panel(
        axes[0, 0], rows, "final_max_rift_extension",
        "Final maximum rift extension", "Dimensionless",
    )
    _panel(
        axes[0, 1], rows, "cumulative_breakup_area_km2",
        "Cumulative continental breakup", "million km²", 1.0e-6,
    )
    _panel(
        axes[0, 2], rows, "cumulative_continental_thinning_volume_km3",
        "Cumulative continental thinning", "million km³", 1.0e-6,
    )
    _panel(
        axes[1, 0], rows, "final_plate_count",
        "Final plate count", "Plates",
    )
    _panel(
        axes[1, 1], rows, "final_mean_continental_root_km",
        "Final mean continental root", "km",
    )
    _panel(
        axes[1, 2], rows, "final_mean_craton_strength",
        "Final mean craton strength", "Dimensionless",
    )
    fig.suptitle(
        "Moon Tectonics v0.26 — four-mode, four-seed 500-Myr ensemble",
        y=0.99,
    )
    fig.text(
        0.5,
        0.012,
        "Bars are ensemble means; error bars are population standard deviations. Each seed shares identical initial geometry across modes.",
        ha="center",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = [_load(seed, mode) for seed in SEEDS for mode in MODES]
    _write_csv(rows, ROOT / "V126_PLUME_RIFTING_4MODE_500MYR.csv")
    _write_figure(rows, ROOT / "V126_PLUME_RIFTING_4MODE_500MYR.png")


if __name__ == "__main__":
    main()
