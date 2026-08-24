"""Maps and history plots for v0.24 cratonic-lithosphere memory."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tectonics.lithosphere import continental_material_fields
from .raster import rasterize_cells


def _map(mesh, data, path: Path, title: str, label: str, cmap: str, dpi: int, vmin=None, vmax=None):
    fig = plt.figure(figsize=(12, 6.8))
    ax = fig.add_subplot(111, projection="mollweide")
    lon_edges, lat_edges, grid = rasterize_cells(mesh, np.asarray(data, dtype=float))
    image = ax.pcolormesh(
        lon_edges,
        lat_edges,
        grid,
        cmap=cmap,
        shading="auto",
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.08, fraction=0.05, label=label)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_craton_maps(mesh, state, radius_km: float, out: Path, dpi: int = 180) -> None:
    if (
        state.continental_lithosphere_age_myr is None
        or state.mantle_depletion_fraction is None
        or state.craton_strength is None
    ):
        return
    out.mkdir(parents=True, exist_ok=True)
    areas = mesh.physical_cell_areas_km2(float(radius_km))
    fraction, _ = continental_material_fields(state, areas)
    present = fraction > 0.01
    age = np.where(present, np.asarray(state.continental_lithosphere_age_myr, dtype=float), np.nan)
    depletion = np.where(present, np.asarray(state.mantle_depletion_fraction, dtype=float), np.nan)
    strength = np.where(present, np.asarray(state.craton_strength, dtype=float), np.nan)
    time = float(state.time_myr)
    _map(
        mesh,
        age,
        out / "continental_lithosphere_age_final.png",
        f"Continental lithosphere effective age — t={time:g} Myr",
        "Effective age, Myr",
        "plasma",
        dpi,
        0.0,
    )
    _map(
        mesh,
        depletion,
        out / "mantle_depletion_fraction_final.png",
        f"Continental mantle-root depletion — t={time:g} Myr",
        "Depletion fraction",
        "viridis",
        dpi,
        0.0,
        0.72,
    )
    _map(
        mesh,
        strength,
        out / "craton_strength_final.png",
        f"Cratonic strength memory — t={time:g} Myr",
        "Craton strength",
        "magma",
        dpi,
        0.0,
        1.0,
    )


def save_craton_history(rows, path: Path, dpi: int = 160) -> None:
    if not rows:
        return
    time = np.asarray([row["time_myr"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, [row["mean_craton_strength"] for row in rows], label="mean strength")
    ax.plot(time, [row["mean_mantle_depletion_fraction"] for row in rows], label="mean depletion")
    ax.plot(
        time,
        [row["cratonic_fraction_of_continental_material"] for row in rows],
        label="cratonic share of continent",
    )
    ax.set_xlabel("Time, Myr")
    ax.set_ylabel("Dimensionless fraction")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("v0.24 continental-lithosphere maturation")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


__all__ = ["save_craton_maps", "save_craton_history"]
