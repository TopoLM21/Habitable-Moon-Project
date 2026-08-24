"""Maps and histories for v0.25 mantle-plume forcing."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .raster import rasterize_cells


def _map(mesh, data, path: Path, title: str, label: str, cmap: str, dpi: int, vmin=0.0, vmax=None):
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


def save_plume_maps(mesh, plume_state, out: Path, dpi: int = 180) -> None:
    out.mkdir(parents=True, exist_ok=True)
    time = float(plume_state.time_myr)
    _map(
        mesh,
        plume_state.last_flux,
        out / "mantle_plume_flux_final.png",
        f"Mantle-plume forcing — t={time:g} Myr",
        "Normalized plume flux",
        "inferno",
        dpi,
        0.0,
        1.0,
    )
    _map(
        mesh,
        plume_state.cumulative_exposure_myr,
        out / "mantle_plume_exposure_final.png",
        f"Cumulative mantle-plume exposure — t={time:g} Myr",
        "Flux-weighted exposure, Myr",
        "magma",
        dpi,
    )
    _map(
        mesh,
        plume_state.cumulative_root_erosion_km,
        out / "mantle_plume_root_erosion_final.png",
        f"Cumulative imposed continental-root erosion — t={time:g} Myr",
        "Root erosion accumulated at cell, km",
        "cividis",
        dpi,
    )


def save_plume_history(rows, path: Path, dpi: int = 160) -> None:
    if not rows:
        return
    time = np.asarray([row["time_myr"] for row in rows], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(time, [row["active_plume_count"] for row in rows], label="active plumes")
    axes[0].plot(time, [row["max_surface_flux"] for row in rows], label="maximum flux")
    axes[0].plot(time, [row["affected_surface_area_fraction"] for row in rows], label="affected surface")
    axes[0].set_ylabel("Count / fraction")
    axes[0].legend()

    axes[1].plot(time, [row["mean_continental_age_loss_myr"] for row in rows], label="age loss, Myr")
    axes[1].plot(time, [row["mean_craton_strength_loss"] for row in rows], label="strength loss")
    axes[1].plot(time, [row["mean_continental_depletion_loss"] for row in rows], label="depletion loss")
    axes[1].set_ylabel("Step response")
    axes[1].legend()

    axes[2].plot(time, [row["mean_root_erosion_this_step_km"] for row in rows], label="mean root erosion")
    axes[2].plot(time, [row["max_root_erosion_this_step_km"] for row in rows], label="maximum root erosion")
    axes[2].plot(time, [row["cumulative_mean_surface_exposure_myr"] for row in rows], label="mean cumulative exposure")
    axes[2].set_xlabel("Time, Myr")
    axes[2].set_ylabel("km / Myr")
    axes[2].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("v0.25 mantle-plume forcing and craton response")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


__all__ = ["save_plume_maps", "save_plume_history"]
