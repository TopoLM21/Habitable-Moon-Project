"""Static maps, histories and GIF frames for v0.28 plume magmatism."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tectonics.plume_magmatism import (
    magmatic_topography_fields,
    total_igneous_volume_field,
)

from .raster import rasterize_cells
from .topology import build_gif


def _map(mesh, values, path: Path, title: str, label: str, cmap: str, dpi: int,
         *, vmin=None, vmax=None) -> None:
    fig = plt.figure(figsize=(12, 6.8))
    ax = fig.add_subplot(111, projection="mollweide")
    lon, lat, grid = rasterize_cells(mesh, np.asarray(values, dtype=float))
    image = ax.pcolormesh(
        lon, lat, grid, cmap=cmap, shading="auto", rasterized=True,
        vmin=vmin, vmax=vmax,
    )
    fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.08,
                 fraction=0.05, label=label)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_plume_magmatism_maps(mesh, state, radius_km: float, params,
                              output: Path, dpi: int = 180) -> None:
    output.mkdir(parents=True, exist_ok=True)
    areas = mesh.physical_cell_areas_km2(float(radius_km))
    extrusive_h = np.maximum(state.extrusive_volume_km3, 0.0) / np.maximum(areas, 1e-30)
    dyke_h = np.maximum(state.dyke_volume_km3, 0.0) / np.maximum(areas, 1e-30)
    underplate_h = np.maximum(state.underplate_volume_km3, 0.0) / np.maximum(areas, 1e-30)
    _, load, intrusive, total_h_m = magmatic_topography_fields(
        mesh, state, radius_km, params
    )
    support = 1000.0 * extrusive_h + load + intrusive
    age = np.asarray(state.track_age_myr, dtype=float).copy()
    age[total_igneous_volume_field(state) <= params.mapped_track_volume_km3] = np.nan
    time = float(state.time_myr)
    _map(mesh, total_h_m / 1000.0,
         output / "plume_igneous_thickness_final.png",
         f"Permanent plume-derived igneous crust — t={time:g} Myr",
         "Total added igneous thickness, km", "magma", dpi, vmin=0.0)
    _map(mesh, extrusive_h,
         output / "plume_extrusive_basalt_final.png",
         f"Extrusive plume basalt — t={time:g} Myr",
         "Extrusive basalt thickness, km", "inferno", dpi, vmin=0.0)
    _map(mesh, dyke_h + underplate_h,
         output / "plume_intrusive_crust_final.png",
         f"Dykes, sills and underplate — t={time:g} Myr",
         "Intrusive igneous thickness, km", "copper", dpi, vmin=0.0)
    _map(mesh, age, output / "plume_track_age_final.png",
         f"Transported age since last plume emplacement — t={time:g} Myr",
         "Track age, Myr", "viridis_r", dpi, vmin=0.0)
    _map(mesh, support, output / "plume_magmatic_isostatic_support_final.png",
         f"Density-aware magmatic isostatic support — t={time:g} Myr",
         "Local Airy-limit support, m", "plasma", dpi, vmin=0.0)


def save_plume_magmatism_history(rows, path: Path, dpi: int = 160) -> None:
    if not rows:
        return
    time = np.asarray([row["time_myr"] for row in rows], dtype=float)
    million = 1.0e6
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(time, [r["surface_extrusive_volume_km3"] / million for r in rows],
                 label="extrusive basalt")
    axes[0].plot(time, [r["surface_dyke_volume_km3"] / million for r in rows],
                 label="dykes and sills")
    axes[0].plot(time, [r["surface_underplate_volume_km3"] / million for r in rows],
                 label="underplate")
    axes[0].set_ylabel("Surface reservoir, million km³")
    axes[0].legend()
    axes[1].plot(time, [r["cumulative_generated_total_volume_km3"] / million for r in rows],
                 label="cumulative generated")
    axes[1].plot(time, [r["deep_recycled_total_volume_km3"] / million for r in rows],
                 label="deep recycled")
    axes[1].plot(time, [r["global_igneous_ledger_error_km3"] for r in rows],
                 label="ledger error, km³")
    axes[1].set_ylabel("Volume, million km³ / km³")
    axes[1].legend()
    axes[2].plot(time, [r["maximum_igneous_thickness_km"] for r in rows],
                 label="maximum igneous thickness, km")
    axes[2].plot(time, [r["maximum_density_aware_support_m"] / 1000.0 for r in rows],
                 label="maximum isostatic support, km")
    axes[2].plot(time, [r["mapped_track_area_fraction"] for r in rows],
                 label="mapped track area fraction")
    axes[2].set_xlabel("Time, Myr")
    axes[2].set_ylabel("Thickness / support / fraction")
    axes[2].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("v0.28 permanent plume-magmatic crust")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_plume_magmatism_frame(mesh, lithosphere, topography, plume_state,
                               state, radius_km: float, plate_count: int,
                               path: Path, dpi: int = 105) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    areas = mesh.physical_cell_areas_km2(float(radius_km))
    thickness = total_igneous_volume_field(state) / np.maximum(areas, 1e-30)
    age = np.asarray(state.track_age_myr, dtype=float).copy()
    age[thickness <= 0.0] = np.nan
    fig = plt.figure(figsize=(20, 5.8))
    axes = [fig.add_subplot(1, 4, i, projection="mollweide") for i in range(1, 5)]
    lon, lat, plates = rasterize_cells(mesh, lithosphere.cell_plate)
    axes[0].pcolormesh(lon, lat, plates, cmap="tab20", shading="auto", rasterized=True)
    axes[0].set_title(f"Plates: {plate_count}")
    _, _, productivity = rasterize_cells(mesh, state.last_emplacement_productivity)
    image = axes[1].pcolormesh(lon, lat, productivity, cmap="magma", shading="auto",
                               rasterized=True, vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=axes[1], orientation="horizontal", pad=0.08, fraction=0.05)
    if len(plume_state.centers_unit):
        centers = np.asarray(plume_state.centers_unit, dtype=float)
        axes[1].scatter(np.arctan2(centers[:, 1], centers[:, 0]),
                        np.arcsin(np.clip(centers[:, 2], -1.0, 1.0)),
                        marker="*", s=55, c="cyan", edgecolors="black", linewidths=0.5)
    axes[1].set_title("Current emplacement productivity")
    _, _, igneous = rasterize_cells(mesh, thickness)
    image = axes[2].pcolormesh(lon, lat, igneous, cmap="inferno", shading="auto",
                               rasterized=True, vmin=0.0)
    fig.colorbar(image, ax=axes[2], orientation="horizontal", pad=0.08, fraction=0.05)
    axes[2].set_title("Permanent igneous thickness, km")
    _, _, elevation = rasterize_cells(mesh, topography.elevation_m)
    image = axes[3].pcolormesh(lon, lat, elevation, cmap="terrain", shading="auto",
                               rasterized=True, vmin=-7000.0, vmax=3500.0)
    fig.colorbar(image, ax=axes[3], orientation="horizontal", pad=0.08, fraction=0.05)
    axes[3].set_title("Total surface elevation, m")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Moon tectonics v0.28 — permanent plume track — "
                 f"t={lithosphere.time_myr:g} Myr")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_plume_magmatism_gif(frame_paths, path: Path,
                              frame_duration_ms: int = 350) -> None:
    if len(frame_paths) >= 2:
        build_gif(frame_paths, path, frame_duration_ms)


__all__ = [
    "save_plume_magmatism_maps",
    "save_plume_magmatism_history",
    "save_plume_magmatism_frame",
    "build_plume_magmatism_gif",
]
