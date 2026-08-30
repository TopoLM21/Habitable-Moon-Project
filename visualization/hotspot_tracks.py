"""Maps, histories and GIF frames for v0.29 hotspot tracks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tectonics.plume_magmatism import total_igneous_volume_field

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


def save_hotspot_track_maps(mesh, magmatic_state, state, radius_km: float,
                            output: Path, dpi: int = 180) -> None:
    output.mkdir(parents=True, exist_ok=True)
    areas = mesh.physical_cell_areas_km2(float(radius_km))
    underplate_h = np.maximum(magmatic_state.underplate_volume_km3, 0.0) / np.maximum(areas, 1e-30)
    age = np.asarray(magmatic_state.track_age_myr, dtype=float).copy()
    age[total_igneous_volume_field(magmatic_state) <= 1e-9] = np.nan
    time = float(state.time_myr)
    _map(mesh, age, output / "hotspot_track_age_final.png",
         f"Age-progressive hotspot tracks — t={time:g} Myr",
         "Age since last emplacement, Myr", "viridis_r", dpi, vmin=0.0)
    _map(mesh, state.thermal_anomaly, output / "magmatic_thermal_anomaly_final.png",
         f"Transported magmatic thermal memory — t={time:g} Myr",
         "Normalized thermal anomaly", "inferno", dpi, vmin=0.0, vmax=1.0)
    _map(mesh, state.last_dike_localization, output / "rift_dike_localization_final.png",
         f"Syn-rift dyke localization — t={time:g} Myr",
         "Localization factor", "magma", dpi, vmin=0.0, vmax=1.0)
    _map(mesh, state.underplate_eclogite_fraction,
         output / "underplate_eclogite_fraction_final.png",
         f"Old underplate eclogitization — t={time:g} Myr",
         "Eclogitized fraction", "cividis", dpi, vmin=0.0, vmax=1.0)
    _map(mesh, underplate_h * state.underplate_eclogite_fraction,
         output / "eclogitized_underplate_thickness_final.png",
         f"Dense eclogitized underplate — t={time:g} Myr",
         "Equivalent thickness, km", "copper_r", dpi, vmin=0.0)


def save_hotspot_track_history(rows, path: Path, dpi: int = 160) -> None:
    if not rows:
        return
    time = np.asarray([row["time_myr"] for row in rows], dtype=float)
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(time, [r["maximum_head_productivity"] for r in rows], label="broad head")
    axes[0].plot(time, [r["maximum_tail_productivity"] for r in rows], label="narrow tail")
    axes[0].set_ylabel("Maximum productivity")
    axes[0].legend()
    axes[1].plot(time, [r["maximum_thermal_anomaly"] for r in rows], label="thermal anomaly")
    axes[1].plot(time, [r["maximum_thermal_extension_forcing"] for r in rows], label="extension forcing")
    axes[1].plot(time, [r["maximum_dike_localization"] for r in rows], label="dyke localization")
    axes[1].set_ylabel("Normalized response")
    axes[1].legend()
    axes[2].plot(time, [r["eclogitized_underplate_volume_km3"] / 1e6 for r in rows], label="eclogitized")
    axes[2].plot(time, [r["cumulative_delaminated_underplate_volume_km3"] / 1e6 for r in rows], label="foundered")
    axes[2].set_ylabel("Underplate, million km³")
    axes[2].legend()
    axes[3].plot(time, [r["hotspot_track_age_distance_correlation"] for r in rows], label="age-distance correlation")
    axes[3].axhline(0.0, color="black", linewidth=0.8)
    axes[3].set_ylabel("Correlation")
    axes[3].set_xlabel("Time, Myr")
    axes[3].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("v0.29 plume-head/tail hotspot-track evolution")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_hotspot_track_frame(mesh, lithosphere, topography, plume_state,
                             magmatic_state, state, radius_km: float,
                             plate_count: int, path: Path,
                             dpi: int = 105) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    areas = mesh.physical_cell_areas_km2(float(radius_km))
    thickness = total_igneous_volume_field(magmatic_state) / np.maximum(areas, 1e-30)
    age = np.asarray(magmatic_state.track_age_myr, dtype=float).copy()
    age[thickness <= 1e-12] = np.nan
    current = np.asarray(state.last_head_productivity) + np.asarray(state.last_tail_productivity)
    fig = plt.figure(figsize=(20, 10.5))
    axes = [fig.add_subplot(2, 3, i, projection="mollweide") for i in range(1, 7)]
    lon, lat, _ = rasterize_cells(mesh, lithosphere.cell_plate)

    panels = [
        (lithosphere.cell_plate, "tab20", None, None, f"Plates: {plate_count}"),
        (current, "magma", 0.0, 1.0, "Current head + tail productivity"),
        (thickness, "inferno", 0.0, None, "Permanent igneous thickness, km"),
        (age, "viridis_r", 0.0, None, "Hotspot-track age, Myr"),
        (state.thermal_anomaly, "plasma", 0.0, 1.0, "Transported magmatic heat"),
        (topography.elevation_m, "terrain", -7000.0, 3500.0, "Surface elevation, m"),
    ]
    for ax, (values, cmap, vmin, vmax, title) in zip(axes, panels):
        _, _, grid = rasterize_cells(mesh, values)
        image = ax.pcolormesh(lon, lat, grid, cmap=cmap, shading="auto",
                              rasterized=True, vmin=vmin, vmax=vmax)
        fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.08, fraction=0.05)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    if len(plume_state.centers_unit):
        centers = np.asarray(plume_state.centers_unit, dtype=float)
        axes[1].scatter(np.arctan2(centers[:, 1], centers[:, 0]),
                        np.arcsin(np.clip(centers[:, 2], -1.0, 1.0)),
                        marker="*", s=55, c="cyan", edgecolors="black", linewidths=0.5)
    fig.suptitle(f"Moon tectonics v0.29 — plume heads, tails and hotspot chains — t={lithosphere.time_myr:g} Myr")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_hotspot_track_gif(frame_paths, path: Path,
                            frame_duration_ms: int = 350) -> None:
    if len(frame_paths) >= 2:
        build_gif(frame_paths, path, frame_duration_ms)


__all__ = [
    "save_hotspot_track_maps",
    "save_hotspot_track_history",
    "save_hotspot_track_frame",
    "build_hotspot_track_gif",
]
