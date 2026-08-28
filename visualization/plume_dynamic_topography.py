"""Maps, histories and GIF frames for v0.27 plume dynamic topography."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .raster import rasterize_cells
from .topology import build_gif


def _mollweide_map(
    mesh,
    values,
    path: Path,
    title: str,
    colorbar_label: str,
    cmap: str,
    dpi: int,
    *,
    vmin=None,
    vmax=None,
) -> None:
    fig = plt.figure(figsize=(12, 6.8))
    ax = fig.add_subplot(111, projection="mollweide")
    lon_edges, lat_edges, grid = rasterize_cells(
        mesh, np.asarray(values, dtype=float)
    )
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
    fig.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        pad=0.08,
        fraction=0.05,
        label=colorbar_label,
    )
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_plume_dynamic_topography_maps(
    mesh, state, output: Path, dpi: int = 180
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    time = float(state.time_myr)
    limit = max(
        100.0,
        float(np.max(np.abs(state.target_dynamic_topography_m))),
        float(np.max(np.abs(state.realized_dynamic_topography_m))),
    )
    _mollweide_map(
        mesh,
        state.target_dynamic_topography_m,
        output / "plume_dynamic_topography_target_final.png",
        f"Instantaneous plume dynamic-topography target — t={time:g} Myr",
        "Dynamic-topography target, m",
        "coolwarm",
        dpi,
        vmin=-limit,
        vmax=limit,
    )
    _mollweide_map(
        mesh,
        state.realized_dynamic_topography_m,
        output / "plume_dynamic_topography_realized_final.png",
        f"Realized transient plume dynamic topography — t={time:g} Myr",
        "Dynamic topography, m",
        "coolwarm",
        dpi,
        vmin=-limit,
        vmax=limit,
    )
    _mollweide_map(
        mesh,
        state.cumulative_positive_support_m_myr,
        output / "plume_dynamic_support_cumulative_final.png",
        f"Cumulative positive plume support — t={time:g} Myr",
        "Integrated positive support, m Myr",
        "inferno",
        dpi,
        vmin=0.0,
    )


def save_plume_dynamic_topography_history(rows, path: Path, dpi: int = 160) -> None:
    if not rows:
        return
    time = np.asarray([row["time_myr"] for row in rows], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(
        time,
        [row["maximum_target_uplift_m"] for row in rows],
        label="instantaneous target",
    )
    axes[0].plot(
        time,
        [row["maximum_realized_uplift_m"] for row in rows],
        label="realized maximum",
    )
    axes[0].plot(
        time,
        [row["minimum_realized_subsidence_m"] for row in rows],
        label="compensating minimum",
    )
    axes[0].set_ylabel("Dynamic topography, m")
    axes[0].legend()

    axes[1].plot(
        time,
        [row["rms_realized_anomaly_m"] for row in rows],
        label="global RMS anomaly",
    )
    axes[1].plot(
        time,
        [row["plume_weighted_mean_uplift_m"] for row in rows],
        label="plume-weighted uplift",
    )
    axes[1].set_ylabel("Relief, m")
    axes[1].legend()

    axes[2].plot(
        time,
        [row["affected_surface_area_fraction"] for row in rows],
        label="surface above uplift threshold",
    )
    axes[2].plot(
        time,
        [row["maximum_absolute_vertical_rate_m_per_myr"] for row in rows],
        label="maximum |vertical rate|",
    )
    axes[2].set_xlabel("Time, Myr")
    axes[2].set_ylabel("Fraction / m Myr$^{-1}$")
    axes[2].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("v0.27 transient plume dynamic topography")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_plume_dynamic_topography_frame(
    mesh,
    lithosphere,
    topography,
    plume_state,
    dynamic_state,
    plate_count: int,
    path: Path,
    dpi: int = 105,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 6.2))
    axes = [
        fig.add_subplot(1, 3, index, projection="mollweide")
        for index in (1, 2, 3)
    ]
    lon_edges, lat_edges, plate_grid = rasterize_cells(
        mesh, lithosphere.cell_plate
    )
    axes[0].pcolormesh(
        lon_edges,
        lat_edges,
        plate_grid,
        cmap="tab20",
        shading="auto",
        rasterized=True,
    )
    axes[0].set_title(f"Plates: {plate_count}")

    _, _, dynamic_grid = rasterize_cells(
        mesh, dynamic_state.realized_dynamic_topography_m
    )
    limit = max(
        100.0,
        float(np.max(np.abs(dynamic_state.realized_dynamic_topography_m))),
    )
    image = axes[1].pcolormesh(
        lon_edges,
        lat_edges,
        dynamic_grid,
        cmap="coolwarm",
        shading="auto",
        rasterized=True,
        vmin=-limit,
        vmax=limit,
    )
    fig.colorbar(image, ax=axes[1], orientation="horizontal", pad=0.08, fraction=0.05)
    if len(plume_state.centers_unit):
        centers = np.asarray(plume_state.centers_unit, dtype=float)
        axes[1].scatter(
            np.arctan2(centers[:, 1], centers[:, 0]),
            np.arcsin(np.clip(centers[:, 2], -1.0, 1.0)),
            marker="*",
            s=65,
            c="yellow",
            edgecolors="black",
            linewidths=0.5,
        )
    axes[1].set_title("Transient dynamic topography, m")

    _, _, elevation_grid = rasterize_cells(mesh, topography.elevation_m)
    surface = axes[2].pcolormesh(
        lon_edges,
        lat_edges,
        elevation_grid,
        cmap="terrain",
        shading="auto",
        rasterized=True,
        vmin=-7000.0,
        vmax=3500.0,
    )
    fig.colorbar(surface, ax=axes[2], orientation="horizontal", pad=0.08, fraction=0.05)
    axes[2].set_title("Total surface elevation, m")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        f"Moon tectonics v0.27 — transient plume-supported relief — "
        f"t={lithosphere.time_myr:g} Myr"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_plume_dynamic_topography_gif(
    frame_paths, path: Path, frame_duration_ms: int = 350
) -> None:
    if len(frame_paths) >= 2:
        build_gif(frame_paths, path, frame_duration_ms)


__all__ = [
    "save_plume_dynamic_topography_maps",
    "save_plume_dynamic_topography_history",
    "save_plume_dynamic_topography_frame",
    "build_plume_dynamic_topography_gif",
]
