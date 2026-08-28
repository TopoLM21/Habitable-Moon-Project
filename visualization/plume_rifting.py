"""Diagnostics and animation frames for v0.26 plume-driven rifting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .raster import rasterize_cells
from .topology import build_gif


def _map(mesh, data, path: Path, title: str, label: str, cmap: str, dpi: int, *, vmin=0.0, vmax=None):
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


def save_plume_rifting_maps(mesh, state, out: Path, dpi: int = 180) -> None:
    out.mkdir(parents=True, exist_ok=True)
    time = float(state.time_myr)
    _map(
        mesh,
        state.last_extension_forcing,
        out / "plume_extension_forcing_final.png",
        f"Plume-driven continental extension — t={time:g} Myr",
        "External extension forcing",
        "magma",
        dpi,
        vmin=0.0,
        vmax=1.0,
    )
    _map(
        mesh,
        state.cumulative_extension_impulse_myr,
        out / "plume_extension_impulse_final.png",
        f"Cumulative plume extension impulse — t={time:g} Myr",
        "Forcing-weighted time, Myr",
        "inferno",
        dpi,
    )
    _map(
        mesh,
        state.last_dynamic_uplift_m,
        out / "plume_dynamic_uplift_final.png",
        f"Diagnostic plume dynamic uplift — t={time:g} Myr",
        "Diagnostic uplift, m",
        "terrain",
        dpi,
    )
    _map(
        mesh,
        state.last_magmatic_productivity,
        out / "plume_magmatic_productivity_final.png",
        f"Diagnostic plume magmatic productivity — t={time:g} Myr",
        "Normalized productivity",
        "plasma",
        dpi,
        vmin=0.0,
        vmax=1.0,
    )


def save_plume_rifting_history(rows, path: Path, dpi: int = 160) -> None:
    if not rows:
        return
    time = np.asarray([row["time_myr"] for row in rows], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(time, [row["max_surface_extension_forcing"] for row in rows], label="maximum")
    axes[0].plot(time, [row["mean_continental_extension_forcing"] for row in rows], label="continental mean")
    axes[0].plot(time, [row["forced_surface_area_fraction"] for row in rows], label="forced surface")
    axes[0].set_ylabel("Forcing / fraction")
    axes[0].legend()

    axes[1].plot(time, [row["max_dynamic_uplift_m"] for row in rows], label="maximum uplift")
    axes[1].plot(time, [row["mean_dynamic_uplift_m"] for row in rows], label="mean uplift")
    axes[1].set_ylabel("Diagnostic uplift, m")
    axes[1].legend()

    axes[2].plot(time, [row["max_magmatic_productivity"] for row in rows], label="maximum productivity")
    axes[2].plot(time, [row["cumulative_mean_extension_impulse_myr"] for row in rows], label="mean cumulative impulse")
    axes[2].set_xlabel("Time, Myr")
    axes[2].set_ylabel("Normalized / Myr")
    axes[2].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("v0.26 plume-driven rifting diagnostics")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_plume_rifting_frame(mesh, lithosphere, plume_state, rifting_state, plate_count: int, path: Path, dpi: int = 105) -> None:
    """Save a compact plate/forcing/root frame for an animated research view."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 6.2))
    axes = [fig.add_subplot(1, 3, index, projection="mollweide") for index in (1, 2, 3)]

    lon_edges, lat_edges, plate_grid = rasterize_cells(mesh, lithosphere.cell_plate)
    axes[0].pcolormesh(lon_edges, lat_edges, plate_grid, cmap="tab20", shading="auto", rasterized=True)
    axes[0].set_title(f"Plates: {plate_count}")

    _, _, forcing_grid = rasterize_cells(mesh, rifting_state.last_extension_forcing)
    forcing_image = axes[1].pcolormesh(
        lon_edges,
        lat_edges,
        forcing_grid,
        cmap="magma",
        shading="auto",
        rasterized=True,
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(forcing_image, ax=axes[1], orientation="horizontal", pad=0.08, fraction=0.05)
    if len(plume_state.centers_unit):
        centers = np.asarray(plume_state.centers_unit, dtype=float)
        lon = np.arctan2(centers[:, 1], centers[:, 0])
        lat = np.arcsin(np.clip(centers[:, 2], -1.0, 1.0))
        axes[1].scatter(lon, lat, marker="*", s=65, c="cyan", edgecolors="black", linewidths=0.5)
    axes[1].set_title("Plume extension forcing")

    root = (
        np.zeros(mesh.cell_count, dtype=float)
        if lithosphere.mantle_lithosphere_thickness_km is None
        else np.asarray(lithosphere.mantle_lithosphere_thickness_km, dtype=float)
    )
    _, _, root_grid = rasterize_cells(mesh, root)
    root_image = axes[2].pcolormesh(
        lon_edges,
        lat_edges,
        root_grid,
        cmap="cividis",
        shading="auto",
        rasterized=True,
        vmin=0.0,
        vmax=220.0,
    )
    fig.colorbar(root_image, ax=axes[2], orientation="horizontal", pad=0.08, fraction=0.05)
    extension = (
        np.zeros(mesh.cell_count, dtype=float)
        if lithosphere.rift_extension is None
        else np.asarray(lithosphere.rift_extension, dtype=float)
    )
    active = extension >= 0.35
    if np.any(active):
        points = np.asarray(mesh.centroids)[active]
        axes[2].scatter(
            np.arctan2(points[:, 1], points[:, 0]),
            np.arcsin(np.clip(points[:, 2], -1.0, 1.0)),
            s=3.0,
            c="red",
            linewidths=0,
            alpha=0.7,
        )
    axes[2].set_title("Mantle-root thickness; red = rift memory")

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Moon tectonics v0.26 — t={lithosphere.time_myr:g} Myr")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_plume_rifting_gif(frame_paths, path: Path, frame_duration_ms: int = 350) -> None:
    if len(frame_paths) >= 2:
        build_gif(frame_paths, path, frame_duration_ms)


__all__ = [
    "save_plume_rifting_maps",
    "save_plume_rifting_history",
    "save_plume_rifting_frame",
    "build_plume_rifting_gif",
]
