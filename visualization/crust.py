"""Visualization for v0.3 oceanic crust evolution."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tectonics.crust import OceanicSnapshot, OceanicCrustState, StepDiagnostics
from tectonics.kinematics import BoundaryType
from tectonics.mesh import SphereMesh

BOUNDARY_LABELS = {
    BoundaryType.INACTIVE: "inactive",
    BoundaryType.DIVERGENT: "divergent",
    BoundaryType.CONVERGENT: "convergent",
    BoundaryType.TRANSFORM: "transform",
}


def _lon_lat(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.arctan2(points[:, 1], points[:, 0])
    lat = np.arcsin(np.clip(points[:, 2], -1.0, 1.0))
    return lon, lat


def save_crust_frame(
    mesh: SphereMesh,
    snapshot: OceanicSnapshot,
    path: str | Path,
    dpi: int = 130,
) -> None:
    lon, lat = _lon_lat(mesh.centroids)
    fig = plt.figure(figsize=(12, 6.8))
    ax = fig.add_subplot(111, projection="mollweide")
    sc = ax.scatter(
        lon,
        lat,
        c=snapshot.state.crust_age_myr,
        cmap="viridis",
        s=2.5,
        linewidths=0,
        rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.08, fraction=0.05)
    cb.set_label("Oceanic crust age, Myr")

    for kind in BoundaryType:
        pts = np.asarray([b.midpoint for b in snapshot.boundaries if b.boundary_type == kind])
        if len(pts) == 0:
            continue
        blon, blat = _lon_lat(pts)
        ax.scatter(blon, blat, s=4.0, linewidths=0, alpha=0.75, label=BOUNDARY_LABELS[kind])

    diag = snapshot.diagnostics
    extra = ""
    if diag is not None:
        extra = (
            f" | pre-resolve gaps={100.0*diag.pre_resolution_gap_fraction:.1f}%"
            f" overlaps={100.0*diag.pre_resolution_overlap_fraction:.1f}%"
            f" | created={diag.created_area_km2/1e6:.3f} Mkm²"
            f" subducted={diag.subducted_area_km2/1e6:.3f} Mkm²"
        )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.15))
    ax.set_title(
        f"v0.3 oceanic crust — t = {snapshot.state.time_myr:g} Myr\n"
        f"mean age={np.mean(snapshot.state.crust_age_myr):.2f} Myr  max age={np.max(snapshot.state.crust_age_myr):.2f} Myr"
        f"{extra}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_crust_age_map(mesh: SphereMesh, state: OceanicCrustState, path: str | Path, dpi: int = 180) -> None:
    lon, lat = _lon_lat(mesh.centroids)
    fig = plt.figure(figsize=(12, 6.5))
    ax = fig.add_subplot(111, projection="mollweide")
    sc = ax.scatter(lon, lat, c=state.crust_age_myr, cmap="viridis", s=2.5, linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.08, fraction=0.05, label="Oceanic crust age, Myr")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Final oceanic crust age — t = {state.time_myr:g} Myr")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_plate_map(mesh: SphereMesh, state: OceanicCrustState, path: str | Path, dpi: int = 180) -> None:
    lon, lat = _lon_lat(mesh.centroids)
    fig = plt.figure(figsize=(12, 6.5))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.scatter(lon, lat, c=state.cell_plate, cmap="tab20", s=2.2, linewidths=0, rasterized=True)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Top-surface plate ownership — t = {state.time_myr:g} Myr")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_boundary_map(mesh: SphereMesh, state: OceanicCrustState, boundaries, path: str | Path, dpi: int = 180) -> None:
    lon, lat = _lon_lat(mesh.centroids)
    fig = plt.figure(figsize=(12, 6.5))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.scatter(lon, lat, c=state.cell_plate, cmap="tab20", s=1.0, alpha=0.16, linewidths=0)
    for kind in BoundaryType:
        pts = np.asarray([b.midpoint for b in boundaries if b.boundary_type == kind])
        if len(pts) == 0:
            continue
        blon, blat = _lon_lat(pts)
        ax.scatter(blon, blat, s=5.0, linewidths=0, label=BOUNDARY_LABELS[kind])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13))
    ax.set_title(f"Boundary kinematics on evolving top-surface ownership — t = {state.time_myr:g} Myr")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_budget_history(
    times: np.ndarray,
    created_area_km2: np.ndarray,
    subducted_area_km2: np.ndarray,
    mean_age_myr: np.ndarray,
    path: str | Path,
    dpi: int = 160,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, created_area_km2 / 1e6, label="created oceanic area per step")
    ax.plot(times, subducted_area_km2 / 1e6, label="subducted oceanic area per step")
    ax.set_xlabel("Time, Myr")
    ax.set_ylabel("Area, million km² per step")
    ax.set_title("v0.3 crust budget")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(times, mean_age_myr, linestyle="--", label="mean crust age")
    ax2.set_ylabel("Mean oceanic crust age, Myr")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_gap_overlap_history(
    times: np.ndarray,
    gaps: np.ndarray,
    overlaps: np.ndarray,
    path: str | Path,
    dpi: int = 160,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, 100.0 * gaps, label="pre-resolution gap fraction")
    ax.plot(times, 100.0 * overlaps, label="pre-resolution overlap fraction")
    ax.set_xlabel("Time, Myr")
    ax.set_ylabel("Reference-cell fraction, %")
    ax.set_title("v0.3 unresolved geometry before crust budget repair")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def build_gif(frame_paths: list[Path], output_path: str | Path, frame_duration_ms: int = 350) -> None:
    if not frame_paths:
        raise ValueError("No frames supplied")
    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
    try:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=int(frame_duration_ms),
            loop=0,
            optimize=False,
        )
    finally:
        for image in images:
            image.close()
