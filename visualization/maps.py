"""Diagnostic maps for the tectonic kinematics prototype."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tectonics.kinematics import BoundaryType
from tectonics.simulation import PrototypeResult


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


def save_plate_map(result: PrototypeResult, path: str | Path, dpi: int = 180) -> None:
    lon, lat = _lon_lat(result.mesh.centroids)
    fig = plt.figure(figsize=(12, 6.5))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.scatter(
        lon,
        lat,
        c=result.plates.cell_plate,
        cmap="tab20",
        s=2.2,
        linewidths=0,
        rasterized=True,
    )
    ax.grid(True, alpha=0.35)
    ax.set_title(
        f"Initial connected plates — {len(result.plates.plates)} plates, "
        f"{result.mesh.cell_count:,} cells"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_boundary_map(result: PrototypeResult, path: str | Path, dpi: int = 180) -> None:
    fig = plt.figure(figsize=(12, 6.5))
    ax = fig.add_subplot(111, projection="mollweide")

    # Very faint plate cells provide geographic context.
    lon, lat = _lon_lat(result.mesh.centroids)
    ax.scatter(lon, lat, c=result.plates.cell_plate, cmap="tab20", s=1.0, alpha=0.16, linewidths=0)

    for kind in BoundaryType:
        points = np.asarray([b.midpoint for b in result.boundaries if b.boundary_type == kind])
        if len(points) == 0:
            continue
        blon, blat = _lon_lat(points)
        ax.scatter(blon, blat, s=5.0, linewidths=0, label=BOUNDARY_LABELS[kind])

    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13))
    ax.set_title("Boundary kinematics from relative Euler rotation")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_velocity_histogram(result: PrototypeResult, path: str | Path, dpi: int = 180) -> None:
    values = np.asarray([b.normal_rate_km_per_myr for b in result.boundaries], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(values, bins=60)
    ax.axvline(0.0, linewidth=1.0)
    ax.set_xlabel("Normal relative rate, km/Myr (positive = divergence)")
    ax.set_ylabel("Boundary-edge count")
    ax.set_title("Distribution of plate-boundary normal motion")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
