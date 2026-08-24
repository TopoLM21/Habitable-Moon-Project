"""Maps, diagnostics and animation for v0.2 rigid plate patches."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tectonics.evolution import EvolutionSnapshot
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


def save_evolution_frame(
    result: PrototypeResult,
    snapshot: EvolutionSnapshot,
    path: str | Path,
    dpi: int = 130,
) -> None:
    fig = plt.figure(figsize=(12, 6.6))
    ax = fig.add_subplot(111, projection="mollweide")

    # Draw each material plate separately.  No ownership is reassigned to the
    # fixed grid, so rigid patches stay rigid; white gaps and overdrawn regions
    # are the unresolved divergence/convergence that v0.3 must handle.
    for plate_id in range(len(result.plates.plates)):
        mask = result.plates.cell_plate == plate_id
        lon, lat = _lon_lat(snapshot.marker_positions[mask])
        ax.scatter(
            lon,
            lat,
            c=np.full(mask.sum(), plate_id),
            cmap="tab20",
            vmin=0,
            vmax=max(len(result.plates.plates) - 1, 1),
            s=2.0,
            linewidths=0,
            rasterized=True,
        )

    # Plot the two material copies of every original plate boundary.  At t=0
    # they coincide; later they split apart or pass across each other.
    for kind in BoundaryType:
        idx = np.asarray([i for i, b in enumerate(result.boundaries) if b.boundary_type == kind], dtype=int)
        if len(idx) == 0:
            continue
        pa = snapshot.boundary_side_a[idx]
        pb = snapshot.boundary_side_b[idx]
        lon_a, lat_a = _lon_lat(pa)
        lon_b, lat_b = _lon_lat(pb)
        ax.scatter(lon_a, lat_a, s=2.0, linewidths=0, alpha=0.65, label=BOUNDARY_LABELS[kind])
        ax.scatter(lon_b, lat_b, s=2.0, linewidths=0, alpha=0.65)

    cov = snapshot.coverage
    mean_sep = float(np.mean(snapshot.boundary_separation_km)) if len(snapshot.boundary_separation_km) else 0.0
    max_sep = float(np.max(snapshot.boundary_separation_km)) if len(snapshot.boundary_separation_km) else 0.0
    ax.grid(True, alpha=0.30)
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13))
    ax.set_title(
        f"v0.2 rigid plate patches — t = {snapshot.time_myr:g} Myr\n"
        f"mean paired-edge separation={mean_sep:,.0f} km  max={max_sep:,.0f} km  |  "
        f"diagnostic gaps={100*cov.uncovered_cell_fraction:.1f}% overlaps={100*cov.multiply_covered_cell_fraction:.1f}%"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_boundary_separation_history(
    times: np.ndarray,
    mean_sep: np.ndarray,
    p95_sep: np.ndarray,
    max_sep: np.ndarray,
    path: str | Path,
    dpi: int = 160,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, mean_sep, label="mean")
    ax.plot(times, p95_sep, label="95th percentile")
    ax.plot(times, max_sep, label="maximum")
    ax.set_xlabel("Time, Myr")
    ax.set_ylabel("Separation of paired original plate edges, km")
    ax.set_title("v0.2 unresolved boundary separation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_coverage_history(
    times: np.ndarray,
    uncovered: np.ndarray,
    multiply_covered: np.ndarray,
    path: str | Path,
    dpi: int = 160,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, 100.0 * uncovered, label="under-covered reference cells")
    ax.plot(times, 100.0 * multiply_covered, label="multiply covered reference cells")
    ax.set_xlabel("Time, Myr")
    ax.set_ylabel("Reference-cell fraction, %")
    ax.set_title("Diagnostic only: gaps and overlaps that v0.3 must resolve")
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
