"""Time evolution for the v0.2 kinematic plate prototype.

v0.2 deliberately has no crust creation, destruction, age or subduction.
Every initial surface cell is a Lagrangian material marker rigidly attached to
one plate.  Each plate rotates around its Euler pole, so the plate patch keeps
its shape exactly.  Adjacent plate edges are allowed to separate or overlap.
Those gaps/overlaps are *diagnostics*, not silently repaired; v0.3 will add the
crust physics that resolves them through spreading and subduction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .kinematics import BoundaryRecord
from .mesh import SphereMesh
from .plates import PlateSystem


Array = np.ndarray


@dataclass(slots=True)
class CoverageDiagnostics:
    """Coarse diagnostic of gaps and overlaps on the fixed reference mesh."""

    uncovered_cell_fraction: float
    single_covered_cell_fraction: float
    multiply_covered_cell_fraction: float
    nearest_marker_angle_deg_mean: float
    nearest_marker_angle_deg_max: float


@dataclass(slots=True)
class EvolutionSnapshot:
    time_myr: float
    marker_positions: Array              # one rigidly advected marker per initial cell
    boundary_side_a: Array               # (B, 3), advected copy attached to plate A
    boundary_side_b: Array               # (B, 3), advected copy attached to plate B
    boundary_separation_km: Array         # (B,), great-circle separation of the two copies
    coverage: CoverageDiagnostics


def rotate_points_by_plate(points: Array, plate_ids: Array, system: PlateSystem, time_myr: float) -> Array:
    """Rotate points with Rodrigues' formula using fixed Euler poles.

    The state is evaluated analytically from t=0 instead of incrementally, so
    no integration drift accumulates.
    """
    points = np.asarray(points, dtype=np.float64)
    plate_ids = np.asarray(plate_ids, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if plate_ids.shape != (len(points),):
        raise ValueError("plate_ids must have shape (N,)")

    axes_by_plate = np.asarray([p.euler_axis for p in system.plates], dtype=np.float64)
    speeds_by_plate = np.asarray([p.angular_speed_rad_per_myr for p in system.plates], dtype=np.float64)
    axes = axes_by_plate[plate_ids]
    angles = speeds_by_plate[plate_ids] * float(time_myr)

    cos_a = np.cos(angles)[:, None]
    sin_a = np.sin(angles)[:, None]
    axial = np.sum(axes * points, axis=1)[:, None]
    rotated = points * cos_a + np.cross(axes, points) * sin_a + axes * axial * (1.0 - cos_a)
    rotated /= np.linalg.norm(rotated, axis=1, keepdims=True)
    return rotated


def _boundary_side_positions(
    boundaries: list[BoundaryRecord],
    system: PlateSystem,
    time_myr: float,
) -> tuple[Array, Array]:
    if not boundaries:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, empty.copy()

    midpoints = np.asarray([b.midpoint for b in boundaries], dtype=np.float64)
    plate_a = np.asarray([b.plate_a for b in boundaries], dtype=np.int32)
    plate_b = np.asarray([b.plate_b for b in boundaries], dtype=np.int32)
    return (
        rotate_points_by_plate(midpoints, plate_a, system, time_myr),
        rotate_points_by_plate(midpoints, plate_b, system, time_myr),
    )


def _coverage_diagnostics(mesh: SphereMesh, marker_positions: Array) -> CoverageDiagnostics:
    """Measure unresolved kinematic gaps/overlaps without changing plate geometry.

    Each material marker votes for its nearest fixed diagnostic cell.  Zero
    votes indicate a locally under-covered region; multiple votes indicate a
    locally over-covered region.  This is intentionally only a diagnostic.
    """
    target_tree = cKDTree(mesh.centroids)
    _, target_cell = target_tree.query(marker_positions, k=1, workers=-1)
    multiplicity = np.bincount(target_cell, minlength=mesh.cell_count)

    marker_tree = cKDTree(marker_positions)
    nearest_distance, _ = marker_tree.query(mesh.centroids, k=1, workers=-1)
    chord = np.clip(nearest_distance, 0.0, 2.0)
    angle = 2.0 * np.arcsin(0.5 * chord)

    return CoverageDiagnostics(
        uncovered_cell_fraction=float(np.mean(multiplicity == 0)),
        single_covered_cell_fraction=float(np.mean(multiplicity == 1)),
        multiply_covered_cell_fraction=float(np.mean(multiplicity > 1)),
        nearest_marker_angle_deg_mean=float(np.rad2deg(np.mean(angle))),
        nearest_marker_angle_deg_max=float(np.rad2deg(np.max(angle))),
    )


def snapshot_at_time(
    mesh: SphereMesh,
    initial_system: PlateSystem,
    initial_boundaries: list[BoundaryRecord],
    radius_km: float,
    time_myr: float,
) -> EvolutionSnapshot:
    marker_positions = rotate_points_by_plate(
        mesh.centroids,
        initial_system.cell_plate,
        initial_system,
        time_myr,
    )
    side_a, side_b = _boundary_side_positions(initial_boundaries, initial_system, time_myr)

    if len(side_a):
        dot = np.clip(np.sum(side_a * side_b, axis=1), -1.0, 1.0)
        separation = np.arccos(dot) * float(radius_km)
    else:
        separation = np.empty(0, dtype=np.float64)

    return EvolutionSnapshot(
        time_myr=float(time_myr),
        marker_positions=marker_positions,
        boundary_side_a=side_a,
        boundary_side_b=side_b,
        boundary_separation_km=separation,
        coverage=_coverage_diagnostics(mesh, marker_positions),
    )


def snapshot_times(duration_myr: float, interval_myr: float) -> Array:
    if duration_myr < 0.0:
        raise ValueError("duration_myr must be non-negative")
    if interval_myr <= 0.0:
        raise ValueError("interval_myr must be positive")

    count = int(np.floor(duration_myr / interval_myr + 1e-12))
    times = np.arange(count + 1, dtype=np.float64) * interval_myr
    if times[-1] < duration_myr - 1e-10:
        times = np.append(times, float(duration_myr))
    return times
