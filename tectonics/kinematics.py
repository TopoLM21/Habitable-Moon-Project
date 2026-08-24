"""Rigid-plate velocities and boundary classification on a sphere."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .mesh import SphereMesh
from .plates import PlateSystem


Array = np.ndarray


class BoundaryType(IntEnum):
    INACTIVE = 0
    DIVERGENT = 1
    CONVERGENT = 2
    TRANSFORM = 3


@dataclass(slots=True)
class BoundaryRecord:
    face_a: int
    face_b: int
    vertex_u: int
    vertex_v: int
    plate_a: int
    plate_b: int
    midpoint: Array
    normal_rate_km_per_myr: float
    tangential_rate_km_per_myr: float
    relative_speed_km_per_myr: float
    boundary_type: BoundaryType


def angular_velocity_vectors(system: PlateSystem) -> Array:
    return np.asarray(
        [p.euler_axis * p.angular_speed_rad_per_myr for p in system.plates],
        dtype=np.float64,
    )


def cell_velocities_km_per_myr(mesh: SphereMesh, system: PlateSystem, radius_km: float) -> Array:
    omega = angular_velocity_vectors(system)[system.cell_plate]
    return np.cross(omega, mesh.centroids) * float(radius_km)


def _safe_unit(v: Array) -> Array:
    n = float(np.linalg.norm(v))
    if n < 1e-14:
        raise ValueError("Degenerate tangent direction")
    return v / n


def classify_boundaries(
    mesh: SphereMesh,
    system: PlateSystem,
    radius_km: float,
    normal_threshold_km_per_myr: float,
    inactive_speed_km_per_myr: float,
) -> list[BoundaryRecord]:
    """Vectorized boundary kinematics over the icosphere edge table.

    v0.9.5 performs this classification several times per geological step.
    The original scalar loop was numerically clear but dominated long 4-Gyr
    integrations.  This implementation evaluates all inter-plate edges with
    the same equations in NumPy, then materializes BoundaryRecord objects only
    for the much smaller boundary subset.
    """
    edges = np.asarray(mesh.shared_edges, dtype=np.int64)
    if edges.size == 0:
        return []
    face_a_all, face_b_all, vertex_u_all, vertex_v_all = edges.T
    owner = np.asarray(system.cell_plate, dtype=np.int32)
    plate_a_all = owner[face_a_all]
    plate_b_all = owner[face_b_all]
    mask = plate_a_all != plate_b_all
    if not np.any(mask):
        return []

    face_a = face_a_all[mask]; face_b = face_b_all[mask]
    vertex_u = vertex_u_all[mask]; vertex_v = vertex_v_all[mask]
    plate_a = plate_a_all[mask]; plate_b = plate_b_all[mask]

    midpoint = mesh.vertices[vertex_u] + mesh.vertices[vertex_v]
    midpoint /= np.linalg.norm(midpoint, axis=1, keepdims=True)

    omega = angular_velocity_vectors(system)
    velocity_a = np.cross(omega[plate_a], midpoint) * float(radius_km)
    velocity_b = np.cross(omega[plate_b], midpoint) * float(radius_km)
    relative = velocity_b - velocity_a

    direction = mesh.centroids[face_b] - mesh.centroids[face_a]
    direction -= midpoint * np.sum(direction * midpoint, axis=1, keepdims=True)
    dnorm = np.linalg.norm(direction, axis=1, keepdims=True)
    if np.any(dnorm < 1e-14):
        raise ValueError("Degenerate tangent direction")
    normal = direction / dnorm
    tangent = np.cross(midpoint, normal)
    tnorm = np.linalg.norm(tangent, axis=1, keepdims=True)
    if np.any(tnorm < 1e-14):
        raise ValueError("Degenerate tangent direction")
    tangent /= tnorm

    normal_rate = np.sum(relative * normal, axis=1)
    tangential_rate = np.sum(relative * tangent, axis=1)
    relative_speed = np.linalg.norm(relative, axis=1)

    kinds = np.full(len(face_a), int(BoundaryType.TRANSFORM), dtype=np.int8)
    kinds[normal_rate > float(normal_threshold_km_per_myr)] = int(BoundaryType.DIVERGENT)
    kinds[normal_rate < -float(normal_threshold_km_per_myr)] = int(BoundaryType.CONVERGENT)
    kinds[relative_speed < float(inactive_speed_km_per_myr)] = int(BoundaryType.INACTIVE)

    return [
        BoundaryRecord(
            face_a=int(fa), face_b=int(fb), vertex_u=int(vu), vertex_v=int(vv),
            plate_a=int(pa), plate_b=int(pb), midpoint=mp.copy(),
            normal_rate_km_per_myr=float(nr),
            tangential_rate_km_per_myr=float(tr),
            relative_speed_km_per_myr=float(rs),
            boundary_type=BoundaryType(int(kind)),
        )
        for fa, fb, vu, vv, pa, pb, mp, nr, tr, rs, kind in zip(
            face_a, face_b, vertex_u, vertex_v, plate_a, plate_b, midpoint,
            normal_rate, tangential_rate, relative_speed, kinds
        )
    ]


def rigid_motion_residual(mesh: SphereMesh, system: PlateSystem) -> float:
    """Maximum derivative of r_i·r_j for neighboring cells on one plate.

    A rigid rotation preserves angular distance, therefore the derivative must
    be zero up to floating-point error.
    """
    omega = angular_velocity_vectors(system)
    worst = 0.0
    for face_a, face_b, _, _ in mesh.shared_edges:
        plate_a = int(system.cell_plate[face_a])
        if plate_a != int(system.cell_plate[face_b]):
            continue
        r1 = mesh.centroids[face_a]
        r2 = mesh.centroids[face_b]
        u1 = np.cross(omega[plate_a], r1)
        u2 = np.cross(omega[plate_a], r2)
        residual = abs(float(np.dot(u1, r2) + np.dot(r1, u2)))
        worst = max(worst, residual)
    return worst
