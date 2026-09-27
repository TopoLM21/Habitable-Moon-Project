"""Spatial support for onset diagnostics on a fixed reference sphere.

Smoothing regularizes a scalar loading field over a physical length. Material
positions and velocities are derived from membrane displacements, never from a
prescribed velocity field. Connected intact regions are diagnostic candidates;
this module does not turn them into independently moving tectonic plates.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .mesh import connected_components


def _arc(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1),
                      np.einsum("...i,...i->...", a, b))


class NonlocalLoading:
    """Area-conservative Helmholtz filter with no flux through inactive cells.

    The finite-volume system is ``(M + length_km**2 L) u = M f``.
    M contains physical face areas in km²; symmetric edge conductances in L
    are the ratio of shared-edge length to centroid separation. The most recent
    active-mask factorization is reused between calls.
    """

    def __init__(self, mesh, radius_km: float, length_km: float):
        if not np.isfinite(radius_km) or radius_km <= 0:
            raise ValueError("radius_km must be positive and finite")
        if not np.isfinite(length_km) or length_km < 0:
            raise ValueError("length_km must be nonnegative and finite")
        self.mesh = mesh
        self.length_km = float(length_km)
        self.areas = mesh.physical_cell_areas_km2(radius_km)
        edges = np.asarray(mesh.shared_edges, dtype=int)
        self._left, self._right = edges[:, 0], edges[:, 1]
        self._conductance = (
            _arc(mesh.vertices[edges[:, 2]], mesh.vertices[edges[:, 3]]) /
            _arc(mesh.centroids[self._left], mesh.centroids[self._right]))
        self._mask = None
        self._indices = None
        self._factor = None

    def apply(self, values, active):
        values = np.asarray(values, dtype=float)
        active = np.asarray(active)
        if values.ndim not in (1, 2) or values.shape[0] != self.mesh.cell_count:
            raise ValueError("values must have one row per mesh cell")
        if not np.all(np.isfinite(values)):
            raise ValueError("values must be finite")
        if active.shape != (self.mesh.cell_count,) or active.dtype != bool:
            raise ValueError("active must be a boolean mask with one entry per cell")
        result = np.zeros_like(values)
        if not np.any(active):
            return result
        if self.length_km == 0:
            result[active] = values[active]
            return result
        if self._mask is None or not np.array_equal(active, self._mask):
            self._indices = np.flatnonzero(active)
            local = np.full(self.mesh.cell_count, -1, dtype=int)
            local[self._indices] = np.arange(len(self._indices))
            use = active[self._left] & active[self._right]
            a, b = local[self._left[use]], local[self._right[use]]
            conductance = self._conductance[use] * self.length_km**2
            laplacian = sparse.coo_matrix(
                (np.concatenate((conductance, conductance, -conductance, -conductance)),
                 (np.concatenate((a, b, a, b)), np.concatenate((a, b, b, a)))),
                shape=(len(self._indices), len(self._indices))).tocsc()
            matrix = sparse.diags(self.areas[self._indices], format="csc") + laplacian
            self._factor = splu(matrix)
            self._mask = active.copy()
        weights = self.areas[self._indices]
        if values.ndim == 2:
            weights = weights[:, None]
        result[self._indices] = self._factor.solve(weights * values[self._indices])
        return result


def material_positions(mesh, vertex_basis, displacement_rad):
    """Move reference vertices tangentially, then return unit face centroids.

    This normalized linear displacement is consistent with the existing
    small-strain membrane. It does not advect its mesh or change connectivity.
    """
    basis = np.asarray(vertex_basis, dtype=float)
    displacement = np.asarray(displacement_rad, dtype=float)
    if basis.shape != (mesh.vertex_count, 3, 2):
        raise ValueError("vertex_basis must have shape (vertex_count, 3, 2)")
    if displacement.shape != (mesh.vertex_count, 2):
        raise ValueError("displacement_rad must have shape (vertex_count, 2)")
    if not np.all(np.isfinite(basis)) or not np.all(np.isfinite(displacement)):
        raise ValueError("basis and displacement must be finite")
    if not np.any(displacement):
        return mesh.centroids.copy()
    vertices = mesh.vertices + np.einsum("vij,vj->vi", basis, displacement)
    lengths = np.linalg.norm(vertices, axis=1, keepdims=True)
    if np.any(lengths == 0):
        raise ValueError("displacement collapses a vertex")
    vertices /= lengths
    centers = vertices[mesh.faces].sum(axis=1)
    lengths = np.linalg.norm(centers, axis=1, keepdims=True)
    if np.any(lengths == 0):
        raise ValueError("displacement collapses a face centroid")
    return centers / lengths


def face_velocities(previous, current, radius_km: float, dt_myr: float):
    """Great-circle displacement per time, tangent at the current positions.

    The endpoint tangent is minus log_current(previous). atan2 avoids the
    spurious finite speed of acos near identical unit vectors.
    """
    old, new = np.asarray(previous, dtype=float), np.asarray(current, dtype=float)
    if old.ndim != 2 or old.shape[1] != 3 or new.shape != old.shape:
        raise ValueError("positions must have matching (cell_count, 3) shapes")
    if not np.isfinite(radius_km) or radius_km <= 0 or not np.isfinite(dt_myr) or dt_myr <= 0:
        raise ValueError("radius and time step must be positive and finite")
    if not np.all(np.isfinite(old)) or not np.all(np.isfinite(new)):
        raise ValueError("positions must be finite")
    old_length = np.linalg.norm(old, axis=1, keepdims=True)
    new_length = np.linalg.norm(new, axis=1, keepdims=True)
    if np.any(old_length == 0) or np.any(new_length == 0):
        raise ValueError("positions must be nonzero")
    old, new = old / old_length, new / new_length
    cross = np.cross(old, new)
    sine = np.linalg.norm(cross, axis=1)
    cosine = np.clip(np.einsum("fi,fi->f", old, new), -1., 1.)
    if np.any((sine < 1e-14) & (cosine < 0)):
        raise ValueError("antipodal positions have no unique displacement direction")
    angle = np.arctan2(sine, cosine)
    factor = np.divide(angle, sine, out=np.ones_like(angle), where=sine > 0)
    # The cross-product form avoids subtracting nearly coincident positions.
    tangent = np.cross(cross, new)
    return tangent * (factor * (radius_km / dt_myr))[:, None]


def regional_motion(mesh, positions, velocity_km_myr, intact_bool):
    """Area-weighted rigid-rotation fit for each connected intact region.

    Fit ``v = omega cross r`` separately in each region, where r is a unit
    vector and omega has units km/Myr. The residual measures deformation;
    low residual alone does not establish that a region is a tectonic plate.
    Single-cell regions are underdetermined and naturally have zero tangential
    residual. No inertia, kinetic energy, or motion is invented by this fit.
    """
    position = np.asarray(positions, dtype=float)
    velocity = np.asarray(velocity_km_myr, dtype=float)
    intact = np.asarray(intact_bool)
    if position.shape != (mesh.cell_count, 3) or velocity.shape != position.shape:
        raise ValueError("positions and velocities must have shape (cell_count, 3)")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        raise ValueError("positions and velocities must be finite")
    if not np.allclose(np.linalg.norm(position, axis=1), 1., rtol=1e-10, atol=1e-12):
        raise ValueError("positions must be unit vectors")
    if intact.shape != (mesh.cell_count,) or intact.dtype != bool:
        raise ValueError("intact_bool must be a boolean cell mask")
    area = mesh.areas_unit_sphere
    total_area = float(np.sum(area))
    intact_area = float(np.sum(area[intact]))
    components = connected_components(np.flatnonzero(intact), mesh.neighbors)
    residual_sum = 0.
    speed_sum = float(np.sum(area[intact] * np.sum(velocity[intact]**2, axis=1)))
    largest_area = 0.
    for component in components:
        indices = np.asarray(component, dtype=int)
        r, v, w = position[indices], velocity[indices], area[indices]
        region_area = float(np.sum(w))
        largest_area = max(largest_area, region_area)
        normal = region_area * np.eye(3) - np.einsum("f,fi,fj->ij", w, r, r)
        rhs = np.sum(w[:, None] * np.cross(r, v), axis=0)
        omega = np.linalg.lstsq(normal, rhs, rcond=1e-12)[0]
        residual = v - np.cross(omega, r)
        residual_sum += float(np.sum(w * np.sum(residual**2, axis=1)))
    rms_speed = np.sqrt(speed_sum / intact_area) if intact_area else 0.
    rms_residual = np.sqrt(residual_sum / intact_area) if intact_area else 0.
    return {
        "candidate_region_count": len(components),
        "intact_area_fraction": intact_area / total_area,
        "largest_candidate_area_fraction": largest_area / total_area,
        "intact_rms_speed_km_myr": float(rms_speed),
        "rigid_fit_residual_km_myr": float(rms_residual),
        "rigid_fit_residual_fraction": float(np.sqrt(residual_sum / speed_sum)) if speed_sum else 0.,
    }
