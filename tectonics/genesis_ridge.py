"""Seeded geometric tracking of localized weak bands on a reference sphere.

This is an experimental *diagnostic*, not a fracture constitutive law. It
finds transverse maxima of a supplied scalar along supplied unoriented normal
lines. Seeds, analysis scales and contrast thresholds are explicit inputs;
they do not establish physical nucleation, propagation speed or fracture work.
Paths are continuous spherical polylines, never independently selected mesh
edges. Material cells, contact histories and production checkpoints are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral, Real

import numpy as np
from scipy.spatial import cKDTree


class RidgeUnavailable(RuntimeError):
    """The sampled field does not resolve a supported transverse maximum."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class RidgeParameters:
    fit_radius_km: float = 1200.
    probe_distance_km: float = 600.
    step_km: float = 100.
    max_correction_km: float = 250.
    projection_tolerance_km: float = .01
    min_peak_value: float = .1
    min_transverse_contrast: float = .025
    min_orientation_coherence: float = .7
    max_along_curvature_ratio: float = .5
    max_fit_residual: float = .08
    min_fit_condition: float = 1e-4
    max_cell_scale_to_probe: float = .75
    max_turn_degrees: float = 30.
    max_branch_length_km: float = 6000.
    min_samples: int = 12
    max_projection_iterations: int = 12

    def validate(self, radius_km):
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"ridge.{item.name} must be finite and positive")
        if (isinstance(radius_km, bool) or not isinstance(radius_km, Real)
                or not np.isfinite(radius_km) or radius_km <= 0):
            raise ValueError("Ridge radius must be finite and positive")
        if (not isinstance(self.min_samples, Integral) or self.min_samples < 6
                or not isinstance(self.max_projection_iterations, Integral)
                or not self.projection_tolerance_km < self.step_km <= self.probe_distance_km
                or not self.probe_distance_km < self.fit_radius_km < np.pi*radius_km/4
                or self.max_correction_km > self.probe_distance_km
                or self.max_branch_length_km < self.step_km
                or self.max_turn_degrees >= 90
                or any(getattr(self, name) >= 1 for name in (
                    "min_peak_value", "min_transverse_contrast", "min_orientation_coherence",
                    "max_along_curvature_ratio", "max_fit_residual", "min_fit_condition",
                    "max_cell_scale_to_probe"))):
            raise ValueError("Invalid ridge analysis scales or numerical guards")


@dataclass(frozen=True)
class RidgeSample:
    point_xyz: np.ndarray
    normal_xyz: np.ndarray
    value: float
    gradient: np.ndarray
    hessian: np.ndarray
    basis: np.ndarray
    fit_residual: float
    fit_condition: float
    orientation_coherence: float
    sample_count: int
    effective_sample_count: float
    transverse_contrast: float = 0.


@dataclass(frozen=True)
class RidgePath:
    points_xyz: np.ndarray
    arclength_km: np.ndarray
    values: np.ndarray
    transverse_contrast: np.ndarray
    seed_index: int
    left_stop: str
    right_stop: str
    closed: bool


def _unit(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all() or not np.isclose(np.linalg.norm(value), 1., atol=1e-12, rtol=0):
        raise ValueError("Ridge query must be a finite unit three-vector")
    return value


def _basis(point):
    axis = np.eye(3)[np.argmin(np.abs(point))]
    first = np.cross(point, axis)
    first /= np.linalg.norm(first)
    return np.column_stack((first, np.cross(point, first)))


def _advance(point, tangent, distance_km, radius_km):
    tangent = tangent - np.dot(tangent, point)*point
    tangent /= np.linalg.norm(tangent)
    angle = distance_km/radius_km
    moved = np.cos(angle)*point + np.sin(angle)*tangent
    return moved/np.linalg.norm(moved)


def _transport(vectors, source, target):
    # Exact parallel transport along the minor great-circle arc. All local
    # reconstruction samples lie within pi/4 of the target by construction.
    return vectors - (np.sum(vectors*target, axis=-1)/(1+np.sum(source*target, axis=-1)))[..., None]*(source+target)


def _distance(a, b, radius_km):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a*b, axis=-1))*radius_km


class RidgeField:
    """Fixed-scale reconstruction of cell-centred scalar and line directions.

    Values lie in [0, 1]. Normals are unit 3-D tangent vectors at the spherical
    cell centres, with arbitrary signs. Inactive normals must be zero. The
    scalar fit includes inactive cells; excluding them would erase gaps and
    bias the measured contrast. No support radius is enlarged on coarse grids.
    """

    def __init__(self, mesh, values, plane_normals_xyz, radius_km, *, active=None, parameters=None):
        self.p = parameters or RidgeParameters()
        self.p.validate(radius_km)
        self.radius_km = float(radius_km)
        self.centers = np.array(mesh.centroids, dtype=float, copy=True)
        self.areas = np.array(mesh.areas_unit_sphere, dtype=float, copy=True)
        self.values = np.array(values, dtype=float, copy=True)
        self.normals = np.array(plane_normals_xyz, dtype=float, copy=True)
        self.active = np.ones(len(self.centers), dtype=bool) if active is None else np.array(active, copy=True)
        n = len(self.centers)
        if (not n or self.centers.shape != (n, 3) or not np.isfinite(self.centers).all()
                or not np.allclose(np.linalg.norm(self.centers, axis=1), 1., atol=1e-12, rtol=0)
                or self.areas.shape != (n,) or not np.isfinite(self.areas).all() or np.any(self.areas <= 0)
                or self.values.shape != (n,) or not np.isfinite(self.values).all()
                or np.any((self.values < 0)|(self.values > 1))
                or self.active.dtype != bool or self.active.shape != (n,)
                or self.normals.shape != (n, 3) or not np.isfinite(self.normals).all()):
            raise ValueError("Invalid cell samples for ridge reconstruction")
        if (not np.allclose(np.linalg.norm(self.normals[self.active], axis=1), 1., atol=1e-10, rtol=0)
                or np.any(np.abs(np.sum(self.normals*self.centers, axis=1)) > 1e-10)
                or np.any(self.normals[~self.active] != 0)):
            raise ValueError("Active ridge normals must be unit tangent vectors; inactive normals must be zero")
        for value in (self.centers, self.areas, self.values, self.normals, self.active):
            value.setflags(write=False)
        self.tree = cKDTree(self.centers)
        self.chord_radius = 2*np.sin(self.p.fit_radius_km/(2*self.radius_km))

    def _core(self, point):
        _, nearest = self.tree.query(point)
        scale = np.sqrt(self.areas[nearest])*self.radius_km
        if scale > self.p.max_cell_scale_to_probe*self.p.probe_distance_km:
            raise RidgeUnavailable("underresolved_cells")
        if not self.active[nearest] or self.values[nearest] < self.p.min_peak_value:
            raise RidgeUnavailable("inactive_core")

    def _fit(self, point, *, need_orientation=True):
        indices = np.asarray(sorted(self.tree.query_ball_point(point, self.chord_radius)), dtype=int)
        if len(indices) < self.p.min_samples:
            raise RidgeUnavailable("insufficient_samples")
        centers = self.centers[indices]
        dot = np.clip(centers@point, -1., 1.)
        tangent = centers-dot[:, None]*point
        sine = np.linalg.norm(tangent, axis=1)
        angles = np.arctan2(sine, dot)
        basis = _basis(point)
        log = tangent*np.divide(angles, sine, out=np.ones_like(angles), where=sine > 1e-15)[:, None]
        xy = log@basis*self.radius_km/self.p.fit_radius_km
        x, y = xy.T
        radial = angles*self.radius_km/self.p.fit_radius_km
        kernel = np.maximum(1-radial, 0.)**4*(1+4*radial)
        weights = self.areas[indices]*kernel
        weights /= weights.sum()
        # Frobenius-orthonormal coordinates for the symmetric Hessian keep
        # singular values/conditioning invariant under a tangent-basis turn.
        design = np.column_stack((np.ones(len(x)), x, y, .5*x*x, x*y/np.sqrt(2.), .5*y*y))
        coeff, _, rank, singular = np.linalg.lstsq(design*np.sqrt(weights)[:, None],
            self.values[indices]*np.sqrt(weights), rcond=None)
        condition = singular[-1]/singular[0]
        effective = 1/np.sum(weights**2)
        if rank != 6 or condition < self.p.min_fit_condition or effective < 6:
            raise RidgeUnavailable("ill_conditioned_fit")
        # A supported query must lie inside the local sample fan. Use angular
        # coverage, not axis-aligned quadrants: the guard must not depend on
        # the arbitrary tangent basis chosen for the least-squares solve.
        supported = xy[(radial > .15)&(weights > .001)]
        angles = np.sort(np.arctan2(supported[:, 1], supported[:, 0]))
        if len(angles) < 3 or np.max(np.diff(np.r_[angles, angles[0]+2*np.pi])) >= np.pi:
            raise RidgeUnavailable("one_sided_support")
        residual = float(np.sqrt(np.sum(weights*(design@coeff-self.values[indices])**2)))
        if residual > self.p.max_fit_residual:
            raise RidgeUnavailable("poor_scalar_fit")
        normal, coherence = np.zeros(3), 0.
        if need_orientation:
            oriented_weights = weights*self.active[indices]
            if oriented_weights.sum() == 0:
                raise RidgeUnavailable("no_active_orientation")
            vectors = _transport(self.normals[indices], centers, point)@basis
            dyad = np.einsum("f,fi,fj->ij", oriented_weights, vectors, vectors)/oriented_weights.sum()
            eigen, axes = np.linalg.eigh(dyad)
            coherence = float((eigen[1]-eigen[0])/max(eigen.sum(), 1e-15))
            if coherence < self.p.min_orientation_coherence:
                raise RidgeUnavailable("ambiguous_orientation")
            normal = basis@axes[:, 1]
        return RidgeSample(point.copy(), normal, float(coeff[0]), coeff[1:3]/self.p.fit_radius_km,
            np.array([[coeff[3], coeff[4]/np.sqrt(2.)], [coeff[4]/np.sqrt(2.), coeff[5]]])/self.p.fit_radius_km**2,
            basis, residual, float(condition), coherence, len(indices), float(effective))

    def project(self, seed):
        """Find a nearby supported transverse maximum; never search globally."""
        from dataclasses import replace

        point = _unit(seed).copy()
        correction = 0.
        for _ in range(self.p.max_projection_iterations):
            self._core(point)
            sample = self._fit(point)
            n = sample.basis.T@sample.normal_xyz
            transverse = float(n@sample.hessian@n)
            # Contrast-scale curvature floor rejects constant fields including
            # floating-point remnants; this is a numerical, not rock threshold.
            if transverse >= -self.p.min_transverse_contrast/self.p.probe_distance_km**2:
                raise RidgeUnavailable("no_transverse_maximum")
            shift = -float(sample.gradient@n)/transverse
            if abs(shift) <= self.p.projection_tolerance_km:
                break
            correction += abs(shift)
            if correction > self.p.max_correction_km:
                raise RidgeUnavailable("projection_out_of_range")
            point = _advance(point, sample.normal_xyz, shift, self.radius_km)
        else:
            raise RidgeUnavailable("projection_not_converged")
        t = np.array([-n[1], n[0]])
        if abs(float(t@sample.hessian@t)) > self.p.max_along_curvature_ratio*abs(transverse):
            raise RidgeUnavailable("not_an_elongated_ridge")
        if sample.value < self.p.min_peak_value:
            raise RidgeUnavailable("weak_peak")
        sides = [self._fit(_advance(point, sample.normal_xyz, sign*self.p.probe_distance_km,
                                  self.radius_km), need_orientation=False).value for sign in (-1, 1)]
        contrast = float(sample.value-max(sides))
        if contrast < max(self.p.min_transverse_contrast, 2*sample.fit_residual):
            raise RidgeUnavailable("unresolved_transverse_contrast")
        return replace(sample, transverse_contrast=contrast)

    def _branch(self, seed, sign, other_branch=None):
        samples = [seed]
        tangent = sign*np.cross(seed.point_xyz, seed.normal_xyz)
        start_tangent = tangent.copy()
        travelled = 0.
        while travelled < self.p.max_branch_length_km-self.p.projection_tolerance_km:
            previous = samples[-1]
            step = min(self.p.step_km, self.p.max_branch_length_km-travelled)
            if step <= self.p.projection_tolerance_km:
                break
            try:
                predictor = _advance(previous.point_xyz, tangent, step, self.radius_km)
                self._core(_advance(previous.point_xyz, tangent, step/2, self.radius_km))
                current = self.project(predictor)
                distance = float(_distance(previous.point_xyz, current.point_xyz, self.radius_km))
                if not .25*step <= distance <= 1.5*step:
                    raise RidgeUnavailable("nonadvancing_projection")
                midpoint = previous.point_xyz+current.point_xyz
                self._core(midpoint/np.linalg.norm(midpoint))
                heading = _transport(tangent, previous.point_xyz, current.point_xyz)
                next_tangent = np.cross(current.point_xyz, current.normal_xyz)
                next_tangent *= 1 if next_tangent@heading >= 0 else -1
                if next_tangent@heading < np.cos(np.deg2rad(self.p.max_turn_degrees)):
                    raise RidgeUnavailable("abrupt_path_turn")
                if travelled+distance > self.p.max_branch_length_km+1e-8:
                    return samples, "length_limit", False
                if other_branch is not None and len(other_branch) > 5:
                    other_points = np.asarray([item.point_xyz for item in other_branch[5:]])
                    if np.min(_distance(other_points, current.point_xyz, self.radius_km)) < .75*self.p.step_km:
                        raise RidgeUnavailable("other_branch_approach")
                # Guard proximity to earlier samples. This is not an exact
                # spherical segment-intersection test and cannot certify every
                # unresolved crossing. No branch is inserted at an approach.
                if len(samples) > 6:
                    near = _distance(np.asarray([item.point_xyz for item in samples[:-5]]),
                                     current.point_xyz, self.radius_km)
                    if near.min() < .75*self.p.step_km:
                        to_seed = float(_distance(current.point_xyz, seed.point_xyz, self.radius_km))
                        end_heading = _transport(next_tangent, current.point_xyz, seed.point_xyz)
                        if to_seed < .75*self.p.step_km and end_heading@start_tangent > .95:
                            closing_distance = float(_distance(previous.point_xyz, seed.point_xyz, self.radius_km))
                            if travelled+closing_distance > self.p.max_branch_length_km+1e-8:
                                return samples, "length_limit", False
                            if closing_distance > 1.5*self.p.step_km:
                                raise RidgeUnavailable("path_self_approach")
                            return samples+[seed], "closed_path", True
                        raise RidgeUnavailable("path_self_approach")
            except RidgeUnavailable as exc:
                return samples, exc.reason, False
            samples.append(current)
            tangent = next_tangent
            travelled += distance
        return samples, "length_limit", False

    def trace(self, seed):
        """Trace both ways from one explicit seed, with resolved stopping reasons.

        Distance along this static field is not elapsed physical time. The
        method has no branching or contact insertion rule. It cannot identify
        every unresolved junction from a single scalar and normal per cell.
        """
        start = self.project(seed)
        right, right_stop, closed = self._branch(start, 1.)
        if closed:
            samples, left_stop, seed_index = right, "not_traced_closed_path", 0
        else:
            left, left_stop, left_closed = self._branch(start, -1., other_branch=right)
            if left_closed:
                samples, right_stop, closed, seed_index = left, "not_traced_closed_path", True, 0
            else:
                samples, seed_index = list(reversed(left[1:]))+right, len(left)-1
        points = np.asarray([sample.point_xyz for sample in samples])
        distance = _distance(points[:-1], points[1:], self.radius_km)
        return RidgePath(points, np.r_[0., np.cumsum(distance)],
            np.asarray([sample.value for sample in samples]),
            np.asarray([sample.transverse_contrast for sample in samples]),
            seed_index, left_stop, right_stop, closed)
