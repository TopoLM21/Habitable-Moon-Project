"""Fixed material geometry for an experimental, two-front crack history.

A supplied simple open spherical polyline is a *possible* growth support,
not a fracture prediction. A finite seed notch and its subsequent active
interval are explicit inputs. Nothing here chooses a seed, cuts material
cells, spends energy, or converts a ridge into a physical crack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from numbers import Real

import numpy as np


_ANGLE_TOL = 1e-11


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _immutable(values):
    """Own an immutable byte buffer: callers cannot re-enable array writes."""
    values = np.asarray(values, dtype="<f8")
    return np.frombuffer(values.tobytes(), dtype="<f8").reshape(values.shape)


def _angle(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a*b, axis=-1))


def _on_arc(point, first, last, length):
    return abs(float(_angle(first, point)+_angle(point, last))-length) <= _ANGLE_TOL


def _validate_simple(points, angles):
    normals = np.cross(points[:-1], points[1:])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    for i in range(len(angles)):
        for j in range(i+1, len(angles)):
            adjacent = j == i+1
            common = points[j] if adjacent else None
            cross = np.cross(normals[i], normals[j])
            norm = np.linalg.norm(cross)
            if norm <= _ANGLE_TOL:
                # Coincident great circles: overlap or a nonadjacent endpoint
                # touch is invalid. Adjacent segments may share only one end.
                for point in (points[i], points[i+1], points[j], points[j+1]):
                    if adjacent and _angle(point, common) <= _ANGLE_TOL:
                        continue
                    if (_on_arc(point, points[i], points[i+1], angles[i])
                            and _on_arc(point, points[j], points[j+1], angles[j])):
                        raise ValueError("Crack support overlaps or touches itself")
            else:
                intersection = cross/norm
                for point in (intersection, -intersection):
                    if adjacent and _angle(point, common) <= _ANGLE_TOL:
                        continue
                    if (_on_arc(point, points[i], points[i+1], angles[i])
                            and _on_arc(point, points[j], points[j+1], angles[j])):
                        raise ValueError("Crack support intersects or touches itself")


@dataclass(frozen=True, eq=False)
class ReferenceCrackPath:
    """Simple open minor-arc polyline on the reference material sphere.

    Points are finite unit vectors; lengths exposed by this module are metres.
    Both arrays are copied into immutable storage. ``fingerprint`` identifies
    this exact ordered geometry and radius, not a rotation-equivalence class.
    """

    points_xyz: np.ndarray
    radius_km: float
    arclength_m: np.ndarray = field(init=False, repr=False)
    fingerprint: str = field(init=False)

    def __post_init__(self):
        radius = _number(self.radius_km, "Crack radius")
        points = np.asarray(self.points_xyz)
        if (radius <= 0 or not np.isfinite(radius*1000.)
                or points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2
                or not np.issubdtype(points.dtype, np.number) or np.iscomplexobj(points)):
            raise ValueError("Crack support needs positive radius and at least two unit vectors")
        points = np.asarray(points, dtype=float)
        if (not np.isfinite(points).all()
                or not np.allclose(np.linalg.norm(points, axis=1), 1., atol=1e-12, rtol=0)):
            raise ValueError("Crack support points must be finite unit vectors")
        angles = _angle(points[:-1], points[1:])
        if np.any(angles <= _ANGLE_TOL) or np.any(angles >= np.pi-_ANGLE_TOL):
            raise ValueError("Crack segments must be distinct, non-antipodal minor arcs")
        _validate_simple(points, angles)
        arclength = np.r_[0., np.cumsum(angles)*(radius*1000.)]
        if not np.isfinite(arclength).all() or np.any(np.diff(arclength) <= 0):
            raise ValueError("Crack arclength must be finite and strictly increasing")
        points = _immutable(points)
        arclength = _immutable(arclength)
        identity = hashlib.sha256(b"reference-crack-path-0.1\0")
        identity.update(np.asarray([radius], dtype="<f8").tobytes())
        identity.update(points.tobytes())
        object.__setattr__(self, "radius_km", radius)
        object.__setattr__(self, "points_xyz", points)
        object.__setattr__(self, "arclength_m", arclength)
        object.__setattr__(self, "fingerprint", identity.hexdigest())

    @classmethod
    def from_ridge(cls, ridge, radius_km):
        """Copy a diagnostic ridge's geometry, without asserting activation."""
        if ridge.closed:
            raise ValueError("Closed ridge supports need a separate topology model")
        return cls(ridge.points_xyz, radius_km)

    @property
    def length_m(self):
        return float(self.arclength_m[-1])

    def point_at(self, arclength_m):
        """Interpolate unit vectors at scalar or array material coordinates."""
        coordinate = np.asarray(arclength_m)
        if (not np.issubdtype(coordinate.dtype, np.number) or np.iscomplexobj(coordinate)
                or not np.isfinite(coordinate).all()
                or np.any(coordinate < 0) or np.any(coordinate > self.length_m)):
            raise ValueError("Crack coordinate must lie within the finite reference support")
        coordinate = np.asarray(coordinate, dtype=float)
        index = np.searchsorted(self.arclength_m, coordinate, side="right")-1
        index = np.minimum(index, len(self.points_xyz)-2)
        first, last = self.points_xyz[index], self.points_xyz[index+1]
        length = self.arclength_m[index+1]-self.arclength_m[index]
        fraction = (coordinate-self.arclength_m[index])/length
        angle = length/(self.radius_km*1000.)
        tangent = last-np.sum(first*last, axis=-1)[..., None]*first
        tangent /= np.linalg.norm(tangent, axis=-1)[..., None]
        travelled = fraction*angle
        result = np.cos(travelled)[..., None]*first+np.sin(travelled)[..., None]*tangent
        result /= np.linalg.norm(result, axis=-1)[..., None]
        # Preserve every supplied material vertex exactly, including endpoints.
        result = np.where((fraction == 0)[..., None], first, result)
        result = np.where((fraction == 1)[..., None], last, result)
        return result


@dataclass(frozen=True)
class CrackInterval:
    """An irreversible active interval on a separately identified support."""

    left_m: float
    right_m: float

    def __post_init__(self):
        left = _number(self.left_m, "Left crack front")
        right = _number(self.right_m, "Right crack front")
        if not 0 <= left < right:
            raise ValueError("Crack interval must have nonnegative fronts and positive length")
        object.__setattr__(self, "left_m", left)
        object.__setattr__(self, "right_m", right)

    @property
    def length_m(self):
        return self.right_m-self.left_m

    def validate(self, path):
        if not isinstance(path, ReferenceCrackPath) or self.right_m > path.length_m:
            raise ValueError("Crack interval extends beyond its reference support")

    def grow(self, path, *, left_m=None, right_m=None):
        """Return a new interval; no retreat, clipping or topology changes."""
        self.validate(path)
        grown = CrackInterval(self.left_m if left_m is None else left_m,
                              self.right_m if right_m is None else right_m)
        grown.validate(path)
        if grown.left_m > self.left_m or grown.right_m < self.right_m:
            raise ValueError("Crack fronts cannot retreat")
        return grown


def initial_interval(path, seed_m, notch_length_m):
    """Build an explicitly supplied centred seed notch without clipping it."""
    seed = _number(seed_m, "Crack seed coordinate")
    length = _number(notch_length_m, "Crack seed notch length")
    if length <= 0:
        raise ValueError("Crack seed notch must have positive length")
    interval = CrackInterval(seed-length/2, seed+length/2)
    interval.validate(path)
    return interval
