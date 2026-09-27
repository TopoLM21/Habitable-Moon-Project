"""Mesh-independent weak-band fixtures for geometric ridge extraction.

These fields prescribe a crack *location* for a geometric benchmark. They are
not damage solutions or evidence that a planetary shell will fracture there.
All lengths are physical kilometres. ``width_km`` is the Gaussian standard
deviation, not a cell count or the full support width. A smooth cutoff makes
the transverse support exactly zero beyond three standard deviations.

The transverse plane normals are unit vectors in the spherical tangent plane,
in world coordinates. They are not components in a mesh's chord-face frame.
An extractor that needs those components must perform its own projection.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


def _unit_positions(positions):
    """Accept (..., 3) vectors, returning their directions without mutation."""
    result = np.asarray(positions, dtype=float)
    if result.ndim < 1 or result.shape[-1] != 3 or not np.isfinite(result).all():
        raise ValueError("Positions must be finite vectors with shape (..., 3)")
    length = np.linalg.norm(result, axis=-1)
    if np.any(length <= 0):
        raise ValueError("Position vectors must have positive length")
    return result / length[..., None]


def _smooth_cutoff(coordinate, plateau, support):
    """One within plateau, zero outside support, with a C2 transition."""
    argument = np.clip((np.abs(coordinate) - plateau) / (support - plateau), 0., 1.)
    # Evaluate 1 - smootherstep without small negative roundoff at its ends.
    value = 1. - argument**3 * (10. + argument * (-15. + 6.*argument))
    return np.where(np.abs(coordinate) >= support, 0., np.clip(value, 0., 1.))


@dataclass(frozen=True)
class ReferenceSamples:
    """Scalar samples, compact-support mask, and transverse tangent normals."""

    values: np.ndarray
    active: np.ndarray
    plane_normals: np.ndarray


@dataclass(frozen=True)
class RidgeReference:
    """An analytic band sampled independently of a triangular mesh.

    ``great_circle`` is an equatorial segment centred on +x, with transverse
    normal toward +z. ``small_circle`` is a closed circle about +z at the
    requested polar angle, with transverse normal toward increasing polar
    angle. ``uniform`` has no ridge: its scalar is constant in a cap about +x.
    Its support edge is deliberately kept away from the seed.

    ``rotation`` maps these reference directions into world coordinates. To
    test orientation relative to a fixed mesh, rotate this field alone. To
    test coordinate objectivity, rotate both the field and the mesh points.
    """

    kind: str = "great_circle"
    radius_km: float = 6371.
    width_km: float = 600.
    half_length_km: float = 4500.
    taper_km: float = 600.
    small_circle_radius_degrees: float = 40.
    uniform_cap_degrees: float = 18.
    uniform_value: float = .8
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3), repr=False,
                                compare=False)

    def __post_init__(self):
        if self.kind not in {"great_circle", "small_circle", "uniform"}:
            raise ValueError("Reference kind must be great_circle, small_circle or uniform")
        numeric = [self.radius_km, self.width_km, self.half_length_km,
                   self.taper_km, self.small_circle_radius_degrees,
                   self.uniform_cap_degrees, self.uniform_value]
        if not np.isfinite(numeric).all():
            raise ValueError("Reference parameters must be finite")
        if self.radius_km <= 0 or self.width_km <= 0:
            raise ValueError("Radius and Gaussian width must be positive")
        if not 0 < self.taper_km < self.half_length_km < np.pi*self.radius_km:
            raise ValueError("Require 0 < taper < half length < pi * radius")
        if not 0 < self.small_circle_radius_degrees < 180:
            raise ValueError("Small-circle polar angle must be between 0 and 180 degrees")
        if not 0 < self.uniform_cap_degrees < 90 or not 0 < self.uniform_value <= 1:
            raise ValueError("Uniform cap must be below 90 degrees with value in (0, 1]")
        # Active normals must remain defined throughout the analytic support.
        if self.kind == "great_circle" and 3*self.width_km >= .5*np.pi*self.radius_km:
            raise ValueError("Great-circle support must exclude both poles")
        pole_distance = min(self.small_circle_radius_degrees,
                            180. - self.small_circle_radius_degrees)
        if (self.kind == "small_circle"
                and 3*self.width_km >= np.deg2rad(pole_distance)*self.radius_km):
            raise ValueError("Small-circle support must exclude both poles")
        rotation = np.asarray(self.rotation, dtype=float)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0., atol=1e-12)
                or not np.isclose(np.linalg.det(rotation), 1., rtol=0., atol=1e-12)):
            raise ValueError("Rotation must be a proper orthogonal 3 by 3 matrix")
        rotation = rotation.copy()
        rotation.setflags(write=False)
        object.__setattr__(self, "rotation", rotation)

    @property
    def seed(self):
        """An exact ridge point; the control uses the centre of its uniform cap."""
        if self.kind == "small_circle":
            angle = np.deg2rad(self.small_circle_radius_degrees)
            local = np.array([np.sin(angle), 0., np.cos(angle)])
        else:
            local = np.array([1., 0., 0.])
        return self.rotation @ local

    def rotated(self, rotation):
        """Apply another world rotation to this field, leaving this object intact."""
        return replace(self, rotation=np.asarray(rotation, dtype=float) @ self.rotation)

    def _local_positions(self, positions):
        return _unit_positions(positions) @ self.rotation

    def signed_distance_km(self, positions):
        """Signed transverse distance to the supporting great or small circle.

        For the open great-circle segment this ignores its longitudinal ends;
        use ``distance_km`` for distance to the actual finite centreline.
        A uniform field does not prescribe a centreline or a distance to one.
        """
        if self.kind == "uniform":
            raise ValueError("Uniform reference has no prescribed ridge")
        local = self._local_positions(positions)
        vertical = np.clip(local[..., 2], -1., 1.)
        if self.kind == "great_circle":
            return self.radius_km*np.arcsin(vertical)
        return self.radius_km*(np.arccos(vertical)
                               - np.deg2rad(self.small_circle_radius_degrees))

    def distance_km(self, positions):
        """Exact shortest spherical distance to the prescribed finite curve."""
        cross = np.abs(self.signed_distance_km(positions))
        if self.kind == "small_circle":
            return cross
        local = self._local_positions(positions)
        longitude = np.arctan2(local[..., 1], local[..., 0])
        half_angle = self.half_length_km/self.radius_km
        endpoint_longitude = np.clip(longitude, -half_angle, half_angle)
        nearest_dot = (local[..., 0]*np.cos(endpoint_longitude)
                       + local[..., 1]*np.sin(endpoint_longitude))
        endpoint_distance = self.radius_km*np.arccos(np.clip(nearest_dot, -1., 1.))
        return np.where(np.abs(longitude) <= half_angle, cross, endpoint_distance)

    def sample(self, positions):
        """Sample the analytic scalar and plane directions at arbitrary points."""
        local = self._local_positions(positions)
        vertical = np.clip(local[..., 2], -1., 1.)
        if self.kind == "uniform":
            active = local[..., 0] >= np.cos(np.deg2rad(self.uniform_cap_degrees))
            values = np.where(active, self.uniform_value, 0.)
        else:
            if self.kind == "great_circle":
                transverse = self.radius_km*np.arcsin(vertical)
            else:
                transverse = self.radius_km*(np.arccos(vertical)
                                             - np.deg2rad(self.small_circle_radius_degrees))
            scaled = transverse/self.width_km
            values = np.exp(-.5*scaled**2)*_smooth_cutoff(scaled, 2., 3.)
            if self.kind == "great_circle":
                longitudinal = self.radius_km*np.arctan2(local[..., 1], local[..., 0])
                values *= _smooth_cutoff(longitudinal,
                                         self.half_length_km-self.taper_km,
                                         self.half_length_km)
            active = values > 0.

        axis = np.array([0., 0., 1.])
        normals = axis - vertical[..., None]*local
        if self.kind == "small_circle":
            normals = -normals
        lengths = np.linalg.norm(normals, axis=-1)
        normals = np.divide(normals, lengths[..., None], out=np.zeros_like(normals),
                            where=(active & (lengths > 1e-14))[..., None])
        return ReferenceSamples(values=values, active=active,
                                plane_normals=normals @ self.rotation.T)

    def centerline(self, count=401):
        """Exact curve points including endpoints (or a repeated closed start)."""
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 2:
            raise ValueError("Centerline count must be an integer of at least two")
        if self.kind == "uniform":
            return np.empty((0, 3))
        if self.kind == "great_circle":
            half_angle = self.half_length_km/self.radius_km
            angle = np.linspace(-half_angle, half_angle, count)
            local = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(count)))
        else:
            polar = np.deg2rad(self.small_circle_radius_degrees)
            angle = np.linspace(0., 2*np.pi, count)
            local = np.column_stack((np.sin(polar)*np.cos(angle),
                                     np.sin(polar)*np.sin(angle),
                                     np.full(count, np.cos(polar))))
        return local @ self.rotation.T

    def metadata(self):
        """JSON-compatible parameters and interpretation for benchmark records."""
        length = None
        if self.kind == "great_circle":
            length = 2*self.half_length_km
        elif self.kind == "small_circle":
            length = 2*np.pi*self.radius_km*np.sin(np.deg2rad(self.small_circle_radius_degrees))
        return {
            "kind": self.kind, "radius_km": self.radius_km,
            "gaussian_sigma_km": self.width_km,
            "transverse_support_half_width_km": 3*self.width_km,
            "half_length_km": self.half_length_km, "taper_km": self.taper_km,
            "small_circle_radius_degrees": self.small_circle_radius_degrees,
            "uniform_cap_degrees": self.uniform_cap_degrees,
            "uniform_value": self.uniform_value,
            "rotation": self.rotation.tolist(), "seed": self.seed.tolist(),
            "centerline_length_km": length,
            "closed": self.kind == "small_circle",
            "interpretation": "Prescribed geometric reference; not a fracture simulation",
        }


def great_circle_reference(**parameters):
    return RidgeReference(kind="great_circle", **parameters)


def small_circle_reference(**parameters):
    return RidgeReference(kind="small_circle", **parameters)


def uniform_reference(**parameters):
    return RidgeReference(kind="uniform", **parameters)
