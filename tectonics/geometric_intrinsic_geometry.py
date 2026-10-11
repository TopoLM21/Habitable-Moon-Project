"""Portable rigid footprints with authoritative body rays and recorded motion.

The body polygon is ordinary, strictly validated saved binary64 geometry.
Rigid motion appends the complete physical angular velocity and timestep; it
does not replace the body with rounded world coordinates.  Those coordinates
are an explicit display projection; external searches also need certified
projection bounds.  They are not
an area measure or an input to material cutting.  Native cuts and inherited
support planes need their own portable lineage before differential transport
can use this representation.

Projection replays the ordered SO(3) history in deterministic Decimal
arithmetic.  Precision grows with the body's smallest represented feature and
the physical increments, so an arbitrary common rotation does not erase a
thin body merely because its world coordinates have a large common offset.
There is no area floor and no change to the legacy world-coordinate contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from functools import lru_cache
import math
from numbers import Real

import numpy as np

from .spherical_polygons import normalize_polygon, polygon_area


VERSION = "spherical-intrinsic-geometry-1"
_UNIT_TOLERANCE = 16. * np.finfo(float).eps


def _real(value, message):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(message)
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(message) from error
    if not math.isfinite(result):
        raise ValueError(message)
    return result


def _triple(values, message):
    try:
        values = tuple(values)
    except TypeError as error:
        raise ValueError(message) from error
    if len(values) != 3:
        raise ValueError(message)
    return tuple(_real(value, message) for value in values)


@dataclass(frozen=True, slots=True)
class RotationIncrement:
    """One full, ordered physical increment; zero angular velocity is valid."""

    omega_rad_per_myr: tuple[float, float, float]
    dt_myr: float

    def __post_init__(self):
        omega = _triple(self.omega_rad_per_myr,
                        "Intrinsic rotation requires a finite explicit three-component omega")
        dt = _real(self.dt_myr, "Intrinsic rotation requires a finite positive real timestep")
        if dt <= 0.:
            raise ValueError("Intrinsic rotation requires a finite positive real timestep")
        # Do not use a squared norm: nonzero subnormal vectors and large
        # components are valid even when their squared binary64 norm is not.
        object.__setattr__(self, "omega_rad_per_myr", omega)
        object.__setattr__(self, "dt_myr", dt)


def _validated_body(values):
    try:
        raw = tuple(_triple(row, "Intrinsic body rays must be finite real triples")
                    for row in values)
    except TypeError as error:
        raise ValueError("Intrinsic body requires a positive convex unit polygon") from error
    checked = normalize_polygon(raw)
    if len(raw) < 3 or checked.shape != (len(raw), 3):
        raise ValueError("Intrinsic body requires a positive polygon without repeated vertices")
    if any(abs(math.hypot(*row)-1.) > _UNIT_TOLERANCE for row in raw):
        raise ValueError("Intrinsic body rays must be unit vectors")
    # Reproduce the persisted fragment's difference-form orientation while
    # preserving each saved binary64 coordinate, rather than renormalizing it.
    array = np.asarray(raw, dtype=float)
    orientation = math.fsum(float(np.dot(array[0], np.cross(
        array[index]-array[0], array[index+1]-array[0])))
        for index in range(1, len(array)-1))
    if orientation < 0.:
        raw = raw[::-1]
    area = polygon_area(raw)
    if not math.isfinite(area) or area <= 0.:
        raise ValueError("Intrinsic body requires strictly positive finite spherical area")
    return raw


@dataclass(frozen=True, slots=True)
class IntrinsicGeometry:
    """Immutable body geometry and complete rigid history, independently saved."""

    body_rays: tuple[tuple[float, float, float], ...]
    rotations: tuple[RotationIncrement, ...] = ()
    version: str = VERSION

    def __post_init__(self):
        if self.version != VERSION:
            raise ValueError("Unsupported intrinsic geometry version")
        body = _validated_body(self.body_rays)
        try:
            history = tuple(self.rotations)
        except TypeError as error:
            raise ValueError("Intrinsic geometry needs an explicit ordered rotation history") from error
        if any(type(step) is not RotationIncrement for step in history):
            raise ValueError("Intrinsic geometry needs validated full rotation increments")
        history = tuple(RotationIncrement(step.omega_rad_per_myr, step.dt_myr) for step in history)
        object.__setattr__(self, "body_rays", body)
        object.__setattr__(self, "rotations", history)

    @property
    def body_polygon(self):
        """Read-only alias for the authoritative saved body rays."""
        return self.body_rays

    @property
    def area_sr(self):
        """Measure the authoritative body, never its world projection."""
        return polygon_area(self.body_rays)

    @property
    def projection_precision(self):
        return _projection_precision(self.body_rays, self.rotations, self.area_sr)

    def rotated(self, omega_rad_per_myr, dt_myr):
        step = RotationIncrement(omega_rad_per_myr, dt_myr)
        return IntrinsicGeometry(self.body_rays, self.rotations+(step,), self.version)

    def decimal_world_rays(self):
        """Normalized world rays at adaptive Decimal precision, without float feedback."""
        precision = self.projection_precision
        with localcontext(Context(prec=precision)):
            rays = tuple(tuple(Decimal.from_float(value) for value in row)
                         for row in self.body_rays)
            # Positive scaling of a ray changes neither its area nor its edge
            # planes.  Normalize only before the replay and after the full
            # history, retaining high precision throughout every increment.
            rays = tuple(_decimal_unit(row) for row in rays)
            for step in self.rotations:
                omega = tuple(Decimal.from_float(value) for value in step.omega_rad_per_myr)
                magnitude = sum(value*value for value in omega).sqrt()
                if magnitude == 0:
                    continue
                axis = tuple(value/magnitude for value in omega)
                angle = magnitude*Decimal.from_float(step.dt_myr)
                sine, cosine, haversine = _decimal_rotation_terms(angle, precision)
                moved = []
                for ray in rays:
                    crossing = _decimal_cross(axis, ray)
                    projection = sum(a*b for a, b in zip(axis, ray))
                    moved.append(tuple(ray[index]*cosine+crossing[index]*sine
                        +projection*axis[index]*haversine for index in range(3)))
                rays = tuple(moved)
            return tuple(_decimal_unit(row) for row in rays)

    def world_polygon(self):
        """Final binary64 projection; thin projected area may disagree with the body."""
        return tuple(tuple(float(value) for value in row) for row in self.decimal_world_rays())


def _projection_precision(body, history, area):
    # Area and represented coordinate differences both matter: the small
    # feature can be a width near a pole or a cancellation in a long triangle.
    # Eighty guard digits cover accumulated replay arithmetic; grow them with
    # history length rather than imposing a supported feature size.
    depth = max(0, -Decimal.from_float(area).adjusted())
    for row in body:
        for value in row:
            if value:
                depth = max(depth, -Decimal.from_float(abs(value)).adjusted())
    # Every binary64 unit coordinate is an exact decimal with fewer than 1100
    # significant digits.  Use a fixed independent context for exact pairwise
    # differences; a caller's Decimal precision/rounding cannot select a pose.
    with localcontext(Context(prec=1100)):
        for first in body:
            for second in body:
                for left, right in zip(first, second):
                    if left != right:
                        difference = abs(Decimal.from_float(left)-Decimal.from_float(right))
                        depth = max(depth, -difference.adjusted())
    angle_digits = 0
    with localcontext(Context(prec=40)):
        for step in history:
            values = [Decimal.from_float(abs(value)) for value in step.omega_rad_per_myr if value]
            if not values:
                continue
            largest = max(value.adjusted() for value in values)
            smallest = min(value.adjusted() for value in values)
            depth = max(depth, largest-smallest)
            dt = Decimal.from_float(step.dt_myr)
            # Three components add at most one decimal order to the norm.
            angle_digits = max(angle_digits, max(0, largest+dt.adjusted()+2))
    return max(80, depth+80+angle_digits+len(str(max(1, len(history)))))


def _decimal_unit(ray):
    magnitude = sum(value*value for value in ray).sqrt()
    if magnitude == 0:
        raise ValueError("Intrinsic projection produced a zero ray")
    return tuple(value/magnitude for value in ray)


def _decimal_cross(first, second):
    return (first[1]*second[2]-first[2]*second[1],
            first[2]*second[0]-first[0]*second[2],
            first[0]*second[1]-first[1]*second[0])


@lru_cache(maxsize=16)
def _decimal_pi(precision):
    """Gauss-Legendre pi with guard digits; no platform libm dependency."""
    with localcontext(Context(prec=precision+16)) as context:
        one = Decimal(1)
        a, b, t, multiplier = one, one/Decimal(2).sqrt(), one/4, one
        for _ in range(max(4, precision.bit_length()+2)):
            next_a = (a+b)/2
            next_b = (a*b).sqrt()
            t -= multiplier*(a-next_a)**2
            a, b, multiplier = next_a, next_b, multiplier*2
        result = (a+b)**2/(4*t)
        context.prec = precision
        return +result


def _decimal_rotation_terms(angle, precision):
    """Deterministic Rodrigues terms, including a cancellation-free 1-cos."""
    with localcontext(Context(prec=precision+16)) as context:
        pi = _decimal_pi(precision+16)
        period = 2*pi
        turns = (angle/period).to_integral_value(rounding=ROUND_HALF_EVEN)
        reduced = angle-turns*period
        half = reduced/2
        squared = half*half
        sine_half, cosine_half = half, Decimal(1)
        sine_term, cosine_term = half, Decimal(1)
        index = 1
        while True:
            sine_term *= -squared/Decimal((2*index)*(2*index+1))
            cosine_term *= -squared/Decimal((2*index-1)*(2*index))
            next_sine, next_cosine = sine_half+sine_term, cosine_half+cosine_term
            if next_sine == sine_half and next_cosine == cosine_half:
                break
            sine_half, cosine_half = next_sine, next_cosine
            index += 1
        sine = 2*sine_half*cosine_half
        haversine = 2*sine_half*sine_half
        cosine = cosine_half*cosine_half-sine_half*sine_half
        context.prec = precision
        return +sine, +cosine, +haversine


def intrinsic_geometry_to_dict(geometry):
    if type(geometry) is not IntrinsicGeometry:
        raise ValueError("Expected validated intrinsic geometry")
    geometry = IntrinsicGeometry(geometry.body_rays, geometry.rotations, geometry.version)
    return dict(version=geometry.version, body_rays=[list(row) for row in geometry.body_rays],
        rotations=[rotation_increment_to_dict(step) for step in geometry.rotations])


def rotation_increment_to_dict(step):
    if type(step) is not RotationIncrement:
        raise ValueError("Expected validated full intrinsic rotation increment")
    step = RotationIncrement(step.omega_rad_per_myr, step.dt_myr)
    return dict(omega_rad_per_myr=list(step.omega_rad_per_myr), dt_myr=step.dt_myr)


def rotation_increment_from_dict(data):
    if (not isinstance(data, dict) or set(data) != {"omega_rad_per_myr", "dt_myr"}
            or not isinstance(data["omega_rad_per_myr"], list)):
        raise ValueError("Intrinsic rotation needs an explicit full increment schema")
    return RotationIncrement(data["omega_rad_per_myr"], data["dt_myr"])


def intrinsic_geometry_from_dict(data):
    if not isinstance(data, dict) or set(data) != {"version", "body_rays", "rotations"}:
        raise ValueError("Intrinsic geometry needs an explicit supported schema")
    if (not isinstance(data["body_rays"], list) or not isinstance(data["rotations"], list)
            or any(not isinstance(row, list) for row in data["body_rays"])):
        raise ValueError("Intrinsic geometry needs explicit body and ordered history arrays")
    history = [rotation_increment_from_dict(step) for step in data["rotations"]]
    return IntrinsicGeometry(data["body_rays"], tuple(history), data["version"])
