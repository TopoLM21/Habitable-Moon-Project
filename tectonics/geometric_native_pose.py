"""Certified, bounded full SO(3) replay for native spherical geometry.

The rational matrices are disposable exact proper rotations.  Their interval
companions contain the analytic rotations specified by every saved component
of omega and dt.  A clipping predicate must prove its sign from the intervals;
the rational matrix alone does not authorize a physical topology decision.

This native evaluator declares a small-increment domain: each rotation has
theta <= 1/4 radian.  Exact rational alternating series and outward integer
square roots provide bounds without a world-coordinate float projection.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import isqrt

from .geometric_intrinsic_geometry import RotationIncrement


def _fraction(value):
    if type(value) is Fraction:
        return value
    if type(value) is int:
        return Fraction(value)
    raise TypeError("Native intervals require exact Fraction or integer scalars")


@dataclass(frozen=True, slots=True)
class FractionInterval:
    """Closed exact-rational enclosure; a missing hi makes a singleton."""

    lo: Fraction
    hi: Fraction | None = None

    def __post_init__(self):
        low = _fraction(self.lo)
        high = low if self.hi is None else _fraction(self.hi)
        if low > high:
            raise ValueError("Native interval endpoints must be ordered")
        object.__setattr__(self, "lo", low)
        object.__setattr__(self, "hi", high)

    @property
    def midpoint(self):
        return (self.lo + self.hi) / 2

    @property
    def width(self):
        return self.hi - self.lo

    def __add__(self, other):
        other = singleton(other)
        return FractionInterval(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return FractionInterval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -singleton(other)

    def __rsub__(self, other):
        return singleton(other) + -self

    def __mul__(self, other):
        other = singleton(other)
        values = (self.lo * other.lo, self.lo * other.hi,
                  self.hi * other.lo, self.hi * other.hi)
        return FractionInterval(min(values), max(values))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = singleton(other)
        if other.lo <= 0 <= other.hi:
            raise ValueError("Cannot divide by a native interval containing zero")
        reciprocal = FractionInterval(1 / other.hi, 1 / other.lo)
        return self * reciprocal

    def __rtruediv__(self, other):
        return singleton(other) / self


def singleton(value):
    """Return an existing enclosure or enclose an exact scalar."""
    return value if type(value) is FractionInterval else FractionInterval(value)


def _precision(value):
    if type(value) is not int or value <= 0:
        raise ValueError("Native certificate precision must be a positive integer")
    return value


def _precision_bits(precision):
    # 3322/1000 > log2(10); eight extra binary digits are harmless guards.
    return (3322 * _precision(precision) + 999) // 1000 + 8


def _power_two(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def _floor_log_two(value):
    if value <= 0:
        raise ValueError("A binary exponent requires a positive rational")
    numerator, denominator = value.numerator, value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        below = numerator < (denominator << exponent)
    else:
        below = (numerator << -exponent) < denominator
    return exponent - int(below)


def _sqrt_point(value, precision):
    if value < 0:
        raise ValueError("Native square roots require a nonnegative enclosure")
    if not value:
        return FractionInterval(0)
    numerator_root, denominator_root = isqrt(value.numerator), isqrt(value.denominator)
    if (numerator_root * numerator_root == value.numerator
            and denominator_root * denominator_root == value.denominator):
        return FractionInterval(Fraction(numerator_root, denominator_root))
    # Scale relative to sqrt(value), so a positive subnormal-size input never
    # receives an artificial absolute precision or a zero lower bound.
    shift = _precision_bits(precision) - (_floor_log_two(value) // 2)
    scale = _power_two(shift)
    squared = value * scale * scale
    root = isqrt(squared.numerator // squared.denominator)
    return FractionInterval(Fraction(root) / scale, Fraction(root + 1) / scale)


def interval_sqrt(value, precision):
    """Rigorous monotone sqrt, using exact isqrt on a relative binary grid."""
    _precision(precision)
    value = singleton(value)
    if value.lo < 0:
        raise ValueError("Native square roots require a nonnegative enclosure")
    low = _sqrt_point(value.lo, precision)
    high = _sqrt_point(value.hi, precision)
    return FractionInterval(low.lo, high.hi)


def _rounded_interval(value, precision):
    """Outward compression controls denominator growth, never authorizes signs."""
    value = singleton(value)
    maximum = max(abs(value.lo), abs(value.hi))
    if not maximum:
        return FractionInterval(0)
    scale = _power_two(_precision_bits(precision) - _floor_log_two(maximum))
    low, high = value.lo * scale, value.hi * scale
    floor_low = low.numerator // low.denominator
    ceil_high = -((-high.numerator) // high.denominator)
    return FractionInterval(Fraction(floor_low) / scale, Fraction(ceil_high) / scale)


def _rounded_midpoint(value, precision):
    middle = value.midpoint
    if not middle:
        return Fraction(0)
    scale = _power_two(_precision_bits(precision) - _floor_log_two(abs(middle)))
    scaled = middle * scale
    lower = scaled.numerator // scaled.denominator
    remainder = scaled - lower
    rounded = lower + int(remainder > Fraction(1, 2)
                          or (remainder == Fraction(1, 2) and lower % 2))
    return Fraction(rounded) / scale


def _alternating_rotation_terms(u, precision):
    """Enclose cos(sqrt(u)) and sinc(sqrt(u)) by consecutive partial sums."""
    threshold = Fraction(1, 10 ** _precision(precision))

    def terms(sinc):
        partial = term = Fraction(1)
        index = 0
        while True:
            denominator = ((2 * index + 2) * (2 * index + 3)
                           if sinc else (2 * index + 1) * (2 * index + 2))
            following = -term * u / denominator
            next_partial = partial + following
            if abs(following) <= threshold:
                return FractionInterval(min(partial, next_partial),
                                        max(partial, next_partial))
            partial, term, index = next_partial, following, index + 1

    return terms(False), terms(True)


def _quaternion_matrix(quaternion):
    w, x, y, z = quaternion
    norm = w * w + x * x + y * y + z * z
    return ((
        (w * w + x * x - y * y - z * z) / norm,
        2 * (x * y - w * z) / norm,
        2 * (x * z + w * y) / norm,
    ), (
        2 * (x * y + w * z) / norm,
        (w * w - x * x + y * y - z * z) / norm,
        2 * (y * z - w * x) / norm,
    ), (
        2 * (x * z - w * y) / norm,
        2 * (y * z + w * x) / norm,
        (w * w - x * x - y * y + z * z) / norm,
    ))


def _matrix_product(left, right):
    return tuple(tuple(sum((left[i][k] * right[k][j] for k in range(3)), 0)
                       for j in range(3)) for i in range(3))


def _identity(intervals=False):
    return tuple(tuple(FractionInterval(int(i == j)) if intervals else Fraction(int(i == j))
                       for j in range(3)) for i in range(3))


@dataclass(frozen=True, slots=True)
class MatrixCertificate:
    matrix: tuple[tuple[Fraction, ...], ...]
    intervals: tuple[tuple[FractionInterval, ...], ...]


def _rotation_certificate(step, precision):
    half = tuple(Fraction.from_float(value) * Fraction.from_float(step.dt_myr) / 2
                 for value in step.omega_rad_per_myr)
    u = sum((value * value for value in half), Fraction(0))
    if u > Fraction(1, 64):
        raise ValueError("Native certified pose requires every full rotation angle <= 1/4 radian")
    if not u:
        return MatrixCertificate(_identity(), _identity(True))
    cosine, sinc = _alternating_rotation_terms(u, precision + 8)
    quaternion = (cosine,) + tuple(value * sinc for value in half)
    center = _quaternion_matrix(tuple(_rounded_midpoint(value, precision + 8)
                                      for value in quaternion))
    bounds = _quaternion_matrix(quaternion)
    # A center is only scratch geometry; including it in the intervals makes
    # the difference from the true pose directly available to callers.
    bounds = tuple(tuple(_rounded_interval(FractionInterval(
        min(bounds[i][j].lo, center[i][j]), max(bounds[i][j].hi, center[i][j])),
        precision + 8) for j in range(3)) for i in range(3))
    return MatrixCertificate(center, bounds)


@lru_cache(maxsize=64)
def _certify_rotation_history_cached(rotations, precision):
    matrix, intervals = _identity(), _identity(True)
    for step in rotations:
        increment = _rotation_certificate(step, precision)
        matrix = _matrix_product(increment.matrix, matrix)
        product = _matrix_product(increment.intervals, intervals)
        intervals = tuple(tuple(_rounded_interval(FractionInterval(
            min(product[i][j].lo, matrix[i][j]), max(product[i][j].hi, matrix[i][j])),
            precision + 8) for j in range(3)) for i in range(3))
    return MatrixCertificate(matrix, intervals)


def certify_rotation_history(rotations, precision):
    """Replay all saved full increments with a true-pose interval certificate."""
    _precision(precision)
    if type(rotations) is not tuple or any(type(step) is not RotationIncrement for step in rotations):
        raise ValueError("Native pose requires a tuple of validated full rotation increments")
    checked = tuple(RotationIncrement(step.omega_rad_per_myr, step.dt_myr)
                    for step in rotations)
    return _certify_rotation_history_cached(checked, precision)


def _atan_point(value, precision):
    if not value:
        return FractionInterval(0)
    # The direct series is used only after half-angle reduction to <= 1/8.
    threshold = value / (10 ** _precision(precision))
    partial, term, index = value, value, 0
    square = value * value
    while True:
        following = -term * square * (2 * index + 1) / (2 * index + 3)
        next_partial = partial + following
        if abs(following) <= threshold:
            return FractionInterval(min(partial, next_partial), max(partial, next_partial))
        partial, term, index = next_partial, following, index + 1


def interval_atan(value, precision):
    """Rigorous atan of a nonnegative rational enclosure, without libm."""
    _precision(precision)
    reduced = singleton(value)
    if reduced.lo < 0:
        raise ValueError("Native atan requires a nonnegative enclosure")
    multiplier = 1
    while reduced.hi > Fraction(1, 8):
        reduced = reduced / (1 + interval_sqrt(1 + reduced * reduced, precision + 8))
        reduced = _rounded_interval(reduced, precision + 8)
        multiplier *= 2
    lower = _atan_point(reduced.lo, precision + 8)
    upper = _atan_point(reduced.hi, precision + 8)
    return _rounded_interval(FractionInterval(multiplier * lower.lo,
                                              multiplier * upper.hi), precision)


@lru_cache(maxsize=16)
def interval_pi(precision):
    """Rigorous Machin enclosure of pi at the requested relative precision."""
    _precision(precision)
    return _rounded_interval(16 * interval_atan(Fraction(1, 5), precision + 8)
                             - 4 * interval_atan(Fraction(1, 239), precision + 8), precision)
