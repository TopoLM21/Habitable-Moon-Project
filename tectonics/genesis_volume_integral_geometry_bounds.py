"""Source radial and literal Gamma moment enclosures on a fixed native chart.

The source sphere roots are enclosed by integer square roots of exact rational
operands.  Their intersection with the original cohort bounds is transported
to the caller's represented ``lambda = midpoint + halfwidth * xi`` chart.
Rational antiderivatives then enclose ordinary, signed Gamma and separate bank
moments.  Endpoint migration is included; the native clipping endpoints and
native Gamma classification are not treated as source truth.

Only the small scalar moment vectors use rational arithmetic.  The returned
arrays are directed native leaves for tensor contraction.  This proves no
angular convergence, material release, work funding or planetary advance.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from types import MappingProxyType

import numpy as np

from analysis.genesis_volume_operand_validation.independent_geometry_bounds import (
    GeometryBoundRefusal, _source_interval, rational, source_line,
)
from tectonics.genesis_volume_operand_bounds import DirectedInterval, rational_enclosure


SCOPE = "source_radial_gamma_moments_on_literal_native_chart_no_angular_certificate"
MAX_DEGREE = 16


class IntegralGeometryRefusal(ValueError):
    """A source support or a directed moment cannot be certified."""


def _require(condition, message):
    if not condition:
        raise IntegralGeometryRefusal(message)


def _ratio(value):
    value = Fraction(value)
    return [str(value.numerator), str(value.denominator)]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_digest(line):
    return _digest({key: [[_ratio(v) for v in row] for row in line[key]]
                    for key in ("C", "D")} | {
        key: [_ratio(v) for v in line[key]] for key in ("center", "normal", "bounds")
    } | {key: _ratio(line[key]) for key in ("offset", "radius")})


def _chart(source_binding, N, native_interval):
    n = tuple(map(rational, np.asarray(N).reshape(-1)))
    _require(len(n) == 3, "Material N shape differs")
    if native_interval is None:
        midpoint, halfwidth = None, Fraction()
    else:
        represented_n = tuple(map(rational, np.asarray(native_interval.barycentric).reshape(-1)))
        _require(n == represented_n, "Native interval and source N differ")
        midpoint = rational(native_interval.lambda_mid)
        halfwidth = rational(native_interval.lambda_halfwidth)
        _require(halfwidth > 0, "Native chart halfwidth is not positive")
    binding = _digest(dict(source=source_binding, N=[_ratio(v) for v in n],
        midpoint=None if midpoint is None else _ratio(midpoint), halfwidth=_ratio(halfwidth)))
    return n, midpoint, halfwidth, binding


def _primitive(x, degree, mode, root, slope, constant):
    """A continuous antiderivative, evaluated without floating cancellation."""
    power = degree + 1
    ordinary = x**power / power
    if mode == "ordinary":
        return ordinary
    if slope == 0:
        _require(constant != 0, "Literal Gamma contains the whole source line")
        sign = 1 if constant > 0 else -1
        if mode == "signed":
            return sign * ordinary / 2
        return ordinary if mode == ("positive" if sign > 0 else "negative") else Fraction()
    right = (x**power-root**power)/power if x > root else Fraction()
    left = (x**power-root**power)/power if x < root else Fraction()
    positive, negative = (right, left) if slope > 0 else (left, right)
    if mode == "signed":
        return (positive-negative)/2
    return positive if mode == "positive" else negative


def _primitive_range(bounds, degree, mode, root, slope, constant):
    points = [bounds[0], bounds[1]]
    # All derivative sign changes occur at zero or the literal Gamma root.
    if bounds[0] <= 0 <= bounds[1]:
        points.append(Fraction())
    if root is not None and bounds[0] <= root <= bounds[1]:
        points.append(root)
    values = [_primitive(x, degree, mode, root, slope, constant) for x in points]
    return min(values), max(values)


def _moment_vector(low, high, halfwidth, mode, root, slope, constant):
    result = []
    for degree in range(MAX_DEGREE+1):
        left = _primitive_range(low, degree, mode, root, slope, constant)
        right = _primitive_range(high, degree, mode, root, slope, constant)
        lower, upper = halfwidth*(right[0]-left[1]), halfwidth*(right[1]-left[0])
        # Even ordinary and bank integrands are nonnegative everywhere.  This
        # intersects a mathematically known sign, not an empirical error floor.
        if degree % 2 == 0 and mode != "signed":
            lower = max(Fraction(), lower)
        result.append((lower, upper))
    return tuple(result)


def _directed_vector(bounds):
    lower, upper = [], []
    for lo, hi in bounds:
        lo_leaf, hi_leaf = rational_enclosure(lo), rational_enclosure(hi)
        lower.append(lo_leaf.lower)
        upper.append(hi_leaf.upper)
    return DirectedInterval(lower, upper)


def _degrees(max_degree):
    _require(type(max_degree) is int and 0 <= max_degree <= MAX_DEGREE,
             "Moment degree must be an integer from 0 through 16")
    return slice(0, max_degree+1)


def _side(side_power):
    _require(type(side_power) is int and side_power in (0, 1, 2), "Side power must be 0, 1 or 2")
    return side_power


@dataclass(frozen=True, slots=True)
class SourceGeometryMoments:
    """Read-only moment vectors and the exact operands that bind them."""
    source_binding_sha256: str
    line_operand_sha256: str
    barycentric: tuple
    lambda_mid: Fraction | None
    lambda_halfwidth: Fraction
    empty: bool
    source_lambda_bounds: tuple | None
    source_xi_bounds: tuple | None
    literal_phi_coefficients: tuple | None
    literal_gamma_root_xi: Fraction | None
    bank_widths_xi: tuple | None
    rational_moment_enclosures: tuple
    rational_bank_moment_enclosures: tuple | None
    _moments: tuple
    _bank_moments: tuple | None
    binary_places: int
    scope: str = SCOPE
    source_geometry_error_bounds_certified: bool = True
    source_endpoint_migration_included: bool = True
    literal_source_gamma_used: bool = True
    angular_accuracy_certified: bool = False
    physical_creation_admitted: bool = False
    physical_release_admitted: bool = False
    available_for_creation_funding: bool = False
    automated_goal_enabled: bool = False

    def moments(self, max_degree=MAX_DEGREE, side_power=0):
        index, cut = _side(side_power), _degrees(max_degree)
        result = self._moments[index]
        _require(result is not None, "Literal Gamma contains the whole source line")
        return result[cut]

    def bank_moments(self, max_degree=MAX_DEGREE, side_sign=1):
        _require(type(side_sign) is int and side_sign in (-1, 1), "Bank sign must be -1 or +1")
        cut = _degrees(max_degree)
        _require(self._bank_moments is not None, "Literal Gamma contains the whole source line")
        return self._bank_moments[0 if side_sign < 0 else 1][cut]

    def metadata(self):
        def enclosures(value):
            return None if value is None else [dict(lower=_ratio(lo), upper=_ratio(hi)) for lo, hi in value]
        return dict(scope=self.scope, empty=self.empty,
            source_binding_sha256=self.source_binding_sha256,
            line_operand_sha256=self.line_operand_sha256, binary_places=self.binary_places,
            source_lambda_endpoint_enclosures=enclosures(self.source_lambda_bounds),
            source_xi_endpoint_enclosures=enclosures(self.source_xi_bounds),
            literal_phi_coefficients=None if self.literal_phi_coefficients is None else
                [_ratio(v) for v in self.literal_phi_coefficients],
            literal_gamma_root_xi=None if self.literal_gamma_root_xi is None else _ratio(self.literal_gamma_root_xi),
            negative_positive_bank_width_enclosures_xi=enclosures(self.bank_widths_xi),
            source_geometry_error_bounds_certified=True, source_endpoint_migration_included=True,
            literal_source_gamma_used=True, angular_accuracy_certified=False,
            physical_creation_admitted=False, physical_release_admitted=False,
            available_for_creation_funding=False, automated_goal_enabled=False)


@dataclass(frozen=True, slots=True)
class PreparedIntegralGeometry:
    _raw: object
    source_binding_sha256: str
    binary_places: int

    def matches(self, certificate, N, native_interval):
        if type(certificate) is not SourceGeometryMoments:
            return False
        try:
            _, _, _, binding = _chart(self.source_binding_sha256, N, native_interval)
            recomputed = self.evaluate(N, native_interval)
        except (IntegralGeometryRefusal, GeometryBoundRefusal, AttributeError):
            return False
        return (certificate.source_binding_sha256 == self.source_binding_sha256 and
                certificate.line_operand_sha256 == binding and
                certificate.binary_places == self.binary_places and
                all(getattr(certificate, name) == getattr(recomputed, name)
                    for name in certificate.__dataclass_fields__
                    if name not in ("_moments", "_bank_moments")) and
                len(certificate._moments) == len(recomputed._moments) and
                all((a is None and b is None) or (a is not None and b is not None and
                    np.array_equal(a.lower, b.lower) and np.array_equal(a.upper, b.upper))
                    for a, b in zip(certificate._moments, recomputed._moments)) and
                ((certificate._bank_moments is None and recomputed._bank_moments is None) or
                 (certificate._bank_moments is not None and recomputed._bank_moments is not None and
                  len(certificate._bank_moments) == len(recomputed._bank_moments) and
                  all(np.array_equal(a.lower, b.lower) and np.array_equal(a.upper, b.upper)
                      for a, b in zip(certificate._bank_moments, recomputed._bank_moments)))))

    def evaluate(self, N, native_interval):
        try:
            raw = dict(self._raw)
            raw["section"] = dict(raw["section"])
            line = source_line(raw, N)
            truth = _source_interval(line, self.binary_places)
            n, midpoint, halfwidth, binding = _chart(self.source_binding_sha256, N, native_interval)
        except GeometryBoundRefusal as error:
            raise IntegralGeometryRefusal(str(error)) from error
        _require((truth is None) == (native_interval is None),
                 "Source/native positive support classification differs")
        if truth is None:
            exact_zero = tuple((Fraction(), Fraction()) for _ in range(MAX_DEGREE+1))
            zero = _directed_vector(exact_zero)
            return SourceGeometryMoments(self.source_binding_sha256, binding, n, None, Fraction(),
                True, None, None, None, None, None, (exact_zero,)*3, (exact_zero,)*2,
                (zero,)*3, (zero,)*2, self.binary_places)
        low = tuple((x-midpoint)/halfwidth for x in truth["low"])
        high = tuple((x-midpoint)/halfwidth for x in truth["high"])
        _require(low[1] < high[0], "Source interval ordering unresolved on fixed chart")
        phi0 = line["slope"]*midpoint+line["intercept"]
        phi1 = line["slope"]*halfwidth
        root = -phi0/phi1 if phi1 else None
        ordinary = _moment_vector(low, high, halfwidth, "ordinary", root, phi1, phi0)
        if phi0 == 0 and phi1 == 0:
            moments, banks, widths = (ordinary, None, None), None, None
        else:
            signed = _moment_vector(low, high, halfwidth, "signed", root, phi1, phi0)
            squared = tuple((lo/4, hi/4) for lo, hi in ordinary)
            moments = (ordinary, signed, squared)
            banks = tuple(_moment_vector(low, high, halfwidth, mode, root, phi1, phi0)
                          for mode in ("negative", "positive"))
            if phi1:
                left = (max(Fraction(), min(high[0], root)-low[1]),
                        max(Fraction(), min(high[1], root)-low[0]))
                right = (max(Fraction(), high[0]-max(low[1], root)),
                         max(Fraction(), high[1]-max(low[0], root)))
                widths = (left, right) if phi1 > 0 else (right, left)
                # Exact inequalities classify positive source banks even when
                # the irrational endpoint enclosure straddles the Gamma root.
                lambda_root = -line["intercept"]/line["slope"]
                b, a = line["bounds"]; c = line["closest"]
                square = line["margin"]/line["uu"]
                left_positive = b < lambda_root and (lambda_root >= c or (c-lambda_root)**2 < square)
                right_positive = a > lambda_root and (lambda_root <= c or (lambda_root-c)**2 < square)
                positive = (left_positive, right_positive) if phi1 > 0 else (right_positive, left_positive)
                _require(all(not exists or width[0] > 0 for exists, width in zip(positive, widths)),
                         "Positive source Gamma bank unresolved at rational root precision")
            else:
                width = (high[0]-low[1], high[1]-low[0])
                zero = (Fraction(), Fraction())
                widths = (zero, width) if phi0 > 0 else (width, zero)
        directed = tuple(None if vector is None else _directed_vector(vector) for vector in moments)
        directed_banks = None if banks is None else tuple(_directed_vector(v) for v in banks)
        return SourceGeometryMoments(self.source_binding_sha256, binding, n, midpoint, halfwidth,
            False, (truth["low"], truth["high"]), (low, high), (phi0, phi1), root, widths,
            moments, banks, directed, directed_banks, self.binary_places)


def prepare_integral_geometry(raw, *, binary_places=256):
    """Freeze only original geometry operands; no new history is initialized."""
    _require(type(raw) is dict, "Source must be an object")
    _require(type(binary_places) is int and 32 <= binary_places <= 4096,
             "Root enclosure precision must be from 32 through 4096 binary places")
    section = raw.get("section", raw.get("geometry_source", raw))
    _require(type(section) is dict, "Source section must be an object")
    keys = ("triangle_m", "displacement_A_m", "slope_P_m", "normal_xyz", "offset_m", "lambda_bounds")
    _require(all(key in section for key in keys), "Missing source geometry operands")
    _require("seed_center_world_m" in raw and "seed_radius_m" in raw, "Missing original sphere operands")
    try:
        source_line(raw, (1., 0., 0.))
    except GeometryBoundRefusal as error:
        raise IntegralGeometryRefusal(str(error)) from error
    frozen_section = {}
    for key in keys:
        value = section[key]
        if key == "offset_m":
            frozen_section[key] = rational(value)
        else:
            value_array = np.array(value, dtype=np.float64, copy=True)
            array = np.frombuffer(value_array.tobytes(), dtype=value_array.dtype).reshape(value_array.shape)
            frozen_section[key] = array
    original_center = np.array(raw["seed_center_world_m"], copy=True)
    center = np.frombuffer(original_center.tobytes(), dtype=original_center.dtype).reshape(original_center.shape)
    frozen = dict(section=frozen_section, seed_center_world_m=center,
                  seed_radius_m=rational(raw["seed_radius_m"]))
    try:
        baseline = source_line(frozen, (1., 0., 0.))
    except GeometryBoundRefusal as error:
        raise IntegralGeometryRefusal(str(error)) from error
    frozen["section"] = MappingProxyType(frozen_section)
    return PreparedIntegralGeometry(MappingProxyType(frozen), _source_digest(baseline), binary_places)
