"""Centered fixed-material-line clipping and polynomial lambda primitives.

Only the radial integral is analytic.  These helpers neither certify angular
integration nor admit a physical fault, release, or new Genesis history.
The source chart retains the existing float64 X+P and A-P convention before
promotion to long double.  All polynomial arrays have coefficient axis first
and use the centered coordinate xi, lambda = midpoint + halfwidth*xi.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Real

import numpy as np

from .genesis_volume_material_sections import CohortMaterialSection, _validate_cohort_section


LD = np.longdouble
EPS = np.finfo(LD).eps
MAX_INTEGRAL_DEGREE = 16
FORMAT = "genesis-centered-exact-lambda-primitives-0.1"
SCOPE = "radial_polynomial_primitive_no_angular_or_physical_certificate"


class ExactLambdaRefusal(ValueError):
    """The supplied source or represented radial calculation is unresolved."""


def _array(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "ifu":
        raise ExactLambdaRefusal(f"{name} requires real numeric values")
    result = np.asarray(raw, dtype=LD)
    if result.shape != shape or not np.isfinite(result).all():
        raise ExactLambdaRefusal(f"{name} has invalid shape or nonfinite values")
    return result


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ExactLambdaRefusal(f"{name} must be a finite real number")
    result = LD(value)
    if not np.isfinite(result):
        raise ExactLambdaRefusal(f"{name} must be a finite real number")
    return result


def _readonly(value):
    result = np.array(value, dtype=LD, copy=True)
    # The underlying bytes owner is immutable, including through .base.
    return np.frombuffer(result.tobytes(), dtype=LD).reshape(result.shape)


def _fraction(value):
    numerator, denominator = LD(value).as_integer_ratio()
    return Fraction(numerator, denominator)


def _fraction_ld(value):
    """Convert an exact binary rational without overflowing its huge integers."""
    if not value:
        return LD(0)
    numerator, denominator = abs(value.numerator), value.denominator
    keep = np.finfo(LD).nmant + 8
    ns = max(numerator.bit_length()-keep, 0)
    ds = max(denominator.bit_length()-keep, 0)
    result = np.ldexp(LD(numerator >> ns)/LD(denominator >> ds), ns-ds)
    if value < 0:
        result = -result
    if not np.isfinite(result) or result == 0:
        raise ExactLambdaRefusal("Exact source quantity is outside represented long-double range")
    return result


def _det3(matrix):
    return (matrix[0, 0]*(matrix[1, 1]*matrix[2, 2]-matrix[1, 2]*matrix[2, 1])
            -matrix[0, 1]*(matrix[1, 0]*matrix[2, 2]-matrix[1, 2]*matrix[2, 0])
            +matrix[0, 2]*(matrix[1, 0]*matrix[2, 1]-matrix[1, 1]*matrix[2, 0]))


@dataclass(frozen=True, slots=True)
class ExactLambdaSource:
    section: CohortMaterialSection
    seed_center_world_m: np.ndarray
    seed_radius_m: np.longdouble
    birth_C_m: np.ndarray
    birth_D_m: np.ndarray
    lambda_bounds: tuple
    reference_determinant_m3: np.longdouble
    source_section_binding_sha256: str
    format: str = FORMAT
    scope: str = SCOPE
    geometry_operand_accuracy_certified: bool = False
    physical_creation_admitted: bool = False
    physical_release_admitted: bool = False


@dataclass(frozen=True, slots=True)
class FixedNRadialInterval:
    source: ExactLambdaSource
    barycentric: np.ndarray
    lambda_bounds: tuple
    lambda_mid: np.longdouble
    lambda_halfwidth: np.longdouble
    birth_relative_coefficients: np.ndarray
    literal_phi_coefficients: np.ndarray
    literal_phi_lambda_coefficients: np.ndarray
    gamma_root_xi: np.longdouble | None
    gamma_endpoint_distance_xi: np.longdouble | None
    gamma_left_width_xi: np.longdouble | None
    gamma_right_width_xi: np.longdouble | None
    constant_side: np.longdouble | None
    gamma_root_arithmetic_estimate: np.longdouble
    literal_gamma_root_residual_m: np.longdouble | None
    closest_lambda: np.longdouble
    closest_relative_m: np.ndarray
    sphere_halfwidth_lambda: np.longdouble
    sphere_margin_m2: np.longdouble
    sphere_margin_arithmetic_estimate_m2: np.longdouble
    exact_sign_fallback_used: bool
    reference_determinant_m3: np.longdouble
    relative_lambda_mid_to_original_lower: np.longdouble
    relative_lambda_mid_to_original_upper: np.longdouble
    scope: str = SCOPE
    geometry_operand_accuracy_certified: bool = False
    angular_integral_convergence_certified: bool = False
    physical_creation_admitted: bool = False
    physical_release_admitted: bool = False

    @property
    def section(self):
        return self.source.section

    @property
    def sphere_endpoint_residuals_m2(self):
        points = np.stack((self.birth_relative_coefficients[0]-self.birth_relative_coefficients[1],
                           self.birth_relative_coefficients[0]+self.birth_relative_coefficients[1]))
        return _readonly(np.sum(points*points, axis=1, dtype=LD)-self.source.seed_radius_m**2)


@dataclass(frozen=True, slots=True)
class PolyIntegral:
    value: np.ndarray
    absolute_operand_scale: np.ndarray
    roundoff_bound: np.ndarray
    coefficient_degree: int
    side_power: int
    scope: str = SCOPE
    geometry_operand_accuracy_certified: bool = False
    coefficient_construction_accuracy_certified: bool = False
    angular_integral_convergence_certified: bool = False


def prepare_exact_lambda_source(section, center, radius):
    """Authenticate once; retain literal old float64 source-chart arithmetic."""
    try:
        _validate_cohort_section(section)
    except ValueError as exc:
        raise ExactLambdaRefusal(str(exc)) from exc
    center = _array(center, (3,), "Seed center")
    radius = _scalar(radius, "Seed radius")
    if radius <= 0:
        raise ExactLambdaRefusal("Seed radius must be positive")
    # These two operations deliberately happen before the LD conversion.
    C64 = np.asarray(section.triangle_m, np.float64)+np.asarray(section.slope_P_m, np.float64)
    D64 = np.asarray(section.displacement_A_m, np.float64)-np.asarray(section.slope_P_m, np.float64)
    C, D = _array(C64, (3, 3), "Source X+P"), _array(D64, (3, 3), "Source A-P")
    lower, upper = map(LD, section.lambda_bounds)
    if not np.isfinite((lower, upper)).all() or not 0 < lower < upper:
        raise ExactLambdaRefusal("Original lambda bounds must be finite, ordered and positive")
    determinant = _det3(_array(section.triangle_m, (3, 3), "Source triangle"))
    if not np.isfinite(determinant) or determinant <= 0:
        raise ExactLambdaRefusal("Original reference triangle must have positive finite determinant")
    return ExactLambdaSource(section, _readonly(center), radius, _readonly(C), _readonly(D),
                             (lower, upper), determinant, section.binding_sha256)


def _exact_source_line(source, N):
    n = tuple(map(_fraction, N))
    C = [[_fraction(x) for x in row] for row in source.birth_C_m]
    D = [[_fraction(x) for x in row] for row in source.birth_D_m]
    center = tuple(map(_fraction, source.seed_center_world_m))
    u = [sum((n[i]*C[i][j] for i in range(3)), Fraction()) for j in range(3)]
    d = [sum((n[i]*D[i][j] for i in range(3)), Fraction()) for j in range(3)]
    v = [d[j]-center[j] for j in range(3)]
    return u, d, v


def prepare_fixed_N_interval(source, N, center=None, radius=None):
    """Clip a fixed original N line to the ball and original radial bounds.

    Empty and exactly tangent intersections return None.  An ambiguous sphere
    sign is checked with exact binary-rational source operands rather than
    discarded by a tolerance.  A positive interval is never widened or floored.
    """
    if type(source) is CohortMaterialSection:
        source = prepare_exact_lambda_source(source, center, radius)
    elif type(source) is not ExactLambdaSource or center is not None or radius is not None:
        raise ExactLambdaRefusal("An authenticated exact-lambda source is required")
    raw_N = np.asarray(N)
    N = _array(N, (3,), "Original barycentric coordinates")
    input_eps = np.finfo(raw_N.dtype).eps if raw_N.dtype.kind == "f" else EPS
    sum_budget = (64*input_eps if raw_N.dtype.itemsize <= 8 and raw_N.dtype.kind == "f"
                  else 8*EPS)*np.abs(N).sum(dtype=LD)
    if np.any(N < 0) or abs(N.sum(dtype=LD)-1) > sum_budget:
        raise ExactLambdaRefusal("Original barycentric coordinates must lie in the original simplex")
    u = N@source.birth_C_m
    d = N@source.birth_D_m
    v = d-source.seed_center_world_m
    uu, uv = np.dot(u, u), np.dot(u, v)
    if not np.isfinite((uu, uv)).all() or uu <= 0:
        raise ExactLambdaRefusal("Fixed-N birth line has zero or unresolved direction")
    closest = -uv/uu
    p = v+closest*u
    R2 = source.seed_radius_m**2
    p2 = np.dot(p, p)
    margin = R2-p2
    # Own-operand estimate, including the closest-point arithmetic.  It is
    # reported as a floating-arithmetic estimate, not a physical certificate.
    line_scale = np.abs(N)@np.abs(source.birth_D_m)+np.abs(source.seed_center_world_m)
    u_scale = np.abs(N)@np.abs(source.birth_C_m)
    p_error = 32*EPS*(line_scale+abs(closest)*u_scale+np.abs(p))
    margin_error = (8*EPS*(R2+p2)+2*np.dot(np.abs(p), p_error)+np.dot(p_error, p_error))
    exact = None
    fallback = abs(margin) <= margin_error
    if fallback:
        exact = _exact_source_line(source, N)
        eu, _, ev = exact
        e_uu = sum((x*x for x in eu), Fraction())
        e_uv = sum((x*y for x, y in zip(eu, ev)), Fraction())
        e_vv = sum((x*x for x in ev), Fraction())
        e_margin = _fraction(source.seed_radius_m)**2-e_vv+e_uv*e_uv/e_uu
        if e_margin <= 0:
            return None
        margin = _fraction_ld(e_margin)
        closest = _fraction_ld(-e_uv/e_uu) if e_uv else LD(0)
        p = np.array([_fraction_ld(ev[j]-e_uv*eu[j]/e_uu)
                      if ev[j]-e_uv*eu[j]/e_uu else LD(0) for j in range(3)], dtype=LD)
    elif margin < 0:
        return None
    sphere_halfwidth = np.sqrt(margin/uu)
    if not np.isfinite(sphere_halfwidth) or sphere_halfwidth <= 0:
        raise ExactLambdaRefusal("Positive sphere interval is not representable")
    lower, upper = source.lambda_bounds
    left, right = max(-sphere_halfwidth, lower-closest), min(sphere_halfwidth, upper-closest)
    if right <= left:
        return None
    relative_mid = (left+right)/2
    halfwidth = (right-left)/2
    midpoint = closest+relative_mid
    bounds = (closest+left, closest+right)
    if not np.isfinite((midpoint, halfwidth, *bounds)).all() or halfwidth <= 0:
        raise ExactLambdaRefusal("Clipped centered interval is not representable")
    normal = np.asarray(source.section.normal_xyz, dtype=LD)
    offset = LD(source.section.offset_m)
    # Every field uses this SAME represented midpoint and halfwidth.  The
    # closest-point anchors determine the sphere bounds, but their independently
    # rounded sum must not define a second polynomial birth chart.  Form the
    # small birth offsets and literal Gamma with exact binary source operands
    # before LD conversion, avoiding cancellation against world coordinates.
    if exact is None:
        exact = _exact_source_line(source, N)
    eu, ed, ev = exact
    emidpoint, ehalfwidth = _fraction(midpoint), _fraction(halfwidth)
    e_relative0 = [ev[j]+emidpoint*eu[j] for j in range(3)]
    e_relative1 = [ehalfwidth*eu[j] for j in range(3)]
    relative = np.array([[_fraction_ld(x) if x else LD(0) for x in row]
                         for row in (e_relative0, e_relative1)], dtype=LD)
    en = tuple(map(_fraction, normal))
    es = sum((x*y for x, y in zip(en, eu)), Fraction())
    ec = sum((x*y for x, y in zip(en, ed)), Fraction())-_fraction(offset)
    ep0 = es*emidpoint+ec
    ep1 = es*ehalfwidth
    phi0 = _fraction_ld(ep0) if ep0 else LD(0)
    phi1 = _fraction_ld(ep1) if ep1 else LD(0)
    slope = _fraction_ld(es) if es else LD(0)
    constant = _fraction_ld(ec) if ec else LD(0)
    root = side = endpoint_distance = left_width = right_width = root_residual = None
    root_error = LD(0)
    if phi1 == 0:
        side = LD(.5) if phi0 > 0 else LD(-.5) if phi0 < 0 else LD(0)
    else:
        eroot = -ep0/ep1
        root = _fraction_ld(eroot) if eroot else LD(0)
        exact_root_rounding_error = abs(eroot-_fraction(root))
        root_error = (_fraction_ld(exact_root_rounding_error)
                      if exact_root_rounding_error else LD(0))
        eroot_residual = ep0+ep1*_fraction(root)
        root_residual = _fraction_ld(eroot_residual) if eroot_residual else LD(0)
        if eroot >= 1:
            side = LD(-.5) if phi1 > 0 else LD(.5)
        elif eroot <= -1:
            side = LD(.5) if phi1 > 0 else LD(-.5)
        else:
            endpoint_distance = _fraction_ld(1-abs(eroot))
            left_width = _fraction_ld(1+eroot)
            right_width = _fraction_ld(1-eroot)
    return FixedNRadialInterval(source, _readonly(N), bounds, midpoint, halfwidth,
        _readonly(relative), _readonly((phi0, phi1)), _readonly((constant, slope)), root,
        endpoint_distance, left_width, right_width, side,
        root_error, root_residual, closest, _readonly(p), sphere_halfwidth, margin, margin_error, fallback,
        source.reference_determinant_m3, midpoint-lower, upper-midpoint)


def _poly(coefficients):
    raw = np.asarray(coefficients)
    if raw.dtype.kind not in "ifu" or raw.ndim < 1 or raw.shape[0] == 0:
        raise ExactLambdaRefusal("Polynomial coefficients require a nonempty real coefficient axis first")
    result = np.asarray(raw, dtype=LD)
    if not np.isfinite(result).all():
        raise ExactLambdaRefusal("Polynomial coefficients must be finite")
    return result


def poly_add(*polynomials):
    """Add polynomials, broadcasting their trailing component dimensions."""
    if not polynomials:
        raise ExactLambdaRefusal("Polynomial addition requires operands")
    operands = [_poly(p) for p in polynomials]
    try:
        shape = np.broadcast_shapes(*(p.shape[1:] for p in operands))
    except ValueError as exc:
        raise ExactLambdaRefusal("Polynomial component shapes do not broadcast") from exc
    result = np.zeros((max(len(p) for p in operands),)+shape, dtype=LD)
    for p in operands:
        result[:len(p)] += p.reshape((len(p),)+(1,)*(len(shape)-p.ndim+1)+p.shape[1:])
    if not np.isfinite(result).all():
        raise ExactLambdaRefusal("Polynomial addition overflowed")
    return result


def poly_multiply(left, right):
    """Coefficient convolution with elementwise trailing-component products."""
    left, right = _poly(left), _poly(right)
    try:
        shape = np.broadcast_shapes(left.shape[1:], right.shape[1:])
    except ValueError as exc:
        raise ExactLambdaRefusal("Polynomial component shapes do not broadcast") from exc
    result = np.zeros((len(left)+len(right)-1,)+shape, dtype=LD)
    for i in range(len(left)):
        for j in range(len(right)):
            result[i+j] += left[i]*right[j]
    if not np.isfinite(result).all():
        raise ExactLambdaRefusal("Polynomial product overflowed")
    return result


def poly_evaluate(coefficients, xi):
    """Evaluate at one finite centered coordinate, preserving components."""
    coefficients = _poly(coefficients)
    xi = _scalar(xi, "Centered polynomial coordinate")
    value = np.array(coefficients[-1], dtype=LD, copy=True)
    for coefficient in coefficients[-2::-1]:
        value = value*xi+coefficient
    if not np.isfinite(value).all():
        raise ExactLambdaRefusal("Polynomial evaluation overflowed")
    return value


def _degree(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 <= value <= MAX_INTEGRAL_DEGREE:
        raise ExactLambdaRefusal("Exact primitive degree must be an integer from 0 through 16")
    return int(value)


def _side_power(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value not in (0, 1, 2):
        raise ExactLambdaRefusal("Side power must be 0, 1, or 2")
    return int(value)


def _moments(interval, degree, side_power):
    if type(interval) is not FixedNRadialInterval:
        raise ExactLambdaRefusal("An authenticated fixed-N interval is required")
    degree, side_power = _degree(degree), _side_power(side_power)
    moments = np.zeros(degree+1, dtype=LD)
    errors = np.zeros_like(moments)
    h = interval.lambda_halfwidth
    for k in range(0, degree+1, 2):
        moments[k] = 2*h/LD(k+1)
    if side_power == 2:
        moments *= LD(.25)
    elif side_power == 1:
        if interval.constant_side is not None:
            if interval.constant_side == 0:
                raise ExactLambdaRefusal("Literal Gamma contains this whole line; signed-bank integral is unresolved")
            moments *= interval.constant_side
        else:
            g = interval.gamma_root_xi
            s = LD(1) if interval.literal_phi_coefficients[1] > 0 else LD(-1)
            ag = abs(g)
            for k in range(degree+1):
                if k % 2 == 0:
                    moments[k] = -s*h*g**(k+1)/LD(k+1)
                else:
                    # 1-g^(even) = (1-|g|)*(1+|g|+...).  This retains
                    # arbitrarily small represented endpoint bank slivers.
                    geometric_sum = sum((ag**j for j in range(k+1)), LD(0))
                    moments[k] = s*h*interval.gamma_endpoint_distance_xi*geometric_sum/LD(k+1)
                if k % 2 == 0:
                    errors[k] = h*ag**k*interval.gamma_root_arithmetic_estimate
                else:
                    errors[k] = h*interval.gamma_endpoint_distance_xi*k*interval.gamma_root_arithmetic_estimate
    errors += (32+4*degree)*EPS*np.abs(moments)
    return moments, errors


def exact_polynomial_moments(interval, max_degree=16, side_power=0):
    """Return ordinary, paired-H, or H-squared lambda moments through degree16."""
    return _readonly(_moments(interval, max_degree, side_power)[0])


def reference_side_volumes_m3(interval):
    """Positive fixed-N reference-Jacobian measures on each literal bank.

    These are per unit original angular-coordinate area.  Bank widths are
    retained directly; no nearly equal unsigned/signed volumes are subtracted.
    The enclosing angular integration and geometric propagation remain open.
    """
    if type(interval) is not FixedNRadialInterval:
        raise ExactLambdaRefusal("An authenticated fixed-N interval is required")
    h, midpoint, determinant = (interval.lambda_halfwidth, interval.lambda_mid,
                                interval.reference_determinant_m3)
    total = 2*h*determinant*(midpoint**2+h*h/3)
    if interval.constant_side is not None:
        if interval.constant_side == 0:
            raise ExactLambdaRefusal("Literal Gamma contains this whole line; bank measures are unresolved")
        return {LD(-.5): total if interval.constant_side < 0 else LD(0),
                LD(.5): total if interval.constant_side > 0 else LD(0)}
    result = []
    for width, endpoint, sign in ((interval.gamma_left_width_xi, LD(-1), LD(1)),
                                 (interval.gamma_right_width_xi, LD(1), LD(-1))):
        bank_halfwidth = h*width/2
        bank_mid = midpoint+h*(endpoint+sign*width/2)
        value = 2*bank_halfwidth*determinant*(bank_mid**2+bank_halfwidth**2/3)
        if not np.isfinite(value) or value <= 0:
            raise ExactLambdaRefusal("Positive bank reference measure is not representable")
        result.append(value)
    return {LD(-.5): result[0] if interval.literal_phi_coefficients[1] > 0 else result[1],
            LD(.5): result[1] if interval.literal_phi_coefficients[1] > 0 else result[0]}


def integrate_polynomial(interval, coefficients, side_power=0):
    """Integrate represented xi coefficients without endpoint-power subtraction.

    The arithmetic estimate applies to these represented coefficients, moments
    and their reduction.  A caller must also carry its coefficient-construction
    errors and angular convergence; no global-energy scale or absolute floor is
    included here.
    """
    coefficients = _poly(coefficients)
    moments, moment_errors = _moments(interval, len(coefficients)-1, side_power)
    shape = (len(moments),)+(1,)*(coefficients.ndim-1)
    terms = coefficients*moments.reshape(shape)
    value = terms.sum(axis=0, dtype=LD)
    operands = np.abs(terms).sum(axis=0, dtype=LD)
    arithmetic = (8+2*len(moments))*EPS*operands
    arithmetic += (np.abs(coefficients)*moment_errors.reshape(shape)).sum(axis=0, dtype=LD)
    if not np.isfinite((value, operands, arithmetic)).all():
        raise ExactLambdaRefusal("Polynomial primitive overflowed")
    return PolyIntegral(_readonly(value), _readonly(operands), _readonly(arithmetic),
                        len(coefficients)-1, int(side_power))
