"""Validated angle/depth integration of a source-bound curved normal law.

The smooth parts use factorial-normalised bivariate Taylor coefficients.
Polynomial terms are integrated with exact rectangle moments.  Derivatives
of total degree N+1 evaluated over the entire rectangle bound the integral
remainder by the multivariate Taylor theorem.  A pressure sign is never
decided from quadrature samples or a pressure/area/depth cutoff.

This certifies the declared reduced curved potential.  It does not certify
three-dimensional column strain energy, whole-clock collision freedom or
the admission of a material interval.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
from math import factorial

from .geometric_native_pose import FractionInterval as I, _rounded_interval, interval_sqrt
from .geometric_native_spherical_solid import interval_cbrt


def _as_interval(value):
    if type(value) is I:
        return value
    if type(value) in (F, int):
        return I(value)
    raise ValueError("Taylor scalars require literal rational interval values")


def _abs_upper(value):
    return max(abs(value.lo), abs(value.hi))


@lru_cache(maxsize=64)
def _indices(order):
    return tuple((i, degree-i) for degree in range(order+1) for i in range(degree+1))


@dataclass(frozen=True, slots=True)
class TaylorAlgebra:
    order: int
    precision: int = 160
    degree_limits: tuple | None = None

    def __post_init__(self):
        if type(self.order) is not int or not 0 <= self.order <= 32:
            raise ValueError("Validated Taylor order must lie in [0,32]")
        if type(self.precision) is not int or not 64 <= self.precision <= 512:
            raise ValueError("Validated Taylor precision must lie in [64,512]")
        if self.degree_limits is not None and (
                type(self.degree_limits) is not tuple or len(self.degree_limits) != 2
                or any(type(x) is not int or x < 0 for x in self.degree_limits)
                or sum(self.degree_limits) != self.order):
            raise ValueError("Tensor Taylor degree limits must sum to the declared order")

    def constant(self, value):
        if type(value) is TaylorJet:
            if value.algebra != self:
                raise ValueError("Taylor expressions cannot mix derivative domains")
            return value
        return TaylorJet(self, {(0, 0): _rounded_interval(_as_interval(value), self.precision)})

    def variable(self, value, *, coordinate, scale=F(1)):
        if type(coordinate) is not int or coordinate not in (0, 1) or type(scale) is not F:
            raise ValueError("A Taylor variable needs one exact coordinate and rational scale")
        coefficients = {(0, 0): _rounded_interval(_as_interval(value), self.precision)}
        if self.order and scale and (self.degree_limits is None or self.degree_limits[coordinate]):
            coefficients[(1, 0) if coordinate == 0 else (0, 1)] = I(scale)
        return TaylorJet(self, coefficients)

    def bound(self, value):
        return self.constant(value).constant_bounds

    def sqrt(self, value):
        return self.constant(value)._real_power(F(1, 2))

    def cbrt(self, value):
        return self.constant(value)._real_power(F(1, 3))

    def sin(self, value):
        return self.constant(value)._trigonometric(False)

    def cos(self, value):
        return self.constant(value)._trigonometric(True)


class TaylorJet:
    """Finite derivative jet, not a polynomial approximation without a tail.

    A coefficient (i,j) encloses d^(i+j)f/(i! j!) in two scaled
    variables.  The zero coefficient may be an interval: in that case all
    higher coefficients are derivative enclosures over that same box.
    """
    __slots__ = ("algebra", "coefficients")

    def __init__(self, algebra, coefficients):
        if type(algebra) is not TaylorAlgebra or type(coefficients) is not dict:
            raise ValueError("Taylor jets require their declared algebra and coefficients")
        if any(type(key) is not tuple or len(key) != 2 or any(type(k) is not int or k < 0 for k in key)
               or sum(key) > algebra.order or type(value) is not I
               or (algebra.degree_limits is not None and any(k > limit for k, limit in zip(key, algebra.degree_limits)))
               for key, value in coefficients.items()):
            raise ValueError("Invalid factorial-normalised Taylor coefficient")
        self.algebra = algebra
        self.coefficients = {key: value for key, value in coefficients.items() if value.lo or value.hi}

    @property
    def constant_bounds(self):
        return self.coefficients.get((0, 0), I(0))

    def coefficient(self, i, j):
        return self.coefficients.get((i, j), I(0))

    def _other(self, value):
        return self.algebra.constant(value)

    def __add__(self, other):
        other = self._other(other)
        keys = self.coefficients.keys() | other.coefficients.keys()
        return TaylorJet(self.algebra, {key: _rounded_interval(
            self.coefficients.get(key, I(0))+other.coefficients.get(key, I(0)), self.algebra.precision)
            for key in keys})

    __radd__ = __add__

    def __neg__(self):
        return TaylorJet(self.algebra, {key: -value for key, value in self.coefficients.items()})

    def __sub__(self, other):
        return self + -self._other(other)

    def __rsub__(self, other):
        return self._other(other) + -self

    def __mul__(self, other):
        other = self._other(other)
        result = {}
        order, precision = self.algebra.order, self.algebra.precision
        for (i, j), first in self.coefficients.items():
            for (k, ell), second in other.coefficients.items():
                if i+j+k+ell <= order and (self.algebra.degree_limits is None or
                        (i+k <= self.algebra.degree_limits[0] and j+ell <= self.algebra.degree_limits[1])):
                    key = (i+k, j+ell)
                    result[key] = _rounded_interval(result.get(key, I(0))+first*second, precision)
        return TaylorJet(self.algebra, result)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self*self._other(other)._real_power(F(-1))

    def __rtruediv__(self, other):
        return self._other(other)*self._real_power(F(-1))

    def __pow__(self, exponent):
        if type(exponent) is not int:
            raise ValueError("Taylor polynomial powers require literal integer exponents")
        if exponent < 0:
            return self._real_power(F(-1))**(-exponent)
        result, value = self.algebra.constant(1), self
        while exponent:
            if exponent & 1:
                result = result*value
            exponent //= 2
            if exponent:
                value = value*value
        return result

    def _zero_constant_part(self):
        return TaylorJet(self.algebra, {key: value for key, value in self.coefficients.items()
                                       if key != (0, 0)})

    def _real_power(self, exponent):
        base = self.constant_bounds
        if exponent == -1:
            if base.lo <= 0 <= base.hi:
                raise ValueError("A Taylor denominator cannot include zero")
            root = 1/base
        elif exponent == F(1, 2):
            if base.lo <= 0:
                if base == I(0) and not self._zero_constant_part().coefficients:
                    return self.algebra.constant(0)
                raise ValueError("Smooth Taylor square roots require strictly positive arguments")
            root = interval_sqrt(base, self.algebra.precision)
        elif exponent == F(1, 3):
            if base.lo <= 0:
                raise ValueError("Smooth Taylor cube roots require strictly positive arguments")
            root = interval_cbrt(base, self.algebra.precision)
        else:
            raise ValueError("Unsupported validated Taylor real exponent")
        normalized = self._zero_constant_part()*self.algebra.constant(1/base)
        result, power, binomial = self.algebra.constant(root), self.algebra.constant(1), F(1)
        for degree in range(1, self.algebra.order+1):
            power = power*normalized
            if not power.coefficients:
                break
            binomial = binomial*(exponent-(degree-1))/degree
            result = result+power*self.algebra.constant(root*binomial)
        return result

    def _trigonometric(self, cosine):
        from .geometric_native_active_contact_line_work import _trig_ranges_signed
        sine, cos = _trig_ranges_signed(self.constant_bounds.lo, self.constant_bounds.hi,
                                         self.algebra.precision)
        derivatives = (cos, -sine, -cos, sine) if cosine else (sine, cos, -sine, -cos)
        result = self.algebra.constant(derivatives[0])
        delta, power = self._zero_constant_part(), self.algebra.constant(1)
        for degree in range(1, self.algebra.order+1):
            power = power*delta
            if not power.coefficients:
                break
            result = result+power*self.algebra.constant(derivatives[degree % 4]/factorial(degree))
        return result


@dataclass(frozen=True, slots=True)
class ValidatedRectangleIntegral:
    integral_bounds: tuple
    polynomial_integral_bounds: tuple
    remainder_absolute_upper_bounds: tuple
    angle_bounds_radians: I
    material_depth_bounds_m: I
    taylor_order: int
    precision: int
    remainder_axis_absolute_upper_bounds: tuple

    @property
    def whole_rectangle_derivative_remainder_included(self):
        return True


def validated_integrate_rectangle(evaluator, angle_bounds_radians, material_depth_bounds_m, *,
                                 order=8, precision=160):
    """Enclose every returned smooth scalar's whole rectangle integral.

    ``evaluator(theta, depth, algebra)`` must return an immutable tuple of
    expressions using the supplied analytic scalar algebra.  It is invoked
    once at the centre and once with constant terms enclosing the whole box.
    The latter is essential: centre derivatives alone do not bound a tail.
    """
    theta, depth = angle_bounds_radians, material_depth_bounds_m
    if type(theta) is not I or type(depth) is not I or not callable(evaluator):
        raise ValueError("Validated integration requires a callable and two exact rectangles")
    if type(order) is not int or not 0 <= order <= 24:
        raise ValueError("Validated integral order must lie in [0,24]")
    area = theta.width*depth.width
    mid_algebra, full_algebra = TaylorAlgebra(order, precision), TaylorAlgebra(order+1, precision)
    def values(algebra, at_center):
        first = algebra.variable(I(theta.midpoint) if at_center else theta,
                                 coordinate=0, scale=theta.width/2)
        second = algebra.variable(I(depth.midpoint) if at_center else depth,
                                  coordinate=1, scale=depth.width/2)
        answer = evaluator(first, second, algebra)
        if type(answer) is not tuple:
            raise ValueError("An analytic integral evaluator must return an immutable scalar tuple")
        return tuple(algebra.constant(value) for value in answer)
    middle, full = values(mid_algebra, True), values(full_algebra, False)
    if len(middle) != len(full):
        raise ValueError("A rectangle evaluator changed its scalar dimension")
    polynomial, tails, bounds, axis_tails = [], [], [], []
    for center, whole in zip(middle, full):
        exact_moments = I(0)
        for (i, j), coefficient in center.coefficients.items():
            if i % 2 == 0 and j % 2 == 0:
                exact_moments = _rounded_interval(exact_moments+coefficient*area/F((i+1)*(j+1)), precision)
        terms = tuple(_abs_upper(whole.coefficient(i, order+1-i))*area/F((i+1)*(order+2-i))
                      for i in range(order+2))
        remainder = sum(terms, F(0))
        polynomial.append(exact_moments)
        tails.append(remainder)
        bounds.append(_rounded_interval(exact_moments+I(-remainder, remainder), precision))
        axis_tails.append((sum((term*F(i, order+1) for i, term in enumerate(terms)), F(0)),
                           sum((term*F(order+1-i, order+1) for i, term in enumerate(terms)), F(0))))
    return ValidatedRectangleIntegral(tuple(bounds), tuple(polynomial), tuple(tails), theta, depth,
                                      order, precision, tuple(axis_tails))


def validated_integrate_tensor_rectangle(evaluator, angle_bounds_radians, material_depth_bounds_m, *,
                                        angle_order=2, depth_order=8, precision=160):
    """Sequential Taylor proof with independent angle and depth degrees.

    First expand in angle, enclosing the angle remainder on the whole box.
    Then integrate each even angle coefficient using its depth expansion and
    whole-depth derivative remainder.  The angle domain is not collapsed to
    its midpoint; this is useful for tiny native arcs with a much deeper face.
    """
    theta, depth = angle_bounds_radians, material_depth_bounds_m
    if type(theta) is not I or type(depth) is not I or not callable(evaluator):
        raise ValueError("Tensor integration requires an analytic evaluator and exact rectangle")
    if any(type(x) is not int or not 0 <= x <= 24 for x in (angle_order, depth_order)) or angle_order+depth_order+1 > 32:
        raise ValueError("Tensor integration degrees exceed the validated derivative domain")
    area = theta.width*depth.width
    middle_algebra = TaylorAlgebra(angle_order+depth_order, precision, (angle_order, depth_order))
    depth_algebra = TaylorAlgebra(angle_order+depth_order+1, precision, (angle_order, depth_order+1))
    angle_algebra = TaylorAlgebra(angle_order+1, precision, (angle_order+1, 0))
    def evaluate(algebra, theta_zero, depth_zero):
        answer = evaluator(algebra.variable(theta_zero, coordinate=0, scale=theta.width/2),
                           algebra.variable(depth_zero, coordinate=1, scale=depth.width/2), algebra)
        if type(answer) is not tuple:
            raise ValueError("A tensor evaluator must return an immutable scalar tuple")
        return tuple(algebra.constant(x) for x in answer)
    middle = evaluate(middle_algebra, I(theta.midpoint), I(depth.midpoint))
    depth_derivatives = evaluate(depth_algebra, I(theta.midpoint), depth)
    angle_derivatives = evaluate(angle_algebra, theta, depth)
    if not len(middle) == len(depth_derivatives) == len(angle_derivatives):
        raise ValueError("A tensor evaluator changed its scalar dimension")
    bounds, polynomials, tails, axis_tails = [], [], [], []
    for center, along_depth, along_angle in zip(middle, depth_derivatives, angle_derivatives):
        polynomial = I(0)
        for (i, j), coefficient in center.coefficients.items():
            if not i % 2 and not j % 2:
                polynomial = _rounded_interval(polynomial+coefficient*area/F((i+1)*(j+1)), precision)
        depth_tail = sum((_abs_upper(along_depth.coefficient(i, depth_order+1))*area/F((i+1)*(depth_order+2))
                          for i in range(0, angle_order+1, 2)), F(0))
        angle_tail = _abs_upper(along_angle.coefficient(angle_order+1, 0))*area/F(angle_order+2)
        tail = depth_tail+angle_tail
        bounds.append(_rounded_interval(polynomial+I(-tail, tail), precision))
        polynomials.append(polynomial)
        tails.append(tail)
        axis_tails.append((angle_tail, depth_tail))
    return ValidatedRectangleIntegral(tuple(bounds), tuple(polynomials), tuple(tails), theta, depth,
        depth_order, precision, tuple(axis_tails))


@dataclass(frozen=True, slots=True)
class CertifiedNativeCurvedContactIntegral:
    family_id: str
    source_support_sha256: str
    source_epoch_sha256: str
    source_preload_gap_m: F
    reference_receiver_material_depth_bounds_m: I
    reference_donor_material_depth_bounds_m: I
    angle_bounds_radians: I
    time_fraction: F
    potential_bounds_j: I
    activation_potential_bounds_j: I
    conservative_endpoint_potential_change_bounds_j: I
    integrated_pressure_bounds_n: I
    energy_partition_bounds_j: tuple
    literal_axis_and_radial_increment_gradient_bounds: tuple
    gauss_newton_hessian_bounds: tuple
    free_gap_bounds_m: I
    gradient_absolute_width_budgets: tuple
    own_coordinate_widths_satisfied: bool
    integration_cell_count: int
    closed_taylor_cell_count: int
    open_cell_count: int
    unresolved_sign_cell_count: int
    uncertain_endpoint_cell_count: int
    taylor_order: int
    precision: int
    source_force_certificate_reused: bool = False
    activation_history_adopted: bool = False
    native_force_equilibrium_certified: bool = False
    complete_three_dimensional_strain_energy_certified: bool = False
    physical_material_admission_authorized: bool = False
    native_source_authenticated: bool = False
    used_depth_taylor_orders: tuple = ()

    @property
    def whole_time_interval_force_and_collision_certified(self):
        return False

    @property
    def full_native_angle_and_shared_physical_depth_integrated(self):
        return True

    @property
    def pressure_or_area_cutoff_applied(self):
        return False

    @property
    def gauss_newton_integrand_pointwise_positive_semidefinite(self):
        # Every actual integrand is a nonnegative multiple of dg dg.T.
        # Independent choices from its entrywise enclosure need not be PSD.
        return True

    @property
    def angle_taylor_order(self):
        return 2

    @property
    def activation_depth_taylor_order(self):
        return min(self.taylor_order, 8) if self.gradient_absolute_width_budgets else 0

    @property
    def reduced_series_energy_partition_certified(self):
        return True


@dataclass(frozen=True, slots=True)
class _IntegratedCell:
    theta: I
    depth: I
    integrals: tuple
    hessian: tuple
    gap: I
    kind: str
    split_coordinate: int
    root_depth_bounds_m: I | None = None
    depth_taylor_order: int = 0


def _physical_nonnegative(value):
    if value.hi < 0:
        raise ValueError("A certified nonnegative integral enclosure excluded its physical value")
    return I(max(F(0), value.lo), value.hi)


def integrate_clipped_affine_unit_square(constant, angle_slope, depth_slope):
    """Exact moments of min(a+b*x+c*y,0) on [-1,1] squared.

    The returned tuple is (first moment, squared moment, closed area).  A
    rational half-plane clips the complete square; triangle barycentric
    moments integrate the resulting affine function without sampling.
    """
    if any(type(value) is not F for value in (constant, angle_slope, depth_slope)):
        raise ValueError("Clipped affine moments require three literal Fractions")
    def value(point):
        return constant+angle_slope*point[0]+depth_slope*point[1]
    square = ((F(-1), F(-1)), (F(1), F(-1)), (F(1), F(1)), (F(-1), F(1)))
    polygon = []
    for previous, current in zip(square[-1:]+square[:-1], square):
        fp, fc = value(previous), value(current)
        if (fp <= 0) != (fc <= 0):
            fraction = fp/(fp-fc)
            polygon.append(tuple(a+fraction*(b-a) for a, b in zip(previous, current)))
        if fc <= 0:
            polygon.append(current)
    if len(polygon) < 3:
        return (F(0),)*3
    first, second, measure = F(0), F(0), F(0)
    anchor = polygon[0]
    for b, c in zip(polygon[1:-1], polygon[2:]):
        area = abs((b[0]-anchor[0])*(c[1]-anchor[1])-(b[1]-anchor[1])*(c[0]-anchor[0]))/2
        fa, fb, fc = value(anchor), value(b), value(c)
        first += area*(fa+fb+fc)/3
        second += area*(fa*fa+fb*fb+fc*fc+fa*fb+fb*fc+fc*fa)/6
        measure += area
    return first, second, measure


def _intersect(first, second):
    lower, upper = max(first.lo, second.lo), min(first.hi, second.hi)
    if lower > upper:
        raise ValueError("Independent certified integral enclosures were disjoint")
    return I(lower, upper)


def certified_native_curved_contact_integral(source, *, donor_radial_increment_m,
        receiver_radial_increment_m, donor_owner_axis, receiver_owner_axis,
        time_fraction=F(1), gradient_absolute_width_budgets=None, precision=160,
        taylor_order=8, max_cells=4096):
    """Integrate the actual full carrier, refining for eight separate widths.

    Budgets constrain absolute enclosure widths of each exponential-axis or
    radial-coordinate force separately.  ``None`` requests an initial bounded
    evaluation from which a caller can form its actual physical operand scales.
    A zero budget is useful for an exactly open or structurally zero row; it
    never causes a nonzero row to be dropped.  Failure to reach a budget is
    returned explicitly, retaining every force enclosure.
    """
    from .geometric_native_curved_correlated_kernel import (
        validate_curved_correlated_source, bound_curved_correlated_cell,
        evaluate_curved_correlated_scalars,
    )
    source = validate_curved_correlated_source(source)
    if type(precision) is not int or not 64 <= precision <= 512:
        raise ValueError("Certified force precision must lie in [64,512]")
    if type(taylor_order) is not int or not 1 <= taylor_order <= 24:
        raise ValueError("Certified force Taylor order must lie in [1,24]")
    if type(max_cells) is not int or max_cells < 1:
        raise ValueError("A certified integration cell limit must be a positive integer")
    if gradient_absolute_width_budgets is not None and (
            type(gradient_absolute_width_budgets) is not tuple
            or len(gradient_absolute_width_budgets) != 8
            or any(type(x) is not F or x < 0 for x in gradient_absolute_width_budgets)):
        raise ValueError("All eight own-coordinate width budgets must be literal nonnegative Fractions")
    kwargs = dict(donor_radial_increment_m=donor_radial_increment_m,
        receiver_radial_increment_m=receiver_radial_increment_m,
        donor_owner_axis=donor_owner_axis, receiver_owner_axis=receiver_owner_axis,
        time_fraction=time_fraction)
    zero_hessian = tuple((I(0),)*8 for _ in range(8))

    def raw(theta, depth):
        return bound_curved_correlated_cell(source, angle_bounds_radians=theta,
            receiver_depth_bounds_m=depth, **kwargs)

    def analytic(theta, depth, algebra, *, reference=False):
        state = dict(donor_radial_increment_m=F(0), receiver_radial_increment_m=F(0),
            donor_owner_axis=(F(0),)*3, receiver_owner_axis=(F(0),)*3,
            time_fraction=F(1)) if reference else kwargs
        result = evaluate_curved_correlated_scalars(source, theta, depth,
            algebra=algebra, include_gauss_newton=False, **state)
        return (result.potential, result.pressure*result.area)+result.potential_gradient

    def build(theta, depth, uncertain=False, *, active_order=None):
        result = raw(theta, depth)
        effective_gap, jet = result.gap, None
        if result.gap.lo < 0 < result.gap.hi:
            algebra = TaylorAlgebra(2 if gradient_absolute_width_budgets is not None else 1, precision)
            jet = evaluate_curved_correlated_scalars(source,
                algebra.variable(theta, coordinate=0, scale=theta.width/2),
                algebra.variable(depth, coordinate=1, scale=depth.width/2),
                algebra=algebra, include_gauss_newton=False, **kwargs).gap
            if depth.width:
                derivative = jet.coefficient(0, 1)/(depth.width/2)
                if derivative.lo > 0 or derivative.hi < 0:
                    lower, upper = raw(theta, I(depth.lo)).gap, raw(theta, I(depth.hi)).gap
                    monotone = I(lower.lo, upper.hi) if derivative.lo > 0 else I(upper.lo, lower.hi)
                    effective_gap = _intersect(effective_gap, monotone)
        area = theta.width*depth.width
        measure = I(0, area) if uncertain else I(area)
        hessian = tuple(tuple(_rounded_interval(value*measure, precision) for value in row)
                        for row in result.gauss_newton_hessian)
        if effective_gap.lo >= 0:
            return _IntegratedCell(theta, depth, (I(0),)*10, zero_hessian, effective_gap,
                                   "endpoint" if uncertain else "open", 1)
        if not uncertain and effective_gap.hi <= 0 and gradient_absolute_width_budgets is not None:
            if effective_gap.hi < 0:
                hessian = tuple(tuple(_rounded_interval(_intersect(original,
                    result.area*source.effective_normal_stiffness_pa_m*first*second*area), precision)
                    for original, second in zip(row, result.gap_gradient))
                    for row, first in zip(hessian, result.gap_gradient))
            active_order = min(4, taylor_order) if active_order is None else active_order
            integral = validated_integrate_tensor_rectangle(analytic, theta, depth,
                angle_order=2, depth_order=active_order, precision=precision)
            bounds = tuple(_physical_nonnegative(x) if index < 2 else x
                           for index, x in enumerate(integral.integral_bounds))
            if gradient_absolute_width_budgets is None:
                theta_tail = sum((row[0] for row in integral.remainder_axis_absolute_upper_bounds[2:]), F(0))
                depth_tail = sum((row[1] for row in integral.remainder_axis_absolute_upper_bounds[2:]), F(0))
            else:
                theta_tail = sum((row[0]/budget for row, budget in
                    zip(integral.remainder_axis_absolute_upper_bounds[2:], gradient_absolute_width_budgets)
                    if budget), F(0))
                depth_tail = sum((row[1]/budget for row, budget in
                    zip(integral.remainder_axis_absolute_upper_bounds[2:], gradient_absolute_width_budgets)
                    if budget), F(0))
            coordinate = 0 if theta_tail > depth_tail else 1
            return _IntegratedCell(theta, depth, bounds, hessian, effective_gap, "closed", coordinate,
                                   depth_taylor_order=active_order)
        bounds = tuple(_rounded_interval(x*measure, precision) for x in
                       (result.potential, result.pressure*result.area)+result.potential_gradient)
        coordinate, root_depth = 1, None
        if not uncertain and effective_gap.hi > 0:
            if jet is None:
                algebra = TaylorAlgebra(2 if gradient_absolute_width_budgets is not None else 1, precision)
                jet = evaluate_curved_correlated_scalars(source,
                    algebra.variable(theta, coordinate=0, scale=theta.width/2),
                    algebra.variable(depth, coordinate=1, scale=depth.width/2),
                    algebra=algebra, include_gauss_newton=False, **kwargs).gap
            coordinate = 0 if _abs_upper(jet.coefficient(1, 0)) > _abs_upper(jet.coefficient(0, 1)) else 1
            if gradient_absolute_width_budgets is not None:
                center_algebra = TaylorAlgebra(1, precision)
                center = evaluate_curved_correlated_scalars(source,
                    center_algebra.variable(I(theta.midpoint), coordinate=0, scale=theta.width/2),
                    center_algebra.variable(I(depth.midpoint), coordinate=1, scale=depth.width/2),
                    algebra=center_algebra, include_gauss_newton=False, **kwargs).gap
                coefficients = tuple(center.coefficient(i, j) for i, j in ((0, 0), (1, 0), (0, 1)))
                nominal = tuple(value.midpoint for value in coefficients)
                # Full-box degree-two derivatives bound every nonlinear
                # term; source/rounding widths of the affine coefficients
                # remain in the same pointwise gap error enclosure.
                error = sum((value.width/2 for value in coefficients), F(0))+sum(
                    (_abs_upper(jet.coefficient(i, 2-i)) for i in range(3)), F(0))
                below = integrate_clipped_affine_unit_square(nominal[0]-error, nominal[1], nominal[2])
                above = integrate_clipped_affine_unit_square(nominal[0]+error, nominal[1], nominal[2])
                first_moment = I(below[0], above[0])*(area/4)
                squared_moment = I(above[1], below[1])*(area/4)
                k, J = source.effective_normal_stiffness_pa_m, result.area
                lower_closed, upper_closed = above[2], below[2]
                if not nominal[1] and not nominal[2]:
                    # A whole square whose affine upper bound is identically
                    # zero need not be strictly closed; preserve the source
                    # evaluator's zero-pressure boundary convention.
                    if nominal[0]+error == 0:
                        lower_closed = F(0)
                    if nominal[0]-error == 0:
                        upper_closed = F(0)
                closed_measure = I(lower_closed, upper_closed)*(area/4)
                hessian = tuple(tuple(_rounded_interval(_intersect(original,
                    J*k*first*second*closed_measure), precision)
                    for original, second in zip(row, result.gap_gradient))
                    for row, first in zip(hessian, result.gap_gradient))
                clipped_bounds = (J*k*squared_moment/2, -J*k*first_moment)+tuple(
                    J*k*dg*first_moment+k*dJ*squared_moment/2
                    for dg, dJ in zip(result.gap_gradient, result.area_gradient))
                bounds = tuple(_rounded_interval(_intersect(original, improved), precision)
                               for original, improved in zip(bounds, clipped_bounds))
            if coordinate == 1 and depth.width:
                derivative = jet.coefficient(0, 1)/(depth.width/2)
                if derivative.lo > 0 or derivative.hi < 0:
                    middle_gap = raw(theta, I(depth.midpoint)).gap
                    roots = _rounded_interval(I(depth.midpoint)-middle_gap/derivative, precision)
                    lower, upper = max(depth.lo, roots.lo), min(depth.hi, roots.hi)
                    if lower <= upper:
                        root_depth = I(lower, upper)
        return _IntegratedCell(theta, depth, bounds, hessian, effective_gap,
                               "endpoint" if uncertain else "bounded_closed" if effective_gap.hi <= 0 else "unresolved",
                               coordinate, root_depth)

    endpoint, depth = source.native_angle_end_bounds_radians, source.receiver_depth_bounds_m
    cells = []
    if endpoint.lo > 0 and depth.width:
        cells.append(build(I(0, endpoint.lo), depth))
    if endpoint.width and depth.width:
        cells.append(build(I(endpoint.lo, endpoint.hi), depth, True))
    if not cells:
        # A literal zero-measure face has exactly zero integrated mechanics.
        gap = raw(I(0), depth).gap
        cells.append(_IntegratedCell(I(0), depth, (I(0),)*10, zero_hessian, gap, "open", 1))

    def totals():
        return tuple(_rounded_interval(sum((cell.integrals[index] for cell in cells), I(0)), precision)
                     for index in range(10))

    def satisfied(values):
        return gradient_absolute_width_budgets is None or all(
            value.width <= budget for value, budget in zip(values[2:], gradient_absolute_width_budgets))

    values = totals()
    while not satisfied(values):
        if any(not budget and value.width for value, budget in
               zip(values[2:], gradient_absolute_width_budgets)):
            # A finite-width analytic/source enclosure cannot meet an exact
            # zero-width demand.  Preserve the rows and report this honestly.
            break
        eligible = tuple((index, cell) for index, cell in enumerate(cells)
                         if cell.kind not in ("open", "endpoint")
                         and (cell.theta.width or cell.depth.width))
        if not eligible:
            break
        def priority(row):
            cell = row[1]
            unsatisfied = tuple((value.width, budget) for value, budget, total in
                zip(cell.integrals[2:], gradient_absolute_width_budgets, values[2:]) if total.width > budget)
            if any(not budget and width for width, budget in unsatisfied):
                return (1, max((width for width, budget in unsatisfied if not budget), default=F(0)))
            return (0, max((width/budget for width, budget in unsatisfied if budget), default=F(0)))
        index, cell = max(eligible, key=priority)
        if cell.kind == "closed" and cell.depth_taylor_order < taylor_order:
            cells[index] = build(cell.theta, cell.depth,
                active_order=min(taylor_order, cell.depth_taylor_order+2))
            values = totals()
            continue
        if len(cells) >= max_cells:
            break
        if cell.root_depth_bounds_m is not None:
            cuts = sorted(set((cell.depth.lo, cell.root_depth_bounds_m.lo,
                               cell.root_depth_bounds_m.hi, cell.depth.hi)))
            if len(cuts) > 2 and len(cells)+len(cuts)-2 <= max_cells:
                replacements = [build(cell.theta, I(first, last)) for first, last in zip(cuts, cuts[1:])
                                if first < last]
                cells[index:index+1] = replacements
                values = totals()
                continue
        coordinate = cell.split_coordinate
        if coordinate == 0 and not cell.theta.width:
            coordinate = 1
        if coordinate == 1 and not cell.depth.width:
            coordinate = 0
        interval = cell.theta if coordinate == 0 else cell.depth
        half = interval.midpoint
        halves = (I(interval.lo, half), I(half, interval.hi))
        replacements = [build(part, cell.depth) if coordinate == 0 else build(cell.theta, part)
                        for part in halves]
        cells[index:index+1] = replacements
        values = totals()

    if source.source_preload_gap_m >= 0:
        activation = I(0)
    elif gradient_absolute_width_budgets is None:
        reference = bound_curved_correlated_cell(source,
            angle_bounds_radians=I(0, endpoint.lo), receiver_depth_bounds_m=depth,
            donor_radial_increment_m=F(0), receiver_radial_increment_m=F(0),
            donor_owner_axis=(F(0),)*3, receiver_owner_axis=(F(0),)*3,
            include_gauss_newton=False)
        activation = reference.potential*(endpoint.lo*depth.width)
        if endpoint.width:
            tail = bound_curved_correlated_cell(source,
                angle_bounds_radians=I(endpoint.lo, endpoint.hi), receiver_depth_bounds_m=depth,
                donor_radial_increment_m=F(0), receiver_radial_increment_m=F(0),
                donor_owner_axis=(F(0),)*3, receiver_owner_axis=(F(0),)*3,
                include_gauss_newton=False)
            activation += tail.potential*I(0, endpoint.width*depth.width)
    else:
        activation = _certified_activation_integral(source, precision, min(taylor_order, 8))
    hessian = tuple(tuple(_rounded_interval(sum((cell.hessian[i][j] for cell in cells), I(0)), precision)
                          for j in range(8)) for i in range(8))
    energy_parts = tuple(_rounded_interval(values[0]*source.effective_normal_stiffness_pa_m*compliance,
                                           precision) for compliance in source.normal_compliances_m_pa)
    gap = I(min(cell.gap.lo for cell in cells), max(cell.gap.hi for cell in cells))
    return CertifiedNativeCurvedContactIntegral(source.family_id, source.source_support_sha256,
        source.source_epoch_sha256, source.source_preload_gap_m, source.receiver_depth_bounds_m,
        source.receiver_depth_bounds_m+source.donor_depth_offset_m, source.angle_bounds_radians,
        time_fraction, values[0], activation, values[0]-activation, values[1], energy_parts,
        values[2:], hessian, gap,
        () if gradient_absolute_width_budgets is None else gradient_absolute_width_budgets,
        satisfied(values), len(cells), sum(cell.kind == "closed" for cell in cells),
        sum(cell.kind == "open" for cell in cells), sum(cell.kind == "unresolved" for cell in cells),
        sum(cell.kind == "endpoint" for cell in cells), taylor_order, precision,
        native_source_authenticated=source.native_source_authenticated,
        used_depth_taylor_orders=tuple(sorted(set(cell.depth_taylor_order for cell in cells
                                                if cell.depth_taylor_order))))


@lru_cache(maxsize=32)
def _certified_activation_integral(source, precision, taylor_order):
    """The immutable source reference is independent of the current solve."""
    from .geometric_native_curved_correlated_kernel import (
        validate_curved_correlated_source, evaluate_curved_correlated_scalars,
        bound_curved_correlated_cell,
    )
    validate_curved_correlated_source(source)
    endpoint, depth = source.native_angle_end_bounds_radians, source.receiver_depth_bounds_m
    kwargs = dict(donor_radial_increment_m=F(0), receiver_radial_increment_m=F(0),
        donor_owner_axis=(F(0),)*3, receiver_owner_axis=(F(0),)*3, time_fraction=F(1))
    def evaluator(theta, z, algebra):
        response = evaluate_curved_correlated_scalars(source, theta, z,
            algebra=algebra, include_gauss_newton=False, **kwargs)
        return (response.potential,)
    activation = I(0)
    if endpoint.lo and depth.width:
        activation = _physical_nonnegative(validated_integrate_tensor_rectangle(evaluator,
            I(0, endpoint.lo), depth, angle_order=2, depth_order=taylor_order,
            precision=precision).integral_bounds[0])
    if endpoint.width and depth.width:
        response = bound_curved_correlated_cell(source,
            angle_bounds_radians=I(endpoint.lo, endpoint.hi), receiver_depth_bounds_m=depth,
            include_gauss_newton=False, **kwargs)
        activation = _rounded_interval(activation+response.potential*I(0, endpoint.width*depth.width), precision)
    return activation
