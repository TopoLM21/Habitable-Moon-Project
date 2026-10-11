"""Whole-mask cold tensor integrals on authenticated supported Ritz fields.

The native quarter coordinate is integrated analytically.  Only the transverse
coordinate and areal material depth use the existing validated Taylor engine.
Removed endpoint and curved material remains a signed, bounded correction;
neither proxy polygon areas nor positive area floors supply material.

These component integrals do not establish mixed force equilibrium or permit
adoption of a material interval.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
from math import comb
import hashlib
import json

from .geometric_native_pose import FractionInterval as I, interval_pi, _rounded_interval
from .geometric_native_curved_certified_integration import (
    TaylorAlgebra, TaylorJet, validated_integrate_rectangle,
)
from .geometric_native_spherical_solid import interval_cbrt


def _is_zero(value):
    if type(value) is TaylorJet:
        return not value.coefficients
    return type(value) is I and value.lo == value.hi == 0


class QuarterAngularPolynomial:
    """A finite polynomial in cos(theta),sin(theta), retaining analytic jets.

    The identity sin(theta)^2=1-cos(theta)^2 is applied symbolically.  Thus
    every coefficient has at most one sine factor.  Body tensor products have
    degree at most twelve; the declared finite bound also covers the endpoint
    strip's extra delta factors.  Moments are exact before any depth proof.
    """
    __slots__ = ("algebra", "coefficients")

    def __init__(self, algebra, coefficients):
        if type(algebra) is not QuarterAngularAlgebra or type(coefficients) is not dict:
            raise ValueError("Angular polynomials need their immutable scalar algebra")
        result = {}
        for key, value in coefficients.items():
            if type(key) is not tuple or len(key) != 2 or any(type(k) is not int or k < 0 for k in key):
                raise ValueError("Angular powers must be two nonnegative integers")
            i, j = key
            if i+j > algebra.maximum_degree:
                raise ValueError("Cold angular degree exceeds its declared finite bound")
            value = algebra.scalar.constant(value)
            if _is_zero(value):
                continue
            if j < 2:
                result[key] = result[key]+value if key in result else value
                continue
            for k in range(j//2+1):
                reduced = (i+2*k, j % 2)
                term = value*(F((-1)**k*comb(j//2, k)))
                result[reduced] = result[reduced]+term if reduced in result else term
        self.algebra = algebra
        self.coefficients = {key: value for key, value in result.items() if not _is_zero(value)}

    def _other(self, other):
        return self.algebra.constant(other)

    def __add__(self, other):
        other = self._other(other)
        zero = self.algebra.scalar.constant(0)
        return QuarterAngularPolynomial(self.algebra, {key:
            self.coefficients.get(key, zero)+other.coefficients.get(key, zero)
            for key in self.coefficients.keys() | other.coefficients.keys()})

    __radd__ = __add__

    def __neg__(self):
        return QuarterAngularPolynomial(self.algebra, {key: -value for key, value in self.coefficients.items()})

    def __sub__(self, other):
        return self+-self._other(other)

    def __rsub__(self, other):
        return self._other(other)+-self

    def __mul__(self, other):
        other = self._other(other)
        values = {}
        zero = self.algebra.scalar.constant(0)
        for (i, j), first in self.coefficients.items():
            for (k, ell), second in other.coefficients.items():
                key = (i+k, j+ell)
                term = first*second
                values[key] = values[key]+term if key in values else term
        return QuarterAngularPolynomial(self.algebra, values)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._other(other)
        return self*self.algebra.constant(1/other._constant_scalar())

    def __rtruediv__(self, other):
        return self._other(other)/self

    def __pow__(self, exponent):
        if type(exponent) is not int:
            raise ValueError("Angular powers require a literal integer")
        if exponent < 0:
            return self.algebra.constant(self._constant_scalar()**exponent)
        answer = self.algebra.constant(1)
        for _ in range(exponent):
            answer = answer*self
        return answer

    def _constant_scalar(self):
        if any(key != (0, 0) for key in self.coefficients):
            raise ValueError("Analytic nonpolynomial operations cannot discard theta dependence")
        return self.coefficients.get((0, 0), self.algebra.scalar.constant(0))


class QuarterAngularAlgebra:
    """Lift a validated beta/depth scalar algebra to exact angular polynomials."""
    __slots__ = ("scalar", "precision", "maximum_degree")

    def __init__(self, scalar, *, maximum_degree=24):
        if type(scalar) is not TaylorAlgebra:
            raise ValueError("Angular analytic coefficients require the validated Taylor algebra")
        if type(maximum_degree) is not int or not 12 <= maximum_degree <= 32:
            raise ValueError("Cold angular maximum degree must lie in [12,32]")
        self.scalar, self.precision, self.maximum_degree = scalar, scalar.precision, maximum_degree

    def constant(self, value):
        if type(value) is QuarterAngularPolynomial:
            if value.algebra is not self:
                raise ValueError("Angular expressions cannot mix derivative domains")
            return value
        return QuarterAngularPolynomial(self, {(0, 0): value})

    def sine_theta(self):
        return QuarterAngularPolynomial(self, {(0, 1): F(1)})

    def cosine_theta(self):
        return QuarterAngularPolynomial(self, {(1, 0): F(1)})

    def sin(self, value):
        return self.constant(self.scalar.sin(self.constant(value)._constant_scalar()))

    def cos(self, value):
        return self.constant(self.scalar.cos(self.constant(value)._constant_scalar()))

    def sqrt(self, value):
        return self.constant(self.scalar.sqrt(self.constant(value)._constant_scalar()))

    def cbrt(self, value):
        return self.constant(self.scalar.cbrt(self.constant(value)._constant_scalar()))


@lru_cache(maxsize=1024)
def quarter_angular_moment(cosine_degree, sine_degree, precision=160):
    """Enclose the literal integral cos(theta)^i sin(theta)^j on [0,pi/2]."""
    i, j = cosine_degree, sine_degree
    if any(type(k) is not int or k < 0 for k in (i, j)) or i+j > 32:
        raise ValueError("Quarter moments require powers with total degree at most thirty-two")
    if j >= 2:
        return quarter_angular_moment(i, j-2, precision)*F(j-1, i+j)
    if j == 1:
        return I(F(1, i+1))
    if i >= 2:
        return quarter_angular_moment(i-2, 0, precision)*F(i-1, i)
    return interval_pi(precision)/2 if i == 0 else I(1)


def integrate_quarter_angular_polynomial(value):
    if type(value) is not QuarterAngularPolynomial:
        raise ValueError("Exact angular integration requires a declared finite polynomial")
    result = value.algebra.scalar.constant(0)
    for (i, j), coefficient in value.coefficients.items():
        result = result+coefficient*quarter_angular_moment(i, j, value.algebra.precision)
    return result


def _transpose(matrix):
    return tuple(tuple(matrix[j][i] for j in range(3)) for i in range(3))


def _matmul(first, second):
    return tuple(tuple(sum((first[i][k]*second[k][j] for k in range(3)), 0)
                       for j in range(3)) for i in range(3))


def _sym(matrix):
    values = {}
    for i in range(3):
        values[i, i] = matrix[i][i]
        for j in range(i+1, 3):
            values[i, j] = values[j, i] = (matrix[i][j]+matrix[j][i])/2
    return tuple(tuple(values[i, j] for j in range(3)) for i in range(3))


def _sym_gram(first, second):
    values = {}
    for i in range(3):
        values[i, i] = sum((first[k][i]*second[k][i] for k in range(3)), 0)
        for j in range(i+1, 3):
            values[i, j] = values[j, i] = sum((
                first[k][i]*second[k][j]+first[k][j]*second[k][i]
                for k in range(3)), 0)/2
    return tuple(tuple(values[i, j] for j in range(3)) for i in range(3))


def _tensor_inner(first, second, lam, shear):
    return lam*sum((first[i][i] for i in range(3)), 0)*sum((second[i][i] for i in range(3)), 0)+2*shear*sum(
        ((1 if i == j else 2)*first[i][j]*second[i][j]
         for i in range(3) for j in range(i, 3)), 0)


def _symmetric_rows(values, dimension=7):
    if len(values) != dimension*(dimension+1)//2:
        raise ValueError("A complete symmetric tensor is required")
    result = [[None]*dimension for _ in range(dimension)]
    index = 0
    for i in range(dimension):
        for j in range(i, dimension):
            result[i][j] = result[j][i] = values[index]
            index += 1
    return tuple(tuple(row) for row in result)


def _cold_tensor_fields(body, coordinates_m, beta, depth, scalar_algebra, *,
                        theta_sine_bounds=None, theta_cosine_bounds=None):
    """Form tensor fields once, before their angular contractions.

    Green strain contains only the incremental F.  Accepted initial strain
    and the positive nonlocal calibration are assembled from these raw
    integrals by the separate strain material module.
    """
    from .geometric_native_cold_body import evaluate_native_cold_modal_geometry
    algebra = QuarterAngularAlgebra(scalar_algebra)
    sine = algebra.sine_theta() if theta_sine_bounds is None else algebra.constant(theta_sine_bounds)
    cosine = algebra.cosine_theta() if theta_cosine_bounds is None else algebra.constant(theta_cosine_bounds)
    geometry = evaluate_native_cold_modal_geometry(body, sine,
        cosine, algebra.constant(beta), algebra.constant(depth), algebra)
    gradients = geometry.mode_gradient_m_inv
    L = tuple(tuple(sum((coordinates_m[a]*gradients[a][i][j] for a in range(7)), 0)
                    for j in range(3)) for i in range(3))
    LT, H0 = _transpose(L), _sym(gradients[0])
    LTL = _sym_gram(L, L)
    linear = tuple(_sym(value) for value in gradients)
    nonlinear_first = tuple(_sym_gram(L, value) for value in gradients)
    # Remove q0*H0 symbolically, before forming intervals.  This retains the
    # exact old tip amplitude in the calibrated potential at tiny increments.
    green = tuple(tuple(sum((coordinates_m[a]*linear[a][i][j] for a in range(1, 7)), 0)+LTL[i][j]/2
                        for j in range(3)) for i in range(3))
    first = tuple(tuple(tuple((linear[a][i][j] if a else algebra.constant(0))+nonlinear_first[a][i][j]
                             for j in range(3)) for i in range(3)) for a in range(7))
    second = tuple(_sym_gram(gradients[a], gradients[b])
                   for a in range(7) for b in range(a, 7))
    return H0, green, first, second


def _response_values(fields, inner):
    H0, green, first, second = fields
    values = (inner(H0, H0), inner(H0, green))+tuple(inner(H0, value) for value in first)+tuple(
        inner(H0, value) for value in second)+(inner(green, green),)+tuple(inner(green, value) for value in first)+tuple(
        inner(first[a], first[b])+inner(green, second[index])
        for index, (a, b) in enumerate((a, b) for a in range(7) for b in range(a, 7)))
    return values


def _cold_tensor_polynomial_values(body, coordinates_m, beta, depth, scalar_algebra, *, lame_pa, shear_pa):
    fields = _cold_tensor_fields(body, coordinates_m, beta, depth, scalar_algebra)
    values = _response_values(fields, lambda x, y: _tensor_inner(x, y, lame_pa, shear_pa))
    if any(i+j > 12 for value in values for i, j in value.coefficients):
        raise ValueError("The whole-body tensor kernel exceeded its physical degree twelve")
    return values


def _quarter_product(first, second, moment=None):
    """Contract with analytic moments before creating a product polynomial."""
    scalar = first.algebra.scalar
    if len(first.coefficients) > len(second.coefficients):
        first, second = second, first
    answer = scalar.constant(0)
    for (i, j), coefficient in first.coefficients.items():
        moment_second = scalar.constant(0)
        for (k, ell), other in second.coefficients.items():
            factor = quarter_angular_moment(i+k, j+ell, first.algebra.precision) if moment is None else moment(i+k, j+ell)
            moment_second = moment_second+other*factor
        answer = answer+coefficient*moment_second
    return answer


def _quarter_tensor_inner(first, second, lam, shear, moment=None):
    trace_first = sum((first[i][i] for i in range(3)), 0)
    trace_second = sum((second[i][i] for i in range(3)), 0)
    return lam*_quarter_product(trace_first, trace_second, moment)+2*shear*sum((
        (1 if i == j else 2)*_quarter_product(first[i][j], second[i][j], moment)
        for i in range(3) for j in range(i, 3)), 0)


def cold_tensor_angular_kernel(body, coordinates_m, beta, depth, scalar_algebra, *, lame_pa, shear_pa):
    fields = _cold_tensor_fields(body, coordinates_m, beta, depth, scalar_algebra)
    values = _response_values(fields, lambda x, y: _quarter_tensor_inner(x, y, lame_pa, shear_pa))
    factor = scalar_algebra.cos(beta)*body.radius_m**2
    return tuple(value*factor for value in values)


@dataclass(frozen=True, slots=True)
class RawColdTensorIntegral:
    component_id: str
    body_snapshot_sha256: str
    material_snapshot_sha256: str
    coordinates_m: tuple
    reference_tip_strain_stiffness_n_m: I
    reference_tip_strain_stiffness_unintersected_n_m: I
    reference_tip_strain_stiffness_lower_bound_proof: object
    projection_n: I
    projection_gradient_n_m: tuple
    projection_hessian_n_m2: tuple
    incremental_strain_inner_product_j: I
    incremental_local_gradient_n: tuple
    incremental_local_hessian_n_m: tuple
    projection_remainder_n: I
    projection_remainder_gradient_n_m: tuple
    remainder_strain_inner_product_j: I
    remainder_local_gradient_n: tuple
    remainder_local_hessian_n_m: tuple
    quarter_integral_bounds: tuple
    removed_material_correction_bounds: tuple
    actual_material_integral_bounds: tuple
    integration_cell_count: int
    maximum_taylor_remainder_absolute_upper_bounds: tuple
    taylor_order: int
    precision: int
    cell_receipts: tuple
    correction_receipts: tuple
    integral_snapshot_sha256: str
    primary_integrals_use_symbolic_tip_remainder: bool = True
    complete_mixed_force_equilibrium_certified: bool = False
    physical_material_admission_authorized: bool = False


_AUTHENTICATED_INTEGRAL_SNAPSHOTS = set()
_INTEGRAL_CACHE = {}


def _encoded(value):
    if type(value) is F:
        return [value.numerator, value.denominator]
    if type(value) is I:
        return {"lo": _encoded(value.lo), "hi": _encoded(value.hi)}
    if type(value) is tuple:
        return [_encoded(x) for x in value]
    if type(value) is dict:
        return {key: _encoded(x) for key, x in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {key: _encoded(getattr(value, key)) for key in value.__dataclass_fields__
                if key != "integral_snapshot_sha256"}
    if type(value) in (str, int, bool) or value is None:
        return value
    raise TypeError("An integral receipt requires only literal immutable scalar values")


def _snapshot_sha(value):
    return hashlib.sha256(json.dumps(_encoded(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _raw_receipt(body, material, coordinates_m, quarter, correction, cell_count, tails, order, precision,
                 cell_receipts, correction_receipts):
    actual = tuple(_rounded_interval(value-removal, precision) for value, removal in zip(quarter, correction))
    proof = _reference_tip_stiffness_lower_bound(body, material, precision)
    if actual[0].hi < proof.stiffness_lower_bound_n_m:
        raise ValueError("The independent supported strain-norm lower proof disagrees with the Taylor integral")
    k = I(max(actual[0].lo, proof.stiffness_lower_bound_n_m), actual[0].hi)
    prem, prem_first, second = actual[1], actual[2:9], _symmetric_rows(actual[9:37])
    rem_inner, rem_first, rem_second = actual[37], actual[38:45], _symmetric_rows(actual[45:73])
    q0 = coordinates_m[0]
    projection = q0*k+prem
    projection_first = tuple((k if i == 0 else I(0))+prem_first[i] for i in range(7))
    inner = q0*q0*k+2*q0*prem+rem_inner
    first = tuple((q0*k+prem if i == 0 else I(0))+q0*prem_first[i]+rem_first[i] for i in range(7))
    hessian = tuple(tuple((k if i == j == 0 else I(0))+
        (prem_first[j] if i == 0 else I(0))+(prem_first[i] if j == 0 else I(0))+
        rem_second[i][j]+q0*second[i][j] for j in range(7)) for i in range(7))
    values = (body.component_id, body.body_snapshot_sha256, material.material_snapshot_sha256, coordinates_m,
        k, actual[0], proof, projection, projection_first, second, inner, first, hessian,
        prem, prem_first, rem_inner, rem_first, rem_second, quarter, correction, actual,
        cell_count, tails, order, precision, cell_receipts, correction_receipts)
    provisional = RawColdTensorIntegral(*values, "")
    receipt = RawColdTensorIntegral(*values, _snapshot_sha(provisional))
    _AUTHENTICATED_INTEGRAL_SNAPSHOTS.add(receipt.integral_snapshot_sha256)
    return receipt


def validate_native_cold_tensor_integral(value, *, body=None, material=None):
    if type(value) is not RawColdTensorIntegral or value.integral_snapshot_sha256 != _snapshot_sha(value):
        raise ValueError("A complete immutable cold tensor integral is required")
    if value.integral_snapshot_sha256 not in _AUTHENTICATED_INTEGRAL_SNAPSHOTS:
        raise ValueError("Re-sealing a tensor integral does not authenticate its native field")
    if body is not None and (value.component_id != body.component_id or
            value.body_snapshot_sha256 != body.body_snapshot_sha256):
        raise ValueError("The cold tensor integral belongs to another native body")
    if material is not None and value.material_snapshot_sha256 != material.material_snapshot_sha256:
        raise ValueError("The cold tensor integral belongs to another strain material")
    if value.reference_tip_strain_stiffness_n_m.lo <= 0:
        raise ValueError("The supported reference tip strain needs a strictly positive stiffness")
    return value


@dataclass(frozen=True, slots=True)
class ColdTensorCellReceipt:
    beta_bounds_radians: I
    areal_material_depth_bounds_m: I
    layer: str
    lame_pa: F
    shear_pa: F
    polynomial_integral_bounds: tuple
    remainder_absolute_upper_bounds: tuple
    integral_bounds: tuple


@dataclass(frozen=True, slots=True)
class ColdTipStiffnessLowerBoundProof:
    beta_cut_radians: F
    pi_bounds: I
    source_archive_area_bounds_sr: I
    paired_curved_removed_area_bounds_sr: tuple
    transverse_shape_squared_integral_radians: F
    transverse_slope_squared_integral_per_radian: F
    maximum_removed_angular_strain_factor: F
    angular_integral_lower_bound: F
    upper_layer_shear_pa: F
    lower_layer_shear_pa: F
    physical_upper_layer_height_m: F
    physical_lower_layer_height_bounds_m: I
    physical_material_bottom_radius_bounds_m: I
    stiffness_lower_bound_n_m: F
    positive_lame_coefficients_retained: bool = True
    strain_quadratic_form_lower_bound_includes_depth_shear: bool = True


def _reference_tip_stiffness_lower_bound(body, material, precision):
    """Independent positive norm proof, retaining the actual removed material.

    In the transported orthonormal (e,t_beta,t_theta) frame, the old-tip
    symmetric gradient has two diagonal entries chi/r and shear
    chi_beta/(2*r).  Its depth-flow radial entry adds nonnegative norm.
    Thus H0:C:H0 >= G*(4*chi**2+chi_beta**2)/r**2 for nonnegative lambda.
    The polynomial beta moments below are exact, while cos(beta)>=1-b**2/2.
    R**2*integral dz/r**2 is the physical radial height of each source layer.
    """
    b, R = body.beta_cut_radians, body.radius_m
    upper_lam, upper_shear = material.lame_coefficients("upper")
    lower_lam, lower_shear = material.lame_coefficients("lower")
    if min(upper_lam, lower_lam) < 0 or b >= 1:
        raise ValueError("The strain lower proof requires source positive Lamé coefficients and a physical narrow collar")
    pi = interval_pi(precision)
    shape_moment, slope_moment = F(33, 140)*b, F(6, 5)/b
    maximum = 4+F(9, 4)/(b*b)
    removed = body.archive_area_bounds_sr.hi+sum((area.hi for area in body.paired_curved_removed_area_bounds_sr), F(0))
    angular = (pi.lo/2)*(1-b*b/2)*(4*shape_moment+slope_moment)-maximum*removed
    top = R+body.roof_radial_displacement_m
    bottom = interval_cbrt(I(top**3-3*R*R*body.areal_material_thickness_m), precision)
    upper_height = body.physical_elastic_thickness_m
    lower_height = I(top-upper_height)-bottom
    if angular <= 0 or lower_height.lo <= 0:
        raise ValueError("The authentic actual cold supported strain norm lacks a positive geometric lower proof")
    lower = angular*(upper_shear*upper_height+lower_shear*lower_height.lo)
    return ColdTipStiffnessLowerBoundProof(b, pi, body.archive_area_bounds_sr,
        body.paired_curved_removed_area_bounds_sr, shape_moment, slope_moment,
        maximum, angular, upper_shear, lower_shear, upper_height, lower_height, bottom, lower)


@dataclass(frozen=True, slots=True)
class ColdTensorCorrectionReceipt:
    correction_kind: str
    layer: str
    removed_area_bounds_sr: I
    areal_material_depth_bounds_m: I
    complete_supported_quarter_scalar_bounds: tuple
    correction_integral_bounds: tuple
    theta_terminal_cap_width_bounds_radians: I
    beta_bounds_radians: I


@dataclass(frozen=True, slots=True)
class ColdTensorEndpointStripReceipt:
    correction_kind: str
    layer: str
    removed_area_bounds_sr: I
    areal_material_depth_bounds_m: I
    plane_cos_coefficient_bounds: I
    plane_sin_coefficient_bounds: I
    positive_theta_case: str
    endpoint_cap_width_bounds_radians: I
    maximum_positive_delta_upper_bound: F
    polynomial_integral_bounds: tuple
    depth_remainder_absolute_upper_bounds: tuple
    transverse_and_atan_remainder_absolute_upper_bounds: tuple
    complete_supported_quarter_scalar_bounds: tuple
    correction_integral_bounds: tuple


def _small_nonnegative_atan(bounds, precision):
    if bounds.lo < 0 or bounds.hi > F(1, 10):
        raise ValueError("The authenticated endpoint cap must remain in its small-angle domain")
    return _rounded_interval(I(max(F(0), bounds.lo-bounds.hi**3/3), bounds.hi), precision)


def _endpoint_strip_parameters(body, precision):
    if len(body.endpoint_plane_coefficients_in_quarter) != 3:
        raise ValueError("An endpoint strip requires its literal archived source plane")
    a, b, c = body.endpoint_plane_coefficients_in_quarter
    if c.hi >= 0:
        raise ValueError("The native endpoint foreign plane must face the physical donor quarter")
    d, e = _rounded_interval(-a/c, precision+16), _rounded_interval(-b/c, precision+16)
    if d.lo >= 0 and e.lo >= 0:
        return d, e, "whole_quarter", I(0), d.hi+e.hi
    if d.lo > 0 and e.hi < 0:
        return d, e, "quarter_minus_terminal_cap", _small_nonnegative_atan(-e/d, precision+16), d.hi
    if d.hi < 0 and e.lo > 0:
        return d, e, "terminal_cap_only", _small_nonnegative_atan(e/-d, precision+16), e.hi
    raise ValueError("The native endpoint sign pattern needs a separately certified strip partition")


def _terminal_cap_moment(cosine_degree, sine_degree, gamma, precision):
    # theta=pi/2-epsilon, so cos(theta)^i*sin(theta)^j becomes
    # sin(epsilon)^i*cos(epsilon)^j.  Bernoulli bounds retain the complete cap.
    i, j = cosine_degree, sine_degree
    lower = gamma.lo**(i+1)/F(i+1)-(F(i, 6)+F(j, 2))*gamma.hi**(i+3)/F(i+3)
    upper = gamma.hi**(i+1)/F(i+1)
    return _rounded_interval(I(max(F(0), lower), upper), precision)


def _weighted_endpoint_moments(d, e, case, cap, power, precision):
    cache = {}
    def literal_power(value, exponent):
        result = I(1)
        for _ in range(exponent):
            result = result*value
        return result
    def moment(i, j):
        key = (i, j)
        if key in cache:
            return cache[key]
        def positive(k, ell):
            whole = quarter_angular_moment(k, ell, precision)
            terminal = _terminal_cap_moment(k, ell, cap, precision)
            return terminal if case == "terminal_cap_only" else whole-terminal if case == "quarter_minus_terminal_cap" else whole
        answer = I(0)
        for k in range(power+1):
            answer = answer+F(comb(power, k))*literal_power(d, k)*literal_power(e, power-k)*positive(i+k, j+power-k)
        cache[key] = _rounded_interval(answer, precision)
        return cache[key]
    return moment


def _strip_depth_kernel(body, coordinates, depth, scalar, *, lam, shear, parameters):
    d, e, case, cap, _ = parameters
    inner = TaylorAlgebra(scalar.order+1, scalar.precision, (1, scalar.order))
    beta = inner.variable(I(0), coordinate=0, scale=F(1))
    depth_inner = TaylorJet(inner, {(0, j): value for (i, j), value in depth.coefficients.items() if i == 0})
    fields = _cold_tensor_fields(body, coordinates, beta, depth_inner, inner)
    first_moments = _weighted_endpoint_moments(d, e, case, cap, 1, scalar.precision)
    second_moments = _weighted_endpoint_moments(d, e, case, cap, 2, scalar.precision)
    first_values = _response_values(fields, lambda x, y: _quarter_tensor_inner(x, y, lam, shear, first_moments))
    second_values = _response_values(fields, lambda x, y: _quarter_tensor_inner(x, y, lam, shear, second_moments))
    factor = inner.cos(beta)*body.radius_m**2
    answer = []
    for first, second in zip(first_values, second_values):
        first, second = first*factor, second*factor
        coefficients = {(0, j): first.coefficient(0, j)+second.coefficient(1, j)/2
                        for j in range(scalar.order+1)}
        answer.append(TaylorJet(scalar, coefficients))
    return tuple(answer)


def _bounded_primary_values(body, coordinates, beta_box, depth_box, *, precision, lam, shear,
                            sine_bounds=I(0, 1), cosine_bounds=I(0, 1), order=2,
                            include_angular_measure=True):
    algebra = TaylorAlgebra(order, precision, (order, 0))
    beta = algebra.variable(beta_box, coordinate=0, scale=F(1))
    depth = algebra.constant(depth_box)
    fields = _cold_tensor_fields(body, coordinates, beta, depth, algebra,
        theta_sine_bounds=sine_bounds, theta_cosine_bounds=cosine_bounds)
    values = _response_values(fields, lambda x, y: _tensor_inner(x, y, lam, shear))
    factor = algebra.cos(beta)*body.radius_m**2 if include_angular_measure else algebra.constant(1)
    return tuple(value._constant_scalar()*factor for value in values)


def _integrate_endpoint_strip(body, coordinates, layer, z0, z1, lam, shear, precision, order):
    parameters = _endpoint_strip_parameters(body, precision)
    d, e, case, cap, D = parameters
    if D > body.beta_cut_radians:
        raise ValueError("The native endpoint strip extends beyond the supported collar")
    integral = validated_integrate_rectangle(lambda unused, depth, scalar: _strip_depth_kernel(
        body, coordinates, depth, scalar, lam=lam, shear=shear, parameters=parameters),
        I(0, 1), I(z0, z1), order=order, precision=precision)
    if case == "terminal_cap_only":
        sine = I(max(F(0), 1-cap.hi**2/2), 1)
        cosine = I(0, cap.hi)
        angle_length = cap.hi
    else:
        sine = cosine = I(0, 1)
        angle_length = interval_pi(precision).hi/2
    bounded = _bounded_primary_values(body, coordinates, I(0, D), I(z0, z1),
        precision=precision, lam=lam, shear=shear, sine_bounds=sine, cosine_bounds=cosine)
    tails = tuple((max(abs(value.coefficient(0, 0).lo), abs(value.coefficient(0, 0).hi))*D**3/3+
                   max(abs(value.coefficient(1, 0).lo), abs(value.coefficient(1, 0).hi))*D**4/3+
                   max(abs(value.coefficient(2, 0).lo), abs(value.coefficient(2, 0).hi))*D**3/3)
                  *angle_length*(z1-z0) for value in bounded)
    corrected = tuple(_rounded_interval(value+I(-tail, tail), precision)
                      for value, tail in zip(integral.integral_bounds, tails))
    receipt = ColdTensorEndpointStripReceipt("endpoint_archive_exact_theta_strip", layer,
        body.endpoint_archive_area_bounds_sr, I(z0, z1), d, e, case, cap, D,
        integral.polynomial_integral_bounds, integral.remainder_absolute_upper_bounds, tails,
        tuple(value.constant_bounds for value in bounded), corrected)
    return receipt


def _angular_range(value):
    """Bound a whole quarter polynomial without making a quadrature sample."""
    result = I(0)
    for (i, j), coefficient in value.coefficients.items():
        if type(coefficient) is TaylorJet:
            coefficient = coefficient.constant_bounds
        multiplier = I(1) if i == j == 0 else I(0, 1)
        result = _rounded_interval(result+coefficient*multiplier, value.algebra.precision)
    return result


def _cold_tensor_angular_raw_kernel(body, coordinates_m, beta, depth, scalar_algebra, *, lame_pa, shear_pa):
    """Unintegrated version used only for certified removed-mask suprema."""
    # Obtain the coefficient polynomials through the same evaluator, with a
    # temporary integration hook local to this call rather than altering any
    # source field.  The analytic kernel below is shared by both operations.
    return _cold_tensor_polynomial_values(body, coordinates_m, beta, depth, scalar_algebra,
                                          lame_pa=lame_pa, shear_pa=shear_pa)


def integrate_native_cold_tensor_response(body, material, coordinates_m, *, precision=128,
        taylor_order=2, beta_cell_count=1, depth_cell_count=1):
    """Integrate one actual cold supported field and retain every tensor term."""
    from .geometric_native_cold_body import validate_native_cold_body
    from .geometric_native_cold_strain_energy import validate_native_cold_strain_material
    body = validate_native_cold_body(body)
    material = validate_native_cold_strain_material(material, body=body)
    if type(coordinates_m) is not tuple or len(coordinates_m) != 7 or any(type(q) is not F for q in coordinates_m):
        raise ValueError("Seven literal metre coordinates are required")
    if type(precision) is not int or not 64 <= precision <= 512:
        raise ValueError("Cold tensor precision must lie in [64,512]")
    if type(taylor_order) is not int or not 0 <= taylor_order <= 12:
        raise ValueError("Cold tensor Taylor order must lie in [0,12]")
    if any(type(n) is not int or n <= 0 for n in (beta_cell_count, depth_cell_count)):
        raise ValueError("Cold tensor cell counts must be positive literal integers")
    cache_key = (body.body_snapshot_sha256, material.material_snapshot_sha256,
        coordinates_m, precision, taylor_order, beta_cell_count, depth_cell_count)
    if cache_key in _INTEGRAL_CACHE:
        return validate_native_cold_tensor_integral(_INTEGRAL_CACHE[cache_key], body=body, material=material)
    quarter, correction, tails = [I(0)]*73, [I(0)]*73, [F(0)]*73
    cells, removals = [], []
    beta_cut = body.beta_cut_radians
    layers = (("upper", F(0), body.elastic_layer_areal_depth_m),
              ("lower", body.elastic_layer_areal_depth_m, body.areal_material_thickness_m))
    for layer, z0, z1 in layers:
        lam, shear = material.lame_coefficients(layer)
        for bi in range(beta_cell_count):
            beta_box = I(beta_cut*bi/beta_cell_count, beta_cut*(bi+1)/beta_cell_count)
            for zi in range(depth_cell_count):
                depth_box = I(z0+(z1-z0)*zi/depth_cell_count, z0+(z1-z0)*(zi+1)/depth_cell_count)
                evaluator = lambda beta, depth, algebra: cold_tensor_angular_kernel(
                    body, coordinates_m, beta, depth, algebra, lame_pa=lam, shear_pa=shear)
                integral = validated_integrate_rectangle(evaluator, beta_box, depth_box,
                                                        order=taylor_order, precision=precision)
                cells.append(ColdTensorCellReceipt(beta_box, depth_box, layer, lam, shear,
                    integral.polynomial_integral_bounds, integral.remainder_absolute_upper_bounds,
                    integral.integral_bounds))
                for index, value in enumerate(integral.integral_bounds):
                    quarter[index] = _rounded_interval(quarter[index]+value, precision)
                    tails[index] += integral.remainder_absolute_upper_bounds[index]
        if body.endpoint_archive_area_bounds_sr != I(0):
            strip = _integrate_endpoint_strip(body, coordinates_m, layer, z0, z1, lam, shear,
                                             precision, taylor_order)
            removals.append(strip)
            for index, value in enumerate(strip.correction_integral_bounds):
                correction[index] = _rounded_interval(correction[index]+value, precision)
        removed_measures = tuple(("paired_curved_removed", area)
                                 for area in body.paired_curved_removed_area_bounds_sr)
        for kind, area in removed_measures:
            if area == I(0):
                continue
            theta_width = body.paired_curved_cap_theta_width_bounds_radians
            beta_upper = body.paired_curved_beta_upper_bounds_radians
            if type(theta_width) is not I or type(beta_upper) is not I:
                raise ValueError("The curved correction requires its true swept narrow-cap source")
            sine = I(max(F(0), 1-theta_width.hi**2/2), 1)
            cosine = I(0, theta_width.hi)
            values = _bounded_primary_values(body, coordinates_m, I(0, beta_upper.hi), I(z0, z1),
                precision=precision, lam=lam, shear=shear, sine_bounds=sine, cosine_bounds=cosine,
                order=0, include_angular_measure=False)
            bounds = tuple(value.constant_bounds for value in values)
            multiplier = area*body.radius_m**2*(z1-z0)
            corrected = tuple(_rounded_interval(value*multiplier, precision) for value in bounds)
            removals.append(ColdTensorCorrectionReceipt(kind, layer, area, I(z0, z1), bounds, corrected,
                theta_width, I(0, beta_upper.hi)))
            for index, value in enumerate(corrected):
                correction[index] = _rounded_interval(correction[index]+value, precision)
    result = validate_native_cold_tensor_integral(_raw_receipt(body, material, coordinates_m,
        tuple(quarter), tuple(correction), len(cells), tuple(tails), taylor_order, precision,
        tuple(cells), tuple(removals)), body=body, material=material)
    _INTEGRAL_CACHE[cache_key] = result
    return result
