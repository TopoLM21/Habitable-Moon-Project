"""Finite born forces checked against exact moments and physical force scales.

The reference moments below use independent rational alternating trigonometric
series.  They do not reuse production integration, active-set partitions, or
pressure routines, and retain the complete second harmonic.
"""
from fractions import Fraction as F
from decimal import Decimal, localcontext
from math import factorial

import pytest

from tests.test_geometric_native_birth_contact_network import _case
from tectonics.geometric_native_birth_contact_continuum import BornContinuumDeclaration
from tectonics.geometric_native_birth_finite_pose import BornFinitePoseFields
from tectonics.geometric_native_birth_finite_force_bounds import BornFinitePoseJacobianBounds
from tectonics.geometric_native_birth_finite_equilibrium import (
    certify_finite_equilibrium_operands, integrate_born_finite_pose_response,
)
from tectonics.geometric_native_pose import FractionInterval as I


def _synthetic(gap, *, end=F(1, 8), gradients=None, time=I(1)):
    source = _case()["born_attachments"][0]
    count = 7
    if gradients is None:
        gradients = ((F(2), F(0), F(0), F(-2), F(0), F(0), F(5)),) + ((F(0),)*count,)*4
    rows = tuple(tuple(I(value) if type(value) is F else value for value in row) for row in gradients)
    with localcontext() as context:
        context.prec = 160
        end_decimal = str(Decimal(end.numerator)/Decimal(end.denominator))
    declaration = BornContinuumDeclaration(source.parameters, source.reference,
        "0", end_decimal, "1", (("0",)*count,)*3, (("0",)*count,)*3,
        "Independent finite five-term pressure and Jacobian test")
    norms = tuple(sum((max(abs(rows[basis][coordinate].lo), abs(rows[basis][coordinate].hi))
        for basis in range(5)), F(0)) for coordinate in range(count))
    jacobian = BornFinitePoseJacobianBounds(declaration, rows, I(0, end), time,
        (F(0),)*count, norms, norms, 160, 32)
    fields = BornFinitePoseFields(tuple(I(value) if type(value) is F else value for value in gap),
        (I(0),)*5, (I(0),)*5, time, 160, 32)
    return fields, jacobian


def _response(gap, *, end=I(F(1, 8)), density=F(1), stiffness=F(1, 2), gradients=None, **options):
    fields, jacobian = _synthetic(gap, end=end.hi, gradients=gradients)
    return integrate_born_finite_pose_response(fields, jacobian,
        measure_density_m2_per_radian=density, angle_end_bounds=end,
        normal_stiffness_pa_m=stiffness, precision=160, **options)


def _trig(value, *, cosine=False):
    # Every test angle/frequency is <=1; alternating terms decrease.  The
    # first omitted term encloses the remainder with its mathematical sign.
    if value < 0 and not cosine:
        return -_trig(-value)
    first = 0 if cosine else 1
    terms = tuple((-1)**degree*value**(2*degree+first)/factorial(2*degree+first)
        for degree in range(70))
    total = sum(terms, F(0))
    following = value**(140+first)/factorial(140+first)
    return I(total, total+following)


def _integrated_trig(frequency, end, *, sine=False):
    if not frequency:
        return I(0 if sine else end)
    if sine:
        return (1-_trig(F(frequency)*end, cosine=True))/frequency
    return _trig(F(frequency)*end)/frequency


def _basis_moment(left, right, end):
    # Basis = 1, cos(theta), sin(theta), cos(2theta), sin(2theta).
    kind = ("c", "c", "s", "c", "s")
    frequency = (0, 1, 1, 2, 2)
    l, r = frequency[left], frequency[right]
    if kind[left] == kind[right] == "c":
        return (_integrated_trig(l-r, end)+_integrated_trig(l+r, end))/2
    if kind[left] == kind[right] == "s":
        return (_integrated_trig(l-r, end)-_integrated_trig(l+r, end))/2
    if kind[left] == "s":
        l, r = r, l
    return (_integrated_trig(r+l, end, sine=True)+_integrated_trig(r-l, end, sine=True))/2


def _product_integral(left, right, end):
    return sum((l*r*_basis_moment(i, j, end) for i, l in enumerate(left)
        for j, r in enumerate(right)), I(0))


def _contains_interval(actual, expected):
    assert actual.lo <= expected.lo <= expected.hi <= actual.hi


def test_full_five_term_closed_pressure_energy_gradient_and_gn_curvature_match_independent_moments():
    gap = (F(-1), F(3, 10), F(1, 5), F(-1, 4), F(1, 7))
    gradients = (
        (F(2), F(0), F(0), F(-2), F(0), F(0), F(5)),
        (F(1, 3), F(0), F(0), F(1, 7), F(0), F(0), F(2, 9)),
        (F(-2, 11), F(0), F(0), F(1, 13), F(0), F(0), F(3, 10)),
        (F(1, 4), F(0), F(0), F(1, 5), F(0), F(0), F(-2, 7)),
        (F(-1, 3), F(0), F(0), F(-1, 8), F(0), F(0), F(3, 11)),
    )
    end, density, stiffness = F(1, 8), F(7, 3), F(5, 11)
    response = _response(gap, end=I(end), density=density, stiffness=stiffness, gradients=gradients)
    # The coefficient l1 bound proves g<0 on this complete arc.
    assert sum(map(abs, gap[1:]), F(0)) < -gap[0]
    _contains_interval(response.energy_bounds_j, density*stiffness*_product_integral(gap, gap, end)/2)
    for coordinate in range(7):
        column = tuple(row[coordinate] for row in gradients)
        _contains_interval(response.gradient_bounds[coordinate], density*stiffness*_product_integral(gap, column, end))
        for other in range(7):
            second = tuple(row[other] for row in gradients)
            _contains_interval(response.gauss_newton_hessian_bounds[coordinate][other],
                density*stiffness*_product_integral(column, second, end))
    assert response.gap_bounds_m.hi < 0
    assert response.unresolved_angle_parts == ()
    assert response.material_admission_authorized is False


def test_compression_uses_signed_pressure_and_foundation_radial_gradient_has_radius_factor():
    response = _response((F(-2),)+(F(0),)*4, density=F(7), stiffness=F(3))
    _contains_interval(response.energy_bounds_j, I(F(21, 4)))
    _contains_interval(response.gradient_bounds[0], I(F(-21, 2)))
    _contains_interval(response.gradient_bounds[3], I(F(21, 2)))
    _contains_interval(response.gradient_bounds[6], I(F(-105, 4)))
    _contains_interval(response.gauss_newton_hessian_bounds[6][6], I(F(525, 8)))


@pytest.mark.parametrize("gap", [F(0), F(1, 10**80)])
def test_exact_open_or_touching_contact_has_zero_reaction_and_energy(gap):
    response = _response((gap,)+(F(0),)*4)
    assert response.energy_bounds_j == I(0)
    assert response.gradient_bounds == (I(0),)*7
    assert all(value == I(0) for row in response.gauss_newton_hessian_bounds for value in row)


def test_arbitrarily_small_closed_force_is_retained_in_its_own_nonzero_scale():
    gap, end, density, stiffness = F(-1, 10**80), F(1, 10**30), F(10**8), F(10**10)
    response = _response((gap,)+(F(0),)*4, end=I(end), density=density, stiffness=stiffness)
    expected = density*stiffness*gap*5*end
    _contains_interval(response.gradient_bounds[6], I(expected))
    assert response.gradient_bounds[6].hi < 0
    assert response.energy_bounds_j.lo > 0
    assert response.gradient_bounds[6].width < abs(expected)*F(1, 10**100)


def test_a_tiny_second_harmonic_closing_tail_has_positive_energy_without_midpoint_or_area_cutoff():
    # g=-epsilon + 1-cos(2theta). The midpoint is open, while a tiny true
    # compression tail at theta=0 has width of order sqrt(epsilon).
    epsilon, root, end = F(1, 10**60), F(1, 10**30), F(1, 10**29)
    response = _response((1-epsilon, F(0), F(0), F(-1), F(0)), end=I(end))
    # On [0,root/4], 1-cos(2theta)=2sin(theta)^2<=epsilon/8.
    pressure_lower_magnitude = 7*epsilon/8
    energy_lower = pressure_lower_magnitude**2*(root/4)/4
    force_lower_magnitude = pressure_lower_magnitude*(root/4)*5/2
    assert response.energy_bounds_j.lo >= energy_lower
    assert response.gradient_bounds[6].hi <= -force_lower_magnitude
    assert response.energy_bounds_j.hi < epsilon**2*root
    assert response.gauss_newton_hessian_bounds[6][6].lo > 0


def test_native_endpoint_uncertainty_is_retained_in_every_force_and_energy_interval():
    end = I(F(1, 10), F(11, 100))
    response = _response((F(-1),)+(F(0),)*4, end=end, density=F(3), stiffness=F(2))
    _contains_interval(response.energy_bounds_j, 3*end)
    _contains_interval(response.gradient_bounds[6], -30*end)
    _contains_interval(response.gauss_newton_hessian_bounds[6][6], 150*end)


def test_native_coefficient_uncertainty_survives_pressure_energy_and_gradient():
    response = _response((I(F(-2), F(-1)),)+(F(0),)*4)
    _contains_interval(response.energy_bounds_j, I(F(1, 32), F(1, 8)))
    _contains_interval(response.gradient_bounds[6], I(F(-5, 8), F(-5, 16)))


@pytest.mark.parametrize("value", [0., 1., True, F(0), F(-1)])
def test_stiffness_requires_positive_exact_physical_fraction(value):
    fields, jacobian = _synthetic((F(-1),)+(F(0),)*4)
    with pytest.raises((ValueError, TypeError)):
        integrate_born_finite_pose_response(fields, jacobian,
            measure_density_m2_per_radian=F(1), angle_end_bounds=I(F(1, 8)),
            normal_stiffness_pa_m=value)


@pytest.mark.parametrize("value", [0., 1., True, F(0), F(-1)])
def test_measure_requires_positive_exact_fraction_and_cannot_be_silently_cut_off(value):
    fields, jacobian = _synthetic((F(-1),)+(F(0),)*4)
    with pytest.raises((ValueError, TypeError)):
        integrate_born_finite_pose_response(fields, jacobian,
            measure_density_m2_per_radian=value, angle_end_bounds=I(F(1, 8)),
            normal_stiffness_pa_m=F(1))


def test_response_refuses_fields_and_jacobian_from_different_clocks():
    fields, _ = _synthetic((F(-1),)+(F(0),)*4, time=I(0, 1))
    _, jacobian = _synthetic((F(-1),)+(F(0),)*4, time=I(1))
    with pytest.raises(ValueError, match="clock|time"):
        integrate_born_finite_pose_response(fields, jacobian,
            measure_density_m2_per_radian=F(1), angle_end_bounds=I(F(1, 8)),
            normal_stiffness_pa_m=F(1))


def test_full_so3_force_changes_from_affine_force_and_matches_independent_scalar_rotation():
    from tests.test_geometric_native_birth_finite_force_bounds import _sources, _jacobian
    from tectonics.geometric_native_birth_finite_pose import bound_born_finite_pose_fields

    alpha, end = F(1, 4), F(1, 8)
    declaration, _, affine, values = _sources(end=end,
        donor_axis=(F(0), F(0), alpha), receiver_axis=(F(0),)*3, precision=160)
    fields = bound_born_finite_pose_fields(**values)
    jacobian = _jacobian(declaration, values, I(0, end))
    response = integrate_born_finite_pose_response(fields, jacobian,
        measure_density_m2_per_radian=F(1), angle_end_bounds=I(end),
        normal_stiffness_pa_m=F(1, 2), precision=160)
    sine, cosine = _trig(alpha), _trig(alpha, cosine=True)
    lever = values["radius_m"]+values["donor_radial_before_m"]+values["donor_radial_increment_m"]
    s, c = values["sine_dip"], values["cosine_dip"]
    base = values["birth_gap_free_m"]-c*(values["donor_radial_increment_m"]-values["receiver_radial_increment_m"])
    # Independent scalar Rz gives r dot (Rz r-r)=(cos(alpha)-1)sin²theta
    # and h dot (Rz r-r)=sin(alpha)sin(theta).
    normal = (base+lever*c*(1-cosine)/2, I(0), -lever*s*sine,
        -lever*c*(1-cosine)/2, I(0))
    rotation_gradient = (lever*c*sine/2, I(0), -lever*s*cosine,
        -lever*c*sine/2, I(0))
    _contains_interval(response.energy_bounds_j, _product_integral(normal, normal, end)/4)
    _contains_interval(response.gradient_bounds[2], _product_integral(normal, rotation_gradient, end)/2)
    _contains_interval(response.gradient_bounds[5], -_product_integral(normal, rotation_gradient, end)/2)
    affine_energy = F(Decimal(affine.state.stored_energy_decimal))
    assert not response.energy_bounds_j.lo <= affine_energy <= response.energy_bounds_j.hi
    assert response.material_admission_authorized is False


def test_each_coordinate_certificate_uses_its_own_lower_force_scale():
    large, tiny, tolerance = F(10**40), F(1, 10**80), F(1, 10**10)
    certificate = certify_finite_equilibrium_operands(((I(large), I(-large)),
        (I(tiny), I(-tiny))), relative_tolerance=tolerance)
    assert certificate.equilibrium_certified is True
    assert certificate.own_scale_lower_bounds == (2*large, 2*tiny)
    assert certificate.relative_residual_upper_bounds == (F(0), F(0))
    # The huge coordinate cannot hide a 10% residual on the tiny contact.
    with pytest.raises(ValueError, match="coordinate|budget|force"):
        certify_finite_equilibrium_operands(((I(large), I(-large)),
            (I(tiny), I(-tiny*F(9, 10)))), relative_tolerance=tolerance)


def test_zero_force_coordinate_requires_exact_zero_residual():
    certificate = certify_finite_equilibrium_operands(((I(0), I(0)),),
        relative_tolerance=F(1, 10**10))
    assert certificate.own_scale_lower_bounds == (F(0),)
    assert certificate.relative_residual_upper_bounds == (F(0),)
    assert certificate.equilibrium_certified is True
    # A zero midpoint of a nonzero interval cannot certify a dormant port.
    with pytest.raises(ValueError, match="coordinate|budget|force|zero"):
        certify_finite_equilibrium_operands(((I(F(-1, 10**100), F(1, 10**100)),),),
            relative_tolerance=F(1, 10**10))


def test_force_scale_uses_interval_lower_magnitude_and_retains_uncertainty_in_residual():
    certificate = certify_finite_equilibrium_operands(((I(F(9), F(11)), I(-10)),),
        relative_tolerance=F(1, 10))
    assert certificate.own_scale_lower_bounds == (F(19),)
    _contains_interval(certificate.residual_bounds[0], I(-1, 1))
    assert certificate.relative_residual_upper_bounds[0] >= F(1, 19)
    assert certificate.relative_residual_upper_bounds[0] < F(1, 18)
    with pytest.raises(ValueError, match="coordinate|budget|force"):
        certify_finite_equilibrium_operands(((I(F(9), F(11)), I(-10)),),
            relative_tolerance=F(1, 100))
