"""Validated rectangles against exact moments and independent antiderivatives."""
from decimal import Decimal, localcontext
from dataclasses import replace
from fractions import Fraction as F
from math import factorial

import pytest

from tectonics.geometric_native_pose import FractionInterval as I


def _api():
    from tectonics import geometric_native_curved_certified_integration as module
    return module


def _integrate(evaluator, theta=I(0, F(1, 5)), depth=I(F(1), F(3)), **kwargs):
    return _api().validated_integrate_rectangle(evaluator, theta, depth, **kwargs)


def _moment(bounds, exponent):
    return (bounds.hi**(exponent+1)-bounds.lo**(exponent+1))/(exponent+1)


def _contains(actual, expected):
    if type(expected) is I:
        assert actual.lo <= expected.lo <= expected.hi <= actual.hi
    else:
        point = F(expected)
        assert actual.lo <= point <= actual.hi


def _sine(value, cosine=False):
    # Independent alternating series, the first omitted term bounds its tail.
    first = 0 if cosine else 1
    total = sum(((-1)**k*value**(2*k+first)/factorial(2*k+first)
        for k in range(100)), F(0))
    tail = abs(value)**(200+first)/factorial(200+first)
    return I(total-tail, total+tail)


def test_rational_polynomial_integral_matches_independent_separable_moments():
    theta, depth = I(F(-1, 5), F(2, 7)), I(F(1, 3), F(7, 4))
    result = _integrate(lambda t, z, a: (3*t**2*z-2*t*z**3+F(5, 7), t**3*z**2),
        theta, depth, order=6, precision=160)
    expected = (3*_moment(theta, 2)*_moment(depth, 1)
        -2*_moment(theta, 1)*_moment(depth, 3)+F(5, 7)*theta.width*depth.width,
        _moment(theta, 3)*_moment(depth, 2))
    for actual, exact in zip(result.integral_bounds, expected):
        _contains(actual, exact)
        assert actual.width < max(F(1), abs(exact))*F(1, 10**150)
    assert result.remainder_absolute_upper_bounds == (F(0), F(0))


def test_tensor_degrees_keep_complete_mixed_polynomials_and_both_axis_remainders():
    theta, depth = I(F(-1, 10), F(1, 5)), I(F(1, 3), F(2, 3))
    result = _api().validated_integrate_tensor_rectangle(
        lambda t, z, a: (t**2*z**3+3*t*z**2, t**4*z**6),
        theta, depth, angle_order=2, depth_order=3, precision=160)
    exact = (_moment(theta, 2)*_moment(depth, 3)+3*_moment(theta, 1)*_moment(depth, 2),
        _moment(theta, 4)*_moment(depth, 6))
    for actual, expected in zip(result.integral_bounds, exact):
        _contains(actual, expected)
    assert result.remainder_absolute_upper_bounds[0] == 0
    assert all(value > 0 for value in result.remainder_axis_absolute_upper_bounds[1])
    assert result.whole_rectangle_derivative_remainder_included is True


@pytest.mark.parametrize("coefficients, expected", [
    ((F(0), F(1), F(1)), (F(-4, 3), F(4, 3), F(2))),
    ((F(0), F(0), F(1)), (F(-1), F(2, 3), F(2))),
    ((F(-2), F(0), F(0)), (F(-8), F(16), F(4))),
    ((F(2), F(0), F(0)), (F(0), F(0), F(0))),
    ((F(0), F(0), F(0)), (F(0), F(0), F(4))),
])
def test_exact_clipped_affine_moments_match_independent_triangle_and_stripe_integrals(coefficients, expected):
    # The diagonal half-square is one triangle of area 2 and affine vertex
    # values (-2,0,0); the stripe follows direct integration of y on [-1,0].
    result = _api().integrate_clipped_affine_unit_square(*coefficients)
    assert result == expected


def test_six_hundred_digit_closed_corner_keeps_its_exact_positive_area_and_energy():
    epsilon = F(1, 10**600)
    result = _api().integrate_clipped_affine_unit_square(2-epsilon, F(1), F(1))
    # Triangle legs epsilon give area epsilon²/2, with gap vertex values
    # (-epsilon,0,0). This follows from simplex moments, without quadrature.
    assert result == (-epsilon**3/6, epsilon**4/12, epsilon**2/2)
    assert result[1] > 0 and result[2] > 0
    assert float(result[1]) == float(result[2]) == 0.


def test_clipped_moments_are_invariant_under_square_reflection_and_axis_exchange():
    coefficients = (F(-2, 7), F(3, 5), F(-7, 9))
    expected = _api().integrate_clipped_affine_unit_square(*coefficients)
    for a, b, c in ((coefficients[0], coefficients[2], coefficients[1]),
            (coefficients[0], -coefficients[1], coefficients[2]),
            (coefficients[0], coefficients[1], -coefficients[2])):
        assert _api().integrate_clipped_affine_unit_square(a, b, c) == expected


def test_true_sine_and_cosine_integrals_include_the_analytic_taylor_remainder():
    theta, depth = I(0, F(1, 3)), I(0, F(1, 4))
    result = _integrate(lambda t, z, a: (a.sin(t)*z, a.cos(t)*a.sin(z)),
        theta, depth, order=14, precision=160)
    expected = ((1-_sine(theta.hi, True))*_moment(depth, 1),
        _sine(theta.hi)*(1-_sine(depth.hi, True)))
    for actual, exact in zip(result.integral_bounds, expected):
        _contains(actual, exact)
    assert all(value > 0 for value in result.remainder_absolute_upper_bounds)
    assert result.whole_rectangle_derivative_remainder_included is True


def test_derivatives_over_whole_rectangle_bound_a_rapidly_changing_tail():
    theta, depth = I(F(1, 2), F(19, 20)), I(0, F(1, 7))
    result = _integrate(lambda t, z, a: (1/(1-t)**2,), theta, depth,
        order=0, precision=160)
    # Centre samples alone underestimate the integral by more than a factor 2.
    exact = (1/(1-theta.hi)-1/(1-theta.lo))*depth.width
    midpoint_rectangle = theta.width*depth.width/(1-theta.midpoint)**2
    assert exact > 2*midpoint_rectangle
    _contains(result.integral_bounds[0], exact)
    assert result.remainder_absolute_upper_bounds[0] >= abs(exact-midpoint_rectangle)


def test_real_cube_root_integral_matches_independent_decimal_antiderivative():
    theta, depth = I(0, F(1, 5)), I(0, F(1, 6))
    result = _integrate(lambda t, z, a: (a.cbrt(8+t+2*z),),
        theta, depth, order=12, precision=160)
    with localcontext() as context:
        context.prec = 240
        def d(value):
            return Decimal(value.numerator)/Decimal(value.denominator)
        power = Decimal(7)/Decimal(3)
        def primitive(t, z):
            return (Decimal(8)+d(t)+2*d(z))**power*Decimal(9)/Decimal(56)
        exact = (primitive(theta.hi, depth.hi)-primitive(theta.lo, depth.hi)
            -primitive(theta.hi, depth.lo)+primitive(theta.lo, depth.lo))
        _contains(result.integral_bounds[0], exact)


def test_uncertain_literal_coefficient_is_retained_through_the_integral():
    coefficient = I(F(-2), F(-1))
    theta, depth = I(0, F(1, 8)), I(0, F(3, 7))
    result = _integrate(lambda t, z, a: (a.constant(coefficient)*(1+t*z),), theta, depth,
        order=4, precision=160)
    positive_measure = theta.width*depth.width+_moment(theta, 1)*_moment(depth, 1)
    _contains(result.integral_bounds[0], coefficient*positive_measure)


def test_six_hundred_digit_force_and_energy_are_not_replaced_by_zero():
    epsilon = F(1, 10**600)
    theta, depth = I(0, F(1, 10**30)), I(0, F(2))
    result = _integrate(lambda t, z, a: (epsilon*(1+t*z), epsilon**2*(1+t*z)**2),
        theta, depth, order=6, precision=160)
    exact_force = epsilon*(theta.width*depth.width+_moment(theta, 1)*_moment(depth, 1))
    exact_energy = epsilon**2*(theta.width*depth.width
        +2*_moment(theta, 1)*_moment(depth, 1)+_moment(theta, 2)*_moment(depth, 2))
    for actual, expected in zip(result.integral_bounds, (exact_force, exact_energy)):
        _contains(actual, expected)
        assert actual.lo > 0 and float(actual.hi) == 0.
        assert actual.width < abs(expected)*F(1, 10**150)


def test_exact_zero_reactions_remain_zero_without_creating_interval_noise():
    result = _integrate(lambda t, z, a: (0, 0, 0), order=8)
    assert result.integral_bounds == (I(0),)*3
    assert result.remainder_absolute_upper_bounds == (F(0),)*3


@pytest.mark.parametrize("kind", ["sqrt", "cbrt"])
def test_factorial_normalized_root_chain_derivatives_match_independent_algebra(kind):
    algebra = _api().TaylorAlgebra(4, 160)
    x = algebra.variable(F(0), coordinate=0)
    y = algebra.variable(F(0), coordinate=1)
    base, exponent, root = (F(4), F(1, 2), F(2)) if kind == "sqrt" else (F(8), F(1, 3), F(2))
    expression = getattr(algebra, kind)(base+x+2*y)
    for i, j in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 1)):
        degree = i+j
        falling = F(1)
        for k in range(degree):
            falling *= exponent-k
        exact = root*falling*2**j/(base**degree*factorial(i)*factorial(j))
        _contains(expression.coefficient(i, j), exact)


def test_taylor_jets_cannot_mix_derivative_algebras():
    first, second = _api().TaylorAlgebra(4), _api().TaylorAlgebra(5)
    with pytest.raises(ValueError, match="mix|domain"):
        first.variable(F(0), coordinate=0)+second.variable(F(0), coordinate=0)


@pytest.mark.parametrize("evaluator", [lambda t, z, a: [t],
    lambda t, z, a: (a.sqrt(t),), lambda t, z, a: (1/t,)])
def test_invalid_or_nonsmooth_rectangle_expression_refuses(evaluator):
    with pytest.raises(ValueError):
        _integrate(evaluator, I(-F(1), F(1)), I(0, F(1)), order=4)


def _curved_source(*, preload=F(-1, 4), endpoint=I(F(1, 100))):
    from tectonics.geometric_native_curved_correlated_kernel import prepare_curved_correlated_reference
    return prepare_curved_correlated_reference(radius_m=F(100), donor_radial_before_m=F(0),
        receiver_radial_before_m=F(0), receiver_depth_bounds_m=I(0, F(1)),
        angle_bounds_radians=I(0, endpoint.hi),
        start_direction_bounds=(I(1), I(0), I(0)),
        tangent_direction_bounds=(I(0), I(1), I(0)),
        normal_direction_bounds=(I(0), I(0), I(1)),
        shared_depth_axis_per_m=(F(0),)*3, normal_orientation=1,
        source_preload_gap_m=preload, normal_compliances_m_pa=(F(1), F(2), F(3)),
        precision=160, order=32)


def _curved_integral(source, **changes):
    arguments = dict(donor_radial_increment_m=F(0), receiver_radial_increment_m=F(0),
        donor_owner_axis=(F(0),)*3, receiver_owner_axis=(F(0),)*3,
        time_fraction=F(1), precision=160, taylor_order=6, max_cells=2,
        gradient_absolute_width_budgets=(F(1),)*8)
    arguments.update(changes)
    return _api().certified_native_curved_contact_integral(source, **arguments)


def _closed_radial_oracle(preload, end):
    with localcontext() as context:
        context.prec = 240
        radius = Decimal(100)
        bottom = (radius**3-3*radius**2)**(Decimal(1)/Decimal(3))
        measure_per_angle = (radius**2-bottom**2)/2
        theta, gap = Decimal(end.numerator)/Decimal(end.denominator), Decimal(preload.numerator)/Decimal(preload.denominator)
        energy_density, pressure = gap**2/12, -gap/6
        energy = theta*measure_per_angle*energy_density
        force = theta*measure_per_angle*pressure
        receiver_radial_gradient = theta*energy_density*(radius-radius**2/bottom)
        return tuple(F(value) for value in (energy, force, receiver_radial_gradient))


def test_complete_closed_curved_integral_matches_independent_radial_antiderivative():
    source = _curved_source()
    result = _curved_integral(source)
    energy, pressure, radial_gradient = _closed_radial_oracle(source.source_preload_gap_m, F(1, 100))
    _contains(result.potential_bounds_j, energy)
    _contains(result.integrated_pressure_bounds_n, pressure)
    _contains(result.literal_axis_and_radial_increment_gradient_bounds[7], radial_gradient)
    _contains(result.activation_potential_bounds_j, energy)
    _contains(result.conservative_endpoint_potential_change_bounds_j, F(0))
    for part, compliance in zip(result.energy_partition_bounds_j, (F(1), F(2), F(3))):
        _contains(part, energy*compliance/6)
    assert result.closed_taylor_cell_count == 1
    assert result.unresolved_sign_cell_count == 0
    assert source.native_source_authenticated is False
    assert result.activation_history_adopted is False
    assert result.native_force_equilibrium_certified is False


def test_exact_open_whole_face_meets_zero_force_width_budgets():
    source = _curved_source(preload=F(1, 4))
    result = _curved_integral(source, gradient_absolute_width_budgets=(F(0),)*8)
    assert result.potential_bounds_j == I(0)
    assert result.integrated_pressure_bounds_n == I(0)
    assert result.literal_axis_and_radial_increment_gradient_bounds == (I(0),)*8
    assert result.own_coordinate_widths_satisfied is True
    assert result.open_cell_count == 1


def test_closed_face_six_hundred_digit_load_survives_integration_and_own_scale():
    source = _curved_source(preload=-F(1, 10**600))
    result = _curved_integral(source)
    energy, pressure, radial_gradient = _closed_radial_oracle(source.source_preload_gap_m, F(1, 100))
    for actual, expected in ((result.potential_bounds_j, energy),
            (result.integrated_pressure_bounds_n, pressure),
            (result.literal_axis_and_radial_increment_gradient_bounds[7], radial_gradient)):
        _contains(actual, expected)
        assert actual.width < abs(expected)*F(1, 10**8)
        assert float(expected) == 0.
    assert result.potential_bounds_j.lo > 0 and result.integrated_pressure_bounds_n.lo > 0
    assert result.pressure_or_area_cutoff_applied is False


def test_failed_own_coordinate_width_budget_retains_real_force_enclosures():
    result = _curved_integral(_curved_source(),
        gradient_absolute_width_budgets=(F(0),)*8, max_cells=1)
    assert result.own_coordinate_widths_satisfied is False
    assert result.potential_bounds_j.lo > 0
    assert result.literal_axis_and_radial_increment_gradient_bounds[7].hi < 0
    assert result.integration_cell_count == 1


def test_uncertain_complete_terminal_angle_tail_is_kept_in_energy_pressure_and_forces():
    from tectonics.geometric_native_curved_correlated_kernel import _snapshot_hash, _source_snapshot
    endpoint = I(F(1, 100), F(1, 50))
    # This deliberately declared pure frame has no native mask authority.
    # Resealing its uncertain *declared* endpoint exercises the same rigorous
    # tail path that authenticated native receipts use, without claiming a
    # native fixture or source receipt was rebuilt here.
    source = replace(_curved_source(endpoint=endpoint), native_angle_end_bounds_radians=endpoint)
    source = replace(source, source_snapshot_sha256=_snapshot_hash(_source_snapshot(source)))
    result = _curved_integral(source)
    for end in (endpoint.lo, endpoint.midpoint, endpoint.hi):
        energy, pressure, gradient = _closed_radial_oracle(source.source_preload_gap_m, end)
        _contains(result.potential_bounds_j, energy)
        _contains(result.integrated_pressure_bounds_n, pressure)
        _contains(result.literal_axis_and_radial_increment_gradient_bounds[7], gradient)
        _contains(result.activation_potential_bounds_j, energy)
    assert result.uncertain_endpoint_cell_count == 1
    assert result.integration_cell_count == 2
    assert source.native_source_authenticated is False
    assert result.physical_material_admission_authorized is False


def test_monotone_open_face_proves_zero_force_for_the_whole_uncertain_terminal_tail():
    from tectonics.geometric_native_curved_correlated_kernel import (
        prepare_curved_correlated_reference, bound_curved_correlated_cell,
        _snapshot_hash, _source_snapshot,
    )
    endpoint = I(F(1, 10**20), F(2, 10**20))
    source = prepare_curved_correlated_reference(radius_m=F(100),
        donor_radial_before_m=F(0), receiver_radial_before_m=F(0),
        receiver_depth_bounds_m=I(0, F(1)), angle_bounds_radians=I(0, endpoint.hi),
        start_direction_bounds=(I(1), I(0), I(0)),
        tangent_direction_bounds=(I(0), I(1), I(0)),
        normal_direction_bounds=(I(0), I(0), I(1)),
        shared_depth_axis_per_m=(F(0), F(1, 100), F(0)), normal_orientation=1,
        source_preload_gap_m=F(-7, 100), normal_compliances_m_pa=(F(1),)*3,
        precision=100, order=24)
    source = replace(source, native_angle_end_bounds_radians=endpoint)
    source = replace(source, source_snapshot_sha256=_snapshot_hash(_source_snapshot(source)))
    motion = dict(donor_radial_increment_m=F(0), receiver_radial_increment_m=F(0),
        donor_owner_axis=(F(0), F(1, 1000), F(0)), receiver_owner_axis=(F(0),)*3)
    raw = bound_curved_correlated_cell(source, angle_bounds_radians=I(0, endpoint.hi),
        receiver_depth_bounds_m=source.receiver_depth_bounds_m, **motion)
    assert raw.gap.lo < 0 < raw.gap.hi
    result = _api().certified_native_curved_contact_integral(source,
        gradient_absolute_width_budgets=(F(0),)*8, precision=100, taylor_order=4,
        max_cells=4, **motion)
    assert result.free_gap_bounds_m.lo > 0
    assert result.integrated_pressure_bounds_n == I(0)
    assert result.potential_bounds_j == I(0)
    assert result.literal_axis_and_radial_increment_gradient_bounds == (I(0),)*8
    assert result.gauss_newton_hessian_bounds == ((I(0),)*8,)*8
    assert result.own_coordinate_widths_satisfied is True
    assert result.uncertain_endpoint_cell_count == 1
    assert source.native_source_authenticated is False


def test_tiny_closed_depth_wedge_limits_gn_to_actual_contact_area():
    from tectonics.geometric_native_curved_correlated_kernel import (
        prepare_curved_correlated_reference, bound_curved_correlated_cell,
    )
    source = prepare_curved_correlated_reference(radius_m=F(100000),
        donor_radial_before_m=F(0), receiver_radial_before_m=F(0),
        receiver_depth_bounds_m=I(0, F(2000)), angle_bounds_radians=I(0, F(1, 10**20)),
        start_direction_bounds=(I(1), I(0), I(0)),
        tangent_direction_bounds=(I(0), I(0), I(1)),
        normal_direction_bounds=(I(0), I(-1), I(0)),
        shared_depth_axis_per_m=(F(0), F(0), F(-1, 50000)), normal_orientation=1,
        source_preload_gap_m=F(-1, 100),
        normal_compliances_m_pa=(F(1, 10**6), F(1, 10**6), F(1, 10**7)),
        precision=100, order=20)
    alpha = F(1, 5000000)
    with localcontext() as context:
        context.prec = 240
        def d(value):
            return Decimal(value.numerator)/Decimal(value.denominator)
        # At depth zero U=x and n=(2x-y)/sqrt(5). Solve the actual
        # source surface gap analytically, then retain a finite positive
        # compression wedge. This fixture is not a numerical Newton replay.
        sine, cosine = d(_sine(alpha).midpoint), d(_sine(alpha, True).midpoint)
        roof = (Decimal(200000)-Decimal(".01")*Decimal(5).sqrt())/(2*cosine-sine)
        donor_increment = F(roof-Decimal(100000))+F(2, 10**16)
    motion = dict(donor_radial_increment_m=donor_increment,
        receiver_radial_increment_m=F(0), donor_owner_axis=(F(0), F(0), alpha),
        receiver_owner_axis=(F(0),)*3)
    raw = bound_curved_correlated_cell(source, angle_bounds_radians=source.angle_bounds_radians,
        receiver_depth_bounds_m=source.receiver_depth_bounds_m, **motion)
    assert raw.gap.lo < 0 < raw.gap.hi
    budget = abs(F(1, 10**32)*donor_increment)/10**11
    result = _api().certified_native_curved_contact_integral(source,
        gradient_absolute_width_budgets=(F(1),)*6+(budget, F(1)),
        precision=100, taylor_order=3, max_cells=32, **motion)
    full_face_gn_upper = raw.gauss_newton_hessian[6][6].hi*source.angle_bounds_radians.width*source.receiver_depth_bounds_m.width
    actual_gn = result.gauss_newton_hessian_bounds[6][6]
    assert result.own_coordinate_widths_satisfied is True
    assert actual_gn.lo > 0
    assert actual_gn.hi < full_face_gn_upper/F(10**10)
    assert result.potential_bounds_j.lo > 0
    assert result.closed_taylor_cell_count > 0 and result.open_cell_count > 0
    assert result.pressure_or_area_cutoff_applied is False
