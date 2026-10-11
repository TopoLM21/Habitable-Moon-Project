"""Independent analytic/Decimal goldens for radial clipping and primitives."""
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

from tectonics.genesis_volume_geometry import ConicalCohortGeometry
from tectonics.genesis_volume_material_sections import prepare_cohort_material_section
from tectonics.genesis_volume_exact_lambda import (
    ExactLambdaRefusal, exact_polynomial_moments, integrate_polynomial,
    poly_add, poly_evaluate, poly_multiply, prepare_exact_lambda_source,
    prepare_fixed_N_interval, reference_side_volumes_m3,
)


LD = np.longdouble


def _section(*, normal=(1., 0., 0.), offset=2., displacement=None,
             lower=.25, upper=.75, triangle=None):
    X = 4*np.eye(3) if triangle is None else np.asarray(triangle, float)
    coefficient = 2.
    volume = np.linalg.det(X)*(upper**3-lower**3)/6
    cohort = ConicalCohortGeometry(17, 0, 2, (1-upper**3)/coefficient,
        (1-lower**3)/coefficient, 2*volume, volume, tuple(map(tuple, X)),
        upper, lower, (), "original-owner", "original-material")
    U = np.zeros((3, 6)) if displacement is None else displacement
    return prepare_cohort_material_section(cohort, U, normal, offset,
                                           conical_mass_coefficient=coefficient)


def _interval(*, offset=2., center=(2., 0., 0.), radius=.5, N=(1., 0., 0.), **kwargs):
    source = prepare_exact_lambda_source(_section(offset=offset, **kwargs), center, radius)
    return prepare_fixed_N_interval(source, N)


def _decimal(value):
    numerator, denominator = LD(value).as_integer_ratio()
    return Decimal(numerator)/Decimal(denominator)


def _decimal_piecewise(coefficients, h, root, sign=1, power=1):
    # Direct endpoint antiderivatives on separate bank intervals are independent
    # of the producer's centered-parity/factored signed-moment formula.
    with localcontext() as context:
        context.prec = 110
        h, root = _decimal(h), _decimal(root)
        result = Decimal(0)
        for k, coefficient in enumerate(coefficients):
            c = _decimal(coefficient)
            left = (root**(k+1)-Decimal(-1)**(k+1))/Decimal(k+1)
            right = (Decimal(1)-root**(k+1))/Decimal(k+1)
            if power == 0:
                result += c*h*(left+right)
            elif power == 1:
                result += c*h*Decimal(sign)*(right-left)/2
            else:
                result += c*h*(left+right)/4
        return LD(str(result))


def test_exact_ball_roots_original_bounds_and_noncentered_literal_gamma():
    interval = _interval(offset=2.125)
    assert interval.lambda_bounds == (LD(.375), LD(.625))
    assert interval.lambda_mid == LD(.5)
    assert interval.lambda_halfwidth == LD(.125)
    assert interval.closest_lambda == LD(.5)
    assert interval.sphere_margin_m2 == LD(.25)
    np.testing.assert_array_equal(interval.birth_relative_coefficients,
                                  ((0., 0., 0.), (.5, 0., 0.)))
    np.testing.assert_array_equal(interval.literal_phi_coefficients, (-.125, .5))
    np.testing.assert_array_equal(interval.literal_phi_lambda_coefficients, (-2.125, 4.))
    assert interval.gamma_root_xi == LD(.25)
    assert interval.constant_side is None
    assert interval.reference_determinant_m3 == 64
    assert not interval.geometry_operand_accuracy_certified
    assert not interval.angular_integral_convergence_certified
    assert not interval.physical_creation_admitted


def test_original_radial_cohort_cap_is_retained_exactly():
    interval = _interval(center=(2.75, 0., 0.))
    assert interval.lambda_bounds == (LD(.5625), LD(.75))
    assert interval.lambda_mid == LD(.65625)
    assert interval.lambda_halfwidth == LD(.09375)
    assert interval.relative_lambda_mid_to_original_lower == LD(.40625)
    assert interval.relative_lambda_mid_to_original_upper == LD(.09375)
    at_lower = interval.birth_relative_coefficients[0]-interval.birth_relative_coefficients[1]
    at_upper = interval.birth_relative_coefficients[0]+interval.birth_relative_coefficients[1]
    np.testing.assert_array_equal(at_lower, (-.5, 0., 0.))
    np.testing.assert_array_equal(at_upper, (.25, 0., 0.))


def test_empty_outside_original_cohort_and_exact_sphere_tangency():
    assert _interval(center=(10., 0., 0.)) is None
    assert _interval(center=(2., .5, 0.)) is None
    assert _interval(center=(2., .5001, 0.)) is None


def test_positive_near_tangent_sliver_is_not_discarded_by_an_arithmetic_floor():
    y = LD(.5)-np.ldexp(LD(1), -60)
    interval = _interval(center=np.array((LD(2), y, LD(0))))
    assert interval is not None
    assert interval.exact_sign_fallback_used
    with localcontext() as context:
        context.prec = 110
        expected = ((Decimal('.25')-_decimal(y)**2)/16).sqrt()
    assert abs(interval.lambda_halfwidth-LD(str(expected))) <= 8*np.finfo(LD).eps*LD(str(expected))
    assert interval.lambda_halfwidth > 0


def test_original_cohort_endpoint_sliver_keeps_its_own_small_width():
    delta = np.ldexp(LD(1), -56)
    interval = _interval(center=np.array((LD(3.5)-delta, LD(0), LD(0))))
    assert interval is not None
    assert interval.lambda_halfwidth == delta/8
    assert interval.lambda_bounds[1] == LD(.75)
    assert integrate_polynomial(interval, np.array([LD(1e-30)])).value > 0


@pytest.mark.parametrize("normal,sign", (((1., 0., 0.), 1), ((-1., 0., 0.), -1)))
@pytest.mark.parametrize("side_power", (0, 1, 2))
def test_all_polynomial_degrees_through_sixteen_against_decimal_piecewise_goldens(normal, sign, side_power):
    offset = 2.125*sign
    interval = _interval(normal=normal, offset=offset)
    for degree in range(17):
        coefficients = np.array([(-1)**k*(k+1) for k in range(degree+1)], dtype=LD)
        expected = _decimal_piecewise(coefficients, LD(.125), LD(.25), sign, side_power)
        result = integrate_polynomial(interval, coefficients, side_power=side_power)
        assert abs(result.value-expected) <= result.roundoff_bound+8*np.finfo(LD).eps*abs(expected)
        assert result.absolute_operand_scale >= abs(result.value)
        assert not result.geometry_operand_accuracy_certified
        assert not result.coefficient_construction_accuracy_certified


def test_h_squared_quarter_rule_and_signed_symmetry_zeros_are_exact():
    interval = _interval()
    unsigned = exact_polynomial_moments(interval, 16, side_power=0)
    signed = exact_polynomial_moments(interval, 16, side_power=1)
    squared = exact_polynomial_moments(interval, 16, side_power=2)
    np.testing.assert_array_equal(squared, unsigned/4)
    np.testing.assert_array_equal(unsigned[1::2], np.zeros(8, dtype=LD))
    np.testing.assert_array_equal(signed[::2], np.zeros(9, dtype=LD))
    zero = integrate_polynomial(interval, np.zeros((17, 3, 2), dtype=LD), side_power=1)
    np.testing.assert_array_equal(zero.value, np.zeros((3, 2)))
    np.testing.assert_array_equal(zero.absolute_operand_scale, np.zeros((3, 2)))
    np.testing.assert_array_equal(zero.roundoff_bound, np.zeros((3, 2)))


def test_near_endpoint_signed_odd_moments_preserve_positive_cap():
    distance = np.ldexp(LD(1), -40)
    interval = _interval(offset=float(LD(2.5)-distance/2))
    assert interval.gamma_root_xi == 1-distance
    moments = exact_polynomial_moments(interval, 16, side_power=1)
    for k in range(1, 17, 2):
        coefficients = np.zeros(k+1, dtype=LD)
        coefficients[k] = 1
        expected = _decimal_piecewise(coefficients, LD(.125), 1-distance)
        assert moments[k] > 0
        assert abs(moments[k]-expected) <= 32*np.finfo(LD).eps*abs(expected)
    banks = reference_side_volumes_m3(interval)
    assert 0 < banks[LD(.5)] < banks[LD(-.5)]*distance
    unsigned = integrate_polynomial(interval,
        np.array((interval.lambda_mid**2, 2*interval.lambda_mid*interval.lambda_halfwidth,
                  interval.lambda_halfwidth**2))*64).value
    assert abs(sum(banks.values())-unsigned) <= 16*np.finfo(LD).eps*unsigned


def test_gamma_cap_below_root_coordinate_ulp_is_retained_before_rounding():
    tiny = LD('1e-30')
    displacement = np.zeros((3, 6))
    displacement[1, 0] = 1
    interval = _interval(offset=2.5, displacement=displacement,
                         N=np.array((LD(1), tiny, LD(0))))
    assert interval.gamma_root_xi == 1  # The root itself cannot retain 1-2e-30.
    assert interval.constant_side is None
    assert interval.gamma_endpoint_distance_xi == 2*tiny
    odd = exact_polynomial_moments(interval, 16, side_power=1)[1::2]
    np.testing.assert_allclose(odd, np.full(8, interval.lambda_halfwidth*2*tiny), rtol=1e-18, atol=0)
    assert reference_side_volumes_m3(interval)[LD(.5)] > 0


@pytest.mark.parametrize("offset,side", ((1., .5), (3., -.5)))
def test_root_outside_interval_reduces_to_literal_constant_side(offset, side):
    interval = _interval(offset=offset)
    assert interval.constant_side == side
    np.testing.assert_array_equal(exact_polynomial_moments(interval, side_power=1),
                                  side*exact_polynomial_moments(interval))
    banks = reference_side_volumes_m3(interval)
    assert banks[LD(-side)] == 0
    assert banks[LD(side)] > 0


def test_constant_nonzero_gamma_slope_and_coplanar_refusal():
    interval = _interval(normal=(0., 1., 0.), offset=-.125)
    assert interval.gamma_root_xi is None
    assert interval.constant_side == .5
    np.testing.assert_array_equal(exact_polynomial_moments(interval, side_power=1),
                                  exact_polynomial_moments(interval)/2)
    coplanar = _interval(normal=(0., 1., 0.), offset=0.)
    assert coplanar.constant_side == 0
    assert exact_polynomial_moments(coplanar, side_power=0)[0] > 0
    with pytest.raises(ExactLambdaRefusal, match="whole line"):
        exact_polynomial_moments(coplanar, side_power=1)
    with pytest.raises(ExactLambdaRefusal, match="whole line"):
        reference_side_volumes_m3(coplanar)


def test_old_float64_chart_sum_is_preserved_before_longdouble_promotion():
    X = np.eye(3)*1e16
    displacement = np.zeros((3, 6))
    displacement[:, 3:] = np.eye(3)
    section = _section(triangle=X, displacement=displacement, offset=5e15)
    source = prepare_exact_lambda_source(section, (5e15, 0., 0.), 100.)
    expected = np.asarray(section.triangle_m+section.slope_P_m, dtype=LD)
    np.testing.assert_array_equal(source.birth_C_m, expected)
    assert source.birth_C_m[0, 0] == LD(1e16)
    assert LD(section.triangle_m[0, 0])+LD(section.slope_P_m[0, 0]) != source.birth_C_m[0, 0]


def test_literal_gamma_world_dot_cancellation_uses_exact_source_residual():
    section = _section(triangle=np.eye(3)*1e16, normal=(1., 1., 1.), offset=5e15)
    source = prepare_exact_lambda_source(section, (5e15, 0., 0.), 100.)
    interval = prepare_fixed_N_interval(source, (1., 0., 0.))
    with localcontext() as context:
        context.prec = 110
        constant = _decimal(section.normal_xyz[0])*Decimal('5e15')-_decimal(section.offset_m)
        slope_xi = _decimal(section.normal_xyz[0])*Decimal('1e16')*_decimal(interval.lambda_halfwidth)
        root = -constant/slope_xi
    assert interval.literal_phi_coefficients[0] == LD(str(constant))
    assert abs(interval.gamma_root_xi-LD(str(root))) <= 4*np.finfo(LD).eps*abs(LD(str(root)))
    assert interval.literal_phi_coefficients[0] != 0
    assert abs(interval.literal_gamma_root_residual_m) <= 8*np.finfo(LD).eps*abs(interval.literal_phi_coefficients[0])
    assert interval.sphere_endpoint_residuals_m2.shape == (2,)
    assert not interval.geometry_operand_accuracy_certified


def test_accepted_thin_cohort_source_offsets_share_the_native_midpoint_chart():
    # Literal accepted cohort1 chart values, parent raw62579a0bad128830...
    # This small computational regression reproduces its geometry, without
    # restoring a planet or claiming a physical-source admission certificate.
    X = np.array((
        (-2779334.7908648686, 4497293.439988639, 445.91590265775136),
        (-4276968.434620413, 2642996.844734693, 1634765.7483262795),
        (-1633425.240690996, 4276593.808315874, 2644431.4598161923)))
    A = np.array(((.917386949695752, -1.5080456939747349, -.04374257440034894),
                  (.6531170464550741, -.3473096701875744, -.4087710154277778),
                  (.029878477215076627, -.12691612315928352, -.2873058353576092)))
    P = np.array(((90.89980337615366, -148.018601273836, -1.9079926572857926),
                  (141.4353788370711, -86.36862155791569, -57.48793851529024),
                  (53.28246077641166, -140.4433063729462, -90.77453442666696)))
    center = np.array((-2896565.8903007517, 3805614.5844821963, 1426542.6686451025))
    section = _section(triangle=X, displacement=np.concatenate((A, P), axis=1),
                       lower=.9999932803766464, upper=1., normal=(1., 2., 3.),
                       offset=2403825.4588440103*np.sqrt(14))
    source = prepare_exact_lambda_source(section, center, 100.)
    N = np.full(3, LD(1)/3, dtype=LD)
    interval = prepare_fixed_N_interval(source, N)
    assert interval is not None
    assert source.lambda_bounds[1]-source.lambda_bounds[0] < LD('7e-6')

    def fraction(value):
        numerator, denominator = LD(value).as_integer_ratio()
        return Fraction(numerator, denominator)

    def represented(value):
        with localcontext() as context:
            context.prec = 110
            return LD(str(Decimal(value.numerator)/Decimal(value.denominator)))

    u = [sum((fraction(N[i])*fraction((X+P)[i, j]) for i in range(3)), Fraction())
         for j in range(3)]
    d = [sum((fraction(N[i])*fraction((A-P)[i, j]) for i in range(3)), Fraction())
         for j in range(3)]
    midpoint, halfwidth = fraction(interval.lambda_mid), fraction(interval.lambda_halfwidth)
    relative0 = [d[j]+midpoint*u[j]-fraction(center[j]) for j in range(3)]
    relative1 = [halfwidth*u[j] for j in range(3)]
    expected = np.array([[represented(x) for x in row] for row in (relative0, relative1)], dtype=LD)
    np.testing.assert_allclose(interval.birth_relative_coefficients, expected,
                               rtol=4*np.finfo(LD).eps, atol=0)
    assert interval.relative_lambda_mid_to_original_lower == interval.lambda_mid-source.lambda_bounds[0]
    assert interval.relative_lambda_mid_to_original_upper == source.lambda_bounds[1]-interval.lambda_mid
    phi0 = sum((fraction(section.normal_xyz[j])*(d[j]+midpoint*u[j])
                for j in range(3)), Fraction())-fraction(section.offset_m)
    phi1 = sum((fraction(section.normal_xyz[j])*halfwidth*u[j]
                for j in range(3)), Fraction())
    np.testing.assert_allclose(interval.literal_phi_coefficients,
                               (represented(phi0), represented(phi1)),
                               rtol=4*np.finfo(LD).eps, atol=0)
    # Evaluate another native xi with Decimal source coordinates rather than
    # the kernel's closest-point decomposition or polynomial helper.
    xi = LD('.37')
    expected_point = np.array([represented(relative0[j]+fraction(xi)*relative1[j])
                               for j in range(3)], dtype=LD)
    point = interval.birth_relative_coefficients[0]+xi*interval.birth_relative_coefficients[1]
    own_operands = np.abs(interval.birth_relative_coefficients[0])+abs(xi)*np.abs(interval.birth_relative_coefficients[1])
    assert np.all(np.abs(point-expected_point) <= 8*np.finfo(LD).eps*own_operands)
    assert not interval.geometry_operand_accuracy_certified


def test_source_binding_array_immutability_and_public_float64_N_are_preserved():
    source = prepare_exact_lambda_source(_section(), (2., 0., 0.), .5)
    original = np.array((1/3, 1/3, 1/3), dtype=np.float64)
    interval = prepare_fixed_N_interval(source, original)
    # This N line misses the small ball, but it is a valid original public N.
    assert interval is None
    interval = prepare_fixed_N_interval(source, (1., 0., 0.))
    for array in (source.birth_C_m, source.birth_D_m, interval.barycentric,
                  interval.birth_relative_coefficients, interval.literal_phi_coefficients):
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.base.setflags(write=True)
    forged = replace(source.section, offset_m=1.875)
    with pytest.raises(ExactLambdaRefusal, match="immutable geometric binding"):
        prepare_exact_lambda_source(forged, (2., 0., 0.), .5)


def test_polynomial_helpers_first_axis_broadcast_and_vector_primitive():
    scalar = np.array((1., 2., 3.))
    vector = np.array(((4., 5.), (6., 7.)))
    np.testing.assert_array_equal(poly_add(scalar, vector), ((5., 6.), (8., 9.), (3., 3.)))
    product = poly_multiply(scalar, vector)
    np.testing.assert_array_equal(product, ((4., 5.), (14., 17.), (24., 29.), (18., 21.)))
    np.testing.assert_array_equal(poly_evaluate(product, .25),
                                  poly_evaluate(scalar, .25)*poly_evaluate(vector, .25))
    interval = _interval(offset=2.125)
    integrated = integrate_polynomial(interval, product, side_power=1)
    for column in range(2):
        expected = _decimal_piecewise(product[:, column], LD(.125), LD(.25))
        assert abs(integrated.value[column]-expected) <= integrated.roundoff_bound[column]


@pytest.mark.parametrize("radius", (0., -1., np.nan, np.inf, True, '1'))
def test_invalid_radius_refused(radius):
    with pytest.raises(ExactLambdaRefusal):
        prepare_exact_lambda_source(_section(), (2., 0., 0.), radius)


@pytest.mark.parametrize("N", ((-.1, .1, 1.), (.5, .25, .2), (1., np.nan, 0.),
                               (True, False, False), (1., 0.)))
def test_invalid_original_N_refused(N):
    source = prepare_exact_lambda_source(_section(), (2., 0., 0.), .5)
    with pytest.raises(ExactLambdaRefusal):
        prepare_fixed_N_interval(source, N)


@pytest.mark.parametrize("coefficients", ([], [np.nan], [True], ['1'], np.array(1.)))
def test_invalid_polynomial_refused(coefficients):
    with pytest.raises(ExactLambdaRefusal):
        integrate_polynomial(_interval(), coefficients)


def test_degree_side_shape_and_nonfinite_helpers_explicitly_refuse():
    interval = _interval()
    for degree in (-1, 17, True, 2.5):
        with pytest.raises(ExactLambdaRefusal):
            exact_polynomial_moments(interval, degree)
    for side in (-1, 3, True, .5):
        with pytest.raises(ExactLambdaRefusal):
            integrate_polynomial(interval, (1.,), side_power=side)
    with pytest.raises(ExactLambdaRefusal):
        integrate_polynomial(interval, np.ones(18))
    with pytest.raises(ExactLambdaRefusal, match="shapes"):
        poly_add(np.ones((2, 3)), np.ones((2, 4)))
    with pytest.raises(ExactLambdaRefusal, match="shapes"):
        poly_multiply(np.ones((2, 3)), np.ones((2, 4)))
    with pytest.raises(ExactLambdaRefusal):
        poly_evaluate((1.,), np.inf)
    with pytest.raises(ExactLambdaRefusal):
        poly_add()
