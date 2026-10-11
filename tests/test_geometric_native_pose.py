"""Independent full Rodrigues and exact algebra checks of native pose bounds."""
from decimal import Context, Decimal, ROUND_DOWN, localcontext
from fractions import Fraction

import numpy as np
import pytest

from tectonics.geometric_intrinsic_geometry import RotationIncrement
from tectonics.geometric_native_pose import (
    FractionInterval, certify_rotation_history, interval_atan, interval_pi,
    interval_sqrt,
)
from test_geometric_precision import NEXT_DT_MYR, NEXT_OMEGA


def _dot(left, right):
    return sum((a*b for a, b in zip(left, right)), 0)


def _cross(left, right):
    return (left[1]*right[2]-left[2]*right[1],
            left[2]*right[0]-left[0]*right[2],
            left[0]*right[1]-left[1]*right[0])


def _trigonometric(angle):
    square = angle*angle
    sine, cosine = angle, Decimal(1)
    sine_term, cosine_term = sine, cosine
    for index in range(1, 10000):
        sine_term *= -square/Decimal(2*index*(2*index+1))
        cosine_term *= -square/Decimal((2*index-1)*(2*index))
        next_sine, next_cosine = sine+sine_term, cosine+cosine_term
        if next_sine == sine and next_cosine == cosine:
            return sine, cosine
        sine, cosine = next_sine, next_cosine
    raise AssertionError("Independent Rodrigues oracle did not converge")


def _analytic_matrix(history, precision=260):
    """Direct axis-angle action on columns; production uses quaternion bounds."""
    with localcontext(Context(prec=precision)):
        columns = tuple(tuple(Decimal(int(i == j)) for i in range(3))
                        for j in range(3))
        for step in history:
            vector = tuple(Decimal.from_float(value) for value in step.omega_rad_per_myr)
            magnitude = _dot(vector, vector).sqrt()
            if not magnitude:
                continue
            axis = tuple(value/magnitude for value in vector)
            sine, cosine = _trigonometric(magnitude*Decimal.from_float(step.dt_myr))
            transformed = []
            for column in columns:
                crossing, projection = _cross(axis, column), _dot(axis, column)
                transformed.append(tuple(column[i]*cosine+crossing[i]*sine+
                    axis[i]*projection*(Decimal(1)-cosine) for i in range(3)))
            columns = tuple(transformed)
        return tuple(tuple(columns[j][i] for j in range(3)) for i in range(3))


def _assert_analytic_containment(certificate, history, oracle_precision=260):
    expected = _analytic_matrix(history, oracle_precision)
    # This oracle uncertainty is far smaller than the requested interval
    # accuracy. It handles a mathematically zero matrix entry's final Decimal
    # cancellation without allowing an incorrect nonzero physical component.
    oracle_error = Fraction(1, 10**(oracle_precision-20))
    for i in range(3):
        for j in range(3):
            value = Fraction(expected[i][j])
            enclosure = certificate.intervals[i][j]
            assert enclosure.lo-oracle_error <= value <= enclosure.hi+oracle_error
            assert enclosure.lo <= certificate.matrix[i][j] <= enclosure.hi


def _assert_exact_proper_rotation(matrix):
    columns = tuple(tuple(matrix[i][j] for i in range(3)) for j in range(3))
    for i in range(3):
        for j in range(3):
            assert _dot(columns[i], columns[j]) == int(i == j)
    assert _dot(columns[0], _cross(columns[1], columns[2])) == 1


@pytest.mark.parametrize("history", [
    (),
    (RotationIncrement((0., 0., 0.), .71),),
    (RotationIncrement(NEXT_OMEGA[0][0], NEXT_DT_MYR),),
    (RotationIncrement(NEXT_OMEGA[0][1], NEXT_DT_MYR),),
    (RotationIncrement((.023, -.037, .049), .71),
     RotationIncrement((-.16, .09, .13), .29),
     RotationIncrement(NEXT_OMEGA[1][0], NEXT_DT_MYR)),
])
def test_ordered_full_history_contains_independent_rodrigues_and_is_exact_proper_so3(history):
    certificate = certify_rotation_history(history, 80)
    _assert_analytic_containment(certificate, history)
    _assert_exact_proper_rotation(certificate.matrix)


def test_noncommuting_order_changes_the_certified_physical_pose():
    first = RotationIncrement((.16, -.02, .04), .71)
    second = RotationIncrement((-.03, .09, .13), .29)
    ordered = certify_rotation_history((first, second), 70)
    reversed_order = certify_rotation_history((second, first), 70)
    assert ordered.matrix != reversed_order.matrix
    _assert_analytic_containment(ordered, (first, second))
    _assert_analytic_containment(reversed_order, (second, first))
    assert any(a.lo>b.hi or b.lo>a.hi
               for row_a, row_b in zip(ordered.intervals, reversed_order.intervals)
               for a, b in zip(row_a, row_b))


@pytest.mark.parametrize("vector, duration, oracle_precision", [
    ((float(np.nextafter(0., 1.)), 0., 0.), 1., 700),
    ((1e-300, 0., 0.), 1e298, 260),
    ((1e-300, -1e-301, 1e308), 1e-310, 760),
])
def test_subnormal_or_large_components_keep_the_complete_physical_rotation(vector, duration,
        oracle_precision):
    history = (RotationIncrement(vector, duration),)
    certificate = certify_rotation_history(history, 80)
    _assert_analytic_containment(certificate, history, oracle_precision)
    _assert_exact_proper_rotation(certificate.matrix)
    # Every case has a physically nonzero component even when a binary64
    # squared norm or the smallest full increment underflows.
    assert certificate.matrix != certify_rotation_history((), 80).matrix
    if vector[2] == 0.:
        assert certificate.intervals[2][1].lo > 0
    else:
        assert certificate.intervals[2][1].lo > 0
        assert certificate.intervals[2][0].lo > 0


def test_exact_small_increment_domain_boundary_and_refusal_are_explicit():
    at_boundary = (RotationIncrement((.25, 0., 0.), 1.),)
    _assert_analytic_containment(certify_rotation_history(at_boundary, 60), at_boundary)
    outside = (RotationIncrement((float(np.nextafter(.25, 1.)), 0., 0.), 1.),)
    with pytest.raises(ValueError, match="1/4"):
        certify_rotation_history(outside, 60)
    # A finite full input whose angle overflows binary64 must not become an
    # identity or a projected-axis rotation in this declared native domain.
    with pytest.raises(ValueError, match="1/4"):
        certify_rotation_history((RotationIncrement((0., 0., 1e308), 1e308),), 60)


def test_certificates_are_independent_of_external_decimal_rounding_and_precision():
    history = (RotationIncrement(NEXT_OMEGA[0][1], NEXT_DT_MYR),
               RotationIncrement((.023, -.037, .049), .71))
    expected = certify_rotation_history(history, 70)
    with localcontext(Context(prec=7, rounding=ROUND_DOWN, Emax=20, Emin=-20, clamp=1)):
        actual = certify_rotation_history(history, 70)
    assert actual == expected


@pytest.mark.parametrize("value", [Fraction(0), Fraction(4, 9), Fraction(2),
                                   Fraction(1, 10**700), Fraction(10**700)])
def test_square_root_enclosure_has_exact_rational_squared_endpoint_proof(value):
    enclosure = interval_sqrt(value, 70)
    assert 0 <= enclosure.lo <= enclosure.hi
    assert enclosure.lo**2 <= value <= enclosure.hi**2
    if value:
        assert enclosure.lo > 0
        assert enclosure.width/enclosure.lo < Fraction(1, 10**65)


@pytest.mark.parametrize("value", [Fraction(0), Fraction(1, 10**300), Fraction(1, 8),
                                   Fraction(1), Fraction(10**300)])
def test_atan_encloses_independent_high_precision_half_angle_series(value):
    enclosure = interval_atan(value, 70)
    with localcontext(Context(prec=420)):
        argument = Decimal(value.numerator)/Decimal(value.denominator)
        multiplier = Decimal(1)
        while argument > Decimal('.125'):
            argument /= Decimal(1)+(Decimal(1)+argument*argument).sqrt()
            multiplier *= 2
        term = total = argument
        for index in range(1, 10000):
            term *= -argument*argument
            following = total+term/Decimal(2*index+1)
            if following == total:
                break
            total = following
        else:
            raise AssertionError("Independent atan oracle did not converge")
        expected = Fraction(multiplier*total)
    error = Fraction(1, 10**395)
    assert enclosure.lo-error <= expected <= enclosure.hi+error
    if value:
        assert enclosure.lo > 0
        assert enclosure.width/enclosure.lo < Fraction(1, 10**65)


def test_pi_contains_independent_gauss_legendre_value():
    with localcontext(Context(prec=260)):
        a, b, t, scale = Decimal(1), Decimal(1)/Decimal(2).sqrt(), Decimal(1)/4, Decimal(1)
        for _ in range(12):
            following_a, following_b = (a+b)/2, (a*b).sqrt()
            t -= scale*(a-following_a)**2
            a, b, scale = following_a, following_b, 2*scale
        expected = Fraction((a+b)**2/(4*t))
    enclosure = interval_pi(80)
    assert enclosure.lo < expected < enclosure.hi
    assert enclosure.width < Fraction(1, 10**75)


def test_interval_arithmetic_encloses_all_exact_corners_and_rejects_undefined_division():
    left, right = FractionInterval(Fraction(-3, 7), Fraction(2, 5)), FractionInterval(2, 3)
    for operation in (lambda a,b:a+b, lambda a,b:a-b, lambda a,b:a*b, lambda a,b:a/b):
        actual = operation(left, right)
        values = tuple(operation(a,b) for a in (left.lo,left.hi) for b in (right.lo,right.hi))
        assert actual.lo <= min(values) <= max(values) <= actual.hi
    assert (-left).lo == -left.hi and (-left).hi == -left.lo
    assert (1/right).lo == Fraction(1,3) and (1/right).hi == Fraction(1,2)
    with pytest.raises(ValueError):
        left/FractionInterval(-1, 1)
    with pytest.raises(TypeError):
        FractionInterval(0.1)


@pytest.mark.parametrize("precision", [True, 0, -1, 1.5])
def test_invalid_native_precision_cannot_select_a_pose(precision):
    with pytest.raises(ValueError):
        certify_rotation_history((), precision)
