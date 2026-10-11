"""Independent radial moments, source endpoint migration and bank slivers."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.genesis_volume_integral_geometry_bounds import (
    IntegralGeometryRefusal, prepare_integral_geometry,
)
from tectonics.genesis_volume_operand_bounds import DirectedInterval


LD = np.longdouble
F = Fraction


def _raw(*, normal=(1., 0., 0.), offset=2., center=(2., 0., 0.), radius=.5):
    return dict(section=dict(triangle_m=4*np.eye(3), displacement_A_m=np.zeros((3, 3)),
        slope_P_m=np.zeros((3, 3)), normal_xyz=normal, offset_m=offset,
        lambda_bounds=(.25, .75)), seed_center_world_m=center, seed_radius_m=radius)


def _chart(N=(1., 0., 0.), midpoint=LD(.5), halfwidth=LD(.125)):
    return SimpleNamespace(barycentric=np.array(N, dtype=LD), lambda_mid=midpoint,
                           lambda_halfwidth=halfwidth)


def _contains_exact(interval, values):
    assert len(interval.lower) == len(values)
    for lo, hi, value in zip(interval.lower, interval.upper, values):
        assert F(*lo.as_integer_ratio()) <= value <= F(*hi.as_integer_ratio())


@pytest.mark.parametrize("side_power", (0, 1, 2))
def test_exact_centered_moments_through_degree16_include_literal_Gamma(side_power):
    certificate = prepare_integral_geometry(_raw()).evaluate((1., 0., 0.), _chart())
    h = F(1, 8)
    if side_power == 0:
        expected = [2*h/(k+1) if k % 2 == 0 else F() for k in range(17)]
    elif side_power == 1:
        expected = [h/(k+1) if k % 2 else F() for k in range(17)]
    else:
        expected = [h/(2*(k+1)) if k % 2 == 0 else F() for k in range(17)]
    _contains_exact(certificate.moments(16, side_power), expected)
    assert certificate.rational_moment_enclosures[side_power] == tuple((q, q) for q in expected)
    assert certificate.source_endpoint_migration_included
    assert not certificate.angular_accuracy_certified
    assert not certificate.physical_creation_admitted
    assert not certificate.available_for_creation_funding
    assert not certificate.automated_goal_enabled


@pytest.mark.parametrize("normal,offset,sign", (((1.,0.,0.), 1., 1),
                                             ((1.,0.,0.), 3., -1),
                                             ((-1.,0.,0.), -1., -1)))
def test_constant_and_reversed_source_Gamma_banks_are_exact(normal, offset, sign):
    certificate = prepare_integral_geometry(_raw(normal=normal, offset=offset)).evaluate((1.,0.,0.), _chart())
    ordinary = [F(1,4*(k+1)) if k % 2 == 0 else F() for k in range(17)]
    _contains_exact(certificate.bank_moments(16, sign), ordinary)
    _contains_exact(certificate.bank_moments(16, -sign), [F()]*17)
    _contains_exact(certificate.moments(16, 1), [sign*q/2 for q in ordinary])


@pytest.mark.parametrize("side_sign", (-1, 1))
def test_separate_bank_antiderivatives_do_not_subtract_dependent_total_moments(side_sign):
    certificate = prepare_integral_geometry(_raw()).evaluate((1.,0.,0.), _chart())
    expected = [F(side_sign if k % 2 else 1, 8*(k+1)) for k in range(17)]
    _contains_exact(certificate.bank_moments(16, side_sign), expected)
    assert certificate.rational_bank_moment_enclosures[0 if side_sign < 0 else 1] == tuple((q,q) for q in expected)


def test_source_endpoint_migration_is_integrated_on_the_same_shifted_native_chart():
    shifted = LD(.5)+np.ldexp(LD(1), -60)
    certificate = prepare_integral_geometry(_raw()).evaluate((1.,0.,0.), _chart(midpoint=shifted))
    mid = F(*shifted.as_integer_ratio()); h = F(1,8)
    left, right = (F(3,8)-mid)/h, (F(5,8)-mid)/h
    expected = [h*(right**(k+1)-left**(k+1))/(k+1) for k in range(17)]
    _contains_exact(certificate.moments(16,0), expected)
    assert certificate.source_xi_bounds == ((left,left),(right,right))
    assert expected[1] != 0


def test_irrational_source_endpoints_use_proved_integer_root_enclosures():
    certificate = prepare_integral_geometry(_raw(center=(2.,.125,0.))).evaluate((1.,0.,0.), _chart())
    low, high = certificate.source_lambda_bounds
    center, square = F(1,2), F(15,1024)
    assert (center-low[1])**2 <= square <= (center-low[0])**2
    assert (high[0]-center)**2 <= square <= (high[1]-center)**2
    assert high[0]-low[1] > 0
    assert high[1]-high[0] < F(1,2**250)
    # The represented chart would integrate width 1/4, but the source width
    # is strictly smaller and its enclosure excludes that value.
    assert certificate.moments(0,0).upper[0] < LD(.25)


def test_positive_sub_ulp_source_Gamma_bank_and_odd_moments_are_retained():
    raw = _raw(offset=2.5)
    raw["section"]["displacement_A_m"][1,0] = 1.
    tiny = np.ldexp(LD(1), -100)
    N = np.array((LD(1),tiny,LD(0)))
    chart = _chart(N=N)
    # A tiny nonzero barycentric source contribution moves Gamma below the
    # sphere endpoint; its rational positive width is below native ulp(1).
    raw["section"]["normal_xyz"] = (1.,1.,0.)
    certificate = prepare_integral_geometry(raw).evaluate(N, chart)
    positive_width = certificate.bank_widths_xi[1]
    assert positive_width[0] > 0
    assert positive_width[1] < F(1,2**90)
    positive_bank = certificate.bank_moments(16,1)
    assert positive_bank.lower[0] > 0
    assert positive_bank.lower[1] > 0
    assert certificate.literal_gamma_root_xi < 1


def test_original_cohort_caps_inside_large_sphere_remain_literal_bounds():
    certificate = prepare_integral_geometry(_raw(radius=2.)).evaluate((1.,0.,0.), _chart(halfwidth=LD(.25)))
    assert certificate.source_lambda_bounds == ((F(1,4),F(1,4)),(F(3,4),F(3,4)))
    assert certificate.source_xi_bounds == ((F(-1),F(-1)),(F(1),F(1)))


def test_empty_source_requires_matching_native_empty_and_produces_exact_zero():
    prepared = prepare_integral_geometry(_raw(center=(10.,0.,0.)))
    certificate = prepared.evaluate((1.,0.,0.), None)
    assert certificate.empty
    assert np.array_equal(certificate.moments(16,0).lower, np.zeros(17))
    assert np.array_equal(certificate.bank_moments(16,-1).upper, np.zeros(17))
    with pytest.raises(IntegralGeometryRefusal, match="classification differs"):
        prepared.evaluate((1.,0.,0.), _chart())
    with pytest.raises(IntegralGeometryRefusal, match="classification differs"):
        prepare_integral_geometry(_raw()).evaluate((1.,0.,0.), None)


def test_exact_source_tangency_and_whole_line_Gamma_have_explicit_refusals():
    with pytest.raises(IntegralGeometryRefusal, match="tangency"):
        prepare_integral_geometry(_raw(center=(2.,.5,0.))).evaluate((1.,0.,0.), None)
    certificate = prepare_integral_geometry(_raw(normal=(0.,1.,0.),offset=0.)).evaluate((1.,0.,0.), _chart())
    assert certificate.moments(0,0).lower[0] == LD(.25)
    for side_power in (1,2):
        with pytest.raises(IntegralGeometryRefusal, match="whole source line"):
            certificate.moments(16,side_power)
    with pytest.raises(IntegralGeometryRefusal, match="whole source line"):
        certificate.bank_moments(16,1)


def test_source_preparation_is_detached_from_mutable_input_and_bound_to_literal_chart():
    raw = _raw(); original = deepcopy(raw)
    prepared = prepare_integral_geometry(raw)
    raw["section"]["triangle_m"][0,0] = 40.
    certificate = prepared.evaluate((1.,0.,0.), _chart())
    independent = prepare_integral_geometry(original).evaluate((1.,0.,0.), _chart())
    assert certificate.source_binding_sha256 == independent.source_binding_sha256
    assert prepared.matches(certificate,(1.,0.,0.),_chart())
    assert not prepared.matches(certificate,(1.,0.,0.),_chart(midpoint=LD(.5)+np.ldexp(LD(1),-60)))
    with pytest.raises(ValueError):
        certificate.moments(16,0).lower.setflags(write=True)
    with pytest.raises(TypeError):
        prepared._raw["seed_radius_m"] = F(100)


def test_resealed_inflated_moment_array_is_not_a_canonical_geometry_certificate():
    prepared = prepare_integral_geometry(_raw())
    certificate = prepared.evaluate((1.,0.,0.),_chart())
    original = certificate._moments[0]
    inflated = DirectedInterval(original.lower-1, original.upper+1)
    forged = replace(certificate, _moments=(inflated,)+certificate._moments[1:])
    assert not prepared.matches(forged,(1.,0.,0.),_chart())


@pytest.mark.parametrize("changes", ({"_moments": ()}, {"_bank_moments": ()},
                                     {"angular_accuracy_certified": True},
                                     {"physical_creation_admitted": True}))
def test_malformed_or_scope_promoted_certificate_is_rejected(changes):
    prepared = prepare_integral_geometry(_raw())
    certificate = prepared.evaluate((1.,0.,0.),_chart())
    assert not prepared.matches(replace(certificate,**changes),(1.,0.,0.),_chart())


@pytest.mark.parametrize("degree,side_power", ((-1,0),(17,0),(True,0),(1,-1),(1,3),(1,True)))
def test_invalid_moment_requests_are_rejected(degree,side_power):
    certificate = prepare_integral_geometry(_raw()).evaluate((1.,0.,0.),_chart())
    with pytest.raises(IntegralGeometryRefusal):
        certificate.moments(degree,side_power)


def test_source_native_N_mismatch_is_rejected():
    with pytest.raises(IntegralGeometryRefusal,match="N differ"):
        prepare_integral_geometry(_raw()).evaluate((1.,0.,0.),_chart(N=(.5,.5,0.)))
