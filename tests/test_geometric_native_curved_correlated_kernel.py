"""Correlated curved normals against an independent Decimal world oracle.

Reference depth labels share one physical radius; this cancellation must
survive literal owner rotations, current roof displacements and sub-float
compression. Finite differences differentiate complete world energy.
"""
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import json

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tests.test_geometric_native_curved_contact_energy import (
    _d, _dot, _oracle_geometry, _trig,
)


def _api():
    from tectonics import geometric_native_curved_correlated_kernel as module
    return module


def _source(**changes):
    arguments = dict(radius_m=F(100), donor_radial_before_m=F(0),
        receiver_radial_before_m=F(-1, 10), receiver_depth_bounds_m=I(F(6, 5)),
        angle_bounds_radians=I(F(1, 7)),
        start_direction_bounds=(I(F(3, 5)), I(F(4, 5)), I(0)),
        tangent_direction_bounds=(I(F(-4, 5)), I(F(3, 5)), I(0)),
        normal_direction_bounds=(I(0), I(0), I(1)),
        shared_depth_axis_per_m=(F(1, 1000), F(-1, 900), F(1, 800)),
        normal_orientation=1, source_preload_gap_m=F(-2, 5),
        normal_compliances_m_pa=(F(1, 1000000), F(1, 1000000), F(1, 1000000)),
        precision=160, order=40)
    arguments.update(changes)
    return _api().prepare_curved_correlated_reference(**arguments)


def _motion(**changes):
    arguments = dict(donor_radial_increment_m=F(1, 8),
        receiver_radial_increment_m=F(-1, 5),
        donor_owner_axis=(F(1, 100), F(1, 110), F(-1, 90)),
        receiver_owner_axis=(F(-1, 101), F(1, 120), F(1, 98)),
        time_fraction=F(1))
    arguments.update(changes)
    return arguments


def _response(source, motion=None, **changes):
    arguments = _motion() if motion is None else dict(motion)
    arguments.update(changes)
    return _api().bound_curved_correlated_cell(source,
        angle_bounds_radians=source.angle_bounds_radians,
        receiver_depth_bounds_m=source.receiver_depth_bounds_m, **arguments)


def _oracle(source, motion):
    """Use independent Rodrigues transforms and world cross products."""
    theta = _d(source.angle_bounds_radians.lo)
    start = tuple(_d(value.lo) for value in source.start_direction_bounds)
    tangent = tuple(_d(value.lo) for value in source.tangent_direction_bounds)
    cosine, sine = _trig(theta, True), _trig(theta)
    direction = tuple(cosine*u+sine*t for u, t in zip(start, tangent))
    angle_tangent = tuple(-sine*u+cosine*t for u, t in zip(start, tangent))
    radius, receiver_before, donor_before = (source.radius_m,
        source.receiver_radial_before_m, source.donor_radial_before_m)
    offset = ((radius+donor_before)**3-(radius+receiver_before)**3)/(3*radius**2)
    common = dict(radius=radius, direction=direction, tangent=angle_tangent,
        rate=source.shared_depth_axis_per_m)
    time = motion["time_fraction"]
    donor = _oracle_geometry(**common, roof=donor_before+time*motion["donor_radial_increment_m"],
        reference_roof=donor_before, depth=source.receiver_depth_bounds_m.lo+offset,
        owner=tuple(time*axis for axis in motion["donor_owner_axis"]))
    receiver = _oracle_geometry(**common,
        roof=receiver_before+time*motion["receiver_radial_increment_m"],
        reference_roof=receiver_before, depth=source.receiver_depth_bounds_m.lo,
        owner=tuple(time*axis for axis in motion["receiver_owner_axis"]))
    relative = tuple(b-a for a, b in zip(donor[0], receiver[0]))
    gap = _d(source.source_preload_gap_m)+source.normal_orientation*_dot(receiver[3], relative)
    compliance = sum(map(_d, source.normal_compliances_m_pa), Decimal(0))
    compression = min(gap, Decimal(0))
    potential = receiver[2]*compression**2/(2*compliance)
    return gap, receiver[2], potential


def _contains(bounds, expected):
    value = F(expected)
    if bounds.lo == bounds.hi:
        assert abs(bounds.lo-value) < F(1, 10**170)*max(F(1), abs(value))
    else:
        assert bounds.lo <= value <= bounds.hi


def test_correlated_gap_area_and_energy_enclose_independent_world_geometry():
    source, motion = _source(), _motion()
    response = _response(source, motion)
    with localcontext() as context:
        context.prec = 360
        gap, area, energy = _oracle(source, motion)
        _contains(response.free_gap_bounds_m, gap)
        _contains(response.area_density_bounds_m2_per_radian_per_material_m, area)
        _contains(response.potential_density_bounds_j_per_radian_per_material_m, energy)


@pytest.mark.parametrize("coordinate", range(8))
def test_all_literal_axis_and_roof_variations_include_normal_and_area_changes(coordinate):
    source, motion = _source(), _motion(time_fraction=F(3, 4))
    response = _response(source, motion)
    lower, upper = dict(motion), dict(motion)
    increment = F(1, 10**100)
    if coordinate < 6:
        bank, axis = divmod(coordinate, 3)
        key = ("donor_owner_axis", "receiver_owner_axis")[bank]
        a, b = list(lower[key]), list(upper[key])
        a[axis] -= increment
        b[axis] += increment
        lower[key], upper[key] = tuple(a), tuple(b)
    else:
        key = ("donor_radial_increment_m", "receiver_radial_increment_m")[coordinate-6]
        lower[key] -= increment
        upper[key] += increment
    with localcontext() as context:
        context.prec = 440
        a, b = _oracle(source, lower), _oracle(source, upper)
        denominator = 2*_d(increment)
        _contains(response.gap_gradient_bounds[coordinate], (b[0]-a[0])/denominator)
        _contains(response.area_gradient_bounds[coordinate], (b[1]-a[1])/denominator)
        _contains(response.potential_gradient_bounds[coordinate], (b[2]-a[2])/denominator)


@pytest.mark.parametrize("axis", [(F(0),)*3, (F(1, 90), F(-1, 120), F(1, 85))])
def test_same_owner_equal_radial_change_retains_exact_activation_gap(axis):
    source = _source(donor_radial_before_m=F(0), receiver_radial_before_m=F(0),
        source_preload_gap_m=F(0))
    motion = _motion(donor_radial_increment_m=F(1, 8), receiver_radial_increment_m=F(1, 8),
        donor_owner_axis=axis, receiver_owner_axis=axis)
    response = _response(source, motion)
    assert response.free_gap_bounds_m == I(source.source_preload_gap_m)


def test_depth_shear_conjugation_changes_opposite_owner_axis_gap():
    source = _source()
    plain = _source(shared_depth_axis_per_m=(F(0),)*3)
    motion = _motion()
    curved_response, plain_response = _response(source, motion), _response(plain, motion)
    assert (curved_response.free_gap_bounds_m.hi < plain_response.free_gap_bounds_m.lo
        or plain_response.free_gap_bounds_m.hi < curved_response.free_gap_bounds_m.lo)
    with localcontext() as context:
        context.prec = 360
        _contains(curved_response.free_gap_bounds_m, _oracle(source, motion)[0])


def test_full_current_area_derivative_is_a_real_energy_gradient_term():
    source = _source()
    response = _response(source, _motion(donor_owner_axis=(F(0),)*3,
        receiver_owner_axis=(F(0),)*3))
    pressure_only = (-response.area_density_bounds_m2_per_radian_per_material_m
        *response.normal_pressure_bounds_pa*response.gap_gradient_bounds[7])
    full = response.potential_gradient_bounds[7]
    assert full.hi < pressure_only.lo or pressure_only.hi < full.lo


def test_sub_float_compression_is_retained_after_exact_shared_owner_cancellation():
    source = _source(donor_radial_before_m=F(0), receiver_radial_before_m=F(0),
        source_preload_gap_m=-F(1, 10**600))
    response = _response(source, _motion(donor_radial_increment_m=F(0),
        receiver_radial_increment_m=F(0), donor_owner_axis=(F(0),)*3,
        receiver_owner_axis=(F(0),)*3))
    assert response.free_gap_bounds_m == I(-F(1, 10**600))
    assert response.normal_pressure_bounds_pa.lo > 0
    assert response.potential_density_bounds_j_per_radian_per_material_m.lo > 0
    assert float(response.normal_pressure_bounds_pa.hi) == 0.


def test_reference_receipt_is_replayed_instead_of_trusting_forged_saved_offset():
    source = _source()
    forged = replace(source, donor_depth_offset_m=source.donor_depth_offset_m+1)
    with pytest.raises(ValueError, match="reference|replay|offset|source"):
        _response(forged)


def test_a_whole_angle_depth_cell_contains_independent_boundary_world_positions():
    source = _source(angle_bounds_radians=I(0, F(1, 7)),
        receiver_depth_bounds_m=I(F(6, 5), F(7, 5)))
    motion, response = _motion(), _response(source)
    with localcontext() as context:
        context.prec = 360
        for theta in (F(0), F(1, 14), F(1, 7)):
            for depth in (F(6, 5), F(7, 5)):
                point_source = replace(source, angle_bounds_radians=I(theta),
                    receiver_depth_bounds_m=I(depth))
                gap, area, energy = _oracle(point_source, motion)
                _contains(response.free_gap_bounds_m, gap)
                _contains(response.area_density_bounds_m2_per_radian_per_material_m, area)
                _contains(response.potential_density_bounds_j_per_radian_per_material_m, energy)


def test_gauss_newton_curvature_is_distinct_from_complete_energy_hessian():
    source, motion = _source(), _motion()
    response = _response(source, motion)
    step = F(1, 10**50)
    lower, upper = dict(motion), dict(motion)
    lower["receiver_radial_increment_m"] -= step
    upper["receiver_radial_increment_m"] += step
    with localcontext() as context:
        context.prec = 440
        before, middle, after = (_oracle(source, point)[2] for point in (lower, motion, upper))
        true_hessian = F((after-2*middle+before)/_d(step)**2)
    approximation = response.gauss_newton_hessian[7][7]
    assert approximation.lo > 0
    assert true_hessian < approximation.lo or approximation.hi < true_hessian
    assert response.complete_three_dimensional_strain_energy_certified is False
    assert response.physical_material_admission_authorized is False


def test_self_consistently_resealed_pure_receipt_cannot_claim_native_source_authority():
    source = replace(_source(), native_source_authenticated=True)
    def payload(value):
        if type(value) is F:
            return [value.numerator, value.denominator]
        if type(value) is I:
            return {"lo": payload(value.lo), "hi": payload(value.hi)}
        if type(value) is tuple:
            return [payload(item) for item in value]
        return value
    values = tuple(getattr(source, name) for name in source.__dataclass_fields__
        if name != "source_snapshot_sha256")
    digest = hashlib.sha256(json.dumps(payload(values), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    forged = replace(source, source_snapshot_sha256=digest)
    with pytest.raises(ValueError, match="native|authentication|re-sealed"):
        _api().validate_curved_correlated_source(forged)
