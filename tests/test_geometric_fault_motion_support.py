"""Continuous material-bank ownership and analytic finite-pair geometry."""
from dataclasses import replace
from itertools import product
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.test_geometric_distributed_contact_invariants import traces, world
from tectonics.fractional_surface import SurfaceParcel
from tectonics.geometric_bending import ElasticBendingBankParameters, FaultBendingParameters
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_fault_motion_support import (
    UnresolvedFaultMotionSupport, _fragment_interval, continuous_bank_pair_separation_m,
    validate_reference_bending_bank_support,
)
from tectonics.geometric_fault_traces import (
    TraceQuadrature, UnresolvedFaultTrace, _cover_once, _interval,
    declare_fault_trace_set, fault_trace_points,
)
from tectonics.geometric_surface import GeometricFragment, GeometricSurfaceState, split_geometry
from tectonics.spherical_polygons import (
    arc_length, clip_hemisphere, normalize_polygon, polygon_area, polygon_halfspaces,
)


def banks(trace_set, width=10000.):
    bank = ElasticBendingBankParameters(width, width/10., 7e10, .25,
                                       "Explicit synthetic reference clamp")
    return tuple(FaultBendingParameters(trace.trace_id, bank, bank)
                 for trace in trace_set.traces)


def setup(*, order=4):
    _, surface, _ = world()
    trace_set = declare_fault_trace_set(surface, traces(), TraceQuadrature(1, order))
    return surface, trace_set, banks(trace_set)


def test_reference_clamps_belong_to_declared_banks_over_complete_inward_segments():
    surface, trace_set, parameters = setup()
    checked = validate_reference_bending_bank_support(surface, trace_set, parameters)
    assert len(checked.supports) == 2*len(fault_trace_points(surface, trace_set))
    assert checked.roundoff_merged_gap_count == 0
    assert checked.maximum_roundoff_gap_radians == 0.
    points = {point.point_id: point for point in fault_trace_points(surface, trace_set)}
    for item in checked.supports:
        point = points[item.point_id]
        angle = item.width_m/(surface.radius_km*1000.)
        sign = -1. if item.bank == "sub" else 1.
        # Independent equatorial analytic meridian, including ownership sign.
        expected = np.asarray(point.position)*math.cos(angle)
        expected += sign*np.asarray(point.horizontal)*math.sin(angle)
        np.testing.assert_allclose(item.clamp_position, expected, rtol=0., atol=3e-16)
        assert item.clamp_position[2] > 0. if item.bank == "sub" else item.clamp_position[2] < 0.
        assert all(high > low >= 0. for _, low, high in item.fragment_intervals)
        assert max(high for _, _, high in item.fragment_intervals) == angle


@pytest.mark.parametrize("angle", [math.pi/2., math.pi/2.+1e-9, math.pi, 2.*math.pi])
def test_reference_width_must_be_a_declared_minor_local_bank_segment(angle):
    surface, trace_set, _ = setup()
    with pytest.raises(UnresolvedFaultMotionSupport, match="width/radius"):
        validate_reference_bending_bank_support(surface, trace_set,
            banks(trace_set, angle*surface.radius_km*1000.))


def test_same_owner_material_cuts_do_not_change_physical_clamp_positions():
    surface, trace_set, parameters = setup()
    original = validate_reference_bending_bank_support(surface, trace_set, parameters)
    point = fault_trace_points(surface, trace_set)[0]
    target = next(item for item in surface.fragments if item.parcel.plate == 0
                  and np.all(np.asarray(item.polygon)@np.asarray(point.position) >= 0.))
    polygon = np.asarray(target.polygon)
    # The centroid fan crosses inward bank geodesics while retaining all mass.
    centroid = np.sum(polygon, axis=0)
    centroid /= np.linalg.norm(centroid)
    pieces = tuple(normalize_polygon((centroid, a, b))
                   for a, b in zip(polygon, np.roll(polygon, -1, axis=0)))
    children = split_geometry(target, pieces, label="reference-bank-cut")
    changed = replace(surface, fragments=tuple(item for item in surface.fragments
        if item.fragment_id != target.fragment_id)+children)
    rebound = declare_fault_trace_set(changed, trace_set.traces, trace_set.quadrature)
    checked = validate_reference_bending_bank_support(changed, rebound, parameters)
    assert checked.trace_sha256 == original.trace_sha256
    assert checked.surface_sha256 != original.surface_sha256
    assert tuple((item.point_id, item.bank, item.clamp_position) for item in checked.supports) == tuple(
        (item.point_id, item.bank, item.clamp_position) for item in original.supports)
    assert any(any(identity.startswith(target.fragment_id+"/") for identity, _, _ in item.fragment_intervals)
               for item in checked.supports)


def _oblique_material_world():
    # A nonsingular radial image of the octahedron is an exact partition.
    # The northern x-bank boundary x=z*tan(beta) crosses +z meridians.
    beta = .8
    basis = np.asarray(((1., 0., 0.), (0., 1., 0.), (math.sin(beta), 0., math.cos(beta))))
    fragments = []
    radius = 100.
    for index, signs in enumerate(product((-1., 1.), repeat=3)):
        polygon = normalize_polygon(np.asarray(signs)[:, None]*basis)
        area = polygon_area(polygon)*radius**2
        owner = 1 if signs[2] < 0. else (0 if signs[0] > 0. else 2)
        parcel = SurfaceParcel(index, owner, f"origin:{index}", area, area, area, area, 0.)
        fragments.append(GeometricFragment(f"oblique:{index}", tuple(map(tuple, polygon)), parcel))
    surface = GeometricSurfaceState(0., radius, tuple(fragments))
    template = traces()[0]
    physical = []
    for index, contact in enumerate(extract_contacts(surface.fragments, surface.radius_km)):
        # Contact start/end and its normal follow actual a->b orientation.
        a, b = contact.start, contact.end
        if np.dot(np.cross(a, b), contact.normal_a_to_b) < 0.:
            a, b = b, a
        physical.append(replace(template, trace_id=f"oblique-physical:{index}", start=a, end=b,
            subducting_plate=contact.plate_a, overriding_plate=contact.plate_b))
    return surface, declare_fault_trace_set(surface, physical, TraceQuadrature(1, 4))


def test_wrong_owner_interior_is_rejected_even_when_trench_support_is_valid():
    surface, trace_set = _oblique_material_world()
    validate_reference_bending_bank_support(surface, trace_set, banks(trace_set, 100.))
    with pytest.raises(UnresolvedFaultMotionSupport, match="another material owner"):
        validate_reference_bending_bank_support(surface, trace_set, banks(trace_set, 140000.))


@pytest.mark.parametrize("sliver_angle", [.03, 1e-10, 1e-12])
def test_foreign_interior_strip_is_retained_when_both_bank_endpoints_have_correct_owner(sliver_angle):
    _, original, _ = world()
    target = next(item for item in original.fragments if item.parcel.plate == 0
                  and np.any(np.all(np.asarray(item.polygon) == (1., 0., 0.), axis=1))
                  and np.any(np.all(np.asarray(item.polygon) == (0., -1., 0.), axis=1)))
    first = np.asarray((-math.tan(.03), 0., 1.))
    last = np.asarray((-math.tan(.03+sliver_angle), 0., 1.))
    strip = clip_hemisphere(clip_hemisphere(target.polygon, first), -last)
    pieces = (clip_hemisphere(target.polygon, -first), strip,
              clip_hemisphere(target.polygon, last))
    children = list(split_geometry(target, pieces, label="foreign-interior-strip"))
    children[1] = replace(children[1], parcel=replace(children[1].parcel, plate=2))
    surface = replace(original, fragments=tuple(item for item in original.fragments
        if item.fragment_id != target.fragment_id)+tuple(children))
    assert polygon_area(strip) > 0.
    template = traces()[0]
    physical = []
    for index, contact in enumerate(extract_contacts(surface.fragments, surface.radius_km)):
        start, end = contact.start, contact.end
        if np.dot(np.cross(start, end), contact.normal_a_to_b) < 0.:
            start, end = end, start
        midpoint = np.asarray(start)+np.asarray(end)
        is_target = (set((contact.plate_a, contact.plate_b)) == {0, 1}
                     and midpoint[0] > 0. and midpoint[1] < 0.)
        physical.append(replace(template, trace_id="aaa-target" if is_target else f"zzz-other:{index}",
            start=start, end=end, subducting_plate=contact.plate_a, overriding_plate=contact.plate_b))
    trace_set = declare_fault_trace_set(surface, physical, TraceQuadrature(1, 4))
    tiny = ElasticBendingBankParameters(sliver_angle*100., sliver_angle*10., 7e10, .25,
                                       "Explicit support narrower than the foreign strip")
    wide = ElasticBendingBankParameters(10000., 1000., 7e10, .25,
                                       "Clamp deliberately crosses and exits the foreign strip")
    parameters = tuple(FaultBendingParameters(trace.trace_id,
        wide if trace.trace_id == "aaa-target" and trace.subducting_plate == 0 else tiny,
        wide if trace.trace_id == "aaa-target" and trace.overriding_plate == 0 else tiny)
        for trace in trace_set.traces)
    trace = trace_set.traces[0]
    point = fault_trace_points(surface, trace_set)[0]
    sign = -1. if trace.subducting_plate == 0 else 1.
    start = np.asarray(point.position)
    clamp = start*math.cos(.1)+sign*np.asarray(point.horizontal)*math.sin(.1)
    assert np.min(polygon_halfspaces(children[0].polygon)@start) >= -1e-15
    assert np.min(polygon_halfspaces(children[2].polygon)@clamp) >= -1e-15
    with pytest.raises(UnresolvedFaultMotionSupport, match="another material owner.*aaa-target"):
        validate_reference_bending_bank_support(surface, trace_set, parameters)


@pytest.mark.parametrize("sliver_angle", [.03, 1e-10, 1e-12])
def test_spherical_interval_clipping_retains_tiny_positive_foreign_strips(sliver_angle):
    surface, trace_set, _ = setup()
    point = fault_trace_points(surface, trace_set)[0]
    origin, direction = np.asarray(point.position), -np.asarray(point.horizontal)
    target = next(item for item in surface.fragments if item.parcel.plate == 0
                  and np.any(np.all(np.asarray(item.polygon) == (1., 0., 0.), axis=1))
                  and np.any(np.all(np.asarray(item.polygon) == (0., -1., 0.), axis=1)))
    lower, upper = .03, .03+sliver_angle
    strip = clip_hemisphere(clip_hemisphere(target.polygon,
        (-math.tan(lower), 0., 1.)), (math.tan(upper), 0., -1.))
    assert polygon_area(strip) > 0.
    result = _fragment_interval(origin, direction, .1, strip)
    assert result is not None and result[1] > result[0]
    # Independently solve z(theta)=x(theta)*tan(alpha) on the meridian.
    expected = tuple(math.atan(origin[0]*math.tan(angle)) for angle in (lower, upper))
    np.testing.assert_allclose(result, expected, rtol=0., atol=5e-17)


@pytest.mark.parametrize("extent", [1e-10, 1e-12, 1e-20])
def test_saved_short_trace_endpoints_have_exact_interval_incidence(extent):
    start = (math.cos(.03), 0., math.sin(.03)) if extent >= 1e-12 else (1., 0., 0.)
    tangent = (-math.sin(.03), 0., math.cos(.03)) if extent >= 1e-12 else (0., 1., 0.)
    end = tuple(np.asarray(start)*math.cos(extent)+np.asarray(tangent)*math.sin(extent))
    measured = arc_length(start, end)
    assert measured > 0.
    assert _interval(start, end, start, end) == (0., measured)
    assert _interval(start, end, end, start) == (0., measured)


def test_nonendpoint_trace_coordinate_uses_stable_difference_form():
    start = np.asarray((1., 1., 0.))/math.sqrt(2.)
    tangent = np.asarray((-1., 1., 0.))/math.sqrt(2.)
    end = start*math.cos(.83)+tangent*math.sin(.83)
    point = start*math.cos(1e-12)+tangent*math.sin(1e-12)
    expected = arc_length(start, point)
    low, high = _interval(start, end, point, point)
    assert low == high == pytest.approx(expected, rel=8.*np.finfo(float).eps, abs=0.)


@pytest.mark.parametrize("extent", [1e-10, 1e-12, 1e-20])
@pytest.mark.parametrize("kind", ["gap", "overlap"])
def test_tiny_arc_coverage_keeps_relative_positive_gap_and_overlap_detection(extent, kind):
    perturbation = extent*1e-12
    second_start = extent/2.+(perturbation if kind == "gap" else -perturbation)
    with pytest.raises(UnresolvedFaultTrace, match="Missing" if kind == "gap" else "Multiple"):
        _cover_once(((0., extent/2.), (second_start, extent)), extent, "explicit tiny arc")


def test_reference_support_covariance_under_global_passive_rotation():
    surface, trace_set, parameters = setup()
    original = validate_reference_bending_bank_support(surface, trace_set, parameters)
    matrix = Rotation.from_rotvec((.31, -.17, .23)).as_matrix()
    changed = replace(surface, fragments=tuple(replace(item,
        polygon=tuple(map(tuple, np.asarray(item.polygon)@matrix.T))) for item in surface.fragments))
    moved = tuple(replace(item, start=tuple(matrix@item.start), end=tuple(matrix@item.end))
                  for item in trace_set.traces)
    rebound = declare_fault_trace_set(changed, moved, trace_set.quadrature)
    checked = validate_reference_bending_bank_support(changed, rebound, parameters)
    for before, after in zip(original.supports, checked.supports):
        assert before.point_id == after.point_id and before.plate_id == after.plate_id
        np.testing.assert_allclose(after.clamp_position, matrix@before.clamp_position, rtol=0., atol=8e-16)


def test_reference_support_has_no_dense_owner_id_allocation():
    surface, trace_set, parameters = setup()
    ids = {0: 3, 1: 10**12+7}
    changed = replace(surface, fragments=tuple(replace(item,
        parcel=replace(item.parcel, plate=ids[item.parcel.plate])) for item in surface.fragments))
    moved = tuple(replace(item, subducting_plate=ids[item.subducting_plate],
                          overriding_plate=ids[item.overriding_plate]) for item in trace_set.traces)
    rebound = declare_fault_trace_set(changed, moved, trace_set.quadrature)
    checked = validate_reference_bending_bank_support(changed, rebound, parameters)
    assert {item.plate_id for item in checked.supports} == set(ids.values())


def test_arbitrarily_short_positive_reference_bank_is_retained():
    surface, trace_set, _ = setup()
    width = 1e-12
    checked = validate_reference_bending_bank_support(surface, trace_set, banks(trace_set, width))
    assert all(item.width_m == width and item.fragment_intervals for item in checked.supports)
    assert all(max(high for _, _, high in item.fragment_intervals) > 0. for item in checked.supports)


@pytest.mark.parametrize("case", ["missing", "duplicate", "unknown", "wrong_type"])
def test_reference_parameters_must_cover_each_physical_trace_once(case):
    surface, trace_set, parameters = setup()
    changed = {"missing": parameters[:-1], "duplicate": parameters+(parameters[0],),
               "unknown": (replace(parameters[0], trace_id="unknown"),)+parameters[1:],
               "wrong_type": (object(),)+parameters[1:]}[case]
    with pytest.raises(ValueError, match="parameters|Parameters|every physical trace"):
        validate_reference_bending_bank_support(surface, trace_set, changed)


def test_identical_finite_maps_preserve_each_exact_material_pair():
    matrix = Rotation.from_rotvec((.7, -.2, .4)).as_matrix()
    assert continuous_bank_pair_separation_m(traces()[0], matrix, matrix, 1e6) == (0., 0.)


def test_along_strike_sliding_changes_pairing_despite_equal_boundary_sets():
    angle, radius = .13, 1e6
    matrix = Rotation.from_rotvec((0., 0., angle)).as_matrix()
    expected = 2.*radius*math.sin(angle/2.)
    for trace in traces():
        low, high = continuous_bank_pair_separation_m(trace, np.eye(3), matrix, radius)
        assert low == pytest.approx(expected, rel=8e-16)
        assert high == pytest.approx(expected, rel=8e-16)
    # Both bank images still describe exactly the same complete equator.
    assert np.array_equal(matrix@np.asarray((0., 0., 1.)), (0., 0., 1.))


def test_continuous_pair_audit_finds_an_extremum_between_gauss_nodes():
    angle, radius = .4, 1e6
    axis = np.asarray((1., 1., 0.))/math.sqrt(2.)
    matrix = Rotation.from_rotvec(angle*axis).as_matrix()
    trace = traces()[0]
    low, high = continuous_bank_pair_separation_m(trace, np.eye(3), matrix, radius)
    expected = 2.*radius*math.sin(angle/2.)
    assert low == pytest.approx(expected/math.sqrt(2.), rel=1e-15)
    assert high == pytest.approx(expected, rel=1e-15)
    surface, trace_set, _ = setup(order=2)
    nodes = [point for point in fault_trace_points(surface, trace_set) if point.trace_id == trace.trace_id]
    sampled = max(radius*np.linalg.norm(np.asarray(point.position)-matrix@point.position) for point in nodes)
    assert sampled < high*.91


def test_continuous_pair_separation_is_covariant_and_scales_with_radius():
    trace = traces()[0]
    sub = Rotation.from_rotvec((.23, -.1, .7)).as_matrix()
    over = Rotation.from_rotvec((-.11, .4, .1)).as_matrix()
    frame = Rotation.from_rotvec((-.2, .9, -.31)).as_matrix()
    moved = replace(trace, start=tuple(frame@trace.start), end=tuple(frame@trace.end))
    before = continuous_bank_pair_separation_m(trace, sub, over, 1e6)
    after = continuous_bank_pair_separation_m(moved, frame@sub@frame.T, frame@over@frame.T, 2e6)
    np.testing.assert_allclose(after, 2.*np.asarray(before), rtol=3e-15, atol=0.)


@pytest.mark.parametrize("matrix", [np.diag((1., 1., -1.)), 2.*np.eye(3),
    np.zeros((3, 3)), np.eye(2), np.full((3, 3), np.nan), np.eye(3, dtype=bool)])
def test_continuous_pair_rejects_non_so3_maps(matrix):
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        continuous_bank_pair_separation_m(traces()[0], matrix, np.eye(3), 1e6)


@pytest.mark.parametrize("radius", [0., -1., math.inf, math.nan, True])
def test_continuous_pair_requires_an_explicit_positive_radius(radius):
    with pytest.raises(ValueError, match="Sphere radius"):
        continuous_bank_pair_separation_m(traces()[0], np.eye(3), np.eye(3), radius)


@pytest.mark.parametrize("steps", [-1, True, np.bool_(False), .5, 1., math.inf, math.nan, "60", None])
def test_continuous_pair_requires_explicit_nonnegative_integer_roundoff_steps(steps):
    with pytest.raises(ValueError, match="roundoff steps"):
        continuous_bank_pair_separation_m(traces()[0], np.eye(3), np.eye(3), 1e6,
                                          rotation_roundoff_steps=steps)


def test_long_composition_uses_declared_arithmetic_budget_without_matrix_projection():
    increment = Rotation.from_rotvec((.2, -.1, .3)).as_matrix()
    matrix = np.eye(3)
    for _ in range(60):
        matrix = increment@matrix
    assert abs(float(np.linalg.det(matrix))-1.) > 64.*np.finfo(float).eps
    saved = matrix.copy()
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        continuous_bank_pair_separation_m(traces()[0], matrix, matrix, 1e6)
    assert continuous_bank_pair_separation_m(traces()[0], matrix, matrix, 1e6,
        rotation_roundoff_steps=np.int64(60)) == (0., 0.)
    np.testing.assert_array_equal(matrix, saved)


@pytest.mark.parametrize("matrix", [np.diag((1., 1., -1.)), 2.*np.eye(3),
                                   np.zeros((3, 3)), np.diag((1.000001, 1., 1.))])
def test_long_composition_budget_rejects_nonrotational_maps(matrix):
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        continuous_bank_pair_separation_m(traces()[0], matrix, np.eye(3), 1e6,
                                          rotation_roundoff_steps=60)
