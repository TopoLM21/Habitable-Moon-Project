"""Complete original-material point histories outside a preselected thin patch."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import pytest

from tectonics.geometric_intrinsic_geometry import RotationIncrement
from tectonics.geometric_native_polygons import (
    NativePlane, NativePolygon, NativeRay, _side, native_polygon_from_dict, validate_native_polygon,
)


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "analysis/geometric_native_underthrust_validation/force_control_release_02"


def _encoded(raw):
    return json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _cross(a, b):
    return tuple(a[j]*b[k]-a[k]*b[j] for j, k in ((1, 2), (2, 0), (0, 1)))


def _axes(donor, receiver, dt):
    duration = Fraction.from_float(dt)
    return tuple(Fraction.from_float(v)*duration for v in donor), tuple(
        Fraction.from_float(v)*duration for v in receiver)


def _first_two_normal_coefficients(normal, donor, receiver, dt):
    d, r = _axes(donor, receiver, dt)
    first = _cross(tuple(a-b for a, b in zip(r, d)), normal)
    dd, rr, dr = _cross(d, _cross(d, normal)), _cross(r, _cross(r, normal)), _cross(d, _cross(r, normal))
    second = tuple((a+b)/2-c for a, b, c in zip(dd, rr, dr))
    return first, second


def _poly_product(first, second):
    result = [Fraction(0)]*(len(first)+len(second)-1)
    for i, a in enumerate(first):
        for j, b in enumerate(second):
            result[i+j] += a*b
    return tuple(result)


def _expected_characteristic_polynomial(donor, receiver, dt):
    d, r = _axes(donor, receiver, dt)
    a, b = sum(v*v for v in d), sum(v*v for v in r)
    result = (Fraction(0), Fraction(1))
    for factor in ((a, 0, 1), (b, 0, 1), ((a-b)**2, 0, 2*(a+b), 0, 1)):
        result = _poly_product(result, factor)
    return result


def _matrix_product(first, second):
    return tuple(tuple(sum((first[i][k]*second[k][j] for k in range(9)), Fraction(0))
                       for j in range(9)) for i in range(9))


def _cross_matrix(axis):
    x, y, z = axis
    return ((0, -z, y), (z, 0, -x), (-y, x, 0))


def _kronecker_generator(donor, receiver, dt):
    d, r = (_cross_matrix(axis) for axis in _axes(donor, receiver, dt))
    indices = [(i, j) for i in range(3) for j in range(3)]
    return tuple(tuple((d[i][k] if j == l else 0)+(r[j][l] if i == k else 0)
                       for k, l in indices) for i, j in indices)


@pytest.fixture(scope="module")
def actual():
    probe_bytes = (CONTROL / "force_candidate_refusal_probe.json").read_bytes()
    initial_bytes = (CONTROL / "initial.json").read_bytes()
    probe, initial = json.loads(probe_bytes), json.loads(initial_bytes)
    source = {fragment["fragment_id"]: fragment
              for fragment in initial["current_surface"]["fragments"]}
    donor, receiver = tuple(probe["omega"]["1"]), tuple(probe["omega"]["0"])
    return dict(donor=native_polygon_from_dict(source["octant:6"]["geometry"]),
        receiver=native_polygon_from_dict(source["octant:2"]["geometry"]),
        donor_omega=donor, receiver_omega=receiver, dt=probe["admission"]["dt_myr"],
        probe=probe, probe_bytes=probe_bytes, initial_bytes=initial_bytes)


def _certify(actual, point, **budget):
    from tectonics.geometric_native_point_chronology import certify_native_point_chronology
    return certify_native_point_chronology(actual["donor"], actual["receiver"],
        actual["donor_omega"], actual["receiver_omega"], actual["dt"], point, **budget)


@pytest.fixture(scope="module")
def ordinary(actual):
    first, _ = _first_two_normal_coefficients((-1, 0, 0), actual["donor_omega"],
                                              actual["receiver_omega"], actual["dt"])
    return _certify(actual, (first[1]/2, Fraction(1), Fraction(-1)))


def _occupancy_samples(certificate):
    from tectonics.geometric_native_point_chronology import evaluate_native_point_chronology
    events = certificate.occupancy_events
    fractions = [events[0].time_fraction.lo/2] if events else [Fraction(1, 2)]
    fractions += [(a.time_fraction.hi+b.time_fraction.lo)/2 for a, b in zip(events, events[1:])]
    if events:
        fractions.append((events[-1].time_fraction.hi+1)/2)
    return [evaluate_native_point_chronology(certificate, u).occupancy for u in fractions]


def test_actual_ordinary_material_has_one_entry_around_half_the_full_interval(actual, ordinary):
    assert ordinary.donor_omega == actual["donor_omega"]
    assert ordinary.receiver_omega == actual["receiver_omega"]
    assert all(value != 0 for omega in (ordinary.donor_omega, ordinary.receiver_omega) for value in omega)
    assert ordinary.dt_myr == actual["dt"]
    assert [event.kind for event in ordinary.occupancy_events] == ["entry"]
    event, = ordinary.occupancy_events
    assert Fraction(49, 100) < event.time_fraction.lo < event.time_fraction.hi < Fraction(51, 100)
    assert event.time_myr == event.time_fraction*Fraction(actual["dt"])
    assert _occupancy_samples(ordinary) == ["outside", "inside"]
    assert actual["probe"]["accepted_loaded_native_intervals"] == 0
    assert actual["probe"]["input_unchanged"] is True
    assert (CONTROL / "force_candidate_refusal_probe.json").read_bytes() == actual["probe_bytes"]
    assert (CONTROL / "initial.json").read_bytes() == actual["initial_bytes"]


def test_far_original_donor_material_remains_outside_without_a_fabricated_entry(actual):
    certificate = _certify(actual, (1, 1, -1))
    assert certificate.initial_occupancy == "outside"
    assert certificate.occupancy_events == ()
    assert _occupancy_samples(certificate) == ["outside"]


def test_receiver_only_helper_does_not_fabricate_original_donor_membership(actual):
    from tectonics.geometric_native_point_chronology import (
        certify_native_point_receiver_chronology, evaluate_native_point_chronology,
    )
    point = (Fraction(-1), Fraction(1), Fraction(-1))
    with pytest.raises(ValueError):
        _certify(actual, point)
    certificate = certify_native_point_receiver_chronology(actual["receiver"],
        (0., 0., 0.), actual["receiver_omega"], actual["dt"], point)
    assert certificate.initial_occupancy == "inside"
    assert certificate.occupancy_events == ()
    assert evaluate_native_point_chronology(certificate, Fraction(1, 2)).occupancy == "inside"


def test_general_solver_retains_the_previous_positive_patch_three_event_regression(actual):
    from tectonics.geometric_native_donor_sweep import (
        certify_native_donor_patch_sweep, isolate_native_donor_patch_point_events,
    )
    old = certify_native_donor_patch_sweep(actual["donor"], actual["receiver"],
        actual["donor_omega"], actual["receiver_omega"], actual["dt"])
    x, y = old.patch_x_bounds.midpoint, old.patch_y_bounds.midpoint
    previous = isolate_native_donor_patch_point_events(old, x, y)
    current = _certify(actual, (x, y, Fraction(-1)))
    assert [event.kind for event in current.occupancy_events] == ["entry", "exit", "entry"]
    assert len(current.occupancy_events) == len(previous)
    for event, old_event in zip(current.occupancy_events, previous):
        assert event.kind == old_event.kind
        assert event.time_fraction.lo <= old_event.time_fraction.hi
        assert old_event.time_fraction.lo <= event.time_fraction.hi
        assert event.time_fraction.lo > 0
    assert _occupancy_samples(current) == ["outside", "inside", "outside", "inside"]


def test_real_transverse_receiver_plane_exit_and_return_are_not_masked_by_main_plane(actual):
    x_first, _ = _first_two_normal_coefficients((-1, 0, 0), actual["donor_omega"],
                                                actual["receiver_omega"], actual["dt"])
    z_first, z_second = _first_two_normal_coefficients((0, 0, -1), actual["donor_omega"],
                                                       actual["receiver_omega"], actual["dt"])
    turning = -z_first[1]/(2*z_second[1])
    point = (x_first[1]*turning/1000, Fraction(1), -z_first[1]**2/(8*z_second[1]))
    assert all(_side(NativeRay("seed", point), support) == 1 for support in actual["donor"].support_planes)
    certificate = _certify(actual, point)
    assert [event.kind for event in certificate.occupancy_events] == ["entry", "exit", "entry"]
    planes = actual["receiver"].support_planes
    assert [planes[event.support_index].normal for event in certificate.occupancy_events] == [
        (-1, 0, 0), (0, 0, -1), (0, 0, -1)]
    assert _occupancy_samples(certificate) == ["outside", "inside", "outside", "inside"]
    second, third = certificate.occupancy_events[1:]
    assert 0 < second.time_fraction.lo < second.time_fraction.hi < turning
    assert turning < third.time_fraction.lo < third.time_fraction.hi < 4*turning


def test_main_halfspace_entry_is_masked_until_the_actual_transverse_receiver_return(actual):
    from tectonics.geometric_native_point_chronology import evaluate_native_point_chronology
    x_first, _ = _first_two_normal_coefficients((-1, 0, 0), actual["donor_omega"],
                                                actual["receiver_omega"], actual["dt"])
    z_first, z_second = _first_two_normal_coefficients((0, 0, -1), actual["donor_omega"],
                                                       actual["receiver_omega"], actual["dt"])
    turning = -z_first[1]/(2*z_second[1])
    point = (x_first[1]*turning/2, Fraction(1), -z_first[1]**2/(8*z_second[1]))
    certificate = _certify(actual, point)
    assert [event.kind for event in certificate.plane_events] == ["exit", "entry", "entry"]
    planes = actual["receiver"].support_planes
    assert [planes[event.support_index].normal for event in certificate.plane_events] == [
        (0, 0, -1), (-1, 0, 0), (0, 0, -1)]
    entry, = certificate.occupancy_events
    assert entry.kind == "entry" and planes[entry.support_index].normal == (0, 0, -1)
    assert [evaluate_native_point_chronology(certificate, fraction).occupancy for fraction in
            (Fraction(0), turning/16, turning, 4*turning, Fraction(1))] == [
        "outside", "outside", "outside", "inside", "inside"]


def test_exact_initial_boundary_zero_remains_an_explicit_event_window(actual):
    from tectonics.geometric_native_point_chronology import evaluate_native_point_chronology
    certificate = _certify(actual, (0, 1, -1))
    assert certificate.initial_occupancy == "inside"
    support, = [item for item in certificate.plane_chronologies if item.support.normal == (-1, 0, 0)]
    assert support.initial_zero_order == 1
    assert support.coincident is False
    endpoint = evaluate_native_point_chronology(certificate, Fraction(0))
    assert endpoint.occupancy == "event_window"
    assert support.support_index in endpoint.active_support_indices
    assert evaluate_native_point_chronology(certificate, Fraction(1, 2)).occupancy == "inside"


def test_true_full_three_dimensional_synthetic_motion_crosses_the_real_receiver(actual):
    from tectonics.geometric_native_point_chronology import certify_native_point_chronology
    donor, receiver = (.13, -.07, .19), (-.02, .05, .08)
    duration, point = .1, (Fraction(3, 1000), Fraction(2), Fraction(-1))
    certificate = certify_native_point_chronology(actual["donor"], actual["receiver"],
        donor, receiver, duration, point)
    assert [event.kind for event in certificate.occupancy_events] == ["entry"]
    event, = certificate.occupancy_events
    assert Fraction(1, 4) < event.time_fraction.lo < event.time_fraction.hi < Fraction(1, 2)
    for fraction, expected in ((Fraction(1, 4), False), (Fraction(1, 2), True)):
        ray = NativeRay("seed", point).rotated((RotationIncrement(donor, duration*float(fraction)),))
        supports = actual["receiver"].rotated(receiver, duration*float(fraction)).support_planes
        # Existing native predicates independently check the full poses at
        # dyadic times; they do not import the chronology's sign certificate.
        assert all(_side(ray, plane) >= 0 for plane in supports) is expected


def test_exact_cayley_hamilton_identity_uses_both_complete_owner_generators(actual):
    from tectonics.geometric_native_point_chronology import certify_native_point_chronology
    donor, receiver, dt = (.13, -.07, .19), (-.02, .05, .08), .1
    polynomial = _expected_characteristic_polynomial(donor, receiver, dt)
    generator = _kronecker_generator(donor, receiver, dt)
    identity = tuple(tuple(Fraction(i == j) for j in range(9)) for i in range(9))
    value = tuple(tuple(Fraction(0) for _ in range(9)) for _ in range(9))
    for coefficient in reversed(polynomial):
        value = _matrix_product(value, generator)
        value = tuple(tuple(entry+coefficient*identity[i][j] for j, entry in enumerate(row))
                      for i, row in enumerate(value))
    assert all(entry == 0 for row in value for entry in row)
    certificate = certify_native_point_chronology(actual["donor"], actual["receiver"],
        donor, receiver, dt, (Fraction(3, 1000), 2, -1))
    assert certificate.generator_characteristic_polynomial == polynomial
    assert ordinary_polynomial(actual) == _expected_characteristic_polynomial(
        actual["donor_omega"], actual["receiver_omega"], actual["dt"])


def ordinary_polynomial(actual):
    return _certify(actual, (1, 1, -1)).generator_characteristic_polynomial


def test_same_full_three_dimensional_owners_prove_exact_coincidence_without_false_events(actual):
    from tectonics.geometric_native_point_chronology import (
        certify_native_point_chronology, evaluate_native_point_chronology,
    )
    omega = (.13, -.07, .19)
    certificate = certify_native_point_chronology(actual["donor"], actual["receiver"],
        omega, omega, .1, (0, 1, -1))
    support, = [item for item in certificate.plane_chronologies if item.support.normal == (-1, 0, 0)]
    assert support.coincident is True
    assert certificate.plane_events == certificate.occupancy_events == ()
    for fraction in (Fraction(0), Fraction(1, 3), Fraction(1)):
        query = evaluate_native_point_chronology(certificate, fraction)
        assert query.occupancy == "inside"
        assert query.coincident_support_indices == (support.support_index,)


@pytest.mark.parametrize("budget", [dict(max_depth=1), dict(max_cells=1)])
def test_insufficient_proof_budget_refuses_the_unresolved_actual_tiny_roots(actual, budget):
    from tectonics.geometric_native_donor_sweep import certify_native_donor_patch_sweep
    old = certify_native_donor_patch_sweep(actual["donor"], actual["receiver"],
        actual["donor_omega"], actual["receiver_omega"], actual["dt"])
    with pytest.raises(ValueError):
        _certify(actual, (old.patch_x_bounds.midpoint, old.patch_y_bounds.midpoint, Fraction(-1)), **budget)


def _native_polygon(rays):
    vertices = tuple(NativeRay("seed", tuple(ray)) for ray in rays)
    supports = tuple(NativePlane(_cross(a.reference, b.reference))
                     for a, b in zip(vertices, vertices[1:]+vertices[:1]))
    return validate_native_polygon(NativePolygon(vertices, supports))


def test_interior_tangency_refuses_instead_of_inventing_material_entry(actual):
    from tectonics.geometric_native_point_chronology import certify_native_point_chronology
    # g(theta)=401-399*cos(theta)-40*sin(theta) has an exact interior
    # double zero at theta=2*atan(1/20). Its equal-magnitude identity is exact.
    normal, touch = (-399, -40, -401), (399, 40, -401)
    assert 401**2 == 399**2+40**2
    tangent = _cross(normal, touch)
    first = tuple(Fraction(q)-Fraction(v, 1000) for q, v in zip(touch, tangent))
    second = tuple(Fraction(q)+Fraction(v, 1000) for q, v in zip(touch, tangent))
    inward = tuple(Fraction(q)+Fraction(n, 10) for q, n in zip(touch, normal))
    receiver = _native_polygon((first, second, inward))
    omega, duration = (0., 0., 1.), .2
    assert receiver.pose == ()
    with pytest.raises(ValueError):
        certify_native_point_chronology(actual["donor"], receiver, omega, (0., 0., 0.),
            duration, (1, 0, -1), max_depth=192, max_cells=512)


def test_unresolved_simultaneous_receiver_plane_crossings_refuse_atomic_ordering(actual):
    from tectonics.geometric_native_point_chronology import certify_native_point_chronology
    # This original trajectory passes exactly through a native receiver
    # vertex (399,40,-401) at the same theta=2*atan(1/20). Two independent
    # receiver supports cross in opposite directions at one physical time.
    touch, normal_a, normal_b = (399, 40, -401), (-40, 399, 0), (401, 0, 399)
    tangent_a, tangent_b = _cross(normal_a, touch), _cross(normal_b, touch)
    second = tuple(Fraction(q)+Fraction(v, 1000) for q, v in zip(touch, tangent_b))
    third = tuple(Fraction(q)-Fraction(v, 1000) for q, v in zip(touch, tangent_a))
    receiver = _native_polygon((touch, second, third))
    omega, duration = (0., 0., 1.), .2
    with pytest.raises(ValueError):
        certify_native_point_chronology(actual["donor"], receiver, omega, (0., 0., 0.),
            duration, (401, 0, -401), max_depth=192, max_cells=512)


def test_strict_chronology_restart_recomputes_proof_and_preserves_the_input(ordinary):
    from tectonics.geometric_native_point_chronology import (
        native_point_chronology_from_dict, native_point_chronology_to_dict,
    )
    raw = native_point_chronology_to_dict(ordinary)
    copied, before = deepcopy(raw), _encoded(raw)
    loaded = native_point_chronology_from_dict(copied)
    assert _encoded(native_point_chronology_to_dict(loaded)) == before
    assert _encoded(copied) == before


def test_integer_and_fraction_point_inputs_have_the_same_canonical_proof_after_warmup(actual):
    from tectonics.geometric_native_point_chronology import native_point_chronology_to_dict
    first = _certify(actual, (1, 1, -1))
    second = _certify(actual, (Fraction(1), Fraction(1), Fraction(-1)))
    assert _encoded(native_point_chronology_to_dict(first)) == _encoded(native_point_chronology_to_dict(second))
    assert all(type(coordinate) is Fraction for coordinate in second.point)


@pytest.mark.parametrize("point", [(True, 1, -1), (.1, 1, -1), [1, 1, -1], (1, -1), "1,1,-1"])
def test_point_coordinates_have_strict_exact_types_even_after_cache_warmup(actual, ordinary, point):
    with pytest.raises((TypeError, ValueError)):
        _certify(actual, point)


@pytest.mark.parametrize("budget", [dict(precision=True), dict(precision=256.), dict(order=True),
                                   dict(order=24.), dict(max_depth=True), dict(max_depth=192.),
                                   dict(max_cells=True), dict(max_cells=4096.)])
def test_proof_budget_types_cannot_alias_a_cached_valid_certificate(actual, ordinary, budget):
    with pytest.raises(ValueError):
        _certify(actual, ordinary.point, **budget)


@pytest.mark.parametrize("fraction", [0., True, Fraction(-1), Fraction(2), "0"])
def test_query_time_is_exact_and_stays_within_the_original_interval(ordinary, fraction):
    from tectonics.geometric_native_point_chronology import evaluate_native_point_chronology
    with pytest.raises((TypeError, ValueError)):
        evaluate_native_point_chronology(ordinary, fraction)


def test_query_inside_a_root_enclosure_does_not_grant_an_occupancy(ordinary):
    from tectonics.geometric_native_point_chronology import evaluate_native_point_chronology
    event, = ordinary.occupancy_events
    query = evaluate_native_point_chronology(ordinary, event.time_fraction.midpoint)
    assert query.occupancy == "event_window"
    assert event.support_index in query.active_support_indices


@pytest.mark.parametrize("damage", ["occupancy", "float_time", "lost_epoch", "plane_signs",
                                  "certificate_sha256", "active_indices", "doubled_events"])
def test_query_serializer_reproves_derived_answers_against_their_actual_certificate(ordinary, damage):
    from tectonics.geometric_native_point_chronology import (
        evaluate_native_point_chronology, native_point_chronology_query_to_dict,
    )
    event, = ordinary.occupancy_events
    query = evaluate_native_point_chronology(ordinary, event.time_fraction.midpoint)
    if damage == "occupancy":
        forged = replace(query, occupancy="inside")
    elif damage == "float_time":
        forged = replace(query, time_fraction=float(query.time_fraction))
    elif damage == "lost_epoch":
        forged = replace(query, time_myr=Fraction(0))
    elif damage == "plane_signs":
        forged = replace(query, plane_signs=(1, 1, 1))
    elif damage == "certificate_sha256":
        forged = replace(query, certificate_sha256="0"*64)
    elif damage == "active_indices":
        forged = replace(query, active_support_indices=())
    else:
        assert query.event_sequence
        forged = replace(query, event_sequence=2*query.event_sequence)
    with pytest.raises(ValueError):
        native_point_chronology_query_to_dict(forged)


def test_query_restart_replays_exact_certificate_time_and_full_polygon_answer(ordinary):
    from tectonics.geometric_native_point_chronology import (
        evaluate_native_point_chronology, native_point_chronology_query_from_dict,
        native_point_chronology_query_to_dict,
    )
    event, = ordinary.occupancy_events
    query = evaluate_native_point_chronology(ordinary, event.time_fraction.midpoint)
    raw = native_point_chronology_query_to_dict(query)
    copied, before = deepcopy(raw), _encoded(raw)
    loaded = native_point_chronology_query_from_dict(copied, certificate=ordinary)
    assert _encoded(native_point_chronology_query_to_dict(loaded)) == before
    assert _encoded(copied) == before
    assert loaded.occupancy == "event_window"
    assert loaded.time_myr == loaded.time_fraction*Fraction(ordinary.dt_myr)


@pytest.mark.parametrize("damage", ["occupancy", "time_myr", "plane_signs", "event_sequence"])
def test_query_dto_cannot_grant_a_false_state_even_with_an_intact_certificate(ordinary, damage):
    from tectonics.geometric_native_point_chronology import (
        evaluate_native_point_chronology, native_point_chronology_query_from_dict,
        native_point_chronology_query_to_dict,
    )
    event, = ordinary.occupancy_events
    raw = deepcopy(native_point_chronology_query_to_dict(
        evaluate_native_point_chronology(ordinary, event.time_fraction.midpoint)))
    if damage == "occupancy":
        raw["occupancy"] = "inside"
    elif damage == "time_myr":
        raw["time_myr"] = [0, 1]
    elif damage == "plane_signs":
        raw["plane_signs"] = [1, 1, 1]
    else:
        assert raw["event_sequence"]
        raw["event_sequence"] *= 2
    with pytest.raises(ValueError):
        native_point_chronology_query_from_dict(raw, certificate=ordinary)


def test_query_restart_rejects_another_valid_source_certificate(actual, ordinary):
    from tectonics.geometric_native_point_chronology import (
        evaluate_native_point_chronology, native_point_chronology_query_from_dict,
        native_point_chronology_query_to_dict,
    )
    raw = native_point_chronology_query_to_dict(evaluate_native_point_chronology(ordinary, Fraction(1, 4)))
    another_point = _certify(actual, (1, 1, -1))
    with pytest.raises(ValueError):
        native_point_chronology_query_from_dict(raw, certificate=another_point)


@pytest.mark.parametrize("damage", ["lost_event", "zero_time", "wrong_characteristic", "projected_omega",
                                  "float_precision", "source_id", "force_acceptance", "extra_field"])
def test_resealed_chronology_dto_rejects_forged_physical_proof_and_authority(ordinary, damage):
    from tectonics.geometric_native_point_chronology import (
        native_point_chronology_from_dict, native_point_chronology_to_dict,
    )
    raw = deepcopy(native_point_chronology_to_dict(ordinary))
    if damage == "float_precision":
        raw["precision"] = float(raw["precision"])
    elif damage == "source_id":
        raw["source_fragment_id"] = "foreign-material"
    elif damage == "force_acceptance":
        raw["admission_authorized"] = True
    elif damage == "extra_field":
        raw["receiver_scope_is_complete"] = True
    else:
        target = {"lost_event": "occupancy_events", "zero_time": "occupancy_events",
                  "wrong_characteristic": "generator_characteristic_polynomial",
                  "projected_omega": "donor_omega"}[damage]
        def corrupt(item):
            if not isinstance(item, dict):
                return False
            for key, value in item.items():
                if key == target or (damage == "projected_omega" and key == "donor_omega_rad_per_myr"):
                    if damage == "lost_event":
                        value.clear()
                    elif damage == "zero_time":
                        value[0]["time_fraction"]["lo"] = [0, 1]
                    elif damage == "wrong_characteristic":
                        value[-1] = [0, 1]
                    else:
                        value[0] = 0.
                    return True
                if corrupt(value):
                    return True
            return False
        assert corrupt(raw), f"Missing chronology field for {damage}"
    seal_keys = [key for key in raw if key.endswith("sha256")]
    assert len(seal_keys) == 1
    seal, = seal_keys
    del raw[seal]
    raw[seal] = hashlib.sha256(_encoded(raw)).hexdigest()
    with pytest.raises((TypeError, ValueError)):
        native_point_chronology_from_dict(raw)


def test_a_forged_frozen_event_sequence_is_reproved_before_query(ordinary):
    from tectonics.geometric_native_point_chronology import evaluate_native_point_chronology
    forged = replace(ordinary, occupancy_events=())
    with pytest.raises(ValueError):
        evaluate_native_point_chronology(forged, Fraction(1, 2))


def test_real_swept_only_material_enters_and_exits_before_the_actual_endpoint_loss(actual):
    from tectonics.geometric_native_point_chronology import (
        certify_native_point_chronology, evaluate_native_point_chronology,
    )
    from tectonics.geometric_native_swept_domain import certify_native_swept_domain
    # The complete 6/2 cone supplies a strictly positive difference between
    # its initial and terminal ratios. Its mirrored 5/1 source pair carries
    # true material that contacts the receiver and then leaves before t=1.
    cone = certify_native_swept_domain(actual["donor"], actual["receiver"],
        actual["donor_omega"], actual["receiver_omega"], actual["dt"])
    delta = cone.initial_q_bounds.lo-cone.terminal_q_bounds.hi
    assert delta > 0
    yabs = (cone.initial_q_bounds.lo+cone.terminal_q_bounds.hi)/2
    first, _ = _first_two_normal_coefficients((-1, 0, 0), actual["donor_omega"],
                                              actual["receiver_omega"], actual["dt"])
    point = (first[1]*delta/100, -yabs, Fraction(1))
    source = {fragment["fragment_id"]: fragment for fragment in
              json.loads(actual["initial_bytes"])["current_surface"]["fragments"]}
    donor = native_polygon_from_dict(source["octant:5"]["geometry"])
    receiver = native_polygon_from_dict(source["octant:1"]["geometry"])
    assert source["octant:5"]["parcel"]["plate"] == 1
    assert source["octant:1"]["parcel"]["plate"] == 0
    ray = NativeRay("seed", point)
    assert all(_side(ray, plane) == 1 for plane in donor.support_planes)
    certificate = certify_native_point_chronology(donor, receiver, actual["donor_omega"],
        actual["receiver_omega"], actual["dt"], point)
    assert [event.kind for event in certificate.occupancy_events] == ["entry", "exit"]
    assert _occupancy_samples(certificate) == ["outside", "inside", "outside"]
    fractions = (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1))
    main = next(plane for plane in receiver.support_planes if plane.normal == (-1, 0, 0))
    for fraction, expected in zip(fractions, (-1, 1, 1, -1, -1)):
        dt = actual["dt"]*float(fraction)
        if not fraction:
            moved, support = ray, main
        else:
            moved = ray.rotated((RotationIncrement(actual["donor_omega"], dt),))
            support = main.rotated((RotationIncrement(actual["receiver_omega"], dt),))
        assert _side(moved, support) == expected
    for plane in certificate.plane_chronologies:
        if plane.support.normal in ((0, -1, 0), (0, 0, 1)):
            assert plane.events == ()
            assert all(cell.status == "no_root" and cell.sign == 1 for cell in plane.cells)
    entry, exit_event = certificate.occupancy_events
    assert 0 < entry.time_fraction.lo < entry.time_fraction.hi < Fraction(1, 4)
    assert Fraction(1, 2) < exit_event.time_fraction.lo < exit_event.time_fraction.hi < Fraction(3, 4)
    final = evaluate_native_point_chronology(certificate, Fraction(1))
    assert final.occupancy == "outside"
    assert entry.time_fraction.hi < 1
    # Native endpoint containment independently refuses this already
    # contacted material. A final loss footprint cannot replace its history.
    assessment, = [item for item in actual["probe"]["admission"]["actual_overlap_assessments"]
                    if item["fragment_a"] == "octant:1" and item["fragment_b"] == "octant:5"]
    overlap = native_polygon_from_dict(assessment["native_overlap"])
    moved = ray.rotated((RotationIncrement(actual["donor_omega"], actual["dt"]),))
    assert any(_side(moved, plane) == -1 for plane in overlap.support_planes)
    assert (CONTROL / "force_candidate_refusal_probe.json").read_bytes() == actual["probe_bytes"]
    assert (CONTROL / "initial.json").read_bytes() == actual["initial_bytes"]
