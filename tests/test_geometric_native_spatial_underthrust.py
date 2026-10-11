"""Native continuous admission controls retain true tiny support and polarity."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import json
import math

import pytest

from tests.test_geometric_spatial_underthrust import (
    GEOMETRIC_DT, MOTION, admission_fixture, birth,
)
from tectonics.geometric_intrinsic_geometry import RotationIncrement
from tectonics.geometric_native_polygons import NativeRay, NativePlane, _primitive, native_intersection
from tectonics.geometric_native_spatial_underthrust import (
    _arc, _material_support_interval, _native_velocity_status, _trace_arc, _rotated, _polygon_arcs,
    attach_native_underthrust_losses, make_native_spatial_underthrust_resolver,
    native_intersect_support_arcs, native_support_covers_arc,
)
from tectonics.geometric_native_surface import (
    advance_native_surface, enroll_native_surface, native_surface_to_dict,
)
from tectonics.geometric_transport import UnresolvedPolarityError


def ray(values):
    return NativeRay("seed", _primitive(values))


def xy_arc(first, last):
    return _arc(ray(first), ray(last), NativePlane((0, 0, 1)))


@pytest.fixture(scope="module")
def accepted():
    _, origin, seed, step = admission_fixture()
    before = enroll_native_surface(origin)
    omega = {owner: MOTION[owner] for owner in seed.plate_ids}
    histories = tuple((owner, ()) for owner in seed.plate_ids)
    resolve, audit = make_native_spatial_underthrust_resolver(before, omega,
        GEOMETRIC_DT, origin_surface=origin, initial_mechanics=seed,
        owner_histories=histories, step=step)
    transaction = advance_native_surface(before, omega, GEOMETRIC_DT,
        birth_factory=birth, polarity_resolver=resolve)
    after_histories = tuple((owner, (RotationIncrement(omega[owner], GEOMETRIC_DT),))
                           for owner in seed.plate_ids)
    attached, receipt = attach_native_underthrust_losses(before, transaction,
        origin_surface=origin, initial_mechanics=seed, owner_histories_after=after_histories,
        step=step, admission_audit=audit)
    return before, origin, seed, step, omega, after_histories, transaction, audit, attached, receipt


def test_actual_declared_continuous_field_binds_two_native_losses_to_endpoint_support(accepted):
    before, _, _, _, _, _, transaction, _, attached, receipt = accepted
    assert len(transaction.losses) == len(attached.losses) == len(receipt["attachments"]) == 2
    assert len(transaction.births) == 2
    assert len(attached.contacts) == 4
    assert all(loss.attachment_status == "unresolved_native_support" for loss in transaction.losses)
    endpoint = {contact.contact_id for contact in attached.contacts}
    for loss, row in zip(attached.losses, receipt["attachments"]):
        assert loss.attachment_status == "geometric_contact"
        assert loss.contact_ids and set(loss.contact_ids) <= endpoint
        assert loss.polarity_basis == "solved_spatial_underthrust_admission"
        assert row["loss_fragment_id"] == loss.fragment.fragment_id
        assert row["supports"]
        assert all(support["donor_material_id"] == loss.fragment.parcel.material_id
                   for support in row["supports"])
    assert attached.diagnostics["attached_loss_count"] == 2
    assert attached.diagnostics["unresolved_attachment_count"] == 0
    assert attached.state is transaction.state
    assert before.time_myr == 0.


def test_rehashed_false_admission_manifest_cannot_grant_native_attachment(accepted):
    before, origin, seed, step, _, histories, transaction, audit, _, _ = accepted
    changed = deepcopy(audit)
    changed["node_manifest_sha256"] = "0"*64
    original = json.dumps(native_surface_to_dict(before), sort_keys=True)
    with pytest.raises(ValueError, match="fresh field/location replay"):
        attach_native_underthrust_losses(before, transaction, origin_surface=origin,
            initial_mechanics=seed, owner_histories_after=histories,
            step=step, admission_audit=changed)
    assert json.dumps(native_surface_to_dict(before), sort_keys=True) == original
    assert all(loss.attachment_status == "unresolved_native_support" for loss in transaction.losses)


def test_forged_constitutive_step_fails_original_node_replay(accepted):
    before, origin, seed, step, omega, _, _, _, _, _ = accepted
    with pytest.raises(ValueError, match="authenticated node/field replay"):
        make_native_spatial_underthrust_resolver(before, omega, GEOMETRIC_DT,
            origin_surface=origin, initial_mechanics=seed,
            owner_histories=tuple((owner, ()) for owner in seed.plate_ids),
            step=replace(step, node_manifest_sha256="0"*64))


def test_unloaded_extra_pose_interval_cannot_change_original_memory_clock(accepted):
    before, origin, seed, step, omega, _, _, _, _, _ = accepted
    histories = tuple((owner, (RotationIncrement((0., 0., 0.), GEOMETRIC_DT),))
                      for owner in seed.plate_ids)
    with pytest.raises(ValueError, match="memory clocks disagree"):
        make_native_spatial_underthrust_resolver(before, omega, GEOMETRIC_DT,
            origin_surface=origin, initial_mechanics=seed, owner_histories=histories, step=step)


def test_locked_original_field_refuses_before_any_birth_and_preserves_input():
    _, origin, seed, step = admission_fixture(locked=True)
    before = enroll_native_surface(origin)
    original = json.dumps(native_surface_to_dict(before), sort_keys=True)
    omega = {owner: MOTION[owner] for owner in seed.plate_ids}
    resolve, _ = make_native_spatial_underthrust_resolver(before, omega, GEOMETRIC_DT,
        origin_surface=origin, initial_mechanics=seed,
        owner_histories=tuple((owner, ()) for owner in seed.plate_ids), step=step)
    calls = []
    def forbidden_birth(*args):
        calls.append(args)
        raise AssertionError("Locked native admission reached birth")
    with pytest.raises(UnresolvedPolarityError):
        advance_native_surface(before, omega, GEOMETRIC_DT,
            birth_factory=forbidden_birth, polarity_resolver=resolve)
    assert not calls
    assert json.dumps(native_surface_to_dict(before), sort_keys=True) == original


def test_complete_native_arc_union_requires_no_positive_uncovered_gap():
    whole = xy_arc((1, 0, 0), (0, 1, 0))
    first = xy_arc((1, 0, 0), (1, 1, 0))
    second = xy_arc((1, 1, 0), (0, 1, 0))
    assert native_support_covers_arc(whole, (second, first))
    tiny = Fraction(1, 10**150)
    left = xy_arc((1, 0, 0), (2*tiny, 1, 0))
    right = xy_arc((tiny, 1, 0), (0, 1, 0))
    # Both absolute binary64 material angles round to pi/2, but the positive
    # native gap must still prevent whole-arc coverage.
    assert not native_support_covers_arc(whole, (left, right))
    assert native_intersect_support_arcs(left, right) is None


def test_foreign_plane_and_antipodal_quarter_never_supply_native_support():
    whole = xy_arc((1, 0, 0), (0, 1, 0))
    tiny = Fraction(1, 10**150)
    foreign = _arc(ray((1, 0, 0)), ray((0, 1, -tiny)), NativePlane((0, tiny, 1)))
    assert native_intersect_support_arcs(whole, foreign) is None
    remote = xy_arc((-1, 0, 0), (0, -1, 0))
    assert native_intersect_support_arcs(whole, remote) is None


@pytest.mark.parametrize("y,status,maximum", [
    (-1e-20, "spatial_underthrust_admissible", -9e-18),
    (-1e-19, "spatial_underthrust_admissible", 0.),
    (-2e-19, "mixed_convergence", None),
])
def test_true_tiny_pole_arc_keeps_full_off_axis_velocity_and_exact_zero(accepted, y, status, maximum):
    _, _, seed, _, _, _, _, _, _, _ = accepted
    trace = replace(seed.fault_traces.traces[0], start=(0., -1., 0.), end=(0., 0., 1.))
    carrier = _trace_arc(trace, ())
    actual = _arc(ray((0., y, 1.)), carrier.end, carrier.plane)
    omega = {trace.subducting_plate: (0., 0., 0.),
             trace.overriding_plate: (0., 1e-19, -1.)}
    computed, low, high, width = _native_velocity_status(carrier, actual, trace, omega, 100.)
    assert computed == status
    assert actual.length_radians > 0.
    assert width >= 0.
    if maximum is not None:
        assert high == pytest.approx(maximum, rel=2e-15, abs=0.)
        assert low < 0. and high <= 0.
    low_angle, high_angle = _material_support_interval(carrier, actual, math.pi/2.)
    assert low_angle == math.nextafter(math.pi/2., -math.inf)
    assert high_angle == math.pi/2.
    assert actual.end is carrier.end


def test_tampered_native_arc_extent_cannot_claim_continuous_support():
    actual = xy_arc((1, 0, 0), (0, 1, 0))
    with pytest.raises(ValueError, match="length differs"):
        native_support_covers_arc(actual, (replace(actual, length_radians=1.),))
    with pytest.raises(ValueError, match="explicit finite real"):
        native_support_covers_arc(actual, (replace(actual, length_radians=True),))


def test_true_finite_pose_intersection_retains_positive_endpoint_velocity_and_refuses(accepted):
    before, _, seed, _, _, _, _, _, _, _ = accepted
    # Full force-solved vectors from the declared affine source.  Their tiny
    # components are part of the recorded analytic poses.  At this actual
    # finite-rotation intersection, one real endpoint opens by ~8.68e-30.
    omega = {
        0: (-1.4150917253688593e-36, 1.866601350707532e-19, 28.24520547917484),
        1: (1.4150917253688593e-36, -2.1223480037729766e-19, 34.86999452082516),
    }
    dt = 3.168808781402895e-08
    fragments = {fragment.fragment_id: fragment for fragment in before.fragments}
    overlap = native_intersection(fragments["octant:2"].geometry.rotated(omega[0], dt),
                                  fragments["octant:6"].geometry.rotated(omega[1], dt))
    assert overlap is not None
    trace = next(trace for trace in seed.fault_traces.traces
                 if trace.start == (0., 1., 0.) and trace.end == (0., 0., -1.))
    original = _trace_arc(trace, ())
    moved = _rotated(original, (RotationIncrement(omega[0], dt),))
    actual = tuple(shared for edge in _polygon_arcs(overlap)
        if (shared := native_intersect_support_arcs(moved, edge)) is not None)
    assert len(actual) == 1
    before_arc = _rotated(actual[0], (RotationIncrement(tuple(-x for x in omega[0]), dt),))
    endpoints = {}
    status, _, _, _ = _native_velocity_status(original, before_arc, trace,
        omega, before.radius_km, endpoint_audit=endpoints)
    assert status == "mixed_convergence"
    low, high = endpoints["endpoints"]
    assert low["upper_km_per_myr"] < 0.
    assert high["lower_km_per_myr"] > 0.
    assert high["upper_km_per_myr"] == pytest.approx(8.682890018650322e-30, rel=1e-13, abs=0.)
