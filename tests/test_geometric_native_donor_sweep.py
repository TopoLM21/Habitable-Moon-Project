"""Continuous donor material events on a real two-dimensional native patch.

The immutable saved force candidate supplies the complete Euler vectors and
the original native donor, receiver, and overlap.  Positive point samples alone
are not accepted as evidence for a material family.
"""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import pytest

from tectonics.geometric_intrinsic_geometry import RotationIncrement
from tectonics.geometric_native_polygons import (
    _ray_same, _side, native_polygon_from_dict, native_polygon_to_dict,
)


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "analysis/geometric_native_underthrust_validation/force_control_release_02"


def _encoded(raw):
    return json.dumps(raw, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


@pytest.fixture(scope="module")
def actual():
    probe_bytes = (CONTROL / "force_candidate_refusal_probe.json").read_bytes()
    initial_bytes = (CONTROL / "initial.json").read_bytes()
    probe, initial = json.loads(probe_bytes), json.loads(initial_bytes)
    fragments = {fragment["fragment_id"]: fragment
                 for fragment in initial["current_surface"]["fragments"]}
    assessment, = [item for item in probe["admission"]["actual_overlap_assessments"]
                    if item["fragment_a"] == "octant:2" and item["fragment_b"] == "octant:6"]
    receiver, = assessment["receiver_arcs"]
    assert receiver["overriding_plate"] == 0 and receiver["subducting_plate"] == 1
    assert fragments["octant:6"]["parcel"]["plate"] == 1
    assert fragments["octant:2"]["parcel"]["plate"] == 0
    return dict(probe=probe, initial=initial,
        probe_bytes=probe_bytes, initial_bytes=initial_bytes,
        donor=native_polygon_from_dict(fragments["octant:6"]["geometry"]),
        receiver=native_polygon_from_dict(fragments["octant:2"]["geometry"]),
        overlap=native_polygon_from_dict(assessment["native_overlap"]),
        donor_omega=tuple(probe["omega"]["1"]),
        receiver_omega=tuple(probe["omega"]["0"]),
        duration=probe["admission"]["dt_myr"])


@pytest.fixture(scope="module")
def certificate(actual):
    from tectonics.geometric_native_donor_sweep import certify_native_donor_patch_sweep
    return certify_native_donor_patch_sweep(actual["donor"], actual["receiver"],
        actual["donor_omega"], actual["receiver_omega"], actual["duration"])


def test_actual_patch_preserves_full_solved_motion_and_unaccepted_source(actual, certificate):
    assert certificate.donor_omega == actual["donor_omega"]
    assert certificate.receiver_omega == actual["receiver_omega"]
    assert all(component != 0 for omega in (certificate.donor_omega,
                                           certificate.receiver_omega) for component in omega)
    assert certificate.dt_myr == actual["duration"]
    assert certificate.donor_polygon == actual["donor"]
    assert certificate.receiver_polygon == actual["receiver"]
    assert actual["probe"]["accepted_loaded_native_intervals"] == 0
    assert actual["probe"]["input_unchanged"] is True
    assert (CONTROL / "force_candidate_refusal_probe.json").read_bytes() == actual["probe_bytes"]
    assert (CONTROL / "initial.json").read_bytes() == actual["initial_bytes"]


def test_true_positive_two_dimensional_donor_patch_has_no_area_floor(certificate):
    x, y = certificate.patch_x_bounds, certificate.patch_y_bounds
    area = certificate.patch_area_bounds
    assert 0 < x.lo < x.hi and 0 < y.lo < y.hi
    assert x.width > 0 and y.width > 0
    assert 0 < area.lo < area.hi < Fraction(1, 10**100)
    # The same physical patch must exist as a native polygon, rather than an
    # arc label or a caller-supplied assertion that material has positive area.
    native_area = certificate.patch_geometry.area_bounds()
    assert native_area.lo > 0
    assert native_area.lo <= area.hi and area.lo <= native_area.hi
    assert len(certificate.patch_geometry.vertices) == 4


def test_patch_is_strictly_inside_original_donor_and_actual_endpoint_overlap(actual, certificate):
    step = RotationIncrement(actual["donor_omega"], actual["duration"])
    for ray in certificate.patch_geometry.vertices:
        assert ray.kind == "seed"
        assert all(_side(ray, support) == 1 for support in actual["donor"].support_planes)
        moved = ray.rotated((step,))
        assert all(_side(moved, support) == 1 for support in actual["overlap"].support_planes)


def test_receiver_plane_family_has_exactly_three_ordered_crossings_for_every_patch_point(certificate):
    fractions = certificate.sample_fractions
    signs = certificate.gap_sample_bounds
    assert len(fractions) == len(signs) == 5
    assert fractions[0] == 0 and fractions[-1] == 1
    assert all(a < b for a, b in zip(fractions, fractions[1:]))
    assert signs[0].hi < 0 < signs[1].lo
    assert signs[2].hi < 0 < signs[3].lo
    assert signs[4].lo > 0
    # Rolle's theorem and the whole-time third derivative bound exclude an
    # extra hidden event anywhere beyond the whole-patch sign proofs.
    assert certificate.gap_third_derivative_bounds.lo > 0
    events = certificate.event_windows
    assert [event.kind for event in events] == ["entry", "exit", "entry"]
    assert 0 < certificate.first_entry_negative_fraction < fractions[1]
    assert certificate.first_entry_negative_gap_bounds.hi < 0
    for index, (event, low, high) in enumerate(zip(events, fractions, fractions[1:])):
        expected_low = certificate.first_entry_negative_fraction if index == 0 else low
        assert event.time_fraction.lo == expected_low and event.time_fraction.hi == high
        expected = event.time_fraction * Fraction(certificate.dt_myr)
        assert event.time_myr == expected


def test_all_receiver_supports_are_proved_over_the_entire_patch_and_time(actual, certificate):
    assert {support.normal for support in actual["receiver"].supports} == {
        (-1, 0, 0), (0, 1, 0), (0, 0, -1)}
    assert len(certificate.receiver_positive_bounds) == 2
    assert all(bound.lo > 0 for bound in certificate.receiver_positive_bounds)
    # Thus the crossings of the moving -X plane are crossings of the real
    # receiver polygon, not of an unbounded surrogate halfspace.


def test_original_donor_endpoint_domain_retains_owner_journal(actual, certificate):
    from tectonics.geometric_native_buried_support import native_pullback
    step = RotationIncrement(actual["donor_omega"], actual["duration"])
    pulled_back = native_pullback(actual["overlap"], (step,))
    proved = certificate.endpoint_donor_domain
    assert proved.pose == pulled_back.pose
    assert proved.supports == pulled_back.supports
    assert all(_ray_same(a, b) for a, b in zip(proved.vertices, pulled_back.vertices))
    assert certificate.endpoint_donor_domain.pose[-1].omega_rad_per_myr == tuple(
        -value for value in actual["donor_omega"])
    assert certificate.endpoint_donor_domain.pose[-1].dt_myr == actual["duration"]


def test_native_donor_certificate_is_immutable(certificate):
    with pytest.raises(AttributeError):
        certificate.dt_myr = 1.


def test_distinct_true_donor_points_have_distinct_ordered_event_times(certificate):
    from tectonics.geometric_native_donor_sweep import isolate_native_donor_patch_point_events
    box_x, box_y = certificate.patch_x_bounds, certificate.patch_y_bounds
    first_x, second_x = box_x.lo+box_x.width/4, box_x.lo+3*box_x.width/4
    first = isolate_native_donor_patch_point_events(certificate, first_x, box_y.midpoint)
    second = isolate_native_donor_patch_point_events(certificate, second_x, box_y.midpoint)
    for events in (first, second):
        assert [event.kind for event in events] == ["entry", "exit", "entry"]
        assert all(a.time_fraction.hi < b.time_fraction.lo
                   for a, b in zip(events, events[1:]))
        for event, window in zip(events, certificate.event_windows):
            assert window.time_fraction.lo < event.time_fraction.lo
            assert event.time_fraction.hi < window.time_fraction.hi
            assert event.time_myr == event.time_fraction*Fraction(certificate.dt_myr)
    # A donor volume carries spatially varying crossing times. Copying one
    # receiver-arc event clock to every point would fail these inequalities.
    assert first[0].time_fraction.hi < second[0].time_fraction.lo
    assert second[1].time_fraction.hi < first[1].time_fraction.lo
    assert first[2].time_fraction.hi < second[2].time_fraction.lo


@pytest.mark.parametrize("query", ["float_x", "bool_y", "x_below", "x_above", "y_below", "y_above"])
def test_point_events_require_exact_coordinates_inside_the_whole_proved_patch(certificate, query):
    from tectonics.geometric_native_donor_sweep import isolate_native_donor_patch_point_events
    bx, by = certificate.patch_x_bounds, certificate.patch_y_bounds
    x, y = bx.midpoint, by.midpoint
    if query == "float_x":
        x = float(x)
    elif query == "bool_y":
        y = True
    elif query == "x_below":
        x = bx.lo-bx.width
    elif query == "x_above":
        x = bx.hi+bx.width
    elif query == "y_below":
        y = by.lo-by.width
    else:
        y = by.hi+by.width
    with pytest.raises(ValueError):
        isolate_native_donor_patch_point_events(certificate, x, y)


def test_donor_proof_dto_replays_the_same_complete_material_family(certificate):
    from tectonics.geometric_native_donor_sweep import (
        native_donor_patch_sweep_from_dict, native_donor_patch_sweep_to_dict,
    )
    dto = native_donor_patch_sweep_to_dict(certificate)
    copied = deepcopy(dto)
    before = _encoded(copied)
    loaded = native_donor_patch_sweep_from_dict(copied)
    assert _encoded(native_donor_patch_sweep_to_dict(loaded)) == _encoded(dto)
    assert _encoded(copied) == before
    assert dto["admission_authorized"] is False
    assert dto["scope"] == "positive_original_donor_2D_patch_not_complete_swept_domain"


@pytest.mark.parametrize("motion", ["zero", "projected_z", "reverse_donor", "reverse_receiver"])
def test_changed_motion_cannot_claim_the_actual_complete_donor_sweep(actual, motion):
    from tectonics.geometric_native_donor_sweep import (
        UnresolvedNativeDonorSweep, certify_native_donor_patch_sweep,
    )
    donor, receiver = actual["donor_omega"], actual["receiver_omega"]
    if motion == "zero":
        donor = receiver = (0., 0., 0.)
    elif motion == "projected_z":
        donor, receiver = (0., 0., donor[2]), (0., 0., receiver[2])
    elif motion == "reverse_donor":
        donor = tuple(-value for value in donor)
    else:
        receiver = tuple(-value for value in receiver)
    with pytest.raises(UnresolvedNativeDonorSweep):
        certify_native_donor_patch_sweep(actual["donor"], actual["receiver"],
            donor, receiver, actual["duration"])


@pytest.mark.parametrize("geometry", ["swapped_owners", "moved_donor", "moved_receiver", "foreign_octant"])
def test_altered_native_parent_geometry_refuses_the_bounded_axis_proof(actual, geometry):
    from tectonics.geometric_native_donor_sweep import certify_native_donor_patch_sweep
    donor, receiver = actual["donor"], actual["receiver"]
    if geometry == "swapped_owners":
        donor, receiver = receiver, donor
    elif geometry == "moved_donor":
        donor = donor.rotated(actual["donor_omega"], actual["duration"])
    elif geometry == "moved_receiver":
        receiver = receiver.rotated(actual["receiver_omega"], actual["duration"])
    else:
        raw = deepcopy(native_polygon_to_dict(receiver))
        for ray, vector in zip(raw["vertices"], ([-1, 0, 0], [0, 0, -1], [0, -1, 0])):
            ray["reference"] = vector
        for support, vector in zip(raw["supports"], ([0, -1, 0], [-1, 0, 0], [0, 0, -1])):
            support["normal"] = vector
        receiver = native_polygon_from_dict(raw)
    with pytest.raises(ValueError):
        certify_native_donor_patch_sweep(donor, receiver, actual["donor_omega"],
                                       actual["receiver_omega"], actual["duration"])


@pytest.mark.parametrize("duration", [0., -1., True, float("nan"), float("inf"), "1"])
def test_donor_sweep_requires_finite_positive_same_physical_time(actual, duration):
    from tectonics.geometric_native_donor_sweep import certify_native_donor_patch_sweep
    with pytest.raises((TypeError, ValueError)):
        certify_native_donor_patch_sweep(actual["donor"], actual["receiver"],
            actual["donor_omega"], actual["receiver_omega"], duration)


@pytest.mark.parametrize("omega", [(0., 1.), (True, 0., 1.), (float("nan"), 0., 1.),
                                 (0., float("inf"), 1.), [0., 0., 1.], "0,0,1"])
def test_donor_sweep_requires_immutable_complete_finite_owner_vectors(actual, omega):
    from tectonics.geometric_native_donor_sweep import certify_native_donor_patch_sweep
    with pytest.raises((TypeError, ValueError)):
        certify_native_donor_patch_sweep(actual["donor"], actual["receiver"],
            omega, actual["receiver_omega"], actual["duration"])


@pytest.mark.parametrize("precision,order", [(True, 20), (256., 20), (31, 20), (1025, 20),
                                          (256, True), (256, 20.), (256, 3), (256, 65)])
def test_proof_budget_types_and_domain_are_explicit(actual, precision, order):
    from tectonics.geometric_native_donor_sweep import certify_native_donor_patch_sweep
    with pytest.raises(ValueError):
        certify_native_donor_patch_sweep(actual["donor"], actual["receiver"],
            actual["donor_omega"], actual["receiver_omega"], actual["duration"],
            precision=precision, order=order)


@pytest.mark.parametrize("damage", ["wrong_area", "hidden_event", "lost_receiver_plane",
                                  "wrong_owner", "source_id", "force_acceptance", "float_precision"])
def test_resealed_donor_dto_cannot_change_geometry_event_count_or_authority(certificate, damage):
    from tectonics.geometric_native_donor_sweep import (
        native_donor_patch_sweep_from_dict, native_donor_patch_sweep_to_dict,
    )
    dto = deepcopy(native_donor_patch_sweep_to_dict(certificate))
    if damage == "source_id":
        dto["source_fragment_id"] = "unrelated-positive-material"
    elif damage == "force_acceptance":
        dto["admission_authorized"] = True
    elif damage == "float_precision":
        dto["precision"] = float(dto["precision"])
    else:
        # Locate named proof fields without assuming the transport nesting.
        def corrupt(raw):
            if not isinstance(raw, dict):
                return False
            targets = dict(wrong_area="patch_area_bounds", hidden_event="event_windows",
                           lost_receiver_plane="receiver_positive_bounds", wrong_owner="donor_omega")
            target = targets[damage]
            for key, value in raw.items():
                if key == target or (damage == "wrong_owner" and key == "donor_omega_rad_per_myr"):
                    if damage == "wrong_area":
                        value["lo"] = [0, 1]
                    elif damage == "hidden_event":
                        value.pop()
                    elif damage == "lost_receiver_plane":
                        value.pop()
                    else:
                        value[0] = 0.
                    return True
                if corrupt(value):
                    return True
            return False
        assert corrupt(dto), f"Missing donor proof field for {damage}"
    seal_keys = [key for key in dto if key.endswith("sha256")]
    assert len(seal_keys) == 1
    seal_key, = seal_keys
    del dto[seal_key]
    dto[seal_key] = hashlib.sha256(_encoded(dto)).hexdigest()
    with pytest.raises((TypeError, ValueError)):
        native_donor_patch_sweep_from_dict(dto)


@pytest.mark.parametrize("field", ["patch_x_bounds", "patch_area_bounds", "receiver_positive_bounds"])
def test_forged_frozen_proof_object_must_be_reproved(certificate, field):
    from tectonics.geometric_native_donor_sweep import validate_native_donor_patch_sweep
    from tectonics.geometric_native_pose import FractionInterval
    wrong = (FractionInterval(0), FractionInterval(0)) if field == "receiver_positive_bounds" else FractionInterval(0)
    forged = replace(certificate, **{field: wrong})
    with pytest.raises(ValueError):
        validate_native_donor_patch_sweep(forged)
