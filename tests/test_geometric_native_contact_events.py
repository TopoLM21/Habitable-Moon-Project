"""Complete native receiver-arc events and strict diagnostic replay."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import hashlib
import json

import pytest

from analysis.geometric_native_contact_event_validation.run_event_control import (
    actual_arc_case, DEFAULT_PROBE, encoded,
)
from tectonics.geometric_native_polygons import NativePlane, NativeRay


@pytest.fixture(scope="module")
def actual():
    raw = json.loads(DEFAULT_PROBE.read_bytes())
    return raw, actual_arc_case(raw)


def test_actual_control_retains_full_native_arc_and_every_owner_component(actual):
    raw, (arc, plane, donor, receiver, duration, evidence) = actual
    assert arc.start.kind == arc.end.kind == "cross"
    assert 1.5 < arc.length_radians < 1.6
    assert plane == NativePlane((1, 0, 0))
    assert donor == tuple(raw["omega"]["1"])
    assert receiver == tuple(raw["omega"]["0"])
    assert all(value != 0. for omega in (donor, receiver) for value in omega)
    assert duration == raw["admission"]["dt_myr"]
    assert evidence["accepted_loaded_native_intervals"] == 0
    assert evidence["input_unchanged"] is True
    assert evidence["receiver_arc"]["end"]["planes"][0]["rotations"][0]["omega_rad_per_myr"] == list(receiver)


@pytest.mark.parametrize("damage", ["missing_overlap", "changed_source_owner", "changed_duration",
                                  "accepted_step", "mutable_state"])
def test_saved_actual_control_rejects_false_lineage_or_force_acceptance(actual, damage):
    original, _ = actual
    raw = deepcopy(original)
    if damage == "missing_overlap":
        raw["admission"]["actual_overlap_assessments"] = []
    elif damage == "changed_source_owner":
        raw["omega"]["0"][0] = 0.
    elif damage == "changed_duration":
        raw["admission"]["dt_myr"] *= 2
    elif damage == "accepted_step":
        raw["accepted_loaded_native_intervals"] = 1
    else:
        raw["input_unchanged"] = False
    with pytest.raises(ValueError):
        actual_arc_case(raw)


def test_actual_control_returns_independent_evidence_copy(actual):
    raw, _ = actual
    before = encoded(raw)
    *_, evidence = actual_arc_case(raw)
    evidence["receiver_arc"]["end"]["planes"].clear()
    assert encoded(raw) == before


@pytest.fixture(scope="module")
def certificate(actual):
    from tectonics.geometric_native_contact_events import certify_native_contact_arc_events
    _, (arc, plane, donor, receiver, duration, _) = actual
    return certify_native_contact_arc_events(plane, arc, donor, receiver, duration)


def test_complete_actual_arc_proves_strict_turn_and_positive_recrossing_band(certificate):
    assert certificate.topology == "initial_exit_later_entry_positive_material_band"
    turning = certificate.turning_fraction
    assert Fraction(1, 10**12) < turning.lo < turning.hi < Fraction(1, 10**10)
    width = certificate.recrossing_ratio_width
    assert Fraction(1, 10**55) < width.lo < width.hi < Fraction(1, 10**53)
    assert certificate.maximum_ratio.lo > certificate.initial_ratio.hi
    # Neither the positive material band nor its critical time can be replaced
    # by a binary64 endpoint or discarded below an absolute time/area floor.
    assert width.lo > 0
    assert turning.lo > 0


def test_turning_time_uses_the_same_original_physical_duration(actual, certificate):
    _, (_, _, _, _, duration, _) = actual
    expected = certificate.turning_fraction * Fraction(duration)
    assert certificate.turning_time_myr.lo == expected.lo
    assert certificate.turning_time_myr.hi == expected.hi


def test_native_event_certificate_is_immutable(certificate):
    with pytest.raises(AttributeError):
        certificate.topology = "always_converging"


def test_event_diagnostic_reloads_by_full_native_proof(actual, certificate):
    from tectonics.geometric_native_contact_events import (
        native_contact_arc_events_from_dict, native_contact_arc_events_to_dict,
    )
    dto = native_contact_arc_events_to_dict(certificate)
    copied = deepcopy(dto)
    before = encoded(copied)
    loaded = native_contact_arc_events_from_dict(copied)
    assert encoded(native_contact_arc_events_to_dict(loaded)) == encoded(dto)
    assert encoded(copied) == before
    assert actual[0]["accepted_loaded_native_intervals"] == 0


@pytest.mark.parametrize("damage", ["additional_root", "wrong_topology", "not_a_mapping"])
def test_event_diagnostic_rejects_false_schema_and_false_topology(certificate, damage):
    from tectonics.geometric_native_contact_events import (
        native_contact_arc_events_from_dict, native_contact_arc_events_to_dict,
    )
    dto = deepcopy(native_contact_arc_events_to_dict(certificate))
    if damage == "additional_root":
        dto["accepted_loaded_native_intervals"] = 1
    elif damage == "wrong_topology":
        # The recursive lookup remains independent of DTO nesting choices.
        def replace_topology(raw):
            if isinstance(raw, dict):
                for key, value in raw.items():
                    if key == "topology":
                        raw[key] = "always_converging"
                        return True
                    if replace_topology(value):
                        return True
            return False
        assert replace_topology(dto)
    else:
        dto = []
    with pytest.raises((TypeError, ValueError)):
        native_contact_arc_events_from_dict(dto)


@pytest.mark.parametrize("damage", ["zero_motion", "projected_offaxis", "reversed_donor_plane"])
def test_degenerate_or_changed_motion_cannot_claim_the_actual_positive_band(actual, damage):
    from tectonics.geometric_native_contact_events import (
        UnresolvedNativeContactEvents, certify_native_contact_arc_events,
    )
    _, (arc, plane, donor, receiver, duration, _) = actual
    if damage == "zero_motion":
        donor = receiver = (0., 0., 0.)
    elif damage == "projected_offaxis":
        # A rotation-axis projection changes the actual physical trajectory.
        donor, receiver = (0., 0., donor[2]), (0., 0., receiver[2])
    else:
        plane = NativePlane((-1, 0, 0))
    with pytest.raises(UnresolvedNativeContactEvents):
        certify_native_contact_arc_events(plane, arc, donor, receiver, duration)


@pytest.mark.parametrize("duration", [0., -1., True, float("nan"), float("inf"), "1"])
def test_event_sweep_rejects_nonphysical_or_untyped_time(actual, duration):
    from tectonics.geometric_native_contact_events import certify_native_contact_arc_events
    _, (arc, plane, donor, receiver, _, _) = actual
    with pytest.raises((TypeError, ValueError)):
        certify_native_contact_arc_events(plane, arc, donor, receiver, duration)


@pytest.mark.parametrize("omega", [(0., 1.), (0., True, 1.), (0., float("nan"), 1.),
                                 (0., float("inf"), 1.), "0,0,1"])
def test_event_sweep_requires_all_three_finite_explicit_owner_components(actual, omega):
    from tectonics.geometric_native_contact_events import certify_native_contact_arc_events
    _, (arc, plane, _, receiver, duration, _) = actual
    with pytest.raises((TypeError, ValueError)):
        certify_native_contact_arc_events(plane, arc, omega, receiver, duration)


def test_exact_endpoint_material_keeps_both_boundary_events(certificate):
    from tectonics.geometric_native_contact_events import isolate_native_contact_ratio_events
    events = isolate_native_contact_ratio_events(certificate, Fraction(0))
    assert [event.kind for event in events] == ["exit", "entry"]
    assert events[0].time_fraction.lo == events[0].time_fraction.hi == 0
    assert events[1].time_fraction.lo == events[1].time_fraction.hi == 1
    assert events[0].time_myr.lo == 0
    assert events[1].time_myr.lo == events[1].time_myr.hi == Fraction(certificate.dt_myr)


def test_positive_interior_material_exits_then_reenters_at_actual_times(certificate):
    from tectonics.geometric_native_contact_events import isolate_native_contact_ratio_events
    ratio = (certificate.initial_ratio.hi + certificate.maximum_ratio.lo) / 2
    events = isolate_native_contact_ratio_events(certificate, ratio)
    assert [event.kind for event in events] == ["entry", "exit", "entry"]
    assert events[0].time_fraction.lo == events[0].time_fraction.hi == 0
    first, second = events[1:]
    assert 0 < first.time_fraction.lo < first.time_fraction.hi < certificate.turning_fraction.lo
    assert certificate.turning_fraction.hi < second.time_fraction.lo < second.time_fraction.hi < 1
    for event in events:
        expected = event.time_fraction * Fraction(certificate.dt_myr)
        assert event.time_myr.lo == expected.lo and event.time_myr.hi == expected.hi


@pytest.mark.parametrize("choice", ["later_entry", "always_inside", "threshold"])
def test_other_arc_material_classes_or_explicit_threshold_refusal(certificate, choice):
    from tectonics.geometric_native_contact_events import (
        isolate_native_contact_ratio_events, UnresolvedNativeContactEvents,
    )
    if choice == "threshold":
        with pytest.raises(UnresolvedNativeContactEvents):
            isolate_native_contact_ratio_events(certificate, certificate.initial_ratio.midpoint)
        return
    ratio = certificate.initial_ratio.lo/2 if choice == "later_entry" else 2*certificate.maximum_ratio.hi
    events = isolate_native_contact_ratio_events(certificate, ratio)
    if choice == "later_entry":
        assert [event.kind for event in events] == ["exit", "entry"]
        assert events[0].time_fraction.lo == 0
        assert certificate.turning_fraction.hi < events[1].time_fraction.lo < events[1].time_fraction.hi < 1
    else:
        assert len(events) == 1 and events[0].kind == "entry"
        assert events[0].time_fraction.lo == events[0].time_fraction.hi == 0


def test_resealed_false_positive_band_is_rejected_by_analytic_replay(certificate):
    from tectonics.geometric_native_contact_events import (
        native_contact_arc_events_from_dict, native_contact_arc_events_to_dict,
    )
    dto = native_contact_arc_events_to_dict(certificate)
    dto["certificate"]["recrossing_ratio_width"]["lo"] = [0, 1]
    del dto["certificate_sha256"]
    dto["certificate_sha256"] = hashlib.sha256(encoded(dto)).hexdigest()
    with pytest.raises(ValueError, match="replay"):
        native_contact_arc_events_from_dict(dto)


def test_forged_frozen_certificate_cannot_authorize_an_event_query(certificate):
    from tectonics.geometric_native_contact_events import isolate_native_contact_ratio_events
    forged = replace(certificate, topology="always_converging")
    with pytest.raises(ValueError, match="replay"):
        isolate_native_contact_ratio_events(forged, Fraction(0))


@pytest.mark.parametrize("damage", ["float_precision", "bool_order", "bool_plane", "float_plane",
                                  "admission_claim", "aliased_admission", "unknown_trajectory"])
def test_resealed_native_event_dto_keeps_explicit_types_and_authority(certificate, damage):
    from tectonics.geometric_native_contact_events import (
        native_contact_arc_events_from_dict, native_contact_arc_events_to_dict,
    )
    dto = deepcopy(native_contact_arc_events_to_dict(certificate))
    if damage == "float_precision":
        dto["precision"] = float(dto["precision"])
    elif damage == "bool_order":
        dto["order"] = True
    elif damage == "bool_plane":
        dto["trajectory"]["donor_plane_before"]["normal"][0] = True
    elif damage == "float_plane":
        dto["trajectory"]["donor_plane_before"]["normal"][0] = 1.
    elif damage == "admission_claim":
        dto["admission_authorized"] = True
    elif damage == "aliased_admission":
        dto["admission_authorized"] = 0
    else:
        dto["trajectory"]["projected_omega"] = [0., 0., 1.]
    del dto["certificate_sha256"]
    dto["certificate_sha256"] = hashlib.sha256(encoded(dto)).hexdigest()
    with pytest.raises((TypeError, ValueError)):
        native_contact_arc_events_from_dict(dto)


@pytest.mark.parametrize("damage", ["zero_arc", "wrong_length", "reversed_orientation", "off_plane"])
def test_event_proof_requires_the_complete_authenticated_positive_receiver_arc(actual, damage):
    from tectonics.geometric_native_contact_events import certify_native_contact_arc_events
    _, (arc, plane, donor, receiver, duration, _) = actual
    if damage == "zero_arc":
        arc = replace(arc, end=arc.start)
    elif damage == "wrong_length":
        arc = replace(arc, length_radians=1.)
    elif damage == "reversed_orientation":
        arc = replace(arc, start=arc.end, end=arc.start)
    else:
        arc = replace(arc, end=NativeRay("seed", (1, 0, 0)))
    with pytest.raises(ValueError):
        certify_native_contact_arc_events(plane, arc, donor, receiver, duration)
