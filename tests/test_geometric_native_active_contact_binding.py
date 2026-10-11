"""Actual receiver labels select a unique, frozen bank path at proved seams."""
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
import hashlib
import json

import pytest

from tectonics.geometric_native_contact_reference_transport import (
    ContactHistoryTransportLaw, TRANSPORT_LAW_MODEL,
)
from tectonics.geometric_native_physical_history import native_physical_history_binding_to_dict
from tectonics.geometric_native_pose import FractionInterval as I
from tests.test_geometric_native_physical_history import get_physical_case


F = Fraction
TINY_RECEIVER_THREE = (F(0), F(1), F(1, 10**55))
ORDINARY_RECEIVER_TWO = (F(0), F(1), F(-1))


@lru_cache(maxsize=1)
def get_active_binding_case():
    from tectonics.geometric_native_active_contact_binding import certify_native_active_contact_binding
    original = get_physical_case()
    history = original["binding"]
    source = native_physical_history_binding_to_dict(history)
    law = ContactHistoryTransportLaw(TRANSPORT_LAW_MODEL,
        "Explicit receiver-attached complete configuration; bank changes only at proved quarter seams")
    paths = {
        "two_changes": certify_native_active_contact_binding(history, "octant:3",
            TINY_RECEIVER_THREE, law=law, expected_source_history_dto=source),
        "unchanged": certify_native_active_contact_binding(history, "octant:2",
            ORDINARY_RECEIVER_TWO, law=law, expected_source_history_dto=source),
    }
    return dict(original=original, history=history, source=source, law=law, paths=paths)


@pytest.fixture(scope="module")
def active_binding_case():
    return get_active_binding_case()


def decimal(value):
    if type(value) is Decimal:
        return value
    if type(value) is float:
        value = F.from_float(value)
    value = F(value)
    return Decimal(value.numerator)/Decimal(value.denominator)


def cross(first, second):
    return tuple(first[j]*second[k]-first[k]*second[j]
        for j, k in ((1, 2), (2, 0), (0, 1)))


def dot(first, second):
    return sum((a*b for a, b in zip(first, second)), Decimal(0))


def sine_cosine(angle):
    sine, cosine = angle, Decimal(1)
    sine_term, cosine_term = angle, Decimal(1)
    for index in range(1, 3000):
        sine_term *= -angle*angle/Decimal((2*index)*(2*index+1))
        cosine_term *= -angle*angle/Decimal((2*index-1)*(2*index))
        sine += sine_term
        cosine += cosine_term
        if max(abs(sine_term), abs(cosine_term)) < Decimal("1e-590"):
            return sine, cosine
    raise AssertionError("Independent complete rotation series did not converge")


def rotate(vector, axis, time):
    """Independent Rodrigues map, retaining every literal Euler component."""
    size = dot(axis, axis).sqrt()
    if not size:
        return vector
    sine, cosine = sine_cosine(size*time)
    first = cross(axis, vector)
    second = cross(axis, first)
    return tuple(v+sine/size*a+(1-cosine)/(size*size)*b
        for v, a, b in zip(vector, first, second))


def unit_direction(exact):
    vector = tuple(map(decimal, exact))
    size = dot(vector, vector).sqrt()
    return tuple(value/size for value in vector)


def path_axes(path):
    bank = next(row for row in path.history.material_state.trace_bindings
        if row.trace_id == path.receiver_trace_id)
    duration = decimal(path.history.dt_myr)
    axes = {owner: tuple(decimal(value)*duration for value in omega)
        for owner, omega in path.history.omega_records}
    return axes[bank.subducting_plate], axes[bank.overriding_plate]


def independent_receiver_pullback(path, time):
    donor, receiver = path_axes(path)
    point = unit_direction(path.receiver_reference_coordinates)
    world = rotate(point, receiver, time)
    value = rotate(world, donor, -time)
    difference = tuple(a-b for a, b in zip(receiver, donor))
    derivative = rotate(cross(difference, world), donor, -time)
    return value, derivative


def independent_event_root(path, event):
    """Newton refines an already certified root solely for the test oracle.

    A midpoint is only an initial numerical guess here. Production attachment
    and material identity must remain expressions at the entire implicit root.
    """
    with localcontext() as context:
        context.prec = 620
        time = decimal((event.time_fraction_bounds.lo+event.time_fraction_bounds.hi)/2)
        component = event.seam_coordinate_index
        for _ in range(16):
            point, derivative = independent_receiver_pullback(path, time)
            change = point[component]/derivative[component]
            time -= change
            if abs(change) < Decimal("1e-580"):
                assert decimal(event.time_fraction_bounds.lo) < time < decimal(event.time_fraction_bounds.hi)
                return time, independent_receiver_pullback(path, time)[0]
        raise AssertionError("Independent seam root did not converge")


def reseal(raw, key):
    raw[key] = hashlib.sha256(json.dumps({name: item for name, item in raw.items() if name != key},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def restore_arguments(case, path):
    return dict(expected_history=case["history"], expected_law=case["law"],
        expected_source_history_dto=case["source"],
        expected_receiver_fragment_id=path.receiver_fragment_id,
        expected_receiver_reference_coordinates=path.receiver_reference_coordinates)


def test_actual_tiny_receiver_label_changes_seven_to_six_and_back_to_new_seven_material(active_binding_case):
    path = active_binding_case["paths"]["two_changes"]
    assert path.receiver_fragment_id == "octant:3"
    assert path.receiver_reference_coordinates == TINY_RECEIVER_THREE
    assert [(event.old_donor_fragment_id, event.new_donor_fragment_id)
        for event in path.events] == [("octant:7", "octant:6"), ("octant:6", "octant:7")]
    first, second = path.events
    assert 0 < first.time_fraction_bounds.lo < first.time_fraction_bounds.hi < second.time_fraction_bounds.lo
    assert second.time_fraction_bounds.hi < 1
    assert [segment.donor_fragment_id for segment in path.segments] == ["octant:7", "octant:6", "octant:7"]
    assert first.donor_reference_coordinates_bounds[0].hi < second.donor_reference_coordinates_bounds[0].lo
    for event in path.events:
        assert event.seam_coordinate_index == 2
        assert event.donor_reference_coordinates_bounds[2] == I(F(0))
        assert event.donor_reference_coordinates_bounds[0].lo > 0
        assert event.fresh_owner_depth_derivative_bounds.lo > 0
    assert len({segment.active_binding_id for segment in path.segments}) == 3


def test_frozen_segment_labels_are_the_initial_label_or_the_proved_switch_expression(active_binding_case):
    path = active_binding_case["paths"]["two_changes"]
    assert path.segments[0].start_time_fraction_bounds == I(F(0))
    assert path.segments[-1].end_time_fraction_bounds == I(F(1))
    assert path.segments[0].anchor_event_sequence is None
    assert path.segments[0].frozen_reference_coordinates_bounds == path.receiver_unit_direction_bounds
    for before, after, event in zip(path.segments, path.segments[1:], path.events):
        assert before.end_time_fraction_bounds == after.start_time_fraction_bounds == event.time_fraction_bounds
        assert after.anchor_event_sequence == event.sequence
        assert after.frozen_reference_coordinates_bounds == event.donor_reference_coordinates_bounds
    assert path.segments[0].frozen_reference_coordinates_bounds != path.segments[1].frozen_reference_coordinates_bounds
    assert path.segments[1].frozen_reference_coordinates_bounds != path.segments[2].frozen_reference_coordinates_bounds


def test_complete_literal_euler_motion_independently_brackets_both_tiny_seams(active_binding_case):
    path = active_binding_case["paths"]["two_changes"]
    history = active_binding_case["history"]
    for _, certificate in path.donor_chronologies:
        assert all(value != 0 for omega in (certificate.donor_omega, certificate.receiver_omega) for value in omega)
        assert F.from_float(certificate.dt_myr) == history.dt_myr
    with localcontext() as context:
        context.prec = 620
        for index, event in enumerate(path.events):
            lower = independent_receiver_pullback(path, decimal(event.time_fraction_bounds.lo))[0][2]
            upper = independent_receiver_pullback(path, decimal(event.time_fraction_bounds.hi))[0][2]
            assert lower > 0 > upper if index == 0 else lower < 0 < upper
            _, original = independent_event_root(path, event)
            for axis in (0, 1):
                bounds = event.donor_reference_coordinates_bounds[axis]
                assert bounds.lo <= F(original[axis]) <= bounds.hi
            assert abs(original[2]) < Decimal("1e-580")


def test_ordinary_receiver_keeps_one_original_binding_despite_continuously_entering_material(active_binding_case):
    path = active_binding_case["paths"]["unchanged"]
    assert path.events == ()
    segment, = path.segments
    assert segment.donor_fragment_id == "octant:6"
    assert segment.start_time_fraction_bounds == I(F(0))
    assert segment.end_time_fraction_bounds == I(F(1))
    assert segment.anchor_event_sequence is None
    assert segment.frozen_reference_coordinates_bounds == path.receiver_unit_direction_bounds


@pytest.mark.parametrize("label", ["two_changes", "unchanged"])
def test_active_binding_restart_replays_all_chronology_from_external_receiver_and_source(active_binding_case, label):
    from tectonics.geometric_native_active_contact_binding import (
        native_active_contact_binding_from_dict, native_active_contact_binding_to_dict,
    )
    path = active_binding_case["paths"][label]
    raw = native_active_contact_binding_to_dict(path)
    restored = native_active_contact_binding_from_dict(deepcopy(raw), **restore_arguments(active_binding_case, path))
    assert native_active_contact_binding_to_dict(restored) == raw
    changed = restore_arguments(active_binding_case, path)
    changed["expected_receiver_reference_coordinates"] = (F(0), F(1), F(1, 10**54))
    with pytest.raises(ValueError):
        native_active_contact_binding_from_dict(deepcopy(raw), **changed)


def test_binding_path_leaves_original_source_force_memory_and_material_records_unchanged(active_binding_case):
    history = active_binding_case["history"]
    assert native_physical_history_binding_to_dict(history) == active_binding_case["source"]
    assert history.accepted_before.elapsed_time_s == 0
    assert history.accepted_before.cumulative_source_work_j == 0
    assert history.accepted_before.receipts == ()


@pytest.mark.parametrize("mutation", ["incoming_bank", "frozen_label", "root_clock", "event_identity", "force_adoption"])
def test_resealed_binding_cannot_replace_a_causal_bank_label_or_clock(active_binding_case, mutation):
    from tectonics.geometric_native_active_contact_binding import (
        native_active_contact_binding_from_dict, native_active_contact_binding_to_dict,
    )
    path = active_binding_case["paths"]["two_changes"]
    raw = deepcopy(native_active_contact_binding_to_dict(path))
    if mutation == "incoming_bank":
        raw["events"][1]["new_donor_fragment_id"] = "octant:6"
    elif mutation == "frozen_label":
        raw["events"][1]["donor_reference_coordinates_bounds"] = deepcopy(raw["events"][0]["donor_reference_coordinates_bounds"])
    elif mutation == "root_clock":
        raw["events"][0]["time_fraction_bounds"] = deepcopy(raw["events"][1]["time_fraction_bounds"])
    elif mutation == "event_identity":
        raw["events"][1]["consumed_binding_id"] = raw["events"][0]["consumed_binding_id"]
    else:
        raw["physical_pair_admission_authorized"] = True
        raw["accepted_native_force_intervals"] = 1
    reseal(raw, "path_sha256")
    with pytest.raises(ValueError):
        native_active_contact_binding_from_dict(raw, **restore_arguments(active_binding_case, path))


@pytest.mark.parametrize("coordinates", [(F(0), F(1), F(0)), (F(0), True, F(1)),
    (0., 1., 1.), (F(1), F(1), F(1)), (F(0), F(-1), F(1))])
def test_unproved_endpoint_or_foreign_receiver_label_is_refused(active_binding_case, coordinates):
    from tectonics.geometric_native_active_contact_binding import certify_native_active_contact_binding
    with pytest.raises(ValueError):
        certify_native_active_contact_binding(active_binding_case["history"], "octant:3", coordinates,
            law=active_binding_case["law"], expected_source_history_dto=active_binding_case["source"])


def test_binding_path_reports_its_actual_single_carrier_scope(active_binding_case):
    from tectonics.geometric_native_active_contact_binding import native_active_contact_binding_to_dict
    path = active_binding_case["paths"]["two_changes"]
    raw = native_active_contact_binding_to_dict(path)
    assert raw["preceding_binding_consumed_once"]
    assert not raw["root_midpoint_selected"]
    assert not raw["receiving_carrier_reprojected_after_switch"]
    assert not raw["interface_or_work_weight_assigned_to_point"]
    assert not raw["all_receiver_angles_scheduled"]
    assert not raw["physical_pair_admission_authorized"]
    assert raw["accepted_native_force_intervals"] == 0
    assert all(event.first_entry_proved and not event.material_reentry for event in path.events)
