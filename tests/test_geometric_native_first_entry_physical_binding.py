"""Incoming material cannot inherit an old spring through common owners.

The historical control supplies an actual 6 -> 3 -> 2 receiving sequence.
These checks preserve the first entry and both different original spring
pairs, and compare its full-vector relative geometry to Decimal Rodrigues.
No additional force solve is performed beyond the shared physical fixture.
"""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import pytest

from tectonics.geometric_native_first_entry_material import (
    certify_native_first_entry_material_point,
    evaluate_native_first_entry_material_point,
)
from tectonics.geometric_native_first_entry_physical_binding import (
    certify_native_first_entry_physical_binding,
    native_first_entry_physical_binding_from_dict,
    native_first_entry_physical_binding_to_dict,
    validate_native_first_entry_physical_binding,
)
from tectonics.geometric_native_domain_journal import (
    evaluate_native_domain_journal_owner_point,
)
from tectonics.geometric_native_pose import FractionInterval as I
from tests.test_geometric_native_physical_pair_time import decimal, rotate


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "analysis/geometric_native_first_entry_validation/control_release_02"
LABELS = ("main_plane_reentry", "transverse_masked_main_entry", "ordinary_entry",
          "no_entry", "swept_only_owner_entry_exit", "empty_donor_four",
          "retained_overriding_material")


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _reseal(raw):
    raw["binding_sha256"] = hashlib.sha256(_encoded({key: value
        for key, value in raw.items() if key != "binding_sha256"})).hexdigest()
    return raw


@pytest.fixture(scope="module")
def entry_case():
    # Imported lazily so the shared helper owns its one actual force replay.
    from tests.test_geometric_native_physical_history import get_physical_case

    physical = get_physical_case()
    history, state = physical["binding"], physical["material_state"]
    saved = json.loads((CONTROL / "point_certificates.json").read_bytes())
    contexts = {row["label"]: certify_native_first_entry_material_point(state,
        row["certificate"]["original_parent_id"], tuple(Fraction(*pair)
            for pair in row["certificate"]["original_point"])) for row in saved}
    assert tuple(contexts) == LABELS
    bindings = {label: certify_native_first_entry_physical_binding(context, history)
                for label, context in contexts.items()}
    return dict(physical=physical, history=history, state=state,
                contexts=contexts, bindings=bindings)


def test_actual_six_to_three_keeps_different_original_physical_pairs(entry_case):
    binding = entry_case["bindings"]["transverse_masked_main_entry"]
    row, = binding.rows
    assert row.desired_fragment_pair == ("octant:6", "octant:3")
    assert row.original_donor_trace_pair == ("octant:6", "octant:2")
    assert row.original_receiver_trace_pair == ("octant:7", "octant:3")
    assert row.donor_trace_id != row.receiver_trace_id
    assert row.same_original_fragment_pair is False
    assert row.handover_status == binding.status == "unresolved_material_handover"
    assert binding.physical_pair_admission_authorized is False
    traces = {trace.trace_id: trace for trace in entry_case["history"].initial_mechanics.fault_traces.traces}
    donor, receiver = traces[row.donor_trace_id], traces[row.receiver_trace_id]
    assert (donor.subducting_plate, donor.overriding_plate) == (
        receiver.subducting_plate, receiver.overriding_plate) == (1, 0)


def test_same_fragment_pair_does_not_make_interior_donor_the_original_tip(entry_case):
    row, = entry_case["bindings"]["ordinary_entry"].rows
    assert row.desired_fragment_pair == row.original_donor_trace_pair == (
        "octant:6", "octant:2")
    assert row.original_receiver_trace_pair == row.original_donor_trace_pair
    assert row.same_original_fragment_pair is True
    assert row.donor_trace_id == row.receiver_trace_id
    assert row.donor_carrier_plane_value != 0
    assert row.incoming_original_label_on_donor_carrier is False
    assert row.handover_status == "unresolved_material_handover"


@pytest.mark.parametrize("label", LABELS)
def test_seven_contexts_keep_exact_first_windows_and_zero_physical_acceptance(entry_case, label):
    chronology = entry_case["contexts"][label]
    binding = entry_case["bindings"][label]
    expected = evaluate_native_first_entry_material_point(chronology, Fraction(1))
    assert binding.time_fraction_bounds == expected.first_entry_time_fraction
    assert binding.first_receiver_ancestry_status == expected.first_entry_receiver_ancestry_status
    assert binding.chronology is chronology
    assert binding.history is entry_case["history"]
    if binding.time_fraction_bounds is None:
        assert binding.status == "no_first_entry" and binding.rows == ()
    else:
        assert binding.time_fraction_bounds.lo < binding.time_fraction_bounds.hi
        assert binding.status == "unresolved_material_handover"
        assert tuple(row.receiver_fragment_id for row in binding.rows) == expected.receiver_original_fragment_ids
    raw = native_first_entry_physical_binding_to_dict(binding)
    for name in ("first_root_midpoint_used", "original_nodal_history_assigned_to_incoming_material",
                 "finite_material_reservoir_assigned_to_point", "nodal_work_distributed_to_incoming_material",
                 "candidate_mechanics_adopted", "physical_pair_admission_authorized"):
        assert raw[name] is False
    assert raw["accepted_loaded_native_intervals"] == raw["accepted_source_work_j"] == 0


def test_later_receiver_two_never_replaces_first_receiver_three(entry_case):
    chronology = entry_case["contexts"]["transverse_masked_main_entry"]
    binding = entry_case["bindings"]["transverse_masked_main_entry"]
    queries = json.loads((CONTROL / "material_point_queries.json").read_bytes())
    times = tuple(Fraction(*row["query"]["time_fraction"]) for row in queries
                  if row["label"] == "transverse_masked_main_entry")
    current_receivers = []
    for time in times:
        current = evaluate_native_domain_journal_owner_point(chronology.receiver_chronology, time)
        current_receivers.append(tuple(identity for identity, query in current.receiver_queries
                                       if query.occupancy == "inside"))
    assert ("octant:3",) in current_receivers
    assert ("octant:2",) in current_receivers
    assert tuple(row.receiver_fragment_id for row in binding.rows) == ("octant:3",)
    for time in times:
        point = evaluate_native_first_entry_material_point(chronology, time)
        assert point.receiver_original_fragment_ids == ("octant:3",)
        assert point.archive_debits <= 1
    assert native_first_entry_physical_binding_to_dict(binding)["first_root_midpoint_used"] is False


def test_exit_reentry_has_one_unchanged_root_and_original_node_references(entry_case):
    chronology = entry_case["contexts"]["main_plane_reentry"]
    binding = entry_case["bindings"]["main_plane_reentry"]
    first = chronology.first_entry_certificate.first_entry_time_fraction
    windows = [event.time_fraction for event in chronology.first_entry_certificate.plane_chronology.events
               if event.time_fraction.lo > 0]
    assert len(windows) >= 3
    for window in windows:
        point = evaluate_native_first_entry_material_point(chronology, window.hi)
        # The first root's upper enclosure endpoint is still an event window;
        # later exit/reentry windows retain the already allocated archive.
        assert point.archive_debits == (0 if window == first else 1)
        assert point.first_entry_time_fraction == first == binding.time_fraction_bounds
    nodes = entry_case["history"].original_points
    for row in binding.rows:
        assert row.original_donor_point_ids == tuple(point.point_id for point in nodes
                                                     if point.trace_id == row.donor_trace_id)
        assert row.original_receiver_point_ids == tuple(point.point_id for point in nodes
                                                        if point.trace_id == row.receiver_trace_id)
    assert len({point.point_id for point in nodes}) == len(nodes)


def _atan(value):
    """Independent Decimal alternating atan after reciprocal/half-angle reduction."""
    if value < 0:
        return -_atan(-value)
    if value > 1:
        pi = 16*_atan(Decimal(1)/5)-4*_atan(Decimal(1)/239)
        return pi/2-_atan(1/value)
    multiplier = 1
    while value > Decimal("0.1"):
        value /= 1+(1+value*value).sqrt()
        multiplier *= 2
    partial = term = value
    threshold = Decimal(10)**(-650)
    for degree in range(1, 2000):
        term *= -value*value
        increment = term/Decimal(2*degree+1)
        partial += increment
        if abs(increment) < threshold:
            return multiplier*partial
    raise AssertionError("Independent carrier-coordinate oracle did not converge")


def _contains(interval, value):
    return decimal(interval.lo) <= value <= decimal(interval.hi)


@pytest.mark.parametrize("label", ("main_plane_reentry", "transverse_masked_main_entry",
                                  "ordinary_entry", "swept_only_owner_entry_exit"))
def test_relative_entry_full_omega_and_carrier_angle_have_independent_decimal_bounds(entry_case, label):
    binding, history = entry_case["bindings"][label], entry_case["history"]
    point = binding.chronology.original_point
    traces = {trace.trace_id: trace for trace in history.initial_mechanics.fault_traces.traces}
    with localcontext() as context:
        context.prec = 700
        for row in binding.rows:
            donor, receiver = traces[row.donor_trace_id], traces[row.receiver_trace_id]
            for time in (binding.time_fraction_bounds.lo, binding.time_fraction_bounds.midpoint,
                         binding.time_fraction_bounds.hi):
                axes = {owner: tuple(decimal(component)*decimal(history.dt_myr)*decimal(time)
                    for component in omega) for owner, omega in history.omega_records}
                world = rotate(tuple(map(decimal, point)), axes[donor.subducting_plate])
                relative = rotate(world, tuple(-value for value in axes[receiver.overriding_plate]))
                assert all(_contains(interval, value) for interval, value
                           in zip(row.relative_entry_coordinates, relative))
                assert row.receiver_native_angle_bounds is not None
                start = tuple(Decimal.from_float(value) for value in receiver.start)
                end = tuple(Decimal.from_float(value) for value in receiver.end)
                cosine = sum(a*b for a, b in zip(start, relative))
                sine = sum(a*b for a, b in zip(end, relative))
                assert cosine > 0 and sine >= 0
                assert _contains(row.receiver_native_angle_bounds, _atan(sine/cosine))
    assert binding.time_fraction_bounds.lo < binding.time_fraction_bounds.hi


@pytest.mark.parametrize("label", LABELS)
def test_binding_roundtrip_uses_actual_external_history_and_context(entry_case, label):
    value = entry_case["bindings"][label]
    raw = native_first_entry_physical_binding_to_dict(value)
    replayed = native_first_entry_physical_binding_from_dict(deepcopy(raw),
        chronology=value.chronology, history=entry_case["history"])
    assert _encoded(raw) == _encoded(native_first_entry_physical_binding_to_dict(replayed))


@pytest.mark.parametrize("field,value", (
    ("first_root_midpoint_used", True),
    ("original_nodal_history_assigned_to_incoming_material", True),
    ("nodal_work_distributed_to_incoming_material", True),
    ("finite_material_reservoir_assigned_to_point", True),
    ("candidate_mechanics_adopted", True),
    ("physical_pair_admission_authorized", True),
    ("accepted_loaded_native_intervals", 1),
    ("accepted_source_work_j", 1),
    ("physical_pair_admission_authorized", 0),
))
def test_rehashed_dto_cannot_promote_material_closure_or_allocate_work(entry_case, field, value):
    binding = entry_case["bindings"]["transverse_masked_main_entry"]
    raw = native_first_entry_physical_binding_to_dict(binding)
    raw[field] = value
    with pytest.raises(ValueError):
        native_first_entry_physical_binding_from_dict(_reseal(raw),
            chronology=binding.chronology, history=entry_case["history"])


@pytest.mark.parametrize("damage", ("receiver_pair", "trace_swap", "off_carrier", "root_midpoint",
                                  "source_point", "tuple_alias", "float_rational"))
def test_rehashed_dto_cannot_relabel_old_banks_or_replace_exact_event(entry_case, damage):
    binding = entry_case["bindings"]["transverse_masked_main_entry"]
    raw = native_first_entry_physical_binding_to_dict(binding)
    row = raw["rows"][0]
    if damage == "receiver_pair":
        row["original_receiver_trace_pair"][0] = "octant:6"
    elif damage == "trace_swap":
        row["receiver_trace_id"] = row["donor_trace_id"]
    elif damage == "off_carrier":
        row["incoming_original_label_on_donor_carrier"] = True
        row["donor_carrier_plane_value"] = [0, 1]
    elif damage == "root_midpoint":
        midpoint = binding.time_fraction_bounds.midpoint
        raw["time_fraction_bounds"] = dict(lo=[midpoint.numerator, midpoint.denominator],
                                           hi=[midpoint.numerator, midpoint.denominator])
    elif damage == "source_point":
        raw["original_point"][0] = [0, 1]
    elif damage == "tuple_alias":
        raw["rows"] = tuple(raw["rows"])
    else:
        raw["original_point"][2][1] = float(raw["original_point"][2][1])
    with pytest.raises(ValueError):
        native_first_entry_physical_binding_from_dict(_reseal(raw),
            chronology=binding.chronology, history=entry_case["history"])


def test_dto_cannot_borrow_another_material_trajectory(entry_case):
    value = entry_case["bindings"]["transverse_masked_main_entry"]
    raw = native_first_entry_physical_binding_to_dict(value)
    with pytest.raises(ValueError):
        native_first_entry_physical_binding_from_dict(raw,
            chronology=entry_case["contexts"]["ordinary_entry"], history=entry_case["history"])


@pytest.mark.parametrize("damage", ("row_boolean_alias", "mutable_rows", "work_alias", "reset_preload"))
def test_warm_typed_cache_rejects_history_and_material_aliases(entry_case, damage):
    value = entry_case["bindings"]["transverse_masked_main_entry"]
    if damage == "row_boolean_alias":
        tampered = replace(value, rows=(replace(value.rows[0], same_original_fragment_pair=0),))
    elif damage == "mutable_rows":
        tampered = replace(value, rows=list(value.rows))
    elif damage == "work_alias":
        history = replace(value.history, before_work_ledgers=(("cumulative_source_work_j", Fraction(0)),))
        tampered = replace(value, history=history)
    else:
        node = value.history.quadrature_histories[0]
        assert node.previous_state.gap_free_m != 0
        reset = replace(node.previous_state, gap_free_m=0.0)
        history = replace(value.history, quadrature_histories=(replace(node, previous_state=reset),
            *value.history.quadrature_histories[1:]))
        tampered = replace(value, history=history)
    with pytest.raises(ValueError):
        validate_native_first_entry_physical_binding(tampered)


def test_binding_observation_preserves_existing_clocks_memory_and_endpoint_work(entry_case):
    history = entry_case["history"]
    before, candidate = history.accepted_before, history.actual_candidate
    old = before.states, before.spatial_memory, before.free_anchor_fields, before.elapsed_time_s
    candidate_old = candidate.states, candidate.spatial_step, candidate.candidate_diagnostics
    work = history.before_work_ledgers, history.unaccepted_candidate_work_ledgers
    for value in entry_case["bindings"].values():
        validate_native_first_entry_physical_binding(value)
    assert (before.states, before.spatial_memory, before.free_anchor_fields, before.elapsed_time_s) == old
    assert (candidate.states, candidate.spatial_step, candidate.candidate_diagnostics) == candidate_old
    assert (history.before_work_ledgers, history.unaccepted_candidate_work_ledgers) == work
    assert before.cumulative_source_work_j == 0.0
    assert dict(history.unaccepted_candidate_work_ledgers)["source_work_j"] > 0.0
    with pytest.raises(FrozenInstanceError):
        entry_case["bindings"]["ordinary_entry"].status = "admitted"
