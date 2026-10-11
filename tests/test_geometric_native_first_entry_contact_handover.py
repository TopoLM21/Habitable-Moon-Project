"""Authentic first-entry field handover with receiver-conditioned history."""
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
from tectonics.geometric_native_first_entry_material import certify_native_first_entry_material_point
from tectonics.geometric_native_first_entry_physical_binding import (
    certify_native_first_entry_physical_binding,
)
from tectonics.geometric_native_physical_history import native_physical_history_binding_to_dict
from tectonics.geometric_native_pose import FractionInterval as I
from tests.test_geometric_native_physical_history import get_physical_case
from tests.test_geometric_native_contact_field_history import independent_field, enclosed
from tests.test_geometric_native_physical_pair_time import decimal


F = Fraction
LABELS = ("main_plane_reentry", "transverse_masked_main_entry", "ordinary_entry",
          "no_entry", "swept_only_owner_entry_exit", "empty_donor_four",
          "retained_overriding_material")
ENTRY_LABELS = ("main_plane_reentry", "transverse_masked_main_entry", "ordinary_entry",
                "swept_only_owner_entry_exit")


@lru_cache(maxsize=1)
def get_first_entry_handover_case():
    """One genuine force source and one replay of each original root context."""
    from tectonics.geometric_native_first_entry_contact_handover import (
        certify_native_first_entry_contact_handover,
    )
    case = get_physical_case()
    history = case["binding"]
    source = native_physical_history_binding_to_dict(history)
    contexts = {row["label"]: certify_native_first_entry_material_point(case["material_state"],
        row["certificate"]["original_parent_id"], tuple(F(*pair)
            for pair in row["certificate"]["original_point"])) for row in case["point_certificates"]}
    assert tuple(contexts) == LABELS
    bindings = {label: certify_native_first_entry_physical_binding(context, history)
        for label, context in contexts.items()}
    law = ContactHistoryTransportLaw(TRANSPORT_LAW_MODEL,
        "Explicit receiver-attached complete-configuration transport")
    handovers = {label: certify_native_first_entry_contact_handover(binding,
        law=law, expected_source_history_dto=source) for label, binding in bindings.items()}
    return dict(case=case, history=history, source=source, contexts=contexts,
        bindings=bindings, law=law, handovers=handovers)


@pytest.fixture(scope="module")
def entry_handover_case():
    return get_first_entry_handover_case()


@pytest.mark.parametrize("label", LABELS)
def test_all_actual_contexts_commit_only_the_proved_first_receiver(entry_handover_case, label):
    case = entry_handover_case
    value, old = case["handovers"][label], case["bindings"][label]
    assert value.binding is old
    assert value.additional_handover_work_j == value.additional_handover_dissipation_j == F(0)
    assert not value.physical_pair_admission_authorized
    assert not value.candidate_mechanics_adopted
    if label not in ENTRY_LABELS:
        assert value.status == "no_first_entry"
        assert value.attachments == ()
        return
    assert value.status == "first_entry_field_attachment_committed"
    attachment, = value.attachments
    row, = old.rows
    assert attachment.attachment_committed
    assert attachment.desired_fragment_pair == row.desired_fragment_pair
    assert attachment.original_donor_trace_id == row.donor_trace_id
    assert attachment.original_receiver_trace_id == row.receiver_trace_id
    assert attachment.source_field_query.trace_id == row.receiver_trace_id
    assert attachment.source_field_query.time_fraction_bounds == old.time_fraction_bounds
    assert attachment.interface_measure.time_fraction_bounds == old.time_fraction_bounds
    assert attachment.transport_family.revision == 1
    assert len(attachment.transport_family.consumed_binding_ids) == 1
    assert attachment.transport_family.active_banks.sub_fragment_id == old.chronology.original_parent_id
    assert attachment.transport_family.active_banks.over_fragment_id == row.receiver_fragment_id
    assert attachment.finite_point_area_m2 is attachment.finite_point_work_j is None
    assert attachment.source_field_query.cumulative_work_density_j_m2 is None


def test_actual_six_to_three_uses_receiver_seven_history_and_incoming_six_beam(entry_handover_case):
    case = entry_handover_case
    old = case["bindings"]["transverse_masked_main_entry"]
    value = case["handovers"]["transverse_masked_main_entry"]
    attachment, = value.attachments
    assert attachment.desired_fragment_pair == ("octant:6", "octant:3")
    assert attachment.original_donor_trace_pair == ("octant:6", "octant:2")
    assert attachment.original_receiver_trace_pair == ("octant:7", "octant:3")
    assert attachment.original_donor_trace_id != attachment.original_receiver_trace_id
    assert attachment.source_field_query.trace_id == attachment.original_receiver_trace_id
    assert attachment.sub_material_chart.trace_id == attachment.original_donor_trace_id
    assert attachment.sub_material_chart.reference_coordinates == old.chronology.original_point
    assert attachment.sub_material_chart.shape.hi < 1
    assert attachment.transport_family.original_banks.sub_fragment_id == "octant:7"
    assert attachment.transport_family.active_banks.sub_fragment_id == "octant:6"
    assert attachment.transport_family.original_banks.over_fragment_id == "octant:3"
    assert attachment.transport_family.active_banks.over_fragment_id == "octant:3"
    assert attachment.expression_context["source_expression_frozen_for_later_geometry_variations"]
    assert not attachment.expression_context["first_root_midpoint_used"]


@pytest.mark.parametrize("label", ENTRY_LABELS)
def test_implicit_source_field_preserves_loaded_memory_strains_and_energy_density(entry_handover_case, label):
    case = entry_handover_case
    attachment, = case["handovers"][label].attachments
    field = attachment.source_field_query
    assert field.previous_plastic_slip_bounds_m.lo == field.previous_plastic_slip_bounds_m.hi == 0
    assert field.plastic_increment_bounds_m.hi < 0
    assert field.cumulative_slip_bounds_m.lo > 0
    assert field.normal_pressure_bounds_pa.lo > 0
    assert field.bulk_energy_density_bounds_j_m2.lo > 0
    assert field.contact_energy_density_bounds_j_m2.lo > 0
    assert field.shear_traction_bounds_pa == I(F(0))
    for angle in (field.computational_angle_bounds.lo, field.computational_angle_bounds.hi):
        for time in (field.time_fraction_bounds.lo, field.time_fraction_bounds.hi):
            for name, expected in independent_field(case["history"], field.trace_id, angle, time).items():
                enclosed(getattr(field, name), expected)
    assert attachment.source_free_gap_bounds == attachment.transported_free_gap_bounds
    assert attachment.source_free_jump_bounds == attachment.transported_free_jump_bounds
    assert field.source_previous_memory is case["history"].previous_memory
    assert field.source_unaccepted_candidate_memory is case["history"].unaccepted_candidate_memory


def atan_small(value):
    result = term = value
    for index in range(1, 3000):
        term *= -value*value
        addition = term/Decimal(2*index+1)
        result += addition
        if abs(addition) < Decimal("1e-560"):
            return result
    raise AssertionError("Independent incoming bank angle failed to converge")


def cross(first, second):
    return (first[1]*second[2]-first[2]*second[1], first[2]*second[0]-first[0]*second[2],
            first[0]*second[1]-first[1]*second[0])


def dot(first, second):
    return sum((a*b for a, b in zip(first, second)), Decimal(0))


@pytest.mark.parametrize("label", ENTRY_LABELS)
def test_incoming_reference_uses_own_bank_interior_amplitude_at_same_implicit_clock(entry_handover_case, label):
    case, history = entry_handover_case, entry_handover_case["history"]
    value, old = case["handovers"][label], case["bindings"][label]
    attachment, = value.attachments
    trace = next(trace for trace in history.initial_mechanics.fault_traces.traces
        if trace.trace_id == attachment.original_donor_trace_id)
    parameter = next(row for row in history.initial_mechanics.bending_parameters if row.trace_id == trace.trace_id)
    old_radial = {row[0]: row[1:] for row in history.accepted_before.transverse_displacement_m}
    new_radial = {row[0]: row[1:] for row in history.actual_candidate.transverse_displacement_m}
    node = next(node for node in history.quadrature_histories
        if node.trace_id == attachment.original_receiver_trace_id)
    with localcontext() as context:
        context.prec = 600
        point = tuple(map(decimal, old.chronology.original_point))
        start, end = (tuple(decimal(F.from_float(x)) for x in values) for values in (trace.start, trace.end))
        inward = tuple(-x for x in cross(start, end))
        transverse = dot(point, inward)
        in_plane_norm = (dot(point, start)**2+dot(point, end)**2).sqrt()
        radius = decimal(F.from_float(history.origin_surface.radius_km)*1000)
        distance = radius*atan_small(transverse/in_plane_norm)
        coordinate = distance/decimal(F.from_float(parameter.sub.width_m))
        shape = (1-coordinate)**2*(coordinate+2)/2
        enclosed(attachment.sub_material_chart.shape, shape)
        enclosed(attachment.sub_material_chart.inward_distance_m, distance)
        assert shape < 1
        sine, cosine = (decimal(F.from_float(x)) for x in node.original_frame)
        for time in (old.time_fraction_bounds.lo, old.time_fraction_bounds.hi):
            fraction = decimal(time)
            ws0, ws1 = (decimal(F.from_float(radial[attachment.original_donor_trace_id][0]))
                for radial in (old_radial, new_radial))
            wo0, wo1 = (decimal(F.from_float(radial[attachment.original_receiver_trace_id][1]))
                for radial in (old_radial, new_radial))
            sub_tip, over_tip = ws0+fraction*(ws1-ws0), wo0+fraction*(wo1-wo0)
            enclosed(attachment.sub_tip_displacement_bounds_m, sub_tip)
            enclosed(attachment.over_tip_displacement_bounds_m, over_tip)
            enclosed(attachment.sub_material_chart.displacement_m, shape*sub_tip)
            difference = over_tip-shape*sub_tip
            enclosed(attachment.reference_gap_bounds, cosine*difference)
            enclosed(attachment.reference_jump_bounds, -sine*difference)


def test_event_reference_cancels_exactly_and_later_geometry_keeps_first_source_clock_frozen(entry_handover_case):
    from tectonics.geometric_native_first_entry_contact_handover import (
        evaluate_native_first_entry_contact_free_anchors,
        native_first_entry_contact_handover_to_dict,
    )
    value = entry_handover_case["handovers"]["transverse_masked_main_entry"]
    attachment, = value.attachments
    before = native_first_entry_contact_handover_to_dict(value)
    reference = attachment.reference_gap_bounds, attachment.reference_jump_bounds
    source = attachment.source_free_gap_bounds, attachment.source_free_jump_bounds
    actual = evaluate_native_first_entry_contact_free_anchors(value, 0, reference,
        geometry_context=attachment.expression_context)
    assert actual == source
    future_geometry = (reference[0]+F(1, 7), reference[1]-F(2, 11))
    future_context = dict(attachment.expression_context, subsequent_pose_expression="later independent pose")
    future = evaluate_native_first_entry_contact_free_anchors(value, 0, future_geometry,
        geometry_context=future_context)
    assert future[0].lo <= source[0].lo+F(1, 7) <= source[0].hi+F(1, 7) <= future[0].hi
    assert future[1].lo <= source[1].lo-F(2, 11) <= source[1].hi-F(2, 11) <= future[1].hi
    assert attachment.source_field_query.time_fraction_bounds == value.binding.time_fraction_bounds
    assert native_first_entry_contact_handover_to_dict(value) == before
    with pytest.raises(ValueError):
        evaluate_native_first_entry_contact_free_anchors(value, 0, future_geometry,
            geometry_context=attachment.expression_context)


def test_exit_and_reentry_keep_first_receiver_and_single_consumed_attachment(entry_handover_case):
    case = entry_handover_case
    attachment, = case["handovers"]["transverse_masked_main_entry"].attachments
    assert attachment.transport_family.active_banks.over_fragment_id == "octant:3"
    assert attachment.transport_family.revision == 1
    assert "octant:2" not in attachment.desired_fragment_pair
    source, = case["handovers"]["main_plane_reentry"].attachments
    first = case["contexts"]["main_plane_reentry"].first_entry_certificate.first_entry_time_fraction
    assert source.source_field_query.time_fraction_bounds == first
    assert source.transport_family.revision == 1
    assert source.source_field_query.source_expression_context["first_entry_time_fraction_bounds"] == {
        "lo": [first.lo.numerator, first.lo.denominator], "hi": [first.hi.numerator, first.hi.denominator]}


def test_all_source_ledgers_and_global_force_clock_remain_literal_unaccepted_records(entry_handover_case):
    from tectonics.geometric_native_first_entry_contact_handover import native_first_entry_contact_handover_to_dict
    case, history = entry_handover_case, entry_handover_case["history"]
    assert native_physical_history_binding_to_dict(history) == case["source"]
    for value in case["handovers"].values():
        raw = native_first_entry_contact_handover_to_dict(value)
        assert raw["complete_original_source_history"] == case["source"]
        assert not raw["first_root_midpoint_used"]
        assert not raw["whole_first_entry_domain_integration_adopted"]
        assert not raw["whole_spatial_bank_and_clamp_support_certified"]
        assert raw["accepted_loaded_native_intervals"] == 0
        assert raw["accepted_source_work_j"] == [0, 1]
        for item in raw["attachments"]:
            assert item["finite_point_area_m2"] is None
            assert item["finite_point_work_j"] is None
            assert item["transport_family"]["receipts"][0]["source_binding_consumed"]
            assert item["transport_family"]["receipts"][0]["signed_transfer_external_work_identity"] == "0"
    assert history.accepted_before.elapsed_time_s == 0
    assert history.accepted_before.cumulative_source_work_j == 0
    assert history.accepted_before.receipts == ()


def test_full_handover_roundtrip_requires_original_binding_and_full_source(entry_handover_case):
    from tectonics.geometric_native_first_entry_contact_handover import (
        native_first_entry_contact_handover_from_dict, native_first_entry_contact_handover_to_dict,
    )
    case = entry_handover_case
    value = case["handovers"]["transverse_masked_main_entry"]
    raw = native_first_entry_contact_handover_to_dict(value)
    rebuilt = native_first_entry_contact_handover_from_dict(deepcopy(raw),
        expected_binding=case["bindings"]["transverse_masked_main_entry"],
        expected_source_history_dto=case["source"], expected_law=case["law"])
    assert native_first_entry_contact_handover_to_dict(rebuilt) == raw


def reseal(raw):
    raw["handover_sha256"] = hashlib.sha256(json.dumps({key: item for key, item in raw.items()
        if key != "handover_sha256"}, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return raw


@pytest.mark.parametrize("mutation", ["source_trace", "own_beam", "receiver", "root", "memory", "energy", "clock",
    "area", "work", "active_banks", "original_work", "midpoint", "duplicate"])
def test_rehashed_handover_cannot_forge_source_geometry_energy_or_first_ancestry(entry_handover_case, mutation):
    from tectonics.geometric_native_first_entry_contact_handover import (
        native_first_entry_contact_handover_from_dict, native_first_entry_contact_handover_to_dict,
    )
    case = entry_handover_case
    raw = deepcopy(native_first_entry_contact_handover_to_dict(case["handovers"]["transverse_masked_main_entry"]))
    attachment = raw["attachments"][0]
    if mutation == "source_trace":
        attachment["source_field_query"]["trace_id"] = attachment["original_donor_trace_id"]
    elif mutation == "own_beam":
        attachment["sub_material_chart"]["source"]["trace_id"] = attachment["original_receiver_trace_id"]
    elif mutation == "receiver":
        attachment["desired_fragment_pair"][1] = "octant:2"
    elif mutation == "root":
        attachment["expression_context"]["first_entry_time_fraction_bounds"] = {"lo": [0, 1], "hi": [0, 1]}
    elif mutation in ("memory", "energy", "clock"):
        key = {"memory": "plastic_slip_bounds_m", "energy": "total_energy_density_bounds_j_m2",
               "clock": "time_fraction_bounds"}[mutation]
        attachment["source_field_query"][key] = {"lo": [0, 1], "hi": [0, 1]}
    elif mutation in ("area", "work"):
        attachment["finite_point_area_m2" if mutation == "area" else "finite_point_work_j"] = [1, 1]
    elif mutation == "active_banks":
        attachment["transport_family"]["active_banks"]["sub_fragment_id"] = "octant:7"
    elif mutation == "original_work":
        raw["complete_original_source_history"]["actual_candidate"]["states"][0][1]["cumulative_external_work_j"] += 1
    elif mutation == "midpoint":
        raw["first_root_midpoint_used"] = True
    else:
        raw["attachments"].append(deepcopy(attachment))
    with pytest.raises(ValueError):
        native_first_entry_contact_handover_from_dict(reseal(raw),
            expected_binding=case["bindings"]["transverse_masked_main_entry"],
            expected_source_history_dto=case["source"], expected_law=case["law"])
