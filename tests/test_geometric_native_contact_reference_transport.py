"""Loaded actual-state transport, reference cancellation and lineage controls."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import hashlib
import json

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.geometric_native_physical_history import native_physical_history_binding_to_dict
from tectonics.geometric_native_contact_reference_transport import (
    ContactHistoryTransportLaw, TRANSPORT_LAW_MODEL,
    contact_reference_transport_from_dict, contact_reference_transport_to_dict,
    contact_reference_transport_symbols_from_dict, contact_reference_transport_symbols_to_dict,
    evaluate_contact_reference_free_anchors, initialize_contact_reference_transport,
    evaluate_contact_reference_transport_symbols, initialize_contact_reference_transport_symbols,
    make_contact_bank_attachment, transport_contact_reference,
    transport_contact_reference_symbols,
    validate_contact_reference_transport, validate_contact_reference_transport_ledger,
)
from tests.test_geometric_native_physical_history import get_physical_case, independent_dense_pair


F = Fraction


@pytest.fixture(scope="module")
def actual_source():
    case = get_physical_case()
    history = case["binding"]
    node = history.quadrature_histories[4]
    binding = next(row for row in history.material_state.trace_bindings if row.trace_id == node.trace_id)
    source = native_physical_history_binding_to_dict(history)
    law = ContactHistoryTransportLaw(TRANSPORT_LAW_MODEL,
        "Declared reduced complete-configuration transport for independent local controls")
    context = dict(model="declared_reference_expression", original_point_id=node.point_id,
        source_epoch="unaccepted_candidate", expression="literal original finite-pose projection")
    target_context = dict(model="declared_reference_expression", original_point_id=node.point_id,
        source_epoch="unaccepted_candidate", expression="explicit independently declared incoming projection")
    banks = make_contact_bank_attachment(binding.donor_fragment_id, binding.donor_material_id,
        binding.subducting_plate, binding.receiver_fragment_id, binding.receiver_material_id,
        binding.overriding_plate, correspondence=dict(model="declared_original_material_correspondence",
        original_point_id=node.point_id, receiver_material_frame="unchanged",
        incoming_label_expression="independent complete expression; no midpoint"))
    return dict(case=case, history=history, node=node, source=source, law=law,
        context=context, target_context=target_context, banks=banks)


def initialize(source, *, epoch="unaccepted_candidate"):
    return initialize_contact_reference_transport(source["history"], source["node"].point_id,
        law=source["law"], source_epoch=epoch, source_binding_id="original-active",
        h_old=(I(F(-2), F(-1)), I(F(3), F(4))),
        geometry_context=dict(source["context"], source_epoch=epoch),
        expected_source_history_dto=source["source"])


def transition(source, state, **changes):
    material = source["node"].original_material
    parameters = dict(target_binding_id="incoming-active", target_banks=source["banks"],
        h_new=(I(F(-5), F(-3)), I(F(7), F(11))),
        geometry_context=source["target_context"], target_columns=material.columns,
        target_contact_law=material.law, target_water=material.water,
        target_area_m2=source["node"].original_area_m2,
        expected_source_history_dto=source["source"])
    parameters.update(changes)
    return transport_contact_reference(state, **parameters)


@pytest.mark.parametrize("epoch", ["before", "unaccepted_candidate"])
def test_complete_real_configuration_energy_work_and_exact_epoch_are_preserved(actual_source, epoch):
    original = initialize(actual_source, epoch=epoch)
    transferred = transition(actual_source, original)
    node = actual_source["node"]
    expected = node.previous_state if epoch == "before" else node.unaccepted_candidate_state
    assert original.local_state is transferred.local_state is expected
    assert expected.gap_free_m < 0
    assert expected.traction_normal_pa < 0
    assert expected.bulk_energy_j > 0
    assert expected.contact_energy_j > 0
    assert expected.cumulative_external_work_j > 0
    if epoch == "unaccepted_candidate":
        assert expected.plastic_slip_m < 0
        assert expected.cumulative_slip_m > 0
        assert expected.elapsed_time_s > node.previous_state.elapsed_time_s
    assert transferred.revision == 1
    assert transferred.active_binding_id == "incoming-active"
    assert transferred.consumed_binding_ids == ("original-active",)
    assert transferred.active_history_count == 1
    assert transferred.work_lineage_id == original.work_lineage_id
    receipt = transferred.receipts[0]
    assert receipt.signed_external_work_j == receipt.dissipation_j == receipt.energy_change_j == F(0)
    assert original.receipts == ()
    assert original.active_binding_id == "original-active"
    assert not transferred.global_force_candidate_adopted
    assert not transferred.nodal_work_assigned_to_material_volume
    assert actual_source["history"].accepted_before.elapsed_time_s == 0
    assert native_physical_history_binding_to_dict(actual_source["history"]) == actual_source["source"]


def test_event_geometry_cancels_symbolically_and_future_geometry_preserves_loaded_reference(actual_source):
    state = transition(actual_source, initialize(actual_source))
    free = tuple(F.from_float(value) for value in (state.local_state.gap_free_m, state.local_state.jump_free_m))
    at_event = evaluate_contact_reference_free_anchors(state, state.reference_geometry,
        geometry_context=actual_source["target_context"])
    assert at_event == tuple(I(value) for value in free)
    different_context = dict(actual_source["target_context"], clock="later exact clock")
    future = evaluate_contact_reference_free_anchors(state, (F(-2), F(13)),
        geometry_context=different_context)
    assert future == (I(free[0]+1, free[0]+3), I(free[1]+2, free[1]+6))
    # There is no chosen root/geometry midpoint in the transported anchor.
    assert future[0].lo < future[0].hi and future[1].lo < future[1].hi
    with pytest.raises(ValueError):
        evaluate_contact_reference_free_anchors(state, (F(-2), F(13)),
            geometry_context=actual_source["target_context"])


def test_consumption_chain_forbids_reused_identity_and_assembled_forks(actual_source):
    initial = initialize(actual_source)
    first = transition(actual_source, initial)
    second = transition(actual_source, first, target_binding_id="incoming-next",
        geometry_context=dict(actual_source["target_context"], next_revision=True))
    assert second.consumed_binding_ids == ("original-active", "incoming-active")
    assert second.active_binding_id == "incoming-next"
    for reused in ("original-active", "incoming-active", "incoming-next"):
        with pytest.raises(ValueError):
            transition(actual_source, second, target_binding_id=reused)
    fork = transition(actual_source, initial, target_binding_id="fork-active")
    assert validate_contact_reference_transport_ledger((second,)) == (second,)
    for duplicate in ((first, second), (first, fork), (second, second)):
        with pytest.raises(ValueError):
            validate_contact_reference_transport_ledger(duplicate)
    before = transition(actual_source, initialize(actual_source, epoch="before"),
        target_binding_id="before-epoch-active")
    assert before.work_lineage_id == second.work_lineage_id
    with pytest.raises(ValueError):
        validate_contact_reference_transport_ledger((before, second))
    with pytest.raises(ValueError):
        validate_contact_reference_transport_ledger([second])


@pytest.mark.parametrize("parameter", ["columns", "law", "water", "area", "area_alias"])
def test_zero_work_transport_refuses_changed_compliance_law_water_or_measure(actual_source, parameter):
    state = initialize(actual_source)
    material = actual_source["node"].original_material
    changes = {
        "columns": dict(target_columns=replace(material.columns,
            sub_length_m=material.columns.sub_length_m+1)),
        "law": dict(target_contact_law=replace(material.law,
            normal_stiffness_pa_m=material.law.normal_stiffness_pa_m*2)),
        "water": dict(target_water=0.),
        "area": dict(target_area_m2=actual_source["node"].original_area_m2+1),
        "area_alias": dict(target_area_m2=F.from_float(actual_source["node"].original_area_m2)),
    }[parameter]
    with pytest.raises(ValueError):
        transition(actual_source, state, **changes)


@pytest.mark.parametrize("parameter", ["receiver", "material", "owner_alias"])
def test_declared_attachment_cannot_change_receiver_or_forge_material(actual_source, parameter):
    banks = actual_source["banks"]
    changes = {
        "receiver": dict(over_fragment_id=banks.sub_fragment_id,
            over_material_id=banks.sub_material_id, over_owner=banks.sub_owner),
        "material": dict(sub_material_id="invented-new-material"),
        "owner_alias": dict(sub_owner=float(banks.sub_owner)),
    }[parameter]
    with pytest.raises(ValueError):
        transition(actual_source, initialize(actual_source), target_banks=replace(banks, **changes))


def test_restart_replays_complete_source_and_transition_lineage(actual_source):
    state = transition(actual_source, initialize(actual_source))
    raw = contact_reference_transport_to_dict(state)
    rebuilt = contact_reference_transport_from_dict(deepcopy(raw), history=actual_source["history"],
        expected_source_history_dto=actual_source["source"], law=actual_source["law"])
    assert contact_reference_transport_to_dict(rebuilt) == raw
    assert rebuilt.local_state is actual_source["node"].unaccepted_candidate_state
    assert raw["original_local_state"]["plastic_slip_m"] < 0
    assert raw["active_history_count"] == raw["receipts"][0]["original_work_lineage_count"] == 1


@pytest.mark.parametrize("field", ["plastic", "work", "area", "clock", "consumed", "active_count", "geometry", "revision", "source_work"])
def test_resealed_restart_cannot_change_actual_state_or_transition_history(actual_source, field):
    raw = contact_reference_transport_to_dict(transition(actual_source, initialize(actual_source)))
    if field in ("plastic", "work", "area", "clock"):
        name = dict(plastic="plastic_slip_m", work="cumulative_external_work_j",
            area="area_m2", clock="elapsed_time_s")[field]
        raw["original_local_state"][name] += 1
    elif field == "consumed":
        raw["consumed_binding_ids"] = []
    elif field == "active_count":
        raw["active_history_count"] = 2
    elif field == "geometry":
        raw["reference_geometry"][0]["lo"] = [-6, 1]
    elif field == "revision":
        raw["receipts"][0]["revision"] = 2
    else:
        raw["source_history"]["actual_candidate"]["states"][0][1]["cumulative_external_work_j"] += 1
    raw["state_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in raw.items()
        if key != "state_sha256"}, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError):
        contact_reference_transport_from_dict(raw, history=actual_source["history"],
            expected_source_history_dto=actual_source["source"], law=actual_source["law"])


def test_memory_mutation_hash_only_source_and_literal_alias_cannot_authenticate(actual_source):
    state = transition(actual_source, initialize(actual_source))
    forged = replace(state, local_state=replace(state.local_state, plastic_slip_m=0.))
    with pytest.raises(ValueError):
        validate_contact_reference_transport(forged)
    with pytest.raises(ValueError):
        transition(actual_source, state,
            expected_source_history_dto={"binding_sha256": actual_source["history"].binding_sha256})
    raw = contact_reference_transport_to_dict(state)
    raw["receipts"] = tuple(raw["receipts"])
    with pytest.raises(ValueError):
        contact_reference_transport_from_dict(raw, history=actual_source["history"],
            expected_source_history_dto=actual_source["source"], law=actual_source["law"])


@pytest.mark.parametrize("epoch", ["accepted_candidate", "root_midpoint", True])
def test_only_real_before_or_unaccepted_endpoint_source_is_available(actual_source, epoch):
    with pytest.raises(ValueError):
        initialize(actual_source, epoch=epoch)


@pytest.mark.parametrize("model", ["implicit_default", "same_plate_ids", True])
def test_transport_requires_explicit_supported_law(model):
    with pytest.raises(ValueError):
        ContactHistoryTransportLaw(model, "explicit provenance")


@pytest.fixture(scope="module")
def declared_cross_pair_family(actual_source):
    """Declared field law only; this fixture proves no spatial root admission."""
    history = actual_source["history"]
    receiving = next(row for row in history.material_state.trace_bindings
        if row.donor_fragment_id == "octant:7" and row.receiver_fragment_id == "octant:3")
    donor = next(row for row in history.material_state.trace_bindings
        if row.donor_fragment_id == "octant:6")
    original_context = dict(model="original_continuous_receiving_interface_expression",
        trace_id=receiving.trace_id, material_coordinate="original theta",
        clock="implicit exact u; no selected midpoint")
    target_context = dict(model="declared_incoming_continuous_expression",
        trace_id=receiving.trace_id, source_donor=donor.donor_fragment_id,
        receiving_material=receiving.receiver_fragment_id,
        spatial_first_root_proved=False)
    original = initialize_contact_reference_transport_symbols(history, receiving.trace_id,
        law=actual_source["law"], source_binding_id="receiving-original-field",
        geometry_context=original_context, expected_source_history_dto=actual_source["source"])
    banks = make_contact_bank_attachment(donor.donor_fragment_id, donor.donor_material_id,
        donor.subducting_plate, receiving.receiver_fragment_id, receiving.receiver_material_id,
        receiving.overriding_plate, correspondence=target_context)
    trace = next(row for row in history.initial_mechanics.fault_traces.traces
        if row.trace_id == receiving.trace_id)
    family = transport_contact_reference_symbols(original, target_binding_id="declared-six-three-field",
        target_banks=banks, geometry_context=target_context, target_columns=trace.columns,
        target_contact_law=trace.law, target_water=trace.water,
        expected_source_history_dto=actual_source["source"])
    node = next(row for row in history.quadrature_histories if row.trace_id == receiving.trace_id)
    return dict(family=family, original=original, context=target_context, node=node)


@pytest.mark.parametrize("clock", [I(F(0), F(1)), I(F(1, 10**400))])
def test_symbol_field_retains_original_continuous_anchors_on_exact_clock(actual_source, declared_cross_pair_family, clock):
    field = declared_cross_pair_family
    family, node = field["family"], field["node"]
    reference = (I(F(-5), F(-3)), I(F(7), F(11)))
    result = evaluate_contact_reference_transport_symbols(family, I(node.computational_angle),
        clock, reference, reference, geometry_context=field["context"],
        reference_geometry_context=field["context"], precision=128)
    samples = (F(0), F(1, 3), F(1)) if clock.lo != clock.hi else (clock.lo,)
    for time in samples:
        independent = independent_dense_pair(actual_source["history"], node, time)
        for bounds, name in zip(result, ("free_gap_bounds_m", "free_jump_bounds_m")):
            assert bounds.lo <= F(independent[name]) <= bounds.hi
    if clock.lo == clock.hi:
        assert result[1].hi < 0
        assert float(result[1].hi) == 0
    assert family.source_contact_time_path is actual_source["history"].contact_time_path
    assert family.source_continuous_memory is actual_source["history"].contact_time_path.step.previous_memory


def test_cross_pair_symbol_family_has_no_nodal_area_work_or_midpoint_state(actual_source, declared_cross_pair_family):
    family = declared_cross_pair_family["family"]
    assert family.original_banks.sub_fragment_id == "octant:7"
    assert family.active_banks.sub_fragment_id == "octant:6"
    assert family.original_banks.over_fragment_id == family.active_banks.over_fragment_id == "octant:3"
    assert family.consumed_binding_ids == ("receiving-original-field",)
    assert family.finite_point_area_m2 is family.finite_point_work_j is None
    assert not family.global_force_candidate_adopted
    raw = contact_reference_transport_symbols_to_dict(family)
    assert raw["finite_point_area_m2"] is raw["finite_point_work_j"] is None
    assert "original continuous" in raw["memory_identity"]
    assert raw["receipts"][0]["signed_transfer_external_work_identity"] == "0"
    assert raw["active_banks"]["correspondence"]["spatial_first_root_proved"] is False
    rebuilt = contact_reference_transport_symbols_from_dict(deepcopy(raw),
        history=actual_source["history"], expected_source_history_dto=actual_source["source"],
        law=actual_source["law"])
    assert contact_reference_transport_symbols_to_dict(rebuilt) == raw


@pytest.mark.parametrize("mutation", ["finite_area", "work", "receipt", "memory", "tuple"])
def test_symbol_restart_refuses_finite_measure_work_forged_receipt_or_type_alias(actual_source, declared_cross_pair_family, mutation):
    raw = contact_reference_transport_symbols_to_dict(declared_cross_pair_family["family"])
    if mutation == "finite_area":
        raw["finite_point_area_m2"] = actual_source["node"].original_area_m2
    elif mutation == "work":
        raw["finite_point_work_j"] = actual_source["node"].unaccepted_candidate_state.cumulative_external_work_j
    elif mutation == "receipt":
        raw["receipts"][0]["source_binding_consumed"] = False
    elif mutation == "memory":
        raw["memory_identity"] = "reset all incoming plastic memory to zero"
    else:
        raw["receipts"] = tuple(raw["receipts"])
    raw["state_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in raw.items()
        if key != "state_sha256"}, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError):
        contact_reference_transport_symbols_from_dict(raw, history=actual_source["history"],
            expected_source_history_dto=actual_source["source"], law=actual_source["law"])


def test_symbol_current_context_cannot_change_same_expression_or_invent_source(actual_source, declared_cross_pair_family):
    field = declared_cross_pair_family
    with pytest.raises(ValueError):
        evaluate_contact_reference_transport_symbols(field["family"], I(field["node"].computational_angle),
            I(F(1, 3)), (F(1), F(2)), (F(3), F(4)), geometry_context=field["context"],
            reference_geometry_context=field["context"])
    raw = contact_reference_transport_symbols_to_dict(field["family"])
    with pytest.raises(ValueError):
        contact_reference_transport_symbols_from_dict(raw, history=actual_source["history"],
            expected_source_history_dto={"binding_sha256": actual_source["history"].binding_sha256},
            law=actual_source["law"])
