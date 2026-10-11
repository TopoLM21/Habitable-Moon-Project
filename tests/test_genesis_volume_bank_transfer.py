"""Contextual transfer gates; the full accepted-control replay is run once separately."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis_volume_bank_transfer import (
    bind_volume_bank_transfer, validate_volume_bank_transfer_payload,
    volume_bank_transfer_to_dict)
from tectonics.genesis_volume_banks import prepare_volume_banks
from tectonics.genesis_volume_basal import assemble_genesis_volume_basal
from tectonics.genesis_volume_checkpoint import _canonical
from tectonics.genesis_volume_dynamics import coefficient_arrays
from tectonics.genesis_volume_slip_dynamics import VolumeSlipModel


@pytest.fixture(scope="module")
def context():
    path = Path(__file__).resolve().parents[1]/(
        "analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json")
    original = path.read_bytes()
    model = VolumeSlipModel(json.loads(original))
    state = model.initial_state()
    assembly, basal = model._assembly(state.geometry, state.fault_history.damage)
    face = state.geometry.faces[0]
    edges = [(face[i], face[(i+1)%3]) for i in range(3)]
    banks = prepare_volume_banks(assembly, edges, basal=basal)
    transfer = bind_volume_bank_transfer(model, state, banks)
    yield model, state, banks, transfer
    assert path.read_bytes() == original


def reseal(raw):
    body = {k: v for k, v in raw.items() if k != "transfer_sha256"}
    raw["transfer_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return raw


def test_material_and_memory_are_transferred_without_birth_or_increment(context):
    model, state, banks, transfer = context
    raw = volume_bank_transfer_to_dict(transfer)
    e0, e1 = coefficient_arrays(state.mechanics)
    np.testing.assert_array_equal(transfer.elastic_constant, e0)
    np.testing.assert_array_equal(transfer.elastic_inverse_scale, e1)
    np.testing.assert_array_equal(transfer.displacement_m, banks.R@state.displacement_m)
    np.testing.assert_array_equal(transfer.velocity_m_s, banks.R@state.velocity_m_s)
    assert not np.any(banks.J@transfer.displacement_m)
    assert not np.any(banks.J@transfer.velocity_m_s)
    assert raw["cohort_ids"] == [c.id for c in state.mechanics.cohorts]
    assert raw["time_myr"] == state.time_myr
    assert raw["accepted_step_sizes_myr"] == list(state.accepted_step_sizes_myr)
    assert raw["transition_work_j"] == raw["transition_time_increment_myr"] == 0
    assert raw["memory_update_increment_m"] == 0
    assert not raw["history_reinitialized"]
    assert not raw["connected_material_owner_ids_changed"]
    assert validate_volume_bank_transfer_payload(raw, expected_transfer=transfer) == raw


def test_computational_support_cannot_activate_actual_source(context):
    model, state, banks, transfer = context
    raw = volume_bank_transfer_to_dict(transfer)
    assert raw["support_kind"] == "explicit_computational_edge_split"
    assert len(raw["supplied_computational_cut_edges"]) == 3
    assert raw["active_weak_plane_faces"] == []
    assert raw["physical_cut_status"] == "no_eligible_cut"
    assert not raw["sharp_physical_banks_created"]
    assert not raw["physical_release_admitted"]
    assert not raw["endpoint_force_is_release_traction"]
    assert len(raw["unresolved_physical_inputs"]) == 5
    no_cut = bind_volume_bank_transfer(model, state)
    plain = volume_bank_transfer_to_dict(no_cut)
    assert plain["support_kind"] == "original_connected_domain"
    assert plain["supplied_computational_cut_edges"] == []
    assert plain["parent_checkpoint_sha256"] == raw["parent_checkpoint_sha256"]
    assert plain["full_cohort_memory_sha256"] == raw["full_cohort_memory_sha256"]


@pytest.mark.parametrize("field,value", [
    ("physical_cut_status", "admitted"),
    ("sharp_physical_banks_created", True),
    ("physical_release_admitted", True),
    ("spatial_underthrust_admitted", True),
    ("physical_creation_work_assigned", True),
    ("endpoint_force_is_release_traction", True),
    ("unresolved_physical_inputs", []),
    ("parent_checkpoint_sha256", "0"*64),
    ("formation_checkpoint_sha256", "0"*64),
    ("time_myr", 1.),
    ("accepted_steps", True),
    ("full_cohort_memory_sha256", "0"*64),
    ("birth_history_sha256", "0"*64),
    ("support_kind", "physical_material_cut"),
    ("transition_work_j", 1.),
])
def test_new_seal_cannot_change_bound_claims(context, field, value):
    transfer = context[3]
    raw = volume_bank_transfer_to_dict(transfer)
    raw[field] = value
    with pytest.raises(ValueError, match="accepted parent/support"):
        validate_volume_bank_transfer_payload(reseal(raw), expected_transfer=transfer)


def test_sealed_cohort_reordering_and_extra_field_are_refused(context):
    transfer = context[3]
    for mutation in (lambda p: p["cohort_ids"].reverse(),
                     lambda p: p.update(new_creation_law={"Gc": 2e7})):
        raw = volume_bank_transfer_to_dict(transfer)
        mutation(raw)
        with pytest.raises(ValueError):
            validate_volume_bank_transfer_payload(reseal(raw), expected_transfer=transfer)


def test_current_context_requires_mass_force_and_actual_drag(context):
    model, state, banks, _ = context
    incomplete = prepare_volume_banks(banks.parent_assembly, banks.topology.cut_edges)
    with pytest.raises(ValueError, match="basal mass, drag and force"):
        bind_volume_bank_transfer(model, state, incomplete)


def test_array_forgery_and_mutation_are_refused(context):
    transfer = context[3]
    forged = replace(transfer, elastic_inverse_scale=transfer.elastic_inverse_scale+1e-6)
    with pytest.raises(ValueError, match="accepted memory binding"):
        volume_bank_transfer_to_dict(forged)
    for name in ("displacement_m", "velocity_m_s", "elastic_constant", "elastic_inverse_scale"):
        with pytest.raises(ValueError):
            getattr(transfer, name).setflags(write=True)


def test_law_valid_operator_with_other_damage_is_not_current_context(context):
    model, state, banks, _ = context
    other, other_basal = model._assembly(state.geometry, state.fault_history.damage+.005)
    support = prepare_volume_banks(other, banks.topology.cut_edges, basal=other_basal)
    with pytest.raises(ValueError, match="different parent operator"):
        bind_volume_bank_transfer(model, state, support)


def test_source_law_valid_but_other_drag_is_not_current_context(context):
    model, state, banks, _ = context
    parameters = replace(model.contact_parameters,
                         basal_drag_pa_s_m=model.contact_parameters.basal_drag_pa_s_m/2)
    other_basal = assemble_genesis_volume_basal(banks.parent_assembly, parameters)
    support = prepare_volume_banks(banks.parent_assembly, banks.topology.cut_edges, basal=other_basal)
    with pytest.raises(ValueError, match="basal mass, drag and force"):
        bind_volume_bank_transfer(model, state, support)


def test_unknown_or_unbound_context_is_refused(context):
    model, state, banks, transfer = context
    for bad_model, bad_state in ((object(), state), (model, object())):
        with pytest.raises(ValueError):
            bind_volume_bank_transfer(bad_model, bad_state, banks)
    with pytest.raises(ValueError):
        bind_volume_bank_transfer(model, state, object())
    with pytest.raises(ValueError):
        validate_volume_bank_transfer_payload([], expected_transfer=transfer)
