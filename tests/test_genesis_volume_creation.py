"""Source-context and replay tests for separate cohesive creation histories."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis_volume_banks import prepare_volume_banks
from tectonics.genesis_volume_checkpoint import _canonical
from tectonics.genesis_volume_creation import (
    bind_volume_creation_context, creation_context_to_dict,
    make_computational_creation_state, step_computational_creation_state,
    creation_fixture_state_to_dict, creation_fixture_state_from_dict)
from tectonics.genesis_volume_creation_law import ProspectiveCreationLaw
from tectonics.genesis_volume_fracture_surface import prepare_volume_fracture_surface
from tectonics.genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict
from tectonics.genesis_volume_slip_dynamics import VolumeSlipModel


@pytest.fixture(scope="module")
def fixture():
    path = Path(__file__).resolve().parents[1]/(
        "analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json")
    before = path.read_bytes()
    model = VolumeSlipModel(json.loads(before))
    state = model.initial_state()
    baseline = _canonical(volume_slip_checkpoint_to_dict(model, state))
    law = ProspectiveCreationLaw(1e6, 4e6, 1e9, 2e9)  # Explicit mechanical fixture.
    context = bind_volume_creation_context(model, state, law)
    parent, basal = model._assembly(state.geometry, state.fault_history.damage)
    face = state.geometry.faces[0]
    banks = prepare_volume_banks(parent, [(face[i], face[(i+1)%3]) for i in range(3)], basal=basal)
    surface = prepare_volume_fracture_surface(banks)
    initial = make_computational_creation_state(context, surface)
    u = np.zeros(banks.ndof)
    u[parent.ndof:] = np.tile((.9, -.6, .7, 200., -300., 100.), (banks.ndof-parent.ndof)//6)
    first, response = step_computational_creation_state(context, surface, initial, u)
    terminal, last = step_computational_creation_state(context, surface, first, .25*u)
    yield model, state, context, surface, initial, first, terminal, response, last, u
    assert path.read_bytes() == before
    assert _canonical(volume_slip_checkpoint_to_dict(model, state)) == baseline


def test_actual_source_has_no_declared_fracture_law_or_credit(fixture):
    model, state, *_ = fixture
    raw = creation_context_to_dict(bind_volume_creation_context(model, state))
    assert raw["source_calculated_creation_law"] is None
    assert raw["declared_prospective_creation_law"] is None
    assert raw["physical_cut_status"] == "no_eligible_cut"
    assert raw["historical_dissipation_available_for_creation_j"] == 0
    assert raw["creation_work_debited_from_Genesis_j"] == 0
    assert not raw["existing_damage_or_slip_work_reclassified"]
    assert not raw["material_history_reinitialized"]
    assert raw["cohort_ids"] == [c.id for c in state.mechanics.cohorts]


def test_explicit_prospective_law_never_becomes_recovered_history(fixture):
    raw = creation_context_to_dict(fixture[2])
    assert raw["declared_prospective_creation_law"]["fracture_energy_j_m2"] == 1e6
    assert raw["source_calculated_creation_law"] is None
    assert not raw["physical_creation_admitted"] and not raw["physical_release_admitted"]
    assert raw["interface_area_convention"] == "paired_Gamma_once_Gc_includes_both_banks"


def test_current_birth_and_inherited_work_are_literal(fixture):
    model, state, context, *_ = fixture
    raw = creation_context_to_dict(context)
    packet = volume_slip_checkpoint_to_dict(model, state)
    assert raw["parent_checkpoint_sha256"] == packet["checkpoint_sha256"]
    assert raw["time_myr"] == state.time_myr
    assert raw["accepted_step_sizes_myr"] == list(state.accepted_step_sizes_myr)
    for kind, source in (("runtime", packet["runtime_state"]),
                         ("mechanics", packet["runtime_state"]["mechanics"]["state"]),
                         ("fault", packet["runtime_state"]["fault_history"])):
        assert raw["inherited_genesis_ledgers"][kind] == {k: v for k, v in source.items() if k.endswith("_j")}


def test_surface_work_supplies_stored_energy_and_creation_once(fixture):
    _, _, _, surface, initial, first, terminal, response, last, _ = fixture
    assert initial.cumulative_cohesive_work_j == initial.cumulative_fracture_work_j == 0
    assert first.cumulative_fracture_work_j > 0
    assert first.cumulative_fracture_work_j <= surface.total_area_m2*fixture[2].law.fracture_energy_j_m2*(1+1e-14)
    assert terminal.cumulative_fracture_work_j == first.cumulative_fracture_work_j
    assert last.integrated_fracture_increment_j == 0
    assert terminal.cumulative_cohesive_work_j == pytest.approx(
        terminal.stored_interface_energy_j+terminal.cumulative_fracture_work_j, rel=2e-14)
    assert terminal.cumulative_cohesive_work_j == pytest.approx(
        response.integrated_mechanical_work_j+last.integrated_mechanical_work_j, rel=2e-14)


def test_unchanged_path_does_not_charge_creation_twice(fixture):
    context, surface, first, u = fixture[2], fixture[3], fixture[5], fixture[-1]
    again, response = step_computational_creation_state(context, surface, first, u)
    assert response.integrated_fracture_increment_j == response.integrated_mechanical_work_j == 0
    assert again.cumulative_fracture_work_j == first.cumulative_fracture_work_j


def test_full_path_replay_and_midpoint_resume_are_exact(fixture):
    context, surface, first, terminal, u = fixture[2], fixture[3], fixture[5], fixture[6], fixture[-1]
    restored = creation_fixture_state_from_dict(creation_fixture_state_to_dict(first), context=context, surface=surface)
    resumed, _ = step_computational_creation_state(context, surface, restored, .25*u)
    expected = creation_fixture_state_to_dict(terminal)
    assert _canonical(creation_fixture_state_to_dict(resumed)) == _canonical(expected)
    restored = creation_fixture_state_from_dict(json.loads(_canonical(expected)), context=context, surface=surface)
    assert _canonical(creation_fixture_state_to_dict(restored)) == _canonical(expected)


@pytest.mark.parametrize("field,value", [
    ("cumulative_fracture_work_j", 0.),
    ("cumulative_cohesive_work_j", 0.),
    ("stored_interface_energy_j", -1.),
    ("physical_creation_admitted", True),
    ("physical_release_admitted", True),
    ("context_binding_sha256", "0"*64),
    ("surface_binding_sha256", "0"*64),
    ("format", "real_planetary_fracture_checkpoint"),
])
def test_resealed_archive_claims_cannot_replace_replayed_memory(fixture, field, value):
    context, surface, terminal = fixture[2], fixture[3], fixture[6]
    raw = creation_fixture_state_to_dict(terminal)
    raw[field] = value
    raw["state_sha256"] = hashlib.sha256(_canonical({k: v for k, v in raw.items() if k != "state_sha256"})).hexdigest()
    with pytest.raises(ValueError, match="full path replay"):
        creation_fixture_state_from_dict(raw, context=context, surface=surface)


def test_source_surface_and_law_mismatch_refused(fixture):
    model, state, context, surface, initial, *_ = fixture
    no_law = bind_volume_creation_context(model, state)
    with pytest.raises(ValueError, match="explicit prospective"):
        make_computational_creation_state(no_law, surface)
    other = bind_volume_creation_context(model, state, ProspectiveCreationLaw(2e6,4e6,1e9,2e9))
    with pytest.raises(ValueError, match="source/surface"):
        step_computational_creation_state(other, surface, initial, np.zeros(surface.banks.ndof))
    forged = replace(initial, cumulative_fracture_work_j=1.)
    with pytest.raises(ValueError, match="source/surface"):
        step_computational_creation_state(context, surface, forged, np.zeros(surface.banks.ndof))


def test_changed_context_and_mutable_fields_refused(fixture):
    context, first = fixture[2], fixture[5]
    with pytest.raises(ValueError, match="accepted source binding"):
        creation_context_to_dict(replace(context, payload_json="{}"))
    for name in ("bank_displacement_from_fixture_origin_m", "local_gap_m", "maximum_separation_m"):
        with pytest.raises(ValueError):
            getattr(first, name).setflags(write=True)


@pytest.mark.parametrize("value", [True, 1j, float("nan"), float("inf")])
def test_invalid_fixture_path_fields_refused(fixture, value):
    context, surface, initial = fixture[2], fixture[3], fixture[4]
    bad = [0.]*surface.banks.ndof
    bad[0] = value
    with pytest.raises(ValueError):
        step_computational_creation_state(context, surface, initial, bad)
