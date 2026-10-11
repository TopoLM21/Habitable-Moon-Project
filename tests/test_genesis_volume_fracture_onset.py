"""Activation witnesses retain the actual accepted Genesis source and clocks."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis_volume_checkpoint import _canonical
from tectonics.genesis_volume_fracture_onset import (
    bind_calculated_volume_fracture_onset, calculated_volume_fracture_onset_to_dict,
    calculated_volume_fracture_onset_from_dict)
from tectonics.genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict
from tectonics.genesis_volume_slip_dynamics import VolumeSlipModel


@pytest.fixture(scope="module")
def accepted():
    path = Path(__file__).resolve().parents[1]/(
        "analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json")
    original = path.read_bytes()
    model = VolumeSlipModel(json.loads(original))
    state = model.initial_state()
    baseline = _canonical(volume_slip_checkpoint_to_dict(model, state))
    onset = bind_calculated_volume_fracture_onset(model, state)
    yield model, state, onset
    assert path.read_bytes() == original
    assert _canonical(volume_slip_checkpoint_to_dict(model, state)) == baseline


def test_accepted_start_orientation_prerequisite_has_no_fabricated_surface(accepted):
    model, state, onset = accepted
    raw = calculated_volume_fracture_onset_to_dict(onset)
    assert raw["new_activation_event"] is None
    assert raw["newly_activating_faces"] == raw["existing_active_faces"] == []
    assert raw["selected_sharp_material_surface"] is None
    assert raw["source_creation_energy_law"] is None
    assert raw["material_surface_status"] == "no_calculated_activation_event"
    assert raw["event_evaluation_scope"] == "next_interval_start_at_current_accepted_state"
    assert raw["time_myr"] == state.time_myr
    assert raw["accepted_step_sizes_myr"] == []
    assert raw["accepted_steps"] == state.mechanics.accepted_steps
    assert np.array_equal(raw["computed_face_stress_pa"], model.face_response(state)["stress_pa"])
    assert raw["accepted_fault_history"] == raw["prepared_fault_history"]


def test_no_seed_or_historical_energy_credit_enters_genesis(accepted):
    model, state, onset = accepted
    raw = calculated_volume_fracture_onset_to_dict(onset)
    packet = volume_slip_checkpoint_to_dict(model, state)
    assert raw["parent_checkpoint_sha256"] == packet["checkpoint_sha256"]
    assert raw["formation_checkpoint_sha256"] == packet["source_checkpoint_sha256"]
    assert raw["inherited_runtime_ledgers"] == {
        k: v for k, v in packet["runtime_state"].items() if k.endswith("_j")}
    assert raw["creation_work_debited_from_Genesis_j"] == 0
    for key in ("physical_creation_admitted", "physical_release_admitted",
                "physical_fault_contact_admission_claimed", "history_reinitialized",
                "activation_accepted", "clock_advanced"):
        assert raw[key] is False
    assert raw["live_cohorts"] == [dict(cohort_id=c.id, face_index=c.face_index,
        owner_id=c.owner_id, material_id=c.material_id) for c in state.geometry.cohorts]


def test_onset_archive_recomputes_full_source(accepted):
    model, state, onset = accepted
    raw = json.loads(_canonical(calculated_volume_fracture_onset_to_dict(onset)))
    restored = calculated_volume_fracture_onset_from_dict(raw, model=model, state=state)
    assert restored == onset
    with pytest.raises(FrozenInstanceError):
        onset.payload_json = "{}"
    with pytest.raises(ValueError, match="immutable"):
        calculated_volume_fracture_onset_to_dict(replace(onset, payload_json="{}"))


@pytest.mark.parametrize("field,value", [
    ("time_myr", 0.), ("parent_checkpoint_sha256", "0"*64),
    ("formation_checkpoint_sha256", "0"*64), ("newly_activating_faces", [0]),
    ("new_activation_event", {"event": "invented"}),
    ("selected_sharp_material_surface", {"edge": [0, 1]}),
    ("physical_creation_admitted", True), ("physical_release_admitted", True),
    ("creation_work_debited_from_Genesis_j", 1.), ("clock_advanced", True),
    ("history_reinitialized", True), ("unexpected_field", 0.),
])
def test_resealed_onset_substitutions_are_refused(accepted, field, value):
    model, state, onset = accepted
    raw = calculated_volume_fracture_onset_to_dict(onset)
    raw[field] = value
    raw["onset_sha256"] = hashlib.sha256(_canonical({
        k: v for k, v in raw.items() if k != "onset_sha256"})).hexdigest()
    with pytest.raises(ValueError, match="full accepted-source"):
        calculated_volume_fracture_onset_from_dict(raw, model=model, state=state)


def test_resealed_stress_is_not_an_external_event_input(accepted):
    model, state, onset = accepted
    raw = calculated_volume_fracture_onset_to_dict(onset)
    raw["computed_face_stress_pa"][0][0] += 1e6
    raw["onset_sha256"] = hashlib.sha256(_canonical({
        k: v for k, v in raw.items() if k != "onset_sha256"})).hexdigest()
    with pytest.raises(ValueError, match="full accepted-source"):
        calculated_volume_fracture_onset_from_dict(raw, model=model, state=state)


def test_an_initial_seeded_plane_is_not_accepted_genesis_history(accepted):
    model, state, _ = accepted
    damage = state.fault_history.damage.copy()
    damage[0] = .7
    modified = replace(state, fault_history=replace(state.fault_history, damage=damage))
    with pytest.raises(ValueError, match="literal Genesis"):
        bind_calculated_volume_fracture_onset(model, modified)


@pytest.mark.parametrize("model,state", [(None, None), (object(), object())])
def test_onset_requires_typed_source_bound_runtime(model, state):
    with pytest.raises(ValueError, match="accepted Genesis"):
        bind_calculated_volume_fracture_onset(model, state)
