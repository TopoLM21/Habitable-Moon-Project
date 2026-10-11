"""Actual-source volume weakening, conservation and untrusted restart checks."""
from copy import deepcopy
from dataclasses import fields, replace
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.genesis_volume_checkpoint import _canonical, volume_checkpoint_to_dict, volume_checkpoint_from_dict
from tectonics.genesis_volume_dynamics import VolumeModel
from tectonics.genesis_volume_dynamics import radial_moments
from tectonics.genesis_volume_loading import volume_mechanics_energy
from tectonics.genesis_volume_fault_checkpoint import (
    load_volume_fault_checkpoint, save_volume_fault_checkpoint,
    volume_fault_checkpoint_from_dict, volume_fault_checkpoint_to_dict)
from tectonics.genesis_volume_fault_dynamics import (
    DAMAGE_COUPLING, VolumeFaultActivationBoundary, VolumeFaultModel,
    retained_live_fraction, volume_face_response)


@pytest.fixture(scope="module")
def source():
    path = Path(__file__).resolve().parents[1]/(
        "analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json")
    original = path.read_bytes()
    yield json.loads(original)
    assert path.read_bytes() == original


@pytest.fixture(scope="module")
def initial(source):
    model = VolumeFaultModel(source)
    return model, model.initial_state()


@pytest.fixture(scope="module")
def first(initial):
    model, state = initial
    after = model.step(state, 1e-8)
    return model, state, after, volume_fault_checkpoint_to_dict(model, after)


def reseal(packet):
    body = {k: v for k, v in packet.items() if k != "checkpoint_sha256"}
    packet["checkpoint_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return packet


def test_initial_fault_fields_and_nonzero_cohort_memory_are_literal_source(initial, source):
    model, state = initial
    np.testing.assert_array_equal(state.fault_history.damage, source["current_shell_memory"]["damage"])
    assert state.mechanics.elastic_energy_j == source["cohort_mechanics"]["state"]["elastic_energy_j"]
    assert state.mechanics.elastic_energy_j > 4e19
    assert not state.fault_history.fault_active.any()
    assert state.damage_coupling == DAMAGE_COUPLING
    assert state.time_myr == state.fault_history.time_myr
    assert not state.geometry.stress_free_reinitialization
    assert model.validate_state(state) is state


def test_real_step_advances_damage_clock_without_changing_source_or_input(first):
    model, before, after, _ = first
    assert not np.array_equal(before.fault_history.damage, after.fault_history.damage)
    assert after.fault_history.accepted_steps == after.mechanics.accepted_steps == 1
    assert before.mechanics.accepted_steps == 0
    assert before.fault_history.accepted_steps == 0
    assert after.fault_history.time_myr == after.thermal.time_myr == after.geometry.time_myr == after.time_myr
    np.testing.assert_array_equal(after.thermal.damage, before.thermal.damage)
    assert not after.fault_history.fault_active.any()
    assert not after.fault_history.fault_candidate_age_myr.any()
    assert model.validate_state(after) is after


def test_only_final_solve_enters_mass_work_and_memory_journals(first):
    model, before, after, _ = first
    assert len(after.accepted_step_sizes_myr) == 1
    assert after.history.accepted_intervals == before.history.accepted_intervals+1
    assert len(after.mechanics.cohorts) == 560
    f = after.last_report["fault_evolution"]
    assert f["damage_energy_change_j"] != 0
    assert f["damage_energy_change_j"] == after.mechanics.damage_energy_change_j
    assert not f["physical_banks_admitted"]
    assert not f["combined_heat_mechanical_reservoir_closed"]
    energy = after.mechanics.elastic_energy_j+after.kinetic_energy_j
    m = after.mechanics
    expected = (m.initial_energy_j+after.initial_kinetic_energy_j+after.external_work_j
        +m.thermal_work_j-m.viscous_dissipation_j-after.basal_dissipation_j
        +m.damage_energy_change_j-m.remelt_energy_removed_j+after.phase_kinetic_energy_change_j)
    assert abs(energy-expected) < 1e-12*m.initial_energy_j
    assert after.last_report["momentum_equation_relative_residual"] < 1e-11
    assert model.validate_state(after) is after


def test_retained_fraction_uses_born_material_mass_not_equal_cohort_counts(first):
    _, before, after, _ = first
    retained = retained_live_fraction(before.geometry, after.geometry)
    assert np.all((retained > .99) & (retained < 1))
    old = np.zeros(80); new = np.zeros(80)
    for c in before.geometry.cohorts:
        old[c.face_index] += c.volume_m3
    for c in after.geometry.cohorts:
        new[c.face_index] += c.volume_m3
    np.testing.assert_allclose(retained, old/new, rtol=5e-15)


def test_source_geometry_assembly_respects_changed_damage(initial):
    model, state = initial
    original, _ = model._assembly(state.geometry, state.fault_history.damage)
    new_damage = state.fault_history.damage+.01
    updated, _ = model._assembly(state.geometry, new_damage)
    assert updated is not original
    assert np.max(np.abs(updated.elasticity_pa-original.elasticity_pa)) > 1e8
    assert updated.geometry is original.geometry


def test_face_stress_is_volume_mean_of_all_cohorts_rotated_to_current_axes(first):
    model, _, state, _ = first
    a, _ = model._assembly(state.geometry, state.fault_history.damage)
    p = model.face_response(state)
    assert p["stress_pa"].shape == (80, 3)
    assert p["world_stress_pa"].shape == (80, 3, 3)
    volume = np.zeros(80)
    for c in state.geometry.cohorts:
        volume[c.face_index] += c.volume_m3
    np.testing.assert_allclose(p["live_volume_m3"], volume, rtol=1e-15)
    frame = p["current_basis_xyz"]
    local = frame.transpose(0, 2, 1)@p["world_stress_pa"]@frame
    np.testing.assert_allclose(p["stress_pa"], np.stack((local[:, 0, 0], local[:, 1, 1], local[:, 0, 1]), axis=1))
    assert np.max(np.abs(p["stress_pa"])) > 1e6
    # A missing cohort memory must not be replaced by zeros or a face average.
    with pytest.raises(ValueError, match="identity"):
        volume_face_response(a, SimpleNamespace(cohorts=state.mechanics.cohorts[:-1]),
            state.fault_history.damage, state.geometry.vertices_m, model.shell_parameters)


def test_actual_direct_and_replayed_continuation_are_complete_exact(first):
    model, _, after, packet = first
    resumed_model, restored = volume_fault_checkpoint_from_dict(packet,
        expected_source_checkpoint_sha256=model.source_packet["source"]["checkpoint_sha256"])
    assert _canonical(volume_fault_checkpoint_to_dict(resumed_model, restored)) == _canonical(packet)
    direct = model.step(after, 1e-8)
    resumed = resumed_model.step(restored, 1e-8)
    assert _canonical(volume_fault_checkpoint_to_dict(model, direct)) == _canonical(
        volume_fault_checkpoint_to_dict(resumed_model, resumed))


def test_fault_checkpoint_text_stream_and_atomic_file(first, tmp_path):
    model, _, after, packet = first
    stream = io.StringIO()
    save_volume_fault_checkpoint(stream, model, after)
    stream.seek(0)
    m, s = load_volume_fault_checkpoint(stream)
    assert _canonical(volume_fault_checkpoint_to_dict(m, s)) == _canonical(packet)
    path = tmp_path/'actual.json'
    save_volume_fault_checkpoint(path, model, after)
    assert json.loads(path.read_text()) == packet
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize("field", ["damage", "water_access", "fault_candidate_age_myr", "peak_tensile_pa"])
def test_resealed_forged_fault_history_refused_even_if_mechanical_fields_unchanged(first, field):
    _, _, _, packet = first
    forged = deepcopy(packet)
    forged["runtime_state"]["fault_history"][field][0] += .01
    with pytest.raises(ValueError, match="replay"):
        volume_fault_checkpoint_from_dict(reseal(forged))


def test_resealed_predictor_stress_cannot_be_used_to_invent_damage_evolution(first):
    _, _, _, packet = first
    forged = deepcopy(packet)
    forged["runtime_state"]["last_report"]["fault_evolution"]["predictor_driver"]["stress_pa"][0][0] *= 2
    with pytest.raises(ValueError, match="replay"):
        volume_fault_checkpoint_from_dict(reseal(forged))


def test_damage_policy_forgery_is_refused_before_replay(first):
    _, _, _, packet = first
    forged = deepcopy(packet)
    forged["runtime_state"]["damage_coupling"] = "ignored_damage"
    with pytest.raises(ValueError, match="semantics"):
        volume_fault_checkpoint_from_dict(reseal(forged))


def test_duplicate_json_history_key_is_refused(first):
    _, _, _, packet = first
    raw = json.dumps(packet).replace('"damage_coupling": ', '"damage_coupling": "forged", "damage_coupling": ', 1)
    with pytest.raises(ValueError, match="Duplicate"):
        load_volume_fault_checkpoint(io.StringIO(raw))


def test_wrong_expected_source_pin_refused(first):
    _, _, _, packet = first
    with pytest.raises(ValueError, match="binding"):
        volume_fault_checkpoint_from_dict(packet, expected_source_checkpoint_sha256='a'*64)


def test_calculated_activation_boundary_does_not_publish_a_fake_bank_or_step(first):
    # Explicit gate fixture tests refusal, not formation of a new planetary fault.
    model, _, after, _ = first
    d = np.full(80, .7)
    age = np.full(80, .05)
    f = replace(after.fault_history, damage=d, fault_candidate_age_myr=age)
    a, _ = model._assembly(after.geometry, d)
    energy = volume_mechanics_energy(after.mechanics, radial_moments(a),
        d[a.face_indices], model.shell_parameters)
    difference = energy-after.mechanics.elastic_energy_j
    m = replace(after.mechanics, elastic_energy_j=energy,
        damage_energy_change_j=after.mechanics.damage_energy_change_j+difference)
    candidate = replace(after, fault_history=f, mechanics=m)
    model.validate_state(candidate)
    with pytest.raises(VolumeFaultActivationBoundary) as result:
        model.step(candidate, 1e-8)
    event = result.value.activation_report
    assert event["time_myr"] == after.time_myr
    assert not event["sharp_banks_created"]
    assert result.value.candidate_history.fault_active.all()
    assert not after.fault_history.fault_active.any()
    assert after.mechanics.accepted_steps == 1


def test_frozen_old_control_checkpoint_format_is_still_supported(source):
    model = VolumeModel(source)
    state = model.initial_state()
    packet = volume_checkpoint_to_dict(model, state)
    assert packet["format"] == 'genesis-volume-continuation-checkpoint-0.1'
    resumed_model, restored = volume_checkpoint_from_dict(packet)
    assert _canonical(volume_checkpoint_to_dict(resumed_model, restored)) == _canonical(packet)
