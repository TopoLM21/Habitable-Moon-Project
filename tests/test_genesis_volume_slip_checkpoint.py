"""Literal-source, complete restart and resealed slip-history refusal checks."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis_volume_checkpoint import _canonical
from tectonics.genesis_volume_slip_checkpoint import (
    FORMAT, SCOPE, load_volume_slip_checkpoint, save_volume_slip_checkpoint,
    volume_slip_checkpoint_from_dict, volume_slip_checkpoint_to_dict)
from tectonics.genesis_volume_slip_dynamics import (
    MECHANICAL_WORK_CLOSURE, PLANE_TRANSPORT, SLIP_CLOSURE, VolumeSlipModel)


@pytest.fixture(scope="module")
def source():
    path = Path(__file__).resolve().parents[1] / (
        "analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json")
    original = path.read_bytes()
    yield json.loads(original)
    assert path.read_bytes() == original


@pytest.fixture(scope="module")
def initial(source):
    model = VolumeSlipModel(source)
    state = model.initial_state()
    return model, state, volume_slip_checkpoint_to_dict(model, state)


@pytest.fixture(scope="module")
def first(initial):
    model, before, _ = initial
    after = model.step(before, 1e-8)
    return model, before, after, volume_slip_checkpoint_to_dict(model, after)


def reseal(packet):
    body = {k: v for k, v in packet.items() if k != "checkpoint_sha256"}
    packet["checkpoint_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return packet


def test_initial_export_keeps_literal_genesis_memory_and_separate_zero_runtime_slip(initial, source):
    model, state, packet = initial
    assert packet["format"] == FORMAT == "genesis-volume-slip-checkpoint-0.1"
    assert packet["scope"] == SCOPE
    assert packet["source_packet"] == source
    assert packet["source_checkpoint_sha256"] == source["source"]["checkpoint_sha256"]
    assert state.mechanics.elastic_energy_j == source["cohort_mechanics"]["state"]["elastic_energy_j"]
    assert state.mechanics.elastic_energy_j > 4e19
    np.testing.assert_array_equal(state.fault_history.damage, source["current_shell_memory"]["damage"])
    assert not state.fault_history.fault_active.any()
    assert state.physical_mechanical_work_j == 0
    assert state.fault_friction_dissipation_j == 0
    assert state.fault_viscous_dissipation_j == 0
    np.testing.assert_array_equal(state.plane_normal_world, np.zeros((80, 3)))
    np.testing.assert_array_equal(state.slip_direction_world, np.zeros((80, 3)))
    assert state.slip_closure == SLIP_CLOSURE
    assert state.plane_transport == PLANE_TRANSPORT
    assert state.mechanical_work_closure == MECHANICAL_WORK_CLOSURE
    assert not state.geometry.stress_free_reinitialization
    assert model.validate_state(state) is state


def test_initial_checkpoint_full_source_replay_is_exact(initial):
    model, _, packet = initial
    restored_model, restored = volume_slip_checkpoint_from_dict(packet,
        expected_source_checkpoint_sha256=model.source_packet["source"]["checkpoint_sha256"])
    assert _canonical(volume_slip_checkpoint_to_dict(restored_model, restored)) == _canonical(packet)


def test_actual_interval_replay_and_next_interval_match_every_field_exactly(first):
    model, before, after, packet = first
    assert after.mechanics.accepted_steps == before.mechanics.accepted_steps + 1
    assert after.fault_history.accepted_steps == after.mechanics.accepted_steps
    assert after.time_myr == after.fault_history.time_myr
    assert after.fault_friction_dissipation_j == after.fault_viscous_dissipation_j == 0
    restored_model, restored = volume_slip_checkpoint_from_dict(packet,
        expected_source_checkpoint_sha256=model.source_packet["source"]["checkpoint_sha256"])
    assert _canonical(volume_slip_checkpoint_to_dict(restored_model, restored)) == _canonical(packet)
    direct, resumed = model.step(after, 1e-8), restored_model.step(restored, 1e-8)
    assert _canonical(volume_slip_checkpoint_to_dict(model, direct)) == _canonical(
        volume_slip_checkpoint_to_dict(restored_model, resumed))


def test_actual_interval_stream_and_atomic_file_round_trip(first, tmp_path):
    model, _, state, packet = first
    stream = io.StringIO()
    save_volume_slip_checkpoint(stream, model, state)
    stream.seek(0)
    resumed_model, restored = load_volume_slip_checkpoint(stream)
    assert _canonical(volume_slip_checkpoint_to_dict(resumed_model, restored)) == _canonical(packet)
    path = tmp_path / "actual.json"
    save_volume_slip_checkpoint(path, model, state)
    assert json.loads(path.read_text()) == packet
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("field", ["plane_normal_world", "slip_direction_world"])
def test_resealed_fabricated_world_plane_or_flow_is_refused(first, field):
    _, _, _, packet = first
    forged = deepcopy(packet)
    forged["runtime_state"][field][0][0] = 1.
    with pytest.raises(ValueError, match="replay"):
        volume_slip_checkpoint_from_dict(reseal(forged))


@pytest.mark.parametrize("field", [
    "physical_mechanical_work_j", "fault_friction_dissipation_j", "fault_viscous_dissipation_j"])
def test_resealed_fabricated_runtime_slip_ledger_is_refused(first, field):
    _, _, _, packet = first
    forged = deepcopy(packet)
    forged["runtime_state"][field] += 1e17
    with pytest.raises(ValueError, match="replay"):
        volume_slip_checkpoint_from_dict(reseal(forged))


@pytest.mark.parametrize("field", ["damage", "fault_candidate_age_myr", "peak_tensile_pa"])
def test_resealed_changed_evolution_history_is_refused(first, field):
    _, _, _, packet = first
    forged = deepcopy(packet)
    forged["runtime_state"]["fault_history"][field][0] += .01
    with pytest.raises(ValueError, match="replay"):
        volume_slip_checkpoint_from_dict(reseal(forged))


def test_resealed_invented_active_plane_cannot_enter_the_source_history(first):
    _, _, _, packet = first
    forged = deepcopy(packet)
    forged["runtime_state"]["fault_history"]["fault_active"][0] = True
    with pytest.raises(ValueError, match="replay"):
        volume_slip_checkpoint_from_dict(reseal(forged))


def test_resealed_predictor_driver_cannot_supply_a_replacement_stress_history(first):
    _, _, _, packet = first
    forged = deepcopy(packet)
    forged["runtime_state"]["last_report"]["fault_evolution"]["predictor_driver"]["stress_pa"][0][0] *= 2
    with pytest.raises(ValueError, match="replay"):
        volume_slip_checkpoint_from_dict(reseal(forged))


@pytest.mark.parametrize("field", [
    "damage_coupling", "slip_closure", "plane_transport", "mechanical_work_closure"])
def test_resealed_changed_constitutive_policy_is_refused(initial, field):
    _, _, packet = initial
    forged = deepcopy(packet)
    forged["runtime_state"][field] = "replacement_policy"
    with pytest.raises(ValueError, match="semantics"):
        volume_slip_checkpoint_from_dict(reseal(forged))


def test_resealed_changed_contact_drag_cannot_reuse_the_actual_motion_or_work(first):
    _, _, _, packet = first
    forged = deepcopy(packet)
    contact = json.loads(forged["runtime_state"]["contact_parameters_json"])
    contact["basal_drag_pa_s_m"] *= 2
    forged["runtime_state"]["contact_parameters_json"] = json.dumps(contact, sort_keys=True,
        separators=(",", ":"), allow_nan=False)
    with pytest.raises(ValueError, match="replay"):
        volume_slip_checkpoint_from_dict(reseal(forged))


def test_resealed_source_packet_cannot_replace_actual_embedded_genesis_damage(initial):
    _, _, packet = initial
    forged = deepcopy(packet)
    source = forged["source_packet"]
    source["current_shell_memory"]["damage"][0] += .1
    source["packet_sha256"] = hashlib.sha256(_canonical(
        {k: v for k, v in source.items() if k != "packet_sha256"})).hexdigest()
    with pytest.raises(ValueError, match="source replay"):
        volume_slip_checkpoint_from_dict(reseal(forged))


def test_external_source_pin_cannot_be_replaced_by_the_packet(initial):
    _, _, packet = initial
    with pytest.raises(ValueError, match="binding"):
        volume_slip_checkpoint_from_dict(packet, expected_source_checkpoint_sha256="a" * 64)


def test_duplicate_runtime_key_is_refused_before_seal_or_replay(initial):
    _, _, packet = initial
    encoded = json.dumps(packet).replace('"slip_closure": ',
        '"slip_closure": "forged", "slip_closure": ', 1)
    with pytest.raises(ValueError, match="Duplicate"):
        load_volume_slip_checkpoint(io.StringIO(encoded))


def test_duplicate_embedded_contact_key_is_refused(initial):
    _, _, packet = initial
    forged = deepcopy(packet)
    contact = forged["runtime_state"]["contact_parameters_json"]
    forged["runtime_state"]["contact_parameters_json"] = contact.replace('"basal_drag_pa_s_m":',
        '"basal_drag_pa_s_m":1e14,"basal_drag_pa_s_m":', 1)
    with pytest.raises(ValueError, match="Duplicate"):
        volume_slip_checkpoint_from_dict(reseal(forged))


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_number_is_refused(number):
    with pytest.raises(ValueError, match="Nonfinite"):
        load_volume_slip_checkpoint(io.StringIO('{"value":' + number + '}'))


@pytest.mark.parametrize("field,value", [
    ("physical_mechanical_work_j", True), ("plane_normal_world", [[0., 0., 0.]])])
def test_boolean_ledger_or_wrong_face_vector_shape_is_refused(initial, field, value):
    _, _, packet = initial
    forged = deepcopy(packet)
    forged["runtime_state"][field] = value
    with pytest.raises(ValueError, match="finite"):
        volume_slip_checkpoint_from_dict(reseal(forged))
