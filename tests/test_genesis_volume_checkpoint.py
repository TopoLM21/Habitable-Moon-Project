"""Actual-origin restart and adversarial history checks for volume saves."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis_volume_checkpoint import (
    _validate_runtime_schema, load_volume_checkpoint, save_volume_checkpoint,
    volume_checkpoint_from_dict, volume_checkpoint_to_dict,
)
from tectonics.genesis_volume_dynamics import VolumeModel


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def reseal(packet):
    body = {key: value for key, value in packet.items() if key != "checkpoint_sha256"}
    packet["checkpoint_sha256"] = hashlib.sha256(canonical(body)).hexdigest()
    return packet


@pytest.fixture(scope="module")
def actual_source():
    path = Path(__file__).resolve().parents[1]/(
        "analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json")
    before = path.read_bytes()
    yield json.loads(before)
    assert path.read_bytes() == before


@pytest.fixture(scope="module")
def initial_model(actual_source):
    model = VolumeModel(actual_source)
    return model, model.initial_state()


@pytest.fixture(scope="module")
def first_checkpoint(initial_model):
    model, initial = initial_model
    first = model.step(initial, 1e-8)  # actual .01 year, not a frozen mechanics fixture
    return model, first, volume_checkpoint_to_dict(model, first)


def test_initial_checkpoint_preserves_nonzero_computed_source_memory(initial_model, actual_source):
    model, initial = initial_model
    packet = volume_checkpoint_to_dict(model, initial)
    loaded_model, loaded = volume_checkpoint_from_dict(packet)
    assert loaded.mechanics.initial_energy_j > 4e19
    assert loaded.mechanics.elastic_energy_j == initial.mechanics.elastic_energy_j
    assert loaded.history == initial.history
    assert loaded.geometry == initial.geometry
    assert loaded.accepted_step_sizes_myr == ()
    assert canonical(volume_checkpoint_to_dict(loaded_model, loaded)) == canonical(packet)
    assert loaded.mechanics.initial_energy_j == pytest.approx(
        actual_source["instantaneous_cohort_plane_stress_3d"]["elastic_energy_j"], rel=3e-15)
    assert not loaded.geometry.stress_free_reinitialization


def test_actual_two_steps_equal_first_step_restart_exactly(first_checkpoint):
    model, first, saved = first_checkpoint
    resumed_model, restored = volume_checkpoint_from_dict(saved,
        expected_source_checkpoint_sha256=model.source_packet["source"]["checkpoint_sha256"])
    np.testing.assert_array_equal(restored.displacement_m, first.displacement_m)
    np.testing.assert_array_equal(restored.velocity_m_s, first.velocity_m_s)
    assert restored.mechanics == first.mechanics
    assert restored.last_report == first.last_report
    assert restored.last_certificate == first.last_certificate
    direct = model.step(first, 1e-8)
    resumed = resumed_model.step(restored, 1e-8)
    assert canonical(volume_checkpoint_to_dict(model, direct)) == canonical(
        volume_checkpoint_to_dict(resumed_model, resumed))
    assert len(resumed.accepted_step_sizes_myr) == 2


def test_file_and_text_stream_roundtrip(first_checkpoint, tmp_path):
    model, state, expected = first_checkpoint
    path = tmp_path/"continued.json"
    save_volume_checkpoint(path, model, state)
    loaded_model, loaded = load_volume_checkpoint(path)
    assert canonical(volume_checkpoint_to_dict(loaded_model, loaded)) == canonical(expected)
    stream = io.StringIO()
    save_volume_checkpoint(stream, model, state)
    stream.seek(0)
    loaded_model, loaded = load_volume_checkpoint(stream)
    assert canonical(volume_checkpoint_to_dict(loaded_model, loaded)) == canonical(expected)
    assert path.read_text().count('"checkpoint_base64"') == 1


def test_checksum_change_and_wrong_external_source_binding_refused(first_checkpoint):
    _, _, packet = first_checkpoint
    forged = deepcopy(packet)
    forged["runtime_state"]["velocity_m_s"][0] += .1
    with pytest.raises(ValueError, match="seal"):
        volume_checkpoint_from_dict(forged)
    with pytest.raises(ValueError, match="binding"):
        volume_checkpoint_from_dict(packet, expected_source_checkpoint_sha256="a"*64)


def test_resealed_balanced_work_forgery_refused_by_whole_step_replay(first_checkpoint):
    _, _, packet = first_checkpoint
    forged = deepcopy(packet)
    memory = forged["runtime_state"]["mechanics"]
    # The total constitutive work balance remains unchanged. A checksum and
    # final elastic-energy comparison alone cannot detect this substituted past.
    memory["state"]["mechanical_work_j"] += 1e12
    memory["state"]["thermal_work_j"] -= 1e12
    body = {key: value for key, value in memory.items() if key != "packet_sha256"}
    memory["packet_sha256"] = hashlib.sha256(canonical(body)).hexdigest()
    reseal(forged)
    with pytest.raises(ValueError):
        volume_checkpoint_from_dict(forged)


def test_resealed_changed_cohort_memory_refused(first_checkpoint):
    _, _, packet = first_checkpoint
    forged = deepcopy(packet)
    memory = forged["runtime_state"]["mechanics"]
    memory["state"]["cohorts"][0]["strain0"][0][0] += 1e-5
    body = {key: value for key, value in memory.items() if key != "packet_sha256"}
    memory["packet_sha256"] = hashlib.sha256(canonical(body)).hexdigest()
    reseal(forged)
    with pytest.raises(ValueError):
        volume_checkpoint_from_dict(forged)


@pytest.mark.parametrize("field", ("displacement_m", "velocity_m_s"))
def test_resealed_motion_forgery_refused(first_checkpoint, field):
    _, _, packet = first_checkpoint
    forged = deepcopy(packet)
    forged["runtime_state"][field][0] += 1.
    reseal(forged)
    with pytest.raises(ValueError):
        volume_checkpoint_from_dict(forged)


def test_resealed_changed_contact_parameters_refused(first_checkpoint):
    _, _, packet = first_checkpoint
    forged = deepcopy(packet)
    parameters = json.loads(forged["runtime_state"]["contact_parameters_json"])
    parameters["basal_drag_pa_s_m"] *= 1.01
    forged["runtime_state"]["contact_parameters_json"] = canonical(parameters).decode()
    reseal(forged)
    with pytest.raises(ValueError):
        volume_checkpoint_from_dict(forged)


def test_missing_runtime_step_cannot_promote_endpoint_to_initial_state(first_checkpoint):
    _, _, packet = first_checkpoint
    forged = deepcopy(packet)
    forged["runtime_state"]["accepted_step_sizes_myr"] = []
    reseal(forged)
    with pytest.raises(ValueError):
        volume_checkpoint_from_dict(forged)


def test_duplicate_or_nonfinite_json_is_refused_before_loading():
    with pytest.raises(ValueError, match="Duplicate"):
        load_volume_checkpoint(io.StringIO('{"format":"one","format":"two"}'))
    with pytest.raises(ValueError, match="Nonfinite"):
        load_volume_checkpoint(io.StringIO('{"format":NaN}'))


def test_resealed_extra_runtime_field_refused(first_checkpoint):
    _, _, packet = first_checkpoint
    forged = deepcopy(packet)
    forged["runtime_state"]["historical_natural_metric_was_recovered"] = True
    reseal(forged)
    with pytest.raises(ValueError, match="fields"):
        volume_checkpoint_from_dict(forged)


def test_physical_semantics_are_string_claims_and_refuse_substitution(first_checkpoint):
    model, state, packet = first_checkpoint
    runtime = packet["runtime_state"]
    assert _validate_runtime_schema(model, runtime) == state.accepted_step_sizes_myr
    for field in ("velocity_closure", "kinematics", "activation_closure"):
        assert type(runtime[field]) is str
        for malformed in (None, 0., "substituted_physical_history"):
            changed = {**runtime, field: malformed}
            with pytest.raises(ValueError, match="physical semantics"):
                _validate_runtime_schema(model, changed)
