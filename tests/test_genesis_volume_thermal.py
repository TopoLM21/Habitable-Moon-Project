"""Real-source thermal continuation, independent fluxes and strict replay."""
from copy import deepcopy
from dataclasses import asdict, replace
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis import SECONDS_PER_MYR, temperatures
from tectonics.genesis_bulk_birth import bulk_birth_from_dict
from tectonics.genesis_shell import rock_temperature
from tectonics.genesis_volume_geometry import build_genesis_volume_geometry
from tectonics.genesis_volume_thermal import (
    BIRTH_CHART_SCOPE, VolumeThermalBoundaryParameters,
    initial_volume_thermal_state, prepare_volume_thermal_step,
    thermal_from_dict, thermal_to_dict,
)


@pytest.fixture(scope="module")
def actual_packet():
    path = Path(__file__).resolve().parents[1]/"analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def actual_source(actual_packet):
    return (initial_volume_thermal_state(actual_packet),
            bulk_birth_from_dict(actual_packet["bulk_birth_history"]),
            build_genesis_volume_geometry(actual_packet))


@pytest.fixture(scope="module")
def actual_step(actual_source):
    return prepare_volume_thermal_step(*actual_source, 1e-6)


def _reseal(raw):
    body = {key: value for key, value in raw.items() if key != "sha256"}
    raw["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def test_actual_source_clocks_profiles_and_ledgers_are_inherited(actual_packet, actual_source):
    state, history, geometry = actual_source
    with np.load(io.BytesIO(base64.b64decode(actual_packet["source"]["checkpoint_base64"])), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
    assert state.time_myr == state.initial_time_myr == history.time_myr == geometry.time_myr
    assert state.accepted_steps == 0 and state.source_accepted_intervals == 438
    np.testing.assert_array_equal(state.layer_mass_kg, actual_packet["current_columns"]["layer_mass_kg"])
    np.testing.assert_array_equal(state.column_enthalpy, actual_packet["current_columns"]["specific_enthalpy_j_kg"])
    np.testing.assert_array_equal(state.damage, actual_packet["current_shell_memory"]["damage"])
    assert asdict(state.global_state) == metadata["thermal_state"]
    assert asdict(state.orbit) == metadata["orbit"]
    for name in ("initial_column_energy_j", "boundary_energy_j", "tidal_heat_received_j"):
        assert getattr(state, name) == metadata["state"][name]
    assert not state.combined_heat_mechanical_energy_closed
    assert state.birth_chart_scope == BIRTH_CHART_SCOPE
    old_energy = state.global_state.energy[0]
    mutable_snapshot = state.global_state
    mutable_snapshot.energy[0] = 0.
    assert state.global_state.energy[0] == old_energy
    for array in (state.layer_mass_kg, state.column_enthalpy, state.damage):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_actual_one_year_step_has_real_births_and_one_shared_clock(actual_source, actual_step):
    before, history, geometry = actual_source
    step = actual_step
    assert step.state.time_myr == step.history.time_myr == step.geometry.time_myr
    assert step.state.global_state.time_myr == step.state.orbit.time_myr == step.state.time_myr
    assert step.dt_myr == step.state.time_myr-before.time_myr
    assert step.state.accepted_steps == 1 and step.history.accepted_intervals == 439
    assert len(step.geometry.cohorts) == 560 and len(step.newborn_cohort_ids) == 80
    assert len(history.cohorts) == 480 and len(geometry.cohorts) == 480
    assert before.accepted_steps == 0 and before.intervals == ()
    np.testing.assert_array_equal(before.layer_mass_kg, step.state.layer_mass_kg)
    assert np.any(step.state.column_enthalpy != before.column_enthalpy)
    np.testing.assert_array_equal(step.birth_temperature_k, np.full(560, history.solidus_k))
    assert step.geometry.vertices_m == geometry.vertices_m
    assert step.geometry.source_packet_json == geometry.source_packet_json
    assert all(v == 0 for v in step.removed_radial_moments.volume_m3)
    old_volume = dict(zip((c.id for c in geometry.cohorts), step.before_radial_moments.volume_m3))
    new_volume = dict(zip((c.id for c in step.geometry.cohorts), step.radial_moments.volume_m3))
    assert all(new_volume[identity] == value for identity, value in old_volume.items())


def test_column_fluxes_use_actual_top_chord_area_and_close_separate_ledger(actual_source, actual_step):
    before, _, geometry = actual_source
    step = actual_step
    p = before.source_shell_parameters
    thermal = before.source_boundary_parameters.thermal
    X = np.asarray(geometry.vertices_m)[np.asarray(geometry.faces)]
    area = .5*np.linalg.norm(np.cross(X[:, 1]-X[:, 0], X[:, 2]-X[:, 0]), axis=1)
    dz = before.layer_mass_kg/(p.density_kg_m3*area[:, None])
    stable = .2*p.density_kg_m3*thermal.silicate_heat_capacity_j_kg_k*dz.min()**2/p.conductivity_w_m_k
    seconds = step.dt_myr*SECONDS_PER_MYR
    assert seconds < stable  # Exactly one physical Euler conduction substep.
    old = rock_temperature(before.column_enthalpy, thermal)
    tm0, ts0 = temperatures(np.asarray(before.global_state.energy), thermal)
    tm1, ts1 = temperatures(np.asarray(step.state.global_state.energy), thermal)
    flux = np.empty((len(area), p.column_layers+1))
    flux[:, 0] = 2*p.conductivity_w_m_k*(.5*(ts0+ts1)-old[:, 0])/dz[:, 0]
    flux[:, -1] = 2*p.conductivity_w_m_k*(old[:, -1]-.5*(tm0+tm1))/dz[:, -1]
    flux[:, 1:-1] = p.conductivity_w_m_k*(old[:, :-1]-old[:, 1:])/(.5*(dz[:, :-1]+dz[:, 1:]))
    expected_delta = (flux[:, :-1]-flux[:, 1:])*area[:, None]*seconds/before.layer_mass_kg
    np.testing.assert_allclose(step.state.column_enthalpy,
        before.column_enthalpy+expected_delta, rtol=0, atol=0)
    expected_boundary = np.dot(area, flux[:, 0]-flux[:, -1])*seconds
    actual_delta = np.sum(before.layer_mass_kg*(step.state.column_enthalpy-before.column_enthalpy))
    ledger_delta = step.state.boundary_energy_j-before.boundary_energy_j
    assert actual_delta == pytest.approx(expected_boundary, rel=2e-8)
    assert ledger_delta == pytest.approx(expected_boundary, rel=2e-8)
    source_spherical_area = np.asarray(json.loads(before.source_packet_json)["current_geometry"]["face_area_sr"])*(thermal.radius_km*1000)**2
    assert np.sum(area) < .95*np.sum(source_spherical_area)


def _profile_mean(profile, surface, mantle, layer_mass, lo, hi):
    bounds = np.r_[0., np.cumsum(layer_mass)/sum(layer_mass)]
    q = np.r_[0., .5*(bounds[:-1]+bounds[1:]), 1.]
    t = np.r_[surface, profile, mantle]
    partition = np.r_[lo, q[(q > lo) & (q < hi)], hi]
    value = np.interp(partition, q, t)
    return np.sum(np.diff(partition)*.5*(value[:-1]+value[1:]))/(hi-lo)


def test_temperatures_are_independent_profile_integrals_on_after_live_intervals(actual_source, actual_step):
    state, _, _ = actual_source
    thermal = state.source_boundary_parameters.thermal
    old = rock_temperature(state.column_enthalpy, thermal)
    new = rock_temperature(actual_step.state.column_enthalpy, thermal)
    tm0, ts0 = temperatures(np.asarray(state.global_state.energy), thermal)
    tm1, ts1 = temperatures(np.asarray(actual_step.state.global_state.energy), thermal)
    for index in (0, 65, 237, 479, 480, 559):
        c = actual_step.geometry.cohorts[index]
        mass = state.layer_mass_kg[c.face_index]
        expected0 = _profile_mean(old[c.face_index], ts0, tm0, mass, c.q_lo, c.live_q_hi)
        expected1 = _profile_mean(new[c.face_index], ts1, tm1, mass, c.q_lo, c.live_q_hi)
        assert actual_step.before_temperature_k[index] == pytest.approx(expected0, rel=3e-12)
        assert actual_step.after_temperature_k[index] == pytest.approx(expected1, rel=3e-12)


def test_actual_deformed_surface_affects_future_flux_and_activation_but_not_old_reference(actual_source, actual_step):
    state, history, geometry = actual_source
    X = np.asarray(geometry.vertices_m)
    q = np.zeros((len(X), 6))
    q[:, :3] = 2e-4*X
    q[:, 3:] = q[:, :3]
    moved = prepare_volume_thermal_step(state, history, geometry, 1e-6, displacement_m=q.ravel())
    assert np.any(moved.state.column_enthalpy != actual_step.state.column_enthalpy)
    assert moved.geometry.vertices_m == geometry.vertices_m
    faces = np.asarray(geometry.faces)
    for index, c in enumerate(moved.geometry.cohorts):
        np.testing.assert_array_equal(moved.activation_top_triangles_m[index], (X+q[:, :3])[faces[c.face_index]])
    np.testing.assert_array_equal(state.column_enthalpy,
        json.loads(state.source_packet_json)["current_columns"]["specific_enthalpy_j_kg"])
    assert moved.state.intervals[0].surface_vertices_m != actual_step.state.intervals[0].surface_vertices_m


def test_actual_restart_replays_all_thermal_inputs_then_continues_exactly(actual_packet, actual_step):
    raw = thermal_to_dict(actual_step.state, include_source=False)
    assert raw["source_packet"] is None
    restored = thermal_from_dict(json.loads(json.dumps(raw)), source_packet=actual_packet)
    assert thermal_to_dict(restored, include_source=False) == raw
    direct = prepare_volume_thermal_step(actual_step.state, actual_step.history, actual_step.geometry, 1e-6)
    resumed = prepare_volume_thermal_step(restored, actual_step.history, actual_step.geometry, 1e-6)
    assert thermal_to_dict(direct.state, include_source=False) == thermal_to_dict(resumed.state, include_source=False)
    assert direct.history == resumed.history and direct.geometry == resumed.geometry
    np.testing.assert_array_equal(direct.after_temperature_k, resumed.after_temperature_k)
    assert len(direct.geometry.cohorts) == 640


@pytest.mark.parametrize("mutation", ("enthalpy_and_ledger", "clock", "counter", "mass", "source", "extra", "lost_prefix"))
def test_jointly_resealed_false_thermal_state_is_rejected_by_full_source_replay(actual_packet, actual_step, mutation):
    raw = deepcopy(thermal_to_dict(actual_step.state, include_source=False))
    if mutation == "enthalpy_and_ledger":
        raw["column_enthalpy"][0][0] += 1.
        raw["boundary_energy_j"] += raw["layer_mass_kg"][0][0]
    elif mutation == "clock":
        raw["global_state"]["time_myr"] += 1e-6
    elif mutation == "counter":
        raw["accepted_steps"] += 1
    elif mutation == "mass":
        raw["layer_mass_kg"][0][0] *= 1.001
    elif mutation == "source":
        raw["source_checkpoint_sha256"] = "0"*64
    elif mutation == "extra":
        raw["unrequested"] = 0
    else:
        raw["intervals"][0]["before_time_myr"] = 0.
    _reseal(raw)
    with pytest.raises(ValueError):
        thermal_from_dict(raw, source_packet=actual_packet)


@pytest.mark.parametrize("dt", (0., -1., float("nan"), float("inf"), True))
def test_invalid_trial_is_refused_without_mutating_accepted_state(actual_source, dt):
    state, history, geometry = actual_source
    original = state.column_enthalpy.copy()
    with pytest.raises(ValueError):
        prepare_volume_thermal_step(state, history, geometry, dt)
    np.testing.assert_array_equal(state.column_enthalpy, original)
    assert not state.intervals


def test_wrong_physics_and_missing_restart_source_are_refused(actual_source, actual_step):
    state, history, geometry = actual_source
    with pytest.raises(ValueError, match="shell parameters"):
        prepare_volume_thermal_step(state, history, geometry, 1e-6,
            replace(state.source_shell_parameters, conductivity_w_m_k=4.))
    boundary = state.source_boundary_parameters
    with pytest.raises(ValueError, match="boundary parameters"):
        prepare_volume_thermal_step(state, history, geometry, 1e-6,
            source_boundary_parameters=VolumeThermalBoundaryParameters(
                replace(boundary.thermal, radius_km=boundary.thermal.radius_km+1), boundary.tides))
    with pytest.raises(ValueError, match="embedded source"):
        thermal_from_dict(thermal_to_dict(actual_step.state, include_source=False))


def test_embedded_source_roundtrip_and_no_unmodelled_raft_strain(actual_packet, actual_source):
    state, _, _ = actual_source
    packet = thermal_to_dict(state)
    restored = thermal_from_dict(packet)
    assert thermal_to_dict(restored) == packet
    assert all(value is None for value in actual_packet["unknown_fields"].values())
