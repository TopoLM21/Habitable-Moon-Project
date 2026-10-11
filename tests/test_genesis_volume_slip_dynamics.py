"""Actual inactive continuation and bounded top material-plane transport."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis_volume_dynamics import coefficient_arrays
from tectonics.genesis_volume_checkpoint import _canonical
from tectonics.genesis_volume_loading import volume_mechanics_to_dict
from tectonics.genesis_volume_slip_dynamics import (VolumeSlipModel,
    VolumeSlipMaterialBoundary, transport_top_face_plane, _lift_plane)


@pytest.fixture(scope='module')
def initial():
    path = Path(__file__).resolve().parents[1]/(
        'analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json')
    original = path.read_bytes()
    model = VolumeSlipModel(json.loads(original))
    yield model, model.initial_state()
    assert path.read_bytes() == original


@pytest.fixture(scope='module')
def actual100(initial):
    model, state = initial
    return model, state, model.step(state, 1e-4)


def test_actual_initial_history_and_energy_remain_literal(initial):
    model, state = initial
    assert state.mechanics.elastic_energy_j == 4.276852389239655e19
    assert len(state.mechanics.cohorts) == 480
    assert not state.fault_history.fault_active.any()
    assert not state.plane_normal_world.any()
    assert not state.slip_direction_world.any()
    assert state.physical_mechanical_work_j == 0
    assert state.fault_friction_dissipation_j == state.fault_viscous_dissipation_j == 0
    assert model.validate_state(state) is state


def test_real_100_year_interval_matches_preceding_sealed_control(actual100):
    _, _, after = actual100
    path = Path(__file__).resolve().parents[1]/(
        'analysis/genesis_volume_fault_validation/evidence/control/after_first.json')
    old = json.loads(path.read_text())['runtime_state']
    assert _canonical(volume_mechanics_to_dict(after.mechanics)) == _canonical(old['mechanics'])
    np.testing.assert_array_equal(after.displacement_m, old['displacement_m'])
    np.testing.assert_array_equal(after.velocity_m_s, old['velocity_m_s'])
    np.testing.assert_array_equal(after.fault_history.damage, old['fault_history']['damage'])
    assert after.mechanics.elastic_energy_j == 4.033056725693343e19
    assert after.fault_history.damage.max() == .03692826969004512


def test_private_predictor_does_not_double_count_new_material_or_work(actual100):
    model, before, after = actual100
    assert len(after.mechanics.cohorts) == 560
    assert len(after.accepted_step_sizes_myr) == 1
    assert after.fault_history.accepted_steps == after.mechanics.accepted_steps == 1
    assert after.history.accepted_intervals == before.history.accepted_intervals+1
    assert after.physical_mechanical_work_j == after.mechanics.mechanical_work_j
    assert after.fault_friction_dissipation_j == after.fault_viscous_dissipation_j == 0
    report = after.last_report['weak_plane_slip']
    assert report['activation_event'] is None
    assert not np.asarray(report['shear_increment']).any()
    assert not report['sharp_banks_created']
    assert not report['physical_fault_contact_admission_claimed']
    assert report['coupled_momentum_relative_residual'] < 1e-11
    assert model.validate_state(after) is after


def test_runtime_work_cannot_be_added_without_matching_history(actual100):
    model, _, state = actual100
    with pytest.raises(ValueError, match='runtime loss'):
        model.validate_state(replace(state, fault_friction_dissipation_j=1e12))


def test_physical_and_net_work_cannot_disagree(actual100):
    model, _, state = actual100
    with pytest.raises(ValueError, match='total/net'):
        model.validate_state(replace(state, physical_mechanical_work_j=state.physical_mechanical_work_j+1e12))


def test_thermal_baseline_cannot_be_replaced_by_runtime_damage(actual100):
    model, _, state = actual100
    thermal = replace(state.thermal, damage=state.fault_history.damage)
    with pytest.raises(ValueError, match='thermal damage baseline'):
        model.validate_state(replace(state, thermal=thermal))


def test_plane_vectors_are_immutable_and_bound_to_material_orientation(initial):
    model, state = initial
    with pytest.raises(ValueError):
        state.plane_normal_world.setflags(write=True)
    malformed = np.ones_like(state.plane_normal_world)
    with pytest.raises(ValueError, match='co-normal binding'):
        model.validate_state(replace(state, plane_normal_world=malformed))


def test_top_face_conormal_transport_includes_stretch_not_only_rotation(initial):
    _, state = initial
    vertices = np.asarray(state.geometry.vertices_m)
    active = np.ones(80, bool)
    normals = np.broadcast_to([1/np.sqrt(2), 1/np.sqrt(2)], (80, 2)).copy()
    stretch = np.diag([1.04, .98, 1.01])
    normal2, world, tangent = transport_top_face_plane(normals, active,
        vertices, vertices@stretch.T, state.geometry.faces)
    assert np.max(np.abs(normal2-normals)) > .001
    np.testing.assert_allclose(np.linalg.norm(world, axis=1), 1., rtol=0, atol=3e-15)
    np.testing.assert_allclose(np.sum(world*tangent, axis=1), 0., rtol=0, atol=3e-15)
    recovered, _ = _lift_plane(normal2, active, vertices@stretch.T, state.geometry.faces)
    np.testing.assert_array_equal(world, recovered)
    roundtrip, _, _ = transport_top_face_plane(normal2, active,
        vertices@stretch.T, vertices, state.geometry.faces)
    np.testing.assert_allclose(roundtrip, normals, rtol=0, atol=2e-15)


def test_top_transport_is_objective_under_whole_body_rotation(initial):
    _, state = initial
    vertices = np.asarray(state.geometry.vertices_m)
    active = np.ones(80, bool)
    normal = np.broadcast_to([.6, .8], (80, 2))
    angle = .37
    rotation = np.array([[np.cos(angle),-np.sin(angle),0],
                         [np.sin(angle),np.cos(angle),0],[0,0,1]])
    n0, m0 = _lift_plane(normal, active, vertices, state.geometry.faces)
    n2, n, m = transport_top_face_plane(normal, active, vertices,
                                      vertices@rotation.T, state.geometry.faces)
    np.testing.assert_allclose(n2, normal, rtol=0, atol=7e-16)
    np.testing.assert_allclose(n, n0@rotation.T, rtol=0, atol=7e-16)
    np.testing.assert_allclose(m, m0@rotation.T, rtol=0, atol=7e-16)


def test_full_retirement_refuses_seeded_mechanical_defect_without_publishing(initial, monkeypatch):
    # Explicit event fixture tests the guard, never authentic activation.
    import tectonics.genesis_volume_slip_dynamics as dynamics
    model, state = initial
    active = np.zeros(80, bool); active[0] = True
    normal = np.zeros((80, 2)); normal[0] = [1., 0.]
    times = np.full(80, -1.); times[0] = state.time_myr
    candidate = replace(state.fault_history, fault_active=active,
                        plane_normal=normal, activation_time_myr=times)
    monkeypatch.setattr(dynamics, 'prepare_volume_fault_activation',
                        lambda *args:(candidate, {'event':'synthetic_guard_fixture'}))
    monkeypatch.setattr(dynamics, 'retained_live_fraction', lambda *args:np.zeros(80))
    with pytest.raises(VolumeSlipMaterialBoundary, match='fully retired'):
        model.step(state, 1e-8)
    assert state.mechanics.accepted_steps == 0
    assert not state.fault_history.fault_active.any()
    assert np.max(np.abs(coefficient_arrays(state.mechanics)[0])) > 0
