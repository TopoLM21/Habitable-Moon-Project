"""Calculated source inheritance and prospective weakening, not seeded planets."""
from dataclasses import fields, replace
import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis_faults import load_fault_checkpoint
from tectonics.genesis_onset_support import NonlocalLoading, _arc
from tectonics.genesis_tides import advance_tidal_orbit
from tectonics.genesis_volume_fault_evolution import (
    VolumeFaultEvolutionStepLimit, initial_volume_fault_history,
    evaluate_volume_fault_evolution, prepare_volume_fault_activation,
    volume_fault_history_to_dict, volume_fault_history_from_dict,
    validate_volume_fault_history, _nonlocal_loading, _source)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def source():
    packet = json.loads((ROOT/"analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json").read_text())
    model, state, thermal, orbit, _ = load_fault_checkpoint(io.BytesIO(
        base64.b64decode(packet["source"]["checkpoint_base64"], validate=True)))
    return packet, model, state, thermal, orbit


@pytest.fixture(scope="module")
def history(source):
    return initial_volume_fault_history(source[0])


def inputs(source, history, *, dt=1e-6, stress=None):
    packet, model, state, _, orbit = source
    orbit = replace(orbit, time_myr=history.time_myr)
    after, _ = advance_tidal_orbit(orbit, model.tides_p, history.time_myr+dt)
    tensor = np.asarray(packet["instantaneous_plane_stress_3d"]["stress_local_pa"])
    if stress is None:
        stress = np.column_stack((tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 0, 1]))
    return dict(after_time_myr=after.time_myr,
        physical_vertices_m=np.asarray(packet["current_geometry"]["vertices_unit_xyz"])*state.radius_km*1000,
        face_stress_pa=stress, face_mean_temperature_k=np.full(80, 1389.9464437792699),
        retained_fraction=np.ones(80), active_material=np.ones(80, dtype=bool),
        surface_temperature_k=1379.89288755854, ocean_fraction=0.,
        orbit_before=orbit, orbit_after=after)


def test_actual_source_every_fault_array_and_totals_inherited_literally(source, history):
    _, _, state, _, _ = source
    for f in fields(history):
        value = getattr(history, f.name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(state, f.name), err_msg=f.name)
    assert history.friction_work_j == state.friction_work_j
    assert history.viscous_fault_work_j == state.viscous_fault_work_j
    assert history.first_fracture_time_myr == state.first_fracture_time_myr
    assert history.time_myr == history.initial_time_myr == state.time_myr
    assert history.accepted_steps == 0 and not history.fault_active.any()
    assert history.damage.max() == .03498505110952482


def test_all_arrays_own_immutable_bytes(history):
    for f in fields(history):
        a = getattr(history, f.name)
        if isinstance(a, np.ndarray):
            with pytest.raises(ValueError):
                a.setflags(write=True)


def test_actual_small_interval_evolves_damage_without_inventing_active_fault(source, history):
    before = volume_fault_history_to_dict(history)
    after, report = evaluate_volume_fault_evolution(history, source[0], **inputs(source, history))
    assert after.time_myr == inputs(source, history)["after_time_myr"]
    assert after.accepted_steps == 1 and after.damage.max() > history.damage.max()
    assert not after.fault_active.any() and not after.fault_candidate_age_myr.any()
    assert report["nonlocal_area_conservation_relative_residual"] < 3e-15
    assert not report["sharp_banks_created"]
    assert not report["source_spherical_filter_equals_current_chord_filter_claimed"]
    assert volume_fault_history_to_dict(history) == before
    for name in ("plane_normal", "activation_time_myr", "cumulative_shear", "signed_shear",
            "last_shear_increment", "friction_work_cell_j", "viscous_work_cell_j"):
        np.testing.assert_array_equal(getattr(after, name), getattr(history, name))


def test_current_area_filter_matches_independent_dense_graph_and_not_spherical_filter(source):
    c = _source(source[0])
    x = c.mesh.vertices*c.state.radius_km*1000
    # This deformed geometry is an algebraic filter fixture, not planet history.
    x = x*np.array([1.03, .98, 1.01])
    tri = x[c.mesh.faces]
    areas = .5*np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)/1e6
    values = np.linspace(.2, 40., c.mesh.cell_count)
    active = np.ones(c.mesh.cell_count, dtype=bool)
    result = _nonlocal_loading(c, values, active, areas)
    matrix = np.diag(areas)
    for a, b, u, v in c.mesh.shared_edges:
        edge = np.arctan2(np.linalg.norm(np.cross(c.mesh.vertices[u], c.mesh.vertices[v])),
            np.dot(c.mesh.vertices[u], c.mesh.vertices[v]))
        separation = np.arctan2(np.linalg.norm(np.cross(c.mesh.centroids[a], c.mesh.centroids[b])),
            np.dot(c.mesh.centroids[a], c.mesh.centroids[b]))
        weight = edge/separation*c.model.onset_p.regularization_km**2
        matrix[a, a] += weight; matrix[b, b] += weight
        matrix[a, b] -= weight; matrix[b, a] -= weight
    np.testing.assert_allclose(result, np.linalg.solve(matrix, areas*values), rtol=4e-15, atol=1e-13)
    assert np.dot(areas, result) == pytest.approx(np.dot(areas, values), rel=3e-15)
    legacy = NonlocalLoading(c.mesh, c.state.radius_km, c.model.onset_p.regularization_km).apply(values, active)
    assert np.max(np.abs(legacy-result)) > .01


def test_inactive_filter_boundary_has_no_flux(source):
    c = _source(source[0]); areas = np.ones(c.mesh.cell_count)
    active = np.zeros(c.mesh.cell_count, dtype=bool); active[[0, 25, 50]] = True
    values = np.arange(c.mesh.cell_count, dtype=float)+1
    result = _nonlocal_loading(c, values, active, areas)
    np.testing.assert_array_equal(result[~active], 0.)
    assert np.dot(areas, result) == pytest.approx(np.dot(areas[active], values[active]), rel=1e-15)
    np.testing.assert_array_equal(_nonlocal_loading(c, values, np.zeros_like(active), areas), 0.)


def test_water_requires_real_ocean_and_cool_surface_and_growth_dilutes(source, history):
    # Explicit constitutive unit fixture: supplied wet/cold boundaries are
    # not claimed to have arisen in this source's actual thermal trajectory.
    wet = replace(history, water_access=np.full(80, .2))
    args = inputs(source, wet, stress=np.zeros((80, 3)))
    args.update(retained_fraction=np.full(80, .5), surface_temperature_k=300.,
        face_mean_temperature_k=np.full(80, 700.), ocean_fraction=.1)
    after, _ = evaluate_volume_fault_evolution(wet, source[0], **args)
    assert np.all(after.water_access > .1)
    assert np.all(after.water_access < .101)
    args["ocean_fraction"] = 0.
    dry, _ = evaluate_volume_fault_evolution(wet, source[0], **args)
    expected = .1*np.exp(-1e-6/source[1].onset_p.loss_timescale_myr)
    np.testing.assert_allclose(dry.water_access, expected, rtol=1e-15)


def test_candidate_age_and_display_weak_duration_are_separate(source, history):
    # Deliberate high-damage constitutive fixture, not actual planet state.
    ready = replace(history, damage=np.full(80, .66), fault_candidate_age_myr=np.full(80, .001))
    args = inputs(source, ready, stress=np.zeros((80, 3)))
    args["face_mean_temperature_k"] = np.full(80, 700.)
    after, _ = evaluate_volume_fault_evolution(ready, source[0], **args)
    np.testing.assert_allclose(after.fault_candidate_age_myr, .001+1e-6, rtol=0, atol=3e-17)
    assert not after.fault_active.any()
    assert after.first_fracture_time_myr == after.time_myr
    args["retained_fraction"] = np.full(80, .9)
    interrupted, _ = evaluate_volume_fault_evolution(ready, source[0], **args)
    np.testing.assert_array_equal(interrupted.fault_candidate_age_myr, 0.)
    np.testing.assert_array_equal(interrupted.weak_duration_myr, 0.)


def test_activation_uses_actual_ready_gate_and_stress_without_mutating_accepted(source, history):
    stress = np.tile([2e7, -1e7, 3e6], (80, 1))
    same, report = prepare_volume_fault_activation(history, source[0], stress)
    assert same is history and report is None
    # Explicit timer/plane-selection fixture, not producer activation evidence.
    fixture = replace(history, damage=np.full(80, .66), fault_candidate_age_myr=np.full(80, .05))
    candidate, report = prepare_volume_fault_activation(fixture, source[0], stress)
    assert candidate.fault_active.all() and not fixture.fault_active.any()
    np.testing.assert_array_equal(candidate.activation_time_myr, fixture.time_myr)
    np.testing.assert_allclose(np.linalg.norm(candidate.plane_normal, axis=1), 1., rtol=0, atol=2e-15)
    assert not report["sharp_banks_created"]
    again, event = prepare_volume_fault_activation(candidate, source[0], -stress)
    assert again is candidate and event is None
    with pytest.raises(ValueError, match="future volume plastic"):
        evaluate_volume_fault_evolution(candidate, source[0], **inputs(source, candidate))


def test_rejected_damage_trial_leaves_all_accepted_arrays_and_timers_intact(source, history):
    before = volume_fault_history_to_dict(history)
    with pytest.raises(VolumeFaultEvolutionStepLimit):
        evaluate_volume_fault_evolution(history, source[0], **inputs(source, history, stress=np.full((80, 3), 1e10)))
    assert volume_fault_history_to_dict(history) == before


def test_strict_plain_json_roundtrip_preserves_literal_arrays(source, history):
    body = json.loads(json.dumps(volume_fault_history_to_dict(history), allow_nan=False))
    restored = volume_fault_history_from_dict(body, source[0])
    assert volume_fault_history_to_dict(restored) == body
    body["injected"] = True
    with pytest.raises(ValueError, match="schema"):
        volume_fault_history_from_dict(body, source[0])


@pytest.mark.parametrize("changes", [
    {"damage": np.full(80, -1.)}, {"water_access": np.full(80, 1.1)},
    {"accepted_steps": True}, {"fault_active": np.zeros(80, dtype=int)},
    {"activation_time_myr": np.zeros(80)}, {"plane_normal": np.ones((80, 2))},
    {"time_myr": float("nan")}, {"weak_duration_myr": np.full(80, 10.)},
    {"source_checkpoint_sha256": "bad"}])
def test_history_validation_rejects_invalid_material_state(history, changes):
    with pytest.raises(ValueError):
        replace(history, **changes)


@pytest.mark.parametrize("first", [True, -1., float("nan"), 10.])
def test_first_diffuse_fracture_clock_rejects_invalid_values(history, first):
    with pytest.raises(ValueError):
        replace(history, first_fracture_time_myr=first)


def test_history_source_binding_and_clocks_are_checked(source, history):
    mismatch = replace(history, source_identity="0"*64)
    with pytest.raises(ValueError, match="source"):
        validate_volume_fault_history(mismatch, source[0])
    args = inputs(source, history)
    args["orbit_before"] = replace(args["orbit_before"], time_myr=0.)
    with pytest.raises(ValueError, match="clocks"):
        evaluate_volume_fault_evolution(history, source[0], **args)


def test_inactive_face_unavailable_temperature_has_no_constitutive_driver(source, history):
    # Authenticated source with an explicit mask/temperature unit fixture;
    # this does not claim an actual remelting event in its planet trajectory.
    args = inputs(source, history)
    inactive = 12
    args["active_material"][inactive] = False
    args["face_mean_temperature_k"][inactive] = 0.
    args["retained_fraction"][inactive] = 0.
    args["face_stress_pa"][inactive] = [1e200, 1e200, 1e200]
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        after, report = evaluate_volume_fault_evolution(history, source[0], **args)
    assert after.damage[inactive] == after.water_access[inactive] == 0.
    assert after.fault_candidate_age_myr[inactive] == after.weak_duration_myr[inactive] == 0.
    assert after.peak_tensile_pa[inactive] == history.peak_tensile_pa[inactive]
    assert after.tidal_stress_mpa[inactive] == 0.
    assert report["raw_loading_per_myr"][inactive] == 0.
    assert report["nonlocal_loading_per_myr"][inactive] == 0.
    assert report["healing_per_myr"][inactive] == 0.
    assert report["mean_solid_temperature_available"][inactive] is False
    assert not after.fault_active.any()
    assert after.friction_work_j == history.friction_work_j
    assert after.viscous_fault_work_j == history.viscous_fault_work_j
    np.testing.assert_array_equal(after.cumulative_shear, history.cumulative_shear)


def test_active_face_zero_temperature_is_rejected(source, history):
    args = inputs(source, history)
    args["face_mean_temperature_k"][0] = 0.
    assert args["active_material"][0]
    with pytest.raises(ValueError, match="positive temperatures on active"):
        evaluate_volume_fault_evolution(history, source[0], **args)
