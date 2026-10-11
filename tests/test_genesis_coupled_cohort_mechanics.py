"""Independent parallel-layer controls in the Coupled reference operator.

These are declared analytical mechanics fixtures. Their prospective cohort
history starts empty and is advanced explicitly; it is never presented as a
formation checkpoint or used to repair a missing producer history.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.genesis_bulk_birth import initial_bulk_birth, advance_bulk_birth
from tectonics.genesis_cohort_mechanics import (
    initial_cohort_mechanics, prepare_cohort_loading, evaluate_cohort_loading,
    accept_cohort_loading, cohort_mechanics_energy,
)
from tectonics.genesis_coupled import CoupledModel
from test_genesis_contact import _source_bytes


@pytest.fixture(scope="module")
def operator(tmp_path_factory):
    source = _source_bytes(tmp_path_factory.mktemp("cohort_operator"),
        traction_pa=0., elastic=(0., 0., 0.), active=False)[0]
    return CoupledModel(source)


def _loading(model):
    state = model.initial()
    p, mesh = model.source_model.p, model.original_mesh
    history = initial_bulk_birth(mesh, model.layer_mass_kg, model.radius_m/1000,
        {"purpose": "analytical parallel-layer mechanics control"}, model.source_model.thermal.solidus_k)
    memory = initial_cohort_mechanics(history)
    n, layers = model.layer_mass_kg.shape
    columns = np.broadcast_to(np.linspace(550., 1250., layers), (n, layers)).copy()
    born = advance_bulk_birth(history, before_front_fraction=np.zeros(n),
        after_front_fraction=np.ones(n), before_time_myr=0., after_time_myr=.0001,
        mesh=mesh, radius_km=model.radius_m/1000, solidus_k=model.source_model.thermal.solidus_k)
    gate = prepare_cohort_loading(memory, history, born,
        before_column_temperature=np.full_like(columns, 1700.), after_column_temperature=columns,
        before_surface_temperature_k=1700., after_surface_temperature_k=500.,
        before_mantle_temperature_k=1700., after_mantle_temperature_k=1300.,
        before_established=False, after_established=True, parameters=p,
        before_damage=state.damage, activation_mesh=mesh, activation_radius_km=model.radius_m/1000)
    initial_strain = np.tile([2e-5, 2e-5, 0.], (n, 1))
    activated = evaluate_cohort_loading(gate, initial_strain, None, state.damage)
    memory = accept_cohort_loading(gate, activated)
    after = advance_bulk_birth(born, before_front_fraction=np.ones(n),
        after_front_fraction=np.ones(n), before_time_myr=.0001, after_time_myr=.0501,
        mesh=mesh, radius_km=model.radius_m/1000, solidus_k=model.source_model.thermal.solidus_k)
    local = prepare_cohort_loading(memory, born, after,
        before_column_temperature=columns, after_column_temperature=columns,
        before_surface_temperature_k=500., after_surface_temperature_k=500.,
        before_mantle_temperature_k=1300., after_mantle_temperature_k=1300.,
        before_established=True, after_established=True, parameters=p,
        before_damage=state.damage, activation_mesh=mesh, activation_radius_km=model.radius_m/1000)
    loading = SimpleNamespace(dt_myr=.05, thermal_state=replace(model.source.thermal_state, time_myr=.0501),
        fraction=np.ones(n), depth_km=np.full(n, p.column_depth_km),
        memory=local.mean_memory, effective_b=local.effective_b,
        water_access=state.water_access.copy(), cohort_loading=local)
    return state, loading


def _force_from_cohorts(model, state, local):
    """Assemble directly from individual volume stress resultants."""
    geometry = model._geometry(state.cut_edges)
    p, membrane = model.source_model.p, geometry.membrane
    nu = p.poisson_ratio
    D = np.array([[1., nu, 0.], [nu, 1., 0.], [0., 0., (1-nu)/2]])/(1-nu**2)
    factor = p.young_modulus_pa*(p.residual_stiffness+(1-p.residual_stiffness)*(1-state.damage)**2)
    stresses = (local.memory@D.T)*factor[local.face_index, None]
    resultant = np.zeros((len(state.damage), 3))
    np.add.at(resultant, local.face_index, stresses*local.volume_m3[:, None])
    local_force = np.einsum("fai,fa->fi", membrane.b, resultant)/model.radius_m
    force = np.zeros(membrane.ndof)
    np.add.at(force, membrane.dofs.ravel(), local_force.ravel())
    return force


def test_depth_temperature_controls_reaction_and_retains_energy_variance(operator, monkeypatch):
    state, loading = _loading(operator)
    local = loading.cohort_loading
    assert np.ptp(local.maxwell_r) > .5
    reaction = _force_from_cohorts(operator, state, local)
    monkeypatch.setattr(operator, "_external", lambda geometry, depth: reaction.copy())
    history, elastic, stress, work, correction, _ = operator._solve(state, loading, state.damage)
    np.testing.assert_array_equal(history.displacement_m, state.contact.displacement_m)
    np.testing.assert_allclose(elastic, local.mean_memory, rtol=3e-15, atol=1e-18)
    assert work == 0. and correction == pytest.approx(0., abs=1.)
    response = evaluate_cohort_loading(local, np.zeros_like(elastic), None, state.damage)
    np.testing.assert_allclose(stress, response.stress, rtol=3e-15, atol=1e-7)
    memory = accept_cohort_loading(local, response)
    energy = operator._bulk_energy(elastic, state.damage, loading.fraction,
        cohort_mechanics=memory, birth_history=local.history)
    average_energy = operator._bulk_energy(elastic, state.damage, loading.fraction)
    assert energy > 1.05*average_energy
    assert energy == pytest.approx(cohort_mechanics_energy(memory, local.history,
        state.damage, operator.source_model.p), rel=1e-15)
    assert response.viscous_dissipation_j > 0
    assert abs(response.balance_residual_j)/energy < 3e-13


def test_free_geometry_equilibrates_layer_stress_without_erasing_depth_memory(operator):
    state, loading = _loading(operator)
    history, elastic, stress, work, correction, _ = operator._solve(state, loading, state.damage)
    local = loading.cohort_loading
    expected = -local.mean_memory[0, 0]/local.effective_b[0]
    assert history.displacement_m[-1] == pytest.approx(operator.radius_m*expected,
        rel=2e-10, abs=1e-6)
    np.testing.assert_allclose(elastic, 0., atol=1e-13, rtol=0.)
    np.testing.assert_allclose(stress, 0., atol=1e-2, rtol=0.)
    increment = operator._geometry(state.cut_edges)._strain(history.displacement_m-state.contact.displacement_m)
    response = evaluate_cohort_loading(local, increment, None, state.damage)
    assert np.min(response.elastic[:, 0]) < 0 < np.max(response.elastic[:, 0])
    assert response.energy_j > 0
    assert work == response.mechanical_work_j
    assert response.viscous_dissipation_j > 0
    assert abs(response.balance_residual_j)/max(response.energy_j, local.old_total_energy_j, abs(work)) < 3e-12
    assert history.equilibrium_residual <= operator.contact_parameters.equilibrium_tolerance
