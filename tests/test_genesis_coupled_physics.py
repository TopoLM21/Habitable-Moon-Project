"""Analytical mechanical limits of the coupled incremental contact solver."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.genesis import SECONDS_PER_MYR
from tectonics.genesis_coupled import CoupledModel
from tectonics.genesis_shell import maxwell_factors
from test_genesis_contact import _source_bytes


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return _source_bytes(tmp_path_factory.mktemp("coupled_physics"),
                         traction_pa=0., elastic=(0., 0., 0.), active=False)[0]


def _loading(model, state, *, x=1., delta_temperature=0., dt_myr=.0001):
    """Prescribe a uniform constitutive interval, independent of climate."""
    count = model.original_mesh.cell_count
    r, b = maxwell_factors(dt_myr*SECONDS_PER_MYR,
                         np.full(count, dt_myr*SECONDS_PER_MYR/x))
    memory = r[:, None]*state.elastic_strain
    memory[:, :2] -= (b*model.source_model.p.linear_expansion_per_k*delta_temperature)[:, None]
    return SimpleNamespace(dt_myr=dt_myr,
        thermal_state=replace(model.source.thermal_state, time_myr=state.time_myr+dt_myr),
        fraction=np.ones(count), depth_km=np.full(count, model.source_model.p.column_depth_km),
        memory=memory, effective_b=b, water_access=state.water_access.copy())


def _internal_force(model, state, elastic):
    geometry = model._geometry(state.cut_edges)
    membrane, p = geometry.membrane, model.source_model.p
    degradation = p.residual_stiffness+(1-p.residual_stiffness)*(1-state.damage)**2
    stress = (elastic@membrane.d.T)*(p.young_modulus_pa*degradation[:, None])
    volume = model.layer_mass_kg.sum(axis=1)/p.density_kg_m3
    local = np.einsum("fai,fa,f->fi", membrane.b, stress, volume/model.radius_m)
    force = np.zeros(membrane.ndof)
    np.add.at(force, membrane.dofs.ravel(), local.ravel())
    return force


@pytest.mark.parametrize("x", [1e-5, .1, 1., 4.])
def test_free_uniform_cooling_contracts_radius_without_creating_stress(source, x):
    model = CoupledModel(source)
    state = model.initial()
    assert len(state.cut_edges) == 0
    # A previous stress-free contraction ensures the law acts on delta q;
    # applying a new Maxwell factor to total displacement would fail this.
    state.contact.displacement_m[-1] = -50.
    delta_temperature = -2.
    loading = _loading(model, state, x=x, delta_temperature=delta_temperature)
    history, elastic, stress, _, _, _ = model._solve(state, loading, state.damage)
    contraction = model.radius_m*model.source_model.p.linear_expansion_per_k*delta_temperature
    assert history.displacement_m[-1] == pytest.approx(-50.+contraction, rel=1e-10, abs=1e-8)
    np.testing.assert_allclose(history.displacement_m[:-1], 0., atol=1e-7, rtol=0.)
    np.testing.assert_allclose(elastic, 0., atol=2e-14, rtol=0.)
    np.testing.assert_allclose(stress, 0., atol=1e-3, rtol=0.)
    assert history.equilibrium_residual <= model.contact_parameters.equilibrium_tolerance
    assert state.contact.displacement_m[-1] == -50.


@pytest.mark.parametrize("x", [.01, .5, 2.])
def test_fixed_strain_reaction_holds_geometry_while_maxwell_memory_relaxes(source, monkeypatch, x):
    model = CoupledModel(source)
    state = model.initial()
    initial = np.tile([2e-5, -1e-5, 3e-5], (model.original_mesh.cell_count, 1))
    state = replace(state, elastic_strain=initial)
    loading = _loading(model, state, x=x)
    # The external reaction balances the relaxed stress at zero displacement.
    reaction = _internal_force(model, state, initial*np.exp(-x))
    monkeypatch.setattr(model, "_external", lambda geometry, depth: reaction.copy())
    history, elastic, stress, work, loss, _ = model._solve(state, loading, state.damage)
    np.testing.assert_array_equal(history.displacement_m, state.contact.displacement_m)
    np.testing.assert_allclose(elastic, initial*np.exp(-x), atol=0., rtol=3e-16)
    assert work == 0. and loss == 0.
    old_energy = model._bulk_energy(initial, state.damage, loading.fraction)
    new_energy = model._bulk_energy(elastic, state.damage, loading.fraction)
    assert new_energy == pytest.approx(old_energy*np.exp(-2*x), rel=1e-14)
    assert np.max(np.abs(stress)) > 0


@pytest.mark.parametrize("x", [.05, .5, 2.])
def test_constant_uniform_stress_produces_analytical_maxwell_creep(source, monkeypatch, x):
    model = CoupledModel(source)
    state = model.initial()
    strain = 2e-5
    initial = np.tile([strain, strain, 0.], (model.original_mesh.cell_count, 1))
    state = replace(state, elastic_strain=initial)
    state.contact.displacement_m[-1] = -25.
    loading = _loading(model, state, x=x)
    force = _internal_force(model, state, initial)
    monkeypatch.setattr(model, "_external", lambda geometry, depth: force.copy())
    history, elastic, _, _, _, _ = model._solve(state, loading, state.damage)
    # Maxwell creep under constant stress: delta strain = sigma/E * dt/tau.
    expected_increment = model.radius_m*strain*x
    assert history.displacement_m[-1] == pytest.approx(-25.+expected_increment, rel=2e-10, abs=1e-7)
    np.testing.assert_allclose(elastic, initial, atol=1e-13, rtol=0.)
    np.testing.assert_allclose(history.displacement_m[:-1], 0., atol=1e-7, rtol=0.)
    assert history.equilibrium_residual <= model.contact_parameters.equilibrium_tolerance


def test_signed_loading_correction_is_not_standalone_dissipation(source, monkeypatch):
    model = CoupledModel(source)
    state = model.initial()
    initial = np.tile([2e-5, 2e-5, 0.], (model.original_mesh.cell_count, 1))
    state = replace(state, elastic_strain=initial)
    loading = _loading(model, state, x=1.)
    # An unloading interval that leaves a smaller tensile elastic strain.
    target_elastic = .1*initial
    force = _internal_force(model, state, target_elastic)
    monkeypatch.setattr(model, "_external", lambda geometry, depth: force.copy())
    _, elastic, _, work, correction, _ = model._solve(state, loading, state.damage)
    np.testing.assert_allclose(elastic, target_elastic, atol=1e-13, rtol=0.)
    assert work < 0 and correction < 0
    predictor_release = (model._bulk_energy(initial, state.damage, loading.fraction)
                         -model._bulk_energy(loading.memory, state.damage, loading.fraction))
    assert predictor_release+correction > 0
