"""Physical and rollback controls for the thermal split-shell coupling."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from tectonics.genesis import (
    ENERGY_SCALE, SECONDS_PER_MYR, GenesisParameters, mantle_enthalpy, surface_enthalpy,
)
from tectonics.genesis_coupled_thermal import advance_thermal_loading, evolve_coupled_damage
from tectonics.genesis_faults import FaultModel
from tectonics.genesis_mobile import _RetryStep
from tectonics.genesis_onset import OnsetParameters
from tectonics.genesis_seams import split_mesh, rebuild_seam_mesh
from tectonics.genesis_shell import ShellParameters, rock_enthalpy
from tectonics.genesis_tides import TidalParameters


def _fixture(*, temperature=1000., surface=None, tides=False, viscosity=1e24):
    p = ShellParameters(subdivisions=1, initial_temperature_anomaly_k=0.,
                        viscosity_reference_pa_s=viscosity)
    model = FaultModel(p, GenesisParameters(), OnsetParameters(),
        TidalParameters(enabled=tides, spin_state="synchronous_zero_obliquity"))
    state, global_state, orbit = model.initial()
    energy = [mantle_enthalpy(temperature, model.thermal)/ENERGY_SCALE,
              surface_enthalpy(temperature if surface is None else surface, model.thermal)/ENERGY_SCALE,
              0., 0.]
    energy[3] = global_state.initial_total_energy-sum(energy[:2])
    global_state = replace(global_state, energy=energy)
    columns = np.full_like(state.column_enthalpy, rock_enthalpy(temperature, model.thermal))
    state = replace(state, column_enthalpy=columns,
        boundary_energy_j=float(np.sum(state.layer_mass_kg*columns))-state.initial_column_energy_j,
        elastic_strain=np.tile([.001, -.001, .002], (model.mesh.cell_count, 1)),
        membrane_established=True)
    # Every face is independent, including when its banks still coincide.
    topology = split_mesh(model.mesh, np.asarray(model.mesh.shared_edges)[:, 2:])
    return model, state, global_state, orbit, topology


def _advance(fixture, *, target=.0001, mesh=None, **kwargs):
    model, state, global_state, orbit, topology = fixture
    args = dict(mesh=topology.mesh if mesh is None else mesh, radius_km=state.radius_km,
        layer_mass_kg=state.layer_mass_kg, column_enthalpy=state.column_enthalpy,
        elastic_strain=state.elastic_strain, damage=state.damage, water_access=state.water_access,
        boundary_energy_j=state.boundary_energy_j, thermal_state=global_state,
        orbit=orbit, target_myr=target)
    args.update(kwargs)
    return advance_thermal_loading(model, **args)


def _constant_boundary(monkeypatch):
    def fixed(global_state, orbit, thermal, tides, target_myr, max_step_myr):
        from tectonics.genesis import diagnose
        result = replace(global_state, time_myr=target_myr)
        return result, replace(orbit, time_myr=target_myr), 0., [diagnose(result, thermal)]
    monkeypatch.setattr("tectonics.genesis_coupled_thermal.advance_orbit_thermal", fixed)


def test_shared_clocks_fixed_material_and_separate_conductive_energy_ledger():
    fixture = _fixture(tides=True)
    model, state, thermal, orbit, topology = fixture
    before = deepcopy((state, thermal, orbit, topology.mesh))
    result = _advance(fixture)
    assert result.thermal_state.time_myr == result.orbit.time_myr == .0001
    assert result.dt_myr == .0001
    assert result.orbit.eccentricity < orbit.eccentricity
    assert result.tidal_heat_received_j > 0
    assert result.orbit.dissipated_energy_j == pytest.approx(result.tidal_heat_received_j, rel=3e-12)
    old_energy = float(np.sum(state.layer_mass_kg*state.column_enthalpy))
    new_energy = float(np.sum(state.layer_mass_kg*result.column_enthalpy))
    boundary_delta = result.boundary_energy_j-state.boundary_energy_j
    assert abs((new_energy-old_energy)-boundary_delta)/old_energy < 4e-16
    assert boundary_delta < 0
    for name in ("layer_mass_kg", "column_enthalpy", "elastic_strain", "damage", "water_access"):
        np.testing.assert_array_equal(getattr(state, name), getattr(before[0], name))
    assert thermal == before[1] and orbit == before[2]
    np.testing.assert_array_equal(topology.mesh.vertices, before[3].vertices)
    np.testing.assert_array_equal(topology.mesh.faces, before[3].faces)
    assert len(topology.mesh.shared_edges) == 0


def test_conduction_uses_moved_face_areas_without_removing_or_creating_material():
    fixture = _fixture()
    model, state, _, _, topology = fixture
    vertices = topology.mesh.vertices.copy()
    triangle = topology.mesh.faces[0]
    center = topology.mesh.centroids[0]
    vertices[triangle] = .9*vertices[triangle]+.1*center
    vertices[triangle] /= np.linalg.norm(vertices[triangle], axis=1)[:, None]
    mesh = rebuild_seam_mesh(topology, vertices)
    result = _advance(fixture, mesh=mesh)
    area = mesh.physical_cell_areas_km2(state.radius_km)
    recovered_mass = result.column_depth_km*area*1e9*model.p.density_kg_m3
    np.testing.assert_allclose(recovered_mass, state.layer_mass_kg.sum(axis=1), rtol=3e-16)
    assert result.column_depth_km[0] > model.p.column_depth_km
    np.testing.assert_allclose(result.column_depth_km[1:], model.p.column_depth_km, rtol=4e-16)
    residual = np.sum(state.layer_mass_kg*result.column_enthalpy)-state.initial_column_energy_j-result.boundary_energy_j
    assert abs(residual)/state.initial_column_energy_j < 4e-16


def test_maxwell_relaxes_analytically_once_with_no_boundary_temperature_change(monkeypatch):
    _constant_boundary(monkeypatch)
    fixture = _fixture(viscosity=1e20)
    model, state, *_ = fixture
    dt = .0001
    result = _advance(fixture, target=dt)
    exact = np.exp(-dt*SECONDS_PER_MYR*model.p.young_modulus_pa/model.p.viscosity_reference_pa_s)
    assert 0 < exact < .2
    np.testing.assert_allclose(result.memory, exact*state.elastic_strain, rtol=3e-15)
    np.testing.assert_array_equal(result.column_enthalpy, state.column_enthalpy)
    np.testing.assert_array_equal(result.retained, 1.)
    np.testing.assert_array_equal(result.retained_thermal_strain, 0.)
    assert result.boundary_energy_j == state.boundary_energy_j


def test_newly_frozen_material_dilutes_damage_and_inherits_no_shear_memory(monkeypatch):
    _constant_boundary(monkeypatch)
    fixture = _fixture(temperature=1500., surface=500.)
    model, state, _, _, _ = fixture
    result = _advance(fixture, target=.001, damage=np.full(model.mesh.cell_count, .8))
    old_fraction, _, _ = model._phase(state.column_enthalpy, 500., 1500.)
    assert np.all(result.fraction > old_fraction)
    assert np.all(result.retained < 1.)
    # Mean damage integrated over the solid column is conserved on accretion;
    # the growing stress-free fraction contributes no inherited shear memory.
    np.testing.assert_allclose(result.damage0*result.fraction, .8*old_fraction, rtol=2e-16)
    np.testing.assert_allclose(result.memory[:, 2]*result.fraction,
        state.elastic_strain[:, 2]*old_fraction*result.maxwell_r, rtol=3e-16)


def test_early_thermal_event_shortens_every_loading_interval(monkeypatch):
    from tectonics.genesis import diagnose
    fixture = _fixture(viscosity=1e20)
    event_time = .00005
    def event(global_state, orbit, thermal, tides, target_myr, max_step_myr):
        result = replace(global_state, time_myr=event_time, stopped_reason="thermal_event_control")
        return result, replace(orbit, time_myr=event_time), 0., [diagnose(result, thermal)]
    monkeypatch.setattr("tectonics.genesis_coupled_thermal.advance_orbit_thermal", event)
    result = _advance(fixture, target=.01)
    assert result.thermal_state.time_myr == result.orbit.time_myr == result.dt_myr == event_time
    model, state, *_ = fixture
    exact = np.exp(-event_time*SECONDS_PER_MYR*model.p.young_modulus_pa/model.p.viscosity_reference_pa_s)
    np.testing.assert_allclose(result.memory, state.elastic_strain*exact, rtol=2e-15)


def test_cooling_creates_contraction_loading_without_secular_tidal_strain():
    without = _advance(_fixture())
    with_tides = _advance(_fixture(tides=True))
    assert np.all(without.retained_thermal_strain < 0)
    # Heat changes Maxwell/cooling by its tiny thermal contribution, whereas
    # the large periodic elastic tide is evaluated only by the damage helper.
    assert np.max(np.abs(with_tides.memory-without.memory)) < 1e-9
    assert with_tides.tidal_parameters.enabled
    np.testing.assert_allclose(without.memory[:, 2],
                               without.maxwell_r*.002, rtol=2e-15)


def test_water_access_requires_liquid_and_does_not_consume_global_water():
    liquid_fixture = _fixture(surface=500.)
    steam_fixture = _fixture(surface=1000.)
    n = liquid_fixture[0].mesh.cell_count
    liquid = _advance(liquid_fixture, damage=np.full(n, .8))
    steam = _advance(steam_fixture, damage=np.full(n, .8))
    assert liquid.ocean_fraction > .5 and liquid.water_access.min() > 0
    assert steam.ocean_fraction == 0
    np.testing.assert_array_equal(steam.water_access, 0.)
    assert liquid.rows[-1]["total_water_mass_kg"] == steam.rows[-1]["total_water_mass_kg"]


def test_healing_is_analytic_without_stress_or_tides(monkeypatch):
    _constant_boundary(monkeypatch)
    fixture = _fixture()
    model, _, _, _, topology = fixture
    n = model.mesh.cell_count
    loading = _advance(fixture, damage=np.full(n, .6))
    result, tide = evolve_coupled_damage(model, loading, topology.mesh,
                                        model.thermal.radius_km, np.zeros((n, 3)))
    healing = 1/model.p.cold_healing_timescale_myr+((1000.-700.)/650.)**4/model.p.hot_healing_timescale_myr
    np.testing.assert_allclose(result, .6*np.exp(-healing*loading.dt_myr), rtol=1e-15)
    np.testing.assert_array_equal(tide, 0.)


def test_nonlocal_damage_does_not_cross_disconnected_banks():
    fixture = _fixture()
    model, _, _, _, topology = fixture
    loading = _advance(fixture)
    stress = np.zeros((model.mesh.cell_count, 3))
    stress[0, :2] = 3*model.p.tensile_strength_pa
    isolated, _ = evolve_coupled_damage(model, loading, topology.mesh,
                                        model.thermal.radius_km, stress)
    closed, _ = evolve_coupled_damage(model, loading, model.mesh,
                                     model.thermal.radius_km, stress)
    assert isolated[0] > 0
    np.testing.assert_array_equal(isolated[1:], 0.)
    assert np.any(closed[1:] > 0)


def test_periodic_tides_supply_damage_loading_without_adding_to_memory():
    fixture = _fixture(tides=True)
    model, _, _, _, topology = fixture
    loading = _advance(fixture)
    copy = loading.memory.copy()
    damage, tidal_peak = evolve_coupled_damage(model, loading, topology.mesh,
        model.thermal.radius_km, np.zeros((model.mesh.cell_count, 3)))
    assert tidal_peak.max() > 0
    assert np.isfinite(damage).all()
    np.testing.assert_array_equal(copy, loading.memory)


def test_melting_cannot_silently_leave_an_unsupported_contact_shell():
    fixture = _fixture(temperature=2300.)
    before = fixture[1].column_enthalpy.copy()
    with pytest.raises(_RetryStep, match="coupled_partial_melt_limit"):
        _advance(fixture)
    np.testing.assert_array_equal(fixture[1].column_enthalpy, before)


def test_inconsistent_clocks_rejected_before_advancement():
    fixture = _fixture()
    with pytest.raises(ValueError, match="clocks"):
        _advance(fixture, orbit=replace(fixture[3], time_myr=.1))
