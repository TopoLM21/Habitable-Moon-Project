"""Primary material is a phase view of existing mass, not an injected crust."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from tectonics.genesis import (ENERGY_SCALE, GenesisParameters, SECONDS_PER_MYR,
                               mantle_enthalpy, surface_enthalpy)
from tectonics.genesis_shell import ShellParameters, mantle_traction
from tectonics.genesis_starter import StarterModel, StarterParameters
from tectonics.genesis_starter_material import primary_material_inventory, independent_mantle_omega
from tectonics.genesis_tides import TidalParameters, SYNCHRONOUS_SPIN
from tectonics.mesh import build_icosphere


def model(traction=20000., thermal=None):
    return StarterModel(build_icosphere(1), thermal or GenesisParameters(),
        TidalParameters(enabled=True, spin_state=SYNCHRONOUS_SPIN),
        ShellParameters(subdivisions=1, convective_traction_pa=traction), StarterParameters())


def thermal_state(owner, surface_k, mantle_k=1800.):
    """An exact thermal inventory fixture, without pretending to simulate its history."""
    state = owner.initial_state()
    source = state.thermal_context.thermal
    energy = [mantle_enthalpy(mantle_k, owner.thermal)/ENERGY_SCALE,
              surface_enthalpy(surface_k, owner.thermal)/ENERGY_SCALE]
    source.energy = energy+[0., source.initial_total_energy-sum(energy)]
    return state


def test_molten_start_has_zero_primary_material_and_no_liquid_ocean():
    owner = model()
    inventory = primary_material_inventory(owner, owner.initial_state())
    assert inventory.surface_solid_fraction == 0.
    assert not inventory.fully_solid_surface
    assert not np.any(inventory.primary_mass_kg)
    assert not np.any(inventory.primary_thickness_km)
    assert not np.any(inventory.primary_enthalpy_j)
    assert not np.any(inventory.liquid_water_mass_kg)
    assert inventory.vapor_water_mass_kg.sum() == pytest.approx(
        owner.thermal.water_volume_km3*1e9*owner.thermal.water_density_kg_m3)


@pytest.mark.parametrize("surface_k,solid", [(2300., 0.), (2000., 0.),
    (1850., .25), (1700., .5), (1550., .75), (1400., 1.), (500., 1.)])
def test_phase_partition_closes_mass_water_and_enthalpy(surface_k, solid):
    owner = model()
    state = thermal_state(owner, surface_k)
    inventory = primary_material_inventory(owner, state)
    areas = owner.areas*1e6
    assert inventory.surface_solid_fraction == pytest.approx(solid, abs=2e-15)
    np.testing.assert_allclose(inventory.primary_mass_kg,
        areas*owner.thermal.surface_column_kg_m2*solid, rtol=2e-14, atol=0)
    np.testing.assert_allclose(inventory.primary_mass_kg+inventory.surface_remaining_mass_kg,
        areas*owner.thermal.surface_column_kg_m2, rtol=2e-15)
    np.testing.assert_allclose(inventory.source_silicate_mass_kg,
        areas*owner.thermal.silicate_column_kg_m2, rtol=2e-15)
    np.testing.assert_allclose(inventory.primary_thickness_km,
        np.full(owner.mesh.cell_count, owner.thermal.surface_layer_depth_m/1000*solid),
        rtol=2e-14, atol=0)
    np.testing.assert_allclose(inventory.source_enthalpy_j,
        areas*sum(state.thermal_context.thermal.energy[:2])*ENERGY_SCALE, rtol=2e-14)
    assert abs(inventory.silicate_mass_relative_residual) < 1e-14
    assert abs(inventory.water_mass_relative_residual) < 1e-14
    assert abs(inventory.thermal_energy_relative_residual) < 1e-14
    assert inventory.fully_solid_surface == (solid == 1.)


def test_source_surface_mass_is_subtracted_from_mantle_not_added_to_planet():
    thin = model(thermal=replace(GenesisParameters(), surface_layer_depth_m=500.))
    thick = model(thermal=replace(GenesisParameters(), surface_layer_depth_m=2000.))
    a = primary_material_inventory(thin, thermal_state(thin, 500.))
    b = primary_material_inventory(thick, thermal_state(thick, 500.))
    np.testing.assert_allclose(a.source_silicate_mass_kg, b.source_silicate_mass_kg, rtol=2e-15)
    np.testing.assert_allclose(b.primary_mass_kg-a.primary_mass_kg,
        a.mantle_mass_kg-b.mantle_mass_kg, rtol=2e-12)
    np.testing.assert_allclose(b.primary_thickness_km, 2., rtol=2e-15, atol=0)
    np.testing.assert_allclose(a.primary_thickness_km, .5, rtol=2e-15, atol=0)


def test_passive_column_is_not_chemical_crust_or_second_global_energy_store():
    owner = model()
    before = thermal_state(owner, 500.)
    after = deepcopy(before)
    h = after.thermal_context.column_enthalpy*.5
    boundary = ((h-after.thermal_context.column_enthalpy).sum()*owner.loading.layer_mass_kg_m2
                +after.thermal_context.boundary_energy_j_m2)
    after.thermal_context = replace(after.thermal_context, column_enthalpy=h,
                                    boundary_energy_j_m2=boundary)
    a = primary_material_inventory(owner, before)
    b = primary_material_inventory(owner, after)
    assert a.mechanical_lid_thickness_km != b.mechanical_lid_thickness_km
    assert a.primary_thickness_km[0] != a.mechanical_lid_thickness_km
    for field in ("primary_mass_kg", "surface_remaining_mass_kg", "mantle_mass_kg",
                  "primary_thickness_km", "source_enthalpy_j", "source_silicate_mass_kg"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def test_condensation_moves_water_between_phases_without_inventing_inventory():
    owner = model()
    steam = primary_material_inventory(owner, thermal_state(owner, 1000.))
    ocean = primary_material_inventory(owner, thermal_state(owner, 500.))
    assert not np.any(steam.liquid_water_mass_kg)
    assert np.all(ocean.liquid_water_mass_kg > 0)
    assert np.all(ocean.vapor_water_mass_kg > 0)
    np.testing.assert_allclose(steam.liquid_water_mass_kg+steam.vapor_water_mass_kg,
        ocean.liquid_water_mass_kg+ocean.vapor_water_mass_kg, rtol=2e-15)


@pytest.fixture(scope="module")
def cooled():
    owner = model(0.)
    # A real cooling trajectory gives a finite lid without forcing a partition.
    state = owner.advance(owner.initial_state(), 2.)
    return owner, state


def test_independent_mantle_omega_reconstructs_prescribed_tangential_velocity(cooled):
    cold_owner, state = cooled
    owner = model()
    before = deepcopy(state)
    omega = independent_mantle_omega(owner, state)
    sample = owner.loading.sample(state.thermal_context)
    traction = mantle_traction(owner.mesh, replace(owner.shell, seed=owner.parameters.seed),
                               np.full(owner.mesh.cell_count, sample.lid_thickness_km))
    expected = traction[owner.mesh.faces].mean(axis=1)/owner.parameters.basal_drag_pa_s_m
    x = owner.mesh.centroids
    expected -= x*np.sum(expected*x, axis=1)[:, None]
    actual = np.cross(omega, x)*(owner.thermal.radius_km*1000)/SECONDS_PER_MYR
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-24)
    roundoff = 4*np.finfo(float).eps*np.max(np.linalg.norm(omega, axis=1))
    np.testing.assert_allclose(np.sum(omega*x, axis=1), 0., atol=roundoff)
    assert np.max(np.linalg.norm(actual, axis=1)) > 0
    assert state.thermal_context.thermal == before.thermal_context.thermal
    np.testing.assert_array_equal(state.thermal_context.column_enthalpy, before.thermal_context.column_enthalpy)
    assert not np.any(independent_mantle_omega(cold_owner, state))


def test_mantle_field_does_not_depend_on_domain_labels_or_plate_rotation(cooled):
    _, state = cooled
    owner = model()
    omega = independent_mantle_omega(owner, state)
    altered = deepcopy(state)
    altered.system.cell_plate[:] = np.arange(owner.mesh.cell_count)
    # This deliberately invalid mechanical partition must not be consulted by
    # an independent mantle driver; only its shared thermal context is needed.
    np.testing.assert_array_equal(omega, independent_mantle_omega(owner, altered))
    assert not np.any(independent_mantle_omega(owner, owner.initial_state()))


def test_dry_inventory_has_no_water_energy():
    owner = model(thermal=replace(GenesisParameters(), water_volume_km3=0.))
    inventory = primary_material_inventory(owner, thermal_state(owner, 500.))
    assert not np.any(inventory.liquid_water_mass_kg)
    assert not np.any(inventory.vapor_water_mass_kg)
    assert not np.any(inventory.water_enthalpy_j)
    assert inventory.water_mass_relative_residual == 0.
