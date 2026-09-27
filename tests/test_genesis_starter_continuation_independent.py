"""Independent import, clock, null-load and installed-hook checks."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.dynamics import DynamicsParameters, angular_velocity_vectors, update_plate_dynamics
from tectonics.genesis import GenesisParameters
from tectonics.genesis_mature import (
    load_experimental_genesis_mature_import, save_experimental_genesis_mature_import,
)
from tectonics.genesis_shell import ShellParameters
from tectonics.genesis_starter import StarterModel, StarterParameters
from tectonics.genesis_starter_continuation import (
    YoungWorldCoupling, _period, build_starter_continuation,
)
from tectonics.genesis_starter_material import primary_material_inventory
from tectonics.genesis_tides import SYNCHRONOUS_SPIN, TidalParameters, advance_tidal_orbit
from tectonics.mesh import build_icosphere
from tectonics.lithosphere import advance_lithosphere
from tectonics.plates import Plate, PlateSystem
from tectonics.simulation import load_config
from tectonics.transport import initialize_transport_state


@pytest.fixture(scope="module")
def source():
    model = StarterModel(build_icosphere(1), GenesisParameters(),
        TidalParameters(enabled=True, spin_state=SYNCHRONOUS_SPIN),
        ShellParameters(subdivisions=1, convective_traction_pa=50000.), StarterParameters())
    state = model.advance(model.initial_state(), 2.)
    assert state.stopped_reason == "first_partition"
    config = load_config(Path(__file__).resolve().parents[1]/"configs"/"canonical_moon.yaml")
    return model, state, config


def imported(source):
    model, state, config = source
    bundle, cfg, report = build_starter_continuation(model, deepcopy(state), config)
    return model, deepcopy(state), bundle, cfg, report


def test_actual_import_uses_surface_inventory_and_preserves_zero_continents(source):
    model, state, bundle, cfg, report = imported(source)
    material = primary_material_inventory(model, state)
    cp = bundle.checkpoint
    np.testing.assert_array_equal(cp.state.cell_plate, state.system.cell_plate)
    np.testing.assert_array_equal(cp.state.tidal_damage, state.damage)
    np.testing.assert_array_equal(cp.state.continental_volume_km3, 0.)
    np.testing.assert_array_equal(cp.state.continental_fraction, 0.)
    np.testing.assert_allclose(cp.state.oceanic_volume_km3,
        material.primary_mass_kg/(material.primary_density_kg_m3*1e9), rtol=2e-15)
    assert not np.isclose(report["initial_primary_thickness_km"], report["initial_mechanical_lid_thickness_km"])
    assert cp.plume_state.plume_ids.size == 0
    assert cp.cycle.cumulative_generated_volume_km3 == 0.
    assert cfg["lithosphere"]["initial_continental_fraction"] == 0.
    assert not bundle.report["physical_handoff_certified"]


def test_saved_import_roundtrips_through_existing_public_integrity_checked_loader(source, tmp_path):
    _, _, bundle, _, _ = imported(source)
    path = save_experimental_genesis_mature_import(tmp_path/"import", bundle)
    loaded = load_experimental_genesis_mature_import(path)
    assert loaded.checkpoint.events == json.loads(json.dumps(bundle.checkpoint.events))
    np.testing.assert_array_equal(loaded.checkpoint.state.oceanic_volume_km3,
                                  bundle.checkpoint.state.oceanic_volume_km3)


def test_unpartitioned_source_is_refused_without_manufacturing_initial_plates(source):
    model, _, config = source
    with pytest.raises(ValueError, match="first-partition"):
        build_starter_continuation(model, model.initial_state(), config)


def test_zero_mantle_and_zero_force_control_does_not_receive_a_velocity_floor(source):
    model, state, config = source
    quiet = StarterModel(model.mesh, model.thermal, model.tides,
                         replace(model.shell, convective_traction_pa=0.), model.parameters)
    state = deepcopy(state)
    for plate in state.system.plates:
        plate.angular_speed_rad_per_myr = 0.
    bundle, cfg, _ = build_starter_continuation(quiet, state, config)
    cp = bundle.checkpoint
    np.testing.assert_array_equal(cp.mantle_flow.cell_omega_rad_per_myr, 0.)
    # Explicitly zero force scale isolates the dynamics null-load invariant;
    # the legacy min_active setting must never create a positive rotation.
    parameters = DynamicsParameters(force_speed_scale_deg_per_myr=0., min_active_speed_deg_per_myr=1.)
    result, _, _, _ = update_plate_dynamics(
        quiet.mesh, cp.state, cp.system, cp.baseline, quiet.thermal.radius_km,
        1., 1., .1, parameters, mantle_flow=cp.mantle_flow,
        thermal_lithosphere_thickness_km=cp.thermal.thermal_lithosphere_thickness_km,
    )
    np.testing.assert_array_equal(angular_velocity_vectors(result), 0.)


def test_young_heat_continues_orbit_without_mutating_archived_first_partition(source):
    model, state, bundle, cfg, _ = imported(source)
    cp = bundle.checkpoint
    coupling = YoungWorldCoupling(model, state, cfg, cp)
    old_damage = coupling.source_state.damage.copy()
    old_energy = state.thermal_context.thermal.energy.copy()
    old_time = state.time_myr
    thermal, _ = coupling.advance_heat(cp.thermal, .2)
    assert thermal.time_myr == old_time + .2
    assert coupling.source_state.thermal_context.orbit.time_myr == thermal.time_myr
    # The first-partition archive is stable; live continued damage now resides
    # in coupling.fracture and is tested through the installed material hook.
    np.testing.assert_array_equal(coupling.source_state.damage, old_damage)
    np.testing.assert_array_equal(state.thermal_context.thermal.energy, old_energy)
    middle = old_time + .1
    expected = advance_tidal_orbit(state.thermal_context.orbit, model.tides, middle)[0].eccentricity
    assert coupling.at(middle) == expected
    assert coupling.at(old_time) == state.thermal_context.orbit.eccentricity
    with pytest.raises(ValueError, match="outside"):
        coupling.at(thermal.time_myr + .01)
    sample = model.loading.sample(coupling.source_state.thermal_context)
    assert abs(sample.thermal["relative_energy_residual"]) < 1e-10


def test_installed_hooks_follow_post_topology_velocities_transport_and_liquid_water(source):
    model, state, bundle, cfg, _ = imported(source)
    cp = bundle.checkpoint
    coupling = YoungWorldCoupling(model, state, cfg, cp)
    merged_state = deepcopy(cp.state)
    merged_state.cell_plate[:] = 0
    merged_system = PlateSystem(merged_state.cell_plate.copy(),
        (Plate(0, 0, np.array([0., 0., 1.]), .007),))
    remapped = initialize_transport_state(1)
    remapped.cumulative_commit_count = 11
    seen = {}

    class FakeManager:
        params = SimpleNamespace(split_enabled=False)

        def update(self, mesh, material, system, boundaries, radius_km, dt_myr):
            return merged_system, None, []

    def original_lithosphere(*args, **kwargs):
        seen["period"] = args[6]
        return advance_lithosphere(*args, **kwargs)

    def original_hydrosphere(mesh, material, topography, hydrosphere, *args, **kwargs):
        seen["water"] = hydrosphere.water_volume_km3
        return hydrosphere, None

    base = SimpleNamespace(
        update_plate_dynamics=lambda *args, **kwargs: (cp.system, None, None, None),
        advance_hydrosphere=original_hydrosphere, PlateTopologyManager=FakeManager,
        remap_transport_state=lambda *args, **kwargs: remapped,
    )
    runner = SimpleNamespace(base=base,
        v124=SimpleNamespace(_original_advance_lithosphere=original_lithosphere))
    coupling.install(runner)
    prototype = base.build_prototype(cfg)
    assert prototype.mesh is model.mesh
    np.testing.assert_array_equal(prototype.plates.cell_plate, cp.system.cell_plate)
    assert base.eccentricity_history_from_config(cfg) is coupling
    thermal, _ = base.advance_thermal_state(cp.thermal, .1)
    merged_state.time_myr = thermal.time_myr
    runner.v124._original_advance_lithosphere(model.mesh, cp.system, cp.state, .1,
        model.thermal.radius_km, model.thermal.surface_gravity_m_s2, -123., coupling,
        transport_state=cp.transport_state)
    assert seen["period"] == _period(model, coupling.source_state.thermal_context.orbit)
    base.update_plate_dynamics()
    base.PlateTopologyManager().update(model.mesh, merged_state, cp.system, [],
                                      model.thermal.radius_km, .1)
    base.remap_transport_state(cp.system, merged_system, cp.transport_state)
    # A stale pre-merge velocity would yield a different mean speed. The real
    # topology/transport hooks must select the returned one-plate system here.
    base.advance_hydrosphere(model.mesh, merged_state, cp.topo,
                            replace(cp.hydrosphere, water_volume_km3=0.))
    row = coupling.history[-1]
    expected_speed = np.linalg.norm(np.cross(np.array([0., 0., .007]), model.mesh.centroids), axis=1)*model.thermal.radius_km
    assert row["plate_count"] == 1 and row["transport_commits"] == 11
    assert row["mean_surface_speed_km_myr"] == pytest.approx(model.areas@expected_speed/model.areas.sum())
    fraction = model.loading.sample(coupling.source_state.thermal_context).thermal["ocean_fraction"]
    assert seen["water"] == pytest.approx(fraction*model.thermal.water_volume_km3)


def test_young_mechanical_thickness_is_not_reset_from_zero_chemical_ages(source):
    model, state, bundle, cfg, _ = imported(source)
    cp = bundle.checkpoint
    coupling = YoungWorldCoupling(model, state, cfg, cp)
    thermal, _ = coupling.advance_heat(cp.thermal, 1.)
    cp.state.time_myr = thermal.time_myr
    cp.state.crust_age_myr[:] = 0.
    coupling.mechanical_fields(cp.state)
    sample = model.loading.sample(coupling.source_state.thermal_context)
    expected = np.maximum(sample.lid_thickness_km-cp.state.crust_thickness_km, 0.)
    np.testing.assert_array_equal(cp.state.mantle_lithosphere_thickness_km, expected)
    assert np.max(cp.state.mantle_lithosphere_thickness_km) > 0.
