"""Independent physical controls for the coarse starter and its loading basis."""
from copy import deepcopy
from dataclasses import replace
import math

import numpy as np
import pytest

import tectonics.genesis_starter as starter_module
import tectonics.genesis_starter_loading as loading_module
from tectonics.genesis import GenesisParameters
from tectonics.genesis_material import face_frames
from tectonics.genesis_shell import ShellParameters
from tectonics.genesis_starter import StarterModel, StarterParameters, _principal
from tectonics.genesis_starter_loading import smooth_mantle_tensor
from tectonics.genesis_tides import SYNCHRONOUS_SPIN, TidalParameters
from tectonics.mesh import build_icosphere


def model(*, parameters=None, traction=20000.0):
    return StarterModel(
        build_icosphere(1), GenesisParameters(),
        TidalParameters(enabled=True, spin_state=SYNCHRONOUS_SPIN),
        ShellParameters(subdivisions=1, convective_traction_pa=traction),
        parameters or StarterParameters(),
    )


@pytest.mark.parametrize("seed", [20260927, 912])
def test_seeded_material_and_real_cooling_cannot_create_plates_without_loading(seed):
    owner = model(traction=0.0, parameters=StarterParameters(
        seed=seed, cooling_contrast_fraction=0.0, tidal_mechanics=False,
    ))
    initial = owner.initial_state()
    final = owner.advance(initial, 3.0)
    diagnosis = owner.diagnose(final)
    assert diagnosis["lid_thickness_km"] > 1.0
    assert diagnosis["surface_temperature_k"] < owner.thermal.initial_surface_temperature_k - 1000.0
    assert diagnosis["ocean_fraction"] > 0.0
    assert diagnosis["water_access_fraction"] > 0.0
    assert final.thermal_context.orbit.dissipated_energy_j > 0.0
    # Heating, water, seed-dependent strength and elapsed time alone do not
    # substitute for a mechanical load or a prescribed plate population.
    np.testing.assert_array_equal(final.damage, 0.0)
    np.testing.assert_array_equal(final.yield_ratio, 0.0)
    np.testing.assert_array_equal(final.cooling_stress_pa, 0.0)
    assert not np.any(final.eligible)
    assert len(final.system.plates) == 1 and not final.events
    assert final.first_fracture_time_myr is None
    assert final.time_myr == 3.0
    np.testing.assert_array_equal(initial.damage, 0.0)
    assert initial.time_myr == 0.0


def test_differential_cooling_stress_has_no_global_mean_and_does_not_move_mesh():
    owner = model(traction=0.0, parameters=StarterParameters(tidal_mechanics=False))
    before = {name: getattr(owner.mesh, name).copy()
              for name in ("vertices", "faces", "centroids", "areas_unit_sphere")}
    final = owner.advance(owner.initial_state(), 2.0)
    assert np.max(np.abs(final.cooling_stress_pa)) > 1.0
    average = float(owner.areas @ final.cooling_stress_pa / owner.areas.sum())
    assert abs(average) < 1e-8
    for name, field in before.items():
        np.testing.assert_array_equal(getattr(owner.mesh, name), field)
    assert final.thermal_context.thermal.time_myr == final.thermal_context.orbit.time_myr


def test_mantle_tensor_and_yield_invariants_do_not_depend_on_face_basis(monkeypatch):
    mesh = build_icosphere(2)
    frames = face_frames(mesh)
    original = smooth_mantle_tensor(mesh, 3491)
    angles = np.linspace(-2.5, 2.1, mesh.cell_count)
    c, s = np.cos(angles), np.sin(angles)
    rotated = np.stack((frames[:, :, 0]*c[:, None] + frames[:, :, 1]*s[:, None],
                        -frames[:, :, 0]*s[:, None] + frames[:, :, 1]*c[:, None]), axis=2)
    monkeypatch.setattr(loading_module, "face_frames", lambda selected: rotated)
    transformed = smooth_mantle_tensor(mesh, 3491)

    def world_tensor(values, basis):
        local = np.empty((mesh.cell_count, 2, 2))
        local[:, 0, 0], local[:, 1, 1] = values[:, 0], values[:, 1]
        local[:, 0, 1] = local[:, 1, 0] = values[:, 2]
        return np.einsum("nia,nab,njb->nij", basis, local, basis)

    np.testing.assert_allclose(world_tensor(original, frames), world_tensor(transformed, rotated),
                               rtol=0.0, atol=9e-16)
    for first, second in zip(_principal(original), _principal(transformed)):
        np.testing.assert_allclose(first, second, rtol=0.0, atol=9e-16)


def _fixed_material_sample(owner, time):
    sample = owner.loading.sample(owner.loading.initial())
    return replace(sample, thermal=dict(sample.thermal, time_myr=time,
                                        surface_temperature_k=300.0, ocean_fraction=0.0),
                   lid_thickness_km=5.0, mean_lid_temperature_k=700.0)


def _constant_strength_model():
    return model(traction=0.0, parameters=StarterParameters(
        cooling_contrast_fraction=0.0, strength_variation_fraction=0.0,
        damage_strength_reduction=0.0, hot_strength_fraction=1.0,
        water_weakening=False, regularization_km=0.0,
    ))


def test_zero_mean_tidal_cycle_drives_phase_averaged_irreversible_overstress(monkeypatch):
    owner = _constant_strength_model()
    strength = owner.shell.tensile_strength_pa
    cycle = np.zeros((2, owner.mesh.cell_count, 3))
    cycle[0, :, :2], cycle[1, :, :2] = 2.0 * strength, -2.0 * strength
    np.testing.assert_array_equal(cycle.mean(axis=0), 0.0)
    monkeypatch.setattr(starter_module, "tidal_stress_cycle", lambda *args, **kwargs: cycle)
    state = owner.initial_state()
    dt = 0.007
    owner._material_sample(state, _fixed_material_sample(owner, 0.0),
                           _fixed_material_sample(owner, dt))
    # Isotropic tension has ratio 2 for half the cycle; isotropic compression
    # has zero shear and no tensile loading for the other half.
    drive = 0.5 / owner.shell.damage_timescale_myr
    healing = 1.0 / owner.shell.cold_healing_timescale_myr
    expected = drive / (drive + healing) * (-math.expm1(-(drive + healing) * dt))
    np.testing.assert_allclose(state.damage, expected, rtol=0.0, atol=2e-16)
    np.testing.assert_array_equal(state.yield_ratio, 2.0)
    assert expected > 0.0


def test_constant_load_damage_integral_is_independent_of_material_step(monkeypatch):
    owner = _constant_strength_model()
    strength = owner.shell.tensile_strength_pa
    cycle = np.zeros((2, owner.mesh.cell_count, 3))
    cycle[0, :, :2], cycle[1, :, :2] = 2.0 * strength, -2.0 * strength
    monkeypatch.setattr(starter_module, "tidal_stress_cycle", lambda *args, **kwargs: cycle)
    single = owner.initial_state()
    single.damage[:] = 0.2
    fine = deepcopy(single)
    owner._material_sample(single, _fixed_material_sample(owner, 0.0),
                           _fixed_material_sample(owner, 0.1))
    for index in range(10):
        owner._material_sample(fine, _fixed_material_sample(owner, index * 0.01),
                               _fixed_material_sample(owner, (index + 1) * 0.01))
    np.testing.assert_allclose(fine.damage, single.damage, rtol=0.0, atol=4e-16)
    np.testing.assert_array_equal(fine.yield_ratio, single.yield_ratio)


def test_restarting_an_actual_cooling_state_preserves_the_next_physical_result(tmp_path):
    owner = model()
    state = owner.advance(owner.initial_state(), 1.0)
    checkpoint = tmp_path / "independent_starter.npz"
    owner.save_state(checkpoint, state)
    restored = model().load_state(checkpoint)
    resumed = model().advance(restored, 2.0)
    continued = owner.advance(state, 2.0)
    for name in ("damage", "cooling_stress_pa", "water_access", "yield_ratio", "strength_pa",
                 "eligible", "split_band"):
        np.testing.assert_array_equal(getattr(resumed, name), getattr(continued, name))
    np.testing.assert_array_equal(resumed.thermal_context.column_enthalpy,
                                  continued.thermal_context.column_enthalpy)
    np.testing.assert_array_equal(resumed.system.cell_plate, continued.system.cell_plate)
    assert resumed.thermal_context.thermal == continued.thermal_context.thermal
    assert resumed.thermal_context.orbit == continued.thermal_context.orbit
    assert resumed.events == continued.events
    assert resumed.thermal_samples == continued.thermal_samples


def test_stronger_mantle_control_reaches_a_real_partition_without_prescribed_plate_count():
    owner = model(traction=50000.0)
    initial = owner.initial_state()
    final = owner.advance(initial, 2.0)
    assert final.stopped_reason == "first_partition"
    assert len(final.system.plates) == 2
    assert len(final.events) == 1
    assert final.time_myr == final.events[0]["time_myr"] < 2.0
    assert np.any(final.split_band) and np.all(final.eligible[final.split_band])
    assert final.kinematic_fit_relative_residual is not None
    assert not owner.diagnose(final)["mature_handoff_ready"]
