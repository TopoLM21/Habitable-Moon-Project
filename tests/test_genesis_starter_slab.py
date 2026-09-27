"""Young pull must be earned by actual stored subduction, with no force seed."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from tectonics import dynamics
from tectonics.cpu_runtime import CpuExecution
from tectonics.genesis_starter_slab import (
    slab_development_fraction, young_slab_pull_multiplier, young_slab_pull,
)
from tectonics.kinematics import classify_boundaries
from tectonics.lithosphere import initialize_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from tectonics.subduction_memory import (
    SlabZone, SubductionMemoryParameters, advance_subduction_memory,
    initialize_subduction_memory, memory_to_json,
)


def test_unformed_slabs_have_no_pull_and_pair_direction_matters():
    parameters = SubductionMemoryParameters()
    memory = initialize_subduction_memory()
    assert young_slab_pull_multiplier(None, 0, 1, parameters) == 0.0
    assert young_slab_pull_multiplier(memory, 0, 1, parameters) == 0.0
    memory.zones[(0, 1)] = SlabZone(0, 1)
    assert young_slab_pull_multiplier(memory, 0, 1, parameters) == 0.0
    memory.zones[(0, 1)].slab_length_km = 900.0
    assert young_slab_pull_multiplier(memory, 0, 1, parameters) == 0.5
    assert young_slab_pull_multiplier(memory, 1, 0, parameters) == 0.0
    assert young_slab_pull_multiplier(memory, 0, 1, replace(parameters, enabled=False)) == 0.0
    memory.zones[(0, 1)].broken_off = True
    assert young_slab_pull_multiplier(memory, 0, 1, parameters) == 0.0


def test_development_is_continuous_bounded_and_independent_of_elapsed_age():
    lengths = [0.0, 0.001, 1.0, 90.0, 900.0, 1800.0, 2000.0]
    values = [slab_development_fraction(SlabZone(0, 1, slab_length_km=x), 1800.) for x in lengths]
    assert values == sorted(values)
    np.testing.assert_allclose(values, np.clip(np.asarray(lengths) / 1800., 0., 1.))
    ancient = SlabZone(0, 1, active_age_myr=4000., slab_length_km=0.)
    assert slab_development_fraction(ancient, 1800.) == 0.


@pytest.mark.parametrize("reference", [0.0, -1.0, np.nan, np.inf])
def test_invalid_normalization_is_rejected(reference):
    with pytest.raises(ValueError, match="reference length"):
        slab_development_fraction(None, reference)


@pytest.mark.parametrize("length", [-1.0, np.nan, np.inf])
def test_invalid_saved_length_is_rejected(length):
    with pytest.raises(ValueError, match="Stored slab length"):
        slab_development_fraction(SlabZone(0, 1, slab_length_km=length), 1800.)


def _world():
    mesh = build_icosphere(1)
    system = random_plate_system(mesh, 4, 8722, 0.2, 0.1, 0.3)
    state = initialize_lithosphere(mesh, system, continental_fraction=0., continental_nuclei=0)
    return mesh, state, system


def test_actual_convergence_grows_pull_and_stationary_memory_does_not():
    mesh, state, system = _world()
    parameters = SubductionMemoryParameters()
    boundaries = classify_boundaries(mesh, system, 5287., 4., 1.)
    memory = initialize_subduction_memory()
    advance_subduction_memory(mesh, state, boundaries, 5287., 1., memory, parameters)
    assert memory.zones
    for key, zone in memory.zones.items():
        assert zone.slab_length_km > 0.
        assert young_slab_pull_multiplier(memory, *key, parameters) > 0.
    lengths = {key: zone.slab_length_km for key, zone in memory.zones.items()}
    advance_subduction_memory(mesh, state, [], 5287., 1., memory, parameters)
    assert lengths == {key: zone.slab_length_km for key, zone in memory.zones.items()}


def test_residual_is_scaled_once_per_zone_without_mutating_memory():
    parameters = SubductionMemoryParameters()
    memory = initialize_subduction_memory()
    memory.zones[(0, 1)] = SlabZone(0, 1, active=False, slab_length_km=900.,
                                   trench_length_km=100., torque_axis=np.array([0., 0., 1.]))
    memory.zones[(0, 2)] = SlabZone(0, 2, active=False, slab_length_km=0.,
                                   trench_length_km=300., torque_axis=np.array([0., 0., 1.]))
    original = dynamics.residual_pull_by_plate
    old_vec, old_fraction = original(memory, 3, parameters, 1.25, 1.85)
    before = memory_to_json(deepcopy(memory))
    with young_slab_pull(parameters):
        new_vec, new_fraction = dynamics.residual_pull_by_plate(memory, 3, parameters, 1.25, 1.85)
    # One quarter of the trench length has a half-developed slab.
    np.testing.assert_allclose(new_vec, old_vec / 8.)
    np.testing.assert_allclose(new_fraction, old_fraction / 8.)
    assert memory_to_json(memory) == before
    assert dynamics.residual_pull_by_plate is original


def test_active_slabs_are_not_counted_again_as_residual_pull():
    parameters = SubductionMemoryParameters()
    memory = initialize_subduction_memory()
    memory.zones[(0, 1)] = SlabZone(0, 1, active=True, slab_length_km=1800., trench_length_km=100.)
    with young_slab_pull(parameters):
        vec, fraction = dynamics.residual_pull_by_plate(memory, 2, parameters, 1.25, 1.85)
    assert not np.any(vec)
    assert not np.any(fraction)


def test_existing_breakoff_factor_is_preserved_and_hooks_restore_after_failure(monkeypatch):
    parameters = SubductionMemoryParameters()
    memory = initialize_subduction_memory()
    memory.zones[(0, 1)] = SlabZone(0, 1, slab_length_km=900.)
    custom = lambda *args: 0.3
    monkeypatch.setattr(dynamics, "slab_pull_multiplier_for_pair", custom)
    old_residual = dynamics.residual_pull_by_plate
    with pytest.raises(RuntimeError, match="injected"):
        with young_slab_pull(parameters):
            assert dynamics.slab_pull_multiplier_for_pair(memory, 0, 1) == pytest.approx(.15)
            raise RuntimeError("injected dynamics failure")
    assert dynamics.slab_pull_multiplier_for_pair is custom
    assert dynamics.residual_pull_by_plate is old_residual


@pytest.mark.parametrize("development", [0., .01, .5, 1.])
def test_reference_and_optimized_dynamics_apply_identical_development(development):
    mesh, state, system = _world()
    parameters = SubductionMemoryParameters()
    memory = initialize_subduction_memory()
    boundaries = classify_boundaries(mesh, system, 5287., 4., 1.)
    advance_subduction_memory(mesh, state, boundaries, 5287., 1., memory, parameters)
    for zone in memory.zones.values():
        zone.slab_length_km = development * parameters.slab_length_cap_km
    args = (mesh, state, system, system, 5287., 1., 4., 1., dynamics.DynamicsParameters())
    kwargs = dict(subduction_memory=memory, subduction_memory_params=parameters)
    saved = memory_to_json(deepcopy(memory))
    with young_slab_pull(parameters):
        with CpuExecution(boundary_forces=False):
            reference = dynamics.update_plate_dynamics(*args, **kwargs)
        with CpuExecution(boundary_forces=True):
            optimized = dynamics.update_plate_dynamics(*args, **kwargs)
    np.testing.assert_array_equal(reference[3], optimized[3])
    np.testing.assert_array_equal(dynamics.angular_velocity_vectors(reference[0]),
                                  dynamics.angular_velocity_vectors(optimized[0]))
    assert reference[1] == optimized[1]
    assert memory_to_json(memory) == saved


def test_full_development_recovers_mature_surface_force():
    mesh, state, system = _world()
    parameters = SubductionMemoryParameters()
    memory = initialize_subduction_memory()
    boundaries = classify_boundaries(mesh, system, 5287., 4., 1.)
    advance_subduction_memory(mesh, state, boundaries, 5287., 1., memory, parameters)
    for zone in memory.zones.values():
        zone.slab_length_km = parameters.slab_length_cap_km
    args = (mesh, state, system, system, 5287., 1., 4., 1., dynamics.DynamicsParameters())
    kwargs = dict(subduction_memory=memory, subduction_memory_params=parameters)
    mature = dynamics.update_plate_dynamics(*args, **kwargs)
    with young_slab_pull(parameters):
        young = dynamics.update_plate_dynamics(*args, **kwargs)
    np.testing.assert_array_equal(mature[3], young[3])
    np.testing.assert_array_equal(dynamics.angular_velocity_vectors(mature[0]),
                                  dynamics.angular_velocity_vectors(young[0]))
