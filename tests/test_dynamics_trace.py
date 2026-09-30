"""Observational force tracing must reproduce the actual dynamics update."""
import numpy as np
import pytest

from tectonics.cpu_runtime import CpuExecution
from tectonics.dynamics import (
    DynamicsParameters, angular_velocity_vectors, center_net_rotation,
    system_from_omega, update_plate_dynamics,
)
from tectonics.genesis_starter_slab import young_slab_pull
from tectonics.kinematics import BoundaryType, classify_boundaries
from tectonics.lithosphere import initialize_lithosphere
from tectonics.mantle import initialize_mantle_flow
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from tectonics.subduction_memory import (
    SubductionMemoryParameters, advance_subduction_memory, initialize_subduction_memory,
)


def _world():
    mesh = build_icosphere(1)
    system = random_plate_system(mesh, 4, 8722, 0.2, 0.1, 0.3)
    state = initialize_lithosphere(mesh, system, continental_fraction=0., continental_nuclei=0)
    state.mantle_lithosphere_thickness_km = np.full(mesh.cell_count, 50.)
    state.mantle_lithosphere_density_anomaly_kg_m3 = np.full(mesh.cell_count, 40.)
    return mesh, state, system


@pytest.mark.parametrize("optimized", [False, True])
def test_trace_does_not_change_update_and_replays_every_vector_stage(optimized):
    mesh, state, system = _world()
    mantle = initialize_mantle_flow(mesh, system)
    parameters = DynamicsParameters()
    subparameters = SubductionMemoryParameters()
    memory = initialize_subduction_memory()
    boundaries = classify_boundaries(mesh, system, 5287., 4., 1.)
    advance_subduction_memory(mesh, state, boundaries, 5287., 2., memory, subparameters)
    rollback = np.full((len(system.plates), 3), 1.e-5)
    args = mesh, state, system, system, 5287., 2., 4., 1., parameters
    kwargs = dict(mantle_flow=mantle, subduction_memory=memory,
                  subduction_memory_params=subparameters, rollback_omega_rad_per_myr=rollback)
    trace = {}
    with young_slab_pull(subparameters), CpuExecution(boundary_forces=optimized):
        ordinary = update_plate_dynamics(*args, **kwargs)
        traced = update_plate_dynamics(*args, **kwargs, trace=trace)
    np.testing.assert_array_equal(angular_velocity_vectors(ordinary[0]), angular_velocity_vectors(traced[0]))
    np.testing.assert_array_equal(ordinary[3], traced[3])
    assert ordinary[1] == traced[1]
    np.testing.assert_allclose(trace["boundary_drive_raw"],
                               trace["ridge_drive_raw"] + trace["slab_drive_raw"], atol=1.e-11)
    expected_relative = trace["drive_scale_rad_per_myr"] * (
        trace["ridge_drive_normalized"] + trace["slab_drive_normalized"]
        + trace["residual_slab_drive"] + trace["gpe_component"]) + rollback
    np.testing.assert_allclose(trace["relative_drive"], expected_relative, atol=1.e-17)
    np.testing.assert_array_equal(trace["common_mantle"], parameters.mantle_memory_fraction * trace["mantle_omega"])
    expected_target = trace["common_mantle"] + expected_relative * trace["drag_factor"][:, None]
    np.testing.assert_allclose(trace["target_omega"], expected_target, atol=1.e-17)
    np.testing.assert_allclose(trace["resistance_omega"],
        trace["collision_resistance_omega"] + trace["transform_resistance_omega"], atol=1.e-17)
    expected_relaxed = trace["current_omega"] + trace["alpha"] * (expected_target - trace["current_omega"])
    np.testing.assert_allclose(trace["relaxed_omega"], expected_relaxed, atol=1.e-17)
    np.testing.assert_array_equal(trace["post_gauge_omega"], trace["relaxed_omega"] - trace["mean_rotation"])
    np.testing.assert_allclose(trace["final_omega"], angular_velocity_vectors(traced[0]), atol=1.e-18)
    assert np.linalg.norm(trace["ridge_drive_raw"]) > 0.
    assert np.linalg.norm(trace["slab_drive_raw"]) > 0.


def test_net_rotation_removal_preserves_boundary_relative_velocities():
    mesh, state, system = _world()
    omega = angular_velocity_vectors(system)
    system = system_from_omega(state.cell_plate, system, omega + np.array([.03, -.02, .015]))
    centered = center_net_rotation(mesh, state, system, 5287.)
    before = classify_boundaries(mesh, system, 5287., 4., 1.)
    after = classify_boundaries(mesh, centered, 5287., 4., 1.)
    assert len(before) == len(after)
    for a, b in zip(before, after):
        np.testing.assert_allclose(
            [a.normal_rate_km_per_myr, a.tangential_rate_km_per_myr, a.relative_speed_km_per_myr],
            [b.normal_rate_km_per_myr, b.tangential_rate_km_per_myr, b.relative_speed_km_per_myr],
            atol=1.e-13)
        assert a.boundary_type == b.boundary_type


def test_low_speed_classification_really_gates_force_and_slab_birth():
    """Document the current zero-force zone without changing its thresholds."""
    mesh, state, system = _world()
    system = system_from_omega(state.cell_plate, system, angular_velocity_vectors(system) * .001)
    mantle = initialize_mantle_flow(mesh, system)
    subparameters = SubductionMemoryParameters()
    memory = initialize_subduction_memory()
    boundaries = classify_boundaries(mesh, system, 5287., 4., 1.)
    assert all(b.boundary_type == BoundaryType.INACTIVE for b in boundaries)
    assert any(b.normal_rate_km_per_myr < 0. for b in boundaries)
    assert any(b.normal_rate_km_per_myr > 0. for b in boundaries)
    advance_subduction_memory(mesh, state, boundaries, 5287., 1., memory, subparameters)
    assert not memory.zones
    trace = {}
    with young_slab_pull(subparameters):
        update_plate_dynamics(mesh, state, system, system, 5287., 1., 4., 1.,
            DynamicsParameters(ridge_gpe_min_factor=0.), mantle_flow=mantle,
            subduction_memory=memory, subduction_memory_params=subparameters, trace=trace)
    assert not np.any(trace["ridge_drive_raw"])
    assert not np.any(trace["slab_drive_raw"])


def test_no_young_ridge_or_slab_force_without_thermal_geometry_or_stored_slab():
    mesh, state, system = _world()
    state.mantle_lithosphere_thickness_km[:] = 0.
    memory = initialize_subduction_memory()
    subparameters = SubductionMemoryParameters()
    trace = {}
    with young_slab_pull(subparameters):
        update_plate_dynamics(mesh, state, system, system, 5287., 1., 4., 1.,
            DynamicsParameters(ridge_gpe_min_factor=0.),
            mantle_flow=initialize_mantle_flow(mesh, system), subduction_memory=memory,
            subduction_memory_params=subparameters, trace=trace)
    assert not np.any(trace["ridge_drive_raw"])
    assert not np.any(trace["slab_drive_raw"])
    assert trace["slab_edges"]  # The absence is physical development, not absent contact.
    assert all(edge["applied_multiplier"] == 0. for edge in trace["slab_edges"])


def test_slab_length_equals_integrated_actual_classified_convergence():
    mesh, state, system = _world()
    parameters = SubductionMemoryParameters()
    memory = initialize_subduction_memory()
    boundaries = classify_boundaries(mesh, system, 5287., 4., 1.)
    assert any(b.boundary_type == BoundaryType.CONVERGENT for b in boundaries)
    dt = .125
    advance_subduction_memory(mesh, state, boundaries, 5287., dt, memory, parameters)
    assert memory.zones
    for zone in memory.zones.values():
        expected = parameters.slab_length_growth_efficiency * zone.convergence_rate_km_per_myr * dt
        assert zone.slab_length_km == pytest.approx(expected)
        assert zone.slab_depth_km == pytest.approx(expected * np.sin(np.deg2rad(zone.dip_deg)))
    lengths = {key: zone.slab_length_km for key, zone in memory.zones.items()}
    advance_subduction_memory(mesh, state, [], 5287., dt, memory, parameters)
    assert lengths == {key: zone.slab_length_km for key, zone in memory.zones.items()}
