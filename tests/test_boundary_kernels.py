"""Production boundary-force opt-in, parity, bounded-cache and cleanup checks."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from tectonics import boundary_kernels, dynamics
from tectonics.cpu_runtime import CpuExecution, current_execution
from tectonics.kinematics import classify_boundaries
from tectonics.lithosphere import CrustType, initialize_lithosphere
from tectonics.mantle import initialize_mantle_flow
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from test_gpu_stage_profile import assert_exact


def world():
    mesh = build_icosphere(1)
    system = random_plate_system(mesh, 4, 8722, 0.2, 0.1, 0.3)
    state = initialize_lithosphere(mesh, system, 0.5, 4, 7.0, 35.0, 500.0,
                                  radius_km=5287.0)
    continental = state.crust_type == int(CrustType.CONTINENTAL)
    state.crust_thickness_km[continental] = np.linspace(
        40.0, 70.0, np.count_nonzero(continental))
    return mesh, state, system


@pytest.mark.parametrize("mode", ["modern", "mantle", "legacy"])
def test_production_preserves_whole_dynamics_and_inputs(mode):
    mesh, state, system = world()
    kwargs = {"thermal_lithosphere_thickness_km": 133.75}
    if mode == "mantle":
        kwargs["mantle_flow"] = initialize_mantle_flow(mesh, system)
        kwargs["rollback_omega_rad_per_myr"] = np.linspace(-1e-6, 1e-6, 12).reshape(4, 3)
    elif mode == "legacy":
        state.mantle_lithosphere_thickness_km = None
        state.mantle_lithosphere_density_anomaly_kg_m3 = None
        state.continental_fraction = None
        state.craton_strength = None
    params = dynamics.DynamicsParameters(
        slab_buoyancy_exponent=0.73, slab_thermal_exponent=0.27)
    args = (mesh, state, system, system, 5287.0, 4.0, 4.0, 1.0, params)
    saved = deepcopy((state, system, kwargs))
    original = dynamics.update_plate_dynamics
    with CpuExecution(boundary_forces=False):
        expected = original(*args, **kwargs)
    with CpuExecution(boundary_forces=True) as execution:
        for _ in range(2):
            assert_exact(expected, original(*args, **kwargs))
        report = execution.numerical_report()["boundary_forces"]
        assert report["enabled"] is True
        assert report["backend"] == "prepared_cpu_boundary_forces"
        assert report["calls"] == 2
        assert report["inclusive_seconds"] > 0.0
        assert report["cached_edges"] > 0
        assert report["cached_geometry_numeric_bytes"] > 0
    assert execution._boundary_geometry is None
    assert execution.numerical_report()["boundary_forces"] == report
    assert dynamics.update_plate_dynamics is original
    assert_exact(saved, (state, system, kwargs))


def test_opt_out_and_no_context_keep_reference_without_preparation(monkeypatch):
    mesh, state, system = world()
    args = (mesh, state, system, system, 5287.0, 4.0, 4.0, 1.0,
            dynamics.DynamicsParameters())
    calls = []
    reference = dynamics._boundary_force_terms_reference

    def counted_reference(*args, **kwargs):
        calls.append(1)
        return reference(*args, **kwargs)

    def unexpected(*args, **kwargs):
        pytest.fail("Disabled production boundary kernel must not execute")

    monkeypatch.setattr(dynamics, "_boundary_force_terms_reference", counted_reference)
    monkeypatch.setattr(boundary_kernels, "prepared_cpu", unexpected)
    expected = dynamics.update_plate_dynamics(*args)
    with CpuExecution() as execution:
        assert execution.boundary_forces_enabled is False
        assert_exact(expected, dynamics.update_plate_dynamics(*args))
        assert execution._boundary_geometry is None
        assert execution.numerical_report()["boundary_forces"]["calls"] == 0
    with CpuExecution(boundary_forces=False):
        assert_exact(expected, dynamics.update_plate_dynamics(*args))
    assert len(calls) == 3


def test_enabled_dispatch_does_not_call_scalar_reference(monkeypatch):
    mesh, state, system = world()

    def unexpected(*args, **kwargs):
        pytest.fail("Enabled production boundary calculation must use prepared kernel")

    monkeypatch.setattr(dynamics, "_boundary_force_terms_reference", unexpected)
    with CpuExecution(boundary_forces=True) as execution:
        dynamics.update_plate_dynamics(mesh, state, system, system, 5287.0,
                                       4.0, 4.0, 1.0, dynamics.DynamicsParameters())
        assert execution.boundary_calls == 1


def test_production_replaces_cache_for_new_mesh_and_cleans_on_exit():
    mesh, state, system = world()
    moved = deepcopy(mesh)
    moved.centroids = np.roll(moved.centroids, 1, axis=1)
    args = (state, system, system, 5287.0, 4.0, 4.0, 1.0,
            dynamics.DynamicsParameters())
    with CpuExecution(boundary_forces=False):
        expected = dynamics.update_plate_dynamics(mesh, *args)
        expected_moved = dynamics.update_plate_dynamics(moved, *args)
    with CpuExecution(boundary_forces=True) as execution:
        assert_exact(expected, dynamics.update_plate_dynamics(mesh, *args))
        first = execution._boundary_geometry
        assert_exact(expected_moved, dynamics.update_plate_dynamics(moved, *args))
        assert execution._boundary_geometry is not first
        assert execution._boundary_geometry.mesh is moved
        assert_exact(expected, dynamics.update_plate_dynamics(mesh, *args))
        assert execution._boundary_geometry is not first
        assert execution._boundary_geometry.mesh is mesh
        assert execution.boundary_calls == 3
    assert execution._boundary_geometry is None
    assert current_execution() is None


def test_production_cleans_cache_on_kernel_error_without_mutating_state(monkeypatch):
    mesh, state, system = world()
    saved = deepcopy((state, system))

    def fail(*args, **kwargs):
        raise ValueError("injected production boundary error")

    monkeypatch.setattr(boundary_kernels, "prepared_cpu", fail)
    with pytest.raises(ValueError, match="injected production boundary error"):
        with CpuExecution(boundary_forces=True) as execution:
            dynamics.update_plate_dynamics(mesh, state, system, system, 5287.0,
                                           4.0, 4.0, 1.0, dynamics.DynamicsParameters())
    assert execution._boundary_geometry is None
    assert execution.boundary_calls == 0
    assert current_execution() is None
    assert_exact(saved, (state, system))


@pytest.mark.parametrize("kind", ["float32", "nonfinite", "shape"])
def test_production_rejects_unsupported_dynamic_array_before_caching(kind):
    mesh, state, system = world()
    boundaries = classify_boundaries(mesh, system, 5287.0, 4.0, 1.0)
    if kind == "float32":
        state.tidal_damage = state.tidal_damage.astype(np.float32)
    elif kind == "nonfinite":
        state.tidal_damage[0] = np.nan
    else:
        state.tidal_damage = state.tidal_damage[:-1]
    with CpuExecution(boundary_forces=True) as execution:
        with pytest.raises(ValueError, match="tidal_damage.*FP64"):
            execution.calculate_boundary_forces(
                mesh, state, boundaries, 5287.0, len(system.plates),
                dynamics.DynamicsParameters())
        assert not execution._boundary_geometry.edges
        assert execution.boundary_calls == 0
    assert execution._boundary_geometry is None


def test_bounded_custom_midpoint_cache_eviction_preserves_exact_output():
    mesh, state, system = world()
    boundaries = classify_boundaries(mesh, system, 5287.0, 4.0, 1.0)
    geometry = boundary_kernels.BoundaryGeometry(mesh)
    assert geometry.max_cached_edges == 2 * len(mesh.shared_edges)
    # Exercise the actual production limit using distinct custom midpoint keys.
    original = boundaries[0]
    custom = []
    for i in range(geometry.max_cached_edges + 3):
        midpoint = original.midpoint + np.array([1e-7 * i, 0.0, 0.0])
        midpoint /= np.linalg.norm(midpoint)
        custom.append(replace(original, midpoint=midpoint))
    args = (state, custom, 5287.0, len(system.plates),
            dynamics.DynamicsParameters())
    saved = deepcopy((state, custom))
    with CpuExecution():
        expected = dynamics._boundary_force_terms_reference(mesh, *args)
        for _ in range(2):
            assert_exact(expected, boundary_kernels.prepared_cpu(geometry, *args))
            assert len(geometry.edges) == geometry.max_cached_edges
    assert_exact(saved, (state, custom))
