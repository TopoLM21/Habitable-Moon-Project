"""CPU-only full dynamics checks for the isolated process-level experiment."""
from copy import deepcopy

import numpy as np
import pytest

from analysis.run_boundary_candidate import BoundaryCandidateContext
import tectonics.dynamics as dynamics
from tectonics.cpu_runtime import CpuExecution
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
    state.crust_thickness_km[continental] = np.linspace(40.0, 70.0, np.count_nonzero(continental))
    return mesh, state, system


@pytest.mark.parametrize("mode", ["modern", "mantle", "legacy"])
def test_candidate_preserves_complete_dynamics_and_inputs(mode):
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
    params = dynamics.DynamicsParameters(slab_buoyancy_exponent=0.73, slab_thermal_exponent=0.27)
    args = (mesh, state, system, system, 5287.0, 4.0, 4.0, 1.0, params)
    saved = deepcopy((state, system, kwargs))
    original = dynamics.update_plate_dynamics
    with CpuExecution(numeric_kernels=True):
        expected = original(*args, **kwargs)
        with BoundaryCandidateContext() as context:
            for _ in range(2):
                actual = dynamics.update_plate_dynamics(*args, **kwargs)
                assert_exact(expected, actual)
            report = context.report()
            assert report["calls"] == 2
            assert report["cached_edges"] > 0
            assert report["inclusive_seconds"] > 0.0
    assert dynamics.update_plate_dynamics is original
    assert context.geometry is None
    assert_exact(saved, (state, system, kwargs))


def test_candidate_context_restores_runner_alias_after_error(monkeypatch):
    import run_long_evolution_v123 as base
    import analysis.run_boundary_candidate as candidate_module

    original = dynamics.update_plate_dynamics
    assert base.update_plate_dynamics is original
    mesh, state, system = world()
    saved = deepcopy(state)

    def fail(*args, **kwargs):
        raise ValueError("injected candidate failure")

    monkeypatch.setattr(candidate_module, "prepared_cpu", fail)
    context = BoundaryCandidateContext()
    with pytest.raises(ValueError, match="injected candidate failure"):
        with context:
            assert base.update_plate_dynamics is dynamics.update_plate_dynamics
            assert dynamics.update_plate_dynamics is not original
            base.update_plate_dynamics(mesh, state, system, system, 5287.0, 4.0, 4.0, 1.0,
                                       dynamics.DynamicsParameters())
    assert dynamics.update_plate_dynamics is original
    assert base.update_plate_dynamics is original
    assert context.geometry is None and not context._installed
    assert context.replacements == []
    assert context.calls == 0
    assert_exact(saved, state)


def test_candidate_rejects_nested_contexts_and_inactive_calls():
    original = dynamics.update_plate_dynamics
    context = BoundaryCandidateContext()
    with pytest.raises(RuntimeError, match="not active"):
        context.calculate(*([None] * 9))
    with context:
        active = dynamics.update_plate_dynamics
        with pytest.raises(RuntimeError, match="already active"):
            context.__enter__()
        with pytest.raises(RuntimeError, match="Another boundary candidate"):
            with BoundaryCandidateContext():
                pytest.fail("Nested context must not enter")
        assert dynamics.update_plate_dynamics is active
    assert dynamics.update_plate_dynamics is original


def test_new_mesh_object_replaces_geometry_cache():
    mesh, state, system = world()
    moved_mesh = deepcopy(mesh)
    # Keep edge vertices/midpoints unchanged while changing centroid geometry;
    # a cache keyed by boundary records alone would incorrectly reuse normals.
    moved_mesh.centroids = np.roll(moved_mesh.centroids, 1, axis=1)
    params = dynamics.DynamicsParameters()
    args = (state, system, system, 5287.0, 4.0, 4.0, 1.0, params)
    original = dynamics.update_plate_dynamics
    with CpuExecution(numeric_kernels=True):
        expected_initial = original(mesh, *args)
        expected_moved = original(moved_mesh, *args)
        with BoundaryCandidateContext() as context:
            assert_exact(expected_initial, dynamics.update_plate_dynamics(mesh, *args))
            first_geometry = context.geometry
            assert_exact(expected_moved, dynamics.update_plate_dynamics(moved_mesh, *args))
            assert context.geometry is not first_geometry
            assert context.geometry.mesh is moved_mesh
            assert_exact(expected_initial, dynamics.update_plate_dynamics(mesh, *args))
            assert context.geometry is not first_geometry
            assert context.geometry.mesh is mesh
            assert context.calls == 3
    assert context.geometry is None
