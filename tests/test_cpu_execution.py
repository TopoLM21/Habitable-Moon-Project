from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from tectonics.cpu_runtime import CpuExecution, current_execution, query_workers
from tectonics.lithosphere import initialize_lithosphere
from tectonics.mantle import advance_mantle_flow, initialize_mantle_flow
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from tectonics.transport import (
    SubgridTransportParameters,
    _median_cell_spacing_rad,
    _median_cell_spacing_rad_batched,
    build_transport_map,
    initialize_transport_state,
)
from visualization.raster import rasterize_cells


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_parallel_transport_is_exact_including_ordered_diagnostics(workers):
    mesh = build_icosphere(2)
    system = random_plate_system(mesh, 6, 20260819, 0.2, 0.15, 0.6)
    state = initialize_lithosphere(mesh, system, continental_fraction=0.28, continental_nuclei=4)
    reference_state = initialize_transport_state(6)
    parallel_state = deepcopy(reference_state)
    reference = []
    params = SubgridTransportParameters()
    for _ in range(8):
        reference.append(deepcopy(build_transport_map(mesh, system, state, 4.0, reference_state, params)))
    with CpuExecution(workers):
        for expected in reference:
            actual = build_transport_map(mesh, system, state, 4.0, parallel_state, params)
            for name in ("covered", "source"):
                np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
            for actual_map, expected_map in zip(actual.source_to_target, expected.source_to_target, strict=True):
                np.testing.assert_array_equal(actual_map, expected_map)
            np.testing.assert_array_equal(actual.state.residual_quaternions, expected.state.residual_quaternions)
            np.testing.assert_array_equal(actual.state.hold_age_myr, expected.state.hold_age_myr)
            assert asdict(actual.diagnostics) == asdict(expected.diagnostics)


def test_batched_regular_mesh_spacing_matches_scalar_exactly():
    for subdivisions in (1, 3, 5):
        mesh = build_icosphere(subdivisions)
        neighbors = np.asarray(mesh.neighbors, dtype=np.int32)
        assert _median_cell_spacing_rad_batched(mesh, neighbors) == _median_cell_spacing_rad(mesh)


def test_failed_parallel_preparation_does_not_partly_commit(monkeypatch):
    import tectonics.transport as transport
    mesh = build_icosphere(1)
    system = random_plate_system(mesh, 4, 3, 0.2, 0.15, 0.6)
    state = initialize_lithosphere(mesh, system)
    memory = initialize_transport_state(4)
    before = deepcopy(memory)
    def fail(*args, **kwargs):
        raise RuntimeError("test failure")
    monkeypatch.setattr(transport, "_optimal_assignment", fail)
    params = SubgridTransportParameters(min_changed_fraction=0.0, min_p75_cell_spacing_fraction=0.0)
    with CpuExecution(2), pytest.raises(RuntimeError, match="test failure"):
        build_transport_map(mesh, system, state, 4.0, memory, params)
    np.testing.assert_array_equal(memory.residual_quaternions, before.residual_quaternions)
    np.testing.assert_array_equal(memory.hold_age_myr, before.hold_age_myr)
    assert memory.cumulative_commit_count == 0
    assert current_execution() is None


def test_mantle_batching_preserves_exact_fixed_grid_field():
    mesh = build_icosphere(3)
    system = random_plate_system(mesh, 6, 9, 0.2, 0.15, 0.6)
    state = initialize_mantle_flow(mesh, system)
    expected = advance_mantle_flow(mesh, state, 4.0, 0.82)
    with CpuExecution(2):
        actual = advance_mantle_flow(mesh, state, 4.0, 0.82)
    np.testing.assert_array_equal(actual[0].cell_omega_rad_per_myr, expected[0].cell_omega_rad_per_myr)
    assert asdict(actual[1]) == asdict(expected[1])


def test_raster_cache_uses_mesh_identity_resolution_and_current_values():
    meshes = [build_icosphere(2), build_icosphere(2)]
    meshes[1].centroids = meshes[1].centroids[:, [1, 2, 0]].copy()
    values = np.arange(meshes[0].cell_count, dtype=float)
    cases = [(meshes[0], values, 40, 20), (meshes[0], values + 7, 40, 20),
             (meshes[0], values, 60, 30), (meshes[1], values, 40, 20)]
    expected = [rasterize_cells(*case) for case in cases]
    with CpuExecution(2) as execution:
        for case, want in zip(cases, expected, strict=True):
            actual = rasterize_cells(*case)
            for a, b in zip(actual, want, strict=True):
                np.testing.assert_array_equal(a, b)
        assert len(execution.geometry(meshes[0]).rasters) == 2
        for width in range(41, 49):
            rasterize_cells(meshes[0], values, width, 20)
        assert len(execution.geometry(meshes[0]).rasters) == 4


def test_execution_is_opt_in_bounded_and_cleaned_up():
    assert query_workers() == -1
    with CpuExecution(2) as execution:
        assert query_workers() == 1
        with pytest.raises(RuntimeError):
            with CpuExecution(1):
                pass
        for _ in range(4):
            execution.geometry(build_icosphere(0))
        assert len(execution._meshes) == 2
    assert query_workers() == -1
    assert not execution._meshes
    for workers in (0, -1, 33, True, 1.5):
        with pytest.raises(ValueError):
            CpuExecution(workers)


def test_pool_uses_separate_workers_but_returns_input_order():
    from threading import Barrier, get_ident
    barrier = Barrier(4)
    def work(index):
        barrier.wait(timeout=5)
        return index, get_ident(), query_workers()
    with CpuExecution(4) as execution:
        results = execution.ordered_map(work, range(4))
    assert [row[0] for row in results] == list(range(4))
    assert len({row[1] for row in results}) == 4
    assert all(row[2] == 1 for row in results)
