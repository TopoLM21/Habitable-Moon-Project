from copy import deepcopy
from dataclasses import fields, is_dataclass
from threading import Barrier, get_ident
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.cpu_runtime import CpuExecution, current_execution
from tectonics.lithosphere import (advance_lithosphere, initialize_lithosphere,
                                  refresh_mechanical_lithosphere)
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from tectonics.tides import constant_eccentricity
from tectonics.transport import SubgridTransportParameters, initialize_transport_state


def exact(a, b):
    if isinstance(a, np.ndarray):
        assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
    elif is_dataclass(a):
        for field in fields(a):
            exact(getattr(a, field.name), getattr(b, field.name))
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            exact(x, y)
    else:
        assert a == b


def test_wired_root_contrast_reuses_indices_but_not_evolving_values():
    from tectonics.plume_rifting import _neighbor_root_contrast
    mesh = build_icosphere(2)
    rng = np.random.default_rng(871)
    roots = [rng.uniform(-5, 300, mesh.cell_count) for _ in range(2)]
    references = [_neighbor_root_contrast(mesh, SimpleNamespace(mantle_lithosphere_thickness_km=root))
                  for root in roots]
    with CpuExecution() as execution:
        for root, want in zip(roots, references, strict=True):
            actual = _neighbor_root_contrast(mesh, SimpleNamespace(mantle_lithosphere_thickness_km=root))
            exact(want, actual)
        assert execution.geometry(mesh).neighbors.shape == (mesh.cell_count, 3)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_wired_proxy_and_polarity_use_local_values_without_changing_ties(dtype, monkeypatch):
    import tectonics.dynamics as dynamics
    import tectonics.subduction_memory as memory
    state = SimpleNamespace(
        mantle_lithosphere_thickness_km=np.array([-0., 50., 50., 100., 150., -1.], dtype=dtype),
        mantle_lithosphere_density_anomaly_kg_m3=np.array([50., 60., 60., -3., 90., 4.], dtype=dtype),
        crust_age_myr=np.arange(6, dtype=dtype), crust_type=np.zeros(6, dtype=np.int8))
    boundaries = [SimpleNamespace(face_a=i, face_b=i + 1, plate_a=2, plate_b=1) for i in range(5)]
    proxies = np.asarray([memory._local_proxy(state, i) for i in range(6)])
    expected = [(dynamics._choose_subducting_side(state, b), memory.choose_subducting_side(state, b))
                for b in boundaries]
    def no_full_array(*args):
        raise AssertionError("Optimized per-face lookup built a full field")
    with CpuExecution():
        monkeypatch.setattr(dynamics, "mantle_lithosphere_negative_buoyancy_proxy", no_full_array)
        monkeypatch.setattr(memory, "mantle_lithosphere_negative_buoyancy_proxy", no_full_array)
        exact(proxies, np.asarray([memory._local_proxy(state, i) for i in range(6)]))
        assert expected == [(dynamics._choose_subducting_side(state, b), memory.choose_subducting_side(state, b))
                            for b in boundaries]


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
@pytest.mark.parametrize("optional_fields", [False, True])
def test_single_source_path_matches_complete_conservative_step(workers, optional_fields):
    mesh = build_icosphere(2)
    system = random_plate_system(mesh, 6, 72814, 0.2, 3.0, 5.0)
    state = initialize_lithosphere(mesh, system, continental_fraction=0.28, continental_nuclei=4)
    if optional_fields:
        rng = np.random.default_rng(371)
        state.continental_lithosphere_age_myr = rng.uniform(0, 900, mesh.cell_count)
        state.mantle_depletion_fraction = rng.uniform(0, 1, mesh.cell_count)
        state.craton_strength = rng.uniform(0, 1, mesh.cell_count)
        refresh_mechanical_lithosphere(state, 0.0)
    else:
        state.rift_extension = state.extension_age_myr = None
        state.mantle_lithosphere_thickness_km = state.mantle_lithosphere_density_anomaly_kg_m3 = None
        state.continental_fraction = state.continental_volume_km3 = None
    saved_input = deepcopy(state)
    params = SubgridTransportParameters(min_changed_fraction=0.0, min_p75_cell_spacing_fraction=0.0)
    transport = initialize_transport_state(len(system.plates))
    reference_transport = deepcopy(transport)
    arguments = (mesh, system, state, 4., 5287., 7.12, 47., constant_eccentricity(.00047))
    with CpuExecution(numeric_kernels=False, single_source_cells=False):
        reference = advance_lithosphere(*arguments, transport_state=reference_transport, transport_parameters=params)
    with CpuExecution(single_source_cells=True, cell_workers=workers) as execution:
        actual = advance_lithosphere(*arguments, transport_state=transport, transport_parameters=params)
        assert execution.cell_calls == 1 and execution.cells_prepared > 0
        assert execution.cell_tasks == workers
    exact(reference, actual)
    exact(reference_transport, transport)
    exact(saved_input, state)
    assert current_execution() is None and execution.cell_pool is None


def test_cell_pool_is_separate_ordered_and_closes_on_error():
    barrier = Barrier(4)
    def work(index):
        barrier.wait(timeout=5)
        return index, get_ident()
    with CpuExecution(workers=2, single_source_cells=True, cell_workers=4) as execution:
        results = execution.ordered_cell_map(work, range(4))
        assert [row[0] for row in results] == list(range(4))
        assert len({row[1] for row in results}) == 4
        assert execution.pool is not execution.cell_pool
    assert execution.pool is execution.cell_pool is None
    for invalid in (0, -1, 3, 16, True, 2.0):
        with pytest.raises(ValueError):
            CpuExecution(cell_workers=invalid)


def test_cell_preparation_failure_does_not_commit_any_outputs(monkeypatch):
    import tectonics.lithosphere_kernels as kernels
    n = 8
    outputs = {"plate": np.full(n, -7)}
    state = SimpleNamespace(crust_type=np.zeros(n), crust_age_myr=np.zeros(n), crust_thickness_km=np.ones(n))
    def fail(targets, **kwargs):
        if targets[0] == 0:
            raise RuntimeError("test worker failure")
        return targets, {"plate": np.zeros(len(targets))}, get_ident()
    monkeypatch.setattr(kernels, "_prepare_single_source", fail)
    with CpuExecution(single_source_cells=True, cell_workers=4) as execution:
        with pytest.raises(RuntimeError, match="test worker failure"):
            kernels.fill_single_source_cells(execution, np.arange(n), covered=np.ones((1, n), bool),
                source=np.arange(n)[None, :], areas=np.ones(n), state=state,
                fraction=np.zeros(n), volume=np.zeros(n), dt_myr=4., copied_fields={}, outputs=outputs)
        np.testing.assert_array_equal(outputs["plate"], -7)
        assert execution.cell_calls == 0
    assert execution.cell_pool is None


def test_single_source_empty_and_signed_zero_fields():
    from tectonics.lithosphere import CrustType
    from tectonics.lithosphere_kernels import _prepare_single_source, fill_single_source_cells
    areas = np.array([2., 3., 7., 11.])
    source = np.array([[3, 2, 1, 0]], dtype=np.int32)
    state = SimpleNamespace(crust_type=np.array([CrustType.OCEANIC, CrustType.CONTINENTAL] * 2),
                            crust_age_myr=np.array([-0., 10., 20., 30.]),
                            crust_thickness_km=np.array([-0., 30., 7., 40.]))
    fraction = np.array([-0., .5, 0., 1.])
    volume = np.array([-0., 20., 0., 80.])
    copied = {"damage": np.array([-0., 0., .4, 1.])}
    targets = np.arange(4)
    _, actual, _ = _prepare_single_source(targets, covered=np.ones((1, 4), bool),
        source=source, areas=areas, crust_type=state.crust_type, age=state.crust_age_myr,
        thickness=state.crust_thickness_km, fraction=fraction, volume=volume,
        dt_myr=4., copied_fields=copied)
    expected_fraction, expected_volume, expected_h = [], [], []
    for target in targets:
        src = source[:, target]
        winner = int(src[0])
        expected_fraction.append(float(np.sum(fraction[src] * areas[src]) / areas[target]))
        expected_volume.append(float(np.sum(volume[src])))
        if state.crust_type[winner] == CrustType.CONTINENTAL:
            wf = max(float(fraction[winner]), 1e-12)
            expected_h.append(float(volume[winner] / max(areas[winner] * wf, 1e-30)))
        else:
            expected_h.append(float(state.crust_thickness_km[winner] * areas[winner] / areas[target]))
    exact(np.asarray(expected_fraction), actual["fraction"])
    exact(np.asarray(expected_volume), actual["volume"])
    exact(np.asarray(expected_h), actual["thickness"])
    exact(copied["damage"][source[0]], actual["damage"])
    with CpuExecution(single_source_cells=True, cell_workers=8) as execution:
        fill_single_source_cells(execution, np.array([], dtype=int), covered=None,
            source=None, areas=None, state=None, fraction=None, volume=None,
            dt_myr=4., copied_fields={}, outputs={})
        assert execution.cell_calls == execution.cell_tasks == execution.cells_prepared == 0


def test_single_source_opt_in_does_not_change_legacy_transport():
    mesh = build_icosphere(1)
    system = random_plate_system(mesh, 4, 111, .2, .15, .6)
    state = initialize_lithosphere(mesh, system)
    args = (mesh, system, state, 4., 5287., 7.12, 47., constant_eccentricity(.00047))
    with CpuExecution(numeric_kernels=False, single_source_cells=False):
        expected = advance_lithosphere(*args)
    with CpuExecution(single_source_cells=True, cell_workers=4) as execution:
        actual = advance_lithosphere(*args)
        assert execution.cell_calls == 0
    exact(expected, actual)


def test_failed_cell_pool_creation_cleans_up_transport_pool(monkeypatch):
    import tectonics.cpu_runtime as runtime
    made = []
    real_pool = runtime.ThreadPoolExecutor
    def make_pool(**kwargs):
        if made:
            raise RuntimeError("test pool creation failure")
        made.append(real_pool(**kwargs))
        return made[0]
    monkeypatch.setattr(runtime, "ThreadPoolExecutor", make_pool)
    execution = CpuExecution(workers=2, cell_workers=4)
    with pytest.raises(RuntimeError, match="test pool creation failure"):
        with execution:
            pass
    assert current_execution() is None
    assert execution.pool is execution.cell_pool is None
    with pytest.raises(RuntimeError, match="shutdown"):
        made[0].submit(lambda: None)
