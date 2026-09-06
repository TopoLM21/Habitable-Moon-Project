"""Independent exact tests for the isolated halo cut; no production hook."""
from dataclasses import replace

import numpy as np
import pytest

from analysis.probe_gpu_topography import (
    CpuHalo, GpuHalo, as_result, compare_results, prepare_seeds,
)
from tectonics.kinematics import BoundaryRecord, BoundaryType
from tectonics.lithosphere import CrustType, initialize_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from tectonics.topography import TopographyParameters, tectonic_forcing


def fixture_world():
    mesh = build_icosphere(1)
    plates = random_plate_system(mesh, 4, 8192, 0.2, 0.1, 0.3)
    state = initialize_lithosphere(mesh, plates, 0.28, 2, 7.0, 35.0, 500.0, radius_km=5287.0)
    rng = np.random.default_rng(89032)
    state.continental_fraction[:] = rng.choice([0.0, 0.1, 0.5, 1.0], size=mesh.cell_count)
    state.crust_age_myr[:] = rng.choice([0.0, 10.0, 100.0, 500.0], size=mesh.cell_count)
    state.crust_type[:] = rng.choice([int(CrustType.OCEANIC), int(CrustType.CONTINENTAL)],
                                    size=mesh.cell_count)
    boundaries = []
    for i, (a, b, u, v) in enumerate(mesh.shared_edges):
        boundaries.append(BoundaryRecord(
            int(a), int(b), int(u), int(v), int(state.cell_plate[a]), int(state.cell_plate[b]),
            np.array([1.0, 0.0, 0.0]), float(rng.uniform(-100, 100)), 0.0, 100.0,
            BoundaryType(i % 4),
        ))
    return mesh, state, boundaries + boundaries[::7]


@pytest.mark.parametrize("material", [False, True])
@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("halo", [-0.1, 0.0, 0.4, 1.5])
def test_cpu_halo_cut_matches_original_bytes(material, external, halo):
    mesh, state, boundaries = fixture_world()
    params = TopographyParameters(material_aware_boundary_forcing=material,
                                  boundary_one_ring_fraction=halo)
    arcs = np.linspace(-0.3, 2.3, mesh.cell_count) if external else None
    args = mesh, state, boundaries, params, 5287.0, arcs
    seeds, tags = prepare_seeds(*args)
    actual = as_result(CpuHalo(mesh).calculate(seeds, halo, external), tags)
    assert compare_results(tectonic_forcing(*args), actual)["byte_exact"]


def test_cpu_halo_cut_empty_boundaries_and_nondefault_amounts():
    mesh, state, boundaries = fixture_world()
    params = TopographyParameters(ridge_uplift_m=-1.0, trench_min_extra_depth_m=-4.0,
                                  trench_max_extra_depth_m=-2.0, arc_uplift_m=-12.0,
                                  continental_collision_uplift_m=-5.0)
    for selected in ([], boundaries):
        for arcs in (None, np.linspace(-1.0, 3.0, mesh.cell_count)):
            args = mesh, state, selected, params, 5287.0, arcs
            seeds, tags = prepare_seeds(*args)
            actual = as_result(CpuHalo(mesh).calculate(seeds, 0.4, arcs is not None), tags)
            assert compare_results(tectonic_forcing(*args), actual)["byte_exact"]


@pytest.mark.parametrize("external", [False, True])
def test_cuda_halo_cut_matches_original_bytes(external):
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA is unavailable")
    if count < 1:
        pytest.skip("CUDA device is unavailable")
    mesh, state, boundaries = fixture_world()
    backend = GpuHalo(mesh, 0)
    for material in (False, True):
        for halo in (0.0, 0.4, 1.5):
            params = TopographyParameters(material_aware_boundary_forcing=material,
                                          boundary_one_ring_fraction=halo)
            arcs = np.linspace(-0.3, 2.3, mesh.cell_count) if external else None
            args = mesh, state, boundaries, params, 5287.0, arcs
            seeds, tags = prepare_seeds(*args)
            actual = as_result(backend.calculate(seeds, halo, external), tags)
            assert compare_results(tectonic_forcing(*args), actual)["byte_exact"]
