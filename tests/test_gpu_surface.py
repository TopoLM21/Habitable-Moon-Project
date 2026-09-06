"""Public-API parity and ownership checks for the resident sediment pipeline."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace

import numpy as np
import pytest

from tectonics.cpu_runtime import CpuExecution
from tectonics.lithosphere import initialize_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from tectonics.sediment import SedimentBudgetState, SedimentParameters, advance_sediments
from tectonics.topography import TopographyState


RADIUS_KM = 5287.0


def _cuda_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    return cp


def _case(mesh=None, *, seed=20260905, radius_km=RADIUS_KM):
    if mesh is None:
        mesh = build_icosphere(2)
    rng = np.random.default_rng(seed)
    n = mesh.cell_count
    areas = mesh.physical_cell_areas_km2(radius_km)
    plates = random_plate_system(mesh, 4, 20260821, 0.2, 0.1, 0.4)
    previous = initialize_lithosphere(mesh, plates, 0.45, 3, radius_km=radius_km)
    previous.sediment_volume_km3 = areas * rng.uniform(0.0, 2.0, n)
    state = deepcopy(previous)
    state.time_myr = 704.0
    state.continental_fraction = rng.uniform(0.0, 1.0, n)
    state.continental_fraction[::5] = 0.0
    state.continental_fraction[1::5] = 1.0
    state.continental_fraction[2::17] = 1.0e-13
    state.continental_volume_km3 = (
        areas * state.continental_fraction * rng.uniform(0.05, 45.0, n)
    )
    # Material fraction and visible crust type are deliberately independent.
    state.crust_type = (np.arange(n) % 3 == 0).astype(np.int8)
    state.crust_thickness_km = rng.uniform(5.0, 45.0, n)
    topography = TopographyState(704.0, rng.uniform(-1200.0, 3500.0, n))
    source = np.arange(n, dtype=np.int32)
    source[0] = 1  # Split one source across two targets; source 0 is lost.
    source[2] = -1
    source[3] = n + 7
    budget = SedimentBudgetState(700.0, 123.5, 42.25, 7.125, 3.0)
    erosivity = rng.uniform(-0.25, 3.0, n)
    return mesh, {
        "previous_lithosphere": previous,
        "state": state,
        "topography": topography,
        "source_index": source,
        "budget": budget,
        "dt_myr": 4.0,
        "radius_km": radius_km,
        "params": SedimentParameters(),
        "rift_recycled_volume_km3": 11.75,
        "erosivity_field": erosivity,
        "sea_level_m": 150.0,
    }


def _assert_dataclass_exact(actual, expected):
    assert type(actual) is type(expected)
    for field in fields(expected):
        candidate = getattr(actual, field.name)
        reference = getattr(expected, field.name)
        if isinstance(reference, np.ndarray):
            assert isinstance(candidate, np.ndarray), field.name
            assert candidate.shape == reference.shape, field.name
            assert candidate.dtype == reference.dtype, field.name
            assert candidate.tobytes() == reference.tobytes(), field.name
        else:
            assert candidate == reference, field.name


def _assert_result_exact(actual, expected):
    for candidate, reference in zip(actual, expected, strict=True):
        _assert_dataclass_exact(candidate, reference)


def _cpu_result(mesh, inputs):
    with CpuExecution(cell_kernels=True):
        return advance_sediments(mesh, **deepcopy(inputs))


def _configure_spill(mesh, inputs):
    areas = mesh.physical_cell_areas_km2(inputs["radius_km"])
    neighbors = np.asarray(mesh.neighbors)
    # The central cell receives from source IDs both below and above its own,
    # while also spilling itself. This exercises CPU scatter accumulation order.
    center = next(
        i for i, row in enumerate(neighbors)
        if np.any(row < i) and np.any(row > i)
    )
    lower_id = int(neighbors[center][neighbors[center] < center][0])
    higher_id = int(neighbors[center][neighbors[center] > center][0])
    thickness = np.full(mesh.cell_count, 0.5)
    thickness[[lower_id, center, higher_id]] = [30.0, 20.0, 35.0]
    inputs["previous_lithosphere"].sediment_volume_km3 = areas * thickness
    inputs["source_index"] = np.arange(mesh.cell_count, dtype=np.int32)
    inputs["topography"].elevation_m[:] = 0.0
    inputs["params"] = replace(
        inputs["params"], erosion_diffusion_per_myr=0.0,
        sediment_reworking_rate_per_myr=0.0,
    )


@pytest.mark.parametrize(
    "kind", ["mixed", "disabled", "no_sweeps", "spill", "legacy", "clipped_deposition"]
)
def test_gpu_surface_public_step_is_byte_exact(kind):
    _cuda_or_skip()
    from tectonics.gpu_runtime import GpuExecution

    mesh, inputs = _case()
    if kind == "disabled":
        inputs["params"] = replace(inputs["params"], enabled=False)
    elif kind == "no_sweeps":
        inputs["params"] = replace(inputs["params"], routing_sweeps=0)
    elif kind == "spill":
        _configure_spill(mesh, inputs)
    elif kind == "legacy":
        inputs["state"].continental_fraction = None
        inputs["state"].continental_volume_km3 = None
        inputs["previous_lithosphere"].sediment_volume_km3 = None
        inputs["erosivity_field"] = None
    elif kind == "clipped_deposition":
        inputs["params"] = replace(
            inputs["params"], land_deposition_fraction_per_sweep=-0.2,
            basin_deposition_fraction_per_sweep=1.2,
        )
    expected = _cpu_result(mesh, inputs)
    candidate_inputs = deepcopy(inputs)
    with CpuExecution(cell_kernels=True), GpuExecution(surface_pipeline=True) as execution:
        actual = advance_sediments(mesh, **candidate_inputs)
        report = execution.report()["surface_pipeline"]
    _assert_result_exact(actual, expected)
    assert actual[0] is candidate_inputs["state"]
    for name in ("previous_lithosphere", "topography", "budget"):
        _assert_dataclass_exact(candidate_inputs[name], inputs[name])
    np.testing.assert_array_equal(candidate_inputs["source_index"], inputs["source_index"])
    if inputs["erosivity_field"] is not None:
        np.testing.assert_array_equal(candidate_inputs["erosivity_field"], inputs["erosivity_field"])
    if kind == "disabled":
        assert report is None
    else:
        assert report["calls"] == 1
    if kind == "disabled":
        assert actual[1] is candidate_inputs["topography"]
        assert actual[3].eroded_bedrock_volume_km3 == 0.0
    if kind == "mixed":
        assert actual[3].transported_to_deep_reservoir_km3 > 0.0
        assert actual[3].eroded_bedrock_volume_km3 > 0.0
        assert actual[3].reworked_sediment_volume_km3 > 0.0
    if kind == "spill":
        initial_sediment = inputs["previous_lithosphere"].sediment_volume_km3
        assert not np.array_equal(actual[0].sediment_volume_km3, initial_sediment)
        assert actual[3].max_sediment_thickness_m > 10000.0


def test_gpu_surface_reuses_workspace_but_refreshes_all_step_inputs():
    _cuda_or_skip()
    from tectonics.gpu_runtime import GpuExecution

    mesh, first = _case()
    _, second = _case(mesh, seed=20260906)
    second["state"].cell_plate[:] = np.arange(mesh.cell_count, dtype=np.int32) % 7 + 50
    second["source_index"] = np.roll(np.arange(mesh.cell_count, dtype=np.int32), 11)
    second["state"].time_myr += 4.0
    _, third = _case(mesh, seed=20260907, radius_km=RADIUS_KM * 0.7)
    cases = [first, second, third]
    expected = [_cpu_result(mesh, case) for case in cases]
    reports = []
    actual = []
    with GpuExecution(surface_pipeline=True) as execution:
        for case in cases:
            actual.append(advance_sediments(mesh, **deepcopy(case)))
            reports.append(execution.report()["surface_pipeline"].copy())
        assert len(execution._meshes) == 1
    for candidate, reference in zip(actual, expected, strict=True):
        _assert_result_exact(candidate, reference)
    assert reports[-1]["calls"] == len(cases)
    assert reports[0]["buffer_allocations"] > 0
    assert reports[1]["buffer_allocations"] == reports[0]["buffer_allocations"]


@pytest.mark.parametrize("bad_shape", [(1,), (320, 1)])
def test_gpu_surface_invalid_erosivity_does_not_mutate_inputs(bad_shape):
    _cuda_or_skip()
    from tectonics.gpu_runtime import GpuExecution

    mesh, inputs = _case()
    inputs["erosivity_field"] = np.ones(bad_shape)
    before = deepcopy(inputs)
    with GpuExecution(surface_pipeline=True):
        with pytest.raises(ValueError, match="erosivity_field must match cell count"):
            advance_sediments(mesh, **inputs)
    for name in ("previous_lithosphere", "state", "topography", "budget"):
        _assert_dataclass_exact(inputs[name], before[name])


def test_gpu_surface_is_explicitly_opt_in():
    # Construction must not import CuPy or initialize the CUDA runtime.
    from tectonics.gpu_runtime import GpuExecution, current_execution

    assert current_execution() is None
    assert GpuExecution().surface_pipeline is False
    assert GpuExecution(surface_pipeline=True).surface_pipeline is True


def test_gpu_surface_dense_mesh_is_byte_exact():
    _cuda_or_skip()
    from tectonics.gpu_runtime import GpuExecution

    mesh, inputs = _case(build_icosphere(6))
    assert mesh.cell_count == 81920
    expected = _cpu_result(mesh, inputs)
    with GpuExecution(surface_pipeline=True) as execution:
        actual = advance_sediments(mesh, **deepcopy(inputs))
        report = execution.report()["surface_pipeline"]
    _assert_result_exact(actual, expected)
    assert report["calls"] == 1
    assert report["workspace_bytes"] == 20 * mesh.cell_count * 8
    assert report["host_upload_bytes"] == report["host_download_bytes"] == 6 * mesh.cell_count * 8


@pytest.mark.parametrize("invalid", ["topography", "parameters"])
def test_gpu_surface_nonfinite_values_do_not_mutate_state(invalid):
    _cuda_or_skip()
    from tectonics.gpu_runtime import GpuExecution

    mesh, inputs = _case()
    if invalid == "topography":
        inputs["topography"].elevation_m[0] = np.nan
    else:
        inputs["params"] = replace(inputs["params"], erosion_diffusion_per_myr=np.inf)
    before = deepcopy(inputs)
    with GpuExecution(surface_pipeline=True):
        with pytest.raises(ValueError, match="must be finite"):
            advance_sediments(mesh, **inputs)
    for name in ("previous_lithosphere", "state", "topography", "budget"):
        _assert_dataclass_exact(inputs[name], before[name])
