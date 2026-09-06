from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tectonics.mesh import build_icosphere
from tectonics.sediment import SedimentParameters, _route_mobile


def _cuda_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    return cp


def test_gpu_routing_matches_original_byte_for_byte():
    _cuda_or_skip()
    from tectonics.gpu_runtime import GpuExecution, current_execution

    mesh = build_icosphere(2)
    rng = np.random.default_rng(20260905)
    cases = []
    params = SedimentParameters()
    for kind in ("normal", "flat", "empty", "tiny", "no_sweeps", "clipped_deposition"):
        z = rng.normal(0.0, 100.0, mesh.cell_count)
        stationary = rng.uniform(0.0, 1000.0, mesh.cell_count)
        mobile = rng.uniform(0.0, 100.0, mesh.cell_count)
        case_params = params
        if kind == "flat":
            z[:] = 0.0
        elif kind == "empty":
            mobile[:] = 0.0
        elif kind == "tiny":
            mobile *= 1.0e-25
        elif kind == "no_sweeps":
            case_params = replace(params, routing_sweeps=0)
        elif kind == "clipped_deposition":
            case_params = replace(
                params,
                land_deposition_fraction_per_sweep=-0.2,
                basin_deposition_fraction_per_sweep=1.2,
            )
        cases.append((z, stationary, mobile, case_params))

    expected = [_route_mobile(mesh, *case, 0.0) for case in cases]
    with GpuExecution() as execution:
        actual = [_route_mobile(mesh, *case, 0.0) for case in cases]
        assert current_execution() is execution
        assert execution.routing_calls == len(cases)
        assert len(execution._meshes) == 1
    assert current_execution() is None
    for reference, candidate in zip(expected, actual, strict=True):
        assert reference.dtype == candidate.dtype == np.float64
        assert reference.tobytes() == candidate.tobytes()


def test_gpu_execution_is_opt_in_and_rejects_nesting():
    _cuda_or_skip()
    from tectonics.gpu_runtime import GpuExecution, current_execution

    with GpuExecution() as execution:
        assert current_execution() is execution
        with pytest.raises(RuntimeError, match="already active"):
            with GpuExecution():
                pass
    assert current_execution() is None


def test_gpu_routing_precedes_cpu_cell_kernel(monkeypatch):
    import tectonics.gpu_runtime as gpu_runtime

    class FakeGpu:
        def route_mobile(self, *args):
            return np.array([456.0])

    monkeypatch.setattr(gpu_runtime, "_active", FakeGpu())
    mesh = build_icosphere(0)
    inputs = mesh, np.zeros(1), np.ones(1), np.zeros(1), SedimentParameters(), 0.0
    assert _route_mobile(*inputs)[0] == 456.0


def test_gpu_arc_painting_matches_cpu_with_strict_tolerance():
    _cuda_or_skip()
    from scipy.spatial import cKDTree
    from types import SimpleNamespace

    from tectonics.gpu_runtime import GpuExecution
    from tectonics.volcanic_arc import _paint_gaussian_batch

    mesh = build_icosphere(3)
    owner = np.arange(mesh.cell_count, dtype=np.int32) % 3
    state = SimpleNamespace(cell_plate=owner)
    centers = [mesh.centroids[11], mesh.centroids[217], mesh.centroids[901]]
    plates = [int(owner[11]), int(owner[217]), int(owner[901])]
    amplitudes = [0.75, 1.1, 0.4]
    radius_km = 1737.4
    sigma_km = 140.0
    outer_km = 320.0
    expected = np.zeros(mesh.cell_count, dtype=np.float64)
    _paint_gaussian_batch(
        mesh, cKDTree(mesh.centroids), centers, plates, amplitudes, state,
        radius_km, sigma_km, outer_km, expected, 1,
    )
    batches = [{
        "centers": centers,
        "plates": plates,
        "amplitudes": amplitudes,
        "sigma_km": sigma_km,
        "outer_km": outer_km,
    }]
    with GpuExecution(arc_painting=True) as execution:
        actual = execution.paint_volcanic_arcs(mesh, owner, radius_km, batches)
        report = execution.report()
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-14)
    assert np.array_equal(actual > 0.05, expected > 0.05)
    assert report["arc_calls"] == 1 and report["arc_tasks"] == len(centers)
    assert report["tolerance_kernels"] == ["volcanic_arc_painting"]
