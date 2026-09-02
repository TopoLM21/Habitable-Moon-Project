"""Isolated FP64 GPU feasibility probe, NOT a simulation backend or GUI option.

Compare the original CPU loop, batched CPU, and deterministic GPU gather on the
same fixed inputs. Report cold setup, warm resident work, and transfers separately.
Requires CuPy only in the separate GPU-probe environment.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from tectonics.mesh import build_icosphere
from tectonics.sediment import SedimentParameters, _route_mobile
from tectonics.sediment_kernels import route_mobile_batched


CUDA_SOURCE = r'''
extern "C" __global__ void emit_flux(
    const double* z, const int* neighbors, const double* mobile,
    double* sediment, double* edges, int n, double sea, double land, double basin) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double v = mobile[i];
    double drops[3];
    double total = 0.0;
    bool downhill = false;
    for (int k = 0; k < 3; ++k) {
        edges[3*i+k] = 0.0;
        double d = z[i] - z[neighbors[3*i+k]];
        drops[k] = d > 1.0e-9 ? d : 0.0;
        total += drops[k];
        downhill = downhill || d > 1.0e-9;
    }
    if (!(v > 0.0)) return;
    if (!downhill) { sediment[i] += v; return; }
    double dep = z[i] <= sea ? basin : land;
    dep = dep < 0.0 ? 0.0 : (dep > 1.0 ? 1.0 : dep);
    sediment[i] += v * dep;
    double move = v * (1.0 - dep);
    double denominator = total > 1.0e-30 ? total : 1.0e-30;
    for (int k = 0; k < 3; ++k)
        if (drops[k] > 0.0) edges[3*i+k] = move * (drops[k] / denominator);
}
extern "C" __global__ void gather_flux(
    const double* edges, const int* incoming, double* next, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double total = 0.0;
    // Source IDs are sorted, retaining the CPU scatter-add accumulation order.
    for (int k = 0; k < 3; ++k) total += edges[incoming[3*i+k]];
    next[i] = total;
}
'''


class GpuRoutingProbe:
    def __init__(self, mesh):
        import cupy as cp
        self.cp = cp
        nbs = np.asarray(mesh.neighbors, dtype=np.int32)
        if nbs.shape != (mesh.cell_count, 3):
            raise ValueError("GPU probe requires a closed triangular mesh")
        sources = np.sort(nbs, axis=1)
        matches = nbs[sources] == np.arange(mesh.cell_count)[:, None, None]
        if not np.all(matches.sum(axis=2) == 1):
            raise ValueError("GPU gather requires reciprocal, unique neighbours")
        incoming = sources * 3 + np.argmax(matches, axis=2)
        self.n = mesh.cell_count
        self.neighbors = cp.asarray(nbs)
        self.incoming = cp.asarray(incoming.astype(np.int32))
        module = cp.RawModule(code=CUDA_SOURCE, options=("--std=c++17", "--fmad=false"))
        self.emit = module.get_function("emit_flux")
        self.gather = module.get_function("gather_flux")

    def run(self, z, stationary, mobile, params, sea_level_m):
        cp = self.cp
        z = cp.asarray(z, dtype=cp.float64)
        sed = cp.asarray(stationary, dtype=cp.float64).copy()
        mob = cp.asarray(mobile, dtype=cp.float64).copy()
        next_mob = cp.empty_like(mob)
        edges = cp.empty((self.n, 3), dtype=cp.float64)
        grid, block = ((self.n + 255) // 256,), (256,)
        for _ in range(max(int(params.routing_sweeps), 0)):
            self.emit(grid, block, (z, self.neighbors, mob, sed, edges, np.int32(self.n),
                      np.float64(sea_level_m), np.float64(params.land_deposition_fraction_per_sweep),
                      np.float64(params.basin_deposition_fraction_per_sweep)))
            self.gather(grid, block, (edges, self.incoming, next_mob, np.int32(self.n)))
            mob, next_mob = next_mob, mob
            total = float(cp.sum(mob).get())
            # Near the stopping threshold use the exact NumPy reduction order.
            # All physical fluxes are non-negative; far above it reduction
            # roundoff cannot change the decision by a factor of four.
            if total <= 4e-12:
                total = float(np.sum(mob.get()))
            if total <= 1e-12:
                break
        return sed + mob


def comparison(expected, actual):
    difference = np.abs(expected - actual)
    return {"byte_exact": expected.dtype == actual.dtype and expected.tobytes() == actual.tobytes(),
            "different_cells": int(np.count_nonzero(expected != actual)),
            "max_absolute_difference": float(np.max(difference)),
            "sum_difference": float(np.sum(actual) - np.sum(expected))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subdivisions", nargs="+", type=int, default=[5, 6, 7])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--performance-cores", action="store_true")
    args = parser.parse_args()
    from analysis.windows_cpu_affinity import performance_core_benchmark
    with performance_core_benchmark(args.performance_cores) as affinity:
        measure(args, affinity)


def measure(args, affinity):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cold = perf_counter()
    import cupy as cp
    properties = cp.cuda.runtime.getDeviceProperties(0)
    report = {"gpu": properties["name"].decode(), "cupy": cp.__version__, "numpy": np.__version__,
              "backend_is_integrated": False, "float_dtype": "float64", "cpu_environment": affinity, "cases": []}
    params = SedimentParameters()
    small = build_icosphere(1)
    first_probe = GpuRoutingProbe(small)
    rng = np.random.default_rng(31)
    checks = []
    for kind in ("normal", "flat", "empty", "tiny", "no_sweeps", "clipped_deposition"):
        z = rng.normal(0, 100, small.cell_count)
        stationary = rng.uniform(0, 1000, small.cell_count)
        mobile = rng.uniform(0, 100, small.cell_count)
        case_params = params
        if kind == "flat": z[:] = 0
        if kind == "empty": mobile[:] = 0
        if kind == "tiny": mobile *= 1e-25
        if kind == "no_sweeps": case_params = replace(params, routing_sweeps=0)
        if kind == "clipped_deposition":
            case_params = replace(params, land_deposition_fraction_per_sweep=-0.2, basin_deposition_fraction_per_sweep=1.2)
        expected = _route_mobile(small, z, stationary, mobile, case_params, 0.0)
        actual = first_probe.run(z, stationary, mobile, case_params, 0.0).get()
        checks.append({"kind": kind, **comparison(expected, actual)})
    report["small_validation_cases"] = checks
    report["startup_compilation_and_small_validation_seconds"] = perf_counter() - cold
    (output / "measurements.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for subdivisions in args.subdivisions:
        mesh_started = perf_counter()
        mesh = build_icosphere(subdivisions)
        mesh_seconds = perf_counter() - mesh_started
        rng = np.random.default_rng(20260902 + subdivisions)
        xyz = mesh.centroids
        z = 3000 * (xyz[:, 2] + 0.2*np.sin(8*xyz[:, 0])) + rng.normal(0, 100, mesh.cell_count)
        stationary = rng.uniform(0, 10000, mesh.cell_count)
        mobile = rng.uniform(0, 1000, mesh.cell_count)
        mobile[::5] = 0
        started = perf_counter()
        expected = _route_mobile(mesh, z, stationary, mobile, params, 0.0)
        original_seconds = perf_counter() - started
        batch_times = []
        for _ in range(args.repeat):
            started = perf_counter()
            batched = route_mobile_batched(mesh, z, stationary, mobile, params, 0.0)
            batch_times.append(perf_counter() - started)
        if not comparison(expected, batched)["byte_exact"]:
            raise RuntimeError("Batched CPU reference changed")
        started = perf_counter()
        probe = GpuRoutingProbe(mesh)
        actual = probe.run(z, stationary, mobile, params, 0.0).get()
        cold_mesh_seconds = perf_counter() - started
        if not report["cases"]:
            report["first_case_elapsed_including_cpu_reference_seconds"] = perf_counter() - cold
        dz, ds, dm = [cp.asarray(value) for value in (z, stationary, mobile)]
        resident_times, transfer_times = [], []
        for _ in range(args.repeat):
            cp.cuda.get_current_stream().synchronize()
            started = perf_counter()
            probe.run(dz, ds, dm, params, 0.0)
            cp.cuda.get_current_stream().synchronize()
            resident_times.append(perf_counter() - started)
            started = perf_counter()
            transferred = probe.run(z, stationary, mobile, params, 0.0).get()
            transfer_times.append(perf_counter() - started)
            if not np.array_equal(actual, transferred):
                raise RuntimeError("GPU output is not repeatable")
        case = {"subdivisions": subdivisions, "cells": mesh.cell_count,
                "cpu_mesh_setup_seconds": mesh_seconds, "original_cpu_seconds": original_seconds,
                "batched_cpu_seconds": batch_times, "gpu_cold_mesh_and_first_call_seconds": cold_mesh_seconds,
                "gpu_resident_seconds": resident_times, "gpu_with_field_transfers_seconds": transfer_times,
                "comparison": comparison(expected, actual),
                "median_batched_cpu_seconds": statistics.median(batch_times),
                "median_gpu_with_transfers_seconds": statistics.median(transfer_times)}
        report["cases"].append(case)
        (output / "measurements.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(case, indent=2), flush=True)


if __name__ == "__main__":
    main()
