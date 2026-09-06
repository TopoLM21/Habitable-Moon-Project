"""Measure a brute-force CUDA painter on real volcanic-arc tasks.

The existing CPU path still prepares the physical arc centers.  CUDA evaluates
all center/cell candidates and takes one deterministic maximum per cell, avoiding
thousands of tiny NumPy calls and unordered atomics.  This is an isolated
candidate because CUDA transcendental functions are not byte-identical to NumPy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tectonics.checkpoint import load_checkpoint
from tectonics.cpu_runtime import CpuExecution
from tectonics.lithosphere import boundary_records_for_state
from tectonics.simulation import build_prototype, load_config
from tectonics.topology import PlateTopologyManager, PlateTopologyParameters
from tectonics.volcanic_arc import VolcanicArcParameters
import tectonics.volcanic_arc as volcanic_arc


CUDA_SOURCE = r'''
extern "C" __global__ void paint_arcs(
    const double* centroids, const int* owner,
    const double* centers, const int* plates, const double* amplitudes,
    const double* sigma, const double* minimum_dot,
    double* field, int cells, int tasks, double radius) {
    int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= cells) return;
    double x = centroids[3*cell];
    double y = centroids[3*cell+1];
    double z = centroids[3*cell+2];
    int plate = owner[cell];
    double best = 0.0;
    for (int task = 0; task < tasks; ++task) {
        if (plates[task] != plate || amplitudes[task] <= 0.0) continue;
        double dot = x*centers[3*task] + y*centers[3*task+1] + z*centers[3*task+2];
        if (dot < minimum_dot[task]) continue;
        dot = dot < -1.0 ? -1.0 : (dot > 1.0 ? 1.0 : dot);
        double distance = acos(dot) * radius;
        double scaled = distance / sigma[task];
        double value = amplitudes[task] * exp(-0.5 * scaled * scaled);
        best = best > value ? best : value;
    }
    field[cell] = best;
}
'''


def dataclass_from_config(kind, values):
    return kind(**{name: values[name] for name in kind.__dataclass_fields__ if name in values})


def capture_tasks(mesh, state, memory, radius_km, params, boundaries, *, paint=True):
    batches = []
    original = volcanic_arc._paint_gaussian_batch

    def capture(mesh_arg, tree, centers, plate_ids, amplitudes, state_arg,
                radius_arg, sigma_km, outer_km, field, workers):
        batches.append(
            {
                "centers": np.asarray(centers, dtype=np.float64),
                "plates": np.asarray(plate_ids, dtype=np.int32),
                "amplitudes": np.asarray(amplitudes, dtype=np.float64),
                "sigma_km": float(sigma_km),
                "outer_km": float(outer_km),
            }
        )
        if paint:
            return original(
                mesh_arg, tree, centers, plate_ids, amplitudes, state_arg,
                radius_arg, sigma_km, outer_km, field, workers,
            )
        return None

    volcanic_arc._paint_gaussian_batch = capture
    try:
        with CpuExecution(1, arc_kernels=True):
            field, diagnostics = volcanic_arc.compute_volcanic_arc_forcing(
                mesh, state, memory, radius_km, params, boundaries
            )
    finally:
        volcanic_arc._paint_gaussian_batch = original
    return field, diagnostics, batches


def prepare_arrays(batches, radius_km):
    nonempty = [batch for batch in batches if len(batch["centers"])]
    if not nonempty:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    centers = np.concatenate([batch["centers"] for batch in nonempty])
    plates = np.concatenate([batch["plates"] for batch in nonempty])
    amplitudes = np.concatenate([batch["amplitudes"] for batch in nonempty])
    sigma = np.concatenate([
        np.full(len(batch["centers"]), batch["sigma_km"], dtype=np.float64)
        for batch in nonempty
    ])
    minimum_dot = np.concatenate([
        np.full(
            len(batch["centers"]),
            np.cos(min(batch["outer_km"] / max(radius_km, 1.0e-9), np.pi)),
            dtype=np.float64,
        )
        for batch in nonempty
    ])
    return centers, plates, amplitudes, sigma, minimum_dot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--gpu-device", type=int, default=0)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("--output already exists")

    config = load_config(args.config)
    prototype = build_prototype(config)
    topology = dataclass_from_config(PlateTopologyParameters, config["plate_topology"])
    checkpoint = load_checkpoint(args.checkpoint, PlateTopologyManager(topology))
    params = dataclass_from_config(VolcanicArcParameters, config.get("volcanic_arc", {}))
    radius_km = float(config["moon"]["radius_km"])
    normal = float(config["classification"]["normal_threshold_km_per_myr"])
    inactive = float(config["classification"]["inactive_speed_km_per_myr"])
    boundaries = boundary_records_for_state(
        prototype.mesh, checkpoint.state, checkpoint.system,
        radius_km, normal, inactive,
    )

    cpu_field, diagnostics, batches = capture_tasks(
        prototype.mesh, checkpoint.state, checkpoint.subduction_memory,
        radius_km, params, boundaries,
    )
    centers, plates, amplitudes, sigma, minimum_dot = prepare_arrays(batches, radius_km)

    cpu_times = []
    for _ in range(args.repeat):
        started = perf_counter()
        candidate, _, _ = capture_tasks(
            prototype.mesh, checkpoint.state, checkpoint.subduction_memory,
            radius_km, params, boundaries,
        )
        cpu_times.append(perf_counter() - started)
        if candidate.tobytes() != cpu_field.tobytes():
            raise RuntimeError("CPU arc result was not repeatable")

    preparation_times = []
    for _ in range(args.repeat):
        started = perf_counter()
        _, _, prepared = capture_tasks(
            prototype.mesh, checkpoint.state, checkpoint.subduction_memory,
            radius_km, params, boundaries, paint=False,
        )
        preparation_times.append(perf_counter() - started)
        if sum(len(batch["centers"]) for batch in prepared) != len(centers):
            raise RuntimeError("Arc task preparation was not repeatable")

    import cupy as cp
    cp.cuda.Device(args.gpu_device).use()
    module = cp.RawModule(code=CUDA_SOURCE, options=("--std=c++17", "--fmad=false"))
    kernel = module.get_function("paint_arcs")
    device_centroids = cp.asarray(np.asarray(prototype.mesh.centroids, dtype=np.float64))
    device_owner = cp.asarray(np.asarray(checkpoint.state.cell_plate, dtype=np.int32))
    device_centers = cp.asarray(centers)
    device_plates = cp.asarray(plates)
    device_amplitudes = cp.asarray(amplitudes)
    device_sigma = cp.asarray(sigma)
    device_minimum_dot = cp.asarray(minimum_dot)
    device_field = cp.empty(prototype.mesh.cell_count, dtype=cp.float64)
    grid = ((prototype.mesh.cell_count + 255) // 256,)
    block = (256,)

    def run_gpu():
        kernel(
            grid,
            block,
            (
                device_centroids, device_owner, device_centers, device_plates,
                device_amplitudes, device_sigma, device_minimum_dot, device_field,
                np.int32(prototype.mesh.cell_count), np.int32(len(centers)),
                np.float64(radius_km),
            ),
        )
        return device_field.get()

    gpu_field = run_gpu()
    gpu_times = []
    for _ in range(args.repeat):
        started = perf_counter()
        gpu_field = run_gpu()
        gpu_times.append(perf_counter() - started)

    difference = gpu_field - cpu_field
    cpu_median = statistics.median(cpu_times)
    preparation_median = statistics.median(preparation_times)
    gpu_median = statistics.median(gpu_times)
    report = {
        "input": {
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "cells": prototype.mesh.cell_count,
            "boundaries": len(boundaries),
            "tasks": len(centers),
            "repeat": args.repeat,
        },
        "cpu_full_arc_seconds": cpu_times,
        "cpu_task_preparation_seconds": preparation_times,
        "gpu_paint_only_seconds": gpu_times,
        "cpu_full_arc_median_seconds": cpu_median,
        "cpu_task_preparation_median_seconds": preparation_median,
        "gpu_paint_only_median_seconds": gpu_median,
        "estimated_hybrid_median_seconds": preparation_median + gpu_median,
        "estimated_hybrid_speedup": cpu_median / (preparation_median + gpu_median),
        "comparison": {
            "byte_exact": cpu_field.tobytes() == gpu_field.tobytes(),
            "different_cells": int(np.count_nonzero(cpu_field != gpu_field)),
            "max_absolute_difference": float(np.max(np.abs(difference))),
            "rms_difference": float(np.sqrt(np.mean(difference * difference))),
            "forced_mask_equal": bool(np.array_equal(cpu_field > 0.05, gpu_field > 0.05)),
        },
        "cpu_diagnostics": {
            name: getattr(diagnostics, name) for name in diagnostics.__dataclass_fields__
        },
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
