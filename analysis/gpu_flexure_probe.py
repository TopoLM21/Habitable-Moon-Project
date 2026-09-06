"""Compare the CPU flexure solve with the experimental CUDA CG implementation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
from tectonics.flexure import FlexureParameters, solve_flexural_response
from tectonics.gpu_flexure import GpuFlexureSolver, solve_flexural_response_gpu
from tectonics.gpu_runtime import GpuExecution
from tectonics.simulation import build_prototype, load_config
from tectonics.topology import PlateTopologyManager, PlateTopologyParameters


def dataclass_from_config(kind, values):
    return kind(**{name: values[name] for name in kind.__dataclass_fields__ if name in values})


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
    params = dataclass_from_config(FlexureParameters, config.get("flexure", {}))
    radius_km = float(config["moon"]["radius_km"])
    gravity = float(config["moon"]["surface_gravity_m_s2"])
    # A checkpointed real relief field supplies realistic length scales and
    # amplitudes without modifying or rerunning the trajectory.
    local_target = np.asarray(checkpoint.topo.elevation_m, dtype=np.float64)

    solve_flexural_response(
        prototype.mesh, checkpoint.state, local_target, radius_km, gravity, params
    )
    cpu_times = []
    cpu_result = None
    cpu_diagnostics = None
    for _ in range(args.repeat):
        started = perf_counter()
        cpu_result, cpu_diagnostics, _, _ = solve_flexural_response(
            prototype.mesh, checkpoint.state, local_target, radius_km, gravity, params
        )
        cpu_times.append(perf_counter() - started)

    with GpuExecution(args.gpu_device) as execution:
        solver = GpuFlexureSolver(execution, prototype.mesh, radius_km)
        solve_flexural_response_gpu(
            execution, solver, prototype.mesh, checkpoint.state,
            local_target, radius_km, gravity, params,
        )
        gpu_times = []
        gpu_result = None
        gpu_diagnostics = None
        gpu_report = None
        for _ in range(args.repeat):
            started = perf_counter()
            gpu_result, gpu_diagnostics, _, _, gpu_report = solve_flexural_response_gpu(
                execution, solver, prototype.mesh, checkpoint.state,
                local_target, radius_km, gravity, params,
            )
            gpu_times.append(perf_counter() - started)
        device = execution.report()

    assert cpu_result is not None and gpu_result is not None
    difference = gpu_result - cpu_result
    scale = max(float(np.max(np.abs(cpu_result))), 1.0e-300)
    report = {
        "input": {
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "cells": prototype.mesh.cell_count,
            "repeat": args.repeat,
        },
        "device": device,
        "cpu_seconds": cpu_times,
        "gpu_seconds": gpu_times,
        "cpu_median_seconds": statistics.median(cpu_times),
        "gpu_median_seconds": statistics.median(gpu_times),
        "speedup": statistics.median(cpu_times) / statistics.median(gpu_times),
        "comparison": {
            "byte_exact": cpu_result.tobytes() == gpu_result.tobytes(),
            "max_absolute_difference_m": float(np.max(np.abs(difference))),
            "rms_difference_m": float(np.sqrt(np.mean(difference * difference))),
            "max_relative_to_peak": float(np.max(np.abs(difference)) / scale),
            "mean_difference_m": float(np.mean(difference)),
        },
        "cpu_diagnostics": asdict(cpu_diagnostics),
        "gpu_diagnostics": asdict(gpu_diagnostics),
        "gpu_cg": asdict(gpu_report),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
