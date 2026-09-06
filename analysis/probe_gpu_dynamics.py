"""Time exact-order GPE torque candidates against real checkpoint inputs.

This is not a whole-run speedup measurement. Both candidate paths reuse static
geometry; its full cold construction cost is recorded separately. The CUDA
steady samples include dynamic input validation/grouping, uploads, ordered
reduction, download and the same NumPy final normalisation as the reference.
"""
from __future__ import annotations

import argparse
import hashlib
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
from tectonics.dynamics import DynamicsParameters
from tectonics.gpu_dynamics import GpeGeometry, GpuGpeProbe, gpe_reference, gpe_prepared_cpu
from tectonics.simulation import build_prototype, load_config
from tectonics.topology import PlateTopologyManager, PlateTopologyParameters


def _parameters(kind, values):
    return kind(**{name: values[name] for name in kind.__dataclass_fields__ if name in values})


def _hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _comparison(reference, candidate):
    return {
        name: {
            "byte_exact": (expected.shape == actual.shape and expected.dtype == actual.dtype
                           and expected.tobytes() == actual.tobytes()),
            "different_values": int(np.count_nonzero(expected != actual)),
            "max_absolute_difference": float(np.max(np.abs(expected - actual), initial=0.0)),
        }
        for name, expected, actual in zip(("drive", "weight"), reference, candidate)
    }


def _measure(call, reference, repeat):
    samples = []
    comparisons = []
    for _ in range(repeat):
        start = perf_counter()
        result = call()
        samples.append(perf_counter() - start)
        comparison = _comparison(reference, result)
        comparisons.append(comparison)
        if not all(value["byte_exact"] for value in comparison.values()):
            break
    return {
        "wall_seconds": samples,
        "median_seconds": statistics.median(samples),
        "comparison": comparisons[-1],
        "comparisons": comparisons,
        "all_outputs_byte_exact": all(
            value["byte_exact"] for item in comparisons for value in item.values()
        ),
    }


def _record_speedup(case, key, cold_cost):
    """Never publish a speedup estimate for a numerically rejected candidate."""
    candidate = case[key]
    exact = case["reference"]["all_outputs_byte_exact"] and candidate["all_outputs_byte_exact"]
    if key == "gpu":
        exact = exact and all(value["byte_exact"] for value in case["gpu_first_comparison"].values())
    candidate["eligible_for_exact_integration"] = exact
    candidate["loop_speedup"] = None
    candidate["estimated_calls_to_amortize_cold_cost"] = None
    if exact:
        saving = case["reference"]["median_seconds"] - candidate["median_seconds"]
        candidate["loop_speedup"] = case["reference"]["median_seconds"] / candidate["median_seconds"]
        candidate["estimated_calls_to_amortize_cold_cost"] = cold_cost / saving if saving > 0.0 else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.gpu_device < 0:
        parser.error("--repeat must be positive and --gpu-device non-negative")
    if args.output.exists():
        parser.error("--output already exists")

    source_paths = [args.config.resolve(), ROOT / "tectonics" / "dynamics.py"]
    source_paths.extend(path.resolve() / name for path in args.checkpoint for name in ("meta.json", "state.npz"))
    hashes_before = {str(path): _hash_file(path) for path in source_paths}
    config = load_config(args.config)
    mesh = build_prototype(config).mesh
    params = _parameters(DynamicsParameters, config.get("plate_dynamics", {}))
    topo_params = _parameters(PlateTopologyParameters, config.get("plate_topology", {}))
    start = perf_counter()
    geometry = GpeGeometry.prepare(mesh)
    geometry_seconds = perf_counter() - start
    gpu = None
    gpu_init_seconds = None
    if not args.cpu_only:
        start = perf_counter()
        gpu = GpuGpeProbe(geometry, args.gpu_device)
        gpu_init_seconds = perf_counter() - start
    report = {
        "scope": "GPE neighbour loop only; not an integrated dynamics or full-run gain",
        "config": str(args.config.resolve()),
        "config_sha256": hashes_before[str(args.config.resolve())],
        "reference_source_sha256": hashes_before[str(ROOT / "tectonics" / "dynamics.py")],
        "cells": mesh.cell_count,
        "edges": len(geometry.source),
        "reference_thickness_km": params.gpe_reference_thickness_km,
        "geometry_preparation_seconds": geometry_seconds,
        "geometry_host_bytes": geometry.host_bytes,
        "gpu_initialization_seconds": gpu_init_seconds,
        "repeat": args.repeat,
        "gpu_sample_scope": "host validation/grouping + evolving uploads + kernels + download + NumPy normalisation",
        "exactness_method": "scalar NumPy static geometry, FP64 no FMA, stable cell/neighbour ordered sums",
        "cases": [],
    }
    if gpu is not None:
        props = gpu.cp.cuda.runtime.getDeviceProperties(args.gpu_device)
        name = props["name"]
        report["gpu_name"] = name.decode() if isinstance(name, bytes) else str(name)
    for checkpoint_path in args.checkpoint:
        checkpoint_hashes_before = {name: hashes_before[str(checkpoint_path.resolve() / name)]
                                    for name in ("meta.json", "state.npz")}
        checkpoint = load_checkpoint(checkpoint_path, PlateTopologyManager(topo_params))
        state = checkpoint.state
        plates = len(checkpoint.system.plates)
        if len(state.cell_plate) != mesh.cell_count:
            raise ValueError(f"Checkpoint mesh size differs from configuration: {checkpoint_path}")
        inputs = (mesh, state, plates, params.gpe_reference_thickness_km)
        candidate_inputs = (geometry, state, plates, params.gpe_reference_thickness_km)
        reference = gpe_reference(*inputs)
        case = {
            "checkpoint": str(checkpoint_path.resolve()),
            "time_myr": state.time_myr,
            "plates": plates,
            "nonzero_weight_plates": int(np.count_nonzero(reference[1])),
            "reference": _measure(lambda: gpe_reference(*inputs), reference, args.repeat),
            "prepared_cpu": _measure(lambda: gpe_prepared_cpu(*candidate_inputs), reference, args.repeat),
            "input_sha256": checkpoint_hashes_before,
        }
        if gpu is not None:
            start = perf_counter()
            first = gpu.calculate(state, plates, params.gpe_reference_thickness_km)
            case["gpu_first_call_seconds"] = perf_counter() - start
            case["gpu_first_comparison"] = _comparison(reference, first)
            case["gpu"] = _measure(
                lambda: gpu.calculate(state, plates, params.gpe_reference_thickness_km), reference, args.repeat
            )
        for key in ("prepared_cpu", "gpu"):
            if key not in case:
                continue
            cold_cost = geometry_seconds + ((gpu_init_seconds or 0.0) if key == "gpu" else 0.0)
            if key == "gpu":
                cold_cost += case["gpu_first_call_seconds"]
            _record_speedup(case, key, cold_cost)
        checkpoint_hashes_after = {name: _hash_file(checkpoint_path / name) for name in checkpoint_hashes_before}
        case["input_files_unchanged"] = checkpoint_hashes_before == checkpoint_hashes_after
        report["cases"].append(case)
        print(f"Checkpoint {state.time_myr:g} Myr: reference="
              f"{case['reference']['median_seconds'] * 1000:.3f} ms", flush=True)
        for key in ("prepared_cpu", "gpu"):
            if key in case:
                print(f"  {key}: {case[key]['median_seconds'] * 1000:.3f} ms; "
                      f"byte_exact={case[key]['eligible_for_exact_integration']}", flush=True)
    report["hashes_before"] = hashes_before
    report["hashes_after"] = {str(path): _hash_file(path) for path in source_paths}
    report["inputs_unchanged"] = hashes_before == report["hashes_after"]
    if not report["inputs_unchanged"]:
        for case in report["cases"]:
            for key in ("prepared_cpu", "gpu"):
                if key in case:
                    case[key]["eligible_for_exact_integration"] = False
                    case[key]["loop_speedup"] = None
                    case[key]["estimated_calls_to_amortize_cold_cost"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(f"Report: {args.output.resolve()}", flush=True)
    if not report["inputs_unchanged"]:
        raise SystemExit("Input configuration, checkpoint or reference source changed during probe")
    if any(
        not case[key]["eligible_for_exact_integration"]
        for case in report["cases"] for key in ("prepared_cpu", "gpu") if key in case
    ):
        raise SystemExit("A candidate differs from the reference; it is not eligible for exact integration")


if __name__ == "__main__":
    main()
