"""Time the public sediment step on checkpoint fields with an identity source map.

This surface-only probe is not a tectonic timestep or a continued trajectory.
It includes host preparation inside advance_sediments, transfers, routing, spill,
relief response and diagnostic reductions. Loading, input deepcopy, context entry
and correctness comparisons are outside warm timings. Each sample starts from
the same checkpoint fields; no sample consumes the preceding sample's output.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, fields
from datetime import datetime, timezone
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
from tectonics.cpu_runtime import CpuExecution
from tectonics.gpu_runtime import GpuExecution
from tectonics.sediment import SedimentParameters, advance_sediments
from tectonics.simulation import build_initial_mesh, load_config
from tectonics.topology import PlateTopologyManager, PlateTopologyParameters


def dataclass_from_config(kind, values):
    return kind(**{field.name: values[field.name] for field in fields(kind) if field.name in values})


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_outputs(reference, candidate):
    """Check every returned dataclass field, including untouched state arrays."""
    differences = []
    arrays_checked = scalars_checked = 0
    for group, expected, actual in zip(
        ("state", "topography", "budget", "diagnostics"), reference, candidate, strict=True
    ):
        if type(expected) is not type(actual):
            differences.append({"field": group, "reason": "dataclass type mismatch"})
            continue
        for field in fields(expected):
            left, right = getattr(expected, field.name), getattr(actual, field.name)
            name = f"{group}.{field.name}"
            if isinstance(left, np.ndarray):
                arrays_checked += 1
                if not isinstance(right, np.ndarray) or left.shape != right.shape or left.dtype != right.dtype:
                    differences.append({"field": name, "reason": "array type, shape or dtype mismatch"})
                elif left.tobytes() != right.tobytes():
                    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
                    maximum = float(np.max(delta, initial=0.0))
                    differences.append({
                        "field": name,
                        "reason": "array bytes differ",
                        "different_values": int(np.count_nonzero(left != right)),
                        "max_absolute_difference": maximum if np.isfinite(maximum) else None,
                    })
            else:
                scalars_checked += 1
                if isinstance(left, (float, np.floating)):
                    equal = (isinstance(right, (float, np.floating))
                             and np.float64(left).tobytes() == np.float64(right).tobytes())
                else:
                    equal = type(left) is type(right) and left == right
                if not equal:
                    differences.append({"field": name, "reason": "scalar differs",
                                        "reference": repr(left), "candidate": repr(right)})
    return {"byte_exact": not differences, "arrays_checked": arrays_checked,
            "scalars_checked": scalars_checked, "differences": differences}


def measure_backend(name, mesh, template, repeat, device, reference=None):
    """Keep one execution context active for a cold call and warm repetitions."""
    comparisons = []
    warm_seconds = []
    with ExitStack() as stack:
        started = perf_counter()
        stack.enter_context(CpuExecution(cell_kernels=True))
        gpu = None
        if name != "cpu":
            gpu = stack.enter_context(GpuExecution(device, surface_pipeline=name == "gpu_surface"))
            gpu.cp.cuda.get_current_stream().synchronize()
        context_entry_seconds = perf_counter() - started

        for sample in range(repeat + 1):
            # Mutating the local state is part of the public API; replenish every
            # input before starting the timer instead of timing the deepcopy.
            inputs = deepcopy(template)
            if gpu is not None:
                gpu.cp.cuda.get_current_stream().synchronize()
            started = perf_counter()
            result = advance_sediments(mesh, **inputs)
            if gpu is not None:
                gpu.cp.cuda.get_current_stream().synchronize()
            elapsed = perf_counter() - started
            if reference is None:
                reference = result
            comparison = compare_outputs(reference, result)
            comparison["sample"] = "cold" if sample == 0 else sample
            comparisons.append(comparison)
            if sample == 0:
                cold_call_seconds = elapsed
            else:
                warm_seconds.append(elapsed)

        execution_report = gpu.report() if gpu is not None else {"backend": "cpu", "cell_kernels": True}
        diagnostics = asdict(result[3])
    median = statistics.median(warm_seconds)
    measurement = {
        "context_entry_seconds": context_entry_seconds,
        "cold_public_call_seconds": cold_call_seconds,
        "cold_context_and_call_seconds": context_entry_seconds + cold_call_seconds,
        "warm_public_call_seconds": warm_seconds,
        "median_warm_seconds": median,
        "minimum_warm_seconds": min(warm_seconds),
        "maximum_warm_seconds": max(warm_seconds),
        "all_outputs_byte_exact": all(item["byte_exact"] for item in comparisons),
        "comparisons": comparisons,
        "final_diagnostics": diagnostics,
        "execution_report": execution_report,
    }
    print(f"{name}: median {1000.0 * median:.3f} ms; "
          f"all outputs byte exact: {measurement['all_outputs_byte_exact']}", flush=True)
    return measurement, reference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "canonical_moon.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; existing paths are rejected")
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--dt", type=float, default=4.0)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--sea-level", type=float, default=None,
                        help="Override checkpoint's last recorded sea level in metres")
    parser.add_argument("--cpu-only", action="store_true", help="Verify the probe without initializing CUDA")
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if not np.isfinite(args.dt) or args.dt <= 0.0:
        parser.error("--dt must be positive and finite")
    if args.sea_level is not None and not np.isfinite(args.sea_level):
        parser.error("--sea-level must be finite")
    if args.gpu_device < 0:
        parser.error("--gpu-device must be non-negative")
    output = args.output.resolve()
    if output.exists():
        parser.error("--output already exists")

    setup_started = perf_counter()
    config = load_config(args.config)
    topology = dataclass_from_config(PlateTopologyParameters, config.get("plate_topology", {}))
    checkpoint = load_checkpoint(args.checkpoint, PlateTopologyManager(topology))
    mesh = build_initial_mesh(config)
    if len(checkpoint.state.cell_plate) != mesh.cell_count:
        parser.error("checkpoint and config mesh cell counts differ")
    if checkpoint.sediment_budget is None:
        parser.error("checkpoint must contain a sediment budget")
    params = dataclass_from_config(SedimentParameters, config.get("sediment", {}))
    if not params.enabled:
        parser.error("sediment is disabled in the supplied config")
    if args.sea_level is not None:
        sea_level = args.sea_level
        sea_level_source = "command_line"
    elif checkpoint.hydrosphere_rows and "sea_level_m" in checkpoint.hydrosphere_rows[-1]:
        sea_level = float(checkpoint.hydrosphere_rows[-1]["sea_level_m"])
        sea_level_source = "checkpoint_last_hydrosphere_row"
    else:
        sea_level = float(config.get("hydrosphere", {}).get("reference_sea_level_m", 0.0))
        sea_level_source = "config_reference_sea_level"
    state = deepcopy(checkpoint.state)
    state.time_myr += args.dt
    template = {
        "previous_lithosphere": checkpoint.state,
        "state": state,
        "topography": checkpoint.topo,
        "source_index": np.arange(mesh.cell_count, dtype=np.int32),
        "budget": checkpoint.sediment_budget,
        "dt_myr": args.dt,
        "radius_km": float(config["moon"]["radius_km"]),
        "params": params,
        "rift_recycled_volume_km3": 0.0,
        "erosivity_field": None,
        "sea_level_m": sea_level,
    }
    input_setup_seconds = perf_counter() - setup_started
    order = ["cpu"] if args.cpu_only else ["cpu", "gpu_routing", "gpu_surface"]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "public advance_sediments on checkpoint fields; surface-only, not a full timestep",
        "source_map": "identity; no lithosphere transport, gaps, overlaps or topology evolution",
        "timing_scope": "public function including CPU reductions, uploads and downloads; final stream synchronization included",
        "excluded_from_warm_timing": ["input loading", "mesh construction", "deepcopy", "context entry", "correctness comparison"],
        "cold_scope": "first call per backend context; process, driver and on-disk kernel caches may already be warm",
        "sample_policy": "repeat identical checkpoint input; contiguous backend batches retain each context's buffers",
        "backend_order": order,
        "input": {
            "config": str(args.config.resolve()),
            "config_sha256": file_sha256(args.config),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_hashes": {name: file_sha256(args.checkpoint / name) for name in ("meta.json", "state.npz")},
            "checkpoint_time_myr": float(checkpoint.state.time_myr),
            "cells": mesh.cell_count,
            "radius_km": template["radius_km"],
            "dt_myr": args.dt,
            "sea_level_m": sea_level,
            "sea_level_source": sea_level_source,
            "erosivity": "uniform default",
            "rift_recycled_volume_km3": 0.0,
            "sediment_parameters": asdict(params),
            "warm_repeats": args.repeat,
        },
        "input_setup_seconds": input_setup_seconds,
        "environment": {"python": sys.version, "python_executable": sys.executable, "numpy": np.__version__},
        "measurements": {},
    }
    reference = None
    for name in order:
        measurement, reference = measure_backend(name, mesh, template, args.repeat, args.gpu_device, reference)
        report["measurements"][name] = measurement
    cpu_median = report["measurements"]["cpu"]["median_warm_seconds"]
    for name, measurement in report["measurements"].items():
        measurement["isolated_surface_speedup_vs_cpu"] = cpu_median / measurement["median_warm_seconds"]
    report["all_outputs_byte_exact"] = all(
        measurement["all_outputs_byte_exact"] for measurement in report["measurements"].values()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents an overwrite if another process creates
    # the chosen file while this benchmark is running.
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Report: {output}", flush=True)
    return 0 if report["all_outputs_byte_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
