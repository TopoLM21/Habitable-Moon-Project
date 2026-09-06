"""Isolated exact tectonic-forcing halo study, not a production backend.

The original CPU function still selects subducting sides and computes all
trench transcendental functions. With halo=0 it produces seed maxima and tags.
A separate CPU or CUDA self-plus-neighbor maximum spreads only those seeds.
This deliberately measures CPU preparation AND round-trip transfers; it does
not mistake a tiny CUDA stencil time for whole-topography acceleration.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
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
from tectonics.lithosphere import boundary_records_for_state
from tectonics.simulation import build_initial_mesh, load_config
from tectonics.topography import TopographyParameters, tectonic_forcing
from tectonics.topology import PlateTopologyManager, PlateTopologyParameters
from tectonics.volcanic_arc import VolcanicArcParameters, compute_volcanic_arc_forcing


COMPONENTS = ("ridge", "trench_depth", "arc", "collision")
CUDA_SOURCE = r'''
extern "C" __global__ void halo_maximum(
    const double* seeds, const int* neighbors, double* output,
    int n, double halo, int external_arc) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double v[4];
    for (int kind = 0; kind < 4; ++kind) {
        double best = seeds[kind*n+i];
        if (halo > 0.0 && !(kind == 2 && external_arc)) {
            for (int k = 0; k < 3; ++k) {
                double candidate = seeds[kind*n+neighbors[3*i+k]] * halo;
                best = candidate > best ? candidate : best;
            }
        }
        v[kind] = best;
        output[kind*n+i] = best;
    }
    output[4*n+i] = ((v[0] + v[2]) + v[3]) - v[1];
}
'''


def dataclass_from_config(kind, values):
    return kind(**{field.name: values[field.name] for field in fields(kind) if field.name in values})


def mesh_neighbors(mesh):
    """The gather cut requires the reciprocal triangular mesh used by the model."""
    neighbors = np.ascontiguousarray(mesh.neighbors, dtype=np.int32)
    n = mesh.cell_count
    if neighbors.shape != (n, 3) or np.any(neighbors < 0) or np.any(neighbors >= n):
        raise ValueError("probe requires three valid neighbors per mesh cell")
    for slot in range(3):
        if not np.all(np.any(neighbors[neighbors[:, slot]] == np.arange(n)[:, None], axis=1)):
            raise ValueError("probe requires reciprocal mesh neighbors")
    return neighbors


def prepare_seeds(mesh, state, boundaries, params, radius_km, arc_uplift_forcing):
    _, tags, components = tectonic_forcing(
        mesh, state, boundaries, replace(params, boundary_one_ring_fraction=0.0),
        radius_km, arc_uplift_forcing,
    )
    seeds = np.stack([components[key] for key in COMPONENTS])
    if not np.all(np.isfinite(seeds)):
        raise ValueError("this isolated probe supports finite seed fields only")
    return seeds, tags


def as_result(packed, tags):
    return packed[4], tags, {key: packed[i] for i, key in enumerate(COMPONENTS)}


class CpuHalo:
    def __init__(self, mesh):
        self.neighbors = mesh_neighbors(mesh)

    def calculate(self, seeds, halo, external_arc):
        packed = np.empty((5, seeds.shape[1]), dtype=np.float64)
        packed[:4] = seeds
        if halo > 0.0:
            for kind in range(4):
                if kind == 2 and external_arc:
                    continue
                for slot in range(3):
                    np.maximum(packed[kind], seeds[kind, self.neighbors[:, slot]] * halo,
                               out=packed[kind])
        packed[4] = packed[0] + packed[2] + packed[3] - packed[1]
        return packed


class GpuHalo:
    def __init__(self, mesh, device):
        import cupy as cp
        self.cp = cp
        self.device = cp.cuda.Device(device)
        self.device.use()
        neighbors = mesh_neighbors(mesh)
        self.n = mesh.cell_count
        self.neighbors = cp.asarray(neighbors)
        self.seeds = cp.empty((4, self.n), dtype=cp.float64)
        self.output = cp.empty((5, self.n), dtype=cp.float64)
        self.module = cp.RawModule(code=CUDA_SOURCE, options=("--std=c++17", "--fmad=false"))
        self.kernel = self.module.get_function("halo_maximum")
        self.synchronize()

    def synchronize(self):
        self.cp.cuda.get_current_stream().synchronize()

    def calculate(self, seeds, halo, external_arc):
        self.device.use()
        self.seeds.set(seeds)
        self.kernel(((self.n + 255) // 256,), (256,), (
            self.seeds, self.neighbors, self.output, np.int32(self.n),
            np.float64(halo), np.int32(external_arc),
        ))
        return self.output.get()


def compare_results(reference, candidate):
    differences = []
    arrays = [("forcing", reference[0], candidate[0])]
    arrays.extend((key, reference[2][key], candidate[2][key]) for key in COMPONENTS)
    for key, left, right in arrays:
        if left.shape != right.shape or left.dtype != right.dtype or left.tobytes() != right.tobytes():
            differences.append({"array": key, "different_cells": int(np.count_nonzero(left != right)),
                                "max_absolute_difference_m": float(np.max(np.abs(left - right), initial=0.0))})
    tags_equal = reference[1] == candidate[1]
    return {"byte_exact": not differences and tags_equal, "arrays_checked": len(arrays),
            "tags_equal": tags_equal, "differences": differences}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def measure(function, repeat, reference):
    times, comparisons = [], []
    for sample in range(repeat + 1):
        started = perf_counter()
        result = function()
        elapsed = perf_counter() - started
        comparisons.append(compare_results(reference, result))
        if sample:
            times.append(elapsed)
        else:
            cold = elapsed
    return {"cold_call_seconds": cold, "warm_seconds": times,
            "median_seconds": statistics.median(times),
            "all_outputs_byte_exact": all(item["byte_exact"] for item in comparisons),
            "comparisons": comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "canonical_moon.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--internal-arcs", action="store_true",
                        help="Use local boundary arcs instead of the model's separately computed arc field")
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.gpu_device < 0:
        parser.error("--repeat must be positive and --gpu-device non-negative")
    output = args.output.resolve()
    if output.exists():
        parser.error("--output already exists")
    inputs = [args.config.resolve(), args.checkpoint.resolve() / "meta.json",
              args.checkpoint.resolve() / "state.npz"]
    hashes_before = {str(path): file_sha256(path) for path in inputs}
    setup_started = perf_counter()
    config = load_config(args.config)
    with CpuExecution(1, arc_kernels=True, cell_kernels=True):
        mesh = build_initial_mesh(config)
        checkpoint = load_checkpoint(args.checkpoint, PlateTopologyManager(
            dataclass_from_config(PlateTopologyParameters, config.get("plate_topology", {}))))
        if len(checkpoint.state.cell_plate) != mesh.cell_count:
            parser.error("checkpoint and config cell counts differ")
        radius = float(config["moon"]["radius_km"])
        params = dataclass_from_config(TopographyParameters, config.get("topography_evolution", {}))
        boundaries = boundary_records_for_state(
            mesh, checkpoint.state, checkpoint.system, radius,
            float(config["classification"]["normal_threshold_km_per_myr"]),
            float(config["classification"]["inactive_speed_km_per_myr"]),
        )
        external = None
        if not args.internal_arcs:
            external, _ = compute_volcanic_arc_forcing(
                mesh, checkpoint.state, checkpoint.subduction_memory, radius,
                dataclass_from_config(VolcanicArcParameters, config.get("volcanic_arc", {})), boundaries,
            )
        call_args = (mesh, checkpoint.state, boundaries, params, radius, external)
        reference = tectonic_forcing(*call_args)
        cpu_halo = CpuHalo(mesh)
        setup_seconds = perf_counter() - setup_started
        seeds, tags = prepare_seeds(*call_args)
        records = {"cpu_original": measure(lambda: tectonic_forcing(*call_args), args.repeat, reference)}

        def hybrid(backend):
            current_seeds, current_tags = prepare_seeds(*call_args)
            return as_result(backend.calculate(current_seeds, params.boundary_one_ring_fraction,
                                                external is not None), current_tags)

        records["cpu_seed_and_batched_halo"] = measure(lambda: hybrid(cpu_halo), args.repeat, reference)
        records["cpu_halo_only"] = measure(lambda: as_result(cpu_halo.calculate(
            seeds, params.boundary_one_ring_fraction, external is not None), tags), args.repeat, reference)
        prep_times = []
        for _ in range(args.repeat):
            started = perf_counter()
            prepare_seeds(*call_args)
            prep_times.append(perf_counter() - started)
        gpu_setup_seconds = None
        if not args.cpu_only:
            started = perf_counter()
            gpu_halo = GpuHalo(mesh, args.gpu_device)
            gpu_setup_seconds = perf_counter() - started
            records["cpu_seed_and_gpu_halo"] = measure(lambda: hybrid(gpu_halo), args.repeat, reference)
            records["gpu_halo_roundtrip_only"] = measure(lambda: as_result(gpu_halo.calculate(
                seeds, params.boundary_one_ring_fraction, external is not None), tags), args.repeat, reference)
    hashes_after = {str(path): file_sha256(path) for path in inputs}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Isolated tectonic_forcing on one checkpoint, not advance_topography or full timestep",
        "production_backend_modified": False,
        "transcendental_math": "Unchanged original CPU scalar trench function",
        "input": {"config": str(args.config.resolve()), "checkpoint": str(args.checkpoint.resolve()),
                  "cells": mesh.cell_count, "boundaries": len(boundaries),
                  "repeat": args.repeat, "external_arcs": external is not None,
                  "halo_fraction": params.boundary_one_ring_fraction},
        "setup_excluded_seconds": setup_seconds,
        "gpu_setup_excluded_seconds": gpu_setup_seconds,
        "cpu_seed_preparation_seconds": prep_times,
        "cpu_seed_preparation_median_seconds": statistics.median(prep_times),
        "measurements": records,
        "gpu_stencil_transfers_per_call": {"upload_bytes": 4 * mesh.cell_count * 8,
                                            "download_bytes": 5 * mesh.cell_count * 8},
        "hashes_before": hashes_before, "hashes_after": hashes_after,
        "inputs_unchanged": hashes_before == hashes_after,
        "all_outputs_byte_exact": all(item["all_outputs_byte_exact"] for item in records.values()),
        "timing_limitations": "Warm fixed-state repeats; loading, mesh setup, arc field calculation and comparisons excluded. GPU halo includes host upload and download. No whole-model speedup inferred.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2))
    for name, item in records.items():
        print(f"{name}: {item['median_seconds'] * 1000:.3f} ms; byte_exact={item['all_outputs_byte_exact']}")
    print(f"report: {output}")
    return 0 if report["inputs_unchanged"] and report["all_outputs_byte_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
