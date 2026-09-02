"""Isolated candidate kernels, NOT installed in the production integrator.

Compare unchanged reference functions with bounded NumPy alternatives on copied
checkpoint fields. Whole-model speedups cannot be inferred from these timings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def root_contrast_batch(mesh, state):
    """Candidate for the fixed three-neighbour triangulation; no shared writes."""
    if state.mantle_lithosphere_thickness_km is None:
        return np.zeros(mesh.cell_count, dtype=np.float64)
    from tectonics.plume_rifting import _neighbor_root_contrast
    if any(len(neighbors) != 3 for neighbors in mesh.neighbors):
        return _neighbor_root_contrast(mesh, state)
    # Include index construction in this probe, i.e. do not claim free caching.
    neighbors = np.asarray(mesh.neighbors, dtype=np.intp)
    roots = np.asarray(state.mantle_lithosphere_thickness_km, dtype=np.float64)
    return np.abs(roots - np.mean(roots[neighbors], axis=1))


def local_proxy_scalar(state, face):
    """Candidate preserving the original two maxima and float64 multiplication."""
    if state.mantle_lithosphere_thickness_km is not None and state.mantle_lithosphere_density_anomaly_kg_m3 is not None:
        h = np.maximum(np.float64(state.mantle_lithosphere_thickness_km[face]), 0.0)
        drho = np.maximum(np.float64(state.mantle_lithosphere_density_anomaly_kg_m3[face]), 0.0)
        return max(float(h * drho), 0.0)
    return max(float(state.crust_age_myr[face]), 0.0)


def paired_measure(reference, candidate, repeat=3):
    expected, actual = reference(), candidate()
    if expected.dtype != actual.dtype or expected.shape != actual.shape or expected.tobytes() != actual.tobytes():
        raise RuntimeError("Candidate differs from reference")
    samples = {"reference": [], "candidate": []}
    for iteration in range(repeat):
        cases = [("reference", reference), ("candidate", candidate)]
        if iteration % 2:
            cases.reverse()
        for name, function in cases:
            start = perf_counter()
            result = function()
            samples[name].append(perf_counter() - start)
            if result.tobytes() != expected.tobytes():
                raise RuntimeError("Repeated candidate differs")
    return {"exact_bytes": True, "values": int(expected.size), "seconds": samples,
            "median_seconds": {name: statistics.median(values) for name, values in samples.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "results/cpu_performance"):
        parser.error("Output must be inside results/cpu_performance")
    output.mkdir(parents=True, exist_ok=False)
    from execution_policy import apply_process_priority
    apply_process_priority("below_normal")
    from tectonics.checkpoint import load_checkpoint
    from tectonics.cpu_runtime import CpuExecution
    from tectonics.plume_rifting import _neighbor_root_contrast
    from tectonics.simulation import build_initial_mesh, load_config
    from tectonics.subduction_memory import _local_proxy
    from tectonics.topology import PlateTopologyManager, PlateTopologyParameters
    source = ROOT / "results/gui_runs/stable_copy_0_700"
    config_path = source / "gui_runtime_config.yaml"
    paths = [config_path, *[source / f"gui_checkpoint_{age:06d}_Myr" / name
                           for age in (320, 620, 700) for name in ("meta.json", "state.npz")]]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    rows = []
    with CpuExecution(numeric_kernels=False):
        mesh = build_initial_mesh(load_config(config_path))
        for age in (320, 620, 700):
            checkpoint = load_checkpoint(source / f"gui_checkpoint_{age:06d}_Myr",
                                         PlateTopologyManager(PlateTopologyParameters()))
            state = checkpoint.state
            # Reference calls intentionally recompute the full array per face.
            # Sample 4096 deterministic faces, not a simulated full time step.
            faces = np.linspace(0, mesh.cell_count - 1, 4096, dtype=np.intp)
            before = {name: value.copy() for name, value in vars_from_state(state).items()}
            contrast = paired_measure(lambda: _neighbor_root_contrast(mesh, state),
                                      lambda: root_contrast_batch(mesh, state))
            proxy = paired_measure(lambda: np.asarray([_local_proxy(state, int(face)) for face in faces]),
                                   lambda: np.asarray([local_proxy_scalar(state, int(face)) for face in faces]))
            unchanged = all(value.tobytes() == vars_from_state(state)[name].tobytes() for name, value in before.items())
            if not unchanged:
                raise RuntimeError("Kernel mutated checkpoint fields")
            row = {"kind": "real_checkpoint", "age_myr": age, "cells": mesh.cell_count,
                   "root_contrast": contrast, "local_proxy_4096_faces": proxy, "inputs_unchanged": unchanged}
            rows.append(row)
            print(json.dumps(row), flush=True)
    for cells in (81920, 327680):
        # Synthetic scaling isolates kernels; this is NOT a high-resolution run.
        rng = np.random.default_rng(81931 + cells)
        state = SimpleNamespace(mantle_lithosphere_thickness_km=rng.uniform(-5, 220, cells),
                                mantle_lithosphere_density_anomaly_kg_m3=rng.uniform(-30, 100, cells),
                                crust_age_myr=rng.uniform(0, 800, cells))
        neighbors = tuple(((i - 1) % cells, (i + 1) % cells, (i + 2) % cells) for i in range(cells))
        mesh = SimpleNamespace(cell_count=cells, neighbors=neighbors)
        faces = np.linspace(0, cells - 1, 4096, dtype=np.intp)
        row = {"kind": "synthetic_three_neighbor_scaling", "cells": cells,
               "root_contrast": paired_measure(lambda: _neighbor_root_contrast(mesh, state),
                                                lambda: root_contrast_batch(mesh, state)),
               "local_proxy_4096_faces": paired_measure(
                   lambda: np.asarray([_local_proxy(state, int(face)) for face in faces]),
                   lambda: np.asarray([local_proxy_scalar(state, int(face)) for face in faces]))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    unchanged = hashes == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if not unchanged:
        raise RuntimeError("Input checkpoint files changed")
    (output / "candidates.json").write_text(json.dumps({"production_enabled": False,
        "rows": rows, "source_files_unchanged": unchanged, "source_sha256": hashes}, indent=2), encoding="utf-8")


def vars_from_state(state):
    # Do not use dataclasses.asdict: it would hide mutation behind deep copies.
    return {name: getattr(state, name) for name in state.__dataclass_fields__
            if isinstance(getattr(state, name), np.ndarray)}


if __name__ == "__main__":
    main()
