"""Compare short coupled continuations without mistaking cut meshes for plates."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tectonics.genesis_coupled import load_coupled_checkpoint
from tectonics.genesis_seam_diagnostics import cohort_bonded_traces
from tectonics.mesh import connected_components


def inspect_run(path):
    path = Path(path).resolve()
    checkpoint = path / "coupled_checkpoint.npz"
    model, state, thermal, orbit = load_coupled_checkpoint(checkpoint)
    report = model.diagnostics(state, thermal, orbit)
    with (path / "coupled_history.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    if float(history[-1]["time_myr"]) != state.time_myr:
        raise ValueError("History and checkpoint end at different times")
    geometry = model._geometry(state.cut_edges)
    topology = geometry.topology
    bonded = cohort_bonded_traces(state.cohorts, 2 * len(state.cut_edges)).reshape(-1, 2).any(axis=1)
    neighbors = [set(items) for items in topology.mesh.neighbors]
    # Actual split IDs preserve crack-tip kinematic links. Coincident original
    # positions alone must never join the two independently moving banks.
    owners = [[] for _ in range(topology.mesh.vertex_count)]
    for face, vertices in enumerate(topology.mesh.faces):
        for vertex in vertices:
            owners[vertex].append(face)
    for faces in owners:
        for a, b in zip(faces, faces[1:]):
            neighbors[a].add(b)
            neighbors[b].add(a)
    for a, b in topology.seam_faces[bonded]:
        neighbors[a].add(b)
        neighbors[b].add(a)
    regions = sorted(connected_components(range(topology.mesh.cell_count), neighbors), key=len, reverse=True)
    if len(regions) != report["cohesive_component_count"]:
        raise ValueError("Independent face-graph count disagrees with model diagnostics")
    source = model.source_model
    parameters = {name: asdict(value) for name, value in (
        ("thermal", source.thermal), ("shell", source.p), ("onset", source.onset_p),
        ("tides", source.tides_p), ("mobile", source.mobile_p), ("faults", source.fault_p),
        ("coupled", model.parameters), ("contact", model.contact_parameters), ("law", model.law_parameters))}
    areas = topology.mesh.areas_unit_sphere
    return {"path": str(path), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "source_sha256": model.source_hash, "cells": topology.mesh.cell_count,
            "parameters": parameters, "final": report, "history": history,
            "cohesive_component_sizes": list(map(len, regions)),
            "cohesive_reference_area_fractions": [float(areas[region].sum()/areas.sum()) for region in regions]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    target = args.output
    if target.exists() and any(target.iterdir()):
        raise ValueError("Audit output must be new or empty")
    runs = sorted([inspect_run(path) for path in args.run], key=lambda item: item["cells"])
    reference = json.loads(json.dumps(runs[0]["parameters"]))
    reference["shell"].pop("subdivisions")
    for run in runs:
        if run["final"]["source_time_myr"] != runs[0]["final"]["source_time_myr"]:
            raise ValueError("Comparison requires matching source ages")
        candidate = json.loads(json.dumps(run["parameters"]))
        candidate["shell"].pop("subdivisions")
        if candidate != reference:
            raise ValueError("Physical parameters differ beyond spatial resolution")
        times = [float(row["elapsed_years"]) for row in run["history"]]
        reference_times = [float(row["elapsed_years"]) for row in runs[0]["history"]]
        if len(times) != len(reference_times) or not np.allclose(times, reference_times, rtol=0, atol=1e-7):
            raise ValueError("Comparison requires matching observation times")
    target.mkdir(parents=True, exist_ok=True)
    (target / "comparison.json").write_text(json.dumps({
        "format": "seam-continuation-audit-0.1", "runs": runs,
        "scope": "Legacy mesh-dependent extraction; short response only, not plate formation or convergence certification."
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    try:
        for run in runs:
            t = [float(row["elapsed_years"]) for row in run["history"]]
            for ax, key, title in zip(axes,
                    ("max_opening_m", "max_abs_jump_m", "cohesive_component_count"),
                    ("Максимальное раскрытие · м", "Максимальный сдвиг · м", "Области с учётом сцепления")):
                ax.plot(t, [float(row[key]) for row in run["history"]], label=f"{run['cells']} ячеек")
                ax.set_title(title)
                ax.set_xlabel("Физических лет после исходного снимка")
                ax.grid(alpha=.2)
        axes[0].legend()
        axes[2].set_yticks(sorted(set(float(row["cohesive_component_count"]) for run in runs for row in run["history"])))
        traction = runs[0]["parameters"]["shell"]["convective_traction_pa"] / 1000.
        age = runs[0]["final"]["source_time_myr"]
        fig.suptitle(f"Одинаковые параметры · {traction:g} кПа · продолжение с {age:g} млн лет")
        fig.text(.5, .03, "Границы зависят от сетки; этот опыт не подтверждает появление устойчивых плит.", ha="center")
        fig.tight_layout(rect=(0, .08, 1, .93))
        fig.savefig(target / "comparison.png", dpi=150)
    finally:
        plt.close(fig)
    for run in runs:
        print(f"{run['cells']} cells: cohesive sizes {run['cohesive_component_sizes']}; "
              f"opening {run['final']['max_opening_m']:.6g} m")


if __name__ == "__main__":
    main()
