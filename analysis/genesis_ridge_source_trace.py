"""Try the experimental ridge tracker at one explicit, fixed real-source seed.

The seed is a diagnostic shear maximum selected in a previous read-only audit.
It is not a physical nucleation event. Defaults stay fixed across all fields
and resolutions; a rejected or short path is a result, not a reason to tune.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tectonics.genesis_faults import load_fault_checkpoint
from tectonics.genesis_material import face_frames
from tectonics.genesis_ridge import RidgeField, RidgeParameters, RidgeUnavailable


def source_data(checkpoint):
    model, state, _, _, _ = load_fault_checkpoint(checkpoint)
    mesh = model.mesh_for(state)
    normals = np.einsum("fij,fj->fi", face_frames(mesh), state.plane_normal)
    normals -= np.einsum("fi,fi->f", normals, mesh.centroids)[:, None] * mesh.centroids
    length = np.linalg.norm(normals, axis=1)
    normals = np.divide(normals, length[:, None], out=np.zeros_like(normals),
                        where=state.fault_active[:, None])
    return model, state, mesh, normals


def run_case(checkpoint, fixed_seed, parameters, output):
    _, state, mesh, normals = source_data(checkpoint)
    tree = cKDTree(mesh.centroids)
    _, nearest = tree.query(fixed_seed)
    seed_offset_km = state.radius_km * np.arctan2(
        np.linalg.norm(np.cross(mesh.centroids[nearest], fixed_seed)),
        mesh.centroids[nearest] @ fixed_seed)
    report = {"checkpoint": str(checkpoint.relative_to(ROOT)),
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "cells": mesh.cell_count, "radius_km": state.radius_km,
              "nearest_seed_cell": int(nearest),
              "nearest_seed_cell_center_xyz": mesh.centroids[nearest].tolist(),
              "fixed_seed_to_nearest_center_km": float(seed_offset_km),
              "active_mask": "source fault_active, without an additional damage threshold",
              "results": []}
    traces = {}
    for name, values in (("damage", state.damage),
                         ("cumulative_shear_divided_by_0p1", state.cumulative_shear/.1)):
        # Deliberately no clipping or per-run normalization: [0,1] validation
        # must reject a source outside the stated fixed analysis scale.
        field = RidgeField(mesh, values, normals, state.radius_km,
                           active=state.fault_active, parameters=parameters)
        result = {"field": name,
                  "scalar_scale": 1. if name == "damage" else .1,
                  "scalar_max": float(values.max())}
        try:
            path = field.trace(fixed_seed)
        except RidgeUnavailable as exc:
            result.update(status="seed_rejected", reason=exc.reason,
                          points=0, total_length_km=0.)
        else:
            nearest_path = tree.query(path.points_xyz)[1]
            array_path = output / f"trace_{mesh.cell_count}_{name}.npz"
            np.savez_compressed(array_path, points_xyz=path.points_xyz,
                                arclength_km=path.arclength_km, values=path.values,
                                transverse_contrast=path.transverse_contrast,
                                nearest_source_cell=nearest_path)
            result.update(status="traced", points=len(path.points_xyz),
                          total_length_km=float(path.arclength_km[-1]),
                          left_stop=path.left_stop, right_stop=path.right_stop,
                          closed=path.closed, seed_index=path.seed_index,
                          points_xyz=path.points_xyz.tolist(),
                          arclength_km=path.arclength_km.tolist(),
                          values=path.values.tolist(),
                          transverse_contrast=path.transverse_contrast.tolist(),
                          nearest_source_cell=nearest_path.tolist(),
                          path_archive=array_path.name)
            traces[name] = path
        report["results"].append(result)
    return report, (mesh, state, normals, traces)


def draw_case(report, source, fixed_seed, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection, LineCollection

    mesh, state, normals, traces = source
    east = np.cross([0., 0., 1.], fixed_seed)
    east /= np.linalg.norm(east)
    north = np.cross(fixed_seed, east)
    basis = np.column_stack((east, north))
    # Explicit local orthographic display, not the tracker reconstruction.
    xy = mesh.vertices @ basis * state.radius_km
    centers_xy = mesh.centroids @ basis * state.radius_km
    visible = mesh.centroids @ fixed_seed > np.cos(3000/state.radius_km)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    reasons = {
        "underresolved_cells": "Недостаточное разрешение ячеек",
        "no_transverse_maximum": "Нет поперечного максимума",
        "unresolved_transverse_contrast": "Контраст недостаточно разрешён",
        "not_an_elongated_ridge": "Нет вытянутой полосы",
        "ambiguous_orientation": "Неоднозначное направление",
        "projection_out_of_range": "Коррекция выходит за допустимый радиус",
        "poor_scalar_fit": "Недостаточная точность приближения поля",
        "length_limit": "Достигнута заданная длина поиска",
    }
    for ax, name, values, label in zip(
            axes, ("damage", "cumulative_shear_divided_by_0p1"),
            (state.damage, state.cumulative_shear/.1),
            ("Повреждение D", "Накопленный сдвиг / фиксированный масштаб 0.1")):
        collection = PolyCollection(xy[mesh.faces[visible]], array=values[visible],
                                    clim=(0, 1), cmap="viridis", edgecolor=".5", linewidth=.2)
        ax.add_collection(collection)
        tangent = np.cross(mesh.centroids, normals) @ basis
        half_length = 65.
        segments = np.stack((centers_xy-half_length*tangent,
                             centers_xy+half_length*tangent), axis=1)
        ax.add_collection(LineCollection(segments[visible & state.fault_active],
                                          colors="white", alpha=.65, linewidth=.8))
        ax.scatter([0], [0], s=65, marker="x", color="red", zorder=4)
        result = next(item for item in report["results"] if item["field"] == name)
        if name in traces:
            path = traces[name]
            path_xy = path.points_xyz @ basis * state.radius_km
            ax.plot(path_xy[:, 0], path_xy[:, 1], color="red", lw=2., zorder=3)
            text = (f"{len(path_xy)} точек; {path.arclength_km[-1]:.1f} км\n"
                    f"Остановка слева: {reasons.get(path.left_stop, path.left_stop)}\n"
                    f"Остановка справа: {reasons.get(path.right_stop, path.right_stop)}")
        else:
            text = "Начальная точка отклонена:\n" + reasons.get(result["reason"], result["reason"])
        ax.set(xlim=(-2800, 2800), ylim=(-2800, 2800), aspect="equal",
               xlabel="Локальное направление на восток, км",
               ylabel="Локальное направление на север, км", title=label)
        ax.text(.02, .02, text, transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="white", alpha=.9, edgecolor=".7"))
        fig.colorbar(collection, ax=ax, shrink=.75)
    fig.suptitle(f"{mesh.cell_count} ячеек: заданная начальная точка и фиксированные параметры поиска\n"
                 "Геометрическая проба на сохранённом поле; физическое зарождение трещины не рассчитывается")
    fig.savefig(output / f"source_trace_{mesh.cell_count}.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/genesis_runs/ridge_tracking_20260924/source_trace")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "source_trace.json"
    if destination.exists():
        raise FileExistsError(destination)
    base = ROOT / "results/genesis_runs"
    paths = [base / case / "fault_checkpoint.npz" for case in (
        "fault_damage_final_20260922", "fine_fault_audit_20260924/strong_1280",
        "fine_fault_audit_20260924/strong_5120")]
    _, _, fine_mesh, _ = source_data(paths[-1])
    fixed_seed = fine_mesh.centroids[5096].copy()
    parameters = RidgeParameters()
    reports = []
    for checkpoint in paths:
        report, source = run_case(checkpoint, fixed_seed, parameters, args.output)
        reports.append(report)
        draw_case(report, source, fixed_seed, args.output)
    payload = {"purpose": "Explicit static-field tracking trial, not physical fracture nucleation or propagation.",
               "seed_origin": "Full-precision centroid of cell 5096 in the saved 5120-cell source, chosen in the preceding scalar-contrast audit.",
               "seed_xyz": fixed_seed.tolist(),
               "matched_seed_policy": "Identical spherical unit vector on every mesh; nearest source cell IDs and physical offsets recorded, no snapping of the actual query.",
               "ridge_parameters": asdict(parameters),
               "ridge_implementation_sha256": hashlib.sha256((ROOT / "tectonics/genesis_ridge.py").read_bytes()).hexdigest(),
               "physical_time": "Source age remains 1.4 Myr. Path arclength is not a time increment.",
               "scaling": "Damage unchanged; accumulated shear divided by an explicit fixed 0.1. No per-run maximum normalization or clipping.",
               "limits": "No seed stress/energy criterion, front history, propagation law, branching, contact insertion, material transport or source-state changes.",
               "cases": reports}
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination)
    for report in reports:
        print(report["cells"], report["nearest_seed_cell"],
              [(item["field"], item["status"], item.get("reason"), item["total_length_km"])
               for item in report["results"]])


if __name__ == "__main__":
    main()
