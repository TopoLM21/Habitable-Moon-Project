"""Read-only lateral-contrast audit of saved diffuse weak-plane fields.

Neither the reported local maxima nor their count are physical crack seeds.
IDW3 and nearest-cell sampling are diagnostic reconstructions, not a new
constitutive length scale or an energy-consistent localization model.
"""
from __future__ import annotations

import argparse
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


def quantiles(values):
    values = np.asarray(values, float)
    if not len(values):
        return None
    return dict(zip(("min", "p05", "p25", "median", "p75", "p95", "max"),
                    np.quantile(values, [0, .05, .25, .5, .75, .95, 1]).tolist()))


def sample(points, tree, values, method):
    """Spherical angular inverse-distance weighting, exact at cell centres."""
    count = 1 if method == "nearest" else 3
    distance, indices = tree.query(points, k=count)
    if count == 1:
        return values[indices]
    distance = 2 * np.arcsin(np.clip(distance / 2, 0, 1))
    weights = 1 / np.maximum(distance, 1e-14) ** 2
    weights /= weights.sum(axis=1)[:, None]
    return (values[indices] * weights).sum(axis=1)


def lateral_points(points, normals, distance_km, radius_km):
    angle = np.asarray(distance_km) / radius_km
    return (np.cos(angle)[..., None] * points
            + np.sin(angle)[..., None] * normals)


def transport_normals(normals, origins, targets):
    """Parallel transport along the unique shorter great-circle segment."""
    cross = np.cross(origins, targets)
    cosine = np.einsum("...i,...i->...", origins, targets)
    return (normals + np.cross(cross, normals)
            + np.cross(cross, np.cross(cross, normals)) / (1 + cosine)[..., None])


def inspect(path):
    model, state, _, _, _ = load_fault_checkpoint(path)
    mesh = model.mesh_for(state)
    centers = mesh.centroids
    tree = cKDTree(centers)
    eligible = state.fault_active & (state.damage >= model.fault_p.activation_damage)
    normal = np.einsum("fij,fj->fi", face_frames(mesh), state.plane_normal)
    normal -= np.einsum("fi,fi->f", normal, centers)[:, None] * centers
    lengths = np.linalg.norm(normal, axis=1)
    normal = np.divide(normal, lengths[:, None], out=np.zeros_like(normal), where=lengths[:, None] > 0)
    volume = state.layer_mass_kg.sum(axis=1) / model.p.density_kg_m3
    fields = {"damage": state.damage, "cumulative_shear": state.cumulative_shear,
              "cumulative_dissipation_j_per_reference_m3":
              (state.friction_work_cell_j + state.viscous_work_cell_j) / volume}
    edges = np.asarray(mesh.shared_edges, int)
    spacing = state.radius_km * np.arccos(np.clip(np.einsum("fi,fi->f", centers[edges[:, 0]], centers[edges[:, 1]]), -1, 1))
    report = {"checkpoint": str(path.relative_to(ROOT)),
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "cells": mesh.cell_count, "radius_km": state.radius_km,
              "eligible_cells": int(eligible.sum()),
              "neighbor_center_spacing_km": quantiles(spacing),
              "fields_on_eligible": {name: quantiles(values[eligible]) for name, values in fields.items()},
              "lateral_contrasts": []}
    both = eligible[edges[:, 0]] & eligible[edges[:, 1]]
    a, b = edges[both, 0], edges[both, 1]
    carried = transport_normals(normal[a], centers[a], centers[b])
    angles = np.degrees(np.arccos(np.clip(np.abs(np.einsum("fi,fi->f", carried, normal[b])), 0, 1)))
    report["eligible_neighbor_unoriented_plane_difference_degrees"] = quantiles(angles)
    data = {}
    for distance in (400., 800., 1600.):
        sides = [lateral_points(centers, normal, sign * distance, state.radius_km) for sign in (-1, 1)]
        for method in ("nearest", "idw3"):
            for name, values in fields.items():
                left, right = [sample(point, tree, values, method) for point in sides]
                contrast = values - np.maximum(left, right)
                relative = contrast / np.maximum(values, 1e-30)
                ridge = eligible & (contrast > 0)
                nonzero = eligible & (values > 0)
                result = {"distance_km": distance, "sampling": method, "field": name,
                          "absolute_contrast": quantiles(contrast[eligible]),
                          "relative_contrast_nonzero": quantiles(relative[nonzero]),
                          "positive_both_sides_cells": int(ridge.sum()),
                          "positive_both_sides_area_fraction": float(mesh.areas_unit_sphere[ridge].sum() / (4*np.pi)),
                          "over_one_percent_relative_cells": int(np.count_nonzero(eligible & (relative > .01))),
                          "over_five_percent_relative_cells": int(np.count_nonzero(eligible & (relative > .05)))}
                report["lateral_contrasts"].append(result)
                data[distance, method, name] = relative
    report["sampling_and_offset_consistency"] = []
    for name in fields:
        for lower in (0., .01, .05):
            robust = eligible.copy()
            for distance in (400., 800.):
                for method in ("nearest", "idw3"):
                    robust &= data[distance, method, name] > lower
            report["sampling_and_offset_consistency"].append(
                {"field": name, "relative_contrast_lower_bound": lower,
                 "cells_positive_in_both_methods_and_offsets": int(robust.sum())})
    # Candidate ranking is diagnostic and contains no nucleation threshold.
    # Pick strongest 800-km two-sided shear contrast, with only an explicit
    # physical separation to avoid reporting neighbouring cells as new seeds.
    shear_contrast = data[800., "idw3", "cumulative_shear"] * state.cumulative_shear
    ordered = np.argsort(-np.where(eligible, shear_contrast, -np.inf))
    selected = []
    for index in ordered:
        if shear_contrast[index] <= 0:
            break
        if selected:
            separation = state.radius_km * np.arccos(np.clip(centers[selected] @ centers[index], -1, 1))
            if separation.min() < 1600.:
                continue
        selected.append(int(index))
        if len(selected) >= 5:
            break
    report["diagnostic_candidates"] = [
        {"cell": index, "point_xyz": centers[index].tolist(), "normal_xyz": normal[index].tolist(),
         "latitude_degrees": float(np.degrees(np.arcsin(centers[index, 2]))),
         "longitude_degrees": float(np.degrees(np.arctan2(centers[index, 1], centers[index, 0]))),
         "damage": float(state.damage[index]), "cumulative_shear": float(state.cumulative_shear[index]),
         "relative_shear_contrast_400_km": float(data[400., "idw3", "cumulative_shear"][index]),
         "relative_shear_contrast_800_km": float(data[800., "idw3", "cumulative_shear"][index]),
         "relative_damage_contrast_800_km": float(data[800., "idw3", "damage"][index])}
        for index in selected]
    for entry, index in zip(report["diagnostic_candidates"], selected):
        near = np.asarray(tree.query_ball_point(centers[index], 2*np.sin(800/state.radius_km/2)), int)
        near = near[eligible[near]]
        carried = transport_normals(normal[near], centers[near], centers[index])
        angles = np.degrees(np.arccos(np.clip(np.abs(carried @ normal[index]), 0, 1)))
        entry["eligible_cells_within_800_km"] = len(near)
        entry["plane_difference_within_800_km_degrees"] = quantiles(angles)
    return report, (mesh, state, normal, tree, fields, data, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/genesis_runs/ridge_tracking_20260924/source_audit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "source_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    cases = ["fault_damage_final_20260922", "fine_fault_audit_20260924/strong_1280",
             "fine_fault_audit_20260924/strong_5120"]
    reports, collections = [], []
    for case in cases:
        report, data = inspect(ROOT / "results/genesis_runs" / case / "fault_checkpoint.npz")
        reports.append(report)
        collections.append(data)
    payload = {"purpose": "Read-only scalar lateral contrast; no inferred physical crack seeds or contact cuts.",
               "sampling_limitations": "Cell-centred source, nearest and inverse angular-distance three-neighbour reconstructions. Small offsets on coarse meshes underresolve contrast. Positive contrast alone does not establish narrow transverse localization, connectivity, front history or fracture energy.",
               "work_normalization": "Cumulative friction+viscous Joules divided by fixed total material column volume (layer mass/density), not per-cell raw Joules and not current solid volume.",
               "candidate_selection": "Five largest positive absolute two-sided 800-km shear contrasts, separated by 1600 km only for readable diagnostics. No nucleation rule is inferred.",
               "cases": reports}
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for column, (report, source) in enumerate(zip(reports, collections)):
        mesh, state, normal, tree, fields, data, selected = source
        for row, name in enumerate(("damage", "cumulative_shear")):
            ax = axes[row, column]
            x = np.linspace(-2400, 2400, 241)
            for index in selected[:3]:
                point = lateral_points(mesh.centroids[index], normal[index], x, state.radius_km)
                values = sample(point, tree, fields[name], "idw3")
                ax.plot(x, values, label=f"cell {index}")
            ax.axvline(0, color=".6", lw=.7)
            ax.set(title=f"{report['cells']} cells · {name}", xlabel="Transverse offset (km)", ylabel=name)
            ax.legend(fontsize=8)
    fig.suptitle("Diagnostic maxima of diffuse fields: finite lateral contrast does not establish a discrete crack")
    fig.savefig(args.output / "candidate_cross_sections.png", dpi=160)
    plt.close(fig)
    print(destination)
    for report in reports:
        print(json.dumps({k: report[k] for k in ("cells", "neighbor_center_spacing_km", "diagnostic_candidates")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
