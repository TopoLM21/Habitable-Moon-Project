"""Read-only audit of local weak-plane sampling on triangular mesh edges.

This deliberately calls the production selector without altering source fault
states. Synthetic smooth fields probe the selector, not a geological model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tectonics.genesis_contact import ContactParameters, select_seams
from tectonics.genesis_faults import load_fault_checkpoint
from tectonics.genesis_material import face_frames
from tectonics.mesh import build_icosphere, connected_components


def quantiles(values):
    array = np.asarray(values, dtype=float)
    return dict(zip(("min", "p25", "median", "p75", "max"),
                    np.quantile(array, [0, .25, .5, .75, 1]).tolist())) if array.size else None


def inspect(mesh, state, angle=35.):
    edges = np.asarray(mesh.shared_edges, dtype=int)
    cuts = select_seams(mesh, state, ContactParameters(alignment_degrees=angle))
    keys = {tuple(sorted(edge)) for edge in cuts}
    selected = np.array([tuple(sorted(edge[2:])) in keys for edge in edges])
    neighbors = [[] for _ in range(mesh.cell_count)]
    for a, b, _, _ in edges[~selected]:
        neighbors[a].append(b)
        neighbors[b].append(a)
    parts = sorted(([int(x) for x in p] for p in connected_components(
        range(mesh.cell_count), tuple(tuple(x) for x in neighbors))), key=len, reverse=True)
    labels = np.empty(mesh.cell_count, dtype=int)
    for i, part in enumerate(parts):
        labels[part] = i
    degree = np.bincount(edges[selected, 2:].ravel(), minlength=mesh.vertex_count)
    face_cut_count = np.bincount(edges[selected, :2].ravel(), minlength=mesh.cell_count)
    frames = face_frames(mesh)
    plane = np.einsum("fij,fj->fi", frames, state.plane_normal)
    en = np.cross(mesh.vertices[edges[:, 2]], mesh.vertices[edges[:, 3]])
    en /= np.linalg.norm(en, axis=1)[:, None]
    misalignment = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ei,eji->ej", en,
                                                   plane[edges[:, :2]])), 0, 1)))
    face_normals = np.cross(frames[:, :, 0], frames[:, :, 1])
    left_n, right_n = face_normals[edges[:, 0]], face_normals[edges[:, 1]]
    k = np.cross(left_n, right_n)
    p = plane[edges[:, 0]]
    moved = p + np.cross(k, p) + np.cross(k, np.cross(k, p)) / (1 + np.einsum("ei,ei->e", left_n, right_n))[:, None]
    plane_difference = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ei,ei->e", moved, plane[edges[:, 1]])), 0, 1)))
    angular_candidates = misalignment <= angle
    candidate_count = np.bincount(edges[:, :2][angular_candidates], minlength=mesh.cell_count)
    eligible = state.fault_active & (state.damage >= .65)
    angular_lengths = np.arccos(np.clip(np.einsum("ei,ei->e", mesh.vertices[edges[:, 2]], mesh.vertices[edges[:, 3]]), -1, 1))
    radius = float(getattr(state, "radius_km", 1.))
    chips = []
    for i, part in enumerate(parts):
        if len(part) != 2:
            continue
        a, b = part
        boundary = np.flatnonzero((labels[edges[:, 0]] == i) ^ (labels[edges[:, 1]] == i))
        internal = np.flatnonzero(((edges[:, 0] == a) & (edges[:, 1] == b)) |
                                  ((edges[:, 0] == b) & (edges[:, 1] == a)))
        assert len(boundary) == 4 and len(internal) == 1 and selected[boundary].all()
        assert not selected[internal[0]]
        # Transport both tangent normals to their midpoint tangent plane before
        # comparing. At these spacings its difference from exact parallel
        # transport is tiny, but both raw and projected angles are recorded.
        center = mesh.centroids[part].sum(axis=0)
        center /= np.linalg.norm(center)
        pp = plane[part] - np.outer(plane[part] @ center, center)
        pp /= np.linalg.norm(pp, axis=1)[:, None]
        angle_between = float(np.degrees(np.arccos(np.clip(abs(pp[0] @ pp[1]), 0, 1))))
        chips.append({"faces": part, "boundary_edge_indices": boundary.tolist(),
                      "internal_edge_index": int(internal[0]),
                      "plane_difference_degrees": angle_between,
                      "exact_plane_difference_degrees": float(plane_difference[internal[0]]),
                      "boundary_adjacent_plane_difference_degrees": plane_difference[boundary].tolist(),
                      "perimeter_km_if_radius_available": float(angular_lengths[boundary].sum()*radius),
                      "max_boundary_misalignment_degrees": float(misalignment[boundary].max()),
                      "min_boundary_misalignment_degrees": float(misalignment[boundary].min()),
                      "internal_edge_misalignment_degrees": misalignment[internal[0]].tolist(),
                      "damage": state.damage[part].tolist(),
                      "area_fraction": float(mesh.areas_unit_sphere[part].sum() / (4 * np.pi)),
                      "boundary_vertex_degrees": degree[np.unique(edges[boundary, 2:])].tolist()})
    report = {"cells": mesh.cell_count, "alignment_degrees": angle, "cut_edges": len(cuts),
              "components": len(parts), "component_sizes": [len(x) for x in parts],
              "two_cell_components": len(chips),
              "largest_area_fraction": max(float(mesh.areas_unit_sphere[p].sum() / (4*np.pi)) for p in parts),
              "vertex_cut_degree_histogram": np.bincount(degree).tolist(),
              "face_cut_count_histogram": np.bincount(face_cut_count, minlength=4).tolist(),
              "eligible_face_angular_candidate_count_histogram": np.bincount(candidate_count[eligible], minlength=4).tolist(),
              "radius_km_if_available": radius,
              "mean_cut_length_km_if_radius_available": float(angular_lengths[selected].mean()*radius),
              "total_cut_length_km_if_radius_available": float(angular_lengths[selected].sum()*radius),
              "eligible_surface_area_km2_if_radius_available": float(mesh.areas_unit_sphere[eligible].sum()*radius**2),
              "area_per_length_spacing_km_if_radius_available": float(mesh.areas_unit_sphere[eligible].sum()*radius / angular_lengths[selected].sum()),
              "chip_plane_difference_degrees": quantiles([c["plane_difference_degrees"] for c in chips]),
              "chip_exact_plane_difference_degrees": quantiles([c["exact_plane_difference_degrees"] for c in chips]),
              "chip_boundary_adjacent_plane_difference_degrees": quantiles([a for c in chips for a in c["boundary_adjacent_plane_difference_degrees"]]),
              "chip_perimeter_km_if_radius_available": quantiles([c["perimeter_km_if_radius_available"] for c in chips]),
              "chip_boundary_misalignment_degrees": quantiles([c["max_boundary_misalignment_degrees"] for c in chips]),
              "chip_internal_misalignment_degrees": quantiles([a for c in chips for a in c["internal_edge_misalignment_degrees"]]),
              "chips": chips}
    return report, (edges, selected, parts, plane, face_cut_count)


def synthetic(subdivision):
    mesh = build_icosphere(subdivision)
    coarse = build_icosphere(0)
    anchor = coarse.centroids[0]
    # Fix the field in global space once, independently of refinement.
    face = coarse.faces[0]
    first = np.cross(coarse.vertices[face[0]], coarse.vertices[face[1]])
    second = np.cross(coarse.vertices[face[1]], coarse.vertices[face[2]])
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    if first @ second < 0:
        second *= -1
    direction = first + second
    direction /= np.linalg.norm(direction)
    frames = face_frames(mesh)
    normals = np.einsum("fji,j->fi", frames, direction)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    active = (mesh.centroids @ anchor) > np.cos(np.radians(15.))
    state = SimpleNamespace(fault_active=active, damage=np.ones(mesh.cell_count), plane_normal=normals)
    return mesh, state, anchor, direction


def draw_local(mesh, state, data, path, title, center=None, zoom_degrees=10.):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    edges, selected, parts, plane, face_count = data
    chips = [p for p in parts if len(p) == 2]
    if center is None:
        center = mesh.centroids[chips[len(chips)//2]].sum(axis=0)
        center /= np.linalg.norm(center)
    east = np.cross([0., 0., 1.], center)
    east /= np.linalg.norm(east)
    north = np.cross(center, east)
    basis = np.stack((east, north), axis=1)
    xy = mesh.vertices @ basis * 180 / np.pi
    fxy = mesh.centroids @ basis * 180 / np.pi
    visible = (mesh.centroids @ center) > np.cos(np.radians(zoom_degrees))
    visible_edges = visible[edges[:, 0]] | visible[edges[:, 1]]
    labels = np.zeros(mesh.cell_count)
    for part in chips:
        labels[part] = 1
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    polygon = PolyCollection(xy[mesh.faces[visible]], array=labels[visible], cmap="Greys", clim=(0, 3),
                             edgecolor="none", alpha=.8)
    ax.add_collection(polygon)
    ax.add_collection(LineCollection(xy[edges[visible_edges, 2:]], colors="0.8", linewidths=.8))
    ax.add_collection(LineCollection(xy[edges[visible_edges & selected, 2:]], colors="#d64b30", linewidths=2.))
    # Weak-plane tangent, unoriented: compare line geometry to selected edges.
    tangent = np.cross(mesh.centroids, plane)
    arrows = tangent @ basis
    extent = .38 * np.median(np.linalg.norm(xy[edges[visible_edges, 2]] - xy[edges[visible_edges, 3]], axis=1))
    segment = np.stack((fxy - arrows*extent, fxy + arrows*extent), axis=1)
    ax.add_collection(LineCollection(segment[visible & state.fault_active], colors="#185b93", linewidths=1.4))
    ax.set(xlim=(-zoom_degrees, zoom_degrees), ylim=(-zoom_degrees, zoom_degrees), aspect="equal",
           xlabel="Локальное направление на восток · градусы", ylabel="Локальное направление на север · градусы", title=title)
    ax.text(.02, .02, "Красное: выбранные рёбра разреза\nСинее: направление слабых плоскостей\nСерое: двухъячеечные компоненты сетки",
            transform=ax.transAxes, fontsize=10, bbox=dict(facecolor="white", alpha=.9, edgecolor="0.8"))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    output = ROOT / "results/genesis_runs/seam_extraction_audit_20260924"
    output.mkdir(exist_ok=True)
    cases = ["fault_damage_final_20260922", "fine_fault_audit_20260924/strong_1280",
             "fine_fault_audit_20260924/strong_5120"]
    reports = []
    for case in cases:
        checkpoint = ROOT / "results/genesis_runs" / case / "fault_checkpoint.npz"
        model, state, _, _, _ = load_fault_checkpoint(checkpoint)
        mesh = model.mesh_for(state)
        report, data = inspect(mesh, state)
        report["source"] = case
        report["source_checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        report["angle_sweep"] = []
        for angle in (20., 25., 29., 30., 31., 32., 33., 35., 40., 45.):
            entry, _ = inspect(mesh, state, angle)
            report["angle_sweep"].append({k: entry[k] for k in
                                          ("alignment_degrees", "cut_edges", "components", "two_cell_components", "largest_area_fraction")})
        reports.append(report)
        if mesh.cell_count == 5120:
            draw_local(mesh, state, data, output / "actual_5120_chip_geometry.png",
                       "5120 ячеек: выбор рёбер создаёт ромбы при почти параллельных слабых плоскостях")
    synthetics = []
    for subdivision in (2, 3, 4, 5):
        mesh, state, anchor, direction = synthetic(subdivision)
        report, data = inspect(mesh, state)
        report["active_cells"] = int(state.fault_active.sum())
        report["construction"] = "Constant 3D direction projected into chord planes, active only within a 15-degree cap"
        report["global_direction"] = direction.tolist()
        report["cap_center"] = anchor.tolist()
        synthetics.append(report)
        if subdivision == 4:
            draw_local(mesh, state, data, output / "smooth_plane_false_fragments.png",
                       "Контроль с гладким полем: выбор рёбер тоже создаёт двухъячеечные компоненты",
                       center=anchor, zoom_degrees=17.)
    result = {"production_cases": reports, "smooth_plane_control": synthetics}
    (output / "geometry_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"production_cases": [{k: r[k] for k in ("cells", "cut_edges", "components", "two_cell_components",
                                                               "chip_plane_difference_degrees", "chip_internal_misalignment_degrees",
                                                               "face_cut_count_histogram", "angle_sweep")} for r in reports],
                      "smooth_plane_control": [{k: r[k] for k in ("cells", "active_cells", "cut_edges", "components", "two_cell_components",
                                                                  "chip_plane_difference_degrees")} for r in synthetics]}, indent=2))


if __name__ == "__main__":
    main()
