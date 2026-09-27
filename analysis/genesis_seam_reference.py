"""Reproduce edge-selector ambiguity with a smooth, unlocalized weak-plane field.

This is a geometric counterexample, not a fracture simulation. Damage is uniform,
shear is zero, and no crack centerline or fracture work is prescribed. The active
cap avoids the unavoidable singularity of a globally smooth spherical line field.
Only the *orientation of the same field relative to the mesh* is varied.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tectonics.genesis_contact import ContactParameters, select_seams
from tectonics.genesis_material import face_frames
from tectonics.mesh import build_icosphere, connected_components


def reference_axes():
    """Place a small cap inside one base icosahedron face, away from its edges."""
    base = build_icosphere(0)
    return base.centroids[0].copy(), face_frames(base)[0, :, 0].copy()


def uniform_cap_state(mesh, angle_degrees, *, cap_degrees=18., damage=.8,
                      center=None, tangent=None):
    """Project one constant spatial vector into every chord-face tangent plane.

    The normal direction varies smoothly because of sphere curvature. It has no
    dependence on cell id, stress, damage gradients, or the selected edges.
    ``center`` and ``tangent`` can be jointly rotated with the mesh for a
    coordinate-objectivity check.
    """
    default_center, default_tangent = reference_axes()
    center = default_center if center is None else np.asarray(center, dtype=float)
    tangent = default_tangent if tangent is None else np.asarray(tangent, dtype=float)
    if (center.shape != (3,) or tangent.shape != (3,)
            or not np.isfinite(center).all() or not np.isfinite(tangent).all()
            or not np.isclose(np.linalg.norm(center), 1.)
            or not np.isclose(np.linalg.norm(tangent), 1.)
            or not np.isclose(center @ tangent, 0., atol=1e-12)):
        raise ValueError("Reference axes must be finite orthogonal unit vectors")
    if not np.isfinite([angle_degrees, cap_degrees, damage]).all() or not 0 < cap_degrees < 20.:
        raise ValueError("Reference cap must lie inside the base face (0 < angle < 20 degrees)")
    if not 0 <= damage <= 1:
        raise ValueError("Reference damage must lie in [0, 1]")
    angle = np.deg2rad(angle_degrees)
    direction = np.cos(angle)*tangent + np.sin(angle)*np.cross(center, tangent)
    active = mesh.centroids @ center >= np.cos(np.deg2rad(cap_degrees))
    normals = np.einsum("fij,i->fj", face_frames(mesh), direction)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths[active] <= 1e-12):
        raise ValueError("Reference projection vanishes in the active cap")
    normals = np.divide(normals, lengths[:, None], out=np.zeros_like(normals),
                        where=active[:, None])
    return SimpleNamespace(fault_active=active, damage=np.full(mesh.cell_count, damage),
                           plane_normal=normals, cumulative_shear=np.zeros(mesh.cell_count))


def cut_graph_summary(mesh, cuts):
    """Count topological components, without treating cohesive cuts as rupture."""
    selected = {tuple(sorted(map(int, edge))) for edge in cuts}
    neighbors = [[] for _ in range(mesh.cell_count)]
    total_length = 0.
    for a, b, u, v in mesh.shared_edges:
        if (u, v) in selected:
            total_length += np.arccos(np.clip(mesh.vertices[u] @ mesh.vertices[v], -1., 1.))
        else:
            neighbors[a].append(b)
            neighbors[b].append(a)
    regions = connected_components(range(mesh.cell_count), tuple(map(tuple, neighbors)))
    regions.sort(key=len, reverse=True)
    sizes = [len(region) for region in regions]
    return {"cut_count": len(selected), "cut_length_unit_sphere": float(total_length),
            "component_count": len(sizes), "two_cell_component_count": sizes.count(2),
            "component_sizes": sizes,
            "largest_component_area_fraction": float(max(
                mesh.areas_unit_sphere[region].sum() for region in regions)/(4*np.pi))}


def evaluate_reference(mesh, angle_degrees, *, parameters=None):
    parameters = parameters or ContactParameters()
    state = uniform_cap_state(mesh, angle_degrees)
    cuts = select_seams(mesh, state, parameters)
    report = cut_graph_summary(mesh, cuts)
    report.update(angle_degrees=float(angle_degrees), cell_count=mesh.cell_count,
                  active_cell_count=int(state.fault_active.sum()))
    return report, state, cuts


def _plot(output, meshes, sweeps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    center, first = reference_axes()
    second = np.cross(center, first)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, mesh in zip(axes[0], meshes):
        data, state, cuts = evaluate_reference(mesh, 0.)
        # Gnomonic coordinates turn great-circle edges into straight lines.
        denominator = mesh.vertices @ center
        xy = np.column_stack((mesh.vertices @ first, mesh.vertices @ second))
        xy = np.divide(xy, denominator[:, None], out=np.zeros_like(xy),
                       where=np.abs(denominator[:, None]) > 1e-12)*180/np.pi
        visible = [edge[2:] for edge in mesh.shared_edges
                   if state.fault_active[edge[0]] or state.fault_active[edge[1]]]
        ax.add_collection(LineCollection(xy[np.asarray(visible)], color="#c6cbd0", linewidth=.7))
        if len(cuts):
            ax.add_collection(LineCollection(xy[cuts], color="#c23934", linewidth=1.6))
        theta = np.linspace(0, 2*np.pi, 200)
        radius = np.tan(np.deg2rad(18))*180/np.pi
        ax.plot(radius*np.cos(theta), radius*np.sin(theta), color="#44779a", ls="--", lw=1)
        ax.set(xlim=(-23, 23), ylim=(-23, 23), aspect="equal",
               title=f"{mesh.cell_count} cells: {data['two_cell_component_count']} two-cell pieces\n"
                     f"{data['cut_count']} cuts; direction angle 0°", xlabel="Projected angle (degrees)")
    axes[0, 0].set_ylabel("Projected angle (degrees)")
    for mesh, sweep in zip(meshes, sweeps):
        angle = [row["angle_degrees"] for row in sweep]
        axes[1, 0].plot(angle, [row["two_cell_component_count"] for row in sweep], label=str(mesh.cell_count))
        axes[1, 1].plot(angle, [row["cut_length_unit_sphere"] for row in sweep], label=str(mesh.cell_count))
    axes[1, 0].set(xlabel="Direction relative to mesh (degrees)", ylabel="Two-cell components",
                   title="Same uniform field, different orientation")
    axes[1, 1].set(xlabel="Direction relative to mesh (degrees)", ylabel="Selected edge length / sphere radius",
                   title="Cut length also depends on discretization")
    for ax in axes[1, :2]:
        ax.grid(alpha=.2)
        ax.legend(title="Total cells")
    ax = axes[1, 2]
    ax.axis("off")
    ax.text(0, .98, "Analytic cause on equilateral triangles\n\n"
            "Unoriented edge normals are 60° apart.\n"
            "A ±35° acceptance window can select\n"
            "two edge families at once. Cutting both\n"
            "leaves only one uncut neighbor per face:\n"
            "two triangles form isolated rhombi.\n\n"
            "This input has uniform damage D = 0.8,\n"
            "zero shear and no crack centerlines.\n"
            "It tests extraction, not fracture physics.\n\n"
            "Raw cut components remain connected\n"
            "by the contact model's cohesive forces.\n"
            "They are not established plates.", va="top", fontsize=11, linespacing=1.5)
    fig.suptitle("Smooth weak-plane field produces mesh-sized components under the legacy selector", fontsize=15)
    fig.savefig(output / "reference.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if any((output / name).exists() for name in ("reference.json", "reference.png")):
        raise FileExistsError("Reference results already exist; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    meshes = [build_icosphere(subdivision) for subdivision in (2, 3, 4)]
    sweeps = [[evaluate_reference(mesh, angle)[0] for angle in range(61)] for mesh in meshes]
    result = {"format": "genesis-seam-reference-0.1", "contact_parameters": asdict(ContactParameters()),
              "input": {"cap_degrees": 18., "uniform_damage": .8, "cumulative_shear": 0.,
                        "prescribed_crack_centerlines": 0},
              "interpretation": "Geometric counterexample only: raw cut components are not physically ruptured plates.",
              "analytic_limit": "On equilateral triangles, edge-normal families are 60 degrees apart; a 35-degree half-window can select two families, isolating pairs of triangles.",
              "sources": [
                  {"url": "https://www.scipedia.com/public/Cervera_Chiumenti_2006b",
                   "relevance": "Damage regularization and crack-path tracking address distinct mesh dependence problems; not a calibration of this planetary model."},
                  {"url": "https://arxiv.org/abs/2006.03617",
                   "relevance": "Example of a diffuse-to-sharp method that resolves propagation near tips; not a justification for thresholding all aligned mesh edges."}],
              "angle_sweeps": sweeps}
    (output / "reference.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _plot(output, meshes, sweeps)
    print(json.dumps({"output": str(output), "angle_zero": [sweep[0] for sweep in sweeps],
                      "angle_thirty": [sweep[30] for sweep in sweeps]}, indent=2))


if __name__ == "__main__":
    main()
