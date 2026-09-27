"""Continuous crack support insertion, material conservation and mechanics audit."""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tectonics.genesis_crack_path import CrackInterval, ReferenceCrackPath
from tectonics.genesis_faults import load_fault_checkpoint
from tectonics.genesis_material import material_column_depth
from tectonics.genesis_mobile import _external_force
from tectonics.genesis_path_mesh import insert_crack_path
from tectonics.genesis_path_material import ConservativeSubdivision
from tectonics.genesis_seams import split_mesh
from tectonics.genesis_seam_diagnostics import seam_connectivity
from tectonics.genesis_shell import Membrane, mantle_traction
from tectonics.genesis_shell_release import FrozenShell
from tectonics.genesis_unilateral import UnilateralShell
from tectonics.mesh import build_icosphere
from analysis.genesis_path_mechanics_audit import audit_remeshing

BASE = ROOT/"results/genesis_runs"
SOURCE = BASE/"fine_fault_audit_20260924/strong_5120/fault_checkpoint.npz"
TRACE = BASE/"ridge_tracking_20260924/source_trace_verified/trace_5120_cumulative_shear_divided_by_0p1.npz"


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quality(mesh):
    xyz = mesh.vertices[mesh.faces]
    sides = np.roll(xyz, -1, axis=1)-xyz
    quality = 2*np.sqrt(3.)*np.linalg.norm(np.cross(sides[:, 0], -sides[:, 2]), axis=1)/np.sum(sides*sides, axis=(1, 2))
    return float(np.min(quality))


def _force(mesh, traction, radius_m, young):
    membrane = Membrane(mesh, .25)
    force = _external_force(membrane, traction, radius_m/1000., young)*(young*radius_m*1000.)
    return np.einsum("vij,vj->vi", membrane.vertex_basis, force[:-1].reshape(-1, 2)), force[-1]


def _relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a), np.linalg.norm(b), 1e-300))


def _max_relative(a, b):
    return float(np.max(np.abs(a-b)/np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)))


def source_case(output):
    hashes = {str(SOURCE.relative_to(ROOT)): _hash(SOURCE), str(TRACE.relative_to(ROOT)): _hash(TRACE)}
    model, state, thermal, orbit, _ = load_fault_checkpoint(SOURCE)
    mesh = model.mesh_for(state)
    with np.load(TRACE, allow_pickle=False) as data:
        path = ReferenceCrackPath(data["points_xyz"], state.radius_km)
    started = perf_counter()
    inserted = insert_crack_path(mesh, path, front_coordinates_m=path.length_m*np.array([.25, .5, .75]))
    insertion_seconds = perf_counter()-started
    projection = ConservativeSubdivision(mesh, inserted.mesh, inserted.parent_face)
    bundle = projection.fault_fields(state)
    parent = inserted.parent_face
    mass, enthalpy = bundle["layer_mass_kg"], bundle["column_enthalpy"]
    def totals(value):
        summed = np.zeros((mesh.cell_count,)+value.shape[1:])
        np.add.at(summed, parent, value)
        return summed
    mass_error = _relative(totals(mass), state.layer_mass_kg)
    heat_error = _relative(totals(mass*enthalpy), state.layer_mass_kg*state.column_enthalpy)
    work_error = max(_relative(totals(bundle[key]), getattr(state, key)) for key in ("friction_work_cell_j", "viscous_work_cell_j"))
    parent_mass_error = _max_relative(totals(mass), state.layer_mass_kg)
    parent_heat_error = _max_relative(totals(mass*enthalpy), state.layer_mass_kg*state.column_enthalpy)
    parent_work_error = max(_max_relative(totals(bundle[key]), getattr(state, key)) for key in ("friction_work_cell_j", "viscous_work_cell_j"))
    old = FrozenShell.from_fault_checkpoint(SOURCE)
    mass_fraction = mass.sum(axis=1)/state.layer_mass_kg.sum(axis=1)[parent]
    depth = old.depth_m[parent]*mesh.areas_unit_sphere[parent]*mass_fraction/inserted.mesh.areas_unit_sphere
    traction = mantle_traction(inserted.mesh, model.p, depth/1000.)
    force, radial = _force(inserted.mesh, traction, old.radius_m, model.p.young_modulus_pa)
    refined = FrozenShell(inserted.mesh, old.radius_m, depth, old.elasticity_pa[parent],
        bundle["elastic_strain"], force, radial_force_n=radial)
    old_energy = .5*np.einsum("fi,fij,fj,f->f", old.elastic_strain, old.elasticity_pa, old.elastic_strain, old.reference_volume_m3)
    new_energy = .5*np.einsum("fi,fij,fj,f->f", refined.elastic_strain, refined.elasticity_pa, refined.elastic_strain, refined.reference_volume_m3)
    parent_elastic_error = _max_relative(totals(new_energy), old_energy)
    mechanics = audit_remeshing(old, refined, inserted)
    topology = split_mesh(inserted.mesh, inserted.cut_edges)
    connection = seam_connectivity(topology, np.zeros(2*len(inserted.cut_edges), dtype=bool))
    child_depth = material_column_depth(inserted.mesh, state.radius_km, mass, model.p.density_kg_m3)
    source_depth = material_column_depth(mesh, state.radius_km, state.layer_mass_kg, model.p.density_kg_m3)
    arrays = dict(vertices=inserted.mesh.vertices, faces=inserted.mesh.faces,
        parent_face=parent, path_vertex_ids=inserted.path_vertex_ids,
        path_arclength_m=inserted.path_arclength_m, vertex_parent_face=inserted.vertex_parent_face,
        vertex_barycentric=inserted.vertex_barycentric, area_fraction=projection.area_fraction,
        frame_rotation=projection.frame_rotation, cut_edges=inserted.cut_edges,
        split_vertices=topology.mesh.vertices, split_faces=topology.mesh.faces,
        split_parent_vertex=topology.parent_vertex, **bundle)
    metadata = {"format": "genesis-inserted-material-diagnostic-0.1", "time_myr": state.time_myr,
        "source_sha256": hashes, "radius_km": state.radius_km,
        "is_evolving_checkpoint": False, "physical_crack_activated": False,
        "scalar_source_history": {f.name: getattr(state, f.name) for f in fields(state)
                                  if not isinstance(getattr(state, f.name), np.ndarray)}}
    archive = output/"source_material.npz"
    np.savez_compressed(archive, metadata=np.asarray(json.dumps(metadata, allow_nan=False)), **arrays)
    report = {"source_sha256": hashes, "age_myr": state.time_myr,
        "path_length_km": path.length_m/1000, "insertion_wall_seconds": insertion_seconds,
        "old_faces": mesh.cell_count, "new_faces": inserted.mesh.cell_count,
        "new_vertices": inserted.mesh.vertex_count, "support_edges": len(inserted.cut_edges),
        "parent_area_max_relative_error": projection.area_relative_error,
        "layer_mass_relative_error": mass_error, "column_heat_relative_error": heat_error,
        "dissipation_work_relative_error": work_error,
        "max_per_parent_layer_mass_relative_error": parent_mass_error,
        "max_per_parent_column_heat_relative_error": parent_heat_error,
        "max_per_parent_dissipation_work_relative_error": parent_work_error,
        "max_per_parent_elastic_energy_relative_error": parent_elastic_error,
        "column_depth_relative_error": _relative(child_depth, source_depth[parent]),
        "stored_elastic_energy_relative_error": _relative(refined.initial_energy_j, old.initial_energy_j),
        "old_min_triangle_quality": _quality(mesh), "new_min_triangle_quality": _quality(inserted.mesh),
        "hypothetical_connectivity_all_support_edges_cut_and_unbonded": connection,
        "zero_cut_remeshing_mechanics": mechanics,
        "archive_sha256": _hash(archive), "source_unchanged": all(_hash(ROOT/p) == digest for p, digest in hashes.items()),
        "checks": {"per_parent_area": projection.area_relative_error < 2e-12,
            "mass_heat_work": max(parent_mass_error, parent_heat_error, parent_work_error) < 2e-14,
            "depth": _relative(child_depth, source_depth[parent]) < 2e-12,
            "isotropic_stored_energy": parent_elastic_error < 2e-14,
            "open_path_no_detached_chips": connection["cut_component_count"] == connection["cohesive_component_count"] == 1}}
    return report, (mesh, inserted, bundle, state)


def controlled_case(subdivisions):
    mesh = build_icosphere(subdivisions)
    center = np.array([.31, -.72, .62]); center /= np.linalg.norm(center)
    tangent = np.cross(center, [.8, .1, .3]); tangent /= np.linalg.norm(tangent)
    normal = np.cross(center, tangent)
    radius_m, length_m = 5.3e6, 2.5e6
    angle = length_m/(2*radius_m)
    path = ReferenceCrackPath(np.array([np.cos(angle)*center-np.sin(angle)*tangent,
                                       np.cos(angle)*center+np.sin(angle)*tangent]), radius_m/1000)
    insertion = insert_crack_path(mesh, path, front_coordinates_m=length_m*np.array([.25, .5, .75]))
    refined = insertion.mesh
    traction = (refined.vertices@normal)[:, None]*(normal-(refined.vertices@normal)[:, None]*refined.vertices)
    force, radial = _force(refined, traction, radius_m, 60e9)
    stiffness = np.broadcast_to(60e9*Membrane(refined, .25).d, (refined.cell_count, 3, 3))
    shell = FrozenShell(refined, radius_m, np.full(refined.cell_count, 1e4), stiffness,
        np.zeros((refined.cell_count, 3)), force, radial_force_n=radial)
    first, last = CrackInterval(.25*length_m, .5*length_m), CrackInterval(.25*length_m, .75*length_m)
    seed, trial = insertion.cuts_for(first), insertion.cuts_for(last)
    release = UnilateralShell(shell).compare_extension(seed, trial)
    return {"old_faces": mesh.cell_count, "refined_faces": refined.cell_count,
        "support_length_m": path.length_m, "seed_length_m": first.length_m,
        "extension_length_m": last.length_m-first.length_m,
        "seed_edges": len(seed), "trial_edges": len(trial),
        "release_j": release.release_j, "mean_release_j_m2": release.mean_release_j_m2,
        "added_area_m2": release.added_area_m2,
        "admissible_contact": release.admissible_open_crack,
        "rejection_reasons": release.rejection_reasons,
        "force_pullback_relative_error": release.force_pullback_relative_error,
        "stiffness_pullback_relative_error": release.stiffness_pullback_relative_error,
        "gauge_pullback_relative_error": release.gauge_pullback_relative_error,
        "release_identity_relative_error": release.relative_release_identity_error,
        "max_added_strain": release.after.max_added_strain,
        "min_triangle_quality": _quality(refined),
        "checks": {"small_contact_solution": release.admissible_open_crack,
            "same_space_stiffness_load": max(release.stiffness_pullback_relative_error, release.force_pullback_relative_error) < 1e-11,
            "release_identity": release.relative_release_identity_error < 1e-10,
            "exact_area_once": abs(release.added_area_m2/(625000.*10000.)-1) < 1e-12}}, insertion, release


def plot(case, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    mesh, inserted, bundle, state = case
    path = inserted.path
    center = path.point_at(path.length_m/2)
    east = np.cross([0., 0., 1.], center); east /= np.linalg.norm(east)
    basis = np.column_stack((east, np.cross(center, east)))
    path_xy = path.points_xyz@basis*state.radius_km
    low, high = path_xy.min(axis=0)-200., path_xy.max(axis=0)+200.
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, current, values, title in zip(axes, (mesh, inserted.mesh),
            (state.cumulative_shear, bundle["cumulative_shear"]),
            ("Исходные материальные ячейки", "Ячейки разделены по непрерывной линии")):
        xy = current.vertices@basis*state.radius_km
        visible = current.centroids@center > np.cos(2500/state.radius_km)
        ax.add_collection(PolyCollection(xy[current.faces[visible]], array=values[visible], cmap="viridis",
                                         clim=(0, float(state.cumulative_shear.max())), edgecolor="#718096", linewidth=.6))
        ax.plot(path_xy[:, 0], path_xy[:, 1], color="#ef4444", lw=2.)
        if current is inserted.mesh:
            p = xy[inserted.path_vertex_ids]
            ax.scatter(p[:, 0], p[:, 1], color="#ef4444", s=14, zorder=4)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal", title=title,
               xlabel="Локальное направление, км", ylabel="Локальное направление, км")
    fig.suptitle("Траектория 1532 км внутри сетки 5120 ячеек", fontsize=15)
    fig.text(.5, .04, "Цвет — сохранённый накопленный сдвиг. Линия задана по статическому полю; физический рост ещё не включён.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .075, 1, .94))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    actual, case = source_case(args.output)
    controlled = []
    for subdivisions in (2, 3, 4, 5):
        report, insertion, release = controlled_case(subdivisions)
        controlled.append(report)
        print(json.dumps({"mesh": report["old_faces"], "checks": report["checks"], "release_j": report["release_j"]}), flush=True)
    source_files = ("tectonics/genesis_path_mesh.py", "tectonics/genesis_path_material.py",
        "tectonics/genesis_shell_release.py", "tectonics/genesis_unilateral.py",
        "analysis/genesis_path_mechanics_audit.py", "analysis/genesis_path_insertion_validation.py")
    report = {"format": "genesis-path-insertion-validation-0.1", "source": actual,
        "ready_for_evolving_contact": False,
        "evolving_contact_blockers": ["unchanged_equilibrium_projection_missing", "slender_element_reference_limit",
                                      "diffuse_to_cohesive_history_conversion_missing"],
        "same_mesh_controlled_extensions": controlled,
        "implementation_sha256": {name: _hash(ROOT/name) for name in source_files},
        "limitations": ["The ridge is a supplied static geometric support, not physical nucleation or propagation.",
            "Material conservation does not preserve the old discrete FEM equilibrium; zero-cut remeshing is audited separately.",
            "The refined diagnostic archive is not a Fault/Coupled checkpoint; existing source history remains unchanged.",
            "Fracture release is compared only on the same already-inserted mesh and does not include remeshing energy.",
            "Conforming insertion can create slender elements; geometry quality and total-reference guards remain material limitations.",
            "No refinement-independent crack energy, finite sliding, new cohesive-history conversion or mature handoff is claimed."]}
    (args.output/"validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    plot(case, args.output/"source_insertion.png")
    print(json.dumps({"source_checks": actual["checks"], "new_faces": actual["new_faces"], "source_unchanged": actual["source_unchanged"]}), flush=True)
    return 0 if all(actual["checks"].values()) and all(all(r["checks"].values()) for r in controlled) else 1


if __name__ == "__main__":
    raise SystemExit(main())
