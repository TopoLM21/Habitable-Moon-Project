"""Read-only finite-geometry audit of fixed-reference embedded-path motion.

This diagnostic does not change admissibility limits, reference geometry or
constitutive state. Objective chord metrics are compared with the fixed-frame
linear model; the latter is not justified by small objective strain alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _positions(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite (vertex, 3) positions")
    return value


def triangle_geometry(reference_positions, moved_positions, faces):
    """Return objective per-face stretch and reference-frame Green strain.

    The 3x2 deformation gradient maps orthonormal reference chord coordinates
    into world space. Green strain is in the reference material frame, while
    ``rotation`` maps the reference frame into the current polar frame. It
    transforms under a superposed rotation and must rotate stress directions.
    """
    old = _positions(reference_positions, "Reference")
    new = _positions(moved_positions, "Moved")
    faces = np.asarray(faces)
    if (new.shape != old.shape or faces.ndim != 2 or faces.shape[1] != 3
            or faces.dtype.kind not in "iu" or np.any(faces < 0)
            or np.any(faces >= len(old))):
        raise ValueError("Matching positions and valid integer triangles required")
    a, b = old[faces], new[faces]
    old_edges = np.stack((a[:, 1]-a[:, 0], a[:, 2]-a[:, 0]), axis=2)
    new_edges = np.stack((b[:, 1]-b[:, 0], b[:, 2]-b[:, 0]), axis=2)
    e1 = old_edges[:, :, 0]
    first_lengths = np.linalg.norm(e1, axis=1)
    if np.any(first_lengths <= 0):
        raise ValueError("Collapsed reference triangle")
    e1 = e1/first_lengths[:, None]
    normal = np.cross(old_edges[:, :, 0], old_edges[:, :, 1])
    lengths = np.linalg.norm(normal, axis=1)
    if np.any(lengths <= 0):
        raise ValueError("Collapsed reference triangle")
    normal /= lengths[:, None]
    frame = np.stack((e1, np.cross(normal, e1)), axis=2)
    chart = np.einsum("fji,fjk->fik", frame, old_edges)
    gradient = new_edges@np.linalg.inv(chart)
    metric = np.einsum("fji,fjk->fik", gradient, gradient)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    if np.any(eigenvalues <= 0):
        raise ValueError("Collapsed moved triangle")
    stretches = np.sqrt(eigenvalues)
    invstretch = np.einsum("fij,fj,fkj->fik", eigenvectors, 1/stretches, eigenvectors)
    rotated_frame = gradient@invstretch
    moved_normal = np.cross(rotated_frame[:, :, 0], rotated_frame[:, :, 1])
    rotation = (rotated_frame@frame.transpose(0, 2, 1)
                + moved_normal[:, :, None]*normal[:, None, :])
    green = .5*(metric-np.eye(2))

    def quality(triangles):
        edges = triangles[:, [1, 2, 0]]-triangles
        twicearea = np.linalg.norm(np.cross(edges[:, 0], -edges[:, 2]), axis=1)
        return 2*np.sqrt(3)*twicearea/np.sum(edges*edges, axis=(1, 2))

    return {"gradient": gradient, "frame": frame, "rotation": rotation,
        "green": green, "stretch": stretches-1,
        "normal_rotation_rad": np.arctan2(np.linalg.norm(np.cross(normal, moved_normal), axis=1),
                                           np.einsum("fi,fi->f", normal, moved_normal)),
        "old_quality": quality(a), "new_quality": quality(b)}


def engineering(matrix):
    return np.stack((matrix[:, 0, 0], matrix[:, 1, 1], 2*matrix[:, 0, 1]), axis=1)


def _max_principal(engineering_strain):
    value = np.asarray(engineering_strain)
    matrices = np.zeros((len(value), 2, 2))
    matrices[:, 0, 0], matrices[:, 1, 1] = value[:, 0], value[:, 1]
    matrices[:, 0, 1] = matrices[:, 1, 0] = value[:, 2]/2
    return float(np.max(np.abs(np.linalg.eigvalsh(matrices)), initial=0))


def audit(model, state):
    """Inspect a committed state, keeping all production acceptance rules."""
    basis = model.basis
    mesh = basis.topology.mesh
    fine = np.asarray(basis.displacement_operator@state.displacement_m)
    tangential = np.einsum("vij,vj->vi", basis.membrane.vertex_basis,
                           fine[:-1].reshape(-1, 2))
    radius, radial = basis.radius_m, float(fine[-1])
    old = radius*mesh.vertices
    moved_unit = mesh.vertices+tangential/radius
    moved_unit /= np.linalg.norm(moved_unit, axis=1)[:, None]
    moved = (radius+radial)*moved_unit
    linear_positions = old+tangential+radial*mesh.vertices
    finite = triangle_geometry(old, moved, mesh.faces)
    linear_finite = triangle_geometry(old, linear_positions, mesh.faces)
    geometric = basis.geometric_strain(state.displacement_m)
    assumed = basis.strain(state.displacement_m)
    actual_green = engineering(finite["green"])
    linear_green = engineering(linear_finite["green"])
    raw_h = linear_finite["gradient"]-linear_finite["frame"]
    analytic_quadratic = engineering(.5*np.einsum("fji,fjk->fik", raw_h, raw_h))
    rotations = finite["rotation"]
    skew = np.stack((rotations[:, 2, 1]-rotations[:, 1, 2],
        rotations[:, 0, 2]-rotations[:, 2, 0], rotations[:, 1, 0]-rotations[:, 0, 1]), axis=1)
    rotation_angle = np.arctan2(.5*np.linalg.norm(skew, axis=1),
                                .5*(np.trace(rotations, axis1=1, axis2=2)-1))
    # Every split face edge matters; seams have two independent bank edges.
    edges = np.unique(np.sort(mesh.faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
    old_diff, new_diff = old[edges[:, 1]]-old[edges[:, 0]], moved[edges[:, 1]]-moved[edges[:, 0]]
    old_length = np.linalg.norm(old_diff, axis=1)
    edge_strain = np.linalg.norm(new_diff, axis=1)/old_length-1
    relative_edge_motion = np.linalg.norm(new_diff-old_diff, axis=1)/old_length
    shortest, maxmoving = int(np.argmin(old_length)), int(np.argmax(np.linalg.norm(tangential, axis=1)))
    midpoint = old[edges[shortest]].mean(axis=0)
    midpoint /= np.linalg.norm(midpoint)
    angle = np.arctan2(np.linalg.norm(np.cross(midpoint, mesh.vertices[maxmoving])),
                       np.dot(midpoint, mesh.vertices[maxmoving]))
    banks = moved[basis.topology.bank_vertices]
    bank_diff = (banks[:, 1]-banks[:, 0]).reshape(-1, 3)
    linear_jump = np.asarray(model.jump_operator@state.displacement_m).reshape(-1, 2)
    normals, tangents = model.geometry.interface_normal, model.geometry.interface_tangent
    fixed_jump = np.stack((np.einsum("ij,ij->i", bank_diff, normals),
                           np.einsum("ij,ij->i", bank_diff, tangents)), axis=1)
    # Diagnostic mean of two material polar rotations. It is not a new contact
    # model, especially if independently rotating banks eventually disagree.
    rotations = np.repeat(finite["rotation"][basis.topology.seam_faces], 2, axis=0)
    bank_normals = np.einsum("tbij,tj->tbi", rotations, normals)
    bank_tangents = np.einsum("tbij,tj->tbi", rotations, tangents)
    transported_normal = bank_normals.sum(axis=1)
    transported_normal /= np.linalg.norm(transported_normal, axis=1)[:, None]
    transported_tangent = bank_tangents.sum(axis=1)
    transported_tangent -= np.einsum("ij,ij->i", transported_tangent, transported_normal)[:, None]*transported_normal
    transported_tangent /= np.linalg.norm(transported_tangent, axis=1)[:, None]
    transported_jump = np.stack((np.einsum("ij,ij->i", bank_diff, transported_normal),
        np.einsum("ij,ij->i", bank_diff, transported_tangent)), axis=1)
    worst = int(np.argmax(np.max(np.abs(finite["stretch"]), axis=1)))
    report = {"elapsed_years": state.elapsed_years, "stopped_reason": state.stopped_reason,
        "existing_geometry_metrics": model.geometry_metrics(state),
        "maximum_tangential_motion_m": float(np.linalg.norm(tangential, axis=1).max()),
        "shortest_chord_edge_m": float(old_length[shortest]),
        "shortest_edge_vertices": edges[shortest].tolist(), "maximum_motion_vertex": maxmoving,
        "max_motion_to_shortest_edge_midpoint_km": float(angle*radius/1000),
        "shortest_edge_length_strain": float(edge_strain[shortest]),
        "shortest_edge_relative_vector_change": float(relative_edge_motion[shortest]),
        "maximum_edge_length_strain_abs": float(np.max(np.abs(edge_strain))),
        "maximum_edge_relative_vector_change": float(relative_edge_motion.max()),
        "maximum_principal_stretch_abs": float(np.max(np.abs(finite["stretch"]))),
        "maximum_normal_rotation_rad": float(finite["normal_rotation_rad"].max()),
        "maximum_polar_rotation_rad": float(rotation_angle.max()),
        "maximum_affine_displacement_gradient_spectral_norm": float(np.linalg.svd(raw_h, compute_uv=False).max()),
        "maximum_tangent_motion_radius_fraction": float(np.linalg.norm(tangential, axis=1).max()/radius),
        "maximum_sphere_retraction_position_correction_m": float(np.linalg.norm(moved-linear_positions, axis=1).max()),
        "max_finite_green_minus_linear_strain": _max_principal(actual_green-geometric),
        "max_affine_quadratic_strain": _max_principal(analytic_quadratic),
        "max_sphere_retraction_strain_correction": _max_principal(actual_green-linear_green),
        "linear_affine_identity_error": float(np.max(np.abs(linear_green-geometric-analytic_quadratic))),
        "max_assumed_minus_compatible_strain": _max_principal(assumed-geometric),
        "min_reference_quality": float(finite["old_quality"].min()),
        "min_moved_quality": float(finite["new_quality"].min()),
        "minimum_quality_ratio": float(np.min(finite["new_quality"]/finite["old_quality"])),
        "max_fixed_frame_gap_difference_m": float(np.max(np.abs(fixed_jump[:, 0]-linear_jump[:, 0]))),
        "max_fixed_frame_slip_difference_m": float(np.max(np.abs(fixed_jump[:, 1]-linear_jump[:, 1]))),
        "max_corotated_gap_difference_m": float(np.max(np.abs(transported_jump[:, 0]-linear_jump[:, 0]))),
        "max_corotated_slip_difference_m": float(np.max(np.abs(transported_jump[:, 1]-linear_jump[:, 1]))),
        "worst_stretch_face": worst, "worst_stretch_parent_face": int(basis.subdivision.parent_face[worst]),
        "interpretation": "Diagnostic only. Small objective strain does not validate frozen constitutive/contact frames or permit changing guards."}
    arrays = {"moved_positions_m": moved, "actual_green_strain": actual_green,
        "affine_green_strain": linear_green, "geometric_linear_strain": geometric,
        "assumed_linear_strain": assumed, "principal_stretches_minus_one": finite["stretch"],
        "normal_rotation_rad": finite["normal_rotation_rad"], "face_polar_rotation": finite["rotation"],
        "edge_vertices": edges, "edge_length_strain": edge_strain,
        "linear_jump_m": linear_jump, "fixed_frame_actual_jump_m": fixed_jump,
        "mean_corotated_actual_jump_m": transported_jump}
    return report, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"results/genesis_runs/path_dynamics_verified_20260926")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from analysis.genesis_path_dynamics_validation import (CoupledModel, SOURCE, TRACE,
        ReferenceCrackPath, insert_crack_path, EmbeddedPathBasis, PathMechanics)
    source_files = [SOURCE, TRACE, args.input/"released_mechanics_checkpoint.npz"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    coupled = CoupledModel(SOURCE.read_bytes(), source_path=str(SOURCE))
    with np.load(TRACE, allow_pickle=False) as data:
        path = ReferenceCrackPath(data["points_xyz"], coupled.source.source_state.radius_km)
    inserted = insert_crack_path(coupled.original_mesh, path,
        front_coordinates_m=path.length_m*np.array([.25, .5, .75]))
    basis = EmbeddedPathBasis(coupled.original_mesh, inserted, coupled.radius_m,
                              coupled.source_model.p.poisson_ratio)
    model = PathMechanics(basis, basis.subdivision.intensive(coupled.source.depth_m),
                         coupled.contact_parameters, coupled.law_parameters)
    state = model.load_state(args.input/"released_mechanics_checkpoint.npz")
    report, arrays = audit(model, state)
    report["input_sha256"] = hashes
    report["inputs_unchanged"] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
                                      for p, digest in hashes.items())
    report["audit_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"audit.json").exists() or (args.output/"fields.npz").exists():
        raise ValueError("Choose an unused audit output directory")
    np.savez_compressed(args.output/"fields.npz", **arrays)
    report["fields_sha256"] = hashlib.sha256((args.output/"fields.npz").read_bytes()).hexdigest()
    (args.output/"audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
