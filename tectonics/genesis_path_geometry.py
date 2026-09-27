"""Local validity measures for the fixed-reference embedded-path experiment.

These are numerical admissibility policies, not calibrated physical error
estimates. They permit no reference reset, stress rotation, contact-history
change, large sliding or collision search. Small objective strain alone is
insufficient: the constitutive and contact operators still use fixed frames.

The independent analysis/genesis_path_geometry_audit.py established the finite
chord identities used here. This implementation deliberately has no dependency
on analysis code, and its measurements can be checked against that oracle.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Real

import numpy as np

from .genesis_seams import rebuild_seam_mesh


@dataclass(frozen=True)
class PathGeometryParameters:
    """Opt-in tolerances for a bounded, fixed-reference calculation.

    Defaults retain a 2% numerical geometry budget and an absolute 1e-4
    linear-strain discrepancy (2% of the usual .005 strain cap). Upper limits
    follow the existing contact experiment's .05 geometry and .01 strain
    ceilings; they are policy ceilings, not proofs of a 2% solution error.
    """
    max_displacement_gradient: float = .02
    max_material_rotation_rad: float = .02
    max_tangent_motion_radius_fraction: float = .02
    max_linear_strain_error: float = 1e-4
    max_relative_contact_jump_error: float = .02

    def validate(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"path_geometry.{item.name} must be finite and positive")
            ceiling = 5e-4 if item.name == "max_linear_strain_error" else .05
            if value > ceiling:
                raise ValueError(f"path_geometry.{item.name} exceeds the fixed-reference experiment limit")


def _finite_array(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite real values with shape {shape}")
    return np.asarray(value, dtype=float)


def _strain_matrix(value):
    result = np.empty((len(value), 2, 2))
    result[:, 0, 0], result[:, 1, 1] = value[:, 0], value[:, 1]
    result[:, 0, 1] = result[:, 1, 0] = value[:, 2]/2
    return result


def _principal_norm(value):
    return float(np.max(np.abs(np.linalg.eigvalsh(value)), initial=0))


def _angle(first, second):
    return np.arctan2(np.linalg.norm(np.cross(first, second), axis=-1),
                      np.einsum("...i,...i->...", first, second))


def _triangle_kinematics(reference, moved, faces):
    """Map reference orthonormal chord coordinates into deformed world space.

    The thin 3x2 deformation gradient retains out-of-plane rotation. Its polar
    frame gives a proper 3D material rotation; Green strain is measured in the
    reference frame. No spherical retraction is performed by this helper.
    """
    reference = np.asarray(reference)
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("Reference positions must have shape (vertex, 3)")
    reference = _finite_array(reference, reference.shape, "Reference positions")
    moved = _finite_array(moved, reference.shape, "Moved positions")
    faces = np.asarray(faces)
    if (faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu"
            or np.any(faces < 0) or np.any(faces >= len(reference))):
        raise ValueError("Valid integer triangle connectivity is required")
    old, new = reference[faces], moved[faces]
    old_edges = np.stack((old[:, 1]-old[:, 0], old[:, 2]-old[:, 0]), axis=2)
    new_edges = np.stack((new[:, 1]-new[:, 0], new[:, 2]-new[:, 0]), axis=2)
    first_length = np.linalg.norm(old_edges[:, :, 0], axis=1)
    old_normal = np.cross(old_edges[:, :, 0], old_edges[:, :, 1])
    normal_length = np.linalg.norm(old_normal, axis=1)
    if np.any(first_length <= 0) or np.any(normal_length <= 0):
        raise ValueError("Collapsed reference triangle")
    first = old_edges[:, :, 0]/first_length[:, None]
    old_normal /= normal_length[:, None]
    frame = np.stack((first, np.cross(old_normal, first)), axis=2)
    chart = np.einsum("fji,fjk->fik", frame, old_edges)
    try:
        gradient = new_edges@np.linalg.inv(chart)
        metric = gradient.transpose(0, 2, 1)@gradient
        eigenvalues, eigenvectors = np.linalg.eigh(metric)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Invalid triangle deformation") from exc
    if not np.isfinite(metric).all() or np.any(eigenvalues <= 0):
        raise ValueError("Collapsed or nonfinite moved triangle")
    inverse_stretch = np.einsum("fij,fj,fkj->fik", eigenvectors,
                                1/np.sqrt(eigenvalues), eigenvectors)
    rotated_frame = gradient@inverse_stretch
    new_normal = np.cross(rotated_frame[:, :, 0], rotated_frame[:, :, 1])
    rotation = rotated_frame@frame.transpose(0, 2, 1)+new_normal[:, :, None]*old_normal[:, None, :]
    h = gradient-frame
    skew = np.stack((rotation[:, 2, 1]-rotation[:, 1, 2],
        rotation[:, 0, 2]-rotation[:, 2, 0], rotation[:, 1, 0]-rotation[:, 0, 1]), axis=1)
    rotation_angle = np.arctan2(.5*np.linalg.norm(skew, axis=1),
                               .5*(np.trace(rotation, axis1=1, axis2=2)-1))
    result = {"gradient": gradient, "frame": frame, "rotation": rotation,
        "green": .5*(metric-np.eye(2)), "rotation_angle": rotation_angle,
        "gradient_norm": np.linalg.svd(h, compute_uv=False)[:, 0]}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite triangle kinematics")
    return result


def local_geometry_metrics(model, state):
    """Measure a trial without changing state or choosing acceptance limits.

    Raises ValueError for invalid geometry. Callers apply PathGeometryParameters
    and existing strain/sliding/penetration guards to the returned scalars.
    Contact comparisons use the mean of the two polar-rotated material bank
    frames, with their disagreement measured separately. They do not define a
    finite-sliding contact law, nor certify global collision-free geometry.
    """
    basis = model.basis
    mesh = basis.topology.mesh
    q = _finite_array(state.displacement_m, (basis.ndof,), "Generalized displacement")
    fine = _finite_array(basis.displacement_operator@q,
                         (2*mesh.vertex_count+1,), "Split displacement")
    radius, radial = float(basis.radius_m), float(fine[-1])
    if not np.isfinite(radius) or radius <= 0 or radius+radial <= 0:
        raise ValueError("Moved reference radius must remain finite and positive")
    tangential = np.einsum("vij,vj->vi", basis.membrane.vertex_basis,
                           fine[:-1].reshape(-1, 2))
    unit = mesh.vertices+tangential/radius
    norm = np.linalg.norm(unit, axis=1)
    if not np.isfinite(norm).all() or np.any(norm <= 0):
        raise ValueError("Invalid spherical retraction")
    unit /= norm[:, None]
    # Explicitly validate outward orientation, face area and hemisphere limits.
    # Valid metrics alone would also describe an inverted triangle as a stretch.
    rebuild_seam_mesh(basis.topology, unit)
    reference, moved = radius*mesh.vertices, (radius+radial)*unit
    affine = reference+tangential+radial*mesh.vertices
    finite = _triangle_kinematics(reference, moved, mesh.faces)
    linear_positions = _triangle_kinematics(reference, affine, mesh.faces)
    compatible = _finite_array(basis.geometric_strain(q), (mesh.cell_count, 3),
                               "Compatible linear strain")
    jump = np.asarray(model.jump_operator@q).reshape(-1, 2)
    count = len(jump)
    normals = _finite_array(model.geometry.interface_normal, (count, 3), "Interface normals")
    tangents = _finite_array(model.geometry.interface_tangent, (count, 3), "Interface tangents")
    # Seam then endpoint ordering matches build_interface_geometry.
    rotations = np.repeat(finite["rotation"][basis.topology.seam_faces], 2, axis=0)
    bank_normals = np.einsum("tbij,tj->tbi", rotations, normals)
    bank_tangents = np.einsum("tbij,tj->tbi", rotations, tangents)
    mean_normal = bank_normals.sum(axis=1)
    mean_normal_length = np.linalg.norm(mean_normal, axis=1)
    if np.any(mean_normal_length <= 1e-12):
        raise ValueError("Opposed bank normals have no defined mean frame")
    mean_normal /= mean_normal_length[:, None]
    mean_tangent = bank_tangents.sum(axis=1)
    mean_tangent -= np.einsum("ij,ij->i", mean_tangent, mean_normal)[:, None]*mean_normal
    mean_tangent_length = np.linalg.norm(mean_tangent, axis=1)
    if np.any(mean_tangent_length <= 1e-12):
        raise ValueError("Bank tangents have no defined mean frame")
    mean_tangent /= mean_tangent_length[:, None]
    banks = moved[basis.topology.bank_vertices]
    difference = (banks[:, 1]-banks[:, 0]).reshape(-1, 3)
    actual_jump = np.stack((np.einsum("ij,ij->i", difference, mean_normal),
                            np.einsum("ij,ij->i", difference, mean_tangent)), axis=1)
    bank_gaps = np.einsum("ti,tbi->tb", difference, bank_normals)
    jump_error = np.abs(actual_jump-jump)
    onset = model.law_parameters.damage_onset_opening_m
    if not np.isfinite(onset) or onset <= 0:
        raise ValueError("Contact onset opening must be finite and positive")
    result = {
        "gradient_norm": float(max(np.max(finite["gradient_norm"], initial=0),
                                    np.max(linear_positions["gradient_norm"], initial=0))),
        "material_rotation_rad": float(np.max(finite["rotation_angle"], initial=0)),
        "tangent_motion_radius_fraction": float(np.max(np.linalg.norm(tangential, axis=1), initial=0)/radius),
        "finite_green_strain": _principal_norm(finite["green"]),
        "linear_strain_error": _principal_norm(finite["green"]-_strain_matrix(compatible)),
        "relative_contact_jump_error": float(np.max(jump_error/np.maximum(np.abs(jump), onset), initial=0)),
        "bank_frame_angular_mismatch": float(max(np.max(_angle(bank_normals[:, 0], bank_normals[:, 1]), initial=0),
                                                 np.max(_angle(bank_tangents[:, 0], bank_tangents[:, 1]), initial=0))),
        "finite_gap_min_m": float(np.min(actual_jump[:, 0])) if count else 0.,
        "geometric_penetration_m": float(max(0., -np.min(bank_gaps, initial=0),
                                              -np.min(actual_jump[:, 0], initial=0))),
        "normal_jump_error_m": float(np.max(jump_error[:, 0], initial=0)),
        "tangential_jump_error_m": float(np.max(jump_error[:, 1], initial=0)),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Nonfinite local geometry metric")
    return result
