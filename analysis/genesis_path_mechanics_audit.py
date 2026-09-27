"""Measure the mechanical changes caused by conforming spherical refinement.

These are *zero-cut remeshing* diagnostics. Neither relaxed energy nor the
change in equilibrium potential is a fracture release rate. Crack comparisons
must subsequently use one common refined material mesh and unchanged loads.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import norm as sparse_norm

from tectonics.genesis_path_basis import tangent_prolongation



def _norm(value):
    return float(sparse_norm(value) if sparse.issparse(value) else np.linalg.norm(value))


def _difference(before, after):
    absolute = _norm(after-before)
    return absolute, absolute/max(_norm(before), _norm(after), np.finfo(float).tiny)


def audit_remeshing(old_shell, new_shell, insertion):
    """Return JSON-ready zero-cut mechanical audit, never a propagation test.

    Relative pullback errors use the maximum of the old and pulled-back norms,
    so zero prestress/forcing remains well defined. Absolute errors are also
    included. Existing validity guards remain visible; no failed admissibility
    or changed equilibrium is repaired by this diagnostic.
    """
    if old_shell.radius_m != new_shell.radius_m:
        raise ValueError("Remeshing audit requires one unchanged physical radius")
    if (not np.array_equal(insertion.mesh.faces, new_shell.mesh.faces)
            or not np.array_equal(insertion.mesh.vertices, new_shell.mesh.vertices)):
        raise ValueError("Refined shell does not match insertion geometry")
    p = tangent_prolongation(old_shell.mesh, insertion)
    old, new = old_shell.solve([]), new_shell.solve([])
    radius = old_shell.radius_m
    report = {"interpretation": "zero_cut_remeshing_not_fracture_energy",
              "relative_error_scale": "max(old_norm,pulled_back_norm)",
              "old_equilibrium_admissible": bool(old.admissible_open_crack),
              "new_equilibrium_admissible": bool(new.admissible_open_crack),
              "old_rejection_reasons": list(old.rejection_reasons),
              "new_rejection_reasons": list(new.rejection_reasons),
              "old_equilibrium_residual": old.equilibrium_residual,
              "new_equilibrium_residual": new.equilibrium_residual,
              "old_max_added_strain": old.max_added_strain,
              "new_max_added_strain": new.max_added_strain,
              "old_max_motion_edge_fraction": old.max_motion_edge_fraction,
              "new_max_motion_edge_fraction": new.max_motion_edge_fraction,
              "old_max_strain_limit": old_shell.max_strain,
              "new_max_strain_limit": new_shell.max_strain,
              "old_max_motion_edge_fraction_limit": old_shell.max_motion_edge_fraction,
              "new_max_motion_edge_fraction_limit": new_shell.max_motion_edge_fraction}
    for name, before, after in (
            ("stiffness", old.bulk_matrix, p.T@new.bulk_matrix@p),
            ("prestress", old.initial_bulk_force, p.T@new.initial_bulk_force),
            ("force", old.external_force, p.T@new.external_force),
            ("gauge", old.constraints, new.constraints@p)):
        absolute, relative = _difference(before, after)
        report[name+"_pullback_absolute_error"] = absolute
        report[name+"_pullback_relative_error"] = relative

    radial = np.zeros(old.membrane.ndof)
    radial[-1] = 1.
    lifted_radial = p@radial
    strain = np.einsum("fai,fi->fa", new.membrane.b,
                       lifted_radial[new.membrane.dofs])/radius
    expected = np.array([1., 1., 0.])/radius
    report["radial_patch_strain_relative_error"] = float(np.max(np.abs(strain-expected))*radius)
    old_radial_energy = float(radial@(old.bulk_matrix@radial))
    new_radial_energy = float(lifted_radial@(new.bulk_matrix@lifted_radial))
    report["radial_patch_energy_relative_error"] = abs(new_radial_energy-old_radial_energy)/max(
        abs(old_radial_energy), abs(new_radial_energy), np.finfo(float).tiny)
    rotation_lift_error, rotation_strain = [], []
    for axis in np.eye(3):
        q = np.zeros(old.membrane.ndof)
        q[:-1] = np.einsum("vij,vi->vj", old.membrane.vertex_basis,
                           radius*np.cross(axis, old_shell.mesh.vertices)).ravel()
        expected = np.zeros(new.membrane.ndof)
        expected[:-1] = np.einsum("vij,vi->vj", new.membrane.vertex_basis,
                           radius*np.cross(axis, new_shell.mesh.vertices)).ravel()
        lifted = p@q
        rotation_lift_error.append(_difference(expected, lifted)[1])
        rotation_strain.append(float(np.max(np.abs(np.einsum("fai,fi->fa",
            new.membrane.b, lifted[new.membrane.dofs])/radius))))
    report["rotation_patch_displacement_relative_error"] = max(rotation_lift_error)
    report["rotation_patch_max_strain"] = max(rotation_strain)

    lifted = p@old.displacement_m
    lifted_reduced = .5*float(lifted@(new.bulk_matrix@lifted))+float(
        (new.initial_bulk_force-new.external_force)@lifted)
    initial_change = new_shell.initial_energy_j-old_shell.initial_energy_j
    # Difference the small reduced potentials first, avoiding cancellation
    # against a potentially much larger common prestress energy.
    potential_change = initial_change+new.reduced_potential_j-old.reduced_potential_j
    lifted_change = initial_change+lifted_reduced-old.reduced_potential_j
    report.update(
        old_initial_energy_j=old_shell.initial_energy_j,
        new_initial_energy_j=new_shell.initial_energy_j,
        initial_energy_change_j=initial_change,
        old_reduced_potential_j=old.reduced_potential_j,
        new_reduced_potential_j=new.reduced_potential_j,
        zero_cut_remeshing_potential_change_j=potential_change,
        zero_cut_remeshing_lifted_potential_change_j=lifted_change,
        zero_cut_remeshing_relaxation_j=lifted_reduced-new.reduced_potential_j,
        equilibrium_stored_energy_change_j=new.stored_energy_j-old.stored_energy_j,
        equilibrium_external_work_change_j=new.external_potential_work_j-old.external_potential_work_j)
    if any(isinstance(value, (float, np.floating)) and not np.isfinite(value)
           for value in report.values()):
        raise ValueError("Nonfinite remeshing audit")
    return report
