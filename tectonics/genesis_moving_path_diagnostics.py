"""Reaction diagnostics on a material path in an unbroken moving shell.

Only the parent shell has evolved. Its current strain, volume and force balance
are inherited exactly by the tied assumed-strain enrichment. Child material
shares belong to MaterialPathSupport, not to a new geometric area partition.
The returned path state is a current-reference mechanical snapshot; the moving
owner must retain its mesh, last velocity and thermal history for continuation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .genesis_path_birth import recover_tied_tractions, classify_tied_onset
from .genesis_path_dynamics import PathMechanics
from .genesis_path_geometry import PathGeometryParameters
from .genesis_shell import Membrane


@dataclass(frozen=True)
class MovingPathObservation:
    model: object
    state: object
    stress_pa: np.ndarray
    water_per_trace: np.ndarray
    recovery: object
    onset: object
    parent_force_relative_error: float
    parent_energy_relative_error: float
    relative_drag_force_norm_n: float


def observe_material_path(moving_model, state, support, water_access, force_at_geometry,
                          law_parameters=None):
    """Observe a solved, entirely continuous shell without advancing time.

    ``force_at_geometry(mesh, radius_m, depth_m, membrane)`` supplies the same
    physical load convention as the moving solve. Last accepted parent forces
    and drag are authoritative; the callback supplies the fine relative load.
    The reconstructed endpoint tractions retain the existing prior convention.
    This does not discover a path or evolve an activated interface.
    """
    moving_model._validate_state(state)
    if state.accepted_steps < 1:
        raise ValueError("Path observation needs a solved moving-shell state")
    mesh = moving_model.mesh_for(state)
    water = np.asarray(water_access)
    if (water.shape != (mesh.cell_count,) or water.dtype.kind not in "fiu"
            or not np.isfinite(water).all() or np.any((water < 0) | (water > 1))):
        raise ValueError("Parent water must be finite fractions")
    if not callable(force_at_geometry):
        raise ValueError("Path observation requires a physical force callback")
    basis = support.basis_at(mesh, state.radius_m)
    projection = basis.subdivision
    volume = support.extensive(state.last_volume_m3)
    fine_area = projection.mesh.areas_unit_sphere*state.radius_m**2
    depth = volume/fine_area
    model = PathMechanics(basis, depth, moving_model.parameters, law_parameters,
                          geometry_parameters=PathGeometryParameters())
    elasticity = projection.intensive(state.last_young_modulus_pa)[:, None, None]*basis.membrane.d
    elastic = projection.tensor(state.elastic_strain, engineering=True)
    stress = np.einsum("fij,fj->fi", elasticity, elastic)
    _, internal = basis.bulk(volume, elasticity, elastic)
    parent_internal = moving_model.force_diagnostics(state)["internal_force_n"]
    force_error = float(np.linalg.norm(internal[:basis.nparent]-parent_internal)
        /max(np.linalg.norm(parent_internal), 1.))
    parent_stress = moving_model.stress(state)
    parent_energy = .5*float(np.sum(state.last_volume_m3*np.einsum(
        "fi,fi->f", state.elastic_strain, parent_stress)))
    fine_energy = .5*float(np.sum(volume*np.einsum("fi,fi->f", elastic, stress)))
    energy_error = abs(parent_energy-fine_energy)/max(abs(parent_energy), 1.)
    if force_error > 1e-10 or energy_error > 1e-10:
        raise ValueError("Moving tied support failed to inherit parent mechanics")
    fine_membrane = Membrane(projection.mesh, moving_model.poisson_ratio)
    fine_force = np.asarray(force_at_geometry(projection.mesh, state.radius_m, depth, fine_membrane))
    if fine_force.shape != (fine_membrane.ndof,) or not np.isfinite(fine_force).all():
        raise ValueError("Fine physical force callback returned an invalid force")
    parent_vertex = basis.topology.parent_vertex
    area = basis.fine_drag_area_m2[:-1:2]
    total = np.bincount(parent_vertex, weights=area, minlength=projection.mesh.vertex_count)
    split = np.zeros(basis.membrane.ndof)
    split[:-1] = (fine_force[:-1].reshape(-1, 2)[parent_vertex]
        *(area/total[parent_vertex])[:, None]).ravel()
    split[-1] = fine_force[-1]
    external = basis.external(state.last_external_force_n, split)
    # Both banks inherit exactly the same background velocity. W is balanced
    # by their current nodal areas, so their relative drag is zero to roundoff.
    background_increment = basis.parent_displacement_operator@state.last_increment_current_m
    fine_drag = (basis.fine_drag_area_m2*moving_model.parameters.basal_drag_pa_s_m
                 /(state.last_step_years*365.25*86400.))*background_increment
    relative_drag = np.asarray(basis.W.T@fine_drag).ravel()
    drag = np.r_[state.last_drag_force_n, relative_drag]
    path_state = model.initial(elastic)
    path_state.elapsed_years = state.elapsed_years
    path_state.last_step_years = state.last_step_years
    path_state.accepted_steps = state.accepted_steps
    path_state.rejected_steps = state.rejected_steps
    path_state.equilibrium_residual = state.equilibrium_residual
    path_state.constraint_reaction_n = external-internal-drag
    path_state.constraint_reaction_n[:basis.nparent] = 0.
    for name in ("external_work_j", "drag_work_j", "bulk_work_j",
                 "bulk_loading_correction_j", "mechanical_remainder_j"):
        setattr(path_state, name, getattr(state, name))
    water_trace = projection.intensive(water)[basis.topology.seam_faces].mean(axis=1).repeat(2)
    recovery = recover_tied_tractions(model, path_state, stress)
    onset = classify_tied_onset(recovery, water_trace, model.law_parameters)
    return MovingPathObservation(model, path_state, stress, water_trace, recovery, onset,
        force_error, float(energy_error), float(np.linalg.norm(relative_drag)))
