"""Finite background motion with a persistent, small local cohesive opening.

The parent assumed-strain membrane remains the background discretization.
Enrichment offsets are transported by a full material polar rotation, never
reset on reference updates. Their compatible *increments* enter the current
assumed-strain operator. This corotational, small-local-opening model is not a
fully compatible finite-deformation split-surface finite element. In particular
the change of contact frames under background strain has geometric power that
is reported separately; it is not silently classified as heat or fracture.

Only the already born material vertex is released. Material contact areas and
birth tractions persist; newly solid material is delegated to the cohort owner.
No front propagation, second nucleation, remelting, or plate handoff is done.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .genesis_contact import SECONDS_PER_YEAR
from .genesis_contact_geometry import build_interface_geometry
from .genesis_contact_growth import CohortState
from .genesis_crack_path import CrackInterval
from .genesis_material import face_frames, move_mesh, rotate_tensor
from .genesis_moving_tied import (_array, _positive, _kinematics,
    _current_increment, MovingTiedRetry)
from .genesis_path_activation import ExtrinsicPathMechanics
from .genesis_path_material import _normal_rotation, _owned
from .genesis_seams import rebuild_seam_mesh
from .genesis_shell import Membrane, maximum_total_strain


VERSION = "genesis-moving-contact-0.1"


class MovingContactRetry(RuntimeError):
    """The complete pure trial must be retried with a smaller interval."""


@dataclass(frozen=True)
class MovingContactLoading:
    dt_years: float
    volume_m3: np.ndarray
    young_modulus_pa: np.ndarray
    memory: np.ndarray
    effective_b: np.ndarray
    water_per_trace: np.ndarray
    external_force: object


@dataclass
class MovingContactState:
    vertices: np.ndarray
    radius_m: float
    enrichment_m: np.ndarray
    elastic_strain: np.ndarray
    history: object
    elapsed_years: float
    last_step_years: float
    accepted_steps: int
    rejected_steps: int
    continuation_steps: int
    equilibrium_residual: float
    external_work_j: float
    drag_work_j: float
    bulk_work_j: float
    bulk_loading_correction_j: float
    mechanical_remainder_j: float
    contact_geometric_work_j: float
    cohort_parameter_energy_j: float
    last_increment_current_m: np.ndarray
    last_external_force_n: np.ndarray
    last_drag_force_n: np.ndarray
    last_volume_m3: np.ndarray
    last_young_modulus_pa: np.ndarray
    constraint_reaction_n: np.ndarray
    stopped_reason: str | None = None


def transport_enrichment(old_basis, new_basis, enrichment_m, old_radius_m, radius_m):
    """Full parent material rotation, corrected to the moving vertex tangent.

    A shortest rotation of vertex normals alone would lose material spin about
    the vertex. The polar rotation includes that spin and exactly carries a
    common spatial rigid rotation, even when the vertex itself does not move.
    Length is preserved: the local opening uses a corotational metric, rather
    than receiving an implicit affine rescaling on every reference update.
    """
    ids = old_basis.enrichment_vertices
    q = _array(enrichment_m, (2*len(ids),), "enrichment_m").reshape(-1, 2)
    if (not np.array_equal(ids, new_basis.enrichment_vertices)
            or not np.array_equal(old_basis.insertion.vertex_parent_face,
                                  new_basis.insertion.vertex_parent_face)):
        raise ValueError("Enrichment transport requires identical material identities")
    old, new = old_basis.subdivision.parent_mesh, new_basis.subdivision.parent_mesh
    if np.array_equal(old.vertices, new.vertices):
        return q.ravel().copy()
    polar, _ = _kinematics(old, new, old_radius_m, radius_m)
    owner = old_basis.insertion.vertex_parent_face[ids]
    a, b = face_frames(old)[owner], face_frames(new)[owner]
    na, nb = np.cross(a[:, :, 0], a[:, :, 1]), np.cross(b[:, :, 0], b[:, :, 1])
    rotation = b@polar[owner]@a.transpose(0, 2, 1)+nb[:, :, None]*na[:, None, :]
    oldxyz, newxyz = (x.subdivision.mesh.vertices[ids] for x in (old_basis, new_basis))
    arrived = np.einsum("vij,vj->vi", rotation, oldxyz)
    rotation = _normal_rotation(arrived, newxyz)@rotation
    # The unsplit and duplicated banks use the same coordinate tangent frame.
    oldframe = Membrane(old_basis.subdivision.mesh, .25).vertex_basis[ids]
    newframe = Membrane(new_basis.subdivision.mesh, .25).vertex_basis[ids]
    world = np.einsum("vij,vj->vi", oldframe, q)
    return np.einsum("vji,vjk,vk->vi", newframe, rotation, world).ravel()


class MovingContactMechanics:
    """Continue one born local contact on a materially moving parent sphere."""

    def __init__(self, moving_tied_model, support, born_model, born_state, moving_state):
        from .genesis_moving_contact_cohorts import MovingContactCohorts
        if not isinstance(born_model, ExtrinsicPathMechanics):
            raise ValueError("A traction-consistent born extrinsic contact is required")
        moving_tied_model._validate_state(moving_state)
        born_model._validate_state(born_state)
        if (born_state.elapsed_years != moving_state.elapsed_years
                or born_state.elapsed_years != born_model.birth_time_years
                or np.any(born_state.displacement_m != 0)):
            raise ValueError("Moving import requires the paired zero-gap birth state")
        self.moving_model = moving_tied_model
        self.support = support
        self.parameters = moving_tied_model.parameters
        self.law_parameters = born_model.law_parameters
        self.poisson_ratio = moving_tied_model.poisson_ratio
        basis = support.basis_at(moving_tied_model.mesh_for(moving_state), moving_state.radius_m)
        if (not np.array_equal(basis.topology.mesh.vertices, born_model.basis.topology.mesh.vertices)
                or not np.array_equal(basis.topology.mesh.faces, born_model.basis.topology.mesh.faces)
                or basis.radius_m != born_model.basis.radius_m):
            raise ValueError("Born contact and moving support geometry differ")
        self.birth_time_years = born_model.birth_time_years
        self.birth_accepted_steps = moving_state.accepted_steps
        self.birth_child_area_m2 = _owned(basis.subdivision.mesh.areas_unit_sphere*basis.radius_m**2)
        self.initial_volume_m3 = _owned(support.extensive(moving_state.last_volume_m3))
        if not np.allclose(self.initial_volume_m3, born_model.reference_volume_m3, rtol=2e-13, atol=0):
            raise ValueError("Birth material volumes disagree with the moving owner")
        arc = basis.insertion.path_arclength_m
        ends = [np.flatnonzero(np.isclose(arc, x, rtol=0, atol=1e-7)) for x in
                (born_state.active_interval.left_m, born_state.active_interval.right_m)]
        if any(len(x) != 1 for x in ends):
            raise ValueError("Birth fronts must identify unique existing material vertices")
        self.active_support_indices = tuple(int(x[0]) for x in ends)
        self.active_tip_vertex_ids = tuple(int(basis.insertion.path_vertex_ids[x]) for x in self.active_support_indices)
        self.free = _owned(basis.free_dofs(born_state.active_interval))
        self.ndof, self.nparent = basis.ndof, basis.nparent
        self.history_model = MovingContactCohorts(born_state.cohorts,
            born_model.birth_traction_pa, np.repeat(born_model.geometry.edge_length_m, 2),
            self.law_parameters)
        self.configuration = {"version": VERSION, "moving_fingerprint": moving_tied_model.fingerprint,
            "born_fingerprint": born_model.fingerprint, "active_support_indices": self.active_support_indices,
            "birth_time_years": self.birth_time_years, "birth_accepted_steps": self.birth_accepted_steps,
            "law": asdict(self.law_parameters)}
        digest = hashlib.sha256(json.dumps(self.configuration, sort_keys=True).encode())
        for value in (self.birth_child_area_m2, self.initial_volume_m3, self.free,
                      support.insertion.vertex_barycentric, support.area_fraction):
            digest.update(np.ascontiguousarray(value).tobytes())
        self.fingerprint = digest.hexdigest()
        drag = np.r_[moving_state.last_drag_force_n, np.zeros(self.ndof-self.nparent)]
        young = basis.subdivision.intensive(moving_state.last_young_modulus_pa)
        _, internal = basis.bulk(self.initial_volume_m3,
            young[:, None, None]*basis.membrane.d, born_state.elastic_strain)
        contact = born_model.jump_operator.T@(
            born_state.cohorts.area_ref_m2[:, None]*born_state.cohorts.traction_pa).ravel()
        external = internal+contact+drag+born_state.constraint_reaction_n
        # Parent forcing is authoritative from the physical moving solve;
        # its small accepted residual must not be replaced by an invented load.
        external[:self.nparent] = moving_state.last_external_force_n
        self.initial_state = MovingContactState(moving_state.vertices.copy(), moving_state.radius_m,
            born_state.displacement_m[self.nparent:].copy(), born_state.elastic_strain.copy(),
            self.history_model.initial(), moving_state.elapsed_years, moving_state.last_step_years,
            moving_state.accepted_steps, moving_state.rejected_steps, 0, moving_state.equilibrium_residual,
            moving_state.external_work_j, moving_state.drag_work_j, moving_state.bulk_work_j,
            moving_state.bulk_loading_correction_j, moving_state.mechanical_remainder_j,
            0., 0., np.r_[moving_state.last_increment_current_m, np.zeros(self.ndof-self.nparent)],
            external, drag, self.initial_volume_m3.copy(),
            young,
            born_state.constraint_reaction_n.copy())
        self._shapes = {"vertices": moving_state.vertices.shape,
            "enrichment_m": (self.ndof-self.nparent,), "elastic_strain": born_state.elastic_strain.shape,
            **{x: (self.ndof,) for x in ("last_increment_current_m", "last_external_force_n",
                "last_drag_force_n", "constraint_reaction_n")},
            "last_volume_m3": self.initial_volume_m3.shape, "last_young_modulus_pa": self.initial_volume_m3.shape}
        self.initial_state.equilibrium_residual = self.force_diagnostics(self.initial_state)["relative_residual"]
        self._validate_state(self.initial_state)

    def mesh_for(self, state):
        return self.moving_model.mesh_for(state)

    def basis_for(self, state):
        return self.support.basis_at(self.mesh_for(state), state.radius_m)

    def active_interval_for(self, basis):
        indices = self.active_support_indices
        if tuple(int(basis.insertion.path_vertex_ids[i]) for i in indices) != self.active_tip_vertex_ids:
            raise ValueError("Active contact fronts changed material identity")
        return CrackInterval(*(float(basis.insertion.path_arclength_m[i]) for i in indices))

    def _geometry(self, basis, volume):
        depth = volume/(basis.subdivision.mesh.areas_unit_sphere*basis.radius_m**2)
        geometry = build_interface_geometry(basis.subdivision.mesh, basis.topology,
            basis.radius_m, depth, basis.membrane)
        relative = (geometry.jump_operator@basis.W).tocsr()
        return geometry, sparse.hstack((sparse.csr_matrix((relative.shape[0], basis.nparent)), relative), format="csr")

    def geometry_for(self, state):
        return self._geometry(self.basis_for(state), state.last_volume_m3)

    def jump(self, state):
        basis = self.basis_for(state)
        _, j = self._geometry(basis, state.last_volume_m3)
        return (j[:, self.nparent:]@state.enrichment_m).reshape(-1, 2)

    jumps = jump

    def _external(self, basis, volume, callback):
        radius = basis.radius_m
        parent_volume = np.bincount(self.support.parent_face, weights=volume,
            minlength=basis.subdivision.parent_mesh.cell_count)
        parent = basis.subdivision.parent_mesh
        parent_force = _array(callback(parent, radius,
            parent_volume/(parent.areas_unit_sphere*radius**2), basis.parent_membrane),
            (basis.nparent,), "parent external force")
        fine = basis.subdivision.mesh
        membrane = Membrane(fine, self.poisson_ratio)
        fine_force = _array(callback(fine, radius, volume/(fine.areas_unit_sphere*radius**2), membrane),
            (membrane.ndof,), "fine external force")
        ids = basis.topology.parent_vertex
        area = basis.fine_drag_area_m2[:-1:2]
        total = np.bincount(ids, weights=area, minlength=fine.vertex_count)
        split = np.r_[(fine_force[:-1].reshape(-1, 2)[ids]*(area/total[ids])[:, None]).ravel(), fine_force[-1]]
        return basis.external(parent_force, split)

    def _residual(self, basis, internal, contact, drag, external, young, volume, coefficient):
        residual = internal+contact+drag-external
        floor = self.moving_model._force_floor(young, volume, basis.radius_m, coefficient)
        physical_scale = max(float(np.linalg.norm(x[self.free])) for x in (internal, contact, external))
        scale = max(physical_scale, floor)
        return residual, float(np.linalg.norm(residual[self.free])/scale), scale, floor, physical_scale

    def stress(self, state):
        basis = self.basis_for(state)
        return state.last_young_modulus_pa[:, None]*(state.elastic_strain@basis.membrane.d.T)

    def force_diagnostics(self, state):
        basis = self.basis_for(state)
        geometry, j = self._geometry(basis, state.last_volume_m3)
        c = state.last_young_modulus_pa[:, None, None]*basis.membrane.d
        _, internal = basis.bulk(state.last_volume_m3, c, state.elastic_strain)
        force = np.zeros((j.shape[0]//2, 2))
        for cohort in (state.history.initial, state.history.added):
            np.add.at(force, cohort.trace_index, cohort.area_ref_m2[:, None]*cohort.traction_pa)
        contact = np.asarray(j.T@force.ravel()).ravel()
        coefficient = basis.drag_area_m2*self.parameters.basal_drag_pa_s_m/(state.last_step_years*SECONDS_PER_YEAR)
        residual, ratio, scale, floor, physical = self._residual(basis, internal, contact,
            state.last_drag_force_n, state.last_external_force_n, state.last_young_modulus_pa,
            state.last_volume_m3, coefficient)
        return {"internal_force_n": internal, "contact_force_n": contact,
            "drag_force_n": state.last_drag_force_n.copy(),
            "external_force_n": state.last_external_force_n.copy(), "free_dofs": self.free.copy(),
            "residual_force_n": residual, "relative_residual": ratio,
            "raw_relative_residual": float(np.linalg.norm(residual[self.free])/max(physical, 1.)),
            "force_scale_n": scale, "force_floor_n": floor,
            "absolute_roundoff_tolerance_n": floor*self.parameters.equilibrium_tolerance,
            "current_interface_area_m2": geometry.interface_area_m2.copy(),
            "reference_interface_area_m2": self.history_model.birth_edge_length_m_per_trace
                *self.history_model.front_depth(state.history)/2,
            "reference_to_current_area_ratio": self.history_model.birth_edge_length_m_per_trace
                *self.history_model.front_depth(state.history)/(2*geometry.interface_area_m2)}

    def _validate_state(self, state, *, check_force=True):
        if not isinstance(state, MovingContactState):
            raise ValueError("State must be MovingContactState")
        for name, shape in self._shapes.items():
            _array(getattr(state, name), shape, name)
        counts = {"accepted_steps", "rejected_steps", "continuation_steps"}
        for item in fields(state):
            name, value = item.name, getattr(state, item.name)
            if name in self._shapes or name in {"history", "stopped_reason"}:
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
                raise ValueError("Invalid moving contact scalar "+name)
            if name in counts and (not isinstance(value, (int, np.integer)) or value < 0):
                raise ValueError("Invalid moving contact count "+name)
        if (state.radius_m <= 0 or state.elapsed_years < self.birth_time_years
                or state.last_step_years <= 0 or state.continuation_steps > state.accepted_steps
                or state.accepted_steps != self.birth_accepted_steps+state.continuation_steps
                or state.last_step_years > state.elapsed_years+1e-9
                or state.equilibrium_residual < 0 or state.equilibrium_residual > self.parameters.equilibrium_tolerance
                or state.drag_work_j < 0 or np.any(state.last_volume_m3 <= 0)
                or np.any(state.last_young_modulus_pa <= 0)):
            raise ValueError("Invalid moving contact clock, material, or residual")
        if state.stopped_reason is not None and (not isinstance(state.stopped_reason, str) or not state.stopped_reason):
            raise ValueError("Invalid stop reason")
        basis = self.basis_for(state)
        if not np.array_equal(basis.free_dofs(self.active_interval_for(basis)), self.free):
            raise ValueError("Moving contact release changed material identity")
        tied = np.setdiff1d(np.arange(self.ndof), self.free)
        if (np.any(state.enrichment_m[tied-self.nparent] != 0)
                or np.any(state.constraint_reaction_n[self.free] != 0)):
            raise ValueError("Tied enrichment or free reactions are inconsistent")
        _, j = self._geometry(basis, state.last_volume_m3)
        jumps = (j[:, self.nparent:]@state.enrichment_m).reshape(-1, 2)
        self.history_model.validate(state.history, jumps[:, 0], jumps[:, 1])
        observed = np.asarray(j[:, self.free].power(2).sum(axis=1)).ravel().reshape(-1, 2).sum(axis=1) > 0
        for cohort in (state.history.initial, state.history.added):
            if np.any(cohort.birth_time_myr > state.elapsed_years/1e6+1e-14):
                raise ValueError("Saved cohort is younger than the mechanical state")
            untouched = ~observed[cohort.trace_index]
            if any(np.any(getattr(cohort, name)[untouched] != 0) for name in
                    ("plastic_slip_m", "cumulative_slip_m", "max_opening_m", "damage", "traction_pa",
                     "friction_work_j", "viscous_work_j", "fracture_work_j", "shear_remainder_j")):
                raise ValueError("Permanently tied contact material contains invented history")
        try:
            self.moving_model._geometry_check(self.mesh_for(state), state.radius_m, state.elastic_strain)
        except MovingTiedRetry as exc:
            raise ValueError(str(exc)) from exc
        relative_q = np.r_[np.zeros(self.nparent), state.enrichment_m]
        geometry, _ = self._geometry(basis, state.last_volume_m3)
        if (maximum_total_strain(basis.strain(relative_q)) > self.parameters.max_added_strain
                or maximum_total_strain(basis.geometric_strain(relative_q)) > self.parameters.max_added_strain
                or np.max(np.abs(jumps)/np.repeat(geometry.edge_length_m, 2)[:, None]) > self.parameters.max_jump_edge_fraction
                or np.max(-jumps[:, 0], initial=0.) > self.parameters.max_penetration_m):
            raise ValueError("Saved moving contact violates its local geometric domain")
        expected_front = np.repeat(np.min((state.last_volume_m3/self.birth_child_area_m2)[basis.topology.seam_faces], axis=1), 2)
        if not np.allclose(self.history_model.front_depth(state.history), expected_front, rtol=2e-12, atol=1e-9):
            raise ValueError("Contact material front disagrees with stored solid volume")
        coefficient = basis.drag_area_m2*self.parameters.basal_drag_pa_s_m/(state.last_step_years*SECONDS_PER_YEAR)
        if not np.allclose(state.last_drag_force_n, coefficient*state.last_increment_current_m, rtol=2e-12, atol=1e-5):
            raise ValueError("Saved drag force disagrees with current-area motion")
        if check_force:
            force = self.force_diagnostics(state)
            if force["relative_residual"] > self.parameters.equilibrium_tolerance*(1+1e-6):
                raise ValueError("Saved moving contact force balance is invalid")
            if abs(force["relative_residual"]-state.equilibrium_residual) > max(1e-14, 1e-5*self.parameters.equilibrium_tolerance):
                raise ValueError("Saved moving contact residual disagrees with material fields")
            if not np.allclose(state.constraint_reaction_n[tied], -force["residual_force_n"][tied], rtol=1e-9, atol=1e-3):
                raise ValueError("Saved tied reactions disagree with material forces")

    def trial(self, state, loading):
        self._validate_state(state)
        if state.stopped_reason:
            raise ValueError("Cannot advance stopped moving contact")
        if not isinstance(loading, MovingContactLoading):
            raise ValueError("Loading must be MovingContactLoading")
        years = _positive(loading.dt_years, "dt_years")
        if state.elapsed_years+years == state.elapsed_years:
            raise ValueError("Time increment cannot be represented")
        count = len(state.last_volume_m3)
        volume = _array(loading.volume_m3, (count,), "volume_m3", positive=True)
        young = _array(loading.young_modulus_pa, (count,), "young_modulus_pa", positive=True)
        memory = _array(loading.memory, (count, 3), "memory")
        beta = _array(loading.effective_b, (count,), "effective_b", positive=True)
        ntrace = len(self.history_model.birth_traction_pa)
        water = _array(loading.water_per_trace, (ntrace,), "water_per_trace")
        if np.any(beta > 1) or np.any((water < 0) | (water > 1)) or not callable(loading.external_force):
            raise ValueError("Invalid Maxwell factor, water fraction or force callback")
        if np.any(volume < state.last_volume_m3*(1-2e-13)):
            raise ValueError("Moving contact remelting is not implemented")
        oldbasis, oldmesh = self.basis_for(state), self.mesh_for(state)
        old_geometry, oldj = self._geometry(oldbasis, state.last_volume_m3)
        oldjump = (oldj[:, self.nparent:]@state.enrichment_m).reshape(-1, 2)
        front = np.repeat(np.min((volume/self.birth_child_area_m2)[oldbasis.topology.seam_faces], axis=1), 2)
        prepared, parameter_energy = self.history_model.prepare(state.history, front,
            oldjump[:, 0], oldjump[:, 1], (state.elapsed_years+years)/1e6)
        dt = years*SECONDS_PER_YEAR
        p = self.parameters

        def evaluate(mesh, radius, q):
            basis = self.support.basis_at(mesh, radius)
            parent_rotation, parent_increment = _kinematics(oldmesh, mesh, state.radius_m, radius)
            parent_current = rotate_tensor(parent_increment, parent_rotation, engineering=True)
            background = basis.subdivision.tensor(parent_current, engineering=True)
            child_rotation = (basis.subdivision.frame_rotation@parent_rotation[self.support.parent_face]
                @oldbasis.subdivision.frame_rotation.transpose(0, 2, 1))
            carried = transport_enrichment(oldbasis, basis, state.enrichment_m, state.radius_m, radius)
            dq = q-carried
            relative = np.asarray(basis.strain_operator[:, self.nparent:]@dq).reshape(-1, 3)/radius
            increment = background+relative
            current_memory = rotate_tensor(memory, child_rotation, engineering=True)
            elastic = current_memory+beta[:, None]*increment
            elasticity = young[:, None, None]*basis.membrane.d
            stress = np.einsum("fij,fj->fi", elasticity, elastic)
            _, internal = basis.bulk(volume, elasticity, elastic)
            geometry, j = self._geometry(basis, volume)
            jump = (j[:, self.nparent:]@q).reshape(-1, 2)
            history, traction, tangent = self.history_model.evaluate(prepared,
                jump[:, 0], jump[:, 1], dt, water)
            contact = np.asarray(j.T@traction.ravel()).ravel()
            external = self._external(basis, volume, loading.external_force)
            parent_motion, _ = _current_increment(oldmesh, mesh, state.radius_m, radius, basis.parent_membrane)
            motion = np.r_[parent_motion, dq]
            coefficient = basis.drag_area_m2*p.basal_drag_pa_s_m/dt
            drag = coefficient*motion
            residual, ratio, scale, _, _ = self._residual(basis, internal, contact, drag,
                external, young, volume, coefficient)
            return dict(basis=basis, geometry=geometry, j=j, q=q, jump=jump, history=history,
                tangent=tangent, traction=traction, increment=increment, memory=current_memory,
                elastic=elastic, stress=stress, elasticity=elasticity, motion=motion,
                coefficient=coefficient, drag=drag, external=external, residual=residual, ratio=ratio, scale=scale)

        mesh, radius, q = oldmesh, state.radius_m, state.enrichment_m.copy()
        response = evaluate(mesh, radius, q)
        for _ in range(p.max_newton_iterations):
            if response["ratio"] <= p.equilibrium_tolerance:
                break
            b, geometry, j = response["basis"], response["geometry"], response["j"]
            stiffness, _ = b.bulk(volume, response["elasticity"]*beta[:, None, None], np.zeros_like(memory))
            tangent = sparse.coo_matrix((response["tangent"].ravel(),
                (geometry.contact_rows, geometry.contact_columns)), shape=(j.shape[0], j.shape[0])).tocsr()
            matrix = (stiffness+j.T@tangent@j+sparse.diags(response["coefficient"]))[self.free][:, self.free].tocsr()
            scaling = 1/np.sqrt(np.maximum(np.abs(matrix.diagonal()), 1.))
            diagonal = sparse.diags(scaling)
            solved = scaling*spsolve(diagonal@matrix@diagonal, -scaling*response["residual"][self.free])
            if not np.isfinite(solved).all():
                raise MovingContactRetry("moving_contact_linear_solve_limit")
            correction = np.zeros(self.ndof)
            correction[self.free] = solved
            for power in range(18):
                factor = .5**power
                try:
                    newmesh = move_mesh(mesh, b.parent_membrane.vertex_basis,
                        correction[:self.nparent-1].reshape(-1, 2)*factor/radius)
                    newradius = radius+factor*correction[self.nparent-1]
                    newbasis = self.support.basis_at(newmesh, newradius)
                    newq = transport_enrichment(b, newbasis, q+factor*correction[self.nparent:], radius, newradius)
                    trial = evaluate(newmesh, newradius, newq)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if trial["ratio"] < response["ratio"] or trial["ratio"] <= p.equilibrium_tolerance:
                    mesh, radius, q, response = newmesh, newradius, newq, trial
                    break
            else:
                raise MovingContactRetry("moving_contact_equilibrium_limit")
        if response["ratio"] > p.equilibrium_tolerance:
            raise MovingContactRetry("moving_contact_equilibrium_limit")
        r, b = response, response["basis"]
        try:
            self.moving_model._geometry_check(mesh, radius, r["elastic"], old_mesh=oldmesh, old_radius=state.radius_m)
        except MovingTiedRetry as exc:
            raise MovingContactRetry(str(exc)) from exc
        relative_q = np.r_[np.zeros(self.nparent), q]
        jumps = r["jump"]
        if (maximum_total_strain(b.strain(relative_q)) > p.max_added_strain
                or maximum_total_strain(b.geometric_strain(relative_q)) > p.max_added_strain):
            raise MovingContactRetry("moving_contact_local_strain_limit")
        if np.max(np.abs(jumps)/np.repeat(r["geometry"].edge_length_m, 2)[:, None]) > p.max_jump_edge_fraction:
            raise MovingContactRetry("moving_contact_small_sliding_limit")
        if np.max(np.abs(jumps-oldjump)) > p.max_step_jump_m:
            raise MovingContactRetry("moving_contact_jump_step_limit")
        if np.max(-jumps[:, 0], initial=0.) > p.max_penetration_m:
            raise MovingContactRetry("moving_contact_penetration_limit")
        fine = b.W@q
        xyz = b.topology.mesh.vertices+np.einsum("vij,vj->vi", b.membrane.vertex_basis,
            fine[:-1].reshape(-1, 2))/radius
        xyz /= np.linalg.norm(xyz, axis=1)[:, None]
        try:
            rebuild_seam_mesh(b.topology, xyz)
        except ValueError as exc:
            raise MovingContactRetry("moving_contact_split_mesh_limit") from exc
        stress0 = np.einsum("fij,fj->fi", r["elasticity"], r["memory"])
        work = float(np.sum(volume*np.einsum("fi,fi->f", .5*(stress0+r["stress"]), r["increment"])))
        energy_change = .5*float(np.sum(volume*(np.einsum("fi,fi->f", r["elastic"], r["stress"])
            -np.einsum("fi,fi->f", r["memory"], stress0))))
        external_work, drag_work = float(r["external"]@r["motion"]), float(r["drag"]@r["motion"])
        interface_change = self.history_model.energy(r["history"], jumps[:, 0], jumps[:, 1])-self.history_model.energy(state.history, oldjump[:, 0], oldjump[:, 1])
        dissipated = 0.
        for name in ("friction_work_j", "viscous_work_j", "fracture_work_j"):
            dissipated += sum(float(getattr(c, name).sum()) for c in (r["history"].initial, r["history"].added))
            dissipated -= sum(float(getattr(c, name).sum()) for c in (state.history.initial, state.history.added))
        geometric_work = float(np.sum(r["traction"]*(jumps-oldjump))
            -(r["j"].T@r["traction"].ravel())@r["motion"])
        reaction = -r["residual"].copy()
        reaction[self.free] = 0.
        return replace(state, vertices=mesh.vertices.copy(), radius_m=float(radius), enrichment_m=q.copy(),
            elastic_strain=r["elastic"], history=r["history"], elapsed_years=state.elapsed_years+years,
            last_step_years=years, accepted_steps=state.accepted_steps+1,
            continuation_steps=state.continuation_steps+1, equilibrium_residual=r["ratio"],
            external_work_j=state.external_work_j+external_work, drag_work_j=state.drag_work_j+drag_work,
            bulk_work_j=state.bulk_work_j+work,
            bulk_loading_correction_j=state.bulk_loading_correction_j+work-energy_change,
            mechanical_remainder_j=state.mechanical_remainder_j+external_work-work-interface_change+parameter_energy-dissipated-drag_work,
            contact_geometric_work_j=state.contact_geometric_work_j+geometric_work,
            cohort_parameter_energy_j=state.cohort_parameter_energy_j+parameter_energy,
            last_increment_current_m=r["motion"].copy(), last_external_force_n=r["external"].copy(),
            last_drag_force_n=r["drag"].copy(), last_volume_m3=volume, last_young_modulus_pa=young,
            constraint_reaction_n=reaction)

    def save_state(self, path, state):
        self._validate_state(state)
        arrays = {name: getattr(state, name) for name in self._shapes}
        metadata = {item.name: getattr(state, item.name) for item in fields(state)
            if item.name not in self._shapes and item.name != "history"}
        metadata.update(version=VERSION, fingerprint=self.fingerprint)
        for label in ("initial", "added"):
            arrays.update({label+"_"+f.name: getattr(getattr(state.history, label), f.name) for f in fields(CohortState)})
        arrays["metadata"] = np.array(json.dumps(metadata, allow_nan=False, sort_keys=True))
        path = Path(path)
        temporary = Path(str(path)+".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)

    def load_state(self, path):
        from .genesis_moving_contact_cohorts import MovingContactHistory
        names = {label+"_"+f.name for label in ("initial", "added") for f in fields(CohortState)}
        scalars = {f.name for f in fields(MovingContactState)}-set(self._shapes)-{"history"}
        try:
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != set(self._shapes)|names|{"metadata"}:
                    raise ValueError("Moving contact checkpoint has unexpected or missing arrays")
                encoded = data["metadata"]
                if encoded.shape != () or encoded.dtype.kind not in "US":
                    raise ValueError("Moving contact checkpoint metadata must be scalar JSON")
                metadata = json.loads(str(encoded))
                if set(metadata) != scalars|{"version", "fingerprint"}:
                    raise ValueError("Moving contact checkpoint has invalid metadata")
                if metadata.pop("version") != VERSION or metadata.pop("fingerprint") != self.fingerprint:
                    raise ValueError("Moving contact checkpoint belongs to a different owner")
                cohorts = [CohortState(**{f.name: data[label+"_"+f.name].copy() for f in fields(CohortState)})
                    for label in ("initial", "added")]
                state = MovingContactState(**metadata, **{name: data[name].copy() for name in self._shapes},
                    history=MovingContactHistory(*cohorts))
            self._validate_state(state)
            return state
        except (TypeError, KeyError, AttributeError, IndexError) as exc:
            raise ValueError("Malformed moving contact checkpoint") from exc
