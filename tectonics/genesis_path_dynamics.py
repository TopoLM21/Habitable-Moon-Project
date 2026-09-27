"""Physical-time mechanics on a fixed embedded path (research discretization).

The parent background uses assumed strains; the displayed child mesh has a
different compatible strain. Both are guarded. Inserting support leaves every
new jump constrained to zero. ``release`` is an EXPLICIT cohesive intervention,
not a strength/energy criterion or a conversion of diffuse fault history.

Loading is supplied by the caller, so Maxwell memory and thermal contraction
are applied once per accepted interval, never per Newton iteration. The helper
for isothermal loading is a controlled mechanical experiment, not a climate or
orbit integrator. Reference interface depths stay fixed: evolving interfaces
and automatic propagation must be supplied by a later coupled owner.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from numbers import Integral, Real

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .genesis_contact import ContactParameters, SECONDS_PER_YEAR
from .genesis_contact_geometry import build_interface_geometry
from .genesis_contact_growth import (CohortState, append_cohorts, cohort_energy,
    empty_cohorts, evaluate_cohorts, validate_cohorts)
from .genesis_contact_law import ContactLawParameters
from .genesis_crack_path import CrackInterval
from .genesis_seams import rebuild_seam_mesh
from .genesis_shell import maximum_total_strain, maxwell_factors


VERSION = "genesis-path-mechanics-0.1"


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _array(value, shape, name, *, positive=False, fraction=False):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    if positive and np.any(value <= 0):
        raise ValueError(f"{name} must be positive")
    if fraction and np.any((value < 0) | (value > 1)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


@dataclass(frozen=True)
class PathLoading:
    dt_years: float
    volume_m3: np.ndarray
    elasticity: np.ndarray
    memory: np.ndarray
    effective_b: np.ndarray
    external_force: np.ndarray
    water_access: np.ndarray


@dataclass
class PathState:
    elapsed_years: float
    displacement_m: np.ndarray
    elastic_strain: np.ndarray
    cohorts: CohortState
    active_interval: CrackInterval | None
    constraint_reaction_n: np.ndarray
    equilibrium_residual: float = 0.
    released_reaction_norm_n: float = 0.
    released_reaction_measured: bool = False
    drag_work_j: float = 0.
    external_work_j: float = 0.
    bulk_work_j: float = 0.
    bulk_loading_correction_j: float = 0.
    mechanical_remainder_j: float = 0.
    accepted_steps: int = 0
    rejected_steps: int = 0
    last_step_years: float = 0.
    stopped_reason: str | None = None


class _PathRetry(RuntimeError):
    """Rejected trial; no state or constitutive history has been committed."""


class PathMechanics:
    def __init__(self, basis, depth_m, parameters=None, law_parameters=None, *, geometry_parameters=None):
        self.basis = basis
        self.parameters = parameters or ContactParameters()
        self.law_parameters = law_parameters or ContactLawParameters()
        self.parameters.validate()
        self.law_parameters.validate()
        # The default retains the historical guard and checkpoint fingerprint.
        # Opt-in local checks replace only the global displacement/min-edge
        # proxy; they do not change the reference geometry or force operator.
        if geometry_parameters is not None:
            from .genesis_path_geometry import PathGeometryParameters
            if not isinstance(geometry_parameters, PathGeometryParameters):
                raise ValueError("geometry_parameters must be PathGeometryParameters or None")
            geometry_parameters.validate()
        self.geometry_parameters = geometry_parameters
        self.depth_m = _array(depth_m, (basis.membrane.mesh.cell_count,), "depth_m", positive=True).copy()
        self.geometry = build_interface_geometry(basis.subdivision.mesh, basis.topology,
            basis.radius_m, self.depth_m, basis.membrane)
        raw_jump = (self.geometry.jump_operator@basis.displacement_operator).tocsr()
        # Duplicate bank background interpolation is identical. Avoid tiny
        # floating cancellation injecting contact force into parent DOFs.
        background = raw_jump[:, :basis.nparent]
        if background.nnz and np.max(np.abs(background.data)) > 2e-12:
            raise ValueError("Embedded background does not tie paired banks")
        self.jump_operator = sparse.hstack((sparse.csr_matrix(background.shape),
            raw_jump[:, basis.nparent:]), format="csr")
        self.reference_volume_m3 = basis.subdivision.mesh.areas_unit_sphere*basis.radius_m**2*self.depth_m
        self.trace_depth_m = np.repeat(np.min(self.depth_m[basis.topology.seam_faces], axis=1), 2)
        edges = np.asarray(basis.subdivision.mesh.shared_edges)[:, 2:]
        xyz = basis.subdivision.mesh.vertices[edges]
        lengths = np.arctan2(np.linalg.norm(np.cross(xyz[:, 0], xyz[:, 1]), axis=1),
                            np.einsum("ij,ij->i", xyz[:, 0], xyz[:, 1]))*basis.radius_m
        self.shortest_edge_m = float(lengths.min())
        digest = hashlib.sha256()
        digest.update(json.dumps({"version": VERSION, "radius_m": basis.radius_m,
            "parameters": asdict(self.parameters), "law": asdict(self.law_parameters)}, sort_keys=True).encode())
        if geometry_parameters is not None:
            digest.update(json.dumps({"local_geometry_policy": asdict(geometry_parameters)}, sort_keys=True).encode())
        for value in (basis.topology.mesh.vertices, basis.topology.mesh.faces, self.depth_m,
                      basis.subdivision.parent_mesh.vertices, basis.subdivision.parent_mesh.faces,
                      basis.subdivision.parent_face, basis.parent_membrane.d, basis.membrane.d,
                      basis.insertion.path.points_xyz, basis.insertion.path.arclength_m,
                      basis.insertion.path_vertex_ids, basis.insertion.path_arclength_m,
                      basis.topology.parent_vertex, basis.topology.cut_edges,
                      basis.topology.seam_faces, basis.topology.bank_vertices,
                      basis.strain_operator.data, basis.strain_operator.indices,
                      basis.strain_operator.indptr, basis.displacement_operator.data,
                      basis.displacement_operator.indices, basis.displacement_operator.indptr,
                      self.jump_operator.data, self.jump_operator.indices, self.jump_operator.indptr,
                      basis.drag_area_m2):
            value = np.ascontiguousarray(value)
            digest.update(str((value.dtype.str, value.shape)).encode())
            digest.update(value.tobytes())
        self.fingerprint = digest.hexdigest()

    def initial(self, elastic_strain, time_years=0.):
        n = self.basis.subdivision.mesh.cell_count
        elastic = _array(elastic_strain, (n, 3), "elastic_strain").copy()
        if not np.isfinite(time_years) or time_years < 0:
            raise ValueError("Initial time must be finite and nonnegative")
        count = len(self.trace_depth_m)
        cohorts = append_cohorts(empty_cohorts(), np.arange(count), np.zeros(count), self.trace_depth_m,
            np.repeat(self.geometry.edge_length_m, 2), np.zeros(count), np.zeros(count),
            float(time_years)/1e6, self.law_parameters, 1e-9)
        return PathState(float(time_years), np.zeros(self.basis.ndof), elastic, cohorts, None,
                         np.zeros(self.basis.ndof))

    def release(self, state, interval):
        """Explicitly release previously tied jumps, with no fabricated work.

        A fresh zero-gap cohesive surface has zero traction. Released reactions
        are reported; no invented force is added to cancel them. The subsequent
        motion is an imposed-compliance transient, not evidence of nucleation.
        """
        self._validate_state(state)
        if state.stopped_reason:
            raise ValueError("Cannot release a stopped mechanics state")
        if not isinstance(interval, CrackInterval):
            raise ValueError("Release requires an explicit CrackInterval")
        old_free = self.basis.free_dofs(state.active_interval)
        new_free = self.basis.free_dofs(interval)
        if (state.active_interval is not None and
                (interval.left_m > state.active_interval.left_m or interval.right_m < state.active_interval.right_m)):
            raise ValueError("An activated interval cannot shrink or heal")
        if not np.all(np.isin(old_free, new_free)):
            raise ValueError("Release requires a nested, resolved active interval")
        selected = np.setdiff1d(new_free, old_free)
        if not len(selected):
            raise ValueError("Interval does not release any independent bank motion")
        reaction = state.constraint_reaction_n.copy()
        measured = state.accepted_steps > 0
        removed = float(np.linalg.norm(reaction[selected])) if measured else 0.
        reaction[new_free] = 0.
        return replace(state, active_interval=interval, constraint_reaction_n=reaction,
            released_reaction_norm_n=removed, released_reaction_measured=measured)

    def _validate_state(self, state):
        if not isinstance(state, PathState):
            raise ValueError("Path mechanics state must be a PathState")
        b = self.basis
        arrays = {"displacement_m": (b.ndof,),
                  "elastic_strain": (b.subdivision.mesh.cell_count, 3),
                  "constraint_reaction_n": (b.ndof,)}
        for name, shape in arrays.items():
            value = getattr(state, name)
            if not isinstance(value, np.ndarray) or value.dtype.kind not in "fiu":
                raise ValueError(f"State {name} must be a real numeric array")
            _array(value, shape, name)
        q, reaction = state.displacement_m, state.constraint_reaction_n
        counts = {"accepted_steps", "rejected_steps"}
        nonnegative = {"elapsed_years", "equilibrium_residual", "released_reaction_norm_n",
                       "drag_work_j", "last_step_years"}
        for item in fields(state):
            name, value = item.name, getattr(state, item.name)
            if name in arrays or name in {"cohorts", "active_interval", "stopped_reason", "released_reaction_measured"}:
                continue
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value)):
                raise ValueError(f"Invalid state {name}")
            if name in counts and (not isinstance(value, Integral) or value < 0):
                raise ValueError(f"State {name} must be a nonnegative integer")
            if name in nonnegative and value < 0:
                raise ValueError(f"State {name} must be nonnegative")
        if (not isinstance(state.released_reaction_measured, bool)
                or (state.released_reaction_measured and (state.accepted_steps == 0 or state.active_interval is None))
                or (not state.released_reaction_measured and state.released_reaction_norm_n != 0)):
            raise ValueError("Released reaction must distinguish an accepted measurement from initialization")
        if (state.stopped_reason is not None
                and (not isinstance(state.stopped_reason, str) or not state.stopped_reason)):
            raise ValueError("State stopped_reason must be a nonempty string or None")
        if state.active_interval is not None and not isinstance(state.active_interval, CrackInterval):
            raise ValueError("State active_interval must be a CrackInterval or None")
        free = b.free_dofs(state.active_interval)
        tied = np.setdiff1d(np.arange(b.ndof), free)
        if np.any(q[tied] != 0) or np.any(reaction[free] != 0):
            raise ValueError("Tied jumps or free-DOF reactions are inconsistent")
        self._validate_contact_history(state.cohorts)
        c = state.cohorts
        if (not np.array_equal(c.trace_index, np.arange(len(self.trace_depth_m)))
                or not np.array_equal(c.z_hi_ref_m, self.trace_depth_m)
                or np.any(c.z_lo_ref_m != 0)
                or not np.array_equal(c.area_ref_m2, self.geometry.interface_area_m2)):
            raise ValueError("Checkpoint does not cover the fixed reference interfaces")
        tolerance_years = 128*np.finfo(float).eps*max(abs(state.elapsed_years), 1.)
        if (np.any(c.birth_time_myr > state.elapsed_years/1e6+tolerance_years/1e6)
                or (len(c.birth_time_myr) and np.any(c.birth_time_myr != c.birth_time_myr[0]))
                or np.any(c.birth_gap_m != 0) or np.any(c.birth_jump_m != 0)
                or not np.all(c.bonded)):
            raise ValueError("Contact history disagrees with fixed, initially tied interface birth")
        born_years = float(c.birth_time_myr[0])*1e6 if len(c.birth_time_myr) else 0.
        if state.last_step_years > state.elapsed_years-born_years+tolerance_years:
            raise ValueError("Last mechanics step exceeds the elapsed history")
        if ((state.accepted_steps == 0 and state.last_step_years != 0)
                or (state.accepted_steps > 0 and state.last_step_years <= 0)):
            raise ValueError("Last mechanics step disagrees with the accepted step count")
        if state.equilibrium_residual > self.parameters.equilibrium_tolerance:
            raise ValueError("Accepted mechanics residual exceeds its solver tolerance")
        gap, jump = (self.jump_operator@q).reshape(-1, 2).T
        relative_opening = np.maximum(gap-c.birth_gap_m, 0)
        if np.any(relative_opening > c.max_opening_m+1e-10*np.maximum(c.max_opening_m, 1.)):
            raise ValueError("Current opening exceeds irreversible cohesive history")
        expected = self._stored_contact_traction(c, gap, jump)
        if not np.allclose(c.traction_pa, expected, rtol=1e-10, atol=1e-5):
            raise ValueError("Stored cohesive traction disagrees with current displacement and history")
        # In this monotone, fixed-support owner these endpoint traces have
        # never had an independent displacement jump. They cannot carry a
        # fabricated plastic/fracture history, even if locally self-consistent.
        observed = np.asarray(self.jump_operator[:, free].power(2).sum(axis=1)).ravel()
        untouched = observed.reshape(-1, 2).sum(axis=1) == 0
        if any(np.any(getattr(c, name)[untouched] != 0) for name in
                ("plastic_slip_m", "cumulative_slip_m", "max_opening_m", "damage",
                 "traction_pa", "friction_work_j", "viscous_work_j",
                 "fracture_work_j", "shear_remainder_j")):
            raise ValueError("Permanently tied contact traces must retain virgin history")
        if state.accepted_steps == 0:
            if (abs(state.elapsed_years-born_years) > tolerance_years
                    or np.any(q != 0) or np.any(reaction != 0)
                    or any(getattr(state, name) != 0 for name in
                        ("equilibrium_residual", "released_reaction_norm_n", "drag_work_j",
                         "external_work_j", "bulk_work_j", "bulk_loading_correction_j", "mechanical_remainder_j"))
                    or any(np.any(getattr(c, name) != 0) for name in
                        ("plastic_slip_m", "cumulative_slip_m", "max_opening_m", "damage",
                         "friction_work_j", "viscous_work_j", "fracture_work_j", "shear_remainder_j"))):
                raise ValueError("Unadvanced mechanics state contains invented displacement or work history")

    def _validate_contact_history(self, cohorts):
        validate_cohorts(cohorts, self.law_parameters, len(self.trace_depth_m))

    def _stored_contact_traction(self, cohorts, gap, jump):
        c, law = cohorts, self.law_parameters
        normal = law.normal_stiffness_pa_m*(np.minimum(gap, 0)
            +(1-c.damage)*np.maximum(gap-c.birth_gap_m, 0))
        shear = law.tangential_stiffness_pa_m*(jump-c.birth_jump_m-c.plastic_slip_m)
        free_contact = (gap >= 0) & (c.damage >= 1)
        if not np.allclose(c.plastic_slip_m[free_contact],
                (jump-c.birth_jump_m)[free_contact], rtol=1e-10, atol=1e-10):
            raise ValueError("Free-open contact has an inconsistent plastic reference")
        shear[free_contact] = 0.
        return np.column_stack((normal, shear))

    def _evaluate_contact(self, cohorts, gap, jump, dt, water):
        return evaluate_cohorts(cohorts, gap, jump, dt, water, self.law_parameters)

    def _contact_energy(self, cohorts, gap, jump):
        return cohort_energy(cohorts, gap, jump, self.law_parameters)

    def _validate_loading(self, loading, active):
        b, n = self.basis, self.basis.subdivision.mesh.cell_count
        _positive(loading.dt_years, "dt_years")
        volume = _array(loading.volume_m3, (n,), "volume_m3", positive=True)
        c = _array(loading.elasticity, (n, 3, 3), "elasticity")
        if not np.allclose(c, c.transpose(0, 2, 1), rtol=1e-13, atol=0) or np.any(np.linalg.eigvalsh(c) <= 0):
            raise ValueError("Elasticity must be symmetric positive definite")
        _array(loading.memory, (n, 3), "memory")
        factors = _array(loading.effective_b, (n,), "effective_b", positive=True)
        if np.any(factors > 1):
            raise ValueError("effective_b must not exceed one")
        _array(loading.external_force, (b.ndof,), "external_force")
        _array(loading.water_access, (n,), "water_access", fraction=True)
        if active and not np.allclose(volume, self.reference_volume_m3, rtol=1e-10, atol=0):
            raise ValueError("Released-path interface growth/remelting is not implemented")

    def isothermal_loading(self, state, dt_years, volume_m3, elasticity,
                           viscosity_pa_s, young_modulus_pa, external_force, water_access):
        """Exact Maxwell factors for a fixed-temperature, fixed-phase interval."""
        dt = _positive(dt_years, "dt_years")*SECONDS_PER_YEAR
        young = _positive(young_modulus_pa, "young_modulus_pa")
        eta = _array(viscosity_pa_s, (len(state.elastic_strain),), "viscosity_pa_s", positive=True)
        r, beta = maxwell_factors(dt, eta/young)
        return PathLoading(float(dt_years), volume_m3, elasticity,
            r[:, None]*state.elastic_strain, beta, external_force, water_access)

    def geometry_metrics(self, state, *, include_local=True):
        b, q = self.basis, state.displacement_m
        fine = b.displacement_operator@q
        jumps = (self.jump_operator@q).reshape(-1, 2)
        metrics = {"constitutive_added_strain": maximum_total_strain(b.strain(q)),
            "geometric_added_strain": maximum_total_strain(b.geometric_strain(q)),
            "elastic_strain": maximum_total_strain(state.elastic_strain),
            "motion_edge_fraction": float(np.linalg.norm(fine[:-1].reshape(-1, 2), axis=1).max())/self.shortest_edge_m,
            "radial_fraction": abs(float(fine[-1]))/b.radius_m,
            "max_opening_m": float(np.max(jumps[:, 0], initial=0)),
            "max_slip_m": float(np.max(np.abs(jumps[:, 1]), initial=0)),
            "penetration_m": float(np.max(-jumps[:, 0], initial=0)),
            "jump_edge_fraction": float(np.max(np.abs(jumps)/np.repeat(self.geometry.edge_length_m, 2)[:, None], initial=0))}
        if self.geometry_parameters is not None and include_local:
            from .genesis_path_geometry import local_geometry_metrics
            metrics.update(local_geometry_metrics(self, state))
        return metrics

    def _check_geometry(self, state, old):
        p, b = self.parameters, self.basis
        m = self.geometry_metrics(state, include_local=False)
        if max(m["constitutive_added_strain"], m["geometric_added_strain"], m["radial_fraction"]) > p.max_added_strain:
            raise _PathRetry("path_reference_strain_limit")
        if m["elastic_strain"] > .03:
            raise _PathRetry("path_elastic_limit")
        if self.geometry_parameters is None and m["motion_edge_fraction"] > p.max_motion_edge_fraction:
            raise _PathRetry("path_reference_motion_limit")
        if m["jump_edge_fraction"] > p.max_jump_edge_fraction:
            raise _PathRetry("path_small_sliding_limit")
        increment = self.jump_operator@(state.displacement_m-old.displacement_m)
        if np.max(np.abs(increment), initial=0) > p.max_step_jump_m:
            raise _PathRetry("path_jump_step_limit")
        fine = b.displacement_operator@state.displacement_m
        xyz = b.topology.mesh.vertices+np.einsum("vij,vj->vi", b.membrane.vertex_basis,
            fine[:-1].reshape(-1, 2))/b.radius_m
        xyz /= np.linalg.norm(xyz, axis=1)[:, None]
        try:
            rebuild_seam_mesh(b.topology, xyz)
        except ValueError as exc:
            raise _PathRetry("path_moved_mesh_limit") from exc
        banks = xyz[b.topology.bank_vertices]
        diff = (banks[:, 1]-banks[:, 0]).reshape(-1, 3)*(b.radius_m+fine[-1])
        gap = np.einsum("ij,ij->i", diff, self.geometry.interface_normal)
        if max(m["penetration_m"], float(np.max(-gap, initial=0))) > p.max_penetration_m:
            raise _PathRetry("path_penetration_limit")
        if self.geometry_parameters is not None:
            from .genesis_path_geometry import local_geometry_metrics
            try:
                local = local_geometry_metrics(self, state)
            except ValueError as exc:
                raise _PathRetry("path_local_geometry_invalid") from exc
            limits = self.geometry_parameters
            checks = (("gradient_norm", limits.max_displacement_gradient, "gradient"),
                ("material_rotation_rad", limits.max_material_rotation_rad, "rotation"),
                ("tangent_motion_radius_fraction", limits.max_tangent_motion_radius_fraction, "tangent_motion"),
                ("finite_green_strain", p.max_added_strain, "finite_strain"),
                ("linear_strain_error", limits.max_linear_strain_error, "linearization"),
                ("relative_contact_jump_error", limits.max_relative_contact_jump_error, "contact_frame"),
                ("bank_frame_angular_mismatch", 2*limits.max_material_rotation_rad, "bank_frame"),
                ("geometric_penetration_m", p.max_penetration_m, "penetration"))
            for name, bound, reason in checks:
                if local[name] > bound:
                    raise _PathRetry("path_local_"+reason+"_limit")

    def trial(self, state, loading):
        self._validate_state(state)
        self._validate_loading(loading, state.active_interval is not None)
        if state.stopped_reason:
            raise ValueError("Cannot advance a stopped mechanics state")
        b, p = self.basis, self.parameters
        dt = loading.dt_years*SECONDS_PER_YEAR
        matrix, _ = b.bulk(loading.volume_m3,
            loading.elasticity*loading.effective_b[:, None, None], np.zeros_like(loading.memory))
        _, initial_force = b.bulk(loading.volume_m3, loading.elasticity, loading.memory)
        initial_stress = np.einsum("fij,fj->fi", loading.elasticity, loading.memory)
        drag = b.drag_area_m2*p.basal_drag_pa_s_m/dt
        free = b.free_dofs(state.active_interval)
        water = np.repeat(np.mean(loading.water_access[b.topology.seam_faces], axis=1), 2)
        j = self.jump_operator
        floor = float(np.max(np.abs(loading.elasticity)))*float(loading.volume_m3.sum())/b.radius_m*1e-10

        def evaluate(q):
            jump = (j@q).reshape(-1, 2)
            cohorts, force, tangent = self._evaluate_contact(state.cohorts, jump[:, 0], jump[:, 1], dt, water)
            contact = j.T@force.ravel()
            bulk = initial_force+matrix@(q-state.displacement_m)
            residual = bulk+contact+drag*(q-state.displacement_m)-loading.external_force
            scale = max(np.linalg.norm(bulk[free]), np.linalg.norm(contact[free]),
                np.linalg.norm(loading.external_force[free]), floor)
            return cohorts, tangent, residual, float(np.linalg.norm(residual[free])/scale)

        q = state.displacement_m.copy()
        response = evaluate(q)
        for _ in range(p.max_newton_iterations):
            if response[-1] <= p.equilibrium_tolerance:
                break
            tangent = sparse.coo_matrix((response[1].ravel(),
                (self.geometry.contact_rows, self.geometry.contact_columns)),
                shape=(j.shape[0], j.shape[0])).tocsr()
            jacobian = (matrix+j.T@tangent@j+sparse.diags(drag))[free][:, free].tocsr()
            scaling = 1/np.sqrt(np.maximum(np.abs(jacobian.diagonal()), 1.))
            diagonal = sparse.diags(scaling)
            correction = scaling*spsolve(diagonal@jacobian@diagonal, -scaling*response[2][free])
            if not np.isfinite(correction).all():
                raise _PathRetry("path_linear_solve_limit")
            for power in range(18):
                candidate = q.copy()
                candidate[free] += correction*.5**power
                trial = evaluate(candidate)
                if trial[-1] < response[-1] or trial[-1] <= p.equilibrium_tolerance:
                    q, response = candidate, trial
                    break
            else:
                raise _PathRetry("path_equilibrium_limit")
        if response[-1] > p.equilibrium_tolerance:
            raise _PathRetry("path_equilibrium_limit")
        delta = q-state.displacement_m
        inc = b.strain(delta)
        elastic = loading.memory+loading.effective_b[:, None]*inc
        stress = np.einsum("fij,fj->fi", loading.elasticity, elastic)
        work = float(np.sum(np.einsum("fi,fi->f", .5*(initial_stress+stress), inc)*loading.volume_m3))
        energy_change = .5*float(np.sum((np.einsum("fi,fi->f", elastic, stress)
            -np.einsum("fi,fi->f", loading.memory, initial_stress))*loading.volume_m3))
        external_work, drag_work = float(loading.external_force@delta), float(drag@delta**2)
        gap0, jump0 = (j@state.displacement_m).reshape(-1, 2).T
        gap1, jump1 = (j@q).reshape(-1, 2).T
        interface_change = (self._contact_energy(response[0], gap1, jump1)
            -self._contact_energy(state.cohorts, gap0, jump0))
        dissipated = sum(float(np.sum(getattr(response[0], name)-getattr(state.cohorts, name)))
            for name in ("friction_work_j", "viscous_work_j", "fracture_work_j"))
        reaction = -response[2].copy()
        reaction[free] = 0.
        result = replace(state, elapsed_years=state.elapsed_years+loading.dt_years,
            displacement_m=q, elastic_strain=elastic, cohorts=response[0], constraint_reaction_n=reaction,
            equilibrium_residual=response[-1], last_step_years=loading.dt_years,
            accepted_steps=state.accepted_steps+1, external_work_j=state.external_work_j+external_work,
            drag_work_j=state.drag_work_j+drag_work, bulk_work_j=state.bulk_work_j+work,
            bulk_loading_correction_j=state.bulk_loading_correction_j+work-energy_change,
            mechanical_remainder_j=state.mechanical_remainder_j+external_work-work-interface_change-dissipated-drag_work)
        self._check_geometry(result, state)
        return result

    def advance(self, state, target_years, loading_factory, max_step_years=100., min_step_years=.01):
        """Rollback-safe adaptive stepping; callback must be pure and retryable.

        The callback receives (accepted_state, dt_years). It must recompute
        memory for that dt, never reuse loading from a rejected larger step.
        Its thermal/orbit data are externally owned, not saved by this kernel.
        """
        maximum = _positive(max_step_years, "max_step_years")
        minimum = _positive(min_step_years, "min_step_years")
        if minimum > maximum or not np.isfinite(target_years) or target_years < state.elapsed_years:
            raise ValueError("Invalid target or timestep bounds")
        self._validate_state(state)
        dt = maximum
        while state.elapsed_years < target_years and not state.stopped_reason:
            remaining = target_years-state.elapsed_years
            attempted = min(dt, remaining)
            if state.elapsed_years+attempted == state.elapsed_years:
                return replace(state, stopped_reason="path_time_resolution_limit")
            loading = loading_factory(state, attempted)
            if loading.dt_years != attempted:
                raise ValueError("Loading factory changed the requested mechanics interval")
            try:
                candidate = self.trial(state, loading)
            except _PathRetry as exc:
                state = replace(state, rejected_steps=state.rejected_steps+1)
                if attempted <= min(minimum, remaining)*(1+1e-12):
                    return replace(state, stopped_reason=str(exc))
                dt = max(minimum, attempted/2)
            else:
                state = candidate
                # Conservative growth after a successful interval; local error
                # is not estimated, so this is an admissibility controller.
                dt = min(maximum, attempted*1.5)
        return state

    def save_state(self, path, state):
        """Mechanics-only snapshot; owner must separately save loading history."""
        self._validate_state(state)
        arrays, metadata = {}, {"version": VERSION, "fingerprint": self.fingerprint}
        for item in fields(state):
            value = getattr(state, item.name)
            if isinstance(value, np.ndarray):
                arrays[item.name] = value
            elif item.name == "cohorts":
                arrays.update({"cohort_"+f.name: getattr(value, f.name) for f in fields(value)})
            elif item.name == "active_interval":
                metadata[item.name] = None if value is None else asdict(value)
            else:
                metadata[item.name] = value
        arrays["metadata"] = np.array(json.dumps(metadata, allow_nan=False, sort_keys=True))
        np.savez_compressed(path, **arrays)

    def load_state(self, path):
        """Restore only a complete, admissible state of this exact discretization."""
        state_arrays = {"displacement_m", "elastic_strain", "constraint_reaction_n"}
        cohort_arrays = {"cohort_"+item.name for item in fields(CohortState)}
        scalar_names = {item.name for item in fields(PathState)}-state_arrays-{"cohorts"}
        try:
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != state_arrays | cohort_arrays | {"metadata"}:
                    raise ValueError("Path mechanics checkpoint has missing or unexpected arrays")
                encoded = data["metadata"]
                if encoded.shape != () or encoded.dtype.kind not in "US":
                    raise ValueError("Path mechanics metadata must be a scalar JSON string")
                metadata = json.loads(str(encoded))
                if not isinstance(metadata, dict) or set(metadata) != scalar_names | {"version", "fingerprint"}:
                    raise ValueError("Path mechanics checkpoint has missing or unexpected metadata")
                if metadata.pop("version") != VERSION or metadata.pop("fingerprint") != self.fingerprint:
                    raise ValueError("Path mechanics checkpoint belongs to different geometry or parameters")
                arrays = {key: data[key].copy() for key in data.files if key != "metadata"}
            cohorts = CohortState(**{f.name: arrays.pop("cohort_"+f.name) for f in fields(CohortState)})
            interval = metadata.pop("active_interval")
            if interval is not None and (not isinstance(interval, dict) or set(interval) != {"left_m", "right_m"}):
                raise ValueError("Invalid checkpoint active interval")
            state = PathState(**metadata, **arrays, cohorts=cohorts,
                active_interval=None if interval is None else CrackInterval(**interval))
            self._validate_state(state)
            self._check_geometry(state, state)
        except (TypeError, KeyError, AttributeError, _PathRetry) as exc:
            raise ValueError("Malformed or inadmissible path mechanics checkpoint") from exc
        return state
