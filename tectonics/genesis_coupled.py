"""Evolving thermal/orbit/Maxwell shell with persistent split-bank contact.

This bounded reference-geometry model advances physical clocks together. New
cuts inherit material motion and existing contact history. Solidification adds
independent material-depth cohorts; deformation does not create material. Column,
global-thermal and mechanical ledgers are distinct, not a claimed closed shared
energy reservoir. No mature-plate projection is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from numbers import Integral, Real
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .genesis import diagnose, temperatures, SECONDS_PER_MYR
from .genesis_contact import ContactModel, ContactParameters, ContactState, select_seams
from .genesis_contact_law import ContactLawParameters
from .genesis_contact_growth import (CohortState, empty_cohorts, append_cohorts,
    evaluate_cohorts, aggregate_cohorts, cohort_energy, remap_cohorts)
from .genesis_coupled_thermal import advance_thermal_loading, evolve_coupled_damage
from .genesis_fault_law import select_plane
from .genesis_material import material_column_depth
from .genesis_mobile import _RetryStep, _external_force
from .genesis_onset_support import face_velocities
from .genesis_shell import Membrane, mantle_traction, maximum_total_strain
from .genesis_seam_diagnostics import seam_connectivity, cohort_bonded_traces

COUPLED_VERSION = "genesis-coupled-0.2"


@dataclass(frozen=True)
class CoupledParameters:
    max_step_years: float = 100.
    min_step_years: float = .01
    max_interface_area_change_fraction: float = .02
    growth_layer_m: float = 1.
    bonding_gap_tolerance_m: float = 1e-9
    max_cohort_count: int = 100000

    def validate(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid coupled parameter {item.name}")
        if self.min_step_years > self.max_step_years or self.max_interface_area_change_fraction > .05:
            raise ValueError("Coupled step or interface reference limits are invalid")
        if not isinstance(self.max_cohort_count, Integral):
            raise ValueError("Coupled cohort count must be an integer")


@dataclass
class CoupledState:
    time_myr: float
    contact: ContactState
    cohorts: CohortState
    cut_edges: np.ndarray
    interface_birth_area_m2: np.ndarray
    interface_depth_ref_m: np.ndarray
    column_enthalpy: np.ndarray
    elastic_strain: np.ndarray
    damage: np.ndarray
    water_access: np.ndarray
    fault_active: np.ndarray
    plane_normal: np.ndarray
    fault_candidate_age_myr: np.ndarray
    velocity_km_myr: np.ndarray
    tidal_stress_mpa: np.ndarray
    boundary_energy_j: float
    tidal_heat_received_j: float
    initial_column_energy_j: float
    accepted_steps: int = 0
    rejected_steps: int = 0
    new_seam_count: int = 0
    last_step_years: float = 0.
    interface_area_change_fraction: float = 0.
    maxwell_relaxation_release_j: float = 0.
    bulk_parameter_energy_change_j: float = 0.
    thermal_eigenstrain_energy_change_j: float = 0.
    bulk_increment_work_j: float = 0.
    bulk_increment_energy_loss_j: float = 0.
    mechanical_energy_remainder_j: float = 0.
    interface_birth_energy_j: float = 0.
    stopped_reason: str | None = None


class CoupledModel:
    @classmethod
    def from_fault_checkpoint(cls, path, parameters=None, contact_parameters=None, law_parameters=None):
        model = cls(Path(path).read_bytes(), parameters, contact_parameters, law_parameters, str(Path(path).resolve()))
        state = model.initial()
        return model, state, model.source.thermal_state, model.source.orbit

    def __init__(self, source_bytes, parameters=None, contact_parameters=None, law_parameters=None, source_path="embedded"):
        self.parameters = parameters or CoupledParameters()
        self.parameters.validate()
        self.contact_parameters = contact_parameters or ContactParameters()
        self.law_parameters = law_parameters or ContactLawParameters()
        # Empty-cut assembly also permits a continuous load-bearing shell to
        # enter this mode before its first actual crack has formed.
        self.source = ContactModel(source_bytes, self.contact_parameters, self.law_parameters, source_path,
                                   reference_cuts=np.empty((0, 2), dtype=np.int64))
        self.source_model = self.source.source_model
        self.source_bytes, self.source_path = bytes(source_bytes), source_path
        self.source_hash = self.source.source_hash
        self.source_time_myr = self.source.source_time_myr
        self.radius_m = self.source.radius_m
        self.layer_mass_kg = self.source.layer_mass_kg.copy()
        self.original_mesh = self.source_model.mesh_for(self.source.source_state)
        self.original_membrane = Membrane(self.original_mesh, self.source_model.p.poisson_ratio)
        self.reference_column_depth_m = 1000*material_column_depth(self.original_mesh, self.radius_m/1000,
            self.layer_mass_kg, self.source_model.p.density_kg_m3)
        self._geometry_cache = {b"": self.source}

    def _geometry(self, cuts):
        key = np.asarray(cuts, dtype=np.int64).tobytes()
        if key not in self._geometry_cache:
            value = ContactModel(self.source_bytes, self.contact_parameters, self.law_parameters,
                                 self.source_path, reference_cuts=cuts)
            if len(self._geometry_cache) > 3:
                self._geometry_cache.clear()
            self._geometry_cache[key] = value
        return self._geometry_cache[key]

    def initial(self):
        source = self.source.source_state
        cuts = select_seams(self.original_mesh, source, self.contact_parameters)
        geometry = self._geometry(cuts)
        tm, ts = temperatures(np.asarray(self.source.thermal_state.energy), self.source_model.thermal)
        fraction, _, _ = self.source_model._phase(source.column_enthalpy, ts, tm)
        depth = self._reference_front(geometry, fraction)
        if len(depth) > self.parameters.max_cohort_count:
            raise ValueError("Initial contact exceeds coupled max_cohort_count")
        contact = geometry.initial()
        gap, jump = (geometry.jump_operator@contact.displacement_m).reshape(-1, 2).T
        cohorts = append_cohorts(empty_cohorts(), np.arange(len(depth)), np.zeros_like(depth), depth,
            np.repeat(geometry.edge_length_m, 2), gap, jump, source.time_myr, self.law_parameters,
            self.parameters.bonding_gap_tolerance_m)
        contact = replace(contact, **aggregate_cohorts(cohorts, len(depth)))
        return CoupledState(source.time_myr, contact, cohorts, geometry.topology.cut_edges.copy(),
            np.repeat(geometry.edge_length_m, 2)*depth/2, depth, source.column_enthalpy.copy(), source.elastic_strain.copy(),
            source.damage.copy(), source.water_access.copy(), source.fault_active.copy(), source.plane_normal.copy(),
            source.fault_candidate_age_myr.copy(), np.zeros_like(source.velocity_km_myr),
            source.tidal_stress_mpa.copy(), source.boundary_energy_j, source.tidal_heat_received_j,
            source.initial_column_energy_j)

    def _reference_front(self, geometry, fraction):
        """Solid material coordinate, unaffected by displacement or column area."""
        return np.repeat(np.min((fraction*self.reference_column_depth_m)[geometry.topology.seam_faces], axis=1), 2)

    def _interface_area_change(self, geometry, fraction, column_depth_km):
        reference = self._reference_front(geometry, fraction)
        actual = np.repeat(np.min((fraction*column_depth_km*1000)[geometry.topology.seam_faces], axis=1), 2)
        return float(np.max(np.abs(actual/reference-1))) if len(reference) else 0.

    def _grow_interfaces(self, state, fraction, birth_time_myr):
        """Activate a thermally predicted layer at the old mechanical geometry.

        This thermal-first split is first order in time. Pending material less
        than growth_layer_m is resolved on the next front increment, not merged
        into a cohort with an existing nonlinear constitutive history.
        """
        geometry = self._geometry(state.cut_edges)
        front = self._reference_front(geometry, fraction)
        pending = front-state.interface_depth_ref_m
        if np.any(pending < -1e-8):
            raise _RetryStep("coupled_interface_remelting_limit")
        selected = np.flatnonzero(pending >= self.parameters.growth_layer_m)
        if not len(selected):
            return state
        if len(state.cohorts.trace_index)+len(selected) > self.parameters.max_cohort_count:
            raise _RetryStep("coupled_cohort_count_limit")
        gap, jump = (geometry.jump_operator@state.contact.displacement_m).reshape(-1, 2).T
        cohorts = append_cohorts(state.cohorts, selected, state.interface_depth_ref_m[selected], front[selected],
            np.repeat(geometry.edge_length_m, 2), gap, jump, birth_time_myr, self.law_parameters,
            self.parameters.bonding_gap_tolerance_m)
        depth = state.interface_depth_ref_m.copy()
        depth[selected] = front[selected]
        born_energy = .5*self.law_parameters.normal_stiffness_pa_m*float(np.dot(
            cohorts.area_ref_m2[-len(selected):], np.minimum(gap[selected], 0)**2))
        return replace(state, cohorts=cohorts, interface_depth_ref_m=depth,
            interface_birth_area_m2=np.repeat(geometry.edge_length_m, 2)*depth/2,
            contact=replace(state.contact, **aggregate_cohorts(cohorts, len(front))),
            interface_birth_energy_j=state.interface_birth_energy_j+born_energy)

    def mesh_for(self, state):
        return self._geometry(state.cut_edges).mesh_for(state.contact)

    def _phase_fields(self, state, thermal):
        mesh = self.mesh_for(state)
        radius = (self.radius_m+state.contact.displacement_m[-1])/1000.
        tm, ts = temperatures(np.asarray(thermal.energy), self.source_model.thermal)
        fraction, mean, temp = self.source_model._phase(state.column_enthalpy, ts, tm)
        depth = material_column_depth(mesh, radius, self.layer_mass_kg, self.source_model.p.density_kg_m3)
        return mesh, radius, fraction, mean, temp, depth

    def _grow_cuts(self, state, thermal):
        """Only duplicate existing material corners; never change cell masses."""
        fp = self.source_model.fault_p
        active, normals = state.fault_active.copy(), state.plane_normal.copy()
        activate = (~active & (state.damage >= fp.activation_damage)
                    & (state.fault_candidate_age_myr >= fp.activation_persistence_myr))
        if fp.enabled and np.any(activate):
            p = self.source_model.p
            degradation = p.residual_stiffness+(1-p.residual_stiffness)*(1-state.damage)**2
            stress = (state.elastic_strain@self.original_membrane.d.T)*(p.young_modulus_pa*degradation[:, None])
            friction = fp.friction_dry+(fp.friction_wet-fp.friction_dry)*state.water_access
            normals[activate] = select_plane(stress[activate], friction[activate])
            active[activate] = True
        proxy = SimpleNamespace(fault_active=active, plane_normal=normals, damage=state.damage)
        candidates = select_seams(self.original_mesh, proxy, self.contact_parameters)
        cuts = np.unique(np.concatenate((state.cut_edges, candidates)), axis=0)
        state = replace(state, fault_active=active, plane_normal=normals)
        if np.array_equal(cuts, state.cut_edges):
            return state
        from .genesis_coupled_topology import transfer_contact_history
        old, new = self._geometry(state.cut_edges), self._geometry(cuts)
        history = transfer_contact_history(old, new, state.contact)
        _, _, fraction, _, _, _ = self._phase_fields(state, thermal)
        depth = self._reference_front(new, fraction)
        old_indices = {tuple(edge): i for i, edge in enumerate(old.topology.cut_edges)}
        mapping = np.empty(2*len(old.topology.cut_edges), dtype=np.int64)
        selected = []
        for j, edge in enumerate(new.topology.cut_edges):
            if tuple(edge) in old_indices:
                i = old_indices[tuple(edge)]
                depth[2*j:2*j+2] = state.interface_depth_ref_m[2*i:2*i+2]
                mapping[2*i:2*i+2] = [2*j, 2*j+1]
            else:
                selected.extend([2*j, 2*j+1])
        selected = np.asarray(selected, dtype=np.int64)
        if len(state.cohorts.trace_index)+len(selected) > self.parameters.max_cohort_count:
            raise _RetryStep("coupled_cohort_count_limit")
        cohorts = remap_cohorts(state.cohorts, mapping)
        gap, jump = (new.jump_operator@history.displacement_m).reshape(-1, 2).T
        cohorts = append_cohorts(cohorts, selected, np.zeros(len(selected)), depth[selected],
            np.repeat(new.edge_length_m, 2), gap, jump, state.time_myr, self.law_parameters,
            self.parameters.bonding_gap_tolerance_m)
        # New coincident traces have zero birth energy; compute explicitly so
        # numerical compression cannot disappear from the property ledger.
        born_energy = .5*self.law_parameters.normal_stiffness_pa_m*float(np.dot(
            cohorts.area_ref_m2[-len(selected):], np.minimum(gap[selected], 0)**2))
        history = replace(history, **aggregate_cohorts(cohorts, len(depth)))
        return replace(state, contact=history, cohorts=cohorts, cut_edges=new.topology.cut_edges.copy(),
                       interface_birth_area_m2=np.repeat(new.edge_length_m, 2)*depth/2,
                       interface_depth_ref_m=depth, interface_birth_energy_j=state.interface_birth_energy_j+born_energy,
                       new_seam_count=state.new_seam_count+len(cuts)-len(old.topology.cut_edges))

    def _bulk_energy(self, elastic, damage, fraction):
        p = self.source_model.p
        factor = p.young_modulus_pa*(p.residual_stiffness+(1-p.residual_stiffness)*(1-damage)**2)
        volume = self.layer_mass_kg.sum(axis=1)*fraction/p.density_kg_m3
        return .5*float(np.sum(np.einsum("fi,ij,fj->f", elastic, self.original_membrane.d, elastic)*factor*volume))

    def _interface_energy(self, state, geometry):
        gap, jump = (geometry.jump_operator@state.contact.displacement_m).reshape(-1, 2).T
        return cohort_energy(state.cohorts, gap, jump, self.law_parameters)

    def _external(self, geometry, depth_km):
        p = self.source_model.p
        traction = mantle_traction(self.original_mesh, p, depth_km)
        force = _external_force(self.original_membrane, traction, self.radius_m/1000, p.young_modulus_pa)
        force *= p.young_modulus_pa*self.radius_m*1000
        area = geometry.drag_area_m2[0:-1:2]
        parent = geometry.topology.parent_vertex
        total = np.bincount(parent, weights=area, minlength=self.original_mesh.vertex_count)
        out = np.zeros(geometry.membrane.ndof)
        out[:-1] = (force[:-1].reshape(-1, 2)[parent]*(area/total[parent])[:, None]).ravel()
        out[-1] = force[-1]
        return out

    def _solve(self, state, loading, damage):
        geometry = self._geometry(state.cut_edges)
        membrane, p, controls = geometry.membrane, self.source_model.p, self.contact_parameters
        dt_s = loading.dt_myr*SECONDS_PER_MYR
        oldq = state.contact.displacement_m
        volume = self.layer_mass_kg.sum(axis=1)*loading.fraction/p.density_kg_m3
        degradation = p.residual_stiffness+(1-p.residual_stiffness)*(1-damage)**2
        elasticity = (p.young_modulus_pa*degradation)[:, None, None]*membrane.d
        stiffness = elasticity*loading.effective_b[:, None, None]
        local = np.einsum("fai,fab,fbj,f->fij", membrane.b, stiffness, membrane.b, volume/self.radius_m**2)
        matrix = sparse.coo_matrix((local.ravel(), (membrane.rr, membrane.cc)),
                                   shape=(membrane.ndof, membrane.ndof)).tocsr()
        initial_stress = np.einsum("fij,fj->fi", elasticity, loading.memory)
        local_force = np.einsum("fai,fa,f->fi", membrane.b, initial_stress, volume/self.radius_m)
        initial_force = np.zeros(membrane.ndof)
        np.add.at(initial_force, membrane.dofs.ravel(), local_force.ravel())
        external = self._external(geometry, loading.depth_km)
        drag = geometry.drag_area_m2*controls.basal_drag_pa_s_m/dt_s
        water = np.repeat(np.mean(loading.water_access[geometry.topology.seam_faces], axis=1), 2)
        old = state.contact

        def evaluate(q):
            jump = (geometry.jump_operator@q).reshape(-1, 2)
            cohorts, force, tangent = evaluate_cohorts(state.cohorts, jump[:, 0], jump[:, 1], dt_s, water, self.law_parameters)
            contact_force = geometry.jump_operator.T@force.ravel()
            bulk = initial_force+matrix@(q-oldq)
            residual = bulk+contact_force+drag*(q-oldq)-external
            scale = max(np.linalg.norm(bulk), np.linalg.norm(contact_force), np.linalg.norm(external),
                        p.young_modulus_pa*float(volume.sum())/self.radius_m*1e-10)
            return (cohorts, tangent), residual, float(np.linalg.norm(residual)/scale)

        q = oldq.copy()
        response, residual, norm = evaluate(q)
        for _ in range(controls.max_newton_iterations):
            if norm <= controls.equilibrium_tolerance:
                break
            tangent = sparse.coo_matrix((response[1].ravel(),
                (geometry.contact_rows, geometry.contact_columns)),
                shape=(geometry.jump_operator.shape[0], geometry.jump_operator.shape[0])).tocsr()
            jacobian = matrix+geometry.jump_operator.T@tangent@geometry.jump_operator+sparse.diags(drag)
            scaling = 1/np.sqrt(np.maximum(np.abs(jacobian.diagonal()), 1.))
            diagonal = sparse.diags(scaling)
            correction = scaling*spsolve(diagonal@jacobian@diagonal, -scaling*residual)
            if not np.isfinite(correction).all():
                raise _RetryStep("coupled_linear_solve_limit")
            for power in range(18):
                candidate = q+correction*.5**power
                trial = evaluate(candidate)
                if trial[-1] < norm or trial[-1] <= controls.equilibrium_tolerance:
                    q, (response, residual, norm) = candidate, trial
                    break
            else:
                raise _RetryStep("coupled_equilibrium_limit")
        if norm > controls.equilibrium_tolerance:
            raise _RetryStep("coupled_equilibrium_limit")
        delta = q-oldq
        inc = geometry._strain(delta)
        elastic = loading.memory+loading.effective_b[:, None]*inc
        stress = np.einsum("fij,fj->fi", elasticity, elastic)
        if maximum_total_strain(elastic) > self.source_model.mobile_p.max_elastic_strain:
            raise _RetryStep("coupled_elastic_limit")
        history = replace(old, elapsed_years=(loading.thermal_state.time_myr-self.source_time_myr)*1e6,
            displacement_m=q, **aggregate_cohorts(response[0], len(water)),
            drag_work_j=old.drag_work_j+float(np.dot(drag, delta**2)), external_work_j=old.external_work_j+float(external@delta),
            equilibrium_residual=norm, last_step_years=loading.dt_myr*1e6, accepted_steps=old.accepted_steps+1)
        self._check_geometry(geometry, history, old)
        work = float(np.sum(np.einsum("fi,fi->f", .5*(initial_stress+stress), inc)*volume))
        energy_change = self._bulk_energy(elastic, damage, loading.fraction)-self._bulk_energy(loading.memory, damage, loading.fraction)
        return history, elastic, stress, work, work-energy_change, response[0]

    def _geometry_utilization(self, geometry, contact, previous=None, *, geometric_gap_m=None):
        """Measure the actual reference guards without resetting their origin.

        Geometric gaps are supplied after mesh validation. Omitting them gives
        the reference and linear-jump metrics, which can be checked before
        attempting to construct a potentially invalid moved mesh. Per-step
        jump metrics are available only when a previous accepted state exists.
        """
        p = self.contact_parameters
        q = contact.displacement_m
        strain = maximum_total_strain(geometry._strain(q))
        shortest = float(np.min(geometry.edge_length_m)) if len(geometry.edge_length_m) else self.radius_m*.1
        motion = float(np.linalg.norm(q[:-1].reshape(-1, 2), axis=1).max())
        radial = float(abs(q[-1])/self.radius_m)
        metrics = {"max_added_strain": strain, "max_motion_m": motion,
            "motion_reference_length_m": shortest, "max_motion_edge_fraction": motion/shortest,
            "radial_motion_fraction": radial, "added_strain_utilization": strain/p.max_added_strain,
            "motion_utilization": motion/(p.max_motion_edge_fraction*shortest),
            "radial_motion_utilization": radial/p.max_added_strain}
        metrics["reference_geometry_utilization"] = max(metrics[name] for name in
            ("added_strain_utilization", "motion_utilization", "radial_motion_utilization"))
        jump = (geometry.jump_operator@q).reshape(-1, 2)
        maximum = lambda values: float(np.max(values)) if np.size(values) else 0.
        fraction = maximum(np.abs(jump)/np.repeat(geometry.edge_length_m, 2)[:, None])
        linear_penetration = max(0., maximum(-jump[:, 0]))
        metrics.update(max_linear_jump_m=maximum(np.abs(jump)), max_jump_edge_fraction=fraction,
            small_sliding_utilization=fraction/p.max_jump_edge_fraction,
            max_linear_penetration_m=linear_penetration)
        if geometric_gap_m is not None:
            geometric_penetration = max(0., maximum(-np.asarray(geometric_gap_m)))
            penetration = max(linear_penetration, geometric_penetration)
            metrics.update(max_geometric_penetration_m=geometric_penetration,
                max_contact_penetration_m=penetration, penetration_utilization=penetration/p.max_penetration_m)
        if previous is not None:
            old = (geometry.jump_operator@previous.displacement_m).reshape(-1, 2)
            increment = maximum(np.abs(jump-old))
            metrics.update(max_step_jump_m=increment, step_jump_utilization=increment/p.max_step_jump_m)
        return metrics

    def _check_geometry(self, geometry, contact, previous=None):
        metrics = self._geometry_utilization(geometry, contact, previous)
        p = self.contact_parameters
        # Compare the original quantities, preserving strict boundary behavior
        # even if division rounds a just-exceeded utilization back to one.
        if (metrics["max_added_strain"] > p.max_added_strain
                or metrics["max_motion_m"] > p.max_motion_edge_fraction*metrics["motion_reference_length_m"]
                or metrics["radial_motion_fraction"] > p.max_added_strain):
            raise _RetryStep("coupled_reference_geometry_limit")
        try:
            mesh = geometry.mesh_for(contact)
        except ValueError as exc:
            raise _RetryStep("coupled_mesh_quality_limit") from exc
        if len(geometry.edge_length_m):
            actual = geometry._geometric_jump(contact, mesh)
            metrics = self._geometry_utilization(geometry, contact, previous, geometric_gap_m=actual[:, 0])
            if metrics["max_contact_penetration_m"] > p.max_penetration_m:
                raise _RetryStep("coupled_penetration_limit")
            if metrics["max_jump_edge_fraction"] > p.max_jump_edge_fraction:
                raise _RetryStep("coupled_small_sliding_limit")
            if previous is not None and metrics["max_step_jump_m"] > p.max_step_jump_m:
                raise _RetryStep("coupled_jump_step_limit")

    def _trial(self, before, thermal, orbit, target, max_step):
        state = self._grow_cuts(before, thermal)
        geometry = self._geometry(state.cut_edges)
        mesh, radius, old_fraction, _, _, _ = self._phase_fields(state, thermal)
        loading = advance_thermal_loading(self.source_model, mesh=mesh, radius_km=radius,
            layer_mass_kg=self.layer_mass_kg, column_enthalpy=state.column_enthalpy,
            elastic_strain=state.elastic_strain, damage=state.damage, water_access=state.water_access,
            boundary_energy_j=state.boundary_energy_j, thermal_state=thermal, orbit=orbit,
            target_myr=target, max_step_myr=max_step)
        area_change = self._interface_area_change(geometry, loading.fraction, loading.column_depth_km)
        if area_change > self.parameters.max_interface_area_change_fraction:
            raise _RetryStep("coupled_interface_geometry_limit")
        state = self._grow_interfaces(state, loading.fraction, loading.thermal_state.time_myr)
        history, elastic, stress, work, loss, cohorts = self._solve(state, loading, loading.damage0)
        damage, tidal_peak = evolve_coupled_damage(self.source_model, loading, geometry.topology.mesh, radius, stress)
        if np.max(np.abs(damage-loading.damage0)) > self.source_model.mobile_p.max_damage_increment:
            raise _RetryStep("coupled_damage_step_limit")
        if np.max(np.abs(damage-loading.damage0)) > 1e-12:
            history, elastic, stress, work, loss, cohorts = self._solve(state, loading, damage)
        moved = geometry.mesh_for(history)
        next_radius = (self.radius_m+history.displacement_m[-1])/1000
        velocity = face_velocities(mesh.centroids, moved.centroids, .5*(radius+next_radius), loading.dt_myr)
        # Exact accounting of the constitutive predictor's stored-energy
        # changes; these are NOT deposited as heat into overlapping reservoirs.
        e0 = self._bulk_energy(state.elastic_strain, state.damage, old_fraction)
        inherited = loading.retained[:, None]*state.elastic_strain
        e1 = self._bulk_energy(inherited, damage, loading.fraction)
        relaxed = inherited*loading.maxwell_r[:, None]
        e2 = self._bulk_energy(relaxed, damage, loading.fraction)
        e3 = self._bulk_energy(loading.memory, damage, loading.fraction)
        provisional = replace(state, time_myr=loading.thermal_state.time_myr, contact=history, cohorts=cohorts,
            column_enthalpy=loading.column_enthalpy, elastic_strain=elastic, damage=damage,
            water_access=loading.water_access, tidal_stress_mpa=tidal_peak, velocity_km_myr=velocity,
            fault_candidate_age_myr=np.where(damage>=self.source_model.fault_p.activation_damage,
                                             state.fault_candidate_age_myr+loading.dt_myr, 0.),
            boundary_energy_j=loading.boundary_energy_j,
            tidal_heat_received_j=state.tidal_heat_received_j+loading.tidal_heat_received_j,
            accepted_steps=state.accepted_steps+1, last_step_years=loading.dt_myr*1e6,
            interface_area_change_fraction=area_change,
            maxwell_relaxation_release_j=state.maxwell_relaxation_release_j+e1-e2,
            bulk_parameter_energy_change_j=state.bulk_parameter_energy_change_j+e1-e0,
            thermal_eigenstrain_energy_change_j=state.thermal_eigenstrain_energy_change_j+e3-e2,
            bulk_increment_work_j=state.bulk_increment_work_j+work,
            bulk_increment_energy_loss_j=state.bulk_increment_energy_loss_j+loss)
        # Motion changes column depth at fixed mass, even after conduction has
        # finished. Check geometric distortion independently of solidification.
        _, _, final_fraction, _, _, final_depth = self._phase_fields(provisional, loading.thermal_state)
        actual_area_change = self._interface_area_change(geometry, final_fraction, final_depth)
        if actual_area_change > self.parameters.max_interface_area_change_fraction:
            raise _RetryStep("coupled_interface_geometry_limit")
        provisional.interface_area_change_fraction = actual_area_change
        interface_change = self._interface_energy(provisional, geometry)-self._interface_energy(state, geometry)
        contact_loss = sum(float(np.sum(getattr(history, name)-getattr(state.contact, name))) for name in
                           ("friction_work_cell_j", "viscous_work_cell_j", "fracture_work_cell_j"))
        remainder = history.external_work_j-state.contact.external_work_j-work-interface_change-contact_loss-(history.drag_work_j-state.contact.drag_work_j)
        provisional.mechanical_energy_remainder_j = state.mechanical_energy_remainder_j+remainder
        if loading.thermal_state.stopped_reason:
            provisional.stopped_reason = loading.thermal_state.stopped_reason
        return provisional, loading.thermal_state, loading.orbit

    def step(self, state, thermal, orbit, target_myr, max_step_myr=.001):
        if (state.stopped_reason or thermal.stopped_reason or not np.isfinite(target_myr)
                or target_myr <= state.time_myr or state.time_myr != thermal.time_myr or state.time_myr != orbit.time_myr):
            raise ValueError("Coupled physical clocks must agree and advance from an unstopped state")
        if not np.isfinite(max_step_myr) or max_step_myr <= 0:
            raise ValueError("Thermal maximum step must be finite and positive")
        dt = min(target_myr-state.time_myr, self.parameters.max_step_years/1e6)
        # Absolute Myr clocks accumulate several ulps across small mechanical
        # steps. Do not solve a spurious sub-second tail with drag/dt so large
        # that Newton corrections are smaller than displacement roundoff.
        time_tolerance = 128*np.spacing(max(abs(state.time_myr), abs(target_myr), 1.))
        while state.time_myr < target_myr-time_tolerance:
            end = min(target_myr, state.time_myr+dt)
            if target_myr-end <= time_tolerance:
                end = target_myr
            try:
                following, next_thermal, next_orbit = self._trial(state, thermal, orbit, end, max_step_myr)
            except _RetryStep as exc:
                state = replace(state, rejected_steps=state.rejected_steps+1)
                if dt <= self.parameters.min_step_years/1e6*(1+1e-9):
                    state = replace(state, stopped_reason=str(exc))
                    break
                dt = max(dt/2, self.parameters.min_step_years/1e6)
                continue
            state, thermal, orbit = following, next_thermal, next_orbit
            if state.stopped_reason:
                break
        return state, thermal, orbit, [self.diagnostics(state, thermal, orbit)]

    def fields(self, state, thermal):
        geometry = self._geometry(state.cut_edges)
        mesh, radius, fraction, mean, temp, column_depth = self._phase_fields(state, thermal)
        jump = geometry._geometric_jump(state.contact, mesh).reshape(-1, 2, 2)
        banks = mesh.vertices[geometry.topology.bank_vertices]
        centers = banks.mean(axis=(1, 2))
        centers /= np.linalg.norm(centers, axis=1)[:, None]
        return {"damage": state.damage, "water_access": state.water_access, "temperature_k": mean,
            "lid_thickness_km": column_depth*fraction, "column_depth_km": column_depth,
            "column_mass_kg": self.layer_mass_kg.sum(axis=1), "seam_centers_xyz": centers,
            "seam_gap_m": jump[:, :, 0], "seam_slip_m": jump[:, :, 1],
            "seam_damage": state.contact.interface_damage.reshape(-1, 2),
            "velocity_km_myr": state.velocity_km_myr, "speed_cm_yr": np.linalg.norm(state.velocity_km_myr, axis=1)/10,
            "tidal_stress_mpa": state.tidal_stress_mpa}

    def diagnostics(self, state, thermal, orbit):
        data = self.fields(state, thermal)
        mesh, _, fraction, _, _, _ = self._phase_fields(state, thermal)
        g = diagnose(thermal, self.source_model.thermal)
        c = state.contact
        average = lambda x: float(np.average(x, weights=mesh.areas_unit_sphere))
        maximum = lambda x: float(np.max(x)) if np.size(x) else 0.
        geometry = self._geometry(state.cut_edges)
        limits = self._geometry_utilization(geometry, c, geometric_gap_m=data["seam_gap_m"])
        elastic = maximum_total_strain(state.elastic_strain)
        limits.update(max_elastic_strain=elastic,
            elastic_strain_utilization=elastic/self.source_model.mobile_p.max_elastic_strain,
            interface_area_utilization=state.interface_area_change_fraction/self.parameters.max_interface_area_change_fraction,
            cohort_count_utilization=len(state.cohorts.trace_index)/self.parameters.max_cohort_count)
        connectivity = seam_connectivity(geometry.topology,
            cohort_bonded_traces(state.cohorts, 2*len(state.cut_edges)))
        return {"time_myr": state.time_myr, "source_time_myr": self.source_time_myr,
            "elapsed_years": (state.time_myr-self.source_time_myr)*1e6,
            "surface_temperature_k": g["surface_temperature_k"], "mantle_temperature_k": g["mantle_temperature_k"],
            "mantle_melt_fraction": g["mantle_melt_fraction"], "ocean_fraction": g["ocean_fraction"],
            "mean_lid_thickness_km": average(data["lid_thickness_km"]), "mean_damage": average(state.damage),
            "mean_water_access": average(state.water_access), "eccentricity": orbit.eccentricity,
            "mean_speed_cm_yr": average(data["speed_cm_yr"]),
            "seam_count": len(state.cut_edges), "new_seam_count": state.new_seam_count,
            # Preserve the old CSV key while exposing actual cohesive links.
            "component_count": connectivity["cut_component_count"], **connectivity,
            "max_opening_m": max(0., maximum(data["seam_gap_m"])),
            "max_penetration_m": max(0., maximum(-data["seam_gap_m"])),
            "max_abs_jump_m": maximum(np.abs(data["seam_slip_m"])),
            **limits,
            "equilibrium_residual": c.equilibrium_residual,
            "interface_area_change_fraction": state.interface_area_change_fraction,
            "interface_cohort_count": len(state.cohorts.trace_index),
            "interface_represented_area_m2": float(state.interface_birth_area_m2.sum()),
            "interface_pending_depth_m": maximum(self._reference_front(self._geometry(state.cut_edges), fraction)-state.interface_depth_ref_m),
            "interface_born_unbonded_area_m2": float(state.cohorts.area_ref_m2[~state.cohorts.bonded].sum()),
            "interface_birth_energy_j": state.interface_birth_energy_j,
            "relative_mass_residual": (float(self.layer_mass_kg.sum())-self.source.initial_mass_kg)/self.source.initial_mass_kg,
            "relative_global_energy_residual": g["relative_energy_residual"],
            "relative_column_energy_residual": (float(np.sum(self.layer_mass_kg*state.column_enthalpy))
                -state.initial_column_energy_j-state.boundary_energy_j)/max(abs(state.initial_column_energy_j), 1.),
            "orbit_heat_transfer_relative_residual": (state.tidal_heat_received_j-orbit.dissipated_energy_j)/max(orbit.dissipated_energy_j, 1.),
            "bulk_elastic_energy_j": self._bulk_energy(state.elastic_strain, state.damage, fraction),
            "interface_elastic_energy_j": self._interface_energy(state, self._geometry(state.cut_edges)),
            "friction_work_j": float(c.friction_work_cell_j.sum()), "viscous_work_j": float(c.viscous_work_cell_j.sum()),
            "fracture_work_j": float(c.fracture_work_cell_j.sum()), "drag_work_j": c.drag_work_j,
            "maxwell_relaxation_release_j": state.maxwell_relaxation_release_j,
            "bulk_parameter_energy_change_j": state.bulk_parameter_energy_change_j,
            "thermal_eigenstrain_energy_change_j": state.thermal_eigenstrain_energy_change_j,
            "bulk_increment_work_j": state.bulk_increment_work_j, "bulk_increment_energy_loss_j": state.bulk_increment_energy_loss_j,
            "mechanical_energy_remainder_j": state.mechanical_energy_remainder_j,
            "accepted_steps": state.accepted_steps, "rejected_steps": state.rejected_steps,
            "last_step_years": state.last_step_years, "stopped_reason": state.stopped_reason}


# Kept here as public imports for CLI/GUI and historical callers.
from .genesis_coupled_checkpoint import save_coupled_checkpoint, load_coupled_checkpoint, _validate_coupled
