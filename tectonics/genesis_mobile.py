"""Moving material shell with corotational Maxwell memory and fixed layer mass.

This is a Lagrangian, fixed-connectivity, spherical membrane experiment, not
plate contact or subduction. Its mechanical radius may change; the global
thermal/orbital model retains its canonical radius (one-way thin-shell forcing).
Column enthalpy excludes mechanical work and is not added to global energy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .genesis import GenesisParameters, GenesisState, SECONDS_PER_MYR, M_EARTH, initial_state, temperatures, diagnose
from .genesis_shell import (ShellParameters, Membrane, initialize_shell, rock_temperature,
    lid_geometry, maxwell_factors, mantle_traction, principal_tensile, maximum_total_strain)
from .genesis_onset import OnsetParameters, update_water_access, advance_orbit_thermal
from .genesis_onset_support import NonlocalLoading, face_velocities, regional_motion
from .genesis_tides import (TidalParameters, TidalOrbitState, initial_tidal_orbit,
    tidal_strain_cycle, tidal_heat_flux_w_m2, tidal_diagnostics, validate_tidal_orbit)
from .genesis_material import (rebuild_material_mesh, face_deformation, polar_increment,
    rotate_tensor, move_mesh, material_layer_mass, material_column_depth, geometry_diagnostics)
from .mesh import build_icosphere

MOBILE_VERSION = "genesis-mobile-0.1"


@dataclass(frozen=True)
class MobileParameters:
    max_incremental_strain: float = .005
    max_vertex_motion_rad: float = .01
    max_damage_increment: float = .02
    max_elastic_strain: float = .03
    equilibrium_tolerance: float = 1e-6
    max_newton_iterations: int = 30
    min_step_myr: float = 1e-6
    min_face_quality: float = .15
    min_area_ratio: float = .15
    max_area_ratio: float = 6.
    max_radius_change_fraction: float = .1

    def validate(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"Invalid mobile parameter {f.name}")
        if not isinstance(self.max_newton_iterations, Integral) or self.max_newton_iterations > 200:
            raise ValueError("max_newton_iterations must be an integer <= 200")
        if (self.max_incremental_strain > .02 or self.max_vertex_motion_rad > .05
                or self.max_damage_increment > .1 or self.max_elastic_strain > .05
                or not 0 < self.min_area_ratio < 1 < self.max_area_ratio
                or self.min_face_quality >= 1 or self.max_radius_change_fraction > .15
                or self.equilibrium_tolerance > 1e-3):
            raise ValueError("Mobile controls exceed the supported numerical/elastic domain")


def mobile_parameters_from_config(config):
    section = dict(config.get("genesis_mobile", {}))
    if section.pop("schema_version", 1) != 1:
        raise ValueError("Unsupported genesis_mobile schema")
    try:
        p = MobileParameters(**section)
    except TypeError as exc:
        raise ValueError("Unknown mobile parameters") from exc
    p.validate()
    return p


@dataclass
class MobileState:
    time_myr: float
    vertices: np.ndarray
    radius_km: float
    layer_mass_kg: np.ndarray
    column_enthalpy: np.ndarray
    elastic_strain: np.ndarray
    damage: np.ndarray
    water_access: np.ndarray
    peak_tensile_pa: np.ndarray
    weak_duration_myr: np.ndarray
    path_length_km: np.ndarray
    velocity_km_myr: np.ndarray
    tidal_stress_mpa: np.ndarray
    initial_column_energy_j: float
    initial_mass_kg: float
    membrane_established: bool = False
    boundary_energy_j: float = 0.
    tidal_heat_received_j: float = 0.
    first_fracture_time_myr: float | None = None
    equilibrium_residual: float = 0.
    last_incremental_strain: float = 0.
    last_step_myr: float = 0.
    accepted_steps: int = 0
    rejected_steps: int = 0
    newton_iterations: int = 0
    stopped_reason: str | None = None


class _RetryStep(RuntimeError):
    pass


def _external_force(membrane, traction, radius_km, young):
    mesh = membrane.mesh
    area = np.zeros(mesh.vertex_count)
    np.add.at(area, mesh.faces.ravel(), np.repeat(mesh.areas_unit_sphere/3, 3))
    force = np.zeros(membrane.ndof)
    force[:-1] = (np.einsum("nij,ni->nj", membrane.vertex_basis, traction)
                  *area[:, None]*radius_km/young).ravel()
    c = membrane.constraints.toarray()
    return force-c.T@np.linalg.solve(c@c.T, c@force)


def _force_residual(membrane, stress, depth, traction, radius_km, young):
    local = np.einsum("fai,fa,f->fi", membrane.b, stress/young,
                      membrane.mesh.areas_unit_sphere*np.maximum(depth, 1e-4))
    internal = np.zeros(membrane.ndof)
    np.add.at(internal, membrane.dofs.ravel(), local.ravel())
    external = _external_force(membrane, traction, radius_km, young)
    # Rigid rotations are constrained, as in the earlier membrane solver.
    c = membrane.constraints.toarray()
    residual = internal-external
    residual -= c.T@np.linalg.solve(c@c.T, c@residual)
    # A stress-free uniform control has only floating point cancellation.
    # Reference strain 1e-7 supplies an absolute floor (6 kPa for E=60 GPa);
    # at tolerance 1e-6 this permits millipascal-scale force roundoff only.
    reference = 1e-7*float(np.sum(membrane.mesh.areas_unit_sphere*np.maximum(depth, 1e-4)))
    scale = max(np.linalg.norm(external), np.linalg.norm(internal), reference)
    return float(np.linalg.norm(residual)/scale)


class MobileModel:
    def __init__(self, shell_p, thermal, onset_p, tides_p, mobile_p=None):
        self.p, self.thermal, self.onset_p, self.tides_p = shell_p, thermal, onset_p, tides_p
        self.mobile_p = mobile_p or MobileParameters()
        thermal.validate(); shell_p.validate(thermal); onset_p.validate(); tides_p.validate(); self.mobile_p.validate()
        if thermal.tidal_heat_flux_w_m2 != 0:
            raise ValueError("Mobile computes tidal heating from orbit; prescribed tidal flux must be zero")
        for a, b in ((tides_p.satellite_radius_km, thermal.radius_km),
                     (tides_p.satellite_mass_kg, thermal.mass_earth*M_EARTH)):
            if not math.isclose(a, b, rel_tol=1e-12):
                raise ValueError("Tidal and thermal satellite parameters must agree")
        self.mesh = build_icosphere(shell_p.subdivisions)

    def initial(self):
        shell = initialize_shell(self.mesh, self.p, self.thermal)
        mass = material_layer_mass(self.mesh, self.thermal.radius_km, self.p.density_kg_m3,
                                   self.p.column_depth_km, self.p.column_layers)
        n = self.mesh.cell_count
        state = MobileState(0., self.mesh.vertices.copy(), self.thermal.radius_km, mass,
            shell.column_enthalpy.copy(), np.zeros((n, 3)), np.zeros(n), np.zeros(n),
            np.zeros(n), np.zeros(n), np.zeros(n), np.zeros((n, 3)), np.zeros(n),
            float(np.sum(mass*shell.column_enthalpy)), float(np.sum(mass)))
        return state, initial_state(self.thermal), initial_tidal_orbit(self.tides_p)

    def mesh_for(self, state):
        return rebuild_material_mesh(self.mesh, state.vertices)

    def _column_geometry(self, state, mesh):
        return material_column_depth(mesh, state.radius_km, state.layer_mass_kg, self.p.density_kg_m3)

    def _prepare_trial_state(self, state):
        """Optional material-law history, prepared without mutating the input."""
        return state

    def _mechanical_response(self, state, rotation, elastic_trial, effective_b,
                             damage, membrane, dt_myr, *, deformation=None):
        degradation = self.p.residual_stiffness+(1-self.p.residual_stiffness)*(1-damage)**2
        stress = (elastic_trial@membrane.d.T)*(self.p.young_modulus_pa*degradation[:, None])
        return elastic_trial, stress, None

    def _finish_trial_state(self, before, after, old_mesh, memory, effective_b, fraction, dt_myr):
        return after

    def checkpoint_array_shapes(self):
        return {}

    def _phase(self, h, surface, mantle):
        temp = rock_temperature(h, self.thermal)
        depth, mean = lid_geometry(temp, surface, mantle, self.p, self.thermal)
        return depth/self.p.column_depth_km, mean, temp

    def _conduct(self, state, mesh, dt, ts0, ts1, tm0, tm1):
        area = mesh.physical_cell_areas_km2(state.radius_km)*1e6
        dz = state.layer_mass_kg/(self.p.density_kg_m3*area[:, None])
        stable = .20*self.p.density_kg_m3*self.thermal.silicate_heat_capacity_j_kg_k*float(dz.min())**2/self.p.conductivity_w_m_k
        count = max(1, math.ceil(dt*SECONDS_PER_MYR/stable))
        if count > 100000:
            raise _RetryStep("mobile_column_resolution_limit")
        ds = dt*SECONDS_PER_MYR/count
        h = state.column_enthalpy.copy()
        boundary = state.boundary_energy_j
        for i in range(count):
            f = (i+.5)/count
            temp = rock_temperature(h, self.thermal)
            flux = np.empty((mesh.cell_count, self.p.column_layers+1))
            k = self.p.conductivity_w_m_k
            flux[:, 0] = 2*k*(ts0+f*(ts1-ts0)-temp[:, 0])/dz[:, 0]
            flux[:, -1] = 2*k*(temp[:, -1]-tm0-f*(tm1-tm0))/dz[:, -1]
            flux[:, 1:-1] = k*(temp[:, :-1]-temp[:, 1:])/(.5*(dz[:, :-1]+dz[:, 1:]))
            h += (flux[:, :-1]-flux[:, 1:])*area[:, None]*ds/state.layer_mass_kg
            boundary += float(np.dot(area, flux[:, 0]-flux[:, -1])*ds)
        return h, boundary

    def _equilibrium(self, state, old_mesh, fraction, memory, effective_b, damage, guess=None, *, dt_myr=None):
        """Material-tangent Newton, checking forces on every moved candidate.

        Memory and thermal/phase factors are fixed from the start of the step.
        They are never repeatedly relaxed by the equilibrium iteration.
        """
        mp, p = self.mobile_p, self.p
        mesh, radius = guess if guess is not None else (old_mesh, state.radius_km)
        degradation = p.residual_stiffness+(1-p.residual_stiffness)*(1-damage)**2

        def evaluate(candidate, r):
            membrane = Membrane(candidate, p.poisson_ratio)
            deformation = face_deformation(old_mesh, candidate, state.radius_km, r)
            rotation, inc = polar_increment(deformation)
            elastic = rotate_tensor(memory+effective_b[:, None]*inc, rotation, engineering=True)
            depth = material_column_depth(candidate, r, state.layer_mass_kg, p.density_kg_m3)*fraction
            # The closed-membrane gate is applied before Newton. Changing its
            # active set during a geometric correction creates a false jump
            # in support as a just-born layer crosses the diagnostic threshold.
            active = fraction > 0
            elastic[~active] = 0.
            elastic, stress, tangent = self._mechanical_response(
                state, rotation, elastic, effective_b, damage, membrane, dt_myr,
                deformation=deformation)
            traction = mantle_traction(candidate, p, np.where(active, depth, 0.))
            residual = _force_residual(membrane, stress, depth, traction, r, p.young_modulus_pa)
            return membrane, inc, elastic, stress, depth, active, traction, tangent, residual

        current = evaluate(mesh, radius)
        for iteration in range(mp.max_newton_iterations+1):
            membrane, inc, elastic, stress, depth, active, traction, tangent, residual = current
            if residual <= mp.equilibrium_tolerance:
                return mesh, radius, elastic, stress, depth, residual, inc, iteration
            if iteration == mp.max_newton_iterations:
                break
            stiffness = degradation*effective_b
            stiffness[~active] = 1e-9
            if tangent is None:
                correction_eigen = -(stress@np.linalg.inv(membrane.d).T)/(p.young_modulus_pa*stiffness[:, None])
                _, _, _, radial = membrane.solve(correction_eigen, depth, stiffness,
                                                 p.young_modulus_pa, traction, radius)
            else:
                _, _, radial = membrane.solve_correction(stress, depth, tangent,
                                                        p.young_modulus_pa, traction, radius)
            delta = membrane.last_displacement_rad
            accepted = False
            # Line search affects only the Newton correction, never dt/ledgers.
            for power in range(12):
                factor = .5**power
                try:
                    candidate = move_mesh(mesh, membrane.vertex_basis, delta*factor)
                    new_radius = radius*math.exp(float(np.clip(radial*factor, -.2, .2)))
                    trial = evaluate(candidate, new_radius)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if trial[-1] < residual or trial[-1] <= mp.equilibrium_tolerance:
                    mesh, radius, current, accepted = candidate, new_radius, trial, True
                    break
            if not accepted:
                break
        raise _RetryStep("mobile_equilibrium_limit")

    def _trial(self, state, global_state, orbit, target, max_step):
        state = self._prepare_trial_state(state)
        p, mp = self.p, self.mobile_p
        mesh = self.mesh_for(state)
        next_global, next_orbit, heating, rows = advance_orbit_thermal(
            global_state, orbit, self.thermal, self.tides_p, target, max_step)
        dt = next_global.time_myr-state.time_myr
        tm0, ts0 = temperatures(np.asarray(global_state.energy), self.thermal)
        tm1, ts1 = temperatures(np.asarray(next_global.energy), self.thermal)
        h, boundary = self._conduct(state, mesh, dt, ts0, ts1, tm0, tm1)
        old_fraction, _, old_temp = self._phase(state.column_enthalpy, ts0, tm0)
        fraction, mean, temp = self._phase(h, ts1, tm1)
        column_depth = self._column_geometry(state, mesh)
        active = fraction*column_depth >= p.min_load_bearing_thickness_km
        old_active = old_fraction*column_depth >= p.min_load_bearing_thickness_km
        established = state.membrane_established or bool(np.all(active))
        if state.membrane_established:
            if np.any(fraction*column_depth < .25*p.min_load_bearing_thickness_km):
                raise _RetryStep("mobile_partial_melt_limit")
            active = fraction > 0
            old_active = old_fraction > 0
        elif not established:
            # Isolated frozen rafts cannot be treated as an equilibrated closed
            # membrane on an almost zero-stiffness ocean. Their motion needs a
            # separate fluid/contact model. Continue thermal formation only.
            active = np.zeros_like(active)
            old_active = np.zeros_like(old_active)
        shared = np.minimum(old_fraction, fraction)
        retained = np.divide(shared, fraction, out=np.zeros_like(fraction), where=active)*old_active
        weights = np.clip(shared[:, None]*p.column_layers-np.arange(p.column_layers), 0, 1)
        delta_t = np.divide(np.sum((temp-old_temp)*weights, axis=1), weights.sum(axis=1),
                            out=np.zeros(mesh.cell_count), where=weights.sum(axis=1)>0)
        eta = np.clip(p.viscosity_reference_pa_s*np.exp(np.clip(p.activation_energy_j_mol/8.314462618
                      *(1/np.maximum(mean, 1)-1/p.viscosity_reference_temperature_k), -60, 60)),
                      p.viscosity_min_pa_s, p.viscosity_max_pa_s)
        r, b = maxwell_factors(dt*SECONDS_PER_MYR, eta/p.young_modulus_pa)
        effective_b = retained*b+(1-retained)
        memory = (retained*r)[:, None]*state.elastic_strain
        memory[:, :2] -= (retained*b*p.linear_expansion_per_k*delta_t)[:, None]
        damage0 = state.damage*retained
        if np.any(active):
            solved = self._equilibrium(state, mesh, fraction, memory, effective_b, damage0, dt_myr=dt)
        else:
            solved = (mesh, state.radius_km, np.zeros_like(memory), np.zeros_like(memory),
                      fraction*column_depth, 0., np.zeros_like(memory), 0)
        moved, radius, elastic, stress, depth, residual, inc, iterations = solved
        active = (fraction > 0) if established else np.zeros_like(active)
        liquid = rows[-1]["ocean_fraction"]
        water = update_water_access(state.water_access, damage0, retained, active, ts1, liquid, dt, self.onset_p)
        strength = p.tensile_strength_pa*(1-(1-self.onset_p.wet_strength_fraction)*water)
        tide_params = replace(self.tides_p, eccentricity=.5*(orbit.eccentricity+next_orbit.eccentricity),
                              semimajor_axis_km=.5*(orbit.semimajor_axis_km+next_orbit.semimajor_axis_km))
        membrane = Membrane(moved, p.poisson_ratio)
        tide = np.einsum("ab,pfb->pfa", membrane.d, tidal_strain_cycle(moved, tide_params))
        tide *= (p.young_modulus_pa*(p.residual_stiffness+(1-p.residual_stiffness)*(1-damage0)**2))[None, :, None]
        tide[:, ~active] = 0.
        tensile = np.stack([principal_tensile(stress+t) for t in tide])
        tidal_peak = np.max(np.stack([principal_tensile(t) for t in tide]), axis=0)/1e6
        loading = np.mean(np.maximum(tensile/strength-1, 0)**2, axis=0)/p.damage_timescale_myr
        loading = NonlocalLoading(moved, radius, self.onset_p.regularization_km).apply(loading, active)
        hot = np.clip((mean-700)/650, 0, 1)
        healing = 1/p.cold_healing_timescale_myr+hot**4/p.hot_healing_timescale_myr
        rate = loading+healing
        equilibrium = loading/rate
        damage = np.where(active, np.clip(equilibrium+(damage0-equilibrium)*np.exp(-rate*dt), 0, 1), 0.)
        if np.max(np.abs(damage-damage0)) > mp.max_damage_increment:
            raise _RetryStep("mobile_damage_step_limit")
        peak = np.maximum(state.peak_tensile_pa, principal_tensile(stress))
        if np.any(active) and np.max(np.abs(damage-damage0)) > 1e-12:
            solved = self._equilibrium(state, mesh, fraction, memory, effective_b, damage, (moved, radius), dt_myr=dt)
            moved, radius, elastic, stress, depth, residual, inc, iterations2 = solved
            iterations += iterations2
        peak = np.maximum(peak, principal_tensile(stress))
        inc_max = maximum_total_strain(inc)
        angle = np.arctan2(np.linalg.norm(np.cross(mesh.vertices, moved.vertices), axis=1),
                           np.sum(mesh.vertices*moved.vertices, axis=1)).max()
        if inc_max > mp.max_incremental_strain or angle > mp.max_vertex_motion_rad:
            raise _RetryStep("mobile_motion_step_limit")
        if maximum_total_strain(elastic) > mp.max_elastic_strain:
            raise _RetryStep("mobile_elastic_limit")
        geom = geometry_diagnostics(moved, self.mesh, radius, self.thermal.radius_km)
        if (geom["min_face_quality"] < mp.min_face_quality or geom["min_area_ratio"] < mp.min_area_ratio
                or geom["max_area_ratio"] > mp.max_area_ratio):
            raise _RetryStep("mobile_mesh_quality_limit")
        if abs(radius/self.thermal.radius_km-1) > mp.max_radius_change_fraction:
            raise _RetryStep("mobile_thin_shell_limit")
        velocity = face_velocities(mesh.centroids, moved.centroids, .5*(state.radius_km+radius), dt)
        path = state.path_length_km+np.linalg.norm(velocity, axis=1)*dt
        first = state.first_fracture_time_myr
        if first is None and np.any(damage >= p.damage_threshold):
            first = next_global.time_myr
        next_state = replace(state, time_myr=next_global.time_myr, vertices=moved.vertices.copy(), radius_km=radius,
            membrane_established=established,
            column_enthalpy=h, elastic_strain=elastic, damage=damage, water_access=water,
            peak_tensile_pa=peak, weak_duration_myr=np.where(damage>=p.damage_threshold, state.weak_duration_myr+dt, 0.),
            path_length_km=path, velocity_km_myr=velocity, tidal_stress_mpa=tidal_peak,
            boundary_energy_j=boundary, tidal_heat_received_j=state.tidal_heat_received_j+heating*dt*SECONDS_PER_MYR*self.thermal.area_m2,
            first_fracture_time_myr=first, equilibrium_residual=residual, last_incremental_strain=inc_max,
            last_step_myr=dt, accepted_steps=state.accepted_steps+1, newton_iterations=iterations)
        next_state = self._finish_trial_state(state, next_state, mesh, memory, effective_b, fraction, dt)
        return next_state, next_global, next_orbit, rows

    def step(self, state, global_state, orbit, target_myr, max_step_myr=.01):
        if (state.stopped_reason or global_state.stopped_reason or not math.isfinite(target_myr)
                or target_myr <= state.time_myr or global_state.time_myr != state.time_myr
                or orbit.time_myr != state.time_myr):
            raise ValueError("Mobile clocks must agree and advance from an unstopped state")
        # Retry from an immutable start state. Rejected heat/mass/damage never
        # leak into accepted ledgers. Each requested interval is deterministic.
        rows, trial_dt = [], target_myr-state.time_myr
        while state.time_myr < target_myr-1e-14:
            end = min(target_myr, state.time_myr+trial_dt)
            try:
                next_state, next_global, next_orbit, step_rows = self._trial(state, global_state, orbit, end, max_step_myr)
            except _RetryStep as exc:
                state = replace(state, rejected_steps=state.rejected_steps+1)
                if trial_dt <= self.mobile_p.min_step_myr*(1+1e-9):
                    state = replace(state, stopped_reason=str(exc))
                    break
                trial_dt = max(trial_dt/2, self.mobile_p.min_step_myr)
                continue
            state, global_state, orbit = next_state, next_global, next_orbit
            # Keep event rows; only the final accepted sample is a regular row.
            rows.extend(step_rows[:-1])
            if global_state.stopped_reason:
                break
        rows.append(self.diagnostics(state, global_state, orbit)[0])
        return state, global_state, orbit, rows

    def fields(self, state, global_state):
        mesh = self.mesh_for(state)
        tm, ts = temperatures(np.asarray(global_state.energy), self.thermal)
        fraction, mean, temp = self._phase(state.column_enthalpy, ts, tm)
        depth = self._column_geometry(state, mesh)*fraction
        membrane = Membrane(mesh, self.p.poisson_ratio)
        degradation = self.p.residual_stiffness+(1-self.p.residual_stiffness)*(1-state.damage)**2
        stress = (state.elastic_strain@membrane.d.T)*(self.p.young_modulus_pa*degradation[:, None])
        edges = np.asarray(mesh.shared_edges)
        damaged = state.damage >= self.p.damage_threshold
        _, total_strain = polar_increment(face_deformation(self.mesh, mesh, self.thermal.radius_km, state.radius_km))
        return {"temperature_k": mean, "solid_fraction": 1-np.clip((temp[:, 0]-self.thermal.solidus_k)/(self.thermal.liquidus_k-self.thermal.solidus_k), 0, 1),
            "lid_thickness_km": depth, "tensile_stress_mpa": principal_tensile(stress)/1e6,
            "peak_tensile_stress_mpa": state.peak_tensile_pa/1e6, "damage": state.damage,
            "failed_edges": damaged[edges[:, 0]] != damaged[edges[:, 1]], "water_access": state.water_access,
            "speed_cm_yr": np.linalg.norm(state.velocity_km_myr, axis=1)*.1,
            "displacement_km": state.path_length_km, "tidal_stress_mpa": state.tidal_stress_mpa,
            "material_centroids": mesh.centroids, "velocity_km_myr": state.velocity_km_myr,
            "column_depth_km": self._column_geometry(state, mesh), "total_hencky_strain": total_strain,
            "column_mass_kg": state.layer_mass_kg.sum(axis=1)}

    def diagnostics(self, state, global_state, orbit):
        mesh = self.mesh_for(state)
        area = mesh.areas_unit_sphere
        pnow = replace(self.tides_p, semimajor_axis_km=orbit.semimajor_axis_km, eccentricity=orbit.eccentricity)
        g = diagnose(global_state, replace(self.thermal, tidal_heat_flux_w_m2=tidal_heat_flux_w_m2(pnow)))
        data = self.fields(state, global_state)
        active = data["lid_thickness_km"] >= self.p.min_load_bearing_thickness_km
        motion = regional_motion(mesh, mesh.centroids, state.velocity_km_myr, active & (state.damage < self.p.damage_threshold))
        energy = float(np.sum(state.layer_mass_kg*state.column_enthalpy))
        geom = geometry_diagnostics(mesh, self.mesh, state.radius_km, self.thermal.radius_km)
        s = {"time_myr": state.time_myr, "first_fracture_time_myr": state.first_fracture_time_myr,
            "solid_surface_fraction": float(np.average(active, weights=area)),
            "mean_lid_thickness_km": float(np.average(data["lid_thickness_km"], weights=area)),
            "max_lid_thickness_km": float(data["lid_thickness_km"].max()),
            "damaged_area_fraction": float(np.average(state.damage>=self.p.damage_threshold, weights=area)),
            "mean_damage": float(np.average(state.damage, weights=area)),
            "max_tensile_stress_mpa": float(data["tensile_stress_mpa"].max()),
            "peak_tensile_stress_mpa": float(state.peak_tensile_pa.max()/1e6),
            "intact_region_count": motion["candidate_region_count"],
            "relative_column_energy_residual": (energy-state.initial_column_energy_j-state.boundary_energy_j)/state.initial_column_energy_j,
            "relative_material_mass_residual": (float(state.layer_mass_kg.sum())-state.initial_mass_kg)/state.initial_mass_kg,
            "mechanical_equilibrium_residual": state.equilibrium_residual,
            "mechanical_radius_km": state.radius_km,
            "membrane_established": state.membrane_established,
            "max_total_membrane_strain": maximum_total_strain(data["total_hencky_strain"]),
            "max_elastic_strain": maximum_total_strain(state.elastic_strain),
            "last_incremental_strain": state.last_incremental_strain, "last_step_myr": state.last_step_myr,
            "accepted_steps": state.accepted_steps, "rejected_steps": state.rejected_steps,
            "newton_iterations": state.newton_iterations, "stopped_reason": state.stopped_reason, **geom}
        o = {"time_myr": state.time_myr, "mean_water_access": float(np.average(state.water_access, weights=area)),
            "max_water_access": float(state.water_access.max()),
            "mean_speed_cm_yr": float(np.average(data["speed_cm_yr"], weights=area)),
            "max_speed_cm_yr": float(data["speed_cm_yr"].max()), "max_displacement_km": float(state.path_length_km.max()),
            "persistent_weak_area_fraction": float(np.average(state.weak_duration_myr>=self.onset_p.persistence_time_myr, weights=area)),
            "max_tidal_stress_mpa": float(state.tidal_stress_mpa.max()), "tidal_heat_received_j": state.tidal_heat_received_j,
            "orbit_heat_transfer_relative_residual": (state.tidal_heat_received_j-orbit.dissipated_energy_j)/max(orbit.dissipated_energy_j, 1.),
            "eccentricity": orbit.eccentricity, "semimajor_axis_km": orbit.semimajor_axis_km,
            "tidal_heat_flux_w_m2": tidal_heat_flux_w_m2(pnow), **motion}
        return g, s, o, tidal_diagnostics(pnow)


def save_mobile_checkpoint(path, model, state, thermal, orbit, controls, provenance):
    arrays = {f.name: getattr(state, f.name) for f in fields(state) if isinstance(getattr(state, f.name), np.ndarray)}
    scalars = {f.name: getattr(state, f.name) for f in fields(state) if f.name not in arrays}
    parameters = {"shell": asdict(model.p), "thermal": asdict(model.thermal), "onset": asdict(model.onset_p),
                  "tides": asdict(model.tides_p), "mobile": asdict(model.mobile_p)}
    meta = {"format": MOBILE_VERSION, "parameters": parameters,
        "parameter_hash": hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest(),
        "state": scalars, "thermal_state": asdict(thermal), "orbit": asdict(orbit),
        "controls": controls, "provenance": provenance}
    target = Path(path)
    temporary = Path(str(target)+".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, metadata=np.array(json.dumps(meta, allow_nan=False)), **arrays)
    temporary.replace(target)


def load_mobile_checkpoint(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            p = meta["parameters"]
            if meta["format"] != MOBILE_VERSION or hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest() != meta["parameter_hash"]:
                raise ValueError("Mobile checkpoint version/parameter hash mismatch")
            model = MobileModel(ShellParameters(**p["shell"]), GenesisParameters(**p["thermal"]),
                OnsetParameters(**p["onset"]), TidalParameters(**p["tides"]), MobileParameters(**p["mobile"]))
            state = MobileState(**meta["state"], **{k: archive[k].copy() for k in archive.files if k != "metadata"})
        thermal, orbit = GenesisState(**meta["thermal_state"]), TidalOrbitState(**meta["orbit"])
        _validate_loaded(model, state, thermal, orbit)
        return model, state, thermal, orbit, meta
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed mobile checkpoint") from exc


def _validate_loaded(model, state, thermal, orbit):
    validate_tidal_orbit(orbit, model.tides_p)
    n, v, layers = model.mesh.cell_count, model.mesh.vertex_count, model.p.column_layers
    if not all(math.isfinite(t) and t >= 0 and t == state.time_myr for t in (state.time_myr, thermal.time_myr, orbit.time_myr)):
        raise ValueError("Mobile checkpoint clocks differ")
    shapes = {"vertices": (v, 3), "layer_mass_kg": (n, layers), "column_enthalpy": (n, layers),
              "elastic_strain": (n, 3), "velocity_km_myr": (n, 3)}
    shapes.update(model.checkpoint_array_shapes())
    if not isinstance(state.membrane_established, bool):
        raise ValueError("Mobile membrane_established must be boolean")
    for f in fields(state):
        value = getattr(state, f.name)
        if isinstance(value, np.ndarray):
            if value.shape != shapes.get(f.name, (n,)) or not np.isfinite(value).all():
                raise ValueError(f"Invalid mobile array {f.name}")
        elif isinstance(value, Real) and not math.isfinite(value):
            raise ValueError(f"Invalid mobile scalar {f.name}")
    expected_mass = material_layer_mass(model.mesh, model.thermal.radius_km, model.p.density_kg_m3, model.p.column_depth_km, layers)
    if not np.array_equal(state.layer_mass_kg, expected_mass) or state.initial_mass_kg != float(expected_mass.sum()):
        raise ValueError("Mobile material mass changed")
    if not state.membrane_established and (np.any(state.elastic_strain != 0)
            or np.any(state.path_length_km != 0) or not np.array_equal(state.vertices, model.mesh.vertices)
            or state.radius_km != model.thermal.radius_km):
        raise ValueError("Unestablished mobile membrane cannot carry mechanical history")
    if (state.radius_km <= 0 or state.initial_column_energy_j <= 0 or np.any(state.column_enthalpy <= 0)
            or any(np.any((x<0)|(x>1)) for x in (state.damage, state.water_access))
            or any(np.any(x<0) for x in (state.path_length_km, state.weak_duration_myr, state.tidal_stress_mpa, state.peak_tensile_pa))
            or np.any(state.weak_duration_myr > state.time_myr+1e-12)
            or state.tidal_heat_received_j < 0 or state.equilibrium_residual < 0
            or any(not isinstance(getattr(state, name), Integral) or getattr(state, name)<0
                   for name in ("accepted_steps", "rejected_steps", "newton_iterations"))
            or (state.first_fracture_time_myr is not None and not 0 <= state.first_fracture_time_myr <= state.time_myr)):
        raise ValueError("Nonphysical mobile checkpoint")
    if (len(thermal.energy) != 4 or not all(math.isfinite(e) and e >= 0 for e in thermal.energy)
            or not math.isfinite(thermal.initial_total_energy) or thermal.initial_total_energy <= 0
            or any(not math.isfinite(t) or not 0 <= t <= thermal.time_myr for t in thermal.events.values())):
        raise ValueError("Invalid mobile thermal history")
    tm, ts = temperatures(np.asarray(thermal.energy), model.thermal)
    fraction, _, _ = model._phase(state.column_enthalpy, ts, tm)
    if state.membrane_established and np.any(fraction <= 0):
        raise ValueError("Established mobile membrane must have a connected solid layer")
    g, s, o, _ = model.diagnostics(state, thermal, orbit)
    if any(abs(value)>1e-8 for value in (g["relative_energy_residual"], s["relative_column_energy_residual"], o["orbit_heat_transfer_relative_residual"])):
        raise ValueError("Mobile energy ledger mismatch")
    mp = model.mobile_p
    if (s["min_face_quality"] < mp.min_face_quality or s["min_area_ratio"] < mp.min_area_ratio
            or s["max_area_ratio"] > mp.max_area_ratio or s["max_elastic_strain"] > mp.max_elastic_strain
            or s["last_incremental_strain"] < 0 or s["last_incremental_strain"] > mp.max_incremental_strain
            or s["mechanical_equilibrium_residual"] > mp.equilibrium_tolerance
            or state.last_step_myr < 0 or state.last_step_myr > state.time_myr+1e-12
            or abs(state.radius_km/model.thermal.radius_km-1) > mp.max_radius_change_fraction):
        raise ValueError("Mobile checkpoint violates geometry/elastic validity guards")
