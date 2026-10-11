"""Actual-source volume damage evolution up to a genuine weak-plane gate.

The inherited Fault law describes finite-width shear bands in continuous
material. A gate is not a sharp cut, a new plate, or underthrust admission.
Until a work-consistent three-dimensional slip law is available, reaching
that gate refuses further advancement rather than ignoring the new fault.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math

import numpy as np

from .genesis import diagnose, temperatures
from .genesis_shell import principal_tensile
from .genesis_volume_dynamics import VolumeModel, VolumeState, coefficient_arrays
from .genesis_volume_operator import isotropic_elasticity, _immutable


FORMAT = "genesis-source-bound-volume-fault-evolution-0.1"
DAMAGE_COUPLING = "thermal_first_predictor_damage_then_original_memory_resolve"


@dataclass(frozen=True, slots=True)
class VolumeFaultState(VolumeState):
    fault_history: object = None
    damage_coupling: str = DAMAGE_COUPLING


class VolumeFaultActivationBoundary(RuntimeError):
    """Calculated activation at an accepted start; no attempted step accepted."""
    def __init__(self, candidate_history, activation_report):
        self.candidate_history = candidate_history
        self.activation_report = activation_report
        super().__init__("Calculated Genesis weak-plane activation requires a "
                         "work-consistent 3-D slip law; the volume step was not accepted")


def _face_frames(vertices, faces):
    triangle = np.asarray(vertices, float)[np.asarray(faces, int)]
    edge = triangle[:, 1]-triangle[:, 0]
    cross = np.cross(edge, triangle[:, 2]-triangle[:, 0])
    lengths, areas = np.linalg.norm(edge, axis=1), np.linalg.norm(cross, axis=1)
    if np.any(lengths <= 0) or np.any(areas <= 0) or not np.isfinite(triangle).all():
        raise ValueError("Volume fault driver requires nondegenerate current faces")
    if np.any(np.einsum("fi,fi->f", cross, triangle.mean(axis=1)) <= 0):
        raise ValueError("Volume fault driver face orientation inverted")
    x, z = edge/lengths[:, None], cross/areas[:, None]
    return np.stack((x, np.cross(z, x), z), axis=2)


def _stress_tensor(stress):
    result = np.zeros(np.shape(stress)[:-1]+(3, 3))
    result[..., 0, 0], result[..., 1, 1], result[..., 2, 2] = np.moveaxis(stress[..., :3], -1, 0)
    for i, j, k in ((0, 1, 3), (0, 2, 4), (1, 2, 5)):
        result[..., i, j] = result[..., j, i] = stress[..., k]
    return result


def volume_face_response(assembly, mechanics, damage, physical_vertices_m, parameters):
    """Integrate all cohort memory, then project onto CURRENT top-face axes.

    The radial mean of strain0+strain1/lambda and the degree-two angular rule
    integrate stress exactly for the conical field. No face-mean material
    replaces the individual memory or its energy in the mechanical operator.
    """
    if tuple(int(i) for i in assembly.cohort_ids) != tuple(c.id for c in mechanics.cohorts):
        raise ValueError("Fault stress driver lost live cohort identity")
    e0, e1 = coefficient_arrays(mechanics)
    mean = e0.mean(axis=1)+assembly.inverse_scale_mean[:, None]*e1.mean(axis=1)
    degradation = (parameters.residual_stiffness+(1-parameters.residual_stiffness)
                   *(1-np.asarray(damage)[assembly.face_indices])**2)
    C = isotropic_elasticity(parameters.young_modulus_pa, parameters.poisson_ratio)
    stress = (mean@C.T)*degradation[:, None]
    world = assembly.basis_xyz@_stress_tensor(stress)@assembly.basis_xyz.transpose(0, 2, 1)
    n = len(assembly.geometry.faces)
    volume = np.zeros(n)
    world_mean = np.zeros((n, 3, 3))
    temperature = np.zeros(n)
    for i, cohort in enumerate(mechanics.cohorts):
        f, v = int(assembly.face_indices[i]), float(assembly.volume_m3[i])
        volume[f] += v
        world_mean[f] += v*world[i]
        temperature[f] += v*cohort.temperature_k
    live = volume > 0
    world_mean[live] /= volume[live, None, None]
    temperature[live] /= volume[live]
    frames = _face_frames(physical_vertices_m, assembly.geometry.faces)
    local = frames.transpose(0, 2, 1)@world_mean@frames
    tangential = np.stack((local[:, 0, 0], local[:, 1, 1], local[:, 0, 1]), axis=1)
    return dict(stress_pa=_immutable(tangential), mean_temperature_k=_immutable(temperature),
                world_stress_pa=_immutable(world_mean), live_volume_m3=_immutable(volume),
                active_material=_immutable(live, bool), current_basis_xyz=_immutable(frames))


def retained_live_fraction(before_geometry, after_geometry):
    """Retained mass divided by endpoint mass, directly from cohort intervals."""
    old = {c.id: c for c in before_geometry.cohorts}
    n = len(after_geometry.faces)
    retained, total = np.zeros(n), np.zeros(n)
    for c in after_geometry.cohorts:
        mass = after_geometry.column_mass_kg[c.face_index]
        total[c.face_index] += mass*(c.live_q_hi-c.q_lo)
        previous = old.get(c.id)
        if previous is not None:
            retained[c.face_index] += mass*max(0., min(c.live_q_hi, previous.live_q_hi)-c.q_lo)
    if np.any(retained > total*(1+5e-15)):
        raise ValueError("Volume retained material exceeds endpoint material")
    return np.divide(retained, total, out=np.zeros(n), where=total > 0)


class VolumeFaultModel(VolumeModel):
    """Couple genuine stress-driven damage while refusing unsupported slip."""

    state_type = VolumeFaultState

    def __init__(self, source_packet, contact_parameters=None):
        super().__init__(source_packet, contact_parameters)
        from .genesis_volume_fault_evolution import initial_volume_fault_history
        self.initial_fault_history = initial_volume_fault_history(self.source_packet)
        if np.any(self.initial_fault_history.fault_active):
            raise VolumeFaultActivationBoundary(self.initial_fault_history,
                {"inherited_active_weak_planes": True, "physical_banks_admitted": False})

    def _state_damage(self, state):
        return np.asarray(state.fault_history.damage)

    def initial_state(self):
        base = self._build_initial_state()
        state = VolumeFaultState(**{f.name: getattr(base, f.name) for f in fields(VolumeState)},
            fault_history=self.initial_fault_history)
        self.validate_state(state)
        return state

    def validate_state(self, state):
        from .genesis_volume_fault_evolution import VolumeFaultHistory, validate_volume_fault_history
        if type(state) is not VolumeFaultState or state.damage_coupling != DAMAGE_COUPLING:
            raise ValueError("Volume fault state changed its damage-coupling semantics")
        f = state.fault_history
        if (type(f) is not VolumeFaultHistory
                or f.source_checkpoint_sha256 != self.source_operator.source_checkpoint_sha256
                or f.source_identity != self.source_operator.source_identity
                or f.time_myr != state.time_myr or f.accepted_steps != state.mechanics.accepted_steps):
            raise ValueError("Volume fault history lost source/clock/accepted-step binding")
        validate_volume_fault_history(f, self.source_packet)
        if np.any(f.fault_active):
            raise ValueError("An active weak-plane cannot be accepted by the elastic volume branch")
        initial = self.initial_fault_history
        if f.accepted_steps == 0:
            for entry in fields(initial):
                actual, expected = getattr(f, entry.name), getattr(initial, entry.name)
                if isinstance(actual, np.ndarray):
                    same = np.array_equal(actual, expected)
                else:
                    same = actual == expected
                if not same:
                    raise ValueError("Initial volume fault history differs from literal Genesis source")
        frozen = ("plane_normal", "activation_time_myr", "cumulative_shear", "signed_shear",
                  "last_shear_increment", "shear_stress_pa", "normal_stress_pa", "shear_strength_pa",
                  "friction_work_cell_j", "viscous_work_cell_j")
        if any(not np.array_equal(getattr(f, k), getattr(initial, k)) for k in frozen):
            raise ValueError("Inactive volume continuation changed inherited weak-plane/slip history")
        if f.friction_work_j != initial.friction_work_j or f.viscous_fault_work_j != initial.viscous_fault_work_j:
            raise ValueError("Volume damage cannot invent inherited frictional work")
        return super().validate_state(state)

    def face_response(self, state):
        damage = self._state_damage(state)
        assembly, _ = self._assembly(state.geometry, damage)
        vertices = np.asarray(state.geometry.vertices_m)+state.displacement_m.reshape(-1, 6)[:, :3]
        return volume_face_response(assembly, state.mechanics, damage, vertices, self.shell_parameters)

    def step(self, state, dt_myr):
        from .genesis_volume_fault_evolution import (
            evaluate_volume_fault_evolution, prepare_volume_fault_activation)
        from .genesis_volume_thermal import prepare_volume_thermal_step
        self.validate_state(state)
        if isinstance(dt_myr, bool) or not math.isfinite(dt_myr) or dt_myr <= 0:
            raise ValueError("Volume fault step must advance real Genesis time")
        candidate, event = prepare_volume_fault_activation(state.fault_history,
            self.source_packet, self.face_response(state)["stress_pa"])
        if event is not None:
            raise VolumeFaultActivationBoundary(candidate, event)
        thermal = prepare_volume_thermal_step(state.thermal, state.history, state.geometry,
            dt_myr, displacement_m=state.displacement_m)
        retained = retained_live_fraction(state.geometry, thermal.geometry)
        before_damage = self._state_damage(state)
        damage0 = before_damage*retained
        provisional = replace(state.fault_history, time_myr=thermal.state.time_myr,
            accepted_steps=state.fault_history.accepted_steps+1, damage=damage0)
        # This private predictor has no published journal, damage or work. The
        # final solve starts again from the original accepted memory and q/v.
        predictor = self._advance_trial(state, thermal, before_damage, damage0,
            requested_dt_myr=dt_myr, extra_state_fields={"fault_history": provisional})
        driver = self.face_response(predictor)
        vertices = np.asarray(thermal.geometry.vertices_m)+predictor.displacement_m.reshape(-1, 6)[:, :3]
        _, surface = temperatures(np.asarray(thermal.state.global_state.energy),
                                  thermal.state.source_boundary_parameters.thermal)
        ocean = diagnose(thermal.state.global_state,
                         thermal.state.source_boundary_parameters.thermal)["ocean_fraction"]
        faults, evolution_report = evaluate_volume_fault_evolution(state.fault_history,
            self.source_packet, after_time_myr=thermal.state.time_myr,
            physical_vertices_m=vertices, face_stress_pa=driver["stress_pa"],
            face_mean_temperature_k=driver["mean_temperature_k"], retained_fraction=retained,
            active_material=driver["active_material"], surface_temperature_k=surface,
            ocean_fraction=ocean, orbit_before=state.thermal.orbit, orbit_after=thermal.state.orbit)
        final = self._advance_trial(state, thermal, before_damage, np.asarray(faults.damage),
            requested_dt_myr=dt_myr, extra_state_fields={"fault_history": faults})
        # As in Genesis Mobile, retain both predictor and final accepted
        # endpoint stresses. A damage re-solve must not lose a new secular peak.
        final_tensile = principal_tensile(self.face_response(final)["stress_pa"])
        faults = replace(faults, peak_tensile_pa=np.maximum(faults.peak_tensile_pa, final_tensile))
        final = replace(final, fault_history=faults)
        delta = float(final.mechanics.damage_energy_change_j-state.mechanics.damage_energy_change_j)
        report = dict(final.last_report)
        report["fault_evolution"] = {**evolution_report,
            "damage_coupling": DAMAGE_COUPLING,
            "damage_energy_change_j": delta,
            "final_face_tensile_pa": final_tensile.tolist(),
            "combined_heat_mechanical_reservoir_closed": False,
            "physical_banks_admitted": False,
            "predictor_driver": {
                "physical_vertices_m": vertices.tolist(),
                "stress_pa": driver["stress_pa"].tolist(),
                "mean_temperature_k": driver["mean_temperature_k"].tolist(),
                "retained_fraction": retained.tolist(),
                "active_material": driver["active_material"].tolist(),
                "surface_temperature_k": float(surface), "ocean_fraction": float(ocean),
                "elastic_strain0": coefficient_arrays(predictor.mechanics)[0].tolist(),
                "elastic_strain1": coefficient_arrays(predictor.mechanics)[1].tolist(),
                "displacement_m": predictor.displacement_m.tolist(),
                "damage": damage0.tolist()}}
        final = replace(final, last_report=report)
        self.validate_state(final)
        return final
