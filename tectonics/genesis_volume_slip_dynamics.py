"""Actual-source volume evolution with work-consistent finite-width slip.

The inherited material history is not replaced. The new constitutive closure
uses interval-average traction, with an explicitly frozen flow through each
work interval. Top-face co-normal transport occurs only after acceptance.
Neither a sharp cut nor through-depth finite plane transport is inferred.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math

import numpy as np
from scipy import linalg

from .genesis import diagnose, temperatures
from .genesis_faults import transport_plane_normals
from .genesis_shell import principal_tensile
from .genesis_volume_dynamics import VolumeModel, VolumeState, coefficient_arrays
from .genesis_volume_fault_dynamics import (DAMAGE_COUPLING, _face_frames,
    retained_live_fraction, volume_face_response)
from .genesis_volume_fault_evolution import (VolumeFaultHistory, _source,
    initial_volume_fault_history, validate_volume_fault_history,
    prepare_volume_fault_activation, evaluate_volume_fault_evolution)
from .genesis_volume_fault_slip import (prepare_volume_fault_slip,
    evaluate_volume_fault_slip, assemble_volume_fault_average_force_tangent,
    VolumeSlipIncrementBoundary, TRACTION_CLOSURE)
from .genesis_volume_loading import average_volume_loading_stiffness_factors
from .genesis_volume_operator import _immutable, algorithmic_stiffness


SLIP_CLOSURE = "volume_mean_time_average_traction_with_constant_rate_face_shear"
PLANE_TRANSPORT = "top_face_covector_endpoint_transport_with_frozen_interval_flow"
MECHANICAL_WORK_CLOSURE = "base_mechanics_net_driver_and_explicit_total_slip_work"


@dataclass(frozen=True, slots=True)
class VolumeSlipState(VolumeState):
    fault_history: object = None
    damage_coupling: str = DAMAGE_COUPLING
    slip_closure: str = SLIP_CLOSURE
    plane_transport: str = PLANE_TRANSPORT
    mechanical_work_closure: str = MECHANICAL_WORK_CLOSURE
    physical_mechanical_work_j: float = 0.
    fault_friction_dissipation_j: float = 0.
    fault_viscous_dissipation_j: float = 0.
    plane_normal_world: object = None
    slip_direction_world: object = None

    def __post_init__(self):
        VolumeState.__post_init__(self)
        if type(self.fault_history) is not VolumeFaultHistory:
            raise ValueError("Volume slip requires explicit inherited fault history")
        n = len(self.fault_history.damage)
        for name in ("plane_normal_world", "slip_direction_world"):
            array = np.asarray(getattr(self, name), float)
            if array.shape != (n, 3) or not np.isfinite(array).all():
                raise ValueError("Volume slip world-plane coverage or finite values mismatch")
            object.__setattr__(self, name, _immutable(array))
        for name in ("physical_mechanical_work_j", "fault_friction_dissipation_j",
                     "fault_viscous_dissipation_j"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Volume slip runtime work must be finite")
            if name != "physical_mechanical_work_j" and value < 0:
                raise ValueError("Volume slip runtime dissipation must be nonnegative")


class VolumeSlipMaterialBoundary(ValueError):
    """An active defect cannot be passed to unrelated newborn material."""


def _vertices(state):
    return np.asarray(state.geometry.vertices_m)+state.displacement_m.reshape(-1, 6)[:, :3]


def _lift_plane(normal2, active, vertices, faces):
    frame = _face_frames(vertices, faces)
    n = np.einsum("fij,fj->fi", frame[:, :, :2], normal2)
    tangent2 = np.stack((-normal2[:, 1], normal2[:, 0]), axis=1)
    m = np.einsum("fij,fj->fi", frame[:, :, :2], tangent2)
    return np.where(active[:, None], n, 0.), np.where(active[:, None], m, 0.)


def transport_top_face_plane(normal2, active, before_vertices, after_vertices, faces):
    """Source inverse-transpose co-normal rule on the current TOP face only."""
    indices = np.asarray(faces, int)
    frames = [_face_frames(v, faces) for v in (before_vertices, after_vertices)]
    charts = []
    for vertices, frame in zip((before_vertices, after_vertices), frames):
        triangle = np.asarray(vertices)[indices]
        edges = np.stack((triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0]), axis=2)
        charts.append(frame[:, :, :2].transpose(0, 2, 1)@edges)
    deformation = np.linalg.solve(charts[0].transpose(0, 2, 1),
                                  charts[1].transpose(0, 2, 1)).transpose(0, 2, 1)
    transported = transport_plane_normals(normal2, deformation)
    world_n, world_m = _lift_plane(transported, active, after_vertices, faces)
    return transported, world_n, world_m


class VolumeSlipModel(VolumeModel):
    """One source-bound shared-clock branch; no synthetic activation policy."""

    state_type = VolumeSlipState

    def __init__(self, source_packet, contact_parameters=None):
        super().__init__(source_packet, contact_parameters)
        self.initial_fault_history = initial_volume_fault_history(self.source_packet)
        self.fault_parameters = _source(self.source_packet).model.fault_p

    def _state_damage(self, state):
        return np.asarray(state.fault_history.damage)

    def _additional_dissipation(self, state):
        return state.fault_friction_dissipation_j+state.fault_viscous_dissipation_j

    def _trial_additional_dissipation(self, state, trial_fields):
        return (trial_fields['fault_friction_dissipation_j']+
                trial_fields['fault_viscous_dissipation_j'])

    def initial_state(self):
        base = self._build_initial_state()
        history = self.initial_fault_history
        normal, tangent = _lift_plane(history.plane_normal, history.fault_active,
                                     _vertices(base), base.geometry.faces)
        state = VolumeSlipState(**{f.name: getattr(base, f.name) for f in fields(VolumeState)},
            fault_history=history, plane_normal_world=normal, slip_direction_world=tangent)
        self.validate_state(state)
        return state

    def validate_state(self, state):
        if (type(state) is not VolumeSlipState or state.damage_coupling != DAMAGE_COUPLING
                or state.slip_closure != SLIP_CLOSURE or state.plane_transport != PLANE_TRANSPORT
                or state.mechanical_work_closure != MECHANICAL_WORK_CLOSURE):
            raise ValueError("Volume slip state changed declared constitutive semantics")
        h = state.fault_history
        validate_volume_fault_history(h, self.source_packet)
        if not np.array_equal(state.thermal.damage,
                              np.asarray(self.source_packet['current_shell_memory']['damage'])):
            raise ValueError("Volume slip changed the literal source thermal damage baseline")
        if h.time_myr != state.time_myr or h.accepted_steps != state.mechanics.accepted_steps:
            raise ValueError("Volume slip material clocks/counters differ")
        initial = self.initial_fault_history
        if h.accepted_steps == 0:
            for field in fields(initial):
                a, b = getattr(h, field.name), getattr(initial, field.name)
                same = np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
                if not same:
                    raise ValueError("Initial volume slip history differs from literal Genesis source")
            if state.physical_mechanical_work_j != 0 or self._additional_dissipation(state) != 0:
                raise ValueError("Initial volume slip cannot invent prior runtime work")
        if np.any(initial.fault_active & ~h.fault_active):
            raise ValueError("Volume slip silently removed an inherited material defect")
        active = h.fault_active
        normal, tangent = _lift_plane(h.plane_normal, active, _vertices(state), state.geometry.faces)
        if (not np.allclose(state.plane_normal_world, normal, rtol=0, atol=3e-12)
                or not np.allclose(state.slip_direction_world, tangent, rtol=0, atol=3e-12)):
            raise ValueError("Volume slip world vectors lost their top material co-normal binding")
        for name in ("cumulative_shear", "friction_work_cell_j", "viscous_work_cell_j"):
            if np.any(getattr(h, name) < getattr(initial, name)):
                raise ValueError("Volume slip changed inherited accumulated material history")
        runtime = ((h.friction_work_j-initial.friction_work_j, state.fault_friction_dissipation_j),
                   (h.viscous_fault_work_j-initial.viscous_fault_work_j, state.fault_viscous_dissipation_j))
        for literal_delta, ledger in runtime:
            if not math.isclose(literal_delta, ledger, rel_tol=3e-12, abs_tol=1e-8):
                raise ValueError("Volume slip runtime loss differs from inherited-plus-current work")
        net, total, losses = (state.mechanics.mechanical_work_j,
                             state.physical_mechanical_work_j, self._additional_dissipation(state))
        if abs(total-net-losses) > 5e-12*max(abs(total), abs(net), losses, 1.):
            raise ValueError("Volume slip total/net work double-counted or omitted dissipation")
        return super().validate_state(state)

    def face_response(self, state):
        damage = self._state_damage(state)
        assembly, _ = self._assembly(state.geometry, damage)
        return volume_face_response(assembly, state.mechanics, damage, _vertices(state), self.shell_parameters)

    def _mechanical_trial(self, state, after, basal, loading, dt, extra_state_fields):
        history = extra_state_fields['fault_history']
        nworld = np.asarray(extra_state_fields['plane_normal_world'])
        mworld = np.asarray(extra_state_fields['slip_direction_world'])
        prepared = prepare_volume_fault_slip(loading, face_indices=after.face_indices,
            basis_xyz=after.basis_xyz, plane_normal_world=nworld, slip_direction_world=mworld,
            active=history.fault_active, damage=history.damage, water=state.fault_history.water_access,
            parameters=self.fault_parameters)
        M, D = basal.mass_matrix, basal.basal_drag_matrix
        inertia_drag = 2*M/(dt*dt)+D/dt

        def evaluate(increment):
            local = increment[after.dofs]
            inc0 = np.einsum('cqai,ci->cqa', after.B0, local)
            inc1 = np.einsum('cqai,ci->cqa', after.B1, local)
            response = evaluate_volume_fault_slip(prepared, inc0, inc1)
            force = assemble_volume_fault_average_force_tangent(after, prepared, response)
            residual = (2*(M@(increment-dt*state.velocity_m_s))/(dt*dt)
                        +D@increment/dt+force.force_n-basal.external_force_n)
            scale = max(np.linalg.norm(basal.external_force_n), np.linalg.norm(force.force_n), 1.)
            return response, force, residual, float(np.linalg.norm(residual)/scale)

        if not np.any(history.fault_active):
            increment, _, _, _, _, _ = super()._mechanical_trial(
                state, after, basal, loading, dt, extra_state_fields)
            response, force, residual, relative = evaluate(increment)
            iterations = 0
        else:
            # Equilibrate with positive physical ELASTIC/inertial support, not
            # by adding a stiffness floor to a nonassociated material tangent.
            reference = algorithmic_stiffness(after, average_volume_loading_stiffness_factors(loading))+inertia_drag
            diagonal = np.asarray(reference.diagonal())
            if np.any(diagonal <= 0) or not np.isfinite(diagonal).all():
                raise ValueError("Volume slip lacks physical inertial support")
            scaling = 1/np.sqrt(diagonal)
            increment = np.zeros(after.ndof)
            response, force, residual, relative = evaluate(increment)
            for iterations in range(1, self.contact_parameters.max_newton_iterations+1):
                if relative <= min(self.contact_parameters.equilibrium_tolerance, 1e-11):
                    break
                tangent = force.tangent+inertia_drag
                dense = tangent.toarray()*scaling[:, None]*scaling[None, :]
                delta = scaling*linalg.solve(dense, -scaling*residual,
                                             assume_a='gen', check_finite=True)
                accepted_trial = None
                factor = 1.
                for _ in range(24):
                    trial = increment+factor*delta
                    try:
                        candidate = evaluate(trial)
                    except VolumeSlipIncrementBoundary:
                        factor *= .5
                        continue
                    if candidate[3] < relative or candidate[3] <= 1e-11:
                        accepted_trial = (trial, candidate)
                        break
                    factor *= .5
                if accepted_trial is None:
                    raise ValueError("Volume slip nonassociated momentum line search failed")
                increment, (response, force, residual, relative) = accepted_trial
            if relative > min(self.contact_parameters.equilibrium_tolerance, 1e-11):
                raise ValueError("Volume slip coupled momentum did not converge")
        gamma = np.asarray(response.shear_increment)
        active = history.fault_active
        # Every trial started with the SAME accepted-start world vectors.
        # Advance the top-face material covector only now, after the solve.
        end_vertices = np.asarray(after.geometry.vertices_m)+(state.displacement_m+increment).reshape(-1, 6)[:, :3]
        start_normal2 = _face_frames(_vertices(state), state.geometry.faces)[:, :, :2].transpose(0, 2, 1)@nworld[..., None]
        normal2, end_n, end_m = transport_top_face_plane(start_normal2[..., 0], active,
            _vertices(state), end_vertices, after.geometry.faces)
        keep = lambda old, new: np.where(active, new, old)
        history = replace(history, plane_normal=normal2,
            cumulative_shear=history.cumulative_shear+np.abs(gamma),
            signed_shear=history.signed_shear+gamma,
            last_shear_increment=keep(history.last_shear_increment, gamma),
            shear_stress_pa=keep(history.shear_stress_pa, response.shear_stress_pa),
            normal_stress_pa=keep(history.normal_stress_pa, response.normal_stress_pa),
            shear_strength_pa=keep(history.shear_strength_pa, response.yield_strength_pa),
            friction_work_cell_j=history.friction_work_cell_j+response.friction_work_cell_j,
            viscous_work_cell_j=history.viscous_work_cell_j+response.viscous_work_cell_j,
            friction_work_j=history.friction_work_j+response.friction_work_j,
            viscous_fault_work_j=history.viscous_fault_work_j+response.viscous_fault_work_j)
        fields_after = dict(fault_history=history, plane_normal_world=end_n, slip_direction_world=end_m,
            physical_mechanical_work_j=state.physical_mechanical_work_j+response.physical_mechanical_work_j,
            fault_friction_dissipation_j=state.fault_friction_dissipation_j+response.friction_work_j,
            fault_viscous_dissipation_j=state.fault_viscous_dissipation_j+response.viscous_fault_work_j)
        report = dict(closure=SLIP_CLOSURE, traction_closure=TRACTION_CLOSURE,
            plane_transport=PLANE_TRANSPORT, mechanical_work_closure=MECHANICAL_WORK_CLOSURE,
            frozen_start_plane_normal_world=nworld.tolist(), frozen_start_slip_direction_world=mworld.tolist(),
            shear_increment=gamma.tolist(), yielding=response.yielding.tolist(),
            shear_stress_pa=response.shear_stress_pa.tolist(), normal_stress_pa=response.normal_stress_pa.tolist(),
            trial_shear_stress_pa=response.trial_shear_stress_pa.tolist(),
            yield_strength_pa=response.yield_strength_pa.tolist(), active=active.tolist(),
            damage=history.damage.tolist(), water=state.fault_history.water_access.tolist(),
            endpoint_water_access=history.water_access.tolist(),
            water_coupling="accepted_start_lagged_water_fixed_over_both_private_solves",
            persisted_traction_scope="last_interval_average_on_frozen_start_plane",
            plastic_increment0=response.plastic_increment0.tolist(),
            average_strain0=response.average_strain0.tolist(), average_strain1=response.average_strain1.tolist(),
            friction_work_cell_j=response.friction_work_cell_j.tolist(),
            viscous_work_cell_j=response.viscous_work_cell_j.tolist(),
            friction_work_j=response.friction_work_j, viscous_fault_work_j=response.viscous_fault_work_j,
            physical_mechanical_work_j=response.physical_mechanical_work_j,
            net_constitutive_driver_work_j=response.loading_response.mechanical_work_j,
            plastic_work_j=response.plastic_work_j, work_identity_residual_j=response.balance_residual_j,
            coupled_momentum_relative_residual=relative, newton_iterations=iterations,
            finite_width_continuous_material=True, sharp_banks_created=False,
            exact_through_depth_finite_plane_transport_claimed=False,
            physical_fault_contact_admission_claimed=False)
        return (increment, response.loading_response, response.average_strain0,
                response.average_strain1, {'weak_plane_slip':report,
                    'mechanical_work_j':response.physical_mechanical_work_j,
                    'net_constitutive_driver_work_j':response.loading_response.mechanical_work_j,
                    'fault_friction_dissipation_j':response.friction_work_j,
                    'fault_viscous_dissipation_j':response.viscous_fault_work_j}, fields_after)

    def step(self, state, dt_myr):
        from .genesis_volume_thermal import prepare_volume_thermal_step
        self.validate_state(state)
        if isinstance(dt_myr, bool) or not math.isfinite(dt_myr) or dt_myr <= 0:
            raise ValueError("Volume slip must advance genuine Genesis time")
        start_history, activation = prepare_volume_fault_activation(state.fault_history,
            self.source_packet, self.face_response(state)['stress_pa'])
        start_n, start_m = state.plane_normal_world.copy(), state.slip_direction_world.copy()
        new_active = start_history.fault_active & ~state.fault_history.fault_active
        if np.any(new_active):
            n, m = _lift_plane(start_history.plane_normal, start_history.fault_active,
                               _vertices(state), state.geometry.faces)
            start_n[new_active], start_m[new_active] = n[new_active], m[new_active]
        thermal = prepare_volume_thermal_step(state.thermal, state.history, state.geometry,
                                              dt_myr, displacement_m=state.displacement_m)
        retained = retained_live_fraction(state.geometry, thermal.geometry)
        if np.any(start_history.fault_active & (retained <= 0)):
            raise VolumeSlipMaterialBoundary("Active weak-plane material fully retired; no newborn inheritance admitted")
        before_damage = self._state_damage(state)
        damage0 = before_damage*retained
        provisional = replace(start_history, time_myr=thermal.state.time_myr,
            accepted_steps=start_history.accepted_steps+1, damage=damage0)
        common = dict(plane_normal_world=start_n, slip_direction_world=start_m)
        predictor = self._advance_trial(state, thermal, before_damage, damage0,
            requested_dt_myr=dt_myr, extra_state_fields={**common, 'fault_history':provisional})
        driver = self.face_response(predictor)
        vertices = _vertices(predictor)
        _, surface = temperatures(np.asarray(thermal.state.global_state.energy),
                                  thermal.state.source_boundary_parameters.thermal)
        ocean = diagnose(thermal.state.global_state,
                         thermal.state.source_boundary_parameters.thermal)['ocean_fraction']
        faults, evolution = evaluate_volume_fault_evolution(start_history, self.source_packet,
            after_time_myr=thermal.state.time_myr, physical_vertices_m=vertices,
            face_stress_pa=driver['stress_pa'], face_mean_temperature_k=driver['mean_temperature_k'],
            retained_fraction=retained, active_material=driver['active_material'],
            surface_temperature_k=surface, ocean_fraction=ocean,
            orbit_before=state.thermal.orbit, orbit_after=thermal.state.orbit,
            weak_plane_slip_available=True)
        final = self._advance_trial(state, thermal, before_damage, faults.damage,
            requested_dt_myr=dt_myr, extra_state_fields={**common, 'fault_history':faults})
        final_tensile = principal_tensile(self.face_response(final)['stress_pa'])
        final = replace(final, fault_history=replace(final.fault_history,
            peak_tensile_pa=np.maximum(final.fault_history.peak_tensile_pa, final_tensile)))
        report = dict(final.last_report)
        report['weak_plane_slip']['activation_event'] = (None if activation is None else
            {**activation, 'future_volume_plastic_slip_implemented':True})
        report['fault_evolution'] = {**evolution, 'damage_coupling':DAMAGE_COUPLING,
            'damage_energy_change_j':final.mechanics.damage_energy_change_j-state.mechanics.damage_energy_change_j,
            'final_face_tensile_pa':final_tensile.tolist(),
            'combined_heat_mechanical_reservoir_closed':False, 'physical_banks_admitted':False,
            'predictor_driver':{
                'physical_vertices_m':vertices.tolist(), 'stress_pa':driver['stress_pa'].tolist(),
                'mean_temperature_k':driver['mean_temperature_k'].tolist(),
                'retained_fraction':retained.tolist(), 'active_material':driver['active_material'].tolist(),
                'surface_temperature_k':float(surface), 'ocean_fraction':float(ocean),
                'elastic_strain0':coefficient_arrays(predictor.mechanics)[0].tolist(),
                'elastic_strain1':coefficient_arrays(predictor.mechanics)[1].tolist(),
                'displacement_m':predictor.displacement_m.tolist(), 'damage':damage0.tolist(),
                'weak_plane_slip':predictor.last_report['weak_plane_slip']}}
        final = replace(final, last_report=report)
        self.validate_state(final)
        return final
