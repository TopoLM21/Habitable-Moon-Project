"""Source-bound prospective volume mechanics, with discrete work balance.

The reference is the authentic current Genesis material, with its nonzero
cohort prestress. Discrete midpoint momenta and a linear displacement work
path are distinct: this is not a reconstruction of continuous past velocity,
a finite-strain constitutive law, or an admission of separated fault banks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import json
import math

import numpy as np
from scipy import linalg

from .genesis import SECONDS_PER_MYR
from .genesis_bulk_birth import bulk_birth_from_dict
from .genesis_bulk_export import validate_genesis_bulk_material
from .genesis_contact import ContactParameters
from .genesis_shell import ShellParameters
from .genesis_volume_basal import (assemble_genesis_volume_basal,
    conical_interval_mass_matrix, recover_source_last_accepted_vertex_rate)
from .genesis_volume_geometry import validate_genesis_volume_geometry
from .genesis_volume_loading import (RadialMoments, initial_volume_mechanics,
    prepare_volume_loading, evaluate_volume_loading, accept_volume_loading,
    volume_mechanics_energy, average_volume_loading_coefficients,
    average_volume_loading_stiffness_factors)
from .genesis_volume_operator import (assemble_genesis_volume_operator,
    assemble_updated_volume_operator, evaluate_genesis_volume_operator,
    algorithmic_stiffness, _immutable)
from .genesis_volume_temporal import certify_cone_homothetic_path

FORMAT = "genesis-source-bound-volume-dynamics-0.1"
KINEMATICS = "discrete_midpoint_momenta_with_linear_displacement_work_path"
ACTIVATION = "thermal_first_comoving_kinetic_mass_flux"
VELOCITY_CLOSURE = "last_accepted_source_surface_chord_rate_with_current_comoving_radial_extension"


def radial_moments(assembly):
    return RadialMoments(tuple(assembly.volume_m3),
        tuple(assembly.inverse_scale_mean), tuple(assembly.inverse_scale_variance))


def coefficient_arrays(mechanics):
    n = len(mechanics.cohorts)
    return (np.array([c.strain0 for c in mechanics.cohorts]).reshape(n, 3, 6),
            np.array([c.strain1 for c in mechanics.cohorts]).reshape(n, 3, 6))


@dataclass(frozen=True, slots=True)
class VolumeState:
    geometry: object
    thermal: object
    history: object
    mechanics: object
    displacement_m: np.ndarray
    velocity_m_s: np.ndarray
    contact_parameters_json: str
    initial_kinetic_energy_j: float
    kinetic_energy_j: float
    external_work_j: float = 0.
    basal_dissipation_j: float = 0.
    phase_kinetic_energy_change_j: float = 0.
    global_balance_residual_j: float = 0.
    last_report: dict | None = None
    last_certificate: object | None = None
    accepted_step_sizes_myr: tuple = ()
    velocity_closure: str = VELOCITY_CLOSURE
    kinematics: str = KINEMATICS
    activation_closure: str = ACTIVATION

    def __post_init__(self):
        for name in ("displacement_m", "velocity_m_s"):
            value = np.asarray(getattr(self, name), float)
            if value.ndim != 1 or not np.isfinite(value).all():
                raise ValueError("Volume coordinates/momenta must be finite full vectors")
            object.__setattr__(self, name, _immutable(value))
        object.__setattr__(self, "accepted_step_sizes_myr", tuple(self.accepted_step_sizes_myr))

    @property
    def time_myr(self):
        return self.mechanics.time_myr

    def __deepcopy__(self, memo):
        return self


class VolumeModel:
    """Actual-source continuation with thermal-first accepted mass changes."""

    state_type = VolumeState

    def __init__(self, source_packet, contact_parameters=None):
        validate_genesis_bulk_material(source_packet)
        # Keep the supplied authenticated DTO, including its saved scalar
        # reductions; validation's independent rebuild may differ by ulps.
        self.source_packet = json.loads(json.dumps(source_packet, allow_nan=False))
        self.shell_parameters = ShellParameters(**self.source_packet["source"]["parameters"]["shell"])
        self.elastic_strain_limit = float(self.source_packet["source"]["parameters"]["mobile"]["max_elastic_strain"])
        self.contact_parameters = contact_parameters or ContactParameters()
        if type(self.contact_parameters) is not ContactParameters:
            raise ValueError("Volume continuation requires explicit ContactParameters")
        self.contact_parameters.validate()
        self.contact_parameters_json = json.dumps(asdict(self.contact_parameters), sort_keys=True,
            separators=(",", ":"), allow_nan=False)
        self.source_operator = assemble_genesis_volume_operator(self.source_packet)
        self._cache = {}

    def _assembly(self, geometry, damage):
        # Cached typed objects are immutable; archive loads replay from source.
        key = id(geometry)
        saved = self._cache.get(key)
        if saved is not None and saved[0] is geometry and np.array_equal(saved[1], damage):
            return saved[2], saved[3]
        source_damage = np.asarray(self.source_packet["current_shell_memory"]["damage"])
        a = (self.source_operator if geometry is self.source_operator.geometry
             and np.array_equal(damage, source_damage) else
             assemble_updated_volume_operator(self.source_operator, geometry, damage))
        basal = assemble_genesis_volume_basal(a, self.contact_parameters)
        self._cache[key] = (geometry, np.array(damage, copy=True), a, basal)
        if len(self._cache) > 4:
            self._cache.pop(next(iter(self._cache)))
        return a, basal

    def initial_state(self):
        state = self._build_initial_state()
        self.validate_state(state)
        return state

    def _build_initial_state(self):
        from .genesis_volume_thermal import initial_volume_thermal_state
        a = self.source_operator
        thermal = initial_volume_thermal_state(self.source_packet)
        history = bulk_birth_from_dict(self.source_packet["bulk_birth_history"])
        records = self.source_packet["cohort_mechanics"]["state"]["cohorts"]
        temperatures = {r["id"]: r["temperature_k"] for r in records}
        damage = np.asarray(self.source_packet["current_shell_memory"]["damage"])
        mechanics = initial_volume_mechanics(a.source_identity, a.time_myr, a.cohort_ids,
            [temperatures[int(i)] for i in a.cohort_ids], a.source_elastic_strain,
            radial_moments(a), damage[a.face_indices], self.shell_parameters)
        if not math.isclose(mechanics.elastic_energy_j, a.source_elastic_energy_j, rel_tol=3e-12):
            raise ValueError("Volume import changed the actual source elastic energy")
        # Preserve the exact saved scalar after the independent tensor check;
        # subsequent contraction roundoff remains in the reported residual.
        mechanics = replace(mechanics, initial_energy_j=a.source_elastic_energy_j,
                            elastic_energy_j=a.source_elastic_energy_j)
        known_rate = recover_source_last_accepted_vertex_rate(self.source_packet)
        velocity_a = known_rate.velocity_m_s
        # Recovered accepted surface chord rate, with an explicit current
        # co-moving radial extension; continuous old 3-D motion is unavailable.
        velocity = np.column_stack((velocity_a, velocity_a)).ravel()
        _, basal = self._assembly(a.geometry, damage)
        kinetic = float(.5*velocity@(basal.mass_matrix@velocity))
        state = VolumeState(a.geometry, thermal, history, mechanics, np.zeros(a.ndof),
            velocity, self.contact_parameters_json, kinetic, kinetic)
        return state

    def _state_damage(self, state):
        return np.asarray(state.thermal.damage)

    def _additional_dissipation(self, state):
        """Extra runtime losses of an explicitly declared constitutive branch."""
        return 0.

    def validate_state(self, state):
        if type(state) is not self.state_type or state.contact_parameters_json != self.contact_parameters_json:
            raise ValueError("Volume state/continuation parameters mismatch")
        if (state.velocity_closure != VELOCITY_CLOSURE or state.kinematics != KINEMATICS
                or state.activation_closure != ACTIVATION):
            raise ValueError("Volume state changed its declared prospective closures")
        if (state.geometry.source_checkpoint_sha256 != self.source_operator.source_checkpoint_sha256
                or state.mechanics.source_identity != self.source_operator.source_identity):
            raise ValueError("Volume state changed its genuine formation source")
        validate_genesis_volume_geometry(state.geometry)
        clocks = (state.geometry.time_myr, state.thermal.time_myr, state.history.time_myr,
                  state.mechanics.time_myr)
        if len(set(clocks)) != 1:
            raise ValueError("Volume material, mechanics and thermal clocks disagree")
        if (state.displacement_m.shape != (self.source_operator.ndof,)
                or state.velocity_m_s.shape != state.displacement_m.shape):
            raise ValueError("Volume state lost physical nodal degrees of freedom")
        damage = self._state_damage(state)
        a, basal = self._assembly(state.geometry, damage)
        if tuple(int(i) for i in a.cohort_ids) != tuple(c.id for c in state.mechanics.cohorts):
            raise ValueError("Volume memory and actual live material IDs differ")
        energy = volume_mechanics_energy(state.mechanics, radial_moments(a),
            damage[a.face_indices], self.shell_parameters)
        kinetic = float(.5*state.velocity_m_s@(basal.mass_matrix@state.velocity_m_s))
        scale = max(energy, state.mechanics.initial_energy_j, kinetic, 1.)
        if (not math.isclose(energy, state.mechanics.elastic_energy_j, rel_tol=5e-12, abs_tol=1e-6)
                or abs(kinetic-state.kinetic_energy_j) > 5e-12*max(kinetic, 1.)):
            raise ValueError("Volume accepted elastic/kinetic energy disagrees with its fields")
        m = state.mechanics
        expected = (m.initial_energy_j+state.initial_kinetic_energy_j+state.external_work_j
            +m.thermal_work_j-m.viscous_dissipation_j-state.basal_dissipation_j
            -self._additional_dissipation(state)
            +m.damage_energy_change_j-m.remelt_energy_removed_j+state.phase_kinetic_energy_change_j)
        residual = energy+kinetic-expected
        if abs(residual-state.global_balance_residual_j) > 5e-11*scale or abs(residual) > 5e-10*scale:
            raise ValueError("Volume cumulative physical/discrete work balance mismatch")
        if len(state.accepted_step_sizes_myr) != state.mechanics.accepted_steps:
            raise ValueError("Volume accepted transition journal count differs")
        if state.last_certificate is not None:
            cert = state.last_certificate
            q = state.displacement_m.reshape(-1, 6)
            if (cert.source_identity != a.source_identity or float(cert.after_time_myr) != state.time_myr
                    or not cert.whole_clock_all_pair_noninterpenetration_certified
                    or not np.array_equal(np.asarray(cert.displacement_A_xyz, float), q[:, :3])
                    or not np.array_equal(np.asarray(cert.slope_P_xyz, float), q[:, 3:])
                    or float(cert.lambda_min) > min(c.inner_scale for c in state.geometry.cohorts)):
                raise ValueError("Volume geometry certificate is not bound to this accepted clock/source")
        return state

    def step(self, state, dt_myr):
        from .genesis_volume_thermal import prepare_volume_thermal_step
        self.validate_state(state)
        if isinstance(dt_myr, bool) or not math.isfinite(dt_myr) or dt_myr <= 0:
            raise ValueError("Volume step must advance actual Genesis time")
        thermal = prepare_volume_thermal_step(state.thermal, state.history, state.geometry,
            dt_myr, displacement_m=state.displacement_m)
        return self._advance_trial(state, thermal, self._state_damage(state),
            np.asarray(thermal.state.damage), requested_dt_myr=dt_myr)

    def _advance_trial(self, state, thermal, before_damage, after_damage, *,
                       requested_dt_myr, extra_state_fields=None):
        """One private solve from the original memory and a fixed thermal trial.

        The frozen-damage control and the prospective fault-evolution model
        share this calculation. A predictor does not publish any history or
        contribute work to a later accepted solve.
        """
        before, old_basal = self._assembly(state.geometry, before_damage)
        after, basal = self._assembly(thermal.geometry, after_damage)
        loading = prepare_volume_loading(state.mechanics, after_ids=after.cohort_ids,
            after_temperature_k=thermal.after_temperature_k, birth_temperature_k=thermal.birth_temperature_k,
            radial_moments=thermal.radial_moments, before_radial_moments=thermal.before_radial_moments,
            removed_radial_moments=thermal.removed_radial_moments,
            before_damage=before_damage[before.face_indices],
            after_damage=after_damage[after.face_indices], parameters=self.shell_parameters,
            after_time_myr=thermal.state.time_myr, before_temperature_k=thermal.before_temperature_k)
        dt = (thermal.state.time_myr-state.time_myr)*SECONDS_PER_MYR
        M, D = basal.mass_matrix, basal.basal_drag_matrix
        increment, response, mean0, mean1, trial_report, trial_fields = self._mechanical_trial(
            state, after, basal, loading, dt, extra_state_fields or {})
        mechanics = accept_volume_loading(loading, response)
        q = state.displacement_m+increment
        velocity = 2*increment/dt-state.velocity_m_s
        avg_force = evaluate_genesis_volume_operator(after, np.zeros(after.ndof),
            elastic_constant=mean0, elastic_inverse_scale=mean1).force
        end_response = evaluate_genesis_volume_operator(after, np.zeros(after.ndof),
            elastic_constant=response.strain0, elastic_inverse_scale=response.strain1)
        if end_response.maximum_elastic_principal_strain > self.elastic_strain_limit:
            raise ValueError("Volume step exceeds the inherited elastic strain limit")
        residual_vector = 2*(M@(increment-dt*state.velocity_m_s))/(dt*dt)+D@increment/dt+avg_force-basal.external_force_n
        force_scale = max(np.linalg.norm(basal.external_force_n), np.linalg.norm(avg_force), 1.)
        residual = float(np.linalg.norm(residual_vector)/force_scale)
        if residual > self.contact_parameters.equilibrium_tolerance:
            raise ValueError("Volume discrete momentum equation did not converge")
        kinetic = float(.5*velocity@(M@velocity))
        external = float(basal.external_force_n@increment)
        drag = float(increment@(D@increment)/dt)
        if drag < -1e-12*max(abs(external), 1.):
            raise ValueError("Physical basal drag produced negative dissipation")
        drag = max(drag, 0.)
        phase, born_kinetic, removed_kinetic = self._phase_kinetic_flux(before, after, state.velocity_m_s)
        certificate = self._certificate(state, after, q)
        global_expected = (mechanics.initial_energy_j+state.initial_kinetic_energy_j
            +state.external_work_j+external+mechanics.thermal_work_j-mechanics.viscous_dissipation_j
            -state.basal_dissipation_j-drag+mechanics.damage_energy_change_j
            -mechanics.remelt_energy_removed_j+state.phase_kinetic_energy_change_j+phase
            -self._trial_additional_dissipation(state, trial_fields))
        balance = mechanics.elastic_energy_j+kinetic-global_expected
        report = {"kinematics": KINEMATICS, "activation": ACTIVATION,
            "requested_dt_myr": float(requested_dt_myr), "dt_myr": thermal.state.time_myr-state.time_myr,
            "before_time_myr": state.time_myr, "after_time_myr": thermal.state.time_myr,
            "momentum_equation_relative_residual": residual,
            "end_static_force_relative_residual": float(np.linalg.norm(end_response.force-basal.external_force_n)/force_scale),
            "external_work_j": external, "basal_dissipation_j": drag,
            "mechanical_work_j": response.mechanical_work_j, "thermal_work_j": response.thermal_work_j,
            "viscous_dissipation_j": response.viscous_dissipation_j,
            "birth_kinetic_energy_j": born_kinetic, "remelt_kinetic_energy_j": removed_kinetic,
            "global_balance_residual_j": balance,
            "maximum_elastic_principal_strain": end_response.maximum_elastic_principal_strain,
            "live_cohort_count": len(after.cohort_ids), "live_mass_kg": basal.total_live_mass_kg,
            "whole_interval_all_pair_geometry_certified": True,
            "static_equilibrium_claimed": False, "old_C5_to_C1_admission_claimed": False,
            "full_historical_3d_recovery_claimed": False,
            "audit_input": self._audit_input(state, after, loading, increment, basal)}
        report.update(trial_report)
        state_fields = {**(extra_state_fields or {}), **trial_fields}
        accepted = self.state_type(thermal.geometry, thermal.state, thermal.history, mechanics, q, velocity,
            self.contact_parameters_json, state.initial_kinetic_energy_j, kinetic,
            state.external_work_j+external, state.basal_dissipation_j+drag,
            state.phase_kinetic_energy_change_j+phase, balance, report, certificate,
            state.accepted_step_sizes_myr+(float(requested_dt_myr),), **state_fields)
        self.validate_state(accepted)
        return accepted

    def _trial_additional_dissipation(self, state, trial_fields):
        return 0.

    def _mechanical_trial(self, state, after, basal, loading, dt, extra_state_fields):
        """Original linear trial; extensions must supply the same physical means."""
        zero = np.zeros_like(loading.old0)
        avg0, avg1 = average_volume_loading_coefficients(loading, zero, zero)
        theta = average_volume_loading_stiffness_factors(loading)
        initial_force = evaluate_genesis_volume_operator(after, np.zeros(after.ndof),
            elastic_constant=avg0, elastic_inverse_scale=avg1).force
        M, D = basal.mass_matrix, basal.basal_drag_matrix
        stiffness = algorithmic_stiffness(after, theta)
        tangent = stiffness+2*M/(dt*dt)+D/dt
        rhs = basal.external_force_n-initial_force+2*(M@state.velocity_m_s)/dt
        # Exact coordinate equilibration, no penalty, constraints or springs.
        diagonal = np.asarray(tangent.diagonal())
        if np.any(diagonal <= 0) or not np.isfinite(diagonal).all():
            raise ValueError("Volume physical tangent has unresolved inertial support")
        scale = 1/np.sqrt(diagonal)
        dense = tangent.toarray()*scale[:, None]*scale[None, :]
        solution = linalg.solve(dense, scale*rhs, assume_a="pos", check_finite=True)
        increment = scale*solution
        local = increment[after.dofs]
        inc0 = np.einsum("cqai,ci->cqa", after.B0, local)
        inc1 = np.einsum("cqai,ci->cqa", after.B1, local)
        response = evaluate_volume_loading(loading, inc0, inc1)
        mean0, mean1 = average_volume_loading_coefficients(loading, inc0, inc1)
        return increment, response, mean0, mean1, {}, {}

    def _phase_kinetic_flux(self, before, after, velocity):
        previous = {c.id: c for c in before.geometry.cohorts}
        current = {c.id: c for c in after.geometry.cohorts}
        born, removed = 0., 0.
        for i, c in enumerate(after.geometry.cohorts):
            if c.id not in previous:
                local = velocity[after.dofs[i]]
                mass = conical_interval_mass_matrix(c.top_triangle_m,
                    after.geometry.column_mass_kg[c.face_index], after.geometry.density_kg_m3, c.q_lo, c.live_q_hi)
                born += float(.5*local@mass@local)
        for i, c in enumerate(before.geometry.cohorts):
            retained = current.get(c.id)
            lo = c.q_lo if retained is None else retained.live_q_hi
            if lo < c.live_q_hi:
                local = velocity[before.dofs[i]]
                mass = conical_interval_mass_matrix(c.top_triangle_m,
                    before.geometry.column_mass_kg[c.face_index], before.geometry.density_kg_m3, lo, c.live_q_hi)
                removed += float(.5*local@mass@local)
        return born-removed, born, removed

    def _certificate(self, state, after, q):
        before_q, after_q = state.displacement_m.reshape(-1, 6), q.reshape(-1, 6)
        lower = min(c.inner_scale for c in (*state.geometry.cohorts, *after.geometry.cohorts))
        lower = float(np.nextafter(lower, 0.))
        # Tie the certified auxiliary domain to the actual analytic material
        # map using exact mass-q cubics, not a floating spatial tolerance.
        Q = Fraction
        vertices = tuple(tuple(Q(x) for x in row) for row in after.geometry.vertices_m)
        denominators = {}
        for index, face in enumerate(after.geometry.faces):
            x, y, z = (vertices[i] for i in face)
            determinant = (x[0]*(y[1]*z[2]-y[2]*z[1])
                -x[1]*(y[0]*z[2]-y[2]*z[0])+x[2]*(y[0]*z[1]-y[1]*z[0]))
            denominators[index] = Q(after.geometry.density_kg_m3)*determinant
        cubics = [1-6*Q(after.geometry.column_mass_kg[c.face_index])*Q(c.live_q_hi)
                  /denominators[c.face_index]
                  for c in (*state.geometry.cohorts, *after.geometry.cohorts)]
        bottom_cube = min(cubics)
        if bottom_cube <= 0:
            raise ValueError("Volume material reaches the exact conical centre")
        if Q(lower)**3 > bottom_cube:
            # Halving expands only an auxiliary proof domain; it neither
            # changes material placement nor supplies a contact tolerance.
            while Q(lower)**3 > bottom_cube:
                lower *= .5
                if lower <= 0:
                    raise ValueError("Volume exact positive material domain is unresolved")
        return certify_cone_homothetic_path(after.geometry.vertices_m, after.geometry.faces,
            after_q[:, :3], after_q[:, 3:], lambda_min=lower, source_identity=after.source_identity,
            before_time_myr=state.time_myr, after_time_myr=after.time_myr,
            before_displacement_A_xyz=before_q[:, :3], before_slope_P_xyz=before_q[:, 3:])

    def _audit_input(self, state, after, loading, increment, basal):
        return {"before_displacement_m": state.displacement_m.tolist(),
            "before_velocity_m_s": state.velocity_m_s.tolist(), "increment_m": increment.tolist(),
            "before_cohorts": [asdict(c) for c in state.geometry.cohorts],
            "after_cohorts": [asdict(c) for c in after.geometry.cohorts],
            "basis_xyz": after.basis_xyz.tolist(), "old0": loading.old0.tolist(), "old1": loading.old1.tolist(),
            "thermal0": loading.thermal0.tolist(), "maxwell_ratio": loading.ratio.tolist(),
            "elasticity_pa": loading.elasticity_pa.tolist(),
            "basal_traction_at_vertices_pa": basal.basal_traction_at_vertices_pa.tolist(),
            "basal_drag_pa_s_m": self.contact_parameters.basal_drag_pa_s_m}
