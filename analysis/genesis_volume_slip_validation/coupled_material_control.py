"""Explicit synthetic active coupled control on an unchanged Genesis substrate.

This is a mechanical fixture with a prescribed weak-plane/water seed and
additional dead body force. It is never an authentic planetary continuation,
activation result, or checkpoint. Original geometry, six-component elastic
memory, thermal state, birth history and source bytes are reused unchanged.
Only the declared fixture seed/forcing/continuation drag differ.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tectonics.genesis import SECONDS_PER_MYR
from tectonics.genesis_contact import ContactParameters
from tectonics.genesis_volume_dynamics import coefficient_arrays
from tectonics.genesis_volume_fault_dynamics import _face_frames
from tectonics.genesis_volume_fault_slip import (
    assemble_volume_fault_average_force_tangent, evaluate_volume_fault_slip,
    prepare_volume_fault_slip)
from tectonics.genesis_volume_operator import _immutable
from tectonics.genesis_volume_slip_checkpoint import (
    _slip_state_payload, volume_slip_checkpoint_to_dict)
from tectonics.genesis_volume_slip_dynamics import VolumeSlipModel, _lift_plane, _vertices


FORMAT = "genesis-volume-slip-coupled-fixture-0.1"
SCOPE = "declared_synthetic_active_coupled_volume_test_not_actual_genesis"
FORCE_DEFINITION = ("fixed_dead_body_force=(source_K+source_reference_basal_D/dt_seconds)"
                    "@declared_affine_shear_compression_displacement;"
                    "added_to_unchanged_source_basal_load_without_prestress_cancellation")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class DeclaredSeedAndBodyForceModel(VolumeSlipModel):
    """Test-only subclass, intentionally refused by the authentic save API."""

    def __init__(self, packet, *, dt_myr=1e-6, face_index=0,
                 basal_drag_pa_s_m=1e7, shear_strain=.0024,
                 compression_strain=.001, seed_water=.25):
        self.synthetic_force_n = None
        self.synthetic_seed = None
        self.fixture_dt_myr = float(dt_myr)
        self.fixture_face_index = int(face_index)
        self.fixture_shear_strain = float(shear_strain)
        self.fixture_compression_strain = float(compression_strain)
        self.fixture_seed_water = float(seed_water)
        super().__init__(packet, ContactParameters(basal_drag_pa_s_m=basal_drag_pa_s_m))
        self.literal_initial_fault_history = self.initial_fault_history

    def _assembly(self, geometry, damage):
        operator, basal = super()._assembly(geometry, damage)
        if self.synthetic_force_n is not None:
            basal = replace(basal, external_force_n=_immutable(
                basal.external_force_n+self.synthetic_force_n))
        return operator, basal

    def seed_unchanged_mechanical_substrate(self):
        """Declare the fixture edits only after authentic initial-state validation."""
        if self.synthetic_seed is not None:
            raise ValueError("The synthetic fixture seed cannot be applied twice")
        literal = super().initial_state()
        h = literal.fault_history
        face = self.fixture_face_index
        require(0 <= face < len(h.damage), "Synthetic seed face outside actual material coverage")
        require(not h.fault_active.any(), "Fixture requires the literal source to have no active planes")
        active, plane, time, water = (h.fault_active.copy(), h.plane_normal.copy(),
                                    h.activation_time_myr.copy(), h.water_access.copy())
        active[face], plane[face], time[face], water[face] = True, (1., 0.), literal.time_myr, self.fixture_seed_water
        seed_history = replace(h, fault_active=active, plane_normal=plane,
                               activation_time_myr=time, water_access=water)
        world_n, world_m = _lift_plane(plane, active, _vertices(literal), literal.geometry.faces)
        seeded = replace(literal, fault_history=seed_history,
                         plane_normal_world=world_n, slip_direction_world=world_m)
        # This test-specific initial baseline is explicit in the fixture.
        # No source packet/checkpoint is edited or written under an authentic format.
        self.initial_fault_history = seed_history
        self.synthetic_seed = dict(face_index=face, plane_normal2=[1., 0.],
            water_access=self.fixture_seed_water, activation_time_myr=literal.time_myr,
            fault_active=True, origin="prescribed_synthetic_seed_not_calculated_Genesis_activation")
        self.validate_state(seeded)
        operator, basal = self._assembly(literal.geometry, h.damage)
        n, m = world_n[face], world_m[face]
        imposed_strain_tensor = (-self.fixture_compression_strain*np.outer(n, n)
            +.5*self.fixture_shear_strain*(np.outer(m, n)+np.outer(n, m)))
        displacement = np.asarray(literal.geometry.vertices_m)@imposed_strain_tensor.T
        target = np.column_stack((displacement, displacement)).ravel()
        dt_seconds = self.fixture_dt_myr*SECONDS_PER_MYR
        self.synthetic_force_n = _immutable((operator.stiffness+basal.basal_drag_matrix/dt_seconds)@target)
        self.synthetic_strain_tensor_world = _immutable(imposed_strain_tensor)
        self.synthetic_target_displacement_m = _immutable(target)
        # Seeding altered no coefficient, geometry, mass, thermal or birth field.
        require(literal.geometry is seeded.geometry and literal.mechanics is seeded.mechanics
                and literal.history is seeded.history and literal.thermal is seeded.thermal,
                "Synthetic seed replaced the inherited mechanical/formation substrate")
        return literal, seeded

    def _audit_input(self, state, after, loading, increment, basal):
        result = super()._audit_input(state, after, loading, increment, basal)
        require(self.synthetic_force_n is not None, "Synthetic forcing is missing")
        result.update(synthetic_additional_external_force_n=self.synthetic_force_n.tolist(),
            total_external_force_n=basal.external_force_n.tolist(),
            source_basal_external_force_n=(basal.external_force_n-self.synthetic_force_n).tolist(),
            forcing_scope=SCOPE)
        return result

    def _mechanical_trial(self, state, after, basal, loading, dt, extra_state_fields):
        result = super()._mechanical_trial(state, after, basal, loading, dt, extra_state_fields)
        increment, local, mean0, mean1, trial_report, trial_fields = result
        h = extra_state_fields["fault_history"]
        prepared = prepare_volume_fault_slip(loading, face_indices=after.face_indices,
            basis_xyz=after.basis_xyz,
            plane_normal_world=extra_state_fields["plane_normal_world"],
            slip_direction_world=extra_state_fields["slip_direction_world"],
            active=h.fault_active, damage=h.damage, water=state.fault_history.water_access,
            parameters=self.fault_parameters)
        local_increment = increment[after.dofs]
        inc0 = np.einsum("cqai,ci->cqa", after.B0, local_increment)
        inc1 = np.einsum("cqai,ci->cqa", after.B1, local_increment)
        response = evaluate_volume_fault_slip(prepared, inc0, inc1)
        force = assemble_volume_fault_average_force_tangent(after, prepared, response)
        difference = force.tangent-force.tangent.T
        relative = float(np.linalg.norm(difference.data)/max(np.linalg.norm(force.tangent.data), 1.))
        require(np.array_equal(response.loading_response.strain0, local.strain0),
                "Fixture tangent witness used a different accepted material response")
        report = {**trial_report, "synthetic_fixture_tangent_asymmetry_relative_norm":relative,
                  "synthetic_fixture_scope":SCOPE}
        return increment, local, mean0, mean1, report, trial_fields


def synthetic_runtime_view(model, state):
    return dict(format="synthetic-volume-slip-runtime-view-0.1", scope=SCOPE,
        source_packet=model.source_packet,
        source_checkpoint_sha256=state.geometry.source_checkpoint_sha256,
        runtime_state=_slip_state_payload(state), authentic_checkpoint_claimed=False)


def state_facts(state):
    m = state.mechanics
    losses = state.fault_friction_dissipation_j+state.fault_viscous_dissipation_j
    expected = (m.initial_energy_j+state.initial_kinetic_energy_j+state.external_work_j
        +m.thermal_work_j-m.viscous_dissipation_j-state.basal_dissipation_j-losses
        +m.damage_energy_change_j-m.remelt_energy_removed_j+state.phase_kinetic_energy_change_j)
    return dict(time_myr=state.time_myr, accepted_steps=m.accepted_steps,
        live_cohort_count=len(m.cohorts), elastic_energy_j=m.elastic_energy_j,
        kinetic_energy_j=state.kinetic_energy_j, external_work_j=state.external_work_j,
        basal_dissipation_j=state.basal_dissipation_j,
        physical_mechanical_work_j=state.physical_mechanical_work_j,
        net_constitutive_driver_work_j=m.mechanical_work_j,
        fault_friction_dissipation_j=state.fault_friction_dissipation_j,
        fault_viscous_dissipation_j=state.fault_viscous_dissipation_j,
        constitutive_total_net_split_residual_j=state.physical_mechanical_work_j-m.mechanical_work_j-losses,
        global_balance_residual_j=m.elastic_energy_j+state.kinetic_energy_j-expected,
        weak_plane_count=int(state.fault_history.fault_active.sum()),
        max_shear_increment=float(np.max(np.abs(state.fault_history.last_shear_increment))))


def build_control(source_packet, *, dt_years=1., **model_options):
    model = DeclaredSeedAndBodyForceModel(source_packet,
        dt_myr=dt_years/1e6, **model_options)
    literal, seeded = model.seed_unchanged_mechanical_substrate()
    first = model.step(seeded, model.fixture_dt_myr)
    second = model.step(first, model.fixture_dt_myr)
    for before, after in ((seeded, first), (first, second)):
        model.validate_state(after)
        report = after.last_report["weak_plane_slip"]
        require(report["friction_work_j"] > 0 and report["viscous_fault_work_j"] > 0,
                "Synthetic active interval did not exercise both finite-width slip losses")
        require(after.last_report["synthetic_fixture_tangent_asymmetry_relative_norm"] > 0,
                "Synthetic active interval did not exercise nonassociated mean-force tangent")
        require(report["newton_iterations"] > 0, "Synthetic coupled Newton path was not exercised")
        require(np.array_equal(report["water"], before.fault_history.water_access),
                "Accepted-start lagged water changed during the final solve")
        require(np.array_equal(after.last_report["fault_evolution"]["predictor_driver"]
            ["weak_plane_slip"]["water"], before.fault_history.water_access),
                "The predictor used a different lagged water value")
        require(not np.array_equal(report["water"], report["endpoint_water_access"]),
                "The fixture failed to distinguish lagged and updated endpoint water")
        require(after.history.intervals[:len(before.history.intervals)] == before.history.intervals,
                "A coupled interval replaced its accepted formation history")
    checkpoint_refused = False
    try:
        volume_slip_checkpoint_to_dict(model, second)
    except ValueError:
        checkpoint_refused = True
    require(checkpoint_refused, "Synthetic model was admitted to an authentic checkpoint")
    packets = dict(initial_packet=synthetic_runtime_view(model, seeded),
                   after_first_packet=synthetic_runtime_view(model, first),
                   after_second_packet=synthetic_runtime_view(model, second))
    declaration = dict(format=FORMAT, scope=SCOPE,
        synthetic_forcing_and_seed_not_planetary_continuation=True,
        actual_weak_plane_activation_observed=False, sharp_banks_created=False,
        actual_planet_activation_claimed=False, physical_fault_contact_admission_claimed=False,
        original_point_1_completed=False,
        genuine_planetary_continuation_claimed=False, authentic_checkpoint_claimed=False,
        actual_source_replaced=False, initial_memory_unchanged=True,
        initial_geometry_thermal_and_birth_unchanged=True,
        synthetic_seed=model.synthetic_seed,
        synthetic_force_n=model.synthetic_force_n.tolist(), force_definition=FORCE_DEFINITION,
        synthetic_continuation_basal_drag_pa_s_m=model.contact_parameters.basal_drag_pa_s_m,
        synthetic_strain_tensor_world=model.synthetic_strain_tensor_world.tolist(),
        synthetic_target_displacement_m=model.synthetic_target_displacement_m.tolist(),
        requested_dt_years=float(dt_years), accepted_interval_count=2,
        source_fault_material_parameters_unchanged=True,
        producer_result=dict(status="PASS", checkpoint_api_refused_synthetic_model=checkpoint_refused,
            states={"literal_substrate":state_facts(literal), "seeded_initial":state_facts(seeded),
                    "after_first":state_facts(first), "after_second":state_facts(second)}),
        **packets)
    return model, literal, seeded, first, second, declaration


def run(source_path, output, dt_years=1.):
    source_path, output = Path(source_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source_bytes = source_path.read_bytes()
    result = build_control(json.loads(source_bytes), dt_years=dt_years)
    fixture = result[-1]
    fixture.update(source_file_sha256=hashlib.sha256(source_bytes).hexdigest(),
                   source_file_bytes=len(source_bytes), source_rawbytes_preserved=True)
    (output/"source_material_export.json").write_bytes(source_bytes)
    fixture["fixture_sha256"] = hashlib.sha256(canonical(fixture)).hexdigest()
    (output/"coupled_fixture.json").write_text(json.dumps(fixture, sort_keys=True,
        indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    require(source_path.read_bytes() == source_bytes, "Immutable authentic source bytes were changed")
    summary = dict(format=FORMAT, scope=SCOPE, fixture_sha256=fixture["fixture_sha256"],
                   source_file_sha256=fixture["source_file_sha256"], **fixture["producer_result"])
    (output/"control_report.json").write_text(json.dumps(summary, sort_keys=True, indent=2,
        allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT/
        "analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt-years", type=float, default=1.)
    args = parser.parse_args()
    require(np.isfinite(args.dt_years) and args.dt_years > 0, "Fixture interval must be positive")
    run(args.source, args.output, args.dt_years)


if __name__ == "__main__":
    main()
