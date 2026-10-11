"""Run the volume continuation from the immutable actual Genesis source.

No mature test shell or replacement material initialization is constructed.
The algebraic audit below computes cohort mass, isotropic tensor energy and
cumulative constitutive work independently of the volume operator/loading
implementations. This controller is separate from the standalone file audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tectonics.genesis_volume_checkpoint import (  # noqa: E402
    load_volume_checkpoint, save_volume_checkpoint, volume_checkpoint_to_dict,
)
from tectonics.genesis_volume_dynamics import VolumeModel  # noqa: E402


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor(engineering):
    xx, yy, zz, xy, xz, yz = engineering
    return np.array(((xx, xy/2, xz/2), (xy/2, yy, yz/2), (xz/2, yz/2, zz)))


def algebraic_state_audit(model, state):
    """Use tensor Hooke energy, not production B/C/energy helper routines."""
    geometry = state.geometry
    material = {record.id: record for record in state.mechanics.cohorts}
    require(set(material) == {cohort.id for cohort in geometry.cohorts},
            "Independent live material identity match failed")
    E = model.shell_parameters.young_modulus_pa
    nu = model.shell_parameters.poisson_ratio
    mu = E/(2*(1+nu))
    lam = E*nu/((1+nu)*(1-2*nu))
    residual = model.shell_parameters.residual_stiffness
    energy_terms, mass_terms = [], []
    for cohort in geometry.cohorts:
        memory = material[cohort.id]
        a, b = cohort.outer_scale, cohort.inner_scale
        denominator = a*a+a*b+b*b
        mean = 1.5*(a+b)/denominator
        variance = .75*((a-b)/denominator)**2
        damage = float(state.thermal.damage[cohort.face_index])
        stiffness = residual+(1-residual)*(1-damage)**2
        for e0, e1 in zip(memory.strain0, memory.strain1):
            average = tensor(np.asarray(e0)+mean*np.asarray(e1))
            contrast = tensor(e1)
            density = (mu*np.sum(average*average)+.5*lam*np.trace(average)**2
                       +variance*(mu*np.sum(contrast*contrast)+.5*lam*np.trace(contrast)**2))
            energy_terms.append(cohort.volume_m3*stiffness*density/3)
        mass_terms.append(geometry.column_mass_kg[cohort.face_index]
                          *(cohort.live_q_hi-cohort.q_lo))
        require(math.isclose(cohort.mass_kg, mass_terms[-1], rel_tol=3e-15),
                "Independent cohort mass-q identity failed")
        require(math.isclose(cohort.volume_m3*geometry.density_kg_m3,
                             cohort.mass_kg, rel_tol=3e-15),
                "Independent cohort reference mass-volume identity failed")
    energy = math.fsum(energy_terms)
    scale = max(energy, state.mechanics.elastic_energy_j, 1.)
    require(abs(energy-state.mechanics.elastic_energy_j) <= 8e-14*scale,
            "Independent tensor energy differs from volume mechanics")
    memory = state.mechanics
    expected = (memory.initial_energy_j+memory.mechanical_work_j+memory.thermal_work_j
                -memory.viscous_dissipation_j+memory.damage_energy_change_j
                -memory.remelt_energy_removed_j)
    require(abs(energy-expected) <= 8e-12*max(scale, abs(expected)),
            "Independent cumulative constitutive work identity failed")
    mass = math.fsum(mass_terms)
    require(math.isclose(mass, float(state.history.live_mass_kg), rel_tol=8e-15),
            "Independent inherited live mass differs from birth history")
    clocks = (geometry.time_myr, state.thermal.time_myr, state.history.time_myr,
              state.mechanics.time_myr)
    require(all(clock == clocks[0] for clock in clocks),
            "Material, geometry, thermal and mechanics clocks differ")
    return dict(time_myr=clocks[0], live_cohort_count=len(material),
                independently_summed_live_mass_kg=mass,
                independent_tensor_elastic_energy_j=energy,
                production_elastic_energy_j=memory.elastic_energy_j,
                independent_constitutive_balance_residual_j=energy-expected,
                global_balance_residual_j=state.global_balance_residual_j,
                kinetic_energy_j=state.kinetic_energy_j,
                external_work_j=state.external_work_j,
                basal_dissipation_j=state.basal_dissipation_j,
                accepted_steps=len(state.accepted_step_sizes_myr))


def run(source_path, output, dt_myr):
    output.mkdir(parents=True, exist_ok=False)
    source_bytes = source_path.read_bytes()
    packet = json.loads(source_bytes)
    model = VolumeModel(packet)
    initial = model.initial_state()
    audit = {"initial": algebraic_state_audit(model, initial)}
    save_volume_checkpoint(output/"initial.json", model, initial)
    first = model.step(initial, dt_myr)
    audit["after_first"] = algebraic_state_audit(model, first)
    save_volume_checkpoint(output/"after_first.json", model, first)
    direct = model.step(first, dt_myr)
    audit["direct_after_second"] = algebraic_state_audit(model, direct)
    direct_packet = save_volume_checkpoint(output/"direct_after_second.json", model, direct)
    resumed_model, restored = load_volume_checkpoint(output/"after_first.json",
        expected_source_checkpoint_sha256=packet["source"]["checkpoint_sha256"])
    resumed = resumed_model.step(restored, dt_myr)
    audit["resumed_after_second"] = algebraic_state_audit(resumed_model, resumed)
    resumed_packet = save_volume_checkpoint(output/"resumed_after_second.json", resumed_model, resumed)
    require(canonical(direct_packet) == canonical(resumed_packet),
            "Direct and replay-loaded continuation checkpoints differ")
    require(canonical(volume_checkpoint_to_dict(resumed_model, restored)) ==
            canonical(json.loads((output/"after_first.json").read_text())),
            "Replay-loaded first-step checkpoint differs")
    current_source = source_path.read_bytes()
    require(current_source == source_bytes, "Immutable actual Genesis source changed")
    body = dict(format="actual-genesis-volume-continuation-control-0.1", status="PASS",
                actual_source_file=str(source_path.resolve()),
                actual_source_file_sha256=hashlib.sha256(source_bytes).hexdigest(),
                source_checkpoint_sha256=packet["source"]["checkpoint_sha256"],
                birth_source_identity=packet["birth_source_identity"],
                no_replacement_material_initialization=True,
                source_memory_energy_j=packet["cohort_mechanics"]["state"]["elastic_energy_j"],
                source_recomputed_three_dimensional_energy_j=packet[
                    "instantaneous_cohort_plane_stress_3d"]["elastic_energy_j"],
                source_face_mean_energy_j=packet["instantaneous_plane_stress_3d"]["elastic_energy_of_face_mean_j"],
                requested_dt_myr=dt_myr, state_audits=audit,
                direct_and_resumed_entire_packets_exact=True,
                checkpoint_loader_replayed_every_runtime_step=True,
                point_one_complete=False,
                combined_heat_and_mechanical_energy_closed=False,
                scope="connected_source_body_volume_increment_and_constitutive_work_control",
                limitations=["Historic pre-gate raft motion and natural metric remain unknown",
                             "Mechanical balance is discrete midpoint work; continuous dynamics and a combined heat reservoir are not certified",
                             "Connected body geometry is not continental plate ownership",
                             "This control does not establish reverse C5-to-C1 physical contact admission"])
    body["report_sha256"] = hashlib.sha256(canonical(body)).hexdigest()
    (output/"control_report.json").write_text(json.dumps(body, sort_keys=True, indent=2)+"\n")
    print(json.dumps(dict(status="PASS", output=str(output.resolve()),
                          source_checkpoint_sha256=body["source_checkpoint_sha256"],
                          direct_and_resumed_entire_packets_exact=True,
                          audits=audit), sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT/
        "analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt-years", type=float, default=1.)
    args = parser.parse_args()
    require(math.isfinite(args.dt_years) and args.dt_years > 0, "Control step must be positive")
    run(args.source, args.output, args.dt_years/1e6)


if __name__ == "__main__":
    main()
