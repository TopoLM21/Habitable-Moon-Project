"""Continue the real formation source with calculated volume damage.

The preceding frozen-damage and damage-evolving controls stay immutable. This is a new bounded
calculation from the same authentic source, with unchanged forcing and fault
thresholds. It does not claim an active physical split or full heat closure.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.genesis_volume_continuation_validation.actual_control import (
    algebraic_state_audit, canonical, require)
from tectonics.genesis_volume_slip_checkpoint import (
    load_volume_slip_checkpoint, save_volume_slip_checkpoint,
    volume_slip_checkpoint_to_dict)
from tectonics.genesis_volume_slip_dynamics import VolumeSlipModel


def audit_state(model, state):
    # Only the mechanical ENERGY calculation sees the calculated current
    # damage. Actual thermal snapshots are unchanged and saved separately.
    view = SimpleNamespace(**{f.name: getattr(state, f.name) for f in fields(state)})
    view.thermal = SimpleNamespace(time_myr=state.thermal.time_myr,
                                  damage=state.fault_history.damage)
    audit = algebraic_state_audit(model, view)
    require(state.fault_history.time_myr == state.time_myr, "Fault clock differs")
    require(state.fault_history.accepted_steps == state.mechanics.accepted_steps,
            "The predictor was counted as a second accepted interval")
    require(not state.fault_history.fault_active.any(),
            "The actual inactive formation source unexpectedly activated")
    require(np.array_equal(state.thermal.damage, model.initial_fault_history.damage),
            "The separate inherited thermal baseline was changed")
    audit.update(max_damage=float(state.fault_history.damage.max()),
        max_water_access=float(state.fault_history.water_access.max()),
        max_candidate_age_myr=float(state.fault_history.fault_candidate_age_myr.max()),
        active_weak_plane_count=int(state.fault_history.fault_active.sum()),
        cumulative_damage_energy_change_j=state.mechanics.damage_energy_change_j,
        inherited_fault_friction_work_j=state.fault_history.friction_work_j,
        inherited_fault_viscous_work_j=state.fault_history.viscous_fault_work_j)
    audit.update(physical_mechanical_work_j=state.physical_mechanical_work_j,
        fault_friction_dissipation_j=state.fault_friction_dissipation_j,
        fault_viscous_dissipation_j=state.fault_viscous_dissipation_j)
    require(state.physical_mechanical_work_j == state.mechanics.mechanical_work_j,
            'Actual inactive total and net work differ')
    require(state.fault_friction_dissipation_j == state.fault_viscous_dissipation_j == 0,
            'Actual source acquired manufactured plastic loss')
    return audit


def run(source_path, output, dt_years):
    require(dt_years == 100., 'This fixed parity control requires two 100-year intervals')
    output.mkdir(parents=True, exist_ok=False)
    source_bytes = source_path.read_bytes()
    packet = json.loads(source_bytes)
    model = VolumeSlipModel(packet)
    initial = model.initial_state()
    audits = {"initial": audit_state(model, initial)}
    save_volume_slip_checkpoint(output/'initial.json', model, initial)
    first = model.step(initial, dt_years/1e6)
    audits["after_first"] = audit_state(model, first)
    save_volume_slip_checkpoint(output/'after_first.json', model, first)
    direct = model.step(first, dt_years/1e6)
    audits["direct_after_second"] = audit_state(model, direct)
    direct_packet = save_volume_slip_checkpoint(output/'direct_after_second.json', model, direct)
    resumed_model, restored = load_volume_slip_checkpoint(output/'after_first.json',
        expected_source_checkpoint_sha256=packet["source"]["checkpoint_sha256"])
    require(canonical(volume_slip_checkpoint_to_dict(resumed_model, restored)) ==
        canonical(json.loads((output/'after_first.json').read_text())), "First replay differs")
    resumed = resumed_model.step(restored, dt_years/1e6)
    audits["resumed_after_second"] = audit_state(resumed_model, resumed)
    resumed_packet = save_volume_slip_checkpoint(output/'resumed_after_second.json', resumed_model, resumed)
    require(canonical(direct_packet) == canonical(resumed_packet), "Full direct/restart differs")
    require((output/'direct_after_second.json').read_bytes() ==
            (output/'resumed_after_second.json').read_bytes(), "Raw saved packets differ")
    for name in ('initial.json', 'after_first.json', 'direct_after_second.json', 'resumed_after_second.json'):
        earlier = ROOT/'analysis/genesis_volume_fault_validation/evidence/control'/name
        old = json.loads(earlier.read_text())['runtime_state']
        new = json.loads((output/name).read_text())['runtime_state']
        for field in ('geometry', 'thermal', 'history', 'mechanics', 'fault_history',
                      'displacement_m', 'velocity_m_s', 'last_certificate'):
            require(canonical(old[field]) == canonical(new[field]),
                    f'Actual inactive physics differs from prior sealed control: {name}/{field}')
    require(source_path.read_bytes() == source_bytes, "Immutable material source was changed")
    require(not np.array_equal(initial.fault_history.damage, direct.fault_history.damage),
            "The volume weakening history remained frozen")
    body = dict(format='actual-genesis-volume-weak-plane-slip-control-0.1', status='PASS',
        source_file=str(source_path.resolve()), source_file_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_checkpoint_sha256=packet['source']['checkpoint_sha256'],
        unchanged_source_forcing_and_activation_parameters=True,
        no_replacement_initialization=True,
        starts_at_same_formation_source_as_prior_frozen_control=True,
        prospective_3d_weak_plane_slip_available=True,
        actual_source_activation_manufactured=False,
        all_actual_physics_equal_prior_sealed_damage_control=True,
        prior_frozen_control_rewritten=False,
        requested_dt_years=dt_years, state_audits=audits,
        direct_resumed_entire_packets_byte_equal=True,
        checkpoint_loader_replays_every_predictor_and_final_solve=True,
        actual_weak_plane_activation_observed=False,
        sharp_physical_banks_created=False, point_one_complete=False,
        whole_heat_mechanical_closure_claimed=False,
        old_reverse_C5_admission_claimed=False,
        scope='real_formation_source_connected_volume_with_available_average_traction_slip',
        limitations=['No source active weak plane or sharp-cut witness',
            'Actual control has no active material weak plane',
            'Physical banks/contact/underthrust path not admitted',
            'Discrete midpoint momenta and affine work path remain distinct',
            'Historical full natural metric/raft motion remains unavailable'])
    body['report_sha256'] = hashlib.sha256(canonical(body)).hexdigest()
    (output/'control_report.json').write_text(json.dumps(body, sort_keys=True, indent=2)+'\n')
    print(json.dumps(body, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT/
        'analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dt-years', type=float, default=100.)
    args = parser.parse_args()
    require(np.isfinite(args.dt_years) and args.dt_years > 0, 'Time increment must be positive')
    run(args.source, args.output, args.dt_years)


if __name__ == '__main__':
    main()
