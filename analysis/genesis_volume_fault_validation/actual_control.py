"""Continue the real formation source with calculated volume damage.

The preceding frozen-damage control stays immutable. This is a new bounded
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
from tectonics.genesis_volume_fault_checkpoint import (
    load_volume_fault_checkpoint, save_volume_fault_checkpoint,
    volume_fault_checkpoint_to_dict)
from tectonics.genesis_volume_fault_dynamics import VolumeFaultModel


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
            "An unsupported active plane was silently accepted")
    require(np.array_equal(state.thermal.damage, model.initial_fault_history.damage),
            "The separate inherited thermal baseline was changed")
    audit.update(max_damage=float(state.fault_history.damage.max()),
        max_water_access=float(state.fault_history.water_access.max()),
        max_candidate_age_myr=float(state.fault_history.fault_candidate_age_myr.max()),
        active_weak_plane_count=int(state.fault_history.fault_active.sum()),
        cumulative_damage_energy_change_j=state.mechanics.damage_energy_change_j,
        inherited_fault_friction_work_j=state.fault_history.friction_work_j,
        inherited_fault_viscous_work_j=state.fault_history.viscous_fault_work_j)
    return audit


def run(source_path, output, dt_years):
    output.mkdir(parents=True, exist_ok=False)
    source_bytes = source_path.read_bytes()
    packet = json.loads(source_bytes)
    model = VolumeFaultModel(packet)
    initial = model.initial_state()
    audits = {"initial": audit_state(model, initial)}
    save_volume_fault_checkpoint(output/'initial.json', model, initial)
    first = model.step(initial, dt_years/1e6)
    audits["after_first"] = audit_state(model, first)
    save_volume_fault_checkpoint(output/'after_first.json', model, first)
    direct = model.step(first, dt_years/1e6)
    audits["direct_after_second"] = audit_state(model, direct)
    direct_packet = save_volume_fault_checkpoint(output/'direct_after_second.json', model, direct)
    resumed_model, restored = load_volume_fault_checkpoint(output/'after_first.json',
        expected_source_checkpoint_sha256=packet["source"]["checkpoint_sha256"])
    require(canonical(volume_fault_checkpoint_to_dict(resumed_model, restored)) ==
        canonical(json.loads((output/'after_first.json').read_text())), "First replay differs")
    resumed = resumed_model.step(restored, dt_years/1e6)
    audits["resumed_after_second"] = audit_state(resumed_model, resumed)
    resumed_packet = save_volume_fault_checkpoint(output/'resumed_after_second.json', resumed_model, resumed)
    require(canonical(direct_packet) == canonical(resumed_packet), "Full direct/restart differs")
    require((output/'direct_after_second.json').read_bytes() ==
            (output/'resumed_after_second.json').read_bytes(), "Raw saved packets differ")
    require(source_path.read_bytes() == source_bytes, "Immutable material source was changed")
    require(not np.array_equal(initial.fault_history.damage, direct.fault_history.damage),
            "The volume weakening history remained frozen")
    body = dict(format='actual-genesis-volume-fault-evolution-control-0.1', status='PASS',
        source_file=str(source_path.resolve()), source_file_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_checkpoint_sha256=packet['source']['checkpoint_sha256'],
        unchanged_source_forcing_and_activation_parameters=True,
        no_replacement_initialization=True,
        starts_at_same_formation_source_as_prior_frozen_control=True,
        prior_frozen_control_rewritten=False,
        requested_dt_years=dt_years, state_audits=audits,
        direct_resumed_entire_packets_byte_equal=True,
        checkpoint_loader_replays_every_predictor_and_final_solve=True,
        actual_weak_plane_activation_observed=False,
        sharp_physical_banks_created=False, point_one_complete=False,
        whole_heat_mechanical_closure_claimed=False,
        old_reverse_C5_admission_claimed=False,
        scope='real_formation_source_connected_volume_damage_and_persistence',
        limitations=['No source active weak plane or sharp-cut witness',
            'Next actual weak-plane gate refuses unsupported 3D plastic advancement',
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
