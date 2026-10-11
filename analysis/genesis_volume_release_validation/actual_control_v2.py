"""Source-forced private trajectories with source-born operative coordinates.

The accepted planet is publicly replayed once.  Its material history remains
unchanged; the explicit cohesive law and finite seed are computational inputs.
The added displacement and Maxwell memory start at zero at seed insertion.
Their native coordinate journal avoids lossy recovery from rounded full18
expansion views.  No mature material history is initialized again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tectonics.genesis_volume_creation_law import ProspectiveCreationLaw
from tectonics.genesis_volume_dynamics import coefficient_arrays
from tectonics.genesis_volume_embedded_partition import prepare_embedded_partition
from tectonics.genesis_volume_enriched_memory import enriched_memory_to_dict
from tectonics.genesis_volume_finite_seed import prepare_volume_finite_seed
from tectonics.genesis_volume_nucleation import bind_volume_nucleation_proposal
from tectonics.genesis_volume_release_history import (
    make_native_release_birth, native_release_state_to_dict, restart_native_release_state)
from tectonics.genesis_volume_release_solve_native import (
    solve_native_release_work, volume_native_release_solve_to_dict)
from tectonics.genesis_volume_release_work import (
    prepare_volume_release_work, volume_release_work_to_dict, evaluate_native_release_work,
    volume_release_work_response_to_dict, replay_native_release_work_response)
from tectonics.genesis_volume_seed_memory import prepare_source_seed_memory, restart_source_seed_memory
from tectonics.genesis_volume_seed_reaction import prepare_volume_seed_reaction
from tectonics.genesis_volume_slip_checkpoint import load_volume_slip_checkpoint, save_volume_slip_checkpoint

PARENT = ROOT/'analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json'
PREVIOUS = ROOT/'analysis/genesis_volume_memory_validation/evidence/actual_control/control_fixture.json'
PARENT_RAW = '62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7'
PARENT_SELF = '5d282ceeccc8d76acc38427d08a7e65575fddb509fc4b93115cbe6d318e91349'
FORMATION = 'cd79ee2a5139c4b91abdb1b4714a761c9153cc9f4490f5aa63fa2facba458c89'
PREVIOUS_RAW = 'a98ff7cc89aeb59eb004c05b8ac87f75f2e39a4f10d4454fdf93c288153c9c0e'
FORMAT = 'genesis-volume-native-release-work-control-0.2'
SCOPE = 'source_born_native_coordinate_private_fixed_material_work_gate_no_physical_release'


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def pin(path):
    h = hashlib.sha256(); size = 0
    with Path(path).open('rb') as stream:
        while data := stream.read(1 << 20):
            h.update(data); size += len(data)
    return {'bytes': size, 'raw_sha256': h.hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sealed(body, key):
    return {**body, key: hashlib.sha256(canonical(body)).hexdigest()}


def trajectory_record(context, before, before_native, old_maximum, response, *, label, solver=None):
    raw = volume_release_work_response_to_dict(response)
    before_raw, after_raw = enriched_memory_to_dict(before), enriched_memory_to_dict(response.after_memory)
    # Full journal replay retains the original material and memory, including
    # the two-step continuation.  Neither replay advances the accepted planet.
    replayed_memory = restart_source_seed_memory(context.birth, after_raw)
    require(enriched_memory_to_dict(replayed_memory) == after_raw, 'Source memory restart differs')
    before_native_raw = native_release_state_to_dict(before_native)
    after_native_raw = native_release_state_to_dict(response.after_native)
    replayed_native = restart_native_release_state(context, after_native_raw, replayed_memory)
    require(native_release_state_to_dict(replayed_native) == after_native_raw,
        'Source-born operative coordinate restart differs')
    replayed = replay_native_release_work_response(context, raw, before, before_native, old_maximum)
    require(volume_release_work_response_to_dict(replayed) == raw, 'Work trajectory restart differs')
    return dict(label=label, before_memory=before_raw, after_memory=after_raw,
        before_native=before_native_raw, after_native=after_native_raw,
        old_maximum_separation_m=np.asarray(old_maximum).tolist(), response=raw,
        source_memory_restart_exact=True, source_work_restart_exact=True,
        source_native_restart_exact=True,
        solver=None if solver is None else volume_native_release_solve_to_dict(solver))


def run(output):
    output.mkdir(exist_ok=False)
    parent_bytes = PARENT.read_bytes()
    require(hashlib.sha256(parent_bytes).hexdigest() == PARENT_RAW, 'Genuine parent raw pin differs')
    parent = json.loads(parent_bytes)
    require(parent['checkpoint_sha256'] == PARENT_SELF, 'Genuine parent self pin differs')
    require(pin(PREVIOUS)['raw_sha256'] == PREVIOUS_RAW, 'Previous memory fixture raw pin differs')
    previous = json.loads(PREVIOUS.read_bytes())
    require(previous['fixture_sha256'] == hashlib.sha256(canonical({k:v for k,v in previous.items()
        if k != 'fixture_sha256'})).hexdigest(), 'Previous memory fixture seal differs')
    require(previous['parent_raw_sha256'] == PARENT_RAW, 'Previous memory source differs')
    prior = previous['cases'][0]
    d = prior['seed_definition']
    # The only public planetary restore performed by this control.
    model, state = load_volume_slip_checkpoint(PARENT, expected_source_checkpoint_sha256=FORMATION)
    require(len(state.geometry.cohorts) == 800 and len(state.accepted_step_sizes_myr) == 4,
        'Genuine material/accepted journal differs')
    save_volume_slip_checkpoint(output/'parent_before.json', model, state)
    require((output/'parent_before.json').read_bytes() == parent_bytes, 'Genuine source replay differs')
    assembly, basal = model._assembly(state.geometry, state.fault_history.damage)
    e0, e1 = coefficient_arrays(state.mechanics)
    seed = prepare_volume_finite_seed(state.geometry, state.displacement_m, d['normal_xyz'],
        d['offset_m'], d['seed_center_world_m'], d['seed_radius_m'],
        cohort_id=d['original_cohort_id'], quadrature_order=8)
    p = prepare_embedded_partition(assembly, seed, e0, e1, basal=basal, quadrature_order=12)
    require(p.binding_sha256 == prior['embedded_partition']['embedded_partition_sha256'],
        'Existing finite compact partition was changed')
    reaction = prepare_volume_seed_reaction(model, state, seed, partition=p, quadrature_order=12)
    birth = prepare_source_seed_memory(model, state, seed, partition=p, reaction=reaction, quadrature_order=12)
    proposal = bind_volume_nucleation_proposal(model, state, law=None)
    # Explicit mathematical probe parameters, never recovered toughness or a
    # newly selected Genesis fracture.  This law has a resolvable virgin branch.
    law = ProspectiveCreationLaw(1e9, 1e10, 1e12, 1e12)
    context = prepare_volume_release_work(birth, seed, law, source_nucleation_proposal=proposal,
        parent_velocity_m_s=state.velocity_m_s)
    raw_context = volume_release_work_to_dict(context)
    require(context.modes.rank == 15 and context.modes.nullity == 3,
        'Complete compact jump kernel/range differs')
    require('accepted_Genesis_has_no_new_source_activation' in context.physical_refusals,
        'Genuine source activation was invented')
    print(json.dumps(dict(status='SOURCE_READY', cohort=d['original_cohort_id'],
        rank=context.modes.rank, nullity=context.modes.nullity)), flush=True)
    trajectories = []
    before = birth.initial_memory
    before_native = make_native_release_birth(context)
    history = context.cohesive.seed.initial_maximum_separation_m
    summaries = []
    for index, dt in enumerate((1e-4, 7e-5), 1):
        result = solve_native_release_work(context, before, before_native, history, before.time_myr+dt)
        response = result.response
        require(response.balanced_trajectory, 'Calculated source-forced trajectory is unbalanced')
        record = trajectory_record(context, before, before_native, history, response,
            label=f'source_forced_virgin_step_{index}', solver=result)
        trajectories.append(record)
        raw = record['response']
        summaries.append(dict(label=record['label'], balanced_trajectory=raw['balanced_trajectory'],
            projected_momentum_relative_error=raw['projected_momentum_relative_error'],
            external_work_j=raw['external_work_j'], work_balance_residual_j=raw['work_balance_residual_j'],
            bulk_work_decomposition_j=raw['bulk_work_decomposition_j'], cohesive_work_j=raw['cohesive_work_j'],
            cohesive_endpoint_work_agrees_at_integral_precision=raw['cohesive_endpoint_work_agrees_at_integral_precision'],
            source_memory_restart_exact=True, source_work_restart_exact=True, source_native_restart_exact=True))
        before, before_native = response.after_memory, response.after_native
        history = response.cohesive_response.new_maximum_separation_m
        print(json.dumps(dict(status='TRAJECTORY_PASS', **summaries[-1])), flush=True)
    # Independent non-equilibrium trajectory must retain its work identity
    # while refusing momentum admission; external force is still the source F.
    dw = 1e-4*np.sin(.37*(1+np.arange(context.modes.rank)))
    native_birth = make_native_release_birth(context)
    probe = evaluate_native_release_work(context, birth.initial_memory, native_birth,
        context.cohesive.seed.initial_maximum_separation_m, np.r_[np.zeros(p.parent_ndof), dw],
        birth.initial_memory.time_myr+1e-4)
    require(not probe.balanced_trajectory, 'Prescribed unbalanced path was admitted')
    trajectories.append(trajectory_record(context, birth.initial_memory, native_birth,
        context.cohesive.seed.initial_maximum_separation_m, probe, label='prescribed_unbalanced_path'))
    save_volume_slip_checkpoint(output/'parent_after.json', model, state)
    require((output/'parent_after.json').read_bytes() == parent_bytes == PARENT.read_bytes(),
        'Accepted Genesis or original anchor changed')
    flags = dict(physical_creation_admitted=False, physical_release_admitted=False, funding_admitted=False,
        history_reinitialized=False, accepted_planet_memory_changed=False, automated_goal_enabled=False,
        point_one_complete=False, new_Genesis_time_steps=0, new_material_cohorts=0,
        creation_work_debited_from_Genesis_j=0.)
    fixture = sealed(dict(format=FORMAT, scope=SCOPE, parent_raw_sha256=PARENT_RAW,
        parent_checkpoint_sha256=PARENT_SELF, formation_checkpoint_sha256=FORMATION,
        previous_memory_fixture_raw_sha256=PREVIOUS_RAW,
        previous_memory_fixture_sha256=previous['fixture_sha256'],
        operative_coordinate_history_starts_at_source_birth_zero=True,
        operative_coordinate_history_recovered_from_full18=False,
        full18_expansion_is_rounded_arithmetic_witness_only=True,
        cases=[dict(name=prior['name'], previous_case_name=prior['name'], context=raw_context,
            trajectories=trajectories)], **flags), 'fixture_sha256')
    (output/'control_fixture.json').write_bytes(canonical(fixture)+b'\n')
    report = sealed(dict(format=FORMAT+'-report', scope=SCOPE, status='PASS',
        parent_raw_sha256=PARENT_RAW, parent_checkpoint_sha256=PARENT_SELF,
        formation_checkpoint_sha256=FORMATION, fixture_sha256=fixture['fixture_sha256'],
        previous_memory_fixture_pin=pin(PREVIOUS), source_time_myr=state.time_myr, cohort_count=800,
        public_planetary_restore_count=1, whole_parent_before_after_byte_identical=True,
        complete_jump_kernel_tied=True, surface_rank=15, tied_nullity=3,
        computational_source_forced_steps=2, prescribed_unbalanced_path_refused=True,
        operative_coordinate_history_starts_at_source_birth_zero=True,
        operative_coordinate_history_recovered_from_full18=False,
        full18_expansion_is_rounded_arithmetic_witness_only=True,
        trajectories=summaries, physical_refusals=list(context.physical_refusals), **flags), 'report_sha256')
    (output/'report.json').write_bytes(canonical(report)+b'\n')
    print(json.dumps(dict(status='PASS', report=pin(output/'report.json'),
        self_sha256=report['report_sha256'], fixture=pin(output/'control_fixture.json'),
        fixture_sha256=fixture['fixture_sha256'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
