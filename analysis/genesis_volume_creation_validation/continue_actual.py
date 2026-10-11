"""Bounded real Genesis continuation with prospective side-support snapshots.

The accepted parent is continued for two 100-year intervals with its existing
forcing, birth law and activation parameters. Creation parameters and actual
fracture surfaces remain unknown. Explicit edge supports merely sample the
newly calculated material; they never cut, release or reinitialize it.
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

from analysis.genesis_volume_creation_validation.actual_control import (
    PARENT_RAW_SHA256, FORMATION_SHA256, canonical, require, surface_packet,
)
from analysis.genesis_volume_slip_validation.actual_control import audit_state
from tectonics.genesis_volume_banks import prepare_volume_banks
from tectonics.genesis_volume_creation import (
    bind_volume_creation_context, creation_context_to_dict,
)
from tectonics.genesis_volume_fracture_surface import prepare_volume_fracture_surface
from tectonics.genesis_volume_slip_checkpoint import (
    load_volume_slip_checkpoint, save_volume_slip_checkpoint,
    volume_slip_checkpoint_to_dict,
)


FORMAT = "genesis-volume-creation-birth-growth-control-0.1"
SCOPE = "real_Genesis_material_birth_growth_with_prospective_uncut_side_support"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def no_creation_context(model, state):
    payload = creation_context_to_dict(bind_volume_creation_context(model, state))
    require(payload["physical_cut_status"] == "no_eligible_cut",
            "Bounded real continuation unexpectedly acquired an eligible cut")
    require(not payload["physical_creation_admitted"]
            and not payload["physical_release_admitted"],
            "Bounded continuation admitted uncalculated creation or release")
    require(payload["source_calculated_creation_law"] is None
            and payload["declared_prospective_creation_law"] is None,
            "Unknown real creation law was replaced by a fixture law")
    require(payload["creation_work_debited_from_Genesis_j"] == 0.
            and payload["historical_dissipation_available_for_creation_j"] == 0.
            and not payload["existing_damage_or_slip_work_reclassified"],
            "Real historical losses were reclassified as surface-creation credit")
    return payload


def snapshot(model, state, name, checkpoint_path, cut_edges, previous_ids):
    checkpoint_bytes = checkpoint_path.read_bytes()
    packet = json.loads(checkpoint_bytes)
    require(canonical(volume_slip_checkpoint_to_dict(model, state)) == canonical(packet),
            "Surface snapshot differs from its literal accepted checkpoint")
    context = no_creation_context(model, state)
    parent, basal = model._assembly(state.geometry, state.fault_history.damage)
    banks = prepare_volume_banks(parent, cut_edges, basal=basal)
    surface = prepare_volume_fracture_surface(banks)
    ids = {int(cohort.id) for cohort in state.geometry.cohorts}
    require(previous_ids <= ids, "Bounded control lost an existing material cohort")
    new_ids = ids-previous_ids
    cut_faces = set(surface.side_face_indices.ravel().tolist())
    new_cut_face_ids = {int(cohort.id) for cohort in state.geometry.cohorts
                       if int(cohort.id) in new_ids and cohort.face_index in cut_faces}
    paired_ids = new_ids & set(surface.side_cohort_ids.ravel().tolist())
    unmatched_ids = new_ids & set(surface.unmatched_cohort_ids.tolist())
    require(new_cut_face_ids <= paired_ids | unmatched_ids,
            "Newborn material on a supplied side support is not represented")
    paired_new = np.isin(surface.side_cohort_ids, sorted(new_ids)).any(axis=1)
    item = dict(name=name, computational_fixture=True,
        checkpoint_file=checkpoint_path.name,
        checkpoint_raw_sha256=digest(checkpoint_bytes), checkpoint=packet,
        creation_context=context, time_myr=state.time_myr,
        cohort_count=len(state.geometry.cohorts),
        cut_edges=banks.topology.cut_edges.tolist(), bank_ndof=banks.ndof,
        topology=dict(parent_vertex=banks.topology.parent_vertex.tolist(),
                      faces=banks.topology.mesh.faces.tolist()),
        geometry=surface_packet(surface),
        new_cohort_ids=sorted(new_ids),
        new_cut_face_cohort_ids=sorted(new_cut_face_ids),
        new_cohort_ids_in_paired_surface=sorted(paired_ids),
        new_cohort_ids_in_unmatched_surface=sorted(unmatched_ids),
        paired_patches_containing_newborn_material=int(paired_new.sum()),
        actual_physical_surface_created=False,
        actual_creation_work_assigned=False)
    summary = dict(name=name, time_myr=state.time_myr,
        cohort_count=len(state.geometry.cohorts), newborn_cohort_count=len(new_ids),
        new_cut_face_cohort_ids=sorted(new_cut_face_ids),
        new_cohort_ids_in_paired_surface=sorted(paired_ids),
        new_cohort_ids_in_unmatched_surface=sorted(unmatched_ids),
        paired_patches_containing_newborn_material=int(paired_new.sum()),
        paired_patch_count=surface.patch_count,
        unmatched_solid_side_patch_count=len(surface.unmatched_edge_indices),
        paired_interface_area_m2=surface.total_area_m2,
        unmatched_solid_side_area_m2=float(surface.unmatched_area_m2.sum()),
        surface_binding_sha256=surface.binding_sha256,
        checkpoint_sha256=packet["checkpoint_sha256"],
        checkpoint_raw_sha256=item["checkpoint_raw_sha256"],
        context_sha256=context["context_sha256"],
        actual_active_weak_plane_count=len(context["active_weak_plane_faces"]),
        actual_cut_status=context["physical_cut_status"],
        real_creation_law_still_unknown=True,
        actual_physical_surface_created=False,
        actual_creation_work_assigned=False)
    return item, summary, ids


def run(parent_path, output, dt_years):
    require(np.isfinite(dt_years) and dt_years == 100.,
            "This bounded control requires exactly two 100-year intervals")
    output.mkdir(parents=True, exist_ok=False)
    parent_bytes = parent_path.read_bytes()
    require(digest(parent_bytes) == PARENT_RAW_SHA256,
            "Continuation requires the sealed accepted 0.8682 Myr parent")
    parent_packet = json.loads(parent_bytes)
    # Restore the genuine parent once, including its two accepted intervals.
    # The only other untrusted load below is the newly saved first interval,
    # which independently replays all three accepted intervals for restart.
    model, initial = load_volume_slip_checkpoint(parent_path,
        expected_source_checkpoint_sha256=FORMATION_SHA256)
    require(len(initial.geometry.cohorts) == 640
            and abs(initial.time_myr-.8682) < 1e-14,
            "Accepted parent material catalogue or clock differs")
    require(canonical(volume_slip_checkpoint_to_dict(model, initial)) == canonical(parent_packet),
            "Semantic replay changed the accepted parent")
    audits = {"initial": audit_state(model, initial)}
    save_volume_slip_checkpoint(output/"initial.json", model, initial)
    require((output/"initial.json").read_bytes() == parent_bytes,
            "Restored accepted parent archive is not byte-identical")
    faces = np.asarray(initial.geometry.faces, int)
    first_face = faces[0]
    cycle = [(int(first_face[a]), int(first_face[b])) for a, b in ((0, 1), (1, 2), (2, 0))]
    initial_ids = {int(cohort.id) for cohort in initial.geometry.cohorts}
    snapshots, summaries = [], []
    item, summary, ids = snapshot(model, initial, "initial", output/"initial.json", cycle, initial_ids)
    snapshots.append(item); summaries.append(summary)

    first = model.step(initial, dt_years/1e6)
    require(abs(first.time_myr-initial.time_myr-dt_years/1e6) < 1e-14,
            "First real material interval has the wrong duration")
    require(len(first.geometry.cohorts) == 720,
            "First real interval did not calculate the expected 80 newborn cohorts")
    audits["after_first"] = audit_state(model, first)
    save_volume_slip_checkpoint(output/"after_first.json", model, first)
    item, summary, ids = snapshot(model, first, "after_first", output/"after_first.json", cycle, ids)
    snapshots.append(item); summaries.append(summary)

    direct = model.step(first, dt_years/1e6)
    require(abs(direct.time_myr-initial.time_myr-2*dt_years/1e6) < 1e-14,
            "Second real material interval has the wrong duration")
    require(len(direct.geometry.cohorts) == 800,
            "Second real interval did not calculate the expected 80 newborn cohorts")
    audits["direct_after_second"] = audit_state(model, direct)
    direct_packet = save_volume_slip_checkpoint(output/"direct_after_second.json", model, direct)
    item, summary, ids = snapshot(model, direct, "direct_after_second",
                                  output/"direct_after_second.json", cycle, ids)
    snapshots.append(item); summaries.append(summary)

    resumed_model, restored = load_volume_slip_checkpoint(output/"after_first.json",
        expected_source_checkpoint_sha256=FORMATION_SHA256)
    require(canonical(volume_slip_checkpoint_to_dict(resumed_model, restored)) ==
            canonical(json.loads((output/"after_first.json").read_bytes())),
            "First accepted real interval failed full semantic restart replay")
    no_creation_context(resumed_model, restored)
    resumed = resumed_model.step(restored, dt_years/1e6)
    audits["resumed_after_second"] = audit_state(resumed_model, resumed)
    resumed_context = no_creation_context(resumed_model, resumed)
    resumed_packet = save_volume_slip_checkpoint(output/"resumed_after_second.json", resumed_model, resumed)
    require(canonical(direct_packet) == canonical(resumed_packet),
            "Full real direct and restart checkpoints differ")
    require((output/"direct_after_second.json").read_bytes() ==
            (output/"resumed_after_second.json").read_bytes(),
            "Entire real direct and restart checkpoint bytes differ")
    require(canonical(snapshots[-1]["creation_context"]) == canonical(resumed_context),
            "Real restarted creation context differs from direct continuation")
    require(parent_path.read_bytes() == parent_bytes,
            "Immutable accepted parent source was changed")
    require(initial_ids <= ids and len(ids-initial_ids) == 160,
            "Real continuation did not preserve ancestry while calculating newborn material")
    require(not np.array_equal(initial.fault_history.damage, direct.fault_history.damage),
            "Calculated real weakening history remained frozen")

    fixture = dict(format=FORMAT, scope=SCOPE,
        parent_raw_sha256=PARENT_RAW_SHA256, parent_checkpoint=parent_packet,
        formation_checkpoint_sha256=FORMATION_SHA256,
        requested_dt_years=dt_years, real_accepted_intervals=2,
        snapshots=snapshots,
        unchanged_source_forcing_birth_and_activation_parameters=True,
        continuous_material_history=True, no_replacement_initialization=True,
        surface_supports_are_computational_fixtures_only=True,
        no_new_creation_law_applied_to_Genesis=True,
        no_physical_fracture_or_contact_admission=True)
    fixture["fixture_sha256"] = digest(canonical(fixture))
    (output/"growth_fixture.json").write_bytes(canonical(fixture)+b"\n")
    report = dict(format="actual-genesis-volume-creation-birth-growth-report-0.1",
        status="PASS", scope=SCOPE, parent_raw_sha256=PARENT_RAW_SHA256,
        parent_checkpoint_sha256=parent_packet["checkpoint_sha256"],
        formation_checkpoint_sha256=FORMATION_SHA256,
        initial_time_myr=initial.time_myr, final_time_myr=direct.time_myr,
        requested_dt_years=dt_years, real_accepted_intervals=2,
        initial_cohort_count=len(initial.geometry.cohorts),
        final_cohort_count=len(direct.geometry.cohorts), newborn_cohort_count=160,
        parent_replayed_once_from_formation_checkpoint=True,
        restart_replays_all_three_first_checkpoint_intervals=True,
        unchanged_source_forcing_birth_and_activation_parameters=True,
        continuous_material_history=True, no_replacement_initialization=True,
        actual_weak_plane_activation_manufactured=False,
        direct_resumed_entire_packets_byte_equal=True,
        immutable_parent_entire_packet_byte_unchanged=True,
        real_creation_law_still_unknown=True,
        sharp_physical_banks_created=False, ties_released=False,
        actual_creation_work_assigned=False,
        physical_banks_contact_underthrust_admitted=False,
        whole_heat_mechanical_closure_claimed=False, point_one_complete=False,
        state_audits=audits, surface_snapshots=summaries,
        fixture_sha256=fixture["fixture_sha256"])
    report["report_sha256"] = digest(canonical(report))
    (output/"control_report.json").write_bytes(canonical(report)+b"\n")
    print(json.dumps(report, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, default=ROOT/
        "analysis/genesis_volume_slip_validation/evidence/actual_control/direct_after_second.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt-years", type=float, default=100.)
    args = parser.parse_args()
    run(args.parent_checkpoint, args.output, args.dt_years)


if __name__ == "__main__":
    main()
