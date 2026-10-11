"""One accepted temporal interval and an independently proved subsequent refusal.

The joint restart repeats the original physical solve.  The second force and
complete material candidate remain separate from the accepted checkpoint.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.geometric_native_active_line_validation.validation_support import (
    digest, encoded, guard_sources, value_sha, verify_baseline, write,
)


def run_implicit_motion_control(out):
    """Use the canonical cached physical source and one fresh second solve."""
    from tectonics.geometric_native_implicit_epoch_admission import (
        native_implicit_epoch_admission_refusal_to_dict,
        native_implicit_epoch_admission_refusal_from_dict,
    )
    from tectonics.geometric_native_implicit_material_continuation import (
        native_implicit_material_continuation_to_dict,
    )
    from tectonics.geometric_native_implicit_mantle_loading import (
        native_implicit_spatial_mantle_load_to_dict,
        native_implicit_load_force_error_bounds,
    )
    from tectonics.geometric_native_implicit_underthrust_motion import (
        native_implicit_force_candidate_to_dict,
        native_implicit_underthrust_motion_to_dict,
        native_implicit_underthrust_motion_from_dict,
    )
    from tectonics.geometric_native_implicit_underthrust_motion_io import (
        save_native_implicit_underthrust_motion_checkpoint,
        load_native_implicit_underthrust_motion_checkpoint,
    )
    from tectonics.geometric_native_physical_history import native_physical_history_binding_to_dict
    from tests.test_geometric_native_implicit_underthrust_motion import get_implicit_refusal_case

    started = time.perf_counter()
    before, immutable = guard_sources(require_frozen=True)
    verify_baseline()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write(out/"sources_before.json", before)
    case = get_implicit_refusal_case()
    first, state = case["first"], case["before"]
    mesh, history = first["original"]["mesh"], first["history"]
    source = native_physical_history_binding_to_dict(history)
    write(out/"source_history_before.json", source)
    motion = native_implicit_underthrust_motion_to_dict(state)
    write(out/"accepted_motion_before.json", motion)
    replay = native_implicit_underthrust_motion_from_dict(deepcopy(motion),
        expected_history=history, expected_source_history_dto=source, expected_law=first["law"])
    write(out/"accepted_motion_dto_replayed.json", native_implicit_underthrust_motion_to_dict(replay))
    checkpoint_provenance = dict(control="actual original temporal admission and fresh subsequent refusal",
        source_history_sha256=value_sha(source))
    save_native_implicit_underthrust_motion_checkpoint(out/"accepted_checkpoint_before.json", state,
        provenance=checkpoint_provenance)
    _, restarted, saved_provenance = load_native_implicit_underthrust_motion_checkpoint(
        out/"accepted_checkpoint_before.json", mesh=mesh)
    if saved_provenance != checkpoint_provenance:
        raise AssertionError("Joint restart changed its declared provenance")
    write(out/"accepted_motion_joint_replayed.json", native_implicit_underthrust_motion_to_dict(restarted))
    save_native_implicit_underthrust_motion_checkpoint(out/"accepted_checkpoint_replayed.json", restarted,
        provenance=saved_provenance)
    load, candidate, material = case["load"], case["candidate"], case["material_candidate"]
    write(out/"subsequent_source_load.json", native_implicit_spatial_mantle_load_to_dict(
        mesh, state.current_surface, load))
    write(out/"subsequent_force_candidate.json", native_implicit_force_candidate_to_dict(
        candidate, previous=state, load=load))
    write(out/"unadopted_material_candidate.json", native_implicit_material_continuation_to_dict(material))
    temporal_refusal = native_implicit_epoch_admission_refusal_to_dict(case["temporal_refusal"])
    if (not case["temporal_refusal"].positive_newborn_contact_proved
            or not case["temporal_refusal"].immediate_unsupported_incoming_proved
            or case["temporal_refusal"].physical_pair_admission_authorized):
        raise AssertionError("The fresh refusal lacks an actual positive incoming newborn witness")
    write(out/"subsequent_temporal_refusal.json", temporal_refusal)
    refusal_args = dict(expected_previous=state, expected_candidate=candidate,
        expected_load=load, expected_material_epoch=material)
    refusal_replayed = native_implicit_epoch_admission_refusal_from_dict(
        deepcopy(temporal_refusal), **refusal_args)
    write(out/"subsequent_temporal_refusal_replayed.json",
        native_implicit_epoch_admission_refusal_to_dict(refusal_replayed))
    source_error = native_implicit_load_force_error_bounds(load, dict(candidate.omega_rad_per_myr))
    write(out/"subsequent_source_operator_error.json", dict(
        load_sha256=source_error["load_sha256"],
        relative_residual_norm_upper=[source_error["relative_residual_norm_upper"].numerator,
            source_error["relative_residual_norm_upper"].denominator],
        material_midpoint_debit=source_error["material_midpoint_debit"]))
    branch_counts = Counter(piece.branch for row in candidate.spatial_step.certificates for piece in row.pieces)
    if branch_counts != {"up_dip_plastic": 2, "down_dip_plastic": 2}:
        raise AssertionError("Fresh second continuous fields changed their actual constitutive branches")
    semantic_refusals = []
    for label, mutation in (
        ("mechanical_work", lambda raw: raw["snapshot"].update(cumulative_source_work_j=0.)),
        ("atom_support", lambda raw: raw["temporal_admission"]["mechanical_atom_core_margins"][0]
            .update(start_margin=[0, 1])),
        ("material_mask", lambda raw: raw["temporal_admission"]["material_coarea"]["carrier_masks"].pop()),
        ("source", lambda raw: raw["original_physical_history"]["actual_candidate"].update(dt_s=1.)),
        ("receipt", lambda raw: raw["snapshot"]["receipts"][0].update(source_work_j=0.)),
    ):
        forged = deepcopy(motion)
        mutation(forged)
        forged["state_sha256"] = value_sha(forged["snapshot"])
        try:
            native_implicit_underthrust_motion_from_dict(forged, expected_history=history,
                expected_source_history_dto=source, expected_law=first["law"])
        except (TypeError, ValueError) as error:
            semantic_refusals.append(dict(label=label, rejected=True, root_state_checksum_resealed_only=True,
                nested_proof_checksums_resealed=False, independently_supplied_source_unchanged=True,
                exception_type=type(error).__name__, message=str(error), forged_dto_sha256=value_sha(forged)))
        else:
            raise AssertionError("A resealed semantic force substitution was accepted: "+label)
    write(out/"semantic_refusals.json", semantic_refusals)
    write(out/"accepted_motion_after_refusal.json", native_implicit_underthrust_motion_to_dict(state))
    write(out/"source_history_after.json", native_physical_history_binding_to_dict(history))
    if encoded(motion) != case["refusal"].preserved_before_bytes or len(state.receipts) != 1:
        raise AssertionError("The subsequent refusal adopted an unproved force or material candidate")
    for path, original_bytes in first["original"]["original_files"].items():
        if path.read_bytes() != original_bytes:
            raise AssertionError("An original source artifact changed: "+str(path))
    after, immutable_after = guard_sources(require_frozen=True)
    write(out/"sources_after.json", after)
    if before != after or immutable != immutable_after:
        raise AssertionError("A Python source changed during the force control")
    pairs = (("source_history_before.json", "source_history_after.json"),
        ("accepted_motion_before.json", "accepted_motion_dto_replayed.json"),
        ("accepted_motion_before.json", "accepted_motion_joint_replayed.json"),
        ("accepted_motion_before.json", "accepted_motion_after_refusal.json"),
        ("accepted_checkpoint_before.json", "accepted_checkpoint_replayed.json"),
        ("subsequent_temporal_refusal.json", "subsequent_temporal_refusal_replayed.json"))
    byte_pairs = []
    for initial, repeated in pairs:
        if (out/initial).read_bytes() != (out/repeated).read_bytes():
            raise AssertionError("A restart/preservation pair differs: "+initial)
        byte_pairs.append(dict(first=initial, second=repeated,
            bytes=(out/initial).stat().st_size, sha256=digest(out/initial)))
    first_metrics, next_metrics = dict(first["step"].diagnostics), dict(candidate.diagnostics)
    balance_keys = ("material_reaction_relative_error", "angular_signal_relative_residual",
        "transverse_force_relative_residual", "cumulative_energy_balance_relative_error")
    if any(first_metrics[key] >= 5e-10 or next_metrics[key] >= 5e-10 for key in balance_keys):
        raise AssertionError("A complete independent force balance failed")
    report = dict(format="native-implicit-motion-physical-admission-refusal-control-1", status="PASS",
        scope="one genuine accepted interval; subsequent genuine force and complete material candidate refused",
        seconds=time.perf_counter()-started, source_count=len(before),
        prior_704_sources_unchanged=True, source_before_after_unchanged=True,
        accepted_loaded_native_intervals=len(state.receipts), native_force_admission_proved=True,
        native_force_admission_scope="first original physical interval only",
        full_first_interval_balances_passed=True, first_interval_balance_metrics={key:first_metrics[key] for key in balance_keys},
        subsequent_candidate_solved=True, subsequent_candidate_refused=True,
        subsequent_force_balances_passed=True, subsequent_balance_metrics={key:next_metrics[key] for key in balance_keys},
        subsequent_interval_accepted=False, subsequent_intervals_passed=False,
        refusal_preserved_before_bytes=True, positive_newborn_contact_proved=True,
        positive_newborn_contact_witness_count=len(case["temporal_refusal"].geometry.positive_contacts),
        immediate_unsupported_newborn_entry_proved=True,
        safe_truncation_with_same_euler_vectors_exists=False,
        unadopted_material_component_count=len(material.components),
        unadopted_new_archive_count=len(material.new_archives),
        subsequent_continuous_branch_counts=dict(branch_counts),
        subsequent_whole_line_down_dip_claimed=False,
        joint_restart_passed=True, joint_restart_repeats_original_genuine_force=True,
        complete_balances_passed=True, semantic_refusal_count=len(semantic_refusals), byte_pairs=byte_pairs,
        entire_point_1_completed=False,
        remaining_block="Positive new-material receiving contacts have no declared physical pair, constitutive birth state or finite bank law",
        physical_inputs_invented=False, continuum_work_added_to_Gauss_ledger=False)
    return write(out/"motion_control_report.json", report, "report_sha256")


run_control = run_implicit_motion_control
