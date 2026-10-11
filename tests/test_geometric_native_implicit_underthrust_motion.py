"""Actual original forces require all temporal gates before implicit acceptance."""
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import math

import pytest

from tectonics.geometric_native_contact_reference_transport import (
    ContactHistoryTransportLaw, TRANSPORT_LAW_MODEL,
)
from tectonics.geometric_native_implicit_underthrust_motion import (
    accept_native_implicit_underthrust_motion, native_implicit_underthrust_motion_from_dict,
    native_implicit_underthrust_motion_to_dict, validate_native_implicit_underthrust_motion_state,
    advance_native_implicit_underthrust_motion, UnresolvedNativeImplicitUnderthrustMotion,
    validate_native_implicit_force_candidate, native_implicit_force_candidate_to_dict,
    solve_native_implicit_underthrust_candidate, assess_native_implicit_underthrust_candidate,
)
from tectonics.geometric_native_implicit_underthrust_motion_io import (
    save_native_implicit_underthrust_motion_checkpoint, load_native_implicit_underthrust_motion_checkpoint,
)
from tectonics.geometric_native_physical_history import (
    _typed_fingerprint, native_physical_history_binding_to_dict,
)
from tests.test_geometric_native_physical_history import get_physical_case


@lru_cache(maxsize=1)
def get_implicit_motion_case():
    original = get_physical_case()
    history = original["binding"]
    source = native_physical_history_binding_to_dict(history)
    law = ContactHistoryTransportLaw(TRANSPORT_LAW_MODEL,
        "Explicit complete temporal original receiver configuration; positive fresh quarter changes only")
    step = accept_native_implicit_underthrust_motion(history, law=law,
        expected_source_history_dto=source)
    return dict(original=original, history=history, source=source, law=law, step=step,
        raw=native_implicit_underthrust_motion_to_dict(step.state))


@pytest.fixture(scope="module")
def implicit_motion_case():
    return get_implicit_motion_case()


def test_actual_original_force_is_admitted_only_after_complete_temporal_proofs(implicit_motion_case):
    case = implicit_motion_case
    state, candidate = case["step"].state, case["history"].actual_candidate
    assert len(state.receipts) == 1
    assert state.states == candidate.states
    assert state.transverse_displacement_m == candidate.transverse_displacement_m
    assert state.spatial_memory == candidate.spatial_step.memory
    assert state.free_anchor_fields == tuple((row[0], *(field.coefficients for field in row[1:]))
        for row in candidate.spatial_step.anchor_fields)
    assert state.elapsed_time_s == candidate.dt_s
    assert state.current_surface.phase == "partition"
    assert len(state.admission.interface_integral.carrier_masks) == 16
    assert len(state.admission.original_bank_support.parent_supports) == 8
    assert len(state.admission.original_bank_support.live_clamps) == 8
    assert len(state.admission.mechanical_atom_core_margins) == 16
    assert all(min(start, end) > 0 for _, start, end in state.admission.mechanical_atom_core_margins)
    assert not case["history"].accepted_before.receipts
    assert not case["history"].accepted_before.current_inventory.records
    assert case["history"].accepted_before.cumulative_source_work_j == 0


def test_admitted_original_gauss_work_is_kept_separate_from_continuum_work(implicit_motion_case):
    state = implicit_motion_case["step"].state
    metrics = dict(implicit_motion_case["history"].actual_candidate.candidate_diagnostics)
    assert state.cumulative_source_work_j == metrics["source_work_j"]
    assert state.cumulative_basal_dissipation_j == metrics["basal_dissipation_j"]
    assert state.cumulative_bending_numerical_dissipation_j == metrics["bending_numerical_dissipation_j"]
    assert dict(implicit_motion_case["step"].diagnostics)["accepted_cumulative_energy_balance_relative_error"] < 5e-10
    raw = implicit_motion_case["raw"]
    assert not raw["snapshot"]["continuous_contact_work_added_to_mechanical_ledger"]
    assert not raw["temporal_admission"]["endpoint_arc_refusal_overridden"]
    assert not raw["temporal_admission"]["continuous_work_added_to_Gauss_work"]
    assert all(path.read_bytes() == before for path, before in implicit_motion_case["original"]["original_files"].items())


def test_loaded_implicit_partition_preserves_curved_material_and_hot_histories(implicit_motion_case):
    state = implicit_motion_case["step"].state
    assert state.current_surface.source_material_state is implicit_motion_case["history"].material_state
    assert len(state.current_surface.original_parent_allocations) == 8
    assert len(state.current_surface.archive_cohorts) == 3
    assert len([item for item in state.current_surface.components if item.material_kind == "hot_birth"]) == 5
    assert len(state.current_surface.paired_corrections) == 1
    correction, = state.current_surface.paired_corrections
    assert correction.area_bounds_sr.lo > 0
    assert correction.negative_component_id != correction.positive_component_id


def test_loaded_state_strict_dto_restart_replays_every_temporal_gate(implicit_motion_case):
    case = implicit_motion_case
    replay = native_implicit_underthrust_motion_from_dict(deepcopy(case["raw"]),
        expected_history=case["history"], expected_source_history_dto=case["source"],
        expected_law=case["law"])
    assert native_implicit_underthrust_motion_to_dict(replay) == case["raw"]


@pytest.mark.parametrize("field,value", (("elapsed_time_s", 0.),
    ("cumulative_source_work_j", 0.), ("states", ()), ("owner_histories", ()), ("receipts", ())))
def test_typed_accepted_state_cannot_reuse_integrity_after_changing_physics(implicit_motion_case, field, value):
    changed = replace(implicit_motion_case["step"].state, **{field: value})
    with pytest.raises(ValueError, match="complete force, temporal material and support replay"):
        validate_native_implicit_underthrust_motion_state(changed)


@pytest.mark.parametrize("mutation", ("mechanical_work", "atom_support", "material_mask", "source", "receipt"))
def test_resealed_loaded_state_mutation_refuses(implicit_motion_case, mutation):
    case = implicit_motion_case
    raw = deepcopy(case["raw"])
    if mutation == "mechanical_work":
        raw["snapshot"]["cumulative_source_work_j"] = math.nextafter(
            raw["snapshot"]["cumulative_source_work_j"], math.inf)
    elif mutation == "atom_support":
        raw["temporal_admission"]["mechanical_atom_core_margins"][0]["start_margin"] = [0, 1]
    elif mutation == "material_mask":
        raw["temporal_admission"]["material_coarea"]["carrier_masks"].pop()
    elif mutation == "source":
        raw["original_physical_history"]["actual_candidate"]["dt_s"] += 1.
    elif mutation == "receipt":
        raw["snapshot"]["receipts"][0]["source_work_j"] = math.nextafter(
            raw["snapshot"]["receipts"][0]["source_work_j"], math.inf)
    raw["state_sha256"] = hashlib.sha256(json.dumps(raw["snapshot"], sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError, match="independent complete force and material replay"):
        native_implicit_underthrust_motion_from_dict(raw, expected_history=case["history"],
            expected_source_history_dto=case["source"], expected_law=case["law"])


def test_joint_writer_retains_original_and_existing_destination(implicit_motion_case, tmp_path):
    state = implicit_motion_case["step"].state
    path = tmp_path/"accepted.json"
    save_native_implicit_underthrust_motion_checkpoint(path, state,
        provenance=dict(control="original temporal material admission"))
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_native_implicit_underthrust_motion_checkpoint(path, state)
    assert path.read_bytes() == before
    raw = json.loads(before)
    assert raw["implicit_underthrust_motion"]["snapshot"]["accepted_loaded_native_intervals"] == 1


def test_joint_checkpoint_detects_integrity_failure_before_source_replay(implicit_motion_case, tmp_path):
    path = tmp_path/"changed.json"
    save_native_implicit_underthrust_motion_checkpoint(path, implicit_motion_case["step"].state)
    raw = json.loads(path.read_bytes())
    raw["implicit_underthrust_motion"]["snapshot"]["elapsed_time_s"] = 0.
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="integrity mismatch"):
        load_native_implicit_underthrust_motion_checkpoint(path,
            mesh=implicit_motion_case["original"]["mesh"])


@lru_cache(maxsize=1)
def get_implicit_second_trial_case():
    """Share the subsequent genuine force solve with loading/admission controls."""
    from analysis.geometric_underthrust_validation.underthrust_case import DT_S, SOURCE_INTERVALS
    from tectonics.geometric_native_implicit_mantle_loading import (
        assemble_native_implicit_physical_spatial_mantle_load,
    )
    first = get_implicit_motion_case()
    state, mesh = first["step"].state, first["original"]["mesh"]
    before_bytes = json.dumps(native_implicit_underthrust_motion_to_dict(state), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    load = assemble_native_implicit_physical_spatial_mantle_load(mesh, state.current_surface,
        SOURCE_INTERVALS[1])
    candidate = solve_native_implicit_underthrust_candidate(mesh, state, load, DT_S,
        provenance="Explicit second spatial source and interval selected before its force result")
    return dict(first=first, before=state, before_bytes=before_bytes, load=load,
        candidate=candidate, current_load=load, next_candidate=candidate)


@lru_cache(maxsize=1)
def get_implicit_refusal_case():
    """One complete fresh material candidate; never adopt its unsupported pair."""
    trial = get_implicit_second_trial_case()
    try:
        assess_native_implicit_underthrust_candidate(trial["first"]["original"]["mesh"],
            trial["before"], trial["load"], trial["candidate"])
    except UnresolvedNativeImplicitUnderthrustMotion as refusal:
        return dict(trial, material_candidate=refusal.material_candidate,
            temporal_refusal=refusal.temporal_refusal, refusal=refusal)
    raise AssertionError("A newly covered hot receiver has no declared physical pair")


@pytest.fixture(scope="module")
def implicit_refusal_case():
    return get_implicit_refusal_case()


def test_subsequent_genuine_force_balances_but_cannot_adopt_new_material_contact(implicit_refusal_case):
    case = implicit_refusal_case
    before, candidate, refusal = case["before"], case["candidate"], case["refusal"]
    assert refusal.before_state is before
    assert refusal.current_load is case["load"]
    assert refusal.preserved_before_bytes == case["before_bytes"]
    current = json.dumps(native_implicit_underthrust_motion_to_dict(before), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    assert current == case["before_bytes"]
    assert len(before.receipts) == 1
    assert candidate.before_state_sha256 == before.state_sha256
    assert candidate.current_load_sha256 == case["load"].load_sha256
    assert candidate.spatial_step.memory.stage_count == before.spatial_memory.stage_count+1
    assert all(local.step_count == dict(before.states)[identity].step_count+1
        for identity, local in candidate.states)
    metrics = dict(candidate.diagnostics)
    assert all(metrics[name] < 5e-10 for name in ("angular_signal_relative_residual",
        "transverse_force_relative_residual", "material_reaction_relative_error",
        "cumulative_energy_balance_relative_error"))
    assert metrics["anchor_transport_error_m"] < before.max_anchor_transport_error_m
    assert candidate.omega_rad_per_myr[0][1] != candidate.omega_rad_per_myr[1][1]
    raw = native_implicit_force_candidate_to_dict(candidate, previous=before, load=case["load"])
    assert not raw["candidate_mechanics_adopted"]
    assert raw["new_temporal_material_admission_required"]
    assert raw["before_accepted_intervals"] == 1


def test_unadopted_material_continuation_keeps_both_old_and_new_births(implicit_refusal_case):
    case = implicit_refusal_case
    epoch = case["material_candidate"]
    assert _typed_fingerprint(epoch.previous_epoch) == _typed_fingerprint(case["before"].current_surface)
    assert epoch.time_myr > case["before"].current_surface.time_myr
    assert epoch.omega_rad_per_myr == case["candidate"].omega_rad_per_myr
    assert len(epoch.original_parent_allocations) == 8
    assert len(epoch.current_original_parent_reservoirs) == 8
    assert epoch.components and epoch.new_archives
    assert case["before"].current_surface is not epoch


@pytest.mark.parametrize("field,value", (("dt_s", 1.), ("omega_rad_per_myr", ()),
    ("states", ()), ("free_anchor_fields", ()), ("diagnostics", ())))
def test_changed_trial_cannot_reuse_genuine_solve_authority(implicit_refusal_case, field, value):
    case = implicit_refusal_case
    with pytest.raises(ValueError, match="complete genuine current force solve"):
        validate_native_implicit_force_candidate(replace(case["candidate"], **{field: value}),
            previous=case["before"], load=case["load"])


def test_changed_load_cannot_reuse_hash_of_the_genuine_trial(implicit_refusal_case):
    case = implicit_refusal_case
    changed = replace(case["load"], torque_nm=(0.,)*len(case["load"].torque_nm))
    assert changed.load_sha256 == case["load"].load_sha256
    with pytest.raises(ValueError, match="complete genuine current force solve"):
        validate_native_implicit_force_candidate(case["candidate"], previous=case["before"], load=changed)


def test_actual_second_continuous_branches_are_not_claimed_all_down_dip(implicit_refusal_case):
    branches = [piece.branch for certificate in implicit_refusal_case["candidate"].spatial_step.certificates
        for piece in certificate.pieces]
    assert branches.count("up_dip_plastic") == 2
    assert branches.count("down_dip_plastic") == 2
