"""Actual differential force, conservative material events and retained history."""
from dataclasses import asdict, replace
from functools import lru_cache
import json
import math

import numpy as np
import pytest

from analysis.geometric_initiation_validation.admission_case import ledger
from tests.underthrust_case import (
    DT_S, PROVENANCE, SECOND_SOURCE, SOURCE, advance_case, initialize_case, physical_inputs,
)
from tectonics.fractional_surface import EXTENSIVE_FIELDS
from tectonics.geometric_boundary_io import _validate_pair
from tectonics.geometric_contact_dynamics import UnresolvedContactDynamics
from tectonics.geometric_spatial_mantle_loading import (
    PrescribedSpatialMantleSource, _body as source_body, _hash as source_hash,
    assemble_physical_spatial_mantle_load,
)
from tectonics.geometric_transport import UnresolvedPolarityError
from tectonics.geometric_underthrust_motion import (
    _seal, advance_underthrust_motion, underthrust_motion_from_dict,
    underthrust_motion_to_dict, validate_underthrust_motion_state,
)
from tectonics.thermal import SECONDS_PER_MYR


def snapshot(value):
    return json.dumps(asdict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


@lru_cache(maxsize=1)
def accepted_interval():
    return advance_case()


def test_independently_loaded_force_step_retains_every_true_positive_loss_and_birth():
    mesh, origin, initial, step = accepted_interval()
    assert step.rotation_increment_radians[0] != step.rotation_increment_radians[1]
    assert not np.array_equal(step.state.owner_rotations[0][1], step.state.owner_rotations[1][1])
    assert len(step.geometric_transaction.losses) >= 2
    assert step.geometric_transaction.births
    assert all(loss.fragment.parcel.area_km2 > 0. for loss in step.geometric_transaction.losses)
    assert all(loss.fragment.parcel.plate == 1 and loss.receiver_plate == 0
               for loss in step.geometric_transaction.losses)
    assert all(loss.polarity_basis == "solved_spatial_underthrust_admission"
               and loss.polarity_evidence_ids for loss in step.geometric_transaction.losses)
    # Positive endpoint pockets below any plausible geographic cutoff remain
    # physical events; the full computed rotation vectors are retained.
    assert min(loss.fragment.parcel.area_km2 for loss in step.geometric_transaction.losses) < 1e-35
    assert any(abs(row[1]) > 0. for row in step.rotation_increment_radians)
    assert all(birth.parcel.age_myr == birth.parcel.cold_mantle_volume_km3
               == birth.parcel.density_excess_mass_kg == 0. for birth in step.geometric_transaction.births)
    _validate_pair(step.state.current_surface, step.state.current_inventory)
    assert len(step.state.current_inventory.cohorts) == len(step.geometric_transaction.losses)
    assert len(step.state.current_inventory.transactions) == len(step.state.receipts) == 1
    assert initial.current_surface == origin and not initial.receipts
    assert all(local.plastic_slip_m == local.cumulative_slip_m == 0. for _, local in initial.states)


def test_actual_closing_pocket_is_admitted_when_its_whole_parent_contact_opens():
    _, _, _, first = accepted_interval()
    audit = first.admission_audit
    opening = {row["contact_id"] for row in audit["contacts"] if row["status"] == "nonconvergent"}
    witnesses = audit["actual_overlap_assessments"]
    admitted = [row for row in witnesses if row["comparison"] and opening & set(row["contact_ids"])]
    assert admitted
    evidence = {identity for row in witnesses for identity in row["accepted_evidence_ids"]}
    assert all(set(loss.polarity_evidence_ids) <= evidence for loss in first.geometric_transaction.losses)
    for row in admitted:
        for arc in row["receiver_arcs"]:
            assert arc["status"] == "spatial_underthrust_admissible"
            for support in arc["supports"]:
                bound = support["normal_velocity_rounding_bound_km_per_myr"]
                assert support["normal_velocity_min_km_per_myr"] < -bound
                assert support["normal_velocity_max_km_per_myr"] <= bound


def test_all_four_material_reservoirs_balance_with_actual_hot_births():
    _, origin, _, first = accepted_interval()
    born = first.geometric_transaction.births
    residuals = ledger(origin, first.state.current_surface, first.state.current_inventory, born)
    assert set(residuals) == set(EXTENSIVE_FIELDS)
    assert max(residuals.values()) < 5e-12
    assert any(cohort.parcel.cold_mantle_volume_km3 > 0. for cohort in first.state.current_inventory.cohorts)
    # New material is a separate explicitly hot reservoir, never an old bank.
    assert not {birth.parcel.material_id for birth in born} & {
        reference.material_id for reference in first.state.buried_registry.records}


def test_same_physical_clock_drives_mechanics_surface_archive_and_each_original_node():
    _, origin, initial, first = accepted_interval()
    assert first.state.elapsed_time_s == first.state.spatial_memory.elapsed_time_s == DT_S
    assert first.state.current_surface.time_myr == origin.time_myr+DT_S/SECONDS_PER_MYR
    assert first.state.current_inventory.time_myr == first.state.current_surface.time_myr
    seed = dict(initial.states)
    assert tuple(identity for identity, _ in first.state.states) == tuple(seed)
    for identity, current in first.state.states:
        assert current.step_count == seed[identity].step_count+1
        assert current.elapsed_time_s == seed[identity].elapsed_time_s+DT_S
        assert current.cumulative_slip_m >= abs(current.plastic_slip_m)
    assert first.state.spatial_memory.stage_count == 1
    assert first.state.initial_mechanics == initial.initial_mechanics
    assert first.state.preparation_work_j == initial.preparation_work_j
    initial_by_material = {fragment.parcel.material_id: fragment.parcel for fragment in origin.fragments}
    for fragment in first.state.current_surface.fragments:
        before = initial_by_material.get(fragment.parcel.material_id)
        if before is not None:
            assert fragment.parcel.age_myr == pytest.approx(before.age_myr+DT_S/SECONDS_PER_MYR,
                                                           rel=16*np.finfo(float).eps)
            assert fragment.parcel.material_fields == before.material_fields
            assert fragment.parcel.specific_properties == before.specific_properties


def test_source_work_and_force_balances_use_the_computed_actual_motion():
    mesh, origin, initial, first = accepted_interval()
    assert first.state.receipts[-1].current_load == assemble_physical_spatial_mantle_load(mesh, origin, SOURCE)
    assert first.state.cumulative_source_work_j == first.state.receipts[-1].source_work_j
    for step in (first,):
        receipt = step.state.receipts[-1]
        expected = float(np.asarray(receipt.current_load.torque_nm)@np.asarray(step.rotation_increment_radians).reshape(-1))
        assert receipt.source_work_j == pytest.approx(expected, rel=3e-13)
        diagnostics = dict(step.diagnostics)
        assert diagnostics["angular_signal_relative_residual"] < 2e-10
        assert diagnostics["transverse_force_relative_residual"] < 2e-10
        assert diagnostics["material_reaction_relative_error"] < 2e-10
        assert diagnostics["incremental_energy_balance_relative_error"] < 2e-10
        assert diagnostics["cumulative_energy_balance_relative_error"] < 2e-10
        assert diagnostics["continuous_gap_max_m"] <= 0.
        assert diagnostics["anchor_transport_error_m"] < initial.max_anchor_transport_error_m


def test_every_actual_archive_cohort_has_receiver_support_and_original_bank_identity():
    _, _, _, first = accepted_interval()
    registry = first.state.buried_registry
    assert {row.event_id for row in registry.records} == {
        cohort.event_id for cohort in first.state.current_inventory.cohorts}
    assert all(cohort.supports and not cohort.unresolved_support
               and cohort.loss.fragment.parcel == cohort.parcel
               for cohort in first.state.current_inventory.cohorts)
    assert all(row.acceptance_rotation == first.state.owner_rotations[1][1]
               for row in registry.records)


def test_zero_viscosity_sliding_stays_at_yield_without_resetting_original_pair_memory():
    _, _, initial, first = accepted_interval()
    assert any(local.plastic_slip_m < 0. for _, local in first.state.states)
    assert dict(first.diagnostics)["viscous_work_j"] == 0.
    assert dict(first.diagnostics)["friction_work_j"] == 0.
    assert first.state.initial_mechanics.polarity_history == initial.initial_mechanics.polarity_history
    assert first.state.spatial_memory.physical_trace_sha256 == initial.initial_mechanics.fault_traces.trace_sha256


def test_finite_viscosity_is_an_independent_physical_input_and_has_positive_dissipation():
    _, _, _, first = advance_case(viscosity_pa_s_m=1e9)
    assert first.geometric_transaction.losses and dict(first.diagnostics)["viscous_work_j"] > 0.
    assert dict(first.diagnostics)["incremental_energy_balance_relative_error"] < 2e-10


def test_sparse_owner_relabeling_preserves_computed_motion_and_actual_donor_material():
    _, _, _, original = accepted_interval()
    _, _, _, sparse = advance_case(owners=(10**12, 7))
    # Canonical owner order reverses while physical source/material identities do not.
    np.testing.assert_allclose(sparse.rotation_increment_radians,
        tuple(reversed(original.rotation_increment_radians)), rtol=3e-12, atol=1e-21)
    first_loss = {loss.fragment.parcel.material_id: loss.fragment.parcel.area_km2
                  for loss in original.geometric_transaction.losses}
    sparse_loss = {loss.fragment.parcel.material_id: loss.fragment.parcel.area_km2
                   for loss in sparse.geometric_transaction.losses}
    assert sparse_loss == pytest.approx(first_loss, rel=3e-11)
    assert all(loss.fragment.parcel.plate == 7 and loss.receiver_plate == 10**12
               for loss in sparse.geometric_transaction.losses)


def test_stale_initial_source_is_refused_after_true_material_geometry_changes():
    mesh, origin, initial, first = accepted_interval()
    before = snapshot(first.state)
    with pytest.raises(ValueError):
        advance_underthrust_motion(mesh, origin, first.state, initial.initial_load, DT_S, provenance=PROVENANCE)
    assert snapshot(first.state) == before


@pytest.mark.parametrize("source", [SOURCE, SECOND_SOURCE])
def test_actual_current_source_continuation_needs_new_physical_neighbor_pairs_atomically(source):
    mesh, origin, initial, first = accepted_interval()
    load = assemble_physical_spatial_mantle_load(mesh, first.state.current_surface, source)
    assert load.surface_sha256 != initial.initial_load.surface_sha256
    assert load == assemble_physical_spatial_mantle_load(mesh, first.state.current_surface, source)
    before = snapshot(first.state)
    with pytest.raises(UnresolvedPolarityError) as caught:
        advance_underthrust_motion(mesh, origin, first.state, load, DT_S, provenance=PROVENANCE)
    assert any(item.reason == "new_material_requires_new_physical_pair" for item in caught.value.overlaps)
    assert all(item.area_km2 > 0. for item in caught.value.overlaps)
    assert snapshot(first.state) == before


def test_rehashed_false_source_operator_is_refused_before_material_transport(monkeypatch):
    import tectonics.geometric_underthrust_motion as engine
    mesh, origin, state, load = initialize_case()
    torque = list(load.torque_nm)
    torque[-1] *= 1.01
    changed = replace(load, torque_nm=tuple(torque))
    changed = replace(changed, load_sha256=source_hash(source_body(changed)))
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("False source reached material transport")
    monkeypatch.setattr(engine, "advance_geometric_surface", forbidden)
    with pytest.raises(ValueError):
        advance_underthrust_motion(mesh, origin, state, changed, DT_S, provenance=PROVENANCE)
    assert not calls


@pytest.mark.parametrize("dt", [True, 0., -1., float("nan"), 1e12])
def test_invalid_or_unsupported_force_interval_keeps_all_original_state_bytes(dt):
    mesh, origin, state, load = initialize_case()
    before = snapshot(state)
    with pytest.raises((ValueError, UnresolvedContactDynamics)):
        advance_underthrust_motion(mesh, origin, state, load, dt, provenance=PROVENANCE)
    assert snapshot(state) == before


def test_tight_finite_pose_error_bound_blocks_atomically_before_accepting_material():
    mesh, origin, state, load = initialize_case(max_anchor_transport_error_m=1e-12)
    before = snapshot(state)
    with pytest.raises(UnresolvedContactDynamics, match="anchor error"):
        advance_underthrust_motion(mesh, origin, state, load, DT_S, provenance=PROVENANCE)
    assert snapshot(state) == before


@pytest.mark.parametrize("field", ["states", "owner_rotations", "elapsed_time_s", "receipt"])
def test_saved_state_or_rehashed_receipt_cannot_bypass_complete_physical_replay(field):
    mesh, origin, _, first = accepted_interval()
    state = first.state
    if field == "states":
        states = ((state.states[0][0], replace(state.states[0][1], plastic_slip_m=0.)), *state.states[1:])
        changed = replace(state, states=states)
    elif field == "owner_rotations":
        poses = list(state.owner_rotations)
        poses[0] = (poses[0][0], tuple(map(tuple, np.eye(3))))
        changed = replace(state, owner_rotations=tuple(poses))
    elif field == "elapsed_time_s":
        changed = replace(state, elapsed_time_s=state.elapsed_time_s+1.)
    else:
        receipt = _seal(replace(state.receipts[0], source_work_j=state.receipts[0].source_work_j*1.01))
        changed = replace(state, receipts=(receipt,))
    with pytest.raises(ValueError):
        validate_underthrust_motion_state(mesh, origin, changed)


def test_boolean_zero_alias_cannot_replace_serialized_physical_snapshot_value():
    mesh, origin, state, _ = initialize_case()
    raw = underthrust_motion_to_dict(mesh, origin, state)
    raw["snapshot"]["elapsed_time_s"] = False
    with pytest.raises(ValueError):
        underthrust_motion_from_dict(raw, mesh, origin)


def test_reverse_differential_motion_does_not_assign_old_pair_to_hot_newborn_material():
    mesh, origin, _, first = accepted_interval()
    reverse = PrescribedSpatialMantleSource(SOURCE.angular_velocity_rad_s,
        SOURCE.basal_drag_pa_s_m, 1., (-.2, 0., 0.),
        "Explicit reversal of imposed spatial amplitude before trial solve")
    load = assemble_physical_spatial_mantle_load(mesh, first.state.current_surface, reverse)
    before = snapshot(first.state)
    with pytest.raises((UnresolvedPolarityError, UnresolvedContactDynamics)) as caught:
        advance_underthrust_motion(mesh, origin, first.state, load, DT_S, provenance=PROVENANCE)
    if isinstance(caught.value, UnresolvedPolarityError):
        assert any(item.reason == "new_material_requires_new_physical_pair" for item in caught.value.overlaps)
    assert snapshot(first.state) == before
