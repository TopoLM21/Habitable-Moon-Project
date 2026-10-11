"""Accepted source-generated common motion and atomic differential refusal."""
from dataclasses import asdict, replace
from functools import lru_cache
import json
import math

import numpy as np
import pytest

from tests.test_geometric_bending_contact_dynamics import BOUNDS, DT, prepared
from tests.test_geometric_bending_fault_transport_io import nonempty_fixture
from tests.test_geometric_distributed_contact_invariants import PARAMETERS, PROVENANCE as BASAL_PROVENANCE, traces, world
from tests.test_geometric_bending_fault_transport import _rodrigues
from tectonics.geometric_basal_loading import (
    _body as default_body, _hash as default_hash, assemble_geometric_basal_load,
)
from tectonics.geometric_bending_contact_dynamics import initialize_bending_contact_dynamics
from tectonics.geometric_bending_motion import (
    _seal, advance_bending_motion, bending_motion_from_dict, bending_motion_to_dict,
    initialize_bending_motion, validate_bending_motion_state,
)
from tectonics.geometric_boundary import initialize_boundary, polarity_evidence
from tectonics.geometric_boundary_io import _validate_pair
from tectonics.geometric_contact_dynamics import UnresolvedContactDynamics
from tectonics.geometric_fault_traces import TraceQuadrature, declare_fault_trace_set
from tectonics.geometric_polarity import PolarityEvidence
from tectonics.geometric_rigid_mantle_loading import PrescribedRigidMantleSource, assemble_physical_rigid_mantle_load
from tectonics.geometric_surface import totals
from tectonics.thermal import SECONDS_PER_MYR


PROVENANCE = "Physical force-generated common material motion with source before each interval"
SOURCE = PrescribedRigidMantleSource((2e-12, -1e-12, 3e-12), 1e14, PROVENANCE)
SECOND_SOURCE = PrescribedRigidMantleSource((-1e-12, 2e-12, 1.5e-12), 1.3e14,
    "Independently changed fixed global mantle flow for second physical interval")


@lru_cache(maxsize=1)
def motion_fixture():
    mesh, _, _ = world()
    origin, trace_set, _, template = prepared(gap=0.)
    load = assemble_physical_rigid_mantle_load(mesh, origin, SOURCE)
    seed = initialize_bending_contact_dynamics(origin, trace_set, load, template.bending_parameters, **BOUNDS)
    initial = initialize_bending_motion(origin, initialize_boundary(origin), seed, load)
    first = advance_bending_motion(mesh, origin, initial, load, DT, provenance=PROVENANCE)
    second_load = assemble_physical_rigid_mantle_load(mesh, first.state.current_surface, SECOND_SOURCE)
    second = advance_bending_motion(mesh, origin, first.state, second_load, DT, provenance=PROVENANCE)
    return mesh, origin, initial, first, second


@lru_cache(maxsize=1)
def archive_motion_fixture():
    mesh, origin, _, _, transport = nonempty_fixture()
    template = transport.initial_mechanics
    trace_set = declare_fault_trace_set(origin,
        tuple(replace(trace, initial_gap_m=0., initial_jump_m=0.) for trace in template.fault_traces.traces),
        template.fault_traces.quadrature)
    load = assemble_physical_rigid_mantle_load(mesh, origin, SOURCE)
    history = tuple(PolarityEvidence(**row) for row in polarity_evidence(transport.reference_inventory))
    seed = initialize_bending_contact_dynamics(origin, trace_set, load, template.bending_parameters,
                                              history=history, **BOUNDS)
    initial = initialize_bending_motion(origin, transport.reference_inventory, seed, load)
    first = advance_bending_motion(mesh, origin, initial, load, 1e8, provenance=PROVENANCE)
    return mesh, origin, initial, first


def test_two_noncommuting_force_steps_move_actual_material_and_archive_for_the_same_clock():
    mesh, origin, initial, first, second = motion_fixture()
    assert first.mechanical_step.rotation_increment_radians == ((.002, -.001, .003),)*2
    expected_first = _rodrigues(tuple(value*DT for value in SOURCE.angular_velocity_rad_s))
    expected_second = _rodrigues(tuple(value*DT for value in SECOND_SOURCE.angular_velocity_rad_s))
    assert np.linalg.norm(expected_second@expected_first-expected_first@expected_second) > 1e-6
    np.testing.assert_allclose(second.state.common_rotation, expected_second@expected_first,
                               rtol=0., atol=16*np.finfo(float).eps)
    assert len(second.state.motion_receipts) == len(second.state.mechanics.segments) == 2
    assert second.state.current_surface.time_myr == second.state.current_inventory.time_myr == 2.*DT/SECONDS_PER_MYR
    assert second.state.mechanics.elapsed_time_s == 2.*DT
    assert totals(second.state.current_surface.fragments) == totals(origin.fragments)
    original = {fragment.fragment_id: fragment for fragment in origin.fragments}
    for moved in second.state.current_surface.fragments:
        before = original[moved.fragment_id]
        np.testing.assert_allclose(moved.polygon,
            np.asarray(before.polygon)@(expected_second@expected_first).T,
            rtol=0., atol=64*np.finfo(float).eps)
        assert moved.parcel.age_myr == pytest.approx(before.parcel.age_myr+2.*DT/SECONDS_PER_MYR,
                                                    rel=4*np.finfo(float).eps)
        assert replace(moved.parcel, age_myr=before.parcel.age_myr) == before.parcel
    assert not first.geometric_transaction.births and not first.geometric_transaction.losses
    assert not second.geometric_transaction.births and not second.geometric_transaction.losses
    _validate_pair(second.state.current_surface, second.state.current_inventory)
    assert initial.current_surface == origin and not initial.motion_receipts
    assert validate_bending_motion_state(mesh, origin, second.state) == second.state


def test_stage_sources_use_current_footprints_and_separate_work_preserving_original_epoch():
    _, origin, initial, first, second = motion_fixture()
    receipt = second.state.mechanics.segments[-1]
    assert receipt.reference_configuration.surface == first.state.current_surface
    assert receipt.current_load.surface_sha256 != second.state.mechanics.initial_load.surface_sha256
    assert receipt.current_load.source == SECOND_SOURCE
    assert second.state.mechanics.initial_mechanics == initial.mechanics.initial_mechanics
    assert second.state.mechanics.preparation_work_j == initial.mechanics.preparation_work_j
    expected_work = math.fsum(segment.source_work_j for segment in second.state.mechanics.segments)
    assert second.state.mechanics.cumulative_source_work_j == expected_work
    for step in (first, second):
        increment = np.asarray(step.mechanical_step.rotation_increment_radians).reshape(-1)
        assert step.mechanical_step.source_work_j == pytest.approx(
            np.asarray(step.state.mechanics.segments[-1].current_load.torque_nm)@increment, rel=3e-12)
        assert step.mechanical_step.torque_relative_residual < 1e-12
        assert step.mechanical_step.incremental_energy_balance_relative_error < 1e-12
        assert all(local.gap_free_m == local.jump_free_m == local.plastic_slip_m == 0.
                   for _, local in step.state.mechanics.states)
    # Mechanical point clocks advance through the real constitutive interval.
    original_points = dict(initial.mechanics.states)
    assert all(local.step_count == original_points[identity].step_count+2
               and local.elapsed_time_s == original_points[identity].elapsed_time_s+2.*DT
               for identity, local in second.state.mechanics.states)
    assert origin.time_myr == 0.


def test_two_existing_cohorts_keep_all_budgets_payloads_and_live_rotated_support():
    mesh, origin, initial, first = archive_motion_fixture()
    old = {cohort.event_id: cohort for cohort in initial.current_inventory.cohorts}
    assert len(old) == len(first.state.current_inventory.cohorts) == 2
    assert totals(first.state.current_surface.fragments) == totals(origin.fragments)
    for cohort in first.state.current_inventory.cohorts:
        before = old[cohort.event_id]
        assert cohort.loss == before.loss and cohort.payload_sha256 == before.payload_sha256
        assert cohort.parcel == before.parcel
        assert cohort.supports and not cohort.unresolved_support
        assert cohort.subducting_plate == before.subducting_plate
        assert cohort.overriding_plate == before.overriding_plate
        assert cohort.acceptance_time_myr == before.acceptance_time_myr
    assert len(first.state.current_inventory.transactions) == len(initial.current_inventory.transactions)+1
    assert first.state.current_inventory.time_myr == origin.time_myr+1e8/SECONDS_PER_MYR
    assert first.state.current_inventory.surface_digest != initial.current_inventory.surface_digest
    assert first.state.mechanics.initial_mechanics.polarity_history
    _validate_pair(first.state.current_surface, first.state.current_inventory)
    assert validate_bending_motion_state(mesh, origin, first.state) == first.state


def test_differential_mechanical_solution_is_refused_before_any_material_transaction(monkeypatch):
    mesh, _, _ = world()
    origin, _, load, seed = prepared(mixed=True)
    state = initialize_bending_motion(origin, initialize_boundary(origin), seed, load)
    before = json.dumps(asdict(state), sort_keys=True)
    def forbidden_transaction(*args, **kwargs):
        raise AssertionError("Rejected differential motion reached material transport")
    monkeypatch.setattr("tectonics.geometric_bending_motion.advance_geometric_surface", forbidden_transaction)
    with pytest.raises(UnresolvedContactDynamics, match="differential motion"):
        advance_bending_motion(mesh, origin, state, load, DT, provenance=PROVENANCE)
    assert json.dumps(asdict(state), sort_keys=True) == before


def test_rehashed_default_torque_is_refused_by_physical_source_reassembly_before_motion():
    mesh, origin, initial, _, _ = motion_fixture()
    _, _, lid = world()
    load = assemble_geometric_basal_load(mesh, origin, PARAMETERS, lid, BASAL_PROVENANCE)
    fake = replace(load, torque_nm=tuple(1.01*value for value in load.torque_nm))
    fake = replace(fake, load_sha256=default_hash(default_body(fake)))
    before = json.dumps(asdict(initial), sort_keys=True)
    with pytest.raises(ValueError, match="trusted source reassembly"):
        advance_bending_motion(mesh, origin, initial, fake, DT, provenance=PROVENANCE)
    assert json.dumps(asdict(initial), sort_keys=True) == before


def test_sparse_actual_owners_use_compact_force_source_and_motion_coordinates():
    mesh, original, _ = world()
    owners = {0: 3, 1: 1000000}
    origin = replace(original, fragments=tuple(replace(fragment,
        parcel=replace(fragment.parcel, plate=owners[fragment.parcel.plate])) for fragment in original.fragments))
    _, _, _, template = prepared(gap=0.)
    trace_set = declare_fault_trace_set(origin, traces(sub=3, over=1000000, gap=0.), TraceQuadrature(1, 2))
    source = replace(SOURCE, angular_velocity_rad_s=(1e-12, 0., 1e-12))
    load = assemble_physical_rigid_mantle_load(mesh, origin, source)
    seed = initialize_bending_contact_dynamics(origin, trace_set, load, template.bending_parameters, **BOUNDS)
    state = initialize_bending_motion(origin, initialize_boundary(origin), seed, load)
    first = advance_bending_motion(mesh, origin, state, load, 1e8, provenance=PROVENANCE)
    assert first.state.mechanics.plate_ids == (3, 1000000)
    assert np.shape(load.drag_nm_s) == (6, 6)
    assert len(first.mechanical_step.angular_velocity_rad_s) == 2
    assert {fragment.parcel.plate for fragment in first.state.current_surface.fragments} == {3, 1000000}


def test_rehashed_motion_increment_cannot_replace_the_accepted_force_receipt():
    mesh, origin, _, first, _ = motion_fixture()
    receipt = first.state.motion_receipts[0]
    edited = _seal(replace(receipt, rotation_increment_radians=tuple(2.*x for x in receipt.rotation_increment_radians)))
    changed = replace(first.state, motion_receipts=(edited,))
    with pytest.raises(ValueError, match="force/geometry/archive replay"):
        validate_bending_motion_state(mesh, origin, changed)


def test_json_restart_keeps_real_current_pair_and_complete_motion_source_history():
    mesh, origin, _, first, _ = motion_fixture()
    raw = bending_motion_to_dict(mesh, origin, first.state)
    restored = bending_motion_from_dict(json.loads(json.dumps(raw)), mesh, origin)
    assert restored == first.state
    _validate_pair(restored.current_surface, restored.current_inventory)
    assert restored.current_surface.time_myr > origin.time_myr
