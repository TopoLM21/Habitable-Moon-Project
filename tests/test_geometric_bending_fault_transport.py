"""Physical predictor receipts stay distinct from unaccepted finite geometry."""
from dataclasses import replace
from functools import lru_cache
import json
import math

import numpy as np
import pytest

from tests.test_geometric_bending_contact_dynamics import prepared, DT
from tectonics.geometric_basal_loading import assemble_geometric_basal_load
from tectonics.geometric_bending_contact_dynamics import advance_bending_contact_dynamics
from tectonics.geometric_bending_fault_transport import (
    MODEL, _seal_receipt, advance_bending_fault_transport,
    bending_fault_transport_from_dict, bending_fault_transport_to_dict,
    initialize_bending_fault_transport, reassemble_candidate_basal_load,
    validate_bending_fault_transport_state,
)
from tectonics.geometric_boundary import initialize_boundary, surface_digest
from tectonics.geometric_boundary_io import _validate_pair
from tectonics.geometric_fault_motion_support import UnresolvedFaultMotionSupport
from tectonics.geometric_surface import totals
from tectonics.thermal import SECONDS_PER_MYR


PROVENANCE = "Explicit reference-source bending predictor; candidate geometry awaits material admission"


@lru_cache(maxsize=None)
def fixture():
    surface, _, load, mechanics = prepared(mixed=True)
    inventory = initialize_boundary(surface)
    initial = initialize_bending_fault_transport(surface, inventory, mechanics, load)
    first = advance_bending_fault_transport(surface, initial, load, DT, provenance=PROVENANCE)
    return surface, load, initial, first


def _rodrigues(vector):
    """Independent closed-form exponential, without the atlas SO3 library."""
    vector = np.asarray(vector)
    theta = float(np.linalg.norm(vector))
    if theta == 0.:
        return np.eye(3)
    x, y, z = vector/theta
    cross = np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))
    return np.eye(3)+math.sin(theta)*cross+(1.-math.cos(theta))*(cross@cross)


def test_initial_material_epoch_retains_original_mechanics_source_and_inventory():
    surface, load, initial, _ = fixture()
    assert initial.initial_mechanics is initial.current_mechanics
    assert initial.reference_inventory == initialize_boundary(surface)
    assert initial.origin_surface_sha256 == surface_digest(surface)
    assert initial.receipts == ()
    assert initial.material_atlas.elapsed_time_myr == 0.
    assert validate_bending_fault_transport_state(surface, initial, load) == initial


def test_predictor_is_exactly_one_original_physical_step_and_one_same_dt_material_map():
    surface, load, initial, step = fixture()
    direct = advance_bending_contact_dynamics(surface, initial.current_mechanics, load, DT)
    assert step.predictor == direct
    assert step.state.current_mechanics == direct.state
    assert step.state.current_mechanics.step_count == initial.current_mechanics.step_count+1
    assert step.state.material_atlas.elapsed_time_myr == DT/SECONDS_PER_MYR
    assert step.state.reference_inventory is initial.reference_inventory
    assert step.state.current_mechanics.basal_source_sha256 == initial.reference_basal_source_sha256
    assert step.model == MODEL
    assert step.unresolved_material_pairing
    assert step.candidate_surface.phase == "advected_candidates"
    assert totals(step.candidate_surface.fragments) == totals(surface.fragments)
    # A candidate clock cannot masquerade as an accepted joint surface/archive.
    with pytest.raises(ValueError):
        _validate_pair(step.candidate_surface, step.state.reference_inventory)


def test_chronological_maps_follow_true_so3_increments_from_the_two_predictor_steps():
    surface, load, _, first = fixture()
    second = advance_bending_fault_transport(surface, first.state, load, DT, provenance=PROVENANCE)
    rotations = dict(second.state.material_atlas.owner_rotations)
    for owner in second.state.current_mechanics.plate_ids:
        first_vector = dict(first.state.receipts[0].rotation_vectors)[owner]
        second_vector = dict(second.state.receipts[1].rotation_vectors)[owner]
        np.testing.assert_allclose(rotations[owner], _rodrigues(second_vector)@_rodrigues(first_vector),
                                   rtol=0., atol=8*np.finfo(float).eps)
    assert second.state.current_mechanics.step_count == 2
    assert second.state.material_atlas.elapsed_time_myr == math.fsum((DT/SECONDS_PER_MYR, DT/SECONDS_PER_MYR))
    assert validate_bending_fault_transport_state(surface, second.state, load) == second.state


def test_full_arc_mismatch_bounds_cover_material_nodes_and_keep_anchor_defects_diagnostic():
    _, _, initial, first = fixture()
    atlas = first.state.material_atlas
    rotations = {owner: np.asarray(matrix) for owner, matrix in atlas.owner_rotations}
    traces = {trace.trace_id: trace for trace in atlas.reference_traces.traces}
    limits = {identity: (low, high) for identity, low, high in first.trace_material_pair_separation_m}
    assert any(high > 0. for _, _, high in first.trace_material_pair_separation_m)
    saved = dict(first.state.current_mechanics.states)
    old = dict(initial.current_mechanics.states)
    for point, defect in zip(atlas.reference_points, first.point_geometry_defects):
        trace = traces[point.trace_id]
        distance = atlas.radius_km*1000.*np.linalg.norm(
            (rotations[trace.overriding_plate]-rotations[trace.subducting_plate])@point.position)
        low, high = limits[point.trace_id]
        assert low-1e-11 <= distance <= high+1e-11
        assert defect.frozen_gap_increment_m == saved[point.point_id].gap_free_m-old[point.point_id].gap_free_m
        assert defect.frozen_jump_increment_m == saved[point.point_id].jump_free_m-old[point.point_id].jump_free_m
        assert defect.geometric_gap_increment_m-defect.frozen_gap_increment_m == defect.gap_anchor_defect_m
    assert any(np.linalg.norm(item.geometric_minus_linear_displacement_m) > 0.
               for item in first.point_geometry_defects)


def test_exact_diagnostic_uses_both_rotated_radial_tips_not_unrotated_radial_offsets():
    """Independent position oracle resolves the radial/SO3 cross contribution."""
    _, _, initial, first = fixture()
    atlas = first.state.material_atlas
    rotations = {owner: np.asarray(matrix, dtype=np.longdouble)
                 for owner, matrix in atlas.owner_rotations}
    traces = {trace.trace_id: trace for trace in atlas.reference_traces.traces}
    old_radial = {identity: (sub, over) for identity, sub, over
                  in initial.current_mechanics.transverse_displacement_m}
    new_radial = {identity: (sub, over) for identity, sub, over
                  in first.state.current_mechanics.transverse_displacement_m}
    radius = np.longdouble(atlas.radius_km*1000.)
    cross_terms = []
    for point, diagnostic in zip(atlas.reference_points, first.point_geometry_defects):
        trace = traces[point.trace_id]
        position = np.asarray(point.position, dtype=np.longdouble)
        sub, over = rotations[trace.subducting_plate], rotations[trace.overriding_plate]
        w_sub, w_over = map(np.longdouble, new_radial[point.trace_id])
        w_sub0, w_over0 = map(np.longdouble, old_radial[point.trace_id])
        exact = over.T@((radius+w_over)*(over@position)-(radius+w_sub)*(sub@position))
        exact -= (w_over0-w_sub0)*position
        np.testing.assert_allclose(diagnostic.geometric_relative_displacement_m,
                                   np.asarray(exact, dtype=float), rtol=0., atol=5e-11)
        assert diagnostic.geometric_gap_increment_m == pytest.approx(
            float(exact@np.asarray(point.normal, dtype=np.longdouble)), abs=5e-11)
        assert diagnostic.geometric_jump_increment_m == pytest.approx(
            float(exact@np.asarray(point.down_dip, dtype=np.longdouble)), abs=5e-11)
        unrotated = radius*(over.T@(over@position-sub@position))
        unrotated += ((w_over-w_sub)-(w_over0-w_sub0))*position
        cross_terms.append(float(np.linalg.norm(exact-unrotated)))
    # The fixture really distinguishes the full finite tips from the shortcut.
    assert max(cross_terms) > 1e-8


def test_restart_replays_same_physical_second_step_and_identical_serialized_history():
    surface, load, _, first = fixture()
    raw = bending_fault_transport_to_dict(surface, first.state, load)
    restored = bending_fault_transport_from_dict(json.loads(json.dumps(raw)), surface, load)
    direct = advance_bending_fault_transport(surface, first.state, load, DT, provenance=PROVENANCE)
    resumed = advance_bending_fault_transport(surface, restored, load, DT, provenance=PROVENANCE)
    assert resumed == direct
    assert json.dumps(bending_fault_transport_to_dict(surface, resumed.state, load), sort_keys=True) == json.dumps(
        bending_fault_transport_to_dict(surface, direct.state, load), sort_keys=True)


def test_initial_evolved_mechanics_has_explicit_epoch_without_resetting_work_or_point_history():
    surface, load, _, first = fixture()
    mechanics = first.predictor.state
    initialized = initialize_bending_fault_transport(surface, first.state.reference_inventory, mechanics, load)
    assert initialized.initial_mechanics == initialized.current_mechanics == mechanics
    assert initialized.material_atlas.elapsed_time_myr == 0.
    assert initialized.current_mechanics.cumulative_source_work_j != 0.
    next_step = advance_bending_fault_transport(surface, initialized, load, DT, provenance=PROVENANCE)
    assert next_step.state.current_mechanics.elapsed_time_s-mechanics.elapsed_time_s == DT
    assert next_step.state.current_mechanics.step_count-mechanics.step_count == 1
    assert len(next_step.state.receipts) == 1
    assert next_step.state.material_atlas.elapsed_time_myr == DT/SECONDS_PER_MYR
    assert validate_bending_fault_transport_state(surface, next_step.state, load) == next_step.state


def test_rehashed_duration_cannot_change_verified_source_work_or_geometry():
    surface, load, _, first = fixture()
    receipt = first.state.receipts[0]
    edited = _seal_receipt(replace(receipt, dt_s=2.*receipt.dt_s))
    forged = replace(first.state, receipts=(edited,))
    with pytest.raises(ValueError, match="mechanical replay"):
        validate_bending_fault_transport_state(surface, forged, load)


@pytest.mark.parametrize("name", ("source_work_j", "basal_dissipation_j", "before_step_count"))
def test_rehashed_work_and_step_receipts_must_match_original_predictor(name):
    surface, load, _, first = fixture()
    receipt = first.state.receipts[0]
    edited = _seal_receipt(replace(receipt, **{name: getattr(receipt, name)+1}))
    with pytest.raises(ValueError, match="mechanical replay"):
        validate_bending_fault_transport_state(surface, replace(first.state, receipts=(edited,)), load)


def test_wrong_source_and_unresolved_candidate_source_reassembly_are_rejected():
    surface, load, initial, first = fixture()
    from tests.test_geometric_distributed_contact_invariants import PARAMETERS, world
    mesh, _, lid = world()
    other = assemble_geometric_basal_load(mesh, surface, replace(PARAMETERS,
        convective_traction_pa=2.*PARAMETERS.convective_traction_pa),
                                         lid, load.provenance)
    with pytest.raises(ValueError, match="basal source changed"):
        validate_bending_fault_transport_state(surface, first.state, other)
    assert reassemble_candidate_basal_load(mesh, surface, initial, load) == load
    with pytest.raises(ValueError):
        reassemble_candidate_basal_load(mesh, surface, first.state, load)


def test_initializer_requires_actual_material_support_for_both_cantilever_banks():
    surface, _, load, mechanics = prepared(mixed=True)
    # A wide but otherwise legal frozen beam would reach beyond its own bank.
    changed = tuple(replace(item, sub=replace(item.sub, width_m=170000.))
                    for item in mechanics.bending_parameters)
    from tectonics.geometric_bending_contact_dynamics import initialize_bending_contact_dynamics
    from tests.test_geometric_bending_contact_dynamics import BOUNDS
    unsupported = initialize_bending_contact_dynamics(surface, mechanics.fault_traces, load, changed, **BOUNDS)
    with pytest.raises(UnresolvedFaultMotionSupport):
        initialize_bending_fault_transport(surface, initialize_boundary(surface), unsupported, load)


@pytest.mark.parametrize("dt", (True, 0., -1., float("nan")))
def test_invalid_physical_interval_never_advances_mechanics_or_material(dt):
    surface, load, initial, _ = fixture()
    with pytest.raises(ValueError):
        advance_bending_fault_transport(surface, initial, load, dt, provenance=PROVENANCE)
    assert initial.receipts == () and initial.material_atlas.elapsed_time_myr == 0.


def test_checkpoint_schema_rejects_unknown_fields_and_acceptance_claims():
    surface, load, _, first = fixture()
    raw = bending_fault_transport_to_dict(surface, first.state, load)
    with pytest.raises(ValueError, match="schema"):
        bending_fault_transport_from_dict(dict(raw, accepted=True), surface, load)
    with pytest.raises(ValueError, match="schema"):
        bending_fault_transport_from_dict(dict(raw, model="live_coupled_finite_contact_dynamics"), surface, load)
