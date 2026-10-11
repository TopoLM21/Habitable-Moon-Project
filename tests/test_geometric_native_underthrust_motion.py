"""Native force candidates retain the exact kinematic refusal and all memory."""
from dataclasses import asdict, replace

import numpy as np
import pytest

from analysis.geometric_native_underthrust_validation.run_native_underthrust_control import (
    encoded, independent_basal_dissipation, independent_material_ledger,
    independent_source_work, relative_error,
)
from analysis.geometric_underthrust_validation.underthrust_case import (
    ANCHOR_ERROR_BOUND_M, DT_S, SOURCE_INTERVALS, physical_inputs,
)
from tectonics.geometric_native_buried_support import native_buried_registry_to_dict
from tectonics.geometric_native_mantle_loading import assemble_native_physical_spatial_mantle_load
from tectonics.geometric_native_polygons import native_polygon_to_dict
from tectonics.geometric_native_surface import native_surface_to_dict
from tectonics.geometric_native_underthrust_motion import (
    UnresolvedNativeUnderthrustMotion, advance_native_underthrust_motion,
    initialize_native_underthrust_motion, validate_native_underthrust_motion_state,
)
from tectonics.geometric_native_underthrust_motion_io import (
    load_native_underthrust_motion_checkpoint, save_native_underthrust_motion_checkpoint,
)
from tectonics.geometric_underthrust_motion import HotBirthParameters


def accepted_snapshot(state):
    return encoded(dict(surface=native_surface_to_dict(state.current_surface),
        registry=native_buried_registry_to_dict(state.current_inventory),
        initial_inventory=asdict(state.initial_inventory), initial_load=asdict(state.initial_load),
        initial_mechanics=asdict(state.initial_mechanics),
        owner_histories=[dict(owner=owner, history=[asdict(step) for step in history])
                         for owner, history in state.owner_histories],
        nodes=[(identity, asdict(local)) for identity, local in state.states],
        radial=state.transverse_displacement_m, anchor_fields=state.free_anchor_fields,
        spatial_memory=asdict(state.spatial_memory), receipts=[asdict(row) for row in state.receipts],
        elapsed_time_s=state.elapsed_time_s, source_work_j=state.cumulative_source_work_j,
        basal_dissipation_j=state.cumulative_basal_dissipation_j,
        bending_dissipation_j=state.cumulative_bending_numerical_dissipation_j))


@pytest.fixture(scope="module")
def prepared():
    mesh, origin, inventory, seed, legacy_load = physical_inputs()
    state = initialize_native_underthrust_motion(mesh, origin, inventory, seed, legacy_load,
        birth_parameters=HotBirthParameters(2., 0., 0.,
            "Declared hot departure-footprint closure for the native physical control"),
        max_anchor_transport_error_m=ANCHOR_ERROR_BOUND_M)
    load = assemble_native_physical_spatial_mantle_load(mesh, state.current_surface,
                                                       SOURCE_INTERVALS[0])
    return mesh, origin, inventory, seed, legacy_load, state, load


@pytest.fixture(scope="module")
def refused_candidate(prepared):
    mesh, origin, _, _, _, state, load = prepared
    before = accepted_snapshot(state)
    with pytest.raises(UnresolvedNativeUnderthrustMotion) as refused:
        advance_native_underthrust_motion(mesh, origin, state, load, DT_S,
            provenance="Independent affine source drives the original native material pairs")
    assert accepted_snapshot(state) == before
    return refused.value


def test_native_initialization_preserves_the_prepared_physical_epoch(prepared):
    mesh, origin, inventory, seed, legacy_load, state, load = prepared
    assert state.initial_inventory == inventory
    assert state.initial_mechanics == seed
    assert state.initial_load == legacy_load
    assert state.states == seed.states
    assert state.transverse_displacement_m == seed.transverse_displacement_m
    assert state.preparation_work_j == seed.preparation_work_j
    assert state.owner_histories == ((0, ()), (1, ()))
    assert state.current_inventory.records == state.current_inventory.transactions == ()
    assert state.current_surface.time_myr == origin.time_myr
    assert load.source == legacy_load.source == SOURCE_INTERVALS[0]
    assert {sample.source_fragment_id for sample in load.samples} == {
        fragment.fragment_id for fragment in origin.fragments}
    validate_native_underthrust_motion_state(mesh, origin, state)


def test_unaccepted_force_candidate_has_independent_sample_work_and_drag(prepared, refused_candidate):
    _, _, _, _, _, before, load = prepared
    candidate = refused_candidate.candidate
    metrics = dict(candidate.candidate_diagnostics)
    increments = np.asarray(candidate.rotation_increment_radians)
    assert increments.shape == (2, 3)
    assert np.linalg.norm(increments[0]-increments[1]) > 0.
    assert candidate.dt_s == DT_S
    assert np.any(increments[:, :2] != 0.)
    assert relative_error(independent_source_work(before.current_surface, load, increments),
                          metrics["source_work_j"]) < 3e-14
    assert relative_error(independent_basal_dissipation(before.current_surface, load, increments,
                                                       DT_S), metrics["basal_dissipation_j"]) < 3e-14
    assert before.receipts == ()
    assert before.cumulative_source_work_j == before.cumulative_basal_dissipation_j == 0.
    assert before.elapsed_time_s == before.initial_mechanics.elapsed_time_s


def test_candidate_diagnostic_copies_cannot_change_retained_evidence(prepared, refused_candidate):
    _, _, _, _, _, state, _ = prepared
    candidate = refused_candidate.candidate
    before = accepted_snapshot(state)
    audit, reactions = candidate.admission_audit, candidate.points
    original_audit, original_reactions = encoded(audit), encoded(reactions)
    audit.clear()
    reactions.clear()
    assert encoded(candidate.admission_audit) == original_audit
    assert encoded(candidate.points) == original_reactions
    with pytest.raises(AttributeError):
        candidate.dt_s = 1.
    assert accepted_snapshot(state) == before


def test_true_mixed_convergence_preserves_positive_geometry_and_all_accepted_reservoirs(prepared,
        refused_candidate):
    _, origin, _, _, _, state, _ = prepared
    assert len(refused_candidate.overlaps) == 1
    overlap = refused_candidate.overlaps[0]
    assert {overlap.fragment_a, overlap.fragment_b} == {"octant:2", "octant:6"}
    assert overlap.reason == "mixed_convergence"
    assert overlap.area_km2 > 0.
    native_polygon_to_dict(overlap.geometry)
    assessments = refused_candidate.candidate.admission_audit["actual_overlap_assessments"]
    mixed = [row for candidate in assessments for row in candidate["receiver_arcs"]
             if row["status"] == "mixed_convergence"]
    assert mixed
    assert any(support["normal_velocity_max_km_per_myr"] > 0.
               for row in mixed for support in row["supports"])
    assert state.current_inventory.records == state.current_inventory.transactions == ()
    assert state.current_surface.time_myr == origin.time_myr
    ledger = independent_material_ledger(
        (fragment.parcel for fragment in origin.fragments),
        (fragment.parcel for fragment in state.current_surface.fragments), (), ())
    assert ledger["maximum_relative_residual"] == 0.


def test_initial_joint_restart_reproduces_the_same_full_force_refusal(prepared, refused_candidate,
        tmp_path):
    mesh, origin, _, _, _, state, _ = prepared
    metadata = {"control": "native loaded candidate refused", "accepted_native_intervals": 0}
    direct = save_native_underthrust_motion_checkpoint(tmp_path/"initial.json", mesh, origin, state,
                                                      provenance=metadata)
    restored_origin, restored, restored_metadata = load_native_underthrust_motion_checkpoint(direct,
                                                                                           mesh=mesh)
    assert restored_origin == origin and restored_metadata == metadata
    assert accepted_snapshot(restored) == accepted_snapshot(state)
    resaved = save_native_underthrust_motion_checkpoint(tmp_path/"resaved.json", mesh, restored_origin,
                                                       restored, provenance=restored_metadata)
    assert direct.read_bytes() == resaved.read_bytes()
    load = assemble_native_physical_spatial_mantle_load(mesh, restored.current_surface,
                                                       SOURCE_INTERVALS[0])
    with pytest.raises(UnresolvedNativeUnderthrustMotion) as resumed:
        advance_native_underthrust_motion(mesh, restored_origin, restored, load, DT_S,
            provenance="Independent affine source drives the original native material pairs")
    assert resumed.value.candidate.rotation_increment_radians == refused_candidate.candidate.rotation_increment_radians
    assert resumed.value.candidate.candidate_diagnostics == refused_candidate.candidate.candidate_diagnostics
    assert resumed.value.candidate.admission_audit == refused_candidate.candidate.admission_audit
    assert [(overlap.fragment_a, overlap.fragment_b, overlap.area_km2, overlap.reason)
            for overlap in resumed.value.overlaps] == [
        (overlap.fragment_a, overlap.fragment_b, overlap.area_km2, overlap.reason)
        for overlap in refused_candidate.overlaps]
    assert accepted_snapshot(restored) == accepted_snapshot(state)


@pytest.mark.parametrize("field,value", [("elapsed_time_s", True),
    ("cumulative_source_work_j", 1.), ("owner_histories", ((0, ()),))])
def test_forged_physical_state_refuses_before_creating_a_destination(prepared, tmp_path, field, value):
    mesh, origin, _, _, _, state, _ = prepared
    destination = tmp_path/"never_created"/"bad.json"
    with pytest.raises((ValueError, TypeError)):
        save_native_underthrust_motion_checkpoint(destination, mesh, origin,
                                                  replace(state, **{field: value}))
    assert not destination.parent.exists()
