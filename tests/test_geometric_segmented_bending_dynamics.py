"""Incremental contact/bending dynamics with an explicit changing-source ledger.

The fixed-branch oracle assembles its own beam/contact matrix.  Restart tests
also edit and reseal receipts, so a valid checksum cannot substitute for replay.
"""
from dataclasses import replace
from functools import lru_cache
import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import tectonics.fractional_dynamics as fractional_dynamics
from analysis.geometric_initiation_validation.admission_case import snapshot
from tests.test_geometric_bending_contact_dynamics import (
    DT, OPENING_DT, _bending_energy, _fixed_branch_solution, _stiffnesses, prepared,
)
from tests.test_geometric_distributed_contact_invariants import PARAMETERS, world
from tectonics.geometric_basal_loading import assemble_geometric_basal_load
from tectonics.geometric_bending_contact_dynamics import (
    advance_bending_contact_dynamics, continuous_bending_contact_gap_bounds,
)
from tectonics.geometric_bending_reference import (
    current_bending_reference, validate_reference_rotation,
)
from tectonics.geometric_contact_dynamics import UnresolvedContactDynamics
from tectonics.geometric_fault_traces import fault_trace_points
from tectonics.geometric_rigid_mantle_loading import (
    PhysicalRigidMantleLoad, PrescribedRigidMantleSource,
    assemble_physical_rigid_mantle_load,
)
from tectonics.geometric_segmented_bending_dynamics import (
    _seal_segment, advance_segmented_bending_dynamics, initialize_segmented_bending_dynamics,
    segmented_bending_dynamics_from_dict, segmented_bending_dynamics_to_dict,
    validate_segmented_bending_dynamics_state,
)
from tectonics.geometric_surface import rotate_surface
from tectonics.thermal import SECONDS_PER_MYR


PROVENANCE = "Explicit incremental-source control, preserving material contact history"


def _load(surface, *, amplitude=1., seed=None, drag=1., mesh=None):
    base_mesh, _, lid = world()
    parameters = replace(PARAMETERS,
        convective_traction_pa=PARAMETERS.convective_traction_pa*amplitude,
        basal_drag_pa_s_m=PARAMETERS.basal_drag_pa_s_m*drag,
        seed=PARAMETERS.seed if seed is None else seed)
    return assemble_geometric_basal_load(base_mesh if mesh is None else mesh,
        surface, parameters, lid, PROVENANCE)


@lru_cache(maxsize=None)
def fixture(*, evolved=False):
    surface, _, load, seed = prepared(mixed=True)
    if evolved:
        seed = advance_bending_contact_dynamics(surface, seed, load, DT).state
    state = initialize_segmented_bending_dynamics(surface, seed, load)
    return surface, seed, load, state


def _triples(value):
    return np.asarray(value, dtype=float)


def _assert_local_close(actual, expected, *, rel=2e-10):
    assert tuple(identity for identity, _ in actual) == tuple(identity for identity, _ in expected)
    from dataclasses import fields
    for (_, left), (_, right) in zip(actual, expected):
        for field in fields(left):
            a, b = getattr(left, field.name), getattr(right, field.name)
            if isinstance(a, float):
                assert a == pytest.approx(b, rel=rel, abs=5e-12)
            else:
                assert a == b


def _independent_increment_solution(surface, seed, previous, load, dt_s):
    """Series springs + exact beam-tip stiffness on a declared linear branch.

    Rotational coordinates are the new increment, whereas each radial tip is
    an absolute displacement.  Previous free anchors and plastic slips supply
    the affine reaction terms; no production network builder is used.
    """
    radius = surface.radius_km*1000.
    rotation_count = 3*len(load.plate_ids)
    dimension = rotation_count+2*len(seed.fault_traces.traces)
    old_tips = np.asarray([row[1:] for row in previous.transverse_displacement_m])
    qold = np.r_[np.zeros(rotation_count), old_tips.ravel()/radius]
    matrix = np.zeros((dimension, dimension))
    matrix[:rotation_count, :rotation_count] = np.asarray(load.drag_nm_s)/dt_s
    rhs = np.r_[load.torque_nm, np.zeros(dimension-rotation_count)]
    by_trace = {trace.trace_id: trace for trace in seed.fault_traces.traces}
    trace_positions = {trace.trace_id: index for index, trace in enumerate(seed.fault_traces.traces)}
    previous_points = dict(previous.states)
    for identity, stiffnesses in _stiffnesses(surface, seed).items():
        index = trace_positions[identity]
        for bank_index, stiffness in enumerate(stiffnesses):
            matrix[rotation_count+2*index+bank_index,
                   rotation_count+2*index+bank_index] = stiffness*radius**2
    records = []
    for point in fault_trace_points(surface, seed.fault_traces):
        trace, old = by_trace[point.trace_id], previous_points[point.point_id]
        index = trace_positions[point.trace_id]
        horizontal = np.zeros(dimension)
        lever = radius*np.cross(point.position, point.horizontal)
        sub, over = load.plate_ids.index(trace.subducting_plate), load.plate_ids.index(trace.overriding_plate)
        horizontal[3*sub:3*sub+3], horizontal[3*over:3*over+3] = -lever, lever
        transverse = np.zeros(dimension)
        transverse[rotation_count+2*index:rotation_count+2*index+2] = (-radius, radius)
        sine, cosine = math.sin(trace.dip_radians), math.cos(trace.dip_radians)
        normal, shear = sine*horizontal+cosine*transverse, cosine*horizontal-sine*transverse
        kn = 1./(1./trace.columns.normal_stiffness_pa_m+1./trace.law.normal_stiffness_pa_m)
        slipping = trace.law.cohesion_pa == trace.law.friction_dry == trace.law.friction_wet == 0.
        compliance = 1./trace.columns.shear_stiffness_pa_m+1./trace.law.tangential_stiffness_pa_m
        kt = 1./(compliance+(dt_s/trace.law.viscosity_pa_s_m if slipping else 0.))
        affine_normal = old.gap_free_m-float(normal@qold)
        affine_shear = old.jump_free_m-old.plastic_slip_m-float(shear@qold)
        matrix += point.area_m2*(kn*np.outer(normal, normal)+kt*np.outer(shear, shear))
        rhs -= point.area_m2*(kn*affine_normal*normal+kt*affine_shear*shear)
        records.append((point, trace, old, normal, shear, kn, kt, slipping))
    return np.linalg.solve(matrix, rhs), qold, tuple(records)


def test_evolved_seed_keeps_original_preparation_absolute_tips_and_local_history():
    surface, seed, load, state = fixture(evolved=True)
    assert state.initial_mechanics == seed
    assert state.initial_load == load
    assert state.states == seed.states
    assert state.transverse_displacement_m == seed.transverse_displacement_m
    assert state.rotation_path_material_radians == seed.rotation_radians
    assert state.elapsed_time_s == seed.elapsed_time_s
    assert state.step_count == seed.step_count
    assert state.cumulative_source_work_j == seed.cumulative_source_work_j
    assert state.cumulative_basal_dissipation_j == seed.cumulative_basal_dissipation_j
    assert state.cumulative_bending_numerical_dissipation_j == seed.cumulative_bending_numerical_dissipation_j
    assert state.segments == ()
    assert any(local.plastic_slip_m != 0. for _, local in state.states)
    assert validate_segmented_bending_dynamics_state(surface, state) == state


def test_same_source_increment_matches_frozen_solver_after_an_evolved_seed():
    surface, seed, load, initial = fixture(evolved=True)
    direct = advance_bending_contact_dynamics(surface, seed, load, DT)
    before = snapshot(surface), snapshot(initial)
    step = advance_segmented_bending_dynamics(surface, initial, load, DT, provenance=PROVENANCE)
    expected_increment = _triples(direct.state.rotation_radians)-_triples(seed.rotation_radians)
    np.testing.assert_allclose(step.rotation_increment_radians, expected_increment, rtol=2e-10, atol=5e-15)
    np.testing.assert_allclose(step.state.rotation_path_material_radians,
                               direct.state.rotation_radians, rtol=2e-10, atol=5e-15)
    np.testing.assert_allclose([row[1:] for row in step.state.transverse_displacement_m],
        [row[1:] for row in direct.state.transverse_displacement_m], rtol=2e-10, atol=5e-12)
    _assert_local_close(step.state.states, direct.state.states)
    for name in ("source_work_j", "basal_dissipation_j", "bending_work_j",
                 "bending_stored_energy_increment_j", "bending_numerical_dissipation_j"):
        assert getattr(step, name) == pytest.approx(getattr(direct, name), rel=2e-10)
    assert step.state.step_count == seed.step_count+1
    assert step.state.elapsed_time_s == seed.elapsed_time_s+DT
    assert step.torque_signal_relative_residual < 2e-9
    assert step.transverse_force_relative_residual < 2e-9
    assert step.incremental_energy_balance_relative_error < 2e-9
    assert step.minimum_certified_eigenvalue > 0.
    assert (snapshot(surface), snapshot(initial)) == before


def test_changing_source_work_is_sum_of_segments_and_cannot_be_current_torque_dot_path():
    surface, seed, _, initial = fixture(evolved=True)
    first_load = _load(surface, amplitude=.5)
    first = advance_segmented_bending_dynamics(surface, initial, first_load, DT,
                                              provenance=PROVENANCE)
    second_load = _load(surface, amplitude=1.3, drag=1.2)
    second = advance_segmented_bending_dynamics(surface, first.state, second_load, DT,
                                               provenance=PROVENANCE)
    expected = math.fsum((seed.cumulative_source_work_j,
        float(np.asarray(first_load.torque_nm)@_triples(first.rotation_increment_radians).ravel()),
        float(np.asarray(second_load.torque_nm)@_triples(second.rotation_increment_radians).ravel())))
    assert second.state.cumulative_source_work_j == pytest.approx(expected, rel=3e-12)
    false_total = float(np.asarray(second_load.torque_nm)
                        @_triples(second.state.rotation_path_material_radians).ravel())
    assert abs(false_total-expected) > .01*abs(expected)
    assert len(second.state.segments) == 2
    assert second.state.initial_mechanics == seed
    assert second.state.segments[0].current_load == first_load
    assert second.state.segments[1].current_load == second_load
    assert second.state.segments[0].source_work_j == first.source_work_j
    assert second.state.segments[1].source_work_j == second.source_work_j
    for _, local in second.state.states:
        assert local.elapsed_time_s == seed.initial_loading_dt_s+seed.elapsed_time_s+2.*DT
    local_work = math.fsum(local.cumulative_external_work_j for _, local in second.state.states)
    bending = _bending_energy(surface, replace(seed,
        transverse_displacement_m=second.state.transverse_displacement_m))
    accounted = math.fsum((local_work, bending, second.state.cumulative_basal_dissipation_j,
                          second.state.cumulative_bending_numerical_dissipation_j))
    assert accounted == pytest.approx(math.fsum((seed.preparation_work_j, expected)), rel=2e-9)
    forged = replace(second.state, cumulative_source_work_j=false_total)
    with pytest.raises(ValueError):
        validate_segmented_bending_dynamics_state(surface, forged)


def test_evolved_changing_load_matches_independent_locked_and_rate_sliding_matrix():
    surface, seed, _, initial = fixture(evolved=True)
    load = _load(surface, amplitude=.7, drag=1.3)
    solution, qold, records = _independent_increment_solution(surface, seed, initial, load, DT)
    step = advance_segmented_bending_dynamics(surface, initial, load, DT, provenance=PROVENANCE)
    radius = surface.radius_km*1000.
    np.testing.assert_allclose(step.rotation_increment_radians,
                               solution[:6].reshape((-1, 3)), rtol=2e-10, atol=5e-15)
    np.testing.assert_allclose([row[1:] for row in step.state.transverse_displacement_m],
                               solution[6:].reshape((-1, 2))*radius, rtol=2e-10, atol=5e-12)
    local = dict(step.state.states)
    branches = set()
    for point, trace, old, normal, shear, kn, kt, slipping in records:
        gap = old.gap_free_m+float(normal@(solution-qold))
        jump = old.jump_free_m+float(shear@(solution-qold))
        stress = kt*(jump-old.plastic_slip_m)
        assert local[point.point_id].traction_normal_pa == pytest.approx(kn*gap, rel=2e-10)
        assert local[point.point_id].traction_shear_pa == pytest.approx(stress, rel=2e-10, abs=5e-9)
        if slipping:
            assert local[point.point_id].plastic_slip_m == pytest.approx(
                old.plastic_slip_m+DT/trace.law.viscosity_pa_s_m*stress, rel=2e-10, abs=5e-12)
        else:
            cohesion = trace.law.cohesion_pa*(1.-trace.water+trace.water*trace.law.wet_cohesion_fraction)
            assert abs(stress) < cohesion
            assert local[point.point_id].plastic_slip_m == old.plastic_slip_m
        branches.add(slipping)
    assert branches == {False, True}


def test_full_arc_opening_rejects_a_solution_whose_quadrature_nodes_all_remain_closed():
    surface, trace_set, load, seed = prepared(mixed=True, gap=-1.)
    solution, records = _fixed_branch_solution(surface, load, seed, OPENING_DT)
    assert max(trace.initial_gap_m+float(normal@solution)
               for trace, normal, _, _, _ in records) < -.01
    transverse = tuple((trace.trace_id, float(sub), float(over)) for trace, (sub, over)
        in zip(trace_set.traces, solution[6:].reshape((-1, 2))*surface.radius_km*1000.))
    bounds = continuous_bending_contact_gap_bounds(surface, trace_set, load.plate_ids,
                                                    solution[:6].reshape((-1, 3)), transverse)
    assert max(high for _, _, high in bounds) > .01
    initial = initialize_segmented_bending_dynamics(surface, seed, load)
    before = snapshot(surface), snapshot(initial)
    with pytest.raises(UnresolvedContactDynamics):
        advance_segmented_bending_dynamics(surface, initial, load, OPENING_DT,
                                           provenance=PROVENANCE)
    assert (snapshot(surface), snapshot(initial)) == before


def test_later_opening_keeps_the_previous_accepted_segment_and_plastic_history():
    surface, _, _, initial = fixture()
    load = initial.initial_load
    accepted = advance_segmented_bending_dynamics(surface, initial, load, OPENING_DT,
                                                  provenance=PROVENANCE).state
    assert len(accepted.segments) == 1
    assert any(local.plastic_slip_m != 0. for _, local in accepted.states)
    before = snapshot(surface), snapshot(accepted)
    with pytest.raises(UnresolvedContactDynamics):
        advance_segmented_bending_dynamics(surface, accepted, load, OPENING_DT,
                                           provenance=PROVENANCE)
    assert (snapshot(surface), snapshot(accepted)) == before


def test_restart_replays_changed_sources_with_identical_second_step_and_bytes():
    surface, _, _, initial = fixture(evolved=True)
    first_load = _load(surface, amplitude=.6)
    first = advance_segmented_bending_dynamics(surface, initial, first_load, DT,
                                              provenance=PROVENANCE)
    raw = segmented_bending_dynamics_to_dict(surface, first.state)
    restored = segmented_bending_dynamics_from_dict(json.loads(json.dumps(raw)), surface)
    assert restored == first.state
    second_load = _load(surface, amplitude=1.1, drag=1.3)
    direct = advance_segmented_bending_dynamics(surface, first.state, second_load, DT,
                                               provenance=PROVENANCE)
    resumed = advance_segmented_bending_dynamics(surface, restored, second_load, DT,
                                                provenance=PROVENANCE)
    assert direct == resumed
    encode = lambda state: json.dumps(segmented_bending_dynamics_to_dict(surface, state),
                                      sort_keys=True, separators=(",", ":")).encode()
    assert encode(direct.state) == encode(resumed.state)


def test_truthful_rigid_sources_with_changed_omega_and_duration_keep_seed_and_union_restart():
    """A thermal/quadratic seed can retain two independently prescribed sources."""
    surface, seed, seed_load, initial = fixture(evolved=True)
    mesh, _, _ = world()
    first_source = PrescribedRigidMantleSource((2e-15, -1e-15, 1.5e-15), 1e14,
        "Explicit first spatial rigid mantle field; independent of solved plate motion")
    first_load = assemble_physical_rigid_mantle_load(mesh, surface, first_source)
    first_dt = .75*DT
    first = advance_segmented_bending_dynamics(surface, initial, first_load, first_dt,
                                              provenance=PROVENANCE)
    raw = segmented_bending_dynamics_to_dict(surface, first.state)
    assert raw["initial_load"]["format"] != raw["segments"][0]["current_load"]["format"]
    assert raw["segments"][0]["current_load"]["source"]["angular_velocity_rad_s"] == list(
        first_source.angular_velocity_rad_s)
    restored = segmented_bending_dynamics_from_dict(json.loads(json.dumps(raw)), surface)
    assert restored == first.state
    assert restored.initial_load == seed_load
    assert restored.initial_mechanics == seed
    assert isinstance(restored.segments[0].current_load, PhysicalRigidMantleLoad)
    second_source = PrescribedRigidMantleSource((-1.5e-15, .8e-15, 2.4e-15), 1.2e14,
        "Explicit changed rigid mantle field and independent positive drag")
    second_load = assemble_physical_rigid_mantle_load(mesh, surface, second_source)
    second_dt = 1.3*DT
    direct = advance_segmented_bending_dynamics(surface, first.state, second_load, second_dt,
                                               provenance=PROVENANCE)
    resumed = advance_segmented_bending_dynamics(surface, restored, second_load, second_dt,
                                                provenance=PROVENANCE)
    assert direct == resumed
    first_work = float(np.asarray(first_load.torque_nm)
                       @_triples(first.rotation_increment_radians).ravel())
    second_work = float(np.asarray(second_load.torque_nm)
                        @_triples(direct.rotation_increment_radians).ravel())
    assert first.source_work_j == pytest.approx(first_work, rel=3e-12)
    assert direct.source_work_j == pytest.approx(second_work, rel=3e-12)
    expected = math.fsum((seed.cumulative_source_work_j, first_work, second_work))
    assert direct.state.cumulative_source_work_j == pytest.approx(expected, rel=3e-12)
    false_total = float(np.asarray(second_load.torque_nm)
                        @_triples(direct.state.rotation_path_material_radians).ravel())
    assert abs(false_total-expected) > .01*abs(expected)
    assert direct.state.elapsed_time_s == math.fsum((seed.elapsed_time_s, first_dt, second_dt))
    assert [receipt.dt_s for receipt in direct.state.segments] == [first_dt, second_dt]
    assert all(local.elapsed_time_s == seed.initial_loading_dt_s+direct.state.elapsed_time_s
               for _, local in direct.state.states)
    bending = _bending_energy(surface, replace(seed,
        transverse_displacement_m=direct.state.transverse_displacement_m))
    accounted = math.fsum((*(local.cumulative_external_work_j for _, local in direct.state.states),
        bending, direct.state.cumulative_basal_dissipation_j,
        direct.state.cumulative_bending_numerical_dissipation_j))
    assert accounted == pytest.approx(math.fsum((seed.preparation_work_j, expected)), rel=2e-9)
    direct_raw = segmented_bending_dynamics_to_dict(surface, direct.state)
    resumed_raw = segmented_bending_dynamics_to_dict(surface, resumed.state)
    assert json.dumps(direct_raw, sort_keys=True) == json.dumps(resumed_raw, sort_keys=True)
    assert all(isinstance(receipt.current_load, PhysicalRigidMantleLoad)
               for receipt in direct.state.segments)


@lru_cache(maxsize=1)
def _accepted_epoch_fixture():
    surface, _, seed_load, seed = prepared(gap=0.)
    initial = initialize_segmented_bending_dynamics(surface, seed, seed_load)
    mesh, _, _ = world()
    source = PrescribedRigidMantleSource((2e-12, -1e-12, 1.5e-12), 1e14,
        "Explicit rigid mantle field for finite common material motion")
    first_load = assemble_physical_rigid_mantle_load(mesh, surface, source)
    first = advance_segmented_bending_dynamics(surface, initial, first_load, DT,
                                              provenance=PROVENANCE)
    increment = np.asarray(first.rotation_increment_radians)
    np.testing.assert_array_equal(increment[0], increment[1])
    omega = {owner: increment[index]*SECONDS_PER_MYR/DT
             for index, owner in enumerate(first.state.plate_ids)}
    elapsed = DT/SECONDS_PER_MYR
    current_surface = rotate_surface(surface, omega, elapsed)
    assert current_surface.phase == "partition"
    rotation = Rotation.from_rotvec(increment[0]).as_matrix()
    return surface, seed, initial, first, current_surface, rotation, elapsed


def test_positive_accepted_reference_epoch_replays_exact_geometry_source_and_original_seed():
    surface, seed, _, first, current_surface, rotation, elapsed = _accepted_epoch_fixture()
    mesh, _, _ = world()
    second_source = PrescribedRigidMantleSource((-1.1e-12, .6e-12, 1.8e-12), 1.2e14,
        "Changed explicit global mantle field on accepted moved material")
    current_load = assemble_physical_rigid_mantle_load(mesh, current_surface, second_source)
    second_dt = .7*DT
    step = advance_segmented_bending_dynamics(surface, first.state, current_load, second_dt,
        reference_rotation=rotation, reference_elapsed_time_myr=elapsed,
        reference_surface=current_surface, reference_roundoff_steps=1, provenance=PROVENANCE)
    receipt = step.state.segments[-1]
    reference = receipt.reference_configuration
    assert reference.surface == current_surface
    assert reference.uses_supplied_reference_surface
    assert reference.reference_elapsed_time_myr == elapsed
    assert reference.reference_roundoff_steps == 1
    assert reference.surface.time_myr == surface.time_myr+elapsed
    assert reference.physical_dt_s == 0.
    for original, moved in zip(surface.fragments, reference.surface.fragments):
        assert moved.parcel.age_myr == original.parcel.age_myr+elapsed
        assert replace(moved.parcel, age_myr=original.parcel.age_myr) == original.parcel
    assert receipt.current_load == current_load
    assert receipt.current_load.surface_sha256 == reference.current_surface_sha256
    assert step.state.initial_mechanics == seed
    assert seed.surface_sha256 != reference.current_surface_sha256
    np.testing.assert_array_equal(step.rotation_increment_radians[0], step.rotation_increment_radians[1])
    second_work = float(np.asarray(current_load.torque_nm)
                        @_triples(step.rotation_increment_radians).ravel())
    assert step.source_work_j == pytest.approx(second_work, rel=3e-12)
    assert step.state.cumulative_source_work_j == pytest.approx(math.fsum((
        seed.cumulative_source_work_j, first.source_work_j, second_work)), rel=3e-12)
    assert step.state.elapsed_time_s == DT+second_dt
    assert all(local.gap_free_m == local.jump_free_m == local.plastic_slip_m == 0.
               for _, local in step.state.states)
    assert validate_segmented_bending_dynamics_state(surface, step.state) == step.state
    raw = segmented_bending_dynamics_to_dict(surface, step.state)
    assert raw["segments"][1]["reference_configuration"]["reference_surface"] is not None
    resumed = segmented_bending_dynamics_from_dict(json.loads(json.dumps(raw)), surface)
    assert resumed == step.state
    assert resumed.segments[1].reference_configuration.surface == current_surface
    assert json.dumps(segmented_bending_dynamics_to_dict(surface, resumed), sort_keys=True) == json.dumps(raw, sort_keys=True)


@pytest.mark.parametrize("mutation", ["age", "source_hash"])
def test_positive_epoch_rejects_changed_material_age_or_source_digest_atomically(mutation):
    surface, _, _, first, current_surface, rotation, elapsed = _accepted_epoch_fixture()
    mesh, _, _ = world()
    source = PrescribedRigidMantleSource((1e-12, .8e-12, -1e-12), 1e14,
        "Explicit next source on the accepted material epoch")
    if mutation == "age":
        fragment = current_surface.fragments[0]
        current_surface = replace(current_surface, fragments=(replace(fragment,
            parcel=replace(fragment.parcel, age_myr=fragment.parcel.age_myr+.001)),
            *current_surface.fragments[1:]))
    load = assemble_physical_rigid_mantle_load(mesh, current_surface, source)
    if mutation == "source_hash":
        load = replace(load, load_sha256="0"*64)
    before = snapshot(surface), snapshot(first.state), snapshot(current_surface)
    with pytest.raises(ValueError):
        advance_segmented_bending_dynamics(surface, first.state, load, DT,
            reference_rotation=rotation, reference_elapsed_time_myr=elapsed,
            reference_surface=current_surface, reference_roundoff_steps=1, provenance=PROVENANCE)
    assert (snapshot(surface), snapshot(first.state), snapshot(current_surface)) == before


def test_fifty_nine_so3_products_keep_certified_saved_matrix_through_core_replay_and_json():
    """A verified product budget must reach every outer serialization gate."""
    surface, _, seed_load, seed = prepared(gap=0.)
    state = initialize_segmented_bending_dynamics(surface, seed, seed_load)
    one = Rotation.from_rotvec((.2, -.1, .3)).as_matrix()
    accumulated = np.eye(3)
    for _ in range(59):
        accumulated = one@accumulated
    error = max(float(np.max(np.abs(accumulated.T@accumulated-np.eye(3)))),
                abs(float(np.linalg.det(accumulated))-1.))
    # The original cloud counterexample exceeds this strict one-matrix gate.
    # On arithmetic implementations with less drift, retained-matrix replay
    # below still verifies that no projection was silently introduced.
    if error > 64*np.finfo(float).eps:
        with pytest.raises(ValueError):
            validate_reference_rotation(accumulated)
    reference = current_bending_reference(surface, seed, accumulated,
                                          reference_roundoff_steps=59)
    mesh, _, _ = world()
    load = assemble_physical_rigid_mantle_load(mesh, reference.surface,
        PrescribedRigidMantleSource((2e-15, -1e-15, 1.5e-15), 1e14,
            "Explicit rigid mantle source in the certified accumulated coordinate view"))
    step = advance_segmented_bending_dynamics(surface, state, load, DT,
        reference_rotation=accumulated, reference_roundoff_steps=59, provenance=PROVENANCE)
    receipt = step.state.segments[-1]
    np.testing.assert_array_equal(receipt.reference_rotation, accumulated)
    np.testing.assert_array_equal(receipt.reference_configuration.rotation, accumulated)
    assert receipt.reference_configuration.reference_roundoff_steps == 59
    assert validate_segmented_bending_dynamics_state(surface, step.state) == step.state
    raw = segmented_bending_dynamics_to_dict(surface, step.state)
    restored = segmented_bending_dynamics_from_dict(json.loads(json.dumps(raw)), surface)
    assert restored == step.state
    np.testing.assert_array_equal(restored.segments[-1].reference_rotation, accumulated)
    assert json.dumps(segmented_bending_dynamics_to_dict(surface, restored), sort_keys=True) == json.dumps(raw, sort_keys=True)


def test_real_common_material_motion_can_accumulate_past_the_small_increment_bound():
    """Finite rigid pose is distinct from each bounded deformation increment."""
    from tectonics.geometric_bending_motion import (
        advance_bending_motion, initialize_bending_motion, validate_bending_motion_state,
    )
    from tectonics.geometric_boundary import initialize_boundary
    mesh, _, _ = world()
    surface, _, seed_load, seed = prepared(gap=0.)
    state = initialize_bending_motion(surface, initialize_boundary(surface), seed, seed_load)
    # This mantle field is declared before any plate velocity is computed.
    omega = (3e-12, -2e-12, 4e-12)
    source = PrescribedRigidMantleSource(omega, 1e14,
        "Declared common mantle flow for finite pose beyond the per-step chart bound")
    expected_increment = np.asarray(omega)*DT
    assert np.linalg.norm(expected_increment) < seed.max_rotation_radians
    for _ in range(3):
        load = assemble_physical_rigid_mantle_load(mesh, state.current_surface, source)
        step = advance_bending_motion(mesh, surface, state, load, DT, provenance=PROVENANCE)
        np.testing.assert_array_equal(step.mechanical_step.rotation_increment_radians[0], expected_increment)
        np.testing.assert_array_equal(step.mechanical_step.rotation_increment_radians[1], expected_increment)
        state = step.state
    assert np.linalg.norm(Rotation.from_matrix(state.common_rotation).as_rotvec()) > seed.max_rotation_radians
    assert np.max(np.linalg.norm(state.mechanics.rotation_path_material_radians, axis=1)) > seed.max_rotation_radians
    assert all(local.gap_free_m == local.jump_free_m == local.plastic_slip_m == 0.
               for _, local in state.mechanics.states)
    assert all(sub == over == 0. for _, sub, over in state.mechanics.transverse_displacement_m)
    assert len(state.current_inventory.transactions) == 3
    assert len(state.motion_receipts) == state.mechanics.segment_count == 3
    assert state.current_surface.time_myr > surface.time_myr
    assert validate_bending_motion_state(mesh, surface, state) == state


def test_differential_segment_increment_keeps_its_explicit_small_chart_bound():
    """Removing a cumulative rigid limit does not relax differential steps."""
    from tectonics.geometric_segmented_bending_dynamics import _domain
    surface, seed, _, initial = fixture(evolved=True)
    radial = np.asarray([row[1:] for row in initial.transverse_displacement_m])
    increments = np.array(((1.01*seed.max_rotation_radians, 0., 0.),
                           (-1.01*seed.max_rotation_radians, 0., 0.)))
    before = snapshot(surface), snapshot(initial)
    with pytest.raises(UnresolvedContactDynamics, match="small-increment bound"):
        _domain(initial, increments, radial)
    assert (snapshot(surface), snapshot(initial)) == before


def test_evolved_mechanics_and_material_path_are_covariant_under_a_passive_reference(monkeypatch):
    surface, seed, _, initial = fixture(evolved=True)
    load = _load(surface, amplitude=.8)
    original = advance_segmented_bending_dynamics(surface, initial, load, DT,
                                                 provenance=PROVENANCE)
    rotation = Rotation.from_rotvec((.31, -.17, .24)).as_matrix()
    reference = current_bending_reference(surface, seed, rotation)
    mesh, _, lid = world()
    rotated_mesh = replace(mesh, vertices=mesh.vertices@rotation.T,
                           centroids=mesh.centroids@rotation.T)
    source = fractional_dynamics.prescribed_vertex_traction

    def same_physical_source_in_current_coordinates(current_mesh, **kwargs):
        assert current_mesh is rotated_mesh
        return source(mesh, **kwargs)@rotation.T

    monkeypatch.setattr(fractional_dynamics, "prescribed_vertex_traction",
                        same_physical_source_in_current_coordinates)
    rotated_load = assemble_geometric_basal_load(rotated_mesh, reference.surface,
        replace(PARAMETERS, convective_traction_pa=.8*PARAMETERS.convective_traction_pa),
        lid, "Same explicit source represented in the passive SO3 coordinate frame")
    changed = advance_segmented_bending_dynamics(surface, initial, rotated_load, DT,
        reference_rotation=rotation, provenance=PROVENANCE)
    np.testing.assert_allclose(changed.rotation_increment_radians,
        _triples(original.rotation_increment_radians)@rotation.T, rtol=5e-9, atol=5e-15)
    np.testing.assert_allclose(changed.state.rotation_path_material_radians,
        original.state.rotation_path_material_radians, rtol=5e-9, atol=5e-15)
    np.testing.assert_allclose([row[1:] for row in changed.state.transverse_displacement_m],
        [row[1:] for row in original.state.transverse_displacement_m], rtol=5e-9, atol=5e-12)
    _assert_local_close(changed.state.states, original.state.states, rel=5e-9)
    assert changed.source_work_j == pytest.approx(original.source_work_j, rel=5e-9)
    assert changed.bending_energy_j == pytest.approx(original.bending_energy_j, rel=5e-9)
    assert changed.reference_surface_sha256 == reference.current_surface_sha256
    for old, current in zip(original.points, changed.points):
        np.testing.assert_allclose(current.force_on_sub_n, rotation@old.force_on_sub_n,
                                   rtol=5e-9, atol=.1)


@pytest.mark.parametrize("mutation", ["duration", "source", "source_work", "increment",
                                       "sequence", "before_hash", "duration_bool"])
def test_resealed_receipt_must_replay_its_recorded_mechanics(mutation):
    surface, _, _, initial = fixture(evolved=True)
    load = _load(surface, amplitude=.6)
    evolved = advance_segmented_bending_dynamics(surface, initial, load, DT,
                                                 provenance=PROVENANCE).state
    segment = evolved.segments[0]
    values = {}
    if mutation == "duration":
        values["dt_s"] = 2.*segment.dt_s
    elif mutation == "duration_bool":
        values["dt_s"] = True
    elif mutation == "source":
        values["current_load"] = _load(surface, amplitude=.3)
    elif mutation == "source_work":
        values["source_work_j"] = segment.source_work_j*1.001
    elif mutation == "increment":
        vectors = [list(row) for row in segment.rotation_increment_radians]
        vectors[0][0] += 1e-7
        values["rotation_increment_radians"] = tuple(map(tuple, vectors))
    elif mutation == "sequence":
        values["sequence"] = segment.sequence+1
    else:
        values["before_mechanics_sha256"] = "0"*64
    before = snapshot(surface), snapshot(evolved)
    with pytest.raises(ValueError):
        edited = _seal_segment(replace(segment, **values))
        forged = replace(evolved, segments=(edited,))
        validate_segmented_bending_dynamics_state(surface, forged)
    assert (snapshot(surface), snapshot(evolved)) == before


@pytest.mark.parametrize("dt", [True, 0., -1., float("nan"), float("inf")])
def test_invalid_duration_rejects_atomically(dt):
    surface, _, load, initial = fixture(evolved=True)
    before = snapshot(surface), snapshot(initial)
    with pytest.raises(ValueError):
        advance_segmented_bending_dynamics(surface, initial, load, dt, provenance=PROVENANCE)
    assert (snapshot(surface), snapshot(initial)) == before


@pytest.mark.parametrize("edit", ["origin_clock", "source_clock", "rotation_scale", "reflection"])
def test_invalid_source_surface_or_reference_support_rejects_atomically(edit):
    surface, _, load, initial = fixture(evolved=True)
    before = snapshot(surface), snapshot(initial)
    supplied_surface, supplied_load, rotation = surface, load, None
    if edit == "origin_clock":
        supplied_surface = replace(surface, time_myr=surface.time_myr+1.)
    elif edit == "source_clock":
        supplied_load = _load(replace(surface, time_myr=surface.time_myr+1.))
    elif edit == "rotation_scale":
        rotation = np.eye(3)*1.001
    else:
        rotation = np.diag((1., 1., -1.))
    with pytest.raises(ValueError):
        advance_segmented_bending_dynamics(supplied_surface, initial, supplied_load, DT,
            reference_rotation=rotation, provenance=PROVENANCE)
    assert (snapshot(surface), snapshot(initial)) == before


@pytest.mark.parametrize("mutation", ["missing", "extra", "format", "clock_bool",
                                       "point", "tip", "source_work", "harmonic"])
def test_restart_schema_and_preserved_seed_binding_are_strict(mutation):
    surface, _, _, initial = fixture(evolved=True)
    raw = json.loads(json.dumps(segmented_bending_dynamics_to_dict(surface, initial)))
    if mutation == "missing":
        raw.pop("initial_mechanics")
    elif mutation == "extra":
        raw["unrecorded_source_transition"] = True
    elif mutation == "format":
        raw["format"] = "geometric-frozen-bending-contact-dynamics-1"
    elif mutation == "clock_bool":
        raw["elapsed_time_s"] = True
    elif mutation == "point":
        raw["states"].pop()
    elif mutation == "tip":
        raw["transverse_displacement_m"][0][1] = True
    elif mutation == "source_work":
        raw["cumulative_source_work_j"] *= 1.001
    else:
        raw["free_anchor_harmonics"][0]["gap"][0] += .01
    with pytest.raises(ValueError):
        segmented_bending_dynamics_from_dict(raw, surface)
