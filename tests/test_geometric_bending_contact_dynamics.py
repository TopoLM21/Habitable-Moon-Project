"""Preparation, work accounting and strict restart of contact-driven bending."""
from dataclasses import replace
from functools import lru_cache
import json
import math

import numpy as np
import pytest

from analysis.geometric_initiation_validation.admission_case import snapshot
from tests.test_geometric_distributed_contact_invariants import (
    PARAMETERS, PROVENANCE, traces, world,
)
from tectonics.geometric_basal_loading import assemble_geometric_basal_load
from tectonics.geometric_bending import ElasticBendingBankParameters, FaultBendingParameters
from tectonics.geometric_bending_contact_dynamics import (
    advance_bending_contact_dynamics, bending_contact_dynamics_from_dict,
    bending_contact_dynamics_to_dict, continuous_bending_contact_gap_bounds,
    initialize_bending_contact_dynamics,
    validate_bending_contact_dynamics_state,
)
from tectonics.geometric_boundary import initialize_boundary
from tectonics.geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint
from tectonics.geometric_contact_dynamics import UnresolvedContactDynamics
from tectonics.geometric_contact_mechanics import solve_compressed_contact
from tectonics.geometric_fault_traces import (
    TraceQuadrature, declare_fault_trace_set, fault_trace_points,
)


DT = 1e9
OPENING_DT = 1e10
BOUNDS = dict(initial_loading_dt_s=1e6, max_rotation_radians=.01,
              max_tip_slope=.01, max_bending_strain=.001,
              max_deflection_radius_ratio=.001)


@lru_cache(maxsize=None)
def prepared(*, mixed=False, gap=-2.):
    """Eight material octants, four physical arcs and only eight contact nodes."""
    mesh, surface, lid = world()
    load = assemble_geometric_basal_load(mesh, surface, PARAMETERS, lid, PROVENANCE)
    declared = traces(gap=gap)
    if mixed:
        # Explicit strength contrasts guarantee that both constitutive branches
        # are exercised without selecting a strength from the computed answer.
        declared = tuple(replace(trace, law=replace(trace.law, cohesion_pa=0.,
            friction_dry=0., friction_wet=0.)) if index % 2 else trace
            for index, trace in enumerate(declared))
    trace_set = declare_fault_trace_set(surface, declared, TraceQuadrature(1, 2))
    sub = ElasticBendingBankParameters(10000., 2000., 7e10, .25,
                                      "Explicit synthetic free cantilever, subducting bank")
    over = ElasticBendingBankParameters(15000., 3000., 8e10, .3,
                                       "Explicit synthetic free cantilever, overriding bank")
    banks = tuple(FaultBendingParameters(trace.trace_id, sub, over)
                  for trace in trace_set.traces)
    state = initialize_bending_contact_dynamics(surface, trace_set, load, banks, **BOUNDS)
    return surface, trace_set, load, state


def _stiffnesses(surface, state):
    """Cantilever end stiffness from the analytic beam solution, independently."""
    declared = {trace.trace_id: trace for trace in state.fault_traces.traces}
    answer = {}
    for parameters in state.bending_parameters:
        trace = declared[parameters.trace_id]
        start, end = np.asarray(trace.start), np.asarray(trace.end)
        angle = math.atan2(float(np.linalg.norm(np.cross(start, end))), float(start@end))
        length = surface.radius_km*1000.*angle
        answer[parameters.trace_id] = tuple(
            length*bank.young_modulus_pa*bank.elastic_thickness_m**3
            /(4.*(1.-bank.poisson_ratio**2)*bank.width_m**3)
            for bank in (parameters.sub, parameters.over))
    return answer


def _bending_energy(surface, state):
    stiffness = _stiffnesses(surface, state)
    return math.fsum(.5*k*w*w
        for trace_id, sub, over in state.transverse_displacement_m
        for k, w in zip(stiffness[trace_id], (sub, over)))


def _fixed_branch_solution(surface, load, state, dt_s):
    """Independent matrix for high-cohesion lock and zero-strength rate slip."""
    radius = surface.radius_km*1000.
    count = 3*len(load.plate_ids)
    declared = {trace.trace_id: trace for trace in state.fault_traces.traces}
    trace_indices = {trace.trace_id: index for index, trace in enumerate(state.fault_traces.traces)}
    dimension = count+2*len(declared)
    matrix = np.zeros((dimension, dimension))
    matrix[:count, :count] = np.asarray(load.drag_nm_s)/dt_s
    rhs = np.r_[np.asarray(load.torque_nm), np.zeros(dimension-count)]
    for trace_id, pair in _stiffnesses(surface, state).items():
        index = trace_indices[trace_id]
        for bank_index, stiffness in enumerate(pair):
            matrix[count+2*index+bank_index, count+2*index+bank_index] = stiffness*radius**2
    records = []
    for point in fault_trace_points(surface, state.fault_traces):
        trace = declared[point.trace_id]
        horizontal = np.zeros(dimension)
        lever = radius*np.cross(point.position, point.horizontal)
        sub, over = load.plate_ids.index(trace.subducting_plate), load.plate_ids.index(trace.overriding_plate)
        horizontal[3*sub:3*sub+3], horizontal[3*over:3*over+3] = -lever, lever
        transverse = np.zeros(dimension)
        index = trace_indices[point.trace_id]
        transverse[count+2*index:count+2*index+2] = (-radius, radius)
        sine, cosine = math.sin(trace.dip_radians), math.cos(trace.dip_radians)
        normal = sine*horizontal+cosine*transverse
        shear = cosine*horizontal-sine*transverse
        kn = 1./(1./trace.columns.normal_stiffness_pa_m+1./trace.law.normal_stiffness_pa_m)
        shear_compliance = 1./trace.columns.shear_stiffness_pa_m+1./trace.law.tangential_stiffness_pa_m
        slipping = trace.law.cohesion_pa == trace.law.friction_dry == trace.law.friction_wet == 0.
        if slipping:
            shear_compliance += dt_s/trace.law.viscosity_pa_s_m
        kt = 1./shear_compliance
        matrix += point.area_m2*(kn*np.outer(normal, normal)+kt*np.outer(shear, shear))
        rhs -= point.area_m2*(kn*trace.initial_gap_m*normal+kt*trace.initial_jump_m*shear)
        records.append((trace, normal, shear, kt, slipping))
    return np.linalg.solve(matrix, rhs), tuple(records)


def test_preparation_retains_explicit_initial_clamp_work_without_a_bending_reset():
    surface, trace_set, load, state = prepared()
    assert state.step_count == 0 and state.elapsed_time_s == 0.
    assert state.cumulative_source_work_j == state.cumulative_basal_dissipation_j == 0.
    assert state.cumulative_bending_numerical_dissipation_j == 0.
    assert all(sub == over == 0. for _, sub, over in state.transverse_displacement_m)
    np.testing.assert_array_equal(state.rotation_radians, np.zeros((2, 3)))
    by_trace = {trace.trace_id: trace for trace in trace_set.traces}
    expected = []
    for point in fault_trace_points(surface, trace_set):
        trace = by_trace[point.trace_id]
        local = solve_compressed_contact(columns=trace.columns, law=trace.law,
            area_m2=point.area_m2, gap_free_m=trace.initial_gap_m,
            jump_free_m=trace.initial_jump_m, dt_s=BOUNDS["initial_loading_dt_s"],
            water=trace.water)
        expected.append((point.point_id, local.state))
    assert state.states == tuple(expected)
    assert state.preparation_work_j == math.fsum(local.cumulative_external_work_j
                                                for _, local in expected) > 0.
    assert validate_bending_contact_dynamics_state(surface, state, load) == state


def test_mixed_contact_step_accounts_for_beam_energy_and_backward_euler_work():
    surface, _, load, initial = prepared(mixed=True)
    before = snapshot(surface), snapshot(initial)
    previous = initial
    accumulated_bending_remainder = 0.
    for _ in range(2):
        step = advance_bending_contact_dynamics(surface, previous, load, DT)
        stiffness = _stiffnesses(surface, previous)
        old = {trace_id: (sub, over) for trace_id, sub, over in previous.transverse_displacement_m}
        remainder = math.fsum(.5*k*(w-old[trace_id][index])**2
            for trace_id, sub, over in step.state.transverse_displacement_m
            for index, (k, w) in enumerate(zip(stiffness[trace_id], (sub, over))))
        delta_energy = _bending_energy(surface, step.state)-_bending_energy(surface, previous)
        assert step.bending_numerical_dissipation_j == pytest.approx(remainder, rel=3e-12)
        assert step.bending_stored_energy_increment_j == pytest.approx(delta_energy, rel=3e-12)
        assert step.bending_energy_j == pytest.approx(_bending_energy(surface, step.state), rel=3e-12)
        assert step.bending_work_j == pytest.approx(delta_energy+remainder, rel=3e-12)
        assert remainder >= 0.
        accumulated_bending_remainder = math.fsum((accumulated_bending_remainder, remainder))
        assert step.state.cumulative_bending_numerical_dissipation_j == pytest.approx(
            accumulated_bending_remainder, rel=3e-12)
        contact_stored = math.fsum(point.step.bulk_energy_increment_j
                                  +point.step.contact_energy_increment_j for point in step.points)
        contact_remainder = math.fsum(point.step.backward_euler_remainder_j for point in step.points)
        assert step.stored_energy_increment_j == pytest.approx(contact_stored+delta_energy, rel=3e-12)
        assert step.numerical_dissipation_j == pytest.approx(contact_remainder+remainder, rel=3e-12)
        accounted = math.fsum((step.basal_dissipation_j, step.stored_energy_increment_j,
            step.friction_work_j, step.viscous_work_j, step.numerical_dissipation_j))
        assert abs(step.source_work_j-accounted) <= (
            5e-10*max(abs(step.source_work_j), abs(accounted))+step.energy_rounding_tolerance_j)
        assert step.torque_signal_relative_residual < 2e-9
        assert step.incremental_energy_balance_relative_error < 2e-9
        assert step.minimum_certified_eigenvalue > 0.
        assert any(point.step.slip_increment_m == 0. for point in step.points)
        assert any(point.step.slip_increment_m != 0. for point in step.points)
        assert any(sub != 0. or over != 0.
                   for _, sub, over in step.state.transverse_displacement_m)
        source = float(np.asarray(load.torque_nm)@np.asarray(step.state.rotation_radians).ravel())
        assert step.state.cumulative_source_work_j == pytest.approx(source, rel=3e-12)
        all_contact_work = math.fsum(local.cumulative_external_work_j
                                    for _, local in step.state.states)
        cumulative_accounted = math.fsum((all_contact_work,
            step.state.cumulative_basal_dissipation_j, _bending_energy(surface, step.state),
            step.state.cumulative_bending_numerical_dissipation_j))
        cumulative_source = math.fsum((step.state.preparation_work_j,
                                      step.state.cumulative_source_work_j))
        assert cumulative_accounted == pytest.approx(cumulative_source, rel=2e-9)
        previous = step.state
    assert (snapshot(surface), snapshot(initial)) == before


def test_joint_checkpoint_resume_is_exact_and_keeps_beam_and_point_histories(tmp_path):
    surface, _, load, initial = prepared(mixed=True)
    middle = advance_bending_contact_dynamics(surface, initial, load, DT).state
    archive = initialize_boundary(surface)
    archive_before = snapshot(archive)
    raw = bending_contact_dynamics_to_dict(surface, middle, load)
    path = tmp_path/"middle.json"
    save_boundary_checkpoint(path, surface, archive, provenance={"bending_dynamics": raw})
    resumed_surface, resumed_archive, metadata = load_boundary_checkpoint(path)
    restored = bending_contact_dynamics_from_dict(metadata["bending_dynamics"], resumed_surface, load)
    assert restored == middle
    direct = advance_bending_contact_dynamics(surface, middle, load, DT)
    resumed = advance_bending_contact_dynamics(resumed_surface, restored, load, DT)
    assert resumed == direct
    assert direct.state.step_count == 2 and direct.state.elapsed_time_s == 2.*DT
    assert all(local.elapsed_time_s == BOUNDS["initial_loading_dt_s"]+2.*DT
               for _, local in direct.state.states)
    direct_path, resumed_path = tmp_path/"direct.json", tmp_path/"resumed.json"
    save_boundary_checkpoint(direct_path, surface, archive, provenance={"bending_dynamics":
        bending_contact_dynamics_to_dict(surface, direct.state, load)})
    save_boundary_checkpoint(resumed_path, resumed_surface, resumed_archive, provenance={"bending_dynamics":
        bending_contact_dynamics_to_dict(resumed_surface, resumed.state, load)})
    assert direct_path.read_bytes() == resumed_path.read_bytes()
    assert snapshot(archive) == snapshot(resumed_archive) == archive_before


def test_balanced_preparation_source_and_bending_ledger_tampering_is_rejected():
    surface, _, load, initial = prepared()
    evolved = advance_bending_contact_dynamics(surface, initial, load, DT).state
    delta = max(abs(evolved.preparation_work_j), abs(evolved.cumulative_source_work_j))*1e-4
    balanced = replace(evolved, preparation_work_j=evolved.preparation_work_j+delta,
                       cumulative_source_work_j=evolved.cumulative_source_work_j-delta)
    with pytest.raises(ValueError, match="source work"):
        validate_bending_contact_dynamics_state(surface, balanced, load)
    with pytest.raises(ValueError, match="preparation work"):
        validate_bending_contact_dynamics_state(surface, replace(evolved,
            preparation_work_j=evolved.preparation_work_j+delta), load)
    with pytest.raises(ValueError, match="work|dissipation"):
        validate_bending_contact_dynamics_state(surface, replace(evolved,
            cumulative_bending_numerical_dissipation_j=evolved.cumulative_bending_numerical_dissipation_j+delta), load)


@pytest.mark.parametrize("field,value", [
    ("elapsed_time_s", True), ("step_count", True), ("initial_loading_dt_s", True),
    ("max_tip_slope", True), ("max_bending_strain", float("nan")),
    ("max_deflection_radius_ratio", -.001),
    ("cumulative_bending_numerical_dissipation_j", -.1),
])
def test_restart_rejects_invalid_clocks_bounds_and_work(field, value):
    surface, _, load, initial = prepared()
    raw = bending_contact_dynamics_to_dict(surface, initial, load)
    raw[field] = value
    with pytest.raises(ValueError):
        bending_contact_dynamics_from_dict(raw, surface, load)


@pytest.mark.parametrize("mutation", ["missing", "extra", "format", "coordinates",
                                       "point", "bank", "tip", "bank_parameter", "bound"])
def test_checkpoint_schema_and_frozen_beam_binding_are_strict(mutation):
    surface, _, load, initial = prepared()
    raw = json.loads(json.dumps(bending_contact_dynamics_to_dict(surface, initial, load)))
    if mutation == "missing":
        raw.pop("bending_source_sha256")
    elif mutation == "extra":
        raw["unstated_flexural_viscosity"] = 1e18
    elif mutation == "format":
        raw["format"] = "geometric-frozen-distributed-contact-dynamics-1"
    elif mutation == "coordinates":
        raw["coordinate_convention"] = "moving_reference_frames"
    elif mutation == "point":
        raw["states"].pop()
    elif mutation == "bank":
        raw["bending_parameters"].pop()
    elif mutation == "tip":
        raw["transverse_displacement_m"][0][1] = True
    elif mutation == "bank_parameter":
        raw["bending_parameters"][0]["sub"]["young_modulus_pa"] *= 2.
    else:
        raw["max_tip_slope"] *= 2.
    with pytest.raises(ValueError):
        bending_contact_dynamics_from_dict(raw, surface, load)


def test_tip_coordinates_cannot_reset_while_evolved_local_histories_are_kept():
    surface, _, load, initial = prepared()
    evolved = advance_bending_contact_dynamics(surface, initial, load, DT).state
    reset = replace(evolved, transverse_displacement_m=initial.transverse_displacement_m)
    with pytest.raises(ValueError, match="anchors|work|bending"):
        validate_bending_contact_dynamics_state(surface, reset, load)


@pytest.mark.parametrize("field,value", [
    ("initial_loading_dt_s", 0.), ("max_rotation_radians", True),
    ("max_tip_slope", 0.), ("max_bending_strain", -.1),
    ("max_deflection_radius_ratio", float("inf")),
])
def test_initialization_requires_explicit_finite_positive_validity_bounds(field, value):
    surface, trace_set, load, state = prepared()
    before = snapshot(state)
    with pytest.raises(ValueError):
        initialize_bending_contact_dynamics(surface, trace_set, load,
            state.bending_parameters, **dict(BOUNDS, **{field: value}))
    assert snapshot(state) == before


def test_small_slope_rejection_preserves_uncommitted_surface_and_initial_history():
    surface, trace_set, load, ordinary = prepared()
    state = initialize_bending_contact_dynamics(surface, trace_set, load,
        ordinary.bending_parameters, **dict(BOUNDS, max_tip_slope=1e-12))
    before = snapshot(surface), snapshot(state)
    with pytest.raises(UnresolvedContactDynamics, match="slope|bound|line search|validity"):
        advance_bending_contact_dynamics(surface, state, load, DT)
    assert (snapshot(surface), snapshot(state)) == before


def test_full_arc_closure_rejects_mixed_solution_that_closes_every_quadrature_node():
    # This smaller declared preload admits every nodal return map, yet the
    # physical trace opens near an endpoint. Preserve it as a rejection case.
    surface, trace_set, load, state = prepared(mixed=True, gap=-1.)
    unconstrained, records = _fixed_branch_solution(surface, load, state, OPENING_DT)
    nodal_gaps = [trace.initial_gap_m+float(normal@unconstrained)
                  for trace, normal, _, _, _ in records]
    assert max(nodal_gaps) < -.01
    for trace, _, shear, stiffness, slipping in records:
        if not slipping:
            # The independent solution really is on the locked branch.
            shear_stress = abs(stiffness*(trace.initial_jump_m+float(shear@unconstrained)))
            wet_cohesion = trace.law.cohesion_pa*(1.-trace.water+trace.water*trace.law.wet_cohesion_fraction)
            assert shear_stress < wet_cohesion
    rotations = unconstrained[:6].reshape((2, 3))
    transverse = tuple((trace.trace_id, radius_sub, radius_over)
        for trace, (radius_sub, radius_over) in zip(trace_set.traces,
            unconstrained[6:].reshape((-1, 2))*surface.radius_km*1000.))
    bounds = continuous_bending_contact_gap_bounds(surface, trace_set, load.plate_ids,
                                                   rotations, transverse)
    assert max(high for _, _, high in bounds) > .01
    before = snapshot(surface), snapshot(state)
    with pytest.raises(UnresolvedContactDynamics, match="opening|[Cc]losed|line search"):
        advance_bending_contact_dynamics(surface, state, load, OPENING_DT)
    assert (snapshot(surface), snapshot(state)) == before


def test_later_opening_rejection_preserves_the_previous_accepted_bending_history():
    surface, _, load, initial = prepared(mixed=True)
    accepted = advance_bending_contact_dynamics(surface, initial, load, OPENING_DT).state
    assert accepted.step_count == 1 and any(local.plastic_slip_m != 0.
                                           for _, local in accepted.states)
    before = snapshot(surface), snapshot(accepted)
    with pytest.raises(UnresolvedContactDynamics, match="opening|[Cc]losed|line search"):
        advance_bending_contact_dynamics(surface, accepted, load, OPENING_DT)
    assert (snapshot(surface), snapshot(accepted)) == before
