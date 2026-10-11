"""Resolved point strain, continuous closure and fixed-grid restart checks."""
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from analysis.geometric_initiation_validation.admission_case import snapshot
from tests.test_geometric_contact_dynamics import setup as mean_setup
from tectonics.geometric_contact_dynamics import UnresolvedContactDynamics
from tectonics.geometric_distributed_contact_dynamics import (
    _closed_support, advance_distributed_contact_dynamics, continuous_contact_gap_bounds,
    distributed_contact_dynamics_from_dict, distributed_contact_dynamics_to_dict,
    initialize_distributed_contact_dynamics, validate_distributed_contact_dynamics_state,
)
from tectonics.geometric_fault_traces import (
    PhysicalFaultTrace, TraceQuadrature, declare_fault_trace_set, fault_trace_points,
)
from tectonics.geometric_polarity import PolarityEvidence


DT = 1e10


def setup(*, panel_count=2, gauss_order=4, gap=-20., friction=.2, cohesion=2e6):
    surface, contacts, experiments, _, load, _ = mean_setup(gap=gap, friction=friction, cohesion=cohesion)
    traces = []
    for contact, experiment in zip(contacts, experiments):
        h = np.asarray(contact.normal_a_to_b)*(1. if experiment.fault.overriding_plate == contact.plate_b else -1.)
        start, end = contact.start, contact.end
        if np.dot(np.cross(start, end), h) < 0.:
            start, end = end, start
        traces.append(PhysicalFaultTrace(experiment.fault.fault_id, start, end,
            experiment.fault.subducting_plate, experiment.fault.overriding_plate,
            experiment.fault.dip_radians, experiment.columns, experiment.law,
            experiment.contact_height_m, experiment.water, experiment.pore_pressure_pa,
            gap, 0., "synthetic physical arc, independent of material segmentation"))
    trace_set = declare_fault_trace_set(surface, traces, TraceQuadrature(panel_count, gauss_order))
    initial = initialize_distributed_contact_dynamics(surface, trace_set, load,
        initial_loading_dt_s=DT, max_rotation_radians=.01)
    return surface, contacts, trace_set, load, initial


def test_distributed_locked_solution_matches_independent_point_stiffness_matrix():
    surface, _, trace_set, load, initial = setup()
    points = fault_trace_points(surface, trace_set)
    traces = {trace.trace_id: trace for trace in trace_set.traces}
    matrix = np.asarray(load.drag_nm_s)/DT
    preload = np.zeros(6)
    old = dict(initial.states)
    for point in points:
        trace = traces[point.trace_id]
        lever = surface.radius_km*1000.*np.cross(point.position, point.horizontal)
        row = np.zeros((2, 3))
        row[trace.subducting_plate], row[trace.overriding_plate] = -lever, lever
        row = row.ravel()
        sine, cosine = math.sin(trace.dip_radians), math.cos(trace.dip_radians)
        kn = 1./(1./trace.columns.normal_stiffness_pa_m+1./trace.law.normal_stiffness_pa_m)
        kt = 1./(1./trace.columns.shear_stiffness_pa_m+1./trace.law.tangential_stiffness_pa_m)
        matrix += point.area_m2*(kn*sine*sine+kt*cosine*cosine)*np.outer(row, row)
        history = old[point.point_id]
        preload += point.area_m2*(sine*history.traction_normal_pa+cosine*history.traction_shear_pa)*row
    expected = np.linalg.solve(matrix, np.asarray(load.torque_nm)-preload)
    step = advance_distributed_contact_dynamics(surface, initial, load, DT)
    np.testing.assert_allclose(np.asarray(step.state.rotation_radians).ravel(), expected, rtol=2e-10, atol=2e-18)
    assert all(point.step.slip_increment_m == 0. for point in step.points)
    assert step.torque_signal_relative_residual < 2e-10
    assert step.incremental_energy_balance_relative_error < 2e-9
    jumps = [point.step.state.jump_m for point in step.points if point.trace_id == points[0].trace_id]
    assert max(jumps)-min(jumps) > 1e-7


def test_point_sliding_work_and_physical_reactions_reuse_global_engine():
    surface, _, _, load, initial = setup(friction=0., cohesion=0.)
    before = snapshot(surface), snapshot(initial)
    step = advance_distributed_contact_dynamics(surface, initial, load, DT)
    assert any(point.step.slip_increment_m != 0. for point in step.points)
    assert step.viscous_work_j > 0.
    assert step.friction_work_j == 0.
    for point in step.points:
        np.testing.assert_array_equal(point.force_on_sub_n, -np.asarray(point.force_on_over_n))
        np.testing.assert_array_equal(point.torque_on_sub_nm, -np.asarray(point.torque_on_over_nm))
        expected = np.cross(surface.radius_km*1000.*np.asarray(point.position), point.force_on_sub_n)
        np.testing.assert_array_equal(point.torque_on_sub_nm, expected)
    combined = math.fsum((step.basal_dissipation_j, step.stored_energy_increment_j,
        step.friction_work_j, step.viscous_work_j, step.numerical_dissipation_j))
    assert abs(step.source_work_j-combined) <= 5e-10*max(abs(step.source_work_j), abs(combined))+step.energy_rounding_tolerance_j
    assert (snapshot(surface), snapshot(initial)) == before


def test_exact_restart_and_no_silent_changed_quadrature_or_mean_history_migration():
    surface, _, trace_set, load, initial = setup()
    first = advance_distributed_contact_dynamics(surface, initial, load, DT).state
    raw = json.loads(json.dumps(distributed_contact_dynamics_to_dict(surface, first, load)))
    loaded = distributed_contact_dynamics_from_dict(raw, surface, load)
    assert loaded == first
    assert advance_distributed_contact_dynamics(surface, loaded, load, DT) == advance_distributed_contact_dynamics(surface, first, load, DT)
    changed = declare_fault_trace_set(surface, trace_set.traces, TraceQuadrature(1, 8))
    with pytest.raises(ValueError, match="point histories"):
        validate_distributed_contact_dynamics_state(surface, replace(first, fault_traces=changed), load)
    _, _, _, _, _, mean_initial = mean_setup()
    with pytest.raises(ValueError, match="arc-mean"):
        validate_distributed_contact_dynamics_state(surface, mean_initial, load)
    raw["states"].pop()
    with pytest.raises(ValueError, match="point histories"):
        distributed_contact_dynamics_from_dict(raw, surface, load)


def test_continuous_support_detects_interior_opening_missed_by_all_nodes():
    surface, _, trace_set, _, _ = setup(panel_count=1, gauss_order=2, gap=-1.)
    trace = trace_set.traces[0]
    midpoint = np.asarray(trace.start)+np.asarray(trace.end)
    midpoint /= np.linalg.norm(midpoint)
    delta = 1.01*np.cross(midpoint, trace.normal_sub_to_over)/(surface.radius_km*1000.*math.sin(trace.dip_radians))
    q = np.zeros((2, 3))
    q[trace.overriding_plate] = delta
    points = fault_trace_points(surface, trace_set)
    traces = {item.trace_id: item for item in trace_set.traces}
    sampled_gaps = []
    for point in points:
        item = traces[point.trace_id]
        relative = q[item.overriding_plate]-q[item.subducting_plate]
        sampled_gaps.append(item.initial_gap_m+math.sin(item.dip_radians)*surface.radius_km*1000.*np.cross(point.position, point.horizontal)@relative)
    assert max(sampled_gaps) < 0.
    assert max(maximum for _, _, maximum in continuous_contact_gap_bounds(surface, trace_set, (0, 1), q)) > 0.
    with pytest.raises(UnresolvedContactDynamics, match="Continuous"):
        _closed_support(surface, trace_set, (0, 1), q.ravel())


def test_physical_history_scope_keeps_original_evidence_through_checkpoint():
    surface, contacts, trace_set, load, _ = setup()
    origin = PolarityEvidence("original-material", contacts[0].contact_id, 1, 0,
                              "historical explicit material evidence")
    prepared = initialize_distributed_contact_dynamics(surface, trace_set, load,
        initial_loading_dt_s=DT, max_rotation_radians=.01, history=(origin,))
    assert prepared.polarity_history[0].origin == origin
    raw = distributed_contact_dynamics_to_dict(surface, prepared, load)
    loaded = distributed_contact_dynamics_from_dict(raw, surface, load)
    assert loaded.polarity_history == prepared.polarity_history
    raw["polarity_history"][0]["origin"]["subducting_plate"], raw["polarity_history"][0]["origin"]["overriding_plate"] = 0, 1
    with pytest.raises(ValueError, match="direction|owners|orientation|oppos"):
        distributed_contact_dynamics_from_dict(raw, surface, load)
