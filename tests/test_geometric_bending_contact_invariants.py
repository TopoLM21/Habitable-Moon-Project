"""Independent balance, symmetry and elastic-limit checks of coupled bending.

The matrix oracle below derives the two-bank cantilever and contact energies
directly from their physical parameters. It does not use solver network rows,
bending assembly or its quadrature-point constructors.
"""
from dataclasses import asdict, replace
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import tectonics.fractional_dynamics as fractional_dynamics
from tectonics.geometric_basal_loading import assemble_geometric_basal_load
from tectonics.geometric_bending import (
    ElasticBendingBankParameters, FaultBendingParameters,
)
from tectonics.geometric_bending_contact_dynamics import (
    advance_bending_contact_dynamics, initialize_bending_contact_dynamics,
)
from tectonics.geometric_contact_dynamics import UnresolvedContactDynamics
from tectonics.geometric_distributed_contact_dynamics import (
    advance_distributed_contact_dynamics, initialize_distributed_contact_dynamics,
)
from tectonics.geometric_fault_traces import TraceQuadrature, declare_fault_trace_set
from tests.test_geometric_distributed_contact_invariants import (
    DT, PARAMETERS, PROVENANCE, traces, world,
)


def _declared():
    # The finite viscosity certifies the full two-dimensional contact branch
    # family; the large cohesion makes the solutions below purely elastic.
    return tuple(replace(trace, law=replace(trace.law, friction_dry=0., friction_wet=0.,
        viscosity_pa_s_m=1e16)) for trace in traces())


def _banks(declared, *, modulus_multiplier=1.):
    return tuple(FaultBendingParameters(trace.trace_id,
        ElasticBendingBankParameters(20000.+1500.*index, 2000.,
            modulus_multiplier*1e10, .25, "Declared sub-bank clamped cantilever"),
        ElasticBendingBankParameters(24000.+1000.*index, 1800.,
            modulus_multiplier*1.3e10, .22, "Declared over-bank clamped cantilever"))
        for index, trace in enumerate(declared))


def _prepare(mesh, surface, lid, *, declared=None, banks=None,
             source_parameters=PARAMETERS, **bounds):
    declared = _declared() if declared is None else declared
    banks = _banks(declared) if banks is None else banks
    load = assemble_geometric_basal_load(mesh, surface, source_parameters, lid, PROVENANCE)
    trace_set = declare_fault_trace_set(surface, declared, TraceQuadrature(1, 6))
    limits = dict(max_rotation_radians=.01, max_tip_slope=.01,
                  max_bending_strain=.01, max_deflection_radius_ratio=.001)
    limits.update(bounds)
    initial = initialize_bending_contact_dynamics(surface, trace_set, load, banks,
        initial_loading_dt_s=1e6, **limits)
    return load, trace_set, initial


def _independent_locked_problem(surface, trace_set, load, banks):
    """Direct Galerkin energy Hessian, in well-scaled [q, w/R] coordinates."""
    radius = surface.radius_km*1000.
    ids = load.plate_ids
    q_size = 3*len(ids)
    dimension = q_size+2*len(trace_set.traces)
    stiffness = np.zeros((dimension, dimension))
    stiffness[:q_size, :q_size] = np.asarray(load.drag_nm_s)/DT
    source = np.r_[np.asarray(load.torque_nm), np.zeros(dimension-q_size)]
    preload = np.zeros(dimension)
    nodes, weights = np.polynomial.legendre.leggauss(trace_set.quadrature.gauss_order)
    bank_by_trace = {item.trace_id: item for item in banks}
    springs, records = [], []
    for index, trace in enumerate(trace_set.traces):
        start, end = np.asarray(trace.start), np.asarray(trace.end)
        extent = math.atan2(np.linalg.norm(np.cross(start, end)), float(start@end))
        horizontal = np.cross(start, end)/np.linalg.norm(np.cross(start, end))
        tangent = np.cross(horizontal, start)
        length = radius*extent
        bank_pair = bank_by_trace[trace.trace_id]
        for side, bank in enumerate((bank_pair.sub, bank_pair.over)):
            rigidity = bank.young_modulus_pa*bank.elastic_thickness_m**3/(12.*(1.-bank.poisson_ratio**2))
            spring = 3.*rigidity*length/bank.width_m**3
            position = q_size+2*index+side
            stiffness[position, position] += spring*radius**2
            springs.append(spring)
        kn = 1./(trace.columns.sub_length_m/trace.columns.sub_normal_modulus_pa
                 +trace.columns.over_length_m/trace.columns.over_normal_modulus_pa
                 +1./trace.law.normal_stiffness_pa_m)
        kt = 1./(trace.columns.sub_length_m/trace.columns.sub_shear_modulus_pa
                 +trace.columns.over_length_m/trace.columns.over_shear_modulus_pa
                 +1./trace.law.tangential_stiffness_pa_m)
        sine, cosine = math.sin(trace.dip_radians), math.cos(trace.dip_radians)
        for panel in range(trace_set.quadrature.panel_count):
            panel_width = extent/trace_set.quadrature.panel_count
            for node, weight in zip(nodes, weights):
                angle = panel_width*(panel+.5+.5*node)
                radial = math.cos(angle)*start+math.sin(angle)*tangent
                area = radius*.5*panel_width*weight*trace.contact_height_m/sine
                lever = radius*np.cross(radial, horizontal)
                horizontal_row = np.zeros(dimension)
                a, b = ids.index(trace.subducting_plate), ids.index(trace.overriding_plate)
                horizontal_row[3*a:3*a+3], horizontal_row[3*b:3*b+3] = -lever, lever
                vertical_row = np.zeros(dimension)
                vertical_row[q_size+2*index:q_size+2*index+2] = (-radius, radius)
                normal_row = sine*horizontal_row+cosine*vertical_row
                shear_row = cosine*horizontal_row-sine*vertical_row
                stiffness += area*(kn*np.outer(normal_row, normal_row)+kt*np.outer(shear_row, shear_row))
                preload += area*(kn*trace.initial_gap_m*normal_row+kt*trace.initial_jump_m*shear_row)
                records.append((trace, radial, area, kn, kt, normal_row, shear_row))
    coordinates = np.linalg.solve(stiffness, source-preload)
    return coordinates, np.asarray(springs), tuple(records)


def _coordinates(state, radius):
    return np.r_[np.asarray(state.rotation_radians).ravel(),
        np.asarray([item[1:] for item in state.transverse_displacement_m]).ravel()/radius]


def _elastic_energy(state):
    return math.fsum(local.bulk_energy_j+local.contact_energy_j for _, local in state.states)


def test_locked_coupled_solution_matches_independent_cantilever_matrix_and_virtual_work():
    mesh, surface, lid = world()
    banks = _banks(_declared())
    load, trace_set, initial = _prepare(mesh, surface, lid, banks=banks)
    expected, springs, records = _independent_locked_problem(surface, trace_set, load, banks)
    step = advance_bending_contact_dynamics(surface, initial, load, DT)
    radius = surface.radius_km*1000.
    q_size = 3*len(load.plate_ids)
    actual = _coordinates(step.state, radius)
    np.testing.assert_allclose(actual, expected, rtol=3e-9, atol=3e-18)
    assert all(point.step.slip_increment_m == 0. for point in step.points)
    assert np.linalg.norm(actual[q_size:]) > 1e-7
    expected_bending = .5*np.dot(springs, (expected[q_size:]*radius)**2)
    assert step.bending_energy_j == pytest.approx(expected_bending, rel=3e-9)
    assert step.bending_stored_energy_increment_j == pytest.approx(expected_bending, rel=3e-9)
    assert step.bending_numerical_dissipation_j == pytest.approx(expected_bending, rel=3e-9)
    contact_increment, contact_remainder, anchor_work = 0., 0., 0.
    for trace, _, area, kn, kt, normal_row, shear_row in records:
        dg, dj = float(normal_row@expected), float(shear_row@expected)
        gap, jump = trace.initial_gap_m+dg, trace.initial_jump_m+dj
        contact_increment += .5*area*(kn*(gap**2-trace.initial_gap_m**2)
                                      +kt*(jump**2-trace.initial_jump_m**2))
        contact_remainder += .5*area*(kn*dg**2+kt*dj**2)
        anchor_work += area*(kn*gap*dg+kt*jump*dj)
    assert _elastic_energy(step.state)-_elastic_energy(initial) == pytest.approx(contact_increment, rel=3e-9)
    assert step.contact_work_j == pytest.approx(anchor_work, rel=3e-9)
    source_work = float(np.asarray(load.torque_nm)@expected[:q_size])
    drag_work = float(expected[:q_size]@np.asarray(load.drag_nm_s)@expected[:q_size]/DT)
    accounted = math.fsum((contact_increment, contact_remainder, 2.*expected_bending, drag_work))
    assert source_work == pytest.approx(accounted, rel=3e-8, abs=1e-4)
    assert step.source_work_j == pytest.approx(source_work, rel=3e-9)
    # Physical force and torque resultants reproduce the anchor work, including
    # the transverse term which Euler torque alone cannot represent.
    tips = {identity: np.asarray((sub, over)) for identity, sub, over in step.state.transverse_displacement_m}
    trace_by_id = {item.trace_id: item for item in trace_set.traces}
    reaction_work = 0.
    for point in step.points:
        trace = trace_by_id[point.trace_id]
        sub, over = load.plate_ids.index(trace.subducting_plate), load.plate_ids.index(trace.overriding_plate)
        radial = np.asarray(point.position)
        sub_tip, over_tip = tips[point.trace_id]
        q = np.asarray(step.state.rotation_radians)
        u_sub = radius*np.cross(q[sub], radial)+sub_tip*radial
        u_over = radius*np.cross(q[over], radial)+over_tip*radial
        reaction_work += np.dot(point.force_on_sub_n, u_sub)+np.dot(point.force_on_over_n, u_over)
    assert reaction_work == pytest.approx(-anchor_work, rel=3e-9)
    assert step.incremental_energy_balance_relative_error < 2e-9


def test_source_free_preload_relaxes_into_bending_without_inventing_external_work():
    mesh, surface, lid = world()
    load, _, initial = _prepare(mesh, surface, lid,
        source_parameters=replace(PARAMETERS, convective_traction_pa=0.))
    assert np.linalg.norm(load.torque_nm) == 0.
    step = advance_bending_contact_dynamics(surface, initial, load, DT)
    assert initial.preparation_work_j > 0.
    assert step.source_work_j == 0.
    assert step.state.cumulative_source_work_j == 0.
    assert step.state.preparation_work_j == initial.preparation_work_j
    assert step.bending_energy_j > 0.
    assert _elastic_energy(step.state) < _elastic_energy(initial)
    release = _elastic_energy(initial)-_elastic_energy(step.state)
    spent = math.fsum((step.bending_energy_j, step.basal_dissipation_j,
                      step.numerical_dissipation_j, step.friction_work_j, step.viscous_work_j))
    assert release == pytest.approx(spent, rel=2e-10)
    final_accounted = math.fsum((_elastic_energy(step.state), step.bending_energy_j,
        step.state.cumulative_basal_dissipation_j,
        step.state.cumulative_bending_numerical_dissipation_j,
        *(local.cumulative_numerical_dissipation_j+local.cumulative_friction_work_j
          +local.cumulative_viscous_work_j for _, local in step.state.states)))
    assert initial.preparation_work_j == pytest.approx(final_accounted, rel=2e-10)


def test_bending_solution_is_covariant_under_passive_rotation_of_geometry_and_source(monkeypatch):
    mesh, surface, lid = world()
    declared = _declared()
    load, _, initial = _prepare(mesh, surface, lid, declared=declared)
    step = advance_bending_contact_dynamics(surface, initial, load, DT)
    rotation = Rotation.from_rotvec([.31, -.17, .24]).as_matrix()
    rotated_mesh = replace(mesh, vertices=mesh.vertices@rotation.T, centroids=mesh.centroids@rotation.T)
    rotated_surface = replace(surface, fragments=tuple(replace(item,
        polygon=tuple(map(tuple, np.asarray(item.polygon)@rotation.T))) for item in surface.fragments))
    rotated_traces = tuple(replace(item, start=tuple(rotation@item.start), end=tuple(rotation@item.end))
                           for item in declared)
    source = fractional_dynamics.prescribed_vertex_traction

    def rotated_source(current_mesh, **kwargs):
        assert current_mesh is rotated_mesh
        return source(mesh, **kwargs)@rotation.T

    monkeypatch.setattr(fractional_dynamics, "prescribed_vertex_traction", rotated_source)
    changed_load, _, changed_initial = _prepare(rotated_mesh, rotated_surface, lid, declared=rotated_traces)
    changed = advance_bending_contact_dynamics(rotated_surface, changed_initial, changed_load, DT)
    np.testing.assert_allclose(changed.angular_velocity_rad_s,
        np.asarray(step.angular_velocity_rad_s)@rotation.T, rtol=5e-9, atol=1e-27)
    np.testing.assert_allclose([item[1:] for item in changed.state.transverse_displacement_m],
        [item[1:] for item in step.state.transverse_displacement_m], rtol=5e-9, atol=1e-12)
    for original, rotated in zip(step.points, changed.points):
        assert rotated.step.traction_pa == pytest.approx(original.step.traction_pa, rel=5e-9, abs=1e-5)
        np.testing.assert_allclose(rotated.force_on_sub_n, rotation@original.force_on_sub_n, rtol=5e-9, atol=.1)
    assert changed.source_work_j == pytest.approx(step.source_work_j, rel=5e-9)
    assert changed.bending_energy_j == pytest.approx(step.bending_energy_j, rel=5e-9)


def test_sparse_owner_relabel_preserves_tip_deflections_and_point_history():
    mesh, surface, lid = world()
    declared = _declared()
    load, _, initial = _prepare(mesh, surface, lid, declared=declared)
    step = advance_bending_contact_dynamics(surface, initial, load, DT)
    labels = {0: 10**12, 1: 7}
    changed_surface = replace(surface, fragments=tuple(replace(item,
        parcel=replace(item.parcel, plate=labels[item.parcel.plate])) for item in reversed(surface.fragments)))
    changed_traces = tuple(replace(item, subducting_plate=labels[item.subducting_plate],
                                  overriding_plate=labels[item.overriding_plate]) for item in declared)
    changed_load, _, changed_initial = _prepare(mesh, changed_surface, lid, declared=changed_traces)
    changed = advance_bending_contact_dynamics(changed_surface, changed_initial, changed_load, DT)
    assert changed_load.plate_ids == (7, 10**12)
    for index, owner in enumerate(load.plate_ids):
        target = changed_load.plate_ids.index(labels[owner])
        np.testing.assert_allclose(changed.angular_velocity_rad_s[target], step.angular_velocity_rad_s[index],
                                   rtol=5e-9, atol=1e-27)
    assert [item[0] for item in changed.state.transverse_displacement_m] == [item[0] for item in step.state.transverse_displacement_m]
    np.testing.assert_allclose([item[1:] for item in changed.state.transverse_displacement_m],
        [item[1:] for item in step.state.transverse_displacement_m], rtol=5e-9, atol=1e-12)
    for original, changed_point in zip(step.points, changed.points):
        assert changed_point.step.traction_pa == pytest.approx(original.step.traction_pa, rel=5e-9, abs=1e-5)
    assert changed.bending_energy_j == pytest.approx(step.bending_energy_j, rel=5e-9)


def test_infinite_bending_stiffness_limit_approaches_frozen_no_bending_contact_solution():
    mesh, surface, lid = world()
    declared = _declared()
    load, trace_set, _ = _prepare(mesh, surface, lid, declared=declared)
    frozen_initial = initialize_distributed_contact_dynamics(surface, trace_set, load,
        initial_loading_dt_s=1e6, max_rotation_radians=.01)
    frozen = advance_distributed_contact_dynamics(surface, frozen_initial, load, DT)
    frozen_q = np.asarray(frozen.state.rotation_radians)
    errors, deflections = [], []
    for multiplier in (1e3, 1e5, 1e7):
        _, _, initial = _prepare(mesh, surface, lid, declared=declared,
                                 banks=_banks(declared, modulus_multiplier=multiplier))
        step = advance_bending_contact_dynamics(surface, initial, load, DT)
        errors.append(np.linalg.norm(np.asarray(step.state.rotation_radians)-frozen_q)/np.linalg.norm(frozen_q))
        deflections.append(np.linalg.norm([item[1:] for item in step.state.transverse_displacement_m]))
    assert errors[0] > errors[1] > errors[2]
    assert deflections[0] > deflections[1] > deflections[2]
    assert errors[-1] < 2e-4


@pytest.mark.parametrize("bound", ["max_tip_slope", "max_bending_strain", "max_deflection_radius_ratio"])
def test_rejected_small_deformation_bound_keeps_every_accepted_history_unchanged(bound):
    mesh, surface, lid = world()
    load, _, initial = _prepare(mesh, surface, lid, **{bound: 1e-12})
    before = asdict(initial), asdict(surface)
    with pytest.raises(UnresolvedContactDynamics, match="slope|strain|deflection|[Bb]ending|[Tt]ransverse|line search"):
        advance_bending_contact_dynamics(surface, initial, load, DT)
    assert (asdict(initial), asdict(surface)) == before


def test_elastic_oracle_predicts_opening_and_coupled_step_rejects_it_atomically():
    mesh, surface, lid = world()
    declared = _declared()
    banks = _banks(declared)
    load, trace_set, initial = _prepare(mesh, surface, lid, declared=declared, banks=banks,
        source_parameters=replace(PARAMETERS, convective_traction_pa=120000.))
    unconstrained, _, records = _independent_locked_problem(surface, trace_set, load, banks)
    # No other elastic branch is available: this independently assembled
    # locked solution requires a positive free gap on real physical support.
    gaps = [trace.initial_gap_m+float(normal@unconstrained)
            for trace, _, _, _, _, normal, _ in records]
    assert max(gaps) > .01
    assert np.max(np.abs(unconstrained[:6])) < initial.max_rotation_radians
    assert np.max(np.abs(unconstrained[6:])) < initial.max_deflection_radius_ratio
    before = asdict(initial), asdict(surface)
    with pytest.raises(UnresolvedContactDynamics, match="opening|[Cc]losed|line search"):
        advance_bending_contact_dynamics(surface, initial, load, DT)
    assert (asdict(initial), asdict(surface)) == before
