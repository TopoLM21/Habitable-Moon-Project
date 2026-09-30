"""Independent mechanical checks for the explicit young-world SI closure."""
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from tectonics.dynamics import DynamicsParameters, angular_velocity_vectors, system_from_omega
from tectonics.genesis import SECONDS_PER_MYR
from tectonics.kinematics import classify_boundaries
from tectonics.lithosphere import initialize_lithosphere
from tectonics.mantle import MantleFlowState
from tectonics.mesh import build_icosphere
from tectonics.plates import Plate, PlateSystem, random_plate_system
from tectonics.young_plate_dynamics import (basal_torque_system, ridge_torques_nm,
                                           update_young_plate_dynamics)
from tectonics.subduction_memory import initialize_subduction_memory
from tectonics.young_boundary import YoungBoundaryState, YoungSlabSegment, SlabThermalCohort


def world(count=4):
    mesh = build_icosphere(2)
    system = random_plate_system(mesh,count,824,0.2,0.1,0.3)
    state = initialize_lithosphere(mesh,system,continental_fraction=0.,continental_nuclei=0)
    state.mantle_lithosphere_thickness_km = np.full(mesh.cell_count,50.)
    state.mantle_lithosphere_density_anomaly_kg_m3 = np.full(mesh.cell_count,40.)
    state.crust_age_myr[:] = 100.
    return mesh,state,system


def flow(field):
    return MantleFlowState(0.,np.asarray(field).copy(),1.)


def update(mesh,state,system,field,*,params=None,dt=1.,normal=4.,inactive=1.,memory=None,strength=None):
    trace = {}
    params = params or DynamicsParameters(force_model="young_si_v1",velocity_relaxation_myr=45.)
    result = update_young_plate_dynamics(mesh,state,system,system,5287.,dt,normal,inactive,
        params,mantle_flow=flow(field),subduction_memory=memory,trace=trace,
        young_slab_strength_pa=(np.full(mesh.cell_count, 1e12) if strength is None else strength))
    return result,trace


def test_full_sphere_drag_has_analytic_si_tensor_and_torque():
    mesh,state,system = world(1)
    omega = np.array([.002,-.003,.001])
    radius,beta = 5287.,1e14
    drag,torque,_ = basal_torque_system(mesh,state.cell_plate,1,radius,
        flow(np.tile(omega,(mesh.cell_count,1))),beta)
    exact_drag = np.eye(3)*(8*np.pi/3)*beta*(radius*1000.)**4
    np.testing.assert_allclose(drag[0],exact_drag,rtol=3e-15,atol=np.linalg.norm(exact_drag)*1e-16)
    np.testing.assert_allclose(torque[0],exact_drag@omega/SECONDS_PER_MYR,rtol=3e-15)


@pytest.mark.parametrize("tangent_encoding",[False,True])
def test_exact_rigid_flow_recovered_without_memory_multiplier_or_speed_cap(tangent_encoding):
    mesh,state,system = world()
    exact = np.array([[.4,-.3,.2],[-.1,.2,.5],[.3,.2,-.2],[.1,-.1,.3]])
    field = exact[state.cell_plate].copy()
    if tangent_encoding:
        field = np.cross(mesh.centroids,np.cross(field,mesh.centroids))
    params = DynamicsParameters(force_model="young_si_v1",velocity_relaxation_myr=.01,
        mantle_memory_fraction=.22,remove_net_rotation=True,max_speed_deg_per_myr=.001)
    result,trace = update(mesh,state,system,field,params=params,dt=1.)
    np.testing.assert_allclose(trace["target_omega"],exact,rtol=3e-14,atol=1e-15)
    np.testing.assert_allclose(angular_velocity_vectors(result[0]),exact,rtol=3e-14,atol=1e-15)
    assert trace["remove_net_rotation"] is False


def test_mixed_flow_matches_overdetermined_velocity_oracle_and_balances_torque():
    mesh,state,system = world()
    x = mesh.centroids
    matrix = np.array([[.001,.002,0.],[.002,-.001,.001],[0.,.001,0.]])
    velocity = x@matrix + np.cross([.003,.001,-.002],x)
    velocity -= x*np.sum(velocity*x,axis=1)[:,None]
    field = np.cross(x,velocity)
    _,trace = update(mesh,state,system,field)
    areas = mesh.physical_cell_areas_km2(5287.)
    for pid in range(4):
        mask = state.cell_plate==pid
        design = np.stack([np.cross(axis,x[mask]) for axis in np.eye(3)],axis=2)
        a = np.sqrt(areas[mask])
        oracle = np.linalg.lstsq((design*a[:,None,None]).reshape(-1,3),
            (velocity[mask]*a[:,None]).ravel(),rcond=None)[0]
        np.testing.assert_allclose(trace["target_omega"][pid],oracle,rtol=2e-13,atol=1e-17)
        driving = np.linalg.norm(trace["basal_driving_torque_nm"][pid])
        assert np.linalg.norm(trace["target_torque_residual_nm"][pid])/driving < 5e-15


def test_temporal_relaxation_and_mechanical_power_budget():
    mesh,state,system = world()
    field = np.cross(mesh.centroids,np.cross([.02,-.01,.015],mesh.centroids))
    _,trace = update(mesh,state,system,field,dt=2.)
    alpha = -np.expm1(-2./45.)
    np.testing.assert_allclose(trace["final_omega"],trace["current_omega"]
        +alpha*(trace["target_omega"]-trace["current_omega"]),atol=1e-17)
    work = np.sum(trace["transient_torque_residual_nm"]*trace["final_omega"])/SECONDS_PER_MYR
    assert trace["basal_drag_dissipation_w"] >= 0.
    assert trace["basal_source_power_w"]-trace["basal_drag_dissipation_w"] == pytest.approx(work,rel=3e-14)
    assert np.linalg.norm(trace["transient_torque_residual_nm"]) > 1e10


def test_common_frame_rotation_changes_absolute_motion_but_preserves_slip():
    mesh,state,system = world()
    field = np.cross(mesh.centroids,np.cross([.02,-.01,.015],mesh.centroids))
    common = np.array([.013,.004,-.009])
    _,before = update(mesh,state,system,field)
    moved_system = system_from_omega(state.cell_plate,system,angular_velocity_vectors(system)+common)
    _,after = update(mesh,state,moved_system,field+common)
    np.testing.assert_allclose(after["target_omega"],before["target_omega"]+common,atol=3e-17)
    np.testing.assert_allclose(after["final_omega"],before["final_omega"]+common,atol=3e-17)
    np.testing.assert_allclose(after["transient_torque_residual_nm"],before["transient_torque_residual_nm"],
        atol=np.linalg.norm(before["basal_driving_torque_nm"])*5e-15)


def test_plate_relabeling_does_not_change_the_physical_solution():
    mesh,state,system = world()
    field = np.cross(mesh.centroids,mesh.centroids@np.diag([.001,-.003,.002]))
    _,before = update(mesh,state,system,field)
    permutation = np.array([2,0,3,1])
    inverse = np.argsort(permutation)
    moved = deepcopy(state)
    moved.cell_plate = permutation[state.cell_plate]
    plates = tuple(Plate(pid,system.plates[old].seed_cell,system.plates[old].euler_axis,
        system.plates[old].angular_speed_rad_per_myr) for pid,old in enumerate(inverse))
    changed = PlateSystem(moved.cell_plate,plates)
    _,after = update(mesh,moved,changed,field)
    np.testing.assert_allclose(after["final_omega"][permutation],before["final_omega"],atol=1e-17)


def test_uniform_thermal_geometry_has_no_phantom_ridge_despite_young_ages():
    mesh,state,system = world()
    edges = classify_boundaries(mesh,system,5287.,0.,0.)
    for edge in edges:
        state.crust_age_myr[edge.face_a] = .1
    torque,length = ridge_torques_nm(mesh,state,edges,5287.,9.81,4)
    np.testing.assert_array_equal(torque,0.)
    assert length==0.


def test_real_thermal_contrast_is_continuous_and_independent_of_display_threshold():
    mesh,state,system = world()
    edges = classify_boundaries(mesh,system,5287.,4.,1.)
    face = edges[0].face_a
    state.crust_age_myr[face] = .1
    state.mantle_lithosphere_thickness_km[face] = 1.
    field = np.zeros((mesh.cell_count,3))
    _,ordinary = update(mesh,state,system,field)
    _,hidden = update(mesh,state,system,field,normal=1e6,inactive=1e6)
    assert np.linalg.norm(ordinary["ridge_torque_nm"]) > 0.
    np.testing.assert_array_equal(ordinary["ridge_torque_nm"],hidden["ridge_torque_nm"])
    np.testing.assert_array_equal(ordinary["target_omega"],hidden["target_omega"])
    # A vanishing density contrast generates a vanishing dimensional force;
    # there is no shared active-boundary normalization or speed floor.
    state.mantle_lithosphere_density_anomaly_kg_m3 *= 1e-6
    _,tiny = update(mesh,state,system,field)
    np.testing.assert_allclose(tiny["ridge_torque_nm"],ordinary["ridge_torque_nm"]*1e-6,rtol=3e-14)


def slab_memory(moment):
    memory = initialize_subduction_memory()
    memory.young_boundary_state = YoungBoundaryState()
    memory.young_boundary_state.segments["accepted-test-parcel"] = YoungSlabSegment(
        key="accepted-test-parcel",contact_key="edge",subducting_plate=0,overriding_plate=1,
        source_anchor=[1.,0.,0.],receiver_anchor=[0.,1.,0.],midpoint=[1.,0.,0.],
        trench_length_km=50.,accepted_area_km2=100.,oceanic_volume_km3=700.,
        cold_mantle_volume_km3=5000.,density_excess_mass_kg=float(np.linalg.norm(moment)),
        buoyancy_moment_kg=list(moment),
        thermal_cohorts=[SlabThermalCohort(acceptance_time_myr=memory.time_myr,
            accepted_area_km2=100.,oceanic_volume_km3=700.,cold_mantle_volume_km3=5000.,
            initial_density_excess_mass_kg=float(np.linalg.norm(moment)),
            initial_buoyancy_moment_kg=list(moment),initial_thickness_km=50.)])
    return memory


def test_accepted_slab_si_units_and_continuity_without_length_normalization():
    mesh,state,system = world()
    field = np.zeros((mesh.cell_count,3))
    moment = np.array([0.,-3e17,4e17])
    params = DynamicsParameters(force_model="young_si_v1",
        young_slab_force_model="full_transmission_upper_bound")
    _,large = update(mesh,state,system,field,memory=slab_memory(moment),params=params)
    _,small = update(mesh,state,system,field,memory=slab_memory(moment*1e-8),params=params)
    _,zero = update(mesh,state,system,field,memory=slab_memory(np.zeros(3)),params=params)
    expected = moment*(5287.*1000.)*9.81
    np.testing.assert_allclose(large["slab_torque_nm"][0],expected,rtol=2e-15)
    np.testing.assert_array_equal(large["slab_torque_nm"][1:],0.)
    np.testing.assert_allclose(small["target_omega"],large["target_omega"]*1e-8,rtol=1e-14,atol=1e-30)
    np.testing.assert_array_equal(zero["target_omega"],0.)


def test_total_mechanical_power_includes_actual_ridge_and_slab():
    mesh,state,system = world()
    edges = classify_boundaries(mesh,system,5287.,4.,1.)
    face = edges[0].face_a
    state.crust_age_myr[face] = .1
    state.mantle_lithosphere_thickness_km[face] = 1.
    field = np.tile([.001,-.003,.005],(mesh.cell_count,1))
    params = DynamicsParameters(force_model="young_si_v1",
        young_slab_force_model="full_transmission_upper_bound")
    _,trace = update(mesh,state,system,field,memory=slab_memory([0.,-3e17,4e17]),params=params)
    assert np.linalg.norm(trace["ridge_torque_nm"]) > 0.
    assert np.linalg.norm(trace["slab_torque_nm"]) > 0.
    assert trace["total_source_power_w"] == pytest.approx(trace["basal_source_power_w"]
        +trace["ridge_power_w"]+trace["slab_power_w"],rel=1e-14)
    assert trace["total_source_power_w"]-trace["basal_drag_dissipation_w"] == pytest.approx(
        trace["transient_net_power_w"],rel=2e-14)


def test_default_preserves_accepted_inventory_but_applies_no_unvalidated_slab_force():
    mesh,state,system = world()
    memory = slab_memory([0.,-3e17,4e17])
    before = asdict(memory.young_boundary_state)
    field = np.tile([.001,-.003,.005],(mesh.cell_count,1))
    _,with_inventory = update(mesh,state,system,field,memory=memory)
    _,without_inventory = update(mesh,state,system,field)
    assert with_inventory["young_slab_force_model"] == "disabled_pending_closure"
    np.testing.assert_array_equal(with_inventory["slab_torque_nm"],0.)
    np.testing.assert_array_equal(with_inventory["target_omega"],without_inventory["target_omega"])
    assert asdict(memory.young_boundary_state) == before


@pytest.mark.parametrize("dt",[0.,-1.,np.nan,np.inf])
def test_invalid_time_step_rejected(dt):
    mesh,state,system = world()
    with pytest.raises(ValueError):
        update(mesh,state,system,np.zeros((mesh.cell_count,3)),dt=dt)


def connected_slab_world():
    from tectonics.young_boundary import (advance_contact_geometry, accept_slab_material,
        SlabAcceptance, contact_key, synchronize_young_zones)
    from tectonics.subduction_memory import SubductionMemoryParameters
    mesh, state, system = world()
    edges = classify_boundaries(mesh, system, 5287., 0., 0.)
    edge = edges[0]
    memory = initialize_subduction_memory()
    memory.young_boundary_state = YoungBoundaryState(connectivity_model="local_edge_transfer_v1",
                                                     mantle_depth_km=2000.)
    advance_contact_geometry(mesh, edges, 5287., 1., memory.young_boundary_state)
    contact = memory.young_boundary_state.contacts[contact_key(edge)]
    area = contact.trench_length_km*400.
    event = SlabAcceptance(edge.face_a, edge.face_b, edge.plate_a, edge.plate_b,
        area, area*7., area*50., area*50.*1e9*40., contact.key,
        contact.trench_length_km, contact.midpoint, contact.torque_direction_ab)
    state.time_myr = 10.
    accept_slab_material(mesh, memory.young_boundary_state, [event], state.time_myr)
    synchronize_young_zones(memory, state, SubductionMemoryParameters())
    params = DynamicsParameters(force_model="young_si_v1", young_slab_force_model="viscous_sinking_v1",
        young_velocity_response_model="quasistatic", young_slab_mantle_viscosity_pa_s=1e21,
        young_slab_mantle_depth_km=2000.)
    return mesh, state, system, memory, params


def test_coupled_sinking_balances_returned_torques_and_all_dissipation():
    mesh, state, system, memory, params = connected_slab_world()
    field = np.tile([.001, -.003, .005], (mesh.cell_count, 1))
    _, trace = update(mesh, state, system, field, params=params, memory=memory)
    rhs = (trace['basal_driving_torque_nm']+trace['ridge_torque_nm']+trace['slab_torque_nm']
        +trace['slab_constraint_reaction_torque_nm'])
    oracle = np.linalg.solve(trace['total_drag_tensor_nm_s'], rhs.ravel()).reshape(-1, 3)
    np.testing.assert_allclose(trace['final_omega']/SECONDS_PER_MYR, oracle, rtol=2e-13, atol=1e-30)
    assert np.linalg.norm(trace['transient_torque_residual_nm'])/np.linalg.norm(rhs) < 2e-14
    assert trace['slab_bending_dissipation_w'] > 0
    assert trace['slab_mantle_dissipation_w'] > 0
    assert trace['total_source_power_w'] == pytest.approx(trace['total_dissipation_w'], rel=2e-14)
    np.testing.assert_allclose(np.sum(trace['slab_torque_nm'], axis=0), 0.,
                              atol=np.linalg.norm(trace['slab_torque_nm'])*1e-15)


def test_sinking_quasistatic_response_is_independent_of_numerical_dt_and_old_filter():
    mesh, state, system, memory, params = connected_slab_world()
    field = np.zeros((mesh.cell_count, 3))
    _, a = update(mesh, state, system, field, params=params, memory=memory, dt=1.)
    _, b = update(mesh, state, system, field,
        params=replace(params, velocity_relaxation_myr=1e12), memory=memory, dt=.5)
    np.testing.assert_array_equal(a['final_omega'], b['final_omega'])
    assert a['alpha'] == b['alpha'] == 1.


def test_sinking_viscous_resistance_remains_after_thermal_buoyancy_vanishes():
    mesh, state, system, memory, params = connected_slab_world()
    for segment in memory.young_boundary_state.segments.values():
        for cohort in segment.thermal_cohorts:
            cohort.initial_density_excess_mass_kg = 0.
            cohort.initial_buoyancy_moment_kg = [0., 0., 0.]
    field = np.tile([.001, -.003, .005], (mesh.cell_count, 1))
    _, trace = update(mesh, state, system, field, params=params, memory=memory)
    np.testing.assert_array_equal(trace['slab_torque_nm'], 0.)
    assert trace['slab_mantle_dissipation_w'] > 0.
    assert trace['total_source_power_w'] == pytest.approx(trace['total_dissipation_w'], rel=2e-14)


def test_weak_neck_fails_without_deleting_material_in_the_pure_force_query():
    mesh, state, system, memory, params = connected_slab_world()
    before = asdict(memory.young_boundary_state)
    field = np.zeros((mesh.cell_count, 3))
    _, trace = update(mesh, state, system, field, params=params, memory=memory,
                      strength=np.zeros(mesh.cell_count))
    assert len(trace['slab_neck_failures']) == 1
    assert trace['slab_neck_failures'][0]['tension_n'] > 0.
    assert trace['slab_neck_failures'][0]['capacity_n'] == 0.
    failed = trace['slab_neck_failures'][0]
    assert failed['tension_n'] == pytest.approx(failed['gravitational_feed_force_n']
        -failed['bending_feed_resistance_n']-failed['mantle_feed_resistance_n']
        +failed['no_eduction_reaction_n'], rel=1e-13)
    assert failed['cold_hinge_thickness_km'] > 0.
    assert failed['trench_length_km'] > 0.
    np.testing.assert_array_equal(trace['slab_torque_nm'], 0.)
    np.testing.assert_array_equal(trace['final_omega'], 0.)
    assert asdict(memory.young_boundary_state) == before


def test_attached_slab_cannot_withdraw_without_a_material_return_law():
    mesh, state, system, memory, params = connected_slab_world()
    field = np.tile([.03, -.04, .02], (mesh.cell_count, 1))
    _, trace = update(mesh, state, system, field, params=params, memory=memory)
    for section in trace['slab_sections']:
        assert section['feed_m_s'] >= -1e-22
        assert section['neck_tension_n'] <= section['neck_capacity_n']*(1.+1e-10)
    assert abs(trace['slab_constraint_power_w']) < max(trace['total_source_power_w'], 1.)*1e-12
