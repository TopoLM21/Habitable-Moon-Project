from copy import deepcopy
from dataclasses import replace
import itertools

import numpy as np
import pytest

from tectonics.breakoff import SlabBreakoffParameters, advance_slab_breakoff
from tectonics.kinematics import classify_boundaries
from tectonics.lithosphere import initialize_lithosphere, advance_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system, PlateSystem
from tectonics.rollback import RollbackParameters, advance_rollback
from tectonics.subduction_memory import (SubductionMemoryParameters, initialize_subduction_memory,
    advance_subduction_memory, memory_to_json, memory_from_json, remap_subduction_memory)
from tectonics.tides import constant_eccentricity
from tectonics.transport import TransportMap, TransportDiagnostics, initialize_transport_state
from tectonics.young_boundary import (YoungBoundaryState,SlabAcceptance,advance_contact_geometry,
    accept_slab_material,accepted_slab_torques_nm,synchronize_young_zones,
    boundary_state_to_json,boundary_state_from_json,young_ocean_overlap_order)
from tectonics.young_boundary import (slab_thermal_deficit_fraction,limit_attached_inventory,
    accepted_slab_inventory_diagnostics,remesh_young_inventory,contact_key,
    accepted_slab_force_sections,commit_slab_neck_failures)


def world():
    mesh = build_icosphere(1)
    system = random_plate_system(mesh,4,8722,.002,.001,.003)
    state = initialize_lithosphere(mesh,system,continental_fraction=0.,continental_nuclei=0)
    area = mesh.physical_cell_areas_km2(1000.)
    state.crust_age_myr[:] = 50.
    state.oceanic_volume_km3 = area*7.
    state.mantle_lithosphere_thickness_km = np.full(mesh.cell_count,30.)
    state.mantle_lithosphere_density_anomaly_kg_m3 = np.full(mesh.cell_count,60.)
    return mesh,state,system


def event(mesh,**kwargs):
    values = dict(source_face=0,target_face=1,subducting_plate=0,overriding_plate=1,
        accepted_area_km2=10.,oceanic_volume_km3=70.,cold_mantle_volume_km3=300.,
        density_excess_mass_kg=1.8e13,contact_key='edge-a',trench_length_km=10.,
        midpoint=[1.,0.,0.],torque_direction=[0.,0.,1.])
    values.update(kwargs)
    return SlabAcceptance(**values)


def memory():
    result = initialize_subduction_memory()
    result.young_boundary_state = YoungBoundaryState()
    return result


def test_weak_signed_geometry_is_display_independent_but_not_phantom_material():
    mesh,state,system = world()
    slow = classify_boundaries(mesh,system,1000.,4.,1.)
    physical = classify_boundaries(mesh,system,1000.,0.,0.)
    params = SubductionMemoryParameters(model='accepted_material_v1')
    a,b = initialize_subduction_memory(),initialize_subduction_memory()
    advance_subduction_memory(mesh,state,slow,1000.,400.,a,params)
    advance_subduction_memory(mesh,state,physical,1000.,400.,b,params)
    assert memory_to_json(a) == memory_to_json(b)
    assert sum(c.cumulative_closure_area_km2 for c in a.young_boundary_state.contacts.values()) > 0.
    assert sum(c.cumulative_opening_area_km2 for c in a.young_boundary_state.contacts.values()) > 0.
    assert not a.zones
    assert a.cumulative_subducted_area_km2 == 0.
    assert not np.any(accepted_slab_torques_nm(a,4,1000.,5.))


def test_accepted_inventory_has_si_units_and_no_length_cap_amplitude():
    mesh,state,_ = world()
    m = memory()
    accept_slab_material(mesh,m.young_boundary_state,[event(mesh)],1.)
    np.testing.assert_allclose(accepted_slab_torques_nm(m,4,1000.,5.)[0],
                               [0.,0.,1.8e13*1e6*5.],rtol=1e-15)
    state.time_myr = 1.
    synchronize_young_zones(m,state,SubductionMemoryParameters(slab_length_cap_km=.1))
    assert m.zones[(0,1)].slab_length_km == 1.
    before = accepted_slab_torques_nm(m,4,1000.,5.)
    synchronize_young_zones(m,state,SubductionMemoryParameters(slab_length_cap_km=1800.))
    np.testing.assert_array_equal(before,accepted_slab_torques_nm(m,4,1000.,5.))


@pytest.mark.parametrize('scale',[0.,1e-12,1e-6,1.])
def test_no_floor_or_denominator_jump_as_cold_buoyancy_vanishes(scale):
    mesh,_,_ = world()
    m = memory()
    accept_slab_material(mesh,m.young_boundary_state,[event(mesh,
        cold_mantle_volume_km3=300.*scale,density_excess_mass_kg=1.8e13*scale)],1.)
    np.testing.assert_allclose(accepted_slab_torques_nm(m,4,1000.,5.)[0],
                               [0.,0.,1.8e13*1e6*5.*scale],rtol=1e-15)


def test_opposite_segments_cancel_without_unit_axis_renormalization():
    mesh,state,_ = world()
    m = memory()
    events=[event(mesh),event(mesh,source_face=2,contact_key='edge-b',torque_direction=[0.,0.,-1.])]
    accept_slab_material(mesh,m.young_boundary_state,events,1.)
    synchronize_young_zones(m,state,SubductionMemoryParameters())
    assert not np.any(accepted_slab_torques_nm(m,4,1000.,5.))
    assert not np.any(m.zones[(0,1)].torque_axis)
    assert m.young_boundary_state.cumulative_accepted_oceanic_volume_km3 == 140.


def test_duplicate_transaction_rejected_before_any_ledger_mutation():
    mesh,_,_ = world()
    inv=YoungBoundaryState()
    accept_slab_material(mesh,inv,[event(mesh)],1.)
    before=boundary_state_to_json(inv)
    with pytest.raises(ValueError,match='duplicate'):
        accept_slab_material(mesh,inv,[event(mesh)],1.)
    assert before == boundary_state_to_json(inv)
    with pytest.raises(ValueError,match='Duplicate slab source'):
        accept_slab_material(mesh,inv,[event(mesh),event(mesh)],2.)
    assert before == boundary_state_to_json(inv)


def test_relabeling_plate_ids_preserves_polarity_and_physical_torque():
    mesh,state,_=world()
    original_owner=state.cell_plate.copy()
    source=np.array([0,1,2,3])
    reference_order=young_ocean_overlap_order(mesh,state,source,0)
    reference=None
    for ids in itertools.permutations(range(4)):
        ids=np.asarray(ids)
        state.cell_plate=ids[original_owner]
        np.testing.assert_array_equal(reference_order,young_ocean_overlap_order(mesh,state,source,0))
        m=memory()
        accept_slab_material(mesh,m.young_boundary_state,[event(mesh,
            subducting_plate=int(ids[0]),overriding_plate=int(ids[1]))],1.)
        force=accepted_slab_torques_nm(m,4,1000.,5.)[ids]
        if reference is None:
            reference=force
        else:
            np.testing.assert_array_equal(reference,force)


def test_checkpoint_and_resume_preserve_every_accepted_inventory_field():
    mesh,state,system=world()
    a=memory()
    advance_contact_geometry(mesh,classify_boundaries(mesh,system,1000.,4.,1.),1000.,1.,a.young_boundary_state)
    accept_slab_material(mesh,a.young_boundary_state,[event(mesh)],1.)
    synchronize_young_zones(a,state,SubductionMemoryParameters())
    saved=memory_to_json(a)
    b=memory_from_json(saved)
    assert memory_to_json(b) == saved
    next_events=[event(mesh,source_face=2,contact_key='second')]
    for obj in (a,b):
        accept_slab_material(mesh,obj.young_boundary_state,next_events,2.)
    assert memory_to_json(a) == memory_to_json(b)
    np.testing.assert_array_equal(accepted_slab_torques_nm(a,4,1000.,5.),accepted_slab_torques_nm(b,4,1000.,5.))
    with pytest.raises(ValueError,match='model version'):
        boundary_state_from_json(dict(version='unsupported'))
    assert 'young_boundary_state' not in memory_to_json(initialize_subduction_memory())


def test_legacy_breakoff_and_rollback_cannot_mutate_new_si_inventory():
    mesh,state,_=world()
    m=memory()
    accept_slab_material(mesh,m.young_boundary_state,[event(mesh)],1.)
    synchronize_young_zones(m,state,SubductionMemoryParameters())
    zone=m.zones[(0,1)]
    zone.slab_length_km=1800.;zone.slab_depth_km=1100.;zone.active_age_myr=1000.
    zone.breakoff_damage=1.2;zone.active=False
    before=boundary_state_to_json(m.young_boundary_state)
    advance_slab_breakoff(mesh,state,m,1000.,1.,SlabBreakoffParameters())
    omega,forcing,_=advance_rollback(mesh,state,m,1000.,1.,RollbackParameters())
    assert not zone.broken_off
    assert not omega.any() and not forcing.any()
    assert before == boundary_state_to_json(m.young_boundary_state)


def _map_with_one_collision(mesh,state,system,source_face,target_face,transport):
    count=len(system.plates)
    covered=np.zeros((count,mesh.cell_count),dtype=bool)
    sources=np.full((count,mesh.cell_count),-1,dtype=int)
    destinations=[]
    for pid in range(count):
        source=np.flatnonzero(state.cell_plate==pid)
        target=source.copy()
        target[source==source_face]=target_face
        covered[pid,target]=True
        sources[pid,target]=source
        destinations.append(target)
    return TransportMap(covered,sources,tuple(destinations),transport,
                        TransportDiagnostics(1,1,0.,0.,0.,0.,0.))


def test_raster_loss_is_the_same_single_conservative_slab_transaction(monkeypatch):
    mesh,state,system=world()
    boundary=classify_boundaries(mesh,system,1000.,4.,1.)[0]
    source,target=boundary.face_a,boundary.face_b
    transport=initialize_transport_state(4)
    tmap=_map_with_one_collision(mesh,state,system,source,target,transport)
    monkeypatch.setattr('tectonics.transport.build_transport_map',lambda *a,**k:tmap)
    m=memory()
    seen=[]
    def sink(events,new_state):
        seen.extend(events)
        accept_slab_material(mesh,m.young_boundary_state,events,new_state.time_myr)
        synchronize_young_zones(m,new_state,SubductionMemoryParameters())
    updated,_,_,diag=advance_lithosphere(mesh,system,state,1.,1000.,5.,24.,
        constant_eccentricity(0.),transport_state=transport,young_subduction_sink=sink)
    assert len(seen)==1
    assert seen[0].source_face==source and seen[0].target_face==target
    assert seen[0].subducting_plate==state.cell_plate[source]
    assert seen[0].overriding_plate==state.cell_plate[target]
    assert sum(e.oceanic_volume_km3 for e in seen)==diag.oceanic_subducted_volume_km3
    assert m.young_boundary_state.cumulative_accepted_oceanic_volume_km3==diag.oceanic_subducted_volume_km3
    assert abs(diag.oceanic_volume_balance_error_km3)<1e-6
    assert np.sum(updated.oceanic_volume_km3)+diag.oceanic_subducted_volume_km3==pytest.approx(
        np.sum(state.oceanic_volume_km3)+diag.oceanic_created_volume_km3,rel=2e-14)
    assert np.linalg.norm(accepted_slab_torques_nm(m,4,1000.,5.))>0.


def test_topology_relabeling_conserves_inventory_and_reorders_torque():
    mesh,state,system=world()
    b=classify_boundaries(mesh,system,1000.,4.,1.)[0]
    m=memory()
    accept_slab_material(mesh,m.young_boundary_state,[event(mesh,source_face=b.face_a,
        target_face=b.face_b,subducting_plate=b.plate_a,overriding_plate=b.plate_b)],1.)
    synchronize_young_zones(m,state,SubductionMemoryParameters())
    before=accepted_slab_torques_nm(m,4,1000.,5.)
    ids=np.array([3,2,0,1])
    new_plates=tuple(replace(system.plates[int(i)],plate_id=int(ids[i])) for i in np.argsort(ids))
    new_system=PlateSystem(ids[state.cell_plate],new_plates)
    remap_subduction_memory(mesh,system,new_system,m)
    np.testing.assert_array_equal(before,accepted_slab_torques_nm(m,4,1000.,5.)[ids])
    assert m.young_boundary_state.cumulative_accepted_oceanic_volume_km3==70.


def test_finite_slab_diffusion_matches_independent_heat_equation():
    from scipy.sparse import diags
    from scipy.sparse.linalg import expm_multiply
    from tectonics.genesis import SECONDS_PER_MYR
    count=240
    diagonal=np.full(count,-2.)
    diagonal[[0,-1]]=-3.  # Dirichlet hot bath at both outer cell faces.
    matrix=diags([np.ones(count-1),diagonal,np.ones(count-1)],[-1,0,1],format='csr')*count**2
    fourier=.03
    numerical=float(expm_multiply(matrix*fourier,np.ones(count)).mean())
    age=fourier*(30e3)**2/(1e-6*SECONDS_PER_MYR)
    analytic=slab_thermal_deficit_fraction(age,30.,1e-6)
    assert analytic==pytest.approx(numerical,abs=2e-5)
    assert slab_thermal_deficit_fraction(0.,30.,1e-6)==1.
    assert slab_thermal_deficit_fraction(1e-14,30.,1e-6)>1.-1e-6
    assert slab_thermal_deficit_fraction(4.*age,60.,1e-6)==pytest.approx(analytic,rel=1e-14)


def test_warming_reduces_current_buoyancy_without_consuming_material_again():
    mesh,_,_=world()
    m=memory()
    accept_slab_material(mesh,m.young_boundary_state,[event(mesh)],1.)
    m.time_myr=1.
    original=np.linalg.norm(accepted_slab_torques_nm(m,4,1000.,5.))
    m.time_myr=11.
    warmed=np.linalg.norm(accepted_slab_torques_nm(m,4,1000.,5.))
    assert 0.<warmed<.05*original
    d=accepted_slab_inventory_diagnostics(m)
    assert d['cumulative_accepted_oceanic_volume_km3']==70.
    assert d['attached_oceanic_volume_km3']==70.
    assert d['current_negative_buoyancy_mass_kg']<.05*d['initial_negative_buoyancy_mass_kg']


def test_physical_depth_transfers_oldest_material_without_losing_volume():
    mesh,_,_=world()
    m=memory()
    inv=m.young_boundary_state
    inv.mantle_depth_km=1.
    accept_slab_material(mesh,inv,[event(mesh)],1.)
    accept_slab_material(mesh,inv,[event(mesh)],2.)
    limit_attached_inventory(inv,90.)
    cohorts=next(iter(inv.segments.values())).thermal_cohorts
    assert cohorts[0].deep_transfer_fraction==1.
    assert cohorts[1].deep_transfer_fraction==0.
    m.time_myr=2.
    d=accepted_slab_inventory_diagnostics(m)
    assert d['cumulative_accepted_oceanic_volume_km3']==140.
    assert d['deep_oceanic_volume_km3']==70.
    assert d['attached_oceanic_volume_km3']==70.
    assert d['attached_area_km2']==10.
    restored=memory_from_json(memory_to_json(m))
    assert accepted_slab_inventory_diagnostics(restored)==d


def test_hierarchical_refinement_preserves_parcel_mass_coldness_and_vector_torque():
    mesh,state,system=world()
    m=memory()
    boundaries=classify_boundaries(mesh,system,1000.,4.,1.)
    b=boundaries[0]
    advance_contact_geometry(mesh,boundaries,1000.,100.,m.young_boundary_state)
    accept_slab_material(mesh,m.young_boundary_state,[event(mesh,source_face=b.face_a,
        target_face=b.face_b,subducting_plate=b.plate_a,overriding_plate=b.plate_b,
        contact_key=contact_key(b))],1.)
    m.time_myr=4.
    old=boundary_state_to_json(m.young_boundary_state)
    fine=build_icosphere(2)
    ancestors=np.arange(fine.cell_count)//4
    refined=deepcopy(m)
    refined.young_boundary_state=remesh_young_inventory(mesh,fine,state.cell_plate,
        state.cell_plate[ancestors],ancestors,m.young_boundary_state,1000.)
    np.testing.assert_allclose(accepted_slab_torques_nm(refined,4,1000.,5.),
                               accepted_slab_torques_nm(m,4,1000.,5.),rtol=2e-15)
    a=accepted_slab_inventory_diagnostics(m);c=accepted_slab_inventory_diagnostics(refined)
    for name in a:
        if not name.endswith('count'):
            assert c[name]==pytest.approx(a[name],rel=2e-15,abs=1e-10)
    for name in ('cumulative_closure_area_km2','cumulative_opening_area_km2'):
        before=sum(getattr(x,name) for x in m.young_boundary_state.contacts.values())
        after=sum(getattr(x,name) for x in refined.young_boundary_state.contacts.values())
        assert after==pytest.approx(before,rel=2e-14)
    assert boundary_state_to_json(m.young_boundary_state)==old
    assert len(refined.young_boundary_state.segments)>len(m.young_boundary_state.segments)


def connected_world():
    mesh, state, system = world()
    m = memory()
    m.young_boundary_state.connectivity_model = 'local_edge_transfer_v1'
    boundaries = classify_boundaries(mesh, system, 1000., 0., 0.)
    advance_contact_geometry(mesh, boundaries, 1000., 1., m.young_boundary_state)
    return mesh, state, system, m, boundaries


def contact_event(mesh, contact, **kwargs):
    values = dict(source_face=contact.face_a, target_face=contact.face_b,
        subducting_plate=contact.plate_a, overriding_plate=contact.plate_b,
        contact_key=contact.key, trench_length_km=contact.trench_length_km,
        midpoint=contact.midpoint, torque_direction=contact.torque_direction_ab)
    values.update(kwargs)
    return event(mesh, **values)


def test_force_sections_group_cohorts_and_follow_current_contact_polarity():
    mesh, state, _, m, boundaries = connected_world()
    b = boundaries[0]
    c = m.young_boundary_state.contacts[contact_key(b)]
    accept_slab_material(mesh, m.young_boundary_state, [contact_event(mesh, c)], 1.)
    accept_slab_material(mesh, m.young_boundary_state, [contact_event(mesh, c)], 2.)
    state.time_myr = 2.
    params = SubductionMemoryParameters()
    synchronize_young_zones(m, state, params)
    before = memory_to_json(m)
    (section,) = accepted_slab_force_sections(m, params=params)
    assert memory_to_json(m) == before  # Force assembly is a read-only query.
    assert section.accepted_area_km2 == 20.
    assert section.oceanic_volume_km3 == 140.
    assert section.cold_mantle_volume_km3 == 600.
    assert section.area_thickness_cubed_km5 == 20.*30.**3
    assert section.slab_length_km == pytest.approx(20./c.trench_length_km)
    expected_mass = 1.8e13*(1.+slab_thermal_deficit_fraction(1., 30., 1e-6))
    assert section.density_excess_mass_kg == pytest.approx(expected_mass)
    assert section.source_face == c.face_a and section.receiver_face == c.face_b
    np.testing.assert_allclose(section.torque_direction, c.torque_direction_ab)
    # A saved initial vector is historical; a present oriented edge owns force.
    for cohort in next(iter(m.young_boundary_state.segments.values())).thermal_cohorts:
        cohort.initial_buoyancy_moment_kg = [0., 0., 0.]
    assert accepted_slab_force_sections(m, params=params) == (section,)


def test_adjacent_grid_hop_transfers_attachment_without_duplicating_material():
    mesh, state, _, m, boundaries = connected_world()
    adjacent = next((a, b) for a in boundaries for b in boundaries
        if contact_key(a) != contact_key(b)
        and {a.plate_a, a.plate_b} == {b.plate_a, b.plate_b}
        and {a.vertex_u, a.vertex_v} & {b.vertex_u, b.vertex_v})
    old, new = adjacent
    contact = m.young_boundary_state.contacts[contact_key(old)]
    accept_slab_material(mesh, m.young_boundary_state, [contact_event(mesh, contact)], 1.)
    segment = next(iter(m.young_boundary_state.segments.values()))
    cohorts = deepcopy(segment.thermal_cohorts)
    advance_contact_geometry(mesh, [new], 1000., 1., m.young_boundary_state)
    assert segment.attached and segment.contact_key == contact_key(new)
    assert segment.thermal_cohorts == cohorts
    state.time_myr = 2.
    synchronize_young_zones(m, state, SubductionMemoryParameters())
    (section,) = accepted_slab_force_sections(m)
    assert section.accepted_area_km2 == 10.
    assert section.oceanic_volume_km3 == 70.
    assert m.young_boundary_state.cumulative_accepted_oceanic_volume_km3 == 70.
    expected = m.young_boundary_state.contacts[contact_key(new)]
    sign = 1. if expected.plate_a == section.subducting_plate else -1.
    np.testing.assert_allclose(section.torque_direction, sign*np.asarray(expected.torque_direction_ab))


def test_disconnected_slab_cannot_jump_to_remote_pair_or_reconnect_via_new_acceptance():
    mesh, state, _, m, boundaries = connected_world()
    old, remote = next((a, b) for a in boundaries for b in boundaries
        if {a.plate_a, a.plate_b} == {b.plate_a, b.plate_b}
        and not ({a.vertex_u, a.vertex_v} & {b.vertex_u, b.vertex_v}))
    c = m.young_boundary_state.contacts[contact_key(old)]
    parcel = contact_event(mesh, c)
    accept_slab_material(mesh, m.young_boundary_state, [parcel], 1.)
    advance_contact_geometry(mesh, [remote], 1000., 1., m.young_boundary_state)
    detached = next(iter(m.young_boundary_state.segments.values()))
    assert not detached.attached
    assert accepted_slab_force_sections(m, params=SubductionMemoryParameters()) == ()
    advance_contact_geometry(mesh, [old], 1000., 1., m.young_boundary_state)
    accept_slab_material(mesh, m.young_boundary_state, [parcel], 3.)
    state.time_myr = 3.
    synchronize_young_zones(m, state, SubductionMemoryParameters())
    assert len(m.young_boundary_state.segments) == 2
    assert not detached.attached
    (section,) = accepted_slab_force_sections(m)
    assert section.oceanic_volume_km3 == 70.
    ledger = accepted_slab_inventory_diagnostics(m)
    assert ledger['cumulative_accepted_oceanic_volume_km3'] == 140.
    assert ledger['attached_oceanic_volume_km3'] == 70.
    assert ledger['unresolved_or_detached_oceanic_volume_km3'] == 70.


@pytest.mark.parametrize('failure', ['absent', 'wrong_pair', 'detached', 'broken_off', 'deep'])
def test_force_sections_exclude_all_disconnected_or_deep_material(failure):
    mesh, state, _, m, boundaries = connected_world()
    c = m.young_boundary_state.contacts[contact_key(boundaries[0])]
    accept_slab_material(mesh, m.young_boundary_state, [contact_event(mesh, c)], 1.)
    state.time_myr = 1.
    synchronize_young_zones(m, state, SubductionMemoryParameters())
    s = next(iter(m.young_boundary_state.segments.values()))
    if failure == 'absent':
        c.present = False
    elif failure == 'wrong_pair':
        c.plate_b = 999
    elif failure == 'detached':
        s.attached = False
    elif failure == 'broken_off':
        m.zones[(s.subducting_plate, s.overriding_plate)].broken_off = True
    else:
        s.thermal_cohorts[0].deep_transfer_fraction = 1.
    assert accepted_slab_force_sections(m) == ()


def test_force_section_refinement_conserves_extensive_geometry_and_bending_width():
    mesh, state, _, m, boundaries = connected_world()
    c = m.young_boundary_state.contacts[contact_key(boundaries[0])]
    accept_slab_material(mesh, m.young_boundary_state, [contact_event(mesh, c)], 1.)
    state.time_myr = 4.
    params = SubductionMemoryParameters()
    synchronize_young_zones(m, state, params)
    (coarse,) = accepted_slab_force_sections(m, params=params)
    fine_mesh = build_icosphere(2)
    ancestors = np.arange(fine_mesh.cell_count)//4
    fine = deepcopy(m)
    fine.young_boundary_state = remesh_young_inventory(mesh, fine_mesh, state.cell_plate,
        state.cell_plate[ancestors], ancestors, m.young_boundary_state, 1000.)
    sections = accepted_slab_force_sections(fine, params=params)
    assert len(sections) == 2
    for name in ('trench_length_km', 'accepted_area_km2', 'oceanic_volume_km3',
                 'cold_mantle_volume_km3', 'area_thickness_cubed_km5', 'density_excess_mass_kg'):
        assert sum(getattr(s, name) for s in sections) == pytest.approx(getattr(coarse, name), rel=2e-14)
    for s in sections:
        assert s.slab_length_km == pytest.approx(coarse.slab_length_km, rel=2e-14)
    bending_width = lambda s: s.trench_length_km*s.area_thickness_cubed_km5/s.accepted_area_km2
    assert sum(map(bending_width, sections)) == pytest.approx(bending_width(coarse), rel=2e-14)
    assert memory_to_json(memory_from_json(memory_to_json(fine))) == memory_to_json(fine)


def test_connectivity_is_explicit_and_legacy_checkpoint_shape_is_preserved():
    inv = YoungBoundaryState()
    legacy = boundary_state_to_json(inv)
    assert 'connectivity_model' not in legacy
    assert boundary_state_from_json(legacy).connectivity_model == 'legacy_fixed_contacts'
    inv.connectivity_model = 'local_edge_transfer_v1'
    saved = boundary_state_to_json(inv)
    assert boundary_state_to_json(boundary_state_from_json(saved)) == saved
    saved['connectivity_model'] = 'unknown'
    with pytest.raises(ValueError, match='connectivity model'):
        boundary_state_from_json(saved)


def test_remote_raster_collision_cannot_add_force_to_existing_same_pair_trench():
    mesh, state, _, m, boundaries = connected_world()
    a, b = next((a, b) for a in boundaries for b in boundaries
        if {a.plate_a, a.plate_b} == {b.plate_a, b.plate_b}
        and not ({a.face_a, a.face_b} & {b.face_a, b.face_b}))
    contact = m.young_boundary_state.contacts[contact_key(a)]
    accept_slab_material(mesh, m.young_boundary_state, [contact_event(mesh, contact)], 1.)
    distant = contact_event(mesh, contact, source_face=b.face_a, target_face=b.face_b)
    original = deepcopy(distant)
    accept_slab_material(mesh, m.young_boundary_state, [distant], 2.)
    assert distant == original  # The producer's transaction remains immutable.
    state.time_myr = 2.
    synchronize_young_zones(m, state, SubductionMemoryParameters())
    (section,) = accepted_slab_force_sections(m)
    assert section.oceanic_volume_km3 == 70.
    diagnostic = accepted_slab_inventory_diagnostics(m)
    assert diagnostic['cumulative_accepted_oceanic_volume_km3'] == 140.
    assert diagnostic['unresolved_or_detached_oceanic_volume_km3'] == 70.
    assert diagnostic['unresolved_or_detached_fraction_of_accepted_volume'] == .5


def test_tensile_failure_commits_once_and_preserves_volume_history_on_resume_and_remesh():
    mesh, state, _, m, boundaries = connected_world()
    c = m.young_boundary_state.contacts[contact_key(boundaries[0])]
    accept_slab_material(mesh, m.young_boundary_state, [contact_event(mesh, c)], 1.)
    segment = next(iter(m.young_boundary_state.segments.values()))
    segment.thermal_cohorts[0].deep_transfer_fraction = .25
    state.time_myr = 2.
    params = SubductionMemoryParameters()
    synchronize_young_zones(m, state, params)
    failure = dict(contact_key=c.key, subducting_plate=c.plate_a, overriding_plate=c.plate_b,
        tension_n=100., capacity_n=10., iteration=0)
    commit_slab_neck_failures(m, [failure, failure], state.time_myr)
    synchronize_young_zones(m, state, params)
    assert not segment.attached
    assert not m.zones[(c.plate_a, c.plate_b)].active
    assert m.zones[(c.plate_a, c.plate_b)].slab_depth_km == 0.
    assert not np.any(m.zones[(c.plate_a, c.plate_b)].torque_axis)
    ledger = accepted_slab_inventory_diagnostics(m)
    assert ledger['mechanical_detachment_count'] == 1
    assert ledger['cumulative_accepted_oceanic_volume_km3'] == 70.
    assert ledger['deep_oceanic_volume_km3'] == 17.5
    assert ledger['unresolved_or_detached_oceanic_volume_km3'] == 52.5
    saved = memory_to_json(m)
    resumed = memory_from_json(saved)
    commit_slab_neck_failures(resumed, [failure], 3.)
    assert memory_to_json(resumed) == saved
    (history,) = resumed.young_boundary_state.mechanical_detachments
    assert history['retained_oceanic_volume_km3'] == 52.5
    assert history['time_myr'] == 2.
    fine = build_icosphere(2)
    ancestors = np.arange(fine.cell_count)//4
    refined = remesh_young_inventory(mesh, fine, state.cell_plate, state.cell_plate[ancestors],
        ancestors, resumed.young_boundary_state, 1000.)
    assert refined.mechanical_detachments == [history]
    assert all(not s.attached for s in refined.segments.values())


def test_invalid_neck_failure_transaction_cannot_partially_detach_inventory():
    mesh, state, _, m, boundaries = connected_world()
    c = m.young_boundary_state.contacts[contact_key(boundaries[0])]
    accept_slab_material(mesh, m.young_boundary_state, [contact_event(mesh, c)], 1.)
    before = memory_to_json(m)
    valid = dict(contact_key=c.key, subducting_plate=c.plate_a, overriding_plate=c.plate_b,
        tension_n=100., capacity_n=10., iteration=0)
    invalid = {**valid, 'contact_key': 'other', 'capacity_n': float('nan')}
    with pytest.raises(ValueError, match='finite'):
        commit_slab_neck_failures(m, [valid, invalid], 2.)
    assert memory_to_json(m) == before


def test_ordered_layers_put_new_material_shallow_and_preserve_each_thermal_deficit():
    mesh, state, _, m, boundaries = connected_world()
    inv = m.young_boundary_state
    inv.buoyancy_geometry_model = 'ordered_thermal_cohorts_v1'
    c = inv.contacts[contact_key(boundaries[0])]
    accept_slab_material(mesh, inv, [contact_event(mesh, c, accepted_area_km2=20.,
        oceanic_volume_km3=140., cold_mantle_volume_km3=400., density_excess_mass_kg=2e13)], 1.)
    accept_slab_material(mesh, inv, [contact_event(mesh, c, cold_mantle_volume_km3=600.,
        density_excess_mass_kg=3e13)], 3.)
    accept_slab_material(mesh, inv, [contact_event(mesh, c, accepted_area_km2=30.,
        oceanic_volume_km3=210., cold_mantle_volume_km3=900., density_excess_mass_kg=6e13)], 5.)
    next(iter(inv.segments.values())).thermal_cohorts[0].deep_transfer_fraction = .25
    state.time_myr = 5.
    params = SubductionMemoryParameters()
    synchronize_young_zones(m, state, params)
    saved = memory_to_json(m)
    (legacy,) = accepted_slab_force_sections(m, params=params)
    assert legacy.buoyancy_layers == ()
    (section,) = accepted_slab_force_sections(m, params=params,
        buoyancy_model='ordered_thermal_cohorts_v1')
    assert memory_to_json(m) == saved
    assert replace(section, buoyancy_layers=()) == legacy
    layers = section.buoyancy_layers
    assert [layer.acceptance_time_myr for layer in layers] == [5., 3., 1.]
    assert [layer.age_myr for layer in layers] == [0., 2., 4.]
    assert [layer.accepted_area_km2 for layer in layers] == [30., 10., 15.]
    np.testing.assert_allclose([layer.arc_start_km for layer in layers],
        np.array([0., 30., 40.])/c.trench_length_km)
    np.testing.assert_allclose([layer.arc_end_km for layer in layers],
        np.array([30., 40., 55.])/c.trench_length_km)
    np.testing.assert_allclose([layer.density_excess_mass_kg for layer in layers],
        [6e13, 3e13*slab_thermal_deficit_fraction(2., 60., 1e-6),
         1.5e13*slab_thermal_deficit_fraction(4., 20., 1e-6)])
    for name in ('accepted_area_km2', 'cold_mantle_volume_km3',
                 'density_excess_mass_kg', 'area_thickness_cubed_km5'):
        assert sum(getattr(layer, name) for layer in layers) == pytest.approx(getattr(section, name))
    assert layers[-1].arc_end_km == pytest.approx(section.slab_length_km)


@pytest.mark.parametrize('geometry_model', ['uniform_thermal_mass_v1', 'ordered_thermal_cohorts_v1'])
def test_equal_time_depth_transfer_has_explicit_versioned_order(geometry_model):
    mesh, state, _, m, boundaries = connected_world()
    inv = m.young_boundary_state
    inv.buoyancy_geometry_model = geometry_model
    c = inv.contacts[contact_key(boundaries[0])]
    accept_slab_material(mesh, inv, [contact_event(mesh, c)], 1.)
    accept_slab_material(mesh, inv, [contact_event(mesh, c)], 2.)
    segment = next(iter(inv.segments.values()))
    old = segment.thermal_cohorts[0]
    # Two equally old but thermally different parcels; the scalar area has no
    # resolved order within this batch, as after same-time raster acceptance.
    twin = replace(old, initial_thickness_km=60., initial_density_excess_mass_kg=3.6e13,
        cold_mantle_volume_km3=600.)
    segment.thermal_cohorts.insert(1, twin)
    inv.mantle_depth_km = 20./c.trench_length_km  # Retain 20 of 30 km² at 90°.
    reversed_inventory = deepcopy(inv)
    next(iter(reversed_inventory.segments.values())).thermal_cohorts.reverse()
    for candidate in (inv, reversed_inventory):
        limit_attached_inventory(candidate, 90.)
    original = next(iter(inv.segments.values())).thermal_cohorts
    reverse = next(iter(reversed_inventory.segments.values())).thermal_cohorts
    if geometry_model == 'ordered_thermal_cohorts_v1':
        assert [cohort.deep_transfer_fraction for cohort in original] == [.5, .5, 0.]
        assert original == list(reversed(reverse))
    else:
        # The 0.4 FIFO tie behaviour is intentionally unchanged on resume.
        assert [cohort.deep_transfer_fraction for cohort in original] == [1., 0., 0.]
        assert original != list(reversed(reverse))
    assert sum(cohort.accepted_area_km2*(1.-cohort.deep_transfer_fraction)
        for cohort in original) == 20.
    before = deepcopy(original)
    limit_attached_inventory(inv, 90.)
    assert original == before  # Re-evaluating capacity cannot consume twice.


def test_ordered_layer_aggregation_is_independent_of_same_time_cohort_order():
    mesh, state, _, m, boundaries = connected_world()
    inv = m.young_boundary_state
    inv.buoyancy_geometry_model = 'ordered_thermal_cohorts_v1'
    c = inv.contacts[contact_key(boundaries[0])]
    accept_slab_material(mesh, inv, [contact_event(mesh, c)], 1.)
    cohorts = next(iter(inv.segments.values())).thermal_cohorts
    original = cohorts[0]
    cohorts.append(replace(original, initial_thickness_km=60.,
        cold_mantle_volume_km3=600., initial_density_excess_mass_kg=3.6e13))
    state.time_myr = 4.
    params = SubductionMemoryParameters()
    synchronize_young_zones(m, state, params)
    (section,) = accepted_slab_force_sections(m, params=params,
        buoyancy_model='ordered_thermal_cohorts_v1')
    cohorts.reverse()
    (reordered,) = accepted_slab_force_sections(m, params=params,
        buoyancy_model='ordered_thermal_cohorts_v1')
    assert reordered == section
    assert len(section.buoyancy_layers) == 1
    (layer,) = section.buoyancy_layers
    assert layer.density_excess_mass_kg == pytest.approx(
        1.8e13*slab_thermal_deficit_fraction(3., 30., 1e-6)
        +3.6e13*slab_thermal_deficit_fraction(3., 60., 1e-6))
    assert layer.area_thickness_cubed_km5 == 10.*30.**3+10.*60.**3


def test_ordered_layer_remesh_and_checkpoint_preserve_lengths_and_thermal_mass():
    mesh, state, _, m, boundaries = connected_world()
    inv = m.young_boundary_state
    inv.buoyancy_geometry_model = 'ordered_thermal_cohorts_v1'
    c = inv.contacts[contact_key(boundaries[0])]
    for time in (1., 3.):
        accept_slab_material(mesh, inv, [contact_event(mesh, c)], time)
    next(iter(inv.segments.values())).thermal_cohorts[0].deep_transfer_fraction = .25
    state.time_myr = 4.
    params = SubductionMemoryParameters()
    synchronize_young_zones(m, state, params)
    query = lambda obj: accepted_slab_force_sections(obj, params=params,
        buoyancy_model='ordered_thermal_cohorts_v1')
    (coarse,) = query(m)
    fine_mesh = build_icosphere(2)
    ancestors = np.arange(fine_mesh.cell_count)//4
    fine = deepcopy(m)
    fine.young_boundary_state = remesh_young_inventory(mesh, fine_mesh, state.cell_plate,
        state.cell_plate[ancestors], ancestors, inv, 1000.)
    assert fine.young_boundary_state.buoyancy_geometry_model == 'ordered_thermal_cohorts_v1'
    refined = query(fine)
    assert len(refined) == 2
    for i, old in enumerate(coarse.buoyancy_layers):
        children = [section.buoyancy_layers[i] for section in refined]
        for child in children:
            assert child.arc_start_km == pytest.approx(old.arc_start_km, rel=2e-14)
            assert child.arc_end_km == pytest.approx(old.arc_end_km, rel=2e-14)
        for name in ('accepted_area_km2', 'cold_mantle_volume_km3',
                     'density_excess_mass_kg', 'area_thickness_cubed_km5'):
            assert sum(getattr(child, name) for child in children) == pytest.approx(getattr(old, name), rel=2e-14)
    assert query(memory_from_json(memory_to_json(fine))) == refined


def test_buoyancy_geometry_is_explicit_and_legacy_checkpoint_shape_is_preserved():
    inv = YoungBoundaryState()
    legacy = boundary_state_to_json(inv)
    assert 'buoyancy_geometry_model' not in legacy
    assert boundary_state_from_json(legacy).buoyancy_geometry_model == 'uniform_thermal_mass_v1'
    inv.buoyancy_geometry_model = 'ordered_thermal_cohorts_v1'
    saved = boundary_state_to_json(inv)
    assert boundary_state_to_json(boundary_state_from_json(saved)) == saved
    saved['buoyancy_geometry_model'] = 'unknown'
    with pytest.raises(ValueError, match='buoyancy geometry model'):
        boundary_state_from_json(saved)
    with pytest.raises(ValueError, match='buoyancy geometry model'):
        accepted_slab_force_sections(None, buoyancy_model='unknown')
