import math
from types import SimpleNamespace
import numpy as np
import pytest
from tectonics.mesh import build_icosphere
from tectonics.geological_area_transport import (
    MaterialPatch, transfer_rigid_patches, inspect_two_owner_contact, require_single_owner, UnresolvedContact,
)


def patch(mesh, face=0, owner=0):
    return MaterialPatch('one',owner,mesh.vertices[mesh.faces[face]],3e20,2e26,'mafic')


def test_unequal_area_transfer_preserves_source_not_receiver_cell_area():
    mesh=build_icosphere(1);p=patch(mesh);omega={0:np.array([.17,.11,-.09])}
    assert mesh.areas_unit_sphere.max()/mesh.areas_unit_sphere.min()>1.1
    result=transfer_rigid_patches(mesh,[p],omega,1.,5287.)
    assert len(result.pieces)>1
    assert result.diagnostics['max_source_area_relative_residual']<1e-12
    assert math.fsum(x.mass_kg for x in result.pieces)==pytest.approx(p.mass_kg,rel=1e-12)
    assert math.fsum(x.enthalpy_j for x in result.pieces)==pytest.approx(p.enthalpy_j,rel=1e-12)
    assert all(x.identity==p.identity and x.owner==p.owner and x.composition==p.composition for x in result.pieces)
    for x in result.pieces:assert x.area_km2<mesh.physical_cell_areas_km2(5287.)[x.target]


def test_fixed_reference_avoids_rebinning_diffusion_and_source_order_bias():
    mesh=build_icosphere(0);p=patch(mesh);q=patch(mesh,1,1)
    q=MaterialPatch('two',q.owner,q.reference_vertices,4e20,3e26,'felsic')
    omega={0:[0.,0.,.01],1:[0.,0.,-.01]}
    transfer_rigid_patches(mesh,[p,q],omega,.5,5287.)
    half=transfer_rigid_patches(mesh,[q,p],omega,1.,5287.)
    whole=transfer_rigid_patches(mesh,[p,q],omega,1.,5287.)
    def rows(out):return [(x.identity,x.owner,x.target,x.area_km2,x.mass_kg) for x in out.pieces]
    assert rows(half)==rows(whole)
    np.testing.assert_array_equal(p.reference_vertices,mesh.vertices[mesh.faces[0]])


def test_true_collision_keeps_both_identities_and_refuses_a_single_winner():
    mesh=build_icosphere(0);p=patch(mesh)
    q=MaterialPatch('two',1,p.reference_vertices.copy(),5e20,9e26,'felsic')
    result=transfer_rigid_patches(mesh,[p,q],{0:[0,0,0],1:[0,0,0]},1.,5287.)
    contact=inspect_two_owner_contact(mesh,result,5287.)
    assert contact['collision_area_km2']==pytest.approx(mesh.physical_cell_areas_km2(5287.)[0],rel=1e-12)
    assert {p.identity for p in result.pieces}=={'one','two'}
    assert math.fsum(p.mass_kg for p in result.pieces)==pytest.approx(8e20,rel=1e-12)
    with pytest.raises(UnresolvedContact):require_single_owner(contact)


def test_duplicate_source_identity_is_rejected():
    mesh=build_icosphere(0);p=patch(mesh)
    with pytest.raises(ValueError,match='Duplicate'):transfer_rigid_patches(mesh,[p,p],{0:[0,0,0]},1,5287)


def test_fragment_pickup_is_separate_event_with_measured_kinematics():
    from tectonics.connectivity import repair_plate_connectivity
    from tectonics.geological_contact import repair_audit
    from tectonics.plates import Plate,PlateSystem
    from tectonics.topology import _make_system_from_groups,_repaired_angular_velocities
    mesh=build_icosphere(2);owner=np.ones(mesh.cell_count,dtype=int)
    owner[np.abs(mesh.centroids[:,2])>.6]=0;crumb=int(np.argmax(mesh.centroids[:,0]));owner[crumb]=0
    before=PlateSystem(owner,tuple(Plate(i,int(np.flatnonzero(owner==i)[0]),np.array([0.,0.,1.]),.003+.001*i) for i in range(2)))
    repair=repair_plate_connectivity(mesh,owner,5287.,minimum_independent_area_km2=10e6,minimum_independent_cells=30)
    after=_make_system_from_groups(mesh,before,repair.cell_plate,_repaired_angular_velocities(mesh,before,repair.cell_plate,5287.))
    result=repair_audit(mesh,before,after,repair,5287.,10e6)
    events=result['fragment_transfers']
    assert len(events)==1 and events[0]['cells']==[crumb]
    assert events[0]['parent']==0 and events[0]['child']==1
    assert events[0]['max_speed_change_km_per_myr']>0
    assert events[0]['mechanical_work_closed'] is False
    assert events[0]['material_fields_changed'] is False
    assert result['max_pure_child_omega_error']<1e-17


def test_rigid_sphere_covers_each_unequal_receiver_without_area_duplication():
    mesh=build_icosphere(1)
    patches=[MaterialPatch(str(i),0,mesh.vertices[face],1.,2.,'mafic') for i,face in enumerate(mesh.faces)]
    result=transfer_rigid_patches(mesh,patches,{0:[.1,.04,-.07]},1.,5287.)
    areas=mesh.physical_cell_areas_km2(5287.)
    covered=np.bincount([p.target for p in result.pieces],weights=[p.area_km2 for p in result.pieces],minlength=mesh.cell_count)
    np.testing.assert_allclose(covered,areas,rtol=1e-12,atol=0)


def test_two_hemisphere_collision_matches_independent_spherical_lune():
    from analysis.geological_area_transport.experiment import fixture
    mesh,r,patches,omega=fixture(0)
    result=transfer_rigid_patches(mesh,patches,omega,1.,r)
    contact=inspect_two_owner_contact(mesh,result,r)
    expected=4*.01*r*r
    assert contact['collision_area_km2']==pytest.approx(expected,rel=1e-12)
    assert contact['gap_area_km2']==pytest.approx(expected,rel=1e-12)
    assert result.diagnostics['max_source_area_relative_residual']<1e-12


def test_single_owner_does_not_fill_uncovered_area_implicitly():
    contact=dict(cells=[dict(owners=[0],exclusive_owner=0,gap_area_km2=.01)],contacts=[])
    with pytest.raises(UnresolvedContact):require_single_owner(contact)
