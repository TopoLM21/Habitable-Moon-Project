from dataclasses import replace
import json,math
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tectonics.mesh import SphereMesh,_build_topology,_subdivide,_spherical_triangle_area
from tectonics.material_contacts import MaterialBoundaryRegistry,_interval_coverage
from tectonics.plates import Plate,PlateSystem
from tectonics.regional_area import AreaMaterialInventory
from test_regional_area import inventory,R


def test_double_coverage_cannot_pay_for_an_uncovered_bank_segment():
    # Both data sets have total length 1. Only one covers the whole bank.
    assert _interval_coverage([(0.,.5),(.5,1.)])==(1.,0.)
    assert _interval_coverage([(0.,.5),(0.,.5)])==(.5,.5)
    assert _interval_coverage([(0.,1.),(.25,.75),(.25,.75)])==(1.,1.)
    assert _interval_coverage([])==(0.,0.)


def test_local_coincident_banks_reject_double_coverage_and_a_gap_together():
    # One quarter-circle bank faces two identical eighth-circle banks.
    # Integral lengths match, but half the first bank has no neighbour.
    reg=MaterialBoundaryRegistry.__new__(MaterialBoundaryRegistry)
    reg.owners=reg.bank_owners=np.arange(3)
    arc=lambda x,y:np.array([[math.cos(x),math.sin(x),0.],[math.cos(y),math.sin(y),0.]])
    reg.reference=np.array([arc(0.,math.pi/2),arc(0.,math.pi/4),arc(0.,math.pi/4)])
    reg.outward=np.array([[0.,0.,1.],[0.,0.,-1.],[0.,0.,-1.]])
    # Old-partner displacement is only a paired-bank diagnostic. An unused
    # zero-length-free bank is appended away from this local example.
    reg.reference=np.concatenate([reg.reference,[arc(math.pi,3*math.pi/2)]])
    reg.outward=np.vstack([reg.outward,[0.,0.,1.]])
    reg.owners=reg.bank_owners=np.arange(4);reg.identity='local-independent-bank-control'
    q=Rotation.from_rotvec(np.zeros((4,3))).as_quat()
    _,diag=reg.evaluate(q,R)
    assert diag['multiply_covered_bank_length_km']==pytest.approx(math.pi*R/4,rel=2e-14)
    assert diag['unmatched_bank_length_km']==pytest.approx(3*math.pi*R/4,rel=2e-14)
    with pytest.raises(ValueError,match='not geometrically closed'):reg.require_shared_support(q,R)


def hemispheres(level):
    vertices=np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,1],[0,0,-1]],float)
    faces=np.array([[4,i,(i+1)%4] for i in range(4)]+[[5,(i+1)%4,i] for i in range(4)])
    for _ in range(level):vertices,faces=_subdivide(vertices,faces)
    center=np.sum(vertices[faces],axis=1);center/=np.linalg.norm(center,axis=1)[:,None]
    areas=np.array([_spherical_triangle_area(*t) for t in vertices[faces]])
    mesh=SphereMesh(vertices,faces,center,areas,*_build_topology(faces))
    return MaterialBoundaryRegistry(mesh,(center[:,2]<0).astype(int))


@pytest.mark.parametrize('level',[0,1,2])
@pytest.mark.parametrize('slide',[0.,.4,2.3])
def test_equatorial_contact_survives_sliding_and_changed_material_neighbours(level,slide):
    reg=hemispheres(level);q=Rotation.from_rotvec([[0.,0.,0.],[0.,0.,slide]]).as_quat()
    contacts,diag=reg.require_shared_support(q,R)
    assert math.fsum(c.length_km for c in contacts)==pytest.approx(2*math.pi*R,rel=3e-14)
    assert diag['unmatched_bank_length_km']<1e-8
    omega=np.array([0.,0.,.7])
    for c in contacts:
        mid=c.start+c.end;mid/=np.linalg.norm(mid)
        relative=R*np.cross(omega,mid)
        assert abs(relative@c.normal_a_to_b)<1e-10 # tangential slip, no opening
        # Unit arbitrary line-resultant force: common geometry gives equal
        # relative-velocity power and Euler torque power; no runtime SI claim.
        force=np.array([.7,-.3,.2]);torque_b=R*np.cross(mid,force)
        assert torque_b@omega==pytest.approx(force@relative,rel=2e-14,abs=1e-12)


def test_common_rotation_covariance_and_gapped_banks_are_not_contacts():
    reg=hemispheres(1);q=Rotation.from_rotvec([[.4,-.3,.7]]*2).as_quat()
    _,d=reg.require_shared_support(q,R)
    assert d['coincident_contact_length_km']==pytest.approx(2*math.pi*R,rel=3e-14)
    q=Rotation.from_rotvec([[0.,0.,0.],[.1,0.,0.]]).as_quat()
    _,d=reg.evaluate(q,R)
    assert d['coincident_contact_length_km']==0.
    assert d['old_partner_signed_separation_min_km']<0<d['old_partner_signed_separation_max_km']
    with pytest.raises(ValueError,match='gap/overlap'):reg.require_shared_support(q,R)


def test_native_material_registry_restart_and_failure_before_inventory_mutation():
    inv=inventory();owner=(inv.mesh.centroids[:,2]<0).astype(int)
    inv.coarse.arrays['owner']=owner.copy();inv.leaves.arrays['owner']=owner.copy()
    inv.quaternions=Rotation.from_rotvec(np.zeros((2,3))).as_quat()
    inv.area_params=replace(inv.area_params,contact_policy='require_shared_support')
    saved=inv.checkpoint();copy=AreaMaterialInventory.restore(json.loads(json.dumps(saved)),inv.params,inv.area_params)
    _,a=inv.material_boundary_registry().require_shared_support(inv.quaternions,R)
    _,b=copy.material_boundary_registry().require_shared_support(copy.quaternions,R)
    assert a==b
    system=PlateSystem(owner,(Plate(0,0,np.array([0.,0.,1.]),0.),Plate(1,1,np.array([1.,0.,0.]),.1)))
    with pytest.raises(ValueError,match='not geometrically closed'):
        inv.runtime_step(inv.mesh,None,system,None,[],None,R,1.,None,None,7.)
    assert inv.checkpoint()==saved
