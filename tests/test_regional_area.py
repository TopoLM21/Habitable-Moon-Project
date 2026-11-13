from copy import deepcopy
from dataclasses import replace
import json,math
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.spatial.transform import Rotation
from tectonics.mesh import build_icosphere
from tectonics.regional_material import RegionalMaterialParameters
from tectonics.regional_area import AreaMaterialInventory,ActiveAreaParameters
from tectonics.integral_arcs import ArcSegment
from tectonics.lithosphere import LithosphereState
from tectonics.continental import ContinentalCycleParameters
from tectonics.plates import Plate,PlateSystem

R=5287.
def inventory(sub=1,edge=400.,limit=131072):
    mesh=build_icosphere(sub);n=mesh.cell_count;a=mesh.physical_cell_areas_km2(R)
    state=LithosphereState(0.,np.zeros(n,int),np.zeros(n,np.int8),np.full(n,100.),np.full(n,7.),np.zeros(n),
        continental_fraction=np.zeros(n),continental_volume_km3=np.zeros(n),sediment_volume_km3=np.zeros(n),
        oceanic_volume_km3=a*7,oceanic_age_moment_km3_myr=a*700)
    params=RegionalMaterialParameters(material_subdivisions=sub)
    ap=ActiveAreaParameters(max_edge_km=edge,max_leaves=limit)
    return AreaMaterialInventory.initialize(mesh,state,R,params,ap)

def source():
    return ArcSegment(np.array([1.,0,0]),np.array([math.cos(1),math.sin(1),0]),np.array([0.,0,1]),0,.7,105.)

@pytest.mark.parametrize('sub',[1,2])
@pytest.mark.parametrize('edge',[500.,250.])
def test_physical_strip_measure_independent_of_material_parents(sub,edge):
    inv=inventory(sub,edge);s=source();before=math.fsum(inv.leaves.arrays['area'])
    inv.refine([s],[],R,140.,320.);field,diag=inv.integrated_field([s],[],R,140.,320.)
    exact=.7*R*quad(lambda d:math.cos(d/R)*math.exp(-.5*((d-105)/140)**2),0,425.,epsabs=1e-9)[0]
    exact_area=R*R*math.sin(425./R)
    assert math.fsum(field*inv.leaves.arrays['area'])==pytest.approx(exact,rel=3e-14)
    assert diag['source_supported_area_sum_km2']==pytest.approx(exact_area,rel=3e-14)
    assert diag['excluded_material_source_km2']==0.
    assert math.fsum(inv.leaves.arrays['area'])==pytest.approx(before,rel=3e-15)
    assert len(inv.roots)>20*4**sub

def test_rotation_preserves_partial_material_integrals_and_history():
    inv=inventory();s=source();inv.refine([s],[],R,140.,320.)
    values,diag=inv.integrated_field([s],[],R,140.,320.)
    inv.leaves.arrays['potential']=np.linspace(.1,.4,len(inv.roots))
    saved=inv.checkpoint();rot=Rotation.from_rotvec([.6,-.4,.2]);w=np.array([.6,-.4,.2])
    system=PlateSystem(inv.coarse.arrays['owner'].copy(),(Plate(0,0,w/np.linalg.norm(w),np.linalg.norm(w)),))
    inv.rotate(system,1.)
    moved=replace(s,start=rot.apply(s.start),end=rot.apply(s.end),toward=rot.apply(s.toward))
    other,_=inv.integrated_field([moved],[],R,140.,320.)
    assert other==pytest.approx(values,rel=2e-11,abs=1e-12)
    for key in ('area','basalt_volume','basalt_age_moment','potential'):
        assert inv.leaves.arrays[key].tolist()==saved['arrays']['leaf_'+key]

def test_refinement_preserves_nonuniform_volume_age_and_maturation_moments_and_restart():
    inv=inventory(edge=250.);a=inv.leaves.arrays
    a['potential']=np.linspace(.1,.4,len(a['area']))
    a['basalt_volume']*=np.linspace(.5,1.5,len(a['area']));a['basalt_age_moment']=a['basalt_volume']*np.linspace(10.,100.,len(a['area']))
    before={k:math.fsum(a[k]) for k in ('area','basalt_volume','basalt_age_moment')}
    moment=math.fsum(a['basalt_volume']*a['potential'])
    inv.refine([source()],[],R,140.,320.);a=inv.leaves.arrays
    for k in before:assert math.fsum(a[k])==pytest.approx(before[k],rel=2e-15)
    assert math.fsum(a['basalt_volume']*a['potential'])==pytest.approx(moment,rel=3e-15)
    inv.aggregate();saved=inv.checkpoint()
    restored=AreaMaterialInventory.restore(json.loads(json.dumps(saved)),inv.params,inv.area_params)
    assert restored.checkpoint()==saved
    for item in (inv,restored):
        item.refine([source()],[],R,140.,320.)
        f,_=item.integrated_field([source()],[],R,140.,320.)
        item.leaves.react(2.,f,ContinentalCycleParameters(analytic_maturation=True));item.aggregate()
    assert inv.checkpoint()==restored.checkpoint()

def test_numerical_refinement_budget_rejects_atomically_without_losing_material():
    inv=inventory(edge=50.,limit=100);before=inv.checkpoint()
    with pytest.raises(RuntimeError,match='leaf budget'):inv.refine([source()],[],R,140.,320.)
    assert inv.checkpoint()==before

def test_restart_rejects_corrupt_material_geometry():
    inv=inventory();saved=inv.checkpoint();saved['arrays']['reference_triangles'][0][0][0]+=.001
    with pytest.raises(ValueError,match='triangle|geometry'):AreaMaterialInventory.restore(saved,inv.params,inv.area_params)

def test_additive_sources_do_not_duplicate_material_area():
    inv=inventory();s=source();inv.refine([s],[],R,140.,320.)
    f,_=inv.integrated_field([s],[],R,140.,320.);before=inv.checkpoint()
    doubled,_=inv.integrated_field([s,s],[],R,140.,320.)
    assert doubled==pytest.approx(f*2,rel=3e-15)
    assert inv.checkpoint()==before
