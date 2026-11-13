from copy import deepcopy
from dataclasses import replace
import json,math
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from tectonics.regional_material import RegionalMaterialInventory,RegionalMaterialParameters,_pvalue,conservative_projection
from tectonics.mesh import build_icosphere
from tectonics.continental import ContinentalCycleParameters,initialize_continental_cycle
from tectonics.lithosphere import LithosphereState
from tectonics.plates import Plate,PlateSystem

R=5287.
def columns(sub=1):
    mesh=build_icosphere(sub);a=mesh.physical_cell_areas_km2(R);n=len(a)
    inv=RegionalMaterialInventory({'points':mesh.centroids,'owner':np.zeros(n,dtype=int),'area':a,'felsic_area':np.zeros(n),
        'felsic_volume':np.zeros(n),'felsic_age_moment':np.zeros(n),'basalt_volume':a*7,
        'basalt_age_moment':a*700,'potential':np.zeros(n)},RegionalMaterialParameters(material_subdivisions=sub))
    return mesh,inv

def shell(mesh):
    n=mesh.cell_count
    return LithosphereState(0.,np.zeros(n,int),np.zeros(n,np.int8),np.ones(n)*100,np.ones(n)*7,np.zeros(n),
        continental_fraction=np.zeros(n),continental_volume_km3=np.zeros(n),sediment_volume_km3=np.zeros(n))

@pytest.mark.parametrize('k',[0.,1e-10,.05,4.])
@pytest.mark.parametrize('s',[0.,10.])
def test_maturation_mass_history_against_independent_ode(k,s):
    v=30.;p=.2;q=.018;d=.004;dt=4.
    ref=solve_ivp(lambda t,y:[s-k*y[0],q*y[0]-(k+d)*y[1]],(0,dt),[v,v*p],method='Radau',rtol=1e-11,atol=1e-12)
    got=_pvalue(dt,np.array([v]),np.array([p]),np.array([s]),np.array([k]),np.array([q]),d)[0]
    assert got==pytest.approx(ref.y[1,-1]/ref.y[0,-1],rel=3e-8,abs=1e-10)

def test_analytic_maturation_event_and_budget_independent_of_time_partition():
    params=ContinentalCycleParameters(analytic_maturation=True,arc_maturation_rate_per_myr=.018,juvenile_threshold=.75)
    expected=-params.arc_potential_decay_myr*math.log(1-params.juvenile_threshold/(.7*.018*params.arc_potential_decay_myr))
    results=[]
    for dt in (4.,2.,1.):
        _,inv=columns();first=None;born=returned=0.
        for i in range(round(100/dt)):
            x=inv.react(dt,np.ones(80)*.7,params);born+=x['born_felsic_km3'];returned+=x['returned_basalt_km3']
            if first is None:first=x['first_event_myr']
        assert first==pytest.approx(expected,abs=1e-9)
        # Sum of up to 100 rounded increments: bound arithmetic, not geology.
        assert math.fsum(inv.arrays['felsic_volume'])==pytest.approx(born,rel=128*np.finfo(float).eps)
        assert returned==pytest.approx(math.fsum(inv.arrays['area'])*7,rel=3e-15)
        assert np.all(inv.arrays['basalt_volume']==0)
        results.append(math.fsum(inv.arrays['felsic_volume']))
    assert results==pytest.approx([results[0]]*3,rel=1e-14)

def test_transient_maturation_is_not_missed_by_endpoint_test():
    _,inv=columns();inv.arrays['basalt_volume'][:]=1000;inv.arrays['basalt_age_moment'][:]=0
    params=ContinentalCycleParameters(analytic_maturation=True,arc_maturation_rate_per_myr=.01,juvenile_threshold=.25)
    end=_pvalue(100.,np.array([1000.]),np.array([0.]),np.array([1.]),np.array([.1]),np.array([.01]),1/params.arc_potential_decay_myr)[0]
    assert end<params.juvenile_threshold
    x=inv.react(100.,np.ones(80),params,np.ones(80),np.ones(80)*.1)
    assert x['events']==80 and 0<x['first_event_myr']<100

def test_rotation_projection_and_restart_preserve_material_not_cell_labels():
    mesh,inv=columns(2);a=inv.arrays
    a['owner'][mesh.centroids[:,0]<0]=1
    a['felsic_area']=a['area']*.2;a['felsic_volume']=a['felsic_area']*35
    a['felsic_age_moment']=a['felsic_volume']*300;a['basalt_volume']*=.8;a['basalt_age_moment']*=.8
    a['potential']=np.linspace(.1,.4,len(a['area']))
    system=PlateSystem(a['owner'].copy(),(Plate(0,0,np.array([0.,0.,1.]),.002),Plate(1,1,np.array([1.,0.,0.]),.003)))
    inv.rotate(system,4.);saved=inv.checkpoint();restored=RegionalMaterialInventory.restore(json.loads(json.dumps(saved)),inv.params)
    assert saved==restored.checkpoint()
    for sub in (1,2,3):
        receiver=build_icosphere(sub);state,cycle,d=inv.project(receiver,shell(receiver),initialize_continental_cycle(receiver),R)
        assert math.fsum(state.oceanic_volume_km3)==pytest.approx(math.fsum(a['basalt_volume']),rel=2e-15)
        assert math.fsum(state.continental_volume_km3)==pytest.approx(math.fsum(a['felsic_volume']),rel=2e-15)
        assert math.fsum(state.oceanic_age_moment_km3_myr)==pytest.approx(math.fsum(a['basalt_age_moment']),rel=2e-15)
        assert np.all((state.continental_fraction>=0)&(state.continental_fraction<=1))
        assert d['max_receiver_area_relative_error']<inv.params.projection_area_rtol
        assert saved==inv.checkpoint() # projection may not alter authoritative history

def test_projection_rejects_incompatible_total_area():
    mesh,inv=columns()
    with pytest.raises(ValueError,match='area mismatch'):
        conservative_projection(inv.arrays['points'],inv.arrays['area']*1.01,mesh,R,inv.params)
