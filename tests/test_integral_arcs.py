from copy import deepcopy
from dataclasses import replace
import json,math
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.spatial.transform import Rotation
from tectonics.mesh import build_icosphere
from tectonics.integral_arcs import ArcSegment,integrate_sources,SphereReceiver
from tectonics.lithosphere import LithosphereState
from tectonics.continental import ContinentalCycleParameters,initialize_continental_cycle,advance_continental_cycle
from tectonics.material_exchange import MaterialExchangeBudget,MaterialExchangeParameters,surface_volumes
from tectonics.thermal import ThermalParameters,initialize_thermal_state

R=5287.

def segment(angle=1.):
    return ArcSegment(np.array([1.,0.,0.]),np.array([math.cos(angle),math.sin(angle),0.]),
                      np.array([0.,0.,1.]),0,.7,105.)

def physical_integral(sigma=140.,outer=320.,angle=1.):
    return .7*angle*R*quad(lambda d:math.cos(d/R)*math.exp(-.5*((d-105.)/sigma)**2),0,105.+outer,
                          epsabs=1e-10,epsrel=1e-12)[0]

def state(mesh,f):
    n=mesh.cell_count;a=mesh.physical_cell_areas_km2(R)
    return LithosphereState(0.,np.zeros(n,np.int32),np.full(n,int(f>=.5),np.int8),np.ones(n)*100,
        np.full(n,35. if f>=.5 else 7.),np.zeros(n),continental_fraction=np.full(n,f),
        continental_volume_km3=a*f*35.)

@pytest.mark.parametrize('sub',[1,2,3])
@pytest.mark.parametrize('order',[8,16,32])
def test_physical_source_integral_independent_of_receiver_mesh(sub,order):
    mesh=build_icosphere(sub)
    field,diag=integrate_sources(mesh,np.zeros(mesh.cell_count,int),[segment()],R,140.,320.,order=order)
    assert math.fsum(field)==pytest.approx(physical_integral(),rel=1e-7 if order==8 else 2e-14)
    assert abs(diag['deposition_residual_km2'])<1e-8
    assert diag['outside_overriding_integral_km2']==0

def test_segmentation_does_not_multiply_source_budget():
    mesh=build_icosphere(2);s=segment();mid=np.array([math.cos(.37),math.sin(.37),0.])
    a,_=integrate_sources(mesh,np.zeros(mesh.cell_count,int),[s],R,140.,320.,order=16)
    b,_=integrate_sources(mesh,np.zeros(mesh.cell_count,int),[replace(s,end=mid),replace(s,start=mid)],R,140.,320.,order=16)
    assert math.fsum(a)==pytest.approx(math.fsum(b),rel=3e-15)

def test_rotation_preserves_receivers_and_source_work_measure():
    mesh=build_icosphere(2);s=segment();rot=Rotation.from_rotvec([.6,-.4,.2]).as_matrix()
    moved=replace(mesh,vertices=mesh.vertices@rot.T,centroids=mesh.centroids@rot.T)
    translated=replace(s,start=rot@s.start,end=rot@s.end,toward=rot@s.toward)
    owner=np.zeros(mesh.cell_count,int)
    a,_=integrate_sources(mesh,owner,[s],R,140.,320.,order=12)
    b,_=integrate_sources(moved,owner,[translated],R,140.,320.,order=12)
    assert b==pytest.approx(a,rel=2e-14,abs=1e-9)

def test_excluded_source_area_is_accounted_instead_of_renormalized_into_other_regions():
    mesh=build_icosphere(2);owner=(mesh.centroids[:,1]>.4).astype(int)
    field,d=integrate_sources(mesh,owner,[segment()],R,140.,320.,order=16)
    assert np.all(field[owner==1]==0)
    assert 0<d['outside_overriding_integral_km2']<d['source_integral_km2']
    assert sum(field)+d['outside_overriding_integral_km2']==pytest.approx(physical_integral(),rel=2e-14)

def test_locator_checks_triangle_containment():
    mesh=build_icosphere(3)
    assert np.array_equal(SphereReceiver(mesh).locate(mesh.centroids),np.arange(mesh.cell_count))

@pytest.mark.parametrize('f',[.25,.5,1.])
def test_mixed_material_thickening_counts_footprint_once_and_uses_finite_enthalpy(f):
    totals=[]
    for sub in (1,2,3):
        mesh=build_icosphere(sub);s=state(mesh,f);a=mesh.physical_cell_areas_km2(R)
        tparams=ThermalParameters();thermal=initialize_thermal_state(.5,R,7.12,tparams)
        budget,s,thermal=MaterialExchangeBudget.initialize(mesh,s,thermal,.5,R,7.12,tparams)
        integral,_=integrate_sources(mesh,s.cell_plate,[segment()],R,140.,320.,order=16)
        # Zero maturation isolates the already-existing continental material;
        # no threshold is tuned to obtain an event.
        p=ContinentalCycleParameters(analytic_maturation=True,arc_maturation_rate_per_myr=0.)
        cycle=initialize_continental_cycle(mesh);energy=math.fsum(budget.data['energy_m_j']+budget.data['energy_c_j'])
        for _ in range(4):
            s.time_myr+=2.
            s,cycle,d=advance_continental_cycle(mesh,s,[],cycle,2.,R,p,volcanic_arc_forcing=integral/a,integral_arc_forcing=True)
            budget.exchange([d.arc_thickening_volume_km3,0.],[0.,0.],surface_volumes(s),tparams)
        assert math.fsum(budget.data['energy_m_j']+budget.data['energy_c_j'])==energy
        assert max(map(abs,budget.diagnostics(tparams)['origin_mass_residual_kg']))<1e-6
        restored=MaterialExchangeBudget.restore(json.loads(json.dumps(budget.checkpoint())),budget.options)
        assert restored.checkpoint()==budget.checkpoint()
        totals.append(cycle.cumulative_generated_volume_km3)
    assert totals==pytest.approx([physical_integral()*f*p.continental_arc_thickening_km_per_myr*8]*3,rel=3e-14)

def test_conversion_preserves_preexisting_material_and_only_funds_oceanic_footprint():
    mesh=build_icosphere(1);s=state(mesh,.4);a=mesh.physical_cell_areas_km2(R)
    # Old fringe is thicker than the juvenile reference. Converting its label
    # must not delete that pre-existing material or turn it into negative birth.
    s.continental_volume_km3[:]=a*.4*70;s.oceanic_volume_km3=a*.6*7
    cycle=initialize_continental_cycle(mesh);cycle.felsic_potential[:]=1.
    p=ContinentalCycleParameters(analytic_maturation=True,continental_arc_thickening_km_per_myr=0.)
    s.time_myr=1.
    result,cycle,d=advance_continental_cycle(mesh,s,[],cycle,1.,R,p,volcanic_arc_forcing=np.ones(mesh.cell_count),integral_arc_forcing=True)
    born=sum(a)*.6*p.juvenile_continental_thickness_km
    assert cycle.cumulative_generated_volume_km3==pytest.approx(born,rel=2e-15)
    assert sum(result.continental_volume_km3)==pytest.approx(sum(s.continental_volume_km3)+born)
    assert d.juvenile_arc_area_created_km2==pytest.approx(sum(a)*.6)

def test_zero_source_does_not_create_material():
    mesh=build_icosphere(1);s=state(mesh,.4);cycle=initialize_continental_cycle(mesh)
    result,cycle,d=advance_continental_cycle(mesh,s,[],cycle,4.,R,ContinentalCycleParameters(analytic_maturation=True),
        volcanic_arc_forcing=np.zeros(mesh.cell_count),integral_arc_forcing=True)
    assert cycle.cumulative_generated_volume_km3==0
    assert np.array_equal(result.continental_volume_km3,s.continental_volume_km3)
