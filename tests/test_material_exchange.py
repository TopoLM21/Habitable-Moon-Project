from copy import deepcopy
from dataclasses import asdict, replace
from decimal import Decimal, localcontext
import json
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tectonics.mesh import build_icosphere
from tectonics.lithosphere import LithosphereState
from tectonics.thermal import (ThermalParameters, initialize_thermal_state, _geometry, SECONDS_PER_MYR,
                              radiogenic_power_w, tidal_heating_power_w, convective_state)
from tectonics.tides import constant_eccentricity
from tectonics.material_exchange import (MaterialExchangeBudget, MaterialExchangeParameters,
                                        linear_thermal_exchange, surface_volumes)


def initialized(options=MaterialExchangeParameters()):
    mesh=build_icosphere(1);n=mesh.cell_count
    types=np.asarray(mesh.centroids[:,0]>0,dtype=np.int8)
    state=LithosphereState(0.,np.zeros(n,np.int32),types,np.ones(n),np.where(types==1,35.,7.),np.zeros(n))
    p=ThermalParameters();thermal=initialize_thermal_state(.5,5287.,7.12,p)
    budget,state,new=MaterialExchangeBudget.initialize(mesh,state,thermal,.5,5287.,7.12,p,options)
    return budget,state,new,p,thermal


def advance(budget,thermal,dt,p):
    return budget.advance(thermal,dt,.5,5287.,7.12,47.,5.,constant_eccentricity(.00047),p)[0]


def decimal_energy(budget):
    with localcontext() as c:
        c.prec=70
        return sum(Decimal(x) for key in ('energy_m_j','energy_c_j') for x in budget.data[key])


@pytest.mark.parametrize('temperature',[(1800.,600.),(600.,1800.)])
def test_reversible_two_body_heat_exchange_matches_analytic_solution(temperature):
    caps=np.array([1e20,2e20]);g=1e6;h=2.
    result,q,_=linear_thermal_exchange(temperature,caps,(g,0.),(0.,0.,0.),288.,h)
    average=float(np.dot(caps,temperature)/sum(caps))
    difference=(temperature[0]-temperature[1])*math.exp(-g*sum(1/caps)*h*SECONDS_PER_MYR)
    exact=[average+caps[1]/sum(caps)*difference,average-caps[0]/sum(caps)*difference]
    assert result==pytest.approx(exact,rel=1e-14)
    assert q[2]==0
    assert np.dot(caps,result)==pytest.approx(np.dot(caps,temperature),rel=1e-15)
    assert np.sign(q[3])==np.sign(temperature[0]-temperature[1])
    assert float(np.dot(caps,np.log(result/np.asarray(temperature))))>=0


def test_initialization_carves_crust_without_creating_heat_or_mass():
    b,s,t,p,old=initialized()
    initial=_geometry(.5,5287.,p)[1]*p.mantle_heat_capacity_j_kg_k*old.mantle_temperature_k
    assert decimal_energy(b)==Decimal(initial)
    assert b.diagnostics(p)['mass_residual_kg']==pytest.approx(0.,abs=math.ulp(b.data['total_mass_kg']))
    assert t.mantle_temperature_k>old.mantle_temperature_k
    assert p.surface_temperature_k<b.temperatures(p)[1]<t.mantle_temperature_k
    assert all(x==0 for x in b.diagnostics(p)['origin_mass_residual_kg'])


def test_birth_and_recycling_transfer_donor_enthalpy_and_provenance():
    b,s,t,p,_=initialized();before=decimal_energy(b);m0=b.pool_masses().copy()
    born=np.array([1e6,2e6]);returned=np.array([2e5,3e5]);vol=surface_volumes(s)+born-returned
    entropy0=m0[0]*math.log(b.temperatures(p)[0])+sum(m0[1:])*math.log(b.temperatures(p)[1])
    b.exchange(born,returned,vol,p)
    m=b.pool_masses();temps=b.temperatures(p)
    entropy=m[0]*math.log(temps[0])+sum(m[1:])*math.log(temps[1])
    assert entropy>=entropy0
    with localcontext() as c:
        c.prec=70
        assert abs(decimal_energy(b)-before)<Decimal('1e10')
    diag=b.diagnostics(p)
    assert max(map(abs,diag['origin_mass_residual_kg']))<1e9
    assert diag['created_kg']>0 and diag['recycled_kg']>0
    assert np.array_equal(b.data['surface_volumes_km3'],vol)
    assert np.all(np.asarray(b.data['origin_hi'])[1:,0]>0) # initial mantle material really reached both crust pools


def test_small_exchange_is_preserved_below_large_total_energy_roundoff():
    b,s,t,p,_=initialized();before=deepcopy(b.data['energy_m_j']);tm=b.temperatures(p)[0]
    vol=surface_volumes(s)+[1.,0.]
    b.exchange([1.,0.],[0.,0.],vol,p)
    with localcontext() as c:
        c.prec=70
        delta=sum(map(Decimal,b.data['energy_m_j']))-sum(map(Decimal,before))
        expected=-Decimal(1e9)*Decimal(b.options.continental_rock_density_kg_m3)*Decimal(p.mantle_heat_capacity_j_kg_k)*Decimal(tm)
        assert abs((delta-expected)/expected)<Decimal('3e-16')


@pytest.mark.parametrize('kind',['mantle','crust','geometry'])
def test_invalid_transfer_is_atomic(kind):
    b,s,t,p,_=initialized();snapshot=b.checkpoint();target=surface_volumes(s)
    born,returned=np.zeros(2),np.zeros(2)
    if kind=='mantle':born[0]=b.data['total_mass_kg']/(2800e9)
    if kind=='crust':returned[0]=target[0]*2
    if kind=='geometry':target[0]+=1e6
    with pytest.raises(ValueError):b.exchange(born,returned,target,p)
    assert b.checkpoint()==snapshot


def test_combined_thermal_history_and_exchange_restart():
    b,s,t,p,_=initialized()
    t=advance(b,t,4.,p)
    vol=surface_volumes(s)+[1e6,2e6]
    b.exchange([1e6,2e6],[0.,0.],vol,p)
    t=replace(t,mantle_temperature_k=float(b.temperatures(p)[0]))
    resumed=MaterialExchangeBudget.restore(json.loads(json.dumps(b.checkpoint())),b.options)
    x=advance(b,t,4.,p);y=advance(resumed,t,4.,p)
    assert asdict(x)==asdict(y)
    assert b.checkpoint()==resumed.checkpoint()
    d=b.diagnostics(p)
    exchange=max(d['surface_output_j'],d['advected_out_j'],d['orbital_input_j'])
    assert abs(d['energy_residual_j'])<2e-14*exchange


def test_two_box_time_and_tolerance_refinement():
    finals=[]
    for tol in (1.,.1,.01,.001):
        options=MaterialExchangeParameters(temperature_atol_k=tol,temperature_rtol=tol/1e4,
                                          exchange_rtol=tol/1e3,max_step_myr=20.,initial_step_myr=20.)
        b,s,t,p,_=initialized(options)
        # A finite hot addition creates a thermal transient, not a stress seed.
        v=surface_volumes(s);born=v*.1;b.exchange(born,[0.,0.],v+born,p)
        t=replace(t,mantle_temperature_k=float(b.temperatures(p)[0]))
        if not finals:
            masses=b.pool_masses();cp=p.mantle_heat_capacity_j_kg_k
            area=b.data['area_m2'];h=math.fsum(v+born)*1e9/area
            # Independent ODE reference: no exponential update or controller
            # from the implementation under test is used here.
            def rhs(time,temp):
                layer=convective_state(temp[0],5287.,7.12,p)[4]*1000.
                basal=p.thermal_conductivity_w_m_k*area/(layer-h/2)*(temp[0]-temp[1])
                surface=2*p.thermal_conductivity_w_m_k*area/h*(temp[1]-p.surface_temperature_k)
                rm=radiogenic_power_w(t.system_age_myr+time,masses[0],p)
                rc=radiogenic_power_w(t.system_age_myr+time,sum(masses[1:]),p)
                tide=tidal_heating_power_w(.00047,5287.,47.,5.,p)
                return SECONDS_PER_MYR*np.array([(rm+tide-basal)/(cp*masses[0]),
                                                 (rc+basal-surface)/(cp*sum(masses[1:]))])
            reference=solve_ivp(rhs,(0.,20.),b.temperatures(p),method='DOP853',rtol=1e-12,atol=1e-9)
            assert reference.success
            exact=reference.y[:,-1]
        advance(b,t,20.,p)
        finals.append(b.temperatures(p))
        assert abs(b.diagnostics(p)['energy_residual_j'])<1e15
    errors=np.max(abs(np.asarray(finals)-exact),axis=1)
    # Step controllers can select identical steps at adjacent tolerances.
    assert np.all(np.diff(errors)<=1e-10)
    assert errors[-1]<errors[0]/10
    assert errors[-1]<.02


def test_thermal_failure_does_not_mutate_inventory():
    b,s,t,p,_=initialized(MaterialExchangeParameters(max_trials=1,initial_step_myr=20.,max_step_myr=20.,
                                                    exchange_rtol=1e-15,temperature_atol_k=1e-10))
    snapshot=b.checkpoint()
    with pytest.raises(RuntimeError):advance(b,t,20.,p)
    assert b.checkpoint()==snapshot


@pytest.mark.parametrize('key', ['origin_hi','energy_c_j','surface_volumes_km3','next_step_myr'])
def test_corrupt_restart_cannot_silently_replace_inventory(key):
    b,_,_,_,_=initialized();saved=b.checkpoint()
    if key=='next_step_myr':saved[key]=float('nan')
    elif key=='origin_hi':saved[key][0][0]=float('nan')
    else:saved[key][0]=float('nan')
    with pytest.raises(ValueError):MaterialExchangeBudget.restore(saved,b.options)


def test_two_hour_terminal_interval_retains_small_energy_exchange_and_restart():
    b,s,t,p,_=initialized();t=advance(b,t,1.03,p)
    before=deepcopy(b.data);resumed=MaterialExchangeBudget.restore(b.checkpoint(),b.options)
    dt=2/(365.25*24*1e6);target=t.time_myr+dt
    x=advance(b,t,dt,p);y=advance(resumed,t,dt,p)
    assert x.time_myr==y.time_myr==target
    assert b.checkpoint()==resumed.checkpoint()
    with localcontext() as c:
        c.prec=70
        out=sum(map(Decimal,b.data['surface_output_j']))-sum(map(Decimal,before['surface_output_j']))
        temp=np.array([math.fsum(before['energy_m_j'])/(b.pool_masses()[0]*p.mantle_heat_capacity_j_kg_k),
                       math.fsum(before['energy_c_j'])/(sum(b.pool_masses()[1:])*p.mantle_heat_capacity_j_kg_k)])
        h=math.fsum(before['surface_volumes_km3'])*1e9/before['area_m2']
        flux=2*p.thermal_conductivity_w_m_k*before['area_m2']/h*(temp[1]-p.surface_temperature_k)
        exact=Decimal(flux)*Decimal(target-t.time_myr)*Decimal(SECONDS_PER_MYR)
        assert abs((out-exact)/exact)<Decimal('1e-10')
