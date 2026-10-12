import json
import numpy as np
from analysis.genesis_aggregate_coupled.core import step
from tectonics.genesis_transfer_bridge import encoded


def fixture():
    return np.eye(3)*2,np.eye(3)*.3,np.array([.2,-.3,.1]),np.array([.05,-.02,.01]),[dict(H=np.diag([4.,6.]),L=np.array([[1.,0.,.2],[0.,1.,-.3]]),R=np.eye(2)*.2,old=np.array([.1,-.2]),theta=np.array([.01,.02]),dofs=np.arange(3))]


def test_reduced_step_balances_energy_and_nodal_impulse():
    M,D,F,v,b=fixture();r=step(M,D,F,v,.1,b)
    assert r['ledger']['momentum_relative']<1e-14
    assert r['ledger']['balance_relative']<1e-14
    assert r['ledger']['maxwell']>0
    # Maxwell heating and reversible thermal work are equal/opposite entries.
    heat=r['ledger']['maxwell']-r['ledger']['thermal_work']
    assert abs(r['ledger']['elastic_after']-r['ledger']['elastic_before']+heat-r['ledger']['internal_work']+r['ledger']['numerical_memory'])<1e-14


def test_restart_reproduces_second_step_and_frozen_reserve_not_heat(tmp_path):
    M,D,F,v,b=fixture();r=step(M,D,F,v,.1,b);b[0]['old']=r['memory'][0]
    saved=dict(velocity=r['velocity'].tolist(),memory=r['memory'].tolist(),unresolved_reserve_j=7.4)
    p=tmp_path/'state.json';p.write_bytes(encoded(saved));read=json.loads(p.read_bytes())
    direct=step(M,D,F,r['velocity'],.1,b)
    b[0]['old']=np.array(read['memory'][0]);restart=step(M,D,F,np.array(read['velocity']),.1,b)
    np.testing.assert_array_equal(direct['memory'],restart['memory']);np.testing.assert_array_equal(direct['velocity'],restart['velocity'])
    assert read['unresolved_reserve_j']==7.4
    assert 'unresolved_reserve_j' not in restart['ledger']


def test_zero_thermal_uniform_memory_relaxes_without_artificial_work():
    M,D,F,v,b=fixture();b[0]['L'][:]=0;b[0]['theta'][:]=0
    r=step(M,D*0,F*0,v*0,1.,b)
    np.testing.assert_allclose(r['memory'][0],b[0]['old']/1.2)
    assert r['ledger']['thermal_work']==0 and r['ledger']['external_work']==0
    assert r['ledger']['balance_relative']<1e-14


def test_coupled_feedback_conserves_total_heat_and_mechanics():
    from tectonics.genesis_implicit_enthalpy import solve
    M,D,F,v,b=fixture();mass=np.array([2.,3.]);cp=np.ones(2);E0=np.array([1000.,1800.]);dt=.1
    conduction=solve(mass,E0/mass,cp,np.zeros(2),np.full(2,1400.),np.full(2,2000.),np.zeros(2),[(0,1,.4)],dt)
    base=mass*conduction['specific_h'];T=base/mass;old=b[0]['old'].copy()
    for _ in range(20):
        b[0]['theta']=np.array([1e-4*(T[0]-E0[0]/mass[0]),0.])
        r=step(M,D,F,v,dt,b);E=base.copy();E[0]+=r['ledger']['maxwell']-r['ledger']['thermal_work'];nextT=E/mass
        if np.max(np.abs(T-nextT))<1e-12:break
        T=nextT
    else:raise AssertionError('coupling did not converge')
    L=r['ledger']
    total=(E.sum()-E0.sum()+L['kinetic_after']-L['kinetic_before']+L['elastic_after']-L['elastic_before']-L['external_work']+L['drag']+L['numerical_kinetic']+L['numerical_memory'])
    assert abs(total)<1e-11
    np.testing.assert_array_equal(b[0]['old'],old)


def test_series_reduction_preserves_linear_fourier_flux_and_total_resistance():
    from analysis.genesis_aggregate_coupled.adapter import series_column_edges
    # Four fine layers with unequal widths; two merged cells, same source k/rho.
    widths=np.array([1.,2.,3.,4.]);rho=2.;area=3.;k=5.
    mass=np.array([sum(widths[:2]),sum(widths[2:]),1.,1.])*rho*area
    edges=series_column_edges([0,1],mass,rho,area,k,3,2)
    # Exact linear profile sampled at the actual aggregate mass centres.
    T=np.array([10.-1.5,10.-6.5,0.,10.])
    flux=np.array([G*(T[i]-T[j]) for i,j,G in edges])
    np.testing.assert_allclose(flux,k*area,rtol=1e-15)
    assert abs(sum(1/G for _,_,G in edges)-sum(widths)/(k*area))<1e-15


def test_series_reduction_with_single_group_keeps_both_boundary_resistances():
    from analysis.genesis_aggregate_coupled.adapter import series_column_edges
    e=series_column_edges([0],np.array([80.,1.,1.]),1.,1.,4.,2,1)
    assert e==[(2,0,.1),(0,1,.1)]
    assert sum(1/G for _,_,G in e)==20.
