"""Mapped state and native thermal coupling contracts, without new laws."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from tectonics.experimental_edge_weakening import scale
from tectonics.genesis_transfer_bridge import encoded
from analysis.genesis_geological_handoff.spherical_coupled import SphericalCoupled
START=Path('/workspace/spherical-coupled-preflight-initial.json')


@pytest.fixture(scope='module')
def model():
    if not START.exists():pytest.skip('Local transferred source required')
    return SphericalCoupled.load(START)


def test_mapping_inventory_energy_momentum_and_explicit_price(model):
    x=model.origin['mapping']
    assert abs(x['mass_relative'])<1e-15 and abs(x['enthalpy_relative'])<1e-15
    assert abs(x['elastic_relative_change'])<1e-12 and x['restricted_momentum_relative']<1e-12
    assert x['mapping_heat_j']==0 and x['kinetic_after_j']!=x['kinetic_before_j']
    assert x['internal_force_relative']>0
    before=encoded(model.model.state)
    with pytest.raises(ValueError,match='SPHERICAL_REQUIRES_GEOMETRY_PROVIDER'):model.model.advance(.015625)
    assert encoded(model.model.state)==before


def test_thermoelastic_work_and_heat_contract(model):
    a=deepcopy(model);m=a.model;oldT=m.temperature(m.state['energy']);T=oldT.copy();T[m.solid_nodes]+=.001
    D=np.minimum(1,np.array(m.state['damage'])+.0001)
    r=a.mechanical(m,T,D,oldT,.001*31557600.,a.material_mass(),a.basal());L=r['ledger']
    expected=L['maxwell']-L['thermal_work']-r['damage_energy_change']+r['basal_budget']['heat_j']
    assert r['heat'].sum()==pytest.approx(expected,rel=1e-12,abs=1e3)
    assert L['thermal_work']!=0 and L['maxwell']>0 and r['damage_energy_change']<0
    assert L['numerical_memory']>0
    assert L['balance_relative']<5e-9


def test_damage_release_uses_full_memory_without_heat(model):
    a=deepcopy(model);m=a.model;inventory=encoded(dict(mass=m.mass.tolist(),H=m.state['energy']))
    before=a.base_energy()@scale(m,m.state['damage']);r=a.commit(m,np.full(len(m.mesh.shared_edges),.2))
    assert r['release_j']>=0 and r['elastic_before_j']==pytest.approx(before,rel=1e-13)
    assert r['elastic_after_j']+r['release_j']==pytest.approx(before,rel=1e-13)
    assert a.commit(m,np.full(len(m.mesh.shared_edges),.2))['release_j']==0
    assert encoded(dict(mass=m.mass.tolist(),H=m.state['energy']))==inventory


def test_factored_drag_and_refinement_preserve_linear_solution():
    from analysis.genesis_aggregate_coupled.core import step
    M=np.diag([2.,3.]);D=1e6*np.ones((2,2));F=np.array([2.,-3.]);dt=.25
    block=dict(H=np.eye(2),L=np.eye(2),R=np.zeros((2,2)),old=np.zeros(2),theta=np.zeros(2),dofs=np.arange(2))
    diagonal=np.array([2.,3.],np.longdouble)/dt+dt
    inverse=1/diagonal;beta=np.longdouble(1e6)
    expected=inverse*F-inverse*(beta*np.sum(inverse*F)/(1+beta*inverse.sum()))
    r=step(M,D,F,np.zeros(2),dt,[block],refinement_steps=3,drag_action=lambda v:1e6*np.full(2,v.sum()))
    np.testing.assert_allclose(r['velocity'],expected,rtol=1e-10,atol=1e-12)
    assert r['ledger']['drag']>=0 and r['ledger']['balance_relative']<1e-9
