"""Independent mechanical identities, preserved inventory, and restart safety."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from analysis.genesis_aggregate_coupled.core import step
from analysis.genesis_geological_handoff.equilibrated_start import (
    CompatibleProjection, equilibrate, protected_state, point_stress,
)
from analysis.genesis_geological_handoff.spherical_coupled import SphericalCoupled
from tectonics.genesis_transfer_bridge import encoded
from tectonics.simplified_genesis_nodal import rate

START = Path('/workspace/spherical-coupled-preflight-initial.json')


def test_compatible_projection_gauge_energy_and_self_stress():
    L = np.array([[[-1.,1.,0.],[0.,-1.,1.],[-1.,0.,1.]]])
    H = np.array([np.diag([2.,3.,5.])]);M=np.diag([1.,2.,3.]);J=np.ones((3,1))
    op = CompatibleProjection(L,H,np.array([[0,1,2]]),M,J)
    displacement=np.array([.2,-.1,.4]);displacement-=(J.T@M@displacement).item()/(J.T@M@J).item()
    expected=op.strain(displacement)
    actual=op.solve(op.force(expected))
    np.testing.assert_allclose(actual,displacement,atol=2e-15)
    assert np.linalg.norm(J.T@M@actual)<1e-14
    # A genuine self-equilibrated component survives the compatible correction.
    self_stress=np.array([[-.5,-1/3,1/5]])
    assert np.linalg.norm(op.force(self_stress))<1e-15
    initial=expected+self_stress;correction=op.solve(-op.force(initial))
    final=initial+op.strain(correction)
    np.testing.assert_allclose(final,self_stress,atol=1e-15)
    U0=.5*np.einsum('qi,qij,qj->',initial,H,initial)
    U1=.5*np.einsum('qi,qij,qj->',final,H,final)
    assert U1-U0==pytest.approx(.5*(op.force(initial)+op.force(final))@correction,abs=1e-14)


@pytest.mark.parametrize('dt',[.01,1.,100.])
def test_uniform_rate_stationary_force_allows_creep_and_self_memory_decay(dt):
    L=np.array([[-1.,1.,0.],[0.,-1.,1.],[-1.,0.,1.]])
    H=np.diag([2.,3.,5.]);R=.2;M=np.diag([1.,2.,3.]);D=np.diag([2.,4.,7.])
    F=np.array([.2,.4,-.3]);v=np.linalg.solve(D+L.T@H@L/R,F)
    self_stress=np.array([-.5,-1/3,1/5]);e=L@v/R+self_stress
    blocks=[dict(L=L,H=H,R=np.eye(3)*R,old=e,theta=np.zeros(3),dofs=np.arange(3))]
    r=step(M,D,F,v,dt,blocks)
    np.testing.assert_allclose(r['velocity'],v,rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(L.T@H@r['memory'][0]+D@r['velocity'],F,atol=2e-14)
    np.testing.assert_allclose(r['memory'][0]-L@v/R,self_stress/(1+dt*R),atol=2e-15)
    assert r['ledger']['balance_relative']<1e-12
    assert r['ledger']['maxwell']>0 and r['ledger']['drag']>0


@pytest.fixture(scope='module')
def prepared():
    if not START.exists():pytest.skip('Local full spherical history required')
    a=SphericalCoupled.load(START);before=deepcopy(a.model.state);protected=protected_state(a)
    ledger=equilibrate(a,allow_weakened_scenario=True)
    return a,before,protected,ledger


def test_real_inventory_compatibility_force_torque_and_signed_energy(prepared):
    a,before,protected,ledger=prepared
    assert protected_state(a)==protected
    for key in ('relative','force_relative','torque_relative'):
        assert ledger['final_balance'][key]<=1e-8
    assert ledger['force_derivative_relative']<=1e-8
    du=np.array(a.model.state['nodal_displacement'])-before['nodal_displacement']
    delta=np.array(a.model.state['spherical_memory'])-before['spherical_memory']
    np.testing.assert_allclose(delta,np.einsum('qij,qj->qi',a.baseL,du[a.ids[a.f]]),atol=2e-18)
    assert ledger['thermal_heat_j']==0 and ledger['time_advanced_years']==0
    assert ledger['elastic_identity_relative']<1e-10
    assert ledger['elastic_after_j']>0 and np.linalg.norm(point_stress(a))>0
    assert ledger['signed_mechanical_change_j']!=0
    np.testing.assert_array_equal(a.model.state['damage'],before['damage'])


def test_real_restart_and_no_accidental_second_projection(prepared,tmp_path):
    a,_,_,_=prepared;path=tmp_path/'start.json';a.save(path);b=SphericalCoupled.load(path)
    assert encoded(a.model.state)==encoded(b.model.state)
    assert encoded(a.model.config)==encoded(b.model.config)
    assert encoded(a.origin)==encoded(b.origin)
    with pytest.raises(ValueError,match='ALREADY_EQUILIBRATED'):equilibrate(b,allow_weakened_scenario=True)
    with pytest.raises(FileExistsError):a.save(path)


def test_real_frozen_constitutive_tangent_has_no_fast_force_change(prepared):
    # Independent force assembly on an explicit Euler tangent displacement in
    # memory space. This is a diagnostic only, not a thermal/evolution step.
    a=deepcopy(prepared[0]);m=a.model;L,_,f,nodes,e=a.arrays()
    v=np.array(m.state['nodal_velocity']);R=rate(m,m.temperature(m.state['energy'])[nodes])
    derivative=np.einsum('qij,qj->qi',L,v[a.ids[f]])-R[:,None]*e
    initial=a.internal_force()
    m.state['spherical_memory']=(e+1e4*derivative).tolist()
    assert np.linalg.norm(derivative)>0  # self-equilibrated memory still relaxes
    assert np.linalg.norm(a.internal_force()-initial)/np.linalg.norm(initial)<1e-10


def test_failed_projection_is_transactional(prepared):
    a,before,_,_=prepared;b=deepcopy(a);b.model.state=deepcopy(before);b.origin.pop('equilibrated_start')
    state=encoded(b.model.state);origin=encoded(b.origin)
    with pytest.raises(ValueError,match='EQUILIBRATION_NOT_CONVERGED'):
        equilibrate(b,max_iterations=1,allow_weakened_scenario=True)
    assert encoded(b.model.state)==state and encoded(b.origin)==origin
