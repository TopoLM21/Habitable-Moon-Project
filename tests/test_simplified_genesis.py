from copy import deepcopy
import numpy as np
import pytest
from tectonics.mesh import build_icosphere
from tectonics.simplified_genesis import SimplifiedGenesis,VERSION,rotational_step
from tectonics.genesis_transfer_bridge import encoded


from analysis.simplified_genesis_validation.synthetic import synthetic


def test_complete_synthetic_evolution_birth_motion_restart(tmp_path):
    m=synthetic();initial=deepcopy(m.state);m.advance(25.)
    assert m.state['events'] and len(m.state['omega'])==2
    assert m.state['events'][0]['angular_momentum_relative']<1e-12
    assert np.max(m.state['damage'])>=.65 and initial['damage']==[0.]*m.nf
    assert m.state['passive_reserve_j']==12345.
    m.save(tmp_path/'one.json');restored=SimplifiedGenesis.load(tmp_path/'one.json')
    m.advance(25.);restored.advance(25.)
    assert encoded(m.state)==encoded(restored.state)
    assert np.linalg.norm(np.diff(m.state['omega'],axis=0))>1e-16
    assert np.linalg.norm(np.array(m.state['rotation'][0])-m.state['rotation'][1])>1e-8
    assert np.min(m.temperature(m.state['energy']))<1000.
    assert abs(m.state['last_ledger']['whole_balance_j'])<=m.state['last_ledger']['whole_allowance_j']
    assert not m.mature_compatibility()['direct_native_mature']


def test_no_load_does_not_invent_plates_and_failure_is_transactional():
    m=synthetic();m.state['memory']=np.zeros((m.nf,2,6)).tolist();m.advance(25.)
    assert len(m.state['omega'])==1 and m.state['events']==[]
    before=encoded(m.state)
    with pytest.raises(ValueError,match='STEP_RANGE'):m.advance(26.)
    assert before==encoded(m.state)


def test_torque_momentum_and_energy_balance():
    I=np.diag([2.,3.,4.]);D=np.diag([.1,.2,.3]);tau=np.array([1.,2.,3.]);old=np.array([.3,-.2,.1]);dt=.5
    new,l=rotational_step(I,D,tau,old,dt)
    np.testing.assert_allclose(I@(new-old),dt*(tau-D@new),rtol=1e-14,atol=1e-14)
    assert abs(l['rotation_balance'])<1e-14 and l['drag']>0 and l['numerical_rotation']>0


def test_force_reaction_closes_constrained_linear_balance():
    from tectonics.simplified_genesis import constrained_force_reaction
    A=np.array([[1.,2.,0.],[0.,1.,3.],[2.,0.,1.]])
    force=np.array([4.,5.,6.]);D=np.eye(3);old=np.array([.1,.2,.3]);new=np.array([.3,.1,.4]);dt=.25
    reaction=constrained_force_reaction(A,force,D,old,new,dt)
    np.testing.assert_allclose(A@(new-old)/dt,force-D@new+reaction,atol=2e-15)


def test_healing_draws_thermal_energy_and_reserve_is_not_heat():
    m=synthetic();m.state['damage']=np.full(m.nf,.2).tolist()
    # Small finite prestress, below the activation threshold, pays healing.
    m.state['memory']=(np.ones((m.nf,2,6))*1e-5).tolist()
    m.advance(25.)
    assert m.state['last_ledger']['damage_energy_change']>0
    assert m.state['passive_reserve_j']==12345.
    assert m.state['last_ledger']['moment_balance_relative']<1e-10


def test_contact_limit_failure_preserves_state():
    m=synthetic();m.config['max_rotation_rad']=1e-30
    m.config['face_torque']=np.full((m.nf,3),1e18).tolist();m.face_torque=np.array(m.config['face_torque'])
    before=encoded(m.state)
    with pytest.raises(ValueError,match='FINITE_CONTACT_REMESH_LIMIT'):m.advance(25.)
    assert encoded(m.state)==before
