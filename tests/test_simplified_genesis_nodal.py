"""Opt-in integration tests use a locally authenticated source, never fetch data."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.genesis_transfer_bridge import encoded
from tectonics.simplified_genesis_nodal import operators,material_mass,mechanical,birth_operator,capture
from tectonics.simplified_genesis_phase import promote


@pytest.fixture
def model():
    path=Path('/workspace/genesis-nodal-source-01.json')
    if not path.is_file():pytest.skip('Run local nodal_source import; no archive retrieval in tests')
    return SimplifiedGenesis.load(path)


def test_shared_response_and_boundary_conditions(model):
    from analysis.genesis_geological_handoff.nodal_controls import run
    assert run('/workspace/genesis-nodal-source-01.json')['status']=='PASS'


def test_coupled_energy_phase_and_exact_restart(model,tmp_path):
    initial_mass=model.mass.copy();model.advance(1.)
    L=model.state['last_ledger'];assert abs(L['whole_balance_j'])<=L['whole_allowance_j']
    assert L['momentum_relative']<1e-8 and L['phase']['added_latent_heat_j']==0
    model.save(tmp_path/'restart.json');again=SimplifiedGenesis.load(tmp_path/'restart.json')
    model.advance(1.);again.advance(1.)
    assert encoded(model.state)==encoded(again.state)
    np.testing.assert_array_equal(model.mass,initial_mass)


def test_birth_mass_capture_and_mixed_moment_blocks(model):
    # A controlled endpoint admission; no new material/latent inventory.
    m=model;cell=m.config['phase_cells'][0];i=cell['node'];T0=m.temperature(m.state['energy']);T=T0.copy()
    T[i]=m.config['solidus']-.001;before=deepcopy(m.state);after=deepcopy(before);after['time_myr']+=1e-6
    E=np.asarray(before['energy']).copy();E[i]=m.mass[i]*m.cp[i]*T[i]
    M=material_mass(m)
    _,report,_=promote(m.config,before,after,T0,T,E,m.material_inertia(),np.zeros((1,3)))
    assert report['new_load_bearing_cells']==1 and report['added_latent_heat_j']==0
    m.state=after;newM=material_mass(m);o=operators(m);_,local=birth_operator(m,cell);ids=o['dofs'][cell['face']]
    expected=M.copy();expected[np.ix_(ids,ids)]+=local
    np.testing.assert_array_equal(newM,expected)
    v,heat,error=capture(M,newM,np.asarray(before['nodal_velocity']))
    assert heat>=0 and error<1e-10
    m.state['nodal_velocity']=v.tolist()
    r=mechanical(m,T,np.asarray(m.state['damage']),T,31557600.,newM)
    assert len(r['memory'])==m.nf+1 and r['memory'][-1].shape==(6,)
    assert r['ledger']['balance_relative']<5e-10
    second=deepcopy(after)
    _,again,_=promote(m.config,after,second,T,T,E,m.material_inertia(),np.zeros((1,3)))
    assert again['new_load_bearing_cells']==0 and len(second['born_material'])==1


def test_failure_is_transactional_and_mature_export_refused(model):
    from tectonics.genesis_geological_thermal import eligibility
    before=encoded(model.state)
    with pytest.raises(ValueError,match='STEP_RANGE'):model.advance(26.)
    assert encoded(model.state)==before
    assert 'Shared nodal deformation and momentum require an explicit handoff map' in eligibility(model)['reasons']
