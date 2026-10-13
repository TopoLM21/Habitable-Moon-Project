"""Native thermal/phase laws coupled to explicit angular Maxwell memory."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from tectonics.experimental_angular_coupled import AngularCoupled,mechanical,commit_damage,base_face_energy
from tectonics.simplified_genesis_nodal import material_mass
from tectonics.simplified_genesis_basal import assemble
from tectonics.experimental_edge_weakening import scale
from tectonics.genesis_transfer_bridge import encoded

SOURCE=Path('/workspace/genesis-source-angular-01/initial.json')


@pytest.fixture
def coupled():
    if not SOURCE.exists():pytest.skip('Local initialized source required')
    return AngularCoupled.from_source_checkpoint(SOURCE)


def test_coupled_fields_inventory_whole_balance_and_restart(coupled,tmp_path):
    a=coupled;m=a.model;mass=m.mass.copy();before=deepcopy(m.state)
    L=a.advance(.0625)
    assert m.state['energy']!=before['energy'] and m.state['damage']!=before['damage']
    assert m.state['orbit']['time_myr']==m.state['time_myr']>before['time_myr']
    np.testing.assert_array_equal(m.mass,mass)
    assert L['phase']['new_phase_solid_kg']>0 and L['phase']['added_latent_heat_j']==0
    assert abs(L['whole_balance_j'])<=L['whole_allowance_j']
    assert L['edge_damage_heat_j']==0 and L['edge_fracture_release_j']>0
    p=tmp_path/'coupled.json';a.save(p);b=AngularCoupled.load(p)
    a.advance(.0625);b.advance(.0625)
    assert encoded(a.model.state)==encoded(b.model.state)
    with pytest.raises(FileExistsError):a.save(p)


def test_node_heat_is_maxwell_minus_work_and_modulus_change_plus_drag(coupled):
    m=coupled.model;oldT=m.temperature(m.state['energy']);T=oldT.copy()
    angular=np.asarray(m.state['angular_memory']);angular[:,0,0,3]+=2e-5;angular[:,2,0,3]-=2e-5
    m.state['angular_memory']=angular.tolist();m.state['memory']=angular.mean(axis=1).tolist()
    T[m.solid_nodes]+=.001
    D=np.minimum(1,np.array(m.state['damage'])+.001)
    r=mechanical(m,T,D,oldT,.0625*31557600.,material_mass(m),assemble(m))
    L=r['ledger'];expected=L['maxwell']-L['thermal_work']-r['damage_energy_change']+L['drag']
    assert r['heat'].sum()==pytest.approx(expected,rel=1e-10,abs=1e3)
    assert r['damage_energy_change']<0
    assert r['damage_energy_change']==pytest.approx(base_face_energy(m)@(scale(m,D)-scale(m,m.state['damage'])),rel=1e-12)
    assert L['maxwell']>=0 and L['drag']>=0 and L['numerical_memory']>=0
    # Numerical dissipation is deliberately NOT deposited as heat.
    assert L['balance_relative']<5e-10


def test_full_angular_damage_release_not_just_mean_and_not_twice(coupled):
    m=coupled.model;e=np.array(m.state['angular_memory']);e[:,0,0,3]+=1e-4;e[:,1,0,3]-=1e-4
    m.state['angular_memory']=e.tolist();m.state['memory']=e.mean(axis=1).tolist()
    base=base_face_energy(m);D=np.array(m.state['damage']);old=base@scale(m,D)
    inventory=encoded(dict(H=m.state['energy'],mass=m.config['mass']))
    edge=np.full(len(m.mesh.shared_edges),.8);result=commit_damage(m,edge)
    assert old==pytest.approx(result['elastic_before_j'],rel=1e-13)
    assert result['elastic_before_j']-result['elastic_after_j']==pytest.approx(result['release_j'],rel=1e-13)
    assert commit_damage(m,edge)['release_j']==0
    assert encoded(dict(H=m.state['energy'],mass=m.config['mass']))==inventory


def test_controlled_native_birth_zero_memory_capture_and_next_step(coupled,tmp_path):
    a=coupled;m=a.model;cell=m.config['phase_cells'][0];i=cell['node']
    # Explicit controlled thermal initial condition, not evidence of a real birth.
    m.state['energy'][i]=float(m.mass[i]*m.cp[i]*(m.config['solidus']-1.))
    mass=m.mass.copy();before=deepcopy(m.state);M=material_mass(m)
    front_before=assemble(m)['geometry']['fronts'][cell['face']]
    L=a.advance(.0625);born=m.state['born_material']
    assert len(born)==1 and born[0]['cell_id']==cell['id']
    np.testing.assert_array_equal(m.state['angular_born_memory'][cell['id']],np.zeros((3,6)))
    assert born[0]['birth_time_myr']==m.state['time_myr']
    assert assemble(m)['geometry']['fronts'][cell['face']]==cell['q_hi']>front_before
    assert L['phase']['added_latent_heat_j']==0 and L['phase']['capture_nodal_momentum_relative']<1e-10
    assert L['phase']['capture_heat_j']>=0 and abs(L['whole_balance_j'])<=L['whole_allowance_j']
    assert np.linalg.norm(material_mass(m)-M)>0
    np.testing.assert_array_equal(m.mass,mass)
    p=tmp_path/'born.json';a.save(p);b=AngularCoupled.load(p)
    a.advance(.0625);b.advance(.0625)
    assert encoded(a.model.state)==encoded(b.model.state)
    assert len(m.state['born_material'])==1
    assert np.linalg.norm(m.state['angular_born_memory'][cell['id']])>0
    assert m.state['last_ledger']['phase']['new_load_bearing_cells']==0


def test_guard_and_transactional_failure(coupled):
    m=coupled.model;before=encoded(m.state)
    for years in (0.,-1.,26.,float('nan')):
        with pytest.raises(ValueError,match='STEP_RANGE'):coupled.advance(years)
    with pytest.raises(ValueError,match='ANGULAR_REQUIRES_COUPLED_DRIVER'):m.advance(.0625)
    assert encoded(m.state)==before
    with pytest.raises(ValueError,match='UNALLOCATED_THERMAL_HISTORY'):
        AngularCoupled.from_source_checkpoint('/workspace/genesis-source-angular-01/fine_final.json')
    # Force a normal native physical guard; the complete angular state rolls back.
    m.config['max_rotation_rad']=1e-30
    with pytest.raises(ValueError,match='REFERENCE_CHART'):coupled.advance(.0625)
    assert encoded(m.state)==before
