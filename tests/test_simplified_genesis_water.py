from copy import deepcopy
from dataclasses import replace,asdict
import numpy as np
import pytest
from tectonics.genesis import GenesisParameters
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.genesis_transfer_bridge import encoded
from tectonics.mesh import build_icosphere
from analysis.genesis_geological_handoff.thermal_experiment import thermal_source
from analysis.simplified_genesis_validation.synthetic import synthetic


@pytest.fixture(scope='module')
def water_source():
    m,_=thermal_source();c,s=deepcopy(m.config),deepcopy(m.state)
    p=replace(GenesisParameters(**c['thermal_parameters']),water_volume_km3=GenesisParameters().water_volume_km3)
    c['thermal_parameters']=asdict(p);c['mass'][-1]=p.water_column_kg_m2*p.area_m2
    s['energy'][-1]=c['mass'][-1]*(p.water_heat_capacity_j_kg_k*647.54+p.water_latent_heat_j_kg)
    return SimplifiedGenesis(c,s)


def test_opt_in_preserves_source_and_default(water_source):
    original=encoded(water_source.state);config=encoded(water_source.config)
    m=water_source.with_condensation()
    assert encoded(m.state)==original and encoded(water_source.config)==config
    np.testing.assert_allclose(m.temperature(m.state['energy']),water_source.temperature(water_source.state['energy']),atol=1e-10,rtol=0)
    assert 'water_eos' not in water_source.config
    old=deepcopy(water_source)
    with pytest.raises(ValueError,match='UNSUPPORTED_WATER_PHASE'):old.advance(25.)
    assert encoded(old.state)==original


def test_condensation_energy_mass_and_restart(water_source,tmp_path):
    m=water_source.with_condensation();mass=m.mass.copy();m.advance(25.)
    first=m.state['last_ledger']['water']
    assert first['temperature_after_k']<647.096
    assert first['liquid_mass_kg']>0 and first['latent_inventory_change_j']<0
    assert first['added_latent_heat_j']==0
    assert abs(first['residual_j'])<=first['roundoff_allowance_j']
    assert first['total_mass_kg']==pytest.approx(first['vapor_mass_kg']+first['liquid_mass_kg'])
    m.save(tmp_path/'restart.json');again=SimplifiedGenesis.load(tmp_path/'restart.json')
    m.advance(25.);again.advance(25.)
    assert encoded(m.state)==encoded(again.state)
    assert np.array_equal(m.mass,mass)
    assert m.state['last_ledger']['water']['liquid_mass_kg']>first['liquid_mass_kg']
    assert abs(m.state['last_ledger']['whole_balance_j'])<=m.state['last_ledger']['whole_allowance_j']


def test_uniform_persistent_damage_cannot_create_arbitrary_plates():
    m=synthetic(build_icosphere(1));m.state['damage']=[.9]*m.nf
    m.state['candidate_age']=[.05]*m.nf;m.state['memory']=np.zeros((m.nf,2,6)).tolist()
    m.advance(25.)
    assert min(m.state['damage'])>=m.faults['activation_damage']
    assert min(m.state['candidate_age'])>=m.faults['activation_persistence_myr']
    assert len(m.state['omega'])==1 and not m.state['events']


def test_staggered_persistence_in_diffuse_damage_is_not_a_belt(water_source):
    from tectonics.genesis_fault_law import WeakPlaneParameters
    c=deepcopy(water_source.config)
    c['faults']['activation_persistence_myr']=WeakPlaneParameters().activation_persistence_myr
    m=SimplifiedGenesis(c,water_source.state).with_condensation()
    m.state['owner']=[0]*m.nf;m.state['omega']=[[0.,0.,0.]]
    m.state['rotation']=[np.eye(3).tolist()];m.state['events']=[]
    m.state['consumed']=[False]*m.nf;m.state['damage']=[.9]*m.nf
    weak=abs(m.mesh.centroids[:,0])<.12
    m.state['candidate_age']=np.where(weak,.05,0.).tolist()
    m.state['memory']=np.zeros((m.nf,2,6)).tolist()
    before=encoded(m.state)
    with pytest.raises(ValueError,match='DIFFUSE_DAMAGE_NOT_A_LOCALIZED_BELT'):
        m.advance(25.)
    assert encoded(m.state)==before


@pytest.mark.parametrize('uniform',[False,True])
def test_localization_admission_closed_belt_and_uniform_damage(water_source,uniform):
    from tectonics.genesis_starter_topology import select_starter_cut
    from tectonics.plates import Plate,PlateSystem
    m=water_source.with_condensation();owner=np.zeros(m.nf,dtype=int)
    weak=abs(m.mesh.centroids[:,0])<.12
    damage=np.full(m.nf,.9) if uniform else np.where(weak,.9,.04)
    system=PlateSystem(owner,(Plate(0,0,np.array([0.,0.,1.]),0.),))
    cut=select_starter_cut(m.mesh,system,weak,damage,**m.config['selector'])
    assert cut is not None and np.all(weak[cut])
    if uniform:
        with pytest.raises(ValueError,match='DIFFUSE_DAMAGE_NOT_A_LOCALIZED_BELT'):
            m.validate_cut_localization(owner,damage,cut)
    else:
        m.validate_cut_localization(owner,damage,cut)
