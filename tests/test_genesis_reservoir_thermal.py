from dataclasses import asdict,replace
import json,math
import pytest
from tectonics.genesis import GenesisParameters,melt_fraction
from tectonics.genesis_reservoir_thermal import Reservoir,CoupledThermalState,specific_h,temperature,conduct,mechanical_exchange,total,validate,save,load,migrate


def sample():
    p=GenesisParameters()
    cells=(Reservoir('deep_mantle','silicate',10.,10*specific_h(1700.,'silicate',p)),
      Reservoir('column:0:0','silicate',1.,specific_h(1500.,'silicate',p)),
      Reservoir('surface_water','water',10.,10*specific_h(600.,'water',p)),
      Reservoir('elastic:column:0:0','elastic',0.,100000.))
    return CoupledThermalState(0.,cells,json.dumps(asdict(p)),json.dumps({'silicate_mass_kg':11.}),math.fsum(c.energy_j for c in cells))


def test_closed_coupled_relaxation_conduction_and_phase_change():
    s=sample();p=GenesisParameters();start=s;old_phase=melt_fraction(temperature(s.cells[1],p),p)
    for _ in range(20):
        U=s.cells[-1].energy_j
        s=mechanical_exchange(s,kind='maxwell',mechanical_id='elastic:column:0:0',recipient='column:0:0',energy_j=U*(-math.expm1(-.02)))
        s,journal=conduct(s,[('deep_mantle','column:0:0',100.),('column:0:0','surface_water',1000.)],.1)
        assert all(r['donor_entry_j']+r['recipient_entry_j']==0 for r in journal)
    assert total(s)==pytest.approx(total(start),rel=1e-14)
    assert melt_fraction(temperature(s.cells[1],p),p)<old_phase
    assert temperature(s.cells[2],p)>600
    assert temperature(s.cells[0],p)<1700
    assert [c.mass_kg for c in s.cells]==[c.mass_kg for c in start.cells]
    assert s.historical_opening_balance is None


def test_latent_heat_is_counted_once_across_solidification():
    p=GenesisParameters();s=sample();c=s.cells[1]
    assert specific_h(2000.,'silicate',p)-specific_h(1400.,'silicate',p)==pytest.approx(600*p.silicate_heat_capacity_j_kg_k+p.silicate_latent_heat_j_kg)


def test_numerical_loss_never_heats():
    s=sample();r=mechanical_exchange(s,kind='numerical',mechanical_id='elastic:column:0:0',energy_j=123.)
    assert r.cells[:-1]==s.cells[:-1]
    assert total(r)+r.numerical_sink_j==pytest.approx(total(s))
    with pytest.raises(ValueError):mechanical_exchange(s,kind='numerical',mechanical_id='elastic:column:0:0',recipient='column:0:0',energy_j=123.)


@pytest.mark.parametrize('kind',['drag','damage','thermal_work'])
def test_unknown_coupling_not_guessed(kind):
    with pytest.raises(ValueError,match='policy'):mechanical_exchange(sample(),kind=kind,mechanical_id='elastic:column:0:0',energy_j=1.)


def test_overdraw_cfl_and_duplicate_edges_are_transactional():
    s=sample()
    with pytest.raises(ValueError):mechanical_exchange(s,kind='maxwell',mechanical_id='elastic:column:0:0',recipient='column:0:0',energy_j=1e8)
    with pytest.raises(ValueError,match='CFL'):conduct(s,[('deep_mantle','column:0:0',100.)],100.)
    with pytest.raises(ValueError,match='edge'):conduct(s,[('deep_mantle','column:0:0',100.),('column:0:0','deep_mantle',100.)],.1)
    assert s==sample()


def test_versioned_restart_and_exclusive_save(tmp_path):
    s=sample();p=tmp_path/'state';save(s,p);r=load(p);assert r==s
    q=tmp_path/'again';save(r,q);assert q.read_bytes()==p.read_bytes()
    with pytest.raises(FileExistsError):save(s,p)
    packet=json.loads(p.read_bytes());packet['state']['cells'][0]['energy_j']+=1;p.write_text(json.dumps(packet))
    with pytest.raises(ValueError,match='Seal'):load(p)


def test_unknown_opening_balance_and_mass_cannot_change():
    with pytest.raises(ValueError):validate(replace(sample(),historical_opening_balance=0.))
    s=sample();cells=list(s.cells);cells[0]=replace(cells[0],mass_kg=11.)
    with pytest.raises(ValueError,match='mass'):validate(replace(s,cells=tuple(cells)))


def test_migration_authentication():
    with pytest.raises(ValueError,match='SHA'):migrate(b'{}','0'*64)


def test_migrated_restart_requires_original_history(tmp_path):
    import hashlib
    s=sample();raw=b'preserved original bytes';legacy=tmp_path/'legacy';legacy.write_bytes(raw)
    s=replace(s,origin_json=json.dumps({'legacy_raw_sha256':hashlib.sha256(raw).hexdigest()}))
    p=tmp_path/'migrated';save(s,p)
    with pytest.raises(ValueError,match='legacy'):load(p)
    assert load(p,legacy_path=legacy)==s
    legacy.write_bytes(b'different')
    with pytest.raises(ValueError,match='legacy'):load(p,legacy_path=legacy)


def test_maxwell_cannot_be_routed_to_arbitrary_column():
    s=sample();cells=tuple(replace(c,identity='column:1:0') if c.identity=='column:0:0' else c for c in s.cells)
    with pytest.raises(ValueError,match='Located'):mechanical_exchange(replace(s,cells=cells),kind='maxwell',mechanical_id='elastic:column:0:0',recipient='column:1:0',energy_j=10.)
