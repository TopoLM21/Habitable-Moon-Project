from copy import deepcopy
import pytest
from analysis.genesis_energy_inventory.sidecar import exchanges,seal,validate,write_new


def ledger():
    return dict(physical_drag_j=2.,physical_maxwell_j=5.,numerical_kinetic_j=11.,numerical_memory_j=13.,thermal_work_j=3.,damage_energy_change_j=-7.,birth_kinetic_inflow_j=17.,external_work_j=-19.)


def fixture():
    return seal(dict(applied_to_thermal_state=False,historical_opening_balance=None,exchanges=exchanges(ledger()),units={'energy':'J','time':'Myr'}))


def test_signs_and_separate_sinks():
    r=exchanges(ledger())
    assert r['physical_dissipation_unassigned']['value_j']==7
    assert r['numerical_dissipation_diagnostic']['value_j']==24
    assert r['thermomechanical_counterentry_pending']['value_j']==-3
    assert r['damage_destination_pending']['value_j']==7
    assert r['birth_donor_debit_pending']['value_j']==-17
    assert r['traction_donor_debit_pending']['value_j']==19
    assert r['physical_dissipation_unassigned']['recipient']=='unassigned'


@pytest.mark.parametrize('key',list(ledger()))
def test_missing_fields_never_become_zero(key):
    value=ledger();del value[key]
    with pytest.raises(KeyError):exchanges(value)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),True])
def test_invalid_numbers_refused(value):
    l=ledger();l['thermal_work_j']=value
    with pytest.raises(ValueError):exchanges(l)


@pytest.mark.parametrize('mutation',['apply_heat','zero_history','assign_drag','mix_numerical','change_units'])
def test_resealed_policy_changes_refused(mutation):
    expected=fixture();value=deepcopy(expected);del value['sha256']
    if mutation=='apply_heat':value['applied_to_thermal_state']=True
    if mutation=='zero_history':value['historical_opening_balance']=0
    if mutation=='assign_drag':value['exchanges']['physical_dissipation_unassigned']['recipient']='crust'
    if mutation=='mix_numerical':value['exchanges']['physical_dissipation_unassigned']['value_j']+=24
    if mutation=='change_units':value['units']['energy']='W'
    with pytest.raises(ValueError):validate(seal(value),expected)


def test_write_readback_and_no_overwrite(tmp_path):
    p=tmp_path/'sidecar.json';value=fixture();write_new(p,value);before=p.read_bytes()
    with pytest.raises(FileExistsError):write_new(p,value)
    assert p.read_bytes()==before
