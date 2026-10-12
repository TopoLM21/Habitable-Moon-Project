import pytest
from tectonics.genesis_energy_ports import boundary_budget


def ledger():
    # Delta mechanics = Wext + Wthermal - Qmaxwell - drag - damage - numerical.
    return dict(external_work_j=10.,physical_drag_j=3.,damage_energy_change_j=-2.,
        numerical_kinetic_j=1.,numerical_memory_j=1.,kinetic_before_j=0.,kinetic_after_j=0.,
        elastic_before_j=100.,elastic_after_j=101.,physical_maxwell_j=7.,thermal_work_j=5.)


def test_domain_choice_changes_ports_not_conservation():
    for included in (False,True):
        r=boundary_budget(ledger(),2.,interface_in_control_volume=included)
        assert r['balance_residual_j']==0
        assert r['storage_increments']['interface_unresolved_j']==(3 if included else 0)
        assert r['net_boundary_input_j']==(10 if included else 7)


def test_internal_transfers_not_counted_twice():
    L=ledger();a=boundary_budget(L,2.);L.update(physical_maxwell_j=900.,thermal_work_j=898.)
    assert a==boundary_budget(L,2.)


def test_negative_external_work_is_allowed():
    L=ledger();L['external_work_j']=-10;L['elastic_after_j']=81
    assert boundary_budget(L,2.)['balance_residual_j']==0


@pytest.mark.parametrize('key,value',[('damage_energy_change_j',1.),('physical_drag_j',-1.),('numerical_memory_j',-1.),('external_work_j',float('nan')),('birth_kinetic_inflow_j',1.)])
def test_undeclared_sources_rejected(key,value):
    L=ledger();L[key]=value
    with pytest.raises(ValueError):boundary_budget(L,2.)
