from types import SimpleNamespace
import numpy as np
import pytest
from analysis.genesis_coupled_split_control.thermal import phase_guard


def test_phase_guard_retains_solid_and_partial_unowned():
    phase_guard([{'material_id':1},{'material_id':None}],np.array([1300.,1500.,1700.]),SimpleNamespace(solidus_k=1400.),np.array([1300.,1600.,1700.]))


@pytest.mark.parametrize('temps,match', [([1400.,1500.,1700.],'OLD_MATERIAL'),([1300.,1400.,1700.],'NEW_SOLID'),([1300.,1500.,600.],'WATER_PHASE')])
def test_unsupported_events_stop(temps,match):
    with pytest.raises(ValueError,match=match):phase_guard([{'material_id':1},{'material_id':None,'face':0,'layer':0}],np.array(temps),SimpleNamespace(solidus_k=1400.),np.array([1300.,1600.,1700.]))


def test_healing_cannot_be_hidden_by_damage_on_another_face():
    from analysis.genesis_coupled_split_control.mechanical import healing_guard
    with pytest.raises(ValueError,match='UNFUNDED_LOCAL_HEALING'):
        healing_guard([.1,.1],[.09,.2])
    healing_guard([.1,.1],[.1,.11])


@pytest.mark.parametrize('fraction',[0.,1e-6])
def test_damage_delta_resolves_zero_beside_large_elastic_energy(fraction):
    from analysis.genesis_coupled_split_control.audit import damage_delta
    triangle=np.eye(3)*1e6;mass=3e20;rho=3000.;cohort={'q_lo':0.,'live_q_hi':1e-3}
    old=np.zeros((3,6));old[:,0]=.1;C0=np.eye(6)*1e11;C1=(1-fraction)*C0
    value=damage_delta(triangle,mass,rho,cohort,old,np.zeros_like(old),C0,C1)
    expected=.5*mass/rho*.001*.1**2*(C1[0,0]-C0[0,0])
    if fraction==0:assert value==0.
    else:assert value==pytest.approx(expected,rel=2e-12)
