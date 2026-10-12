from types import SimpleNamespace as S
import numpy as np
import pytest
from analysis.genesis_connected_step_validation.run import gate,guarded_step,GateBoundary


def history():
    return S(damage=np.zeros(80),fault_candidate_age_myr=np.zeros(80),fault_active=np.zeros(80,dtype=bool),membrane_established=True)


PARAM=S(activation_damage=.65,activation_persistence_myr=.05,enabled=True)


@pytest.mark.parametrize('face',range(80))
def test_every_face_stops_before_step(face):
    h=history();h.damage[face]=.65;h.fault_candidate_age_myr[face]=.05
    model=S(fault_parameters=PARAM,step=lambda *_:pytest.fail('Step must not be called'))
    with pytest.raises(GateBoundary):guarded_step(model,S(fault_history=h),lambda _:None)


@pytest.mark.parametrize('field,value',[('damage',.65),('fault_candidate_age_myr',.001),('fault_active',True)])
def test_partial_candidate_and_existing_activity_stop(field,value):
    h=history();getattr(h,field)[79]=value
    with pytest.raises(GateBoundary):gate(h,PARAM)


def test_unknown_creation_law_does_not_block_closed_gate():
    calls=[];state=S(fault_history=history());candidate=S(fault_history=history())
    model=S(fault_parameters=PARAM,creation_law=None,step=lambda s,dt:(calls.append((s,dt)) or candidate))
    kept=[];assert guarded_step(model,state,kept.append) is candidate
    assert calls==[(state,.0001)] and kept==[candidate]


def test_endpoint_candidate_retained_then_stopped():
    state=S(fault_history=history());candidate=S(fault_history=history());candidate.fault_history.damage[40]=.65
    model=S(fault_parameters=PARAM,step=lambda *_:candidate);kept=[]
    with pytest.raises(GateBoundary):guarded_step(model,state,kept.append)
    assert kept==[candidate]


def test_omitted_face_or_nonfinite_damage_refused():
    h=history();h.damage=h.damage[:-1]
    with pytest.raises(GateBoundary):gate(h,PARAM)
    h=history();h.damage[5]=np.nan
    with pytest.raises(GateBoundary):gate(h,PARAM)
