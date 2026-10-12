import pytest
from analysis.genesis_intact_time_sensitivity.metrics import difference,material,compare


def packet(parts):
    return {'runtime_state':{'geometry':{'cohorts':[{'id':i+1,'face_index':0,'q_lo':lo,'live_q_hi':hi,'mass_kg':100*(hi-lo)} for i,(lo,hi) in enumerate(parts)],'column_mass_kg':[100]*80,'vertices_m':[[0,0,0]]}}}


def test_newborn_partition_does_not_change_integrated_mass():
    a=material(packet([(0,.2),(.2,.4)]),{1});b=material(packet([(0,.2),(.2,.3),(.3,.4)]),{1})
    assert a['mass_kg']==b['mass_kg']==40 and a['newborn_mass_kg']==b['newborn_mass_kg']==20
    assert a['cohorts']!=b['cohorts'] and a['front_q']==b['front_q']


@pytest.mark.parametrize('parts',[[(0,.2),(.3,.4)],[(0,.3),(.2,.4)]])
def test_gap_overlap_refused(parts):
    with pytest.raises(ValueError):material(packet(parts),{1})


def test_zero_metric_and_unlike_array_refused():
    assert difference([0,0],[0,0])['symmetric_relative_max']==0
    with pytest.raises(ValueError):difference([1,2],[1])


def test_wrong_time_not_called_convergence():
    with pytest.raises(ValueError,match='endpoint'):compare({'time_myr':0},{'time_myr':1})
