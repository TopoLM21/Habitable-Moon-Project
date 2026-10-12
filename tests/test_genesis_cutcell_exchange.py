import numpy as np
import pytest
from tectonics.genesis_cutcell_exchange import apply_exchange


def test_signed_work_and_maxwell_exact_two_reservoir_balance():
    out,rows=apply_exchange([100.,200.,300.],[1,2,None],[2,1],[5.,7.],[-3.,11.])
    assert out.tolist()==[96.,208.,300.]
    assert sum(out)-600+sum(x['mechanical_entry_j'] for x in rows)==0
    assert all(x['thermal_entry_j']==-x['mechanical_entry_j'] for x in rows)


def test_correctors_apply_to_base_not_accumulate():
    base=np.array([100.]);first,_=apply_exchange(base,[1],[1],[3.],[1.])
    second,_=apply_exchange(base,[1],[1],[4.],[1.])
    assert base[0]==100 and first[0]==102 and second[0]==103


@pytest.mark.parametrize('cells,ids,Q,W', [([1,1],[1],[1.],[0.]),([1],[2],[1.],[0.]),([1],[1],[-1.],[0.]),([1],[1],[0.],[101.]),([1],[1],[float('nan')],[0.])])
def test_reject_bad_mapping_negative_heat_and_exhaustion(cells,ids,Q,W):
    with pytest.raises(ValueError):apply_exchange([100.]*len(cells),cells,ids,Q,W)


def test_analytic_relaxing_constrained_material_heat_and_thermal_work():
    from types import SimpleNamespace
    from tectonics.genesis_volume_loading import RadialMoments
    from tectonics.genesis_cutcell_exchange import local_work
    old=.1; expansion=.02; ratio=2.; strain=(old-expansion)/(1+ratio)
    e=np.zeros((1,3,6));e[:,:,0]=strain
    th=np.zeros_like(e);th[:,:,0]=expansion
    loading=SimpleNamespace(elasticity_pa=np.eye(6)[None,:,:],radial_moments=RadialMoments((3.,),(1.,),(0.,)),ratio=np.array([ratio]),thermal0=th)
    Q,W=local_work(loading,e,np.zeros_like(e))
    assert Q[0]==pytest.approx(3*ratio*strain**2)
    assert W[0]==pytest.approx(-3*strain*expansion)
    dU=1.5*(strain**2-old**2);numerical=1.5*(strain-old)**2
    assert dU==pytest.approx(W[0]-Q[0]-numerical)
