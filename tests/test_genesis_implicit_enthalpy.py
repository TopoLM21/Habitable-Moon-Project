import numpy as np
import pytest
from tectonics.genesis_implicit_enthalpy import solve


def run(T,latent=0.,dt=100.):
    T=np.array(T,dtype=float);n=len(T);cp=np.full(n,1200.);L=np.full(n,latent);ts=np.full(n,1400.);tl=np.full(n,2000.)
    h=cp*T+L*np.clip((T-ts)/(tl-ts),0,1)
    result=solve(np.ones(n),h,cp,L,ts,tl,np.zeros(n),[(i,i+1,10.) for i in range(n-1)],dt)
    assert result['specific_h'].sum()==pytest.approx(h.sum(),rel=1e-12)
    assert min(result['temperature_k'])>=min(T)-1e-8 and max(result['temperature_k'])<=max(T)+1e-8
    return result


def test_two_cell_analytic_BE_large_step():
    r=run([1000.,2000.],dt=1e6);difference=1000/(1+2*1e6*10/1200)
    assert r['temperature_k']==pytest.approx([1500-difference/2,1500+difference/2])


@pytest.mark.parametrize('initial',[[1300.,2100.],[1390.,1410.],[1990.,2010.],[1800.,1200.]])
def test_cross_phase_boundaries_with_conservation_and_no_overshoot(initial):run(initial,latent=400000.,dt=1e5)


def test_constant_state_has_no_flux():
    r=run([1600.,1600.,1600.],latent=400000.)
    assert np.all(r['exchange_j']==0)


def test_order_monotonicity_under_warmer_initial_state():
    cold=run([1300.,1800.,1500.],latent=400000.)
    warm=run([1310.,1810.,1510.],latent=400000.)
    assert np.all(warm['temperature_k']>=cold['temperature_k'])
