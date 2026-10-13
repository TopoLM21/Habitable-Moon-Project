"""Small analytical controls for the independent diagnostic, not the planet."""
import numpy as np
from analysis.genesis_geological_handoff.shear_diagnostic import setup,smooth,initial,advance


def test_filter_constant_and_discrete_mode():
    c=setup(128,True);x=(np.arange(128)+.5)/128
    assert np.max(abs(smooth(c,np.ones(128))-1))<1e-14
    mode=np.cos(2*np.pi*x)
    expected=mode/(1+4*8**2*np.sin(np.pi/128)**2)
    assert np.max(abs(smooth(c,mode)-expected))<1e-14
    kernel=smooth(c,np.eye(1,128,0)[0])
    assert kernel.min()>0 and abs(kernel.sum()-1)<1e-14


def test_uniform_elastic_maxwell_limit():
    c=setup(128,False);s=initial(c);dt=.025
    r=advance(c,s,dt)
    expected=dt/(1+dt/100)
    assert abs(r['tau']-expected)<1e-14
    assert np.max(np.abs(r['kappa']))==0
    assert r['ledger']['plastic']==0


def test_uniform_active_return_and_energy():
    c=setup(128,False);c['c0'][:]=1
    s=initial(c);s.update(tau=1.2,time=1.2,gamma=np.full(128,1.2).tolist())
    dt=.025;r=advance(c,s,dt)
    # Independent scalar solution, positive active branch with no softening.
    expected=(1.2+dt+dt/.1)/(1+dt/100+dt/.1)
    assert abs(r['tau']-expected)<1e-14
    assert np.ptp(r['kappa'])==0
    assert abs(r['kappa'][0]-dt/.1*(expected-1))<1e-14
    ledger=r['ledger']
    balance=.5*(expected**2-1.2**2)+ledger['maxwell']+ledger['plastic']+ledger['numerical']-ledger['external']
    assert abs(balance)<1e-14


def test_uniform_weakening_does_not_create_pattern():
    c=setup(128,True);c['c0'][:]=1;s=initial(c)
    for _ in range(100):s=advance(c,s,.025)
    assert np.ptp(s['kappa'])==0
    assert s['peak_cv']<1e-14
