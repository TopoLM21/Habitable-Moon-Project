import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm
from analysis.genesis_backward_euler_prototype.core import Memory
from analysis.genesis_forced_memory_diagnosis.exponential import weights,update
from analysis.genesis_forced_memory_diagnosis.prototype import step


@pytest.mark.parametrize('z',[0.,1e-10,1e-4,9.99e-4,.001,.008,1.,20.,1000.])
def test_exact_material_matches_independent_ode_and_heat_integral(z):
    e=np.array([.7,-.2]);f=np.array([-.3,.5]);H=np.array([[3.,.4],[.4,2.]])
    r=update(e,f,H,z)
    A=np.zeros((3,3));A[:2,:2]=-z*np.eye(2);A[:2,2]=f
    at=lambda t:(expm(t*A)@np.r_[e,1.])[:2]
    np.testing.assert_allclose(r['endpoint'],at(1),rtol=1e-13,atol=1e-15)
    heat=quad(lambda t:z*(at(t)@H@at(t)),0,1,epsabs=2e-12,epsrel=2e-12)[0]
    assert r['heat_j']==pytest.approx(heat,rel=3e-10,abs=1e-14)
    assert abs(r['elastic_after_j']-r['elastic_before_j']-r['forcing_work_j']+r['heat_j'])<2e-10
    assert r['heat_j']>=0


def test_pure_relaxation_analytic_energy_and_zero_forcing():
    H=np.array([[4.]]);r=update([2.],[0.],H,3.)
    assert r['elastic_after_j']==pytest.approx(8*np.exp(-6))
    assert r['heat_j']==pytest.approx(8*(-np.expm1(-6)))


def test_elastic_limit_and_constant_equilibrium():
    r=update([.3],[.2],[[4.]],0)
    np.testing.assert_allclose(r['endpoint'],[.5]);np.testing.assert_allclose(r['mean'],[.4]);assert r['heat_j']==0
    r=update([.3],[.6],[[4.]],2)
    np.testing.assert_allclose(r['endpoint'],[.3]);assert r['heat_j']==pytest.approx(.72)


def audit(r):
    L=r['ledger'];scale=sum(abs(v) for v in L.values())
    assert abs(L['balance_residual_j'])<5e-13*max(1,scale)
    np.testing.assert_allclose(r['impulse_residual'],0,atol=3e-13)


def test_manufactured_force_thermal_momentum_and_energy():
    M=np.diag([2.,3.]);D=np.diag([.2,.4]);v0=np.array([.2,-.1]);v1=np.array([.1,.4]);h=.7
    m=Memory(np.array([[1.,-.2],[.3,.8]]),np.array([[4.,.2],[.2,3.]]),np.array([.2,-.1]),np.array([.01,-.03]),.7)
    exact=update(m.old,m.B@(h*v1)-m.thermal_increment,m.H,h*m.relaxation_per_s)
    F=M@(v1-v0)/h+D@v1+m.B.T@m.H@exact['endpoint']
    old=m.old.copy();r=step(M,D,F,v0,h,[m]);audit(r)
    np.testing.assert_allclose(r['velocity'],v1,atol=3e-16);np.testing.assert_array_equal(m.old,old)


def test_signed_coupling_is_not_a_nonnegative_dissipation():
    m=Memory(np.ones((1,1)),np.ones((1,1)),np.ones(1),np.zeros(1),1.)
    end=update([1],[.1],[[1]],1)['endpoint'][0]
    r=step([[1]],[[0]],[.1+end],[0],1,[m]);audit(r)
    assert r['ledger']['signed_coupling_defect_j']<-.02
    assert r['ledger']['signed_coupling_defect_j']+r['ledger']['numerical_kinetic_j']<0


@pytest.mark.parametrize('gamma',[0.,.01,1.,100.])
def test_scalar_mode_stable_and_stiff_limit_decays(gamma):
    # Independent analytic amplification: char poly x²-(1+a)/d*x+a/d,
    # d=1+h²*b. Jury conditions hold for h>0, 0<=a<=1, b>0.
    for h in [.001,.1,1.,10.,1e6]:
        a,b,*_=weights(h*gamma);d=1+h*h*b
        expected=np.array([[1,-h*a],[h*b,a]])/d
        columns=[]
        for v,e in [(1.,0.),(0.,1.)]:
            m=Memory(np.ones((1,1)),np.ones((1,1)),np.array([e]),np.zeros(1),gamma)
            r=step([[1]],[[0]],[0],[v],h,[m]);columns.append([r['velocity'][0],r['memory'][0]['endpoint'][0]])
        np.testing.assert_allclose(np.array(columns).T,expected,atol=3e-16,rtol=1e-8)
        radius=max(abs(np.linalg.eigvals(expected)));assert radius<=1+1e-14
        if h==1e6:assert radius<1e-3


@pytest.mark.parametrize('bad',[-1.,float('nan'),float('inf')])
def test_reject_invalid_relaxation(bad):
    with pytest.raises(ValueError):weights(bad)
