from dataclasses import replace
import numpy as np
import pytest
from scipy.linalg import expm
from analysis.genesis_backward_euler_prototype.core import Memory,step,conical_memory,conical_endpoint
from tectonics.genesis_volume_operator import conical_strain_operators,ANGULAR_BARYCENTRIC
from tectonics.genesis_volume_loading import coefficient_inner_product,RadialMoments,isotropic_elasticity


def audit(r):
    L=r['ledger'];scale=sum(abs(x) for k,x in L.items() if k!='balance_residual_j')
    assert abs(L['balance_residual_j'])<2e-13*max(scale,1e-300)
    assert r['impulse_relative_residual']<2e-13
    for k in ['physical_drag_j','physical_maxwell_j','numerical_kinetic_j','numerical_memory_j']:assert L[k]>=0


@pytest.mark.parametrize('z',[1e-4,1.,2.,10.,1e4,3e10,5.62e18])
def test_actual_solve_monotone_drag(z):
    r=step([[1]],[[1]],[0],[0],[1],z);audit(r)
    assert r['velocity'][0]==pytest.approx(1/(1+z),rel=3e-15)
    assert r['velocity'][0]>0
    assert r['ledger']['physical_drag_j']==pytest.approx(z/(1+z)**2)
    assert r['ledger']['numerical_kinetic_j']==pytest.approx(.5*(z/(1+z))**2)


def test_force_free_momentum_and_comoving_birth():
    M=np.diag([2.,3.]);birth=np.diag([.5,.7]);v=np.array([1.,-2.])
    r=step(M,np.zeros((2,2)),[0,0],[0,0],v,.3,birth_mass=birth);audit(r)
    np.testing.assert_allclose(r['velocity'],v)
    np.testing.assert_allclose((M+birth)@r['velocity']-M@v,r['birth_momentum'])
    assert r['ledger']['numerical_kinetic_j']<1e-28


def test_internal_forces_preserve_translation_momentum():
    m=Memory(np.array([[-1.,1.]]),np.array([[5.]]),np.array([.2]),np.array([0.]),.7)
    r=step(np.diag([2.,3.]),np.zeros((2,2)),[0,0],[0,0],[.5,-.3],.4,[m]);audit(r)
    assert np.sum(r['internal_force'])==0
    assert np.dot([2,3],r['velocity'])==pytest.approx(.1)


def test_manufactured_forced_thermal_birth_solve_and_ledgers():
    dt=.3;M=np.diag([2.,3.]);birth=np.diag([.4,.6]);D=np.diag([.7,.8]);v0=np.array([.1,-.2]);want=np.array([.3,.1])
    m=Memory(np.array([[1.,-.2],[.3,.8]]),np.array([[4.,.2],[.2,3.]]),np.array([.02,-.01]),np.array([.003,.001]),.9)
    b=Memory(np.array([[.2,.5]]),np.array([[2.]]),np.zeros(1),np.array([.004]),0.,True)
    force=(M+birth)@(want-v0)/dt+D@want
    for c in [m,b]:force+=c.B.T@c.H@((c.old+c.B@(dt*want)-c.thermal_increment)/(1+dt*c.relaxation_per_s))
    before=m.old.copy();r=step(M,D,force,[.2,.4],v0,dt,[m,b],birth);audit(r)
    np.testing.assert_allclose(r['velocity'],want,rtol=2e-15,atol=2e-16)
    np.testing.assert_array_equal(m.old,before)
    assert r['ledger']['external_work_j']==pytest.approx(force@(dt*want))
    assert r['ledger']['birth_kinetic_inflow_j']==pytest.approx(.5*v0@birth@v0)


def test_full_maxwell_coupling_small_step_limit_against_matrix_exponential():
    # v'=-.3v-2e; e'=v-.7e, no external/thermal input.
    A=np.array([[-.3,-2.],[1.,-.7]]);exact=expm(.4*A)@np.array([.3,.1]);errors=[]
    for n in [8,16,32]:
        v=.3;e=.1;q=0.
        for _ in range(n):
            m=Memory(np.ones((1,1)),np.array([[2.]]),np.array([e]),np.zeros(1),.7)
            r=step([[1]],[[.3]],[0],[q],[v],.4/n,[m]);audit(r)
            v=r['velocity'][0];e=r['memory'][0][0];q=r['displacement'][0]
        errors.append(np.linalg.norm(np.array([v,e])-exact))
    assert 1.8<errors[0]/errors[1]<2.1
    assert 1.8<errors[1]/errors[2]<2.1


def test_pure_memory_relaxation_has_separate_physical_and_numerical_loss():
    m=Memory(np.zeros((1,1)),np.array([[6.]]),np.array([.2]),np.zeros(1),2.)
    r=step([[1]],[[0]],[0],[0],[0],.5,[m]);audit(r)
    assert r['memory'][0][0]==pytest.approx(.1)
    assert r['ledger']['physical_maxwell_j']==pytest.approx(.06)
    assert r['ledger']['numerical_memory_j']==pytest.approx(.03)


def test_real_conical_operator_energy_and_virtual_work_adapter():
    X=np.array([[0.,0.,10.],[2.,0.,10.],[0.,2.,10.]])
    B0,B1=conical_strain_operators(X,np.eye(3),ANGULAR_BARYCENTRIC)
    C=isotropic_elasticity(60e9,.25);rng=np.random.default_rng(911)
    e0=rng.normal(size=(3,6))*1e-5;e1=rng.normal(size=(3,6))*1e-5;du=rng.normal(size=18)*1e-5
    m=conical_memory(B0,B1,e0,e1,np.zeros((3,6)),C,2.,1.2,.03,.01)
    radial=RadialMoments((2.,),(1.2,),(.03,))
    energy=coefficient_inner_product(e0[None],e1[None],e0[None],e1[None],C,radial)[0]
    work=coefficient_inner_product(e0[None],e1[None],(B0@du)[None],(B1@du)[None],C,radial)[0]
    assert m.old@m.H@m.old==pytest.approx(energy,rel=2e-15)
    assert m.old@m.H@m.B@du==pytest.approx(work,rel=3e-15)
    r=step(np.eye(18),np.eye(18),np.zeros(18),np.zeros(18),np.zeros(18),.1,[m]);audit(r)
    a0,a1=conical_endpoint(B0,B1,e0,e1,np.zeros((3,6)),r['displacement'],.1,.01)
    rebuilt=conical_memory(B0,B1,a0,a1,np.zeros((3,6)),C,2.,1.2,.03,.01)
    np.testing.assert_allclose(rebuilt.old,r['memory'][0],rtol=2e-13,atol=2e-20)
    energy1=.5*coefficient_inner_product(a0[None],a1[None],a0[None],a1[None],C,radial)[0]
    assert energy1==pytest.approx(r['ledger']['elastic_after_j'],rel=2e-14)


def test_newborn_relaxes_on_following_step_and_rejects_changed_birth_closure():
    m=Memory(np.ones((1,1)),np.ones((1,1)),np.zeros(1),np.zeros(1),0.,True)
    r=step([[1]],[[0]],[1],[0],[0],.1,[m],[[.2]]);audit(r)
    old=replace(m,old=r['memory'][0],relaxation_per_s=2.,born=False)
    audit(step([[1.2]],[[0]],[0],r['displacement'],r['velocity'],.1,[old]))
    with pytest.raises(ValueError):step([[1]],[[0]],[0],[0],[0],.1,[replace(m,relaxation_per_s=2.)])


def test_negative_birth_mass_refused():
    with pytest.raises(ValueError):step([[1]],[[0]],[0],[0],[0],1.,birth_mass=[[-.1]])


def test_elastic_oscillator_loss_is_numerical_not_physical():
    m=Memory(np.ones((1,1)),np.array([[4.]]),np.array([.2]),np.zeros(1),0.)
    r=step([[1]],[[0]],[0],[0],[.3],10.,[m]);audit(r)
    assert r['ledger']['physical_drag_j']==0
    assert r['ledger']['physical_maxwell_j']==0
    assert r['ledger']['numerical_kinetic_j']+r['ledger']['numerical_memory_j']>0
    assert r['ledger']['kinetic_after_j']+r['ledger']['elastic_after_j']<.001
