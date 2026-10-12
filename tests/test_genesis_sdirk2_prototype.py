import numpy as np
import pytest
from scipy.linalg import expm
from analysis.genesis_sdirk2_prototype.core import step,A,B,C,GAMMA,ENERGY_MATRIX
from analysis.genesis_sdirk2_prototype.model import Coupled,Block
from analysis.genesis_sdirk2_prototype.audit import audit


def model(alpha=.01,g=.7,ports=None,regions=('solid',),Bmat=None):
    bm=np.array([[1.]]) if Bmat is None else np.asarray(Bmat,float)
    nv=bm.shape[1]
    return Coupled(np.eye(nv),.3*np.eye(nv),[2.],[3.],[4.],[100.],[120.],[0.],[],
        [Block(bm,np.array([[2.]]),g,np.array([alpha]),0)],
        ports or (lambda *a:(np.zeros(1),np.zeros(nv))),regions)


def check(m,y,r):
    L=audit(m,y,r)
    scale=abs(m.energy(y))+abs(m.energy(r['after']))+sum(abs(L[k]) for k in ('external_work_j','source_heat_j','signed_numerical_j'))
    assert abs(L['balance_residual_j'])<2e-10*max(scale,1.)
    assert np.max(np.abs(L['stage_equation_defects']))<2e-9*max(np.max(np.abs(y)),1.)
    assert np.max(np.abs(L['independent_rate_defects']))<2e-8
    assert np.linalg.norm(L['impulse_residual'])<2e-9
    assert L['physical_drag_j']>=0 and L['physical_maxwell_j']>=0
    return L


def test_tableau_order_stability_and_indefinite_energy_matrix():
    assert sum(B)==pytest.approx(1.) and B@C==pytest.approx(.5)
    assert np.min(np.linalg.eigvalsh(ENERGY_MATRIX))<0<np.max(np.linalg.eigvalsh(ENERGY_MATRIX))
    for z in (-1.,-10.,-1e4,-1e8):
        R=(1+(1-2*GAMMA)*z)/(1-GAMMA*z)**2
        r=step(lambda t,y:z*y,0.,np.ones(1),1.,np.ones(1),lambda y:None)
        assert r['after'][0]==pytest.approx(R,abs=1e-12)
        assert abs(R)<1
    assert abs(R)<1e-6


def test_analytic_maxwell_heat_second_order_and_variable_steps():
    m=model(alpha=0.,g=2.,Bmat=[[0.]])
    y0=np.array([60.,0.,.3]);end=.8
    exact=np.array([60.+.09*(1-np.exp(-4*end)),0.,.3*np.exp(-2*end)])
    for variable in (False,True):
        errors=[]
        for n in (10,20,40):
            y=y0.copy();t=0.;weights=np.tile([.5,1.5],n//2) if variable else np.ones(n)
            for h in end*weights/weights.sum():
                r=step(m.rhs,t,y,h,np.ones(3),m.guard);check(m,y,r);y=r['after'];t+=h
            errors.append(np.linalg.norm(y-exact))
        assert all(3.6<a/b<4.5 for a,b in zip(errors,errors[1:]))


def test_coupled_manufactured_all_fields_second_order():
    # Same equations: prescribed v=.2+.1t,T=10+.2t; solve e analytically.
    g=.7;alpha=.01;v0=.2;acc=.1;rate=.2;e0=.1
    c0=v0-alpha*rate
    def exact(t):
        e=e0*np.exp(-g*t)+c0*(1-np.exp(-g*t))/g+acc*(t/g-(1-np.exp(-g*t))/g**2)
        return np.array([6*(10+rate*t),v0+acc*t,e])
    def ports(t,T,v,mem):
        _,vv,e=exact(t)
        return np.array([6*rate-g*2*e*e-2*e*alpha*rate]),np.array([acc+.3*vv+2*e])
    m=model(ports=ports);errors=[]
    for n in (8,16,32):
        y=exact(0);t=0.
        for _ in range(n):
            r=step(m.rhs,t,y,1/n,np.ones(3),m.guard);check(m,y,r);y=r['after'];t+=1/n
        errors.append(np.abs(y-exact(1)))
    for a,b in zip(errors,errors[1:]):assert np.all((a/b>3.7)&(a/b<4.3))


def test_negative_numerical_work_explicit_never_added_to_heat():
    m=model(alpha=0.,g=2.,Bmat=[[0.]])
    y=np.array([60.,0.,.3]);r=step(m.rhs,0.,y,.5,np.ones(3),m.guard);L=check(m,y,r)
    assert L['signed_numerical_j']<0
    assert r['after'][0]-y[0]==pytest.approx(L['physical_maxwell_j'])
    assert L['thermal_feedback_j'][0]==pytest.approx(L['physical_maxwell_j'])


def test_translation_momentum_with_internal_memory():
    m=model(alpha=0.,Bmat=[[-1.,1.]])
    m.D=np.zeros((2,2));y=np.array([60.,.5,-.3,.1])
    r=step(m.rhs,0.,y,.4,np.ones(4),m.guard);check(m,y,r)
    assert r['after'][1:3].sum()==pytest.approx(.2,abs=1e-14)


def test_unconditional_temperature_positivity_not_claimed():
    m=Coupled([[1.]],[[0.]],[1.],[1.],[0.],[100.],[120.],[0.],[],[],
        lambda t,T,*_:(-T,np.zeros(1)),('solid',))
    y=np.array([1.,0.]);r=step(m.rhs,0.,y,.1,np.ones(2),m.guard);check(m,y,r)
    assert r['after'][0]>0
    with pytest.raises(ValueError,match='NONPOSITIVE'):step(m.rhs,0.,y,3.,np.ones(2),m.guard)
    np.testing.assert_array_equal(y,[1.,0.])


def test_phase_crossing_stops_and_exact_event_restart_has_no_hidden_heat():
    # Fixed synthetic enthalpy law, no actual remelt/cohort creation.
    def make(region):return Coupled([[1.]],[[0.]],[1.],[1.],[2.],[2.],[4.],[0.],[],[],
        lambda *_:(np.array([1.]),np.zeros(1)),(region,))
    m=make('solid');y=np.array([1.75,0.])
    with pytest.raises(ValueError,match='PHASE_EVENT'):step(m.rhs,0.,y,.5,np.ones(2),m.guard)
    r=step(m.rhs,0.,y,.25,np.ones(2),m.guard);L=check(m,y,r)
    assert r['after'][0]==2. and L['signed_numerical_j']==0
    mush=make('mush');rr=step(mush.rhs,.25,r['after'],.25,np.ones(2),mush.guard);check(mush,r['after'],rr)
    assert mush.thermal(rr['after'][:1])[0]==pytest.approx(2.125)
    assert (mush.thermal(rr['after'][:1])[0]-2)/2==pytest.approx(.0625)


def test_independent_audit_detects_corrupted_stage():
    m=model();y=np.array([60.,.2,.1]);r=step(m.rhs,0.,y,.1,np.ones(3),m.guard)
    r['stages'][0,2]+=.01
    L=audit(m,y,r)
    assert np.max(np.abs(L['independent_rate_defects']))>.001


def test_nonpositive_effective_capacity_rejected():
    m=model(alpha=100.);y=np.array([60.,0.,1.])
    with pytest.raises(ValueError,match='CAPACITY'):step(m.rhs,0.,y,.1,np.ones(3),m.guard)


def test_real_production_operators_on_manufactured_patch():
    from tectonics.genesis_volume_operator import conical_strain_operators,ANGULAR_BARYCENTRIC
    from tectonics.genesis_volume_loading import isotropic_elasticity,coefficient_inner_product,RadialMoments
    from tectonics.genesis_volume_basal import local_consistent_mass_matrix,local_basal_drag_and_force
    from analysis.genesis_backward_euler_prototype.core import conical_memory
    X=np.array([[0.,0.,10.],[2.,0.,10.],[0.,2.,10.]])
    B0,B1=conical_strain_operators(X,np.eye(3),ANGULAR_BARYCENTRIC)
    Cmat=isotropic_elasticity(60e9,.25)
    cm=conical_memory(B0,B1,np.zeros((3,6)),np.zeros((3,6)),np.zeros((3,6)),Cmat,2.,1.2,.03,.01)
    M=local_consistent_mass_matrix(6000.,.9,.8)
    drag=local_basal_drag_and_force(X,.8,np.zeros((3,3)),1.)
    D=drag[0]
    expansion=np.zeros(36);expansion[:18]=np.tile([1e-5]*3+[0.]*3,3)
    m=Coupled(M,D,[6000.],[1200.],[4e5],[1400.],[1800.],[0.],[],
        [Block(cm.B,cm.H,.01,expansion,0)],lambda *_:(np.array([1e4]),np.zeros(18)),('solid',))
    y=np.concatenate(([6000*1200*1000.],np.ones(18)*1e-9,np.zeros(36)))
    scale=np.concatenate(([1e4],np.full(18,1e-8),np.full(36,1e-8)))
    r=step(m.rhs,0.,y,.001,scale,m.guard);L=audit(m,y,r)
    assert abs(L['balance_residual_j'])<2e-6
    assert np.linalg.norm(L['impulse_residual'])<1e-10
    e=r['after'][19:];e0=e[:18].reshape(3,6)-1.2*e[18:].reshape(3,6)/np.sqrt(.03);e1=e[18:].reshape(3,6)/np.sqrt(.03)
    energy=coefficient_inner_product(e0[None],e1[None],e0[None],e1[None],Cmat,RadialMoments((2.,),(1.2,),(.03,)))[0]/2
    assert energy==pytest.approx(.5*e@cm.H@e,rel=1e-13)

def test_mass_form_matches_rhs_stages_without_tolerance_change():
    m=model();y=np.array([60.,.2,.1])
    a=step(m.rhs,0.,y,.1,np.ones(3),m.guard)
    b=step(m.rhs,0.,y,.1,np.ones(3),m.guard,implicit_residual=m.implicit_residual)
    np.testing.assert_allclose(a['after'],b['after'],rtol=1e-12,atol=1e-13)
    check(m,y,b)


def test_near_phase_boundary_second_order_without_crossing():
    # Exponential heating ends below solidus. No crossing is silently accepted.
    m=Coupled([[1.]],[[0.]],[1.],[1.],[2.],[2.],[4.],[0.],[],[],
        lambda t,*_:(np.array([np.exp(t)]),np.zeros(1)),('solid',))
    end=np.log(1.999);errors=[]
    for n in (8,16,32):
        y=np.array([1.,0.]);t=0.
        for _ in range(n):
            r=step(m.rhs,t,y,end/n,np.ones(2),m.guard);check(m,y,r)
            y=r['after'];t+=end/n
        errors.append(abs(y[0]-1.999))
    assert all(3.8<a/b<4.2 for a,b in zip(errors,errors[1:]))


def test_conduction_pair_conserves_total_enthalpy():
    m=Coupled([[1.]],[[0.]],[1.,2.],[1.,1.],[0.,0.],[100.,100.],[120.,120.],[0.,0.],[(0,1,3.)],[],
        lambda *_:(np.zeros(2),np.zeros(1)),('solid','solid'))
    y=np.array([10.,40.,0.]);r=step(m.rhs,0.,y,.1,np.ones(3),m.guard);L=check(m,y,r)
    assert r['after'][:2].sum()==pytest.approx(50.)
    assert sum(L['conduction_node_j'])==0.
    assert np.all(m.thermal(r['after'][:2])>0)
