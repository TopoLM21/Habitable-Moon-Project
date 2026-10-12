import numpy as np
import pytest
from analysis.genesis_sdirk2_prototype.thin_diagnosis import refined_solve,solve_stage,step
from analysis.genesis_sdirk2_prototype.core import step as generic,GAMMA,ENERGY_MATRIX
from analysis.genesis_sdirk2_prototype.model import Coupled,Block
from analysis.genesis_sdirk2_prototype.audit import audit
from analysis.genesis_sdirk2_prototype.decision import energy_bound


def problem():
    return Coupled([[2.]],[[.3]],[2.],[3.],[4.],[100.],[120.],[0.],[],
        [Block(np.array([[1.]]),np.array([[2.]]),.7,np.array([.01]),0)],
        lambda *_:(np.array([.1]),np.array([.2])),('solid',))


def test_elimination_matches_original_coupled_stage_equations():
    m=problem();y=np.array([60.,.2,.1]);scale=np.ones(3)
    r=step(m,0.,y,.2,scale)
    reference=generic(m.rhs,0.,y,.2,scale,m.guard,implicit_residual=m.implicit_residual)
    np.testing.assert_allclose(r['after'],reference['after'],rtol=1e-12,atol=1e-13)
    assert all(d['original_scaled_residual']<=2e-10 for d in r['details'])
    L=audit(m,y,r)
    assert abs(L['balance_residual_j'])<1e-12
    assert np.max(abs(np.array(L['independent_rate_defects'])))<1e-11
    assert np.linalg.norm(L['impulse_residual'])<1e-13
    assert abs(L['signed_numerical_j'])<=energy_bound(m,r)['absolute_signed_numerical_bound_j']
    np.testing.assert_array_equal(y,[60.,.2,.1])


def test_equilibration_refinement_solves_same_system():
    ld=np.longdouble
    M=np.array([[1e-10,2e-6],[2e-6,4.1]],ld)
    wanted=np.array([[1.,2.],[-.5,.3]],ld);rhs=M@wanted
    got,info=refined_solve(M,rhs)
    np.testing.assert_allclose(got,wanted,rtol=2e-13,atol=2e-13)
    assert info['backward_errors'][-1]<32*np.finfo(ld).eps


def test_be_stage_uses_unchanged_residual_threshold():
    m=problem();y=np.array([60.,.2,.1]);Y,detail=solve_stage(m,y,.5,.5,np.ones(3))
    assert detail['original_scaled_residual']<2e-10
    assert np.max(abs(m.implicit_residual(.5,Y,y,.5,np.ones(3))))<2e-10


def test_negative_numerical_energy_is_already_in_unforced_scalar_decay():
    # Independent closed-form scalar calculation: no heat coupling or forcing.
    h=.5;g=2.;initial=.3;metric=2.
    z1=initial/(1+GAMMA*h*g)
    z2=(initial-h*(1-GAMMA)*g*z1)/(1+GAMMA*h*g)
    physical=h*g*metric*((1-GAMMA)*z1*z1+GAMMA*z2*z2)
    defect=.5*metric*(initial**2-z2**2)-physical
    rates=np.array([-g*z1,-g*z2])
    rk_defect=.5*h*h*metric*(rates@ENERGY_MATRIX@rates)
    bound=.5*h*h*metric*np.max(abs(np.linalg.eigvalsh(ENERGY_MATRIX)))*(rates@rates)
    assert defect==pytest.approx(rk_defect,abs=1e-16)
    assert -bound<=defect<0


def test_refinement_rejects_invalid_system():
    with pytest.raises(ValueError):refined_solve(np.zeros((2,2)),np.ones((2,1)))
    with pytest.raises(ValueError):refined_solve(np.array([[np.nan]]),np.ones((1,1)))


def test_specialized_solver_refuses_other_phase_branch():
    m=problem();m.regions=('mush',)
    with pytest.raises(ValueError,match='SCOPE'):solve_stage(m,np.array([60.,.2,.1]),.1,.1,np.ones(3))
