from types import SimpleNamespace as NS
from dataclasses import replace
import numpy as np
import pytest
from tectonics.genesis_volume_operator import conical_strain_operators,conical_radial_moments,ANGULAR_BARYCENTRIC
from tectonics.genesis_volume_loading import RadialMoments,isotropic_elasticity
from analysis.genesis_backward_euler_prototype.core import conical_memory,step
from analysis.genesis_be_candidate.solver import solve_trial
from analysis.genesis_be_candidate.audit import block
from analysis.genesis_be_candidate.adapter import no_remelt


def fixture():
    X=np.array([[0.,0.,10.],[2.,0.,10.],[0.,2.,10.]])
    frame=np.eye(3);mass=100.;rho=3.;lo=0.;hi=.1
    # Independently integrate radial moments for this manufactured conical band.
    from analysis.genesis_volume_continuation_validation.independent_audit import radial_rule,conical_scales
    q,w=radial_rule(lo,hi);lam,_=conical_scales(X,mass,rho,q);mu=float(w@(1/lam));var=float(w@((1/lam-mu)**2));V=mass*(hi-lo)/rho
    B0,B1=conical_strain_operators(X,frame,ANGULAR_BARYCENTRIC);rng=np.random.default_rng(92)
    e0=rng.normal(size=(3,6))*.001;e1=rng.normal(size=(3,6))*.0001;t=np.full((3,6),.00001)
    C=isotropic_elasticity(1000.,.25);mom=RadialMoments((V,),(mu,),(var,));ratio=.4
    loading=NS(old0=e0[None],old1=e1[None],thermal0=t[None],elasticity_pa=C[None],radial_moments=mom,ratio=np.array([ratio]))
    M=np.eye(18)*3;D=np.eye(18)*.2;F=rng.normal(size=18)*.01;v0=rng.normal(size=18)*.002;dt=.3
    return X,frame,mass,rho,dict(q_lo=lo,live_q_hi=hi),B0,B1,loading,M,D,F,v0,dt


def test_batched_adapter_equals_prototype_and_independent_quadrature():
    X,frame,mass,rho,c,B0,B1,l,M,D,F,v0,dt=fixture()
    du,v,e0,e1,L=solve_trial(M,D,F,v0,dt,np.arange(18)[None],B0[None],B1[None],l)
    mom=l.radial_moments;m=conical_memory(B0,B1,l.old0[0],l.old1[0],l.thermal0[0],l.elasticity_pa[0],mom.volume_m3[0],mom.inverse_radius_mean[0],mom.inverse_radius_variance[0],l.ratio[0]/dt)
    p=step(M,D,F,np.zeros(18),v0,dt,[m]);np.testing.assert_allclose(v,p['velocity'],rtol=3e-13,atol=1e-16)
    Q,force,error=block(X,frame,mass,rho,c,l.elasticity_pa[0],l.old0[0],l.old1[0],e0[0],e1[0],l.thermal0[0],l.ratio[0],du)
    for k,x in Q.items():assert x==pytest.approx(L[k],rel=3e-12,abs=1e-13)
    np.testing.assert_allclose(M@(v-v0)/dt+D@v+force,F,atol=3e-14)
    with pytest.raises(ValueError,match='memory'):block(X,frame,mass,rho,c,l.elasticity_pa[0],l.old0[0],l.old1[0],e0[0]*1.01,e1[0],l.thermal0[0],l.ratio[0],du)


def test_no_retirement_hidden_in_adapter():
    c=NS(id=1,q_lo=0.,live_q_hi=.1,mass_kg=1.,material_id='a',owner_id='b',face_index=0)
    no_remelt(NS(cohorts=[c]),NS(cohorts=[c]))
    changed=NS(**{**vars(c),'live_q_hi':.09})
    with pytest.raises(ValueError):no_remelt(NS(cohorts=[c]),NS(cohorts=[changed]))
    with pytest.raises(ValueError):no_remelt(NS(cohorts=[c]),NS(cohorts=[]))


def test_be_format_rejects_old_or_tampered_checkpoint():
    from analysis.genesis_be_candidate.checkpoint import verify,seal
    from analysis.genesis_be_candidate.adapter import FORMAT,BEState
    from analysis.genesis_connected_step_validation.run import PARENT_SHA
    from dataclasses import fields
    state={f.name:None for f in fields(BEState)};state['inherited_parent_sha256']=PARENT_SHA
    p=seal(dict(format=FORMAT,parent_sha256=PARENT_SHA,source_packet={},state=state,report={}))
    verify(p)
    p['state']['interval_ledger']={'numerical_memory_j':0}
    with pytest.raises(ValueError):verify(p)
    with pytest.raises(ValueError):verify({'format':'genesis-volume-slip-checkpoint-0.1'})


def test_independent_boolean_binding_is_exact():
    from analysis.genesis_be_candidate.audit import close
    assert close([True,False],[True,False],'flags')==0
    with pytest.raises(ValueError):close([True,False],[True,True],'flags')
    with pytest.raises(ValueError):close([1,0],[True,False],'flags')


def test_restart_requires_authenticated_parent_bytes(monkeypatch):
    from analysis.genesis_be_candidate.checkpoint import validate_dependencies
    from pathlib import Path
    monkeypatch.setattr(Path,'read_bytes',lambda _:b'{"forged":true}')
    with pytest.raises(ValueError,match='parent'):validate_dependencies({}, {})
