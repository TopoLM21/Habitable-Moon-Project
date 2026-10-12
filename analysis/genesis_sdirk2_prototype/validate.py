"""Finite manufactured validation, including saved real-cohort operators. No planet run."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .core import step,ENERGY_MATRIX,GAMMA
from .model import Coupled,Block
from .audit import audit
from analysis.genesis_adaptive_control.controller import time_limit

ROOT=Path(__file__).resolve().parents[2]
SAVED=Path('/workspace/genesis-forced-pair-25-01/001_whole.json')
SAVED_SHA='4ac5efaeb55948a226379bbc184f298248a2e09a67a423699748790e0d3a0539'
PARENT=ROOT/'analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json'
PARENT_SHA='62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7'


def real_patch():
    from tectonics.genesis_volume_operator import conical_strain_operators,conical_radial_moments,ANGULAR_BARYCENTRIC
    from tectonics.genesis_volume_basal import local_consistent_mass_matrix,local_basal_drag_and_force
    from analysis.genesis_backward_euler_prototype.core import conical_memory
    raw=SAVED.read_bytes();raw0=PARENT.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SAVED_SHA
    assert hashlib.sha256(raw0).hexdigest()==PARENT_SHA
    p=json.loads(raw)['packet'];parent=json.loads(raw0)
    g=p['state']['geometry'];cohort=g['cohorts'][0];inp=p['inputs'];X=np.array(cohort['top_triangle_m'])
    B0,B1=conical_strain_operators(X,np.array(inp['basis_xyz'][0]),ANGULAR_BARYCENTRIC)
    mean,var=conical_radial_moments(cohort['outer_scale'],cohort['inner_scale'])
    dt=(p['thermal']['time_myr']-parent['runtime_state']['thermal']['time_myr'])*365.25*86400*1e6
    gamma=inp['maxwell_ratio'][0]/dt
    cm=conical_memory(B0,B1,np.zeros((3,6)),np.zeros((3,6)),np.zeros((3,6)),np.array(inp['elasticity_pa'][0]),cohort['volume_m3'],mean,var,gamma)
    M=local_consistent_mass_matrix(cohort['mass_kg'],cohort['outer_scale'],cohort['inner_scale'])
    # This is an isolated manufactured cohort patch, not the global basal assembly.
    D,_,_=local_basal_drag_and_force(X,cohort['inner_scale'],np.zeros((3,3)),inp['basal_drag_pa_s_m'])
    params=parent['source_packet']['source']['parameters'];gp=params['thermal'];sp=params['shell']
    U=np.zeros(36);U[:18]=np.tile([sp['linear_expansion_per_k']]*3+[0.]*3,3)
    mass=cohort['mass_kg'];cp=gp['silicate_heat_capacity_j_kg_k'];T0=gp['solidus_k']-10.
    amp=np.tile([1e-5,-2e-5,1e-5,0.,0.,0.],3);omega=.01
    def exact(t):
        sn=np.sin(omega*t);cs=np.cos(omega*t);ex=np.exp(-gamma*t)
        e=cm.B@amp*(gamma*sn-omega*cs+omega*ex)/(gamma**2+omega**2)-U*omega*(gamma*cs+omega*sn-gamma*ex)/(gamma**2+omega**2)
        return np.concatenate(([mass*cp*(T0+sn)],amp*sn,e))
    def ports(t,T,v,mem):
        y=exact(t);vv=y[1:19];e=y[19:];rate=omega*np.cos(omega*t)
        Q=mass*cp*rate-gamma*(e@cm.H@e)-(e@cm.H@U)*rate
        F=M@(amp*rate)+D@vv+cm.B.T@cm.H@e
        return np.array([Q]),F
    m=Coupled(M,D,[mass],[cp],[gp['silicate_latent_heat_j_kg']], [gp['solidus_k']],[gp['liquidus_k']],[0.],[],[Block(cm.B,cm.H,gamma,U,0)],ports,('solid',))
    scale=np.concatenate(([mass*cp],np.full(18,1e-5),np.full(36,1e-5)))
    return m,exact,scale,dict(cohort_id=cohort['id'],mass_kg=mass,source_sha256=SAVED_SHA,
        canonical_sha256=PARENT_SHA,operator_dimension=18,memory_dimension=36,relaxation_per_s=gamma,
        scope='isolated real-geometry/coefficient patch with manufactured states and forcing; not global planet')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    with time_limit(120):
        m,exact,scale,provenance=real_patch();patch_report={}
        # One bounded real-operator attempt. A nonlinear failure is evidence,
        # never a reason to relax tolerances or change physical coefficients.
        try:
            y=exact(0.)
            r=step(m.rhs,0.,y,12.5,scale,m.guard,implicit_residual=m.implicit_residual)
            L=audit(m,y,r)
            patch_report=dict(status='SOLVED_NOT_ADMITTED',audit=L)
        except ValueError as exc:
            patch_report=dict(status='BLOCKED_STAGE_SOLVE',reason=str(exc))
        patch_report['mass_condition_number']=float(np.linalg.cond(m.M))
        patch_report['maximum_drag_matrix_entry']=float(np.max(np.abs(m.D)))
        # Analytic Maxwell decay with located thermal feedback and independent drag.
        simple=Coupled([[1.]],[[.3]],[2.],[3.],[4.],[100.],[120.],[0.],[],
            [Block(np.zeros((1,1)),np.array([[2.]]),2.,np.zeros(1),0)],
            lambda *_:(np.zeros(1),np.zeros(1)),('solid',))
        y0=np.array([60.,.2,.3]);end=.8
        truth=np.array([60.+.09*(1-np.exp(-4*end)),.2*np.exp(-.3*end),.3*np.exp(-2*end)])
        convergence={};negative=None
        for variable in (False,True):
            errors=[];balance=[]
            for n in (10,20,40):
                y=y0.copy();t=0.;bal=0.
                weights=np.tile([.5,1.5],n//2) if variable else np.ones(n)
                for h in end*weights/weights.sum():
                    r=step(simple.rhs,t,y,h,np.ones(3),simple.guard);L=audit(simple,y,r)
                    bal=max(bal,abs(L['balance_residual_j']))
                    y=r['after'];t+=h
                errors.append(float(np.linalg.norm(y-truth)));balance.append(bal)
            ratios=[errors[i]/errors[i+1] for i in range(2)]
            assert all(3.5<x<4.6 for x in ratios)
            convergence['variable' if variable else 'uniform']=dict(errors=errors,ratios=ratios,max_balance_residual_j=balance)
        negative=audit(simple,y0,step(simple.rhs,0.,y0,.5,np.ones(3),simple.guard))
        assert negative['signed_numerical_j']<0

        manifest={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(__file__).parent.glob('*.py')}
        report=dict(status='PROTOTYPE_VERIFIED_WITH_ADMISSION_BLOCKERS',scheme='SDIRK2 gamma=1-1/sqrt(2)',
            gamma=GAMMA,energy_matrix_eigenvalues=np.linalg.eigvalsh(ENERGY_MATRIX).tolist(),
            real_patch=provenance,real_patch_attempt=patch_report,analytic_convergence=convergence,
            negative_numerical_example=negative,code_manifest=manifest,new_planetary_steps=0,
            admission='BLOCKED: signed numerical energy may be negative; real thin-cohort stage solve fails; physical interval event bounds and dynamic-coefficient/damage/orbit stages absent')
        assert hashlib.sha256(SAVED.read_bytes()).hexdigest()==SAVED_SHA
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
