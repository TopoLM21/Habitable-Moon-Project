"""Reproducible bounded numerical evidence, never a Genesis driver."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from .core import Memory,step


def run():
    started=time.monotonic();records=[]
    def capture(name,r):
        L=r['ledger'];scale=sum(abs(x) for k,x in L.items() if k!='balance_residual_j')
        rel=abs(L['balance_residual_j'])/max(scale,np.finfo(float).tiny)
        if rel>2e-13 or r['impulse_relative_residual']>2e-13:raise ValueError('Numerical algebra verification failed')
        records.append(dict(case=name,velocity=r['velocity'].tolist(),memory=[x.tolist() for x in r['memory']],
            ledger=L,balance_relative=rel,impulse_relative=r['impulse_relative_residual']))
    for z in [1e-4,1.,2.,10.,1e4,3e10,5.62e18]:
        r=step([[1]],[[1]],[0],[0],[1],z);capture('drag_z_'+str(z),r)
        records[-1].update(exact_velocity=float(np.exp(-z)),analytic_BE_velocity=1/(1+z))
    errors=[];A=np.array([[-.3,-2.],[1.,-.7]]);exact=expm(.4*A)@np.array([.3,.1])
    for n in [8,16,32]:
        v=.3;e=.1;q=0.
        for i in range(n):
            m=Memory(np.ones((1,1)),np.array([[2.]]),np.array([e]),np.zeros(1),.7)
            r=step([[1]],[[.3]],[0],[q],[v],.4/n,[m]);capture(f'coupled_n{n}_step{i+1}',r)
            v=r['velocity'][0];e=r['memory'][0][0];q=r['displacement'][0]
        errors.append(float(np.linalg.norm(np.array([v,e])-exact)))
    m=Memory(np.array([[1.,-.2],[.3,.8]]),np.array([[4.,.2],[.2,3.]]),np.array([.02,-.01]),np.array([.003,.001]),.9)
    b=Memory(np.array([[.2,.5]]),np.array([[2.]]),np.zeros(1),np.array([.004]),0.,True)
    dt=.3;M=np.diag([2.,3.]);birth=np.diag([.4,.6]);D=np.diag([.7,.8]);v0=np.array([.1,-.2]);want=np.array([.3,.1])
    F=(M+birth)@(want-v0)/dt+D@want
    for c in [m,b]:F+=c.B.T@c.H@((c.old+c.B@(dt*want)-c.thermal_increment)/(1+dt*c.relaxation_per_s))
    capture('manufactured_external_thermal_birth',step(M,D,F,[.2,.4],v0,dt,[m,b],birth))
    # Only hash accepted data; no checkpoint model load, replay or new trajectory.
    parent=Path('analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json')
    pin=hashlib.sha256(parent.read_bytes()).hexdigest()
    if pin!='62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7':raise ValueError('Canonical parent changed')
    paths=[Path(__file__),Path(__file__).with_name('core.py'),Path('tectonics/genesis_volume_dynamics.py'),Path('tectonics/genesis_volume_loading.py'),Path('tectonics/genesis_volume_operator.py')]
    return dict(status='PASS_OPT_IN_BE_LINEAR_PROTOTYPE',base_sha='0006ccb7a1f13e4dcc1a40104f70c5caf52c8bd5',seconds=time.monotonic()-started,
        parent_sha256=pin,default_changed=False,genesis_evolution=False,new_physical_birth_assumption=False,
        coupled_exact=exact.tolist(),coupled_errors=errors,error_ratios=[errors[0]/errors[1],errors[1]/errors[2]],
        max_balance_relative=max(r['balance_relative'] for r in records),max_impulse_relative=max(r['impulse_relative'] for r in records),
        cases=records,pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run()
    with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ['cases','pins']}))
