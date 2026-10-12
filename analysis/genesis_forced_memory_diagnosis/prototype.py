"""Manufactured-only endpoint BE momentum + exact frozen-rate Maxwell.

No Genesis adapter. The signed coupling defect is NOT physical heat and is
not guaranteed dissipative. This exposes why a coefficient swap is insufficient.
"""
import numpy as np
from scipy.linalg import solve
from analysis.genesis_backward_euler_prototype.core import symmetric
from .exponential import weights, update


def step(M, D, force, v0, dt, memories):
    v0=np.asarray(v0,float);F=np.asarray(force,float);n=len(v0)
    if not np.isfinite(dt) or dt<=0:raise ValueError('Positive dt')
    if any(x.shape!=(n,) or not np.isfinite(x).all() for x in (v0,F)):raise ValueError('Finite vectors')
    M=symmetric(M,n,'M',True);D=symmetric(D,n,'D');matrix=M/dt+D;rhs=M@v0/dt+F;checked=[]
    for m in memories:
        if m.born:raise ValueError('Fixed old material only')
        e=np.asarray(m.old,float);B=np.asarray(m.B,float);th=np.asarray(m.thermal_increment,float)
        if e.ndim!=1 or B.shape!=(len(e),n) or th.shape!=e.shape or not all(np.isfinite(x).all() for x in (e,B,th)):raise ValueError('Finite material fields')
        H=symmetric(m.H,len(e),'H',True);z=dt*m.relaxation_per_s;a,b,*_=weights(z)
        matrix=matrix+dt*b*(B.T@H@B);rhs=rhs-B.T@H@(a*e-b*th)
        checked.append((e,B,th,H,z))
    v=solve(matrix,rhs,assume_a='pos');du=dt*v;dv=v-v0
    U0=U1=heat=thermal=defect=0.;material=[];internal=np.zeros(n)
    for e,B,th,H,z in checked:
        mech=B@du;r=update(e,mech-th,H,z);end=r['endpoint'];mean=r['mean'];material.append(r)
        U0+=r['elastic_before_j'];U1+=r['elastic_after_j'];heat+=r['heat_j']
        thermal-=mean@H@th;defect+=(end-mean)@H@mech;internal+=B.T@H@end
    K0=.5*v0@M@v0;K1=.5*v@M@v;numK=.5*dv@M@dv;drag=dt*v@D@v;work=F@du
    residual=K1+U1-K0-U0-work-thermal+drag+heat+numK+defect
    return dict(velocity=v,memory=material,impulse_residual=M@dv-dt*(F-D@v-internal),
                ledger=dict(elastic_before_j=U0,elastic_after_j=U1,kinetic_before_j=K0,kinetic_after_j=K1,
                external_work_j=work,thermal_work_j=thermal,physical_drag_j=drag,physical_maxwell_j=heat,
                numerical_kinetic_j=numK,signed_coupling_defect_j=defect,balance_residual_j=residual))
