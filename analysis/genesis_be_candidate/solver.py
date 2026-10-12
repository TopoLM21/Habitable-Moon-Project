"""Batched conical BE solver, equivalent to the opt-in block prototype."""
import numpy as np
from scipy.linalg import solve
from tectonics.genesis_volume_loading import coefficient_inner_product
from analysis.genesis_backward_euler_prototype.core import conical_memory


def solve_trial(M,D,F,v0,dt,dofs,B0,B1,loading):
    M=np.asarray(M);D=np.asarray(D);n=len(v0);A=M/dt+D;rhs=M@v0/dt+F
    for i,idx in enumerate(dofs):
        m=conical_memory(B0[i],B1[i],loading.old0[i],loading.old1[i],loading.thermal0[i],
            loading.elasticity_pa[i],loading.radial_moments.volume_m3[i],
            loading.radial_moments.inverse_radius_mean[i],loading.radial_moments.inverse_radius_variance[i],loading.ratio[i]/dt)
        a=1/(1+loading.ratio[i]);A[np.ix_(idx,idx)]+=dt*a*(m.B.T@m.H@m.B)
        np.add.at(rhs,idx,-m.B.T@m.H@(a*(m.old-m.thermal_increment)))
    scale=1/np.sqrt(np.diag(A));v=scale*solve(A*scale[:,None]*scale[None,:],scale*rhs,assume_a='pos')
    du=dt*v;local=du[dofs];inc0=np.einsum('cqai,ci->cqa',B0,local);inc1=np.einsum('cqai,ci->cqa',B1,local)
    a=1/(1+loading.ratio[:,None,None]);e0=a*(loading.old0+inc0-loading.thermal0);e1=a*(loading.old1+inc1)
    inner=lambda a0,a1,b0,b1:coefficient_inner_product(a0,a1,b0,b1,loading.elasticity_pa,loading.radial_moments)
    de0=e0-loading.old0;de1=e1-loading.old1;zero=np.zeros_like(e0)
    energy=.5*inner(e0,e1,e0,e1).sum();initial=.5*inner(loading.old0,loading.old1,loading.old0,loading.old1).sum()
    ledger=dict(elastic_after_j=float(energy),elastic_before_after_damage_j=float(initial),
        internal_work_j=float(inner(e0,e1,inc0,inc1).sum()),thermal_work_j=float(-inner(e0,e1,loading.thermal0,zero).sum()),
        physical_maxwell_j=float(np.dot(loading.ratio,inner(e0,e1,e0,e1))),numerical_memory_j=float(.5*inner(de0,de1,de0,de1).sum()),
        numerical_kinetic_j=float(.5*(v-v0)@M@(v-v0)),physical_drag_j=float(dt*v@D@v),external_work_j=float(F@du),
        kinetic_after_j=float(.5*v@M@v))
    return du,v,e0,e1,ledger
