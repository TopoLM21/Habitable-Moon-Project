"""Instantaneous conservative reduced-state projection; no time integrator."""
import numpy as np


def moments(weights, coordinate, values, metric):
    """Weighted affine projection; retains zeroth/first work-conjugate moments.

    Metric is constant SPD within a group. Coordinates are centered/scaled for
    conditioning; the returned slope is per physical coordinate unit.
    """
    w=np.asarray(weights,float); z=np.asarray(coordinate,float); x=np.asarray(values,float)
    C=np.asarray(metric,float)
    if w.ndim!=1 or z.shape!=w.shape or x.ndim!=2 or len(x)!=len(w) or C.shape!=(x.shape[1],x.shape[1]): raise ValueError('shape')
    if not all(np.isfinite(a).all() for a in (w,z,x,C)) or np.any(w<=0): raise ValueError('finite positive weights')
    if not np.allclose(C,C.T,rtol=0,atol=0): raise ValueError('symmetric metric')
    np.linalg.cholesky(C)
    center=float(w@z/w.sum()); dz=z-center; scale=float(np.max(np.abs(dz)))
    B=np.ones((len(w),1)) if scale==0 else np.column_stack((np.ones(len(w)),dz/scale))
    coeff=np.linalg.solve(B.T@(w[:,None]*B),B.T@(w[:,None]*x))
    fit=B@coeff; residual=x-fit
    norm=lambda a:float(np.einsum('n,ni,ij,nj->',w,a,C,a))
    before=.5*norm(x); retained=.5*norm(fit); lost=.5*norm(residual)
    return dict(center=center,mean=coeff[0],slope=np.zeros(x.shape[1]) if scale==0 else coeff[1]/scale,
        fit=fit,residual=residual,energy_before=before,energy_retained=retained,energy_residual=lost,
        energy_balance=before-retained-lost,
        zeroth_error=(w[:,None]*residual).sum(axis=0),first_error=(w[:,None]*dz[:,None]*residual).sum(axis=0))


def thermal_bins(mass,enthalpy,groups,count):
    """Conservative specific-enthalpy means, not temperature averages."""
    m=np.asarray(mass,float); E=np.asarray(enthalpy,float); g=np.asarray(groups)
    if m.shape!=E.shape or g.shape!=m.shape or g.dtype.kind not in 'iu' or np.any(g<0) or np.any(g>=count): raise ValueError('groups')
    if not np.isfinite(m).all() or not np.isfinite(E).all() or np.any(m<=0): raise ValueError('inventory')
    M=np.bincount(g,weights=m,minlength=count); H=np.bincount(g,weights=E,minlength=count)
    if np.any(M==0): raise ValueError('empty group')
    h=H/M; residual=E-m*h[g]
    return dict(mass=M,enthalpy=H,specific_enthalpy=h,residual_enthalpy=residual,
                mass_residual=float(m.sum()-M.sum()),energy_residual=float(E.sum()-H.sum()))


def rigid_motion(M,positions,velocity):
    """Six global modes retain linear and angular momentum; never create plates."""
    p=np.asarray(positions,float);v=np.asarray(velocity,float)
    B=np.zeros((p.size,6));T=np.tile(np.eye(3),(len(p),1));B[:,:3]=T
    radius=float(np.max(np.linalg.norm(p,axis=1)))
    for k in range(3): B[:,3+k]=np.cross(np.eye(3)[k],p).ravel()/radius
    fit=B@np.linalg.solve(B.T@M@B,B.T@M@v);r=v-fit
    return dict(fit=fit,residual=r,moment_residual=B.T@M@r,
                kinetic_before=float(.5*v@M@v),kinetic_retained=float(.5*fit@M@fit),kinetic_residual=float(.5*r@M@r))
