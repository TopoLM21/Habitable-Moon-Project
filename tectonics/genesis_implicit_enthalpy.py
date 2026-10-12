"""Backward-Euler graph conduction in enthalpy, with piecewise-linear phase law."""
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import spsolve


def phase(T,ts,tl):return np.clip((T-ts)/(tl-ts),0.,1.)


def invert(h,cp,L,ts,tl,offset):
    x=h-offset
    return np.where(x<=cp*ts,x/cp,np.where(x>=cp*tl+L,(x-L)/cp,(x+L*ts/(tl-ts))/(cp+L/(tl-ts))))


def solve(mass,h0,cp,latent,solidus,liquidus,offset,edges,dt_seconds):
    mass,h0,cp,L,ts,tl,off=[np.asarray(x,dtype=float) for x in (mass,h0,cp,latent,solidus,liquidus,offset)]
    n=len(mass)
    if any(x.shape!=(n,) or not np.isfinite(x).all() for x in (mass,h0,cp,L,ts,tl,off)):raise ValueError('Finite complete fields')
    if np.any(mass<=0) or np.any(cp<=0) or np.any(L<0) or np.any(tl<=ts) or np.any(h0<0) or not np.isfinite(dt_seconds) or dt_seconds<=0:raise ValueError('Physical solver bounds')
    rows=[];cols=[];values=[];seen=set()
    for i,j,G in edges:
        if not (0<=i<n and 0<=j<n and i!=j and np.isfinite(G) and G>0) or frozenset((i,j)) in seen:raise ValueError('Invalid/duplicate edge')
        seen.add(frozenset((i,j)));rows.extend([i,j,i,j]);cols.extend([i,j,j,i]);values.extend([G,G,-G,-G])
    A=coo_matrix((values,(rows,cols)),shape=(n,n)).tocsr();scale=mass*cp
    T0=invert(h0,cp,L,ts,tl,off);T=T0.copy();phi0=phase(T0,ts,tl)
    if np.any(T0<0):raise ValueError('Negative temperature')
    ii=np.array([e[0] for e in edges],dtype=int);jj=np.array([e[1] for e in edges],dtype=int);G=np.array([e[2] for e in edges])
    tol=3e-11+8*np.spacing(max(1.,float(T0.max())))*(1+dt_seconds*A.diagonal()/scale)
    def residual(x):
        out=mass*(cp*(x-T0)+L*(phase(x,ts,tl)-phi0))
        flux=dt_seconds*G*(x[ii]-x[jj]);np.add.at(out,ii,flux);np.add.at(out,jj,-flux)
        return out/scale
    for iteration in range(40):
        r=residual(T)
        if np.max(np.abs(r)/tol)<=1:break
        capacity=mass*(cp+L/(tl-ts)*((T>ts)&(T<tl)))
        J=diags(1/scale)@(diags(capacity)+dt_seconds*A)
        direction=spsolve(J,-r);alpha=1.;norm=np.linalg.norm(r)
        for backtrack in range(30):
            trial=T+alpha*direction
            if np.all(trial>=0) and np.linalg.norm(residual(trial))<norm:break
            alpha*=.5
        else:raise ValueError('Newton line search failed')
        T=trial
    else:raise ValueError('Newton budget exhausted')
    if T.min()<T0.min()-1e-8 or T.max()>T0.max()+1e-8:raise ValueError('Maximum principle')
    h=cp*T+L*phase(T,ts,tl)+off
    exchanges=dt_seconds*G*(T[ii]-T[jj])
    return dict(temperature_k=T,specific_h=h,melt_fraction=phase(T,ts,tl),exchange_j=exchanges,
      iterations=iteration,residual_k=residual(T),tolerance_k=tol,initial_temperature_k=T0,
      mass_kg=mass.copy(),dt_seconds=dt_seconds)
