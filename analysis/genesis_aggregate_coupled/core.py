"""BE update of reduced Maxwell blocks, with the unchanged nodal impulse equation."""
import numpy as np
from scipy.linalg import solve


def step(M,D,F,v0,dt,blocks):
    if not np.isfinite(dt) or dt<=0: raise ValueError('positive dt')
    A=M/dt+D;rhs=M@v0/dt+F;prepared=[]
    for b in blocks:
        H,L,R,e,theta,idx=(b[k] for k in ('H','L','R','old','theta','dofs'))
        # H R is symmetric PSD for a projected positive Maxwell relaxation.
        S=np.linalg.solve(np.eye(len(e))+dt*R,np.eye(len(e)))
        HS=H@S
        if not np.allclose(HS,HS.T,rtol=1e-10,atol=1e-12*np.max(np.abs(HS))): raise ValueError('nonreciprocal relaxation')
        A[np.ix_(idx,idx)]+=dt*L.T@HS@L
        np.add.at(rhs,idx,-L.T@HS@(e-theta));prepared.append((b,S))
    scale=1/np.sqrt(np.diag(A))
    v=scale*solve(A*scale[:,None]*scale[None,:],scale*rhs,assume_a='pos')
    du=dt*v;mem=[];force=np.zeros_like(v);rows=[]
    for b,S in prepared:
        H,L,R,e,theta,idx=(b[k] for k in ('H','L','R','old','theta','dofs'))
        inc=L@du[idx];new=S@(e+inc-theta);de=new-e;mem.append(new)
        np.add.at(force,idx,L.T@H@new)
        rows.append(dict(elastic_before=float(.5*e@H@e),elastic_after=float(.5*new@H@new),
            maxwell=float(dt*new@H@R@new),thermal_work=float(-new@H@theta),
            numerical_memory=float(.5*de@H@de),internal_work=float(new@H@inc)))
    totals={k:sum(r[k] for r in rows) for k in rows[0]}
    K0=float(.5*v0@M@v0);K1=float(.5*v@M@v)
    totals.update(kinetic_before=K0,kinetic_after=K1,drag=float(dt*v@D@v),external_work=float(F@du),numerical_kinetic=float(.5*(v-v0)@M@(v-v0)))
    totals['balance']=K1-K0+totals['elastic_after']-totals['elastic_before']-totals['external_work']-totals['thermal_work']+totals['maxwell']+totals['numerical_memory']+totals['numerical_kinetic']+totals['drag']
    residual=M@(v-v0)/dt+D@v+force-F
    totals['momentum_relative']=float(np.linalg.norm(residual)/max(np.linalg.norm(force),np.linalg.norm(F),1.))
    totals['balance_relative']=abs(totals['balance'])/max(max(abs(x) for x in totals.values()),1.)
    if min(totals[k] for k in ['maxwell','numerical_memory','numerical_kinetic','drag']) < -1e-12: raise ValueError('negative dissipation')
    return dict(velocity=v,displacement_increment=du,memory=np.array(mem),internal_force=force,rows=rows,ledger=totals)
