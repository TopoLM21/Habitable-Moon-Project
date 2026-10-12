"""Opt-in backward Euler for fixed-coefficient mechanical trials; no runtime hook.

M dv/dt + D v + sum(B.T H e) = F, de/dt = Bv - thermal_rate - gamma e.
Birth follows the existing thermal-first, comoving-v0, zero-memory convention.
Remelt, damage changes and geometry certification deliberately have no adapter.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve, cholesky


@dataclass(frozen=True)
class Memory:
    B: np.ndarray
    H: np.ndarray
    old: np.ndarray
    thermal_increment: np.ndarray
    relaxation_per_s: float
    born: bool = False


def symmetric(a, n, name, positive=False):
    a=np.asarray(a,float)
    if a.shape!=(n,n) or not np.isfinite(a).all():raise ValueError(name+' dimensions/finite')
    if not np.allclose(a,a.T,rtol=0,atol=32*np.finfo(float).eps*max(np.linalg.norm(a,2),1e-300)):
        raise ValueError(name+' symmetry')
    if positive:cholesky(a,lower=True)
    elif np.linalg.eigvalsh(a)[0]<-64*np.finfo(float).eps*max(np.linalg.norm(a,2),1e-300):
        raise ValueError(name+' must be positive semidefinite')
    return a


def step(M_before,D,force,q0,v0,dt,memories=(),birth_mass=None):
    """Actual prototype solve, update and explicit physical/numerical ledgers.

    Force is sampled at the endpoint; D,H,gamma fixed within this trial.
    All quantities SI. Returns new arrays; never mutates supplied history.
    """
    v0=np.asarray(v0,float);q0=np.asarray(q0,float);F=np.asarray(force,float);n=len(v0)
    if not np.isfinite(dt) or dt<=0:raise ValueError('Positive dt required')
    if any(x.shape!=(n,) or not np.isfinite(x).all() for x in (v0,q0,F)):raise ValueError('Vector dimensions/finite')
    M0=symmetric(M_before,n,'M before',True);D=symmetric(D,n,'D')
    Min=np.zeros_like(M0) if birth_mass is None else symmetric(birth_mass,n,'birth mass')
    M=M0+Min
    A=M/dt+D;rhs=M@v0/dt+F;checked=[]
    for m in memories:
        B=np.asarray(m.B,float);e=np.asarray(m.old,float);t=np.asarray(m.thermal_increment,float)
        if B.shape!=(len(e),n) or t.shape!=e.shape or not all(np.isfinite(x).all() for x in (B,e,t)):
            raise ValueError('Memory dimensions/finite')
        H=symmetric(m.H,len(e),'H',True);g=m.relaxation_per_s
        if not np.isfinite(g) or g<0:raise ValueError('Nonnegative relaxation required')
        if m.born and (np.any(e) or g!=0):raise ValueError('Birth must use existing zero-memory, zero-first-step-relaxation convention')
        a=1/(1+dt*g)
        A=A+dt*a*(B.T@H@B);rhs=rhs-B.T@H@(a*(e-t))
        checked.append((B,H,e,t,g,a))
    scale=1/np.sqrt(np.diag(A))
    v=scale*solve(A*scale[:,None]*scale[None,:],scale*rhs,assume_a='pos',check_finite=True)
    du=dt*v;q=q0+du;dv=v-v0
    energy0=energy1=visc=nummemory=thermal=internalwork=0.;internal=np.zeros(n);new=[]
    for B,H,e,t,g,a in checked:
        e1=a*(e+B@du-t);de=e1-e;new.append(e1)
        energy0+=.5*e@H@e;energy1+=.5*e1@H@e1
        visc+=dt*g*(e1@H@e1);nummemory+=.5*de@H@de
        thermal-=e1@H@t;internal+=B.T@H@e1;internalwork+=e1@H@(B@du)
    K0=.5*v0@M0@v0;K1=.5*v@M@v;birthK=.5*v0@Min@v0
    numK=.5*dv@M@dv;drag=dt*v@D@v;external=F@du
    residual=K1+energy1-K0-energy0-birthK-external-thermal+drag+visc+numK+nummemory
    impulse=M@v-M0@v0-Min@v0-dt*(F-D@v-internal)
    impulse_scale=np.linalg.norm(M,2)*(np.linalg.norm(v)+np.linalg.norm(v0))+dt*(np.linalg.norm(F)+np.linalg.norm(D,2)*np.linalg.norm(v))
    impulse_scale+=dt*sum(np.linalg.norm(B.T@H,2)*np.linalg.norm(e1) for (B,H,*_),e1 in zip(checked,new))
    ledger=dict(kinetic_before_j=float(K0),kinetic_after_j=float(K1),elastic_before_j=float(energy0),
        elastic_after_j=float(energy1),birth_kinetic_inflow_j=float(birthK),external_work_j=float(external),
        thermal_work_j=float(thermal),physical_drag_j=float(drag),physical_maxwell_j=float(visc),
        numerical_kinetic_j=float(numK),numerical_memory_j=float(nummemory),
        internal_work_j=float(internalwork),balance_residual_j=float(residual))
    return dict(displacement=q,velocity=v,memory=new,ledger=ledger,impulse_residual=impulse,
                birth_momentum=Min@v0,internal_force=internal,
                impulse_relative_residual=float(np.linalg.norm(impulse)/max(impulse_scale,np.finfo(float).tiny)))


def conical_memory(B0,B1,old0,old1,thermal0,C,volume,mean,variance,relaxation_per_s,born=False):
    """Reuse Genesis conical B and exact radial mean/variance energy metric.

    Stable coordinates: (e0+mean*e1, sqrt(variance)*e1), not two cancelling
    thin-band coefficients. This adapter does not prepare a Genesis time step.
    """
    if volume<=0 or variance<0 or mean<=0:raise ValueError('Invalid radial moments')
    B0,B1,old0,old1,t=map(np.asarray,(B0,B1,old0,old1,thermal0))
    root=np.sqrt(variance)
    B=np.concatenate(((B0+mean*B1).reshape(18,-1),(root*B1).reshape(18,-1)))
    e=np.concatenate(((old0+mean*old1).ravel(),(root*old1).ravel()))
    thermal=np.concatenate((t.ravel(),np.zeros(18)))
    H=volume/3*np.kron(np.eye(6),C)
    return Memory(B,H,e,thermal,relaxation_per_s,born)


def conical_endpoint(B0,B1,old0,old1,thermal0,increment,dt,relaxation_per_s):
    """Recover original strain0/strain1 without dividing by tiny variance."""
    a=1/(1+dt*relaxation_per_s)
    return (a*(np.asarray(old0)+np.asarray(B0)@increment-np.asarray(thermal0)),
            a*(np.asarray(old1)+np.asarray(B1)@increment))
