"""Fixed-geometry/constant-C coupled enthalpy, momentum and Maxwell equations.

Physical Maxwell/thermoelastic feedback matches genesis_cutcell_exchange;
drag remains a separate reservoir. No damage, birth, remelt or orbit integrator.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve
from tectonics.genesis_implicit_enthalpy import invert,phase
from analysis.genesis_backward_euler_prototype.core import symmetric


@dataclass(frozen=True)
class Block:
    B: np.ndarray
    H: np.ndarray
    relaxation: float
    expansion: np.ndarray
    node: int


class Coupled:
    def __init__(self,M,D,mass,cp,latent,solidus,liquidus,offset,edges,blocks,ports,regions):
        self.nv=len(M);self.nt=len(mass)
        self.M=symmetric(M,self.nv,'M',True);self.D=symmetric(D,self.nv,'D')
        self.mass,self.cp,self.latent,self.ts,self.tl,self.offset=[np.asarray(x,float) for x in (mass,cp,latent,solidus,liquidus,offset)]
        if any(x.shape!=(self.nt,) or not np.isfinite(x).all() for x in (self.mass,self.cp,self.latent,self.ts,self.tl,self.offset)):raise ValueError('THERMAL_FIELDS')
        if np.any(self.mass<=0) or np.any(self.cp<=0) or np.any(self.latent<0) or np.any(self.tl<=self.ts):raise ValueError('THERMAL_PARAMETERS')
        self.edges=tuple(edges);self.blocks=tuple(blocks);self.ports=ports;self.regions=tuple(regions)
        if len(regions)!=self.nt or any(r not in ('solid','mush','liquid') for r in regions):raise ValueError('PHASE_REGION')
        seen=set()
        for i,j,g in self.edges:
            if i==j or not 0<=i<self.nt or not 0<=j<self.nt or not np.isfinite(g) or g<=0 or frozenset((i,j)) in seen:raise ValueError('EDGE')
            seen.add(frozenset((i,j)))
        self.slices=[];cursor=self.nt+self.nv
        for m in self.blocks:
            n=len(m.H);symmetric(m.H,n,'H',True)
            if m.B.shape!=(n,self.nv) or m.expansion.shape!=(n,) or not np.isfinite(m.B).all() or not np.isfinite(m.expansion).all() or not 0<=m.node<self.nt or not np.isfinite(m.relaxation) or m.relaxation<0:raise ValueError('MEMORY_BLOCK')
            self.slices.append(slice(cursor,cursor+n));cursor+=n
        self.size=cursor

    def unpack(self,y):
        y=np.asarray(y,float)
        if y.shape!=(self.size,) or not np.isfinite(y).all():raise ValueError('STATE')
        return y[:self.nt],y[self.nt:self.nt+self.nv],[y[s] for s in self.slices]

    def thermal(self,E):return invert(E/self.mass,self.cp,self.latent,self.ts,self.tl,self.offset)

    def guard(self,y):
        E,_,_=self.unpack(y);T=self.thermal(E)
        if np.any(E<0) or np.any(T<=0):raise ValueError('NONPOSITIVE_THERMAL_STATE')
        for i,r in enumerate(self.regions):
            if (r=='solid' and T[i]>self.ts[i]) or (r=='mush' and not self.ts[i]<=T[i]<=self.tl[i]) or (r=='liquid' and T[i]<self.tl[i]):
                raise ValueError('PHASE_EVENT_REQUIRES_LOCALIZATION')

    def rates_and_force(self,t,y):
        E,v,mem=self.unpack(y);T=self.thermal(E)
        Q,F=self.ports(t,T,v,mem);Q=np.asarray(Q,float).copy();F=np.asarray(F,float)
        if Q.shape!=(self.nt,) or F.shape!=(self.nv,) or not np.isfinite(Q).all() or not np.isfinite(F).all():raise ValueError('PORTS')
        capacity=self.mass*(self.cp+self.latent/(self.tl-self.ts)*((T>self.ts)&(T<self.tl)))
        # At a phase endpoint, use the explicitly selected one-sided smooth region.
        for i,r in enumerate(self.regions):
            if r=='mush' and T[i] in (self.ts[i],self.tl[i]):capacity[i]=self.mass[i]*(self.cp[i]+self.latent[i]/(self.tl[i]-self.ts[i]))
        effective=capacity.copy();internal=np.zeros(self.nv)
        for i,j,g in self.edges:
            flux=g*(T[i]-T[j]);Q[i]-=flux;Q[j]+=flux
        for m,e in zip(self.blocks,mem):
            Q[m.node]+=m.relaxation*(e@m.H@e)
            effective[m.node]-=e@m.H@m.expansion
            internal+=m.B.T@m.H@e
        if np.any(effective<=0):raise ValueError('NONPOSITIVE_EFFECTIVE_CAPACITY')
        Tdot=Q/effective
        force=F-self.D@v-internal
        de=[m.B@v-m.expansion*Tdot[m.node]-m.relaxation*e for m,e in zip(self.blocks,mem)]
        return np.concatenate((capacity*Tdot,force,*de))

    def rhs(self,t,y):
        r=self.rates_and_force(t,y)
        r[self.nt:self.nt+self.nv]=solve(self.M,r[self.nt:self.nt+self.nv],assume_a='pos')
        return r

    def implicit_residual(self,t,y,base,gh,scale):
        # Keep momentum in mass form for thin cohorts; do not invert the nearly
        # singular inertia inside a finite-difference Newton residual.
        v=slice(self.nt,self.nt+self.nv)
        r=self.rates_and_force(t,y)
        residual=y-base-gh*r
        residual[v]=self.M@(y[v]-base[v])-gh*r[v]
        rows=scale.copy()
        matrix=np.abs(self.M)+gh*np.abs(self.D)
        for m in self.blocks:matrix+=gh*gh*np.abs(m.B.T@m.H@m.B)
        rows[v]=matrix@scale[v]
        return residual/rows

    def energy(self,y):
        E,v,mem=self.unpack(y)
        return float(E.sum()+.5*v@self.M@v+sum(.5*e@m.H@e for m,e in zip(self.blocks,mem)))
