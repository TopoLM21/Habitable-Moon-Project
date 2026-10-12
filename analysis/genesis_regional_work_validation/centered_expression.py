"""Exact expression graph of one source-bound scalar virtual-work density.

Compact displacement bubble is differentiated symbolically. No radial
integration, numerical gradients, tensor assembler, or mechanical solver.
"""
from fractions import Fraction as F
import numpy as np
from analysis.genesis_volume_operand_validation.independent_coefficients import _literal_vertices,_inverse3,rational_array


class Graph:
    def __init__(self):self.nodes=[];self.cache={};self.derivatives={}
    def node(self,op,*args):
        key=(op,*args)
        if key not in self.cache:self.cache[key]=len(self.nodes);self.nodes.append(key)
        return self.cache[key]
    def const(self,x):return self.node('c',str(F(x)))
    def value(self,i):return F(self.nodes[i][1]) if self.nodes[i][0]=='c' else None
    def add(self,a,b):
        av,bv=self.value(a),self.value(b)
        if av==0:return b
        if bv==0:return a
        if av is not None and bv is not None:return self.const(av+bv)
        return self.node('+',*sorted((a,b)))
    def mul(self,a,b):
        av,bv=self.value(a),self.value(b)
        if av==0 or bv==0:return self.const(0)
        if av==1:return b
        if bv==1:return a
        if av is not None and bv is not None:return self.const(av*bv)
        return self.node('*',*sorted((a,b)))
    def scale(self,a,b):return self.mul(a,self.const(b))
    def sub(self,a,b):return self.add(a,self.scale(b,-1))
    def sum(self,values):
        out=self.const(0)
        for value in values:out=self.add(out,value)
        return out
    def diff(self,i,axis):
        key=(i,axis)
        if key in self.derivatives:return self.derivatives[key]
        n=self.nodes[i]
        if n[0]=='c':out=self.const(0)
        elif n[0]=='x':out=self.const(int(n[1]==axis))
        elif n[0]=='+':out=self.add(self.diff(n[1],axis),self.diff(n[2],axis))
        else:out=self.add(self.mul(self.diff(n[1],axis),n[2]),self.mul(n[1],self.diff(n[2],axis)))
        self.derivatives[key]=out;return out


def build(raw,probe,geometry_center):
    if any(probe.get('mean_z_m',[0]*18)):raise ValueError('Scope requires zero mean added memory')
    g=Graph();u,v,lam=[g.node('x',i) for i in range(3)];one=g.const(1)
    N=[g.sub(g.sub(one,u),v),u,v];section=raw['section']
    X,A,P=[np.asarray(section[k],float) for k in ('triangle_m','displacement_A_m','slope_P_m')]
    # Preserve the frozen represented sums used by the original source.
    C,D=rational_array(X+P),rational_array(A-P);Xr=rational_array(X)
    Q=rational_array(raw['basis_xyz']);center=list(map(F,raw['seed_center_world_m']));R=F(raw['seed_radius_m'])
    uc,vc,lc=geometry_center
    nc=[1-uc-vc,uc,vc]
    du0,dv0,dl0=[g.sub(x,g.const(c)) for x,c in zip((u,v,lam),geometry_center)]
    offsets=[]
    for j in range(3):
        base=sum(nc[i]*(lc*C[i,j]+D[i,j]) for i in range(3))-center[j]
        xu=lc*(C[1,j]-C[0,j])+D[1,j]-D[0,j]
        xv=lc*(C[2,j]-C[0,j])+D[2,j]-D[0,j]
        xl=sum(nc[i]*C[i,j] for i in range(3))
        offsets.append(g.sum([g.const(base),g.scale(du0,xu),g.scale(dv0,xv),g.scale(dl0,xl),
            g.scale(g.mul(du0,dl0),C[1,j]-C[0,j]),g.scale(g.mul(dv0,dl0),C[2,j]-C[0,j])]))
    distance=g.sum(g.mul(x,x) for x in offsets);sphere=g.sub(distance,g.const(R*R))
    normal=list(map(F,section['normal_xyz']))
    # Collapse plane coefficients before interval evaluation; exact bilinear identity.
    cv=[sum(C[i,j]*normal[j] for j in range(3)) for i in range(3)]
    dv=[sum(D[i,j]*normal[j] for j in range(3)) for i in range(3)]
    pc=sum(nc[i]*(lc*cv[i]+dv[i]) for i in range(3))-F(section['offset_m'])
    phi=g.sum([g.const(pc),g.scale(du0,lc*(cv[1]-cv[0])+dv[1]-dv[0]),
        g.scale(dv0,lc*(cv[2]-cv[0])+dv[2]-dv[0]),g.scale(dl0,sum(nc[i]*cv[i] for i in range(3))),
        g.scale(g.mul(du0,dl0),cv[1]-cv[0]),g.scale(g.mul(dv0,dl0),cv[2]-cv[0])])
    lo,hi=map(F,section['lambda_bounds']);r=g.scale(g.sub(lam,g.const(lo)),1/(hi-lo))
    radial=g.scale(g.mul(r,g.sub(one,r)),4);angular=g.scale(g.mul(g.mul(N[0],N[1]),N[2]),27)
    S=g.sub(one,g.scale(distance,1/(R*R)));w=g.mul(g.mul(angular,radial),g.mul(S,S))
    du,dv,dl=[g.diff(w,i) for i in range(3)]
    inv=_inverse3(np.column_stack((Xr[1]-Xr[0],Xr[2]-Xr[0],Xr[0])))
    lp=g.sub(g.mul(lam,dl),g.add(g.mul(u,du),g.mul(v,dv)))
    L=[g.sum([g.scale(du,inv[0,j]),g.scale(dv,inv[1,j]),g.scale(lp,inv[2,j])]) for j in range(3)]
    a=list(map(F,probe['enrichment_increment_m']))
    velocity=[g.sum(g.mul(N[i],g.add(g.const(a[6*i+j]),g.scale(g.sub(lam,one),a[6*i+3+j]))) for i in range(3)) for j in range(3)]
    lt=[g.sum(g.scale(velocity[j],Q[j,i]) for j in range(3)) for i in range(3)]
    lg=[g.sum(g.scale(L[j],Q[j,i]) for j in range(3)) for i in range(3)]
    B0,B1=_literal_vertices(X,np.asarray(raw['basis_xyz'],float))
    ba=[]
    for component in range(6):
        b0=g.sum(g.scale(N[i],sum(B0[i,component,j]*a[j] for j in range(18))) for i in range(3))
        b1=g.sum(g.scale(N[i],sum(B1[i,component,j]*a[j] for j in range(18))) for i in range(3))
        ba.append(g.add(g.mul(lam,b0),b1))
    Ga=[]
    for k,(i,j) in enumerate(((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))):
        extra=g.mul(lt[i],lg[j])
        if i!=j:extra=g.add(extra,g.mul(lt[j],lg[i]))
        Ga.append(g.add(g.mul(w,ba[k]),extra))
    sample=np.array(((2/3,1/6,1/6),(1/6,2/3,1/6),(1/6,1/6,2/3)))
    ainv=rational_array(np.linalg.inv(sample));e0=rational_array(probe.get('mean_e0',raw['memory0']));e1=rational_array(probe.get('mean_e1',raw['memory1']))
    m0,m1=ainv@e0,ainv@e1
    memory=[g.sum(g.mul(N[i],g.add(g.scale(lam,m0[i,j]),g.const(m1[i,j]))) for i in range(3)) for j in range(6)]
    elasticity=rational_array(raw['elasticity_pa'])
    density=g.sum(g.mul(Ga[i],g.sum(g.scale(memory[j],elasticity[i,j]) for j in range(6))) for i in range(6))
    det=sum(Xr[0,j]*(Xr[1,(j+1)%3]*Xr[2,(j+2)%3]-Xr[1,(j+2)%3]*Xr[2,(j+1)%3]) for j in range(3))
    if det<=0:raise ValueError('Positive reference measure required')
    density=g.scale(density,det)
    return dict(nodes=g.nodes,outputs=dict(sphere=sphere,gamma=phi,density_without_H=density),
                geometry_center=list(map(str,geometry_center)),variables=['N1','N2','lambda'],measure='dN1 dN2 dlambda',units='J per dN1 dN2 dlambda',
                virtual_work_only=True,mean_added_memory_zero=True)


def interval_evaluate(graph,box):
    values=[]
    for node in graph['nodes']:
        op=node[0]
        if op=='c':x=F(node[1]);value=(x,x)
        elif op=='x':value=box[node[1]]
        elif op=='+':
            a,b=values[node[1]],values[node[2]];value=(a[0]+b[0],a[1]+b[1])
        elif op=='*':
            a,b=values[node[1]],values[node[2]];products=[x*y for x in a for y in b];value=(min(products),max(products))
        else:raise ValueError('Unknown graph operation')
        values.append(value)
    return {k:values[i] for k,i in graph['outputs'].items()}


def block_bound(graph,box):
    ranges=interval_evaluate(graph,box);q=ranges['sphere'];phi=ranges['gamma'];lo,hi=ranges['density_without_H']
    if q[0]>0:return (F(),F()),'outside',ranges
    if phi[0]>0:lo,hi=lo/2,hi/2
    elif phi[1]<0:lo,hi=-hi/2,-lo/2
    else:upper=max(abs(lo),abs(hi))/2;lo,hi=-upper,upper
    state='inside' if q[1]<=0 else 'unknown_sphere'
    if state=='unknown_sphere':lo,hi=min(F(),lo),max(F(),hi)
    volume=F(1)
    for a,b in box:volume*=b-a
    return (volume*lo,volume*hi),state,ranges
