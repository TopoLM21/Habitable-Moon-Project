"""Spherical strip/triangle coarea integrals; no area renormalization.

At distance d from the trench, triangle half-spaces cut intervals out of the
trench coordinate s. Their exact lengths multiply cos(d/R) ds dd. Only
one-dimensional quadrature on smooth pieces remains.
"""
from functools import lru_cache
import math
import numpy as np


@lru_cache(maxsize=8)
def _gauss(order):return np.polynomial.legendre.leggauss(order)


def strip_intersection_integrals(triangles,segment,radius,sigma,outer,order):
    """Positive [source-weighted area, support area] for each material triangle."""
    order=int(order);count=len(triangles)
    if not count:return np.zeros((0,2))
    a=np.asarray(segment.start);b=np.asarray(segment.end)
    normal=np.cross(a,b);normal/=np.linalg.norm(normal)
    if normal@segment.toward<0:normal=-normal
    tangent=b-a*(a@b);tangent/=np.linalg.norm(tangent)
    length=math.atan2(np.linalg.norm(np.cross(a,b)),a@b)
    end_tangent=-math.sin(length)*a+math.cos(length)*tangent
    lo=max(0.,segment.offset_km-outer)/radius;hi=(segment.offset_km+outer)/radius
    if not 0<length<math.pi or not 0<=lo<hi<math.pi/2:raise ValueError('Invalid coarea strip geometry')
    normals=np.cross(triangles,np.roll(triangles,-1,axis=1));normals/=np.linalg.norm(normals,axis=2)[:,:,None]
    normals*=np.where(np.einsum('nij,nj->ni',normals,np.sum(triangles,axis=1))>=0,1.,-1.)[:,:,None]
    A=normals@a;B=normals@tangent;C=normals@normal
    # Critical latitudes: vertices, great-circle edge extrema, and intersections
    # with either end of the finite strip. Superfluous divisions are harmless.
    critical=[np.full((count,1),lo),np.full((count,1),hi),np.arcsin(np.clip(triangles@normal,-1,1))]
    extrema=np.arcsin(np.sqrt(np.maximum(0.,1-C*C)))
    critical.extend([extrema,-extrema])
    for direction in (tangent,end_tangent):
        p=np.cross(normals,direction);norm=np.linalg.norm(p,axis=2)
        p=np.divide(p,norm[:,:,None],out=np.zeros_like(p),where=norm[:,:,None]>1e-14)
        delta=np.arcsin(np.clip(p@normal,-1,1));critical.extend([delta,-delta])
    bounds=np.sort(np.clip(np.concatenate(critical,axis=1),lo,hi),axis=1)
    left=bounds[:,:-1];right=bounds[:,1:];take=right>left
    rows=np.broadcast_to(np.arange(count)[:,None],left.shape)[take]
    left=left[take];right=right[take]
    x,w=_gauss(order);delta=(left[:,None]+right[:,None])/2+(right-left)[:,None]*x/2
    aa=A[rows,None,:];bb=B[rows,None,:];cc=C[rows,None,:]
    magnitude=np.hypot(aa,bb);rhs=-cc*np.tan(delta[:,:,None])
    h=np.divide(rhs,magnitude,out=np.zeros_like(rhs),where=magnitude>1e-14)
    phase=np.arctan2(bb,aa);angle=np.arccos(np.clip(h,-1,1));valid=(np.abs(h)<1)&(magnitude>1e-14)
    roots=np.stack([(phase-angle)%(2*math.pi),(phase+angle)%(2*math.pi)],axis=-1)
    roots=np.where(valid[:,:,:,None]&(roots>0)&(roots<length),roots,length).reshape(len(rows),order,6)
    endpoints=np.sort(np.concatenate([np.zeros((len(rows),order,1)),roots,np.full((len(rows),order,1),length)],axis=2),axis=2)
    mid=(endpoints[:,:,:-1]+endpoints[:,:,1:])/2;span=np.diff(endpoints,axis=2)
    inside=np.all(aa[:,:,None,:]*np.cos(mid[:,:,:,None])+bb[:,:,None,:]*np.sin(mid[:,:,:,None])
                  +cc[:,:,None,:]*np.tan(delta[:,:,None,None])>=0,axis=3)
    arc=np.sum(span*inside,axis=2)
    measure=radius**2*np.cos(delta)*arc*w[None,:]*(right-left)[:,None]/2
    phi=segment.amplitude*np.exp(-.5*((delta*radius-segment.offset_km)/sigma)**2)
    pieces=np.column_stack([np.sum(measure*phi,axis=1),np.sum(measure,axis=1)])
    result=np.zeros((count,2));np.add.at(result,rows,pieces)
    if np.any(~np.isfinite(result)) or np.any(result<0):raise ValueError('Invalid coarea intersection integral')
    return result
