"""Finite-width volcanic production integrated over physical source geometry.

Active sources are oriented geodesic trench segments with a fixed Gaussian
cross-profile. Independent segments contribute additively (not a maximum over
mesh samples). This is an explicit effective line-source approximation. Rock
production/enthalpy are still handled by the continental cycle/finite reservoir.
"""
from dataclasses import dataclass
import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ArcSegment:
    start: np.ndarray
    end: np.ndarray
    toward: np.ndarray
    overriding_plate: int
    amplitude: float
    offset_km: float


class SphereReceiver:
    """Locate physical quadrature points in actual spherical triangles.

    Nearest-centroid sampling is only a candidate search. Triangle containment
    decides ownership; candidate search is expanded or fails explicitly.
    """
    def __init__(self,mesh):
        self.mesh=mesh;self.tree=cKDTree(mesh.centroids)
        tri=mesh.vertices[mesh.faces]
        normals=np.cross(tri,np.roll(tri,-1,axis=1))
        normals/=np.linalg.norm(normals,axis=2)[:,:,None]
        normals*=np.where(np.einsum('nij,nj->ni',normals,mesh.centroids)>=0,1.,-1.)[:,:,None]
        self.normals=normals

    def locate(self,points):
        points=np.asarray(points,dtype=float);result=np.full(len(points),-1,np.int32)
        for k in (4,16,64):
            todo=np.flatnonzero(result<0)
            if not len(todo):break
            _,candidates=self.tree.query(points[todo],k=min(k,self.mesh.cell_count))
            if candidates.ndim==1:candidates=candidates[:,None]
            dots=np.einsum('nkei,ni->nke',self.normals[candidates],points[todo])
            inside=np.all(dots>=-2e-13,axis=2)
            good=np.any(inside,axis=1)
            result[todo[good]]=candidates[good,np.argmax(inside[good],axis=1)]
        if np.any(result<0):raise ValueError('Physical arc quadrature point could not be located')
        return result


def strip_quadrature(segment,radius_km,sigma_km,outer_km,spacing_km=200.,order=8):
    """Spherical Fermi coordinates, Jacobian cos(d/R) ds dd, on the upper side.

    Trench arc length is exact. Subdivision uses a physical length, never the
    receiving mesh resolution. The lower profile tail stops at the trench.
    """
    a=np.asarray(segment.start,float);a=a/np.linalg.norm(a)
    b=np.asarray(segment.end,float);b=b/np.linalg.norm(b)
    angle=math.atan2(float(np.linalg.norm(np.cross(a,b))),float(a@b))
    if not 0<angle<math.pi-1e-8:raise ValueError('Arc segment must be a nondegenerate minor geodesic')
    normal=np.cross(a,b);normal/=np.linalg.norm(normal)
    if normal@segment.toward<0:normal=-normal
    tangent=(b-a*float(a@b))/math.sin(angle)
    lo=max(0.,segment.offset_km-outer_km);hi=segment.offset_km+outer_km
    if not 0<=lo<hi<math.pi*radius_km/2:raise ValueError('Arc profile outside the oriented strip domain')
    nodes,weights=leggauss(int(order))
    panels=max(1,math.ceil(angle*radius_km/spacing_km))
    s=((np.arange(panels)[:,None]+(nodes+1)[None,:]/2)*angle/panels).ravel()
    ws=np.tile(weights,panels)*(angle*radius_km/(2*panels))
    d=lo+(nodes+1)*(hi-lo)/2
    wd=weights*(hi-lo)/2*np.cos(d/radius_km)*np.exp(-.5*((d-segment.offset_km)/sigma_km)**2)
    center=np.cos(s)[:,None]*a+np.sin(s)[:,None]*tangent
    points=(center[:,None,:]*np.cos(d/radius_km)[None,:,None]
            +normal[None,None,:]*np.sin(d/radius_km)[None,:,None]).reshape(-1,3)
    weighted=(ws[:,None]*wd[None,:]*segment.amplitude).ravel()
    return points,weighted


def cap_quadrature(center,radius_km,sigma_km,outer_km,amplitude,spacing_km=200.,order=8):
    """Physical cap integral for the existing localized post-breakoff pulse."""
    c=np.array(center,dtype=float,copy=True);c/=np.linalg.norm(c)
    axis=np.array([0.,0.,1.]) if abs(c[2])<.9 else np.array([1.,0.,0.])
    e=np.cross(c,axis);e/=np.linalg.norm(e);f=np.cross(c,e)
    x,w=leggauss(int(order));d=(x+1)*outer_km/2
    wd=w*outer_km/2*radius_km*np.sin(d/radius_km)*np.exp(-.5*(d/sigma_km)**2)*amplitude
    count=max(16,math.ceil(2*math.pi*outer_km/spacing_km))
    theta=(np.arange(count)+.5)*(2*math.pi/count)
    tangent=np.cos(theta)[:,None]*e+np.sin(theta)[:,None]*f
    points=(np.cos(d/radius_km)[None,:,None]*c+np.sin(d/radius_km)[None,:,None]*tangent[:,None,:]).reshape(-1,3)
    return points,np.tile(wd,count)*(2*math.pi/count)


def integrate_sources(mesh,owner,segments,radius_km,sigma_km,outer_km,spacing_km=200.,order=8,caps=()):
    """Deposit integrated source weights once; retain excluded-region budget."""
    if min(radius_km,sigma_km,outer_km,spacing_km)<=0 or not 2<=order<=32:
        raise ValueError('Invalid integral arc quadrature settings')
    receiver=SphereReceiver(mesh);owners=np.asarray(owner);field=np.zeros(mesh.cell_count)
    total=excluded=0.;count=0
    inputs=[]
    for segment in segments:
        if not math.isfinite(segment.amplitude) or segment.amplitude<0:raise ValueError('Invalid arc amplitude')
        points,weights=strip_quadrature(segment,radius_km,sigma_km,outer_km,spacing_km,order)
        inputs.append((points,weights,segment.overriding_plate))
    for center,plate,amplitude,cap_sigma,cap_outer in caps:
        points,weights=cap_quadrature(center,radius_km,cap_sigma,cap_outer,amplitude,spacing_km,order)
        inputs.append((points,weights,plate))
    for points,weights,plate in inputs:
        cells=receiver.locate(points);mask=owners[cells]==plate
        field+=np.bincount(cells[mask],weights=weights[mask],minlength=mesh.cell_count)
        total+=math.fsum(weights);excluded+=math.fsum(weights[~mask]);count+=len(points)
    error=math.fsum([*field,excluded,-total])
    if abs(error)>64*math.ulp(max(total,1.)):raise ValueError('Arc source deposition budget failed')
    return field,{'source_integral_km2':total,'outside_overriding_integral_km2':excluded,
                  'deposition_residual_km2':error,'quadrature_points':count}
