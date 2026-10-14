"""Explicit geometric roundoff budgets; material amounts are never discarded.

Binary64 vertex error is budgeted as 64 ulps per boundary construction stage.
The spherical boundary tube has area <= 2*perimeter*delta + pi*delta**2.
This is an engineering forward-error budget, not an exact predicate proof.
The independent cell-partition audit remains mandatory.
"""
import math
import numpy as np

def area_budget(vertices,radius,stages=1):
    v=np.asarray(vertices,dtype=np.longdouble)
    if len(v)<3:return 0.
    v=v/np.sqrt(np.sum(v*v,axis=1))[:,None]
    w=np.roll(v,-1,axis=0)
    perimeter=float(np.sum(np.arctan2(np.sqrt(np.sum(np.cross(v,w)**2,axis=1)),np.sum(v*w,axis=1))))
    delta=64*np.finfo(float).eps*max(1,stages)
    return (2*perimeter*delta+math.pi*delta*delta)*radius*radius

def conservative_weights(areas,expected,budget):
    values=np.asarray(areas,dtype=float);total=math.fsum(values)
    if expected<=0 or total<=0 or np.any(values<0) or not np.isfinite(values).all():
        raise ValueError('No positive geometric support for material allocation')
    error=total-expected
    if abs(error)>budget:raise ValueError(f'Geometric closure exceeds boundary budget: {error} > {budget}')
    weights=values/total
    largest=int(np.argmax(weights));weights[largest]=1.-math.fsum(w for i,w in enumerate(weights) if i!=largest)
    return weights,error

def convex_roundoff(vertices,radius):
    """Repair a rounded convex boundary only inside its area error budget.

    A genuine concavity or bow tie exceeds the budget and remains an error.
    """
    from scipy.spatial import ConvexHull
    from .spherical_polygons import polygon_area,normalize_polygon
    v=np.asarray(vertices,dtype=float);centre=v.sum(0);centre/=np.linalg.norm(centre)
    axis=np.eye(3)[np.argmin(abs(centre))];u=np.cross(centre,axis);u/=np.linalg.norm(u);w=np.cross(centre,u)
    if np.any(v@centre<=0):raise ValueError('Convex roundoff repair requires a hemisphere')
    xy=np.c_[v@u,v@w]/(v@centre)[:,None]
    hull=ConvexHull(xy);out=normalize_polygon(v[hull.vertices])
    difference=abs(polygon_area(out)-polygon_area(v))*radius**2
    if difference>area_budget(v,radius,stages=len(v)+2):
        raise ValueError(f'Nonconvex polygon exceeds roundoff repair budget: {difference} vs {area_budget(v,radius,stages=len(v)+2)}; vertices={v.tolist()}')
    return out
