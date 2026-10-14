"""Convex great-circle clipping in a receiving cell's gnomonic chart.

Great circles become straight lines. Extended arithmetic on local coordinate
*differences* avoids reconstructing planes from nearly coincident global rays.
Output remains ordinary binary64; material closure uses a separate area budget.
"""
import numpy as np
LD=np.longdouble

def frame(centre):
    c=np.asarray(centre,dtype=LD);c/=np.sqrt(np.dot(c,c))
    u=np.cross(c,np.eye(3,dtype=LD)[np.argmin(abs(c))]);u/=np.sqrt(np.dot(u,u));return c,u,np.cross(c,u)

def project(v,basis):
    c,u,w=basis;v=np.asarray(v,dtype=LD);d=v@c
    if np.any(d<=0):raise ValueError('Gnomonic clipping crosses chart horizon')
    p=np.c_[v@u,v@w]/d[:,None]
    p=clean_vertices(p)
    if len(p)<3:return np.empty((0,2),dtype=LD)
    return convex_boundary(p)

def clean_vertices(points):
    if not len(points):return np.empty((0,2),dtype=LD)
    bound=8*np.finfo(float).eps*(1+float(np.max(np.abs(points))))
    clean=[]
    for x in points:
        if not clean or np.linalg.norm(x-clean[-1])>bound:clean.append(x)
    if len(clean)>1 and np.linalg.norm(clean[0]-clean[-1])<=bound:clean.pop()
    return np.asarray(clean,dtype=LD).reshape((-1,2))

def signed_area(p):
    if len(p)<3:return LD(0)
    a=p[1:-1]-p[0];b=p[2:]-p[0]
    return np.sum(a[:,0]*b[:,1]-a[:,1]*b[:,0])/2

def convex_boundary(p):
    """Remove rounded interior/collinear vertices before forming halfplanes.

    A short spurious edge must not define a halfplane across a whole cell.
    Hull area changes beyond the coordinate error tube remain hard failures.
    """
    ordered=p[np.lexsort((p[:,1],p[:,0]))]
    def chain(points):
        result=[]
        for x in points:
            while len(result)>1 and side(x,result[-2],result[-1])<=0:result.pop()
            result.append(x)
        return result
    hull=np.asarray(chain(ordered)[:-1]+chain(ordered[::-1])[:-1],dtype=LD)
    original=abs(signed_area(p));actual=abs(signed_area(hull))
    edges=np.roll(p,-1,axis=0)-p;perimeter=np.sum(np.sqrt(np.sum(edges*edges,axis=1)))
    budget=64*np.finfo(float).eps*(1+np.max(abs(p)))*perimeter*(len(p)+2)
    if abs(actual-original)>budget:raise ValueError('Gnomonic boundary is genuinely nonconvex')
    return hull if len(hull)>=3 and actual>0 else np.empty((0,2),dtype=LD)

def unproject(p,basis):
    if len(p)<3:return np.empty((0,3))
    c,u,w=basis;v=c+p[:,0,None]*u+p[:,1,None]*w
    v/=np.sqrt(np.sum(v*v,axis=1))[:,None]
    return np.asarray(v,dtype=float)

def side(p,a,b):
    e=b-a;q=p-a;terms=np.array([e[0]*q[1],e[1]*q[0]])
    value=terms[0]-terms[1]
    bound=8*np.finfo(LD).eps*np.sum(abs(terms))
    return LD(0) if abs(value)<=bound else value

def clip(p,a,b,positive=True):
    if len(p)<3:return np.empty((0,2),dtype=LD)
    out=[];sign=1 if positive else -1
    for u,v in zip(p,np.roll(p,-1,axis=0)):
        du=sign*side(u,a,b);dv=sign*side(v,a,b)
        if (du<0<dv) or (dv<0<du):out.append(u+(v-u)*(du/(du-dv)))
        if dv>=0:out.append(v)
    clean=clean_vertices(out)
    if len(clean)<3:return np.empty((0,2),dtype=LD)
    return convex_boundary(np.asarray(clean))

def common_frame(a,b,centre):
    basis=frame(centre)
    if np.any(np.asarray(a)@basis[0]<=0) or np.any(np.asarray(b)@basis[0]<=0):
        basis=frame(np.asarray(a).mean(0)+np.asarray(b).mean(0))
    return basis

def intersection(a,b,centre):
    basis=common_frame(a,b,centre);p=project(a,basis);q=project(b,basis)
    if len(q)<3:return np.empty((0,3))
    for u,v in zip(q,np.roll(q,-1,axis=0)):
        p=clip(p,u,v)
        if len(p)<3:break
    return unproject(p,basis)

def difference(a,b,centre):
    basis=common_frame(a,b,centre);p=project(a,basis);q=project(b,basis);parts=[]
    if len(q)<3:return (unproject(p,basis),) if len(p)>=3 else ()
    for u,v in zip(q,np.roll(q,-1,axis=0)):
        cut=clip(p,u,v,False)
        if len(cut)>=3:parts.append(unproject(cut,basis))
        p=clip(p,u,v)
        if len(p)<3:break
    return tuple(parts)

def vacancy(domain,occupied,centre):
    """One bounded-precision overlay instead of exponentially repeated cuts.

    Quantization is tied to the same 64-ulp-per-stage boundary error budget.
    No material lives in this complement yet; sub-budget gaps do not create rock.
    """
    import shapely
    from shapely.geometry import Polygon
    basis=frame(centre);stages=2+sum(len(p) for p in occupied)
    grid=float(64*np.finfo(float).eps*stages)
    d=shapely.set_precision(Polygon(np.asarray(project(domain,basis),dtype=float)),grid)
    polys=[shapely.set_precision(Polygon(np.asarray(project(p,basis),dtype=float)),grid) for p in occupied]
    gap=shapely.difference(d,shapely.union_all(polys,grid_size=grid),grid_size=grid)
    triangles=shapely.constrained_delaunay_triangles(gap)
    parts=tuple(unproject(np.asarray(p.exterior.coords,dtype=LD)[:-1],basis) for p in triangles.geoms if p.geom_type=='Polygon')
    return parts,grid,stages
