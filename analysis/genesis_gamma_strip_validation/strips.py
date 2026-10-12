"""Fixed transverse quarters and exact Gamma-root strip bounds; no work-sign targeting."""
from fractions import Fraction as F
from itertools import product


def evaluate(poly,point):
    value=F()
    for k,c in poly.items():
        term=c
        for x,n in zip(point,k):term*=x**n
        value+=term
    return value


def root_window(gamma,box,axis):
    if any(any(n>1 for n in k) for k in gamma):raise ValueError('Multiaffine Gamma required')
    other=[a for a in range(3) if a!=axis];roots=[];slopes=[]
    for corner in product((0,1),repeat=2):
        point=[F(),F(),F()]
        for a,c in zip(other,corner):point[a]=box[a][c]
        point[axis]=box[axis][0];left=evaluate(gamma,point)
        point[axis]=box[axis][1];right=evaluate(gamma,point)
        slope=right-left;slopes.append(slope)
        if not slope:return None
        roots.append(-left/slope)
    if not(all(s>0 for s in slopes) or all(s<0 for s in slopes)):return None
    # Affine numerator/denominator in each transverse coordinate with fixed
    # denominator sign: coordinatewise extrema of their ratio occur at corners.
    low=max(F(),min(F(1),min(roots)));high=max(F(),min(F(1),max(roots)))
    return dict(axis=axis,normalized_root_bounds=[str(low),str(high)],
                corner_roots=list(map(str,roots)),slope_sign=1 if slopes[0]>0 else -1)


def partition(gamma,box):
    candidates=[x for axis in range(3) if (x:=root_window(gamma,box,axis)) is not None]
    if not candidates:return dict(axis=None,quarters=[],reason='no_uniform_monotone_axis')
    score=lambda x:(F(x['normalized_root_bounds'][1])-F(x['normalized_root_bounds'][0]),x['axis'])
    axis=min(candidates,key=score)['axis'];other=[a for a in range(3) if a!=axis];quarters=[]
    for sides in product((0,1),repeat=2):
        q=[tuple(b) for b in box]
        for a,side in zip(other,sides):
            lo,hi=q[a];mid=(lo+hi)/2;q[a]=(lo,mid) if side==0 else (mid,hi)
        window=root_window(gamma,q,axis);assert window is not None
        lo,hi=q[axis];l,u=map(F,window['normalized_root_bounds'])
        cuts=sorted(set((lo,lo+(hi-lo)*l,lo+(hi-lo)*u,hi)))
        slabs=[]
        for left,right in zip(cuts,cuts[1:]):
            child=list(q);child[axis]=(left,right);slabs.append([[str(x) for x in b] for b in child])
        quarters.append(dict(box=[[str(x) for x in b] for b in q],window=window,slabs=slabs))
    return dict(axis=axis,quarters=quarters,reason='root_strips',axis_candidates=candidates)
