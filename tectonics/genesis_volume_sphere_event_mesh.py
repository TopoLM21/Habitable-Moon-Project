"""Measure-preserving anisotropic refinement guided by curved source events.

Experimental only: analytic half-ball trials regress at the original work cap.
This is not the default angular refinement or a certified convergence method.
Root brackets choose a cut, never a side assignment or a discarded volume.
Both sides of each cut remain in the exact source domain. Every new vertex is
an exact convex combination, so previously exact affine Gamma signs persist.
"""
from fractions import Fraction as F
import numpy as np
from .genesis_volume_rational_fields import rational,fraction
from .genesis_volume_event_partition import (
    literal_cell,signed_double_area,source_events,on_cell,split_cell,ratio,sign_certificate,
)


def fan(cell,point):
    return tuple(literal_cell((cell[i],cell[(i+1)%3],point)) for i in range(3)
                 if signed_double_area((cell[i],cell[(i+1)%3],point))>0)


def parallel_cut(cell,opposite,t):
    cell=literal_cell(cell);t=F(t)
    if not 0<t<1:raise ValueError('The cut must be inside the triangle')
    a,b,c=cell[opposite],cell[(opposite+1)%3],cell[(opposite+2)%3]
    p=tuple((1-t)*v+t*w for v,w in zip(b,a))
    q=tuple((1-t)*v+t*w for v,w in zip(c,a))
    result=((a,p,q),(b,c,q),(b,q,p))
    if any(signed_double_area(v)<=0 for v in result):raise AssertionError('Nonpositive cut child')
    if sum(signed_double_area(v) for v in result)!=signed_double_area(cell):raise AssertionError('Cut lost measure')
    return tuple(map(literal_cell,result))


def bracket_root(event,start,end,relative_bits=12):
    def point(t):return tuple((1-t)*a+t*b for a,b in zip(start,end))
    lo,hi=F(),F(1);flo,fhi=event.evaluate(start),event.evaluate(end)
    if flo*fhi>=0:raise ValueError('A strict root sign bracket is required')
    for iteration in range(128):
        mid=(lo+hi)/2;value=event.evaluate(point(mid))
        if value==0:return mid,dict(lower=ratio(mid),upper=ratio(mid),exact_root=True,iterations=iteration+1)
        if value*flo>0:lo,flo=mid,value
        else:hi,fhi=mid,value
        if hi-lo<=min(lo,1-hi)/2**relative_bits:
            return (lo+hi)/2,dict(lower=ratio(lo),upper=ratio(hi),exact_root=False,iterations=iteration+1)
    return None,dict(reason='Unresolved root bracket retained; midpoint fallback')


class SphereEventRefiner:
    def __init__(self,raw):
        events,_=source_events(raw)
        self.events={k:events[k] for k in ('Delta','sphere_cohort_lower','sphere_cohort_upper','Gamma_sphere_resultant')}
        s=raw['section'];X,A,P=(np.asarray(s[k],float) for k in ('triangle_m','displacement_A_m','slope_P_m'))
        lam=sum(map(fraction,s['lambda_bounds']))/2
        self.metric_map=lam*rational(X+P)+rational(A-P)

    def chart(self,vertices):
        for name,event in self.events.items():
            cert=sign_certificate(event,vertices)
            lo,hi=(F(*map(int,cert[k])) for k in ('lower','upper'))
            if lo<=0<=hi and lo!=hi:
                values=[abs(event.evaluate(n)) for n in vertices]
                opposite=max(range(3),key=lambda i:values[i])
                return opposite,2,name
        return 0,1,None

    def split(self,vertices):
        cell=literal_cell(vertices)
        positions=np.asarray(cell,object)@self.metric_map
        lengths=[sum(v*v for v in positions[i]-positions[(i+1)%3]) for i in range(3)]
        edge=max(range(3),key=lambda i:lengths[i])
        for name,event in self.events.items():
            values=tuple(event.evaluate(n) for n in cell)
            candidates=[]
            for i in (edge,):
                j=(i+1)%3
                fm,fv=values[i],values[j]
                if fm*fv<0:
                    estimate=fm/(fm-fv)
                    candidates.append((min(estimate,1-estimate),i))
            if candidates:
                _,i=min(candidates);j=(i+1)%3;k=(i+2)%3
                t,bracket=bracket_root(event,cell[i],cell[j])
                if t is not None:
                    # Avoid endlessly peeling tiny endpoint triangles. The work
                    # and depth caps are unchanged; split quality must improve.
                    if not F(1,4)<=t<=F(3,4):
                        continue
                    p=tuple((1-t)*a+t*b for a,b in zip(cell[i],cell[j]))
                    children=(literal_cell((cell[i],p,cell[k])),literal_cell((p,cell[j],cell[k])))
                    return children,dict(method='event-guided-edge-cut',event=name,edge=i,
                        parameter=ratio(t),root_bracket=bracket,exact_partition=True,exact_curved_event_alignment=False)
            # A whole conic can lie inside a cell without edge sign changes.
            # Use its exact stationary point if the restricted event is quadratic.
            local=on_cell(event,cell)
            if local.degree<=2:
                p=local.coefficients;aa=2*p.get((2,0),F());bb=p.get((1,1),F());cc=2*p.get((0,2),F())
                u,v=p.get((1,0),F()),p.get((0,1),F());det=aa*cc-bb*bb
                if det:
                    s,t=(bb*v-cc*u)/det,(bb*u-aa*v)/det
                    if s>0 and t>0 and s+t<1:
                        n=tuple((1-s-t)*a+s*b+t*c for a,b,c in zip(*cell));fn=event.evaluate(n)
                        if any(fn*f<0 for f in values):
                            children=fan(cell,n)
                            return children,dict(method='interior-conic-stationary-fan',event=name,
                                point=list(map(ratio,n)),exact_partition=True,exact_curved_event_alignment=False)
        i=edge;j=(i+1)%3;k=(i+2)%3
        p=tuple((a+b)/2 for a,b in zip(cell[i],cell[j]))
        children=(literal_cell((cell[i],p,cell[k])),literal_cell((p,cell[j],cell[k])))
        return children,dict(method='physical-longest-edge-bisection',edge=edge,
            exact_partition=True,exact_curved_event_alignment=False)
