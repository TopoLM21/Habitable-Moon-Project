"""Exact rational inner/outer radial tubes for each Gamma bank of a cell.

The bilinear birth map belongs to the convex hull of its six slab corners.
Distances from the centroid line are bounded at those corners. Radius R+eps
contains every source fiber; R-eps is contained in every fiber. Independent
plane deviations give necessary/sufficient bank conditions. Bounds concern
reference measure det(X)*lambda**2 dN dlambda, without empirical EPS.
"""
from fractions import Fraction as F
import numpy as np
from .genesis_volume_rational_fields import rational,fraction
from .genesis_volume_event_partition import (
    literal_cell,signed_double_area,source_events,sign_certificate,multiply,add,scale,ratio,digest,
    radial_support_classification,
)
from .genesis_volume_operand_bounds import rational_enclosure
from analysis.genesis_volume_operand_validation.independent_geometry_bounds import sqrt_enclosure


def sphere_interval(u,v,radius,cohort,outer):
    if radius<=0:return None
    A,B,V=u@u,u@v,v@v
    if A==0:
        return cohort if (V<=radius*radius if outer else V<radius*radius) else None
    margin=radius*radius-V+B*B/A
    if margin<0:return None
    root=sqrt_enclosure(margin/A,binary_places=256)[int(outer)]
    center=-B/A
    lo,hi=max(cohort[0],center-root),min(cohort[1],center+root)
    return (lo,hi) if hi>lo else None


def halfspace(interval,slope,intercept):
    if interval is None:return None
    lo,hi=interval
    if slope>0:lo=max(lo,-intercept/slope)
    elif slope<0:hi=min(hi,-intercept/slope)
    elif intercept<=0:return None
    return (lo,hi) if hi>lo else None


class PreparedBankBounds:
    def __init__(self,raw):
        s=raw['section'];X,A,P=(np.asarray(s[k],float) for k in ('triangle_m','displacement_A_m','slope_P_m'))
        self.C,self.D=rational(X+P),rational(A-P)
        self.X=rational(X);self.center=rational(raw['seed_center_world_m'])
        if any(fraction(a)!=fraction(b) for a,b in zip(raw['seed_center_world_m'],np.asarray(raw['seed_center_world_m'],float))):
            raise ValueError('Event bounds require literal binary64 center')
        if any(fraction(v)!=fraction(float(v)) for v in (raw['seed_radius_m'],s['offset_m'])):
            raise ValueError('Event bounds require literal binary64 radius and offset')
        self.radius=fraction(raw['seed_radius_m']);self.normal=rational(np.asarray(s['normal_xyz'],float))
        self.offset=fraction(s['offset_m']);self.cohort=tuple(map(fraction,np.asarray(s['lambda_bounds'],float)))
        self.det=sum(self.X[0,j]*(self.X[1,(j+1)%3]*self.X[2,(j+2)%3]-self.X[1,(j+2)%3]*self.X[2,(j+1)%3]) for j in range(3))
        if self.det<=0 or not 0<self.cohort[0]<self.cohort[1] or self.radius<=0:raise ValueError('Invalid source measure')
        self.events,_=source_events(raw)
        self.closest_phi=add(multiply(self.events['A'],self.events['h']),scale(multiply(self.events['B'],self.events['g']),-1))
        self.cache={};self.skipped=0

    def certificate(self,vertices):
        vertices=literal_cell(vertices)
        if any(sum(n)!=1 for n in vertices):raise ValueError('Bank cell must lie in exact simplex')
        if vertices in self.cache:return self.cache[vertices]
        n=np.array(vertices,object);middle=n.sum(axis=0)/3
        corners=np.vstack([n@(lam*self.C+self.D) for lam in self.cohort])
        reference=np.vstack([np.tile(middle@(lam*self.C+self.D),(3,1)) for lam in self.cohort])
        deviation=corners-reference
        eps2=max(row@row for row in deviation)
        eps=sqrt_enclosure(eps2,binary_places=128)[1]
        plane_deviations=[row@self.normal for row in deviation]
        plane_low,plane_high=min(plane_deviations),max(plane_deviations)
        eta=max(abs(plane_low),abs(plane_high))
        corner_phi=[row@self.normal-self.offset for row in corners]
        u=middle@self.C;d=middle@self.D;v=d-self.center
        distance2=sum(max(F(),min(corners[:,j])-self.center[j],self.center[j]-max(corners[:,j]))**2 for j in range(3))
        outer=None if distance2>self.radius**2 else sphere_interval(u,v,self.radius+eps,self.cohort,True)
        inner=sphere_interval(u,v,self.radius-eps,self.cohort,False)
        slope,intercept=u@self.normal,d@self.normal-self.offset
        if outer is not None and not slope and not intercept and not eta:
            raise ValueError('Literal Gamma contains an unresolved candidate cell')
        factor=signed_double_area(vertices)*self.det/6
        def measure(interval):return F() if interval is None else factor*(interval[1]**3-interval[0]**3)
        radial_names=('A','Delta','sphere_cohort_lower','sphere_cohort_upper',
                      'sphere_derivative_half_lower','sphere_derivative_half_upper')
        polynomials={k:self.events[k] for k in radial_names}
        polynomials.update(resultant=self.events['Gamma_sphere_resultant'],closest_phi=self.closest_phi)
        events={k:sign_certificate(p,vertices) for k,p in polynomials.items()}
        radial_classification=radial_support_classification(events)
        radial_excluded=radial_classification=='EMPTY_CERTIFIED_ON_SUPPLIED_CELL'
        def bound(k,edge):return F(*map(int,events[k][edge]))
        # If q(lambda_Gamma)>0, the Gamma root is outside the sphere chord.
        # Its bank is fixed by phi at the closest point, including g=0.
        separated=bound('A','lower')>0 and bound('resultant','lower')>0
        ranges={};reasons={};outer_banks={};inner_banks={}
        for side in (-1,1):
            delta_min,delta_max=(plane_low,plane_high) if side>0 else (-plane_high,-plane_low)
            outer_banks[side]=halfspace(outer,side*slope,side*intercept+delta_max)
            inner_banks[side]=halfspace(inner,side*slope,side*intercept+delta_min)
            upper=measure(outer_banks[side]);lower=measure(inner_banks[side])
            excluded=separated and (bound('closest_phi','lower')>0 if side<0 else bound('closest_phi','upper')<0)
            cohort_excluded=max(side*v for v in corner_phi)<=0
            if excluded or cohort_excluded or radial_excluded:
                if lower:raise AssertionError('Independent bank bounds disagree')
                upper=F()
            if not 0<=lower<=upper:raise AssertionError('Reversed exact bank measure bound')
            ranges[side]=(lower,upper);reasons[side]='sphere_cohort_Bernstein' if radial_excluded else 'cohort_Gamma_sign' if cohort_excluded else 'sphere_Gamma_separation' if excluded else 'radial_tube_and_plane'
        body=dict(format='genesis-exact-cell-bank-bounds-1',vertices=[[ratio(v) for v in row] for row in vertices],
            reference_triangle_double_area=ratio(signed_double_area(vertices)),reference_det_X=ratio(self.det),
            outer_bank_intervals={str(s):None if v is None else list(map(ratio,v)) for s,v in outer_banks.items()},
            inner_bank_intervals={str(s):None if v is None else list(map(ratio,v)) for s,v in inner_banks.items()},
            source_event_bounds={k:dict(lower=v['lower'],upper=v['upper']) for k,v in events.items()},
            radial_support_classification=radial_classification,
            displacement_norm_squared_upper=ratio(eps2),displacement_norm_upper=ratio(eps),plane_deviation_upper=ratio(eta),plane_deviation_bounds=list(map(ratio,(plane_low,plane_high))),
            box_distance_squared_lower=ratio(distance2),bank_measure_bounds={str(s):list(map(ratio,ranges[s])) for s in (-1,1)},
            bank_reasons={str(s):reasons[s] for s in (-1,1)},
            source_measure='det(X)*lambda^2 dN1 dN2 dlambda',whole_cell_bounds_certified=True,
            unknown_cells_discarded=False,angular_accuracy_certified=False)
        body['sha256']=digest(body);self.cache[vertices]=body
        if all(hi==0 for lo,hi in ranges.values()):self.skipped+=1
        return body

    def bounds(self,vertices):
        cert=self.certificate(vertices)
        return {s:(rational_enclosure(F(*map(int,cert['bank_measure_bounds'][str(s)][0]))).lower,
                   rational_enclosure(F(*map(int,cert['bank_measure_bounds'][str(s)][1]))).upper) for s in (-1,1)}


def verify_bank_certificate(raw,certificate):
    vertices=tuple(tuple(F(*map(int,v)) for v in row) for row in certificate['vertices'])
    if PreparedBankBounds(raw).certificate(vertices)!=certificate:
        raise ValueError('Exact bank certificate differs')
    return True
