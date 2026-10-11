"""Independent discriminant bank-fiber enclosures for rational audit points."""
from fractions import Fraction as F
from math import isqrt
import numpy as np
from tectonics.genesis_volume_rational_fields import rational,fraction


def fiber(raw,N,side):
    s=raw['section'];X,A,P=(np.asarray(s[k],float) for k in ('triangle_m','displacement_A_m','slope_P_m'))
    n=np.array(N,object);u=n@rational(X+P);d=n@rational(A-P);v=d-rational(raw['seed_center_world_m'])
    a=u@u;b=2*(u@v);c=v@v-fraction(raw['seed_radius_m'])**2
    disc=b*b-4*a*c
    if a<=0:raise ValueError('Independent bank audit needs a noncollapsed line')
    if disc<=0:return F(),F()
    scale=2**768;k=isqrt((disc*scale*scale).__floor__());r0=F(k,scale);r1=r0 if r0*r0==disc else F(k+1,scale)
    lo,hi=map(fraction,np.asarray(s['lambda_bounds'],float))
    left=[max(lo,(-b-r1)/(2*a)),max(lo,(-b-r0)/(2*a))]
    right=[min(hi,(-b+r0)/(2*a)),min(hi,(-b+r1)/(2*a))]
    normal=rational(np.asarray(s['normal_xyz'],float));g=side*(u@normal);h=side*(d@normal-fraction(s['offset_m']))
    if not g and not h:return None
    if g>0:left=[max(x,-h/g) for x in left]
    elif g<0:right=[min(x,-h/g) for x in right]
    elif h<0:return F(),F()
    x=rational(X)
    det=x[0,0]*(x[1,1]*x[2,2]-x[1,2]*x[2,1])-x[0,1]*(x[1,0]*x[2,2]-x[1,2]*x[2,0])+x[0,2]*(x[1,0]*x[2,1]-x[1,1]*x[2,0])
    return det*max(F(),right[0]**3-left[1]**3)/3,det*max(F(),right[1]**3-left[0]**3)/3


def audit_certificates(raw,certificates):
    checks=skipped=0
    for c in certificates:
        cell=tuple(tuple(F(*map(int,v)) for v in row) for row in c['vertices'])
        area=F(*map(int,c['reference_triangle_double_area']))/2
        points=cell+(tuple(sum(row[j] for row in cell)/3 for j in range(3)),)
        for n in points:
            for side in (-1,1):
                target=fiber(raw,n,side)
                if target is None:skipped+=1;continue
                lo,hi=(F(*map(int,v)) for v in c['bank_measure_bounds'][str(side)])
                if not lo<=area*target[0]<=area*target[1]<=hi:
                    raise AssertionError('Independent bank fiber escapes the cell envelope')
                checks+=1
    return dict(independent_bank_fiber_checks=checks,whole_Gamma_fibers_not_used=skipped,
                all_sampled_bank_fibers_contained=True,proof_for_whole_cells_is_separate=True)
