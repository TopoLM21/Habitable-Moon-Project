"""Separate discriminant clipping and polynomial contraction at rational nodes.

Uses the preserved independent coefficient/contraction algebra, not runtime
fields or moment vectors. Roots use integer isqrt at an independent precision;
bank antiderivatives use the independent endpoint translator.
"""
from fractions import Fraction as F
from math import isqrt
import numpy as np
from analysis.genesis_volume_operand_validation.independent_coefficients import rational_array, fraction
from analysis.genesis_volume_integral_validation.independent_integrals import (
    source_component_polynomials, signed_endpoint_integral, bank_endpoint_integral,
)


def independent_targets(raw,N,chart,**options):
    n=rational_array(N)
    s=raw['section']
    X,A,P=(np.asarray(s[k],float) for k in ('triangle_m','displacement_A_m','slope_P_m'))
    u=n@rational_array(X+P); v=n@rational_array(A-P)-rational_array(raw['seed_center_world_m'])
    a,b,c=u@u,2*(u@v),v@v-fraction(raw['seed_radius_m'])**2
    discriminant=b*b-4*a*c
    if discriminant<=0:
        raise ValueError('Audit needs a positive source interval')
    scale=2**768
    root=isqrt((discriminant*scale*scale).__floor__())
    rlo,rhi=F(root,scale),F(root+1,scale)
    if rlo*rlo==discriminant:
        rhi=rlo
    lo,hi=map(fraction,s['lambda_bounds'])
    left=(max(lo,(-b-rhi)/(2*a)),max(lo,(-b-rlo)/(2*a)))
    right=(min(hi,(-b+rlo)/(2*a)),min(hi,(-b+rhi)/(2*a)))
    if left[1]>=right[0]:
        raise ValueError('Audit interval is not positive')
    mid,half=map(fraction,(chart.lambda_mid,chart.lambda_halfwidth))
    left=tuple((x-mid)/half for x in left);right=tuple((x-mid)/half for x in right)
    normal=rational_array(np.asarray(s['normal_xyz'],float))
    slope=normal@u
    intercept=normal@(n@rational_array(A-P))-fraction(s['offset_m'])
    phi0,phi1=slope*mid+intercept,slope*half
    witness={'native_exact_binary_ratio':{'interval.'+k:dict(shape=[],numerator_denominator=[[str(v.numerator),str(v.denominator)]])
        for k,v in (('lambda_mid',mid),('lambda_halfwidth',half))}}
    polynomials=source_component_polynomials(raw,n,witness,**options)
    moments={}
    for side in range(5):
        result=[]
        for degree in range(17):
            p=(F(),)*degree+(F(1),)
            fun=(lambda endpoint: signed_endpoint_integral(p,F(),endpoint,side,phi0,phi1)) if side<3 else (
                lambda endpoint: bank_endpoint_integral(p,F(),endpoint,side-3,phi0,phi1))
            aa,bb=fun(left),fun(right)
            result.append((half*(bb[0]-aa[1]),half*(bb[1]-aa[0])))
        moments[side]=result
    targets={}
    for name,terms in polynomials.items():
        lower=upper=F()
        for side,polynomial in terms:
            for coefficient,(a,b) in zip(polynomial,moments[side]):
                ends=(coefficient*a,coefficient*b)
                lower+=min(ends);upper+=max(ends)
        targets[name]=(lower,upper)
    return targets
