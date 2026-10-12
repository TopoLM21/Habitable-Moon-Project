"""Independent inverse polynomial transform and exact monomial antiderivative audit.

Does not import the experimental Bernstein transform or its enclosure routine.
"""
from fractions import Fraction as F
from itertools import product
from math import comb,prod


def recover(cert,box):
    d=cert['degree'];assert len(d)==3 and all(isinstance(n,int) and 0<=n<=8 for n in d)
    assert len(cert['coefficients'])==prod(n+1 for n in d)
    values={k:F(v) for k,v in zip(product(*(range(n+1) for n in d)),cert['coefficients']) if F(v)}
    for axis,n in enumerate(d):
        out={}
        for k,value in values.items():
            i=k[axis]
            for power in range(i,n+1):
                j=list(k);j[axis]=power;j=tuple(j)
                out[j]=out.get(j,F())+value*comb(n,i)*comb(n-i,power-i)*(-1)**(power-i)
        values={k:v for k,v in out.items() if v}
    # Convert local t=(x-lo)/h back to original physical power coefficients.
    for axis in range(3):
        lo,hi=box[axis];assert hi>lo
        out={}
        for k,value in values.items():
            n=k[axis]
            for power in range(n+1):
                j=list(k);j[axis]=power;j=tuple(j)
                out[j]=out.get(j,F())+value*comb(n,power)*(-lo)**(n-power)/(hi-lo)**n
        values={k:v for k,v in out.items() if v}
    return values


def audit(polys,record):
    box=[tuple(map(F,b)) for b in record['box']]
    for name,p in polys.items():assert recover(record['polynomials'][name],box)==p
    c={k:list(map(F,v['coefficients'])) for k,v in record['polynomials'].items()}
    volume=prod(hi-lo for lo,hi in box);K=c['density_without_H']
    exact=sum(v*prod((hi**(e+1)-lo**(e+1))/(e+1) for e,(lo,hi) in zip(k,box)) for k,v in polys['density_without_H'].items())
    assert exact==F(record['whole_box_integral_without_H'])==volume*sum(K)/len(K)
    low=volume*sum(x for x in K if x<0)/len(K);high=volume*sum(x for x in K if x>0)/len(K)
    assert low==F(record['negative_envelope_integral']) and high==F(record['positive_envelope_integral'])
    q,g=c['sphere'],c['gamma'];assert min(g)!=0 or max(g)!=0
    if min(q)>0:bounds=(F(),F());state='outside'
    elif min(g)>=0 or max(g)<=0:
        s=1 if min(g)>=0 else -1
        if max(q)<=0:bounds=(s*exact/2,)*2;state='inside_gamma_fixed'
        else:bounds=(low/2,high/2) if s==1 else (-high/2,-low/2);state='sphere_boundary_gamma_fixed'
    else:
        bounds=(-(high-low)/2,(high-low)/2)
        state='gamma_uncertain_inside' if max(q)<=0 else 'sphere_and_gamma_uncertain'
    assert bounds==tuple(map(F,record['bounds'])) and state==record['state']
    return dict(exact_polynomial_identities=3,exact_integral_verified=True,signed_mask_enclosure_verified=True)
