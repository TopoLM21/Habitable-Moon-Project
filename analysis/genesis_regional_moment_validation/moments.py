"""Exact polynomial moments and signed mask bounds on unchanged source boxes."""
from fractions import Fraction as F
from itertools import product
from math import comb, prod


def degree(p):
    return tuple(max((k[a] for k in p),default=0) for a in range(3))


def transform_axis(values,axis,matrix):
    out={}
    for k,v in values.items():
        for i,row in enumerate(matrix):
            m=row[k[axis]]
            if m:
                j=list(k);j[axis]=i;j=tuple(j)
                out[j]=out.get(j,F())+m*v
    return {k:v for k,v in out.items() if v}


def coefficients(p,box):
    """Separable exact power -> local power -> tensor Bernstein conversion."""
    d=degree(p)
    if prod(n+1 for n in d)>729:raise ValueError('729 coefficient cap')
    values=p
    for axis,n in enumerate(d):
        lo,hi=box[axis];h=hi-lo
        if h<=0:raise ValueError('Nonpositive box width')
        local=[[F(comb(k,i))*lo**(k-i)*h**i if i<=k else F() for k in range(n+1)] for i in range(n+1)]
        values=transform_axis(values,axis,local)
    for axis,n in enumerate(d):
        bern=[[F(comb(i,k),comb(n,k)) if k<=i else F() for k in range(n+1)] for i in range(n+1)]
        values=transform_axis(values,axis,bern)
    return dict(degree=list(d),coefficients=[str(values.get(k,F())) for k in product(*(range(n+1) for n in d))])


def enclose(polys,box):
    cert={key:coefficients(p,box) for key,p in polys.items()}
    return enclose_certificates(cert,box)


def enclose_certificates(cert,box):
    b={k:list(map(F,v['coefficients'])) for k,v in cert.items()}
    qlo,qhi=min(b['sphere']),max(b['sphere']);glo,ghi=min(b['gamma']),max(b['gamma'])
    if glo==ghi==0:raise ValueError('Identically zero Gamma needs ownership semantics')
    K=b['density_without_H'];volume=prod(hi-lo for lo,hi in box)
    integral=volume*sum(K)/len(K)
    # Nonnegative Bernstein basis: K_minus <= min(K(x),0),
    # max(K(x),0) <= K_plus, pointwise, without sampling.
    neg=volume*sum(min(F(),v) for v in K)/len(K)
    pos=volume*sum(max(F(),v) for v in K)/len(K)
    if qlo>0:state='outside';bounds=(F(),F())
    elif glo>=0 or ghi<=0:
        sign=1 if glo>=0 else -1
        if qhi<=0:state='inside_gamma_fixed';bounds=(sign*integral/2,)*2
        else:
            state='sphere_boundary_gamma_fixed'
            bounds=(neg/2,pos/2) if sign==1 else (-pos/2,-neg/2)
    else:
        state='gamma_uncertain_inside' if qhi<=0 else 'sphere_and_gamma_uncertain'
        absolute=pos-neg;bounds=(-absolute/2,absolute/2)
    return dict(box=[[str(x) for x in row] for row in box],polynomials=cert,state=state,
                bounds=list(map(str,bounds)),whole_box_integral_without_H=str(integral),
                negative_envelope_integral=str(neg),positive_envelope_integral=str(pos))
