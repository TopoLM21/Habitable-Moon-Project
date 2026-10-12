"""Exact SPD quadratic bounds. Signed cross terms are retained by exact LDL identity."""
from fractions import Fraction as F


def ldl(matrix):
    C=[list(map(F,r)) for r in matrix];n=len(C)
    if any(C[i][j]!=C[j][i] for i in range(n) for j in range(n)):raise ValueError('Symmetric C required')
    L=[[F(i==j) for j in range(n)] for i in range(n)];D=[]
    for j in range(n):
        d=C[j][j]-sum(L[j][k]**2*D[k] for k in range(j))
        if d<=0:raise ValueError('Strict positive definiteness not proved')
        D.append(d)
        for i in range(j+1,n):L[i][j]=(C[i][j]-sum(L[i][k]*L[j][k]*D[k] for k in range(j)))/d
    return L,D


def quadratic(ranges,L,D):
    lower=upper=F()
    for j,d in enumerate(D):
        lo=hi=F()
        for i in range(len(D)):
            endpoints=[L[i][j]*v for v in ranges[i]];lo+=min(endpoints);hi+=max(endpoints)
        lower+=d*(0 if lo<=0<=hi else min(lo*lo,hi*hi));upper+=d*max(lo*lo,hi*hi)
    return lower,upper


def block(ranges,box,state,L,D,det):
    volume=F(1)
    for lo,hi in box:volume*=hi-lo
    if state=='outside':return dict(stiffness_a_j=['0','0'],memory_norm_j=['0','0'])
    inside=state in ('inside_gamma_fixed','gamma_uncertain_inside')
    out={}
    for prefix,name,factor in [('strain','stiffness_a_j',F(1,4)),('memory','memory_norm_j',F(1))]:
        lo,hi=quadratic([ranges[prefix+'_'+str(i)] for i in range(6)],L,D)
        out[name]=[str(det*volume*factor*(lo if inside else 0)),str(det*volume*factor*hi)]
    return out


def responses(W,K,ell2):
    """Exact correlated affine response; t is dimensionless, a is in metres."""
    rows=[]
    for h in (F(1,4),F(1,8),F(1,16)):
        for t in (-h,h):
            work=sorted(t*x for x in W);increment=sorted(t*x for x in K)
            energy=[work[e]+t*t*K[e]/2 for e in (0,1)]
            response=[W[e]+increment[e] for e in (0,1)]
            rows.append(dict(t=str(t),internal_virtual_work_j=list(map(str,work)),
                incremental_directional_force_times_a_j=list(map(str,increment)),
                total_directional_force_times_a_j=list(map(str,response)),
                frozen_elastic_energy_change_j=list(map(str,energy)),
                centered_force_secant_a_j=list(map(str,K)),
                centered_energy_curvature_a_j=list(map(str,K)),
                stiffness_normalized_n_m=[str(x/ell2) for x in K]))
    return rows
