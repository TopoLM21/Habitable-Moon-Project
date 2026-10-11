"""Independent polynomial identity, enclosure and exact-cover audit.

Does not import the Bernstein conversion or experimental controller.
"""
from fractions import Fraction as F
from itertools import product
from math import comb


def bernstein_value(c,x):
    degrees=c['degree'];values=iter(map(F,c['coefficients']));result=F()
    for powers in product(*(range(d+1) for d in degrees)):
        v=next(values)
        for i,d,t in zip(powers,degrees,x):v*=comb(d,i)*t**i*(1-t)**(d-i)
        result+=v
    return result


def audit(report):
    raw=report['source_geometry'];section=raw['section']
    assert all(v==0 for k in ('slope_P_m','displacement_A_m') for row in section[k] for v in row)
    X=[[F(v) for v in row] for row in section['triangle_m']]
    # Independent six-permutation determinant.
    det=(X[0][0]*X[1][1]*X[2][2]+X[0][1]*X[1][2]*X[2][0]+X[0][2]*X[1][0]*X[2][1]
         -X[0][2]*X[1][1]*X[2][0]-X[0][1]*X[1][0]*X[2][2]-X[0][0]*X[1][2]*X[2][1])
    assert det>0
    center=[F(4,9),F(5,18),F(5,18)];normal=list(map(F,section['normal_xyz']))
    radius=F(raw['seed_radius_m']);sphere=list(map(F,raw['seed_center_world_m']))
    delta=F(section['offset_m'])-sum(n*c for n,c in zip(normal,sphere));norm2=sum(n*n for n in normal)
    assert norm2>0 and radius>0
    empty=[]
    if delta>=0 and delta*delta>=radius*radius*norm2:empty.append(1)
    if delta<=0 and delta*delta>=radius*radius*norm2:empty.append(-1)
    assert empty==report['empty_banks']
    identity_checks=0;bits=0
    for identifier,r in enumerate(report['records']):
        assert r['id']==identifier and 0<=r['triangle']<3
        box=[tuple(map(F,v)) for v in r['box']];assert all(lo<hi for lo,hi in box)
        b=[F(j==r['triangle']) for j in range(3)];c=[F(j==(r['triangle']+1)%3) for j in range(3)]
        area=(b[1]-center[1])*(c[2]-center[2])-(b[2]-center[2])*(c[1]-center[1]);assert area>0
        def direct(x):
            rho,t,lam=x;N=[(1-rho)*a+rho*((1-t)*bb+t*cc) for a,bb,cc in zip(center,b,c)]
            position=[lam*sum(n*row[j] for n,row in zip(N,X)) for j in range(3)]
            return dict(Q=sum((x-y)**2 for x,y in zip(position,sphere))-radius**2,
                        G=sum(n*x for n,x in zip(normal,position))-F(section['offset_m']),K=area*det*rho*lam**2)
        for name,cap in [('Q',(2,2,2)),('G',(1,1,1)),('K',(1,0,2))]:
            cert=r['polynomials'][name];assert all(0<=d<=u for d,u in zip(cert['degree'],cap))
            size=1
            for d in cert['degree']:size*=d+1
            assert len(cert['coefficients'])==size
            # Unisolvent tensor grid: exact identity check, not evidence by random sampling.
            for u in product(*(tuple(F(k,d) for k in range(d+1)) if d else (F(),) for d in cap)):
                physical=[lo+(hi-lo)*x for x,(lo,hi) in zip(u,box)]
                assert bernstein_value(cert,u)==direct(physical)[name];identity_checks+=1
            for v in map(F,cert['coefficients']):bits=max(bits,v.numerator.bit_length(),v.denominator.bit_length())
        ranges={k:(min(map(F,v['coefficients'])),max(map(F,v['coefficients']))) for k,v in r['polynomials'].items()}
        assert ranges['G']!=(F(),F())
        volume=(box[0][1]-box[0][0])*(box[1][1]-box[1][0])*(box[2][1]-box[2][0])
        for bank,side in enumerate((-1,1)):
            gl,gu=ranges['G'] if side==1 else (-ranges['G'][1],-ranges['G'][0])
            if side in empty:state='geometric_empty';bounds=(F(),F())
            elif ranges['Q'][0]>0 or gu<0:state='outside';bounds=(F(),F())
            elif ranges['Q'][1]<=0 and gl>=0:
                state='inside';lo,hi=box[0];ll,lh=box[2]
                exact=area*det*(hi**2-lo**2)/2*(box[1][1]-box[1][0])*(lh**3-ll**3)/3;bounds=(exact,exact)
            else:
                state='unknown';kl,ku=ranges['K'];bounds=(volume*min(F(),kl),volume*max(F(),ku))
            assert r['states'][bank]==state and tuple(map(F,r['bounds'][bank]))==bounds
    records=report['records'];assert len(records)<=report['max_blocks']<=2000
    assert report['maximum_depth']==max(r['depth'] for r in records)<=report['max_depth']<=24
    roots=records[:3];cohort=list(map(F,section['lambda_bounds']));active=set(range(3));created=set(active)
    for i,r in enumerate(roots):
        assert r['triangle']==i and r['depth']==0
        assert [list(map(F,x)) for x in r['box']]==[[F(),F(1)],[F(),F(1)],cohort]
    for split in report['splits']:
        p=records[split['parent']];assert p['id'] in active
        axis=split['axis'];assert axis==p['depth']%3 and len(split['children'])==2
        base=[list(map(F,v)) for v in p['box']];mid=sum(base[axis])/2
        active.remove(p['id'])
        for position,identifier in enumerate(split['children']):
            assert identifier not in created and identifier<len(records)
            child=records[identifier];expected=[list(x) for x in base];expected[axis][1-position]=mid
            assert child['triangle']==p['triangle'] and child['depth']==p['depth']+1
            assert [list(map(F,v)) for v in child['box']]==expected
            active.add(identifier);created.add(identifier)
    assert created==set(range(len(records))) and active==set(report['leaves'])
    totals=[[sum(F(records[i]['bounds'][bank][end]) for i in active) for end in (0,1)] for bank in (0,1)]
    assert totals==[list(map(F,v)) for v in report['integral_bounds']]
    errors=[max(abs(F(v)-lo),abs(hi-F(v))) for v,(lo,hi) in zip(report['reference'],totals)]
    assert errors==list(map(F,report['error_bounds']))
    assert report['certificate_gate_passed']==all(e<=F(a) for e,a in zip(errors,report['allowance']))
    assert report['processed_blocks']==len(records) and report['polynomial_transforms']==3*len(records)
    assert report['stored_bernstein_coefficients']==sum(len(c['coefficients']) for r in records for c in r['polynomials'].values())
    assert report['radial_evaluations']==0
    return dict(exact_polynomial_identity_checks=identity_checks,processed_blocks=len(records),leaves=len(active),
                exact_cover=True,enclosure_rules_verified=True,cost_verified=True,max_coefficient_bits=bits,
                exact_rational_arithmetic=True,new_radial_evaluations=0)
