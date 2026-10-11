"""Independent quadratic-formula audit of the analytic fixtures' ray records.

No Sturm sequence or ray restriction implementation is imported. This auditor
deliberately refuses degree >2; it does not certify general deformed sources.
"""
from fractions import Fraction as F
from math import isqrt
from pathlib import Path
import argparse
import hashlib
import json
from analysis.genesis_volume_banks_validation.independent_bank import fiber


def decode(value):return F(*map(int,value))


def value(p,x):return sum(c*x**i for i,c in enumerate(p))


def roots(p):
    if len(p)>3:raise ValueError('Quadratic analytic-fixture audit only')
    if len(p)<2:return []
    if len(p)==2:
        r=-p[0]/p[1];return [(r,r)] if 0<=r<=1 else []
    c,b,a=p
    if c==0:exact=[F(),-b/a]
    elif a+b+c==0:exact=[F(1),c/a]
    else:exact=None
    if exact is not None:return [(r,r) for r in sorted(set(exact)) if 0<=r<=1]
    d=b*b-4*a*c
    if d<0:return []
    if not d:
        r=-b/(2*a);return [(r,r)] if 0<=r<=1 else []
    scale=2**512;k=isqrt((d*scale*scale).__floor__())
    low=F(k,scale);high=low if low*low==d else F(k+1,scale)
    result=[]
    for sign in (-1,1):
        bounds=sorted((-b+sign*s)/(2*a) for s in (low,high))
        lo,hi=bounds
        if hi<0 or lo>1:continue
        if lo<0 or hi>1:raise ValueError('Unresolved boundary root in independent audit')
        result.append((lo,hi))
    return sorted(result)


def audit_ray(raw,row):
    a=tuple(map(decode,row['start']));b=tuple(map(decode,row['end']))
    assert sum(a)==sum(b)==1 and min(a+b)>=0
    polynomials={k:tuple(map(decode,p)) for k,p in row['polynomials'].items()}
    R=F(raw['seed_radius_m']);offset=F(raw['section']['offset_m'])
    assert raw['seed_center_world_m']==[1.,1.,9.]
    assert raw['section']['triangle_m']==[[0.,0.,10.],[4.,0.,10.],[0.,4.,10.]]
    assert raw['section']['slope_P_m']==raw['section']['displacement_A_m']==[[0.,0.,0.]]*3
    assert raw['section']['normal_xyz']==[0.,0.,1.]
    assert raw['section']['lambda_bounds']==[.8,1.]
    # Direct geometry is quadratic in rho: three exact values determine it.
    for rho in (F(),F(1,2),F(1)):
        n=tuple(x+rho*(y-x) for x,y in zip(a,b));x,y=4*n[1],4*n[2]
        A=x*x+y*y+100;B=-x-y-90;E=83-R*R
        expected=dict(A=A,Delta=B*B-A*E,g=F(10),
            Gamma_sphere_resultant=A*offset**2+20*B*offset+100*E)
        for name,lam in zip(('lower','upper'),map(F,raw['section']['lambda_bounds'])):
            expected['sphere_cohort_'+name]=A*lam*lam+2*B*lam+E
            expected['Gamma_cohort_'+name]=10*lam-offset
        assert set(polynomials)==set(expected)
        for name,p in polynomials.items():assert len(p)<=3 and value(p,rho)==expected[name]
    boxes=[];root_count=0
    for name,p in polynomials.items():
        independent=roots(p);given=row['isolated'][name]
        assert given['identically_zero']==(not p)
        isolated=[tuple(map(decode,box)) for box in given['roots']]
        assert len(independent)==len(isolated)
        for (lo,hi),(u,v) in zip(sorted(isolated),independent):
            if lo==hi:assert value(p,lo)==0 and u<=lo<=v
            else:assert lo<=u<=v<=hi
        boxes.extend(isolated);root_count+=len(independent)
    intervals=[(decode(r['lo']),decode(r['hi']),r['kind'],r['integrated']) for r in row['intervals']]
    assert intervals[0][0]==0 and intervals[-1][1]==1
    assert all(lo<hi for lo,hi,_,_ in intervals)
    assert all(x[1]==y[0] for x,y in zip(intervals,intervals[1:]))
    assert sum(hi-lo for lo,hi,_,_ in intervals)==1
    checks=0
    for lo,hi,kind,integrated in intervals:
        assert kind in ('root_free','root_box')
        if kind=='root_box':assert integrated;continue
        assert all(not (lo<x<hi) if x==y else max(lo,x)>=min(hi,y) for x,y in boxes)
        if integrated:continue
        for t in (F(1,4),F(1,2),F(3,4)):
            rho=lo+(hi-lo)*t;n=tuple(x+rho*(y-x) for x,y in zip(a,b))
            for side in (-1,1):assert fiber(raw,n,side)==(F(),F());checks+=1
    return root_count,checks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    results=[]
    for directory in args.runs:
        report=json.loads((directory/'report.json').read_bytes())
        path=directory/'ray_certificates.json';rays=json.loads(path.read_bytes())
        checked=[audit_ray(report['source_geometry'],r) for r in rays]
        assert len(rays)==report['ray_partitions_built']
        results.append(dict(case=report['case'],rays=len(rays),distinct_event_roots=sum(r[0] for r in checked),
            sampled_zero_fiber_checks=sum(r[1] for r in checked),all_partitions_complete_without_overlap=True,
            all_root_boxes_retained=True,ray_records_path=str(path),
            ray_records_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            proof_scope='Recorded rational rays of analytic quadratic fixtures, including planned rays not evaluated'))
    with args.output.open('x') as f:json.dump(dict(cases=results,independent_of_sturm=True),f,indent=2);f.write('\n')
    print(json.dumps(results))


if __name__=='__main__':main()
