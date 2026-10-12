"""Independent root identity, closed-form restriction, masks and cover audit."""
from fractions import Fraction as F
from itertools import product
from math import comb,prod
from pathlib import Path
import argparse,hashlib,json,time
from .audit import audit as root_audit
from analysis.genesis_regional_work_validation.identity import expand
from analysis.genesis_volume_mapped_validation.coverage_audit import audit as coverage_audit


def restricted(cert,axis,side):
    """Closed-form binomial restriction, independent of de Casteljau recursion."""
    d=cert['degree'];keys=list(product(*(range(n+1) for n in d)))
    values=dict(zip(keys,map(F,cert['coefficients'])));out=[];n=d[axis]
    for k in keys:
        i=k[axis];value=F()
        js=range(i+1) if side==0 else range(i,n+1)
        for j in js:
            source=list(k);source[axis]=j
            weight=F(comb(i,j),2**i) if side==0 else F(comb(n-i,j-i),2**(n-i))
            value+=weight*values[tuple(source)]
        out.append(str(value))
    return dict(degree=d,coefficients=out)


def verify_bounds(row,cert):
    expected_hash=hashlib.sha256(json.dumps(cert,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert expected_hash==row['coefficient_sha256']
    box=[tuple(map(F,b)) for b in row['box']];assert all(hi>lo for lo,hi in box)
    vol=prod(hi-lo for lo,hi in box)
    data={k:list(map(F,v['coefficients'])) for k,v in cert.items()};k=data['density_without_H'];q=data['sphere'];g=data['gamma']
    lower=vol*sum(v for v in k if v<0)/len(k);upper=vol*sum(v for v in k if v>0)/len(k);moment=vol*sum(k)/len(k)
    assert list(map(F,[row['negative_envelope_integral'],row['positive_envelope_integral'],row['whole_box_integral_without_H']]))==[lower,upper,moment]
    assert min(g)!=0 or max(g)!=0
    if min(q)>0:bounds=(F(),F());state='outside'
    elif min(g)>=0 or max(g)<=0:
        sign=1 if min(g)>=0 else -1
        if max(q)<=0:bounds=(sign*moment/2,)*2;state='inside_gamma_fixed'
        else:bounds=(lower/2,upper/2) if sign>0 else (-upper/2,-lower/2);state='sphere_boundary_gamma_fixed'
    else:
        bounds=(-(upper-lower)/2,(upper-lower)/2)
        state='gamma_uncertain_inside' if max(q)<=0 else 'sphere_and_gamma_uncertain'
    assert tuple(map(F,row['bounds']))==bounds and row['state']==state
    return bounds


def audit(report,source):
    start=time.monotonic();cover=coverage_audit(source['source'],source['coverage'])
    root=[tuple(map(F,b)) for b in report['root']];decode=lambda x:F(*map(int,x))
    assert root[:2]==[tuple(map(decode,b)) for b in source['coverage']['coordinate_bounds'][1:]]
    assert root[2]==tuple(map(F,source['source']['section']['lambda_bounds']))
    assert root[0][0]>=0 and root[1][0]>=0 and root[0][1]+root[1][1]<=1
    det=list(map(decode,source['coverage']['determinant_bounds']));assert det[0]>0 or det[1]<0
    assert any(source['source']['section']['normal_xyz'])
    polys,_=expand(source['graph']);rows=report['records'];cert=report['root_certificates']
    root_audit(polys,{**rows[0],'polynomials':cert})
    assert rows[0]['id']==0 and rows[0]['box']==report['root']
    verify_bounds(rows[0],cert);certificates={0:cert};active={0};seen={0};depth={0:0};cost=0
    snapshots={x['processed_blocks']:x for x in report['trace']}
    def check_trace():
        if len(seen) in snapshots:
            total=[sum(F(rows[i]['bounds'][e]) for i in active) for e in (0,1)]
            assert total==list(map(F,snapshots[len(seen)]['bounds']))
    check_trace()
    for split in report['splits']:
        parent=split['parent'];axis=split['axis'];assert parent in active and axis in (0,1,2)
        assert len(split['children'])==2
        pc=certificates.pop(parent);active.remove(parent);box=[list(map(F,b)) for b in rows[parent]['box']];mid=sum(box[axis])/2
        moments=[]
        for side,i in enumerate(split['children']):
            assert i not in seen and i<len(rows) and rows[i]['id']==i
            expected=[list(b) for b in box];expected[axis][1-side]=mid
            assert expected==[list(map(F,b)) for b in rows[i]['box']]
            c={k:restricted(v,axis,side) for k,v in pc.items()};cost+=3
            verify_bounds(rows[i],c);certificates[i]=c;seen.add(i);active.add(i);depth[i]=depth[parent]+1
            moments.append(F(rows[i]['whole_box_integral_without_H']))
        assert sum(moments)==F(rows[parent]['whole_box_integral_without_H'])
        check_trace()
        if len(seen) in (511,999,1499):print('audited',len(seen),'blocks',flush=True)
    assert seen==set(range(len(rows))) and active==set(report['leaves'])
    total=[sum(F(rows[i]['bounds'][e]) for i in active) for e in (0,1)]
    assert total==list(map(F,report['bounds']))
    assert report['sign']==('positive' if total[0]>0 else 'negative' if total[1]<0 else 'unresolved')
    assert len(rows)==report['processed_blocks']<=report['max_blocks']==1999
    assert max(depth.values())==report['maximum_depth']<=11
    assert report['half_restrictions']==3*len(report['splits']) and report['root_polynomial_transforms']==3
    assert report['radial_evaluations']==report['mechanical_solves']==report['new_partition_splits']==0
    groups={};signed_interior={'positive':F(),'negative':F(),'positive_count':0,'negative_count':0}
    for i in active:
        r=rows[i];g=groups.setdefault(r['state'],dict(count=0,width=F(),lower=F(),upper=F()))
        lo,hi=map(F,r['bounds']);g['count']+=1;g['width']+=hi-lo;g['lower']+=lo;g['upper']+=hi
        if r['state']=='inside_gamma_fixed' and lo:
            key='positive' if lo>0 else 'negative';signed_interior[key]+=lo;signed_interior[key+'_count']+=1
    for g in groups.values():
        for k in ['width','lower','upper']:g[k]=str(g[k])
    return dict(exact_root_polynomial_identities=3,independent_child_polynomial_restrictions=cost,
        original_coverage=cover,exact_partition_cover=True,whole_density_integral_conserved=True,
        signed_mask_bounds_verified=True,trace_and_final_sum_verified=True,sign=report['sign'],
        groups=groups,signed_exact_interior={k:str(v) if isinstance(v,F) else v for k,v in signed_interior.items()},processed_blocks=len(rows),seconds=time.monotonic()-start,radial_evaluations=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=Path('/workspace/genesis-regional-moment-region-01.json'));parser.add_argument('--output',type=Path,default=Path('/workspace/genesis-regional-moment-region-audit-01.json'));args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    path=args.input;r=json.loads(path.read_bytes())
    sourcepath=Path(r['source']);assert hashlib.sha256(sourcepath.read_bytes()).hexdigest()==r['source_sha256']
    result=audit(r,json.loads(sourcepath.read_bytes()));result['input_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['groups','original_coverage','signed_exact_interior']}),flush=True)
