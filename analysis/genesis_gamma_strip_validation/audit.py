"""Independent polynomial, strip-cover, replacement and whole-region sum audit.

No root-window selector, experimental conversion or integration routine is used.
Unchanged regional contributions reuse a hash-bound prior audit receipt.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import argparse,hashlib,json,time
from analysis.genesis_regional_work_validation.identity import expand
from analysis.genesis_regional_moment_validation.audit import audit as audit_polynomial


def value(poly,point):
    out=F()
    for exponents,coefficient in poly.items():
        term=coefficient
        for a in range(3):term*=point[a]**exponents[a]
        out+=term
    return out


def old_cover(old,source):
    assert old['root']==source['root']==old['records'][0]['box']
    active={0};seen={0};rows=old['records']
    for split in old['splits']:
        parent=split['parent'];axis=split['axis'];assert parent in active and axis in (0,1,2)
        active.remove(parent);box=[list(map(F,b)) for b in rows[parent]['box']];mid=sum(box[axis])/2
        assert len(split['children'])==2
        for side,i in enumerate(split['children']):
            assert i not in seen and i<len(rows) and rows[i]['id']==i
            expected=[list(b) for b in box];expected[axis][1-side]=mid
            assert expected==[list(map(F,b)) for b in rows[i]['box']]
            active.add(i);seen.add(i)
    assert active==set(old['leaves']) and seen==set(range(len(rows)))
    assert [sum(F(rows[i]['bounds'][e]) for i in active) for e in (0,1)]==list(map(F,old['bounds']))


def strip_cover(plan,parent_box,gamma):
    axis=plan['axis']
    if axis is None:
        assert plan['quarters']==[];return []
    assert axis in (0,1,2) and len(plan['quarters'])==4
    assert all(all(n<=1 for n in k) for k in gamma)
    other=[a for a in range(3) if a!=axis];expected_children=[]
    for sides,q in zip(product((0,1),repeat=2),plan['quarters']):
        expected=[list(b) for b in parent_box]
        for a,side in zip(other,sides):
            mid=sum(parent_box[a])/2;expected[a][1-side]=mid
        assert expected==[list(map(F,b)) for b in q['box']]
        roots=[];slopes=[]
        for corner in product((0,1),repeat=2):
            point=[F(),F(),F()]
            for a,side in zip(other,corner):point[a]=expected[a][side]
            point[axis]=expected[axis][0];left=value(gamma,point)
            point[axis]=expected[axis][1];right=value(gamma,point)
            slopes.append(right-left);assert right!=left;roots.append(-left/(right-left))
        assert all(x>0 for x in slopes) or all(x<0 for x in slopes)
        assert roots==list(map(F,q['window']['corner_roots']))
        low=max(F(),min(F(1),min(roots)));high=max(F(),min(F(1),max(roots)))
        assert [low,high]==list(map(F,q['window']['normalized_root_bounds']))
        lo,hi=expected[axis];cuts=sorted(set([lo,lo+(hi-lo)*low,lo+(hi-lo)*high,hi]))
        assert len(q['slabs'])==len(cuts)-1
        for (lo,hi),child in zip(zip(cuts,cuts[1:]),q['slabs']):
            child_expected=[list(b) for b in expected];child_expected[axis]=[lo,hi]
            assert child_expected==[list(map(F,b)) for b in child]
            expected_children.append(child)
    return expected_children


def audit(report,old,source,receipt):
    start=time.monotonic()
    assert receipt['exact_partition_cover'] and receipt['signed_mask_bounds_verified'] and receipt['trace_and_final_sum_verified']
    old_cover(old,source);polys,_=expand(source['graph'])
    fixed=(1258,1255,1270,1267,1257,1256,1269,1268,1301,1300,1305,1304,1230,1225,1229,1226)
    assert report['ids']==list(fixed) and len(set(fixed))==16
    records=report['records'];assert [r['id'] for r in records]==list(fixed[:len(records)])
    total=list(map(F,old['bounds']));localblocks=0;childcount=0;counts={'positive':0,'negative':0,'unresolved':0};lower_lift=F();upper_drop=F()
    for row in records:
        i=row['id'];parent=old['records'][i];assert i in old['leaves'] and parent['state']=='gamma_uncertain_inside'
        assert row['old_bounds']==parent['bounds']
        expected=strip_cover(row['plan'],[list(map(F,b)) for b in parent['box']],polys['gamma'])
        children=row['children'];assert len(children)<=len(expected)
        subtotal=[F(),F()];moment=F()
        for r,b in zip(children,expected):
            assert r['box']==b;audit_polynomial(polys,r);childcount+=1
            for e in (0,1):subtotal[e]+=F(r['bounds'][e])
            moment+=F(r['whole_box_integral_without_H'])
        quarters=0;remaining=len(children)
        for q in row['plan']['quarters']:
            quarters+=1
            if remaining<len(q['slabs']):break
            remaining-=len(q['slabs'])
        if 'quarters_started' in row:
            assert 0<=row['quarters_started']<=quarters
            assert len(children)<=sum(len(q['slabs']) for q in row['plan']['quarters'][:row['quarters_started']])
            quarters=row['quarters_started']
        localblocks+=quarters+len(children)
        prior=list(map(F,parent['bounds']))
        if row['complete']:
            assert len(children)==len(expected)>0 and quarters==4
            assert moment==F(parent['whole_box_integral_without_H'])
            assert subtotal==list(map(F,row['summed_child_bounds']))
            new=[max(prior[0],subtotal[0]),min(prior[1],subtotal[1])]
        else:
            assert row['summed_child_bounds'] is None;new=prior
        assert new[0]<=new[1] and new==list(map(F,row['accepted_bounds']))
        sign='positive' if new[0]>0 else 'negative' if new[1]<0 else 'unresolved'
        assert sign==row['sign'];counts[sign]+=1
        for e in (0,1):total[e]+=new[e]-prior[e]
        lower_lift+=new[0]-prior[0];upper_drop+=prior[1]-new[1]
        print('audited selected block',i,sign,flush=True)
    assert total==list(map(F,report['regional_bounds']))
    replacements={r['id']:list(map(F,r['accepted_bounds'])) for r in records}
    direct=[sum(replacements.get(i,list(map(F,old['records'][i]['bounds'])))[e] for i in old['leaves']) for e in (0,1)]
    assert direct==total
    sign='positive' if total[0]>0 else 'negative' if total[1]<0 else 'unresolved'
    assert sign==report['regional_sign']
    assert report['new_local_blocks']==localblocks<=report['max_new_local_blocks']==256
    assert report['old_processed_blocks']==old['processed_blocks']==1729
    assert report['combined_processed_blocks']==1729+localblocks<=1999
    assert report['polynomial_transforms']==3*(len(records)+childcount)
    assert localblocks+len(records)<=256 and 1729+localblocks+len(records)<=1999
    assert report['radial_evaluations']==report['mechanical_solves']==0
    return dict(original_cover_verified=True,local_strip_covers_verified=True,
        exact_child_polynomial_identities=3*childcount,whole_density_integral_conserved=True,
        unchanged_contributions_reused_from_hash_bound_audit=True,regional_sum_verified_two_ways=True,
        selected_signs=counts,unattempted_ids=list(fixed[len(records):]),regional_sign=sign,
        lower_lift_j=str(lower_lift),upper_drop_j=str(upper_drop),regional_bounds=list(map(str,total)),
        seconds=time.monotonic()-start,new_local_blocks=localblocks,combined_processed_blocks=1729+localblocks,
        parent_rechecks=len(records),new_block_work_with_parent_rechecks=localblocks+len(records),
        combined_block_work_with_parent_rechecks=1729+localblocks+len(records))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    r=json.loads(args.input.read_bytes());oldpath=Path(r['source']);old=json.loads(oldpath.read_bytes())
    assert hashlib.sha256(oldpath.read_bytes()).hexdigest()==r['source_sha256']
    ap=Path(r['prior_audit']);assert hashlib.sha256(ap.read_bytes()).hexdigest()==r['prior_audit_sha256'];receipt=json.loads(ap.read_bytes())
    assert receipt['input_sha256']==r['source_sha256']
    sp=Path(old['source']);assert hashlib.sha256(sp.read_bytes()).hexdigest()==old['source_sha256']
    result=audit(r,old,json.loads(sp.read_bytes()),receipt);result['input_sha256']=hashlib.sha256(args.input.read_bytes()).hexdigest()
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('regional_bounds','lower_lift_j','upper_drop_j')}),flush=True)
