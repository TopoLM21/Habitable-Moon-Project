"""Independent inverse Bernstein transforms, antiderivatives, sums and cover."""
from fractions import Fraction as F
from pathlib import Path
from math import prod
import argparse,hashlib,json,time,sys
from analysis.genesis_regional_work_validation.identity import expand
from analysis.genesis_regional_moment_validation.audit import recover
from analysis.genesis_w15_coupling_validation.audit import add
sys.set_int_max_str_digits(30000)


def audit(r):
    start=time.monotonic();loaded={}
    for k,pin in r['input_pins'].items():
        raw=Path(pin['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==pin['sha256'];loaded[k]=json.loads(raw)
    old=loaded['vector'];geometry=loaded['geometry'];receipt=loaded['vector_audit']
    assert receipt['input_sha256']==r['input_pins']['vector']['sha256'] and receipt['all_signed_bounds_and_sums_verified']
    assert old['input_pins']['source']['sha256']==r['input_pins']['geometry']['sha256']
    polys,_=expand(old['graph']);q=list(map(F,old['q_diagnostic_m']));direct={}
    for i in range(18):direct=add(direct,polys['b_'+str(i)],q[i])
    assert direct==polys['adjoint_work']
    geo_rows=geometry['work']['records'];active={0};seen={0}
    for split in geometry['work']['splits']:
        p=split['parent'];assert p in active;active.remove(p);axis=split['axis'];box=[list(map(F,b)) for b in geo_rows[p]['box']];mid=sum(box[axis])/2
        for side,i in enumerate(split['children']):
            assert i not in seen;child=[b[:] for b in box];child[axis][1-side]=mid
            assert child==[list(map(F,b)) for b in geo_rows[i]['box']];seen.add(i);active.add(i)
    assert sorted(active)==[row['id'] for row in r['records']]==geometry['work']['leaves']
    prior={row['id']:row for row in old['records']};totals={k:[F(),F()] for k in old['totals']};inverse_count=0;moments=0
    width=lambda i:sum(F(prior[i]['bounds']['b_'+str(j)][1])-F(prior[i]['bounds']['b_'+str(j)][0]) for j in range(18))
    selected=sorted((i for i in active if geo_rows[i]['state']=='inside_gamma_fixed'),key=lambda i:(-width(i),i))[:2]
    assert selected==[g['id'] for g in r['gate']]
    for row in r['records']:
        box=[tuple(map(F,b)) for b in geo_rows[row['id']]['box']];volume=prod(hi-lo for lo,hi in box)
        previous={k:tuple(map(F,v)) for k,v in prior[row['id']]['bounds'].items()}
        if row['method']=='prior_verified_bound':expected=previous
        else:
            for key,cert in row['geometry_certificates'].items():assert recover(cert,box)==polys[key];inverse_count+=1
            sphere=list(map(F,row['geometry_certificates']['sphere']['coefficients']));gamma=list(map(F,row['geometry_certificates']['gamma']['coefficients']))
            assert min(gamma)!=0 or max(gamma)!=0
            fresh={}
            for key in totals:
                cert=row['density_certificates'].get(key)
                if cert is not None:assert recover(cert,box)==polys[key];inverse_count+=1
                if min(sphere)>0:
                    assert row['method']=='exact_outside';bound=(F(),F())
                elif max(sphere)<=0 and (min(gamma)>=0 or max(gamma)<=0):
                    assert row['method']=='exact_inside_moments'
                    # Independent termwise three-dimensional antiderivative.
                    exact=sum(v*prod((hi**(n+1)-lo**(n+1))/F(n+1) for n,(lo,hi) in zip(powers,box)) for powers,v in polys[key].items());moments+=1
                    if cert is not None:assert exact==volume*sum(map(F,cert['coefficients']))/len(cert['coefficients'])
                    value=(1 if min(gamma)>=0 else -1)*exact/2;bound=(value,value)
                else:
                    assert row['method']=='signed_bernstein_mask' and cert is not None
                    values=list(map(F,cert['coefficients']));negative=volume*sum(x for x in values if x<0)/len(values);positive=volume*sum(x for x in values if x>0)/len(values)
                    if min(gamma)>=0:bound=(negative/2,positive/2)
                    elif max(gamma)<=0:bound=(-positive/2,-negative/2)
                    else:bound=(-(positive-negative)/2,(positive-negative)/2)
                assert bound==tuple(map(F,row['moment_bounds'][key]));fresh[key]=bound
            expected={k:(max(previous[k][0],fresh[k][0]),min(previous[k][1],fresh[k][1])) for k in totals}
        for key,bound in expected.items():
            assert bound[0]<=bound[1] and bound==tuple(map(F,row['bounds'][key]))
            for e in (0,1):totals[key][e]+=bound[e]
    for gate in r['gate']:
        row=next(x for x in r['records'] if x['id']==gate['id'])
        improved=sum(F(row['bounds']['b_'+str(j)][1])-F(row['bounds']['b_'+str(j)][0]) < F(prior[row['id']]['bounds']['b_'+str(j)][1])-F(prior[row['id']]['bounds']['b_'+str(j)][0]) for j in range(18))
        assert gate['strictly_narrower_components']==improved==18
    assert r['vector_control_enabled'] is True
    assert totals=={k:list(map(F,v)) for k,v in r['totals'].items()}
    forward=[sum(min(q[i]*x for x in totals['b_'+str(i)]) for i in range(18)),sum(max(q[i]*x for x in totals['b_'+str(i)]) for i in range(18))]
    assert forward==list(map(F,r['forward_work_j']))
    overlap=[max(forward[0],totals['adjoint_work'][0]),min(forward[1],totals['adjoint_work'][1])]
    assert overlap==list(map(F,r['mutual_work_intersection_j'])) and overlap[0]<=overlap[1]
    assert r['mechanical_solves']==r['new_partition_splits']==0 and not r['physical_admission']
    return dict(status='PASS',exact_full_cover=True,global_mutual_work_identity=True,reciprocity_defect_j=['0','0'],
        independent_inverse_polynomial_checks=inverse_count,independent_exact_moments=moments,
        all_signed_bounds_intersections_sums_verified=True,addressed_improvement_gate_verified=True,
        previous_verified_bounds_retained=True,seconds=time.monotonic()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    data=args.input.read_bytes();out=audit(json.loads(data));out['input_sha256']=hashlib.sha256(data).hexdigest()
    with args.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out),flush=True)
