"""Independent coverage, uncertainty, and exact-node work-ledger replay.

Adapted for the existing-grid-only experiment; verifies zero mixed work.
Reconstructs the node requests from saved rectangles, literal Gauss operands,
and saved ray partitions. Does not use the experimental planner/cache code.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from analysis.genesis_volume_ray_enclosure_validation.audit import audit_budget
from analysis.genesis_volume_ray_validation.audit import audit_ray,decode

BASE=((3,3),(5,5));MIXED=((3,5),(5,3))


def key(c):return (c['triangle'],*(decode(c[k]) for k in ('s0','s1','t0','t1')))


def children(c,axis):
    i,a,b,u,v=c
    if axis=='s':
        m=(a+b)/2;return [(i,a,m,u,v),(i,m,b,u,v)]
    assert axis=='t'
    m=(u+v)/2;return [(i,a,b,u,m),(i,a,b,m,v)]


def audit_cost(report,rays):
    history=[]
    if report['mode']=='diagnosis':
        history.extend((key(c),BASE) for c in report['cell_records'])
        history.extend((key(c),MIXED) for c in report['cell_records'] if c['diagnostic'] is not None)
    else:
        lazy=report.get('mixed_rules_on_selected_parents_only',False)
        pairs=BASE if lazy or report.get('directions_from_existing_5x5',False) else BASE+MIXED
        leaves={(i,F(),F(1),F(),F(1)) for i in range(3)}
        history.extend((c,pairs) for c in sorted(leaves))
        for r in report['refinements']:
            i,*values=r['parent'];parent=(int(i),*map(F,values));assert parent in leaves
            if lazy:history.append((parent,MIXED))
            added=children(parent,r['axis']);leaves.remove(parent);leaves.update(added)
            history.extend((c,pairs) for c in added)
        assert leaves=={key(c) for c in report['cell_records']}
    raymap={(r['triangle'],decode(r['t'])):r for r in rays}
    assert len(raymap)==len(rays)
    nodes={n:tuple(F(float(x)) for x in (np.polynomial.legendre.leggauss(n)[0]+1)/2) for n in (3,5)}
    seen=set();requests=mixed_new=0
    for (i,a,b,u,v),pairs in history:
        for ns,nt in pairs:
            for node_t in nodes[nt]:
                t=u+(v-u)*node_t;r=raymap[i,t]
                start=tuple(map(decode,r['start']));end=tuple(map(decode,r['end']))
                for interval in r['intervals']:
                    if not interval['integrated'] or interval['kind']=='root_box':continue
                    lo,hi=decode(interval['lo']),decode(interval['hi'])
                    for node_s in nodes[ns]:
                        s=a+(b-a)*node_s
                        rho=lo+(hi-lo)*(1-(1-s)**2)
                        N=tuple((1-rho)*x+rho*y for x,y in zip(start,end))
                        requests+=1
                        if N not in seen:
                            seen.add(N);mixed_new+=int((ns,nt) in MIXED)
    assert len(seen)==report['radial_line_evaluations']<=3400
    assert requests==report['quadrature_node_requests']
    assert requests-len(seen)==report['exact_node_cache_hits']
    assert mixed_new==report['mixed_rule_new_evaluations']
    assert report['scalar_bank_fiber_calls']==2*len(seen)
    return dict(independent_unique_source_nodes=len(seen),independent_requests=requests,
        exact_identical_node_reuses=requests-len(seen),mixed_new_evaluations=mixed_new,
        all_mixed_work_charged=True,cache_never_merges_distinct_exact_N=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    results=[]
    for directory in args.runs:
        report=json.loads((directory/'report.json').read_bytes());budget=audit_budget(report)
        path=directory/'ray_certificates.json';rays=json.loads(path.read_bytes());root_count=zero_checks=0
        for row in rays:
            r,z=audit_ray(report['source_geometry'],row);root_count+=r;zero_checks+=z
            for event in row['isolated'].values():
                for lo,hi in event['roots']:
                    lo,hi=decode(lo),decode(hi);assert lo==hi or hi-lo<=min(lo,1-hi)/2**40
            assert all(r['integrated'] and r['treatment']=='integral_enclosure' for r in row['intervals'] if r['kind']=='root_box')
        assert report['directions_from_existing_5x5'] and report['direction_new_radial_evaluations']==0
        cost=audit_cost(report,rays)
        assert cost['mixed_new_evaluations']==0
        result=dict(path=str(directory),case=report['case'],mode=report['mode'],
            lazy_mixed=report.get('mixed_rules_on_selected_parents_only',False),rays=len(rays),roots=root_count,
            zero_fiber_checks=zero_checks,ray_records_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**budget,**cost)
        results.append(result);print(json.dumps(result),flush=True)
    with args.output.open('x') as f:json.dump(dict(cases=results,scope='Analytic fixtures and recorded rays; no general angular remainder certificate'),f,indent=2);f.write('\n')


if __name__=='__main__':main()
