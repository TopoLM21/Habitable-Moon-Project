"""Independent DAG interval and rectangular-cover certificate replay.

No experimental evaluator, radial producer or mechanical solver is imported.
The mapping from source physics to the graph is an explicit algebraic contract;
the graph's exact arithmetic and all domain coverage are checked here.
"""
from fractions import Fraction as F
from pathlib import Path
from itertools import product
import argparse
import hashlib
import json
import time
from analysis.genesis_volume_mapped_validation.coverage_audit import audit as audit_coverage


def evaluate(nodes,box):
    values=[]
    for i,node in enumerate(nodes):
        if node[0]=='c':result=(F(node[1]),)*2
        elif node[0]=='x':assert node[1] in (0,1,2);result=box[node[1]]
        else:
            assert node[0] in ('+','*') and all(0<=v<i for v in node[1:])
            a,b=values[node[1]],values[node[2]]
            corners=[x+y if node[0]=='+' else x*y for x,y in product(a,b)]
            result=(min(corners),max(corners))
        values.append(result)
    return values


def audit(report):
    coverage=audit_coverage(report['source'],report['coverage'])
    root=[tuple(map(F,b)) for b in report['root']]
    decode=lambda x:F(*map(int,x))
    assert root[:2]==[tuple(map(decode,b)) for b in report['coverage']['coordinate_bounds'][1:]]
    assert root[2]==tuple(map(F,report['source']['section']['lambda_bounds']))
    assert root[0][0]>=0 and root[1][0]>=0 and root[0][1]+root[1][1]<=1
    # Noncollapsed source map and nonzero plane normal imply Gamma has zero volume.
    den=list(map(decode,report['coverage']['determinant_bounds']))
    assert den[0]>0 or den[1]<0
    assert any(report['source']['section']['normal_xyz'])
    graph=report['graph'];rows=report['records'];counts={};widths={};start=time.monotonic()
    for i,r in enumerate(rows):
        assert r['id']==i
        box=[tuple(map(F,b)) for b in r['box']];assert all(lo<hi for lo,hi in box)
        values=evaluate(graph['nodes'],box)
        ranges={key:values[value] for key,value in graph['outputs'].items()}
        assert ranges=={k:tuple(map(F,b)) for k,b in r['ranges'].items()}
        volume=F(1)
        for lo,hi in box:volume*=hi-lo
        q=ranges['sphere'];phi=ranges['gamma'];K=ranges['density_without_H']
        if q[0]>0:bound=(F(),F());state='outside'
        else:
            H=(F(1,2),)*2 if phi[0]>0 else (F(-1,2),)*2 if phi[1]<0 else (F(-1,2),F(1,2))
            products=[x*y for x,y in product(K,H)]
            state='inside' if q[1]<=0 else 'unknown_sphere'
            if state=='unknown_sphere':products.append(F())
            bound=(volume*min(products),volume*max(products))
        assert bound==tuple(map(F,r['bounds'])) and state==r['state']
    active={0};seen={0};assert rows[0]['depth']==0 and [tuple(map(F,b)) for b in rows[0]['box']]==root
    snapshots={t['blocks']:t for t in report['trace']}
    def check_trace(size):
        if size in snapshots:
            total=[sum(F(rows[i]['bounds'][end]) for i in active) for end in (0,1)]
            assert total==list(map(F,snapshots[size]['bounds']))
    check_trace(1)
    for split in report['splits']:
        assert split['parent'] in active;parent=rows[split['parent']];axis=split['axis']
        assert axis==parent['depth']%3 and len(split['children'])==2
        box=[list(map(F,b)) for b in parent['box']];mid=sum(box[axis])/2;active.remove(parent['id'])
        for side,identifier in enumerate(split['children']):
            assert identifier not in seen and identifier<len(rows)
            expected=[list(x) for x in box];expected[axis][1-side]=mid;child=rows[identifier]
            assert [list(map(F,b)) for b in child['box']]==expected and child['depth']==parent['depth']+1
            seen.add(identifier);active.add(identifier)
        check_trace(len(seen))
    assert active==set(report['leaves']) and seen==set(range(len(rows)))
    total=[sum(F(rows[i]['bounds'][end]) for i in active) for end in (0,1)]
    assert total==list(map(F,report['bounds']))
    assert report['sign']==('positive' if total[0]>0 else 'negative' if total[1]<0 else 'unresolved')
    for i in active:
        r=rows[i];phi=list(map(F,r['ranges']['gamma']));key=r['state']+('_gamma_uncertain' if phi[0]<=0<=phi[1] else '_gamma_fixed')
        counts[key]=counts.get(key,0)+1;widths[key]=widths.get(key,F())+F(r['bounds'][1])-F(r['bounds'][0])
    assert len(rows)==report['processed_blocks']<=report['max_blocks']<=2000
    assert max(r['depth'] for r in rows)==report['maximum_depth']<=report['max_depth']<=24
    assert report['graph_node_visits']==len(rows)*len(graph['nodes'])
    assert report['radial_evaluations']==report['mechanical_solves']==0
    return dict(coverage=coverage,exact_rectangle_cover=True,all_interval_operations_verified=True,
        trace_sums_verified=True,processed_blocks=len(rows),leaves=len(active),graph_node_visits=report['graph_node_visits'],
        leaf_classes=counts,width_by_class_j={k:str(v) for k,v in widths.items()},seconds=time.monotonic()-start,
        scope='Whole-domain inclusion conditional on documented source-to-density expression; no angular quadrature estimate used',
        radial_evaluations=0,mechanical_solves=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(json.loads(a.input.read_bytes()));result['input_sha256']=hashlib.sha256(a.input.read_bytes()).hexdigest()
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
