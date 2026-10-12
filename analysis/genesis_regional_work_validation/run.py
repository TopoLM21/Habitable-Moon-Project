"""Bounded first regional virtual-work certificate on the real saved source."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import heapq
import json
import shutil
import time
import numpy as np
from analysis.genesis_saved_operator_sensitivity.compare import load
from tectonics.genesis_volume_source_coverage import source_coverage
from analysis.genesis_volume_operand_validation.independent_coefficients import exact_source_coefficients,rational_array
from analysis.genesis_volume_operand_validation.independent_contractions import _memory_coefficients
from analysis.genesis_regional_work_validation.expression import build,interval_evaluate,block_bound


def point_checks(raw,probe,graph,root):
    """Compare distinct symbolic-gradient construction to saved-source algebra.

    Three polynomial-density evaluations, no radial integral or mechanical solve.
    Shared literal B-vertex conventions are explicit; this is not a new angular proof.
    """
    u,v=[sum(b)/2 for b in root[:2]];N=np.array([1-u-v,u,v],object)
    lm,lh=sum(root[2])/2,(root[2][1]-root[2][0])/2
    fields=exact_source_coefficients(raw,N,lm,lh)
    a=rational_array(probe['enrichment_increment_m']);C=rational_array(raw['elasticity_pa'])
    memory=_memory_coefficients(raw,N,fields['lam'],probe['mean_e0'],probe['mean_e1'])
    X=rational_array(raw['section']['triangle_m']);det=sum(X[0,j]*(X[1,(j+1)%3]*X[2,(j+2)%3]-X[1,(j+2)%3]*X[2,(j+1)%3]) for j in range(3))
    checks=[]
    for xi in (F(-1,2),F(),F(1,2)):
        Ga=sum((p@a*xi**i for i,p in enumerate(fields['Ghat_without_H'])),np.zeros(6,object))
        m=sum((p*xi**i for i,p in enumerate(memory)),np.zeros(6,object))
        target=det*(Ga@C@m);x=(u,v,lm+lh*xi)
        actual=interval_evaluate(graph,tuple((v,v) for v in x))['density_without_H']
        assert actual==(target,target)
        checks.append(dict(point=list(map(str,x)),density_without_H=str(target),exact_match=True))
    return checks


def solve(graph,root,max_blocks=2000,max_depth=24,max_seconds=120):
    if max_blocks>2000 or max_depth>24 or max_seconds>120:raise ValueError('First-series caps cannot increase')
    start=time.monotonic();rows=[];leaves=set();heap=[];splits=[];trace=[];total=[F(),F()]
    def add(box,depth):
        bounds,state,ranges=block_bound(graph,box);identifier=len(rows)
        row=dict(id=identifier,box=[[str(v) for v in b] for b in box],depth=depth,
                 bounds=list(map(str,bounds)),state=state,ranges={k:list(map(str,b)) for k,b in ranges.items()})
        rows.append(row);leaves.add(identifier);total[0]+=bounds[0];total[1]+=bounds[1]
        if depth<max_depth and bounds[1]>bounds[0]:heapq.heappush(heap,(-(bounds[1]-bounds[0]),identifier))
        return identifier
    add(root,0)
    while True:
        if len(rows) in (1,31,127,511,1999):trace.append(dict(blocks=len(rows),bounds=list(map(str,total)),seconds=time.monotonic()-start))
        # Require persistence at two recorded refinements, not just a convenient early bound.
        if len(trace)>=2 and all(F(t['bounds'][0])>0 for t in trace[-2:]):reason='positive_sign_confirmed';break
        if len(trace)>=2 and all(F(t['bounds'][1])<0 for t in trace[-2:]):reason='negative_sign_confirmed';break
        if len(rows)+2>max_blocks:reason='block_cap';break
        if time.monotonic()-start>=max_seconds:reason='time_cap';break
        if not heap:reason='depth_or_exact_leaf_limit';break
        _,identifier=heapq.heappop(heap);parent=rows[identifier];leaves.remove(identifier)
        total[0]-=F(parent['bounds'][0]);total[1]-=F(parent['bounds'][1])
        box=[tuple(map(F,b)) for b in parent['box']];axis=parent['depth']%3;lo,hi=box[axis];mid=(lo+hi)/2
        children=[]
        for interval in ((lo,mid),(mid,hi)):
            child=list(box);child[axis]=interval;children.append(add(tuple(child),parent['depth']+1))
        splits.append(dict(parent=identifier,axis=axis,children=children))
    if trace[-1]['blocks']!=len(rows):trace.append(dict(blocks=len(rows),bounds=list(map(str,total)),seconds=time.monotonic()-start))
    return dict(records=rows,splits=splits,leaves=sorted(leaves),trace=trace,bounds=list(map(str,total)),
                sign='positive' if total[0]>0 else 'negative' if total[1]<0 else 'unresolved',reason=reason,
                processed_blocks=len(rows),graph_node_visits=len(rows)*len(graph['nodes']),maximum_depth=max(r['depth'] for r in rows),
                seconds=time.monotonic()-start,max_blocks=max_blocks,max_depth=max_depth,max_seconds=max_seconds)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    free=shutil.disk_usage('/workspace').free
    if free<128*1024**2:raise ValueError('128 MiB reserve required')
    start=time.monotonic();data,probes,paths=load();case=data['fixture']['cases'][0];s=case['supports']['fixed_100m_clipped_support']
    raw={**case['source_inputs'],'seed_center_world_m':s['seed_center_world_m'],'seed_radius_m':s['seed_radius_m']}
    probe=probes['saved_accepted_interval_mean'];_,coverage=source_coverage(raw)
    decode=lambda p:F(*map(int,p))
    root=tuple(tuple(map(decode,b)) for b in coverage['coordinate_bounds'][1:])+ (tuple(map(F,raw['section']['lambda_bounds'])),)
    # Whole outer rectangle lies in the barycentric simplex in this selected case.
    assert root[0][0]>=0 and root[1][0]>=0 and root[0][1]+root[1][1]<=1
    graph=build(raw,probe);checks=point_checks(raw,probe,graph,root)
    preflight=dict(graph_nodes=len(graph['nodes']),density_checks=len(checks),free_bytes=free,build_seconds=time.monotonic()-start)
    print(json.dumps(preflight),flush=True)
    report=solve(graph,root)
    report.update(source=raw,probe=probe,coverage=coverage,graph=graph,point_checks=checks,
        preflight=preflight,root=[[str(v) for v in b] for b in root],radial_evaluations=0,mechanical_solves=0,
        independent_density_points=3,units='J, integrated over N1,N2,lambda on the entire saved support',
        work_definition='Integral H*(Ghat_without_H*a)^T*C*accepted_mean_memoryhat*det(X); z=0, other added virtual-work terms zero',
        omitted_terms='Unchanged whole-parent work and cohesive surface-creation cost; no finite-opening activation claim',
        physical_Gamma_selected=raw['physical_Gamma_selected'],source_activation_accepted=raw['source_activation_accepted'],
        free_bytes_before=free,free_bytes_before_write=shutil.disk_usage('/workspace').free,
        working_tolerance_changed=False,physical_admission=False,
        input_pins={k:dict(path=str(v),sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k,v in paths.items()},
        code_pins={str(v):hashlib.sha256(v.read_bytes()).hexdigest() for v in (Path(__file__),Path('analysis/genesis_regional_work_validation/expression.py'))})
    with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(sign=report['sign'],bounds=[float(F(v)) for v in report['bounds']],blocks=report['processed_blocks'],
                         seconds=report['seconds'],reason=report['reason'],radial_evaluations=0)),flush=True)


if __name__=='__main__':main()
