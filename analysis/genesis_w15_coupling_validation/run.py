"""Single bounded fixed-cover series; all unfinished leaves retain root bounds."""
from fractions import Fraction as F
from pathlib import Path
import argparse,hashlib,json,time,sys
from .core import build,bounds,contract,observed,Q_DIAGNOSTIC
from analysis.genesis_regional_work_validation.expression import interval_evaluate
sys.set_int_max_str_digits(30000)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh path required')
    start=time.monotonic()
    paths={'source':Path('/workspace/genesis-w15-mode1-joint-01.json'),
           'source_audit':Path('/workspace/genesis-w15-mode1-joint-audit-01.json')}
    loaded={k:json.loads(p.read_bytes()) for k,p in paths.items()}
    old=loaded['source'];receipt=loaded['source_audit']
    assert hashlib.sha256(paths['source'].read_bytes()).hexdigest()==receipt['input_sha256']
    assert receipt['new_case_exact_W15_membership'] and receipt['exact_squared_analytic_kernel_check']
    for pin in old['input_pins'].values():assert hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['sha256']
    root=tuple(tuple(map(F,b)) for b in old['work']['root'])
    graph,_,_,det=build(old['source'],old['probe'],tuple(sum(b)/2 for b in root))
    root_ranges=interval_evaluate(graph,root);evaluations=1
    leaves=old['work']['leaves'];assert len(leaves)==796
    totals={k:[F(),F()] for k in root_ranges if k not in ('sphere','gamma')};rows=[];trace=[]
    for i in leaves:
        box=tuple(tuple(map(F,b)) for b in old['work']['records'][i]['box'])
        fallback=time.monotonic()-start>=120
        ranges=root_ranges if fallback else interval_evaluate(graph,box)
        evaluations+=not fallback
        local=bounds(ranges,box)
        rows.append(dict(id=i,root_fallback=fallback,bounds={k:list(map(str,v)) for k,v in local.items()}))
        for k,v in local.items():
            for e in (0,1):totals[k][e]+=v[e]
        if len(rows) in (1,64,256,512,796):
            t=dict(leaves=len(rows),seconds=time.monotonic()-start,evaluations=evaluations)
            trace.append(t);print(json.dumps(t),flush=True)
    forward=contract([totals['b_'+str(i)] for i in range(18)])
    adjoint=totals['adjoint_work'];overlap=(max(forward[0],adjoint[0]),min(forward[1],adjoint[1]));assert overlap[0]<=overlap[1]
    report=dict(format='w15-coupling-vector-fixed-cover-1',graph=graph,determinant=det,records=rows,
        totals={k:list(map(str,v)) for k,v in totals.items()},forward_work_j=list(map(str,forward)),
        reciprocal_work_intersection_j=list(map(str,overlap)),q_diagnostic_m=list(map(str,Q_DIAGNOSTIC)),
        observables={k:observed(v) for k,v in totals.items()},trace=trace,
        seconds=time.monotonic()-start,graph_evaluations=evaluations,graph_nodes=len(graph['nodes']),
        max_graph_evaluations=797,max_compute_seconds=120,external_process_limit_seconds=180,
        root_fallback_leaves=sum(r['root_fallback'] for r in rows),new_partition_splits=0,
        mechanical_solves=0,radial_integrations=0,physical_admission=False,surface_law_selected=False,
        global_parent_ndof=252,local_components=18,units='N; diagnostic mutual work J',
        selected_parent_dofs=old['source']['selected_parent_dofs'],
        input_pins={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()})
    with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(seconds=report['seconds'],evaluations=evaluations,fallback=report['root_fallback_leaves'],
        signs=[report['observables']['b_'+str(i)]['sign'] for i in range(18)],adjoint_work_j=list(map(float,adjoint)))),flush=True)


if __name__=='__main__':main()
