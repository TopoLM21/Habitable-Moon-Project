"""One fixed-leaf series; no adaptive or mechanical solve."""
from pathlib import Path
from fractions import Fraction as F
import argparse,hashlib,json,time,sys
sys.set_int_max_str_digits(30000)
import numpy as np
from .expression import build
from .bounds import ldl,block,responses
from analysis.genesis_regional_work_validation.expression import interval_evaluate
from analysis.genesis_volume_operand_validation.independent_coefficients import exact_source_coefficients,rational_array


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    paths={k:Path(v) for k,v in dict(region='/workspace/genesis-regional-moment-region-01.json',
        region_audit='/workspace/genesis-regional-moment-region-audit-01.json',
        work='analysis/genesis_gamma_strip_validation/summary.json',
        fixture='analysis/genesis_volume_exact_lambda_validation/evidence/actual_control/source_fixture.json').items()}
    region=json.loads(paths['region'].read_bytes());receipt=json.loads(paths['region_audit'].read_bytes())
    assert hashlib.sha256(paths['region'].read_bytes()).hexdigest()==receipt['input_sha256']
    paths['source']=Path(region['source']);source=json.loads(paths['source'].read_bytes())
    assert hashlib.sha256(paths['source'].read_bytes()).hexdigest()==region['source_sha256']
    work=json.loads(paths['work'].read_bytes());W=list(map(F,work['regional_bounds']));assert W[0]>0
    raw,probe=source['source'],source['probe'];root=tuple(tuple(map(F,b)) for b in source['root'])
    start=time.monotonic();graph=build(raw,probe,tuple(sum(b)/2 for b in root))
    assert list(map(list,graph['nodes']))==source['graph']['nodes']
    assert all(graph['outputs'][k]==v for k,v in source['graph']['outputs'].items())
    L,D=ldl(raw['elasticity_pa']);X=rational_array(raw['section']['triangle_m'])
    det=sum(X[0,j]*(X[1,(j+1)%3]*X[2,(j+2)%3]-X[1,(j+2)%3]*X[2,(j+1)%3]) for j in range(3));assert det>0
    # Three independent density evaluations, no radial integration.
    u,v,lam=[sum(b)/2 for b in root];N=np.array([1-u-v,u,v],object);lh=(root[2][1]-root[2][0])/2
    f=exact_source_coefficients(raw,N,lam,lh);a=rational_array(probe['enrichment_increment_m']);checks=[]
    for xi in (F(-1,2),F(),F(1,2)):
        g=sum((m@a*xi**i for i,m in enumerate(f['Ghat_without_H'])),np.zeros(6,object))
        rr=interval_evaluate(graph,tuple((z,z) for z in (u,v,lam+xi*lh)))
        assert all(rr['strain_'+str(i)]==(g[i],g[i]) for i in range(6));checks.append(dict(xi=str(xi),matched=True))
    fixture=json.loads(paths['fixture'].read_bytes());support=fixture['cases'][0]['supports']['fixed_100m_clipped_support'];B=np.asarray(support['original_declared_added_basis'],float)
    af=np.asarray(probe['enrichment_increment_m']);fit=B@np.linalg.lstsq(B,af,rcond=None)[0];residual=af-fit
    eligibility=dict(basis=B.tolist(),basis_shape=list(B.shape),kernel_tied=support['original_kernel_coordinates_tied'],
        norm_residual_m=float(np.linalg.norm(residual)),relative_residual=float(np.linalg.norm(residual)/np.linalg.norm(af)),
        released_mode_admitted=False,unchanged_full18_diagnostic_only=True)
    root_ranges=interval_evaluate(graph,root);rows=[];totals={'stiffness_a_j':[F(),F()],'memory_norm_j':[F(),F()]};evaluations=1
    for i in region['leaves']:
        parent=region['records'][i];box=tuple(tuple(map(F,b)) for b in parent['box']);fallback=time.monotonic()-start>=120
        ranges=root_ranges if fallback else interval_evaluate(graph,box);evaluations+=not fallback
        result=block(ranges,box,parent['state'],L,D,det)
        rows.append(dict(id=i,box=parent['box'],geometry_state=parent['state'],root_fallback=fallback,
            ranges={k:list(map(str,v)) for k,v in ranges.items()},**result))
        for key in totals:
            for e in (0,1):totals[key][e]+=F(result[key][e])
    assert evaluations<=866 and len(rows)==865
    ell2=sum(x*x for x in a);Eu=totals['memory_norm_j'][1];assert Eu>0
    lower_from_work=W[0]**2/Eu
    K=[max(totals['stiffness_a_j'][0],lower_from_work),totals['stiffness_a_j'][1]];assert 0<K[0]<=K[1]
    result=dict(source=raw,probe=probe,graph=graph,root=source['root'],determinant=str(det),
        L=[[str(x) for x in row] for row in L],D=list(map(str,D)),point_checks=checks,eligibility=eligibility,
        records=rows,work_bounds_j=list(map(str,W)),ell_squared_m2=str(ell2),
        raw_totals={k:list(map(str,v)) for k,v in totals.items()},cauchy_lower_j=str(lower_from_work),
        stiffness_a_j=list(map(str,K)),stiffness_v_n_m=[str(x/ell2) for x in K],responses=responses(W,K,ell2),
        graph_evaluations=evaluations,max_graph_evaluations=866,max_seconds=120,seconds=time.monotonic()-start,
        radial_evaluations=0,mechanical_solves=0,physical_admission=False,tolerance_changed=False,
        input_pins={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()})
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(K_a_j=[float(x) for x in K],K_v_n_m=[float(x/ell2) for x in K],
        E_upper_j=float(Eu),seconds=result['seconds'],evaluations=evaluations,relative_projection_residual=eligibility['relative_residual'])),flush=True)


if __name__=='__main__':main()
