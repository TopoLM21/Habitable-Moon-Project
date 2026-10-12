"""Independent polynomial reciprocity, interval arithmetic and cover audit."""
from fractions import Fraction as F
from pathlib import Path
import argparse,hashlib,json,time,sys
import numpy as np
from analysis.genesis_regional_work_validation.audit import evaluate
from analysis.genesis_regional_work_validation.identity import expand
from analysis.genesis_volume_mapped_validation.coverage_audit import audit as coverage_audit
from analysis.genesis_volume_operand_validation.independent_coefficients import _literal_vertices,rational_array,exact_source_coefficients
sys.set_int_max_str_digits(30000)


def add(a,b,scale=F(1)):
    out=dict(a)
    for k,v in b.items():out[k]=out.get(k,F())+scale*v
    return {k:v for k,v in out.items() if v}


def multiply(a,b):
    out={}
    for k,v in a.items():
        for l,w in b.items():
            p=tuple(x+y for x,y in zip(k,l));out[p]=out.get(p,F())+v*w
    return {k:v for k,v in out.items() if v}


def audit(r):
    start=time.monotonic();loaded={}
    for k,pin in r['input_pins'].items():
        raw=Path(pin['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==pin['sha256'];loaded[k]=json.loads(raw)
    old=loaded['source'];assert loaded['source_audit']['input_sha256']==r['input_pins']['source']['sha256']
    assert loaded['source_audit']['new_case_exact_W15_membership']
    raw=old['source'];assert raw['seed_radius_m']==100
    assert r['selected_parent_dofs']==raw['selected_parent_dofs'] and len(set(r['selected_parent_dofs']))==18
    graph=r['graph'];base=old['graph'];assert graph['nodes'][:len(base['nodes'])]==base['nodes']
    assert all(graph['outputs'][k]==base['outputs'][k] for k in ('sphere','gamma'))
    assert r['mechanical_solves']==r['radial_integrations']==r['new_partition_splits']==0
    assert r['physical_admission'] is False and r['surface_law_selected'] is False
    extended={**graph,'outputs':{**graph['outputs'],**{k:v for k,v in base['outputs'].items() if k.startswith('strain_')}}}
    polys,max_terms=expand(extended)
    X=np.asarray(raw['section']['triangle_m']);B0,B1=_literal_vertices(X,np.asarray(raw['basis_xyz']))
    C=rational_array(raw['elasticity_pa']);assert np.array_equal(C,C.T)
    det=F(r['determinant']);xr=rational_array(X)
    assert det==sum(xr[0,j]*(xr[1,(j+1)%3]*xr[2,(j+2)%3]-xr[1,(j+2)%3]*xr[2,(j+1)%3]) for j in range(3))>0
    N=[{(0,0,0):F(1),(1,0,0):F(-1),(0,1,0):F(-1)}, {(1,0,0):F(1)}, {(0,1,0):F(1)}]
    parent=[]
    for i in range(18):
        col=[]
        for j in range(6):
            p={}
            for k in range(3):p=add(p,multiply(N[k],{(0,0,1):B0[k,j,i],(0,0,0):B1[k,j,i]}))
            col.append(p)
        parent.append(col)
    # All 18 identities, globally in N1,N2,lambda; exact rational coefficients.
    for i in range(18):
        expected={}
        for j in range(6):
            for k in range(6):
                if C[k,j]:expected=add(expected,multiply(polys['strain_'+str(k)],parent[i][j]),det*C[k,j])
        assert expected==polys['b_'+str(i)]
    q=list(map(F,r['q_diagnostic_m']));assert q==[F((-1)**i,8192*(i+1)) for i in range(18)]
    direct={}
    for i in range(18):direct=add(direct,polys['b_'+str(i)],q[i])
    assert direct==polys['adjoint_work']
    print(json.dumps(dict(polynomial_identities=19,seconds=time.monotonic()-start)),flush=True)
    # Additional source-coefficient regressions, not substituted for global proof.
    root=tuple(tuple(map(F,b)) for b in old['work']['root']);u,v,lam=[sum(b)/2 for b in root];half=(root[2][1]-root[2][0])/2
    fields=exact_source_coefficients(raw,np.array([1-u-v,u,v],object),lam,half)
    a=rational_array(old['probe']['enrichment_increment_m'])
    for xi in (F(-1,2),F(),F(1,2)):
        B=sum((m*xi**j for j,m in enumerate(fields['Bhat'])),np.zeros((6,18),object))
        G=sum((m*xi**j for j,m in enumerate(fields['Ghat_without_H'])),np.zeros((6,18),object))
        wanted=det*(B.T@C@(G@a));values=evaluate(graph['nodes'],[(x,x) for x in (u,v,lam+xi*half)])
        assert all(values[graph['outputs']['b_'+str(i)]]==(wanted[i],wanted[i]) for i in range(18))
    coverage=coverage_audit(raw,old['coverage'])
    records=old['work']['records'];active={0};seen={0}
    for split in old['work']['splits']:
        p=split['parent'];axis=split['axis'];assert p in active;active.remove(p)
        box=[list(map(F,b)) for b in records[p]['box']];mid=sum(box[axis])/2
        for side,i in enumerate(split['children']):
            assert i not in seen
            child=[b[:] for b in box];child[axis][1-side]=mid
            assert child==[list(map(F,b)) for b in records[i]['box']]
            seen.add(i);active.add(i)
    assert sorted(active)==old['work']['leaves']==[row['id'] for row in r['records']]
    assert tuple(tuple(map(F,b)) for b in records[0]['box'])==root
    totals={k:[F(),F()] for k in r['totals']};root_values=evaluate(graph['nodes'],root);evaluations=1
    for row in r['records']:
        box=[tuple(map(F,b)) for b in records[row['id']]['box']]
        values=root_values if row['root_fallback'] else evaluate(graph['nodes'],box);evaluations+=not row['root_fallback']
        sphere=values[graph['outputs']['sphere']];phi=values[graph['outputs']['gamma']]
        volume=(box[0][1]-box[0][0])*(box[1][1]-box[1][0])*(box[2][1]-box[2][0])
        for k in totals:
            value=values[graph['outputs'][k]]
            if sphere[0]>0:bound=(F(),F())
            else:
                signs=[F(1,2)] if phi[0]>0 else [F(-1,2)] if phi[1]<0 else [F(-1,2),F(1,2)]
                ends=[x*h for x in value for h in signs]
                if sphere[1]>0:ends.append(F())
                bound=(min(ends)*volume,max(ends)*volume)
            assert bound==tuple(map(F,row['bounds'][k]))
            for e in (0,1):totals[k][e]+=bound[e]
    assert totals=={k:list(map(F,v)) for k,v in r['totals'].items()}
    forward=[F(),F()]
    for i in range(18):
        ends=[q[i]*x for x in totals['b_'+str(i)]];forward[0]+=min(ends);forward[1]+=max(ends)
    assert forward==list(map(F,r['forward_work_j']))
    overlap=[max(forward[0],totals['adjoint_work'][0]),min(forward[1],totals['adjoint_work'][1])]
    assert overlap==list(map(F,r['reciprocal_work_intersection_j'])) and overlap[0]<=overlap[1]
    assert evaluations==r['graph_evaluations']<=797
    return dict(status='PASS',source_pins=True,source_to_parent_action_polynomial_identities=18,
        exact_global_mutual_work_identity=True,reciprocity_defect_j=['0','0'],source_point_regressions=3,
        independent_interval_evaluations=evaluations,exact_full_cover=True,coverage=coverage,
        all_signed_bounds_and_sums_verified=True,maximum_polynomial_terms=max_terms,
        seconds=time.monotonic()-start,mechanical_solves=0,physical_admission=False,
        scope='Represented frozen source contract; reciprocity is not magnitude accuracy or coupled equilibrium')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Fresh path required')
    data=args.input.read_bytes();out=audit(json.loads(data));out['input_sha256']=hashlib.sha256(data).hexdigest()
    with args.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k!='coverage'}),flush=True)
