"""Independent arithmetic replay, SPD identity and source-bound Cauchy check."""
from fractions import Fraction as F
from pathlib import Path
import argparse,hashlib,json,time,sys
sys.set_int_max_str_digits(30000)
from analysis.genesis_regional_work_validation.audit import evaluate
from analysis.genesis_regional_work_validation.identity import expand
from analysis.genesis_gamma_strip_validation.audit import old_cover


def witness(B,a):
    rows=[list(map(F,col)) for col in zip(*B)];m=len(rows);n=len(a);pivots=[];rank=0
    for column in range(n):
        k=next((i for i in range(rank,m) if rows[i][column]),None)
        if k is None:continue
        rows[rank],rows[k]=rows[k],rows[rank];v=rows[rank][column];rows[rank]=[x/v for x in rows[rank]]
        for i in range(m):
            if i!=rank:
                v=rows[i][column]
                if v:rows[i]=[x-v*y for x,y in zip(rows[i],rows[rank])]
        pivots.append(column);rank+=1
        if rank==m:break
    for free in (i for i in range(n) if i not in pivots):
        w=[F()]*n;w[free]=F(1)
        for i,p in enumerate(pivots):w[p]=-rows[i][free]
        if sum(x*F(y) for x,y in zip(w,a)):
            scale=max(abs(x) for x in w);return [x/scale for x in w]
    raise AssertionError('No nonmembership witness')


def squared_form(ranges,L,D):
    terms=[]
    for column in range(6):
        low=high=F()
        for i in range(6):
            c=L[i][column];a,b=ranges[i]
            low+=c*(a if c>=0 else b);high+=c*(b if c>=0 else a)
        candidates=[low*low,high*high]
        if low<=0<=high:candidates.append(F())
        terms.append((D[column]*min(candidates),D[column]*max(candidates)))
    return tuple(sum(t[e] for t in terms) for e in (0,1))


def audit(r):
    start=time.monotonic();loaded={}
    for k,pin in r['input_pins'].items():
        path=Path(pin['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==pin['sha256'];loaded[k]=json.loads(path.read_bytes())
    old,source=loaded['region'],loaded['source'];receipt=loaded['region_audit'];old_cover(old,source)
    assert receipt['input_sha256']==r['input_pins']['region']['sha256'] and receipt['signed_mask_bounds_verified']
    work=loaded['work'];assert work['artifacts']['genesis-regional-moment-region-01.json']['sha256']==r['input_pins']['region']['sha256']
    wp=work['artifacts']['genesis-gamma-strips-audit-01.json'];path=Path(wp['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==wp['sha256']
    wa=json.loads(path.read_bytes());assert wa['regional_sum_verified_two_ways'] and wa['regional_bounds']==r['work_bounds_j']
    assert r['source']==source['source'] and r['probe']==source['probe']
    assert r['graph']['nodes']==source['graph']['nodes'] and r['root']==source['root']
    assert all(r['graph']['outputs'][k]==v for k,v in source['graph']['outputs'].items())
    L=[list(map(F,row)) for row in r['L']];D=list(map(F,r['D']));C=[list(map(F,row)) for row in r['source']['elasticity_pa']]
    assert all(d>0 for d in D)
    assert C==[[sum(L[i][k]*D[k]*L[j][k] for k in range(6)) for j in range(6)] for i in range(6)]
    X=[list(map(F,row)) for row in r['source']['section']['triangle_m']]
    det=X[0][0]*X[1][1]*X[2][2]+X[0][1]*X[1][2]*X[2][0]+X[0][2]*X[1][0]*X[2][1]-X[0][2]*X[1][1]*X[2][0]-X[0][1]*X[1][0]*X[2][2]-X[0][0]*X[1][2]*X[2][1]
    assert det==F(r['determinant'])>0
    # Bind Cauchy-Schwarz to exactly the already certified work integrand,
    # by global coefficient identity, not a handful of density samples.
    polys,_=expand(r['graph']);cross={}
    for i in range(6):
        for j in range(6):
            factor=det*C[i][j]
            if not factor:continue
            for powers,x in polys['strain_'+str(i)].items():
                for other,y in polys['memory_'+str(j)].items():
                    key=tuple(a+b for a,b in zip(powers,other))
                    cross[key]=cross.get(key,F())+factor*x*y
    assert {k:v for k,v in cross.items() if v}==polys['density_without_H']
    totals={k:[F(),F()] for k in ('stiffness_a_j','memory_norm_j')};count=1
    assert [x['id'] for x in r['records']]==old['leaves']
    root=[tuple(map(F,b)) for b in r['root']];rootvalues=evaluate(r['graph']['nodes'],root)
    for row in r['records']:
        oldrow=old['records'][row['id']];assert row['box']==oldrow['box'] and row['geometry_state']==oldrow['state']
        box=[tuple(map(F,b)) for b in row['box']];values=rootvalues if row['root_fallback'] else evaluate(r['graph']['nodes'],box);count+=not row['root_fallback']
        ranges={k:values[v] for k,v in r['graph']['outputs'].items()};assert ranges=={k:tuple(map(F,v)) for k,v in row['ranges'].items()}
        volume=(box[0][1]-box[0][0])*(box[1][1]-box[1][0])*(box[2][1]-box[2][0])
        for name,prefix,scale in [('stiffness_a_j','strain',F(1,4)),('memory_norm_j','memory',F(1))]:
            if oldrow['state']=='outside':bound=(F(),F())
            else:
                a,b=squared_form([ranges[prefix+'_'+str(i)] for i in range(6)],L,D)
                if oldrow['state'] not in ('inside_gamma_fixed','gamma_uncertain_inside'):a=F()
                bound=(a*volume*det*scale,b*volume*det*scale)
            assert bound==tuple(map(F,row[name]))
            for e in (0,1):totals[name][e]+=bound[e]
    assert {k:list(map(str,v)) for k,v in totals.items()}==r['raw_totals']
    W=list(map(F,r['work_bounds_j']));assert W[0]>0
    cauchy=W[0]**2/totals['memory_norm_j'][1];assert cauchy==F(r['cauchy_lower_j'])
    K=[max(totals['stiffness_a_j'][0],cauchy),totals['stiffness_a_j'][1]];assert K==list(map(F,r['stiffness_a_j'])) and 0<K[0]<=K[1]
    a=list(map(F,r['probe']['enrichment_increment_m']));ell2=sum(x*x for x in a);assert ell2==F(r['ell_squared_m2'])
    assert [x/ell2 for x in K]==list(map(F,r['stiffness_v_n_m']))
    for row,t in zip(r['responses'],[F(-1,4),F(1,4),F(-1,8),F(1,8),F(-1,16),F(1,16)]):
        assert F(row['t'])==t
        force_delta=sorted(t*x for x in K);linear=sorted(t*x for x in W)
        assert force_delta==list(map(F,row['incremental_directional_force_times_a_j']))
        assert linear==list(map(F,row['internal_virtual_work_j']))
        assert [W[e]+force_delta[e] for e in (0,1)]==list(map(F,row['total_directional_force_times_a_j']))
        assert [linear[e]+t*t*K[e]/2 for e in (0,1)]==list(map(F,row['frozen_elastic_energy_change_j']))
        assert K==list(map(F,row['centered_force_secant_a_j']))==list(map(F,row['centered_energy_curvature_a_j']))
    B=r['eligibility']['basis'];assert B==loaded['fixture']['cases'][0]['supports']['fixed_100m_clipped_support']['original_declared_added_basis']
    w=witness(B,a)
    assert all(sum(w[i]*F(B[i][j]) for i in range(18))==0 for j in range(15))
    dot=sum(w[i]*a[i] for i in range(18));assert dot!=0
    assert count==r['graph_evaluations']<=866 and r['radial_evaluations']==r['mechanical_solves']==0
    return dict(exact_SPD_identity=True,global_work_strain_memory_identity=True,all_interval_bounds_verified=True,old_cover_and_geometry_receipt_bound=True,
        full_stiffness_sum_verified=True,cauchy_lower_verified=True,symmetric_response_formulas_verified=True,
        correlation_preserved_in_centered_differences=True,exact_basis_nonmembership=True,
        left_null_witness=list(map(str,w)),witness_dot_probe_m=str(dot),
        diagnostic_full18_not_released_motion=True,graph_evaluations=count,seconds=time.monotonic()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    r=json.loads(args.input.read_bytes());out=audit(r);out['input_sha256']=hashlib.sha256(args.input.read_bytes()).hexdigest()
    with args.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('left_null_witness','witness_dot_probe_m')}),flush=True)
