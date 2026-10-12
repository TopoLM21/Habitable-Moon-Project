"""New-case membership and independent work/stiffness cover audit."""
from fractions import Fraction as F
from pathlib import Path
import argparse,hashlib,json,time,sys
sys.set_int_max_str_digits(30000)
import numpy as np
from analysis.genesis_regional_moment_validation.regional_audit import audit as audit_work
from analysis.genesis_regional_work_validation.audit import evaluate
from analysis.genesis_regional_work_validation.identity import expand
from analysis.genesis_directional_stiffness_validation.audit import squared_form


def audit(r):
    start=time.monotonic();loaded={}
    for key,pin in r['input_pins'].items():
        p=Path(pin['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256'];loaded[key]=json.loads(p.read_bytes())
    source=loaded['fixture']['cases'][0];s=source['supports']['fixed_100m_clipped_support'];B=s['original_declared_added_basis']
    expected_raw={**source['source_inputs'],'seed_center_world_m':s['seed_center_world_m'],'seed_radius_m':s['seed_radius_m']}
    assert r['source']==expected_raw
    p=r['probe'];candidate=loaded['selection']['candidate'];a=list(map(F,p['enrichment_increment_m']));coords=[F(1,8192)]+[F()]*14
    assert p['enrichment_increment_m']==candidate['enrichment_increment_m']
    assert all(a[i]==sum(F(B[i][j])*coords[j] for j in range(15)) for i in range(18))
    assert p['parent_increment_m']==[0.]*18 and p['name']=='W15_mode_1_mixed_jump'
    assert p['mean_e0']==r['source']['accepted_interval_mean_e0'] and p['mean_e1']==r['source']['accepted_interval_mean_e1']
    section=r['source']['section'];n=np.asarray(section['normal_xyz']);X=np.asarray(section['triangle_m']);A=np.asarray(section['displacement_A_m']);P=np.asarray(section['slope_P_m'])
    za=list(map(F,(X+A)@n-section['offset_m']));zp=list(map(F,(X+P)@n))
    norm2=sum(x*x for x in a);leak2=F()
    for j in range(3):
        dot=sum(za[i]*a[6*i+j]+zp[i]*a[6*i+3+j] for i in range(3))
        length2=sum(x*x for x in za+zp);assert length2>0
        leak2+=dot*dot/length2/norm2
    assert leak2<=F(256*np.finfo(float).eps)**2
    graph=r['graph'];wg=r['work_graph'];assert graph['nodes']==wg['nodes']
    assert wg['outputs']=={key:graph['outputs'][key] for key in ('sphere','gamma','density_without_H')}
    newsource=dict(source=r['source'],coverage=r['coverage'],graph=wg,probe=p,root=r['work']['root'])
    assert r['coverage']==loaded['tree']['coverage'] and r['work']['root']==loaded['tree']['root']
    assert r['work']['splits']==loaded['tree']['splits'][:len(r['work']['splits'])]
    # Uses inverse root conversion and independent closed-form child restrictions.
    workaudit=audit_work(r['work'],newsource)
    L=[list(map(F,row)) for row in r['L']];D=list(map(F,r['D']));C=[list(map(F,row)) for row in r['source']['elasticity_pa']]
    assert all(x>0 for x in D)
    assert C==[[sum(L[i][k]*D[k]*L[j][k] for k in range(6)) for j in range(6)] for i in range(6)]
    xr=[list(map(F,row)) for row in section['triangle_m']]
    det=(xr[0][0]*xr[1][1]*xr[2][2]+xr[0][1]*xr[1][2]*xr[2][0]+xr[0][2]*xr[1][0]*xr[2][1]
         -xr[0][2]*xr[1][1]*xr[2][0]-xr[0][1]*xr[1][0]*xr[2][2]-xr[0][0]*xr[1][2]*xr[2][1]);assert det==F(r['determinant'])>0
    polys,_=expand(graph);cross={}
    for i in range(6):
        for j in range(6):
            if not C[i][j]:continue
            for powers,x in polys['strain_'+str(i)].items():
                for other,y in polys['memory_'+str(j)].items():
                    k=tuple(a+b for a,b in zip(powers,other));cross[k]=cross.get(k,F())+det*C[i][j]*x*y
    assert {k:v for k,v in cross.items() if v}==polys['density_without_H']
    rows=r['work']['records'];evaluations=0
    for row in rows:
        box=[tuple(map(F,b)) for b in row['box']];values=evaluate(graph['nodes'],box);evaluations+=1
        ranges={k:values[v] for k,v in graph['outputs'].items()};volume=(box[0][1]-box[0][0])*(box[1][1]-box[1][0])*(box[2][1]-box[2][0])
        for key,prefix,factor in [('stiffness_a_j','strain',F(1,4)),('memory_norm_j','memory',F(1))]:
            if row['state']=='outside':bound=(F(),F())
            else:
                lo,hi=squared_form([ranges[prefix+'_'+str(i)] for i in range(6)],L,D)
                if row['state'] not in ('inside_gamma_fixed','gamma_uncertain_inside'):lo=F()
                bound=(det*volume*factor*lo,det*volume*factor*hi)
            assert bound==tuple(map(F,row[key]))
    leaves=r['work']['leaves'];K=[sum(F(rows[i]['stiffness_a_j'][e]) for i in leaves) for e in (0,1)];E=[sum(F(rows[i]['memory_norm_j'][e]) for i in leaves) for e in (0,1)]
    assert K==list(map(F,r['stiffness_raw_a_j'])) and E==list(map(F,r['memory_norm_j']))
    active={0}
    snapshots={t['processed_blocks']:t for t in r['work']['trace']}
    def trace_check(count):
        if count in snapshots:assert [sum(F(rows[i]['stiffness_a_j'][e]) for i in active) for e in (0,1)]==list(map(F,snapshots[count]['stiffness_a_j']))
    trace_check(1);count=1
    for split in r['work']['splits']:
        active.remove(split['parent']);active.update(split['children']);count+=2;trace_check(count)
    W=list(map(F,r['work']['bounds']));gap=W[0] if W[0]>0 else -W[1] if W[1]<0 else F()
    cauchy=gap**2/E[1] if E[1] else F();assert cauchy==F(r['cauchy_lower_j'])
    K[0]=max(K[0],cauchy);assert K==list(map(F,r['stiffness_a_j'])) and 0<=K[0]<=K[1]
    assert norm2==F(r['ell_squared_m2']) and [x/norm2 for x in K]==list(map(F,r['stiffness_v_n_m']))
    assert evaluations==r['graph_evaluations']==r['work']['processed_blocks']<=1999
    assert not r['old_observable_values_reused'] and r['radial_evaluations']==r['mechanical_solves']==0
    return dict(new_case_exact_W15_membership=True,exact_squared_analytic_kernel_check=True,analytic_kernel_leakage_squared=str(leak2),
        work=workaudit,positive_material_identity=True,global_work_strain_memory_identity=True,all_stiffness_bounds_verified=True,
        stiffness_trace_and_final_sum_verified=True,new_source_bound_no_old_work_transfer=True,
        stiffness_sign='positive' if K[0]>0 else 'zero' if K[1]==0 else 'unresolved',
        graph_evaluations=evaluations,seconds=time.monotonic()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    r=json.loads(args.input.read_bytes());out=audit(r);out['input_sha256']=hashlib.sha256(args.input.read_bytes()).hexdigest()
    with args.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(dict(work_sign=out['work']['sign'],stiffness_sign=out['stiffness_sign'],blocks=out['graph_evaluations'],seconds=out['seconds'],membership=True,kernel=True)),flush=True)
