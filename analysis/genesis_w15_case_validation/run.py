"""One shared bounded regional tree for NEW W15 work and stiffness."""
from fractions import Fraction as F
from pathlib import Path
import argparse,hashlib,json,time,sys,shutil
sys.set_int_max_str_digits(30000)
from .preconditions import verify,distance_to_zero
from analysis.genesis_directional_stiffness_validation.expression import build
from analysis.genesis_directional_stiffness_validation.bounds import ldl,block
from analysis.genesis_regional_work_validation.expression import interval_evaluate
from analysis.genesis_regional_work_validation.run import point_checks
from analysis.genesis_regional_work_validation.identity import expand
from analysis.genesis_regional_moment_validation.moments import enclose,enclose_certificates
from analysis.genesis_regional_moment_validation.restrict import halve
from analysis.genesis_regional_moment_validation.regional import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    if shutil.disk_usage('/workspace').free<128*1024**2:raise ValueError('128 MiB reserve required')
    paths={k:Path(v) for k,v in dict(tree='/workspace/genesis-regional-work-centered-01.json',
        selection='analysis/genesis_w15_probe_selection/preflight.json',
        fixture='analysis/genesis_volume_exact_lambda_validation/evidence/actual_control/source_fixture.json').items()}
    old=json.loads(paths['tree'].read_bytes());selection=json.loads(paths['selection'].read_bytes());fixture=json.loads(paths['fixture'].read_bytes())
    assert old['processed_blocks']==1999 and old['maximum_depth']==11
    assert hashlib.sha256(paths['fixture'].read_bytes()).hexdigest()==selection['source_sha256']
    raw=old['source'];candidate=selection['candidate'];B=fixture['cases'][0]['supports']['fixed_100m_clipped_support']['original_declared_added_basis']
    preconditions=verify(candidate,B,raw)
    probe={k:candidate[k] for k in ('enrichment_increment_m','parent_increment_m','mean_e0','mean_e1')}
    probe['name']='W15_mode_1_mixed_jump';assert probe['mean_e0']==old['probe']['mean_e0'] and probe['mean_e1']==old['probe']['mean_e1']
    start=time.monotonic();root=tuple(tuple(map(F,b)) for b in old['root']);graph=build(raw,probe,tuple(sum(b)/2 for b in root))
    checks=point_checks(raw,probe,graph,root)
    workgraph={**graph,'outputs':{k:graph['outputs'][k] for k in ('sphere','gamma','density_without_H')}}
    polys,_=expand(workgraph);L,D=ldl(raw['elasticity_pa']);X=[list(map(F,r)) for r in raw['section']['triangle_m']]
    det=sum(X[0][j]*(X[1][(j+1)%3]*X[2][(j+2)%3]-X[1][(j+2)%3]*X[2][(j+1)%3]) for j in range(3));assert det>0
    full=enclose(polys,root);root_cert=full['polynomials'];certificates={0:root_cert};active={0};rows=[];splits=[];trace=[]
    total=[F(),F()];stotal=[F(),F()];mtotal=[F(),F()]
    def record(i,full):
        assert i==len(rows)
        coeff=full.pop('polynomials');box=tuple(tuple(map(F,b)) for b in full['box'])
        ranges=interval_evaluate(graph,box);quadratic=block(ranges,box,full['state'],L,D,det)
        row=dict(id=i,**full,coefficient_sha256=digest(coeff),**quadratic);rows.append(row)
        for e in (0,1):total[e]+=F(row['bounds'][e]);stotal[e]+=F(row['stiffness_a_j'][e]);mtotal[e]+=F(row['memory_norm_j'][e])
    record(0,full)
    def snapshot():
        trace.append(dict(processed_blocks=len(rows),bounds=list(map(str,total)),stiffness_a_j=list(map(str,stotal)),seconds=time.monotonic()-start))
        print(json.dumps(dict(blocks=len(rows),work_J=[float(x) for x in total],stiffness_a_J=[float(x) for x in stotal],seconds=trace[-1]['seconds'])),flush=True)
    snapshot();reason='fixed_partition_complete'
    for split in old['splits']:
        if time.monotonic()-start>=120:reason='time_cap';break
        if len(rows)+2>1999:reason='block_cap';break
        parent=split['parent'];axis=split['axis'];assert parent in active
        cert=certificates.pop(parent);pairs={k:halve(v,axis) for k,v in cert.items()};active.remove(parent)
        for e in (0,1):total[e]-=F(rows[parent]['bounds'][e]);stotal[e]-=F(rows[parent]['stiffness_a_j'][e]);mtotal[e]-=F(rows[parent]['memory_norm_j'][e])
        for side,i in enumerate(split['children']):
            cc={k:p[side] for k,p in pairs.items()};certificates[i]=cc;active.add(i)
            box=tuple(tuple(map(F,b)) for b in old['records'][i]['box']);record(i,enclose_certificates(cc,box))
        splits.append(split)
        if len(rows) in (31,127,511,999,1499,1999):snapshot()
    if trace[-1]['processed_blocks']!=len(rows):snapshot()
    workreport=dict(root_certificates=root_cert,root=old['root'],records=rows,splits=splits,leaves=sorted(active),bounds=list(map(str,total)),trace=trace,
        sign='positive' if total[0]>0 else 'negative' if total[1]<0 else 'unresolved',reason=reason,processed_blocks=len(rows),max_blocks=1999,
        max_compute_seconds=120,maximum_depth=max(old['records'][i]['depth'] for i in range(len(rows))),root_polynomial_transforms=3,
        half_restrictions=3*len(splits),radial_evaluations=0,mechanical_solves=0,new_partition_splits=0,physical_admission=False)
    ell2=sum(F(v)**2 for v in probe['enrichment_increment_m']);gap=distance_to_zero(total)
    cauchy=gap**2/mtotal[1] if mtotal[1] else F();K=[max(stotal[0],cauchy),stotal[1]];assert K[0]<=K[1]
    result=dict(case_name='W15-мода №1, смешанный скачок',case_id=probe['name'],preconditions=preconditions,source=raw,probe=probe,
        coverage=old['coverage'],graph=graph,work_graph=workgraph,work=workreport,
        L=[[str(x) for x in row] for row in L],D=list(map(str,D)),determinant=str(det),
        stiffness_raw_a_j=list(map(str,stotal)),memory_norm_j=list(map(str,mtotal)),cauchy_lower_j=str(cauchy),
        stiffness_a_j=list(map(str,K)),ell_squared_m2=str(ell2),stiffness_v_n_m=[str(x/ell2) for x in K],
        point_checks=checks,seconds=time.monotonic()-start,graph_evaluations=len(rows),
        radial_evaluations=0,mechanical_solves=0,physical_admission=False,tolerance_changed=False,old_observable_values_reused=False,
        input_pins={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()})
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(work_sign=workreport['sign'],work_J=[float(x) for x in total],stiffness_N_m=[float(x/ell2) for x in K],blocks=len(rows),seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
