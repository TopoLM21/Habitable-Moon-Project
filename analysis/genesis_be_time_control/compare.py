"""Read-only common-time/common-material comparison of BE100/50/25."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from analysis.genesis_connected_step_validation.run import PARENT,PARENT_SHA
from analysis.genesis_volume_continuation_validation.independent_audit import mass_matrix
from analysis.genesis_kinetic_diagnostic.run import fronts,bands,compare_velocities
from analysis.genesis_intact_time_sensitivity.metrics import difference


def compare(directory):
    report=json.loads((directory/'report.json').read_bytes())
    if report['status']!='PASS':raise ValueError('Incomplete series')
    for name,pin in report['artifacts'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=pin['sha256']:raise ValueError('Changed series artifact '+name)
    raw=PARENT.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PARENT_SHA:raise ValueError('Canonical root changed')
    root=json.loads(raw)['runtime_state'];g0=root['geometry'];old={c['id']:c for c in g0['cohorts']}
    path=Path('/workspace/genesis-be-candidate-01/candidate.json');raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='b919855cd5a27fcec4f49eedaadab4a490775b801b10c54a667f6d4d14c6b892':raise ValueError('BE100 changed')
    base_report=Path('/workspace/genesis-be-candidate-01/final_report.json')
    baseline_summary=json.loads(Path('analysis/genesis_be_candidate/summary.json').read_bytes())
    if hashlib.sha256(base_report.read_bytes()).hexdigest()!=baseline_summary['report_sha256']:raise ValueError('BE100 cost report changed')
    base_cost=json.loads(base_report.read_bytes())
    packets={'100':json.loads(raw),'50':json.loads((directory/'50/step_2.json').read_bytes()),'25':json.loads((directory/'25/step_4.json').read_bytes())}
    M0=mass_matrix(g0['vertices_m'],g0['faces'],g0['column_mass_kg'],g0['density_kg_m3'],g0['cohorts']);oldmass=sum(c['mass_kg'] for c in old.values());q0=np.asarray(root['displacement_m']);f0=fronts(g0)
    out={};state={}
    for name,p in packets.items():
        s=p['state'];g=s['geometry'];rows={c['id']:c for c in g['cohorts']}
        if abs(s['thermal']['time_myr']-.8685)>1e-14:raise ValueError('Unequal physical endpoint')
        for field in ['vertices_m','faces','column_mass_kg','density_kg_m3']:
            if g[field]!=g0[field]:raise ValueError('Changed reference chart')
        for i,c in old.items():
            if i not in rows or any(rows[i][k]!=c[k] for k in ['q_lo','live_q_hi','mass_kg','face_index','owner_id','material_id']):raise ValueError('Old support changed')
        dt=(s['thermal']['time_myr']-root['thermal']['time_myr'])*365.25*86400*1e6
        u=np.asarray(s['displacement_m']);v=np.asarray(s['velocity_m_s']);mean=(u-q0)/dt;L=s['be_accumulated_ledger'];num=L['numerical_kinetic_j']+L['numerical_memory_j'];physical=L['physical_drag_j']+L['physical_maxwell_j']
        memory={c['id']:c for c in s['mechanics']['cohorts']};state[name]=dict(u=u,v=v,mean=mean,front=fronts(g),memory=memory,damage=s['fault_history']['damage'],stress=p['report']['final_face_tensile_pa'])
        out[name]=dict(cohorts=len(rows),ledger=L,numerical_total_j=num,numerical_to_abs_external_work=num/abs(L['external_work_j']),
            numerical_to_physical_dissipation=num/physical,max_damage=max(s['fault_history']['damage']),
            old_endpoint_rms_m_s=float(np.sqrt(v@M0@v/oldmass)),old_full100_mean_rms_m_s=float(np.sqrt(mean@M0@mean/oldmass)),
            mechanical_solve_seconds=base_cost['stages']['single_BE_interval'] if name=='100' else sum(report['stages'][name+'_step_'+str(i)] for i in range(1,int(100/int(name))+1)),
            control_seconds=base_cost['seconds'] if name=='100' else report['branches'][name]['seconds'])
    comparisons={}
    for a,b in [('100','50'),('50','25'),('100','25')]:
        x,y=state[a],state[b];common=bands(f0,np.minimum(x['front'],y['front']))
        Mn=mass_matrix(g0['vertices_m'],g0['faces'],g0['column_mass_kg'],g0['density_kg_m3'],common)
        newmass=float(np.dot(g0['column_mass_kg'],np.minimum(x['front'],y['front'])-f0))
        comparisons[a+'_vs_'+b]=dict(top_A_m=difference(x['u'].reshape(-1,6)[:,:3],y['u'].reshape(-1,6)[:,:3]),
            P_m=difference(x['u'].reshape(-1,6)[:,3:],y['u'].reshape(-1,6)[:,3:]),face_tensile_pa=difference(x['stress'],y['stress']),damage=difference(x['damage'],y['damage']),
            old_endpoint_velocity=compare_velocities(M0,oldmass,x['v'],y['v']),old_full100_mean_velocity=compare_velocities(M0,oldmass,x['mean'],y['mean']),
            common_newborn_endpoint_velocity=compare_velocities(Mn,newmass,x['v'],y['v']),
            common_newborn_full100_mean_velocity=compare_velocities(Mn,newmass,x['mean'],y['mean']),
            unmatched_material_kg=float(np.dot(g0['column_mass_kg'],np.abs(x['front']-y['front']))),
            old_memory={k:difference([x['memory'][i][k] for i in sorted(old)],[y['memory'][i][k] for i in sorted(old)]) for k in ['strain0','strain1','temperature_k']},
            numerical_dissipation_ratio=out[a]['numerical_total_j']/out[b]['numerical_total_j'])
    # Intermediate common time: after 50 years, not after unequal substep counts.
    mid50=json.loads((directory/'50/step_1.json').read_bytes())['state'];mid25=json.loads((directory/'25/step_2.json').read_bytes())['state']
    midway=dict(time_50=mid50['thermal']['time_myr'],time_25=mid25['thermal']['time_myr'],
        old_endpoint_velocity=compare_velocities(M0,oldmass,np.asarray(mid50['velocity_m_s']),np.asarray(mid25['velocity_m_s'])))
    return dict(status='PASS_BOUNDED_BE_TIME_MATERIAL_CONTROL',base_sha='9d237ab22fdd1b2f5aeb7525dd599c2024e6f11c',branches=out,comparisons=comparisons,midway=midway,
        new_substeps=report['new_substeps'],seconds=report['seconds'],peak_rss_kib=report['peak_rss_kib'],
        cumulative_audits={k:v['cumulative'] for k,v in report['branches'].items()},
        restart_byte_identical={k:v['restart_byte_identical'] for k,v in report['branches'].items()},
        max_independent_impulse=max(s['independent_final']['coordinate_momentum_relative'] for b in report['branches'].values() for s in b['steps']),
        max_independent_balance=max(s['independent_final']['balance_relative'] for b in report['branches'].values() for s in b['steps']),
        canonical_parent_unchanged=True,original_800_supports_unchanged=True,physical_error_bound_claimed=False,
        total_output_bytes=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()),
        report_sha256=hashlib.sha256((directory/'report.json').read_bytes()).hexdigest(),directory=str(directory))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=compare(a.input)
    with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:r[k] for k in ['status','new_substeps','seconds','branches','comparisons']}))
