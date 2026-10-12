"""Read-only BE vs existing midpoint at the same material endpoint."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from analysis.genesis_volume_continuation_validation.independent_audit import mass_matrix
from analysis.genesis_intact_time_sensitivity.metrics import difference
from analysis.genesis_kinetic_diagnostic.run import compare_velocities
from analysis.genesis_connected_step_validation.run import PARENT,PARENT_SHA


def compare(directory):
    report_path=next(directory/name for name in ['final_report.json','verified_report.json','report.json'] if (directory/name).exists())
    report=json.loads(report_path.read_bytes())
    if report['status']!='PASS':raise ValueError('Adapter run not PASS')
    for name,pin in report['artifacts'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=pin['sha256']:raise ValueError('Run artifacts changed')
    raw=PARENT.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PARENT_SHA:raise ValueError('Parent changed')
    parent=json.loads(raw)['runtime_state'];candidate=json.loads((directory/'candidate.json').read_bytes());be=candidate['state']
    baseline=Path('/workspace/genesis-connected-step-08685-01/candidate.json');raw=baseline.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='4fa1eb80eed9d19ddb44a08c747dfe3fe267a84bd953380f649d4d3834c60e98':raise ValueError('Midpoint changed')
    mid=json.loads(raw)['runtime_state'];g=be['geometry']
    if g!=mid['geometry'] or be['thermal']!=mid['thermal']:raise ValueError('Different material or thermal endpoint')
    M=mass_matrix(g['vertices_m'],g['faces'],g['column_mass_kg'],g['density_kg_m3'],g['cohorts']);mass=sum(c['mass_kg'] for c in g['cohorts'])
    dt=(be['thermal']['time_myr']-parent['thermal']['time_myr'])*365.25*86400*1e6
    u0=np.asarray(parent['displacement_m']);uB=np.asarray(be['displacement_m']);uM=np.asarray(mid['displacement_m'])
    vB=np.asarray(be['velocity_m_s']);vM=np.asarray(mid['velocity_m_s']);meanB=(uB-u0)/dt;meanM=(uM-u0)/dt
    LM=mid['last_report'];LB=be['interval_ledger'];comparisons={}
    pairs={'physical_drag_j':'basal_dissipation_j','physical_maxwell_j':'viscous_dissipation_j','internal_work_j':'mechanical_work_j',
        'external_work_j':'external_work_j','thermal_work_j':'thermal_work_j','birth_kinetic_inflow_j':'birth_kinetic_energy_j'}
    for k,m in pairs.items():comparisons[k]=dict(midpoint=LM[m],backward_euler=LB[k],**difference(LM[m],LB[k]))
    oldids={c['id'] for c in parent['geometry']['cohorts']};memM={c['id']:c for c in mid['mechanics']['state']['cohorts']};memB={c['id']:c for c in be['mechanics']['cohorts']}
    memory={k:difference([memM[i][k] for i in sorted(oldids)],[memB[i][k] for i in sorted(oldids)]) for k in ['strain0','strain1','temperature_k']}
    numerical=LB['numerical_kinetic_j']+LB['numerical_memory_j'];physical=LB['physical_drag_j']+LB['physical_maxwell_j']
    return dict(status='PASS_SINGLE_BE_CANDIDATE_COMPARISON',base_sha='1475fee46622d7216ac179ada0870cddbc9112f9',
        time_myr=be['thermal']['time_myr'],cohorts=len(g['cohorts']),seconds=report['seconds'],peak_rss_kib=report['peak_rss_kib'],stages=report['stages'],
        geometry_and_thermal_identical_to_midpoint=True,canonical_parent_unchanged=True,restart_byte_identical=report['restart_byte_identical'],
        endpoint_velocity=compare_velocities(M,mass,vM,vB),mean_velocity=compare_velocities(M,mass,meanM,meanB),
        BE_endpoint_vs_mean_rms_m_s=float(np.sqrt((vB-meanB)@M@(vB-meanB)/mass)),
        midpoint_endpoint_vs_mean_rms_m_s=float(np.sqrt((vM-meanM)@M@(vM-meanM)/mass)),
        top_displacement_m=difference(uM.reshape(-1,6)[:,:3],uB.reshape(-1,6)[:,:3]),
        top_displacement_max_vector_difference_m=float(np.max(np.linalg.norm((uB-uM).reshape(-1,6)[:,:3],axis=1))),
        director_P_m=difference(uM.reshape(-1,6)[:,3:],uB.reshape(-1,6)[:,3:]),
        face_tensile_pa=difference(LM['fault_evolution']['final_face_tensile_pa'],candidate['report']['final_face_tensile_pa']),
        damage=difference(mid['fault_history']['damage'],be['fault_history']['damage']),
        max_damage=dict(midpoint=max(mid['fault_history']['damage']),BE=max(be['fault_history']['damage'])),
        old_memory=memory,work=comparisons,BE_ledger=LB,
        numerical_total_j=numerical,numerical_to_physical_dissipation=numerical/physical,numerical_to_final_elastic=numerical/LB['elastic_after_j'],
        independent_final=report['independent_final'],after_gate=report['after_gate'],
        no_full_time_accuracy_claim=True,new_Genesis_intervals=report['new_physical_intervals'],
        directory=str(directory),report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        total_artifact_bytes=sum(p.stat().st_size for p in directory.iterdir() if p.is_file()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=compare(a.input)
    with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:r[k] for k in ['status','seconds','endpoint_velocity','mean_velocity','top_displacement_m','face_tensile_pa','damage','numerical_total_j','numerical_to_physical_dissipation']}))
