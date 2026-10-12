"""Read-only saved trial: face ports, damage release, capacities and flux scales."""
import json,hashlib,math
from pathlib import Path
import numpy as np
from tectonics.genesis_energy_ports import boundary_budget
from analysis.genesis_volume_continuation_validation import independent_audit as q
from analysis.genesis_volume_fault_validation import independent_audit as fault
from analysis.genesis_be_candidate.audit import close
from analysis.genesis_connected_step_validation.run import PARENT,PARENT_SHA


def main():
    raw=PARENT.read_bytes();assert hashlib.sha256(raw).hexdigest()==PARENT_SHA;r=json.loads(raw)
    raw=Path('/workspace/genesis-cutcell-mechanical-02/candidate.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=='5a0cfcd3edfd57f4f05b59e8b408b09c25ac2a704068609e22029ab865230f2e';p=json.loads(raw)
    raw=Path('/workspace/genesis-implicit-cutcell-01/after.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==p['thermal_sha256'];th=json.loads(raw)
    s=p['state'];g=s['geometry'];L=s['interval_ledger'];before=r['runtime_state'];shell=r['source_packet']['source']['parameters']['shell'];tp=r['source_packet']['source']['parameters']['thermal']
    X=np.array(g['vertices_m']);faces=np.array(g['faces']);mass=np.array(g['column_mass_kg']);rho=g['density_kg_m3'];v=np.array(s['velocity_m_s']);du=np.array(s['displacement_m'])-before['displacement_m'];dt=(s['thermal']['time_myr']-before['thermal']['time_myr'])*365.25*86400*1e6
    # Independent quadrature's basal assembly supports restricting material by face.
    traction=q.prescribed_traction(X,faces,mass,rho,g['cohorts'],r['source_packet']['current_geometry']['radius_km'],shell)
    D,F=q.basal_drag_and_force(X,faces,mass,rho,g['cohorts'],p['inputs']['basal_drag_pa_s_m'],traction[faces])
    W=float(F@du);Q=float(dt*v@D@v)
    close(W,L['external_work_j'],'external port',rtol=3e-10,atol=1e3);close(Q,L['physical_drag_j'],'interface port',rtol=3e-10,atol=1e3)
    # Independently integrate each face without reconstructing a mantle volume.
    from tectonics.genesis_volume_basal import local_basal_drag_and_force
    face_rows=[];old={x['id']:x for x in before['mechanics']['state']['cohorts']};damage=[]
    for f,face in enumerate(faces):
        cohorts=[c for c in g['cohorts'] if c['face_index']==f]
        idx=(6*face[:,None]+np.arange(6)).ravel();scale=min(c['inner_scale'] for c in cohorts)
        df,ff,area=local_basal_drag_and_force(X[face],scale,traction[face],p['inputs']['basal_drag_pa_s_m'])
        qf=float(dt*v[idx]@df@v[idx]);wf=float(ff@du[idx]);capacity=sum(c['mass_kg'] for c in cohorts)*tp['silicate_heat_capacity_j_kg_k']
        face_rows.append(dict(face=f,area_m2=area,drag_j=qf,traction_work_j=wf,drag_flux_w_m2=qf/dt/area,retained_solid_capacity_j_k=capacity,all_drag_in_face_solid_homogenized_delta_k=qf/capacity))
        for c in cohorts:
            e=old[c['id']];C0=q.isotropic_matrix(shell['young_modulus_pa'],shell['poisson_ratio'],before['fault_history']['damage'][f],shell['residual_stiffness']);C1=q.isotropic_matrix(shell['young_modulus_pa'],shell['poisson_ratio'],s['fault_history']['damage'][f],shell['residual_stiffness'])
            release=-q.coefficient_energy(X[face],mass[f],rho,c,e['strain0'],e['strain1'],C1-C0)
            damage.append(dict(id=c['id'],face=f,release_j=release,all_release_as_cohort_heat_delta_k=release/(c['mass_kg']*tp['silicate_heat_capacity_j_kg_k'])))
    close(math.fsum(x['drag_j'] for x in face_rows),Q,'face drag sum',rtol=3e-10,atol=1e3)
    close(math.fsum(x['traction_work_j'] for x in face_rows),W,'face work sum',rtol=3e-10,atol=1e3)
    release=math.fsum(x['release_j'] for x in damage);close(release,-L['damage_energy_change_j'],'damage release',rtol=3e-8,atol=1e3)
    assert min(x['release_j'] for x in damage)>=0
    delta=float(np.sum(np.array(p['thermal']['energy_j'],dtype=np.longdouble)-np.array(th['energy_j'],dtype=np.longdouble)))
    ports=boundary_budget(L,delta);expanded=boundary_budget(L,delta,interface_in_control_volume=True)
    allowance=json.loads(Path('/workspace/genesis-cutcell-mechanical-02/completion.json').read_text())['feedback_audit']['combined_allowance_j']
    assert abs(ports['balance_residual_j'])<=allowance and abs(expanded['balance_residual_j'])<=allowance
    water_received=-math.fsum(e[2] for e in th['paired_exchanges'] if e[0]==3361)
    deep_received=math.fsum(e[2] for e in th['paired_exchanges'] if e[1]==3360)
    Csolid=sum(x['retained_solid_capacity_j_k'] for x in face_rows);Ccolumns=sum(th['mass_kg'][:3360])*tp['silicate_heat_capacity_j_kg_k']
    terms=dict(traction=W,drag=Q,damage_release=release,maxwell=L['physical_maxwell_j'],thermal_work=L['thermal_work_j'],numerical=L['numerical_kinetic_j']+L['numerical_memory_j'])
    ratios={k:dict(energy_j=x,mean_power_w=x/dt,over_maxwell=x/L['physical_maxwell_j'],over_initial_elastic=x/L['elastic_before_j'],over_top_conduction= x/abs(water_received),homogenized_retained_solid_delta_k=x/Csolid) for k,x in terms.items()}
    result=dict(status='PASS',new_evolution_steps=0,base_sha='7e67a44d85ef834b31517142fcbe9825c69321d9',open_budget=ports,expanded_interface_budget=expanded,balance_allowance_j=allowance,dt_seconds=dt,scales=dict(retained_solid_capacity_j_k=Csolid,column_sensible_capacity_j_k=Ccolumns,top_conduction_to_water_j=water_received,bottom_conduction_to_deep_j=deep_received,top_conduction_mean_power_w=water_received/dt,basal_area_m2=sum(x['area_m2'] for x in face_rows)),terms=ratios,face_ports=face_rows,damage_release_by_cohort=damage,healing_cohorts=0,max_face_homogenized_drag_delta_k=max(x['all_drag_in_face_solid_homogenized_delta_k'] for x in face_rows),max_cohort_damage_heat_scenario_delta_k=max(x['all_release_as_cohort_heat_delta_k'] for x in damage),thermal_feedback_applied=False)
    result['full_split_interval']=full_split_budget(p,th,L)
    out=Path('/workspace/genesis-open-boundary-budget-01.json')
    with out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('face_ports','damage_release_by_cohort')},indent=2))


def full_split_budget(p,th,L):
    # Same literal mass*h endpoint construction as the saved conduction producer.
    cut_raw=Path('/workspace/genesis-cutcell-feasibility-02/candidate.json').read_bytes()
    assert hashlib.sha256(cut_raw).hexdigest()=='6006521e7b3e22ce23894e63248d0145ee15d3065333c949701c730c6cab47ad'
    cut=json.loads(cut_raw)['cells']
    projection_raw=Path('/workspace/genesis-coupled-thermal-projection-03/projected.json').read_bytes()
    assert hashlib.sha256(projection_raw).hexdigest()=='d25fe1fd4e8b56b18ab8ec697b197c326d723e9421148e5d3726fb7affdd45a4'
    cells={c['identity']:c for c in json.loads(projection_raw)['state']['cells']}
    before=np.array([x['mass_kg']*x['specific_h'] for x in cut]+[(cells[k]['energy_j']/cells[k]['mass_kg'])*cells[k]['mass_kg'] for k in ('deep_mantle','surface_water')])
    conducted=np.array(th['energy_j']);end=np.array(p['thermal']['energy_j'])
    conduction_delta=float(np.sum(conducted.astype(np.longdouble)-before.astype(np.longdouble)))
    conduction_allowance=float(np.sum(8*(np.spacing(before)+np.spacing(conducted))))
    assert abs(conduction_delta)<=conduction_allowance
    thermal_delta=float(np.sum(end.astype(np.longdouble)-before.astype(np.longdouble)))
    combined=boundary_budget(L,thermal_delta)
    feedback_allowance=json.loads(Path('/workspace/genesis-cutcell-mechanical-02/completion.json').read_text())['feedback_audit']['combined_allowance_j']
    allowance=conduction_allowance+feedback_allowance
    assert abs(combined['balance_residual_j'])<=allowance
    return dict(status='PASS',conduction_global_residual_j=conduction_delta,conduction_endpoint_roundoff_allowance_j=conduction_allowance,
                complete_thermal_delta_j=thermal_delta,open_budget=combined,combined_stage_allowance_j=allowance,
                note='Whole existing split interval; its larger endpoint-rounding bound does not replace local feedback or mechanical checks.')


if __name__=='__main__':main()
