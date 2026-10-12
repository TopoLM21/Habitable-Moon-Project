"""Frozen endpoint coefficient-error identity for the thermal moment trial.

No nodal solve, thermal integration, damage update or new planetary trial.
"""
import json,time
from pathlib import Path
import numpy as np
from analysis.genesis_thermal_moment.adapter import MomentAdapter
from analysis.genesis_volume_continuation_validation import independent_audit as q
from tectonics.genesis_transfer_bridge import digest,encoded
from tectonics.genesis import SECONDS_PER_MYR

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/workspace/genesis-thermal-moment-25-01')


def main():
    started=time.perf_counter();report=json.loads((OUT/'report.json').read_bytes())
    for name,pin in report['input_sha256'].items():assert digest(Path(name).read_bytes())==pin
    p=json.loads(Path('/workspace/genesis-exploratory-100-01/continuation/initial.json').read_bytes())['packet']
    root=json.loads((ROOT/'analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json').read_bytes())
    full=json.loads(Path('/workspace/genesis-forced-coupled-25-01/candidate.json').read_bytes());r=json.loads((OUT/'after.json').read_bytes())
    a=MomentAdapter(p,root['source_packet'],full['thermal_packet']['fixed_cut_geometry'],full['inputs'],report['contact_parameters'])
    dt=(r['time_myr']-a.time)*SECONDS_PER_MYR;sp=a.shell;alpha=sp['linear_expansion_per_k'];unit=np.array([1,1,1,0,0,0.])
    ag=np.array(r['memory']).reshape(80,2,6);Tag=np.array(r['temperature_k']);dag=np.array(r['fault_history']['damage'])
    end={x['id']:x for x in full['state']['mechanics']['cohorts']}
    duf=np.array(full['state']['displacement_m'])-a.u0;dua=np.array(r['displacement_m'])-a.u0
    N,wa=q.angular_rule();interp=N@np.linalg.inv(q.SOURCE_ANGULAR_POINTS)
    parts={k:[] for k in ['kinematics','thermal_mean','thermal_gradient','relaxation_coefficient','unresolved_memory_relaxation','actual_difference','identity_residual']}
    full_project_energy=0.;face_rows=[]
    rate=lambda T:sp['young_modulus_pa']/np.clip(sp['viscosity_reference_pa_s']*np.exp(np.clip(sp['activation_energy_j_mol']/8.314462618*(1/T-1/sp['viscosity_reference_temperature_k']),-60,60)),sp['viscosity_min_pa_s'],sp['viscosity_max_pa_s'])
    for f,b in enumerate(a.blocks):
        rhs=np.zeros((2,6));temp=np.zeros(2);relax=np.zeros((2,2));rate_e=np.zeros((2,6))
        for c in a.g['cohorts']:
            if c['face_index']!=f:continue
            rad,wr=q.radial_rule(c['q_lo'],c['live_q_hi']);lam,_=q.conical_scales(a.X[a.faces[f]],a.g['column_mass_kg'][f],a.g['density_kg_m3'],rad)
            height=float(a.X[a.faces[f,0]]@a.frames[f,:,2]);z=np.broadcast_to(height*(lam-1),(len(wa),len(wr))).ravel()
            w=(c['mass_kg']/a.g['density_kg_m3']*wa[:,None]*wr[None,:]).ravel();phi=np.column_stack((np.ones(len(w)),(z-b['center'])/b['scale']))
            e=end[c['id']];strain=((interp@np.array(e['strain0']))[:,None,:]+(interp@np.array(e['strain1']))[:,None,:]/lam[None,:,None]).reshape(-1,6)
            tau=rate(e['temperature_k']);Gc=phi.T@(w[:,None]*phi);moment=phi.T@(w[:,None]*strain)
            rhs+=moment;temp+=phi.T@(w*e['temperature_k']);relax+=tau*Gc;rate_e+=tau*moment
        pf=np.linalg.solve(b['G'],rhs);Tf=np.linalg.solve(b['G'],temp);Rf=np.linalg.solve(b['G'],relax)
        residual_rate=np.linalg.solve(b['G'],rate_e)-Rf@pf
        P=a.P[f];w=a.weights[f];Ra=np.linalg.solve(b['G'],P.T@((w*rate(Tag[[a.solid_group[f],a.solid_second[f]]]))[:,None]*P));S=np.linalg.inv(np.eye(2)+dt*Ra);H=np.kron(b['G'],a.C(f,dag))
        dT=a.temperature_coefficients(Tag,f)-Tf
        term=dict(kinematics=S@(b['L']@(dua-duf)[b['dofs']]).reshape(2,6),
                  thermal_mean=-alpha*(S@np.array([dT[0],0.])[:,None])*unit,
                  thermal_gradient=-alpha*(S@np.array([0.,dT[1]])[:,None])*unit,
                  relaxation_coefficient=-dt*S@(Ra-Rf)@pf,
                  unresolved_memory_relaxation=dt*S@residual_rate)
        actual=ag[f]-pf;remainder=actual-sum(term.values());term.update(actual_difference=actual,identity_residual=remainder)
        for k,x in term.items():parts[k].append(float(x.ravel()@H@x.ravel()))
        full_project_energy+=float(.5*pf.ravel()@H@pf.ravel())
        face_rows.append(dict(face=f,full_temperature_mean_k=float(Tf[0]),full_temperature_slope_k_per_m=float(Tf[1]/b['scale']),aggregate_temperature_k=float(Tag[a.solid_group[f]]),mean_error_k=float(dT[0]),equation_identity_max_abs=float(np.max(np.abs(remainder)))))
    squared={k:sum(v) for k,v in parts.items()};norms={k:float(np.sqrt(v)) for k,v in squared.items()}
    assert norms['identity_residual']<1e-8*norms['actual_difference']
    full_ledger=full['state']['interval_ledger']
    result=dict(status='SAVED_ALGEBRA_IDENTITY_PASS_NO_NEW_EVOLUTION',new_planetary_trials=0,new_nodal_solves=0,
        input_after_sha256=digest((OUT/'after.json').read_bytes()),
        decomposition=dict(metric='sqrt(sum(delta_coeff^T H_at_aggregate_damage delta_coeff)), sqrt(J); term norms do not add because cross terms exist',energy_metric_norms=norms,
            identity_relative=norms['identity_residual']/norms['actual_difference'],full_endpoint_projection_energy_at_aggregate_damage_j=full_project_energy,faces=face_rows),
        energy=dict(aggregate_thermal_work_j=r['ledger']['thermal_work'],full_thermal_work_j=full_ledger['thermal_work_j'],aggregate_numerical_memory_j=r['ledger']['numerical_memory'],full_numerical_memory_j=full_ledger['numerical_memory_j']),
        seconds=time.perf_counter()-started)
    with (OUT/'localization.json').open('xb') as f:f.write(encoded(result))
    print(json.dumps({k:v for k,v in result.items() if k!='decomposition'},indent=2));print(json.dumps(norms,indent=2))


if __name__=='__main__':main()
