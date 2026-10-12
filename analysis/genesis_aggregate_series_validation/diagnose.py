"""Frozen endpoint algebra and one stiffness-assembly timing per representation.

No nodal solve, thermal integration, damage update or new planetary trial.
"""
import json,time
from pathlib import Path
import numpy as np
from analysis.genesis_aggregate_coupled.adapter import Adapter
from analysis.genesis_volume_continuation_validation import independent_audit as q
from analysis.genesis_backward_euler_prototype.core import conical_memory
from tectonics.genesis_volume_operator import conical_radial_moments
from tectonics.genesis_transfer_bridge import digest,encoded
from tectonics.genesis import SECONDS_PER_MYR

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/workspace/genesis-aggregate-series-25-01')


def main():
    started=time.perf_counter();report=json.loads((OUT/'report.json').read_bytes())
    for name,pin in report['input_sha256'].items():assert digest(Path(name).read_bytes())==pin
    p=json.loads(Path('/workspace/genesis-exploratory-100-01/continuation/initial.json').read_bytes())['packet']
    root=json.loads((ROOT/'analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json').read_bytes())
    full=json.loads(Path('/workspace/genesis-forced-coupled-25-01/candidate.json').read_bytes());r=json.loads((OUT/'after.json').read_bytes())
    a=Adapter(p,root['source_packet'],full['thermal_packet']['fixed_cut_geometry'],full['inputs'],report['contact_parameters'],thermal_reduction='series_resistance')
    dt=(r['time_myr']-a.time)*SECONDS_PER_MYR;sp=a.shell;alpha=sp['linear_expansion_per_k'];unit=np.array([1,1,1,0,0,0.])
    ag=np.array(r['memory']).reshape(80,2,6);Tag=np.array(r['temperature_k']);dag=np.array(r['fault_history']['damage'])
    old={x['id']:x for x in p['state']['mechanics']['cohorts']};end={x['id']:x for x in full['state']['mechanics']['cohorts']}
    duf=np.array(full['state']['displacement_m'])-a.u0;dua=np.array(r['displacement_m'])-a.u0
    N,wa=q.angular_rule();interp=N@np.linalg.inv(q.SOURCE_ANGULAR_POINTS)
    parts={k:[] for k in ['kinematics','thermal_mean','thermal_gradient','relaxation_coefficient','unresolved_memory_relaxation','actual_difference','identity_residual']}
    full_project_energy=0.;face_rows=[];full_setup=[];agg_setup=[]
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
        ra=rate(Tag[a.solid_group[f]]);S=1/(1+dt*ra);H=np.kron(b['G'],a.C(f,dag))
        dT=np.array([Tag[a.solid_group[f]],0.])-Tf
        term=dict(kinematics=S*(b['L']@(dua-duf)[b['dofs']]).reshape(2,6),
                  thermal_mean=-S*alpha*np.array([dT[0],0.])[:,None]*unit,
                  thermal_gradient=-S*alpha*np.array([0.,dT[1]])[:,None]*unit,
                  relaxation_coefficient=-S*dt*(ra*np.eye(2)-Rf)@pf,
                  unresolved_memory_relaxation=S*dt*residual_rate)
        actual=ag[f]-pf;remainder=actual-sum(term.values());term.update(actual_difference=actual,identity_residual=remainder)
        for k,x in term.items():parts[k].append(float(x.ravel()@H@x.ravel()))
        full_project_energy+=float(.5*pf.ravel()@H@pf.ravel())
        face_rows.append(dict(face=f,full_temperature_mean_k=float(Tf[0]),full_temperature_slope_k_per_m=float(Tf[1]/b['scale']),aggregate_temperature_k=float(Tag[a.solid_group[f]]),mean_error_k=float(dT[0]),equation_identity_max_abs=float(np.max(np.abs(remainder)))))
        theta=(np.array([Tag[a.solid_group[f]],0.])-b['old_temperature_coeff'])[:,None]*alpha*unit
        agg_setup.append((b,H,ra,theta.ravel(),a.old[f]))
    # Static thermal representability: compare to best mass/enthalpy mean of full endpoint.
    mass=np.array(p['thermal']['mass_kg']);Ef=np.array(full['thermal']['energy_j']);Tf=np.array(full['thermal']['temperature_k']);Eg=np.bincount(a.map,weights=Ef,minlength=a.n);meanT=a.temperature(Eg)
    solid=np.array([i for i,c in enumerate(a.cuts) if c['material_id'] is not None]);within=Tf-meanT[a.map];between=Tag-meanT
    # A single assembly-only benchmark. Frozen stored full mechanical inputs;
    # no solution, time advance or physical state from the assembled systems.
    inp=full['inputs'];rows=list(a.g['cohorts'])
    for i,c in enumerate(rows):
        f=c['face_index'];B0,B1=q.independent_strain_operators(a.X[a.faces[f]],a.frames[f],q.SOURCE_ANGULAR_POINTS)
        mu,var=conical_radial_moments(c['outer_scale'],c['inner_scale'])
        full_setup.append((a.blocks[f]['dofs'],B0,B1,np.array(inp['old0'][i]),np.array(inp['old1'][i]),np.array(inp['thermal0'][i]),np.array(inp['elasticity_pa'][i]),c['volume_m3'],mu,var,inp['maxwell_ratio'][i]))
    start=time.perf_counter();Af=a.M/dt+a.D;rhsf=a.M@a.v0/dt+a.F
    for idx,B0,B1,e0,e1,th,C,V,mu,var,ratio in full_setup:
        mem=conical_memory(B0,B1,e0,e1,th,C,V,mu,var,ratio/dt);factor=1/(1+ratio)
        Af[np.ix_(idx,idx)]+=dt*factor*mem.B.T@mem.H@mem.B
        np.add.at(rhsf,idx,-mem.B.T@mem.H@(factor*(mem.old-mem.thermal_increment)))
    full_seconds=time.perf_counter()-start
    start=time.perf_counter();Aa=a.M/dt+a.D;rhsa=a.M@a.v0/dt+a.F
    for b,H,ra,theta,e in agg_setup:
        S=np.linalg.solve(np.eye(12)+dt*np.eye(12)*ra,np.eye(12));HS=H@S;idx=b['dofs'];L=b['L']
        Aa[np.ix_(idx,idx)]+=dt*L.T@HS@L;np.add.at(rhsa,idx,-L.T@HS@(e-theta))
    agg_seconds=time.perf_counter()-start
    assert np.isfinite(Aa).all() and np.isfinite(Af).all()
    squared={k:sum(v) for k,v in parts.items()};norms={k:float(np.sqrt(v)) for k,v in squared.items()}
    assert norms['identity_residual']<1e-8*norms['actual_difference']
    full_ledger=full['state']['interval_ledger']
    result=dict(status='SAVED_ALGEBRA_IDENTITY_PASS_NO_NEW_EVOLUTION',new_planetary_trials=0,new_nodal_solves=0,
        input_after_sha256=digest((OUT/'after.json').read_bytes()),
        decomposition=dict(metric='sqrt(sum(delta_coeff^T H_at_aggregate_damage delta_coeff)), sqrt(J); term norms do not add because cross terms exist',energy_metric_norms=norms,
            identity_relative=norms['identity_residual']/norms['actual_difference'],full_endpoint_projection_energy_at_aggregate_damage_j=full_project_energy,faces=face_rows),
        thermal=dict(full_profile_within_group_max_difference_k=float(np.max(np.abs(within))),aggregate_vs_full_group_mean_max_k=float(np.max(np.abs(between))),
            within_solid_group_mass_rms_k=float(np.sqrt(np.average(within[solid]**2,weights=mass[solid]))),aggregate_solid_mean_error_mass_rms_k=float(np.sqrt(np.average(between[a.map[solid]]**2,weights=mass[solid])))),
        energy=dict(aggregate_thermal_work_j=r['ledger']['thermal_work'],full_thermal_work_j=full_ledger['thermal_work_j'],aggregate_numerical_memory_j=r['ledger']['numerical_memory'],full_numerical_memory_j=full_ledger['numerical_memory_j']),
        assembly_benchmark=dict(full_seconds=full_seconds,aggregate_seconds=agg_seconds,speed_ratio=full_seconds/agg_seconds,repeats=1,scope='one frozen stiffness/RHS assembly each, prepared operands; no thermal/source/damage/audit work, no nodal solve; different reduced/full representations',full_blocks=800,aggregate_blocks=80),
        seconds=time.perf_counter()-started)
    with (OUT/'localization.json').open('xb') as f:f.write(encoded(result))
    print(json.dumps({k:v for k,v in result.items() if k!='decomposition'},indent=2));print(json.dumps(norms,indent=2))


if __name__=='__main__':main()
