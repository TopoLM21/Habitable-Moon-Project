"""Read-only orthogonal decomposition of saved BE work; no evolution or solve."""
import json,time,resource
from pathlib import Path
import numpy as np
from .adapter import MomentAdapter
from analysis.genesis_volume_continuation_validation import independent_audit as q
from tectonics.genesis_transfer_bridge import digest,encoded
from tectonics.genesis import SECONDS_PER_MYR

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/workspace/genesis-thermal-moment-25-01')

def quadratic_parts(vectors,H):
    names=list(vectors);out={k:float(.5*vectors[k]@H@vectors[k]) for k in names}
    for i,k in enumerate(names):
        for j in names[i+1:]:out[k+' × '+j]=float(vectors[k]@H@vectors[j])
    return out

def main():
    start=time.perf_counter();r=json.loads((OUT/'after.json').read_bytes());report=json.loads((OUT/'report.json').read_bytes())
    pins=dict(report['input_sha256']);pins[str(OUT/'after.json')]='18476901044d7c8844a5809725f3100b8f026fa52005d5e276b27040acbd273d'
    for path,pin in pins.items():assert digest(Path(path).read_bytes())==pin
    p=json.loads(Path('/workspace/genesis-exploratory-100-01/continuation/initial.json').read_bytes())['packet']
    src=json.loads((ROOT/'analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json').read_bytes())['source_packet']
    full=json.loads(Path('/workspace/genesis-forced-coupled-25-01/candidate.json').read_bytes());lf=full['state']['interval_ledger'];la=r['ledger']
    a=MomentAdapter(p,src,full['thermal_packet']['fixed_cut_geometry'],full['inputs'],report['contact_parameters'])
    dt=(r['time_myr']-a.time)*SECONDS_PER_MYR;ag=np.array(r['memory']);T=np.array(r['temperature_k']);da=np.array(r['fault_history']['damage']);df=np.array(full['state']['fault_history']['damage'])
    du=np.array(r['displacement_m'])-a.u0;old={c['id']:c for c in p['state']['mechanics']['cohorts']};end={c['id']:c for c in full['state']['mechanics']['cohorts']}
    N,wa=q.angular_rule();interp=N@np.linalg.inv(q.SOURCE_ANGULAR_POINTS);tot={};faces=[];forcing={};material_rows=[]
    blocks=a.mechanical_blocks(T,da)
    def add(k,x):tot[k]=tot.get(k,0.)+float(x)
    for f,b in enumerate(a.blocks):
        Ca=a.C(f,da);Cf=a.C(f,df);C0=a.C(f,a.history.damage);Ha=np.kron(b['G'],Ca);Hf=np.kron(b['G'],Cf)
        rhs0=np.zeros((2,6));rhs1=rhs0.copy();parts=[]
        for c in a.g['cohorts']:
            if c['face_index']!=f:continue
            rad,wr=q.radial_rule(c['q_lo'],c['live_q_hi']);lam,_=q.conical_scales(a.X[a.faces[f]],a.g['column_mass_kg'][f],a.g['density_kg_m3'],rad)
            z=np.broadcast_to(float(a.X[a.faces[f,0]]@a.frames[f,:,2])*(lam-1),(len(wa),len(wr))).ravel();w=(c['mass_kg']/a.g['density_kg_m3']*wa[:,None]*wr[None,:]).ravel();phi=np.column_stack((np.ones(len(w)),(z-b['center'])/b['scale']))
            def value(e):return ((interp@np.array(e['strain0']))[:,None,:]+(interp@np.array(e['strain1']))[:,None,:]/lam[None,:,None]).reshape(-1,6)
            e0=value(old[c['id']]);e1=value(end[c['id']]);rhs0+=phi.T@(w[:,None]*e0);rhs1+=phi.T@(w[:,None]*e1);parts.append((c['id'],phi,w,e0,e1))
        p0=np.linalg.solve(b['G'],rhs0);p1=np.linalg.solve(b['G'],rhs1);assert np.allclose(p0.ravel(),a.old[f],rtol=1e-12,atol=1e-16)
        dp=(p1-p0).ravel();err=ag[f]-p1.ravel();dag=ag[f]-a.old[f]
        nf=0.;nr=0.;orth=0.
        for cid,phi,w,e0,e1 in parts:
            residual0=e0-phi@p0;residual1=e1-phi@p1;dr=residual1-residual0;de=e1-e0;active=phi@(p1-p0)
            inner=lambda x,C,y:float(np.einsum('n,ni,ij,nj->',w,x,C,y))
            nf+=.5*inner(de,Cf,de);nr+=.5*inner(dr,Cf,dr);orth+=inner(active,Cf,dr)
            ea=phi@ag[f].reshape(2,6);eap0=phi@p0;dea=ea-eap0
            material_rows.append(dict(id=cid,aggregate_n=.5*inner(dea,Ca,dea),aggregate_stored=max(.5*inner(ea,Ca,ea),.5*inner(eap0,Ca,eap0)),full_n=.5*inner(de,Cf,de),full_stored=max(.5*inner(e0,Cf,e0),.5*inner(e1,Cf,e1))))
            ep=phi@p1;oldT=old[cid]['temperature_k'];newT=end[cid]['temperature_k']
            theta=np.tile(a.shell['linear_expansion_per_k']*(newT-oldT)*np.array([1.,1.,1.,0.,0.,0.]),(len(w),1))
            add('full_projected_maxwell',dt*a.rate(newT)*inner(ep,Cf,ep))
            add('full_projected_thermal_work',-inner(ep,Cf,theta))
            add('full_projected_endpoint_energy',.5*inner(ep,Cf,ep))
            add('initial_reserve_direct',.5*inner(residual0,C0,residual0));add('full_endpoint_unresolved_energy',.5*inner(residual1,Cf,residual1))
        npf=.5*dp@Hf@dp;npa=.5*dp@Ha@dp;na=.5*dag@Ha@dag
        cross=dp@Ha@err;errsq=.5*err@Ha@err
        for k,v in dict(full_n=nf,full_projected_n=npf,full_unresolved_increment_n=nr,orthogonality_cross=orth,projected_at_aggregate_damage_n=npa,damage_metric_delta=npa-npf,endpoint_cross=cross,endpoint_error_square=errsq,aggregate_n=na).items():add(k,v)
        bm=blocks[f];vectors=dict(deformation=b['L']@du[b['dofs']],thermal=-bm['theta'],relaxation=-dt*bm['R']@ag[f])
        assert np.max(np.abs(sum(vectors.values())-dag))<1e-12
        for k,v in quadratic_parts(vectors,Ha).items():forcing[k]=forcing.get(k,0.)+v
        u0=.5*a.old[f]@Ha@a.old[f];u1=.5*ag[f]@Ha@ag[f]
        faces.append(dict(face=f,aggregate_n=float(na),full_active_n=float(npf),full_n=float(nf),aggregate_stored_max=float(max(u0,u1)),aggregate_n_over_stored=float(na/max(u0,u1))))
    assert np.isclose(tot['full_n'],lf['numerical_memory_j'],rtol=1e-10)
    assert np.isclose(tot['aggregate_n'],la['numerical_memory'],rtol=1e-10)
    assert np.isclose(tot['initial_reserve_direct'],a.reserve,rtol=1e-10)
    delta=tot['aggregate_n']-tot['full_n'];recomposed=tot['damage_metric_delta']+tot['endpoint_cross']+tot['endpoint_error_square']-tot['full_unresolved_increment_n']-tot['orthogonality_cross']
    assert abs(delta-recomposed)<1e-10*max(abs(delta),1.)
    assert abs(sum(forcing.values())-tot['aggregate_n'])<1e-10*tot['aggregate_n']
    cfg=json.loads((ROOT/'analysis/genesis_adaptive_control/config.json').read_bytes());h=25.
    norm={}
    for name,n,U0,U1,Q,D,W,TW in [('aggregate',tot['aggregate_n'],a.active_initial,la['elastic_after'],la['maxwell'],la['drag'],la['external_work'],la['thermal_work']),('full',tot['full_n'],full['state']['mechanics']['elastic_energy_j']-lf['elastic_after_j']+lf['elastic_before_j'],lf['elastic_after_j'],lf['physical_maxwell_j'],lf['physical_drag_j'],lf['external_work_j'],lf['thermal_work_j'])]:
        norm[name]=dict(n_j=n,initial_energy_j=U0,endpoint_energy_j=U1,maxwell_j=Q,drag_j=D,external_work_j=W,thermal_work_j=TW,n_over_initial=n/U0,n_over_maxwell=n/Q,n_over_physical_dissipation=n/(Q+D),n_over_abs_external=n/abs(W),n_over_abs_thermal=n/abs(TW),n_over_sum_abs_work=n/(abs(W)+abs(TW)),global_stored_budget_j=cfg['global_work_atol_j_per_year']*h+cfg['global_work_rtol_stored']*max(U0,U1))
        norm[name]['global_budget_score']=n/norm[name]['global_stored_budget_j']
    fh=r['fault_history'];ff=full['state']['fault_history'];threshold=src['source']['parameters']['faults']['activation_damage']
    norm['full_projected_active']=dict(n_j=tot['full_projected_n'],initial_active_energy_j=a.active_initial,endpoint_active_energy_j=tot['full_projected_endpoint_energy'],maxwell_of_projected_field_j=tot['full_projected_maxwell'],thermal_work_of_projected_field_j=tot['full_projected_thermal_work'],n_over_initial_active=tot['full_projected_n']/a.active_initial,n_over_maxwell_projected=tot['full_projected_n']/tot['full_projected_maxwell'],n_over_abs_thermal_projected=tot['full_projected_n']/abs(tot['full_projected_thermal_work']))
    material_gate={name:max(x[name+'_n']/(cfg['local_work_atol_j_per_year']*h+cfg['local_work_rtol_stored']*x[name+'_stored']) for x in material_rows) for name in ['aggregate','full']}
    events=dict(activation_damage=threshold,aggregate_damage_max=float(max(da)),full_damage_max=float(max(df)),minimum_aggregate_margin=float(threshold-max(da)),damage_max_error=float(max(abs(da-df))),aggregate_active_count=int(sum(fh['fault_active'])),full_active_count=int(sum(ff['fault_active'])),aggregate_max_candidate_age_myr=max(fh['fault_candidate_age_myr']),full_max_candidate_age_myr=max(ff['fault_candidate_age_myr']))
    for path,pin in pins.items():assert digest(Path(path).read_bytes())==pin
    result=dict(status='SAVED_SINK_IDENTITIES_PASS_NOT_ADMISSION',new_planetary_steps=0,input_sha256=pins,decomposition_j=tot,difference_j=delta,recomposition_residual_j=delta-recomposed,aggregate_forcing_quadratic_j=forcing,normalization=norm,faces=faces,event_margins=events,original_800_support_local_budget_scores=material_gate,local_group_budget_score=max(x['aggregate_n']/(cfg['local_work_atol_j_per_year']*h+cfg['local_work_rtol_stored']*x['aggregate_stored_max']) for x in faces),seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with (ROOT/'analysis/genesis_thermal_moment/sink_audit_verified.json').open('xb') as file:file.write(encoded(result))
    print(json.dumps({k:v for k,v in result.items() if k not in ['faces','input_sha256']},indent=2))

if __name__=='__main__':main()
