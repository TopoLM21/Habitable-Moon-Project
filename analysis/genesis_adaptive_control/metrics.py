"""SI output comparisons and positive numerical-work budgets; no evolution."""
import math
import numpy as np
from tectonics.genesis_volume_loading import coefficient_inner_product,RadialMoments
from tectonics.genesis_volume_operator import conical_radial_moments
from analysis.genesis_be_candidate.audit import face_driver
from analysis.genesis_volume_fault_validation.independent_audit import physical_face_geometry


def observe_packet(p,source):
    s=p['state'];g=s['geometry'];inp=p['inputs'];cohorts=g['cohorts']
    ids=[x['id'] for x in cohorts]
    if ids!=[x['id'] for x in s['mechanics']['cohorts']]:raise ValueError('Memory ordering')
    e0=np.array([x['strain0'] for x in s['mechanics']['cohorts']]);e1=np.array([x['strain1'] for x in s['mechanics']['cohorts']])
    old0=np.array(inp['old0']);old1=np.array(inp['old1']);C=np.array(inp['elasticity_pa'])
    radial=[conical_radial_moments(x['outer_scale'],x['inner_scale']) for x in cohorts]
    moments=RadialMoments(tuple(x['volume_m3'] for x in cohorts),tuple(x[0] for x in radial),tuple(x[1] for x in radial))
    inner=lambda a,b,c,d:coefficient_inner_product(a,b,c,d,C,moments)
    U=.5*inner(e0,e1,e0,e1);U0=.5*inner(old0,old1,old0,old1);N=.5*inner(e0-old0,e1-old1,e0-old0,e1-old1)
    Q=np.array(inp['maxwell_ratio'])*2*U;L=s['interval_ledger']
    for name,value in [('numerical_memory_j',N.sum()),('physical_maxwell_j',Q.sum()),('elastic_after_j',U.sum())]:
        if not math.isclose(value,L[name],rel_tol=3e-10,abs_tol=1e3):raise ValueError('Local ledger '+name)
    v=np.array(s['velocity_m_s']).reshape(-1,6);faces=np.array(g['faces']);points=[]
    # Physical conical field is affine in N and lambda. Its vector norm maximum
    # on a cohort occurs at these 3 angular vertices × 2 radial endpoints.
    for x in cohorts:
        vv=v[faces[x['face_index']]]
        points.append([vv[:,:3]+(lam-1)*vv[:,3:] for lam in (x['outer_scale'],x['inner_scale'])])
    faults=s['fault_history'];T=np.array(p['thermal']['temperature_k']);m=np.array(p['thermal']['mass_kg'])
    if np.any(m<=0):raise ValueError('Positive cell masses')
    current=np.asarray(g['vertices_m'])+np.asarray(s['displacement_m']).reshape(-1,6)[:,:3]
    frames,_=physical_face_geometry(current,faces)
    world_stress=projected_stress_world(face_driver(s,source)['stress_pa'],frames)
    return dict(ids=ids,stress_pa=world_stress.reshape(len(faults['damage']),-1),
                velocity_m_s=np.array(points).reshape(len(ids),6,3),damage=np.array(faults['damage']),temperature_k=T,
                enthalpy_j_kg=np.array(p['thermal']['energy_j'])/m,local_work_j=N,local_stored_j=np.maximum(U0,U),
                numerical_kinetic_j=L['numerical_kinetic_j'],global_stored_j=max(L['elastic_before_j']+L['kinetic_before_j'],L['elastic_after_j']+L['kinetic_after_j']),
                physical_heat_j=Q,physical_drag_j=L['physical_drag_j'],activation_damage=source['source']['parameters']['faults']['activation_damage'],
                gate_flags=bool(any(faults['fault_active']) or any(faults['fault_candidate_age_myr'])))


def assess(whole,half1,half2,h,c):
    observations=(whole,half1,half2)
    if any(o['ids']!=whole['ids'] for o in observations):raise ValueError('Material set/order changed')
    for o in observations:
        for k in ('stress_pa','velocity_m_s','damage','temperature_k','enthalpy_j_kg','local_work_j','local_stored_j','physical_heat_j','numerical_kinetic_j','global_stored_j','physical_drag_j','activation_damage'):
            if not np.isfinite(o[k]).all():raise ValueError('Nonfinite '+k)
        for k in ('local_work_j','physical_heat_j','numerical_kinetic_j','physical_drag_j','local_stored_j','global_stored_j'):
            if np.any(np.asarray(o[k])<0):return dict(fatal='NEGATIVE_DISSIPATION_OR_STORED_ENERGY')
        if o['gate_flags'] or np.any(np.asarray(o['damage'])>=o['activation_damage']-c['damage_gate_margin']):return dict(fatal='GATE_BOUNDARY_STOP')
    checks={};factors=[]
    def output(name,a,b,atol,rtol=0,vector=False):
        a=np.asarray(a);b=np.asarray(b)
        if a.shape!=b.shape:raise ValueError('Observable shape '+name)
        if vector:
            delta=np.linalg.norm(a-b,axis=-1);scale=np.maximum(np.linalg.norm(a,axis=-1),np.linalg.norm(b,axis=-1))
        else:delta=np.abs(a-b);scale=np.maximum(np.abs(a),np.abs(b))
        budget=atol+rtol*scale;local=float(np.max(delta/budget));rms=float(np.sqrt(np.mean(delta**2)))
        global_score=rms/(atol+rtol*float(np.sqrt(np.mean(scale**2))))
        checks[name]=dict(local_ratio=local,global_ratio=global_score,max_difference=float(delta.max()),rms_difference=rms)
        ratio=max(local,global_score);factors.append(c['safety']/math.sqrt(max(ratio,1e-30)))
        return ratio
    for name,atol,rtol,vector in [('stress_pa',c['stress_atol_pa'],c['stress_rtol'],True),('velocity_m_s',c['velocity_atol_m_s'],c['velocity_rtol'],True),('damage',c['damage_atol'],0,False),('temperature_k',c['temperature_atol_k'],0,False),('enthalpy_j_kg',c['enthalpy_atol_j_kg'],0,False)]:
        output(name,whole[name],half2[name],atol,rtol,vector)
    gap=np.minimum(whole['activation_damage']-np.asarray(whole['damage']),half2['activation_damage']-np.asarray(half2['damage']))-c['damage_gate_margin']
    gate_error=float(np.max(np.abs(np.asarray(whole['damage'])-half2['damage'])/(c['damage_error_fraction_of_margin']*gap)))
    checks['damage_margin']=dict(local_ratio=gate_error,global_ratio=gate_error)
    factors.append(c['safety']/math.sqrt(max(gate_error,1e-30)))
    work=work_budget([half1,half2],h,c);checks['numerical_work']=work
    local_ratio=work['local_ratio'];global_ratio=work['global_ratio']
    # Solve N*r² <= A*r+R: absolute rate and stored-energy budget have
    # different h scalings. This avoids unnecessarily severe linear shrinkage.
    factors.append(c['safety']*work['predicted_step_factor'])
    score=max(max(x['local_ratio'],x['global_ratio']) for x in checks.values())
    return dict(score=score,next_factor=min(c['max_factor'],max(c['min_factor'],min(factors))),checks=checks,
                estimator='BE step doubling: |fine-coarse|/(2^1-1); no extrapolation or proof of global error')


def work_budget(parts,h,c):
    local=sum(np.asarray(p['local_work_j']) for p in parts)
    stored=np.maximum.reduce([p['local_stored_j'] for p in parts])
    local_budget=c['local_work_atol_j_per_year']*h+c['local_work_rtol_stored']*stored
    total=float(local.sum()+sum(p['numerical_kinetic_j'] for p in parts))
    global_budget=c['global_work_atol_j_per_year']*h+c['global_work_rtol_stored']*max(p['global_stored_j'] for p in parts)
    def allowed_factor(N,A,R):
        N=np.atleast_1d(N);A=np.broadcast_to(A,N.shape);R=np.broadcast_to(R,N.shape)
        positive=N>0
        if not np.any(positive):return c['max_factor']
        n=N[positive]/(A[positive]+R[positive]);a=A[positive]/(A[positive]+R[positive])
        return float(np.min((a+np.sqrt(a*a+4*n*(1-a)))/(2*n)))
    factor=min(allowed_factor(local,c['local_work_atol_j_per_year']*h,c['local_work_rtol_stored']*stored),
               allowed_factor(total,c['global_work_atol_j_per_year']*h,c['global_work_rtol_stored']*max(p['global_stored_j'] for p in parts)))
    return dict(predicted_step_factor=factor,local_ratio=float(np.max(local/local_budget)),global_ratio=total/global_budget,total_j=total,global_budget_j=global_budget,
                max_local_j=float(local.max()),physical_heat_j=float(sum(np.sum(p['physical_heat_j']) for p in parts)))


def projected_stress_world(stress,frames):
    stress=np.asarray(stress);f=np.asarray(frames)[:,:,:2]
    plane=np.zeros((len(stress),2,2));plane[:,0,0]=stress[:,0];plane[:,1,1]=stress[:,1]
    plane[:,0,1]=plane[:,1,0]=stress[:,2]
    return f@plane@f.transpose(0,2,1)
