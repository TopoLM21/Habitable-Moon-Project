"""Bounded opt-in step doubling; accepted state is unchanged after any rejection.

No change to source physics, safety gates, or the ordinary runner/defaults.
"""
from copy import deepcopy
import time
import numpy as np
from .simplified_genesis import SimplifiedGenesis
from .experimental_edge_weakening import MODE,face_damage
from .genesis_transfer_bridge import encoded

DEFAULT=dict(initial_years=.25,min_years=1/64,max_years=4.,stress_tolerance=.01,
             damage_tolerance=.002,temperature_tolerance_k=.05,reserve_increment_tolerance=.05,
             slow_damage_increment=.01,slow_stress_change=.10,max_attempts=240,wall_seconds=180.,
             growth_rule='predict absolute errors quadratically and relative reserve increment linearly')


def edge_field(m):
    # Same physical stress reconstruction as the edge rule, independent of saved pre-update traction.
    from .simplified_genesis_nodal import operators
    d=np.array(m.state.get('experimental_edge_fatigue',{}).get('damage',np.zeros(len(m.mesh.shared_edges))))
    effective=m.scale(np.maximum(m.state['damage'],face_damage(m.mesh,d)))
    sigma=[];volume=m.weights.sum(axis=1).copy()
    for f in range(m.nf):
        e=m.P[f]@np.array(m.state['memory'][f]);sigma.append(np.sum(m.weights[f,:,None]*(e@(m.C[f]*effective[f])),axis=0))
    sigma=np.array(sigma)
    for x in m.state.get('born_material',[]):
        f=x['face'];sigma[f]+=x['volume_m3']*(m.C[f]*effective[f])@np.array(x['memory']);volume[f]+=x['volume_m3']
    sigma/=volume[:,None];tensor=np.zeros((m.nf,3,3));tensor[:,range(3),range(3)]=sigma[:,:3]
    for k,(i,j) in enumerate(((0,1),(0,2),(1,2)),3):tensor[:,i,j]=tensor[:,j,i]=sigma[:,k]
    frames=operators(m)['frames'];world=np.einsum('fai,fij,fbj->fab',frames,tensor,frames);out=[]
    for f,g,i,j in m.mesh.shared_edges:
        a,b=m.mesh.vertices[[i,j]];rad=a+b;rad/=np.linalg.norm(rad);t=b-a;t/=np.linalg.norm(t);n=np.cross(t,rad)
        traction=.5*(world[f]+world[g])@n;out.append(traction-n*(traction@n))
    return d,np.array(out)


def attempt(model,years,settings=DEFAULT):
    before=encoded(model.state);coarse=SimplifiedGenesis(model.config,model.state);fine=SimplifiedGenesis(model.config,model.state)
    try:
        coarse.advance(years);fine.advance(years/2);fine.advance((coarse.state['time_myr']-fine.state['time_myr'])*1e6)
    except ValueError as e:
        assert encoded(model.state)==before
        return None,dict(accepted=False,guard=str(e),error_ratio=None)
    dc,tc=edge_field(coarse);df,tf=edge_field(fine);d0,t0=edge_field(model)
    denom=max(float(np.linalg.norm(tf,axis=1).max()),1.)
    stress=float(np.linalg.norm(tc-tf,axis=1).max()/denom);damage=float(np.max(abs(dc-df)))
    old_damage=float(np.max(abs(np.array(coarse.state['damage'])-np.array(fine.state['damage']))))
    temp=float(np.max(abs(coarse.temperature(coarse.state['energy'])-fine.temperature(fine.state['energy']))))
    prior=model.state.get('experimental_edge_fracture_reserve_j',0.)
    rc=coarse.state.get('experimental_edge_fracture_reserve_j',0.)-prior;rf=fine.state.get('experimental_edge_fracture_reserve_j',0.)-prior
    reserve=abs(rc-rf)/max(abs(rf),1.)
    phase_same=[p['cell_id'] for p in coarse.state['born_material']]==[p['cell_id'] for p in fine.state['born_material']]
    masks_same=np.array_equal(dc>=1,df>=1)
    candidate_same=coarse.state['experimental_edge_fatigue']['partition_owner']==fine.state['experimental_edge_fatigue']['partition_owner']
    error=max(stress/settings['stress_tolerance'],damage/settings['damage_tolerance'],old_damage/settings['damage_tolerance'],temp/settings['temperature_tolerance_k'],reserve/settings['reserve_increment_tolerance'])
    accepted=error<=1 and phase_same and masks_same and candidate_same
    slow_damage=float(np.max(abs(df-d0)));slow_stress=float(np.linalg.norm(tf-t0,axis=1).max()/max(float(np.linalg.norm(t0,axis=1).max()),1.))
    row=dict(accepted=bool(accepted),guard=None,error_ratio=error,stress_difference=stress,damage_difference=damage,
        old_damage_difference=old_damage,temperature_difference_k=temp,reserve_increment_relative=reserve,
        event_agreement=bool(phase_same and masks_same and candidate_same),damage_increment=slow_damage,stress_change=slow_stress,
        slow=bool(slow_damage<=settings['slow_damage_increment'] and slow_stress<=settings['slow_stress_change']),
        input_state_unchanged=encoded(model.state)==before)
    assert row['input_state_unchanged']
    return (fine if accepted else None),row


def integrate(model,horizon_years,settings=None):
    settings={**DEFAULT,**(settings or {})}
    if model.config.get('experimental_geological_edge_fatigue',{}).get('mode')!=MODE:raise ValueError('ADAPTIVE_REQUIRES_EXPLICIT_EDGE_WEAKENING')
    if not np.isfinite(horizon_years) or horizon_years<=0:raise ValueError('ADAPTIVE_HORIZON')
    if not 0<settings['min_years']<=settings['initial_years']<=settings['max_years']<=model.config['max_step_years']:raise ValueError('ADAPTIVE_STEP_BOUNDS')
    if any(not np.isfinite(settings[k]) or settings[k]<=0 for k in ('stress_tolerance','damage_tolerance','temperature_tolerance_k','reserve_increment_tolerance','wall_seconds')):raise ValueError('ADAPTIVE_POSITIVE_TOLERANCES')
    clock=time.monotonic();current=SimplifiedGenesis(model.config,model.state);target=current.state['time_myr']+horizon_years/1e6
    h=settings['initial_years'];rows=[];accepted=[];low_count=0;status='COMPLETED'
    for index in range(settings['max_attempts']):
        if current.state['time_myr']>=target:break
        if time.monotonic()-clock>settings['wall_seconds']:status='WALL_BUDGET_STOP';break
        remaining=(target-current.state['time_myr'])*1e6;h=min(h,remaining)
        proposed,row=attempt(current,h,settings);row.update(years_before=(current.state['time_myr']-model.state['time_myr'])*1e6,dt_years=h)
        rows.append(row)
        if not row['accepted']:
            low_count=0
            if h<=settings['min_years']*(1+1e-9):status='MIN_STEP_REJECTED';break
            h=max(settings['min_years'],h/2);continue
        current=proposed;d,t=edge_field(current)
        accepted.append(dict(years=(current.state['time_myr']-model.state['time_myr'])*1e6,dt_years=h,damage=d.tolist(),traction_pa=t.tolist(),
            reserve_j=current.state['experimental_edge_fracture_reserve_j'],reason=current.state['experimental_edge_fatigue']['reason'],
            energy_fraction=abs(current.state['last_ledger']['whole_balance_j'])/current.state['last_ledger']['whole_allowance_j']))
        if current.state['experimental_edge_fatigue']['partition_owner'] is not None:status='HANDOFF_PENDING';break
        # BE step-doubling absolute defects scale ~h²; dividing the reserve
        # defect by its O(h) increment gives an O(h) relative estimator.
        # This predicts a proposal only; every larger step is checked again.
        predicted=max(4*row['stress_difference']/settings['stress_tolerance'],
            4*max(row['damage_difference'],row['old_damage_difference'])/settings['damage_tolerance'],
            4*row['temperature_difference_k']/settings['temperature_tolerance_k'],
            2*row['reserve_increment_relative']/settings['reserve_increment_tolerance'])
        row['predicted_doubled_error_ratio']=predicted
        low_count=low_count+1 if row['slow'] and predicted<.8 else 0
        if low_count>=2:h=min(settings['max_years'],h*2);low_count=0
    else:status='ATTEMPT_BUDGET_STOP'
    return current,dict(status=status,settings=settings,attempts=rows,accepted=accepted,seconds=time.monotonic()-clock,
        start_time_myr=model.state['time_myr'],end_time_myr=current.state['time_myr'],target_time_myr=target)
