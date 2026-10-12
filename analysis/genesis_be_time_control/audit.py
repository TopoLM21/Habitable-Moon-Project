"""Chain audit: independent mechanics/profile checks, thermal-input replay."""
import hashlib,json
import numpy as np
from analysis.genesis_be_candidate.audit import audit as mechanical_audit,close
from analysis.genesis_connected_step_validation.run import PARENT,PARENT_SHA

# Independent list, not imported from the producer accumulation routine.
WORKS=('external_work_j','internal_work_j','thermal_work_j','physical_drag_j','physical_maxwell_j',
       'numerical_kinetic_j','numerical_memory_j','birth_kinetic_inflow_j','damage_energy_change_j',
       'remelt_kinetic_outflow_j','remelt_elastic_outflow_j')


def before_view(root,previous):
    if previous is None:return root
    s=previous['state']
    return dict(source_packet=root['source_packet'],runtime_state={**s,'mechanics':{'state':s['mechanics']},
        'last_report':{'audit_input':previous['report']['final']['inputs']}})


def verify_link(root,previous,packet):
    required={'format','parent_sha256','predecessor_sha256','step_sizes_myr','source_packet','state','report','sha256'}
    if set(packet)!=required or packet['format']!='genesis-be-chain-0.1':raise ValueError('Chain schema')
    body={k:v for k,v in packet.items() if k!='sha256'}
    if hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=packet['sha256']:raise ValueError('Chain seal')
    if packet['parent_sha256']!=PARENT_SHA or packet['state']['inherited_parent_sha256']!=PARENT_SHA:raise ValueError('Root binding')
    if packet['source_packet']!=root['source_packet']:raise ValueError('Source changed')
    prefix=PARENT_SHA if previous is None else previous['sha256'];sizes=[] if previous is None else previous['step_sizes_myr']
    if packet['predecessor_sha256']!=prefix or packet['step_sizes_myr'][:-1]!=sizes:raise ValueError('Predecessor/clock prefix')
    dt=packet['step_sizes_myr'][-1]
    if dt not in (5e-5,2.5e-5) or sum(packet['step_sizes_myr'])>.0001+1e-14:raise ValueError('Bounded interval')
    before=before_view(root,previous)['runtime_state'];after=packet['state']
    if abs(after['thermal']['time_myr']-before['thermal']['time_myr']-dt)>1e-14:raise ValueError('Clock step')
    return dt


def replay_thermal(root,previous,packet):
    """Replays prescribed thermal inputs only; no mechanical solve or model.step."""
    from tectonics.genesis_volume_thermal import initial_volume_thermal_state,_advance_core,thermal_to_dict
    from tectonics.genesis_bulk_birth import bulk_birth_from_dict,bulk_birth_to_dict
    source=root['source_packet'];before=before_view(root,previous)['runtime_state'];s=packet['state'];raw=s['thermal']
    records=raw['intervals']
    if records[:-1]!=before['thermal']['intervals']:raise ValueError('Thermal journal prefix')
    top=np.asarray(before['geometry']['vertices_m'])+np.asarray(before['displacement_m']).reshape(-1,6)[:,:3]
    close(records[-1]['surface_vertices_m'],top,'Actual thermal surface',0,0)
    state=initial_volume_thermal_state(source);history=bulk_birth_from_dict(source['bulk_birth_history'])
    for row in records:
        if row['before_time_myr']!=state.time_myr:raise ValueError('Thermal source clock')
        state,history,*_= _advance_core(state,history,row['requested_after_time_myr'],row['surface_vertices_m'],row['max_global_step_myr'])
    if thermal_to_dict(state,include_source=False)!=raw or bulk_birth_to_dict(history)!=s['history']:raise ValueError('Thermal or birth replay differs')
    return state,history


def verify_sum(root,packets,audits):
    if len(packets)!=len(audits) or not packets:raise ValueError('Missing interval audit')
    c=packets[-1]['state']['be_accumulated_ledger'];last=audits[-1]['final']['ledger'];first=audits[0]['final']['ledger']
    expected={k:sum(a['final']['ledger'].get(k,0.) for a in audits) for k in WORKS}
    expected.update({k:first[k] for k in ('kinetic_before_j','elastic_before_j')})
    expected.update({k:last[k] for k in ('kinetic_after_j','elastic_after_j')})
    if set(c)!=set(expected)|{'balance_residual_j'}:raise ValueError('Cumulative schema')
    for k,v in expected.items():close(c[k],v,'cumulative '+k,rtol=3e-8,atol=1e-6 if k in ('numerical_kinetic_j','birth_kinetic_inflow_j','kinetic_before_j','kinetic_after_j') else 1e-8)
    for k,field in [('kinetic_before_j','kinetic_energy_j'),('elastic_before_j',None)]:
        value=root['runtime_state'][field] if field else root['runtime_state']['mechanics']['state']['elastic_energy_j']
        close(c[k],value,'root energy '+k,rtol=5e-12,atol=1e-6 if field else 1e-8)
    net=expected['kinetic_after_j']+expected['elastic_after_j']-expected['kinetic_before_j']-expected['elastic_before_j']-expected['external_work_j']-expected['thermal_work_j']-expected['birth_kinetic_inflow_j']-expected['damage_energy_change_j']+sum(expected[k] for k in ['physical_drag_j','physical_maxwell_j','numerical_kinetic_j','numerical_memory_j','remelt_kinetic_outflow_j','remelt_elastic_outflow_j'])
    relative=abs(net)/max(abs(v) for v in expected.values())
    if relative>5e-10:raise ValueError('Cumulative balance')
    return dict(status='PASS',intervals=len(packets),independent_totals=expected,balance_residual_j=net,balance_relative=relative)


def audit_step(root,previous,packet):
    dt=verify_link(root,previous,packet)
    thermal,history=replay_thermal(root,previous,packet)
    result=mechanical_audit(before_view(root,previous),packet,None,dt_myr=dt,chained=True)
    result['thermal_and_birth_replay']='PASS'
    return result,thermal,history


def verify_prefixes(root,packets,audits):
    if len(packets)!=len(audits):raise ValueError('Missing prefix audit')
    result=None
    for n in range(1,len(packets)+1):result=verify_sum(root,packets[:n],audits[:n])
    if result is None:raise ValueError('Empty chain')
    return result
