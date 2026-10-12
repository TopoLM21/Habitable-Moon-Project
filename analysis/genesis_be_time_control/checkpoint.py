"""Restore a complete BE chain by audit, without repeating mechanics."""
import hashlib,json
import numpy as np
from tectonics.genesis_volume_geometry import GenesisVolumeGeometry,ConicalCohortGeometry,validate_genesis_volume_geometry
from tectonics.genesis_volume_loading import VolumeCohortMemory
from tectonics.genesis_volume_fault_evolution import volume_fault_history_from_dict
from tectonics.genesis_volume_temporal import cone_homothetic_from_dict
from analysis.genesis_be_candidate.adapter import BEState,BEMemory,state_payload
from analysis.genesis_connected_step_validation.run import PARENT,PARENT_SHA
from .audit import audit_step,verify_prefixes


def restore(root,packets):
    raw=PARENT.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PARENT_SHA or json.loads(raw)!=root:raise ValueError('Unauthenticated root')
    previous=None;audits=[]
    for packet in packets:
        result,thermal,history=audit_step(root,previous,packet);audits.append(result);previous=packet
    totals=verify_prefixes(root,packets,audits);s=packets[-1]['state'];source=root['source_packet']
    tuples=lambda v:tuple(tuples(x) for x in v) if isinstance(v,list) else v
    raw=dict(s['geometry']);raw['cohorts']=tuple(ConicalCohortGeometry(**{k:tuples(v) for k,v in c.items()}) for c in raw['cohorts'])
    raw={k:tuples(v) for k,v in raw.items()}
    raw['source_packet_json']=json.dumps(source,sort_keys=True,separators=(',',':'),ensure_ascii=False)
    raw['runtime_birth_history_json']=json.dumps(s['history'],sort_keys=True,separators=(',',':'),ensure_ascii=False)
    geometry=GenesisVolumeGeometry(**raw);validate_genesis_volume_geometry(geometry)
    m=s['mechanics'];memory=BEMemory(m['source_identity'],m['time_myr'],m['accepted_steps'],tuple(VolumeCohortMemory(**c) for c in m['cohorts']),m['elastic_energy_j'])
    state=BEState(geometry,thermal,history,memory,volume_fault_history_from_dict(s['fault_history'],source),np.asarray(s['displacement_m']),np.asarray(s['velocity_m_s']),
        cone_homothetic_from_dict(s['last_certificate']),s['interval_ledger'],s['be_accumulated_ledger'])
    if state_payload(state)!=s:raise ValueError('Typed chain restart differs')
    return state,totals
