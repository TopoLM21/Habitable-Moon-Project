"""Separate audited restart format, requiring the pinned inherited parent."""
from dataclasses import fields
import hashlib,json
from pathlib import Path
import numpy as np
from tectonics.genesis_bulk_birth import bulk_birth_from_dict
from tectonics.genesis_volume_geometry import GenesisVolumeGeometry,ConicalCohortGeometry,validate_genesis_volume_geometry
from tectonics.genesis_volume_thermal import thermal_from_dict
from tectonics.genesis_volume_fault_evolution import volume_fault_history_from_dict
from tectonics.genesis_volume_loading import VolumeCohortMemory
from tectonics.genesis_volume_temporal import cone_homothetic_from_dict
from .adapter import BEState,BEMemory,FORMAT,state_payload
from .audit import audit
from analysis.genesis_connected_step_validation.run import PARENT_SHA,PARENT


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)


def seal(packet):return {**packet,'sha256':hashlib.sha256(canonical(packet).encode()).hexdigest()}


def verify(packet):
    if set(packet)!={'format','parent_sha256','source_packet','state','report','sha256'}:raise ValueError('BE top-level schema')
    body={k:v for k,v in packet.items() if k!='sha256'}
    if packet['format']!=FORMAT or packet['parent_sha256']!=PARENT_SHA or seal(body)!=packet:raise ValueError('BE seal/source format')
    if set(packet['state'])!={f.name for f in fields(BEState)}:raise ValueError('BE state schema')
    if packet['state']['inherited_parent_sha256']!=PARENT_SHA:raise ValueError('Inherited parent pin')


def validate_dependencies(before,midpoint):
    raw=PARENT.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PARENT_SHA or json.loads(raw)!=before:raise ValueError('Pinned parent dependency')
    raw=Path('/workspace/genesis-connected-step-08685-01/candidate.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!='4fa1eb80eed9d19ddb44a08c747dfe3fe267a84bd953380f649d4d3834c60e98' or json.loads(raw)!=midpoint:raise ValueError('Pinned thermal oracle dependency')


def restore(packet,before,midpoint):
    validate_dependencies(before,midpoint)
    verify(packet);evidence=audit(before,packet,midpoint);s=packet['state'];source=packet['source_packet']
    tuples=lambda v:tuple(tuples(x) for x in v) if isinstance(v,list) else v
    raw=dict(s['geometry']);raw['cohorts']=tuple(ConicalCohortGeometry(**{k:tuples(v) for k,v in c.items()}) for c in raw['cohorts'])
    raw={k:tuples(v) for k,v in raw.items()};raw['source_packet_json']=canonical(source);raw['runtime_birth_history_json']=canonical(s['history'])
    geometry=GenesisVolumeGeometry(**raw);validate_genesis_volume_geometry(geometry)
    m=s['mechanics'];memory=BEMemory(m['source_identity'],m['time_myr'],m['accepted_steps'],tuple(VolumeCohortMemory(**c) for c in m['cohorts']),m['elastic_energy_j'])
    state=BEState(geometry,thermal_from_dict(s['thermal'],source_packet=source),bulk_birth_from_dict(s['history']),memory,
        volume_fault_history_from_dict(s['fault_history'],source),np.asarray(s['displacement_m']),np.asarray(s['velocity_m_s']),
        cone_homothetic_from_dict(s['last_certificate']),s['interval_ledger'],s['be_accumulated_ledger'])
    if state_payload(state)!=s:raise ValueError('Restored BE state differs')
    return state,evidence
