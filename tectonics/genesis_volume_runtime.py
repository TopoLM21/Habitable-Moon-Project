"""Explicit intact execution policy around the unchanged VolumeSlipModel.

BE reuses the independently tested chain implementation. No automatic scheme
selection, numerical-heat conversion, active-plane or remelt extension.
"""
import hashlib,json,io
from pathlib import Path
import numpy as np
from analysis.genesis_connected_step_validation.run import PARENT_SHA,FORMATION_SHA,gate
from analysis.genesis_be_candidate.adapter import state_payload
from analysis.genesis_be_time_control.chain import transition
from analysis.genesis_be_time_control.audit import audit_step
from analysis.genesis_be_time_control.checkpoint import restore
from .genesis_volume_checkpoint import _reject_duplicate_keys,_reject_nonfinite_constant
from .genesis_volume_slip_checkpoint import load_volume_slip_checkpoint,volume_slip_checkpoint_to_dict

FORMAT='genesis-volume-integrator-runtime-0.1'
MODES={'midpoint':(.0001,'legacy_default'), 'be50':(5e-5,'engineering_candidate'), 'be25':(2.5e-5,'control_mode')}


def metadata(mode):
    if mode not in MODES:raise ValueError('Integrator must be midpoint, be50 or be25')
    return dict(integrator=mode,scheme='midpoint' if mode=='midpoint' else 'backward_euler_velocity_and_maxwell_endpoint',
        dt_myr=MODES[mode][0],role=MODES[mode][1],physical_accuracy_certified=False,
        numerical_dissipation_destination='diagnostic_only_no_thermal_feedback',scope='intact_stop_before_activation_no_remelt')


def seal(body):
    return {**body,'sha256':hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()}


def validate_bundle(packet,mode):
    required={'format','integrator_metadata','root_checkpoint_utf8','chain','sha256'}
    if not isinstance(packet,dict) or set(packet)!=required or packet['format']!=FORMAT:raise ValueError('Runtime bundle schema')
    if mode=='midpoint' or packet['integrator_metadata']!=metadata(mode):raise ValueError('Integrator mismatch: select the stored BE mode explicitly; journals cannot be mixed')
    body={k:v for k,v in packet.items() if k!='sha256'}
    if seal(body)!=packet:raise ValueError('Runtime bundle seal')
    raw=packet['root_checkpoint_utf8'].encode('utf8')
    if hashlib.sha256(raw).hexdigest()!=PARENT_SHA:raise ValueError('BE requires the validated 0.8684 Myr root')
    chain=packet['chain']
    if not isinstance(chain,list):raise ValueError('Missing chain')
    for i,p in enumerate(chain):
        if p['step_sizes_myr']!=[MODES[mode][0]]*(i+1):raise ValueError('Mode/dt journal mismatch')
    return raw


class VolumeRuntime:
    @classmethod
    def from_root(cls,model,state,*,integrator='midpoint',root_bytes=None):
        metadata(integrator);model.validate_state(state)
        obj=cls();obj.model=model;obj.state=state;obj.mode=integrator;obj.chain=[];obj.audits=[];obj.root_bytes=root_bytes
        if integrator!='midpoint':
            if root_bytes is None or hashlib.sha256(root_bytes).hexdigest()!=PARENT_SHA:raise ValueError('Explicit authenticated root bytes required for BE')
            obj.root=json.loads(root_bytes)
            if volume_slip_checkpoint_to_dict(model,state)!=obj.root:raise ValueError('BE cannot silently rebase an existing midpoint state')
        else:obj.root=None
        gate(state.fault_history,model.fault_parameters)
        return obj

    def attach_chain(self,packets):
        if self.mode=='midpoint' or self.chain:raise ValueError('Attach requires a fresh explicit BE runtime')
        bundle=seal(dict(format=FORMAT,integrator_metadata=metadata(self.mode),root_checkpoint_utf8=self.root_bytes.decode('utf8'),chain=packets))
        validate_bundle(bundle,self.mode)
        if packets:
            state,total=restore(self.root,packets,root_bytes=self.root_bytes,max_elapsed_myr=None)
            gate(state.fault_history,self.model.fault_parameters)
            self.state=state;self.chain=list(packets);self.last_audit=total
        return self

    def advance(self,*,dt_myr=None):
        dt=MODES[self.mode][0] if dt_myr is None else dt_myr
        if not np.isfinite(dt) or dt<=0:raise ValueError('Positive finite dt required')
        if self.mode!='midpoint' and dt!=MODES[self.mode][0]:raise ValueError('BE dt differs from saved mode')
        gate(self.state.fault_history,self.model.fault_parameters)
        if self.mode=='midpoint':
            before=volume_slip_checkpoint_to_dict(self.model,self.state)
            candidate=self.model.step(self.state,dt)
            gate(candidate.fault_history,self.model.fault_parameters)
            from analysis.genesis_volume_slip_validation.independent_audit import audit_actual_interval
            proof=audit_actual_interval(before,volume_slip_checkpoint_to_dict(self.model,candidate))
            self.state=candidate;self.last_audit=proof
        else:
            previous=self.chain[-1] if self.chain else None
            candidate,packet=transition(self.model,self.state,previous,dt,max_elapsed_myr=None)
            proof,*_=audit_step(self.root,previous,packet,max_elapsed_myr=None)
            # Audit the cumulative chain before committing the new runtime state.
            prospective=self.chain+[packet]
            restored,total=restore(self.root,prospective,root_bytes=self.root_bytes,max_elapsed_myr=None)
            if state_payload(restored)!=state_payload(candidate):raise ValueError('Advance/restart state mismatch')
            self.chain=prospective;self.state=restored;self.last_audit=total
        return self.state

    def packet(self):
        if self.mode=='midpoint':return volume_slip_checkpoint_to_dict(self.model,self.state)
        if self.chain and state_payload(self.state)!=self.chain[-1]['state']:raise ValueError('Runtime state lost chain binding')
        return seal(dict(format=FORMAT,integrator_metadata=metadata(self.mode),root_checkpoint_utf8=self.root_bytes.decode('utf8'),chain=self.chain))

    def save(self,path):
        # Exclusive creation: never replace accepted history or an existing file.
        packet=self.packet()
        with Path(path).open('x',encoding='utf8',newline='') as f:
            json.dump(packet,f,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

    def describe(self):
        return dict(integrator_metadata=metadata(self.mode),time_myr=self.state.time_myr,
                    cohorts=len(self.state.geometry.cohorts),BE_intervals=len(self.chain),gate=gate(self.state.fault_history,self.model.fault_parameters))


def load_runtime(path,*,integrator='midpoint'):
    metadata(integrator);raw=Path(path).read_bytes();packet=json.loads(raw,object_pairs_hook=_reject_duplicate_keys,parse_constant=_reject_nonfinite_constant)
    if not isinstance(packet,dict):raise ValueError('Checkpoint must be an object')
    if packet.get('format')==FORMAT:
        root_raw=validate_bundle(packet,integrator)
        model,state=load_volume_slip_checkpoint(io.StringIO(root_raw.decode('utf8')),expected_source_checkpoint_sha256=FORMATION_SHA)
        return model.configured_runtime(state,integrator=integrator,root_bytes=root_raw).attach_chain(packet['chain'])
    if packet.get('format')!='genesis-volume-slip-checkpoint-0.1':raise ValueError('Unsupported checkpoint format; pack a validated chain explicitly')
    if integrator!='midpoint' and hashlib.sha256(raw).hexdigest()!=PARENT_SHA:raise ValueError('BE starts only from the authenticated canonical root or its own bundle')
    model,state=load_volume_slip_checkpoint(path,expected_source_checkpoint_sha256=FORMATION_SHA)
    return model.configured_runtime(state,integrator=integrator,root_bytes=raw)
