"""Unapplied exchange journal, derived only from pinned saved BE50 states."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from .audit import ROOT,ROOT_SHA,STEPS,read_pinned

FORMAT='genesis-unapplied-energy-exchanges-0.1'
SCHEME='backward_euler_velocity_and_maxwell_endpoint'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def seal(body):
    return {**body,'sha256':hashlib.sha256(canonical(body)).hexdigest()}


def exchanges(ledger):
    def required(key):
        value=ledger[key]
        if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Nonfinite/non-numeric '+key)
        return value
    drag,maxwell=required('physical_drag_j'),required('physical_maxwell_j')
    nk,nm=required('numerical_kinetic_j'),required('numerical_memory_j')
    if min(drag,maxwell,nk,nm)<0:raise ValueError('Negative dissipation')
    return dict(
      physical_dissipation_unassigned=dict(value_j=math.fsum([drag,maxwell]),recipient='unassigned',
        components_j=dict(basal_drag=drag,maxwell=maxwell),source_fields=['physical_drag_j','physical_maxwell_j']),
      thermomechanical_counterentry_pending=dict(value_j=-required('thermal_work_j'),counterparty='unassigned',source_fields=['thermal_work_j'],source_multiplier=-1),
      damage_destination_pending=dict(value_j=-required('damage_energy_change_j'),recipient='unassigned',source_fields=['damage_energy_change_j'],source_multiplier=-1),
      birth_donor_debit_pending=dict(value_j=-required('birth_kinetic_inflow_j'),donor='unassigned',source_fields=['birth_kinetic_inflow_j'],source_multiplier=-1),
      traction_donor_debit_pending=dict(value_j=-required('external_work_j'),donor='unassigned',source_fields=['external_work_j'],source_multiplier=-1),
      numerical_dissipation_diagnostic=dict(value_j=math.fsum([nk,nm]),destination='diagnostic_only_never_heat',
        components_j=dict(kinetic=nk,memory=nm),source_fields=['numerical_kinetic_j','numerical_memory_j']))


def build(directory):
    root,pin=read_pinned(ROOT,ROOT_SHA);pins=[pin];before=root['runtime_state']['thermal']['time_myr'];start=before
    rows=[];previous=ROOT_SHA
    for index,(name,sha) in enumerate(STEPS,1):
        packet,pin=read_pinned(directory/name,sha);pins.append(pin)
        state=packet['state'];after=state['thermal']['time_myr']
        if (packet['predecessor_sha256']!=previous or packet['step_sizes_myr']!=[5e-5]*index
            or state['scheme']!=SCHEME or not math.isclose(after-before,5e-5,rel_tol=0,abs_tol=1e-14)):
            raise ValueError('BE50 chain/clock/scheme differs')
        rows.append(dict(index=index,before_time_myr=before,after_time_myr=after,
          source_raw_sha256=sha,source_packet_sha256=packet['sha256'],source_field='state.interval_ledger',
          exchanges=exchanges(state['interval_ledger'])))
        previous=packet['sha256'];before=after
    total=exchanges(state['be_accumulated_ledger'])
    for key,value in total.items():
        summed=math.fsum(row['exchanges'][key]['value_j'] for row in rows)
        if not math.isclose(summed,value['value_j'],rel_tol=1e-12,abs_tol=1e-9):raise ValueError('Window sum differs: '+key)
    return seal(dict(format=FORMAT,integrator='be50',scheme=SCHEME,dt_myr=5e-5,
      units=dict(energy='J',time='Myr'),applied_to_thermal_state=False,historical_opening_balance=None,
      semantics='Unapplied exchange records for this window; not a reservoir, temperature increment or calculated heating',
      combined_heating_j=None,combined_global_energy_closed=False,new_Genesis_steps=0,
      before_time_myr=start,after_time_myr=before,input_pins=pins,intervals=rows,
      window=dict(source_raw_sha256=pins[-1]['sha256'],source_field='state.be_accumulated_ledger',exchanges=total)))


def validate(value,expected):
    if not isinstance(value,dict) or 'sha256' not in value:raise ValueError('Sidecar schema')
    body={k:v for k,v in value.items() if k!='sha256'}
    if seal(body)!=value:raise ValueError('Sidecar seal mismatch')
    # Full comparison to a fresh derivation binds signs, unknowns, units, clocks,
    # inputs and separation of numerical/physical losses, even after resealing.
    if canonical(value)!=canonical(expected):raise ValueError('Sidecar differs from pinned-source derivation')
    return True


def write_new(path,value):
    encoded=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    with path.open('x') as f:f.write(encoded)
    validate(json.loads(path.read_bytes()),value)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('/workspace/genesis-be-time-control-01/50'))
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--output',type=Path);group.add_argument('--verify',type=Path)
    a=p.parse_args()
    if a.output is not None and a.output.exists():raise ValueError('Output exists')
    value=build(a.input)
    if a.output is not None:write_new(a.output,value)
    else:validate(json.loads(a.verify.read_bytes()),value)
    print(json.dumps(dict(status='PASS',sidecar_sha256=value['sha256'],applied_to_thermal_state=False,new_Genesis_steps=0)))

if __name__=='__main__':main()
