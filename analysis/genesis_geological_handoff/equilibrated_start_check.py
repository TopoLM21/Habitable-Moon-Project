"""Bounded, predeclared equilibrium and coupled temporal experiment."""
from copy import deepcopy
from pathlib import Path
import json
import time

import numpy as np

from tectonics.genesis_transfer_bridge import encoded, digest
from .spherical_coupled import SphericalCoupled
from .equilibrated_start import equilibrate, point_stress, equilibrium_balance

START = Path('/workspace/spherical-coupled-preflight-initial.json')
OUT = Path('/workspace/genesis-spherical-equilibrated-01')
REPORT = Path(__file__).with_name('equilibrated_start_check.json')
SOURCE_SHA = 'a2e18177f1737fffc20d4edcf8e86a0f755b08f4891f1f8d6ec74756aa9cfdfc'


def snapshot(a):
    m=a.model
    return dict(stress=point_stress(a),T=m.temperature(m.state['energy']),
                old_damage=np.array(m.state['damage']),edge_damage=np.array(m.state['experimental_edge_fatigue']['damage']),
                velocity=np.array(m.state['nodal_velocity']))


def rms(sigma,w):
    return float(np.sqrt(np.sum(w[:,None]*sigma**2*np.array([1.,1.,1.,2.,2.,2.]))/w.sum()))


def compare(coarse,fine,initial,w):
    error=rms(coarse['stress']-fine['stress'],w)
    change=rms(fine['stress']-initial['stress'],w)
    temp=float(abs(coarse['T']-fine['T']).max());temp_change=float(abs(fine['T']-initial['T']).max())
    result=dict(stress_error_rms_pa=error,fine_stress_change_rms_pa=change,
        stress_increment_relative=error/max(change,np.finfo(float).tiny),
        temperature_error_max_k=temp,fine_temperature_change_max_k=temp_change,
        temperature_increment_relative=temp/max(temp_change,np.finfo(float).tiny),
        old_damage_error_max=float(abs(coarse['old_damage']-fine['old_damage']).max()),
        edge_damage_error_max=float(abs(coarse['edge_damage']-fine['edge_damage']).max()),
        velocity_relative=float(np.linalg.norm(coarse['velocity']-fine['velocity'])/max(np.linalg.norm(fine['velocity']),np.finfo(float).tiny)))
    result['field_gate']=result['stress_increment_relative']<=.05 and result['temperature_increment_relative']<=.02 and max(result['old_damage_error_max'],result['edge_damage_error_max'])<=1e-5
    return result


def main():
    if OUT.exists() or REPORT.exists():raise FileExistsError('Preserve prior experiment')
    start=time.monotonic();solve_seconds=0.
    if digest(START.read_bytes())!=SOURCE_SHA:raise ValueError('Source changed')
    OUT.mkdir()
    a=SphericalCoupled.load(START)
    a.origin['equilibration_input']=dict(path=str(START),sha256=SOURCE_SHA,code_base='23f12c22effc1650a0c4e7449443aa25cf632fe8')
    ledger=equilibrate(a,allow_weakened_scenario=True)
    a.save(OUT/'initial.json')
    fresh=SphericalCoupled.load(OUT/'initial.json')
    assert encoded(fresh.model.state)==encoded(a.model.state)
    assert encoded(fresh.model.config)==encoded(a.model.config)
    initial=snapshot(a);weights=a.arrays()[1];M=a.material_mass();mass=a.model.mass.copy()
    print(json.dumps(dict(equilibrium_seconds=ledger['seconds'],balance=ledger['final_balance'],elastic_change_j=ledger['signed_elastic_change_j'])),flush=True)
    runs={};fields={};restart_exact=False

    def advance(obj,dt):
        nonlocal solve_seconds
        if solve_seconds>240 or time.monotonic()-start>420:raise TimeoutError('Predeclared budget')
        tick=time.monotonic();L=obj.advance(dt);seconds=time.monotonic()-tick;solve_seconds+=seconds
        m=obj.model
        if m.state['experimental_edge_fatigue']['partition_owner'] is not None:raise RuntimeError('Certified partition: stop')
        np.testing.assert_array_equal(m.mass,mass)
        if m.state.get('born_material'):raise RuntimeError('Birth requires matched point-field diagnostics: stop')
        row=dict(seconds=seconds,whole_allowance_fraction=abs(L['whole_balance_j'])/L['whole_allowance_j'],
            mechanical_balance_relative=L['moment_balance_relative'],momentum_relative=L['momentum_relative'],
            maxwell_j=L['maxwell'],drag_j=L['basal']['heat_j'],numerical_memory_j=L['numerical_memory'],
            numerical_kinetic_j=L['numerical_kinetic'],thermal_work_j=L['thermal_work'],external_work_j=L['external_work'],
            elastic_j=L['elastic_after'],kinetic_j=L['kinetic_after'],edge_release_j=L['edge_fracture_release_j'],
            source_energy_j=L['source_energy'],new_phase_solid_kg=L['phase']['new_phase_solid_kg'],
            old_damage_change_max=float(abs(np.array(m.state['damage'])-initial['old_damage']).max()),
            edge_damage_max=max(m.state['experimental_edge_fatigue']['damage']),corrector_iterations=L['corrector_iterations'],
            births=len(m.state.get('born_material',[])),certificate=m.state['experimental_edge_fatigue']['reason'])
        assert row['whole_allowance_fraction']<=1
        assert row['mechanical_balance_relative']<=5e-9 and row['momentum_relative']<=1e-8
        assert min(row[k] for k in ('maxwell_j','drag_j','numerical_memory_j','numerical_kinetic_j'))>=0
        return row

    for name,dt,duration in [('coarse',.25,1.),('fine',.125,1.),('short_coarse',.0625,.0625),('short_fine',.03125,.0625)]:
        obj=deepcopy(a);rows=[];trace={}
        for i in range(round(duration/dt)):
            row=advance(obj,dt);row['years']=(i+1)*dt;rows.append(row)
            if name!='fine' or (i+1)%2==0:trace[str((i+1)*dt)]=snapshot(obj)
            if name=='fine' and i==0:
                fine_first=snapshot(obj);obj.save(OUT/'restart_before.json');resumed=SphericalCoupled.load(OUT/'restart_before.json')
            if name=='fine' and i==1:
                advance(resumed,dt);restart_exact=encoded(obj.model.state)==encoded(resumed.model.state);assert restart_exact
            print(json.dumps(dict(run=name,**row)),flush=True)
        if name in ('coarse','fine'):obj.save(OUT/(name+'_final.json'))
        runs[name]=rows;fields[name]=trace
    comparisons={}
    for t in fields['coarse']:
        c=compare(fields['coarse'][t],fields['fine'][t],initial,weights)
        for k in ('maxwell_j','drag_j','numerical_memory_j','numerical_kinetic_j'):
            cv=sum(row[k] for row in runs['coarse'] if row['years']<=float(t))
            fv=sum(row[k] for row in runs['fine'] if row['years']<=float(t))
            c[k]=dict(coarse=cv,fine=fv,relative_difference=abs(cv-fv)/max(abs(fv),np.finfo(float).tiny))
        c['heat_gate']=max(c[k]['relative_difference'] for k in ('maxwell_j','drag_j'))<=.05
        c['temporal_gate']=c['field_gate'] and c['heat_gate']
        comparisons[t]=c
    short=compare(fields['short_coarse']['0.0625'],fields['short_fine']['0.0625'],initial,weights)
    # Isolated, explicitly budgeted small thermal perturbation. No parameter,
    # beta, traction or threshold changes and no modification to the main path.
    control=deepcopy(a);m=control.model;node=int(m.solid_nodes[0,0]);delta=float(m.mass[node]*m.cp[node]*.001)
    old=m.state['energy'][node];m.state['energy'][node]+=delta;imposed=m.state['energy'][node]-old
    actual_T=float(m.temperature(m.state['energy'])[node]-initial['T'][node])
    control_row=advance(control,.125);control_field=snapshot(control)
    perturb=dict(node=node,requested_delta_temperature_k=.001,actual_delta_temperature_k=actual_T,imposed_enthalpy_j=imposed,
        stress_response_rms_pa=rms(control_field['stress']-fine_first['stress'],weights),
        temperature_response_max_k=float(abs(control_field['T']-fine_first['T']).max()),
        velocity_response_norm=float(np.linalg.norm(control_field['velocity']-fine_first['velocity'])),ledger=control_row)
    assert perturb['stress_response_rms_pa']>0 and perturb['temperature_response_max_k']>0
    np.savez_compressed(OUT/'fields.npz',**{f'{name}_{t}_{k}':v for name,trace in fields.items() for t,s in trace.items() for k,v in s.items()},
                        **{f'initial_{k}':v for k,v in initial.items()})
    assert digest(START.read_bytes())==SOURCE_SHA
    previous=json.loads(Path(__file__).with_name('spherical_coupled_check.json').read_text())
    prior_short={k:sum(row[oldkey] for row in previous['runs']['fine']) for k,oldkey in
                 [('numerical_memory_j','numerical_memory_j'),('drag_j','drag_heat_j'),('maxwell_j','maxwell_j')]}
    new_short={k:sum(row[k] for row in runs['short_fine']) for k in prior_short}
    result=dict(code_base='23f12c22effc1650a0c4e7449443aa25cf632fe8',source_sha256=SOURCE_SHA,
        initialization=ledger,runs=runs,comparisons=comparisons,short_comparison=short,controlled_thermal=perturb,
        previous_short_fine=prior_short,new_short_fine=new_short,
        note='Different initial states and short steps: suppressed transient is measured, not claimed as increased accuracy of original history.',
        initial_restart_exact=True,midrun_restart_exact=restart_exact,
        temporal_gate=all(c['temporal_gate'] for c in comparisons.values()),
        solve_seconds=solve_seconds,wall_seconds=time.monotonic()-start,output=str(OUT),
        files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in OUT.iterdir()})
    REPORT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(temporal_gate=result['temporal_gate'],comparisons=comparisons,controlled=perturb,wall_seconds=result['wall_seconds'],solve_seconds=solve_seconds),indent=2),flush=True)


if __name__=='__main__':main()
