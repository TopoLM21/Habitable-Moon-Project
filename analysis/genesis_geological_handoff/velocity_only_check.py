"""One bounded coupled pair after exact preservation of fracture-driving memory."""
from copy import deepcopy
from pathlib import Path
import json,time
import numpy as np
from tectonics.genesis_transfer_bridge import encoded,digest
from .spherical_coupled import SphericalCoupled
from .velocity_only_start import initialize,stress_inventory,protected_state
from .equilibrated_start_check import snapshot,compare

START=Path('/workspace/spherical-coupled-preflight-initial.json')
SOURCE_SHA='a2e18177f1737fffc20d4edcf8e86a0f755b08f4891f1f8d6ec74756aa9cfdfc'
WEAK=Path('/workspace/genesis-spherical-equilibrated-01/initial.json')
OUT=Path('/workspace/genesis-spherical-velocity-only-01')
REPORT=Path(__file__).with_name('velocity_only_check.json')


def main():
    if OUT.exists() or REPORT.exists():raise FileExistsError('Preserve prior run')
    start=time.monotonic();solve_seconds=0.;weak_sha=digest(WEAK.read_bytes())
    if digest(START.read_bytes())!=SOURCE_SHA:raise ValueError('Unexpected primary source')
    OUT.mkdir();a=SphericalCoupled.load(START);before=stress_inventory(a);protected=protected_state(a)
    a.origin['velocity_only_input']=dict(path=str(START),sha256=SOURCE_SHA,code_base='72dd1ab5d31023254604a2c8c8d5acb3a35d2fee')
    L=initialize(a);after=stress_inventory(a)
    assert encoded(before)==encoded(after) and protected_state(a)==protected
    assert after['edges_above_shear']==418 and after['faces_above_tensile']==84
    a.save(OUT/'initial.json');fresh=SphericalCoupled.load(OUT/'initial.json')
    assert encoded(fresh.model.state)==encoded(a.model.state)
    print(json.dumps(dict(initialization=L,inventory_before=before,inventory_after=after,pre_continuation_gate=True)),flush=True)
    initial=snapshot(a);weights=a.arrays()[1];mass=a.model.mass.copy();fields={};runs={};restart=False

    def step(obj,dt):
        nonlocal solve_seconds
        if time.monotonic()-start>180 or solve_seconds>90:raise TimeoutError('Predeclared budget')
        tick=time.monotonic();r=obj.advance(dt);seconds=time.monotonic()-tick;solve_seconds+=seconds
        m=obj.model;np.testing.assert_array_equal(m.mass,mass)
        if m.state['experimental_edge_fatigue']['partition_owner'] is not None:raise RuntimeError('Certified partition: stop')
        if m.state.get('born_material'):raise RuntimeError('New birth: stop bounded comparison')
        row=dict(seconds=seconds,whole_allowance_fraction=abs(r['whole_balance_j'])/r['whole_allowance_j'],
            mechanical_balance_relative=r['moment_balance_relative'],momentum_relative=r['momentum_relative'],
            maxwell_j=r['maxwell'],drag_j=r['basal']['heat_j'],numerical_memory_j=r['numerical_memory'],numerical_kinetic_j=r['numerical_kinetic'],
            thermal_work_j=r['thermal_work'],external_work_j=r['external_work'],elastic_j=r['elastic_after'],kinetic_j=r['kinetic_after'],
            edge_release_j=r['edge_fracture_release_j'],edge_damage_max=max(m.state['experimental_edge_fatigue']['damage']),
            old_damage_max=max(m.state['damage']),births=0,certificate=m.state['experimental_edge_fatigue']['reason'])
        assert row['whole_allowance_fraction']<=1 and row['mechanical_balance_relative']<=5e-9 and row['momentum_relative']<=1e-8
        assert min(row[k] for k in ('maxwell_j','drag_j','numerical_memory_j','numerical_kinetic_j'))>=0
        return row

    for name,dt in [('coarse',.03125),('fine',.015625)]:
        obj=deepcopy(a);rows=[];trace={}
        for i in range(round(.0625/dt)):
            row=step(obj,dt);row['years']=(i+1)*dt;rows.append(row)
            if name=='coarse' or (i+1)%2==0:trace[str((i+1)*dt)]=snapshot(obj)
            if name=='fine' and i==0:
                obj.save(OUT/'restart_before.json');resume=SphericalCoupled.load(OUT/'restart_before.json')
            if name=='fine' and i==1:
                step(resume,dt);restart=encoded(resume.model.state)==encoded(obj.model.state);assert restart
            print(json.dumps(dict(run=name,**row)),flush=True)
        obj.save(OUT/(name+'_final.json'));runs[name]=rows;fields[name]=trace
        if name=='fine':fine=obj
    comparisons={}
    for t in fields['coarse']:
        c=compare(fields['coarse'][t],fields['fine'][t],initial,weights)
        for k in ('maxwell_j','drag_j','numerical_memory_j','numerical_kinetic_j'):
            cv=sum(x[k] for x in runs['coarse'] if x['years']<=float(t));fv=sum(x[k] for x in runs['fine'] if x['years']<=float(t))
            c[k]=dict(coarse=cv,fine=fv,relative_difference=abs(cv-fv)/max(abs(fv),np.finfo(float).tiny))
        c['heat_gate']=max(c[k]['relative_difference'] for k in ('maxwell_j','drag_j'))<=.05
        c['temporal_gate']=c['field_gate'] and c['heat_gate'];comparisons[t]=c
    old=json.loads(Path(__file__).with_name('spherical_coupled_check.json').read_text())
    baseline={name:dict(seconds=sum(x['seconds'] for x in old['runs'][name]),
                         maxwell_j=sum(x['maxwell_j'] for x in old['runs'][name]),drag_j=sum(x['drag_heat_j'] for x in old['runs'][name]),
                         numerical_memory_j=sum(x['numerical_memory_j'] for x in old['runs'][name])) for name in runs}
    previous=SphericalCoupled.load('/workspace/genesis-spherical-coupled-01/fine_final.json')
    baseline_field=compare(snapshot(fine),snapshot(previous),initial,weights)
    np.savez_compressed(OUT/'fields.npz',**{f'{name}_{t}_{k}':v for name,trace in fields.items() for t,s in trace.items() for k,v in s.items()},
                        **{f'initial_{k}':v for k,v in initial.items()})
    assert digest(START.read_bytes())==SOURCE_SHA and digest(WEAK.read_bytes())==weak_sha
    result=dict(code_base='72dd1ab5d31023254604a2c8c8d5acb3a35d2fee',source_sha256=SOURCE_SHA,weakened_scenario_sha256=weak_sha,
        initialization=L,inventory_before=before,inventory_after=after,pre_continuation_memory_gate=True,
        runs=runs,comparisons=comparisons,original_pilot_baseline=baseline,final_vs_original_pilot=baseline_field,
        final_inventory=stress_inventory(fine),initial_restart_exact=True,midrun_restart_exact=restart,
        temporal_gate=all(c['temporal_gate'] for c in comparisons.values()),solve_seconds=solve_seconds,wall_seconds=time.monotonic()-start,
        output=str(OUT),files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in OUT.iterdir()})
    REPORT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(temporal_gate=result['temporal_gate'],comparisons=comparisons,baseline=baseline,final_vs_original_pilot=baseline_field,
                         wall_seconds=result['wall_seconds'],solve_seconds=solve_seconds),indent=2),flush=True)


if __name__=='__main__':main()
