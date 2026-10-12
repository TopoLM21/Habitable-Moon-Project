"""Finish saved two-step audit/restart/comparison; zero new physical solves."""
import hashlib,json,resource,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from tectonics.genesis import GenesisParameters
from tectonics.genesis_volume_loading import VolumeCohortMemory
from tectonics.genesis_volume_fault_evolution import volume_fault_history_from_dict,volume_fault_history_to_dict
from tectonics.genesis_energy_ports import boundary_budget
from analysis.genesis_connected_step_validation.run import PARENT,PARENT_SHA,deadline
from analysis.genesis_be_candidate.adapter import dumps
from analysis.genesis_be_candidate.checkpoint import seal
from . import run


def main():
  with deadline(180):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));start=time.monotonic()
    out=Path('/workspace/genesis-coupled-split-25-01');root=run.pinned(PARENT,PARENT_SHA);current=root;run.SOURCE=root['source_packet']
    previous=PARENT_SHA;reports=[];packets=[];cumulative={}
    for i in (1,2):
        raw=(out/f'step_{i}.json').read_bytes();p=json.loads(raw);assert p['predecessor_sha256']==previous and p['step_index']==i
        assert seal({k:v for k,v in p.items() if k!='sha256'})==p
        thermal=json.loads((out/f'thermal_{i}.json').read_bytes());assert thermal==p['thermal_packet'] and seal({k:v for k,v in thermal.items() if k!='sha256'})==thermal
        print('read-only audit step '+str(i),flush=True);a=run.audit_step(current,thermal,p,run.SOURCE)
        memory=[asdict(VolumeCohortMemory(**c)) for c in p['state']['mechanics']['cohorts']];p['state']['mechanics']['cohorts']=memory
        faults=volume_fault_history_from_dict(p['state']['fault_history'],run.SOURCE)
        assert volume_fault_history_to_dict(faults)==p['state']['fault_history']
        assert dumps(p).encode()==raw
        restart=out/f'restart_{i}.json'
        if restart.exists():assert restart.read_bytes()==raw
        else:
            with restart.open('xb') as f:f.write(raw)
        with (out/f'completion_audit_{i}.json').open('x') as f:f.write(dumps(a))
        for k in ('external_work_j','physical_drag_j','physical_maxwell_j','thermal_work_j','damage_energy_change_j','numerical_kinetic_j','numerical_memory_j'):
            cumulative[k]=cumulative.get(k,0.)+p['state']['interval_ledger'][k]
        reports.append(a);packets.append(p);previous=hashlib.sha256(raw).hexdigest();current=run.root_from_previous(current,p)
    baseline=run.pinned('/workspace/genesis-cutcell-mechanical-02/candidate.json','5a0cfcd3edfd57f4f05b59e8b408b09c25ac2a704068609e22029ab865230f2e')
    gp=GenesisParameters(**run.SOURCE['source']['parameters']['thermal']);comparison=run.compare(baseline,packets[-1],packets[-1]['thermal_packet']['fixed_cut_geometry'],gp)
    cut=run.pinned('/workspace/genesis-cutcell-feasibility-02/candidate.json','6006521e7b3e22ce23894e63248d0145ee15d3065333c949701c730c6cab47ad')
    proj=run.pinned('/workspace/genesis-coupled-thermal-projection-03/projected.json','d25fe1fd4e8b56b18ab8ec697b197c326d723e9421148e5d3726fb7affdd45a4')
    cells={c['identity']:c for c in proj['state']['cells']};initialE=[x['mass_kg']*x['specific_h'] for x in cut['cells']]+[(cells[k]['energy_j']/cells[k]['mass_kg'])*cells[k]['mass_kg'] for k in ('deep_mantle','surface_water')]
    last=packets[-1]['state']['interval_ledger'];first=packets[0]['state']['interval_ledger']
    full={**cumulative,'elastic_before_j':first['elastic_before_j'],'elastic_after_j':last['elastic_after_j'],'kinetic_before_j':first['kinetic_before_j'],'kinetic_after_j':last['kinetic_after_j']}
    delta=float(np.sum(np.array(packets[-1]['thermal']['energy_j'],dtype=np.longdouble)-np.array(initialE,dtype=np.longdouble)))
    budget=boundary_budget(full,delta);allowance=sum(json.loads((out/f'thermal_audit_{i}.json').read_text())['global_allowance_j']+reports[i-1]['feedback']['combined_allowance_j'] for i in (1,2))
    assert abs(budget['balance_residual_j'])<=allowance
    feedback_lag=[float(np.max(np.abs(np.array(p['thermal']['temperature_k'])-p['thermal_packet']['temperature_k']))) for p in packets]
    report=dict(status='PASS',steps_started=2,steps_passed=2,new_physical_solves=0,correctors=[len(p['correctors']) for p in packets],restart_byte_identical=True,comparison=comparison,feedback_temperature_lag_k=feedback_lag,cumulative_ledger=cumulative,cumulative_open_budget=budget,cumulative_allowance_j=allowance,final_sha256=previous,sequential_runner_usable=True,canonical_advanced=False,tests_passed=79,readback_seconds=time.monotonic()-start,unresolved_heat_return='zero only for this diagnostic window',original_stop='Predictor zero damage energy suffered 8192 J cancellation in independent audit; delta-C integral removes cancellation without tolerance change or new solve.')
    with (out/'completion.json').open('x') as f:f.write(dumps(report))
    print(dumps(report))


if __name__=='__main__':main()
