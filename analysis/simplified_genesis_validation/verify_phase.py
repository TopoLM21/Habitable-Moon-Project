"""Read-only real trajectory inspection; artificial birth/restart exercise."""
import json,time
from pathlib import Path
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.simplified_genesis_phase import solid_inventory,subdivide_uniform_cell
from tectonics.genesis_transfer_bridge import encoded,digest
from .phase_synthetic import freezing_model

ROOT=Path(__file__).resolve().parents[2];OUT=Path('/workspace/simplified-genesis-phase-pilot-01')

def main():
    start=time.perf_counter();first=SimplifiedGenesis.load(OUT/'initial.json');last=SimplifiedGenesis.load(OUT/'final.json');prior=SimplifiedGenesis.load('/workspace/simplified-genesis-pilot-01/final.json')
    rows=[json.loads((OUT/f'step_{i}.json').read_bytes())['payload']['state'] for i in range(1,4)]
    # Exact same physical trajectory on this horizon: phase tracking is a
    # conservative diagnostic until a whole cell reaches the solidus.
    fields=['time_myr','energy','memory','candidate_age','owner','omega','rotation','events','passive_reserve_j']
    assert all(encoded(last.state[k])==encoded(prior.state[k]) for k in fields)
    damage_roundoff=float(np.max(abs(np.array(last.state['damage'])-prior.state['damage'])))
    assert damage_roundoff<=4*np.finfo(float).eps
    cfg=first.config
    def inventory(model):
        return solid_inventory([dict(mass_kg=c['mass_kg'],energy_j=model.state['energy'][c['node']],cp=cfg['cp'][c['node']],latent=cfg['latent'][c['node']]) for c in cfg['phase_cells']],cfg['solidus'],cfg['liquidus'])
    delta=inventory(last)-inventory(first);net=sum(s['last_ledger']['phase']['new_phase_solid_kg']-s['last_ledger']['phase']['mush_remelted_kg'] for s in rows)
    assert np.isclose(delta,net,rtol=1e-10)
    m,node=freezing_model();E0=sum(m.state['energy']);m.advance(25.);birth=json.loads(encoded(m.state['last_ledger']['phase']));born=json.loads(encoded(m.state['born_material']))
    dest=Path('/workspace/simplified-genesis-phase-synthetic-01');dest.mkdir(exist_ok=False);m.save(dest/'birth.json');restored=SimplifiedGenesis.load(dest/'birth.json');m.advance(25.);restored.advance(25.);assert encoded(m.state)==encoded(restored.state)
    c=dict(id='material',face=0,layer=0,q_lo=.1,q_hi=.2,mass_kg=3e12,energy_j=3e12*(1200*1600+4e5/3),cp=1200.,latent=4e5)
    children=subdivide_uniform_cell(c,.37);regrid=solid_inventory(children,1400,2000)-solid_inventory([c],1400,2000)
    report=dict(status='CONSERVATIVE_PHASE_AND_MECHANICAL_BIRTH_PASS',real=dict(intervals=3,years=75,new_phase_solid_kg=net,independent_phase_inventory_change_kg=delta,inventory_roundoff_kg=delta-net,new_load_bearing_cells=len(last.state.get('born_material',[])),unchanged_vs_previous_pilot_fields=fields,damage_difference_from_previous_pilot=damage_roundoff,owners=len(last.state['omega']),thermal_mass_kg=float(sum(last.mass)),phase_cells=len(cfg['phase_cells']),min_remaining_cell_temperature_k=float(min(last.temperature(last.state['energy'])[x['node']] for x in cfg['phase_cells']))),synthetic=dict(birth= birth,zero_memory_at_birth=all(x['memory']==[0.]*6 for x in born),memory_norm_after_next_cooling=float(np.linalg.norm(m.state['born_material'][0]['memory'])),restart_next_step_exact=True,repartition_phase_mass_change_kg=regrid,whole_balance_j=m.state['last_ledger']['whole_balance_j'],whole_allowance_j=m.state['last_ledger']['whole_allowance_j']),seconds=time.perf_counter()-start,new_real_intervals=0,artifact_hashes={p.name:digest(p.read_bytes()) for p in OUT.glob('*.json')})
    for name,pin in cfg['source_pins'].items():assert digest(Path(name).read_bytes())==pin
    with (ROOT/'analysis/simplified_genesis_validation/phase_result.json').open('xb') as f:f.write(encoded(report))
    (ROOT/'analysis/simplified_genesis_validation/phase_pilot_report.json').write_bytes((OUT/'report.json').read_bytes());print(json.dumps(report,indent=2))

if __name__=='__main__':main()
