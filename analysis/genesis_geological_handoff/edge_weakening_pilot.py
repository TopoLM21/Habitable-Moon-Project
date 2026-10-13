"""One bounded comparison: same early source, 20x25 years, no tuning/retries."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,time
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.experimental_edge_weakening import MODE,scale
from tectonics.genesis_geological_thermal import ThermalGeologicalContinuation
from tectonics.genesis_transfer_bridge import encoded
from .nodal_retirement_check import SOURCE,OPTION

ROOT=Path(__file__).parent
OUTPUT=Path('/workspace/genesis-edge-weakening-pilot-01')


def main():
    start=time.monotonic();blob=SOURCE.read_bytes();pin=hashlib.sha256(blob).hexdigest()
    assert pin=='94fbbb3cb88a15147d33ab04c8f791470dd92e3d73cf8cba4afc0f361cd583fc'
    assert not OUTPUT.exists(),'Single pilot: output already exists; do not retry automatically'
    controls=json.loads((ROOT/'edge_weakening_controls.json').read_text());assert controls['local']['restart_exact'] and controls['local']['inventory_exact']
    baseline=json.loads((ROOT/'nodal_retirement_result.json').read_text())
    prior_path=Path(baseline['pilot']['output'])/'checkpoint.json';assert hashlib.sha256(prior_path.read_bytes()).hexdigest()==baseline['pilot']['checkpoint_sha256']
    source=SimplifiedGenesis.load(SOURCE);c=deepcopy(source.config);c['experimental_geological_edge_fatigue']={**OPTION,'mode':MODE}
    m=SimplifiedGenesis(c,source.state);OUTPUT.mkdir();rows=[];status='COMPLETED_SINGLE_SHELL';last=None
    for _ in range(20):
        if time.monotonic()-start>120:status='BUDGET_STOP';break
        before=deepcopy(m.state);clock=time.monotonic()
        try:m.advance(25.)
        except ValueError as e:
            assert encoded(m.state)==encoded(before);status='GUARD_STOP: '+str(e);break
        last=before;s=m.state;L=s['last_ledger'];f=s['experimental_edge_fatigue']
        deltaH=np.sum(np.array(s['energy'],np.longdouble)-np.array(before['energy'],np.longdouble));deltaR=s['experimental_edge_fracture_reserve_j']-before.get('experimental_edge_fracture_reserve_j',0.)
        independent=deltaH+L['elastic_after']-L['elastic_before']+L['kinetic_after']-L['kinetic_before']+deltaR-L['source_energy']-L['external_work']+L['numerical_memory']+L['numerical_kinetic']
        assert abs(independent)<=L['whole_allowance_j'];assert L['edge_damage_heat_j']==0
        rows.append(dict(time_myr=s['time_myr'],seconds=time.monotonic()-clock,
            max_edge_damage=max(f['damage']),saturated=sum(f['failed_edges']),max_shear_before_edge_update_pa=max(f['shear_traction_pa']),reason=f['reason'],
            reserve_j=s['experimental_edge_fracture_reserve_j'],release_j=L['edge_fracture_release_j'],fracture_heat_j=L['edge_damage_heat_j'],
            energy_fraction=float(abs(independent)/L['whole_allowance_j']),old_damage_max=max(s['damage']),
            effective_scale_min=float(min(scale(m,s['damage']))),newborn=len(s['born_material'])))
        if f['partition_owner'] is not None:
            try:
                g=ThermalGeologicalContinuation.from_source(m,experimental_nodal_retirement=True,fracture_edges=np.array(f['failed_edges']))
                g.save(OUTPUT/'geological-handoff');status='CERTIFIED_GEOLOGICAL_HANDOFF'
            except ValueError as e:status='HANDOFF_GUARD_STOP: '+str(e)
            break
    m.save(OUTPUT/'checkpoint.json');restart=None
    if last is not None:
        replay=SimplifiedGenesis(c,last);replay.advance(25.);restart=encoded(replay.state)==encoded(m.state);assert restart
    assert SOURCE.read_bytes()==blob
    assert hashlib.sha256(prior_path.read_bytes()).hexdigest()==baseline['pilot']['checkpoint_sha256']
    result=dict(status=status,source_sha256=pin,source_unchanged=True,baseline_unchanged=True,
        start_time_myr=source.state['time_myr'],end_time_myr=m.state['time_myr'],accepted_steps=len(rows),dt_years=25.,
        parameters=c['experimental_geological_edge_fatigue'],output=str(OUTPUT),checkpoint_sha256=hashlib.sha256((OUTPUT/'checkpoint.json').read_bytes()).hexdigest(),
        domains=len(set(m.state['owner'])),mass_array_exact=m.config['mass']==source.config['mass'],restart_exact=restart,seconds=time.monotonic()-start,rows=rows,
        baseline=dict(end_time_myr=baseline['pilot']['end_time_myr'],domains=baseline['pilot']['domains'],last=baseline['pilot']['rows'][-1]))
    (ROOT/'edge_weakening_pilot.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
