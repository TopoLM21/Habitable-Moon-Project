"""Verify approved coarse handoff and run a bounded real edge-incubation pilot."""
import json,time,hashlib
from copy import deepcopy
from pathlib import Path
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.genesis_geological_thermal import ThermalGeologicalContinuation,eligibility
from tectonics.genesis_transfer_bridge import encoded

ROOT=Path(__file__).parent
SOURCE=Path('/workspace/genesis-moving-basal-source-01.json')
OUTPUT=Path('/workspace/genesis-edge-retirement-pilot-01')
OPTION={'shear_strength_pa':1e6,'damage_time_years':100.,'mode':'incubation_then_geological_retirement'}


def main():
    start=time.monotonic();blob=SOURCE.read_bytes();m=SimplifiedGenesis.load(SOURCE);original=encoded(dict(config=m.config,state=m.state))
    assert not OUTPUT.exists(),'Use a fresh output directory'
    OUTPUT.mkdir()
    # Manufactured partition ONLY for adapter verification; never assigned in real pilot.
    sign=m.mesh.centroids[:,2]>0;edges=np.array([sign[f]!=sign[g] for f,g,_,_ in m.mesh.shared_edges])
    g=ThermalGeologicalContinuation.from_source(m,experimental_nodal_retirement=True,fracture_edges=edges)
    assert encoded(dict(config=m.config,state=m.state))==original
    np.testing.assert_array_equal(g.record['thermal_inventory']['mass_kg'],m.config['mass'])
    np.testing.assert_array_equal(g.record['thermal_inventory']['energy_j'],m.state['energy'])
    assert g.record['mapping']['energy']['mapping_heat_j']==0
    assert g.record['mapping']['Maxwellmemory']['heat_j']==0
    mapping=deepcopy(g.record['mapping']);g.step(25e-6);g.save(OUTPUT/'manufactured-mature-restart')
    resumed=ThermalGeologicalContinuation.load(OUTPUT/'manufactured-mature-restart')
    g.step(25e-6);resumed.step(25e-6)
    from analysis.genesis_geological_handoff.experiment import exact_saved_state
    exact_saved_state(g.save(OUTPUT/'manufactured-direct'),resumed.save(OUTPUT/'manufactured-resumed'))
    assert any('Shared nodal' in x for x in eligibility(m)['reasons'])
    bad=edges.copy();bad[np.flatnonzero(bad)[0]]=False
    try:ThermalGeologicalContinuation.from_source(m,experimental_nodal_retirement=True,fracture_edges=bad)
    except ValueError as e:assert str(e)=='NOT_SEPARATING'
    else:raise AssertionError('Open seam accepted')
    c=deepcopy(m.config);c['experimental_geological_edge_fatigue']=OPTION
    pilot=SimplifiedGenesis(c,m.state);rows=[];before_last=None;status='COMPLETED_SINGLE_SHELL'
    benchmark=None
    for i in range(20):
        if time.monotonic()-start>120:status='BUDGET_STOP';break
        before_last=deepcopy(pilot.state);clock=time.monotonic();pilot.advance(25.);elapsed=time.monotonic()-clock
        if benchmark is None:benchmark=elapsed
        x=pilot.state['experimental_edge_fatigue'];L=pilot.state['last_ledger']
        rows.append(dict(time_myr=pilot.state['time_myr'],seconds=elapsed,max_edge_damage=max(x['damage']),
            failed_edges=sum(x['failed_edges']),max_shear_pa=max(x['shear_traction_pa']),reason=x['reason'],
            newborn=len(pilot.state.get('born_material',[])),whole_energy_j=L['whole_balance_j'],allowance_j=L['whole_allowance_j']))
        if x['partition_owner'] is not None:
            try:
                actual=ThermalGeologicalContinuation.from_source(pilot,experimental_nodal_retirement=True,fracture_edges=np.array(x['failed_edges']))
                actual.save(OUTPUT/'actual-geological-handoff');actual.step(25e-6);actual.save(OUTPUT/'actual-geological-step')
                status='REAL_GEOLOGICAL_HANDOFF'
            except ValueError as e:status='HANDOFF_CANDIDATE_BLOCKED: '+str(e)
            break
    pilot.save(OUTPUT/'checkpoint.json')
    replay=SimplifiedGenesis(c,before_last);replay.advance(25.)
    assert encoded(replay.state)==encoded(pilot.state)
    # Early source's ordinary/default first step remains an independent control.
    control=SimplifiedGenesis.load(SOURCE);control.advance(25.)
    opted=SimplifiedGenesis(c,m.state);opted.advance(25.)
    for key in ('energy','memory','nodal_velocity','damage'):
        assert control.state[key]==opted.state[key]
    assert SOURCE.read_bytes()==blob
    result=dict(status=status,source_sha256=hashlib.sha256(blob).hexdigest(),source_unchanged=True,
        manufactured_adapter=dict(mass_H_exact=True,no_mapping_heat=True,source_unchanged=True,restart_exact=True,
            open_seam_rejected=True,default_guard_retained=True,mapping=mapping,native_steps=g.record['steps']),
        pilot=dict(output=str(OUTPUT),start_time_myr=m.state['time_myr'],end_time_myr=pilot.state['time_myr'],
            steps=len(rows),years=(pilot.state['time_myr']-m.state['time_myr'])*1e6,domains=len(set(pilot.state['owner'])),
            last_step_restart_exact=True,first_step_control_identical=True,first_step_seconds=benchmark,
            checkpoint_sha256=hashlib.sha256((OUTPUT/'checkpoint.json').read_bytes()).hexdigest(),rows=rows),seconds=time.monotonic()-start)
    (ROOT/'nodal_retirement_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**result,'manufactured_adapter':{k:v for k,v in result['manufactured_adapter'].items() if k!='native_steps'},'pilot':{k:v for k,v in result['pilot'].items() if k!='rows'}},indent=2))


if __name__=='__main__':main()
