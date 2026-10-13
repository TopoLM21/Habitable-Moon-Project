"""Read-only comparison of the single moving-basal pilot with saved nodal history."""
import hashlib,json
from pathlib import Path
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.simplified_genesis_basal import geometry
from tectonics.genesis_transfer_bridge import encoded


def run():
    out=Path(__file__).parent;root=Path('/workspace/genesis-moving-basal-pilot-01')
    r=json.loads((root/'report.json').read_bytes())
    rows=list(map(json.loads,(root/'history.jsonl').read_text().splitlines()))
    old=list(map(json.loads,Path('/workspace/genesis-nodal-pilot-01/history.jsonl').read_text().splitlines()))
    bytime={round(x['time_myr'],9):x for x in old}
    model=SimplifiedGenesis.load(root/'checkpoint.json')
    source=SimplifiedGenesis.load(r['source_file']);prior=SimplifiedGenesis.load('/workspace/genesis-nodal-source-01.json')
    assert encoded(source.state)==encoded(prior.state)
    c=json.loads(json.dumps(source.config));del c['nodal']['moving_basal'];assert c==prior.config
    assert model.config==source.config
    assert hashlib.sha256((root/'checkpoint.json').read_bytes()).hexdigest()==r['final_checkpoint_sha256']
    fields=('radial_strain_mean','tensile_mean_pa','tensile_std_pa','tensile_max_pa','damage_min','damage_max','damage_mean','born_cells','solid_mean_k','whole_energy_residual_j')
    previous=bytime[round(r['final']['time_myr'],9)]
    r['same_time_comparison']=dict(time_myr=r['final']['time_myr'],fixed_basal={k:previous[k] for k in fields},moving_basal={k:r['final'][k] for k in fields})
    g=geometry(model)
    r.update(current_endpoint_basal_geometry=dict(area_m2=float(g['area'].sum()),depth_min_m=float(g['depth'].min()),depth_max_m=float(g['depth'].max()),depth_mean_m=float(g['depth'].mean())),
        mantle_velocity_source=model.config['nodal']['moving_basal']['mantle_velocity_source'],beta_pa_s_m=1e14,
        inherited_initial_state_exact=True,only_config_addition='nodal.moving_basal',
        last_basal_ledger=model.state['last_ledger']['basal'],last_whole_balance_over_allowance=abs(model.state['last_ledger']['whole_balance_j'])/model.state['last_ledger']['whole_allowance_j'],
        max_nodal_momentum_relative=max(x['nodal_momentum_relative'] for x in rows[1:]),
        total_relative_drag_heat_j=sum(x['basal_heat_j'] for x in rows[1:]),
        median_step_seconds=float(np.median([x['step_seconds'] for x in rows[1:]])),
        prior_median_step_seconds=float(np.median([x['step_seconds'] for x in old[1:len(rows)]])),
        cost_caveat='Saved runs, not an isolated benchmark; tests overlapped part of both pilots.',
        tests=dict(addressed='14 passed in 6.44s',regression='13 passed in 10.39s'),
        history_sha256=hashlib.sha256((root/'history.jsonl').read_bytes()).hexdigest(),
        executed_code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('tectonics/simplified_genesis_basal.py'),Path('tectonics/simplified_genesis_nodal.py')]})
    (out/'moving_basal_result.json').write_bytes(encoded(r))
    (out/'moving_basal_history_sample.jsonl').write_text(''.join(json.dumps(x)+'\n' for i,x in enumerate(rows) if i%50==0 or i==len(rows)-1))
    print(json.dumps({k:r[k] for k in ['status','completed_real_steps','elapsed_seconds','final_checkpoint_sha256','same_time_comparison','current_endpoint_basal_geometry','median_step_seconds','prior_median_step_seconds','max_whole_energy_residual_over_allowance','phase_audit_all_passed','last_real_step_restart_exact']},indent=2))


if __name__=='__main__':run()
