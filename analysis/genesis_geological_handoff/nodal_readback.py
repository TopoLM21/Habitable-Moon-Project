"""Summarize saved nodal pilot and old observations; never advance either model."""
import hashlib,json
from pathlib import Path
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.genesis_shell import principal_tensile
from tectonics.genesis_transfer_bridge import encoded


def readback(root,out):
    root=Path(root);out=Path(out);r=json.loads((root/'report.json').read_bytes())
    new=list(map(json.loads,(root/'history.jsonl').read_text().splitlines()))
    old=list(map(json.loads,Path('/workspace/genesis-real-progression-01/history.jsonl').read_text().splitlines()))
    old+=list(map(json.loads,Path('/workspace/genesis-real-condensation-01/history.jsonl').read_text().splitlines()))
    bytime={round(x['time_myr'],9):x for x in old}
    m=SimplifiedGenesis.load(root/'checkpoint.json');src=SimplifiedGenesis.load(r['source_file'])
    assert m.config==src.config and m.state['time_myr']==r['final']['time_myr']
    assert hashlib.sha256((root/'checkpoint.json').read_bytes()).hexdigest()==r['final_checkpoint_sha256']
    same=SimplifiedGenesis.load('/workspace/genesis-real-progression-01/checkpoint.json')
    T=same.temperature(same.state['energy']);D=np.asarray(same.state['damage'])
    stress=principal_tensile(same.constitutive(T,D,np.asarray(same.state['memory']),T,0.)[1])
    row=next(x for x in new if abs(x['time_myr']-same.state['time_myr'])<1e-10)
    common=dict(time_myr=row['time_myr'],old=dict(radial_strain_mean=0.,damage_mean=float(D.mean()),damage_span=float(D.max()-D.min()),tensile_mean_pa=float(stress.mean()),tensile_std_pa=float(stress.std())),new={k:row[k] for k in ('radial_strain_mean','damage_mean','tensile_mean_pa','tensile_std_pa')})
    common['new']['damage_span']=row['damage_max']-row['damage_min']
    oldfinal=bytime[round(r['final']['time_myr'],9)]
    fields=('damage_mean','damage_min','damage_max','solid_mean_k','water_k','born_cells')
    timing=dict(new_median_step_seconds=float(np.median([x['step_seconds'] for x in new[1:]])),old_median_step_seconds=float(np.median([x['step_seconds'] for x in old if 'step_seconds' in x])),caveat='Saved runs on same environment at different times; regression tests overlapped part of this pilot, not an isolated benchmark.')
    r.update(common_time_comparison=common,same_final_time_old_observations={k:oldfinal[k] for k in fields},cost=timing,kinetic_import=m.config['nodal']['kinetic_import'],controls=json.loads((out/'nodal_controls_result.json').read_bytes()),
        tests=dict(nodal_and_moment='15 passed in 3.47s',legacy_and_handoff='34 passed in 24.51s'),history_sha256=hashlib.sha256((root/'history.jsonl').read_bytes()).hexdigest(),checkpoint_load_verified=True,
        max_nodal_momentum_relative=max(x['nodal_momentum_relative'] for x in new[1:]),
        code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('tectonics/simplified_genesis_nodal.py'),Path('analysis/genesis_aggregate_coupled/core.py')]})
    (out/'nodal_pilot_result.json').write_bytes(encoded(r))
    (out/'nodal_history_sample.jsonl').write_text(''.join(json.dumps(x)+'\n' for i,x in enumerate(new) if i%50==0 or i==len(new)-1))
    print(json.dumps({k:r[k] for k in ['status','completed_real_steps','elapsed_seconds','final','common_time_comparison','same_final_time_old_observations','cost','final_checkpoint_sha256','last_real_step_restart_exact']},indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    readback(args.root,args.output)
