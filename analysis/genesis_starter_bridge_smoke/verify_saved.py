"""Read/save/reload verification only. Does not execute another time step."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from tectonics.checkpoint import save_checkpoint
from tectonics.genesis_starter_continuation import load_starter_source, _load_cp, YoungWorldCoupling
from tectonics.genesis_starter_fracture import YoungShellFracture
from tectonics.genesis_transfer_bridge import assess_starter_entry
from tectonics.simulation import load_config

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'analysis/genesis_origin_starter_validation/evidence'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def equal_npz(a,b):
    with np.load(a,allow_pickle=False) as x, np.load(b,allow_pickle=False) as y:
        assert set(x.files)==set(y.files)
        for key in x.files: assert np.array_equal(x[key],y[key]), key
        return len(x.files)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args(); start=time.perf_counter(); root=args.directory; step=root/'step'
    pins={row['path']:row['raw_sha256'] for row in json.loads((SOURCE/'package_manifest.json').read_bytes())['files']}
    verified={}
    for name in ['partition/starter_checkpoint.npz','partition/parameters.json','partition/partial_checkpoint.npz',
                 'partition/molten_checkpoint.npz','partition/report.json']:
        assert sha(SOURCE/name)==pins[name], name
        verified[name]=pins[name]
    model,origin,_=load_starter_source(SOURCE/'partition/starter_checkpoint.npz')
    partial=model.load_state(SOURCE/'partition/partial_checkpoint.npz')
    partial_gate=assess_starter_entry(model,partial)
    assert partial_gate['admitted'] and partial_gate['owner_count']==1
    molten=model.load_state(SOURCE/'partition/molten_checkpoint.npz')
    assert len(molten.system.plates)==1 and not molten.events and not np.any(molten.damage)
    assert origin.stopped_reason=='first_partition' and len(origin.system.plates)==2
    assert len(origin.events)==1 and origin.events[0]['kind']=='split'
    assert np.all(origin.damage[origin.split_band]>=model.parameters.rupture_damage)
    report=json.loads((step/'continuation.json').read_bytes())
    assert report['status']=='completed' and all(report['checks'].values())
    for name,pin in report['checkpoint_sha256'].items(): assert sha(step/name)==pin
    m,s,_=load_starter_source(step/'young_context/starter_checkpoint.npz')
    cfg=load_config(step/'mature_config.yaml'); cp=_load_cp(step/'mature_checkpoint',cfg)
    fracture=YoungShellFracture.load(m,step/'young_context/fracture_memory.npz')
    out=root/'readback'; out.mkdir(exist_ok=False)
    m.save_state(out/'starter.npz',s); fracture.save(out/'fracture.npz'); save_checkpoint(out/'mature',cp)
    counts=dict(starter=equal_npz(step/'young_context/starter_checkpoint.npz',out/'starter.npz'),
                fracture=equal_npz(step/'young_context/fracture_memory.npz',out/'fracture.npz'),
                mature=equal_npz(step/'mature_checkpoint/state.npz',out/'mature/state.npz'))
    assert json.loads((step/'mature_checkpoint/meta.json').read_bytes())==json.loads((out/'mature/meta.json').read_bytes())
    restored=m.load_state(out/'starter.npz'); f2=YoungShellFracture.load(m,out/'fracture.npz'); cp2=_load_cp(out/'mature',cfg)
    coupling=YoungWorldCoupling(m,restored,cfg,cp2,report['history'],f2)
    assert coupling.source_state.solidification_history==s.solidification_history
    assert coupling.fracture.time_myr==s.time_myr==cp2.state.time_myr
    assert np.array_equal(coupling.fracture.damage,cp2.state.tidal_damage)
    assert s.solidification_history.cohorts==origin.solidification_history.cohorts
    old=origin.solidification_history.intervals
    assert s.solidification_history.intervals[:len(old)]==old
    initial=_load_cp(step/'initial_import/mature_checkpoint',cfg)
    fields_changed=[]
    with np.load(step/'initial_import/mature_checkpoint/state.npz',allow_pickle=False) as a, np.load(step/'mature_checkpoint/state.npz',allow_pickle=False) as b:
        fields_changed=[k for k in a.files if k in b.files and not np.array_equal(a[k],b[k])]
    result=dict(status='PASS_OLD_STARTER_BRANCH_ONLY',one_new_mature_step=True,step_myr=report['step_myr'],
        start_time_myr=origin.time_myr,final_time_myr=s.time_myr, source_pins=verified,
        source_traction_pa=model.shell.convective_traction_pa,source_starter_seed=model.parameters.seed,
        source_partition_event=origin.events[0],partition_reexecuted=False,partial_starter_gate=partial_gate,
        volume_starter_gate=assess_starter_entry(None,dict(source_owner_ids=['one-body'])),
        all_runner_checks=report['checks'],material_ledger=report['material_ledger'],
        mean_speed_km_myr=report['final_mean_surface_speed_km_myr'],transport_commits=report['transport_commits'],
        new_fracture_events=report['young_fracture_events'],mature_changed_fields=fields_changed,
        restart_equal_npz_fields=counts,mature_metadata_equal=True,restart_coupling_constructed=True,
        original_birth_cohorts_and_interval_prefix_preserved=True,
        source_files_unchanged=all(sha(SOURCE/k)==v for k,v in verified.items()),
        continuation_sha256=sha(step/'continuation.json'),
        physical_handoff_certified=False,readback_seconds=time.perf_counter()-start,
        artifact_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
    with (root/'verification.json').open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
