"""Independent replay and bounded one-year temporal pair on the nested mesh."""
from pathlib import Path
import json
import time
import numpy as np
from tectonics.experimental_angular_coupled import AngularCoupled
from tectonics.genesis_transfer_bridge import encoded, digest
from .coupled_spatial_check import OUT as SPATIAL, field, compare
from .edge_spatial_audit import ROOT

OUT = Path('/workspace/genesis-coupled-spatial-pilot-01')


def main():
    if OUT.exists():
        raise FileExistsError('Preserve previous pilot')
    prior = json.loads((ROOT/'coupled_spatial_check.json').read_text())
    assert prior['spatial_gate_passed']
    assert all(x['births'] == 0 and x['whole_allowance_fraction'] <= 1
               for name in ('coarse_full', 'fine_full') for x in prior['runs'][name]['rows'])
    start = time.monotonic()
    original_hashes = {p: digest(p.read_bytes()) for p in SPATIAL.glob('*.json')}
    # Replay from the independently serialized initial state, not two copies of
    # one loaded final state. This also checks construction versus reconstruction.
    direct = AngularCoupled.load(SPATIAL/'fine_initial.json')
    direct.advance(.0625)
    direct.advance(.0625)
    resumed = AngularCoupled.load(SPATIAL/'fine_final.json')
    replay_exact = encoded(direct.model.state) == encoded(resumed.model.state)
    assert replay_exact
    direct.advance(.0625)
    resumed.advance(.0625)
    restart_exact = encoded(direct.model.state) == encoded(resumed.model.state)
    assert restart_exact
    print(json.dumps(dict(replay_exact=replay_exact, restart_exact=restart_exact)), flush=True)
    OUT.mkdir()
    runs = {}; fields = {}; deadline = time.monotonic()+180
    for name, dt in [('dt_0125', .125), ('dt_00625', .0625)]:
        a = AngularCoupled.load(SPATIAL/'fine_initial.json')
        rows = []; trace = {}
        for i in range(round(1/dt)):
            if time.monotonic() > deadline:
                raise TimeoutError('180second pilot budget')
            tick = time.monotonic(); L = a.advance(dt); m = a.model
            rows.append(dict(years=(i+1)*dt, seconds=time.monotonic()-tick,
                whole_balance_j=L['whole_balance_j'], whole_allowance_j=L['whole_allowance_j'],
                whole_allowance_fraction=abs(L['whole_balance_j'])/L['whole_allowance_j'],
                mechanical_balance_relative=L['moment_balance_relative'], momentum_relative=L['momentum_relative'],
                births=len(m.state.get('born_material', [])),
                saturated=sum(m.state['experimental_edge_fatigue']['failed_edges']),
                certificate=m.state['experimental_edge_fatigue']['reason']))
            if ((i+1)*dt*8).is_integer():
                trace[str((i+1)*dt)] = field(m)
            assert rows[-1]['whole_allowance_fraction'] <= 1
            if rows[-1]['certificate'] != 'NOT_SEPARATING':
                raise RuntimeError('Partition reached: bounded pilot stopped')
        a.save(OUT/(name+'_final.json'))
        runs[name] = rows; fields[name] = trace
        print(json.dumps(dict(run=name, solve_seconds=sum(r['seconds'] for r in rows))), flush=True)
    differences = {t: compare(fields['dt_0125'][t], fields['dt_00625'][t], a.model.G[:,0,0])
                   for t in fields['dt_0125']}
    initial = field(AngularCoupled.load(SPATIAL/'fine_initial.json').model)
    end = fields['dt_00625']['1.0']
    assert all(digest(p.read_bytes()) == h for p,h in original_hashes.items())
    np.savez_compressed(OUT/'fields.npz', **{f'{name}_{t}_{k}':v for name,trace in fields.items() for t,f in trace.items() for k,v in f.items()})
    result = dict(spatial_result_sha256=digest((ROOT/'coupled_spatial_check.json').read_bytes()),
        replay_from_initial_exact=replay_exact, independent_restart_exact=restart_exact,
        duration_years=1., faces=320, steps=[.125,.0625], runs=runs, temporal_comparisons=differences,
        evolution=dict(temperature_max_change_k=float(abs(end['T']-initial['T']).max()),
            old_damage_max_change=float(abs(end['oldD']-initial['oldD']).max()),
            face_damage_max=float(end['faceD'].max())),
        seconds=time.monotonic()-start, output=str(OUT),
        files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in OUT.iterdir()})
    (ROOT/'coupled_spatial_pilot.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(seconds=result['seconds'], evolution=result['evolution'], final_difference=differences['1.0'])),flush=True)


if __name__ == '__main__':
    main()
