"""Bounded fractional-interface experiment from a saved young shell."""
import argparse, time, json, math
from pathlib import Path
from collections import Counter
from tectonics.unified_geological_genesis import UnifiedGeologicalGenesis as U
from tectonics.genesis_transfer_bridge import encoded
from analysis.geological_continuation.run import snap
from analysis.geological_contact.experiment import balances
from analysis.genesis_geological_handoff.experiment import exact_saved_state

def run(output, dt, duration=1., fractional=True, source='/workspace/unified-geological-genesis-02/1.0/initial'):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();e=U.load(source)
    if fractional:e.enable_fractional_transport()
    else:
        e.enable_mature_cycle(continental_displacement='retire_added_continental_area');e.record['contact_diagnostics']=True
    initial=e.save(root/'initial');history=[snap(e)];error=None
    first=len(e.record['steps']);first_event=len(e.record['events'])
    initialM=math.fsum(e.record['thermal_inventory']['mass_kg']);initialH=math.fsum(e.record['thermal_inventory']['energy_j']);initialExport=e.record['cumulative_exported_heat_j']
    for i in range(round(duration/dt)):
        if time.monotonic()-start>240:error='240s wall budget';break
        try:e.step(dt)
        except Exception as exc:
            import traceback
            error=repr(exc);(root/'failure_trace.txt').write_text(traceback.format_exc())
            if hasattr(exc,'geometry_failure'):(root/'geometry_failure.json').write_text(json.dumps(exc.geometry_failure,indent=2))
            break
        history.append(snap(e));print('accepted',dt,i+1,time.monotonic()-start,flush=True)
    e.save(root/'accepted');restart=False
    if error is None and time.monotonic()-start<180:
        try:
            # Repeat from the serialized initial state to the same final time.
            resumed=U.load(initial)
            for i in range(round(duration/dt)):resumed.step(dt)
            exact_saved_state(root/'accepted',resumed.save(root/'replayed'));restart=True
        except Exception as exc:error='restart: '+repr(exc)
    result=dict(source=source,fractional=fractional,dt=dt,duration=duration,error=error,seconds=time.monotonic()-start,
        history=history,balances=balances(e),restart_exact=restart,
        incremental_mass_relative_error=(math.fsum(e.record['thermal_inventory']['mass_kg'])-initialM)/initialM,
        incremental_H_relative_error=(math.fsum(e.record['thermal_inventory']['energy_j'])-initialH+e.record['cumulative_exported_heat_j']-initialExport)/initialH,
        events=dict(Counter(x['kind'] for x in e.record['events'][first_event:])),steps=e.record['steps'][first:])
    (root/'result.json').write_bytes(encoded(result));print(json.dumps({k:v for k,v in result.items() if k not in ('history','steps')},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--dt',type=float,required=True);p.add_argument('--duration',type=float,default=1.)
    p.add_argument('--native',action='store_true');p.add_argument('--source',default='/workspace/unified-geological-genesis-02/1.0/initial')
    a=p.parse_args();run(a.output,a.dt,a.duration,not a.native,a.source)
