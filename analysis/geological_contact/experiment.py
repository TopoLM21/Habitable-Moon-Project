"""Bounded common-start contact/control experiments; never resets saved time."""
import argparse,json,math,time
from pathlib import Path
from tectonics.unified_geological_genesis import UnifiedGeologicalGenesis as U
from tectonics.genesis_transfer_bridge import encoded
from analysis.geological_continuation.run import snap
from analysis.genesis_geological_handoff.experiment import exact_saved_state


def balances(e):
    r=e.record;m=math.fsum(r['thermal_inventory']['mass_kg']);h=math.fsum(r['thermal_inventory']['energy_j'])
    return dict(mass_relative_residual=(m-r['initial_total_mass_kg'])/r['initial_total_mass_kg'],
        H_relative_residual=(h+r['cumulative_exported_heat_j']-r['initial_H_j'])/r['initial_H_j'],
        water_mass_kg=r['water_mass_kg'],water_volume_km3=e.checkpoint.hydrosphere.water_volume_km3)


def run(output,source,dt,duration,attachment,displacement):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();e=U.load(source);e.record['contact_diagnostics']=True
    e.record['gap_attachment']=None if attachment=='legacy' else 'adjacent_front'
    e.record['continental_displacement']={'off':None,'on':'retire_displaced_basalt','area':'retire_added_continental_area'}[displacement]
    initial=snap(e);first=len(e.record['steps']);events=len(e.record['events']);history=[initial];error=None
    for i in range(round(duration/dt)):
        if time.monotonic()-start>240:error='bounded wall limit240s';break
        try:e.step(dt)
        except Exception as exc:error=repr(exc);break
        history.append(snap(e))
    e.save(root/'accepted')
    rows=e.record['steps'][first:]
    result=dict(source=source,dt=dt,duration=duration,attachment=attachment,displacement=displacement,
        error=error,elapsed_seconds=time.monotonic()-start,initial=initial,history=history,final=snap(e),
        balances=balances(e),steps=rows,events=e.record['events'][events:])
    if error is None:
        # Independent load and continuation from the real final checkpoint.
        resumed=U.load(root/'accepted');e.step(dt);resumed.step(dt)
        exact_saved_state(e.save(root/'restart_direct'),resumed.save(root/'restart_loaded'))
        result['restart_exact']=True
    result['total_seconds_including_restart']=time.monotonic()-start
    (root/'result.json').write_bytes(encoded(result));print(json.dumps({k:v for k,v in result.items() if k not in ('history','steps','events')},indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--source',required=True);p.add_argument('--dt',type=float,default=1);p.add_argument('--duration',type=float,default=40);p.add_argument('--attachment',choices=['legacy','adjacent'],default='legacy');p.add_argument('--displacement',choices=['off','on','area'],required=True)
    a=p.parse_args();run(a.output,a.source,a.dt,a.duration,a.attachment,a.displacement)
