"""Bounded real-source pair; exclusive output, no publication or source edits."""
import argparse,json,math,time
from pathlib import Path
from collections import Counter
import numpy as np
from tectonics.unified_geological_genesis import UnifiedGeologicalGenesis
from tectonics.genesis_geological_handoff import sha
from tectonics.genesis_transfer_bridge import encoded
from tectonics.dynamics import angular_velocity_vectors
from tectonics.mesh import connected_components
from analysis.genesis_geological_handoff.experiment import exact_saved_state

SOURCE_SHA='a2e18177f1737fffc20d4edcf8e86a0f755b08f4891f1f8d6ec74756aa9cfdfc'

def metrics(e):
    c=e.checkpoint;r=e.record;w=angular_velocity_vectors(c.system)
    v=np.linalg.norm(np.cross(w[c.state.cell_plate],e.mesh.centroids*r['radius_km']),axis=1)
    a=e.mesh.physical_cell_areas_km2(r['radius_km'])
    assert all(len(connected_components(np.flatnonzero(c.state.cell_plate==i),e.mesh.neighbors))==1 for i in range(len(c.system.plates)))
    inv=r['thermal_inventory'];M=math.fsum(inv['mass_kg']);H=math.fsum(inv['energy_j'])
    return dict(time_myr=c.state.time_myr,plates=len(c.system.plates),mean_speed_km_per_myr=float(np.average(v,weights=a)),
        max_speed_km_per_myr=float(max(v)),moving_area_fraction=float(a[v>1.].sum()/a.sum()),
        largest_plate_fraction=float(np.bincount(c.state.cell_plate,weights=a).max()/a.sum()),
        mantle_temperature_k=c.thermal.mantle_temperature_k,water=e.water_diagnostics(),
        mass_relative_residual=(M-r['initial_total_mass_kg'])/r['initial_total_mass_kg'],
        H_relative_residual=(H+r['cumulative_exported_heat_j']-r['initial_H_j'])/r['initial_H_j'],
        cumulative_exported_heat_j=r['cumulative_exported_heat_j'],transport_commits=c.transport_state.cumulative_commit_count,
        max_hold_age_myr=float(max(c.transport_state.hold_age_myr)),connected=True,
        continental_volume_km3=float(c.state.continental_volume_km3.sum()))

def run(source,root):
    start=time.monotonic();root=Path(root);root.mkdir(parents=True,exist_ok=False)
    assert sha(Path(source).read_bytes())==SOURCE_SHA
    repository=Path(__file__).resolve().parents[2]
    code_paths=['tectonics/unified_geological_genesis.py','tectonics/genesis_implicit_enthalpy.py',
        'tectonics/genesis_geological_handoff.py','tectonics/dynamics.py','tectonics/lithosphere.py',
        'tectonics/topology.py','tectonics/transport.py','tectonics/topography.py','tectonics/hydrosphere.py',
        'analysis/unified_geological_genesis/experiment.py']
    result=dict(source_sha256=SOURCE_SHA,duration_myr=160.,runs=[],code_sha256={p:sha((repository/p).read_bytes()) for p in code_paths})
    for dt in (1.,.5):
        ts=time.monotonic();e=UnifiedGeologicalGenesis.from_checkpoint(source);initial=metrics(e)
        arm=root/str(dt);arm.mkdir();e.save(arm/'initial');history=[initial]
        for i in range(round(160/dt)):
            if time.monotonic()-start>180:raise TimeoutError('Bounded 180s scenario budget')
            e.step(dt)
            if (i+1)%round(20/dt)==0:history.append(metrics(e))
            if i+1==round(80/dt):
                cp=e.save(arm/'mid');resumed=UnifiedGeologicalGenesis.load(cp)
                # Same next macro step, including event controller and every native substate.
                resumed.step(dt)
            elif i+1==round(80/dt)+1:
                exact_saved_state(e.save(arm/'direct_restart_check'),resumed.save(arm/'resumed_restart_check'))
        e.save(arm/'final');rows=e.record['steps'];final=metrics(e)
        summary=dict(dt_myr=dt,initial=initial,final=final,history=history,restart_exact=True,
            accepted_substeps=len(rows),events=Counter(x['kind'] for x in e.record['events']),
            event_times_myr=[x['time_myr'] for x in e.record['events']],
            fracture_band_cell_counts=[len(x['cut']) for x in e.record['events'] if x['kind']=='geological_weak_band_split'],
            last_fracture_time_myr=max((x['time_myr'] for x in e.record['events'] if x['kind']=='geological_weak_band_split'),default=None),
            transported_winning_cells=sum(x['transported_winning_cells'] for x in rows),
            generated_volume_km3=sum(x['material']['generated_volume_km3'] for x in rows),
            recycled_volume_km3=sum(x['material']['recycled_volume_km3'] for x in rows),
            max_material_residual_km3=max(abs(x['material']['volume_residual_km3']) for x in rows),
            max_water_relative_error=max(abs(x['water']['relative_volume_error']) for x in rows),
            max_thermal_residual_over_allowance=max(abs(x['thermal']['global_residual_j'])/x['thermal']['allowance_j'] for x in rows),
            max_native_tidal_strain=max(x['max_tidal_strain'] for x in e.checkpoint.lithosphere_rows),
            elapsed_seconds=time.monotonic()-ts)
        result['runs'].append(summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('history','initial')},indent=2),flush=True)
        (root/'result.json').write_bytes(encoded(result))
    a,b=result['runs'];comparison={}
    for key in ('generated_volume_km3','recycled_volume_km3','transported_winning_cells'):
        comparison[key]=dict(coarse=a[key],fine=b[key],relative_difference=abs(a[key]-b[key])/max(abs(b[key]),1e-30))
    for key in ('plates','mean_speed_km_per_myr','mantle_temperature_k','largest_plate_fraction'):
        x,y=a['final'][key],b['final'][key];comparison[key]=dict(coarse=x,fine=y,relative_difference=abs(x-y)/max(abs(y),1e-30))
    result.update(comparison=comparison,source_unchanged=sha(Path(source).read_bytes())==SOURCE_SHA,elapsed_seconds=time.monotonic()-start)
    (root/'result.json').write_bytes(encoded(result));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');args=p.parse_args();run(args.source,args.output)
