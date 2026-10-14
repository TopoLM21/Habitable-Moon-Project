"""Bounded 500 Myr paired continuation; all output directories exclusive."""
import argparse,json,math,time
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import numpy as np
from tectonics.unified_geological_genesis import UnifiedGeologicalGenesis
from tectonics.genesis_transfer_bridge import encoded
from tectonics.genesis_geological_handoff import sha
from tectonics.hydrosphere import diagnose_hydrosphere,HydrosphereParameters
from tectonics.topography import TopographyParameters
from analysis.unified_geological_genesis.experiment import metrics
from analysis.genesis_geological_handoff.experiment import exact_saved_state
from .diagnostics import snapshot

def snap(e):
    c=e.checkpoint;r=e.record;w=diagnose_hydrosphere(e.mesh,c.state,c.topo,c.hydrosphere,r['radius_km'],HydrosphereParameters(),TopographyParameters())
    return snapshot(e.mesh,c.state,c.system,c.topo,w,r['radius_km'])

def run(root,dt,duration=500.,resume=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    e=UnifiedGeologicalGenesis.load(resume or '/workspace/unified-geological-genesis-02/1.0/initial')
    if resume is None:e.enable_mature_cycle()
    saved=e.save(root/'initial');history=[snap(e)];last=0;resumed=UnifiedGeologicalGenesis.load(saved);resumed.step(dt)
    target=e.source['original_time_myr']+duration
    count=round((target-e.checkpoint.state.time_myr)/dt)
    try:
        for i in range(count):
            if time.monotonic()-start>300:raise TimeoutError('300s bounded scenario budget')
            e.step(dt);last=i+1
            if i==0:exact_saved_state(e.save(root/'direct_initial_restart'),resumed.save(root/'resumed_initial_restart'))
            if (i+1)%round(20/dt)==0:history.append(snap(e));print(dt,i+1,'Myr',e.checkpoint.state.time_myr,'seconds',time.monotonic()-start,flush=True)
            if abs(e.checkpoint.state.time_myr-(e.source['original_time_myr']+160))<1e-9:e.save(root/'at160')
            if i+1==count//2:saved=e.save(root/'mid');resumed=UnifiedGeologicalGenesis.load(saved);resumed.step(dt)
            elif i+1==count//2+1:exact_saved_state(e.save(root/'direct_restart'),resumed.save(root/'resumed_restart'))
    except Exception as error:
        e.save(root/'last_accepted');(root/'failure.json').write_text(json.dumps(dict(accepted_steps=last,time=e.checkpoint.state.time_myr,error=repr(error),elapsed=time.monotonic()-start),indent=2));raise
    e.save(root/'final');rows=e.record['steps'];g=[x['geometry'] for x in rows];r=e.record
    totalM=math.fsum(r['thermal_inventory']['mass_kg']);totalH=math.fsum(r['thermal_inventory']['energy_j'])
    result=dict(dt_myr=dt,duration_myr=duration,resumed_from=resume,history=history,final=snap(e),restart_exact=True,
        events=Counter(x['kind'] for x in r['events']),substeps=len(rows),elapsed_seconds=time.monotonic()-start,
        raw_extra_components=sum(x['raw_footprint_extra_components'] for x in g),post_overlap_extra_components=sum(x['post_overlap_extra_components'] for x in g),
        max_assignment_error_km=max(x['max_assignment_error_km'] for x in g),
        generated_volume_km3=math.fsum(x['material']['generated_volume_km3'] for x in rows),
        recycled_volume_km3=math.fsum(x['material']['recycled_volume_km3'] for x in rows),
        max_material_residual_km3=max(abs(x['material']['volume_residual_km3']) for x in rows),
        mass_residual_kg=totalM-r['initial_total_mass_kg'],mass_relative_residual=(totalM-r['initial_total_mass_kg'])/r['initial_total_mass_kg'],
        H_residual_j=totalH+r['cumulative_exported_heat_j']-r['initial_H_j'],
        H_relative_residual=(totalH+r['cumulative_exported_heat_j']-r['initial_H_j'])/r['initial_H_j'],
        water_mass_kg=r['water_mass_kg'],water_mass_change_kg=0.,max_water_relative_error=max(abs(x['water']['relative_volume_error']) for x in rows),
        slab_zones=len(e.checkpoint.subduction_memory.zones),slab_births=e.checkpoint.subduction_memory.births,
        source_sha256=r['source_sha256'],first_split_geometry=next(x['geometry'] for x in r['events'] if x['kind']=='geological_weak_band_split'))
    (root/'result.json').write_bytes(encoded(result));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--dt',type=float,required=True);p.add_argument('--duration',type=float,default=500.);p.add_argument('--resume')
    a=p.parse_args();run(a.output,a.dt,a.duration,a.resume)
