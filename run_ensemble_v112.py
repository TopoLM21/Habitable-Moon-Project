#!/usr/bin/env python3
"""Multi-seed ensemble driver for Moon Tectonics v0.12 physical-topology model.

The v0.11 material-layer model uses ensembles and resolution sweeps before any
further force-law tuning.  This driver uses a cheaper mesh by default and runs
checkpointed segments so a long seed does not have to restart from zero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from collections import Counter

import numpy as np
import yaml

from tectonics.mesh import build_icosphere
from tectonics.lithosphere import CrustType


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--config', default='configs/canonical_moon.yaml')
    p.add_argument('--output', default='ensemble_v112_topology')
    p.add_argument('--seeds', default='20260806,20260807,20260808,20260809,20260810,20260811,20260812,20260813')
    p.add_argument('--subdivisions', type=int, default=3, help='Cheap ensemble mesh; canonical production mesh is 5')
    p.add_argument('--end-time', type=float, default=4000.0)
    p.add_argument('--dt', type=float, default=4.0)
    p.add_argument('--segment-myr', type=float, default=500.0)
    p.add_argument('--frame-interval', type=float, default=100.0, help='Diagnostic GIF frame cadence; 0 disables frames')
    return p.parse_args()


def _load_json(path: Path):
    with path.open('r',encoding='utf-8') as f: return json.load(f)


def summarize_seed(seed:int, cfg:dict, checkpoint:Path) -> dict:
    meta=_load_json(checkpoint/'meta.json')
    with np.load(checkpoint/'state.npz',allow_pickle=False) as z:
        crust=z['crust_type']; h=z['crust_thickness_km']; owner=z['state_cell_plate']
        material_fraction=z['continental_fraction'].copy() if 'continental_fraction' in z.files else (crust==int(CrustType.CONTINENTAL)).astype(float)
        material_volume=z['continental_volume_km3'].copy() if 'continental_volume_km3' in z.files else None
    mesh=build_icosphere(int(cfg['mesh']['subdivisions']))
    areas=mesh.physical_cell_areas_km2(float(cfg['moon']['radius_km']))
    total=float(np.sum(areas)); cont=crust==int(CrustType.CONTINENTAL)
    plate_count=len(meta['system_plates'])
    plate_area=np.bincount(owner,weights=areas,minlength=plate_count)
    event_counts=Counter(str(e.get('kind','?')) for e in meta.get('events',[]))
    lith=meta.get('lithosphere_rows',[])
    cont_area=float(np.sum(areas*material_fraction))
    cont_vol=float(np.sum(material_volume)) if material_volume is not None else float(np.sum(areas[cont]*h[cont]))
    mean_h=cont_vol/max(cont_area,1e-30)
    p90=float(np.quantile(h[cont],.90)) if np.any(cont) else 0.0
    transport=meta.get('transport_state') or {}
    return {
        'seed':seed,
        'mesh_subdivisions':int(cfg['mesh']['subdivisions']),
        'mesh_cells':int(mesh.cell_count),
        'time_myr':float(meta['time_myr']),
        'plate_count':int(plate_count),
        'largest_plate_fraction':float(np.max(plate_area)/total) if plate_count else 0.0,
        'continental_area_fraction':cont_area/total,
        'visible_continental_area_fraction':float(np.sum(areas[cont])/total),
        'continental_volume_km3':cont_vol,
        'mean_continental_thickness_km':mean_h,
        'p90_continental_thickness_km':p90,
        'topology_event_count':len(meta.get('events',[])),
        'event_counts':dict(event_counts),
        'last_topology_event_myr':max((float(e.get('time_myr',0.0)) for e in meta.get('events',[])),default=0.0),
        'transport_commit_count':int(transport.get('cumulative_commit_count',0)),
        'max_transport_hold_age_myr':float(transport.get('max_hold_age_myr',0.0)),
        'max_abs_transport_volume_error_km3':max((abs(float(r.get('conservative_transport_volume_error_km3',0.0))) for r in lith),default=0.0),
        'cumulative_collision_redistribution_km3':sum(float(r.get('collision_overflow_redistributed_volume_km3',0.0)) for r in lith),
    }


def main():
    a=parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    base=yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    seeds=[int(x.strip()) for x in a.seeds.split(',') if x.strip()]
    rows=[]
    for seed in seeds:
        cfg=json.loads(json.dumps(base))
        cfg['mesh']['subdivisions']=int(a.subdivisions)
        cfg['plates']['seed']=int(seed)
        seed_dir=out/f'seed_{seed}'
        seed_dir.mkdir(parents=True,exist_ok=True)
        cfg_path=seed_dir/'config.yaml'
        cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        current=0.0; resume=None
        while current < float(a.end_time)-1e-9:
            target=min(current+float(a.segment_myr),float(a.end_time))
            cp=seed_dir/f'cp_{int(round(target)):04d}'
            cmd=[sys.executable,'run_long_evolution_v112.py','--config',str(cfg_path),'--output',str(seed_dir/'run'),'--end-time',str(target),'--dt',str(a.dt),'--checkpoint',str(cp)]
            if float(a.frame_interval) > 0:
                cmd += ['--frame-interval',str(a.frame_interval)]
            if abs(target-float(a.end_time)) < 1e-9:
                cmd += ['--finalize']
            if resume is not None: cmd += ['--resume',str(resume)]
            print(f'[seed {seed}] {current:.0f}->{target:.0f} Myr',flush=True)
            subprocess.run(cmd,check=True)
            current=target; resume=cp
        rows.append(summarize_seed(seed,cfg,resume))
        (out/'ensemble_summary.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    # Compact aggregate useful for deciding whether the v0.11 convergence WATCH items are systematic.
    largest=np.array([r['largest_plate_fraction'] for r in rows],float)
    thick=np.array([r['mean_continental_thickness_km'] for r in rows],float)
    aggregate={
        'seed_count':len(rows),
        'mesh_subdivisions':int(a.subdivisions),
        'end_time_myr':float(a.end_time),
        'largest_plate_fraction':{'mean':float(np.mean(largest)),'median':float(np.median(largest)),'min':float(np.min(largest)),'max':float(np.max(largest))},
        'mean_continental_thickness_km':{'mean':float(np.mean(thick)),'median':float(np.median(thick)),'min':float(np.min(thick)),'max':float(np.max(thick))},
        'watch_superplate_over_50pct_count':int(np.sum(largest>.50)),
        'watch_mean_thickness_below_30km_count':int(np.sum(thick<30.0)),
    }
    (out/'ensemble_aggregate.json').write_text(json.dumps(aggregate,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(aggregate,indent=2,ensure_ascii=False))


if __name__=='__main__': main()
