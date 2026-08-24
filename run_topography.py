#!/usr/bin/env python3
"""Run Moon Tectonics v0.6: dynamic plates + lithosphere + tectonic relief."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

from tectonics.dynamics import DynamicsParameters, center_net_rotation, update_plate_dynamics
from tectonics.evolution import snapshot_times
from tectonics.lithosphere import advance_lithosphere,boundary_records_for_state,initialize_lithosphere
from tectonics.plates import PlateSystem
from tectonics.simulation import build_prototype,load_config
from tectonics.tides import eccentricity_history_from_config, tidal_strain_amplitude
from tectonics.topography import TopographyParameters,advance_topography,equilibrium_elevation,initialize_topography
from visualization.topography import build_gif,save_final_maps,save_process_history,save_topography_frame,save_topography_history


def args():
    p=argparse.ArgumentParser(description='v0.6 tectonically generated topography')
    p.add_argument('--config',default='configs/canonical_moon.yaml'); p.add_argument('--output',default=None); p.add_argument('--duration',type=float,default=None); p.add_argument('--interval',type=float,default=None); p.add_argument('--dt',type=float,default=None); p.add_argument('--no-gif',action='store_true'); return p.parse_args()

def substeps(start,end,dt):
    out=[];t=start
    while t+dt<end-1e-12:out.append(dt);t+=dt
    if end>t+1e-12:out.append(end-t)
    return out

def dataclass_params(cls,cfg):
    return cls(**{k:cfg[k] for k in cls.__dataclass_fields__ if k in cfg})

def main():
    a=args(); cfg=load_config(a.config); proto=build_prototype(cfg); lc=cfg['lithosphere']; tc=cfg['tides']; evo=cfg['topography_evolution']; dynp=dataclass_params(DynamicsParameters,cfg['plate_dynamics']); topop=dataclass_params(TopographyParameters,cfg['topography'])
    duration=float(a.duration if a.duration is not None else evo['duration_myr']); interval=float(a.interval if a.interval is not None else evo['frame_interval_myr']); dt=float(a.dt if a.dt is not None else evo['time_step_myr'])
    out=Path(a.output or cfg['output'].get('topography_directory','outputs_v06')); frames=out/'frames'; frames.mkdir(parents=True,exist_ok=True)
    ecc=eccentricity_history_from_config(cfg,Path(a.config).resolve().parent.parent)
    state=initialize_lithosphere(proto.mesh,proto.plates,float(lc['initial_continental_fraction']),int(lc['continental_nuclei']),float(lc['oceanic_thickness_km']),float(lc['continental_thickness_km']),float(lc['initial_continental_age_myr']))
    radius=float(cfg['moon']['radius_km']); grav=float(cfg['moon']['surface_gravity_m_s2']); period=float(cfg['moon']['rotation_period_hours']); pmj=float(cfg['primary']['mass_jupiter']); normal=float(cfg['classification']['normal_threshold_km_per_myr']); inactive=float(cfg['classification']['inactive_speed_km_per_myr'])
    baseline=center_net_rotation(proto.mesh,state,proto.plates,radius) if dynp.remove_net_rotation else proto.plates; system=PlateSystem(cell_plate=state.cell_plate.copy(),plates=baseline.plates)
    boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive); topo=initialize_topography(proto.mesh,state,boundaries,topop); target,_=equilibrium_elevation(proto.mesh,state,boundaries,topop)
    times=snapshot_times(duration,interval); rows=[]; frame_paths=[]; last_topo_diag=None

    for fi,tgt in enumerate(times):
        if tgt>state.time_myr:
            for dti in substeps(state.time_myr,float(tgt),dt):
                system,dyn_diag,_,_=update_plate_dynamics(proto.mesh,state,system,baseline,radius,float(dti),normal,inactive,dynp)
                state,strain,weak,lith_diag=advance_lithosphere(proto.mesh,system,state,float(dti),radius,grav,period,ecc,primary_mass_jupiter=pmj,love_h2=float(tc['love_h2']),reference_eccentricity=float(tc['eccentricity_rms']),oceanic_thickness_km=float(lc['oceanic_thickness_km']),continental_thickness_km=float(lc['continental_thickness_km']),max_continental_thickness_km=float(lc['max_continental_thickness_km']),collision_accretion_fraction=float(lc['collision_accretion_fraction']),tidal_damage_rate_per_myr=float(tc['damage_rate_per_myr']),tidal_damage_relaxation_myr=float(tc['damage_relaxation_myr']),continental_rift_damage_threshold=float(tc['continental_rift_damage_threshold']))
                system=PlateSystem(cell_plate=state.cell_plate.copy(),plates=system.plates)
                boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive)
                topo,last_topo_diag,target=advance_topography(proto.mesh,state,boundaries,topo,float(dti),radius,topop)
                rows.append({k:getattr(last_topo_diag,k) for k in last_topo_diag.__dataclass_fields__})
        boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive)
        fp=frames/f'frame_{fi:04d}_{tgt:08.3f}_Myr.png'; save_topography_frame(proto.mesh,state,topo,system,boundaries,last_topo_diag,fp,int(cfg['output'].get('topography_dpi',120))); frame_paths.append(fp)
        if last_topo_diag: print(f"t={state.time_myr:7.2f} Myr | relief={last_topo_diag.min_elevation_m:7.0f}..{last_topo_diag.max_elevation_m:7.0f} m | cont mean={last_topo_diag.mean_continental_elevation_m:6.0f} m | ocean mean={last_topo_diag.mean_oceanic_elevation_m:7.0f} m")

    save_final_maps(proto.mesh,state,topo,target,out,int(cfg['output'].get('dpi',180)))
    if rows:
        save_topography_history(rows,out/'topography_history.png'); save_process_history(rows,out/'topographic_process_history.png')
        with (out/'topography_history.csv').open('w',newline='',encoding='utf-8') as h:
            w=csv.DictWriter(h,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    if not a.no_gif: build_gif(frame_paths,out/'history.gif',int(evo.get('gif_frame_duration_ms',350)))
    final=rows[-1] if rows else {'min_elevation_m':float(np.min(topo.elevation_m)),'max_elevation_m':float(np.max(topo.elevation_m)),'mean_elevation_m':float(np.mean(topo.elevation_m)),'mean_continental_elevation_m':0,'mean_oceanic_elevation_m':0,'reference_exposed_fraction':float(np.mean(topo.elevation_m>0)),'eroded_volume_km3':0}
    summary={'version':'0.6','model':'v0.5 force-driven lithosphere + tectonically generated effective topography','duration_myr':duration,'time_step_myr':dt,'frame_interval_myr':interval,'mesh_cells':proto.mesh.cell_count,'plate_count':len(system.plates),'min_elevation_m':final['min_elevation_m'],'max_elevation_m':final['max_elevation_m'],'mean_elevation_m':final['mean_elevation_m'],'mean_continental_elevation_m':final['mean_continental_elevation_m'],'mean_oceanic_elevation_m':final['mean_oceanic_elevation_m'],'reference_datum_exposed_fraction':final['reference_exposed_fraction'],'cumulative_eroded_volume_km3':float(sum(r['eroded_volume_km3'] for r in rows)),'topography_parameters':{k:getattr(topop,k) for k in TopographyParameters.__dataclass_fields__},'notes':['Zero elevation is a reference datum, not yet a solved global sea level.','Oceanic bathymetry deepens with sqrt(crust age); divergent boundaries add ridge relief.','Convergent boundaries create trenches and overriding arcs; continent-continent convergence creates orogenic uplift.','Continental crustal thickness from the lithosphere model contributes Airy-style buoyancy and long-lived mountain elevation.','Positive relief undergoes simple diffusive erosion; sediment transport and river incision are deferred.','100 Myr is a validation horizon. Multi-Gyr production runs should wait until thermal/orbital secular evolution is added.']}
    with (out/'summary_v06.json').open('w',encoding='utf-8') as h:json.dump(summary,h,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2));print('Saved:',out.resolve())
if __name__=='__main__':main()
