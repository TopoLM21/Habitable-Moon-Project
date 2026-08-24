#!/usr/bin/env python3
"""Run v0.7: dynamic plate birth, welding and disappearance + v0.6 relief."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from tectonics.dynamics import DynamicsParameters,center_net_rotation,update_plate_dynamics
from tectonics.evolution import snapshot_times
from tectonics.lithosphere import advance_lithosphere,boundary_records_for_state,initialize_lithosphere
from tectonics.plates import PlateSystem
from tectonics.simulation import build_prototype,load_config
from tectonics.tides import eccentricity_history_from_config
from tectonics.topography import TopographyParameters,advance_topography,equilibrium_elevation,initialize_topography
from tectonics.topology import PlateTopologyManager,PlateTopologyParameters
from visualization.topography import save_final_maps,save_process_history,save_topography_history
from visualization.topology import build_gif,save_plate_count_history,save_plate_map,save_plate_size_history,save_topology_frame

def parse_args():
    p=argparse.ArgumentParser(description='v0.7 dynamic plate topology')
    p.add_argument('--config',default='configs/canonical_moon.yaml');p.add_argument('--output',default=None);p.add_argument('--duration',type=float,default=None);p.add_argument('--interval',type=float,default=None);p.add_argument('--dt',type=float,default=None);p.add_argument('--no-gif',action='store_true');return p.parse_args()
def substeps(start,end,dt):
    out=[];t=start
    while t+dt<end-1e-12:out.append(dt);t+=dt
    if end>t+1e-12:out.append(end-t)
    return out
def dc(cls,cfg):return cls(**{k:cfg[k] for k in cls.__dataclass_fields__ if k in cfg})

def main():
    a=parse_args();cfg=load_config(a.config);proto=build_prototype(cfg);lc=cfg['lithosphere'];tc=cfg['tides'];evo=cfg['topology_evolution'];dynp=dc(DynamicsParameters,cfg['plate_dynamics']);topop=dc(TopographyParameters,cfg['topography']);topp=dc(PlateTopologyParameters,cfg['plate_topology'])
    duration=float(a.duration if a.duration is not None else evo['duration_myr']);interval=float(a.interval if a.interval is not None else evo['frame_interval_myr']);dt=float(a.dt if a.dt is not None else evo['time_step_myr'])
    out=Path(a.output or cfg['output'].get('topology_directory','outputs_v07'));frames=out/'frames';frames.mkdir(parents=True,exist_ok=True)
    ecc=eccentricity_history_from_config(cfg,Path(a.config).resolve().parent.parent)
    state=initialize_lithosphere(proto.mesh,proto.plates,float(lc['initial_continental_fraction']),int(lc['continental_nuclei']),float(lc['oceanic_thickness_km']),float(lc['continental_thickness_km']),float(lc['initial_continental_age_myr']))
    radius=float(cfg['moon']['radius_km']);grav=float(cfg['moon']['surface_gravity_m_s2']);period=float(cfg['moon']['rotation_period_hours']);pmj=float(cfg['primary']['mass_jupiter']);normal=float(cfg['classification']['normal_threshold_km_per_myr']);inactive=float(cfg['classification']['inactive_speed_km_per_myr'])
    baseline=center_net_rotation(proto.mesh,state,proto.plates,radius) if dynp.remove_net_rotation else proto.plates;system=PlateSystem(cell_plate=state.cell_plate.copy(),plates=baseline.plates)
    boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive);topo=initialize_topography(proto.mesh,state,boundaries,topop);target,_=equilibrium_elevation(proto.mesh,state,boundaries,topop)
    manager=PlateTopologyManager(topp);times=snapshot_times(duration,interval);topo_rows=[];relief_rows=[];events=[];frame_paths=[];last_td=None
    for fi,tgt in enumerate(times):
        if tgt>state.time_myr:
            for dti in substeps(state.time_myr,float(tgt),dt):
                system,_,_,_=update_plate_dynamics(proto.mesh,state,system,baseline,radius,float(dti),normal,inactive,dynp)
                state,_,_,_=advance_lithosphere(proto.mesh,system,state,float(dti),radius,grav,period,ecc,primary_mass_jupiter=pmj,love_h2=float(tc['love_h2']),reference_eccentricity=float(tc['eccentricity_rms']),oceanic_thickness_km=float(lc['oceanic_thickness_km']),continental_thickness_km=float(lc['continental_thickness_km']),max_continental_thickness_km=float(lc['max_continental_thickness_km']),collision_accretion_fraction=float(lc['collision_accretion_fraction']),tidal_damage_rate_per_myr=float(tc['damage_rate_per_myr']),tidal_damage_relaxation_myr=float(tc['damage_relaxation_myr']),continental_rift_damage_threshold=float(tc['continental_rift_damage_threshold']))
                system=PlateSystem(cell_plate=state.cell_plate.copy(),plates=system.plates);boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive)
                system,last_td,new_events=manager.update(proto.mesh,state,system,boundaries,radius,float(dti))
                if last_td.topology_changed:
                    # Reset the slow mantle-memory template to the mechanically continuous post-event motion.
                    baseline=PlateSystem(cell_plate=system.cell_plate.copy(),plates=system.plates)
                    boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive)
                    for ev in new_events:
                        events.append({'time_myr':ev.time_myr,'kind':ev.kind,'parents':list(ev.parents),'children':list(ev.children),'affected_cells':ev.affected_cells,'detail':ev.detail})
                        print(f"TOPOLOGY t={ev.time_myr:.1f}: {ev.kind} parents={ev.parents} children={ev.children} | {ev.detail}")
                topo,last_relief,target=advance_topography(proto.mesh,state,boundaries,topo,float(dti),radius,topop)
                topo_rows.append({k:getattr(last_td,k) for k in last_td.__dataclass_fields__})
                relief_rows.append({k:getattr(last_relief,k) for k in last_relief.__dataclass_fields__})
        boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive)
        fp=frames/f'frame_{fi:04d}_{tgt:08.3f}_Myr.png';save_topology_frame(proto.mesh,state,topo,system,boundaries,last_td,fp,int(cfg['output'].get('topology_dpi',120)));frame_paths.append(fp)
        print(f"t={state.time_myr:7.2f} Myr | plates={len(system.plates):2d} | relief={np.min(topo.elevation_m):7.0f}..{np.max(topo.elevation_m):7.0f} m")
    save_final_maps(proto.mesh,state,topo,target,out,int(cfg['output'].get('dpi',180)));save_plate_map(proto.mesh,state,system,out/'plate_map_final.png',int(cfg['output'].get('dpi',180)))
    if topo_rows:
        save_plate_count_history(topo_rows,out/'plate_count_history.png');save_plate_size_history(topo_rows,out/'plate_size_history.png')
        with (out/'topology_history.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(topo_rows[0].keys()));w.writeheader();w.writerows(topo_rows)
    if relief_rows:save_topography_history(relief_rows,out/'topography_history.png');save_process_history(relief_rows,out/'topographic_process_history.png')
    with (out/'topology_events.json').open('w',encoding='utf-8') as h:json.dump(events,h,ensure_ascii=False,indent=2)
    if not a.no_gif:build_gif(frame_paths,out/'history.gif',int(evo.get('gif_frame_duration_ms',350)))
    counts=np.bincount(state.cell_plate,minlength=len(system.plates));summary={'version':'0.7','model':'v0.6 tectonic relief + dynamic plate topology','duration_myr':duration,'time_step_myr':dt,'frame_interval_myr':interval,'mesh_cells':proto.mesh.cell_count,'initial_plate_count':len(proto.plates.plates),'final_plate_count':len(system.plates),'topology_event_count':len(events),'split_count':sum(e['kind']=='split' for e in events),'merge_count':sum(e['kind']=='merge' for e in events),'absorbed_small_plate_count':sum(e['kind']=='absorb' for e in events),'final_min_plate_cells':int(np.min(counts)),'final_max_plate_cells':int(np.max(counts)),'min_elevation_m':float(np.min(topo.elevation_m)),'max_elevation_m':float(np.max(topo.elevation_m)),'events':events,'notes':['Plate IDs are compacted after every topology event.','Breakup requires a young damaged rift band that geometrically disconnects a plate into large connected children.','Continental plates weld only after a persistent convergent contact; short collisions do not instantly merge identities.','Very small plates are absorbed by the neighbour with the longest common boundary.','Post-event Euler velocities are inherited continuously and become the new mantle-memory baseline.','This remains an effective plate-graph model; brittle fracture and 3-D mantle convection are deferred.']}
    with (out/'summary_v07.json').open('w',encoding='utf-8') as h:json.dump(summary,h,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2));print('Saved:',out.resolve())
if __name__=='__main__':main()
