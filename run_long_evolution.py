#!/usr/bin/env python3
"""v0.9.3 checkpoint/resume runner for long tectonic integrations.

``--end-time`` is an absolute simulation time.  A resumed run continues from
its checkpoint to that time.  Keeping checkpoint boundaries aligned to the
internal time step preserves the same stepping sequence as a monolithic run.
"""
from __future__ import annotations
import argparse,csv,json
from dataclasses import replace
from pathlib import Path
import numpy as np

from tectonics.checkpoint import RunCheckpoint,load_checkpoint,save_checkpoint
from tectonics.continental import ContinentalCycleParameters,advance_continental_cycle,initialize_continental_cycle
from tectonics.dynamics import DynamicsParameters,center_net_rotation,update_plate_dynamics
from tectonics.lithosphere import CrustType,advance_lithosphere,boundary_records_for_state,initialize_lithosphere
from tectonics.plates import PlateSystem
from tectonics.simulation import build_prototype,load_config
from tectonics.tides import eccentricity_history_from_config
from tectonics.thermal import ThermalParameters,advance_thermal_state,initialize_thermal_state,diagnose_thermal_state
from tectonics.topography import TopographyParameters,advance_topography,equilibrium_elevation,initialize_topography
from tectonics.topology import PlateTopologyManager,PlateTopologyParameters
from visualization.continental import save_continental_cycle_history,save_continental_flux_history,save_continental_maps
from visualization.thermal import save_activity_history,save_heat_budget,save_thermal_history
from visualization.topography import save_final_maps,save_process_history,save_topography_history
from visualization.topology import build_gif,save_plate_count_history,save_plate_map,save_plate_size_history,save_topology_frame
from visualization.rifting import save_rift_maps, save_rift_history
from visualization.bathymetry import save_bathymetry_components, save_bathymetry_limit_history


def dc(cls,cfg): return cls(**{k:cfg[k] for k in cls.__dataclass_fields__ if k in cfg})

def parse_args():
    p=argparse.ArgumentParser(description='v0.9.3 long integration with progressive rifting and calibrated ocean/trench bathymetry')
    p.add_argument('--config',default='configs/canonical_moon.yaml')
    p.add_argument('--output',default='outputs_v093_long')
    p.add_argument('--resume',default=None,help='Checkpoint directory to resume')
    p.add_argument('--end-time',type=float,required=True,help='Absolute target simulation time, Myr')
    p.add_argument('--dt',type=float,default=None)
    p.add_argument('--checkpoint',default=None,help='Checkpoint directory to write at segment end')
    p.add_argument('--save-frame',action='store_true')
    p.add_argument('--finalize',action='store_true',help='Write final maps/plots/summary and GIF from saved frames')
    return p.parse_args()

def step_sizes(start,end,dt):
    if end < start-1e-10: raise ValueError('end-time precedes checkpoint time')
    span=end-start
    n=int(round(span/dt))
    if abs(n*dt-span)>1e-9:
        raise ValueError(f'Segment span {span} Myr is not an integer multiple of dt={dt}; align checkpoints to dt to preserve deterministic stepping')
    return [dt]*n

def build_checkpoint(state,cycle,thermal,topo,system,baseline,manager,initial_cont_frac,initial_cont_vol,topo_rows,lithosphere_rows,relief_rows,cycle_rows,thermal_rows,events):
    return RunCheckpoint(state,cycle,thermal,topo,system,baseline,manager,initial_cont_frac,initial_cont_vol,topo_rows,lithosphere_rows,relief_rows,cycle_rows,thermal_rows,events)

def main():
    a=parse_args();cfg=load_config(a.config);proto=build_prototype(cfg);lc=cfg['lithosphere'];tc=cfg['tides'];rc=cfg.get('continental_rifting',{});evo=cfg['thermal_evolution']
    dyn0=dc(DynamicsParameters,cfg['plate_dynamics']);topop=dc(TopographyParameters,cfg['topography']);topp=dc(PlateTopologyParameters,cfg['plate_topology']);cc0=dc(ContinentalCycleParameters,cfg['continental_cycle']);thp=dc(ThermalParameters,cfg['thermal'])
    dt=float(a.dt if a.dt is not None else evo['time_step_myr']);out=Path(a.output);frames=out/'frames';checkpoints=out/'checkpoints';frames.mkdir(parents=True,exist_ok=True);checkpoints.mkdir(parents=True,exist_ok=True)
    ecc=eccentricity_history_from_config(cfg,Path(a.config).resolve().parent.parent)
    mass=float(cfg['moon']['mass_earth']);radius=float(cfg['moon']['radius_km']);grav=float(cfg['moon']['surface_gravity_m_s2']);period=float(cfg['moon']['rotation_period_hours']);pmj=float(cfg['primary']['mass_jupiter']);normal=float(cfg['classification']['normal_threshold_km_per_myr']);inactive=float(cfg['classification']['inactive_speed_km_per_myr'])
    areas=proto.mesh.physical_cell_areas_km2(radius)
    manager=PlateTopologyManager(topp)
    if a.resume:
        cp=load_checkpoint(a.resume,manager)
        state,cycle,thermal,topo,system,baseline=cp.state,cp.cycle,cp.thermal,cp.topo,cp.system,cp.baseline
        initial_cont_frac=cp.initial_continental_area_fraction;initial_cont_vol=cp.initial_continental_volume_km3
        topo_rows,lithosphere_rows,relief_rows,cycle_rows,thermal_rows,events=cp.topology_rows,cp.lithosphere_rows,cp.relief_rows,cp.cycle_rows,cp.thermal_rows,cp.events
        print(f'Resumed checkpoint at t={state.time_myr:.1f} Myr with {len(system.plates)} plates')
    else:
        state=initialize_lithosphere(proto.mesh,proto.plates,float(lc['initial_continental_fraction']),int(lc['continental_nuclei']),float(lc['oceanic_thickness_km']),float(lc['continental_thickness_km']),float(lc['initial_continental_age_myr']));cycle=initialize_continental_cycle(proto.mesh)
        thermal=initialize_thermal_state(mass,radius,grav,thp)
        baseline=center_net_rotation(proto.mesh,state,proto.plates,radius) if dyn0.remove_net_rotation else proto.plates;system=PlateSystem(cell_plate=state.cell_plate.copy(),plates=baseline.plates)
        boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive);topo=initialize_topography(proto.mesh,state,boundaries,topop)
        initial_cont_frac=float(np.sum(areas[state.crust_type==int(CrustType.CONTINENTAL)])/np.sum(areas));initial_cont_vol=float(np.sum(areas[state.crust_type==int(CrustType.CONTINENTAL)]*state.crust_thickness_km[state.crust_type==int(CrustType.CONTINENTAL)]))
        topo_rows=[];lithosphere_rows=[];relief_rows=[];cycle_rows=[];events=[]
        d0=diagnose_thermal_state(thermal,mass,radius,grav,period,pmj,ecc.at(0.0),thp);thermal_rows=[{k:getattr(d0,k) for k in d0.__dataclass_fields__}]
    end=float(a.end_time);last_td=None;last_relief=None
    for dti in step_sizes(float(state.time_myr),end,dt):
        thermal,thdiag=advance_thermal_state(thermal,dti,mass,radius,grav,period,pmj,ecc,thp);thermal_rows.append({k:getattr(thdiag,k) for k in thdiag.__dataclass_fields__})
        activity=thermal.tectonic_activity_factor
        dynp=replace(dyn0,force_speed_scale_deg_per_myr=dyn0.force_speed_scale_deg_per_myr*activity)
        ccp=replace(cc0,arc_maturation_rate_per_myr=cc0.arc_maturation_rate_per_myr*activity,continental_arc_thickening_km_per_myr=cc0.continental_arc_thickening_km_per_myr*activity,subduction_erosion_km_per_myr=cc0.subduction_erosion_km_per_myr*activity,delamination_rate_per_myr=cc0.delamination_rate_per_myr*activity)
        system,_,_,_=update_plate_dynamics(proto.mesh,state,system,baseline,radius,dti,normal,inactive,dynp)
        previous_state=state
        state,_,_,lith_diag=advance_lithosphere(proto.mesh,system,state,dti,radius,grav,period,ecc,primary_mass_jupiter=pmj,love_h2=float(tc['love_h2']),reference_eccentricity=float(tc['eccentricity_rms']),oceanic_thickness_km=float(lc['oceanic_thickness_km']),continental_thickness_km=float(lc['continental_thickness_km']),max_continental_thickness_km=float(lc['max_continental_thickness_km']),collision_accretion_fraction=float(lc['collision_accretion_fraction']),tidal_damage_rate_per_myr=float(tc['damage_rate_per_myr'])*max(activity,.5),tidal_damage_relaxation_myr=float(tc['damage_relaxation_myr']),tidal_damage_background_fraction=float(tc.get('damage_background_fraction',0.10)),continental_extension_rate_per_myr=float(rc.get('extension_rate_per_myr',0.012)),continental_extension_relaxation_myr=float(rc.get('extension_relaxation_myr',90.0)),continental_extension_min_duration_myr=float(rc.get('min_duration_myr',32.0)),continental_rift_extension_threshold=float(rc.get('extension_threshold',0.70)),continental_thinning_km_per_myr=float(rc.get('thinning_km_per_myr',0.16)),continental_min_breakup_thickness_km=float(rc.get('min_breakup_thickness_km',19.0)),tidal_thinning_boost_max_fraction=float(rc.get('tidal_thinning_boost_max_fraction',0.25)))
        lithosphere_rows.append({k:getattr(lith_diag,k) for k in lith_diag.__dataclass_fields__})
        system=PlateSystem(cell_plate=state.cell_plate.copy(),plates=system.plates);boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive)
        state,cycle,cycle_diag=advance_continental_cycle(proto.mesh,state,boundaries,cycle,dti,radius,ccp,oceanic_thickness_km=float(lc['oceanic_thickness_km']),previous_lithosphere=previous_state,plate_system=system)
        boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive);system,last_td,new_events=manager.update(proto.mesh,state,system,boundaries,radius,dti)
        if last_td.topology_changed:
            baseline=PlateSystem(cell_plate=system.cell_plate.copy(),plates=system.plates);boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive)
            for ev in new_events:
                rec={'time_myr':ev.time_myr,'kind':ev.kind,'parents':list(ev.parents),'children':list(ev.children),'affected_cells':ev.affected_cells,'detail':ev.detail};events.append(rec);print(f"TOPOLOGY t={ev.time_myr:.1f}: {ev.kind} parents={ev.parents} children={ev.children}")
        topo,last_relief,target=advance_topography(proto.mesh,state,boundaries,topo,dti,radius,topop)
        topo_rows.append({k:getattr(last_td,k) for k in last_td.__dataclass_fields__});relief_rows.append({k:getattr(last_relief,k) for k in last_relief.__dataclass_fields__});cycle_rows.append({k:getattr(cycle_diag,k) for k in cycle_diag.__dataclass_fields__})
    boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive)
    contf=100*float(np.sum(areas[state.crust_type==int(CrustType.CONTINENTAL)])/np.sum(areas))
    print(f"SEGMENT END t={state.time_myr:7.1f} Myr | Tm={thermal.mantle_temperature_k:7.1f} K | activity={thermal.tectonic_activity_factor:5.3f} | plates={len(system.plates):2d} | continent={contf:5.2f}%")
    cp=build_checkpoint(state,cycle,thermal,topo,system,baseline,manager,initial_cont_frac,initial_cont_vol,topo_rows,lithosphere_rows,relief_rows,cycle_rows,thermal_rows,events)
    cp_path=Path(a.checkpoint) if a.checkpoint else checkpoints/f'checkpoint_{int(round(state.time_myr)):04d}_Myr'
    save_checkpoint(cp_path,cp);print('Checkpoint:',cp_path.resolve())
    if a.save_frame or a.finalize:
        fp=frames/f'frame_{state.time_myr:08.1f}_Myr.png';save_topology_frame(proto.mesh,state,topo,system,boundaries,last_td,fp,int(cfg['output'].get('thermal_dpi',120)));print('Frame:',fp.resolve())
    if a.finalize:
        target,_=equilibrium_elevation(proto.mesh,state,boundaries,topop)
        save_final_maps(proto.mesh,state,topo,target,out,int(cfg['output'].get('dpi',180)));save_plate_map(proto.mesh,state,system,out/'plate_map_final.png',int(cfg['output'].get('dpi',180)));save_continental_maps(proto.mesh,state,cycle,out,int(cfg['output'].get('dpi',180)))
        save_rift_maps(proto.mesh,state,out,int(cfg['output'].get('dpi',180)))
        if lithosphere_rows: save_rift_history(lithosphere_rows,out/'rift_history.png')
        if topo_rows: save_plate_count_history(topo_rows,out/'plate_count_history.png');save_plate_size_history(topo_rows,out/'plate_size_history.png')
        if relief_rows: save_topography_history(relief_rows,out/'topography_history.png');save_process_history(relief_rows,out/'topographic_process_history.png')
        if cycle_rows: save_continental_cycle_history(cycle_rows,out/'continental_cycle_history.png');save_continental_flux_history(cycle_rows,out/'continental_flux_history.png')
        save_thermal_history(thermal_rows,out/'thermal_history.png');save_heat_budget(thermal_rows,out/'heat_budget.png');save_activity_history(thermal_rows,out/'tectonic_activity_history.png')
        save_bathymetry_components(proto.mesh,state,boundaries,topop,out,int(cfg['output'].get('dpi',180)));save_bathymetry_limit_history(relief_rows,out/'bathymetry_limit_history.png')
        with (out/'thermal_history.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(thermal_rows[0].keys()));w.writeheader();w.writerows(thermal_rows)
        if lithosphere_rows:
            with (out/'rift_history.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(lithosphere_rows[0].keys()));w.writeheader();w.writerows(lithosphere_rows)
        with (out/'topology_events.json').open('w',encoding='utf-8') as h:json.dump(events,h,ensure_ascii=False,indent=2)
        all_frames=sorted(frames.glob('frame_*_Myr.png'))
        if all_frames: build_gif(all_frames,out/'history.gif',int(evo.get('gif_frame_duration_ms',350)))
        cont=state.crust_type==int(CrustType.CONTINENTAL);final_diag=thermal_rows[-1]
        summary={'version':'0.9.3','model':'v0.9.3 full geology with progressive rifting and physicalized ocean/trench bathymetry','duration_myr':state.time_myr,'time_step_myr':dt,'checkpoint_resume':True,'system_age_start_myr':thp.system_age_at_start_myr,'system_age_final_myr':thermal.system_age_myr,'moon_mass_earth':mass,'moon_radius_km':radius,'surface_gravity_m_s2':grav,'rotation_period_hours':period,'primary_mass_jupiter':pmj,'initial_mantle_temperature_k':thp.initial_mantle_temperature_k,'final_mantle_temperature_k':thermal.mantle_temperature_k,'final_tectonic_activity_factor':thermal.tectonic_activity_factor,'final_thermal_lithosphere_thickness_km':thermal.thermal_lithosphere_thickness_km,'final_convective_heat_flux_w_m2':final_diag['convective_heat_flux_w_m2'],'final_radiogenic_heat_flux_w_m2':final_diag['radiogenic_heat_flux_w_m2'],'final_tidal_heat_flux_w_m2':final_diag['tidal_heat_flux_w_m2'],'initial_plate_count':len(proto.plates.plates),'final_plate_count':len(system.plates),'topology_event_count':len(events),'initial_continental_area_fraction':initial_cont_frac,'final_continental_area_fraction':float(np.sum(areas[cont])/np.sum(areas)),'initial_continental_volume_km3':initial_cont_vol,'final_continental_volume_km3':float(np.sum(areas[cont]*state.crust_thickness_km[cont])),'min_elevation_m':float(np.min(topo.elevation_m)),'max_elevation_m':float(np.max(topo.elevation_m)),'final_deepest_normal_ocean_m':float(relief_rows[-1].get('deepest_normal_ocean_m',0.0)) if relief_rows else 0.0,'final_deepest_trench_anomaly_m':float(relief_rows[-1].get('deepest_trench_anomaly_m',0.0)) if relief_rows else 0.0,'total_numerical_min_clip_cells':int(sum(int(r.get('numerical_min_clip_cells',0)) for r in relief_rows)),'total_numerical_max_clip_cells':int(sum(int(r.get('numerical_max_clip_cells',0)) for r in relief_rows)),'cumulative_generated_continental_area_km2':cycle.cumulative_generated_area_km2,'cumulative_recycled_continental_area_km2':cycle.cumulative_recycled_area_km2,'cumulative_generated_continental_volume_km3':cycle.cumulative_generated_volume_km3,'cumulative_recycled_continental_volume_km3':cycle.cumulative_recycled_volume_km3,'cumulative_breakup_area_km2':float(sum(float(r.get('tidally_rifted_continental_area_km2',0.0)) for r in lithosphere_rows)),'cumulative_continental_thinning_volume_km3':float(sum(float(r.get('continental_thinning_volume_km3',0.0)) for r in lithosphere_rows)),'final_mean_tidal_damage':float(lithosphere_rows[-1]['mean_tidal_damage']) if lithosphere_rows else 0.0,'final_max_rift_extension':float(lithosphere_rows[-1]['max_rift_extension']) if lithosphere_rows else 0.0,'notes':['Long integration was performed through deterministic checkpoint/resume segments aligned to the 4 Myr internal step.','Checkpoint stores full lithosphere, topography, plate dynamics/baseline, thermal state, continental-cycle memory and topology collision memory.','Oceanic background bathymetry uses a saturating plate-cooling relation; trench deflection is bounded and non-additive across duplicate mesh edges.','The +/-18/12 km elevation rails are numerical safety bounds only; calibrated runs should record zero clip cells.']}
        with (out/'summary_v093.json').open('w',encoding='utf-8') as h:json.dump(summary,h,ensure_ascii=False,indent=2)
        print(json.dumps(summary,ensure_ascii=False,indent=2));print('Finalized:',out.resolve())

if __name__=='__main__': main()
