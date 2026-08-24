#!/usr/bin/env python3
"""Run v0.9: v0.8 geology coupled to a long-term mantle heat budget."""
from __future__ import annotations
import argparse,csv,json
from dataclasses import replace
from pathlib import Path
import numpy as np
from tectonics.continental import ContinentalCycleParameters,advance_continental_cycle,initialize_continental_cycle
from tectonics.dynamics import DynamicsParameters,center_net_rotation,update_plate_dynamics
from tectonics.evolution import snapshot_times
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

def parse_args():
    p=argparse.ArgumentParser(description='v0.9 thermal evolution + full tectonic prototype')
    p.add_argument('--config',default='configs/canonical_moon.yaml');p.add_argument('--output',default=None);p.add_argument('--duration',type=float,default=None);p.add_argument('--interval',type=float,default=None);p.add_argument('--dt',type=float,default=None);p.add_argument('--no-gif',action='store_true');return p.parse_args()
def substeps(start,end,dt):
    out=[];t=start
    while t+dt<end-1e-12:out.append(dt);t+=dt
    if end>t+1e-12:out.append(end-t)
    return out
def dc(cls,cfg):return cls(**{k:cfg[k] for k in cls.__dataclass_fields__ if k in cfg})

def main():
    a=parse_args();cfg=load_config(a.config);proto=build_prototype(cfg);lc=cfg['lithosphere'];tc=cfg['tides'];evo=cfg['thermal_evolution']
    dyn0=dc(DynamicsParameters,cfg['plate_dynamics']);topop=dc(TopographyParameters,cfg['topography']);topp=dc(PlateTopologyParameters,cfg['plate_topology']);cc0=dc(ContinentalCycleParameters,cfg['continental_cycle']);thp=dc(ThermalParameters,cfg['thermal'])
    duration=float(a.duration if a.duration is not None else evo['duration_myr']);interval=float(a.interval if a.interval is not None else evo['frame_interval_myr']);dt=float(a.dt if a.dt is not None else evo['time_step_myr'])
    out=Path(a.output or cfg['output'].get('thermal_directory','outputs_v09'));frames=out/'frames';frames.mkdir(parents=True,exist_ok=True)
    ecc=eccentricity_history_from_config(cfg,Path(a.config).resolve().parent.parent)
    state=initialize_lithosphere(proto.mesh,proto.plates,float(lc['initial_continental_fraction']),int(lc['continental_nuclei']),float(lc['oceanic_thickness_km']),float(lc['continental_thickness_km']),float(lc['initial_continental_age_myr']));cycle=initialize_continental_cycle(proto.mesh)
    mass=float(cfg['moon']['mass_earth']);radius=float(cfg['moon']['radius_km']);grav=float(cfg['moon']['surface_gravity_m_s2']);period=float(cfg['moon']['rotation_period_hours']);pmj=float(cfg['primary']['mass_jupiter']);normal=float(cfg['classification']['normal_threshold_km_per_myr']);inactive=float(cfg['classification']['inactive_speed_km_per_myr'])
    thermal=initialize_thermal_state(mass,radius,grav,thp)
    baseline=center_net_rotation(proto.mesh,state,proto.plates,radius) if dyn0.remove_net_rotation else proto.plates;system=PlateSystem(cell_plate=state.cell_plate.copy(),plates=baseline.plates)
    boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive);topo=initialize_topography(proto.mesh,state,boundaries,topop);target,_=equilibrium_elevation(proto.mesh,state,boundaries,topop)
    manager=PlateTopologyManager(topp);times=snapshot_times(duration,interval);topo_rows=[];relief_rows=[];cycle_rows=[];thermal_rows=[];events=[];frame_paths=[];last_td=None
    areas=proto.mesh.physical_cell_areas_km2(radius);initial_cont_frac=float(np.sum(areas[state.crust_type==int(CrustType.CONTINENTAL)])/np.sum(areas));initial_cont_vol=float(np.sum(areas[state.crust_type==int(CrustType.CONTINENTAL)]*state.crust_thickness_km[state.crust_type==int(CrustType.CONTINENTAL)]))
    # Initial thermal diagnostic for plots/summary.
    d0=diagnose_thermal_state(thermal,mass,radius,grav,period,pmj,ecc.at(0.0),thp);thermal_rows.append({k:getattr(d0,k) for k in d0.__dataclass_fields__})
    for fi,tgt in enumerate(times):
        if tgt>state.time_myr:
            for dti in substeps(state.time_myr,float(tgt),dt):
                thermal,thdiag=advance_thermal_state(thermal,float(dti),mass,radius,grav,period,pmj,ecc,thp);thermal_rows.append({k:getattr(thdiag,k) for k in thdiag.__dataclass_fields__})
                activity=thermal.tectonic_activity_factor
                dynp=replace(dyn0,force_speed_scale_deg_per_myr=dyn0.force_speed_scale_deg_per_myr*activity)
                ccp=replace(cc0,arc_maturation_rate_per_myr=cc0.arc_maturation_rate_per_myr*activity,continental_arc_thickening_km_per_myr=cc0.continental_arc_thickening_km_per_myr*activity,subduction_erosion_km_per_myr=cc0.subduction_erosion_km_per_myr*activity,delamination_rate_per_myr=cc0.delamination_rate_per_myr*activity)
                system,_,_,_=update_plate_dynamics(proto.mesh,state,system,baseline,radius,float(dti),normal,inactive,dynp)
                previous_state=state
                state,_,_,_=advance_lithosphere(proto.mesh,system,state,float(dti),radius,grav,period,ecc,primary_mass_jupiter=pmj,love_h2=float(tc['love_h2']),reference_eccentricity=float(tc['eccentricity_rms']),oceanic_thickness_km=float(lc['oceanic_thickness_km']),continental_thickness_km=float(lc['continental_thickness_km']),max_continental_thickness_km=float(lc['max_continental_thickness_km']),collision_accretion_fraction=float(lc['collision_accretion_fraction']),tidal_damage_rate_per_myr=float(tc['damage_rate_per_myr'])*max(activity,.5),tidal_damage_relaxation_myr=float(tc['damage_relaxation_myr']),continental_rift_damage_threshold=float(tc['continental_rift_damage_threshold']))
                system=PlateSystem(cell_plate=state.cell_plate.copy(),plates=system.plates);boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive)
                state,cycle,cycle_diag=advance_continental_cycle(proto.mesh,state,boundaries,cycle,float(dti),radius,ccp,oceanic_thickness_km=float(lc['oceanic_thickness_km']),previous_lithosphere=previous_state,plate_system=system)
                boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive);system,last_td,new_events=manager.update(proto.mesh,state,system,boundaries,radius,float(dti))
                if last_td.topology_changed:
                    baseline=PlateSystem(cell_plate=system.cell_plate.copy(),plates=system.plates);boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive)
                    for ev in new_events:
                        rec={'time_myr':ev.time_myr,'kind':ev.kind,'parents':list(ev.parents),'children':list(ev.children),'affected_cells':ev.affected_cells,'detail':ev.detail};events.append(rec);print(f"TOPOLOGY t={ev.time_myr:.1f}: {ev.kind} parents={ev.parents} children={ev.children}")
                topo,last_relief,target=advance_topography(proto.mesh,state,boundaries,topo,float(dti),radius,topop)
                topo_rows.append({k:getattr(last_td,k) for k in last_td.__dataclass_fields__});relief_rows.append({k:getattr(last_relief,k) for k in last_relief.__dataclass_fields__});cycle_rows.append({k:getattr(cycle_diag,k) for k in cycle_diag.__dataclass_fields__})
        boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive);fp=frames/f'frame_{fi:04d}_{tgt:08.3f}_Myr.png';save_topology_frame(proto.mesh,state,topo,system,boundaries,last_td,fp,int(cfg['output'].get('thermal_dpi',120)));frame_paths.append(fp)
        contf=100*float(np.sum(areas[state.crust_type==int(CrustType.CONTINENTAL)])/np.sum(areas));print(f"t={state.time_myr:7.2f} Myr | Tm={thermal.mantle_temperature_k:7.1f} K | activity={thermal.tectonic_activity_factor:5.3f} | plates={len(system.plates):2d} | continent={contf:5.2f}%")
    save_final_maps(proto.mesh,state,topo,target,out,int(cfg['output'].get('dpi',180)));save_plate_map(proto.mesh,state,system,out/'plate_map_final.png',int(cfg['output'].get('dpi',180)));save_continental_maps(proto.mesh,state,cycle,out,int(cfg['output'].get('dpi',180)))
    if topo_rows:save_plate_count_history(topo_rows,out/'plate_count_history.png');save_plate_size_history(topo_rows,out/'plate_size_history.png')
    if relief_rows:save_topography_history(relief_rows,out/'topography_history.png');save_process_history(relief_rows,out/'topographic_process_history.png')
    if cycle_rows:save_continental_cycle_history(cycle_rows,out/'continental_cycle_history.png');save_continental_flux_history(cycle_rows,out/'continental_flux_history.png')
    save_thermal_history(thermal_rows,out/'thermal_history.png');save_heat_budget(thermal_rows,out/'heat_budget.png');save_activity_history(thermal_rows,out/'tectonic_activity_history.png')
    with (out/'thermal_history.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(thermal_rows[0].keys()));w.writeheader();w.writerows(thermal_rows)
    with (out/'topology_events.json').open('w',encoding='utf-8') as h:json.dump(events,h,ensure_ascii=False,indent=2)
    if not a.no_gif:build_gif(frame_paths,out/'history.gif',int(evo.get('gif_frame_duration_ms',350)))
    cont=state.crust_type==int(CrustType.CONTINENTAL);final_diag=thermal_rows[-1]
    summary={'version':'0.9','model':'v0.8 geology coupled to global mantle thermal evolution','duration_myr':duration,'time_step_myr':dt,'system_age_start_myr':thp.system_age_at_start_myr,'system_age_final_myr':thermal.system_age_myr,'moon_mass_earth':mass,'moon_radius_km':radius,'surface_gravity_m_s2':grav,'primary_mass_jupiter':pmj,'initial_mantle_temperature_k':thp.initial_mantle_temperature_k,'final_mantle_temperature_k':thermal.mantle_temperature_k,'initial_tectonic_activity_factor':1.0,'final_tectonic_activity_factor':thermal.tectonic_activity_factor,'final_thermal_lithosphere_thickness_km':thermal.thermal_lithosphere_thickness_km,'final_convective_heat_flux_w_m2':final_diag['convective_heat_flux_w_m2'],'final_radiogenic_heat_flux_w_m2':final_diag['radiogenic_heat_flux_w_m2'],'final_tidal_heat_flux_w_m2':final_diag['tidal_heat_flux_w_m2'],'initial_plate_count':len(proto.plates.plates),'final_plate_count':len(system.plates),'topology_event_count':len(events),'initial_continental_area_fraction':initial_cont_frac,'final_continental_area_fraction':float(np.sum(areas[cont])/np.sum(areas)),'initial_continental_volume_km3':initial_cont_vol,'final_continental_volume_km3':float(np.sum(areas[cont]*state.crust_thickness_km[cont])),'min_elevation_m':float(np.min(topo.elevation_m)),'max_elevation_m':float(np.max(topo.elevation_m)),'notes':['Moon mass sets mantle heat capacity/radiogenic inventory; radius sets cooling area and mantle-depth scale; surface gravity enters Rayleigh convection scaling.','Radiogenic heat uses an effective exponential mixture, not isotope-by-isotope geochemistry yet.','Tidal heat uses a synchronous eccentricity-tide formula with configurable effective k2/Q.','Thermal activity scales effective plate driving and continental-cycle rates; this is still not a 3-D mantle solver.']}
    with (out/'summary_v09.json').open('w',encoding='utf-8') as h:json.dump(summary,h,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2));print('Saved:',out.resolve())
if __name__=='__main__':main()
