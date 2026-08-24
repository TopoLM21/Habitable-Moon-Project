#!/usr/bin/env python3
"""Moon Tectonics v0.14 passive-hydrosphere checkpoint/resume runner.

Forward development from the reconstructed v0.10 baseline; adds a conservative continental-material layer decoupled from discrete plate ownership.
The conservative transport architecture and sub-grid parameters are recovered
from those reports; several v0.9.8 mantle/GPE/collapse coefficients are
explicit reconstruction assumptions because their original source was lost.
"""
from __future__ import annotations
import argparse,csv,json
from dataclasses import replace
from pathlib import Path
import numpy as np

from tectonics.checkpoint import RunCheckpoint,load_checkpoint,save_checkpoint
from tectonics.continental import ContinentalCycleParameters,advance_continental_cycle,initialize_continental_cycle
from tectonics.dynamics import DynamicsParameters,center_net_rotation,update_plate_dynamics
from tectonics.mantle import MantleFlowParameters,advance_mantle_flow,initialize_mantle_flow
from tectonics.transport import SubgridTransportParameters,initialize_transport_state,remap_transport_state
from tectonics.lithosphere import CrustType,advance_lithosphere,boundary_records_for_state,initialize_lithosphere,initialize_oceanic_crust_ages,continental_material_fields
from tectonics.hydrosphere import HydrosphereParameters,advance_hydrosphere,diagnose_hydrosphere,initialize_hydrosphere
from tectonics.late_tectonics import LateTectonicsParameters,advance_late_tectonics
from tectonics.plates import PlateSystem
from tectonics.simulation import build_prototype,load_config
from tectonics.tides import eccentricity_history_from_config
from tectonics.thermal import ThermalParameters,advance_thermal_state,initialize_thermal_state,diagnose_thermal_state
from tectonics.topography import TopographyParameters,advance_topography,equilibrium_elevation,initialize_topography
from tectonics.topology import PlateTopologyManager,PlateTopologyParameters
from visualization.continental import save_continental_cycle_history,save_continental_flux_history,save_continental_maps,save_continental_history_frame
from visualization.thermal import save_activity_history,save_heat_budget,save_thermal_history
from visualization.topography import save_final_maps,save_process_history,save_topography_history
from visualization.topology import build_gif,save_plate_count_history,save_plate_map,save_plate_size_history,save_topology_frame,save_plate_history_frame
from visualization.rifting import save_rift_maps, save_rift_history
from visualization.bathymetry import save_bathymetry_components, save_bathymetry_limit_history
from visualization.late_tectonics import save_late_tectonic_maps,save_late_history,save_nucleation_history
from visualization.hydrosphere import save_hydrosphere_frame,save_hydrosphere_history,save_final_hydrosphere_maps


def dc(cls,cfg): return cls(**{k:cfg[k] for k in cls.__dataclass_fields__ if k in cfg})

def parse_args():
    p=argparse.ArgumentParser(description='v0.14 conserved-water sea-level + v0.13 stable topology long integration')
    p.add_argument('--config',default='configs/canonical_moon.yaml')
    p.add_argument('--output',default='outputs_v114_hydrosphere')
    p.add_argument('--resume',default=None,help='Checkpoint directory to resume')
    p.add_argument('--end-time',type=float,required=True,help='Absolute target simulation time, Myr')
    p.add_argument('--dt',type=float,default=None)
    p.add_argument('--checkpoint',default=None,help='Checkpoint directory to write at segment end')
    p.add_argument('--save-frame',action='store_true')
    p.add_argument('--frame-interval',type=float,default=None,help='Save diagnostic animation frames every N Myr (must align to dt)')
    p.add_argument('--finalize',action='store_true',help='Write final maps/plots/summary and GIF from saved frames')
    p.add_argument('--surface-only-frames',action='store_true',help='When saving animation frames, render only the hydrosphere surface map (faster long visual runs)')
    return p.parse_args()

def assert_plate_consistency(state, system, where: str) -> None:
    owner=np.asarray(state.cell_plate,dtype=np.int64)
    if owner.size == 0:
        raise RuntimeError(f"{where}: empty plate-owner field")
    lo=int(np.min(owner)); hi=int(np.max(owner)); pcount=len(system.plates)
    if lo < 0 or hi >= pcount:
        raise RuntimeError(f"{where}: inconsistent plate ownership range [{lo},{hi}] for {pcount} plates")
    present=np.unique(owner)
    expected=np.arange(pcount,dtype=present.dtype)
    if not np.array_equal(present,expected):
        raise RuntimeError(f"{where}: non-compact plate ids {present.tolist()} for {pcount} plates")

def step_sizes(start,end,dt):
    if end < start-1e-10: raise ValueError('end-time precedes checkpoint time')
    span=end-start
    n=int(round(span/dt))
    if abs(n*dt-span)>1e-9:
        raise ValueError(f'Segment span {span} Myr is not an integer multiple of dt={dt}; align checkpoints to dt to preserve deterministic stepping')
    return [dt]*n

def build_checkpoint(state,cycle,thermal,topo,system,baseline,manager,initial_cont_frac,initial_cont_vol,topo_rows,lithosphere_rows,relief_rows,cycle_rows,thermal_rows,late_rows,events,mantle_flow,transport_state,hydrosphere,hydrosphere_rows):
    return RunCheckpoint(
        state=state,cycle=cycle,thermal=thermal,topo=topo,system=system,baseline=baseline,manager=manager,
        initial_continental_area_fraction=initial_cont_frac,initial_continental_volume_km3=initial_cont_vol,
        topology_rows=topo_rows,lithosphere_rows=lithosphere_rows,relief_rows=relief_rows,cycle_rows=cycle_rows,thermal_rows=thermal_rows,events=events,late_rows=late_rows,
        mantle_flow=mantle_flow,transport_state=transport_state,hydrosphere=hydrosphere,hydrosphere_rows=hydrosphere_rows,
    )

def main():
    a=parse_args();cfg=load_config(a.config);proto=build_prototype(cfg);lc=cfg['lithosphere'];tc=cfg['tides'];rc=cfg.get('continental_rifting',{});evo=cfg['thermal_evolution']
    dyn0=dc(DynamicsParameters,cfg['plate_dynamics']);topop=dc(TopographyParameters,cfg['topography']);topp=dc(PlateTopologyParameters,cfg['plate_topology']);cc0=dc(ContinentalCycleParameters,cfg['continental_cycle']);thp=dc(ThermalParameters,cfg['thermal']);latep=dc(LateTectonicsParameters,cfg.get('late_tectonics',{}));mantlep=dc(MantleFlowParameters,cfg.get('mantle_flow',{}));transportp=dc(SubgridTransportParameters,cfg.get('subgrid_transport',{}));hydrop=dc(HydrosphereParameters,cfg.get('hydrosphere',{}))
    dt=float(a.dt if a.dt is not None else evo['time_step_myr']);out=Path(a.output);frames=out/'frames';plate_frames=out/'plate_frames';continental_frames=out/'continental_frames';hydro_frames=out/'hydrosphere_frames';checkpoints=out/'checkpoints';frames.mkdir(parents=True,exist_ok=True);plate_frames.mkdir(parents=True,exist_ok=True);continental_frames.mkdir(parents=True,exist_ok=True);hydro_frames.mkdir(parents=True,exist_ok=True);checkpoints.mkdir(parents=True,exist_ok=True)
    ecc=eccentricity_history_from_config(cfg,Path(a.config).resolve().parent.parent)
    mass=float(cfg['moon']['mass_earth']);radius=float(cfg['moon']['radius_km']);grav=float(cfg['moon']['surface_gravity_m_s2']);period=float(cfg['moon']['rotation_period_hours']);pmj=float(cfg['primary']['mass_jupiter']);normal=float(cfg['classification']['normal_threshold_km_per_myr']);inactive=float(cfg['classification']['inactive_speed_km_per_myr'])
    areas=proto.mesh.physical_cell_areas_km2(radius)
    manager=PlateTopologyManager(topp)
    if a.resume:
        cp=load_checkpoint(a.resume,manager)
        state,cycle,thermal,topo,system,baseline=cp.state,cp.cycle,cp.thermal,cp.topo,cp.system,cp.baseline
        mantle_flow=cp.mantle_flow
        transport_state=cp.transport_state
        if mantle_flow is None or transport_state is None:
            raise ValueError('v0.11 resume requires a compatible v0.10-reconstructed/v0.11 checkpoint; start a fresh run when upgrading from older layouts')
        initial_cont_frac=cp.initial_continental_area_fraction;initial_cont_vol=cp.initial_continental_volume_km3
        topo_rows,lithosphere_rows,relief_rows,cycle_rows,thermal_rows,late_rows,events=cp.topology_rows,cp.lithosphere_rows,cp.relief_rows,cp.cycle_rows,cp.thermal_rows,cp.late_rows,cp.events
        hydrosphere=cp.hydrosphere
        if hydrosphere is None:
            raise ValueError('v0.14 resume requires a v0.14 hydrosphere checkpoint; start a fresh v0.14 run instead of silently recalibrating water at resume time')
        hydrosphere_rows=cp.hydrosphere_rows
        last_hydro_diag=diagnose_hydrosphere(proto.mesh,state,topo,hydrosphere,radius,hydrop)
        print(f'Resumed checkpoint at t={state.time_myr:.1f} Myr with {len(system.plates)} plates | sea={last_hydro_diag.sea_level_m:+.1f} m')
    else:
        state=initialize_lithosphere(proto.mesh,proto.plates,float(lc['initial_continental_fraction']),int(lc['continental_nuclei']),float(lc['oceanic_thickness_km']),float(lc['continental_thickness_km']),float(lc['initial_continental_age_myr']),radius_km=radius);cycle=initialize_continental_cycle(proto.mesh)
        thermal=initialize_thermal_state(mass,radius,grav,thp)
        baseline=center_net_rotation(proto.mesh,state,proto.plates,radius) if dyn0.remove_net_rotation else proto.plates;system=PlateSystem(cell_plate=state.cell_plate.copy(),plates=baseline.plates)
        mantle_flow=initialize_mantle_flow(proto.mesh,baseline)
        transport_state=initialize_transport_state(len(system.plates))
        boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive)
        state=initialize_oceanic_crust_ages(proto.mesh,state,boundaries,radius,spreading_rate_km_per_myr=float(lc.get('initial_oceanic_age_spreading_rate_km_per_myr',30.0)),max_age_myr=float(lc.get('initial_oceanic_age_max_myr',160.0)),unseeded_age_myr=float(lc.get('initial_oceanic_age_unseeded_myr',120.0)))
        topo=initialize_topography(proto.mesh,state,boundaries,topop)
        init_frac_field,init_vol_field=continental_material_fields(state,areas);initial_cont_frac=float(np.sum(areas*init_frac_field)/np.sum(areas));initial_cont_vol=float(np.sum(init_vol_field))
        topo_rows=[];lithosphere_rows=[];relief_rows=[];cycle_rows=[];late_rows=[];events=[]
        hydrosphere=initialize_hydrosphere(proto.mesh,topo,radius,hydrop)
        last_hydro_diag=diagnose_hydrosphere(proto.mesh,state,topo,hydrosphere,radius,hydrop)
        hydrosphere_rows=[{k:getattr(last_hydro_diag,k) for k in last_hydro_diag.__dataclass_fields__}]
        print(f'Initialized hydrosphere: water={hydrosphere.water_volume_km3/1e9:.3f} billion km3 | sea={last_hydro_diag.sea_level_m:+.3f} m | land={100*last_hydro_diag.land_area_fraction:.2f}%')
        d0=diagnose_thermal_state(thermal,mass,radius,grav,period,pmj,ecc.at(0.0),thp);thermal_rows=[{k:getattr(d0,k) for k in d0.__dataclass_fields__}]
    end=float(a.end_time);last_td=None;last_relief=None
    frame_interval=None if a.frame_interval is None else float(a.frame_interval)
    if frame_interval is not None:
        if frame_interval <= 0 or abs(round(frame_interval/dt)*dt-frame_interval)>1e-9:
            raise ValueError(f'frame-interval={frame_interval} must be a positive integer multiple of dt={dt}')

    def save_animation_frames():
        fp=frames/f'frame_{state.time_myr:08.1f}_Myr.png'
        pp=plate_frames/f'plate_{state.time_myr:08.1f}_Myr.png'
        cpng=continental_frames/f'continental_{state.time_myr:08.1f}_Myr.png'
        hpng=hydro_frames/f'surface_{state.time_myr:08.1f}_Myr.png'
        if not a.surface_only_frames:
            save_topology_frame(proto.mesh,state,topo,system,boundaries,last_td,fp,int(cfg['output'].get('thermal_dpi',120)))
            save_plate_history_frame(proto.mesh,state,system,pp,int(cfg['output'].get('thermal_dpi',120)))
            save_continental_history_frame(proto.mesh,state,cpng,int(cfg['output'].get('thermal_dpi',120)))
        save_hydrosphere_frame(proto.mesh,state,topo,hydrosphere,last_hydro_diag,radius,hpng,int(cfg['output'].get('thermal_dpi',120)))
        print('Surface frame:',hpng.resolve())

    # Include t=0 in fresh animation sets. Resumed segments inherit prior frames.
    if frame_interval is not None and not a.resume:
        save_animation_frames()
    for dti in step_sizes(float(state.time_myr),end,dt):
        assert_plate_consistency(state,system,"step start")
        thermal,thdiag=advance_thermal_state(thermal,dti,mass,radius,grav,period,pmj,ecc,thp);thermal_rows.append({k:getattr(thdiag,k) for k in thdiag.__dataclass_fields__})
        activity=thermal.tectonic_activity_factor
        mantle_flow,mantle_diag=advance_mantle_flow(proto.mesh,mantle_flow,dti,activity,mantlep)
        dynp=replace(dyn0,force_speed_scale_deg_per_myr=dyn0.force_speed_scale_deg_per_myr*activity)
        ccp=replace(cc0,arc_maturation_rate_per_myr=cc0.arc_maturation_rate_per_myr*activity,continental_arc_thickening_km_per_myr=cc0.continental_arc_thickening_km_per_myr*activity,subduction_erosion_km_per_myr=cc0.subduction_erosion_km_per_myr*activity,delamination_rate_per_myr=cc0.delamination_rate_per_myr*activity)
        system,dyn_diag,_,_=update_plate_dynamics(
            proto.mesh,state,system,baseline,radius,dti,normal,inactive,dynp,
            mantle_flow=mantle_flow,
            thermal_lithosphere_thickness_km=thermal.thermal_lithosphere_thickness_km,
        )
        pre_lith_boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive)
        collision_suppression=manager.extension_suppression_field(proto.mesh,state,pre_lith_boundaries)
        previous_state=state
        # v0.9.7: rift accumulation/thinning follows the waning tectonic
        # activity instead of remaining at formation-era speed forever.
        rift_activity_scale=float(max(activity,0.0)) ** float(rc.get('activity_scaling_exponent',0.75))
        state,_,_,lith_diag=advance_lithosphere(proto.mesh,system,state,dti,radius,grav,period,ecc,primary_mass_jupiter=pmj,love_h2=float(tc['love_h2']),reference_eccentricity=float(tc['eccentricity_rms']),oceanic_thickness_km=float(lc['oceanic_thickness_km']),continental_thickness_km=float(lc['continental_thickness_km']),max_continental_thickness_km=float(lc['max_continental_thickness_km']),collision_accretion_fraction=float(lc['collision_accretion_fraction']),tidal_damage_rate_per_myr=float(tc['damage_rate_per_myr'])*max(activity,.5),tidal_damage_relaxation_myr=float(tc['damage_relaxation_myr']),tidal_damage_background_fraction=float(tc.get('damage_background_fraction',0.10)),continental_extension_rate_per_myr=float(rc.get('extension_rate_per_myr',0.012))*rift_activity_scale,continental_extension_relaxation_myr=float(rc.get('extension_relaxation_myr',90.0)),continental_extension_min_duration_myr=float(rc.get('min_duration_myr',32.0)),continental_rift_extension_threshold=float(rc.get('extension_threshold',0.70)),continental_thinning_km_per_myr=float(rc.get('thinning_km_per_myr',0.16))*rift_activity_scale,continental_min_breakup_thickness_km=float(rc.get('min_breakup_thickness_km',19.0)),continental_breakup_min_extension_forcing=float(rc.get('breakup_min_extension_forcing',0.75)),continental_extension_requires_two_plate_flanks=bool(rc.get('extension_requires_two_plate_flanks',True)),tidal_thinning_boost_max_fraction=float(rc.get('tidal_thinning_boost_max_fraction',0.25)),continental_extension_suppression=collision_suppression,transport_state=transport_state,transport_parameters=transportp)
        lithosphere_rows.append({k:getattr(lith_diag,k) for k in lith_diag.__dataclass_fields__ if k!='material_source_index'})
        system=PlateSystem(cell_plate=state.cell_plate.copy(),plates=system.plates);boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive)
        state,cycle,cycle_diag=advance_continental_cycle(proto.mesh,state,boundaries,cycle,dti,radius,ccp,oceanic_thickness_km=float(lc['oceanic_thickness_km']),previous_lithosphere=previous_state,plate_system=system,transport_source_index=lith_diag.material_source_index)
        boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive)
        late_diag=advance_late_tectonics(proto.mesh,state,system,boundaries,dti,radius,activity,latep,last_split_time_myr=manager.last_split_time_myr)
        late_rows.append({k:getattr(late_diag,k) for k in late_diag.__dataclass_fields__})
        if late_diag.nucleated_rift:
            print(f"LATE RIFT t={late_diag.time_myr:.1f}: plate={late_diag.nucleated_plate} path_cells={late_diag.nucleated_path_cells} score={late_diag.nucleation_score:.3f}")
        pre_topology_system=system
        system,last_td,new_events=manager.update(proto.mesh,state,system,boundaries,radius,dti)
        assert_plate_consistency(state,system,f"after topology t={state.time_myr:.1f}")
        if last_td.topology_changed:
            transport_state=remap_transport_state(pre_topology_system,system,transport_state)
            # Critical v0.9.8 rule: baseline/mantle memory is NOT overwritten
            # by the current slowed plate velocities after topology changes.
            boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive)
            for ev in new_events:
                rec={'time_myr':ev.time_myr,'kind':ev.kind,'parents':list(ev.parents),'children':list(ev.children),'affected_cells':ev.affected_cells,'detail':ev.detail};events.append(rec);print(f"TOPOLOGY t={ev.time_myr:.1f}: {ev.kind} parents={ev.parents} children={ev.children}")
        topo,last_relief,target=advance_topography(proto.mesh,state,boundaries,topo,dti,radius,topop)
        hydrosphere,last_hydro_diag=advance_hydrosphere(proto.mesh,state,topo,hydrosphere,radius,hydrop)
        hydrosphere_rows.append({k:getattr(last_hydro_diag,k) for k in last_hydro_diag.__dataclass_fields__})
        topo_rows.append({k:getattr(last_td,k) for k in last_td.__dataclass_fields__});relief_rows.append({k:getattr(last_relief,k) for k in last_relief.__dataclass_fields__});cycle_rows.append({k:getattr(cycle_diag,k) for k in cycle_diag.__dataclass_fields__})
        if frame_interval is not None and abs((state.time_myr/frame_interval)-round(state.time_myr/frame_interval))<1e-9:
            save_animation_frames()
    boundaries=boundary_records_for_state(proto.mesh,state,system,radius,normal,inactive)
    contf=100*float(np.sum(areas[state.crust_type==int(CrustType.CONTINENTAL)])/np.sum(areas))
    print(f"SEGMENT END t={state.time_myr:7.1f} Myr | Tm={thermal.mantle_temperature_k:7.1f} K | activity={thermal.tectonic_activity_factor:5.3f} | plates={len(system.plates):2d} | continent={contf:5.2f}% | transport_commits={transport_state.cumulative_commit_count} | mantle_rms={mantle_diag.rms_speed_deg_per_myr if 'mantle_diag' in locals() else 0.0:.3f} deg/Myr | sea={last_hydro_diag.sea_level_m:+.1f} m | land={100*last_hydro_diag.land_area_fraction:.1f}%")
    cp=build_checkpoint(state,cycle,thermal,topo,system,baseline,manager,initial_cont_frac,initial_cont_vol,topo_rows,lithosphere_rows,relief_rows,cycle_rows,thermal_rows,late_rows,events,mantle_flow,transport_state,hydrosphere,hydrosphere_rows)
    cp_path=Path(a.checkpoint) if a.checkpoint else checkpoints/f'checkpoint_{int(round(state.time_myr)):04d}_Myr'
    save_checkpoint(cp_path,cp);print('Checkpoint:',cp_path.resolve())
    if a.save_frame or a.finalize:
        save_animation_frames()
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
        save_late_tectonic_maps(proto.mesh,state,out,int(cfg['output'].get('dpi',180)));save_late_history(late_rows,out/'late_tectonics_history.png');save_nucleation_history(late_rows,out/'late_rift_nucleation_history.png')
        save_final_hydrosphere_maps(proto.mesh,state,topo,hydrosphere,radius,hydrop,out,int(cfg['output'].get('dpi',180)));save_hydrosphere_history(hydrosphere_rows,out/'hydrosphere_history.png')
        with (out/'thermal_history.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(thermal_rows[0].keys()));w.writeheader();w.writerows(thermal_rows)
        if lithosphere_rows:
            with (out/'rift_history.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(lithosphere_rows[0].keys()));w.writeheader();w.writerows(lithosphere_rows)
        if late_rows:
            with (out/'late_tectonics_history.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(late_rows[0].keys()));w.writeheader();w.writerows(late_rows)
        with (out/'topology_events.json').open('w',encoding='utf-8') as h:json.dump(events,h,ensure_ascii=False,indent=2)
        if hydrosphere_rows:
            with (out/'hydrosphere_history.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(hydrosphere_rows[0].keys()));w.writeheader();w.writerows(hydrosphere_rows)
        all_frames=sorted(frames.glob('frame_*_Myr.png'))
        all_plate_frames=sorted(plate_frames.glob('plate_*_Myr.png'))
        all_continental_frames=sorted(continental_frames.glob('continental_*_Myr.png'))
        all_hydro_frames=sorted(hydro_frames.glob('surface_*_Myr.png'))
        def build_animation_if_possible(frame_paths, output_path):
            if len(frame_paths) >= 2:
                build_gif(frame_paths, output_path, int(evo.get('gif_frame_duration_ms',350)))
            elif len(frame_paths) == 1:
                print(f'WARNING: not writing {output_path.name}: only one animation frame exists. Use --frame-interval to save multiple frames.')

        build_animation_if_possible(all_frames, out/'history.gif')
        build_animation_if_possible(all_plate_frames, out/'plate_history.gif')
        build_animation_if_possible(all_continental_frames, out/'continental_history.gif')
        build_animation_if_possible(all_hydro_frames, out/'surface_history.gif')
        cont=state.crust_type==int(CrustType.CONTINENTAL);final_frac_field,final_vol_field=continental_material_fields(state,areas);final_diag=thermal_rows[-1]
        summary={'version':'0.14-hydrosphere','model':'v0.13 stable topology + passive fixed-inventory global hydrosphere with solved sea level','duration_myr':state.time_myr,'time_step_myr':dt,'checkpoint_resume':True,'system_age_start_myr':thp.system_age_at_start_myr,'system_age_final_myr':thermal.system_age_myr,'moon_mass_earth':mass,'moon_radius_km':radius,'surface_gravity_m_s2':grav,'rotation_period_hours':period,'primary_mass_jupiter':pmj,'initial_mantle_temperature_k':thp.initial_mantle_temperature_k,'final_mantle_temperature_k':thermal.mantle_temperature_k,'final_tectonic_activity_factor':thermal.tectonic_activity_factor,'final_thermal_lithosphere_thickness_km':thermal.thermal_lithosphere_thickness_km,'final_convective_heat_flux_w_m2':final_diag['convective_heat_flux_w_m2'],'final_radiogenic_heat_flux_w_m2':final_diag['radiogenic_heat_flux_w_m2'],'final_tidal_heat_flux_w_m2':final_diag['tidal_heat_flux_w_m2'],'initial_plate_count':len(proto.plates.plates),'final_plate_count':len(system.plates),'topology_event_count':len(events),'initial_continental_area_fraction':initial_cont_frac,'final_continental_area_fraction':float(np.sum(areas*final_frac_field)/np.sum(areas)),'final_visible_continental_area_fraction':float(np.sum(areas[cont])/np.sum(areas)),'initial_continental_volume_km3':initial_cont_vol,'final_continental_volume_km3':float(np.sum(final_vol_field)),'min_elevation_m':float(np.min(topo.elevation_m)),'max_elevation_m':float(np.max(topo.elevation_m)),'final_deepest_normal_ocean_m':float(relief_rows[-1].get('deepest_normal_ocean_m',0.0)) if relief_rows else 0.0,'final_deepest_trench_anomaly_m':float(relief_rows[-1].get('deepest_trench_anomaly_m',0.0)) if relief_rows else 0.0,'total_numerical_min_clip_cells':int(sum(int(r.get('numerical_min_clip_cells',0)) for r in relief_rows)),'total_numerical_max_clip_cells':int(sum(int(r.get('numerical_max_clip_cells',0)) for r in relief_rows)),'cumulative_generated_continental_area_km2':cycle.cumulative_generated_area_km2,'cumulative_recycled_continental_area_km2':cycle.cumulative_recycled_area_km2,'cumulative_generated_continental_volume_km3':cycle.cumulative_generated_volume_km3,'cumulative_recycled_continental_volume_km3':cycle.cumulative_recycled_volume_km3,'cumulative_breakup_area_km2':float(sum(float(r.get('tidally_rifted_continental_area_km2',0.0)) for r in lithosphere_rows)),'cumulative_continental_thinning_volume_km3':float(sum(float(r.get('continental_thinning_volume_km3',0.0)) for r in lithosphere_rows)),'final_mean_tidal_damage':float(lithosphere_rows[-1]['mean_tidal_damage']) if lithosphere_rows else 0.0,'final_max_rift_extension':float(lithosphere_rows[-1]['max_rift_extension']) if lithosphere_rows else 0.0,'final_largest_plate_fraction':float(late_rows[-1]['largest_plate_fraction']) if late_rows else 0.0,'final_mean_intraplate_stress':float(late_rows[-1]['mean_intraplate_stress']) if late_rows else 0.0,'final_mean_collision_seam_weakness':float(late_rows[-1]['mean_collision_seam_weakness']) if late_rows else 0.0,'final_mean_supercontinent_heat':float(late_rows[-1]['mean_supercontinent_heat']) if late_rows else 0.0,'late_rift_nucleation_count':int(sum(1 for r in late_rows if r.get('nucleated_rift'))),'cumulative_numerical_continental_volume_correction_km3':0.0,'max_abs_conservative_transport_volume_error_km3':float(max((abs(float(r.get('conservative_transport_volume_error_km3',0.0))) for r in lithosphere_rows),default=0.0)),'cumulative_collision_overflow_redistributed_volume_km3':float(sum(float(r.get('collision_overflow_redistributed_volume_km3',0.0)) for r in lithosphere_rows)),'max_collision_raw_thickness_km':float(max((float(r.get('collision_raw_max_thickness_km',0.0)) for r in lithosphere_rows),default=0.0)),'max_collision_post_redistribution_thickness_km':float(max((float(r.get('collision_post_redistribution_max_thickness_km',0.0)) for r in lithosphere_rows),default=0.0)),'subgrid_transport_cumulative_commit_count':int(transport_state.cumulative_commit_count),'final_transport_mean_residual_angle_deg':float(lithosphere_rows[-1].get('transport_mean_residual_angle_deg',0.0)) if lithosphere_rows else 0.0,'final_transport_max_residual_angle_deg':float(lithosphere_rows[-1].get('transport_max_residual_angle_deg',0.0)) if lithosphere_rows else 0.0,'max_transport_hold_age_myr':float(transport_state.max_hold_age_myr),'final_mantle_rms_speed_deg_per_myr':float(dyn_diag.mantle_rms_speed_deg_per_myr) if 'dyn_diag' in locals() else 0.0,'final_mean_plate_mantle_slip_deg_per_myr':float(dyn_diag.mean_plate_mantle_slip_deg_per_myr) if 'dyn_diag' in locals() else 0.0,'final_mean_continental_plate_speed_deg_per_myr':float(dyn_diag.mean_continental_plate_speed_deg_per_myr) if 'dyn_diag' in locals() else 0.0,'water_volume_km3':float(hydrosphere.water_volume_km3),'initial_sea_level_m':float(hydrosphere_rows[0]['sea_level_m']) if hydrosphere_rows else 0.0,'final_sea_level_m':float(last_hydro_diag.sea_level_m),'final_land_area_fraction':float(last_hydro_diag.land_area_fraction),'final_ocean_area_fraction':float(last_hydro_diag.ocean_area_fraction),'final_shallow_sea_area_fraction':float(last_hydro_diag.shallow_sea_area_fraction),'final_exposed_continental_material_area_fraction':float(last_hydro_diag.exposed_continental_material_area_fraction),'final_submerged_continental_material_area_fraction':float(last_hydro_diag.submerged_continental_material_area_fraction),'final_mean_ocean_depth_m':float(last_hydro_diag.mean_ocean_depth_m),'final_max_ocean_depth_m':float(last_hydro_diag.max_ocean_depth_m),'max_abs_hydrosphere_volume_error_km3':float(max((abs(float(r.get('volume_error_km3',0.0))) for r in hydrosphere_rows),default=0.0)),'notes':['Long integration was performed through deterministic checkpoint/resume segments aligned to the 4 Myr internal step.','Checkpoint stores full lithosphere, topography, fixed-grid mantle-flow field, plate dynamics, residual transport state, thermal state, continental-cycle memory and topology collision memory.','Oceanic background bathymetry uses a saturating plate-cooling relation; trench deflection is bounded and non-additive across duplicate mesh edges.','The +/-18/12 km elevation rails are numerical safety bounds only; calibrated runs should record zero clip cells.','Continental collision and permanent welding are separate stages; mature contacts mechanically couple before any merge is allowed.','v0.9.6 collision/weld calibration is retained unchanged.','v0.11 keeps plate ownership discrete but transports continental footprint fraction and volume independently, removing one-cell collision towers as a numerical artefact.','v0.12 expresses split/disconnect/microplate thresholds in km or km2 and late-rift path/band/seam scales in physical distance rather than mesh-cell counts.','v0.13 adds 20-Myr microplate-area hysteresis, explicit zero-area plate vanish/ID compaction, and invariants that mechanical collision coupling cannot change topology.','v0.14 conserves one global water inventory and solves an equipotential sea level from evolving basin geometry; hydrosphere is passive and does not yet affect erosion, loading, climate or tectonic forces.','Fresh v0.14 worlds initialize a mature oceanic age field from divergent-ridge distance instead of the legacy all-zero ocean age, removing the artificial whole-ocean early subsidence transient.']}
        with (out/'summary_v114.json').open('w',encoding='utf-8') as h:json.dump(summary,h,ensure_ascii=False,indent=2)
        print(json.dumps(summary,ensure_ascii=False,indent=2));print('Finalized:',out.resolve())

if __name__=='__main__': main()
