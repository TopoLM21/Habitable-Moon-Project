#!/usr/bin/env python3
"""Run Moon Tectonics v0.5: lithosphere + eccentricity tides + evolving Euler poles."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

from tectonics.dynamics import DynamicsParameters, center_net_rotation, update_plate_dynamics
from tectonics.evolution import snapshot_times
from tectonics.lithosphere import LithosphereSnapshot, advance_lithosphere, boundary_records_for_state, initialize_lithosphere
from tectonics.plates import PlateSystem
from tectonics.simulation import build_prototype, load_config
from tectonics.tides import eccentricity_history_from_config, semi_major_axis_from_period, M_JUPITER, tidal_strain_amplitude
from visualization.dynamics import build_gif,save_dynamics_frame,save_euler_poles,save_force_history,save_speed_history
from visualization.lithosphere import save_final_maps,save_histories,save_tidal_history


def parse_args():
    p=argparse.ArgumentParser(description='v0.5 effective force-driven plate dynamics')
    p.add_argument('--config',default='configs/canonical_moon.yaml'); p.add_argument('--output',default=None)
    p.add_argument('--duration',type=float,default=None); p.add_argument('--interval',type=float,default=None); p.add_argument('--dt',type=float,default=None); p.add_argument('--no-gif',action='store_true'); return p.parse_args()


def substeps(start,end,dt):
    out=[]; t=start
    while t+dt<end-1e-12: out.append(dt); t+=dt
    if end>t+1e-12: out.append(end-t)
    return out


def _params(cfg):
    d=cfg['plate_dynamics']; fields=DynamicsParameters.__dataclass_fields__
    return DynamicsParameters(**{k:d[k] for k in fields if k in d})


def main():
    a=parse_args(); cfg=load_config(a.config); proto=build_prototype(cfg); lc=cfg['lithosphere']; tc=cfg['tides']; evo=cfg['dynamics_evolution']; dynp=_params(cfg)
    duration=float(a.duration if a.duration is not None else evo['duration_myr']); interval=float(a.interval if a.interval is not None else evo['frame_interval_myr']); dt=float(a.dt if a.dt is not None else evo['time_step_myr'])
    out=Path(a.output or cfg['output'].get('dynamics_directory','outputs_v05')); frames=out/'frames'; frames.mkdir(parents=True,exist_ok=True)
    ecc=eccentricity_history_from_config(cfg,Path(a.config).resolve().parent.parent)
    state=initialize_lithosphere(proto.mesh,proto.plates,float(lc['initial_continental_fraction']),int(lc['continental_nuclei']),float(lc['oceanic_thickness_km']),float(lc['continental_thickness_km']),float(lc['initial_continental_age_myr']))
    baseline_system=center_net_rotation(proto.mesh,state,proto.plates,float(cfg['moon']['radius_km'])) if dynp.remove_net_rotation else proto.plates; current_system=PlateSystem(cell_plate=state.cell_plate.copy(),plates=baseline_system.plates)
    radius=float(cfg['moon']['radius_km']); grav=float(cfg['moon']['surface_gravity_m_s2']); period=float(cfg['moon']['rotation_period_hours']); pmj=float(cfg.get('primary',{}).get('mass_jupiter',5.0))
    normal_thr=float(cfg['classification']['normal_threshold_km_per_myr']); inactive=float(cfg['classification']['inactive_speed_km_per_myr'])
    times=snapshot_times(duration,interval); lith_rows=[]; dyn_rows=[]; frame_paths=[]; last_strain=np.zeros(proto.mesh.cell_count); last_weak=np.zeros(proto.mesh.cell_count); last_lith=None; last_dyn=None

    for fi,target in enumerate(times):
        if target>state.time_myr:
            for dti in substeps(state.time_myr,float(target),dt):
                current_system,last_dyn,_,_=update_plate_dynamics(proto.mesh,state,current_system,baseline_system,radius,float(dti),normal_thr,inactive,dynp)
                state,last_strain,last_weak,last_lith=advance_lithosphere(proto.mesh,current_system,state,float(dti),radius,grav,period,ecc,primary_mass_jupiter=pmj,love_h2=float(tc['love_h2']),reference_eccentricity=float(tc['eccentricity_rms']),oceanic_thickness_km=float(lc['oceanic_thickness_km']),continental_thickness_km=float(lc['continental_thickness_km']),max_continental_thickness_km=float(lc['max_continental_thickness_km']),collision_accretion_fraction=float(lc['collision_accretion_fraction']),tidal_damage_rate_per_myr=float(tc['damage_rate_per_myr']),tidal_damage_relaxation_myr=float(tc['damage_relaxation_myr']),continental_rift_damage_threshold=float(tc['continental_rift_damage_threshold']))
                current_system=PlateSystem(cell_plate=state.cell_plate.copy(),plates=current_system.plates)
                last_dyn.time_myr=state.time_myr
                lith_rows.append({k:getattr(last_lith,k) for k in last_lith.__dataclass_fields__}); dyn_rows.append({k:getattr(last_dyn,k) for k in last_dyn.__dataclass_fields__})
        else:
            e=ecc.at(0.0); last_strain=tidal_strain_amplitude(proto.mesh.centroids,e,radius,grav,period,pmj,float(tc['love_h2'])); ref=tidal_strain_amplitude(proto.mesh.centroids,float(tc['eccentricity_rms']),radius,grav,period,pmj,float(tc['love_h2'])); last_weak=np.clip(last_strain/max(float(np.max(ref)),1e-30),0,2)
        boundaries=boundary_records_for_state(proto.mesh,state,current_system,radius,normal_thr,inactive)
        fp=frames/f'frame_{fi:04d}_{target:08.3f}_Myr.png'; save_dynamics_frame(proto.mesh,state,current_system,boundaries,last_dyn,fp,int(cfg['output'].get('dynamics_dpi',120))); frame_paths.append(fp)
        if last_dyn and last_lith:
            print(f"t={state.time_myr:7.2f} Myr | speed={last_dyn.mean_speed_deg_per_myr:.3f}°/Myr max={last_dyn.max_speed_deg_per_myr:.3f} | pole turn={last_dyn.mean_axis_turn_deg:.2f}° | cont={100*last_lith.continental_area_fraction:5.1f}%")

    save_final_maps(proto.mesh,state,last_strain,last_weak,out,int(cfg['output'].get('dpi',180)))
    save_euler_poles(current_system,out/'euler_poles_final.png',title=f'v0.5 final Euler poles — t={state.time_myr:g} Myr')
    if dyn_rows:
        save_speed_history(dyn_rows,out/'plate_speed_history.png'); save_force_history(dyn_rows,out/'force_balance_history.png')
        with (out/'plate_dynamics_history.csv').open('w',newline='',encoding='utf-8') as h:
            w=csv.DictWriter(h,fieldnames=list(dyn_rows[0].keys())); w.writeheader(); w.writerows(dyn_rows)
    if lith_rows:
        save_histories(lith_rows,out/'surface_history.png'); save_tidal_history(lith_rows,out/'tidal_history.png')
        with (out/'lithosphere_history.csv').open('w',newline='',encoding='utf-8') as h:
            w=csv.DictWriter(h,fieldnames=list(lith_rows[0].keys())); w.writeheader(); w.writerows(lith_rows)
    if not a.no_gif: build_gif(frame_paths,out/'history.gif',int(evo.get('gif_frame_duration_ms',350)))

    a_m=semi_major_axis_from_period(pmj*M_JUPITER,period)
    initial_speeds=np.rad2deg(np.abs([p.angular_speed_rad_per_myr for p in baseline_system.plates])); final_speeds=np.rad2deg(np.abs([p.angular_speed_rad_per_myr for p in current_system.plates]))
    summary={
        'version':'0.5','model':'v0.4 lithosphere + quasi-static effective slab/ridge/collision/mantle plate dynamics','duration_myr':duration,'time_step_myr':dt,'frame_interval_myr':interval,'mesh_cells':proto.mesh.cell_count,'plate_count':len(current_system.plates),
        'canonical_primary_mass_jupiter':pmj,'canonical_primary_radius_jupiter':float(cfg.get('primary',{}).get('radius_jupiter',1.03)),'configured_semi_major_axis_primary_radii':float(cfg.get('orbit',{}).get('semi_major_axis_primary_radii',10.5)),'orbital_period_hours':period,'semi_major_axis_km_from_period_and_primary_mass':float(a_m/1000.0),'eccentricity_source':('csv' if tc.get('eccentricity_history_csv') else 'constant rms'),'eccentricity_rms_fallback':float(tc['eccentricity_rms']),'love_h2_assumption':float(tc['love_h2']),
        'initial_mean_plate_speed_deg_per_myr':float(np.mean(initial_speeds)),'final_mean_plate_speed_deg_per_myr':float(np.mean(final_speeds)),'final_max_plate_speed_deg_per_myr':float(np.max(final_speeds)),
        'mean_euler_pole_turn_deg_per_step':float(np.mean([r['mean_axis_turn_deg'] for r in dyn_rows])) if dyn_rows else 0.0,'max_recorded_euler_pole_turn_deg':float(np.max([r['max_axis_turn_deg'] for r in dyn_rows])) if dyn_rows else 0.0,
        'final_continental_fraction':float(last_lith.continental_area_fraction if last_lith else lc['initial_continental_fraction']),'final_max_tidal_displacement_m':float(np.max(last_strain)*radius*1000.0),
        'dynamics_parameters':{k:getattr(dynp,k) for k in DynamicsParameters.__dataclass_fields__},
        'notes':['Primary mass corrected to the canonical 5 M_J from the orbital-system configuration.','At fixed 47 h period, leading GM/a^3 tidal amplitude is nearly independent of primary mass because Kepler sets GM/a^3 ~ n^2; the semi-major axis is not independent and is recomputed.','Euler angular velocities now evolve after every geological step rather than remaining fixed.','Slab pull and ridge push are directional geometric proxies; collision/transform terms add resistance; tidal damage reduces boundary resistance.','Force coefficients map effective geological driving to deg/Myr and are calibration parameters, not literal SI forces.','No full 3D mantle convection or explicit slab geometry yet.']}
    with (out/'summary_v05.json').open('w',encoding='utf-8') as h: json.dump(summary,h,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2)); print('Saved:',out.resolve())
if __name__=='__main__': main()
