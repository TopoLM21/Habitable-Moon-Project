#!/usr/bin/env python3
"""Run Moon Tectonics v0.4: continents + improved transport + eccentricity tides."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from tectonics.evolution import snapshot_times
from tectonics.lithosphere import LithosphereSnapshot, advance_lithosphere, boundary_records_for_state, initialize_lithosphere
from tectonics.simulation import build_prototype, load_config
from tectonics.tides import eccentricity_history_from_config, tidal_strain_amplitude
from visualization.lithosphere import build_gif,save_final_maps,save_histories,save_lithosphere_frame,save_tidal_history


def args():
    p=argparse.ArgumentParser(description='v0.4 lithosphere evolution')
    p.add_argument('--config',default='configs/canonical_moon.yaml'); p.add_argument('--output',default=None)
    p.add_argument('--duration',type=float,default=None); p.add_argument('--interval',type=float,default=None); p.add_argument('--dt',type=float,default=None); p.add_argument('--no-gif',action='store_true'); return p.parse_args()

def substeps(start,end,dt):
    out=[]; t=start
    while t+dt<end-1e-12: out.append(dt); t+=dt
    if end>t+1e-12: out.append(end-t)
    return out

def main():
    a=args(); cfg=load_config(a.config); proto=build_prototype(cfg); lc=cfg['lithosphere']; tc=cfg['tides']; evo=cfg['lithosphere_evolution']
    duration=float(a.duration if a.duration is not None else evo['duration_myr']); interval=float(a.interval if a.interval is not None else evo['frame_interval_myr']); dt=float(a.dt if a.dt is not None else evo['time_step_myr'])
    out=Path(a.output or cfg['output'].get('lithosphere_directory','outputs_v04')); frames=out/'frames'; frames.mkdir(parents=True,exist_ok=True)
    ecc=eccentricity_history_from_config(cfg,Path(a.config).resolve().parent.parent)
    state=initialize_lithosphere(proto.mesh,proto.plates,float(lc['initial_continental_fraction']),int(lc['continental_nuclei']),float(lc['oceanic_thickness_km']),float(lc['continental_thickness_km']),float(lc['initial_continental_age_myr']))
    radius=float(cfg['moon']['radius_km']); grav=float(cfg['moon']['surface_gravity_m_s2']); period=float(cfg['moon']['rotation_period_hours'])
    times=snapshot_times(duration,interval); rows=[]; frame_paths=[]; last_strain=np.zeros(proto.mesh.cell_count); last_weak=np.zeros(proto.mesh.cell_count); last_diag=None
    for fi,target in enumerate(times):
        if target>state.time_myr:
            for dti in substeps(state.time_myr,float(target),dt):
                state,last_strain,last_weak,last_diag=advance_lithosphere(proto.mesh,proto.plates,state,float(dti),radius,grav,period,ecc,primary_mass_jupiter=float(cfg.get('primary',{}).get('mass_jupiter',5.0)),love_h2=float(tc['love_h2']),reference_eccentricity=float(tc['eccentricity_rms']),oceanic_thickness_km=float(lc['oceanic_thickness_km']),continental_thickness_km=float(lc['continental_thickness_km']),max_continental_thickness_km=float(lc['max_continental_thickness_km']),collision_accretion_fraction=float(lc['collision_accretion_fraction']),tidal_damage_rate_per_myr=float(tc['damage_rate_per_myr']),tidal_damage_relaxation_myr=float(tc['damage_relaxation_myr']),continental_rift_damage_threshold=float(tc['continental_rift_damage_threshold']))
                rows.append({k:getattr(last_diag,k) for k in last_diag.__dataclass_fields__})
        else:
            e=ecc.at(0.0); last_strain=tidal_strain_amplitude(proto.mesh.centroids,e,radius,grav,period,float(cfg.get('primary',{}).get('mass_jupiter',5.0)),float(tc['love_h2'])); ref=tidal_strain_amplitude(proto.mesh.centroids,float(tc['eccentricity_rms']),radius,grav,period,float(cfg.get('primary',{}).get('mass_jupiter',5.0)),float(tc['love_h2'])); last_weak=np.clip(last_strain/max(float(np.max(ref)),1e-30),0,2)
        boundaries=boundary_records_for_state(proto.mesh,state,proto.plates,radius,float(cfg['classification']['normal_threshold_km_per_myr']),float(cfg['classification']['inactive_speed_km_per_myr']))
        snap=LithosphereSnapshot(state,boundaries,last_strain,last_weak,last_diag); fp=frames/f'frame_{fi:04d}_{target:08.3f}_Myr.png'; save_lithosphere_frame(proto.mesh,snap,fp,int(cfg['output'].get('lithosphere_dpi',120))); frame_paths.append(fp)
        if last_diag: print(f"t={state.time_myr:7.2f} Myr | cont={100*last_diag.continental_area_fraction:5.1f}% | gaps={100*last_diag.gap_fraction:5.2f}% overlaps={100*last_diag.overlap_fraction:5.2f}% | e={last_diag.eccentricity:.6f} | tide={last_diag.max_radial_displacement_m:.2f} m")
    save_final_maps(proto.mesh,state,last_strain,last_weak,out,int(cfg['output'].get('dpi',180)))
    if rows:
        save_histories(rows,out/'surface_history.png'); save_tidal_history(rows,out/'tidal_history.png')
        with (out/'lithosphere_history.csv').open('w',newline='',encoding='utf-8') as h:
            w=csv.DictWriter(h,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if not a.no_gif: build_gif(frame_paths,out/'history.gif',int(evo.get('gif_frame_duration_ms',350)))
    surface=float(np.sum(proto.mesh.physical_cell_areas_km2(radius)))
    summary={'version':'0.4','model':'backward semi-Lagrangian rigid plates + oceanic/continental crust + eccentricity-tide fatigue','duration_myr':duration,'time_step_myr':dt,'frame_interval_myr':interval,'mesh_cells':proto.mesh.cell_count,'plate_count':len(proto.plates.plates),'surface_area_km2':surface,'eccentricity_source':('csv' if tc.get('eccentricity_history_csv') else 'constant rms'),'eccentricity_rms_fallback':float(tc['eccentricity_rms']),'canonical_primary_mass_jupiter':float(cfg.get('primary',{}).get('mass_jupiter',5.0)),'love_h2_assumption':float(tc['love_h2']),'initial_continental_fraction':float(lc['initial_continental_fraction']),'final_continental_fraction':float(last_diag.continental_area_fraction if last_diag else lc['initial_continental_fraction']),'final_mean_tidal_damage':float(np.mean(state.tidal_damage)),'final_max_tidal_damage':float(np.max(state.tidal_damage)),'final_max_tidal_displacement_m':float(np.max(last_strain)*radius*1000.0),'mean_gap_fraction':float(np.mean([r['gap_fraction'] for r in rows])) if rows else 0,'mean_overlap_fraction':float(np.mean([r['overlap_fraction'] for r in rows])) if rows else 0,'total_tidally_rifted_continental_area_km2':float(np.sum([r['tidally_rifted_continental_area_km2'] for r in rows])) if rows else 0,'total_continental_collision_area_km2':float(np.sum([r['continental_collision_area_km2'] for r in rows])) if rows else 0,'mean_abs_numerical_continental_area_correction_km2_per_step':float(np.mean(np.abs([r['numerical_continental_area_correction_km2'] for r in rows]))) if rows else 0,'notes':['v0.4 fixes the major v0.3 forward-remap artifact with backward semi-Lagrangian coverage.','Continental crust resists oceanic subduction; continent-continent overlap thickens crust.','Eccentricity drives a degree-2 cyclic tidal strain amplitude and accumulated lithosphere damage.','If tides.eccentricity_history_csv is set, eccentricity is interpolated from the N-body history instead of using the RMS fallback.','Plate Euler poles are still fixed; self-consistent force-driven plate motion remains future work.']}
    with (out/'summary_v04.json').open('w',encoding='utf-8') as h: json.dump(summary,h,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2)); print('Saved:',out.resolve())
if __name__=='__main__': main()
