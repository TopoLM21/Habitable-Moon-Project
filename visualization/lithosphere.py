"""Visualization helpers for v0.4 lithosphere + tides."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tectonics.lithosphere import CrustType, LithosphereSnapshot, LithosphereState


def _lon_lat(points):
    lon=np.arctan2(points[:,1],points[:,0]); lat=np.arcsin(np.clip(points[:,2],-1,1)); return lon,lat


def save_lithosphere_frame(mesh, snapshot: LithosphereSnapshot, path, dpi=120):
    lon,lat=_lon_lat(mesh.centroids)
    fig=plt.figure(figsize=(13,7.2))
    ax1=fig.add_subplot(121,projection='mollweide')
    # type map: ocean=0, continent=1
    sc1=ax1.scatter(lon,lat,c=snapshot.state.crust_type,cmap='coolwarm',vmin=0,vmax=1,s=2.6,linewidths=0,rasterized=True)
    ax1.grid(True,alpha=.25); ax1.set_title('Crust type (blue oceanic / red continental)')
    ax2=fig.add_subplot(122,projection='mollweide')
    sc2=ax2.scatter(lon,lat,c=snapshot.state.tidal_damage,cmap='magma',vmin=0,vmax=1,s=2.6,linewidths=0,rasterized=True)
    ax2.grid(True,alpha=.25); ax2.set_title('Accumulated tidal weakening / damage')
    fig.colorbar(sc2,ax=ax2,orientation='horizontal',pad=.08,fraction=.05,label='damage index')
    d=snapshot.diagnostics
    subtitle=f"t={snapshot.state.time_myr:g} Myr"
    if d:
        subtitle += (f"  e={d.eccentricity:.6f}  max cyclic displacement={d.max_radial_displacement_m:.2f} m\n"
                     f"gaps={100*d.gap_fraction:.2f}% overlaps={100*d.overlap_fraction:.2f}%  "
                     f"continental area={100*d.continental_area_fraction:.1f}%")
    fig.suptitle('Moon Tectonics v0.4 — '+subtitle)
    fig.tight_layout(); fig.savefig(path,dpi=dpi,bbox_inches='tight'); plt.close(fig)


def _save_scalar(mesh, values, path, title, label, cmap='viridis', dpi=180, vmin=None, vmax=None):
    lon,lat=_lon_lat(mesh.centroids)
    fig=plt.figure(figsize=(12,6.5)); ax=fig.add_subplot(111,projection='mollweide')
    sc=ax.scatter(lon,lat,c=values,cmap=cmap,s=2.5,linewidths=0,rasterized=True,vmin=vmin,vmax=vmax)
    fig.colorbar(sc,ax=ax,orientation='horizontal',pad=.08,fraction=.05,label=label)
    ax.grid(True,alpha=.3); ax.set_title(title); fig.tight_layout(); fig.savefig(path,dpi=dpi); plt.close(fig)


def save_final_maps(mesh, state: LithosphereState, tidal_strain, weakening, output_dir: Path, dpi=180):
    output_dir=Path(output_dir)
    _save_scalar(mesh,state.crust_type,output_dir/'crust_type_final.png',f'Crust type — t={state.time_myr:g} Myr','0 oceanic / 1 continental','coolwarm',dpi,0,1)
    _save_scalar(mesh,state.crust_age_myr,output_dir/'crust_age_final.png',f'Crust age — t={state.time_myr:g} Myr','Myr','viridis',dpi)
    _save_scalar(mesh,state.crust_thickness_km,output_dir/'crust_thickness_final.png',f'Crust thickness — t={state.time_myr:g} Myr','km','plasma',dpi)
    _save_scalar(mesh,state.tidal_damage,output_dir/'tidal_damage_final.png',f'Accumulated tidal damage — t={state.time_myr:g} Myr','damage index','magma',dpi,0,1)
    _save_scalar(mesh,tidal_strain*1e6,output_dir/'tidal_strain_amplitude.png','Eccentricity-driven cyclic tidal strain amplitude','microstrain','cividis',dpi)
    _save_scalar(mesh,weakening,output_dir/'tidal_weakening_index.png','Geological tidal weakening forcing','relative to canonical e_rms','inferno',dpi)


def save_histories(rows, path, dpi=160):
    t=np.asarray([r['time_myr'] for r in rows])
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(t,100*np.asarray([r['continental_area_fraction'] for r in rows]),label='continental area %')
    ax.plot(t,100*np.asarray([r['gap_fraction'] for r in rows]),label='gap % before resolution')
    ax.plot(t,100*np.asarray([r['overlap_fraction'] for r in rows]),label='overlap % before resolution')
    ax.set_xlabel('Time, Myr'); ax.set_ylabel('% of surface'); ax.grid(True,alpha=.3); ax.legend(); ax.set_title('v0.4 surface geometry and continents')
    fig.tight_layout(); fig.savefig(path,dpi=dpi); plt.close(fig)


def save_tidal_history(rows,path,dpi=160):
    t=np.asarray([r['time_myr'] for r in rows]); e=np.asarray([r['eccentricity'] for r in rows]); disp=np.asarray([r['max_radial_displacement_m'] for r in rows]); dmg=np.asarray([r['mean_tidal_damage'] for r in rows])
    fig,ax=plt.subplots(figsize=(10,6)); ax.plot(t,disp,label='max cyclic radial displacement, m'); ax.plot(t,dmg,label='mean damage index')
    ax.set_xlabel('Time, Myr'); ax.grid(True,alpha=.3); ax.legend(loc='upper left'); ax.set_title('v0.4 eccentricity tide and lithosphere weakening')
    ax2=ax.twinx(); ax2.plot(t,e,linestyle='--',label='eccentricity'); ax2.set_ylabel('eccentricity')
    fig.tight_layout(); fig.savefig(path,dpi=dpi); plt.close(fig)


def build_gif(frame_paths, output_path, frame_duration_ms=350):
    images=[Image.open(p).convert('P',palette=Image.Palette.ADAPTIVE) for p in frame_paths]
    try:
        images[0].save(output_path,save_all=True,append_images=images[1:],duration=int(frame_duration_ms),loop=0,optimize=False)
    finally:
        for im in images: im.close()
