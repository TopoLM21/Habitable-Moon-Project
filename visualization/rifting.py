"""Diagnostics for v0.9.2 progressive continental rifting."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tectonics.lithosphere import CrustType, LithosphereState
from tectonics.mesh import SphereMesh


def _lon_lat(points: np.ndarray):
    lon=np.arctan2(points[:,1],points[:,0]);lat=np.arcsin(np.clip(points[:,2],-1.0,1.0));return lon,lat


def save_rift_maps(mesh: SphereMesh, state: LithosphereState, out_dir: str | Path, dpi: int = 180) -> None:
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    lon,lat=_lon_lat(mesh.centroids)
    ext=np.zeros(mesh.cell_count) if state.rift_extension is None else np.asarray(state.rift_extension,float)
    age=np.zeros(mesh.cell_count) if state.extension_age_myr is None else np.asarray(state.extension_age_myr,float)
    cont=state.crust_type==int(CrustType.CONTINENTAL)

    for values,name,label in [
        (ext,'rift_extension_final.png','Accumulated extensional rift index'),
        (age,'extension_age_final.png','Persistent extension memory, Myr'),
    ]:
        fig=plt.figure(figsize=(12,6.5));ax=fig.add_subplot(111,projection='mollweide')
        shown=np.where(cont,values,np.nan)
        sc=ax.scatter(lon,lat,c=shown,cmap='viridis',s=2.4,linewidths=0,rasterized=True)
        fig.colorbar(sc,ax=ax,orientation='horizontal',pad=0.08,fraction=0.05,label=label)
        ax.grid(True,alpha=0.3);ax.set_title(f'v0.9.2 continental rifting — t = {state.time_myr:g} Myr')
        fig.tight_layout();fig.savefig(out/name,dpi=dpi);plt.close(fig)


def save_rift_history(rows: list[dict], path: str | Path, dpi: int = 160) -> None:
    if not rows:return
    t=np.asarray([float(r['time_myr']) for r in rows])
    active=np.asarray([float(r.get('actively_extending_continental_area_km2',0.0)) for r in rows])/1e6
    breakup=np.asarray([float(r.get('tidally_rifted_continental_area_km2',0.0)) for r in rows])/1e6
    maxext=np.asarray([float(r.get('max_rift_extension',0.0)) for r in rows])
    damage=np.asarray([float(r.get('mean_tidal_damage',0.0)) for r in rows])
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(t,active,label='actively extending continental area')
    ax.plot(t,np.cumsum(breakup),label='cumulative completed breakup area')
    ax.set_xlabel('Time, Myr');ax.set_ylabel('Area, million km²');ax.grid(True,alpha=0.3);ax.legend(loc='upper left')
    ax2=ax.twinx();ax2.plot(t,maxext,linestyle='--',label='max rift extension');ax2.plot(t,damage,linestyle=':',label='mean tidal damage')
    ax2.set_ylabel('Dimensionless index');ax2.legend(loc='upper right')
    ax.set_title('v0.9.2 progressive rifting diagnostics')
    fig.tight_layout();fig.savefig(path,dpi=dpi);plt.close(fig)
