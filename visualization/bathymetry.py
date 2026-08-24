"""Diagnostics for v0.9.3 ocean-floor and trench calibration."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from tectonics.lithosphere import CrustType, LithosphereState
from tectonics.mesh import SphereMesh
from tectonics.topography import TopographyParameters, topography_components


def _lon_lat(points):
    lon=np.arctan2(points[:,1],points[:,0]); lat=np.arcsin(np.clip(points[:,2],-1.0,1.0)); return lon,lat


def save_bathymetry_components(mesh: SphereMesh, state: LithosphereState, boundaries, params: TopographyParameters, outdir: str|Path, dpi: int=180):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    comp=topography_components(mesh,state,boundaries,params)
    lon,lat=_lon_lat(mesh.centroids)
    ocean=state.crust_type==int(CrustType.OCEANIC)

    base=np.full(mesh.cell_count,np.nan,dtype=float); base[ocean]=comp['base'][ocean]
    fig=plt.figure(figsize=(12,6.5)); ax=fig.add_subplot(111,projection='mollweide')
    sc=ax.scatter(lon,lat,c=base,cmap='viridis',s=2.2,linewidths=0,rasterized=True)
    fig.colorbar(sc,ax=ax,orientation='horizontal',pad=.08,fraction=.05,label='Normal ocean-floor elevation, m')
    ax.grid(True,alpha=.3); ax.set_title('v0.9.3 normal oceanic bathymetry (before trench forcing)')
    fig.tight_layout(); fig.savefig(out/'normal_ocean_bathymetry_final.png',dpi=dpi); plt.close(fig)

    trench=comp['trench_depth']
    fig=plt.figure(figsize=(12,6.5)); ax=fig.add_subplot(111,projection='mollweide')
    sc=ax.scatter(lon,lat,c=trench,cmap='magma',s=2.2,linewidths=0,rasterized=True,vmin=0)
    fig.colorbar(sc,ax=ax,orientation='horizontal',pad=.08,fraction=.05,label='Extra trench depth, m')
    ax.grid(True,alpha=.3); ax.set_title('v0.9.3 bounded subduction-trench anomaly')
    fig.tight_layout(); fig.savefig(out/'trench_anomaly_final.png',dpi=dpi); plt.close(fig)


def save_bathymetry_limit_history(rows: list[dict], path: str|Path, dpi: int=170):
    if not rows: return
    t=np.asarray([r['time_myr'] for r in rows],dtype=float)
    min_e=np.asarray([r['min_elevation_m'] for r in rows],dtype=float)
    base=np.asarray([r.get('deepest_normal_ocean_m',0.0) for r in rows],dtype=float)
    trench=np.asarray([r.get('deepest_trench_anomaly_m',0.0) for r in rows],dtype=float)
    clips=np.asarray([r.get('numerical_min_clip_cells',0) for r in rows],dtype=float)
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(t,min_e/1000.0,label='deepest realized topography')
    ax.plot(t,base/1000.0,label='deepest normal ocean floor')
    ax.plot(t,(base+trench)/1000.0,linestyle='--',label='base + deepest trench anomaly (diagnostic envelope)')
    ax.set_xlabel('Time, Myr'); ax.set_ylabel('Elevation, km'); ax.grid(True,alpha=.3); ax.legend(loc='lower left')
    ax.set_title('v0.9.3 bathymetry remains emergent, not floor-limited')
    ax2=ax.twinx(); ax2.plot(t,clips,label='safety-floor clip cells',linestyle=':'); ax2.set_ylabel('Cells clipped at numerical floor')
    fig.tight_layout(); fig.savefig(path,dpi=dpi); plt.close(fig)
