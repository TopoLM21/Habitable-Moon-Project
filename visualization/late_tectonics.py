"""Diagnostics for v0.9.4 late boundary nucleation."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tectonics.lithosphere import LithosphereState
from tectonics.mesh import SphereMesh


def _lon_lat(points):
    return np.arctan2(points[:,1],points[:,0]), np.arcsin(np.clip(points[:,2],-1.0,1.0))


def _save_field(mesh: SphereMesh, field: np.ndarray, title: str, label: str, path: str|Path, dpi:int=180):
    lon,lat=_lon_lat(mesh.centroids)
    fig=plt.figure(figsize=(12,6.5));ax=fig.add_subplot(111,projection='mollweide')
    sc=ax.scatter(lon,lat,c=field,cmap='viridis',s=2.4,linewidths=0,rasterized=True)
    fig.colorbar(sc,ax=ax,orientation='horizontal',pad=.08,fraction=.05,label=label)
    ax.grid(True,alpha=.3);ax.set_title(title);fig.tight_layout();fig.savefig(path,dpi=dpi);plt.close(fig)


def save_late_tectonic_maps(mesh: SphereMesh,state:LithosphereState,out_dir:str|Path,dpi:int=180):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);n=mesh.cell_count
    seam=np.zeros(n) if state.collision_seam_weakness is None else state.collision_seam_weakness
    stress=np.zeros(n) if state.intraplate_stress is None else state.intraplate_stress
    heat=np.zeros(n) if state.supercontinent_heat is None else state.supercontinent_heat
    _save_field(mesh,seam,'Inherited collision-seam weakness','seam weakness',out/'collision_seam_weakness_final.png',dpi)
    _save_field(mesh,stress,'Accumulated intraplate stress','normalized stress',out/'intraplate_stress_final.png',dpi)
    _save_field(mesh,heat,'Supercontinent thermal-cap memory','normalized heat memory',out/'supercontinent_heat_final.png',dpi)


def save_late_history(rows:list[dict],path:str|Path,dpi:int=160):
    if not rows:return
    t=np.asarray([r['time_myr'] for r in rows],float)
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(t,[r['largest_plate_fraction'] for r in rows],label='largest plate fraction')
    ax.plot(t,[r['mean_intraplate_stress'] for r in rows],label='mean intraplate stress')
    ax.plot(t,[r['mean_collision_seam_weakness'] for r in rows],label='mean seam weakness')
    ax.plot(t,[r['mean_supercontinent_heat'] for r in rows],label='mean supercontinent heat')
    ax.plot(t,[r['old_ocean_fraction'] for r in rows],label='ocean > age threshold')
    ax.set_xlabel('Time, Myr');ax.set_ylabel('fraction / normalized field');ax.set_title('v0.9.4 late-boundary nucleation diagnostics');ax.grid(True,alpha=.3);ax.legend()
    fig.tight_layout();fig.savefig(path,dpi=dpi);plt.close(fig)


def save_nucleation_history(rows:list[dict],path:str|Path,dpi:int=160):
    if not rows:return
    t=np.asarray([r['time_myr'] for r in rows],float)
    active=np.asarray([r['active_internal_rift_cells'] for r in rows],float)
    nuc=np.asarray([1.0 if r['nucleated_rift'] else 0.0 for r in rows])
    fig,ax=plt.subplots(figsize=(10,6));ax.plot(t,active,label='maturing internal-rift cells')
    ax.set_xlabel('Time, Myr');ax.set_ylabel('cell count');ax.grid(True,alpha=.3)
    ax2=ax.twinx();ax2.scatter(t[nuc>0],nuc[nuc>0],marker='x',label='new rift nucleated');ax2.set_ylim(0,1.2);ax2.set_ylabel('nucleation event')
    ax.set_title('Internally nucleated rifts');fig.tight_layout();fig.savefig(path,dpi=dpi);plt.close(fig)
