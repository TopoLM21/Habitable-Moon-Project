from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from tectonics.lithosphere import LithosphereState
from tectonics.mesh import SphereMesh
from tectonics.subduction_memory import SubductionMemoryState
from tectonics.volcanic_arc import VolcanicArcParameters, compute_volcanic_arc_forcing, projected_arc_center
from .raster import rasterize_cells


def _lonlat(xyz):
    xyz=np.asarray(xyz,float)
    return np.rad2deg(np.arctan2(xyz[:,1],xyz[:,0])),np.rad2deg(np.arcsin(np.clip(xyz[:,2],-1,1)))


def save_volcanic_arc_maps(mesh: SphereMesh, state: LithosphereState, memory: SubductionMemoryState,
                           radius_km: float, params: VolcanicArcParameters, output_dir: str|Path, dpi: int=180, boundaries=None):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    field,diag=compute_volcanic_arc_forcing(mesh,state,memory,radius_km,params,boundaries)
    lon_edges,lat_edges,data=rasterize_cells(mesh,field,width=720,height=360)
    fig,ax=plt.subplots(figsize=(12,5.8));sc=ax.pcolormesh(np.rad2deg(lon_edges),np.rad2deg(lat_edges),data,shading='auto')
    centers=[]
    for key in sorted(memory.zones):
        z=memory.zones[key]
        if (z.active and not z.broken_off) or (z.broken_off and z.post_breakoff_age_myr<=params.post_breakoff_duration_myr):
            centers.append(projected_arc_center(z,radius_km,params)[0])
    if centers:
        alon,alat=_lonlat(np.asarray(centers));ax.scatter(alon,alat,s=22,facecolors='none',edgecolors='k',linewidths=.7)
    fig.colorbar(sc,ax=ax,label='dimensionless arc forcing')
    ax.set(xlim=(-180,180),ylim=(-90,90),xlabel='longitude',ylabel='latitude',title=f'Slab-geometry volcanic-arc forcing | t={state.time_myr:.0f} Myr');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'volcanic_arc_forcing_final.png',dpi=dpi);plt.close(fig)
    return diag


def save_volcanic_arc_history(rows, path: str|Path):
    if not rows:return
    t=np.array([float(r.get('time_myr',0)) for r in rows])
    active=np.array([float(r.get('active_arc_zones',0)) for r in rows])
    pulses=np.array([float(r.get('post_breakoff_pulses',0)) for r in rows])
    dist=np.array([float(r.get('mean_trench_arc_distance_km',0)) for r in rows])
    depth=np.array([float(r.get('mean_target_slab_depth_km',0)) for r in rows])
    forcing=np.array([float(r.get('max_arc_forcing',0)) for r in rows])
    fig,ax=plt.subplots(figsize=(10,5));ax.plot(t,active,label='active arcs');ax.plot(t,pulses,label='post-breakoff pulses');ax.set(xlabel='Time (Myr)',ylabel='Count',title='Slab-geometry volcanic arcs');ax.legend(loc='upper left');ax.grid(alpha=.2)
    ax2=ax.twinx();ax2.plot(t,dist,label='mean trench-arc distance',alpha=.75);ax2.plot(t,depth,label='target slab depth',alpha=.75);ax2.set_ylabel('km');ax2.legend(loc='upper right')
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4));ax.plot(t,forcing);ax.set(xlabel='Time (Myr)',ylabel='max forcing',title='Volcanic-arc forcing amplitude');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(Path(path).with_name('volcanic_arc_forcing_history.png'),dpi=160);plt.close(fig)
