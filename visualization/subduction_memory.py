from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _lonlat(xyz):
    xyz=np.asarray(xyz,float); return np.rad2deg(np.arctan2(xyz[:,1],xyz[:,0])), np.rad2deg(np.arcsin(np.clip(xyz[:,2],-1,1)))


def save_subduction_memory_maps(mesh,memory,out_dir,dpi=180):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(12,5.8))
    if memory.zones:
        pts=np.array([z.trench_midpoint for z in memory.zones.values()]); lon,lat=_lonlat(pts)
        lengths=np.array([z.slab_length_km for z in memory.zones.values()]); active=np.array([z.active for z in memory.zones.values()])
        sc=ax.scatter(lon,lat,c=lengths,s=18+55*active,alpha=.8)
        fig.colorbar(sc,ax=ax,label='remembered slab length (km)')
    ax.set(xlim=(-180,180),ylim=(-90,90),xlabel='longitude',ylabel='latitude',title='Subduction memory: trench/slab states')
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'subduction_memory_final.png',dpi=dpi); plt.close(fig)


def save_subduction_memory_history(rows,path):
    if not rows: return
    t=np.array([r['time_myr'] for r in rows],float)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(t,[r['active_zone_count'] for r in rows],label='active')
    ax.plot(t,[r['residual_zone_count'] for r in rows],label='residual')
    ax.set(xlabel='time (Myr)',ylabel='zones',title='Subduction-memory history'); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)
