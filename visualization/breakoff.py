from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _lonlat(xyz):
    xyz=np.asarray(xyz,float)
    return np.rad2deg(np.arctan2(xyz[:,1],xyz[:,0])), np.rad2deg(np.arcsin(np.clip(xyz[:,2],-1,1)))


def save_breakoff_history(rows,path):
    if not rows: return
    t=np.array([r['time_myr'] for r in rows],float)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(t,[r['active_necking_zones'] for r in rows],label='necking zones')
    ax.plot(t,[r['broken_off_tombstones'] for r in rows],label='recent breakoffs')
    ax.plot(t,[r['cumulative_breakoffs'] for r in rows],label='cumulative breakoffs')
    ax.set(xlabel='time (Myr)',ylabel='count',title='Slab-breakoff history')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def save_breakoff_maps(mesh,memory,out_dir,dpi=180):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    zs=[memory.zones[k] for k in sorted(memory.zones)]
    fig,ax=plt.subplots(figsize=(12,5.8))
    if zs:
        pts=np.asarray([z.trench_midpoint for z in zs],float);lon,lat=_lonlat(pts)
        damage=np.asarray([float(getattr(z,'breakoff_damage',0.0)) for z in zs])
        broken=np.asarray([bool(getattr(z,'broken_off',False)) for z in zs])
        sc=ax.scatter(lon[~broken],lat[~broken],c=damage[~broken],vmin=0,vmax=1,s=25+65*damage[~broken],alpha=.85)
        if np.any(~broken): fig.colorbar(sc,ax=ax,label='necking / breakoff damage')
        if np.any(broken): ax.scatter(lon[broken],lat[broken],marker='x',s=85,label='recent breakoff')
    ax.set(xlim=(-180,180),ylim=(-90,90),xlabel='longitude',ylabel='latitude',title='Collision-triggered slab breakoff')
    ax.grid(alpha=.2);ax.legend(loc='upper right') if any(bool(getattr(z,'broken_off',False)) for z in zs) else None
    fig.tight_layout();fig.savefig(out/'slab_breakoff_final.png',dpi=dpi);plt.close(fig)
