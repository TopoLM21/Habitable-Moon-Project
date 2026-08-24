from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def save_rollback_history(rows,path):
    if not rows:return
    t=np.array([r['time_myr'] for r in rows],float)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(t,[r['mean_rollback_rate_km_per_myr'] for r in rows],label='mean rollback')
    ax.plot(t,[r['max_rollback_rate_km_per_myr'] for r in rows],label='max rollback')
    ax.set(xlabel='time (Myr)',ylabel='km/Myr',title='Slab rollback / trench migration');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)

def save_backarc_history(rows,path):
    if not rows:return
    t=np.array([r['time_myr'] for r in rows],float)
    fig,ax=plt.subplots(figsize=(9,5));ax.plot(t,[r['backarc_forced_area_km2']/1e6 for r in rows])
    ax.set(xlabel='time (Myr)',ylabel='million km²',title='Back-arc extensional forcing area');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)

def save_rollback_maps(mesh,memory,backarc_forcing,out_dir,dpi=180):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    xyz=mesh.centroids;lon=np.rad2deg(np.arctan2(xyz[:,1],xyz[:,0]));lat=np.rad2deg(np.arcsin(np.clip(xyz[:,2],-1,1)))
    fig,ax=plt.subplots(figsize=(12,5.8));sc=ax.scatter(lon,lat,c=np.asarray(backarc_forcing,float),s=5)
    if memory.zones:
        p=np.array([memory.zones[k].trench_midpoint for k in sorted(memory.zones)]);plon=np.rad2deg(np.arctan2(p[:,1],p[:,0]));plat=np.rad2deg(np.arcsin(np.clip(p[:,2],-1,1)))
        rates=np.array([memory.zones[k].rollback_rate_km_per_myr for k in sorted(memory.zones)]);ax.scatter(plon,plat,s=15+35*np.clip(rates/2.0,0,1),facecolors='none',edgecolors='k',linewidths=.6)
    fig.colorbar(sc,ax=ax,label='back-arc extension forcing');ax.set(xlim=(-180,180),ylim=(-90,90),xlabel='longitude',ylabel='latitude',title='Rollback-driven back-arc extension');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'backarc_extension_forcing_final.png',dpi=dpi);plt.close(fig)

    fig,ax=plt.subplots(figsize=(12,5.8))
    if memory.zones:
        p=np.array([memory.zones[k].trench_midpoint for k in sorted(memory.zones)]);plon=np.rad2deg(np.arctan2(p[:,1],p[:,0]));plat=np.rad2deg(np.arcsin(np.clip(p[:,2],-1,1)))
        dist=np.array([memory.zones[k].rollback_distance_km for k in sorted(memory.zones)]);sc=ax.scatter(plon,plat,c=dist,s=30)
        fig.colorbar(sc,ax=ax,label='integrated effective rollback (km)')
    ax.set(xlim=(-180,180),ylim=(-90,90),xlabel='longitude',ylabel='latitude',title='Slab rollback / trench-migration memory');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'rollback_distance_final.png',dpi=dpi);plt.close(fig)
