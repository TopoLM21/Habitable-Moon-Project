"""v0.7 topology diagnostics and frames."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tectonics.kinematics import BoundaryType
from .raster import rasterize_cells


def _lon_lat(points):
    return np.arctan2(points[:,1],points[:,0]), np.arcsin(np.clip(points[:,2],-1,1))


def save_topology_frame(mesh,state,topo,system,boundaries,topo_diag,path,dpi=120):
    fig=plt.figure(figsize=(12,6.8)); ax=fig.add_subplot(111,projection='mollweide')
    lim=max(6000.0,float(np.percentile(np.abs(topo.elevation_m),99)))
    xe,ye,grid=rasterize_cells(mesh,topo.elevation_m)
    sc=ax.pcolormesh(xe,ye,grid,cmap='terrain',vmin=-lim,vmax=lim,shading='auto',rasterized=True)
    fig.colorbar(sc,ax=ax,orientation='horizontal',pad=.08,fraction=.05,label='Elevation relative to reference datum, m')
    for kind in (BoundaryType.DIVERGENT,BoundaryType.CONVERGENT,BoundaryType.TRANSFORM):
        pts=np.asarray([b.midpoint for b in boundaries if b.boundary_type==kind])
        if len(pts):
            x,y=_lon_lat(pts); ax.scatter(x,y,s=2.8,linewidths=0,alpha=.55,label=kind.name.lower())
    title=f'Dynamic plate topology — t={state.time_myr:g} Myr | plates={len(system.plates)}'
    if topo_diag:
        title+=f"\nplate cells min/mean/max={topo_diag.min_plate_cells}/{topo_diag.mean_plate_cells:.0f}/{topo_diag.max_plate_cells}"
        if topo_diag.topology_changed: title+=' | TOPOLOGY EVENT'
    ax.set_title(title);ax.grid(True,alpha=.3);ax.legend(loc='lower center',ncol=3,bbox_to_anchor=(.5,-.15));fig.tight_layout();fig.savefig(path,dpi=dpi,bbox_inches='tight');plt.close(fig)



def save_plate_history_frame(mesh,state,system,path,dpi=120):
    """Filled plate map with representative Euler-motion arrows."""
    fig=plt.figure(figsize=(12,6.8));ax=fig.add_subplot(111,projection='mollweide')
    xe,ye,grid=rasterize_cells(mesh,state.cell_plate)
    ax.pcolormesh(xe,ye,grid,cmap='tab20',shading='auto',rasterized=True)
    qx=[];qy=[];qu=[];qv=[];speeds=[]
    for pid,plate in enumerate(system.plates):
        cells=np.flatnonzero(state.cell_plate==pid)
        if not len(cells):continue
        p=np.mean(mesh.centroids[cells],axis=0);p/=max(np.linalg.norm(p),1e-30)
        lon=np.arctan2(p[1],p[0]);lat=np.arcsin(np.clip(p[2],-1,1))
        vel=float(plate.angular_speed_rad_per_myr)*np.cross(np.asarray(plate.euler_axis,dtype=float),p)
        east=np.array([-np.sin(lon),np.cos(lon),0.0]);north=np.array([-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)])
        latdot=float(np.dot(vel,north));londot=float(np.dot(vel,east))/max(float(np.cos(lat)),0.15)
        # Display a 20-Myr displacement vector; direction matters more than exact length.
        qx.append(lon);qy.append(lat);qu.append(londot*20.0);qv.append(latdot*20.0);speeds.append(abs(float(plate.angular_speed_rad_per_myr)))
    if qx:
        ax.quiver(qx,qy,qu,qv,angles='xy',scale_units='xy',scale=1.0,width=0.003)
    ax.grid(True,alpha=.3)
    mean_speed=np.rad2deg(np.mean(speeds)) if speeds else 0.0
    ax.set_title(f'Plate partition — t={state.time_myr:g} Myr | plates={len(system.plates)} | mean |ω|={mean_speed:.2f}°/Myr')
    fig.tight_layout();fig.savefig(path,dpi=dpi,bbox_inches='tight');plt.close(fig)

def save_plate_map(mesh,state,system,path,dpi=180):
    lon,lat=_lon_lat(mesh.centroids);fig=plt.figure(figsize=(12,6.5));ax=fig.add_subplot(111,projection='mollweide')
    sc=ax.scatter(lon,lat,c=state.cell_plate,cmap='tab20',s=2.5,linewidths=0,rasterized=True)
    ax.grid(True,alpha=.3);ax.set_title(f'Final dynamic plate partition — {len(system.plates)} plates at t={state.time_myr:g} Myr');fig.tight_layout();fig.savefig(path,dpi=dpi);plt.close(fig)


def save_plate_count_history(rows,path,dpi=160):
    if not rows:return
    t=np.asarray([r['time_myr'] for r in rows]);n=np.asarray([r['plate_count_after'] for r in rows])
    fig,ax=plt.subplots(figsize=(10,5.8));ax.step(t,n,where='post',label='plate count')
    for r in rows:
        if r['split_events'] or r['merge_events'] or r['absorbed_small_plates']:
            ax.axvline(r['time_myr'],alpha=.25,linewidth=1)
    ax.set_xlabel('Time, Myr');ax.set_ylabel('Active plate count');ax.set_title('Plate birth/death history');ax.grid(True,alpha=.3);ax.legend();fig.tight_layout();fig.savefig(path,dpi=dpi);plt.close(fig)


def save_plate_size_history(rows,path,dpi=160):
    if not rows:return
    t=np.asarray([r['time_myr'] for r in rows]);fig,ax=plt.subplots(figsize=(10,5.8))
    ax.plot(t,[r['min_plate_cells'] for r in rows],label='minimum');ax.plot(t,[r['mean_plate_cells'] for r in rows],label='mean');ax.plot(t,[r['max_plate_cells'] for r in rows],label='maximum')
    ax.set_xlabel('Time, Myr');ax.set_ylabel('Surface cells per plate');ax.set_title('Plate-size distribution');ax.grid(True,alpha=.3);ax.legend();fig.tight_layout();fig.savefig(path,dpi=dpi);plt.close(fig)


def build_gif(paths,out,frame_duration_ms=350):
    images=[Image.open(p).convert('P',palette=Image.Palette.ADAPTIVE) for p in paths]
    try:images[0].save(out,save_all=True,append_images=images[1:],duration=int(frame_duration_ms),loop=0,optimize=False)
    finally:
        for im in images:im.close()
