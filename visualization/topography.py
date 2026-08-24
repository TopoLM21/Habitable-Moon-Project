"""Visual diagnostics for v0.6 tectonic topography."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tectonics.kinematics import BoundaryType
from tectonics.lithosphere import CrustType, LithosphereState
from tectonics.mesh import SphereMesh
from tectonics.topography import TopographyDiagnostics, TopographyState
from tectonics.plates import PlateSystem


def _lon_lat(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon=np.arctan2(points[:,1],points[:,0]); lat=np.arcsin(np.clip(points[:,2],-1,1)); return lon,lat


def save_topography_frame(mesh: SphereMesh, lith: LithosphereState, topo: TopographyState, system: PlateSystem, boundaries, diag: TopographyDiagnostics|None, path: str|Path, dpi:int=120)->None:
    lon,lat=_lon_lat(mesh.centroids)
    fig=plt.figure(figsize=(12,6.8)); ax=fig.add_subplot(111,projection='mollweide')
    lim=max(6000.0,float(np.percentile(np.abs(topo.elevation_m),99)))
    sc=ax.scatter(lon,lat,c=topo.elevation_m,cmap='terrain',vmin=-lim,vmax=lim,s=2.6,linewidths=0,rasterized=True)
    fig.colorbar(sc,ax=ax,orientation='horizontal',pad=0.08,fraction=0.05,label='Elevation relative to reference datum, m')
    # Only show tectonically diagnostic boundaries; the relief itself is primary.
    for kind in (BoundaryType.DIVERGENT,BoundaryType.CONVERGENT,BoundaryType.TRANSFORM):
        pts=np.asarray([b.midpoint for b in boundaries if b.boundary_type==kind])
        if len(pts):
            blon,blat=_lon_lat(pts); ax.scatter(blon,blat,s=2.8,linewidths=0,alpha=.55,label=kind.name.lower())
    title=f'v0.6 tectonic relief — t={topo.time_myr:g} Myr'
    if diag:
        title+=f"\nmin={diag.min_elevation_m:,.0f} m max={diag.max_elevation_m:,.0f} m | reference exposed={100*diag.reference_exposed_fraction:.1f}% | erosion={diag.eroded_volume_km3:,.0f} km³/step"
    ax.set_title(title); ax.grid(True,alpha=.3); ax.legend(loc='lower center',ncol=3,bbox_to_anchor=(.5,-.15)); fig.tight_layout(); fig.savefig(path,dpi=dpi,bbox_inches='tight'); plt.close(fig)


def save_final_maps(mesh:SphereMesh,lith:LithosphereState,topo:TopographyState,target:np.ndarray,out:Path,dpi:int=180)->None:
    out.mkdir(parents=True,exist_ok=True); lon,lat=_lon_lat(mesh.centroids)
    for name,data,title,label,cmap in [
        ('elevation_final.png',topo.elevation_m,f'Final tectonic elevation — t={topo.time_myr:g} Myr','Elevation, m','terrain'),
        ('equilibrium_elevation_final.png',target,f'Instantaneous tectonic/isostatic target — t={topo.time_myr:g} Myr','Target elevation, m','terrain'),
        ('crust_thickness_final.png',lith.crust_thickness_km,f'Crust thickness — t={topo.time_myr:g} Myr','Crust thickness, km','viridis'),
    ]:
        fig=plt.figure(figsize=(12,6.5)); ax=fig.add_subplot(111,projection='mollweide'); sc=ax.scatter(lon,lat,c=data,cmap=cmap,s=2.5,linewidths=0,rasterized=True); fig.colorbar(sc,ax=ax,orientation='horizontal',pad=.08,fraction=.05,label=label); ax.grid(True,alpha=.3); ax.set_title(title); fig.tight_layout(); fig.savefig(out/name,dpi=dpi); plt.close(fig)


def save_topography_history(rows:list[dict],path:str|Path,dpi:int=160)->None:
    t=np.asarray([r['time_myr'] for r in rows]);
    fig,ax=plt.subplots(figsize=(10,6));
    ax.plot(t,[r['max_elevation_m'] for r in rows],label='maximum'); ax.plot(t,[r['mean_continental_elevation_m'] for r in rows],label='mean continent'); ax.plot(t,[r['mean_oceanic_elevation_m'] for r in rows],label='mean ocean'); ax.plot(t,[r['min_elevation_m'] for r in rows],label='minimum');
    ax.set_xlabel('Time, Myr'); ax.set_ylabel('Elevation, m'); ax.set_title('v0.6 topographic evolution'); ax.grid(True,alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=dpi); plt.close(fig)


def save_process_history(rows:list[dict],path:str|Path,dpi:int=160)->None:
    t=np.asarray([r['time_myr'] for r in rows]); fig,ax=plt.subplots(figsize=(10,6));
    ax.plot(t,[r['ridge_cells'] for r in rows],label='ridge cells'); ax.plot(t,[r['trench_cells'] for r in rows],label='trench cells'); ax.plot(t,[r['arc_cells'] for r in rows],label='arc cells'); ax.plot(t,[r['collision_cells'] for r in rows],label='collision cells'); ax.set_xlabel('Time, Myr'); ax.set_ylabel('Active boundary-adjacent cells'); ax.set_title('Topographic tectonic processes'); ax.grid(True,alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=dpi); plt.close(fig)


def build_gif(frame_paths:list[Path],output_path:str|Path,frame_duration_ms:int=350)->None:
    images=[Image.open(p).convert('P',palette=Image.Palette.ADAPTIVE) for p in frame_paths]
    try: images[0].save(output_path,save_all=True,append_images=images[1:],duration=int(frame_duration_ms),loop=0,optimize=False)
    finally:
        for im in images: im.close()
