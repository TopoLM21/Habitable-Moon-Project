"""Diagnostics for v0.16 crust / mantle-lithosphere separation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tectonics.lithosphere import mantle_lithosphere_negative_buoyancy_proxy
from .raster import rasterize_cells


def _map(mesh,data,path,title,label,cmap='viridis',dpi=180,vmin=None,vmax=None):
    fig=plt.figure(figsize=(12,6.8)); ax=fig.add_subplot(111,projection='mollweide')
    xe,ye,grid=rasterize_cells(mesh,np.asarray(data,dtype=float))
    sc=ax.pcolormesh(xe,ye,grid,cmap=cmap,shading='auto',rasterized=True,vmin=vmin,vmax=vmax)
    fig.colorbar(sc,ax=ax,orientation='horizontal',pad=.08,fraction=.05,label=label)
    ax.grid(True,alpha=.3); ax.set_title(title); fig.tight_layout(); fig.savefig(path,dpi=dpi,bbox_inches='tight'); plt.close(fig)


def save_lithosphere_split_maps(mesh,state,out:Path,dpi:int=180)->None:
    if state.mantle_lithosphere_thickness_km is None or state.mantle_lithosphere_density_anomaly_kg_m3 is None:
        return
    out.mkdir(parents=True,exist_ok=True)
    h=np.asarray(state.mantle_lithosphere_thickness_km,dtype=float)
    drho=np.asarray(state.mantle_lithosphere_density_anomaly_kg_m3,dtype=float)
    crust=np.asarray(state.crust_thickness_km,dtype=float)
    proxy=mantle_lithosphere_negative_buoyancy_proxy(state)
    _map(mesh,crust,out/'chemical_crust_thickness_final.png',f'Chemical crust thickness — t={state.time_myr:g} Myr','Crust thickness, km','viridis',dpi)
    _map(mesh,h,out/'mantle_lithosphere_thickness_final.png',f'Mantle lithosphere thickness — t={state.time_myr:g} Myr','Mantle lithosphere, km','magma',dpi,0.0,max(155.0,float(np.nanpercentile(h,99))))
    _map(mesh,crust+h,out/'mechanical_lithosphere_total_thickness_final.png',f'Crust + mantle mechanical lithosphere — t={state.time_myr:g} Myr','Total mechanical thickness, km','plasma',dpi)
    _map(mesh,drho,out/'mantle_lithosphere_density_anomaly_final.png',f'Mantle-lithosphere density anomaly — t={state.time_myr:g} Myr',r'$\Delta\rho$, kg/m³','coolwarm',dpi)
    _map(mesh,proxy,out/'slab_negative_buoyancy_proxy_final.png',f'Integrated negative-buoyancy proxy — t={state.time_myr:g} Myr','km × kg/m³','inferno',dpi,0.0)


def save_lithosphere_split_frame(mesh,state,path:Path,dpi:int=120)->None:
    if state.mantle_lithosphere_thickness_km is None:
        return
    h=np.asarray(state.mantle_lithosphere_thickness_km,dtype=float)
    fig=plt.figure(figsize=(12,6.8)); ax=fig.add_subplot(111,projection='mollweide')
    xe,ye,grid=rasterize_cells(mesh,h)
    sc=ax.pcolormesh(xe,ye,grid,cmap='magma',shading='auto',rasterized=True,vmin=0,vmax=155)
    fig.colorbar(sc,ax=ax,orientation='horizontal',pad=.08,fraction=.05,label='Mantle lithosphere thickness, km')
    ax.grid(True,alpha=.3)
    ax.set_title(f'v0.16 mechanical mantle lithosphere — t={state.time_myr:g} Myr')
    fig.tight_layout(); fig.savefig(path,dpi=dpi,bbox_inches='tight'); plt.close(fig)


def save_lithosphere_split_history(rows,path:Path,dpi:int=160)->None:
    rows=[r for r in rows if 'mean_oceanic_mantle_lithosphere_thickness_km' in r]
    if not rows:return
    t=np.asarray([r['time_myr'] for r in rows],dtype=float)
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(t,[r['mean_oceanic_mantle_lithosphere_thickness_km'] for r in rows],label='mean oceanic mantle lithosphere')
    ax.plot(t,[r['p90_oceanic_mantle_lithosphere_thickness_km'] for r in rows],label='p90 oceanic')
    ax.plot(t,[r['mean_continental_mantle_lithosphere_thickness_km'] for r in rows],label='mean continental/mixed root')
    ax.set_xlabel('Time, Myr');ax.set_ylabel('Mantle lithosphere thickness, km');ax.grid(True,alpha=.3);ax.legend();ax.set_title('v0.16 crust / mantle-lithosphere separation')
    fig.tight_layout();fig.savefig(path,dpi=dpi);plt.close(fig)
