"""v0.23 conservative sediment diagnostics."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .raster import rasterize_cells
from tectonics.sediment import sediment_thickness_m


def save_sediment_maps(mesh,state,radius_km,out_dir,dpi=180):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    th=sediment_thickness_m(mesh,state,radius_km)
    xe,ye,data=rasterize_cells(mesh,th,width=720,height=360)
    fig=plt.figure(figsize=(12,6.5));ax=fig.add_subplot(111,projection='mollweide')
    vmax=max(float(np.nanpercentile(data,99.5)),100.0)
    pc=ax.pcolormesh(xe,ye,data,shading='auto',cmap='cividis',vmin=0,vmax=vmax,rasterized=True)
    fig.colorbar(pc,ax=ax,orientation='horizontal',pad=.08,fraction=.05,label='Sediment thickness, m')
    ax.grid(True,alpha=.28);ax.set_title(f'Conservative sediment thickness — t={state.time_myr:g} Myr')
    fig.tight_layout();fig.savefig(out/'sediment_thickness_final.png',dpi=dpi,bbox_inches='tight');plt.close(fig)


def save_sediment_history(rows,path,dpi=160):
    if not rows:return
    p=Path(path);t=np.asarray([float(r['time_myr']) for r in rows])
    fig,ax=plt.subplots(figsize=(10,5.7))
    ax.plot(t,[float(r['surface_sediment_volume_km3'])/1e6 for r in rows],label='surface sediment')
    ax.plot(t,[float(r['deep_recycled_sediment_volume_km3'])/1e6 for r in rows],label='deep recycled sediment')
    ax.plot(t,[float(r['cumulative_rift_recycled_volume_km3'])/1e6 for r in rows],label='rift recycled crust')
    ax.set(xlabel='Time, Myr',ylabel='million km³',title='Conservative continental-material reservoirs');ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(p,dpi=dpi);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.7))
    ax.plot(t,[float(r['mean_sediment_thickness_m']) for r in rows],label='mean thickness')
    ax.plot(t,[float(r['max_sediment_thickness_m']) for r in rows],label='max thickness')
    ax.set(xlabel='Time, Myr',ylabel='m',title='Sediment thickness history');ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(p.with_name('sediment_thickness_history.png'),dpi=dpi);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.7))
    ax.plot(t,[float(r['eroded_bedrock_volume_km3'])/1e6 for r in rows],label='bedrock erosion / step')
    ax.plot(t,[float(r['reworked_sediment_volume_km3'])/1e6 for r in rows],label='sediment reworked / step')
    ax.plot(t,[float(r['transported_to_deep_reservoir_km3'])/1e6 for r in rows],label='subducted sediment / step')
    ax.set(xlabel='Time, Myr',ylabel='million km³ per step',title='Sediment fluxes');ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(p.with_name('sediment_flux_history.png'),dpi=dpi);plt.close(fig)
