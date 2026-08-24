"""v0.17 diagnostics for thermal/GPE ridge push."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from tectonics.dynamics import DynamicsParameters, mantle_lithosphere_ridge_gpe_proxy, plate_ridge_push_factors
from .raster import rasterize_cells


def _map(mesh,data,path,title,label,cmap='viridis',dpi=180,vmin=None,vmax=None):
    fig=plt.figure(figsize=(12,6.8)); ax=fig.add_subplot(111,projection='mollweide')
    xe,ye,grid=rasterize_cells(mesh,np.asarray(data,dtype=float))
    sc=ax.pcolormesh(xe,ye,grid,cmap=cmap,shading='auto',rasterized=True,vmin=vmin,vmax=vmax)
    fig.colorbar(sc,ax=ax,orientation='horizontal',pad=.08,fraction=.05,label=label)
    ax.grid(True,alpha=.3);ax.set_title(title);fig.tight_layout();fig.savefig(path,dpi=dpi,bbox_inches='tight');plt.close(fig)


def save_ridge_push_maps(mesh,state,radius_km,plate_count,params:DynamicsParameters,out:Path,dpi:int=180)->None:
    if state.mantle_lithosphere_thickness_km is None or state.mantle_lithosphere_density_anomaly_kg_m3 is None:
        return
    out.mkdir(parents=True,exist_ok=True)
    proxy=mantle_lithosphere_ridge_gpe_proxy(state)
    factors=plate_ridge_push_factors(mesh,state,radius_km,plate_count,params)
    cell_factor=factors[np.asarray(state.cell_plate,dtype=np.int32)]
    _map(mesh,proxy,out/'ridge_push_gpe_proxy_final.png',f'Thermal ridge-push GPE proxy — t={state.time_myr:g} Myr',r'$\Delta\rho H^2$ proxy, kg/m³ × km²','inferno',dpi,0.0)
    _map(mesh,cell_factor,out/'ridge_push_plate_factor_final.png',f'Plate-side thermal ridge-push factor — t={state.time_myr:g} Myr','Multiplier relative to v0.16 ridge push','viridis',dpi,0.0,max(2.4,float(np.nanpercentile(cell_factor,99))))


def save_ridge_push_history(rows,path:Path,dpi:int=160)->None:
    rows=[r for r in rows if 'mean_ridge_push_factor' in r]
    if not rows:return
    t=np.asarray([r['time_myr'] for r in rows],dtype=float)
    mean=np.asarray([r['mean_ridge_push_factor'] for r in rows],dtype=float)
    lo=np.asarray([r['min_ridge_push_factor'] for r in rows],dtype=float)
    hi=np.asarray([r['max_ridge_push_factor'] for r in rows],dtype=float)
    fig,ax=plt.subplots(figsize=(10,6))
    ax.fill_between(t,lo,hi,alpha=.18,label='ridge-side min–max')
    ax.plot(t,mean,label='boundary-length weighted mean')
    ax.axhline(1.0,linewidth=1.0,linestyle='--',label='v0.16 constant ridge push')
    ax.set_xlabel('Time, Myr');ax.set_ylabel('Thermal ridge-push multiplier')
    ax.grid(True,alpha=.3);ax.legend();ax.set_title('v0.17 thermal/GPE ridge-push evolution')
    fig.tight_layout();fig.savefig(path,dpi=dpi);plt.close(fig)


def save_ridge_push_age_calibration(path:Path,params:DynamicsParameters,dpi:int=160)->None:
    # Ideal single ridge flank: H^2 and therefore ridge force are ~linear in age
    # until the plate-cooling cap. Use the same reference normalization as dynamics.
    from tectonics.lithosphere import oceanic_thermal_lithosphere_total_thickness_km
    age=np.linspace(0,200,401)
    h=np.maximum(oceanic_thermal_lithosphere_total_thickness_km(age)-7.0,0.0)
    ref=max(params.ridge_gpe_reference_density_anomaly_kg_m3*params.ridge_gpe_reference_mantle_thickness_km**2,1e-12)
    # For a simple flank, the end-member GPE is the relevant ridge-to-flank contrast.
    raw=(params.ridge_gpe_reference_density_anomaly_kg_m3*h*h)/ref
    factor=params.ridge_gpe_calibration_gain*np.maximum(raw,0.0)**params.ridge_gpe_exponent
    factor=np.clip(factor,params.ridge_gpe_min_factor,params.ridge_gpe_max_factor)
    fig,ax=plt.subplots(figsize=(9,5.5));ax.plot(age,factor,label='v0.17 thermal/GPE factor');ax.axhline(1.0,linestyle='--',linewidth=1,label='v0.16 constant')
    ax.axvline(80.0,linestyle=':',linewidth=1,label='80 Myr reference')
    ax.set_xlabel('Characteristic oceanic-lithosphere age, Myr');ax.set_ylabel('Ridge-push multiplier');ax.grid(True,alpha=.3);ax.legend();ax.set_title('Ridge-push thermal calibration')
    fig.tight_layout();fig.savefig(path,dpi=dpi);plt.close(fig)
