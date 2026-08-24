"""Diagnostics for v0.22 variable-rigidity flexural isostasy."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .raster import rasterize_cells


def _filled(mesh, values, path: Path, title: str, label: str, *, cmap=None, symmetric=False, dpi=180):
    lon_edges,lat_edges,data=rasterize_cells(mesh,np.asarray(values,dtype=float),width=720,height=360)
    fig=plt.figure(figsize=(12,6.5));ax=fig.add_subplot(111,projection='mollweide')
    kwargs={}
    if cmap is not None: kwargs['cmap']=cmap
    if symmetric:
        vmax=float(np.nanpercentile(np.abs(data),99.5)) if np.any(np.isfinite(data)) else 1.0
        vmax=max(vmax,1e-9); kwargs.update(vmin=-vmax,vmax=vmax)
    pc=ax.pcolormesh(lon_edges,lat_edges,data,shading='auto',**kwargs)
    fig.colorbar(pc,ax=ax,orientation='horizontal',pad=.08,fraction=.05,label=label)
    ax.grid(True,alpha=.28);ax.set_title(title);fig.tight_layout();fig.savefig(path,dpi=dpi,bbox_inches='tight');plt.close(fig)


def save_flexure_maps(mesh, components: dict, out_dir: str|Path, dpi: int=180)->None:
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    _filled(mesh,components['effective_elastic_thickness_km'],out/'effective_elastic_thickness_final.png','Effective elastic thickness $T_e$','km',cmap='viridis',dpi=dpi)
    _filled(mesh,components['flexural_parameter_km'],out/'flexural_parameter_final.png','Local flexural parameter','km',cmap='viridis',dpi=dpi)
    _filled(mesh,components['flexural_local_source'],out/'flexural_local_source_final.png','Local-isostatic / tectonic source before flexure','Local target anomaly, m',cmap='coolwarm',symmetric=True,dpi=dpi)
    _filled(mesh,components['flexural_response'],out/'flexural_response_final.png','Elastic-plate flexural response','Flexural response, m',cmap='coolwarm',symmetric=True,dpi=dpi)
    _filled(mesh,components['flexural_correction'],out/'flexural_correction_final.png','Flexural correction relative to local isostasy','Correction, m',cmap='coolwarm',symmetric=True,dpi=dpi)


def save_flexure_history(rows: list[dict], path: str|Path, dpi: int=160)->None:
    if not rows:return
    p=Path(path); t=np.asarray([float(r.get('time_myr',0.0)) for r in rows])
    fig,ax=plt.subplots(figsize=(10,5.5))
    ax.plot(t,[float(r.get('mean_elastic_thickness_km',0.0)) for r in rows],label='mean $T_e$')
    ax.plot(t,[float(r.get('mean_flexural_parameter_km',0.0)) for r in rows],label='mean flexural parameter')
    ax.set(xlabel='Time, Myr',ylabel='km',title='Effective elastic lithosphere');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(p,dpi=dpi);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.5))
    ax.plot(t,[float(r.get('max_abs_flexural_correction_m',0.0)) for r in rows],label='max |correction|')
    ax.plot(t,[float(r.get('rms_flexural_correction_m',0.0)) for r in rows],label='RMS correction')
    ax.set(xlabel='Time, Myr',ylabel='m',title='Flexural correction amplitude');ax.grid(alpha=.25);ax.legend();
    ax2=ax.twinx();ax2.plot(t,[float(r.get('flexure_cg_iterations',0.0)) for r in rows],alpha=.55,label='CG iterations');ax2.set_ylabel('CG iterations')
    fig.tight_layout();fig.savefig(p.with_name('flexure_solver_history.png'),dpi=dpi);plt.close(fig)
