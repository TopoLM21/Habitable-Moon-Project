"""Visual diagnostics for v0.8 continental crust cycle."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tectonics.lithosphere import CrustType
from tectonics.lithosphere import continental_material_fields,effective_continental_thickness_km
from .raster import rasterize_cells


def _lon_lat(points):
    return np.arctan2(points[:,1],points[:,0]), np.arcsin(np.clip(points[:,2],-1,1))


def save_continental_maps(mesh,state,cycle,out:Path,dpi:int=180)->None:
    out.mkdir(parents=True,exist_ok=True);lon,lat=_lon_lat(mesh.centroids)
    type_data=(state.crust_type==int(CrustType.CONTINENTAL)).astype(float)
    maps=[
        ('crust_type_final.png',type_data,'Continental (1) / oceanic (0) crust','Crust class','coolwarm'),
        ('continental_age_final.png',np.where(type_data>0,state.crust_age_myr,np.nan),'Continental crust age','Age, Myr','plasma'),
        ('felsic_potential_final.png',cycle.felsic_potential,'Felsic / island-arc maturation potential','Potential','viridis'),
    ]
    for name,data,title,label,cmap in maps:
        fig=plt.figure(figsize=(12,6.5));ax=fig.add_subplot(111,projection='mollweide')
        sc=ax.scatter(lon,lat,c=data,cmap=cmap,s=2.5,linewidths=0,rasterized=True)
        fig.colorbar(sc,ax=ax,orientation='horizontal',pad=.08,fraction=.05,label=label)
        ax.grid(True,alpha=.3);ax.set_title(f'{title} — t={state.time_myr:g} Myr');fig.tight_layout();fig.savefig(out/name,dpi=dpi);plt.close(fig)



def save_continental_history_frame(mesh,state,path,dpi:int=120)->None:
    """Save continental thickness only; oceanic cells are masked.

    This is deliberately separate from the plate-ID animation so a material
    footprint that stalls while plate ownership moves is visually obvious.
    """
    # Use the independent v0.11 material layer when present.  A 5% threshold
    # removes only numerically tiny fractional fringe material from display.
    # Radius cancels in the effective thickness ratio, so unit-sphere areas are sufficient.
    areas=mesh.areas_unit_sphere
    frac,vol=continental_material_fields(state,areas)
    # Stored volume is physical km^3. Infer R^2 from cells whose legacy visible
    # thickness is synced to the material layer; this avoids threading radius
    # through every plotting call while still rendering fractional fringes with
    # their true material thickness.
    cont=frac>=0.05
    vis=(state.crust_type==int(CrustType.CONTINENTAL))&(frac>1e-6)&(state.crust_thickness_km>0)
    if state.continental_volume_km3 is not None and np.any(vis):
        r2=np.median(np.asarray(state.continental_volume_km3)[vis]/(mesh.areas_unit_sphere[vis]*frac[vis]*state.crust_thickness_km[vis]))
        physical_areas=mesh.areas_unit_sphere*r2
        h=effective_continental_thickness_km(frac,np.asarray(state.continental_volume_km3),physical_areas)
    else:
        h=np.asarray(state.crust_thickness_km,dtype=float)
    data=np.where(cont,h,np.nan)
    fig=plt.figure(figsize=(12,6.8));ax=fig.add_subplot(111,projection='mollweide')
    xe,ye,grid=rasterize_cells(mesh,data)
    sc=ax.pcolormesh(xe,ye,grid,cmap='viridis',shading='auto',rasterized=True)
    if np.any(cont):
        fig.colorbar(sc,ax=ax,orientation='horizontal',pad=.08,fraction=.05,label='Continental crust thickness, km')
    ax.grid(True,alpha=.3)
    area_pct=100.0*float(np.sum(mesh.areas_unit_sphere*frac))/float(np.sum(mesh.areas_unit_sphere))
    mean_h=float(np.average(state.crust_thickness_km[state.crust_type==int(CrustType.CONTINENTAL)],weights=mesh.areas_unit_sphere[state.crust_type==int(CrustType.CONTINENTAL)])) if np.any(state.crust_type==int(CrustType.CONTINENTAL)) else 0.0
    ax.set_title(f'Continental material — t={state.time_myr:g} Myr | material area={area_pct:.1f}% | mean visible h={mean_h:.1f} km')
    fig.tight_layout();fig.savefig(path,dpi=dpi,bbox_inches='tight');plt.close(fig)

def save_continental_cycle_history(rows,path,dpi=160):
    if not rows:return
    t=np.asarray([r['time_myr'] for r in rows],dtype=float)
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(t,100*np.asarray([r['continental_area_fraction'] for r in rows]),label='continental area fraction')
    ax.set_xlabel('Time, Myr');ax.set_ylabel('Continental surface, %');ax.grid(True,alpha=.3)
    ax2=ax.twinx();ax2.plot(t,np.asarray([r['continental_volume_km3'] for r in rows])/1e9,linestyle='--',label='continental volume')
    ax2.set_ylabel('Continental crust volume, billion km³')
    ax.set_title('Continental crust area and volume')
    lines=ax.get_lines()+ax2.get_lines();ax.legend(lines,[x.get_label() for x in lines],loc='best')
    fig.tight_layout();fig.savefig(path,dpi=dpi);plt.close(fig)


def save_continental_flux_history(rows,path,dpi=160):
    if not rows:return
    t=np.asarray([r['time_myr'] for r in rows],dtype=float)
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(t,np.asarray([r['juvenile_arc_area_created_km2'] for r in rows])/1e6,label='juvenile continent created')
    ax.plot(t,np.asarray([r['subduction_erosion_area_km2'] for r in rows])/1e6,label='continental area recycled')
    ax.set_xlabel('Time, Myr');ax.set_ylabel('Area per step, million km²');ax.grid(True,alpha=.3)
    ax2=ax.twinx();
    ax2.plot(t,np.asarray([r['arc_thickening_volume_km3'] for r in rows])/1e6,linestyle='--',label='arc thickening volume')
    ax2.plot(t,np.asarray([r['delaminated_volume_km3'] for r in rows])/1e6,linestyle=':',label='delaminated volume')
    ax2.set_ylabel('Volume per step, million km³')
    ax.set_title('Continental crust creation and recycling')
    lines=ax.get_lines()+ax2.get_lines();ax.legend(lines,[x.get_label() for x in lines],loc='best')
    fig.tight_layout();fig.savefig(path,dpi=dpi);plt.close(fig)
