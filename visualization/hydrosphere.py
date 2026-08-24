"""Filled sea-level/shoreline maps for v0.14 hydrosphere."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

from tectonics.hydrosphere import HydrosphereDiagnostics, HydrosphereState, HydrosphereParameters, diagnose_hydrosphere
from tectonics.lithosphere import LithosphereState, continental_material_fields
from tectonics.topography import TopographyState, TopographyParameters, material_subgrid_surface_elevations
from .raster import rasterize_cells


def _surface_cmap():
    # Hard visual split at sea level: deep->shallow blues below zero, then
    # lowland greens/browns and highland pale tones above zero.  This is much
    # easier to read as a moving coastline than a single continuous terrain map.
    water=plt.cm.Blues(np.linspace(0.95,0.28,128))
    land=plt.cm.terrain(np.linspace(0.28,1.0,128))
    return LinearSegmentedColormap.from_list('v014_land_ocean',np.vstack([water,land]))


def _display_surface_fields(mesh, lithosphere, topography, hydrosphere, diag, radius_km, hydrop, topop):
    """Return representative relative height, land fraction and mean water depth.

    For mixed cells, the representative height follows the majority surface
    patch (dry or wet) so a mostly continental coastal cell is not painted as
    abyssal ocean merely because its scalar area-mean elevation is negative.
    """
    if not (bool(getattr(hydrop, 'subgrid_material_hypsometry', False)) and topop is not None):
        rel=np.asarray(topography.elevation_m,float)-float(diag.sea_level_m)
        return rel,(rel>=0.0).astype(float),np.maximum(-rel,0.0)
    f,ocean,cont=material_subgrid_surface_elevations(mesh,lithosphere,topography,radius_km,topop)
    orl=ocean-float(diag.sea_level_m); crl=cont-float(diag.sea_level_m)
    odry=orl>=0.0; cdry=crl>=0.0
    ow=1.0-f; cw=f
    land=ow*odry+cw*cdry
    dryw=ow*odry+cw*cdry; wetw=ow*(~odry)+cw*(~cdry)
    drynum=ow*odry*np.maximum(orl,0.0)+cw*cdry*np.maximum(crl,0.0)
    wetnum=ow*(~odry)*np.minimum(orl,0.0)+cw*(~cdry)*np.minimum(crl,0.0)
    dryrep=np.divide(drynum,np.maximum(dryw,1e-30))
    wetrep=np.divide(wetnum,np.maximum(wetw,1e-30))
    rep=np.where(land>=0.5,dryrep,wetrep)
    mean_depth=ow*np.maximum(-orl,0.0)+cw*np.maximum(-crl,0.0)
    return rep,land,mean_depth


def save_hydrosphere_frame(mesh, lithosphere: LithosphereState, topography: TopographyState,
                           hydrosphere: HydrosphereState, diag: HydrosphereDiagnostics,
                           radius_km: float, path: str|Path, dpi: int=120,
                           hydrosphere_params: HydrosphereParameters | None = None,
                           topography_params: TopographyParameters | None = None) -> None:
    hp=hydrosphere_params or HydrosphereParameters()
    relative,land_fraction,_ = _display_surface_fields(
        mesh,lithosphere,topography,hydrosphere,diag,radius_km,hp,topography_params
    )
    lon_edges, lat_edges, field = rasterize_cells(mesh, relative, width=720, height=360)
    fig = plt.figure(figsize=(12.8, 6.8))
    ax = fig.add_subplot(111, projection='mollweide')
    # terrain gives an immediately readable water/land surface; zero is the
    # physically solved coastline, not the historical arbitrary datum.
    vmax = max(3500.0, float(np.percentile(relative, 98.5)))
    vmin = min(-7000.0, float(np.percentile(relative, 1.5)))
    norm=TwoSlopeNorm(vmin=vmin,vcenter=0.0,vmax=vmax)
    im = ax.pcolormesh(lon_edges, lat_edges, field, cmap=_surface_cmap(), norm=norm, shading='flat', rasterized=True)
    # Coastline from rasterised relative elevation.
    lon_c = 0.5*(lon_edges[:-1]+lon_edges[1:])
    lat_c = 0.5*(lat_edges[:-1]+lat_edges[1:])
    _,_,land_field = rasterize_cells(mesh, land_fraction, width=720, height=360)
    try:
        ax.contour(lon_c, lat_c, land_field, levels=[0.5], linewidths=0.65)
    except ValueError:
        pass
    ax.grid(True, alpha=0.22)
    ax.set_title(
        f"v0.15 material-aware surface — t={topography.time_myr:g} Myr | sea level {diag.sea_level_m:+.1f} m | "
        f"land {100*diag.land_area_fraction:.1f}%\n"
        f"mean ocean depth {diag.mean_ocean_depth_m/1000:.2f} km | max {diag.max_ocean_depth_m/1000:.2f} km"
    )
    cb=fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, shrink=0.72)
    cb.set_label('Topography relative to sea level, m')
    fig.tight_layout()
    fig.savefig(path,dpi=dpi,bbox_inches='tight')
    plt.close(fig)


def save_hydrosphere_history(rows: list[dict], path: str|Path, dpi: int=160) -> None:
    if not rows: return
    t=np.asarray([r['time_myr'] for r in rows],float)
    sea=np.asarray([r['sea_level_m'] for r in rows],float)
    land=100*np.asarray([r['land_area_fraction'] for r in rows],float)
    shallow=100*np.asarray([r['shallow_sea_area_fraction'] for r in rows],float)
    fig,ax=plt.subplots(figsize=(10,5.8))
    ax.plot(t,sea,label='sea level')
    ax.set_xlabel('Time, Myr');ax.set_ylabel('Sea level relative to original datum, m');ax.grid(True,alpha=.3)
    ax2=ax.twinx();ax2.plot(t,land,label='land fraction');ax2.plot(t,shallow,label='shallow sea fraction')
    ax2.set_ylabel('Surface area, %')
    lines=ax.get_lines()+ax2.get_lines();ax.legend(lines,[x.get_label() for x in lines],loc='best')
    ax.set_title('Conserved-water sea level and exposed surface')
    fig.tight_layout();fig.savefig(path,dpi=dpi);plt.close(fig)


def save_final_hydrosphere_maps(mesh, lithosphere: LithosphereState, topography: TopographyState,
                                 hydrosphere: HydrosphereState, radius_km: float,
                                 params: HydrosphereParameters, out_dir: str|Path, dpi: int=180,
                                 topography_params: TopographyParameters | None = None) -> None:
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    diag=diagnose_hydrosphere(mesh,lithosphere,topography,hydrosphere,radius_km,params,topography_params)
    relative,land_fraction,depth=_display_surface_fields(mesh,lithosphere,topography,hydrosphere,diag,radius_km,params,topography_params)
    areas=mesh.physical_cell_areas_km2(radius_km);cont,_=continental_material_fields(lithosphere,areas)
    if bool(params.subgrid_material_hypsometry) and topography_params is not None:
        f,ocean_s,cont_s=material_subgrid_surface_elevations(mesh,lithosphere,topography,radius_km,topography_params)
        submerged_cont=f*(cont_s<diag.sea_level_m)
    else:
        submerged_cont=np.where(depth>0,cont,0.0)
    fields=[
        ('surface_relative_sea_level.png',relative,'terrain','Elevation relative to sea level, m'),
        ('water_depth.png',depth,'Blues','Water depth, m'),
        ('continental_shelf.png',np.where(submerged_cont>0,submerged_cont,np.nan),'viridis','Submerged continental-material fraction'),
        ('land_fraction.png',land_fraction,'viridis','Sub-grid land fraction'),
    ]
    for name,values,cmap,label in fields:
        lon_edges,lat_edges,data=rasterize_cells(mesh,values,width=720,height=360)
        fig=plt.figure(figsize=(12.8,6.8));ax=fig.add_subplot(111,projection='mollweide')
        
        if name=='surface_relative_sea_level.png':
            finite=data[np.isfinite(data)];vvmin=min(-7000.0,float(np.percentile(finite,1.5)));vvmax=max(3500.0,float(np.percentile(finite,98.5)))
            im=ax.pcolormesh(lon_edges,lat_edges,data,cmap=_surface_cmap(),norm=TwoSlopeNorm(vmin=vvmin,vcenter=0.0,vmax=vvmax),shading='flat',rasterized=True)
        else:
            im=ax.pcolormesh(lon_edges,lat_edges,data,cmap=cmap,shading='flat',rasterized=True)
        ax.grid(True,alpha=.22);ax.set_title(f'{label} — t={topography.time_myr:g} Myr, sea={diag.sea_level_m:+.1f} m')
        cb=fig.colorbar(im,ax=ax,orientation='horizontal',pad=.08,shrink=.72);cb.set_label(label)
        fig.tight_layout();fig.savefig(out/name,dpi=dpi,bbox_inches='tight');plt.close(fig)
