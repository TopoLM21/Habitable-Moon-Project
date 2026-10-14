"""Independent sorted-wet-set sea solution and ordered basin attribution."""
import json
from pathlib import Path
import numpy as np
from tectonics.unified_geological_genesis import UnifiedGeologicalGenesis as U
from tectonics.topography import TopographyParameters,material_topography_endmembers,material_subgrid_surface_elevations
from analysis.geological_continuation.run import snap


def exact_sea(mesh,f,o,c,radius,volume):
    # Independent of native bisection: solve the cubic on successive wet sets.
    L=np.longdouble
    heights=np.r_[o,c].astype(L);weights=np.r_[mesh.areas_unit_sphere*(1-f),mesh.areas_unit_sphere*f].astype(L)
    active=weights>0;heights=heights[active];weights=weights[active]
    order=np.argsort(heights);heights=heights[order];weights=weights[order]
    floor=L(radius)+heights/1000;W=np.cumsum(weights);B=np.cumsum(weights*floor**3)
    roots=np.cbrt((3*L(volume)+B)/W)
    eligible=np.r_[roots[:-1]<=floor[1:],True]& (roots>=floor)
    index=int(np.flatnonzero(eligible)[0]);sea=(roots[index]-L(radius))*1000
    z=L(radius)+sea/1000;wet=heights<sea
    # Factored cubic avoids cancellation in the independent volume readback.
    check=np.sum(weights[wet]*(z-floor[wet])*(z*z+z*floor[wet]+floor[wet]**2)/3)
    land=float(np.sum(weights[~wet])/np.sum(weights))
    return dict(sea_m=float(sea),land_fraction=land,water_relative_error=float((check-L(volume))/L(volume)))


def decompose(initial,final):
    mesh=final.mesh;r=final.record['radius_km'];V=final.record['water_volume_km3'];p=TopographyParameters()
    f0,h0,o0,c0=material_topography_endmembers(mesh,initial.checkpoint.state,r,p)
    f,h,o,c=material_topography_endmembers(mesh,final.checkpoint.state,r,p)
    _,io,ic=material_subgrid_surface_elevations(mesh,initial.checkpoint.state,initial.checkpoint.topo,r,p)
    _,fo,fc=material_subgrid_surface_elevations(mesh,final.checkpoint.state,final.checkpoint.topo,r,p)
    stages=[('actual_initial',f0,io,ic),('initial_age_depth_equilibrium',f0,o0,c0),
        ('final_ocean_age_depth',f0,o,c0),('final_continental_footprint_at_reference_height',f,o,np.full(len(c),p.continental_reference_elevation_m)),
        ('final_continental_thickness_Airy',f,o,c),('final_relief_including_forcing_and_relaxation',f,fo,fc)]
    result=[];previous=None
    for name,frac,oc,co in stages:
        d=exact_sea(mesh,frac,oc,co,r,V);d['stage']=name;d['increment_m']=None if previous is None else d['sea_m']-previous;previous=d['sea_m'];result.append(d)
    return dict(time_myr=final.checkpoint.state.time_myr,radius_km=r,datum='fixed reference sphere, zero topographic elevation; no re-zeroing',
        reference_sea_level_state_m=final.checkpoint.hydrosphere.reference_sea_level_m,stages=result,
        native_sea_m=snap(final)['sea_level_m'],independent_minus_native_m=result[-1]['sea_m']-snap(final)['sea_level_m'],
        water_change_km3=V-initial.record['water_volume_km3'],phase_change=0,
        attribution='ordered counterfactual decomposition, not unique additive causal attribution; actual final residual includes lag/erosion/ridge/trench/collision forcing',
        initial_mean_floor_m=float(np.average(initial.checkpoint.topo.elevation_m,weights=mesh.areas_unit_sphere)),
        final_mean_floor_m=float(np.average(final.checkpoint.topo.elevation_m,weights=mesh.areas_unit_sphere)),
        max_flexural_correction_m=max(abs(row.get('max_abs_flexural_correction_m',0)) for row in final.checkpoint.topography_rows) if hasattr(final.checkpoint,'topography_rows') else None)

if __name__=='__main__':
    initial=U.load('/workspace/unified-geological-genesis-02/1.0/initial')
    rows=[decompose(initial,U.load(p)) for p in ['/workspace/geological-continuation-04/dt1/final','/workspace/geological-continuation-06/dt05/final']]
    Path('analysis/geological_contact/sea_level.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
