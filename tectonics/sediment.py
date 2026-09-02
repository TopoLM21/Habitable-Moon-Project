"""v0.23 conservative surface erosion and sediment routing.

The module replaces the old topography-only erosion sink with an explicit
continental-bedrock -> surface-sediment -> deep-recycled material path.
Climate is intentionally external: callers may supply an erosivity field, while
production defaults use a spatially uniform factor.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .lithosphere import CrustType, LithosphereState, continental_material_fields, effective_continental_thickness_km
from .mesh import SphereMesh
from .topography import TopographyState

Array=np.ndarray

@dataclass(slots=True)
class SedimentBudgetState:
    time_myr: float = 0.0
    cumulative_eroded_bedrock_volume_km3: float = 0.0
    cumulative_reworked_sediment_volume_km3: float = 0.0
    deep_recycled_sediment_volume_km3: float = 0.0
    cumulative_rift_recycled_volume_km3: float = 0.0

@dataclass(slots=True, frozen=True)
class SedimentParameters:
    enabled: bool = True
    erosion_diffusion_per_myr: float = 0.018
    max_erosion_fraction_per_step: float = 0.22
    sediment_reworking_rate_per_myr: float = 0.040
    routing_sweeps: int = 8
    land_deposition_fraction_per_sweep: float = 0.10
    basin_deposition_fraction_per_sweep: float = 0.45
    sediment_density_kg_m3: float = 2400.0
    mantle_density_kg_m3: float = 3300.0
    max_bedrock_erosion_km_per_step: float = 0.75
    # The legacy v0.22 erosion field represented surface denudation after rapid
    # isostatic rebound. Converting that relief change into excavated bedrock
    # requires somewhat more rock volume than the scalar surface lowering.
    bedrock_volume_multiplier: float = 1.50
    burial_soft_limit_km: float = 10.0
    burial_spill_fraction_per_step: float = 0.35

@dataclass(slots=True)
class SedimentDiagnostics:
    time_myr: float
    dt_myr: float
    eroded_bedrock_volume_km3: float
    reworked_sediment_volume_km3: float
    transported_to_deep_reservoir_km3: float
    rift_recycled_volume_km3: float
    surface_sediment_volume_km3: float
    deep_recycled_sediment_volume_km3: float
    cumulative_rift_recycled_volume_km3: float
    mean_sediment_thickness_m: float
    max_sediment_thickness_m: float
    area_fraction_over_1km: float
    step_surface_mass_error_km3: float


def initialize_sediment_budget(time_myr: float = 0.0) -> SedimentBudgetState:
    return SedimentBudgetState(time_myr=float(time_myr))


def sediment_volume_field(state: LithosphereState) -> Array:
    if state.sediment_volume_km3 is None:
        return np.zeros(len(state.crust_thickness_km), dtype=np.float64)
    return np.maximum(np.asarray(state.sediment_volume_km3,dtype=np.float64),0.0)


def sediment_thickness_m(mesh: SphereMesh, state: LithosphereState, radius_km: float) -> Array:
    A=mesh.physical_cell_areas_km2(float(radius_km))
    return 1000.0*sediment_volume_field(state)/np.maximum(A,1e-30)


def sediment_net_surface_factor(params: SedimentParameters) -> float:
    """Local Airy limit for deposited sediment after mantle displacement."""
    rm=max(float(params.mantle_density_kg_m3),1e-9)
    return float(np.clip((rm-float(params.sediment_density_kg_m3))/rm,0.0,1.0))


def _advect_surface_sediment(previous: LithosphereState, source_index: Array) -> tuple[Array,float]:
    old=sediment_volume_field(previous)
    src=np.asarray(source_index,dtype=np.int64)
    n=len(old)
    if src.shape!=(n,):
        raise ValueError('source_index must match cell count')
    valid=(src>=0)&(src<n)
    out=np.zeros(n,dtype=np.float64)
    if np.any(valid):
        counts=np.bincount(src[valid],minlength=n).astype(np.float64)
        # Usually exactly one-to-one. Division makes the operation conservative
        # even if a diagnostic/non-production map happens to duplicate a source.
        out[valid]=old[src[valid]]/np.maximum(counts[src[valid]],1.0)
        used=counts>0
    else:
        used=np.zeros(n,dtype=bool)
    deep=float(np.sum(old[~used]))
    return out,deep


def _route_mobile(mesh: SphereMesh, elevation_m: Array, stationary: Array, mobile: Array,
                  params: SedimentParameters, sea_level_m: float) -> Array:
    from .cpu_runtime import current_execution
    execution = current_execution()
    if execution is not None and execution.cell_kernels:
        from .sediment_kernels import route_mobile_batched
        return route_mobile_batched(mesh, elevation_m, stationary, mobile, params, sea_level_m)
    z=np.asarray(elevation_m,dtype=np.float64)
    sed=np.asarray(stationary,dtype=np.float64).copy()
    mob=np.asarray(mobile,dtype=np.float64).copy()
    neighbors=np.asarray(mesh.neighbors,dtype=np.int32)
    ns=max(int(params.routing_sweeps),0)
    for _ in range(ns):
        next_mob=np.zeros_like(mob)
        for i in np.flatnonzero(mob>0.0):
            v=float(mob[i])
            if v<=0.0: continue
            nbs=neighbors[int(i)]
            drops=z[i]-z[nbs]
            lower=drops>1e-9
            if not np.any(lower):
                sed[i]+=v
                continue
            dep=(float(params.basin_deposition_fraction_per_sweep) if z[i] <= float(sea_level_m)
                 else float(params.land_deposition_fraction_per_sweep))
            dep=float(np.clip(dep,0.0,1.0))
            sed[i]+=v*dep
            move=v*(1.0-dep)
            ids=nbs[lower]
            w=np.asarray(drops[lower],dtype=np.float64)
            w/=max(float(np.sum(w)),1e-30)
            np.add.at(next_mob,ids,move*w)
        mob=next_mob
        if float(np.sum(mob))<=1e-12:
            break
    sed += mob
    return sed


def _spill_overthick_sediment(mesh: SphereMesh, elevation_m: Array, sediment: Array,
                              areas_km2: Array, params: SedimentParameters) -> Array:
    """Conservatively spread part of burial excess instead of hard-clipping it.

    This is an effective unresolved compaction/basin-spill closure. Deep basins
    may exceed the soft limit, but a fraction of the excess is transferred each
    step to lower neighboring accommodation, preventing one-cell sediment towers.
    """
    sed=np.asarray(sediment,dtype=np.float64).copy()
    limit=max(float(params.burial_soft_limit_km),0.0)
    frac=float(np.clip(params.burial_spill_fraction_per_step,0.0,1.0))
    if limit<=0.0 or frac<=0.0: return sed
    z=np.asarray(elevation_m,dtype=np.float64)
    nbs=np.asarray(mesh.neighbors,dtype=np.int32)
    th=sed/np.maximum(areas_km2,1e-30)
    transfer=np.zeros_like(sed)
    for i in np.flatnonzero(th>limit):
        excess=(float(th[i])-limit)*float(areas_km2[i])*frac
        if excess<=0.0: continue
        ids=nbs[int(i)]
        # Prefer lower surface and thinner sediment columns.
        score=np.maximum(z[i]-z[ids],0.0)+250.0*np.maximum(float(th[i])-th[ids],0.0)
        good=score>1e-12
        if not np.any(good): continue
        w=score[good];w=w/max(float(np.sum(w)),1e-30)
        transfer[i]-=excess
        np.add.at(transfer,ids[good],excess*w)
    sed=np.maximum(sed+transfer,0.0)
    return sed


def advance_sediments(
    mesh: SphereMesh,
    previous_lithosphere: LithosphereState,
    state: LithosphereState,
    topography: TopographyState,
    source_index: Array,
    budget: SedimentBudgetState,
    dt_myr: float,
    radius_km: float,
    params: SedimentParameters,
    *,
    rift_recycled_volume_km3: float = 0.0,
    erosivity_field: Array | None = None,
    sea_level_m: float = 0.0,
) -> tuple[LithosphereState,TopographyState,SedimentBudgetState,SedimentDiagnostics]:
    if dt_myr<=0.0: raise ValueError('dt_myr must be positive')
    A=mesh.physical_cell_areas_km2(float(radius_km))
    before_surface=float(np.sum(sediment_volume_field(previous_lithosphere)))
    advected,deep_step=_advect_surface_sediment(previous_lithosphere,source_index)
    state.sediment_volume_km3=advected
    if not bool(params.enabled):
        nb=SedimentBudgetState(float(state.time_myr),budget.cumulative_eroded_bedrock_volume_km3,
            budget.cumulative_reworked_sediment_volume_km3,budget.deep_recycled_sediment_volume_km3+deep_step,
            budget.cumulative_rift_recycled_volume_km3+float(rift_recycled_volume_km3))
        th=1000.0*advected/np.maximum(A,1e-30)
        d=SedimentDiagnostics(float(state.time_myr),float(dt_myr),0.0,0.0,deep_step,float(rift_recycled_volume_km3),float(np.sum(advected)),nb.deep_recycled_sediment_volume_km3,nb.cumulative_rift_recycled_volume_km3,float(np.sum(A*th)/np.sum(A)),float(np.max(th)),float(np.sum(A[th>1000.0])/np.sum(A)),0.0)
        return state,topography,nb,d

    z=np.asarray(topography.elevation_m,dtype=np.float64)
    neigh=np.asarray(mesh.neighbors,dtype=np.int32)
    neigh_mean=np.mean(z[neigh],axis=1)
    local_excess=np.maximum(z-neigh_mean,0.0)
    frac=min(float(params.erosion_diffusion_per_myr)*float(dt_myr),float(params.max_erosion_fraction_per_step))
    eros=np.ones(mesh.cell_count,dtype=np.float64) if erosivity_field is None else np.clip(np.asarray(erosivity_field,dtype=np.float64),0.0,None)
    if eros.shape!=(mesh.cell_count,): raise ValueError('erosivity_field must match cell count')

    cf,cv=continental_material_fields(state,A)
    bedrock_h=effective_continental_thickness_km(cf,cv,A)
    remove_m=np.minimum(frac*float(params.bedrock_volume_multiplier)*local_excess*eros,float(params.max_bedrock_erosion_km_per_step)*1000.0)
    remove_m=np.where((z>0.0)&(cf>1e-12),remove_m,0.0)
    # Erode that thickness only over the continental footprint.
    requested=A*cf*(remove_m/1000.0)
    removed=np.minimum(requested,cv)
    cv=np.maximum(cv-removed,0.0)
    state.continental_volume_km3=cv
    eff=effective_continental_thickness_km(cf,cv,A)
    visible=np.asarray(state.crust_type)==int(CrustType.CONTINENTAL)
    state.crust_thickness_km[visible]=eff[visible]
    eroded=float(np.sum(removed))

    # Existing loose sediment is easier to remobilize than bedrock, but only a
    # bounded fraction is moved each step. Relief contrast suppresses reworking
    # on flat basin floors automatically.
    sed=advected.copy()
    sed_h=1000.0*sed/np.maximum(A,1e-30)
    slope_signal=np.clip(local_excess/500.0,0.0,1.0)
    rework_frac=np.clip(float(params.sediment_reworking_rate_per_myr)*float(dt_myr)*slope_signal,0.0,0.65)
    reworked=sed*rework_frac
    sed-=reworked
    reworked_total=float(np.sum(reworked))
    routed=_route_mobile(mesh,z,sed,removed+reworked,params,float(sea_level_m))
    routed=_spill_overthick_sediment(mesh,z,routed,A,params)
    state.sediment_volume_km3=routed

    # Preserve the old erosion's direct geomorphic response while now giving the
    # removed mass an explicit destination. Sediment redistribution changes the
    # surface by its locally compensated thickness; the full elastic response is
    # captured by the next topographic equilibrium solve.
    mean_remove=np.zeros_like(remove_m)
    nz=requested>1e-30
    mean_remove[nz]=remove_m[nz]*(removed[nz]/requested[nz])*cf[nz]
    delta_sed_h=1000.0*(routed-advected)/np.maximum(A,1e-30)
    net=sediment_net_surface_factor(params)
    new_z=z-mean_remove+net*delta_sed_h
    new_topo=TopographyState(time_myr=float(state.time_myr),elevation_m=new_z)

    after_surface=float(np.sum(routed))
    # Sediment sub-ledger: old surface - transported-deep + newly eroded = new surface.
    surface_err=after_surface-(before_surface-deep_step+eroded)
    nb=SedimentBudgetState(
        time_myr=float(state.time_myr),
        cumulative_eroded_bedrock_volume_km3=float(budget.cumulative_eroded_bedrock_volume_km3+eroded),
        cumulative_reworked_sediment_volume_km3=float(budget.cumulative_reworked_sediment_volume_km3+reworked_total),
        deep_recycled_sediment_volume_km3=float(budget.deep_recycled_sediment_volume_km3+deep_step),
        cumulative_rift_recycled_volume_km3=float(budget.cumulative_rift_recycled_volume_km3+float(rift_recycled_volume_km3)),
    )
    th=1000.0*routed/np.maximum(A,1e-30)
    diag=SedimentDiagnostics(
        time_myr=float(state.time_myr),dt_myr=float(dt_myr),
        eroded_bedrock_volume_km3=eroded,reworked_sediment_volume_km3=reworked_total,
        transported_to_deep_reservoir_km3=deep_step,rift_recycled_volume_km3=float(rift_recycled_volume_km3),
        surface_sediment_volume_km3=after_surface,deep_recycled_sediment_volume_km3=nb.deep_recycled_sediment_volume_km3,
        cumulative_rift_recycled_volume_km3=nb.cumulative_rift_recycled_volume_km3,
        mean_sediment_thickness_m=float(np.sum(A*th)/np.sum(A)),max_sediment_thickness_m=float(np.max(th)),
        area_fraction_over_1km=float(np.sum(A[th>1000.0])/np.sum(A)),step_surface_mass_error_km3=float(surface_err),
    )
    return state,new_topo,nb,diag


def continental_material_ledger_error_km3(initial_continental_volume_km3: float, generated_volume_km3: float,
        state: LithosphereState, budget: SedimentBudgetState, cycle_recycled_volume_km3: float) -> float:
    bedrock=float(np.sum(np.maximum(np.asarray(state.continental_volume_km3,dtype=np.float64),0.0))) if state.continental_volume_km3 is not None else 0.0
    surface=float(np.sum(sediment_volume_field(state)))
    lhs=float(initial_continental_volume_km3)+float(generated_volume_km3)
    rhs=bedrock+surface+float(budget.deep_recycled_sediment_volume_km3)+float(budget.cumulative_rift_recycled_volume_km3)+float(cycle_recycled_volume_km3)
    return lhs-rhs

__all__=['SedimentBudgetState','SedimentParameters','SedimentDiagnostics','initialize_sediment_budget','sediment_volume_field','sediment_thickness_m','sediment_net_surface_factor','advance_sediments','continental_material_ledger_error_km3']
