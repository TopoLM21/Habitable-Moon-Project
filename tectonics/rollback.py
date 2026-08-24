"""v0.19 slab rollback, trench migration and back-arc extension.

This is an effective 2.5-D coupling built on v0.18 slab memory.  A deep,
mature, negatively buoyant active slab can retreat oceanward relative to the
overriding plate.  We represent that in two deliberately small pieces:

1. a plate-scale angular-velocity contribution on the overriding plate;
2. a local extensional forcing band landward of the trench, consumed by the
   existing conservative continental-rifting machinery.

No new plate boundary is created directly by this module.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .mesh import SphereMesh
from .lithosphere import LithosphereState
from .subduction_memory import SubductionMemoryState

Array=np.ndarray


@dataclass(slots=True)
class RollbackParameters:
    enabled: bool = True
    min_slab_length_km: float = 550.0
    full_slab_length_km: float = 1400.0
    min_slab_depth_km: float = 350.0
    full_slab_depth_km: float = 900.0
    max_rollback_rate_km_per_myr: float = 2.0
    buoyancy_reference_factor: float = 1.0
    rollback_distance_cap_km: float = 1600.0
    backarc_inner_distance_km: float = 120.0
    backarc_peak_distance_km: float = 280.0
    backarc_outer_distance_km: float = 650.0
    backarc_max_extension_forcing: float = 0.15
    minimum_active_age_myr: float = 20.0


@dataclass(slots=True)
class RollbackDiagnostics:
    time_myr: float
    active_rollback_zones: int
    mean_rollback_rate_km_per_myr: float
    max_rollback_rate_km_per_myr: float
    mean_rollback_distance_km: float
    max_rollback_distance_km: float
    backarc_forced_area_km2: float
    max_backarc_extension_forcing: float


def _smooth01(x: float) -> float:
    x=float(np.clip(x,0.0,1.0)); return x*x*(3.0-2.0*x)


def zone_rollback_rate(z, params: RollbackParameters) -> float:
    if not params.enabled or not z.active or z.active_age_myr < float(params.minimum_active_age_myr):
        return 0.0
    lf=_smooth01((z.slab_length_km-float(params.min_slab_length_km))/max(float(params.full_slab_length_km-params.min_slab_length_km),1e-9))
    df=_smooth01((z.slab_depth_km-float(params.min_slab_depth_km))/max(float(params.full_slab_depth_km-params.min_slab_depth_km),1e-9))
    bf=float(np.clip(z.buoyancy_factor/max(float(params.buoyancy_reference_factor),1e-9),0.35,1.35))
    return min(float(params.max_rollback_rate_km_per_myr), float(params.max_rollback_rate_km_per_myr)*lf*df*np.sqrt(bf))


def _toward_overriding(z) -> Array:
    r=np.asarray(z.trench_midpoint,dtype=np.float64)
    t=np.cross(np.asarray(z.torque_axis,dtype=np.float64),r)
    n=float(np.linalg.norm(t)); return np.zeros(3) if n<1e-14 else t/n


def advance_rollback(
    mesh: SphereMesh,
    state: LithosphereState,
    memory: SubductionMemoryState,
    radius_km: float,
    dt_myr: float,
    params: RollbackParameters,
) -> tuple[Array, Array, RollbackDiagnostics]:
    """Update rollback bookkeeping and return (plate omega, backarc forcing)."""
    pcount=int(np.max(state.cell_plate))+1
    omega_sum=np.zeros((pcount,3),dtype=np.float64); omega_w=np.zeros(pcount,dtype=np.float64)
    forcing=np.zeros(mesh.cell_count,dtype=np.float64)
    rates=[]; distances=[]
    areas=mesh.physical_cell_areas_km2(radius_km)
    for key in sorted(memory.zones):
        z=memory.zones[key]
        if bool(getattr(z, "broken_off", False)):
            z.rollback_rate_km_per_myr = 0.0
            distances.append(float(getattr(z, "rollback_distance_km", 0.0)))
            continue
        # v0.19 fields are added dynamically for backwards-compatible v0.18
        # checkpoints reconstructed into the current dataclass.
        if not hasattr(z,'rollback_distance_km'): z.rollback_distance_km=0.0
        if not hasattr(z,'rollback_rate_km_per_myr'): z.rollback_rate_km_per_myr=0.0
        rate=zone_rollback_rate(z,params)
        z.rollback_rate_km_per_myr=float(rate)
        if rate<=0.0:
            distances.append(float(z.rollback_distance_km)); continue
        z.rollback_distance_km=min(float(params.rollback_distance_cap_km),float(z.rollback_distance_km)+rate*float(dt_myr))
        rates.append(rate); distances.append(float(z.rollback_distance_km))
        over=int(z.overriding_plate)
        if 0<=over<pcount:
            w=max(float(z.trench_length_km),1e-9)
            # Upper plate moves landward, so the trench retreats oceanward in
            # its reference frame.  omega x r points along +toward_overriding.
            omega_sum[over]+=w*(rate/max(float(radius_km),1e-9))*np.asarray(z.torque_axis,dtype=np.float64)
            omega_w[over]+=w

        # Local back-arc band: same physical mesh coordinates, restricted to
        # the overriding plate and the landward half-space behind the trench.
        r=np.asarray(z.trench_midpoint,dtype=np.float64); t=_toward_overriding(z)
        if not np.any(t): continue
        cells=np.flatnonzero(np.asarray(state.cell_plate)==over)
        if not len(cells): continue
        c=mesh.centroids[cells]
        ang=np.arccos(np.clip(c@r,-1.0,1.0)); dist=ang*float(radius_km)
        tangent=c-(c@r)[:,None]*r[None,:]
        landward=(tangent@t)>0.0
        inner=float(params.backarc_inner_distance_km); peak=float(params.backarc_peak_distance_km); outer=float(params.backarc_outer_distance_km)
        shape=np.zeros(len(cells),dtype=np.float64)
        left=landward&(dist>=inner)&(dist<=peak); right=landward&(dist>peak)&(dist<=outer)
        shape[left]=(dist[left]-inner)/max(peak-inner,1e-9)
        shape[right]=(outer-dist[right])/max(outer-peak,1e-9)
        amplitude=float(params.backarc_max_extension_forcing)*min(rate/max(float(params.max_rollback_rate_km_per_myr),1e-9),1.0)
        forcing[cells]=np.maximum(forcing[cells],amplitude*np.clip(shape,0.0,1.0))

    nz=omega_w>0
    omega_sum[nz]/=omega_w[nz,None]
    forced=forcing>1e-9
    diag=RollbackDiagnostics(
        time_myr=float(state.time_myr),active_rollback_zones=len(rates),
        mean_rollback_rate_km_per_myr=float(np.mean(rates)) if rates else 0.0,max_rollback_rate_km_per_myr=float(np.max(rates)) if rates else 0.0,
        mean_rollback_distance_km=float(np.mean(distances)) if distances else 0.0,max_rollback_distance_km=float(np.max(distances)) if distances else 0.0,
        backarc_forced_area_km2=float(np.sum(areas[forced])) if np.any(forced) else 0.0,max_backarc_extension_forcing=float(np.max(forcing)) if len(forcing) else 0.0,
    )
    return omega_sum,forcing,diag

__all__=['RollbackParameters','RollbackDiagnostics','zone_rollback_rate','advance_rollback']
