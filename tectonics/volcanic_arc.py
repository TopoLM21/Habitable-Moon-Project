"""v0.21 slab-geometry volcanic-arc forcing.

Arc magmatism is displaced landward from the trench according to the remembered
slab dip and a dehydration/melting depth.  The module is intentionally 2.5-D:
it produces a dimensionless surface forcing field which the existing
continental-cycle and topography modules consume.  It does not create crust by
itself.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

from .cpu_runtime import current_execution
from .lithosphere import LithosphereState, continental_material_fields
from .kinematics import BoundaryRecord, BoundaryType
from .mesh import SphereMesh
from .subduction_memory import SubductionMemoryState

Array = np.ndarray


@dataclass(slots=True, frozen=True)
class VolcanicArcParameters:
    enabled: bool = True
    reference_slab_depth_km: float = 105.0
    min_slab_depth_km: float = 70.0
    max_slab_depth_km: float = 130.0
    convergence_reference_km_per_myr: float = 50.0
    convergence_depth_sensitivity_km: float = 12.0
    min_active_slab_depth_km: float = 55.0
    min_active_age_myr: float = 8.0
    arc_half_width_km: float = 140.0
    arc_outer_width_km: float = 320.0
    max_trench_arc_distance_km: float = 420.0
    min_trench_arc_distance_km: float = 55.0
    max_forcing: float = 1.35
    post_breakoff_enabled: bool = True
    post_breakoff_peak_fraction: float = 0.22
    post_breakoff_decay_myr: float = 10.0
    post_breakoff_duration_myr: float = 32.0
    post_breakoff_width_km: float = 260.0


@dataclass(slots=True)
class VolcanicArcDiagnostics:
    time_myr: float
    active_arc_zones: int
    post_breakoff_pulses: int
    forced_area_km2: float
    juvenile_arc_forced_area_km2: float
    continental_arc_forced_area_km2: float
    mean_target_slab_depth_km: float
    mean_trench_arc_distance_km: float
    max_trench_arc_distance_km: float
    max_arc_forcing: float


def target_slab_depth_km(convergence_rate_km_per_myr: float, params: VolcanicArcParameters) -> float:
    """Effective slab depth beneath the volcanic front.

    Faster descent is assigned a slightly shallower target, consistent with the
    observed inverse relationship between sub-arc slab depth and descent speed.
    The dependence is deliberately weak and bounded by the observed arc range.
    """
    ref=max(float(params.convergence_reference_km_per_myr),1e-9)
    x=(float(convergence_rate_km_per_myr)-ref)/ref
    depth=float(params.reference_slab_depth_km)-float(params.convergence_depth_sensitivity_km)*np.tanh(x)
    return float(np.clip(depth,float(params.min_slab_depth_km),float(params.max_slab_depth_km)))


def trench_arc_distance_km(dip_deg: float, target_depth_km: float, params: VolcanicArcParameters) -> float:
    dip=np.deg2rad(float(np.clip(dip_deg,5.0,85.0)))
    dist=float(target_depth_km)/max(float(np.tan(dip)),1e-6)
    return float(np.clip(dist,float(params.min_trench_arc_distance_km),float(params.max_trench_arc_distance_km)))


def _toward_overriding(zone) -> Array:
    r=np.asarray(zone.trench_midpoint,dtype=np.float64)
    r=r/max(float(np.linalg.norm(r)),1e-30)
    torque=np.asarray(zone.torque_axis,dtype=np.float64)
    t=np.cross(torque,r)
    t-=r*float(np.dot(t,r))
    n=float(np.linalg.norm(t))
    if n < 1e-12:
        # Deterministic tangent fallback.
        ref=np.array([0.0,0.0,1.0]) if abs(float(r[2]))<0.9 else np.array([1.0,0.0,0.0])
        t=ref-r*float(np.dot(ref,r));n=float(np.linalg.norm(t))
    return t/max(n,1e-30)


def projected_arc_center(zone, radius_km: float, params: VolcanicArcParameters) -> tuple[Array,float,float]:
    depth=target_slab_depth_km(float(zone.convergence_rate_km_per_myr),params)
    distance=trench_arc_distance_km(float(zone.dip_deg),depth,params)
    r=np.asarray(zone.trench_midpoint,dtype=np.float64);r=r/max(float(np.linalg.norm(r)),1e-30)
    t=_toward_overriding(zone)
    theta=distance/max(float(radius_km),1e-9)
    p=np.cos(theta)*r+np.sin(theta)*t
    p=p/max(float(np.linalg.norm(p)),1e-30)
    return p,float(depth),float(distance)


def _paint_gaussian(mesh: SphereMesh, tree: cKDTree, center: Array, plate_id: int, state: LithosphereState,
                    radius_km: float, sigma_km: float, outer_km: float, amplitude: float, field: Array) -> None:
    if amplitude <= 0.0 or outer_km <= 0.0:
        return
    theta=min(float(outer_km)/max(float(radius_km),1e-9),np.pi)
    chord=2.0*np.sin(0.5*theta)+1e-12
    cells=np.asarray(tree.query_ball_point(np.asarray(center,dtype=float),r=chord),dtype=np.int32)
    if not len(cells):
        return
    cells=cells[np.asarray(state.cell_plate[cells],dtype=np.int32)==int(plate_id)]
    if not len(cells):
        return
    dist=np.arccos(np.clip(mesh.centroids[cells]@center,-1.0,1.0))*float(radius_km)
    sig=max(float(sigma_km),1e-9)
    values=float(amplitude)*np.exp(-0.5*(dist/sig)**2)
    field[cells]=np.maximum(field[cells],values)


def _paint_gaussian_batch(
    mesh: SphereMesh,
    tree: cKDTree,
    centers: list[Array],
    plate_ids: list[int],
    amplitudes: list[float],
    state: LithosphereState,
    radius_km: float,
    sigma_km: float,
    outer_km: float,
    field: Array,
    workers: int,
) -> None:
    """Paint many independent fronts with one parallel spatial-tree query.

    Geometry and Gaussian values are still evaluated in their original order;
    only the read-only neighborhood lookup is batched. This keeps every field
    value bitwise identical to repeated ``_paint_gaussian`` calls.
    """
    if not centers or outer_km <= 0.0:
        return
    theta=min(float(outer_km)/max(float(radius_km),1e-9),np.pi)
    chord=2.0*np.sin(0.5*theta)+1e-12
    center_array=np.asarray(centers,dtype=float)
    cell_groups=tree.query_ball_point(center_array,r=chord,workers=int(workers))
    sig=max(float(sigma_km),1e-9)
    for center,plate_id,amplitude,group in zip(center_array,plate_ids,amplitudes,cell_groups):
        if amplitude <= 0.0:
            continue
        cells=np.asarray(group,dtype=np.int32)
        if not len(cells):
            continue
        cells=cells[np.asarray(state.cell_plate[cells],dtype=np.int32)==int(plate_id)]
        if not len(cells):
            continue
        dist=np.arccos(np.clip(mesh.centroids[cells]@center,-1.0,1.0))*float(radius_km)
        values=float(amplitude)*np.exp(-0.5*(dist/sig)**2)
        field[cells]=np.maximum(field[cells],values)


def _boundary_toward_overriding(mesh: SphereMesh, b: BoundaryRecord, subducting_plate: int) -> tuple[Array,Array]:
    r=np.asarray(b.midpoint,dtype=np.float64);r=r/max(float(np.linalg.norm(r)),1e-30)
    d=np.asarray(mesh.centroids[int(b.face_b)]-mesh.centroids[int(b.face_a)],dtype=np.float64)
    d-=r*float(np.dot(d,r)); d=d/max(float(np.linalg.norm(d)),1e-30)
    toward=d if int(subducting_plate)==int(b.plate_a) else -d
    return r,toward


def _project_point(r: Array, toward: Array, distance_km: float, radius_km: float) -> Array:
    theta=float(distance_km)/max(float(radius_km),1e-9)
    p=np.cos(theta)*np.asarray(r,dtype=float)+np.sin(theta)*np.asarray(toward,dtype=float)
    return p/max(float(np.linalg.norm(p)),1e-30)


def compute_volcanic_arc_forcing(
    mesh: SphereMesh,
    state: LithosphereState,
    memory: SubductionMemoryState | None,
    radius_km: float,
    params: VolcanicArcParameters,
    boundaries: list[BoundaryRecord] | None = None,
) -> tuple[Array, VolcanicArcDiagnostics]:
    field=np.zeros(mesh.cell_count,dtype=np.float64)
    areas=mesh.physical_cell_areas_km2(float(radius_km))
    frac,_=continental_material_fields(state,areas)
    if (not params.enabled) or memory is None:
        d=VolcanicArcDiagnostics(float(state.time_myr),0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
        return field,d
    execution=current_execution()
    optimized=execution is not None and execution.arc_kernels
    tree=execution.geometry(mesh).tree if optimized else cKDTree(mesh.centroids)
    # A physical volcanic front can be narrower than a coarse diagnostic mesh.
    # Represent it as an unresolved sub-grid band instead of silently losing it.
    cell_scale_km=float(np.sqrt(np.mean(areas))) if len(areas) else 0.0
    active_sigma=max(float(params.arc_half_width_km),0.45*cell_scale_km)
    active_outer=max(float(params.arc_outer_width_km),1.05*cell_scale_km)
    break_sigma=max(float(params.post_breakoff_width_km),0.45*cell_scale_km)
    break_outer=max(2.2*float(params.post_breakoff_width_km),1.05*cell_scale_km)
    depths=[];distances=[];active=0;pulses=0
    if optimized:
        by_pair: dict[tuple[int, int], list[BoundaryRecord]] = {}
        if boundaries is not None:
            for b in boundaries:
                if b.boundary_type == BoundaryType.CONVERGENT:
                    pair=tuple(sorted((int(b.plate_a),int(b.plate_b))))
                    by_pair.setdefault(pair,[]).append(b)

        active_centers=[];active_plates=[];active_amplitudes=[]
        break_centers=[];break_plates=[];break_amplitudes=[]
        for key in sorted(memory.zones):
            z=memory.zones[key]
            if bool(getattr(z,'broken_off',False)):
                age=float(getattr(z,'post_breakoff_age_myr',0.0))
                if bool(params.post_breakoff_enabled) and age <= float(params.post_breakoff_duration_myr):
                    center,depth,distance=projected_arc_center(z,radius_km,params)
                    amp=float(params.post_breakoff_peak_fraction)*np.exp(-age/max(float(params.post_breakoff_decay_myr),1e-9))
                    break_centers.append(center);break_plates.append(int(z.overriding_plate));break_amplitudes.append(amp)
                    pulses+=1;depths.append(depth);distances.append(distance)
                continue
            if (not bool(z.active)) or float(z.active_age_myr)<float(params.min_active_age_myr) or float(z.slab_depth_km)<float(params.min_active_slab_depth_km):
                continue
            _,depth,distance=projected_arc_center(z,radius_km,params)
            conv=np.clip(float(z.convergence_rate_km_per_myr)/max(float(params.convergence_reference_km_per_myr),1e-9),0.15,float(params.max_forcing))
            amp=min(float(params.max_forcing),float(conv))
            pair=tuple(sorted((int(z.subducting_plate),int(z.overriding_plate))))
            zone_boundaries=tuple(by_pair.get(pair,())) if boundaries is not None else ()
            painted=False
            for b in zone_boundaries:
                r,toward=_boundary_toward_overriding(mesh,b,int(z.subducting_plate))
                center=_project_point(r,toward,distance,radius_km)
                active_centers.append(center);active_plates.append(int(z.overriding_plate));active_amplitudes.append(amp)
                painted=True
            if not painted:
                center,_,_=projected_arc_center(z,radius_km,params)
                active_centers.append(center);active_plates.append(int(z.overriding_plate));active_amplitudes.append(amp)
            active+=1;depths.append(depth);distances.append(distance)

        _paint_gaussian_batch(mesh,tree,active_centers,active_plates,active_amplitudes,state,
                              radius_km,active_sigma,active_outer,field,execution.workers)
        _paint_gaussian_batch(mesh,tree,break_centers,break_plates,break_amplitudes,state,
                              radius_km,break_sigma,break_outer,field,execution.workers)
        execution.arc_calls+=1
        execution.arc_tasks+=len(active_centers)+len(break_centers)
    else:
        for key in sorted(memory.zones):
            z=memory.zones[key]
            if bool(getattr(z,'broken_off',False)):
                age=float(getattr(z,'post_breakoff_age_myr',0.0))
                if bool(params.post_breakoff_enabled) and age <= float(params.post_breakoff_duration_myr):
                    center,depth,distance=projected_arc_center(z,radius_km,params)
                    amp=float(params.post_breakoff_peak_fraction)*np.exp(-age/max(float(params.post_breakoff_decay_myr),1e-9))
                    _paint_gaussian(mesh,tree,center,int(z.overriding_plate),state,radius_km,
                                    break_sigma,break_outer,amp,field)
                    pulses+=1;depths.append(depth);distances.append(distance)
                continue
            if (not bool(z.active)) or float(z.active_age_myr)<float(params.min_active_age_myr) or float(z.slab_depth_km)<float(params.min_active_slab_depth_km):
                continue
            _,depth,distance=projected_arc_center(z,radius_km,params)
            # Once a slab is deep enough to feed the mantle wedge, arc productivity
            # follows convergence much like the legacy v0.8 cycle.
            conv=np.clip(float(z.convergence_rate_km_per_myr)/max(float(params.convergence_reference_km_per_myr),1e-9),0.15,float(params.max_forcing))
            amp=min(float(params.max_forcing),float(conv))
            painted=False
            if boundaries is not None:
                for b in boundaries:
                    if b.boundary_type != BoundaryType.CONVERGENT:
                        continue
                    pa,pb=int(b.plate_a),int(b.plate_b)
                    if {pa,pb}!={int(z.subducting_plate),int(z.overriding_plate)}:
                        continue
                    r,toward=_boundary_toward_overriding(mesh,b,int(z.subducting_plate))
                    center=_project_point(r,toward,distance,radius_km)
                    _paint_gaussian(mesh,tree,center,int(z.overriding_plate),state,radius_km,active_sigma,active_outer,amp,field)
                    painted=True
            if not painted:
                center,_,_=projected_arc_center(z,radius_km,params)
                _paint_gaussian(mesh,tree,center,int(z.overriding_plate),state,radius_km,active_sigma,active_outer,amp,field)
            active+=1;depths.append(depth);distances.append(distance)
    forced=field>0.05
    # A mixed cell contributes continuously to oceanic/island-arc versus
    # continental-arc area; this mirrors v0.15 material-aware topography.
    juvenile_area=float(np.sum(areas*forced*(1.0-np.clip(frac,0.0,1.0))))
    continental_area=float(np.sum(areas*forced*np.clip(frac,0.0,1.0)))
    diag=VolcanicArcDiagnostics(
        time_myr=float(state.time_myr),active_arc_zones=int(active),post_breakoff_pulses=int(pulses),
        forced_area_km2=float(np.sum(areas[forced])) if np.any(forced) else 0.0,
        juvenile_arc_forced_area_km2=juvenile_area,continental_arc_forced_area_km2=continental_area,
        mean_target_slab_depth_km=float(np.mean(depths)) if depths else 0.0,
        mean_trench_arc_distance_km=float(np.mean(distances)) if distances else 0.0,
        max_trench_arc_distance_km=float(np.max(distances)) if distances else 0.0,
        max_arc_forcing=float(np.max(field)) if len(field) else 0.0,
    )
    return field,diag


__all__=['VolcanicArcParameters','VolcanicArcDiagnostics','target_slab_depth_km','trench_arc_distance_km','projected_arc_center','compute_volcanic_arc_forcing']
