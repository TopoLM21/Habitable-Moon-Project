"""v0.9.4 late boundary nucleation and supercontinent-cycle memory.

The long v0.9.3 run exposed a structural failure mode: continent-continent
collisions could weld plates, while the model had almost no mechanism for a
new boundary to nucleate later *inside* the welded plate.  This module adds an
effective, material-memory model for that missing part of the plate cycle.

It is deliberately not a random plate generator.  New rift bands require a
large plate plus accumulated intraplate stress.  Their preferred paths follow
old collision seams, thermally insulated continental lids, and/or very old
oceanic lithosphere.  Once nucleated, a band must mature for tens of Myr before
the existing topology solver is allowed to split the plate.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from .kinematics import BoundaryRecord, BoundaryType
from .lithosphere import CrustType, LithosphereState
from .mesh import SphereMesh, connected_components
from .plates import PlateSystem

Array = np.ndarray


@dataclass(slots=True)
class LateTectonicsParameters:
    enabled: bool = True
    seam_relaxation_myr: float = 1800.0
    seam_gain_per_myr: float = 0.024
    seam_one_ring_fraction: float = 0.45
    # v0.12 physical spatial scale. ``None`` keeps the legacy one-ring rule.
    seam_spread_scale_km: float | None = None
    stress_relaxation_myr: float = 750.0
    plate_area_threshold_fraction: float = 0.18
    plate_area_full_stress_fraction: float = 0.55
    size_stress_rate_per_myr: float = 0.0020
    old_ocean_age_threshold_myr: float = 180.0
    old_ocean_stress_rate_per_myr: float = 0.0010
    supercontinent_heat_relaxation_myr: float = 1200.0
    supercontinent_continental_area_threshold_fraction: float = 0.10
    supercontinent_heat_rate_per_myr: float = 0.0012
    nucleation_min_plate_fraction: float = 0.20
    nucleation_score_threshold: float = 0.54
    nucleation_cooldown_myr: float = 80.0
    seam_weight: float = 0.38
    stress_weight: float = 0.34
    supercontinent_heat_weight: float = 0.18
    old_ocean_weight: float = 0.10
    rift_band_half_width_rings: int = 1
    rift_band_half_width_km: float | None = None
    rift_seed_extension: float = 0.12
    rift_extension_rate_per_myr: float = 0.012
    rift_thinning_km_per_myr: float = 0.095
    rift_min_path_cells: int = 20
    rift_max_path_cells: int = 420
    rift_min_path_length_km: float | None = None
    rift_max_path_length_km: float | None = None
    min_interior_rift_stress: float = 0.30
    # v0.24: old depleted continental roots divert newly nucleated rifts into
    # juvenile/orogenic belts and slow any band that does cross a cratonic core.
    craton_nucleation_penalty: float = 0.58
    craton_extension_resistance_gain: float = 0.68
    craton_min_extension_factor: float = 0.28


@dataclass(slots=True)
class LateTectonicsDiagnostics:
    time_myr: float
    largest_plate_fraction: float
    mean_intraplate_stress: float
    max_intraplate_stress: float
    mean_collision_seam_weakness: float
    max_collision_seam_weakness: float
    mean_supercontinent_heat: float
    max_supercontinent_heat: float
    old_ocean_fraction: float
    active_internal_rift_cells: int
    nucleated_rift: bool
    nucleated_plate: int
    nucleated_path_cells: int
    nucleated_path_length_km: float
    nucleation_score: float


def _field(value: Array | None, n: int) -> Array:
    if value is None:
        return np.zeros(n, dtype=np.float64)
    return np.asarray(value, dtype=np.float64).copy()


def _plate_boundary_cells(mesh: SphereMesh, owner: Array, plate: int) -> list[int]:
    cells=[]
    for c in np.flatnonzero(owner == plate):
        if any(int(owner[nb]) != plate for nb in mesh.neighbors[int(c)]):
            cells.append(int(c))
    return cells


def _edge_center_distance_km(mesh: SphereMesh, a: int, b: int, radius_km: float) -> float:
    return float(np.arccos(np.clip(np.dot(mesh.centroids[int(a)], mesh.centroids[int(b)]), -1.0, 1.0))) * float(radius_km)


def _path_length_km(mesh: SphereMesh, path: list[int], radius_km: float) -> float:
    if len(path) < 2:
        return 0.0
    return float(sum(_edge_center_distance_km(mesh, a, b, radius_km) for a, b in zip(path[:-1], path[1:])))


def _component_span_km(mesh: SphereMesh, cells: list[int] | Array, radius_km: float) -> float:
    idx=np.asarray(cells,dtype=np.int32)
    if len(idx)<2:return 0.0
    pts=np.asarray(mesh.centroids[idx],dtype=float)
    i=int(np.argmin(pts@pts[0]));j=int(np.argmin(pts@pts[i]))
    return float(np.arccos(np.clip(np.dot(pts[i],pts[j]),-1.0,1.0)))*float(radius_km)


def _dijkstra_path(
    mesh: SphereMesh,
    owner: Array,
    plate: int,
    start: int,
    goal: int,
    preference: Array,
    max_cells: int,
    radius_km: float,
) -> list[int]:
    """Shortest material path biased toward high weakness/stress preference."""
    dist={int(start):0.0}; prev: dict[int,int]={}; heap=[(0.0,int(start))]
    visited=0
    while heap:
        d,c=heapq.heappop(heap)
        if d != dist.get(c):
            continue
        visited += 1
        if c == goal:
            break
        # Hard guard only prevents pathological graph work; it is not a path cap.
        if visited > max(5000, 30*max_cells):
            break
        for nb in mesh.neighbors[c]:
            if int(owner[nb]) != plate:
                continue
            # High preference can reduce cost ~4x but never make it zero.
            local=0.25 + 0.75*(1.0-float(np.clip(preference[nb],0.0,1.0)))
            # Physical edge length prevents a fine mesh from making a path
            # artificially more expensive merely because it contains more cells.
            nd=d+local*_edge_center_distance_km(mesh,c,nb,radius_km)
            if nd < dist.get(int(nb),1e100):
                dist[int(nb)]=nd;prev[int(nb)]=c;heapq.heappush(heap,(nd,int(nb)))
    if goal not in dist:
        return []
    path=[int(goal)];cur=int(goal)
    while cur != start:
        cur=prev[cur];path.append(cur)
    path.reverse()
    if len(path) > int(max_cells):
        return []
    return path


def _choose_cross_plate_path(
    mesh: SphereMesh,
    owner: Array,
    plate: int,
    score: Array,
    params: LateTectonicsParameters,
    radius_km: float,
) -> list[int]:
    boundary=_plate_boundary_cells(mesh,owner,plate)
    if len(boundary) < 2:
        return []
    # Start at the boundary nearest the strongest weak/stressed interior.
    pcells=np.flatnonzero(owner==plate)
    seed=int(pcells[np.argmax(score[pcells])])
    dots=mesh.centroids[np.asarray(boundary)] @ mesh.centroids[seed]
    start=int(boundary[int(np.argmax(dots))])
    # Opposite boundary point gives a plate-crossing band rather than a small notch.
    dots2=mesh.centroids[np.asarray(boundary)] @ mesh.centroids[start]
    order=np.argsort(dots2)  # most angularly distant first
    best=[];best_value=-1e30
    for idx in order[:min(40,len(order))]:
        goal=int(boundary[int(idx)])
        max_cells=int(params.rift_max_path_cells) if params.rift_max_path_length_km is None else int(mesh.cell_count)
        path=_dijkstra_path(mesh,owner,plate,start,goal,score,max_cells,radius_km)
        plen=_path_length_km(mesh,path,radius_km)
        if params.rift_min_path_length_km is not None:
            if plen < float(params.rift_min_path_length_km): continue
        elif len(path) < int(params.rift_min_path_cells):
            continue
        if params.rift_max_path_length_km is not None and plen > float(params.rift_max_path_length_km):
            continue
        vals=score[np.asarray(path,dtype=np.int32)]
        # Favour a physically weak path while still spanning a large angular distance.
        ang=float(np.arccos(np.clip(np.dot(mesh.centroids[start],mesh.centroids[goal]),-1.0,1.0)))
        value=float(np.mean(vals))+0.18*ang
        if value>best_value:
            best_value=value;best=path
    return best


def _grow_band(mesh: SphereMesh, owner: Array, plate: int, path: list[int], rings: int) -> set[int]:
    band=set(map(int,path));front=set(band)
    for _ in range(max(0,int(rings))):
        nxt=set()
        for c in front:
            for nb in mesh.neighbors[c]:
                if int(owner[nb])==plate and int(nb) not in band:
                    nxt.add(int(nb))
        band.update(nxt);front=nxt
    return band


def _grow_band_km(mesh: SphereMesh, owner: Array, plate: int, path: list[int], radius_km: float, width_km: float) -> set[int]:
    """Grow a path by physical geodesic distance rather than neighbor rings."""
    band=set(map(int,path))
    if width_km<=0.0 or not band:return band
    dist={c:0.0 for c in band};heap=[(0.0,c) for c in sorted(band)];heapq.heapify(heap)
    while heap:
        d,c=heapq.heappop(heap)
        if d!=dist.get(c):continue
        for nb in mesh.neighbors[c]:
            nb=int(nb)
            if int(owner[nb])!=plate:continue
            nd=d+_edge_center_distance_km(mesh,c,nb,radius_km)
            if nd>float(width_km)+1e-9:continue
            if nd<dist.get(nb,1e100):dist[nb]=nd;heapq.heappush(heap,(nd,nb))
    return set(dist)


def _seam_spread_kernel(mesh: SphereMesh, owner: Array, sources: set[int], radius_km: float, scale_km: float) -> Array:
    """Resolution-stable max Gaussian-like kernel around collision contacts."""
    out=np.zeros(mesh.cell_count,dtype=np.float64)
    if not sources:return out
    scale=max(float(scale_km),1e-9);limit=3.0*scale
    # If the requested physical seam is narrower than a mesh cell, store the
    # unresolved fault as a diluted cell-average weakness.  This keeps the
    # area-integrated weakening approximately invariant under mesh refinement.
    mean_linear_km=float(np.sqrt(np.mean(mesh.physical_cell_areas_km2(radius_km))))
    amplitude=float(np.clip(scale/max(mean_linear_km,1e-9),0.0,1.0))
    dist={int(c):0.0 for c in sources};heap=[(0.0,int(c)) for c in sorted(sources)];heapq.heapify(heap)
    while heap:
        d,c=heapq.heappop(heap)
        if d!=dist.get(c):continue
        out[c]=max(out[c],amplitude*float(np.exp(-d/scale)))
        for nb in mesh.neighbors[c]:
            nb=int(nb)
            if int(owner[nb])!=int(owner[c]):continue
            nd=d+_edge_center_distance_km(mesh,c,nb,radius_km)
            if nd>limit:continue
            if nd<dist.get(nb,1e100):dist[nb]=nd;heapq.heappush(heap,(nd,nb))
    return out


def advance_late_tectonics(
    mesh: SphereMesh,
    state: LithosphereState,
    system: PlateSystem,
    boundaries: list[BoundaryRecord],
    dt_myr: float,
    radius_km: float,
    tectonic_activity_factor: float,
    params: LateTectonicsParameters,
    *,
    last_split_time_myr: float = -1e30,
) -> LateTectonicsDiagnostics:
    n=mesh.cell_count
    seam=_field(state.collision_seam_weakness,n)
    stress=_field(state.intraplate_stress,n)
    heat=_field(state.supercontinent_heat,n)
    ext=np.zeros(n,dtype=np.float64) if state.rift_extension is None else np.asarray(state.rift_extension,dtype=np.float64).copy()
    ext_age=np.zeros(n,dtype=np.float64) if state.extension_age_myr is None else np.asarray(state.extension_age_myr,dtype=np.float64).copy()
    owner=np.asarray(state.cell_plate,dtype=np.int32)
    areas=mesh.physical_cell_areas_km2(radius_km);total_area=float(np.sum(areas))

    if not params.enabled:
        state.collision_seam_weakness=seam;state.intraplate_stress=stress;state.supercontinent_heat=heat
        return LateTectonicsDiagnostics(float(state.time_myr),0.0,float(np.mean(stress)),float(np.max(stress)),float(np.mean(seam)),float(np.max(seam)),float(np.mean(heat)),float(np.max(heat)),0.0,0,False,-1,0,0.0,0.0)

    seam*=np.exp(-float(dt_myr)/max(float(params.seam_relaxation_myr),1e-9))
    stress*=np.exp(-float(dt_myr)/max(float(params.stress_relaxation_myr),1e-9))
    heat*=np.exp(-float(dt_myr)/max(float(params.supercontinent_heat_relaxation_myr),1e-9))

    # Collision seams are material weaknesses. Record them before topology welding.
    seam_direct=set()
    for b in boundaries:
        if b.boundary_type != BoundaryType.CONVERGENT:
            continue
        if int(state.crust_type[b.face_a]) != int(CrustType.CONTINENTAL) or int(state.crust_type[b.face_b]) != int(CrustType.CONTINENTAL):
            continue
        seam_direct.add(int(b.face_a));seam_direct.add(int(b.face_b))
    if seam_direct:
        if params.seam_spread_scale_km is not None:
            kernel=_seam_spread_kernel(mesh,owner,seam_direct,radius_km,float(params.seam_spread_scale_km))
            seam+=float(params.seam_gain_per_myr)*float(dt_myr)*kernel
        else:
            direct=np.asarray(sorted(seam_direct),dtype=np.int32)
            seam[direct]+=float(params.seam_gain_per_myr)*float(dt_myr)
            for c in direct:
                for nb in mesh.neighbors[int(c)]:
                    seam[int(nb)]+=float(params.seam_gain_per_myr)*float(params.seam_one_ring_fraction)*float(dt_myr)
    seam=np.clip(seam,0.0,1.0)

    plate_areas=np.bincount(owner,weights=areas,minlength=len(system.plates)).astype(float)
    cont_mask=state.crust_type==int(CrustType.CONTINENTAL)
    ocean_mask=~cont_mask
    cont_areas=np.bincount(owner,weights=areas*cont_mask,minlength=len(system.plates)).astype(float)
    largest=float(np.max(plate_areas)/total_area) if len(plate_areas) else 0.0

    # Large plates accumulate membrane stress because their boundaries cannot
    # accommodate all mantle-driving torque coherently. Cooling does not turn
    # this off; lower activity merely slows the accumulation moderately.
    act_scale=0.55+0.45*float(np.clip(tectonic_activity_factor,0.0,1.5))
    lo=float(params.plate_area_threshold_fraction);hi=max(float(params.plate_area_full_stress_fraction),lo+1e-6)
    for p,pa in enumerate(plate_areas):
        frac=float(pa/total_area)
        size_score=float(np.clip((frac-lo)/(hi-lo),0.0,1.0))
        cells=np.flatnonzero(owner==p)
        if len(cells)==0: continue
        stress[cells]+=float(params.size_stress_rate_per_myr)*size_score*act_scale*float(dt_myr)
        # A large continental lid thermally insulates the mantle beneath it.
        global_cont_frac=float(cont_areas[p]/total_area)
        lid_score=float(np.clip((global_cont_frac-float(params.supercontinent_continental_area_threshold_fraction))/0.16,0.0,1.0))
        if lid_score>0:
            cc=cells[cont_mask[cells]]
            heat[cc]+=float(params.supercontinent_heat_rate_per_myr)*lid_score*float(dt_myr)

    old_ocean=ocean_mask & (state.crust_age_myr>=float(params.old_ocean_age_threshold_myr))
    stress[old_ocean]+=float(params.old_ocean_stress_rate_per_myr)*act_scale*float(dt_myr)
    stress=np.clip(stress,0.0,1.0);heat=np.clip(heat,0.0,1.0)

    old_score=np.clip((state.crust_age_myr-float(params.old_ocean_age_threshold_myr))/max(float(params.old_ocean_age_threshold_myr),1.0),0.0,1.0)*ocean_mask
    craton_strength=(
        np.zeros(n,dtype=np.float64)
        if state.craton_strength is None
        else np.clip(np.asarray(state.craton_strength,dtype=np.float64),0.0,1.0)
    )
    craton_extension_factor=np.clip(
        1.0-float(params.craton_extension_resistance_gain)*craton_strength,
        float(params.craton_min_extension_factor),
        1.0,
    )
    score=np.clip(
        float(params.seam_weight)*seam+
        float(params.stress_weight)*stress+
        float(params.supercontinent_heat_weight)*heat+
        float(params.old_ocean_weight)*old_score,
        0.0,1.0,
    )
    score*=np.clip(1.0-float(params.craton_nucleation_penalty)*craton_strength,0.10,1.0)

    # Keep an already nucleated internal band maturing even before an actual
    # kinematic gap exists. Cells on an existing plate boundary are excluded.
    boundary_cell=np.zeros(n,dtype=bool)
    for fa,fb,_,_ in mesh.shared_edges:
        if owner[fa]!=owner[fb]: boundary_cell[fa]=True;boundary_cell[fb]=True
    active=(ext>0.05)&(stress>=float(params.min_interior_rift_stress))&(~boundary_cell)
    if np.any(active):
        forcing=np.clip(0.5+score[active],0.5,1.5)*craton_extension_factor[active]
        ext[active]+=float(params.rift_extension_rate_per_myr)*forcing*float(dt_myr)
        ext_age[active]+=craton_extension_factor[active]*float(dt_myr)
        cactive=np.flatnonzero(active & cont_mask)
        if len(cactive):
            f=np.clip(0.5+score[cactive],0.5,1.5)*craton_extension_factor[cactive]
            state.crust_thickness_km[cactive]=np.maximum(16.0,state.crust_thickness_km[cactive]-float(params.rift_thinning_km_per_myr)*f*float(dt_myr))

    nucleated=False;nplate=-1;npath=0;npath_km=0.0;nscore=0.0
    # Do not seed another line while a sizeable interior line is maturing.
    if params.rift_min_path_length_km is not None:
        active_components=connected_components(np.flatnonzero(active),mesh.neighbors) if np.any(active) else []
        has_maturing=any(_component_span_km(mesh,c,radius_km)>=float(params.rift_min_path_length_km) for c in active_components)
    else:
        has_maturing=int(np.sum(active))>=int(params.rift_min_path_cells)
    if (not has_maturing) and float(state.time_myr)-float(last_split_time_myr)>=float(params.nucleation_cooldown_myr):
        order=sorted(range(len(system.plates)),key=lambda p:(-plate_areas[p],p))
        for p in order:
            pfrac=float(plate_areas[p]/total_area)
            if pfrac<float(params.nucleation_min_plate_fraction): continue
            cells=np.flatnonzero(owner==p)
            maxscore=float(np.max(score[cells])) if len(cells) else 0.0
            meanstress=float(np.mean(stress[cells])) if len(cells) else 0.0
            # Large-plate stress is a gate; local seams/heat choose the path.
            if maxscore<float(params.nucleation_score_threshold) or meanstress<0.18:
                continue
            path=_choose_cross_plate_path(mesh,owner,p,score,params,radius_km)
            if not path: continue
            if params.rift_band_half_width_km is not None:
                band=_grow_band_km(mesh,owner,p,path,radius_km,float(params.rift_band_half_width_km))
            else:
                band=_grow_band(mesh,owner,p,path,int(params.rift_band_half_width_rings))
            idx=np.asarray(sorted(band),dtype=np.int32)
            ext[idx]=np.maximum(ext[idx],float(params.rift_seed_extension)*craton_extension_factor[idx])
            ext_age[idx]=np.maximum(ext_age[idx],float(dt_myr)*craton_extension_factor[idx])
            stress[idx]=np.maximum(stress[idx],0.45)
            cc=idx[cont_mask[idx]]
            if len(cc):
                state.crust_thickness_km[cc]=np.maximum(16.0,state.crust_thickness_km[cc]-0.5*float(params.rift_thinning_km_per_myr)*craton_extension_factor[cc]*float(dt_myr))
            nucleated=True;nplate=int(p);npath=int(len(idx));npath_km=_path_length_km(mesh,path,radius_km);nscore=float(np.mean(score[idx]));break

    state.rift_extension=np.clip(ext,0.0,4.0)
    state.extension_age_myr=np.clip(ext_age,0.0,5000.0)
    state.collision_seam_weakness=seam
    state.intraplate_stress=stress
    state.supercontinent_heat=heat

    return LateTectonicsDiagnostics(
        time_myr=float(state.time_myr),
        largest_plate_fraction=largest,
        mean_intraplate_stress=float(np.mean(stress)),
        max_intraplate_stress=float(np.max(stress)),
        mean_collision_seam_weakness=float(np.mean(seam)),
        max_collision_seam_weakness=float(np.max(seam)),
        mean_supercontinent_heat=float(np.mean(heat)),
        max_supercontinent_heat=float(np.max(heat)),
        old_ocean_fraction=float(np.sum(areas[old_ocean])/max(np.sum(areas[ocean_mask]),1e-30)) if np.any(ocean_mask) else 0.0,
        active_internal_rift_cells=int(np.sum(active)),
        nucleated_rift=bool(nucleated),
        nucleated_plate=int(nplate),
        nucleated_path_cells=int(npath),
        nucleated_path_length_km=float(npath_km),
        nucleation_score=float(nscore),
    )


__all__=["LateTectonicsParameters","LateTectonicsDiagnostics","advance_late_tectonics"]
