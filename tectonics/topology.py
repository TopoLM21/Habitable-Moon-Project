"""v0.7 dynamic plate topology: breakup, collision welding and plate death.

The earlier prototypes evolved crust and Euler velocities while keeping a fixed
set of plate identities.  v0.7 makes the *plate graph* itself dynamic.

This is an effective geological model, not a brittle-fracture continuum solver:

* breakup: a young, tidally weakened rift band must geometrically disconnect a
  plate into at least two sufficiently large connected blocks;
* welding/merge: a continent-continent convergent contact must persist for a
  configurable time and length before the two plates are mechanically welded;
* plate death: a plate that shrinks below a minimum cell count is absorbed by
  the neighbour with the longest shared boundary.

After every topology event IDs are compacted to 0..N-1, which preserves the
indexing assumptions of the kinematics/dynamics modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict, deque

import numpy as np

from .dynamics import angular_velocity_vectors, system_from_omega
from .kinematics import BoundaryRecord, BoundaryType, classify_boundaries
from .lithosphere import CrustType, LithosphereState, continental_material_fields, effective_continental_thickness_km
from .mesh import SphereMesh, connected_components
from .plates import Plate, PlateSystem

Array = np.ndarray


@dataclass(slots=True)
class PlateTopologyParameters:
    # Breakup: recent oceanic rift with inherited extensional strain acts as the cut.
    split_enabled: bool = True
    split_young_ocean_age_myr: float = 16.0
    split_min_postrift_extension: float = 0.55
    split_max_thinned_continent_km: float = 24.0
    split_min_extension_age_myr: float = 24.0
    # v0.9.4: mature internally nucleated bands may cut old oceanic or
    # continental lithosphere if material stress/seam memory is high.
    split_late_stress_threshold: float = 0.42
    split_late_seam_threshold: float = 0.34
    # Legacy fallback for pre-v0.9.2 states that do not carry rift_extension.
    split_min_postrift_damage: float = 0.035
    split_min_rift_cells: int = 8
    split_min_child_cells: int = 180
    split_min_rift_span_km: float | None = None
    split_min_child_area_km2: float | None = None
    split_differential_speed_deg_per_myr: float = 0.08
    split_cooldown_myr: float = 12.0
    disconnect_min_child_cells: int = 180
    disconnect_min_child_area_km2: float | None = None

    # v0.9.5 progressive continent-continent collision.  A collision first
    # forms a deforming mechanical coupling zone while the two plate IDs remain
    # distinct.  Permanent welding is only allowed after a long mature
    # collision followed by a separate low-relative-speed quiet phase.
    merge_enabled: bool = True
    merge_min_continental_boundary_km: float = 900.0
    collision_min_continental_fraction: float = 0.25
    collision_initial_max_relative_speed_km_per_myr: float = 65.0
    collision_contact_max_divergence_km_per_myr: float = 8.0
    collision_coupling_start_myr: float = 20.0
    collision_coupling_timescale_myr: float = 100.0
    collision_coupling_max_step_fraction: float = 0.08
    weld_min_collision_age_myr: float = 160.0
    weld_quiet_persistence_myr: float = 80.0
    weld_max_relative_speed_km_per_myr: float = 8.0
    weld_max_normal_divergence_km_per_myr: float = 1.5
    # ``area_weighted`` is the historical effective rule. ``inertia_tensor``
    # is an opt-in diagnostic alternative that conserves angular momentum for
    # the actual cell geometry of both welded plates.
    merge_kinematics_rule: str = "area_weighted"

    # v0.9.6: active continent-continent collision suppresses local extension.
    # This prevents a compressional weld zone from being thinned by a nearby
    # remapping gap at the same time.  Suppression is local and fades to zero
    # away from the collisional contact.
    collision_rift_suppression_start_myr: float = 32.0
    collision_rift_suppression_max: float = 0.75
    collision_rift_suppression_one_ring_fraction: float = 0.45

    # Very small plates are treated as doomed microplates/fragments.
    min_plate_cells: int = 90
    min_plate_area_km2: float | None = None
    # A plate must remain below the physical microplate-area threshold for
    # this long before numerical cleanup may absorb it.  This hysteresis
    # prevents a one-step coarse-grid area fluctuation from changing the
    # topology event class.  Zero preserves legacy immediate cleanup.
    min_plate_persistence_myr: float = 0.0
    max_events_per_step: int = 2


@dataclass(slots=True)
class TopologyEvent:
    time_myr: float
    kind: str
    parents: tuple[int, ...]
    children: tuple[int, ...]
    affected_cells: int
    detail: str


@dataclass(slots=True)
class TopologyDiagnostics:
    time_myr: float
    plate_count_before: int
    plate_count_after: int
    split_events: int
    merge_events: int
    absorbed_small_plates: int
    active_collision_pairs: int
    mature_collision_pairs: int
    quiet_weld_pairs: int
    max_collision_age_myr: float
    max_quiet_weld_age_myr: float
    min_plate_cells: int
    mean_plate_cells: float
    max_plate_cells: int
    min_plate_area_km2: float
    mean_plate_area_km2: float
    max_plate_area_km2: float
    topology_changed: bool


def _edge_length_km(mesh: SphereMesh, b: BoundaryRecord, radius_km: float) -> float:
    u = mesh.vertices[b.vertex_u]
    v = mesh.vertices[b.vertex_v]
    return float(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))) * float(radius_km)


def _plate_cell_counts(cell_plate: Array, pcount: int) -> Array:
    return np.bincount(np.asarray(cell_plate, dtype=np.int32), minlength=pcount).astype(np.int64)


def _plate_area_weights(mesh: SphereMesh, cell_plate: Array, radius_km: float, pcount: int) -> Array:
    areas = mesh.physical_cell_areas_km2(radius_km)
    return np.bincount(np.asarray(cell_plate, dtype=np.int32), weights=areas, minlength=pcount).astype(float)


def _plate_inertia_tensor(
    mesh: SphereMesh,
    cells: Array,
    cell_areas_km2: Array,
) -> Array:
    """Return a thin-shell inertia tensor up to a shared density/radius factor."""

    indices = np.asarray(cells, dtype=np.int32)
    points = np.asarray(mesh.centroids[indices], dtype=np.float64)
    weights = np.asarray(cell_areas_km2[indices], dtype=np.float64)
    second_moment = np.einsum("i,ij,ik->jk", weights, points, points)
    return float(np.sum(weights)) * np.eye(3) - second_moment


def _merged_angular_velocity(
    mesh: SphereMesh,
    owner: Array,
    omega: Array,
    a: int,
    b: int,
    radius_km: float,
    rule: str,
) -> Array:
    areas = _plate_area_weights(mesh, owner, radius_km, len(omega))
    if rule == "area_weighted":
        return (areas[a] * omega[a] + areas[b] * omega[b]) / max(
            float(areas[a] + areas[b]), 1e-30
        )
    if rule != "inertia_tensor":
        raise ValueError(
            "merge_kinematics_rule must be 'area_weighted' or 'inertia_tensor'"
        )
    cell_areas = mesh.physical_cell_areas_km2(radius_km)
    inertia_a = _plate_inertia_tensor(mesh, np.flatnonzero(owner == a), cell_areas)
    inertia_b = _plate_inertia_tensor(mesh, np.flatnonzero(owner == b), cell_areas)
    inertia = inertia_a + inertia_b
    angular_momentum = inertia_a @ omega[a] + inertia_b @ omega[b]
    try:
        return np.linalg.solve(inertia, angular_momentum)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(inertia, rcond=1.0e-12) @ angular_momentum


def _component_area_km2(component: list[int] | Array, cell_areas_km2: Array) -> float:
    cells=np.asarray(component,dtype=np.int32)
    return 0.0 if len(cells)==0 else float(np.sum(np.asarray(cell_areas_km2,dtype=float)[cells]))


def _component_span_km(mesh: SphereMesh, component: list[int] | Array, radius_km: float) -> float:
    cells=np.asarray(component,dtype=np.int32)
    if len(cells)<2:return 0.0
    pts=np.asarray(mesh.centroids[cells],dtype=float)
    i=int(np.argmin(pts@pts[0])); j=int(np.argmin(pts@pts[i]))
    return float(np.arccos(np.clip(np.dot(pts[i],pts[j]),-1.0,1.0)))*float(radius_km)


def _large_component(component:list[int],cell_areas_km2:Array,*,min_cells:int,min_area_km2:float|None)->bool:
    return _component_area_km2(component,cell_areas_km2)>=float(min_area_km2) if min_area_km2 is not None else len(component)>=int(min_cells)


def _continental_fraction_field(state:LithosphereState,mesh:SphereMesh,radius_km:float)->Array:
    frac,_=continental_material_fields(state,mesh.physical_cell_areas_km2(radius_km))
    return np.asarray(frac,dtype=float)


def _make_system_from_groups(
    mesh: SphereMesh,
    old_system: PlateSystem,
    new_owner_raw: Array,
    group_omega: dict[int, Array],
) -> PlateSystem:
    """Compact arbitrary raw owner labels and build a matching PlateSystem."""
    raw = np.asarray(new_owner_raw, dtype=np.int64)
    labels = sorted(int(x) for x in np.unique(raw))
    remap = {lab: i for i, lab in enumerate(labels)}
    owner = np.fromiter((remap[int(x)] for x in raw), dtype=np.int32, count=len(raw))

    plates: list[Plate] = []
    for lab in labels:
        new_id = remap[lab]
        cells = np.flatnonzero(raw == lab)
        if len(cells) == 0:
            raise RuntimeError("Empty plate group during topology compaction")
        w = np.asarray(group_omega[lab], dtype=float)
        speed = float(np.linalg.norm(w))
        if speed > 1e-15:
            axis = w / speed
        else:
            # Deterministic fallback: centroid direction, then z-axis.
            c = np.sum(mesh.centroids[cells], axis=0)
            cn = float(np.linalg.norm(c))
            axis = c / cn if cn > 1e-14 else np.array([0.0, 0.0, 1.0])
        plates.append(
            Plate(
                plate_id=new_id,
                seed_cell=int(cells[0]),
                euler_axis=np.asarray(axis, dtype=float),
                angular_speed_rad_per_myr=speed,
            )
        )
    return PlateSystem(cell_plate=owner, plates=tuple(plates))


def _assign_cut_band_to_components(
    mesh: SphereMesh,
    plate_cells: Array,
    cut_cells: set[int],
    all_components: list[list[int]],
    seed_components: list[list[int]],
) -> Array:
    """Partition a rifted parent into two connected child domains.

    The two large seed components define the children.  Any additional small
    non-rift components are first attached to the nearer seed by spherical
    centroid, then the removed rift band is filled by multi-source BFS.  This
    prevents numerical island fragments from becoming accidental new plates.
    """
    parent = set(int(x) for x in plate_cells)
    if len(seed_components) != 2:
        raise ValueError("Binary split requires exactly two seed components")
    centers=[]
    for comp in seed_components:
        c=np.sum(mesh.centroids[np.asarray(comp,dtype=np.int32)],axis=0); c/=max(float(np.linalg.norm(c)),1e-30); centers.append(c)
    label: dict[int,int]={}
    seed_sets=[set(map(int,c)) for c in seed_components]
    for comp in all_components:
        cset=set(map(int,comp))
        if cset==seed_sets[0]: lab=0
        elif cset==seed_sets[1]: lab=1
        else:
            c=np.sum(mesh.centroids[np.asarray(comp,dtype=np.int32)],axis=0); c/=max(float(np.linalg.norm(c)),1e-30)
            lab=int(np.argmax([float(np.dot(c,centers[0])),float(np.dot(c,centers[1]))]))
        for cell in comp: label[int(cell)]=lab
    q=deque(sorted(label))
    while q:
        cell=q.popleft();lab=label[cell]
        for nb in mesh.neighbors[cell]:
            if nb not in parent or nb in label: continue
            label[int(nb)]=lab;q.append(int(nb))
    if len(label)!=len(parent):
        missing=parent.difference(label)
        # Extremely defensive fallback: nearest seed centroid. This should only
        # be reachable for graph/pathological input, never normal closed mesh.
        for cell in missing:
            lab=int(np.argmax([float(np.dot(mesh.centroids[cell],centers[0])),float(np.dot(mesh.centroids[cell],centers[1]))]));label[cell]=lab
    out=np.full(mesh.cell_count,-1,dtype=np.int32)
    for cell,lab in label.items():out[cell]=int(lab)
    return out


def _attempt_split(
    mesh: SphereMesh,
    state: LithosphereState,
    system: PlateSystem,
    params: PlateTopologyParameters,
    radius_km: float,
) -> tuple[PlateSystem | None, TopologyEvent | None]:
    if not params.split_enabled or len(system.plates) < 1:
        return None, None

    omega = angular_velocity_vectors(system)
    cell_areas=mesh.physical_cell_areas_km2(radius_km)
    cont_frac,cont_volume=continental_material_fields(state,cell_areas)
    cont_thickness=effective_continental_thickness_km(cont_frac,cont_volume,cell_areas)
    cont_material=cont_frac>=float(params.collision_min_continental_fraction)
    # v0.9.2 marks genuine continental breakup with a preserved extension
    # memory.  Ordinary newborn seafloor at a pre-existing plate boundary has
    # rift_extension=0 and therefore cannot accidentally split a plate.
    if state.rift_extension is not None:
        postrift = np.asarray(state.rift_extension, dtype=np.float64) >= float(params.split_min_postrift_extension)
        age_mem = np.zeros_like(state.crust_thickness_km, dtype=np.float64) if state.extension_age_myr is None else np.asarray(state.extension_age_myr, dtype=np.float64)
        young_ocean_rift = (
            (state.crust_type == int(CrustType.OCEANIC))
            & (state.crust_age_myr <= float(params.split_young_ocean_age_myr) + 1e-9)
        )
        mature_thinned_rift = (
            cont_material
            & (cont_thickness <= float(params.split_max_thinned_continent_km))
            & (age_mem >= float(params.split_min_extension_age_myr))
        )
        late_stress = np.zeros_like(state.crust_thickness_km, dtype=np.float64) if state.intraplate_stress is None else np.asarray(state.intraplate_stress, dtype=np.float64)
        late_seam = np.zeros_like(state.crust_thickness_km, dtype=np.float64) if state.collision_seam_weakness is None else np.asarray(state.collision_seam_weakness, dtype=np.float64)
        mature_internal_boundary = (
            (age_mem >= float(params.split_min_extension_age_myr))
            & ((late_stress >= float(params.split_late_stress_threshold)) | (late_seam >= float(params.split_late_seam_threshold)))
        )
        candidate_global = postrift & (young_ocean_rift | mature_thinned_rift | mature_internal_boundary)
    else:
        # Backwards-compatible fallback for older synthetic tests/checkpoints.
        candidate_global = (
            (state.crust_type == int(CrustType.OCEANIC))
            & (state.crust_age_myr <= float(params.split_young_ocean_age_myr) + 1e-9)
            & (np.asarray(state.tidal_damage, dtype=np.float64) >= float(params.split_min_postrift_damage))
        )

    # Prefer larger, genuinely damaged plates first.
    counts = _plate_cell_counts(state.cell_plate, len(system.plates))
    plate_order = sorted(range(len(system.plates)), key=lambda p: (-int(counts[p]), p))

    for parent in plate_order:
        plate_cells = np.flatnonzero(state.cell_plate == parent)
        parent_area=float(np.sum(cell_areas[plate_cells]))
        if params.split_min_child_area_km2 is not None:
            if parent_area < 2.0*float(params.split_min_child_area_km2): continue
        elif len(plate_cells) < 2 * int(params.split_min_child_cells):
            continue
        all_cut=[int(x) for x in plate_cells if candidate_global[int(x)]]
        if not all_cut:
            continue
        cut_components=connected_components(all_cut,mesh.neighbors)
        candidates=[]
        for cc in cut_components:
            span=_component_span_km(mesh,cc,radius_km); area=_component_area_km2(cc,cell_areas)
            if params.split_min_rift_span_km is not None:
                if span<float(params.split_min_rift_span_km): continue
            elif len(cc)<int(params.split_min_rift_cells): continue
            candidates.append((span,area,cc))
        candidates.sort(key=lambda x:(-x[0],-x[1],min(x[2])))
        chosen=None
        for span,rift_area,cc in candidates:
            cut=set(map(int,cc)); remaining=[int(x) for x in plate_cells if int(x) not in cut]
            components=connected_components(remaining,mesh.neighbors)
            large=[c for c in components if _large_component(c,cell_areas,min_cells=params.split_min_child_cells,min_area_km2=params.split_min_child_area_km2)]
            if len(large)<2: continue
            large.sort(key=lambda c:(-_component_area_km2(c,cell_areas),-len(c),min(c)))
            chosen=(cut,components,large[:2],span,rift_area); break
        if chosen is None: continue
        cut,components,seeds,rift_span_km,rift_area_km2=chosen
        child_label = _assign_cut_band_to_components(mesh, plate_cells, cut, components, seeds)

        raw_owner = np.asarray(state.cell_plate, dtype=np.int64).copy()
        next_raw = int(np.max(raw_owner)) + 1
        raw_owner[(state.cell_plate == parent) & (child_label == 0)] = parent
        raw_owner[(state.cell_plate == parent) & (child_label == 1)] = next_raw

        # Unchanged plates inherit their omega.  Children get a small differential
        # rotation around the axis through their two area-centroid directions.
        group_omega: dict[int, Array] = {p: omega[p].copy() for p in range(len(system.plates)) if p != parent}
        c0 = np.sum(mesh.centroids[np.flatnonzero(raw_owner == parent)], axis=0)
        c1 = np.sum(mesh.centroids[np.flatnonzero(raw_owner == next_raw)], axis=0)
        c0 /= max(float(np.linalg.norm(c0)), 1e-30)
        c1 /= max(float(np.linalg.norm(c1)), 1e-30)
        axis = np.cross(c0, c1)
        an = float(np.linalg.norm(axis))
        if an < 1e-12:
            axis = np.asarray(system.plates[parent].euler_axis, dtype=float)
        else:
            axis /= an
        delta = np.deg2rad(float(params.split_differential_speed_deg_per_myr)) * axis
        group_omega[parent] = omega[parent] - 0.5 * delta
        group_omega[next_raw] = omega[parent] + 0.5 * delta

        trial = _make_system_from_groups(mesh, system, raw_owner, group_omega)
        # Ensure the freshly created child contact is, on average, not closing.
        bounds = classify_boundaries(mesh, trial, radius_km, 0.0, 0.0)
        child_ids = sorted(set(int(trial.cell_plate[x]) for x in np.flatnonzero(state.cell_plate == parent)))
        if len(child_ids) == 2:
            rates = [b.normal_rate_km_per_myr for b in bounds if {b.plate_a, b.plate_b} == set(child_ids)]
            if rates and float(np.mean(rates)) < 0.0:
                group_omega[parent] = omega[parent] + 0.5 * delta
                group_omega[next_raw] = omega[parent] - 0.5 * delta
                trial = _make_system_from_groups(mesh, system, raw_owner, group_omega)

        # Parent/child ids after compaction are inferred from ownership.
        children = tuple(sorted(set(int(trial.cell_plate[x]) for x in plate_cells)))
        event = TopologyEvent(
            time_myr=float(state.time_myr),
            kind="split",
            parents=(int(parent),),
            children=children,
            affected_cells=int(len(plate_cells)),
            detail=f"rift_cells={len(cut)}; rift_span={rift_span_km:.0f} km; rift_area={rift_area_km2:.0f} km2; child_areas_km2="+",".join(f"{float(np.sum(cell_areas[trial.cell_plate==c])):.0f}" for c in children),
        )
        return trial, event
    return None, None


def _merge_pair(
    mesh: SphereMesh,
    state: LithosphereState,
    system: PlateSystem,
    a: int,
    b: int,
    radius_km: float,
    kind: str,
    detail: str,
    *,
    velocity_rule: str = "area_weighted",
) -> tuple[PlateSystem, TopologyEvent]:
    if a == b:
        raise ValueError("Cannot merge a plate with itself")
    a, b = sorted((int(a), int(b)))
    owner = np.asarray(state.cell_plate, dtype=np.int64).copy()
    cells_a = np.flatnonzero(owner == a)
    cells_b = np.flatnonzero(owner == b)
    if not len(cells_a) or not len(cells_b):
        raise ValueError("Cannot merge an empty plate")

    omega = angular_velocity_vectors(system)
    merged_w = _merged_angular_velocity(
        mesh, owner, omega, a, b, radius_km, velocity_rule
    )
    owner[owner == b] = a
    group_omega: dict[int, Array] = {}
    for p in range(len(system.plates)):
        if p == b:
            continue
        group_omega[p] = merged_w.copy() if p == a else omega[p].copy()
    out = _make_system_from_groups(mesh, system, owner, group_omega)
    merged_new = int(out.cell_plate[cells_a[0]])
    event = TopologyEvent(
        time_myr=float(state.time_myr),
        kind=kind,
        parents=(a, b),
        children=(merged_new,),
        affected_cells=int(len(cells_a) + len(cells_b)),
        detail=f"{detail}; merge_kinematics_rule={velocity_rule}",
    )
    return out, event


def _attempt_disconnected_split(
    mesh: SphereMesh,
    state: LithosphereState,
    system: PlateSystem,
    params: PlateTopologyParameters,
    radius_km: float,
) -> tuple[PlateSystem | None, TopologyEvent | None]:
    """Split one plate whose surface domain already has two large components.

    A rigid plate cannot consist of two macroscopic surface patches separated by
    another plate.  Semi-Lagrangian advection, subduction and rifting can remove
    the connecting neck before the explicit rift-band detector fires.  This is
    therefore treated as a topology invariant repair, not a random plate birth.
    Only the second large component is detached per event; tiny disconnected
    crumbs remain with the parent and can later be absorbed/cleaned up.
    """
    omega = angular_velocity_vectors(system)
    counts = _plate_cell_counts(system.cell_plate, len(system.plates))
    cell_areas=mesh.physical_cell_areas_km2(radius_km)
    order = sorted(range(len(system.plates)), key=lambda p: (-int(counts[p]), p))
    for parent in order:
        cells = np.flatnonzero(system.cell_plate == parent)
        comps = connected_components(cells, mesh.neighbors)
        large=[c for c in comps if _large_component(c,cell_areas,min_cells=params.disconnect_min_child_cells,min_area_km2=params.disconnect_min_child_area_km2)]
        if len(large) < 2:
            continue
        large.sort(key=lambda c: (-_component_area_km2(c,cell_areas),-len(c),min(c)))
        detached = np.asarray(large[1], dtype=np.int32)
        raw_owner = np.asarray(system.cell_plate, dtype=np.int64).copy()
        next_raw = int(np.max(raw_owner)) + 1
        raw_owner[detached] = next_raw

        group_omega: dict[int, Array] = {p: omega[p].copy() for p in range(len(system.plates)) if p != parent}
        # Children inherit the parent's motion initially.  A very small
        # differential rotation prevents exact dynamical degeneracy; subsequent
        # force balance can then evolve each independently.
        c0 = np.sum(mesh.centroids[np.flatnonzero(raw_owner == parent)], axis=0)
        c1 = np.sum(mesh.centroids[detached], axis=0)
        c0 /= max(float(np.linalg.norm(c0)), 1e-30)
        c1 /= max(float(np.linalg.norm(c1)), 1e-30)
        axis = np.cross(c0, c1)
        an = float(np.linalg.norm(axis))
        if an < 1e-12:
            axis = np.asarray(system.plates[parent].euler_axis, dtype=float)
        else:
            axis /= an
        delta = 0.25 * np.deg2rad(float(params.split_differential_speed_deg_per_myr)) * axis
        group_omega[parent] = omega[parent] - 0.5 * delta
        group_omega[next_raw] = omega[parent] + 0.5 * delta
        trial = _make_system_from_groups(mesh, system, raw_owner, group_omega)
        children = tuple(sorted(set(int(trial.cell_plate[x]) for x in cells)))
        event = TopologyEvent(
            time_myr=float(state.time_myr),
            kind="disconnect_split",
            parents=(int(parent),),
            children=children,
            affected_cells=int(len(detached)),
            detail=f"macroscopic disconnected component detached; child_cells={len(detached)}; child_area_km2={float(np.sum(cell_areas[detached])):.0f}",
        )
        return trial, event
    return None, None


def _compact_empty_plates(
    mesh: SphereMesh,
    state: LithosphereState,
    system: PlateSystem,
) -> tuple[PlateSystem, TopologyEvent | None]:
    """Remove plate IDs that no longer own any surface cell.

    Conservative remapping/subduction can erase the final cell of a very small
    plate between topology updates.  Such a plate has physically vanished and
    must be compacted explicitly.  This is an invariant repair, not a microplate
    threshold decision, and therefore has no persistence delay.
    """
    owner=np.asarray(system.cell_plate,dtype=np.int64)
    present=sorted(int(x) for x in np.unique(owner))
    expected=list(range(len(system.plates)))
    if present == expected:
        return system, None
    missing=tuple(p for p in expected if p not in set(present))
    if not present:
        raise RuntimeError("All plate IDs vanished from the surface")
    omega=angular_velocity_vectors(system)
    group_omega={p:omega[p].copy() for p in present}
    out=_make_system_from_groups(mesh,system,owner,group_omega)
    event=TopologyEvent(
        time_myr=float(state.time_myr),
        kind="vanish",
        parents=missing,
        children=(),
        affected_cells=0,
        detail=f"zero-area plate IDs removed after transport: {missing}",
    )
    return out,event


def _smallest_doomed_plate(mesh:SphereMesh,system:PlateSystem,params:PlateTopologyParameters,radius_km:float)->int|None:
    counts = _plate_cell_counts(system.cell_plate, len(system.plates))
    if params.min_plate_area_km2 is not None:
        areas=_plate_area_weights(mesh,system.cell_plate,radius_km,len(system.plates)); doomed=np.flatnonzero(areas<float(params.min_plate_area_km2))
        return None if not len(doomed) else int(doomed[np.argmin(areas[doomed])])
    doomed = np.flatnonzero(counts < int(params.min_plate_cells))
    if not len(doomed):
        return None
    return int(doomed[np.argmin(counts[doomed])])


def _longest_boundary_neighbor(mesh: SphereMesh, system: PlateSystem, plate: int, radius_km: float) -> int | None:
    length: dict[int, float] = defaultdict(float)
    for fa, fb, u, v in mesh.shared_edges:
        pa = int(system.cell_plate[fa]); pb = int(system.cell_plate[fb])
        if pa == pb or (pa != plate and pb != plate):
            continue
        other = pb if pa == plate else pa
        angle = float(np.arccos(np.clip(np.dot(mesh.vertices[u], mesh.vertices[v]), -1.0, 1.0)))
        length[other] += angle * float(radius_km)
    if not length:
        return None
    return max(length, key=lambda p: (length[p], -p))


@dataclass
class PlateTopologyManager:
    params: PlateTopologyParameters
    collision_age_myr: dict[tuple[int, int], float] = field(default_factory=dict)
    quiet_weld_age_myr: dict[tuple[int, int], float] = field(default_factory=dict)
    small_plate_age_myr: dict[int, float] = field(default_factory=dict)
    last_split_time_myr: float = -1e30

    def _update_small_plate_memory(
        self,
        mesh: SphereMesh,
        system: PlateSystem,
        radius_km: float,
        dt_myr: float,
    ) -> None:
        """Track how long each plate has continuously remained a microplate.

        The area threshold is physical; the persistence clock adds temporal
        hysteresis so a coarse cell entering/leaving the plate does not cause
        an instantaneous cleanup event.  Plates above the threshold forget
        their previous small-plate age immediately.
        """
        threshold = self.params.min_plate_area_km2
        if threshold is None:
            counts = _plate_cell_counts(system.cell_plate, len(system.plates))
            small = counts < int(self.params.min_plate_cells)
        else:
            areas = _plate_area_weights(mesh, system.cell_plate, radius_km, len(system.plates))
            small = areas < float(threshold)
        next_age: dict[int, float] = {}
        for pid in np.flatnonzero(small):
            pid = int(pid)
            next_age[pid] = float(self.small_plate_age_myr.get(pid, 0.0)) + float(dt_myr)
        self.small_plate_age_myr = next_age

    def _smallest_persistent_doomed_plate(
        self,
        mesh: SphereMesh,
        system: PlateSystem,
        radius_km: float,
    ) -> int | None:
        persistence = max(float(self.params.min_plate_persistence_myr), 0.0)
        counts = _plate_cell_counts(system.cell_plate, len(system.plates))
        if self.params.min_plate_area_km2 is not None:
            metric = _plate_area_weights(mesh, system.cell_plate, radius_km, len(system.plates))
            eligible = [
                int(pid) for pid in np.flatnonzero(metric < float(self.params.min_plate_area_km2))
                if float(self.small_plate_age_myr.get(int(pid), 0.0)) + 1e-12 >= persistence
            ]
        else:
            metric = counts.astype(np.float64)
            eligible = [
                int(pid) for pid in np.flatnonzero(counts < int(self.params.min_plate_cells))
                if float(self.small_plate_age_myr.get(int(pid), 0.0)) + 1e-12 >= persistence
            ]
        if not eligible:
            return None
        return min(eligible, key=lambda pid: (float(metric[pid]), pid))

    def _update_collision_memory(
        self,
        mesh: SphereMesh,
        state: LithosphereState,
        boundaries: list[BoundaryRecord],
        radius_km: float,
        dt_myr: float,
    ) -> dict[tuple[int, int], tuple[float, float, float]]:
        """Advance progressive continental-collision memory.

        Returns ``pair -> (continental boundary length, mean relative speed,
        mean normal rate)`` for contacts that remain mechanically contiguous.

        The important v0.9.5 distinction is that collision maturity and final
        welding are separate clocks.  Mature colliding blocks remain distinct
        plates.  Only after their relative motion becomes genuinely quiet does
        the weld clock begin.
        """
        length = defaultdict(float)
        speed_sum = defaultdict(float)
        normal_sum = defaultdict(float)
        convergent_length = defaultdict(float)
        cont_frac=_continental_fraction_field(state,mesh,radius_km)
        min_frac=float(self.params.collision_min_continental_fraction)
        for b in boundaries:
            fa=float(cont_frac[b.face_a]); fb=float(cont_frac[b.face_b])
            if fa<min_frac or fb<min_frac:
                continue
            pair = tuple(sorted((int(b.plate_a), int(b.plate_b))))
            l = _edge_length_km(mesh, b, radius_km)*min(fa,fb)
            length[pair] += l
            speed_sum[pair] += l * float(b.relative_speed_km_per_myr)
            normal_sum[pair] += l * float(b.normal_rate_km_per_myr)
            if b.boundary_type == BoundaryType.CONVERGENT:
                convergent_length[pair] += l

        active: dict[tuple[int, int], tuple[float, float, float]] = {}
        new_collision: dict[tuple[int, int], float] = {}
        new_quiet: dict[tuple[int, int], float] = {}
        for pair, l in length.items():
            if l < float(self.params.merge_min_continental_boundary_km):
                continue
            mean_speed = speed_sum[pair] / max(l, 1e-30)
            mean_normal = normal_sum[pair] / max(l, 1e-30)
            old_collision = float(self.collision_age_myr.get(pair, 0.0))

            # A new collision must actually contain convergence.  Once mature,
            # the contact is allowed to evolve into a near-stationary weld zone
            # without resetting simply because it is no longer classified as
            # convergent by the boundary threshold.
            initiating = (
                convergent_length[pair] > 0.0
                and mean_speed <= float(self.params.collision_initial_max_relative_speed_km_per_myr)
            )
            maintaining = (
                old_collision > 0.0
                and mean_normal <= float(self.params.collision_contact_max_divergence_km_per_myr)
            )
            if not (initiating or maintaining):
                continue

            age = old_collision + float(dt_myr)
            new_collision[pair] = age
            active[pair] = (l, mean_speed, mean_normal)

            quiet = (
                age >= float(self.params.weld_min_collision_age_myr)
                and mean_speed <= float(self.params.weld_max_relative_speed_km_per_myr)
                and mean_normal <= float(self.params.weld_max_normal_divergence_km_per_myr)
            )
            if quiet:
                new_quiet[pair] = float(self.quiet_weld_age_myr.get(pair, 0.0)) + float(dt_myr)

        self.collision_age_myr = new_collision
        self.quiet_weld_age_myr = new_quiet
        return active

    def extension_suppression_field(
        self,
        mesh: SphereMesh,
        state: LithosphereState,
        boundaries: list[BoundaryRecord],
    ) -> Array:
        """Return local 0..1 suppression of continental extension.

        A mature continent-continent collision is a compressional/orogenic
        environment.  Numerical opening gaps adjacent to that same contact
        must not simultaneously drive full-rate continental rifting.  The
        protection is deliberately local: direct collision cells receive the
        full maturity-scaled suppression and one same-plate neighbour ring a
        weaker fraction.  Outside known collision pairs the field is zero.
        """
        out = np.zeros(mesh.cell_count, dtype=np.float64)
        start = float(self.params.collision_rift_suppression_start_myr)
        maximum = float(self.params.collision_rift_suppression_max)
        ring = float(self.params.collision_rift_suppression_one_ring_fraction)
        if maximum <= 0.0 or not self.collision_age_myr:
            return out
        cont_frac=(np.asarray(state.crust_type)==int(CrustType.CONTINENTAL)).astype(float) if state.continental_fraction is None else np.asarray(state.continental_fraction,dtype=float)
        min_frac=float(self.params.collision_min_continental_fraction)
        direct: set[int] = set()
        for b in boundaries:
            pair = tuple(sorted((int(b.plate_a), int(b.plate_b))))
            age = float(self.collision_age_myr.get(pair, 0.0))
            if age < start:
                continue
            if float(cont_frac[b.face_a])<min_frac or float(cont_frac[b.face_b])<min_frac:
                continue
            maturity = min(1.0, (age - start) / max(float(self.params.weld_min_collision_age_myr) - start, 1e-9))
            strength = maximum * (0.35 + 0.65 * maturity)
            for cell in (int(b.face_a), int(b.face_b)):
                out[cell] = max(out[cell], strength)
                direct.add(cell)
        for cell in direct:
            for nb in mesh.neighbors[cell]:
                if float(cont_frac[nb])<min_frac:
                    continue
                if int(state.cell_plate[nb]) != int(state.cell_plate[cell]):
                    continue
                out[nb] = max(out[nb], ring * out[cell])
        return np.clip(out, 0.0, 1.0)

    def _apply_collision_coupling(
        self,
        mesh: SphereMesh,
        system: PlateSystem,
        active: dict[tuple[int, int], tuple[float, float, float]],
        radius_km: float,
        dt_myr: float,
    ) -> PlateSystem:
        """Mechanically couple mature collision partners without merging IDs.

        Each pair relaxes a small fraction toward its area-weighted common
        angular velocity.  Pair angular momentum in this effective kinematic
        sense is preserved, while relative motion decays gradually over tens to
        hundreds of Myr instead of disappearing in a single topology event.
        """
        if not active or len(system.plates) < 2:
            return system
        omega = angular_velocity_vectors(system).copy()
        areas = _plate_area_weights(mesh, system.cell_plate, radius_km, len(system.plates))
        changed = False
        for pair in sorted(active):
            age = float(self.collision_age_myr.get(pair, 0.0))
            if age < float(self.params.collision_coupling_start_myr):
                continue
            a, b = pair
            if a >= len(system.plates) or b >= len(system.plates):
                continue
            maturity = min(1.0, age / max(float(self.params.weld_min_collision_age_myr), 1e-9))
            alpha = maturity * float(dt_myr) / max(float(self.params.collision_coupling_timescale_myr), 1e-9)
            alpha = float(np.clip(alpha, 0.0, float(self.params.collision_coupling_max_step_fraction)))
            if alpha <= 0.0:
                continue
            common = (areas[a] * omega[a] + areas[b] * omega[b]) / max(float(areas[a] + areas[b]), 1e-30)
            omega[a] += alpha * (common - omega[a])
            omega[b] += alpha * (common - omega[b])
            changed = True
        if not changed:
            return system
        # Mechanical coupling changes Euler velocities only.  It must never
        # compact IDs or otherwise alter topology; empty-ID cleanup is handled
        # explicitly at the start of update() as a `vanish` event.
        return system_from_omega(system.cell_plate, system, omega)

    def _remap_collision_memory(self, old_system: PlateSystem, new_system: PlateSystem) -> None:
        """Carry collision/weld clocks across plate-ID compaction.

        Topology events renumber plate IDs globally.  v0.9.4 used to clear all
        collision memory after every event, which made long-lived weld zones
        impossible whenever unrelated rifting happened elsewhere.  We map each
        old plate to the new plate containing the largest number of its material
        cells and preserve clocks for pairs that remain distinct.
        """
        old_owner = np.asarray(old_system.cell_plate, dtype=np.int32)
        new_owner = np.asarray(new_system.cell_plate, dtype=np.int32)
        mapping: dict[int, int] = {}
        for old_id in range(len(old_system.plates)):
            cells = np.flatnonzero(old_owner == old_id)
            if not len(cells):
                continue
            vals, counts = np.unique(new_owner[cells], return_counts=True)
            mapping[old_id] = int(vals[int(np.argmax(counts))])

        def remap(src: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
            out: dict[tuple[int, int], float] = {}
            for (a, b), age in src.items():
                if a not in mapping or b not in mapping:
                    continue
                na, nb = mapping[a], mapping[b]
                if na == nb:
                    continue
                pair = tuple(sorted((na, nb)))
                out[pair] = max(float(age), float(out.get(pair, 0.0)))
            return out

        old_small = dict(self.small_plate_age_myr)
        new_small: dict[int, float] = {}
        for old_id, age in old_small.items():
            if old_id not in mapping:
                continue
            new_id = mapping[old_id]
            new_small[new_id] = max(float(age), float(new_small.get(new_id, 0.0)))
        self.small_plate_age_myr = new_small
        self.collision_age_myr = remap(self.collision_age_myr)
        self.quiet_weld_age_myr = remap(self.quiet_weld_age_myr)

    def update(
        self,
        mesh: SphereMesh,
        state: LithosphereState,
        system: PlateSystem,
        boundaries: list[BoundaryRecord],
        radius_km: float,
        dt_myr: float,
    ) -> tuple[PlateSystem, TopologyDiagnostics, list[TopologyEvent]]:
        before = len(system.plates)
        events: list[TopologyEvent] = []
        split_n = merge_n = absorb_n = 0
        current = system

        # Topology invariant zero: transport may have removed the final surface
        # cell of a plate.  Compact that dead ID synchronously before any
        # collision coupling or split/merge logic touches plate-indexed arrays.
        compacted, vev = _compact_empty_plates(mesh, state, current)
        if vev is not None:
            old_current=current
            current=compacted
            state.cell_plate=current.cell_plate.copy()
            events.append(vev)
            self._remap_collision_memory(old_current,current)

        self._update_small_plate_memory(mesh, current, radius_km, dt_myr)

        # Topology invariant first: if one plate ID already occupies two large
        # disconnected surface domains, detach one before evaluating new welds.
        disconnected, dev = _attempt_disconnected_split(mesh, state, current, self.params, radius_km)
        if disconnected is not None and dev is not None and len(events) < int(self.params.max_events_per_step):
            old_current = current
            current = disconnected
            state.cell_plate = current.cell_plate.copy()
            events.append(dev); split_n += 1
            self.last_split_time_myr = float(state.time_myr)
            self._remap_collision_memory(old_current, current)
            active = {}
        else:
            active = self._update_collision_memory(mesh, state, boundaries, radius_km, dt_myr)
            # Progressive coupling changes Euler velocities but deliberately
            # keeps collision partners as separate plate IDs.
            current = self._apply_collision_coupling(mesh, current, active, radius_km, dt_myr)

        # Final welding is a rare *second stage*: a mature collision must then
        # spend a long separate interval at genuinely low relative speed.
        if self.params.merge_enabled and not events and len(events) < int(self.params.max_events_per_step):
            mature = [
                (quiet_age, self.collision_age_myr.get(pair, 0.0), pair, active[pair])
                for pair, quiet_age in self.quiet_weld_age_myr.items()
                if quiet_age >= float(self.params.weld_quiet_persistence_myr) and pair in active
            ]
            if mature:
                mature.sort(key=lambda x: (-x[0], -x[1], x[2]))
                quiet_age, collision_age, pair, (length, speed, normal_rate) = mature[0]
                old_current = current
                current, ev = _merge_pair(
                    mesh, state, current, pair[0], pair[1], radius_km,
                    "merge",
                    f"mature collision {collision_age:.1f} Myr + quiet weld phase {quiet_age:.1f} Myr; "
                    f"boundary={length:.0f} km; mean_rel_speed={speed:.1f} km/Myr; mean_normal={normal_rate:.1f} km/Myr",
                    velocity_rule=self.params.merge_kinematics_rule,
                )
                state.cell_plate = current.cell_plate.copy()
                events.append(ev); merge_n += 1
                self._remap_collision_memory(old_current, current)

        # Breakup after merging, but not repeatedly within a short cooldown.
        if (
            self.params.split_enabled
            and len(events) < int(self.params.max_events_per_step)
            and float(state.time_myr) - float(self.last_split_time_myr) >= float(self.params.split_cooldown_myr)
        ):
            trial, ev = _attempt_split(mesh, state, current, self.params, radius_km)
            if trial is not None and ev is not None:
                old_current = current
                current = trial
                state.cell_plate = current.cell_plate.copy()
                events.append(ev); split_n += 1
                self.last_split_time_myr = float(state.time_myr)
                self._remap_collision_memory(old_current, current)

        # Remove tiny plates/fragments.  One absorption per remaining event slot.
        while len(events) < int(self.params.max_events_per_step) and len(current.plates) > 1:
            doomed = self._smallest_persistent_doomed_plate(mesh,current,radius_km)
            if doomed is None:
                break
            nb = _longest_boundary_neighbor(mesh, current, doomed, radius_km)
            if nb is None:
                break
            old_current = current
            current, ev = _merge_pair(
                mesh, state, current, doomed, nb, radius_km,
                "absorb",
                (f"plate below min_plate_area_km2={self.params.min_plate_area_km2:.0f} for {self.small_plate_age_myr.get(doomed,0.0):.1f} Myr; absorbed into longest-boundary neighbour" if self.params.min_plate_area_km2 is not None else f"plate below min_plate_cells={self.params.min_plate_cells} for {self.small_plate_age_myr.get(doomed,0.0):.1f} Myr; absorbed into longest-boundary neighbour"),
                velocity_rule=self.params.merge_kinematics_rule,
            )
            state.cell_plate = current.cell_plate.copy()
            events.append(ev); absorb_n += 1
            self._remap_collision_memory(old_current, current)

        counts = _plate_cell_counts(current.cell_plate, len(current.plates))
        plate_areas=_plate_area_weights(mesh,current.cell_plate,radius_km,len(current.plates))
        diag = TopologyDiagnostics(
            time_myr=float(state.time_myr),
            plate_count_before=int(before),
            plate_count_after=int(len(current.plates)),
            split_events=int(split_n),
            merge_events=int(merge_n),
            absorbed_small_plates=int(absorb_n),
            active_collision_pairs=int(len(active)),
            mature_collision_pairs=int(sum(age >= float(self.params.weld_min_collision_age_myr) for age in self.collision_age_myr.values())),
            quiet_weld_pairs=int(len(self.quiet_weld_age_myr)),
            max_collision_age_myr=float(max(self.collision_age_myr.values(), default=0.0)),
            max_quiet_weld_age_myr=float(max(self.quiet_weld_age_myr.values(), default=0.0)),
            min_plate_cells=int(np.min(counts)) if len(counts) else 0,
            mean_plate_cells=float(np.mean(counts)) if len(counts) else 0.0,
            max_plate_cells=int(np.max(counts)) if len(counts) else 0,
            min_plate_area_km2=float(np.min(plate_areas)) if len(plate_areas) else 0.0,
            mean_plate_area_km2=float(np.mean(plate_areas)) if len(plate_areas) else 0.0,
            max_plate_area_km2=float(np.max(plate_areas)) if len(plate_areas) else 0.0,
            topology_changed=bool(events),
        )
        return current, diag, events


__all__ = [
    "PlateTopologyParameters",
    "TopologyEvent",
    "TopologyDiagnostics",
    "PlateTopologyManager",
]
