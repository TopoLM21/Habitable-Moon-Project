"""Persistent 2.5-D subduction/slab memory for v0.18+.

The surface solver remains rigid-plate/effective.  This module remembers the
subsurface slab that corresponds to an oriented plate pair
``subducting -> overriding``.  Active memory does not add a second mature slab
pull on top of the calibrated v0.17 force; it only supplies a small residual
pull for a short time after the surface contact is lost/reclassified.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable
import numpy as np

from .kinematics import BoundaryRecord, BoundaryType
from .lithosphere import CrustType, LithosphereState, mantle_lithosphere_negative_buoyancy_proxy
from .mesh import SphereMesh
from .plates import PlateSystem

Array = np.ndarray


@dataclass(slots=True)
class SubductionMemoryParameters:
    enabled: bool = True
    slab_length_growth_efficiency: float = 0.90
    slab_length_cap_km: float = 1800.0
    slab_depth_cap_km: float = 1100.0
    initial_dip_deg: float = 35.0
    mature_dip_deg: float = 55.0
    dip_maturation_myr: float = 80.0
    residual_pull_gain: float = 0.05
    residual_decay_myr: float = 24.0
    detach_after_inactive_myr: float = 80.0
    # Optional experiment. Production v0.18 leaves this disabled because strong
    # hysteresis systematically changed the topology distribution.
    polarity_hysteresis_enabled: bool = False
    polarity_advantage_ratio: float = 1.10
    polarity_memory_myr: float = 24.0


@dataclass(slots=True)
class SlabZone:
    subducting_plate: int
    overriding_plate: int
    active: bool = True
    active_age_myr: float = 0.0
    inactive_age_myr: float = 0.0
    slab_length_km: float = 0.0
    slab_depth_km: float = 0.0
    dip_deg: float = 35.0
    trench_length_km: float = 0.0
    convergence_rate_km_per_myr: float = 0.0
    buoyancy_factor: float = 1.0
    cumulative_subducted_area_km2: float = 0.0
    trench_midpoint: Array = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float64))
    torque_axis: Array = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float64))
    rollback_distance_km: float = 0.0
    rollback_rate_km_per_myr: float = 0.0
    # v0.20 collision-triggered necking / breakoff state.  These live on the
    # remembered slab so checkpoint/resume and topology remaps preserve the
    # finite-time path to detachment.
    continental_collision_age_myr: float = 0.0
    breakoff_damage: float = 0.0
    last_front_continental_fraction: float = 0.0
    broken_off: bool = False
    post_breakoff_age_myr: float = 0.0
    breakoff_time_myr: float = -1.0

    def key(self) -> tuple[int, int]:
        return int(self.subducting_plate), int(self.overriding_plate)


@dataclass(slots=True)
class SubductionMemoryState:
    time_myr: float = 0.0
    zones: dict[tuple[int, int], SlabZone] = field(default_factory=dict)
    births: int = 0
    detachments: int = 0
    breakoffs: int = 0
    cumulative_subducted_area_km2: float = 0.0


@dataclass(slots=True)
class SubductionMemoryDiagnostics:
    time_myr: float
    zone_count: int
    active_zone_count: int
    residual_zone_count: int
    births: int
    detachments: int
    cumulative_subducted_area_km2: float
    mean_slab_length_km: float
    max_slab_length_km: float
    mean_slab_depth_km: float
    max_slab_depth_km: float
    mean_residual_pull_fraction: float
    max_residual_pull_fraction: float


def initialize_subduction_memory(time_myr: float = 0.0) -> SubductionMemoryState:
    return SubductionMemoryState(time_myr=float(time_myr))


def _edge_length_km(mesh: SphereMesh, b: BoundaryRecord, radius_km: float) -> float:
    u = mesh.vertices[b.vertex_u]
    v = mesh.vertices[b.vertex_v]
    return float(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0)) * radius_km)


def _normal_ab(mesh: SphereMesh, b: BoundaryRecord) -> Array:
    r = np.asarray(b.midpoint, dtype=np.float64)
    d = np.asarray(mesh.centroids[b.face_b] - mesh.centroids[b.face_a], dtype=np.float64)
    d -= r * float(np.dot(d, r))
    n = float(np.linalg.norm(d))
    return np.zeros(3, dtype=np.float64) if n < 1e-14 else d / n


def _local_proxy(state: LithosphereState, face: int) -> float:
    if state.mantle_lithosphere_thickness_km is not None and state.mantle_lithosphere_density_anomaly_kg_m3 is not None:
        from .cpu_runtime import current_execution
        execution = current_execution()
        if execution is not None and execution.numeric_kernels:
            from .lithosphere import mantle_lithosphere_negative_buoyancy_at
            return max(mantle_lithosphere_negative_buoyancy_at(state, face), 0.0)
        p = mantle_lithosphere_negative_buoyancy_proxy(state)
        return max(float(p[face]), 0.0)
    return max(float(state.crust_age_myr[face]), 0.0)


def choose_subducting_side(
    state: LithosphereState,
    b: BoundaryRecord,
    memory: SubductionMemoryState | None = None,
    params: SubductionMemoryParameters | None = None,
) -> int | None:
    ta = int(state.crust_type[b.face_a]); tb = int(state.crust_type[b.face_b])
    if ta == int(CrustType.OCEANIC) and tb == int(CrustType.CONTINENTAL):
        return int(b.plate_a)
    if tb == int(CrustType.OCEANIC) and ta == int(CrustType.CONTINENTAL):
        return int(b.plate_b)
    if ta != int(CrustType.OCEANIC) or tb != int(CrustType.OCEANIC):
        return None
    aa = _local_proxy(state, b.face_a); ab = _local_proxy(state, b.face_b)
    preferred = int(b.plate_a) if aa > ab + 1e-9 else (int(b.plate_b) if ab > aa + 1e-9 else min(int(b.plate_a), int(b.plate_b)))
    if memory is None or params is None or not params.polarity_hysteresis_enabled:
        return preferred
    a, c = int(b.plate_a), int(b.plate_b)
    za = memory.zones.get((a, c)); zb = memory.zones.get((c, a))
    old = za if za is not None and za.inactive_age_myr <= params.polarity_memory_myr else zb if zb is not None and zb.inactive_age_myr <= params.polarity_memory_myr else None
    if old is None:
        return preferred
    old_sub = int(old.subducting_plate)
    old_proxy = aa if old_sub == a else ab
    other_proxy = ab if old_sub == a else aa
    return preferred if other_proxy > max(old_proxy, 1e-12) * float(params.polarity_advantage_ratio) else old_sub


def _buoyancy_factor(state: LithosphereState, face: int, reference: float = 6000.0, exponent: float = 0.75) -> float:
    # 100 km * 60 kg/m3 = 6000 is the v0.16 reference integrated anomaly.
    p = _local_proxy(state, face)
    if state.mantle_lithosphere_thickness_km is None:
        return float(np.clip(1.0 + 0.9 * min(p / 80.0, 1.5), 0.02, 2.8))
    return float(np.clip(p / max(reference, 1e-9), 0.02, 2.8)) ** float(exponent)


def advance_subduction_memory(
    mesh: SphereMesh,
    state: LithosphereState,
    boundaries: Iterable[BoundaryRecord],
    radius_km: float,
    dt_myr: float,
    memory: SubductionMemoryState,
    params: SubductionMemoryParameters,
) -> tuple[SubductionMemoryState, SubductionMemoryDiagnostics]:
    if dt_myr <= 0.0:
        raise ValueError("dt_myr must be positive")
    if not params.enabled:
        memory.time_myr = float(state.time_myr)
        return memory, diagnose_subduction_memory(memory, params)

    # Aggregate current mesh-edge segments by oriented plate pair.
    agg: dict[tuple[int, int], dict[str, object]] = {}
    for b in boundaries:
        if b.boundary_type != BoundaryType.CONVERGENT:
            continue
        sub = choose_subducting_side(state, b, memory, params)
        if sub is None:
            continue
        pa, pb = int(b.plate_a), int(b.plate_b)
        over = pb if sub == pa else pa
        key = (sub, over)
        length = _edge_length_km(mesh, b, radius_km)
        if length <= 0.0:
            continue
        normal = _normal_ab(mesh, b)
        if not np.any(normal):
            continue
        if sub == pa:
            face = int(b.face_a); toward = +normal
        else:
            face = int(b.face_b); toward = -normal
        r = np.asarray(b.midpoint, dtype=np.float64)
        torque = np.cross(r, toward)
        d = agg.setdefault(key, {"length":0.0,"conv":0.0,"mid":np.zeros(3),"torque":np.zeros(3),"buoy":0.0})
        d["length"] = float(d["length"]) + length
        d["conv"] = float(d["conv"]) + length * max(-float(b.normal_rate_km_per_myr), 0.0)
        d["mid"] = np.asarray(d["mid"]) + length * r
        d["torque"] = np.asarray(d["torque"]) + length * torque
        d["buoy"] = float(d["buoy"]) + length * _buoyancy_factor(state, face)

    active_keys = set(agg)
    births_now = 0
    for key, d in agg.items():
        length = max(float(d["length"]), 1e-12)
        conv = float(d["conv"]) / length
        mid = np.asarray(d["mid"], dtype=np.float64); mid /= max(float(np.linalg.norm(mid)), 1e-30)
        tor = np.asarray(d["torque"], dtype=np.float64); tor /= max(float(np.linalg.norm(tor)), 1e-30)
        buoy = float(d["buoy"]) / length
        z = memory.zones.get(key)
        if z is not None and bool(getattr(z, "broken_off", False)):
            # v0.20 breakoff tombstone: the detached slab cannot be recreated
            # by the same surface contact on the immediately following step.
            continue
        if z is None:
            z = SlabZone(subducting_plate=key[0], overriding_plate=key[1], dip_deg=float(params.initial_dip_deg))
            memory.zones[key] = z; memory.births += 1; births_now += 1
        z.active = True; z.inactive_age_myr = 0.0; z.active_age_myr += float(dt_myr)
        maturity = 1.0 - np.exp(-z.active_age_myr / max(float(params.dip_maturation_myr), 1e-9))
        z.dip_deg = float(params.initial_dip_deg + (params.mature_dip_deg - params.initial_dip_deg) * maturity)
        increment = max(conv, 0.0) * float(dt_myr) * float(params.slab_length_growth_efficiency)
        z.slab_length_km = min(float(params.slab_length_cap_km), z.slab_length_km + increment)
        z.slab_depth_km = min(float(params.slab_depth_cap_km), z.slab_length_km * np.sin(np.deg2rad(z.dip_deg)))
        z.trench_length_km = length; z.convergence_rate_km_per_myr = conv; z.buoyancy_factor = buoy
        z.trench_midpoint = mid; z.torque_axis = tor
        area = length * max(conv, 0.0) * float(dt_myr)
        z.cumulative_subducted_area_km2 += area; memory.cumulative_subducted_area_km2 += area

    detach: list[tuple[int,int]] = []
    for key, z in list(memory.zones.items()):
        if bool(getattr(z, "broken_off", False)):
            z.active = False
            continue
        if key in active_keys:
            continue
        z.active = False; z.inactive_age_myr += float(dt_myr)
        if z.inactive_age_myr >= float(params.detach_after_inactive_myr):
            detach.append(key)
    for key in detach:
        memory.zones.pop(key, None); memory.detachments += 1

    memory.time_myr = float(state.time_myr)
    return memory, diagnose_subduction_memory(memory, params)


def residual_pull_by_plate(
    memory: SubductionMemoryState | None,
    plate_count: int,
    params: SubductionMemoryParameters,
    slab_pull_weight: float,
    slab_buoyancy_calibration_gain: float,
) -> tuple[Array, Array]:
    """Return plate-scale residual-drive vectors and per-plate fractions.

    Multiple remembered trench segments are trench-length weighted instead of
    being naively summed, matching the normalization of surface boundary drive.
    """
    vec = np.zeros((int(plate_count), 3), dtype=np.float64)
    weight = np.zeros(int(plate_count), dtype=np.float64)
    frac_sum = np.zeros(int(plate_count), dtype=np.float64)
    if memory is None or not params.enabled or params.residual_pull_gain <= 0.0:
        return vec, frac_sum
    for key in sorted(memory.zones):
        z=memory.zones[key]
        if z.active or bool(getattr(z, "broken_off", False)) or z.inactive_age_myr < 0.0:
            continue
        sub = int(z.subducting_plate)
        if sub < 0 or sub >= int(plate_count):
            continue
        frac = float(params.residual_pull_gain) * np.exp(-z.inactive_age_myr / max(float(params.residual_decay_myr), 1e-9))
        w = max(float(z.trench_length_km), 1e-9)
        strength = float(slab_pull_weight) * float(slab_buoyancy_calibration_gain) * float(z.buoyancy_factor) * frac
        vec[sub] += w * strength * np.asarray(z.torque_axis, dtype=np.float64)
        frac_sum[sub] += w * frac
        weight[sub] += w
    nz = weight > 0.0
    vec[nz] /= weight[nz, None]
    frac_sum[nz] /= weight[nz]
    return vec, frac_sum


def remap_subduction_memory(
    mesh: SphereMesh,
    old_system: PlateSystem,
    new_system: PlateSystem,
    memory: SubductionMemoryState,
) -> SubductionMemoryState:
    """Remap slab plate IDs through split/merge/vanish topology changes.

    The trench midpoint chooses the geographically appropriate child when an
    old plate splits.  This is deterministic and avoids attaching a remembered
    slab to a remote child merely because it has larger total area.
    """
    old_owner = np.asarray(old_system.cell_plate, dtype=np.int32)
    new_owner = np.asarray(new_system.cell_plate, dtype=np.int32)
    out: dict[tuple[int,int], SlabZone] = {}
    for z in memory.zones.values():
        mapped=[]
        for old_pid in (int(z.subducting_plate), int(z.overriding_plate)):
            cells=np.flatnonzero(old_owner==old_pid)
            if len(cells)==0:
                mapped.append(-1); continue
            dots=mesh.centroids[cells] @ np.asarray(z.trench_midpoint,dtype=np.float64)
            cell=int(cells[int(np.argmax(dots))])
            mapped.append(int(new_owner[cell]))
        sub,over=mapped
        if sub < 0 or over < 0 or sub == over:
            memory.detachments += 1
            continue
        nz=SlabZone(**{k:v for k,v in asdict(z).items() if k not in {"trench_midpoint","torque_axis"}})
        nz.subducting_plate=sub; nz.overriding_plate=over
        nz.trench_midpoint=np.asarray(z.trench_midpoint,dtype=np.float64).copy(); nz.torque_axis=np.asarray(z.torque_axis,dtype=np.float64).copy()
        key=(sub,over)
        old=out.get(key)
        if old is None:
            out[key]=nz
        else:
            total=max(old.trench_length_km+nz.trench_length_km,1e-9)
            old.cumulative_subducted_area_km2 += nz.cumulative_subducted_area_km2
            old.slab_length_km=max(old.slab_length_km,nz.slab_length_km); old.slab_depth_km=max(old.slab_depth_km,nz.slab_depth_km)
            old.breakoff_damage=max(float(getattr(old,"breakoff_damage",0.0)),float(getattr(nz,"breakoff_damage",0.0)))
            old.continental_collision_age_myr=max(float(getattr(old,"continental_collision_age_myr",0.0)),float(getattr(nz,"continental_collision_age_myr",0.0)))
            old.last_front_continental_fraction=max(float(getattr(old,"last_front_continental_fraction",0.0)),float(getattr(nz,"last_front_continental_fraction",0.0)))
            old.broken_off=bool(getattr(old,"broken_off",False) or getattr(nz,"broken_off",False))
            old.post_breakoff_age_myr=max(float(getattr(old,"post_breakoff_age_myr",0.0)),float(getattr(nz,"post_breakoff_age_myr",0.0)))
            old.breakoff_time_myr=max(float(getattr(old,"breakoff_time_myr",-1.0)),float(getattr(nz,"breakoff_time_myr",-1.0)))
            old.trench_midpoint=(old.trench_midpoint*old.trench_length_km+nz.trench_midpoint*nz.trench_length_km)/total; old.trench_midpoint/=max(np.linalg.norm(old.trench_midpoint),1e-30)
            old.torque_axis=(old.torque_axis*old.trench_length_km+nz.torque_axis*nz.trench_length_km)/total; old.torque_axis/=max(np.linalg.norm(old.torque_axis),1e-30)
            old.trench_length_km=total
    memory.zones=out
    return memory


def diagnose_subduction_memory(memory: SubductionMemoryState, params: SubductionMemoryParameters) -> SubductionMemoryDiagnostics:
    zs=[memory.zones[k] for k in sorted(memory.zones)]
    lengths=np.array([z.slab_length_km for z in zs],dtype=float) if zs else np.zeros(0)
    depths=np.array([z.slab_depth_km for z in zs],dtype=float) if zs else np.zeros(0)
    residual=np.array([float(params.residual_pull_gain)*np.exp(-z.inactive_age_myr/max(float(params.residual_decay_myr),1e-9)) for z in zs if (not z.active and not bool(getattr(z,"broken_off",False)))],dtype=float)
    return SubductionMemoryDiagnostics(
        time_myr=float(memory.time_myr), zone_count=len(zs), active_zone_count=sum(int(z.active and not bool(getattr(z,"broken_off",False))) for z in zs), residual_zone_count=sum(int((not z.active) and not bool(getattr(z,"broken_off",False))) for z in zs),
        births=int(memory.births), detachments=int(memory.detachments), cumulative_subducted_area_km2=float(memory.cumulative_subducted_area_km2),
        mean_slab_length_km=float(np.mean(lengths)) if len(lengths) else 0.0, max_slab_length_km=float(np.max(lengths)) if len(lengths) else 0.0,
        mean_slab_depth_km=float(np.mean(depths)) if len(depths) else 0.0, max_slab_depth_km=float(np.max(depths)) if len(depths) else 0.0,
        mean_residual_pull_fraction=float(np.mean(residual)) if len(residual) else 0.0, max_residual_pull_fraction=float(np.max(residual)) if len(residual) else 0.0,
    )


def memory_to_json(memory: SubductionMemoryState | None) -> dict | None:
    if memory is None: return None
    return {
        "time_myr":float(memory.time_myr),"births":int(memory.births),"detachments":int(memory.detachments),"breakoffs":int(getattr(memory,"breakoffs",0)),"cumulative_subducted_area_km2":float(memory.cumulative_subducted_area_km2),
        "zones":[{
            "subducting_plate":int(z.subducting_plate),"overriding_plate":int(z.overriding_plate),"active":bool(z.active),"active_age_myr":float(z.active_age_myr),"inactive_age_myr":float(z.inactive_age_myr),
            "slab_length_km":float(z.slab_length_km),"slab_depth_km":float(z.slab_depth_km),"dip_deg":float(z.dip_deg),"trench_length_km":float(z.trench_length_km),"convergence_rate_km_per_myr":float(z.convergence_rate_km_per_myr),"buoyancy_factor":float(z.buoyancy_factor),
            "cumulative_subducted_area_km2":float(z.cumulative_subducted_area_km2),"rollback_distance_km":float(z.rollback_distance_km),"rollback_rate_km_per_myr":float(z.rollback_rate_km_per_myr),
            "continental_collision_age_myr":float(getattr(z,"continental_collision_age_myr",0.0)),"breakoff_damage":float(getattr(z,"breakoff_damage",0.0)),"last_front_continental_fraction":float(getattr(z,"last_front_continental_fraction",0.0)),
            "broken_off":bool(getattr(z,"broken_off",False)),"post_breakoff_age_myr":float(getattr(z,"post_breakoff_age_myr",0.0)),"breakoff_time_myr":float(getattr(z,"breakoff_time_myr",-1.0)),
            "trench_midpoint":[float(x) for x in z.trench_midpoint],"torque_axis":[float(x) for x in z.torque_axis]
        } for _,z in sorted(memory.zones.items())]
    }


def memory_from_json(data: dict | None) -> SubductionMemoryState | None:
    if data is None: return None
    m=SubductionMemoryState(time_myr=float(data.get("time_myr",0.0)),births=int(data.get("births",0)),detachments=int(data.get("detachments",0)),breakoffs=int(data.get("breakoffs",0)),cumulative_subducted_area_km2=float(data.get("cumulative_subducted_area_km2",0.0)))
    for d in data.get("zones",[]):
        z=SlabZone(subducting_plate=int(d["subducting_plate"]),overriding_plate=int(d["overriding_plate"]),active=bool(d.get("active",True)),active_age_myr=float(d.get("active_age_myr",0.0)),inactive_age_myr=float(d.get("inactive_age_myr",0.0)),slab_length_km=float(d.get("slab_length_km",0.0)),slab_depth_km=float(d.get("slab_depth_km",0.0)),dip_deg=float(d.get("dip_deg",35.0)),trench_length_km=float(d.get("trench_length_km",0.0)),convergence_rate_km_per_myr=float(d.get("convergence_rate_km_per_myr",0.0)),buoyancy_factor=float(d.get("buoyancy_factor",1.0)),cumulative_subducted_area_km2=float(d.get("cumulative_subducted_area_km2",0.0)),rollback_distance_km=float(d.get("rollback_distance_km",0.0)),rollback_rate_km_per_myr=float(d.get("rollback_rate_km_per_myr",0.0)),continental_collision_age_myr=float(d.get("continental_collision_age_myr",0.0)),breakoff_damage=float(d.get("breakoff_damage",0.0)),last_front_continental_fraction=float(d.get("last_front_continental_fraction",0.0)),broken_off=bool(d.get("broken_off",False)),post_breakoff_age_myr=float(d.get("post_breakoff_age_myr",0.0)),breakoff_time_myr=float(d.get("breakoff_time_myr",-1.0)),trench_midpoint=np.asarray(d.get("trench_midpoint",[1,0,0]),dtype=np.float64),torque_axis=np.asarray(d.get("torque_axis",[0,0,1]),dtype=np.float64))
        m.zones[z.key()]=z
    return m


__all__=["SubductionMemoryParameters","SlabZone","SubductionMemoryState","SubductionMemoryDiagnostics","initialize_subduction_memory","choose_subducting_side","advance_subduction_memory","residual_pull_by_plate","remap_subduction_memory","diagnose_subduction_memory","memory_to_json","memory_from_json"]
