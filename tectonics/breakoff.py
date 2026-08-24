"""v0.20 collision-triggered slab breakoff.

The model is deliberately effective/2.5-D.  A remembered oceanic slab does
not detach the instant buoyant continental lithosphere reaches the trench.
Instead the continental front loads the slab/continent neck in tension and a
bounded damage variable accumulates over a finite geologic interval.

The mature active slab-pull calibration from v0.17 is *not* strengthened here.
Breakoff only removes that pull once damage reaches unity.  Broken slabs remain
as short-lived tombstones so the same oriented plate pair cannot immediately
recreate an identical slab on the next 4-Myr step.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .lithosphere import LithosphereState, continental_material_fields
from .mesh import SphereMesh
from .subduction_memory import SubductionMemoryState

Array = np.ndarray


@dataclass(slots=True)
class SlabBreakoffParameters:
    enabled: bool = True
    continental_fraction_onset: float = 0.35
    continental_fraction_full: float = 0.75
    front_search_radius_km: float = 450.0
    front_weight_scale_km: float = 220.0
    min_slab_length_km: float = 650.0
    full_slab_length_km: float = 1200.0
    min_slab_depth_km: float = 250.0
    full_slab_depth_km: float = 650.0
    weak_slab_breakoff_time_myr: float = 10.0
    strong_slab_breakoff_time_myr: float = 22.0
    strong_slab_buoyancy_factor: float = 1.15
    damage_relaxation_myr: float = 20.0
    post_breakoff_cooldown_myr: float = 40.0


@dataclass(slots=True)
class SlabBreakoffDiagnostics:
    time_myr: float
    active_necking_zones: int
    broken_off_tombstones: int
    cumulative_breakoffs: int
    mean_breakoff_damage: float
    max_breakoff_damage: float
    mean_collision_front_fraction: float
    max_collision_front_fraction: float
    mean_target_breakoff_time_myr: float
    breakoffs_this_step: int


def _smooth01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _front_continental_fraction(
    mesh: SphereMesh,
    state: LithosphereState,
    zone,
    radius_km: float,
    params: SlabBreakoffParameters,
) -> float:
    """Area/gaussian weighted continental footprint at the incoming plate front."""
    pid = int(zone.subducting_plate)
    cells = np.flatnonzero(np.asarray(state.cell_plate, dtype=np.int32) == pid)
    if not len(cells):
        return 0.0
    areas = mesh.physical_cell_areas_km2(float(radius_km))
    frac, _ = continental_material_fields(state, areas)
    r = np.asarray(zone.trench_midpoint, dtype=np.float64)
    dots = np.clip(mesh.centroids[cells] @ r, -1.0, 1.0)
    dist = np.arccos(dots) * float(radius_km)
    keep = dist <= float(params.front_search_radius_km)
    if not np.any(keep):
        # Nearest incoming-plate cell is still preferable to silently returning
        # zero on very coarse meshes.
        j = int(np.argmin(dist))
        return float(np.clip(frac[cells[j]], 0.0, 1.0))
    cc = cells[keep]
    dd = dist[keep]
    scale = max(float(params.front_weight_scale_km), 1e-9)
    w = areas[cc] * np.exp(-0.5 * (dd / scale) ** 2)
    sw = float(np.sum(w))
    if sw <= 0.0:
        return float(np.mean(frac[cc]))
    return float(np.sum(w * frac[cc]) / sw)


def _target_breakoff_time_myr(zone, params: SlabBreakoffParameters) -> float:
    """Old/strong slabs resist necking longer than weak/young slabs.

    ``buoyancy_factor`` is our existing effective proxy for a cold, mature
    incoming lithosphere.  It is imperfect as rheology, so the interpolation is
    deliberately modest and bounded by literature-scale 10--22 Myr values.
    """
    strength = _smooth01(float(zone.buoyancy_factor) / max(float(params.strong_slab_buoyancy_factor), 1e-9))
    return float(params.weak_slab_breakoff_time_myr) + strength * (
        float(params.strong_slab_breakoff_time_myr) - float(params.weak_slab_breakoff_time_myr)
    )


def slab_pull_multiplier_for_pair(memory: SubductionMemoryState | None, subducting_plate: int, overriding_plate: int) -> float:
    """Suppress the surface slab-pull proxy while a just-broken slab tombstone exists."""
    if memory is None:
        return 1.0
    z = memory.zones.get((int(subducting_plate), int(overriding_plate)))
    if z is not None and bool(getattr(z, "broken_off", False)):
        return 0.0
    return 1.0


def advance_slab_breakoff(
    mesh: SphereMesh,
    state: LithosphereState,
    memory: SubductionMemoryState,
    radius_km: float,
    dt_myr: float,
    params: SlabBreakoffParameters,
) -> tuple[SubductionMemoryState, SlabBreakoffDiagnostics]:
    if dt_myr <= 0.0:
        raise ValueError("dt_myr must be positive")
    if not params.enabled:
        return memory, diagnose_slab_breakoff(memory)

    remove: list[tuple[int, int]] = []
    damage_values: list[float] = []
    front_values: list[float] = []
    target_times: list[float] = []
    active_necking = 0
    breakoffs_now = 0

    for key in sorted(memory.zones):
        z = memory.zones[key]
        if bool(getattr(z, "broken_off", False)):
            z.active = False
            z.rollback_rate_km_per_myr = 0.0
            z.post_breakoff_age_myr = float(getattr(z, "post_breakoff_age_myr", 0.0)) + float(dt_myr)
            if z.post_breakoff_age_myr >= float(params.post_breakoff_cooldown_myr):
                remove.append(key)
            continue

        front = _front_continental_fraction(mesh, state, z, radius_km, params)
        z.last_front_continental_fraction = float(front)
        front_values.append(front)
        length_factor = _smooth01(
            (float(z.slab_length_km) - float(params.min_slab_length_km)) /
            max(float(params.full_slab_length_km - params.min_slab_length_km), 1e-9)
        )
        depth_factor = _smooth01(
            (float(z.slab_depth_km) - float(params.min_slab_depth_km)) /
            max(float(params.full_slab_depth_km - params.min_slab_depth_km), 1e-9)
        )
        continental_factor = _smooth01(
            (front - float(params.continental_fraction_onset)) /
            max(float(params.continental_fraction_full - params.continental_fraction_onset), 1e-9)
        )
        target = _target_breakoff_time_myr(z, params)
        target_times.append(target)

        # Collision-triggered breakoff begins only after the former oceanic
        # subduction contact has stalled/reclassified.  A continent merely
        # approaching an otherwise active trench must not start necking.
        if (not bool(z.active)) and continental_factor > 0.0 and length_factor > 0.0 and depth_factor > 0.0:
            active_necking += 1
            z.continental_collision_age_myr = float(getattr(z, "continental_collision_age_myr", 0.0)) + float(dt_myr)
            rate = continental_factor * length_factor * depth_factor / max(target, 1e-9)
            z.breakoff_damage = min(1.25, float(getattr(z, "breakoff_damage", 0.0)) + float(dt_myr) * rate)
        else:
            z.continental_collision_age_myr = max(0.0, float(getattr(z, "continental_collision_age_myr", 0.0)) - float(dt_myr))
            relax = np.exp(-float(dt_myr) / max(float(params.damage_relaxation_myr), 1e-9))
            z.breakoff_damage = float(getattr(z, "breakoff_damage", 0.0)) * float(relax)

        damage_values.append(float(z.breakoff_damage))
        if z.breakoff_damage >= 1.0:
            z.broken_off = True
            z.active = False
            z.inactive_age_myr = 0.0
            z.post_breakoff_age_myr = 0.0
            z.breakoff_time_myr = float(state.time_myr + dt_myr)
            z.rollback_rate_km_per_myr = 0.0
            memory.breakoffs = int(getattr(memory, "breakoffs", 0)) + 1
            memory.detachments += 1
            breakoffs_now += 1

    for key in remove:
        memory.zones.pop(key, None)

    memory.time_myr = float(state.time_myr + dt_myr)
    diag = SlabBreakoffDiagnostics(
        time_myr=float(memory.time_myr),
        active_necking_zones=int(active_necking),
        broken_off_tombstones=sum(int(bool(getattr(z, "broken_off", False))) for z in memory.zones.values()),
        cumulative_breakoffs=int(getattr(memory, "breakoffs", 0)),
        mean_breakoff_damage=float(np.mean(damage_values)) if damage_values else 0.0,
        max_breakoff_damage=float(np.max(damage_values)) if damage_values else 0.0,
        mean_collision_front_fraction=float(np.mean(front_values)) if front_values else 0.0,
        max_collision_front_fraction=float(np.max(front_values)) if front_values else 0.0,
        mean_target_breakoff_time_myr=float(np.mean(target_times)) if target_times else 0.0,
        breakoffs_this_step=int(breakoffs_now),
    )
    return memory, diag


def diagnose_slab_breakoff(memory: SubductionMemoryState) -> SlabBreakoffDiagnostics:
    zs = [memory.zones[k] for k in sorted(memory.zones)]
    live = [z for z in zs if not bool(getattr(z, "broken_off", False))]
    damage = np.asarray([float(getattr(z, "breakoff_damage", 0.0)) for z in live], dtype=float)
    front = np.asarray([float(getattr(z, "last_front_continental_fraction", 0.0)) for z in live], dtype=float)
    return SlabBreakoffDiagnostics(
        time_myr=float(memory.time_myr),
        active_necking_zones=sum(int(float(getattr(z, "breakoff_damage", 0.0)) > 0.0) for z in live),
        broken_off_tombstones=sum(int(bool(getattr(z, "broken_off", False))) for z in zs),
        cumulative_breakoffs=int(getattr(memory, "breakoffs", 0)),
        mean_breakoff_damage=float(np.mean(damage)) if len(damage) else 0.0,
        max_breakoff_damage=float(np.max(damage)) if len(damage) else 0.0,
        mean_collision_front_fraction=float(np.mean(front)) if len(front) else 0.0,
        max_collision_front_fraction=float(np.max(front)) if len(front) else 0.0,
        mean_target_breakoff_time_myr=0.0,
        breakoffs_this_step=0,
    )


__all__ = [
    "SlabBreakoffParameters", "SlabBreakoffDiagnostics",
    "advance_slab_breakoff", "diagnose_slab_breakoff",
    "slab_pull_multiplier_for_pair",
]
