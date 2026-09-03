"""v0.8 continental-crust life cycle.

This module sits on top of the v0.7 plate/lithosphere/topography machinery and
adds a deliberately effective (not full petrological) long-term continental
cycle:

* ocean-ocean subduction builds felsic island arcs; persistent arcs mature into
  juvenile continental crust;
* ocean-continent subduction thickens the overriding continental arc;
* strongly damaged, rapidly converging continental margins can undergo
  subduction erosion and, if thinned far enough, be recycled to oceanic crust;
* over-thickened continental crust delaminates lower crust back into the mantle.

The state keeps a continuous ``felsic_potential`` field so new continents do
not appear in a single arbitrary time step.  All coefficients are effective
prototype parameters and are exposed in YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree

from .evolution import rotate_points_by_plate
from .kinematics import BoundaryRecord, BoundaryType
from .lithosphere import CrustType, LithosphereState, continental_material_fields
from .mesh import SphereMesh
from .plates import PlateSystem

Array = np.ndarray


@dataclass(slots=True)
class ContinentalCycleState:
    time_myr: float
    felsic_potential: Array
    cumulative_generated_area_km2: float = 0.0
    cumulative_recycled_area_km2: float = 0.0
    cumulative_generated_volume_km3: float = 0.0
    cumulative_recycled_volume_km3: float = 0.0


@dataclass(slots=True)
class ContinentalCycleDiagnostics:
    time_myr: float
    dt_myr: float
    continental_area_fraction: float
    continental_volume_km3: float
    juvenile_arc_area_created_km2: float
    arc_thickening_volume_km3: float
    subduction_erosion_area_km2: float
    subduction_erosion_volume_km3: float
    delaminated_volume_km3: float
    gravitational_collapse_redistributed_volume_km3: float
    mean_felsic_potential_oceanic: float
    max_felsic_potential: float


@dataclass(slots=True, frozen=True)
class ContinentalCycleParameters:
    convergence_reference_km_per_myr: float = 50.0
    arc_maturation_rate_per_myr: float = 0.018
    arc_potential_decay_myr: float = 250.0
    juvenile_threshold: float = 1.0
    juvenile_continental_thickness_km: float = 26.0
    juvenile_seed_age_myr: float = 0.0
    continental_arc_thickening_km_per_myr: float = 0.045
    max_arc_thickening_km_per_step: float = 0.8
    subduction_erosion_km_per_myr: float = 0.018
    subduction_erosion_damage_threshold: float = 0.28
    recycle_below_thickness_km: float = 21.0
    delamination_threshold_km: float = 60.0
    delamination_target_km: float = 54.0
    delamination_rate_per_myr: float = 0.035
    # v0.9.8 reconstruction: conservative lateral gravitational collapse.
    # Exact lost-source coefficients are not recoverable; disabled by default
    # so legacy v0.9.7 behaviour/tests remain unchanged.
    gravitational_collapse_enabled: bool = False
    gravitational_collapse_threshold_km: float = 45.0
    gravitational_collapse_target_km: float = 39.0
    gravitational_collapse_rate_per_myr: float = 0.020
    gravitational_collapse_neighbor_rings: int = 2


def initialize_continental_cycle(mesh: SphereMesh) -> ContinentalCycleState:
    return ContinentalCycleState(
        time_myr=0.0,
        felsic_potential=np.zeros(mesh.cell_count, dtype=np.float64),
    )


def _copy_lithosphere(state: LithosphereState) -> LithosphereState:
    return LithosphereState(
        time_myr=float(state.time_myr),
        cell_plate=np.asarray(state.cell_plate, dtype=np.int32).copy(),
        crust_type=np.asarray(state.crust_type, dtype=np.int8).copy(),
        crust_age_myr=np.asarray(state.crust_age_myr, dtype=np.float64).copy(),
        crust_thickness_km=np.asarray(state.crust_thickness_km, dtype=np.float64).copy(),
        tidal_damage=np.asarray(state.tidal_damage, dtype=np.float64).copy(),
        rift_extension=None if state.rift_extension is None else np.asarray(state.rift_extension, dtype=np.float64).copy(),
        extension_age_myr=None if state.extension_age_myr is None else np.asarray(state.extension_age_myr, dtype=np.float64).copy(),
        collision_seam_weakness=None if state.collision_seam_weakness is None else np.asarray(state.collision_seam_weakness, dtype=np.float64).copy(),
        intraplate_stress=None if state.intraplate_stress is None else np.asarray(state.intraplate_stress, dtype=np.float64).copy(),
        supercontinent_heat=None if state.supercontinent_heat is None else np.asarray(state.supercontinent_heat, dtype=np.float64).copy(),
        continental_fraction=None if state.continental_fraction is None else np.asarray(state.continental_fraction, dtype=np.float64).copy(),
        continental_volume_km3=None if state.continental_volume_km3 is None else np.asarray(state.continental_volume_km3, dtype=np.float64).copy(),
        mantle_lithosphere_thickness_km=None if state.mantle_lithosphere_thickness_km is None else np.asarray(state.mantle_lithosphere_thickness_km, dtype=np.float64).copy(),
        mantle_lithosphere_density_anomaly_kg_m3=None if state.mantle_lithosphere_density_anomaly_kg_m3 is None else np.asarray(state.mantle_lithosphere_density_anomaly_kg_m3, dtype=np.float64).copy(),
        sediment_volume_km3=None if state.sediment_volume_km3 is None else np.asarray(state.sediment_volume_km3, dtype=np.float64).copy(),
        continental_lithosphere_age_myr=None if state.continental_lithosphere_age_myr is None else np.asarray(state.continental_lithosphere_age_myr, dtype=np.float64).copy(),
        mantle_depletion_fraction=None if state.mantle_depletion_fraction is None else np.asarray(state.mantle_depletion_fraction, dtype=np.float64).copy(),
        craton_strength=None if state.craton_strength is None else np.asarray(state.craton_strength, dtype=np.float64).copy(),
    )


def _continental_volume_km3(areas_km2: Array, state: LithosphereState) -> float:
    _, volume = continental_material_fields(state, areas_km2)
    return float(np.sum(volume))



def _advect_felsic_potential(
    mesh: SphereMesh,
    potential: Array,
    previous_lithosphere: LithosphereState,
    current_lithosphere: LithosphereState,
    plate_system: PlateSystem,
    dt_myr: float,
) -> Array:
    """Semi-Lagrangian transport of arc maturation with plate material."""
    tree = cKDTree(mesh.centroids)
    out = np.zeros(mesh.cell_count, dtype=np.float64)
    for plate_id in range(len(plate_system.plates)):
        targets = np.flatnonzero(current_lithosphere.cell_plate == plate_id)
        if not len(targets):
            continue
        pids = np.full(len(targets), plate_id, dtype=np.int32)
        back = rotate_points_by_plate(mesh.centroids[targets], pids, plate_system, -float(dt_myr))
        _, src = tree.query(back, k=1, workers=-1)
        src = np.asarray(src, dtype=np.int32)
        valid = np.asarray(previous_lithosphere.cell_plate, dtype=np.int32)[src] == plate_id
        if np.any(valid):
            out[targets[valid]] = np.asarray(potential, dtype=np.float64)[src[valid]]
    return out

def _gravitational_collapse(
    mesh: SphereMesh,
    state: LithosphereState,
    areas: Array,
    params: ContinentalCycleParameters,
    dt_myr: float,
) -> float:
    """Conservatively move excess continental material within a plate.

    Thickness describes the continental part of a cell, while
    ``continental_fraction`` describes how much of its footprint that material
    occupies.  Collapse therefore has to transfer ``area * fraction *
    thickness`` rather than the full geometric column volume.  The distinction
    matters at mixed coastline cells.
    """
    if not params.gravitational_collapse_enabled:
        return 0.0
    threshold = float(params.gravitational_collapse_threshold_km)
    target = min(float(params.gravitational_collapse_target_km), threshold)
    relaxation = 1.0 - np.exp(-float(params.gravitational_collapse_rate_per_myr) * float(dt_myr))
    if relaxation <= 0.0:
        return 0.0
    material_fraction, _ = continental_material_fields(state, areas)
    cont = state.crust_type == int(CrustType.CONTINENTAL)
    sources = np.flatnonzero(
        cont & (material_fraction > 0.0) & (state.crust_thickness_km > threshold)
    )
    moved_total = 0.0
    # Thickest columns act first; deterministic index tie-break keeps resume stable.
    sources = np.asarray(sorted((int(x) for x in sources), key=lambda x: (-float(state.crust_thickness_km[x]), x)), dtype=np.int32)
    for src in sources:
        h = float(state.crust_thickness_km[src])
        source_fraction = float(material_fraction[src])
        movable = max(h - target, 0.0) * float(areas[src]) * source_fraction * relaxation
        if movable <= 0.0:
            continue
        remaining = movable
        seen = {int(src)}
        frontier = [int(src)]
        candidates: list[int] = []
        for _ in range(max(int(params.gravitational_collapse_neighbor_rings), 0)):
            nxt: list[int] = []
            for cell in frontier:
                for nb in mesh.neighbors[cell]:
                    nb = int(nb)
                    if nb in seen:
                        continue
                    seen.add(nb)
                    if (
                        not cont[nb]
                        or material_fraction[nb] <= 0.0
                        or int(state.cell_plate[nb]) != int(state.cell_plate[src])
                    ):
                        continue
                    nxt.append(nb)
                    candidates.append(nb)
            frontier = nxt
            if not frontier:
                break
        candidates.sort(key=lambda x: (float(state.crust_thickness_km[x]), x))
        actually_moved = 0.0
        for dst in candidates:
            if remaining <= 1e-9:
                break
            destination_fraction = float(material_fraction[dst])
            capacity = (
                max(threshold - float(state.crust_thickness_km[dst]), 0.0)
                * float(areas[dst])
                * destination_fraction
            )
            if capacity <= 0.0:
                continue
            amount = min(capacity, remaining)
            state.crust_thickness_km[dst] += amount / (
                float(areas[dst]) * destination_fraction
            )
            remaining -= amount
            actually_moved += amount
        if actually_moved > 0.0:
            state.crust_thickness_km[src] -= actually_moved / (
                float(areas[src]) * source_fraction
            )
            moved_total += actually_moved
    return float(moved_total)


def advance_continental_cycle(
    mesh: SphereMesh,
    lithosphere: LithosphereState,
    boundaries: list[BoundaryRecord],
    cycle: ContinentalCycleState,
    dt_myr: float,
    radius_km: float,
    params: ContinentalCycleParameters,
    *,
    oceanic_thickness_km: float = 7.0,
    previous_lithosphere: LithosphereState | None = None,
    plate_system: PlateSystem | None = None,
    transport_source_index: Array | None = None,
    volcanic_arc_forcing: Array | None = None,
) -> tuple[LithosphereState, ContinentalCycleState, ContinentalCycleDiagnostics]:
    if dt_myr <= 0.0:
        raise ValueError("dt_myr must be positive")
    if params.convergence_reference_km_per_myr <= 0.0:
        raise ValueError("convergence_reference_km_per_myr must be positive")

    areas = mesh.physical_cell_areas_km2(radius_km)
    state = _copy_lithosphere(lithosphere)
    material_fraction, material_volume = continental_material_fields(state, areas)
    state.continental_fraction = material_fraction
    state.continental_volume_km3 = material_volume
    if transport_source_index is not None:
        src = np.asarray(transport_source_index, dtype=np.int32)
        if src.shape != (mesh.cell_count,):
            raise ValueError("transport_source_index must have shape (cell_count,)")
        potential = np.zeros(mesh.cell_count, dtype=np.float64)
        valid = (src >= 0) & (src < mesh.cell_count)
        potential[valid] = np.asarray(cycle.felsic_potential, dtype=np.float64)[src[valid]]
    elif previous_lithosphere is not None and plate_system is not None:
        potential = _advect_felsic_potential(
            mesh, cycle.felsic_potential, previous_lithosphere, lithosphere, plate_system, dt_myr
        )
    else:
        potential = np.asarray(cycle.felsic_potential, dtype=np.float64).copy()

    # Arc-production/thickening/erosion intensities are capped per cell so a
    # vertex touching several boundary edges cannot gain several time steps of
    # magmatism at once purely because of mesh tessellation.
    juvenile_intensity = np.zeros(mesh.cell_count, dtype=np.float64)
    continental_arc_intensity = np.zeros(mesh.cell_count, dtype=np.float64)
    erosion_intensity = np.zeros(mesh.cell_count, dtype=np.float64)

    external_arc = None
    if volcanic_arc_forcing is not None:
        external_arc = np.clip(np.asarray(volcanic_arc_forcing, dtype=np.float64), 0.0, 2.0)
        if external_arc.shape != (mesh.cell_count,):
            raise ValueError("volcanic_arc_forcing must have shape (cell_count,)")
        # v0.21: one material-aware arc forcing field feeds the existing crust
        # cycle. Oceanic footprint matures toward juvenile felsic crust, while
        # continental footprint thickens. No extra mass source is introduced.
        juvenile_intensity = external_arc * (1.0 - np.clip(material_fraction, 0.0, 1.0))
        continental_arc_intensity = external_arc * np.clip(material_fraction, 0.0, 1.0)

    for b in boundaries:
        if b.boundary_type != BoundaryType.CONVERGENT:
            continue
        a, c = int(b.face_a), int(b.face_b)
        ta = int(state.crust_type[a]); tb = int(state.crust_type[c])
        conv = min(abs(float(b.normal_rate_km_per_myr)) / params.convergence_reference_km_per_myr, 2.0)
        if conv <= 0.0:
            continue

        if external_arc is None and ta == int(CrustType.OCEANIC) and tb == int(CrustType.OCEANIC):
            # Legacy boundary-local arc placement, retained for old runners/tests.
            if state.crust_age_myr[a] > state.crust_age_myr[c] + 1e-9:
                upper = c
            elif state.crust_age_myr[c] > state.crust_age_myr[a] + 1e-9:
                upper = a
            else:
                upper = min(a, c)
            juvenile_intensity[upper] = max(juvenile_intensity[upper], conv)
            for nb in mesh.neighbors[upper]:
                if state.cell_plate[nb] == state.cell_plate[upper] and state.crust_type[nb] == int(CrustType.OCEANIC):
                    juvenile_intensity[nb] = max(juvenile_intensity[nb], 0.45 * conv)

        elif ta != tb:
            # Forearc tectonic erosion remains tied to the plate interface. Arc
            # magmatism itself is geometry-displaced in v0.21 when forcing is supplied.
            upper = a if ta == int(CrustType.CONTINENTAL) else c
            if external_arc is None:
                continental_arc_intensity[upper] = max(continental_arc_intensity[upper], conv)
                for nb in mesh.neighbors[upper]:
                    if state.cell_plate[nb] == state.cell_plate[upper] and state.crust_type[nb] == int(CrustType.CONTINENTAL):
                        continental_arc_intensity[nb] = max(continental_arc_intensity[nb], 0.35 * conv)
            erosion_intensity[upper] = max(erosion_intensity[upper], conv)

    # Felsic/island-arc maturation.  Potential decays away from active arcs.
    decay = np.exp(-float(dt_myr) / max(params.arc_potential_decay_myr, 1e-9))
    potential *= decay
    potential += juvenile_intensity * params.arc_maturation_rate_per_myr * float(dt_myr)

    ocean_before = state.crust_type == int(CrustType.OCEANIC)
    mature = ocean_before & (potential >= params.juvenile_threshold)
    juvenile_area = float(np.sum(areas[mature]))
    juvenile_volume = 0.0
    if np.any(mature):
        old_material_volume = material_volume[mature].copy()
        state.crust_type[mature] = int(CrustType.CONTINENTAL)
        state.crust_age_myr[mature] = float(params.juvenile_seed_age_myr)
        state.crust_thickness_km[mature] = float(params.juvenile_continental_thickness_km)
        material_fraction[mature] = 1.0
        material_volume[mature] = areas[mature] * float(params.juvenile_continental_thickness_km)
        state.tidal_damage[mature] *= 0.7
        if state.rift_extension is not None:
            state.rift_extension[mature] = 0.0
        if state.extension_age_myr is not None:
            state.extension_age_myr[mature] = 0.0
        if state.continental_lithosphere_age_myr is not None:
            state.continental_lithosphere_age_myr[mature] = 0.0
        if state.mantle_depletion_fraction is not None:
            state.mantle_depletion_fraction[mature] = 0.0
        if state.craton_strength is not None:
            state.craton_strength[mature] = 0.0
        # v0.23 ledger counts only newly generated *tracked continental*
        # material.  Oceanic basalt replaced by the juvenile arc was never part
        # of the continental reservoir and must not be subtracted here.
        juvenile_volume = float(np.sum(material_volume[mature] - old_material_volume))
        potential[mature] = 0.0

    # Continental volcanic arcs add lower/middle-crustal volume.
    cont = state.crust_type == int(CrustType.CONTINENTAL)
    thicken = cont & (continental_arc_intensity > 0.0)
    arc_thickening_volume = 0.0
    if np.any(thicken):
        delta = np.minimum(
            params.max_arc_thickening_km_per_step,
            continental_arc_intensity[thicken] * params.continental_arc_thickening_km_per_myr * float(dt_myr),
        )
        state.crust_thickness_km[thicken] += delta
        material_volume[thicken] += areas[thicken] * material_fraction[thicken] * delta
        # Mixed coastline cells only contain continental material over their
        # fractional footprint.
        arc_thickening_volume = float(np.sum(areas[thicken] * material_fraction[thicken] * delta))

    # Slow v0.9.8-style gravitational collapse conservatively spreads
    # over-thickened continental crust within the same plate.
    collapse_redistributed_volume = _gravitational_collapse(
        mesh, state, areas, params, dt_myr
    )
    # Gravitational collapse changes only lateral thickness distribution and is
    # conservative. Sync visible continental material volume to that field.
    if collapse_redistributed_volume > 0.0:
        vis = state.crust_type == int(CrustType.CONTINENTAL)
        material_volume[vis] = areas[vis] * material_fraction[vis] * state.crust_thickness_km[vis]

    # Tectonic/subduction erosion removes damaged continental forearc volume.
    erosion_cells = (
        (state.crust_type == int(CrustType.CONTINENTAL))
        & (erosion_intensity > 0.0)
        & (state.tidal_damage >= params.subduction_erosion_damage_threshold)
    )
    erosion_volume = 0.0
    recycled_area = 0.0
    if np.any(erosion_cells):
        idx = np.flatnonzero(erosion_cells)
        damage_factor = np.clip(
            (state.tidal_damage[idx] - params.subduction_erosion_damage_threshold)
            / max(1.0 - params.subduction_erosion_damage_threshold, 1e-9),
            0.0,
            1.0,
        )
        delta = erosion_intensity[idx] * (0.25 + 0.75 * damage_factor) * params.subduction_erosion_km_per_myr * float(dt_myr)
        delta = np.minimum(delta, np.maximum(state.crust_thickness_km[idx] - params.recycle_below_thickness_km, 0.0) + 0.4)
        before = state.crust_thickness_km[idx].copy()
        state.crust_thickness_km[idx] = np.maximum(0.0, before - delta)
        removed_h = before - state.crust_thickness_km[idx]
        removed_v = areas[idx] * material_fraction[idx] * removed_h
        material_volume[idx] = np.maximum(0.0, material_volume[idx] - removed_v)
        erosion_volume = float(np.sum(removed_v))

        recycle = idx[state.crust_thickness_km[idx] < params.recycle_below_thickness_km]
        if len(recycle):
            recycled_area = float(np.sum(areas[recycle] * material_fraction[recycle]))
            # Remaining thin felsic crust is assumed tectonically removed and
            # replaced by newly formed oceanic lithosphere at the surface.
            erosion_volume += float(np.sum(material_volume[recycle]))
            state.crust_type[recycle] = int(CrustType.OCEANIC)
            state.crust_age_myr[recycle] = 0.0
            state.crust_thickness_km[recycle] = float(oceanic_thickness_km)
            material_fraction[recycle] = 0.0
            material_volume[recycle] = 0.0
            state.tidal_damage[recycle] *= 0.5
            if state.rift_extension is not None:
                state.rift_extension[recycle] = 0.0
            if state.extension_age_myr is not None:
                state.extension_age_myr[recycle] = 0.0
            if state.continental_lithosphere_age_myr is not None:
                state.continental_lithosphere_age_myr[recycle] = 0.0
            if state.mantle_depletion_fraction is not None:
                state.mantle_depletion_fraction[recycle] = 0.0
            if state.craton_strength is not None:
                state.craton_strength[recycle] = 0.0
            potential[recycle] = 0.0

    # Dense lower crust in very thick orogens/arcs can delaminate.  This removes
    # volume without necessarily destroying the continental surface area.
    cont = state.crust_type == int(CrustType.CONTINENTAL)
    delam = cont & (state.crust_thickness_km > params.delamination_threshold_km)
    delam_volume = 0.0
    if np.any(delam):
        idx = np.flatnonzero(delam)
        excess = np.maximum(state.crust_thickness_km[idx] - params.delamination_target_km, 0.0)
        frac = 1.0 - np.exp(-params.delamination_rate_per_myr * float(dt_myr))
        delta = np.minimum(excess, excess * frac)
        state.crust_thickness_km[idx] -= delta
        delam_removed = areas[idx] * material_fraction[idx] * delta
        material_volume[idx] = np.maximum(0.0, material_volume[idx] - delam_removed)
        delam_volume = float(np.sum(delam_removed))

    # Potential has no meaning under already continental crust.
    potential[state.crust_type == int(CrustType.CONTINENTAL)] = 0.0
    state.continental_fraction = material_fraction
    state.continental_volume_km3 = material_volume

    new_cycle = ContinentalCycleState(
        time_myr=float(state.time_myr),
        felsic_potential=potential,
        cumulative_generated_area_km2=float(cycle.cumulative_generated_area_km2 + juvenile_area),
        cumulative_recycled_area_km2=float(cycle.cumulative_recycled_area_km2 + recycled_area),
        cumulative_generated_volume_km3=float(cycle.cumulative_generated_volume_km3 + juvenile_volume + arc_thickening_volume),
        cumulative_recycled_volume_km3=float(cycle.cumulative_recycled_volume_km3 + erosion_volume + delam_volume),
    )

    total_area = float(np.sum(areas))
    cont_mask = state.crust_type == int(CrustType.CONTINENTAL)
    ocean_mask = ~cont_mask
    diag = ContinentalCycleDiagnostics(
        time_myr=float(state.time_myr),
        dt_myr=float(dt_myr),
        continental_area_fraction=float(np.sum(areas * material_fraction) / total_area),
        continental_volume_km3=_continental_volume_km3(areas, state),
        juvenile_arc_area_created_km2=juvenile_area,
        arc_thickening_volume_km3=arc_thickening_volume,
        subduction_erosion_area_km2=recycled_area,
        subduction_erosion_volume_km3=erosion_volume,
        delaminated_volume_km3=delam_volume,
        gravitational_collapse_redistributed_volume_km3=collapse_redistributed_volume,
        mean_felsic_potential_oceanic=float(np.mean(potential[ocean_mask])) if np.any(ocean_mask) else 0.0,
        max_felsic_potential=float(np.max(potential)) if len(potential) else 0.0,
    )
    return state, new_cycle, diag


__all__ = [
    "ContinentalCycleState",
    "ContinentalCycleDiagnostics",
    "ContinentalCycleParameters",
    "initialize_continental_cycle",
    "advance_continental_cycle",
]
