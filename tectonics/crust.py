"""v0.3 oceanic-crust evolution on a fixed spherical diagnostic mesh.

This module upgrades the rigid kinematic v0.2 prototype with a single top
surface layer of oceanic crust:

- every mesh cell is occupied by exactly one top crust parcel after each step;
- existing parcels are advected by their plate's rigid Euler rotation;
- target cells left empty after advection are filled by *new* oceanic crust,
  representing spreading at divergent gaps;
- target cells receiving multiple parcels keep one top parcel while the rest
  are removed from the surface budget, representing subduction/consumption.

The algorithm is intentionally simplified.  It does not yet include continents,
real slab geometry, force balance, or self-consistent plate-motion changes.
However, it enforces the key surface bookkeeping that v0.2 lacked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .evolution import rotate_points_by_plate
from .kinematics import BoundaryRecord, classify_boundaries
from .mesh import SphereMesh
from .plates import PlateSystem

Array = np.ndarray


@dataclass(slots=True)
class OceanicCrustState:
    time_myr: float
    cell_plate: Array            # (N,), top-surface plate ownership
    crust_age_myr: Array         # (N,), top-surface oceanic crust age


@dataclass(slots=True)
class StepDiagnostics:
    time_myr: float
    dt_myr: float
    pre_resolution_gap_fraction: float
    pre_resolution_overlap_fraction: float
    created_area_km2: float
    subducted_area_km2: float
    mean_age_myr: float
    max_age_myr: float


@dataclass(slots=True)
class OceanicSnapshot:
    state: OceanicCrustState
    boundaries: list[BoundaryRecord]
    diagnostics: StepDiagnostics | None


def initialize_oceanic_crust(initial_system: PlateSystem) -> OceanicCrustState:
    """Initial top-surface state for v0.3.

    The prototype begins with oceanic crust everywhere and zero crust age.  The
    initial plate ownership comes directly from the v0.1/v0.2 partition.
    """
    cell_plate = np.asarray(initial_system.cell_plate, dtype=np.int32).copy()
    age = np.zeros_like(cell_plate, dtype=np.float64)
    return OceanicCrustState(time_myr=0.0, cell_plate=cell_plate, crust_age_myr=age)


def state_as_plate_system(state: OceanicCrustState, initial_system: PlateSystem) -> PlateSystem:
    return PlateSystem(cell_plate=np.asarray(state.cell_plate, dtype=np.int32), plates=initial_system.plates)


def boundary_records_for_state(
    mesh: SphereMesh,
    state: OceanicCrustState,
    initial_system: PlateSystem,
    radius_km: float,
    normal_threshold_km_per_myr: float,
    inactive_speed_km_per_myr: float,
) -> list[BoundaryRecord]:
    current = state_as_plate_system(state, initial_system)
    return classify_boundaries(
        mesh=mesh,
        system=current,
        radius_km=radius_km,
        normal_threshold_km_per_myr=normal_threshold_km_per_myr,
        inactive_speed_km_per_myr=inactive_speed_km_per_myr,
    )


def advance_oceanic_crust(
    mesh: SphereMesh,
    initial_system: PlateSystem,
    state: OceanicCrustState,
    dt_myr: float,
    radius_km: float,
) -> tuple[OceanicCrustState, StepDiagnostics]:
    """Advance the single-layer oceanic crust model by one time step.

    Existing top-surface parcels are advected rigidly with their current plate
    ownership.  The fixed diagnostic mesh is then reoccupied by nearest-cell
    assignment:

    - zero incoming parcels -> a gap, filled with newborn crust age 0;
    - one incoming parcel    -> simple advection and aging;
    - multiple parcels       -> overlap, one parcel survives on top and the
                                rest are removed from the surface (subduction).

    Winner rule in overlaps: the youngest parcel remains on top.  This is a
    simple proxy for older/denser oceanic lithosphere being preferentially
    consumed in ocean-ocean interactions.
    """
    if dt_myr <= 0.0:
        raise ValueError("dt_myr must be positive")

    n = mesh.cell_count
    cell_areas = mesh.physical_cell_areas_km2(radius_km)
    aged = np.asarray(state.crust_age_myr, dtype=np.float64) + float(dt_myr)

    advected_positions = rotate_points_by_plate(
        mesh.centroids,
        np.asarray(state.cell_plate, dtype=np.int32),
        initial_system,
        float(dt_myr),
    )

    target_tree = cKDTree(mesh.centroids)
    _, target_cells = target_tree.query(advected_positions, k=1, workers=-1)

    incoming: list[list[int]] = [[] for _ in range(n)]
    for source_idx, target_idx in enumerate(np.asarray(target_cells, dtype=np.int32)):
        incoming[int(target_idx)].append(int(source_idx))

    multiplicity = np.fromiter((len(lst) for lst in incoming), dtype=np.int32, count=n)
    gap_mask = multiplicity == 0
    overlap_mask = multiplicity > 1

    new_plate = np.empty(n, dtype=np.int32)
    new_age = np.empty(n, dtype=np.float64)

    created_area = float(np.sum(cell_areas[gap_mask]))
    subducted_area = 0.0

    # Resolve targets that receive at least one incoming parcel.
    for target_idx, candidates in enumerate(incoming):
        if not candidates:
            continue
        if len(candidates) == 1:
            winner = candidates[0]
            new_plate[target_idx] = int(state.cell_plate[winner])
            new_age[target_idx] = float(aged[winner])
            continue

        cand = np.asarray(candidates, dtype=np.int32)
        cand_age = aged[cand]
        cand_plate = np.asarray(state.cell_plate[cand], dtype=np.int32)
        # Prefer younger oceanic crust; break ties deterministically by plate id.
        best_order = np.lexsort((cand_plate, cand_age))
        winner = int(cand[best_order[0]])
        new_plate[target_idx] = int(state.cell_plate[winner])
        new_age[target_idx] = float(aged[winner])

        losers = cand[best_order[1:]]
        if len(losers):
            subducted_area += float(np.sum(cell_areas[losers]))

    # Fill advection gaps with newborn oceanic crust.  Assign the new crust to
    # the nearest surviving advected parcel, which approximates the local plate
    # that opens the rift.  This creates zero-age crust strips around divergent
    # openings without reusing the v0.2 nearest-marker ownership remap globally.
    if np.any(gap_mask):
        marker_tree = cKDTree(advected_positions)
        _, nearest_source = marker_tree.query(mesh.centroids[gap_mask], k=1, workers=-1)
        gap_sources = np.asarray(nearest_source, dtype=np.int32)
        new_plate[gap_mask] = np.asarray(state.cell_plate, dtype=np.int32)[gap_sources]
        new_age[gap_mask] = 0.0

    new_state = OceanicCrustState(
        time_myr=float(state.time_myr + dt_myr),
        cell_plate=new_plate,
        crust_age_myr=new_age,
    )
    diagnostics = StepDiagnostics(
        time_myr=new_state.time_myr,
        dt_myr=float(dt_myr),
        pre_resolution_gap_fraction=float(np.mean(gap_mask)),
        pre_resolution_overlap_fraction=float(np.mean(overlap_mask)),
        created_area_km2=created_area,
        subducted_area_km2=float(subducted_area),
        mean_age_myr=float(np.mean(new_age)),
        max_age_myr=float(np.max(new_age)),
    )
    return new_state, diagnostics


__all__ = [
    "OceanicCrustState",
    "StepDiagnostics",
    "OceanicSnapshot",
    "initialize_oceanic_crust",
    "state_as_plate_system",
    "boundary_records_for_state",
    "advance_oceanic_crust",
]
