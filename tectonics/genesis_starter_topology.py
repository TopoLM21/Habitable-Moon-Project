"""Reuse mature plate splitting for a physically supplied early-shell weak band.

This module contains no damage law or random partition. A supplied connected
weak region must already separate its parent plate into two sufficiently large
surviving domains. This includes a closed band on a boundaryless global shell.
An isolated patch, an open-ended line, and an entirely eligible shell do not
establish a separating cut and are therefore left unsplit.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .mesh import SphereMesh, connected_components
from .plates import PlateSystem
from .topology import (
    TopologyEvent,
    _component_span_km,
    _split_prepared_plate_cut,
)


def _positive(name: str, value: float, *, allow_zero: bool = False) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or (not allow_zero and value == 0.0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def _validate_system(mesh: SphereMesh, system: PlateSystem) -> np.ndarray:
    owner = np.asarray(system.cell_plate)
    if owner.shape != (mesh.cell_count,) or owner.dtype.kind not in "iu":
        raise ValueError("plate ownership must be an integer field matching the mesh")
    if not len(system.plates) or not np.array_equal(np.unique(owner), np.arange(len(system.plates))):
        raise ValueError("plate labels must be compact and every plate must own cells")
    for index, plate in enumerate(system.plates):
        cells = np.flatnonzero(owner == index)
        if plate.plate_id != index or plate.seed_cell not in cells:
            raise ValueError("plate order and seeds must agree with ownership")
        if len(connected_components(cells, mesh.neighbors)) != 1:
            raise ValueError("starter input plates must be connected")
        axis = np.asarray(plate.euler_axis, dtype=np.float64)
        if (axis.shape != (3,) or not np.all(np.isfinite(axis))
                or not np.isclose(np.linalg.norm(axis), 1.0, rtol=0.0, atol=1e-12)
                or not np.isfinite(plate.angular_speed_rad_per_myr)):
            raise ValueError("plate rotations must have finite speeds and unit Euler axes")
    return owner


def canonicalize_plate_seeds(mesh: SphereMesh, system: PlateSystem) -> PlateSystem:
    """Refresh stale geometric representatives after material transport.

    Seeds identify an owned raster cell; they are not material markers. A
    transport commit can move a plate away from its old seed without changing
    its ID or rotation. Only those stale representatives are replaced, using
    the lowest owned cell just as the mature topology constructor does. The
    input system, ownership and Euler arrays are never modified. Invalid
    labels, disconnected domains and invalid rotations still fail validation.
    """
    owner = np.asarray(system.cell_plate)
    if owner.shape != (mesh.cell_count,) or owner.dtype.kind not in "iu":
        raise ValueError("plate ownership must be an integer field matching the mesh")
    if not len(system.plates) or not np.array_equal(np.unique(owner), np.arange(len(system.plates))):
        raise ValueError("plate labels must be compact and every plate must own cells")
    plates = []
    changed = False
    for index, plate in enumerate(system.plates):
        cells = np.flatnonzero(owner == index)
        if plate.seed_cell in cells:
            plates.append(plate)
        else:
            plates.append(replace(plate, seed_cell=int(cells[0])))
            changed = True
    result = replace(system, plates=tuple(plates)) if changed else system
    _validate_system(mesh, result)
    return result


def _separated_parts(mesh, owner, cut, cell_areas, minimum_area):
    parent = int(owner[int(cut[0])])
    cells = np.flatnonzero(owner == parent)
    remaining = np.setdiff1d(cells, cut, assume_unique=True)
    components = connected_components(remaining, mesh.neighbors)
    large = [c for c in components if float(np.sum(cell_areas[c])) >= minimum_area]
    if len(large) < 2:
        return None
    large.sort(key=lambda c: (-float(np.sum(cell_areas[c])), -len(c), min(c)))
    return parent, cells, components, large[:2]


def select_starter_cut(
    mesh: SphereMesh,
    system: PlateSystem,
    eligible: np.ndarray,
    preference: np.ndarray,
    radius_km: float,
    min_child_area_km2: float,
    min_band_span_km: float,
) -> np.ndarray | None:
    """Choose a separating connected eligible component without inventing a path.

    Candidates are ranked by their area-weighted mean physical preference,
    then decreasing angular span, increasing area, parent ID and first cell.
    Preference is finite and nonnegative, with no prescribed physical units.
    Returned sorted cell IDs are always a subset of ``eligible``. Complete
    eligibility is deliberately insufficient: two coherent surviving blocks
    are required, rather than a partition inferred from an arbitrary seed.
    """
    owner = _validate_system(mesh, system)
    radius = _positive("radius_km", radius_km)
    minimum = _positive("min_child_area_km2", min_child_area_km2)
    span_limit = _positive("min_band_span_km", min_band_span_km, allow_zero=True)
    mask = np.asarray(eligible)
    weights = np.asarray(preference, dtype=np.float64)
    if mask.shape != (mesh.cell_count,) or mask.dtype.kind != "b":
        raise ValueError("eligible must be a boolean field matching the mesh")
    if (weights.shape != mask.shape or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)):
        raise ValueError("preference must be a finite nonnegative field matching the mesh")
    areas = mesh.physical_cell_areas_km2(radius)
    candidates = []
    for parent in range(len(system.plates)):
        cells = np.flatnonzero((owner == parent) & mask)
        for component in connected_components(cells, mesh.neighbors):
            cut = np.asarray(sorted(component), dtype=np.int32)
            span = _component_span_km(mesh, cut, radius)
            if span < span_limit or _separated_parts(mesh, owner, cut, areas, minimum) is None:
                continue
            area = float(np.sum(areas[cut]))
            preference_mean = float(np.sum(weights[cut] * (areas[cut] / area)))
            rank = (-preference_mean, -span, area, parent, int(cut[0]))
            candidates.append((rank, cut))
    if not candidates:
        return None
    return min(candidates, key=lambda entry: entry[0])[1].copy()


def split_starter_band(
    mesh: SphereMesh,
    system: PlateSystem,
    cut: np.ndarray,
    radius_km: float,
    min_child_area_km2: float,
    min_band_span_km: float,
    *,
    time_myr: float = 0.0,
    differential_speed_deg_per_myr: float = 0.0,
) -> tuple[PlateSystem | None, TopologyEvent | None]:
    """Apply a selected cut with the existing mature ownership/rotation operator.

    ``cut`` must be a connected, unique set of cells in a single parent. It is
    the caller's responsibility to supply physical eligibility (normally by
    ``select_starter_cut``). Geometrically nonseparating or undersized bands
    return ``(None, None)``. Invalid fields raise ``ValueError``. By default
    children inherit the parent's motion: no artificial separating kick is
    introduced. The caller may fit their motion to independently supplied flow.
    No source state is mutated, and no crust/age/damage fields are synthesized.
    """
    owner = _validate_system(mesh, system)
    radius = _positive("radius_km", radius_km)
    minimum = _positive("min_child_area_km2", min_child_area_km2)
    span_limit = _positive("min_band_span_km", min_band_span_km, allow_zero=True)
    time = _positive("time_myr", time_myr, allow_zero=True)
    differential = _positive("differential_speed_deg_per_myr", differential_speed_deg_per_myr, allow_zero=True)
    selected = np.asarray(cut)
    if selected.ndim != 1 or selected.dtype.kind not in "iu":
        raise ValueError("cut must be a one-dimensional integer array")
    if not len(selected):
        return None, None
    if (np.any(selected < 0) or np.any(selected >= mesh.cell_count)
            or len(np.unique(selected)) != len(selected)):
        raise ValueError("cut must contain unique in-range cell IDs")
    selected = np.sort(selected.astype(np.int32))
    if len(np.unique(owner[selected])) != 1:
        raise ValueError("cut must belong to a single parent plate")
    if len(connected_components(selected, mesh.neighbors)) != 1:
        raise ValueError("cut must be connected")
    span = _component_span_km(mesh, selected, radius)
    areas = mesh.physical_cell_areas_km2(radius)
    parts = _separated_parts(mesh, owner, selected, areas, minimum)
    if span < span_limit or parts is None:
        return None, None
    parent, cells, components, seeds = parts
    result, event = _split_prepared_plate_cut(
        mesh, system, parent, set(map(int, selected)), components, seeds, cells, radius,
        time_myr=time, differential_speed_deg_per_myr=differential,
        rift_span_km=span, rift_area_km2=float(np.sum(areas[selected])),
    )
    # The mature manager has a subsequent connectivity-repair stage; the starter
    # emits usable connected children directly or declines this candidate.
    if any(len(connected_components(np.flatnonzero(result.cell_plate == p), mesh.neighbors)) != 1
           for p in range(len(result.plates))):
        return None, None
    return result, event


__all__ = ["canonicalize_plate_seeds", "select_starter_cut", "split_starter_band"]
