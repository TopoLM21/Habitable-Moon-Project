"""Connected plate ownership, independent of the transported material fields.

Each retained component is an immutable seed. Unresolved fragments are filled
from adjacent seeds by a simultaneous geodesic flood, so a label can never jump
across another plate. This changes mechanical ownership, not crust inventories.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .mesh import SphereMesh


@dataclass(slots=True)
class ConnectivityRepair:
    cell_plate: np.ndarray
    parent_by_plate: tuple[int, ...]
    components_before: int
    promoted_components: int
    reassigned_cells: int
    detached_cells_before: int


def repair_plate_connectivity(
    mesh: SphereMesh,
    cell_plate: np.ndarray,
    radius_km: float,
    *,
    minimum_independent_area_km2: float | None,
    minimum_independent_cells: int,
) -> ConnectivityRepair:
    """Return one connected domain per ID, preserving all retained domains.

    Keep the largest component of every existing plate, even a small plate still
    in its persistence interval. Promote other components at the independently
    surviving plate threshold. No velocity kick or physical scalar edit occurs.
    IDs returned here are contiguous; parent_by_plate maps them to input IDs.
    """
    owner = np.asarray(cell_plate, dtype=np.int32)
    if owner.shape != (mesh.cell_count,) or np.any(owner < 0):
        raise ValueError("cell_plate must contain one nonnegative ID per cell")
    if minimum_independent_cells < 1:
        raise ValueError("minimum_independent_cells must be positive")
    if minimum_independent_area_km2 is not None and minimum_independent_area_km2 <= 0:
        raise ValueError("minimum_independent_area_km2 must be positive")
    edges = np.asarray(mesh.shared_edges, dtype=np.int64)[:, :2]
    same = owner[edges[:, 0]] == owner[edges[:, 1]]
    links = edges[same]
    graph = coo_matrix((np.ones(len(links), dtype=np.int8),
                        (links[:, 0], links[:, 1])), shape=(len(owner), len(owner))).tocsr()
    count, component = connected_components(graph, directed=False)
    ids = np.unique(owner)
    if count == len(ids) and np.array_equal(ids, np.arange(len(ids))):
        return ConnectivityRepair(owner.copy(), tuple(map(int, ids)), count, 0, 0, 0)

    areas = mesh.physical_cell_areas_km2(radius_km)
    component_area = np.bincount(component, weights=areas)
    component_cells = np.bincount(component)
    # Every component contains one original ID. Stable sorting avoids repeated
    # full-mesh scans when historical files contain many fragments.
    order = np.argsort(component, kind="stable")
    offsets = np.r_[0, np.cumsum(component_cells)]
    first = order[offsets[:-1]]
    component_parent = owner[first]
    keep_label = np.full(count, -1, dtype=np.int32)
    parents = list(map(int, ids))
    promoted = 0
    detached_cells = 0
    for label, pid in enumerate(ids):
        candidates = np.flatnonzero(component_parent == pid)
        candidates = sorted(candidates, key=lambda c: (-component_area[c], int(first[c])))
        keep_label[candidates[0]] = label
        detached_cells += int(component_cells[candidates[1:]].sum())
        for comp in candidates[1:]:
            independent = (component_area[comp] >= minimum_independent_area_km2
                           if minimum_independent_area_km2 is not None
                           else component_cells[comp] >= minimum_independent_cells)
            if independent:
                keep_label[comp] = len(parents)
                parents.append(int(pid))
                promoted += 1
    repaired = keep_label[component]
    unknown = repaired < 0
    frontier = edges[unknown[edges[:, 0]] != unknown[edges[:, 1]]]
    heap: list[tuple[float, int, int]] = []
    for a, b in frontier:
        source, target = (int(b), int(a)) if unknown[a] else (int(a), int(b))
        distance = float(np.arccos(np.clip(mesh.centroids[source] @ mesh.centroids[target], -1, 1)))
        heap.append((distance, int(repaired[source]), target))
    heapq.heapify(heap)
    while heap:
        distance, label, cell = heapq.heappop(heap)
        if repaired[cell] >= 0:
            continue
        repaired[cell] = label
        for neighbor in mesh.neighbors[cell]:
            if repaired[neighbor] < 0:
                step = float(np.arccos(np.clip(mesh.centroids[cell] @ mesh.centroids[neighbor], -1, 1)))
                heapq.heappush(heap, (distance + step, label, int(neighbor)))
    if np.any(repaired < 0):
        raise RuntimeError("Connected spherical mesh flood left unassigned cells")
    inherited_owner = np.asarray(parents, dtype=np.int32)[repaired]
    reassigned = int(np.count_nonzero(inherited_owner != owner))
    return ConnectivityRepair(repaired, tuple(parents), count, promoted, reassigned, detached_cells)
