"""Optional geometric estimates along existing plate-boundary chains.

This module does not change ownership, boundary classification, or forces.
``normals`` are coarse geometric estimates, not replacement finite-volume face
normals. In particular, smoothing cannot infer whether a short jog is a real
transform segment or a raster artifact. Keep ``raw_normals`` for conservative
fluxes and validate any use of the estimates in dynamics separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import SphereMesh


@dataclass(slots=True)
class BoundaryGeometry:
    edges: np.ndarray
    midpoints: np.ndarray
    raw_normals: np.ndarray
    normals: np.ndarray
    support_counts: np.ndarray
    protected_edges: np.ndarray


def _unit(v: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(length < 1e-14):
        raise ValueError("Degenerate boundary geometry")
    return v / length


def estimate_boundary_normals(
    mesh: SphereMesh,
    cell_plate: np.ndarray,
    radius_km: float,
    *,
    half_width_km: float = 0.0,
    max_corner_turn_degrees: float = 75.0,
) -> BoundaryGeometry:
    """Estimate oriented normals using nearby edges of the same boundary.

    Output edge order is the inter-plate subset of ``mesh.shared_edges``, as
    used by ``classify_boundaries``. Both normal fields point from face_a to
    face_b. The raw field uses that classifier's centroid-difference formula.

    Support follows arc length along a nonbranching, same-pair edge chain,
    never proximity across unrelated/disconnected boundaries. All edges at
    triple junctions, branching vertices, and sharp corners are protected;
    support cannot cross those vertices. Neighboring normals are parallel
    transported to the central midpoint and weighted by edge length and a
    triangular distance kernel. A zero width returns the raw normals exactly.

    The corner guard preserves resolved right-angle transforms, but cannot
    certify the physical meaning of smaller bends or unresolved segments.
    """
    radius = float(radius_km)
    width = float(half_width_km)
    corner = float(max_corner_turn_degrees)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius_km must be finite and positive")
    if not np.isfinite(width) or width < 0:
        raise ValueError("half_width_km must be finite and nonnegative")
    if not np.isfinite(corner) or not 0 <= corner <= 180:
        raise ValueError("max_corner_turn_degrees must be in [0, 180]")
    owner = np.asarray(cell_plate)
    if owner.shape != (mesh.cell_count,):
        raise ValueError("cell_plate must match the mesh")
    if not np.issubdtype(owner.dtype, np.integer) or np.any(owner < 0):
        raise ValueError("cell_plate must contain nonnegative integer IDs")
    all_edges = np.asarray(mesh.shared_edges, dtype=np.int64).reshape(-1, 4)
    edges = all_edges[owner[all_edges[:, 0]] != owner[all_edges[:, 1]]]
    count = len(edges)
    if not count:
        empty = np.empty((0, 3), dtype=np.float64)
        return BoundaryGeometry(edges, empty.copy(), empty.copy(), empty.copy(),
                                np.empty(0, dtype=np.int32), np.empty(0, dtype=bool))
    fa, fb, vu, vv = edges.T
    midpoint = _unit(mesh.vertices[vu] + mesh.vertices[vv])
    direction = mesh.centroids[fb] - mesh.centroids[fa]
    direction -= midpoint * np.sum(direction * midpoint, axis=1, keepdims=True)
    raw = _unit(direction)
    result = raw.copy()
    support = np.ones(count, dtype=np.int32)
    protected = np.zeros(count, dtype=bool)
    if width == 0:
        return BoundaryGeometry(edges, midpoint, raw, result, support, protected)

    lengths = radius * np.arccos(np.clip(np.sum(mesh.vertices[vu] * mesh.vertices[vv], axis=1), -1, 1))
    pair = np.sort(np.column_stack((owner[fa], owner[fb])), axis=1)
    sign = np.where(owner[fa] < owner[fb], 1.0, -1.0)
    canonical = raw * sign[:, None]
    touch: dict[int, list[int]] = {}
    for i, (_, _, u, v) in enumerate(edges):
        touch.setdefault(int(u), []).append(i)
        touch.setdefault(int(v), []).append(i)

    # Two boundary edges can be continued only when they represent the same
    # plate pair and meet at a resolved, non-sharp point of the chain.
    links: dict[tuple[int, int], int] = {}
    for vertex, incident in touch.items():
        allowed = len(incident) == 2
        if allowed:
            a, b = incident
            allowed = bool(np.array_equal(pair[a], pair[b]))
        if allowed:
            r = mesh.vertices[vertex]
            away = []
            for i in incident:
                other = int(vv[i] if vu[i] == vertex else vu[i])
                p = mesh.vertices[other]
                away.append(_unit(p - r * np.dot(p, r)))
            turn = np.degrees(np.arccos(np.clip(-np.dot(away[0], away[1]), -1, 1)))
            allowed = bool(turn <= corner + 1e-10)
        if allowed:
            links[(a, vertex)] = b
            links[(b, vertex)] = a
        else:
            protected[incident] = True

    for center in range(count):
        if protected[center]:
            continue
        selected = {center: 0.0}
        for initial_vertex in (int(vu[center]), int(vv[center])):
            current, vertex, distance = center, initial_vertex, 0.0
            while (current, vertex) in links:
                nxt = links[(current, vertex)]
                distance += 0.5 * (lengths[current] + lengths[nxt])
                if distance >= width or nxt == center:
                    break
                # On a closed loop take the shorter along-chain distance.
                previous = selected.get(nxt, np.inf)
                if distance >= previous:
                    break
                selected[nxt] = float(distance)
                vertex = int(vv[nxt] if vu[nxt] == vertex else vu[nxt])
                current = nxt
        indices = np.asarray(sorted(selected), dtype=np.int64)
        if len(indices) == 1:
            continue
        source = midpoint[indices]
        target = midpoint[center]
        denominator = 1.0 + source @ target
        # Parallel transport along the shortest great circle is undefined at
        # antipodes. Such global smoothing is outside this local estimator.
        if np.any(denominator <= 1e-10):
            continue
        normals = canonical[indices]
        transported = normals - ((normals @ target) / denominator)[:, None] * (source + target)
        weights = lengths[indices] * (1.0 - np.array([selected[int(i)] for i in indices]) / width)
        average = np.sum(weights[:, None] * transported, axis=0)
        average -= target * np.dot(average, target)
        size = float(np.linalg.norm(average))
        # An ambiguous/U-shaped neighborhood must not reverse owner polarity.
        if size <= 1e-12 * float(np.sum(weights)) or np.dot(average, canonical[center]) <= 0:
            continue
        result[center] = sign[center] * average / size
        support[center] = len(indices)
    return BoundaryGeometry(edges, midpoint, raw, result, support, protected)


__all__ = ["BoundaryGeometry", "estimate_boundary_normals"]
