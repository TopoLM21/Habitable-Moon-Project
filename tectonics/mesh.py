"""Icosahedral spherical surface mesh.

The first prototype treats triangular faces as surface cells.  This gives a
nearly uniform spherical mesh with no polar singularity.  Plate boundaries are
shared face edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


Array = np.ndarray


@dataclass(slots=True)
class SphereMesh:
    vertices: Array               # (V, 3), unit vectors
    faces: Array                  # (F, 3), vertex indices
    centroids: Array              # (F, 3), unit vectors
    areas_unit_sphere: Array      # (F,), steradians on unit sphere
    neighbors: tuple[tuple[int, ...], ...]
    shared_edges: tuple[tuple[int, int, int, int], ...]
    # Each shared edge is (face_a, face_b, vertex_u, vertex_v), face_a < face_b.

    @property
    def cell_count(self) -> int:
        return int(self.faces.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    def physical_cell_areas_km2(self, radius_km: float) -> Array:
        return self.areas_unit_sphere * float(radius_km) ** 2


def _normalize(v: Array) -> Array:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("Cannot normalize a zero vector")
    return v / norm


def _base_icosahedron() -> tuple[Array, Array]:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        ],
        dtype=np.float64,
    )
    vertices = _normalize(vertices)

    faces = np.array(
        [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _subdivide(vertices: Array, faces: Array) -> tuple[Array, Array]:
    vertex_list = [v.copy() for v in vertices]
    midpoint_cache: dict[tuple[int, int], int] = {}

    def midpoint_index(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        cached = midpoint_cache.get(key)
        if cached is not None:
            return cached
        midpoint = vertices[a] + vertices[b]
        midpoint /= np.linalg.norm(midpoint)
        idx = len(vertex_list)
        vertex_list.append(midpoint)
        midpoint_cache[key] = idx
        return idx

    new_faces: list[tuple[int, int, int]] = []
    for a, b, c in faces:
        ab = midpoint_index(int(a), int(b))
        bc = midpoint_index(int(b), int(c))
        ca = midpoint_index(int(c), int(a))
        new_faces.extend(
            [
                (int(a), ab, ca),
                (int(b), bc, ab),
                (int(c), ca, bc),
                (ab, bc, ca),
            ]
        )

    return np.asarray(vertex_list, dtype=np.float64), np.asarray(new_faces, dtype=np.int64)


def _spherical_triangle_area(a: Array, b: Array, c: Array) -> float:
    """Robust unit-sphere triangle area (steradians)."""
    numerator = abs(float(np.dot(a, np.cross(b, c))))
    denominator = 1.0 + float(np.dot(a, b) + np.dot(b, c) + np.dot(c, a))
    return 2.0 * np.arctan2(numerator, denominator)


def _build_topology(faces: Array) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, int, int, int], ...]]:
    edge_owner: dict[tuple[int, int], int] = {}
    neighbor_sets = [set() for _ in range(len(faces))]
    shared: list[tuple[int, int, int, int]] = []

    for face_idx, (a, b, c) in enumerate(faces):
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            edge = (u, v) if u < v else (v, u)
            other = edge_owner.get(edge)
            if other is None:
                edge_owner[edge] = face_idx
            else:
                neighbor_sets[face_idx].add(other)
                neighbor_sets[other].add(face_idx)
                fa, fb = sorted((other, face_idx))
                shared.append((fa, fb, edge[0], edge[1]))

    if len(shared) != (3 * len(faces)) // 2:
        raise RuntimeError("Mesh is not a closed triangular manifold")

    neighbors = tuple(tuple(sorted(items)) for items in neighbor_sets)
    shared_edges = tuple(sorted(shared))
    return neighbors, shared_edges


def build_icosphere(subdivisions: int = 4) -> SphereMesh:
    if subdivisions < 0:
        raise ValueError("subdivisions must be non-negative")

    vertices, faces = _base_icosahedron()
    for _ in range(subdivisions):
        vertices, faces = _subdivide(vertices, faces)

    face_vertices = vertices[faces]
    centroids = _normalize(face_vertices.sum(axis=1))
    areas = np.fromiter(
        (_spherical_triangle_area(a, b, c) for a, b, c in face_vertices),
        dtype=np.float64,
        count=len(faces),
    )
    neighbors, shared_edges = _build_topology(faces)

    return SphereMesh(
        vertices=vertices,
        faces=faces,
        centroids=centroids,
        areas_unit_sphere=areas,
        neighbors=neighbors,
        shared_edges=shared_edges,
    )


def connected_components(indices: Iterable[int], neighbors: tuple[tuple[int, ...], ...]) -> list[list[int]]:
    remaining = set(int(i) for i in indices)
    components: list[list[int]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            current = stack.pop()
            for nxt in neighbors[current]:
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
                    component.append(nxt)
        components.append(component)
    return components
