"""Material-preserving cuts through a spherical triangular shell.

Only the displacement connectivity changes: each face keeps its original
index, orientation, and material column. Face corners are joined across intact
edges. Corners meeting only across a cut receive independent vertex indices;
an isolated cut tip remains joined through the uncut fan around that vertex.

Geometric areas are evaluated face by face. Once banks move independently,
neither coverage of 4 pi nor absence of overlap follows from this topology.
Contact, opening budgets, and any overlap checks belong to the seam mechanics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import SphereMesh


@dataclass(slots=True)
class SeamTopology:
    """A split material mesh and its immutable reference edge identities.

    ``cut_edges[k]`` contains the sorted original vertex IDs. ``seam_faces[k]``
    gives the two incident faces in ascending order. ``bank_vertices[k, b]``
    contains the split vertex IDs on bank/face ``b``, ordered as ``cut_edges[k]``.
    Both banks may share one or both crack-tip vertices.

    ``original_shared_edges`` retains every reference edge, whereas
    ``intact_shared_edges`` contains only uncut reference edges. Their entries
    follow SphereMesh's (face_a, face_b, vertex_u, vertex_v) convention.
    ``mesh.shared_edges`` contains only intact edges, with split vertex IDs.
    """

    mesh: SphereMesh
    parent_vertex: np.ndarray
    original_faces: np.ndarray
    cut_edges: np.ndarray
    seam_faces: np.ndarray
    bank_vertices: np.ndarray
    original_shared_edges: tuple[tuple[int, int, int, int], ...]
    intact_shared_edges: tuple[tuple[int, int, int, int], ...]

    @property
    def seam_count(self) -> int:
        return int(len(self.cut_edges))

    @property
    def original_vertex_count(self) -> int:
        return int(self.parent_vertex.max()) + 1


class _UnionFind:
    def __init__(self, count: int):
        self.parents = np.arange(count, dtype=np.int64)

    def root(self, item: int) -> int:
        while self.parents[item] != item:
            self.parents[item] = self.parents[self.parents[item]]
            item = int(self.parents[item])
        return item

    def join(self, first: int, second: int):
        first, second = self.root(first), self.root(second)
        if first != second:
            # Deterministic roots make the output independent of edge order.
            low, high = sorted((first, second))
            self.parents[high] = low


def _face_geometry(vertices: np.ndarray, faces: np.ndarray):
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
        raise ValueError("Seam vertices must be finite with shape (vertex_count, 3)")
    if not np.allclose(np.linalg.norm(vertices, axis=1), 1., rtol=0., atol=2e-12):
        raise ValueError("Seam vertices must be unit vectors")
    triangles = vertices[faces]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    dots = np.stack((np.einsum("fi,fi->f", a, b), np.einsum("fi,fi->f", b, c),
                     np.einsum("fi,fi->f", c, a)), axis=1)
    if np.any(dots <= 0.):
        raise ValueError("Seam mesh edge reaches or spans 90 degrees")
    oriented_volume = np.einsum("fi,fi->f", a, np.cross(b, c))
    if np.any(oriented_volume <= 1e-14):
        raise ValueError("Seam mesh face is inverted or collapsed")
    areas = 2. * np.arctan2(oriented_volume, 1. + dots.sum(axis=1))
    centers = triangles.sum(axis=1)
    centers /= np.linalg.norm(centers, axis=1)[:, None]
    return vertices, centers, areas


def _closed_edges(mesh: SphereMesh):
    """Validate the source manifold from faces, without trusting cached edges."""
    faces = np.asarray(mesh.faces)
    vertices = np.asarray(mesh.vertices)
    if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0
            or not np.issubdtype(faces.dtype, np.integer)):
        raise ValueError("Source faces must be a nonempty integer (cell_count, 3) array")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Source vertices must have shape (vertex_count, 3)")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("Source face vertex index is out of range")
    if np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
        raise ValueError("Source face contains repeated vertices")
    if len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces):
        raise ValueError("Source mesh contains duplicate faces")
    if len(np.unique(faces)) != len(vertices):
        raise ValueError("Source mesh contains unused vertices")
    _, computed_centers, computed_areas = _face_geometry(vertices, faces)

    owners: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_id, triangle in enumerate(faces):
        for local_u, local_v in ((0, 1), (1, 2), (2, 0)):
            u, v = int(triangle[local_u]), int(triangle[local_v])
            key = (min(u, v), max(u, v))
            # Local endpoint indices are always ordered by the original IDs.
            low, high = (local_u, local_v) if u < v else (local_v, local_u)
            owners.setdefault(key, []).append((face_id, low, high))

    connected_corners = _UnionFind(3 * len(faces))
    shared = []
    for (u, v), adjacent in owners.items():
        if len(adjacent) != 2:
            raise ValueError("Source mesh must be a closed two-face-per-edge manifold")
        (fa, au, av), (fb, bu, bv) = adjacent
        if (av - au) % 3 == (bv - bu) % 3:
            raise ValueError("Source edge faces have inconsistent orientation")
        connected_corners.join(3 * fa + au, 3 * fb + bu)
        connected_corners.join(3 * fa + av, 3 * fb + bv)
        shared.append((fa, fb, u, v))
    # Two closed fans sharing only a vertex are edge-manifold but not a surface
    # manifold. Reject that bow-tie input before constructing material banks.
    vertex_roots: dict[int, int] = {}
    for corner, vertex in enumerate(faces.ravel()):
        root = connected_corners.root(corner)
        previous = vertex_roots.setdefault(int(vertex), root)
        if previous != root:
            raise ValueError("Source mesh has a nonmanifold vertex fan")
    # Face values are copied rather than recomputed at the split transition.
    # Do not preserve stale or malformed geometry caches from an input mesh.
    centers, areas = np.asarray(mesh.centroids), np.asarray(mesh.areas_unit_sphere)
    if (centers.shape != computed_centers.shape or areas.shape != computed_areas.shape
            or not np.all(np.isfinite(centers)) or not np.all(np.isfinite(areas))
            or not np.allclose(centers, computed_centers, rtol=0., atol=2e-12)
            or not np.allclose(areas, computed_areas, rtol=2e-12, atol=0.)):
        raise ValueError("Source face geometry is inconsistent with its vertices")
    return faces, owners, tuple(sorted(shared))


def _canonical_cuts(cut_edges, owners):
    cuts = np.asarray(cut_edges)
    if cuts.size == 0:
        if cuts.shape not in ((0,), (0, 2)):
            raise ValueError("Cut edges must have shape (seam_count, 2)")
        return np.empty((0, 2), dtype=np.int64)
    if (cuts.ndim != 2 or cuts.shape[1] != 2
            or not np.issubdtype(cuts.dtype, np.integer)):
        raise ValueError("Cut edges must be an integer (seam_count, 2) array")
    cuts = np.sort(cuts, axis=1)
    keys = [tuple(int(value) for value in edge) for edge in cuts]
    if len(set(keys)) != len(keys):
        raise ValueError("Cut edges must be unique, including reversed duplicates")
    if any(key not in owners for key in keys):
        raise ValueError("Every cut must be an existing source edge")
    return np.asarray(sorted(keys), dtype=np.int64).reshape(-1, 2)


def split_mesh(mesh: SphereMesh, cut_edges) -> SeamTopology:
    """Duplicate material vertex DOFs across selected reference mesh edges.

    ``mesh`` must be a closed oriented manifold. ``cut_edges`` contains pairs
    of source vertex IDs, in either endpoint or row order. An uncut vertex keeps
    its original index; when a vertex is split, its first corner component keeps
    that index and additional copies are appended. Material faces are never
    reordered, subdivided, or deleted.
    """
    original_faces, owners, original_shared = _closed_edges(mesh)
    cuts = _canonical_cuts(cut_edges, owners)
    cut_set = {tuple(edge) for edge in cuts}
    corners = _UnionFind(3 * mesh.cell_count)
    intact_reference = []
    for fa, fb, u, v in original_shared:
        if (u, v) in cut_set:
            continue
        (_, au, av), (_, bu, bv) = owners[(u, v)]
        corners.join(3 * fa + au, 3 * fb + bu)
        corners.join(3 * fa + av, 3 * fb + bv)
        intact_reference.append((fa, fb, u, v))

    original_count = mesh.vertex_count
    parent_vertex = list(range(original_count))
    split_faces = np.empty_like(original_faces, dtype=np.int64)
    vertex_components: list[dict[int, int]] = [{} for _ in range(original_count)]
    for corner, original_vertex in enumerate(original_faces.ravel()):
        vertex = int(original_vertex)
        components = vertex_components[vertex]
        component = corners.root(corner)
        if component not in components:
            if not components:
                components[component] = vertex
            else:
                components[component] = len(parent_vertex)
                parent_vertex.append(vertex)
        split_faces.flat[corner] = components[component]
    parent_vertex = np.asarray(parent_vertex, dtype=np.int64)

    neighbor_sets = [set() for _ in range(mesh.cell_count)]
    intact_split = []
    for fa, fb, u, v in intact_reference:
        (_, au, av), (_, bu, bv) = owners[(u, v)]
        su, sv = int(split_faces[fa, au]), int(split_faces[fa, av])
        assert su == split_faces[fb, bu] and sv == split_faces[fb, bv]
        neighbor_sets[fa].add(fb)
        neighbor_sets[fb].add(fa)
        intact_split.append((fa, fb, min(su, sv), max(su, sv)))

    seam_faces = np.empty((len(cuts), 2), dtype=np.int64)
    banks = np.empty((len(cuts), 2, 2), dtype=np.int64)
    for index, edge in enumerate(cuts):
        for bank, (face, low, high) in enumerate(owners[tuple(edge)]):
            seam_faces[index, bank] = face
            banks[index, bank] = split_faces[face, (low, high)]

    # Splitting alone has no geometric effect. Copy the cached face values to
    # preserve bitwise identity at the transition from the closed shell.
    split = SphereMesh(
        vertices=mesh.vertices[parent_vertex].copy(), faces=split_faces,
        centroids=mesh.centroids.copy(), areas_unit_sphere=mesh.areas_unit_sphere.copy(),
        neighbors=tuple(tuple(sorted(items)) for items in neighbor_sets),
        shared_edges=tuple(sorted(intact_split)),
    )
    return SeamTopology(
        mesh=split, parent_vertex=parent_vertex, original_faces=original_faces.copy(),
        cut_edges=cuts, seam_faces=seam_faces, bank_vertices=banks,
        original_shared_edges=original_shared,
        intact_shared_edges=tuple(intact_reference),
    )


def rebuild_seam_mesh(template: SeamTopology | SphereMesh,
                      vertices: np.ndarray) -> SphereMesh:
    """Update split-bank geometry while retaining every material face.

    The mesh may contain gaps or overlap, so its area sum is deliberately not
    constrained to 4 pi. This validates individual faces only; it does not
    certify a collision-free global configuration.
    """
    mesh = template.mesh if isinstance(template, SeamTopology) else template
    if np.shape(vertices) != (mesh.vertex_count, 3):
        raise ValueError("Seam vertices must have shape (vertex_count, 3)")
    vertices, centers, areas = _face_geometry(vertices, mesh.faces)
    return SphereMesh(
        vertices=vertices.copy(), faces=mesh.faces.copy(), centroids=centers,
        areas_unit_sphere=areas, neighbors=mesh.neighbors,
        shared_edges=mesh.shared_edges,
    )
