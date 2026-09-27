"""Conforming, conservative geometry for a supplied spherical crack support.

This module inserts a *prescribed* simple polyline into a closed material mesh.
It neither chooses a physical crack nor evolves one. Every original material
triangle is partitioned in its own gnomonic plane, where minor great-circle
arcs are straight. Edge intersections are shared by both original neighbors.
The resulting mesh remains closed until explicitly passed to ``split_mesh``.

All prospective front positions must be inserted together. Changing the active
interval only selects existing edges; it never changes the trial space.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree

from .genesis_crack_path import CrackInterval, ReferenceCrackPath
from .genesis_seams import _closed_edges, _face_geometry
from .mesh import SphereMesh, _build_topology


_ANGLE_TOL = 2e-12
_PLANAR_TOL = 2e-13


def _angle(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1),
                      np.sum(a*b, axis=-1))


def _edge(a, b):
    return (min(int(a), int(b)), max(int(a), int(b)))


def _cross2(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


@dataclass(frozen=True)
class PathMeshInsertion:
    """A closed subdivision and a fixed ordered support on that subdivision.

    ``parent_face`` identifies the original material cell of each child.
    ``vertex_parent_face`` and ``vertex_barycentric`` describe each vertex by
    its ray intersection with the *chordal* triangle of one original cell:
    normalizing the weighted original vertices reconstructs that unit vector.
    They are interpolation ancestry, not a claim of conserved nodal forces.
    Arrays are owned by this result; the supplied mesh is never modified.
    """

    mesh: SphereMesh
    parent_face: np.ndarray
    path_vertex_ids: np.ndarray
    path_arclength_m: np.ndarray
    vertex_parent_face: np.ndarray
    vertex_barycentric: np.ndarray
    path: ReferenceCrackPath

    @property
    def cut_edges(self):
        """All support edges, ordered along the path with sorted endpoints."""
        return np.sort(np.column_stack((self.path_vertex_ids[:-1],
                                        self.path_vertex_ids[1:])), axis=1)

    def cuts_for(self, interval: CrackInterval):
        """Select an explicit active interval without rebuilding its mesh.

        An absent front is an error: inserting it would change the geometry
        and invalidate comparisons made in this common discrete trial space.
        """
        if not isinstance(interval, CrackInterval):
            raise ValueError("Active support requires a CrackInterval")
        interval.validate(self.path)
        selected = []
        for coordinate in (interval.left_m, interval.right_m):
            distances = np.abs(self.path_arclength_m-coordinate)
            index = int(np.argmin(distances))
            tolerance = 64*np.finfo(float).eps*max(self.path.length_m, 1.)
            if distances[index] > tolerance:
                raise ValueError("Crack front was not preinserted; supply front_coordinates_m")
            selected.append(index)
        left, right = selected
        if right <= left:
            raise ValueError("Crack interval is below the mesh geometry resolution")
        return self.cut_edges[left:right].copy()


def _oriented(triangle, xy):
    a, b, c = triangle
    signed = float(_cross2(xy[b]-xy[a], xy[c]-xy[a]))
    if abs(signed) <= _PLANAR_TOL:
        raise ValueError("Crack insertion produces an unresolved planar sliver")
    return (a, b, c) if signed > 0 else (a, c, b)


def _owners(triangles):
    result = {}
    for i, (a, b, c) in enumerate(triangles):
        for edge in (_edge(a, b), _edge(b, c), _edge(c, a)):
            result.setdefault(edge, []).append(i)
    return result


def _ear_clip(polygon, xy):
    """Triangulate a simple cavity side, retaining collinear boundary nodes."""
    if len(polygon) < 3:
        return []
    polygon = list(polygon)
    signed = sum(float(_cross2(xy[a], xy[b]))
                 for a, b in zip(polygon, polygon[1:]+polygon[:1]))
    if signed < 0:
        polygon.reverse()
    result = []
    while len(polygon) > 3:
        found = False
        for i, b in enumerate(polygon):
            a, c = polygon[i-1], polygon[(i+1) % len(polygon)]
            if _cross2(xy[b]-xy[a], xy[c]-xy[a]) <= _PLANAR_TOL:
                continue
            others = [p for p in polygon if p not in (a, b, c)]
            if others:
                p = xy[others]
                inside = ((_cross2(xy[b]-xy[a], p-xy[a]) >= -_PLANAR_TOL)
                          & (_cross2(xy[c]-xy[b], p-xy[b]) >= -_PLANAR_TOL)
                          & (_cross2(xy[a]-xy[c], p-xy[c]) >= -_PLANAR_TOL))
                if inside.any():
                    continue
            result.append((a, b, c))
            del polygon[i]
            found = True
            break
        if not found:
            raise ValueError("Cannot resolve constrained triangulation cavity")
    result.append(_oriented(tuple(polygon), xy))
    return result


def _constrain(triangles, first, last, xy, protected):
    """Recover a straight constraint by retriangulating its crossed cavity."""
    wanted = _edge(first, last)
    owners = _owners(triangles)
    if wanted in owners:
        return triangles
    direction = xy[last]-xy[first]
    removed = set()
    for (u, v), incident in owners.items():
        if first in (u, v) or last in (u, v):
            continue
        side_u = float(_cross2(direction, xy[u]-xy[first]))
        side_v = float(_cross2(direction, xy[v]-xy[first]))
        side_first = float(_cross2(xy[v]-xy[u], xy[first]-xy[u]))
        side_last = float(_cross2(xy[v]-xy[u], xy[last]-xy[u]))
        if side_u*side_v < 0 and side_first*side_last < 0:
            if min(abs(side_u), abs(side_v), abs(side_first), abs(side_last)) <= _PLANAR_TOL:
                raise ValueError("Constraint passes too close to an unresolved mesh vertex")
            if (u, v) in protected:
                raise ValueError("Crack support constraints cross inside a material face")
            removed.update(incident)
    if not removed:
        raise ValueError("Could not recover a crack segment in its material face")
    cavity = _owners([triangles[i] for i in sorted(removed)])
    boundary = [edge for edge, faces in cavity.items() if len(faces) == 1]
    neighbors = {}
    for a, b in boundary:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    if (first not in neighbors or last not in neighbors
            or any(len(items) != 2 for items in neighbors.values())):
        raise ValueError("Constraint cavity is not a simple polygon")
    sides = []
    for initial in sorted(neighbors[first]):
        chain, previous, current = [first], first, initial
        while current != last:
            if current in chain:
                raise ValueError("Constraint cavity does not contain both endpoints")
            chain.append(current)
            following = neighbors[current]
            previous, current = current, (following[0] if following[1] == previous
                                          else following[1])
        chain.append(last)
        sides.extend(_ear_clip(chain, xy))
    result = [triangle for i, triangle in enumerate(triangles) if i not in removed]+sides
    final_edges = _owners(result)
    if wanted not in final_edges or not protected.issubset(final_edges):
        raise ValueError("Constrained triangulation lost a required material edge")
    return result


def _triangulate_face(ids, constraints, vertices, center, vertex_edges):
    ids = np.asarray(sorted(ids), dtype=np.int64)
    if len(ids) == 3:
        return [tuple(ids)]
    normal = center/np.linalg.norm(center)
    axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    east = np.cross(axis, normal)
    east /= np.linalg.norm(east)
    north = np.cross(normal, east)
    points = vertices[ids]
    denominators = points@normal
    if np.any(denominators <= 0):
        raise ValueError("Material face cannot be represented in one gnomonic chart")
    xy = np.column_stack((points@east, points@north))/denominators[:, None]
    xy -= xy.mean(axis=0)
    scale = np.max(np.ptp(xy, axis=0))
    xy /= scale
    try:
        triangulation = Delaunay(xy)
    except QhullError as exc:
        raise ValueError("Crack insertion has unresolved planar geometry") from exc
    if len(np.unique(triangulation.simplices)) != len(ids):
        raise ValueError("Planar triangulation omitted a material vertex")
    # Projection roundoff can make Qhull emit an exterior simplex composed of
    # an original edge's endpoints and its inserted intersection(s). Those
    # points are collinear by construction. Identify them from the actual
    # intersection provenance, not an area tolerance: a real thin material
    # triangle must still be retained or rejected as unresolved.
    triangles = []
    for triangle in triangulation.simplices:
        a, b, c = (int(v) for v in triangle)
        if vertex_edges[ids[a]] & vertex_edges[ids[b]] & vertex_edges[ids[c]]:
            continue
        triangles.append(_oriented((a, b, c), xy))
    if len({v for triangle in triangles for v in triangle}) != len(ids):
        raise ValueError("Planar triangulation contains an unresolved material vertex")
    local = {int(v): i for i, v in enumerate(ids)}
    protected = set()
    for a, b in sorted(constraints):
        first, last = local[a], local[b]
        triangles = _constrain(triangles, first, last, xy, protected)
        protected.add(_edge(first, last))
    return [tuple(int(ids[i]) for i in triangle) for triangle in triangles]


def insert_crack_path(mesh: SphereMesh, path: ReferenceCrackPath, *,
                      front_coordinates_m=()) -> PathMeshInsertion:
    """Subdivide a closed shell so a supplied continuous support follows edges.

    Original vertex IDs and coordinates are retained. Path bends, exact hits on
    old vertices, coincident old edges, and interior endpoints are supported.
    Near-degenerate geometry that cannot meet the existing seam mesh precision
    is rejected instead of snapping a path onto unrelated old-edge directions.
    ``front_coordinates_m`` are optional additional fixed prospective fronts.
    """
    if not isinstance(path, ReferenceCrackPath):
        raise ValueError("A ReferenceCrackPath is required")
    faces, _, shared = _closed_edges(mesh)
    coordinates = np.asarray(front_coordinates_m)
    if (coordinates.ndim != 1 or not np.issubdtype(coordinates.dtype, np.number)
            or np.iscomplexobj(coordinates) or not np.isfinite(coordinates).all()
            or np.any(coordinates < 0) or np.any(coordinates > path.length_m)):
        raise ValueError("Prospective fronts must be finite support coordinates")
    edges = np.asarray([(u, v) for _, _, u, v in shared], dtype=np.int64)
    edge_a, edge_b = mesh.vertices[edges[:, 0]], mesh.vertices[edges[:, 1]]
    edge_normals = np.cross(edge_a, edge_b)
    edge_normals /= np.linalg.norm(edge_normals, axis=1)[:, None]
    edge_lengths = _angle(edge_a, edge_b)
    events = [(float(s), p.copy(), None) for s, p in zip(path.arclength_m, path.points_xyz)]
    events.extend((float(s), path.point_at(float(s)), None) for s in coordinates)
    supplied_events = list(events)
    radius_m = path.radius_km*1000.
    for i, (a, b) in enumerate(zip(path.points_xyz[:-1], path.points_xyz[1:])):
        length = float(_angle(a, b))
        normal = np.cross(a, b)
        normal /= np.linalg.norm(normal)
        intersections = np.cross(normal, edge_normals)
        norms = np.linalg.norm(intersections, axis=1)
        regular = norms > _ANGLE_TOL
        regular_indices = np.flatnonzero(regular)
        candidates = intersections[regular]/norms[regular, None]
        for points in (candidates, -candidates):
            ea, eb = edge_a[regular], edge_b[regular]
            along = _angle(a, points)
            on_path = np.abs(along+_angle(points, b)-length) <= _ANGLE_TOL
            on_edge = np.abs(_angle(ea, points)+_angle(points, eb)-edge_lengths[regular]) <= _ANGLE_TOL
            selected = on_path & on_edge
            for point, distance, edge_index in zip(points[selected], along[selected],
                                                    regular_indices[selected]):
                events.append((float(path.arclength_m[i]+distance*radius_m),
                               point.copy(), tuple(edges[edge_index])))
        # Coincident great circles: retain each old endpoint inside this arc.
        for edge_index in np.flatnonzero(~regular):
            key = tuple(edges[edge_index])
            for point in (edge_a[edge_index], edge_b[edge_index]):
                along = float(_angle(a, point))
                if abs(along+float(_angle(point, b))-length) <= _ANGLE_TOL:
                    events.append((float(path.arclength_m[i]+along*radius_m), point.copy(), key))
            # A supplied endpoint/front inside a coincident edge is also on
            # that edge, although no isolated great-circle crossing exists.
            for coordinate, point, _ in supplied_events:
                if path.arclength_m[i] <= coordinate <= path.arclength_m[i+1]:
                    distance = float(_angle(edge_a[edge_index], point)
                                     + _angle(point, edge_b[edge_index]))
                    if abs(distance-edge_lengths[edge_index]) <= _ANGLE_TOL:
                        events.append((coordinate, point.copy(), key))
    events.sort(key=lambda item: item[0])
    vertices = [point.copy() for point in mesh.vertices]
    vertex_edges = [set() for _ in vertices]
    for edge in edges:
        key = tuple(edge)
        for vertex in edge:
            vertex_edges[vertex].add(key)
    old_tree = cKDTree(mesh.vertices)
    support_ids, support_s = [], []
    for coordinate, point, source_edge in events:
        distance, old_id = old_tree.query(point)
        if distance <= _ANGLE_TOL:
            vertex = int(old_id)
        elif support_ids and np.linalg.norm(vertices[support_ids[-1]]-point) <= _ANGLE_TOL:
            vertex = support_ids[-1]
        else:
            vertex = len(vertices)
            vertices.append(point.copy())
            vertex_edges.append(set())
        if source_edge is not None:
            vertex_edges[vertex].add(source_edge)
        if support_ids and vertex == support_ids[-1]:
            # Supplied front coordinates must survive intersection roundoff.
            exact = np.r_[path.arclength_m, coordinates]
            matched = exact[np.abs(exact-coordinate) <= _ANGLE_TOL*radius_m]
            if len(matched):
                support_s[-1] = float(matched[np.argmin(np.abs(matched-coordinate))])
            continue
        support_ids.append(vertex)
        support_s.append(coordinate)
    if len(support_ids) != len(set(support_ids)):
        raise ValueError("Inserted support revisits a mesh vertex at finite precision")
    support_s = np.asarray(support_s)
    support_s[0], support_s[-1] = 0., path.length_m
    if np.any(np.diff(support_s) <= _ANGLE_TOL*radius_m):
        raise ValueError("Prospective fronts or intersections are below mesh resolution")
    vertices = np.asarray(vertices)
    triangles = mesh.vertices[faces]
    inward = np.cross(triangles, np.roll(triangles, -1, axis=1))
    inward /= np.linalg.norm(inward, axis=2)[:, :, None]
    face_ids = [set(int(v) for v in face) for face in faces]
    constraints = [set() for _ in faces]
    vertex_faces = [[] for _ in vertices]
    for i, face in enumerate(faces):
        for vertex in face:
            vertex_faces[int(vertex)].append(i)

    def containing(point):
        found = np.flatnonzero(np.min(inward@point, axis=1) >= -_ANGLE_TOL)
        if not len(found):
            raise ValueError("A support point is outside the supplied closed shell")
        return found

    for vertex in support_ids:
        for face in containing(vertices[vertex]):
            face_ids[face].add(vertex)
            if int(face) not in vertex_faces[vertex]:
                vertex_faces[vertex].append(int(face))
    for a, b in zip(support_ids[:-1], support_ids[1:]):
        midpoint = vertices[a]+vertices[b]
        midpoint /= np.linalg.norm(midpoint)
        for face in containing(midpoint):
            if a not in face_ids[face] or b not in face_ids[face]:
                raise ValueError("A support segment crosses an unresolved material edge")
            constraints[face].add(_edge(a, b))
    new_faces, parents = [], []
    for parent, original in enumerate(faces):
        if len(face_ids[parent]) == 3:
            children = [tuple(int(v) for v in original)]
        else:
            children = _triangulate_face(face_ids[parent], constraints[parent],
                                         vertices, mesh.centroids[parent], vertex_edges)
            children = sorted(tuple(triangle[np.argmin(triangle):]+triangle[:np.argmin(triangle)])
                              for triangle in children)
        new_faces.extend(children)
        parents.extend([parent]*len(children))
    new_faces = np.asarray(new_faces, dtype=np.int64)
    parents = np.asarray(parents, dtype=np.int64)
    _, centers, areas = _face_geometry(vertices, new_faces)
    neighbors, new_edges = _build_topology(new_faces)
    refined = SphereMesh(vertices.copy(), new_faces, centers, areas, neighbors, new_edges)
    _closed_edges(refined)
    recovered = np.bincount(parents, weights=areas, minlength=mesh.cell_count)
    if not np.allclose(recovered, mesh.areas_unit_sphere, rtol=2e-12, atol=0):
        raise ValueError("Crack subdivision does not preserve each material parent area")
    existing_edges = {_edge(u, v) for _, _, u, v in new_edges}
    if any(_edge(a, b) not in existing_edges for a, b in zip(support_ids[:-1], support_ids[1:])):
        raise ValueError("Refined material mesh omitted a support segment")
    physical_length = float(np.sum(_angle(vertices[support_ids[:-1]],
                                          vertices[support_ids[1:]]))*radius_m)
    if abs(physical_length-path.length_m) > 8*_ANGLE_TOL*radius_m:
        raise ValueError("Insertion changed the prescribed crack support length")
    vertex_parent = np.asarray([min(items) for items in vertex_faces], dtype=np.int64)
    coefficients = np.linalg.solve(np.transpose(mesh.vertices[faces[vertex_parent]], (0, 2, 1)),
                                   vertices[..., None])[..., 0]
    barycentric = coefficients/coefficients.sum(axis=1)[:, None]
    if np.any(barycentric < -2e-10):
        raise ValueError("Vertex ancestry lies outside its original material cell")
    barycentric = np.maximum(barycentric, 0.)
    barycentric /= barycentric.sum(axis=1)[:, None]
    return PathMeshInsertion(refined, parents, np.asarray(support_ids, dtype=np.int64),
                             support_s, vertex_parent, barycentric, path)
