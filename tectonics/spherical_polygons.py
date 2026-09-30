"""Convex spherical polygons with minor great-circle edges.

The supported polygons fit in an open hemisphere.  Vertices are unit vectors,
ordered counterclockwise when viewed from outside the sphere.  Areas are in
steradians; empty and lower-dimensional results have shape ``(0, 3)``.

There is deliberately no minimum area or edge length.  A filtered dot-product
predicate treats a vertex as lying on a clipping plane only within its floating
point roundoff bound.  ``GeometryDiagnostics`` makes these ambiguous predicates
and exactly lower-dimensional results visible to callers.  Coordinates and
intersection arithmetic use extended precision where NumPy provides it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


Polygon = np.ndarray
_LD = np.longdouble
_EPS = np.finfo(np.float64).eps


@dataclass
class GeometryDiagnostics:
    """Arithmetic events, not permission to discard positive-area material."""

    ambiguous_plane_predicates: int = 0
    maximum_ambiguous_distance: float = 0.0
    duplicate_vertices_removed: int = 0
    lower_dimensional_results: int = 0
    rounded_intersections_reused: int = 0
    maximum_reused_endpoint_distance: float = 0.0


def _empty() -> Polygon:
    return np.empty((0, 3), dtype=np.float64)


def _unit(vector: Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=_LD)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("Expected a finite three-dimensional vector")
    norm = np.sqrt(np.sum(value * value))
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector")
    return np.asarray(value / norm, dtype=np.float64)


def _vertices(vertices: Sequence[Sequence[float]]) -> Polygon:
    values = np.asarray(vertices, dtype=np.float64)
    if values.size == 0:
        return _empty()
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("Polygon vertices must be a finite (N, 3) array")
    return np.asarray([_unit(value) for value in values])


def _deduplicate(vertices: Polygon, diagnostics: GeometryDiagnostics | None) -> Polygon:
    kept: list[np.ndarray] = []
    for value in vertices:
        if kept and np.array_equal(value, kept[-1]):
            if diagnostics is not None:
                diagnostics.duplicate_vertices_removed += 1
        else:
            kept.append(value)
    if len(kept) > 1 and np.array_equal(kept[0], kept[-1]):
        kept.pop()
        if diagnostics is not None:
            diagnostics.duplicate_vertices_removed += 1
    return np.asarray(kept, dtype=np.float64).reshape((-1, 3))


def _triangle_signed_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.longdouble:
    a, b, c = (np.asarray(value, dtype=_LD) for value in (a, b, c))
    # Difference form avoids subtracting O(1) cross products for tiny polygons.
    numerator = np.dot(a, np.cross(b - a, c - a))
    denominator = 1 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
    return 2 * np.arctan2(numerator, denominator)


def _signed_area(vertices: Polygon) -> float:
    if len(vertices) < 3:
        return 0.0
    return float(sum((_triangle_signed_area(vertices[0], vertices[i], vertices[i + 1])
                      for i in range(1, len(vertices) - 1)), _LD(0)))


def polygon_area(vertices: Sequence[Sequence[float]]) -> float:
    """Return a small convex polygon's area without a small-area cutoff."""
    return abs(_signed_area(_vertices(vertices)))


def arc_length(a: Sequence[float], b: Sequence[float]) -> float:
    """Minor arc length in radians, stable also for very short arcs."""
    u, v = (_unit(value).astype(_LD) for value in (a, b))
    return float(2 * np.arctan2(np.sqrt(np.sum((u - v) ** 2)),
                                np.sqrt(np.sum((u + v) ** 2))))


def _edge_normal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(_LD), b.astype(_LD)
    # Windows NumPy may alias longdouble to float64.  The difference form is
    # essential there: cross(a,b) loses relative accuracy on short clipped edges.
    cross = np.cross(a, b - a)
    if np.all(cross == 0):
        raise ValueError("Polygon contains a zero-length or antipodal edge")
    return _unit(cross)


def _distance(vertex: np.ndarray, normal: np.ndarray,
              diagnostics: GeometryDiagnostics | None) -> np.longdouble:
    products = vertex.astype(_LD) * normal.astype(_LD)
    distance = np.sum(products)
    # Normal and coordinates are stored as doubles.  This bound accounts for
    # their last-bit rounding as well as the dot product; it is not an area or
    # physical length tolerance.  Axis-aligned planes have no cancellation.
    bound = 8 * _EPS * np.sum(np.abs(products))
    if distance != 0 and abs(distance) <= bound:
        if diagnostics is not None:
            diagnostics.ambiguous_plane_predicates += 1
            diagnostics.maximum_ambiguous_distance = max(
                diagnostics.maximum_ambiguous_distance, float(abs(distance)))
        return _LD(0)
    return distance


def _edge_distance(vertex: np.ndarray, a: np.ndarray, b: np.ndarray,
                   diagnostics: GeometryDiagnostics | None) -> np.longdouble:
    """Signed plane distance preserving exact incidence at defining vertices.

    A rounded unit normal need not have an exactly zero dot product with its
    own endpoints.  The difference-form determinant does, including on Windows
    where longdouble has no extra precision.  Its error bound scales with the
    determinant's arithmetic terms rather than imposing a geometric tolerance.
    """
    if np.array_equal(vertex, a) or np.array_equal(vertex, b):
        return _LD(0)
    a, b, vertex = a.astype(_LD), b.astype(_LD), vertex.astype(_LD)
    ab, av = b-a, vertex-a
    normal = np.cross(a, ab)
    norm = np.sqrt(np.sum(normal*normal))
    cross = np.cross(ab, av)
    determinant = np.dot(a, cross)
    terms = np.array((abs(ab[1]*av[2])+abs(ab[2]*av[1]),
                      abs(ab[2]*av[0])+abs(ab[0]*av[2]),
                      abs(ab[0]*av[1])+abs(ab[1]*av[0])))
    arithmetic_bound = 16*_EPS*np.dot(np.abs(a), terms)
    # Coordinates produced by intersections/normalization carry last-bit
    # uncertainty too.  Propagate componentwise rounding through the three
    # determinant gradients; short-edge uncertainty is thereby amplified by
    # exactly its defining geometry, not by a fixed angular/area tolerance.
    coordinate_bound = 4*_EPS*(
        np.dot(np.abs(np.cross(b, vertex-b)), np.abs(a))
        + np.dot(np.abs(np.cross(vertex, a-vertex)), np.abs(b))
        + np.dot(np.abs(normal), np.abs(vertex)))
    bound = arithmetic_bound+coordinate_bound
    if determinant != 0 and abs(determinant) <= bound:
        if diagnostics is not None:
            diagnostics.ambiguous_plane_predicates += 1
            diagnostics.maximum_ambiguous_distance = max(
                diagnostics.maximum_ambiguous_distance, float(abs(determinant)/norm))
        return _LD(0)
    return determinant/norm


def normalize_polygon(vertices: Sequence[Sequence[float]], *,
                      diagnostics: GeometryDiagnostics | None = None) -> Polygon:
    """Normalize/orient a convex small polygon; reject invalid input.

    An explicitly empty input is permitted.  Nonempty degenerate, nonconvex,
    antipodal, or hemisphere-spanning input is an error.  Exact consecutive
    duplicates (including a repeated closing vertex) are harmless.
    """
    result = _deduplicate(_vertices(vertices), diagnostics)
    if not len(result):
        return result
    if len(result) < 3:
        raise ValueError("A nonempty polygon needs at least three distinct vertices")
    centre = _unit(np.sum(result.astype(_LD), axis=0))
    if np.any(result.astype(_LD) @ centre.astype(_LD) <= 0):
        raise ValueError("Polygon must fit in an open hemisphere about its vertex centre")
    area = _signed_area(result)
    if area == 0:
        raise ValueError("Polygon has zero area")
    if area < 0:
        result = result[::-1].copy()
    for edge_index, (a, b) in enumerate(zip(result, np.roll(result, -1, axis=0))):
        _edge_normal(a, b)
        # An edge's own endpoints define its plane.  Re-testing their rounded
        # unit normal can spuriously put them outside, especially near an axis.
        if any(_edge_distance(vertex, a, b, diagnostics) < 0 for index, vertex in enumerate(result)
               if index not in (edge_index, (edge_index + 1) % len(result))):
            raise ValueError("Polygon is not convex or has self-intersecting edges")
    return result


def polygon_halfspaces(vertices: Sequence[Sequence[float]]) -> np.ndarray:
    """Inward unit normals, one per positively oriented polygon edge."""
    polygon = normalize_polygon(vertices)
    return np.asarray([_edge_normal(a, b) for a, b in
                       zip(polygon, np.roll(polygon, -1, axis=0))]).reshape((-1, 3))


def _finish(vertices: list[np.ndarray], diagnostics: GeometryDiagnostics | None) -> Polygon:
    result = _deduplicate(np.asarray(vertices, dtype=float).reshape((-1, 3)), diagnostics)
    if len(result) < 3 or _signed_area(result) == 0:
        if diagnostics is not None and len(result):
            diagnostics.lower_dimensional_results += 1
        return _empty()
    if _signed_area(result) < 0:
        result = result[::-1].copy()
    return result


def _clip(polygon: Polygon, normal: np.ndarray,
          diagnostics: GeometryDiagnostics | None, *, plane_points=None) -> Polygon:
    if not len(polygon):
        return _empty()
    if plane_points is None:
        distances = [_distance(vertex, normal, diagnostics) for vertex in polygon]
    else:
        a, b = plane_points
        sign = 1 if np.dot(normal, _edge_normal(a, b)) > 0 else -1
        distances = [sign*_edge_distance(vertex, a, b, diagnostics) for vertex in polygon]
    if all(value >= 0 for value in distances):
        return polygon.copy()
    if all(value <= 0 for value in distances):
        if diagnostics is not None and any(value == 0 for value in distances):
            diagnostics.lower_dimensional_results += 1
        return _empty()
    output: list[np.ndarray] = []
    for i, b in enumerate(polygon):
        a = polygon[i - 1]
        da, db = distances[i - 1], distances[i]
        if (da < 0 < db) or (db < 0 < da):
            # Positive endpoint weights select the crossing on the minor arc.
            crossing = _unit(abs(db) * a.astype(_LD) + abs(da) * b.astype(_LD))
            # A true intersection at an existing endpoint can return the next
            # binary64 neighbor after normalization.  Reuse that endpoint if
            # their coordinate-rounding intervals overlap.  This is local to
            # a newly computed crossing; input polygons and positive-area
            # fragments are never filtered by size.
            for endpoint in (a, b):
                difference = float(np.linalg.norm(crossing-endpoint))
                # Rotation and normalized interpolation involve sums across
                # coordinates: cancellation can put an O(eps) error in a
                # coordinate whose true value is zero.  A normwise unit-vector
                # rounding bound is required, not relative error per component.
                bound = 4*_EPS*max(float(np.linalg.norm(crossing)), float(np.linalg.norm(endpoint)))
                if difference <= bound:
                    if diagnostics is not None and not np.array_equal(crossing, endpoint):
                        diagnostics.rounded_intersections_reused += 1
                        diagnostics.maximum_reused_endpoint_distance = max(
                            diagnostics.maximum_reused_endpoint_distance,
                            difference)
                    crossing = endpoint.copy()
                    break
            output.append(crossing)
        if db >= 0:
            output.append(b.copy())
    return _finish(output, diagnostics)


def clip_hemisphere(vertices: Sequence[Sequence[float]], normal: Sequence[float], *,
                    diagnostics: GeometryDiagnostics | None = None) -> Polygon:
    """Clip to ``normal dot x >= 0`` along exact great-circle boundaries."""
    return _clip(normalize_polygon(vertices, diagnostics=diagnostics), _unit(normal), diagnostics)


def intersect_convex(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]], *,
                     diagnostics: GeometryDiagnostics | None = None) -> Polygon:
    """Convex intersection; boundary-only intersections return an empty array."""
    result = normalize_polygon(a, diagnostics=diagnostics)
    other = normalize_polygon(b, diagnostics=diagnostics)
    if not len(result) or not len(other):
        return _empty()
    for u, v in zip(other, np.roll(other, -1, axis=0)):
        result = _clip(result, _edge_normal(u, v), diagnostics, plane_points=(u, v))
        if not len(result):
            break
    return result


def subtract_convex(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]], *,
                    diagnostics: GeometryDiagnostics | None = None) -> tuple[Polygon, ...]:
    """Partition ``a minus b`` into convex pieces with disjoint interiors.

    Each successive edge of b cuts an outside piece from the current remainder.
    Boundary duplication has zero area and no minimum-area filter is applied.
    """
    remainder = normalize_polygon(a, diagnostics=diagnostics)
    other = normalize_polygon(b, diagnostics=diagnostics)
    if not len(remainder):
        return ()
    if not len(other):
        return (remainder,)
    parts: list[Polygon] = []
    for u, v in zip(other, np.roll(other, -1, axis=0)):
        normal = _edge_normal(u, v)
        outside = _clip(remainder, -normal, diagnostics, plane_points=(u, v))
        if len(outside):
            parts.append(outside)
        remainder = _clip(remainder, normal, diagnostics, plane_points=(u, v))
        if not len(remainder):
            break
    return tuple(parts)


def rotate_polygon(vertices: Sequence[Sequence[float]], omega_rad_per_myr: Sequence[float],
                   dt_myr: float) -> Polygon:
    """Rigid Rodrigues rotation, independent of the cell containing the polygon."""
    polygon = normalize_polygon(vertices)
    omega = np.asarray(omega_rad_per_myr, dtype=_LD)
    if omega.shape != (3,) or not np.all(np.isfinite(omega)) or not math.isfinite(dt_myr):
        raise ValueError("Rotation requires finite omega and dt")
    magnitude = np.sqrt(np.sum(omega * omega))
    if magnitude == 0 or dt_myr == 0 or not len(polygon):
        return polygon.copy()
    axis = omega / magnitude
    angle = magnitude * _LD(dt_myr)
    sine, cosine = np.sin(angle), np.cos(angle)
    values = polygon.astype(_LD)
    rotated = values * cosine + np.cross(axis, values) * sine + (
        values @ axis)[:, None] * axis * (2 * np.sin(angle / 2) ** 2)
    return _vertices(rotated)


def shared_boundary_arcs(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]], *,
                         adjacent_only: bool = True,
                         diagnostics: GeometryDiagnostics | None = None
                         ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Positive-length coincident edge overlaps, oriented as polygon a.

    By default inward normals must oppose each other: coincident outer edges of
    overlapping/identical polygons are not an interface between adjacent regions.
    Shared endpoints alone do not produce an arc.  Collinear split edges may
    produce multiple consecutive arcs, preserving their geometric provenance.
    """
    first = normalize_polygon(a, diagnostics=diagnostics)
    second = normalize_polygon(b, diagnostics=diagnostics)
    arcs: list[tuple[np.ndarray, np.ndarray]] = []
    for u, v in zip(first, np.roll(first, -1, axis=0)):
        normal = _edge_normal(u, v)
        extent = arc_length(u, v)
        for x, y in zip(second, np.roll(second, -1, axis=0)):
            other_normal = _edge_normal(x, y)
            if adjacent_only and float(np.dot(normal, other_normal)) >= 0:
                continue
            # Use the longer edge to define the supporting plane.  The plane
            # reconstructed from a tiny edge has O(eps / edge_length) angular
            # uncertainty: extending it over a long neighbor would spuriously
            # reject real shared arcs after a common rigid rotation.
            if arc_length(x, y) > extent:
                support = other_normal * (1 if np.dot(normal, other_normal) > 0 else -1)
                probes = (u, v)
            else:
                support = normal
                probes = (x, y)
            if any(_distance(point, support, diagnostics) != 0 for point in probes):
                continue
            tangent = np.cross(support.astype(_LD), u.astype(_LD))
            tx = float(np.arctan2(np.dot(tangent, x), np.dot(u.astype(_LD), x)))
            ty = float(np.arctan2(np.dot(tangent, y), np.dot(u.astype(_LD), y)))
            # Conditional unwrapping avoids adding/subtracting pi from a tiny
            # angular interval and thereby destroying its low-order bits.
            if ty - tx > math.pi:
                ty -= 2 * math.pi
            elif ty - tx < -math.pi:
                ty += 2 * math.pi
            for shift in (-2 * math.pi, 0.0, 2 * math.pi):
                candidates = [(0.0, u), (extent, v), (tx + shift, x), (ty + shift, y)]
                low = max(0.0, min(tx, ty) + shift)
                high = min(extent, max(tx, ty) + shift)
                if high <= low:
                    continue
                start = min(candidates, key=lambda item: abs(item[0] - low))[1]
                end = min(candidates, key=lambda item: abs(item[0] - high))[1]
                if not np.array_equal(start, end):
                    arcs.append((start.copy(), end.copy()))
                break
    return tuple(arcs)
