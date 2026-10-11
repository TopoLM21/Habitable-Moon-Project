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
from fractions import Fraction
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
    squared_norm = np.sum(value*value)
    if (not np.isfinite(squared_norm) or squared_norm < np.finfo(_LD).tiny):
        # Keep ordinary normalization bits, but avoid an underflowed/overflowed
        # sum of squares when the vector itself is representable.  This is an
        # arithmetic-range branch, not a minimum geometric length.
        scale = np.max(np.abs(value))
        if scale == 0.:
            raise ValueError("Cannot normalize a zero vector")
        value = value/scale
        squared_norm = np.sum(value*value)
    norm = np.sqrt(squared_norm)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector")
    return np.asarray(value / norm, dtype=np.float64)


class _SupportedPolygon(np.ndarray):
    """Scratch clipping coordinates together with their original edge planes.

    A cut endpoint is rounded when stored.  Reconstructing its incoming plane
    from that rounded point can manufacture a second, different plane.  The
    metadata is local to polygon arithmetic; external coordinates and saved
    surface formats remain ordinary coordinate sequences.
    """

    def __array_finalize__(self, original):
        self.edge_supports = getattr(original, "edge_supports", None)
        self.support_vertices = getattr(original, "support_vertices", None)


def _supported(vertices, supports=None):
    result = np.asarray(vertices, dtype=float).reshape((-1, 3)).view(_SupportedPolygon)
    if supports is None:
        supports = tuple((a.copy(), b.copy()) for a, b in
                         zip(result, np.roll(result, -1, axis=0)))
    result.edge_supports = tuple(supports)
    result.support_vertices = np.asarray(result).copy()
    return result


def _supports(vertices):
    supports = getattr(vertices, "edge_supports", None)
    snapshot = getattr(vertices, "support_vertices", None)
    if (supports is not None and len(supports) == len(vertices)
            and snapshot is not None and np.array_equal(vertices, snapshot)):
        return supports
    return tuple((np.asarray(a).copy(), np.asarray(b).copy()) for a, b in
                 zip(vertices, np.roll(vertices, -1, axis=0)))


def _reverse_polygon(vertices):
    supports = _supports(vertices)
    count = len(vertices)
    reversed_supports = tuple(tuple(reversed(supports[(count-2-i) % count]))
                              for i in range(count))
    return _supported(np.asarray(vertices)[::-1].copy(), reversed_supports)


def _vertices(vertices: Sequence[Sequence[float]]) -> Polygon:
    # Rotations/intersections may supply extended-precision coordinates.
    # Normalize before the final binary64 rounding in _unit: an earlier cast
    # tilts short collinear edges and fabricates vacancies on their extension.
    values = np.asarray(vertices, dtype=_LD)
    if values.size == 0:
        return _empty()
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("Polygon vertices must be a finite (N, 3) array")
    result = np.asarray([_unit(value) for value in values])
    return _supported(result, _supports(vertices))


def _deduplicate(vertices: Polygon, diagnostics: GeometryDiagnostics | None) -> Polygon:
    kept: list[np.ndarray] = []
    incoming = []
    supports = _supports(vertices)
    for index, value in enumerate(vertices):
        if kept and np.array_equal(value, kept[-1]):
            if diagnostics is not None:
                diagnostics.duplicate_vertices_removed += 1
        else:
            kept.append(value)
            incoming.append(supports[(index-1) % len(vertices)])
    if len(kept) > 1 and np.array_equal(kept[0], kept[-1]):
        kept.pop()
        incoming[0] = incoming.pop()
        if diagnostics is not None:
            diagnostics.duplicate_vertices_removed += 1
    edges = tuple(incoming[(index+1) % len(incoming)] for index in range(len(incoming)))
    return _supported(kept, edges)


def _triangle_signed_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.longdouble:
    a, b, c = (np.asarray(value, dtype=_LD) for value in (a, b, c))
    # Short edges prefer differences; a long fan triangle ending at two
    # different tiny pole coordinates needs the direct triple product.
    numerator, _ = _determinant_value_and_error(a, b, c)
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


def _determinant_absolute_products(a, b, c):
    """Absolute six-product expansion, also for coordinate error boxes."""
    return (abs(a[0]*b[1]*c[2])+abs(a[1]*b[2]*c[0])+abs(a[2]*b[0]*c[1])
            +abs(a[0]*b[2]*c[1])+abs(a[1]*b[0]*c[2])+abs(a[2]*b[1]*c[0]))


def _polygon_projection_area_error(vertices, coordinate_errors):
    """Area envelope for a *certified* rounded projection of unit vertices.

    The caller supplies component errors from the executed rigid map.  This
    is not a tolerance for imported polygons.  Determinants are multilinear:
    retain their three first derivatives and every mixed error product.  The
    cross/dot interval for atan2 then bounds each fan triangle without an
    angular, coordinate, or area floor.  Infinity means no projection proof.
    """
    points = np.asarray(vertices, dtype=_LD)
    errors = np.asarray(coordinate_errors, dtype=_LD)
    if (points.shape != errors.shape or points.ndim != 2 or points.shape[1] != 3
            or not np.isfinite(errors).all() or np.any(errors < 0.)):
        raise ValueError("Projection needs finite componentwise coordinate errors")
    if len(points) < 3:
        return math.inf
    total = _LD(0.)
    a, ea = points[0], errors[0]
    for index in range(1, len(points)-1):
        b, c = points[index], points[index+1]
        eb, ec = errors[index], errors[index+1]
        determinant, determinant_arithmetic = _determinant_value_and_error(a, b, c)
        first = (np.dot(np.abs(_edge_cross(b, c)), ea)
                 +np.dot(np.abs(_edge_cross(c, a)), eb)
                 +np.dot(np.abs(_edge_cross(a, b)), ec))
        # The adaptive cross evaluations themselves have operation errors.
        first_arithmetic = _LD(16.*_EPS)*(
            _determinant_absolute_products(ea, b, c)
            +_determinant_absolute_products(a, eb, c)
            +_determinant_absolute_products(a, b, ec))
        mixed = (_determinant_absolute_products(ea, eb, c)
                 +_determinant_absolute_products(ea, b, ec)
                 +_determinant_absolute_products(a, eb, ec)
                 +_determinant_absolute_products(ea, eb, ec))
        determinant_error = determinant_arithmetic+first+first_arithmetic+mixed
        denominator = _LD(1.)+np.dot(a, b)+np.dot(b, c)+np.dot(c, a)
        denominator_error = _LD(16.*_EPS)*(
            _LD(1.)+np.dot(np.abs(a), np.abs(b))
            +np.dot(np.abs(b), np.abs(c))+np.dot(np.abs(c), np.abs(a)))
        for u, v, eu, ev in ((a, b, ea, eb), (b, c, eb, ec), (c, a, ec, ea)):
            denominator_error += (np.dot(np.abs(u), ev)+np.dot(np.abs(v), eu)
                                  +np.dot(eu, ev))
        # Dot product of the nominal atan2 vector with every perturbed vector.
        lower_dot = (denominator*denominator+determinant*determinant
                     -abs(denominator)*denominator_error
                     -abs(determinant)*determinant_error)
        if lower_dot <= 0.:
            return math.inf
        upper_cross = (abs(denominator)*determinant_error
                       +abs(determinant)*denominator_error)
        angle_error = _LD(2.)*np.arctan(upper_cross/lower_dot)
        total += angle_error+_LD(16.*_EPS)*abs(_triangle_signed_area(a, b, c))
    # Round the returned envelope outward, including possible subnormal bounds.
    return math.nextafter(float(total), math.inf)


def arc_length(a: Sequence[float], b: Sequence[float]) -> float:
    """Minor arc length in radians, stable also for very short arcs."""
    u, v = (_unit(value).astype(_LD) for value in (a, b))
    return float(2 * np.arctan2(np.sqrt(np.sum((u - v) ** 2)),
                                np.sqrt(np.sum((u + v) ** 2))))


def _difference_with_error(value, anchor):
    """Subtract saved floating coordinates with an explicit operation bound.

    Sterbenz's lemma makes subtraction of same-sign comparable floating
    operands exact, including when longdouble is binary64.  Outside that
    interval the ordinary absolute subtraction bound is retained.
    """
    value, anchor = np.asarray(value, dtype=_LD), np.asarray(anchor, dtype=_LD)
    exact = ((value == anchor) | (value == 0.) | (anchor == 0.)
             | ((value*anchor > 0.) & (np.abs(value) >= np.abs(anchor)/_LD(2.))
                & (np.abs(value) <= _LD(2.)*np.abs(anchor))))
    return value-anchor, np.where(exact, _LD(0.),
                                  _LD(_EPS)*(np.abs(value)+np.abs(anchor)))


def _edge_cross(a, b):
    """Choose each cross component's less uncertain algebraic evaluation.

    Direct products preserve a long edge's tiny pole components; differences
    preserve the plane of a short clipped edge.  The choice compares operation
    error bounds and neither changes coordinates nor infers coincident planes.
    """
    return _edge_cross_with_error(a, b)[0]


def _edge_cross_with_error(a, b):
    """Adaptive cross product and the bounds of its chosen evaluations."""
    a, b = np.asarray(a, dtype=_LD), np.asarray(b, dtype=_LD)
    delta, delta_error = _difference_with_error(b, a)
    values, errors = [], []
    for j, k in ((1, 2), (2, 0), (0, 1)):
        direct_terms = (a[j]*b[k], a[k]*b[j])
        difference_terms = (a[j]*delta[k], a[k]*delta[j])
        direct_bound = _LD(8.*_EPS)*sum(map(abs, direct_terms))
        difference_bound = (_LD(8.*_EPS)*sum(map(abs, difference_terms))
                            +abs(a[j])*delta_error[k]+abs(a[k])*delta_error[j])
        use_difference = difference_bound <= direct_bound
        terms = difference_terms if use_difference else direct_terms
        values.append(terms[0]-terms[1])
        factors = ((a[j], delta[k]), (a[k], delta[j])) if use_difference else (
            (a[j], b[k]), (a[k], b[j]))
        # A relative product-error bound does not cover gradual underflow.
        # Keep the chosen cross coordinates, but deny a certificate when a
        # nonzero product enters that arithmetic regime.  Normalization could
        # otherwise amplify an underflowed zero into a representable component
        # whose alleged error is also zero.
        underflowed = any(x != 0. and y != 0. and abs(term) < np.finfo(_LD).tiny
                          for (x, y), term in zip(factors, terms))
        errors.append(_LD(np.inf) if underflowed else min(difference_bound, direct_bound))
    return np.asarray(values, dtype=_LD), np.asarray(errors, dtype=_LD)


def _edge_normal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(_LD), b.astype(_LD)
    cross = _edge_cross(a, b)
    if np.all(cross == 0):
        raise ValueError("Polygon contains a zero-length or antipodal edge")
    return _unit(cross)


def _determinant_value_and_error(a, b, vertex):
    """A saved triple product with the smaller proven operation-error bound."""
    a, b, vertex = (np.asarray(value, dtype=_LD) for value in (a, b, vertex))
    ab, ab_error = _difference_with_error(b, a)
    av, av_error = _difference_with_error(vertex, a)
    difference_determinant = np.dot(a, np.cross(ab, av))
    terms = np.array((abs(ab[1]*av[2])+abs(ab[2]*av[1]),
                      abs(ab[2]*av[0])+abs(ab[0]*av[2]),
                      abs(ab[0]*av[1])+abs(ab[1]*av[0])))
    difference_bound = _LD(16*_EPS)*np.dot(np.abs(a), terms)
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        difference_bound += abs(a[i])*(
            ab_error[j]*abs(av[k])+ab_error[k]*abs(av[j])
            +av_error[k]*abs(ab[j])+av_error[j]*abs(ab[k])
            +ab_error[j]*av_error[k]+ab_error[k]*av_error[j])
    direct_terms = np.asarray((a[0]*b[1]*vertex[2], a[1]*b[2]*vertex[0],
                              a[2]*b[0]*vertex[1], -a[0]*b[2]*vertex[1],
                              -a[1]*b[0]*vertex[2], -a[2]*b[1]*vertex[0]), dtype=_LD)
    direct_determinant = np.sum(direct_terms)
    direct_bound = _LD(16*_EPS)*np.sum(np.abs(direct_terms))
    return ((direct_determinant, direct_bound) if direct_bound < difference_bound
            else (difference_determinant, difference_bound))


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
    own endpoints.  Adaptive direct/difference determinants preserve long
    pole edges and short clipped edges, including when longdouble has no extra
    precision.  Their error bounds scale with the actual arithmetic terms.
    """
    if np.array_equal(vertex, a) or np.array_equal(vertex, b):
        return _LD(0)
    a, b, vertex = a.astype(_LD), b.astype(_LD), vertex.astype(_LD)
    normal = _edge_cross(a, b)
    norm = np.sqrt(np.sum(normal*normal))
    determinant, arithmetic_bound = _determinant_value_and_error(a, b, vertex)
    # Keep the existing coordinate-rounding contract for constructed vertices
    # and supporting rays.  Literal saved-input separation is proved separately
    # in intersect_convex; it must not be confused with construction incidence.
    coordinate_bound = 4*_EPS*(
        np.dot(np.abs(_edge_cross(b, vertex)), np.abs(a))
        +np.dot(np.abs(_edge_cross(vertex, a)), np.abs(b))
        +np.dot(np.abs(normal), np.abs(vertex)))
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
        result = _reverse_polygon(result)
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
    return np.asarray([_edge_normal(a, b) for a, b in _supports(polygon)]).reshape((-1, 3))


def _finish(vertices: list[np.ndarray], diagnostics: GeometryDiagnostics | None,
            incoming_supports=None) -> Polygon:
    edges = (None if incoming_supports is None else
             tuple(incoming_supports[(i+1) % len(incoming_supports)]
                   for i in range(len(incoming_supports))))
    result = _deduplicate(_supported(vertices, edges), diagnostics)
    if len(result) < 3 or _signed_area(result) == 0:
        if diagnostics is not None and len(result):
            diagnostics.lower_dimensional_results += 1
        return _empty()
    if _signed_area(result) < 0:
        result = _reverse_polygon(result)
    return result


def _plane_crossing_error(subject_support, clipping_support, crossing):
    """Executed nested plane intersection's normalized coordinate envelope.

    A plane-plane crossing replaces the interpolated crossing below, so its
    own cancellation and two inherited plane errors must reach endpoint reuse.
    A nonpositive lower norm gives no certificate; no endpoint may be reused
    on the strength of the unrelated interpolation error in that case.
    """
    first, first_error = _edge_cross_with_error(*subject_support)
    second, second_error = _edge_cross_with_error(*clipping_support)
    if not np.isfinite(first_error).all() or not np.isfinite(second_error).all():
        return None
    line, line_error = _edge_cross_with_error(first, second)
    if not np.isfinite(line_error).all():
        return None
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        line_error[i] += (
            abs(first[j])*second_error[k]+abs(first[k])*second_error[j]
            +abs(second[k])*first_error[j]+abs(second[j])*first_error[k]
            +first_error[j]*second_error[k]+first_error[k]*second_error[j])
    norm = np.hypot.reduce(line)
    norm_error = (np.hypot.reduce(line_error)
                  +_LD(8.*_EPS)*norm)
    lower_norm = norm-norm_error
    if not np.isfinite(lower_norm) or lower_norm <= 0.:
        return None
    envelope = (line_error/lower_norm+(np.abs(line)/norm)*(norm_error/lower_norm)
                +_LD(4.*_EPS)*np.abs(np.asarray(crossing, dtype=_LD)))
    if not np.isfinite(envelope).all() or np.any(envelope < 0.):
        return None
    return envelope


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
    incoming_supports = []
    supports = _supports(polygon)
    if plane_points is None:
        axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
        first = _unit(_edge_cross(normal, axis))
        clipping_support = (first, _unit(_edge_cross(normal, first)))
    else:
        clipping_support = (plane_points if np.dot(normal, _edge_normal(*plane_points)) > 0.
                            else tuple(reversed(plane_points)))
    for i, b in enumerate(polygon):
        a = polygon[i - 1]
        da, db = distances[i - 1], distances[i]
        subject_support = supports[(i-1) % len(polygon)]
        if (da < 0 < db) or (db < 0 < da):
            # Positive endpoint weights select the crossing on the minor arc.
            weighted = abs(db)*a.astype(_LD)+abs(da)*b.astype(_LD)
            weighted_norm = np.sqrt(np.sum(weighted*weighted))
            crossing = _unit(weighted)
            plane_crossing_error = None
            used_plane_crossing = False
            if plane_points is not None:
                # Both original planes survive intervening cuts.  Their
                # intersection avoids feeding rounded cut endpoints back into
                # a newly invented supporting plane.
                line = _edge_cross(_edge_cross(*subject_support),
                                   _edge_cross(*plane_points))
                if np.any(line != 0.):
                    used_plane_crossing = True
                    crossing = _unit(line)
                    if np.dot(crossing.astype(_LD), weighted) < 0.:
                        crossing = -crossing
                    plane_crossing_error = _plane_crossing_error(
                        subject_support, plane_points, crossing)
            absolute_terms = abs(db)*np.abs(a.astype(_LD))+abs(da)*np.abs(b.astype(_LD))
            # Propagate actual interpolation cancellation component by
            # component.  A generic O(1) cancellation can create a last-bit
            # zero-coordinate residual; an actual tiny pole displacement has
            # tiny arithmetic terms and therefore remains fully resolved.
            crossing_error = _LD(4.*_EPS)*(absolute_terms/weighted_norm+np.abs(crossing))
            if plane_crossing_error is not None:
                crossing_error = plane_crossing_error
            # A true intersection at an existing endpoint can return the next
            # binary64 neighbor after normalization.  Reuse that endpoint if
            # their coordinate-rounding intervals overlap.  This is local to
            # a newly computed crossing; input polygons and positive-area
            # fragments are never filtered by size.
            endpoints = (a, b) if plane_points is None else (a, b, *plane_points)
            if used_plane_crossing and plane_crossing_error is None:
                endpoints = ()
            for endpoint in endpoints:
                endpoint_error = crossing_error+_LD(4.*_EPS)*np.abs(endpoint)
                if np.any(np.abs(crossing.astype(_LD)-endpoint.astype(_LD)) > endpoint_error):
                    continue
                # Incidence must be compatible with that same construction
                # error envelope, rather than with a geographic distance.
                distance = (_distance(endpoint, normal, diagnostics) if plane_points is None else
                            _edge_distance(endpoint, *plane_points, diagnostics))
                support_normal = normal if plane_points is None else _edge_normal(*plane_points)
                incident = abs(distance) <= np.dot(np.abs(support_normal), endpoint_error)
                if not incident:
                    continue
                subject_distance = _edge_distance(endpoint, *subject_support, diagnostics)
                if abs(subject_distance) > np.dot(np.abs(_edge_normal(*subject_support)),
                                                 endpoint_error):
                    continue
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
            incoming_supports.append(subject_support if da > 0. else clipping_support)
        if db >= 0:
            output.append(b.copy())
            incoming_supports.append(subject_support if da >= 0. or db > 0. else clipping_support)
    return _finish(output, diagnostics, incoming_supports)


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
    if (_saved_binary64_rays(a) and _saved_binary64_rays(b)
            and _saved_inputs_separated(result, other)):
        return _empty()
    for u, v in _supports(other):
        result = _clip(result, _edge_normal(u, v), diagnostics, plane_points=(u, v))
        if not len(result):
            break
    return result


def _saved_binary64_rays(vertices):
    """Literal saved rays, without coercing another caller's input precision."""
    if isinstance(vertices, _SupportedPolygon):
        return False
    return all(type(coordinate) in (float, np.float64)
               for row in vertices for coordinate in row)


def _saved_ray_products_resolved(a, b, vertex):
    """Whether both determinant evaluations permit relative product bounds.

    A nonzero product in gradual underflow needs an absolute IEEE bound.
    Declining the fast sign sends the unchanged rays to exact arithmetic.
    Check the left-associated intermediate products as well as final terms.
    """
    tiny = np.finfo(_LD).tiny
    def resolved(x, y):
        product = x*y
        return (np.isfinite(product) and
                (x == 0. or y == 0. or abs(product) >= tiny))
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1),
                    (0, 2, 1), (1, 0, 2), (2, 1, 0)):
        if not resolved(a[i], b[j]) or not resolved(a[i]*b[j], vertex[k]):
            return False
    ab, av = b-a, vertex-a
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        if not resolved(ab[j], av[k]) or not resolved(ab[k], av[j]):
            return False
        cross_component = ab[j]*av[k]-ab[k]*av[j]
        if not resolved(a[i], cross_component):
            return False
    return True


def _saved_ray_side(a, b, vertex, exact_rays):
    """Sign of the exact dyadic determinant of three saved binary64 rays."""
    if np.array_equal(vertex, a) or np.array_equal(vertex, b):
        return 0
    a, b, vertex = (np.asarray(value, dtype=_LD) for value in (a, b, vertex))
    value, bound = _determinant_value_and_error(a, b, vertex)
    if (np.isfinite(value) and np.isfinite(bound) and bound > 0.
            and _saved_ray_products_resolved(a, b, vertex)
            and abs(value) > np.nextafter(bound, _LD(np.inf))):
        return 1 if value > 0. else -1
    def exact(ray):
        key = tuple(map(float, ray))
        if key not in exact_rays:
            exact_rays[key] = tuple(Fraction.from_float(x) for x in key)
        return exact_rays[key]
    a, b, vertex = map(exact, (a, b, vertex))
    determinant = sum(a[i]*(b[j]*vertex[k]-b[k]*vertex[j])
                      for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)))
    return (determinant > 0)-(determinant < 0)


def _saved_inputs_separated(first, second):
    """Prove no positive intersection of validated ordinary saved inputs.

    Normalization changes an exact ray only by a positive scale.  The initial
    supports retain those input rays and their orientation.  A candidate edge
    must contain every vertex of its own polygon in its inward halfspace; all
    vertices of the other polygon must be nonpositive, with at least one
    strictly outside.  Exact shared endpoints or rational fallback certify
    zeros.  Scratch inherited planes have a different construction contract
    and never enter this saved-domain proof.
    """
    exact_rays = {}
    for region, other in ((first, second), (second, first)):
        own_vertices = tuple(u for u, _ in _supports(region))
        other_vertices = tuple(u for u, _ in _supports(other))
        for u, v in _supports(region):
            if any(_saved_ray_side(u, v, point, exact_rays) < 0
                   for point in own_vertices):
                continue
            sides = tuple(_saved_ray_side(u, v, point, exact_rays)
                          for point in other_vertices)
            if max(sides) <= 0 and min(sides) < 0:
                return True
    return False


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
    for u, v in _supports(other):
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


def _rigid_rotation_projection(vertices, omega_rad_per_myr, dt_myr):
    """The actual Rodrigues result and its componentwise arithmetic envelope.

    This proof is used only for temporary differential-motion candidates.
    Input/output normalization uses the same 4eps component envelope as the
    incidence predicates.  The remaining bounds follow the executed norm,
    axis division, trigonometry, cross/dot products and three-term sum; the
    working-precision epsilon also covers platforms where longdouble is double.
    """
    polygon = normalize_polygon(vertices)
    omega = np.asarray(omega_rad_per_myr, dtype=_LD)
    if (omega.shape != (3,) or not np.isfinite(omega).all()
            or not math.isfinite(dt_myr)):
        raise ValueError("Rotation requires finite omega and dt")
    values = np.asarray(polygon, dtype=_LD)
    magnitude = np.sqrt(np.sum(omega*omega))
    if magnitude == 0. and np.any(omega != 0.) and dt_myr != 0.:
        # A working-precision squared norm can underflow.  The ordinary map's
        # identity branch is not a proof for this nonzero physical vector.
        return polygon.copy(), np.full(values.shape, math.inf, dtype=_LD)
    if magnitude == 0. or dt_myr == 0. or not len(polygon):
        return polygon.copy(), _LD(4.*_EPS)*np.abs(values)
    work_eps = _LD(np.finfo(_LD).eps)
    axis = omega/magnitude
    angle = magnitude*_LD(dt_myr)
    sine, cosine = np.sin(angle), np.cos(angle)
    haversine = _LD(2.)*np.sin(angle/_LD(2.))**2
    cross = np.cross(axis, values)
    projection = values@axis
    first = values*cosine
    second = cross*sine
    third = projection[:, None]*axis*haversine
    rotated = first+second+third
    result = _vertices(rotated)

    value_error = _LD(4.*_EPS)*np.abs(values)
    axis_error = _LD(8.)*work_eps*np.abs(axis)
    angle_error = _LD(8.)*work_eps*abs(angle)
    sine_error = ((abs(cosine)+angle_error)*angle_error
                  +_LD(4.)*work_eps*abs(sine))
    cosine_error = ((abs(sine)+angle_error)*angle_error
                    +_LD(4.)*work_eps*abs(cosine))
    haversine_error = ((abs(sine)+angle_error)*angle_error
                       +_LD(8.)*work_eps*abs(haversine))
    cross_error = np.empty_like(values)
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        cross_error[:, i] = (
            abs(axis[j])*value_error[:, k]+abs(axis[k])*value_error[:, j]
            +axis_error[j]*np.abs(values[:, k])+axis_error[k]*np.abs(values[:, j])
            +axis_error[j]*value_error[:, k]+axis_error[k]*value_error[:, j]
            +_LD(8.)*work_eps*(abs(axis[j])*np.abs(values[:, k])
                              +abs(axis[k])*np.abs(values[:, j])))
    projection_error = (value_error@np.abs(axis)+np.abs(values)@axis_error
                        +value_error@axis_error
                        +_LD(8.)*work_eps*(np.abs(values)@np.abs(axis)))
    first_error = (np.abs(values)*cosine_error+abs(cosine)*value_error
                   +value_error*cosine_error+_LD(4.)*work_eps*np.abs(first))
    second_error = (np.abs(cross)*sine_error+abs(sine)*cross_error
                    +cross_error*sine_error+_LD(4.)*work_eps*np.abs(second))
    p, ep = np.abs(projection)[:, None], projection_error[:, None]
    ax, eax, hv, ehv = np.abs(axis), axis_error, abs(haversine), haversine_error
    third_error = (ep*ax*hv+p*eax*hv+p*ax*ehv
                   +ep*eax*hv+ep*ax*ehv+p*eax*ehv+ep*eax*ehv
                   +_LD(8.)*work_eps*np.abs(third))
    raw_error = (first_error+second_error+third_error
                 +_LD(8.)*work_eps*(np.abs(first)+np.abs(second)+np.abs(third)))
    norm = np.sqrt(np.sum(rotated*rotated, axis=1))
    norm_error = (np.sqrt(np.sum(raw_error*raw_error, axis=1))
                  +_LD(8.)*work_eps*norm)
    lower_norm = norm-norm_error
    if np.any(lower_norm <= 0.):
        return result, np.full(values.shape, math.inf, dtype=_LD)
    errors = (raw_error/lower_norm[:, None]
              +np.abs(rotated)*(norm_error/(norm*lower_norm))[:, None]
              +_LD(4.*_EPS)*np.abs(np.asarray(result, dtype=_LD)))
    # Both computations above must produce the ordinary, full rigid map.
    if not np.array_equal(result, rotate_polygon(vertices, omega_rad_per_myr, dt_myr)):
        raise RuntimeError("Rigid projection certificate differs from the executed map")
    return result, errors


def _shared_minor_arc(a0, a1, b0, b1, *, diagnostics=None):
    """Positive shared arc from saved Cartesian endpoints, oriented as a.

    A small interval next to a long arc's endpoint cannot be represented by
    subtracting two global angles near pi/2.  Oriented wedge membership and
    endpoint ordering therefore use the same adaptive determinant predicates
    as polygon incidence; every returned endpoint retains its saved values.
    """
    a0, a1, b0, b1 = (np.asarray(point, dtype=float) for point in (a0, a1, b0, b1))
    length_a, length_b = arc_length(a0, a1), arc_length(b0, b1)
    if not (0. < length_a < math.pi and 0. < length_b < math.pi):
        return None
    x, y = (a0, a1) if length_a >= length_b else (b0, b1)
    probes = (b0, b1) if length_a >= length_b else (a0, a1)
    if any(_edge_distance(point, x, y, diagnostics) != 0. for point in probes):
        return None
    normal_a, normal_b = _edge_normal(a0, a1), _edge_normal(b0, b1)

    def contains(point, start, end, normal):
        if np.array_equal(point, start) or np.array_equal(point, end):
            return True
        return (_edge_distance(point, normal, start, diagnostics) >= 0.
                and _edge_distance(point, end, normal, diagnostics) >= 0.)

    candidates = [point for point in (a0, a1, b0, b1)
                  if contains(point, a0, a1, normal_a)
                  and contains(point, b0, b1, normal_b)]
    if len(candidates) < 2:
        return None
    first = last = candidates[0]
    for point in candidates[1:]:
        before, _ = _determinant_value_and_error(normal_a, point, first)
        after, _ = _determinant_value_and_error(normal_a, last, point)
        if before > 0.:
            first = point
        if after > 0.:
            last = point
    if np.array_equal(first, last) or arc_length(first, last) == 0.:
        return None
    return first.copy(), last.copy()


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
            shared = _shared_minor_arc(u, v, x, y, diagnostics=diagnostics)
            if shared is not None:
                arcs.append(shared)
    return tuple(arcs)
