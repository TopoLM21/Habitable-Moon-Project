"""Analytic spherical geometry and conservative partition checks."""

import math

import numpy as np
import pytest

from tectonics.mesh import build_icosphere
from tectonics.spherical_polygons import (
    GeometryDiagnostics, arc_length, clip_hemisphere, intersect_convex,
    normalize_polygon, polygon_area, polygon_halfspaces, rotate_polygon,
    shared_boundary_arcs, subtract_convex,
)


OCTANT = np.eye(3)


def patch(points):
    """A gnomonic chart: straight chart lines are exactly great circles."""
    return normalize_polygon([(x, y, 1.) for x, y in points])


def test_octant_area_and_orientation_and_scale():
    assert polygon_area(OCTANT) == pytest.approx(math.pi / 2, rel=2e-16)
    np.testing.assert_array_equal(normalize_polygon(OCTANT), OCTANT)
    np.testing.assert_array_equal(normalize_polygon(OCTANT[::-1]), OCTANT)
    np.testing.assert_allclose(normalize_polygon(OCTANT * np.array([2., 3., 7.])[:, None]), OCTANT)
    np.testing.assert_array_equal(polygon_halfspaces(OCTANT), OCTANT[[2, 0, 1]])


@pytest.mark.parametrize('bad', [
    [(1., 0., 0.), (0., 1., 0.)],
    [(0., 0., 0.), (0., 1., 0.), (0., 0., 1.)],
    [(float('nan'), 0., 0.), (0., 1., 0.), (0., 0., 1.)],
    [(1., 0., 0.), (0., 1., 0.), (-1., 0., 0.)],
    [(1., 0., 0.), (2., 0., 0.), (3., 0., 0.)],
    [(-1., -1., 1.), (1., 1., 1.), (-1., 1., 1.), (1., -1., 1.)],
    [(-1., -1., 1.), (1., -1., 1.), (0., 0., 1.), (1., 1., 1.), (-1., 1., 1.)],
])
def test_invalid_polygon_rejected(bad):
    with pytest.raises(ValueError):
        normalize_polygon(bad)


def test_empty_operations_and_duplicate_closing_vertex():
    assert normalize_polygon([]).shape == (0, 3)
    assert intersect_convex([], OCTANT).shape == (0, 3)
    assert subtract_convex([], OCTANT) == ()
    np.testing.assert_array_equal(subtract_convex(OCTANT, [])[0], OCTANT)
    diagnostics = GeometryDiagnostics()
    np.testing.assert_array_equal(normalize_polygon([*OCTANT, OCTANT[0]], diagnostics=diagnostics), OCTANT)
    assert diagnostics.duplicate_vertices_removed == 1


def test_clip_octant_through_pole_halves_area():
    half = clip_hemisphere(OCTANT, [1., -1., 0.])
    assert polygon_area(half) == pytest.approx(math.pi / 4, rel=4e-16)
    assert np.min(half @ np.array([1., -1., 0.])) >= -1e-16
    remaining = clip_hemisphere(OCTANT, [-1., 1., 0.])
    assert polygon_area(remaining) + polygon_area(half) == pytest.approx(math.pi / 2, rel=4e-16)


def test_coincident_clip_edges_and_point_contacts_are_zero_area():
    diagnostics = GeometryDiagnostics()
    np.testing.assert_array_equal(clip_hemisphere(OCTANT, [1., 0., 0.]), OCTANT)
    assert not len(clip_hemisphere(OCTANT, [-1., 0., 0.], diagnostics=diagnostics))
    assert diagnostics.lower_dimensional_results == 1
    assert not len(intersect_convex(OCTANT, normalize_polygon([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])))
    assert subtract_convex(OCTANT, OCTANT) == ()


@pytest.mark.parametrize('width', [1e-5, 1e-10, 1e-14])
def test_no_positive_area_cutoff_for_thin_fragment(width):
    strip = patch([(0., 0.), (width, 0.), (width, .1), (0., .1)])
    assert polygon_area(strip) > 0
    clipped = clip_hemisphere(strip, [1., 0., 0.])
    assert polygon_area(clipped) == pytest.approx(polygon_area(strip), rel=3e-15)
    tiny = patch([(0., 0.), (width, 0.), (0., width)])
    assert polygon_area(tiny) == pytest.approx(width ** 2 / 2, rel=max(1e-9, width ** 2))


def test_convex_difference_hole_partition_is_disjoint_and_complete():
    outer = patch([(-.2, -.2), (.2, -.2), (.2, .2), (-.2, .2)])
    inner = patch([(-.05, -.05), (.08, -.05), (.08, .07), (-.05, .07)])
    pieces = subtract_convex(outer, inner)
    assert len(pieces) == 4
    assert math.fsum(map(polygon_area, pieces)) + polygon_area(inner) == pytest.approx(polygon_area(outer), rel=1e-14)
    for i, piece in enumerate(pieces):
        assert polygon_area(intersect_convex(piece, inner)) < 1e-17
        for other in pieces[i + 1:]:
            assert polygon_area(intersect_convex(piece, other)) < 1e-17


@pytest.mark.parametrize('seed', range(8))
def test_rotated_mesh_triangle_intersection_difference_partition(seed):
    mesh = build_icosphere(1)
    a = normalize_polygon(mesh.vertices[mesh.faces[seed]])
    omega = np.random.default_rng(seed).normal(size=3) * .03
    b = rotate_polygon(a, omega, 1.)
    overlap = intersect_convex(a, b)
    parts = subtract_convex(a, b)
    assert polygon_area(overlap) + math.fsum(map(polygon_area, parts)) == pytest.approx(polygon_area(a), rel=2e-14)
    assert polygon_area(intersect_convex(a, b)) == pytest.approx(polygon_area(intersect_convex(b, a)), rel=2e-14)


def test_rigid_rotation_preserves_area_orientation_and_cycle():
    mesh = build_icosphere(2)
    a = normalize_polygon(mesh.vertices[mesh.faces[123]])
    rotated = rotate_polygon(a, [.9, -.2, .13], 11.)
    assert polygon_area(rotated) == pytest.approx(polygon_area(a), rel=3e-15)
    np.testing.assert_allclose(rotate_polygon(rotated, [-.9, .2, -.13], 11.), a, atol=2e-16)
    np.testing.assert_array_equal(normalize_polygon(rotated), rotated)
    np.testing.assert_array_equal(rotate_polygon(a, [0., 0., 0.], 100.), a)


def test_shared_partial_arc_uses_endpoints_and_opposing_normals():
    left = patch([(-.2, -.2), (0., -.2), (0., .2), (-.2, .2)])
    right = patch([(0., -.1), (.2, -.1), (.2, .1), (0., .1)])
    arcs = shared_boundary_arcs(left, right)
    assert len(arcs) == 1
    assert arc_length(*arcs[0]) == pytest.approx(2 * math.atan(.1), rel=2e-15)
    np.testing.assert_allclose(arcs[0][0], right[0], atol=0)
    np.testing.assert_allclose(arcs[0][1], right[-1], atol=0)
    assert shared_boundary_arcs(left, left) == ()
    assert len(shared_boundary_arcs(left, left, adjacent_only=False)) == 4


def test_shared_single_point_not_arc_and_distinct_nearby_planes_not_shared():
    a = patch([(-.2, -.2), (0., -.2), (0., 0.), (-.2, 0.)])
    b = patch([(0., 0.), (.2, 0.), (.2, .2), (0., .2)])
    assert shared_boundary_arcs(a, b) == ()
    shifted = patch([(1e-10, -.2), (.2, -.2), (.2, 0.), (1e-10, 0.)])
    assert shared_boundary_arcs(a, shifted) == ()


def test_shared_arcs_survive_common_rotation():
    mesh = build_icosphere(2)
    a, b, iu, iv = mesh.shared_edges[93]
    first = normalize_polygon(mesh.vertices[mesh.faces[a]])
    second = normalize_polygon(mesh.vertices[mesh.faces[b]])
    expected = arc_length(mesh.vertices[iu], mesh.vertices[iv])
    for omega in ([0., 0., 0.], [.001, -.004, .002], [3., -.4, 2.]):
        arcs = shared_boundary_arcs(rotate_polygon(first, omega, 1.), rotate_polygon(second, omega, 1.))
        assert len(arcs) == 1
        assert arc_length(*arcs[0]) == pytest.approx(expected, rel=2e-14)


def test_complete_rotated_mesh_partition_recovers_donor_area():
    mesh = build_icosphere(1)
    original = normalize_polygon(mesh.vertices[mesh.faces[18]])
    rotated = rotate_polygon(original, [.13, -.04, .02], 1.)
    areas = [polygon_area(intersect_convex(rotated, mesh.vertices[face])) for face in mesh.faces]
    assert sum(area > 0 for area in areas) >= 3
    assert math.fsum(areas) == pytest.approx(polygon_area(original), rel=3e-14)


@pytest.mark.parametrize('motion', [1e-3, 1e-6, 1e-9, 1e-12])
def test_short_mesh_fragments_remain_valid_and_partition_donor(motion):
    # Windows longdouble can equal double.  Direct cross(a,b) on tiny edges
    # used to reject narrow fragments after clipping; cross(a,b-a) is required.
    mesh = build_icosphere(2)
    for cell in (0, 18, 77, 91, 133, 217, 301):
        original = mesh.vertices[mesh.faces[cell]]
        moved = rotate_polygon(original, np.array([1., -.2, .6]) * motion, 1.)
        neighbors = np.flatnonzero(np.any(np.isin(mesh.faces, mesh.faces[cell]), axis=1))
        areas = []
        for neighbor in neighbors:
            fragment = intersect_convex(moved, mesh.vertices[mesh.faces[neighbor]])
            if len(fragment):
                normalize_polygon(fragment)
                areas.append(polygon_area(fragment))
        assert math.fsum(areas) == pytest.approx(polygon_area(moved), rel=1e-14)


@pytest.mark.parametrize('width', [1e-3, 1e-6, 1e-9, 1e-12])
def test_common_rotation_preserves_short_arc_against_long_neighbor(width):
    left = patch([(-1., -1.), (0., -1.), (0., 1.), (-1., 1.)])
    right = patch([(0., 1. - width), (1., 1. - width), (1., 1.), (0., 1.)])
    omega = [.31, .17, -.24]
    before = shared_boundary_arcs(left, right)
    after = shared_boundary_arcs(rotate_polygon(left, omega, 1.), rotate_polygon(right, omega, 1.))
    assert len(before) == len(after) == 1
    # Endpoint coordinate rounding is an absolute angular error, not a fixed
    # relative tolerance for a real arc only 5e-13 radians long.
    assert arc_length(*after[0]) == pytest.approx(arc_length(*before[0]), abs=3e-16)
