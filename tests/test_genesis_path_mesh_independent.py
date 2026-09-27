"""Independent geometry controls for a supplied, conforming crack support.

These cases test geometric insertion, not fracture initiation or propagation.
In particular, a simple open support must never manufacture detached plates.
"""
from collections import Counter

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tectonics.genesis_crack_path import CrackInterval, ReferenceCrackPath
from tectonics.genesis_path_mesh import insert_crack_path
from tectonics.genesis_seams import split_mesh
from tectonics.mesh import SphereMesh, build_icosphere, connected_components


RADIUS_KM = 5300.


def _unit(v):
    values = np.asarray(v, dtype=float)
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def _angle(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1),
                      np.sum(a*b, axis=-1))


def _oblique_path(angle=.319):
    """A fixed physical arc, independent of refinement and original edges."""
    start = _unit([.213, -.567, .812])
    first = _unit(np.cross(start, [.127, .911, -.031]))
    second = np.cross(start, first)
    tangent = np.cos(angle)*first + np.sin(angle)*second
    end = np.cos(1.013)*start + np.sin(1.013)*tangent
    return ReferenceCrackPath(np.asarray([start, _unit(end)]), RADIUS_KM)


def _assert_partition(source, inserted):
    """Independently check topology and geometry, without trusting caches."""
    refined = inserted.mesh
    faces = refined.vertices[refined.faces]
    volume = np.einsum("fi,fi->f", faces[:, 0],
                       np.cross(faces[:, 1], faces[:, 2]))
    assert np.all(volume > 1e-14)
    denominator = 1. + np.einsum("fi,fi->f", faces[:, 0], faces[:, 1])
    denominator += np.einsum("fi,fi->f", faces[:, 1], faces[:, 2])
    denominator += np.einsum("fi,fi->f", faces[:, 2], faces[:, 0])
    area = 2*np.arctan2(volume, denominator)
    np.testing.assert_allclose(refined.areas_unit_sphere, area, rtol=3e-12)
    np.testing.assert_allclose(
        np.bincount(inserted.parent_face, weights=area,
                    minlength=source.cell_count),
        source.areas_unit_sphere, rtol=3e-11, atol=2e-15)
    assert np.sum(area) == pytest.approx(4*np.pi, rel=3e-14)

    # Every child stays inside its assigned original material triangle.
    parent_triangles = source.vertices[source.faces[inserted.parent_face]]
    for k in range(3):
        normal = np.cross(parent_triangles[:, k], parent_triangles[:, (k+1) % 3])
        normal = _unit(normal)
        distances = np.einsum("fi,fji->fj", normal, faces)
        assert distances.min() > -3e-12

    edge_counts = Counter()
    directed = Counter()
    for a, b, c in refined.faces:
        for first, last in ((a, b), (b, c), (c, a)):
            edge_counts[tuple(sorted((int(first), int(last))))] += 1
            directed[(int(first), int(last))] += 1
    assert set(edge_counts.values()) == {2}
    assert all(count == directed[(b, a)] == 1
               for (a, b), count in directed.items())
    assert refined.vertex_count-len(edge_counts)+refined.cell_count == 2
    assert len(connected_components(range(refined.cell_count), refined.neighbors)) == 1


def _assert_exact_support(inserted, path):
    coordinates = inserted.path_arclength_m
    ids = inserted.path_vertex_ids
    assert len(ids) == len(coordinates)
    assert coordinates[0] == 0
    assert coordinates[-1] == pytest.approx(path.length_m, rel=1e-14)
    assert np.all(np.diff(coordinates) > 0)
    assert len(set(ids)) == len(ids)
    expected = path.point_at(coordinates)
    actual = inserted.mesh.vertices[ids]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=4e-12)
    np.testing.assert_allclose(
        _angle(actual[:-1], actual[1:])*RADIUS_KM*1000.,
        np.diff(coordinates), rtol=5e-9, atol=1e-6)
    expected_edges = np.sort(np.column_stack([ids[:-1], ids[1:]]), axis=1)
    np.testing.assert_array_equal(inserted.cut_edges, expected_edges)


def _assert_open_cut(inserted, interval):
    cuts = inserted.cuts_for(interval)
    degree = Counter(cuts.ravel())
    assert Counter(degree.values()) == {1: 2, 2: len(cuts)-1}
    topology = split_mesh(inserted.mesh, cuts)
    components = connected_components(range(topology.mesh.cell_count),
                                      topology.mesh.neighbors)
    assert len(components) == 1
    assert len(components[0]) == inserted.mesh.cell_count
    # Tip copies stay attached through their remaining intact material fan;
    # each internal support vertex separates into exactly two banks.
    copies = np.bincount(topology.parent_vertex,
                        minlength=inserted.mesh.vertex_count)
    for vertex, count in degree.items():
        assert copies[vertex] == count
    return cuts


@pytest.mark.parametrize("subdivisions", [2, 3, 4])
@pytest.mark.parametrize("angle", [.137, .619, 1.113])
def test_oblique_path_is_exact_and_does_not_detach_mesh_chips(subdivisions, angle):
    mesh = build_icosphere(subdivisions)
    path = _oblique_path(angle)
    inserted = insert_crack_path(mesh, path)
    _assert_partition(mesh, inserted)
    _assert_exact_support(inserted, path)
    _assert_open_cut(inserted, CrackInterval(0., path.length_m))
    # The endpoint is inside material, far from any old mesh vertex: an edge
    # staircase or nearest-vertex approximation cannot satisfy this test.
    assert np.min(_angle(mesh.vertices, path.points_xyz[0])) > 1e-4
    assert inserted.mesh.vertex_count > mesh.vertex_count


@pytest.mark.parametrize("subdivisions", [2, 4])
def test_bent_support_and_both_tips_inside_one_material_triangle(subdivisions):
    mesh = build_icosphere(subdivisions)
    triangle = mesh.vertices[mesh.faces[17]]
    barycentric = np.array([[.62, .23, .15], [.28, .49, .23],
                            [.23, .31, .46], [.12, .61, .27]])
    path = ReferenceCrackPath(_unit(barycentric@triangle), RADIUS_KM)
    inserted = insert_crack_path(mesh, path)
    _assert_partition(mesh, inserted)
    _assert_exact_support(inserted, path)
    _assert_open_cut(inserted, CrackInterval(0., path.length_m))
    # All bends must be explicit vertices; omitting one would replace its two
    # minor arcs by a different diagonal through the material cell.
    for point in path.points_xyz:
        assert np.min(_angle(inserted.mesh.vertices[inserted.path_vertex_ids], point)) < 1e-12


@pytest.mark.parametrize("subdivisions", [2, 3, 4])
def test_existing_great_circle_edge_chain_is_reused_without_geometric_offset(subdivisions):
    mesh = build_icosphere(subdivisions)
    original = build_icosphere(0)
    path = ReferenceCrackPath(original.vertices[[0, 1]], RADIUS_KM)
    inserted = insert_crack_path(mesh, path)
    _assert_partition(mesh, inserted)
    _assert_exact_support(inserted, path)
    _assert_open_cut(inserted, CrackInterval(0., path.length_m))
    assert inserted.mesh.vertex_count == mesh.vertex_count
    assert inserted.mesh.cell_count == mesh.cell_count
    assert len(inserted.cut_edges) == 2**subdivisions


def test_preinserted_fronts_make_nested_cuts_without_changing_material_mesh():
    mesh = build_icosphere(3)
    path = _oblique_path()
    fronts = path.length_m*np.asarray([.17, .43, .59, .86])
    inserted = insert_crack_path(mesh, path, front_coordinates_m=fronts)
    before_vertices = inserted.mesh.vertices.copy()
    before_faces = inserted.mesh.faces.copy()
    _assert_partition(mesh, inserted)
    _assert_exact_support(inserted, path)
    old = set()
    for end in fronts[1:]:
        interval = CrackInterval(fronts[0], end)
        cuts = _assert_open_cut(inserted, interval)
        active = set(map(tuple, cuts))
        assert old < active
        old = active
        assert sum(_angle(inserted.mesh.vertices[a], inserted.mesh.vertices[b])
                   for a, b in cuts)*RADIUS_KM*1000. == pytest.approx(
                       interval.length_m, rel=1e-11)
    np.testing.assert_array_equal(inserted.mesh.vertices, before_vertices)
    np.testing.assert_array_equal(inserted.mesh.faces, before_faces)


@pytest.mark.parametrize("subdivisions", [2, 4])
def test_oblique_path_through_existing_vertex_has_no_duplicate_or_spur(subdivisions):
    mesh = build_icosphere(subdivisions)
    center = mesh.vertices[10]
    tangent = _unit(np.cross(center, [.187, -.461, .929]))
    angles = np.array([-.253, .627])
    points = np.cos(angles[:, None])*center + np.sin(angles[:, None])*tangent
    path = ReferenceCrackPath(_unit(points), RADIUS_KM)
    inserted = insert_crack_path(mesh, path)
    _assert_partition(mesh, inserted)
    _assert_exact_support(inserted, path)
    _assert_open_cut(inserted, CrackInterval(0., path.length_m))
    distances = _angle(inserted.mesh.vertices, center)
    assert np.count_nonzero(distances < 1e-12) == 1
    assert np.min(_angle(inserted.mesh.vertices[inserted.path_vertex_ids], center)) < 1e-12


def test_resolved_submetre_near_vertex_pass_is_not_snapped_to_old_vertex():
    mesh = build_icosphere(3)
    center = mesh.vertices[10]
    tangent = _unit(np.cross(center, [.187, -.461, .929]))
    offset = 1e-7  # 0.53 m on this sphere, versus hundreds of km per cell.
    shifted = _unit(center + offset*np.cross(center, tangent))
    angles = np.array([-.253, .627])
    points = np.cos(angles[:, None])*shifted + np.sin(angles[:, None])*tangent
    path = ReferenceCrackPath(_unit(points), RADIUS_KM)
    inserted = insert_crack_path(mesh, path)
    _assert_partition(mesh, inserted)
    _assert_exact_support(inserted, path)
    _assert_open_cut(inserted, CrackInterval(0., path.length_m))
    assert 10 not in inserted.path_vertex_ids
    closest = np.min(_angle(inserted.mesh.vertices[inserted.path_vertex_ids], center))
    assert closest >= .999*offset


def test_oblique_insertion_commutes_with_rigid_rotation():
    mesh = build_icosphere(3)
    path = _oblique_path(.619)
    rotation = Rotation.from_rotvec([.37, -.91, .28]).as_matrix()
    rotated = SphereMesh(mesh.vertices@rotation.T, mesh.faces.copy(),
                         mesh.centroids@rotation.T, mesh.areas_unit_sphere.copy(),
                         mesh.neighbors, mesh.shared_edges)
    rotated_path = ReferenceCrackPath(path.points_xyz@rotation.T, RADIUS_KM)
    first = insert_crack_path(mesh, path)
    second = insert_crack_path(rotated, rotated_path)
    _assert_partition(rotated, second)
    _assert_exact_support(second, rotated_path)
    np.testing.assert_array_equal(second.parent_face, first.parent_face)
    np.testing.assert_array_equal(second.mesh.faces, first.mesh.faces)
    np.testing.assert_array_equal(second.path_vertex_ids, first.path_vertex_ids)
    np.testing.assert_allclose(second.mesh.vertices, first.mesh.vertices@rotation.T,
                               rtol=0, atol=3e-12)
    np.testing.assert_allclose(second.path_arclength_m, first.path_arclength_m,
                               rtol=3e-12, atol=2e-7)
