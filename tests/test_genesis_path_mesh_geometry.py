"""Geometry contracts for prescribed supports, without a propagation law."""
import numpy as np
import pytest

from tectonics.genesis_crack_path import CrackInterval, ReferenceCrackPath
from tectonics.genesis_path_mesh import insert_crack_path
from tectonics.genesis_seams import split_mesh
from tectonics.mesh import build_icosphere, connected_components


def _path(points, radius=5300.):
    points = np.asarray(points, dtype=float)
    return ReferenceCrackPath(points/np.linalg.norm(points, axis=1)[:, None], radius)


@pytest.mark.parametrize("level", [0, 2, 4])
def test_subdivision_preserves_parents_and_interpolation_ancestry(level):
    mesh = build_icosphere(level)
    original_vertices, original_faces = mesh.vertices.copy(), mesh.faces.copy()
    path = _path([[1., .2, .1], [1., .3, .4], [1., .8, .5]])
    insertion = insert_crack_path(mesh, path)
    np.testing.assert_array_equal(mesh.vertices, original_vertices)
    np.testing.assert_array_equal(mesh.faces, original_faces)
    np.testing.assert_array_equal(insertion.mesh.vertices[:mesh.vertex_count], mesh.vertices)
    recovered = np.bincount(insertion.parent_face, insertion.mesh.areas_unit_sphere)
    np.testing.assert_allclose(recovered, mesh.areas_unit_sphere, rtol=3e-13, atol=0)
    parent_triangles = mesh.vertices[mesh.faces[insertion.vertex_parent_face]]
    reconstructed = np.einsum("ni,nij->nj", insertion.vertex_barycentric, parent_triangles)
    reconstructed /= np.linalg.norm(reconstructed, axis=1)[:, None]
    np.testing.assert_allclose(reconstructed, insertion.mesh.vertices, atol=5e-15, rtol=0)
    np.testing.assert_allclose(insertion.vertex_barycentric.sum(axis=1), 1., atol=5e-16)
    assert np.all(insertion.vertex_barycentric >= 0)


@pytest.mark.parametrize("level", [0, 2, 4])
def test_existing_mesh_edge_support_is_reused_without_refinement(level):
    mesh = build_icosphere(level)
    path = ReferenceCrackPath(mesh.vertices[[0, 1]], 5300.)
    inserted = insert_crack_path(mesh, path)
    assert inserted.mesh.vertex_count == mesh.vertex_count
    assert inserted.mesh.cell_count == mesh.cell_count
    assert len(inserted.cut_edges) == 2**level
    np.testing.assert_array_equal(inserted.mesh.faces, mesh.faces)
    split = split_mesh(inserted.mesh, inserted.cut_edges)
    assert len(connected_components(range(mesh.cell_count), split.mesh.neighbors)) == 1


def test_explicit_fronts_share_one_mesh_and_absent_front_is_not_snapped():
    mesh = build_icosphere(2)
    path = _path([[1., .12, .23], [1., .72, .53]])
    first, middle, last = path.length_m*np.array([.17, .53, .89])
    inserted = insert_crack_path(mesh, path, front_coordinates_m=[first, middle, last])
    small = inserted.cuts_for(CrackInterval(first, middle))
    large = inserted.cuts_for(CrackInterval(first, last))
    assert {tuple(edge) for edge in small} < {tuple(edge) for edge in large}
    np.testing.assert_array_equal(inserted.path_arclength_m[
        np.isin(inserted.path_arclength_m, [first, middle, last])], [first, middle, last])
    with pytest.raises(ValueError, match="not preinserted"):
        inserted.cuts_for(CrackInterval(first, last-1.))


def test_interior_bent_support_is_one_open_cut_with_joined_tips():
    mesh = build_icosphere(0)
    triangle = mesh.vertices[mesh.faces[0]]
    points = np.array([[.6, .2, .2], [.2, .5, .3], [.25, .2, .55]])@triangle
    inserted = insert_crack_path(mesh, _path(points))
    split = split_mesh(inserted.mesh, inserted.cut_edges)
    assert len(connected_components(range(inserted.mesh.cell_count), split.mesh.neighbors)) == 1
    assert len(inserted.cut_edges) == 2
    for tip in inserted.path_vertex_ids[[0, -1]]:
        assert np.count_nonzero(split.parent_vertex == tip) == 1
    assert np.count_nonzero(split.parent_vertex == inserted.path_vertex_ids[1]) == 2


def test_deterministic_insertion_and_permuted_front_list():
    mesh = build_icosphere(2)
    path = _path([[1., .2, .1], [1., .3, .4], [1., .8, .5]])
    front = path.length_m*np.array([.2, .7])
    a = insert_crack_path(mesh, path, front_coordinates_m=front)
    b = insert_crack_path(mesh, path, front_coordinates_m=front[::-1])
    for name in ("parent_face", "path_vertex_ids", "path_arclength_m",
                 "vertex_parent_face", "vertex_barycentric"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    np.testing.assert_array_equal(a.mesh.vertices, b.mesh.vertices)
    np.testing.assert_array_equal(a.mesh.faces, b.mesh.faces)


def test_fine_control_drops_only_topologically_collinear_qhull_exterior():
    # At 20480 cells Qhull emits a spurious projected triangle on old edge
    # (1661, 6546): its third vertex is an intersection on that same edge.
    # Its floating signed area, 1.57e-14, must not dictate a material cutoff.
    mesh = build_icosphere(5)
    center = np.array([.31, -.72, .62])
    center /= np.linalg.norm(center)
    tangent = np.cross(center, [.8, .1, .3])
    tangent /= np.linalg.norm(tangent)
    radius_m, length_m = 5.3e6, 2.5e6
    angle = length_m/(2*radius_m)
    path = ReferenceCrackPath(np.array([np.cos(angle)*center-np.sin(angle)*tangent,
                                       np.cos(angle)*center+np.sin(angle)*tangent]), radius_m/1000)
    inserted = insert_crack_path(mesh, path,
                                 front_coordinates_m=length_m*np.array([.25, .5, .75]))
    recovered = np.bincount(inserted.parent_face, inserted.mesh.areas_unit_sphere)
    np.testing.assert_allclose(recovered, mesh.areas_unit_sphere, rtol=3e-13, atol=0)
    triangles = inserted.mesh.vertices[inserted.mesh.faces]
    sides = np.roll(triangles, -1, axis=1)-triangles
    quality = (2*np.sqrt(3.)*np.linalg.norm(np.cross(sides[:, 0], -sides[:, 2]), axis=1)
               /np.sum(sides*sides, axis=(1, 2)))
    assert quality.min() > .05  # There was no actual near-degenerate child.
    assert inserted.mesh.cell_count == 20546
    split_mesh(inserted.mesh, inserted.cut_edges)


def test_true_unresolved_grazing_sliver_is_still_rejected():
    mesh = build_icosphere(3)
    center = mesh.vertices[10]
    tangent = np.cross(center, [.187, -.461, .929])
    tangent /= np.linalg.norm(tangent)
    shifted = center+1e-8*np.cross(center, tangent)
    shifted /= np.linalg.norm(shifted)
    angles = np.array([-.253, .627])
    points = np.cos(angles[:, None])*shifted+np.sin(angles[:, None])*tangent
    with pytest.raises(ValueError, match="unresolved planar sliver"):
        insert_crack_path(mesh, _path(points))


@pytest.mark.parametrize("coordinates", [[-1.], [np.nan], [[0.]], [True], [1j]])
def test_invalid_fronts_are_rejected(coordinates):
    mesh = build_icosphere(0)
    path = ReferenceCrackPath(mesh.vertices[[0, 1]], 5300.)
    with pytest.raises(ValueError, match="Prospective fronts"):
        insert_crack_path(mesh, path, front_coordinates_m=coordinates)
