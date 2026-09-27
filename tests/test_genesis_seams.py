"""Cut topology preserves material cells while separating displacement banks."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tectonics.genesis_material import material_column_depth, material_layer_mass
from tectonics.genesis_seams import rebuild_seam_mesh, split_mesh
from tectonics.mesh import build_icosphere, connected_components


@pytest.fixture(scope="module")
def mesh():
    return build_icosphere(1)


def _patch_boundary(mesh, patch):
    return [(u, v) for fa, fb, u, v in mesh.shared_edges if (fa in patch) != (fb in patch)]


def _opened_triangle(topology, face=0, scale=.8):
    vertices = topology.mesh.vertices.copy()
    indices = topology.mesh.faces[face]
    center = topology.mesh.centroids[face]
    vertices[indices] = center + scale * (vertices[indices] - center)
    vertices[indices] /= np.linalg.norm(vertices[indices], axis=1)[:, None]
    return vertices


def test_no_cut_preserves_original_indices_topology_and_geometry_exactly(mesh):
    result = split_mesh(mesh, [])
    np.testing.assert_array_equal(result.mesh.vertices, mesh.vertices)
    np.testing.assert_array_equal(result.mesh.faces, mesh.faces)
    np.testing.assert_array_equal(result.parent_vertex, np.arange(mesh.vertex_count))
    np.testing.assert_array_equal(result.mesh.centroids, mesh.centroids)
    np.testing.assert_array_equal(result.mesh.areas_unit_sphere, mesh.areas_unit_sphere)
    assert result.mesh.shared_edges == mesh.shared_edges
    assert result.mesh.neighbors == mesh.neighbors
    assert result.original_shared_edges == result.intact_shared_edges == mesh.shared_edges
    assert result.cut_edges.shape == (0, 2)
    assert result.seam_faces.shape == (0, 2)
    assert result.bank_vertices.shape == (0, 2, 2)
    assert result.seam_count == 0
    assert result.original_vertex_count == mesh.vertex_count


def test_single_cut_edge_retains_both_shared_crack_tips(mesh):
    fa, fb, u, v = mesh.shared_edges[8]
    result = split_mesh(mesh, [(v, u)])
    assert result.mesh.vertex_count == mesh.vertex_count
    np.testing.assert_array_equal(result.mesh.faces, mesh.faces)
    np.testing.assert_array_equal(result.bank_vertices[0], [[u, v], [u, v]])
    np.testing.assert_array_equal(result.seam_faces[0], [fa, fb])
    assert fb not in result.mesh.neighbors[fa]
    assert fa not in result.mesh.neighbors[fb]
    assert len(result.mesh.shared_edges) == len(mesh.shared_edges) - 1
    assert len(connected_components(range(mesh.cell_count), result.mesh.neighbors)) == 1


def test_two_edge_crack_duplicates_internal_vertex_but_keeps_endpoints(mesh):
    u, center, v = map(int, mesh.faces[0])
    result = split_mesh(mesh, [(u, center), (center, v)])
    assert result.mesh.vertex_count == mesh.vertex_count + 1
    assert np.count_nonzero(result.parent_vertex == center) == 2
    assert np.count_nonzero(result.parent_vertex == u) == 1
    assert np.count_nonzero(result.parent_vertex == v) == 1
    for edge, banks in zip(result.cut_edges, result.bank_vertices):
        interior = int(np.flatnonzero(edge == center)[0])
        assert banks[0, interior] != banks[1, interior]
        assert banks[0, 1 - interior] == banks[1, 1 - interior]


@pytest.mark.parametrize("patch", [{0}, {0, 3}])
def test_closed_cut_loop_separates_patch_and_duplicates_all_boundary_vertices(mesh, patch):
    result = split_mesh(mesh, _patch_boundary(mesh, patch))
    boundary_vertices = np.unique(result.cut_edges)
    assert result.mesh.vertex_count == mesh.vertex_count + len(boundary_vertices)
    components = {frozenset(c) for c in connected_components(range(mesh.cell_count), result.mesh.neighbors)}
    assert frozenset(patch) in components
    assert len(components) == 2
    assert np.all(result.bank_vertices[:, 0] != result.bank_vertices[:, 1])
    np.testing.assert_array_equal(result.parent_vertex[result.mesh.faces], mesh.faces)
    for edge, faces, banks in zip(result.cut_edges, result.seam_faces, result.bank_vertices):
        np.testing.assert_array_equal(result.parent_vertex[banks], np.broadcast_to(edge, banks.shape))
        for face, endpoints in zip(faces, banks):
            assert set(endpoints).issubset(result.mesh.faces[face])
    for fa, fb, u, v in result.mesh.shared_edges:
        assert {u, v}.issubset(result.mesh.faces[fa])
        assert {u, v}.issubset(result.mesh.faces[fb])
        assert fb in result.mesh.neighbors[fa] and fa in result.mesh.neighbors[fb]


def test_cut_order_and_endpoint_order_do_not_change_bank_identity(mesh):
    cuts = np.asarray(_patch_boundary(mesh, {0, 3}))
    forward = split_mesh(mesh, cuts)
    reverse = split_mesh(mesh, cuts[::-1, ::-1])
    for name in ("cut_edges", "seam_faces", "bank_vertices", "parent_vertex", "original_faces"):
        np.testing.assert_array_equal(getattr(forward, name), getattr(reverse, name))
    np.testing.assert_array_equal(forward.mesh.faces, reverse.mesh.faces)


def test_all_edges_cut_yields_independent_triangles(mesh):
    result = split_mesh(mesh, [(u, v) for _, _, u, v in mesh.shared_edges])
    assert result.mesh.vertex_count == 3 * mesh.cell_count
    assert len(np.unique(result.mesh.faces)) == 3 * mesh.cell_count
    assert result.mesh.shared_edges == ()
    assert result.intact_shared_edges == ()
    assert all(neighbors == () for neighbors in result.mesh.neighbors)
    np.testing.assert_array_equal(result.parent_vertex[result.mesh.faces], mesh.faces)
    np.testing.assert_array_equal(result.mesh.vertices[result.mesh.faces], mesh.vertices[mesh.faces])


def test_separation_preserves_face_mass_and_energy_then_opening_changes_only_depth(mesh):
    result = split_mesh(mesh, _patch_boundary(mesh, {0}))
    mass = material_layer_mass(mesh, 5287., 3000., 40., 8)
    split_mass = material_layer_mass(result.mesh, 5287., 3000., 40., 8)
    np.testing.assert_array_equal(split_mass, mass)
    # A material column can already have different layer masses and enthalpy;
    # neither quantity is remapped when its geometric footprint changes.
    mass *= np.linspace(.8, 1.2, 8)[None, :]
    enthalpy = np.random.default_rng(12).uniform(1e6, 2e6, mass.shape)
    saved_mass, saved_enthalpy = mass.copy(), enthalpy.copy()
    moved = rebuild_seam_mesh(result, _opened_triangle(result))
    depth = material_column_depth(moved, 5287., mass, 3000.)
    assert depth[0] > 40.
    np.testing.assert_allclose(depth[1:], 40., atol=4e-14)
    recovered_mass = moved.physical_cell_areas_km2(5287.) * depth * 3000. * 1e9
    np.testing.assert_allclose(recovered_mass, mass.sum(axis=1), rtol=3e-16)
    np.testing.assert_array_equal(mass, saved_mass)
    np.testing.assert_array_equal(enthalpy, saved_enthalpy)
    np.testing.assert_array_equal(np.sum(mass * enthalpy, axis=1), np.sum(saved_mass * saved_enthalpy, axis=1))


@pytest.mark.parametrize("scale", [.8, 1.1])
def test_open_geometry_does_not_require_four_pi_coverage(mesh, scale):
    result = split_mesh(mesh, _patch_boundary(mesh, {0}))
    moved = rebuild_seam_mesh(result, _opened_triangle(result, scale=scale))
    area_difference = moved.areas_unit_sphere.sum() - 4 * np.pi
    assert np.sign(area_difference) == np.sign(scale - 1.)
    assert abs(area_difference) > .001
    assert moved.neighbors is result.mesh.neighbors
    assert moved.shared_edges is result.mesh.shared_edges
    np.testing.assert_array_equal(moved.faces, result.mesh.faces)
    np.testing.assert_allclose(moved.areas_unit_sphere[1:], mesh.areas_unit_sphere[1:], atol=1e-16)


def test_rotating_the_globe_rotates_banks_without_changing_topology_or_areas(mesh):
    rotation = Rotation.from_rotvec([.9, -.4, .3]).as_matrix()
    cuts = _patch_boundary(mesh, {0, 3})
    result = split_mesh(mesh, cuts)
    rotated_source = rebuild_seam_mesh(mesh, mesh.vertices @ rotation.T)
    rotated = split_mesh(rotated_source, cuts)
    np.testing.assert_array_equal(result.mesh.faces, rotated.mesh.faces)
    np.testing.assert_array_equal(result.parent_vertex, rotated.parent_vertex)
    np.testing.assert_array_equal(result.bank_vertices, rotated.bank_vertices)
    np.testing.assert_allclose(rotated.mesh.vertices, result.mesh.vertices @ rotation.T, atol=2e-16)
    np.testing.assert_allclose(rotated.mesh.areas_unit_sphere, result.mesh.areas_unit_sphere, atol=2e-16)
    moved = rebuild_seam_mesh(result, _opened_triangle(result))
    rotated_moved = rebuild_seam_mesh(rotated, moved.vertices @ rotation.T)
    np.testing.assert_allclose(rotated_moved.areas_unit_sphere, moved.areas_unit_sphere, atol=2e-16)


@pytest.mark.parametrize("cuts", [
    [[0., 1.]], [[True, False]], [0, 1], [[-1, 2]], [[0, 0]],
    [[0, 99999]], np.empty((0, 3)),
])
def test_invalid_cut_format_and_nonexistent_edges_are_rejected(mesh, cuts):
    with pytest.raises(ValueError):
        split_mesh(mesh, cuts)


def test_duplicate_cuts_are_rejected_even_with_reversed_endpoints(mesh):
    _, _, u, v = mesh.shared_edges[0]
    with pytest.raises(ValueError, match="unique"):
        split_mesh(mesh, [(u, v), (v, u)])


def test_open_source_is_rejected(mesh):
    with pytest.raises(ValueError, match="closed"):
        split_mesh(replace(mesh, faces=mesh.faces[1:]), [])


def test_duplicate_source_faces_are_rejected(mesh):
    with pytest.raises(ValueError, match="duplicate faces"):
        split_mesh(replace(mesh, faces=np.vstack((mesh.faces, mesh.faces[0]))), [])


def test_three_faces_sharing_a_source_edge_are_rejected(mesh):
    a, b, c = mesh.faces[0]
    nonmanifold = replace(mesh,
                          vertices=np.vstack((mesh.vertices, mesh.vertices[c])),
                          faces=np.vstack((mesh.faces, [a, b, mesh.vertex_count])))
    with pytest.raises(ValueError, match="two-face-per-edge"):
        split_mesh(nonmanifold, [])


@pytest.mark.parametrize("kind", ["areas", "centers", "shape"])
def test_stale_cached_source_geometry_is_rejected(mesh, kind):
    areas, centers = mesh.areas_unit_sphere.copy(), mesh.centroids.copy()
    if kind == "areas":
        areas[0] *= 2.
    elif kind == "centers":
        centers[0] = np.nan
    else:
        areas = areas[:-1]
    with pytest.raises(ValueError, match="geometry is inconsistent"):
        split_mesh(replace(mesh, areas_unit_sphere=areas, centroids=centers), [])


def test_nonmanifold_source_vertex_fan_is_rejected():
    mesh = build_icosphere(0)
    second_vertices = np.arange(mesh.vertex_count) + mesh.vertex_count - 1
    second_vertices[0] = 0
    bow_tie = replace(mesh,
                      vertices=np.vstack((mesh.vertices, mesh.vertices[1:])),
                      faces=np.vstack((mesh.faces, second_vertices[mesh.faces])))
    with pytest.raises(ValueError, match="nonmanifold vertex fan"):
        split_mesh(bow_tie, [])


@pytest.mark.parametrize("kind", ["float_faces", "out_of_range", "unused_vertex", "repeated_vertex", "reversed_face"])
def test_invalid_source_face_connectivity_is_rejected(mesh, kind):
    faces, vertices = mesh.faces.copy(), mesh.vertices.copy()
    if kind == "float_faces":
        faces = faces.astype(float)
    elif kind == "out_of_range":
        faces[0, 0] = len(vertices)
    elif kind == "unused_vertex":
        vertices = np.vstack((vertices, vertices[0]))
    elif kind == "repeated_vertex":
        faces[0, 0] = faces[0, 1]
    else:
        faces[0] = faces[0, ::-1]
    with pytest.raises(ValueError):
        split_mesh(replace(mesh, faces=faces, vertices=vertices), [])


@pytest.mark.parametrize("kind", ["nan", "nonunit", "shape", "inverted", "collapsed", "long_edge"])
def test_invalid_open_geometry_is_rejected(mesh, kind):
    result = split_mesh(mesh, _patch_boundary(mesh, {0}))
    vertices = result.mesh.vertices.copy()
    a, b, c = result.mesh.faces[0]
    if kind == "nan":
        vertices[a, 0] = np.nan
    elif kind == "nonunit":
        vertices[a] *= 1.1
    elif kind == "shape":
        vertices = vertices[:-1]
    elif kind == "inverted":
        vertices[[a, b]] = vertices[[b, a]]
    elif kind == "collapsed":
        vertices[a] = vertices[b]
    else:
        vertices[a] = -vertices[b]
    with pytest.raises(ValueError):
        rebuild_seam_mesh(result, vertices)


def test_split_does_not_mutate_or_alias_source_material_geometry(mesh):
    faces, vertices = mesh.faces.copy(), mesh.vertices.copy()
    result = split_mesh(mesh, _patch_boundary(mesh, {0}))
    result.mesh.vertices[:] = 0.
    result.mesh.faces[:] = 0
    result.original_faces[:] = 0
    np.testing.assert_array_equal(mesh.faces, faces)
    np.testing.assert_array_equal(mesh.vertices, vertices)
