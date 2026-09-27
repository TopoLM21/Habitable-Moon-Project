"""Material identity and parent mechanics under moving path observations."""
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import norm as sparse_norm
from scipy.spatial.transform import Rotation

from tectonics.genesis_crack_path import ReferenceCrackPath
from tectonics.genesis_material import rebuild_material_mesh
from tectonics.genesis_moving_support import MaterialPathSupport
from tectonics.genesis_path_mesh import insert_crack_path
from tectonics.mesh import build_icosphere


def _source():
    mesh = build_icosphere(1)
    points = np.array([[1., .12, .23], [1., .42, .49], [1., .72, .53]])
    points /= np.linalg.norm(points, axis=1)[:, None]
    path = ReferenceCrackPath(points, 5300.)
    inserted = insert_crack_path(mesh, path,
        front_coordinates_m=path.length_m * np.array([.2, .5, .8]))
    return mesh, inserted


@pytest.fixture(scope="module")
def support():
    parent, insertion = _source()
    return MaterialPathSupport(parent, insertion, 5.3e6, .25)


def _deformed(mesh):
    x, y, z = mesh.vertices.T
    vertices = mesh.vertices + .07 * np.column_stack((x*y, y*z, z*x))
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    return rebuild_material_mesh(mesh, vertices)


def _relative(a, b):
    norm = sparse_norm if sparse.issparse(a) else np.linalg.norm
    return norm(a-b) / max(norm(a), norm(b), 1e-300)


def test_rigid_rotation_and_radius_preserve_material_support(support):
    matrix = Rotation.from_rotvec([.37, -.52, .19]).as_matrix()
    moved = rebuild_material_mesh(support.parent_mesh, support.parent_mesh.vertices @ matrix.T)
    radius = 1.03 * support.radius_m
    basis = support.basis_at(moved, radius)
    inserted = basis.insertion
    np.testing.assert_allclose(inserted.mesh.vertices,
        support.insertion.mesh.vertices @ matrix.T, rtol=0, atol=8e-16)
    np.testing.assert_array_equal(inserted.path.points_xyz,
        inserted.mesh.vertices[inserted.path_vertex_ids])
    assert len(inserted.path.points_xyz) > len(support.insertion.path.points_xyz)
    assert inserted.path.length_m == pytest.approx(
        1.03 * support.insertion.path.length_m, rel=3e-14)
    np.testing.assert_allclose(inserted.path_arclength_m,
        1.03 * support.insertion.path_arclength_m, rtol=3e-14, atol=3e-9)
    assert basis.radius_m == radius
    np.testing.assert_array_equal(basis.free_dofs(), np.arange(basis.nparent))


def test_nonuniform_motion_retains_every_material_identity(support):
    moved = _deformed(support.parent_mesh)
    basis = support.basis_at(moved, support.radius_m)
    old, current = support.insertion, basis.insertion
    for name in ("parent_face", "path_vertex_ids", "vertex_parent_face", "vertex_barycentric"):
        np.testing.assert_array_equal(getattr(old, name), getattr(current, name))
    np.testing.assert_array_equal(old.mesh.faces, current.mesh.faces)
    np.testing.assert_array_equal(old.cut_edges, current.cut_edges)
    corners = moved.vertices[moved.faces[old.vertex_parent_face]]
    expected = np.einsum("vi,vij->vj", old.vertex_barycentric, corners)
    expected /= np.linalg.norm(expected, axis=1)[:, None]
    np.testing.assert_allclose(current.mesh.vertices, expected, rtol=0, atol=3e-16)
    np.testing.assert_array_equal(current.path.points_xyz,
        current.mesh.vertices[current.path_vertex_ids])
    np.testing.assert_array_equal(current.path_arclength_m, current.path.arclength_m)
    assert current.path.length_m != old.path.length_m


def test_fixed_mass_shares_do_not_follow_current_child_areas(support):
    masses = np.random.default_rng(912).uniform(1e10, 1e20,
        (support.parent_mesh.cell_count, 3))
    before = support.extensive(masses)
    moved = _deformed(support.parent_mesh)
    basis = support.basis_at(moved, .99 * support.radius_m)
    np.testing.assert_array_equal(support.extensive(masses), before)
    assert np.max(np.abs(basis.subdivision.area_fraction - support.area_fraction)) > 1e-5
    assert _relative(basis.subdivision.extensive(masses), before) > 1e-5
    summed = np.zeros_like(masses)
    np.add.at(summed, support.parent_face, before)
    np.testing.assert_allclose(summed, masses, rtol=3e-16, atol=0)


@pytest.mark.parametrize("geometry", ["initial", "rotated", "nonuniform"])
def test_tied_parent_stiffness_force_and_energy_with_fixed_volume(support, geometry):
    mesh = support.parent_mesh
    if geometry == "rotated":
        rotation = Rotation.from_rotvec([-.4, .3, .21]).as_matrix()
        mesh = rebuild_material_mesh(mesh, mesh.vertices @ rotation.T)
    elif geometry == "nonuniform":
        mesh = _deformed(mesh)
    basis = support.basis_at(mesh, .97 * support.radius_m)
    rng = np.random.default_rng(622)
    # Fixed material amounts do not get recomputed from current child areas.
    volume = support.parent_mesh.areas_unit_sphere * support.radius_m**2 * rng.uniform(
        8e3, 15e3, mesh.cell_count)
    membrane = basis.parent_membrane
    elasticity = 6e10 * rng.uniform(.1, 1, mesh.cell_count)[:, None, None] * membrane.d
    memory = rng.normal(0, 1e-5, (mesh.cell_count, 3))
    child_volume = support.extensive(volume)
    child_c = basis.subdivision.intensive(elasticity)
    child_memory = basis.subdivision.tensor(memory, engineering=True)
    k, g = basis.bulk(child_volume, child_c, child_memory)
    local = np.einsum("fai,fab,fbj,f->fij", membrane.b, elasticity, membrane.b,
        volume / basis.radius_m**2)
    expected_k = sparse.coo_matrix((local.ravel(), (membrane.rr, membrane.cc)),
        shape=(membrane.ndof, membrane.ndof)).tocsr()
    local_g = np.einsum("fai,fab,fb,f->fi", membrane.b, elasticity, memory,
        volume / basis.radius_m)
    expected_g = np.zeros(membrane.ndof)
    np.add.at(expected_g, membrane.dofs.ravel(), local_g.ravel())
    assert _relative(k[:basis.nparent, :basis.nparent], expected_k) < 2e-14
    assert _relative(g[:basis.nparent], expected_g) < 2e-14
    energy = .5 * np.einsum("fi,fij,fj,f->", memory, elasticity, memory, volume)
    child_energy = .5 * np.einsum("fi,fij,fj,f->", child_memory, child_c, child_memory, child_volume)
    assert child_energy == pytest.approx(energy, rel=2e-14)


def test_observations_are_independent_of_previous_observation_geometry(support):
    moved = _deformed(support.parent_mesh)
    first = support.basis_at(moved, support.radius_m)
    support.basis_at(support.parent_mesh, .93 * support.radius_m)
    repeated = support.basis_at(moved, support.radius_m)
    np.testing.assert_array_equal(first.insertion.mesh.vertices, repeated.insertion.mesh.vertices)
    np.testing.assert_array_equal(first.insertion.path_arclength_m, repeated.insertion.path_arclength_m)
    np.testing.assert_array_equal(first.strain_operator.data, repeated.strain_operator.data)


def test_input_mutations_do_not_change_owned_ancestry_or_amounts():
    parent, insertion = _source()
    support = MaterialPathSupport(parent, insertion, 5.3e6, .25)
    owned_vertices = support.parent_mesh.vertices.copy()
    owned_weights = support.insertion.vertex_barycentric.copy()
    mass = np.arange(parent.cell_count, dtype=float) + 1
    expected = support.extensive(mass)
    parent.vertices[:] = 0
    parent.faces[:] = 0
    insertion.mesh.vertices[:] = 0
    insertion.vertex_barycentric[:] = 0
    insertion.parent_face[:] = 0
    np.testing.assert_array_equal(support.parent_mesh.vertices, owned_vertices)
    np.testing.assert_array_equal(support.insertion.vertex_barycentric, owned_weights)
    np.testing.assert_array_equal(support.extensive(mass), expected)
    support.basis_at(support.parent_mesh, support.radius_m)


@pytest.mark.parametrize("name", ["parent_face", "area_fraction"])
def test_initial_shares_have_immutable_byte_storage(support, name):
    value = getattr(support, name)
    with pytest.raises(ValueError):
        value.setflags(write=True)


def test_stored_mesh_and_ancestry_arrays_are_immutable(support):
    arrays = [getattr(mesh, name) for mesh in (support.parent_mesh, support.insertion.mesh)
        for name in ("vertices", "faces", "centroids", "areas_unit_sphere")]
    arrays += [getattr(support.insertion, name) for name in
        ("parent_face", "path_vertex_ids", "path_arclength_m", "vertex_parent_face", "vertex_barycentric")]
    for array in arrays:
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize("change", ["faces", "neighbors", "edges", "resolution"])
def test_changed_parent_topology_is_rejected(support, change):
    mesh = support.parent_mesh
    if change == "faces":
        mesh = replace(mesh, faces=mesh.faces[::-1].copy())
    elif change == "neighbors":
        mesh = replace(mesh, neighbors=tuple(reversed(mesh.neighbors)))
    elif change == "edges":
        mesh = replace(mesh, shared_edges=tuple(reversed(mesh.shared_edges)))
    else:
        mesh = build_icosphere(2)
    with pytest.raises(ValueError, match="topology"):
        support.basis_at(mesh, support.radius_m)


@pytest.mark.parametrize("radius", [0., -1., np.inf, np.nan, True, None])
def test_invalid_current_radius_rejected(support, radius):
    with pytest.raises(ValueError, match="radius"):
        support.basis_at(support.parent_mesh, radius)


@pytest.mark.parametrize("value", [3., np.zeros(4), np.array([None]), np.array([1+2j])])
def test_invalid_material_amounts_rejected(support, value):
    with pytest.raises(ValueError, match="amounts"):
        support.extensive(value)


def test_nonfinite_geometry_rejected_without_changing_support(support):
    vertices = support.parent_mesh.vertices.copy()
    vertices[0, 0] = np.nan
    original = support.insertion.mesh.vertices.copy()
    with pytest.raises(ValueError, match="finite"):
        support.basis_at(replace(support.parent_mesh, vertices=vertices), support.radius_m)
    np.testing.assert_array_equal(support.insertion.mesh.vertices, original)
