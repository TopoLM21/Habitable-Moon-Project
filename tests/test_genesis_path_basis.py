"""Mechanical contracts for tied support and explicit relative-bank motion."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import norm as sparse_norm
from scipy.spatial.transform import Rotation

from tectonics.genesis_contact import ContactModel
from tectonics.genesis_crack_path import CrackInterval, ReferenceCrackPath
from tectonics.genesis_material import rebuild_material_mesh
from tectonics.genesis_path_basis import EmbeddedPathBasis, tangent_prolongation
from tectonics.genesis_path_mesh import insert_crack_path
from tectonics.mesh import build_icosphere


def _fixture(level=1):
    parent = build_icosphere(level)
    points = np.array([[1., .12, .23], [1., .42, .49], [1., .72, .53]])
    points /= np.linalg.norm(points, axis=1)[:, None]
    path = ReferenceCrackPath(points, 5300.)
    inserted = insert_crack_path(parent, path,
        front_coordinates_m=path.length_m*np.array([.2, .5, .8]))
    return EmbeddedPathBasis(parent, inserted, 5.3e6, .25)


@pytest.fixture(scope="module")
def basis():
    return _fixture()


def _relative(a, b):
    norm = sparse_norm if sparse.issparse(a) else np.linalg.norm
    return float(norm(a-b)/max(norm(a), norm(b), 1e-300))


def _contact_geometry(basis):
    geometry = object.__new__(ContactModel)
    geometry.topology = basis.topology
    geometry.membrane = basis.membrane
    geometry.radius_m = basis.radius_m
    geometry.depth_m = np.full(basis.topology.mesh.cell_count, 1e4)
    geometry.source_state = SimpleNamespace(water_access=np.zeros(basis.topology.mesh.cell_count))
    geometry._build_interfaces(basis.subdivision.mesh)
    return geometry


def _material(basis):
    parent = basis.subdivision.parent_mesh
    rng = np.random.default_rng(481)
    volume = parent.areas_unit_sphere*basis.radius_m**2*rng.uniform(8e3, 15e3, parent.cell_count)
    elasticity = 6e10*rng.uniform(.1, 1, parent.cell_count)[:, None, None]*basis.parent_membrane.d
    memory = rng.normal(0, 1e-5, (parent.cell_count, 3))
    return volume, elasticity, memory


def test_locked_basis_reproduces_parent_stiffness_prestress_and_energy(basis):
    volume, elasticity, memory = _material(basis)
    child_volume = basis.subdivision.extensive(volume)
    child_c = basis.subdivision.intensive(elasticity)
    child_memory = basis.subdivision.tensor(memory, engineering=True)
    matrix, force = basis.bulk(child_volume, child_c, child_memory)
    m = basis.parent_membrane
    local = np.einsum("fai,fab,fbj,f->fij", m.b, elasticity, m.b,
                      volume/basis.radius_m**2)
    expected_k = sparse.coo_matrix((local.ravel(), (m.rr, m.cc)),
                                   shape=(m.ndof, m.ndof)).tocsr()
    local_force = np.einsum("fai,fab,fb,f->fi", m.b, elasticity, memory,
                            volume/basis.radius_m)
    expected_g = np.zeros(m.ndof)
    np.add.at(expected_g, m.dofs.ravel(), local_force.ravel())
    assert _relative(matrix[:basis.nparent, :basis.nparent], expected_k) < 2e-14
    assert _relative(force[:basis.nparent], expected_g) < 2e-14
    before = .5*np.einsum("fi,fij,fj,f->", memory, elasticity, memory, volume)
    after = .5*np.einsum("fi,fij,fj,f->", child_memory, child_c, child_memory, child_volume)
    assert abs(after-before)/before < 2e-14


def test_rigid_rotations_and_radial_patch_are_preserved(basis):
    original = basis.subdivision.parent_mesh
    refined = basis.topology.mesh
    for axis in np.eye(3):
        z = np.zeros(basis.ndof)
        world = basis.radius_m*np.cross(axis, original.vertices)
        z[:basis.nparent-1] = np.einsum("vij,vi->vj",
            basis.parent_membrane.vertex_basis, world).ravel()
        q = basis.displacement_operator@z
        expected = basis.radius_m*np.cross(axis, refined.vertices)
        actual = np.einsum("vij,vj->vi", basis.membrane.vertex_basis, q[:-1].reshape(-1, 2))
        assert _relative(actual, expected) < 2e-14
        assert np.max(np.abs(basis.strain(z))) < 2e-14
        assert np.max(np.abs(basis.geometric_strain(z))) < 2e-12
    z = np.zeros(basis.ndof)
    z[basis.nparent-1] = 13.
    expected = np.broadcast_to(np.array([13., 13., 0.])/basis.radius_m,
                               (refined.cell_count, 3))
    np.testing.assert_allclose(basis.strain(z), expected, rtol=1e-14, atol=1e-20)
    np.testing.assert_allclose(basis.geometric_strain(z), expected, rtol=0, atol=1e-20)


def test_contact_jump_has_no_background_and_independent_enrichment(basis):
    j = _contact_geometry(basis).jump_operator
    assert sparse_norm(j@basis.parent_displacement_operator) < 1e-14
    enrichment = (j@basis.W).toarray()
    assert np.linalg.matrix_rank(enrichment) == basis.ndof-basis.nparent
    assert np.linalg.matrix_rank(basis.displacement_operator.toarray()) == basis.ndof


def test_drag_is_positive_orthogonal_and_parent_background_unchanged(basis):
    fine = sparse.diags(basis.fine_drag_area_m2)
    cross = basis.parent_displacement_operator.T@fine@basis.W
    scale = max(sparse_norm(fine@basis.W), 1.)
    assert sparse_norm(cross)/scale < 3e-16
    relative = basis.W.T@fine@basis.W
    expected = sparse.diags(basis.drag_area_m2[basis.nparent:])
    assert _relative(relative, expected) < 3e-16
    assert np.all(basis.drag_area_m2[basis.nparent:] > 0)
    assert basis.drag_area_m2[basis.nparent-1] == 0
    m = basis.subdivision.parent_mesh
    expected_parent = np.zeros(m.vertex_count)
    np.add.at(expected_parent, m.faces.ravel(), np.repeat(m.areas_unit_sphere/3, 3))
    np.testing.assert_array_equal(basis.drag_area_m2[:basis.nparent-1],
                                 np.repeat(expected_parent*basis.radius_m**2, 2))
    rng = np.random.default_rng(331)
    z = rng.normal(size=basis.ndof)
    dz = z[basis.nparent:]
    relative_velocity = basis.W@dz
    assert np.dot(basis.drag_area_m2[basis.nparent:], dz**2) == pytest.approx(
        np.dot(basis.fine_drag_area_m2, relative_velocity**2), rel=1e-15)


def test_bulk_tangent_and_force_are_energy_derivatives(basis):
    volume, elasticity, memory = _material(basis)
    v = basis.subdivision.extensive(volume)
    c = basis.subdivision.intensive(elasticity)
    e = basis.subdivision.tensor(memory, engineering=True)
    k, g = basis.bulk(v, c, e)
    rng = np.random.default_rng(612)
    z, direction = rng.normal(size=(2, basis.ndof))
    def energy(q):
        inc = basis.strain(q)
        # Reduced potential avoids cancellation against inherited memory energy.
        return float(np.einsum("fi,fij,fj,f->", inc, c, e+.5*inc, v))
    step = 1e-3
    derivative = (energy(z+step*direction)-energy(z-step*direction))/(2*step)
    assert derivative == pytest.approx(float(direction@(g+k@z)), rel=1e-10)
    delta_force = k@(z+step*direction)+g-(k@(z-step*direction)+g)
    assert _relative(delta_force/(2*step), k@direction) < 3e-11
    assert _relative(k, k.T) < 1e-14


def test_assumed_and_geometric_strains_are_explicitly_different(basis):
    z = np.zeros(basis.ndof)
    z[:basis.nparent] = np.random.default_rng(735).normal(size=basis.nparent)
    assert _relative(basis.strain(z), basis.geometric_strain(z)) > 1e-5
    z[:basis.nparent] = 0.
    z[basis.nparent:] = np.random.default_rng(2).normal(size=basis.ndof-basis.nparent)
    np.testing.assert_allclose(basis.strain(z), basis.geometric_strain(z), rtol=1e-14, atol=1e-20)


def test_unconstrained_bulk_has_only_three_rigid_rotation_modes():
    small = _fixture(0)
    volume = small.subdivision.mesh.areas_unit_sphere*small.radius_m**2*1e4
    c = np.broadcast_to(small.membrane.d*6e10, (len(volume), 3, 3))
    k, _ = small.bulk(volume, c, np.zeros((len(volume), 3)))
    scale = 1/np.sqrt(k.diagonal())
    eigenvalues = np.linalg.eigvalsh(k.toarray()*scale[:, None]*scale[None, :])
    assert np.max(np.abs(eigenvalues[:3])) < 1e-12
    assert eigenvalues[3] > 1e-5


def test_active_intervals_leave_tips_tied_and_spaces_nested(basis):
    path = basis.insertion.path
    np.testing.assert_array_equal(basis.free_dofs(), np.arange(basis.nparent))
    full = basis.free_dofs(CrackInterval(0., path.length_m))
    np.testing.assert_array_equal(full, np.arange(basis.ndof))
    small = basis.free_dofs(CrackInterval(.2*path.length_m, .5*path.length_m))
    larger = basis.free_dofs(CrackInterval(.2*path.length_m, .8*path.length_m))
    assert set(small).issubset(larger)
    assert set(larger) < set(full)
    coords = dict(zip(basis.insertion.path_vertex_ids, basis.insertion.path_arclength_m))
    allowed = [i for i, vertex in enumerate(basis.enrichment_vertices)
               if .2*path.length_m < coords[vertex] < .5*path.length_m]
    expected = np.r_[np.arange(basis.nparent),
        (basis.nparent+2*np.asarray(allowed)[:, None]+np.arange(2)).ravel()]
    np.testing.assert_array_equal(small, expected)
    with pytest.raises(ValueError, match="not preinserted"):
        basis.free_dofs(CrackInterval(.2*path.length_m+1., .8*path.length_m))


def test_single_edge_seed_has_no_independent_bank_motion(basis):
    arc = basis.insertion.path_arclength_m
    interval = CrackInterval(float(arc[1]), float(arc[2]))
    np.testing.assert_array_equal(basis.free_dofs(interval), basis.free_dofs())


def test_support_of_one_existing_edge_has_no_enrichment():
    parent = build_icosphere(0)
    path = ReferenceCrackPath(parent.vertices[[0, 1]], 5300.)
    insertion = insert_crack_path(parent, path)
    basis = EmbeddedPathBasis(parent, insertion, 5.3e6, .25)
    assert basis.ndof == basis.nparent
    assert basis.W.shape == (basis.membrane.ndof, 0)
    assert not len(basis.enrichment_vertices)
    np.testing.assert_array_equal(basis.displacement_operator.toarray(), np.eye(basis.nparent))
    np.testing.assert_array_equal(basis.free_dofs(CrackInterval(0, path.length_m)), basis.free_dofs())


def test_external_work_preserves_background_and_explicit_enrichment(basis):
    rng = np.random.default_rng(142)
    parent = rng.normal(size=basis.nparent)
    fine = rng.normal(size=basis.membrane.ndof)
    z = rng.normal(size=basis.ndof)
    force = basis.external(parent, fine)
    np.testing.assert_array_equal(force[:basis.nparent], parent)
    expected = parent@z[:basis.nparent]+fine@(basis.W@z[basis.nparent:])
    assert force@z == pytest.approx(expected, rel=1e-14)


def test_joint_world_rotation_preserves_both_strains_and_work(basis):
    rotation = Rotation.from_rotvec([.41, -.51, .17]).as_matrix()
    original = basis.subdivision.parent_mesh
    inserted = basis.insertion
    rotated_parent = rebuild_material_mesh(original, original.vertices@rotation.T)
    rotated_child = rebuild_material_mesh(inserted.mesh, inserted.mesh.vertices@rotation.T)
    rotated_path = ReferenceCrackPath(inserted.path.points_xyz@rotation.T, inserted.path.radius_km)
    rotated_insertion = replace(inserted, mesh=rotated_child, path=rotated_path,
        path_arclength_m=inserted.path_arclength_m*rotated_path.length_m/inserted.path.length_m)
    other = EmbeddedPathBasis(rotated_parent, rotated_insertion, basis.radius_m, .25)
    rng = np.random.default_rng(41)
    z = rng.normal(size=basis.ndof)
    rotated_z = z.copy()
    old_vectors = np.einsum("vij,vj->vi", basis.parent_membrane.vertex_basis,
                           z[:basis.nparent-1].reshape(-1, 2))
    rotated_z[:basis.nparent-1] = np.einsum("vij,vi->vj", other.parent_membrane.vertex_basis,
                                                        old_vectors@rotation.T).ravel()
    first_bank = basis.enrichment_vertices
    old_relative = np.einsum("vij,vj->vi", basis.membrane.vertex_basis[first_bank],
                            z[basis.nparent:].reshape(-1, 2))
    rotated_z[basis.nparent:] = np.einsum("vij,vi->vj", other.membrane.vertex_basis[first_bank],
                                                        old_relative@rotation.T).ravel()
    np.testing.assert_allclose(other.strain(rotated_z), basis.strain(z), rtol=2e-12, atol=2e-18)
    np.testing.assert_allclose(other.geometric_strain(rotated_z), basis.geometric_strain(z), rtol=2e-12, atol=2e-18)
    assert np.dot(other.drag_area_m2, rotated_z**2) == pytest.approx(
        np.dot(basis.drag_area_m2, z**2), rel=3e-14)


@pytest.mark.parametrize("bad", [None, np.nan, True, -1., 0.])
def test_invalid_radius_rejected(basis, bad):
    with pytest.raises(ValueError):
        EmbeddedPathBasis(basis.subdivision.parent_mesh, basis.insertion, bad, .25)


@pytest.mark.parametrize("bad", [np.nan, True, -1., .5, 1.])
def test_invalid_poisson_ratio_rejected(basis, bad):
    with pytest.raises(ValueError):
        EmbeddedPathBasis(basis.subdivision.parent_mesh, basis.insertion, basis.radius_m, bad)


@pytest.mark.parametrize("field", ["volume", "elasticity", "memory", "displacement", "parent_force", "fine_force"])
@pytest.mark.parametrize("problem", ["shape", "nan", "complex"])
def test_invalid_method_arrays_rejected(basis, field, problem):
    n = basis.topology.mesh.cell_count
    values = dict(volume=np.ones(n), elasticity=np.broadcast_to(np.eye(3), (n, 3, 3)).copy(),
                  memory=np.zeros((n, 3)), displacement=np.zeros(basis.ndof),
                  parent_force=np.zeros(basis.nparent), fine_force=np.zeros(basis.membrane.ndof))
    if problem == "shape":
        values[field] = values[field][:-1]
    elif problem == "nan":
        values[field].flat[0] = np.nan
    else:
        values[field] = values[field].astype(complex)
    with pytest.raises(ValueError):
        if field == "displacement":
            basis.strain(values[field])
        elif field in {"parent_force", "fine_force"}:
            basis.external(values["parent_force"], values["fine_force"])
        else:
            basis.bulk(values["volume"], values["elasticity"], values["memory"])


@pytest.mark.parametrize("problem", ["negative_volume", "zero_volume", "negative_modulus", "nonsymmetric"])
def test_invalid_constitutive_data_rejected(basis, problem):
    n = basis.topology.mesh.cell_count
    volume, c = np.ones(n), np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    if problem == "negative_volume":
        volume[0] = -1.
    elif problem == "zero_volume":
        volume[0] = 0.
    elif problem == "negative_modulus":
        c[0, 0, 0] = -1.
    else:
        c[0, 0, 1] = .2
    with pytest.raises(ValueError):
        basis.bulk(volume, c, np.zeros((n, 3)))


def test_interpolation_public_analysis_api_is_preserved(basis):
    from analysis.genesis_path_mechanics_audit import tangent_prolongation as diagnostic
    assert diagnostic is tangent_prolongation
    matrix = tangent_prolongation(basis.subdivision.parent_mesh, basis.insertion)
    assert matrix.shape == (2*basis.insertion.mesh.vertex_count+1, basis.nparent)
