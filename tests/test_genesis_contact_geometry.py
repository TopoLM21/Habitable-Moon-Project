"""Physical trace geometry independent of saved fault-model configuration."""
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.genesis_contact import ContactModel
from tectonics.genesis_contact_geometry import build_interface_geometry
from tectonics.genesis_crack_path import ReferenceCrackPath
from tectonics.genesis_path_mesh import insert_crack_path
from tectonics.genesis_seams import split_mesh
from tectonics.genesis_shell import Membrane
from tectonics.mesh import build_icosphere
from test_genesis_contact import _source_bytes


def _geometry(mesh, cuts, depth=None):
    topo = split_mesh(mesh, cuts)
    membrane = Membrane(topo.mesh, .25)
    depth = np.full(mesh.cell_count, 1e4) if depth is None else depth
    geometry = build_interface_geometry(mesh, topo, 5.3e6, depth, membrane)
    return topo, membrane, geometry


@pytest.fixture(scope="module")
def mesh():
    return build_icosphere(1)


def test_empty_and_shared_tip_traces_have_no_artificial_displacement_mode(mesh):
    _, membrane, empty = _geometry(mesh, [])
    assert empty.jump_operator.shape == (0, membrane.ndof)
    assert empty.interface_area_m2.shape == empty.contact_rows.shape == (0,)
    assert empty.interface_normal.shape == empty.interface_tangent.shape == (0, 3)
    _, _, tip = _geometry(mesh, [mesh.shared_edges[0][2:]])
    assert tip.jump_operator.shape[0] == 4
    assert tip.jump_operator.nnz == 0
    assert tip.interface_area_m2.sum() > 0


def test_jump_measures_bank_difference_with_positive_opening_and_work_conjugacy(mesh):
    a, b, c = mesh.faces[0]
    topo, membrane, geometry = _geometry(mesh, [(a, b), (b, c)])
    random = np.random.default_rng(513)
    q = random.normal(size=membrane.ndof)
    physical = np.einsum("vij,vj->vi", membrane.vertex_basis, q[:-1].reshape(-1, 2))
    banks = physical[topo.bank_vertices]
    delta = (banks[:, 1]-banks[:, 0]).reshape(-1, 3)
    expected = np.column_stack((np.sum(delta*geometry.interface_normal, axis=1),
                                np.sum(delta*geometry.interface_tangent, axis=1)))
    np.testing.assert_allclose((geometry.jump_operator@q).reshape(-1, 2), expected,
                               rtol=2e-15, atol=2e-15)
    direction = mesh.centroids[topo.seam_faces[:, 1]]-mesh.centroids[topo.seam_faces[:, 0]]
    assert np.all(np.sum(geometry.interface_normal[::2]*direction, axis=1) > 0)
    traction = random.normal(size=expected.shape)
    force = geometry.jump_operator.T@(traction*geometry.interface_area_m2[:, None]).ravel()
    assert np.dot(q, force) == pytest.approx(np.sum(expected*traction*geometry.interface_area_m2[:, None]), rel=3e-15)
    assert geometry.jump_operator[:, -1].nnz == 0


def test_continuous_copied_material_displacement_has_zero_jump_to_roundoff(mesh):
    topo, membrane, geometry = _geometry(mesh, [edge[2:] for edge in mesh.shared_edges])
    values = np.random.default_rng(8).normal(size=(mesh.vertex_count, 2))
    q = np.r_[values[topo.parent_vertex].ravel(), 12.]
    np.testing.assert_allclose(geometry.jump_operator@q, 0., rtol=0.,
                               atol=4*np.finfo(float).eps*np.max(np.abs(values)))


def test_reference_area_counts_two_quadrature_endpoints_once(mesh):
    depth = np.linspace(2e3, 12e3, mesh.cell_count)
    topo, _, geometry = _geometry(mesh, [edge[2:] for edge in mesh.shared_edges], depth)
    a, b = mesh.vertices[topo.cut_edges].transpose(1, 0, 2)
    # Independent chord-to-angle conversion, with the radius in physical m.
    expected_length = 2*np.arcsin(np.linalg.norm(a-b, axis=1)/2)*5.3e6
    np.testing.assert_allclose(geometry.edge_length_m, expected_length, rtol=3e-16)
    expected_area = expected_length*np.minimum(depth[topo.seam_faces[:, 0]], depth[topo.seam_faces[:, 1]])
    np.testing.assert_allclose(geometry.interface_area_m2.reshape(-1, 2).sum(axis=1), expected_area, rtol=4e-16)
    np.testing.assert_allclose(np.linalg.norm(geometry.interface_normal, axis=1), 1., atol=2e-16)
    np.testing.assert_allclose(np.linalg.norm(geometry.interface_tangent, axis=1), 1., atol=2e-16)
    np.testing.assert_allclose(np.sum(geometry.interface_normal*geometry.interface_tangent, axis=1), 0., atol=2e-16)


def test_arbitrary_inserted_child_mesh_needs_no_checkpoint(mesh):
    points = np.array([[1., .2, .1], [1., .3, .4], [1., .8, .5]])
    points /= np.linalg.norm(points, axis=1)[:, None]
    path = ReferenceCrackPath(points, 5300.)
    insertion = insert_crack_path(mesh, path)
    assert insertion.mesh.cell_count > mesh.cell_count
    _, _, geometry = _geometry(insertion.mesh, insertion.cut_edges)
    assert geometry.edge_length_m.sum() == pytest.approx(path.length_m, rel=2e-14)
    assert geometry.interface_area_m2.sum() == pytest.approx(path.length_m*1e4, rel=2e-14)
    assert geometry.jump_operator.nnz > 0


def test_checkpoint_model_uses_identical_geometry_helper(tmp_path):
    data, *_ = _source_bytes(tmp_path)
    model = ContactModel(data)
    original = model.source_model.mesh_for(model.source_state)
    geometry = build_interface_geometry(original, model.topology, model.radius_m,
                                         model.depth_m, model.membrane)
    for item in fields(geometry):
        expected, actual = getattr(geometry, item.name), getattr(model, item.name)
        if item.name == "jump_operator":
            for array in ("data", "indices", "indptr"):
                np.testing.assert_array_equal(getattr(actual, array), getattr(expected, array))
        else:
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("radius", [0., -1., np.nan, np.inf, True])
def test_invalid_radius_rejected(mesh, radius):
    topo = split_mesh(mesh, [])
    with pytest.raises(ValueError, match="radius"):
        build_interface_geometry(mesh, topo, radius, np.ones(mesh.cell_count), Membrane(topo.mesh, .25))


@pytest.mark.parametrize("bad", [0., -1., np.nan, np.inf])
def test_invalid_solid_depth_rejected(mesh, bad):
    topo = split_mesh(mesh, [])
    depth = np.ones(mesh.cell_count)
    depth[0] = bad
    with pytest.raises(ValueError, match="solid depth"):
        build_interface_geometry(mesh, topo, 5.3e6, depth, Membrane(topo.mesh, .25))


def test_incompatible_reference_and_basis_are_rejected(mesh):
    topo = split_mesh(mesh, [])
    membrane = Membrane(topo.mesh, .25)
    with pytest.raises(ValueError, match="solid depth"):
        build_interface_geometry(mesh, topo, 5.3e6, [1.], membrane)
    other = build_icosphere(0)
    with pytest.raises(ValueError, match="reference mesh"):
        build_interface_geometry(other, topo, 5.3e6, np.ones(other.cell_count), membrane)
    bad_basis = SimpleNamespace(vertex_basis=np.zeros((1, 3, 2)), ndof=membrane.ndof)
    with pytest.raises(ValueError, match="vertex DOFs"):
        build_interface_geometry(mesh, topo, 5.3e6, np.ones(mesh.cell_count), bad_basis)
