"""Material geometry, frame objectivity, and column conservation invariants."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tectonics.genesis_material import (
    face_deformation, face_frames, geometry_diagnostics, material_column_depth,
    material_layer_mass, move_mesh, polar_increment, rebuild_material_mesh,
    rotate_tensor,
)
from tectonics.genesis_shell import Membrane
from tectonics.mesh import SphereMesh, build_icosphere


@pytest.fixture(scope="module")
def mesh():
    return build_icosphere(2)


def test_rigid_three_dimensional_rotation_preserves_geometry_mass_and_enthalpy(mesh):
    spatial_rotation = Rotation.from_rotvec([.9, -.4, .3]).as_matrix()
    rotated = rebuild_material_mesh(mesh, mesh.vertices @ spatial_rotation.T)
    np.testing.assert_allclose(rotated.areas_unit_sphere, mesh.areas_unit_sphere, rtol=2e-15)
    np.testing.assert_allclose(face_frames(rotated), np.einsum("ij,fjk->fik", spatial_rotation, face_frames(mesh)), atol=2e-15)
    deformation = face_deformation(mesh, rotated, 5287., 5287.)
    np.testing.assert_allclose(deformation, np.broadcast_to(np.eye(2), deformation.shape), atol=2e-15)
    rotation, strain = polar_increment(deformation)
    np.testing.assert_allclose(strain, 0., atol=2e-15)
    stress = np.random.default_rng(4).normal(size=(mesh.cell_count, 3))
    np.testing.assert_allclose(rotate_tensor(stress, rotation), stress, atol=5e-15)
    mass = material_layer_mass(mesh, 5287., 3000., 40., 32)
    enthalpy = np.random.default_rng(1).uniform(2e6, 3e6, mass.shape)
    depth = material_column_depth(rotated, 5287., mass, 3000.)
    recovered_mass = rotated.physical_cell_areas_km2(5287.)[:, None] * depth[:, None] * 1e9 * 3000. / 32
    assert np.sum(recovered_mass * enthalpy) == pytest.approx(np.sum(mass * enthalpy), rel=3e-16)
    assert rotated.areas_unit_sphere.sum() == pytest.approx(4 * np.pi, rel=2e-16)
    np.testing.assert_allclose(depth, 40., atol=8e-14)


def test_nonuniform_material_motion_changes_depth_without_changing_column_budgets(mesh):
    membrane = Membrane(mesh, .25)
    n = mesh.vertices
    velocity = .06 * (np.array([0., 0., 1.]) - n * n[:, 2:3])
    delta = np.einsum("vij,vi->vj", membrane.vertex_basis, velocity)
    moved = move_mesh(mesh, membrane.vertex_basis, delta)
    mass = material_layer_mass(mesh, 5287., 3000., 40., 32)
    saved_mass = mass.copy()
    enthalpy = np.random.default_rng(1).uniform(2e6, 3e6, mass.shape)
    original_energy = np.sum(mass * enthalpy, axis=1)
    depth = material_column_depth(moved, 5260., mass, 3000.)
    np.testing.assert_array_equal(mass, saved_mass)
    assert np.ptp(depth) > 5.
    recovered_mass = moved.physical_cell_areas_km2(5260.)[:, None] * depth[:, None] * 1e9 * 3000. / 32
    np.testing.assert_allclose(np.sum(recovered_mass * enthalpy, axis=1), original_energy, rtol=6e-16)
    assert np.sum(moved.physical_cell_areas_km2(5260.) * depth * 1e9 * 3000.) == pytest.approx(mass.sum(), rel=3e-16)
    assert moved.neighbors is mesh.neighbors and moved.shared_edges is mesh.shared_edges
    np.testing.assert_array_equal(moved.faces, mesh.faces)
    diagnostics = geometry_diagnostics(moved, mesh, 5260., 5287.)
    assert diagnostics["min_area_ratio"] < 1 < diagnostics["max_area_ratio"]
    assert .9 < diagnostics["min_face_quality"] <= 1
    assert abs(diagnostics["total_area_relative_residual"]) < 3e-16


def test_uniform_radius_contraction_has_exact_isotropic_hencky_strain(mesh):
    contraction = .94
    deformation = face_deformation(mesh, mesh, 5287., 5287. * contraction)
    rotation, strain = polar_increment(deformation)
    np.testing.assert_allclose(rotation, np.broadcast_to(np.eye(2), rotation.shape), atol=5e-16)
    np.testing.assert_allclose(strain[:, :2], np.log(contraction), atol=5e-16)
    np.testing.assert_allclose(strain[:, 2], 0., atol=2e-16)
    mass = material_layer_mass(mesh, 5287., 3000., 40., 32)
    np.testing.assert_allclose(material_column_depth(mesh, 5287. * contraction, mass, 3000.), 40. / contraction**2, rtol=4e-16)


def test_material_deformation_matches_membrane_at_small_increment(mesh):
    membrane = Membrane(mesh, .25)
    delta = np.random.default_rng(21).normal(0., 1e-8, (mesh.vertex_count, 2))
    radial_increment = 3e-8
    moved = move_mesh(mesh, membrane.vertex_basis, delta)
    _, measured = polar_increment(face_deformation(mesh, moved, 5287., 5287. * (1 + radial_increment)))
    dofs = np.r_[delta.ravel(), radial_increment]
    expected = np.einsum("fai,fi->fa", membrane.b, dofs[membrane.dofs])
    # The finite chord update and logarithmic strain converge to the existing
    # membrane B operator; differences are quadratic in the increment.
    np.testing.assert_allclose(measured, expected, rtol=0., atol=8e-14)


def test_polar_decomposition_uses_right_hencky_and_engineering_shear():
    angle = .6
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    directions = np.array([[.8, -.6], [.6, .8]])
    stretches = np.array([1.12, .93])
    stretch_tensor = (directions * stretches) @ directions.T
    expected_log = (directions * np.log(stretches)) @ directions.T
    calculated_rotation, hencky = polar_increment(rotation @ stretch_tensor)
    np.testing.assert_allclose(calculated_rotation, rotation, atol=3e-16)
    np.testing.assert_allclose(hencky, [expected_log[0, 0], expected_log[1, 1], 2 * expected_log[0, 1]], atol=3e-16)


@pytest.mark.parametrize("engineering", [False, True])
def test_tensor_rotation_preserves_eigenvalues_and_inverse(engineering):
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    source = np.array([7., 2., 3.])
    transformed = rotate_tensor(source, rotation, engineering=engineering)
    shear_scale = 2 if engineering else 1
    old_tensor = np.array([[source[0], source[2] / shear_scale], [source[2] / shear_scale, source[1]]])
    new_tensor = np.array([[transformed[0], transformed[2] / shear_scale], [transformed[2] / shear_scale, transformed[1]]])
    np.testing.assert_allclose(np.linalg.eigvalsh(new_tensor), np.linalg.eigvalsh(old_tensor), atol=3e-15)
    np.testing.assert_allclose(rotate_tensor(transformed, rotation.T, engineering=engineering), source, atol=4e-15)


@pytest.mark.parametrize("case", ["inverted", "collapsed", "nonunit", "nonfinite", "large_edge"])
def test_invalid_material_mesh_is_rejected(mesh, case):
    vertices = mesh.vertices.copy()
    a, b, c = mesh.faces[0]
    if case == "inverted":
        vertices[[a, b]] = vertices[[b, a]]
    elif case == "collapsed":
        vertices[a] = vertices[b]
    elif case == "nonunit":
        vertices[a] *= 1.01
    elif case == "nonfinite":
        vertices[a, 0] = np.nan
    else:
        vertices[a] = -vertices[b]
    with pytest.raises(ValueError):
        rebuild_material_mesh(mesh, vertices)


def test_incomplete_spherical_cover_is_rejected(mesh):
    # A topology with one face removed must not silently lose material area.
    incomplete = SphereMesh(mesh.vertices, mesh.faces[:-1], mesh.centroids[:-1],
                            mesh.areas_unit_sphere[:-1], mesh.neighbors, mesh.shared_edges)
    with pytest.raises(ValueError, match="4 pi"):
        rebuild_material_mesh(incomplete, mesh.vertices)


@pytest.mark.parametrize("deformation", [np.zeros((2, 2)), np.diag([1., -1.]), np.diag([1., 1e-15]), np.full((2, 2), np.nan)])
def test_nonphysical_deformation_is_rejected(deformation):
    with pytest.raises(ValueError):
        polar_increment(deformation)


def test_mass_and_movement_input_guards(mesh):
    with pytest.raises(ValueError, match="positive"):
        material_layer_mass(mesh, 0., 3000., 40., 32)
    with pytest.raises(ValueError, match="integer"):
        material_layer_mass(mesh, 5287., 3000., 40., 2.5)
    with pytest.raises(ValueError, match="positive"):
        material_column_depth(mesh, 5287., np.zeros((mesh.cell_count, 4)), 3000.)
    basis = Membrane(mesh, .25).vertex_basis
    with pytest.raises(ValueError, match="orthonormal"):
        move_mesh(mesh, basis * 2, np.zeros((mesh.vertex_count, 2)))
    with pytest.raises(ValueError, match="tangent"):
        move_mesh(mesh, np.stack((mesh.vertices, basis[:, :, 0]), axis=2), np.zeros((mesh.vertex_count, 2)))
