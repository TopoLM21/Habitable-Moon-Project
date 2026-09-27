"""Independent objectivity and finite-strain identities for geometry audit."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from analysis.genesis_path_geometry_audit import engineering, triangle_geometry
from tectonics.mesh import build_icosphere


def test_finite_rigid_rotation_has_zero_distortion_but_rotating_frame():
    mesh = build_icosphere(1)
    old = 5.3e6*mesh.vertices
    rotation = Rotation.from_rotvec([.3, -.4, .2]).as_matrix()
    new = old@rotation.T
    values = triangle_geometry(old, new, mesh.faces)
    np.testing.assert_allclose(values["stretch"], 0, atol=7e-16)
    np.testing.assert_allclose(values["green"], 0, atol=8e-16)
    np.testing.assert_allclose(values["rotation"],
                              np.broadcast_to(rotation, values["rotation"].shape), atol=1e-15)
    assert values["normal_rotation_rad"].max() > .5
    np.testing.assert_allclose(values["old_quality"], values["new_quality"], atol=5e-16)
    # Fixed-frame small strain falsely treats a finite rigid rotation as strain.
    frame = values["frame"]
    h = values["gradient"]-frame
    projected = np.einsum("fji,fjk->fik", frame, h)
    fixed_linear = .5*(projected+projected.transpose(0, 2, 1))
    assert np.max(np.abs(fixed_linear)) > .1


def test_rigid_rotation_requires_stress_transport_for_frame_objectivity():
    mesh = build_icosphere(1)
    old = mesh.vertices
    rotation = Rotation.from_rotvec([.4, -.2, .3]).as_matrix()
    values = triangle_geometry(old, old@rotation.T, mesh.faces)
    frame = values["frame"]
    stress = frame@np.diag([3., 1.])@frame.transpose(0, 2, 1)
    rotated_stress = values["rotation"]@stress@values["rotation"].transpose(0, 2, 1)
    expected = rotation@stress@rotation.T
    np.testing.assert_allclose(rotated_stress, expected, atol=3e-15)
    normal = frame[:, :, 0]
    new_normal = normal@rotation.T
    traction = np.einsum("fij,fj->fi", stress, normal)
    actual_traction = np.einsum("fij,fj->fi", rotated_stress, new_normal)
    frozen_traction = np.einsum("fij,fj->fi", stress, new_normal)
    np.testing.assert_allclose(actual_traction, traction@rotation.T, atol=4e-15)
    assert np.linalg.norm(frozen_traction-actual_traction) > 1


def test_stretch_is_invariant_under_superposed_rotation_and_translation():
    mesh = build_icosphere(1)
    old = mesh.vertices*2000
    stretch = np.diag([1.002, .999, 1.001])
    rotation = Rotation.from_rotvec([.6, .2, -.3]).as_matrix()
    new = old@stretch
    a = triangle_geometry(old, new, mesh.faces)
    b = triangle_geometry(old, new@rotation.T+[42., -17., 2.], mesh.faces)
    np.testing.assert_allclose(a["stretch"], b["stretch"], atol=6e-16)
    np.testing.assert_allclose(a["green"], b["green"], atol=6e-16)
    np.testing.assert_allclose(b["rotation"], rotation@a["rotation"], atol=1e-15)


def test_green_linear_error_is_exact_quadratic_gradient_term():
    old = np.array([[0., 0., 0.], [3., 0., 0.], [1., 2., 0.]])
    affine = np.array([[1.02, .03, 0], [-.01, .99, 0], [.02, .005, 1.]])
    values = triangle_geometry(old, old@affine.T, np.array([[0, 1, 2]]))
    h = values["gradient"]-values["frame"]
    projected = values["frame"].transpose(0, 2, 1)@h
    linear = .5*(projected+projected.transpose(0, 2, 1))
    quadratic = .5*h.transpose(0, 2, 1)@h
    np.testing.assert_allclose(values["green"]-linear, quadratic, atol=8e-17)
    # Out-of-plane motion contributes to Green strain even when the planar
    # symmetric gradient is zero; normals must be measured separately.
    assert values["normal_rotation_rad"][0] > .01


def test_uniform_expansion_known_green_strain():
    mesh = build_icosphere(1)
    values = triangle_geometry(mesh.vertices, 1.003*mesh.vertices, mesh.faces)
    np.testing.assert_allclose(values["stretch"], .003, atol=6e-16)
    expected = np.broadcast_to(np.array([.003+.5*.003**2, .003+.5*.003**2, 0]),
                               (mesh.cell_count, 3))
    np.testing.assert_allclose(engineering(values["green"]), expected, atol=6e-16)


@pytest.mark.parametrize("invalid", [np.array([[0, 0, 0]]), np.array([[0, 1, 1]])])
def test_collapsed_reference_rejected(invalid):
    positions = np.array([[0., 0., 0.], [1., 0., 0.]])
    with pytest.raises(ValueError, match="Collapsed"):
        triangle_geometry(positions, positions, invalid)


def test_collapsed_current_rejected():
    old = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    new = old.copy()
    new[2] = new[1]
    with pytest.raises(ValueError, match="Collapsed moved"):
        triangle_geometry(old, new, np.array([[0, 1, 2]]))
