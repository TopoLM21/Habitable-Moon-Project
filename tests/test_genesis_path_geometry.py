"""Independent local geometry checks for the opt-in fixed-reference guard."""
from dataclasses import FrozenInstanceError, fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from analysis.genesis_path_geometry_audit import audit, triangle_geometry
from tectonics.genesis_crack_path import ReferenceCrackPath
from tectonics.genesis_material import rebuild_material_mesh
from tectonics.genesis_path_basis import EmbeddedPathBasis
from tectonics.genesis_path_dynamics import PathMechanics
from tectonics.genesis_path_geometry import (
    PathGeometryParameters, _triangle_kinematics, local_geometry_metrics)
from tectonics.genesis_path_mesh import insert_crack_path
from tectonics.mesh import build_icosphere


def _model(rotation=None):
    parent = build_icosphere(1)
    points = np.array([[1., .12, .23], [1., .42, .49], [1., .72, .53]])
    points /= np.linalg.norm(points, axis=1)[:, None]
    if rotation is not None:
        parent = rebuild_material_mesh(parent, parent.vertices@rotation.T)
        points = points@rotation.T
    path = ReferenceCrackPath(points, 5300.)
    insertion = insert_crack_path(parent, path)
    basis = EmbeddedPathBasis(parent, insertion, 5.3e6, .25)
    return PathMechanics(basis, np.full(basis.topology.mesh.cell_count, 1e4))


@pytest.fixture(scope="module")
def model():
    return _model()


def _initial(model):
    return model.initial(np.zeros((model.basis.topology.mesh.cell_count, 3)))


def test_zero_motion_has_zero_metrics_and_changes_nothing(model):
    state = _initial(model)
    before = state.displacement_m.copy()
    vertices = model.basis.topology.mesh.vertices.copy()
    result = local_geometry_metrics(model, state)
    for key, value in result.items():
        assert abs(value) < 3e-13, key
    np.testing.assert_array_equal(before, state.displacement_m)
    np.testing.assert_array_equal(vertices, model.basis.topology.mesh.vertices)


def test_uniform_radial_expansion_has_known_nonlinear_error(model):
    state = _initial(model)
    strain = .004
    state.displacement_m[model.basis.nparent-1] = strain*model.basis.radius_m
    result = local_geometry_metrics(model, state)
    assert result["gradient_norm"] == pytest.approx(strain, abs=2e-13)
    assert result["finite_green_strain"] == pytest.approx(strain+.5*strain**2, abs=2e-13)
    assert result["linear_strain_error"] == pytest.approx(.5*strain**2, abs=2e-13)
    assert result["material_rotation_rad"] < 1e-13
    assert result["relative_contact_jump_error"] == 0


def test_independent_audit_matches_metrics_for_relative_bank_motion(model):
    state = _initial(model)
    rng = np.random.default_rng(876)
    state.displacement_m[:] = rng.normal(size=model.basis.ndof)*.2
    expected, _ = audit(model, state)
    result = local_geometry_metrics(model, state)
    assert result["linear_strain_error"] == pytest.approx(expected["max_finite_green_minus_linear_strain"], rel=1e-10, abs=2e-16)
    assert result["material_rotation_rad"] == pytest.approx(expected["maximum_polar_rotation_rad"], abs=2e-16)
    assert result["tangent_motion_radius_fraction"] == pytest.approx(expected["maximum_tangent_motion_radius_fraction"])
    assert result["normal_jump_error_m"] == pytest.approx(expected["max_corotated_gap_difference_m"], abs=1e-12)
    assert result["tangential_jump_error_m"] == pytest.approx(expected["max_corotated_slip_difference_m"], abs=1e-12)
    assert result["bank_frame_angular_mismatch"] <= 2*result["material_rotation_rad"]+1e-12


def test_finite_rigid_rotation_is_objective_but_requires_frame_control():
    mesh = build_icosphere(1)
    rotation = Rotation.from_rotvec([.3, -.2, .4]).as_matrix()
    values = _triangle_kinematics(mesh.vertices, mesh.vertices@rotation.T, mesh.faces)
    np.testing.assert_allclose(values["green"], 0, atol=1e-15)
    np.testing.assert_allclose(values["rotation"], np.broadcast_to(rotation, values["rotation"].shape), atol=1e-15)
    assert np.min(values["rotation_angle"]) > .5
    assert values["gradient_norm"].max() > .5


def test_triangle_metrics_are_covariant_under_coordinate_rotation():
    mesh = build_icosphere(1)
    rotation = Rotation.from_rotvec([.3, -.2, .4]).as_matrix()
    moved = mesh.vertices@np.array([[1.001, .002, 0], [0, .999, .001], [0, 0, 1.002]])
    values = _triangle_kinematics(mesh.vertices, moved, mesh.faces)
    other = _triangle_kinematics(mesh.vertices@rotation.T, moved@rotation.T, mesh.faces)
    np.testing.assert_allclose(other["green"], values["green"], atol=1e-15)
    np.testing.assert_allclose(other["gradient_norm"], values["gradient_norm"], atol=1e-15)
    np.testing.assert_allclose(other["rotation_angle"], values["rotation_angle"], atol=1e-15)
    np.testing.assert_allclose(other["rotation"], rotation@values["rotation"]@rotation.T, atol=2e-15)


def test_slender_triangle_detects_large_gradient_despite_tiny_motion_to_edge_ratio():
    old = np.array([[0., 0., 0.], [1000., 0., 0.], [500., .01, 0.]])
    new = old.copy()
    new[2, 1] += .001
    result = _triangle_kinematics(old, new, np.array([[0, 1, 2]]))
    assert .001/500 < 1e-5
    assert result["gradient_norm"][0] == pytest.approx(.1)
    assert result["gradient_norm"][0] > PathGeometryParameters().max_displacement_gradient


def test_triangle_green_strain_matches_independent_oracle():
    mesh = build_icosphere(1)
    moved = mesh.vertices@np.array([[1.003, .002, 0], [0, .999, .004], [0, 0, 1.001]])
    actual = _triangle_kinematics(mesh.vertices, moved, mesh.faces)
    expected = triangle_geometry(mesh.vertices, moved, mesh.faces)
    np.testing.assert_allclose(actual["green"], expected["green"], atol=1e-16)
    np.testing.assert_allclose(actual["rotation"], expected["rotation"], atol=1e-16)
    h = actual["gradient"]-actual["frame"]
    projected = actual["frame"].transpose(0, 2, 1)@h
    linear = .5*(projected+projected.transpose(0, 2, 1))
    np.testing.assert_allclose(actual["green"]-linear, .5*h.transpose(0, 2, 1)@h, atol=3e-16)


def test_whole_local_metric_pipeline_is_covariant_under_world_rotation(model):
    rotation = Rotation.from_rotvec([.2, -.4, .1]).as_matrix()
    other = _model(rotation)
    first, second = _initial(model), _initial(other)
    assert model.basis.ndof == other.basis.ndof
    # Pure parent infinitesimal rotation avoids any dependence on triangulation
    # tie-breaking or bank-copy ordering under rotation of the inserted support.
    axis = np.array([1e-5, -2e-5, 3e-5])
    for m, s, spin in ((model, first, axis), (other, second, rotation@axis)):
        world = m.basis.radius_m*np.cross(spin, m.basis.subdivision.parent_mesh.vertices)
        s.displacement_m[:m.basis.nparent-1] = np.einsum("vij,vi->vj",
            m.basis.parent_membrane.vertex_basis, world).ravel()
    a, b = local_geometry_metrics(model, first), local_geometry_metrics(other, second)
    for key in a:
        assert b[key] == pytest.approx(a[key], rel=3e-5, abs=1e-10), key


@pytest.mark.parametrize("value", [True, np.bool_(True), 0., -1., np.nan, np.inf, "1"])
@pytest.mark.parametrize("name", [item.name for item in fields(PathGeometryParameters)])
def test_geometry_parameters_reject_invalid_caps(name, value):
    with pytest.raises(ValueError, match="finite and positive"):
        replace(PathGeometryParameters(), **{name: value}).validate()


@pytest.mark.parametrize("name", [item.name for item in fields(PathGeometryParameters)])
def test_geometry_parameters_have_bounded_small_geometry_caps(name):
    ceiling = 5e-4 if name == "max_linear_strain_error" else .05
    replace(PathGeometryParameters(), **{name: ceiling}).validate()
    with pytest.raises(ValueError, match="experiment limit"):
        replace(PathGeometryParameters(), **{name: ceiling*1.0001}).validate()


def test_geometry_parameters_are_frozen():
    parameters = PathGeometryParameters()
    parameters.validate()
    with pytest.raises(FrozenInstanceError):
        parameters.max_linear_strain_error = .01


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_motion_is_rejected(model, value):
    state = _initial(model)
    state.displacement_m[0] = value
    with pytest.raises(ValueError, match="finite real"):
        local_geometry_metrics(model, state)


def test_collapsed_radius_is_rejected(model):
    state = _initial(model)
    state.displacement_m[model.basis.nparent-1] = -model.basis.radius_m
    with pytest.raises(ValueError, match="radius"):
        local_geometry_metrics(model, state)


def test_inverted_moved_mesh_is_rejected(model):
    state = _initial(model)
    # A local vertex moves past its neighboring face; spherical normalization
    # alone cannot detect the resulting inversion.
    state.displacement_m[:2] = model.basis.radius_m*np.array([10., 10.])
    with pytest.raises(ValueError):
        local_geometry_metrics(model, state)


def test_nonfinite_interface_axis_is_rejected(model):
    altered = SimpleNamespace(basis=model.basis, jump_operator=model.jump_operator,
        law_parameters=model.law_parameters,
        geometry=SimpleNamespace(interface_normal=model.geometry.interface_normal.copy(),
                                 interface_tangent=model.geometry.interface_tangent.copy()))
    altered.geometry.interface_normal[0, 0] = np.nan
    with pytest.raises(ValueError, match="Interface normals"):
        local_geometry_metrics(altered, _initial(model))
