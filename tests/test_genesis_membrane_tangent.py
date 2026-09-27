"""Virtual-work and compatibility checks for constitutive Newton corrections."""
import numpy as np
import pytest

from tectonics.genesis_shell import Membrane
from tectonics.mesh import build_icosphere


@pytest.fixture
def membrane():
    return Membrane(build_icosphere(1), 0.25)


def test_isotropic_correction_matches_eigenstrain_solve(membrane):
    mesh = membrane.mesh
    young = 6e10
    rng = np.random.default_rng(142)
    stress = rng.normal(size=(mesh.cell_count, 3)) * 3e6
    thickness = 2 + mesh.centroids[:, 0]
    stiffness = 0.6 + 0.3 * mesh.centroids[:, 1]
    traction = rng.normal(size=(mesh.vertex_count, 3)) * 1000
    eigen = -np.einsum("ab,fb->fa", np.linalg.inv(membrane.d), stress)
    eigen /= (young * stiffness)[:, None]
    expected, _, expected_residual, expected_radial = membrane.solve(
        eigen, thickness, stiffness, young, traction, 5287.)
    expected_displacement = membrane.last_displacement_rad.copy()
    tangent = young * stiffness[:, None, None] * membrane.d[None, :, :]
    actual, residual, radial = membrane.solve_correction(
        stress, thickness, tangent, young, traction, 5287.)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-15)
    np.testing.assert_allclose(membrane.last_displacement_rad,
                               expected_displacement, rtol=2e-12, atol=1e-15)
    assert radial == pytest.approx(expected_radial, rel=2e-12, abs=1e-15)
    assert residual < 1e-11
    assert expected_residual < 1e-11


def test_zero_stress_has_no_correction(membrane):
    count = membrane.mesh.cell_count
    strain, residual, radial = membrane.solve_correction(
        np.zeros((count, 3)), np.ones(count),
        np.broadcast_to(6e10 * membrane.d, (count, 3, 3)), 6e10)
    np.testing.assert_array_equal(strain, 0.)
    np.testing.assert_array_equal(membrane.last_displacement_rad, 0.)
    assert residual == 0.
    assert radial == 0.


def test_uniform_stress_relaxes_through_free_radius(membrane):
    mesh = membrane.mesh
    count = mesh.cell_count
    young = 6e10
    stress = np.zeros((count, 3))
    stress[:, :2] = 4e6
    tangent = np.broadcast_to(young * membrane.d, (count, 3, 3))
    strain, residual, radial = membrane.solve_correction(
        stress, 2 + mesh.centroids[:, 0], tangent, young)
    expected = -4e6 * (1 - 0.25) / young
    np.testing.assert_allclose(strain[:, :2], expected, rtol=0, atol=1e-16)
    np.testing.assert_allclose(strain[:, 2], 0., rtol=0, atol=1e-16)
    np.testing.assert_allclose(membrane.last_displacement_rad, 0., rtol=0, atol=1e-16)
    assert radial == pytest.approx(expected, abs=1e-16)
    assert residual < 1e-11


def test_already_balanced_stress_has_no_correction(membrane):
    mesh = membrane.mesh
    count = mesh.cell_count
    young = 6e10
    depth = 2 + mesh.centroids[:, 0]
    stiffness = 0.6 + 0.3 * mesh.centroids[:, 1]
    eigen = np.zeros((count, 3))
    eigen[:, :2] = (1e-3 * mesh.centroids[:, 2] ** 2)[:, None]
    traction = 1000 * np.cross(mesh.vertices, [0.5, 0.2, 1.])
    _, stress, _, _ = membrane.solve(eigen, depth, stiffness, young, traction, 5287.)
    tangent = young * stiffness[:, None, None] * membrane.d[None, :, :]
    correction, _, radial = membrane.solve_correction(
        stress, depth, tangent, young, traction, 5287.)
    np.testing.assert_allclose(correction, 0., rtol=0, atol=2e-15)
    np.testing.assert_allclose(membrane.last_displacement_rad, 0., rtol=0, atol=2e-15)
    assert abs(radial) < 2e-15


def test_nonsymmetric_tangent_balances_virtual_work_without_rotation(membrane):
    mesh = membrane.mesh
    count = mesh.cell_count
    young = 6e10
    radius = 5287.
    rng = np.random.default_rng(145)
    stress = rng.normal(size=(count, 3)) * 2e6
    depth = 3 + mesh.centroids[:, 0]
    tangent = young * np.broadcast_to(membrane.d, (count, 3, 3)).copy()
    tangent[:, 2, 0] += young * (0.2 + 0.08 * mesh.centroids[:, 1])
    tangent[:, 2, 1] += young * (0.1 + 0.04 * mesh.centroids[:, 2])
    traction = rng.normal(size=(mesh.vertex_count, 3)) * 500
    inputs = [value.copy() for value in (stress, depth, tangent, traction)]
    strain, residual, radial = membrane.solve_correction(
        stress, depth, tangent, young, traction, radius)
    displacement = np.r_[membrane.last_displacement_rad.ravel(), radial]

    # Independently assemble virtual work in dense element loops. Checking the
    # unsymmetric operator here catches accidental tangent symmetrization.
    matrix = np.zeros((membrane.ndof, membrane.ndof))
    force = np.zeros(membrane.ndof)
    for face in range(count):
        dofs = membrane.dofs[face]
        b = membrane.b[face]
        area = mesh.areas_unit_sphere[face]
        matrix[np.ix_(dofs, dofs)] += (
            area * depth[face] * b.T @ tangent[face] @ b / young)
        force[dofs] -= area * depth[face] * (b.T @ stress[face]) / young
    external = np.zeros(membrane.ndof)
    for face, vertices in enumerate(mesh.faces):
        for vertex in vertices:
            external[2 * vertex:2 * vertex + 2] += (
                mesh.areas_unit_sphere[face] / 3 * radius / young
                * membrane.vertex_basis[vertex].T @ traction[vertex])
    constraints = membrane.constraints.toarray()
    external -= constraints.T @ np.linalg.solve(
        constraints @ constraints.T, constraints @ external)
    force += external
    assert np.linalg.norm(matrix @ displacement - force) < 1e-11 * np.linalg.norm(force)
    np.testing.assert_allclose(constraints @ displacement, 0., rtol=0, atol=1e-15)
    expected_strain = np.einsum("fai,fi->fa", membrane.b, displacement[membrane.dofs])
    np.testing.assert_allclose(strain, expected_strain, rtol=0, atol=1e-16)
    assert residual < 1e-11
    for value, original in zip((stress, depth, tangent, traction), inputs):
        np.testing.assert_array_equal(value, original)
    symmetric = (tangent + tangent.transpose(0, 2, 1)) / 2
    symmetric_strain, _, _ = membrane.solve_correction(
        stress, depth, symmetric, young, traction, radius)
    assert np.linalg.norm(strain - symmetric_strain) > 0.01 * np.linalg.norm(strain)


def test_zero_depth_has_no_internal_force_but_keeps_regularizing_tangent(membrane):
    count = membrane.mesh.cell_count
    strain, residual, radial = membrane.solve_correction(
        np.full((count, 3), 4e6), np.zeros(count),
        np.broadcast_to(6e10 * membrane.d, (count, 3, 3)), 6e10)
    np.testing.assert_array_equal(strain, 0.)
    assert residual == 0.
    assert radial == 0.


@pytest.mark.parametrize("name,value", [
    ("stress_pa", np.zeros((2, 3))),
    ("stress_pa", np.nan),
    ("thickness_km", -1.),
    ("thickness_km", np.inf),
    ("tangent_pa", np.zeros((3, 3))),
    ("tangent_pa", np.nan),
    ("traction_pa", np.zeros((2, 3))),
    ("traction_pa", np.inf),
    ("young_pa", 0.),
    ("young_pa", np.nan),
    ("young_pa", True),
    ("radius_km", 0.),
    ("radius_km", np.inf),
])
def test_correction_rejects_invalid_inputs(membrane, name, value):
    count = membrane.mesh.cell_count
    values = dict(stress_pa=np.zeros((count, 3)), thickness_km=np.ones(count),
                  tangent_pa=np.broadcast_to(6e10 * membrane.d, (count, 3, 3)),
                  young_pa=6e10,
                  traction_pa=np.zeros((membrane.mesh.vertex_count, 3)),
                  radius_km=5287.)
    if name.endswith("_pa") and name != "young_pa" and np.ndim(value) == 0:
        value = np.full_like(values[name], value)
    if name == "thickness_km" and np.ndim(value) == 0:
        value = np.full(count, value)
    values[name] = value
    with pytest.raises(ValueError, match=name):
        membrane.solve_correction(**values)


def test_nonfinite_linear_solution_does_not_overwrite_displacement(membrane, monkeypatch):
    count = membrane.mesh.cell_count
    membrane.last_displacement_rad.fill(0.012)
    monkeypatch.setattr("tectonics.genesis_shell.spsolve",
                        lambda matrix, rhs: np.full(len(rhs), np.nan))
    with pytest.raises(RuntimeError, match="correction did not converge"):
        membrane.solve_correction(
            np.zeros((count, 3)), np.ones(count),
            np.broadcast_to(6e10 * membrane.d, (count, 3, 3)), 6e10)
    np.testing.assert_array_equal(membrane.last_displacement_rad, 0.012)
