"""Conservation, geometry and kinematic invariants for onset support."""
import numpy as np
import pytest

from tectonics.genesis_onset_support import (
    NonlocalLoading, face_velocities, material_positions, regional_motion,
)
from tectonics.genesis_shell import Membrane
from tectonics.mesh import build_icosphere, connected_components


@pytest.fixture(scope="module")
def mesh():
    return build_icosphere(2)


def test_filter_preserves_constants_bounds_and_area_mean(mesh):
    smoother = NonlocalLoading(mesh, 5287., 2000.)
    active = np.ones(mesh.cell_count, dtype=bool)
    np.testing.assert_allclose(smoother.apply(np.full(mesh.cell_count, 7.), active), 7., atol=3e-14)
    values = np.random.default_rng(42).uniform(0., 3., mesh.cell_count)
    result = smoother.apply(values, active)
    assert result.min() >= values.min()
    assert result.max() <= values.max()
    area = mesh.areas_unit_sphere
    assert np.dot(result, area) == pytest.approx(np.dot(values, area), rel=3e-15)
    assert np.var(result) < np.var(values)


def test_filter_no_flux_across_molten_band_and_conserves_each_component(mesh):
    active = np.abs(mesh.centroids[:, 2]) > .25
    values = np.where(mesh.centroids[:, 2] > 0, 3., 0.)
    values[~active] = 1000.
    smoother = NonlocalLoading(mesh, 5287., 10000.)
    result = smoother.apply(values, active)
    assert np.all(result[~active] == 0.)
    for component in connected_components(np.flatnonzero(active), mesh.neighbors):
        area = mesh.areas_unit_sphere[component]
        assert np.dot(result[component], area) == pytest.approx(np.dot(values[component], area), abs=1e-12)
    np.testing.assert_allclose(result[active], values[active], atol=1e-12)


def test_filter_empty_zero_length_and_changed_mask(mesh):
    values = np.arange(mesh.cell_count, dtype=float)
    active = np.ones(mesh.cell_count, dtype=bool)
    smoother = NonlocalLoading(mesh, 5287., 500.)
    first = smoother.apply(values, active)
    factor = smoother._factor
    np.testing.assert_array_equal(smoother.apply(values, active), first)
    assert smoother._factor is factor
    active[0] = False
    assert smoother.apply(values, active)[0] == 0.
    assert smoother._factor is not factor
    np.testing.assert_array_equal(smoother.apply(values, np.zeros_like(active)), 0.)
    np.testing.assert_array_equal(NonlocalLoading(mesh, 5287., 0.).apply(values, active)[active], values[active])


def test_filter_harmonic_attenuation_is_consistent_across_refinement():
    errors = []
    radius, length = 5287., 1800.
    attenuation = 1. / (1. + 2. * (length / radius)**2)
    for subdivisions in (1, 2, 3):
        current = build_icosphere(subdivisions)
        harmonic = current.centroids[:, 2]
        result = NonlocalLoading(current, radius, length).apply(harmonic, np.ones(current.cell_count, dtype=bool))
        errors.append(np.sqrt(np.average((result - attenuation * harmonic)**2, weights=current.areas_unit_sphere)))
    # Centroid connections are not exactly orthogonal to shared edges. This
    # two-point flux approximation has a small geometric bias; demand bounded
    # continuum error, without falsely claiming monotonic convergence.
    assert max(errors) < .005
    assert abs(errors[-1] - errors[-2]) < .001


def test_material_positions_are_unit_and_zero_displacement_is_exact(mesh):
    membrane = Membrane(mesh, .25)
    zeros = np.zeros((mesh.vertex_count, 2))
    np.testing.assert_array_equal(material_positions(mesh, membrane.vertex_basis, zeros), mesh.centroids)
    displacement = np.random.default_rng(2).normal(0, .005, zeros.shape)
    moved = material_positions(mesh, membrane.vertex_basis, displacement)
    np.testing.assert_allclose(np.linalg.norm(moved, axis=1), 1., atol=3e-16)
    assert np.max(np.linalg.norm(moved - mesh.centroids, axis=1)) > .001


def test_velocity_zero_exact_and_small_displacement_has_no_acos_floor(mesh):
    np.testing.assert_array_equal(face_velocities(mesh.centroids, mesh.centroids, 5287., .002), 0.)
    old = np.array([[1., 0., 0.]])
    theta = 1e-12
    new = np.array([[np.cos(theta), np.sin(theta), 0.]])
    velocity = face_velocities(old, new, 5287., .002)
    assert np.linalg.norm(velocity) == pytest.approx(theta * 5287. / .002, rel=1e-15)
    assert np.dot(velocity[0], new[0]) == pytest.approx(0., abs=1e-30)


def test_great_circle_speed_and_endpoint_tangent():
    theta = .07
    old = np.array([[1., 0., 0.]])
    new = np.array([[np.cos(theta), np.sin(theta), 0.]])
    expected = 5000. * theta / .003 * np.array([[-np.sin(theta), np.cos(theta), 0.]])
    np.testing.assert_allclose(face_velocities(old, new, 5000., .003), expected, rtol=3e-15)
    with pytest.raises(ValueError, match="antipodal"):
        face_velocities(old, -old, 5000., .003)


def test_rotation_velocity_fit_and_zero_motion(mesh):
    active = np.ones(mesh.cell_count, dtype=bool)
    omega = np.array([3., -4., 7.])
    velocity = np.cross(omega, mesh.centroids)
    row = regional_motion(mesh, mesh.centroids, velocity, active)
    assert row["candidate_region_count"] == 1
    assert row["rigid_fit_residual_fraction"] < 2e-15
    assert row["intact_rms_speed_km_myr"] > 1.
    stationary = regional_motion(mesh, mesh.centroids, np.zeros_like(velocity), active)
    assert stationary["rigid_fit_residual_fraction"] == 0.
    assert stationary["intact_rms_speed_km_myr"] == 0.
    empty = regional_motion(mesh, mesh.centroids, velocity, ~active)
    assert empty["candidate_region_count"] == 0
    assert empty["intact_area_fraction"] == 0.
    assert all(np.isfinite(value) for value in empty.values())


def test_separate_regions_fit_independent_rotations_and_detect_deformation(mesh):
    active = np.abs(mesh.centroids[:, 2]) > .25
    omega = np.where((mesh.centroids[:, 2] > 0)[:, None], [1., 2., 3.], [-3., 1., -1.])
    velocity = np.cross(omega, mesh.centroids)
    row = regional_motion(mesh, mesh.centroids, velocity, active)
    assert row["candidate_region_count"] == 2
    assert row["rigid_fit_residual_fraction"] < 3e-15
    velocity *= (1. + .5 * mesh.centroids[:, :1])
    assert regional_motion(mesh, mesh.centroids, velocity, active)["rigid_fit_residual_fraction"] > .05
