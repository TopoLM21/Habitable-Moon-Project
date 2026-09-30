"""Independent velocity-space checks for the optional rigid mantle projection."""
from dataclasses import replace

import numpy as np
import pytest

from tectonics.mantle import (
    MantleFlowState, plate_mean_mantle_omega, plate_rigid_mantle_fit,
)
from tectonics.mesh import build_icosphere


def _state(field):
    return MantleFlowState(0., np.asarray(field).copy(), 1.)


def _owners(mesh):
    x = mesh.centroids
    return ((x[:, 0] > .1).astype(np.int32)
            + 2 * (x[:, 1] > -.15).astype(np.int32))


def _tangent_omega(mesh, velocity_per_radius):
    return np.cross(mesh.centroids, velocity_per_radius)


@pytest.mark.parametrize("subdivisions", [0, 2, 3])
@pytest.mark.parametrize("encoding", ["full_euler", "minimum_norm"])
def test_exact_rigid_flow_recovered_on_arbitrary_plates(subdivisions, encoding):
    mesh = build_icosphere(subdivisions)
    owner = _owners(mesh)
    exact = np.array([[.001, -.002, .004], [.003, .001, -.005],
                      [-.002, .003, .002], [.004, -.001, .001]])
    field = exact[owner]
    if encoding == "minimum_norm":
        field = _tangent_omega(mesh, np.cross(field, mesh.centroids))
    fit = plate_rigid_mantle_fit(mesh, owner, 4, 5287., _state(field))
    np.testing.assert_allclose(fit.omega_rad_per_myr, exact, rtol=2e-14, atol=1e-17)
    np.testing.assert_allclose(fit.relative_residual, 0., atol=4e-15)
    np.testing.assert_allclose(fit.represented_kinetic_fraction, 1., atol=4e-15)
    np.testing.assert_array_equal(fit.moment_rank, 3)


def test_tangent_encoding_exposes_legacy_two_thirds_attenuation():
    mesh = build_icosphere(2)
    exact = np.array([.001, -.002, .004])
    owner = np.zeros(mesh.cell_count, dtype=np.int32)
    field = _tangent_omega(mesh, np.cross(exact, mesh.centroids))
    legacy = plate_mean_mantle_omega(mesh, owner, 1, 5287., _state(field))
    fit = plate_rigid_mantle_fit(mesh, owner, 1, 5287., _state(field))
    np.testing.assert_allclose(legacy[0], 2/3*exact, atol=1e-17)
    np.testing.assert_allclose(fit.omega_rad_per_myr[0], exact, atol=1e-17)


def _mixed_field(mesh):
    x = mesh.centroids
    matrix = np.array([[.004, .001, -.002], [.001, -.003, .002], [-.002, .002, -.001]])
    velocity = x @ matrix
    velocity -= x * np.sum(velocity * x, axis=1)[:, None]
    velocity += np.cross([.001, -.002, .004], x)
    return _tangent_omega(mesh, velocity)


def test_mixed_field_matches_direct_overdetermined_velocity_fit():
    mesh = build_icosphere(2)
    # Deliberately unequal areas ensure an unweighted fit cannot pass.
    mesh = replace(mesh, areas_unit_sphere=mesh.areas_unit_sphere
        * np.exp(2 * mesh.centroids[:, 2]))
    owner, field = _owners(mesh), _mixed_field(mesh)
    fit = plate_rigid_mantle_fit(mesh, owner, 4, 5287., _state(field))
    old = plate_mean_mantle_omega(mesh, owner, 4, 5287., _state(field))
    for pid in range(4):
        mask = owner == pid
        x, a = mesh.centroids[mask], mesh.areas_unit_sphere[mask]
        # Independent oracle: stack the actual velocity design matrix rather
        # than constructing the production code's 3x3 normal equations.
        design = np.stack([np.cross(axis, x) for axis in np.eye(3)], axis=2)
        velocity = np.cross(field[mask], x)
        direct = np.linalg.lstsq((design * np.sqrt(a)[:, None, None]).reshape(-1, 3),
            (velocity * np.sqrt(a)[:, None]).ravel(), rcond=None)[0]
        np.testing.assert_allclose(fit.omega_rad_per_myr[pid], direct, atol=2e-17)
        old_error = np.sum(a[:, None] * (velocity - np.cross(old[pid], x))**2)
        new_error = np.sum(a[:, None] * (velocity - np.cross(direct, x))**2)
        assert new_error < old_error
    assert np.all(fit.relative_residual > .05)
    np.testing.assert_allclose(fit.represented_kinetic_fraction,
        1-fit.relative_residual**2, atol=3e-15)


def test_radial_gauge_does_not_change_physical_projection():
    mesh = build_icosphere(2)
    owner, field = _owners(mesh), _mixed_field(mesh)
    gauge = np.sin(np.arange(mesh.cell_count))[:, None] * mesh.centroids * .02
    first = plate_rigid_mantle_fit(mesh, owner, 4, 5287., _state(field))
    second = plate_rigid_mantle_fit(mesh, owner, 4, 5287., _state(field + gauge))
    for name in ("omega_rad_per_myr", "local_rms_speed_km_per_myr",
                 "relative_residual", "represented_kinetic_fraction"):
        np.testing.assert_allclose(getattr(first, name), getattr(second, name),
                                   rtol=3e-14, atol=1e-16)


def test_common_rigid_rotation_adds_same_euler_vector_and_keeps_residual():
    mesh = build_icosphere(2)
    owner, field = _owners(mesh), _mixed_field(mesh)
    common = np.array([-.003, .005, .007])
    added = _tangent_omega(mesh, np.cross(common, mesh.centroids))
    first = plate_rigid_mantle_fit(mesh, owner, 4, 5287., _state(field))
    second = plate_rigid_mantle_fit(mesh, owner, 4, 5287., _state(field + added))
    np.testing.assert_allclose(second.omega_rad_per_myr,
        first.omega_rad_per_myr + common, atol=4e-17)
    np.testing.assert_allclose(first.residual_rms_speed_km_per_myr,
        second.residual_rms_speed_km_per_myr, rtol=4e-15)
    # Common rotation is a gauge for relative plate motion at any boundary point.
    for a, b, u, v in mesh.shared_edges:
        pa, pb = owner[a], owner[b]
        if pa != pb:
            midpoint = mesh.vertices[u] + mesh.vertices[v]
            midpoint /= np.linalg.norm(midpoint)
            relative = np.cross(first.omega_rad_per_myr[pb]
                                - first.omega_rad_per_myr[pa], midpoint)
            updated = np.cross(second.omega_rad_per_myr[pb]
                               - second.omega_rad_per_myr[pa], midpoint)
            np.testing.assert_allclose(relative, updated, atol=4e-17)


def test_rigid_fit_is_equivariant_under_world_rotation_and_plate_relabeling():
    mesh = build_icosphere(2)
    owner, field = _owners(mesh), _mixed_field(mesh)
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    permute = np.array([2, 0, 3, 1])
    turned = replace(mesh, vertices=mesh.vertices @ rotation.T,
                     centroids=mesh.centroids @ rotation.T)
    first = plate_rigid_mantle_fit(mesh, owner, 4, 5287., _state(field))
    second = plate_rigid_mantle_fit(turned, permute[owner], 4, 5287.,
                                   _state(field @ rotation.T))
    np.testing.assert_allclose(second.omega_rad_per_myr[permute],
        first.omega_rad_per_myr @ rotation.T, atol=1e-17)
    np.testing.assert_allclose(second.relative_residual[permute],
        first.relative_residual, atol=2e-15)


@pytest.mark.parametrize("subdivisions", [0, 1, 2, 3])
def test_smooth_nonrigid_field_has_grid_independent_analytic_energy(subdivisions):
    mesh = build_icosphere(subdivisions)
    x = mesh.centroids
    rigid = np.array([.001, -.002, .004])
    matrix = np.diag([1., -1., 0.])
    amplitude = .006
    velocity = amplitude * (x @ matrix - x * np.einsum("ni,ij,nj->n", x, matrix, x)[:, None])
    velocity += np.cross(rigid, x)
    fit = plate_rigid_mantle_fit(mesh, np.zeros(mesh.cell_count, dtype=np.int32),
                                1, 5287., _state(_tangent_omega(mesh, velocity)))
    # For a traceless symmetric matrix, sphere mean |P M r|^2=tr(M^2)/5.
    rigid_square = 2/3 * float(rigid @ rigid)
    residual_square = amplitude**2 * np.trace(matrix @ matrix)/5
    np.testing.assert_allclose(fit.omega_rad_per_myr[0], rigid, atol=3e-17)
    assert fit.represented_kinetic_fraction[0] == pytest.approx(
        rigid_square/(rigid_square+residual_square), abs=4e-15)
    assert fit.residual_rms_speed_km_per_myr[0] == pytest.approx(
        5287*np.sqrt(residual_square), rel=4e-15)


def test_empty_and_one_cell_patches_have_defined_minimum_norm_solution():
    mesh = build_icosphere(1)
    owner = np.ones(mesh.cell_count, dtype=np.int32)
    owner[0] = 0
    field = np.zeros((mesh.cell_count, 3))
    field[0] = np.cross(mesh.centroids[0], [.001, -.002, .004])
    fit = plate_rigid_mantle_fit(mesh, owner, 3, 5287., _state(field))
    np.testing.assert_allclose(fit.omega_rad_per_myr[0], field[0], atol=1e-17)
    np.testing.assert_array_equal(fit.moment_rank, [2, 3, 0])
    np.testing.assert_array_equal(fit.omega_rad_per_myr[1:], 0.)
    np.testing.assert_allclose(fit.relative_residual, 0., atol=2e-15)
    assert fit.plate_area_km2[2] == 0.
    np.testing.assert_array_equal(fit.represented_kinetic_fraction[1:], 0.)


def test_projection_does_not_mutate_or_depend_on_radius_for_euler_fit():
    mesh = build_icosphere(1)
    owner, field = _owners(mesh), _mixed_field(mesh)
    original_owner, original_field = owner.copy(), field.copy()
    state = _state(field)
    first = plate_rigid_mantle_fit(mesh, owner, 4, 1000., state)
    second = plate_rigid_mantle_fit(mesh, owner, 4, 2000., state)
    np.testing.assert_array_equal(owner, original_owner)
    np.testing.assert_array_equal(state.cell_omega_rad_per_myr, original_field)
    np.testing.assert_array_equal(first.omega_rad_per_myr, second.omega_rad_per_myr)
    np.testing.assert_array_equal(2*first.fitted_rms_speed_km_per_myr,
                                  second.fitted_rms_speed_km_per_myr)


@pytest.mark.parametrize("count,radius", [(0, 1.), (True, 1.), (1.5, 1.),
                                         (1, 0.), (1, float("nan"))])
def test_invalid_projection_parameters_fail_clearly(count, radius):
    mesh = build_icosphere(0)
    with pytest.raises(ValueError):
        plate_rigid_mantle_fit(mesh, np.zeros(mesh.cell_count, dtype=np.int32),
                              count, radius, _state(np.zeros((mesh.cell_count, 3))))
