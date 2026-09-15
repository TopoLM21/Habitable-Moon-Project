"""Analytic spherical geometry and isolation tests for optional ridge frames."""
import numpy as np
import pytest

from tectonics.boundary_geometry import estimate_boundary_normals
from tectonics.kinematics import angular_velocity_vectors, classify_boundaries
from tectonics.mesh import SphereMesh, build_icosphere
from tectonics.plates import Plate, PlateSystem


def _unit(x):
    x = np.asarray(x, dtype=float)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _chain(points, links=None, pairs=None):
    """Minimal mesh exposing exact great-circle segments and their flanks."""
    vertices = _unit(points)
    if links is None:
        links = [(i, i + 1) for i in range(len(vertices) - 1)]
    centroids, shared, owner = [], [], []
    for i, (u, v) in enumerate(links):
        midpoint = _unit(vertices[u] + vertices[v])
        normal = _unit(np.cross(vertices[u], vertices[v]))
        centroids.extend([_unit(midpoint - .001 * normal), _unit(midpoint + .001 * normal)])
        shared.append((2 * i, 2 * i + 1, u, v))
        owner.extend((0, 1) if pairs is None else pairs[i])
    n = 2 * len(links)
    mesh = SphereMesh(vertices, np.zeros((n, 3), dtype=int), np.array(centroids),
                      np.full(n, 4 * np.pi / n), tuple(() for _ in range(n)), tuple(shared))
    return mesh, np.array(owner, dtype=np.int32)


def test_zero_width_reproduces_existing_kinematics_and_does_not_mutate():
    mesh = build_icosphere(2)
    owner = (mesh.centroids @ _unit([.31, .57, .76]) > 0).astype(np.int32)
    system = PlateSystem(owner, (Plate(0, 0, _unit([1, 2, 3]), .001),
                                 Plate(1, 1, _unit([-2, 1, 1]), .002)))
    before = owner.copy()
    geo = estimate_boundary_normals(mesh, owner, 5287.)
    bounds = classify_boundaries(mesh, system, 5287., 4., 1.)
    omega = angular_velocity_vectors(system)
    a, b = geo.edges[:, 0], geo.edges[:, 1]
    relative = np.cross(omega[owner[b]] - omega[owner[a]], geo.midpoints) * 5287.
    np.testing.assert_allclose(np.sum(relative * geo.normals, axis=1),
                               [b.normal_rate_km_per_myr for b in bounds], atol=1e-12)
    np.testing.assert_array_equal(geo.normals, geo.raw_normals)
    np.testing.assert_array_equal(owner, before)


def test_great_circle_keeps_analytic_divergence_and_transform():
    theta = np.linspace(-.3, .3, 31)
    mesh, owner = _chain(np.column_stack((np.zeros_like(theta), np.sin(theta), np.cos(theta))))
    geo = estimate_boundary_normals(mesh, owner, 5287., half_width_km=450.)
    assert np.max(geo.support_counts) > 3
    np.testing.assert_allclose(geo.normals, geo.raw_normals, atol=1e-12)
    transform = np.cross(np.array([.003, 0., 0.]), geo.midpoints) * 5287.
    divergent = np.cross(np.array([0., -.003, 0.]), geo.midpoints) * 5287.
    np.testing.assert_allclose(np.sum(transform * geo.normals, axis=1), 0., atol=1e-12)
    np.testing.assert_allclose(np.sum(divergent * geo.normals, axis=1),
                               .003 * 5287. * geo.midpoints[:, 2], atol=1e-12)


def test_digitized_oblique_great_circle_reduces_normal_error():
    mesh = build_icosphere(3)
    axis = _unit([.31, .57, .76])
    owner = (mesh.centroids @ axis > 0).astype(np.int32)
    geo = estimate_boundary_normals(mesh, owner, 5287., half_width_km=1200.)
    sign = np.where(owner[geo.edges[:, 0]] < owner[geo.edges[:, 1]], 1., -1.)
    expected = _unit(axis - geo.midpoints * (geo.midpoints @ axis)[:, None]) * sign[:, None]
    raw_error = np.mean(np.sum((geo.raw_normals - expected) ** 2, axis=1))
    estimate_error = np.mean(np.sum((geo.normals - expected) ** 2, axis=1))
    assert np.count_nonzero(geo.support_counts > 1) > len(geo.edges) // 3
    # Compare with the known continuous boundary, not a preselected smoothing
    # factor: the sharp-corner guard intentionally leaves some raster jogs.
    assert estimate_error < raw_error
    np.testing.assert_allclose(np.linalg.norm(geo.normals, axis=1), 1., atol=1e-12)
    np.testing.assert_allclose(np.sum(geo.normals * geo.midpoints, axis=1), 0., atol=1e-12)
    assert np.all(np.sum(geo.normals * geo.raw_normals, axis=1) > 0.)


def test_same_pair_disconnected_chains_do_not_share_support():
    points = [[-.08, 0, 1], [-.04, 0, 1], [0, 0, 1], [.04, 0, 1], [.08, 0, 1],
              [-.08, .001, 1], [-.04, .009, 1], [0, .001, 1], [.04, .009, 1], [.08, .001, 1]]
    links = [(i, i + 1) for i in range(4)] + [(i, i + 1) for i in range(5, 9)]
    mesh, owner = _chain(points, links)
    isolated, owners = _chain(points[:5])
    both = estimate_boundary_normals(mesh, owner, 5287., half_width_km=1000.)
    one = estimate_boundary_normals(isolated, owners, 5287., half_width_km=1000.)
    np.testing.assert_array_equal(both.normals[:4], one.normals)
    np.testing.assert_array_equal(both.support_counts[:4], one.support_counts)


def test_triple_junction_and_right_angle_corners_are_protected():
    mesh, owner = _chain([[-.1, 0, 1], [-.05, 0, 1], [0, 0, 1], [0, .05, 1], [0, .1, 1]])
    geo = estimate_boundary_normals(mesh, owner, 5287., half_width_km=2000.)
    assert geo.protected_edges[1] and geo.protected_edges[2]
    np.testing.assert_array_equal(geo.normals[1:3], geo.raw_normals[1:3])
    mesh, owner = _chain([[-.1, 0, 1], [0, 0, 1], [.1, 0, 1], [0, .1, 1]],
                         links=[(0, 1), (1, 2), (1, 3)], pairs=[(0, 1), (0, 2), (1, 2)])
    geo = estimate_boundary_normals(mesh, owner, 5287., half_width_km=2000.)
    assert np.all(geo.protected_edges)
    np.testing.assert_array_equal(geo.normals, geo.raw_normals)
    assert np.all(geo.support_counts == 1)


def test_owner_id_renumbering_does_not_change_geometric_result():
    mesh = build_icosphere(2)
    owner = (mesh.centroids[:, 0] > 0).astype(np.int32)
    first = estimate_boundary_normals(mesh, owner, 5287., half_width_km=2000.)
    renumbered = estimate_boundary_normals(mesh, np.where(owner == 0, 77, 3), 5287., half_width_km=2000.)
    np.testing.assert_array_equal(first.edges, renumbered.edges)
    np.testing.assert_array_equal(first.normals, renumbered.normals)


def test_uniform_plate_empty_and_bad_parameters():
    mesh = build_icosphere(0)
    owner = np.zeros(mesh.cell_count, dtype=np.int32)
    assert estimate_boundary_normals(mesh, owner, 5287.).normals.shape == (0, 3)
    for radius, width, corner in [(0, 0, 75), (5287, -1, 75), (5287, np.nan, 75), (5287, 1, 181)]:
        with pytest.raises(ValueError):
            estimate_boundary_normals(mesh, owner, radius, half_width_km=width, max_corner_turn_degrees=corner)
    with pytest.raises(ValueError):
        estimate_boundary_normals(mesh, owner.astype(float), 5287.)
