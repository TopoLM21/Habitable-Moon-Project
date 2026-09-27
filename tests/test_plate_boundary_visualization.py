"""Geographic overlays must not invent contacts or draw across the world."""
from types import SimpleNamespace

import numpy as np

from tectonics.mesh import build_icosphere
from visualization.plate_boundaries import plate_boundary_segments


def _edge(lon, lat):
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    vertices = np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))
    return SimpleNamespace(vertices=vertices, shared_edges=((0, 1, 0, 1),))


def test_one_plate_has_no_internal_grid_edges():
    mesh = build_icosphere(1)
    assert plate_boundary_segments(mesh, np.zeros(mesh.cell_count, dtype=int)) == []


def test_contact_is_shared_edge_not_centroid_connection():
    mesh = _edge([10, 12], [0, 0])
    segments = np.asarray(plate_boundary_segments(mesh, [3, 9]))
    np.testing.assert_allclose(segments[0, 0], np.deg2rad([10, 0]), atol=1e-14)
    np.testing.assert_allclose(segments[-1, 1], np.deg2rad([12, 0]), atol=1e-14)
    assert plate_boundary_segments(mesh, [3, 3]) == []


def test_antimeridian_contact_reaches_both_map_borders_without_crossing_map():
    segments = np.asarray(plate_boundary_segments(_edge([179, -179], [10, 10]), [0, 1]))
    assert np.isfinite(segments).all()
    assert np.max(np.abs(segments[:, 1, 0] - segments[:, 0, 0])) < np.deg2rad(3)
    lon = segments[:, :, 0]
    assert np.isclose(lon, np.pi).any() and np.isclose(lon, -np.pi).any()


def test_opposite_signed_longitudes_on_same_meridian_remain_finite():
    segments = np.asarray(plate_boundary_segments(_edge([180, -180], [10, 20]), [0, 1]))
    assert np.isfinite(segments).all()
    assert np.max(np.abs(segments[:, 1, 0] - segments[:, 0, 0])) < 1e-12


def test_polar_endpoint_uses_approaching_meridian():
    segments = np.asarray(plate_boundary_segments(_edge([110, 0], [80, 90]), [0, 1]))
    assert np.isfinite(segments).all()
    np.testing.assert_allclose(segments[:, :, 0], np.deg2rad(110), atol=1e-12)
