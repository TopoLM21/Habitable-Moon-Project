"""Material path controls, independent of a fracture criterion or mesh edges."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tectonics.genesis_crack_path import (
    CrackInterval, ReferenceCrackPath, initial_interval,
)


def _equator(degrees):
    radians = np.deg2rad(degrees)
    return np.column_stack((np.cos(radians), np.sin(radians), np.zeros(len(radians))))


def _local(points):
    vectors = np.column_stack((np.asarray(points, dtype=float), np.ones(len(points))))
    return vectors/np.linalg.norm(vectors, axis=1)[:, None]


def test_arclength_uses_spherical_distance_in_metres_and_interpolates_vertices():
    points = _equator([0., 30., 90.])
    path = ReferenceCrackPath(points, 1000.)
    np.testing.assert_allclose(path.arclength_m, np.deg2rad([0., 30., 90.])*1e6,
                               rtol=2e-15)
    np.testing.assert_array_equal(path.point_at(path.arclength_m), points)
    np.testing.assert_allclose(path.point_at(np.deg2rad(45.)*1e6), _equator([45.])[0],
                               atol=3e-16)
    assert path.point_at(np.array([[0., path.length_m]])).shape == (1, 2, 3)


def test_material_geometry_cannot_change_through_input_alias_or_reenabled_writes():
    points = _equator([0., 40., 70.])
    expected = points.copy()
    path = ReferenceCrackPath(points, 3000.)
    identity = path.fingerprint
    points[:] = np.nan
    np.testing.assert_array_equal(path.points_xyz, expected)
    for array in (path.points_xyz, path.arclength_m):
        with pytest.raises(ValueError):
            array.flat[0] = 5.
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        path.radius_km = 2.
    assert path.fingerprint == identity
    assert ReferenceCrackPath(expected.copy(), 3000.).fingerprint == identity
    assert ReferenceCrackPath(expected, 3001.).fingerprint != identity
    assert ReferenceCrackPath(expected[::-1], 3000.).fingerprint != identity


def test_rotating_material_sphere_preserves_lengths_and_front_positions():
    points = _local([[-.6, 0.], [0., .25], [.5, .15], [.8, -.15]])
    path = ReferenceCrackPath(points, 5200.)
    rotation = Rotation.from_rotvec([.61, -.48, .18]).as_matrix()
    rotated = ReferenceCrackPath(points@rotation.T, path.radius_km)
    np.testing.assert_allclose(rotated.arclength_m, path.arclength_m, atol=3e-9, rtol=0)
    coordinate = np.linspace(0, path.length_m, 81)
    # The two total lengths differ by roundoff; compare normalized coordinates.
    np.testing.assert_allclose(rotated.point_at(coordinate/path.length_m*rotated.length_m),
                               path.point_at(coordinate)@rotation.T, atol=5e-16, rtol=0)


def test_subdividing_same_great_circle_does_not_change_front_geometry():
    coarse = ReferenceCrackPath(_equator([-60., 60.]), 6300.)
    fine = ReferenceCrackPath(_equator(np.linspace(-60., 60., 29)), 6300.)
    assert fine.length_m == pytest.approx(coarse.length_m, rel=5e-16)
    fractions = np.linspace(0, 1, 93)
    np.testing.assert_allclose(coarse.point_at(fractions*coarse.length_m),
                               fine.point_at(fractions*fine.length_m), atol=5e-16)


def test_simple_support_can_continue_across_hemispheres_without_being_closed():
    path = ReferenceCrackPath(_equator(np.arange(0., 300., 30.)), 1000.)
    assert path.length_m == pytest.approx(1.5*np.pi*1e6)
    np.testing.assert_allclose(path.point_at(path.length_m/2), _equator([135.])[0],
                               atol=3e-16)


@pytest.mark.parametrize("points", [
    [[0., 0., 0.], [1., 0., 0.]],
    [[1., 0., 0.], [1., 0., 0.]],
    [[1., 0., 0.], [-1., 0., 0.]],
    [[np.nan, 0., 0.], [0., 1., 0.]],
    [[2., 0., 0.], [0., 1., 0.]],
    [[1., 0., 0.]],
    [],
    [1., 0., 0.],
    [[1.+0.j, 0., 0.], [0., 1., 0.]],
])
def test_invalid_support_points_are_rejected(points):
    with pytest.raises(ValueError):
        ReferenceCrackPath(points, 5000.)


@pytest.mark.parametrize("radius", [0., -1., np.nan, np.inf, True, "5000"])
def test_invalid_radius_is_rejected(radius):
    with pytest.raises(ValueError):
        ReferenceCrackPath(_equator([0., 30.]), radius)


@pytest.mark.parametrize("xy", [
    [[-1., -1.], [1., 1.], [-1., 1.], [1., -1.]],  # crossing interiors
    [[-1., 0.], [1., 0.], [1., 1.], [0., 0.]],  # endpoint in another arc
    [[-1., 0.], [1., 0.], [0., 0.]],  # adjacent reversal
    [[-1., 0.], [1., 0.], [1., 1.], [-1., 0.]],  # closed loop
    [[-1., 0.], [1., 0.], [1., 1.], [0., 0.], [-.5, 0.]],  # collinear overlap
])
@pytest.mark.parametrize("rotation", [np.eye(3), Rotation.from_rotvec([.4, -.2, .7]).as_matrix()])
def test_self_crossings_touches_and_overlaps_are_rejected_in_any_orientation(xy, rotation):
    with pytest.raises(ValueError, match="intersects|overlaps|touches"):
        ReferenceCrackPath(_local(xy)@rotation.T, 5000.)


def test_long_minor_arcs_intersect_on_opposite_hemisphere_too():
    # First arc crosses the negative x axis; the final arc crosses it north-south.
    points = np.array([[-1., -1., 0.], [-1., 1., 0.], [-1., 0., 1.], [-1., 0., -1.]])
    points /= np.linalg.norm(points, axis=1)[:, None]
    with pytest.raises(ValueError, match="intersects"):
        ReferenceCrackPath(points, 5000.)


def test_ridge_adapter_copies_geometry_but_declines_closed_topology():
    points = _equator([-15., 0., 25.])
    ridge = SimpleNamespace(points_xyz=points, closed=False, seed_index=1)
    path = ReferenceCrackPath.from_ridge(ridge, 5000.)
    np.testing.assert_array_equal(path.points_xyz, points)
    ridge.closed = True
    with pytest.raises(ValueError, match="Closed"):
        ReferenceCrackPath.from_ridge(ridge, 5000.)


@pytest.mark.parametrize("coordinate", [-1., np.inf, np.nan, True, "1", 1+0j])
def test_interpolation_does_not_clip_or_accept_invalid_coordinates(coordinate):
    path = ReferenceCrackPath(_equator([0., 45.]), 1000.)
    with pytest.raises(ValueError):
        path.point_at(coordinate)
    with pytest.raises(ValueError):
        path.point_at(path.length_m+1.)


def test_seed_notch_is_explicit_and_growth_preserves_old_active_material():
    path = ReferenceCrackPath(_equator([0., 45.]), 1000.)
    seed = initial_interval(path, 1000., 40.)
    assert seed == CrackInterval(980., 1020.)
    left = seed.grow(path, left_m=920.)
    right = left.grow(path, right_m=1100.)
    assert seed == CrackInterval(980., 1020.)
    assert left == CrackInterval(920., 1020.)
    assert right.length_m-seed.length_m == 140.
    assert right.grow(path) == right
    assert right.grow(path, left_m=0., right_m=path.length_m).length_m == path.length_m
    with pytest.raises(FrozenInstanceError):
        seed.left_m = 0.


@pytest.mark.parametrize("left,right", [(-1., 2.), (1., 1.), (2., 1.),
                                          (np.nan, 1.), (0., np.inf), (False, 1.)])
def test_invalid_active_intervals_rejected(left, right):
    with pytest.raises(ValueError):
        CrackInterval(left, right)


def test_growth_rejects_healing_and_missing_support():
    path = ReferenceCrackPath(_equator([0., 45.]), 1000.)
    seed = initial_interval(path, 1000., 40.)
    for kwargs in ({"left_m": 981.}, {"right_m": 1019.},
                   {"left_m": -1.}, {"right_m": path.length_m+1.}):
        with pytest.raises(ValueError):
            seed.grow(path, **kwargs)
    with pytest.raises(ValueError):
        CrackInterval(0., path.length_m+1.).grow(path)


@pytest.mark.parametrize("seed,length", [(10., 30.), (1000., 0.), (1000., -1.),
                                          (1000., np.inf), (np.nan, 1.), (True, 1.)])
def test_seed_notch_cannot_be_silently_clipped_or_automatically_chosen(seed, length):
    path = ReferenceCrackPath(_equator([0., 45.]), 1000.)
    with pytest.raises(ValueError):
        initial_interval(path, seed, length)
    with pytest.raises(ValueError):
        initial_interval(path, path.length_m, 20.)
