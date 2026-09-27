"""Geometry controls with known centre lines; no fracture solution is assumed."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from analysis.genesis_ridge_reference import RidgeReference
from tectonics.genesis_material import rebuild_material_mesh
from tectonics.genesis_ridge import RidgeField, RidgeParameters, RidgeUnavailable
from tectonics.mesh import build_icosphere


@pytest.fixture(scope="module")
def mesh():
    return build_icosphere(4)


def _field(mesh, reference, parameters=None):
    samples = reference.sample(mesh.centroids)
    return RidgeField(mesh, samples.values, samples.plane_normals,
                      reference.radius_km, active=samples.active,
                      parameters=parameters)


@pytest.mark.parametrize("kind", ["great_circle", "small_circle"])
@pytest.mark.parametrize("rotation", [np.eye(3), Rotation.from_rotvec([.4, .6, .7]).as_matrix()])
def test_known_curve_is_traced_without_selecting_mesh_edges(mesh, kind, rotation):
    reference = RidgeReference(kind=kind, rotation=rotation)
    path = _field(mesh, reference).trace(reference.seed)
    assert len(path.points_xyz) > 50
    assert np.all(np.diff(path.arclength_km) > 0)
    np.testing.assert_allclose(np.linalg.norm(path.points_xyz, axis=1), 1., atol=1e-14)
    # Tolerance is physical and fixed for these controls, not a cell count.
    assert np.max(reference.distance_km(path.points_xyz)) < 20.
    assert np.all(path.transverse_contrast >= RidgeParameters().min_transverse_contrast)
    assert not path.closed


def test_joint_rotation_is_objective_up_to_unoriented_path_order(mesh):
    reference = RidgeReference(kind="small_circle")
    original_field = _field(mesh, reference)
    path = original_field.trace(reference.seed)
    rotation = Rotation.from_rotvec([.31, -.62, .27]).as_matrix()
    moved = rebuild_material_mesh(mesh, mesh.vertices @ rotation.T)
    rotated_reference = reference.rotated(rotation)
    rotated_field = _field(moved, rotated_reference)
    rotated = rotated_field.trace(rotated_reference.seed)
    expected = path.points_xyz @ rotation.T
    assert len(rotated.points_xyz) == len(expected)
    reverse = np.linalg.norm(rotated.points_xyz[0]-expected[-1]) < np.linalg.norm(rotated.points_xyz[0]-expected[0])
    if reverse:
        expected = expected[::-1]
        expected_values = path.values[::-1]
        expected_stops = (path.right_stop, path.left_stop)
    else:
        expected_values = path.values
        expected_stops = (path.left_stop, path.right_stop)
    np.testing.assert_allclose(rotated.points_xyz, expected, rtol=0., atol=2e-12)
    np.testing.assert_allclose(rotated.values, expected_values, rtol=0., atol=2e-12)
    assert (rotated.left_stop, rotated.right_stop) == expected_stops
    assert rotated.closed == path.closed
    original_sample = original_field.project(reference.seed)
    rotated_sample = rotated_field.project(rotated_reference.seed)
    # This guard can decide acceptance, so it must not depend on the arbitrary
    # tangent-basis axes used for the polynomial coefficient representation.
    assert rotated_sample.fit_condition == pytest.approx(original_sample.fit_condition,
                                                         rel=0., abs=2e-12)


def test_arbitrary_per_cell_normal_signs_do_not_change_trace(mesh):
    reference = RidgeReference(kind="small_circle")
    samples = reference.sample(mesh.centroids)
    signs = np.random.default_rng(4101).choice([-1., 1.], mesh.cell_count)
    changed = RidgeField(mesh, samples.values, samples.plane_normals*signs[:, None],
                         reference.radius_km, active=samples.active)
    original = _field(mesh, reference).trace(reference.seed)
    flipped = changed.trace(reference.seed)
    np.testing.assert_array_equal(flipped.points_xyz, original.points_xyz)
    np.testing.assert_array_equal(flipped.values, original.values)
    assert (flipped.left_stop, flipped.right_stop) == (original.left_stop, original.right_stop)


def test_uniform_high_damage_does_not_generate_a_path(mesh):
    reference = RidgeReference(kind="uniform")
    field = _field(mesh, reference)
    with pytest.raises(RidgeUnavailable, match="no_transverse_maximum"):
        field.trace(reference.seed)


@pytest.mark.parametrize("subdivision", [2, 3])
def test_coarse_cells_fail_explicitly_at_the_same_physical_scale(subdivision):
    reference = RidgeReference()
    field = _field(build_icosphere(subdivision), reference)
    with pytest.raises(RidgeUnavailable, match="underresolved_cells"):
        field.trace(reference.seed)
    assert field.p.fit_radius_km == 1200.
    assert field.p.probe_distance_km == 600.


def test_isotropic_damage_spot_is_not_misclassified_as_a_ridge(mesh):
    reference = RidgeReference()
    samples = reference.sample(mesh.centroids)
    distance = reference.radius_km*np.arccos(np.clip(mesh.centroids[:, 0], -1., 1.))
    active = distance < 1800.
    values = np.where(active, np.exp(-.5*(distance/600.)**2), 0.)
    normal = np.where(active[:, None], samples.plane_normals, 0.)
    field = RidgeField(mesh, values, normal, reference.radius_km, active=active)
    with pytest.raises(RidgeUnavailable, match="not_an_elongated_ridge"):
        field.project(reference.seed)


def test_perpendicular_crossing_stops_on_ambiguous_plane_direction(mesh):
    reference = RidgeReference()
    first = reference.sample(mesh.centroids)
    other = reference.rotated(Rotation.from_rotvec([np.pi/2, 0., 0.]).as_matrix())
    second = other.sample(mesh.centroids)
    # Each cell stores only one direction, selected from its stronger band.
    normals = np.where((first.values > second.values)[:, None],
                       first.plane_normals, second.plane_normals)
    field = RidgeField(mesh, np.maximum(first.values, second.values), normals,
                       reference.radius_km, active=first.active | second.active)
    with pytest.raises(RidgeUnavailable, match="ambiguous_orientation"):
        field.trace(reference.seed)


def test_resolved_gap_stops_trace_instead_of_joining_disconnected_bands(mesh):
    reference = RidgeReference()
    samples = reference.sample(mesh.centroids)
    longitude = reference.radius_km*np.arctan2(mesh.centroids[:, 1], mesh.centroids[:, 0])
    gap = (longitude > 1200.) & (longitude < 2200.)
    active = samples.active & ~gap
    values = np.where(active, samples.values, 0.)
    normals = np.where(active[:, None], samples.plane_normals, 0.)
    field = RidgeField(mesh, values, normals, reference.radius_km, active=active)
    path = field.trace(reference.seed)
    positions = reference.radius_km*np.arctan2(path.points_xyz[:, 1], path.points_xyz[:, 0])
    assert positions.min() < -2000.
    assert positions.max() < 1200.
    assert len(path.points_xyz) > 20
    assert not path.closed


def test_projection_recovers_offset_seed_with_along_path_amplitude_variation(mesh):
    reference = RidgeReference()
    samples = reference.sample(mesh.centroids)
    longitude = reference.radius_km*np.arctan2(mesh.centroids[:, 1], mesh.centroids[:, 0])
    values = samples.values*(.75+.15*np.sin(longitude/2000.))
    field = RidgeField(mesh, values, samples.plane_normals, reference.radius_km,
                       active=samples.active,
                       parameters=replace(RidgeParameters(), max_branch_length_km=2000.))
    angle = 100./reference.radius_km
    projected = field.project(np.array([np.cos(angle), 0., np.sin(angle)]))
    assert reference.distance_km(projected.point_xyz) < .01
    path = field.trace(reference.seed)
    assert np.max(reference.distance_km(path.points_xyz)) < 2.
    assert np.ptp(path.values) > .2


def test_corrected_arc_length_respects_each_branch_budget(mesh):
    reference = RidgeReference()
    samples = reference.sample(mesh.centroids)
    angle = np.deg2rad(20.)
    # A permitted mismatch between prescribed plane and damage-band normal
    # makes the actual projected advance longer than the predictor step.
    normals = (np.cos(angle)*samples.plane_normals
               + np.sin(angle)*np.cross(mesh.centroids, samples.plane_normals))
    field = RidgeField(mesh, samples.values, normals, reference.radius_km,
                       active=samples.active,
                       parameters=replace(RidgeParameters(), max_branch_length_km=1000.))
    path = field.trace(reference.seed)
    lengths = (path.arclength_km[path.seed_index],
               path.arclength_km[-1]-path.arclength_km[path.seed_index])
    assert all(900. <= length <= 1000.+1e-9 for length in lengths)


def test_opposite_branches_do_not_produce_overlapping_open_loop(mesh):
    reference = RidgeReference(kind="small_circle", small_circle_radius_degrees=20.)
    field = _field(mesh, reference,
                   replace(RidgeParameters(), max_branch_length_km=8000.))
    path = field.trace(reference.seed)
    circumference = 2*np.pi*reference.radius_km*np.sin(np.deg2rad(20.))
    assert "other_branch_approach" in (path.left_stop, path.right_stop)
    assert path.arclength_km[-1] <= circumference+2*field.p.step_km
    assert not path.closed


def test_supported_single_branch_can_close_a_known_loop(mesh):
    reference = RidgeReference(kind="small_circle", small_circle_radius_degrees=20.)
    field = _field(mesh, reference,
                   replace(RidgeParameters(), max_branch_length_km=16000.))
    path = field.trace(reference.seed)
    circumference = 2*np.pi*reference.radius_km*np.sin(np.deg2rad(20.))
    assert path.closed
    assert "closed_path" in (path.left_stop, path.right_stop)
    np.testing.assert_array_equal(path.points_xyz[0], path.points_xyz[-1])
    assert abs(path.arclength_km[-1]-circumference) < 200.
    assert np.all(np.diff(path.arclength_km) > 0.)


def test_unactivated_high_damage_cannot_supply_a_trace(mesh):
    field = RidgeField(mesh, np.full(mesh.cell_count, .8),
                       np.zeros_like(mesh.centroids), 6371.,
                       active=np.zeros(mesh.cell_count, dtype=bool))
    with pytest.raises(RidgeUnavailable, match="inactive_core"):
        field.trace(np.array([1., 0., 0.]))


def test_input_snapshot_is_independent_and_query_does_not_mutate_material(mesh):
    reference = RidgeReference()
    samples = reference.sample(mesh.centroids)
    original_values = samples.values.copy()
    original_normals = samples.plane_normals.copy()
    original_centroids = mesh.centroids.copy()
    field = RidgeField(mesh, samples.values, samples.plane_normals,
                       reference.radius_km, active=samples.active)
    samples.values[:] = 0.
    samples.plane_normals[:] = 0.
    samples.active[:] = False
    assert len(field.trace(reference.seed).points_xyz) > 50
    np.testing.assert_array_equal(field.values, original_values)
    np.testing.assert_array_equal(field.normals, original_normals)
    np.testing.assert_array_equal(mesh.centroids, original_centroids)
