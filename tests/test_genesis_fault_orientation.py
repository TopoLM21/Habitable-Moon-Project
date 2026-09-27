"""Material weak planes follow finite deformation, including finite stretch."""
from dataclasses import replace

import numpy as np
import pytest

from tectonics.genesis import GenesisParameters
from tectonics.genesis_faults import FaultModel, transport_plane_normals
from tectonics.genesis_material import face_deformation, move_mesh, polar_increment, rotate_tensor
from tectonics.genesis_onset import OnsetParameters
from tectonics.genesis_shell import Membrane, ShellParameters
from tectonics.genesis_tides import TidalParameters


def rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def test_oblique_material_plane_reorients_under_pure_stretch():
    normal = np.array([[1., 1.]]) / np.sqrt(2)
    deformation = np.array([[[1.2, 0.], [0., .8]]])
    moved = transport_plane_normals(normal, deformation)
    expected = np.array([[1/1.2, 1/.8]])
    expected /= np.linalg.norm(expected, axis=1)[:, None]
    np.testing.assert_allclose(moved, expected, rtol=0, atol=2e-16)
    tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
    material_tangent = np.einsum("fij,fj->fi", deformation, tangent)
    np.testing.assert_allclose(np.sum(moved * material_tangent, axis=1), 0., atol=2e-16)
    # Polar rotation alone is identity here, but is not perpendicular to the
    # deformed material plane. This is the finite-stretch regression.
    assert abs(float(np.sum(normal * material_tangent))) > .1


def test_rotation_scale_and_inactive_normals_are_transported_correctly():
    normal = np.array([[.6, .8], [0., 0.], [-.8, .6]])
    rotate = rotation(.37)
    deformation = np.broadcast_to(1.3 * rotate, (3, 2, 2))
    np.testing.assert_allclose(transport_plane_normals(normal, deformation),
                               normal @ rotate.T, rtol=0, atol=2e-16)


def test_plane_transport_composes_and_is_objective_under_frame_changes():
    normal = np.array([[.6, .8]])
    first = np.array([[[1.1, .15], [0., .9]]])
    second = np.array([[[.96, -.13], [.05, 1.07]]])
    sequential = transport_plane_normals(transport_plane_normals(normal, first), second)
    direct = transport_plane_normals(normal, second @ first)
    np.testing.assert_allclose(sequential, direct, rtol=0, atol=3e-16)
    old_frame = rotation(.42)
    new_frame = rotation(-.63)
    transformed = new_frame[None] @ first @ old_frame.T[None]
    result = transport_plane_normals(normal @ old_frame.T, transformed)
    expected = transport_plane_normals(normal, first) @ new_frame.T
    np.testing.assert_allclose(result, expected, rtol=0, atol=3e-16)


@pytest.mark.parametrize("normal,deformation", [
    ([[1., 0.]], [[[1., 0.], [0., -1.]]]),
    ([[1., 0.]], [[[1., 0.], [0., 0.]]]),
    ([[1., 0.]], [[[1., 0.], [0., 1e-14]]]),
    ([[1., 0.]], [[[np.inf, 0.], [0., 1.]]]),
    ([[np.nan, 0.]], [[[1., 0.], [0., 1.]]]),
    ([[2., 0.]], [[[1., 0.], [0., 1.]]]),
    ([[1., 0.]], [[1., 0.], [0., 1.]]),
])
def test_invalid_plane_or_deformation_is_rejected(normal, deformation):
    with pytest.raises(ValueError, match="[Pp]lane|[Dd]eformation"):
        transport_plane_normals(normal, deformation)


def test_final_history_uses_same_material_plane_as_trial_constitutive_response():
    shell = ShellParameters(subdivisions=1, initial_temperature_anomaly_k=0.,
                            convective_traction_pa=0.)
    model = FaultModel(shell, GenesisParameters(), OnsetParameters(), TidalParameters())
    before, _, _ = model.initial()
    count = model.mesh.cell_count
    before = replace(before, membrane_established=True,
                     fault_active=np.ones(count, dtype=bool),
                     plane_normal=np.broadcast_to([.6, .8], (count, 2)).copy(),
                     activation_time_myr=np.zeros(count), damage=np.full(count, .8))
    old_mesh = model.mesh_for(before)
    old_membrane = Membrane(old_mesh, shell.poisson_ratio)
    rng = np.random.default_rng(780)
    moved = move_mesh(old_mesh, old_membrane.vertex_basis,
                      rng.normal(size=(old_mesh.vertex_count, 2)) * 2e-4)
    new_radius = before.radius_km * 1.0001
    deformation = face_deformation(old_mesh, moved, before.radius_km, new_radius)
    rotate, increment = polar_increment(deformation)
    b = np.full(count, .65)
    memory = np.zeros((count, 3))
    memory[:, 2] = .006
    elastic_trial = rotate_tensor(memory + b[:, None] * increment, rotate, engineering=True)
    elastic, stress, tangent = model._mechanical_response(
        before, rotate, elastic_trial, b, before.damage,
        Membrane(moved, shell.poisson_ratio), .001, deformation=deformation)
    expected = model._return(before, deformation, elastic_trial, b, before.damage, .001)
    assert np.any(expected["shear_increment"] != 0)
    np.testing.assert_array_equal(elastic, expected["elastic_strain"])
    np.testing.assert_array_equal(stress, expected["stress_pa"])
    np.testing.assert_array_equal(tangent, expected["tangent_pa"])
    after = replace(before, vertices=moved.vertices.copy(), radius_km=new_radius,
                    time_myr=.001, elastic_strain=elastic)
    final = model._finish_trial_state(before, after, old_mesh, memory, b,
                                      np.ones(count), .001)
    np.testing.assert_allclose(final.plane_normal,
                               transport_plane_normals(before.plane_normal, deformation),
                               rtol=0, atol=5e-16)
    np.testing.assert_allclose(final.last_shear_increment, expected["shear_increment"],
                               rtol=2e-10, atol=1e-14)
    np.testing.assert_array_equal(final.elastic_strain, elastic)
    np.testing.assert_allclose(final.shear_stress_pa, expected["shear_stress_pa"],
                               rtol=2e-10, atol=2e-5)
    polar_only = np.einsum("fij,fj->fi", rotate, before.plane_normal)
    assert np.max(np.abs(final.plane_normal - polar_only)) > 1e-5
    # Finishing a trial does not modify its restart source.
    np.testing.assert_array_equal(before.plane_normal,
                                   np.broadcast_to([.6, .8], (count, 2)))
    np.testing.assert_array_equal(before.cumulative_shear, 0.)
    assert before.friction_work_j == 0.
