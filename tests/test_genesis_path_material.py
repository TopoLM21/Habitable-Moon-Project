"""Subdivision preserves material histories and objective local tensors."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tectonics.genesis_material import rebuild_material_mesh, material_layer_mass, material_column_depth
from tectonics.genesis_path_material import ConservativeSubdivision
from tectonics.genesis_shell import Membrane
from tectonics.mesh import build_icosphere
from test_genesis_contact import _source_bytes


@pytest.fixture(scope="module")
def projection():
    parent, child = build_icosphere(1), build_icosphere(2)
    return ConservativeSubdivision(parent, child, np.repeat(np.arange(parent.cell_count), 4))


def test_per_parent_mass_enthalpy_work_depth_and_positive_slivers(projection):
    p = projection
    rng = np.random.default_rng(83)
    mass = material_layer_mass(p.parent_mesh, 5300., 3300., 40., 12)*rng.uniform(.5, 2., (80, 12))
    enthalpy = rng.uniform(1e5, 2e6, mass.shape)
    work = rng.uniform(0., 1e20, len(mass))
    child_mass, child_h, child_work = p.extensive(mass), p.intensive(enthalpy), p.extensive(work)
    summed_mass, summed_heat = np.zeros_like(mass), np.zeros_like(mass)
    np.add.at(summed_mass, p.parent_face, child_mass)
    np.add.at(summed_heat, p.parent_face, child_mass*child_h)
    np.testing.assert_allclose(summed_mass, mass, rtol=3e-16, atol=0)
    np.testing.assert_allclose(summed_heat, mass*enthalpy, rtol=4e-16, atol=0)
    np.testing.assert_allclose(np.bincount(p.parent_face, weights=child_work), work, rtol=3e-16, atol=0)
    np.testing.assert_allclose(material_column_depth(p.mesh, 5300., child_mass, 3300.),
        material_column_depth(p.parent_mesh, 5300., mass, 3300.)[p.parent_face], rtol=1e-14, atol=0)
    assert np.min(child_mass) > 0
    assert p.area_relative_error < 1e-13


def test_engineering_strain_energy_and_resolved_weak_plane_stress_preserved(projection):
    p = projection
    rng = np.random.default_rng(33)
    elastic = rng.normal(0., .001, (80, 3))
    angle = rng.uniform(-np.pi, np.pi, 80)
    normals = np.column_stack((np.cos(angle), np.sin(angle)))
    tensor = Membrane(p.parent_mesh, .27).d*60e9
    child_elastic, child_normals = p.tensor(elastic, engineering=True), p.plane_normals(normals)
    old_energy = .5*np.einsum("fi,ij,fj->f", elastic, tensor, elastic)
    new_energy = .5*np.einsum("fi,ij,fj->f", child_elastic, tensor, child_elastic)
    np.testing.assert_allclose(new_energy, old_energy[p.parent_face], rtol=3e-15, atol=1e-10)
    old_stress, new_stress = elastic@tensor, child_elastic@tensor
    np.testing.assert_allclose(p.tensor(old_stress), new_stress, rtol=5e-13, atol=3e-8)

    def resolved(stress, n):
        matrix = np.stack((stress[:, 0], stress[:, 2], stress[:, 2], stress[:, 1]), axis=1).reshape(-1, 2, 2)
        traction = np.einsum("fij,fj->fi", matrix, n)
        tangent = np.column_stack((-n[:, 1], n[:, 0]))
        return np.column_stack((np.sum(traction*n, axis=1), np.sum(traction*tangent, axis=1)))

    np.testing.assert_allclose(resolved(new_stress, child_normals), resolved(old_stress, normals)[p.parent_face], rtol=1e-12, atol=1e-7)


def test_joint_world_rotation_is_objective_and_velocity_remains_tangent(projection):
    p = projection
    world = Rotation.from_rotvec([.61, -.25, .33]).as_matrix()
    old = rebuild_material_mesh(p.parent_mesh, p.parent_mesh.vertices@world.T)
    new = rebuild_material_mesh(p.mesh, p.mesh.vertices@world.T)
    rotated = ConservativeSubdivision(old, new, p.parent_face)
    np.testing.assert_allclose(rotated.frame_rotation, p.frame_rotation, rtol=0, atol=3e-15)
    velocity = np.cross(np.array([3., -7., 11.]), p.parent_mesh.centroids)
    projected = p.tangent_vectors(velocity)
    np.testing.assert_allclose(np.sum(projected*p.mesh.centroids, axis=1), 0., atol=1e-14, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(projected, axis=1), np.linalg.norm(velocity, axis=1)[p.parent_face], rtol=5e-16)
    np.testing.assert_allclose(rotated.tangent_vectors(velocity@world.T), projected@world.T, rtol=2e-14, atol=1e-14)


def test_intensive_identity_inactive_normals_and_owned_geometry(projection):
    p = projection
    flags = np.arange(80) % 3 == 0
    np.testing.assert_array_equal(p.intensive(flags), flags[p.parent_face])
    assert p.intensive(flags).dtype == bool
    np.testing.assert_array_equal(p.plane_normals(np.zeros((80, 2))), 0.)
    np.testing.assert_array_equal(p.extensive(np.zeros((80, 5))), 0.)
    for array in (p.parent_face, p.area_fraction, p.frame_rotation):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    parent, child = deepcopy(p.parent_mesh), deepcopy(p.mesh)
    owned = ConservativeSubdivision(parent, child, p.parent_face)
    parent.vertices[:] = np.nan
    child.vertices[:] = np.nan
    assert np.isfinite(owned.mesh.vertices).all()
    assert np.isfinite(owned.parent_mesh.vertices).all()


@pytest.mark.parametrize("bad", ["ancestry", "outside", "nan", "tensor", "normal", "velocity"])
def test_invalid_parentage_and_fields_are_not_silently_repaired(projection, bad):
    p = projection
    with pytest.raises(ValueError):
        if bad == "ancestry":
            ConservativeSubdivision(p.parent_mesh, p.mesh, np.zeros(p.mesh.cell_count, dtype=int))
        elif bad == "outside":
            ConservativeSubdivision(p.parent_mesh, p.mesh, (p.parent_face+1)%80)
        elif bad == "nan":
            p.extensive(np.full(80, np.nan))
        elif bad == "tensor":
            p.tensor(np.ones((80, 2)))
        elif bad == "normal":
            p.plane_normals(np.full((80, 2), 2.))
        else:
            p.tangent_vectors(p.parent_mesh.centroids)


def test_all_fault_face_arrays_are_projected_without_mutating_source(tmp_path, projection):
    _, _, (state, _, _), _ = _source_bytes(tmp_path)
    p = projection
    original = deepcopy(state)
    bundle = p.fault_fields(state)
    expected = {name for name, value in vars(state).items() if isinstance(value, np.ndarray)}-{"vertices"}
    assert set(bundle) == expected
    np.testing.assert_array_equal(bundle["column_enthalpy"], state.column_enthalpy[p.parent_face])
    np.testing.assert_array_equal(bundle["activation_time_myr"], state.activation_time_myr[p.parent_face])
    for name, value in vars(state).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(original, name))


def test_material_state_from_different_geometry_is_rejected(tmp_path, projection):
    _, _, (state, _, _), _ = _source_bytes(tmp_path)
    state.vertices = state.vertices@Rotation.from_rotvec([.02, -.03, .04]).as_matrix().T
    with pytest.raises(ValueError, match="parent geometry"):
        projection.fault_fields(state)
