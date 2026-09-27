"""Remeshing errors are measured separately from physical crack release."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from analysis.genesis_path_mechanics_audit import tangent_prolongation, audit_remeshing
from tectonics.genesis_crack_path import ReferenceCrackPath
from tectonics.genesis_material import rebuild_material_mesh
from tectonics.genesis_path_material import ConservativeSubdivision
from tectonics.genesis_path_mesh import insert_crack_path
from tectonics.genesis_shell import Membrane
from tectonics.genesis_shell_release import FrozenShell
from tectonics.mesh import build_icosphere


def _insertion(edge=False):
    mesh = build_icosphere(2)
    points = mesh.vertices[[0, 1]] if edge else np.array([[1., .2, .1], [1., .3, .4], [1., .8, .5]])
    points = points/np.linalg.norm(points, axis=1)[:, None]
    return mesh, insert_crack_path(mesh, ReferenceCrackPath(points, 5300.))


def _shell_pair(mesh, insertion):
    material = ConservativeSubdivision(mesh, insertion.mesh, insertion.parent_face)
    elastic = np.random.default_rng(5).normal(size=(mesh.cell_count, 3))*1e-7
    old = FrozenShell.from_uniform(mesh, 5.3e6, 1e4, 60e9, .25,
                                   np.zeros_like(mesh.vertices), elastic_strain=elastic)
    new = FrozenShell.from_uniform(insertion.mesh, 5.3e6, 1e4, 60e9, .25,
            np.zeros_like(insertion.mesh.vertices),
            elastic_strain=material.tensor(elastic, engineering=True))
    return old, new


def test_existing_edge_identity_has_no_remeshing_energy():
    mesh, insertion = _insertion(edge=True)
    p = tangent_prolongation(mesh, insertion)
    np.testing.assert_allclose(p.toarray(), np.eye(p.shape[0]), atol=4e-16, rtol=0)
    old, new = _shell_pair(mesh, insertion)
    result = audit_remeshing(old, new, insertion)
    for key in ("force", "prestress", "stiffness", "gauge"):
        assert result[key+"_pullback_relative_error"] < 2e-15
    scale = abs(result["old_reduced_potential_j"])
    assert abs(result["zero_cut_remeshing_potential_change_j"]) < scale*1e-14
    assert abs(result["zero_cut_remeshing_relaxation_j"]) < scale*1e-14


def test_valid_rotation_and_radial_patches_do_not_imply_nested_stiffness():
    mesh, insertion = _insertion()
    old, new = _shell_pair(mesh, insertion)
    result = audit_remeshing(old, new, insertion)
    assert result["rotation_patch_displacement_relative_error"] < 2e-15
    assert result["rotation_patch_max_strain"] < 3e-12
    assert result["radial_patch_strain_relative_error"] < 2e-15
    assert result["radial_patch_energy_relative_error"] < 2e-14
    assert result["stiffness_pullback_relative_error"] > 1e-4
    assert result["prestress_pullback_relative_error"] > 1e-4
    assert abs(result["initial_energy_change_j"]) < result["old_initial_energy_j"]*2e-14
    assert abs(result["zero_cut_remeshing_potential_change_j"]) > result["old_initial_energy_j"]*1e-5
    assert result["zero_cut_remeshing_relaxation_j"] > 0
    np.testing.assert_allclose(result["zero_cut_remeshing_lifted_potential_change_j"]
            - result["zero_cut_remeshing_potential_change_j"],
            result["zero_cut_remeshing_relaxation_j"], rtol=2e-13)
    assert result["interpretation"] == "zero_cut_remeshing_not_fracture_energy"


def test_tangent_prolongation_is_objective_for_general_displacements():
    mesh, insertion = _insertion()
    world = Rotation.from_rotvec([.41, -.73, .22]).as_matrix()
    old_rotated = rebuild_material_mesh(mesh, mesh.vertices@world.T)
    child_rotated = rebuild_material_mesh(insertion.mesh, insertion.mesh.vertices@world.T)
    rotated = replace(insertion, mesh=child_rotated)
    old, new = Membrane(mesh, .25), Membrane(insertion.mesh, .25)
    old_r, new_r = Membrane(old_rotated, .25), Membrane(child_rotated, .25)
    q = np.random.default_rng(94).normal(size=old.ndof)
    xyz = np.einsum("vij,vj->vi", old.vertex_basis, q[:-1].reshape(-1, 2))
    qr = np.r_[np.einsum("vij,vi->vj", old_r.vertex_basis, xyz@world.T).ravel(), q[-1]]
    fine = tangent_prolongation(mesh, insertion)@q
    fine_r = tangent_prolongation(old_rotated, rotated)@qr
    xyz_fine = np.einsum("vij,vj->vi", new.vertex_basis, fine[:-1].reshape(-1, 2))
    xyz_fine_r = np.einsum("vij,vj->vi", new_r.vertex_basis, fine_r[:-1].reshape(-1, 2))
    np.testing.assert_allclose(xyz_fine@world.T, xyz_fine_r, rtol=2e-13, atol=2e-15)
    assert fine[-1] == fine_r[-1] == q[-1]


@pytest.mark.parametrize("kind", ["negative", "nan", "wrong_geometry", "owner"])
def test_invalid_ancestry_is_rejected(kind):
    mesh, insertion = _insertion()
    weights = insertion.vertex_barycentric.copy()
    owner = insertion.vertex_parent_face.copy()
    if kind == "negative":
        weights[-1, 0] = -.1
    elif kind == "nan":
        weights[-1, 0] = np.nan
    elif kind == "wrong_geometry":
        weights[-1] = [1., 0., 0.]
    else:
        owner[-1] = mesh.cell_count
    with pytest.raises(ValueError, match="ancestry"):
        tangent_prolongation(mesh, replace(insertion,
            vertex_barycentric=weights, vertex_parent_face=owner))


def test_changed_physical_radius_is_not_remeshing():
    mesh, insertion = _insertion()
    old, new = _shell_pair(mesh, insertion)
    new.radius_m += 1.
    with pytest.raises(ValueError, match="unchanged physical radius"):
        audit_remeshing(old, new, insertion)


def test_geometry_partition_can_fail_chord_material_transport_guard():
    mesh = build_icosphere(1)
    points = np.array([[1., .2, .1], [1., .3, .4], [1., .8, .5]])
    points /= np.linalg.norm(points, axis=1)[:, None]
    insertion = insert_crack_path(mesh, ReferenceCrackPath(points, 5300.))
    # Spherical partition and rigid-motion interpolation remain defined,
    # but an extremely oblique child chord is outside the material convention.
    assert tangent_prolongation(mesh, insertion).shape[0] > 2*mesh.vertex_count+1
    with pytest.raises(ValueError, match="share a hemisphere"):
        ConservativeSubdivision(mesh, insertion.mesh, insertion.parent_face)
