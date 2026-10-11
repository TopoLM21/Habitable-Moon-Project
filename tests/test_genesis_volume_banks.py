"""Prospective tied-bank identities on authentic geometry and full memory."""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis_volume_banks import (
    evaluate_volume_banks, prepare_volume_banks, validate_volume_banks,
)
from tectonics.genesis_volume_basal import assemble_genesis_volume_basal
from tectonics.genesis_volume_operator import (
    assemble_genesis_volume_operator, assemble_updated_volume_operator,
    evaluate_genesis_volume_operator,
)


@pytest.fixture(scope="module")
def parent():
    path = Path(__file__).resolve().parents[1]/"analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json"
    return assemble_genesis_volume_operator(json.loads(path.read_text()))


@pytest.fixture(scope="module")
def basal(parent):
    return assemble_genesis_volume_basal(parent)


@pytest.fixture(scope="module")
def cuts(parent):
    a, b, c = parent.geometry.faces[0]
    # Explicit computational loop around one whole face, not a source crack.
    return np.array(((a, b), (b, c), (c, a)), int)


@pytest.fixture(scope="module")
def banks(parent, cuts, basal):
    return prepare_volume_banks(parent, cuts, basal=basal)


@pytest.fixture(scope="module")
def memory(parent):
    # Declared mechanical probe: add angular/radial variations to real source
    # memory.  These coefficients are never published as a Genesis state.
    rng = np.random.default_rng(28465)
    return (parent.memory0+rng.normal(size=parent.memory0.shape)*2e-5,
            rng.normal(size=parent.memory0.shape)*3e-5)


def _relative(actual, expected):
    return np.linalg.norm(actual-expected)/max(np.linalg.norm(expected), 1.)


def test_whole_material_geometry_ids_frames_and_variance_are_preserved(parent, banks):
    t = banks.topology
    assert banks.parent_assembly is parent
    assert banks.parent_assembly.geometry is parent.geometry
    np.testing.assert_array_equal(t.original_faces, parent.geometry.faces)
    np.testing.assert_array_equal(t.parent_vertex[t.mesh.faces], parent.geometry.faces)
    assert banks.ndof == parent.ndof+18
    assert np.array_equal(banks.dofs, banks.face_dofs[parent.face_indices])
    assert len(parent.cohort_ids) == 480
    assert np.all(parent.inverse_scale_variance > 0)
    assert not t.calculated_fracture_surface
    assert not banks.physical_bank_creation_admitted
    assert not banks.physical_contact_release_admitted
    assert not banks.physical_force_work_admission


def test_all_six_tie_constraints_have_exact_prolongation_nullspace(parent, banks):
    R, J = banks.R, banks.J
    assert R.shape == (banks.ndof, parent.ndof)
    assert J.shape == (banks.ndof-parent.ndof, banks.ndof)
    assert (J@R).nnz == 0
    np.testing.assert_array_equal(J[:, parent.ndof:].toarray(), np.eye(18))
    np.testing.assert_array_equal(R[:parent.ndof].toarray(), np.eye(parent.ndof))
    # Full row rank follows from the identity block; tied nullspace has exactly
    # the parent's dimension, including A and P Cartesian degrees of freedom.
    q = np.random.default_rng(89).normal(size=parent.ndof)
    tied = R@q
    np.testing.assert_array_equal(J@tied, np.zeros(18))
    np.testing.assert_array_equal(tied.reshape(-1, 6), q.reshape(-1, 6)[banks.topology.parent_vertex])


def test_elastic_mass_drag_and_reference_force_pull_back_to_parent(parent, basal, banks):
    R = banks.R
    for actual, expected in ((R.T@banks.K@R, parent.stiffness),
                             (R.T@banks.basal.mass_matrix@R, basal.mass_matrix),
                             (R.T@banks.basal.basal_drag_matrix@R, basal.basal_drag_matrix)):
        assert _relative(actual.toarray(), expected.toarray()) < 5e-16
    assert _relative(R.T@banks.basal.external_force_n, basal.external_force_n) < 4e-16
    assert banks.basal.total_live_mass_kg == basal.total_live_mass_kg
    assert banks.basal.cohort_mass_matrices is basal.cohort_mass_matrices
    assert banks.basal.face_basal_drag_matrices is basal.face_basal_drag_matrices
    assert banks.basal.face_external_force_n is basal.face_external_force_n


@pytest.mark.parametrize("vary_beta", (False, True))
def test_full_angular_and_radial_memory_force_energy_tied_identity(parent, banks, memory, vary_beta):
    e0, e1 = memory
    u = np.random.default_rng(142).normal(size=parent.ndof)*.002
    beta = np.linspace(.2, 1., len(parent.cohort_ids)) if vary_beta else None
    old = evaluate_genesis_volume_operator(parent, u, elastic_constant=e0,
        elastic_inverse_scale=e1, strain_factors=beta)
    new = evaluate_volume_banks(banks, banks.R@u, elastic_constant=e0,
        elastic_inverse_scale=e1, strain_factors=beta)
    np.testing.assert_array_equal(new.memory0, old.memory0)
    np.testing.assert_array_equal(new.memory1, old.memory1)
    np.testing.assert_array_equal(new.cohort_volume_mean_stress_pa, old.cohort_volume_mean_stress_pa)
    assert new.energy_j == old.energy_j
    assert new.maximum_elastic_principal_strain == old.maximum_elastic_principal_strain
    assert _relative(banks.R.T@new.force, old.force) < 4e-16
    assert _relative((banks.R.T@new.stiffness@banks.R).toarray(), old.stiffness.toarray()) < 5e-16
    assert (banks.R@u)@new.force == pytest.approx(u@old.force, rel=2e-15)


def test_accepted_memory_transfer_uses_zero_new_increment(parent, banks, memory):
    e0, e1 = memory
    response = evaluate_volume_banks(banks, np.zeros(banks.ndof),
        elastic_constant=e0, elastic_inverse_scale=e1)
    np.testing.assert_array_equal(response.memory0, e0)
    np.testing.assert_array_equal(response.memory1, e1)
    old = evaluate_genesis_volume_operator(parent, np.zeros(parent.ndof),
        elastic_constant=e0, elastic_inverse_scale=e1)
    assert response.energy_j == old.energy_j
    assert _relative(banks.R.T@response.force, old.force) < 4e-16
    # Absolute accepted displacement is copied separately, not applied again
    # to already accumulated endpoint strains at this transfer.
    q = np.random.default_rng(718).normal(size=parent.ndof)*.01
    duplicated_q = banks.R@q
    np.testing.assert_array_equal(banks.J@duplicated_q, np.zeros(banks.J.shape[0]))
    doubled = evaluate_volume_banks(banks, duplicated_q,
        elastic_constant=e0, elastic_inverse_scale=e1)
    assert not np.array_equal(doubled.memory0, response.memory0)


def test_full_memory_is_not_replaced_by_face_or_depth_mean(parent, banks, memory):
    e0, e1 = memory
    real = evaluate_volume_banks(banks, np.zeros(banks.ndof),
        elastic_constant=e0, elastic_inverse_scale=e1)
    collapsed0 = np.broadcast_to((e0+parent.inverse_scale_mean[:, None, None]*e1).mean(axis=1)[:, None], e0.shape)
    collapsed = evaluate_volume_banks(banks, np.zeros(banks.ndof),
        elastic_constant=collapsed0, elastic_inverse_scale=np.zeros_like(e1))
    assert real.energy_j > collapsed.energy_j
    assert _relative(real.force, collapsed.force) > 1e-4


@pytest.mark.parametrize("component", range(6))
def test_free_duplicate_A_and_P_motion_changes_bulk_memory_and_work(banks, memory, component):
    e0, e1 = memory
    zero = evaluate_volume_banks(banks, np.zeros(banks.ndof),
        elastic_constant=e0, elastic_inverse_scale=e1)
    jump = np.zeros(banks.ndof)
    jump[banks.parent_assembly.ndof+component] = .003
    assert np.linalg.norm(banks.J@jump) == .003
    response = evaluate_volume_banks(banks, jump, elastic_constant=e0,
        elastic_inverse_scale=e1)
    assert np.linalg.norm(response.memory0-zero.memory0)+np.linalg.norm(response.memory1-zero.memory1) > 0
    # The free bank DOF assembles genuine cohort strain, not a projected
    # parent solve or an energy-free extra dummy variable.
    elastic_change = banks.K@jump
    assert np.linalg.norm(elastic_change) > 0
    assert jump@elastic_change > 0
    # Subtracting the authentic O(1e15 N) prestress obscures this much smaller
    # increment.  Test the increment tangent directly and bound the separate
    # prestressed superposition by floating roundoff at its actual force scale.
    pure = evaluate_volume_banks(banks, jump, elastic_constant=np.zeros_like(e0),
        elastic_inverse_scale=np.zeros_like(e1))
    assert _relative(pure.force, elastic_change) < 3e-15
    assert pure.energy_j == pytest.approx(.5*jump@elastic_change, rel=3e-15)
    eps = np.finfo(float).eps
    assert np.linalg.norm(response.force-zero.force-elastic_change) < 20*eps*(
        np.linalg.norm(response.force)+np.linalg.norm(zero.force))
    assert abs(response.energy_j-zero.energy_j-.5*(response.force+zero.force)@jump) < 20*eps*(
        abs(response.energy_j)+abs(zero.energy_j))


def test_tied_kinetic_energy_power_linear_and_angular_momentum(parent, basal, banks):
    v = np.random.default_rng(276).normal(size=parent.ndof)*2e-8
    w = banks.R@v
    M, D = banks.basal.mass_matrix, banks.basal.basal_drag_matrix
    assert .5*w@(M@w) == pytest.approx(.5*v@(basal.mass_matrix@v), rel=3e-15)
    assert w@(D@w) == pytest.approx(v@(basal.basal_drag_matrix@v), rel=3e-15)
    assert banks.basal.external_force_n@w == pytest.approx(basal.external_force_n@v, rel=3e-15)
    for axis in np.eye(3):
        translation = np.zeros((parent.ndof//6, 6))
        translation[:, :3] = axis
        q = translation.ravel()
        assert (banks.R@q)@(M@w) == pytest.approx(q@(basal.mass_matrix@v), rel=4e-15)
        rotation = np.cross(axis, np.asarray(parent.geometry.vertices_m))
        q = np.column_stack((rotation, rotation)).ravel()
        assert (banks.R@q)@(M@w) == pytest.approx(q@(basal.mass_matrix@v), rel=8e-15)


def test_full_cell_rigid_translation_and_rotation_remain_elastically_null(parent, banks):
    translation = np.tile(np.array((.1, -.2, .3, 0., 0., 0.)), parent.ndof//6)
    omega = np.array((2e-7, -1e-7, 4e-7))
    rotation = np.cross(omega, np.asarray(parent.geometry.vertices_m))
    rotation = np.column_stack((rotation, rotation)).ravel()
    z = np.zeros((len(parent.cohort_ids), 3, 6))
    for q in (translation, rotation):
        response = evaluate_volume_banks(banks, banks.R@q,
            elastic_constant=z, elastic_inverse_scale=z)
        assert np.max(np.abs(response.memory0))+np.max(np.abs(response.memory1)) < 2e-21


def test_empty_cuts_have_identity_R_and_no_jump_rows(parent):
    banks = prepare_volume_banks(parent, [])
    assert banks.ndof == parent.ndof
    assert banks.J.shape == (0, parent.ndof)
    np.testing.assert_array_equal(banks.R.toarray(), np.eye(parent.ndof))
    np.testing.assert_array_equal(banks.K.toarray(), parent.stiffness.toarray())
    assert banks.basal is None


def test_isolated_edge_tips_remain_tied_through_uncut_corner_fans(parent, cuts):
    banks = prepare_volume_banks(parent, cuts[:1])
    assert banks.topology.seam_count == 1
    assert banks.ndof == parent.ndof
    assert banks.J.shape == (0, parent.ndof)
    np.testing.assert_array_equal(banks.topology.bank_vertices[:, 0],
                                  banks.topology.bank_vertices[:, 1])


def test_updated_operator_requires_explicit_calculated_memory(parent, cuts, memory):
    # Same authentic geometry, altered DECLARED damage for a constitutive
    # mechanical probe; updated operator intentionally has no inherited e0.
    damage = np.full(len(parent.geometry.faces), .3)
    updated = assemble_updated_volume_operator(parent, parent.geometry, damage)
    banks = prepare_volume_banks(updated, cuts)
    assert updated.requires_explicit_memory
    with pytest.raises(ValueError, match="both complete"):
        evaluate_volume_banks(banks, np.zeros(banks.ndof))
    response = evaluate_volume_banks(banks, np.zeros(banks.ndof),
        elastic_constant=memory[0], elastic_inverse_scale=memory[1])
    assert response.energy_j > 0


@pytest.mark.parametrize("field", ("dofs", "B0", "B1", "cohort_ids", "basis_xyz", "cohort_stiffness", "float_dofs", "complex_C"))
def test_malformed_parent_locals_and_source_maps_are_refused(parent, cuts, field):
    if field == "float_dofs":
        field, value = "dofs", parent.dofs.astype(float)
    elif field == "complex_C":
        field, value = "elasticity_pa", parent.elasticity_pa.astype(complex)
    else:
        value = np.array(getattr(parent, field), copy=True)
        value.flat[0] += 1
    malformed = replace(parent, **{field: value})
    with pytest.raises(ValueError):
        prepare_volume_banks(malformed, cuts)


def test_wrong_parent_source_clock_basal_and_support_mutation_are_refused(parent, basal, banks, cuts):
    with pytest.raises(ValueError, match="source or material clock"):
        prepare_volume_banks(replace(parent, time_myr=parent.time_myr+1e-6), cuts)
    with pytest.raises(ValueError, match="basal"):
        prepare_volume_banks(parent, cuts, basal=replace(basal, source_identity="wrong"))
    with pytest.raises(ValueError, match="different parent"):
        validate_volume_banks(banks, parent_assembly=replace(parent, source_identity="wrong"))
    with pytest.raises(ValueError, match="prepared source binding"):
        evaluate_volume_banks(replace(banks, dofs=banks.dofs[:, ::-1]), np.zeros(banks.ndof),
            elastic_constant=parent.memory0, elastic_inverse_scale=parent.memory1)


@pytest.mark.parametrize("cuts", (((1., 2.),), ((True, False),), ((-1, 2),), ((0, 0),), ((0, 1), (1, 0))))
def test_malformed_or_duplicate_support_cuts_are_refused(parent, cuts):
    with pytest.raises(ValueError):
        prepare_volume_banks(parent, cuts)


def test_frozen_array_and_csr_buffers_cannot_be_made_writable(parent, banks, memory):
    response = evaluate_volume_banks(banks, np.zeros(banks.ndof),
        elastic_constant=memory[0], elastic_inverse_scale=memory[1])
    arrays = [banks.dofs, banks.face_dofs, banks.topology.parent_vertex,
        banks.topology.mesh.faces, banks.topology.mesh.vertices,
        response.memory0, response.memory1, response.force,
        banks.basal.external_force_n]
    for matrix in (banks.R, banks.J, banks.K, banks.basal.mass_matrix, banks.basal.basal_drag_matrix):
        arrays.extend((matrix.data, matrix.indices, matrix.indptr))
    for array in arrays:
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        banks.topology.mesh.faces = np.asarray(parent.geometry.faces)


@pytest.mark.parametrize("kind", ("short_u", "nan_u", "missing_e1", "short_e0", "nan_e1", "negative_beta", "complex_u", "logical_e0"))
def test_incomplete_or_nonfinite_bank_probes_are_refused(parent, banks, memory, kind):
    u, e0, e1 = np.zeros(banks.ndof), memory[0], memory[1]
    beta = None
    if kind == "short_u":
        u = u[:-1]
    elif kind == "nan_u":
        u[0] = np.nan
    elif kind == "missing_e1":
        e1 = None
    elif kind == "short_e0":
        e0 = e0[:-1]
    elif kind == "nan_e1":
        e1 = np.array(e1, copy=True)
        e1.flat[0] = np.nan
    elif kind == "complex_u":
        u = u.astype(complex)
        u[0] = 1j
    elif kind == "logical_e0":
        e0 = np.zeros(e0.shape, bool)
    else:
        beta = -np.ones(len(parent.cohort_ids))
    with pytest.raises(ValueError):
        evaluate_volume_banks(banks, u, elastic_constant=e0,
            elastic_inverse_scale=e1, strain_factors=beta)
