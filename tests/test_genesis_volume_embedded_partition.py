"""Compact finite-patch bulk enrichment: geometry, memory and virtual work.

Small cells/tensors are explicit numerical fixtures.  The planetary fixture
uses the unchanged original genuine formation export, not a mature restart.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis_volume_geometry import ConicalCohortGeometry
from tectonics.genesis_volume_material_sections import prepare_cohort_material_section
from tectonics.genesis_volume_operator import (
    ANGULAR_BARYCENTRIC, assemble_genesis_volume_operator,
    evaluate_genesis_volume_operator, isotropic_elasticity,
    tensor_to_engineering_strain,
)
from tectonics.genesis_volume_basal import assemble_genesis_volume_basal
from tectonics.genesis_volume_finite_seed import prepare_volume_finite_seed
from tectonics.genesis_volume_embedded_partition import (
    EmbeddedPartitionRefusal, prepare_cohort_embedded_partition,
    prepare_embedded_partition, prepare_volume_embedded_partition,
    evaluate_cohort_enrichment, evaluate_embedded_partition,
    evaluate_embedded_enrichment_trace, validate_volume_embedded_partition,
    volume_embedded_partition_to_dict, _inverse_birth_point, _chart,
)

X = np.array(((0., 0., 10.), (2., 0., 10.), (0., 3., 10.)))
Q = np.eye(3)
C = isotropic_elasticity(60e9, .25)
E0 = np.array(((.00011, -.00023, .00007, .00004, -.00003, .00002),
               (.00009, -.00014, .00006, .00008, -.00001, .00003),
               (.00017, -.00018, .00008, .00002, -.00005, .00004)))
E1 = np.array(((.00003, -.00002, .00001, -.00001, .00002, .00003),
               (-.00002, .00003, -.00002, .00002, -.00003, .00001),
               (.00001, -.00004, .00002, .00003, .00001, -.00002)))
U = np.array(((.03, -.02, .01, -.015, .01, .003),
              (-.01, .02, -.02, -.02, -.01, .007),
              (.02, .01, .03, .01, -.02, -.005)))


def _geometry(scale=1.):
    x = X*scale
    qlo, qhi, coefficient = .1, .8, .5
    a, b = (1-coefficient*np.array((qlo, qhi)))**(1/3)
    volume = np.linalg.det(x)*(a**3-b**3)/6
    return ConicalCohortGeometry(17, 0, 2, qlo, qhi, 2*volume, volume,
        tuple(map(tuple, x)), a, b, (), "source-owner", "original-material"), coefficient


def _fixture(warped=False, scale=1., radius=.3):
    cohort, coefficient = _geometry(scale)
    displacement = U*scale if warped else np.zeros((3, 6))
    N, lam = np.array((.4, .3, .3)), .91
    center = N@(lam*(np.asarray(cohort.top_triangle_m)+displacement[:, 3:])+displacement[:, :3]-displacement[:, 3:])
    normal = np.array((1., .3, .2)); normal /= np.linalg.norm(normal)
    section = prepare_cohort_material_section(cohort, displacement, normal, float(normal@center),
        conical_mass_coefficient=coefficient)
    return section, center, radius*scale


@pytest.fixture(scope="module")
def local():
    section, center, radius = _fixture()
    return prepare_cohort_embedded_partition(section, C, Q, E0, E1,
        seed_center_m=center, seed_radius_m=radius, quadrature_order=4)


def _material_from_reference(point, section):
    x = np.asarray(section.triangle_m)
    design = np.column_stack((x[1]-x[0], x[2]-x[0], x[0]))
    ulam, vlam, lam = np.linalg.solve(design, point)
    return np.array((1-(ulam+vlam)/lam, ulam/lam, vlam/lam)), float(lam)


def test_frozen_warped_bubble_gradient_uses_reference_jacobian():
    section, center, radius = _fixture(warped=True)
    N, lam = np.array((.39, .31, .30)), .907
    value = evaluate_cohort_enrichment(section, center, radius, Q, N, lam)
    point = value["reference_position_m"]
    step = 1e-5
    differences = []
    for axis in np.eye(3):
        plusN, plusL = _material_from_reference(point+step*axis, section)
        minusN, minusL = _material_from_reference(point-step*axis, section)
        plus = evaluate_cohort_enrichment(section, center, radius, Q, plusN, plusL)["bubble"]
        minus = evaluate_cohort_enrichment(section, center, radius, Q, minusN, minusL)["bubble"]
        differences.append((plus-minus)/(2*step))
    np.testing.assert_allclose(value["gradient_reference_m_inverse"], differences, rtol=2e-8, atol=2e-8)
    wrong = np.linalg.solve(value["birth_jacobian"].T,
                           value["reference_jacobian"].T@value["gradient_reference_m_inverse"])
    assert np.linalg.norm(wrong-value["gradient_reference_m_inverse"]) > 1e-3


@pytest.mark.parametrize("mode", range(18))
def test_all_A_P_enriched_strain_modes_match_cartesian_differentiation(mode):
    section, center, radius = _fixture(warped=True)
    N, lam = np.array((.39, .31, .30)), .907
    value = evaluate_cohort_enrichment(section, center, radius, Q, N, lam)
    point = value["reference_position_m"]
    step = 1e-6
    derivative = np.zeros((3, 3))
    for index, axis in enumerate(np.eye(3)):
        plusN, plusL = _material_from_reference(point+step*axis, section)
        minusN, minusL = _material_from_reference(point-step*axis, section)
        plus = evaluate_cohort_enrichment(section, center, radius, Q, plusN, plusL)["enrichment_trace"][:, mode]
        minus = evaluate_cohort_enrichment(section, center, radius, Q, minusN, minusL)["enrichment_trace"][:, mode]
        derivative[:, index] = (plus-minus)/(2*step)
    expected = tensor_to_engineering_strain((derivative+derivative.T)/2)
    np.testing.assert_allclose(value["enrichment_strain_operator"][:, mode], expected,
                               rtol=3e-8, atol=2e-9)


@pytest.mark.parametrize("N,lam", (((0., .5, .5), .91), ((.4, .3, .3), None)))
def test_every_original_cohort_boundary_remains_intact(N, lam):
    section, center, radius = _fixture(radius=10.)
    if lam is None:
        lam = section.lambda_bounds[0]
    value = evaluate_cohort_enrichment(section, center, radius, Q, np.array(N), lam)
    assert value["bubble"] == 0
    np.testing.assert_array_equal(value["jump_trace"], np.zeros((3, 18)))


def test_outside_seed_ligament_has_zero_jump_even_on_same_infinite_plane():
    section, center, radius = _fixture(radius=.01)
    value = evaluate_cohort_enrichment(section, center, radius, Q, np.array((.3, .3, .4)), .91)
    assert value["bubble"] == 0
    np.testing.assert_array_equal(value["enrichment_trace"], np.zeros((3, 18)))
    np.testing.assert_array_equal(value["enrichment_strain_operator"], np.zeros((6, 18)))


def test_exact_jump_is_bubble_times_full_original_A_P_trace():
    section, center, radius = _fixture()
    value = evaluate_cohort_enrichment(section, center, radius, Q, np.array((.4, .3, .3)), .91)
    # H+−H−=1, so the jump is w*T, retaining slope components.
    np.testing.assert_array_equal(value["jump_trace"], value["bubble"]*value["parent_trace"])
    assert np.linalg.norm(value["jump_trace"][:, [3, 4, 5, 9, 10, 11, 15, 16, 17]]) > 0


def test_cubic_inverse_roundtrips_frozen_heterogeneous_A_P_and_refuses_exterior():
    section, center, radius = _fixture(warped=True)
    for N, lam in ((np.array((.4, .3, .3)), .91), (np.array((.2, .1, .7)), .87)):
        point = _chart(section, N, lam)[1]
        inverse = _inverse_birth_point(section, point)
        np.testing.assert_allclose(inverse[0], N, rtol=0, atol=2e-14)
        assert inverse[1] == pytest.approx(lam, abs=2e-14)
    assert _inverse_birth_point(section, center+np.array((100., 100., 100.))) is None


def test_direct_ball_quadrature_volume_and_both_sides_are_resolved(local):
    expected = 4*np.pi*local.seed_radius_m**3/3
    assert local.reference_volume_weights_m3.sum() == pytest.approx(expected, rel=3e-14)
    assert set(local.side) == {-.5, .5}
    assert len(local.radial_scale) == 4*4**3
    assert np.trace(local.stiffness_enriched) > 0
    assert np.trace(local.mass_enriched) > 0
    assert local.numerical_enrichment_trace_rank == 18


def test_tiny_seed_is_resolved_without_sampling_million_metre_parent_cell():
    section, center, radius = _fixture(scale=1e6, radius=1e-4)
    p = prepare_cohort_embedded_partition(section, C, Q, E0, E1,
        seed_center_m=center, seed_radius_m=radius, quadrature_order=4)
    assert radius == pytest.approx(100.)
    assert p.reference_volume_weights_m3.sum() == pytest.approx(4*np.pi*100**3/3, rel=3e-11)
    assert np.trace(p.mass_enriched) > 1e4
    assert np.trace(p.stiffness_enriched) > 1e12
    assert set(p.side) == {-.5, .5}


def test_full_affine_angular_radial_memory_interpolates_without_reset(local):
    interpolation = local.barycentric@np.linalg.inv(ANGULAR_BARYCENTRIC)
    expected = interpolation@E0+(interpolation@E1)/local.radial_scale[:, None]
    np.testing.assert_array_equal(local.memory0, E0)
    np.testing.assert_array_equal(local.memory1, E1)
    np.testing.assert_allclose(local.point_elastic_strain, expected, rtol=2e-15)
    np.testing.assert_allclose(local.point_stress_pa, expected@C.T, rtol=2e-15)
    assert np.linalg.norm(local.memory1) > 0


def test_elastic_reaction_is_full_bulk_gradient_and_virtual_work(local):
    rng = np.random.default_rng(3214)
    q, a = rng.normal(size=(2, 18))*1e-5
    K = np.block([[local.parent_stiffness, local.stiffness_cross],
                  [local.stiffness_cross.T, local.stiffness_enriched]])
    F = np.r_[local.parent_force_n, local.enrichment_force_n]
    z = np.r_[q, a]
    def energy(x):
        return local.parent_energy_j+F@x+.5*x@K@x
    reaction = -(F+K@z)[18:]
    direction = rng.normal(size=18)
    step = 1e-7
    perturb = np.r_[np.zeros(18), direction*step]
    derivative = (energy(z+perturb)-energy(z-perturb))/(2*step)
    assert derivative == pytest.approx(-reaction@direction, rel=5e-11)
    strain = local.point_elastic_strain+np.einsum("pai,i->pa", local.parent_strain_operator, q)+np.einsum("pai,i->pa", local.enrichment_strain_operator, a)
    direct_added_force = np.einsum("pai,pa,p->i", local.enrichment_strain_operator, strain@C.T, local.reference_volume_weights_m3)
    np.testing.assert_allclose(-reaction, direct_added_force, rtol=2e-13, atol=1e-7)


def test_cross_block_annihilates_all_original_rigid_A_P_modes(local):
    x = local.section.triangle_m
    for axis in np.eye(3):
        translation = np.column_stack((np.tile(axis, (3, 1)), np.zeros((3, 3)))).ravel()
        rotation = np.cross(axis, x)
        rotated = np.column_stack((rotation, rotation)).ravel()
        for mode in (translation, rotated):
            assert np.linalg.norm(local.stiffness_cross.T@mode) < 2e-14*np.linalg.norm(local.stiffness_cross)*np.linalg.norm(mode)


def test_inertia_and_elasticity_have_positive_augmented_work_without_repairs(local):
    rng = np.random.default_rng(6143)
    K = np.block([[local.parent_stiffness, local.stiffness_cross],
                  [local.stiffness_cross.T, local.stiffness_enriched]])
    M = np.block([[local.parent_mass, local.mass_cross], [local.mass_cross.T, local.mass_enriched]])
    for vector in rng.normal(size=(20, 36)):
        assert vector@K@vector > 0
        assert vector@M@vector > 0
    np.testing.assert_allclose(K, K.T, atol=1e-4, rtol=2e-15)
    assert local.scaled_augmented_stiffness_minimum_eigenvalue > -1e-12


def test_ball_quadrature_enriched_bulk_force_stiffness_mass_converge():
    section, center, radius = _fixture(warped=True)
    values = [prepare_cohort_embedded_partition(section, C, Q, E0, E1,
        seed_center_m=center, seed_radius_m=radius, quadrature_order=order) for order in (4, 8, 12)]
    for name in ("stiffness_enriched", "stiffness_cross", "mass_enriched", "enrichment_force_n"):
        errors = [np.linalg.norm(getattr(p, name)-getattr(values[-1], name)) for p in values[:-1]]
        assert errors[0] > 1e-8
        assert errors[1] < errors[0]*.005


@pytest.fixture(scope="module")
def genuine_partition():
    path = Path(__file__).resolve().parents[1]/"analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json"
    packet = json.loads(path.read_text())
    assembly = assemble_genesis_volume_operator(packet)
    basal = assemble_genesis_volume_basal(assembly)
    cohort = assembly.geometry.cohorts[0]
    lam = .5*(cohort.outer_scale+cohort.inner_scale)
    center = lam*np.mean(cohort.top_triangle_m, axis=0)
    normal = assembly.basis_xyz[0]@np.array((.7, .3, .2))
    normal /= np.linalg.norm(normal)
    seed = prepare_volume_finite_seed(assembly.geometry, np.zeros(assembly.ndof), normal,
        float(normal@center), center, 100., cohort_id=cohort.id, quadrature_order=4)
    p = prepare_embedded_partition(assembly, seed, assembly.memory0, assembly.memory1,
                                 basal=basal, quadrature_order=4)
    return p, basal, seed


def test_genuine_tied_operator_exactly_retains_parent_K_M_D_and_force(genuine_partition):
    p, basal, seed = genuine_partition
    for expanded, original in ((p.K, p.parent_assembly.K), (p.M, basal.mass_matrix), (p.D, basal.basal_drag_matrix)):
        np.testing.assert_array_equal(expanded[:p.parent_ndof, :p.parent_ndof].toarray(), original.toarray())
        np.testing.assert_array_equal((p.R.T@expanded@p.R).toarray(), original.toarray())
    np.testing.assert_array_equal(p.R.T@p.external_force_n, basal.external_force_n)
    np.testing.assert_array_equal((p.J@p.R).toarray(), np.zeros((18, p.parent_ndof)))
    assert p.parent_ndof == 252
    assert len(p.memory0) == 480
    assert not p.physical_creation_admitted and not p.physical_release_admitted
    assert p.released_dof_count == 0
    assert np.linalg.norm(p.D[p.parent_ndof:].data) == 0
    np.testing.assert_array_equal(p.external_force_n[p.parent_ndof:], np.zeros(18))


def test_genuine_tied_increment_preserves_full_memory_and_energy(genuine_partition):
    p, _, _ = genuine_partition
    q = np.random.default_rng(710).normal(size=p.parent_ndof)*.01
    old = evaluate_genesis_volume_operator(p.parent_assembly, q, elastic_constant=p.memory0,
                                          elastic_inverse_scale=p.memory1)
    new = evaluate_embedded_partition(p, q)
    assert new.elastic_energy_j == old.energy_j
    np.testing.assert_array_equal(new.generalized_force_n[:p.parent_ndof], old.force)
    np.testing.assert_array_equal(new.parent_memory0, old.memory0)
    np.testing.assert_array_equal(new.parent_memory1, old.memory1)
    assert not new.computational_enrichment_probe
    np.testing.assert_array_equal(new.static_tie_reaction_n, -new.generalized_force_n[p.parent_ndof:])


def test_genuine_virtual_probe_energy_work_and_reaction_match_quadratic_operator(genuine_partition):
    p, _, _ = genuine_partition
    rng = np.random.default_rng(412)
    q, a = rng.normal(size=p.parent_ndof)*.01, rng.normal(size=18)*.001
    response = evaluate_embedded_partition(p, q, a)
    z = np.r_[q, a]
    np.testing.assert_allclose(response.generalized_force_n, p.initial_force_n+p.K@z, rtol=4e-13, atol=.1)
    assert response.elastic_energy_j-p.initial_elastic_energy_j == pytest.approx(response.finite_increment_bulk_work_j, rel=1e-9, abs=1.)
    assert response.computational_enrichment_probe
    np.testing.assert_array_equal(p.memory0, p.parent_assembly.memory0)


def test_genuine_finite_surface_jump_trace_is_fullmode_and_ligament_tapered(genuine_partition):
    p, _, seed = genuine_partition
    trace = evaluate_embedded_enrichment_trace(p, seed.surface.barycentric, seed.surface.radial_scale)
    assert trace.shape == (seed.surface.point_count, 3, 18)
    assert np.linalg.norm(trace[..., [3, 4, 5, 9, 10, 11, 15, 16, 17]]) > 0
    assert np.max(np.abs(trace)) > 0


def test_immutable_DTO_seal_rejects_changed_quadrature_or_reaction(genuine_partition):
    p, _, _ = genuine_partition
    body = volume_embedded_partition_to_dict(p)
    assert body["reaction_semantics"].startswith("negative_static_endpoint")
    assert not body["enriched_memory_representable_by_old_affine_coefficients"]
    for array in (p.memory0, p.local.enrichment_strain_operator, p.K.data, p.local.enrichment_force_n):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(EmbeddedPartitionRefusal, match="binding"):
        validate_volume_embedded_partition(replace(p, initial_elastic_energy_j=p.initial_elastic_energy_j*1.001))


@pytest.mark.parametrize("bad", (0., -1., True, float("nan")))
def test_invalid_seed_radius_is_refused(bad):
    section, center, _ = _fixture()
    with pytest.raises(ValueError):
        prepare_cohort_embedded_partition(section, C, Q, E0, E1,
            seed_center_m=center, seed_radius_m=bad, quadrature_order=4)


def test_missing_or_nonfinite_complete_memory_is_refused():
    section, center, radius = _fixture()
    for bad in (np.zeros((1, 6)), np.full((3, 6), np.nan)):
        with pytest.raises(EmbeddedPartitionRefusal, match="memory"):
            prepare_cohort_embedded_partition(section, C, Q, bad, E1,
                seed_center_m=center, seed_radius_m=radius, quadrature_order=4)


def test_sphere_center_off_declared_plane_is_refused():
    section, center, radius = _fixture()
    with pytest.raises(EmbeddedPartitionRefusal, match="center"):
        prepare_cohort_embedded_partition(section, C, Q, E0, E1,
            seed_center_m=center+.1*section.normal_xyz, seed_radius_m=radius, quadrature_order=4)


def test_literal_plane_side_is_not_silently_translated_to_rounded_sphere_center():
    original, center, radius = _fixture()
    cohort, coefficient = _geometry()
    offset = np.nextafter(center[0], np.inf)
    section = prepare_cohort_material_section(cohort, np.zeros((3, 6)), (1., 0., 0.), offset,
        conical_mass_coefficient=coefficient)
    value = evaluate_cohort_enrichment(section, center, radius, Q, np.array((.4, .3, .3)), .91)
    assert section.normal_xyz@center < section.offset_m
    assert value["side"] == -.5
    assert value["bubble"] > 0


def test_changed_parent_constitutive_operator_is_refused_by_response_binding(genuine_partition):
    p, _, _ = genuine_partition
    changed = replace(p.parent_assembly, elasticity_pa=p.parent_assembly.elasticity_pa*1.001)
    with pytest.raises(EmbeddedPartitionRefusal, match="binding"):
        validate_volume_embedded_partition(replace(p, parent_assembly=changed))


def test_raw_volume_factory_matches_same_finite_seed_and_saved_memory(genuine_partition):
    p, basal, seed = genuine_partition
    rebuilt = prepare_volume_embedded_partition(p.parent_assembly, basal, np.zeros(p.parent_ndof),
        seed.surface.declared_normal_xyz, seed.surface.declared_offset_m,
        cohort_id=seed.selected_cohort_id, seed_center_m=seed.seed_center_world_m,
        seed_radius_m=seed.seed_radius_m, elastic_constant=p.memory0, elastic_inverse_scale=p.memory1,
        quadrature_order=4)
    assert rebuilt.binding_sha256 == p.binding_sha256


def test_genuine_large_world_small_ball_preserves_weak_signed_AA_mass_convergence():
    """Read the accepted source leaf; never restore/replay or accept a state.

    The old float64 world subtraction produced a nonzero 4e-5 relative
    order-eight/twelve error in this weak AA block. Its actual mass is kept;
    a large positive AA Gram scale must not hide its own convergence error.
    """
    root = Path(__file__).resolve().parents[1]
    parent = json.loads((root/"analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json").read_bytes())
    prior = json.loads((root/"analysis/genesis_volume_location_validation/evidence/actual_control/supplied_current_plane_face_0.json").read_bytes())
    geometry = parent["runtime_state"]["geometry"]
    raw = geometry["cohorts"][0]
    cohort = ConicalCohortGeometry(**{**raw,
        "top_triangle_m": tuple(map(tuple, raw["top_triangle_m"])),
        "tetrahedra_m": tuple(tuple(tuple(v) for v in tetra) for tetra in raw["tetrahedra_m"])})
    vertices = np.asarray(geometry["faces"][raw["face_index"]], int)
    pose = np.asarray(parent["runtime_state"]["displacement_m"]).reshape(-1, 6)[vertices]
    triangle = np.asarray(raw["top_triangle_m"])
    N, lam = np.full(3, 1/3.), .5*(raw["inner_scale"]+raw["outer_scale"])
    center = N@(lam*(triangle+pose[:, 3:])+pose[:, :3]-pose[:, 3:])
    normal = np.asarray(prior["normal_xyz"])
    radius = .12*(raw["outer_scale"]-raw["inner_scale"])*np.linalg.norm(N@(triangle+pose[:, 3:]))
    coefficient = 6*geometry["column_mass_kg"][0]/(geometry["density_kg_m3"]*np.linalg.det(triangle.T))
    section = prepare_cohort_material_section(cohort, pose, normal, float(normal@center),
        conical_mass_coefficient=coefficient, cohort_index=0)
    shell = parent["source_packet"]["source"]["parameters"]["shell"]
    damage = parent["runtime_state"]["fault_history"]["damage"][0]
    elasticity = (shell["residual_stiffness"]+(1-shell["residual_stiffness"])*(1-damage)**2)*isotropic_elasticity(shell["young_modulus_pa"], shell["poisson_ratio"])
    basis = np.asarray(parent["source_packet"]["instantaneous_cohort_plane_stress_3d"]["basis_xyz"])[0]
    memory = parent["runtime_state"]["mechanics"]["state"]["cohorts"][0]
    values = [prepare_cohort_embedded_partition(section, elasticity, basis,
        memory["strain0"], memory["strain1"], seed_center_m=center, seed_radius_m=radius,
        quadrature_order=order) for order in (8, 12)]
    A = np.array([6*v+axis for v in range(3) for axis in range(3)])
    P = A+3
    errors = []
    for row in (A, P):
        for column in (A, P):
            selection = np.ix_(row, column)
            first, final = (value.mass_cross[selection] for value in values)
            errors.append(np.linalg.norm(first-final)/max(np.linalg.norm(final), np.finfo(float).tiny))
    weak = values[-1].mass_cross[np.ix_(A, A)]
    positive = values[-1].mass_enriched[np.ix_(A, A)]
    assert np.linalg.norm(weak) > .01
    assert np.linalg.norm(weak) < 1e-5*np.linalg.norm(positive)
    assert max(errors) <= 2e-6, errors
    assert len(values[-1].barycentric) == 4*12**3
    assert all(array.dtype == np.dtype(float) for array in (
        values[-1].barycentric, values[-1].radial_scale, values[-1].bubble,
        values[-1].enrichment_trace, values[-1].reference_volume_weights_m3,
        values[-1].mass_cross, values[-1].enrichment_force_n))
