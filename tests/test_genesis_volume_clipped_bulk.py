"""Regular-field, memory and own-scale checks on declared clipped cells."""
from dataclasses import replace

import numpy as np
import pytest

from tectonics.genesis_volume_geometry import ConicalCohortGeometry
from tectonics.genesis_volume_material_sections import prepare_cohort_material_section
from tectonics.genesis_volume_clipped_domain import prepare_cohort_clipped_domain
from tectonics.genesis_volume_operator import isotropic_elasticity, _immutable
from tectonics.genesis_volume_embedded_partition import _cohort_enrichment_fields
from tectonics.genesis_volume_clipped_bulk import (
    ClippedBulkRefusal, _agreement,
    _seal, _integrate, clipped_bulk_contractions, clipped_bulk_operator_to_dict,
    compare_clipped_bulk_orders, memory_cross_force, prepare_clipped_bulk_operator,
    validate_clipped_bulk_operator, clipped_bulk_projected_blocks,
)

LD = np.longdouble
X = np.array(((0., -5., 10.), (5., -5., 10.), (0., 5., 10.)))


def _section(displacement=None):
    lo, hi, coefficient = .8, .95, .5
    volume = np.linalg.det(X)*(hi**3-lo**3)/6
    cohort = ConicalCohortGeometry(17, 0, 2, (1-hi**3)/coefficient,
        (1-lo**3)/coefficient, 2*volume, volume, tuple(map(tuple, X)),
        hi, lo, (), "unchanged-owner", "unchanged-material")
    return prepare_cohort_material_section(cohort,
        np.zeros((3, 6)) if displacement is None else displacement,
        (0., 1., 0.), 0., conical_mass_coefficient=coefficient)


def _op(order=4, center=(1., 0., 9.), e0=None, e1=None, displacement=None):
    section = _section(displacement)
    domain = prepare_cohort_clipped_domain(section, center, .25,
        radial_order=order, angular_order=order, ray_order=order)
    e0 = (np.arange(18).reshape(3, 6)-8)*1e-4 if e0 is None else e0
    e1 = (np.arange(18).reshape(3, 6)[::-1]-7)*2e-4 if e1 is None else e1
    return prepare_clipped_bulk_operator(domain, isotropic_elasticity(3e8, .27), np.eye(3), e0, e1)


@pytest.fixture(scope="module")
def op():
    return _op()


def test_point_fields_match_original_regular_enrichment_with_centered_offsets(op):
    indices = np.argsort(op.bubble)[-5:]
    for p in indices:
        reference = _cohort_enrichment_fields(op.domain.section,
            np.asarray(op.domain.seed_center_world_m, LD), LD(op.domain.seed_radius_m),
            op.basis_xyz, op.domain.barycentric[p], op.domain.radial_scale[p],
            birth_relative_m=op.domain.centered_birth_offsets_m[p])
        for name in ("parent_trace", "enrichment_trace", "parent_strain_operator", "enrichment_strain_operator"):
            np.testing.assert_allclose(getattr(op, name)[p], reference[name], rtol=3e-13, atol=2e-15)
        np.testing.assert_allclose(op.bubble_gradient_reference_m_inverse[p],
            reference["gradient_reference_m_inverse"], rtol=4e-13, atol=2e-15)


def _bubble_at_reference(op, point):
    # Recover N,lambda directly from the exact linear conical reference map.
    co = np.asarray(np.linalg.solve(X.T, np.asarray(point, float)), LD)
    lam, N = co.sum(), co/co.sum()
    section = op.domain.section
    birth = N@(lam*np.asarray(section.triangle_m+section.slope_P_m, LD)+
               np.asarray(section.displacement_A_m-section.slope_P_m, LD))
    offset = birth-np.asarray(op.domain.seed_center_world_m, LD)
    b, a = section.lambda_bounds
    t = (lam-b)/(a-b)
    return (1-(offset@offset)/op.domain.seed_radius_m**2)**2*27*np.prod(N)*4*t*(1-t)


def test_gradient_is_derivative_with_respect_to_reference_cartesian_metres(op):
    p = int(np.argmax(op.bubble))
    xr = op.domain.radial_scale[p]*(op.domain.barycentric[p]@np.asarray(X, LD))
    h = 1e-6
    difference = np.array([(_bubble_at_reference(op, xr+h*axis)-
        _bubble_at_reference(op, xr-h*axis))/(2*h) for axis in np.eye(3)], LD)
    np.testing.assert_allclose(op.bubble_gradient_reference_m_inverse[p], difference, rtol=2e-7, atol=2e-9)


def test_full_angular_normal_and_shear_memory_is_inherited_without_reset(op):
    from tectonics.genesis_volume_operator import ANGULAR_BARYCENTRIC
    angular = op.domain.barycentric@np.asarray(np.linalg.inv(ANGULAR_BARYCENTRIC), LD)
    expected = angular@np.asarray(op.memory0, LD)+(angular@np.asarray(op.memory1, LD))/op.domain.radial_scale[:, None]
    np.testing.assert_array_equal(op.point_elastic_strain, expected)
    np.testing.assert_array_equal(op.point_stress_pa, expected@np.asarray(op.elasticity_pa, LD).T)
    assert np.any(expected[:, :3] != 0)
    assert np.any(expected[:, 3:] != 0)
    assert np.ptp(expected[:, 5]) > 0


def test_parent_mass_retains_whole_original_cohort_and_physical_metre_units(op):
    translation = np.tile((1., 2., -1., 0., 0., 0.), 3)
    assert translation@op.parent_mass@translation == pytest.approx(6*op.domain.section.mass_kg, rel=2e-15)
    # A clipped support does not replace the old QQ mass by its ball mass.
    assert np.trace(op.parent_mass) > 1000*np.trace(op.mass_enriched)


def test_enriched_mass_bilinear_matches_point_kinetic_energy_and_density(op):
    a = np.cos(.31*(np.arange(18)+1))*2e-3
    v = np.einsum("pai,i->pa", op.enrichment_trace, np.asarray(a, LD))
    point = LD(op.domain.section.mass_kg)/LD(op.domain.section.volume_m3)*np.sum(
        op.domain.reference_volume_weights_m3*np.sum(v*v, axis=1))
    value = np.asarray(a, LD)@op.mass_enriched@np.asarray(a, LD)
    assert abs(value-point) <= LD(2e-17)*abs(point)
    assert value > 0


def test_stiffness_work_adjoint_and_full_point_stress_cross_force(op):
    q, a = (np.asarray(v, LD) for v in
        (1e-4*np.sin(.37*(np.arange(18)+1)), 2e-4*np.cos(.23*(np.arange(18)+1))))
    Bg, Ga = np.einsum("pai,i->pa", op.parent_strain_operator, q), np.einsum("pai,i->pa", op.enrichment_strain_operator, a)
    point = np.einsum("pa,ab,pb,p->", Bg, np.asarray(op.elasticity_pa, LD), Ga, op.domain.reference_volume_weights_m3)
    matrix = q@op.stiffness_cross@a
    budget = np.abs(q)@op.stiffness_cross_budget@np.abs(a)
    assert abs(point-matrix) <= budget
    assert q@op.stiffness_cross@a == pytest.approx(float(a@op.stiffness_cross.T@q), rel=2e-15)
    pointforce = np.einsum("pai,pa,p->i", op.enrichment_strain_operator, op.point_stress_pa, op.domain.reference_volume_weights_m3)
    assert np.all(np.abs(pointforce-op.enrichment_force_n) <= op.enrichment_force_n_budget)


def test_cross_force_retains_every_inherited_memory_component_and_sign(op):
    positive = memory_cross_force(op, op.memory0, op.memory1)
    negative = memory_cross_force(op, -op.memory0, -op.memory1)
    np.testing.assert_array_equal(positive["force_n"], op.enrichment_force_n)
    np.testing.assert_array_equal(negative["force_n"], -positive["force_n"])
    np.testing.assert_array_equal(negative["absolute_roundoff_budget_n"], positive["absolute_roundoff_budget_n"])
    zero = memory_cross_force(op, np.zeros((3, 6)), np.zeros((3, 6)))
    assert np.all(zero["force_n"] == 0)
    assert np.all(zero["absolute_roundoff_budget_n"] == 0)


def test_work_parts_do_not_subtract_large_parent_energy_and_preserve_signed_cross(op):
    q = 1e-4*np.sin(.37*(np.arange(18)+1))
    a = 2e-4*np.cos(.23*(np.arange(18)+1))
    z = 3e-4*np.sin(.19*(np.arange(18)+1))
    forward = clipped_bulk_contractions(op, q, a, mean_z_m=z)
    reverse = clipped_bulk_contractions(op, -q, -a, mean_z_m=z)
    for name in ("parent_work_j", "cross_parent_work_j", "cross_added_work_j", "self_work_j"):
        assert reverse["terms"][name]["value"] == -forward["terms"][name]["value"]
        assert forward["terms"][name]["absolute_roundoff_budget"] >= 0
    assert "energy_difference_j" not in forward
    assert forward["available_for_creation_funding"] is False


def test_zero_declared_fields_have_exact_zero_cross_self_and_budgets(op):
    values = clipped_bulk_contractions(op, np.zeros(18), np.zeros(18), mean_z_m=np.zeros(18))
    assert all(part["value"] == part["absolute_roundoff_budget"] == 0 for part in values["terms"].values())


def test_own_component_comparison_rejects_weak_PP_error_hidden_by_large_AA_norm():
    lo = np.zeros((18, 18), LD)
    lo[0, 0], lo[3, 3] = LD("1e20"), LD("1e-18")
    hi = lo.copy()
    hi[3, 3] = LD("2e-18")
    zero = np.zeros_like(lo)
    mask, report = _agreement(lo, hi, zero, zero, LD("1e-5"))
    assert mask[0, 0] and not mask[3, 3]
    assert report["failed_component_count"] == 1
    assert np.linalg.norm(np.asarray(hi-lo, float))/np.linalg.norm(np.asarray(lo, float)) < 1e-30
    assert not report["positive_scale_floor_used"]


def test_exact_zero_component_has_no_positive_scale_floor():
    mask, report = _agreement(np.zeros(1, LD), np.array([LD("1e-200")]),
        np.zeros(1, LD), np.zeros(1, LD), LD("1e-5"))
    assert not mask[0] and not report["agreed"]


def test_signed_reduction_budget_is_owned_by_each_actual_operand():
    L = np.array([[[1., 0.]], [[-1., 0.]]], LD)
    R = np.array([[[1., 0.]], [[1., 0.]]], LD)
    value, budget = _integrate(L, np.eye(1, dtype=LD), R, np.ones(2, LD))
    assert value[0, 0] == 0 and budget[0, 0] > 0
    assert np.all(value[1:] == 0) and np.all(budget[1:] == 0)
    assert np.all(value[:, 1:] == 0) and np.all(budget[:, 1:] == 0)


def test_finite_order_comparison_retains_all_AP_blocks_and_explicit_probes(op):
    higher = _op(order=6)
    probe = dict(name="declared18", parent_increment_m=(1e-4*np.sin(np.arange(18))).tolist(),
        enrichment_increment_m=(2e-4*np.cos(np.arange(18))).tolist(), mean_z_m=(1e-4*np.ones(18)).tolist())
    comparison = compare_clipped_bulk_orders((op, higher), relative_tolerance=1e-5, probes=(probe,))
    pair = comparison["pair_results"][0]
    assert set(pair["components"]["stiffness_enriched"]["A_P_blocks"]) == {"AA", "AP", "PA", "PP"}
    assert pair["declared_probe_results"][0]["name"] == "declared18"
    assert "self_work_j" in pair["declared_probe_results"][0]["contractions"]
    assert not comparison["physical_contraction_certified"]
    assert not comparison["arbitrary_contraction_certified"]
    assert not comparison["available_for_creation_funding"]


def test_source_changed_between_orders_is_refused(op):
    higher = _op(order=6, e1=op.memory1*1.01)
    with pytest.raises(ClippedBulkRefusal, match="identical original geometry and source"):
        compare_clipped_bulk_orders((op, higher))


def test_literal_archive_and_live_precision_have_distinct_seals(op):
    dto = clipped_bulk_operator_to_dict(op, include_points=False)
    assert "parent_trace" not in dto and dto["points_included"] is False
    assert "barycentric" in dto["domain"]
    assert dto["binding_sha256"] != dto["live_precision_binding_sha256"]
    assert dto["binding_sha256"] != dto["operator_binding_sha256"]
    body = dict(dto)
    seal = body.pop("binding_sha256")
    assert _seal(body) == seal
    assert dto["added_coordinate_count"] == 18 and dto["added_coordinates_tied"]
    assert dto["accepted_time_advance_myr"] == dto["created_material_kg"] == dto["work_debit_j"] == 0
    points = clipped_bulk_operator_to_dict(op)
    assert points["parent_trace"] and points["point_elastic_strain"]
    assert points["domain_geometry_resolved"] is True
    assert points["finite_order_agreement_certified"] is False


def test_arrays_are_immutable_and_LD_low_bits_are_bound(op):
    for name in ("parent_trace", "stiffness_cross", "memory0"):
        with pytest.raises(ValueError):
            getattr(op, name).setflags(write=True)
    changed = np.array(op.enrichment_force_n, LD)
    index = int(np.argmax(np.abs(changed)))
    changed[index] = np.nextafter(changed[index], LD(np.inf))
    assert float(changed[index]) == float(op.enrichment_force_n[index])
    forged = replace(op, enrichment_force_n=_immutable(changed, LD))
    with pytest.raises(ClippedBulkRefusal, match="immutable live/source binding"):
        validate_clipped_bulk_operator(forged)


@pytest.mark.parametrize("bad", (True, np.ones((2, 6)), np.full((3, 6), np.nan)))
def test_partial_boolean_or_nonfinite_memory_is_refused(op, bad):
    with pytest.raises(ClippedBulkRefusal):
        prepare_clipped_bulk_operator(op.domain, op.elasticity_pa, op.basis_xyz, bad, op.memory1)


@pytest.mark.parametrize("bad", (True, 0., float("nan"), 1.))
def test_invalid_convergence_tolerance_is_refused(op, bad):
    with pytest.raises(ClippedBulkRefusal):
        compare_clipped_bulk_orders((op, op), relative_tolerance=bad)


def test_no_source_memory_mutation_or_added_memory_initialization(op):
    before = (op.domain.section.binding_sha256, op.memory0.tobytes(), op.memory1.tobytes())
    result = clipped_bulk_contractions(op, np.ones(18)*1e-4, np.ones(18)*1e-4)
    assert before == (op.domain.section.binding_sha256, op.memory0.tobytes(), op.memory1.tobytes())
    assert not op.physical_creation_admitted and not op.physical_release_admitted
    assert result["physical_contraction_certified"] is False


def test_declared_operative_basis_is_projected_before_Gram_and_preserves_full_fields(op):
    basis = np.zeros((18, 3))
    basis[0, 0], basis[3, 0] = 1., 8.
    basis[7, 1], basis[10, 1] = 1., 8.
    basis[14, 2], basis[17, 2] = 1., 8.
    projection = clipped_bulk_projected_blocks(op, basis)
    assert projection["projection_before_Gram"]
    assert projection["stiffness_enriched"].shape == (3, 3)
    assert projection["stiffness_cross"].shape == (18, 3)
    assert not projection["physical_contraction_certified"]
    point = np.einsum("pai,ij->paj", op.enrichment_strain_operator, np.asarray(basis, LD))
    expected = np.einsum("pai,ab,pbj,p->ij", point, np.asarray(op.elasticity_pa, LD), point,
                          op.domain.reference_volume_weights_m3)
    assert np.all(np.abs(projection["stiffness_enriched"]-expected) <= projection["stiffness_enriched_budget"])
    assert op.enrichment_strain_operator.shape[-1] == 18


def test_identity_projection_matches_literal_blocks_with_own_projection_budgets(op):
    projection = clipped_bulk_projected_blocks(op, np.eye(18))
    for name in ("stiffness_cross", "stiffness_enriched", "mass_cross", "mass_enriched", "enrichment_force_n"):
        assert np.all(np.abs(projection[name]-getattr(op, name)) <= projection[name+"_budget"]+getattr(op, name+"_budget"))


def test_point_work_and_force_adjoints_agree_without_large_energy_subtraction(op):
    a, q, z = (np.sin((np.arange(18)+1)*factor)*1e-4 for factor in (.23, .37, .19))
    contraction = clipped_bulk_contractions(op, q, a, mean_z_m=z)
    total = contraction["terms"]["cross_added_work_j"]["value"]+contraction["terms"]["self_work_j"]["value"]
    forcework = a@np.asarray(contraction["enrichment_force_n"])
    budget = sum(contraction["terms"][name]["absolute_roundoff_budget"] for name in ("cross_added_work_j", "self_work_j"))
    budget += np.abs(a)@np.asarray(contraction["enrichment_force_budget_n"])
    assert abs(total-forcework) <= 4*budget
    assert contraction["projection_before_Gram"]


def test_finite_order_comparison_reports_declared_weak_projected_components(op):
    higher = _op(order=6)
    basis = np.eye(18)[:, [0, 3, 7]]
    comparison = compare_clipped_bulk_orders((op, higher), added_basis=basis)
    components = comparison["pair_results"][0]["declared_projected_components"]
    assert components["stiffness_enriched"]["component_count"] == 9
    assert components["stiffness_cross"]["component_count"] == 54
    assert comparison["projection_before_Gram"]
    assert not comparison["arbitrary_contraction_certified"]


@pytest.mark.parametrize("basis", (np.eye(17), np.ones((18, 19)), np.ones(18), np.full((18, 3), np.nan)))
def test_unbounded_or_invalid_operative_basis_is_refused(op, basis):
    with pytest.raises(ClippedBulkRefusal):
        clipped_bulk_projected_blocks(op, basis)


def test_duplicate_orders_do_not_supply_false_convergence_evidence(op):
    with pytest.raises(ClippedBulkRefusal, match="increase at least one"):
        compare_clipped_bulk_orders((op, op))
