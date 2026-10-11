"""Independent direct-nodal derivatives and component-scale forgery refusals."""
from copy import deepcopy

import numpy as np
import pytest

from analysis.genesis_volume_clipped_validation.independent_bulk_audit import (
    IndependentBulkRefusal, audit_bulk_operator, independent_point_fields, own_component_check,
    independently_check_pair, audit_case)
from tectonics.genesis_volume_clipped_bulk import prepare_clipped_bulk_operator, clipped_bulk_operator_to_dict
from tectonics.genesis_volume_clipped_domain import prepare_cohort_clipped_domain
from tectonics.genesis_volume_geometry import ConicalCohortGeometry
from tectonics.genesis_volume_material_sections import prepare_cohort_material_section
from tectonics.genesis_volume_operator import isotropic_elasticity


@pytest.fixture(scope="module")
def archive():
    X = np.array(((0., 0., 10.), (2., 0., 10.), (0., 3., 10.)))
    qlo, qhi, coefficient = .1, .8, .5
    a, b = (1-coefficient*np.array((qlo, qhi)))**(1/3)
    volume = np.linalg.det(X)*(a**3-b**3)/6
    cohort = ConicalCohortGeometry(17, 0, 2, qlo, qhi, 2*volume, volume,
        tuple(map(tuple, X)), a, b, (), "explicit-source", "explicit-material")
    displacement = np.array(((.03, -.02, .01, -.015, .01, .003),
        (-.01, .02, -.02, -.02, -.01, .007), (.02, .01, .03, .01, -.02, -.005)))
    center = np.array((.4, .3, .3)) @ (.91*(X+displacement[:, 3:])+displacement[:, :3]-displacement[:, 3:])
    normal = np.array((1., .3, .2)); normal /= np.linalg.norm(normal)
    section = prepare_cohort_material_section(cohort, displacement, normal, float(normal@center),
        conical_mass_coefficient=coefficient)
    domain = prepare_cohort_clipped_domain(section, center, .9,
        radial_order=3, angular_order=3, ray_order=3, require_both_sides=True)
    C = isotropic_elasticity(60e9, .25)
    e0 = np.array(((.00011, -.00023, .00007, .00004, -.00003, .00002),
        (.00009, -.00014, .00006, .00008, -.00001, .00003),
        (.00017, -.00018, .00008, .00002, -.00005, .00004)))
    e1 = np.array(((.00003, -.00002, .00001, -.00001, .00002, .00003),
        (-.00002, .00003, -.00002, .00002, -.00003, .00001),
        (.00001, -.00004, .00002, .00003, .00001, -.00002)))
    return clipped_bulk_operator_to_dict(prepare_clipped_bulk_operator(domain, C, np.eye(3), e0, e1))


def test_direct_nodal_gradient_reconstructs_all_full_A_P_source_fields(archive):
    result = audit_bulk_operator(archive)
    assert result["status"] == "PASS"
    assert result["matrix_components_independently_recomputed"]
    assert result["retained_source_memory_force_verified"]
    assert set(result["complete_source_point_fields_checked"]) == {
        "parent_strain_operator", "enrichment_strain_operator", "parent_trace",
        "enrichment_trace", "point_elastic_strain"}
    assert not result["physical_contraction_accuracy_certified"]


@pytest.mark.parametrize("name", ["stiffness_cross", "stiffness_enriched", "mass_cross", "mass_enriched", "enrichment_force_n"])
def test_corrupted_own_component_refused_without_using_strong_block(archive, name):
    forged = deepcopy(archive)
    value = np.asarray(forged[name]); index = np.unravel_index(np.abs(value).argmax(), value.shape)
    value[index] *= 1.01
    forged[name] = value.tolist()
    with pytest.raises(IndependentBulkRefusal, match="own component differs"):
        audit_bulk_operator(forged)


def test_billion_times_larger_neighbour_cannot_hide_small_component_error():
    expected = np.array((1e20, 1e-12, 0.))
    actual = expected.copy(); actual[1] += 1e-13
    with pytest.raises(IndependentBulkRefusal, match="own component differs"):
        own_component_check(actual, expected, np.abs(expected), operations=64)


def test_exact_zero_has_no_positive_absolute_allowance():
    with pytest.raises(IndependentBulkRefusal, match="own component differs"):
        own_component_check([1e-300], [0.], [0.], relative_tolerance=0.)


def test_retained_shear_and_depth_memory_cannot_be_replaced_by_zero(archive):
    forged = deepcopy(archive); forged["memory0"] = np.zeros((3, 6)).tolist()
    forged["memory1"] = np.zeros((3, 6)).tolist()
    with pytest.raises(IndependentBulkRefusal, match="enrichment_force_n own component differs"):
        audit_bulk_operator(forged)


def test_source_slopes_survive_direct_reference_differentiation(archive):
    fields = independent_point_fields(archive["domain"], archive["elasticity_pa"],
        archive["basis_xyz"], archive["memory0"], archive["memory1"])
    assert np.linalg.norm(fields["parent_trace"][:, :, [3,4,5,9,10,11,15,16,17]]) > 0
    assert np.linalg.norm(fields["enrichment_strain_operator"][:, :, [3,4,5,9,10,11,15,16,17]]) > 0


def test_inflated_roundoff_budget_cannot_admit_finite_order_error(archive):
    forged = deepcopy(archive)
    forged["roundoff_component_bounds"]["stiffness_enriched"][0][0] = 1e30
    with pytest.raises(IndependentBulkRefusal, match="Inflated or invalid live component"):
        audit_bulk_operator(forged)


def test_forged_finite_order_summary_cannot_hide_small_signed_component():
    names = ("stiffness_cross","stiffness_enriched","mass_cross","mass_enriched","enrichment_force_n")
    blocks, budgets, components = {}, {}, {}
    for name in names:
        shape = (18,) if name == "enrichment_force_n" else (18,18)
        blocks[name] = np.ones(shape).tolist()
        budgets[name] = np.zeros(shape).tolist()
        report = dict(agreed=True,failed_component_count=0,own_component_relative_scale=True,
                      positive_scale_floor_used=False)
        if len(shape) == 2:
            report["A_P_blocks"] = {label:dict(agreed=True,failed_component_count=0)
                                    for label in ("AA","AP","PA","PP")}
        components[name] = report
    left = dict(bulk_operator=dict(blocks,roundoff_component_bounds=budgets))
    right = deepcopy(left)
    left["bulk_operator"]["stiffness_cross"][0][0] = 1e20
    right["bulk_operator"]["stiffness_cross"][0][0] = 1e20
    left["bulk_operator"]["stiffness_cross"][1][1] = 1e-12
    right["bulk_operator"]["stiffness_cross"][1][1] = 1.1e-12
    pair = dict(components=components,finite_order_agreement=True,declared_probe_results=[])
    with pytest.raises(IndependentBulkRefusal,match="Saved own component finite comparison differs"):
        independently_check_pair(left,right,pair,2e-6)


@pytest.mark.parametrize("primary,baseline,error", [
    (None,None,"Required fixed 100 m support"),
    ({},None,"Required original-radius baseline"),
    ({"orders":[],"fixed_orders":[4,8],"required_final_orders":[4,8],
      "seed_radius_m":100.,"relative_tolerance_fixed_before_execution":2e-6},{},"Fixed source radius/orders")])
def test_missing_or_shortened_required_support_cannot_pass_vacuously(primary,baseline,error):
    means = np.zeros((3,6)).tolist()
    case = dict(source_inputs=dict(section=dict(cohort_id=1),accepted_interval_mean_e0=means,
        accepted_interval_mean_e1=means), declared_probes=[dict(name=n) for n in
        ("mixed_A_P","A_only","P_only","small_mixed","saved_accepted_interval_mean")],
        fixed_100m_clipped_support=primary,original_radius_baseline=baseline)
    case["declared_probes"][-1].update(mean_e0=means,mean_e1=means)
    with pytest.raises(IndependentBulkRefusal,match=error):
        audit_case(case)
