"""Analytic domains and resealed-style geometric forgeries for the new oracle.

No planetary checkpoint is restored and no tectonics producer is imported.
"""
from copy import deepcopy
import math

import numpy as np
import pytest

from analysis.genesis_volume_clipped_validation.independent_geometry_audit import (
    CONTRACTION_NAMES, MOMENT_NAMES, IndependentGeometryRefusal,
    _angular_support, _cell_bounds, _probe_field, analytic_ball_moments,
    analytic_cap_volume, audit_archived_geometry, integrate_radial_lines,
    make_contraction_evaluator, radial_intervals, read_source,
)


def source(**changes):
    value = dict(triangle_m=(np.eye(3)*10).tolist(),
                 displacement_A_m=np.zeros((3, 3)).tolist(),
                 slope_P_m=np.zeros((3, 3)).tolist(), lambda_bounds=[.1, .9],
                 normal_xyz=[0., 0., 1.], offset_m=2.,
                 seed_center_world_m=[2., 2., 2.], seed_radius_m=.3)
    value.update(changes)
    return value


def test_radial_quadratic_and_literal_gamma_have_exact_half_diameters():
    intervals = radial_intervals(source(), [1/3]*3)
    half = .3/(10/math.sqrt(3))
    np.testing.assert_allclose(intervals[-1], [.6-half, .6], rtol=0, atol=3e-16)
    np.testing.assert_allclose(intervals[1], [.6, .6+half], rtol=0, atol=3e-16)


def test_gamma_uses_literal_offset_instead_of_translating_to_center():
    value = source(offset_m=2.01)
    intervals = radial_intervals(value, [1/3]*3)
    assert intervals[-1][1] == pytest.approx(.603, abs=2e-16)
    assert intervals[1][0] == pytest.approx(.603, abs=2e-16)


def test_original_radial_bounds_clip_sphere_before_quadrature():
    intervals = radial_intervals(source(lambda_bounds=[.59, .61]), [1/3]*3)
    assert intervals[-1][0] == pytest.approx(.59)
    assert intervals[1][1] == pytest.approx(.61)


def test_tangent_radial_line_has_no_positive_length_and_nearby_cap_is_positive():
    # Point (2,2,2) with sphere displaced normal to its radial line.
    direction = np.array((1., -1., 0.))/math.sqrt(2)
    value = source(seed_center_world_m=(np.array((2., 2., 2.))+direction*.3).tolist())
    # A clearly exterior line must never gain a numerical-threshold radius.
    value["seed_center_world_m"] = (np.array((2., 2., 2.))+direction*(.3+1e-9)).tolist()
    assert all(v is None for v in radial_intervals(value, [1/3]*3).values())
    value["seed_center_world_m"] = (np.array((2., 2., 2.))+direction*(.3-1e-9)).tolist()
    assert any(v is not None for v in radial_intervals(value, [1/3]*3).values())


def test_positive_sliver_is_retained_when_coarse_barycentric_center_misses_it():
    raw = source(seed_center_world_m=[.001, .001, 4.], offset_m=4., seed_radius_m=.01)
    s = read_source(raw)
    assert all(v is None for v in radial_intervals(s, [1/3]*3).values())
    bounds = _cell_bounds(s, np.eye(3))
    assert sum(v[1] for v in bounds.values()) > 0
    polygon = _angular_support(s)
    assert polygon
    assert sum(float(abs(np.linalg.det(np.asarray(np.column_stack((v[:, 1:], np.ones(3))), float))))/2 for v in polygon) > 0
    with pytest.raises(IndependentGeometryRefusal, match="max_cells|no resolved positive"):
        integrate_radial_lines(raw, max_cells=1, relative_tolerance=.01)


def test_true_miss_is_discarded_by_conservative_corner_distance():
    bounds = _cell_bounds(read_source(source(seed_center_world_m=[-10., -10., -10.])), np.eye(3))
    assert bounds == {-1: (0., 0.), 1: (0., 0.)}


@pytest.fixture(scope="module")
def contained_ball_report():
    return integrate_radial_lines(source(), relative_tolerance=.08, max_cells=2400,
                                  angular_orders=(2, 3), radial_order=3)


def test_independent_radial_route_matches_analytic_ball_and_halfball_moments(contained_ball_report):
    result = np.asarray(contained_ball_report["values"])
    golden = analytic_ball_moments(.3)
    volume = golden["volume_m3"]
    np.testing.assert_allclose(result[:, :2], volume/2, rtol=8e-4, atol=0)
    np.testing.assert_allclose(result[:, 5:8], volume/10, rtol=1.2e-3, atol=0)
    np.testing.assert_allclose(result[:, 4], [-math.pi*.3**3/4, math.pi*.3**3/4], rtol=8e-4, atol=0)
    assert np.max(np.abs(result[:, [2, 3, 8, 9, 10]])) < 5e-4*volume
    lower = np.asarray(contained_ball_report["reference_volume_envelope_lower_m3"])
    upper = np.asarray(contained_ball_report["reference_volume_envelope_upper_m3"])
    assert np.all(lower <= volume/2) and np.all(upper >= volume/2)


def test_finite_order_agreement_never_becomes_physical_certificate(contained_ball_report):
    assert contained_ball_report["positive_branches_conservatively_discovered"]
    assert contained_ball_report["independent_numerical_convergence_verified"]
    for name in ("general_contraction_accuracy_certified", "physical_creation_admitted",
                 "physical_release_admitted", "conditioning_or_roundoff_available_for_funding"):
        assert contained_ball_report[name] is False


def test_small_positive_cap_formula_has_no_large_energy_or_length_floor():
    radius, height = 1e3, 1e-8
    expected = math.pi*height*height*(radius-height/3)
    assert analytic_cap_volume(radius, height) == expected
    assert 0 < expected < 1e-10
    assert analytic_cap_volume(radius, 0) == 0
    assert analytic_cap_volume(radius, 2*radius) == pytest.approx(4*math.pi*radius**3/3)


def ball_dto():
    raw = source()
    nodes, weights = np.polynomial.legendre.leggauss(4)
    g, gw = (nodes+1)/2, weights/2
    an, aw = np.polynomial.legendre.leggauss(12)
    theta, tw = math.pi*(an+1), math.pi*aw
    rows = []
    for side in (-1., 1.):
        for z, zw in zip(g*.3*side, gw*.3):
            radius2 = .3**2-z*z
            for s, sw in zip(g, gw):
                rho = math.sqrt(radius2)*s
                for angle, angleweight in zip(theta, tw):
                    offset = np.array((rho*math.cos(angle), rho*math.sin(angle), z))
                    x = np.array((2., 2., 2.))+offset
                    lam = x.sum()/10
                    N = x/(10*lam)
                    weight = zw*radius2*s*sw*angleweight
                    rows.append((N, lam, side/2, weight, offset))
    raw = {"geometry_source": {k: v for k, v in raw.items() if k not in ("seed_center_world_m", "seed_radius_m")},
           "seed_center_world_m": raw["seed_center_world_m"], "seed_radius_m": raw["seed_radius_m"],
           "barycentric": [v[0].tolist() for v in rows], "radial_scale": [v[1] for v in rows],
           "side": [v[2] for v in rows], "reference_volume_weights_m3": [v[3] for v in rows],
           "birth_volume_weights_m3": [v[3] for v in rows], "centered_birth_offsets_m": [v[4].tolist() for v in rows]}
    V = 4*math.pi*.3**3/3
    values = [[V/2, V/2, 0., 0., sign*math.pi*.3**3/4, V/10, V/10, V/10, 0., 0., 0.] for sign in (-1, 1)]
    report = {"moment_names": list(MOMENT_NAMES), "values": values,
              "own_absolute_operand_scales": (np.ones((2, len(MOMENT_NAMES)))*V).tolist(),
              "observed_embedded_refinement_errors": np.zeros((2, len(MOMENT_NAMES))).tolist(),
              "relative_tolerance": 1e-9}
    return raw, report


def test_source_chart_jacobian_ratio_and_analytic_moments_agree():
    raw, report = ball_dto()
    result = audit_archived_geometry(raw, report)
    assert result["source_node_domain_consistency_verified"]
    assert result["independent_geometric_moment_agreement_verified"]
    assert result["physical_release_admitted"] is False


def test_node_consistency_can_be_checked_without_laundering_unresolved_integration():
    raw, _ = ball_dto()
    result = audit_archived_geometry(raw)
    assert result["source_node_domain_consistency_verified"]
    assert result["independent_geometric_moment_agreement_verified"] is False
    assert result["general_contraction_accuracy_certified"] is False


@pytest.mark.parametrize("forge", ["side", "centered_point", "jacobian_ratio", "weight_moments", "radial", "barycentric"])
def test_changed_array_without_source_math_is_rejected(forge):
    raw, report = ball_dto()
    raw = deepcopy(raw)
    if forge == "side":
        raw["side"][0] *= -1
    elif forge == "centered_point":
        raw["centered_birth_offsets_m"][0][0] += .01
    elif forge == "jacobian_ratio":
        raw["birth_volume_weights_m3"][0] *= 1.01
    elif forge == "weight_moments":
        raw["birth_volume_weights_m3"] = [x*1.01 for x in raw["birth_volume_weights_m3"]]
        raw["reference_volume_weights_m3"] = [x*1.01 for x in raw["reference_volume_weights_m3"]]
    elif forge == "radial":
        raw["radial_scale"][0] = .01
    else:
        raw["barycentric"][0] = [-.1, .5, .6]
    with pytest.raises(IndependentGeometryRefusal):
        audit_archived_geometry(raw, report)


@pytest.mark.parametrize("change", [{"seed_radius_m": 0}, {"lambda_bounds": [.9, .1]},
                                    {"normal_xyz": [0., 0., 2.]}, {"triangle_m": np.zeros((3, 3)).tolist()}])
def test_invalid_original_source_is_refused(change):
    with pytest.raises(IndependentGeometryRefusal):
        read_source(source(**change))


def test_unenriched_rigid_translation_has_zero_strain_but_correct_mass_field():
    s = read_source(source())
    translation = np.tile([1., 2., 3., 0., 0., 0.], (3, 1)).ravel()
    value, strain = _probe_field(s, np.array((.3, .3, .4)), .6, 1, np.eye(3), translation, enriched=False)
    np.testing.assert_array_equal(strain, np.zeros(6))
    np.testing.assert_array_equal(value, [1., 2., 3.])


def test_independent_maxwell_cross_self_work_identity_uses_small_changes():
    s = read_source(source())
    q = np.sin(np.arange(18)+.3)*1e-6
    a = np.cos(np.arange(18)+.7)*1e-5
    z = np.sin(np.arange(18)+1.7)*2e-5
    evaluate, names = make_contraction_evaluator(elasticity_pa=np.diag([5., 6., 7., 2., 3., 4.]),
        basis_xyz=np.eye(3), strain0=np.ones((3, 6))*.2,
        strain1=np.arange(18).reshape(3, 6)*1e-3, density_kg_m3=2,
        parent_probe_m=q, enrichment_probe_m=a, before_enrichment_memory_m=z, maxwell_ratio=.7)
    result = dict(zip(names, evaluate(s, np.array((.31, .34, .35)), .6, 1)))
    assert names == CONTRACTION_NAMES
    for part in ("cross", "self"):
        residual = result["maxwell_work_"+part]-result["maxwell_dissipation_"+part]-result["maxwell_energy_change_"+part]
        own = sum(abs(result["maxwell_"+term+"_"+part]) for term in ("work", "dissipation", "energy_change"))
        assert abs(residual) <= 2e-13*own
    assert result["M_self_half"] >= 0
    assert result["K_self_half"] >= 0
    assert result["maxwell_dissipation_self"] >= 0


def test_weak_cross_does_not_borrow_large_parent_energy():
    s = read_source(source())
    # A full retained stress with tiny enrichment probe still yields a small
    # contracted result; this oracle never evaluates/subtracts parent energies.
    evaluate, names = make_contraction_evaluator(elasticity_pa=np.eye(6)*1e20,
        basis_xyz=np.eye(3), strain0=np.ones((3, 6))*1e4, strain1=np.zeros((3, 6)),
        density_kg_m3=2., parent_probe_m=np.zeros(18), enrichment_probe_m=np.ones(18)*1e-40,
        maxwell_ratio=0.)
    result = dict(zip(names, evaluate(s, np.array((.31, .34, .35)), .6, 1)))
    assert result["K_self_half"] < 1e-50
    assert abs(result["prestress_work"]) < 1e-10
    assert result["maxwell_dissipation_self"] == 0
