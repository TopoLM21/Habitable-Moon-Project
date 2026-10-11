"""Independent regular-field and split quadrature checks for exact lambda."""
from dataclasses import replace
from types import MappingProxyType
from fractions import Fraction

import numpy as np
import pytest

from tectonics.genesis_volume_geometry import ConicalCohortGeometry
from tectonics.genesis_volume_material_sections import prepare_cohort_material_section
from tectonics.genesis_volume_operator import ANGULAR_BARYCENTRIC, isotropic_elasticity, conical_strain_operators
from tectonics.genesis_volume_embedded_partition import _cohort_enrichment_fields
from tectonics.genesis_volume_exact_lambda import poly_evaluate, ExactLambdaRefusal
from tectonics.genesis_volume_clipped_bulk import _parent_blocks
from tectonics.genesis_volume_exact_lambda_bulk import (
    ExactLambdaBulkRefusal, prepare_source, integrate_line, parent_probe_work,
    prepare_line_evaluator, validate_line, validate_source, line_to_dict, source_to_dict,
)

LD = np.longdouble
X = np.array(((0., -5., 10.), (5., -5., 10.), (0., 5., 10.)))


def _source(*, normal=(0., 0., 1.), offset=9.03, displacement=None,
            lower=.8, upper=.95, center=(1., 0., 9.), radius=.25, Q=None):
    coefficient = .5
    volume = np.linalg.det(X)*(upper**3-lower**3)/6
    cohort = ConicalCohortGeometry(17, 0, 2, (1-upper**3)/coefficient,
        (1-lower**3)/coefficient, 2*volume, volume, tuple(map(tuple, X)),
        upper, lower, (), "unchanged-original-owner", "unchanged-original-material")
    section = prepare_cohort_material_section(cohort,
        np.zeros((3, 6)) if displacement is None else displacement,
        normal, offset, conical_mass_coefficient=coefficient)
    e0 = (np.arange(18).reshape(3, 6)-8)*1e-4
    e1 = (np.arange(18).reshape(3, 6)[::-1]-7)*2e-4
    return prepare_source(dict(section=section, elasticity_pa=isotropic_elasticity(3e8, .27),
        basis_xyz=np.eye(3) if Q is None else Q, memory0=e0, memory1=e1,
        accepted_interval_mean_e0=e0*.7, accepted_interval_mean_e1=e1*.4,
        seed_center_world_m=center, seed_radius_m=radius, source_identity="original-source"))


def _N():
    N = np.array((.31, .20, .49), LD)
    return N/N.sum()


def _probe(source):
    j = np.arange(1, 19)
    return dict(name="saved_full_mean", parent_increment_m=1e-4*np.sin(.37*j),
        enrichment_increment_m=2e-4*np.cos(.23*j), mean_z_m=1e-5*np.sin(.19*j),
        mean_e0=source.accepted_interval_mean_e0,
        mean_e1=source.accepted_interval_mean_e1)


@pytest.fixture(scope="module")
def source():
    return _source()


@pytest.fixture(scope="module")
def line(source):
    W = np.column_stack((np.sin(.13*np.arange(18)), np.cos(.27*np.arange(18))))
    return integrate_line(source, _N(), added_basis=W, probes=(_probe(source),))


def _points(source, line, order=22, mean_e0=None, mean_e1=None):
    """Old regular field evaluated at split Gauss nodes; no coefficient Gram."""
    interval = line.interval
    cuts = [-LD(1), LD(1)]
    if interval.constant_side is None:
        cuts.insert(1, interval.gamma_root_xi)
    nodes, gaussweights = np.polynomial.legendre.leggauss(order)
    out = []
    angular = interval.barycentric@np.asarray(np.linalg.inv(ANGULAR_BARYCENTRIC), LD)
    e0 = source.memory0 if mean_e0 is None else mean_e0
    e1 = source.memory1 if mean_e1 is None else mean_e1
    for left, right in zip(cuts, cuts[1:]):
        xi = (left+right)/2+(right-left)/2*np.asarray(nodes, LD)
        weights = interval.lambda_halfwidth*(right-left)/2*np.asarray(gaussweights, LD)
        for x, weight in zip(xi, weights):
            lam = interval.lambda_mid+interval.lambda_halfwidth*x
            offset = interval.birth_relative_coefficients[0]+x*interval.birth_relative_coefficients[1]
            fields = _cohort_enrichment_fields(source.section,
                np.asarray(source.geometry.seed_center_world_m, LD), source.geometry.seed_radius_m,
                source.basis_xyz, interval.barycentric, lam, birth_relative_m=offset)
            elastic = angular@np.asarray(e0, LD)+(angular@np.asarray(e1, LD))/lam
            out.append((x, lam, weight*interval.reference_determinant_m3*lam**2, fields, elastic))
    return out


def test_degree_and_complete_original_coordinate_pattern(line):
    degrees = dict(lam=1, w=6, L=6, Bhat=1, Ghat_without_H=7,
                   T=1, E_without_H=7, memoryhat=1)
    assert {name: len(v)-1 for name, v in line.coefficients.items()} == degrees
    assert line.blocks["mass_cross"].shape == (18, 18)
    for i in range(18):
        for j in range(18):
            if i % 3 != j % 3:
                assert line.blocks["mass_cross"][i, j] == 0
                assert line.roundoff_component_bounds["mass_cross"][i, j] == 0
            else:
                assert line.blocks["mass_cross"][i, j] != 0
    assert sum(line.blocks["mass_cross"].ravel() == 0) == 216
    for rows in (np.arange(18).reshape(3, 6)[:, :3].ravel(), np.arange(18).reshape(3, 6)[:, 3:].ravel()):
        for cols in (np.arange(18).reshape(3, 6)[:, :3].ravel(), np.arange(18).reshape(3, 6)[:, 3:].ravel()):
            assert np.any(line.blocks["mass_cross"][np.ix_(rows, cols)] != 0)


def test_coefficients_match_existing_regular_fields_and_full_memory(source, line):
    for _, lam, _, fields, elastic in _points(source, line, order=4):
        xi = (lam-line.interval.lambda_mid)/line.interval.lambda_halfwidth
        c = {k: poly_evaluate(v, xi) for k, v in line.coefficients.items()}
        np.testing.assert_allclose(c["w"], fields["bubble"], rtol=3e-14, atol=0)
        np.testing.assert_allclose(c["L"], lam*fields["gradient_reference_m_inverse"], rtol=3e-13, atol=2e-14)
        np.testing.assert_allclose(c["T"], fields["parent_trace"], rtol=3e-15, atol=0)
        phi = line.interval.literal_phi_coefficients[0]+xi*line.interval.literal_phi_coefficients[1]
        side = LD(.5) if phi > 0 else LD(-.5)
        np.testing.assert_allclose(c["E_without_H"]*side, fields["enrichment_trace"], rtol=3e-14, atol=0)
        np.testing.assert_allclose(c["Ghat_without_H"]*side/lam, fields["enrichment_strain_operator"], rtol=4e-13, atol=2e-13)
        np.testing.assert_allclose(c["memoryhat"]/lam, elastic, rtol=3e-15, atol=0)
        assert np.any(elastic[:3] != 0) and np.any(elastic[3:] != 0)


def test_full_integrals_against_independent_split_point_quadrature(source, line):
    C, density = np.asarray(source.elasticity_pa, LD), LD(source.section.mass_kg)/LD(source.section.volume_m3)
    values = {k: np.zeros_like(v) for k, v in line.blocks.items()}
    own = {k: np.zeros_like(v) for k, v in line.blocks.items()}
    for _, _, weight, f, elastic in _points(source, line):
        B, G, T, E = (f[k] for k in ("parent_strain_operator", "enrichment_strain_operator", "parent_trace", "enrichment_trace"))
        for name, L, R, middle, factor in (("stiffness_cross", B, G, C, 1),
                ("stiffness_enriched", G, G, C, 1), ("mass_cross", T, E, np.eye(3), density),
                ("mass_enriched", E, E, np.eye(3), density)):
            values[name] += weight*factor*(L.T@middle@R)
            own[name] += abs(weight*factor)*(np.abs(L).T@np.abs(middle)@np.abs(R))
        values["enrichment_force_n"] += weight*(G.T@C@elastic)
        own["enrichment_force_n"] += abs(weight)*(np.abs(G).T@np.abs(C)@np.abs(elastic))
    # Independent Gauss nodes/weights and the old point fields retain their
    # own float64 construction scope, distinct from the exact primitive.
    for name in values:
        bounds = line.roundoff_component_bounds[name]+LD(128*np.finfo(float).eps)*own[name]
        assert np.all(np.abs(values[name]-line.blocks[name]) <= bounds), name


def test_literal_float64_vertex_parent_and_analytic_parent_are_unchanged(source):
    B0, B1 = conical_strain_operators(source.section.triangle_m, source.basis_xyz, np.eye(3))
    np.testing.assert_array_equal(source.literal_vertex_B0, B0)
    np.testing.assert_array_equal(source.literal_vertex_B1, B1)
    parent = _parent_blocks(source.section, source.elasticity_pa, source.basis_xyz, source.memory0, source.memory1)
    for got, expected in zip((source.parent_stiffness, source.parent_mass, source.parent_force_n), parent):
        np.testing.assert_array_equal(got, expected)
    probe = _probe(source)
    _, _, force = _parent_blocks(source.section, source.elasticity_pa, source.basis_xyz,
                                probe["mean_e0"], probe["mean_e1"])
    work = parent_probe_work(source, probe)
    assert work["value"] == np.asarray(probe["parent_increment_m"], LD)@np.asarray(force, LD)
    assert "whole_original_cohort" in work["scope"]
    assert not work["available_for_creation_funding"]


def test_projected_and_declared_work_adjoint_is_before_gram(source, line):
    probe = _probe(source)
    q, a, z = [np.asarray(probe[k], LD) for k in ("parent_increment_m", "enrichment_increment_m", "mean_z_m")]
    contractions = line.probe_contractions[probe["name"]]
    values, budget = contractions["values"], contractions["roundoff_component_bounds"]
    assert "parent_work_j" not in values
    for name, value in (("stiffness_cross_bilinear_j", q@line.blocks["stiffness_cross"]@a),
                        ("stiffness_self_bilinear_j", a@line.blocks["stiffness_enriched"]@a),
                        ("mass_cross_bilinear_kg_m2", q@line.blocks["mass_cross"]@a),
                        ("mass_self_bilinear_kg_m2", a@line.blocks["mass_enriched"]@a),
                        ("cross_parent_work_j", q@line.blocks["stiffness_cross"]@z),
                        ("self_work_j", a@line.blocks["stiffness_enriched"]@z)):
        assert abs(values[name]-value) <= budget[name]+LD(5e-17)*abs(value)
    assert abs(q@values["added_parent_force_n"]-values["cross_parent_work_j"]) <= budget["cross_parent_work_j"]+np.abs(q)@budget["added_parent_force_n"]
    assert abs(a@values["enrichment_force_n"]-values["cross_added_work_j"]-values["self_work_j"]) <= budget["cross_added_work_j"]+budget["self_work_j"]+np.abs(a)@budget["enrichment_force_n"]
    # Weak basis cancellation has its own projection budget; this checks the
    # point/coefficient projection path without replacing it by rounded K.
    W = np.column_stack((np.sin(.13*np.arange(18)), np.cos(.27*np.arange(18))))
    expected = np.asarray(W, LD).T@line.blocks["mass_enriched"]@np.asarray(W, LD)
    assert np.all(np.abs(expected-line.projected_blocks["mass_enriched"]) <=
        line.projected_roundoff_component_bounds["mass_enriched"]+LD(5e-17)*np.abs(expected))


def test_mean_memory_changes_all_direct_terms_and_keeps_sign(source):
    probe = _probe(source)
    positive = integrate_line(source, _N(), probes=(probe,)).probe_contractions[probe["name"]]
    negative_probe = {**probe, "mean_e0": -probe["mean_e0"], "mean_e1": -probe["mean_e1"], "mean_z_m": -probe["mean_z_m"]}
    negative = integrate_line(source, _N(), probes=(negative_probe,)).probe_contractions[probe["name"]]
    for key in ("cross_parent_work_j", "cross_added_work_j", "self_work_j", "added_parent_force_n", "enrichment_force_n"):
        np.testing.assert_array_equal(negative["values"][key], -positive["values"][key])
        np.testing.assert_array_equal(negative["roundoff_component_bounds"][key], positive["roundoff_component_bounds"][key])
    default = {k: v for k, v in probe.items() if k not in ("mean_e0", "mean_e1")}
    without_mean = integrate_line(source, _N(), probes=(default,)).probe_contractions[probe["name"]]
    assert positive["values"]["cross_added_work_j"] != without_mean["values"]["cross_added_work_j"]


def test_exact_zero_probe_and_structural_zero_have_no_positive_budget_floor(source):
    probe = dict(name="zero", parent_increment_m=np.zeros(18), enrichment_increment_m=np.zeros(18),
                 mean_e0=np.zeros((3, 6)), mean_e1=np.zeros((3, 6)), mean_z_m=np.zeros(18))
    line = integrate_line(source, _N(), probes=(probe,))
    for part in line.probe_contractions["zero"].values():
        for value in part.values():
            assert not np.any(value)
    assert np.any(line.blocks["stiffness_enriched"])  # Large AA does not fund a zero/weak probe.


def test_literal_gamma_reverse_changes_cross_only(source, line):
    negative = _source(normal=(0., 0., -1.), offset=-9.03)
    reversed_line = integrate_line(negative, _N())
    for name in ("stiffness_cross", "mass_cross", "enrichment_force_n"):
        np.testing.assert_array_equal(reversed_line.blocks[name], -line.blocks[name])
    for name in ("stiffness_enriched", "mass_enriched"):
        np.testing.assert_array_equal(reversed_line.blocks[name], line.blocks[name])


def test_prepared_native_and_sealed_routes_are_identical_and_inputs_frozen(source):
    probe = _probe(source)
    W = np.column_stack((np.sin(.13*np.arange(18)), np.cos(.27*np.arange(18))))
    evaluator = prepare_line_evaluator(source, added_basis=W, probes=(probe,))
    W[0, 0] = 1000
    probe["mean_z_m"][0] = 2000
    for N in (_N(), np.array((.2, .3, .5), LD)):
        native, sealed = evaluator(N), evaluator.witness(N)
        np.testing.assert_array_equal(native.values, sealed.values)
        np.testing.assert_array_equal(native.budgets, sealed.budgets)
        assert native.names == sealed.names == evaluator.names
        assert native.interval is sealed.interval or native.interval.lambda_bounds == sealed.interval.lambda_bounds


def test_empty_line_schema_and_coplanar_refusal(source):
    evaluator = prepare_line_evaluator(source, probes=(_probe(source),))
    empty = evaluator((1., 0., 0.))
    assert empty.interval is None
    assert not np.any(empty.values) and not np.any(empty.budgets)
    assert len(empty.values) == len(evaluator.names)
    coplanar = _source(normal=(0., 1., 0.), offset=0.)
    with pytest.raises(ExactLambdaRefusal, match="whole line"):
        integrate_line(coplanar, (.3, .2, .5))


def test_live_bindings_immutable_and_subfloat64_tamper_refused(source, line):
    validate_source(source); validate_line(line)
    with pytest.raises(ValueError):
        source.literal_vertex_B0.setflags(write=True)
    with pytest.raises(ValueError):
        line.values.setflags(write=True)
    forged = line.values.copy()
    i = int(np.argmax(np.abs(forged)))
    forged[i] = np.nextafter(forged[i], LD(np.inf))
    assert float(forged[i]) == float(line.values[i])
    with pytest.raises(ExactLambdaBulkRefusal, match="native"):
        validate_line(replace(line, values=forged))
    coeffs = dict(line.coefficients)
    coeffs["w"] = coeffs["w"].copy()
    coeffs["w"][0] = np.nextafter(coeffs["w"][0], LD(np.inf))
    with pytest.raises(ExactLambdaBulkRefusal, match="native"):
        line_to_dict(replace(line, coefficients=MappingProxyType(coeffs)))
    interval = replace(line.interval, gamma_left_width_xi=np.nextafter(line.interval.gamma_left_width_xi, LD(np.inf)))
    with pytest.raises(ExactLambdaBulkRefusal, match="native"):
        validate_line(replace(line, interval=interval))


def test_public_archive_has_distinct_seals_and_no_physical_credit(source, line):
    body, sourcebody = line_to_dict(line), source_to_dict(source)
    assert body["archive_sha256"] != body["live_precision_binding_sha256"]
    assert sourcebody["archive_sha256"] != sourcebody["live_precision_binding_sha256"]
    for key in ("physical_contraction_certified", "available_for_creation_funding",
                "physical_creation_admitted", "physical_release_admitted",
                "geometry_operand_accuracy_certified", "angular_accuracy_certified"):
        assert body[key] is False
    assert body["new_material_count"] == body["accepted_time_advance_myr"] == body["funding_debit_j"] == 0
    assert "coefficients" not in line_to_dict(line, include_coefficients=False)
    native = body["native_exact_binary_ratio"]["coefficient.w"]
    expected = [Fraction(*LD(v).as_integer_ratio()) for v in line.coefficients["w"]]
    assert [Fraction(int(n), int(d)) for n, d in native["numerator_denominator"]] == expected
    assert native["shape"] == [7]


def test_full_birth_displacement_and_rotated_source_frame_are_retained():
    displacement = np.array(((.003, -.007, .004, .011, -.014, .017),
                             (-.003, .002, -.006, -.013, .009, .003),
                             (.002, .004, -.001, .002, -.003, .008)))
    theta = .37
    Q = np.array(((np.cos(theta), -np.sin(theta), 0.),
                  (np.sin(theta), np.cos(theta), 0.), (0., 0., 1.)))
    source = _source(displacement=displacement, Q=Q)
    line = integrate_line(source, _N())
    for x, lam, _, fields, elastic in _points(source, line, order=3):
        c = {k: poly_evaluate(v, x) for k, v in line.coefficients.items()}
        side = .5*np.sign(line.interval.literal_phi_coefficients[0]+x*line.interval.literal_phi_coefficients[1])
        np.testing.assert_allclose(c["Ghat_without_H"]*side/lam,
            fields["enrichment_strain_operator"], rtol=4e-13, atol=2e-13)
        np.testing.assert_allclose(c["L"]/lam, fields["gradient_reference_m_inverse"], rtol=3e-13, atol=2e-14)
        np.testing.assert_allclose(c["memoryhat"]/lam, elastic, rtol=3e-15, atol=0)
    np.testing.assert_array_equal(source.geometry.birth_C_m, np.asarray(X+displacement[:, 3:], LD))
    np.testing.assert_array_equal(source.geometry.birth_D_m, np.asarray(displacement[:, :3]-displacement[:, 3:], LD))


def test_thin_cohort_centered_polynomial_keeps_weak_original_P_components():
    lower, upper = 1-2e-10, 1-2e-11
    center = (1., 0., 10*(lower+upper)/2)
    source = _source(lower=lower, upper=upper, center=center,
                     offset=10*((lower+upper)/2+.17*(upper-lower)))
    line = integrate_line(source, _N())
    assert len(line.coefficients["w"]) == 7
    assert np.all(np.isfinite(line.values))
    angular = 27*np.prod(_N())
    # At xi=0 evaluate radial factors from independently represented cohort
    # distances instead of translating a polynomial from large lambda powers.
    interval = line.interval
    tlow = interval.relative_lambda_mid_to_original_lower/LD(upper-lower)
    thigh = interval.relative_lambda_mid_to_original_upper/LD(upper-lower)
    r = interval.birth_relative_coefficients[0]
    expected = angular*4*tlow*thigh*(1-(r@r)/LD(.25)**2)**2
    assert abs(line.coefficients["w"][0]-expected) <= line.coefficient_roundoff_bounds["w"][0]
    aa = abs(line.blocks["mass_cross"][0, 0])
    pp = abs(line.blocks["mass_cross"][3, 3])
    assert 0 < pp < aa*LD(1e-18)
    assert 0 < line.roundoff_component_bounds["mass_cross"][3, 3] < aa*LD(1e-20)
    assert line.roundoff_component_bounds["mass_cross"][0, 1] == 0


def test_invalid_declared_source_and_probe_fail_locally(source):
    with pytest.raises(ExactLambdaBulkRefusal, match="real"):
        integrate_line(source, _N(), added_basis=np.ones((17, 2)))
    p = _probe(source)
    with pytest.raises(ExactLambdaBulkRefusal, match="uniquely"):
        integrate_line(source, _N(), probes=(p, p))
    bad = dict(name="bad", parent_increment_m=np.zeros(17), enrichment_increment_m=np.zeros(18))
    with pytest.raises(ExactLambdaBulkRefusal, match="Parent"):
        integrate_line(source, _N(), probes=(bad,))
