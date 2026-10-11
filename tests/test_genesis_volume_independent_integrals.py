"""Source-root endpoint integration and independent enclosure regressions."""
from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from analysis.genesis_volume_integral_validation import independent_integrals as proof
from analysis.genesis_volume_operand_validation import independent_contractions as algebra
from analysis.genesis_volume_operand_validation.independent_coefficients import exact_source_coefficients


def _packet(witness, label, value):
    value = algebra._array(value)
    witness.setdefault("native_exact_binary_ratio", {})[label] = dict(
        shape=list(value.shape), numerator_denominator=[algebra._ratio(q) for q in value.ravel()])


def _fixture(*, clipped=False, projected=False):
    N = (Fraction(1, 4), Fraction(1, 4), Fraction(1, 2))
    raw = {"section": {"triangle_m": [[0., 0., 2.], [1., 0., 2.], [0., 1., 2.]],
        "displacement_A_m": np.zeros((3, 3)).tolist(), "slope_P_m": np.zeros((3, 3)).tolist(),
        "lambda_bounds": [.75, 1.25], "normal_xyz": [0., 0., 1.], "offset_m": 2.,
        "mass_kg": 3., "volume_m3": 2.}, "basis_xyz": np.eye(3).tolist(),
        "seed_center_world_m": [.25, .5, 2.], "seed_radius_m": .125 if clipped else 10.,
        "memory0": (np.arange(18.).reshape(3, 6)*.125).tolist(),
        "memory1": (np.arange(18.).reshape(3, 6)*-.03125).tolist(),
        "elasticity_pa": np.eye(6).tolist()}
    mid, half = Fraction(1), Fraction(1, 16) if clipped else Fraction(1, 4)
    witness = {"interval": {"gamma_root_xi": 0., "constant_side": None}}
    for key, value in dict(lambda_mid=mid, lambda_halfwidth=half,
                            reference_determinant_m3=2, gamma_endpoint_distance_xi=1).items():
        _packet(witness, "interval."+key, value)
    _packet(witness, "interval.barycentric", N)
    _packet(witness, "interval.side", [0])
    _packet(witness, "interval.literal_phi_coefficients", [0, 2*half])
    coefficients = exact_source_coefficients(raw, N, mid, half)
    for key, value in coefficients.items():
        native = np.asarray([float(q) for q in value.ravel()]).reshape(value.shape)
        _packet(witness, "coefficient."+key, native)
    options = {}
    if projected:
        W = np.zeros((18, 15)); W[:15] = np.eye(15)
        q, a, z = np.arange(18.)*.03125, np.arange(18.)*-.0625, np.arange(18.)*.125
        probes = [dict(name="accepted_mean", q=q.tolist(), a=a.tolist(), mean_z_m=z.tolist(),
                       mean_e0=raw["memory0"], mean_e1=raw["memory1"])]
        options.update(added_basis=W, parent_basis=None, probes=probes)
    source = algebra.exact_line_contractions(raw, N, witness, coefficients=coefficients,
        gamma_mode="literal_source", weight_mode="literal_source", **options)
    witness["component_names"] = list(source)
    _packet(witness, "values", list(map(float, source.values())))
    _packet(witness, "budgets", np.ones(len(source)))
    return raw, N, witness, source, options


@pytest.mark.parametrize("anchor,endpoint", [(-1, (Fraction(-9, 8), Fraction(-7, 8))),
                                             (1, (Fraction(7, 8), Fraction(9, 8))),
                                             (0, (Fraction(-1, 4), Fraction(1, 4)))])
def test_translated_antiderivative_encloses_hand_polynomial(anchor, endpoint):
    p = (Fraction(2), Fraction(-3), Fraction(5))
    bounds = proof.translated_antiderivative_interval(p, anchor, endpoint)
    def primitive(x):
        return 2*x-Fraction(3, 2)*x*x+Fraction(5, 3)*x**3
    for numerator in range(9):
        x = endpoint[0]+(endpoint[1]-endpoint[0])*Fraction(numerator, 8)
        assert bounds[0] <= primitive(x)-primitive(Fraction(anchor)) <= bounds[1]


@pytest.mark.parametrize("side", [0, 1, 2])
@pytest.mark.parametrize("anchor", [-1, 1])
def test_exact_endpoint_direction_and_gamma_bank(side, anchor):
    endpoint = Fraction(anchor)+Fraction(1, 8)
    result = proof.signed_endpoint_integral((Fraction(2),), anchor, (endpoint, endpoint), side, 0, 1)
    bank = Fraction(-1, 2) if anchor < 0 else Fraction(1, 2)
    expected = Fraction(1, 4)*(1 if side == 0 else bank if side == 1 else Fraction(1, 4))
    assert result == (expected, expected)


def test_literal_gamma_root_inside_endpoint_sliver_is_integrated():
    # Integrate from 1 to 1+epsilon with a literal bank switch halfway through.
    tiny = Fraction(1, 2**180)
    root = Fraction(1)+tiny/2
    target = (Fraction(1)+tiny,)*2
    result = proof.signed_endpoint_integral((Fraction(1), Fraction(1)), 1, target, 1, -root, 1)
    # Constant part cancels; slope leaves a positive quadratic endpoint sliver.
    assert result == (tiny*tiny/8,)*2


def test_gamma_endpoint_interval_straddles_root_without_omitting_bank():
    root = Fraction(5, 4)
    bounds = proof.signed_endpoint_integral((Fraction(1),), 1,
                    (Fraction(9, 8), Fraction(11, 8)), 1, -root, 1)
    assert bounds[0] <= -Fraction(1, 8) <= bounds[1]
    assert bounds[0] <= -Fraction(1, 16) <= bounds[1]


def test_inactive_reference_bank_remains_exact_zero_with_irrational_endpoints():
    raw, N, witness, _, _ = _fixture(clipped=True)
    raw["section"]["offset_m"] = 100.
    target = proof.true_source_integral_enclosures(raw, N, witness)
    assert target[proof.VOLUME_NAMES[1]] == (0, 0)
    assert target[proof.VOLUME_NAMES[0]][0] > 0
    tiny = Fraction(1, 2**180)
    for bank in (0, 1):
        actual = proof.bank_endpoint_integral((Fraction(1),), 1, (1+tiny, 1+tiny), bank,
                                               -(1+tiny/2), 1)
        assert actual == (tiny/2, tiny/2)


def test_identically_gamma_line_refused_and_no_positive_floor():
    with pytest.raises(proof.SourceIntegralRefusal, match="identically"):
        proof.signed_endpoint_integral((Fraction(1),), 0, (0, 1), 1, 0, 0)
    assert proof.translated_antiderivative_interval((Fraction(0),), 1, (1, 2)) == (0, 0)


def test_source_polynomials_reproduce_frozen_independent_schema_without_migration():
    raw, N, witness, exact, options = _fixture(projected=True)
    target = proof.true_source_integral_enclosures(raw, N, witness, **options)
    assert list(target) == list(proof.VOLUME_NAMES)+list(exact)
    assert len(target) == 2+1314+1005+43
    for name, value in exact.items():
        assert target[name] == (value, value)
    # Each half of the literal Gamma cut has exact reference measure.
    assert target[proof.VOLUME_NAMES[0]] == (Fraction(37, 96),)*2
    assert target[proof.VOLUME_NAMES[1]] == (Fraction(61, 96),)*2
    assert not any("parent_probe_work" in name or "funding" in name for name in target)


def test_true_source_sphere_integral_encloses_hand_irrational_volume():
    raw, N, witness, _, _ = _fixture(clipped=True)
    targets = proof.true_source_integral_enclosures(raw, N, witness)
    moments = proof.source_moment_enclosures(raw, N, witness)
    assert moments["source_lower_xi"][1] < -Fraction(9, 10)
    assert moments["source_upper_xi"][0] > Fraction(9, 10)
    assert any(a != 0 or b != 0 for a, b in moments["endpoint_corrections"][0])
    # Source λ halfwidth t satisfies 69*t² = 1/4, from |u|² = 69/16.
    from analysis.genesis_volume_operand_validation.independent_geometry_bounds import sqrt_enclosure
    lo, hi = sqrt_enclosure(Fraction(1, 276), binary_places=300)
    for name, sign in zip(proof.VOLUME_NAMES, (-1, 1)):
        # det=2 and ∫_0^t (1±s)^2 ds = t ± t² + t³/3.
        hand = 2*(lo+sign*lo*lo+lo**3/3)
        hand_hi = 2*(hi+sign*hi*hi+hi**3/3)
        assert targets[name][0] <= min(hand, hand_hi) <= max(hand, hand_hi) <= targets[name][1]


def test_canonical_source_certificate_retains_migration_and_scope():
    raw, N, witness, _, _ = _fixture(clipped=True)
    certificate = proof.build_true_source_integral_certificate(raw, N, witness)
    assert certificate["checked_component_count"] == 1316
    assert certificate["source_endpoint_displacement_integrated"] is True
    assert certificate["angular_accuracy_certified"] is False
    assert certificate["selected_line_uniform_forward_bound_certified"] is False
    assert certificate["available_for_creation_funding"] is False
    assert proof.verify_true_source_integral_certificate(raw, N, witness, certificate)["status"] == "PASS"


@pytest.mark.parametrize("field", ["true_source_integral_enclosure", "endpoint_migration_integral_enclosure"])
def test_consistently_resealed_target_inflation_refused(field):
    raw, N, witness, _, _ = _fixture()
    certificate = proof.build_true_source_integral_certificate(raw, N, witness)
    component = certificate["components"][proof.VOLUME_NAMES[0]]
    numerator, denominator = map(int, component[field]["upper"])
    component[field]["upper"] = [str(numerator+denominator), str(denominator)]
    certificate.pop("certificate_sha256")
    certificate["certificate_sha256"] = algebra._digest(certificate)
    with pytest.raises(proof.SourceIntegralRefusal, match="Canonical"):
        proof.verify_true_source_integral_certificate(raw, N, witness, certificate)


def test_changed_source_root_cannot_hide_behind_resealed_target():
    raw, N, witness, _, _ = _fixture()
    certificate = proof.build_true_source_integral_certificate(raw, N, witness)
    changed = deepcopy(raw); changed["seed_radius_m"] = .125
    with pytest.raises(proof.SourceIntegralRefusal, match="Canonical"):
        proof.verify_true_source_integral_certificate(changed, N, witness, certificate)


def test_runtime_enclosure_audit_reports_missed_own_component():
    targets = {"a": (Fraction(1, 3), Fraction(1, 2)), "zero": (Fraction(0), Fraction(0))}
    assert proof.verify_runtime_enclosures(targets, [0, 0], [1, 0], list(targets))["status"] == "PASS"
    result = proof.verify_runtime_enclosures(targets, [0, 0], [Fraction(2, 5), 0], list(targets))
    assert result["status"] == "OPEN" and result["failures"] == ["a"]
    with pytest.raises(proof.SourceIntegralRefusal, match="schema"):
        proof.verify_runtime_enclosures(targets, [0, 0], [1, 0], ["zero", "a"])


def test_other_material_N_refused_without_normalizing():
    raw, N, witness, _, _ = _fixture()
    with pytest.raises(proof.SourceIntegralRefusal, match="native material N"):
        proof.source_moment_enclosures(raw, (Fraction(1, 3),)*3, witness)
