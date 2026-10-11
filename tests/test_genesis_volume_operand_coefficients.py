"""Exact source coefficient proofs, cancellation, and mathematical negatives."""
from copy import deepcopy
from fractions import Fraction as F

import numpy as np
import pytest

from analysis.genesis_volume_operand_validation.independent_coefficients import (
    CoefficientProofRefusal, FIELDS, audit_coefficients, decode_native_array,
    digest, encode_rational_array, exact_source_coefficients, fraction,
    source_operand_families, verify_coefficient_proof,
)


def source():
    return dict(section=dict(triangle_m=[[0., 0., 2.], [1., 0., 2.], [0., 1., 2.]],
        displacement_A_m=np.zeros((3, 3)).tolist(), slope_P_m=np.zeros((3, 3)).tolist(),
        lambda_bounds=[.75, 1.]), basis_xyz=np.eye(3).tolist(),
        seed_center_world_m=[7/32, 7/16, 7/4], seed_radius_m=1.,
        memory0=np.arange(18.).reshape(3, 6).tolist(),
        memory1=(np.arange(18.).reshape(3, 6)*-.125).tolist())


N = (F(1, 4), F(1, 4), F(1, 2))
LM, LH = F(7, 8), F(1, 64)


def witness(raw=None):
    raw = source() if raw is None else raw
    coefficients = exact_source_coefficients(raw, N, LM, LH)
    records = {"interval.lambda_mid": encode_rational_array(LM),
               "interval.lambda_halfwidth": encode_rational_array(LH),
               "interval.barycentric": encode_rational_array(N)}
    for key, values in coefficients.items():
        native = np.asarray([F(*float(x).as_integer_ratio()) for x in values.flat],
                            dtype=object).reshape(values.shape)
        residual = abs(native-values)
        records["coefficient."+key] = encode_rational_array(native)
        records["coefficient_bound."+key] = encode_rational_array(residual)
    return dict(native_exact_binary_ratio=records)


def reseal(proof):
    proof.pop("coefficient_proof_sha256", None)
    proof["coefficient_proof_sha256"] = digest(proof)
    return proof


def test_bubble_matches_hand_derived_even_polynomial():
    fields = exact_source_coefficients(source(), N, LM, LH)
    angular, alpha, beta = F(27, 32), F(1, 64), F(69, 65536)
    expected = np.array([angular, 0, -angular*(alpha+2*beta), 0,
                         angular*(beta*beta+2*alpha*beta), 0,
                         -angular*alpha*beta*beta], dtype=object)
    assert np.array_equal(fields["w"], expected)
    assert fields["T"][0, 0, 3] == F(-1, 32)
    assert fields["T"][1, 0, 3] == F(1, 256)
    assert fields["lam"].tolist() == [LM, LH]


def test_source_operand_majorants_survive_world_coordinate_cancellation():
    fields, operands = source_operand_families(source(), N, LM, LH)
    assert set(fields) == set(operands) == set(FIELDS)
    for key in FIELDS:
        assert fields[key].shape == operands[key].shape
        assert (operands[key] >= abs(fields[key])).all()
    assert operands["w"][0] > abs(fields["w"][0])
    assert operands["L"][0].max() > abs(fields["L"][0]).max()


def test_exact_proof_and_equality_verifier():
    raw, saved = source(), witness()
    proof = audit_coefficients(raw, N, saved)
    assert proof["coefficient_count"] == 1662
    assert proof["saved_coefficient_budget_failure_count"] == 0
    assert proof["exact_canonical_coefficient_residual_certified"] is True
    assert proof["advertised_forward_construction_budget_authenticated"] is False
    assert proof["geometry_endpoint_accuracy_certified"] is False
    assert verify_coefficient_proof(raw, N, saved, proof)


@pytest.mark.parametrize("field", FIELDS)
def test_consistently_resealed_residual_inflation_is_rejected(field):
    raw, saved = source(), witness()
    proof = audit_coefficients(raw, N, saved)
    records = proof["families"][field]["exact_residual"]["numerator_denominator"]
    n, d = map(int, records[0])
    records[0] = [str(n+d), str(d)]
    reseal(proof)
    with pytest.raises(CoefficientProofRefusal, match="Canonical"):
        verify_coefficient_proof(raw, N, saved, proof)


def test_altered_source_is_rejected_even_after_proof_reseal():
    raw, saved = source(), witness()
    proof = audit_coefficients(raw, N, saved)
    raw["memory0"][0][0] += 1
    with pytest.raises(CoefficientProofRefusal, match="Canonical"):
        verify_coefficient_proof(raw, N, saved, reseal(proof))


def test_missing_or_negative_budget_is_refused():
    raw, saved = source(), witness()
    saved["native_exact_binary_ratio"].pop("coefficient_bound.w")
    with pytest.raises(CoefficientProofRefusal, match="Missing"):
        audit_coefficients(raw, N, saved)
    saved = witness()
    saved["native_exact_binary_ratio"]["coefficient_bound.w"]["numerator_denominator"][0] = ["-1", "1"]
    with pytest.raises(CoefficientProofRefusal, match="Negative"):
        audit_coefficients(raw, N, saved)


def test_missing_budget_coverage_is_an_exact_numeric_refusal():
    raw, saved = source(), witness()
    saved["native_exact_binary_ratio"]["coefficient.w"]["numerator_denominator"][0] = ["100", "1"]
    proof = audit_coefficients(raw, N, saved)
    assert proof["saved_coefficient_budget_status"] == "OPEN"
    assert proof["saved_coefficient_budget_failure_count"] == 1
    assert proof["saved_coefficient_budget_failures"] == [{"family": "w", "index": [0]}]
    assert proof["exact_canonical_coefficient_residual_certified"] is True
    assert proof["source_coefficient_accuracy_against_saved_bounds_certified"] is False


def test_saved_budget_inflation_does_not_claim_forward_operand_authentication():
    raw, saved = source(), witness()
    before = audit_coefficients(raw, N, saved)
    saved["native_exact_binary_ratio"]["coefficient_bound.w"]["numerator_denominator"][0] = ["1000000", "1"]
    after = audit_coefficients(raw, N, saved)
    assert before["families"]["w"]["exact_residual"] == after["families"]["w"]["exact_residual"]
    assert after["saved_coefficient_budget_status"] == "PASS"
    assert after["advertised_forward_construction_budget_authenticated"] is False
    assert before["native_coefficient_witness_sha256"] != after["native_coefficient_witness_sha256"]


def test_other_native_N_is_refused_without_normalization():
    with pytest.raises(CoefficientProofRefusal, match="source N"):
        audit_coefficients(source(), (F(1, 3),)*3, witness())


@pytest.mark.parametrize("value", [True, np.bool_(False), float("inf"), float("nan"), "1/0", object()])
def test_nonfinite_and_boolean_rational_operands_are_refused(value):
    with pytest.raises(CoefficientProofRefusal):
        fraction(value)


def test_native_shape_and_denominator_negatives():
    saved = witness()
    saved["native_exact_binary_ratio"]["coefficient.w"]["shape"] = [2]
    with pytest.raises(CoefficientProofRefusal, match="count"):
        decode_native_array(saved, "coefficient.w")
    saved = witness()
    saved["native_exact_binary_ratio"]["coefficient.w"]["numerator_denominator"][0][1] = "0"
    with pytest.raises(CoefficientProofRefusal, match="nonpositive"):
        decode_native_array(saved, "coefficient.w")


def test_native_coefficient_binding_changes_even_when_residual_is_equal():
    raw, saved = source(), witness()
    proof = audit_coefficients(raw, N, saved)
    changed = deepcopy(saved)
    record = changed["native_exact_binary_ratio"]["coefficient.T"]["numerator_denominator"]
    record[0] = ["0", "1"]
    assert proof["native_coefficient_witness_sha256"] != audit_coefficients(raw, N, changed)["native_coefficient_witness_sha256"]
