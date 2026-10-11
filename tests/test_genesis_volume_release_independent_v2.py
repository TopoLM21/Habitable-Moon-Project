"""Frozen genuine R4m source-kernel regression, without a planet restore.

The compressed record contains only the original authenticated material chart
and literal release-trace diagnostic. It cannot initialize a planet or replace
its history. The v2 actual audit still derives these witnesses afresh from the
complete authenticated parent and previous source-memory fixture.
"""
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zlib

import numpy as np
import pytest

from analysis.genesis_volume_release_validation import independent_audit as old
from analysis.genesis_volume_release_validation import independent_audit_v2 as oracle
from analysis.genesis_volume_release_validation import independent_native_audit_v2 as native

# Authenticated extraction from genuine actual01; no synthetic material/state.
GENUINE_RECORD_SHA256 = "505f866765ec8bf3c55dc0c074227a8bafcdac663e5b436347dc536364b35410"
GENUINE_RECORD_ZLIB_BASE64 = (
    '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'
    'ZgxAVARdAKIg6AIQDcFsOsYFmDEeqInLlJg31Lvs/N2UmNUteHkcV04XCXnJrFxuYeLuZ9RgitcuJ7PMqNulOjC/xinTMTMEokHo'
    'IxANQh+BKBBWp2NGCESD0EcgGoQ+AlEgzI8ycbcfxLrWWcWi7FqdQ3jHXFelZ6DP1AzLxM7N6eGxM+9iRqRe7tNHxvE8i2zDsoEU'
    'eAeTrSZrR5lmMETF0YchKg4Fhmg41o8yjWCIiqMPQ1QcCgzRcEyPMvHUeba1jBNd95Tanea13R5d093NmbzGnPOR43BnIA5tRzSr'
    '27l7tBndxsp0yPaW1DxRyiMRJvRPMs3YF5X/Dvui8d9lXzT+1w4yjdgXlf8O+6Lx32VfNP6n55go7JDpqeey7fpt3njdAqWU3L2D'
    'TJE1xNr15QvqxGvgTa013dt4nl3IrCleq1tQx+i5myiglf5BpjkAURH0AIiKoAdANARrJ5nGAERF0AMgKoIeANEQTE6nwBZ6y4gr'
    'z0rI1lacwLXjmObeWSZE4q1eTwlLas9ZRFYbXwu0h/MZCOEarXELaqaRjPQWK+yzAypz/kUF0ONfVAA9/kUDsHJGZcy/qAB6/IsK'
    'oMe/aACmB+AYY1km84uBC6yJwyy/3D3EZkO78z5OC7E1aioBbxCKJd8VmsRNPfhCXgoN+quwOmrtn4Cbci8q+z3uRWW/x70o7K8d'
    'gfvJvajs97gXlf0e96KwP3FPiJ0ZXHOj2NK/8PZqLoKsJuTNtCJw53WWyOtWnFtwrBhoYil33RNNL+K1kpbuKbFUGHivXe80519U'
    'AF3+RQXQ418UACvOacy/qAC6/IsKoMe/KACmR5xYbsByE+o4n/6MhgrrL3Fv4b1glpUvbOExGx+WcXRrmsGUvxtHW4dwL7voViJp'
    'cB0dj/i4/imnOQRRMXQhiIqhC0E0DGsHncYQRMXQhSAqhi4E0TDMMvIcEPogVveuxGVGHAyXzT2M3v2zTim6iDaqicuUnmaS7rGT'
    'kSegp27UlYQclNziFnNUEvIFClFh9FGIBqOPQjQYq/n4BIWoMPooRIPRRyEajHEBY163FCFhzswKGMcnTjfBbJoQcg73ShVBQG0s'
    'xoWa1ughxj6WCMdt050KyJHFl2+VvBb00bX74FiCp1fBeA2FaDAUFKLC6KMQBcayhPEchWgwFBSiwuijEAXGrOwaq7lxXwUc6bSA'
    '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'
)


@pytest.fixture
def genuine():
    body = zlib.decompress(base64.b64decode(GENUINE_RECORD_ZLIB_BASE64))
    assert hashlib.sha256(body).hexdigest() == GENUINE_RECORD_SHA256
    result = json.loads(body)
    assert result["parent_raw_sha256"] == oracle.PARENT_RAW_SHA256
    assert result["previous_memory_fixture_raw_sha256"] == oracle.PREVIOUS_MEMORY_FIXTURE_RAW_SHA256
    return result


def reseal(raw):
    raw = {key: value for key, value in raw.items() if key != "release_modes_sha256"}
    return {**raw, "release_modes_sha256": oracle.digest(raw)}


def audit_genuine(genuine, raw=None):
    original = genuine["release_modes"]
    return oracle.audit_jump_modes(original if raw is None else raw,
        np.asarray(original["trace_xyz"]), np.asarray(original["reference_area_weights_m2"]),
        source_null_modes=oracle.source_geometric_null_modes(genuine["chart"]))


def function_bodies(module):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(Path(module.__file__).read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def test_regression_is_genuine_r4m_geometry_without_planet_initialization(genuine):
    assert genuine["scope"] == "immutable_genuine_source_geometry_and_literal_trace_numerical_regression_only_no_planet_state"
    assert genuine["seed_definition"]["original_cohort_id"] == 1
    assert 4. < genuine["seed_definition"]["seed_radius_m"] < 4.1
    assert set(genuine["chart"]) == {"X", "A", "P", "n", "d"}
    assert "runtime_state" not in genuine and "state" not in genuine
    assert np.asarray(genuine["release_modes"]["trace_xyz"]).shape == (64, 3, 18)


def test_source_phi_is_original_parent_plane_residual(genuine):
    chart = genuine["chart"]
    X, A, P = (np.asarray(chart[key]) for key in ("X", "A", "P"))
    n = np.asarray(chart["n"])
    a, p = (X+A)@n-chart["d"], (X+P)@n
    phi = oracle.source_geometric_null_modes(chart)
    expected = np.zeros((18, 3))
    for axis in range(3):
        expected[axis::6, axis], expected[axis+3::6, axis] = a, p
    expected /= np.linalg.norm(expected, axis=0)
    np.testing.assert_array_equal(phi, expected)
    np.testing.assert_array_equal(phi, genuine["release_modes"]["analytic_null_modes"])


def test_independent_full_rank_plus_source_geometry_admits_complete_kernel(genuine):
    result = audit_genuine(genuine)
    assert (result["rank"], result["nullity"]) == (15, 3)
    diag = result["source_kernel_report"]
    assert diag["source_geometric_complete_kernel_verified"]
    assert diag["source_geometric_witnesses_exhaust_full_normalized_nullity"]
    assert diag["source_geometric_kernel_machine_operand_relative_residual"] < 256*oracle.EPS
    assert diag["source_geometric_projector_maximum_absolute_difference"] < 2e-15


def test_old_frozen_projector_comparison_refuses_this_same_genuine_source(genuine):
    raw = genuine["release_modes"]
    with pytest.raises(ValueError, match="physical tied kernel projector"):
        old.audit_jump_modes(raw, np.asarray(raw["trace_xyz"]),
                             np.asarray(raw["reference_area_weights_m2"]))


def test_rounded_numeric_projector_drift_is_bounded_conditioning_diagnostic(genuine):
    diag = audit_genuine(genuine)["source_kernel_report"]
    assert diag["rounded_numerical_projector_maximum_absolute_difference"] > 1e-4
    assert 1e-14 < diag["minimum_retained_unscaled_weighted_jump_singular_value_m"] < 1e-12
    assert diag["retained_unscaled_weighted_jump_condition_number"] > 1e12
    assert diag["rounded_numerical_source_subspace_angle"] <= diag["rounded_numerical_source_subspace_angle_bound"]
    assert diag["numerical_projector_is_conditioning_diagnostic_only"]
    assert not diag["conditioning_or_roundoff_available_for_funding"]
    assert not diag["fresh_physical_weak_source_accuracy_certified"]
    assert not diag["declared_scaled_svd_diagnostics_independently_recomputed"]
    assert diag["declared_scaled_svd_conditioning_diagnostics"]["analytic_null_subspace_error"] > 1e-4


def test_actual_source_projector_uses_machine_bound_not_relaxed_numeric_tolerance(genuine):
    diag = audit_genuine(genuine)["source_kernel_report"]
    assert diag["source_projector_comparison_arithmetic_bound"] == 256*oracle.EPS*18
    assert diag["source_projector_comparison_arithmetic_bound"] < 2e-8


def test_resealed_raw_svd_projector_cannot_replace_source_geometric_kernel(genuine):
    raw = genuine["release_modes"].copy()
    numeric = oracle.independent_jump_modes(raw["trace_xyz"], raw["reference_area_weights_m2"])
    raw["tied_null_modes"] = numeric["tied_null_modes"].tolist()
    raw["release_basis"] = numeric["release_basis"].tolist()
    with pytest.raises(ValueError, match="physical tied kernel projector"):
        audit_genuine(genuine, reseal(raw))


def test_resealed_invented_analytic_witness_is_refused(genuine):
    raw = genuine["release_modes"].copy()
    phi = np.asarray(raw["analytic_null_modes"]).copy()
    phi[0, 0] += 1e-12
    raw["analytic_null_modes"] = phi.tolist()
    with pytest.raises(ValueError, match="authentic source geometric kernel witnesses"):
        audit_genuine(genuine, reseal(raw))


def test_independently_changed_original_source_geometry_cannot_reuse_declared_phi(genuine):
    chart = {**genuine["chart"], "d": genuine["chart"]["d"]+1.}
    raw = genuine["release_modes"]
    with pytest.raises(ValueError, match="authentic source geometric kernel witnesses"):
        oracle.audit_jump_modes(raw, np.asarray(raw["trace_xyz"]),
            np.asarray(raw["reference_area_weights_m2"]),
            source_null_modes=oracle.source_geometric_null_modes(chart))


def test_complete_kernel_dimension_cannot_be_inferred_from_three_supplied_witnesses(genuine):
    raw = genuine["release_modes"]
    numeric = oracle.independent_jump_modes(raw["trace_xyz"], raw["reference_area_weights_m2"])
    numeric.update(rank=14, nullity=4)
    with pytest.raises(ValueError, match="do not exhaust the complete tied kernel"):
        oracle.source_complete_kernel(numeric, raw["trace_xyz"], raw["reference_area_weights_m2"],
                                      oracle.source_geometric_null_modes(genuine["chart"]))


def test_source_witness_machine_residual_has_no_absolute_floor(genuine):
    raw = genuine["release_modes"]
    trace = np.asarray(raw["trace_xyz"])*1e-30
    numeric = oracle.independent_jump_modes(trace, raw["reference_area_weights_m2"])
    phi = oracle.source_geometric_null_modes(genuine["chart"]).copy()
    strong = numeric["release_basis"][:, 0]
    phi[:, 0] += .1*strong
    with pytest.raises(ValueError, match="machine source-operand residual"):
        oracle.source_complete_kernel(numeric, trace, raw["reference_area_weights_m2"], phi)


def test_source_trace_machine_residual_refuses_manufactured_jump(genuine):
    raw = genuine["release_modes"]
    trace = np.asarray(raw["trace_xyz"]).copy()
    phi = oracle.source_geometric_null_modes(genuine["chart"])
    numeric = oracle.independent_jump_modes(trace, raw["reference_area_weights_m2"])
    trace[0, 0] += .01*phi[:, 0]
    with pytest.raises(ValueError, match="machine source-operand residual"):
        oracle.source_complete_kernel(numeric, trace, raw["reference_area_weights_m2"], phi)


def test_independent_source_witnesses_are_required(genuine):
    raw = genuine["release_modes"]
    numeric = oracle.independent_jump_modes(raw["trace_xyz"], raw["reference_area_weights_m2"])
    phi = oracle.source_geometric_null_modes(genuine["chart"])
    phi[:, 1] = phi[:, 0]
    with pytest.raises(ValueError, match="must be independent"):
        oracle.source_complete_kernel(numeric, raw["trace_xyz"], raw["reference_area_weights_m2"], phi)


def test_source_subspace_sign_and_basis_rotations_preserve_kernel(genuine):
    raw = genuine["release_modes"].copy()
    rotation = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., -1.]])
    raw["tied_null_modes"] = (np.asarray(raw["tied_null_modes"])@rotation).tolist()
    raw["release_basis"] = (-np.asarray(raw["release_basis"])).tolist()
    assert audit_genuine(genuine, reseal(raw))["nullity"] == 3


def test_resealed_free_complete_kernel_is_refused(genuine):
    raw = genuine["release_modes"].copy()
    raw.update(rank=18, nullity=0, tied_null_modes=[[] for _ in range(18)],
               release_basis=np.eye(18).tolist())
    with pytest.raises(ValueError, match="kernel dimension"):
        audit_genuine(genuine, reseal(raw))


def test_tiny_complete_kernel_coordinate_remains_refused_without_projection(genuine):
    modes = audit_genuine(genuine)
    displacement = modes["declared_tied_null_modes"][:, 0]*1e-40
    with pytest.raises(ValueError, match="complete jump-kernel component"):
        oracle.require_tied_kernel(modes, displacement, "tiny released kernel")


@pytest.mark.parametrize("key", ["physical_release_admitted", "funding_admitted"])
def test_geometric_kernel_diagnostic_cannot_claim_physical_admission(genuine, key):
    raw = genuine["release_modes"].copy()
    raw[key] = True
    with pytest.raises(ValueError, match="complete jump kernel cannot be released"):
        audit_genuine(genuine, reseal(raw))


def test_no_other_literal_force_memory_or_work_function_was_changed():
    previous, current = function_bodies(old), function_bodies(oracle)
    changed = {name for name in previous if previous[name] != current[name]}
    assert changed == {"audit_jump_modes", "audit_release_context"}


def test_no_native_history_or_current_work_function_was_changed():
    from analysis.genesis_volume_release_validation import independent_native_audit as previous
    original, current = function_bodies(previous), function_bodies(native)
    changed = {name for name in original if original[name] != current[name]}
    assert changed == {"audit_fixture"}


def test_v2_direct_script_help_works_without_external_pythonpath(tmp_path):
    import os
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(Path(native.__file__).resolve()), "--help"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--previous-memory-fixture" in result.stdout and "--control-directory" in result.stdout


def test_v2_oracles_remain_producer_free():
    for module in (oracle, native):
        syntax = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(syntax):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("tectonics")
            if isinstance(node, ast.Import):
                assert not any(alias.name.startswith("tectonics") for alias in node.names)
    assert native.literal is oracle
    assert native.FORMAT == "independent-genesis-native-release-work-audit-0.2"


def audit_observable(genuine, raw=None):
    modes = audit_genuine(genuine)
    original = genuine["release_modes"]
    return oracle.audit_observable_source_svd(genuine["observable_source_context"] if raw is None else raw,
        modes, np.asarray(original["trace_xyz"]), np.asarray(original["reference_area_weights_m2"]))


def test_genuine_observable_basis_matches_literal_source_svd_exactly(genuine):
    diag = audit_observable(genuine)
    assert diag["source_observable_svd_construction_verified"]
    assert diag["source_observable_svd_basis_reconstruction_byteexact"]
    assert diag["source_observable_svd_singular_values_reconstruction_byteexact"]
    assert [(c["first_coordinate"], c["stop_coordinate_exclusive"])
            for c in diag["source_observable_svd_clusters"]] == [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]


def test_genuine_weak_pair_dot_failure_remains_explicit_without_admission_credit(genuine):
    diag = audit_observable(genuine)
    assert not diag["former_weak_relative_orthogonality_criterion_all_passed"]
    assert len(diag["literal_weighted_trace_pair_dots"]) == 105
    assert max(abs(p["literal_weighted_trace_dot_m2"])
               for p in diag["literal_weighted_trace_pair_dots"] if p["coordinate_j"] >= 9) > 1e-16
    assert diag["weighted_trace_column_orthogonality_is_arithmetic_diagnostic_only"]
    assert not diag["source_observable_svd_precision_bounds_enter_primary_work_or_momentum_admission"]
    assert not diag["conditioning_or_roundoff_available_for_funding"]
    assert not diag["fresh_physical_weak_source_accuracy_certified"]


@pytest.mark.parametrize("start", [0, 9, 12])
def test_source_observable_svd_accepts_signs_and_degenerate_cluster_rotations(genuine, start):
    raw = genuine["observable_source_context"].copy()
    W = np.asarray(raw["observable_force_basis"]).copy()
    rotation = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., -1.]])
    W[:, start:start+3] = W[:, start:start+3]@rotation
    raw["observable_force_basis"] = W.tolist()
    assert audit_observable(genuine, raw)["source_observable_svd_construction_verified"]


def test_source_observable_svd_rejects_orthogonal_strong_weak_mixing(genuine):
    raw = genuine["observable_source_context"].copy()
    W = np.asarray(raw["observable_force_basis"]).copy()
    angle = 1e-6
    W[:, [0, 9]] = W[:, [0, 9]]@np.asarray([[np.cos(angle), -np.sin(angle)],
                                             [np.sin(angle), np.cos(angle)]])
    np.testing.assert_allclose(W.T@W, np.eye(15), atol=3e-15)
    raw["observable_force_basis"] = W.tolist()
    with pytest.raises(ValueError, match="SVD cluster projector differs"):
        audit_observable(genuine, raw)


def test_source_observable_svd_rejects_mixing_distinct_resolved_weak_clusters(genuine):
    raw = genuine["observable_source_context"].copy()
    W = np.asarray(raw["observable_force_basis"]).copy()
    angle = 1e-5
    W[:, [9, 12]] = W[:, [9, 12]]@np.asarray([[np.cos(angle), -np.sin(angle)],
                                              [np.sin(angle), np.cos(angle)]])
    raw["observable_force_basis"] = W.tolist()
    with pytest.raises(ValueError, match="SVD cluster projector differs"):
        audit_observable(genuine, raw)


def test_source_observable_svd_has_no_absolute_weak_singular_value_floor(genuine):
    raw = genuine["observable_source_context"].copy()
    values = np.asarray(raw["observable_force_singular_values_m"]).copy()
    values[-1] += 1e-17  # The former 1e-16 absolute allowance hid this error.
    raw["observable_force_singular_values_m"] = values.tolist()
    with pytest.raises(ValueError, match="own relative scales"):
        audit_observable(genuine, raw)


def test_source_observable_svd_rejects_claimed_different_basis_scope(genuine):
    raw = genuine["observable_source_context"].copy()
    raw["momentum_force_basis_scope"] = "manufactured_weak_relative_exact_eigenbasis"
    with pytest.raises(ValueError, match="SVD convention scope"):
        audit_observable(genuine, raw)
