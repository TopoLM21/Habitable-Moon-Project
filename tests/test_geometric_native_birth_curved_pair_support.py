"""Correlated signed curved masks and short lived cold native fronts."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
from functools import lru_cache
import json
import math
from pathlib import Path

import pytest

from tectonics.geometric_native_birth_curved_pair_support import (
    _chart_plane_polynomial, _prove_chart_plane, certify_native_curved_pair_support,
    certify_native_cold5_receiver1_immediate_slice,
    certify_native_curved_related_cold_front, native_curved_pair_support_from_dict,
    native_curved_pair_support_to_dict, validate_native_curved_pair_support,
)
from tectonics.geometric_native_birth_pair_exclusion import UnresolvedNativePairExclusion
from tectonics.geometric_native_polygons import NativePlane
from tectonics.geometric_native_pose import FractionInterval as I
from tests.test_geometric_native_implicit_epoch_admission import get_actual_newborn_geometry


def test_negative_derivative_preserves_exact_mask_when_rectangle_has_positive_upper():
    chart = (I(0, F(1, 100)), I(-F(3, 10), -F(2, 10)), I(1))
    plane = NativePlane((-1, 0, 0))
    proof = _prove_chart_plane(chart, plane, plane, (0., 0., 2.),
        (0., 0., 1.), .01, 128, 20, 0, 0)
    assert proof.whole_time_support_bounds.hi > 0
    assert proof.whole_time_derivative_bounds.hi < 0
    assert proof.initial_support_bounds.hi == 0
    assert proof.method == "exact_outside_endpoint_archive_and_negative_derivative"
    assert proof.initial_support_constraint == "signed_correction_equals_complete_swept_archive_minus_endpoint_archive"


def test_initial_mask_assumption_requires_matching_endpoint_support():
    chart = (I(0, F(1, 100)), I(-F(3, 10), -F(2, 10)), I(1))
    with pytest.raises(UnresolvedNativePairExclusion):
        _prove_chart_plane(chart, NativePlane((-1, 0, 0)), NativePlane((0, -1, 0)),
            (0., 0., 2.), (0., 0., 1.), .01, 128, 20, 0, 0)


def test_unoriented_plane_identity_cannot_reverse_the_endpoint_mask():
    chart = (I(-F(1, 100), 0), I(F(2, 10), F(3, 10)), I(1))
    with pytest.raises(UnresolvedNativePairExclusion):
        _prove_chart_plane(chart, NativePlane((1, 0, 0)), NativePlane((-1, 0, 0)),
            (0., 0., 2.), (0., 0., 1.), .01, 128, 20, 0, 0)


def test_positive_derivative_cannot_authorize_the_outside_mask():
    chart = (I(0, F(1, 100)), I(-F(3, 10), -F(2, 10)), I(1))
    plane = NativePlane((-1, 0, 0))
    with pytest.raises(UnresolvedNativePairExclusion):
        _prove_chart_plane(chart, plane, plane, (0., 0., 1.),
            (0., 0., 2.), .01, 128, 20, 0, 0)


def test_direct_z_separation_does_not_need_an_endpoint_mask_assumption():
    chart = (I(0, F(1, 100)), I(-F(3, 10), -F(2, 10)), I(1))
    proof = _prove_chart_plane(chart, NativePlane((0, 0, -1)), NativePlane((-1, 0, 0)),
        (0., 0., 2.), (0., 0., 1.), .01, 128, 20, 0, 1)
    assert proof.whole_time_support_bounds.hi < 0
    assert proof.method == "whole_chart_nonpositive_support"


def test_scalar_bounds_enclose_independent_exact_axis_rotation_samples():
    chart = (I(F(1, 100)), I(-F(1, 5)), I(1))
    coefficients, tail, derivative_tail, support, derivative = _chart_plane_polynomial(
        chart, NativePlane((-1, 0, 0)), (0., 0., 2.), (0., 0., 1.), .01, 128, 20)
    for index in range(33):
        t = index/32
        value = -.01*math.cos(.01*t)-.2*math.sin(.01*t)
        slope = .0001*math.sin(.01*t)-.002*math.cos(.01*t)
        assert float(support.lo) <= value <= float(support.hi)
        assert float(derivative.lo) <= slope <= float(derivative.hi)
    assert len(coefficients) == 21 and tail > 0 and derivative_tail > 0


@lru_cache(maxsize=1)
def get_curved_pair_case():
    epoch = get_actual_newborn_geometry().source_epoch
    root = Path(__file__).resolve().parents[1]
    inventory = json.loads((root/"analysis/geometric_native_birth_contact_validation/control_release_01/native/mixed_pair_inventory.json").read_text())["inventory"]
    omega = tuple((owner, tuple(vector)) for owner, vector in inventory["omega_rad_per_myr"])
    components = {(row.material_kind, row.original_parent_id): row.component_id for row in epoch.components}
    proof = certify_native_curved_pair_support(epoch, omega, inventory["dt_myr"],
        components[("hot_birth", "octant:5")], components[("original_live", "octant:1")],
        precision=256, order=28, provenance="Actual fresh mixed candidate; exact signed curved exclusion")
    return dict(epoch=epoch, omega=omega, dt=inventory["dt_myr"], proof=proof, components=components)


@pytest.fixture(scope="module")
def case():
    return get_curved_pair_case()


def test_actual_hot5_cold1_complete_signed_mask_is_excluded(case):
    proof = case["proof"]
    assert len(proof.proxy_piece_proofs) == 2
    assert len(proof.curved_piece_proofs) == 1
    curve = proof.curved_piece_proofs[0]
    assert curve.whole_time_support_bounds.hi > 0
    assert curve.whole_time_derivative_bounds.hi < 0
    assert curve.initial_support_bounds.hi == 0
    assert curve.method == "exact_outside_endpoint_archive_and_negative_derivative"
    assert float(curve.whole_time_derivative_bounds.hi) < -1e-26


def test_actual_signed_correction_area_and_both_component_ids_are_preserved(case):
    raw = native_curved_pair_support_to_dict(case["proof"])
    assert raw["whole_material_pair_excluded_for_whole_interval"]
    assert not raw["positive_curved_correction_deleted"]
    assert not raw["negative_curved_correction_deleted"]
    assert F(*raw["positive_curved_measure_bounds_sr"]["lo"]) > 0
    assert raw["negative_component_id"] == case["components"][("original_live", "octant:5")]
    assert not raw["curved_area_assigned_to_polygon"]
    assert not raw["force_admission_authorized"]


def test_actual_signed_curved_support_canonical_restart(case):
    raw = native_curved_pair_support_to_dict(case["proof"])
    restored = native_curved_pair_support_from_dict(raw, expected_epoch=case["epoch"])
    assert native_curved_pair_support_to_dict(restored) == raw


@pytest.mark.parametrize("field,value", [("precision", True), ("dt_myr", 1),
    ("whole_material_pair_excluded_for_whole_interval", 1), ("positive_curved_correction_deleted", True)])
def test_saved_checksum_and_equivalent_numeric_types_do_not_grant_admission(case, field, value):
    raw = native_curved_pair_support_to_dict(case["proof"])
    raw[field] = value
    # Keep the existing root checksum: semantic replay must inspect the full
    # typed source and nested physical proof, not accept a checksum alone.
    with pytest.raises(ValueError):
        native_curved_pair_support_from_dict(raw, expected_epoch=case["epoch"])


def test_nested_positive_derivative_tampering_is_refused(case):
    raw = native_curved_pair_support_to_dict(case["proof"])
    raw["curved_pieces"][0]["whole_time_derivative_bounds"] = {"lo": [0, 1], "hi": [1, 1]}
    with pytest.raises(ValueError):
        native_curved_pair_support_from_dict(raw, expected_epoch=case["epoch"])


def test_typed_positive_derivative_tampering_is_refused(case):
    proof = case["proof"]
    wrong = replace(proof, curved_piece_proofs=(replace(proof.curved_piece_proofs[0],
        whole_time_derivative_bounds=I(0, 1)),))
    with pytest.raises(ValueError):
        validate_native_curved_pair_support(wrong)


def test_exact_cold_front_is_visible_before_disappearing_at_endpoint(case):
    raw = certify_native_curved_related_cold_front(case["epoch"], case["omega"], case["dt"],
        sample_fraction=F(1, 100000), precision=256,
        provenance="Interior clock check on exact lower cold5 masks")
    assert raw["whole_material_pair_has_positive_temporal_subset"]
    assert raw["complete_signed_curved_mask_retained"]
    assert raw["sampled_geometries_are_certified_inner_material_subsets"]
    assert raw["initial_contact_starts_at_arbitrarily_small_positive_time"]
    assert all(F(*row["sample_intersection_area_bounds_sr"]["lo"]) > 0 for row in raw["witnesses"])
    assert all(not row["full_interval_endpoint_has_positive_inner_subset"] for row in raw["witnesses"])
    assert not raw["terminal_contact_clock_proved"]
    assert not raw["force_admission_authorized"]


def test_mixed_flux_cold5_arc_has_true_immediate_receiver1_slice(case):
    raw = certify_native_cold5_receiver1_immediate_slice(case["epoch"], case["omega"], case["dt"],
        precision=256, provenance="Source-bound immediate cold5 receiver1 witness")
    edge = raw["immediate_incoming_edge"]
    assert edge["receiver_inward_plane"]["normal"] == [-1, 0, 0]
    assert all(F(*row["lo"]) > 0 for row in edge["endpoint_incoming_flux_bounds_per_myr"])
    assert F(*edge["interior_incoming_flux_bounds_per_myr"]["lo"]) > 0
    assert raw["contact_exists_at_arbitrarily_small_positive_time"]
    assert raw["negative_curved_correction_retained"]
    for row in raw["receiver_original_gnomonic_coordinate_bounds"]:
        assert row[0] == {"lo": [0, 1], "hi": [0, 1]}
        assert F(*row[1]["hi"]) < -F(1, 10**24)
    assert not raw["complete_incoming_arc_proved"]
    assert not raw["force_admission_authorized"]


def test_other_northern_cold_front_can_be_checked_on_its_own_lower_mask(case):
    raw = certify_native_curved_related_cold_front(case["epoch"], case["omega"], case["dt"],
        donor_original_parent_id="octant:7", receiver_original_parent_id="octant:2",
        sample_fraction=F(1, 100000), precision=256,
        provenance="Distinct source-bound cold7 receiver2 interior witness")
    assert raw["whole_material_pair_has_positive_temporal_subset"]
    assert raw["negative_curved_correction_id"] is None
    assert raw["negative_curved_correction_measure_bounds_sr"] == {"lo": [0, 1], "hi": [0, 1]}
    assert raw["initial_contact_starts_at_arbitrarily_small_positive_time"]
    assert all(not row["full_interval_endpoint_has_positive_inner_subset"] for row in raw["witnesses"])


@pytest.mark.parametrize("sample", [0, F(0), F(2)])
def test_cold_front_rejects_outside_or_untyped_physical_sample(case, sample):
    with pytest.raises(ValueError):
        certify_native_curved_related_cold_front(case["epoch"], case["omega"], case["dt"],
            sample_fraction=sample, precision=256, provenance="Invalid sample")


def _audit_inputs():
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root/"analysis/geometric_native_birth_contact_validation/control_release_01/native/accepted_motion_before.json").read_text())["snapshot"]["current_material_epoch"]
    candidate = json.loads((root/"analysis/geometric_native_birth_front_validation/development_native_continuum_01/candidate_packet.json").read_text())
    proof_path = root/"analysis/geometric_native_birth_front_validation/development_geometry_02/curved_pairs/hot5_to_cold1.json"
    return source, candidate, proof_path


def test_independent_audit_reconstructs_actual_ordered_pose_and_current_series():
    from analysis.geometric_native_birth_front_validation.independent_curved_pair_audit import audit_curved_pair
    source, candidate, path = _audit_inputs()
    receipt = audit_curved_pair(path, source_epoch_payload=source, candidate_payload=candidate)
    assert receipt["actual_candidate_sha256"] == candidate["candidate_sha256"]
    assert receipt["observations"][0]["independently_reconstructed_scalar_coefficients"] == 29
    assert receipt["ordered_old_pose_independently_reconstructed"]
    assert receipt["whole_signed_curved_component_exclusion_independently_proved"]
    assert not receipt["whole_material_pair_exclusion_independently_reproved"]


@pytest.mark.parametrize("tamper", ["coefficient", "derivative_tail", "orientation", "signed_measure", "clock", "omega"])
def test_independent_audit_rejects_physical_nested_tampering(tmp_path, tamper):
    from analysis.geometric_native_birth_front_validation.independent_curved_pair_audit import audit_curved_pair
    source, candidate, path = _audit_inputs()
    proof = json.loads(path.read_text())
    piece = proof["curved_pieces"][0]
    if tamper == "coefficient":
        piece["scalar_coefficients"][1] = {"lo": [0, 1], "hi": [1, 1]}
    elif tamper == "derivative_tail":
        piece["derivative_remainder_bound"] = [0, 1]
    elif tamper == "orientation":
        piece["receiver_inward_plane"]["normal"] = [1, 0, 0]
    elif tamper == "signed_measure":
        proof["positive_curved_measure_bounds_sr"] = {"lo": [0, 1], "hi": [0, 1]}
    elif tamper == "clock":
        piece["physical_fractional_clock"] = {"lo": [1, 2], "hi": [1, 1]}
    else:
        proof["omega_rad_per_myr"][0][1][1] *= -1
    changed = tmp_path/"changed_curved_proof.json"
    changed.write_text(json.dumps(proof))
    with pytest.raises(ValueError):
        audit_curved_pair(changed, source_epoch_payload=source, candidate_payload=candidate)
