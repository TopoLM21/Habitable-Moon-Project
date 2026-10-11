"""Physical-state separation and exact handover commuting controls."""
from dataclasses import replace
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

import pytest

from tectonics import geometric_native_common_initialization_source as source_api
from tectonics import geometric_native_common_initialization_handover as api

ROOT = Path(__file__).resolve().parents[1]
PATHS = (
    "analysis/geometric_native_common_general_force_validation/development_native03/native_control.json",
    "analysis/geometric_native_common_source_carrier_validation/release_validation_02/native_control/source_packet.json",
    "analysis/geometric_native_common_mixed_force_validation/release_validation_01/native_control/source_packet.json",
    "analysis/geometric_native_common_cold_material_carrier_validation/release_validation_02/native_control/source_packet.json",
)
QUERY = (F(1), F(0), F(0))


def fraction(raw):
    return F(*raw)


def contains_zero(raw):
    if type(raw) is dict and set(raw) == {"lo", "hi"}:
        return fraction(raw["lo"]) <= 0 <= fraction(raw["hi"])
    return all(contains_zero(x) for x in raw)


@pytest.fixture(scope="module")
def handover():
    packets = tuple(json.loads((ROOT/path).read_bytes()) for path in PATHS)
    context = source_api.bind_native_common_initialization_source(*packets,
        provenance="Complete original accepted state and reference935 initialization test authority")
    system = api.bind_native_common_initialization_handover(context,
        provenance="Exact Cartesian handover controls, preserving historical unknown fields")
    response = api.evaluate_native_common_initialization_handover(system,
        virtual_material_increment_m=QUERY)
    return context, system, response


def test_full32_material_side_catalogue_includes_all_eight_archived_attachments(handover):
    receipt = api.native_common_initialization_handover_receipt(handover[2])
    sides = receipt["side_geometry_receipts"]
    assert len(sides) == 32 and len({(s["point_id"], s["side"]) for s in sides}) == 32
    assert receipt["live_source_side_count"] == 24
    assert receipt["archived_source_side_count"] == 8
    assert {s["target_internal_coordinate_index"] for s in sides} == set(range(92, 100))
    assert sum(s["archive_debits"] for s in sides) == 8


def test_every_original_cartesian_gradient_and_inverse_commutes(handover):
    sides = api.native_common_initialization_handover_receipt(handover[2])["side_geometry_receipts"]
    for side in sides:
        for name in ("roof_bridge_residual", "inverse_residual", "stored_transport_residual",
                     "original_gradient_commuting_residual"):
            assert contains_zero(side[name])
        assert side["matrix_algebra_identity_by_construction"] is True
        assert side["interval_zero_enclosures_are_additional_diagnostics"] is True


def test_nonzero_authentic_roof_correction_is_not_legacy_bq_identity(handover):
    sides = api.native_common_initialization_handover_receipt(handover[2])["side_geometry_receipts"]
    identity = api._identity()
    for side in sides:
        C = api._matrix(side["roof_chart_correction"])
        assert not api._contains_zero(api._msub(C, identity))


def test_existing_archival_arm_and_current_tip_candidate_are_both_retained(handover):
    sides = api.native_common_initialization_handover_receipt(handover[2])["side_geometry_receipts"]
    for side in sides:
        assert contains_zero(side["archived_arm_source_residual_m"])
        assert contains_zero(side["existing_attachment_commuting_residual_m"])
        assert contains_zero(side["current_minus_archived_attachment_commuting_residual_m"])
        assert side["current_tip_affine_candidate_applied"] is False
        assert fraction(side["existing_attachment_minus_current_tip_squared_norm_m2"]["lo"]) > 0


def test_material_reference_change_is_real_for_all32_source_labels(handover):
    receipt = api.native_common_initialization_handover_receipt(handover[2])
    assert receipt["received_material_embedding_preserved"] is False
    for side in receipt["side_geometry_receipts"]:
        assert side["material_position_change_certified"] is True
        assert fraction(side["material_position_change_squared_norm_m2"]["lo"]) > 0


def test_all_admissible_receiver_radial_gap_uses_actual_full_chart_envelope(handover):
    receipt = api.native_common_initialization_handover_receipt(handover[2])
    gaps = receipt["allq_receiver_reference_gap_receipts"]
    assert len(gaps) == 16
    for row in gaps:
        assert fraction(row["squared_radius_separation_lower_m2"]) > 0
        assert row["whole895_cold_envelope_matches935_actual_cold_chart"] is True
        assert row["accepted_current_mechanical_tip_representable_as_declared935_cold_material_point"] is False
        assert row["registered879_material_reference_representable_as_declared935_cold_material_point"] is False
        assert row["historical_physical_material_embedding_preserved"] is None
        assert row["whole_historical_physical_material_embedding_nonrepresentability_certified"] is False
        assert row["internal_virtual_source_attachments_nonrepresentability_certified"] is False
        assert fraction(row["target_minus_old_registered_reference_squared_radius_separation_lower_m2"]) > 0
        assert row["physical_preparation_path_ruled_out"] is False
        assert row["radial_separation_is_conditional_declared_chart_result"] is True


def test_covectors_retain_three_distinct_old_force_bases(handover):
    receipt = api.native_common_initialization_handover_receipt(handover[2])
    for side in receipt["side_geometry_receipts"]:
        rows = side["old_force_basis_comparisons"]
        assert [r["basis"] for r in rows] == ["literal_archival_producer", "archival_old_reference", "current_objective"]
        for row in rows:
            assert contains_zero(row["candidate_force_pullback_commuting_residual_n"])
            assert contains_zero(row["candidate_virtual_work_commuting_residual_j"])
        assert contains_zero(side["covector_virtual_work_pairing_residual_j"])
        assert side["formal_virtual_work_is_physical_path_work"] is False


@pytest.mark.parametrize("query", [(F(0), F(1), F(0)), (F(0), F(0), F(1)),
                                  (F(1, 2), F(-2), F(3, 7))])
def test_independent_cartesian_virtual_increments_preserve_conjugate_work(handover, query):
    response = api.evaluate_native_common_initialization_handover(handover[1],
        virtual_material_increment_m=query)
    receipt = api.native_common_initialization_handover_receipt(response)
    assert response.virtual_material_increment_m == query
    assert receipt["virtual_material_increment_m"] == [[x.numerator, x.denominator] for x in query]
    assert all(contains_zero(s["covector_virtual_work_pairing_residual_j"])
               for s in receipt["side_geometry_receipts"])


def test_source14_force_sign_and_true_parent_projection_are_explicit(handover):
    row = api.native_common_initialization_handover_receipt(handover[2])["source14_covector_comparison"]
    assert set(row["target_diagnostic_indices14"]) == set(range(6)) | set(range(92, 100))
    gradient = [api._interval(v) for v in row["exact_archival_source_potential_gradient14"]]
    force = [api._interval(v) for v in row["exact_archival_source_physical_covector14"]]
    assert all(api._contains_zero(g+f) for g, f in zip(gradient, force))
    assert row["receiver_frame_spin_retained_in_scalar_driver"] is True
    assert row["source14_projection_is_diagnostic_only"] is True
    assert row["unmeasured86_body_coordinates_assigned_zero"] is False
    assert not contains_zero(row["current_objective_minus_exact_archival_gradient14"])


def test_sixteen_contact_offsets_and_rounded_free_anchor_residuals_survive(handover):
    atoms = api.native_common_initialization_handover_receipt(handover[2])["contact_force_work_receipts"]
    assert len(atoms) == 16
    for atom in atoms:
        assert fraction(atom["source_energy_rounding_offset_j"]) != 0
        assert fraction(atom["target_anchor_minus_source_stored_free_anchor_m"]) != 0
        assert contains_zero(atom["target_zero_energy_minus_literal_source_energy_j"])
        assert atom["initial_normal_max_opening_m"] is None
        assert atom["initial_normal_damage"] is None
        assert atom["missing_normal_history_replaced_by_virgin_seed"] is False


def test_eight_beam_literals_do_not_hide_the_aggregate_ledger_rounding(handover):
    receipt = api.native_common_initialization_handover_receipt(handover[2])
    assert len(receipt["beam_energy_receipts"]) == 8
    ledger = receipt["source_energy_work_ledger_receipt"]
    assert fraction(ledger["individual_minus_aggregate_beam_energy_j"]) == -F(3, 131072)
    assert fraction(ledger["literal_source_reference_relabeling_energy_residual_j"]) == -F(3, 131072)
    assert ledger["physical_preparation_or_motion_work_j"] is None
    assert ledger["assumed_physical_duration_s"] is None
    assert ledger["additional_lambda_drive_work_j"] is None
    assert ledger["work_ledgers_changed"] is False


def test_saved_continuum_zero_diagnostics_do_not_fill_missing_historical86_fields(handover):
    row = api.native_common_initialization_handover_receipt(handover[2])["continuum_initialization_receipt"]
    assert row["historical_cold56_deformation_coordinates_m"] is None
    assert row["historical_hot30_deformation_coordinates_m"] is None
    assert row["historical_original_parent_3d_prestress_field"] is None
    assert row["historical_to_declared_continuum_energy_residual_j"] is None
    assert all(value is None for value in row["unknown_physical_inputs"].values())
    assert row["physical_continuum_initialization_ready"] is False
    assert row["missing_historical_fields_replaced_by_zero"] is False


def test_accepted_original_state_and_old_memory_are_unchanged(handover):
    context, _, response = handover
    source = source_api.native_common_initialization_source_evidence(context)
    receipt = api.native_common_initialization_handover_receipt(response)
    assert receipt["accepted_source"] == source["accepted_source"]
    assert receipt["target_retained_reference_clocks_s"] == [[0, 1], [2000000, 1], [1000000, 1], [1000000, 1]]
    assert receipt["accepted_source"]["accepted_state_clocks_s"]["source_geometry"] == [1000000, 1]
    assert response.accepted_original_native_intervals == 1
    assert response.accepted_new_native_intervals == 0
    assert response.physical_initialization_ready is False
    assert response.original_point_1_completed is False


def test_system_restart_uses_the_complete_authenticated_source(handover):
    context, system, _ = handover
    restored = api.native_common_initialization_handover_from_dict(
        json.loads(json.dumps(api.native_common_initialization_handover_to_dict(system))), context=context)
    assert api.native_common_initialization_handover_to_dict(restored) == api.native_common_initialization_handover_to_dict(system)


def test_response_restart_recomputes_the_expected_virtual_increment(handover):
    _, system, response = handover
    dto = json.loads(json.dumps(api.native_common_initialization_handover_response_to_dict(response)))
    restored = api.native_common_initialization_handover_response_from_dict(dto, system=system,
        expected_virtual_material_increment_m=QUERY)
    assert api.native_common_initialization_handover_response_to_dict(restored) == dto
    with pytest.raises(ValueError, match="result restart"):
        api.native_common_initialization_handover_response_from_dict(dto, system=system,
            expected_virtual_material_increment_m=(F(0), F(1), F(0)))


@pytest.mark.parametrize("query", [[F(1), F(0), F(0)], (F(1), F(0)),
                                  (1, F(0), F(0)), (True, F(0), F(0)),
                                  (1.0, F(0), F(0))])
def test_virtual_material_query_requires_exact_complete_typed_scalars(handover, query):
    with pytest.raises(ValueError, match="exact Cartesian"):
        api.evaluate_native_common_initialization_handover(handover[1], virtual_material_increment_m=query)


def test_resealed_physical_success_or_boolean_integer_alias_is_not_registered(handover):
    system, response = handover[1:]
    for field, value in (("physical_initialization_ready", True), ("accepted_new_native_intervals", False)):
        forged = replace(response, **{field: value}, binding_sha256="")
        forged = replace(forged, binding_sha256=api._digest(api._payload(api._body(forged))))
        with pytest.raises(ValueError, match="altered or resealed"):
            api.native_common_initialization_handover_response_to_dict(forged)
    altered_system = replace(system, physical_initialization_ready=True, binding_sha256="")
    altered_system = replace(altered_system, binding_sha256=api._digest(api._payload(api._body(altered_system))))
    with pytest.raises(ValueError, match="altered or resealed"):
        api.native_common_initialization_handover_to_dict(altered_system)


def test_resealed_result_cannot_replace_unknown_normal_history_by_zero(handover):
    dto = api.native_common_initialization_handover_response_to_dict(handover[2])
    receipt = json.loads(dto["receipt_json"])
    receipt["contact_force_work_receipts"][0]["initial_normal_damage"] = [0, 1]
    dto["receipt_json"] = api._json(receipt)
    dto["binding_sha256"] = api._digest({k: v for k, v in dto.items() if k not in ("binding_sha256", "format")})
    with pytest.raises(ValueError, match="result restart"):
        api.native_common_initialization_handover_response_from_dict(dto, system=handover[1],
            expected_virtual_material_increment_m=QUERY)


def test_public_receipt_reads_cannot_mutate_saved_initialization(handover):
    response = handover[2]
    receipt = api.native_common_initialization_handover_receipt(response)
    receipt["physical_initialization_ready"] = True
    receipt["side_geometry_receipts"].clear()
    fresh = api.native_common_initialization_handover_receipt(response)
    assert fresh["physical_initialization_ready"] is False
    assert len(fresh["side_geometry_receipts"]) == 32


def test_exact_nonorthogonal_map_requires_covector_inverse_transpose():
    I = api._I
    scalar = lambda x: I(F(x), F(x))
    T = tuple(tuple(scalar(x) for x in row) for row in ((2, 1, 0), (0, 3, 0), (0, 0, 4)))
    source_force = tuple(scalar(x) for x in (5, -7, 11))
    delta = tuple(scalar(x) for x in (2, 3, -1))
    force_new = api._mv(api._transpose(api._inverse(T)), source_force)
    assert api._contains_zero(api._dot(force_new, api._mv(T, delta))-api._dot(source_force, delta))
    assert not api._contains_zero(api._dot(api._mv(T, source_force), api._mv(T, delta))-api._dot(source_force, delta))


def test_unrounded_fraction_interval_inverse_preserves_an_actual_positive_map():
    scalar = lambda x: api._I(F(x), F(x))
    A = tuple(tuple(scalar(x) for x in row) for row in ((2, 1, 1), (0, 3, 1), (0, 0, 4)))
    residual = api._msub(api._mm(api._inverse(A), A), api._identity())
    assert api._contains_zero(residual)
    assert all(x.lo == x.hi == 0 for row in residual for x in row)
