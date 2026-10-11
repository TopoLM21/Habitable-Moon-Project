"""Synthetic active integration controls; no planetary activation is asserted."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy import linalg

from analysis.genesis_volume_slip_validation.coupled_material_control import (
    FORMAT, SCOPE, build_control, canonical)
from tectonics.genesis_volume_dynamics import coefficient_arrays
from tectonics.genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json"


@pytest.fixture(scope="module")
def coupled():
    source_bytes = SOURCE.read_bytes()
    packet = json.loads(source_bytes)
    solve = linalg.solve
    with patch("tectonics.genesis_volume_slip_dynamics.linalg.solve", wraps=solve) as calls:
        result = build_control(packet)
    assert SOURCE.read_bytes() == source_bytes
    return (*result, [call.kwargs.get("assume_a") for call in calls.call_args_list])


def test_fixture_explicitly_declares_seed_force_and_refuses_planetary_claims(coupled):
    model, literal, seeded, first, second, body, _ = coupled
    assert body["format"] == FORMAT and body["scope"] == SCOPE
    assert body["synthetic_forcing_and_seed_not_planetary_continuation"]
    for name in ("actual_weak_plane_activation_observed", "actual_planet_activation_claimed",
                 "sharp_banks_created", "physical_fault_contact_admission_claimed",
                 "original_point_1_completed", "genuine_planetary_continuation_claimed",
                 "authentic_checkpoint_claimed", "actual_source_replaced"):
        assert body[name] is False
    assert not literal.fault_history.fault_active.any()
    assert seeded.fault_history.fault_active.sum() == 1
    assert body["synthetic_seed"]["origin"] == "prescribed_synthetic_seed_not_calculated_Genesis_activation"
    assert seeded.fault_history.damage[0] < model.fault_parameters.activation_damage
    assert seeded.fault_history.water_access[0] == .25
    assert literal.fault_history.water_access[0] == 0
    assert body["synthetic_continuation_basal_drag_pa_s_m"] == 1e7


def test_original_geometry_memory_thermal_birth_and_source_are_unmodified(coupled):
    model, literal, seeded, _, _, body, _ = coupled
    assert literal.geometry is seeded.geometry
    assert literal.mechanics is seeded.mechanics
    assert literal.thermal is seeded.thermal
    assert literal.history is seeded.history
    assert canonical(model.source_packet) == canonical(json.loads(SOURCE.read_bytes()))
    for a, b in zip(coefficient_arrays(literal.mechanics), coefficient_arrays(seeded.mechanics)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(literal.fault_history.damage, seeded.fault_history.damage)
    assert body["initial_memory_unchanged"] and body["initial_geometry_thermal_and_birth_unchanged"]


def test_two_intervals_exercise_nonassociated_global_solver_and_both_slip_losses(coupled):
    model, _, _, first, second, _, solve_kinds = coupled
    assert solve_kinds.count("gen") >= 4  # predictor and final for both intervals
    for state in (first, second):
        model.validate_state(state)
        report = state.last_report["weak_plane_slip"]
        assert report["newton_iterations"] > 0
        assert report["normal_stress_pa"][0] < 0
        assert state.last_report["synthetic_fixture_tangent_asymmetry_relative_norm"] > 0
        assert report["friction_work_j"] > 0 and report["viscous_fault_work_j"] > 0
        assert 0 < abs(report["shear_increment"][0]) <= model.fault_parameters.max_shear_increment
        assert report["coupled_momentum_relative_residual"] <= 1e-11


def test_cumulative_net_total_and_fault_work_count_only_final_accepted_solves(coupled):
    _, _, seeded, first, second, _, _ = coupled
    reports = [state.last_report["weak_plane_slip"] for state in (first, second)]
    assert second.physical_mechanical_work_j == pytest.approx(
        sum(r["physical_mechanical_work_j"] for r in reports), rel=2e-15)
    assert second.mechanics.mechanical_work_j == pytest.approx(
        sum(r["net_constitutive_driver_work_j"] for r in reports), rel=2e-15)
    assert second.fault_friction_dissipation_j == sum(r["friction_work_j"] for r in reports)
    assert second.fault_viscous_dissipation_j == sum(r["viscous_fault_work_j"] for r in reports)
    losses = second.fault_friction_dissipation_j+second.fault_viscous_dissipation_j
    assert second.physical_mechanical_work_j-second.mechanics.mechanical_work_j == pytest.approx(losses, rel=2e-11)
    assert second.fault_history.friction_work_j-seeded.fault_history.friction_work_j == second.fault_friction_dissipation_j
    assert second.fault_history.viscous_fault_work_j-seeded.fault_history.viscous_fault_work_j == second.fault_viscous_dissipation_j
    np.testing.assert_allclose(second.fault_history.signed_shear,
        np.asarray(reports[0]["shear_increment"])+reports[1]["shear_increment"], rtol=1e-15)


def test_global_cumulative_balance_includes_slip_as_separate_additional_loss(coupled):
    _, _, _, first, second, _, _ = coupled
    for state in (first, second):
        m = state.mechanics
        losses = state.fault_friction_dissipation_j+state.fault_viscous_dissipation_j
        expected = (m.initial_energy_j+state.initial_kinetic_energy_j+state.external_work_j
            +m.thermal_work_j-m.viscous_dissipation_j-state.basal_dissipation_j-losses
            +m.damage_energy_change_j-m.remelt_energy_removed_j+state.phase_kinetic_energy_change_j)
        residual = m.elastic_energy_j+state.kinetic_energy_j-expected
        scale = max(m.initial_energy_j, m.elastic_energy_j, abs(state.external_work_j), 1.)
        assert abs(residual) < 5e-12*scale
        assert residual == pytest.approx(state.global_balance_residual_j, abs=32*np.finfo(float).eps*scale)
        assert losses > 1000*abs(residual)


def test_accepted_start_water_is_frozen_across_predictor_and_final(coupled):
    _, _, seeded, first, second, _, _ = coupled
    for before, after in ((seeded, first), (first, second)):
        report = after.last_report["weak_plane_slip"]
        predictor = after.last_report["fault_evolution"]["predictor_driver"]["weak_plane_slip"]
        np.testing.assert_array_equal(report["water"], before.fault_history.water_access)
        np.testing.assert_array_equal(predictor["water"], before.fault_history.water_access)
        np.testing.assert_array_equal(report["endpoint_water_access"], after.fault_history.water_access)
        assert report["water"][0] != report["endpoint_water_access"][0]


def test_top_face_covector_transport_matches_independent_material_line(coupled):
    _, _, seeded, first, second, _, _ = coupled
    face = 0
    for before, after in ((seeded, first), (first, second)):
        vertices = np.asarray(before.geometry.faces[face])
        triangles = []
        for state in (before, after):
            physical = np.asarray(state.geometry.vertices_m)+state.displacement_m.reshape(-1, 6)[:, :3]
            triangles.append(physical[vertices])
        edges = [np.stack((x[1]-x[0], x[2]-x[0]), axis=1) for x in triangles]
        material_line_coefficients = np.linalg.lstsq(edges[0], before.slip_direction_world[face], rcond=None)[0]
        endpoint_line = edges[1]@material_line_coefficients
        endpoint_normal = np.cross(edges[1][:, 0], edges[1][:, 1])
        endpoint_normal /= np.linalg.norm(endpoint_normal)
        independent_co_normal = np.cross(endpoint_line, endpoint_normal)
        independent_co_normal /= np.linalg.norm(independent_co_normal)
        np.testing.assert_allclose(after.plane_normal_world[face], independent_co_normal, rtol=0, atol=2e-12)
        report = after.last_report["weak_plane_slip"]
        np.testing.assert_array_equal(report["frozen_start_plane_normal_world"], before.plane_normal_world)
        np.testing.assert_array_equal(report["frozen_start_slip_direction_world"], before.slip_direction_world)
        assert np.linalg.norm(after.plane_normal_world[face]-before.plane_normal_world[face]) > 1e-9


def test_shared_formation_clock_birth_journal_and_all_cohort_memory_continue(coupled):
    _, _, seeded, first, second, _, _ = coupled
    assert len(second.accepted_step_sizes_myr) == second.mechanics.accepted_steps == 2
    for before, after in ((seeded, first), (first, second)):
        assert after.time_myr == after.geometry.time_myr == after.thermal.time_myr == after.history.time_myr
        assert after.time_myr == after.fault_history.time_myr == after.mechanics.time_myr
        assert after.history.accepted_intervals == before.history.accepted_intervals+1
        assert after.history.intervals[:len(before.history.intervals)] == before.history.intervals
        assert len(after.mechanics.cohorts) > len(before.mechanics.cohorts)
        assert tuple(c.id for c in after.geometry.cohorts) == tuple(c.id for c in after.mechanics.cohorts)
        assert after.last_certificate.whole_clock_all_pair_noninterpenetration_certified
        assert not np.array_equal(after.fault_history.damage, before.fault_history.damage)
        assert np.count_nonzero(coefficient_arrays(after.mechanics)[1]) > 0


def test_fixed_synthetic_force_is_bound_into_every_global_audit_input(coupled):
    model, _, _, first, second, body, _ = coupled
    for state in (first, second):
        inputs = state.last_report["audit_input"]
        np.testing.assert_array_equal(inputs["synthetic_additional_external_force_n"], body["synthetic_force_n"])
        np.testing.assert_array_equal(inputs["synthetic_additional_external_force_n"], model.synthetic_force_n)
        np.testing.assert_allclose(np.asarray(inputs["source_basal_external_force_n"])
            +inputs["synthetic_additional_external_force_n"], inputs["total_external_force_n"], rtol=0, atol=1.)
        assert inputs["forcing_scope"] == SCOPE


def test_synthetic_views_cannot_be_saved_as_an_authentic_checkpoint(coupled):
    model, _, _, _, second, body, _ = coupled
    with pytest.raises(ValueError, match="source-bound"):
        volume_slip_checkpoint_to_dict(model, second)
    for name in ("initial_packet", "after_first_packet", "after_second_packet"):
        packet = body[name]
        assert packet["format"] == "synthetic-volume-slip-runtime-view-0.1"
        assert packet["scope"] == SCOPE and packet["authentic_checkpoint_claimed"] is False
        assert "checkpoint_sha256" not in packet
