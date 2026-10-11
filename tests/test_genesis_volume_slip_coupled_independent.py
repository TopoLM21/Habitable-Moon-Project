"""Independent force/work checks of explicitly synthetic active intervals."""
import ast
import copy
import json
from pathlib import Path

import pytest

from analysis.genesis_volume_slip_validation import coupled_independent_audit as audit
from analysis.genesis_volume_slip_validation.coupled_material_control import build_control


@pytest.fixture(scope="module")
def coupled_fixture():
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root/"analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json").read_text())
    return build_control(source)[-1]


@pytest.fixture(scope="module")
def coupled_report(coupled_fixture):
    return audit.audit_coupled_fixture(coupled_fixture)


def interval_check(raw):
    return audit.audit_coupled_interval(raw["initial_packet"], raw["after_first_packet"], raw["synthetic_force_n"])


def test_coupled_oracle_imports_no_producer_physics():
    for module in (audit, audit.slip, audit.volume, audit.faults):
        for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(name == "tectonics" or name.startswith("tectonics.") for name in names)


def test_two_active_coupled_intervals_have_independent_full_force_and_work(coupled_report):
    assert coupled_report["status"] == "PASS"
    assert coupled_report["initial_formation_material_binding"]["formation_material_memory_unchanged"]
    assert coupled_report["two_active_coupled_accepted_intervals_independently_checked"]
    for interval in coupled_report["intervals"]:
        assert interval["yielding_face_count"] == 1
        assert interval["maximum_absolute_shear_increment"] > 0
        assert interval["independently_checked_full_18_coordinate_force"]
        assert interval["predictor_stress_and_damage_law_independently_recomputed"]
        assert interval["material_quadrature"]["plastic_work_j"] > 0
        assert interval["maximum_cohort_relative_work_residual"] < 3e-11
        assert interval["discrete_dynamic_balance"]["relative_max_coordinate_momentum_residual"] < 2e-8
        assert interval["cumulative_full_work_relative_residual"] < 3e-9
    assert not coupled_report["actual_planet_activation_claimed"]
    assert not coupled_report["sharp_banks_created"]
    assert not coupled_report["original_point_1_completed"]


@pytest.mark.parametrize("claim", ("actual_source_replaced", "actual_planet_activation_claimed",
    "sharp_banks_created", "physical_fault_contact_admission_claimed", "original_point_1_completed"))
def test_synthetic_control_cannot_claim_planetary_history(coupled_fixture, claim):
    raw = copy.deepcopy(coupled_fixture)
    raw[claim] = True
    with pytest.raises(ValueError, match="unsupported claim"):
        audit.audit_coupled_fixture(raw)


def test_synthetic_runtime_cannot_masquerade_as_authentic_checkpoint(coupled_fixture):
    raw = copy.deepcopy(coupled_fixture)
    raw["initial_packet"]["format"] = audit.slip.FORMAT
    with pytest.raises(ValueError, match="synthetic runtime"):
        audit.audit_coupled_fixture(raw)


def test_unlisted_initial_history_change_is_refused(coupled_fixture):
    raw = copy.deepcopy(coupled_fixture)
    raw["initial_packet"]["runtime_state"]["fault_history"]["fault_candidate_age_myr"][1] += .0001
    with pytest.raises(ValueError, match="undeclared source field"):
        audit.audit_coupled_fixture(raw)


@pytest.mark.parametrize("field", ("friction_work_j", "first_fracture_time_myr"))
def test_unlisted_initial_history_scalar_is_refused(coupled_fixture, field):
    raw = copy.deepcopy(coupled_fixture)
    raw["initial_packet"]["runtime_state"]["fault_history"][field] = 1.
    with pytest.raises(ValueError, match="undeclared source scalar"):
        audit.audit_coupled_fixture(raw)


def test_nonfinite_fixture_is_refused(coupled_fixture):
    raw = copy.deepcopy(coupled_fixture)
    raw["synthetic_force_n"][0] = float("nan")
    with pytest.raises(ValueError, match="JSON compliant"):
        audit.audit_coupled_fixture(raw)


def test_inactive_plane_covector_cannot_be_fabricated(coupled_fixture):
    raw = copy.deepcopy(coupled_fixture)
    raw["after_first_packet"]["runtime_state"]["fault_history"]["plane_normal"][1][0] = 1.
    with pytest.raises(ValueError, match="all accepted 2-D plane covectors"):
        interval_check(raw)


def test_persisted_last_traction_must_match_accepted_integrated_slip(coupled_fixture):
    raw = copy.deepcopy(coupled_fixture)
    raw["after_first_packet"]["runtime_state"]["fault_history"]["shear_stress_pa"][0] += 1e7
    with pytest.raises(ValueError, match="accepted last traction/shear history"):
        interval_check(raw)


def test_accepted_start_water_is_not_substituted_by_endpoint_water(coupled_fixture):
    raw = copy.deepcopy(coupled_fixture)
    report = raw["after_first_packet"]["runtime_state"]["last_report"]["weak_plane_slip"]
    report["water"] = report["endpoint_water_access"]
    with pytest.raises(ValueError, match="lagged water"):
        interval_check(raw)


def test_full_3d_accepted_memory_forgery_is_refused(coupled_fixture):
    raw = copy.deepcopy(coupled_fixture)
    raw["after_first_packet"]["runtime_state"]["mechanics"]["state"]["cohorts"][0]["strain0"][1][4] += 1e-5
    with pytest.raises(ValueError, match="accepted plastic coefficient"):
        interval_check(raw)


@pytest.mark.parametrize("field", ("plastic_work_j", "physical_mechanical_work_j", "net_constitutive_driver_work_j"))
def test_accepted_material_work_forgery_is_refused(coupled_fixture, field):
    raw = copy.deepcopy(coupled_fixture)
    raw["after_first_packet"]["runtime_state"]["last_report"]["weak_plane_slip"][field] *= 1.2
    with pytest.raises(ValueError, match="accepted integrated slip"):
        interval_check(raw)


def test_top_interval_ledger_does_not_hide_net_physical_split(coupled_fixture):
    raw = copy.deepcopy(coupled_fixture)
    report = raw["after_first_packet"]["runtime_state"]["last_report"]
    report["mechanical_work_j"] = report["net_constitutive_driver_work_j"]
    with pytest.raises(ValueError, match="top physical mechanical work"):
        interval_check(raw)


def test_cumulative_slip_loss_cannot_be_omitted(coupled_fixture):
    raw = copy.deepcopy(coupled_fixture)
    raw["after_first_packet"]["runtime_state"]["fault_friction_dissipation_j"] *= 1.2
    with pytest.raises(ValueError, match="cumulative fault_friction"):
        interval_check(raw)


def test_prescribed_synthetic_force_cannot_be_changed_only_in_report(coupled_fixture):
    raw = copy.deepcopy(coupled_fixture)
    raw["after_first_packet"]["runtime_state"]["last_report"]["audit_input"]["synthetic_additional_external_force_n"][0] += 1e12
    with pytest.raises(ValueError, match="declared fixed synthetic force"):
        interval_check(raw)


def test_source_basal_force_cannot_be_cancelled_in_report(coupled_fixture):
    raw = copy.deepcopy(coupled_fixture)
    raw["after_first_packet"]["runtime_state"]["last_report"]["audit_input"]["source_basal_external_force_n"][0] += 1e12
    with pytest.raises(ValueError, match="unchanged source basal force"):
        interval_check(raw)


def test_predictor_is_rebuilt_from_original_material_memory(coupled_fixture):
    raw = copy.deepcopy(coupled_fixture)
    driver = raw["after_first_packet"]["runtime_state"]["last_report"]["fault_evolution"]["predictor_driver"]
    driver["elastic_strain0"][0][1][5] += 1e-5
    with pytest.raises(ValueError, match="predictor endpoint coefficient"):
        interval_check(raw)


def test_predictor_stress_cannot_be_invented_to_drive_damage(coupled_fixture):
    raw = copy.deepcopy(coupled_fixture)
    driver = raw["after_first_packet"]["runtime_state"]["last_report"]["fault_evolution"]["predictor_driver"]
    driver["stress_pa"][0][2] += 1e7
    with pytest.raises(ValueError, match="predictor independently calculated stress"):
        interval_check(raw)


def test_predictor_slip_work_cannot_be_duplicated_into_accepted_total(coupled_fixture):
    raw = copy.deepcopy(coupled_fixture)
    state = raw["after_first_packet"]["runtime_state"]
    state["physical_mechanical_work_j"] += state["last_report"]["fault_evolution"]["predictor_driver"]["weak_plane_slip"]["physical_mechanical_work_j"]
    with pytest.raises(ValueError, match="cumulative physical_mechanical"):
        interval_check(raw)


def test_fixed_declared_force_is_checked_by_full_momentum(coupled_fixture):
    raw = copy.deepcopy(coupled_fixture)
    raw["synthetic_force_n"][0] += 1e16
    # Updating every reported force together cannot change the accepted DOFs.
    inputs = raw["after_first_packet"]["runtime_state"]["last_report"]["audit_input"]
    inputs["synthetic_additional_external_force_n"][0] += 1e16
    inputs["total_external_force_n"][0] += 1e16
    with pytest.raises(ValueError, match="momentum equation|global work"):
        interval_check(raw)
