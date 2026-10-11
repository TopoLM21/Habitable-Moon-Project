"""Declared material controls of the producer-free volume-slip oracle."""
import ast
import copy
from pathlib import Path
import json

import numpy as np
import pytest

from analysis.genesis_volume_slip_validation import independent_audit as audit
from analysis.genesis_volume_slip_validation.material_control import declared_material_fixture
from tectonics.genesis_volume_slip_dynamics import VolumeSlipModel
from tectonics.genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict




@pytest.fixture(scope="module")
def material_fixture():
    return declared_material_fixture()


@pytest.fixture(scope="module")
def actual_initial_packet():
    path = Path(__file__).resolve().parents[1]/"analysis/genesis_volume_fault_validation/evidence/control/initial.json"
    source = json.loads(path.read_text())["source_packet"]
    model = VolumeSlipModel(source)
    return volume_slip_checkpoint_to_dict(model, model.initial_state())


def test_actual_new_branch_starts_with_literal_formation_memory(actual_initial_packet):
    binding = audit.actual_state_binding(actual_initial_packet)
    assert binding["source_active_plane_count"] == 0
    assert binding["original_birth_prefix_preserved"]
    assert binding["runtime_3d_inactive_planes_and_slip_loss_remain_exact_zero"]


@pytest.mark.parametrize("field,value", (("plane_normal_world", .1),
    ("slip_direction_world", .1), ("fault_friction_dissipation_j", 1.),
    ("fault_viscous_dissipation_j", 1.), ("physical_mechanical_work_j", 1.)))
def test_actual_source_cannot_invent_planes_or_slip_work(actual_initial_packet, field, value):
    packet = copy.deepcopy(actual_initial_packet)
    if field.endswith("world"):
        packet["runtime_state"][field][0][0] = value
    else:
        packet["runtime_state"][field] = value
    with pytest.raises(ValueError, match="fabricated|total versus net"):
        audit.actual_state_binding(packet)


def test_actual_state_keeps_inherited_signed_slip_history(actual_initial_packet):
    packet = copy.deepcopy(actual_initial_packet)
    packet["runtime_state"]["fault_history"]["signed_shear"][0] = 1e-8
    with pytest.raises(ValueError, match="reset or forged"):
        audit.actual_state_binding(packet)


def test_oracle_imports_no_producer_physics():
    for module in (audit, audit.volume, audit.faults):
        for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(name == "tectonics" or name.startswith("tectonics.") for name in names)


@pytest.mark.parametrize("sign", (1., -1.))
def test_full_3d_material_slip_matches_independent_work_and_memory(sign):
    fixture = declared_material_fixture(sign=sign)
    report = audit.audit_material_fixture(fixture)
    assert report["yielding_face_count"] == 1
    assert report["plastic_work_j"] > 0
    assert report["friction_work_j"] > 0
    assert report["viscous_fault_work_j"] > 0
    assert report["maximum_relative_cohort_balance_residual"] < 3e-12
    flow = np.array(fixture["producer_result"]["plastic_increment0"])[0, 0]
    assert np.count_nonzero(np.abs(flow[3:]) > 1e-10) == 3
    assert np.sign(fixture["producer_result"]["shear_increment"][0]) == sign


@pytest.mark.parametrize("active,pure_compression", ((False, False), (True, True)))
def test_inactive_and_compression_controls_have_no_invented_slip(active, pure_compression):
    fixture = declared_material_fixture(active=active, pure_compression=pure_compression)
    report = audit.audit_material_fixture(fixture)
    assert report["yielding_face_count"] == 0
    assert report["plastic_work_j"] == 0.


@pytest.mark.parametrize("field", ("shear_increment", "plastic_work_j", "friction_work_j",
    "viscous_fault_work_j", "physical_mechanical_work_j", "mechanical_work_j", "normal_stress_pa"))
def test_slip_work_traction_and_sign_forgery_refused(material_fixture, field):
    fixture = copy.deepcopy(material_fixture)
    value = fixture["producer_result"][field]
    if isinstance(value, list):
        value[0] *= -1. if field == "shear_increment" else 1.2
    else:
        fixture["producer_result"][field] *= 1.2
    with pytest.raises(ValueError, match="material slip"):
        audit.audit_material_fixture(fixture)


@pytest.mark.parametrize("field", ("strain0", "strain1", "average_strain0", "plastic_increment0"))
def test_3d_cohort_coefficient_forgery_refused(material_fixture, field):
    fixture = copy.deepcopy(material_fixture)
    fixture["producer_result"][field][1][2][4] += 1e-6
    with pytest.raises(ValueError, match="material slip cohort"):
        audit.audit_material_fixture(fixture)


def test_full_18_coordinate_force_forgery_refused(material_fixture):
    fixture = copy.deepcopy(material_fixture)
    fixture["producer_result"]["average_force_cohort_n"][1][17] += 1e6
    with pytest.raises(ValueError, match="18-coordinate force"):
        audit.audit_material_fixture(fixture)


def test_synthetic_fixture_cannot_claim_actual_planet_source(material_fixture):
    fixture = copy.deepcopy(material_fixture)
    fixture["scope"] = "actual_genesis_source"
    with pytest.raises(ValueError, match="synthetic material scope"):
        audit.audit_material_fixture(fixture)


def test_reported_material_work_nan_is_refused(material_fixture):
    fixture = copy.deepcopy(material_fixture)
    fixture["producer_result"]["balance_residual_j"] = float('nan')
    with pytest.raises(ValueError):
        audit.audit_material_fixture(fixture)


def test_nonorthogonal_3d_plane_is_refused(material_fixture):
    fixture = copy.deepcopy(material_fixture)
    fixture["slip_direction_world"] = copy.deepcopy(fixture["plane_normal_world"])
    with pytest.raises(ValueError, match="orthonormal"):
        audit.audit_material_fixture(fixture)


def test_fixed_start_3d_plane_is_invariant_under_world_rotation(material_fixture):
    fixture = copy.deepcopy(material_fixture)
    axis = np.array((1., -2., 3.)); axis /= np.linalg.norm(axis)
    a = .63
    K = np.array(((0., -axis[2], axis[1]), (axis[2], 0., -axis[0]), (-axis[1], axis[0], 0.)))
    R = np.eye(3)+np.sin(a)*K+(1-np.cos(a))*(K@K)
    for key in ("plane_normal_world", "slip_direction_world"):
        fixture[key] = (np.array(fixture[key])@R.T).tolist()
    for row in fixture["cohorts"]:
        row["triangle_m"] = (np.array(row["triangle_m"])@R.T).tolist()
        row["basis_xyz"] = (R@np.array(row["basis_xyz"])).tolist()
        row["displacement_increment18_m"] = (np.array(row["displacement_increment18_m"]).reshape(-1, 3)@R.T).ravel().tolist()
    forces = np.array(fixture["producer_result"]["average_force_cohort_n"])
    fixture["producer_result"]["average_force_cohort_n"] = (forces.reshape(-1, 3)@R.T).reshape(forces.shape).tolist()
    report = audit.audit_material_fixture(fixture)
    assert report["yielding_face_count"] == 1


def test_full_radial_memory_has_energy_beyond_face_mean(material_fixture):
    fixture = copy.deepcopy(material_fixture)
    means = []
    for row in fixture["cohorts"]:
        cohort = row["geometry_cohort"]
        q, w = audit.volume.radial_rule(cohort["q_lo"], cohort["q_hi"])
        lam, _ = audit.volume.conical_scales(np.array(row["triangle_m"]),
            row["column_mass_kg"], row["density_kg_m3"], q)
        mean = float(w@(1/lam)); means.append(mean)
        e1 = np.tile((0., 0., 0., 0., 8e-4, -5e-4), (3, 1))
        row["old0"] = (-mean*e1).tolist()
        row["old1"] = e1.tolist()
        row["thermal0"] = np.zeros((3, 6)).tolist()
        row["displacement_increment18_m"] = np.zeros(18).tolist()
    varied = audit.independent_material_response(fixture)
    assert varied["elastic_energy_j"] > 0
    assert varied["plastic_work_j"] == 0.
    for mean, response in zip(means, varied["cohorts"]):
        assert np.max(np.abs(response["strain1"])) > 0
        np.testing.assert_allclose(response["strain0"]+mean*response["strain1"], 0., atol=3e-19)
        assert response["elastic_energy_j"] > 0.
