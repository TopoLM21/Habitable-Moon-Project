"""Explicit prospective seeds never replace inherited Genesis activation/history."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.genesis_volume_checkpoint import _canonical
from tectonics.genesis_volume_creation_law import ProspectiveCreationLaw
from tectonics.genesis_volume_nucleation import (
    ProspectiveVolumeNucleationLaw, VolumeNucleationRefusal,
    VolumeNucleationTieRefusal, VolumeNucleationEigenRefusal,
    VolumeNucleationSupportRefusal, evaluate_volume_nucleation_criterion,
    propose_from_cohort_corner_fields, bind_volume_nucleation_proposal,
    volume_nucleation_proposal_to_dict, volume_nucleation_proposal_from_dict)
from tectonics.genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict
from tectonics.genesis_volume_slip_dynamics import VolumeSlipModel


def law(*, tensile=10., cohesion=1000., friction=.6, radius=.3):
    return ProspectiveVolumeNucleationLaw(tensile, cohesion, friction, radius,
        ProspectiveCreationLaw(1e8, 1e8, 1e10, 2e10))


def corner_fixture():
    cohort = SimpleNamespace(id=7, face_index=3, layer_index=2,
        q_lo=.1, live_q_hi=.4, outer_scale=.9, inner_scale=.75,
        owner_id="original-owner", material_id="original-material")
    X = np.array(((0., 0., 4.), (1., 1., 4.), (-1., 1., 4.)))
    positions = np.stack((.9*X, .75*X))
    stresses = np.zeros((2, 3, 3, 3))
    stresses[0, 0] = np.diag((100., 2., -3.))
    return cohort, stresses, positions


def propose(stress=None, position=None, source=(.8, 0., .6), params=None):
    c, s, x = corner_fixture()
    return propose_from_cohort_corner_fields(c, s if stress is None else stress,
        x if position is None else position, (0., 0., 1.), source,
        law() if params is None else params)


def test_new_rankine_and_classical_unclamped_coulomb_scores_have_independent_formula():
    p = law(tensile=12., cohesion=4., friction=.6)
    stress = np.diag((20., 2., -10.))
    result = evaluate_volume_nucleation_criterion(stress, p)
    np.testing.assert_array_equal(result.principal_stress_pa, (-10., 2., 20.))
    h = np.sqrt(1+.6**2)
    assert float(result.tensile_score) == pytest.approx(20/12-1)
    assert float(result.shear_score) == pytest.approx(((h+.6)*20-(h-.6)*(-10))/(2*4)-1)
    assert float(result.yield_score) == max(float(result.tensile_score), float(result.shear_score))


def test_classical_tensile_extrapolation_is_declared_and_does_not_mimic_old_friction_clamp():
    p = law(tensile=100., cohesion=4., friction=.6)
    response = evaluate_volume_nucleation_criterion(np.eye(3)*10, p)
    assert float(response.shear_score) == pytest.approx(.6*10/4-1)
    assert float(response.tensile_score) == pytest.approx(-.9)


@pytest.mark.parametrize("seed", [7, 23, 41])
def test_convex_corner_certificate_bounds_random_affine_angular_inverse_radius_fields(seed):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(2, 3, 3, 3))*20
    corners = .5*(raw+raw.swapaxes(-1, -2))
    p = law(tensile=12., cohesion=4.)
    maximum = float(evaluate_volume_nucleation_criterion(corners, p).yield_score.max())
    N = rng.dirichlet(np.ones(3), 300)
    inverse_outer, inverse_inner = 1/.93, 1/.79
    inverse_radius = rng.uniform(inverse_outer, inverse_inner, len(N))
    eta = (inverse_radius-inverse_outer)/(inverse_inner-inverse_outer)
    # Independent tensor hull, not the producer's interpolation helper.
    sigma = np.einsum("ni,niab->nab", N, np.stack([
        (1-e)*corners[0]+e*corners[1] for e in eta]))
    value = evaluate_volume_nucleation_criterion(sigma, p).yield_score
    assert np.all(value <= maximum+3e-13*max(1., abs(maximum)))


def test_tensile_unique_maximum_defines_literal_material_anchor_and_normal():
    c, stress, xyz = corner_fixture()
    raw = propose()
    candidate = raw["candidate"]
    assert candidate["cohort_id"] == c.id and candidate["owner_id"] == c.owner_id
    assert candidate["angular_barycentric"] == [1., 0., 0.]
    assert candidate["mass_coordinate_q"] == c.q_lo
    assert candidate["radial_scale"] == c.outer_scale
    np.testing.assert_array_equal(candidate["current_nucleation_point_m"], xyz[0, 0])
    np.testing.assert_array_equal(candidate["normal_world"], (1., 0., 0.))
    assert candidate["dominant_mode"] == "tensile"
    assert candidate["plane_offset_m"] == 0.
    assert candidate["dip_from_current_top_plane_rad"] == pytest.approx(np.pi/2)
    assert candidate["dip_reference"] == "current_lambda_one_column_top_triangle_X_plus_A"
    assert raw["parameter_provenance"] == "explicit_prospective_constitutive_inputs_not_recovered_history"


def test_full_current_corner_placement_and_literal_inner_q_are_preserved():
    c, stress, xyz = corner_fixture()
    stress[1, 0] = stress[0, 0]*2
    # Shift the entire explicit computational chart, preserving its support.
    xyz += np.array((.2, -.3, .4))
    candidate = propose(stress, xyz)["candidate"]
    assert candidate["selected_corner_index"] == 3
    assert candidate["mass_coordinate_q"] == c.live_q_hi
    assert candidate["radial_scale"] == c.inner_scale
    assert candidate["plane_offset_m"] == pytest.approx(.2)
    np.testing.assert_array_equal(candidate["current_nucleation_point_m"], xyz[1, 0])


def test_shear_conjugate_is_selected_by_declared_inherited_orientation_preference():
    _, stress, _ = corner_fixture()
    stress[0, 0] = np.diag((12., 0., -8.))
    p = law(tensile=100., cohesion=1.)
    candidate = propose(stress, params=p)["candidate"]
    theta = np.pi/4-.5*np.arctan(.6)
    n = np.array((np.cos(theta), 0., np.sin(theta)))
    np.testing.assert_allclose(candidate["normal_world"], n, atol=4e-15)
    s = stress[0, 0]
    traction = s@n
    pressure = n@traction
    tau = np.linalg.norm(traction-pressure*n)
    assert (tau+.6*pressure)/p.shear_cohesion_pa-1 == pytest.approx(candidate["shear_score"])
    assert candidate["dominant_mode"] == "shear"
    assert candidate["shear_conjugate_preference"][0] != candidate["shear_conjugate_preference"][1]


def test_seed_is_finite_definition_without_area_cost_or_funding_claim():
    raw = propose()
    candidate = raw["candidate"]
    assert candidate["seed_radius_m"] == .3
    assert candidate["finite_seed_domain"] == "selected_single_original_live_cohort_only"
    assert candidate["finite_seed_definition"] == (
        "current_world_plane_intersection_sphere_intersection_selected_original_cohort")
    assert candidate["finite_seed_reference_area_m2"] is None
    assert candidate["full_creation_cost_j"] is None
    assert candidate["funding_source"] is None
    assert candidate["physical_surface_selected"] is False
    assert candidate["gc_area_convention"] == "both_banks_once_per_reference_material_area"
    assert candidate["prospective_gc_j_m2"] == 1e8
    assert all(raw[k] is False for k in ("positive_finite_seed_area_certified",
        "physical_creation_admitted", "physical_release_admitted", "funding_admitted"))
    assert raw["creation_work_debited_from_Genesis_j"] == 0.


def test_below_threshold_field_requires_no_arbitrary_orientation_even_when_every_corner_ties():
    raw = propose(np.zeros((2, 3, 3, 3)))
    assert raw["status"] == "below_prospective_convex_threshold"
    assert raw["candidate"] is None
    assert raw["maximum_yield_score"] == -1.


def test_tied_distinct_material_corners_are_refused_instead_of_centroid_selection():
    _, stress, _ = corner_fixture()
    stress[0, 1] = stress[0, 0]
    with pytest.raises(VolumeNucleationTieRefusal, match="distinct material corners"):
        propose(stress)


def test_numerically_unresolved_corner_preference_is_not_a_physical_threshold_tolerance():
    _, stress, _ = corner_fixture()
    stress[0, 1] = stress[0, 0]*(1-2*np.finfo(float).eps)
    with pytest.raises(VolumeNucleationTieRefusal):
        propose(stress)


def test_exact_threshold_can_define_proposal_while_literal_negative_score_does_not():
    _, stress, _ = corner_fixture()
    stress[0, 0] = np.diag((10., 0., -1.))
    assert propose(stress)["maximum_yield_score"] == 0.
    stress[0, 0, 0, 0] = np.nextafter(10., 0.)
    assert propose(stress)["candidate"] is None


def test_equal_tensile_and_shear_modes_are_refused():
    _, stress, _ = corner_fixture()
    stress[0, 0] = np.diag((20., 0., -20.))
    with pytest.raises(VolumeNucleationTieRefusal, match="tensile/shear"):
        propose(stress, params=law(tensile=10., cohesion=10., friction=0.))


@pytest.mark.parametrize("stress,params,reason", [
    (np.diag((20., 20., 0.)), law(), "dominant principal"),
    (np.diag((20., -20., -20.)), law(tensile=1000., cohesion=1.), "minimum principal"),
])
def test_degenerate_eigenplanes_are_refused(stress, params, reason):
    _, corners, _ = corner_fixture()
    corners[0, 0] = stress
    with pytest.raises(VolumeNucleationEigenRefusal, match=reason):
        propose(corners, params=params)


def test_equal_shear_conjugate_preference_is_refused_without_axis_bias():
    _, stress, _ = corner_fixture()
    stress[0, 0] = np.diag((12., 0., -8.))
    with pytest.raises(VolumeNucleationTieRefusal, match="conjugates"):
        propose(stress, source=(1., 0., 0.), params=law(tensile=100., cohesion=1.))


def test_supporting_plane_is_refused_even_with_unique_supercritical_tensor():
    _, _, xyz = corner_fixture()
    xyz[:, 2, 0] = abs(xyz[:, 2, 0])
    with pytest.raises(VolumeNucleationSupportRefusal, match="boundary-only"):
        propose(position=xyz)


def test_rotation_changes_world_normal_but_not_spectral_failure_score_or_dip():
    c, stress, xyz = corner_fixture()
    angle = .7
    Q = np.array(((np.cos(angle), -np.sin(angle), 0.),
                  (np.sin(angle), np.cos(angle), 0.), (0., 0., 1.)))
    transformed = Q@stress@Q.T
    raw = propose_from_cohort_corner_fields(c, transformed, xyz@Q.T,
        Q@np.array((0., 0., 1.)), Q@np.array((.8, 0., .6)), law())
    original = propose()
    assert raw["maximum_yield_score"] == pytest.approx(original["maximum_yield_score"])
    np.testing.assert_allclose(raw["candidate"]["normal_world"], Q@np.array((1., 0., 0.)), atol=3e-15)
    assert raw["candidate"]["dip_from_current_top_plane_rad"] == pytest.approx(np.pi/2)


@pytest.mark.parametrize("name,value", [
    ("tensile_strength_pa", 0.), ("shear_cohesion_pa", -1.),
    ("friction_coefficient", -1.), ("seed_radius_m", 0.),
    ("tensile_strength_pa", True), ("seed_radius_m", float("inf")),
])
def test_constitutive_inputs_require_explicit_valid_numbers(name, value):
    parameters = dict(tensile_strength_pa=10., shear_cohesion_pa=4.,
        friction_coefficient=.6, seed_radius_m=.3,
        creation_law=ProspectiveCreationLaw(1e8, 1e8, 1e10, 2e10))
    parameters[name] = value
    with pytest.raises(VolumeNucleationRefusal):
        ProspectiveVolumeNucleationLaw(**parameters)


def test_no_prospective_law_or_creation_energy_is_silently_defaulted():
    with pytest.raises(TypeError):
        ProspectiveVolumeNucleationLaw()
    with pytest.raises(VolumeNucleationRefusal, match="explicit prospective creation"):
        ProspectiveVolumeNucleationLaw(10., 4., .6, .3, None)


@pytest.mark.parametrize("stress", [
    np.ones((3, 2)), np.array(((1., 2., 0.), (0., 3., 0.), (0., 0., 1.))),
    np.eye(3)*float("nan"), [[True, 0., 0.], [0., 1., 0.], [0., 0., 1.]],
])
def test_invalid_tensor_representation_is_refused(stress):
    with pytest.raises(VolumeNucleationRefusal):
        evaluate_volume_nucleation_criterion(stress, law())


def test_response_and_law_are_immutable_owned_values():
    p = law()
    raw = np.diag((20., 0., -5.))
    response = evaluate_volume_nucleation_criterion(raw, p)
    raw[:] = 0
    np.testing.assert_array_equal(response.principal_stress_pa, (-5., 0., 20.))
    with pytest.raises(ValueError):
        response.yield_score.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        p.seed_radius_m = 12


@pytest.fixture(scope="module")
def accepted():
    path = Path(__file__).resolve().parents[1]/(
        "analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json")
    original = path.read_bytes()
    model = VolumeSlipModel(json.loads(original))
    state = model.initial_state()
    checkpoint = _canonical(volume_slip_checkpoint_to_dict(model, state))
    unconfigured = bind_volume_nucleation_proposal(model, state)
    p = law(tensile=12e6, cohesion=4e6, radius=100.)
    configured = bind_volume_nucleation_proposal(model, state, p)
    yield model, state, p, unconfigured, configured
    assert path.read_bytes() == original
    assert _canonical(volume_slip_checkpoint_to_dict(model, state)) == checkpoint


def test_authentic_inactive_source_keeps_gate_and_noevent_for_both_parameter_declarations(accepted):
    model, state, p, unconfigured, configured = accepted
    for proposal in (unconfigured, configured):
        raw = volume_nucleation_proposal_to_dict(proposal)
        assert raw["status"] == "no_calculated_activation_event"
        assert raw["newly_activating_faces"] == raw["existing_active_faces"] == []
        assert raw["corner_catalogue"] == raw["computational_candidate_proposals"] == []
        assert raw["candidate"] is raw["selected_sharp_material_surface"] is None
        assert raw["source_onset"]["new_activation_event"] is None
        assert raw["parent_checkpoint_sha256"] == volume_slip_checkpoint_to_dict(model, state)["checkpoint_sha256"]
        assert raw["inherited_runtime_ledgers"] == raw["source_onset"]["inherited_runtime_ledgers"]
        assert raw["creation_work_debited_from_Genesis_j"] == 0.
        assert raw["historical_dissipation_available_for_creation_j"] == 0.
        assert all(raw[k] is False for k in ("clock_advanced", "activation_accepted",
            "physical_creation_admitted", "physical_release_admitted", "funding_admitted",
            "positive_finite_seed_area_certified", "history_reinitialized"))
    assert volume_nucleation_proposal_to_dict(unconfigured)["declared_prospective_law"] is None
    assert volume_nucleation_proposal_to_dict(configured)["parameter_provenance"] == (
        "explicit_prospective_constitutive_inputs_not_recovered_history")


def test_noevent_does_not_even_evaluate_new_strength_to_bypass_inherited_damage_gate(accepted, monkeypatch):
    model, state, p, _, _ = accepted
    def forbidden(*args, **kwargs):
        raise AssertionError("No-event source must not select a constitutive seed")
    monkeypatch.setattr("tectonics.genesis_volume_nucleation.evaluate_volume_nucleation_criterion", forbidden)
    assert volume_nucleation_proposal_to_dict(bind_volume_nucleation_proposal(model, state, p))["candidate"] is None


def test_full_source_roundtrip_and_outer_immutable_binding(accepted):
    model, state, p, _, configured = accepted
    raw = volume_nucleation_proposal_to_dict(configured)
    assert volume_nucleation_proposal_from_dict(raw, model=model, state=state, law=p) == configured
    with pytest.raises(FrozenInstanceError):
        configured.payload_json = "{}"
    with pytest.raises(VolumeNucleationRefusal, match="immutable"):
        volume_nucleation_proposal_to_dict(replace(configured, payload_json="{}"))


@pytest.mark.parametrize("key,value", [
    ("time_myr", 0.), ("parent_checkpoint_sha256", "0"*64),
    ("newly_activating_faces", [0]), ("candidate", {"cohort_id": 1}),
    ("selected_sharp_material_surface", {"mesh_edge": 1}),
    ("source_recovered_creation_energy_law", {"Gc": 1e8}),
    ("physical_creation_admitted", True), ("funding_admitted", True),
    ("creation_work_debited_from_Genesis_j", 1.),
    ("historical_dissipation_available_for_creation_j", 1.),
])
def test_resealed_source_history_activation_and_funding_substitutions_are_refused(accepted, key, value):
    model, state, p, _, configured = accepted
    raw = volume_nucleation_proposal_to_dict(configured)
    raw[key] = value
    raw["nucleation_sha256"] = hashlib.sha256(_canonical({
        k: v for k, v in raw.items() if k != "nucleation_sha256"})).hexdigest()
    with pytest.raises(VolumeNucleationRefusal, match="complete accepted-source"):
        volume_nucleation_proposal_from_dict(raw, model=model, state=state, law=p)


def test_resealed_constitutive_change_requires_explicit_new_trusted_law(accepted):
    model, state, p, _, configured = accepted
    raw = volume_nucleation_proposal_to_dict(configured)
    raw["declared_prospective_law"]["seed_radius_m"] *= 2
    raw["nucleation_sha256"] = hashlib.sha256(_canonical({
        k: v for k, v in raw.items() if k != "nucleation_sha256"})).hexdigest()
    with pytest.raises(VolumeNucleationRefusal, match="complete accepted-source"):
        volume_nucleation_proposal_from_dict(raw, model=model, state=state, law=p)


@pytest.mark.parametrize("model,state", [(None, None), (object(), object())])
def test_trusted_binding_requires_actual_runtime_types(model, state):
    with pytest.raises(VolumeNucleationRefusal, match="accepted Genesis"):
        bind_volume_nucleation_proposal(model, state)
