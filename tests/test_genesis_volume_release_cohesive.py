"""Straight-path cohesive force is integrated, including irreversible history."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.integrate import quad

from tectonics.genesis_volume_creation_law import (
    CreationLawRefusal, ProspectiveCreationLaw)
from tectonics.genesis_volume_release_cohesive import (
    VolumeReleaseCohesiveRefusal, evaluate_volume_release_cohesive,
    prepare_volume_release_cohesive, validate_volume_release_cohesive,
    volume_release_cohesive_from_dict, volume_release_cohesive_response_to_dict,
    volume_release_cohesive_to_dict)


@pytest.fixture
def law():
    return ProspectiveCreationLaw(1e6, 4e6, 1e9, 2e9)


def chart(law, initial=(0., 0., 0.), *, area=7., trace=None, basis=None):
    return prepare_volume_release_cohesive(law, np.asarray([initial]), [area],
        np.eye(3)[None] if trace is None else trace,
        np.eye(3) if basis is None else basis)


def independent_mean(law, initial, old, new, history, events):
    """Integrate the three traction components separately using scalar QUADPACK."""
    initial = np.asarray(initial, float)
    offset = initial/law.cohesive_stiffness_pa_m
    offset[0] = initial[0]/(law.cohesive_stiffness_pa_m
        if initial[0] >= 0 else law.compression_stiffness_pa_m)
    initial_radius = np.linalg.norm(np.r_[max(offset[0], 0), offset[1:]])
    deltaf = 2*(law.fracture_energy_j_m2
        +.5*law.cohesive_stiffness_pa_m*initial_radius**2)/law.peak_traction_pa
    old, step = np.asarray(old), np.asarray(new)-old

    def value(s, component):
        gap = offset+old+s*step
        effective = np.r_[max(gap[0], 0), gap[1:]]
        maximum = max(float(history), initial_radius, np.linalg.norm(effective))
        if maximum <= law.peak_separation_m:
            damage = 0.
        elif maximum >= deltaf:
            damage = 1.
        else:
            damage = deltaf/(deltaf-law.peak_separation_m)*(1-law.peak_separation_m/maximum)
        traction = law.cohesive_stiffness_pa_m*(1-damage)*effective
        traction[0] += law.compression_stiffness_pa_m*min(gap[0], 0.)
        return float(traction[component])

    return np.array([quad(value, 0, 1, args=(i,), points=events[1:-1],
        epsabs=1e-12 if i == 2 else 1e-7, epsrel=3e-12, limit=250)[0]
        for i in range(3)])


def assert_certified(response):
    tolerance = response.relative_tolerance
    np.testing.assert_array_less(response.mean_force_component_order_error_n,
        tolerance*np.abs(response.mean_generalized_force_n)
        +response.mean_force_component_roundoff_bound_n+np.finfo(float).tiny)
    assert response.work_kernel_error_j <= (
        tolerance*abs(response.mechanical_work_j)+response.work_roundoff_bound_j)
    assert response.force_work_j == pytest.approx(response.mechanical_work_j,
        rel=tolerance, abs=response.work_roundoff_bound_j)
    assert response.mechanical_work_j == pytest.approx(
        response.stored_energy_j-response.old_stored_energy_j
        +response.fracture_dissipation_increment_j, rel=2e-14,
        abs=response.work_roundoff_bound_j)
    assert not response.physical_release_admitted
    assert not response.normal_contact_admitted
    assert not response.funding_admitted


def test_birth_retains_reference_force_without_assigning_creation_energy(law):
    value = chart(law, (1e6, .4e6, -.2e6))
    response = evaluate_volume_release_cohesive(value, np.zeros(3), np.zeros(3),
        value.seed.initial_maximum_separation_m)
    np.testing.assert_allclose(response.mean_generalized_force_n, [7e6, 2.8e6, -1.4e6], rtol=3e-15)
    assert response.old_stored_energy_j == response.stored_energy_j == 0
    assert response.mechanical_work_j == response.fracture_dissipation_increment_j == 0
    assert_certified(response)


@pytest.mark.parametrize("opening", [.002, .004, .03, .4, .6])
def test_pure_opening_exact_energy_and_average_not_endpoint_force(law, opening):
    value = chart(law)
    response = evaluate_volume_release_cohesive(value, np.zeros(3), [opening, 0, 0], [0])
    mean = independent_mean(law, [0, 0, 0], [0, 0, 0], [opening, 0, 0], 0,
                            response.point_event_fractions[0])
    np.testing.assert_allclose(response.mean_traction_local_pa[0], mean, rtol=3e-12, atol=0)
    assert response.mean_generalized_force_n[0] != response.endpoint_generalized_force_n[0]
    if opening >= law.final_separation_m:
        assert response.mechanical_work_j == 7*law.fracture_energy_j_m2
        assert response.endpoint_generalized_force_n[0] == 0
        assert response.mean_generalized_force_n[0] > 0
    assert_certified(response)


@pytest.mark.parametrize("initial,old,new,history", [
    ([1e6, 1.2e6, -.8e6], [0, 0, 0], [.06, .03, -.04], .002),
    ([1e6, 1.2e6, -.8e6], [.1, .02, 0], [.02, -.01, .03], .12),
    ([1e6, 1.2e6, -.8e6], [.02, -.01, .03], [.2, .02, -.04], .12),
    ([-8e6, 1e6, -2e6], [0, 0, 0], [.01, .03, -.05], .003),
    ([-8e6, 1e6, -2e6], [.1, .06, -.02], [-.03, -.02, .03], .13),
    ([0, 0, 0], [0, .03, 0], [0, -.03, 0], .03),
    ([1e6, 1e6, 1e6], [.03, -.02, .06], [.7, .02, -.03], .08),
])
def test_changed_direction_unloading_and_compression_match_separate_component_integrals(
        law, initial, old, new, history):
    value = chart(law, initial)
    response = evaluate_volume_release_cohesive(value, old, new, [history])
    mean = independent_mean(law, initial, old, new, history,
                            response.point_event_fractions[0])
    np.testing.assert_allclose(response.mean_traction_local_pa[0], mean,
                               rtol=2e-10, atol=1e-8)
    assert_certified(response)


def test_load_unload_and_reload_keep_the_entire_irreversible_maximum(law):
    value = chart(law, (1e6, .7e6, -.3e6))
    history = value.seed.initial_maximum_separation_m
    old = np.zeros(3)
    total_work = total_loss = 0.
    loaded_history = None
    for i, new in enumerate(([.06, .03, -.02], [.005, -.002, .003], [.04, .015, -.01], [.2, .03, -.04])):
        response = evaluate_volume_release_cohesive(value, old, new, history)
        if i == 0:
            loaded_history = response.new_maximum_separation_m.copy()
        if i in (1, 2):
            np.testing.assert_array_equal(response.new_maximum_separation_m, loaded_history)
            assert response.fracture_dissipation_increment_j == 0
        total_work += response.mechanical_work_j
        total_loss += response.fracture_dissipation_increment_j
        history, old = response.new_maximum_separation_m, np.asarray(new)
        assert_certified(response)
    assert total_work == pytest.approx(response.stored_energy_j+total_loss, rel=2e-14)


def test_compression_energy_is_reversible_and_contact_is_not_admitted(law):
    value = chart(law, (-8e6, 0, 0))
    response = evaluate_volume_release_cohesive(value, [0, 0, 0], [.001, 0, 0], [0])
    assert response.mechanical_work_j < 0
    assert response.fracture_dissipation_increment_j == 0
    assert response.mean_generalized_force_n[0] == pytest.approx(-49e6)
    assert response.mean_generalized_force_n[1] == response.mean_generalized_force_n[2] == 0
    assert_certified(response)


def test_rotation_and_multiple_reference_area_weights_preserve_force_work_duality(law):
    angle = .7
    basis = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    local_initial = np.array([[1e6, .6e6, -.2e6], [-2e6, .5e6, .1e6]])
    trace = np.array([np.eye(3), .4*np.eye(3)])
    value = prepare_volume_release_cohesive(law, local_initial@basis.T,
                                           [2., 5.], trace, basis)
    response = evaluate_volume_release_cohesive(value, np.zeros(3), [.03, .04, -.006],
                                              value.seed.initial_maximum_separation_m)
    manual_force = np.einsum("pai,pa,p->i", trace, response.mean_traction_world_pa, [2., 5.])
    np.testing.assert_allclose(response.mean_generalized_force_n, manual_force, rtol=2e-15)
    assert_certified(response)


def test_small_force_component_is_certified_without_large_component_floor(law):
    value = chart(law, (1e6, .5e6, 1e-6))
    response = evaluate_volume_release_cohesive(value, np.zeros(3), [.08, .03, -2e-14],
                                              value.seed.initial_maximum_separation_m)
    mean = independent_mean(law, [1e6, .5e6, 1e-6], np.zeros(3), [.08, .03, -2e-14],
                            value.seed.initial_maximum_separation_m[0],
                            response.point_event_fractions[0])
    assert abs(response.mean_generalized_force_n[2]) < 1e-5
    assert response.mean_traction_local_pa[0, 2] == pytest.approx(mean[2], rel=2e-10, abs=1e-20)
    assert response.mean_force_component_roundoff_bound_n[2] < 1e-16
    assert_certified(response)


def test_surface_null_mode_has_no_gap_force_or_work(law):
    trace = np.c_[np.eye(3), np.zeros(3)][None]
    value = chart(law, (1e6, .5e6, -.3e6), trace=trace)
    response = evaluate_volume_release_cohesive(value, np.zeros(4), [0, 0, 0, 1e5],
                                              value.seed.initial_maximum_separation_m)
    np.testing.assert_array_equal(response.new_gap_local_m, np.zeros((1, 3)))
    assert response.mean_generalized_force_n[3] == 0
    assert response.mechanical_work_j == response.force_work_j == 0
    assert_certified(response)


def test_event_subdivision_resolves_history_peak_failure_normal_kink_and_radial_minimum(law):
    value = chart(law)
    response = evaluate_volume_release_cohesive(value, [0, .03, 0], [0, -.7, 0], [.03])
    events = response.point_event_fractions[0]
    for expected in (.03/.73, (.03-.004)/.73, (.03+.004)/.73,
                     .06/.73, (.03+.5)/.73):
        assert min(abs(np.asarray(events)-expected)) < 1e-14
    assert_certified(response)
    compressed = chart(law, (-8e6, .4e6, -.2e6))
    crossing = evaluate_volume_release_cohesive(compressed, np.zeros(3), [.02, 0, 0],
                                              compressed.seed.initial_maximum_separation_m)
    assert min(abs(np.asarray(crossing.point_event_fractions[0])-.2)) < 1e-14


def test_packet_roundtrip_uses_trusted_inputs_and_response_is_sealed(law):
    value = chart(law, (1e6, .5e6, -.2e6))
    packet = volume_release_cohesive_to_dict(value)
    reconstructed = volume_release_cohesive_from_dict(packet, law=law,
        initial_reference_traction_world_pa=[[1e6, .5e6, -.2e6]],
        reference_area_weights_m2=[7.], compact_jump_trace_world=np.eye(3)[None],
        birth_basis_xyz=np.eye(3))
    assert reconstructed.binding_sha256 == value.binding_sha256
    packet["reference_area_weights_m2"][0] = 9
    with pytest.raises(VolumeReleaseCohesiveRefusal, match="trusted"):
        volume_release_cohesive_from_dict(packet, law=law,
            initial_reference_traction_world_pa=[[1e6, .5e6, -.2e6]],
            reference_area_weights_m2=[7.], compact_jump_trace_world=np.eye(3)[None],
            birth_basis_xyz=np.eye(3))
    response = evaluate_volume_release_cohesive(value, np.zeros(3), [.03, .02, -.001],
                                              value.seed.initial_maximum_separation_m)
    raw = volume_release_cohesive_response_to_dict(response)
    assert len(raw["cohesive_response_sha256"]) == 64
    for array in (value.birth_basis_xyz, value.seed.reference_offset_m,
                  response.mean_generalized_force_n, response.new_maximum_separation_m):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        response.mechanical_work_j = 0
    object.__setattr__(value, "binding_sha256", "0"*64)
    with pytest.raises(VolumeReleaseCohesiveRefusal, match="binding"):
        validate_volume_release_cohesive(value)


@pytest.mark.parametrize("field,value", [
    ("area", -1), ("area", True), ("basis", np.diag([1, 1, -1])),
    ("trace", np.ones((1, 2, 3))),
])
def test_invalid_chart_inputs_are_refused(law, field, value):
    with pytest.raises(CreationLawRefusal):
        chart(law, **{field: value})


@pytest.mark.parametrize("options", [
    {"low_order": True}, {"high_order": 16}, {"max_subdivisions": -1},
    {"relative_tolerance": 0}, {"relative_tolerance": True},
])
def test_invalid_quadrature_requests_are_refused(law, options):
    value = chart(law)
    with pytest.raises(CreationLawRefusal):
        evaluate_volume_release_cohesive(value, np.zeros(3), [.03, .01, 0], [0], **options)


def test_missing_irreversible_history_is_refused(law):
    value = chart(law)
    with pytest.raises(CreationLawRefusal, match="history"):
        evaluate_volume_release_cohesive(value, [.03, 0, 0], [.01, 0, 0], [0])


def test_failure_to_converge_refuses_instead_of_projecting_force_from_work(law):
    value = chart(law, (1e6, .8e6, -.4e6))
    with pytest.raises(VolumeReleaseCohesiveRefusal, match="converge"):
        evaluate_volume_release_cohesive(value, np.zeros(3), [.03, -.02, .04],
            value.seed.initial_maximum_separation_m, low_order=1, high_order=2,
            relative_tolerance=1e-13, max_subdivisions=0)


def test_nearly_failed_endpoint_dissipation_subtraction_is_reported_as_unresolved_work(law):
    value = chart(law)
    response = evaluate_volume_release_cohesive(value, [.5-1e-10, 0, 0],
                                              [.5-1e-12, 0, 0], [.5-1e-10])
    # Endpoint energy/dissipation arithmetic loses the tiny new work.  The
    # independently integrated force is retained, and the large arithmetic
    # witness is explicitly separated from traction/order certification.
    assert not response.endpoint_work_agrees_at_integral_precision
    assert not response.numerically_resolved
    assert response.exact_work_arithmetic_dominated
    assert response.work_kernel_error_j > response.work_integral_roundoff_bound_j
    assert response.work_kernel_error_j < response.work_kernel_roundoff_bound_j
    assert response.force_work_j < 1e-12
    assert response.work_integral_roundoff_bound_j < 1e-13
    assert not response.funding_admitted
