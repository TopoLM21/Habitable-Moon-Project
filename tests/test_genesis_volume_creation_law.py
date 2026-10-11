"""Physical identities for the explicit prospective Γ-area creation kernel."""
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest
from scipy.integrate import quad

from tectonics.genesis_volume_creation_law import (
    CreationLawRefusal, CreationLawSofteningRefusal, ProspectiveCreationLaw,
    evaluate_creation_law)


@pytest.fixture
def law():
    return ProspectiveCreationLaw(
        fracture_energy_j_m2=1e6, peak_traction_pa=4e6,
        cohesive_stiffness_pa_m=1e9, compression_stiffness_pa_m=2e9)


def radius(gap):
    return np.linalg.norm([max(gap[0], 0.), gap[1], gap[2]])


def independently_integrated_work(law, old, new, old_maximum):
    """Direct traction integration; no response energy or damage is reused."""
    old, new = np.array(old, float), np.array(new, float)
    increment = new - old
    peak = law.peak_traction_pa / law.cohesive_stiffness_pa_m
    failure = 2 * law.fracture_energy_j_m2 / law.peak_traction_pa

    def power(fraction):
        gap = old + fraction * increment
        z = np.array([max(gap[0], 0.), gap[1], gap[2]])
        history = max(old_maximum, np.linalg.norm(z))
        if history <= peak:
            damage = 0.
        elif history >= failure:
            damage = 1.
        else:
            damage = failure * (history - peak) / (history * (failure - peak))
        traction = law.cohesive_stiffness_pa_m * (1 - damage) * z
        traction[0] += law.compression_stiffness_pa_m * min(gap[0], 0.)
        return traction @ increment

    value, error = quad(power, 0., 1., epsabs=1e-5, epsrel=3e-12, limit=250)
    assert error <= max(abs(value) * 1e-8, 1e-4)
    return value


@pytest.mark.parametrize("direction", np.eye(3).tolist() + [[.6, .48, .64]])
def test_complete_separation_all_three_modes_counts_both_banks_once(law, direction):
    direction = np.array(direction)
    response = evaluate_creation_law(law, np.zeros(3), 1.5 * law.final_separation_m * direction, 0.)
    assert response.damage == 1.
    assert response.stored_energy_j_m2 == 0.
    assert response.fracture_dissipation_j_m2 == law.fracture_energy_j_m2
    assert response.mechanical_work_j_m2 == law.fracture_energy_j_m2
    np.testing.assert_array_equal(response.traction_pa, np.zeros(3))
    reference = independently_integrated_work(law, np.zeros(3),
        1.5 * law.final_separation_m * direction, 0.)
    assert reference == pytest.approx(law.fracture_energy_j_m2, rel=2e-10)
    # Γ is one material surface; no extra factor of two is attached to its banks.
    area = 37.
    assert area * response.mechanical_work_j_m2 == area * law.fracture_energy_j_m2


@pytest.mark.parametrize("component", range(3))
def test_peak_traction_and_initial_stiffness(law, component):
    gap = np.zeros(3)
    gap[component] = law.peak_separation_m
    response = evaluate_creation_law(law, np.zeros(3), gap, 0.)
    assert response.damage == 0.
    assert response.traction_pa[component] == law.peak_traction_pa
    np.testing.assert_allclose(response.tangent_pa_m, law.cohesive_stiffness_pa_m * np.eye(3))
    assert response.stored_energy_j_m2 == .5 * law.peak_traction_pa * law.peak_separation_m
    assert response.fracture_dissipation_j_m2 == 0.


def test_two_shear_components_are_rotationally_covariant(law):
    theta = 1.127
    rotation = np.array([[1., 0, 0], [0, np.cos(theta), -np.sin(theta)],
                         [0, np.sin(theta), np.cos(theta)]])
    old, new = np.array([.001, .0007, -.0012]), np.array([.04, .023, -.034])
    first = evaluate_creation_law(law, old, new, radius(old))
    rotated = evaluate_creation_law(law, rotation @ old, rotation @ new, radius(old))
    np.testing.assert_allclose(rotated.traction_pa, rotation @ first.traction_pa, rtol=2e-14)
    np.testing.assert_allclose(rotated.tangent_pa_m,
        rotation @ first.tangent_pa_m @ rotation.T, rtol=2e-13, atol=1e-6)
    for name in ("maximum_separation_m", "damage", "stored_energy_j_m2",
                 "fracture_dissipation_j_m2", "mechanical_work_j_m2"):
        assert getattr(rotated, name) == pytest.approx(getattr(first, name), rel=3e-14)


@pytest.mark.parametrize("old,new,history", [
    ([0, 0, 0], [.033, .024, .057], 0),
    ([.003, -.001, .0003], [.0038, -.0015, .004], .0035),
    ([.1, .08, 0], [.03, -.02, .01], .13),
    ([.01, 0, 0], [.16, .02, -.03], .08),
    ([-.04, .01, -.015], [.02, -.03, .018], .04),
    ([.08, .06, 0], [-.05, -.04, .05], .1),
    ([-.01, .01, 0], [-.08, .025, .031], .02),
    ([.06, .08, 0], [.06, 0, .08], .1),
    ([0, 0, 0], [-.015, 0, 0], 0),
    ([.03, -.07, .02], [.7, -.09, -.03], .08),
])
def test_endpoint_energy_identity_equals_independent_straight_path_work(law, old, new, history):
    response = evaluate_creation_law(law, old, new, history)
    reference = independently_integrated_work(law, old, new, history)
    assert response.mechanical_work_j_m2 == pytest.approx(reference, rel=3e-9, abs=2e-5)
    assert response.mechanical_work_j_m2 == pytest.approx(
        response.stored_energy_j_m2 - response.old_stored_energy_j_m2
        + response.fracture_dissipation_increment_j_m2, abs=1e-8)
    assert response.fracture_dissipation_increment_j_m2 >= 0


@pytest.mark.parametrize("gap,history", [
    ([.001, .0004, -.0005], 0.),
    ([.06, .08, -.02], 0.),
    ([-.015, .06, -.05], 0.),
    ([.018, .024, -.005], .09),
    ([-.02, .02, -.03], .09),
    ([.7, .01, .02], 0.),
])
def test_consistent_tangent_matches_three_component_finite_differences(law, gap, history):
    gap = np.array(gap)
    old_gap = np.zeros(3)
    response = evaluate_creation_law(law, old_gap, gap, history)
    perturbation = 1e-7
    derivative = np.zeros((3, 3))
    for i in range(3):
        step = np.eye(3)[i] * perturbation
        plus = evaluate_creation_law(law, old_gap, gap + step, history)
        minus = evaluate_creation_law(law, old_gap, gap - step, history)
        derivative[:, i] = (plus.traction_pa - minus.traction_pa) / (2 * perturbation)
    np.testing.assert_allclose(response.tangent_pa_m, derivative, rtol=2e-7, atol=.04)
    np.testing.assert_allclose(response.tangent_pa_m, response.tangent_pa_m.T, rtol=2e-14, atol=1e-7)


def test_unloading_reloading_and_restart_never_pay_creation_twice(law):
    first_gap = np.array([.12, .09, 0])
    first = evaluate_creation_law(law, np.zeros(3), first_gap, 0.)
    unloaded_gap = np.array([.03, .02, 0])
    unloaded = evaluate_creation_law(law, first_gap, unloaded_gap, first.maximum_separation_m)
    assert unloaded.maximum_separation_m == first.maximum_separation_m
    assert unloaded.damage == first.damage
    assert unloaded.fracture_dissipation_increment_j_m2 == 0
    assert unloaded.mechanical_work_j_m2 < 0
    reloaded = evaluate_creation_law(law, unloaded_gap, first_gap,
                                    float(unloaded.maximum_separation_m))
    assert reloaded.fracture_dissipation_increment_j_m2 == 0
    assert reloaded.mechanical_work_j_m2 == pytest.approx(-unloaded.mechanical_work_j_m2)
    np.testing.assert_allclose(reloaded.tangent_pa_m,
        law.cohesive_stiffness_pa_m * (1 - first.damage) * np.eye(3))
    complete = evaluate_creation_law(law, first_gap, [.7, 0, 0], reloaded.maximum_separation_m)
    assert first.mechanical_work_j_m2 + complete.mechanical_work_j_m2 == pytest.approx(law.fracture_energy_j_m2)
    assert (first.fracture_dissipation_j_m2
        + complete.fracture_dissipation_increment_j_m2) == law.fracture_energy_j_m2
    repeated = evaluate_creation_law(law, [.7, 0, 0], [0, .8, .9], complete.maximum_separation_m)
    assert repeated.fracture_dissipation_increment_j_m2 == 0
    assert repeated.mechanical_work_j_m2 == 0


def test_compression_does_not_create_fracture_energy_or_mask_either_shear(law):
    only_compression = evaluate_creation_law(law, [0, 0, 0], [-.08, 0, 0], 0)
    assert only_compression.maximum_separation_m == 0
    assert only_compression.fracture_dissipation_j_m2 == 0
    assert only_compression.traction_pa[0] == -.08 * law.compression_stiffness_pa_m
    opened = evaluate_creation_law(law, [-.08, 0, 0], [0, 0, 0], 0)
    assert only_compression.mechanical_work_j_m2 + opened.mechanical_work_j_m2 == 0
    shear = evaluate_creation_law(law, [0, 0, 0], [-.08, .03, .04], 0)
    reference = evaluate_creation_law(law, [0, 0, 0], [0, .03, .04], 0)
    assert shear.damage == reference.damage
    assert shear.fracture_dissipation_j_m2 == reference.fracture_dissipation_j_m2
    np.testing.assert_array_equal(shear.traction_pa[1:], reference.traction_pa[1:])


def test_broadcasting_and_immutable_arrays_are_independent_of_callers(law):
    old = np.zeros((2, 1, 3))
    new = np.array([[[.02, .03, 0], [.04, 0, .08], [-.02, .1, .05]]])
    histories = np.array([[0.], [.2]])
    response = evaluate_creation_law(law, old, new, histories)
    assert response.traction_pa.shape == (2, 3, 3)
    assert response.tangent_pa_m.shape == (2, 3, 3, 3)
    assert response.damage.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            scalar = evaluate_creation_law(law, old[i, 0], new[0, j], histories[i, 0])
            np.testing.assert_array_equal(response.traction_pa[i, j], scalar.traction_pa)
            assert response.mechanical_work_j_m2[i, j] == scalar.mechanical_work_j_m2
    saved = response.traction_pa.copy()
    old[:] = 123
    new[:] = 456
    histories[:] = 789
    np.testing.assert_array_equal(response.traction_pa, saved)
    for name in (field.name for field in fields(response) if field.name.endswith(("_pa", "_pa_m", "_m", "_m2"))):
        with pytest.raises(ValueError):
            getattr(response, name).setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        response.kernel_only = False
    with pytest.raises(FrozenInstanceError):
        law.fracture_energy_j_m2 = 3e6


def test_zero_gap_kernel_has_no_prestress_and_never_admits_physical_release(law):
    response = evaluate_creation_law(law, [0, 0, 0], [0, 0, 0], 0.)
    assert response.mechanical_work_j_m2 == response.fracture_dissipation_j_m2 == 0
    np.testing.assert_array_equal(response.traction_pa, np.zeros(3))
    assert response.kernel_only and law.kernel_only
    assert not response.physical_release_admitted
    assert not law.physical_release_admitted
    assert set(field.name for field in fields(law)) == {
        "fracture_energy_j_m2", "peak_traction_pa",
        "cohesive_stiffness_pa_m", "compression_stiffness_pa_m"}
    with pytest.raises(TypeError):
        ProspectiveCreationLaw()
    for credit in ("friction_work_j", "viscous_work_j", "fault_damage_work_j"):
        with pytest.raises(TypeError):
            evaluate_creation_law(law, [0, 0, 0], [0, 0, 0], 0, **{credit: 1e6})


@pytest.mark.parametrize("field_name", [field.name for field in fields(ProspectiveCreationLaw)])
@pytest.mark.parametrize("invalid", [True, 0., -1., np.nan, np.inf, 1+0j, "1", [1.], None])
def test_parameter_dtype_shape_and_positivity_are_strict(law, field_name, invalid):
    with pytest.raises(CreationLawRefusal):
        replace(law, **{field_name: invalid})


@pytest.mark.parametrize("toughness", [7_999., 8_000.])
def test_insufficient_toughness_has_typed_refusal(law, toughness):
    with pytest.raises(CreationLawSofteningRefusal):
        replace(law, fracture_energy_j_m2=toughness)


@pytest.mark.parametrize("which,invalid", [
    ("old", [True, 0., 0.]), ("new", [0., 1j, 0.]),
    ("new", np.array([0., 0., 0.], dtype=object)), ("history", True),
    ("old", ["0", "0", "0"]), ("new", [np.nan, 0, 0]),
    ("history", np.inf), ("old", [0., 0.]), ("new", 0.),
    ("history", -.001),
])
def test_gap_and_history_strict_validation(law, which, invalid):
    old, new, history = [0., 0., 0.], [.01, .02, .03], 0.
    if which == "old":
        old = invalid
    elif which == "new":
        new = invalid
    else:
        history = invalid
    with pytest.raises(CreationLawRefusal):
        evaluate_creation_law(law, old, new, history)


def test_history_coverage_and_controlled_roundoff(law):
    old = np.array([.003, .004, 0.])
    length = radius(old)
    with pytest.raises(CreationLawRefusal, match="old history"):
        evaluate_creation_law(law, old, old, length * .99)
    response = evaluate_creation_law(law, old, old, np.nextafter(length, 0.))
    assert response.maximum_separation_m == length
    assert response.mechanical_work_j_m2 == 0
    assert response.fracture_dissipation_increment_j_m2 == 0
    with pytest.raises(CreationLawRefusal, match="broadcast"):
        evaluate_creation_law(law, np.zeros((2, 3)), np.zeros((3, 3)), 0)
    with pytest.raises(CreationLawRefusal, match="explicit"):
        evaluate_creation_law(object(), old, old, length)


def test_already_separated_large_finite_gap_does_not_overflow_unneeded_terms(law):
    response = evaluate_creation_law(law, [0, 0, 0], [1e300, 1e300, 0], 0)
    assert response.mechanical_work_j_m2 == law.fracture_energy_j_m2
    np.testing.assert_array_equal(response.traction_pa, np.zeros(3))
    np.testing.assert_array_equal(response.tangent_pa_m, np.zeros((3, 3)))
    with pytest.raises(CreationLawRefusal, match="finite"):
        evaluate_creation_law(law, [0, 0, 0], [-1e300, 0, 0], 0)


def test_declared_kink_tangents(law):
    zero = evaluate_creation_law(law, [0, 0, 0], [0, 0, 0], 0)
    assert zero.tangent_pa_m[0, 0] == law.cohesive_stiffness_pa_m
    final = evaluate_creation_law(law, [0, 0, 0], [law.final_separation_m, 0, 0], 0)
    np.testing.assert_array_equal(final.tangent_pa_m, np.zeros((3, 3)))
    repeated = evaluate_creation_law(law, [.03, .04, 0], [.03, .04, 0], .05)
    np.testing.assert_allclose(repeated.tangent_pa_m,
        law.cohesive_stiffness_pa_m * (1 - repeated.damage) * np.eye(3))
