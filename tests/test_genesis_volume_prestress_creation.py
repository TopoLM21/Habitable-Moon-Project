"""Prestress-compatible force and work without virtual historical credit."""
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest
from scipy.integrate import quad

from tectonics.genesis_volume_creation_law import (
    CreationLawRefusal, ProspectiveCreationLaw, evaluate_creation_law)
from tectonics.genesis_volume_prestress_creation import (
    PrestressCreationRefusal, evaluate_prestress_creation, prepare_prestress_creation)


@pytest.fixture
def law():
    return ProspectiveCreationLaw(1e6, 4e6, 1e9, 2e9)


def independent_radius(gap):
    return np.linalg.norm([max(gap[0], 0), gap[1], gap[2]])


def independent_work(law, initial_traction, old, new, history):
    """Integrate shifted traction directly, without producer reference data."""
    traction = np.array(initial_traction, float)
    offset = traction / law.cohesive_stiffness_pa_m
    if traction[0] < 0:
        offset[0] = traction[0] / law.compression_stiffness_pa_m
    initial_radius = independent_radius(offset)
    reference_energy = .5 * law.cohesive_stiffness_pa_m * initial_radius**2
    delta0 = law.peak_traction_pa / law.cohesive_stiffness_pa_m
    deltaf = 2 * (law.fracture_energy_j_m2 + reference_energy) / law.peak_traction_pa
    old, new = np.array(old, float), np.array(new, float)
    step = new - old

    def integrand(fraction):
        internal = offset + old + fraction * step
        effective = np.array([max(internal[0], 0), internal[1], internal[2]])
        maximum = max(initial_radius, history, np.linalg.norm(effective))
        if maximum <= delta0:
            damage = 0.
        elif maximum >= deltaf:
            damage = 1.
        else:
            damage = deltaf * (maximum - delta0) / (maximum * (deltaf - delta0))
        value = law.cohesive_stiffness_pa_m * (1 - damage) * effective
        value[0] += law.compression_stiffness_pa_m * min(internal[0], 0.)
        return value @ step

    # Split at the normal positive-part kink.  Asking adaptive quadrature to
    # discover it can underestimate its integration error in an unload path.
    points = []
    if step[0] != 0:
        crossing = -(offset[0] + old[0]) / step[0]
        if 0 < crossing < 1:
            points.append(crossing)
    work, error = quad(integrand, 0, 1, points=points,
                       epsabs=1e-5, epsrel=3e-12, limit=250)
    assert error <= max(abs(work)*1e-8, 1e-4)
    return work


@pytest.mark.parametrize("initial", [
    [0, 0, 0], [1e6, 0, 0], [0, -2e6, 0], [0, 0, 2e6],
    [1e6, 1.2e6, -.8e6], [-8e6, 0, 0], [-8e6, 1e6, -2e6],
    [4e6, 0, 0], [0, 2.4e6, 3.2e6],
])
def test_birth_retains_traction_but_assigns_no_reference_energy_or_damage(law, initial):
    seed = prepare_prestress_creation(law, initial)
    response = evaluate_prestress_creation(seed, [0, 0, 0], [0, 0, 0],
                                         seed.initial_maximum_separation_m)
    np.testing.assert_allclose(response.traction_pa, initial, rtol=2e-15, atol=1e-8)
    assert response.damage == 0
    assert response.stored_energy_j_m2 == response.old_stored_energy_j_m2 == 0
    assert response.fracture_dissipation_j_m2 == response.fracture_dissipation_increment_j_m2 == 0
    assert response.mechanical_work_j_m2 == 0
    assert seed.kernel_only and response.kernel_only
    assert not seed.physical_release_admitted and not response.physical_release_admitted
    assert not seed.normal_contact_admitted and not response.normal_contact_admitted


@pytest.mark.parametrize("initial,direction", [
    ([2e6, 0, 0], [1, 0, 0]),
    ([0, -2e6, 0], [0, -1, 0]),
    ([0, 0, 3e6], [0, 0, 1]),
    ([1e6, 1.2e6, -.8e6], [.6, .48, .64]),
    ([4e6, 0, 0], [.6, .48, .64]),
])
def test_complete_cohesive_failure_costs_original_gc_once(law, initial, direction):
    seed = prepare_prestress_creation(law, initial)
    new = np.array(direction) * 1.3 * seed.final_separation_m - seed.reference_offset_m
    response = evaluate_prestress_creation(seed, [0, 0, 0], new,
                                         seed.initial_maximum_separation_m)
    assert response.damage == 1
    assert response.stored_energy_j_m2 == 0
    assert response.fracture_dissipation_j_m2 == law.fracture_energy_j_m2
    assert response.mechanical_work_j_m2 == law.fracture_energy_j_m2
    assert independent_work(law, initial, np.zeros(3), new,
        seed.initial_maximum_separation_m) == pytest.approx(law.fracture_energy_j_m2, rel=2e-10)
    np.testing.assert_array_equal(response.traction_pa, np.zeros(3))
    area = 127.
    assert area * response.fracture_dissipation_j_m2 == area * law.fracture_energy_j_m2


def test_naive_constant_reference_subtraction_has_the_wrong_fracture_cost(law):
    initial = [3e6, 0, 0]
    seed = prepare_prestress_creation(law, initial)
    final_gap = [1., 0, 0]
    old_base = evaluate_creation_law(law, seed.reference_offset_m, final_gap,
                                   seed.initial_maximum_separation_m)
    assert old_base.mechanical_work_j_m2 == (
        law.fracture_energy_j_m2 - seed.reference_cohesive_energy_j_m2)
    corrected = evaluate_prestress_creation(seed, [0, 0, 0],
        np.array(final_gap)-seed.reference_offset_m, seed.initial_maximum_separation_m)
    assert corrected.mechanical_work_j_m2 == law.fracture_energy_j_m2
    assert corrected.stored_energy_j_m2 == 0
    assert corrected.fracture_dissipation_j_m2 == law.fracture_energy_j_m2


@pytest.mark.parametrize("initial,old,new,history", [
    ([1e6, 1.2e6, -.8e6], [0, 0, 0], [.06, .03, -.04], .002),
    ([1e6, 1.2e6, -.8e6], [.1, .02, 0], [.02, -.01, .03], .12),
    ([1e6, 1.2e6, -.8e6], [.02, -.01, .03], [.2, .02, -.04], .12),
    ([-8e6, 1e6, -2e6], [0, 0, 0], [.01, .03, -.05], .003),
    ([-8e6, 1e6, -2e6], [.1, .06, -.02], [-.03, -.02, .03], .13),
    ([2e6, 1e6, -.3e6], [0, 0, 0], [-.02, 0, 0], .003),
    ([-8e6, 0, 0], [0, 0, 0], [.02, 0, 0], 0),
    ([0, 2.4e6, 3.2e6], [0, 0, 0], [.02, -.04, .06], .004),
    ([1e6, 1e6, 1e6], [.03, -.02, .06], [.7, .02, -.03], .08),
])
def test_independent_traction_integral_equals_relative_energy_plus_new_dissipation(law, initial, old, new, history):
    seed = prepare_prestress_creation(law, initial)
    response = evaluate_prestress_creation(seed, old, new, history)
    reference = independent_work(law, initial, old, new, history)
    assert response.mechanical_work_j_m2 == pytest.approx(reference, rel=4e-9, abs=3e-5)
    assert response.mechanical_work_j_m2 == pytest.approx(
        response.stored_energy_j_m2 - response.old_stored_energy_j_m2
        + response.fracture_dissipation_increment_j_m2, abs=1e-8)
    assert response.fracture_dissipation_increment_j_m2 >= 0


@pytest.mark.parametrize("initial", [[1e6, 2e6, -.3e6], [-6e6, .4e6, -1e6]])
def test_unloading_has_signed_relative_energy_without_fracture_history_credit(law, initial):
    seed = prepare_prestress_creation(law, initial)
    response = evaluate_prestress_creation(seed, [0, 0, 0], -seed.reference_offset_m,
                                         seed.initial_maximum_separation_m)
    assert response.stored_energy_j_m2 == pytest.approx(
        -seed.reference_cohesive_energy_j_m2-seed.reference_compression_energy_j_m2)
    assert response.stored_energy_j_m2 < 0
    assert response.fracture_dissipation_increment_j_m2 == 0
    back = evaluate_prestress_creation(seed, -seed.reference_offset_m, [0, 0, 0],
                                      response.maximum_separation_m)
    assert back.stored_energy_j_m2 == 0
    assert back.fracture_dissipation_j_m2 == 0
    assert back.mechanical_work_j_m2 == pytest.approx(-response.mechanical_work_j_m2)


def test_compression_reference_is_reversible_and_not_a_geometric_contact_law(law):
    seed = prepare_prestress_creation(law, [-8e6, 0, 0])
    closed = evaluate_prestress_creation(seed, [0, 0, 0], [.001, 0, 0], 0)
    assert closed.traction_pa[0] < 0
    assert closed.fracture_dissipation_j_m2 == 0
    assert closed.mechanical_work_j_m2 < 0
    assert not closed.normal_contact_admitted
    full = evaluate_prestress_creation(seed, [0, 0, 0], [1., 0, 0], 0)
    assert full.fracture_dissipation_j_m2 == law.fracture_energy_j_m2
    assert full.stored_energy_j_m2 == -seed.reference_compression_energy_j_m2
    assert full.mechanical_work_j_m2 == law.fracture_energy_j_m2 - seed.reference_compression_energy_j_m2
    # The negative reversible compression change is separate from D=Gc.
    assert full.fracture_dissipation_increment_j_m2 == law.fracture_energy_j_m2


@pytest.mark.parametrize("initial", [[0, 0, 0], [4e6, 0, 0], [1e6, 2e6, 2e6], [-8e6, 1e6, -3e6]])
def test_fracture_dissipation_is_monotone_and_has_the_correct_damage_driver(law, initial):
    seed = prepare_prestress_creation(law, initial)
    radii = np.linspace(seed.initial_maximum_separation_m, seed.final_separation_m * 1.1, 1101)
    gaps = np.stack([radii, np.zeros_like(radii), np.zeros_like(radii)], axis=-1) - seed.reference_offset_m
    response = evaluate_prestress_creation(seed, [0, 0, 0], gaps,
                                         seed.initial_maximum_separation_m)
    assert np.min(np.diff(response.fracture_dissipation_j_m2)) >= 0
    assert response.fracture_dissipation_j_m2[-1] == law.fracture_energy_j_m2
    delta0, deltaf = seed.peak_separation_m, seed.final_separation_m
    sample = (delta0 + deltaf)/2
    step = 1e-6
    lower = evaluate_prestress_creation(seed, [0, 0, 0],
        np.array([sample-step, 0, 0])-seed.reference_offset_m, seed.initial_maximum_separation_m)
    upper = evaluate_prestress_creation(seed, [0, 0, 0],
        np.array([sample+step, 0, 0])-seed.reference_offset_m, seed.initial_maximum_separation_m)
    derivative = (upper.fracture_dissipation_j_m2-lower.fracture_dissipation_j_m2)/(2*step)
    dprime = deltaf*delta0/(sample**2*(deltaf-delta0))
    driving = .5*law.cohesive_stiffness_pa_m*sample**2-seed.reference_cohesive_energy_j_m2
    assert derivative == pytest.approx(driving*dprime, rel=1e-8)
    assert driving >= 0


@pytest.mark.parametrize("initial,new,history", [
    ([1e6, 2e6, -.3e6], [.0001, .0002, -.0003], .003),
    ([1e6, 2e6, -.3e6], [.04, .08, -.03], .003),
    ([-8e6, 1e6, -2e6], [-.01, .04, -.03], .003),
    ([1e6, 2e6, -.3e6], [.01, .02, -.015], .1),
    ([-8e6, 1e6, -2e6], [1., .03, -.04], .003),
])
def test_full_tangent_matches_finite_differences(law, initial, new, history):
    seed = prepare_prestress_creation(law, initial)
    new = np.array(new)
    response = evaluate_prestress_creation(seed, [0, 0, 0], new, history)
    difference = np.zeros((3, 3))
    epsilon = 1e-7
    for i in range(3):
        delta = np.eye(3)[i] * epsilon
        plus = evaluate_prestress_creation(seed, [0, 0, 0], new+delta, history)
        minus = evaluate_prestress_creation(seed, [0, 0, 0], new-delta, history)
        difference[:, i] = (plus.traction_pa-minus.traction_pa)/(2*epsilon)
    np.testing.assert_allclose(response.tangent_pa_m, difference, rtol=3e-7, atol=.04)


def test_tangent_frame_rotations_preserve_both_shears_and_energy(law):
    angle = 1.033
    rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    initial = np.array([-6e6, 1.3e6, -2.4e6])
    old, new = np.array([.01, .001, -.002]), np.array([.04, .05, -.07])
    seed = prepare_prestress_creation(law, initial)
    rotated_seed = prepare_prestress_creation(law, rotation@initial)
    response = evaluate_prestress_creation(seed, old, new, .025)
    rotated = evaluate_prestress_creation(rotated_seed, rotation@old, rotation@new, .025)
    np.testing.assert_allclose(rotated.traction_pa, rotation@response.traction_pa, rtol=4e-14)
    np.testing.assert_allclose(rotated.tangent_pa_m, rotation@response.tangent_pa_m@rotation.T,
                               rtol=3e-13, atol=1e-6)
    assert rotated.mechanical_work_j_m2 == pytest.approx(response.mechanical_work_j_m2, rel=3e-14)
    assert rotated.fracture_dissipation_j_m2 == pytest.approx(response.fracture_dissipation_j_m2, rel=3e-14)


def test_restart_history_and_reloading_do_not_pay_fracture_twice(law):
    seed = prepare_prestress_creation(law, [1e6, 2e6, -1e6])
    first_gap, second_gap = np.array([.09, .05, -.02]), np.array([.025, .01, -.03])
    first = evaluate_prestress_creation(seed, [0, 0, 0], first_gap, seed.initial_maximum_separation_m)
    unloaded = evaluate_prestress_creation(seed, first_gap, second_gap, first.maximum_separation_m)
    resumed = evaluate_prestress_creation(seed, second_gap, first_gap, float(unloaded.maximum_separation_m))
    assert unloaded.fracture_dissipation_increment_j_m2 == resumed.fracture_dissipation_increment_j_m2 == 0
    assert unloaded.mechanical_work_j_m2 == pytest.approx(-resumed.mechanical_work_j_m2)
    final = evaluate_prestress_creation(seed, first_gap, [1, 0, 0], resumed.maximum_separation_m)
    assert first.fracture_dissipation_j_m2 + final.fracture_dissipation_increment_j_m2 == law.fracture_energy_j_m2
    assert first.mechanical_work_j_m2 + final.mechanical_work_j_m2 == pytest.approx(law.fracture_energy_j_m2)
    repeated = evaluate_prestress_creation(seed, [1, 0, 0], [0, 1, 1], final.maximum_separation_m)
    assert repeated.fracture_dissipation_increment_j_m2 == repeated.mechanical_work_j_m2 == 0


def test_zero_prestress_recovers_original_kernel(law):
    seed = prepare_prestress_creation(law, [0, 0, 0])
    old, new, history = [.03, .01, -.02], [-.02, .03, .06], .05
    original = evaluate_creation_law(law, old, new, history)
    prepared = evaluate_prestress_creation(seed, old, new, history)
    for name in ("traction_pa", "tangent_pa_m", "damage", "stored_energy_j_m2",
                 "fracture_dissipation_j_m2", "mechanical_work_j_m2"):
        np.testing.assert_allclose(getattr(prepared, name), getattr(original, name), rtol=3e-14)


def test_per_point_preparation_broadcasts_without_aliasing_or_mutable_history(law):
    initial = np.array([[1e6, 2e6, -.3e6], [-8e6, 1e6, -2e6]])
    seed = prepare_prestress_creation(law, initial)
    initial[:] = 0
    old = np.zeros((3, 1, 3))
    new = np.array([[.04, .02, -.01], [.001, .09, -.03]])
    response = evaluate_prestress_creation(seed, old, new, seed.initial_maximum_separation_m)
    assert response.traction_pa.shape == (3, 2, 3)
    assert response.tangent_pa_m.shape == (3, 2, 3, 3)
    assert response.damage.shape == (3, 2)
    assert seed.final_separation_m[0] != seed.final_separation_m[1]
    for i in range(2):
        individual_seed = prepare_prestress_creation(law, seed.initial_traction_pa[i])
        individual = evaluate_prestress_creation(individual_seed, [0, 0, 0], new[i],
                                                 individual_seed.initial_maximum_separation_m)
        np.testing.assert_array_equal(response.traction_pa[:, i],
                                      np.broadcast_to(individual.traction_pa, (3, 3)))
    for value in (seed, response):
        for member in fields(value):
            array = getattr(value, member.name)
            if isinstance(array, np.ndarray):
                with pytest.raises(ValueError):
                    array.setflags(write=True)
        with pytest.raises(FrozenInstanceError):
            value.physical_release_admitted = True


@pytest.mark.parametrize("traction", [
    [True, 0, 0], [1j, 0, 0], [np.nan, 0, 0], [np.inf, 0, 0],
    ["0", "0", "0"], [0, 0], 0., np.array([0, 0, 0], object),
    [4.1e6, 0, 0], [0, 4.1e6, 0], [3e6, 3e6, 0],
])
def test_invalid_or_overstrength_initial_traction_is_refused(law, traction):
    with pytest.raises(CreationLawRefusal):
        prepare_prestress_creation(law, traction)


@pytest.mark.parametrize("which,invalid", [
    ("old", [True, 0, 0]), ("new", [0, 1j, 0]), ("new", [0, np.nan, 0]),
    ("history", True), ("history", -.01), ("history", np.inf),
    ("new", [0, 0]), ("old", 0), ("history", "0"),
])
def test_invalid_gap_history_dtypes_or_shapes_are_refused(law, which, invalid):
    seed = prepare_prestress_creation(law, [1e6, 0, 0])
    old, new, history = [0, 0, 0], [.03, .02, .04], seed.initial_maximum_separation_m
    if which == "old":
        old = invalid
    elif which == "new":
        new = invalid
    else:
        history = invalid
    with pytest.raises(CreationLawRefusal):
        evaluate_prestress_creation(seed, old, new, history)


def test_history_cannot_erase_initial_prestress_or_old_peak(law):
    seed = prepare_prestress_creation(law, [1e6, 2e6, -1e6])
    with pytest.raises(PrestressCreationRefusal, match="initial and old"):
        evaluate_prestress_creation(seed, -seed.reference_offset_m, [0, 0, 0], 0.)
    with pytest.raises(PrestressCreationRefusal, match="initial and old"):
        evaluate_prestress_creation(seed, [.1, 0, 0], [.01, 0, 0], seed.initial_maximum_separation_m)
    rounded = np.nextafter(seed.initial_maximum_separation_m, 0.)
    response = evaluate_prestress_creation(seed, [0, 0, 0], [0, 0, 0], rounded)
    assert response.maximum_separation_m == seed.initial_maximum_separation_m
    assert response.mechanical_work_j_m2 == 0


def test_explicit_types_limits_and_no_work_credit_parameters(law):
    with pytest.raises(PrestressCreationRefusal):
        prepare_prestress_creation(object(), [0, 0, 0])
    with pytest.raises(PrestressCreationRefusal):
        evaluate_prestress_creation(object(), [0, 0, 0], [0, 0, 0], 0)
    seed = prepare_prestress_creation(law, [0, 0, 0])
    with pytest.raises(PrestressCreationRefusal, match="broadcast"):
        evaluate_prestress_creation(seed, np.zeros((2, 3)), np.zeros((3, 3)), 0)
    with pytest.raises(CreationLawRefusal, match="finite"):
        prepare_prestress_creation(law, [-1e300, 0, 0])
    for credit in ("finite_width_slip_work_j", "damage_work_j", "historical_creation_work_j"):
        with pytest.raises(TypeError):
            evaluate_prestress_creation(seed, [0, 0, 0], [0, 0, 0], 0, **{credit: 1e6})
    complete = evaluate_prestress_creation(seed, [0, 0, 0], [1e300, 1e300, 0], 0)
    assert complete.fracture_dissipation_j_m2 == law.fracture_energy_j_m2
    assert complete.stored_energy_j_m2 == 0


def test_kink_conventions_use_internal_reference_and_old_peak(law):
    seed = prepare_prestress_creation(law, [1e6, 0, 0])
    at_peak = evaluate_prestress_creation(seed, [0, 0, 0],
        [seed.peak_separation_m-seed.reference_offset_m[0], 0, 0], seed.initial_maximum_separation_m)
    np.testing.assert_allclose(at_peak.tangent_pa_m, law.cohesive_stiffness_pa_m*np.eye(3))
    at_final = evaluate_prestress_creation(seed, [0, 0, 0],
        [seed.final_separation_m-seed.reference_offset_m[0], 0, 0], seed.initial_maximum_separation_m)
    np.testing.assert_array_equal(at_final.tangent_pa_m, np.zeros((3, 3)))
    gap = [.03, .04, 0]
    history = independent_radius(seed.reference_offset_m+gap)
    old_peak = evaluate_prestress_creation(seed, gap, gap, history)
    np.testing.assert_allclose(old_peak.tangent_pa_m,
        law.cohesive_stiffness_pa_m*(1-old_peak.damage)*np.eye(3))


def test_machine_peak_frame_excess_has_recorded_snap_and_zero_initial_damage(law):
    initial = np.array([1965754.8543109482, -227903.1988754612, 3476185.838630507])
    # The two standard norm evaluations differ by an ulp for this frame.
    unsnapped_radius = np.hypot.reduce(initial/law.cohesive_stiffness_pa_m)
    assert unsnapped_radius > law.peak_separation_m
    seed = prepare_prestress_creation(law, initial)
    np.testing.assert_array_equal(seed.initial_traction_pa, initial)
    assert seed.reference_peak_snap_factor < 1
    assert seed.initial_maximum_separation_m <= law.peak_separation_m
    np.testing.assert_allclose(seed.reference_traction_pa, initial,
                               rtol=32*np.finfo(float).eps, atol=0)
    response = evaluate_prestress_creation(seed, [0, 0, 0], [0, 0, 0],
                                         seed.initial_maximum_separation_m)
    np.testing.assert_array_equal(response.traction_pa, seed.reference_traction_pa)
    assert response.damage == response.stored_energy_j_m2 == response.fracture_dissipation_j_m2 == 0
    assert response.mechanical_work_j_m2 == 0
    with pytest.raises(PrestressCreationRefusal, match="peak strength"):
        prepare_prestress_creation(law, initial*(1+128*np.finfo(float).eps))


def test_peak_strength_rotations_remain_admissible_within_machine_precision(law):
    initial = np.array([1965754.8543109482, -227903.1988754612, 3476185.838630507])
    for angle in np.linspace(-np.pi, np.pi, 97):
        rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                             [0, np.sin(angle), np.cos(angle)]])
        rotated = rotation@initial
        seed = prepare_prestress_creation(law, rotated)
        response = evaluate_prestress_creation(seed, [0, 0, 0], [0, 0, 0],
                                             seed.initial_maximum_separation_m)
        np.testing.assert_allclose(response.traction_pa, rotated, rtol=32*np.finfo(float).eps, atol=2e-8)
        assert response.damage == response.stored_energy_j_m2 == response.fracture_dissipation_j_m2 == 0
        assert response.mechanical_work_j_m2 == 0


def test_peak_snap_does_not_change_compressive_reference(law):
    initial = np.array([-8e6, law.peak_traction_pa, 0.])
    initial[1] *= 1 + 4*np.finfo(float).eps
    seed = prepare_prestress_creation(law, initial)
    assert seed.reference_peak_snap_factor < 1
    assert seed.reference_offset_m[0] == initial[0]/law.compression_stiffness_pa_m
    assert seed.reference_traction_pa[0] == initial[0]
    assert seed.reference_compression_energy_j_m2 == 16000
    assert seed.initial_maximum_separation_m <= seed.peak_separation_m


@pytest.mark.parametrize("name,value", [
    ("fracture_energy_j_m2", True), ("peak_traction_pa", np.inf),
    ("cohesive_stiffness_pa_m", -1.), ("compression_stiffness_pa_m", 0.),
])
def test_seed_revalidates_law_even_after_explicit_constructor_bypass(law, name, value):
    object.__setattr__(law, name, value)
    with pytest.raises(CreationLawRefusal):
        prepare_prestress_creation(law, [0, 0, 0])


def test_prepared_seed_does_not_alias_the_callers_law(law):
    seed = prepare_prestress_creation(law, [1e6, 2e6, -.3e6])
    assert seed.law is not law
    object.__setattr__(law, "fracture_energy_j_m2", 2e6)
    assert seed.law.fracture_energy_j_m2 == 1e6
    response = evaluate_prestress_creation(seed, [0, 0, 0], [1., 0, 0],
                                         seed.initial_maximum_separation_m)
    assert response.fracture_dissipation_j_m2 == 1e6
