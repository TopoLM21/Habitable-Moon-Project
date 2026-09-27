from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from tectonics.genesis_contact_law import ContactLawParameters, contact_return_map
from tectonics.genesis_extrinsic_contact_law import (
    extrinsic_damage, extrinsic_dissipation, extrinsic_energy, extrinsic_return_map,
)


def response(gap, jump=0., *, initial=(1e6, 3e5), old_slip=0.,
             old_cumulative=None, old_opening=0., water=0., dt=1e9,
             params=None, old_damage=None):
    vector = lambda value: np.atleast_1d(np.asarray(value, dtype=float))
    if old_cumulative is None:
        old_cumulative = np.abs(old_slip)
    return extrinsic_return_map(
        vector(gap), vector(jump), vector(old_slip), vector(old_cumulative),
        vector(old_opening), dt, vector(water), np.atleast_2d(initial),
        params or ContactLawParameters(),
        old_damage=None if old_damage is None else vector(old_damage))


@pytest.mark.parametrize("initial", [(0., 0.), (2e6, 0.), (1e6, 7e5), (0., -1e6)])
def test_birth_preserves_admissible_traction_without_history_or_energy(initial):
    r = response(0., initial=initial, water=1.)
    np.testing.assert_array_equal(r["traction_pa"], [initial])
    for key in ("damage", "plastic_slip_m", "cumulative_slip_m", "max_opening_m",
                "slip_increment_m", "friction_work_j_m2", "viscous_work_j_m2",
                "fracture_work_j_m2", "recoverable_energy_j_m2"):
        np.testing.assert_array_equal(r[key], 0.)


def test_initial_shear_overstrength_is_returned_not_redefined_as_new_strength():
    r = response(0., initial=(1e6, 4e6), water=1.)
    assert r["slip_increment_m"][0] > 0
    assert r["shear_strength_pa"][0] == 1e6
    assert 1e6 < r["traction_pa"][0, 1] < 4e6


@pytest.mark.parametrize("fraction", [0., .2, .5, .999, 1.])
def test_normal_envelope_area_and_dissipation_match_independent_quadrature(fraction):
    p = ContactLawParameters()
    t0 = fraction*p.tensile_strength_pa
    initial = np.array([[t0, 0.]])
    onset = (p.tensile_strength_pa-t0)/p.normal_stiffness_pa_m
    ascending = .5*(t0+p.tensile_strength_pa)*onset
    failure = onset+2*(p.fracture_energy_j_m2-ascending)/p.tensile_strength_pa

    def envelope(gap):
        if gap <= onset:
            return t0+p.normal_stiffness_pa_m*gap
        return max(0., p.tensile_strength_pa*(failure-gap)/(failure-onset))

    for gap in (0., onset/2, onset, onset+.01, failure/4, failure-.01, failure, failure+10):
        integrated = quad(envelope, 0., gap, points=[x for x in (onset, failure) if 0 < x < gap])[0]
        r = response(gap, initial=initial, params=p)
        assert r["traction_pa"][0, 0] == pytest.approx(envelope(gap), abs=1e-8)
        assert r["fracture_work_j_m2"][0] >= 0
        assert r["fracture_work_j_m2"][0]+r["recoverable_energy_j_m2"][0] == pytest.approx(integrated, rel=2e-13, abs=1e-8)
        assert extrinsic_dissipation([gap], initial, p)[0] == r["fracture_work_j_m2"][0]
    assert response(failure, initial=initial)["fracture_work_j_m2"][0] == p.fracture_energy_j_m2
    assert quad(envelope, 0, failure, points=[onset] if onset else None)[0] == pytest.approx(p.fracture_energy_j_m2)


@pytest.mark.parametrize("initial_normal", [0., 1e6, 2e6])
@pytest.mark.parametrize("gap,jump,history", [
    (.04, .01, 0.), (-1., .01, 0.), (-1., 2., 0.), (-1., -2., 0.),
    (3., 0., 0.), (3., 2., 0.), (3., -2., 0.), (1., .01, 3.),
    (1., 2., 3.), (25., 2., 25.), (-1., 2., 25.),
])
def test_jacobian_matches_independent_finite_difference(initial_normal, gap, jump, history):
    args = dict(initial=(initial_normal, -4e5), old_opening=history, water=.35)
    r = response(gap, jump, **args)
    h = 1e-5
    numerical = np.column_stack([
        (response(gap+h, jump, **args)["traction_pa"][0]
         -response(gap-h, jump, **args)["traction_pa"][0])/(2*h),
        (response(gap, jump+h, **args)["traction_pa"][0]
         -response(gap, jump-h, **args)["traction_pa"][0])/(2*h),
    ])
    np.testing.assert_allclose(r["tangent_pa_m"][0], numerical, rtol=1e-7, atol=.02)


@pytest.mark.parametrize("gap,jump,old_opening", [(-1., 2., 0.), (3., -2., 0.), (25., 2., 25.)])
@pytest.mark.parametrize("initial_shear", [-7e5, 8e5])
def test_shifted_shear_energy_return_identity(gap, jump, old_opening, initial_shear):
    p = ContactLawParameters()
    r = response(gap, jump, initial=(1e6, initial_shear), old_opening=old_opening,
                 old_slip=.1, params=p)
    def psi(elastic_slip):
        return initial_shear*elastic_slip+.5*p.tangential_stiffness_pa_m*elastic_slip**2
    released = psi(jump-.1)-psi(jump-r["plastic_slip_m"][0])
    accounted = sum(r[key][0] for key in (
        "friction_work_j_m2", "viscous_work_j_m2", "shear_relaxation_remainder_j_m2"))
    assert released == pytest.approx(accounted)


def test_broken_free_contact_has_signed_relative_potential_and_recloses_without_healing():
    p = ContactLawParameters()
    initial = (1e6, 8e5)
    r = response(25., 2., initial=initial)
    assert r["plastic_slip_m"][0] == pytest.approx(2.+initial[1]/p.tangential_stiffness_pa_m)
    np.testing.assert_array_equal(r["traction_pa"], 0.)
    np.testing.assert_array_equal(r["tangent_pa_m"], 0.)
    assert r["recoverable_energy_j_m2"][0] == pytest.approx(-initial[1]**2/(2*p.tangential_stiffness_pa_m))
    closed = response(-.1, 2.02, initial=initial,
                      old_slip=r["plastic_slip_m"], old_opening=r["max_opening_m"],
                      old_cumulative=r["cumulative_slip_m"], old_damage=r["damage"])
    np.testing.assert_allclose(closed["traction_pa"], [[-1e6, 2e5]])
    assert closed["damage"][0] == 1
    assert closed["fracture_work_j_m2"][0] == 0
    assert closed["shear_strength_pa"][0] == 6e5


def test_unload_to_compression_uses_initial_stress_potential_and_penalty_pressure():
    p = ContactLawParameters()
    initial = np.array([[1e6, 0.]])
    loaded = response(3., initial=initial)
    d = loaded["damage"][0]
    for gap in (1., 0., -.1, -2.):
        r = response(gap, initial=initial, old_opening=3., old_damage=loaded["damage"])
        assert r["damage"][0] == d
        assert r["fracture_work_j_m2"][0] == 0
        expected = (1-d)*1e6+p.normal_stiffness_pa_m*(gap if gap < 0 else (1-d)*gap)
        assert r["traction_pa"][0, 0] == pytest.approx(expected)
        assert r["normal_pressure_pa"][0] == p.normal_stiffness_pa_m*max(-gap, 0.)
        h = 1e-7
        plus = extrinsic_energy([gap+h], [0.], [0.], [3.], initial, p)[0]
        minus = extrinsic_energy([gap-h], [0.], [0.], [3.], initial, p)[0]
        assert (plus-minus)/(2*h) == pytest.approx(expected, rel=5e-6, abs=1.)


def test_reverse_shear_uses_shifted_reference_and_nonnegative_dissipation():
    p = replace(ContactLawParameters(), viscosity_pa_s_m=0.)
    forward = response(-1., 2., params=p, initial=(1e6, 8e5))
    reverse = response(-1., -2., params=p, initial=(1e6, 8e5),
                       old_slip=forward["plastic_slip_m"], old_cumulative=forward["cumulative_slip_m"])
    assert reverse["slip_increment_m"][0] < 0
    assert reverse["traction_pa"][0, 1] < 0
    assert reverse["cumulative_slip_m"][0] > forward["cumulative_slip_m"][0]
    assert reverse["friction_work_j_m2"][0] > 0


def test_fracture_increments_telescope_through_cycles_without_healing():
    opening, damage, fracture = 0., 0., 0.
    for gap in (.04, .2, 1., .5, -2., 1., 8., 2., 21., -3., 2., 40.):
        r = response(gap, initial=(1.5e6, 0.), old_opening=opening, old_damage=damage)
        assert r["damage"][0] >= damage
        assert r["fracture_work_j_m2"][0] >= 0
        fracture += r["fracture_work_j_m2"][0]
        opening, damage = r["max_opening_m"][0], r["damage"][0]
    assert fracture == pytest.approx(ContactLawParameters().fracture_energy_j_m2)


def test_zero_birth_matches_original_intrinsic_law_across_all_branches():
    p = ContactLawParameters()
    gap = np.array([.1, -1., -1., 3., 1., 25., -1.])
    jump = np.array([.01, .01, 2., -2., 2., 6., 2.])
    slip = np.array([0., 0., .1, -.1, 0., 1., .1])
    cumulative = np.abs(slip)
    opening = np.array([0., 0., 0., 1., 3., 21., 21.])
    water = np.linspace(0., 1., len(gap))
    original = contact_return_map(gap, jump, slip, cumulative, opening, 1e9, water, p)
    extrinsic = extrinsic_return_map(gap, jump, slip, cumulative, opening, 1e9, water,
                                     np.zeros((len(gap), 2)), p)
    for key, value in original.items():
        np.testing.assert_allclose(extrinsic[key], value, rtol=1e-12, atol=1e-8, err_msg=key)


def test_vectorization_and_inputs_are_immutable():
    values = [np.array([.04, -1., 3., 25.]), np.array([.01, 2., -2., 6.]),
              np.array([0., .1, -.1, 1.]), np.array([0., .2, .3, 3.]),
              np.array([0., 0., 1., 21.]), np.array([0., .2, .6, 1.]),
              np.array([[0., 1e5], [1e6, -8e5], [2e6, 0.], [1.5e6, 4e5]])]
    originals = [v.copy() for v in values]
    for value in values:
        value.setflags(write=False)
    r = extrinsic_return_map(*values[:5], 1e9, *values[5:], ContactLawParameters())
    for before, after in zip(originals, values):
        np.testing.assert_array_equal(before, after)
    for i in range(4):
        single = response(values[0][i], values[1][i], old_slip=values[2][i],
                          old_cumulative=values[3][i], old_opening=values[4][i],
                          water=values[5][i], initial=values[6][i])
        for key in r:
            np.testing.assert_allclose(r[key][i], single[key][0], err_msg=key)


def test_empty_and_large_finite_history():
    z = np.empty(0)
    r = extrinsic_return_map(z, z, z, z, z, 1., z, np.empty((0, 2)), ContactLawParameters())
    assert r["traction_pa"].shape == (0, 2)
    assert r["tangent_pa_m"].shape == (0, 2, 2)
    birth = np.array([[1e6, 0.]])
    assert extrinsic_damage([1e308], birth, ContactLawParameters())[0] == 1.
    assert extrinsic_dissipation([1e308], birth, ContactLawParameters())[0] == 2e7


@pytest.mark.parametrize("name,value", [
    ("gap", np.nan), ("jump", np.inf), ("old_slip", np.nan),
    ("old_cumulative", -.1), ("old_cumulative", .001), ("old_opening", -1.),
    ("water", 1.01), ("water", -.01), ("dt", True), ("dt", 0.),
    ("dt", np.inf), ("old_damage", .5), ("old_damage", -1.),
    ("initial", (-1., 0.)), ("initial", (2e6+1., 0.)),
    ("initial", (0., np.nan)), ("initial", [1e6]),
])
def test_invalid_trial_and_history_rejected(name, value):
    args = dict(gap=.04, jump=.01, old_slip=.1)
    args[name] = value
    with pytest.raises(ValueError):
        response(**args)


@pytest.mark.parametrize("helper", [extrinsic_damage, extrinsic_dissipation])
def test_helpers_reject_broadcasting_and_corrupt_history(helper):
    p = ContactLawParameters()
    for opening, initial in [(-1., [[1e6, 0.]]), ([np.nan], [[1e6, 0.]]),
                             ([0., 1.], [[1e6, 0.]]), ([[0.]], [[1e6, 0.]])]:
        with pytest.raises(ValueError):
            helper(opening, initial, p)


def test_invalid_energy_current_history_and_no_implicit_broadcasting():
    p = ContactLawParameters()
    with pytest.raises(ValueError, match="include the current gap"):
        extrinsic_energy([2.], [0.], [0.], [1.], [[1e6, 0.]], p)
    with pytest.raises(ValueError):
        response([.1, .2], .1)
    with pytest.raises(ValueError):
        response([[.1]], .1)
    with pytest.raises(ValueError):
        response(.1, params=replace(p, fracture_energy_j_m2=1.))
