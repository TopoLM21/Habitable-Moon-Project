from dataclasses import replace

import numpy as np
import pytest

from tectonics.genesis_contact_law import (
    ContactLawParameters, cohesive_damage, cohesive_dissipation,
    contact_law_parameters_from_config, contact_return_map,
)


def response(gap, jump=0., *, old_slip=0., old_cumulative=None, old_opening=0.,
             water=0., dt=1e9, params=None, old_damage=None):
    def vector(value):
        return np.atleast_1d(np.asarray(value, dtype=float))
    if old_cumulative is None:
        old_cumulative = np.abs(old_slip)
    return contact_return_map(vector(gap), vector(jump), vector(old_slip),
                              vector(old_cumulative), vector(old_opening), dt,
                              vector(water), params or ContactLawParameters(),
                              old_damage=None if old_damage is None else vector(old_damage))


def test_undamaged_elastic_opening_and_sticking():
    p = ContactLawParameters()
    r = response(.1, .01, params=p)
    np.testing.assert_allclose(r["traction_pa"], [[1e6, 1e5]])
    np.testing.assert_allclose(r["tangent_pa_m"], [np.diag([1e7, 1e7])])
    np.testing.assert_array_equal(r["slip_increment_m"], 0)
    assert r["damage"][0] == 0
    assert r["recoverable_energy_j_m2"][0] == pytest.approx(.5*1e7*(.1**2+.01**2))


def test_compression_does_not_accumulate_opening_damage():
    r = response(-2.)
    np.testing.assert_allclose(r["traction_pa"], [[-2e7, 0]])
    assert r["normal_pressure_pa"][0] == 2e7
    assert r["damage"][0] == 0
    assert r["max_opening_m"][0] == 0
    assert r["fracture_work_j_m2"][0] == 0
    assert r["recoverable_energy_j_m2"][0] == 2e7


def test_progressive_softening_and_analytical_negative_tangent():
    p = ContactLawParameters()
    onset, failure = p.damage_onset_opening_m, p.failure_opening_m
    r = response(3., params=p)
    expected = p.tensile_strength_pa*(failure-3)/(failure-onset)
    assert r["traction_pa"][0, 0] == pytest.approx(expected)
    assert 0 < r["damage"][0] < 1
    assert r["tangent_pa_m"][0, 0, 0] == pytest.approx(-p.tensile_strength_pa/(failure-onset))
    assert r["fracture_work_j_m2"][0] == pytest.approx(p.fracture_energy_j_m2*(3-onset)/(failure-onset))


def test_unloading_reloading_preserves_damage_and_fracture_budget():
    p = ContactLawParameters()
    loaded = response(3., params=p)
    unloaded = response(1., old_opening=3., old_damage=loaded["damage"], params=p)
    np.testing.assert_allclose(unloaded["damage"], loaded["damage"])
    assert unloaded["fracture_work_j_m2"][0] == 0
    assert unloaded["traction_pa"][0, 0] == pytest.approx((1-loaded["damage"][0])*p.normal_stiffness_pa_m)
    assert unloaded["tangent_pa_m"][0, 0, 0] > 0
    reloaded = response(2., old_opening=3., params=p)
    assert reloaded["fracture_work_j_m2"][0] == 0
    assert reloaded["max_opening_m"][0] == 3


def test_fully_broken_open_interface_has_no_tension_or_shear_even_with_viscosity():
    r = response(25., 6., old_slip=1., old_cumulative=3., old_opening=21.)
    np.testing.assert_array_equal(r["traction_pa"], 0)
    np.testing.assert_array_equal(r["tangent_pa_m"], 0)
    assert r["damage"][0] == 1
    assert r["plastic_slip_m"][0] == 6
    assert r["cumulative_slip_m"][0] == 8
    assert r["recoverable_energy_j_m2"][0] == 0
    assert r["friction_work_j_m2"][0] == 0
    assert r["viscous_work_j_m2"][0] == 0
    assert r["shear_relaxation_remainder_j_m2"][0] == pytest.approx(.5*1e7*5**2)


def test_reclosure_restores_compression_friction_without_cohesive_healing():
    opened = response(25., 6.)
    closed = response(-1., 6.1, old_opening=opened["max_opening_m"],
                      old_slip=opened["plastic_slip_m"], old_cumulative=opened["cumulative_slip_m"])
    assert closed["damage"][0] == 1
    assert closed["traction_pa"][0, 0] == -1e7
    assert closed["traction_pa"][0, 1] == pytest.approx(1e6)
    assert closed["shear_strength_pa"][0] == 6e6
    assert closed["fracture_work_j_m2"][0] == 0


def test_coulomb_return_with_and_without_viscosity():
    for eta in (0., 1e14):
        p = replace(ContactLawParameters(), viscosity_pa_s_m=eta)
        r = response(-1., 2., params=p)
        strength = p.cohesion_pa+p.friction_dry*p.normal_stiffness_pa_m
        expected_slip = (2*p.tangential_stiffness_pa_m-strength)/(p.tangential_stiffness_pa_m+eta/1e9)
        assert r["slip_increment_m"][0] == pytest.approx(expected_slip)
        assert r["traction_pa"][0, 1] == pytest.approx(strength+eta/1e9*expected_slip)
        assert r["friction_work_j_m2"][0] == pytest.approx(strength*expected_slip)
        assert r["viscous_work_j_m2"][0] == pytest.approx(eta/1e9*expected_slip**2)


def test_reverse_slip_increases_cumulative_and_reverses_traction():
    p = replace(ContactLawParameters(), viscosity_pa_s_m=0.)
    forward = response(-1., 2., params=p)
    reverse = response(-1., -2., old_slip=forward["plastic_slip_m"],
                       old_cumulative=forward["cumulative_slip_m"], params=p)
    assert reverse["slip_increment_m"][0] < 0
    assert reverse["traction_pa"][0, 1] < 0
    assert reverse["cumulative_slip_m"][0] > forward["cumulative_slip_m"][0]
    assert reverse["friction_work_j_m2"][0] > 0


def test_water_weakens_friction_and_cohesion_without_changing_pressure_or_mode_i():
    dry = response(-1., 2., water=0.)
    wet = response(-1., 2., water=1.)
    assert wet["traction_pa"][0, 1] < dry["traction_pa"][0, 1]
    assert wet["slip_increment_m"][0] > dry["slip_increment_m"][0]
    np.testing.assert_array_equal(wet["normal_pressure_pa"], dry["normal_pressure_pa"])
    np.testing.assert_array_equal(wet["damage"], dry["damage"])
    for value in (0., 1.):
        opening = response(3., water=value)
        assert opening["fracture_work_j_m2"][0] == response(3.)["fracture_work_j_m2"][0]


@pytest.mark.parametrize("gap,jump,history", [
    (.1, .01, 0.), (-1., .01, 0.), (-1., 2., 0.), (-1., -2., 0.),
    (3., 0., 0.), (3., 2., 0.), (3., -2., 0.), (1., .01, 3.),
    (1., 2., 3.), (25., 2., 21.), (-1., 2., 21.),
])
def test_algorithmic_tangent_matches_finite_difference(gap, jump, history):
    r = response(gap, jump, old_opening=history, water=.35)
    h = 1e-5
    numerical = np.column_stack((
        (response(gap+h, jump, old_opening=history, water=.35)["traction_pa"][0]
         -response(gap-h, jump, old_opening=history, water=.35)["traction_pa"][0])/(2*h),
        (response(gap, jump+h, old_opening=history, water=.35)["traction_pa"][0]
         -response(gap, jump-h, old_opening=history, water=.35)["traction_pa"][0])/(2*h),
    ))
    np.testing.assert_allclose(r["tangent_pa_m"][0], numerical, rtol=1e-7, atol=.02)


def test_friction_tangent_is_nonsymmetric():
    r = response(-1., 2.)
    assert r["tangent_pa_m"][0, 1, 0] < 0
    assert r["tangent_pa_m"][0, 0, 1] == 0


def test_mode_i_work_matches_fracture_energy_and_storage_at_all_openings():
    p = ContactLawParameters()
    a, b, peak = p.damage_onset_opening_m, p.failure_opening_m, p.tensile_strength_pa
    for g in (.05, a, .8, 4., 19., b, 40.):
        r = response(g, params=p)
        if g <= a:
            integrated = .5*p.normal_stiffness_pa_m*g*g
        else:
            end = min(g, b)
            end_traction = peak*(b-end)/(b-a)
            integrated = .5*peak*a+.5*(peak+end_traction)*(end-a)
        assert r["fracture_work_j_m2"][0]+r["recoverable_energy_j_m2"][0] == pytest.approx(integrated)
        assert r["fracture_work_j_m2"][0] <= p.fracture_energy_j_m2
    assert response(b)["fracture_work_j_m2"][0] == p.fracture_energy_j_m2


def test_fracture_increments_telescope_and_do_not_heal_under_cycles():
    kappa, damage, fracture = 0., 0., 0.
    for gap in (.1, 1., .5, -2., 1., 8., 2., 21., -3., 2., 40.):
        r = response(gap, old_opening=kappa, old_damage=damage)
        assert r["damage"][0] >= damage
        assert r["fracture_work_j_m2"][0] >= 0
        fracture += r["fracture_work_j_m2"][0]
        kappa, damage = r["max_opening_m"][0], r["damage"][0]
    assert fracture == pytest.approx(ContactLawParameters().fracture_energy_j_m2)


@pytest.mark.parametrize("gap,jump,old_opening", [(-1., 2., 0.), (3., -2., 0.), (25., 2., 21.)])
def test_shear_return_energy_is_accounted_including_free_opening(gap, jump, old_opening):
    p = ContactLawParameters()
    r = response(gap, jump, old_opening=old_opening, old_slip=.1, params=p)
    trial_energy = .5*p.tangential_stiffness_pa_m*(jump-.1)**2
    returned_shear_energy = .5*p.tangential_stiffness_pa_m*(jump-r["plastic_slip_m"][0])**2
    accounted = sum(r[key][0] for key in ("friction_work_j_m2", "viscous_work_j_m2", "shear_relaxation_remainder_j_m2"))
    assert trial_energy-returned_shear_energy == pytest.approx(accounted)


def test_history_is_immutable_and_vectorized_results_match_scalar_updates():
    values = [np.array([.1, -1., 3., 25.]), np.array([.01, 2., -2., 6.]),
              np.array([0., .1, -.1, 1.]), np.array([0., .2, .3, 3.]),
              np.array([0., 0., 1., 21.]), np.array([0., .2, .6, 1.])]
    originals = [value.copy() for value in values]
    r = contact_return_map(*values[:5], 1e9, values[5], ContactLawParameters())
    for before, after in zip(originals, values):
        np.testing.assert_array_equal(before, after)
    for i in range(4):
        single = response(values[0][i], values[1][i], old_slip=values[2][i],
                          old_cumulative=values[3][i], old_opening=values[4][i], water=values[5][i])
        for key in r:
            np.testing.assert_allclose(r[key][i], single[key][0])


def test_empty_interface_set():
    z = np.array([])
    r = contact_return_map(z, z, z, z, z, 1., z, ContactLawParameters())
    assert r["traction_pa"].shape == (0, 2)
    assert r["tangent_pa_m"].shape == (0, 2, 2)


@pytest.mark.parametrize("name,value", [
    ("normal_stiffness_pa_m", 0.), ("tangential_stiffness_pa_m", -1.),
    ("tensile_strength_pa", 0.), ("fracture_energy_j_m2", 1.),
    ("cohesion_pa", -1.), ("friction_dry", 3.), ("friction_wet", .8),
    ("wet_cohesion_fraction", 1.1), ("viscosity_pa_s_m", -1.),
    ("normal_stiffness_pa_m", float("inf")), ("friction_dry", True),
    ("fracture_energy_j_m2", 1e308),
])
def test_invalid_parameters_rejected(name, value):
    with pytest.raises(ValueError):
        replace(ContactLawParameters(), **{name: value}).validate()


@pytest.mark.parametrize("keyword,value", [
    ("gap", float("nan")), ("jump", float("inf")), ("dt", 0.),
    ("dt", True), ("old_opening", -1.), ("old_cumulative", -1.),
    ("old_cumulative", .01), ("water", 1.1), ("old_damage", .5),
])
def test_invalid_trial_or_history_rejected(keyword, value):
    args = dict(gap=.1, jump=.1, old_slip=.1)
    args[keyword] = value
    with pytest.raises(ValueError):
        response(**args)


def test_shapes_are_not_implicitly_broadcast():
    with pytest.raises(ValueError):
        response([.1, .2], .1)
    with pytest.raises(ValueError):
        response([[.1]], .1)


def test_config_is_strict_and_defaults_are_valid():
    p = contact_law_parameters_from_config({})
    p.validate()
    assert p.damage_onset_opening_m == .2
    assert p.failure_opening_m == 20.
    changed = contact_law_parameters_from_config({"genesis_contact_law": {"schema_version": 1, "cohesion_pa": 3e6}})
    assert changed.cohesion_pa == 3e6
    for section in ({"typo": 1.}, {"schema_version": 2}, {"schema_version": True}, {"cohesion_pa": "2e6"}):
        with pytest.raises(ValueError):
            contact_law_parameters_from_config({"genesis_contact_law": section})


@pytest.mark.parametrize("function", [cohesive_damage, cohesive_dissipation])
def test_damage_helpers_reject_corrupt_history(function):
    for value in (-1., float("nan")):
        with pytest.raises(ValueError):
            function(value, ContactLawParameters())


def test_large_finite_maximum_opening_remains_fully_damaged():
    p = ContactLawParameters()
    assert cohesive_damage(1e308, p) == 1.
    assert cohesive_dissipation(1e308, p) == p.fracture_energy_j_m2
