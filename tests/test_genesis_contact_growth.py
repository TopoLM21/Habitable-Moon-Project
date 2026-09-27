from dataclasses import fields, replace

import numpy as np
import pytest

from tectonics.genesis_contact_growth import (
    CohortState, aggregate_cohorts, append_cohorts, cohort_energy, empty_cohorts,
    evaluate_cohorts, remap_cohorts, validate_cohorts,
)
from tectonics.genesis_contact_law import ContactLawParameters, contact_return_map


LAW = ContactLawParameters()


def born(*, gap=0., jump=0., area=7., law=LAW, tolerance=.001):
    return append_cohorts(empty_cohorts(), np.array([0]), [0.], [2.], [area],
                          [gap], [jump], 1.4, law, tolerance)


def response(c, gap, jump=0., *, water=.35, dt=1e9, law=LAW):
    return evaluate_cohorts(c, [gap], [jump], dt, [water], law)


@pytest.mark.parametrize("gap,jump", [
    (0., 0.), (.1, .01), (-1., .01), (-1., 2.), (-1., -2.),
    (3., 0.), (3., 2.), (3., -2.), (25., 6.),
])
def test_first_zero_reference_cohort_reduces_to_existing_law(gap, jump):
    c = born()
    new, force, tangent = response(c, gap, jump)
    old = contact_return_map(np.array([gap]), np.array([jump]), np.zeros(1),
        np.zeros(1), np.zeros(1), 1e9, np.array([.35]), LAW)
    for name in ("plastic_slip_m", "cumulative_slip_m", "max_opening_m", "damage", "traction_pa"):
        np.testing.assert_allclose(getattr(new, name), old[name], rtol=1e-14)
    for name, source in (("friction_work_j", "friction_work_j_m2"),
                         ("viscous_work_j", "viscous_work_j_m2"),
                         ("fracture_work_j", "fracture_work_j_m2"),
                         ("shear_remainder_j", "shear_relaxation_remainder_j_m2")):
        np.testing.assert_allclose(getattr(new, name), 7*old[source], rtol=1e-14)
    np.testing.assert_allclose(force, 7*old["traction_pa"], rtol=1e-14)
    np.testing.assert_allclose(tangent, 7*old["tangent_pa_m"], rtol=1e-14)
    assert cohort_energy(new, [gap], [jump], LAW) == pytest.approx(7*old["recoverable_energy_j_m2"][0])
    validate_cohorts(new, LAW, 1)


def test_open_birth_has_no_bridge_no_fictitious_fracture_and_no_healing():
    c = born(gap=3000., jump=4500.)
    assert not c.bonded[0]
    assert c.damage[0] == 1
    assert c.max_opening_m[0] == 0
    assert c.fracture_work_j[0] == 0
    assert cohort_energy(c, [3000.], [4500.], LAW) == 0
    for gap, jump in ((3000., 4500.), (4000., 4503.), (.001, 4504.)):
        c, force, tangent = response(c, gap, jump)
        np.testing.assert_array_equal(force, 0)
        np.testing.assert_array_equal(tangent, 0)
        assert c.fracture_work_j[0] == 0
        assert cohort_energy(c, [gap], [jump], LAW) == 0
    closed, force, _ = response(c, -1., 4506.)
    assert force[0, 0] == -7e7
    assert force[0, 1] > 0
    assert closed.friction_work_j[0] > 0
    assert closed.fracture_work_j[0] == 0
    assert not closed.bonded[0]
    reopened, force, _ = response(closed, .05, 4507.)
    np.testing.assert_array_equal(force, 0)
    assert reopened.damage[0] == 1
    assert reopened.fracture_work_j[0] == 0


def test_compression_uses_actual_gap_and_birth_stores_penalty_energy():
    c = born(gap=-.25, jump=123.)
    assert c.birth_gap_m[0] == 0
    assert c.bonded[0]
    np.testing.assert_array_equal(c.traction_pa, [[-.25e7, 0.]])
    assert cohort_energy(c, [-.25], [123.], LAW) == .5*7e7*.25**2
    new, force, _ = response(c, -.1, 123.)
    np.testing.assert_array_equal(force, [[-7e6, 0.]])
    assert new.max_opening_m[0] == 0
    assert new.fracture_work_j[0] == 0


def test_tiny_positive_birth_gap_offsets_tension_but_not_compression():
    c = born(gap=.0005, jump=2000.)
    assert c.bonded[0]
    assert cohort_energy(c, [.0005], [2000.], LAW) == 0
    _, force, tangent = response(c, .0002, 2000.)
    assert force[0, 0] == 0
    assert tangent[0, 0, 0] == 0
    _, force, _ = response(c, -.1, 2000.)
    assert force[0, 0] == -7e6
    new, force, _ = response(c, .1005, 2000.01)
    assert force[0, 0] == pytest.approx(7e6)
    assert force[0, 1] == pytest.approx(7e5)
    assert new.max_opening_m[0] == pytest.approx(.1)
    assert new.cumulative_slip_m[0] == 0


@pytest.mark.parametrize("birth_gap,gap,jump", [
    (.0005, .1005, .01), (.0005, .0002, .01), (.0005, -.1, .01),
    (.0005, -.1, 2.), (.0005, -.1, -2.), (.0005, 3., 2.),
    (.0005, 3., -2.), (.0005, 25., 5.),
    (30., 40., 5.), (30., -1., .01), (30., -1., 2.), (30., -1., -2.),
])
def test_cohort_algorithmic_tangent_matches_finite_difference(birth_gap, gap, jump):
    c = born(gap=birth_gap, jump=17.)
    jump += 17.
    _, _, tangent = response(c, gap, jump)
    h = 1e-6
    numerical = np.column_stack((
        (response(c, gap+h, jump)[1][0]-response(c, gap-h, jump)[1][0])/(2*h),
        (response(c, gap, jump+h)[1][0]-response(c, gap, jump-h)[1][0])/(2*h),
    ))
    np.testing.assert_allclose(tangent[0], numerical, rtol=3e-7, atol=.2)


def test_growth_preserves_old_history_joules_and_adds_stress_free_shear_reference():
    old, _, _ = response(born(), -1., 2.)
    saved = {f.name: getattr(old, f.name).copy() for f in fields(old)}
    grown = append_cohorts(old, np.array([0]), [2.], [5.], [7.], [-1.], [2.], 1.5, LAW, .001)
    for name, value in saved.items():
        np.testing.assert_array_equal(getattr(old, name), value)
        np.testing.assert_array_equal(getattr(grown, name)[:1], value)
    assert grown.area_ref_m2[1] == 10.5
    assert grown.plastic_slip_m[1] == 0
    assert grown.cumulative_slip_m[1] == 0
    assert grown.birth_jump_m[1] == 2
    assert grown.friction_work_j[1] == 0
    old_energy = cohort_energy(old, [-1.], [2.], LAW)
    new_energy = cohort_energy(grown, [-1.], [2.], LAW)
    assert new_energy-old_energy == pytest.approx(.5*10.5e7)
    summary = aggregate_cohorts(grown, 1)
    assert summary["friction_work_cell_j"][0] == old.friction_work_j[0]
    assert summary["viscous_work_cell_j"][0] == old.viscous_work_j[0]
    assert summary["fracture_work_cell_j"][0] == old.fracture_work_j[0]


def test_cohorts_keep_independent_damage_and_force_is_integrated_not_averaged():
    first, _, _ = response(born(), 3., 1.)
    grown = append_cohorts(first, np.array([0]), [2.], [3.], [7.], [3.], [1.], 1.5, LAW, .001)
    assert grown.damage[0] < 1
    assert grown.damage[1] == 1
    next_state, force, _ = response(grown, 4., 1.5)
    np.testing.assert_allclose(force[0], np.sum(next_state.area_ref_m2[:, None]*next_state.traction_pa, axis=0))
    assert next_state.fracture_work_j[1] == 0
    summary = aggregate_cohorts(next_state, 2)
    np.testing.assert_allclose(summary["traction_pa"][0], force[0]/10.5)
    np.testing.assert_array_equal(summary["traction_pa"][1], 0)
    assert summary["fracture_work_cell_j"].sum() == next_state.fracture_work_j.sum()


def test_complete_rupture_consumes_reference_area_times_fracture_energy_once():
    c = born(gap=.0005)
    for gap in (.1, .3, 1., 4., 30., 2., -1., 40.):
        c, _, _ = response(c, gap)
        validate_cohorts(c, LAW)
    assert c.fracture_work_j.sum() == pytest.approx(7*LAW.fracture_energy_j_m2)
    assert c.damage[0] == 1


def test_calls_leave_inputs_immutable_and_repeatable_for_newton_rollback():
    c, _, _ = response(born(), -1., 2.)
    saved = {f.name: getattr(c, f.name).copy() for f in fields(c)}
    one, force1, tangent1 = response(c, -2., -3.)
    two, force2, tangent2 = response(c, -2., -3.)
    for f in fields(c):
        np.testing.assert_array_equal(getattr(c, f.name), saved[f.name])
        np.testing.assert_array_equal(getattr(one, f.name), getattr(two, f.name))
    np.testing.assert_array_equal(force1, force2)
    np.testing.assert_array_equal(tangent1, tangent2)


def test_remapping_changes_only_trace_identity():
    c, _, _ = response(born(), 3., 1.)
    new = remap_cohorts(c, np.array([2, 0]))
    assert new.trace_index[0] == 2
    for f in fields(c):
        if f.name != "trace_index":
            np.testing.assert_array_equal(getattr(new, f.name), getattr(c, f.name))
    _, force, _ = evaluate_cohorts(new, np.array([0., 0., 3.]), np.array([0., 0., 1.]),
                                  1e9, np.full(3, .35), LAW)
    assert force[2, 0] > 0
    np.testing.assert_array_equal(force[:2], 0)
    with pytest.raises(ValueError, match="injective"):
        remap_cohorts(c, np.array([1, 1]))


def test_empty_state_and_empty_geometry_have_well_defined_shapes():
    c = empty_cohorts()
    new, force, tangent = evaluate_cohorts(c, [], [], 1., [], LAW)
    validate_cohorts(new, LAW, 0)
    assert force.shape == (0, 2)
    assert tangent.shape == (0, 2, 2)
    assert cohort_energy(new, [], [], LAW) == 0
    assert aggregate_cohorts(new, 0)["traction_pa"].shape == (0, 2)
    _, force, tangent = evaluate_cohorts(c, [0., 0.], [0., 0.], 1., [0., 0.], LAW)
    assert force.shape == (2, 2)
    np.testing.assert_array_equal(force, 0)
    np.testing.assert_array_equal(tangent, 0)


@pytest.mark.parametrize("field,value", [
    ("trace_index", np.array([0.])), ("trace_index", np.array([-1])),
    ("bonded", np.array([1])), ("area_ref_m2", np.array([0.])),
    ("z_hi_ref_m", np.array([0.])), ("birth_time_myr", np.array([-1.])),
    ("birth_gap_m", np.array([-1.])), ("birth_jump_m", np.array([np.inf])),
    ("damage", np.array([.5])), ("fracture_work_j", np.array([1.])),
    ("plastic_slip_m", np.array([1.])), ("traction_pa", np.array([0.])),
])
def test_corrupt_cohort_state_is_rejected(field, value):
    with pytest.raises(ValueError):
        validate_cohorts(replace(born(), **{field: value}), LAW, 1)


def test_unbonded_history_cannot_claim_damage_evolution_or_fracture_spending():
    c = born(gap=30.)
    for field, value in (("damage", .5), ("max_opening_m", 30.), ("fracture_work_j", 1.)):
        with pytest.raises(ValueError, match="Unbonded"):
            validate_cohorts(replace(c, **{field: np.array([value])}), LAW)


@pytest.mark.parametrize("dt,water", [(0., .5), (-1., .5), (np.inf, .5), (1., -1.), (1., 2.)])
def test_invalid_time_or_water_is_rejected(dt, water):
    with pytest.raises(ValueError):
        response(born(), 0., dt=dt, water=water)


def test_bonding_tolerance_cannot_bridge_the_damage_process_scale():
    with pytest.raises(ValueError, match="bonding tolerance"):
        born(tolerance=LAW.damage_onset_opening_m)
