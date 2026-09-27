from dataclasses import fields, replace

import numpy as np
import pytest

from tectonics.genesis_contact_growth import (
    CohortState, append_cohorts, cohort_energy, empty_cohorts,
    evaluate_cohorts, validate_cohorts,
)
from tectonics.genesis_contact_law import ContactLawParameters
from tectonics.genesis_extrinsic_contact_law import extrinsic_energy, extrinsic_return_map
from tectonics.genesis_moving_contact_cohorts import MovingContactCohorts, MovingContactHistory


LAW = ContactLawParameters()


def owner(*, n=2, law=LAW, peak=False, shear=0.):
    depth = np.arange(n, dtype=float)+2.
    length = np.arange(n, dtype=float)+7.
    c = append_cohorts(empty_cohorts(), np.arange(n), np.zeros(n), depth,
        length, np.zeros(n), np.zeros(n), 1.4, law, .001)
    traction = np.column_stack((np.full(n, law.tensile_strength_pa*(1 if peak else .5)),
                                np.full(n, shear)))
    c.traction_pa = traction.copy()
    return MovingContactCohorts(c, traction, length, law)


def evaluate(model, history, gap, jump=0., dt=1e9, water=.35):
    return model.evaluate(history, np.broadcast_to(gap, (model.ntraces,)),
        np.broadcast_to(jump, (model.ntraces,)), dt,
        np.broadcast_to(water, (model.ntraces,)))


def snapshot(history):
    return {part: {f.name: getattr(getattr(history, part), f.name).copy()
                   for f in fields(CohortState)} for part in ("initial", "added")}


def assert_snapshot(history, saved):
    for part, arrays in saved.items():
        for name, value in arrays.items():
            np.testing.assert_array_equal(getattr(getattr(history, part), name), value)


@pytest.mark.parametrize("gap,jump", [
    (0., 0.), (.05, .01), (.7, .03), (3., 2.), (-1., -2.), (30., 6.),
])
def test_original_response_exactly_reuses_extrinsic_law(gap, jump):
    m = owner(shear=2e5)
    old = m.initial()
    new, force, tangent = evaluate(m, old, gap, jump)
    n = m.ntraces
    expected = extrinsic_return_map(np.full(n, gap), np.full(n, jump),
        np.zeros(n), np.zeros(n), np.zeros(n), 1e9, np.full(n, .35),
        m.birth_traction_pa, LAW)
    for name in ("plastic_slip_m", "cumulative_slip_m", "max_opening_m", "damage", "traction_pa"):
        np.testing.assert_array_equal(getattr(new.initial, name), expected[name])
    area = old.initial.area_ref_m2
    np.testing.assert_array_equal(force, area[:, None]*expected["traction_pa"])
    np.testing.assert_array_equal(tangent, area[:, None, None]*expected["tangent_pa_m"])
    assert m.energy(new, np.full(n, gap), np.full(n, jump)) == pytest.approx(
        area@expected["recoverable_energy_j_m2"], rel=1e-14)


def test_extrinsic_damage_is_never_validated_as_ordinary_cohesive_damage():
    m = owner(peak=True)
    history, _, _ = evaluate(m, m.initial(), .05)
    assert np.all(history.initial.damage > 0)
    with pytest.raises(ValueError, match="damage disagrees"):
        validate_cohorts(history.initial, LAW, m.ntraces)
    m.validate(history, [.05, .05], [0., 0.])
    prepared, de = m.prepare(history, m.front_depth(history)+1., [.05, .05], [0., 0.], 1.5)
    assert de == 0
    assert np.all(~prepared.added.bonded)


def test_growth_preserves_initial_history_and_every_old_material_area():
    m = owner()
    old, _, _ = evaluate(m, m.initial(), -1., 2.)
    saved = snapshot(old)
    old_front = m.front_depth(old)
    grown, change = m.prepare(old, old_front+[3., 7.], [-1., -1.], [2., 2.], 1.5)
    assert_snapshot(old, saved)
    for name, value in saved["initial"].items():
        np.testing.assert_array_equal(getattr(grown.initial, name), value)
    np.testing.assert_array_equal(grown.added.z_lo_ref_m, old_front)
    np.testing.assert_array_equal(grown.added.z_hi_ref_m, old_front+[3., 7.])
    added_area = m.birth_edge_length_m_per_trace*np.array([3., 7.])/2
    np.testing.assert_array_equal(grown.added.area_ref_m2, added_area)
    assert change == pytest.approx(.5*LAW.normal_stiffness_pa_m*added_area.sum())
    assert m.energy(grown, [-1., -1.], [2., 2.])-m.energy(old, [-1., -1.], [2., 2.]) == pytest.approx(change)
    assert np.all(grown.added.birth_jump_m == 2.)
    assert np.all(grown.added.plastic_slip_m == 0.)
    assert np.all(grown.added.fracture_work_j == 0.)
    assert np.all(grown.added.friction_work_j == 0.)


def test_partial_trace_growth_keeps_contiguous_material_coverage_and_clocks():
    m = owner(n=3)
    h = m.initial()
    first, _ = m.prepare(h, [3., 3., 6.], [0.]*3, [0.]*3, 1.5)
    second, _ = m.prepare(first, [5., 8., 6.], [0.]*3, [0.]*3, 1.6)
    np.testing.assert_array_equal(second.added.trace_index, [0, 2, 0, 1])
    np.testing.assert_array_equal(m.front_depth(second), [5., 8., 6.])
    for trace in range(m.ntraces):
        area = second.initial.area_ref_m2[trace]+second.added.area_ref_m2[second.added.trace_index == trace].sum()
        assert area == m.birth_edge_length_m_per_trace[trace]*m.front_depth(second)[trace]/2
    np.testing.assert_array_equal(second.added.birth_time_myr, [1.5, 1.5, 1.6, 1.6])


def test_new_solid_at_closed_contact_has_no_inherited_extrinsic_prestress():
    m = owner(peak=True, shear=5e5)
    h = m.initial()
    grown, change = m.prepare(h, m.front_depth(h)+4., [0., 0.], [0., 0.], 1.5)
    assert change == 0
    np.testing.assert_array_equal(grown.added.traction_pa, 0)
    _, force, _ = evaluate(m, grown, 0.)
    np.testing.assert_array_equal(force, h.initial.area_ref_m2[:, None]*m.birth_traction_pa)


def test_default_bonding_tolerance_retains_existing_coupled_contact_scale():
    m = owner()
    h, _, _ = evaluate(m, m.initial(), 1e-6)
    grown, _ = m.prepare(h, [3., 4.], [1e-6, 1e-6], [0., 0.], 1.5)
    assert m.bonding_gap_tolerance_m == 1e-9
    assert np.all(~grown.added.bonded)
    corrupt = replace(grown, added=replace(grown.added,
        bonded=np.ones(2, dtype=bool), damage=np.zeros(2)))
    with pytest.raises(ValueError, match="birth-gap decision"):
        m.validate(corrupt)


def test_open_gap_birth_never_bridges_heals_or_claims_fracture_work():
    m = owner()
    opened, _, _ = evaluate(m, m.initial(), 30., 5.)
    grown, change = m.prepare(opened, m.front_depth(opened)+2., [30., 30.], [5., 5.], 1.5)
    assert change == 0
    assert np.all(~grown.added.bonded)
    initial_birth_traction = m.birth_traction_pa.copy()
    for gap, jump in ((40., 7.), (-1., 8.), (.05, 9.), (30., 12.)):
        grown, force, _ = evaluate(m, grown, gap, jump)
        assert np.all(~grown.added.bonded)
        assert np.all(grown.added.damage == 1)
        assert np.all(grown.added.max_opening_m == 0)
        assert np.all(grown.added.fracture_work_j == 0)
        if gap >= 0:
            np.testing.assert_array_equal(force, 0)
            np.testing.assert_array_equal(grown.added.traction_pa, 0)
    np.testing.assert_array_equal(m.birth_traction_pa, initial_birth_traction)


@pytest.mark.parametrize("gap,jump", [(.03, .01), (.7, .03), (3., 2.), (-1., -2.), (30., 6.)])
def test_hybrid_integrated_response_is_sum_of_independent_material_sections(gap, jump):
    m = owner(shear=2e5)
    h, _ = m.prepare(m.initial(), [5., 7.], [0., 0.], [0., 0.], 1.5)
    new, force, tangent = evaluate(m, h, gap, jump)
    extrinsic = extrinsic_return_map(np.full(2, gap), np.full(2, jump),
        h.initial.plastic_slip_m, h.initial.cumulative_slip_m, h.initial.max_opening_m,
        1e9, np.full(2, .35), m.birth_traction_pa, LAW)
    ordinary, ordinary_force, ordinary_tangent = evaluate_cohorts(h.added,
        np.full(2, gap), np.full(2, jump), 1e9, np.full(2, .35), LAW)
    np.testing.assert_allclose(force, ordinary_force+h.initial.area_ref_m2[:, None]*extrinsic["traction_pa"], rtol=0, atol=0)
    np.testing.assert_allclose(tangent, ordinary_tangent+h.initial.area_ref_m2[:, None, None]*extrinsic["tangent_pa_m"], rtol=0, atol=0)
    expected_energy = h.initial.area_ref_m2@extrinsic_energy(np.full(2, gap), np.full(2, jump),
        new.initial.plastic_slip_m, new.initial.max_opening_m, m.birth_traction_pa, LAW)
    expected_energy += cohort_energy(ordinary, np.full(2, gap), np.full(2, jump), LAW)
    assert m.energy(new, np.full(2, gap), np.full(2, jump)) == expected_energy


@pytest.mark.parametrize("gap,jump", [(.03, .01), (.7, .03), (3., 2.), (-1., -2.), (30., 6.)])
def test_combined_algorithmic_tangent_matches_independent_finite_difference(gap, jump):
    m = owner(n=1, shear=2e5)
    h, _ = m.prepare(m.initial(), [5.], [0.], [0.], 1.5)
    _, _, tangent = evaluate(m, h, gap, jump)
    eps = 1e-6
    numerical = np.column_stack((
        (evaluate(m, h, gap+eps, jump)[1][0]-evaluate(m, h, gap-eps, jump)[1][0])/(2*eps),
        (evaluate(m, h, gap, jump+eps)[1][0]-evaluate(m, h, gap, jump-eps)[1][0])/(2*eps),
    ))
    np.testing.assert_allclose(tangent[0], numerical, rtol=4e-7, atol=.3)


def test_newton_probes_and_rejected_growth_cannot_mutate_accepted_history():
    m = owner()
    old, _, _ = evaluate(m, m.initial(), -.1, 1.)
    saved = snapshot(old)
    prepared, _ = m.prepare(old, m.front_depth(old)+2., [-.1, -.1], [1., 1.], 1.5)
    preparation = snapshot(prepared)
    for gap, jump in ((2., 3.), (-2., -3.), (50., 8.)):
        evaluate(m, prepared, gap, jump)
    assert_snapshot(old, saved)
    assert_snapshot(prepared, preparation)
    one = evaluate(m, prepared, .1, .3)
    two = evaluate(m, prepared, .1, .3)
    assert_snapshot(one[0], snapshot(two[0]))
    np.testing.assert_array_equal(one[1], two[1])
    np.testing.assert_array_equal(one[2], two[2])
    with pytest.raises(ValueError, match="remelting"):
        m.prepare(old, m.front_depth(old)-.01, [-.1, -.1], [1., 1.], 1.5)
    assert_snapshot(old, saved)


def test_full_fracture_spends_unchanged_reference_area_budget_once_per_bonded_cohort():
    m = owner()
    h, _ = m.prepare(m.initial(), [5., 7.], [0., 0.], [0., 0.], 1.5)
    for gap in (.01, .1, 1., 5., 30., 2., -.1, 50.):
        h, _, _ = evaluate(m, h, gap)
    old_budget = h.initial.area_ref_m2.sum()*LAW.fracture_energy_j_m2
    new_budget = h.added.area_ref_m2.sum()*LAW.fracture_energy_j_m2
    assert h.initial.fracture_work_j.sum() == pytest.approx(old_budget)
    assert h.added.fracture_work_j.sum() == pytest.approx(new_budget)
    after, _ = m.prepare(h, m.front_depth(h)+100., [50., 50.], [0., 0.], 1.6)
    assert after.initial.fracture_work_j.sum() == pytest.approx(old_budget)
    assert after.added.fracture_work_j.sum() == pytest.approx(new_budget)


def test_owned_birth_metadata_and_trial_arrays_are_not_writable():
    m = owner()
    for array in (m.birth_traction_pa, m.birth_edge_length_m_per_trace,
                  m.initial_cohorts.area_ref_m2, m.initial().initial.traction_pa):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_zero_growth_preserves_history_and_reports_zero_birth_energy():
    m = owner()
    h = m.initial()
    same, change = m.prepare(h, m.front_depth(h), [0., 0.], [0., 0.], 1.5)
    assert same is h
    assert change == 0
    # Floating-point roundoff is not physical remelting.
    same, change = m.prepare(h, m.front_depth(h)*(1-1e-15), [0., 0.], [0., 0.], 1.5)
    assert same is h
    assert change == 0


@pytest.mark.parametrize("clock", [1.3, -1., np.nan, np.inf, True])
def test_invalid_birth_clock_rejected(clock):
    m = owner()
    with pytest.raises(ValueError, match="birth time"):
        m.prepare(m.initial(), [3., 4.], [0., 0.], [0., 0.], clock)


@pytest.mark.parametrize("field,value", [
    ("z_hi_ref_m", [4., 3.]), ("area_ref_m2", [8., 12.]),
    ("birth_time_myr", [1.3, 1.4]), ("birth_jump_m", [1., 0.]),
    ("damage", [.3, .3]), ("fracture_work_j", [1., 0.]),
])
def test_original_metadata_and_extrinsic_history_corruption_is_rejected(field, value):
    m = owner()
    h = m.initial()
    corrupt = replace(h, initial=replace(h.initial, **{field: np.array(value)}))
    with pytest.raises(ValueError):
        m.validate(corrupt)


@pytest.mark.parametrize("field,value", [
    ("z_lo_ref_m", [2.1, 3.]), ("area_ref_m2", [8., 16.]),
    ("birth_time_myr", [1.3, 1.5]), ("damage", [.3, .3]),
])
def test_added_material_metadata_corruption_is_rejected(field, value):
    m = owner()
    h, _ = m.prepare(m.initial(), [4., 5.], [0., 0.], [0., 0.], 1.5)
    corrupt = replace(h, added=replace(h.added, **{field: np.array(value)}))
    with pytest.raises(ValueError):
        m.validate(corrupt)


def test_invalid_initial_area_or_nonzero_birth_reference_is_rejected():
    m = owner()
    for field, value in (("area_ref_m2", [8., 12.]), ("birth_jump_m", [1., 0.])):
        with pytest.raises(ValueError):
            MovingContactCohorts(replace(m.initial_cohorts, **{field: np.array(value)}),
                m.birth_traction_pa, m.birth_edge_length_m_per_trace, LAW)


def test_accepted_traction_and_opening_are_checked_without_advancing_history():
    m = owner()
    h = m.initial()
    with pytest.raises(ValueError, match="current opening"):
        m.validate(h, [.01, .01], [0., 0.])
    with pytest.raises(ValueError, match="traction"):
        m.validate(h, [0., 0.], [.01, .01])
    with pytest.raises(ValueError, match="Both"):
        m.validate(h, [0., 0.])


def test_empty_material_geometry_has_consistent_zero_energy_and_shapes():
    m = owner(n=0)
    h, de = m.prepare(m.initial(), [], [], [], 1.5)
    new, force, tangent = m.evaluate(h, [], [], 1., [])
    assert de == 0
    assert m.energy(new, [], []) == 0
    assert force.shape == (0, 2)
    assert tangent.shape == (0, 2, 2)
