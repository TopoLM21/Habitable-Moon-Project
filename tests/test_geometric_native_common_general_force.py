"""Actual material field and finite-chart checks of the declared101 force."""
import copy
from dataclasses import replace
from fractions import Fraction as F
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import mpmath as mp
import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tectonics import geometric_native_common_general_force as api
from tectonics.geometric_native_common_carrier_source_work import _interval, _fraction

ROOT = Path(__file__).resolve().parents[1]


def overlaps(a, b):
    return max(a.lo, b.lo) <= min(a.hi, b.hi)


def _mp(value):
    if type(value) is F:
        return mp.mpf(value.numerator)/value.denominator
    return mp.mpf(value)


def contains(interval, value):
    return _mp(interval.lo) <= value <= _mp(interval.hi)


def test_diagnostic_finite_owner_rotation_keeps_true_first_and_second_chart_derivatives():
    theta = (F(1, 10000), F(-2, 10000), F(3, 10000))
    rotation, receipt = api._owner_rotation(theta, 1, 96)
    with mp.workdps(220):
        x = tuple(_mp(v) for v in theta)
        def entry(i, j):
            def f(a, b, c):
                return mp.expm(mp.matrix([[0, -c, b], [c, 0, -a], [-b, a, 0]]))[i, j]
            return f
        for i in range(3):
            for j in range(3):
                f, jet = entry(i, j), rotation[i][j]
                assert contains(jet.value, f(*x))
                for a in range(3):
                    orders = tuple(int(k == a) for k in range(3))
                    assert contains(jet.gradient.get(3+a, I(0)), mp.diff(f, x, orders))
                    for b in range(a, 3):
                        orders = tuple(int(k == a)+int(k == b) for k in range(3))
                        assert contains(jet.hessian.get((3+a, 3+b), I(0)), mp.diff(f, x, orders))
    assert receipt['generator_infinity_norm_upper'] == [1, 2000]
    assert receipt['zero_chart_exact_second_order_identity'] is False


def test_diagnostic_zero_owner_chart_is_exact_and_has_no_angle_division():
    rotation, receipt = api._owner_rotation((F(0),)*3, 0, 96)
    assert receipt['zero_chart_exact_second_order_identity'] is True
    assert receipt['value_first_second_tail_upper'] == [[0, 1]]*3
    assert all(rotation[i][j].value == I(int(i == j)) for i in range(3) for j in range(3))
    assert rotation[1][2].gradient[0] == I(-1)
    assert rotation[2][1].gradient[0] == I(1)
    assert rotation[0][0].hessian[1, 1] == I(-1)
    assert rotation[0][1].hessian[0, 1] == I(F(1, 2))


@pytest.mark.parametrize('theta', [(0.1, F(0), F(0)), (F(1), F(0), F(0))])
def test_diagnostic_owner_chart_rejects_binary_float_and_unproved_large_domain(theta):
    with pytest.raises(ValueError):
        api._owner_rotation(theta, 0, 96)


def test_diagnostic_prepared_original_gradient_keeps_body_curvature_and_lambda_cross_terms():
    prep = SimpleNamespace(declaration=SimpleNamespace(radius_m=F(100000), fixed_radial_scale=F(1),
        angular_rate_per_physical_volume_depth_m=(F(0), F(0), F(1, 1000000))))
    x = (api._Jet.variable(F(99001), 36), api._Jet(2000), api._Jet(1000))
    A = tuple(tuple(api._Jet(int(i == j), {36: I(F(1, 100000)) if (i, j) == (0, 0) else I(0)})
                    for j in range(3)) for i in range(3))
    lam = api._Jet.variable(F(3, 10), 100)
    _, actual, _, _ = api._prepared_fields(x, A, prep, lam, 96)
    with mp.workdps(220):
        def entry(i, j):
            def f(c, l):
                xx = mp.matrix([99000+c, 2000, 1000])
                r = mp.sqrt(sum(v*v for v in xx))
                tau = (mp.mpf(100000)**3-r**3)/(3*mp.mpf(100000)**2)
                g = -r*xx/mp.mpf(100000)**2
                W = mp.matrix([[0, -mp.mpf('0.000001'), 0], [mp.mpf('0.000001'), 0, 0], [0, 0, 0]])
                Q = mp.expm(l*tau*W)
                aa = mp.eye(3); aa[0, 0] += (c-1)/100000
                return (Q*(mp.eye(3)+l*(W*xx)*g.T)*aa)[i, j]
            return f
        state = (mp.mpf(1), mp.mpf(3)/10)
        for i in range(3):
            for j in range(3):
                f, jet = entry(i, j), actual[i][j]
                assert contains(jet.value, f(*state))
                assert contains(jet.gradient.get(36, I(0)), mp.diff(f, state, (1, 0)))
                assert contains(jet.hessian.get((36, 36), I(0)), mp.diff(f, state, (2, 0)))
                assert contains(jet.hessian.get((36, 100), I(0)), mp.diff(f, state, (1, 1)))
    assert any(v.hessian.get((36, 36), I(0)) != I(0) for row in actual for v in row)
    assert any(v.hessian.get((36, 100), I(0)) != I(0) for row in actual for v in row)


def test_diagnostic_transported_internal_arm_uses_actual_gradient_body_internal_curvature():
    prep = SimpleNamespace(declaration=SimpleNamespace(radius_m=F(100000), fixed_radial_scale=F(1),
        angular_rate_per_physical_volume_depth_m=(F(0), F(0), F(1, 1000000))))
    x = (api._Jet.variable(F(99000), 36), api._Jet(2000), api._Jet(1000))
    A = api._identity(96)
    _, original, _, _ = api._prepared_fields(x, A, prep, api._Jet.variable(F(1, 2), 100), 96)
    internal = api._Jet.variable(F(1, 1000), 92)
    transported = api._mv(original, (api._Jet(3)+internal, api._Jet(0), api._Jet(0)))
    assert any(v.hessian.get((36, 92), I(0)) != I(0) for v in transported)
    assert any(v.hessian.get((92, 100), I(0)) != I(0) for v in transported)
    normal, _ = api._normal(original, (I(1), I(0), I(0)), 96)
    residual = api._dot(normal, normal).value-I(1)
    assert residual.lo <= 0 <= residual.hi


@pytest.fixture(scope='module')
def context():
    return runpy.run_path(str(ROOT/'analysis/geometric_native_common_general_force_validation/native_control.txt'))['context']()


@pytest.fixture(scope='module')
def zero(context):
    return api.evaluate_native_common_general_force(context['general_system'], (F(0),)*101,
        coordinate_rates=(F(0),)*101)


@pytest.fixture(scope='module')
def finite(context):
    hot = tuple(F(*v) for v in context['source903']['evaluation_queries'][1]['response']['coordinate_increments'])
    cold = tuple(v/4 for v in context['cold_carrier'].bound_coordinate_increments)
    owners = tuple(F((-1 if i % 2 else 1)*(i+1), 10000) for i in range(6))
    internal = tuple(F((-1 if i % 2 else 1)*(i+1), 1000000) for i in range(8))
    state = owners+hot+cold+internal+(F(1, 2),)
    rates = tuple(F((-1 if i % 2 else 1)*(i+1), 10**16) for i in range(100))+(F(2, 10**12),)
    return api.evaluate_native_common_general_force(context['general_system'], state, coordinate_rates=rates)


def test_complete_catalogue_and_physical_scope_do_not_change_clock_or_acceptance(context, zero):
    assert len(zero.coordinate_order) == len(zero.residual_bounds) == 101
    assert all(len(row) == 101 for row in zero.state_tangent_bounds+zero.velocity_tangent_bounds)
    assert zero.source_clocks_s == (F(0), F(2000000), F(1000000), F(1000000))
    assert zero.accepted_old_native_intervals == 1 and zero.accepted_new_native_intervals == 0
    assert zero.clocks_advanced is False
    assert zero.complete_declared_general_state_operator_computed is True
    assert zero.general_state_full_physics_operator_certified is False
    assert zero.reference_preparation_work_paid is False
    assert zero.new_mixed_force_equilibrium_certified is False
    assert zero.whole_clock_all_pair_admission_authorized is False
    assert zero.full_system_temporal_path_work_certified is False
    assert zero.remaining_condition_count == 3


def test_zero_source_contacts_match_authentic907_full_coordinate_derivatives(context, zero):
    old = next(q['response'] for q in context['source907']['evaluation_queries']
               if all(v == [0, 1] for v in q['response']['coordinate_increments']))
    oldatoms = {r['point_id']: r for r in map(json.loads, old['contact_atoms_json'])}
    for new in map(json.loads, zero.contact_atoms_json):
        prior = oldatoms[new['point_id']]
        g = dict((i, _interval(v)) for i, v in new['energy_twojet']['first_derivative_bounds'])
        h = dict(((i, j), _interval(v)) for i, j, v in new['energy_twojet']['second_derivative_bounds'])
        ids = prior['active_coordinate_indices']
        for a, i in enumerate(ids):
            assert overlaps(g.get(i, I(0)), _interval(prior['gradient_bounds'][a]))
            for b, j in enumerate(ids):
                assert overlaps(h.get(api._pair(i, j), I(0)), _interval(prior['hessian_bounds'][a][b]))


def test_all_contact_material_fields_and_true_finite_owner_basal_rows_retained(finite):
    sides = list(map(json.loads, finite.contact_side_receipts_json))
    assert len(sides) == 32
    assert sum(r['allocation'] == 'live' for r in sides) == 24
    assert sum(r['allocation'] == 'archived' for r in sides) == 8
    assert len(finite.contact_atoms_json) == 16 and len(finite.basal_trace_records_json) == 68
    assert all(r['full_body_internal_preparation_cross_curvature_included'] for r in sides)
    rows = list(map(json.loads, finite.basal_trace_records_json))
    assert all(r['owner_rotation_query_is_zero_slice'] is False for r in rows)
    assert all(r['finite_owner_chart_first_and_second_derivatives_retained'] for r in rows)
    basal = json.loads(finite.basal_response_json)
    assert _interval(basal['drag_dissipation_bounds_w']).lo >= 0
    assert any(finite.velocity_tangent_bounds[i][i].lo > 0 for i in range(6, 92))


def test_operator_signs_and_complete_matrix_curvature(finite):
    basal = json.loads(finite.basal_response_json)
    for i in range(101):
        assert overlaps(finite.residual_bounds[i], finite.conservative_gradient_bounds[i]-_interval(basal['residual_bounds'][i]))
        for j in range(101):
            assert finite.conservative_hessian_bounds[i][j] == finite.conservative_hessian_bounds[j][i]
            assert overlaps(finite.state_tangent_bounds[i][j],
                finite.conservative_hessian_bounds[i][j]-_interval(basal['state_tangent_bounds'][i][j]))
            assert finite.velocity_tangent_bounds[i][j] == _interval(basal['drag_bounds'][i][j])
    assert any(not overlaps(finite.state_tangent_bounds[i][j], finite.state_tangent_bounds[j][i])
        for i in range(36) for j in range(i))
    assert all(finite.velocity_tangent_bounds[i][j] == I(0) for i in range(92, 100) for j in range(101))
    assert any(finite.conservative_hessian_bounds[i][100] != I(0) for i in range(36, 92))
    assert any(finite.conservative_hessian_bounds[i][100] != I(0) for i in range(92, 100))


def test_internal_beams_added_once_and_frozen_owner_potential_replaced(finite):
    parts = json.loads(finite.energy_partitions_json)
    assert parts['frozen_owner_load_drag_potential_added'] is False
    assert finite.frozen_owner_load_potential_used is False
    beams = list(map(json.loads, finite.internal_beam_receipts_json))
    assert len(beams) == 8
    assert {b['global_internal_coordinate_index'] for b in beams} == set(range(92, 100))
    assert all(b['original_beam_energy_added_again'] is False for b in beams)
    assert all(finite.proposed_normal_damage_or_fracture_work_applied is False for _ in beams)


def test_same_state_opposite_rates_keep_energy_and_drag_and_reverse_source_power(context, finite):
    opposite = api.evaluate_native_common_general_force(context['general_system'], finite.coordinate_values,
        coordinate_rates=tuple(-v for v in finite.coordinate_rates))
    assert opposite.energy_bounds_j == finite.energy_bounds_j
    assert opposite.conservative_hessian_bounds == finite.conservative_hessian_bounds
    assert opposite.velocity_tangent_bounds == finite.velocity_tangent_bounds
    plus, minus = json.loads(finite.basal_response_json), json.loads(opposite.basal_response_json)
    assert _interval(plus['drag_dissipation_bounds_w']) == _interval(minus['drag_dissipation_bounds_w'])
    assert overlaps(_interval(plus['source_power_bounds_w']), -_interval(minus['source_power_bounds_w']))


def test_system_restart_requires_full_source_and_declared_provenance(context):
    system = context['general_system']
    ports = dict(cold_carrier=context['cold_carrier'], hot_system=context['hot_system'],
        preparations=context['preparations'], basal_trace_system=context['trace_system'],
        cold_energy_system=context['cold_energy_system'])
    dto = api.native_common_general_force_to_dict(system)
    restored = api.native_common_general_force_from_dict(dto, context['source907'], context['source895'], context['source903'], **ports)
    assert api.native_common_general_force_to_dict(restored) == dto
    changed = copy.deepcopy(dto); changed['provenance'] = 'other source binding'
    with pytest.raises(ValueError):
        api.native_common_general_force_from_dict(changed, context['source907'], context['source895'], context['source903'], **ports)
    with pytest.raises(ValueError):
        api.validate_native_common_general_force(replace(system, provenance='resealed context'))


def test_response_restart_rejects_query_and_operator_transplants(context, finite):
    dto = api.native_common_general_force_response_to_dict(finite)
    restored = api.native_common_general_force_response_from_dict(dto, system=context['general_system'],
        expected_coordinates101=finite.coordinate_values, expected_coordinate_rates=finite.coordinate_rates)
    assert api.native_common_general_force_response_to_dict(restored) == dto
    changed = copy.deepcopy(dto); changed['state_tangent_bounds'][0][0]['lo'][0] += 1
    with pytest.raises(ValueError):
        api.native_common_general_force_response_from_dict(changed, system=context['general_system'],
            expected_coordinates101=finite.coordinate_values, expected_coordinate_rates=finite.coordinate_rates)
    with pytest.raises(ValueError):
        api.native_common_general_force_response_from_dict(dto, system=context['general_system'],
            expected_coordinates101=finite.coordinate_values, expected_coordinate_rates=(F(0),)*101)


@pytest.mark.parametrize('rates', [None, (0.0,)*101, (F(0),)*100])
def test_physical_rates_cannot_be_inferred_or_silently_filled(context, rates):
    with pytest.raises(ValueError):
        api.evaluate_native_common_general_force(context['general_system'], (F(0),)*101, coordinate_rates=rates)
