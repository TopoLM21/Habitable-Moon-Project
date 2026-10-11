"""Source-bound NET-power theorem, explicit duration and strict restart controls."""
from dataclasses import replace
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

import pytest

from tectonics import geometric_native_common_basal_work_bound as api


ROOT = Path(__file__).resolve().parents[1]
PATHS = (
    'analysis/geometric_native_common_general_force_validation/development_native03/native_control.json',
    'analysis/geometric_native_common_cold_continuum_energy_validation/release_validation_01/native_control/source_packet.json',
    'analysis/geometric_native_common_cold_preparation_energy_validation/development_native01/native_control.json',
    'analysis/geometric_native_common_hot_mean_validation/release_validation_02/native_control/source_packet.json',
)


@pytest.fixture(scope='module')
def sources():
    return tuple(json.loads((ROOT/name).read_bytes()) for name in PATHS)


@pytest.fixture(scope='module')
def context(sources):
    return api.bind_native_common_preparation_work_source_context(*sources,
        provenance='Authentic four-source necessary-work test authority')


@pytest.fixture(scope='module')
def system(context):
    return api.bind_native_common_basal_work_bound(context,
        provenance='Fixed original 68-row NET-power upper envelope')


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def _tau(direction):
    beta, radius, k, omega = F(120000000000000), F(100000), F.from_float(.17), F.from_float(1.1e-12)
    x, y, _ = direction
    return (-beta*radius*(1+k*x)*omega*y, beta*radius*(1+k*x)*omega*x, F(0))


def _sphere(a, b):
    den = 1+a*a+b*b
    return (2*a/den, 2*b/den, (1-a*a-b*b)/den)


@pytest.mark.parametrize('a,b', [(F(0), F(0)), (F(1), F(0)), (F(-1), F(0)),
    (F(2, 3), F(-4, 5)), (F(-3, 7), F(5, 11)), (F(12), F(-17)), (F(1, 10**6), F(1000))])
def test_unit_direction_and_complete_square_are_global_in_position_and_velocity(a, b):
    direction = _sphere(a, b)
    assert _dot(direction, direction) == 1
    # Normalizing the actual position, rather than dividing by source R,
    # leaves this direction unchanged at arbitrarily different radii.
    for radial in (F(1, 10**15), F(99999), F(10**21)):
        position = tuple(radial*v for v in direction)
        assert _dot(position, position) == radial**2
        assert tuple(v/radial for v in position) == direction
    beta, k = F(120000000000000), F.from_float(.17)
    traction = _tau(direction)
    cap = _dot(traction, traction)/(4*beta)
    for velocity in ((F(0),)*3, (F(-7, 19), F(11, 23), F(13, 29)),
            tuple(v/(2*beta) for v in traction), tuple(-v/beta for v in traction)):
        actual = _dot(traction, velocity)-beta*_dot(velocity, velocity)
        square = beta*sum(((velocity[i]-traction[i]/(2*beta))**2 for i in range(3)), F(0))
        assert actual == cap-square
        assert actual <= cap
    x, _, z = direction
    # An exact sum of nonnegative terms proves the sphere bound; sampling
    # is only an additional algebra control, not the interval certificate.
    difference = k*(1-x)*(2+k*(1+x))+(1+k*x)**2*z*z
    assert difference == (1+k)**2-(1+k*x)**2*(1-z*z)
    assert difference >= 0


def test_generalized_power_chain_rule_uses_all_101_rates_including_lambda():
    beta = F(120000000000000)
    directions = (_sphere(F(2, 3), F(4, 7)), _sphere(F(-2), F(3, 5)), (F(1), F(0), F(0)))
    weights = (F(7, 11), F(13, 17), F(19, 23))
    # Authentic internal arm-only columns vanish; their rates are still
    # present in the full original catalogue and must invent no basal work.
    jacobians = tuple(tuple(tuple(F(0) if 92 <= j < 100 else
        F((s+2)*(axis+3)*(j+1)*(-1 if (s+axis+j)%2 else 1), 97+j)
        for j in range(101)) for axis in range(3)) for s in range(3))
    for rates in (tuple(F(j+1, 10**9) for j in range(101)),
            tuple(F((-1 if j%2 else 1)*(j+1), 10**7) for j in range(101)),
            (F(0),)*100+(F(2, 3),)):
        qforce = [F(0)]*101
        direct_net, source_power, dissipation = F(0), F(0), F(0)
        for w, direction, J in zip(weights, directions, jacobians):
            u = tuple(_dot(row, rates) for row in J)
            traction = _tau(direction)
            slip = tuple(t-beta*v for t, v in zip(traction, u))
            direct_net += w*_dot(slip, u)
            source_power += w*_dot(traction, u)
            dissipation += beta*w*_dot(u, u)
            for j in range(101):
                qforce[j] += w*sum((J[axis][j]*slip[axis] for axis in range(3)), F(0))
        assert _dot(qforce, rates) == direct_net == source_power-dissipation
        assert all(qforce[j] == 0 for j in range(92, 100))
        assert qforce[100] != 0
        assert direct_net <= sum((w*_dot(_tau(q), _tau(q))/(4*beta)
            for w, q in zip(weights, directions)), F(0))


def test_authentic_68_reference_weights_once_and_exact_literal_power(context, system, sources):
    evidence = api.native_common_preparation_work_source_context_evidence(context)['power_evidence']
    rows = evidence['original_positive_reference_rows']
    load = json.loads(sources[3]['hot_mean_system']['complete_current_load_json'])
    assert len(rows) == 68
    assert [row['source_material_cell_index'] for row in rows] == list(range(68))
    assert [row['original_row'] for row in rows] == load['material_cell_measures']
    assert sum(row['original_row']['material_kind'] == 'original_live' for row in rows) == 59
    assert sum(row['original_row']['material_kind'] == 'hot_birth' for row in rows) == 9
    weights = [(F(*r['area_bounds_m2']['lo']), F(*r['area_bounds_m2']['hi']))
        for r in load['material_cell_measures']]
    assert all(lo > 0 for lo, hi in weights)
    upper = sum((hi for _, hi in weights), F(0))
    expected_density = F(120000000000000)*100000**2*F.from_float(1.1e-12)**2*(1+F.from_float(.17))**2/4
    assert system.maximum_net_power_w == upper*expected_density
    assert system.maximum_net_power_density_w_m2 == expected_density
    assert F(62443640184) < system.maximum_net_power_w < F(62443640185)
    assert system.fixed_reference_trace_count == 68
    # Decimal17/100 and the binary64 literal1.17 both round below the
    # authentic coefficient 1+literal0.17 and cannot be silently substituted.
    assert 1+F.from_float(.17) > F(117, 100)
    assert 1+F.from_float(.17) > F.from_float(1.17)


def test_duration_is_an_explicit_parameter_not_an_original_clock(system):
    for duration in (F(0), F(7, 13), F(10**6), F(10**18)):
        response = api.evaluate_native_common_basal_work_bound(system, duration_s=duration)
        assert response.maximum_net_work_j == system.maximum_net_power_w*duration
        assert response.duration_s == duration
        assert response.net_power_is_after_drag is True
        assert response.drag_dissipation_subtracted_twice is False
        assert response.additional_preparation_actuator_work_included is False
        assert response.original_clocks_used_as_physical_duration is False
        assert response.clocks_advanced is False
        assert response.accepted_new_native_intervals == 0


@pytest.mark.parametrize('duration', [None, True, False, 1, .5, F(-1), '1000000'])
def test_missing_undefined_or_negative_duration_fails_closed(system, duration):
    with pytest.raises(ValueError):
        api.evaluate_native_common_basal_work_bound(system, duration_s=duration)


def test_omitting_duration_does_not_infer_a_clock(system):
    with pytest.raises(TypeError):
        api.evaluate_native_common_basal_work_bound(system)


def test_dto_and_response_restart_are_source_and_exact_duration_bound(context, system):
    raw = api.native_common_basal_work_bound_to_dict(system)
    assert api.native_common_basal_work_bound_from_dict(raw, context) is system
    response = api.evaluate_native_common_basal_work_bound(system, duration_s=F(17, 23))
    dto = api.native_common_basal_work_bound_response_to_dict(response)
    assert api.native_common_basal_work_bound_response_from_dict(dto,
        system=system, expected_duration_s=F(17, 23)) is response
    with pytest.raises(ValueError):
        api.native_common_basal_work_bound_response_from_dict(dto,
            system=system, expected_duration_s=F(17, 24))


def test_resealed_bool_int_response_and_system_forgery_are_rejected(context, system):
    response = api.evaluate_native_common_basal_work_bound(system, duration_s=F(1))
    dto = api.native_common_basal_work_bound_response_to_dict(response)
    dto['accepted_new_native_intervals'] = False
    dto['binding_sha256'] = api._digest({k: v for k, v in dto.items() if k not in ('format', 'binding_sha256')})
    with pytest.raises(ValueError):
        api.native_common_basal_work_bound_response_from_dict(dto, system=system, expected_duration_s=F(1))
    forged = replace(system, maximum_net_power_w=system.maximum_net_power_w+1)
    forged = replace(forged, binding_sha256=api._digest(api._body(forged)))
    with pytest.raises(ValueError):
        api.evaluate_native_common_basal_work_bound(forged, duration_s=F(1))
    raw = api.native_common_basal_work_bound_to_dict(system)
    raw['fixed_reference_trace_count'] = 67
    raw['binding_sha256'] = api._digest({k: v for k, v in raw.items() if k not in ('format', 'binding_sha256')})
    with pytest.raises(ValueError):
        api.native_common_basal_work_bound_from_dict(raw, context)


def test_compact_context_is_immutable_and_returns_fresh_evidence(context, system):
    evidence = api.native_common_preparation_work_source_context_evidence(context)
    evidence['power_evidence']['original_positive_reference_rows'].clear()
    assert len(api.native_common_preparation_work_source_context_evidence(context)
        ['power_evidence']['original_positive_reference_rows']) == 68
    assert api.evaluate_native_common_basal_work_bound(system, duration_s=F(3)).maximum_net_work_j == 3*system.maximum_net_power_w
    unregistered = replace(context)
    with pytest.raises(ValueError):
        api.bind_native_common_basal_work_bound(unregistered, provenance='Forged object identity')


def test_context_restart_authenticates_full_frozen_sources(context, sources):
    dto = api.native_common_preparation_work_source_context_to_dict(context)
    assert api.native_common_preparation_work_source_context_from_dict(dto, *sources) is context
    altered = dict(sources[0])
    altered['actual_fixed_reference_basal_trace_count'] = 67
    with pytest.raises(ValueError):
        api.bind_native_common_preparation_work_source_context(altered, *sources[1:])
    altered['sha256'] = hashlib.sha256(json.dumps({k: v for k, v in altered.items() if k != 'sha256'},
        sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError):
        api.bind_native_common_preparation_work_source_context(altered, *sources[1:])
    forged = dict(dto)
    forged['evidence_json'] = dto['evidence_json'].replace('"actual_source_rule_trace_count":68',
        '"actual_source_rule_trace_count":67')
    forged['binding_sha256'] = api._digest({k: v for k, v in forged.items() if k not in ('format', 'binding_sha256')})
    with pytest.raises(ValueError):
        api.native_common_preparation_work_source_context_from_dict(forged, *sources)
