"""Current literal memory checked independently of dense previous replay."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import hashlib
import json

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tectonics import geometric_native_common_contact_state as api
from tests.test_geometric_native_common_contact_ports import _reference, _case


def _decimal(x):
    return D(x.numerator)/D(x.denominator) if type(x) is F else D.from_float(x)


def _field(poly, theta):
    x = _decimal(theta)
    sine = a = x
    cosine = b = D(1)
    for n in range(1, 240):
        a *= -x*x/D(2*n*(2*n+1)); b *= -x*x/D((2*n-1)*2*n)
        sine += a; cosine += b
    basis = D(1), cosine, sine, cosine*cosine-sine*sine, 2*sine*cosine
    return sum((_decimal(F.from_float(c))*v for c, v in zip(poly['coefficients'], basis)), D(0))


def _contains(bounds, value):
    exact = F(value)
    assert bounds.lo-F(1, 10**145) <= exact <= bounds.hi+F(1, 10**145)


@pytest.mark.parametrize('index', range(5))
def test_actual_receiving_state_uses_literal_current_pieces_and_independent_pressure(index):
    response = _case(index)
    observed = api.observe_native_common_contact_state(_reference(), response, precision=160)
    assert observed.receiving_trace_id == response.port.cut.receiving_trace_id
    assert observed.receiving_trace_id != response.port.cut.incoming_donor_trace_id or index in (0,2,4)
    assert observed.source_time_fraction_bounds == I(1)
    assert observed.source_memory_elapsed_time_s == observed.source_native_elapsed_time_s == F(1000000)
    assert observed.source_contact_elapsed_time_s == F(2000000)
    assert observed.source_memory_stage_count == 1 and observed.source_contact_step_count == 2
    assert observed.source_current_node_epoch == 'unaccepted_candidate'
    assert observed.shear_traction_bounds_pa == I(0)
    assert observed.literal_current_memory_used and not observed.source_history_advanced
    assert not observed.face_depth_memory_or_measure_transfer_certified
    assert not observed.old_rod_prestrain_or_energy_transferred
    with localcontext() as context:
        context.prec = 180
        anchors = json.loads(observed.source_endpoint_anchor_fields_json)
        theta = (observed.computational_angle_bounds.lo + observed.computational_angle_bounds.hi)/2
        gap = _field(anchors[1], theta)
        pressure = -_decimal(observed.original_series_normal_stiffness_pa_m)*gap
        _contains(observed.free_gap_bounds_m, gap)
        _contains(observed.normal_pressure_bounds_pa, pressure)
        _contains(observed.contact_reference_gap_bounds_m, -pressure/_decimal(observed.normal_stiffness_pa_m))
        _contains(observed.contact_reference_energy_density_bounds_j_m2,
                  pressure*pressure/(2*_decimal(observed.normal_stiffness_pa_m)))
        for row in observed.memory_piece_intersections:
            piece = json.loads(row.original_piece_json)
            point = (row.computational_angle_bounds.lo + row.computational_angle_bounds.hi)/2
            _contains(row.plastic_slip_bounds_m, _field(piece['plastic_slip'], point))
            _contains(row.cumulative_slip_bounds_m, _field(piece['cumulative_slip'], point))


def test_current_coefficients_are_not_replaced_with_previous_or_endpoint_dense_expression():
    observed = api.observe_native_common_contact_state(_reference(), _case(0), precision=160)
    current = json.loads(observed.source_current_memory_trace_json)
    anchors = json.loads(observed.source_endpoint_anchor_fields_json)
    # This actual source contains a one-ulp difference; replacing saved p with
    # final free j or a dense recomputation would lose the current literal.
    assert current['pieces'][0]['plastic_slip']['coefficients'][0] != anchors[2]['coefficients'][0]
    assert observed.memory_piece_intersections[0].original_piece_json == api._encoded(current['pieces'][0]).decode()
    assert observed.plastic_slip_bounds_m != I(0)


def test_true_hot_curve_cannot_inherit_donor_or_other_born_contact_law():
    with pytest.raises(ValueError, match='no authenticated source receiving'):
        api.observe_native_common_contact_state(_reference(), _case(5), precision=96)


def test_reference_is_independent_of_modal_increment_but_bound_to_actual_material_label():
    from tectonics.geometric_native_common_contact_ports import evaluate_native_common_contact_port
    response = _case(0)
    zero = evaluate_native_common_contact_port(_reference(), response.port,
        parameter_bounds=response.geometry.parameter_bounds,
        physical_radius_cubed_bounds_m3=response.geometry.common_physical_radius_cubed_bounds_m3,
        precision=response.precision)
    a = api.observe_native_common_contact_state(_reference(), response, precision=96)
    b = api.observe_native_common_contact_state(_reference(), zero, precision=96)
    assert a == b
    with pytest.raises(ValueError, match='another physical material query'):
        api.validate_native_common_contact_state_observation(a, reference=_reference(), port_response=_case(1))


def test_serialized_current_state_replays_source_and_rejects_forged_memory_and_clocks():
    response = _case(2)
    observed = api.observe_native_common_contact_state(_reference(), response, precision=96)
    raw = api.native_common_contact_state_observation_to_dict(observed)
    assert api.native_common_contact_state_observation_from_dict(raw, reference=_reference(), port_response=response) == observed
    for key in ('source_contact_elapsed_time_s', 'normal_stiffness_pa_m', 'receiving_trace_id'):
        altered = deepcopy(raw)
        altered[key] = [1,1] if key != 'receiving_trace_id' else _case(0).port.cut.receiving_trace_id
        body = {k:v for k,v in altered.items() if k not in ('format','model','scope','binding_sha256')}
        altered['binding_sha256'] = hashlib.sha256(api._encoded(body)).hexdigest()
        with pytest.raises(ValueError, match='differs from source'):
            api.native_common_contact_state_observation_from_dict(altered, reference=_reference(), port_response=response)
    forged = replace(observed, source_history_advanced=True)
    forged = replace(forged, binding_sha256=api._seal(forged))
    with pytest.raises(ValueError, match='lacks genuine source replay'):
        api.validate_native_common_contact_state_observation(forged)
