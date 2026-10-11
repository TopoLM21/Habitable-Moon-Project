"""Actual-cut body ports checked by independent world Cartesian calculus."""
from dataclasses import replace
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from functools import lru_cache
import json

import pytest

from tectonics.geometric_native_pose import FractionInterval as I


def _api():
    from tectonics import geometric_native_common_contact_ports as api
    return api


@lru_cache(maxsize=1)
def _reference():
    from tectonics.geometric_native_common_reference import bind_native_common_reference
    return bind_native_common_reference(precision=96)


@lru_cache(maxsize=1)
def _ports():
    return _api().bind_native_common_contact_ports(_reference(), precision=96)


def _query(port, fraction=F(1, 3)):
    p = port.cut.parameter_guaranteed_bounds
    cube = port.cut.common_physical_radius_cubed_bounds_m3
    return I(p.lo+(p.hi-p.lo)*fraction), I(cube.lo+(cube.hi-cube.lo)*fraction)


@lru_cache(maxsize=6)
def _case(index):
    port = _ports().ports[index]
    parameter, cube = _query(port)
    dq = tuple(F((-1)**n, (n+1)*10000) for n in range(len(port.donor_mode_order)))
    rq = tuple(F((-1)**(n+1), (n+2)*10000) for n in range(len(port.receiver_mode_order)))
    return _api().evaluate_native_common_contact_port(_reference(), port,
        parameter_bounds=parameter, physical_radius_cubed_bounds_m3=cube,
        donor_coordinates_m=dq, receiver_coordinates_m=rq, precision=96)


def _d(x):
    return D(x.numerator)/D(x.denominator) if type(x) is F else D(x)


def _mid(x):
    return (_d(x.lo)+_d(x.hi))/2


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), D(0))


def _cross(a, b):
    return a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]


def _sin_cos(x):
    sine, cosine, a, b = x, D(1), x, D(1)
    for n in range(1, 140):
        a *= -x*x/D(2*n*(2*n+1))
        b *= -x*x/D((2*n-1)*2*n)
        sine += a
        cosine += b
    return sine, cosine


def _rotate(v, angle):
    norm = _dot(angle, angle).sqrt()
    if not norm:
        return v
    sine, cosine = _sin_cos(norm)
    a, b = _cross(angle, v), _cross(angle, _cross(angle, v))
    return tuple(v[i]+sine/norm*a[i]+(1-cosine)/(norm*norm)*b[i] for i in range(3))


def _atan(x):
    multiplier = 1
    while abs(x) > D(1)/8:
        x = x/(1+(1+x*x).sqrt())
        multiplier *= 2
    result, term = x, x
    for n in range(1, 180):
        term *= -x*x
        result += term/D(2*n+1)
    return multiplier*result


def _owner_inverse(body, v):
    for step in reversed(json.loads(body.original_body.source_owner_history_json)):
        angle = tuple(-_d(F.from_float(x))*_d(F.from_float(step["dt_myr"])) for x in step["omega_rad_per_myr"])
        v = _rotate(v, angle)
    return v


def _world_modes(body, world):
    R = _d(body.radius_m)
    if hasattr(body, "embedding_bank"):
        r = _dot(world, world).sqrt()
        tau = (R**3-r**3)/(3*R*R)
        original = _owner_inverse(body, _rotate(tuple(x/r for x in world),
            tuple(-_d(x)*tau for x in body.angular_rate_per_physical_volume_depth_m)))
        labels = tuple(_dot(tuple(_d(x) for x in axis), original) for axis in
                       (body.original_quarter_start, body.original_quarter_end, body.original_quarter_inward))
        horizontal = (labels[0]*labels[0]+labels[1]*labels[1]).sqrt()
        beta = _atan(labels[2]/horizontal)
        t = beta/_d(body.beta_cut_radians)
        chi = (1-t)**2*(t+2)/2 if beta < _d(body.beta_cut_radians) else D(0)
        modes = [tuple(chi*x/r for x in world)]
    else:
        chi, modes = D(1), []
    for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        modes.append(tuple(chi/R*(world[a] if i == a and a == b else
            world[b] if i == a else world[a] if i == b else D(0)) for i in range(3)))
    return tuple(modes)


def _close(bounds, expected, *, tolerance=D("1e-36")):
    error = D(0) if _d(bounds.lo) <= expected <= _d(bounds.hi) else min(abs(expected-_d(bounds.lo)), abs(expected-_d(bounds.hi)))
    assert error <= tolerance*max(D(1), abs(expected))


def test_six_ports_bind_five_true_cold_cuts_and_actual_curved_hot_receiver():
    value = _ports()
    assert len(value.ports) == 6
    hot = tuple(port for port in value.ports if port.cut.receiver_material_kind == "hot_birth")
    assert len(hot) == 1 and hot[0].cut.kind == "true_curved_envelope"
    assert len(hot[0].donor_mode_order) == 7 and len(hot[0].receiver_mode_order) == 6
    assert all(len(port.donor_mode_order) == len(port.receiver_mode_order) == 7
               for port in value.ports if port not in hot)
    assert not value.contact_constitutive_law_inferred
    assert not value.physical_material_admission_authorized
    assert not value.original_sixteen_coordinate_history_modified


@pytest.mark.parametrize("index", range(6))
def test_genuine_full_radius_body_modes_and_spatial_gradients_match_independent_world_differences(index):
    response = _case(index)
    assert response.geometry.genuine_query_support_certified
    assert response.geometry.every_point_of_parameter_enclosure_is_material
    assert response.geometry.physical_radius_bounds_m.hi < _reference().radius_m
    assert len(response.mode_jump_columns) == (13 if response.receiver_fields.material_kind == "hot_birth" else 14)
    with localcontext() as context:
        context.prec = 145
        # The cut's authenticated point is the input of this independent body
        # oracle.  The geometry module separately proves the actual cut itself.
        world = tuple(_mid(x) for x in response.geometry.world_position_bounds_m)
        for body, receipt in ((response.port.donor_body, response.donor_fields),
                              (response.port.receiver_body, response.receiver_fields)):
            expected = _world_modes(body, world)
            for column, mode in zip(receipt.mode_displacement_columns, expected):
                for bounds, x in zip(column, mode):
                    _close(bounds, x)
            h = D("0.00001")
            for axis in range(3):
                samples = {n: _world_modes(body, tuple(x+(n*h if i == axis else D(0))
                    for i, x in enumerate(world))) for n in (-2, -1, 1, 2)}
                for mode in range(len(expected)):
                    for row in range(3):
                        derivative = (-samples[2][mode][row]+8*samples[1][mode][row]
                            -8*samples[-1][mode][row]+samples[-2][mode][row])/(12*h)
                        _close(receipt.mode_spatial_gradient_columns_per_m[mode][row][axis], derivative,
                               tolerance=D("1e-24"))
            displacement = tuple(sum(_d(q)*mode[i] for q, mode in zip(receipt.coordinates_m, expected)) for i in range(3))
            for bounds, x in zip(receipt.displacement_increment_bounds_m, displacement):
                _close(bounds, x)
        donor_modes = _world_modes(response.port.donor_body, world)
        receiver_modes = _world_modes(response.port.receiver_body, world)
        columns = tuple(tuple(-x for x in mode) for mode in donor_modes)+receiver_modes
        directions = tuple(tuple(_mid(x) for x in vector) for vector in
            (response.geometry.receiver_inward_unit_normal_bounds,
             response.geometry.inward_unit_down_direction_bounds, response.geometry.angular_unit_tangent_bounds))
        for row, direction in zip(response.projected_mode_jacobian_bounds, directions):
            for bounds, column in zip(row, columns):
                _close(bounds, _dot(direction, column))


def test_symbolic_zero_jump_and_displaced_owner_left_columns_preserve_common_point_correlation():
    with localcontext() as context:
        context.prec = 130
        for index, port in enumerate(_ports().ports):
            parameter, cube = _query(port, F(2, 3))
            zero = _api().evaluate_native_common_contact_port(_reference(), port,
                parameter_bounds=parameter, physical_radius_cubed_bounds_m3=cube, precision=96)
            assert zero.displacement_jump_bounds_m == (I(0),)*3
            assert zero.normal_jump_bounds_m == zero.down_jump_bounds_m == zero.angular_jump_bounds_m == I(0)
            finite = _case(index)
            donor_owner, receiver_owner = port.donor_body.original_body.owner, port.receiver_body.original_body.owner
            for (owner, label), bounds in zip(port.owner_rotation_coordinate_order, finite.owner_rotation_jump_columns_m):
                axis = tuple(D(int(i == "xyz".index(label))) for i in range(3))
                # For distinct owners the column acts on its actual deformed
                # endpoint.  For the same owner the shared reference cancels.
                if donor_owner == receiver_owner == owner:
                    assert all(x == I(0) for row in zero.owner_rotation_jump_columns_m for x in row)
                    vector = tuple(_mid(x) for x in finite.displacement_jump_bounds_m)
                else:
                    vector = tuple((int(owner == receiver_owner)*_mid(r)-int(owner == donor_owner)*_mid(d))
                        for r, d in zip(finite.receiver_fields.displaced_point_bounds_m, finite.donor_fields.displaced_point_bounds_m))
                for actual, expected in zip(bounds, _cross(axis, vector)):
                    _close(actual, expected)
            assert finite.owner_columns_are_infinitesimal_left_chart_at_displaced_sides
            assert not finite.finite_owner_motion_or_whole_clock_certified


def test_binding_and_response_serializers_replay_actual_source_instead_of_trusting_a_seal():
    value = _ports()
    raw = _api().native_common_contact_ports_to_dict(value)
    assert _api().native_common_contact_ports_from_dict(raw, _reference()) == value
    response = _case(5)
    exported = _api().native_common_contact_port_response_to_dict(response)
    assert _api().native_common_contact_port_response_from_dict(exported, _reference()) == response
    forged = replace(response, displacement_jump_bounds_m=(I(99),)*3)
    forged = replace(forged, binding_sha256=_api()._seal(forged))
    with pytest.raises(ValueError, match="altered|re-sealed|authentic"):
        _api().validate_native_common_contact_port_response(forged)
    wrong = dict(exported)
    wrong["displacement_jump_bounds_m"] = [{"lo": [99, 1], "hi": [99, 1]}]*3
    with pytest.raises(ValueError, match="replay|Serialized|source"):
        _api().native_common_contact_port_response_from_dict(wrong, _reference())


def test_unknown_trim_root_band_is_rejected_and_hot_original_cold_labels_are_not_invented():
    port = next(p for p in _ports().ports if p.cut.kind == "true_curved_envelope")
    with pytest.raises(ValueError, match="unresolved|root|trim|material"):
        _api().evaluate_native_common_contact_port(_reference(), port, precision=96)
    response = next(_case(i) for i, p in enumerate(_ports().ports) if p == port)
    assert response.receiver_fields.original_unit_direction_bounds is None
    assert response.receiver_fields.material_beta_bounds_radians is None
    assert response.receiver_fields.coordinate_order == ("xx", "yy", "zz", "xy", "xz", "yz")
