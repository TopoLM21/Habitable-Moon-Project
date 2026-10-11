"""Objective material-director contact calculus and authentic source proposals."""
from dataclasses import replace
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from functools import lru_cache

import pytest

from tectonics.geometric_native_pose import FractionInterval as I


def _api():
    from tectonics import geometric_native_common_contact_constitutive as api
    return api


def _d(value):
    return D(value.numerator)/D(value.denominator) if type(value) is F else D(value)


def _mid(value):
    return (_d(value.lo)+_d(value.hi))/2


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), D(0))


def _cross(a, b):
    return a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]


def _rotate(vector, angle):
    length = _dot(angle, angle).sqrt()
    if not length:
        return vector
    sine, cosine, a, b = length, D(1), length, D(1)
    for n in range(1, 100):
        a *= -length*length/D(2*n*(2*n+1))
        b *= -length*length/D((2*n-1)*2*n)
        sine += a
        cosine += b
    av, aav = _cross(angle, vector), _cross(angle, _cross(angle, vector))
    return tuple(vector[i]+sine/length*av[i]+(1-cosine)/(length*length)*aav[i] for i in range(3))


def _close(bounds, value, *, tolerance=D("1e-22")):
    error = D(0) if _d(bounds.lo) <= value <= _d(bounds.hi) else min(
        abs(value-_d(bounds.lo)), abs(value-_d(bounds.hi)))
    assert error <= tolerance*max(D(1), abs(value))


def _finite_projection(donor, receiver, directions, donor_owner, receiver_owner,
                       owner_order, offsets):
    body_count = len(donor[1])+len(receiver[1])
    angles = {owner: [D(0)]*3 for owner, _ in owner_order}
    for (owner, axis), offset in zip(owner_order, offsets[body_count:]):
        angles[owner]["xyz".index(axis)] += offset
    nd = len(donor[1])
    yd = tuple(donor[0][i]+sum((offset*mode[i] for offset, mode in
        zip(offsets[:nd], donor[1])), D(0)) for i in range(3))
    yr = tuple(receiver[0][i]+sum((offset*mode[i] for offset, mode in
        zip(offsets[nd:body_count], receiver[1])), D(0)) for i in range(3))
    yd, yr = _rotate(yd, angles[donor_owner]), _rotate(yr, angles[receiver_owner])
    jump = tuple(r-d for d, r in zip(yd, yr))
    frame = tuple(_rotate(direction, angles[receiver_owner]) for direction in directions)
    return tuple(_dot(e, jump) for e in frame), yd, yr, frame


def _energy(kinematic_data, pressure, kn, kt, branch, offsets):
    projections, _, _, _ = _finite_projection(*kinematic_data, offsets)
    gap = -pressure/kn+projections[0]
    return kn*gap*gap/2+(D(0) if branch == "relaxed_source" else kt*projections[1]**2/2)


def _fd_first(function, zero, i, h):
    return sum((weight*function(tuple(x+n*h if k == i else x for k, x in enumerate(zero)))
        for n, weight in ((-2, D(1)), (-1, D(-8)), (1, D(8)), (2, D(-1)))), D(0))/(12*h)


def _fd_second(function, zero, i, j, hi, hj):
    if i == j:
        return sum((weight*function(tuple(x+n*hi if k == i else x for k, x in enumerate(zero)))
            for n, weight in ((-2, D(-1)), (-1, D(16)), (0, D(-30)),
                              (1, D(16)), (2, D(-1)))), D(0))/(12*hi*hi)
    weights = ((-2, D(1)), (-1, D(-8)), (1, D(8)), (2, D(-1)))
    return sum((wi*wj*function(tuple(x+(ni*hi if k == i else D(0))
        +(nj*hj if k == j else D(0)) for k, x in enumerate(zero)))
        for ni, wi in weights for nj, wj in weights), D(0))/(144*hi*hj)


def _synthetic(shared=False):
    """Small exact algebra fixture, explicitly not an authenticated material case."""
    api = _api()
    donor_owner, receiver_owner = 0, 0 if shared else 1
    owner_order = tuple((o, a) for o in (0, 1) for a in "xyz")
    X = tuple(I(x) for x in (3, -4, 5))
    ud, ur = tuple(I(F(x, 200)) for x in (1, 2, -1)), tuple(I(F(x, 200)) for x in (-2, 1, 3))
    donor_modes = tuple(tuple(I(F(x, 7)) for x in mode) for mode in ((1, 2, 3), (-2, 1, 4)))
    receiver_modes = tuple(tuple(I(F(x, 11)) for x in mode) for mode in ((2, -1, 3), (4, 2, -1)))
    columns = tuple(tuple(-x for x in mode) for mode in donor_modes)+receiver_modes
    jump = tuple(r-d for d, r in zip(ud, ur))
    first = list(columns)
    second = [[(I(0),)*3 for _ in range(10)] for _ in range(10)]
    for a, (owner, axis) in enumerate(owner_order, 4):
        d, r = int(owner == donor_owner), int(owner == receiver_owner)
        v = tuple((r-d)*x+r*y-d*z for x, y, z in zip(X, ur, ud))
        first.append(api._cross(api._axis(axis), v))
        for b, (column, body_owner) in enumerate(zip(columns, (donor_owner,)*2+(receiver_owner,)*2)):
            value = api._cross(api._axis(axis), column) if owner == body_owner else (I(0),)*3
            second[a][b] = second[b][a] = value
        for b, (other, other_axis) in enumerate(owner_order, 4):
            if other == owner:
                second[a][b] = api._symmetric_owner_second(api._axis(axis), api._axis(other_axis), v)
    directions = tuple(tuple(I(int(i == k)) for i in range(3)) for k in range(3))
    result = api._scalar_derivatives(jump, columns, tuple(first), second, directions,
        receiver_owner, owner_order, 4, shared, 96)
    numeric = (((tuple(_mid(x+y) for x, y in zip(X, ud)),
                 tuple(tuple(_mid(x) for x in m) for m in donor_modes))),
               ((tuple(_mid(x+y) for x, y in zip(X, ur)),
                 tuple(tuple(_mid(x) for x in m) for m in receiver_modes))),
               tuple(tuple(_mid(x) for x in e) for e in directions),
               donor_owner, receiver_owner, owner_order)
    return result, numeric


@pytest.mark.parametrize("shared", (False, True))
def test_pure_material_frame_first_and_symmetric_second_partials_match_finite_rotations(shared):
    with localcontext() as context:
        context.prec = 110
        result, numeric = _synthetic(shared)
        zero = (D(0),)*10
        for component in range(3):
            function = lambda offsets: _finite_projection(*numeric, offsets)[0][component]
            _close(result[0][component], function(zero))
            for i in range(10):
                hi = D("1e-7") if i < 4 else D("1e-12")
                _close(result[3][component][i], _fd_first(function, zero, i, hi))
                for j in range(i, 10):
                    hj = D("1e-7") if j < 4 else D("1e-12")
                    _close(result[4][component][i][j], _fd_second(function, zero, i, j, hi, hj))
                    assert result[4][component][i][j] == result[4][component][j][i]
                    if shared and (i >= 4 or j >= 4):
                        assert result[4][component][i][j] == I(0)


def test_pure_material_director_energy_is_objective_under_finite_superposed_rotation():
    with localcontext() as context:
        context.prec = 110
        _, numeric = _synthetic()
        offsets = (D("0.03"), D("-0.02"), D("0.01"), D("0.04"),
                   D("0.13"), D("-0.09"), D("0.07"), D("-0.17"), D("0.11"), D("0.05"))
        xi, yd, yr, frame = _finite_projection(*numeric, offsets)
        rotation = tuple(D(x) for x in ("0.31", "-0.27", "0.19"))
        dyr, dyd = _rotate(yr, rotation), _rotate(yd, rotation)
        moved = tuple(_dot(_rotate(e, rotation), tuple(r-d for r, d in zip(dyr, dyd))) for e in frame)
        assert max(abs(x-y) for x, y in zip(xi, moved)) < D("1e-95")
        for branch in ("relaxed_source", "frozen_memory_predictor"):
            def W(values):
                return D(1000000)*(-D("0.2")+values[0])**2/2+(
                    D(0) if branch == "relaxed_source" else D(1000000)*values[1]**2/2)
            assert abs(W(xi)-W(moved)) < D("1e-90")


@lru_cache(maxsize=3)
def _actual(index):
    # Reuse the frozen847 expensive immutable source/port fixture in the
    # combined gate; all force/state observations below are genuinely new.
    from tests.test_geometric_native_common_contact_ports import _reference, _case
    from tectonics.geometric_native_common_contact_state import observe_native_common_contact_state
    port = _case(index)
    state = observe_native_common_contact_state(_reference(), port, precision=96)
    return _reference(), port, state


def _numeric_actual(port):
    donor = tuple(_mid(x) for x in port.donor_fields.displaced_point_bounds_m)
    receiver = tuple(_mid(x) for x in port.receiver_fields.displaced_point_bounds_m)
    modes_d = tuple(tuple(_mid(x) for x in v) for v in port.donor_fields.mode_displacement_columns)
    modes_r = tuple(tuple(_mid(x) for x in v) for v in port.receiver_fields.mode_displacement_columns)
    directions = tuple(tuple(_mid(x) for x in v) for v in (
        port.geometry.receiver_inward_unit_normal_bounds,
        port.geometry.inward_unit_down_direction_bounds, port.geometry.angular_unit_tangent_bounds))
    return ((donor, modes_d), (receiver, modes_r), directions,
            port.port.donor_body.original_body.owner, port.port.receiver_body.original_body.owner,
            port.port.owner_rotation_coordinate_order)


@pytest.mark.parametrize("branch", ("relaxed_source", "frozen_memory_predictor"))
def test_actual_nonzero_body_energy_force_hessian_include_preload_and_full_receiver_frame(branch):
    reference, port, state = _actual(0)
    value = _api().evaluate_native_common_contact_constitutive(reference, port, state,
        branch=branch, precision=96)
    with localcontext() as context:
        context.prec = 110
        numeric = _numeric_actual(port)
        zero = (D(0),)*len(value.kinematics.coordinate_order)
        body_count = value.kinematics.body_coordinate_count
        function = lambda offsets: _energy(numeric, _mid(state.normal_pressure_bounds_pa),
            _d(state.normal_stiffness_pa_m), _d(state.tangential_stiffness_pa_m), branch, offsets)
        _close(value.recoverable_energy_bounds_j_m2, function(zero))
        for i in range(len(zero)):
            hi = D("1e-7") if i < body_count else D("1e-12")
            _close(value.residual_bounds_per_reference_m2[i], _fd_first(function, zero, i, hi))
            for j in range(i, len(zero)):
                hj = D("1e-7") if j < body_count else D("1e-12")
                _close(value.hessian_bounds_per_reference_m2[i][j], _fd_second(function, zero, i, j, hi, hj))
                assert value.hessian_bounds_per_reference_m2[i][j] == value.hessian_bounds_per_reference_m2[j][i]
        # The preload second term is material, not optional.  Find an actual
        # owner-owner entry whose independently computed full Hessian differs
        # from the stiffness Gram by more than rounding/finite-difference tails.
        first = value.kinematics.scalar_first_derivative_bounds
        differences = []
        for i in range(body_count, len(zero)):
            for j in range(body_count, len(zero)):
                gram = _d(state.normal_stiffness_pa_m)*_mid(first[0][i])*_mid(first[0][j])
                if branch == "frozen_memory_predictor":
                    gram += _d(state.tangential_stiffness_pa_m)*_mid(first[1][i])*_mid(first[1][j])
                differences.append(abs(_mid(value.hessian_bounds_per_reference_m2[i][j])-gram))
        if numeric[3] != numeric[4]:
            assert max(differences) > D(1)


def test_actual_relaxed_source_proposes_plastic_memory_without_advancing_or_assigning_face_energy():
    reference, port, state = _actual(2)
    api = _api()
    relaxed = api.evaluate_native_common_contact_constitutive(reference, port, state, precision=96)
    frozen = api.evaluate_native_common_contact_constitutive(reference, port, state,
        branch="frozen_memory_predictor", precision=96)
    assert relaxed.traction_bounds_pa[1] == I(0)
    assert relaxed.elastic_shear_extension_bounds_m == I(0)
    assert state.tangential_stiffness_pa_m*relaxed.elastic_shear_extension_bounds_m == relaxed.traction_bounds_pa[1]
    assert relaxed.local_tangent_bounds_pa_m[1][1] == I(0)
    assert frozen.local_tangent_bounds_pa_m[1][1] == I(state.tangential_stiffness_pa_m)
    assert frozen.proposed_plastic_increment_bounds_m == I(0)
    _close(state.tangential_stiffness_pa_m*frozen.elastic_shear_extension_bounds_m,
        _mid(frozen.traction_bounds_pa[1]))
    assert relaxed.proposed_plastic_increment_bounds_m == relaxed.trial_elastic_shear_extension_bounds_m
    _close(relaxed.proposed_plastic_increment_bounds_m, _mid(relaxed.kinematics.scalar_increment_bounds_m[1]))
    _close(relaxed.proposed_cumulative_slip_bounds_m,
        _mid(state.cumulative_slip_bounds_m)+abs(_mid(relaxed.proposed_plastic_increment_bounds_m)))
    for value in (relaxed, frozen):
        assert not value.source_memory_proposal_applied
        assert not value.material_memory_or_interface_measure_transferred
        assert not value.old_sixteen_atoms_or_rod_energy_assigned
        assert not value.source_state_or_thermal_state_advanced
        assert not value.force_equilibrium_certified


def test_actual_constitutive_replay_and_resealed_forgery_rejection():
    reference, port, state = _actual(0)
    api = _api()
    value = api.evaluate_native_common_contact_constitutive(reference, port, state, precision=96)
    raw = api.native_common_contact_constitutive_response_to_dict(value)
    replay = api.native_common_contact_constitutive_response_from_dict(raw, reference=reference,
        port_response=port, source_observation=state)
    assert replay == value
    forged = replace(value, recoverable_energy_bounds_j_m2=value.recoverable_energy_bounds_j_m2+1)
    forged = replace(forged, binding_sha256=api._seal(forged))
    with pytest.raises(ValueError):
        api.validate_native_common_contact_constitutive_response(forged)
    changed = replace(state, normal_pressure_bounds_pa=state.normal_pressure_bounds_pa+1)
    with pytest.raises(ValueError):
        api.evaluate_native_common_contact_constitutive(reference, port, changed, precision=96)
    raw["hessian_bounds_per_reference_m2"][0][0]["lo"][0] += 1
    with pytest.raises(ValueError):
        api.native_common_contact_constitutive_response_from_dict(raw, reference=reference,
            port_response=port, source_observation=state)


def test_actual_zero_increment_keeps_nonzero_current_source_pressure_energy_and_geometric_preload():
    from tectonics.geometric_native_common_contact_ports import evaluate_native_common_contact_port
    from tectonics.geometric_native_common_contact_state import observe_native_common_contact_state
    reference, original, state = _actual(0)
    # A deliberately finer authentic observation must survive symbolic
    # zero-increment correlation in a lower-precision force response.
    state = observe_native_common_contact_state(reference, original, precision=160)
    zero = evaluate_native_common_contact_port(reference, original.port,
        parameter_bounds=original.geometry.parameter_bounds,
        physical_radius_cubed_bounds_m3=original.geometry.common_physical_radius_cubed_bounds_m3,
        precision=96)
    assert state.normal_pressure_bounds_pa.lo > 0
    for branch in ("relaxed_source", "frozen_memory_predictor"):
        value = _api().evaluate_native_common_contact_constitutive(reference, zero, state,
            branch=branch, precision=96)
        assert value.kinematics.scalar_increment_bounds_m == (I(0),)*3
        assert value.traction_bounds_pa == (-state.normal_pressure_bounds_pa, I(0))
        assert value.contact_gap_bounds_m == state.contact_reference_gap_bounds_m
        assert value.recoverable_energy_bounds_j_m2 == value.reference_recoverable_energy_bounds_j_m2
        assert value.recoverable_energy_bounds_j_m2.lo > 0
        owner = value.kinematics.body_coordinate_count
        assert any(value.kinematics.scalar_second_derivative_bounds[0][i][j] != I(0)
            for i in range(owner, owner+6) for j in range(owner, owner+6))


def test_actual_shared_owner_hot_scalar_derivatives_cancel_without_inventing_a_hot_contact_law():
    from tests.test_geometric_native_common_contact_ports import _reference, _case
    port = _case(5)
    value = _api().evaluate_native_common_contact_scalar_kinematics(_reference(), port, precision=96)
    assert value.donor_owner == value.receiver_owner
    count, owner = len(value.coordinate_order), value.body_coordinate_count
    assert owner == 13
    for first, second in zip(value.scalar_first_derivative_bounds, value.scalar_second_derivative_bounds):
        assert first[owner:] == (I(0),)*6
        assert all(second[i][j] == I(0) for i in range(count) for j in range(count)
                   if i >= owner or j >= owner)
    assert any(any(x != I(0) for x in column) for column in value.jump_first_derivative_bounds[owner:])
    assert not value.force_equilibrium_certified


def test_actual_hot_receiver_has_no_inherited_law_and_opening_is_rejected():
    from tests.test_geometric_native_common_contact_ports import _reference, _case
    reference, _, state = _actual(0)
    hot = _case(5)
    assert hot.receiver_fields.material_kind == "hot_birth"
    with pytest.raises(ValueError, match="HOT"):
        _api().evaluate_native_common_contact_constitutive(reference, hot, state, precision=96)
    with pytest.raises(ValueError, match="material frame"):
        _api().evaluate_native_common_contact_scalar_kinematics(_reference(), _case(0),
            receiver_frame="fixed_world", precision=96)
    from tectonics.geometric_native_common_contact_ports import evaluate_native_common_contact_port
    port = _case(0)
    nd = len(port.donor_fields.coordinates_m)
    coefficients = port.projected_mode_jacobian_bounds[0][nd:]
    axis = max(range(len(coefficients)), key=lambda i: max(abs(coefficients[i].lo), abs(coefficients[i].hi)))
    assert coefficients[axis].lo > 0 or coefficients[axis].hi < 0
    receiver_q = list(port.receiver_fields.coordinates_m)
    receiver_q[axis] += F(1 if coefficients[axis].lo > 0 else -1)
    opened = evaluate_native_common_contact_port(reference, port.port,
        parameter_bounds=port.geometry.parameter_bounds,
        physical_radius_cubed_bounds_m3=port.geometry.common_physical_radius_cubed_bounds_m3,
        donor_coordinates_m=port.donor_fields.coordinates_m,
        receiver_coordinates_m=tuple(receiver_q), precision=96)
    with pytest.raises(ValueError, match="opening"):
        _api().evaluate_native_common_contact_constitutive(reference, opened, state, precision=96)
