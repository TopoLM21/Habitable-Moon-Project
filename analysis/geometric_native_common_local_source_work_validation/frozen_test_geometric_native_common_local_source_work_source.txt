"""Independent polynomial work tests for the declared local source mechanism.

Genuine controls replay existing material queries and registered integrals.
The mathematical oracle differentiates a truncated polynomial motion map;
it does not import historical tests, solve equilibrium, or create quadrature.
"""
from copy import deepcopy
from dataclasses import fields, make_dataclass, replace
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json

import pytest

from tectonics.geometric_native_pose import FractionInterval as I


@lru_cache(maxsize=1)
def _api():
    from tectonics import geometric_native_common_local_source_work as api
    return api


@lru_cache(maxsize=1)
def _native():
    from analysis.geometric_native_common_local_source_work_validation import native_control
    return native_control


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def _mv(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _transpose(matrix):
    return tuple(zip(*matrix))


def _mm(a, b):
    return tuple(tuple(_dot(row, column) for column in _transpose(b)) for row in a)


def _intervals(value):
    if type(value) in (tuple, list):
        return tuple(_intervals(x) for x in value)
    return value if type(value) is I else I(F(value))


def _encloses(actual, expected):
    if type(actual) is I:
        expected = expected if type(expected) is I else I(expected)
        assert actual.lo <= expected.lo <= expected.hi <= actual.hi, (actual, expected)
    else:
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _encloses(a, b)


def _overlaps(actual, expected):
    if type(actual) is I:
        expected = expected if type(expected) is I else I(expected)
        assert max(actual.lo, expected.lo) <= min(actual.hi, expected.hi), (actual, expected)
    else:
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _overlaps(a, b)


def _nonzero(value):
    return value.lo > 0 or value.hi < 0 if type(value) is I else bool(value)


def _exact_zero(value):
    return value.lo == value.hi == 0 if type(value) is I else value == 0


# Polynomial monomials are sorted coordinate-index tuples. Terms of degree
# greater than two are discarded only after multiplication. Thus the owner
# rotation and the bilinear body/internal vector differentiate independently.
def _pc(value):
    return {} if _exact_zero(value) else {(): value}


def _pv(index):
    return {(index,): F(1)}


def _pa(*polynomials):
    result = {}
    for polynomial in polynomials:
        for powers, value in polynomial.items():
            result[powers] = result.get(powers, F(0))+value
    return {powers: value for powers, value in result.items() if not _exact_zero(value)}


def _ps(polynomial, scalar):
    return {powers: value*scalar for powers, value in polynomial.items()
        if not _exact_zero(value*scalar)}


def _pm(first, second):
    result = {}
    for p, x in first.items():
        for q, y in second.items():
            powers = tuple(sorted(p+q))
            if len(powers) <= 2:
                result[powers] = result.get(powers, F(0))+x*y
    return {powers: value for powers, value in result.items() if not _exact_zero(value)}


def _poly_cross(a, b):
    return tuple(_pa(_pm(a[j], b[k]), _ps(_pm(a[k], b[j]), -1))
        for j, k in ((1, 2), (2, 0), (0, 1)))


def _poly_rotation(vector, indices):
    skew = tuple(_pv(index) for index in indices)
    first = _poly_cross(skew, vector)
    second = _poly_cross(skew, first)
    return tuple(_pa(v, a, _ps(b, F(1, 2))) for v, a, b in zip(vector, first, second))


def _side_polynomial(arguments):
    p, X, v, modes, gradients, coordinates, radial, owner, body, internal = arguments
    arm = tuple(x-y for x, y in zip(p, X))
    r = _pa(_pc(radial), _pv(internal))
    lever = tuple(_pa(_pc(l), _ps(r, q)) for l, q in zip(arm, v))
    tip = tuple(_pa(_pc(x), _ps(r, q)) for x, q in zip(p, v))
    delta = tuple(_ps(r, q) for q in v)
    for coefficient, index, mode, gradient in zip(coordinates, body, modes, gradients):
        a = _pa(_pc(coefficient), _pv(index))
        field = tuple(_pa(_pc(mode[i]), *(_ps(lever[j], gradient[i][j]) for j in range(3)))
            for i in range(3))
        tip = tuple(_pa(x, _pm(a, field[i])) for i, x in enumerate(tip))
        delta = tuple(_pa(x, _pm(a, field[i])) for i, x in enumerate(delta))
    return _poly_rotation(tip, owner), delta


def _polynomial_scalar_twojet(sub, over, directors, count=100):
    sy, sd = _side_polynomial(sub)
    oy, od = _side_polynomial(over)
    difference = tuple(_pa(o, _ps(s, -1)) for o, s in zip(oy, sy))
    increment = tuple(_pa(o, _ps(s, -1)) for o, s in zip(od, sd))
    eta, rows, matrices = [], [], []
    for e in directors:
        frame = _poly_rotation(tuple(_pc(x) for x in e), over[7])
        scalar = _pa(*(_pm(a, b) for a, b in zip(frame, difference)))
        # The reference scalar cancels by its source expression identity.
        # Keeping the constant term as the original increment avoids treating
        # two enclosures of the same reference as independent coordinates.
        eta.append(sum((x*vector.get((), F(0)) for x, vector in zip(e, increment)), F(0)))
        rows.append(tuple(scalar.get((i,), F(0)) for i in range(count)))
        matrices.append(tuple(tuple(scalar.get(tuple(sorted((i, j))), F(0))*(2 if i == j else 1)
            for j in range(count)) for i in range(count)))
    return tuple(eta), tuple(rows), tuple(matrices)


def _toy_inputs():
    def side(offset, owner, body, internal):
        p = tuple(F(offset+3*i+1, 7) for i in range(3))
        X = tuple(F(2*offset-5*i+3, 11) for i in range(3))
        v = tuple(F(offset-i+2, 13) for i in range(3))
        modes = tuple(tuple(F((3*a+5*i+offset)%11-5, 17) for i in range(3)) for a in range(7))
        gradients = tuple(tuple(tuple(F((7*a+3*i+5*j+offset)%13-6, 29)
            for j in range(3)) for i in range(3)) for a in range(7))
        coordinates = tuple(F(a-offset, 31) for a in range(7))
        return p, X, v, modes, gradients, coordinates, F(offset-3, 19), owner, body, internal
    sub = side(2, (0, 1, 2), tuple(range(13, 20)), 92)
    over = side(5, (3, 4, 5), tuple(range(46, 53)), 97)
    directors = ((F(2, 3), F(-3, 5), F(5, 7)), (F(-1, 11), F(7, 13), F(3, 17)))
    return sub, over, directors


def _production_jet(sub, over, directors):
    side = lambda args: _api()._side_source_jet(*(
        _intervals(x) if i < 5 else x for i, x in enumerate(args)), precision=96)
    ss, os = side(sub), side(over)
    return ss, os, _api()._local_source_jet(ss, os, _intervals(directors), over[7], precision=96)


def _dense_jet(result, count=100):
    indices = result["active_coordinate_indices"]
    lookup = {index: i for i, index in enumerate(indices)}
    rows = tuple(tuple(row[lookup[i]] if i in lookup else I(0) for i in range(count))
        for row in result["scalar_first_derivative_bounds"])
    matrices = tuple(tuple(tuple(matrix[lookup[i]][lookup[j]] if i in lookup and j in lookup else I(0)
        for j in range(count)) for i in range(count)) for matrix in result["scalar_second_derivative_bounds"])
    return result["scalar_increment_bounds_m"], rows, matrices


def test_dense_polynomial_driver_twojet_matches_sparse_full_owner_body_internal_operator():
    sub, over, directors = _toy_inputs()
    ss, os, result = _production_jet(sub, over, directors)
    actual = _dense_jet(result)
    expected = _polynomial_scalar_twojet(sub, over, directors)
    _encloses(actual, expected)
    assert len(result["active_coordinate_indices"]) == 22
    assert any(expected[2][0][i][sub[9]] for i in sub[8])
    assert any(expected[2][0][i][j] for i in sub[7] for j in sub[8])
    # These reference arms contain no selected midpoint or renormalized ray.
    _encloses(ss["arm0"], tuple(p-X for p, X in zip(sub[0], sub[1])))
    _encloses(os["arm0"], tuple(p-X for p, X in zip(over[0], over[1])))


def test_common_rigid_rotation_and_exact_global_so3_covariance_keep_complete_scalar_twojets():
    sub, over, directors = _toy_inputs()
    eta, b, h = _polynomial_scalar_twojet(sub, over, directors)
    for component in range(3):
        i, j = sub[7][component], over[7][component]
        assert all(row[i]+row[j] == 0 for row in b)
        for matrix in h:
            assert matrix[i][i]+2*matrix[i][j]+matrix[j][j] == 0
            for index in sub[8]+over[8]+(sub[9], over[9]):
                assert matrix[i][index]+matrix[j][index] == 0
    # A proper rational rotation, with the owner-coordinate chart rotated too.
    Q = ((F(0), F(1), F(0)), (F(0), F(0), F(1)), (F(1), F(0), F(0)))
    def transformed(side):
        p, X, v, modes, gradients, *rest = side
        return (_mv(Q, p), _mv(Q, X), _mv(Q, v), tuple(_mv(Q, mode) for mode in modes),
            tuple(_mm(_mm(Q, G), _transpose(Q)) for G in gradients), *rest)
    _, _, transformed_jet = _production_jet(transformed(sub), transformed(over), tuple(_mv(Q, e) for e in directors))
    rotated = _dense_jet(transformed_jet)
    permutation = list(range(100))
    for indices in (sub[7], over[7]):
        permutation[indices[0]], permutation[indices[1]], permutation[indices[2]] = indices[1], indices[2], indices[0]
    _encloses(rotated[0], eta)
    _encloses(rotated[1], tuple(tuple(row[permutation[i]] for i in range(100)) for row in b))
    _encloses(rotated[2], tuple(tuple(tuple(matrix[permutation[i]][permutation[j]] for j in range(100))
        for i in range(100)) for matrix in h))


def test_finite_bilinear_path_work_and_nonzero_prestress_hessian_match_external_exact_polynomials():
    zero_mode, zero_gradient = (F(0),)*3, ((F(0),)*3,)*3
    modes = ((F(17), F(19), F(23)),)+(zero_mode,)*6
    G = tuple(tuple(F(x) for x in row) for row in ((1, 2, 3), (5, 7, 11), (13, 17, 19)))
    gradients = (G,)+(zero_gradient,)*6
    sub = ((F(2), F(3), F(5)), (F(1),)*3, (F(0), F(1), F(0)), modes,
        gradients, (F(0),)*7, F(0), (0, 1, 2), tuple(range(13, 20)), 92)
    over = ((F(7), F(11), F(13)), (F(7), F(11), F(13)), zero_mode,
        (zero_mode,)*7, (zero_gradient,)*7, (F(0),)*7, F(0), (3, 4, 5), tuple(range(46, 53)), 97)
    e = ((F(1), F(0), F(0)), (F(0), F(1), F(0)))
    outputs = []
    for t in (F(0), F(1, 3), F(2, 3), F(1)):
        current_sub = (*sub[:5], (t,)+(F(0),)*6, 2*t, *sub[7:])
        _, _, jet = _production_jet(current_sub, over, e)
        answer = _api()._atom_algebra(F(3), (F(-5), F(0)), (F(7), F(0)), F(113), jet, precision=96)
        _encloses(jet["scalar_increment_bounds_m"], (-34*t-4*t*t, -84*t-14*t*t))
        expected_energy = 113+510*t+12198*t*t+2856*t**3+168*t**4
        _encloses(answer["potential_bounds_j"], expected_energy)
        indices = {index: i for i, index in enumerate(jet["active_coordinate_indices"])}
        power = answer["potential_gradient_bounds"][indices[13]]+2*answer["potential_gradient_bounds"][indices[92]]
        _encloses(power, 510+24396*t+8568*t*t+672*t**3)
        if t == 0:
            _encloses(answer["potential_hessian_bounds"][indices[13]][indices[92]], F(30))
            _encloses(answer["prestress_curvature_hessian_bounds"][indices[13]][indices[92]], F(30))
            assert jet["scalar_first_derivative_bounds"][0][indices[92]] == I(0)
        outputs.append((answer["potential_bounds_j"], power))
    # Four-node3/8 integration is exact for this independently known cubic
    # power. Endpoint potential is quartic and retains its source E0 offset.
    work = sum((weight*power for weight, (_, power) in zip((1, 3, 3, 1), outputs)), I(0))/8
    _encloses(work, F(15732))
    _encloses(outputs[-1][0]-outputs[0][0], F(15732))


@lru_cache(maxsize=1)
def _carrier():
    return _native().material_system().source_carrier_system


@lru_cache(maxsize=1)
def _material_fields():
    return {(row.point_id, row.side): row for row in _native().original_material_responses()}


def _source_atom(point_id):
    return next(row for row in _carrier().carriers if row.point_id == point_id)


def _actual_inputs(order, coordinates, point_id):
    carrier = _source_atom(point_id)
    basis = carrier.archival_old_reference
    result = []
    for side, owner, p0, v in (("sub", carrier.sub_owner, basis.sub_position_bounds_m, basis.sub_direction_bounds),
            ("over", carrier.over_owner, basis.over_position_bounds_m, basis.over_direction_bounds)):
        field = _material_fields()[point_id, side]
        owner_indices = tuple(i for i, c in enumerate(order) if c[0] == "owner" and c[1] == owner)
        body_indices = tuple(i for i, c in enumerate(order) if c[0] == "body" and c[1] == field.geometry_field_component_id)
        residual_index = order.index(("source_internal", carrier.trace_id, side))
        result.append((p0, field.reference_position_bounds_m, v, field.mode_displacement_columns,
            field.mode_spatial_gradient_columns_per_m, tuple(coordinates[i] for i in body_indices),
            coordinates[residual_index], owner_indices, body_indices, residual_index))
    return *result, basis.receiver_director_bounds


@lru_cache(maxsize=8)
def _actual_atom_oracles(order, coordinates, branch):
    result = {}
    for carrier in _carrier().carriers:
        sub, over, directors = _actual_inputs(order, coordinates, carrier.point_id)
        eta, b, h = _polynomial_scalar_twojet(sub, over, directors)
        k = (carrier.series_stiffness_exact_pa_m[0],
            F(0) if branch == "relaxed_zero_strength_shear" else carrier.series_stiffness_exact_pa_m[1])
        t0 = carrier.literal_source_tractions_pa
        traction = tuple(I(t)+stiffness*x for t, stiffness, x in zip(t0, k, eta))
        area = carrier.original_area_m2
        e0 = sum(carrier.stored_source_energy_components_j, F(0))
        energy = I(e0)+area*sum((t*x+stiffness*x*x/2 for t, stiffness, x in zip(t0, k, eta)), I(0))
        active = set(sub[7]+sub[8]+(sub[9],)+over[7]+over[8]+(over[9],))
        gradient = tuple(area*sum((t*row[i] for t, row in zip(traction, b)), I(0))
            if i in active else F(0) for i in range(100))
        curvature = tuple(tuple(area*sum((t*matrix[i][j] for t, matrix in zip(traction, h)), I(0))
            if i in active and j in active else F(0)
            for j in range(100)) for i in range(100))
        hessian = tuple(tuple(curvature[i][j]+area*sum((stiffness*row[i]*row[j] for stiffness, row in zip(k, b)), I(0))
            if i in active and j in active else F(0)
            for j in range(100)) for i in range(100))
        result[carrier.point_id] = dict(sub=sub, over=over, directors=directors, eta=eta, b=b, h=h,
            traction=traction, k=k, area=area, e0=e0, energy=energy, gradient=gradient,
            hessian=hessian, curvature=curvature)
    return result


def _sparse_dense_atom(atom):
    return _dense_jet(dict(active_coordinate_indices=atom.active_coordinate_indices,
        scalar_increment_bounds_m=atom.scalar_increment_bounds_m,
        scalar_first_derivative_bounds=atom.scalar_first_derivative_bounds,
        scalar_second_derivative_bounds=atom.scalar_second_derivative_bounds))


def _dense_sparse_matrix(atom, field):
    values = getattr(atom, field)
    lookup = {index: i for i, index in enumerate(atom.active_coordinate_indices)}
    return tuple(tuple(values[lookup[i]][lookup[j]] if i in lookup and j in lookup else I(0)
        for j in range(100)) for i in range(100))


def test_actual_catalogue_has_one_hundred_coordinates_and_all_original_labels_at_the_registered_probe():
    system = _native().system()
    projection = _native().previous_projection()
    original, surfaces = _native().original_material_responses(), _native().surface_responses()
    assert system.coordinate_order == _native().material_system().coordinate_order
    assert len(system.coordinate_order) == len(system.bound_coordinate_increments) == 100
    assert system.coordinate_order[:92] == projection.coordinate_order
    assert system.bound_coordinate_increments[:92] == projection.bound_coordinate_increments
    assert all(type(x) is F for x in system.bound_coordinate_increments)
    assert sum(c[0] == "owner" for c in system.coordinate_order) == 6
    assert sum(c[0] == "body" for c in system.coordinate_order) == 86
    assert sum(c[0] == "source_internal" for c in system.coordinate_order) == 8
    assert len(original) == len(system.material_point_fields) == 32
    assert len({(row.point_id, row.side) for row in original}) == 32
    assert sum(row.archived for row in original) == 8
    assert len(surfaces) == 5
    for row in original:
        assert row.material_depth_bounds_m == I(0)
        assert len(row.coordinates_m) == 7 and all(row.coordinates_m)
        indices = tuple(i for i, c in enumerate(system.coordinate_order)
            if c[0] == "body" and c[1] == row.geometry_field_component_id)
        assert tuple(system.bound_coordinate_increments[i] for i in indices) == row.coordinates_m
        assert row.field_active_inside_material_clamp
        assert row.determinant_bounds.lo > 0
        assert row.original_material_membership_authenticated
        assert not row.current_live_mask_membership_authenticated
        assert not row.source_state_or_work_advanced


def test_all_sixteen_nonzero_local_atoms_match_independent_dense_driver_and_prestress_twojets():
    for arguments, response in _native().evaluation_cases():
        expected = _actual_atom_oracles(response.coordinate_order, response.coordinate_increments, response.branch)
        assert len(response.source_atom_responses) == 16
        saw_local_body_force = saw_body_internal_curvature = False
        for atom in response.source_atom_responses:
            oracle = expected[atom.point_id]
            _overlaps(_sparse_dense_atom(atom), (oracle["eta"], oracle["b"], oracle["h"]))
            _overlaps(atom.traction_bounds_pa, oracle["traction"])
            _overlaps(atom.potential_bounds_j, oracle["energy"])
            _overlaps(_dense_sparse_matrix(atom, "potential_hessian_bounds"), oracle["hessian"])
            _overlaps(_dense_sparse_matrix(atom, "prestress_curvature_hessian_bounds"), oracle["curvature"])
            assert atom.traction_bounds_pa[0].hi < 0
            assert any(_nonzero(x) for x in atom.scalar_increment_bounds_m)
            for i, index in enumerate(atom.active_coordinate_indices):
                _overlaps(atom.potential_gradient_bounds[i], oracle["gradient"][index])
                if response.coordinate_order[index][0] == "body":
                    saw_local_body_force |= _nonzero(atom.potential_gradient_bounds[i])
                    for j, other in enumerate(atom.active_coordinate_indices):
                        if response.coordinate_order[other][0] == "source_internal":
                            saw_body_internal_curvature |= _nonzero(atom.prestress_curvature_hessian_bounds[i][j])
            for actual_side, input_side in ((atom.sub, oracle["sub"]), (atom.over, oracle["over"])):
                p0, X, v, modes, gradients, coordinates, radial, *_ = input_side
                arm = tuple(p-X+radial*q for p, X, q in zip(p0, X, v))
                u = tuple(sum((a*mode[i] for a, mode in zip(coordinates, modes)), I(0)) for i in range(3))
                Fbody = tuple(tuple(I(int(i == j))+sum((a*G[i][j] for a, G in zip(coordinates, gradients)), I(0))
                    for j in range(3)) for i in range(3))
                _overlaps(actual_side.internal_arm_bounds_m, arm)
                _overlaps(actual_side.deformation_gradient_bounds, Fbody)
                _overlaps(actual_side.actual_material_position_bounds_m, _add(X, u))
                _overlaps(actual_side.virtual_radial_driver_bounds, _mv(Fbody, v))
        assert saw_local_body_force and saw_body_internal_curvature


def _independent_full_assembly(response):
    expected = _actual_atom_oracles(response.coordinate_order, response.coordinate_increments, response.branch)
    energy = sum((row["energy"] for row in expected.values()), I(0))
    gradient = [sum((row["gradient"][i] for row in expected.values()), I(0)) for i in range(100)]
    hessian = [[sum((row["hessian"][i][j] for row in expected.values()), I(0)) for j in range(100)] for i in range(100)]
    system = _native().system()
    material = _native().material_system()
    energy += material.beam_reference_energy_j
    for block in system.projection_blocks:
        beam = next(row for row in material.beams if (row.trace_id, row.side) == (block.trace_id, block.side))
        internal = response.coordinate_order.index(("source_internal", block.trace_id, block.side))
        driver = block.projection_increment_bounds_m+response.coordinate_increments[internal]
        k, w = beam.stiffness_exact_n_m, beam.reference_displacement_m
        force = k*(w+driver)
        energy += k*(w*driver+driver*driver/2)+block.orthogonal_energy_bounds_j
        gradient[internal] += force
        hessian[internal][internal] += k
        for a, i in enumerate(block.global_coordinate_indices):
            ds = block.projection_gradient_bounds[a]
            gradient[i] += force*ds+block.orthogonal_gradient_bounds_n[a]
            hessian[i][internal] += k*ds
            hessian[internal][i] += k*ds
            for b, j in enumerate(block.global_coordinate_indices):
                hessian[i][j] += (k*ds*block.projection_gradient_bounds[b]
                    +force*block.projection_hessian_bounds[a][b]+block.orthogonal_hessian_bounds_n_m[a][b])
    return energy, tuple(gradient), tuple(tuple(row) for row in hessian)


def test_global_energy_gradient_hessian_own_new_atoms_beams_and_orthogonal_remainders_once():
    for _, response in _native().evaluation_cases():
        expected = _independent_full_assembly(response)
        _overlaps((response.potential_bounds_j, response.potential_gradient_bounds, response.potential_hessian_bounds), expected)
        assert len(response.source_beam_responses) == len(_native().system().projection_blocks) == 8
        for i in response.hot_zero_coordinate_indices:
            assert response.potential_gradient_bounds[i] == I(0)
            assert all(x == I(0) for x in response.potential_hessian_bounds[i])
            assert all(row[i] == I(0) for row in response.potential_hessian_bounds)
        assert len(response.hot_zero_coordinate_indices) == 30
        for block in _native().system().projection_blocks:
            r = response.coordinate_order.index(("source_internal", block.trace_id, block.side))
            assert any(_nonzero(response.potential_hessian_bounds[i][r]) for i in block.global_coordinate_indices)


def _zeros_enclosed(value):
    if type(value) is I:
        assert value.lo <= 0 <= value.hi
    else:
        for x in value:
            _zeros_enclosed(x)


def test_point_forces_hyperstress_virtual_work_and_receiver_spin_close_action_and_moments():
    _, response = _native().evaluation_cases()[0]
    for atom in response.source_atom_responses:
        inputs = _actual_inputs(response.coordinate_order, response.coordinate_increments, atom.point_id)
        for side, force, P, arguments in ((atom.sub, atom.force_on_sub_n, atom.sub_material_hyperstress_bounds_nm, inputs[0]),
                (atom.over, atom.force_on_over_n, atom.over_material_hyperstress_bounds_nm, inputs[1])):
            arm = side.internal_arm_bounds_m
            _overlaps(P, tuple(tuple(f*x for x in arm) for f in force))
            local = {index: i for i, index in enumerate(atom.active_coordinate_indices)}
            for mode, G, index in zip(arguments[3], arguments[4], arguments[8]):
                point_work = _dot(force, mode)
                gradient_work = sum((P[i][j]*G[i][j] for i in range(3) for j in range(3)), I(0))
                _overlaps(-atom.potential_gradient_bounds[local[index]], point_work+gradient_work)
            _overlaps(-atom.potential_gradient_bounds[local[arguments[9]]], _dot(force, side.virtual_radial_driver_bounds))
            _overlaps(_add(side.actual_material_position_bounds_m, side.convected_internal_arm_bounds_m),
                side.virtual_source_position_bounds_m)
        assert atom.force_on_over_n == tuple(-x for x in atom.force_on_sub_n)
        ys, yo = atom.sub.virtual_source_position_bounds_m, atom.over.virtual_source_position_bounds_m
        sub_moment = _cross(ys, atom.force_on_sub_n)
        receiver_couple = _cross(tuple(o-s for o, s in zip(yo, ys)), atom.force_on_sub_n)
        over_moment = _add(_cross(yo, atom.force_on_over_n), receiver_couple)
        _zeros_enclosed(_add(sub_moment, over_moment))
        local = {index: i for i, index in enumerate(atom.active_coordinate_indices)}
        for arguments, moment in ((inputs[0], sub_moment), (inputs[1], over_moment)):
            _overlaps(tuple(-atom.potential_gradient_bounds[local[i]] for i in arguments[7]), moment)
        _zeros_enclosed(atom.world_moment_residual_bounds_nm)
        _zeros_enclosed(atom.virtual_work_residual_bounds)


def test_structural_body_zero_source_fourteen_subspace_preserves_finite_radials_and_old_beams():
    system, material = _native().system(), _native().material_system()
    support = tuple(next(i for i, row in enumerate(system.source_to_global_injection) if row[j] == 1)
        for j in range(14))
    assert support[:6] == tuple(range(6)) and support[6:] == tuple(range(92, 100))
    for arguments, certificate in _native().source_reference_cases():
        coordinates = arguments["source_coordinate_increments"]
        assert len(coordinates) == 14 and all(x == 0 for x in coordinates[:6])
        complete = tuple(sum((row[j]*coordinates[j] for j in range(14)), F(0))
            for row in system.source_to_global_injection)
        assert all(x == 0 for x, c in zip(complete, system.coordinate_order) if c[0] == "body")
        expected = _actual_atom_oracles(system.coordinate_order, complete, arguments["branch"])
        energy = sum((row["energy"] for row in expected.values()), I(0))+material.beam_reference_energy_j
        gradient = [sum((row["gradient"][i] for row in expected.values()), I(0)) for i in support]
        hessian = [[sum((row["hessian"][i][j] for row in expected.values()), I(0)) for j in support] for i in support]
        for beam in material.beams:
            i = beam.source_coordinate_index
            radial, k, w = coordinates[i], beam.stiffness_exact_n_m, beam.reference_displacement_m
            energy += k*(w*radial+radial*radial/2)
            gradient[i] += k*(w+radial)
            hessian[i][i] += k
        _overlaps((certificate.potential_bounds_j, certificate.gradient_bounds, certificate.hessian_bounds),
            (energy, tuple(gradient), tuple(tuple(row) for row in hessian)))
        _zeros_enclosed(certificate.source_first_variation_residual_bounds)
        _zeros_enclosed(certificate.source_second_variation_residual_bounds)
        for residuals in (certificate.atom_scalar_first_identity_residual_bounds,
                certificate.atom_scalar_second_identity_residual_bounds):
            assert len(residuals) == 16
            assert {point_id for point_id, _ in residuals} == set(expected)
            for _, residual in residuals:
                _zeros_enclosed(residual)


def test_relaxed_and_frozen_predictors_keep_literal_source_stores_and_propose_plasticity_without_advancing():
    states_before = _carrier().complete_current_source_states_json
    memory_before = _carrier().complete_current_source_memory_json
    for _, response in _native().evaluation_cases():
        relaxed = response.branch == "relaxed_zero_strength_shear"
        for atom in response.source_atom_responses:
            carrier = _source_atom(atom.point_id)
            assert atom.effective_tangent_stiffness_pa_m == (carrier.series_stiffness_exact_pa_m[0],
                F(0) if relaxed else carrier.series_stiffness_exact_pa_m[1])
            assert atom.stored_source_energy_components_j == carrier.stored_source_energy_components_j
            _overlaps(atom.scalar_free_anchor_bounds_m,
                tuple(I(a)+d for a, d in zip(carrier.stored_free_anchors_m, atom.scalar_increment_bounds_m)))
            assert atom.proposed_plastic_increment_bounds_m == (atom.scalar_increment_bounds_m[1] if relaxed else I(0))
            assert atom.source_work_lineage_id == carrier.original_source_work_lineage_id
            assert json.loads(atom.original_source_state_json) == json.loads(carrier.source_local_state_json)
        assert not response.source_state_or_work_advanced
    assert _carrier().complete_current_source_states_json == states_before
    assert _carrier().complete_current_source_memory_json == memory_before


def test_energy_owner_is_an_executable_guard_against_parallel_source_beam_and_continuous_terms():
    system = _native().system()
    for name in ("parallel_old_projected_source_energized", "separate_source_beam_energized",
            "full_common_calibrated_cold_energy_energized", "continuous_source_measure_energized"):
        for value in (True, 1, F(0)):
            with pytest.raises(ValueError):
                _api().evaluate_native_common_local_source_work(system, **{name: value})


def test_first_query_rejects_untyped_finite_owner_or_unregistered_body_coordinates():
    system = _native().system()
    valid = system.bound_coordinate_increments
    bad = [list(valid), valid[:-1]]
    for index, value in ((0, 0), (0, False), (0, F(1, 10**8)), (6, float(valid[6])),
            (6, valid[6]+F(1, 10**8)), (92, I(0)), (92, [0, 1])):
        row = list(valid)
        row[index] = value
        bad.append(tuple(row))
    for values in bad:
        with pytest.raises(ValueError):
            _api().evaluate_native_common_local_source_work(system, coordinate_increments=values)
    for precision in (True, F(96), 95):
        with pytest.raises(ValueError):
            _api().evaluate_native_common_local_source_work(system, precision=precision)


def _bind_arguments():
    return dict(projection=_native().previous_projection(),
        original_material_responses=_native().original_material_responses(),
        surface_responses=_native().surface_responses())


def test_complete_typed_dependencies_reject_json_equal_aliases_and_resealed_local_field_values():
    system, originals = _native().system(), _native().original_material_responses()
    arguments = _bind_arguments()
    aliases = [replace(originals[0], mode_displacement_columns=list(originals[0].mode_displacement_columns)),
        replace(originals[0], coordinates_m=list(originals[0].coordinates_m))]
    foreign_kind = make_dataclass("NativeCommonOriginalMaterialTransportResponse",
        [(f.name, object) for f in fields(originals[0])], frozen=True)
    aliases.append(foreign_kind(**{f.name: getattr(originals[0], f.name) for f in fields(originals[0])}))
    for alias in aliases:
        assert _api()._digest(alias) == _api()._digest(originals[0])
        changed = dict(arguments, original_material_responses=(alias,)+originals[1:])
        with pytest.raises(ValueError):
            _api().bind_native_common_local_source_work(**changed, provenance=system.provenance)
    altered = replace(originals[0], displacement_increment_bounds_m=tuple(x+F(1, 97)
        for x in originals[0].displacement_increment_bounds_m))
    altered = replace(altered, binding_sha256=_api()._seal(altered))
    with pytest.raises(ValueError):
        _api().bind_native_common_local_source_work(**dict(arguments,
            original_material_responses=(altered,)+originals[1:]), provenance=system.provenance)
    for alias in (replace(system, coordinate_order=list(system.coordinate_order)),
            replace(system, material_point_fields=list(system.material_point_fields))):
        assert _api()._digest(alias) == _api()._digest(system)
        with pytest.raises(ValueError):
            _api().evaluate_native_common_local_source_work(alias)


def _reseal_dto(raw, kind, container):
    value = raw[container] if container in raw else raw
    body = {f.name: value[f.name] for f in fields(kind) if f.name != "binding_sha256"}
    value["binding_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return raw


def test_system_and_query_restart_require_exact_complete_external_context_and_literal_budget_values():
    api, system = _api(), _native().system()
    raw = api.native_common_local_source_work_system_to_dict(system)
    restored = api.native_common_local_source_work_system_from_dict(deepcopy(raw), **_bind_arguments())
    assert api.native_common_local_source_work_system_to_dict(restored) == raw
    originals = _native().original_material_responses()
    changed = dict(_bind_arguments(), original_material_responses=(originals[1],)+originals[1:])
    with pytest.raises(ValueError):
        api.native_common_local_source_work_system_from_dict(deepcopy(raw), **changed)
    arguments, response = _native().evaluation_cases()[0]
    query = api.native_common_local_source_work_response_to_dict(response)
    kwargs = dict(system=system, expected_coordinate_increments=response.coordinate_increments,
        expected_branch=response.branch, expected_precision=response.precision)
    restored = api.native_common_local_source_work_response_from_dict(deepcopy(query), **kwargs)
    assert api.native_common_local_source_work_response_to_dict(restored) == query
    forged = deepcopy(query)
    value = forged["response"] if "response" in forged else forged
    value["potential_bounds_j"] = {"lo": [1, 1], "hi": [1, 1]}
    _reseal_dto(forged, api.NativeCommonLocalSourceWorkResponse, "response")
    with pytest.raises(ValueError):
        api.native_common_local_source_work_response_from_dict(forged, **kwargs)


def test_cached_response_and_reference_certificates_cannot_transplant_another_valid_branch_or_query():
    api, system = _api(), _native().system()
    cases = _native().evaluation_cases()
    first_arguments, first = cases[0]
    for _, other in cases[1:]:
        assert other.coordinate_increments != first.coordinate_increments or other.branch != first.branch
        raw = api.native_common_local_source_work_response_to_dict(other)
        with pytest.raises(ValueError):
            api.native_common_local_source_work_response_from_dict(raw, system=system,
                expected_coordinate_increments=first.coordinate_increments, expected_branch=first.branch,
                expected_precision=first.precision)
    reference_cases = _native().source_reference_cases()
    first_arguments, first = reference_cases[0]
    raw = api.native_common_local_source_reference_certificate_to_dict(first)
    restored = api.native_common_local_source_reference_certificate_from_dict(deepcopy(raw), system=system,
        expected_source_coordinate_increments=first_arguments["source_coordinate_increments"],
        expected_branch=first_arguments["branch"], expected_precision=96)
    assert api.native_common_local_source_reference_certificate_to_dict(restored) == raw
    assert first.cold_body_zero_is_a_symbolic_restriction_not_a_bulk_query
    assert first.source_atoms_and_beams_only
    assert not first.zero_bulk_hessian_claimed
    assert not first.original875_nonzero_anchor_operator_identity_claimed
    assert not first.full_registered_nonzero_bulk_work_identified_with_old_source
    for _, other in reference_cases[1:]:
        with pytest.raises(ValueError):
            api.native_common_local_source_reference_certificate_from_dict(
                api.native_common_local_source_reference_certificate_to_dict(other), system=system,
                expected_source_coordinate_increments=first_arguments["source_coordinate_increments"],
                expected_branch=first_arguments["branch"], expected_precision=96)
