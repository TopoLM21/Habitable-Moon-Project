"""Independent mechanics checks for the paired-carrier source handover.

The source fixtures are saved DTOs; no historical force solver, old test
fixture, or tensor quadrature is run.  Sparse polynomials independently
differentiate the declared local motion, and exact integrals check the
authentic cohesive series preparation rather than a tension spring extension.
"""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json

import pytest

from tectonics.geometric_native_pose import FractionInterval as I


@lru_cache(maxsize=1)
def _api():
    from tectonics import geometric_native_common_carrier_source_work as api
    return api


@lru_cache(maxsize=1)
def _native():
    from analysis.geometric_native_common_carrier_source_work_validation import native_control
    return native_control


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _mv(a, b):
    return tuple(_dot(row, b) for row in a)


def _transpose(a):
    return tuple(zip(*a))


def _mm(a, b):
    return tuple(tuple(_dot(row, col) for col in _transpose(b)) for row in a)


def _add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _iv(value):
    if type(value) in (tuple, list):
        return tuple(_iv(x) for x in value)
    return value if type(value) is I else I(value)


def _overlaps(a, b):
    if type(a) is I:
        b = b if type(b) is I else I(b)
        assert max(a.lo, b.lo) <= min(a.hi, b.hi), (a, b)
    else:
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _overlaps(x, y)


def _encloses(a, b):
    if type(a) is I:
        b = b if type(b) is I else I(b)
        assert a.lo <= b.lo <= b.hi <= a.hi, (a, b)
    else:
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _encloses(x, y)


def _zeros(a):
    if type(a) is I:
        assert a.lo <= 0 <= a.hi, a
    else:
        for value in a:
            _zeros(value)


def _nonzero(a):
    return a.lo > 0 or a.hi < 0 if type(a) is I else bool(a)


def _raw_fraction(value):
    assert type(value) is list and len(value) == 2 and all(type(x) is int for x in value)
    return F(*value)


def _raw_interval(value):
    assert set(value) == {'lo', 'hi'}
    return I(_raw_fraction(value['lo']), _raw_fraction(value['hi']))


def _raw_vector(value):
    return tuple(_raw_interval(x) for x in value)


def _raw_matrix(value):
    return tuple(_raw_vector(row) for row in value)


def _inverse(matrix):
    # Independent cofactor inverse; in particular B and Q are multiplied in
    # their genuine saved order before inversion, not inverted as commuting
    # matrices.  Actual narrow interval inputs enclose their exact inverses.
    columns = tuple(_cross(matrix[(i+1) % 3], matrix[(i+2) % 3]) for i in range(3))
    determinant = _dot(matrix[0], columns[0])
    return tuple(tuple(columns[j][i]/determinant for j in range(3)) for i in range(3))


@lru_cache(maxsize=1)
def _authorities():
    local, transport, source = _native().source_packets()
    fields = {(row['point_id'], row['side']): row
              for row in local['local_source_work_system']['material_point_fields']}
    material = {(row['response']['point_id'], row['response']['side']): row['response']
                for row in transport['original_material_queries']
                if row['response']['material_depth_bounds_m'] == {'lo': [0, 1], 'hi': [0, 1]}}
    carriers = {row['point_id']: row for row in source['carrier_system']['carriers']}
    assert len(fields) == len(material) == 32 and len(carriers) == 16
    assert set(fields) == set(material)
    return local, transport, source, fields, material, carriers


# Independent sparse polynomial algebra: only after multiplying do we discard
# degree greater than two.  Thus body/internal and owner/frame mixed terms
# survive; no production jet or production source scalar operator is used.
def _pc(value):
    return {(): value}


def _pv(index):
    return {(index,): F(1)}


def _pa(*values):
    result = {}
    for value in values:
        for powers, coefficient in value.items():
            result[powers] = result.get(powers, F(0))+coefficient
    return result


def _ps(value, scale):
    return {powers: scale*coefficient for powers, coefficient in value.items()}


def _pm(a, b):
    result = {}
    for p, x in a.items():
        for q, y in b.items():
            powers = tuple(sorted(p+q))
            if len(powers) <= 2:
                result[powers] = result.get(powers, F(0))+x*y
    return result


def _poly_cross(a, b):
    return tuple(_pa(_pm(a[j], b[k]), _ps(_pm(a[k], b[j]), -1))
                 for j, k in ((1, 2), (2, 0), (0, 1)))


def _rotate_polynomial(vector, indices):
    spin = tuple(_pv(index) for index in indices)
    first = _poly_cross(spin, vector)
    second = _poly_cross(spin, first)
    return tuple(_pa(v, a, _ps(b, F(1, 2))) for v, a, b in zip(vector, first, second))


def _side_polynomial(arguments, preparation):
    p0, old, chi, transport, modes, gradients, coefficients, radial, v, owners, body, internal = arguments
    identity = tuple(tuple(F(i == j) for j in range(3)) for i in range(3))
    base = tuple(x+preparation*(y-x) for x, y in zip(old, chi))
    T = tuple(tuple(x+preparation*(y-x) for x, y in zip(row, target))
              for row, target in zip(identity, transport))
    r = _pa(_pc(radial), _pv(internal))
    arm = tuple(_pa(_pc(p-x), _ps(r, direction)) for p, x, direction in zip(p0, old, v))
    point = tuple(_pc(x) for x in base)
    matrix = tuple(tuple(_pc(x) for x in row) for row in T)
    for a, index, mode, gradient in zip(coefficients, body, modes, gradients):
        driver = _pa(_pc(a), _pv(index))
        point = tuple(_pa(x, _ps(driver, m)) for x, m in zip(point, mode))
        matrix = tuple(tuple(_pa(x, _ps(driver, g)) for x, g in zip(row, grow))
                       for row, grow in zip(matrix, gradient))
    tip = tuple(_pa(point[i], *(_pm(matrix[i][j], arm[j]) for j in range(3))) for i in range(3))
    return _rotate_polynomial(tip, owners), point, matrix, arm


def _scalar_oracle(sub, over, directors, preparation):
    sy, _, _, _ = _side_polynomial(sub, preparation)
    oy, _, _, _ = _side_polynomial(over, preparation)
    difference = tuple(_pa(o, _ps(s, -1)) for s, o in zip(sy, oy))
    reference = _sub(over[0], sub[0])
    active = tuple(sorted(set(sub[9]+sub[10]+(sub[11],)+over[9]+over[10]+(over[11],))))
    scalars = []
    for director in directors:
        frame = _rotate_polynomial(tuple(_pc(x) for x in director), over[9])
        scalar = _pa(*(_pm(e, x) for e, x in zip(frame, difference)))
        eta = scalar.get((), F(0))-_dot(director, reference)
        first = tuple(scalar.get((i,), F(0)) for i in active)
        second = tuple(tuple(scalar.get(tuple(sorted((i, j))), F(0))*(2 if i == j else 1)
                             for j in active) for i in active)
        scalars.append((eta, first, second))
    return active, tuple(scalars)


def _cohesive_oracle(gap, kn=F(10**6), compliance=F(1, 500000),
                     strength=F(2*10**6), fracture=F(2*10**7)):
    k = 1/(1/kn+compliance)
    onset_contact = strength/kn
    failure = 2*fracture/strength
    onset = onset_contact+compliance*strength
    if gap < onset:
        traction, tangent = k*gap, k
        potential = k*gap*gap/2
        damage = F(0)
    elif gap < failure:
        traction = strength*(failure-gap)/(failure-onset)
        tangent = -strength/(failure-onset)
        length = gap-onset
        potential = strength*onset/2+strength*(length-length*length/(2*(failure-onset)))
        contact = gap-compliance*traction
        damage = failure*(contact-onset_contact)/(contact*(failure-onset_contact))
    else:
        traction, tangent, potential, damage = F(0), F(0), fracture, F(1)
    contact = gap-compliance*traction
    recoverable = traction*gap/2
    dissipated = potential-recoverable
    return dict(traction=traction, tangent=tangent, potential=potential,
                contact=contact, recoverable=recoverable, fracture=dissipated,
                damage=damage, maximum=max(F(0), contact), onset=onset, failure=failure)


def _loading(gap):
    return _api().evaluate_native_cohesive_series_loading(I(gap),
        contact_stiffness_pa_m=F(10**6), column_compliance_m_pa=F(1, 500000),
        tensile_strength_pa=F(2*10**6), fracture_energy_j_m2=F(2*10**7), precision=96)


def test_authentic_series_cohesive_branches_have_independent_energy_and_damage_balance():
    for gap in (F(-1, 100), F(0), F(3), F(7), F(13), F(19), F(21), F(10000)):
        value, oracle = _loading(gap), _cohesive_oracle(gap)
        assert value.onset_free_gap_m == oracle['onset'] == 6
        assert value.failure_free_gap_m == oracle['failure'] == 20
        for name, key in (
            ('traction_bounds_pa', 'traction'), ('tangent_bounds_pa_m', 'tangent'),
            ('contact_gap_bounds_m', 'contact'), ('loading_potential_density_bounds_j_m2', 'potential'),
            ('recoverable_energy_density_bounds_j_m2', 'recoverable'),
            ('fracture_work_density_bounds_j_m2', 'fracture'),
            ('proposed_damage_bounds', 'damage'), ('proposed_max_opening_bounds_m', 'maximum')):
            _encloses(getattr(value, name), oracle[key])
        _zeros(value.loading_potential_density_bounds_j_m2
               -value.recoverable_energy_density_bounds_j_m2-value.fracture_work_density_bounds_j_m2)
        assert value.fracture_work_density_bounds_j_m2.lo >= 0
    assert _loading(F(13)).tangent_bounds_pa_m.hi < 0
    assert _loading(F(10000)).traction_bounds_pa == I(0)
    assert _loading(F(10000)).proposed_damage_bounds == I(1)


def test_cohesive_loading_work_crosses_real_onset_and_failure_without_tension_extension():
    # Integrate the three affine traction pieces, independently of the
    # production loading potential.  This path crosses both physical events.
    g0, g1, k, strength = F(-1, 100), F(10000), F(10**6, 3), F(2*10**6)
    work = k*(F(6)**2-g0*g0)/2+strength*(F(20)-F(6))/2
    before, after = _loading(g0), _loading(g1)
    _encloses(after.loading_potential_density_bounds_j_m2
              -before.loading_potential_density_bounds_j_m2, work)
    assert work < F(2*10**7)
    _encloses(after.fracture_work_density_bounds_j_m2, F(2*10**7))
    assert after.recoverable_energy_density_bounds_j_m2 == I(0)
    # The old compression quadratic extended to 10km would differ by orders
    # of magnitude; the actual fully failed primitive is constant Gamma.
    assert after.loading_potential_density_bounds_j_m2.hi < k*g1*g1/1000


def test_cohesive_branch_corners_or_snapback_inputs_are_rejected_for_a_twojet():
    api = _api()
    arguments = dict(contact_stiffness_pa_m=F(10**6),
        column_compliance_m_pa=F(1, 500000), tensile_strength_pa=F(2*10**6),
        fracture_energy_j_m2=F(2*10**7), precision=96)
    for gap in (I(6), I(20), I(5, 7), I(19, 21)):
        with pytest.raises(ValueError):
            api.evaluate_native_cohesive_series_loading(gap, **arguments)
    for key, value in (('contact_stiffness_pa_m', 10**6),
                       ('column_compliance_m_pa', F(1, 10000)),
                       ('fracture_energy_j_m2', F(1))):
        with pytest.raises(ValueError):
            api.evaluate_native_cohesive_series_loading(I(13), **dict(arguments, **{key: value}))


def _toy_sides():
    sub = (_iv((3, -2, 5)), _iv((1, 1, 2)), _iv((2, -1, 4)),
        _iv(((1, F(1, 5), 0), (0, 1, F(1, 7)), (F(1, 11), 0, 1))),
        _iv(((F(2, 7), F(-1, 5), F(1, 3)),)),
        _iv((((F(1, 13), F(1, 17), 0), (0, F(-1, 19), F(1, 23)), (F(1, 29), 0, F(1, 31))),)),
        (F(2, 9),), F(1, 11), _iv((F(2, 3), F(-1, 3), F(2, 3))), (0, 1, 2), (6,), 8)
    over = (_iv((-2, 3, 7)), _iv((-1, 1, 2)), _iv((-1, 4, 6)),
        _iv(((1, 0, F(1, 7)), (F(1, 5), 1, 0), (0, F(-1, 11), 1))),
        _iv(((F(-1, 5), F(2, 9), F(1, 7)),)),
        _iv((((F(-1, 17), 0, F(1, 19)), (F(1, 23), F(1, 29), 0), (0, F(1, 31), F(1, 37))),)),
        (F(-1, 7),), F(-1, 13), _iv((F(1, 3), F(2, 3), F(2, 3))), (3, 4, 5), (7,), 9)
    return sub, over


def test_actual_carrier_side_motion_twojet_matches_independent_joint_polynomial():
    preparation = F(2, 5)
    sub, over = _toy_sides()
    for arguments in (sub, over):
        actual = _api()._carrier_side_jet(*arguments, preparation_fraction=preparation, precision=96)
        poly, point, matrix, arm = _side_polynomial(arguments, preparation)
        _encloses(actual['x'], tuple(x[()] for x in point))
        _encloses(actual['y'], tuple(x[()] for x in poly))
        _encloses(actual['F'], tuple(tuple(x[()] for x in row) for row in matrix))
        _encloses(actual['arm'], tuple(x[()] for x in arm))
        for index in arguments[9]+arguments[10]+(arguments[11],):
            _encloses(actual['first'][index], tuple(x.get((index,), F(0)) for x in poly))
            for other in arguments[9]+arguments[10]+(arguments[11],):
                expected = tuple(x.get(tuple(sorted((index, other))), F(0))*(2 if index == other else 1)
                                 for x in poly)
                _encloses(actual['second'].get((index, other), (I(0),)*3), expected)
        _encloses(actual['second'][arguments[10][0], arguments[11]],
                  _mv(arguments[5][0], arguments[8]))
        assert any(_nonzero(x) for x in actual['second'][arguments[10][0], arguments[11]])


def test_local_material_and_microdirector_twojets_are_objective_under_common_rotation():
    api, preparation = _api(), F(2, 5)
    sub, over = _toy_sides()
    directors = _iv(((0, 0, 1), (1, 0, 0)))
    Q = _iv(((F(3, 5), F(-4, 5), 0), (F(4, 5), F(3, 5), 0), (0, 0, 1)))

    def scalar(inputs, frames):
        sides = tuple(api._carrier_side_jet(*row, preparation_fraction=preparation, precision=96)
                      for row in inputs)
        return api._local_source_jet(*sides, frames, inputs[1][9], precision=96)

    def rotated(row):
        p, old, chi, T, modes, gradients, a, r, v, owners, body, internal = row
        return (_mv(Q, p), _mv(Q, old), _mv(Q, chi), _mm(_mm(Q, T), _transpose(Q)),
            tuple(_mv(Q, m) for m in modes), tuple(_mm(_mm(Q, g), _transpose(Q)) for g in gradients),
            a, r, _mv(Q, v), owners, body, internal)

    first, second = scalar((sub, over), directors), scalar((rotated(sub), rotated(over)),
        tuple(_mv(Q, e) for e in directors))
    _overlaps(first['scalar_increment_bounds_m'], second['scalar_increment_bounds_m'])
    # The owner variations rotate as physical spin vectors; scalar body and
    # material internal parameters retain their original coordinate values.
    directions = tuple(F(i+1, 37) for i in range(10))
    rotated_directions = list(directions)
    for owners in (sub[9], over[9]):
        spin = _mv(Q, tuple(I(directions[i]) for i in owners))
        for index, value in zip(owners, spin):
            rotated_directions[index] = value
    for a in range(2):
        _overlaps(_dot(first['scalar_first_derivative_bounds'][a], directions),
                  _dot(second['scalar_first_derivative_bounds'][a], rotated_directions))
        h1 = _dot(directions, _mv(first['scalar_second_derivative_bounds'][a], directions))
        h2 = _dot(rotated_directions, _mv(second['scalar_second_derivative_bounds'][a], rotated_directions))
        _overlaps(h1, h2)


@lru_cache(maxsize=8)
def _response(preparation=F(1)):
    return _native().response(preparation)


def _softening_preparation():
    return dict(_native().preparation_probes())['softening_preparation']


@lru_cache(maxsize=1)
def _physical_points():
    from tectonics.geometric_native_archive_material_carrier import evaluate_native_archive_material_point
    _, _, _, _, material, _ = _authorities()
    return {key: evaluate_native_archive_material_point(_native().archive_carrier(),
        row['original_parent_id'], tuple(_raw_fraction(x) for x in row['original_direction_literal']),
        I(0), F(1), precision=96) for key, row in material.items()}


def _genuine_inputs(response, point_id):
    _, _, _, fields, material, _ = _authorities()
    inputs = []
    for side in ('sub', 'over'):
        key = point_id, side
        field, old = fields[key], material[key]
        point = _physical_points()[key]
        original = _mm(_raw_matrix(old['reference_gradient_bounds']),
                       _raw_matrix(old['material_owner_pose_bounds']))
        transport = _mm(point.deformation_gradient_bounds, _inverse(original))
        indices = tuple(field['body_coordinate_indices'])
        inputs.append((_raw_vector(field['source_reference_position_bounds_m']),
            _raw_vector(field['material_reference_position_bounds_m']), point.position_bounds_m, transport,
            tuple(_raw_vector(x) for x in field['mode_displacement_columns']),
            tuple(_raw_matrix(x) for x in field['mode_spatial_gradient_columns_per_m']),
            tuple(response.coordinate_increments[i] for i in indices),
            response.coordinate_increments[field['residual_coordinate_index']],
            _raw_vector(field['source_virtual_radial_direction_bounds']),
            tuple(field['owner_coordinate_indices']), indices, field['residual_coordinate_index']))
    return tuple(inputs)


def test_all_32_genuine_material_labels_keep_archive_ancestry_and_correct_old_frame_pullback():
    _, _, _, fields, material, _ = _authorities()
    response = _response()
    seen, archived = set(), set()
    for atom in response.source_atom_responses:
        for side in (atom.sub, atom.over):
            key = atom.point_id, side.side
            assert key not in seen
            seen.add(key)
            field, old, physical = fields[key], material[key], _physical_points()[key]
            assert side.original_parent_id == field['original_parent_id'] == old['original_parent_id']
            assert side.original_material_id == field['original_material_id'] == old['original_material_id']
            assert side.owner == field['owner'] == physical.owner
            assert side.archive_cohort_id == field['archive_cohort_id'] == old['archive_cohort_id']
            assert side.original_label_membership == field['original_label_membership']
            assert side.original_ray == tuple(_raw_fraction(x) for x in old['original_direction_literal'])
            assert side.material_depth_bounds_m == I(0)
            if side.archive_cohort_id is not None:
                archived.add(key)
                assert side.side == 'sub' and physical.allocation == 'archived'
                assert physical.archive_debits == 1 and physical.cohort_id == side.archive_cohort_id
            _overlaps(side.carrier_reference_position_bounds_m, physical.position_bounds_m)
            assert side.old_material_reference_position_bounds_m == _raw_vector(
                field['material_reference_position_bounds_m'])
            original = _mm(_raw_matrix(old['reference_gradient_bounds']),
                           _raw_matrix(old['material_owner_pose_bounds']))
            _overlaps(side.old_original_reference_gradient_bounds, original)
            _overlaps(_mm(side.carrier_reference_transport_bounds, original),
                      physical.deformation_gradient_bounds)
            _zeros(side.frame_pullback_identity_residual_bounds)
            combined = _mm(side.deformation_gradient_bounds, original)
            _overlaps(side.total_original_material_deformation_gradient_bounds, combined)
            _overlaps(side.local_material_volume_jacobian_bounds,
                      _dot(combined[0], _cross(combined[1], combined[2])))
            assert side.local_material_volume_jacobian_bounds.lo > 0
            modes = tuple(_raw_vector(x) for x in field['mode_displacement_columns'])
            coordinates = tuple(response.coordinate_increments[i] for i in field['body_coordinate_indices'])
            displacement = tuple(sum((a*mode[i] for a, mode in zip(coordinates, modes)), I(0))
                                 for i in range(3))
            _overlaps(side.actual_material_position_bounds_m,
                      _add(physical.position_bounds_m, displacement))
    assert seen == set(fields) and len(archived) == 8
    assert len({fields[key]['original_parent_id'] for key in archived}) >= 2
    assert not response.whole_support_nonzero_common_material_admission_certified
    assert not response.current_follower_contact_law_defined


def test_noncommuting_old_material_frame_is_pulled_back_in_its_complete_order():
    # The old COMMON B is an actual shear, and Q is a proper nonidentity
    # rotation.  Dropping Q or commuting it with B would fail this identity.
    B = _iv(((1, F(1, 3), 0), (0, 1, 0), (0, 0, 1)))
    Q = _iv(((F(3, 5), F(-4, 5), 0), (F(4, 5), F(3, 5), 0), (0, 0, 1)))
    A = _iv(((F(7, 6), 0, F(1, 11)), (F(1, 13), F(6, 7), 0), (0, F(1, 17), 1)))
    expected = _mm(A, _inverse(_mm(B, Q)))
    actual = _api()._material_transport_pullback(A, B, Q, precision=96)
    _encloses(actual, expected)
    _encloses(_mm(actual, _mm(B, Q)), A)
    assert any(_nonzero(a-b) for row, other in zip(expected, _mm(A, _inverse(B)))
               for a, b in zip(row, other))


def test_genuine_joint_twojets_retain_prestress_and_body_internal_terms_on_softening():
    _, _, _, _, _, carriers = _authorities()
    saw_prestress, saw_mixed = False, False
    for preparation in (F(0), _softening_preparation()):
        response = _response(preparation)
        for atom in response.source_atom_responses:
            inputs = _genuine_inputs(response, atom.point_id)
            source = carriers[atom.point_id]
            directors = tuple(_raw_vector(x) for x in source['archival_old_reference']['receiver_director_bounds'])
            active, scalar = _scalar_oracle(*inputs, directors, preparation)
            assert atom.active_coordinate_indices == active
            for index, (eta, first, second) in enumerate(scalar):
                _overlaps(atom.scalar_increment_bounds_m[index], eta)
                _overlaps(atom.scalar_first_derivative_bounds[index], first)
                _overlaps(atom.scalar_second_derivative_bounds[index], second)
            eta, first, second = scalar[0]
            area = _raw_fraction(source['original_area_m2'])
            traction, tangent = atom.loading_response.traction_bounds_pa, atom.loading_response.tangent_bounds_pa_m
            gradient = tuple(area*traction*b for b in first)
            curvature = tuple(tuple(area*traction*h for h in row) for row in second)
            hessian = tuple(tuple(area*(tangent*first[i]*first[j]+traction*second[i][j])
                                   for j in range(len(active))) for i in range(len(active)))
            _overlaps(atom.potential_gradient_bounds, gradient)
            _overlaps(atom.potential_hessian_bounds, hessian)
            _overlaps(atom.prestress_curvature_hessian_bounds, curvature)
            saw_prestress |= any(_nonzero(x) for row in curvature for x in row)
            for side in inputs:
                for i in side[10]:
                    saw_mixed |= _nonzero(curvature[active.index(i)][active.index(side[11])])
        if preparation:
            assert all(a.loading_response.branch == 'monotone_softening' for a in response.source_atom_responses)
    assert saw_prestress and saw_mixed


def test_material_point_arm_and_director_force_work_close_with_actual_cohesive_forces():
    response = _response(_softening_preparation())
    for atom in response.source_atom_responses:
        inputs = _genuine_inputs(response, atom.point_id)
        local = {index: i for i, index in enumerate(atom.active_coordinate_indices)}
        for side, force, stress, arguments in ((atom.sub, atom.force_on_sub_n,
                atom.sub_material_hyperstress_bounds_nm, inputs[0]),
                (atom.over, atom.force_on_over_n, atom.over_material_hyperstress_bounds_nm, inputs[1])):
            _overlaps(stress, tuple(tuple(f*l for l in side.internal_arm_bounds_m) for f in force))
            _overlaps(_add(side.actual_material_position_bounds_m, side.convected_internal_arm_bounds_m),
                      side.virtual_source_position_bounds_m)
            for mode, gradient, index in zip(arguments[4], arguments[5], arguments[10]):
                physical_work = _dot(force, mode)+sum((stress[i][j]*gradient[i][j]
                                                     for i in range(3) for j in range(3)), I(0))
                _overlaps(-atom.potential_gradient_bounds[local[index]], physical_work)
            _overlaps(-atom.potential_gradient_bounds[local[arguments[11]]],
                      _dot(force, side.virtual_radial_driver_bounds))
        assert any(_nonzero(x) for x in atom.force_on_sub_n)
        assert atom.force_on_over_n == tuple(-x for x in atom.force_on_sub_n)
        _zeros(atom.action_reaction_residual_bounds_n)
        _zeros(atom.world_moment_residual_bounds_nm)
        _zeros(atom.virtual_work_residual_bounds)
        sub_torque = _cross(atom.sub.virtual_source_position_bounds_m, atom.force_on_sub_n)
        over_torque = _add(_cross(atom.over.virtual_source_position_bounds_m, atom.force_on_over_n),
            _cross(_sub(atom.over.virtual_source_position_bounds_m,
                        atom.sub.virtual_source_position_bounds_m), atom.force_on_sub_n))
        _overlaps(atom.complete_attachment_torque_on_sub_nm, sub_torque)
        _overlaps(atom.complete_attachment_torque_on_over_nm, over_torque)
        _zeros(_add(sub_torque, over_torque))


def test_zero_preparation_preserves_saved_relaxed_source_operator_without_replaying_old_physics():
    local, _, _, _, _, _ = _authorities()
    old = local['evaluation_queries'][0]
    assert old['arguments']['branch'] == 'relaxed_zero_strength_shear'
    value, saved = _response(F(0)), old['response']
    assert value.coordinate_increments == tuple(_raw_fraction(x) for x in old['arguments']['coordinate_increments'])
    _overlaps(value.loading_potential_bounds_j, _raw_interval(saved['potential_bounds_j']))
    _overlaps(value.potential_gradient_bounds, _raw_vector(saved['potential_gradient_bounds']))
    _overlaps(value.potential_hessian_bounds, _raw_matrix(saved['potential_hessian_bounds']))
    for atom in value.source_atom_responses:
        assert atom.loading_response.branch == 'intact_or_compressed'
        assert atom.loading_response.traction_bounds_pa.hi < 0
        _zeros(atom.preparation_potential_work_bounds_j)
        assert atom.proposed_fracture_work_bounds_j == I(0)


def test_real_upper_reference_preparation_retains_proposed_fracture_and_exact_reaction_work():
    _, _, _, _, _, carriers = _authorities()
    value, before = _response(), _response(F(0))
    expected_work, expected_fracture = I(0), F(0)
    for atom in value.source_atom_responses:
        source = carriers[atom.point_id]
        law = json.loads(source['complete_original_atom_history_json'])['original_material']['law']
        area, fracture = _raw_fraction(source['original_area_m2']), F.from_float(law['fracture_energy_j_m2'])
        assert atom.loading_response.branch == 'fully_failed_open'
        assert atom.current_free_gap_bounds_m.lo > 20
        assert atom.before_free_gap_bounds_m.hi < 0
        assert atom.loading_response.proposed_damage_bounds == I(1)
        assert atom.loading_response.traction_bounds_pa == I(0)
        assert atom.loading_response.tangent_bounds_pa_m == I(0)
        assert atom.force_on_sub_n == atom.force_on_over_n == (I(0),)*3
        assert all(x == I(0) for x in atom.potential_gradient_bounds)
        assert all(x == I(0) for row in atom.potential_hessian_bounds for x in row)
        _encloses(atom.proposed_fracture_work_bounds_j, area*fracture)
        k = _raw_fraction(source['series_stiffness_exact_pa_m'][0])
        # Exact traction integral: compression/intact K*g up to6, then the
        # genuine descending triangle6..20.  After failure it contributes0.
        initial = k*atom.before_free_gap_bounds_m*atom.before_free_gap_bounds_m/2
        work = area*(fracture-initial)
        _overlaps(atom.preparation_potential_work_bounds_j, work)
        _overlaps(atom.preparation_physical_reaction_work_bounds_j, -work)
        _zeros(atom.preparation_energy_balance_residual_bounds_j)
        _zeros(atom.preparation_potential_work_bounds_j
               -atom.preparation_recoverable_energy_change_bounds_j
               -atom.preparation_fracture_work_increment_bounds_j)
        assert atom.original_source_state_json == source['source_local_state_json']
        expected_work += work
        expected_fracture += area*fracture
    _overlaps(value.preparation_potential_work_bounds_j, expected_work)
    _encloses(value.proposed_fracture_work_bounds_j, expected_fracture)
    _overlaps(value.loading_potential_bounds_j-before.loading_potential_bounds_j, expected_work)
    assert value.preparation_potential_work_bounds_j.lo > 0
    assert not value.source_state_or_work_advanced
    assert value.accepted_new_native_intervals == 0
    assert not value.new_mixed_equilibrium_certified
    assert (value.source_contact_elapsed_time_s, value.source_memory_elapsed_time_s,
            value.source_geometry_elapsed_time_s, value.material_geometry_elapsed_time_s) == (
                F(2000000), F(1000000), F(0), F(1000000))


def test_failed_endpoint_has_one_atom_beam_and_registered_remainder_energy_owner():
    local_packet, transport, _, _, _, carriers = _authorities()
    local, value = local_packet['local_source_work_system'], _response()
    energy = I(_raw_fraction(local['source_beam_literal_reference_energy_j']))
    source_energy = energy
    gradient, hessian = [I(0)]*100, [[I(0) for _ in range(100)] for _ in range(100)]
    for source in carriers.values():
        law = json.loads(source['complete_original_atom_history_json'])['original_material']['law']
        area, fracture = _raw_fraction(source['original_area_m2']), F.from_float(law['fracture_energy_j_m2'])
        t, k = _raw_fraction(source['literal_source_tractions_pa'][0]), _raw_fraction(source['series_stiffness_exact_pa_m'][0])
        literal = sum((_raw_fraction(x) for x in source['stored_source_energy_components_j']), F(0))
        source_energy += literal-area*t*t/(2*k)+area*fracture
    beams = {(x['trace_id'], x['side']): x for x in transport['material_transfer']['beams']}
    remainder = I(0)
    for block in local['projection_blocks']:
        beam = beams[block['trace_id'], block['side']]
        internal = value.coordinate_order.index(('source_internal', block['trace_id'], block['side']))
        body = tuple(block['global_coordinate_indices'])
        driver = _raw_interval(block['projection_increment_bounds_m'])+value.coordinate_increments[internal]
        k, w = _raw_fraction(beam['stiffness_exact_n_m']), _raw_fraction(beam['reference_displacement_m'])
        source_energy += k*(w*driver+driver*driver/2)
        remainder += _raw_interval(block['orthogonal_energy_bounds_j'])
        traction = k*(w+driver)
        first = _raw_vector(block['projection_gradient_bounds'])
        second = _raw_matrix(block['projection_hessian_bounds'])
        orthogonal_g = _raw_vector(block['orthogonal_gradient_bounds_n'])
        orthogonal_h = _raw_matrix(block['orthogonal_hessian_bounds_n_m'])
        gradient[internal] += traction
        hessian[internal][internal] += k
        for a, i in enumerate(body):
            gradient[i] += traction*first[a]+orthogonal_g[a]
            hessian[i][internal] += k*first[a]
            hessian[internal][i] += k*first[a]
            for b, j in enumerate(body):
                hessian[i][j] += k*first[a]*first[b]+traction*second[a][b]+orthogonal_h[a][b]
    _overlaps(value.source_loading_potential_bounds_j, source_energy)
    _overlaps(value.registered_orthogonal_energy_bounds_j, remainder)
    _overlaps(value.loading_potential_bounds_j, source_energy+remainder)
    _overlaps(value.potential_gradient_bounds, tuple(gradient))
    _overlaps(value.potential_hessian_bounds, tuple(tuple(row) for row in hessian))
    _zeros(value.loading_potential_bounds_j-value.recoverable_energy_bounds_j-value.proposed_fracture_work_bounds_j)
    assert len(value.source_atom_responses) == 16 and len(value.source_beam_responses) == 8
    assert len(local['projection_blocks']) == 8
    hot = tuple(local['hot_zero_coordinate_indices'])
    assert len(hot) == 30
    for i in hot:
        assert value.potential_gradient_bounds[i] == I(0)
        assert all(x == I(0) for x in value.potential_hessian_bounds[i])
        assert all(row[i] == I(0) for row in value.potential_hessian_bounds)
    assert not value.hot_constitutive_law_defined


def test_new_virgin_tensile_seed_is_explicit_and_cannot_replace_saved_source_memory():
    api, system = _api(), _native().system()
    assert not system.old_normal_fracture_memory_recovered
    assert system.tensile_history_seed == 'NEW_DECLARED_VIRGIN_MODE_I_PREPARATION_NOT_RECOVERED_OLD_FRACTURE_MEMORY'
    seed = json.loads(system.seed_json)
    local, _, _, _, _, _ = _authorities()
    old = local['local_source_work_system']
    for name in ('complete_current_source_states_json', 'complete_current_source_memory_json',
                 'complete_before_work_ledgers_json', 'complete_unaccepted_candidate_work_ledgers_json'):
        assert seed['local'][name] == old[name]
    for key, value in (('initial_max_opening_m', F(1)), ('initial_damage', F(1, 2)),
                       ('initial_max_opening_m', 0), ('initial_damage', False)):
        with pytest.raises(ValueError):
            api.bind_native_common_carrier_source_work(*_native().source_packets(), _native().archive_carrier(),
                provenance=system.provenance, **{key: value})


def test_queries_keep_registered_body_values_and_reject_unadmitted_branch_or_clock_contexts():
    api, system = _api(), _native().system()
    valid = system.bound_coordinate_increments
    for index, altered in ((0, 0), (0, F(1, 10**8)), (6, valid[6]+F(1, 10**8)),
                           (92, I(0))):
        q = list(valid)
        q[index] = altered
        with pytest.raises(ValueError):
            api.evaluate_native_common_carrier_source_work(system, coordinate_increments=tuple(q))
    for arguments in ({'coordinate_increments': list(valid)},
                      {'coordinate_increments': valid[:-1]},
                      {'time_fraction': F(0)}, {'time_fraction': F(1, 2)},
                      {'time_fraction': F(2)}, {'time_fraction': True},
                      {'preparation_fraction': F(-1, 100)}, {'preparation_fraction': 0},
                      {'branch': 'frozen_elastic_predictor'}, {'precision': True}):
        with pytest.raises(ValueError):
            api.evaluate_native_common_carrier_source_work(system, **arguments)


def _reseal_response(raw):
    body = {key: value for key, value in raw.items() if key not in ('format', 'binding_sha256')}
    raw['binding_sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return raw


def test_joint_restart_rejects_transplanted_queries_resealed_ancestry_and_proposed_history():
    api, system = _api(), _native().system()
    local, transport, source = _native().source_packets()
    contexts = dict(local_source_packet=local, material_transport_packet=transport,
                    source_carrier_packet=source, archive_carrier=_native().archive_carrier())
    raw_system = api.native_common_carrier_source_work_system_to_dict(system)
    restored = api.native_common_carrier_source_work_system_from_dict(deepcopy(raw_system), **contexts)
    assert api.native_common_carrier_source_work_system_to_dict(restored) == raw_system
    raw = api.native_common_carrier_source_work_response_to_dict(_response())
    expected = dict(system=restored, expected_time_fraction=F(1), expected_preparation_fraction=F(1),
                    expected_coordinate_increments=system.bound_coordinate_increments, precision=96)
    rebuilt = api.native_common_carrier_source_work_response_from_dict(deepcopy(raw), **expected)
    assert api.native_common_carrier_source_work_response_to_dict(rebuilt) == raw
    other = api.native_common_carrier_source_work_response_to_dict(_response(F(0)))
    with pytest.raises(ValueError):
        api.native_common_carrier_source_work_response_from_dict(other, **expected)
    for tamper in ('ancestry', 'fracture', 'clock', 'seed'):
        changed = deepcopy(raw)
        if tamper == 'ancestry':
            changed['source_atom_responses'][0]['sub']['original_parent_id'] = 'octant:0'
        elif tamper == 'fracture':
            changed['source_atom_responses'][0]['loading_response']['proposed_damage_bounds'] = {
                'lo': [0, 1], 'hi': [0, 1]}
        elif tamper == 'clock':
            changed['source_contact_elapsed_time_s'] = [2000001, 1]
        else:
            changed['old_normal_fracture_memory_recovered'] = True
        _reseal_response(changed)
        with pytest.raises(ValueError):
            api.native_common_carrier_source_work_response_from_dict(changed, **expected)
    for field in ('coordinate_increments', 'source_atom_responses'):
        changed = replace(_response(), **{field: list(getattr(_response(), field))})
        with pytest.raises(ValueError):
            api.native_common_carrier_source_work_response_to_dict(changed)
