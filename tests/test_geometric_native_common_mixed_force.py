"""Independent mixed force tests: nonlinear geometry, exact work and sources.

Small analytic potentials distinguish finite beam increments, unit Nanson
normal curvature and prestressed contact curvature from their linear parts.
Authentic controls retain the saved complete operators and source histories;
the new owner derivatives do not authorize a moved physical endpoint.
"""
from dataclasses import replace
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json

import pytest


ZERO = F(0)
IDENTITY = tuple(tuple(F(int(i == j)) for j in range(3)) for i in range(3))
ZERO_MATRIX = ((ZERO, ZERO, ZERO),)*3


@lru_cache(maxsize=1)
def _api():
    from tectonics import geometric_native_common_mixed_force as api
    return api


@lru_cache(maxsize=1)
def _native():
    from analysis.geometric_native_common_mixed_force_validation import native_control
    return native_control


def _I(first, second=None):
    from tectonics.geometric_native_pose import FractionInterval
    return FractionInterval(first) if second is None else FractionInterval(first, second)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _fraction(raw):
    assert type(raw) is list and len(raw) == 2 and all(type(x) is int for x in raw)
    value = F(*raw)
    assert [value.numerator, value.denominator] == raw
    return value


def _decode(value):
    if type(value) is dict and set(value) == {"lo", "hi"}:
        return _I(_fraction(value["lo"]), _fraction(value["hi"]))
    if type(value) is list:
        return tuple(_decode(x) for x in value)
    return value


def _contains(actual, expected):
    if hasattr(actual, "lo"):
        assert actual.lo <= expected <= actual.hi, (actual, expected)
    else:
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            _contains(first, second)


def _overlap(actual, expected):
    expected = expected if hasattr(expected, "lo") else _I(expected)
    assert max(actual.lo, expected.lo) <= min(actual.hi, expected.hi), (actual, expected)


def _matrix_interval(matrix):
    return tuple(tuple(_I(x) for x in row) for row in matrix)


def _quadratic_projection(coordinates):
    linear = (F(2, 3), F(-5, 7))
    curvature = ((F(3, 11), F(-2, 13)), (F(-2, 13), F(7, 17)))
    value = F(5, 19)+sum((a*x for a, x in zip(linear, coordinates)), ZERO)+sum(
        (curvature[i][j]*coordinates[i]*coordinates[j]/2 for i in range(2) for j in range(2)), ZERO)
    gradient = tuple(linear[i]+sum((curvature[i][j]*coordinates[j] for j in range(2)), ZERO)
                     for i in range(2))
    return value, gradient, curvature


def _beam_difference(coordinates):
    cold, residual = coordinates[:2], coordinates[2]
    p, _, _ = _quadratic_projection(cold)
    force, stiffness = F(-17, 23), F(29, 31)
    # Difference of the same literal finite beam energy at p+r and p.
    def energy(argument):
        return F(101, 37)+force*argument+stiffness*argument*argument/2
    return energy(p+residual)-energy(p)


def _polynomial_twojet(potential, coordinates, step=F(1, 37)):
    """Independent exact differences for a polynomial of total degree <=4.

    Richardson cancellation removes the sole central-difference step-squared
    term, including diagonal and mixed second variations.
    """
    count = len(coordinates)
    origin = potential(coordinates)
    def displaced(changes):
        point = list(coordinates)
        for index, change in changes.items():
            point[index] += change
        return potential(tuple(point))
    gradient, hessian = [], []
    for i in range(count):
        def first(h):
            return (displaced({i: h})-displaced({i: -h}))/(2*h)
        gradient.append((4*first(step/2)-first(step))/3)
        row = []
        for j in range(count):
            def second(h):
                if i == j:
                    return (displaced({i: h})-2*origin+displaced({i: -h}))/(h*h)
                return sum((a*b*displaced({i: a*h, j: b*h})
                            for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))), ZERO)/(4*h*h)
            row.append((4*second(step/2)-second(step))/3)
        hessian.append(tuple(row))
    return origin, tuple(gradient), tuple(hessian)


def _beam_actual(coordinates):
    p, dp, d2p = _quadratic_projection(coordinates[:2])
    return _api()._internal_beam_correction_twojet(_I(p), tuple(_I(x) for x in dp),
        _matrix_interval(d2p), F(-17, 23), F(29, 31), coordinates[2], precision=96)


def _beam_full_jet(actual):
    gradient = actual["cold_gradient"]+(actual["internal_gradient"],)
    hessian = tuple(tuple(actual["cold_hessian"][i])+(
        actual["cold_internal_hessian"][i],) for i in range(2))+(
        tuple(actual["cold_internal_hessian"])+(actual["internal_hessian"],),)
    return actual["energy_bounds_j"], gradient, hessian


def _jet(value, gradient=None, hessian=None):
    return _api()._MFContactJet(_I(value),
        {i: _I(x) for i, x in (gradient or {}).items()},
        {ij: _I(x) for ij, x in (hessian or {}).items()}, precision=96)


def _jet_entry(value, first=None, second=None):
    if first is None:
        return value.value
    if second is None:
        return value.gradient.get(first, _I(0))
    return value.hessian.get(tuple(sorted((first, second))), _I(0))


def _normal_shear_fixture():
    # cof(A)e_z=(-c1,-c2,1), then normalize its CURRENT physical norm.
    columns = {
        6: _matrix_interval(((ZERO, ZERO, ZERO), ZERO_MATRIX[1], (F(1), ZERO, ZERO))),
        7: _matrix_interval((ZERO_MATRIX[0], ZERO_MATRIX[1], (ZERO, F(1), ZERO))),
    }
    return _api()._mf_nanson_director(_matrix_interval(IDENTITY), columns,
                                     tuple(_I(x) for x in (ZERO, ZERO, F(1))), precision=96)


def _contact_scalar_fixture():
    normal, proof = _normal_shear_fixture()
    sub = tuple(_jet(x) for x in (F(2), F(-1), F(3)))
    jump = (F(5), F(7), F(11))
    columns = {6: (F(13), F(17), F(19)), 7: (F(23), F(29), F(31)),
               92: (F(37), F(41), F(43))}
    over = tuple(_jet(sub[i].value.lo+jump[i],
                      {index: direction[i] for index, direction in columns.items()}) for i in range(3))
    scalar = _api()._mf_contact_scalar_twojet(sub, over, normal, (0, 1, 2), (3, 4, 5),
                                            _I(jump[2]), precision=96)
    return scalar, normal, proof, jump, columns, sub, over


def test_internal_beam_increment_matches_exact_finite_energy_difference_and_all_derivatives():
    coordinates = (F(2, 7), F(-3, 11), F(5, 13))
    _contains(_beam_full_jet(_beam_actual(coordinates)), _polynomial_twojet(_beam_difference, coordinates))


def test_zero_internal_increment_preserves_cold_energy_force_and_tangent_but_keeps_cross_block():
    coordinates = (F(2, 7), F(-3, 11), ZERO)
    actual = _beam_actual(coordinates)
    assert actual["energy_bounds_j"] == _I(0)
    assert actual["cold_gradient"] == (_I(0),)*2
    assert actual["cold_hessian"] == ((_I(0),)*2,)*2
    expected = _polynomial_twojet(_beam_difference, coordinates)
    _contains(_beam_full_jet(actual), expected)
    assert expected[1][2] != 0 and expected[2][0][2] != 0
    assert actual["internal_hessian"] == _I(F(29, 31))


def test_internal_increment_retains_projection_curvature_and_source_force_intercept():
    coordinates = (F(-4, 9), F(1, 5), F(-7, 12))
    actual = _beam_actual(coordinates)
    expected = _polynomial_twojet(_beam_difference, coordinates)
    _contains(_beam_full_jet(actual), expected)
    assert expected[2][0][1] == F(29, 31)*coordinates[2]*F(-2, 13)
    p, _, _ = _quadratic_projection(coordinates[:2])
    assert expected[1][2] == F(-17, 23)+F(29, 31)*(p+coordinates[2])


def test_owner_torque_and_full_drag_twojet_match_literal_mixed_unit_potential():
    theta = tuple(F((-1)**i*(i+1), 101) for i in range(6))
    torque = tuple(F(i-3, 17) for i in range(6))
    drag = tuple(tuple(F(i+j+1, 23)+F(10 if i == j else 0) for j in range(6)) for i in range(6))
    dt = F(1000000)
    potential = lambda point: sum((point[i]*drag[i][j]*point[j]/(2*dt)
        for i in range(6) for j in range(6)), ZERO)-sum((a*b for a, b in zip(torque, point)), ZERO)
    actual = _api()._owner_load_twojet(theta, _matrix_interval(drag), tuple(_I(x) for x in torque), dt,
                                      precision=96)
    expected = _polynomial_twojet(potential, theta)
    _contains((actual["energy_bounds_j"], actual["gradient_bounds"], actual["hessian_bounds"]), expected)
    assert expected[2][0][3] != 0


def test_normalization_removes_diagonal_nanson_area_stretch_from_contact_direction():
    columns = {6: _matrix_interval(((F(1), ZERO, ZERO), ZERO_MATRIX[1], ZERO_MATRIX[2])),
               7: _matrix_interval((ZERO_MATRIX[0], (ZERO, F(1), ZERO), ZERO_MATRIX[2]))}
    normal, _ = _api()._mf_nanson_director(_matrix_interval(IDENTITY), columns,
                                         tuple(_I(x) for x in (ZERO, ZERO, F(1))), precision=96)
    _contains(tuple(x.value for x in normal), (ZERO, ZERO, F(1)))
    for value in normal:
        for index in (6, 7):
            _contains(_jet_entry(value, index), ZERO)
            for other in (6, 7):
                _contains(_jet_entry(value, index, other), ZERO)


def test_unit_nanson_shear_direction_has_full_normalization_curvature():
    normal, _ = _normal_shear_fixture()
    _contains(tuple(x.value for x in normal), (ZERO, ZERO, F(1)))
    _contains(tuple(_jet_entry(x, 6) for x in normal), (F(-1), ZERO, ZERO))
    _contains(tuple(_jet_entry(x, 7) for x in normal), (ZERO, F(-1), ZERO))
    _contains(tuple(_jet_entry(x, 6, 6) for x in normal), (ZERO, ZERO, F(-1)))
    _contains(tuple(_jet_entry(x, 7, 7) for x in normal), (ZERO, ZERO, F(-1)))
    _contains(tuple(_jet_entry(x, 6, 7) for x in normal), (ZERO, ZERO, ZERO))


def test_contact_scalar_retains_material_internal_cross_derivatives_and_normal_curvature():
    scalar, _, _, jump, columns, _, _ = _contact_scalar_fixture()
    _contains(scalar.value, ZERO)
    _contains(_jet_entry(scalar, 6), -jump[0]+columns[6][2])
    _contains(_jet_entry(scalar, 7), -jump[1]+columns[7][2])
    _contains(_jet_entry(scalar, 92), columns[92][2])
    _contains(_jet_entry(scalar, 6, 6), -jump[2]-2*columns[6][0])
    _contains(_jet_entry(scalar, 7, 7), -jump[2]-2*columns[7][1])
    _contains(_jet_entry(scalar, 6, 7), -columns[7][0]-columns[6][1])
    _contains(_jet_entry(scalar, 6, 92), -columns[92][0])
    _contains(_jet_entry(scalar, 7, 92), -columns[92][1])


def test_contact_fresh_owner_jets_preserve_common_rotation_objectivity_and_owner_material_cross_terms():
    scalar, _, _, _, _, _, _ = _contact_scalar_fixture()
    for axis in range(3):
        _contains(_jet_entry(scalar, axis)+_jet_entry(scalar, axis+3), ZERO)
        for other in range(3):
            _contains(sum((_jet_entry(scalar, i, j) for i in (axis, axis+3)
                           for j in (other, other+3)), _I(0)), ZERO)
        for material in (6, 7, 92):
            _contains(_jet_entry(scalar, axis, material)+_jet_entry(scalar, axis+3, material), ZERO)
    assert any(not (_jet_entry(scalar, axis).lo <= 0 <= _jet_entry(scalar, axis).hi) for axis in range(3))
    assert any(not (_jet_entry(scalar, axis, material).lo <= 0 <= _jet_entry(scalar, axis, material).hi)
               for axis in range(6) for material in (6, 7, 92))


def test_contact_potential_contains_prestress_times_full_scalar_curvature():
    scalar, _, _, jump, columns, _, _ = _contact_scalar_fixture()
    anchor, area, stiffness = F(2, 5), F(3, 7), F(17)
    potential = (anchor+scalar).square()*area*stiffness/2
    indices = (6, 7, 92)
    first = (-jump[0]+columns[6][2], -jump[1]+columns[7][2], columns[92][2])
    second = ((-jump[2]-2*columns[6][0], -columns[7][0]-columns[6][1], -columns[92][0]),
              (-columns[7][0]-columns[6][1], -jump[2]-2*columns[7][1], -columns[92][1]),
              (-columns[92][0], -columns[92][1], ZERO))
    _contains(potential.value, area*stiffness*anchor*anchor/2)
    for i, global_i in enumerate(indices):
        _contains(_jet_entry(potential, global_i), area*stiffness*anchor*first[i])
        for j, global_j in enumerate(indices):
            expected = area*stiffness*(first[i]*first[j]+anchor*second[i][j])
            _contains(_jet_entry(potential, global_i, global_j), expected)
    assert second[0][0] != 0 and second[0][2] != 0


def _coordinates(cold_index=0, hot_index=0, residual_sign=0):
    hot, cold, _, _ = _native().source_packets()
    cold_q = tuple(_fraction(x) for x in cold["evaluation_queries"][cold_index]["response"][
        "coordinate_increments"])
    hot_q = tuple(_fraction(x) for x in hot["evaluation_queries"][hot_index]["response"][
        "coordinate_increments"])
    internal = tuple(F(residual_sign*(-1 if i % 2 else 1)*(i+1), 1000000) for i in range(8))
    return (ZERO,)*6+hot_q+cold_q+internal


def _seed():
    return json.loads(_native().source_packets()[2]["carrier_source_work_system"]["seed_json"])


def test_original_hundred_catalogue_is_preserved_with_exact_hot_chart_replacement():
    hot, cold, work, _ = _native().source_packets()
    system = _native().system()
    original = tuple(tuple(x) for x in work["carrier_source_work_system"]["coordinate_order"])
    old_values = tuple(_fraction(x) for x in work["carrier_source_work_system"]["bound_coordinate_increments"])
    assert len(original) == len(old_values) == len(system.coordinate_order) == 100
    assert system.original_coordinate_order == original
    assert system.original_bound_coordinate_increments == old_values
    assert system.coordinate_order[:6] == original[:6]
    assert system.coordinate_order[6:36] == tuple(("body", *x) for x in hot["hot_mean_system"]["coordinate_order"])
    assert system.coordinate_order[36:] == original[36:]
    assert len(set(system.coordinate_order)) == 100
    assert system.bound_coordinate_increments[:6] == (ZERO,)*6
    assert system.bound_coordinate_increments[6:36] == (ZERO,)*30
    assert system.bound_coordinate_increments[36:92] == old_values[36:92]
    assert system.bound_coordinate_increments[92:] == old_values[92:] == (ZERO,)*8
    assert len(cold["evaluation_queries"]) == 2 and len(hot["evaluation_queries"]) == 4


def test_authentic_source_atoms_and_beams_keep_complete_original_ownership_and_positive_areas():
    seed = _seed()
    assert len(seed["source_atoms"]) == 16 and len(seed["attachments"]) == 32 and len(seed["beams"]) == 8
    assert len({(x["field"]["point_id"], x["field"]["side"]) for x in seed["attachments"]}) == 32
    assert all(x["field"]["original_parent_id"] in {"octant:"+str(i) for i in range(8)}
               and all(36 <= index < 92 for index in x["field"]["body_coordinate_indices"])
               for x in seed["attachments"])
    assert sum(x["field"]["original_label_membership"] == "LIVE_ORIGINAL_MATERIAL"
               for x in seed["attachments"]) == 24
    assert sum(x["field"]["original_label_membership"] == "ARCHIVED_ORIGINAL_MATERIAL"
               for x in seed["attachments"]) == 8
    beams = {x["original_parent_id"]: x for x in seed["beams"]}
    assert set(beams) == {"octant:"+str(i) for i in range(8)}
    original = tuple(tuple(x) for x in _native().source_packets()[2]["carrier_source_work_system"][
        "coordinate_order"])
    expected_internal = {0: 99, 1: 93, 2: 95, 3: 97, 4: 98, 5: 92, 6: 94, 7: 96}
    for parent, global_index in expected_internal.items():
        beam = beams["octant:"+str(parent)]
        assert original[global_index] == ("source_internal", beam["trace_id"], beam["side"])
    assert len({_fraction(x["original_area_m2"]) for x in seed["source_atoms"]}) > 1
    assert all(_fraction(x["original_area_m2"]) > 0 for x in seed["source_atoms"])


def test_declared_query_catalogue_exercises_all_saved_constitutive_states_and_internal_signs():
    queries = _native().query_coordinates()
    assert len(queries) == 12 and len({label for label, _ in queries}) == 12
    assert len({q for _, q in queries}) == 12
    assert all(type(q) is tuple and len(q) == 100 and all(type(x) is F for x in q)
               and q[:6] == (ZERO,)*6 for _, q in queries)
    assert sum(q[92:] == (ZERO,)*8 for _, q in queries) == 8
    assert all(all(x != 0 for x in q[92:]) for _, q in queries if any(q[92:]))
    assert any(q[92] > 0 for _, q in queries) and any(q[92] < 0 for _, q in queries)


def test_full_operator_has_hundred_mixed_units_and_symmetric_owner_body_internal_blocks():
    response = _native().response(_coordinates(0, 1, 1))
    assert len(response.gradient_bounds) == len(response.hessian_bounds) == 100
    assert all(len(row) == 100 for row in response.hessian_bounds)
    assert response.hessian_bounds == tuple(zip(*response.hessian_bounds))
    assert response.coordinate_increments == _coordinates(0, 1, 1)
    assert len(response.internal_beam_corrections_json) == 8
    assert len(response.contact_atoms_json) == 16
    assert len(response.source_point_receipts_json) == 32
    assert response.whole_cold_parent_count == 8 and response.whole_hot_body_count == 5
    assert any(not (response.hessian_bounds[owner][body].lo <= 0 <= response.hessian_bounds[owner][body].hi)
               for owner in range(6) for body in range(36, 100))


def test_hot_constitutive_block_is_the_saved_finite_mean_operator_without_second_pullback():
    hot, _, _, _ = _native().source_packets()
    saved = hot["evaluation_queries"][1]["response"]
    response = _native().response(_coordinates(0, 1, 1))
    assert response.gradient_bounds[6:36] == _decode(saved["gradient_bounds_n"])
    assert tuple(tuple(row[6:36]) for row in response.hessian_bounds[6:36]) == _decode(
        saved["hessian_bounds_n_m"])
    assert all(response.hessian_bounds[i][j] == _I(0)
               for i in range(6, 36) for j in tuple(range(6))+tuple(range(36, 100)))
    bodies = hot["hot_mean_system"]["bodies"]
    assert len(bodies) == 5 and all(_decode(body["actual_area_bounds_sr"]).lo > 0 for body in bodies)
    assert sum(_decode(body["actual_area_bounds_sr"]).hi < F(1, 10**40) for body in bodies) == 3
    assert all(any(value != _I(0) for value in response.gradient_bounds[6+6*i:12+6*i])
               for i in range(5))


def test_unsupported_constitutive_states_owner_endpoints_and_literal_kinds_are_rejected():
    api, system = _api(), _native().system()
    for changed_index in (0, 6, 36):
        query = list(system.bound_coordinate_increments)
        query[changed_index] += F(1, 10**7)
        with pytest.raises(ValueError):
            api.evaluate_native_common_mixed_force(system, tuple(query))
    for query in ([ZERO]*100, (ZERO,)*99, (0,)+(ZERO,)*99):
        with pytest.raises(ValueError):
            api.evaluate_native_common_mixed_force(system, query)
    with pytest.raises(ValueError):
        api.evaluate_native_common_mixed_force(system, precision=True)


def test_live_typed_system_and_response_bindings_reject_changed_history_query_and_force():
    api, native = _api(), _native()
    system = native.system()
    with pytest.raises(ValueError):
        api.native_common_mixed_force_to_dict(replace(system, provenance="replacement history"))
    response = native.response(_coordinates(0, 1, 1))
    altered_gradient = (response.gradient_bounds[0]+_I(1),)+response.gradient_bounds[1:]
    with pytest.raises(ValueError):
        api.native_common_mixed_force_response_to_dict(replace(response, gradient_bounds=altered_gradient))
    with pytest.raises(ValueError):
        api.native_common_mixed_force_response_to_dict(replace(response, coordinate_increments=(ZERO,)*100))


def test_full_system_and_response_restart_rebind_saved_authorities_and_exact_query():
    api, native = _api(), _native()
    system = native.system()
    raw = api.native_common_mixed_force_to_dict(system)
    restored = api.native_common_mixed_force_from_dict(raw, *native.source_packets(),
        cold_carrier=native.cold_carrier())
    assert api.native_common_mixed_force_to_dict(restored) == raw
    query = _coordinates(0, 1, 1)
    response = native.response(query)
    dto = api.native_common_mixed_force_response_to_dict(response)
    resumed = api.native_common_mixed_force_response_from_dict(dto, system=restored,
                                                               expected_coordinate_increments=query)
    assert api.native_common_mixed_force_response_to_dict(resumed) == dto
    with pytest.raises(ValueError):
        api.native_common_mixed_force_response_from_dict(dto, system=restored,
            expected_coordinate_increments=_coordinates(0, 1, -1))


def test_saved_components_and_shared_internal_beams_enter_the_total_potential_once():
    native, system = _native(), _native().system()
    hot, cold, _, _ = native.source_packets()
    query = _coordinates(0, 1, 1)
    response = native.response(query)
    partitions = json.loads(response.energy_partitions_json)
    _overlap(_decode(partitions["cold_whole_material_and_eight_beams_bounds_j"]),
             _decode(cold["evaluation_queries"][0]["response"]["energy_j"]))
    _overlap(_decode(partitions["hot_whole_material_and_finite_radial_base_bounds_j"]),
             _decode(hot["evaluation_queries"][1]["response"]["energy_bounds_j"]))
    keys = ("cold_whole_material_and_eight_beams_bounds_j",
            "hot_whole_material_and_finite_radial_base_bounds_j",
            "eight_internal_beam_driver_correction_bounds_j",
            "frozen_owner_load_drag_potential_bounds_j",
            "original_source_atom_loading_potential_bounds_j")
    _overlap(response.energy_bounds_j, sum((_decode(partitions[key]) for key in keys), _I(0)))
    assert partitions["frozen_owner_load_drag_potential_bounds_j"] == _api()._payload(_I(0))
    assert partitions["old891_complete_beam_bulk_operator_added_again"] is False
    assert partitions["old_scalar_hot_foundation_added_again"] is False
    assert response.cold_beams_included_once and response.internal_residuals_shift_existing_beams_once
    banks = cold["cold_continuum_energy_system"]["banks"]
    saved_projection = _decode(cold["evaluation_queries"][0]["response"]["tip_projection_increment_m"])
    summed = _I(0)
    for beam, encoded in zip(system.internal_beams, response.internal_beam_corrections_json):
        receipt = json.loads(encoded)
        expected_bank = next(i for i, bank in enumerate(banks)
                             if bank["original_parent_id"] == beam.original_parent_id)
        assert beam.cold_bank_index == expected_bank
        assert beam.cold_tip_coordinate_index == banks[expected_bank]["tip_coordinate_index"]
        p = saved_projection[expected_bank]
        r = query[beam.global_coordinate_index]
        expected = (beam.literal_reference_force_n+beam.stiffness_exact_n_m*p)*r+beam.stiffness_exact_n_m*r*r/2
        _overlap(_decode(receipt["energy_increment_bounds_j"]), expected)
        assert receipt["old_reference_energy_added_again"] is False
        assert receipt["nonlinear_projection_curvature_included"] is True
        summed += expected
    _overlap(_decode(partitions["eight_internal_beam_driver_correction_bounds_j"]), summed)
    zero_internal = native.response(_coordinates(0, 1))
    assert _decode(json.loads(zero_internal.energy_partitions_json)[
        "eight_internal_beam_driver_correction_bounds_j"]) == _I(0)


def test_declared_endpoint_flags_preserve_complete_source_history_clocks_and_open_physics():
    native, system = _native(), _native().system()
    hot, _, _, transport = native.source_packets()
    assert system.complete_original_material_state_json == transport[
        "material_transfer"]["complete_original_material_state_json"]
    assert system.complete_original_source_clock_work_history_json == hot[
        "hot_mean_system"]["complete_original_source_clock_work_history_json"]
    assert system.complete_current_load_json == hot["hot_mean_system"]["complete_current_load_json"]
    assert system.owner_drag_time_scale_s == 1000000
    assert system.owner_drag_time_scale_is_new_declared_calibration
    assert system.hot_constitutive_law_defined and system.finite_mean_chart_defined
    response = native.response(_coordinates(0, 1, 1))
    assert system.coordinate_units == response.coordinate_units == ("rad",)*6+("m",)*94
    assert system.gradient_units == response.gradient_units == ("N*m",)*6+("N",)*94
    for value in (system, response):
        assert (value.source_geometry_elapsed_time_s, value.contact_geometry_elapsed_time_s,
                value.memory_elapsed_time_s, value.material_geometry_elapsed_time_s) == (
                    F(0), F(2000000), F(1000000), F(1000000))
        assert value.accepted_original_native_intervals == 1 and value.accepted_new_native_intervals == 0
        assert value.original_point_1_remaining_work_blocks == 1
        assert value.remaining_physical_conditions == _api().REMAINING_CONDITIONS
        for name in ("finite_mean_equilibrium_certified",
                     "physical_source_force_prestress_work_handover_certified",
                     "historical_archived_three_dimensional_prestress_recovered",
                     "prepared_reference_full_system_work_certified", "born_hot_receiving_contact_law_certified",
                     "whole_physical_mixed_operator_certified", "new_mixed_force_equilibrium_certified",
                     "physical_material_admission_authorized", "whole_clock_all_pair_admission_authorized",
                     "full_system_temporal_path_work_certified", "clocks_advanced", "original_point_1_completed"):
            assert getattr(value, name) is False


def test_wrong_source_seal_and_resealed_coordinate_names_cannot_create_a_new_operator():
    api, native = _api(), _native()
    hot, cold, work, transport = native.source_packets()
    with pytest.raises(ValueError):
        api.bind_native_common_mixed_force(dict(hot, packet_sha256="0"*64), cold, work, transport,
            cold_carrier=native.cold_carrier(), provenance="rejected source replacement")
    raw = api.native_common_mixed_force_to_dict(native.system())
    changed = dict(raw)
    changed["coordinate_order"] = list(raw["coordinate_order"])
    changed["coordinate_order"][36], changed["coordinate_order"][37] = (
        changed["coordinate_order"][37], changed["coordinate_order"][36])
    changed["binding_sha256"] = hashlib.sha256(_encoded({
        k: v for k, v in changed.items() if k not in ("format", "binding_sha256")})).hexdigest()
    with pytest.raises(ValueError):
        api.native_common_mixed_force_from_dict(changed, *native.source_packets(),
                                                cold_carrier=native.cold_carrier())


def test_prepared_zero_contact_calibration_preserves_every_literal_atom_energy_and_area_once():
    response = _native().response(_coordinates(1, 0))
    original = {x["point_id"]: x for x in _seed()["source_atoms"]}
    summed = ZERO
    for encoded in response.contact_atoms_json:
        receipt = json.loads(encoded)
        source = original[receipt["point_id"]]
        literal = sum((_fraction(x) for x in source["stored_source_energy_components_j"]), ZERO)
        assert receipt["original_area_m2"] == source["original_area_m2"]
        assert json.loads(receipt["original_atom_source_binding_json"]) == source
        assert receipt["original_source_state_json"] == source["source_local_state_json"]
        assert _decode(receipt["energy_bounds_j"]) == _I(literal)
        assert _decode(receipt["recoverable_energy_bounds_j"]) == _I(literal)
        assert _decode(receipt["scalar_increment_twojet"]["value_bounds"]) == _I(0)
        assert receipt["exact_prepared_zero_scalar_energy_identity_used"] is True
        assert receipt["prepared_zero_source_traction_calibration"] is True
        assert receipt["original_source_area_included_once"] is True
        assert receipt["current_area_energy_multiplier_added"] is False
        assert receipt["born_hot_receiving_law_inferred"] is False
        assert receipt["source_state_or_work_advanced"] is False
        assert receipt["proposed_normal_damage_or_fracture_work_applied"] is False
        summed += literal
    partitions = json.loads(response.energy_partitions_json)
    assert _decode(partitions["original_source_atom_loading_potential_bounds_j"]) == _I(summed)


def test_all_thirty_two_original_material_receipts_retain_whole_fifty_six_column_geometry():
    native = _native()
    response = native.response(_coordinates(0, 1, 1))
    authority = _api().native_common_mixed_force_contact_authority_to_dict(
        native.system(), _coordinates(0, 1, 1)[36:92])
    assert authority["binding_sha256"] == response.contact_authority_binding_sha256
    references = {(x["point_id"], x["side"]): x for x in
                  (json.loads(encoded) for encoded in response.source_point_receipts_json)}
    originals = {(x["field"]["point_id"], x["field"]["side"]): x for x in _seed()["attachments"]}
    seen = set()
    for receipt in authority["point_receipts"]:
        key = receipt["point_id"], receipt["side"]
        assert key not in seen
        seen.add(key)
        reference = references[key]
        assert reference["contact_authority_binding_sha256"] == authority["binding_sha256"]
        assert reference["complete_point_receipt_sha256"] == hashlib.sha256(_encoded(receipt)).hexdigest()
        assert reference["current_point_binding_sha256"] == receipt["current"]["binding_sha256"]
        assert reference["prepared_zero_point_binding_sha256"] == receipt["prepared_zero"]["binding_sha256"]
        original = originals[key]
        assert receipt["original_saved_attachment"] == original
        field = original["field"]
        for state_key in ("current", "prepared_zero"):
            point = receipt[state_key]
            assert point["original_parent_id"] == field["original_parent_id"]
            assert point["original_material_id"] == field["original_material_id"]
            assert point["owner"] == field["owner"]
            archived = field["original_label_membership"] == "ARCHIVED_ORIGINAL_MATERIAL"
            assert point["allocation"] == ("archived" if archived else "live")
            assert point["archive_debits"] == int(archived)
            if archived:
                assert point["cohort_id"] == field["archive_cohort_id"]
            else:
                assert field["archive_cohort_id"] is None
            assert point["cohort_id"] == receipt["prepared_zero"]["cohort_id"]
            assert len(point["coordinate_order"]) == 56
            assert len(point["mode_displacement_columns"]) == 56
            assert len(point["mode_original_gradient_columns_per_m"]) == 56
            assert all(len(matrix) == 3 and all(len(row) == 3 for row in matrix)
                       for matrix in point["mode_original_gradient_columns_per_m"])
            assert point["material_membership_authenticated"] is True
            assert _decode(point["volume_jacobian_bounds"]).lo > 0
            assert _decode(point["material_depth_bounds_m"]) == _I(0)
            assert point["physical_material_admission_authorized"] is False
        assert tuple(_fraction(x) for x in receipt["current"]["bound_coordinate_increments"]) == _coordinates(0, 1, 1)[36:92]
        assert tuple(_fraction(x) for x in receipt["prepared_zero"]["bound_coordinate_increments"]) == (ZERO,)*56
    assert seen == set(originals)


