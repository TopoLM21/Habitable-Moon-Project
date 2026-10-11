"""Independent finite-base chart and whole authentic HOT constitutive checks.

An analytic one-ray chart has a rational exact root and exact first/second
derivatives.  A separate literal matrix oracle checks the finite Green law.
Those checks do not rely on saved diagnostic amplitudes or midpoint rays;
the authentic five-body tests retain all original masks and old source data.
"""
from dataclasses import replace
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json

import pytest

from tectonics.geometric_native_pose import FractionInterval as I


ZERO = F(0)
IDENTITY = tuple(tuple(F(int(i == j)) for j in range(3)) for i in range(3))
GENERATORS = (
    IDENTITY,
    ((F(1), ZERO, ZERO), (ZERO, F(-1), ZERO), (ZERO, ZERO, ZERO)),
    ((F(1), ZERO, ZERO), (ZERO, F(1), ZERO), (ZERO, ZERO, F(-2))),
    ((ZERO, F(1), ZERO), (F(1), ZERO, ZERO), (ZERO, ZERO, ZERO)),
    ((ZERO, ZERO, F(1)), (ZERO, ZERO, ZERO), (F(1), ZERO, ZERO)),
    ((ZERO, ZERO, ZERO), (ZERO, ZERO, F(1)), (ZERO, F(1), ZERO)),
)


@lru_cache(maxsize=1)
def _api():
    from tectonics import geometric_native_common_hot_mean as api
    return api


@lru_cache(maxsize=1)
def _native():
    from analysis.geometric_native_common_hot_mean_validation import native_control
    return native_control


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
        return I(_fraction(value["lo"]), _fraction(value["hi"]))
    if type(value) is list:
        return tuple(_decode(x) for x in value)
    return value


def _contains(actual, expected):
    if type(actual) is I:
        assert actual.lo <= expected <= actual.hi, (actual, expected)
    else:
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            _contains(first, second)


def _overlap(actual, expected):
    expected = expected if type(expected) is I else I(expected)
    assert max(actual.lo, expected.lo) <= min(actual.hi, expected.hi)


def _transpose(matrix):
    return tuple(tuple(matrix[j][i] for j in range(3)) for i in range(3))


def _multiply(first, second):
    return tuple(tuple(sum((first[i][k]*second[k][j] for k in range(3)), ZERO)
                       for j in range(3)) for i in range(3))


def _add(first, second):
    return tuple(tuple(a+b for a, b in zip(row_a, row_b))
                 for row_a, row_b in zip(first, second))


def _scale(matrix, coefficient):
    return tuple(tuple(coefficient*x for x in row) for row in matrix)


def _sym(matrix):
    return _scale(_add(matrix, _transpose(matrix)), F(1, 2))


def _tensor_inner(first, second, lame, shear):
    return lame*sum((first[i][i] for i in range(3)), ZERO)*sum(
        (second[i][i] for i in range(3)), ZERO)+2*shear*sum(
        (first[i][j]*second[i][j] for i in range(3) for j in range(3)), ZERO)


def _literal_bulk_twojet(body, h, b):
    """Literal rational differentiation of W(E), E=(F.T F-I)/2."""
    amplitudes = (h,)+b
    deformation = _add(IDENTITY, tuple(tuple(sum(
        (a*K[i][j]/body.radius_m for a, K in zip(amplitudes, GENERATORS)), ZERO)
        for j in range(3)) for i in range(3)))
    columns = tuple(_scale(K, 1/body.radius_m) for K in GENERATORS)
    strain = _scale(_add(_multiply(_transpose(deformation), deformation),
                         _scale(IDENTITY, -1)), F(1, 2))
    first = tuple(_sym(_multiply(_transpose(deformation), K)) for K in columns)
    second = tuple(tuple(_sym(_multiply(_transpose(K), L)) for L in columns)
                   for K in columns)
    inner = lambda A, B: _tensor_inner(A, B, body.lame_modulus_pa,
                                       body.shear_modulus_pa)
    volume = body.original_reference_volume_bounds_m3.lo
    return (volume*inner(strain, strain)/2,
            tuple(volume*inner(strain, D) for D in first),
            tuple(tuple(volume*(inner(first[i], first[j])+inner(strain, second[i][j]))
                        for j in range(6)) for i in range(6)))


def _analytic_body(directions=((F(1), ZERO, ZERO),), area=F(1, 7)):
    """A synthetic discrete angular measure for a closed-form math oracle.

    It is never bound as a native system or offered as an authentic material
    mask.  Each exact unit direction has an equal positive material weight.
    """
    api = _api()
    cells = tuple(json.dumps(dict(
        kind="SYNTHETIC_ANALYTIC_UNIT_RAY", source_index=index,
        source_sha256=hashlib.sha256(_encoded([index, "analytic ray"])).hexdigest(),
        area_bounds_sr=api._payload(I(area/len(directions))),
        direction_bounds=api._payload(tuple(I(x) for x in direction)),
        material_area_is_containing_chart_area=False), sort_keys=True,
        separators=(",", ":")) for index, direction in enumerate(directions))
    return api.NativeCommonHotMeanBody(
        component_id="analytic-unit-ray", material_id="synthetic-test-only",
        original_parent_id="synthetic-test-only", owner=1, radius_m=F(100),
        areal_material_thickness_m=F(2), original_base_radius_bounds_m=I(90),
        prepared_base_radius_bounds_m=I(90), fixed_owner_scale=F(1),
        owner_pose_bounds=tuple(tuple(I(x) for x in row) for row in IDENTITY),
        angular_rate_per_physical_volume_depth_m=(ZERO,)*3,
        actual_area_bounds_sr=I(area), original_reference_volume_bounds_m3=I(F(11, 13)),
        prepared_reference_volume_bounds_m3=I(F(11, 13)), young_modulus_pa=F(10**9),
        shear_modulus_pa=F(4*10**8), lame_modulus_pa=F(4*10**8),
        density_kg_m3=F(2800), gravity_m_s2=F(3, 2),
        source_birth_radial_displacement_m=ZERO, mask_cells=cells,
        complete_common_bank_json="{}", complete_original_common_hot_component_json="{}",
        complete_original_birth_json="{}", complete_foundation_source_json="{}",
        complete_conditional_old19_step_json="{}", conditional_old19_coordinate_decimal="0",
        authentic_old100_coordinate_indices=tuple(range(6)),
        authentic_old100_coordinate_values=(ZERO,)*6,
        saved_nonzero_probe_old_cartesian_amplitudes_m=(F(1),)*6,
        source_bank_snapshot_sha256="analytic-test", source_snapshot_sha256="analytic-test",
        source_support_sha256="analytic-test", source_epoch_sha256="analytic-test")


def _one_ray_oracle():
    # |F e_x| has longitudinal length A=125 and transverse (21,28),
    # so its positive longitudinal root is exactly S=120 metres.
    body = _analytic_body()
    q, b = F(45, 2), (F(1), F(2), F(21), F(28), F(3))
    A, S, alpha = F(125), F(120), body.radius_m/F(90)
    h = S-body.radius_m-b[0]-b[1]
    first = (A*alpha/S, F(-1), F(-1), -b[2]/S, -b[3]/S, ZERO)
    second = [[ZERO for _ in range(6)] for _ in range(6)]
    second[0][0] = -(b[2]**2+b[3]**2)*alpha**2/S**3
    for index, transverse in ((3, b[2]), (4, b[3])):
        second[0][index] = second[index][0] = A*alpha*transverse/S**3
        second[index][index] = -1/S-transverse**2/S**3
    second[3][4] = second[4][3] = -b[2]*b[3]/S**3
    J = tuple(tuple(first[a] if i == 0 else F(int(i == a)) for a in range(6))
              for i in range(6))
    return body, q, b, h, J, tuple(tuple(row) for row in second)


def test_finite_actual_radial_mean_root_and_all_implicit_derivatives_match_closed_form():
    body, q, b, exact_h, exact_J, exact_second = _one_ray_oracle()
    h, J, second, certificate = _api()._solve_finite_mean_body(body, q, b)
    _contains(h, exact_h)
    _contains(J, exact_J)
    _contains(second, exact_second)
    assert certificate["whole_mask_unique_root_certified"]
    assert certificate["literal_root_replaced_by_midpoint"] is False
    assert _decode(certificate["complete_mask_mean_h_derivative_positive_bounds"]).lo > 0
    assert exact_second[0][0] < 0 and exact_second[3][4] < 0
    # Finite radial length is different from reference-normal displacement:
    # q fixes |F e_x|, whose transverse components contribute to the norm.
    reference_normal_mean = F(90)*(exact_h+b[0]+b[1])/body.radius_m
    assert reference_normal_mean != q
    # This deliberately large synthetic shear uses a separate convergence
    # budget.  The default 12-step enclosure already proves the exact root
    # and both derivative orders; further steps tighten that same enclosure.
    refined_h, refined_J, refined_second, refined_certificate = _api()._solve_finite_mean_body(
        body, q, b, root_iterations=24)
    _contains(refined_h, exact_h)
    _contains(refined_J, exact_J)
    _contains(refined_second, exact_second)
    assert h.lo <= refined_h.lo <= refined_h.hi <= h.hi
    assert refined_h.width < F(1, 10**12)
    assert refined_certificate["whole_mask_unique_root_certified"]


def test_full_green_bulk_value_gradient_and_cross_hessian_match_literal_matrix_oracle():
    body, _, b, h, _, _ = _one_ray_oracle()
    actual = _api()._bulk_twojet(body, I(h), b, 96)
    expected = _literal_bulk_twojet(body, h, b)
    _contains(actual, expected)
    assert expected[2][0][3] != 0 and expected[2][3][4] != 0
    # Nonzero finite h,b make the stress-dependent second variation observable.
    zero = _literal_bulk_twojet(body, ZERO, (ZERO,)*5)
    assert expected[2] != zero[2]


def test_chart_force_curvature_is_required_for_one_ray_basal_work_conjugacy():
    api = _api()
    body, q, b, _, _, _ = _one_ray_oracle()
    h, J, hs, _ = api._solve_finite_mean_body(body, q, b)
    physical = api._basal_twojet(body, h, b, 96)
    transformed = api._chain_twojet(*physical[:3], J, hs, 96)
    coefficient = body.density_kg_m3*body.gravity_m_s2*body.radius_m**2*body.actual_area_bounds_sr.lo
    expected_hessian = tuple(tuple(coefficient if i == j == 0 else ZERO for j in range(6))
                             for i in range(6))
    _contains(transformed, (coefficient*q*q/2, (coefficient*q,)+(ZERO,)*5, expected_hessian))
    uncorrected = sum((J[i][3]*physical[2][i][j]*J[j][3]
                       for i in range(6) for j in range(6)), I(0))
    assert uncorrected.lo > 0 or uncorrected.hi < 0
    assert hs[3][3].hi < 0 and physical[1][0].lo > 0


def test_same_finite_field_basal_energy_retains_variance_at_zero_mean():
    api = _api()
    body = _analytic_body(((F(1), ZERO, ZERO), (ZERO, F(1), ZERO)))
    q, b = ZERO, (F(5), ZERO, ZERO, ZERO, ZERO)
    h, _, _, certificate = api._solve_finite_mean_body(body, q, b)
    _contains(h, ZERO)
    energy, _, _, receipts = api._basal_twojet(body, h, b, 96)
    displacement = F(90)*b[0]/body.radius_m
    exact = body.density_kg_m3*body.gravity_m_s2*body.radius_m**2*body.actual_area_bounds_sr.lo*displacement**2/2
    _contains(energy, exact)
    assert energy.lo > 0 and q == 0
    assert len(receipts) == 2 and certificate["true_area_normalization"]


def test_normalized_chart_and_root_do_not_floor_a_positive_extremely_small_area():
    body, q, b, exact_h, exact_J, exact_second = _one_ray_oracle()
    tiny = _analytic_body(area=F(1, 10**80))
    h, J, second, certificate = _api()._solve_finite_mean_body(tiny, q, b)
    _contains(h, exact_h)
    _contains(J, exact_J)
    _contains(second, exact_second)
    assert certificate["positive_source_area_floor_used"] is False
    assert _decode(certificate["actual_source_area_bounds_sr"]) == I(F(1, 10**80))


def test_root_domain_and_literal_numeric_kinds_reject_unproved_queries():
    api, body = _api(), _analytic_body()
    for q, b, kwargs in ((0, (ZERO,)*5, {}), (ZERO, [ZERO]*5, {}),
                        (ZERO, (ZERO,)*5, {"precision": True}),
                        (ZERO, (ZERO,)*5, {"root_iterations": True}),
                        (ZERO, (F(90), ZERO, ZERO, ZERO, ZERO), {})):
        with pytest.raises(ValueError):
            api._solve_finite_mean_body(body, q, b, **kwargs)


def test_complete_authorities_keep_five_positive_masks_and_unchanged_hot_source_zero():
    transport, work, cold, common, normal, curved = _native().source_packets()
    system = _native().system()
    order = tuple(tuple(x) for x in work["carrier_source_work_system"]["coordinate_order"])
    values = tuple(_fraction(x) for x in work["carrier_source_work_system"]["bound_coordinate_increments"])
    assert len(order) == len(values) == 100
    assert system.complete_original_100_coordinate_order == order
    assert system.complete_original_100_coordinate_values == values
    assert len(system.bodies) == 5 and len(system.coordinate_order) == 30
    assert system.bound_coordinate_increments == (ZERO,)*30
    assert len(set(system.coordinate_order)) == 30
    assert system.source_packet_seals == tuple(packet["packet_sha256"] for packet in
                                               (transport, work, cold, common, normal, curved))
    for body in system.bodies:
        assert len(body.authentic_old100_coordinate_indices) == 6
        assert body.authentic_old100_coordinate_values == (ZERO,)*6
        assert all(order[index][0:2] == ("body", body.component_id)
                   and values[index] == 0 for index in body.authentic_old100_coordinate_indices)
        assert body.actual_area_bounds_sr.lo > 0
        assert body.original_reference_volume_bounds_m3.lo > 0
        assert body.young_modulus_pa == 10**9 and body.shear_modulus_pa == 4*10**8
        assert body.lame_modulus_pa == 4*10**8
        assert body.source_birth_radial_displacement_m == 0
        assert body.radius_m == 100000 and body.areal_material_thickness_m == 2000
        _overlap(body.original_reference_volume_bounds_m3,
                 body.actual_area_bounds_sr*body.radius_m**2*body.areal_material_thickness_m)
    assert sum(body.actual_area_bounds_sr.hi < F(1, 10**40) for body in system.bodies) == 3


def test_complete_proxy_measure_and_positive_curved_correction_are_owned_once():
    system = _native().system()
    curve_count = 0
    for body in system.bodies:
        bank = json.loads(body.complete_common_bank_json)
        mask = json.loads(bank["complete_mask_source_json"])
        cells = tuple(json.loads(x) for x in body.mask_cells)
        proxies = tuple(x for x in cells if x["kind"] == "AUTHENTIC_DISJOINT_NATIVE_SOURCE_POLYGON")
        curves = tuple(x for x in cells if x["kind"] == "AUTHENTIC_POSITIVE_PAIRED_CURVED_CORRECTION_ONCE")
        assert len(proxies) == len(mask["proxy_geometries"])
        assert len(curves) == len(bank["paired_corrections_source_json"])
        assert len({x["source_sha256"] for x in cells}) == len(cells)
        for index, cell in enumerate(proxies):
            assert cell["source_index"] == index
            assert cell["source_sha256"] == hashlib.sha256(_encoded(mask["proxy_geometries"][index])).hexdigest()
        for index, cell in enumerate(curves):
            raw = json.loads(bank["paired_corrections_source_json"][index])
            assert cell["area_bounds_sr"] == raw["area_bounds_sr"]
            assert raw["positive_component_id"] == body.component_id
            assert raw["chart_is_archive_domain"] is False
            assert _decode(cell["area_bounds_sr"]).lo > 0
        assert all(x["material_area_is_containing_chart_area"] is False for x in cells)
        summed = sum((_decode(x["area_bounds_sr"]) for x in cells), I(0))
        _overlap(summed, body.actual_area_bounds_sr)
        curve_count += len(curves)
    assert curve_count == 1


def test_new_owner_scaled_source_world_reference_retains_complete_birth_and_history():
    from tectonics.geometric_native_affine_body import _normalized_native_vertex_bounds, _unit_cone_bounds
    from tectonics.geometric_native_joint_spherical_field import rotate_joint_physical_depth
    transport, _, cold, common, _, curved = _native().source_packets()
    system = _native().system()
    certificate = json.loads(cold["cold_continuum_energy_system"]["complete_source_carrier_certificate_json"])
    foundations = common["full_reference_face_birth_support"]["foundations"]
    for body in system.bodies:
        original = json.loads(body.complete_common_bank_json)
        assert body.fixed_owner_scale == _fraction(certificate[
            "donor" if body.owner == 1 else "receiver"]["fixed_radial_scale"])
        _overlap(body.prepared_base_radius_bounds_m,
                 body.fixed_owner_scale*body.original_base_radius_bounds_m)
        _overlap(body.prepared_reference_volume_bounds_m3,
                 body.fixed_owner_scale**3*body.original_reference_volume_bounds_m3)
        assert body.complete_original_birth_json == original["complete_parent_allocation_or_birth_json"]
        assert original["source_owner_history_json"]
        foundation = next(x for x in foundations if x["declaration"]["component_id"] == body.component_id)
        assert json.loads(body.complete_foundation_source_json) == foundation
        assert json.loads(body.complete_original_common_hot_component_json)["body"]["original_body"][
            "component_id"] == body.component_id
        conditional = json.loads(body.complete_conditional_old19_step_json)
        assert conditional["state"]["reference"]["cohort_id"] == body.component_id
        assert 14 <= body.conditional_old19_coordinate_index <= 18
        assert body.conditional_old19_coordinate_decimal == curved["actual_curved_force_trial"][
            "coordinates_decimal"][body.conditional_old19_coordinate_index]
        assert json.loads(body.conditional_old19_radial_displacement_literal_json) == conditional[
            "state"]["radial_displacement_m"]
        # Source polygon vertices already include their authentic placed pose.
        # The new prepared base adds the common depth rotation exactly once;
        # applying archived owner pose a second time would change these cones.
        mask = json.loads(original["complete_mask_source_json"])
        for encoded_cell in body.mask_cells:
            cell = json.loads(encoded_cell)
            if cell["kind"] == "AUTHENTIC_DISJOINT_NATIVE_SOURCE_POLYGON":
                raw = mask["proxy_geometries"][cell["source_index"]]
            else:
                raw = json.loads(original["paired_corrections_source_json"][cell["source_index"]])[
                    "containing_chart"]
            placed_vertices = _normalized_native_vertex_bounds(raw, system.precision)
            prepared_vertices = tuple(rotate_joint_physical_depth(vertex,
                body.angular_rate_per_physical_volume_depth_m, I(body.areal_material_thickness_m),
                precision=system.precision, order=32) for vertex in placed_vertices)
            assert _decode(cell["direction_bounds"]) == _unit_cone_bounds(prepared_vertices,
                                                                          system.precision)
            assert cell["source_polygon_pose_applied_once"] is True
            assert cell["additional_owner_rotation_applied"] is False
            assert cell["source_owner_history_preserved_as_metadata"] is True
    assert system.complete_original_material_state_json == transport[
        "material_transfer"]["complete_original_material_state_json"]
    assert system.reference_model == _api().REFERENCE
    assert system.basal_law == "NEW_OBJECTIVE_FINITE_RADIAL_BASE_DISPLACEMENT_WITH_FULL_VARIANCE"


def test_constitutive_component_flags_preserve_source_clocks_and_leave_admission_open():
    system = _native().system()
    assert system.hot_constitutive_law_defined and system.finite_mean_chart_defined
    assert (system.source_geometry_elapsed_time_s, system.contact_geometry_elapsed_time_s,
            system.memory_elapsed_time_s, system.material_geometry_elapsed_time_s) == (
                F(0), F(2000000), F(1000000), F(1000000))
    assert system.accepted_original_native_intervals == 1 and system.accepted_new_native_intervals == 0
    for name in ("clocks_advanced", "original_point_1_completed",
                 "physical_source_force_prestress_work_handover_certified",
                 "historical_archived_three_dimensional_prestress_recovered",
                 "prepared_reference_full_system_work_certified",
                 "physical_material_admission_authorized", "whole_clock_all_pair_admission_authorized",
                 "full_system_temporal_path_work_certified", "new_mixed_force_equilibrium_certified",
                 "finite_mean_equilibrium_certified"):
        assert getattr(system, name) is False


def test_authentic_source_zero_has_exact_zero_energy_and_force_with_finite_tangent():
    response = _native().response()
    assert response.coordinate_increments == (ZERO,)*30
    assert response.energy_bounds_j == I(0)
    assert response.gradient_bounds_n == (I(0),)*30
    assert len(response.hessian_bounds_n_m) == 30
    assert any(value.lo > 0 for row in response.hessian_bounds_n_m for value in row)
    assert response.hessian_bounds_n_m == tuple(zip(*response.hessian_bounds_n_m))
    assert response.whole_original_hot_material_count == 5
    assert response.positive_paired_curved_correction_count == 1
    assert response.source_scalar_foundation_added_again is False
    assert response.finite_mean_equilibrium_certified is False


def test_all_thirty_new_diagnostic_coordinates_keep_full_implicit_jets_and_coupled_tangents():
    coordinates = _native().diagnostic_coordinates()
    assert type(coordinates) is tuple and len(coordinates) == 30 and all(
        type(x) is F and x != 0 for x in coordinates)
    response = _native().response(coordinates)
    assert response.coordinate_increments == coordinates
    assert response.complete_hot30_operator_certified
    assert response.finite_actual_base_mean_chart_certified
    assert len(response.gradient_bounds_n) == len(response.hessian_bounds_n_m) == 30
    assert all(len(row) == 30 for row in response.hessian_bounds_n_m)
    assert response.hessian_bounds_n_m == tuple(zip(*response.hessian_bounds_n_m))
    assert response.energy_bounds_j.lo >= 0 and response.energy_bounds_j.hi > 0
    for body_index, encoded in enumerate(response.body_responses_json):
        row = json.loads(encoded)
        cert = row["finite_mean_root_certificate"]
        assert cert["whole_mask_unique_root_certified"]
        assert cert["positive_source_area_floor_used"] is False
        assert cert["physical_all_pair_admission_certified"] is False
        assert cert["reference_preparation_work_certified"] is False
        assert row["full_same_field_basal_variance_retained"]
        assert row["chart_force_curvature_term_included"]
        assert len(_decode(row["implicit_physical_coordinate_jacobian"])) == 6
        assert len(_decode(row["implicit_scalar_coordinate_hessian"])) == 6
        _contains(_decode(cert["actual_mean_bounds_m"]), coordinates[6*body_index])
        assert _decode(cert["whole_bracket_deformation_eigenvalue_bounds"]).lo > 0
        assert len(cert["complete_mask_cells"]) == len(_native().system().bodies[body_index].mask_cells)
        for other in range(5):
            if other != body_index:
                assert all(response.hessian_bounds_n_m[6*body_index+i][6*other+j] == I(0)
                           for i in range(6) for j in range(6))
    assert response.accepted_new_native_intervals == 0 and response.clocks_advanced is False
    assert response.finite_mean_equilibrium_certified is False


def test_all_five_isotropic_mean_branches_recover_original_winkler_mean_energy():
    coordinates = tuple(F((-1)**index*(index+1), 1000) if local == 0 else ZERO
                        for index in range(5) for local in range(6))
    response = _native().response(coordinates)
    for index, (body, encoded) in enumerate(zip(_native().system().bodies, response.body_responses_json)):
        row = json.loads(encoded)
        q = coordinates[6*index]
        cert = row["finite_mean_root_certificate"]
        assert cert["exact_isotropic_branch"]
        _overlap(_decode(cert["scalar_affine_amplitude_bounds_m"]),
                 I(body.radius_m*q)/body.prepared_base_radius_bounds_m)
        exact_basal = body.actual_area_bounds_sr*body.density_kg_m3*body.gravity_m_s2*body.radius_m**2*q*q/2
        _overlap(_decode(row["basal_energy_bounds_j"]), exact_basal)
        assert row["physical_source_historical_prestress_recovered"] is False


def test_system_and_response_restart_rebind_complete_authorities_and_exact_query():
    api, native = _api(), _native()
    transport, work, cold, common, normal, curved = native.source_packets()
    system = native.system()
    raw = api.native_common_hot_mean_to_dict(system)
    restored = api.native_common_hot_mean_from_dict(raw, transport, work, cold,
        common_hot_packet=common, normal_support_packet=normal, curved19_packet=curved)
    assert api.native_common_hot_mean_to_dict(restored) == raw
    coordinates = native.diagnostic_coordinates()
    response = native.response(coordinates)
    dto = api.native_common_hot_mean_response_to_dict(response)
    resumed = api.native_common_hot_mean_response_from_dict(dto, system=restored,
                                                          expected_coordinate_increments=coordinates)
    assert api.native_common_hot_mean_response_to_dict(resumed) == dto
    changed = dict(dto, accepted_new_native_intervals=1)
    changed["binding_sha256"] = hashlib.sha256(_encoded({
        k: v for k, v in changed.items() if k not in ("format", "binding_sha256")})).hexdigest()
    with pytest.raises(ValueError, match="resealed HOT response"):
        api.native_common_hot_mean_response_from_dict(changed, system=restored,
                                                      expected_coordinate_increments=coordinates)
    with pytest.raises(ValueError, match="resealed HOT response"):
        api.native_common_hot_mean_response_from_dict(dto, system=restored)


def test_resealed_system_and_changed_typed_binding_cannot_replace_source_history():
    api, native = _api(), _native()
    system = native.system()
    altered = replace(system, complete_original_source_clock_work_history_json="{}")
    with pytest.raises(ValueError, match="typed source binding"):
        api.validate_native_common_hot_mean(altered)
    raw = api.native_common_hot_mean_to_dict(system)
    changed = dict(raw, complete_original_source_clock_work_history_json="{}")
    changed["binding_sha256"] = hashlib.sha256(_encoded({
        k: v for k, v in changed.items() if k not in ("format", "binding_sha256")})).hexdigest()
    transport, work, cold, common, normal, curved = native.source_packets()
    with pytest.raises(ValueError, match="resealed HOT system"):
        api.native_common_hot_mean_from_dict(changed, transport, work, cold,
            common_hot_packet=common, normal_support_packet=normal, curved19_packet=curved)
    with pytest.raises(ValueError, match="typed source binding"):
        api.validate_native_common_hot_mean(replace(system, precision=True))
    with pytest.raises(ValueError, match="unbound authentic material body"):
        api.solve_native_common_hot_finite_mean(replace(system.bodies[0], density_kg_m3=F(1)),
                                               ZERO, (ZERO,)*5)
    with pytest.raises(ValueError, match="unbound authentic material body"):
        api.solve_native_common_hot_finite_mean(_analytic_body(), ZERO, (ZERO,)*5)


def test_saved_source_seal_provenance_and_complete_fraction_query_are_mandatory():
    api, native = _api(), _native()
    transport, work, cold, common, normal, curved = native.source_packets()
    with pytest.raises(ValueError, match="provenance"):
        api.bind_native_common_hot_mean(transport, work, cold, common_hot_packet=common,
            normal_support_packet=normal, curved19_packet=curved, provenance=" ")
    changed = dict(transport, packet_sha256="0"*64)
    with pytest.raises(ValueError, match="immutable source packet seal"):
        api.bind_native_common_hot_mean(changed, work, cold, common_hot_packet=common,
            normal_support_packet=normal, curved19_packet=curved, provenance="rejected test")
    system = native.system()
    for query in ([ZERO]*30, (ZERO,)*29, (0,)+(ZERO,)*29, (F(101),)+(ZERO,)*29):
        with pytest.raises(ValueError, match="30 literal Fraction|amplitude bound"):
            api.evaluate_native_common_hot_mean(system, query)
