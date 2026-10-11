"""Independent checks of the complete smooth cold constitutive extension.

The small-matrix oracle uses literal rational Green-strain polynomials.  Full
source checks read the complete saved authorities and keep their original
material measure, layer cuts, beam ledgers and clocks separate from the new
whole-volume law.  No old cold force solver or registered bulk remainder is
used as a volume integral.
"""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
from functools import lru_cache
from math import factorial
import hashlib
import json

import pytest

from tectonics.geometric_native_pose import FractionInterval as I


ZERO = F(0)
IDENTITY = tuple(tuple(F(int(i == j)) for j in range(3)) for i in range(3))
ZERO_MATRIX = ((ZERO, ZERO, ZERO),)*3


@lru_cache(maxsize=1)
def _api():
    from tectonics import geometric_native_common_cold_continuum_energy as api
    return api


@lru_cache(maxsize=1)
def _native():
    from analysis.geometric_native_common_cold_continuum_energy_validation import native_control
    return native_control


def _fraction(value):
    assert type(value) is list and len(value) == 2 and all(type(v) is int for v in value)
    result = F(*value)
    assert [result.numerator, result.denominator] == value
    return result


def _decode(value):
    if type(value) is dict and set(value) == {"lo", "hi"}:
        return I(_fraction(value["lo"]), _fraction(value["hi"]))
    if type(value) is list:
        return tuple(_decode(v) for v in value)
    return value


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _overlap(actual, expected):
    if type(actual) is I:
        expected = expected if type(expected) is I else I(expected)
        assert max(actual.lo, expected.lo) <= min(actual.hi, expected.hi), (actual, expected)
    elif type(actual) is F:
        if type(expected) is I:
            assert expected.lo <= actual <= expected.hi
        else:
            assert actual == expected
    else:
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _overlap(a, b)


def _contains(actual, expected):
    if type(actual) is I:
        assert actual.lo <= expected <= actual.hi, (actual, expected)
    elif type(actual) is F:
        assert actual == expected
    else:
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _contains(a, b)


def _encloses(actual, expected):
    expected = expected if type(expected) is I else I(expected)
    assert type(actual) is I
    assert actual.lo <= expected.lo and expected.hi <= actual.hi, (actual, expected)


def _transpose(matrix):
    return tuple(tuple(matrix[j][i] for j in range(3)) for i in range(3))


def _multiply(first, second):
    return tuple(tuple(sum((first[i][k]*second[k][j] for k in range(3)), ZERO)
                       for j in range(3)) for i in range(3))


def _add(first, second):
    return tuple(tuple(x+y for x, y in zip(a, b)) for a, b in zip(first, second))


def _scale(matrix, coefficient):
    return tuple(tuple(coefficient*x for x in row) for row in matrix)


def _sym(matrix):
    return _scale(_add(matrix, _transpose(matrix)), F(1, 2))


def _green(gradient):
    deformation = _add(IDENTITY, gradient)
    metric = _multiply(_transpose(deformation), deformation)
    return _scale(_add(metric, _scale(IDENTITY, -1)), F(1, 2))


def _manual_green_twojet(gradient, columns):
    """Exact local value, first variations and all mixed second variations."""
    deformation = _add(IDENTITY, gradient)
    first = tuple(_sym(_multiply(_transpose(deformation), G)) for G in columns)
    second = tuple(tuple(_sym(_multiply(_transpose(G), H)) for H in columns)
                   for G in columns)
    return _green(gradient), first, second


def _tensor_inner(first, second, lame, shear):
    trace_first = sum((first[i][i] for i in range(3)), ZERO)
    trace_second = sum((second[i][i] for i in range(3)), ZERO)
    full = sum((first[i][j]*second[i][j] for i in range(3) for j in range(3)), ZERO)
    return lame*trace_first*trace_second+2*shear*full


def _invert(matrix):
    """Exact independently pivoted elimination for a small literal Gram."""
    count = len(matrix)
    rows = [list(row)+[F(int(i == j)) for j in range(count)]
            for i, row in enumerate(matrix)]
    for column in range(count):
        pivot = next(i for i in range(column, count) if rows[i][column])
        rows[column], rows[pivot] = rows[pivot], rows[column]
        rows[column] = [x/rows[column][column] for x in rows[column]]
        for index in range(count):
            if index == column:
                continue
            coefficient = rows[index][column]
            rows[index] = [x-coefficient*y for x, y in zip(rows[index], rows[column])]
    return tuple(tuple(row[count:]) for row in rows)


def _matrix_inner(first, second):
    return sum((a*b for row_a, row_b in zip(first, second)
                for a, b in zip(row_a, row_b)), ZERO)


def _inverse3(matrix):
    determinant = (matrix[0][0]*(matrix[1][1]*matrix[2][2]-matrix[1][2]*matrix[2][1])
        -matrix[0][1]*(matrix[1][0]*matrix[2][2]-matrix[1][2]*matrix[2][0])
        +matrix[0][2]*(matrix[1][0]*matrix[2][1]-matrix[1][1]*matrix[2][0]))
    cofactors = []
    for i in range(3):
        row = []
        for j in range(3):
            others_i = tuple(k for k in range(3) if k != i)
            others_j = tuple(k for k in range(3) if k != j)
            value = (matrix[others_i[0]][others_j[0]]*matrix[others_i[1]][others_j[1]]
                -matrix[others_i[0]][others_j[1]]*matrix[others_i[1]][others_j[0]])
            row.append((-1)**(i+j)*value/determinant)
        cofactors.append(tuple(row))
    return _transpose(tuple(cofactors))


@lru_cache(maxsize=1)
def _pi_oracle():
    """Machin identity with exact alternating-series outward bounds."""
    def arctangent(denominator):
        value = F(1, denominator)
        total = sum((F((-1)**k)*value**(2*k+1)/(2*k+1) for k in range(90)), ZERO)
        following = total+value**181/181
        return I(min(total, following), max(total, following))
    return 16*arctangent(5)-4*arctangent(239)


def _sin_cos_oracle(value):
    assert type(value) is F and 0 <= value <= F(1, 2)
    sine = sum((F((-1)**k)*value**(2*k+1)/factorial(2*k+1) for k in range(70)), ZERO)
    cosine = sum((F((-1)**k)*value**(2*k)/factorial(2*k) for k in range(70)), ZERO)
    sine_next = sine+value**141/factorial(141)
    cosine_next = cosine+value**140/factorial(140)
    return I(min(sine, sine_next), max(sine, sine_next)), I(min(cosine, cosine_next), max(cosine, cosine_next))


def _synthetic_case():
    gradient = ((F(1, 13), F(-2, 17), F(3, 29)),
                (F(2, 19), F(-1, 31), F(1, 37)),
                (F(-1, 41), F(2, 43), F(1, 47)))
    columns = (((F(1, 5), F(2, 7), ZERO), (ZERO, F(-1, 11), F(1, 13)),
                (F(3, 17), ZERO, F(1, 19))),
               ((ZERO, F(1, 23), F(2, 29)), (F(-1, 31), F(2, 37), ZERO),
                (F(1, 41), F(3, 43), F(-2, 47))))
    return gradient, columns


@lru_cache(maxsize=1)
def _authorities():
    material, work, geometry = _native().source_packets()
    bodies = {b["original_body"]["original_parent_id"]: b["original_body"]
              for b in material["original_cold_body_contexts"]}
    beams = json.loads(work["carrier_source_work_system"]["seed_json"])["beams"]
    return material, work, geometry, bodies, beams


def test_local_green_twojet_matches_independent_literal_matrix_polynomial():
    gradient, columns = _synthetic_case()
    actual = _api()._green_twojet(gradient, columns)
    expected = _manual_green_twojet(gradient, columns)
    _overlap(actual, expected)
    # These entries distinguish the complete quadratic metric from sym(D)
    # and retain mixed columns that a bankwise assembly would discard.
    assert expected[0] != _sym(gradient)
    assert expected[2][0][1] != ZERO_MATRIX
    assert expected[2][0][1] == expected[2][1][0]


def test_local_first_and_cross_second_variations_match_exact_central_differences():
    gradient, columns = _synthetic_case()
    value, first, second = _api()._green_twojet(gradient, columns)
    step = F(7, 31)
    for index, column in enumerate(columns):
        plus = _green(_add(gradient, _scale(column, step)))
        minus = _green(_add(gradient, _scale(column, -step)))
        derivative = _scale(_add(plus, _scale(minus, -1)), 1/(2*step))
        curvature = _scale(_add(_add(plus, minus), _scale(_green(gradient), -2)), 1/(step*step))
        _contains(first[index], derivative)
        _contains(second[index][index], curvature)
    cross = ZERO_MATRIX
    for first_sign, second_sign in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        perturbed = _add(_add(gradient, _scale(columns[0], first_sign*step)),
                         _scale(columns[1], second_sign*step))
        cross = _add(cross, _scale(_green(perturbed), first_sign*second_sign/(4*step*step)))
    _contains(second[0][1], cross)


def test_finite_rigid_left_rotation_preserves_green_value_and_all_variations():
    # A proper exact rational rotation makes this a finite objectivity check,
    # with no infinitesimal-angle or binary64 approximation.
    rotation = ((F(3, 5), F(-4, 5), ZERO),
                (F(4, 5), F(3, 5), ZERO), (ZERO, ZERO, F(1)))
    gradient, columns = _synthetic_case()
    rotated = _add(_multiply(rotation, _add(IDENTITY, gradient)), _scale(IDENTITY, -1))
    rotated_columns = tuple(_multiply(rotation, G) for G in columns)
    base = _api()._green_twojet(gradient, columns)
    transformed = _api()._green_twojet(rotated, rotated_columns)
    _overlap(transformed, base)
    rigid = _add(rotation, _scale(IDENTITY, -1))
    rigid_value, _, _ = _api()._green_twojet(rigid, (ZERO_MATRIX,))
    _contains(rigid_value, ZERO_MATRIX)


def test_full_eight_dimensional_inverse_keeps_authentic_offdiagonal_gram_coupling():
    # Independent Sherman-Morrison result for an exactly specified full SPD
    # metric.  Eight unrelated scalar inverses cannot satisfy this test.
    count = 8
    diagonal = tuple(F(i+2) for i in range(count))
    vector = tuple(F((-1)**i, i+3) for i in range(count))
    matrix = tuple(tuple(diagonal[i]*int(i == j)+vector[i]*vector[j]
                         for j in range(count)) for i in range(count))
    denominator = 1+sum((v*v/d for v, d in zip(vector, diagonal)), ZERO)
    expected = tuple(tuple(F(int(i == j))/diagonal[i]
        -vector[i]*vector[j]/(diagonal[i]*diagonal[j]*denominator)
        for j in range(count)) for i in range(count))
    actual = _api()._inverse_gram(tuple(tuple(I(x) for x in row) for row in matrix), precision=96)
    _contains(actual, expected)
    assert expected == _invert(matrix)
    assert all(expected[i][j] for i in range(count) for j in range(count) if i != j)
    for i in range(count):
        for j in range(count):
            _contains(sum((I(matrix[i][k])*actual[k][j] for k in range(count)), I(0)),
                      F(int(i == j)))
    with pytest.raises(ValueError):
        _api()._inverse_gram(((I(1),)*count,)*count, precision=96)


def test_raw_scalar_channel_order_preserves_full_gram_projection_and_curvature_topology():
    _, first_two = _synthetic_case()
    third = ((F(1, 19), ZERO, F(2, 23)), (ZERO, F(-1, 29), ZERO),
             (F(3, 31), F(1, 37), F(2, 41)))
    columns = first_two+(third,)
    coordinates = (F(1, 17), F(-2, 19), F(3, 23))
    tips = (0, 2)
    lame, shear = F(7, 3), F(5, 7)
    gradient = tuple(tuple(sum((a*G[i][j] for a, G in zip(coordinates, columns)), ZERO)
        for j in range(3)) for i in range(3))
    E, E_first, E_second = _manual_green_twojet(gradient, columns)
    H = tuple(_sym(G) for G in columns)
    initial = ZERO_MATRIX
    for index in tips:
        initial = _add(initial, _scale(H[index], coordinates[index]))
    remainder = _add(E, _scale(initial, -1))
    remainder_first = tuple(_add(value, _scale(H[i], -1)) if i in tips else value
        for i, value in enumerate(E_first))
    pairs = tuple((i, j) for i in range(3) for j in range(i, 3))
    inner = lambda a, b: _tensor_inner(a, b, lame, shear)
    expected = [inner(H[i], H[j]) for k, i in enumerate(tips) for j in tips[k:]]
    expected.extend(inner(H[i], remainder) for i in tips)
    expected.extend(inner(H[i], value) for i in tips for value in remainder_first)
    expected.extend(inner(H[k], E_second[i][j]) for k in tips for i, j in pairs)
    expected.append(inner(remainder, remainder))
    expected.extend(inner(remainder, value) for value in remainder_first)
    expected.extend(inner(remainder_first[i], remainder_first[j])+inner(remainder, E_second[i][j])
        for i, j in pairs)
    actual = _api()._prepared_raw_kernel(columns, coordinates, tips, lame, shear)
    assert len(actual) == len(expected) == 33
    _contains(actual, tuple(expected))
    assert actual[1] != 0  # The off-diagonal Gram channel remains present.
    scale, deviation = F(7, 6), F(1, 9)
    prepared_norm_cap = 1/(scale*(1-deviation))
    assert all(sum((x*x for row in G for x in row), ZERO) <= prepared_norm_cap**2 for G in columns)
    collar = _api().uniform_donor_collar_channels((F(1),)*3, coordinates, tips,
        lame_pa=lame, shear_pa=shear, fixed_scale=scale, carrier_deviation=deviation, precision=96)
    assert len(collar) == 33
    _contains(collar, actual)
    assert collar[1].lo < 0 < collar[1].hi


def test_directed_binary_box_conversion_and_basic_arithmetic_enclose_exact_rational_results():
    algebra = _api()._BoxAlgebra(I(0), 96)
    for value in (F(0), F(1, 10), F(-1, 10), F(2**54+1, 2**54),
                  F(-2**54-1, 2**54), F(1, 2**1100), F(-1, 2**1100)):
        box = algebra.constant(value)
        exact = box.to_interval()
        assert exact == I(F.from_float(box.lo), F.from_float(box.hi))
        _encloses(exact, value)
    first = I(F(-2, 7), F(3, 11))
    second = I(F(-5, 13), F(7, 17))
    negative_denominator = I(F(-7, 5), F(-3, 7))
    a, b, denominator = (algebra.constant(value) for value in (first, second, negative_denominator))
    _encloses((a+b).to_interval(), first+second)
    _encloses((a-b).to_interval(), first-second)
    _encloses((a*b).to_interval(), first*second)
    _encloses((a/denominator).to_interval(), first/negative_denominator)
    assert (algebra.constant(0)+algebra.constant(0)).to_interval() == I(0)
    assert (a*algebra.constant(0)).to_interval() == I(0)
    with pytest.raises(ValueError):
        a/algebra.constant(I(F(-1, 3), F(1, 5)))
    # Equal interval bounds can denote independent material expressions.
    # Caching them to one wrapper must not invent a dependency and turn this
    # product into a square: its true range contains negative values.
    independent_a = algebra.constant(I(-1, 1))
    independent_b = algebra.constant(I(-1, 1))
    _encloses((independent_a*independent_b).to_interval(), I(-1, 1))
    square = (independent_a**2).to_interval()
    _encloses(square, I(0, 1))
    assert square.lo >= 0


def test_binary_box_transcendental_endpoints_retain_independent_rational_outward_bounds():
    algebra = _api()._BoxAlgebra(I(F(1, 100), F(1, 50)), 96)
    lower_sine, lower_cosine = _sin_cos_oracle(F(1, 100))
    upper_sine, upper_cosine = _sin_cos_oracle(F(1, 50))
    _encloses(algebra.sine_theta().to_interval(), I(lower_sine.lo, upper_sine.hi))
    _encloses(algebra.cosine_theta().to_interval(), I(upper_cosine.lo, lower_cosine.hi))
    sine, cosine = _sin_cos_oracle(F(1, 10))
    signed = algebra.constant(I(F(-1, 10), F(1, 10)))
    _encloses(algebra.sin(signed).to_interval(), I(-sine.hi, sine.hi))
    _encloses(algebra.cos(signed).to_interval(), I(cosine.lo, F(1)))
    _encloses(algebra.cbrt(algebra.constant(I(F(1, 8), F(27, 8)))).to_interval(), I(F(1, 2), F(3, 2)))


def test_complete_saved_authorities_keep_original_56_coordinates_and_layer_boundaries():
    material, work, geometry, bodies, beams = _authorities()
    for packet in (material, work, geometry, _native().material_packet()):
        content = {key: value for key, value in packet.items() if key != "packet_sha256"}
        assert hashlib.sha256(_encoded(content)).hexdigest() == packet["packet_sha256"]
    assert set(bodies) == {"octant:"+str(index) for index in range(8)}
    carrier = geometry["cold_material_carrier_system"]
    order = tuple(tuple(row) for row in carrier["coordinate_order"])
    coordinates = tuple(_fraction(x) for x in carrier["bound_coordinate_increments"])
    assert len(order) == len(coordinates) == 56 and all(coordinates)
    assert len(set(order)) == 56
    authentic = work["carrier_source_work_system"]
    original_order = tuple(tuple(row) for row in authentic["coordinate_order"])
    original_coordinates = tuple(_fraction(x) for x in authentic["bound_coordinate_increments"])
    assert coordinates == tuple(original_coordinates[original_order.index(row)] for row in order)
    for parent, body in bodies.items():
        radius, roof, physical = (_fraction(body[key]) for key in
            ("radius_m", "roof_radial_displacement_m", "elastic_thickness_m"))
        top = radius+roof
        exact_cut = (top**3-(top-physical)**3)/(3*radius**2)
        assert _fraction(body["elastic_layer_areal_depth_m"]) == exact_cut
        assert exact_cut != physical
        assert 0 < exact_cut < _fraction(body["areal_material_thickness_m"])
    assert {beam["original_parent_id"] for beam in beams} == set(bodies)


def test_eight_literal_beam_ledgers_preserve_binary_residuals_and_global_once_only_ownership():
    _, _, _, bodies, beams = _authorities()
    assert len(beams) == len({beam["component_id"] for beam in beams}) == 8
    assert len({beam["source_coordinate_index"] for beam in beams}) == 8
    energy_residuals, force_residuals = [], []
    for beam in beams:
        body = bodies[beam["original_parent_id"]]
        stiffness = _fraction(beam["stiffness_exact_n_m"])
        displacement = _fraction(beam["reference_displacement_m"])
        assert stiffness == _fraction(body["source_old_tip_stiffness_n_m"])
        assert displacement == _fraction(body["accepted_tip_displacement_m"])
        energy = _fraction(beam["literal_reduced_beam_binary_energy_evaluation_j"])
        force = _fraction(beam["literal_reduced_beam_binary_force_evaluation_n"])
        energy_residual = energy-stiffness*displacement*displacement/2
        force_residual = force-stiffness*displacement
        assert energy_residual == _fraction(beam["binary_energy_evaluation_minus_exact_j"])
        assert force_residual == _fraction(beam["binary_force_evaluation_minus_exact_n"])
        energy_residuals.append(energy_residual)
        force_residuals.append(force_residual)
        assert beam["energetic_owner"] == "INTERNAL_SOURCE_BEAMS_REPLACE_COMMON_CALIBRATED_BEAM_ENERGY"
        assert not beam["archived_old_binary_force_replayed"]
    # The tiny literal residuals are real authority data.  Replacing them by
    # zero would change both the energy baseline and work-conjugate tip force.
    assert any(energy_residuals) and any(force_residuals)


def test_bound_whole_material_preserves_full_layer_authorities_measure_and_clocks():
    api, native = _api(), _native()
    system = native.system()
    material, work, geometry, bodies, beams = _authorities()
    old_materials = {row["material"]["component_id"]: row["material"]
                     for row in native.material_packet()["cold_components"]}
    assert len(system.banks) == system.original_parent_count == system.source_beam_count == 8
    assert system.nonempty_archive_cohort_count == 3
    assert system.source_beams_included_once
    assert not system.registered_old_bulk_remainders_used_as_continuum_integral
    assert system.material_reference_measure == "AUTHENTIC_ORIGINAL_AREAL_VOLUME_R_SQUARED_DOMEGA_DZ_ONCE"
    assert system.source_packet_seals == tuple(packet["packet_sha256"]
        for packet in (material, work, geometry, native.material_packet()))
    assert system.complete_original_material_state_json == material["material_transfer"]["complete_original_material_state_json"]
    assert json.loads(system.complete_source_carrier_certificate_json) == geometry[
        "cold_material_carrier_system"]["whole_domain_certificate"]
    source_seed = json.loads(work["carrier_source_work_system"]["seed_json"])
    assert json.loads(system.complete_source_clock_work_history_json)[
        "unchanged_source_clocks_and_work_ledger_json"] == source_seed[
            "unchanged_source_clocks_and_work_ledger_json"]
    for bank in system.banks:
        body = bodies[bank.original_parent_id]
        old = old_materials[bank.component_id]
        assert bank.upper_young_modulus_pa == _fraction(body["young_modulus_pa"])
        assert bank.upper_poisson_ratio == _fraction(body["poisson_ratio"])
        assert bank.lower_young_modulus_pa == _fraction(old["lower_young_modulus_pa"])
        assert bank.lower_shear_modulus_pa == _fraction(old["lower_shear_modulus_pa"])
        assert bank.elastic_layer_areal_depth_m == _fraction(body["elastic_layer_areal_depth_m"])
        assert json.loads(bank.complete_original_body_context_json)["original_body"] == body
        assert json.loads(bank.complete_original_cold_strain_material_json) == old
        assert json.loads(bank.complete_source_beam_json) == next(
            beam for beam in beams if beam["component_id"] == bank.component_id)
        # The complete original parcel, including authentic mass reservoirs
        # and density excess, remains in this original-parent authority.
        parent = json.loads(body["complete_parent_source_json"])
        assert parent["parcel"]["material_id"] == bank.original_material_id
        assert parent["fragment_id"] == bank.original_parent_id
        for layer in ("upper", "lower"):
            lame, shear = bank.lame_coefficients(layer)
            assert shear > 0 and 3*lame+2*shear > 0
            diagonal = ((F(2), ZERO, ZERO), (ZERO, F(-1), ZERO), (ZERO, ZERO, F(3)))
            shear_strain = ((ZERO, F(2), ZERO), (F(2), ZERO, ZERO), ZERO_MATRIX[2])
            assert api._tensor_inner(diagonal, shear_strain, lame, shear) == ZERO
            assert api._tensor_inner(shear_strain, shear_strain, lame, shear) == 16*shear
    assert (system.source_geometry_elapsed_time_s, system.contact_geometry_elapsed_time_s,
        system.memory_elapsed_time_s, system.material_geometry_elapsed_time_s) == (
        F(0), F(2000000), F(1000000), F(1000000))
    for attribute in ("historical_full_three_dimensional_prestress_recovered",
        "historical_archived_strain_energy_density_recovered", "new_reference_preparation_work_certified",
        "physical_source_force_prestress_work_handover_certified", "hot_constitutive_law_defined",
        "finite_mean_equilibrium_certified", "new_mixed_force_equilibrium_certified",
        "whole_clock_all_pair_admission_authorized", "physical_material_admission_authorized"):
        assert getattr(system, attribute) is False
    assert system.accepted_new_native_intervals == 0


@lru_cache(maxsize=2)
def _seam_point(parent="octant:1"):
    return _api().bound_native_common_cold_continuum_point(_native().system(), parent,
        (F(-1), F(0), F(100)), I(5000), precision=96)


def test_prepared_reference_pullback_and_all_actual_cross_seam_green_columns_are_retained():
    from tectonics import geometric_native_common_cold_material_carrier as carrier_api
    api, system = _api(), _native().system()
    cold, _ = api._registered(system)
    parent, ray, depth = "octant:1", (F(-1), F(0), F(100)), I(5000)
    prepared = carrier_api.evaluate_native_common_cold_material_point(cold, parent, ray, depth,
        coordinate_increments=(F(0),)*56, precision=96)
    actual = _seam_point()
    _overlap(actual.prepared_reference_gradient_bounds, prepared.original_cartesian_gradient_bounds)
    inverse = _inverse3(prepared.original_cartesian_gradient_bounds)
    _overlap(actual.prepared_reference_inverse_bounds, inverse)
    expected_columns = tuple(_multiply(G, inverse) for G in prepared.mode_original_gradient_columns_per_m)
    _overlap(actual.mode_relative_gradient_columns_per_m, expected_columns)
    expected_D = tuple(tuple(sum((a*G[i][j] for a, G in zip(system.bound_coordinate_increments,
        expected_columns)), I(0)) for j in range(3)) for i in range(3))
    _overlap(actual.relative_gradient_bounds, expected_D)
    expected_E = _green(expected_D)
    _overlap(actual.green_strain_bounds, expected_E)
    left = next(b for b in system.banks if b.original_parent_id == "octant:1")
    right = next(b for b in system.banks if b.original_parent_id == "octant:3")
    i, j = left.tip_coordinate_index, right.tip_coordinate_index
    _, first, second = _manual_green_twojet(expected_D, (expected_columns[i], expected_columns[j]))
    _overlap(actual.green_strain_gradient_per_m[i], first[0])
    _overlap(actual.green_strain_gradient_per_m[j], first[1])
    _overlap(actual.green_strain_hessian_per_m2[i][j], second[0][1])
    assert any(x.lo > 0 or x.hi < 0 for row in second[0][1] for x in row)
    # The physical chart and all56 columns agree on either parent label of
    # this common material seam.  The two source banks remain distinct.
    other = _seam_point("octant:3")
    _overlap(actual.prepared_reference_gradient_bounds, other.prepared_reference_gradient_bounds)
    _overlap(actual.mode_relative_gradient_columns_per_m, other.mode_relative_gradient_columns_per_m)
    _overlap(actual.green_strain_bounds, other.green_strain_bounds)


def test_whole_point_keeps_inactive_pole_and_literal_layer_split_without_changing_source_mass():
    api, system = _api(), _native().system()
    bank = next(b for b in system.banks if b.original_parent_id == "octant:3")
    pole = api.bound_native_common_cold_continuum_point(system, bank.original_parent_id,
        (F(-1), F(0), F(0)), I(5000), precision=96)
    _contains(pole.relative_gradient_bounds, ZERO_MATRIX)
    _contains(pole.green_strain_bounds, ZERO_MATRIX)
    _contains(pole.mode_relative_gradient_columns_per_m, (ZERO_MATRIX,)*56)
    ray = (F(-1), F(0), F(100))
    cut = bank.elastic_layer_areal_depth_m
    upper = api.bound_native_common_cold_continuum_point(system, bank.original_parent_id,
        ray, I(cut), precision=96)
    lower = api.bound_native_common_cold_continuum_point(system, bank.original_parent_id,
        ray, I(cut+F(1, 1000)), precision=96)
    assert (upper.lame_coefficient_pa, upper.shear_modulus_pa) == bank.lame_coefficients("upper")
    assert (lower.lame_coefficient_pa, lower.shear_modulus_pa) == bank.lame_coefficients("lower")
    with pytest.raises(ValueError, match="Split|layer"):
        api.bound_native_common_cold_continuum_point(system, bank.original_parent_id,
            ray, I(cut-F(1, 1000), cut+F(1, 1000)), precision=96)
    assert system.accepted_new_native_intervals == 0


@lru_cache(maxsize=1)
def _response():
    return _native().response()


@lru_cache(maxsize=1)
def _zero_response():
    return _native().response((F(0),)*56)


def _literal_beam_values(system):
    rows = tuple(json.loads(bank.complete_source_beam_json) for bank in system.banks)
    energy = tuple(_fraction(row["literal_reduced_beam_binary_energy_evaluation_j"]) for row in rows)
    force = tuple(_fraction(row["literal_reduced_beam_binary_force_evaluation_n"]) for row in rows)
    return energy, force


def test_actual_full_volume_gram_has_verified_inverse_and_nonzero_within_owner_overlaps():
    system, response = _native().system(), _response()
    matrix = response.integrals.gram_bounds_n_m
    certificate = response.gram_inverse_certificate
    assert len(matrix) == 8 and all(len(row) == 8 for row in matrix)
    assert matrix == certificate.gram_bounds_n_m
    assert certificate.positive_definite_certified
    assert certificate.global_eight_tip_overlap_retained
    assert certificate.preconditioner_residual_infinity_norm_upper >= 0
    assert certificate.inverse_enclosure_error_bound_m_n >= 0
    inverse = certificate.inverse_gram_bounds_m_n
    midpoint = tuple(tuple(x.midpoint for x in row) for row in matrix)
    assert certificate.rational_midpoint_preconditioner_m_n == _invert(midpoint)
    preconditioner = certificate.rational_midpoint_preconditioner_m_n
    if certificate.inverse_enclosure_method == "INTERVAL_LDL_AND_RATIONAL_NEUMANN_RESIDUAL":
        assert len(certificate.interval_ldl_positive_pivots_n_m) == 8
        assert all(pivot.lo > 0 for pivot in certificate.interval_ldl_positive_pivots_n_m)
        assert certificate.preconditioner_residual_infinity_norm_upper < 1
        residual = tuple(tuple(I(int(i == j))-sum((preconditioner[i][k]*matrix[k][j]
            for k in range(8)), I(0)) for j in range(8)) for i in range(8))
        norm = max(sum((max(abs(x.lo), abs(x.hi)) for x in row), ZERO) for row in residual)
        assert certificate.preconditioner_residual_infinity_norm_upper >= norm
        C_norm = max(sum((abs(x) for x in row), ZERO) for row in preconditioner)
        assert certificate.inverse_enclosure_error_bound_m_n >= norm*C_norm/(1-norm)
    else:
        assert certificate.inverse_enclosure_method == "ACTUAL_INTERIOR_GRAM_COERCIVITY_AND_RIGHT_RESIDUAL_NORM"
        assert certificate.interval_ldl_positive_pivots_n_m == ()
        lower = certificate.authentic_interior_gram_coercivity_lower_bounds_n_m
        assert lower == response.integrals.authentic_interior_gram_coercivity_lower_bounds_n_m
        assert len(lower) == 8 and all(type(value) is F and value > 0 for value in lower)
        receipts = response.integrals.quadrature_certificate.cell_receipts
        for index, bank in enumerate(system.banks):
            # Interior cells have only their own tip.  Their strict norm
            # integrals give a physical diagonal lower bound; the remaining
            # material contributes a positive semidefinite Gram.  A chosen
            # numerical floor cannot replace these authentic volume cells.
            exact_lower = sum((max(ZERO, row["interior_tip_gram_bounds_n_m"].lo) for row in receipts
                if row["parent_id"] == bank.original_parent_id and row["theta_piece"] == "interior"
                and row["kind"] != "whole_donor_activation_collar"), ZERO)
            assert lower[index] == exact_lower
            assert all(row["interior_tip_gram_global_index"] == index for row in receipts
                if row["parent_id"] == bank.original_parent_id and row["theta_piece"] == "interior"
                and row["kind"] != "whole_donor_activation_collar")
        residual = tuple(tuple(I(int(i == j))-sum((matrix[i][k]*preconditioner[k][j]
            for k in range(8)), I(0)) for j in range(8)) for i in range(8))
        errors, norms = [], []
        for owner in {bank.owner for bank in system.banks}:
            indices = tuple(i for i, bank in enumerate(system.banks) if bank.owner == owner)
            assert len(indices) == 4
            norm = max(sum((max(abs(residual[i][j].lo), abs(residual[i][j].hi))
                for j in indices), ZERO) for i in indices)
            norms.append(norm)
            errors.append(2*norm/min(lower[i] for i in indices))
        assert certificate.preconditioner_residual_infinity_norm_upper >= max(norms)
        assert certificate.inverse_enclosure_error_bound_m_n >= max(errors)
    nonzero_owners = set()
    for i in range(8):
        for j in range(8):
            assert matrix[i][j] == matrix[j][i]
            _contains(sum((matrix[i][k]*inverse[k][j] for k in range(8)), I(0)), F(int(i == j)))
            if system.banks[i].owner != system.banks[j].owner:
                assert matrix[i][j] == inverse[i][j] == I(0)
            elif i != j and matrix[i][j] != I(0):
                nonzero_owners.add(system.banks[i].owner)
    assert nonzero_owners == {0, 1}


def test_zero_prepared_increment_preserves_all_eight_literal_beams_and_nonlinear_prestress_curvature():
    system, response = _native().system(), _zero_response()
    raw = response.integrals
    energy, forces = _literal_beam_values(system)
    assert response.old_beam_reference_energy_j == sum(energy, ZERO)
    assert response.old_beam_reference_tip_forces_n == forces
    assert response.energy_j == response.beam_energy_j == I(sum(energy, ZERO))
    assert response.orthogonal_continuum_energy_j == I(0)
    assert response.tip_projection_increment_m == (I(0),)*8
    assert raw.remainder_strain_inner_product_j == I(0)
    assert raw.projection_remainder_n == (I(0),)*8
    inverse = response.gram_inverse_certificate.inverse_gram_bounds_m_n
    curvature_seen = False
    for i, bank_i in enumerate(system.banks):
        assert response.gradient_n[bank_i.tip_coordinate_index] == I(forces[i])
        for j, bank_j in enumerate(system.banks):
            a, b = bank_i.tip_coordinate_index, bank_j.tip_coordinate_index
            geometric = sum((forces[k]*sum((inverse[k][m]*raw.projection_hessian_n_m2[m][a][b]
                for m in range(8)), I(0)) for k in range(8)), I(0))
            stiffness = bank_i.source_old_tip_stiffness_n_m*int(i == j)
            _overlap(response.hessian_n_m[a][b], I(stiffness)+geometric)
            curvature_seen |= geometric != I(0)
    assert curvature_seen
    assert response.nonlinear_prestress_curvature_retained
    assert not response.old_beam_stiffness_added_as_second_energy


def test_whole_energy_and_56_derivatives_obey_the_independent_global_projection_law():
    system, response = _native().system(), _response()
    raw = response.integrals
    inverse = response.gram_inverse_certificate.inverse_gram_bounds_m_n
    reference_energy, reference_force = _literal_beam_values(system)
    projected_remainder = tuple(sum((inverse[i][j]*raw.projection_remainder_n[j]
        for j in range(8)), I(0)) for i in range(8))
    projected = tuple(I(response.coordinate_increments[bank.tip_coordinate_index])+r
        for bank, r in zip(system.banks, projected_remainder))
    _overlap(response.tip_projection_increment_m, projected)
    beam = sum((I(reference_energy[i])+reference_force[i]*p
        +system.banks[i].source_old_tip_stiffness_n_m*p*p/2 for i, p in enumerate(projected)), I(0))
    complement = (raw.remainder_strain_inner_product_j-sum((r*p for r, p in
        zip(raw.projection_remainder_n, projected_remainder)), I(0)))/2
    _overlap(response.beam_energy_j, beam)
    _overlap(response.orthogonal_continuum_energy_j, complement)
    _overlap(response.energy_j, beam+complement)
    assert response.orthogonal_continuum_energy_j.lo >= 0
    jacobian_remainder = tuple(tuple(sum((inverse[k][m]*raw.projection_remainder_gradient_n_m[m][i]
        for m in range(8)), I(0)) for i in range(56)) for k in range(8))
    jacobian = tuple(tuple(value+int(i == bank.tip_coordinate_index) for i, value in enumerate(row))
        for bank, row in zip(system.banks, jacobian_remainder))
    projected_force = tuple(I(reference_force[k])+bank.source_old_tip_stiffness_n_m*projected[k]
        for k, bank in enumerate(system.banks))
    for i in range(56):
        expected = raw.remainder_local_gradient_n[i]+sum((projected_force[k]*jacobian[k][i]
            -projected_remainder[k]*raw.projection_remainder_gradient_n_m[k][i]
            for k in range(8)), I(0))
        _overlap(response.gradient_n[i], expected)
    sample = tuple(bank.tip_coordinate_index for bank in system.banks)+(system.banks[1].coordinate_indices[4],)
    for i in sample:
        for j in sample:
            expected = raw.remainder_local_hessian_n_m[i][j]
            for k, bank in enumerate(system.banks):
                projection_second = sum((inverse[k][m]*raw.projection_hessian_n_m2[m][i][j]
                    for m in range(8)), I(0))
                expected += (bank.source_old_tip_stiffness_n_m*jacobian[k][i]*jacobian[k][j]
                    +projected_force[k]*projection_second
                    -raw.projection_remainder_gradient_n_m[k][i]*jacobian_remainder[k][j]
                    -projected_remainder[k]*raw.projection_hessian_n_m2[k][i][j])
            _overlap(response.hessian_n_m[i][j], expected)
    assert len(response.gradient_n) == len(response.hessian_n_m) == 56
    assert all(len(row) == 56 for row in response.hessian_n_m)
    assert all(response.hessian_n_m[i][j] == response.hessian_n_m[j][i]
        for i in range(56) for j in range(56))
    # Actual seam couplings join different banks of the same owner, while
    # separate cold owner volumes do not invent a cross-owner bulk tangent.
    assert any(response.hessian_n_m[a.tip_coordinate_index][b.tip_coordinate_index] != I(0)
        for a in system.banks for b in system.banks
        if a.original_parent_id != b.original_parent_id and a.owner == b.owner)
    assert response.full_fifty_six_cold_gradient_and_hessian_included
    assert response.seam_cross_bank_derivatives_retained
    assert response.accepted_new_native_intervals == 0


def test_typed_guards_reject_changed_layers_beams_clocks_coordinates_and_resealed_receipts():
    api, system = _api(), _native().system()
    bank = system.banks[0]
    for altered in (
        replace(system, banks=(replace(bank, upper_young_modulus_pa=2*bank.upper_young_modulus_pa),)+system.banks[1:]),
        replace(system, banks=(replace(bank, source_reference_force_n=bank.source_reference_force_n+1),)+system.banks[1:]),
        replace(system, source_beams_included_once=1),
        replace(system, material_geometry_elapsed_time_s=system.material_geometry_elapsed_time_s+1),
        replace(system, complete_original_material_state_json="{}"),
    ):
        with pytest.raises(ValueError):
            api.native_common_cold_continuum_energy_to_dict(altered)
    point = _seam_point()
    for altered in (replace(point, precision=True), replace(point, coordinate_increments=(0,)*56),
                    replace(point, fixed_radial_scale=point.fixed_radial_scale+1)):
        with pytest.raises(ValueError):
            api.native_common_cold_continuum_strain_point_to_dict(altered)
    response = _response()
    changed = replace(response, energy_j=response.energy_j+1)
    changed = replace(changed, binding_sha256=api._digest(api._body(changed)))
    with pytest.raises(ValueError):
        api.native_common_cold_continuum_energy_response_to_dict(changed)
    raw = replace(response.integrals, projection_remainder_n=(
        response.integrals.projection_remainder_n[0]+1,)+response.integrals.projection_remainder_n[1:])
    raw = replace(raw, binding_sha256=api._digest(api._body(raw)))
    with pytest.raises(ValueError):
        api.validate_native_common_cold_continuum_integral(raw, system=system)
    outside = list(system.bound_coordinate_increments)
    outside[0] = 2*system.coordinate_envelope_abs_m[0]
    for coordinates in (tuple(outside), list(system.bound_coordinate_increments), (0,)*56,
                        system.bound_coordinate_increments[:-1]):
        with pytest.raises(ValueError):
            api.bound_native_common_cold_continuum_point(system, "octant:1",
                (F(-1), F(0), F(100)), I(5000), coordinate_increments=coordinates)
    with pytest.raises(ValueError):
        api.bound_native_common_cold_continuum_point(system, "octant:1",
            (F(-1), F(0), F(100)), I(5000), precision=True)


def test_complete_system_and_response_restarts_retain_authorities_and_reject_transplants():
    api, native = _api(), _native()
    material, work, geometry = native.source_packets()
    authorities = dict(material_transport_packet=material, source_work_packet=work,
        material_carrier_packet=geometry, cold_carrier=native.cold_carrier(),
        cold_tensor_packet=native.material_packet())
    system_dto = api.native_common_cold_continuum_energy_to_dict(native.system())
    restored = api.native_common_cold_continuum_energy_from_dict(system_dto, **authorities)
    assert api.native_common_cold_continuum_energy_to_dict(restored) == system_dto
    response_dto = api.native_common_cold_continuum_energy_response_to_dict(_response())
    restarted = api.native_common_cold_continuum_energy_response_from_dict(response_dto,
        system=restored, expected_coordinate_increments=restored.bound_coordinate_increments,
        **native.QUADRATURE_OPTIONS)
    assert api.native_common_cold_continuum_energy_response_to_dict(restarted) == response_dto
    with pytest.raises(ValueError):
        api.native_common_cold_continuum_energy_response_from_dict(response_dto,
            system=restored, expected_coordinate_increments=(F(0),)*56, **native.QUADRATURE_OPTIONS)
    changed = deepcopy(system_dto)
    changed["memory_elapsed_time_s"] = [1000001, 1]
    body = {key: value for key, value in changed.items() if key not in ("format", "binding_sha256")}
    changed["binding_sha256"] = hashlib.sha256(_encoded(body)).hexdigest()
    with pytest.raises(ValueError):
        api.native_common_cold_continuum_energy_from_dict(changed, **authorities)
    changed = deepcopy(system_dto)
    changed["source_beams_included_once"] = 1
    with pytest.raises(ValueError):
        api.native_common_cold_continuum_energy_from_dict(changed, **authorities)
    changed_authority = deepcopy(work)
    changed_authority["clocks_advanced"] = True
    with pytest.raises(ValueError):
        api.bind_native_common_cold_continuum_energy(material, changed_authority, geometry,
            cold_carrier=native.cold_carrier(), cold_tensor_packet=native.material_packet(),
            provenance="Changed clock authority must fail")


def test_global_quadrature_certificate_covers_each_parent_layer_seam_and_signed_donor_collar():
    api, system, response = _api(), _native().system(), _response()
    certificate = response.integrals.quadrature_certificate
    cold, _ = api._registered(system)
    assert certificate.material_reference_measure == system.material_reference_measure
    assert certificate.complete_original_parent_ids == tuple(bank.original_parent_id for bank in system.banks)
    assert certificate.nonempty_archive_domain_ids == cold.whole_domain_certificate.complete_archive_domain_ids
    assert len(certificate.nonempty_archive_domain_ids) == 3
    assert certificate.exact_layer_cuts_m == tuple((bank.original_parent_id,
        bank.elastic_layer_areal_depth_m, bank.areal_material_thickness_m) for bank in system.banks)
    expected_volume = _pi_oracle()*sum((bank.radius_m**2*bank.areal_material_thickness_m
        for bank in system.banks), ZERO)/2
    _overlap(certificate.whole_source_volume_bounds_m3, expected_volume)
    expected_active_volume = sum((_pi_oracle()/2*bank.radius_m**2*bank.areal_material_thickness_m
        *_sin_cos_oracle(next(b.beta_cut_radians for b in cold.banks
            if b.original_parent_id == bank.original_parent_id))[0] for bank in system.banks), I(0))
    _overlap(certificate.active_cold_source_volume_bounds_m3, expected_active_volume)
    assert certificate.active_cold_source_volume_bounds_m3.hi < certificate.whole_source_volume_bounds_m3.lo
    for name in ("whole_domain_enclosure_certified", "donor_collar_uniform_enclosure_certified",
        "authentic_original_volume_included_once", "zero_clamp_complement_retained",
        "whole_interval_box_range_certified", "directed_ieee_basic_arithmetic_certified",
        "transcendental_endpoints_rationally_certified"):
        assert getattr(certificate, name) is True
    assert certificate.integration_method == "whole_interval"
    assert certificate.analytic_seam_theta_moments_certified is False
    assert certificate.taylor_order == 0
    assert certificate.beta_cell_count == max(8, _native().QUADRATURE_OPTIONS["beta_cell_count"])
    assert certificate.depth_cell_count == max(2, _native().QUADRATURE_OPTIONS["depth_cell_count"])
    assert certificate.theta_cell_count == _native().QUADRATURE_OPTIONS["theta_cell_count"]
    assert certificate.gauss_difference_used_as_error_certificate is False
    assert certificate.historical_archived_prestress_or_preparation_work_certified is False
    receipts = certificate.cell_receipts
    assert certificate.total_cell_count == len(receipts)
    assert certificate.cell_raw_bounds_archived is False
    assert certificate.cell_raw_bounds_digest_algorithm == "SHA256_CANONICAL_EXACT_DTO_FULL_RAW_BOUNDS_AND_CELL_PROOF_PARAMETERS"
    assert certificate.maximum_absolute_raw_enclosure_width == max(
        row["maximum_absolute_raw_enclosure_width"] for row in receipts)
    assert certificate.maximum_absolute_raw_enclosure_width > 0
    collar_rows = tuple(row for row in receipts if row["kind"] == "whole_donor_activation_collar")
    main_rows = tuple(row for row in receipts if row["kind"] == "whole_interval_theta_beta_depth")
    assert collar_rows and main_rows
    assert len(collar_rows)+len(main_rows) == len(receipts)
    assert len(main_rows) == 8*2*3*certificate.theta_cell_count*certificate.beta_cell_count*certificate.depth_cell_count
    assert len(collar_rows) == 4*2*3
    assert all(row["whole_piece_uniform_bound"] is True for row in receipts)
    assert all(row["directed_ieee_basic_arithmetic"] is True for row in main_rows)
    assert all(row["transcendental_endpoints_rationally_certified"] is True for row in main_rows)
    assert any(row["maximum_absolute_raw_enclosure_width"] > 0 for row in collar_rows)
    for row in receipts:
        assert "integral_bounds" not in row
        count, tip_count = len(row["active_indices"]), len(row["local_tip_indices"])
        assert count in (7, 14) and tip_count in (1, 2)
        pair_count = count*(count+1)//2
        assert row["raw_channel_count"] == tip_count*(tip_count+1)//2+tip_count+tip_count*count+tip_count*pair_count+1+count+pair_count
        seal = row["complete_cell_receipt_canonical_sha256"]
        assert type(seal) is str and len(seal) == 64 and all(character in "0123456789abcdef" for character in seal)
        assert type(row["maximum_absolute_raw_enclosure_width"]) is F
    raw = response.integrals
    assert len(raw.projection_remainder_n) == len(raw.projection_remainder_gradient_n_m) == len(raw.projection_hessian_n_m2) == 8
    assert all(len(row) == 56 for row in raw.projection_remainder_gradient_n_m)
    assert all(len(block) == 56 and all(len(row) == 56 for row in block) for block in raw.projection_hessian_n_m2)
    assert type(raw.remainder_strain_inner_product_j) is I
    assert len(raw.remainder_local_gradient_n) == len(raw.remainder_local_hessian_n_m) == 56
    assert all(len(row) == 56 for row in raw.remainder_local_hessian_n_m)
    assert all(row["source_volume_bounds_m3"].lo >= 0 for row in receipts)
    _overlap(sum((row["source_volume_bounds_m3"] for row in receipts), I(0)), expected_active_volume)

    def complete_boxes(rows, zlo, zhi, blo, bhi, theta_start, theta_stop):
        assert rows
        depth_boxes = sorted({row["depth_bounds"] for row in rows}, key=lambda box: box.lo)
        assert depth_boxes[0].lo == zlo and depth_boxes[-1].hi == zhi
        assert all(left.hi == right.lo for left, right in zip(depth_boxes, depth_boxes[1:]))
        for depth in depth_boxes:
            angles = sorted({(row["integration_angle_lo"], row["integration_angle_hi"])
                for row in rows if row["depth_bounds"] == depth}, key=lambda pair: pair[0].lo)
            _overlap(angles[0][0], theta_start)
            _overlap(angles[-1][1], theta_stop)
            assert all(left[1] == right[0] for left, right in zip(angles, angles[1:]))
            for angle_start, angle_stop in angles:
                beta_boxes = sorted((row["beta_bounds"] for row in rows if row["depth_bounds"] == depth
                    and row["integration_angle_lo"] == angle_start and row["integration_angle_hi"] == angle_stop),
                    key=lambda box: box.lo)
                assert beta_boxes[0].lo == blo and beta_boxes[-1].hi == bhi
                assert all(left.hi == right.lo for left, right in zip(beta_boxes, beta_boxes[1:]))
    # This checks the complete traversal, including the formerly omitted
    # donor activation collar, rather than assigning sample labels volume.
    for bank in system.banks:
        oldbank = next(b for b in cold.banks if b.original_parent_id == bank.original_parent_id)
        for layer, zlo, zhi in (("upper", F(0), bank.elastic_layer_areal_depth_m),
                               ("lower", bank.elastic_layer_areal_depth_m, bank.areal_material_thickness_m)):
            for piece in ("start_seam", "interior", "end_seam"):
                rows = tuple(row for row in main_rows if row["parent_id"] == bank.original_parent_id
                             and row["layer"] == layer and row["theta_piece"] == piece)
                delta = cold.whole_domain_certificate.seam_half_width_radians
                theta_start = I(delta) if piece == "interior" else I(0)
                theta_stop = _pi_oracle()/2-delta if piece == "interior" else I(delta)
                donor = bank.owner == cold.whole_domain_certificate.donor.owner
                collar_end = certificate.donor_activation_collar_beta_upper_radians
                complete_boxes(rows, zlo, zhi, collar_end if donor else F(0),
                    oldbank.beta_cut_radians, theta_start, theta_stop)
                collars = tuple(row for row in collar_rows if row["parent_id"] == bank.original_parent_id
                    and row["layer"] == layer and row["theta_piece"] == piece)
                if donor:
                    complete_boxes(collars, zlo, zhi, F(0), collar_end, theta_start, theta_stop)
                else:
                    assert not collars
    for invalid in (dict(max_cells=0), dict(beta_cell_count=0), dict(depth_cell_count=True),
                    dict(taylor_order=25), dict(integration_method="sampled"), dict(theta_cell_count=True)):
        with pytest.raises(ValueError):
            api.integrate_native_common_cold_continuum_response(system, **invalid)
