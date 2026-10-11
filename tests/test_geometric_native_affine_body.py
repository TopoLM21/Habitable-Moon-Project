"""Independent whole-body geometry checks for the genuine five HOT masks.

The reference-map tests query its declared analytic extension.  A query at a
unit ray does not assert that ray's membership in an implicit material mask.
"""
from dataclasses import replace
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from functools import lru_cache
import json
import math

import pytest

from tectonics.geometric_native_pose import FractionInterval as I


ZERO = F(0)
IDENTITY = ((F(1), ZERO, ZERO), (ZERO, F(1), ZERO), (ZERO, ZERO, F(1)))


def _api():
    from tectonics import geometric_native_affine_body as module
    return module


@lru_cache(maxsize=1)
def _source():
    from tectonics.geometric_native_normal_support import load_native_normal_support_source
    return load_native_normal_support_source()


@lru_cache(maxsize=1)
def _bodies():
    source = _source()
    return tuple(_api().bind_native_affine_hot_body(bank.component_id, source=source,
        provenance="New affine Ritz field on the complete authenticated HOT bank")
        for bank in source.banks if bank.material_kind == "hot_birth")


def _decimal(value):
    return D(value.numerator)/D(value.denominator)


def _sin_cos(angle):
    sine, cosine = angle, D(1)
    sine_term, cosine_term = angle, D(1)
    for index in range(1, 180):
        sine_term *= -angle*angle/D((2*index)*(2*index+1))
        cosine_term *= -angle*angle/D((2*index-1)*(2*index))
        sine += sine_term
        cosine += cosine_term
    return sine, cosine


def _reference_oracle(body, depth):
    """Independent Rodrigues point/differential at the Cartesian unit ray."""
    with localcontext() as context:
        context.prec = 120
        radius = _decimal(body.radius_m)
        top = radius+_decimal(body.roof_radial_displacement_m)
        cube = top**3-3*radius**2*_decimal(depth)
        physical_radius = cube**(D(1)/D(3))
        tau = (radius**3-cube)/(3*radius**2)
        angular_rate = tuple(_decimal(value) for value in
            body.angular_rate_per_physical_volume_depth_m)
        norm = sum((value*value for value in angular_rate), D(0)).sqrt()
        if norm:
            axis = tuple(value/norm for value in angular_rate)
            sine, cosine = _sin_cos(norm*tau)
            cross = ((D(0), -axis[2], axis[1]),
                (axis[2], D(0), -axis[0]), (-axis[1], axis[0], D(0)))
            rotation = tuple(tuple((cosine if i == j else D(0))
                +(1-cosine)*axis[i]*axis[j]+sine*cross[i][j]
                for j in range(3)) for i in range(3))
        else:
            rotation = tuple(tuple(D(i == j) for j in range(3)) for i in range(3))
        point = tuple(physical_radius*rotation[i][0] for i in range(3))
        # (A cross u) at u=(1,0,0) is (0,A_z,-A_y).
        twist = (D(0), angular_rate[2], -angular_rate[1])
        radial_gradient = tuple(tuple(D(i == j)
            -(cube/radius**2)*twist[i]*(D(1) if j == 0 else D(0))
            for j in range(3)) for i in range(3))
        gradient = tuple(tuple(sum((rotation[i][k]*radial_gradient[k][j]
            for k in range(3)), D(0)) for j in range(3)) for i in range(3))
        return point, gradient


def _contains(interval, value):
    value = F(value) if isinstance(value, D) else value
    assert interval.lo <= value <= interval.hi


def test_all_five_actual_hot_masks_keep_their_authoritative_volume_and_source():
    source = _source()
    assert len(_bodies()) == 5
    banks = {bank.component_id: bank for bank in source.banks}
    for body in _bodies():
        bank = banks[body.component_id]
        raw = json.loads(bank.complete_mask_source_json)
        area = I(F(*raw["area_bounds_sr"]["lo"]), F(*raw["area_bounds_sr"]["hi"]))
        assert body.actual_area_bounds_sr == area
        assert body.reference_volume_bounds_m3 == area*source.radius_m**2*bank.areal_material_thickness_m
        assert body.reference_volume_bounds_m3.lo > 0
        assert body.areal_material_thickness_m == bank.areal_material_thickness_m
        assert body.complete_mask_source_json == bank.complete_mask_source_json
        assert body.saved_field_json == source.saved_field_json
        assert body.source_snapshot_sha256 == source.source_snapshot_sha256
        assert body.source_support_sha256 == source.source_support_sha256
        assert body.source_epoch_sha256 == source.source_epoch_sha256
        assert body.reference_origin_m == (ZERO, ZERO, ZERO)


def test_three_real_tiny_hot_reservoirs_are_preserved_without_geometric_cutoff():
    tiny = tuple(body for body in _bodies() if body.actual_area_bounds_sr.hi < F(1, 10**40))
    assert len(tiny) == 3
    assert all(body.actual_area_bounds_sr.lo > 0 for body in tiny)
    assert all(body.reference_volume_bounds_m3.lo > 0 for body in tiny)
    assert min(body.actual_area_bounds_sr.lo for body in tiny) < F(2, 10**47)


def test_affine_coordinates_have_metres_and_both_tensor_shear_entries():
    body = _bodies()[0]
    coordinates = tuple(map(F, (2, -3, 4, 5, -7, 11)))
    inverse_radius = 1/body.radius_m
    expected = ((1+2*inverse_radius, 5*inverse_radius, -7*inverse_radius),
        (5*inverse_radius, 1-3*inverse_radius, 11*inverse_radius),
        (-7*inverse_radius, 11*inverse_radius, 1+4*inverse_radius))
    assert _api().native_affine_body_gradient(body, coordinates) == expected
    assert _api().native_affine_body_gradient(body, (ZERO,)*6) == IDENTITY


@pytest.mark.parametrize("index", range(5))
def test_saved_reference_differential_keeps_depth_twist_and_unit_determinant(index):
    body = _bodies()[index]
    depth = body.areal_material_thickness_m/3
    result = _api().bound_native_affine_reference_gradient(body,
        (I(F(1)), I(ZERO), I(ZERO)), I(depth), precision=64)
    _, expected = _reference_oracle(body, depth)
    for row, oracle in zip(result.reference_gradient_bounds, expected):
        for interval, value in zip(row, oracle):
            _contains(interval, value)
    assert result.determinant_exact == F(1)
    assert result.signed_volume_jacobian_exact == -body.radius_m**2
    assert not result.material_membership_authenticated
    if any(body.angular_rate_per_physical_volume_depth_m):
        assert any(abs(value-D(i == j)) > D("1e-20")
            for i, row in enumerate(expected) for j, value in enumerate(row))


def test_affine_world_point_is_actual_reference_pushforward_and_translation():
    body = _bodies()[0]
    depth = body.areal_material_thickness_m/4
    amplitudes = tuple(map(F, (2, -3, 4, 5, -7, 11)))
    translation = tuple(map(F, (17, -19, 23)))
    result = _api().bound_native_affine_body_point(body,
        (I(F(1)), I(ZERO), I(ZERO)), I(depth), amplitudes,
        translation_m=translation, precision=64)
    point, _ = _reference_oracle(body, depth)
    radius = body.radius_m
    gradient = ((1+F(2)/radius, F(5)/radius, -F(7)/radius),
        (F(5)/radius, 1-F(3)/radius, F(11)/radius),
        (-F(7)/radius, F(11)/radius, 1+F(4)/radius))
    with localcontext() as context:
        context.prec = 110
        expected = tuple(sum((_decimal(gradient[i][j])*point[j] for j in range(3)), D(0))
            +_decimal(translation[i]) for i in range(3))
    for interval, value in zip(result.affine_point_bounds_m, expected):
        _contains(interval, value)
    assert result.incremental_gradient == gradient
    assert not result.material_membership_authenticated


def test_actual_body_does_not_turn_an_original_cold_bank_into_hot_free_affine_material():
    original = next(bank for bank in _source().banks if bank.material_kind == "original_live")
    with pytest.raises((ValueError, TypeError), match="HOT|hot"):
        _api().bind_native_affine_hot_body(original.component_id, source=_source())


@pytest.mark.parametrize("field", ["actual_area_bounds_sr", "reference_volume_bounds_m3", "roof_radial_displacement_m"])
def test_forged_cached_geometry_cannot_manufacture_a_bigger_body(field):
    body = _bodies()[0]
    replacement = I(F(1)) if field.endswith("sr") or field.endswith("m3") else body.roof_radial_displacement_m+1
    forged = replace(body, **{field: replacement})
    with pytest.raises((ValueError, TypeError), match="replay|source|body|authentic"):
        _api().validate_native_affine_hot_body(forged)


def test_singular_or_inverted_increment_is_refused_without_stiffness_floors():
    body = _bodies()[0]
    for scale in (F(-1), F(-2)):
        coordinates = (scale*body.radius_m, ZERO, ZERO, ZERO, ZERO, ZERO)
        with pytest.raises((ValueError, TypeError), match="Jacobian|positive|orientation|singular"):
            _api().native_affine_body_gradient(body, coordinates)


@pytest.mark.parametrize("family_id", ["born-shared-0", "born-shared-3"])
def test_whole_source_face_trace_matches_closed_form_dilatation_integral(family_id):
    source = _source()
    family = next(row for row in source.families if row.family_id == family_id)
    face = _api().bound_native_affine_source_face_trace(family_id, "receiver",
        source=source, depth_cells=4, precision=64)
    assert face.native_angle_end_bounds_radians == family.native_angle_end_bounds_radians
    assert face.full_angle_bounds_radians == I(ZERO, family.native_angle_end_bounds_radians.hi)
    assert face.uncertain_angle_end_bounds_radians.width > 0
    assert face.common_depth_below_shared_top_bounds_m.lo == 0
    depth = face.common_depth_below_shared_top_bounds_m.hi
    top = face.common_reference_top_radius_m
    # The trace of the six-mode virtual displacement is X/R.  The exact
    # oriented area vector dotted with X/R is −r³(A dot T)/R, so both depth
    # and angular integrals have a closed form independent of Q or quadrature.
    radial_integral = top**3*depth-F(3, 2)*source.radius_m**2*depth**2
    total = sum(face.generalized_normal_force_bounds_n_per_pa[:3], I(ZERO))
    with localcontext() as context:
        context.prec = 120
        for endpoint in (family.native_angle_end_bounds_radians.lo,
                family.native_angle_end_bounds_radians.hi):
            sine, cosine = _sin_cos(_decimal(endpoint))
            delta = tuple(_decimal(start.midpoint)*(cosine-1)
                +_decimal(tangent.midpoint)*sine for start, tangent in
                zip(family.start_unit_bounds, family.tangent_unit_bounds))
            projection = sum((_decimal(rate)*value for rate, value in
                zip(family.shared_depth_axis_per_m, delta)), D(0))
            expected = -D(family.receiving_normal_orientation)*projection
            expected *= _decimal(radial_integral/source.radius_m)
            _contains(total, expected)
    donor = _api().bound_native_affine_source_face_trace(family_id, "donor",
        source=source, depth_cells=4, precision=64)
    assert donor.generalized_normal_force_bounds_n_per_pa == tuple(
        -value for value in face.generalized_normal_force_bounds_n_per_pa)
    assert donor.gap_sign == -1 and face.gap_sign == 1
    assert not face.force_equilibrium_certified
    assert not face.physical_material_admission_authorized


def _gauss_legendre_decimal(count):
    """Independent numerical quadrature nodes, refined beyond binary floats."""
    result = []
    for index in range(1, count//2+1):
        root = D.from_float(math.cos(math.pi*(index-F(1, 4))/(count+F(1, 2))))
        for _ in range(20):
            previous, polynomial = D(1), root
            for degree in range(2, count+1):
                previous, polynomial = polynomial, ((2*degree-1)*root*polynomial
                    -(degree-1)*previous)/degree
            derivative = count*(root*polynomial-previous)/(root*root-1)
            correction = polynomial/derivative
            root -= correction
            if abs(correction) < D("1e-110"):
                break
        else:
            raise AssertionError("Independent Legendre node refinement did not converge")
        weight = 2/((1-root*root)*derivative*derivative)
        result.extend(((root, weight), (-root, weight)))
    return tuple(result)


def _rodrigues_vector(vector, angular_rate, tau):
    norm = sum((value*value for value in angular_rate), D(0)).sqrt()
    if not norm:
        return vector
    axis = tuple(value/norm for value in angular_rate)
    sine, cosine = _sin_cos(norm*tau)
    cross = (axis[1]*vector[2]-axis[2]*vector[1],
        axis[2]*vector[0]-axis[0]*vector[2], axis[0]*vector[1]-axis[1]*vector[0])
    projection = sum((a*b for a, b in zip(axis, vector)), D(0))
    return tuple(vector[i]*cosine+cross[i]*sine+axis[i]*projection*(1-cosine)
        for i in range(3))


def _general_face_quadrature_oracle(source, family, face):
    """Rodrigues quadrature plus explicit depth/angle/source error bounds.

    No commutator coefficients, production matrix exponential, or production
    depth moments are used.  The Gauss error bound follows the standard
    2n-derivative remainder on [0,1]; angular midpoint error uses a whole-face
    Lipschitz bound.  The source frame intervals are included in the bound.
    """
    with localcontext() as context:
        context.prec = 120
        radius = _decimal(source.radius_m)
        top_cube = _decimal(face.common_reference_top_radius_m**3)
        depth = _decimal(face.common_depth_below_shared_top_bounds_m.hi)
        endpoint = family.native_angle_end_bounds_radians
        theta = _decimal(endpoint.midpoint)
        theta_high = _decimal(endpoint.hi)
        endpoint_error = _decimal(endpoint.width/2)
        sine, cosine = _sin_cos(theta/2)
        start = tuple(_decimal(value.midpoint) for value in family.start_unit_bounds)
        tangent = tuple(_decimal(value.midpoint) for value in family.tangent_unit_bounds)
        normal = tuple(_decimal(value.midpoint) for value in family.normal_unit_bounds)
        ray = tuple(a*cosine+b*sine for a, b in zip(start, tangent))
        angular_tangent = tuple(-a*sine+b*cosine for a, b in zip(start, tangent))
        angular_rate = tuple(_decimal(value) for value in family.shared_depth_axis_per_m)
        rate_norm = sum((value*value for value in angular_rate), D(0)).sqrt()
        projection = sum((a*b for a, b in zip(angular_rate, angular_tangent)), D(0))
        # Independent Cartesian tensor modes, with both shear entries.
        modes = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
        quadrature = [D(0) for _ in modes]
        count = 32
        for node, weight in _gauss_legendre_decimal(count):
            z = depth*(node+1)/2
            cube = top_cube-3*radius**2*z
            tau = (radius**3-cube)/(3*radius**2)
            rotated_ray = _rodrigues_vector(ray, angular_rate, tau)
            rotated_normal = _rodrigues_vector(normal, angular_rate, tau)
            for index, (i, j) in enumerate(modes):
                if i == j:
                    mixed = rotated_normal[i]*rotated_ray[i]
                    quadratic = rotated_ray[i]**2
                else:
                    mixed = rotated_normal[i]*rotated_ray[j]+rotated_normal[j]*rotated_ray[i]
                    quadratic = 2*rotated_ray[i]*rotated_ray[j]
                value = radius*mixed-cube/radius*projection*quadratic
                quadrature[index] += weight*value/2
        centres = tuple(D(family.receiving_normal_orientation)*theta*depth*value
            for value in quadrature)
        frequency = 2*rate_norm*depth
        derivative_order = 2*count
        derivative_bound = radius*frequency**derivative_order
        derivative_bound += rate_norm/radius*(top_cube*frequency**derivative_order
            +derivative_order*3*radius**2*depth*frequency**(derivative_order-1))
        gauss_constant = D(math.factorial(count))**4/(
            D(2*count+1)*D(math.factorial(2*count))**3)
        depth_error = theta_high*depth*gauss_constant*derivative_bound
        size_bound = depth*(radius+top_cube*rate_norm/radius)
        angular_lipschitz = depth*(radius+3*top_cube*rate_norm/radius)
        angle_error = theta_high**2*angular_lipschitz/4+endpoint_error*size_bound
        frame_error = sum((_decimal(value.width/2) for vector in
            (family.start_unit_bounds, family.tangent_unit_bounds, family.normal_unit_bounds)
            for value in vector), D(0))
        assert frame_error < 1
        source_error = 8*theta_high*size_bound*frame_error
        # This is a high precision regression oracle, not an interval-certified
        # Newton solver.  The generous Decimal120 arithmetic allowance remains
        # tiny beside the analytic full-angle bound and the strong shear mode.
        arithmetic_error = theta_high*size_bound*D("1e-85")
        error = depth_error+angle_error+source_error+arithmetic_error
        return tuple(I(F(value-error), F(value+error)) for value in centres)


def test_general_world_shear_face_modes_match_independent_rodrigues_depth_quadrature():
    source = _source()
    family = next(row for row in source.families if row.family_id == "born-shared-0")
    face = _api().bound_native_affine_source_face_trace(family.family_id, "receiver",
        source=source, depth_cells=4, precision=64)
    oracles = _general_face_quadrature_oracle(source, family, face)
    for actual, oracle in zip(face.generalized_normal_force_bounds_n_per_pa, oracles):
        assert max(actual.lo, oracle.lo) <= min(actual.hi, oracle.hi)
    strongest_shear = max(range(3, 6), key=lambda i: abs(oracles[i].midpoint))
    oracle = oracles[strongest_shear]
    assert oracle.lo > 0 or oracle.hi < 0
    assert oracle.width < abs(oracle.midpoint)*F(1, 10**10)
