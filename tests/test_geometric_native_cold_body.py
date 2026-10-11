"""Source preservation and independent kinematics of actual cold banks."""
from dataclasses import replace
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from functools import lru_cache
import json

import pytest

from tectonics.geometric_native_pose import FractionInterval as I


ZERO = F(0)


def _api():
    from tectonics import geometric_native_cold_body as module
    return module


@lru_cache(maxsize=1)
def _source():
    from tectonics.geometric_native_normal_support import load_native_normal_support_source
    return load_native_normal_support_source()


@lru_cache(maxsize=1)
def _bodies():
    return tuple(_api().bind_native_cold_body(bank.component_id, source=_source(), precision=96)
        for bank in _source().banks if bank.material_kind == "original_live")


def _decimal(value):
    value = value.midpoint if type(value) is I else value
    value = value if type(value) is F else F(value)
    return D(value.numerator)/D(value.denominator)


def _sin_cos(angle):
    sine, cosine = angle, D(1)
    first, second = angle, D(1)
    for index in range(1, 150):
        first *= -angle*angle/D(2*index*(2*index+1))
        second *= -angle*angle/D((2*index-1)*2*index)
        sine += first
        cosine += second
    return sine, cosine


class DecimalAlgebra:
    """Numerical extension queries; midpoint frames confer no mask authority."""
    def constant(self, value):
        return _decimal(value)

    def sin(self, value):
        return _sin_cos(value)[0]

    def cos(self, value):
        return _sin_cos(value)[1]

    def sqrt(self, value):
        return value.sqrt()

    def cbrt(self, value):
        return value**(D(1)/D(3))


def _close(actual, expected, tolerance=D("1e-75")):
    assert abs(actual-expected) <= tolerance*max(D(1), abs(expected))


def test_all_eight_actual_cold_banks_keep_material_source_tip_and_beam_stiffness():
    source = _source()
    before = json.loads(dict(source.complete_source_dto_json)["before"])
    reference = before["original_physical_history"]["original_reference"]
    tips = {row[0]: row[1:] for row in before["snapshot"]["transverse_displacement_m"]}
    banks = {bank.component_id: bank for bank in source.banks}
    assert len(_bodies()) == 8
    assert len({body.component_id for body in _bodies()}) == 8
    for body in _bodies():
        bank = banks[body.component_id]
        raw = json.loads(bank.complete_mask_source_json)
        area = I(F(*raw["area_bounds_sr"]["lo"]), F(*raw["area_bounds_sr"]["hi"]))
        assert body.actual_area_bounds_sr == area
        assert body.complete_mask_source_json == bank.complete_mask_source_json
        assert body.saved_field_json == source.saved_field_json
        assert body.reference_volume_bounds_m3 == area*source.radius_m**2*bank.areal_material_thickness_m
        assert body.areal_material_thickness_m == bank.areal_material_thickness_m
        assert body.accepted_tip_displacement_m == F.from_float(tips[body.trace_id][0 if body.source_bank == "sub" else 1])
        index = body.old_absolute_coordinate_index
        assert body.source_old_tip_stiffness_n_m == F.from_float(reference["stiffness"][index][index])/source.radius_m**2
        assert body.beta_cut_radians == body.width_m/body.radius_m
        assert body.accepted_tip_displacement_m != 0


def test_physical_elastic_thickness_is_not_relabelled_as_areal_material_depth():
    for body in _bodies():
        top = body.radius_m+body.roof_radial_displacement_m
        expected = (top**3-(top-body.elastic_thickness_m)**3)/(3*body.radius_m**2)
        assert body.elastic_layer_areal_depth_m == expected
        assert 0 < expected < body.areal_material_thickness_m
        assert expected != body.elastic_thickness_m
        assert body.areal_material_thickness_m == F(10000)
        assert body.elastic_thickness_m == F(2000 if body.source_bank == "sub" else 3000)


def test_archived_material_and_nonzero_saved_depth_rotation_remain_present():
    before = json.loads(dict(_source().complete_source_dto_json)["before"])
    material = before["snapshot"]["current_material_epoch"]["source_material_state"]
    allocations = {row["live_domain_id"]: row for row in material["parent_allocations"]}
    archived = tuple(body for body in _bodies() if F(*allocations[body.component_id]["archived_area_bounds_sr"]["hi"]) > 0)
    assert len(archived) == 3
    assert {body.original_parent_id for body in archived} == {"octant:5", "octant:6", "octant:7"}
    for body in _bodies():
        assert (body.archive_area_bounds_sr.hi > 0) == (body in archived)
        assert not body.archived_material_used_as_live_support
        assert not body.proxy_polygons_promoted_to_material
    rotated = tuple(body for body in _bodies() if any(body.angular_rate_per_physical_volume_depth_m))
    assert {body.original_parent_id for body in rotated} == {"octant:2", "octant:6"}
    assert any(body.original_parent_id == "octant:5" for body in archived)


@pytest.mark.parametrize("parent", ["octant:0", "octant:4"])
def test_zero_rotation_modal_gradients_match_radial_tip_and_full_isotropic_enrichment(parent):
    body = next(body for body in _bodies() if body.original_parent_id == parent)
    assert not any(body.angular_rate_per_physical_volume_depth_m)
    with localcontext() as context:
        context.prec = 110
        theta = D("0.37")
        beta = _decimal(body.beta_cut_radians)/3
        depth = _decimal(body.elastic_layer_areal_depth_m)/2
        sine_theta, cosine_theta = _sin_cos(theta)
        geometry = _api().evaluate_native_cold_modal_geometry(body,
            sine_theta, cosine_theta, beta, depth, DecimalAlgebra())
        ray = geometry.radial_direction
        radius = sum((value*value for value in geometry.world_point_m), D(0)).sqrt()
        scale = beta/_decimal(body.beta_cut_radians)
        shape = 1-D(3)*scale/2+scale**3/2
        slope = D(3)*(scale**2-1)/(2*_decimal(body.beta_cut_radians))
        tangent = tuple(radius*value for value in geometry.material_beta_gradient_per_m)
        for i in range(3):
            _close(geometry.mode_displacement_per_m[0][i], shape*ray[i])
            for j in range(3):
                expected_tip = shape/radius*(D(i == j)-ray[i]*ray[j])+slope/radius*ray[i]*tangent[j]
                _close(geometry.mode_gradient_m_inv[0][i][j], expected_tip)
                expected_dilation = shape/_decimal(body.radius_m)*D(i == j)+slope/_decimal(body.radius_m)*ray[i]*tangent[j]
                actual_dilation = sum((geometry.mode_gradient_m_inv[k][i][j] for k in (1, 2, 3)), D(0))
                _close(actual_dilation, expected_dilation)
        _close(sum((ray[i]*tangent[i] for i in range(3)), D(0)), D(0))
        _close(sum((value*value for value in tangent), D(0)), D(1))


def test_all_seven_supported_modes_and_gradients_vanish_at_the_actual_clamp():
    with localcontext() as context:
        context.prec = 100
        sine_theta, cosine_theta = _sin_cos(D("0.37"))
        for body in _bodies():
            geometry = _api().evaluate_native_cold_modal_geometry(body,
                sine_theta, cosine_theta, _decimal(body.beta_cut_radians),
                _decimal(body.elastic_layer_areal_depth_m)/2, DecimalAlgebra())
            assert len(geometry.mode_displacement_per_m) == 7
            assert len(geometry.mode_gradient_m_inv) == 7
            assert all(value == 0 for row in geometry.mode_displacement_per_m for value in row)
            assert all(value == 0 for matrix in geometry.mode_gradient_m_inv for row in matrix for value in row)


def test_saved_depth_rotation_modal_gradients_obey_independent_material_chain_derivatives():
    """Numerical source-frame regression, with an allowance for frame bounds.

    The Rodrigues oracle acts directly on vectors, then differentiates the
    material map in beta and physical-volume depth. No production grad-beta,
    reference Jacobian, or rotation matrix is used to construct the oracle.
    """
    body = next(body for body in _bodies() if body.original_parent_id == "octant:2")
    assert any(body.angular_rate_per_physical_volume_depth_m)
    with localcontext() as context:
        context.prec = 110
        theta = D("0.37")
        beta = _decimal(body.beta_cut_radians)/3
        depth = _decimal(body.elastic_layer_areal_depth_m)/2
        st, ct = _sin_cos(theta)
        sb, cb = _sin_cos(beta)
        start, along, inward = (tuple(_decimal(value) for value in vector) for vector in
            (body.current_quarter_start_bounds, body.current_quarter_tangent_bounds,
             body.current_quarter_inward_bounds))
        unrotated_along = tuple(ct*x+st*y for x, y in zip(start, along))
        p = tuple(cb*x+sb*y for x, y in zip(unrotated_along, inward))
        tangent = tuple(-sb*x+cb*y for x, y in zip(unrotated_along, inward))
        radius0 = _decimal(body.radius_m)
        roof = radius0+_decimal(body.roof_radial_displacement_m)
        radius = (roof**3-3*radius0**2*depth)**(D(1)/D(3))
        tau = depth+(radius0**3-roof**3)/(3*radius0**2)
        rate = tuple(_decimal(value) for value in body.angular_rate_per_physical_volume_depth_m)
        norm = sum((value*value for value in rate), D(0)).sqrt()
        axis = tuple(value/norm for value in rate)
        sine, cosine = _sin_cos(norm*tau)

        def dot(first, second):
            return sum((x*y for x, y in zip(first, second)), D(0))

        def cross(first, second):
            return (first[1]*second[2]-first[2]*second[1],
                first[2]*second[0]-first[0]*second[2],
                first[0]*second[1]-first[1]*second[0])

        def rotate(vector):
            mixed = cross(axis, vector)
            axial = dot(axis, vector)
            return tuple(cosine*x+sine*y+(1-cosine)*axial*z
                for x, y, z in zip(vector, mixed, axis))

        ray, ray_beta = rotate(p), rotate(tangent)
        ray_depth = cross(rate, ray)
        point = tuple(radius*x for x in ray)
        point_beta = tuple(radius*x for x in ray_beta)
        point_depth = tuple(-radius0**2/radius**2*x+radius*y
            for x, y in zip(ray, ray_depth))
        scaled = beta/_decimal(body.beta_cut_radians)
        shape = 1-D(3)*scaled/2+scaled**3/2
        slope = D(3)*(scaled**2-1)/(2*_decimal(body.beta_cut_radians))
        expected_beta = [tuple(slope*x+shape*y for x, y in zip(ray, ray_beta))]
        expected_depth = [tuple(shape*x for x in ray_depth)]
        # Independent explicit basis, including both entries of each shear.
        pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
        for first, second in pairs:
            def action(vector):
                return tuple((vector[second] if i == first else D(0))
                    +(vector[first] if first != second and i == second else D(0))
                    for i in range(3))
            value, dbeta, ddepth = action(point), action(point_beta), action(point_depth)
            expected_beta.append(tuple((slope*x+shape*y)/radius0 for x, y in zip(value, dbeta)))
            expected_depth.append(tuple(shape*x/radius0 for x in ddepth))
        geometry = _api().evaluate_native_cold_modal_geometry(body,
            st, ct, beta, depth, DecimalAlgebra())
        frame_error = sum((_decimal(value.hi-value.lo) for vector in
            (body.current_quarter_start_bounds, body.current_quarter_tangent_bounds,
             body.current_quarter_inward_bounds) for value in vector), D(0))
        assert frame_error < D("1e-80")
        # Ninety-six digit source-frame enclosure plus Decimal arithmetic;
        # this numerical regression is not an independent interval proof.
        tolerance = D("1e-70")
        for actual, expected in zip(geometry.world_point_m, point):
            _close(actual, expected, tolerance)
        _close(dot(geometry.material_beta_gradient_per_m, point_beta), D(1), tolerance)
        _close(dot(geometry.material_beta_gradient_per_m, point_depth), D(0), tolerance)
        for mode, gradient in enumerate(geometry.mode_gradient_m_inv):
            for component in range(3):
                _close(dot(gradient[component], point_beta), expected_beta[mode][component], tolerance)
                _close(dot(gradient[component], point_depth), expected_depth[mode][component], tolerance)


def test_a_hot_bank_cannot_replace_a_source_cold_clamp_or_prestress():
    hot = next(bank for bank in _source().banks if bank.material_kind == "hot_birth")
    with pytest.raises((ValueError, TypeError), match="cold|COLD"):
        _api().bind_native_cold_body(hot.component_id, source=_source())


@pytest.mark.parametrize("field", ["accepted_tip_displacement_m", "source_old_tip_stiffness_n_m", "elastic_layer_areal_depth_m"])
def test_cached_source_changes_cannot_reset_prestress_or_change_the_physical_layer(field):
    body = _bodies()[0]
    forged = replace(body, **{field: ZERO if field == "accepted_tip_displacement_m" else getattr(body, field)+1})
    with pytest.raises((ValueError, TypeError), match="source|body|replay|authentic"):
        _api().validate_native_cold_body(forged)
