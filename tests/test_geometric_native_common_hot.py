"""Independent polynomial and changed-basal checks on genuine common HOTs."""
from dataclasses import replace
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from functools import lru_cache
import json

import pytest

from tectonics.geometric_native_pose import FractionInterval as I


def _api():
    from tectonics import geometric_native_common_hot as module
    return module


@lru_cache(maxsize=1)
def _fixture():
    from tectonics.geometric_native_normal_support import load_native_normal_support_source
    from tectonics.geometric_native_common_reference import bind_native_common_reference
    from tectonics.geometric_contact_mechanics import ElasticColumnParameters
    from tectonics.geometric_native_bulk_elasticity import NativeBulkElasticClosure
    source = load_native_normal_support_source()
    reference = bind_native_common_reference(source=source, precision=160)
    parameters = json.loads(source.families[0].complete_carrier_source_json)["declaration"]["parameters"]
    columns = ElasticColumnParameters(**parameters["columns"])
    bodies, assemblies = [], []
    for bank in source.banks:
        if bank.material_kind != "hot_birth":
            continue
        body = _api().bind_native_common_hot_body(bank.component_id, source=source, reference=reference)
        closure = NativeBulkElasticClosure(columns, "over", bank.component_id,
            source.source_support_sha256,
            "Explicit unchanged new HOT isotropic material in the NEW common reference")
        basal = _api().bind_native_common_hot_basal_source(body, precision=64)
        bodies.append(body)
        assemblies.append(_api().assemble_native_common_hot_component(body, closure, basal_source=basal))
    return source, reference, tuple(bodies), tuple(assemblies)


def _contains(actual, expected):
    expected = expected if type(expected) is I else I(F(expected))
    assert actual.lo <= expected.lo <= expected.hi <= actual.hi


def _density(q, radius, shear, lame):
    xx, yy, zz, xy, xz, yz = tuple(value/radius for value in q)
    matrix = ((1+xx, xy, xz), (xy, 1+yy, yz), (xz, yz, 1+zz))
    green = tuple(tuple((sum((matrix[k][i]*matrix[k][j] for k in range(3)), F(0))
        -F(i == j))/2 for j in range(3)) for i in range(3))
    trace = sum((green[i][i] for i in range(3)), F(0))
    return shear*sum((x*x for row in green for x in row), F(0))+lame*trace**2/2


def _vary(q, direction, amount):
    return tuple(x+amount*y for x, y in zip(q, direction))


def _derivatives(q, direction, radius, shear, lame):
    # Exact five-point identities for a quartic energy, no truncation error.
    step = F(7, 11)
    energies = {n: _density(_vary(q, direction, n*step), radius, shear, lame)
        for n in (-2, -1, 0, 1, 2)}
    first = (-energies[2]+8*energies[1]-8*energies[-1]+energies[-2])/(12*step)
    second = (-energies[2]+16*energies[1]-30*energies[0]
        +16*energies[-1]-energies[-2])/(12*step**2)
    return first, second


def _decimal(value):
    return D(value.numerator)/D(value.denominator)


def _rotate_oracle(vector, angular_rate, tau):
    """Independent high precision Rodrigues series, without project rotation."""
    A = tuple(_decimal(value) for value in angular_rate)
    rate = sum((x*x for x in A), D(0)).sqrt()
    if not rate:
        return vector
    axis = tuple(value/rate for value in A)
    angle = rate*tau
    sine, cosine = angle, D(1)
    sterm, cterm = angle, D(1)
    for n in range(1, 90):
        sterm *= -angle**2/D(2*n*(2*n+1))
        cterm *= -angle**2/D((2*n-1)*(2*n))
        sine, cosine = sine+sterm, cosine+cterm
    cross = (axis[1]*vector[2]-axis[2]*vector[1],
        axis[2]*vector[0]-axis[0]*vector[2], axis[0]*vector[1]-axis[1]*vector[0])
    projection = sum((x*y for x, y in zip(axis, vector)), D(0))
    return tuple(cosine*x+sine*y+(1-cosine)*projection*z
        for x, y, z in zip(vector, cross, axis))


def test_all_five_common_hot_bindings_keep_authentic_mask_material_and_volume():
    source, reference, bodies, assemblies = _fixture()
    assert len(bodies) == len(assemblies) == 5
    changed = []
    for body in bodies:
        original, common = body.original_body, body.common_bank
        assert original.complete_mask_source_json == common.complete_mask_source_json
        assert original.saved_field_json == common.saved_field_json == source.saved_field_json
        assert original.angular_rate_per_physical_volume_depth_m == common.historical_angular_rate_per_physical_volume_depth_m
        assert common.angular_rate_per_physical_volume_depth_m == reference.angular_rate_per_physical_volume_depth_m
        assert original.reference_volume_bounds_m3 == common.reference_volume_bounds_m3
        assert original.roof_radial_displacement_m == common.roof_radial_displacement_m
        assert body.binding_sha256 != original.geometry_source_sha256
        assert body.reference_snapshot_sha256 == reference.reference_snapshot_sha256
        assert body.fixed_global_symmetric_generators == original.symmetric_generators
        if body.reference_geometry_changed:
            changed.append(body)
    assert len(changed) == 1
    assert changed[0].component_id.startswith("native-first-entry-actual-vacancy:295fbe04")
    assert changed[0].original_body.angular_rate_per_physical_volume_depth_m == (F(0),)*3
    assert len(tuple(b for b in bodies if b.original_body.actual_area_bounds_sr.hi < F(1, 10**40))) == 3


def test_common_bulk_invariance_keeps_independent_whole_volume_isotropic_polynomial():
    _, _, _, assemblies = _fixture()
    q = tuple(map(F, (170, -230, 410, 50, -70, 110)))
    direction = tuple(map(F, (2, -3, 5, 7, -11, 13)))
    for assembly in assemblies:
        bulk = assembly.bulk
        response = _api().evaluate_native_common_hot_bulk(bulk, q, direction_m=direction)
        radius = bulk.body.radius_m
        shear, lame = bulk.closure.shear_modulus_pa, bulk.closure.lame_lambda_pa
        volume = bulk.body.reference_volume_bounds_m3
        _contains(response.energy_bounds_j, volume*_density(q, radius, shear, lame))
        first, second = _derivatives(q, direction, radius, shear, lame)
        _contains(response.first_directional_derivative_bounds_j, volume*first)
        _contains(response.second_directional_derivative_bounds_j, volume*second)
        assert bulk.fixed_global_bulk_polynomial_exactly_invariant
        assert all(bulk.stiffness_bounds_n_m[i][i].lo > 0 for i in range(6))


def test_common_bulk_all_six_forces_and_36_hessian_entries_match_energy_oracle():
    assembly = next(a for a in _fixture()[3] if a.bulk.body.reference_geometry_changed).bulk
    q = tuple(map(F, (170, -230, 410, 50, -70, 110)))
    radius = assembly.body.radius_m
    shear, lame = assembly.closure.shear_modulus_pa, assembly.closure.lame_lambda_pa
    response = _api().evaluate_native_common_hot_bulk(assembly, q)
    units = tuple(tuple(F(i == j) for i in range(6)) for j in range(6))
    diagonal = []
    volume = assembly.body.reference_volume_bounds_m3
    for i, direction in enumerate(units):
        first, second = _derivatives(q, direction, radius, shear, lame)
        diagonal.append(second)
        _contains(response.generalized_gradient_bounds_n[i], volume*first)
        _contains(response.generalized_hessian_bounds_n_m[i][i], volume*second)
    for i in range(6):
        for j in range(i):
            direction = tuple(x+y for x, y in zip(units[i], units[j]))
            _, second = _derivatives(q, direction, radius, shear, lame)
            cross = (second-diagonal[i]-diagonal[j])/2
            _contains(response.generalized_hessian_bounds_n_m[i][j], volume*cross)
            assert response.generalized_hessian_bounds_n_m[i][j] == response.generalized_hessian_bounds_n_m[j][i]


def test_changed_hot_reference_point_has_independent_nonzero_depth_rotation():
    body = next(b for b in _fixture()[2] if b.reference_geometry_changed)
    depth = body.original_body.areal_material_thickness_m
    result = _api().bound_native_common_hot_body_point(body,
        (I(F(1)), I(F(0)), I(F(0))), I(depth), (F(0),)*6, precision=64)
    with localcontext() as context:
        context.prec = 100
        R = body.radius_m
        cube = (R+body.original_body.roof_radial_displacement_m)**3-3*R**2*depth
        r = _decimal(cube)**(D(1)/D(3))
        tau = _decimal((R**3-cube)/(3*R**2))
        oracle = _rotate_oracle((D(1), D(0), D(0)),
            body.common_bank.angular_rate_per_physical_volume_depth_m, tau)
        for bound, expected in zip(result.reference_point_bounds_m, oracle):
            _contains(bound, r*expected)
        assert any(abs(r*oracle[i]) > D("1") for i in (1, 2))
    assert not result.material_membership_authenticated


def test_changed_basal_is_fresh_at_true_base_and_cannot_reuse_old_zero_rate_receipt():
    from tectonics.geometric_native_affine_body import bind_native_affine_basal_source
    assembly = next(a for a in _fixture()[3] if a.bulk.body.reference_geometry_changed)
    body, new = assembly.bulk.body, assembly.basal.basal_source
    old = bind_native_affine_basal_source(body.original_body, precision=64)
    assert new.physical_volume_depth_bounds_m == I(body.original_body.areal_material_thickness_m)
    assert new.base_radius_bounds_m == old.base_radius_bounds_m
    assert new.base_radius_bounds_m.hi < body.radius_m
    assert old.whole_mask_unit_direction_bounds != new.whole_mask_unit_direction_bounds
    assert new.source_foundation_json == old.source_foundation_json
    assert new.density_kg_m3*new.gravity_m_s2 == F(4200)
    assert new.reference_measure_area_bounds_m2 == body.original_body.actual_area_bounds_sr*body.radius_m**2
    with pytest.raises(ValueError, match="source|receipt|replay"):
        _api().assemble_native_common_hot_basal(body, basal_source=old)


def test_fresh_basal_all_global_modes_enclose_independent_chart_vertex_rotation():
    from tectonics.geometric_native_affine_body import _normalized_native_vertex_bounds
    assembly = next(a for a in _fixture()[3] if a.bulk.body.reference_geometry_changed)
    body, basal = assembly.bulk.body, assembly.basal
    chart = json.loads(body.original_body.complete_mask_source_json)["proxy_geometries"][0]
    # Native chart vertices are containing evidence, not allocated material.
    vertices = _normalized_native_vertex_bounds(chart, 160)
    with localcontext() as context:
        context.prec = 100
        R = body.radius_m
        H = body.original_body.areal_material_thickness_m
        cube = (R+body.original_body.roof_radial_displacement_m)**3-3*R**2*H
        radial = _decimal(cube)**(D(1)/D(3))/_decimal(R)
        tau = _decimal((R**3-cube)/(3*R**2))
        for vertex in vertices:
            vector = tuple(_decimal(x.midpoint) for x in vertex)
            norm = sum((x*x for x in vector), D(0)).sqrt()
            vector = tuple(x/norm for x in vector)
            rotated = _rotate_oracle(vector, body.common_bank.angular_rate_per_physical_volume_depth_m, tau)
            for bound, value in zip(basal.basal_source.whole_mask_unit_direction_bounds, rotated):
                _contains(bound, value)
            for bound, E in zip(basal.normal_displacement_covector_bounds, body.fixed_global_symmetric_generators):
                expected = radial*sum((_decimal(E[i][j])*rotated[i]*rotated[j]
                    for i in range(3) for j in range(3)), D(0))
                _contains(bound, expected)


def test_isotropic_basal_energy_uses_same_field_source_measure_and_all_cross_terms():
    for assembly in _fixture()[3]:
        basal = assembly.basal
        q = (F(3), F(3), F(3), F(0), F(0), F(0))
        response = _api().evaluate_native_common_hot_basal(basal, q, direction_m=q)
        radial = basal.basal_source.base_radius_bounds_m/basal.body.radius_m
        coefficient = F(4200)*basal.reference_measure_area_bounds_m2
        _contains(response.whole_mask_normal_displacement_bounds_m, 3*radial)
        _contains(response.energy_bounds_j, coefficient*radial*radial*F(9, 2))
        assert len(response.generalized_gradient_bounds_n) == 6
        assert all(len(row) == 6 for row in response.generalized_hessian_bounds_n_m)
        for i in range(6):
            assert response.generalized_hessian_bounds_n_m[i][i].lo >= 0
            for j in range(i):
                assert response.generalized_hessian_bounds_n_m[i][j] == response.generalized_hessian_bounds_n_m[j][i]


def test_positive_curved_correction_is_included_in_new_basal_containment():
    sources = tuple(a.basal.basal_source for a in _fixture()[3])
    positive = tuple(s for s in sources if s.positive_correction_containing_chart_count)
    assert len(positive) == 1
    source = positive[0]
    mask = json.loads(source.body.original_body.complete_mask_source_json)
    assert len(source.containing_chart_unit_direction_bounds) == len(mask["proxy_geometries"])+1
    assert not source.containing_charts_assigned_as_material


def test_all_three_active_hot_faces_rebind_only_after_literal_field_equality():
    source, _, bodies, _ = _fixture()
    count = 0
    for family in source.families:
        for role, component_id in (("donor", family.donor_component_id), ("receiver", family.receiver_component_id)):
            body = next((b for b in bodies if b.component_id == component_id), None)
            if body is None:
                continue
            face = _api().bind_native_common_hot_source_face_trace(body, family.family_id, role,
                depth_cells=2, precision=64)
            _api().validate_native_common_hot_source_face_trace(face, body=body)
            assert face.physical_face_embedding_unchanged
            assert face.common_rate_literal == family.shared_depth_axis_per_m
            assert face.generalized_normal_force_bounds_n_per_pa == face.original_face.generalized_normal_force_bounds_n_per_pa
            assert face.original_face.source_preload_gap_m == F.from_float(-.01)
            count += 1
    assert count == 3


@pytest.mark.parametrize("target", ("body", "bulk", "basal_source", "basal", "combined"))
def test_forged_common_binding_and_cached_energy_receipts_are_rejected(target):
    assembly = _fixture()[3][0]
    value, validate = {
        "body": (assembly.bulk.body, _api().validate_native_common_hot_body),
        "bulk": (assembly.bulk, _api().validate_native_common_hot_bulk_assembly),
        "basal_source": (assembly.basal.basal_source, _api().validate_native_common_hot_basal_source),
        "basal": (assembly.basal, _api().validate_native_common_hot_basal_assembly),
        "combined": (assembly, _api().validate_native_common_hot_component_assembly),
    }[target]
    with pytest.raises(ValueError, match="altered|source|replay|authentic"):
        validate(replace(value, binding_sha256="0"*64))


def test_new_same_field_component_retains_narrow_scope_and_zero_new_physical_intervals():
    for assembly in _fixture()[3]:
        response = _api().evaluate_native_common_hot_component(assembly, (F(0),)*6)
        assert response.energy_bounds_j == I(F(0))
        assert response.generalized_gradient_bounds_n == (I(F(0)),)*6
        assert len(response.generalized_hessian_bounds_n_m) == 6
        for item in (assembly, assembly.bulk.body, assembly.bulk, assembly.basal,
                response, response.bulk_response, response.basal_response):
            assert item.scope == "COMMON_REFERENCE_HOT_COMPONENT_ONLY"
            assert not item.force_equilibrium_certified
            assert not item.physical_material_admission_authorized
            assert not item.whole_clock_all_pair_admission_certified
            assert not item.independent_radial_mean_body_mode_introduced
            assert not item.old_scalar_foundation_energy_added_again


def test_common_binding_rejects_changed_source_modulus_under_an_authentic_support_hash():
    assembly = _fixture()[3][0].bulk
    original = assembly.closure
    changed_columns = replace(original.source_columns, over_normal_modulus_pa=1.1e9)
    changed = replace(original, source_columns=changed_columns)
    # The alternate isotropic tensor is mathematically stable, but is not the
    # literal material declaration being transported into this reference.
    assert changed.shear_modulus_pa > 0 and changed.bulk_modulus_pa > 0
    with pytest.raises(ValueError, match="preserved|source columns"):
        _api().assemble_native_common_hot_bulk(assembly.body, changed)
    exported = _api().native_common_hot_to_dict(assembly)
    source_columns = exported["coefficient_kernel"]["closure"]["source_columns"]
    assert source_columns["over_normal_modulus_pa"] == original.source_columns.over_normal_modulus_pa
    assert type(source_columns["over_normal_modulus_pa"]) is float
