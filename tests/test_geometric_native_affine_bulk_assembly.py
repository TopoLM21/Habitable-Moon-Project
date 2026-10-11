"""Independent finite-body energy and coupling on authentic HOT material.

Polynomial energy differences and the analytic inverse isotropic tensor are
independent oracles; no caller-supplied SPD stiffness or fictitious body is used.
"""
from dataclasses import replace
from fractions import Fraction as F
from functools import lru_cache
import json

import pytest

from tectonics.geometric_contact_mechanics import ElasticColumnParameters
from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.geometric_native_bulk_elasticity import NativeBulkElasticClosure


ZERO = F(0)
IDENTITY = ((F(1), ZERO, ZERO), (ZERO, F(1), ZERO), (ZERO, ZERO, F(1)))


def _api():
    from tectonics import geometric_native_affine_bulk_assembly as module
    return module


def _geometry():
    from tectonics import geometric_native_affine_body as module
    return module


@lru_cache(maxsize=1)
def _source():
    from tectonics.geometric_native_normal_support import load_native_normal_support_source
    return load_native_normal_support_source()


@lru_cache(maxsize=1)
def _assemblies():
    source = _source()
    carrier = json.loads(source.families[0].complete_carrier_source_json)
    columns = ElasticColumnParameters(**carrier["declaration"]["parameters"]["columns"])
    result = []
    for bank in source.banks:
        if bank.material_kind != "hot_birth":
            continue
        body = _geometry().bind_native_affine_hot_body(bank.component_id, source=source,
            provenance="New whole-bank affine HOT Ritz deformation")
        closure = NativeBulkElasticClosure(columns, "over", body.component_id,
            body.source_support_sha256,
            "Explicit new uniform HOT tensor extension from supplied source contact E and G")
        result.append(_api().assemble_native_affine_bulk(body, closure))
    return tuple(result)


def _gradient(coordinates, radius):
    xx, yy, zz, xy, xz, yz = (value/radius for value in coordinates)
    return ((1+xx, xy, xz), (xy, 1+yy, yz), (xz, yz, 1+zz))


def _multiply(first, second):
    return tuple(tuple(sum((first[i][k]*second[k][j] for k in range(3)), ZERO)
        for j in range(3)) for i in range(3))


def _density(coordinates, radius, shear, lame):
    """Direct independent StVK polynomial, not a production energy helper."""
    gradient = _gradient(coordinates, radius)
    green = tuple(tuple((sum((gradient[k][i]*gradient[k][j] for k in range(3)), ZERO)
        -F(i == j))/2 for j in range(3)) for i in range(3))
    trace = sum((green[i][i] for i in range(3)), ZERO)
    return shear*sum((entry*entry for row in green for entry in row), ZERO)+lame*trace**2/2


def _encloses(actual, expected):
    expected = expected if type(expected) is I else I(expected)
    assert actual.lo <= expected.lo <= expected.hi <= actual.hi


def _vary(coordinates, direction, amount):
    return tuple(a+amount*b for a, b in zip(coordinates, direction))


def test_actual_whole_mask_stiffness_is_analytic_isotropy_times_source_volume():
    assert len(_assemblies()) == 5
    for assembly in _assemblies():
        radius = assembly.body.radius_m
        shear = assembly.closure.shear_modulus_pa
        lame = assembly.closure.lame_lambda_pa
        expected = tuple(tuple((lame+2*shear if i == j else lame)/radius**2
            if i < 3 and j < 3 else (4*shear/radius**2 if i == j else ZERO)
            for j in range(6)) for i in range(6))
        assert assembly.dof_names == ("xx", "yy", "zz", "xy", "xz", "yz")
        assert assembly.stiffness_per_reference_volume_n_m4 == expected
        volume = assembly.body.reference_volume_bounds_m3
        for i in range(6):
            for j in range(6):
                _encloses(assembly.stiffness_bounds_n_m[i][j], volume*expected[i][j])
        assert assembly.actual_whole_mask_bulk_component_integrated
        assert assembly.material_tensor_is_explicit_new_hot_closure
        assert not assembly.previous_complete_three_dimensional_tensor_inherited
        assert not assembly.arbitrary_discrete_stiffness_supplied
        assert not assembly.source_normal_column_lengths_used_as_geometry


def test_actual_tiny_hot_masks_keep_positive_bulk_stiffness_and_strain_energy():
    tiny = tuple(a for a in _assemblies() if a.body.actual_area_bounds_sr.hi < F(1, 10**40))
    assert len(tiny) == 3
    for assembly in tiny:
        assert all(assembly.stiffness_bounds_n_m[i][i].lo > 0 for i in range(6))
        response = _api().evaluate_native_affine_bulk(assembly, (F(1), ZERO, ZERO, ZERO, ZERO, ZERO))
        assert response.energy_bounds_j.lo > 0
        assert response.generalized_gradient_bounds_n[0].lo > 0
        expected = assembly.reference_volume_bounds_m3*_density(
            (F(1), ZERO, ZERO, ZERO, ZERO, ZERO), assembly.body.radius_m,
            assembly.closure.shear_modulus_pa, assembly.closure.lame_lambda_pa)
        _encloses(response.energy_bounds_j, expected)


def test_zero_increment_has_zero_reference_energy_and_no_hidden_initial_prestress():
    for assembly in _assemblies():
        response = _api().evaluate_native_affine_bulk(assembly, (ZERO,)*6)
        assert response.body_deformation_gradient == IDENTITY
        assert response.energy_bounds_j == I(ZERO)
        assert response.generalized_gradient_bounds_n == (I(ZERO),)*6
        assert response.volume_jacobian == 1
        assert response.generalized_hessian_bounds_n_m == assembly.stiffness_bounds_n_m


def test_whole_body_nonlinear_energy_and_directional_force_match_independent_quartic():
    assembly = _assemblies()[0]
    radius = assembly.body.radius_m
    shear, lame = assembly.closure.shear_modulus_pa, assembly.closure.lame_lambda_pa
    direction = tuple(map(F, (170, -230, 410, 50, -70, 110)))
    gradient = _gradient(direction, radius)
    increment = tuple(tuple(gradient[i][j]-F(i == j) for j in range(3)) for i in range(3))
    square = _multiply(increment, increment)
    trace = sum((increment[i][i] for i in range(3)), ZERO)
    square_trace = sum((square[i][i] for i in range(3)), ZERO)
    contraction = lambda a, b: sum((a[i][j]*b[i][j] for i in range(3) for j in range(3)), ZERO)
    a2 = shear*contraction(increment, increment)+lame*trace**2/2
    a3 = shear*contraction(increment, square)+lame*trace*square_trace/2
    a4 = shear*contraction(square, square)/4+lame*square_trace**2/8
    for time in (F(-2, 3), F(1, 3), F(7, 5)):
        response = _api().evaluate_native_affine_bulk(assembly,
            tuple(time*x for x in direction), direction_m=direction)
        volume = assembly.reference_volume_bounds_m3
        _encloses(response.energy_bounds_j, volume*(a2*time**2+a3*time**3+a4*time**4))
        _encloses(response.first_directional_derivative_bounds_j,
            volume*(2*a2*time+3*a3*time**2+4*a4*time**3))
        _encloses(response.second_directional_derivative_bounds_j,
            volume*(2*a2+6*a3*time+12*a4*time**2))


@pytest.mark.parametrize("coordinate", range(6))
def test_each_generalized_force_and_curvature_matches_exact_independent_energy_differences(coordinate):
    assembly = _assemblies()[0]
    position = tuple(map(F, (170, -230, 410, 50, -70, 110)))
    direction = tuple(F(index == coordinate) for index in range(6))
    response = _api().evaluate_native_affine_bulk(assembly, position)
    step = F(7, 11)
    energies = {index: _density(_vary(position, direction, index*step),
        assembly.body.radius_m, assembly.closure.shear_modulus_pa,
        assembly.closure.lame_lambda_pa) for index in (-2, -1, 0, 1, 2)}
    derivative = (-energies[2]+8*energies[1]-8*energies[-1]+energies[-2])/(12*step)
    curvature = (-energies[2]+16*energies[1]-30*energies[0]
        +16*energies[-1]-energies[-2])/(12*step**2)
    volume = assembly.reference_volume_bounds_m3
    _encloses(response.generalized_gradient_bounds_n[coordinate], volume*derivative)
    _encloses(response.generalized_hessian_bounds_n_m[coordinate][coordinate], volume*curvature)


def test_superposed_world_rotation_preserves_integrated_energy_and_rotates_actual_piola():
    assembly = _assemblies()[0]
    position = tuple(map(F, (170, -230, 410, 50, -70, 110)))
    rotation = ((ZERO, F(-1), ZERO), (F(1), ZERO, ZERO), (ZERO, ZERO, F(1)))
    first = _api().evaluate_native_affine_bulk(assembly, position)
    rotated = _api().evaluate_native_affine_bulk(assembly, position, spatial_rotation=rotation)
    assert rotated.energy_bounds_j == first.energy_bounds_j
    assert rotated.volume_jacobian == first.volume_jacobian
    assert rotated.green_strain == first.green_strain
    assert rotated.generalized_gradient_bounds_n == first.generalized_gradient_bounds_n
    assert rotated.generalized_hessian_bounds_n_m == first.generalized_hessian_bounds_n_m
    assert rotated.deformation_gradient == _multiply(rotation, first.deformation_gradient)
    assert rotated.first_piola_stress_pa == _multiply(rotation, first.first_piola_stress_pa)


def test_reflection_cannot_be_disguised_as_a_rigid_world_rotation():
    assembly = _assemblies()[0]
    reflection = ((F(-1), ZERO, ZERO), (ZERO, F(1), ZERO), (ZERO, ZERO, F(1)))
    with pytest.raises((ValueError, TypeError), match="SO|rotation|proper"):
        _api().evaluate_native_affine_bulk(assembly, (ZERO,)*6, spatial_rotation=reflection)


def test_cached_stiffness_changes_are_refused_by_actual_material_integration_replay():
    assembly = _assemblies()[0]
    rows = tuple(tuple(I(F(1)) if i == j == 0 else item for j, item in enumerate(row))
        for i, row in enumerate(assembly.stiffness_bounds_n_m))
    forged = replace(assembly, stiffness_bounds_n_m=rows)
    with pytest.raises((ValueError, TypeError), match="stiffness|integration|replay"):
        _api().validate_native_affine_bulk_assembly(forged)


def test_supported_hot_component_does_not_claim_mixed_model_force_or_material_admission():
    for assembly in _assemblies():
        assert not assembly.complete_three_dimensional_strain_energy_certified
        assert not assembly.mixed_boundary_conditions_satisfied
        assert not assembly.force_equilibrium_certified
        assert not assembly.material_admission_authorized


@lru_cache(maxsize=1)
def _shared_faces():
    faces = tuple(_geometry().bound_native_affine_source_face_trace(family_id, "receiver",
        source=_source(), depth_cells=4, precision=64)
        for family_id in ("born-shared-0", "born-shared-3"))
    assert faces[0].component_id == faces[1].component_id
    assembly = next(a for a in _assemblies() if a.body.component_id == faces[0].component_id)
    return assembly, faces


def _isotropic_inverse(assembly):
    radius = assembly.body.radius_m
    shear, lame = assembly.closure.shear_modulus_pa, assembly.closure.lame_lambda_pa
    return tuple(tuple(radius**2*((F(i == j)/(2*shear))
        -lame/(2*shear*(2*shear+3*lame))) if i < 3 and j < 3
        else (radius**2/(4*shear) if i == j else ZERO)
        for j in range(6)) for i in range(6))


def test_two_genuine_faces_couple_through_one_actual_hot_body_with_the_analytic_tensor_inverse():
    assembly, faces = _shared_faces()
    response = _api().condense_native_affine_shared_faces(assembly, faces)
    inverse = _isotropic_inverse(assembly)
    assert response.stiffness_inverse_per_reference_volume_m4_n == inverse
    assert response.face_ids == ("born-shared-0", "born-shared-3")
    assert response.face_operators_bounds_m2 == tuple(
        face.generalized_normal_force_bounds_n_per_pa for face in faces)
    # Interior representatives test the independent compliance algebra; the
    # receipts continue to carry their full geometric bounds, not these points.
    operators = tuple(tuple(value.midpoint for value in
        face.generalized_normal_force_bounds_n_per_pa) for face in faces)
    volume = assembly.reference_volume_bounds_m3.midpoint
    for i in range(2):
        for j in range(2):
            expected = sum((operators[i][a]*inverse[a][b]*operators[j][b]
                for a in range(6) for b in range(6)), ZERO)/volume
            _encloses(response.pressure_compliance_bounds_m3_pa[i][j], expected)
            assert response.pressure_compliance_bounds_m3_pa[i][j] == \
                response.pressure_compliance_bounds_m3_pa[j][i]
        for a in range(6):
            expected = sum((inverse[a][b]*operators[i][b] for b in range(6)), ZERO)/volume
            _encloses(response.unit_pressure_body_response_bounds_m_pa[i][a], expected)
    cross = response.pressure_compliance_bounds_m3_pa[0][1]
    assert cross.lo > 0 or cross.hi < 0
    assert response.actual_whole_mask_stiffness_used
    assert response.shared_bank_cross_contact_terms_retained
    assert not response.exact_face_operators_substituted_for_intervals
    assert not response.basal_foundation_included
    assert not response.complete_hot_component_compliance_certified
    assert not response.force_equilibrium_certified
    assert not response.material_admission_authorized


def test_signed_shared_pressure_loading_obeys_true_body_virtual_work():
    assembly, faces = _shared_faces()
    response = _api().condense_native_affine_shared_faces(assembly, faces)
    inverse = _isotropic_inverse(assembly)
    volume = assembly.reference_volume_bounds_m3.midpoint
    operators = tuple(tuple(x.midpoint for x in
        face.generalized_normal_force_bounds_n_per_pa) for face in faces)
    pressures = (F(7), F(-3))
    force = tuple(sum((pressure*operators[j][i] for j, pressure in enumerate(pressures)), ZERO)
        for i in range(6))
    displacement = tuple(sum((inverse[i][j]*force[j] for j in range(6)), ZERO)/volume
        for i in range(6))
    for i, expected in enumerate(displacement):
        interval = sum((pressures[j]*response.unit_pressure_body_response_bounds_m_pa[j][i]
            for j in range(2)), I(ZERO))
        _encloses(interval, expected)
    pressure_work = sum((force[i]*displacement[i] for i in range(6)), ZERO)
    stiffness = assembly.stiffness_per_reference_volume_n_m4
    body_work = volume*sum((displacement[i]*stiffness[i][j]*displacement[j]
        for i in range(6) for j in range(6)), ZERO)
    assert body_work == pressure_work > 0
    condensed_work = sum((pressures[i]*pressures[j]*
        response.pressure_compliance_bounds_m3_pa[i][j]
        for i in range(2) for j in range(2)), I(ZERO))
    _encloses(condensed_work, body_work)


def test_another_bank_or_a_duplicated_face_cannot_supply_shared_material_compliance():
    assembly, faces = _shared_faces()
    with pytest.raises((ValueError, TypeError), match="once|registered|face"):
        _api().condense_native_affine_shared_faces(assembly, (faces[0], faces[0]))
    other = next(a for a in _assemblies() if a.body.component_id != assembly.body.component_id)
    with pytest.raises((ValueError, TypeError), match="bank|body|source"):
        _api().condense_native_affine_shared_faces(other, faces)


@lru_cache(maxsize=1)
def _basal_assemblies():
    return tuple(_api().assemble_native_affine_basal(assembly.body)
        for assembly in _assemblies())


def _square(interval):
    low = ZERO if interval.lo <= 0 <= interval.hi else min(interval.lo**2, interval.hi**2)
    return I(low, max(interval.lo**2, interval.hi**2))


def test_source_basal_fixed_offset_keeps_original_coefficients_and_actual_reference_area_once():
    for assembly in _basal_assemblies():
        source = assembly.basal_source
        raw = json.loads(source.source_foundation_json)["declaration"]["parameters"]
        coefficient = F.from_float(raw["density_kg_m3"])*F.from_float(raw["gravity_m_s2"])
        expected_area = assembly.body.actual_area_bounds_sr*assembly.body.radius_m**2
        assert coefficient == assembly.foundation_stiffness_density_pa_m
        assert assembly.reference_measure_area_bounds_m2 == expected_area
        assert assembly.dof_names == ("xx", "yy", "zz", "xy", "xz", "yz")
        assert len(assembly.stiffness_bounds_n_m) == 6
        for offset in (F(3), F(-5)):
            response = _api().evaluate_native_affine_basal(assembly, (ZERO,)*6,
                external_offset_m=offset)
            _encloses(response.energy_bounds_j, coefficient*expected_area*offset**2/2)
        assert assembly.original_density_and_gravity_preserved
        assert assembly.reference_area_winkler_extension_explicit
        assert not assembly.physical_current_area_winkler_law_inherited
        assert not assembly.old_scalar_foundation_energy_added_again
        assert not assembly.full_mixed_boundary_conditions_satisfied


def test_whole_basal_isotropic_dilation_has_exact_shape_independent_offset_strain_coupling():
    for assembly in _basal_assemblies():
        offset, dilation = F(7), F(-3)
        radius_ratio = assembly.basal_source.base_radius_bounds_m/assembly.body.radius_m
        displacement = I(offset)+dilation*radius_ratio
        coefficient = assembly.foundation_stiffness_density_pa_m*assembly.reference_measure_area_bounds_m2
        direction = (F(1), F(1), F(1), ZERO, ZERO, ZERO)
        response = _api().evaluate_native_affine_basal(assembly,
            (dilation, dilation, dilation, ZERO, ZERO, ZERO), direction_m=direction,
            external_offset_m=offset)
        _encloses(response.whole_mask_normal_displacement_bounds_m, displacement)
        _encloses(response.energy_bounds_j, coefficient*_square(displacement)/2)
        _encloses(sum(response.generalized_gradient_bounds_n[:3], I(ZERO)),
            coefficient*displacement*radius_ratio)
        _encloses(response.first_directional_derivative_bounds_j,
            coefficient*displacement*radius_ratio)
        _encloses(response.second_directional_derivative_bounds_j,
            coefficient*_square(radius_ratio))
        _encloses(sum((assembly.stiffness_bounds_n_m[i][j]
            for i in range(3) for j in range(3)), I(ZERO)),
            coefficient*_square(radius_ratio))
        # A geometric containing cone bounds the integral but never becomes
        # the amount of physical HOT material.
        assert not assembly.basal_source.containing_charts_assigned_as_material
        assert not response.complete_three_dimensional_strain_energy_certified


def test_real_tiny_hot_basal_foundations_keep_positive_dilation_energy_without_a_floor():
    tiny = tuple(a for a in _basal_assemblies() if a.body.actual_area_bounds_sr.hi < F(1, 10**40))
    assert len(tiny) == 3
    for assembly in tiny:
        response = _api().evaluate_native_affine_basal(assembly,
            (F(1), F(1), F(1), ZERO, ZERO, ZERO))
        assert response.energy_bounds_j.lo > 0


def test_basal_cached_coefficient_change_cannot_add_an_extra_foundation_energy_store():
    assembly = _basal_assemblies()[0]
    forged = replace(assembly, foundation_stiffness_density_pa_m=
        2*assembly.foundation_stiffness_density_pa_m)
    with pytest.raises((ValueError, TypeError), match="basal|coefficients|stiffness|source"):
        _api().validate_native_affine_basal_assembly(forged)


def test_combined_hot_component_pays_bulk_and_basal_work_for_the_same_six_actual_modes():
    bulk = _assemblies()[0]
    assembly = _api().assemble_native_affine_hot_component(bulk.body, bulk.closure)
    direction = (F(1), F(1), F(1), ZERO, ZERO, ZERO)
    dilation = F(17)
    response = _api().evaluate_native_affine_hot_component(assembly,
        (dilation, dilation, dilation, ZERO, ZERO, ZERO), direction_m=direction)
    radius = bulk.body.radius_m
    stretch = 1+dilation/radius
    strain = (stretch**2-1)/2
    shear, lame = bulk.closure.shear_modulus_pa, bulk.closure.lame_lambda_pa
    volume = bulk.reference_volume_bounds_m3
    energy_bulk = volume*(3*shear+F(9, 2)*lame)*strain**2
    force_bulk = volume*(6*shear+9*lame)*strain*stretch/radius
    curvature_bulk = volume*(6*shear+9*lame)*(stretch**2+strain)/radius**2
    ratio = assembly.basal.basal_source.base_radius_bounds_m/radius
    foundation = assembly.basal.foundation_stiffness_density_pa_m*assembly.basal.reference_measure_area_bounds_m2
    energy_basal = foundation*_square(dilation*ratio)/2
    force_basal = foundation*dilation*_square(ratio)
    curvature_basal = foundation*_square(ratio)
    _encloses(response.energy_bounds_j, energy_bulk+energy_basal)
    _encloses(response.first_directional_derivative_bounds_j, force_bulk+force_basal)
    _encloses(response.second_directional_derivative_bounds_j, curvature_bulk+curvature_basal)
    assert response.energy_bounds_j == response.bulk_response.energy_bounds_j+response.basal_response.energy_bounds_j
    assert assembly.dof_names == ("xx", "yy", "zz", "xy", "xz", "yz")
    for i in range(6):
        for j in range(6):
            assert assembly.reference_stiffness_bounds_n_m[i][j] == \
                assembly.bulk.stiffness_bounds_n_m[i][j]+assembly.basal.stiffness_bounds_n_m[i][j]
