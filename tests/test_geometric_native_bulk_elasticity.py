"""Independent mechanics for the declared native three-dimensional closure.

These are local constitutive tests; they confer no whole-mask force or material
admission authority. The old scalar source alone does not determine lateral BCs.
"""
from dataclasses import replace
from fractions import Fraction as F

import pytest

from tectonics.geometric_contact_mechanics import ElasticColumnParameters


SOURCE_SUPPORT = "a"*64
GEOMETRY_SOURCE = "b"*64
ZERO = F(0)
IDENTITY = ((F(1), ZERO, ZERO), (ZERO, F(1), ZERO), (ZERO, ZERO, F(1)))


def _api():
    from tectonics import geometric_native_bulk_elasticity as module
    return module


def _columns():
    return ElasticColumnParameters(1000., 1.e9, 4.e8, 1000., 1.e9, 4.e8)


def _closure(**changes):
    arguments = dict(source_columns=_columns(), source_bank="over",
        bank_source_id="actual-shared-native-bank", source_support_sha256=SOURCE_SUPPORT,
        lateral_closure_provenance="Explicit laterally free homogeneous isotropic extension")
    arguments.update(changes)
    return _api().NativeBulkElasticClosure(**arguments)


def _diag(first, second, third):
    return ((F(first), ZERO, ZERO), (ZERO, F(second), ZERO), (ZERO, ZERO, F(third)))


def _matmul(first, second):
    return tuple(tuple(sum((first[i][k]*second[k][j] for k in range(3)), ZERO)
        for j in range(3)) for i in range(3))


def _perturb(first, direction, amount):
    return tuple(tuple(first[i][j]+amount*direction[i][j] for j in range(3))
        for i in range(3))


def test_declared_laterally_free_source_moduli_determine_exact_isotropic_extension():
    closure = _closure()
    assert closure.young_modulus_pa == F(10**9)
    assert closure.shear_modulus_pa == F(4*10**8)
    assert closure.poisson_ratio == F(1, 4)
    assert closure.lame_lambda_pa == F(4*10**8)
    assert closure.bulk_modulus_pa == F(2*10**9, 3)
    assert closure.source_columns == _columns()
    assert closure.source_support_sha256 == SOURCE_SUPPORT
    assert closure.bank_source_id == "actual-shared-native-bank"


def test_bank_specific_moduli_are_taken_from_the_named_actual_source_bank():
    columns = ElasticColumnParameters(1000., 6.e8, 4.e8, 1700., 1.e9, 4.e8)
    donor = _closure(source_columns=columns, source_bank="sub")
    receiver = _closure(source_columns=columns, source_bank="over")
    assert donor.poisson_ratio == F(-1, 4)
    assert donor.lame_lambda_pa == F(-4*10**8, 3)
    assert donor.bulk_modulus_pa == F(4*10**8, 3)
    assert receiver.poisson_ratio == F(1, 4)
    assert donor.source_columns is columns and receiver.source_columns is columns


def test_laterally_free_uniaxial_limit_recovers_original_scalar_normal_energy():
    shortening = F(3, 10000)
    strain = _diag(-shortening, shortening/4, shortening/4)
    response = _api().evaluate_native_bulk_elasticity(_closure(), strain)
    assert response.stress_pa == _diag(-F(10**9)*shortening, 0, 0)
    assert response.energy_density_j_m3 == F(10**9)*shortening**2/2


def test_isochoric_robin_strike_expansion_has_sixty_percent_more_linear_bulk_energy():
    shortening = F(7, 10**5)
    free = _api().evaluate_native_bulk_elasticity(_closure(),
        _diag(-shortening, shortening/4, shortening/4))
    robin = _api().evaluate_native_bulk_elasticity(_closure(),
        _diag(-shortening, shortening, 0))
    assert robin.energy_density_j_m3 == F(8*10**8)*shortening**2
    assert robin.energy_density_j_m3 == F(8, 5)*free.energy_density_j_m3
    assert robin.stress_pa == _diag(-F(8*10**8)*shortening,
        F(8*10**8)*shortening, 0)


def test_symmetric_shear_uses_both_off_diagonal_components_and_physical_work():
    shear = F(3, 1000)
    strain = ((ZERO, shear/2, ZERO), (shear/2, ZERO, ZERO), (ZERO, ZERO, ZERO))
    response = _api().evaluate_native_bulk_elasticity(_closure(), strain,
        strain_direction=strain)
    stress = F(4*10**8)*shear
    assert response.stress_pa == ((ZERO, stress, ZERO), (stress, ZERO, ZERO),
        (ZERO, ZERO, ZERO))
    assert response.energy_density_j_m3 == F(4*10**8)*shear**2/2
    assert response.first_variation_j_m3 == F(4*10**8)*shear**2


def test_arbitrary_symmetric_strain_derivative_matches_independent_quadratic_difference():
    strain = ((F(2, 1000), F(3, 1000), F(-1, 1000)),
        (F(3, 1000), F(-5, 1000), F(4, 1000)),
        (F(-1, 1000), F(4, 1000), F(1, 1000)))
    direction = ((F(3), F(2), F(1)), (F(2), F(-1), F(-2)),
        (F(1), F(-2), F(5)))
    response = _api().evaluate_native_bulk_elasticity(_closure(), strain,
        strain_direction=direction)
    h = F(1, 10000)
    plus = _api().evaluate_native_bulk_elasticity(_closure(),
        _perturb(strain, direction, h)).energy_density_j_m3
    minus = _api().evaluate_native_bulk_elasticity(_closure(),
        _perturb(strain, direction, -h)).energy_density_j_m3
    assert response.first_variation_j_m3 == (plus-minus)/(2*h)
    assert response.stress_pa == ((F(8*10**5), F(24*10**5), F(-8*10**5)),
        (F(24*10**5), F(-48*10**5), F(32*10**5)),
        (F(-8*10**5), F(32*10**5), ZERO))


def test_finite_robin_volume_preservation_does_not_remove_its_transverse_strain_energy():
    alpha = F(999, 1000)
    gradient = _diag(alpha, 1/alpha, 1)
    response = _api().evaluate_native_bulk_deformation(_closure(), gradient,
        reference_density_kg_m3=F(3300), reference_volume_m3=F(12))
    axial = (alpha**2-1)/2
    lateral = (alpha**-2-1)/2
    expected = F(2*10**8)*(axial+lateral)**2+F(4*10**8)*(axial**2+lateral**2)
    assert response.energy_density_reference_j_m3 == expected
    assert response.volume_jacobian == 1
    assert response.physical_volume_m3 == 12
    assert response.current_density_kg_m3 == 3300
    assert response.conserved_mass_kg == 39600
    assert expected > F(10**9)*(alpha-1)**2/2


def test_physical_volume_can_change_while_original_mass_is_retained():
    gradient = _diag(F(11, 10), 1, 1)
    response = _api().evaluate_native_bulk_deformation(_closure(), gradient,
        reference_density_kg_m3=F(3300), reference_volume_m3=F(12))
    assert response.volume_jacobian == F(11, 10)
    assert response.physical_volume_m3 == F(66, 5)
    assert response.current_density_kg_m3 == 3000
    assert response.conserved_mass_kg == 39600
    assert response.current_density_kg_m3*response.physical_volume_m3 == 39600


def test_superposed_rigid_rotation_preserves_green_energy_and_rotates_spatial_stress():
    gradient = ((F(11, 10), F(1, 20), ZERO),
        (ZERO, F(9, 10), ZERO), (ZERO, ZERO, F(21, 20)))
    rotation = ((ZERO, F(-1), ZERO), (F(1), ZERO, ZERO), (ZERO, ZERO, F(1)))
    kwargs = dict(reference_density_kg_m3=F(3300), reference_volume_m3=F(12))
    first = _api().evaluate_native_bulk_deformation(_closure(), gradient, **kwargs)
    rotated = _api().evaluate_native_bulk_deformation(_closure(),
        _matmul(rotation, gradient), **kwargs)
    transpose = tuple(zip(*rotation))
    assert rotated.energy_density_reference_j_m3 == first.energy_density_reference_j_m3
    assert rotated.second_piola_stress_pa == first.second_piola_stress_pa
    assert rotated.first_piola_stress_pa == _matmul(rotation, first.first_piola_stress_pa)
    assert rotated.cauchy_stress_pa == _matmul(_matmul(rotation,
        first.cauchy_stress_pa), transpose)
    assert rotated.volume_jacobian == first.volume_jacobian


def test_finite_deformation_directional_force_matches_exact_quartic_energy_difference():
    gradient = ((F(11, 10), F(1, 20), ZERO),
        (ZERO, F(9, 10), F(1, 30)), (F(1, 50), ZERO, F(21, 20)))
    direction = ((F(1), F(2), F(-1)), (F(3), F(-2), F(1)),
        (ZERO, F(1), F(2)))
    kwargs = dict(reference_density_kg_m3=F(3300), reference_volume_m3=F(12))
    response = _api().evaluate_native_bulk_deformation(_closure(), gradient,
        deformation_gradient_direction=direction, **kwargs)
    def energy(amount):
        return _api().evaluate_native_bulk_deformation(_closure(),
            _perturb(gradient, direction, amount), **kwargs).energy_density_reference_j_m3
    h = F(1, 1000)
    # The Green-strain energy is quartic in this directional variable. This
    # four-point identity returns its exact linear coefficient, without using P.
    independent = (8*(energy(h)-energy(-h))-(energy(2*h)-energy(-2*h)))/(12*h)
    assert response.first_variation_reference_j_m3 == independent


def test_rigid_motion_has_zero_energy_and_zero_stress():
    rotation = ((F(3, 5), F(-4, 5), ZERO),
        (F(4, 5), F(3, 5), ZERO), (ZERO, ZERO, F(1)))
    response = _api().evaluate_native_bulk_deformation(_closure(), rotation,
        reference_density_kg_m3=F(3300), reference_volume_m3=F(1))
    assert response.energy_density_reference_j_m3 == 0
    assert response.first_piola_stress_pa == ((ZERO,)*3,)*3


def test_shared_bank_condensation_contains_real_cross_contact_compliance():
    result = _api().condense_shared_bank_elasticity(
        stiffness=((F(4), F(1)), (F(1), F(2))),
        face_operators=((F(1), ZERO), (ZERO, F(1)), (F(1), F(1))),
        face_ids=("first-born-face", "second-born-face", "third-shared-face"),
        bank_source_id="one-actual-native-bank", source_support_sha256=SOURCE_SUPPORT,
        geometry_source_sha256=GEOMETRY_SOURCE,
        provenance="Declared coupled whole-bank stiffness, not separate springs")
    assert result.unit_face_load_displacements_m_n == (
        (F(2, 7), F(-1, 7)), (F(-1, 7), F(4, 7)), (F(1, 7), F(3, 7)))
    assert result.compliance_m_n == ((F(2, 7), F(-1, 7), F(1, 7)),
        (F(-1, 7), F(4, 7), F(3, 7)), (F(1, 7), F(3, 7), F(4, 7)))
    loads = (F(2), F(-3), F(5))
    elastic_work = sum((loads[i]*result.compliance_m_n[i][j]*loads[j]
        for i in range(3) for j in range(3)), ZERO)/2
    assert elastic_work == F(43, 7)


@pytest.mark.parametrize("changes", [dict(source_bank="receiver"),
    dict(bank_source_id=""), dict(source_support_sha256="self-sealed"),
    dict(lateral_closure_provenance=""), dict(lateral_closure="undeclared")])
def test_closure_requires_actual_bank_binding_and_explicit_lateral_boundary_declaration(changes):
    with pytest.raises((ValueError, TypeError)):
        _closure(**changes)


def test_incompressible_moduli_cannot_be_silently_used_as_finite_lame_elasticity():
    columns = ElasticColumnParameters(1000., 12.e8, 4.e8, 1000., 12.e8, 4.e8)
    with pytest.raises((ValueError, TypeError)):
        _closure(source_columns=columns)


@pytest.mark.parametrize("strain", [((F(1), F(2), ZERO), (ZERO, F(1), ZERO),
    (ZERO, ZERO, F(1))), ((0.,)*3,)*3, ((True, ZERO, ZERO), (ZERO, ZERO, ZERO),
    (ZERO, ZERO, ZERO)), ((F(1), ZERO), (ZERO, F(1)))])
def test_full_bulk_response_rejects_nonsymmetric_or_nonliteral_strain(strain):
    with pytest.raises((ValueError, TypeError)):
        _api().evaluate_native_bulk_elasticity(_closure(), strain)


@pytest.mark.parametrize("gradient", [_diag(-1, 1, 1), _diag(0, 1, 1)])
def test_orientation_reversal_or_collapsed_material_volume_is_refused(gradient):
    with pytest.raises((ValueError, TypeError)):
        _api().evaluate_native_bulk_deformation(_closure(), gradient,
            reference_density_kg_m3=F(3300), reference_volume_m3=F(12))


@pytest.mark.parametrize("stiffness", [((F(1), F(2)), (F(2), F(1))),
    ((F(1), F(2)), (ZERO, F(1))), ((ZERO, ZERO), (ZERO, F(1)))])
def test_shared_bank_cannot_condense_nonpositive_or_nonsymmetric_bulk_energy(stiffness):
    with pytest.raises((ValueError, TypeError)):
        _api().condense_shared_bank_elasticity(stiffness=stiffness,
            face_operators=((F(1), ZERO),), face_ids=("first-born-face",),
            bank_source_id="one-actual-native-bank", source_support_sha256=SOURCE_SUPPORT,
            geometry_source_sha256=GEOMETRY_SOURCE, provenance="Explicit shared-bank stiffness")


def test_exact_bulk_tensor_retains_energy_of_a_mode_far_below_float_resolution():
    shortening = F(1, 10**600)
    response = _api().evaluate_native_bulk_elasticity(_closure(),
        _diag(-shortening, shortening/4, shortening/4))
    assert response.energy_density_j_m3 == F(10**9)*shortening**2/2
    assert response.energy_density_j_m3 > 0
    assert float(response.energy_density_j_m3) == 0.


def _shared_bank():
    return _api().condense_shared_bank_elasticity(
        stiffness=((F(4), F(1)), (F(1), F(2))),
        face_operators=((F(1), ZERO), (ZERO, F(1)), (F(1), F(1))),
        face_ids=("first", "second", "third"), bank_source_id="one-actual-native-bank",
        source_support_sha256=SOURCE_SUPPORT, geometry_source_sha256=GEOMETRY_SOURCE,
        provenance="Explicit coupled source bank operator")


def test_signed_shared_face_loads_advance_one_bank_with_exact_force_and_work_recovery():
    response = _api().evaluate_shared_bank_face_loads(_shared_bank(), (F(2), F(-3), F(5)))
    assert response.body_displacements_m == (F(12, 7), F(1, 7))
    assert response.face_displacements_m == (F(12, 7), F(1, 7), F(13, 7))
    assert response.body_elastic_gradient_n == (F(7), F(2))
    assert response.force_residual_n == (ZERO, ZERO)
    assert response.strain_energy_j == F(43, 7)
    assert response.discrete_declared_operator_equilibrium_verified
    assert not response.force_equilibrium_certified
    assert not response.complete_three_dimensional_strain_energy_certified
    assert not response.material_admission_authorized


def test_condensation_preserves_arbitrarily_weak_positive_bank_mode_without_floor():
    weak = F(1, 10**600)
    result = _api().condense_shared_bank_elasticity(
        stiffness=((F(1), ZERO), (ZERO, weak)),
        face_operators=((F(1), ZERO), (ZERO, F(1))), face_ids=("first", "second"),
        bank_source_id="one-actual-native-bank", source_support_sha256=SOURCE_SUPPORT,
        geometry_source_sha256=GEOMETRY_SOURCE, provenance="Explicit tiny positive body mode")
    assert result.compliance_m_n == ((F(1), ZERO), (ZERO, 1/weak))
    response = _api().evaluate_shared_bank_face_loads(result, (ZERO, weak))
    assert response.body_displacements_m == (ZERO, F(1))
    assert response.strain_energy_j == weak/2
    assert response.strain_energy_j > 0


def test_resealed_cached_compliance_cannot_replace_full_shared_bank_replay():
    genuine = _shared_bank()
    forged = replace(genuine, compliance_m_n=((F(1), ZERO, ZERO),)*3)
    with pytest.raises((ValueError, TypeError), match="replay|condensation|shared|Shared"):
        _api().evaluate_shared_bank_face_loads(forged, (F(1), ZERO, ZERO))
    assert genuine.compliance_m_n[0][1] == F(-1, 7)


def test_local_closure_and_response_do_not_claim_whole_material_mask_or_admission():
    closure = _closure()
    response = _api().evaluate_native_bulk_elasticity(closure, _diag(0, 0, 0))
    assert closure.new_lateral_closure_declared
    assert not closure.previous_three_dimensional_tensor_inherited
    assert not closure.source_column_lengths_are_actual_bank_geometry
    for item in (closure, response, _shared_bank()):
        assert not item.complete_three_dimensional_strain_energy_certified
        assert not item.force_equilibrium_certified
        assert not item.material_admission_authorized
