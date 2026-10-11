"""Physical activation, unilateral work and independently owned HOT support."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import hashlib
import json

import pytest

from tectonics.genesis_contact_law import ContactLawParameters
from tectonics.geometric_contact_mechanics import ElasticColumnParameters, solve_compressed_contact, state_to_dict
from tectonics.geometric_native_birth_contact import (
    BornContactParameters, BornContactReference, BornFoundationParameters,
    BornFoundationReference, activate_born_contact, activate_born_foundation,
    advance_born_contact, advance_born_foundation, effective_normal_stiffness,
    foundation_stiffness_n_m, born_contact_state_to_dict, born_contact_state_from_dict,
    born_foundation_state_to_dict, born_foundation_state_from_dict,
    validate_born_contact_state, validate_born_foundation_state,
)
from tectonics.geometric_native_polygons import NativePlane


def parameters():
    columns = ElasticColumnParameters(1., 200., 80., 1., 200., 80.)
    law = ContactLawParameters(normal_stiffness_pa_m=100., tangential_stiffness_pa_m=80.,
        tensile_strength_pa=10., fracture_energy_j_m2=10., cohesion_pa=0.,
        friction_dry=0., friction_wet=0., viscosity_pa_s_m=0.)
    return BornContactParameters(columns, law, "independent explicit solid HOT test phase")


def reference(**changes):
    row = BornContactReference("HOT:interface", "a"*64, "b"*64,
        "donor:birth:1", "receiver:birth:1", "HOT:donor", "HOT:receiver",
        "parent:donor", "parent:receiver", 1, 0, NativePlane((1, 0, 0), label="declared weak plane"),
        Fraction(1, 100), 2., -.4, 100., "actual gap retained; independent inclined reference area")
    return replace(row, **changes)


def foundation_reference(**changes):
    row = BornFoundationReference("HOT:foundation", "a"*64, "b"*64,
        "receiver:birth:1", "HOT:receiver", "parent:receiver", 0,
        Fraction(1, 100), 3., 3., "separate basal material reference area")
    return replace(row, **changes)


def foundation_parameters():
    return BornFoundationParameters(10., 2., "explicit hydrostatic test substrate")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def reseal(raw):
    raw["state_sha256"] = hashlib.sha256(encoded({k: v for k, v in raw.items()
        if k != "state_sha256"})).hexdigest()


def test_birth_preserves_negative_actual_gap_and_pays_activation_energy():
    params, ref = parameters(), reference()
    state = activate_born_contact(params, ref)
    assert effective_normal_stiffness(params) == 50.
    assert state.gap_free_m == -.4
    assert state.normal_traction_pa == -20.
    assert state.normal_contact_energy_j == pytest.approx(8.)
    assert state.activation_work_j == state.normal_contact_energy_j
    assert state.cumulative_external_work_j == 0.
    assert state.cumulative_numerical_dissipation_j == 0.
    assert state.step_count == 0 and state.elapsed_time_s == 0.


def test_compression_force_and_backward_euler_balance():
    params, ref = parameters(), reference()
    before = activate_born_contact(params, ref)
    result = advance_born_contact(before, params, ref, gap_free_m=-.8, jump_free_m=101., dt_s=2.)
    assert result.normal_traction_pa == -40.
    assert result.energy_gradient_n == (-80., 0.)
    assert result.restoring_force_n == (80., 0.)
    assert result.tangent_n_m == ((100., 0.), (0., 0.))
    assert result.external_work_j == pytest.approx(32.)
    assert result.stored_energy_increment_j == pytest.approx(24.)
    assert result.numerical_dissipation_j == pytest.approx(8.)
    assert result.energy_balance_relative_error < 1e-15
    assert before.normal_contact_energy_j == pytest.approx(8.)


def test_opening_then_reclosure_is_unbonded_and_does_not_reset_actual_gap():
    params, ref = parameters(), reference()
    before = activate_born_contact(params, ref)
    opened = advance_born_contact(before, params, ref, gap_free_m=.2, jump_free_m=120., dt_s=1.)
    assert opened.normal_traction_pa == 0. and opened.normal_tangent_pa_m == 0.
    assert opened.state.normal_contact_energy_j == 0.
    assert opened.external_work_j == 0.
    assert opened.numerical_dissipation_j == pytest.approx(8.)
    reclosed = advance_born_contact(opened.state, params, ref, gap_free_m=-.4, jump_free_m=121., dt_s=1.)
    assert reclosed.normal_traction_pa == -20.
    assert reclosed.state.normal_contact_energy_j == pytest.approx(8.)
    assert reclosed.external_work_j == pytest.approx(24.)
    assert reclosed.numerical_dissipation_j == pytest.approx(16.)
    assert reclosed.state.activation_work_j == pytest.approx(8.)
    assert reclosed.state.cumulative_external_work_j == pytest.approx(24.)
    assert reclosed.state.cumulative_numerical_dissipation_j == pytest.approx(24.)


def test_open_birth_never_acquires_cohesion_or_shear_force():
    params, ref = parameters(), reference(birth_gap_free_m=.25)
    state = activate_born_contact(params, ref)
    result = advance_born_contact(state, params, ref, gap_free_m=.5, jump_free_m=-1e5, dt_s=10.)
    assert state.activation_work_j == 0.
    assert result.state.normal_contact_energy_j == 0.
    assert result.energy_gradient_n == (0., 0.)
    assert result.external_work_j == 0.
    assert result.numerical_dissipation_j == 0.
    assert result.state.shear_traction_pa == 0.


def test_shear_virgin_reference_preserves_independent_cold_history():
    params = parameters()
    cold = solve_compressed_contact(columns=params.columns, law=params.law,
        area_m2=2., gap_free_m=-.4, jump_free_m=-3., dt_s=2., water=0.).state
    original = encoded(state_to_dict(cold))
    ref = reference(birth_jump_free_m=cold.jump_free_m)
    born = activate_born_contact(params, ref)
    assert born.plastic_slip_m == 0.
    evolved = advance_born_contact(born, params, ref, gap_free_m=-.4,
        jump_free_m=ref.birth_jump_free_m+1., dt_s=1.)
    reversed_step = advance_born_contact(evolved.state, params, ref, gap_free_m=-.4,
        jump_free_m=ref.birth_jump_free_m-1., dt_s=1.)
    assert evolved.signed_slip_increment_m == 1.
    assert reversed_step.signed_slip_increment_m == -2.
    assert reversed_step.state.plastic_slip_m == -1.
    assert reversed_step.state.cumulative_slip_m == 3.
    assert reversed_step.external_work_j == 0.
    assert encoded(state_to_dict(cold)) == original


def test_contact_energy_and_reaction_scale_with_interface_area_only():
    params = parameters()
    ref = reference()
    small = replace(ref, interface_area_m2=ref.interface_area_m2*1e-36)
    first = advance_born_contact(activate_born_contact(params, ref), params, ref,
        gap_free_m=-.8, jump_free_m=101., dt_s=2.)
    second = advance_born_contact(activate_born_contact(params, small), params, small,
        gap_free_m=-.8, jump_free_m=101., dt_s=2.)
    assert first.normal_traction_pa == second.normal_traction_pa
    assert first.state.plastic_slip_m == second.state.plastic_slip_m
    assert second.energy_gradient_n[0]/first.energy_gradient_n[0] == pytest.approx(1e-36)
    assert second.external_work_j/first.external_work_j == pytest.approx(1e-36)
    assert second.state.activation_work_j/first.state.activation_work_j == pytest.approx(1e-36)
    assert second.energy_balance_relative_error < 1e-15


def test_positive_tiny_interface_energy_and_work_are_not_thresholded():
    params, ref = parameters(), reference(interface_area_m2=1e-52, birth_gap_free_m=-1e-20)
    born = activate_born_contact(params, ref)
    result = advance_born_contact(born, params, ref, gap_free_m=-2e-20, jump_free_m=100., dt_s=1.)
    assert 0. < born.activation_work_j < 1e-80
    assert result.external_work_j > 0.
    assert result.numerical_dissipation_j > 0.
    assert result.state.normal_contact_energy_j > born.normal_contact_energy_j
    assert result.energy_balance_relative_error < 1e-15


def test_unrepresentable_nonzero_traction_is_refused_before_becoming_zero_force():
    columns = ElasticColumnParameters(1., 4e-200, 4e-200, 1., 4e-200, 4e-200)
    law = ContactLawParameters(normal_stiffness_pa_m=2e-200, tangential_stiffness_pa_m=2e-200,
        tensile_strength_pa=1e-200, fracture_energy_j_m2=1e-200, cohesion_pa=0.,
        friction_dry=0., friction_wet=0., viscosity_pa_s_m=0.)
    params = BornContactParameters(columns, law, "explicit range-limit stress test")
    ref = reference(interface_area_m2=1e300, birth_gap_free_m=-1e-150)
    with pytest.raises(ValueError, match="finite representable range"):
        activate_born_contact(params, ref)


def test_huge_integer_input_is_refused_by_explicit_finite_number_guard():
    with pytest.raises(ValueError, match="finite real number"):
        BornFoundationParameters(10**400, 1., "explicit malformed range test")


@pytest.mark.parametrize("end_gap", [-1., -.4, 0., .4])
def test_convex_contact_remainder_is_nonnegative_and_signed_work_is_balanced(end_gap):
    params, ref = parameters(), reference()
    born = activate_born_contact(params, ref)
    result = advance_born_contact(born, params, ref, gap_free_m=end_gap, jump_free_m=101., dt_s=1.)
    assert result.numerical_dissipation_j >= 0.
    assert result.external_work_j == pytest.approx(
        result.stored_energy_increment_j+result.numerical_dissipation_j, abs=1e-14)
    assert result.state.activation_work_j+result.state.cumulative_external_work_j == pytest.approx(
        result.state.normal_contact_energy_j+result.state.cumulative_numerical_dissipation_j)


def test_hydrostatic_support_uses_independent_basal_area_and_birth_radial_reference():
    params, ref = foundation_parameters(), foundation_reference()
    state = activate_born_foundation(params, ref)
    assert foundation_stiffness_n_m(params, ref) == 60.
    assert state.radial_displacement_m == 3. and state.foundation_energy_j == 0.
    result = advance_born_foundation(state, params, ref, radial_displacement_m=4., dt_s=2.)
    assert result.restoring_force_n == -60.
    assert result.energy_gradient_n == 60. and result.tangent_n_m == 60.
    assert result.state.foundation_energy_j == 30.
    assert result.external_work_j == 60.
    assert result.numerical_dissipation_j == 30.
    assert result.energy_balance_relative_error == 0.


def test_hydrostatic_unloading_preserves_signed_external_work_and_balance():
    params, ref = foundation_parameters(), foundation_reference()
    initial = activate_born_foundation(params, ref)
    loaded = advance_born_foundation(initial, params, ref, radial_displacement_m=4., dt_s=1.)
    unloaded = advance_born_foundation(loaded.state, params, ref, radial_displacement_m=3.5, dt_s=1.)
    assert unloaded.external_work_j == -15.
    assert unloaded.stored_energy_increment_j == -22.5
    assert unloaded.numerical_dissipation_j == 7.5
    assert unloaded.state.foundation_energy_j == 7.5
    assert unloaded.state.cumulative_external_work_j == 45.
    assert unloaded.state.cumulative_numerical_dissipation_j == 37.5


def test_foundation_is_owned_once_when_two_interfaces_share_one_hot_component():
    support_params, support_ref = foundation_parameters(), foundation_reference()
    support = advance_born_foundation(activate_born_foundation(support_params, support_ref),
        support_params, support_ref, radial_displacement_m=4., dt_s=1.)
    params, first_ref = parameters(), reference()
    second_ref = replace(first_ref, family_id="HOT:another-interface", donor_cohort_id="donor:birth:2")
    first, second = activate_born_contact(params, first_ref), activate_born_contact(params, second_ref)
    assert not hasattr(first, "foundation_energy_j")
    assert not hasattr(second, "foundation_energy_j")
    assert support.state.foundation_energy_j == 30.
    assert first.normal_contact_energy_j+second.normal_contact_energy_j+support.state.foundation_energy_j == pytest.approx(46.)


def test_foundation_area_scaling_and_tiny_positive_energy():
    params, ref = foundation_parameters(), foundation_reference(basal_area_m2=1e-36, radial_birth_m=0.)
    result = advance_born_foundation(activate_born_foundation(params, ref), params, ref,
        radial_displacement_m=1e-20, dt_s=1.)
    assert result.state.foundation_energy_j == pytest.approx(1e-75, rel=1e-14, abs=0.)
    assert result.restoring_force_n < 0.
    assert result.external_work_j > 0. and result.numerical_dissipation_j > 0.
    assert result.energy_balance_relative_error < 1e-15


def test_contact_restart_replays_full_history_and_continues_identically():
    params, ref = parameters(), reference()
    state = activate_born_contact(params, ref)
    for gap, jump in ((-.8, 101.), (.2, 100.), (-.4, 99.)):
        state = advance_born_contact(state, params, ref, gap_free_m=gap, jump_free_m=jump, dt_s=2.).state
    raw = born_contact_state_to_dict(state, expected_parameters=params, expected_reference=ref)
    restored = born_contact_state_from_dict(json.loads(encoded(raw)), expected_parameters=params, expected_reference=ref)
    assert encoded(born_contact_state_to_dict(restored, expected_parameters=params, expected_reference=ref)) == encoded(raw)
    a = advance_born_contact(state, params, ref, gap_free_m=-.9, jump_free_m=100., dt_s=3.)
    b = advance_born_contact(restored, params, ref, gap_free_m=-.9, jump_free_m=100., dt_s=3.)
    assert a == b


def test_foundation_restart_replays_history_and_continues_identically():
    params, ref = foundation_parameters(), foundation_reference()
    state = advance_born_foundation(activate_born_foundation(params, ref), params, ref,
        radial_displacement_m=4., dt_s=2.).state
    raw = born_foundation_state_to_dict(state, expected_parameters=params, expected_reference=ref)
    restored = born_foundation_state_from_dict(json.loads(encoded(raw)), expected_parameters=params, expected_reference=ref)
    assert restored == state
    assert advance_born_foundation(restored, params, ref, radial_displacement_m=3.5, dt_s=1.) == advance_born_foundation(
        state, params, ref, radial_displacement_m=3.5, dt_s=1.)


@pytest.mark.parametrize("mutation", [
    lambda raw: raw.update(step_count=False),
    lambda raw: raw.update(normal_contact_energy_j=16.),
    lambda raw: raw["reference"].update(interface_area_m2=2),
    lambda raw: raw["reference"].update(donor_owner=True),
    lambda raw: raw["reference"].update(birth_time_myr=[1, False]),
    lambda raw: raw["parameters"]["law"].update(friction_dry=False),
    lambda raw: raw["history"][0].update(dt_s=2),
    lambda raw: raw.update(unexpected_field=0),
])
def test_resealed_contact_dto_rejects_typed_aliases_and_changed_physics(mutation):
    params, ref = parameters(), reference()
    state = advance_born_contact(activate_born_contact(params, ref), params, ref,
        gap_free_m=-.8, jump_free_m=101., dt_s=2.).state
    raw = deepcopy(born_contact_state_to_dict(state, expected_parameters=params, expected_reference=ref))
    mutation(raw)
    reseal(raw)
    with pytest.raises(ValueError):
        born_contact_state_from_dict(raw, expected_parameters=params, expected_reference=ref)


def test_resealed_reference_or_parameters_cannot_become_their_own_authority():
    params, ref = parameters(), reference()
    raw = born_contact_state_to_dict(activate_born_contact(params, ref), expected_parameters=params, expected_reference=ref)
    changed = deepcopy(raw)
    changed["reference"]["source_epoch_sha256"] = "c"*64
    reseal(changed)
    with pytest.raises(ValueError, match="Independent full birth reference"):
        born_contact_state_from_dict(changed, expected_parameters=params, expected_reference=ref)
    changed = deepcopy(raw)
    changed["parameters"]["provenance"] += ":changed"
    reseal(changed)
    with pytest.raises(ValueError, match="Independent full birth parameters"):
        born_contact_state_from_dict(changed, expected_parameters=params, expected_reference=ref)


def test_resealed_foundation_dto_rejects_energy_and_area_substitution():
    params, ref = foundation_parameters(), foundation_reference()
    state = advance_born_foundation(activate_born_foundation(params, ref), params, ref,
        radial_displacement_m=4., dt_s=1.).state
    raw = born_foundation_state_to_dict(state, expected_parameters=params, expected_reference=ref)
    for key, value in (("foundation_energy_j", 60.), ("step_count", True)):
        changed = deepcopy(raw)
        changed[key] = value
        reseal(changed)
        with pytest.raises(ValueError):
            born_foundation_state_from_dict(changed, expected_parameters=params, expected_reference=ref)
    changed = deepcopy(raw)
    changed["reference"]["basal_area_m2"] = 2.
    reseal(changed)
    with pytest.raises(ValueError, match="Independent full birth reference"):
        born_foundation_state_from_dict(changed, expected_parameters=params, expected_reference=ref)


def test_in_memory_typed_alias_cannot_hide_behind_a_valid_seal():
    params, ref = parameters(), reference()
    state = activate_born_contact(params, ref)
    old = state.cumulative_external_work_j
    object.__setattr__(state, "cumulative_external_work_j", False)
    try:
        with pytest.raises(ValueError, match="literal scalar type"):
            validate_born_contact_state(state, expected_parameters=params, expected_reference=ref)
    finally:
        object.__setattr__(state, "cumulative_external_work_j", old)
    assert validate_born_contact_state(state, expected_parameters=params, expected_reference=ref) is state


@pytest.mark.parametrize("name,replacement", [("normal", (2, 0, 0)),
    ("label", type("TextAlias", (str,), {})("declared weak plane"))])
def test_in_memory_native_plane_canonicalization_cannot_hide_mutation(name, replacement):
    params, ref = parameters(), reference()
    state = activate_born_contact(params, ref)
    independent = deepcopy(ref)
    plane = state.reference.plane
    old = getattr(plane, name)
    object.__setattr__(plane, name, replacement)
    try:
        with pytest.raises(ValueError, match="native birth plane"):
            validate_born_contact_state(state, expected_parameters=params, expected_reference=independent)
    finally:
        object.__setattr__(plane, name, old)
    assert validate_born_contact_state(state, expected_parameters=params, expected_reference=independent) is state


@pytest.mark.parametrize("changes", [dict(friction_dry=.1), dict(friction_wet=.1, friction_dry=.1),
    dict(cohesion_pa=1.), dict(viscosity_pa_s_m=1.)])
def test_nonfrictionless_laws_require_a_future_explicit_model_extension(changes):
    params = parameters()
    with pytest.raises(ValueError, match="zero friction"):
        BornContactParameters(params.columns, replace(params.law, **changes), params.provenance)


def test_birth_validation_rejects_implicit_or_nonphysical_inputs():
    with pytest.raises(ValueError):
        BornFoundationParameters(0., 1., "declared")
    with pytest.raises(ValueError):
        BornFoundationParameters(1., True, "declared")
    with pytest.raises(ValueError):
        reference(donor_owner=True)
    with pytest.raises(ValueError):
        reference(interface_area_m2=0.)
    with pytest.raises(ValueError):
        reference(birth_time_myr=0.)
    params, ref = foundation_parameters(), foundation_reference()
    state = activate_born_foundation(params, ref)
    with pytest.raises(ValueError):
        advance_born_foundation(state, params, ref, radial_displacement_m=4., dt_s=0.)
    with pytest.raises(ValueError):
        validate_born_foundation_state(state, expected_parameters=params,
            expected_reference=replace(ref, owner=1))
