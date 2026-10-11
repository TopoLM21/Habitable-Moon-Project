"""Curved forces retain physical work and each coordinate's own budget."""
from dataclasses import replace
from fractions import Fraction as F

import pytest

from tectonics.geometric_native_pose import FractionInterval as I


def _api():
    from tectonics import geometric_native_curved_equilibrium as module
    return module


def _map(gradient, **changes):
    arguments = dict(coordinate_count=19, donor_angular_indices=(0, 1, 2),
        receiver_angular_indices=(3, 4, 5), donor_radial_index=6,
        receiver_radial_index=14, radius_m=F(100000))
    arguments.update(changes)
    return _api().map_curved_contact_covector(gradient, **arguments)


def test_curved_covector_pullback_preserves_physical_virtual_work():
    gradient = tuple(I(F(index-3, 7)) for index in range(8))
    result = _map(gradient)
    velocity = tuple(F(index+1, 23) for index in range(19))
    physical_velocity = (*velocity[:6], velocity[6]*100000, velocity[14]*100000)
    local_work = sum((force*speed for force, speed in zip(gradient, physical_velocity)), I(0))
    assembled_work = sum((force*speed for force, speed in zip(result, velocity)), I(0))
    assert assembled_work == local_work
    assert all(result[index] == I(0) for index in (7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18))


def test_shared_owner_angular_covectors_aggregate_instead_of_overwrite():
    gradient = tuple(I(F(index+1)) for index in range(8))
    result = _map(gradient, receiver_angular_indices=(0, 1, 2))
    assert result[:3] == (I(5), I(7), I(9))
    assert result[6] == I(700000)
    assert result[14] == I(800000)
    assert all(value == I(0) for index, value in enumerate(result) if index not in (0, 1, 2, 6, 14))


def test_interval_force_uncertainty_survives_radial_pullback():
    gradient = (I(0),)*6+(I(F(1, 10**600), F(2, 10**600)), I(0))
    result = _map(gradient)
    assert result[6] == I(F(1, 10**595), F(2, 10**595))
    assert float(result[6].hi) == 0.


def test_all_nineteen_force_rows_are_certified_against_their_own_operands():
    rows = tuple((I(F(10**30)), I(F(-10**30))) for _ in range(18))
    tiny = F(1, 10**600)
    rows += ((I(tiny), I(-tiny)),)
    certificate = _api().certify_curved_force_operands(rows,
        relative_tolerance=F(1, 10**10), precision=160)
    assert certificate.equilibrium_certified
    assert certificate.own_scale_lower_bounds[-1] == 2*tiny
    assert certificate.relative_residual_upper_bounds == (F(0),)*19


def test_planetary_load_cannot_hide_tiny_coordinate_imbalance():
    rows = tuple((I(F(10**30)), I(F(-10**30))) for _ in range(18))
    tiny = F(1, 10**600)
    rows += ((I(tiny), I(-tiny/2)),)
    with pytest.raises(ValueError, match="coordinate 18|own budget"):
        _api().certify_curved_force_operands(rows,
            relative_tolerance=F(1, 10**10), precision=160)


def test_zero_midpoint_does_not_certify_uncertain_contact_force():
    rows = ((I(-F(1, 10**600), F(1, 10**600)),),)
    with pytest.raises(ValueError, match="own budget|coordinate"):
        _api().certify_curved_force_operands(rows,
            relative_tolerance=F(1, 10**10), precision=160)


def test_uncertain_edge_force_cannot_pass_using_another_coordinates_planetary_scale():
    rows = tuple((I(F(10**30)), I(F(-10**30))) for _ in range(18))
    tiny = F(1, 10**600)
    rows += ((I(tiny), I(-11*tiny/10, -9*tiny/10)),)
    assert sum(rows[-1], I(0)).midpoint == 0
    with pytest.raises(ValueError, match="coordinate 18|own budget"):
        _api().certify_curved_force_operands(rows,
            relative_tolerance=F(1, 10**10), precision=160)


@pytest.mark.parametrize("changes", [dict(radius_m=100000.), dict(radius_m=F(0)),
    dict(coordinate_count=True), dict(donor_angular_indices=(0, 1, 19)),
    dict(donor_radial_index=-1), dict(receiver_radial_index=True),
    dict(receiver_radial_index=6)])
def test_curved_covector_rejects_aliased_or_invalid_physical_coordinate_labels(changes):
    with pytest.raises((ValueError, TypeError)):
        _map((I(0),)*8, **changes)


def test_virgin_curved_reference_preserves_the_existing_activation_state():
    from tests.test_geometric_native_birth_contact_continuum import _continuum_case
    attachment = _continuum_case()["born_continuum_attachments"][0]
    before = attachment.previous_state
    _api().require_virgin_curved_activation_sources((attachment,))
    assert attachment.previous_state is before
    assert before.activation_energy_decimal != "0"
    assert before.history == () and before.step_count == 0


def test_already_advanced_contact_cannot_be_reactivated_as_new_curved_model():
    from tests.test_geometric_native_birth_contact_continuum import _continuum_case
    from tectonics.geometric_native_birth_contact_continuum import solve_born_continuum_network
    inputs = _continuum_case()
    attachment = inputs["born_continuum_attachments"][0]
    next_state = solve_born_continuum_network(**inputs).continuum_contact_steps[0].state
    advanced = replace(attachment, previous_state=next_state)
    with pytest.raises(ValueError, match="virgin|history|first|adopted|activation"):
        _api().require_virgin_curved_activation_sources((advanced,))
    assert advanced.previous_state is next_state
    assert len(next_state.history) == 1


def test_first_birth_cannot_delete_the_paid_source_activation_energy():
    from tests.test_geometric_native_birth_contact_continuum import _continuum_case
    attachment = _continuum_case()["born_continuum_attachments"][0]
    original = attachment.previous_state
    unpaid = replace(original, activation_energy_decimal="0", stored_energy_decimal="0")
    with pytest.raises(ValueError, match="energy|activation|replay|ledger"):
        _api().require_virgin_curved_activation_sources((replace(attachment, previous_state=unpaid),))
    assert original.activation_energy_decimal != "0"


def test_untyped_born_activation_source_cannot_supply_memory_authority():
    with pytest.raises((ValueError, TypeError)):
        _api().require_virgin_curved_activation_sources((object(),))


def test_coarse_contact_upper_force_bound_cannot_become_a_final_lower_scale_floor():
    tiny = F(1, 10**31)
    references = ((I(tiny),),)
    coarse = ((I(-F(1, 10**9), F(0)),),)
    lower, upper = _api().curved_force_width_scales(references, coarse)
    assert lower == (tiny,)
    assert upper == (tiny+F(1, 10**9),)
    refined = ((I(-tiny),),)
    next_lower, next_upper = _api().curved_force_width_scales(references, refined)
    assert next_lower == next_upper == (2*tiny,)
    assert next_upper[0] < upper[0]*F(1, 10**20)


def test_width_scale_refinement_preserves_exact_zero_and_independent_tiny_rows():
    tiny, planet = F(1, 10**600), F(10**30)
    references = ((I(planet), I(-planet)), (I(0),), (I(tiny),))
    covectors = ((I(0), I(0), I(-tiny)),)
    lower, upper = _api().curved_force_width_scales(references, covectors)
    assert lower == upper == (2*planet, F(0), 2*tiny)


@pytest.mark.parametrize("covectors", [((I(0), I(0)),), ((0.,),), [((I(0),),)]])
def test_force_width_scale_helper_refuses_incomplete_or_aliased_covectors(covectors):
    with pytest.raises(ValueError):
        _api().curved_force_width_scales(((I(0),),), covectors)
