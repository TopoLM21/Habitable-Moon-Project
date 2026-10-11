"""Exact physical-coordinate normalization without a native source rebuild."""
from dataclasses import replace
from decimal import Decimal, Inexact, ROUND_FLOOR, localcontext
from fractions import Fraction as F
import json
from types import SimpleNamespace as NS

import pytest

from tectonics.geometric_native_birth_exact_radial import (
    build_native_birth_exact_radial_inputs, exact_radial_before_coordinate,
    normalize_native_birth_radial_before, radial_before_normalization_to_dict,
)


def _physical_inputs(*, radius=100000., radial=(-.05734914863803788, .04870854357657349), hot=0.):
    trace = NS(trace_id="saved-trace")
    previous = NS(initial_mechanics=NS(fault_traces=NS(traces=(trace,))),
        transverse_displacement_m=((trace.trace_id, *radial),))
    foundation = NS(radial_dof_index=8, reference=NS(patch_id="HOT2"),
        previous_state=NS(radial_displacement_m=hot, paid_birth_energy="preserved"))
    inputs = dict(radius_m=radius, rotation_dof_count=6,
        old_stiffness=tuple(tuple(float(i == j) for j in range(8)) for i in range(8)),
        born_foundations=(foundation,), coordinates_before=(0.,)*6+
            tuple(value/radius for value in radial)+(hot/radius,), decimal_precision=160,
        born_continuum_attachments=(object(),), old_states=(object(),),
        old_anchor_operators=(object(),), drag_nm_s=(object(),), source_torque_nm=(object(),),
        dt_s=1e6, old_materials=(object(),), old_point_ids=("unchanged",))
    return inputs, previous


def test_saved_native_alias_is_removed_without_changing_the_physical_bank():
    value, radius = -.05734914863803788, 100000.
    entry = exact_radial_before_coordinate(value, radius, coordinate_index=8, bank_id="cold6")
    exact = F.from_float(value)
    reconstructed = F(Decimal(entry.coordinate_decimal))*F.from_float(radius)
    alias = F.from_float(value/radius)*F.from_float(radius)-exact
    assert entry.physical_before_m == exact
    assert alias == entry.binary64_reconstruction_error_m
    assert 1.4873e-18 < float(alias) < 1.4874e-18
    assert reconstructed-exact == entry.physical_reconstruction_error_m
    assert abs(reconstructed-exact) <= entry.physical_quotient_error_bound_m
    assert entry.physical_quotient_error_bound_m < F(1, 10**158)
    assert str(Decimal(entry.coordinate_decimal)) == entry.coordinate_decimal


def test_nonterminating_quotient_records_actual_signed_error():
    entry = exact_radial_before_coordinate(1., 3., decimal_precision=100)
    assert F(Decimal(entry.coordinate_decimal)) < F(1, 3)
    assert entry.normalized_quotient_error == F(Decimal(entry.coordinate_decimal))-F(1, 3)
    assert entry.physical_reconstruction_error_m == 3*entry.normalized_quotient_error
    assert entry.physical_quotient_error_bound_m == abs(entry.physical_reconstruction_error_m)
    assert 0 < entry.physical_quotient_error_bound_m < F(1, 10**99)


def test_exactly_terminating_division_has_zero_error():
    entry = exact_radial_before_coordinate(.125, 2.)
    assert entry.coordinate_decimal == "0.0625"
    assert entry.physical_reconstruction_error_m == 0
    assert entry.physical_quotient_error_bound_m == 0


def test_precision_and_rounding_are_local_and_independent_of_callers_context():
    reference = exact_radial_before_coordinate(1., 3.)
    with localcontext() as context:
        context.prec = 5
        context.rounding = ROUND_FLOOR
        context.traps[Inexact] = True
        before = (context.prec, context.rounding, dict(context.traps), dict(context.flags))
        actual = exact_radial_before_coordinate(1., 3.)
        assert actual == reference
        assert (context.prec, context.rounding, dict(context.traps), dict(context.flags)) == before


def test_adapter_replaces_only_radial_before_coordinates_and_preserves_every_physical_owner():
    inputs, previous = _physical_inputs()
    original_before = inputs["coordinates_before"]
    normalized, evidence = normalize_native_birth_radial_before(inputs, previous)
    assert normalized is not inputs
    assert inputs["coordinates_before"] is original_before
    assert normalized["coordinates_before"][:6] == original_before[:6]
    assert all(type(value) is str for value in normalized["coordinates_before"][6:])
    for key in inputs:
        if key != "coordinates_before":
            assert normalized[key] is inputs[key]
    assert tuple(item.coordinate_index for item in evidence.coordinates) == (6, 7, 8)
    assert tuple(item.bank_id for item in evidence.coordinates) == ("saved-trace:donor", "saved-trace:receiver", "HOT2")
    assert evidence.radius_m == 100000
    assert evidence.material_admission_authorized is False
    # This block is already expressed in w/R. Normalization must not apply R²
    # again or alter its own stiffness matrix object.
    assert normalized["old_stiffness"][6][6] == 1.
    assert normalized["old_stiffness"] is inputs["old_stiffness"]


def test_independent_foundation_uses_its_actual_saved_radial_displacement():
    inputs, previous = _physical_inputs(hot=.012345678901234567)
    normalized, evidence = normalize_native_birth_radial_before(inputs, previous)
    foundation_entry = evidence.coordinates[-1]
    assert foundation_entry.physical_before_m == F.from_float(.012345678901234567)
    assert foundation_entry.coordinate_decimal == normalized["coordinates_before"][-1]
    assert normalized["born_foundations"][0].previous_state is inputs["born_foundations"][0].previous_state


def test_canonical_adapter_is_idempotent():
    inputs, previous = _physical_inputs()
    first, evidence = normalize_native_birth_radial_before(inputs, previous)
    second, repeated = normalize_native_birth_radial_before(first, previous)
    assert second == first
    assert repeated == evidence


def test_existing_canonical_mechanical_history_is_not_rebased_from_a_rounded_bank():
    inputs, previous = _physical_inputs()
    before = list(inputs["coordinates_before"])
    before[6] = "-5.734914863803788E-7"
    inputs["coordinates_before"] = tuple(before)
    with pytest.raises(ValueError, match="canonical mechanical history"):
        normalize_native_birth_radial_before(inputs, previous)


def test_initial_coordinate_cannot_be_attached_to_a_different_physical_bank():
    inputs, previous = _physical_inputs()
    before = list(inputs["coordinates_before"])
    before[6] += 1e-10
    inputs["coordinates_before"] = tuple(before)
    with pytest.raises(ValueError, match="saved physical bank"):
        normalize_native_birth_radial_before(inputs, previous)


def test_bank_trace_order_is_authenticated_before_normalizing():
    inputs, previous = _physical_inputs()
    previous.transverse_displacement_m = (("wrong-trace", -.05734914863803788, .04870854357657349),)
    with pytest.raises(ValueError, match="canonical immutable order"):
        normalize_native_birth_radial_before(inputs, previous)


def test_missing_physical_bank_is_refused():
    inputs, previous = _physical_inputs()
    previous.initial_mechanics.fault_traces.traces = ()
    previous.transverse_displacement_m = ()
    with pytest.raises(ValueError, match="unchanged physical trace bank"):
        normalize_native_birth_radial_before(inputs, previous)


def test_foundation_coordinates_must_be_registered_exactly_once():
    inputs, previous = _physical_inputs()
    inputs["born_foundations"][0].radial_dof_index = 7
    with pytest.raises(ValueError, match="distinct extra radial"):
        normalize_native_birth_radial_before(inputs, previous)


def test_explicit_precision_changes_only_precision_and_radial_initial_coordinates():
    inputs, previous = _physical_inputs(radius=3.)
    normalized, evidence = normalize_native_birth_radial_before(inputs, previous, decimal_precision=120)
    assert normalized["decimal_precision"] == evidence.decimal_precision == 120
    assert inputs["decimal_precision"] == 160
    assert evidence.maximum_physical_quotient_error_bound_m < F(1, 10**118)


def test_source_bound_wrapper_preserves_the_original_frames_and_source_authentication(monkeypatch):
    inputs, previous = _physical_inputs()
    frames = (object(),)
    sources = (object(), previous, object(), object(), 1e6)
    seen = []
    def builder(*args, **kwargs):
        seen.append((args, kwargs))
        return inputs, frames
    monkeypatch.setattr("tectonics.geometric_native_birth_continuum_motion.build_native_birth_continuum_inputs", builder)
    normalized, actual_frames, evidence = build_native_birth_exact_radial_inputs(*sources)
    assert seen == [(sources, {"decimal_precision": 160})]
    assert actual_frames is frames
    assert normalized["old_states"] is inputs["old_states"]
    assert len(evidence.coordinates) == 3


def test_evidence_roundtrips_exact_rationals_through_json():
    inputs, previous = _physical_inputs()
    _, evidence = normalize_native_birth_radial_before(inputs, previous)
    payload = json.loads(json.dumps(radial_before_normalization_to_dict(evidence), allow_nan=False))
    assert payload["material_admission_authorized"] is False
    assert F(*payload["radius_m"]) == evidence.radius_m
    for entry, saved in zip(evidence.coordinates, payload["coordinates"]):
        assert F(*saved["physical_before_m"]) == entry.physical_before_m
        assert F(*saved["physical_quotient_error_bound_m"]) == entry.physical_quotient_error_bound_m
        assert F(*saved["physical_reconstruction_error_m"]) == entry.physical_reconstruction_error_m


def test_existing_low_level_network_consumes_canonical_decimal_foundation_coordinate():
    from tests.test_geometric_native_birth_contact_network import _case
    from tectonics.geometric_native_birth_contact_network import solve_birth_contact_network
    inputs = _case()
    previous = NS(initial_mechanics=NS(fault_traces=NS(traces=())), transverse_displacement_m=())
    expected = solve_birth_contact_network(**inputs)
    normalized, evidence = normalize_native_birth_radial_before(inputs, previous)
    result = solve_birth_contact_network(**normalized)
    assert evidence.coordinates[0].coordinate_decimal == "0"
    assert result.coordinates_decimal == expected.coordinates_decimal
    assert result.original_point_states == expected.original_point_states
    assert result.born_contact_steps == expected.born_contact_steps
    assert result.born_foundation_steps == expected.born_foundation_steps


@pytest.mark.parametrize("physical,radius", [(float("nan"), 1.), (0., 0.), (0., -1.), (1, 3.),
    (1., True), (1e308, 1e-308)])
def test_nonfinite_nonliteral_or_unrepresentable_inputs_are_refused(physical, radius):
    with pytest.raises(ValueError):
        exact_radial_before_coordinate(physical, radius)


@pytest.mark.parametrize("precision", [True, 99, 160., "160"])
def test_precision_must_be_explicitly_supported(precision):
    with pytest.raises(ValueError, match="100 declared Decimal digits"):
        exact_radial_before_coordinate(1., 3., decimal_precision=precision)
