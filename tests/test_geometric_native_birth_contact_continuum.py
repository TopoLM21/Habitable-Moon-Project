"""Full-arc mechanics against independent exact moments and force balances."""
from dataclasses import replace
from decimal import Decimal, localcontext
import json

import pytest

from tests.test_geometric_native_birth_contact_network import _case
from tectonics.geometric_native_birth_contact_continuum import (
    BornContinuumAttachment, BornContinuumDeclaration, BornContinuumIncrement,
    _advance_field, _response,
    activate_born_continuum_contact, birth_continuum_network_to_dict,
    continuum_field_extrema, solve_born_continuum_network,
    validate_born_continuum_contact_state,
)


def _continuum_case(*, gap=-1., area=1., basal=1., varying=False):
    inputs = _case(gap=gap, interface_area=area, basal_area=basal)
    point = inputs["born_attachments"][0]
    zero = ("0",)*7
    normal = (tuple(str(Decimal.from_float(value)) for value in point.normal_operator), zero, zero)
    if varying:
        normal = (normal[0], zero, ("1", "0", "0", "-1", "0", "0", "0"))
    declaration = BornContinuumDeclaration(point.parameters, point.reference,
        "0", "1", str(Decimal.from_float(area)), normal, (zero,)*3,
        "Full unit native reference arc with independently declared continuum closure")
    previous = activate_born_continuum_contact(declaration)
    attachment = BornContinuumAttachment(declaration, previous)
    return {**inputs, "born_attachments": (), "born_continuum_attachments": (attachment,),
        "decimal_precision": 160}


def test_constant_whole_arc_matches_exact_series_network_and_preserves_original_history():
    result = solve_born_continuum_network(**_continuum_case())
    assert result.coordinates[0] == pytest.approx(5./16.)
    assert result.coordinates[6] == pytest.approx(-1./8.)
    assert result.original_point_states[0][1].step_count == 2
    local = result.continuum_contact_steps[0]
    assert local.closed_angle_parts_decimal == (("0", "1"),)
    assert Decimal(local.state.activation_energy_decimal) == Decimal(".25")
    assert max(result.per_coordinate_relative_residual) < 1e-100
    assert dict(result.diagnostics)["born_midpoint_atoms_used"] is False


def test_open_whole_arc_has_zero_pressure_without_weakening_old_cold_gate():
    result = solve_born_continuum_network(**_continuum_case(gap=1.))
    local = result.continuum_contact_steps[0]
    assert local.closed_angle_parts_decimal == ()
    assert Decimal(local.state.stored_energy_decimal) == 0
    assert all(Decimal(value) == 0 for value in local.gradient_decimal)
    assert result.coordinates[0] == pytest.approx(.25)


def test_sine_gap_integral_includes_real_zero_and_full_closed_subarc():
    inputs = _continuum_case()
    declaration = inputs["born_continuum_attachments"][0].declaration
    # g(theta)=-1/2+sin(theta), root pi/6 within [0,1].
    with localcontext() as context:
        context.prec = 180
        energy, gradient, hessian, parts, bands, uncertainty = _response(
            declaration, (Decimal("-.5"), Decimal(0), Decimal(1)), 160)
        root = Decimal("0.5235987755982988730771072305465838140328615665625176368291574320513027343810348331046724708903528")
        assert len(parts) == 1
        assert abs(parts[0][1]-root) < Decimal("1e-78")
        # Integral (-1/2+sin theta)^2 from 0 to pi/6 is
        # 3*pi/24 - 1 + 3*sqrt(3)/8; keff=1/2, potential prefactor1/4.
        expected = (Decimal(3)*root/4-1+Decimal(3)*Decimal(3).sqrt()/8)/4
        assert abs(energy-expected) < Decimal("1e-78")
        assert energy > 0
        assert len(bands) == 1
        assert bands[0][0] <= root <= bands[0][1]


def test_tiny_positive_closing_subarc_is_integrated_without_a_minimum_area():
    inputs = _continuum_case(area=1e-12, basal=1e-36)
    declaration = inputs["born_continuum_attachments"][0].declaration
    with localcontext() as context:
        context.prec = 180
        epsilon = Decimal("1e-30")
        # sin theta grows from zero: only an arc of width approximately
        # epsilon is closed; midpoint selection would omit the whole force.
        energy, gradient, _, parts, bands, _ = _response(
            declaration, (-epsilon, Decimal(0), Decimal(1)), 160)
        assert energy > 0
        assert gradient[6] > 0
        assert Decimal(".999999999999999999999")*epsilon < parts[0][1] < Decimal("1.000000000000000000001")*epsilon
        expected = Decimal.from_float(1e-12)*epsilon**3/12
        assert abs(energy-expected) < expected*Decimal("1e-45")


def test_actual_varying_arc_newton_solution_changes_the_equilibrium():
    constant = solve_born_continuum_network(**_continuum_case())
    varying = solve_born_continuum_network(**_continuum_case(varying=True))
    assert varying.coordinates_decimal != constant.coordinates_decimal
    assert max(varying.per_coordinate_relative_residual) < 1e-35
    assert max(varying.actual_per_coordinate_relative_residual) < 1e-10
    local = varying.continuum_contact_steps[0]
    assert Decimal(local.numerical_dissipation_decimal) >= 0
    with localcontext() as context:
        context.prec = 160
        balance = Decimal(local.external_work_decimal)-Decimal(local.stored_energy_increment_decimal)-Decimal(local.numerical_dissipation_decimal)
        assert abs(balance) < Decimal("1e-140")


def test_full_field_state_replay_and_strict_export_keep_every_coordinate_and_ledger():
    inputs = _continuum_case(varying=True)
    result = solve_born_continuum_network(**inputs)
    local = result.continuum_contact_steps[0]
    declaration = inputs["born_continuum_attachments"][0].declaration
    validate_born_continuum_contact_state(local.state, expected_declaration=declaration)
    with pytest.raises(ValueError, match="field/work replay"):
        validate_born_continuum_contact_state(replace(local.state, cumulative_external_work_decimal="0"),
            expected_declaration=declaration)
    packet = birth_continuum_network_to_dict(result, **inputs)
    assert json.loads(json.dumps(packet, sort_keys=True)) == packet
    assert packet["result"]["diagnostics"]["whole_temporal_interval_admitted"] is False


def test_whole_arc_extrema_find_internal_compressive_pocket():
    # g=-1/4-cos(theta-1) has an interior minimum on [0,2].
    from tectonics.geometric_native_birth_contact_continuum import _sincos
    with localcontext() as context:
        context.prec = 160
        sine, cosine = _sincos(Decimal(1))
        extremes = continuum_field_extrema(("-0.25", str(-cosine), str(-sine)), "0", "2")
        assert abs(Decimal(extremes[0])+Decimal("1.25")) < Decimal("1e-75")


def test_old_born_midpoint_weights_are_explicitly_refused():
    inputs = _continuum_case()
    with pytest.raises(ValueError, match="midpoint"):
        solve_born_continuum_network(**{**inputs, "born_attachments": _case()["born_attachments"]})


@pytest.mark.parametrize("coefficients", [
    ("0", "0"), ("0", "0", "0", "123"), ["0", "0", "0"],
    ("+0", "0", "0"), ("NaN", "0", "0"), (0., "0", "0"),
])
def test_increment_refuses_incomplete_extra_mutable_or_noncanonical_field(coefficients):
    with pytest.raises(ValueError, match="coefficients|canonical"):
        BornContinuumIncrement(coefficients, ("0", "0", "0"), 1.)


@pytest.mark.parametrize("duration", [True, 1, 0., -1., float("inf"), float("nan")])
def test_increment_requires_its_exact_finite_positive_float_clock(duration):
    with pytest.raises(ValueError, match="float duration"):
        BornContinuumIncrement(("0", "0", "0"), ("0", "0", "0"), duration)


def test_replay_cannot_retain_a_fourth_uncomputed_coefficient_or_mutable_history():
    previous = _continuum_case()["born_continuum_attachments"][0].previous_state
    with localcontext() as context:
        context.prec = 160
        with pytest.raises(ValueError, match="exactly three"):
            _advance_field(previous, (Decimal("-.5"), Decimal(0), Decimal(0), Decimal(123)),
                (Decimal(0),)*3, 1., 160)
    with pytest.raises(ValueError, match="immutable typed increments"):
        replace(previous, history=[])
    with pytest.raises(ValueError, match="step count"):
        replace(previous, step_count=False)


@pytest.mark.parametrize("precision", [True, 160., 0, 99, float("inf")])
def test_state_and_activation_require_exact_positive_integer_precision(precision):
    previous = _continuum_case()["born_continuum_attachments"][0].previous_state
    with pytest.raises(ValueError, match="integer precision"):
        replace(previous, decimal_precision=precision)
    with pytest.raises(ValueError, match="integer precision"):
        activate_born_continuum_contact(previous.declaration, decimal_precision=precision)
