"""Actual primary-material birth comes from phase evolution, never lid depth."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import pytest

from tectonics.genesis import GenesisParameters, melt_fraction
from tectonics import genesis_solidification as solidification
from tectonics.genesis_solidification import (
    SolidificationHistory, advance_solidification_history,
    initial_solidification_history, solidification_history_from_dict,
    solidification_history_to_dict, validate_solidification_history,
)


def sample(thermal, time, temperature):
    return {"time_myr": time, "surface_temperature_k": temperature,
        "surface_melt_fraction": melt_fraction(temperature, thermal)}


def advance(history, thermal, time, temperature):
    return advance_solidification_history(history,
        sample(thermal, history.time_myr, history.surface_temperature_k),
        sample(thermal, time, temperature), thermal)


def reseal(raw):
    body = {key: value for key, value in raw.items() if key != "sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    raw["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    return raw


def test_molten_origin_has_no_solid_cohorts_or_invented_age():
    p = GenesisParameters()
    history = initial_solidification_history(p)
    assert history.time_myr == 0
    assert history.cohorts == history.intervals == ()
    assert history.solid_mass_kg == history.formed_mass_kg == history.remelted_mass_kg == 0
    assert history.mean_birth_time_myr is None
    assert history.mean_age_myr == history.birth_time_mass_moment_kg_myr == 0
    assert history.surface_reservoir_mass_kg == p.area_m2*p.surface_column_kg_m2
    still_molten = advance(history, p, 2., 2100.)
    assert still_molten.accepted_intervals == 1
    assert still_molten.cohorts == ()
    assert history.accepted_intervals == 0


def test_full_phase_crossing_birth_time_matches_independent_clipped_integral():
    p = GenesisParameters()
    history = advance(initial_solidification_history(p), p, 12., 1100.)
    # T(t)=2300-100t. Rock appears at t=3 (T=2000), and reaches
    # solid fraction one at t=9 (T=1400), at constant mass rate M/6.
    mass = p.area_m2*p.surface_column_kg_m2
    moment = mass/6*(9**2-3**2)/2
    assert history.solid_mass_kg == pytest.approx(mass, rel=2e-15)
    assert history.birth_time_mass_moment_kg_myr == pytest.approx(moment, rel=2e-15)
    assert history.mean_birth_time_myr == 6.
    assert history.mean_age_myr == 6.
    assert history.cohorts[0].birth_start_myr == 3.
    assert history.cohorts[0].birth_end_myr == 9.


def test_partial_birth_is_not_entire_solver_interval_midpoint():
    p = GenesisParameters()
    history = advance(initial_solidification_history(p), p, 12., 1700.)
    # T(t)=2300-50t; the first half of the phase interval freezes at 6..12.
    assert history.solid_mass_kg == pytest.approx(history.surface_reservoir_mass_kg/2)
    assert history.mean_birth_time_myr == pytest.approx(9., abs=2e-14)
    assert history.mean_age_myr == pytest.approx(3., abs=2e-14)


def test_proportional_remelting_preserves_old_birth_and_new_freezing_adds_new_birth():
    p = GenesisParameters()
    frozen = advance(initial_solidification_history(p), p, 12., 1100.)
    half = advance(frozen, p, 18., 1700.)
    mass = frozen.surface_reservoir_mass_kg
    assert half.solid_mass_kg == mass/2
    assert half.formed_mass_kg == mass
    assert half.remelted_mass_kg == mass/2
    assert half.birth_time_mass_moment_kg_myr == frozen.birth_time_mass_moment_kg_myr/2
    assert half.mean_birth_time_myr == 6.
    refrozen = advance(half, p, 30., 1100.)
    assert len(refrozen.cohorts) == 2
    assert refrozen.cohorts[1].birth_start_myr == 18.
    assert refrozen.cohorts[1].birth_end_myr == 24.
    assert refrozen.formed_mass_kg == pytest.approx(1.5*mass)
    assert refrozen.remelted_mass_kg == mass/2
    assert refrozen.mean_birth_time_myr == pytest.approx(13.5, abs=2e-14)
    molten = advance(refrozen, p, 42., 2300.)
    assert molten.solid_mass_kg == 0
    assert molten.mean_birth_time_myr is None
    assert molten.formed_mass_kg == molten.remelted_mass_kg
    assert len(molten.cohorts) == 2
    reborn = advance(molten, p, 54., 1100.)
    assert reborn.mean_birth_time_myr == pytest.approx(48., abs=2e-14)
    assert [c.retained_mass_kg for c in reborn.cohorts[:2]] == [0., 0.]


def test_linear_thermal_step_refinement_preserves_mass_and_birth_moment():
    p = GenesisParameters()
    coarse = advance(initial_solidification_history(p), p, 12., 1100.)
    fine = initial_solidification_history(p)
    for time in (1., 2., 3., 4.3, 6., 7.7, 9., 10.9, 12.):
        fine = advance(fine, p, time, 2300.-100*time)
    assert len(fine.cohorts) > len(coarse.cohorts)
    assert fine.solid_mass_kg == pytest.approx(coarse.solid_mass_kg, rel=2e-15)
    assert fine.birth_time_mass_moment_kg_myr == pytest.approx(
        coarse.birth_time_mass_moment_kg_myr, rel=2e-15)
    assert fine.mean_age_myr == pytest.approx(coarse.mean_age_myr, abs=2e-14)


def test_restart_preserves_exact_accepted_records_and_continuation():
    p = GenesisParameters()
    history = advance(initial_solidification_history(p), p, 6., 1700.)
    saved = solidification_history_to_dict(history)
    restored = solidification_history_from_dict(json.loads(json.dumps(saved)), p)
    assert restored == history
    assert advance(restored, p, 10., 1300.) == advance(history, p, 10., 1300.)
    saved["cohorts"][0]["formed_mass_kg"] = 0.
    assert restored.cohorts[0].formed_mass_kg > 0
    with pytest.raises(FrozenInstanceError):
        restored.time_myr = 0
    with pytest.raises(FrozenInstanceError):
        restored.cohorts[0].retained_mass_kg = 0


def test_constructor_owns_caller_lists():
    p = GenesisParameters()
    original = advance(initial_solidification_history(p), p, 6., 1700.)
    cohorts, intervals = list(original.cohorts), list(original.intervals)
    history = SolidificationHistory(original.time_myr, original.surface_temperature_k,
        original.surface_reservoir_mass_kg, original.parameters_sha256, cohorts, intervals)
    cohorts.clear(); intervals.clear()
    assert history == original


def test_immutable_trial_reuse_avoids_replaying_birth_and_restart_still_audits(monkeypatch):
    p = GenesisParameters()
    history = advance(initial_solidification_history(p), p, 12., 1100.)
    assert deepcopy(history) is history
    birth = solidification._birth_interval
    audited_crossings = []

    def count_crossing(interval, thermal):
        audited_crossings.append(interval)
        return birth(interval, thermal)

    monkeypatch.setattr(solidification, "_birth_interval", count_crossing)
    for time in range(13, 113):
        history = advance(deepcopy(history), p, float(time), 1100.)
    assert not audited_crossings
    restored = solidification_history_from_dict(solidification_history_to_dict(history), p)
    assert restored == history
    assert len(audited_crossings) == 1


def test_new_constructed_history_cannot_inherit_incremental_trust():
    p = GenesisParameters()
    history = advance(initial_solidification_history(p), p, 12., 1100.)
    bad = replace(history, cohorts=(replace(history.cohorts[0], birth_start_myr=1.),))
    with pytest.raises(ValueError, match="birth or remelting history changed"):
        validate_solidification_history(bad, p)


@pytest.mark.parametrize("time,temperature", [(0., 1700.), (-1., 1700.), (float("nan"), 1700.),
    (float("inf"), 1700.), (6., float("nan")), (6., float("inf")), (6., -1.)])
def test_nonfinite_or_nonadvancing_samples_are_rejected(time, temperature):
    p = GenesisParameters()
    with pytest.raises(ValueError):
        advance(initial_solidification_history(p), p, time, temperature)


def test_clock_gaps_changed_endpoint_phase_and_parameter_rebinding_are_rejected():
    p = GenesisParameters()
    history = advance(initial_solidification_history(p), p, 6., 1700.)
    with pytest.raises(ValueError, match="disagree"):
        advance_solidification_history(history, sample(p, 7., 1700.), sample(p, 8., 1600.), p)
    with pytest.raises(ValueError, match="disagree"):
        advance_solidification_history(history, sample(p, 6., 1701.), sample(p, 8., 1600.), p)
    wrong_phase = sample(p, 6., 1700.); wrong_phase["surface_melt_fraction"] = .8
    with pytest.raises(ValueError, match="phase law"):
        advance_solidification_history(history, wrong_phase, sample(p, 8., 1600.), p)
    with pytest.raises(ValueError, match="parameters changed"):
        validate_solidification_history(history, replace(p, thermal_diffusivity_m2_s=2e-6))


@pytest.mark.parametrize("mutation", [
    lambda raw: raw["cohorts"][0].update(birth_start_myr=1.),
    lambda raw: raw["cohorts"][0].update(formed_mass_kg=raw["cohorts"][0]["formed_mass_kg"]*.8),
    lambda raw: raw["intervals"][0].update(before_time_myr=.1),
    lambda raw: raw.update(cohorts=[]),
    lambda raw: raw.update(intervals=[]),
    lambda raw: raw.update(surface_reservoir_mass_kg=1.),
])
def test_resealed_but_inconsistent_thermal_birth_history_is_rejected(mutation):
    p = GenesisParameters()
    raw = solidification_history_to_dict(advance(initial_solidification_history(p), p, 12., 1100.))
    mutation(raw)
    with pytest.raises(ValueError):
        solidification_history_from_dict(reseal(raw), p)


@pytest.mark.parametrize("raw", [None, {}, {"format": "older-solidification"}])
def test_missing_old_history_is_not_synthesized_from_current_solid_mass(raw):
    with pytest.raises(ValueError, match="recalculate from molten birth"):
        solidification_history_from_dict(raw, GenesisParameters())
