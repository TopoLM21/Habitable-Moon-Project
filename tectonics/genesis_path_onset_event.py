"""Rollback-safe localization of a sampled, tied-path strength crossing.

The caller owns physical loading and supplies pure trials from ONE accepted
state. The locator only brackets the reconstructed traction criterion. It
does not commit a candidate, insert a cohesive surface, or infer the globally
earliest event under nonmonotone loading. Both bracket endpoints remain
prospective states; the upper endpoint may still exceed the strength.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
import hashlib
from numbers import Integral, Real

import numpy as np

from .genesis_path_birth import (TiedTractionRecovery, TiedOnsetDiagnostic,
    recover_tied_tractions, classify_tied_onset)
from .genesis_path_dynamics import PathState


def _digest(value):
    digest = hashlib.sha256()
    def add(item):
        if is_dataclass(item):
            for field in fields(item):
                digest.update(field.name.encode())
                add(getattr(item, field.name))
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype.str, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, (tuple, list)):
            for child in item:
                add(child)
        else:
            digest.update(repr(item).encode())
    add(value)
    return digest.digest()


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class TiedOnsetSample:
    duration_years: float
    state: PathState
    recovery: TiedTractionRecovery
    onset: TiedOnsetDiagnostic
    governing_trace_indices: np.ndarray


@dataclass(frozen=True)
class TiedOnsetBracket:
    status: str
    lower: TiedOnsetSample
    upper: TiedOnsetSample
    callback_count: int
    time_tolerance_years: float
    strength_tolerance: float

    @property
    def time_width_years(self):
        return self.upper.duration_years-self.lower.duration_years

    @property
    def upper_strength_overshoot(self):
        return max(self.upper.onset.maximum_observed_ratio-1., 0.)

    @property
    def upper_within_strength_tolerance(self):
        return self.upper_strength_overshoot <= self.strength_tolerance


def locate_tied_onset(model, start_state, start_stress_pa, start_water_per_trace,
                     trial_at_duration, upper_duration_years, *,
                     time_tolerance_years, strength_tolerance=1e-6, max_iterations=64):
    """Bracket a local-strength crossing without changing accepted history.

    ``trial_at_duration(dt)`` must return ``(state, stress_pa, water_per_trace)``
    from one fresh ``model.trial(start_state, loading_for_dt)``. Every call uses
    that same starting state, never the preceding prospective endpoint. The
    callback must own no external side effects. The locator checks source
    history immutability, exact clocks, accepted/rejected counts, and all-held
    constraints; callback exceptions propagate without committing a result.

    Start states exceeding strength by more than ``strength_tolerance`` are
    refused. Starts within that tolerance of strength are reported at time
    zero. Otherwise an upper endpoint below strength produces
    ``no_bracket_at_upper``; this is NOT proof of no earlier crossing. A
    bracket with lower score < 1 <= upper score is bisected until its duration
    width is <= ``time_tolerance_years``. Strength tolerance never caps or
    rescales traction: inspect ``upper_within_strength_tolerance`` separately,
    especially for discontinuous forcing or steep changes. No endpoint is
    automatically accepted for physical activation.

    Governing traces have score within ``strength_tolerance`` of the maximum
    observed score. Stored state snapshots are independent copies, not solver
    checkpoints with an owned thermal/orbital history.
    """
    upper_duration = _positive(upper_duration_years, "upper_duration_years")
    time_tolerance = _positive(time_tolerance_years, "time_tolerance_years")
    if (isinstance(strength_tolerance, (bool, np.bool_))
            or not isinstance(strength_tolerance, Real)
            or not np.isfinite(strength_tolerance) or not 0 <= strength_tolerance < 1):
        raise ValueError("strength_tolerance must be finite in [0, 1)")
    if (isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, Integral)
            or max_iterations < 1):
        raise ValueError("max_iterations must be a positive integer")
    if not callable(trial_at_duration):
        raise ValueError("trial_at_duration must be callable")

    def sample(state, stress, water, duration):
        recovery = recover_tied_tractions(model, state, stress)
        onset = classify_tied_onset(recovery, water, model.law_parameters,
                                   relative_tolerance=strength_tolerance)
        if not np.isfinite(onset.maximum_observed_ratio):
            raise ValueError("A finite strength ratio is required for event localization")
        score = np.maximum(onset.normal_ratio, onset.shear_ratio)
        indices = np.flatnonzero(recovery.trace_observed &
            (score >= onset.maximum_observed_ratio-strength_tolerance))
        indices = np.frombuffer(indices.tobytes(), dtype=indices.dtype)
        return TiedOnsetSample(float(duration), deepcopy(state), recovery, onset, indices)

    lower = sample(start_state, start_stress_pa, start_water_per_trace, 0.)
    def protected_values():
        return (start_state, start_stress_pa, start_water_per_trace,
                model.law_parameters, model.parameters, model.geometry_parameters)
    source_digest = _digest(protected_values())
    start_score = lower.onset.maximum_observed_ratio
    if start_score > 1+strength_tolerance:
        raise ValueError("Starting state already exceeds strength; an earlier physical bracket is required")
    if abs(start_score-1.) <= strength_tolerance:
        return TiedOnsetBracket("start_on_strength_surface", lower, lower, 0,
                               time_tolerance, float(strength_tolerance))
    if not np.isfinite(start_state.elapsed_years+upper_duration):
        raise ValueError("Requested event clock overflows")
    if start_state.elapsed_years+upper_duration == start_state.elapsed_years:
        raise ValueError("Requested event duration makes no representable clock progress")
    callback_count = 0

    def evaluate(duration):
        nonlocal callback_count
        expected_time = start_state.elapsed_years+duration
        if expected_time == start_state.elapsed_years:
            raise RuntimeError("Onset bracket reached physical-clock resolution")
        callback_count += 1
        try:
            values = trial_at_duration(float(duration))
        finally:
            if _digest(protected_values()) != source_digest:
                raise ValueError("Onset callback mutated the accepted source history, loading or model parameters")
        if not isinstance(values, (tuple, list)) or len(values) != 3:
            raise ValueError("Onset callback must return (PathState, stress_pa, water_per_trace)")
        state, stress, water = values
        if not isinstance(state, PathState):
            raise ValueError("Onset callback must return a PathState")
        if (state.elapsed_years != expected_time or state.last_step_years != duration
                or state.accepted_steps != start_state.accepted_steps+1
                or state.rejected_steps != start_state.rejected_steps):
            raise ValueError("Onset callback must make one fresh trial from the same accepted state and duration")
        return sample(state, stress, water, duration)

    upper = evaluate(upper_duration)
    if upper.onset.maximum_observed_ratio < 1.:
        return TiedOnsetBracket("no_bracket_at_upper", lower, upper, callback_count,
                               time_tolerance, float(strength_tolerance))
    iterations = 0
    while upper.duration_years-lower.duration_years > time_tolerance:
        if iterations >= max_iterations:
            raise RuntimeError("Onset localization exceeded its bounded iteration count")
        midpoint = lower.duration_years+(upper.duration_years-lower.duration_years)/2
        if midpoint in (lower.duration_years, upper.duration_years):
            raise RuntimeError("Onset bracket reached duration resolution")
        if start_state.elapsed_years+midpoint in (lower.state.elapsed_years, upper.state.elapsed_years):
            raise RuntimeError("Onset bracket reached physical-clock resolution")
        middle = evaluate(midpoint)
        if middle.onset.maximum_observed_ratio < 1.:
            lower = middle
        else:
            upper = middle
        iterations += 1
    return TiedOnsetBracket("bracketed", lower, upper, callback_count,
                           time_tolerance, float(strength_tolerance))
