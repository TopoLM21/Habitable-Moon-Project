"""Time brackets use fresh physical-kernel trials with unchanged accepted history."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from tectonics.genesis_crack_path import CrackInterval
from tectonics.genesis_path_birth import recover_tied_tractions, classify_tied_onset
from tectonics.genesis_path_dynamics import PathLoading
from tectonics.genesis_path_onset_event import locate_tied_onset
from test_genesis_path_dynamics import _model, _material, _same


def _ramp(initial_score=.2, slope=.1):
    """Uniform prescribed stress predictor, with balancing parent loads.

    Relative-bank reactions remain constrained and nonzero. The exact source
    traction criterion scales linearly with the prescribed predictor; these
    are real PathMechanics.trial solves, not mocked event samples.
    """
    model = _model()
    volume, elasticity = _material(model)
    n = len(volume)
    base_memory = np.tile([1e-5, 1e-5, 0.], (n, 1))
    water = np.zeros(len(model.trace_depth_m))

    def trial(old, dt, factor):
        memory = base_memory*factor
        _, force = model.basis.bulk(volume, elasticity, memory)
        force[model.basis.nparent:] = 0.
        loading = PathLoading(dt, volume, elasticity, memory, np.ones(n), force, np.zeros(n))
        state = model.trial(old, loading)
        stress = np.einsum("fij,fj->fi", elasticity, state.elastic_strain)
        return state, stress, water.copy()

    initial = model.initial(np.zeros((n, 3)))
    calibration, stress, _ = trial(initial, 1., 1.)
    peak = classify_tied_onset(recover_tied_tractions(model, calibration, stress), water).maximum_observed_ratio
    start, start_stress, water = trial(initial, 1., initial_score/peak)
    calls = []
    def callback(dt):
        calls.append(dt)
        return trial(start, dt, (initial_score+slope*dt)/peak)
    return model, start, start_stress, water, callback, calls


def _locate(data, upper=20., tolerance=1e-5, **kwargs):
    model, start, stress, water, callback, _ = data
    return locate_tied_onset(model, start, stress, water, callback, upper,
        time_tolerance_years=tolerance, **kwargs)


def test_physical_kernel_ramp_brackets_known_crossing_and_preserves_history():
    data = _ramp()
    model, start, stress, water, callback, calls = data
    before = deepcopy(start)
    report = _locate(data)
    assert report.status == "bracketed"
    assert report.lower.duration_years <= 8. <= report.upper.duration_years
    assert report.time_width_years <= 1e-5
    assert report.lower.onset.maximum_observed_ratio < 1 <= report.upper.onset.maximum_observed_ratio
    assert report.upper_within_strength_tolerance
    assert len(report.upper.governing_trace_indices) > 0
    assert report.callback_count == len(calls) <= 23
    assert calls[0] == 20.
    assert len(set(calls)) == len(calls)
    for sample in (report.lower, report.upper):
        assert sample.state.accepted_steps == start.accepted_steps+1
        assert sample.state.elapsed_years == start.elapsed_years+sample.duration_years
        assert sample.state.active_interval is None
        assert not sample.onset.physical_birth_ready
        assert not sample.governing_trace_indices.flags.writeable
        np.testing.assert_array_equal(sample.state.cohorts.fracture_work_j, 0.)
    _same(before, start)
    # Prospective snapshots do not alias the accepted state.
    report.upper.state.elastic_strain[0, 0] = 100.
    _same(before, start)


def test_lower_upper_refinement_order_does_not_carry_trial_history_forward():
    data = _ramp()
    one = _locate(data, tolerance=1e-4)
    two = _locate(data, tolerance=1e-4)
    _same(one.lower.state, two.lower.state)
    _same(one.upper.state, two.upper.state)
    assert one.callback_count == two.callback_count


def test_unloading_upper_below_strength_does_not_claim_no_earlier_event():
    data = _ramp(initial_score=.8, slope=-.02)
    report = _locate(data, upper=20.)
    assert report.status == "no_bracket_at_upper"
    assert report.callback_count == 1
    assert report.upper.onset.maximum_observed_ratio == pytest.approx(.4)
    assert report.lower.onset.maximum_observed_ratio == pytest.approx(.8)


def test_already_exceeded_source_refuses_without_trial_or_strength_reset():
    data = _ramp(initial_score=1.3)
    before = deepcopy(data[1])
    original_strength = data[0].law_parameters.tensile_strength_pa
    with pytest.raises(ValueError, match="already exceeds strength"):
        _locate(data)
    assert not data[-1]
    _same(before, data[1])
    assert data[0].law_parameters.tensile_strength_pa == original_strength


def test_initial_strength_surface_is_reported_without_callback():
    data = _ramp(initial_score=1.)
    report = _locate(data)
    assert report.status == "start_on_strength_surface"
    assert report.lower.duration_years == report.upper.duration_years == 0.
    assert report.callback_count == 0
    assert not data[-1]


def test_time_resolution_does_not_hide_unresolved_strength_overshoot():
    data = _ramp(initial_score=.2, slope=1e8)
    report = _locate(data, upper=1e-5, tolerance=1e-6)
    assert report.status == "bracketed"
    assert report.time_width_years <= 1e-6
    assert not report.upper_within_strength_tolerance
    assert report.upper_strength_overshoot > 1.
    assert report.upper.state.active_interval is None


@pytest.mark.parametrize("corruption", ["clock", "duration", "steps", "retries", "released", "return", "state"])
def test_rejects_callbacks_violating_fresh_trial_contract(corruption):
    data = _ramp()
    model, start, stress, water, callback, _ = data
    before = deepcopy(start)
    def wrong(dt):
        result, local_stress, local_water = callback(dt)
        if corruption == "clock":
            result = replace(result, elapsed_years=result.elapsed_years+1.)
        elif corruption == "duration":
            result = replace(result, last_step_years=dt/2)
        elif corruption == "steps":
            result = replace(result, accepted_steps=result.accepted_steps+1)
        elif corruption == "retries":
            result = replace(result, rejected_steps=result.rejected_steps+1)
        elif corruption == "released":
            result = model.release(result, CrackInterval(0., model.basis.insertion.path.length_m))
        elif corruption == "return":
            return result
        elif corruption == "state":
            result = None
        return result, local_stress, local_water
    with pytest.raises(ValueError):
        locate_tied_onset(model, start, stress, water, wrong, 20., time_tolerance_years=1e-4)
    _same(before, start)


def test_callback_failure_propagates_without_advancing_accepted_state():
    data = _ramp()
    model, start, stress, water, _, _ = data
    before = deepcopy(start)
    def fail(dt):
        raise RuntimeError("controlled predictor failure")
    with pytest.raises(RuntimeError, match="controlled predictor failure"):
        locate_tied_onset(model, start, stress, water, fail, 20., time_tolerance_years=1e-4)
    _same(before, start)


def test_mutating_callback_is_detected_even_when_it_raises():
    data = _ramp()
    model, start, stress, water, _, _ = data
    def bad(dt):
        start.elastic_strain[0, 0] += 1.
        raise RuntimeError("callback failure")
    with pytest.raises(ValueError, match="mutated the accepted source"):
        locate_tied_onset(model, start, stress, water, bad, 20., time_tolerance_years=1e-4)


def test_callback_cannot_reset_strength_to_make_the_trial_admissible():
    data = _ramp()
    model, start, stress, water, callback, _ = data
    def bad(dt):
        result = callback(dt)
        model.law_parameters = replace(model.law_parameters, tensile_strength_pa=4e6)
        return result
    with pytest.raises(ValueError, match="model parameters"):
        locate_tied_onset(model, start, stress, water, bad, 20., time_tolerance_years=1e-4)


def test_iteration_budget_is_bounded_without_promoting_overshot_endpoint():
    data = _ramp()
    before = deepcopy(data[1])
    with pytest.raises(RuntimeError, match="bounded iteration count"):
        _locate(data, tolerance=1e-12, max_iterations=2)
    assert len(data[-1]) == 3
    _same(before, data[1])


def test_unrepresentable_physical_clock_rejects_before_callback():
    data = list(_ramp())
    data[1] = replace(data[1], elapsed_years=1e16)
    with pytest.raises(ValueError, match="representable clock progress"):
        _locate(data, upper=.1)
    assert not data[-1]


@pytest.mark.parametrize("argument,value", [("upper_duration_years", 0.),
    ("upper_duration_years", np.inf), ("upper_duration_years", True),
    ("time_tolerance_years", 0.), ("time_tolerance_years", np.nan),
    ("strength_tolerance", -.1), ("strength_tolerance", 1.),
    ("strength_tolerance", True), ("max_iterations", 0), ("max_iterations", 2.5)])
def test_invalid_locator_controls_fail_before_callback(argument, value):
    model, start, stress, water, callback, calls = _ramp()
    kwargs = dict(upper_duration_years=20., time_tolerance_years=1e-5)
    kwargs[argument] = value
    with pytest.raises(ValueError):
        locate_tied_onset(model, start, stress, water, callback, **kwargs)
    assert not calls
