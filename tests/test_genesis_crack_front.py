"""Energetic growth controls, irreversible history and exact experimental restart."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from tectonics.genesis_crack_energy import DCBOracle
from tectonics.genesis_crack_front import CrackFrontModel, FrontParameters
from tectonics.genesis_crack_path import CrackInterval, ReferenceCrackPath


def model(extension=.001, length=.15, **parameters):
    # Unit sphere is a convenient reference-coordinate carrier, not beam shape.
    path = ReferenceCrackPath(np.array([[1., 0., 0.], [np.cos(length), np.sin(length), 0.]]), .001)
    return CrackFrontModel(path, DCBOracle(fracture_energy_j_m2=5.), CrackInterval(0., .03),
                           FrontParameters(extension, **parameters))


def opening_at(oracle, length):
    return (16*oracle.fracture_energy_j_m2*length**4
            /(3*oracle.young_pa*oracle.arm_height_m**3))**.5


def ramp(front, count, peak_length=.09):
    state = front.initial()
    for opening in np.linspace(0., opening_at(front.oracle, peak_length), count+1)[1:]:
        state = front.advance(state, opening)
    return state


def test_unloaded_and_subcritical_seed_do_not_grow():
    front = model()
    initial = front.initial()
    for opening in (0., .95*opening_at(front.oracle, .03)):
        state = front.advance(initial, opening)
        assert state.interval == initial.interval
        assert state.events == ()
        assert state.fracture_work_j == state.unresolved_release_j == 0.
        assert state.external_work_j == state.stored_energy_j
        assert state.stop_reason == "energy_arrest"
    assert initial.load_step == 0


def test_every_increment_has_real_available_energy_and_stable_arrest():
    front = model()
    state = ramp(front, 80)
    assert abs(state.interval.length_m-.09) < front.parameters.extension_m
    assert state.stop_reason == "energy_arrest"
    previous = front.seed_interval.right_m
    for event in state.events:
        assert event.old_right_m == previous
        before = front.oracle.stored_energy_j(event.old_right_m, event.opening_m)
        after = front.oracle.stored_energy_j(event.new_right_m, event.opening_m)
        assert event.released_energy_j == before-after
        assert event.released_energy_j >= event.fracture_work_j > 0.
        assert event.unresolved_release_j == event.released_energy_j-event.fracture_work_j
        previous = event.new_right_m
    cost = front.oracle.fracture_energy_j_m2*front.oracle.width_m*(state.interval.length_m-.03)
    assert state.fracture_work_j == pytest.approx(cost, rel=1e-13)
    assert abs(front.energy_residual_j(state)) < 1e-14
    assert state.interval.left_m == 0.
    # The next adjacent segment genuinely fails; the stop is not an event budget.
    next_a = state.interval.length_m+front.parameters.extension_m
    release = state.stored_energy_j-front.oracle.stored_energy_j(next_a, state.opening_m)
    assert release < front.oracle.fracture_cost_j(state.interval.length_m, next_a)


def test_unloading_reloading_and_higher_peak_preserve_fracture_history():
    front = model()
    peak = ramp(front, 80)
    unloaded = front.advance(peak, 0.)
    assert unloaded.events == peak.events
    assert unloaded.interval == peak.interval
    assert unloaded.fracture_work_j == peak.fracture_work_j
    assert unloaded.stored_energy_j == 0.
    assert unloaded.external_work_j < peak.external_work_j
    assert unloaded.external_work_j == pytest.approx(peak.external_work_j-peak.stored_energy_j)
    reload = front.advance(unloaded, peak.opening_m)
    assert reload.events == peak.events
    assert reload.interval == peak.interval
    assert reload.external_work_j == pytest.approx(peak.external_work_j)
    higher = front.advance(reload, opening_at(front.oracle, .105))
    assert higher.interval.right_m > peak.interval.right_m
    assert higher.events[:len(peak.events)] == peak.events
    assert higher.fracture_work_j > peak.fracture_work_j


def test_refinement_reduces_unresolved_release_and_work_error():
    coarse_front, fine_front = model(.002), model(.0005)
    coarse, fine = ramp(coarse_front, 40), ramp(fine_front, 160)
    oracle = fine_front.oracle
    expected_work = (oracle.fracture_energy_j_m2*oracle.width_m*.03/3
                     +4*oracle.fracture_energy_j_m2*oracle.width_m*(.09-.03)/3)
    assert fine.unresolved_release_j < .4*coarse.unresolved_release_j
    assert abs(fine.external_work_j-expected_work) < .4*abs(coarse.external_work_j-expected_work)
    assert abs(fine.interval.length_m-.09) <= .0005
    # Spatial refinement alone cannot resolve energy from a large load jump.
    jumped = fine_front.advance(fine_front.initial(), opening_at(oracle, .09))
    assert jumped.unresolved_release_j > 10*fine.unresolved_release_j


def test_support_exhaustion_truncates_last_increment_and_reports_limit():
    front = model(.004, length=.055)
    state = front.advance(front.initial(), opening_at(front.oracle, .09))
    assert state.stop_reason == "support_exhausted"
    assert state.interval.right_m == front.path.length_m
    assert 0 < state.events[-1].new_right_m-state.events[-1].old_right_m < .004
    assert front.advance(state, state.opening_m).events == state.events
    assert abs(front.energy_residual_j(state)) < 1e-14


def test_extension_budget_failure_is_transactional():
    front = model(max_extensions_per_step=2)
    initial = front.initial()
    with pytest.raises(RuntimeError, match="budget"):
        front.advance(initial, opening_at(front.oracle, .09))
    assert initial == front.initial()


def test_exact_restart_including_unloading_and_energy_history(tmp_path):
    front = model(.0007)
    opening = opening_at(front.oracle, .09)
    loads = [*np.linspace(0., opening, 37), 0., opening*.6, opening, opening*1.2]
    direct = front.initial()
    for value in loads:
        direct = front.advance(direct, value)
    partial = front.initial()
    for value in loads[:23]:
        partial = front.advance(partial, value)
    file = tmp_path/"front.json"
    front.save_checkpoint(file, partial)
    resumed_front, resumed = CrackFrontModel.load_checkpoint(file)
    assert resumed == partial
    assert resumed_front.path.fingerprint == front.path.fingerprint
    for value in loads[23:]:
        resumed = resumed_front.advance(resumed, value)
    assert asdict(resumed) == asdict(direct)
    with pytest.raises(FileExistsError):
        front.save_checkpoint(file, direct)


@pytest.mark.parametrize("damage", ["geometry", "radius", "version", "cost", "event", "load", "unknown", "counter"])
def test_checkpoint_rejects_corrupted_geometry_history_or_energy(tmp_path, damage):
    front = model()
    state = ramp(front, 20)
    data = front.checkpoint_data(state)
    if damage == "geometry":
        data["model"]["points_xyz"][1] = [np.cos(.16), np.sin(.16), 0.]
    elif damage == "radius":
        data["model"]["radius_km"] *= 2
    elif damage == "version":
        data["model"]["version"] = "genesis-contact-0.1"
    elif damage == "cost":
        data["state"]["fracture_work_j"] *= 2
    elif damage == "event":
        data["state"]["events"][0]["old_right_m"] += .0001
    elif damage == "load":
        data["state"]["load_history_m"] = [state.opening_m]*state.load_step
    elif damage == "counter":
        data["state"]["load_step"] = True
    else:
        data["state"]["extra"] = 1
    file = tmp_path/"bad.json"
    file.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        CrackFrontModel.load_checkpoint(file)


@pytest.mark.parametrize("bad", [-.1, np.nan, np.inf, True, "1", None])
def test_invalid_load_rejected_without_changing_history(bad):
    front = model()
    initial = front.initial()
    with pytest.raises(ValueError):
        front.advance(initial, bad)
    assert initial == front.initial()


def test_state_cannot_migrate_between_material_geometry_or_energy_backends():
    front = model()
    state = ramp(front, 20)
    for other in (model(extension=.002), model(length=.2),
                  replace(front, oracle=replace(front.oracle, young_pa=60e9))):
        with pytest.raises(ValueError, match="another"):
            other.advance(state, state.opening_m)
    with pytest.raises(ValueError, match="ledger"):
        front.advance(replace(state, fracture_work_j=0.), state.opening_m)


def test_checkpoint_rejects_forged_event_even_when_final_ledger_matches():
    front = model()
    state = ramp(front, 20)
    bad = replace(state, events=(replace(state.events[0], opening_m=0.),)+state.events[1:])
    with pytest.raises(ValueError, match="history"):
        front.checkpoint_data(bad)


def test_advance_rejects_lost_events_and_invalid_historical_loads():
    front = model()
    state = ramp(front, 20)
    for bad in (replace(state, events=()),
                replace(state, load_history_m=(np.nan,)+state.load_history_m[1:]),
                replace(state, interval=None)):
        with pytest.raises(ValueError):
            front.advance(bad, state.opening_m)


def test_frozen_history_does_not_reproject_when_new_load_arrives():
    front = model(.0013)
    first = ramp(front, 20, .06)
    old_xyz = front.path.point_at([event.new_right_m for event in first.events])
    grown = front.advance(first, opening_at(front.oracle, .09))
    assert grown.events[:len(first.events)] == first.events
    np.testing.assert_array_equal(
        front.path.point_at([event.new_right_m for event in grown.events[:len(first.events)]]), old_xyz)
