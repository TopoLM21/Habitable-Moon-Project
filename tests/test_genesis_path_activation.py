"""Physical controls for an event-triggered, traction-preserving local birth."""
from copy import deepcopy
from dataclasses import fields, replace
import json

import numpy as np
import pytest

from tectonics.genesis_path_activation import (activate_tied_onset, load_born_path,
    ExtrinsicPathMechanics)
from tectonics.genesis_path_dynamics import PathLoading, PathMechanics
from tectonics.genesis_path_onset_event import locate_tied_onset
from test_genesis_path_dynamics import _material, _same
from test_genesis_path_onset_event import _ramp


def _birth_fixture(score=1-5e-7):
    model, state, stress, water, _, _ = _ramp(initial_score=score)
    transition = activate_tied_onset(model, state, stress, water)
    return model, state, transition


def _held_predictor(model, state, factor=1., years=.1):
    volume, elasticity = _material(model)
    memory = state.elastic_strain*factor
    _, force = model.basis.bulk(volume, elasticity, memory)
    force[model.basis.nparent:] = 0.
    return PathLoading(years, volume, elasticity, memory, np.ones(len(volume)),
        force, np.zeros(len(volume)))


def test_birth_replaces_reaction_without_changing_physical_state_or_energy():
    tied, before, report = _birth_fixture()
    after, model = report.state, report.model
    assert report.governing_mode == "tensile"
    assert report.released_dof_count == 2
    assert report.replacement_force_relative_error < 1e-13
    assert report.interface_energy_change_j == 0.
    assert report.strength_ratio == pytest.approx(1-5e-7)
    assert after.active_interval.length_m <= tied.basis.insertion.path.length_m
    for name in ("elapsed_years", "displacement_m", "elastic_strain", "drag_work_j",
            "external_work_j", "bulk_work_j", "bulk_loading_correction_j", "mechanical_remainder_j",
            "accepted_steps", "rejected_steps", "last_step_years"):
        np.testing.assert_array_equal(getattr(after, name), getattr(before, name))
    for field in fields(before.cohorts):
        if field.name != "traction_pa":
            np.testing.assert_array_equal(getattr(after.cohorts, field.name), getattr(before.cohorts, field.name))
    contact_force = model.jump_operator.T@(
        model.geometry.interface_area_m2[:, None]*after.cohorts.traction_pa).ravel()
    np.testing.assert_allclose(after.constraint_reaction_n+contact_force,
        before.constraint_reaction_n, rtol=2e-14, atol=1.)
    assert model.law_parameters == tied.law_parameters
    assert model.fingerprint != tied.fingerprint


def test_load_hold_after_birth_has_no_artificial_release_transient():
    tied, before, report = _birth_fixture()
    model, born = report.model, report.state
    original = deepcopy(born)
    for years in (.01, 1., 100.):
        after = model.trial(born, _held_predictor(model, born, years=years))
        np.testing.assert_array_equal(after.displacement_m, born.displacement_m)
        np.testing.assert_array_equal(after.elastic_strain, born.elastic_strain)
        np.testing.assert_array_equal(after.cohorts.traction_pa, born.cohorts.traction_pa)
        np.testing.assert_array_equal(after.cohorts.fracture_work_j, 0.)
        assert after.drag_work_j == born.drag_work_j
        assert after.mechanical_remainder_j == born.mechanical_remainder_j
    _same(born, original)
    # Same interval under the old zero-traction release visibly relaxes.
    prescribed = tied.release(before, born.active_interval)
    relaxed = tied.trial(prescribed, _held_predictor(tied, prescribed, years=.01))
    assert np.linalg.norm(relaxed.displacement_m-prescribed.displacement_m) > 1e-5


def test_increased_loading_causes_opening_damage_and_nonnegative_fracture_work():
    _, _, report = _birth_fixture()
    model, born = report.model, report.state
    advanced = model.trial(born, _held_predictor(model, born, factor=1.01))
    gap = (model.jump_operator@advanced.displacement_m).reshape(-1, 2)[:, 0]
    assert gap.max() > 0
    assert advanced.cohorts.damage.max() > 0
    assert advanced.cohorts.fracture_work_j.sum() > 0
    assert np.all(advanced.cohorts.fracture_work_j >= 0)
    assert advanced.equilibrium_residual <= model.parameters.equilibrium_tolerance
    model._validate_state(advanced)
    assert np.count_nonzero(gap) <= 2


def test_automatic_location_then_birth_from_admissible_lower_sample():
    model, start, stress, water, callback, _ = _ramp()
    original = deepcopy(start)
    bracket = locate_tied_onset(model, start, stress, water, callback, 20.,
        time_tolerance_years=1e-6)
    sample, local_stress, local_water = callback(bracket.lower.duration_years)
    transition = activate_tied_onset(model, sample, local_stress, local_water)
    assert transition.state.elapsed_years == pytest.approx(start.elapsed_years+8., abs=1e-6)
    _same(start, original)
    above, above_stress, above_water = callback(bracket.upper.duration_years)
    if bracket.upper.onset.maximum_observed_ratio > 1:
        with pytest.raises(ValueError, match="without strength overshoot"):
            activate_tied_onset(model, above, above_stress, above_water)


@pytest.mark.parametrize("score", [.5, 1.001, 4.])
def test_birth_refuses_below_threshold_or_already_overloaded_source(score):
    model, state, stress, water, _, _ = _ramp(initial_score=score)
    original = deepcopy(state)
    with pytest.raises(ValueError, match="strength"):
        activate_tied_onset(model, state, stress, water)
    _same(state, original)


@pytest.mark.parametrize("tolerance", [0., -1., .002, np.inf, np.nan, True])
def test_birth_refuses_invalid_tolerances(tolerance):
    model, state, stress, water, _, _ = _ramp(initial_score=.9999995)
    with pytest.raises(ValueError, match="strength_tolerance"):
        activate_tied_onset(model, state, stress, water, strength_tolerance=tolerance)


def test_birth_copies_state_and_owns_immutable_constitutive_context():
    tied, before, report = _birth_fixture()
    old_elastic = before.elastic_strain.copy()
    report.state.elastic_strain[0, 0] += 1.
    np.testing.assert_array_equal(before.elastic_strain, old_elastic)
    with pytest.raises(ValueError):
        report.model.birth_traction_pa.setflags(write=True)
    with pytest.raises(ValueError):
        report.model.birth_water.setflags(write=True)
    with pytest.raises(ValueError, match="activate_tied_onset"):
        report.model.initial(before.elastic_strain)
    with pytest.raises(ValueError, match="propagation"):
        report.model.release(report.state, report.state.active_interval)


def test_extrinsic_checkpoint_reconstructs_context_and_exact_future(tmp_path):
    tied, _, report = _birth_fixture()
    model, born = report.model, report.state
    one = model.trial(born, _held_predictor(model, born, factor=1.01))
    path = tmp_path/"born.npz"
    model.save_state(path, one)
    restored_model, restored = load_born_path(tied, path)
    assert restored_model.fingerprint == model.fingerprint
    _same(one, restored)
    loading = _held_predictor(model, one, factor=1.005)
    _same(model.trial(one, loading), restored_model.trial(restored, loading))
    with pytest.raises(ValueError):
        tied.load_state(path)


@pytest.mark.parametrize("corruption", ["traction", "water", "clock", "damage", "work", "force", "version", "extra"])
def test_checkpoint_rejects_tampered_birth_and_constitutive_history(tmp_path, corruption):
    tied, _, report = _birth_fixture()
    path = tmp_path/"bad.npz"
    report.model.save_state(path, report.state)
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    metadata = json.loads(str(arrays["metadata"]))
    if corruption == "traction":
        arrays["birth_traction_pa"] *= .9
    elif corruption == "water":
        arrays["birth_water"][:] = .1
    elif corruption == "clock":
        metadata["birth_time_years"] += .1
    elif corruption == "damage":
        arrays["cohort_damage"][0] = .5
    elif corruption == "work":
        arrays["cohort_fracture_work_j"][0] = 1e10
    elif corruption == "force":
        arrays["cohort_traction_pa"] *= .9
    elif corruption == "version":
        metadata["version"] = "invented"
    else:
        arrays["unexpected"] = np.array(1)
    arrays["metadata"] = np.array(json.dumps(metadata))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        load_born_path(tied, path)


def test_birth_checkpoint_rejects_another_discretization_or_law(tmp_path):
    tied, _, report = _birth_fixture()
    path = tmp_path/"born.npz"
    report.model.save_state(path, report.state)
    other = PathMechanics(tied.basis, tied.depth_m,
        law_parameters=replace(tied.law_parameters, tensile_strength_pa=3e6))
    with pytest.raises(ValueError, match="different tied mechanics"):
        load_born_path(other, path)


def test_fixed_interface_material_cannot_be_born_after_its_contact_activation():
    _, _, report = _birth_fixture()
    model, born = report.model, report.state
    advanced = model.trial(born, _held_predictor(model, born, years=2.))
    advanced = model.trial(advanced, _held_predictor(model, advanced, years=.1))
    advanced.cohorts.birth_time_myr[:] = (model.birth_time_years+.5)/1e6
    with pytest.raises(ValueError, match="before contact activation"):
        model._validate_state(advanced)
