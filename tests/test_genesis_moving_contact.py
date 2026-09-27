"""Persistence and admissibility of one moving, materially growing contact."""
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
import json

import numpy as np
import pytest

from tectonics.genesis_moving_contact import MovingContactLoading, MovingContactRetry
from test_genesis_moving_contact_independent import _birth_case, support


@pytest.fixture(scope="module")
def case(support):
    return _birth_case(support)


def same(a, b):
    for field in fields(a):
        x, y = getattr(a, field.name), getattr(b, field.name)
        if isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y, err_msg=field.name)
        elif is_dataclass(x):
            same(x, y)
        else:
            assert x == y, field.name


def loading(case, state, factor=1.01, years=.1):
    model, force, volume, young = case
    return MovingContactLoading(years, volume, young, state.elastic_strain*factor,
        np.ones(len(volume)), np.zeros(len(state.history.initial.trace_index)),
        lambda *a: factor*force(*a))


def test_real_local_opening_survives_next_reference_and_checkpoint(case, tmp_path):
    model = case[0]
    before = deepcopy(model.initial_state)
    first = model.trial(before, loading(case, before))
    assert model.jumps(first)[:, 0].max() > 1e-6
    assert first.history.initial.damage.max() > 0
    assert first.history.initial.fracture_work_j.sum() > 0
    assert np.linalg.norm(first.vertices-before.vertices) > 0
    assert first.equilibrium_residual <= model.parameters.equilibrium_tolerance
    model.save_state(tmp_path/"state.npz", first)
    restored = model.load_state(tmp_path/"state.npz")
    same(first, restored)
    future = loading(case, first)
    second = model.trial(first, future)
    same(second, model.trial(restored, future))
    assert np.max(model.jumps(second)[:, 0]) > np.max(model.jumps(first)[:, 0])
    np.testing.assert_array_equal(first.history.initial.area_ref_m2, second.history.initial.area_ref_m2)
    same(before, model.initial_state)


def test_material_growth_keeps_old_history_and_uses_fresh_sections(case):
    model, force, volume, young = case
    old = model.initial_state
    growth = 1.01
    forcing = MovingContactLoading(1., growth*volume, young, old.elastic_strain/growth,
        np.ones(len(volume)), np.zeros(len(old.history.initial.trace_index)), force)
    new = model.trial(old, forcing)
    same(old.history.initial, new.history.initial)
    assert len(new.history.added.area_ref_m2) == len(old.history.initial.area_ref_m2)
    np.testing.assert_allclose(new.history.added.area_ref_m2,
        .01*old.history.initial.area_ref_m2, rtol=4e-12)
    np.testing.assert_array_equal(new.history.added.traction_pa, 0.)
    np.testing.assert_array_equal(model.jumps(new), 0.)
    assert new.cohort_parameter_energy_j == 0.
    assert new.equilibrium_residual < 1e-12


def test_remelting_rejected_without_erasing_opening_or_history(case):
    model = case[0]
    old = model.trial(model.initial_state, loading(case, model.initial_state))
    original = deepcopy(old)
    predictor = loading(case, old)
    with pytest.raises(ValueError, match="remelting"):
        model.trial(old, replace(predictor, volume_m3=.99*predictor.volume_m3))
    same(old, original)


def test_rejected_step_does_not_commit_prepared_solidification(case):
    model, force, volume, young = case
    old = model.initial_state
    original = deepcopy(old)
    invalid = MovingContactLoading(.1, 1.01*volume, young, old.elastic_strain,
        np.ones(len(volume)), np.zeros(len(old.history.initial.trace_index)),
        lambda *a: np.full(a[-1].ndof, np.nan))
    with pytest.raises(ValueError, match="external force"):
        model.trial(old, invalid)
    same(old, original)


@pytest.mark.parametrize("corruption", ["q", "elastic", "drag", "external", "area", "fracture", "clock", "version", "extra"])
def test_checkpoint_rejects_corruption(case, tmp_path, corruption):
    model = case[0]
    state = model.trial(model.initial_state, loading(case, model.initial_state))
    path = tmp_path/"state.npz"
    model.save_state(path, state)
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(str(data["metadata"]))
    if corruption == "q":
        data["enrichment_m"] *= 2.
    elif corruption == "elastic":
        data["elastic_strain"][0, 0] += .001
    elif corruption == "drag":
        data["last_drag_force_n"][0] += 1e15
    elif corruption == "external":
        data["last_external_force_n"][0] += 1e15
    elif corruption == "area":
        data["initial_area_ref_m2"] *= 1.1
    elif corruption == "fracture":
        data["initial_fracture_work_j"][0] += 1e10
    elif corruption == "clock":
        metadata["elapsed_years"] = 0.
    elif corruption == "version":
        metadata["version"] = "unknown"
    else:
        data["unexpected"] = np.array([0.])
    data["metadata"] = np.array(json.dumps(metadata))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **data)
    with pytest.raises(ValueError):
        model.load_state(path)


def test_current_area_traction_is_explicitly_distinct_from_nominal_law(case):
    model = case[0]
    state = model.initial_state
    diagnostics = model.force_diagnostics(state)
    np.testing.assert_allclose(diagnostics["reference_to_current_area_ratio"], 1., rtol=3e-13)
    force = diagnostics["internal_force_n"]+diagnostics["contact_force_n"]+diagnostics["drag_force_n"]-diagnostics["external_force_n"]
    np.testing.assert_array_equal(force, diagnostics["residual_force_n"])
    assert diagnostics["relative_residual"] < 1e-12


@pytest.mark.parametrize("field,value", [("dt_years",0.), ("dt_years",True), ("effective_b",1.1),
    ("young_modulus_pa",-1.), ("water_per_trace",-0.1)])
def test_invalid_predictor_rejected_before_state_mutation(case, field, value):
    model = case[0]
    before = deepcopy(model.initial_state)
    predictor = loading(case, before)
    if isinstance(getattr(predictor, field), np.ndarray):
        value = np.full_like(getattr(predictor, field), value)
    with pytest.raises(ValueError):
        model.trial(before, replace(predictor, **{field:value}))
    same(before, model.initial_state)
