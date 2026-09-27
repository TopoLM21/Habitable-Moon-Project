"""Moving-shell integration of persistent frictional material shear bands."""
from copy import deepcopy
from dataclasses import fields, replace
import json

import numpy as np
import pytest

from tectonics.genesis import ENERGY_SCALE, GenesisParameters, mantle_enthalpy, surface_enthalpy
from tectonics.genesis_fault_law import WeakPlaneParameters
from tectonics.genesis_faults import FaultModel, load_fault_checkpoint, save_fault_checkpoint
from tectonics.genesis_mobile import MobileModel, MobileParameters, MobileState, _RetryStep
from tectonics.genesis_onset import OnsetParameters
from tectonics.genesis_shell import ShellParameters, rock_enthalpy
from tectonics.genesis_tides import TidalParameters


def _model(*, enabled=True, persistence=0., shell_overrides=None, fault_overrides=None,
           mobile_overrides=None):
    shell = replace(ShellParameters(subdivisions=1, initial_temperature_anomaly_k=0.,
        convective_traction_pa=20000., tensile_strength_pa=1e12), **(shell_overrides or {}))
    fault = replace(WeakPlaneParameters(enabled=enabled, activation_persistence_myr=persistence,
        cohesion_pa=2e5, friction_dry=.2, friction_wet=.1), **(fault_overrides or {}))
    return FaultModel(shell, GenesisParameters(), OnsetParameters(),
        TidalParameters(enabled=False, spin_state="synchronous_zero_obliquity"),
        replace(MobileParameters(), **(mobile_overrides or {})), fault)


def _cold(model, damage=.75):
    """Controlled solid initial state with conserved heat and no invented work."""
    state, thermal, orbit = model.initial()
    energy = [mantle_enthalpy(1000., model.thermal)/ENERGY_SCALE,
              surface_enthalpy(1000., model.thermal)/ENERGY_SCALE, 0., 0.]
    energy[3] = thermal.initial_total_energy-sum(energy[:2])
    thermal = replace(thermal, energy=energy)
    h = np.full_like(state.column_enthalpy, rock_enthalpy(1000., model.thermal))
    state = replace(state, membrane_established=True, column_enthalpy=h,
        boundary_energy_j=float(np.sum(state.layer_mass_kg*h))-state.initial_column_energy_j,
        damage=np.full_like(state.damage, damage))
    return state, thermal, orbit


def _equal(first, second, names=None):
    for left, right in zip(first, second):
        for field in fields(left):
            if names is not None and field.name not in names:
                continue
            a, b = getattr(left, field.name), getattr(right, field.name)
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b, err_msg=field.name)
            else:
                assert a == b, field.name


def _run(model, states, end, dt=.001):
    while states[0].time_myr < end-1e-14:
        *states, _ = model.step(*states, min(end, states[0].time_myr+dt))
        assert states[0].stopped_reason is None
    return tuple(states)


@pytest.fixture(scope="module")
def evolved():
    model = _model(persistence=.002)
    return model, _run(model, _cold(model), .004)


def test_shear_changes_material_motion_and_reaches_global_equilibrium():
    model = _model()
    initial = _cold(model)
    result = _run(model, initial, .003)
    disabled = _model(enabled=False)
    control = _run(disabled, _cold(disabled), .003)
    state, thermal, orbit = result
    assert state.fault_active.all()
    assert np.max(state.cumulative_shear) > 1e-5
    assert np.max(np.abs(state.vertices-control[0].vertices)) > 1e-7
    assert state.friction_work_j > 0 and state.viscous_fault_work_j > 0
    assert np.all(state.friction_work_cell_j >= 0) and np.all(state.viscous_work_cell_j >= 0)
    assert state.friction_work_j == float(state.friction_work_cell_j.sum())
    assert state.viscous_fault_work_j == float(state.viscous_work_cell_j.sum())
    np.testing.assert_array_equal(state.layer_mass_kg, initial[0].layer_mass_kg)
    # Friction is a separate mechanical ledger, not extra heat in the global ODE.
    _equal(result[1:], control[1:])
    global_row, shell, motion, _ = model.diagnostics(*result)
    assert shell["mechanical_equilibrium_residual"] <= model.mobile_p.equilibrium_tolerance
    assert shell["relative_material_mass_residual"] == 0
    assert abs(shell["relative_column_energy_residual"]) < 1e-12
    assert abs(global_row["relative_energy_residual"]) < 1e-12
    assert motion["fault_dissipation_j"] == state.friction_work_j+state.viscous_fault_work_j
    data = model.fields(state, thermal)
    np.testing.assert_array_equal(data["equivalent_slip_km"], model.fault_p.band_width_km*state.cumulative_shear)
    np.testing.assert_allclose(np.linalg.norm(state.plane_normal, axis=1), 1., atol=2e-15)
    assert shell["max_shear_increment"] <= model.fault_p.max_shear_increment


def test_fault_disabled_reproduces_mobile_exactly():
    model = _model(enabled=False)
    mobile = MobileModel(model.p, model.thermal, model.onset_p, model.tides_p, model.mobile_p)
    fault_initial = _cold(model)
    mobile_initial = _cold(mobile)
    result = _run(model, fault_initial, .003)
    control = _run(mobile, mobile_initial, .003)
    _equal(result[:1], control[:1], names={f.name for f in fields(MobileState)})
    _equal(result[1:], control[1:])
    assert not result[0].fault_active.any()
    np.testing.assert_array_equal(result[0].cumulative_shear, 0.)
    np.testing.assert_array_equal(result[0].fault_candidate_age_myr, 0.)
    assert result[0].friction_work_j == result[0].viscous_fault_work_j == 0


def test_candidate_persistence_is_independent_from_damage_display_threshold():
    model = _model(persistence=.002, fault_overrides={"activation_damage": .4})
    state = _cold(model, damage=.5)  # Below the older D >= .65 display threshold.
    state = _run(model, state, .002)
    assert not state[0].fault_active.any()
    np.testing.assert_array_equal(state[0].weak_duration_myr, 0.)
    np.testing.assert_allclose(state[0].fault_candidate_age_myr, .002, atol=1e-16)
    state = _run(model, state, .003)
    assert state[0].fault_active.all()
    np.testing.assert_allclose(state[0].activation_time_myr, .002, atol=1e-16)
    np.testing.assert_array_equal(state[0].weak_duration_myr, 0.)


def test_interrupted_candidate_history_resets_without_premature_activation():
    model = _model(persistence=.002, fault_overrides={"activation_damage": .4})
    first = _run(model, _cold(model, damage=.5), .001)
    interrupted = (replace(first[0], damage=np.zeros_like(first[0].damage)), *first[1:])
    second = _run(model, interrupted, .002)
    np.testing.assert_array_equal(second[0].fault_candidate_age_myr, 0.)
    assert not second[0].fault_active.any()
    restored_damage = (replace(second[0], damage=np.full_like(second[0].damage, .5)), *second[1:])
    third = _run(model, restored_damage, .003)
    np.testing.assert_allclose(third[0].fault_candidate_age_myr, .001, atol=1e-16)
    assert not third[0].fault_active.any()


def test_activation_is_trial_local_and_an_existing_plane_is_not_reselected(evolved):
    model, state = evolved
    pristine = deepcopy(state)
    prepared = model._prepare_trial_state(state[0])
    assert prepared is state[0]
    changed_stress = replace(state[0], elastic_strain=-state[0].elastic_strain)
    unchanged_plane = model._prepare_trial_state(changed_stress)
    np.testing.assert_array_equal(unchanged_plane.plane_normal, state[0].plane_normal)
    np.testing.assert_array_equal(unchanged_plane.activation_time_myr, state[0].activation_time_myr)
    _equal(state, pristine)
    ready = _cold(_model())[0]
    ready_copy = deepcopy(ready)
    newly_prepared = _model()._prepare_trial_state(ready)
    assert newly_prepared.fault_active.all() and not ready.fault_active.any()
    _equal((ready,), (ready_copy,))


def test_rejected_trials_do_not_leak_activation_or_mechanical_work(monkeypatch):
    model = _model(mobile_overrides={"min_step_myr": .001})
    initial = _cold(model)
    pristine = deepcopy(initial)
    real_finish = model._finish_trial_state
    rejected_work = []
    def fail_after_work(*args, **kwargs):
        after = real_finish(*args, **kwargs)
        rejected_work.append(after.friction_work_j)
        assert after.fault_active.all()
        raise _RetryStep("test_fault_rejection")
    monkeypatch.setattr(model, "_finish_trial_state", fail_after_work)
    *result, rows = model.step(*initial, .002)
    assert len(rejected_work) == 2 and all(work > 0 for work in rejected_work)
    _equal(initial, pristine)
    assert result[0].stopped_reason == "test_fault_rejection"
    assert result[0].rejected_steps == 2 and result[0].accepted_steps == 0
    restored = replace(result[0], stopped_reason=None, rejected_steps=0)
    _equal((restored, *result[1:]), pristine)
    assert rows[-1]["time_myr"] == 0


def test_forced_shear_retry_matches_the_same_accepted_time_partition(monkeypatch):
    model = _model()
    initial = _cold(model)
    real_trial = model._trial
    attempted = []
    def reject_first_complete_trial(*args, **kwargs):
        result = real_trial(*args, **kwargs)
        attempted.append(result[0].friction_work_j)
        if len(attempted) == 1:
            raise _RetryStep("fault_shear_step_limit")
        return result
    monkeypatch.setattr(model, "_trial", reject_first_complete_trial)
    *retried, _ = model.step(*initial, .002)
    direct_model = _model()
    direct = _run(direct_model, _cold(direct_model), .002, dt=.001)
    assert retried[0].rejected_steps == 1 and len(attempted) == 3
    _equal((replace(retried[0], rejected_steps=0), *retried[1:]), direct)


def test_actual_shear_guard_subdivides_without_clipping_accepted_slip(monkeypatch):
    model = _model(fault_overrides={"max_shear_increment": 1e-5})
    real_finish = model._finish_trial_state
    accepted_increments = []
    def observe_accepted_return(*args, **kwargs):
        result = real_finish(*args, **kwargs)
        accepted_increments.append(float(np.max(np.abs(result.last_shear_increment))))
        return result
    monkeypatch.setattr(model, "_finish_trial_state", observe_accepted_return)
    result = _run(model, _cold(model), .001)
    assert result[0].rejected_steps > 0 and result[0].accepted_steps > 1
    assert max(accepted_increments) <= model.fault_p.max_shear_increment
    assert np.max(result[0].cumulative_shear) > model.fault_p.max_shear_increment
    assert result[0].friction_work_j > 0


def test_active_planes_do_not_create_shear_from_uniform_free_cooling():
    model = _model(shell_overrides={"convective_traction_pa": 0.})
    initial = _cold(model)
    result = _run(model, initial, .002)
    assert result[0].fault_active.all()
    np.testing.assert_array_equal(result[0].cumulative_shear, 0.)
    np.testing.assert_array_equal(result[0].last_shear_increment, 0.)
    assert result[0].friction_work_j == result[0].viscous_fault_work_j == 0.
    assert result[0].radius_km < initial[0].radius_km
    assert np.max(result[0].path_length_km) < 1e-6


def test_checkpoint_preserves_all_fault_history_and_exact_continuation(tmp_path, evolved):
    model, state = evolved
    checkpoint = tmp_path/"faults.npz"
    save_fault_checkpoint(checkpoint, model, *state, {"step_myr": .001}, {"test": "fault restart"})
    loaded_model, *restored, metadata = load_fault_checkpoint(checkpoint)
    _equal(state, restored)
    assert metadata["controls"] == {"step_myr": .001}
    assert loaded_model.fault_p == model.fault_p
    direct = _run(model, state, .006)
    resumed = _run(loaded_model, restored, .006)
    _equal(direct, resumed)
    assert resumed[0].friction_work_j > state[0].friction_work_j


def _tamper(path, mutation):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files if name != "metadata"}
        meta = json.loads(str(archive["metadata"]))
    mutation(meta, arrays)
    np.savez_compressed(path, metadata=np.array(json.dumps(meta)), **arrays)


@pytest.mark.parametrize("mutation", [
    lambda m, a: a["plane_normal"].__setitem__(0, [2., 0.]),
    lambda m, a: a["plane_normal"].__setitem__((0, 0), np.nan),
    lambda m, a: a.__setitem__("plane_normal", a["plane_normal"][:, :1]),
    lambda m, a: a.__setitem__("fault_active", a["fault_active"].astype(int)),
    lambda m, a: a["activation_time_myr"].__setitem__(0, -1.),
    lambda m, a: a["activation_time_myr"].__setitem__(0, 1.),
    lambda m, a: a["activation_time_myr"].__setitem__(0, 0.),
    lambda m, a: a["fault_candidate_age_myr"].__setitem__(0, 1.),
    lambda m, a: a["fault_candidate_age_myr"].__setitem__(0, -.1),
    lambda m, a: a["cumulative_shear"].__setitem__(0, -1.),
    lambda m, a: a["signed_shear"].__setitem__(0, 1.),
    lambda m, a: a["last_shear_increment"].__setitem__(0, 1.),
    lambda m, a: a["friction_work_cell_j"].__setitem__(0, -1.),
    lambda m, a: a["viscous_work_cell_j"].__setitem__(0, -1.),
    lambda m, a: m["state"].__setitem__("friction_work_j", m["state"]["friction_work_j"]*1.1),
    lambda m, a: m["state"].__setitem__("viscous_fault_work_j", "invalid"),
    lambda m, a: a["shear_strength_pa"].__setitem__(0, -1.),
], ids=["plane_unit", "plane_nan", "plane_shape", "active_dtype", "birth_negative", "birth_future",
        "birth_before_persistence", "age_future", "age_negative", "negative_shear", "signed_shear",
        "increment", "friction_cell", "viscous_cell", "friction_ledger", "work_scalar", "strength"])
def test_fault_checkpoint_rejects_corrupt_history(tmp_path, evolved, mutation):
    model, state = evolved
    path = tmp_path/"corrupt.npz"
    save_fault_checkpoint(path, model, *state, {}, {})
    _tamper(path, mutation)
    with pytest.raises(ValueError):
        load_fault_checkpoint(path)


@pytest.mark.parametrize("mutation", [
    lambda m, a: a["plane_normal"].__setitem__(0, [1., 0.]),
    lambda m, a: a["activation_time_myr"].__setitem__(0, 0.),
    lambda m, a: a["cumulative_shear"].__setitem__(0, .001),
    lambda m, a: (a["friction_work_cell_j"].__setitem__(0, 1.), m["state"].__setitem__("friction_work_j", 1.)),
], ids=["dormant_normal", "dormant_birth", "dormant_shear", "dormant_work"])
def test_unactivated_material_cannot_have_fault_history(tmp_path, mutation):
    model = _model()
    path = tmp_path/"dormant.npz"
    save_fault_checkpoint(path, model, *model.initial(), {}, {})
    _tamper(path, mutation)
    with pytest.raises(ValueError):
        load_fault_checkpoint(path)
