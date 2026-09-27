"""Restart preserves the separate ages and histories of growing interfaces."""
from dataclasses import fields, is_dataclass, replace
import json

import numpy as np
import pytest

from tectonics.genesis_coupled import (COUPLED_VERSION, CoupledModel, CoupledParameters,
    load_coupled_checkpoint, save_coupled_checkpoint)
from tectonics.genesis_faults import save_fault_checkpoint
from tectonics.genesis_shell import rock_enthalpy
from test_genesis_contact import _source_bytes


def _same(a, b):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif is_dataclass(a):
        for f in fields(a):
            _same(getattr(a, f.name), getattr(b, f.name))
    else:
        assert a == b


@pytest.fixture(scope="module")
def grown(tmp_path_factory):
    folder = tmp_path_factory.mktemp("growing_source")
    _, source, (state, thermal, orbit), path = _source_bytes(folder, elastic=(0., 0., 0.))
    # A resolved surface-to-hot-column gradient places the solidification
    # front inside the column. The cold contact fixture alone is fully solid.
    h = np.full_like(state.column_enthalpy, rock_enthalpy(1600., source.thermal))
    state = replace(state, column_enthalpy=h,
        boundary_energy_j=float(np.sum(state.layer_mass_kg*h))-state.initial_column_energy_j)
    save_fault_checkpoint(path, source, state, thermal, orbit, {}, {"test": "contact growth restart"})
    model = CoupledModel(path.read_bytes(), CoupledParameters(growth_layer_m=.001))
    initial = model.initial()
    state, thermal, orbit, _ = model.step(initial, thermal, orbit, 10e-6)
    assert state.stopped_reason is None
    assert len(state.cohorts.trace_index) > len(initial.cohorts.trace_index)
    return model, state, thermal, orbit


def test_grown_contact_roundtrip_and_continuation_are_exact(tmp_path, grown):
    model, state, thermal, orbit = grown
    path = tmp_path/"growing.npz"
    save_coupled_checkpoint(path, model, state, thermal, orbit)
    restored_model, restored, restored_thermal, restored_orbit = load_coupled_checkpoint(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        assert meta["format"] == COUPLED_VERSION == "genesis-coupled-0.2"
        assert {"cohort__birth_time_myr", "cohort__bonded", "interface_depth_ref_m"} <= set(data.files)
        assert "cohorts" not in meta["state"]
    for original, loaded in zip((state, thermal, orbit), (restored, restored_thermal, restored_orbit)):
        _same(original, loaded)
    assert not np.shares_memory(state.cohorts.area_ref_m2, restored.cohorts.area_ref_m2)
    expected = model.step(state, thermal, orbit, 20e-6)
    actual = restored_model.step(restored, restored_thermal, restored_orbit, 20e-6)
    for a, b in zip(expected[:3], actual[:3]):
        _same(a, b)
    assert actual[-1] == expected[-1]
    assert len(actual[0].cohorts.trace_index) > len(state.cohorts.trace_index)


@pytest.mark.parametrize("corruption", [
    "missing_history", "trace_type", "trace_range", "interval_overlap", "interval_gap", "interval_area",
    "represented_front", "represented_area", "future_birth", "pre_source_birth", "bonded_type",
    "bonding_history", "traction", "negative_work", "fracture_work", "aggregate_work", "aggregate_damage",
    "nan_reference", "negative_birth_energy", "missing_birth_energy", "missing_counter",
])
def test_growing_history_corruption_is_rejected(tmp_path, grown, corruption):
    model, state, thermal, orbit = grown
    path = tmp_path/"corrupt.npz"
    save_coupled_checkpoint(path, model, state, thermal, orbit)
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    meta = json.loads(str(arrays["metadata"]))
    if corruption == "missing_history": arrays.pop("cohort__birth_jump_m")
    if corruption == "trace_type": arrays["cohort__trace_index"] = arrays["cohort__trace_index"].astype(float)
    if corruption == "trace_range": arrays["cohort__trace_index"][0] = len(state.interface_depth_ref_m)
    if corruption == "interval_overlap": arrays["cohort__z_lo_ref_m"][-1] -= .01
    if corruption == "interval_gap": arrays["cohort__z_lo_ref_m"][-1] += .01
    if corruption == "interval_area": arrays["cohort__area_ref_m2"][0] *= 1.01
    if corruption == "represented_front": arrays["interface_depth_ref_m"][0] += .1
    if corruption == "represented_area": arrays["interface_birth_area_m2"][0] *= 1.01
    if corruption == "future_birth": arrays["cohort__birth_time_myr"][-1] = state.time_myr+1.
    if corruption == "pre_source_birth": arrays["cohort__birth_time_myr"][0] = model.source_time_myr-1.
    if corruption == "bonded_type": arrays["cohort__bonded"] = arrays["cohort__bonded"].astype(int)
    if corruption == "bonding_history": arrays["cohort__birth_gap_m"][0] = .1
    if corruption == "traction": arrays["cohort__traction_pa"][0, 0] += 1000.
    if corruption == "negative_work": arrays["cohort__friction_work_j"][0] = -1.
    if corruption == "fracture_work": arrays["cohort__fracture_work_j"][0] += 1e8
    if corruption == "aggregate_work": arrays["contact__friction_work_cell_j"][0] += 1e8
    if corruption == "aggregate_damage": arrays["contact__interface_damage"][0] += .01
    if corruption == "nan_reference": arrays["cohort__birth_jump_m"][0] = np.nan
    if corruption == "negative_birth_energy": meta["state"]["interface_birth_energy_j"] = -1.
    if corruption == "missing_birth_energy": meta["state"].pop("interface_birth_energy_j")
    if corruption == "missing_counter": meta["contact_state"].pop("accepted_steps")
    arrays["metadata"] = np.asarray(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        load_coupled_checkpoint(path)


def test_legacy_checkpoint_reports_missing_history_instead_of_inventing_it(tmp_path, grown):
    model, state, thermal, orbit = grown
    path = tmp_path/"legacy.npz"
    save_coupled_checkpoint(path, model, state, thermal, orbit)
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files if not name.startswith("cohort__")}
    meta = json.loads(str(arrays["metadata"]))
    meta["format"] = "genesis-coupled-0.1"
    arrays["metadata"] = np.asarray(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="growth histories.*original fault checkpoint"):
        load_coupled_checkpoint(path)


def test_unknown_format_is_not_read_as_current_cohort_model(tmp_path, grown):
    model, state, thermal, orbit = grown
    path = tmp_path/"future.npz"
    save_coupled_checkpoint(path, model, state, thermal, orbit)
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    meta = json.loads(str(arrays["metadata"]))
    meta["format"] = "genesis-coupled-9.9"
    arrays["metadata"] = np.asarray(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="version or hash"):
        load_coupled_checkpoint(path)
