"""Physical-clock, restart and rollback contracts of evolving split contact."""
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
import json

import numpy as np
import pytest

from tectonics.genesis_contact import save_contact_checkpoint, ContactModel
from tectonics.genesis_coupled import (CoupledModel, CoupledParameters, _RetryStep,
    load_coupled_checkpoint, save_coupled_checkpoint)
from test_genesis_contact import _source_bytes


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return _source_bytes(tmp_path_factory.mktemp("coupled_source"))


@pytest.fixture(scope="module")
def evolved(source):
    model = CoupledModel(source[0])
    state = model.initial()
    state, thermal, orbit, _ = model.step(state, model.source.thermal_state, model.source.orbit,
                                          state.time_myr+10/1e6)
    assert state.stopped_reason is None
    return model, state, thermal, orbit


def _same(a, b, omit=()):
    for field in fields(a):
        if field.name in omit:
            continue
        x, y = getattr(a, field.name), getattr(b, field.name)
        if isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y, err_msg=field.name)
        elif is_dataclass(x):
            _same(x, y)
        else:
            assert x == y, field.name


def test_initial_state_keeps_physical_source_time_and_material(source):
    model = CoupledModel(source[0])
    state = model.initial()
    original = source[2][0]
    assert state.time_myr == model.source.thermal_state.time_myr == model.source.orbit.time_myr
    assert state.contact.elapsed_years == 0
    np.testing.assert_array_equal(model.layer_mass_kg, original.layer_mass_kg)
    np.testing.assert_array_equal(state.column_enthalpy, original.column_enthalpy)
    np.testing.assert_array_equal(state.elastic_strain, original.elastic_strain)


def test_actual_heat_and_mechanics_advance_together_without_input_mutation(source):
    model = CoupledModel(source[0])
    state, thermal, orbit = model.initial(), model.source.thermal_state, model.source.orbit
    copied = deepcopy((state, thermal, orbit))
    new, heat, tide, rows = model.step(state, thermal, orbit, state.time_myr+10/1e6)
    assert new.time_myr == heat.time_myr == tide.time_myr == state.time_myr+10/1e6
    assert new.contact.elapsed_years == pytest.approx(10.)
    assert not np.array_equal(heat.energy, thermal.energy)
    assert not np.array_equal(new.column_enthalpy, state.column_enthalpy)
    assert np.linalg.norm(new.contact.displacement_m) > 0
    for old, original in zip((state, thermal, orbit), copied):
        _same(old, original)
    assert abs(rows[-1]["relative_mass_residual"]) == 0
    assert abs(rows[-1]["relative_column_energy_residual"]) < 1e-12
    assert abs(rows[-1]["relative_global_energy_residual"]) < 1e-12


def test_checkpoint_resumes_bitwise_same_evolution(tmp_path, evolved):
    model, state, thermal, orbit = evolved
    path = tmp_path/"coupled.npz"
    save_coupled_checkpoint(path, model, state, thermal, orbit)
    loaded, restored, heat, tide = load_coupled_checkpoint(path)
    _same(state, restored)
    expected = model.step(state, thermal, orbit, state.time_myr+10/1e6)
    actual = loaded.step(restored, heat, tide, restored.time_myr+10/1e6)
    for a, b in zip(expected[:3], actual[:3]):
        _same(a, b)
    assert expected[-1] == actual[-1]


def test_frozen_contact_cannot_be_relabelled_as_coupled_thermal_time(tmp_path, source):
    model = ContactModel(source[0])
    contact = model.step(model.initial(), 10.)
    path = tmp_path/"contact.npz"
    save_contact_checkpoint(path, model, contact)
    with pytest.raises(ValueError):
        load_coupled_checkpoint(path)
    with pytest.raises(ValueError):
        CoupledModel.from_fault_checkpoint(path)


def test_failed_trial_rolls_back_heat_orbit_topology_and_contact(source, monkeypatch):
    model = CoupledModel(source[0], CoupledParameters(min_step_years=1.))
    state, thermal, orbit = model.initial(), model.source.thermal_state, model.source.orbit
    original = deepcopy((state, thermal, orbit))
    actual_trial = model._trial

    def rejected(*args):
        actual_trial(*args)  # Include a genuine thermal/mechanical calculation.
        raise _RetryStep("test_rejection")

    monkeypatch.setattr(model, "_trial", rejected)
    result, heat, tide, _ = model.step(state, thermal, orbit, state.time_myr+4/1e6)
    assert result.stopped_reason == "test_rejection" and result.rejected_steps == 3
    _same(result, original[0], omit=("stopped_reason", "rejected_steps"))
    _same(heat, original[1]); _same(tide, original[2])
    _same(state, original[0])


def test_roundoff_sized_time_tail_does_not_trigger_false_solver_stop(evolved):
    model, state, thermal, orbit = evolved
    # Put the synchronized test clocks at a Myr-scale age, where accumulating
    # 100-year steps can leave tens of ulps before the requested output time.
    age = 1.4055
    state = replace(state, time_myr=age,
                    contact=replace(state.contact, elapsed_years=(age-model.source_time_myr)*1e6))
    thermal, orbit = replace(thermal, time_myr=age), replace(orbit, time_myr=age)
    target = age+60*np.spacing(age)
    new, heat, tide, _ = model.step(state, thermal, orbit, target)
    assert new.stopped_reason is None and new.rejected_steps == state.rejected_steps
    _same(new, state)
    assert new.time_myr == heat.time_myr == tide.time_myr


def test_additional_cuts_preserve_geometry_and_existing_joules(evolved):
    model, state, thermal, _ = evolved
    original = model.mesh_for(state)
    edges = np.asarray(model.original_mesh.shared_edges)
    existing = set(map(tuple, state.cut_edges))
    new_edge = next(row for row in edges if tuple(row[2:]) not in existing)
    a, b, u, v = new_edge
    normals = state.plane_normal.copy()
    normal3 = np.cross(model.original_mesh.vertices[u], model.original_mesh.vertices[v])
    normal3 /= np.linalg.norm(normal3)
    from tectonics.genesis_material import face_frames
    frames = face_frames(model.original_mesh)
    for cell in (a, b):
        normal2 = normal3@frames[cell]
        normals[cell] = normal2/np.linalg.norm(normal2)
    damage = state.damage.copy(); damage[[a, b]] = .9
    active = state.fault_active.copy(); active[[a, b]] = True
    changed = replace(state, damage=damage, fault_active=active, plane_normal=normals)
    grown = model._grow_cuts(changed, thermal)
    assert len(grown.cut_edges) > len(state.cut_edges)
    moved = model.mesh_for(grown)
    np.testing.assert_array_equal(moved.vertices[moved.faces], original.vertices[original.faces])
    np.testing.assert_array_equal(grown.column_enthalpy, state.column_enthalpy)
    assert grown.contact.fracture_work_cell_j.sum() == pytest.approx(state.contact.fracture_work_cell_j.sum(), rel=1e-15)
    assert grown.contact.drag_work_j == state.contact.drag_work_j


def test_guard_checks_final_thickness_after_mechanical_motion(source, monkeypatch):
    model = CoupledModel(source[0])
    state, thermal, orbit = model.initial(), model.source.thermal_state, model.source.orbit
    phase = model._phase_fields

    def shifted(value, heat):
        result = phase(value, heat)
        if value.time_myr > state.time_myr:
            result = (*result[:-1], result[-1]*1.03)
        return result

    monkeypatch.setattr(model, "_phase_fields", shifted)
    with pytest.raises(_RetryStep, match="interface_geometry"):
        model._trial(state, thermal, orbit, state.time_myr+1e-6, .001)


@pytest.mark.parametrize("corrupt", ["clock", "hash", "temperature", "contact_force", "fracture_work", "column_heat", "counter", "birth_area",
                                    "negative_enthalpy", "elastic_limit", "lost_support"])
def test_malformed_checkpoints_rejected(tmp_path, evolved, corrupt):
    model, state, thermal, orbit = evolved
    path = tmp_path/"coupled.npz"
    save_coupled_checkpoint(path, model, state, thermal, orbit)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    meta = json.loads(str(arrays["metadata"]))
    if corrupt == "clock": meta["thermal_state"]["time_myr"] += 1
    if corrupt == "hash": meta["source_hash"] = "wrong"
    if corrupt == "temperature": meta["thermal_state"]["energy"][0] *= 2
    if corrupt == "contact_force": arrays["contact__traction_pa"][0, 0] += 1000
    if corrupt == "fracture_work": arrays["contact__fracture_work_cell_j"][0] += 1e10
    if corrupt == "column_heat": arrays["column_enthalpy"][0, 0] *= 2
    if corrupt == "counter": meta["state"]["accepted_steps"] = -1
    if corrupt == "birth_area": arrays["interface_birth_area_m2"][0] *= 1.1
    if corrupt == "elastic_limit": arrays["elastic_strain"][:] = [.04, .04, 0.]
    if corrupt in {"negative_enthalpy", "lost_support"}:
        if corrupt == "negative_enthalpy": arrays["column_enthalpy"][0, 0] = -1.
        else: arrays["column_enthalpy"][:] = 1e14
        # A formally balanced heat ledger cannot make invalid material usable.
        meta["state"]["boundary_energy_j"] = (float(np.sum(model.layer_mass_kg*arrays["column_enthalpy"]))
                                              -state.initial_column_energy_j)
    arrays["metadata"] = np.asarray(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        load_coupled_checkpoint(path)


def test_continuous_shell_can_enter_without_artificial_initial_cuts(tmp_path):
    source = _source_bytes(tmp_path, traction_pa=0., elastic=(0., 0., 0.), active=False)
    model = CoupledModel(source[0])
    state = model.initial()
    # Prevent pending fixture damage from becoming a deliberately seeded fault.
    state = replace(state, damage=np.zeros_like(state.damage))
    assert len(state.cut_edges) == 0
    new, thermal, orbit, rows = model.step(state, model.source.thermal_state, model.source.orbit, 1e-6)
    assert new.stopped_reason is None and len(new.cut_edges) == 0
    assert rows[-1]["component_count"] == 1 and rows[-1]["max_opening_m"] == 0
    assert np.max(np.abs(new.elastic_strain)) < 1e-10
    assert new.contact.displacement_m[-1] < 0
