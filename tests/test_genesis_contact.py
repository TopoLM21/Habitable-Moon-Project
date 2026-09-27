"""Material and restart invariants of the split-bank contact continuation."""
from copy import deepcopy
from dataclasses import fields, replace
import io
import json

import numpy as np
import pytest

from tectonics.genesis import ENERGY_SCALE, GenesisParameters, mantle_enthalpy, surface_enthalpy
from tectonics.genesis_contact import (
    ContactModel, ContactParameters, _ContactRetry, load_contact_checkpoint,
    save_contact_checkpoint,
)
from tectonics.genesis_fault_law import WeakPlaneParameters
from tectonics.genesis_faults import FaultModel, save_fault_checkpoint
from tectonics.genesis_mobile import MobileParameters
from tectonics.genesis_onset import OnsetParameters
from tectonics.genesis_shell import ShellParameters, rock_enthalpy
from tectonics.genesis_tides import TidalParameters


def _source_bytes(tmp_path, *, traction_pa=20000., elastic=(.002, -.002, .008), active=True):
    """Small conserved cold snapshot with controlled inherited membrane stress."""
    shell = ShellParameters(subdivisions=1, initial_temperature_anomaly_k=0.,
                            convective_traction_pa=traction_pa, tensile_strength_pa=1e12)
    model = FaultModel(shell, GenesisParameters(), OnsetParameters(),
        TidalParameters(enabled=False, spin_state="synchronous_zero_obliquity"),
        MobileParameters(), WeakPlaneParameters(activation_persistence_myr=0.))
    state, thermal, orbit = model.initial()
    energy = [mantle_enthalpy(1000., model.thermal)/ENERGY_SCALE,
              surface_enthalpy(1000., model.thermal)/ENERGY_SCALE, 0., 0.]
    energy[3] = thermal.initial_total_energy-sum(energy[:2])
    thermal = replace(thermal, energy=energy)
    enthalpy = np.full_like(state.column_enthalpy, rock_enthalpy(1000., model.thermal))
    state = replace(state, membrane_established=True, column_enthalpy=enthalpy,
        boundary_energy_j=float(np.sum(state.layer_mass_kg*enthalpy))-state.initial_column_energy_j,
        damage=np.full_like(state.damage, .8))
    # Select planes once from isotropic stress, then prescribe the controlled
    # initial prestress. This is a short mechanical fixture, not a cooling run.
    if active:
        state = model._prepare_trial_state(state)
    state = replace(state, elastic_strain=np.tile(elastic, (len(state.damage), 1)))
    path = tmp_path/"fault_source.npz"
    save_fault_checkpoint(path, model, state, thermal, orbit, {}, {"test": "contact fixture"})
    return path.read_bytes(), model, (state, thermal, orbit), path


def _equal(first, second, *, omit=()):
    for field in fields(first):
        if field.name in omit:
            continue
        left, right = getattr(first, field.name), getattr(second, field.name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right, err_msg=field.name)
        else:
            assert left == right, field.name


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return _source_bytes(tmp_path_factory.mktemp("contact_source"))


@pytest.fixture(scope="module")
def evolved(source):
    model = ContactModel(source[0])
    state = model.step(model.initial(), 1.)
    state = model.step(state, 10.)
    assert state.stopped_reason is None
    return model, state


def test_initial_split_has_zero_jump_and_preserves_every_material_cell(source):
    model = ContactModel(source[0])
    state = model.initial()
    inherited = source[2][0]
    assert model.topology.mesh.vertex_count > len(inherited.vertices)
    np.testing.assert_array_equal(model.topology.parent_vertex[model.topology.mesh.faces], model.source_model.mesh.faces)
    np.testing.assert_array_equal(model.layer_mass_kg, inherited.layer_mass_kg)
    np.testing.assert_array_equal(model.column_enthalpy, inherited.column_enthalpy)
    np.testing.assert_array_equal(model.jump_operator@state.displacement_m, 0.)
    banks = model.mesh_for(state).vertices[model.topology.bank_vertices]
    np.testing.assert_array_equal(banks[:, 0], banks[:, 1])
    assert model.initial_mass_kg == float(inherited.layer_mass_kg.sum())
    assert model.column_energy_j == float(np.sum(inherited.layer_mass_kg*inherited.column_enthalpy))
    assert model.diagnostics(state)["relative_mass_residual"] == 0.
    assert model.diagnostics(state)["relative_column_energy_residual"] == 0.


def test_file_and_embedded_bytes_constructors_reproduce_same_source(source):
    from_file, initial = ContactModel.from_fault_checkpoint(source[3])
    embedded = ContactModel(source[0])
    assert from_file.source_hash == embedded.source_hash
    np.testing.assert_array_equal(from_file.topology.bank_vertices, embedded.topology.bank_vertices)
    np.testing.assert_array_equal(from_file.external_force, embedded.external_force)
    _equal(initial, embedded.initial())
    _equal(from_file.step(initial, 10.), embedded.step(embedded.initial(), 10.))


def test_contact_continuation_moves_independent_banks_and_converges_to_force_balance(source):
    model = ContactModel(source[0], ContactParameters(alignment_degrees=90.))
    initial = model.initial()
    state = model.step(initial, 10.)
    assert state.stopped_reason is None
    assert state.elapsed_years == 10.
    assert state.equilibrium_residual <= model.parameters.equilibrium_tolerance
    assert np.max(np.abs(state.plastic_slip_m)) > .1
    jump = (model.jump_operator@state.displacement_m).reshape(-1, 2)
    assert jump[:, 0].max() > .5 and jump[:, 0].min() < -.1
    assert np.any(state.interface_damage > 0.)
    assert np.all(state.traction_pa[jump[:, 0] < 0., 0] < 0.)
    assert np.max(-jump[:, 0]) <= model.parameters.max_penetration_m
    banks = model.mesh_for(state).vertices[model.topology.bank_vertices]
    difference = (banks[:, 1]-banks[:, 0]).reshape(-1, 3)*(model.radius_m+state.displacement_m[-1])
    actual_gap = np.einsum("pi,pi->p", difference, model.interface_normal)
    actual_slip = np.einsum("pi,pi->p", difference, model.interface_tangent)
    np.testing.assert_allclose(actual_gap, jump[:, 0], rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(actual_slip, jump[:, 1], rtol=2e-5, atol=2e-7)
    assert np.max(np.linalg.norm(difference, axis=1)) > .5
    # Forces stored with the accepted state must close the global balance,
    # including viscous drag over this exact accepted interval.
    assert state.accepted_steps == 1
    drag = model.drag_area_m2*model.parameters.basal_drag_pa_s_m/(10.*365.25*86400.)
    bulk_force = model.bulk_matrix@state.displacement_m+model.initial_bulk_force
    seam_force = model.jump_operator.T@(state.traction_pa*model.interface_area_m2[:, None]).ravel()
    residual = bulk_force+seam_force+drag*(state.displacement_m-initial.displacement_m)-model.external_force
    scale = max(np.linalg.norm(bulk_force), np.linalg.norm(seam_force), np.linalg.norm(model.external_force))
    assert np.linalg.norm(residual)/scale <= model.parameters.equilibrium_tolerance


def test_seam_opening_preserves_mass_heat_orbit_and_source_snapshot(source, evolved):
    model, state = evolved
    original = deepcopy(model.source_state)
    thermal, orbit = deepcopy(model.thermal_state), deepcopy(model.orbit)
    stored_energy = np.sum(model.layer_mass_kg*model.column_enthalpy, axis=1)
    result = model.step(state, 100.)
    assert result.stopped_reason is None
    mesh = model.mesh_for(result)
    data = model.fields(result)
    diagnostics = model.diagnostics(result)
    assert diagnostics["max_opening_m"] > 1.
    assert abs(diagnostics["signed_area_coverage_residual"]) > 1e-10
    actual_radius_km = (model.radius_m+result.displacement_m[-1])/1000.
    recovered_mass = (mesh.physical_cell_areas_km2(actual_radius_km)*data["column_depth_km"]
                      *model.source_model.p.density_kg_m3*1e9)
    np.testing.assert_allclose(recovered_mass, model.layer_mass_kg.sum(axis=1), rtol=4e-16)
    np.testing.assert_array_equal(np.sum(model.layer_mass_kg*model.column_enthalpy, axis=1), stored_energy)
    _equal(model.source_state, original)
    _equal(model.thermal_state, thermal)
    _equal(model.orbit, orbit)
    assert diagnostics["relative_mass_residual"] == diagnostics["relative_column_energy_residual"] == 0.
    assert result.friction_work_cell_j.sum() > 0.
    assert result.viscous_work_cell_j.sum() > 0.
    assert result.fracture_work_cell_j.sum() > 0.
    assert source[3].read_bytes() == source[0]


def test_no_load_or_inherited_stress_does_not_invent_motion_or_work(tmp_path):
    data, _, _, _ = _source_bytes(tmp_path, traction_pa=0., elastic=(0., 0., 0.))
    model = ContactModel(data)
    state = model.step(model.initial(), 100.)
    assert state.stopped_reason is None and state.elapsed_years == 100.
    assert state.equilibrium_residual == 0.
    np.testing.assert_array_equal(state.displacement_m, 0.)
    np.testing.assert_array_equal(state.traction_pa, 0.)
    np.testing.assert_array_equal(state.plastic_slip_m, 0.)
    np.testing.assert_array_equal(state.interface_damage, 0.)
    assert state.drag_work_j == state.external_work_j == 0.
    for name in ("friction_work_cell_j", "viscous_work_cell_j", "fracture_work_cell_j", "shear_remainder_cell_j"):
        np.testing.assert_array_equal(getattr(state, name), 0.)


def test_no_activated_faults_cannot_be_silently_converted_to_contact(tmp_path):
    data, _, _, _ = _source_bytes(tmp_path, active=False)
    with pytest.raises(ValueError, match="No aligned persistent weak edges"):
        ContactModel(data)


def test_completed_rejected_trials_do_not_leak_slip_work_or_damage(source, monkeypatch):
    model = ContactModel(source[0], ContactParameters(min_step_years=1.))
    initial = model.initial()
    pristine = deepcopy(initial)
    trial = model._trial
    attempts = []
    def reject_after_solve(*args):
        result = trial(*args)
        attempts.append(result)
        raise _ContactRetry("test_rejected_contact")
    monkeypatch.setattr(model, "_trial", reject_after_solve)
    rejected = model.step(initial, 10.)
    assert len(attempts) == 5
    assert attempts[0].friction_work_cell_j.sum() > 0.
    assert attempts[0].interface_damage.max() > 0.
    assert rejected.stopped_reason == "test_rejected_contact"
    assert rejected.rejected_steps == 5 and rejected.accepted_steps == 0
    _equal(initial, pristine)
    _equal(replace(rejected, stopped_reason=None, rejected_steps=0), pristine)


def test_retry_reproduces_the_same_accepted_time_partition(source, monkeypatch):
    model = ContactModel(source[0])
    initial = model.step(model.initial(), 1.)
    pristine = deepcopy(initial)
    trial = model._trial
    attempts = []
    def reject_first_complete_trial(*args):
        result = trial(*args)
        attempts.append(result)
        if len(attempts) == 1:
            raise _ContactRetry("test_step_jump_limit")
        return result
    monkeypatch.setattr(model, "_trial", reject_first_complete_trial)
    retried = model.step(initial, 10.)
    control_model = ContactModel(source[0])
    control = control_model.step(initial, 5.5)
    control = control_model.step(control, 10.)
    assert retried.rejected_steps == 1 and len(attempts) == 3
    _equal(replace(retried, rejected_steps=0), control)
    _equal(initial, pristine)


def test_actual_jump_guard_subdivides_steps_without_clipping_displacement(source, monkeypatch):
    model = ContactModel(source[0], ContactParameters(max_step_jump_m=.02))
    trial = model._trial
    accepted_jumps = []
    def observe(*args):
        result = trial(*args)
        accepted_jumps.append(np.max(np.abs(model.jump_operator@(result.displacement_m-args[0].displacement_m))))
        return result
    monkeypatch.setattr(model, "_trial", observe)
    initial = model.initial()
    state = model.step(initial, 1.)
    assert state.stopped_reason is None
    assert state.rejected_steps > 0 and state.accepted_steps > 1
    assert max(accepted_jumps) <= model.parameters.max_step_jump_m
    assert np.max(np.abs(model.jump_operator@state.displacement_m)) > model.parameters.max_step_jump_m
    np.testing.assert_array_equal(initial.displacement_m, 0.)


def test_checkpoint_embeds_source_and_resumes_exactly(tmp_path, evolved):
    model, state = evolved
    path = tmp_path/"contact.npz"
    save_contact_checkpoint(path, model, state)
    restored_model, restored = load_contact_checkpoint(path)
    _equal(restored, state)
    assert restored_model.source_bytes == model.source_bytes
    assert restored_model.source_hash == model.source_hash
    assert restored_model.parameters == model.parameters
    assert restored_model.law_parameters == model.law_parameters
    expected = model.step(state, 50.)
    actual = restored_model.step(restored, 50.)
    _equal(expected, actual)
    assert restored_model.diagnostics(actual) == model.diagnostics(expected)


def _tamper(path, mutation):
    with np.load(path, allow_pickle=False) as data:
        payload = {name: data[name].copy() for name in data.files}
    metadata = json.loads(str(payload["metadata"]))
    mutation(payload, metadata)
    payload["metadata"] = np.array(json.dumps(metadata))
    stream = io.BytesIO()
    np.savez_compressed(stream, **payload)
    stream.seek(0)
    return stream


@pytest.mark.parametrize("mutation", [
    lambda arrays, meta: arrays["source_checkpoint"].__setitem__(0, arrays["source_checkpoint"][0]^1),
    lambda arrays, meta: meta["parameters"]["contact"].__setitem__("basal_drag_pa_s_m", 2e14),
    lambda arrays, meta: arrays.__setitem__("displacement_m", arrays["displacement_m"][:-1]),
    lambda arrays, meta: arrays["traction_pa"].__setitem__((0, 0), np.nan),
    lambda arrays, meta: arrays["friction_work_cell_j"].__setitem__(0, -1.),
    lambda arrays, meta: arrays["plastic_slip_m"].__setitem__(0, 1e9),
    lambda arrays, meta: arrays["interface_damage"].__setitem__(0, .123),
    lambda arrays, meta: meta["state"].__setitem__("accepted_steps", 2.5),
    lambda arrays, meta: meta["state"].__setitem__("last_step_years", 1e6),
])
def test_checkpoint_rejects_corrupt_source_parameters_and_histories(tmp_path, evolved, mutation):
    model, state = evolved
    path = tmp_path/"contact.npz"
    save_contact_checkpoint(path, model, state)
    with pytest.raises(ValueError):
        load_contact_checkpoint(_tamper(path, mutation))


@pytest.mark.parametrize("key", ["shear_remainder_cell_j", "drag_work_j", "external_work_j"])
def test_unadvanced_checkpoint_cannot_contain_work(tmp_path, source, key):
    model = ContactModel(source[0])
    path = tmp_path/"contact.npz"
    save_contact_checkpoint(path, model, model.initial())
    def mutation(arrays, meta):
        if key in arrays:
            arrays[key][0] = 1.
        else:
            meta["state"][key] = 1.
    with pytest.raises(ValueError):
        load_contact_checkpoint(_tamper(path, mutation))


def test_checkpoint_rejects_stored_force_inconsistent_with_current_banks(tmp_path, evolved):
    model, state = evolved
    path = tmp_path/"contact.npz"
    save_contact_checkpoint(path, model, state)
    with pytest.raises(ValueError):
        load_contact_checkpoint(_tamper(path, lambda arrays, meta: arrays["traction_pa"].__setitem__((0, 0), 1e12)))


def test_checkpoint_rejects_opening_history_behind_the_actual_gap(tmp_path, evolved):
    model, state = evolved
    path = tmp_path/"contact.npz"
    save_contact_checkpoint(path, model, state)
    def mutation(arrays, meta):
        arrays["max_opening_m"][:] = 0.
        arrays["interface_damage"][:] = 0.
    with pytest.raises(ValueError):
        load_contact_checkpoint(_tamper(path, mutation))


def test_checkpoint_rejects_geometry_beyond_reference_strain_guard(tmp_path, evolved):
    model, state = evolved
    path = tmp_path/"contact.npz"
    save_contact_checkpoint(path, model, state)
    def mutation(arrays, meta):
        arrays["displacement_m"][-1] = .1*model.radius_m
    with pytest.raises(ValueError):
        load_contact_checkpoint(_tamper(path, mutation))


@pytest.mark.parametrize("kind", ["boolean", "string", "complex", "scalar"])
def test_checkpoint_rejects_arrays_with_nonnumeric_or_nonarray_types(tmp_path, source, kind):
    model = ContactModel(source[0])
    path = tmp_path/"contact.npz"
    save_contact_checkpoint(path, model, model.initial())
    def mutation(arrays, meta):
        if kind == "scalar":
            del arrays["displacement_m"]
            meta["state"]["displacement_m"] = 0.
        else:
            dtype = {"boolean": bool, "string": str, "complex": complex}[kind]
            arrays["displacement_m"] = arrays["displacement_m"].astype(dtype)
    with pytest.raises(ValueError):
        load_contact_checkpoint(_tamper(path, mutation))


def test_frozen_thermal_and_maxwell_window_stops_before_invalid_long_term_evolution(tmp_path):
    data, _, _, _ = _source_bytes(tmp_path, traction_pa=0., elastic=(0., 0., 0.))
    model = ContactModel(data, ContactParameters(max_frozen_duration_years=2., min_step_years=.25))
    pristine = model.initial()
    state = model.step(pristine, 10.)
    assert state.stopped_reason == "contact_frozen_state_limit"
    assert 0. < state.elapsed_years <= model.frozen_window_years == 2.
    assert state.rejected_steps > 0
    assert model.diagnostics(state)["omitted_maxwell_relaxation_fraction"] <= model.parameters.max_omitted_relaxation_fraction
    np.testing.assert_array_equal(state.displacement_m, 0.)
    assert pristine.elapsed_years == 0. and pristine.accepted_steps == 0
    with pytest.raises(ValueError, match="unstopped"):
        model.step(state, state.elapsed_years+1.)
