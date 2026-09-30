"""Conservation and restart contracts of the moving material-shell experiment."""
from copy import deepcopy
from dataclasses import fields, replace
import json

import numpy as np
import pytest

from tectonics.genesis import (
    ENERGY_SCALE, GenesisParameters, mantle_enthalpy, surface_enthalpy,
)
from tectonics.genesis_mobile import (
    MobileModel, MobileParameters, _RetryStep, load_mobile_checkpoint,
    mobile_parameters_from_config, save_mobile_checkpoint,
)
from tectonics.genesis_onset import OnsetParameters, advance_orbit_thermal
from tectonics.genesis_shell import ShellParameters, rock_enthalpy
from tectonics.genesis_tides import TidalParameters


def _model(*, traction=0., tides=False, shell_overrides=None, thermal_overrides=None,
           mobile_overrides=None):
    thermal = replace(GenesisParameters(), **(thermal_overrides or {}))
    shell = replace(ShellParameters(subdivisions=1, initial_temperature_anomaly_k=0,
                                    convective_traction_pa=traction), **(shell_overrides or {}))
    orbit = TidalParameters(enabled=tides, spin_state="synchronous_zero_obliquity")
    return MobileModel(shell, thermal, OnsetParameters(), orbit,
                       replace(MobileParameters(), **(mobile_overrides or {})))


def _cold(model, temperature=1000.):
    """A solid restart with all removed heat explicitly recorded in the ledgers."""
    state, thermal, orbit = model.initial()
    energy = [mantle_enthalpy(temperature, model.thermal)/ENERGY_SCALE,
              surface_enthalpy(temperature, model.thermal)/ENERGY_SCALE, 0., 0.]
    energy[3] = thermal.initial_total_energy-sum(energy[:2])
    thermal = replace(thermal, energy=energy)
    h = np.full_like(state.column_enthalpy, rock_enthalpy(temperature, model.thermal))
    state = replace(state, column_enthalpy=h,
                    boundary_energy_j=float(np.sum(state.layer_mass_kg*h))-state.initial_column_energy_j)
    return state, thermal, orbit


def _assert_states_equal(first, second):
    for a, b in zip(first, second):
        for field in fields(a):
            left, right = getattr(a, field.name), getattr(b, field.name)
            if isinstance(left, np.ndarray):
                np.testing.assert_array_equal(left, right)
            else:
                assert left == right, field.name


def _run(model, state, end, dt):
    while state[0].time_myr < end-1e-14:
        *state, _ = model.step(*state, min(end, state[0].time_myr+dt))
        assert state[0].stopped_reason is None
        assert state[1].stopped_reason is None
    return tuple(state)


def test_mobile_conserves_mass_column_heat_and_orbital_input_once():
    model = _model(tides=True, thermal_overrides=dict(stellar_flux_w_m2=0,
                   giant_absorbed_flux_w_m2=0, radiogenic_specific_power_w_kg=0))
    state = model.initial()
    initial_mass = state[0].layer_mass_kg.copy()
    for target in (.01, .02, .03):
        *state, _ = model.step(*state, target)
        shell, thermal, orbit = state
        np.testing.assert_array_equal(shell.layer_mass_kg, initial_mass)
        assert shell.time_myr == thermal.time_myr == orbit.time_myr == target
        assert shell.tidal_heat_received_j == pytest.approx(orbit.dissipated_energy_j, rel=1e-12)
        assert thermal.energy[2]*ENERGY_SCALE*model.thermal.area_m2 == pytest.approx(
            orbit.dissipated_energy_j, rel=1e-10)
        global_row, material, motion, _ = model.diagnostics(*state)
        assert abs(global_row["relative_energy_residual"]) < 1e-10
        assert abs(material["relative_column_energy_residual"]) < 1e-12
        assert material["relative_material_mass_residual"] == 0
        assert abs(motion["orbit_heat_transfer_relative_residual"]) < 1e-12


def test_uniform_free_cooling_contracts_radius_without_lateral_motion_or_stress():
    model = _model()
    original = _cold(model)
    result = _run(model, original, .005, .001)
    state = result[0]
    _, material, motion, _ = model.diagnostics(*result)
    assert state.radius_km < original[0].radius_km
    np.testing.assert_allclose(state.vertices, original[0].vertices, atol=1e-11, rtol=0)
    assert material["max_elastic_strain"] < 1e-10
    assert material["max_tensile_stress_mpa"] < 1e-5
    assert motion["max_displacement_km"] < 1e-6
    assert np.max(state.damage) == 0
    assert abs(material["relative_column_energy_residual"]) < 1e-12
    data = model.fields(*result[:2])
    expected_depth = model.p.column_depth_km*(model.thermal.radius_km/state.radius_km)**2
    np.testing.assert_allclose(data["column_depth_km"], expected_depth, rtol=1e-12)


def test_checkpoint_restart_is_exact_after_geometry_and_memory_change(tmp_path):
    model = _model(traction=20000.)
    initial = _cold(model)
    state = _run(model, initial, .002, .001)
    assert np.max(np.abs(state[0].vertices-initial[0].vertices)) > 1e-8
    checkpoint = tmp_path/"mobile.npz"
    save_mobile_checkpoint(checkpoint, model, *state, {"step_myr": .001}, {"test": True})
    loaded_model, *restored, metadata = load_mobile_checkpoint(checkpoint)
    _assert_states_equal(state, restored)
    assert metadata["controls"] == {"step_myr": .001}
    direct = _run(model, state, .004, .001)
    resumed = _run(loaded_model, restored, .004, .001)
    _assert_states_equal(direct, resumed)


def test_viscous_flow_exceeds_old_total_strain_limit_with_small_elastic_memory():
    # A deliberately weak Maxwell viscosity isolates continued material flow
    # without damage softening. This is a constitutive test, not a lunar fit.
    model = _model(traction=50000., shell_overrides=dict(
        linear_expansion_per_k=0., tensile_strength_pa=1e12,
        viscosity_reference_pa_s=1e19, viscosity_min_pa_s=1e19, viscosity_max_pa_s=1e19))
    original = _cold(model)
    coarse = _run(model, original, .007, .0002)
    fine = _run(model, original, .007, .0001)
    _, material, _, _ = model.diagnostics(*fine)
    assert material["max_total_membrane_strain"] > .05
    assert material["max_elastic_strain"] < 1e-4
    assert material["last_incremental_strain"] < model.mobile_p.max_incremental_strain
    assert material["mechanical_equilibrium_residual"] < model.mobile_p.equilibrium_tolerance
    assert np.max(fine[0].path_length_km) > 100.
    assert np.max(np.abs(fine[0].vertices-original[0].vertices)) > .01
    np.testing.assert_array_equal(fine[0].layer_mass_kg, original[0].layer_mass_kg)
    assert abs(material["relative_column_energy_residual"]) < 1e-12
    # Both partitions represent the same loading history. This checks actual
    # material positions and elastic memory, not just a scalar stopping flag.
    np.testing.assert_allclose(coarse[0].vertices, fine[0].vertices, rtol=0, atol=1e-5)
    for name in ("path_length_km", "elastic_strain"):
        relative = np.linalg.norm(getattr(coarse[0], name)-getattr(fine[0], name))/np.linalg.norm(getattr(fine[0], name))
        assert relative < 5e-4


def test_rejected_trial_does_not_leak_heat_or_geometry(monkeypatch):
    model = _model(traction=20000., mobile_overrides={"min_step_myr": .001})
    original = _cold(model)
    pristine = deepcopy(original)
    real_trial = model._trial

    def reject_after_complete_trial(*args):
        real_trial(*args)
        raise _RetryStep("test_forced_rejection")

    monkeypatch.setattr(model, "_trial", reject_after_complete_trial)
    *result, rows = model.step(*original, .002)
    _assert_states_equal(original, pristine)
    assert result[0].stopped_reason == "test_forced_rejection"
    assert result[0].rejected_steps == 2
    assert result[0].accepted_steps == 0
    unchanged = replace(result[0], stopped_reason=None, rejected_steps=0)
    _assert_states_equal((unchanged, *result[1:]), pristine)
    assert rows[-1]["time_myr"] == 0


@pytest.mark.parametrize("tides", [False, True])
def test_freezing_inside_mobile_step_closes_all_clocks_and_energy(tmp_path, tides):
    # Synthetic cold-event fixture: 4 cm transport depth gives k/D=100 W/m2/K
    # so the small mantle reservoir transfers orbit-mean heat to the surface.
    model = _model(tides=tides, thermal_overrides=dict(mantle_mass_fraction=.001,
        mantle_depth_fraction_radius=.04/(GenesisParameters().radius_km*1000), stellar_flux_w_m2=0,
        giant_absorbed_flux_w_m2=0, radiogenic_specific_power_w_kg=0))
    model = MobileModel(model.p, model.thermal, model.onset_p,
                       replace(model.tides_p, eccentricity=.01), model.mobile_p)
    initial = _cold(model, 300.)
    if tides:
        unheated, *_ = advance_orbit_thermal(initial[1], initial[2], model.thermal,
            replace(model.tides_p, enabled=False), .05)
    *state, rows = model.step(*initial, .05)
    shell, thermal, orbit = state
    if tides:
        assert thermal.time_myr > 1.1*unheated.time_myr
    assert 0 < thermal.time_myr < .05
    assert thermal.stopped_reason == "surface_reached_freezing_limit_ice_not_modelled"
    assert shell.time_myr == thermal.time_myr == orbit.time_myr
    assert rows[-1]["surface_temperature_k"] == pytest.approx(273.16, abs=1e-6)
    assert shell.tidal_heat_received_j == pytest.approx(orbit.dissipated_energy_j, rel=1e-10)
    assert thermal.energy[2]*ENERGY_SCALE*model.thermal.area_m2 == pytest.approx(
        orbit.dissipated_energy_j, rel=1e-10, abs=1.)
    global_row, material, motion, _ = model.diagnostics(*state)
    assert global_row["effective_heat_transfer_w_m2_k"] == pytest.approx(100.)
    assert abs(global_row["relative_energy_residual"]) < 1e-10
    assert abs(material["relative_column_energy_residual"]) < 1e-12
    assert abs(motion["orbit_heat_transfer_relative_residual"]) < 1e-10
    path = tmp_path/"frozen.npz"
    save_mobile_checkpoint(path, model, *state, {}, {})
    loaded_model, *restored, _ = load_mobile_checkpoint(path)
    _assert_states_equal(state, restored)
    with pytest.raises(ValueError, match="unstopped"):
        loaded_model.step(*restored, .05)


def _corrupt_checkpoint(path, mutation):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files if key != "metadata"}
        meta = json.loads(str(archive["metadata"]))
    mutation(meta, arrays)
    np.savez_compressed(path, metadata=np.array(json.dumps(meta)), **arrays)


@pytest.mark.parametrize("mutation", [
    lambda m, a: m["parameters"]["shell"].__setitem__("seed", 1),
    lambda m, a: m["state"].__setitem__("time_myr", .1),
    lambda m, a: a["vertices"].__setitem__((0, 0), np.nan),
    lambda m, a: a["vertices"].__setitem__(0, a["vertices"][0]*2),
    lambda m, a: a["layer_mass_kg"].__setitem__((0, 0), a["layer_mass_kg"][0, 0]*1.01),
    lambda m, a: a["column_enthalpy"].__setitem__((0, 0), a["column_enthalpy"][0, 0]*1.01),
    lambda m, a: a["water_access"].__setitem__(0, 1.1),
    lambda m, a: a["damage"].__setitem__(0, -.1),
    lambda m, a: a["path_length_km"].__setitem__(0, -1.),
    lambda m, a: a["weak_duration_myr"].__setitem__(0, 1.),
    lambda m, a: a["elastic_strain"].__setitem__((0, 0), .1),
    lambda m, a: m["state"].__setitem__("tidal_heat_received_j", 1e20),
    lambda m, a: m["state"].__setitem__("boundary_energy_j", 1e28),
    lambda m, a: m["state"].__setitem__("accepted_steps", -1),
    lambda m, a: m["state"].__setitem__("last_step_myr", 1.),
    lambda m, a: m["state"].__setitem__("membrane_established", "false"),
    lambda m, a: m["state"].__setitem__("membrane_established", 1),
    lambda m, a: m["state"].__setitem__("membrane_established", True),
    lambda m, a: m["orbit"].__setitem__("semimajor_axis_km", 1.),
    lambda m, a: m["thermal_state"]["energy"].__setitem__(0, 0.),
    lambda m, a: a.__setitem__("column_enthalpy", a["column_enthalpy"][:-1]),
], ids=["parameters", "clock", "nan_vertex", "nonunit_vertex", "mass", "enthalpy",
        "water", "damage", "path", "age", "elastic_limit", "tidal_heat", "column_ledger",
        "counter", "step_age", "membrane_string", "membrane_integer", "molten_membrane",
        "orbit", "global_energy", "shape"])
def test_malformed_checkpoint_is_rejected(tmp_path, mutation):
    model = _model()
    path = tmp_path/"bad.npz"
    save_mobile_checkpoint(path, model, *model.initial(), {}, {})
    _corrupt_checkpoint(path, mutation)
    with pytest.raises(ValueError):
        load_mobile_checkpoint(path)


@pytest.mark.parametrize("kwargs", [{"max_incremental_strain": .1}, {"min_step_myr": 0},
    {"max_newton_iterations": 2.5}, {"max_elastic_strain": .2},
    {"max_radius_change_fraction": .5}, {"equilibrium_tolerance": .1}])
def test_invalid_mobile_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        replace(MobileParameters(), **kwargs).validate()


def test_config_rejects_unknown_schema_and_controls():
    with pytest.raises(ValueError):
        mobile_parameters_from_config({"genesis_mobile": {"schema_version": 2}})
    with pytest.raises(ValueError):
        mobile_parameters_from_config({"genesis_mobile": {"silently_ignored": True}})
