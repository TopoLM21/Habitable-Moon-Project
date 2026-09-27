"""Conservation, material motion, and restart contracts of coupled genesis."""
from dataclasses import fields, replace
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis import (
    ENERGY_SCALE, mantle_enthalpy, parameters_from_config, surface_enthalpy,
)
from tectonics.genesis_onset import (
    OnsetModel, OnsetParameters, load_onset_checkpoint, save_onset_checkpoint,
    update_water_access,
)
from tectonics.genesis_shell import ShellParameters, rock_enthalpy
from tectonics.genesis_tides import tidal_parameters_from_config
from tectonics.simulation import load_config

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = {"sample_interval_myr": 0.1, "shell_step_myr": 0.01, "max_step_myr": 0.01}


def _model(*, control=False, water=True, exclusive_tidal_heat=False):
    config = load_config(ROOT / "configs/genesis_moon.yaml")
    thermal = parameters_from_config(config)
    if exclusive_tidal_heat:
        thermal = replace(thermal, stellar_flux_w_m2=0, giant_absorbed_flux_w_m2=0,
                          radiogenic_specific_power_w_kg=0)
    shell = ShellParameters(subdivisions=1)
    tides = tidal_parameters_from_config(config, thermal)
    if control:
        shell = replace(shell, initial_temperature_anomaly_k=0, convective_traction_pa=0)
        tides = replace(tides, enabled=False)
    return OnsetModel(shell, thermal, OnsetParameters(water_weakening=water), tides)


def _run(model, end, state=None, dt=0.01):
    state = model.initial() if state is None else state
    for index in range(round(state[0].time_myr / dt) + 1, round(end / dt) + 1):
        *state, _ = model.step(*state, index * dt, dt)
        assert state[0].stopped_reason is None
        assert state[1].stopped_reason is None
    return tuple(state)


@pytest.mark.parametrize("surface,liquid,enabled", [(1000, 0, True), (300, 0, True),
                                                    (300, 1, False), (700, 1, True)])
def test_dry_or_hot_or_disabled_water_does_not_create_access(surface, liquid, enabled):
    n = 4
    result = update_water_access(np.zeros(n), np.ones(n), np.ones(n), np.ones(n, bool),
                                  surface, liquid, 1.0, OnsetParameters(water_weakening=enabled))
    np.testing.assert_array_equal(result, 0)


def test_water_access_is_bounded_damage_sensitive_and_lost_with_remelting_or_drying():
    p = OnsetParameters()
    damage = np.array([0., 0.5, 1., 1.])
    wet = update_water_access(np.zeros(4), damage, np.ones(4),
                              np.array([True, True, True, False]), 300, 1, 0.1, p)
    assert 0 < wet[0] < wet[1] < wet[2] < 1
    assert wet[3] == 0
    dry = update_water_access(wet, damage, np.ones(4), np.ones(4, bool), 900, 0, 0.1, p)
    assert np.all(dry[:3] < wet[:3])
    diluted = update_water_access(wet, damage, np.full(4, 0.25), np.ones(4, bool), 900, 0, 0.1, p)
    np.testing.assert_allclose(diluted, dry * 0.25, rtol=1e-14)
    long_wet = update_water_access(wet, damage, np.ones(4), np.ones(4, bool), 300, 1, 1e6, p)
    np.testing.assert_array_equal(long_wet, 1)


def test_orbital_dissipation_is_received_once_by_thermal_energy_ledger():
    model = _model(exclusive_tidal_heat=True)
    state = model.initial()
    previous_heat = 0.
    for target in (0.01, 0.02, 0.03):
        *state, _ = model.step(*state, target)
        shell, thermal, onset, orbit = state
        assert orbit.dissipated_energy_j > previous_heat
        previous_heat = orbit.dissipated_energy_j
        assert onset.tidal_heat_received_j == pytest.approx(orbit.dissipated_energy_j, rel=1e-12)
        # All other input sources are off; this directly checks the thermal
        # integrator's input ledger, independent of the onset bookkeeping.
        assert thermal.energy[2] * ENERGY_SCALE * model.thermal.area_m2 == pytest.approx(
            orbit.dissipated_energy_j, rel=1e-10)
        global_row, shell_row, motion_row, _ = model.diagnostics(*state)
        assert abs(global_row["relative_energy_residual"]) < 1e-10
        assert abs(shell_row["relative_column_energy_residual"]) < 1e-10
        assert abs(motion_row["orbit_heat_transfer_relative_residual"]) < 1e-12
        np.testing.assert_array_equal(onset.water_access, 0)
        assert shell.time_myr == thermal.time_myr == onset.time_myr == orbit.time_myr


@pytest.mark.parametrize("tides_enabled", [False, True])
def test_freezing_inside_coupled_step_preserves_terminal_event_clocks_and_heat(tmp_path, tides_enabled):
    """A reconstructed cold restart isolates event coupling without a long run.

    Both thermal and column states retain their initial hot energy and record
    the removed energy in their respective output/boundary ledgers. Eccentricity
    is raised within the small-e regime so the changing mean orbital heat has
    a measurable effect on the freezing time.
    """
    base = _model(control=True, exclusive_tidal_heat=True)
    # A smaller, still valid mantle reservoir and efficient solid heat transfer
    # make this regression sensitive to changes in the orbit-mean heat input.
    thermal_p = replace(base.thermal, mantle_mass_fraction=0.001,
                        solid_transfer_w_m2_k=100.)
    model = OnsetModel(base.p, thermal_p, base.onset_p,
                       replace(base.tides_p, enabled=tides_enabled, eccentricity=0.01))
    shell, thermal, onset, orbit = model.initial()
    mantle_k, surface_k = 300., 300.
    energy = [mantle_enthalpy(mantle_k, model.thermal)/ENERGY_SCALE,
              surface_enthalpy(surface_k, model.thermal)/ENERGY_SCALE, 0., 0.]
    energy[3] = thermal.initial_total_energy-sum(energy[:2])
    thermal = replace(thermal, energy=energy)
    depth_fraction = (np.arange(model.p.column_layers)+0.5)/model.p.column_layers
    column_temperature = np.broadcast_to(
        surface_k+(mantle_k-surface_k)*depth_fraction,
        shell.column_enthalpy.shape,
    )
    enthalpy = rock_enthalpy(column_temperature, model.thermal)
    area_m2 = model.mesh.physical_cell_areas_km2(model.thermal.radius_km)*1e6
    column_energy = float(np.sum(enthalpy*area_m2[:, None])
                          * model.p.density_kg_m3
                          * model.p.column_depth_km*1000/model.p.column_layers)
    shell = replace(shell, column_enthalpy=enthalpy.copy(),
                    lid_thickness_km=np.full(model.mesh.cell_count, model.p.column_depth_km),
                    boundary_energy_j=column_energy-shell.initial_column_energy_j)
    before_global, before_shell, _, _ = model.diagnostics(shell, thermal, onset, orbit)
    assert abs(before_global["relative_energy_residual"]) < 1e-12
    assert abs(before_shell["relative_column_energy_residual"]) < 1e-12

    shell, thermal, onset, orbit, rows = model.step(shell, thermal, onset, orbit, 0.05)
    assert 0 < thermal.time_myr < 0.05
    assert thermal.stopped_reason == "surface_reached_freezing_limit_ice_not_modelled"
    assert thermal.events["freezing_limit"] == thermal.time_myr
    assert shell.time_myr == thermal.time_myr == onset.time_myr == orbit.time_myr
    assert rows[-1]["surface_temperature_k"] == pytest.approx(273.16, abs=1e-6)
    assert min(row["surface_temperature_k"] for row in rows) >= 273.16-1e-6
    assert onset.tidal_heat_received_j == pytest.approx(orbit.dissipated_energy_j, rel=1e-10)
    assert thermal.energy[2]*ENERGY_SCALE*model.thermal.area_m2 == pytest.approx(
        orbit.dissipated_energy_j, rel=1e-10, abs=1.)
    after_global, after_shell, motion, _ = model.diagnostics(shell, thermal, onset, orbit)
    assert abs(after_global["relative_energy_residual"]) < 1e-10
    assert abs(after_shell["relative_column_energy_residual"]) < 1e-10
    assert abs(motion["orbit_heat_transfer_relative_residual"]) < 1e-10

    checkpoint = tmp_path/"stopped_at_freezing.npz"
    save_onset_checkpoint(checkpoint, model, shell, thermal, onset, orbit, CONTROLS, {})
    restored_model, *loaded, _ = load_onset_checkpoint(checkpoint)
    assert loaded[1].stopped_reason == thermal.stopped_reason
    with pytest.raises(ValueError, match="stopped|advance|follow"):
        restored_model.step(*loaded, 0.05)


@pytest.mark.parametrize("tides_enabled", [False, True])
def test_uniform_free_cooling_forms_a_lid_without_accumulating_cyclic_tidal_drift(tides_enabled):
    model = _model(control=True)
    if tides_enabled:
        model = OnsetModel(model.p, model.thermal, model.onset_p, replace(model.tides_p, enabled=True))
    shell, thermal, onset, orbit = _run(model, 1.2)
    assert np.min(shell.lid_thickness_km) > 0.5
    assert shell.radial_strain < 0
    np.testing.assert_allclose(onset.displacement_rad, 0, atol=1e-12)
    np.testing.assert_allclose(onset.material_centroids, model.mesh.centroids, atol=1e-12)
    assert np.max(onset.path_length_km) < 1e-7
    assert np.max(np.linalg.norm(onset.velocity_km_myr, axis=1)) < 1e-7
    if tides_enabled:
        assert np.max(onset.tidal_stress_mpa) > 0
        assert orbit.dissipated_energy_j > 0
    else:
        assert orbit.dissipated_energy_j == onset.tidal_heat_received_j == 0


@pytest.fixture(scope="module")
def loaded_case():
    model = _model()
    state = _run(model, 1.2)
    return model, state


def test_loaded_shell_has_solved_material_motion_and_monotone_path(loaded_case):
    model, state = loaded_case
    shell, _, onset, _ = state
    assert np.max(shell.lid_thickness_km) > 0.5
    assert np.max(np.linalg.norm(onset.displacement_rad, axis=1)) > 1e-6
    assert np.max(onset.path_length_km) > 0.01
    assert np.max(np.linalg.norm(onset.material_centroids - model.mesh.centroids, axis=1)) > 1e-6
    assert np.max(onset.water_access) > 0
    assert np.max(onset.tidal_stress_mpa) > 0
    next_state = _run(model, 1.21, state)
    following = next_state[2]
    assert np.all(following.path_length_km >= onset.path_length_km)
    np.testing.assert_allclose(np.linalg.norm(following.material_centroids, axis=1), 1, atol=1e-13)
    # Velocity is tangent to the sphere at the new marker positions.
    radial_velocity = np.sum(following.velocity_km_myr * following.material_centroids, axis=1)
    np.testing.assert_allclose(radial_velocity, 0, atol=1e-9)


def test_water_weakening_switch_does_not_consume_or_remove_the_ocean(loaded_case):
    wet_model, wet_states = loaded_case
    dry_model = _model(water=False)
    dry_states = _run(dry_model, 1.2)
    np.testing.assert_array_equal(dry_states[2].water_access, 0)
    assert np.max(wet_states[2].water_access) > 0
    # Mechanical water access is a constitutive proxy, not a second reservoir.
    assert dry_states[1] == wet_states[1]
    dry_global = dry_model.diagnostics(*dry_states)[0]
    wet_global = wet_model.diagnostics(*wet_states)[0]
    assert dry_global["ocean_mass_kg"] == wet_global["ocean_mass_kg"] > 0
    assert dry_global["vapor_mass_kg"] + dry_global["ocean_mass_kg"] == pytest.approx(
        dry_global["total_water_mass_kg"], rel=1e-14)


def _assert_state_equal(actual, expected):
    for item in fields(expected):
        a, e = getattr(actual, item.name), getattr(expected, item.name)
        if isinstance(e, np.ndarray):
            np.testing.assert_array_equal(a, e)
        else:
            assert a == e, item.name


def test_joint_checkpoint_reproduces_next_steps_including_water_orbit_and_markers(tmp_path, loaded_case):
    model, states = loaded_case
    path = tmp_path / "onset_checkpoint.npz"
    provenance = {"test_source": "loaded canonical integration"}
    save_onset_checkpoint(path, model, *states, CONTROLS, provenance)
    restored_model, *loaded, metadata = load_onset_checkpoint(path)
    assert metadata["controls"] == CONTROLS and metadata["provenance"] == provenance
    assert restored_model.p == model.p
    assert restored_model.tides_p == model.tides_p
    for actual, expected in zip(loaded, states):
        _assert_state_equal(actual, expected)
    expected = _run(model, 1.23, states)
    actual = _run(restored_model, 1.23, tuple(loaded))
    for a, e in zip(actual, expected):
        _assert_state_equal(a, e)
    assert not Path(str(path) + ".tmp").exists()


def test_step_rejects_thermal_clock_mismatch_before_advancing():
    model = _model()
    shell, thermal, onset, orbit = model.initial()
    with pytest.raises(ValueError, match="clock|agree|synchron"):
        model.step(shell, replace(thermal, time_myr=0.001), onset, orbit, 0.01)


@pytest.mark.parametrize("mutation", [
    "hash", "clock", "shape", "nan_array", "missing_array", "water_bounds", "negative_path",
    "nonunit_marker", "nan_scalar", "thermal_budget", "column_budget", "tidal_transfer",
    "future_weak_duration", "future_fracture", "negative_tidal_energy",
    "orbit_angular_momentum", "orbit_eccentricity_gain", "orbit_energy_budget",
])
def test_malformed_joint_checkpoints_are_rejected(tmp_path, loaded_case, mutation):
    model, state = loaded_case
    path = tmp_path / "broken.npz"
    save_onset_checkpoint(path, model, *state, CONTROLS, {})
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files if k != "metadata"}
        metadata = json.loads(str(archive["metadata"]))
    if mutation == "hash":
        metadata["parameters"]["onset"]["wet_strength_fraction"] = 0.9
    elif mutation == "clock":
        metadata["orbit"]["time_myr"] += 0.1
    elif mutation == "shape":
        arrays["onset__displacement_rad"] = arrays["onset__displacement_rad"][:-1]
    elif mutation == "nan_array":
        arrays["onset__velocity_km_myr"][0, 0] = np.nan
    elif mutation == "missing_array":
        del arrays["onset__material_centroids"]
    elif mutation == "water_bounds":
        arrays["onset__water_access"][0] = 1.01
    elif mutation == "negative_path":
        arrays["onset__path_length_km"][0] = -0.1
    elif mutation == "nonunit_marker":
        arrays["onset__material_centroids"][0] *= 2
    elif mutation == "nan_scalar":
        metadata["states"]["onset"]["tidal_heat_received_j"] = float("nan")
    elif mutation == "thermal_budget":
        metadata["thermal_state"]["energy"][0] *= 0.5
    elif mutation == "column_budget":
        arrays["shell__column_enthalpy"][0] *= 0.5
    elif mutation == "tidal_transfer":
        metadata["states"]["onset"]["tidal_heat_received_j"] *= 2
    elif mutation == "future_weak_duration":
        arrays["onset__weak_duration_myr"][0] = state[0].time_myr + 0.1
    elif mutation == "future_fracture":
        metadata["states"]["shell"]["first_fracture_time_myr"] = state[0].time_myr + 0.1
    elif mutation == "negative_tidal_energy":
        metadata["orbit"]["dissipated_energy_j"] = -1
        metadata["states"]["onset"]["tidal_heat_received_j"] = -1
    elif mutation == "orbit_angular_momentum":
        metadata["orbit"]["semimajor_axis_km"] *= 1.01
    elif mutation == "orbit_eccentricity_gain":
        metadata["orbit"]["eccentricity"] = model.tides_p.eccentricity*1.01
    elif mutation == "orbit_energy_budget":
        # Agreement between heat-transfer ledgers is insufficient: their common
        # energy must also equal the loss from the finite orbital reservoir.
        metadata["orbit"]["dissipated_energy_j"] *= 2
        metadata["states"]["onset"]["tidal_heat_received_j"] *= 2
    np.savez_compressed(path, metadata=np.array(json.dumps(metadata)), **arrays)
    with pytest.raises(ValueError):
        load_onset_checkpoint(path)


@pytest.fixture(scope="module")
def cli_source(tmp_path_factory):
    from run_genesis_onset import main

    destination = tmp_path_factory.mktemp("onset_cli") / "first"
    controls = ["--sample-interval-myr", "0.1", "--max-step-myr", "0.01",
                "--shell-step-myr", "0.01", "--no-frames"]
    physics = ["--subdivisions", "1", "--regularization-km", "400"]
    assert main(["--output", str(destination), "--duration-myr", "0.1", *controls, *physics]) == 0
    return destination, controls, physics


def test_cli_resume_preserves_source_parameters_and_matches_continuous_state(cli_source, tmp_path):
    from run_genesis_onset import main

    source, controls, physics = cli_source
    checkpoint = source / "onset_checkpoint.npz"
    source_bytes = checkpoint.read_bytes()
    parameters = (source / "parameters.json").read_bytes()
    resumed, full = tmp_path / "resumed", tmp_path / "full"
    assert main(["--output", str(resumed), "--duration-myr", "0.2", "--resume", str(checkpoint), *controls]) == 0
    assert main(["--output", str(full), "--duration-myr", "0.2", *controls, *physics]) == 0
    actual = load_onset_checkpoint(resumed / "onset_checkpoint.npz")
    expected = load_onset_checkpoint(full / "onset_checkpoint.npz")
    for a, e in zip(actual[1:5], expected[1:5]):
        _assert_state_equal(a, e)
    assert checkpoint.read_bytes() == source_bytes
    assert (source / "parameters.json").read_bytes() == parameters
    saved_physics = json.loads(parameters)
    resumed_physics = json.loads((resumed / "parameters.json").read_text(encoding="utf-8"))
    for name in ("thermal", "shell", "onset", "tides", "controls"):
        assert resumed_physics[name] == saved_physics[name]
    assert resumed_physics["provenance"]["resume_source"] == str(checkpoint.resolve())
    for name in ("genesis_onset.png", "genesis_shell.png", "genesis_history.png"):
        assert (resumed / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("extra", [["--no-tides"], ["--no-water-weakening"],
                                    ["--regularization-km", "200"], ["--shell-step-myr", "0.005"]])
def test_cli_refuses_physics_and_time_control_changes_during_resume(cli_source, tmp_path, extra, capsys):
    from run_genesis_onset import main

    source, controls, _ = cli_source
    destination = tmp_path / "bad_resume"
    assert main(["--output", str(destination), "--duration-myr", "0.2",
                 "--resume", str(source / "onset_checkpoint.npz"), *controls, *extra]) == 1
    assert "Resume" in capsys.readouterr().err
    assert not destination.exists()


def test_cli_protects_existing_output(tmp_path, capsys):
    from run_genesis_onset import main

    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "important.txt"
    sentinel.write_text("keep", encoding="utf-8")
    assert main(["--output", str(output), "--duration-myr", "0.02", "--subdivisions", "1", "--no-frames"]) == 1
    assert "existing results are protected" in capsys.readouterr().err
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding="utf-8") == "keep"
