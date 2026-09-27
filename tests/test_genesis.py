"""Physical invariants and restart/output contracts for the thermal experiment."""
from dataclasses import replace
import csv
import json
import math

import numpy as np
import pytest

from tectonics.genesis import (
    CRITICAL_WATER_K,
    CRITICAL_WATER_PA,
    ENERGY_SCALE,
    SECONDS_PER_MYR,
    GenesisParameters,
    condensation_temperature_k,
    diagnose,
    initial_state,
    load_checkpoint,
    mantle_enthalpy,
    parameters_from_config,
    run_genesis,
    saturation_pressure_pa,
    save_checkpoint,
    surface_enthalpy,
    temperatures,
    vapor_column_kg_m2,
)


@pytest.fixture(scope="module")
def reference_run():
    p = GenesisParameters()
    state, rows = run_genesis(p, duration_myr=2.0, sample_interval_myr=0.1,
                              max_step_myr=0.02)
    return p, state, rows


def test_initial_state_is_fully_molten_and_all_inventory_is_atmospheric():
    p = GenesisParameters()
    row = diagnose(initial_state(p), p)
    expected_water_mass = p.water_volume_km3 * 1e9 * p.water_density_kg_m3
    assert row["mantle_melt_fraction"] == row["surface_melt_fraction"] == 1.0
    assert row["mantle_temperature_k"] == pytest.approx(p.initial_temperature_k)
    assert row["surface_temperature_k"] == pytest.approx(p.initial_surface_temperature_k)
    assert row["ocean_volume_km3"] == row["ocean_mass_kg"] == 0.0
    assert row["vapor_mass_kg"] == pytest.approx(expected_water_mass)
    assert row["steam_pressure_bar"] == pytest.approx(
        expected_water_mass * p.surface_gravity_m_s2 / p.area_m2 / 1e5)
    assert row["cooling_depth_proxy_km"] == 0.0


def test_water_and_integrated_energy_budgets_close_through_condensation(reference_run):
    p, state, rows = reference_run
    expected_water_mass = p.water_volume_km3 * 1e9 * p.water_density_kg_m3
    for row in rows:
        assert row["vapor_mass_kg"] + row["ocean_mass_kg"] == pytest.approx(
            expected_water_mass, rel=2e-15)
        assert row["total_water_mass_kg"] == pytest.approx(expected_water_mass)
        assert 0.0 <= row["ocean_fraction"] <= 1.0
        assert row["vapor_mass_kg"] >= 0.0
        assert row["ocean_mass_kg"] >= 0.0
        assert abs(row["relative_energy_residual"]) < 1e-10

    # Check accumulated input against an analytic integral, independently of
    # the diagnostic residual and mantle/surface heat exchange implementation.
    elapsed_s = state.time_myr * SECONDS_PER_MYR
    constant_input = (p.absorbed_stellar_flux_w_m2 + p.giant_absorbed_flux_w_m2
                      + p.tidal_heat_flux_w_m2) * elapsed_s
    decay = math.log(2) / p.radiogenic_half_life_myr
    initial_radio = (p.silicate_column_kg_m2 * p.radiogenic_specific_power_w_kg
                     * math.exp(-decay * p.system_age_at_start_myr))
    radio_input = initial_radio * -math.expm1(-decay * state.time_myr) / decay * SECONDS_PER_MYR
    assert state.energy[2] * ENERGY_SCALE == pytest.approx(
        constant_input + radio_input, rel=1e-10)
    energy_lost = (state.initial_total_energy - sum(state.energy[:2])) * ENERGY_SCALE
    assert energy_lost > 0
    assert energy_lost == pytest.approx(
        (state.energy[3] - state.energy[2]) * ENERGY_SCALE, rel=1e-10)


def test_surface_condenses_while_mantle_retains_melt_and_events_are_sampled(reference_run):
    p, state, rows = reference_run
    assert {"surface_lid", "mantle_rheology", "ocean_start"} <= state.events.keys()
    assert state.events["surface_lid"] < state.events["ocean_start"] < state.time_myr
    final = rows[-1]
    assert final["surface_temperature_k"] < CRITICAL_WATER_K
    assert final["mantle_temperature_k"] > p.solidus_k
    assert final["surface_melt_fraction"] == 0.0
    assert final["mantle_melt_fraction"] > 0.0
    assert final["ocean_fraction"] > 0.99
    assert final["cooling_depth_proxy_km"] > 0.0
    assert [row["time_myr"] for row in rows] == sorted(row["time_myr"] for row in rows)
    for event, event_time in state.events.items():
        event_row = next(row for row in rows if row["time_myr"] == event_time)
        if event == "ocean_start":
            assert event_row["surface_temperature_k"] == pytest.approx(
                condensation_temperature_k(p), abs=1e-6)
        else:
            reservoir = "surface" if event == "surface_lid" else "mantle"
            assert event_row[f"{reservoir}_melt_fraction"] == pytest.approx(
                p.lid_melt_threshold, abs=1e-8)


def test_larger_stellar_input_prevents_condensation_over_reference_interval(reference_run):
    p, reference, rows = reference_run
    hot = replace(p, stellar_flux_w_m2=2000.0)
    state, history = run_genesis(hot, duration_myr=reference.time_myr,
                                 sample_interval_myr=0.1, max_step_myr=0.02)
    assert "ocean_start" not in state.events
    assert history[-1]["ocean_fraction"] == 0.0
    assert history[-1]["surface_temperature_k"] > rows[-1]["surface_temperature_k"]
    assert history[-1]["mantle_temperature_k"] > rows[-1]["mantle_temperature_k"]


def test_dry_inventory_never_creates_water_or_condensation_event():
    p = GenesisParameters(water_volume_km3=0.0)
    state, rows = run_genesis(p, duration_myr=0.2, sample_interval_myr=0.1)
    assert "ocean_start" not in state.events
    for row in rows:
        assert row["vapor_mass_kg"] == row["ocean_mass_kg"] == 0.0
        assert row["ocean_fraction"] == row["steam_pressure_bar"] == 0.0
        assert abs(row["relative_energy_residual"]) < 1e-10


def test_freezing_stops_at_model_boundary_instead_of_inventing_liquid_water_below_it():
    p = GenesisParameters(stellar_flux_w_m2=0.0, radiogenic_specific_power_w_kg=0.0)
    state, rows = run_genesis(p, duration_myr=2.0, sample_interval_myr=0.1)
    assert state.time_myr < 2.0
    assert state.stopped_reason == "surface_reached_freezing_limit_ice_not_modelled"
    assert state.events["ocean_start"] < state.events["freezing_limit"]
    assert rows[-1]["surface_temperature_k"] == pytest.approx(273.16, abs=1e-6)
    assert min(row["surface_temperature_k"] for row in rows) >= 273.16 - 1e-6


@pytest.mark.parametrize("critical_pressure_ratio", [0.5, 1.0, 1.03, 1.19])
def test_water_partition_and_enthalpy_remain_continuous_at_critical_point(critical_pressure_ratio):
    base = GenesisParameters()
    mass = CRITICAL_WATER_PA * critical_pressure_ratio * base.area_m2 / base.surface_gravity_m_s2
    p = replace(base, water_volume_km3=mass / base.water_density_kg_m3 / 1e9)
    p.validate()
    probe = np.array([CRITICAL_WATER_K - 10, CRITICAL_WATER_K - 5,
                      CRITICAL_WATER_K - 1, CRITICAL_WATER_K - 1e-6,
                      CRITICAL_WATER_K, CRITICAL_WATER_K + 1e-6])
    enthalpies = np.array([surface_enthalpy(t, p) for t in probe])
    vapor = np.array([vapor_column_kg_m2(t, p) for t in probe])
    assert np.all(np.diff(enthalpies) > 0)
    assert np.all(np.diff(vapor) >= 0)
    assert np.all((vapor >= 0) & (vapor <= p.water_column_kg_m2))
    assert enthalpies[-3] == pytest.approx(enthalpies[-2], rel=1e-7)
    assert enthalpies[-1] == pytest.approx(enthalpies[-2], rel=1e-7)
    for t, enthalpy in zip(probe, enthalpies):
        recovered_mantle, recovered_surface = temperatures(np.array([
            mantle_enthalpy(base.initial_temperature_k, p) / ENERGY_SCALE,
            enthalpy / ENERGY_SCALE, 0.0, 0.0]), p)
        assert recovered_mantle == pytest.approx(base.initial_temperature_k)
        assert recovered_surface == pytest.approx(t, abs=1e-6)
    if critical_pressure_ratio < 1:
        onset = condensation_temperature_k(p)
        assert onset < CRITICAL_WATER_K
        assert saturation_pressure_pa(onset) == pytest.approx(
            p.water_column_kg_m2 * p.surface_gravity_m_s2)
    else:
        assert condensation_temperature_k(p) == CRITICAL_WATER_K


def test_half_max_step_converges_on_event_times_and_final_temperatures(reference_run):
    p, coarse, coarse_rows = reference_run
    fine, fine_rows = run_genesis(p, duration_myr=coarse.time_myr,
                                  sample_interval_myr=0.1, max_step_myr=0.01)
    assert fine.events.keys() == coarse.events.keys()
    for event in coarse.events:
        assert fine.events[event] == pytest.approx(coarse.events[event], rel=1e-5, abs=1e-6)
    for field in ("mantle_temperature_k", "surface_temperature_k", "ocean_fraction"):
        assert fine_rows[-1][field] == pytest.approx(coarse_rows[-1][field], rel=1e-6)


def test_checkpoint_resume_at_sample_boundary_matches_uninterrupted_run(tmp_path, reference_run):
    p, uninterrupted, uninterrupted_rows = reference_run
    controls = {"sample_interval_myr": 0.1, "max_step_myr": 0.02}
    first, first_rows = run_genesis(p, duration_myr=1.0, **controls)
    path = tmp_path / "checkpoint.json"
    save_checkpoint(path, first, p, controls=controls, provenance={"case": "restart test"})
    restored, restored_p, payload = load_checkpoint(path)
    assert restored_p == p
    assert restored == first
    assert payload["controls"] == controls
    assert payload["provenance"] == {"case": "restart test"}
    assert not path.with_suffix(".json.tmp").exists()
    resumed, resumed_rows = run_genesis(restored_p, duration_myr=uninterrupted.time_myr,
                                       state=restored, **controls)
    np.testing.assert_allclose(resumed.energy, uninterrupted.energy, rtol=1e-12)
    assert resumed.events == pytest.approx(uninterrupted.events, rel=1e-12)
    combined = first_rows + resumed_rows[1:]
    assert [row["time_myr"] for row in combined] == pytest.approx(
        [row["time_myr"] for row in uninterrupted_rows], rel=1e-12)
    for actual, expected in zip(combined, uninterrupted_rows):
        assert actual["surface_temperature_k"] == pytest.approx(
            expected["surface_temperature_k"], rel=1e-12)


@pytest.mark.parametrize("overrides", [
    {"radius_km": 0}, {"water_volume_km3": -1}, {"stellar_flux_w_m2": -1},
    {"bond_albedo": 1.1}, {"eclipse_fraction": -0.1},
    {"initial_temperature_k": 1000}, {"initial_surface_temperature_k": 1000},
    {"solidus_k": 2500}, {"rheology_transition_low_melt": 0.7},
    {"surface_layer_depth_m": 1e9}, {"water_volume_km3": 1e12},
    {"stellar_flux_w_m2": float("nan")}, {"radius_km": float("inf")},
    {"stellar_flux_w_m2": None}, {"radius_km": True},
])
def test_nonphysical_parameters_are_rejected(overrides):
    with pytest.raises(ValueError):
        replace(GenesisParameters(), **overrides).validate()


@pytest.mark.parametrize("section", [
    None, {"water_volume_km3": 1},
    {"schema_version": 1, "water_volume_km3": None},
    {"schema_version": 1, "water_volume_km3": 1, "stellar_flux_w_m2": None},
    {"schema_version": 1, "water_volume_km3": 1, "stellar_flux_w_m2": "NaN"},
    {"schema_version": 1, "water_volume_km3": 1, "unknown_physics": 1},
])
def test_invalid_or_incomplete_configuration_is_rejected(section):
    with pytest.raises(ValueError):
        parameters_from_config({"genesis": section})


@pytest.mark.parametrize("control", ["duration_myr", "sample_interval_myr", "max_step_myr"])
@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_time_controls_are_rejected(control, bad):
    with pytest.raises(ValueError):
        run_genesis(GenesisParameters(), **{control: bad})


@pytest.mark.parametrize("mutation", [
    "wrong_format", "parameter_hash", "null_parameter", "null_energy",
    "nan_energy", "broken_budget", "future_event",
])
def test_malformed_or_inconsistent_checkpoint_is_rejected(tmp_path, mutation):
    p = GenesisParameters()
    path = tmp_path / "checkpoint.json"
    save_checkpoint(path, initial_state(p), p)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "wrong_format":
        payload["format"] = "mature_tectonics_checkpoint"
    elif mutation == "parameter_hash":
        payload["parameters"]["stellar_flux_w_m2"] += 1
    elif mutation == "null_parameter":
        payload["parameters"]["stellar_flux_w_m2"] = None
    elif mutation == "null_energy":
        payload["state"]["energy"] = None
    elif mutation == "nan_energy":
        payload["state"]["energy"][0] = float("nan")
    elif mutation == "broken_budget":
        payload["state"]["energy"][0] *= 0.5
    elif mutation == "future_event":
        payload["state"]["events"]["ocean_start"] = 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_checkpoint(path)


@pytest.fixture
def tiny_cli_run(tmp_path):
    from run_genesis import main

    destination = tmp_path / "first"
    assert main(["--output", str(destination), "--duration-myr", "0.1"]) == 0
    return destination


def test_cli_produces_consistent_artifacts_and_resumes_to_new_directory(tmp_path, tiny_cli_run):
    from run_genesis import main

    destination = tiny_cli_run
    assert {"parameters.json", "checkpoint.json", "history.csv", "summary.json",
            "genesis_history.png"} <= {path.name for path in destination.iterdir()}
    assert (destination / "genesis_history.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    saved_before = (destination / "checkpoint.json").read_bytes()
    next_destination = tmp_path / "resumed"
    assert main(["--resume", str(destination / "checkpoint.json"),
                 "--output", str(next_destination), "--duration-myr", "0.2"]) == 0
    state, p, payload = load_checkpoint(next_destination / "checkpoint.json")
    summary = json.loads((next_destination / "summary.json").read_text(encoding="utf-8"))
    with (next_destination / "history.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[0]["time_myr"]) == pytest.approx(0.1)
    assert float(rows[-1]["time_myr"]) == pytest.approx(0.2)
    assert state.time_myr == summary["final"]["time_myr"] == pytest.approx(0.2)
    assert payload["provenance"]["resume_source"] == str((destination / "checkpoint.json").resolve())
    expected, _ = run_genesis(p, duration_myr=0.2)
    np.testing.assert_allclose(state.energy, expected.energy, rtol=1e-12)
    assert (destination / "checkpoint.json").read_bytes() == saved_before


def test_cli_refuses_existing_nonempty_output_without_changing_it(tmp_path, capsys):
    from run_genesis import main

    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "important.txt"
    sentinel.write_text("keep this", encoding="utf-8")
    assert main(["--output", str(destination), "--duration-myr", "0.1"]) == 1
    assert "existing runs will not be overwritten" in capsys.readouterr().err
    assert list(destination.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding="utf-8") == "keep this"


@pytest.mark.parametrize("extra", [["--stellar-flux-w-m2", "1400"],
                                    ["--sample-interval-myr", "0.2"]])
def test_cli_refuses_changed_physics_or_controls_on_resume(tmp_path, tiny_cli_run, extra, capsys):
    from run_genesis import main

    destination = tmp_path / "invalid_resume"
    assert main(["--resume", str(tiny_cli_run / "checkpoint.json"),
                 "--output", str(destination), "--duration-myr", "0.2", *extra]) == 1
    assert "Resume" in capsys.readouterr().err
    assert not destination.exists()


def test_cli_reports_null_configuration_cleanly_before_creating_output(tmp_path, capsys):
    from run_genesis import main

    config = tmp_path / "invalid.yaml"
    config.write_text(json.dumps({"genesis": {"schema_version": 1, "water_volume_km3": 1,
                                             "stellar_flux_w_m2": None}}), encoding="utf-8")
    destination = tmp_path / "invalid"
    assert main(["--config", str(config), "--output", str(destination)]) == 1
    assert "Genesis error:" in capsys.readouterr().err
    assert not destination.exists()
