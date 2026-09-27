"""Physical invariants and restart contracts for the passive genesis shell."""
from dataclasses import fields, replace
import csv
import json

import numpy as np
import pytest

from tectonics.genesis import (
    GenesisParameters, SECONDS_PER_MYR, advance as advance_thermal,
    initial_state, temperatures,
)
from tectonics.genesis_shell import (
    Membrane,
    ShellParameters,
    advance_shell,
    diagnose_shell,
    evolve_damage,
    initialize_shell,
    load_shell_checkpoint,
    maxwell_factors,
    maximum_total_strain,
    principal_tensile,
    rock_enthalpy,
    rock_temperature,
    save_shell_checkpoint,
    shell_fields,
)
from tectonics.mesh import build_icosphere


@pytest.fixture(scope="module")
def small_shell():
    p = ShellParameters(subdivisions=1, column_layers=16, column_depth_km=10.0)
    thermal = GenesisParameters()
    mesh = build_icosphere(p.subdivisions)
    return mesh, Membrane(mesh, p.poisson_ratio), p, thermal


@pytest.mark.parametrize("subdivisions", [1, 2])
def test_uniform_cooling_contracts_freely_without_generating_stress(subdivisions):
    mesh = build_icosphere(subdivisions)
    p = ShellParameters(subdivisions=subdivisions)
    membrane = Membrane(mesh, p.poisson_ratio)
    eigen = np.zeros((mesh.cell_count, 3))
    eigen[:, :2] = -0.005
    # Varying thickness and damage must not constrain a compatible contraction.
    thickness = 2.0 + mesh.centroids[:, 0]
    stiffness = 0.6 + 0.3 * mesh.centroids[:, 1]
    strain, stress, residual, radial = membrane.solve(
        eigen, thickness, stiffness, p.young_modulus_pa)
    np.testing.assert_allclose(strain, eigen, rtol=0, atol=1e-14)
    assert np.max(np.abs(stress)) < 1e-3
    assert radial == pytest.approx(-0.005, abs=1e-14)
    assert residual < 1e-10


def test_rigid_rotations_produce_no_membrane_strain(small_shell):
    mesh, membrane, _, _ = small_shell
    for axis in np.eye(3):
        displacement = np.zeros(membrane.ndof)
        rotation = np.cross(axis, mesh.vertices)
        displacement[:-1] = np.einsum(
            "nij,ni->nj", membrane.vertex_basis, rotation).ravel()
        strain = np.einsum("fai,fi->fa", membrane.b, displacement[membrane.dofs])
        np.testing.assert_allclose(strain, 0.0, rtol=0, atol=2e-14)


def test_heterogeneous_cooling_has_balanced_tension_and_compression(small_shell):
    mesh, membrane, p, _ = small_shell
    eigen = np.zeros((mesh.cell_count, 3))
    eigen[:, :2] = (-0.001 * (mesh.centroids[:, 2] ** 2 - 1 / 3))[:, None]
    thickness = 2.0 + mesh.centroids[:, 0]
    _, stress, residual, _ = membrane.solve(
        eigen, thickness, np.ones(mesh.cell_count), p.young_modulus_pa)
    average = (stress[:, 0] + stress[:, 1]) / 2
    assert np.max(average) > 1e6
    assert np.min(average) < -1e6
    assert residual < 1e-10
    # Independent virtual-work check: free radius gives zero net radial force.
    weighted_trace = mesh.areas_unit_sphere * thickness * (stress[:, 0] + stress[:, 1])
    assert abs(np.sum(weighted_trace)) < 1e-10 * np.sum(np.abs(weighted_trace))


def test_silicate_enthalpy_inversion_crosses_both_phase_boundaries():
    p = GenesisParameters()
    temperature = np.array([300.0, p.solidus_k - 1e-6, p.solidus_k,
                            p.solidus_k + 1e-6, 1700.0, p.liquidus_k - 1e-6,
                            p.liquidus_k, p.liquidus_k + 1e-6, 2300.0])
    enthalpy = rock_enthalpy(temperature, p)
    assert np.all(np.diff(enthalpy) > 0)
    np.testing.assert_allclose(rock_temperature(enthalpy, p), temperature,
                               rtol=0, atol=1e-10)
    assert enthalpy[-1] - enthalpy[0] == pytest.approx(
        p.silicate_heat_capacity_j_kg_k * 2000.0 + p.silicate_latent_heat_j_kg)


def test_fully_molten_initial_shell_has_no_lid_damage_or_stress(small_shell):
    mesh, membrane, p, thermal = small_shell
    state = initialize_shell(mesh, p, thermal)
    assert np.min(rock_temperature(state.column_enthalpy, thermal)) > thermal.liquidus_k
    state = advance_shell(state, mesh, membrane, p, thermal, 0.001,
                          2300.0, 2300.0, 2300.0, 2300.0)
    data = shell_fields(state, mesh, p, thermal, 2300.0, 2300.0)
    assert state.first_fracture_time_myr is None
    for name in ("lid_thickness_km", "damage", "stress_pa", "peak_tensile_pa"):
        np.testing.assert_array_equal(getattr(state, name), 0)
    np.testing.assert_array_equal(data["solid_fraction"], 0)
    assert not np.any(data["failed_edges"])


def test_column_flux_matches_independent_one_step_energy_transfer(small_shell):
    mesh, membrane, base, thermal = small_shell
    p = replace(base, initial_temperature_anomaly_k=0.0, convective_traction_pa=0.0)
    state = initialize_shell(mesh, p, thermal)
    dt_myr = 1e-5
    surface, mantle = 2100.0, 2350.0
    dz = p.column_depth_km * 1000 / p.column_layers
    top_flux = 2 * p.conductivity_w_m_k * (surface - 2300.0) / dz
    bottom_flux = 2 * p.conductivity_w_m_k * (2300.0 - mantle) / dz
    result = advance_shell(state, mesh, membrane, p, thermal, dt_myr,
                           surface, surface, mantle, mantle)
    expected = state.column_enthalpy.copy()
    expected[:, 0] += top_flux * dt_myr * SECONDS_PER_MYR / (p.density_kg_m3 * dz)
    expected[:, -1] -= bottom_flux * dt_myr * SECONDS_PER_MYR / (p.density_kg_m3 * dz)
    np.testing.assert_allclose(result.column_enthalpy, expected, rtol=2e-15)
    assert result.boundary_energy_j == pytest.approx(
        thermal.area_m2 * (top_flux - bottom_flux) * dt_myr * SECONDS_PER_MYR,
        rel=2e-15)
    # Advance returns a new state and leaves its restart source unchanged.
    np.testing.assert_array_equal(state.column_enthalpy,
                                  rock_enthalpy(np.full(expected.shape, 2300.0), thermal))


@pytest.fixture(scope="module")
def quenched_columns(small_shell):
    mesh, membrane, base, thermal = small_shell
    p = replace(base, initial_temperature_anomaly_k=0.0, convective_traction_pa=0.0)
    results = []
    for dt in (0.001, 0.0005, 0.00025):
        state = initialize_shell(mesh, p, thermal)
        for step in range(1, round(0.03 / dt) + 1):
            state = advance_shell(state, mesh, membrane, p, thermal, step * dt,
                                  700.0, 700.0, 2300.0, 2300.0)
        results.append(state)
    return mesh, p, thermal, results


def test_solidifying_columns_close_their_separate_boundary_energy_ledger(quenched_columns):
    mesh, p, thermal, results = quenched_columns
    for state in results:
        temperature = rock_temperature(state.column_enthalpy, thermal)
        assert np.all(temperature[:, 0] < thermal.solidus_k)
        assert np.all(temperature[:, -1] > thermal.liquidus_k)
        assert np.all((state.lid_thickness_km > 0.5) & (state.lid_thickness_km < 1))
        assert state.boundary_energy_j < 0
        total = np.sum(state.column_enthalpy * mesh.physical_cell_areas_km2(
            thermal.radius_km)[:, None] * 1e6) * p.density_kg_m3 * p.column_depth_km * 1000 / p.column_layers
        assert total - state.initial_column_energy_j == pytest.approx(
            state.boundary_energy_j, rel=1e-12)
        diagnostics = diagnose_shell(state, mesh, p, thermal, 700.0, 2300.0)
        assert abs(diagnostics["relative_column_energy_residual"]) < 1e-12
        # Perfectly symmetric cooling has no artificial tensile fracture.
        assert diagnostics["max_tensile_stress_mpa"] < 1e-7
        assert diagnostics["damaged_area_fraction"] == 0


def test_halving_column_timestep_converges_through_solidification(quenched_columns):
    _, _, _, (coarse, fine, reference) = quenched_columns
    coarse_error = np.linalg.norm(coarse.column_enthalpy - reference.column_enthalpy)
    fine_error = np.linalg.norm(fine.column_enthalpy - reference.column_enthalpy)
    assert coarse_error > 1
    assert fine_error < 0.45 * coarse_error
    coarse_depth = np.max(abs(coarse.lid_thickness_km - reference.lid_thickness_km))
    fine_depth = np.max(abs(fine.lid_thickness_km - reference.lid_thickness_km))
    assert fine_depth < 0.45 * coarse_depth
    assert fine_depth < 0.003


def test_maxwell_update_matches_analytic_constant_strain_rate_for_any_partition():
    tau = np.array([1e-6, 0.1, 1.0, 100.0, 1e12])
    duration, modulus, rate, old_stress = 3.7, 6e10, 1e-8, 2e6
    expected = (old_stress * np.exp(-duration / tau)
                + modulus * rate * tau * -np.expm1(-duration / tau))
    for count in (1, 2, 17, 100):
        dt = duration / count
        r, b = maxwell_factors(dt, tau)
        assert np.all((r >= 0) & (r <= 1))
        assert np.all((b > 0) & (b <= 1))
        stress = np.full(tau.shape, old_stress)
        for _ in range(count):
            stress = r * stress + modulus * b * rate * dt
        np.testing.assert_allclose(stress, expected, rtol=2e-12, atol=1e-10)


def test_compression_only_heals_and_hot_material_heals_faster():
    p = ShellParameters()
    compressive = np.array([[-20e6, -30e6, 0.0], [-40e6, -25e6, 2e6]])
    tensile = principal_tensile(compressive)
    np.testing.assert_array_equal(tensile, 0.0)
    cold = evolve_damage(np.array([0.0, 0.8]), tensile, p.tensile_strength_pa,
                         500.0, 0.1, p)
    hot = evolve_damage(np.array([0.0, 0.8]), tensile, p.tensile_strength_pa,
                        1400.0, 0.1, p)
    assert cold[0] == hot[0] == 0
    assert 0 < hot[1] < cold[1] < 0.8
    assert cold[1] == pytest.approx(0.8 * np.exp(-0.1 / p.cold_healing_timescale_myr))


def test_damage_is_bounded_and_exact_for_constant_loading_partitions():
    p = ShellParameters()
    old = np.array([0.0, 0.25, 0.9, 1.0])
    tensile = p.tensile_strength_pa * np.array([0.0, 1.1, 3.0, 100.0])
    temperatures = np.array([400.0, 800.0, 1200.0, 1500.0])
    full = evolve_damage(old, tensile, p.tensile_strength_pa, temperatures, 0.02, p)
    split = old.copy()
    for _ in range(20):
        split = evolve_damage(split, tensile, p.tensile_strength_pa, temperatures, 0.001, p)
    assert np.all((full >= 0) & (full <= 1))
    assert full[2] > old[2]
    np.testing.assert_allclose(split, full, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("strain", [[0.03, -0.04, 0.0], [0.0, 0.0, 0.08],
                                    [0.01, -0.03, 0.04]])
def test_total_strain_limit_uses_principal_strains_and_engineering_shear(strain):
    tensor = np.array([[strain[0], strain[2] / 2], [strain[2] / 2, strain[1]]])
    expected = np.max(np.abs(np.linalg.eigvalsh(tensor)))
    assert maximum_total_strain(np.array([strain])) == pytest.approx(expected)


def test_exceeding_small_strain_limit_stops_before_another_advance(small_shell):
    mesh, membrane, base, thermal = small_shell
    p = replace(base, max_total_strain=1e-6)
    state = initialize_shell(mesh, p, thermal)
    result = advance_shell(state, mesh, membrane, p, thermal, 0.001,
                           700.0, 700.0, 2300.0, 2300.0)
    assert maximum_total_strain(result.strain) > p.max_total_strain
    assert result.stopped_reason == "shell_small_strain_limit"
    assert state.stopped_reason is None
    with pytest.raises(ValueError):
        advance_shell(result, mesh, membrane, p, thermal, 0.002,
                      700.0, 700.0, 2300.0, 2300.0)


def test_timestep_refinement_converges_during_nonlinear_damage():
    thermal = GenesisParameters()
    p = ShellParameters(subdivisions=1, convective_traction_pa=50000.0)
    mesh = build_icosphere(p.subdivisions)
    membrane = Membrane(mesh, p.poisson_ratio)
    results = []
    for dt in (0.002, 0.001, 0.0005):
        global_state = initial_state(thermal)
        state = initialize_shell(mesh, p, thermal)
        mantle, surface = temperatures(np.asarray(global_state.energy), thermal)
        for step in range(1, round(0.96 / dt) + 1):
            old_mantle, old_surface = mantle, surface
            global_state, _ = advance_thermal(global_state, thermal, step * dt, 0.01)
            mantle, surface = temperatures(np.asarray(global_state.energy), thermal)
            state = advance_shell(state, mesh, membrane, p, thermal,
                                  global_state.time_myr, old_surface, surface,
                                  old_mantle, mantle)
            assert state.stopped_reason is None
        assert np.max(state.damage) > p.damage_threshold
        assert state.first_fracture_time_myr is not None
        assert np.all(state.peak_tensile_pa + 1e-6 >= principal_tensile(state.stress_pa))
        results.append(state)
    coarse, fine, reference = results
    for field in ("damage", "stress_pa", "column_enthalpy", "lid_thickness_km"):
        coarse_error = np.linalg.norm(getattr(coarse, field) - getattr(reference, field))
        fine_error = np.linalg.norm(getattr(fine, field) - getattr(reference, field))
        assert fine_error < 0.35 * coarse_error


def test_checkpoint_roundtrip_preserves_all_arrays_scalars_and_metadata(tmp_path, small_shell):
    mesh, membrane, p, thermal = small_shell
    state = initialize_shell(mesh, p, thermal)
    state = advance_shell(state, mesh, membrane, p, thermal, 0.01,
                          700.0, 700.0, 2300.0, 2300.0)
    thermal_state = replace(initial_state(thermal), time_myr=state.time_myr)
    controls = {"shell_step_myr": 0.001, "sample_interval_myr": 0.01}
    provenance = {"case": "roundtrip", "nested": {"source": "test"}}
    path = tmp_path / "shell.npz"
    save_shell_checkpoint(path, state, thermal_state, p, thermal, controls, provenance)
    restored, restored_thermal, restored_p, restored_tp, metadata = load_shell_checkpoint(path)
    assert restored_p == p
    assert restored_tp == thermal
    assert restored_thermal == thermal_state
    assert metadata["controls"] == controls
    assert metadata["provenance"] == provenance
    for field in fields(state):
        expected, actual = getattr(state, field.name), getattr(restored, field.name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected
    assert not path.with_suffix(".npz.tmp").exists()
    # A restored state must reproduce the very next solve, including memory.
    expected_next = advance_shell(state, mesh, membrane, p, thermal, 0.02,
                                  700.0, 700.0, 2300.0, 2300.0)
    actual_next = advance_shell(restored, mesh, membrane, p, thermal, 0.02,
                                700.0, 700.0, 2300.0, 2300.0)
    for field in fields(state):
        expected, actual = getattr(expected_next, field.name), getattr(actual_next, field.name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected


@pytest.mark.parametrize("mutation", [
    "format", "hash", "clock", "array_shape", "array_nan", "damage_bounds",
    "negative_enthalpy", "missing_array", "thermal_nan", "thermal_budget",
    "scalar_nan", "future_fracture",
])
def test_malformed_or_inconsistent_shell_checkpoint_is_rejected(tmp_path, small_shell, mutation):
    mesh, _, p, thermal = small_shell
    path = tmp_path / "invalid.npz"
    state = initialize_shell(mesh, p, thermal)
    save_shell_checkpoint(path, state, initial_state(thermal), p, thermal, {}, {})
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files if key != "metadata"}
        metadata = json.loads(str(archive["metadata"]))
    if mutation == "format":
        metadata["format"] = "unrelated_checkpoint"
    elif mutation == "hash":
        metadata["shell_parameters"]["tensile_strength_pa"] *= 2
    elif mutation == "clock":
        metadata["thermal_state"]["time_myr"] = 1.0
    elif mutation == "array_shape":
        arrays["column_enthalpy"] = arrays["column_enthalpy"][:, :-1]
    elif mutation == "array_nan":
        arrays["stress_pa"][0, 0] = float("nan")
    elif mutation == "damage_bounds":
        arrays["damage"][0] = 1.01
    elif mutation == "negative_enthalpy":
        arrays["column_enthalpy"][0, 0] = -1
    elif mutation == "missing_array":
        del arrays["strain"]
    elif mutation == "thermal_nan":
        metadata["thermal_state"]["energy"][0] = float("nan")
    elif mutation == "thermal_budget":
        metadata["thermal_state"]["energy"][0] *= 0.5
    elif mutation == "scalar_nan":
        metadata["shell_state"]["initial_column_energy_j"] = float("nan")
    elif mutation == "future_fracture":
        metadata["shell_state"]["first_fracture_time_myr"] = 1.0
    np.savez_compressed(path, metadata=np.array(json.dumps(metadata)), **arrays)
    with pytest.raises(ValueError):
        load_shell_checkpoint(path)


@pytest.fixture(scope="module")
def loaded_cli_run(tmp_path_factory):
    from run_genesis_shell import main

    root = tmp_path_factory.mktemp("loaded_shell_cli")
    destination = root / "first"
    controls = ["--sample-interval-myr", "0.1", "--max-step-myr", "0.01",
                "--shell-step-myr", "0.01", "--no-frames"]
    physics = ["--subdivisions", "1", "--convective-traction-mpa", "0.02"]
    assert main(["--output", str(destination), "--duration-myr", "1.0",
                 *controls, *physics]) == 0
    return root, destination, controls, physics


def test_cli_artifacts_and_loaded_checkpoint_resume_match_uninterrupted_run(loaded_cli_run):
    from run_genesis_shell import main

    root, destination, controls, physics = loaded_cli_run
    assert {"parameters.json", "shell_checkpoint.npz", "history.csv",
            "shell_history.csv", "summary.json", "shell_fields.npz",
            "genesis_shell.png", "genesis_history.png"} <= {p.name for p in destination.iterdir()}
    for name in ("genesis_shell.png", "genesis_history.png"):
        assert (destination / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    source = destination / "shell_checkpoint.npz"
    saved_bytes = source.read_bytes()
    source_state, _, _, _, _ = load_shell_checkpoint(source)
    assert np.max(source_state.peak_tensile_pa) > 1e6
    assert np.max(source_state.damage) > 0
    resumed, continuous = root / "resumed", root / "continuous"
    assert main(["--output", str(resumed), "--resume", str(source),
                 "--duration-myr", "1.1", *controls]) == 0
    assert main(["--output", str(continuous), "--duration-myr", "1.1",
                 *controls, *physics]) == 0
    actual, actual_thermal, _, _, metadata = load_shell_checkpoint(resumed / "shell_checkpoint.npz")
    expected, expected_thermal, _, _, _ = load_shell_checkpoint(continuous / "shell_checkpoint.npz")
    assert actual_thermal == expected_thermal
    for field in fields(expected):
        value = getattr(expected, field.name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(actual, field.name), value)
        else:
            assert getattr(actual, field.name) == value
    assert actual.time_myr == pytest.approx(1.1)
    assert metadata["provenance"]["resume_source"] == str(source.resolve())
    assert source.read_bytes() == saved_bytes
    summary = json.loads((resumed / "summary.json").read_text(encoding="utf-8"))
    assert summary["shell"]["time_myr"] == actual.time_myr
    with (resumed / "shell_history.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["time_myr"]) for row in rows] == pytest.approx([1.0, 1.1])
    with np.load(resumed / "shell_fields.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["damage"], actual.damage)
        assert archive["faces"].shape == (80, 3)


def test_cli_refuses_existing_output_without_modifying_results(tmp_path, capsys):
    from run_genesis_shell import main

    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "important.txt"
    sentinel.write_text("keep this", encoding="utf-8")
    assert main(["--output", str(destination), "--duration-myr", "0.02",
                 "--subdivisions", "1", "--no-frames"]) == 1
    assert "existing results are protected" in capsys.readouterr().err
    assert list(destination.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding="utf-8") == "keep this"


@pytest.mark.parametrize("extra", [["--subdivisions", "2"],
                                    ["--convective-traction-mpa", "0.1"],
                                    ["--shell-step-myr", "0.005"]])
def test_cli_refuses_changing_saved_physics_or_time_controls(loaded_cli_run, tmp_path, extra, capsys):
    from run_genesis_shell import main

    _, source, controls, _ = loaded_cli_run
    destination = tmp_path / "invalid_resume"
    assert main(["--resume", str(source / "shell_checkpoint.npz"),
                 "--output", str(destination), "--duration-myr", "1.1",
                 *controls, *extra]) == 1
    assert "Resume" in capsys.readouterr().err
    assert not destination.exists()
