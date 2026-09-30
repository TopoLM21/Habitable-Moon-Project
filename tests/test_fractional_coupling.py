"""Independent heat, material and restart checks for parcel/basal integration."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import numpy as np
import pytest

from run_fractional_transport_probe import make_birth_factory
from tectonics.fractional_coupling import (
    EXTENSIVE, advance_coupling, initialize_coupling, ledger_diagnostics,
    load_coupled_checkpoint, save_coupled_checkpoint, thermal_context_to_json,
)
from tectonics.fractional_surface import split_parcel, state_to_json
from tectonics.fractional_surface_io import surface_from_lithosphere, save_fractional_checkpoint
from tectonics.genesis import GenesisParameters
from tectonics.genesis_shell import ShellParameters
from tectonics.genesis_starter import StarterModel
from tectonics.genesis_starter_continuation import build_starter_continuation
from tectonics.genesis_starter_fracture import YoungShellFracture
from tectonics.genesis_tides import TidalParameters
from tectonics.mesh import build_icosphere
from tectonics.simulation import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def source():
    model = StarterModel(build_icosphere(1), GenesisParameters(), TidalParameters(enabled=False),
        ShellParameters(subdivisions=1, convective_traction_pa=50000.))
    archived = model.advance(model.initial_state(), 2.)
    assert archived.stopped_reason == "first_partition"
    config = load_config(ROOT / "configs/canonical_moon.yaml")
    config["young_shell"] = {"mechanics_model_version": "young-mechanics-0.5"}
    bundle, _, imported = build_starter_continuation(model, archived, config)
    cp = bundle.checkpoint
    fracture = YoungShellFracture(model, archived)
    surface = surface_from_lithosphere(model.mesh, cp.state, model.thermal.radius_km,
        fracture_memory=fracture.memory)
    provenance = dict(source_time_myr=surface.time_myr, origin_time_myr=imported["origin_time_myr"],
        radius_km=model.thermal.radius_km, reference_density_kg_m3=model.thermal.surface_layer_density_kg_m3,
        newborn_crust_thickness_km=imported["initial_primary_thickness_km"],
        local_cooling_diffusivity_m2_s=float(config["mechanical_lithosphere"]["thermal_diffusivity_m2_s"]),
        initial_reference_mantle_mass_kg=imported["initial_mantle_material_mass_kg"])
    state = initialize_coupling(model.mesh, surface, model, archived.thermal_context,
        provenance, len(cp.system.plates))
    return model, archived, surface, state


def factory(source):
    model, _, surface, state = source
    return make_birth_factory(surface, model, state.provenance)


def snapshot(state):
    value = asdict(state)
    value["surface"] = state_to_json(state.surface)
    value["thermal_context"] = thermal_context_to_json(state.thermal_context)
    return json.loads(json.dumps(value))


def advance(source, state, dt=.25):
    return advance_coupling(source[0].mesh, state, source[0], dt, birth_factory=factory(source))


def test_coupled_thermal_context_matches_independent_existing_loading_owner(source):
    model, archived, _, state = source
    old_source = thermal_context_to_json(archived.thermal_context)
    old_state = snapshot(state)
    expected, _ = model.loading.advance(state.thermal_context, state.surface.time_myr+.25,
        max_sample_myr=model.parameters.max_loading_interval_myr)
    actual = advance(source, state)
    assert thermal_context_to_json(actual.thermal_context) == thermal_context_to_json(expected)
    assert snapshot(state) == old_state
    assert thermal_context_to_json(archived.thermal_context) == old_source
    assert actual.history[-1]["heat"]["additional_global_heat_sink_j"] == 0.
    assert actual.history[-1]["cooling"]["additional_global_heat_sink_j"] == 0.
    assert actual.surface.time_myr == actual.thermal_context.thermal.time_myr == actual.thermal_context.orbit.time_myr


def test_restart_reproduces_all_surface_heat_archive_and_history_bits(source, tmp_path):
    model, _, _, state = source
    first = advance(source, state)
    direct = advance(source, first)
    path = tmp_path / "coupled.json"
    save_coupled_checkpoint(path, model.mesh, first)
    loaded = load_coupled_checkpoint(path, model.mesh, model, state.provenance)
    assert snapshot(loaded) == snapshot(first)
    resumed = advance(source, loaded)
    assert snapshot(resumed) == snapshot(direct)


def test_independent_chemical_and_cooling_ledgers_close_after_real_transport(source):
    _, _, _, state = source
    first = advance(source, state)
    final = advance(source, first)
    report = ledger_diagnostics(final)
    assert final.removed_material
    assert final.cumulative_births["oceanic_volume_km3"] > 0.
    for key in EXTENSIVE:
        retained = math.fsum(getattr(p, key) for p in final.surface.parcels)
        archived = math.fsum(getattr(loss.parcel, key) for loss in final.removed_material)
        assert archived == pytest.approx(final.cumulative_losses[key], rel=5e-14)
        actual_change = retained+archived-final.initial_totals[key]
        processes = final.cumulative_births[key]+final.cumulative_thermal_sources[key]
        scale = max(retained, archived, abs(processes), 1.)
        assert abs(actual_change-processes)/scale < 5e-14
    density = state.provenance["reference_density_kg_m3"]
    assert report["reference_mantle_mass_change_kg"] == -final.cumulative_births["oceanic_volume_km3"]*density*1e9
    assert abs(report["chemical_reference_mass_relative_residual"]) < 5e-14
    assert final.cumulative_thermal_sources["area_km2"] == 0.
    assert final.cumulative_thermal_sources["oceanic_volume_km3"] == 0.


def test_material_damage_and_distinct_source_histories_survive_dynamic_step(source):
    _, _, initial_surface, initial = source
    state = advance(source, initial)
    fields = {p.material_id: p.material_fields for p in initial_surface.parcels}
    for parcel in state.surface.parcels:
        if parcel.material_id in fields:
            assert parcel.material_fields == fields[parcel.material_id]
            assert parcel.age_myr == .25
    for loss in state.removed_material:
        if loss.parcel.material_id in fields:
            assert loss.parcel.material_fields == fields[loss.parcel.material_id]
    assert state.history[-1]["driving"]["model"] == "fractional_prescribed_basal_v1"
    assert state.history[-1]["endpoint_dynamics"]["torque_relative_residual"] < 1e-11
    assert state.history[-1]["endpoint_dynamics"]["power_relative_residual"] < 1e-11


@pytest.mark.parametrize("dt", [0., -1., float("nan"), float("inf")])
def test_invalid_timestep_rejected_without_mutating_source(source, dt):
    state = source[-1]
    before = snapshot(state)
    with pytest.raises(ValueError, match="time step"):
        advance(source, state, dt)
    assert snapshot(state) == before


def test_rejected_thermal_interval_does_not_publish_material_or_history(source, monkeypatch):
    model, _, _, state = source
    before = snapshot(state)
    original = model.loading.advance
    def stopped(*args, **kwargs):
        context, samples = original(*args, **kwargs)
        thermal = deepcopy(context.thermal)
        thermal.stopped_reason = "test thermal boundary"
        return replace(context, thermal=thermal), samples
    monkeypatch.setattr(model.loading, "advance", stopped)
    with pytest.raises(ValueError, match="Genesis stopped"):
        advance(source, state)
    assert snapshot(state) == before


def test_fractional_transport_only_save_is_not_silently_resumed_as_coupled(source, tmp_path):
    model, _, _, state = source
    path = tmp_path / "transport.json"
    save_fractional_checkpoint(path, model.mesh, state.surface, state.provenance["radius_km"],
        provenance={"experiment": state.provenance})
    with pytest.raises(ValueError, match="heat/basal coupling checkpoint"):
        load_coupled_checkpoint(path, model.mesh, model, state.provenance)


def test_changed_provenance_and_corrupt_checkpoint_are_rejected(source, tmp_path):
    model, _, _, state = source
    path = tmp_path / "coupled.json"
    save_coupled_checkpoint(path, model.mesh, state)
    other = dict(state.provenance, reference_density_kg_m3=2900.)
    with pytest.raises(ValueError, match="identical source"):
        load_coupled_checkpoint(path, model.mesh, model, other)
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["provenance"]["thermal_context"]["thermal"]["energy"][0] *= .9
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_coupled_checkpoint(path, model.mesh, model, state.provenance)


def test_checkpoint_never_overwrites_an_existing_artifact(source, tmp_path):
    model, _, _, state = source
    path = tmp_path / "protected.json"
    path.write_text("protected", encoding="utf-8")
    with pytest.raises(FileExistsError):
        save_coupled_checkpoint(path, model.mesh, state)
    assert path.read_text(encoding="utf-8") == "protected"


def test_clock_mismatch_refused_before_coupled_initialization(source):
    model, archived, surface, state = source
    wrong = replace(surface, time_myr=surface.time_myr+.25)
    with pytest.raises(ValueError, match="clocks disagree"):
        initialize_coupling(model.mesh, wrong, model, archived.thermal_context,
            state.provenance, state.plate_count)


def test_newborn_material_cannot_debit_an_exhausted_reference_mantle(source):
    model, archived, surface, state = source
    tiny = dict(state.provenance, initial_reference_mantle_mass_kg=1.)
    bounded = initialize_coupling(model.mesh, surface, model, archived.thermal_context,
        tiny, state.plate_count)
    before = snapshot(bounded)
    with pytest.raises(ValueError, match="exhausts the reference mantle"):
        advance(source, bounded)
    assert snapshot(bounded) == before


def test_explicit_nondefault_diffusivity_changes_only_rejuvenated_cooling(source):
    model, _, _, initial = source
    mature = advance(source, advance(source, initial, .5), .5)
    # Give one cell an explicit sizable young component rather than relying
    # on this very young Starter's tiny dynamically generated ridge area.
    index = next(i for i, p in enumerate(mature.surface.parcels) if p.material_id.startswith("source:"))
    half = split_parcel(mature.surface.parcels[index], .5)
    younger = replace(half, material_id="test:young", age_myr=.01)
    mixed = replace(mature.surface, parcels=mature.surface.parcels[:index]+(half, younger)+mature.surface.parcels[index+1:],
        known_material_ids=mature.surface.known_material_ids+("test:young",))
    reference = initialize_coupling(model.mesh, mixed, model, mature.thermal_context,
        mature.provenance, mature.plate_count)
    changed = dict(mature.provenance, local_cooling_diffusivity_m2_s=1e-12)
    recomputed = initialize_coupling(model.mesh, mixed, model, mature.thermal_context,
        changed, mature.plate_count)
    for before, after in zip(reference.surface.parcels, recomputed.surface.parcels):
        assert before.area_km2 == after.area_km2
        assert before.oceanic_volume_km3 == after.oceanic_volume_km3
        if before.material_id.startswith("source:"):
            assert before == after
    # Total cold lid can still lie inside chemical crust, so a mechanical
    # change must be visible through driving traction even if cold mantle=0.
    assert not np.allclose(reference.initial_diagnostics["dynamics"]["omega_rad_per_myr"],
        recomputed.initial_diagnostics["dynamics"]["omega_rad_per_myr"], rtol=1e-9, atol=0.)
    assert thermal_context_to_json(mature.thermal_context) == thermal_context_to_json(recomputed.thermal_context)


def test_rehashed_but_inconsistent_removed_material_archive_is_rejected(source, tmp_path):
    from tectonics.fractional_surface_io import _payload_digest
    model, _, _, state = source
    state = advance(source, state)
    assert state.removed_material
    path = tmp_path / "coupled.json"
    save_coupled_checkpoint(path, model.mesh, state)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["provenance"]["removed_material"] = []
    data.pop("payload_sha256")
    data["payload_sha256"] = _payload_digest(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="removed material archive"):
        load_coupled_checkpoint(path, model.mesh, model, state.provenance)


@pytest.fixture
def saved_source(source, tmp_path):
    model, archived, _, _ = source
    folder = tmp_path / "source"
    folder.mkdir()
    path = folder / "starter_checkpoint.npz"
    model.save_state(path, archived)
    (folder / "parameters.json").write_text(json.dumps(
        dict(format="genesis-starter-run-0.1", **model.configuration)), encoding="utf-8")
    return path


def test_public_experiment_restart_is_identical_and_keeps_real_source_files(saved_source, tmp_path):
    import run_fractional_coupling_probe as runner
    files = [saved_source, saved_source.parent / "parameters.json"]
    before = {path: path.read_bytes() for path in files}
    full = runner.execute_probe(saved_source, tmp_path / "full", .5, .25)
    runner.execute_probe(saved_source, tmp_path / "first", .25, .25)
    resumed = runner.execute_probe(saved_source, tmp_path / "second", .25, .25,
        resume=tmp_path / "first/fractional_checkpoint.json")
    assert full["history"] == resumed["history"]
    assert full["checkpoint_sha256"] == resumed["checkpoint_sha256"]
    assert full["source_unchanged"] and resumed["source_unchanged"]
    assert all(path.read_bytes() == content for path, content in before.items())


def test_failed_public_interval_never_publishes_partial_output(saved_source, tmp_path, monkeypatch):
    import run_fractional_coupling_probe as runner
    original = runner.advance_coupling
    calls = []
    def stop_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("test next interval unsupported")
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, "advance_coupling", stop_second)
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="next interval unsupported"):
        runner.execute_probe(saved_source, output, .5, .25)
    assert len(calls) == 2
    assert not output.exists()


def test_public_runner_does_not_overwrite_output(saved_source, tmp_path):
    import run_fractional_coupling_probe as runner
    output = tmp_path / "old"
    output.mkdir()
    marker = output / "report.json"
    marker.write_text("protected", encoding="utf-8")
    with pytest.raises(ValueError, match="never overwritten"):
        runner.execute_probe(saved_source, output, .25, .25)
    assert marker.read_text(encoding="utf-8") == "protected"
