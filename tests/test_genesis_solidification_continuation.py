"""Birth history follows actual cooling, restart and the young-world handover."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.genesis import GenesisParameters
from tectonics.genesis_local_mechanics import refresh_young_material_mechanics
from tectonics.genesis_shell import ShellParameters
from tectonics.genesis_solidification import solidification_history_to_dict
from tectonics.genesis_starter import StarterModel
from tectonics.genesis_starter_continuation import YoungWorldCoupling, build_starter_continuation
from tectonics.genesis_starter_fracture import YoungShellFracture
from tectonics.genesis_starter_material import primary_material_inventory
from tectonics.genesis_tides import SYNCHRONOUS_SPIN, TidalParameters
from tectonics.mesh import build_icosphere
from tectonics.simulation import load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def origin():
    model = StarterModel(build_icosphere(1), GenesisParameters(),
        TidalParameters(enabled=True, spin_state=SYNCHRONOUS_SPIN),
        ShellParameters(subdivisions=1, convective_traction_pa=50000.))
    initial = model.initial_state()
    before = model.advance(initial, .7)
    assert before.stopped_reason is None
    partition = model.advance(before, 2.)
    assert partition.stopped_reason == "first_partition"
    return model, initial, before, partition


def serialized(state):
    return solidification_history_to_dict(state.solidification_history)


def imported(origin):
    model, _, _, source = origin
    config = load_config(ROOT/"configs/canonical_moon.yaml")
    bundle, cfg, report = build_starter_continuation(model, deepcopy(source), config)
    return model, source, bundle, cfg, report


def test_real_molten_birth_has_steam_and_no_solid_material(origin):
    model, initial, _, _ = origin
    row = model.diagnose(initial)
    thermal = model.loading.sample(initial.thermal_context).thermal
    history = initial.solidification_history
    assert thermal["surface_melt_fraction"] == row["mantle_melt_fraction"] == 1.
    assert row["ocean_fraction"] == 0.
    assert thermal["steam_pressure_bar"] > 100.
    assert row["plate_count"] == 0 and row["lid_thickness_km"] == 0.
    assert history.solid_mass_kg == history.formed_mass_kg == 0.
    assert history.mean_birth_time_myr is None and history.mean_age_myr == 0.
    assert not history.cohorts and not history.accepted_intervals


def test_real_solidification_mass_matches_existing_reservoir_and_keeps_initial_unchanged(origin):
    model, initial, _, source = origin
    history = source.solidification_history
    inventory = primary_material_inventory(model, source)
    assert len(history.cohorts) > 1
    assert 0. < history.mean_birth_time_myr < source.time_myr
    assert 0. < history.mean_age_myr < source.time_myr
    assert history.time_myr == source.time_myr
    assert history.accepted_intervals == source.thermal_samples
    assert history.solid_mass_kg == pytest.approx(inventory.primary_mass_kg.sum(), rel=2e-15)
    assert initial.solidification_history.time_myr == 0.
    assert not initial.solidification_history.cohorts


def test_full_birth_history_restarts_exactly_without_duplicate_births(origin, tmp_path):
    model, _, before, whole = origin
    path = tmp_path/"starter.npz"
    model.save_state(path, before)
    restored = model.load_state(path)
    assert serialized(restored) == serialized(before)
    resumed = model.advance(restored, 2.)
    assert serialized(resumed) == serialized(whole)
    np.testing.assert_array_equal(resumed.system.cell_plate, whole.system.cell_plate)
    np.testing.assert_array_equal(resumed.damage, whole.damage)
    assert resumed.events == whole.events
    model.save_state(path, resumed)
    assert serialized(model.load_state(path)) == serialized(resumed)


def test_import_retains_actual_solid_age_and_full_birth_record(origin):
    model, source, bundle, cfg, report = imported(origin)
    history = source.solidification_history
    np.testing.assert_array_equal(bundle.checkpoint.state.crust_age_myr, history.mean_age_myr)
    assert report["initial_primary_mean_birth_time_myr"] == history.mean_birth_time_myr
    assert report["initial_primary_mean_age_myr"] == history.mean_age_myr
    assert bundle.report["source_metadata"]["solidification_history"] == serialized(source)
    assert cfg["young_shell"]["primary_material_age"] == "retained_mass_weighted_solidification_age"
    inventory = primary_material_inventory(model, source)
    np.testing.assert_allclose(bundle.checkpoint.state.oceanic_volume_km3,
        inventory.primary_mass_kg/(inventory.primary_density_kg_m3*1e9), rtol=2e-15)
    np.testing.assert_array_equal(bundle.checkpoint.state.continental_volume_km3, 0.)
    assert not bundle.report["physical_handoff_certified"]


def test_nonzero_birth_age_preserves_primordial_column_and_does_not_age_new_crust(origin):
    model, source, bundle, _, _ = imported(origin)
    material = deepcopy(bundle.checkpoint.state)
    elapsed = .2
    material.time_myr += elapsed
    material.crust_age_myr += elapsed
    material.crust_age_myr[0] = .05
    context, _ = model.loading.advance(source.thermal_context, material.time_myr)
    sample = model.loading.sample(context)
    diagnostic = refresh_young_material_mechanics(material, sample, model,
        origin_time_myr=source.time_myr)
    assert diagnostic["primordial_cell_count"] == model.mesh.cell_count-1
    assert diagnostic["rejuvenated_cell_count"] == 1
    np.testing.assert_array_equal(diagnostic["local_total_lid_thickness_km"][1:], sample.lid_thickness_km)
    assert diagnostic["local_total_lid_thickness_km"][0] <= sample.lid_thickness_km


def test_joint_heat_continues_birth_clock_and_keeps_born_cohorts(origin, tmp_path):
    model, source, bundle, cfg, _ = imported(origin)
    coupling = YoungWorldCoupling(model, source, cfg, bundle.checkpoint)
    old = source.solidification_history
    thermal, _ = coupling.advance_heat(bundle.checkpoint.thermal, .1)
    history = coupling.source_state.solidification_history
    assert history.time_myr == thermal.time_myr
    assert history.cohorts == old.cohorts
    assert history.mean_age_myr == pytest.approx(old.mean_age_myr+.1)
    assert history.accepted_intervals > old.accepted_intervals
    assert source.solidification_history == old
    path = tmp_path/"continued_starter.npz"
    model.save_state(path, coupling.source_state)
    assert serialized(model.load_state(path)) == serialized(coupling.source_state)


@pytest.mark.parametrize("reject_stage", ["fracture", "projection"])
def test_failed_joint_step_does_not_commit_birth_intervals(origin, monkeypatch, reject_stage):
    model, source, bundle, cfg, _ = imported(origin)
    coupling = YoungWorldCoupling(model, source, cfg, bundle.checkpoint)
    before = serialized(coupling.source_state)
    before_context = deepcopy(coupling.source_state.thermal_context)

    def reject(*args, **kwargs):
        raise ValueError("controlled step rejection")

    if reject_stage == "fracture":
        monkeypatch.setattr(YoungShellFracture, "advance", reject)
    else:
        monkeypatch.setattr("tectonics.genesis_starter_continuation.project_thermal", reject)
    with pytest.raises(ValueError, match="step rejection"):
        coupling.advance_heat(bundle.checkpoint.thermal, .1)
    assert serialized(coupling.source_state) == before
    assert coupling.source_state.thermal_context.thermal == before_context.thermal
    np.testing.assert_array_equal(coupling.source_state.thermal_context.column_enthalpy,
                                  before_context.column_enthalpy)


@pytest.mark.parametrize("corruption", ["legacy", "missing", "clock"])
def test_missing_or_incompatible_history_is_refused_without_replacement(origin, tmp_path, corruption):
    model, _, before, _ = origin
    path = tmp_path/"corrupt.npz"
    model.save_state(path, before)
    with np.load(path, allow_pickle=False) as archive:
        contents = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(str(contents["metadata"]))
    if corruption == "legacy":
        metadata["version"] = "genesis-starter-0.1"
    elif corruption == "missing":
        del metadata["solidification_history"]
    else:
        metadata["solidification_history"]["time_myr"] += .1
    contents["metadata"] = np.array(json.dumps(metadata))
    np.savez_compressed(path, **contents)
    with pytest.raises(ValueError, match="(?i)birth|solidification|clock|time"):
        model.load_state(path)


def test_handover_without_birth_history_is_refused(origin):
    model, _, _, source = origin
    source = deepcopy(source)
    source.solidification_history = None
    with pytest.raises(ValueError, match="restart.*birth"):
        build_starter_continuation(model, source, load_config(ROOT/"configs/canonical_moon.yaml"))
