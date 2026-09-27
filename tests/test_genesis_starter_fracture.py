"""Persistent loading, real donor advection and non-replayed young fractures."""
from copy import deepcopy
import json

import numpy as np
import pytest

from tectonics.genesis import GenesisParameters
from tectonics.genesis_shell import ShellParameters
from tectonics.genesis_starter import StarterModel, StarterParameters
from tectonics.genesis_starter_fracture import YoungShellFracture, FIELDS
from tectonics.genesis_starter_topology import select_starter_cut
from tectonics.genesis_tides import TidalParameters, SYNCHRONOUS_SPIN
from tectonics.kinematics import angular_velocity_vectors
from tectonics.mesh import build_icosphere, connected_components


def make_model(subdivisions=2, traction=50000., **parameters):
    return StarterModel(build_icosphere(subdivisions), GenesisParameters(),
        TidalParameters(enabled=True, spin_state=SYNCHRONOUS_SPIN),
        ShellParameters(subdivisions=subdivisions, convective_traction_pa=traction),
        StarterParameters(**parameters))


@pytest.fixture(scope="module")
def real_source():
    model = make_model()
    source = model.advance(model.initial_state(), 1.)
    assert source.stopped_reason == "first_partition"
    return model, source


@pytest.fixture
def geometry():
    model = make_model(3)
    source = model.initial_state()
    source.system.plates[0].angular_speed_rad_per_myr = np.deg2rad(.25)
    return model, source, YoungShellFracture(model, source)


def assert_equal(a, b):
    assert a.time_myr == b.time_myr
    for name in (*FIELDS, "eligible", "consumed_band"):
        np.testing.assert_array_equal(getattr(a.memory, name), getattr(b.memory, name))
    assert a.memory.first_fracture_time_myr == b.memory.first_fracture_time_myr
    assert a.memory.tidal_peak_mpa == b.memory.tidal_peak_mpa
    assert json.dumps(a.events, sort_keys=True) == json.dumps(b.events, sort_keys=True)


def test_continued_material_law_exactly_matches_shared_starter_samples(real_source):
    model, source = real_source
    source_before = deepcopy(source)
    fracture = YoungShellFracture(model, source)
    before = model.loading.sample(source.thermal_context)
    _, samples = model.loading.advance(source.thermal_context, source.time_myr+.2,
        max_sample_myr=model.parameters.max_loading_interval_myr)
    expected = deepcopy(source)
    previous = before
    for sample in samples:
        model._material_sample(expected, previous, sample)
        previous = sample
    fracture.advance(before, samples)
    assert fracture.time_myr == samples[-1].time_myr
    for name in (*FIELDS, "eligible"):
        np.testing.assert_array_equal(getattr(fracture.memory, name), getattr(expected, name))
        np.testing.assert_array_equal(getattr(source, name), getattr(source_before, name))
    assert source.time_myr == source_before.time_myr
    assert not np.array_equal(fracture.damage, source.damage)


def test_bad_sample_clock_rolls_back_all_material_memory(real_source):
    model, source = real_source
    fracture = YoungShellFracture(model, source)
    saved = deepcopy(fracture)
    before = model.loading.sample(source.thermal_context)
    _, samples = model.loading.advance(source.thermal_context, source.time_myr+.1)
    with pytest.raises(ValueError, match="monotonically"):
        fracture.advance(before, [samples[0], samples[0]])
    assert_equal(fracture, saved)
    assert saved.model is model


def test_material_donor_moves_all_memory_and_resets_new_material(real_source):
    model, source = real_source
    fracture = YoungShellFracture(model, source)
    before = deepcopy(fracture)
    n = model.mesh.cell_count
    donors = np.roll(np.arange(n), 11)
    donors[0] = -1
    fracture.transport(donors)
    for name in (*FIELDS, "consumed_band"):
        np.testing.assert_array_equal(getattr(fracture.memory, name)[1:],
            getattr(before.memory, name)[donors[1:]])
    for name in ("damage", "cooling_stress_pa", "water_access", "yield_ratio"):
        assert getattr(fracture.memory, name)[0] == 0.
    assert fracture.memory.strength_pa[0] == model.shell.tensile_strength_pa*model.strength_factor[0]
    assert not fracture.memory.consumed_band[0] and not fracture.memory.eligible[0]
    assert fracture.time_myr == before.time_myr


@pytest.mark.parametrize("kind", ["shape", "float", "low", "high"])
def test_bad_donor_map_is_rejected_without_mutation(real_source, kind):
    model, source = real_source
    fracture = YoungShellFracture(model, source)
    saved = deepcopy(fracture)
    donors = np.arange(model.mesh.cell_count)
    if kind == "shape":
        donors = donors[:-1]
    elif kind == "float":
        donors = donors.astype(float)
    else:
        donors[0] = -2 if kind == "low" else model.mesh.cell_count
    with pytest.raises(ValueError, match="donor"):
        fracture.transport(donors)
    assert_equal(fracture, saved)


def test_global_all_oceanic_shell_can_split_without_age_extension_or_velocity_kick(geometry):
    model, source, fracture = geometry
    weak = np.abs(model.mesh.centroids[:, 0]) < .12
    fracture.set_damage(.8*weak)
    original = deepcopy(source.system)
    result, event = fracture.attempt(source.system)
    assert result is not None and len(result.plates) == 2
    assert event.kind == "split" and event.time_myr == source.time_myr
    np.testing.assert_array_equal(source.system.cell_plate, original.cell_plate)
    np.testing.assert_allclose(angular_velocity_vectors(result),
        np.repeat(angular_velocity_vectors(original), 2, axis=0), rtol=0., atol=1e-18)
    for pid in range(2):
        cells = np.flatnonzero(result.cell_plate == pid)
        assert len(connected_components(cells, model.mesh.neighbors)) == 1
        assert model.areas[cells].sum() >= model.parameters.min_child_area_km2
    assert fracture.diagnose()["continued_split_count"] == 1
    np.testing.assert_array_equal(fracture.memory.consumed_band, weak)
    # A hypothetical later merger cannot replay the unchanged old damage map.
    assert fracture.attempt(original) == (None, None)


def test_new_cross_plate_cut_uses_old_boundary_as_anchor_without_replaying_it(geometry):
    model, source, fracture = geometry
    old_band = np.abs(model.mesh.centroids[:, 0]) < .12
    new_band = np.abs(model.mesh.centroids[:, 2]) < .12
    fracture.set_damage(.8*old_band)
    divided, _ = fracture.attempt(source.system)
    fracture.set_damage(.8*(old_band | new_band))
    p = model.parameters
    # Blanket exclusion leaves a coherent rim and blocks the next cut.
    assert select_starter_cut(model.mesh, divided, new_band & ~old_band,
        fracture.damage, model.thermal.radius_km,
        p.min_child_area_km2, p.min_band_span_km) is None
    result, event = fracture.attempt(divided)
    assert result is not None and len(result.plates) == 3
    for child in event.children:
        np.testing.assert_allclose(angular_velocity_vectors(result)[child],
            angular_velocity_vectors(divided)[event.parents[0]], rtol=0., atol=1e-18)
    assert len(fracture.events) == 2


def test_used_band_rearms_only_after_healing_and_new_loading(geometry):
    model, source, fracture = geometry
    weak = np.abs(model.mesh.centroids[:, 0]) < .12
    fracture.set_damage(.8*weak)
    fracture.attempt(source.system)
    fracture.set_damage(.5*weak)
    fracture.set_damage(.8*weak)
    assert fracture.attempt(source.system) == (None, None)
    fracture.set_damage(.2*weak)
    assert not np.any(fracture.memory.consumed_band)
    assert fracture.attempt(source.system) == (None, None)
    fracture.set_damage(.8*weak)
    result, _ = fracture.attempt(source.system)
    assert result is not None and len(fracture.events) == 2


@pytest.mark.parametrize("kind", ["none", "everywhere", "isolated"])
def test_damage_without_coherent_surviving_blocks_does_not_force_plates(geometry, kind):
    model, source, fracture = geometry
    weak = {"none": np.zeros(model.mesh.cell_count, bool),
        "everywhere": np.ones(model.mesh.cell_count, bool),
        "isolated": model.mesh.centroids[:, 0] > .9}[kind]
    fracture.set_damage(.9*weak)
    assert fracture.attempt(source.system) == (None, None)
    assert not fracture.events


def test_checkpoint_resume_retains_exact_healing_loading_and_latch(real_source, tmp_path):
    model, source = real_source
    fracture = YoungShellFracture(model, source)
    path = tmp_path/"fracture.npz"
    fracture.save(path)
    restored = YoungShellFracture.load(model, path)
    assert_equal(fracture, restored)
    before = model.loading.sample(source.thermal_context)
    _, samples = model.loading.advance(source.thermal_context, source.time_myr+.3)
    fracture.advance(before, samples)
    restored.advance(before, samples)
    assert_equal(fracture, restored)
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize("kind", ["nan", "range", "shape", "eligibility", "clock", "fingerprint"])
def test_corrupt_checkpoint_is_rejected(real_source, tmp_path, kind):
    model, source = real_source
    path = tmp_path/"fracture.npz"
    YoungShellFracture(model, source).save(path)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    if kind in ("nan", "range"):
        arrays["damage"][0] = np.nan if kind == "nan" else 1.1
    elif kind == "shape":
        arrays["water_access"] = arrays["water_access"][:-1]
    elif kind == "eligibility":
        arrays["eligible"][0] = ~arrays["eligible"][0]
    else:
        meta = json.loads(str(arrays["metadata"]))
        meta["time_myr" if kind == "clock" else "fingerprint"] = -1. if kind == "clock" else "other"
        arrays["metadata"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        YoungShellFracture.load(model, path)


def test_event_cannot_bypass_accepted_physical_time(geometry):
    _, source, fracture = geometry
    with pytest.raises(ValueError, match="clock"):
        fracture.attempt(source.system, time_myr=1.)
