"""Whole paired-state refinement, physical conservation and honest rejection."""
from copy import copy, deepcopy
from dataclasses import fields, is_dataclass
import json
from pathlib import Path

import numpy as np
import pytest

from tectonics.checkpoint import load_checkpoint, save_checkpoint
from tectonics.genesis import GenesisParameters
from tectonics.genesis_continuation_remesh import CELL_FIELDS, VOLUME_FIELDS, remesh_continuation_state
from tectonics.genesis_shell import ShellParameters
from tectonics.genesis_starter import StarterModel, StarterParameters
from tectonics.genesis_starter_continuation import build_starter_continuation
from tectonics.genesis_starter_fracture import YoungShellFracture
from tectonics.genesis_starter_loading import StarterLoadingModel
from tectonics.genesis_tides import TidalParameters, SYNCHRONOUS_SPIN
from tectonics.mesh import build_icosphere, connected_components
from tectonics.simulation import load_config
from tectonics.subduction_memory import SlabZone
from tectonics.topology import PlateTopologyManager


def equal(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if is_dataclass(a):
        return type(a) is type(b) and all(equal(getattr(a, f.name), getattr(b, f.name)) for f in fields(a))
    if isinstance(a, dict):
        if a.get("kind") == "split":
            return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


@pytest.fixture(scope="module")
def real_source():
    model = StarterModel(build_icosphere(1), GenesisParameters(),
        TidalParameters(enabled=True, spin_state=SYNCHRONOUS_SPIN),
        ShellParameters(subdivisions=1, convective_traction_pa=50000.), StarterParameters())
    state = model.advance(model.initial_state(), 2.)
    assert state.stopped_reason == "first_partition"
    config = load_config(Path(__file__).resolve().parents[1] / "configs/canonical_moon.yaml")
    bundle, config, _ = build_starter_continuation(model, state, config)
    return model, state, YoungShellFracture(model, state), bundle.checkpoint, config


def source_copy(real_source):
    model, state, fracture, checkpoint, config = real_source
    return model, deepcopy(state), deepcopy(fracture), deepcopy(checkpoint), deepcopy(config)


def test_real_refinement_preserves_all_clocks_damage_and_material_without_new_plates(real_source):
    model, state, fracture, cp, cfg = source_copy(real_source)
    before = deepcopy((state, fracture.memory, cp, cfg))
    result = remesh_continuation_state(model, state, fracture, cp, cfg, 3)
    assert result.model.mesh.cell_count == 1280
    assert result.config["mesh"]["subdivisions"] == 3
    assert result.model.fingerprint != model.fingerprint
    assert result.fracture.model is result.model
    assert result.starter_state.time_myr == state.time_myr == result.checkpoint.state.time_myr
    assert result.fracture.time_myr == fracture.time_myr
    assert equal(result.starter_state.thermal_context, state.thermal_context)
    assert equal(result.checkpoint.thermal, cp.thermal)
    assert equal(result.checkpoint.hydrosphere, cp.hydrosphere)
    assert equal(before, (state, fracture.memory, cp, cfg))
    np.testing.assert_array_equal(result.checkpoint.state.cell_plate, np.repeat(cp.state.cell_plate, 16))
    np.testing.assert_array_equal(result.fracture.damage, result.checkpoint.state.tidal_damage)
    np.testing.assert_array_equal(result.checkpoint.state.continental_volume_km3, 0.)
    for row in result.report["volume_conservation"].values():
        assert abs(row["relative_total_residual"]) < 1e-14
    for pid, plate in enumerate(result.checkpoint.system.plates):
        cells = np.flatnonzero(result.checkpoint.state.cell_plate == pid)
        assert len(connected_components(cells, result.model.mesh.neighbors)) == 1
        assert plate.seed_cell // 16 == cp.system.plates[pid].seed_cell
        assert plate.angular_speed_rad_per_myr == cp.system.plates[pid].angular_speed_rad_per_myr
        np.testing.assert_array_equal(plate.euler_axis, cp.system.plates[pid].euler_axis)


def test_every_grid_field_and_nonzero_volume_is_remapped_by_its_physical_kind(real_source):
    model, state, fracture, cp, cfg = source_copy(real_source)
    n = model.mesh.cell_count
    for name, attributes in CELL_FIELDS.items():
        record = getattr(cp, name)
        assert record is not None, name
        for index, attribute in enumerate(attributes):
            value = getattr(record, attribute)
            if value is None or attribute in ("cell_plate", "crust_type", "tidal_damage"):
                continue
            shape = value.shape
            setattr(record, attribute, np.arange(np.prod(shape), dtype=float).reshape(shape) / (n + 1) + .1 + index)
    result = remesh_continuation_state(model, state, fracture, cp, cfg, 2)
    fine = result.checkpoint
    for name, attributes in CELL_FIELDS.items():
        for attribute in attributes:
            old = getattr(getattr(cp, name), attribute)
            new = getattr(getattr(fine, name), attribute)
            if old is None:
                assert new is None
            elif attribute in VOLUME_FIELDS.get(name, ()):
                np.testing.assert_allclose(new.reshape(n, 4).sum(axis=1), old, rtol=5e-15, atol=0.)
                source_density = old / model.areas
                target_density = new / result.model.areas
                np.testing.assert_allclose(target_density, np.repeat(source_density, 4), rtol=2e-14)
            else:
                np.testing.assert_array_equal(new, np.repeat(old, 4, axis=0))


def test_slab_transport_population_histories_and_collision_seams_survive(real_source):
    model, state, fracture, cp, cfg = source_copy(real_source)
    # A populated plume array whose length happens to equal the number of
    # cells must remain a plume population, not be multiplied with the mesh.
    n = model.mesh.cell_count
    cp.plume_state.centers_unit = np.tile([1., 0., 0.], (n, 1))
    cp.plume_state.ages_myr = np.arange(n, dtype=float)
    cp.plume_state.plume_ids = np.arange(n, dtype=np.int64)
    cp.plume_state.next_plume_id = n
    cp.plume_state.next_birth_time_myr = 177.
    cp.transport_state.residual_quaternions[:] = [np.cos(.03), 0., np.sin(.03), 0.]
    cp.transport_state.hold_age_myr[:] = [3., 5.]
    cp.transport_state.cumulative_commit_count = 7
    cp.subduction_memory.zones[(0, 1)] = SlabZone(0, 1, slab_length_km=271., slab_depth_km=173.,
        active_age_myr=13., cumulative_subducted_area_km2=3e6,
        trench_midpoint=np.array([0., 1., 0.]), torque_axis=np.array([1., 0., 0.]))
    pair_edges = [(a, b) for a, b, _, _ in model.mesh.shared_edges if cp.system.cell_plate[a] != cp.system.cell_plate[b]]
    source_faces = tuple(sorted({face for edge in pair_edges for face in edge}))
    cp.manager.collision_contact_faces[(0, 1)] = source_faces
    cp.manager.collision_age_myr[(0, 1)] = 8.5
    cp.manager.quiet_weld_age_myr[(0, 1)] = 1.25
    cp.manager.small_plate_age_myr[0] = 2.
    cp.lithosphere_rows.append({"historical_source_cell_count": n, "time_myr": .4})
    result = remesh_continuation_state(model, state, fracture, cp, cfg, 2)
    fine = result.checkpoint
    assert equal(fine.subduction_memory, cp.subduction_memory)
    assert equal(fine.transport_state, cp.transport_state)
    assert equal(fine.lithosphere_rows, cp.lithosphere_rows)
    assert fine.manager.collision_age_myr == cp.manager.collision_age_myr
    assert fine.manager.quiet_weld_age_myr == cp.manager.quiet_weld_age_myr
    assert fine.manager.small_plate_age_myr == cp.manager.small_plate_age_myr
    for name in ("centers_unit", "ages_myr", "plume_ids"):
        np.testing.assert_array_equal(getattr(fine.plume_state, name), getattr(cp.plume_state, name))
    expected = {face for a, b, _, _ in result.model.mesh.shared_edges
                if fine.system.cell_plate[a] != fine.system.cell_plate[b] for face in (a, b)}
    assert set(fine.manager.collision_contact_faces[(0, 1)]) == expected
    # Only boundary children inherit the contact, not every child in the
    # coarse adjacent cell; it remains a seam rather than a broad area.
    assert len(expected) < len(source_faces) * 4


def test_live_fracture_and_archived_starter_memory_remain_distinct_and_reloadable(real_source, tmp_path):
    model, state, fracture, cp, cfg = source_copy(real_source)
    fracture.memory.cooling_stress_pa[:] = np.linspace(-2e6, 2e6, model.mesh.cell_count)
    fracture.memory.water_access[:] = .37
    fracture.memory.consumed_band[::3] = True
    fracture.events = [{"time_myr": fracture.time_myr, "kind": "split", "parents": [0], "children": [0, 1]}]
    cp.state.crust_age_myr[:] = np.linspace(0., 4., model.mesh.cell_count)
    result = remesh_continuation_state(model, state, fracture, cp, cfg, 2)
    np.testing.assert_array_equal(result.fracture.memory.cooling_stress_pa, np.repeat(fracture.memory.cooling_stress_pa, 4))
    np.testing.assert_array_equal(result.starter_state.cooling_stress_pa, np.repeat(state.cooling_stress_pa, 4))
    np.testing.assert_array_equal(result.checkpoint.state.crust_age_myr, np.repeat(cp.state.crust_age_myr, 4))
    assert result.fracture.events == fracture.events
    result.model.save_state(tmp_path / "starter.npz", result.starter_state)
    result.fracture.save(tmp_path / "fracture.npz")
    save_checkpoint(tmp_path / "mature", result.checkpoint)
    assert equal(result.model.load_state(tmp_path / "starter.npz"), result.starter_state)
    restored = YoungShellFracture.load(result.model, tmp_path / "fracture.npz")
    assert equal(restored.memory, result.fracture.memory)
    assert restored.events == result.fracture.events
    loaded = load_checkpoint(tmp_path / "mature", PlateTopologyManager(result.checkpoint.manager.params))
    assert equal(loaded.state, result.checkpoint.state)
    assert equal(loaded.system, result.checkpoint.system)


def test_same_mesh_is_an_independent_exact_copy(real_source):
    model, state, fracture, cp, cfg = source_copy(real_source)
    result = remesh_continuation_state(model, state, fracture, cp, cfg, 1)
    assert equal(result.checkpoint, cp)
    assert equal(result.starter_state, state)
    assert equal(result.fracture.memory, fracture.memory)
    assert result.model.fingerprint == model.fingerprint
    result.checkpoint.state.crust_age_myr[0] += 100
    assert result.checkpoint.state.crust_age_myr[0] != cp.state.crust_age_myr[0]


@pytest.mark.parametrize("target", [0, -1, True, 2.5])
def test_coarsening_and_invalid_targets_do_not_mutate_source(real_source, target):
    model, state, fracture, cp, cfg = source_copy(real_source)
    old = deepcopy((state, fracture.memory, cp, cfg))
    with pytest.raises(ValueError, match="Coarsening|integer|1..8"):
        remesh_continuation_state(model, state, fracture, cp, cfg, target)
    assert equal(old, (state, fracture.memory, cp, cfg))


def test_noncanonical_geometry_disagreement_and_stale_contact_fail_explicitly(real_source):
    model, state, fracture, cp, cfg = source_copy(real_source)
    cfg["mesh"]["subdivisions"] = 2
    with pytest.raises(ValueError, match="configuration"):
        remesh_continuation_state(model, state, fracture, cp, cfg, 2)
    cfg["mesh"]["subdivisions"] = 1
    cp.manager.collision_contact_faces[(0, 1)] = (int(np.flatnonzero(cp.state.cell_plate == 0)[0]),)
    with pytest.raises(ValueError, match="footprint"):
        remesh_continuation_state(model, state, fracture, cp, cfg, 2)


def test_nonfinite_or_negative_extensive_material_is_rejected(real_source):
    model, state, fracture, cp, cfg = source_copy(real_source)
    cp.state.oceanic_volume_km3[0] = -1.
    with pytest.raises(ValueError, match="extensive volume"):
        remesh_continuation_state(model, state, fracture, cp, cfg, 2)


def test_excessive_target_is_rejected_before_any_mesh_allocation(real_source, monkeypatch):
    import tectonics.genesis_continuation_remesh as module
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid target must be rejected before building meshes")
    monkeypatch.setattr(module, "build_icosphere", forbidden)
    with pytest.raises(ValueError, match="1..8"):
        remesh_continuation_state(*real_source, 20)


def test_valid_but_coarser_target_is_rejected_without_majority_ownership(real_source):
    fine = remesh_continuation_state(*real_source, 2)
    before = deepcopy((fine.starter_state, fine.fracture.memory, fine.checkpoint, fine.config))
    with pytest.raises(ValueError, match="Coarsening is unsupported"):
        remesh_continuation_state(fine.model, fine.starter_state, fine.fracture, fine.checkpoint, fine.config, 1)
    assert equal(before, (fine.starter_state, fine.fracture.memory, fine.checkpoint, fine.config))


def test_arbitrary_spherical_mesh_cannot_claim_canonical_ancestry(real_source):
    model, state, fracture, cp, cfg = source_copy(real_source)
    model = copy(model)
    model.mesh = deepcopy(model.mesh)
    model.mesh.vertices[[0, 1]] = model.mesh.vertices[[1, 0]]
    with pytest.raises(ValueError, match="canonical unrotated"):
        remesh_continuation_state(model, state, fracture, cp, cfg, 2)


def test_fine_loading_range_does_not_expand_membrane_fem_range():
    shell = ShellParameters(subdivisions=5)
    thermal = GenesisParameters()
    tides = TidalParameters(enabled=True, spin_state=SYNCHRONOUS_SPIN)
    with pytest.raises(ValueError, match="1..4"):
        shell.validate(thermal)
    loading = StarterLoadingModel(thermal, tides, shell)
    assert loading.shell.subdivisions == 5
    assert loading.initial().column_enthalpy.shape == (shell.column_layers,)
