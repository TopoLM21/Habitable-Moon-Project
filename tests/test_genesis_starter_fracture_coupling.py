"""Installed young hooks preserve one damage owner and normal topology remaps."""
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.dynamics import DynamicsParameters, angular_velocity_vectors, update_plate_dynamics
from tectonics.genesis import GenesisParameters
from tectonics.genesis_shell import ShellParameters
from tectonics.genesis_starter import StarterModel
from tectonics.genesis_starter_continuation import YoungWorldCoupling, build_starter_continuation
from tectonics.genesis_starter_fracture import YoungShellFracture
from tectonics.genesis_starter_topology import select_starter_cut
from tectonics.genesis_tides import TidalParameters, SYNCHRONOUS_SPIN
from tectonics.lithosphere import CrustType, advance_lithosphere
from tectonics.mesh import build_icosphere, connected_components
from tectonics.simulation import load_config
from tectonics.subduction_memory import SlabZone, remap_subduction_memory
from tectonics.topology import PlateTopologyManager, PlateTopologyParameters
from tectonics.transport import remap_transport_state


@pytest.fixture(scope="module")
def source():
    model = StarterModel(build_icosphere(2), GenesisParameters(),
        TidalParameters(enabled=True, spin_state=SYNCHRONOUS_SPIN),
        ShellParameters(subdivisions=2, convective_traction_pa=50000.))
    state = model.advance(model.initial_state(), 2.)
    assert state.stopped_reason == "first_partition"
    cfg = load_config(Path(__file__).resolve().parents[1]/"configs/canonical_moon.yaml")
    return model, state, cfg


def installed(source, lithosphere=advance_lithosphere):
    model, source_state, mature_cfg = source
    bundle, cfg, _ = build_starter_continuation(model, source_state, mature_cfg)
    cp = bundle.checkpoint
    coupling = YoungWorldCoupling(model, source_state, cfg, cp)
    base = SimpleNamespace(update_plate_dynamics=update_plate_dynamics,
        advance_hydrosphere=lambda *args, **kwargs: (args[3], None),
        PlateTopologyManager=PlateTopologyManager, remap_transport_state=remap_transport_state)
    runner = SimpleNamespace(base=base,
        v124=SimpleNamespace(_original_advance_lithosphere=lithosphere))
    coupling.install(runner)
    return model, cp, cfg, coupling, runner


def material_step(model, cp, coupling, runner, dt, **kwargs):
    return runner.v124._original_advance_lithosphere(model.mesh, cp.system, cp.state, dt,
        model.thermal.radius_km, model.thermal.surface_gravity_m_s2, 48., coupling,
        transport_state=cp.transport_state, **kwargs)


def test_loading_runs_once_and_installed_material_hook_disables_duplicate_damage(source, monkeypatch):
    seen = {}

    def captured(*args, **kwargs):
        seen["rate"] = kwargs["tidal_damage_rate_per_myr"]
        seen["relaxation"] = kwargs["tidal_damage_relaxation_myr"]
        return advance_lithosphere(*args, **kwargs)

    model, cp, _, coupling, runner = installed(source, captured)
    archive = coupling.source_state.damage.copy()
    incoming = cp.state.tidal_damage.copy()
    old_material_sample = model._material_sample
    durations = []

    def counted(memory, before, after):
        durations.append(after.time_myr-before.time_myr)
        return old_material_sample(memory, before, after)

    monkeypatch.setattr(model, "_material_sample", counted)
    thermal, _ = runner.base.advance_thermal_state(cp.thermal, .3)
    loaded = coupling.fracture.damage.copy()
    assert not np.array_equal(loaded, incoming)
    assert sum(durations) == pytest.approx(.3)
    count_after_heat = len(durations)
    out, _, _, diag = material_step(model, cp, coupling, runner, .3,
        tidal_damage_rate_per_myr=99., tidal_damage_relaxation_myr=.001)
    assert seen == {"rate": 0., "relaxation": np.inf}
    assert len(durations) == count_after_heat
    donor = diag.material_source_index
    expected = np.zeros(model.mesh.cell_count)
    expected[donor >= 0] = loaded[donor[donor >= 0]]
    np.testing.assert_array_equal(out.tidal_damage, expected)
    np.testing.assert_array_equal(coupling.fracture.damage, expected)
    np.testing.assert_array_equal(cp.state.tidal_damage, incoming)
    np.testing.assert_array_equal(coupling.source_state.damage, archive)
    assert out.time_myr == thermal.time_myr == coupling.fracture.time_myr
    assert diag.mean_tidal_damage == pytest.approx(expected.mean())


def test_real_committed_transport_advects_damage_and_resets_new_material_once(source):
    model, cp, _, coupling, runner = installed(source)
    runner.base.advance_thermal_state(cp.thermal, 1.)
    # A deliberately rapid motion exercises actual mature donor selection;
    # this is a transport contract fixture, not a claimed young-world speed.
    for i, plate in enumerate(cp.system.plates):
        plate.angular_speed_rad_per_myr = .45 * (1. if i == 0 else .4)
    loaded = np.linspace(.02, .98, model.mesh.cell_count)
    cooling = np.linspace(1., 1000., model.mesh.cell_count)
    coupling.fracture.set_damage(loaded)
    coupling.fracture.memory.cooling_stress_pa[:] = cooling
    out, _, _, diag = material_step(model, cp, coupling, runner, 1.)
    donor = diag.material_source_index
    assert cp.transport_state.cumulative_commit_count > 0
    assert np.any((donor >= 0) & (donor != np.arange(len(donor))))
    assert np.any(donor < 0)
    expected_damage = np.zeros_like(loaded)
    expected_cooling = np.zeros_like(cooling)
    valid = donor >= 0
    expected_damage[valid] = loaded[donor[valid]]
    expected_cooling[valid] = cooling[donor[valid]]
    np.testing.assert_array_equal(out.tidal_damage, expected_damage)
    np.testing.assert_array_equal(coupling.fracture.memory.cooling_stress_pa, expected_cooling)
    assert not np.any(coupling.fracture.memory.consumed_band[~valid])


def test_actual_continental_breakup_damage_reset_survives_the_young_hook(source):
    model, cp, _, coupling, runner = installed(source)
    runner.base.advance_thermal_state(cp.thermal, 1.)
    # Supply a fully thinned, actively extending continental fixture so the
    # real mature breakup rule runs; this is not initial Genesis material.
    for plate in cp.system.plates:
        plate.angular_speed_rad_per_myr = 0.
    cp.state.crust_type[:] = int(CrustType.CONTINENTAL)
    cp.state.crust_thickness_km[:] = 15.
    cp.state.continental_fraction[:] = 1.
    cp.state.continental_volume_km3[:] = model.areas * 15.
    cp.state.oceanic_volume_km3[:] = 0.
    cp.state.rift_extension[:] = 1.
    cp.state.extension_age_myr[:] = 100.
    coupling.fracture.set_damage(np.full(model.mesh.cell_count, .9))
    out, _, _, diag = material_step(model, cp, coupling, runner, 1.,
        continental_extension_external_forcing=np.ones(model.mesh.cell_count))
    assert diag.tidally_rifted_continental_area_km2 > 0.
    np.testing.assert_array_equal(out.crust_type, int(CrustType.OCEANIC))
    np.testing.assert_allclose(out.tidal_damage, .9*.35, rtol=0., atol=1e-15)
    np.testing.assert_array_equal(coupling.fracture.damage, out.tidal_damage)
    assert diag.mean_tidal_damage == pytest.approx(.9*.35)


def _supply_separating_band(model, cp, coupling):
    # Prescribe a geometric loading fixture while using the real cut selector,
    # split operator and manager. Completely eligible plates cannot be split.
    p = model.parameters
    for axis in np.eye(3):
        for width in (.1, .15, .2, .25, .3):
            band = np.abs(model.mesh.centroids @ axis) < width
            cut = select_starter_cut(model.mesh, cp.system, band, band.astype(float),
                model.thermal.radius_km, p.min_child_area_km2, p.min_band_span_km)
            if cut is not None:
                coupling.fracture.set_damage(band.astype(float))
                coupling.fracture.memory.consumed_band[:] = False
                cp.state.tidal_damage[:] = coupling.fracture.damage
                return
    pytest.fail("The topology fixture needs a separating band")


def test_real_manager_reports_continued_split_for_normal_transport_and_slab_remaps(source):
    model, cp, cfg, coupling, runner = installed(source)
    _supply_separating_band(model, cp, coupling)
    before = deepcopy(cp.system)
    original_material = {f.name: getattr(cp.state, f.name).copy()
        for f in fields(cp.state) if isinstance(getattr(cp.state, f.name), np.ndarray)
        and f.name != "cell_plate"}
    params = PlateTopologyParameters(**cfg["plate_topology"])
    manager = runner.base.PlateTopologyManager(params)
    # The ordinary age-based split is on cooldown; the already loaded young
    # band has its own consumed-band latch instead of an invented crust age.
    manager.last_split_time_myr = cp.state.time_myr
    updated, diag, events = manager.update(model.mesh, cp.state, cp.system, [],
                                           model.thermal.radius_km, 1.)
    assert len(updated.plates) == len(before.plates)+1
    assert diag.topology_changed and diag.split_events == 1
    assert diag.plate_count_before == len(before.plates)
    assert diag.plate_count_after == len(updated.plates)
    counts = np.bincount(updated.cell_plate)
    assert diag.mean_plate_cells == pytest.approx(counts.mean())
    assert len(events) == 1 and events[0].kind == "split"
    assert manager.last_split_time_myr == cp.state.time_myr
    assert len(coupling.fracture.events) == 1
    np.testing.assert_array_equal(cp.state.cell_plate, updated.cell_plate)
    for name, values in original_material.items():
        np.testing.assert_array_equal(getattr(cp.state, name), values)
    old_omega = angular_velocity_vectors(before)
    new_omega = angular_velocity_vectors(updated)
    for pid in range(len(updated.plates)):
        cells = np.flatnonzero(updated.cell_plate == pid)
        assert len(connected_components(cells, model.mesh.neighbors)) == 1
        parent = np.unique(before.cell_plate[cells])
        assert len(parent) == 1
        np.testing.assert_allclose(new_omega[pid], old_omega[parent[0]], rtol=0., atol=1e-18)
    remapped = runner.base.remap_transport_state(before, updated, cp.transport_state)
    assert remapped.residual_quaternions.shape == (len(updated.plates), 4)
    # Exercise the exact downstream remap used by the mature runner when the
    # returned topology_changed flag is true, with real remembered geometry.
    parent = events[0].parents[0]
    other = 1-parent
    child = events[0].children[-1]
    cell = np.flatnonzero(updated.cell_plate == child)[0]
    midpoint = model.mesh.centroids[cell].copy()
    cp.subduction_memory.zones[(parent, other)] = SlabZone(parent, other,
        slab_length_km=100., slab_depth_km=50., trench_midpoint=midpoint,
        cumulative_subducted_area_km2=1200.)
    memory = remap_subduction_memory(model.mesh, before, updated, cp.subduction_memory)
    assert len(memory.zones) == 1
    zone = next(iter(memory.zones.values()))
    assert zone.subducting_plate == child
    assert zone.slab_length_km == 100. and zone.cumulative_subducted_area_km2 == 1200.
    runner.base.advance_hydrosphere(model.mesh, cp.state, cp.topo, cp.hydrosphere)
    assert coupling.history[-1]["plate_count"] == len(updated.plates)
    assert coupling.history[-1]["young_fracture"]["continued_split_count"] == 1


def test_installed_dynamics_has_no_slab_force_before_subduction_memory_exists(source):
    model, cp, cfg, _, runner = installed(source)
    assert not cp.subduction_memory.zones
    p = DynamicsParameters(**cfg["plate_dynamics"])
    assert p.slab_pull_weight > 0.
    args = (model.mesh, cp.state, cp.system, cp.baseline, model.thermal.radius_km,
            1., 0., 0.)
    kwargs = dict(subduction_memory=cp.subduction_memory, mantle_flow=cp.mantle_flow)
    expected = update_plate_dynamics(*args, replace(p, slab_pull_weight=0.), **kwargs)
    observed = runner.base.update_plate_dynamics(*args, p, **kwargs)
    np.testing.assert_array_equal(observed[3], expected[3])
    np.testing.assert_array_equal(angular_velocity_vectors(observed[0]),
                                  angular_velocity_vectors(expected[0]))


@pytest.mark.parametrize("changes", [{"split_enabled": False}, {"max_events_per_step": 0}])
def test_continued_fracture_respects_mature_topology_disable_and_event_budget(source, changes):
    model, cp, cfg, coupling, runner = installed(source)
    _supply_separating_band(model, cp, coupling)
    params = replace(PlateTopologyParameters(**cfg["plate_topology"]), **changes)
    manager = runner.base.PlateTopologyManager(params)
    before = cp.system.cell_plate.copy()
    updated, diag, events = manager.update(model.mesh, cp.state, cp.system, [],
                                           model.thermal.radius_km, 1.)
    np.testing.assert_array_equal(updated.cell_plate, before)
    assert not diag.topology_changed and not events
    assert not coupling.fracture.events


def test_saved_live_fracture_is_distinct_from_archived_first_partition(source, tmp_path):
    model, cp, cfg, coupling, runner = installed(source)
    archive = coupling.source_state.damage.copy()
    thermal, _ = runner.base.advance_thermal_state(cp.thermal, .3)
    state, _, _, _ = material_step(model, cp, coupling, runner, .3)
    coupling.record(state, cp.system, cp.transport_state)
    coupling.fracture.save(tmp_path/"fracture.npz")
    model.save_state(tmp_path/"starter.npz", coupling.source_state)
    loaded = YoungShellFracture.load(model, tmp_path/"fracture.npz")
    restored_archive = model.load_state(tmp_path/"starter.npz")
    np.testing.assert_array_equal(loaded.damage, state.tidal_damage)
    np.testing.assert_array_equal(restored_archive.damage, archive)
    assert not np.array_equal(loaded.damage, archive)
    restored_cp = deepcopy(cp)
    restored_cp.state, restored_cp.thermal = state, thermal
    resumed = YoungWorldCoupling(model, restored_archive, cfg, restored_cp, fracture=loaded)
    assert resumed.fracture.time_myr == restored_archive.time_myr
    np.testing.assert_array_equal(resumed.fracture.memory.cooling_stress_pa,
                                  coupling.fracture.memory.cooling_stress_pa)
