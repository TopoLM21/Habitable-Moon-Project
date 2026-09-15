import numpy as np
import pytest

from tectonics.connectivity import repair_plate_connectivity
from tectonics.mesh import build_icosphere, connected_components
from tectonics.lithosphere import LithosphereState
from tectonics.plates import Plate, PlateSystem
from tectonics.topology import PlateTopologyManager, PlateTopologyParameters
from tectonics.topology import _plate_inertia_tensor


def fragmented_caps(mesh):
    owner = np.ones(mesh.cell_count, dtype=np.int32)
    owner[np.abs(mesh.centroids[:, 2]) > .6] = 0
    crumb = int(np.argmax(mesh.centroids[:, 0]))
    owner[crumb] = 0
    return owner, crumb


def assert_connected(mesh, owner):
    for pid in np.unique(owner):
        assert len(connected_components(np.flatnonzero(owner == pid), mesh.neighbors)) == 1


def test_promotes_large_island_and_floods_crumb_without_crossing_retained_domain():
    mesh = build_icosphere(3)
    owner, crumb = fragmented_caps(mesh)
    saved = owner.copy()
    repair = repair_plate_connectivity(mesh, owner, 5287.,
        minimum_independent_area_km2=None, minimum_independent_cells=30)
    assert repair.promoted_components == 1
    assert repair.reassigned_cells == 1
    assert repair.parent_by_plate == (0, 1, 0)
    assert repair.cell_plate[crumb] == 1
    assert np.array_equal(repair.cell_plate[owner == 1], owner[owner == 1])
    assert np.array_equal(owner, saved)
    assert_connected(mesh, repair.cell_plate)
    again = repair_plate_connectivity(mesh, repair.cell_plate, 5287.,
        minimum_independent_area_km2=None, minimum_independent_cells=30)
    assert np.array_equal(again.cell_plate, repair.cell_plate)
    assert again.promoted_components == again.reassigned_cells == 0


@pytest.mark.parametrize("subdivision", [2, 3, 4])
def test_physical_independent_area_threshold_keeps_caps_across_resolutions(subdivision):
    mesh = build_icosphere(subdivision)
    owner, _ = fragmented_caps(mesh)
    repair = repair_plate_connectivity(mesh, owner, 5287.,
        minimum_independent_area_km2=10e6, minimum_independent_cells=10**9)
    assert repair.promoted_components == 1
    assert len(repair.parent_by_plate) == 3
    assert_connected(mesh, repair.cell_plate)


def test_simultaneous_flood_is_connected_even_for_interlocking_fragment_clouds():
    mesh = build_icosphere(3)
    rng = np.random.default_rng(43)
    owner = rng.integers(0, 5, mesh.cell_count, dtype=np.int32)
    repair = repair_plate_connectivity(mesh, owner, 5287.,
        minimum_independent_area_km2=None, minimum_independent_cells=80)
    assert np.array_equal(np.unique(repair.cell_plate), np.arange(len(repair.parent_by_plate)))
    assert_connected(mesh, repair.cell_plate)
    repeat = repair_plate_connectivity(mesh, owner, 5287.,
        minimum_independent_area_km2=None, minimum_independent_cells=80)
    assert np.array_equal(repair.cell_plate, repeat.cell_plate)


def test_manager_repair_preserves_material_arrays_and_parent_rotation_without_rift_cooldown():
    mesh = build_icosphere(3)
    owner, _ = fragmented_caps(mesh)
    system = PlateSystem(owner.copy(), tuple(
        Plate(pid, int(np.flatnonzero(owner == pid)[0]),
              np.array([0., 0., 1.]), .003 + .001*pid) for pid in range(2)))
    n = mesh.cell_count
    rng = np.random.default_rng(9)
    state = LithosphereState(500., owner.copy(), np.ones(n, dtype=np.int8),
        rng.uniform(10., 800., n), rng.uniform(25., 40., n), rng.random(n),
        continental_fraction=rng.random(n), continental_volume_km3=rng.random(n)*1e6,
        sediment_volume_km3=rng.random(n)*1e4)
    saved = {f: getattr(state, f).copy() for f in state.__dataclass_fields__
             if f != "cell_plate" and isinstance(getattr(state, f), np.ndarray)}
    manager = PlateTopologyManager(PlateTopologyParameters(
        split_enabled=False, merge_enabled=False, min_plate_cells=30,
        max_events_per_step=0, repair_fragmented_plates=True))
    manager.last_split_time_myr = 490.
    out, diag, events = manager.update(mesh, state, system, [], 5287., 4.)
    assert len(out.plates) == 3
    assert events[-1].kind == "connectivity_repair"
    assert diag.connectivity_promoted_components == 1
    assert diag.connectivity_reassigned_cells == 1
    assert manager.last_split_time_myr == 490.
    assert np.allclose(out.plates[2].euler_axis*out.plates[2].angular_speed_rad_per_myr,
                       system.plates[0].euler_axis*system.plates[0].angular_speed_rad_per_myr)
    assert_connected(mesh, out.cell_plate)
    areas = mesh.physical_cell_areas_km2(5287.)
    def momentum(sys):
        return sum((_plate_inertia_tensor(mesh, np.flatnonzero(sys.cell_plate == p.plate_id), areas)
                    @ (p.euler_axis * p.angular_speed_rad_per_myr) for p in sys.plates), np.zeros(3))
    assert np.allclose(momentum(out), momentum(system), rtol=1e-12, atol=1e-8)
    for field, values in saved.items():
        assert np.array_equal(getattr(state, field), values), field


def test_small_connected_existing_plate_keeps_its_identity_during_persistence():
    mesh = build_icosphere(2)
    owner = np.zeros(mesh.cell_count, dtype=np.int32)
    owner[0] = 1
    repair = repair_plate_connectivity(mesh, owner, 5287.,
        minimum_independent_area_km2=None, minimum_independent_cells=100)
    assert np.array_equal(repair.cell_plate, owner)
    assert repair.parent_by_plate == (0, 1)
