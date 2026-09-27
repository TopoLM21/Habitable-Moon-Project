from copy import deepcopy

import numpy as np
import pytest

from tectonics.genesis_starter_topology import select_starter_cut, split_starter_band
from tectonics.kinematics import angular_velocity_vectors
from tectonics.lithosphere import CrustType, LithosphereState
from tectonics.mesh import build_icosphere, connected_components
from tectonics.plates import Plate, PlateSystem
from tectonics.topology import PlateTopologyParameters, _attempt_split


RADIUS = 5287.0
MIN_AREA = 3_100_000.0
MIN_SPAN = 1000.0


@pytest.fixture
def shell():
    mesh = build_icosphere(3)
    system = PlateSystem(
        np.zeros(mesh.cell_count, dtype=np.int32),
        (Plate(0, 0, np.array([0.0, 0.0, 1.0]), np.deg2rad(0.25)),),
    )
    weak = np.abs(mesh.centroids[:, 0]) < 0.12
    return mesh, system, weak


def choose(mesh, system, eligible, preference=None, **overrides):
    args = dict(radius_km=RADIUS, min_child_area_km2=MIN_AREA, min_band_span_km=MIN_SPAN)
    args.update(overrides)
    if preference is None:
        preference = eligible.astype(float)
    return select_starter_cut(mesh, system, eligible, preference, **args)


def split(mesh, system, cut, **overrides):
    args = dict(radius_km=RADIUS, min_child_area_km2=MIN_AREA, min_band_span_km=MIN_SPAN)
    args.update(overrides)
    return split_starter_band(mesh, system, cut, **args)


def test_closed_weak_belt_splits_global_shell_without_velocity_kick(shell):
    mesh, system, weak = shell
    before = deepcopy(system)
    cut = choose(mesh, system, weak)
    assert cut is not None
    assert np.all(weak[cut])
    out, event = split(mesh, system, cut, time_myr=7.0)
    assert out is not None and event is not None
    assert len(out.plates) == 2
    assert event.kind == "split" and event.time_myr == 7.0
    assert event.parents == (0,) and event.children == (0, 1)
    areas = mesh.physical_cell_areas_km2(RADIUS)
    for p in range(2):
        cells = np.flatnonzero(out.cell_plate == p)
        assert np.sum(areas[cells]) >= MIN_AREA
        assert len(connected_components(cells, mesh.neighbors)) == 1
    np.testing.assert_allclose(
        angular_velocity_vectors(out), np.repeat(angular_velocity_vectors(before), 2, axis=0),
        rtol=0.0, atol=1e-18,
    )
    np.testing.assert_array_equal(system.cell_plate, before.cell_plate)
    np.testing.assert_array_equal(angular_velocity_vectors(system), angular_velocity_vectors(before))


@pytest.mark.parametrize("kind", ["none", "all", "patch", "open_belt"])
def test_nonseparating_eligible_regions_do_not_invent_a_partition(shell, kind):
    mesh, system, weak = shell
    eligible = {
        "none": np.zeros(mesh.cell_count, dtype=bool),
        "all": np.ones(mesh.cell_count, dtype=bool),
        "patch": mesh.centroids[:, 0] > 0.9,
        "open_belt": weak & (mesh.centroids[:, 1] < 0.8),
    }[kind]
    assert choose(mesh, system, eligible) is None


def test_physical_area_and_span_thresholds_are_enforced(shell):
    mesh, system, weak = shell
    cut = np.flatnonzero(weak)
    total_area = np.sum(mesh.physical_cell_areas_km2(RADIUS))
    assert choose(mesh, system, weak, min_child_area_km2=0.51 * total_area) is None
    assert choose(mesh, system, weak, min_band_span_km=4.0 * RADIUS) is None
    assert split(mesh, system, cut, min_child_area_km2=0.51 * total_area) == (None, None)
    assert split(mesh, system, cut, min_band_span_km=4.0 * RADIUS) == (None, None)


@pytest.mark.parametrize("kick", [0.0, 0.05])
def test_starter_uses_same_partition_motion_and_event_as_mature_split(shell, kick):
    mesh, system, weak = shell
    n = mesh.cell_count
    state = LithosphereState(
        time_myr=50.0, cell_plate=system.cell_plate.copy(),
        crust_type=np.where(weak, CrustType.OCEANIC, CrustType.CONTINENTAL).astype(np.int8),
        crust_age_myr=np.where(weak, 2.0, 500.0),
        crust_thickness_km=np.where(weak, 7.0, 35.0),
        tidal_damage=np.where(weak, 0.08, 0.0),
    )
    params = PlateTopologyParameters(
        split_min_rift_span_km=MIN_SPAN, split_min_child_area_km2=MIN_AREA,
        split_differential_speed_deg_per_myr=kick,
    )
    mature, mature_event = _attempt_split(mesh, state, system, params, RADIUS)
    cut = choose(mesh, system, weak)
    starter, starter_event = split(
        mesh, system, cut, time_myr=state.time_myr, differential_speed_deg_per_myr=kick,
    )
    assert mature is not None and starter is not None
    np.testing.assert_array_equal(starter.cell_plate, mature.cell_plate)
    np.testing.assert_array_equal(angular_velocity_vectors(starter), angular_velocity_vectors(mature))
    assert starter_event == mature_event


def test_repeat_selection_and_reordered_cut_are_deterministic(shell):
    mesh, system, weak = shell
    cut = choose(mesh, system, weak)
    np.testing.assert_array_equal(cut, choose(mesh, deepcopy(system), weak.copy()))
    forward, event = split(mesh, system, cut)
    backward, reverse_event = split(mesh, system, cut[::-1])
    np.testing.assert_array_equal(forward.cell_plate, backward.cell_plate)
    assert event == reverse_event


def test_existing_child_can_split_again_while_other_plate_is_preserved(shell):
    mesh, system, weak = shell
    initial, _ = split(mesh, system, choose(mesh, system, weak))
    initial.plates[1].angular_speed_rad_per_myr = np.deg2rad(0.4)
    next_weak = np.abs(mesh.centroids[:, 2]) < 0.12
    cut = choose(mesh, initial, next_weak)
    assert cut is not None
    parent = int(initial.cell_plate[cut[0]])
    other = 1 - parent
    before_motion = angular_velocity_vectors(initial).copy()
    result, event = split(mesh, initial, cut)
    assert result is not None and len(result.plates) == 3
    assert event.parents == (parent,)
    np.testing.assert_array_equal(result.cell_plate[initial.cell_plate == other], other)
    np.testing.assert_array_equal(angular_velocity_vectors(result)[other], before_motion[other])
    for child in event.children:
        np.testing.assert_allclose(angular_velocity_vectors(result)[child], before_motion[parent], atol=1e-18)


def test_stronger_supplied_band_wins_without_crossing_ineligible_cells(shell):
    mesh, system, _ = shell
    # Two separated latitude belts both leave a large cap and remainder.
    north = np.abs(mesh.centroids[:, 2] - 0.5) < 0.1
    south = np.abs(mesh.centroids[:, 2] + 0.5) < 0.1
    eligible = north | south
    preference = north.astype(float) + 2.0 * south
    cut = choose(mesh, system, eligible, preference)
    assert cut is not None and np.all(south[cut])
    cut_north = choose(mesh, system, eligible, 2.0 * north + south)
    assert cut_north is not None and np.all(north[cut_north])


@pytest.mark.parametrize("case", ["eligible_shape", "eligible_dtype", "preference_shape", "nan", "negative"])
def test_invalid_selection_fields_are_rejected(shell, case):
    mesh, system, weak = shell
    eligible = weak.copy()
    preference = weak.astype(float)
    if case == "eligible_shape":
        eligible = eligible[:-1]
    elif case == "eligible_dtype":
        eligible = eligible.astype(int)
    elif case == "preference_shape":
        preference = preference[:-1]
    elif case == "nan":
        preference[0] = np.nan
    else:
        preference[0] = -1.0
    with pytest.raises(ValueError):
        choose(mesh, system, eligible, preference)


@pytest.mark.parametrize("name,value", [
    ("radius_km", 0.0), ("radius_km", np.inf), ("min_child_area_km2", -1.0),
    ("min_band_span_km", np.nan), ("min_band_span_km", -1.0),
])
def test_invalid_physical_thresholds_are_rejected(shell, name, value):
    mesh, system, weak = shell
    with pytest.raises(ValueError):
        choose(mesh, system, weak, **{name: value})
    with pytest.raises(ValueError):
        split(mesh, system, np.flatnonzero(weak), **{name: value})


@pytest.mark.parametrize("cut", [np.array([0.0]), np.array([-1]), np.array([100000]),
                                  np.array([0, 0]), np.array([[0]]), np.array([0, 1000])])
def test_invalid_cut_is_rejected(shell, cut):
    mesh, system, _ = shell
    with pytest.raises(ValueError):
        split(mesh, system, cut)


def test_empty_and_isolated_cut_do_not_split(shell):
    mesh, system, _ = shell
    assert split(mesh, system, np.array([], dtype=np.int32)) == (None, None)
    assert split(mesh, system, np.array([0], dtype=np.int32)) == (None, None)


@pytest.mark.parametrize("case", ["labels", "axis", "speed", "seed", "disconnected"])
def test_invalid_plate_system_is_rejected(shell, case):
    mesh, original, weak = shell
    system = deepcopy(original)
    if case == "labels":
        system.cell_plate[:] = 1
    elif case == "axis":
        system.plates[0].euler_axis *= 2
    elif case == "speed":
        system.plates[0].angular_speed_rad_per_myr = np.nan
    elif case == "seed":
        system.plates[0].seed_cell = -1
    else:
        owner = (np.abs(mesh.centroids[:, 2]) > 0.6).astype(np.int32)
        system = PlateSystem(owner, tuple(
            Plate(p, int(np.flatnonzero(owner == p)[0]), np.array([0., 0., 1.]), 0.0)
            for p in range(2)
        ))
    with pytest.raises(ValueError):
        choose(mesh, system, weak)


@pytest.mark.parametrize("name,value", [("time_myr", -1.0), ("time_myr", np.nan),
                                      ("differential_speed_deg_per_myr", -0.1)])
def test_invalid_event_clock_or_kick_is_rejected(shell, name, value):
    mesh, system, weak = shell
    with pytest.raises(ValueError):
        split(mesh, system, np.flatnonzero(weak), **{name: value})
