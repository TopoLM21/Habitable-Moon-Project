"""Regression coverage for sharing the legacy rift shortest-path search."""

import heapq
from types import SimpleNamespace

import numpy as np
import pytest

import tectonics.late_tectonics as late
from tectonics.mesh import build_icosphere


def _legacy_single_goal(mesh, owner, plate, start, goal, preference, max_cells, radius_km):
    """Frozen single-goal algorithm used before the shared search."""
    dist = {int(start): 0.0}
    previous = {}
    heap = [(0.0, int(start))]
    visited = 0
    while heap:
        distance, cell = heapq.heappop(heap)
        if distance != dist.get(cell):
            continue
        visited += 1
        if cell == goal:
            break
        if visited > max(5000, 30 * max_cells):
            break
        for neighbor in mesh.neighbors[cell]:
            if int(owner[neighbor]) != plate:
                continue
            local = 0.25 + 0.75 * (1.0 - float(np.clip(preference[neighbor], 0.0, 1.0)))
            candidate = distance + local * late._edge_center_distance_km(
                mesh, cell, neighbor, radius_km,
            )
            if candidate < dist.get(int(neighbor), 1e100):
                dist[int(neighbor)] = candidate
                previous[int(neighbor)] = cell
                heapq.heappush(heap, (candidate, int(neighbor)))
    if goal not in dist:
        return []
    path = [int(goal)]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()
    return path if len(path) <= int(max_cells) else []


def _independent_paths(mesh, owner, plate, start, goals, preference, max_cells, radius_km):
    return {
        int(goal): _legacy_single_goal(
            mesh, owner, plate, start, goal, preference, max_cells, radius_km,
        )
        for goal in goals
    }


@pytest.mark.parametrize("max_cells", [0, 1, 2, 320])
@pytest.mark.parametrize("constant_preference", [False, True])
def test_shared_paths_match_independent_search_with_limits_and_ties(max_cells, constant_preference):
    mesh = build_icosphere(2)
    owner = np.zeros(mesh.cell_count, dtype=np.int32)
    preference = (
        np.ones(mesh.cell_count)
        if constant_preference
        else np.random.default_rng(901).uniform(-0.2, 1.2, mesh.cell_count)
    )
    goals = [0, *mesh.neighbors[0], 17, 99, 251, 319, 17]
    args = (mesh, owner, 0, 0, goals, preference, max_cells, 5287.0)
    assert late._dijkstra_paths(*args) == _independent_paths(*args)
    assert late._dijkstra_path(mesh, owner, 0, 0, 251, preference, max_cells, 5287.0) == (
        _legacy_single_goal(mesh, owner, 0, 0, 251, preference, max_cells, 5287.0)
    )


def test_shared_search_handles_unreachable_goals_and_empty_request():
    mesh = build_icosphere(2)
    owner = (np.abs(mesh.centroids[:, 2]) <= 0.5).astype(np.int32)
    north = np.flatnonzero(mesh.centroids[:, 2] > 0.5)
    south = np.flatnonzero(mesh.centroids[:, 2] < -0.5)
    preference = np.ones(mesh.cell_count)
    start = int(north[0])
    goals = [start, int(north[-1]), int(south[0])]
    args = (mesh, owner, 0, start, goals, preference, mesh.cell_count, 5287.0)
    paths = late._dijkstra_paths(*args)
    assert paths == _independent_paths(*args)
    assert paths[goals[-1]] == []
    assert late._dijkstra_paths(mesh, owner, 0, start, [], preference, 320, 5287.0) == {}


def test_work_guard_preserves_discovered_but_unsettled_legacy_paths():
    # All leaves have equal positive edge costs and a two-cell path.  The work
    # guard fires while discovered goals are still waiting in heap order.
    n = 5003
    centroids = np.zeros((n, 3))
    centroids[0, 0] = 1.0
    centroids[1:, 1] = 1.0
    mesh = SimpleNamespace(
        centroids=centroids,
        neighbors=(tuple(range(1, n)), *((0,) for _ in range(1, n))),
    )
    goals = [0, 1, 4999, 5000, 5001, 5002]
    args = (mesh, np.zeros(n, dtype=np.int32), 0, 0, goals, np.ones(n), 2, 1.0)
    paths = late._dijkstra_paths(*args)
    assert paths == _independent_paths(*args)
    assert paths[5000] == [0, 5000]  # Goal reached on the guard iteration.
    assert paths[5002] == [0, 5002]  # Goal discovered, but never popped.


@pytest.mark.parametrize("physical_limits", [False, True])
def test_cross_plate_selection_is_identical_with_one_shared_search(monkeypatch, physical_limits):
    mesh = build_icosphere(3)
    owner = (mesh.centroids[:, 2] < 0.0).astype(np.int32)
    score = np.random.default_rng(902).uniform(0.0, 1.0, mesh.cell_count)
    params = late.LateTectonicsParameters(
        rift_min_path_cells=2,
        rift_max_path_cells=mesh.cell_count,
        rift_min_path_length_km=2500.0 if physical_limits else None,
        rift_max_path_length_km=20000.0 if physical_limits else None,
    )
    real_distance = late._edge_center_distance_km
    edge_evaluations = 0

    def counted_distance(*args):
        nonlocal edge_evaluations
        edge_evaluations += 1
        return real_distance(*args)

    monkeypatch.setattr(late, "_edge_center_distance_km", counted_distance)
    actual = late._choose_cross_plate_path(mesh, owner, 0, score, params, 5287.0)
    shared_evaluations = edge_evaluations
    edge_evaluations = 0
    monkeypatch.setattr(late, "_dijkstra_paths", _independent_paths)
    expected = late._choose_cross_plate_path(mesh, owner, 0, score, params, 5287.0)
    assert actual == expected
    assert len(actual) >= 2
    # Deterministic work count, avoiding timing assertions on busy machines.
    assert shared_evaluations < edge_evaluations / 5
