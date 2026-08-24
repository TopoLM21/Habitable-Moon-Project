"""Connected plate generation and Euler-pole assignment."""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from .mesh import SphereMesh, connected_components


Array = np.ndarray


@dataclass(slots=True)
class Plate:
    plate_id: int
    seed_cell: int
    euler_axis: Array             # unit vector
    angular_speed_rad_per_myr: float


@dataclass(slots=True)
class PlateSystem:
    cell_plate: Array             # (F,), integer plate id
    plates: tuple[Plate, ...]


def _angular_distance(a: Array, b: Array) -> Array:
    return np.arccos(np.clip(a @ b, -1.0, 1.0))


def _farthest_point_seeds(centroids: Array, count: int, rng: np.random.Generator) -> list[int]:
    if count < 1 or count > len(centroids):
        raise ValueError("Invalid plate count")
    seeds = [int(rng.integers(len(centroids)))]
    nearest = _angular_distance(centroids, centroids[seeds[0]])
    for _ in range(1, count):
        # A tiny random factor prevents deterministic symmetry ties.
        score = nearest * rng.uniform(0.995, 1.005, size=len(nearest))
        nxt = int(np.argmax(score))
        seeds.append(nxt)
        nearest = np.minimum(nearest, _angular_distance(centroids, centroids[nxt]))
    return seeds


def generate_connected_plates(
    mesh: SphereMesh,
    plate_count: int,
    rng: np.random.Generator,
    boundary_roughness: float = 0.25,
) -> tuple[Array, list[int]]:
    """Partition the face-adjacency graph with stochastic multi-source Dijkstra.

    Every claimed cell is reached through an already claimed cell of the same
    plate, so each resulting plate is connected by construction.
    """
    if not (0.0 <= boundary_roughness <= 1.0):
        raise ValueError("boundary_roughness must be in [0, 1]")

    seeds = _farthest_point_seeds(mesh.centroids, plate_count, rng)
    owner = np.full(mesh.cell_count, -1, dtype=np.int32)
    cost = np.full(mesh.cell_count, np.inf, dtype=np.float64)
    heap: list[tuple[float, int, int]] = []

    for plate_id, seed in enumerate(seeds):
        owner[seed] = plate_id
        cost[seed] = 0.0
        heapq.heappush(heap, (0.0, plate_id, seed))

    while heap:
        current_cost, plate_id, cell = heapq.heappop(heap)
        if current_cost != cost[cell] or owner[cell] != plate_id:
            continue
        c0 = mesh.centroids[cell]
        for neighbor in mesh.neighbors[cell]:
            c1 = mesh.centroids[neighbor]
            edge_angle = float(np.arccos(np.clip(np.dot(c0, c1), -1.0, 1.0)))
            stochastic = 1.0 + boundary_roughness * rng.uniform(-0.75, 1.25)
            proposal = current_cost + edge_angle * max(stochastic, 0.05)
            if proposal < cost[neighbor]:
                cost[neighbor] = proposal
                owner[neighbor] = plate_id
                heapq.heappush(heap, (proposal, plate_id, neighbor))

    if np.any(owner < 0):
        raise RuntimeError("Some mesh cells were not assigned to a plate")

    for plate_id in range(plate_count):
        components = connected_components(np.flatnonzero(owner == plate_id), mesh.neighbors)
        if len(components) != 1:
            raise RuntimeError(f"Plate {plate_id} is disconnected ({len(components)} components)")

    return owner, seeds


def random_plate_system(
    mesh: SphereMesh,
    plate_count: int,
    seed: int,
    boundary_roughness: float,
    min_speed_deg_per_myr: float,
    max_speed_deg_per_myr: float,
) -> PlateSystem:
    rng = np.random.default_rng(seed)
    cell_plate, seed_cells = generate_connected_plates(
        mesh=mesh,
        plate_count=plate_count,
        rng=rng,
        boundary_roughness=boundary_roughness,
    )

    axes = rng.normal(size=(plate_count, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    speeds_deg = rng.uniform(min_speed_deg_per_myr, max_speed_deg_per_myr, size=plate_count)
    signs = rng.choice(np.array([-1.0, 1.0]), size=plate_count)
    speeds_rad = np.deg2rad(speeds_deg * signs)

    plates = tuple(
        Plate(
            plate_id=i,
            seed_cell=seed_cells[i],
            euler_axis=axes[i],
            angular_speed_rad_per_myr=float(speeds_rad[i]),
        )
        for i in range(plate_count)
    )
    return PlateSystem(cell_plate=cell_plate, plates=plates)
