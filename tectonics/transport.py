"""Cumulative sub-grid and conservative plate transport (v0.9.9/v0.10 reconstruction).

This module reconstructs the numerical algorithm documented in the preserved
v0.9.9/v0.10 reports after the original source archive was lost:

* unresolved rigid plate motion is accumulated as a quaternion;
* a raster commit is delayed until that motion is resolvable on the mesh;
* committed plates use a sparse one-to-one source->target assignment;
* every source cell is used exactly once and every target at most once within
  a plate, while different plates are assigned independently;
* the represented rigid rotation is fitted from the realised assignment and
  removed from the residual quaternion.

The implementation is intentionally a discrete parcel remap, not an exact
spherical polygon-overlap remap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree

from .lithosphere import CrustType, LithosphereState
from .mesh import SphereMesh
from .plates import PlateSystem
from .cpu_runtime import current_execution, query_workers

Array = np.ndarray


@dataclass(slots=True)
class SubgridTransportParameters:
    min_changed_fraction: float = 0.18
    min_p75_cell_spacing_fraction: float = 0.30
    continental_preference_min_cells: int = 24
    max_hold_myr: float = 120.0
    forced_min_changed_fraction: float = 0.04
    max_fit_pairs: int = 6000
    initial_candidate_count: int = 8
    max_candidate_count: int = 64
    # Reconstruction safety guard: never fall back to an all-cells dense graph
    # on the canonical 20k-cell mesh.
    absolute_candidate_limit: int = 256


@dataclass(slots=True)
class PlateTransportState:
    residual_quaternions: Array  # (P,4), scalar-first unit quaternions
    hold_age_myr: Array          # (P,)
    cumulative_commit_count: int = 0
    max_hold_age_myr: float = 0.0


@dataclass(slots=True)
class TransportDiagnostics:
    committed_plates: int
    cumulative_commit_count: int
    mean_residual_angle_deg: float
    max_residual_angle_deg: float
    max_hold_age_myr: float
    mean_changed_fraction: float
    max_changed_fraction: float


@dataclass(slots=True)
class TransportMap:
    covered: Array               # (P,N) bool
    source: Array                # (P,N) int32; valid where covered
    source_to_target: tuple[Array, ...]
    state: PlateTransportState
    diagnostics: TransportDiagnostics


def identity_quaternion() -> Array:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _normalize_quaternion(q: Array) -> Array:
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-15:
        return identity_quaternion()
    q = q / norm
    # q and -q encode the same rotation; canonicalise for deterministic state.
    if q[0] < 0.0:
        q = -q
    return q


def quaternion_multiply(a: Array, b: Array) -> Array:
    """Hamilton product: rotation ``b`` is applied first, then ``a``."""
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return _normalize_quaternion(np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dtype=np.float64))


def quaternion_conjugate(q: Array) -> Array:
    q = _normalize_quaternion(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quaternion_from_axis_angle(axis: Array, angle_rad: float) -> Array:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-15 or abs(float(angle_rad)) < 1e-15:
        return identity_quaternion()
    axis = axis / norm
    half = 0.5 * float(angle_rad)
    return _normalize_quaternion(np.concatenate(([np.cos(half)], axis * np.sin(half))))


def quaternion_to_matrix(q: Array) -> Array:
    w, x, y, z = _normalize_quaternion(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def quaternion_from_matrix(r: Array) -> Array:
    """Convert a proper 3x3 rotation matrix to a scalar-first quaternion."""
    r = np.asarray(r, dtype=np.float64)
    tr = float(np.trace(r))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25*s, (r[2,1]-r[1,2])/s, (r[0,2]-r[2,0])/s, (r[1,0]-r[0,1])/s])
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = np.sqrt(max(1.0 + r[0,0] - r[1,1] - r[2,2], 0.0)) * 2.0
            q = np.array([(r[2,1]-r[1,2])/s, 0.25*s, (r[0,1]+r[1,0])/s, (r[0,2]+r[2,0])/s])
        elif i == 1:
            s = np.sqrt(max(1.0 + r[1,1] - r[0,0] - r[2,2], 0.0)) * 2.0
            q = np.array([(r[0,2]-r[2,0])/s, (r[0,1]+r[1,0])/s, 0.25*s, (r[1,2]+r[2,1])/s])
        else:
            s = np.sqrt(max(1.0 + r[2,2] - r[0,0] - r[1,1], 0.0)) * 2.0
            q = np.array([(r[1,0]-r[0,1])/s, (r[0,2]+r[2,0])/s, (r[1,2]+r[2,1])/s, 0.25*s])
    return _normalize_quaternion(q)


def rotate_by_quaternion(points: Array, q: Array) -> Array:
    r = quaternion_to_matrix(q)
    out = np.asarray(points, dtype=np.float64) @ r.T
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


def quaternion_angle_deg(q: Array) -> float:
    q = _normalize_quaternion(q)
    angle = 2.0 * np.arccos(np.clip(abs(float(q[0])), -1.0, 1.0))
    return float(np.rad2deg(angle))


def initialize_transport_state(plate_count: int) -> PlateTransportState:
    q = np.tile(identity_quaternion(), (int(plate_count), 1))
    return PlateTransportState(
        residual_quaternions=q,
        hold_age_myr=np.zeros(int(plate_count), dtype=np.float64),
    )


def _median_cell_spacing_rad(mesh: SphereMesh) -> float:
    vals = []
    c = mesh.centroids
    for i, nbs in enumerate(mesh.neighbors):
        if not nbs:
            continue
        vals.append(min(float(np.arccos(np.clip(np.dot(c[i], c[j]), -1.0, 1.0))) for j in nbs))
    return float(np.median(vals)) if vals else 1.0


def _median_cell_spacing_rad_batched(mesh: SphereMesh, neighbors: Array) -> float:
    """Exact regular-mesh equivalent without one Python call per edge."""
    if mesh.cell_count == 0:
        return 1.0
    nbs = np.asarray(neighbors, dtype=np.int32)
    if nbs.ndim != 2 or nbs.shape[0] != mesh.cell_count or nbs.shape[1] == 0:
        return _median_cell_spacing_rad(mesh)
    centroids = np.asarray(mesh.centroids, dtype=np.float64)
    # einsum preserves the three-product reduction used by np.dot in the
    # scalar path on the triangular icosphere; tests cover exact equality.
    dots = np.einsum("ij,ikj->ik", centroids, centroids[nbs])
    distances = np.arccos(np.clip(dots, -1.0, 1.0))
    return float(np.median(np.min(distances, axis=1)))


def _fit_rotation(source_points: Array, target_points: Array, max_pairs: int) -> Array:
    n = len(source_points)
    if n < 3:
        return np.eye(3, dtype=np.float64)
    if n > int(max_pairs):
        # deterministic evenly spaced sample
        idx = np.linspace(0, n - 1, int(max_pairs), dtype=np.int64)
        a = np.asarray(source_points, dtype=np.float64)[idx]
        b = np.asarray(target_points, dtype=np.float64)[idx]
    else:
        a = np.asarray(source_points, dtype=np.float64)
        b = np.asarray(target_points, dtype=np.float64)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    return r


def _optimal_assignment(mesh: SphereMesh, rotated_sources: Array, params: SubgridTransportParameters, tree: cKDTree | None = None) -> Array:
    """Return one unique target cell for each rotated source point."""
    m = len(rotated_sources)
    n = mesh.cell_count
    if m == 0:
        return np.empty(0, dtype=np.int32)
    if m > n:
        raise ValueError("plate has more source cells than mesh targets")
    if tree is None:
        tree = cKDTree(mesh.centroids)
    k = min(max(int(params.initial_candidate_count), 2), n)
    last_error: Exception | None = None
    while True:
        dist, cand = tree.query(rotated_sources, k=k, workers=query_workers())
        if k == 1:
            dist = dist[:, None]; cand = cand[:, None]
        rows = np.repeat(np.arange(m, dtype=np.int32), k)
        cols = np.asarray(cand, dtype=np.int32).reshape(-1)
        # Sparse matching treats exact zeros as missing edges. Add a tiny
        # deterministic positive floor and tie-breaker.
        cost = np.asarray(dist, dtype=np.float64).reshape(-1)
        cost = np.maximum(cost, 1e-14) + 1e-15 * (cols % 997)
        graph = csr_matrix((cost, (rows, cols)), shape=(m, n))
        try:
            r, c = min_weight_full_bipartite_matching(graph)
            if len(r) == m:
                order = np.argsort(r)
                return np.asarray(c[order], dtype=np.int32)
        except ValueError as exc:
            last_error = exc
        soft_max = min(int(params.max_candidate_count), n)
        hard_max = min(max(soft_max, int(params.absolute_candidate_limit)), n)
        if k >= hard_max:
            raise RuntimeError(
                f"could not build full conservative plate assignment with k<={hard_max}; "
                "refusing catastrophic dense-graph fallback"
            ) from last_error
        k = min(k * 2, hard_max)


def _changed_fraction(tree: cKDTree, source_cells: Array, rotated: Array) -> float:
    if not len(source_cells):
        return 0.0
    _, nearest = tree.query(rotated, k=1, workers=query_workers())
    return float(np.mean(np.asarray(nearest, dtype=np.int32) != np.asarray(source_cells, dtype=np.int32)))


def build_transport_map(
    mesh: SphereMesh,
    system: PlateSystem,
    lithosphere: LithosphereState,
    dt_myr: float,
    transport_state: PlateTransportState,
    params: SubgridTransportParameters | None = None,
) -> TransportMap:
    """Accumulate motion and build the current discrete conservative map.

    ``transport_state`` is mutated in place and also returned inside the map.
    """
    if params is None:
        params = SubgridTransportParameters()
    if dt_myr <= 0.0:
        raise ValueError("dt_myr must be positive")
    pcount = len(system.plates)
    if transport_state.residual_quaternions.shape != (pcount, 4):
        raise ValueError("transport state plate count does not match current system")

    n = mesh.cell_count
    covered = np.zeros((pcount, n), dtype=bool)
    source = np.full((pcount, n), -1, dtype=np.int32)
    source_to_target: list[Array] = []
    execution = current_execution()
    if execution is None:
        tree = cKDTree(mesh.centroids)
        spacing = _median_cell_spacing_rad(mesh)
    else:
        geometry = execution.geometry(mesh)
        tree = geometry.tree
        if geometry.spacing is None:
            regular = bool(mesh.cell_count) and all(
                len(row) == len(mesh.neighbors[0]) for row in mesh.neighbors
            )
            if execution.numeric_kernels and regular and len(mesh.neighbors[0]) > 0:
                if geometry.neighbors is None:
                    geometry.neighbors = np.asarray(mesh.neighbors, dtype=np.int32)
                geometry.spacing = _median_cell_spacing_rad_batched(mesh, geometry.neighbors)
                execution.spacing_kernel_calls += 1
            else:
                geometry.spacing = _median_cell_spacing_rad(mesh)
        spacing = geometry.spacing
    changed_values: list[float] = []
    commits = 0

    owner = np.asarray(lithosphere.cell_plate, dtype=np.int32)
    crust = np.asarray(lithosphere.crust_type, dtype=np.int8)

    def prepare_plate(item):
        pid, plate = item
        src_cells = np.flatnonzero(owner == pid).astype(np.int32)
        if not len(src_cells):
            return pid, src_cells, src_cells.copy(), None, None, None, False
        q_step = quaternion_from_axis_angle(plate.euler_axis, plate.angular_speed_rad_per_myr * float(dt_myr))
        q_total = quaternion_multiply(q_step, transport_state.residual_quaternions[pid])

        cont_cells = src_cells[crust[src_cells] == int(CrustType.CONTINENTAL)]
        relevant = cont_cells if len(cont_cells) >= int(params.continental_preference_min_cells) else src_cells
        relevant_rot = rotate_by_quaternion(mesh.centroids[relevant], q_total)
        changed = _changed_fraction(tree, relevant, relevant_rot)
        dot = np.clip(np.sum(mesh.centroids[relevant] * relevant_rot, axis=1), -1.0, 1.0)
        p75 = float(np.quantile(np.arccos(dot), 0.75)) if len(dot) else 0.0
        enough_motion = (
            changed >= float(params.min_changed_fraction)
            and p75 >= float(params.min_p75_cell_spacing_fraction) * spacing
        )
        forced = (
            transport_state.hold_age_myr[pid] + float(dt_myr) >= float(params.max_hold_myr)
            and changed >= float(params.forced_min_changed_fraction)
        )
        commit = bool(enough_motion or forced)

        if not commit:
            target = src_cells.copy()
            residual = q_total
            hold_age = float(transport_state.hold_age_myr[pid]) + float(dt_myr)
        else:
            desired = rotate_by_quaternion(mesh.centroids[src_cells], q_total)
            target = _optimal_assignment(mesh, desired, params, tree if execution is not None else None)
            represented_r = _fit_rotation(mesh.centroids[src_cells], mesh.centroids[target], params.max_fit_pairs)
            q_fit = quaternion_from_matrix(represented_r)
            residual = quaternion_multiply(q_total, quaternion_conjugate(q_fit))
            hold_age = 0.0
        return pid, src_cells, target, residual, hold_age, changed, commit

    items = enumerate(system.plates)
    prepared = map(prepare_plate, items) if execution is None else execution.ordered_map(prepare_plate, items)
    for pid, src_cells, target, residual, hold_age, changed, commit in prepared:
        if residual is None:
            source_to_target.append(target)
            continue
        changed_values.append(changed)
        transport_state.residual_quaternions[pid] = residual
        transport_state.hold_age_myr[pid] = hold_age
        if commit:
            commits += 1
            transport_state.cumulative_commit_count += 1
        else:
            transport_state.max_hold_age_myr = max(float(transport_state.max_hold_age_myr), float(hold_age))

        # Full one-to-one within this plate by construction.
        covered[pid, target] = True
        source[pid, target] = src_cells
        source_to_target.append(np.asarray(target, dtype=np.int32))

    residual_angles = np.asarray([quaternion_angle_deg(q) for q in transport_state.residual_quaternions], dtype=float)
    diag = TransportDiagnostics(
        committed_plates=int(commits),
        cumulative_commit_count=int(transport_state.cumulative_commit_count),
        mean_residual_angle_deg=float(np.mean(residual_angles)) if len(residual_angles) else 0.0,
        max_residual_angle_deg=float(np.max(residual_angles)) if len(residual_angles) else 0.0,
        max_hold_age_myr=float(transport_state.max_hold_age_myr),
        mean_changed_fraction=float(np.mean(changed_values)) if changed_values else 0.0,
        max_changed_fraction=float(np.max(changed_values)) if changed_values else 0.0,
    )
    return TransportMap(
        covered=covered,
        source=source,
        source_to_target=tuple(source_to_target),
        state=transport_state,
        diagnostics=diag,
    )


def _weighted_quaternion_average(quaternions: Array, weights: Array) -> Array:
    if not len(quaternions):
        return identity_quaternion()
    ref = _normalize_quaternion(quaternions[0])
    aligned = []
    for q in quaternions:
        qn = _normalize_quaternion(q)
        if np.dot(qn, ref) < 0.0:
            qn = -qn
        aligned.append(qn)
    avg = np.average(np.asarray(aligned), axis=0, weights=np.asarray(weights, dtype=float))
    return _normalize_quaternion(avg)


def remap_transport_state(
    old_system: PlateSystem,
    new_system: PlateSystem,
    old_state: PlateTransportState,
) -> PlateTransportState:
    """Carry residual motion through split/merge/absorb topology changes.

    Each new plate receives an overlap-weighted average of residuals from the
    old plates that contributed its cells. Split children therefore inherit the
    parent's motion, while merge products preserve both parents instead of
    resetting transport state.
    """
    old_owner = np.asarray(old_system.cell_plate, dtype=np.int32)
    new_owner = np.asarray(new_system.cell_plate, dtype=np.int32)
    pnew = len(new_system.plates)
    qnew = np.tile(identity_quaternion(), (pnew, 1))
    hnew = np.zeros(pnew, dtype=np.float64)
    for nid in range(pnew):
        cells = np.flatnonzero(new_owner == nid)
        if not len(cells):
            continue
        old_ids, counts = np.unique(old_owner[cells], return_counts=True)
        valid = old_ids < len(old_state.residual_quaternions)
        old_ids = old_ids[valid]; counts = counts[valid]
        if not len(old_ids):
            continue
        qnew[nid] = _weighted_quaternion_average(old_state.residual_quaternions[old_ids], counts)
        hnew[nid] = float(np.average(old_state.hold_age_myr[old_ids], weights=counts))
    return PlateTransportState(
        residual_quaternions=qnew,
        hold_age_myr=hnew,
        cumulative_commit_count=int(old_state.cumulative_commit_count),
        max_hold_age_myr=max(float(old_state.max_hold_age_myr), float(np.max(hnew)) if len(hnew) else 0.0),
    )


__all__ = [
    "SubgridTransportParameters",
    "PlateTransportState",
    "TransportDiagnostics",
    "TransportMap",
    "identity_quaternion",
    "quaternion_multiply",
    "quaternion_from_axis_angle",
    "quaternion_to_matrix",
    "quaternion_angle_deg",
    "rotate_by_quaternion",
    "initialize_transport_state",
    "build_transport_map",
    "remap_transport_state",
]
