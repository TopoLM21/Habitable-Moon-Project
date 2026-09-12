"""Sparse shortest-augmenting-path assignment with an optimality certificate.

This is an opt-in CPU backend for tall candidate lists on a rectangular mesh.
It visits the alternating graph through a heap instead of repeatedly searching
all target columns. A row-minimum warm start leaves only conflicts to resolve.
It still minimizes the original total cost; no greedy result is returned unless
the dual certificate proves it optimal within floating-point roundoff.
Equal-cost assignments need not choose the same targets as SciPy's solver.
"""
from __future__ import annotations

from heapq import heappop, heappush
from itertools import chain
from math import fsum
from time import monotonic
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix


def sparse_minimum_matching(
    graph: csr_matrix, progress: Callable[..., None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a minimum-cost full row matching of a positive canonical CSR.

    ``progress(phase, **stats)`` receives scalar diagnostics, including progress
    inside the Python search. ``ValueError`` means structural infeasibility and
    allows callers to expand their candidate graph. Invalid input or a failed
    numerical certificate raises ``RuntimeError`` instead.

    Dual variables obey ``u[row] + v[col] <= cost`` and ``v <= 0``. Matching
    edges remain tight and unmatched columns retain zero potential. These
    conditions certify optimality for rectangular, not just square, graphs.
    Negative reduced costs are clamped only within 64 machine epsilons of the
    largest absolute cost/potential seen; larger violations abort the solve.
    """
    started = monotonic()
    last_report = started
    stats: dict[str, int | float | str] = {
        "backend": "sparse_ssp", "solver_revision": "no_cardinality_precheck_v2",
    }

    def report(phase: str, *, force: bool = False, **values: int | float) -> None:
        nonlocal last_report
        if progress is None:
            return
        now = monotonic()
        if force or now - last_report >= 1.0:
            last_report = now
            progress(phase, **stats, **values, elapsed_seconds=now - started)

    if not isinstance(graph, csr_matrix):
        raise RuntimeError("sparse assignment requires a CSR matrix")
    try:
        graph.check_format(full_check=True)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("invalid sparse assignment CSR structure") from exc
    if not graph.has_canonical_format:
        raise RuntimeError("sparse assignment requires sorted, unique row edges")
    if not np.isrealobj(graph.data):
        raise RuntimeError("sparse assignment costs must be real numbers")
    costs = np.asarray(graph.data, dtype=np.float64)
    if not np.all(np.isfinite(costs)) or np.any(costs <= 0.0):
        raise RuntimeError("sparse assignment costs must be finite and positive")
    m, original_n = graph.shape
    stats.update(rows=m, original_columns=original_n, edges=int(graph.nnz))
    report("prepare", force=True)
    if m == 0:
        report("complete", force=True, matched=0, augmentations=0)
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if m > original_n or np.any(np.diff(graph.indptr) == 0):
        raise ValueError("no full matching: too few targets or an empty row")

    # The inverse map is monotone, so canonical row order is retained. Unused
    # mesh columns never enter any per-search data structure.
    global_columns, columns = np.unique(graph.indices, return_inverse=True)
    n = len(global_columns)
    if n < m:
        raise ValueError("no full matching: fewer candidate targets than rows")
    indptr = graph.indptr
    stats["columns"] = n
    u = np.minimum.reduceat(costs, indptr[:-1])
    v = np.zeros(n, dtype=np.float64)
    row_match = np.full(m, -1, dtype=np.int64)
    col_match = np.full(n, -1, dtype=np.int64)
    matched_edge = np.full(m, -1, dtype=np.int64)
    matched = 0
    report("warm_start", force=True, matched=matched)
    for row in range(m):
        begin, end = int(indptr[row]), int(indptr[row + 1])
        targets = columns[begin:end]
        tight = np.flatnonzero((costs[begin:end] == u[row]) & (col_match[targets] < 0))
        if len(tight):
            edge = begin + int(tight[0])
            col = int(columns[edge])
            row_match[row], col_match[col], matched_edge[row] = col, row, edge
            matched += 1
        if row % 1024 == 0:
            report("warm_start", matched=matched, processed_rows=row + 1)
    stats["warm_start_matched"] = matched
    report("warm_start", force=True, matched=matched, remaining_rows=m - matched)

    # Do not run a separate cardinality solve: diagnostics on subdivision 6
    # found that precheck taking minutes before this search took ~1 second.
    # A shortest-path search that exhausts all reachable targets without a
    # free one already proves that this graph cannot match every source row.

    epsilon = np.finfo(np.float64).eps
    scale = float(np.max(costs))
    minimum_tolerance = float(np.nextafter(0.0, 1.0))
    tolerance = max(64.0 * epsilon * scale, minimum_tolerance)
    # Generation stamps avoid clearing n-column arrays for every free row.
    col_seen = np.zeros(n, dtype=np.int64)
    col_settled = np.zeros(n, dtype=np.int64)
    col_distance = np.full(n, np.inf, dtype=np.float64)
    row_distance = np.zeros(m, dtype=np.float64)
    predecessor_row = np.full(n, -1, dtype=np.int64)
    predecessor_edge = np.full(n, -1, dtype=np.int64)
    augmentations = 0
    scanned_edges = 0
    searched_rows = 0
    report("augment", force=True, matched=matched, augmentations=augmentations)
    for root in np.flatnonzero(row_match < 0):
        root = int(root)
        generation = augmentations + 1
        heap: list[tuple[float, int]] = []
        reached_rows = [root]
        settled_columns: list[int] = []
        row_distance[root] = 0.0
        row = root
        while True:
            begin, end = int(indptr[row]), int(indptr[row + 1])
            targets = columns[begin:end]
            reduced = costs[begin:end] - u[row] - v[targets]
            if not np.all(np.isfinite(reduced)) or float(np.min(reduced)) < -tolerance:
                raise RuntimeError("sparse assignment lost dual feasibility during search")
            # Only arithmetic roundoff is accepted, never an approximate edge.
            np.maximum(reduced, 0.0, out=reduced)
            distances = row_distance[row] + reduced
            if not np.all(np.isfinite(distances)):
                raise RuntimeError("sparse assignment distance overflow")
            improved = np.flatnonzero(
                (col_settled[targets] != generation)
                & ((col_seen[targets] != generation) | (distances < col_distance[targets]))
            )
            for offset in improved:
                col = int(targets[offset])
                distance = float(distances[offset])
                col_seen[col] = generation
                col_distance[col] = distance
                predecessor_row[col] = row
                predecessor_edge[col] = begin + int(offset)
                # Column id resolves equal distances deterministically.
                heappush(heap, (distance, col))
            scanned_edges += end - begin
            searched_rows += 1
            report("augment", matched=matched, augmentations=augmentations,
                   searched_rows=searched_rows, scanned_edges=scanned_edges,
                   current_search_rows=len(reached_rows), frontier_entries=len(heap))
            popped = 0
            while heap:
                distance, col = heappop(heap)
                if col_settled[col] != generation and distance == col_distance[col]:
                    break
                popped += 1
                if popped % 4096 == 0:
                    report("augment", matched=matched, augmentations=augmentations,
                           searched_rows=searched_rows, scanned_edges=scanned_edges,
                           current_search_rows=len(reached_rows), frontier_entries=len(heap))
            else:
                # Every reachable target is matched, so the reached rows
                # (including the free root) outnumber their target neighbors.
                # This is structural infeasibility: let transport expand k.
                raise ValueError("candidate graph has no full row matching")
            col_settled[col] = generation
            if col_match[col] < 0:
                endpoint, shortest_distance = col, distance
                break
            settled_columns.append(col)
            row = int(col_match[col])
            row_distance[row] = distance
            reached_rows.append(row)

        # All vertices with distance < D were settled before the free endpoint.
        # Vertices at D need no update. Reverse matched edges have zero length,
        # so each reached matched row and its column receive opposite shifts.
        reached = np.asarray(reached_rows, dtype=np.int64)
        settled = np.asarray(settled_columns, dtype=np.int64)
        u[reached] += shortest_distance - row_distance[reached]
        v[settled] -= shortest_distance - col_distance[settled]
        if not np.all(np.isfinite(u[reached])) or not np.all(np.isfinite(v[settled])):
            raise RuntimeError("sparse assignment potential overflow")
        scale = max(scale, float(np.max(np.abs(u[reached]))))
        if len(settled):
            scale = max(scale, float(np.max(np.abs(v[settled]))))
        tolerance = max(64.0 * epsilon * scale, minimum_tolerance)

        col = endpoint
        while True:
            row = int(predecessor_row[col])
            previous_col = int(row_match[row])
            row_match[row] = col
            col_match[col] = row
            matched_edge[row] = predecessor_edge[col]
            if previous_col < 0:
                break
            col = previous_col
        matched += 1
        augmentations += 1
        report("augment", force=augmentations % 128 == 0, matched=matched,
               augmentations=augmentations, searched_rows=searched_rows,
               scanned_edges=scanned_edges)

    # Validate the primal and dual together; a bijection alone cannot certify a
    # minimum-cost assignment. These are linear final passes over the graph,
    # not full target scans inside the shortest-path frontier.
    report("certificate", force=True, matched=matched, augmentations=augmentations)
    rows = np.arange(m, dtype=np.int64)
    if (np.any(row_match < 0) or np.any(row_match >= n)
            or np.any(matched_edge < indptr[:-1]) or np.any(matched_edge >= indptr[1:])):
        raise RuntimeError("sparse assignment returned an incomplete matching")
    if not np.array_equal(col_match[row_match], rows):
        raise RuntimeError("sparse assignment returned duplicate or inconsistent targets")
    if not np.array_equal(columns[matched_edge], row_match):
        raise RuntimeError("sparse assignment returned a target outside its candidate row")
    edge_rows = np.repeat(rows, np.diff(indptr))
    reduced = costs - u[edge_rows] - v[columns]
    matched_slack = costs[matched_edge] - u - v[row_match]
    if not np.all(np.isfinite(reduced)) or not np.all(np.isfinite(matched_slack)):
        raise RuntimeError("sparse assignment certificate contains non-finite slacks")
    unused = col_match < 0
    free_error = float(np.max(np.abs(v[unused]))) if np.any(unused) else 0.0
    max_error = max(0.0, -float(np.min(reduced)), float(np.max(v)),
                    float(np.max(np.abs(matched_slack))), free_error)
    try:
        primal = fsum(float(value) for value in costs[matched_edge])
        dual = fsum(float(value) for value in chain(u, v))
        absolute_potentials = fsum(abs(float(value)) for value in chain(u, v))
    except OverflowError as exc:
        raise RuntimeError("sparse assignment objective certificate overflow") from exc
    gap = primal - dual
    # Tight-edge error plus unused-column error bounds the objective gap.
    # Include a separate sum-rounding allowance when potentials cancel.
    gap_bound = ((m + int(np.count_nonzero(unused))) * tolerance
                 + 64.0 * epsilon * (abs(primal) + absolute_potentials))
    if not np.isfinite(gap) or not np.isfinite(gap_bound):
        raise RuntimeError("sparse assignment objective certificate overflow")
    if max_error > tolerance or abs(gap) > gap_bound:
        raise RuntimeError(
            "sparse assignment failed its optimality certificate: "
            f"error={max_error:.6g}, tolerance={tolerance:.6g}, "
            f"gap={gap:.6g}, gap_bound={gap_bound:.6g}"
        )
    report("complete", force=True, matched=matched, augmentations=augmentations,
           searched_rows=searched_rows, scanned_edges=scanned_edges,
           certificate_max_error=max_error, certificate_tolerance=tolerance,
           primal_cost=primal, dual_cost=dual, primal_dual_gap=gap,
           primal_dual_gap_bound=gap_bound)
    return rows, np.asarray(global_columns[row_match], dtype=np.int64)
