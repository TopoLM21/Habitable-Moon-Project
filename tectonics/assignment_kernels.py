"""CPU kernels for conservative one-to-one material assignment.

Transport supplies a CSR graph with positive edge costs and no more source
rows than target columns. Compaction changes neither those costs nor their
order: it removes unused target columns and maps the selected IDs back.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


def compact_matching(graph: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Solve a transport graph after removing unused global target columns.

    Sorted global column IDs and the rectangular/square distinction are kept
    to preserve SciPy's selected optimum when costs tie. Selected-index parity
    is empirically regression-tested, not guaranteed across SciPy versions.
    Input CSR arrays are never mutated. Hall failures still raise ValueError
    so transport can expand its candidate search exactly as before.
    """
    columns, inverse = np.unique(graph.indices, return_inverse=True)
    if len(columns) < graph.shape[0]:
        raise ValueError("Fewer candidate targets than source cells")
    # Making a rectangular graph square changes tie handling in the solver.
    # One empty column preserves that distinction without adding an edge.
    width = (max(len(columns), graph.shape[0] + 1)
             if graph.shape[1] > graph.shape[0] else len(columns))
    compact = csr_matrix(
        (graph.data.copy(), inverse.astype(graph.indices.dtype), graph.indptr.copy()),
        shape=(graph.shape[0], width),
    )
    rows, targets = min_weight_full_bipartite_matching(compact)
    return rows, columns[targets]
