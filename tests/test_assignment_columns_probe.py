import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from analysis.probe_assignment_columns import compact_matching, same, solve


@pytest.mark.parametrize('tied', [False, True])
def test_compaction_retains_global_target_ids_and_order(tied):
    # Sorted global columns, including internal and trailing empty columns.
    cols = [2, 7, 12, 2, 7, 12, 2, 7, 12]
    weights = (np.ones(9) + 1e-15 * (np.asarray(cols) % 997) if tied
               else np.asarray([1., 4., 8., 3., 1., 7., 6., 4., 1.]))
    graph = csr_matrix((weights, (np.repeat(np.arange(3), 3), cols)), shape=(3, 30))
    before = graph.copy()
    assert same(compact_matching(graph), min_weight_full_bipartite_matching(graph))
    assert np.array_equal(graph.data, before.data)
    assert np.array_equal(graph.indices, before.indices)
    assert np.array_equal(graph.indptr, before.indptr)


def test_insufficient_targets_still_fail_instead_of_returning_partial_matching():
    graph = csr_matrix(([1., 2., 3.], ([0, 1, 2], [7, 7, 7])), shape=(3, 30))
    assert solve(min_weight_full_bipartite_matching, graph) is None
    assert solve(compact_matching, graph) is None


def test_connected_hall_failure_is_preserved():
    graph = csr_matrix(([1., 2., 3., 4.], ([0, 1, 2, 2], [2, 2, 7, 12])), shape=(3, 30))
    assert solve(min_weight_full_bipartite_matching, graph) is None
    assert solve(compact_matching, graph) is None


def test_square_and_rectangular_tied_graphs_keep_selected_optimum():
    rng = np.random.default_rng(619)
    for rows in range(1, 9):
        for columns in (rows, rows + 2, rows + 20):
            for _ in range(10):
                mask = rng.random((rows, columns)) < 0.4
                mask[np.arange(rows), np.arange(rows)] = True
                r, c = np.nonzero(mask)
                costs = rng.integers(1, 4, len(r)).astype(float) + 1e-15 * (c % 997)
                graph = csr_matrix((costs, (r, c)), shape=(rows, columns))
                assert same(solve(compact_matching, graph),
                            solve(min_weight_full_bipartite_matching, graph))


def test_runner_context_restores_solver_after_error_and_rejects_nesting():
    from analysis.run_assignment_candidate import AssignmentCandidate
    import tectonics.transport as transport
    original = transport.min_weight_full_bipartite_matching
    with pytest.raises(ValueError, match='probe failure'):
        with AssignmentCandidate() as candidate:
            with pytest.raises(RuntimeError, match='already active'):
                with AssignmentCandidate():
                    pass
            graph = csr_matrix(([1., 2.], ([0, 1], [3, 8])), shape=(2, 20))
            assert same(transport.min_weight_full_bipartite_matching(graph), original(graph))
            assert candidate.calls == 1
            raise ValueError('probe failure')
    assert transport.min_weight_full_bipartite_matching is original
