from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from tectonics.assignment_kernels import compact_matching
from tectonics.cpu_runtime import CpuExecution, current_execution
from tectonics.lithosphere import initialize_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
import tectonics.transport as transport


def assert_bytes_equal(left, right):
    assert left.dtype == right.dtype
    assert left.shape == right.shape
    assert left.tobytes() == right.tobytes()


def test_production_kernel_keeps_rectangular_tie_selection_and_input_arrays():
    columns = np.tile([2, 7, 12], 3)
    graph = csr_matrix(
        (np.ones(9) + 1e-15 * (columns % 997),
         (np.repeat(np.arange(3), 3), columns)), shape=(3, 30))
    before = graph.copy()
    actual = compact_matching(graph)
    expected = min_weight_full_bipartite_matching(graph)
    for left, right in zip(actual, expected, strict=True):
        assert_bytes_equal(left, right)
    assert actual[1].tolist() == [2, 7, 12]
    for name in ('data', 'indices', 'indptr'):
        assert_bytes_equal(getattr(graph, name), getattr(before, name))


@pytest.mark.parametrize('columns', [[7, 7, 7, 7], [2, 2, 7, 12]])
def test_production_kernel_preserves_insufficient_and_hall_failures(columns):
    graph = csr_matrix(([1., 2., 3., 4.], ([0, 1, 2, 2], columns)), shape=(3, 30))
    before = graph.copy()
    with pytest.raises(ValueError):
        min_weight_full_bipartite_matching(graph)
    with pytest.raises(ValueError):
        compact_matching(graph)
    for name in ('data', 'indices', 'indptr'):
        assert_bytes_equal(getattr(graph, name), getattr(before, name))


@pytest.mark.parametrize('mode', ['no_context', 'default', 'disabled', 'enabled'])
def test_transport_dispatch_is_opt_in_and_retains_original_solver(monkeypatch, mode):
    mesh = build_icosphere(0)
    sources = mesh.centroids[[1, 5, 9]].copy()
    params = transport.SubgridTransportParameters()
    expected = transport._optimal_assignment(mesh, sources, params)
    original = transport.min_weight_full_bipartite_matching
    ordinary_calls = []
    compact_calls = []

    def ordinary(graph):
        ordinary_calls.append(graph.shape)
        return original(graph)

    monkeypatch.setattr(transport, 'min_weight_full_bipartite_matching', ordinary)
    if mode == 'no_context':
        actual = transport._optimal_assignment(mesh, sources, params)
    else:
        options = {} if mode == 'default' else {'assignment_columns': mode == 'enabled'}
        with CpuExecution(**options) as execution:
            calculate = execution.match_assignment

            def compact(graph):
                compact_calls.append(graph.shape)
                return calculate(graph)

            monkeypatch.setattr(execution, 'match_assignment', compact)
            actual = transport._optimal_assignment(mesh, sources, params)
            assert execution.assignment_columns_enabled is (mode == 'enabled')
            assert execution.assignment_calls == len(compact_calls)
    assert_bytes_equal(actual, expected)
    assert bool(compact_calls) is (mode == 'enabled')
    assert bool(ordinary_calls) is (mode != 'enabled')
    assert current_execution() is None
    assert transport.min_weight_full_bipartite_matching is ordinary


@pytest.mark.parametrize('workers', [1, 2])
def test_compact_transport_matches_original_across_ordered_plate_workers(workers):
    mesh = build_icosphere(2)
    system = random_plate_system(mesh, 6, 20260819, 0.2, 0.15, 0.6)
    state = initialize_lithosphere(mesh, system, continental_fraction=0.28, continental_nuclei=4)
    original_memory = transport.initialize_transport_state(6)
    compact_memory = deepcopy(original_memory)
    # Force commits to ensure actual assignment, including shared worker use.
    params = transport.SubgridTransportParameters(
        min_changed_fraction=0., min_p75_cell_spacing_fraction=0.)
    expected = []
    with CpuExecution(workers, assignment_columns=False):
        for _ in range(4):
            expected.append(deepcopy(transport.build_transport_map(
                mesh, system, state, 4., original_memory, params)))
    original_solver = transport.min_weight_full_bipartite_matching
    with CpuExecution(workers, assignment_columns=True) as execution:
        for reference in expected:
            actual = transport.build_transport_map(mesh, system, state, 4., compact_memory, params)
            for name in ('covered', 'source'):
                assert_bytes_equal(getattr(actual, name), getattr(reference, name))
            for left, right in zip(actual.source_to_target, reference.source_to_target, strict=True):
                assert_bytes_equal(left, right)
            for name in ('residual_quaternions', 'hold_age_myr'):
                assert_bytes_equal(getattr(actual.state, name), getattr(reference.state, name))
            assert asdict(actual.diagnostics) == asdict(reference.diagnostics)
        assert execution.assignment_calls >= 4 * len(system.plates)
    assert transport.min_weight_full_bipartite_matching is original_solver
    assert current_execution() is None


@pytest.mark.parametrize('enabled', [False, True])
def test_assignment_expands_candidates_on_matching_failure(enabled):
    mesh = build_icosphere(0)
    sources = mesh.centroids[:3]

    class CandidateTree:
        def __init__(self):
            self.queries = []

        def query(self, points, k, workers):
            self.queries.append(k)
            # The first graph has only two target candidates for three rows.
            targets = np.tile(np.arange(k), (len(points), 1))
            return np.ones_like(targets, dtype=float), targets

    tree = CandidateTree()
    with CpuExecution(assignment_columns=enabled) as execution:
        actual = transport._optimal_assignment(
            mesh, sources, transport.SubgridTransportParameters(initial_candidate_count=2), tree)
        assert execution.assignment_calls == (2 if enabled else 0)
    assert tree.queries == [2, 4]
    assert len(actual) == 3
    assert len(np.unique(actual)) == 3


@pytest.mark.parametrize('workers', [1, 2])
def test_matching_statistics_count_concurrent_attempts_and_are_context_local(workers):
    graph = csr_matrix(([1., 2.], ([0, 1], [3, 8])), shape=(2, 20))
    expected = min_weight_full_bipartite_matching(graph)
    with CpuExecution(workers, assignment_columns=True) as execution:
        results = execution.ordered_map(execution.match_assignment, [graph] * 64)
        for actual in results:
            for left, right in zip(actual, expected, strict=True):
                assert_bytes_equal(left, right)
        assert execution.assignment_calls == 64
        failed = csr_matrix(([1., 2.], ([0, 1], [3, 3])), shape=(2, 20))
        with pytest.raises(ValueError):
            execution.match_assignment(failed)
        report = execution.numerical_report()['assignment_columns']
        assert report['enabled'] is True
        assert report['backend'] == 'cpu_compact_columns'
        assert report['calls'] == 65
        assert report['inclusive_seconds'] >= 0.
    with CpuExecution(assignment_columns=True) as fresh:
        assert fresh.assignment_calls == 0
        assert fresh.assignment_seconds == 0.
