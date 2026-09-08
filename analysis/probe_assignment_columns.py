"""Bounded CPU graph-compaction probe; no production hooks or GPU claims.

Capture assignment graphs from one isolated transport call per checkpoint.
Keep edge costs/order, remove only unused target columns, map IDs back.
This is not a full integration or an end-to-end simulation speed benchmark.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import numpy as np
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Keep the research entry point import-compatible; production owns the kernel.
from tectonics.assignment_kernels import compact_matching


def solve(function, graph):
    try:
        return function(graph)
    except ValueError:
        return None


def same(left, right):
    if left is None or right is None:
        return left is None and right is None
    return all(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
               for a, b in zip(left, right))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    from tectonics.checkpoint import load_checkpoint
    from tectonics.cpu_runtime import CpuExecution
    from tectonics.simulation import build_initial_mesh, load_config
    from tectonics.topology import PlateTopologyManager, PlateTopologyParameters
    import tectonics.transport as transport

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1 or args.output.exists():
        parser.error('repeat must be positive and output must be new')
    paths = [args.config, Path(__file__), ROOT / 'tectonics/transport.py',
             ROOT / 'tectonics/assignment_kernels.py']
    paths += [checkpoint / filename for checkpoint in args.checkpoint
              for filename in ('meta.json', 'state.npz')]
    before = {str(path.resolve()): digest(path) for path in paths}
    config = load_config(args.config)
    mesh = build_initial_mesh(config)
    fields = transport.SubgridTransportParameters.__dataclass_fields__
    params = transport.SubgridTransportParameters(**{
        k: v for k, v in config.get('subgrid_transport', {}).items() if k in fields})
    fields = PlateTopologyParameters.__dataclass_fields__
    topology = PlateTopologyParameters(**{
        k: v for k, v in config.get('plate_topology', {}).items() if k in fields})
    report = {'scope': 'isolated solver graphs; compaction cost included, query and graph building excluded',
              'production_enabled': False, 'gpu_used': False, 'cells': mesh.cell_count,
              'repeat': args.repeat, 'inputs_before': before, 'cases': []}
    all_exact = True
    for path in args.checkpoint:
        cp = load_checkpoint(path, PlateTopologyManager(topology))
        if len(cp.state.cell_plate) != mesh.cell_count:
            raise ValueError('Checkpoint/config mesh mismatch')
        graphs = []
        original = transport.min_weight_full_bipartite_matching

        def capture(graph):
            graphs.append(graph.copy())
            return original(graph)

        transport.min_weight_full_bipartite_matching = capture
        try:
            with CpuExecution(1, numeric_kernels=True):
                transport.build_transport_map(mesh, cp.system, cp.state, 4.0,
                                              deepcopy(cp.transport_state), params)
        finally:
            transport.min_weight_full_bipartite_matching = original
        if not graphs:
            raise RuntimeError('No real assignment graphs captured')
        expected = [solve(original, graph) for graph in graphs]
        snapshots = [(g.data.tobytes(), g.indices.tobytes(), g.indptr.tobytes()) for g in graphs]
        times = {'original': [], 'compact': []}
        exact = True
        for repeat in range(args.repeat):
            order = ['original', 'compact'] if repeat % 2 == 0 else ['compact', 'original']
            for mode in order:
                function = original if mode == 'original' else compact_matching
                started = perf_counter()
                actual = [solve(function, graph) for graph in graphs]
                times[mode].append(perf_counter() - started)
                exact = exact and all(same(a, b) for a, b in zip(expected, actual))
        unchanged = snapshots == [(g.data.tobytes(), g.indices.tobytes(), g.indptr.tobytes()) for g in graphs]
        all_exact = all_exact and exact and unchanged
        case = {'checkpoint': str(path.resolve()), 'time_myr': cp.state.time_myr,
                'graphs': [{'sources': g.shape[0], 'targets': g.shape[1],
                            'active_targets': len(np.unique(g.indices)), 'edges': g.nnz}
                           for g in graphs],
                'all_results_byte_exact': exact, 'input_graphs_unchanged': unchanged,
                'seconds': times, 'median_seconds': {k: statistics.median(v) for k, v in times.items()}}
        report['cases'].append(case)
        print(json.dumps({k: v for k, v in case.items() if k not in ('graphs', 'seconds')}), flush=True)
    report['inputs_after'] = {str(path.resolve()): digest(path) for path in paths}
    report['inputs_unchanged'] = before == report['inputs_after']
    report['all_results_byte_exact'] = all_exact
    report['valid'] = all_exact and report['inputs_unchanged']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
