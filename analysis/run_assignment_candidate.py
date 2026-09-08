"""Research-only compact-column assignment inside the existing GPU surface runner."""
from __future__ import annotations

import argparse
from contextlib import AbstractContextManager, ExitStack
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.probe_assignment_columns import compact_matching
from analysis.benchmark_cpu_modes import compare_checkpoints
import tectonics.transport as transport


class AssignmentCandidate(AbstractContextManager):
    def __init__(self):
        self.calls = 0
        self.seconds = 0.0
        self.original = None

    def calculate(self, graph):
        started = perf_counter()
        try:
            return compact_matching(graph)
        finally:
            self.calls += 1
            self.seconds += perf_counter() - started

    def __enter__(self):
        if self.original is not None or isinstance(
                getattr(transport.min_weight_full_bipartite_matching, '__self__', None), AssignmentCandidate):
            raise RuntimeError('Assignment candidate is already active')
        self.original = transport.min_weight_full_bipartite_matching
        transport.min_weight_full_bipartite_matching = self.calculate
        return self

    def __exit__(self, *exc):
        transport.min_weight_full_bipartite_matching = self.original
        self.original = None


def main():
    # Boundary replacement must see aliases imported by every runner layer.
    import run_long_evolution_v131
    import run_long_evolution_v131_gpu as runner
    if any(arg in ('-h', '--help') for arg in sys.argv[1:]):
        runner.main()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--with-boundary', action='store_true',
                        help='Also enable the research prepared-CPU boundary candidate')
    options, remaining = parser.parse_known_args()
    if options.output.exists():
        raise ValueError('Research output must be a new directory')
    sys.argv = [sys.argv[0], '--output', str(options.output), *remaining]
    boundary_report = None
    with ExitStack() as stack:
        candidate = stack.enter_context(AssignmentCandidate())
        if options.with_boundary:
            from analysis.run_boundary_candidate import BoundaryCandidateContext
            boundary = stack.enter_context(BoundaryCandidateContext())
        runner.main()
        if options.with_boundary:
            boundary_report = boundary.report()
            if boundary_report['calls'] < 1:
                raise RuntimeError('Boundary candidate did not execute')
    report = options.output / 'render_timings.json'
    data = json.loads(report.read_text(encoding='utf-8'))
    comparison = compare_checkpoints(options.reference, options.output / 'checkpoint')
    data['assignment_candidate'] = {'backend': 'cpu_compact_columns', 'calls': candidate.calls,
                                    'inclusive_solver_seconds': candidate.seconds,
                                    'comparison': comparison, 'production_defaults_changed': False}
    if boundary_report is not None:
        data['boundary_candidate'] = boundary_report
    report.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    if not comparison['exact'] or not candidate.calls:
        raise RuntimeError(f'Candidate failed validation: {comparison}')
    print(f'Assignment candidate: {candidate.calls} calls, exact checkpoint; {candidate.seconds:.6f}s', flush=True)


if __name__ == '__main__':
    main()
