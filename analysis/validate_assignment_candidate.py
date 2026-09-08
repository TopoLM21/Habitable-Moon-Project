"""Sequential, exact-validated full-process assignment trials; fresh outputs only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.benchmark_cpu_modes import compare_checkpoints
from analysis.benchmark_render_modes import compare_pngs
from analysis.validate_gpu_surface import (
    checkpoint_hashes, child_environment, require_surface_execution, sha256_file, telemetry,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'resume', 'reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--end-time', type=float, required=True)
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--first-repeat', type=int, default=2)
    parser.add_argument('--candidate-only', action='store_true')
    parser.add_argument('--with-boundary', action='store_true',
                        help='Combine assignment compaction with the boundary candidate')
    parser.add_argument('--production', action='store_true',
                        help='Validate ordinary runner flags instead of research replacements')
    parser.add_argument('--cpu-workers', type=int, choices=range(1, 33), default=1)
    args = parser.parse_args()
    for name in ('config', 'resume', 'reference', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    root = (ROOT / 'results/gpu_surface').resolve()
    if args.output.exists() or args.output == root or not args.output.is_relative_to(root):
        parser.error('output must be new and beneath results/gpu_surface')
    start = json.loads((args.resume / 'meta.json').read_text(encoding='utf-8'))['time_myr']
    end = json.loads((args.reference / 'meta.json').read_text(encoding='utf-8'))['time_myr']
    steps = (args.end_time - start) / 4
    if (args.repeat < 1 or args.first_repeat < 1 or not math.isfinite(steps)
            or steps <= 0 or steps != int(steps) or args.end_time != end):
        parser.error('invalid interval, independent endpoint or repeat count')
    sources = [Path(__file__), ROOT / 'analysis/run_assignment_candidate.py',
               ROOT / 'analysis/probe_assignment_columns.py', ROOT / 'tectonics/transport.py',
               ROOT / 'tectonics/gpu_runtime.py', ROOT / 'tectonics/gpu_surface.py',
               ROOT / 'tectonics/sediment.py', ROOT / 'run_long_evolution_v131_gpu.py',
               ROOT / 'tectonics/assignment_kernels.py', ROOT / 'tectonics/cpu_runtime.py',
               ROOT / 'run_long_evolution_v131_cpu.py']
    if args.with_boundary:
        sources += [ROOT / 'analysis/run_boundary_candidate.py',
                    ROOT / 'analysis/probe_gpu_boundary_forces.py', ROOT / 'tectonics/dynamics.py',
                    ROOT / 'tectonics/boundary_kernels.py']

    def fingerprints():
        return {'config': sha256_file(args.config), 'input': checkpoint_hashes(args.resume),
                'reference': checkpoint_hashes(args.reference),
                'sources': {str(p.relative_to(ROOT)): sha256_file(p) for p in sources}}

    report = {'status': 'running', 'scope': 'Full processes, including startup, CUDA, I/O and final maps',
              'new_gpu_solver': False, 'boundary_candidate_enabled': args.with_boundary,
              'production_runner': args.production, 'cpu_workers': args.cpu_workers,
              'start_myr': start, 'end_myr': end, 'steps': int(steps),
              'reference': str(args.reference), 'fingerprints_before': fingerprints(), 'runs': []}
    args.output.mkdir(parents=True)

    def save():
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    save()
    try:
        for repeat in range(args.first_repeat, args.first_repeat + args.repeat):
            modes = ['original', 'candidate'] if repeat % 2 else ['candidate', 'original']
            if args.candidate_only:
                modes = ['candidate']
            for mode in modes:
                case = args.output / f'{mode}_{repeat}'
                runner = ROOT / ('analysis/run_assignment_candidate.py' if mode == 'candidate' and not args.production
                                 else 'run_long_evolution_v131_gpu.py')
                command = [sys.executable, str(runner), '--gpu-surface', '--cpu-workers', str(args.cpu_workers),
                           '--render-workers', '4', '--cell-kernels', '--config', str(args.config),
                           '--resume', str(args.resume), '--end-time', str(end), '--dt', '4',
                           '--output', str(case), '--checkpoint', str(case / 'checkpoint')]
                if args.production:
                    command += ['--assignment-columns' if mode == 'candidate' else '--no-assignment-columns',
                                '--boundary-forces' if mode == 'candidate' and args.with_boundary
                                else '--no-boundary-forces']
                elif mode == 'candidate':
                    command += ['--reference', str(args.reference)]
                    if args.with_boundary:
                        command += ['--with-boundary']
                row = {'mode': mode, 'repeat': repeat, 'command': command,
                       'telemetry_before': telemetry(0)}
                report['runs'].append(row)
                save()
                print(f'START {case.name}', flush=True)
                with (args.output / f'{case.name}.log').open('x', encoding='utf-8') as log:
                    began = perf_counter()
                    result = subprocess.run(command, cwd=ROOT, env=child_environment(args.output, ROOT),
                                            stdout=log, stderr=subprocess.STDOUT)
                    row['wall_seconds'] = perf_counter() - began
                row['returncode'] = result.returncode
                row['telemetry_after'] = telemetry(0)
                if result.returncode:
                    raise RuntimeError(f'{case.name} failed; see its log')
                execution = json.loads((case / 'render_timings.json').read_text(encoding='utf-8'))
                require_surface_execution(execution.get('gpu_execution', {}))
                if execution['gpu_execution']['surface_pipeline']['calls'] != int(steps):
                    raise RuntimeError('Incomplete GPU surface execution')
                if args.production:
                    numerical = execution.get('numerical_execution', {})
                    for key, backend, enabled in (
                            ('assignment_columns', 'cpu_compact_columns', mode == 'candidate'),
                            ('boundary_forces', 'prepared_cpu_boundary_forces', mode == 'candidate' and args.with_boundary)):
                        info = numerical.get(key, {})
                        if (info.get('backend') != backend or info.get('enabled') is not enabled
                                or not isinstance(info.get('calls'), int)
                                or (enabled and info['calls'] < 1)
                                or (not enabled and info['calls'] != 0)
                                or (enabled and key == 'boundary_forces' and info['calls'] != int(steps))):
                            raise RuntimeError(f'Missing, disabled or inconsistent production {key} execution')
                elif mode == 'candidate':
                    candidate = execution.get('assignment_candidate', {})
                    if candidate.get('backend') != 'cpu_compact_columns' or candidate.get('calls', 0) < 1:
                        raise RuntimeError('Missing assignment candidate provenance')
                    if args.with_boundary:
                        boundary = execution.get('boundary_candidate', {})
                        if (boundary.get('backend') != 'prepared_cpu_boundary_forces'
                                or boundary.get('calls') != int(steps)):
                            raise RuntimeError('Missing or incomplete boundary candidate provenance')
                row['execution_report'] = execution
                row['checkpoint'] = compare_checkpoints(args.reference, case / 'checkpoint')
                row['png'] = compare_pngs(args.reference.parent, case)
                if not row['checkpoint']['exact'] or not row['png']['png_exact'] or not row['png']['png_count']:
                    raise RuntimeError(f'Exact validation failed: {case.name}')
                print(f"DONE {case.name}: {row['wall_seconds']:.3f}s; exact checkpoint and PNG", flush=True)
                save()
        report['status'] = 'exact_validation_passed'
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        try:
            report['fingerprints_after'] = fingerprints()
            report['inputs_and_sources_unchanged'] = report['fingerprints_before'] == report['fingerprints_after']
            if not report['inputs_and_sources_unchanged']:
                raise RuntimeError('Input, reference or code changed')
        except Exception as exc:
            report.update(status='failed', inputs_and_sources_unchanged=False, error=str(exc))
        if report['status'] == 'exact_validation_passed' and not args.candidate_only:
            medians = {mode: statistics.median(r['wall_seconds'] for r in report['runs'] if r['mode'] == mode)
                       for mode in ('original', 'candidate')}
            report['median_wall_seconds'] = medians
            report['observed_time_reduction_percent'] = 100 * (1 - medians['candidate'] / medians['original'])
        save()
    print(f"{report['status']}: {args.output / 'summary.json'}", flush=True)
    return 0 if report['status'] == 'exact_validation_passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
