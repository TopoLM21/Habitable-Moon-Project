"""Isolated exact boundary-force batching experiment; never patches the runner.

The reference is the unchanged production scalar helper. Candidate geometry
uses its scalar NumPy operations and is cached only for this fixed mesh.
Dynamic state and changing boundary ownership are repacked on every invocation.
Python scalar power is intentionally retained. CUDA only replaces ordered sums;
its inclusive samples include CPU preparation and both PCIe transfers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tectonics.dynamics as dynamics
from tectonics.checkpoint import load_checkpoint
from tectonics.cpu_runtime import CpuExecution
from tectonics.kinematics import BoundaryType, classify_boundaries
from tectonics.simulation import build_initial_mesh, load_config
from tectonics.topology import PlateTopologyManager, PlateTopologyParameters

from tectonics.boundary_kernels import (
    BoundaryGeometry, NAMES, prepare_contributions, prepared_cpu, _result,
)


def make_reference():
    """Return the unchanged scalar boundary helper, independent of dispatch."""
    return dynamics._boundary_force_terms_reference


class GpuOrderedReduction:
    def __init__(self, device=0):
        import cupy as cp
        self.cp = cp
        self.device = device
        cp.cuda.Device(device).use()
        self.kernel = cp.RawKernel(r'''
        extern "C" __global__ void reduce_ordered(const int* owner, const double* values,
                                                  double* out, int count, int plates) {
            int component = blockDim.x * blockIdx.x + threadIdx.x;
            if (component >= plates * 6) return;
            int plate = component / 6, column = component % 6;
            double result = 0.0;
            for (int event = 0; event < count; ++event)
                if (owner[event] == plate) result += values[event * 6 + column];
            out[component] = result;
        }
        ''', 'reduce_ordered', options=('--fmad=false',))

    def reduce(self, owners, values, plates):
        cp = self.cp
        cp.cuda.Device(self.device).use()
        output = cp.empty((plates, 6), dtype=cp.float64)
        self.kernel(((plates * 6 + 127) // 128,), (128,),
                    (cp.asarray(owners), cp.asarray(values), output, np.int32(len(owners)), np.int32(plates)))
        return output.get()

    def calculate(self, geometry, *args, **kwargs):
        owners, values, scalars = prepare_contributions(geometry, *args, **kwargs)
        return _result(self.reduce(owners, values, args[3]), scalars)


def comparison(expected, actual):
    result = {}
    for name, left, right in zip(NAMES, expected, actual):
        a, b = np.asarray(left), np.asarray(right)
        exact = a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
        mask = a != b
        result[name] = {"byte_exact": exact, "different_values": int(np.count_nonzero(mask)),
                        "max_absolute_difference": float(np.max(np.abs(a[mask] - b[mask]), initial=0.0))}
    return result


def _params(kind, values):
    return kind(**{name: values[name] for name in kind.__dataclass_fields__ if name in values})


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_hashes(state):
    return {name: hashlib.sha256(value.tobytes()).hexdigest()
            for name in state.__dataclass_fields__
            if isinstance(value := getattr(state, name), np.ndarray)}


def _require_exact(result, label):
    if not all(item['byte_exact'] for item in result.values()):
        raise RuntimeError(f'{label} differs: {result}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeat', type=int, default=9)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--gpu-device', type=int, default=0)
    args = parser.parse_args()
    if args.repeat < 1 or args.output.exists():
        parser.error('repeat must be positive and output must not exist')
    provenance = {str(path.resolve()): _hash(path) for path in (
        args.config, ROOT / 'tectonics/dynamics.py',
        ROOT / 'tectonics/boundary_kernels.py', Path(__file__))}
    config = load_config(args.config)
    params = _params(dynamics.DynamicsParameters, config.get('plate_dynamics', {}))
    topology = _params(PlateTopologyParameters, config.get('plate_topology', {}))
    mesh = build_initial_mesh(config)
    geometry = BoundaryGeometry(mesh)
    reference = make_reference()
    gpu = None
    gpu_init = None
    if not args.cpu_only:
        start = perf_counter()
        gpu = GpuOrderedReduction(args.gpu_device)
        gpu_init = perf_counter() - start
    report = {'scope': 'isolated boundary loop plus ridge factors, excludes classification and full dynamics',
              'production_enabled': False, 'cells': mesh.cell_count,
              'reference_sha256': _hash(ROOT / 'tectonics/dynamics.py'),
              'config': str(args.config.resolve()), 'config_sha256': _hash(args.config),
              'source_sha256': provenance,
              'gpu_initialization_seconds': gpu_init,
              'gpu_scope': 'CPU dynamic preparation + ordered GPU reduction + all uploads/download; allocation included',
              'geometry_policy': 'lazy per fixed mesh edge; newly appearing edges incur exact scalar CPU construction',
              'cases': []}
    if gpu is not None:
        name = gpu.cp.cuda.runtime.getDeviceProperties(args.gpu_device)['name']
        report['gpu_name'] = name.decode() if isinstance(name, bytes) else str(name)
    with CpuExecution(numeric_kernels=True):
        for path in args.checkpoint:
            hashes = {name: _hash(path / name) for name in ('meta.json', 'state.npz')}
            checkpoint = load_checkpoint(path, PlateTopologyManager(topology))
            state, system = checkpoint.state, checkpoint.system
            state_hashes = _state_hashes(state)
            if len(state.cell_plate) != mesh.cell_count:
                raise ValueError('Checkpoint/config mesh sizes differ')
            radius = float(config['moon']['radius_km'])
            classification = config.get('classification', {})
            boundaries = classify_boundaries(mesh, system, radius,
                classification.get('normal_threshold_km_per_myr', 4.0),
                classification.get('inactive_speed_km_per_myr', 1.0))
            inputs = (state, boundaries, radius, len(system.plates), params)
            kwargs = {'mantle_flow': checkpoint.mantle_flow, 'subduction_memory': checkpoint.subduction_memory,
                      'thermal_lithosphere_thickness_km': checkpoint.thermal.thermal_lithosphere_thickness_km}
            expected = reference(mesh, *inputs, **kwargs)
            cache_before = len(geometry.edges)
            start = perf_counter()
            first_cpu = prepared_cpu(geometry, *inputs, **kwargs)
            cold = perf_counter() - start
            case = {'checkpoint': str(path.resolve()), 'time_myr': state.time_myr,
                    'boundaries': len(boundaries), 'plates': len(system.plates),
                    'boundary_kinds': {kind.name: sum(b.boundary_type == kind for b in boundaries) for kind in BoundaryType},
                    'new_cached_edges': len(geometry.edges) - cache_before,
                    'cached_edges': len(geometry.edges), 'geometry_numeric_bytes': geometry.numeric_bytes,
                    'first_prepared_cpu_including_new_geometry_seconds': cold,
                    'first_prepared_cpu_comparison': comparison(expected, first_cpu), 'input_sha256': hashes}
            _require_exact(case['first_prepared_cpu_comparison'], f'{path.name} first prepared CPU')
            calls = {'reference': lambda: reference(mesh, *inputs, **kwargs),
                     'prepared_cpu': lambda: prepared_cpu(geometry, *inputs, **kwargs)}
            if gpu is not None:
                start = perf_counter()
                first_gpu = gpu.calculate(geometry, *inputs, **kwargs)
                case['gpu_first_inclusive_seconds'] = perf_counter() - start
                case['gpu_first_comparison'] = comparison(expected, first_gpu)
                _require_exact(case['gpu_first_comparison'], f'{path.name} first GPU')
                calls['gpu_inclusive'] = lambda: gpu.calculate(geometry, *inputs, **kwargs)
            samples = {name: [] for name in calls}
            comparisons = {}
            for iteration in range(args.repeat):
                order = list(calls)
                if iteration % 2:
                    order.reverse()
                for name in order:
                    start = perf_counter()
                    actual = calls[name]()
                    samples[name].append(perf_counter() - start)
                    found = comparison(expected, actual)
                    comparisons[name] = found
                    _require_exact(found, f'{path.name} {name}')
            for name in calls:
                case[name] = {'wall_seconds': samples[name], 'median_seconds': statistics.median(samples[name]),
                              'comparison': comparisons[name]}
            owners, values, _ = prepare_contributions(geometry, *inputs, **kwargs)
            reductions = {'cpu': [], 'gpu': []}
            prep_samples = []
            for _ in range(args.repeat):
                start = perf_counter()
                prepare_contributions(geometry, *inputs, **kwargs)
                prep_samples.append(perf_counter() - start)
                start = perf_counter()
                output = np.zeros((len(system.plates), 6), dtype=np.float64)
                np.add.at(output, owners, values)
                reductions['cpu'].append(perf_counter() - start)
                if gpu is not None:
                    start = perf_counter()
                    gpu.reduce(owners, values, len(system.plates))
                    reductions['gpu'].append(perf_counter() - start)
            case['warm_cpu_preparation_median_seconds'] = statistics.median(prep_samples)
            case['reduction_only_median_seconds'] = {name: statistics.median(values) for name, values in reductions.items() if values}
            case['dynamic_upload_bytes'] = owners.nbytes + values.nbytes
            case['download_bytes'] = len(system.plates) * 6 * 8
            case['input_files_unchanged'] = hashes == {name: _hash(path / name) for name in hashes}
            case['loaded_state_arrays_unchanged'] = state_hashes == _state_hashes(state)
            if not case['input_files_unchanged']:
                raise RuntimeError('Input checkpoint changed')
            if not case['loaded_state_arrays_unchanged']:
                raise RuntimeError('Loaded checkpoint arrays changed')
            for name in calls:
                case[name]['loop_speedup_vs_reference'] = case['reference']['median_seconds'] / case[name]['median_seconds']
            report['cases'].append(case)
            print(json.dumps(case, indent=2), flush=True)
    report['sources_unchanged'] = provenance == {path: _hash(Path(path)) for path in provenance}
    if not report['sources_unchanged']:
        raise RuntimeError('Configuration or probe/reference source changed during measurement')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(f'Report: {args.output.resolve()}')


if __name__ == '__main__':
    main()
