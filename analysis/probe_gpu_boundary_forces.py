"""Isolated exact boundary-force batching experiment; never patches the runner.

The reference is extracted directly from update_plate_dynamics' AST. Candidate
geometry uses its scalar NumPy operations and is cached only for this fixed mesh.
Dynamic state and changing boundary ownership are repacked on every invocation.
Python scalar power is intentionally retained. CUDA only replaces ordered sums;
its inclusive samples include CPU preparation and both PCIe transfers.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
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
from tectonics.lithosphere import CrustType
from tectonics.simulation import build_initial_mesh, load_config
from tectonics.topology import PlateTopologyManager, PlateTopologyParameters

NAMES = ("drive", "boundary_weight", "collision_length", "transform_length",
         "ridge_len", "slab_len", "coll_len", "trans_len", "ridge_factor_sum",
         "ridge_factor_weight", "ridge_factor_min", "ridge_factor_max")


def make_reference():
    """No copied formula: execute the unchanged initialisation and boundary loop."""
    source = ast.parse(inspect.getsource(dynamics.update_plate_dynamics)).body[0]
    start = next(i for i, node in enumerate(source.body) if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "drive")
    end = next(i for i, node in enumerate(source.body) if isinstance(node, ast.For)
               and isinstance(node.target, ast.Name) and node.target.id == "b")
    tree = ast.parse("def reference(mesh, state, boundaries, radius_km, pcount, params, "
                     "mantle_flow=None, thermal_lithosphere_thickness_km=None, subduction_memory=None):\n pass")
    tree.body[0].body = source.body[start:end + 1] + [
        ast.Return(ast.Tuple([ast.Name(name, ast.Load()) for name in NAMES], ast.Load()))]
    ast.fix_missing_locations(tree)
    namespace = dict(vars(dynamics))
    exec(compile(tree, "<unchanged-boundary-loop>", "exec"), namespace)
    return namespace["reference"]


class BoundaryGeometry:
    """Lazy, bounded-to-mesh static edge cache; new boundaries pay cold cost.

    Mesh vertices and centroids must remain immutable for this object's lifetime.
    Use a new BoundaryGeometry for a changed mesh. Midpoints are included in keys
    so a probe of changed/custom records cannot
    accidentally reuse geometry from a different midpoint. Radius is not cached.
    """
    def __init__(self, mesh):
        self.mesh = mesh
        self.edges = {}

    def pack(self, boundaries, radius):
        packed = np.empty((len(boundaries), 7), dtype=np.float64)
        for i, b in enumerate(boundaries):
            key = (b.face_a, b.face_b, b.vertex_u, b.vertex_v,
                   np.asarray(b.midpoint, dtype=np.float64).tobytes())
            value = self.edges.get(key)
            if value is None:
                normal = dynamics._normal_ab(self.mesh, b)
                value = np.concatenate((
                    [dynamics._boundary_length_km(self.mesh, b, 1.0)],
                    np.cross(b.midpoint, -normal), np.cross(b.midpoint, +normal)))
                if not np.any(normal):
                    value[0] = 0.0
                self.edges[key] = value
            packed[i] = value
        packed[:, 0] *= float(radius)
        return packed

    @property
    def numeric_bytes(self):
        return sum(value.nbytes for value in self.edges.values())


def _ordered_sum(values):
    # np.sum uses pairwise grouping. cumsum retains the original left fold.
    return float(np.cumsum(values, dtype=np.float64)[-1]) if len(values) else 0.0


def prepare_contributions(geometry, state, boundaries, radius_km, pcount, params,
                          mantle_flow=None, thermal_lithosphere_thickness_km=None,
                          subduction_memory=None):
    """Exact local contributions, still ordered by boundary then A/B side."""
    # The scalar reference promotes each field access to Python float. A float32
    # batch would round earlier, so this diagnostic explicitly supports only the
    # finite FP64 fields used by saved production checkpoints. Include checking
    # cost in every inclusive timing; do not silently accept lower precision.
    for name in ('tidal_damage', 'crust_age_myr', 'crust_thickness_km',
                 'continental_fraction', 'craton_strength',
                 'mantle_lithosphere_thickness_km', 'mantle_lithosphere_density_anomaly_kg_m3'):
        value = getattr(state, name)
        if value is not None and (not isinstance(value, np.ndarray) or value.dtype != np.float64
                                  or value.shape != (geometry.mesh.cell_count,)
                                  or not np.all(np.isfinite(value))):
            raise ValueError(f'{name} must be a finite FP64 cell array or None')
    geom = geometry.pack(boundaries, radius_km)
    n = len(boundaries)
    faces = np.asarray([(b.face_a, b.face_b) for b in boundaries], dtype=np.intp).reshape(n, 2)
    owners = np.asarray([(b.plate_a, b.plate_b) for b in boundaries], dtype=np.int32).reshape(n, 2)
    kinds = np.fromiter((int(b.boundary_type) for b in boundaries), np.int8, n)
    length = geom[:, 0]
    valid = length > 0.0
    values = np.zeros((n, 2, 6), dtype=np.float64)
    damage = 0.5 * (state.tidal_damage[faces[:, 0]] + state.tidal_damage[faces[:, 1]])
    resistance = np.clip(1.0 - params.tidal_resistance_reduction * damage, 0.15, 1.0)
    ridge_factors = dynamics.plate_ridge_push_factors(geometry.mesh, state, radius_km, pcount, params)
    crust = np.asarray(state.crust_type)[faces]
    divergent = valid & (kinds == BoundaryType.DIVERGENT)
    convergence = valid & (kinds == BoundaryType.CONVERGENT)
    collision = convergence & np.all(crust == CrustType.CONTINENTAL, axis=1)
    transform = valid & (kinds == BoundaryType.TRANSFORM)

    idx = np.flatnonzero(divergent)
    if state.continental_fraction is None:
        fraction = (crust[idx] == CrustType.CONTINENTAL).astype(np.float64)
    else:
        fraction = np.clip(state.continental_fraction[faces[idx]], 0.0, 1.0)
    factors = fraction + (1.0 - fraction) * ridge_factors[owners[idx]]
    tidal_gain = 1.0 + params.tidal_ridge_enhancement * damage[idx]
    strengths = params.ridge_push_weight * factors * tidal_gain[:, None]
    values[idx, 0, :3] = (length[idx] * strengths[:, 0])[:, None] * geom[idx, 1:4]
    values[idx, 1, :3] = (length[idx] * strengths[:, 1])[:, None] * geom[idx, 4:7]
    values[idx, :, 3] = length[idx, None]
    ridge_sum = _ordered_sum(length[idx] * (factors[:, 0] + factors[:, 1]))
    ridge_weight = _ordered_sum(2.0 * length[idx])
    ridge_min = float(np.min(factors)) if len(idx) else np.inf
    ridge_max = float(np.max(factors)) if len(idx) else -np.inf

    idx = np.flatnonzero(collision)
    mean_h = 0.5 * (state.crust_thickness_km[faces[idx, 0]] + state.crust_thickness_km[faces[idx, 1]])
    buoyancy = np.ones(len(idx), dtype=np.float64)
    if mantle_flow is not None:
        buoyancy += params.continental_buoyancy_resistance_gain * np.maximum(
            mean_h / max(float(params.continental_buoyancy_reference_km), 1e-9) - 1.0, 0.0)
    craton = np.ones(len(idx), dtype=np.float64)
    if state.craton_strength is not None:
        mean_craton = 0.5 * (state.craton_strength[faces[idx, 0]] + state.craton_strength[faces[idx, 1]])
        craton += params.craton_collision_resistance_gain * np.clip(mean_craton, 0.0, 1.0)
    values[idx, :, 4] = (length[idx] * resistance[idx] * buoyancy * craton)[:, None]
    values[transform, :, 5] = (length[transform] * resistance[transform])[:, None]

    idx = np.flatnonzero(convergence & ~collision)
    paired_crust = crust[idx]
    ocean_a = paired_crust[:, 0] == CrustType.OCEANIC
    ocean_b = paired_crust[:, 1] == CrustType.OCEANIC
    mantle = (state.mantle_lithosphere_thickness_km is not None and
              state.mantle_lithosphere_density_anomaly_kg_m3 is not None)
    if mantle:
        pair_h = np.maximum(state.mantle_lithosphere_thickness_km[faces[idx]], 0.0)
        pair_drho = np.maximum(state.mantle_lithosphere_density_anomaly_kg_m3[faces[idx]], 0.0)
        proxy = np.maximum(pair_h * pair_drho, 0.0)
    else:
        proxy = np.asarray(state.crust_age_myr)[faces[idx]]
    choose_a = ((ocean_a & ~ocean_b) | (ocean_a & ocean_b & (
        (proxy[:, 0] > proxy[:, 1] + 1e-9) |
        (~(proxy[:, 1] > proxy[:, 0] + 1e-9) & (owners[idx, 0] <= owners[idx, 1])))))
    # Production crust types are binary. Unknown kinds receive no slab force.
    eligible = ((ocean_a & (ocean_b | (paired_crust[:, 1] == CrustType.CONTINENTAL))) |
                (ocean_b & (ocean_a | (paired_crust[:, 0] == CrustType.CONTINENTAL))))
    idx = idx[eligible]
    side = np.where(choose_a[eligible], 0, 1)
    selected_faces = faces[idx, side]
    sub = owners[idx, side]
    over = owners[idx, 1 - side]
    if mantle:
        h = np.maximum(state.mantle_lithosphere_thickness_km[selected_faces], 0.0)
        rho = np.maximum(state.mantle_lithosphere_density_anomaly_kg_m3[selected_faces], 0.0)
        ref = max(float(params.slab_buoyancy_reference_thickness_km) *
                  float(params.slab_buoyancy_reference_density_anomaly_kg_m3), 1e-9)
        ratio = np.clip((h * rho) / ref, 0.02, 2.8)
        # NumPy/CUDA pow need not equal CPython scalar pow. Preserve this call.
        powered = np.fromiter((float(x) ** float(params.slab_buoyancy_exponent) for x in ratio),
                              dtype=np.float64, count=len(ratio))
        strength = params.slab_pull_weight * float(params.slab_buoyancy_calibration_gain) * powered
    else:
        age = np.maximum(state.crust_age_myr[selected_faces], 0.0)
        age_factor = 1.0 + params.slab_age_gain * np.minimum(age / max(params.slab_age_reference_myr, 1e-9), 1.5)
        thermal_factor = 1.0
        if thermal_lithosphere_thickness_km is not None:
            thermal_factor = float(np.clip(max(float(thermal_lithosphere_thickness_km), 1e-9) /
                max(float(params.slab_thermal_reference_km), 1e-9), 0.55, 2.5)) ** float(params.slab_thermal_exponent)
        strength = params.slab_pull_weight * age_factor * thermal_factor
    multipliers = np.fromiter((dynamics.slab_pull_multiplier_for_pair(subduction_memory, a, b)
                              for a, b in zip(sub, over)), dtype=np.float64, count=len(idx))
    active = multipliers > 0.0
    active_idx, active_side = idx[active], side[active]
    toward = np.where(active_side[:, None] == 0, geom[active_idx, 4:7], geom[active_idx, 1:4])
    values[active_idx, active_side, :3] = (
        length[active_idx] * strength[active] * multipliers[active])[:, None] * toward
    values[active_idx, active_side, 3] = length[active_idx]
    scalars = (_ordered_sum(length[divergent]), _ordered_sum(length[idx]),
               _ordered_sum(length[collision]), _ordered_sum(length[transform]),
               ridge_sum, ridge_weight, ridge_min, ridge_max)
    return owners.ravel(), values.reshape(2 * n, 6), scalars


def _result(output, scalars):
    return (output[:, :3], output[:, 3], output[:, 4], output[:, 5], *scalars)


def prepared_cpu(geometry, *args, **kwargs):
    owners, values, scalars = prepare_contributions(geometry, *args, **kwargs)
    output = np.zeros((args[3], 6), dtype=np.float64)
    np.add.at(output, owners, values)
    return _result(output, scalars)


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
        args.config, ROOT / 'tectonics/dynamics.py', Path(__file__))}
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
