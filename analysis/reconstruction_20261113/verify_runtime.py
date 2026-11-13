"""Bounded new reconstruction checks, never a replay of the lost experiments.

Run from repo root with .venv/bin/python analysis/reconstruction_20261113/verify_runtime.py.
Refuses to overwrite any case directory. Maximum 120 seconds per subprocess.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tectonics.checkpoint import load_checkpoint
from tectonics.topology import PlateTopologyManager, PlateTopologyParameters
from tectonics.sediment import continental_material_ledger_error_km3
from tectonics.plume_magmatism import igneous_ledger_error_km3

HERE = Path(__file__).resolve().parent
OUT = ROOT/'outputs_v131_reconstruction'
BASE = yaml.safe_load((HERE/'coarse_thermal.yaml').read_text())


def checkpoint(case):
    return load_checkpoint(OUT/case/'restart', PlateTopologyManager(PlateTopologyParameters()))


def run_case(name, subdivisions, dt, end=100., resume=None):
    output = OUT/name
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    cfg = deepcopy(BASE)
    cfg['mesh']['subdivisions'] = subdivisions
    path = HERE/f'{name}.yaml'
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite parameters {path}")
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    command = [sys.executable, 'run_long_evolution_v131.py', '--config', str(path),
               '--dt', str(dt), '--end-time', str(end), '--output', str(output),
               '--checkpoint', str(output/'restart')]
    if resume:
        command += ['--resume', str(OUT/resume/'restart')]
    timed = ['/usr/bin/time', '-f', '{"max_rss_kib":%M,"user_s":%U,"system_s":%S,"wall_s":%e}',
             '-o', str(HERE/f'{name}_resources.json'), *command]
    start = time.perf_counter()
    with (HERE/f'{name}.log').open('x') as log:
        result = subprocess.run(timed, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    elapsed = time.perf_counter()-start
    if result.returncode:
        raise RuntimeError(f"{name} returned {result.returncode}; inspect its saved log")
    cp = checkpoint(name)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    item = {
        'case': name, 'source_commit': source, 'subdivisions': subdivisions,
        'dt_myr': dt, 'end_myr': end, 'elapsed_process_s': elapsed,
        'resources': json.loads((HERE/f'{name}_resources.json').read_text()),
        'runtime': json.loads((output/'runtime_stats.json').read_text()),
        'parameters_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'temperature_k': cp.thermal.mantle_temperature_k,
        'plate_count': len(cp.system.plates),
        'topology_events': [e['kind'] for e in cp.events],
        'continental_generated_km3': cp.cycle.cumulative_generated_volume_km3,
        'continental_recycled_km3': cp.cycle.cumulative_recycled_volume_km3,
        'continental_material_ledger_error_km3': continental_material_ledger_error_km3(
            cp.initial_continental_volume_km3, cp.cycle.cumulative_generated_volume_km3,
            cp.state, cp.sediment_budget, cp.cycle.cumulative_recycled_volume_km3),
        'igneous_ledger_error_km3': igneous_ledger_error_km3(cp.plume_magmatism_state),
        'thickness_min_km': float(cp.state.crust_thickness_km.min()),
        'thickness_max_km': float(cp.state.crust_thickness_km.max()),
        'thermal_budget': cp.thermal_integration['history'][-1],
        'full_mass_energy_closure': False,
    }
    (HERE/f'{name}_result.json').write_text(json.dumps(item, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:item[k] for k in ('case','elapsed_process_s','temperature_k','plate_count')}), flush=True)
    return item


def compare_restart():
    first, second = OUT/'grid2_dt4'/'restart', OUT/'restart_second'/'restart'
    with np.load(first/'state.npz') as a, np.load(second/'state.npz') as b:
        differences = [k for k in a.files if not np.array_equal(a[k], b[k])]
        key_match = set(a.files) == set(b.files)
        count = len(a.files)
    ma, mb = (json.loads((p/'meta.json').read_text()) for p in (first, second))
    result = {'array_count': count, 'array_keys_equal': key_match,
              'array_differences': differences, 'metadata_equal': ma == mb,
              'metadata_differences': [k for k in ma if ma[k] != mb.get(k)]}
    (HERE/'restart_comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    if differences or not key_match or ma != mb:
        raise AssertionError(f"Restart differs: {result}")


if __name__ == '__main__':
    for name, sub, dt, end, resume in [
        ('grid2_dt4', 2, 4., 100., None), ('grid2_dt2', 2, 2., 100., None),
        ('grid1_dt4', 1, 4., 100., None), ('grid3_dt4', 3, 4., 100., None),
        ('restart_first', 2, 4., 48., None), ('restart_second', 2, 4., 100., 'restart_first'),
    ]:
        run_case(name, sub, dt, end, resume)
    compare_restart()
