"""Create compact, source-stamped evidence from the completed bounded checks."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tectonics.mesh import build_icosphere
from tectonics.simulation import build_prototype
from visualization.raster import rasterize_cells

HERE = Path(__file__).resolve().parent
OUT = ROOT/'outputs_v131_reconstruction'
cases = [json.loads(p.read_text()) for p in sorted(HERE.glob('measured_grid*_result.json'))]
case_by_name = {d['case']: d for d in cases}
a, b = (case_by_name[n] for n in ('measured_grid2_dt4', 'measured_grid2_dt2'))
summary = {
    'status': 'new reconstruction stage 1, not a reproduced lost regional model',
    'runtime_source_commit': a['source_commit'],
    'base_commit': '831e6e3e0d58e352efab575f561aa677595a4b7f',
    'temperature_step_difference_k': abs(a['temperature_k']-b['temperature_k']),
    'continental_generation_dt2_over_dt4': b['continental_generated_km3']/a['continental_generated_km3'],
    'grid_comparison': 'same thermal inventory; mechanically different seeded raster worlds, NOT geometric convergence',
    'full_mass_energy_closure': False,
    'source_work_for_paid_fracture_reconstructed': False,
    'cases': cases,
    'restart': json.loads((HERE/'restart_comparison.json').read_text()),
    'default_mode_regression': json.loads((HERE/'legacy_comparison.json').read_text()),
}
(HERE/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')

# Preserve two restart points and the independent dt=2 endpoint. The identical
# restart endpoint is represented by the bitwise comparison, not another copy.
for name in ('measured_grid2_dt4', 'measured_grid2_dt2', 'measured_restart_first'):
    dest = HERE/'saved_states'/name
    if dest.exists():
        source = OUT/name/'restart'
        for original in source.iterdir():
            copy = dest/original.name
            if not copy.is_file() or copy.read_bytes() != original.read_bytes():
                raise FileExistsError(f'Refusing to overwrite differing {dest}')
    else:
        shutil.copytree(OUT/name/'restart', dest)

cfg = yaml.safe_load((HERE/'coarse_thermal.yaml').read_text())
mesh = build_icosphere(2)
proto = build_prototype(cfg)
with np.load(OUT/'measured_grid2_dt4/restart/state.npz') as z:
    owner = z['state_cell_plate'].copy()
fig = plt.figure(figsize=(12, 8), layout='constrained')
for index, labels, title in [(1, proto.plates.cell_plate, 'Seeded v0.31 regions | start'),
                              (2, owner, 'Moving regions | 100 Myr, 12 remain')]:
    ax = fig.add_subplot(2, 2, index, projection='mollweide')
    x, y, grid = rasterize_cells(mesh, labels, width=360, height=180)
    ax.pcolormesh(x, y, grid, shading='auto', cmap='tab20', vmin=0, vmax=19)
    ax.grid(alpha=.2); ax.set_title(title, fontsize=11)
ax = fig.add_subplot(2, 2, 3)
for name, label in [('measured_grid2_dt4', 'outer dt = 4 Myr'), ('measured_grid2_dt2', 'outer dt = 2 Myr')]:
    meta = json.loads((OUT/name/'restart/meta.json').read_text())
    rows = meta['thermal_integration']['history']
    ax.plot([0]+[r['time_myr'] for r in rows], [1850]+[r['temperature_k'] for r in rows], label=label)
ax.set(xlabel='Time, Myr', ylabel='Mantle temperature, K', title='Same thermal law; no fitted fracture trigger')
ax.grid(alpha=.2); ax.legend()
ax = fig.add_subplot(2, 2, 4)
values = [a['continental_generated_km3']/1e6, b['continental_generated_km3']/1e6]
bars = ax.bar(['4 Myr step', '2 Myr step'], values, color=['#6788aa', '#bb6847'])
ax.bar_label(bars, fmt='%.1f')
ax.set(ylabel='Generated continental crust, million km³', ylim=(0, max(values)*1.2),
       title='Unresolved mechanical/transport sensitivity')
fig.suptitle('NEW reconstruction control — coarse inherited mechanics, not a lunar history', fontsize=13)
fig.savefig(HERE/'overview.png', dpi=120)
plt.close(fig)

baseline = json.loads((HERE/'baseline_manifest.json').read_text())
changed, absent = [], []
for f, digest in baseline['tracked_sha256'].items():
    path = ROOT/f
    if not path.exists():
        absent.append(f)
    elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        changed.append(f)
preservation = {'base_commit': baseline['commit'], 'missing_original_files': absent,
                'intentionally_changed_original_files': changed,
                'unchanged_original_file_count': len(baseline['tracked_sha256'])-len(changed)-len(absent)}
assert not absent and set(changed) == {'tectonics/checkpoint.py', 'run_long_evolution_v123.py'}
(HERE/'preservation.json').write_text(json.dumps(preservation, indent=2)+'\n')

artifact_paths = list((HERE/'saved_states').rglob('*')) + [HERE/'overview.png', HERE/'summary.json']
manifest = {'base_commit': baseline['commit'], 'runtime_source_commit': a['source_commit'],
            'artifact_files': {str(p.relative_to(ROOT)):
                               {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
                               for p in artifact_paths if p.is_file()}}
(HERE/'artifact_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps({'summary': str(HERE/'summary.json'), 'preservation': preservation,
                  'saved_artifact_bytes': sum(v['bytes'] for v in manifest['artifact_files'].values())}))
