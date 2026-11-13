"""Verify default-mode preservation against the untouched base worktree."""
import json
from pathlib import Path
import subprocess
import sys
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASELINE = Path('/workspace/hm-baseline-831e6e3')
cfg = yaml.safe_load((HERE/'coarse_thermal.yaml').read_text())
cfg['thermal_integration']['enabled'] = False
path = HERE/'legacy_control.yaml'
if path.exists():
    raise FileExistsError(path)
path.write_text(yaml.safe_dump(cfg, sort_keys=False))
outputs = []
for name, cwd in [('legacy_original', BASELINE), ('legacy_current', ROOT)]:
    output = ROOT/'outputs_v131_reconstruction'/name
    if output.exists():
        raise FileExistsError(output)
    with (HERE/f'{name}.log').open('x') as log:
        subprocess.run([sys.executable, 'run_long_evolution_v131.py', '--config', str(path),
                        '--dt', '4', '--end-time', '20', '--output', str(output),
                        '--checkpoint', str(output/'restart')], cwd=cwd,
                       stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True)
    outputs.append(output/'restart')
a, b = outputs
with np.load(a/'state.npz') as x, np.load(b/'state.npz') as y:
    differences = [k for k in x.files if not np.array_equal(x[k], y[k])]
    count = len(x.files)
ma, mb = [json.loads((p/'meta.json').read_text()) for p in outputs]
result = {'baseline_commit': '831e6e3e0d58e352efab575f561aa677595a4b7f',
          'array_count': count, 'array_differences': differences,
          'metadata_equal': ma == mb, 'end_time_myr': 20.}
(HERE/'legacy_comparison.json').write_text(json.dumps(result, indent=2)+'\n')
assert not differences and ma == mb, result
print(json.dumps(result), flush=True)

# A new-mode checkpoint cannot be resumed after silently disabling its ledger.
with (HERE/'disabled_resume.log').open('x') as log:
    result = subprocess.run([
        sys.executable, 'run_long_evolution_v131.py', '--config', str(path),
        '--resume', str(ROOT/'outputs_v131_reconstruction/measured_grid2_dt4/restart'),
        '--end-time', '104', '--dt', '4',
        '--output', str(ROOT/'outputs_v131_reconstruction/disabled_resume')], cwd=ROOT,
        stdout=log, stderr=subprocess.STDOUT, timeout=30)
assert result.returncode != 0
assert 'refusing to discard its history' in (HERE/'disabled_resume.log').read_text()
