"""Verify preserved evidence and all new saves, then emit an immutable manifest."""
import hashlib
import json
import subprocess
import sys
import numpy as np
import scipy
from verify import HERE, ROOT
from tectonics.strain_exchange import StrainExchange

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protected=0
for p in (ROOT/'analysis').glob('*/artifact_manifest.json'):
    if p.parent==HERE:continue
    for name,item in json.loads(p.read_text())['artifact_files'].items():
        assert sha(ROOT/name)==(item['sha256'] if isinstance(item,dict) else item),name
        protected+=1
states=list((HERE/'states').iterdir());snapshots=0
for path in states:
    obj=StrainExchange.load(path);raw=json.loads((path/'meta.json').read_text())
    with np.load(path/'surface.npz') as z:
        raw['coupled']['surface']['arrays']={k:z[k].tolist() for k in z.files}
    assert raw==obj.checkpoint(),str(path)
    snapshots+=sum(p.stat().st_size for p in path.iterdir())
reports=0
for path in HERE.glob('*.json'):
    d=json.loads(path.read_text())
    if not isinstance(d,dict) or 'final' not in d:continue
    o=StrainExchange.load(HERE/'states'/d['case']);assert o.diagnostics()==d['final']
    assert d['status']=='complete';reports+=1
with (HERE/'preservation.json').open('x') as f:
    json.dump(dict(protected_artifacts=protected,mismatches=[],complete_case_reports=reports,
        exact_saved_state_loads=len(states),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        runtime_versions=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__)),f,indent=2)
files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p))
    for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
    and p.suffix!='.pyc' and p.name!='artifact_manifest.json'}
runtime=['tectonics/strain_exchange.py','tectonics/deformation_exchange.py',
    'tectonics/regional_deformation.py','tectonics/conforming_surface.py',
    'tectonics/material_exchange.py','tests/test_strain_exchange.py']
with (HERE/'artifact_manifest.json').open('x') as f:
    json.dump(dict(runtime_commit='278c6354b6610ba57fe42019aa5e8d8e2fe4db10',
        parent_commit='bc2ee6aa2b6ea9c824f78f4dba10e1be0ef69fdb',
        branch='reconstruction/regional-20261113',artifact_files=files,
        runtime_files_sha256={n:sha(ROOT/n) for n in runtime},
        total_bytes=sum(v['bytes'] for v in files.values()),snapshot_directories=len(states),
        snapshot_bytes=snapshots,scope='new coarse pressure/exchange controls, not recovered lunar history'),f,indent=2)
print(json.dumps(dict(artifact_files=len(files),artifact_bytes=sum(v['bytes'] for v in files.values()),
    snapshots=len(states),snapshot_bytes=snapshots,protected=protected,case_reports=reports)),flush=True)
