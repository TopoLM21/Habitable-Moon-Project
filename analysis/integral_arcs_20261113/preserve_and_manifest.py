"""Read-only preservation checks, then a new evidence manifest. No deletions."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

base=json.loads((ROOT/'analysis/reconstruction_20261113/baseline_manifest.json').read_text())
missing=[];changed=[]
for name,digest in base['tracked_sha256'].items():
    p=ROOT/name
    if not p.exists():missing.append(name)
    elif sha(p)!=digest:changed.append(name)
allowed={'run_long_evolution_v123.py','tectonics/checkpoint.py','tectonics/continental.py',
         'tectonics/lithosphere.py','tectonics/topology.py','tectonics/volcanic_arc.py','tests/test_topology.py'}
assert not missing and set(changed)<=allowed,(missing,changed)
prior=[]
for folder in ('reconstruction_20261113','reconstruction_exchange_20261113'):
    m=json.loads((ROOT/'analysis'/folder/'artifact_manifest.json').read_text());bad=[]
    for name,data in m['artifact_files'].items():
        if not (ROOT/name).exists() or sha(ROOT/name)!=data['sha256']:bad.append(name)
    assert not bad,bad
    prior.append({'manifest':folder,'files_checked':len(m['artifact_files']),'changed':bad})
saved=json.loads((HERE/'saved_states_manifest.json').read_text())
assert all(sha(ROOT/x['path'])==x['sha256'] and (ROOT/x['path']).stat().st_nlink==1 for x in saved)
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main','refs/heads/reconstruction/regional-20261113'],cwd=ROOT,text=True)
stats={'original_files_checked':len(base['tracked_sha256']),'missing':missing,
       'intentional_source_changes_since_recovered_baseline':changed,'prior_evidence':prior,
       'independent_restart_files_verified':len(saved),'old_unavailable_archives_recertified':False,
       'remote_before_final_evidence_commit':remote,
       'raw_output_bytes':sum(p.stat().st_size for p in (ROOT/'outputs_v131_integral_arcs').rglob('*') if p.is_file())}
with (HERE/'preservation.json').open('x') as f:json.dump(stats,f,indent=2)
files={}
for p in sorted(HERE.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or p.name=='artifact_manifest.json':continue
    files[str(p.relative_to(ROOT))]={'sha256':sha(p),'bytes':p.stat().st_size}
manifest={'parent_commit':'88d5b52ed72b2f6655d227f500344ac69719064c',
    'numerical_runtime_commit':'ba4965efa4031386264f1e071ba78e64ae19523e',
    'scope':'new reconstruction; final_* runs and matched reference_*; development attempts retained and labelled in RESULT.md',
    'artifact_files':files,'total_bytes':sum(x['bytes'] for x in files.values())}
with (HERE/'artifact_manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
print(json.dumps({'files':len(files),'bytes':manifest['total_bytes'],'preservation':stats},indent=2))
