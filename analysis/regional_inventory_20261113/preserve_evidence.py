"""Copy compact native evidence once; verify earlier files without modifying them."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs_v131_regional'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def states():
    dest=HERE/'saved_states';dest.mkdir(exist_ok=False)
    records=[]
    for original in sorted(OUT.glob('*/restart')):
        if not all((original/n).is_file() for n in ('meta.json','state.npz')):continue
        name=original.parent.name;target=dest/name;target.mkdir()
        result=HERE/f'{name}_result.json'
        source=json.loads(result.read_text())['source_commit'] if result.exists() else (
            '9e6628a9d8069c7c9618c4e689c8208003d4b94d' if name.startswith('initial_same') else '7f7581792b03de62e258a5657c75d0a8ed82199d')
        status='invalid pre-fix restart diagnostic; do not use as accepted continuation' if name=='m3_restart100' else 'completed control or explicit zero-time initialization; limited applicability in RESULT.md'
        files={}
        for n in ('meta.json','state.npz'):
            shutil.copy2(original/n,target/n)
            assert digest(original/n)==digest(target/n)
            assert (original/n).stat().st_ino!=(target/n).stat().st_ino
            files[n]=dict(sha256=digest(target/n),bytes=(target/n).stat().st_size)
        records.append(dict(name=name,source_commit=source,status=status,files=files))
    with (HERE/'saved_states_manifest.json').open('x') as f:json.dump(records,f,indent=2)
    print('Native files:',2*len(records),'bytes:',sum(x['bytes'] for r in records for x in r['files'].values()))


def package():
    earlier=[]
    for folder in ('reconstruction_20261113','reconstruction_exchange_20261113','integral_arcs_20261113'):
        manifest=ROOT/'analysis'/folder/'artifact_manifest.json';data=json.loads(manifest.read_text())
        entries=data.get('artifact_files',data.get('files'))
        if entries is None:raise ValueError(f'Unknown previous manifest schema {manifest}')
        bad=[]
        for path,expected in entries.items():
            expected=expected['sha256'] if isinstance(expected,dict) else expected
            if not (ROOT/path).is_file() or digest(ROOT/path)!=expected:bad.append(path)
        assert not bad,(folder,bad)
        earlier.append(dict(manifest=str(manifest.relative_to(ROOT)),files=len(entries),changed=bad))
    baseline=json.loads((ROOT/'analysis/reconstruction_20261113/baseline_manifest.json').read_text())['tracked_sha256']
    changed=[];missing=[]
    for name,expected in baseline.items():
        p=ROOT/name
        if not p.is_file():missing.append(name)
        elif digest(p)!=expected:changed.append(name)
    assert not missing,missing
    allowed={'run_long_evolution_v123.py','run_long_evolution_v124.py','tectonics/checkpoint.py',
        'tectonics/continental.py','tectonics/lithosphere.py','tectonics/topology.py','tectonics/volcanic_arc.py','tests/test_topology.py'}
    assert set(changed)<=allowed,changed
    check=dict(previous_artifact_checks=earlier,original_tracked_count=len(baseline),missing_original_files=missing,
               intentionally_changed_original_sources=changed,
               unavailable_historical_archives='not present in recovered checkout; cannot recertify')
    with (HERE/'preservation_check.json').open('x') as f:json.dump(check,f,indent=2)
    code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source_paths=['tectonics/regional_material.py','tectonics/material_exchange.py','tectonics/checkpoint.py',
                  'tectonics/volcanic_arc.py','run_long_evolution_v123.py','run_long_evolution_v124.py']
    files={}
    for p in sorted(HERE.rglob('*')):
        if not p.is_file() or p.suffix not in {'.py','.json','.md','.yaml','.log','.txt','.png','.csv','.npz'} or p.name=='artifact_manifest.json':continue
        files[str(p.relative_to(ROOT))]=dict(sha256=digest(p),bytes=p.stat().st_size)
    result=dict(numerical_runtime_commit=code_commit,parent_stage_commit='9e6628a9d8069c7c9618c4e689c8208003d4b94d',
        runtime_files_sha256={p:digest(ROOT/p) for p in source_paths},artifact_files=files,
        total_bytes=sum(x['bytes'] for x in files.values()),scope='new reconstruction, full sensible heat only; negative convergence findings and invalid development controls labelled in RESULT.md')
    with (HERE/'artifact_manifest.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(check));print('Artifact files:',len(files),'bytes:',result['total_bytes'])

if __name__=='__main__':
    {'states':states,'package':package}[sys.argv[1]]()
