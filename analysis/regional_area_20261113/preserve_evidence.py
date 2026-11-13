"""Native copies and hash manifests; old evidence is read-only."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs_v131_regional_area'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def states():
    candidates=sorted(HERE.glob('*_result.json'))
    size=sum((OUT/json.loads(p.read_text())['case']/'restart'/n).stat().st_size for p in candidates for n in ('state.npz','meta.json'))
    if size>30_000_000:raise RuntimeError(f'Native evidence exceeds reviewed compact budget: {size}')
    dest=HERE/'saved_states';dest.mkdir(exist_ok=False);records=[]
    for result in candidates:
        item=json.loads(result.read_text());name=item['case'];source=OUT/name/'restart';target=dest/name;target.mkdir()
        files={}
        for n in ('meta.json','state.npz'):
            shutil.copy2(source/n,target/n)
            assert digest(source/n)==digest(target/n) and (source/n).stat().st_ino!=(target/n).stat().st_ino
            files[n]=dict(sha256=digest(target/n),bytes=(target/n).stat().st_size)
        records.append(dict(name=name,source_commit=item['source_commit'],files=files,
                            status='bounded reconstructed control; source/force/contact limits in RESULT.md'))
    with (HERE/'saved_states_manifest.json').open('x') as f:json.dump(records,f,indent=2)
    print(json.dumps(dict(native_states=len(records),files=2*len(records),bytes=size)))


def package():
    earlier=[]
    for folder in ('reconstruction_20261113','reconstruction_exchange_20261113','integral_arcs_20261113','regional_inventory_20261113'):
        path=ROOT/'analysis'/folder/'artifact_manifest.json';data=json.loads(path.read_text());entries=data.get('artifact_files',data.get('files'))
        changed=[name for name,value in entries.items() if not (ROOT/name).is_file() or digest(ROOT/name)!=(value['sha256'] if isinstance(value,dict) else value)]
        assert not changed,(folder,changed)
        earlier.append(dict(manifest=str(path.relative_to(ROOT)),files=len(entries),changed=changed))
    baseline=json.loads((ROOT/'analysis/reconstruction_20261113/baseline_manifest.json').read_text())['tracked_sha256']
    missing=[name for name in baseline if not (ROOT/name).is_file()]
    changed=[name for name,value in baseline.items() if (ROOT/name).is_file() and digest(ROOT/name)!=value]
    allowed={'run_long_evolution_v123.py','run_long_evolution_v124.py','tectonics/checkpoint.py','tectonics/continental.py',
             'tectonics/lithosphere.py','tectonics/topology.py','tectonics/volcanic_arc.py','tests/test_topology.py'}
    assert not missing and set(changed)<=allowed,(missing,changed)
    check=dict(previous_artifact_checks=earlier,original_tracked_count=len(baseline),missing_original_files=missing,
               intentionally_changed_original_sources=changed,unavailable_historical_archives='not present in recovered checkout; cannot recertify')
    with (HERE/'preservation_check.json').open('x') as f:json.dump(check,f,indent=2)
    files={}
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or path.suffix not in {'.py','.json','.md','.yaml','.log','.txt','.png','.csv','.npz'} or path.name=='artifact_manifest.json':continue
        files[str(path.relative_to(ROOT))]=dict(bytes=path.stat().st_size,sha256=digest(path))
    code=['tectonics/regional_area.py','tectonics/strip_intersection.py','tectonics/regional_material.py',
          'tectonics/material_exchange.py','tectonics/checkpoint.py','run_long_evolution_v123.py','run_long_evolution_v124.py']
    manifest=dict(runtime_commit='4e714a47136f1258cd29bc24190ea71a173dd41a',recovery_commit='287759eec98f3ac053e543c2c4b8d9b942573c8a',
                  previous_completed_stage='a6e78948f1980555c0ca4d9c04d92d72ca4e4020',
                  runtime_files_sha256={name:digest(ROOT/name) for name in code},artifact_files=files,
                  total_bytes=sum(v['bytes'] for v in files.values()),
                  incomplete_diagnostic='source_diagnosis.json is a retained partial JSON from a reporting-only NumPy scalar serialization error; use source_diagnosis_verified.json',
                  scope='new reconstruction; exact sensible-heat ledger is not full physical energy closure; no paid fracture')
    with (HERE/'artifact_manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
    print(json.dumps(check));print(json.dumps(dict(artifact_files=len(files),total_bytes=manifest['total_bytes'])))


if __name__=='__main__':{'states':states,'package':package}[sys.argv[1]]()
