"""Copy compact native states and bind new evidence; never rewrite old data."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs_v131_contact_time'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def states():
    cases=[json.loads(p.read_text()) for p in sorted(HERE.glob('*_result.json'))]
    size=sum((OUT/r['case']/'restart'/n).stat().st_size for r in cases for n in ('meta.json','state.npz'))
    if size>15_000_000:raise RuntimeError(f'Native evidence exceeds compact budget: {size}')
    records=[]
    for result in cases:
        source=OUT/result['case']/'restart';target=HERE/'saved_states'/result['case']
        target.mkdir(parents=True,exist_ok=True);files={}
        for name in ('meta.json','state.npz'):
            src=source/name;dst=target/name
            if not dst.exists():
                with src.open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b)
            assert digest(src)==digest(dst) and src.stat().st_ino!=dst.stat().st_ino
            files[name]=dict(bytes=dst.stat().st_size,sha256=digest(dst))
        records.append(dict(case=result['case'],source_commit=result['source_commit'],
            source_scope=result.get('source_scope','recorded source commit'),
            runtime_sha256=result.get('runtime_sha256',{}),files=files))
    with (HERE/'saved_states_manifest.json').open('x') as f:json.dump(records,f,indent=2)
    print(json.dumps(dict(states=len(records),files=2*len(records),bytes=size)))


def package():
    earlier=[]
    for folder in ('reconstruction_20261113','reconstruction_exchange_20261113','integral_arcs_20261113','regional_inventory_20261113','regional_area_20261113'):
        path=ROOT/'analysis'/folder/'artifact_manifest.json'
        data=json.loads(path.read_text());entries=data.get('artifact_files',data.get('files'))
        changed=[name for name,v in entries.items() if not (ROOT/name).is_file() or digest(ROOT/name)!=(v['sha256'] if isinstance(v,dict) else v)]
        assert not changed,(folder,changed)
        earlier.append(dict(manifest=str(path.relative_to(ROOT)),manifest_sha256=digest(path),files=len(entries),changed=changed))
    baseline=json.loads((ROOT/'analysis/reconstruction_20261113/baseline_manifest.json').read_text())['tracked_sha256']
    missing=[name for name in baseline if not (ROOT/name).is_file()]
    changed=[name for name,v in baseline.items() if (ROOT/name).is_file() and digest(ROOT/name)!=v]
    allowed={'run_long_evolution_v123.py','run_long_evolution_v124.py','tectonics/checkpoint.py','tectonics/continental.py',
             'tectonics/lithosphere.py','tectonics/topology.py','tectonics/volcanic_arc.py','tests/test_topology.py'}
    assert not missing and set(changed)<=allowed,(missing,changed)
    check=dict(previous_artifact_checks=earlier,original_tracked_count=len(baseline),missing_original_files=missing,
        intentionally_changed_original_sources=changed,unavailable_historical_archives='not in recovered checkout; cannot recertify')
    with (HERE/'preservation_check.json').open('x') as f:json.dump(check,f,indent=2)
    files={}
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or path.suffix not in {'.py','.json','.md','.yaml','.log','.txt','.png','.csv','.npz'} or path.name=='artifact_manifest.json':continue
        files[str(path.relative_to(ROOT))]=dict(bytes=path.stat().st_size,sha256=digest(path))
    code=['tectonics/arc_time.py','tectonics/material_contacts.py','tectonics/regional_area.py','tectonics/strip_intersection.py',
          'tectonics/regional_material.py','tectonics/material_exchange.py','tectonics/checkpoint.py','run_long_evolution_v123.py',
          'run_long_evolution_v124.py','tests/test_arc_time.py','tests/test_material_contacts.py']
    manifest=dict(runtime_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        localization_commit='be47df224d0fe7f6d6b0c7aeae553b02d7720a40',contact_registry_commit='0e45555a0e5c839035ab65d08b35e0b2b1edd902',
        previous_stage='df23c6139faa932078585e617e3c78af3ed14cdd',runtime_files_sha256={name:digest(ROOT/name) for name in code},
        artifact_files=files,total_bytes=sum(v['bytes'] for v in files.values()),
        incomplete_run='gate20_g3: no result/state/resource confirmation; exact session outcome unknown; not retried',
        scope='new reconstructed source/contact checks; finite sensible-heat ledger only, legacy proxy forces and unclosed surface; no paid fracture')
    with (HERE/'artifact_manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
    print(json.dumps(check));print(json.dumps(dict(artifact_files=len(files),total_bytes=manifest['total_bytes'])))


if __name__=='__main__':{'states':states,'package':package}[sys.argv[1]]()
