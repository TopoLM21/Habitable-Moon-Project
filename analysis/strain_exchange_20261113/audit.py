"""Exact continuations and independent preservation evidence after the queue."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from verify import HERE, ROOT
from tectonics.strain_exchange import StrainExchange

started=time.perf_counter();restart=[]
for scenario in ('native','smooth'):
    o=StrainExchange.load(HERE/'states'/f'{scenario}_g3_20')
    reference=StrainExchange.load(HERE/'states'/f'{scenario}_g3_100')
    then=time.perf_counter();o.advance(100.)
    assert o.checkpoint()==reference.checkpoint()
    restart.append(dict(scenario=scenario,from_myr=20.,to_myr=100.,
        exact_complete_checkpoint=True,wall_s=time.perf_counter()-then))

protected=[]
for p in sorted((ROOT/'analysis').glob('*/artifact_manifest.json')):
    if p.parent==HERE:continue
    d=json.loads(p.read_text());count=0
    for name,item in d['artifact_files'].items():
        raw=(ROOT/name).read_bytes()
        expected=item['sha256'] if isinstance(item,dict) else item
        assert hashlib.sha256(raw).hexdigest()==expected,name
        count+=1
    protected.append(dict(manifest=str(p.relative_to(ROOT)),files=count))

reports=[]
for path in sorted(HERE.glob('*.json')):
    d=json.loads(path.read_text())
    if 'final' not in d:continue
    f=d['final'];o=StrainExchange.load(HERE/'states'/d['case'])
    assert o.diagnostics()==f
    assert f['external_work_j']==0
    assert f['hydraulic_heat_j']>=0 and f['reservoir_energy_j']>=0
    assert abs(f['thermal_energy_residual_j']) <= 1e-10*max(f['heat_generated_j'],1.)
    reports.append(dict(case=d['case'],complete=d['status']=='complete',
        mechanical_relative_error=abs(f['combined_energy_residual_j'])/max(f['heat_generated_j'],1.),
        origin_residual_max_kg=max(map(abs,f['origin_mass_residual_kg'])),
        mass_residual_kg=f['mass_residual_kg']))

with (HERE/'audit.json').open('x') as f:
    json.dump(dict(runtime_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        restart=restart,protected=protected,protected_count=sum(x['files'] for x in protected),
        report_checks=reports,wall_s=time.perf_counter()-started),f,indent=2)
print(json.dumps(dict(exact_restart=restart,protected=sum(x['files'] for x in protected),reports=len(reports))),flush=True)
