"""Sequential bounded comparisons; no duplicate runs after a recovery."""
import json,time
from verify import run,HERE

cases=[]
for scenario in ('smooth','oblique'):
    for sub in (1,2,3):
        cases.append((f'{scenario}_g{sub}',dict(scenario=scenario,sub=sub,end=100.)))
for sub in (2,3,4):
    cases.append((f'native_g{sub}_20',dict(sub=sub,end=20.)))
for dt in (.5,.25):
    cases.append((f'native_g3_dt{dt:g}_20',dict(sub=3,dt=dt,end=20.)))
for length in (0.,5e5,2e6):
    cases.append((f'native_length{length:g}_20',dict(sub=3,length=length,end=20.)))
cases.append(('native_g3_100',dict(sub=3,end=100.)))

start=time.perf_counter();completed=[];remaining=[]
for index,(name,options) in enumerate(cases):
    if (HERE/f'{name}.json').exists():
        record=json.loads((HERE/f'{name}.json').read_text())
        if record['status']!='complete':raise RuntimeError(f'Inspect stopped case {name} before retrying')
        print('Already completed',name,flush=True);completed.append(name);continue
    if time.perf_counter()-start>150:
        remaining=[x[0] for x in cases[index:]];break
    record=run(name,**options);completed.append(name)
    if record['status']!='complete':raise RuntimeError(f'Inspect stopped case {name}')
summary=dict(completed=completed,remaining=remaining,wall_s=time.perf_counter()-start,
    scope='sequential case times including setup and restore checks; no extrapolated completed horizons')
with (HERE/'queue_summary.json').open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps(summary),flush=True)
