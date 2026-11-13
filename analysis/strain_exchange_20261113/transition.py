"""Same physical 50 Myr state; refine time only across pressure limitation."""
from copy import deepcopy
from dataclasses import replace
import json
import time
from verify import HERE
from tectonics.strain_exchange import StrainExchange

initial=StrainExchange.load(HERE/'states/native_compliance2e8')
initial.advance(50.);initial.save(HERE/'states/capacity50')
results=[]
for dt in (1.,.5,.25):
    o=deepcopy(initial);o.coupled.options=replace(o.coupled.options,step_myr=dt)
    then=time.perf_counter();o.advance(65.)
    path=HERE/'states'/f'capacity65_dt{dt}';o.save(path)
    assert StrainExchange.load(path).checkpoint()==o.checkpoint()
    events=[e for e in o.data['history'] if e['time_myr']>50]
    limited=next((e for e in events if e['negative_quota_m3'] and e['returned_m3']/e['negative_quota_m3']<1-1e-8),None)
    results.append(dict(dt_myr=dt,wall_s=time.perf_counter()-then,final=o.diagnostics(),
        first_limited_step_end_myr=None if limited is None else limited['time_myr'],
        first_limited_step_start_myr=None if limited is None else limited['time_myr']-dt,
        exact_save_load=True))
with (HERE/'transition.json').open('x') as f:
    json.dump(dict(initial_time_myr=50.,initial_state='states/capacity50',results=results,
        scope='same physical memory; endpoint active-set diagnostic, not strict substep event localization'),f,indent=2)
print(json.dumps([dict(dt=x['dt_myr'],first=x['first_limited_step_end_myr'],
    returned_km3=x['final']['basalt_returned_m3']/1e9,wall_s=x['wall_s']) for x in results]),flush=True)
