"""Sequential bounded evidence; no duplicate names and no hidden continuation."""
import json
import time
from verify import HERE, run

cases = []
for scenario, pressure in [('native', 0.), ('smooth', 5e8)]:
    for sub in (2, 3, 4):
        cases.append((f'{scenario}_g{sub}_20', dict(scenario=scenario, pressure=pressure, sub=sub)))
    for dt in (.5, .25):
        cases.append((f'{scenario}_g3_dt{dt}_20', dict(scenario=scenario, pressure=pressure, dt=dt)))
cases += [
    ('oblique_g3_20', dict(scenario='oblique', pressure=5e8)),
    ('oblique_g4_20', dict(scenario='oblique', pressure=5e8, sub=4)),
    ('native_g3_100', dict(end=100.)),
    ('smooth_g3_100', dict(scenario='smooth', pressure=5e8, end=100.)),
    ('native_compliance2e8', dict(compliance=2e8)),
    ('smooth_compliance2e8', dict(scenario='smooth', pressure=5e8, compliance=2e8)),
    ('smooth_pressure3e8', dict(scenario='smooth', pressure=3e8)),
    ('smooth_width500km', dict(scenario='smooth', pressure=5e8, width=5e5)),
    ('smooth_width2000km', dict(scenario='smooth', pressure=5e8, width=2e6)),
    ('smooth_quadrature3', dict(scenario='smooth', pressure=5e8, quadrature=3)),
]
started = time.perf_counter(); results = []
for name, args in cases:
    if time.perf_counter()-started > 240: break
    result = run(name, **args)
    results.append({k:result[k] for k in ('case', 'status', 'failure', 'wall_s')})
    if result['status'] != 'complete': break
with (HERE/'queue_summary.json').open('x') as f:
    json.dump(dict(cases=results, wall_s=time.perf_counter()-started,
        remaining=[n for n, _ in cases[len(results):]], soft_budget_s=240), f, indent=2)
