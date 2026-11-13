"""Bounded stage completion only: 2 Myr cases, no subsequent research queue."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
from tectonics.phase_exchange import PhaseParameters, PhaseStrainExchange

spec = importlib.util.spec_from_file_location('prior_strain_controls', ROOT/'analysis/strain_exchange_20261113/verify.py')
prior = importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)


def run(name, scenario='smooth', sub=3, dt=.5, fraction=1e-7, bins=16, save=False):
    path = HERE/(name+'.json')
    if path.exists(): raise FileExistsError(path)
    start = time.perf_counter()
    phase = PhaseParameters(tuple(1700+(np.arange(bins)+.5)*500/bins), fraction)
    base = prior.setup(scenario, sub, dt, 0. if scenario == 'native' else 5e8)
    x = PhaseStrainExchange.from_fresh(base, phase)
    initial = x.diagnostics(); history = [initial]
    if save: x.save(HERE/'states'/(name+'_initial'))
    then = time.perf_counter()
    while x.coupled.thermal.time_myr < 1.:
        x.step(min(dt, 1.-x.coupled.thermal.time_myr)); history.append(x.diagnostics())
    mid = deepcopy(x.checkpoint())
    if save: x.save(HERE/'states'/(name+'_mid'))
    while x.coupled.thermal.time_myr < 2.:
        x.step(min(dt, 2.-x.coupled.thermal.time_myr)); history.append(x.diagnostics())
    integration_s = time.perf_counter()-then
    # Restart at exactly the same outer step boundary, including controller memory.
    y = PhaseStrainExchange.restore(mid); y.advance(2.)
    assert y.checkpoint() == x.checkpoint()
    if save:
        x.save(HERE/'states'/(name+'_final'))
        assert PhaseStrainExchange.load(HERE/'states'/(name+'_final')).checkpoint() == x.checkpoint()
    d = x.diagnostics(); m = x.coupled.material
    assert d['external_work_j'] == 0 and d['hydraulic_heat_j'] >= 0
    assert abs(d['mass_residual_kg']) < 2e9
    assert abs(d['thermal_energy_residual_j']) < 1e-10*max(d['heat_generated_j'], 1.)
    assert abs(d['combined_energy_residual_j']) < 1e-4*max(d['heat_generated_j'], 1.)
    assert m.data['phase_initialization']['energy_added_j'] == 0
    assert not any(m.phase_state(x.coupled.thermal_params)[1][1])
    raw = json.dumps(x.checkpoint(), sort_keys=True, allow_nan=False).encode()
    result = dict(case=name, source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        scenario=scenario, cells=len(x.coupled.surface.faces), dt_myr=dt, end_myr=2.,
        phase_parameters=m.data['phase_parameters'], initialization=m.data['phase_initialization'],
        initial=initial, final=d, history=history, restart_exact=True,
        checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
        integration_wall_s=integration_s, total_wall_s=time.perf_counter()-start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        scope='short synthetic phase-supply control, unchanged topology; no lunar-history claim')
    with path.open('x') as f: json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(dict(case=name, wall_s=integration_s, born_km3=d['basalt_created_m3']/1e9,
        returned_km3=d['basalt_returned_m3']/1e9, mobile_kg=d['mobile_liquid_kg'])), flush=True)


if __name__ == '__main__':
    run('no_initial_feedstock', fraction=0., save=True)
    for sub in (2, 3, 4): run('limited_g'+str(sub), sub=sub, save=sub == 3)
    for dt, suffix in ((1., 'one'), (.25, 'quarter')): run('limited_dt_'+suffix, dt=dt)
    for sub in (2, 3, 4): run('available_g'+str(sub), sub=sub, fraction=1e-5)
    for bins in (4, 64): run('limited_phase'+str(bins), bins=bins)
    run('native_g3', scenario='native', fraction=1e-5, save=True)
    run('oblique_g3', scenario='oblique', fraction=1e-5)
