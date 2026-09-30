"""Frozen endpoint birth-time quadrature against the exact cold-depth integral.

Tests a uniform birth rate within each old transport step. No trajectories,
thermal state or existing saves are changed. Density excess is not estimated
by this closed-form cold-volume oracle.
"""
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.special import erfinv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_fractional_transport_probe import load_probe_source
from tectonics.fractional_coupling import thermal_context_from_json
from tectonics.genesis import SECONDS_PER_MYR


def main():
    source = ROOT/'analysis/slab_sinking_followup/runs/ordered_sub4_dt1/elapsed_0050'
    _, _, _, model, _ = load_probe_source(source)
    records = []
    for label, dt in [('late_dt1', 1.), ('late_dt0p5', .5), ('late_dt0p25', .25)]:
        path = ROOT/'analysis/fractional_coupling_validation/runs'/label/'fractional_checkpoint.json'
        raw = json.loads(path.read_text(encoding='utf-8'))
        payload = raw['provenance']
        sample = model.loading.sample(thermal_context_from_json(payload['thermal_context']))
        births = [p for p in raw['state']['parcels'] if p['material_id'].startswith('birth:')]
        areas = np.array([p['area_km2'] for p in births])
        ages = np.array([p['age_myr'] for p in births])
        crust = np.array([p['specific_properties'][0] for p in births])
        ts, tm = sample.thermal['surface_temperature_k'], sample.thermal['mantle_temperature_k']
        kappa = payload['experiment']['local_cooling_diffusivity_m2_s']
        factor = 2.*math.sqrt(kappa*SECONDS_PER_MYR)/1000.*erfinv((model.thermal.solidus_k-ts)/(tm-ts))
        ref_h = sample.lid_thickness_km
        critical = (crust/factor)**2
        cap = (ref_h/factor)**2
        lo = np.maximum(np.minimum(ages, cap), critical)
        hi = np.maximum(np.minimum(ages+dt, cap), critical)
        primitive = lambda age: 2.*factor/3.*age**1.5-crust*age
        integral = primitive(hi)-primitive(lo)+(ref_h-crust)*np.maximum(ages+dt-np.maximum(ages,cap),0.)
        exact = float(np.dot(areas, np.maximum(integral/dt,0.)))
        old_volume = math.fsum(p['cold_mantle_volume_km3'] for p in births)
        old_source = payload['cumulative_thermal_sources']['cold_mantle_volume_km3']
        rules = {}
        for count in (1,2,4,8,16):
            nodes, weights = np.polynomial.legendre.leggauss(count)
            values = np.maximum(np.minimum(ref_h, factor*np.sqrt(ages[:,None]+dt*(nodes+1.)*.5))-crust[:,None], 0.)
            volume = float(np.dot(areas,values@(weights*.5)))
            rules[str(count)] = dict(birth_cold_volume_km3=volume,
                relative_error_vs_exact_uniform_births=volume/exact-1.,
                retimed_total_thermal_source_km3=old_source+volume-old_volume)
        records.append(dict(case=label, step_myr=dt,
            exact_birth_cold_volume_km3=exact,
            exact_retimed_total_thermal_source_km3=old_source+exact-old_volume,
            gauss_legendre=rules))
    output=dict(scope=__doc__, cases=records)
    Path(__file__).with_suffix('.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
