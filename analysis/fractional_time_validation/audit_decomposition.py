"""Read-only post-hoc decomposition of the old coupling-1 birth delay.

This is a diagnostic retiming, not a new integrated trajectory or correction
to saved output. It changes ages only for newly born retained parcels, while
holding their endpoint locations, areas and chemical contents fixed.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_fractional_transport_probe import load_probe_source
from tectonics.fractional_surface import state_from_json, surface_totals
from tectonics.fractional_thermal import refresh_fractional_mechanics
from tectonics.fractional_coupling import thermal_context_from_json


def main():
    source = ROOT/'analysis/slab_sinking_followup/runs/ordered_sub4_dt1/elapsed_0050'
    mesh, checkpoint, fracture, model, provenance = load_probe_source(source)
    output = []
    for label, dt in [('late_dt1', 1.), ('late_dt0p5', .5), ('late_dt0p25', .25)]:
        path = ROOT/'analysis/fractional_coupling_validation/runs'/label/'fractional_checkpoint.json'
        raw = json.loads(path.read_text(encoding='utf-8'))
        state = state_from_json(raw['state'])
        payload = raw['provenance']
        sample = model.loading.sample(thermal_context_from_json(payload['thermal_context']))
        retimed = replace(state, parcels=tuple(replace(p, age_myr=p.age_myr+.5*dt)
            if p.material_id.startswith('birth:') else p for p in state.parcels))
        adjusted, cooling = refresh_fractional_mechanics(retimed, sample, model,
            origin_time_myr=provenance['origin_time_myr'],
            thermal_diffusivity_m2_s=payload['experiment']['local_cooling_diffusivity_m2_s'])
        src = payload['cumulative_thermal_sources']
        delta = cooling['thermal_source_delta']
        retained_newborn = [p for p in state.parcels if p.material_id.startswith('birth:')]
        output.append(dict(case=label, step_myr=dt,
            old_thermal_sources=src, retained_birth_midpoint_correction=delta,
            corrected_thermal_sources={k: src[k]+delta.get(k, 0.) for k in src},
            retained_births={k: sum(getattr(p,k) for p in retained_newborn)
                for k in ('area_km2','cold_mantle_volume_km3','density_excess_mass_kg')}))
    value=dict(scope='Post-hoc retained births only: does not recalculate motion or losses', cases=output)
    target=Path(__file__).with_name('audit_decomposition.json')
    target.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value,indent=2))


if __name__ == '__main__':
    main()
