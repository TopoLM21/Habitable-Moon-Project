"""Read-only comparison of aggregate versus spatially ordered cohort buoyancy."""
from pathlib import Path
import json
import sys
import math
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tectonics.dynamics import DynamicsParameters
from tectonics.subduction_memory import SubductionMemoryParameters, memory_from_json
from tectonics.young_boundary import accepted_slab_force_sections, _cohort_current
from tectonics.young_slab_sinking import slab_sinking_system, _shape_quadrature

path = ROOT/'analysis/slab_sinking_validation/runs/validated_sinking_fixed2_sub4_dt1/elapsed_0400'
cfg = yaml.safe_load((path/'mature_config.yaml').read_text(encoding='utf-8'))
meta = json.loads((path/'mature_checkpoint/meta.json').read_text(encoding='utf-8'))
memory = memory_from_json(meta['subduction_memory'])
inv = memory.young_boundary_state
dyn = DynamicsParameters(**cfg['plate_dynamics'])
sub = SubductionMemoryParameters(**cfg['subduction_memory'])
sections = accepted_slab_force_sections(memory, params=sub)
with np.load(path/'mature_checkpoint/state.npz') as arrays:
    incoming = arrays['mantle_lithosphere_thickness_km']
geometry = slab_sinking_system(sections, len(meta['system_plates']), cfg['moon']['radius_km'],
    dyn.young_slab_mantle_viscosity_pa_s, dyn.young_slab_mantle_depth_km, dyn,
    incoming_thickness_km=incoming)
shape = {(x['contact_key'], x['subducting_plate'], x['overriding_plate']):x for x in geometry.sections}
rows = []
for section in sections:
    key = (section.contact_key, section.subducting_plate, section.overriding_plate)
    if key not in shape:
        continue
    cohorts = [c for s in inv.segments.values() if s.attached and
        (s.contact_key,s.subducting_plate,s.overriding_plate)==key for c in s.thermal_cohorts
        if c.deep_transfer_fraction < 1.]
    cohorts = sorted(cohorts, key=lambda c:c.acceptance_time_myr, reverse=True)
    slab = shape[key]
    bend = slab['bend_length_km']*1000.
    dip = math.radians(slab['nominal_dip_deg'])
    def integral(endpoint):
        if endpoint <= 0:
            return 0.
        weights, angles = _shape_quadrature(endpoint, bend, dip)
        return float(weights@np.sin(angles))
    cursor = 0.
    ordered_mass_sine = 0.
    cohort_rows = []
    for c in cohorts:
        retained,cold = _cohort_current(c, inv, memory.time_myr)
        length = c.accepted_area_km2*retained/section.trench_length_km*1000.
        end = cursor+length
        mean_sine = (integral(end)-integral(cursor))/length
        mass = c.initial_density_excess_mass_kg*cold
        ordered_mass_sine += mass*mean_sine
        cohort_rows.append(dict(age_myr=memory.time_myr-c.acceptance_time_myr,
            start_km=cursor/1000.,end_km=end/1000.,mass_kg=mass,mean_sine=mean_sine))
        cursor = end
    uniform_force = slab['gravitational_feed_force_n']
    ordered_force = ordered_mass_sine*dyn.gravity_m_s2
    rows.append(dict(key=key,cohort_count=len(cohorts),slab_length_km=section.slab_length_km,
        uniform_gravitational_force_n=uniform_force,ordered_gravitational_force_n=ordered_force,
        ordered_over_uniform=ordered_force/uniform_force if uniform_force else None,
        cohorts=cohort_rows))
report = dict(scope='Frozen final400 inventory; comparison only, no production change.',
    interpretation='Newest retained cohorts occupy shallowest arc-length intervals; existing force instead spreads current mass uniformly.',
    connected_sections=len(rows),multiple_cohort_sections=sum(x['cohort_count']>1 for x in rows),
    total_uniform_gravitational_force_n=sum(x['uniform_gravitational_force_n'] for x in rows),
    total_ordered_gravitational_force_n=sum(x['ordered_gravitational_force_n'] for x in rows),
    rows=rows)
report['summed_force_ratio']=report['total_ordered_gravitational_force_n']/report['total_uniform_gravitational_force_n']
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
(Path(__file__).parent/'transport_cohort_geometry_probe.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
