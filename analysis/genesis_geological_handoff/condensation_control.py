"""One local 25 vs 2x12.5 year comparison; no global parameter or timestep sweep."""
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.genesis_transfer_bridge import encoded
from analysis.genesis_geological_handoff.real_progression import sha,observe


def run(source,output):
    start=time.monotonic();root=Path(output);root.mkdir(exist_ok=False,parents=True)
    source=Path(source);before=sha(source);m=SimplifiedGenesis.load(source).with_condensation()
    initial=observe(m,0.);a=deepcopy(m);b=deepcopy(m)
    h=m.config['max_step_years'];a.advance(h);b.advance(h/2);b.advance(h/2)
    x=observe(a,time.monotonic()-start);y=observe(b,time.monotonic()-start)
    Ta=a.temperature(a.state['energy']);Tb=b.temperature(b.state['energy']);nodes=a.solid_nodes
    wa=a.state['last_ledger']['water'];wb=b.state['last_ledger']['water']
    created_a=wa['liquid_mass_kg']-initial['liquid_water_mass_kg'];created_b=wb['liquid_mass_kg']-initial['liquid_water_mass_kg']
    memory_a=np.asarray(a.state['memory']);memory_b=np.asarray(b.state['memory'])
    report=dict(source_file=str(source),source_sha256=before,source_unchanged=before==sha(source),
        comparison='one full inherited maximum step vs two half steps at the same source',
        full_years=h,half_years=h/2,initial=initial,full=x,halves=y,
        water_temperature_difference_k=float(Ta[-1]-Tb[-1]),
        max_solid_temperature_difference_k=float(abs(Ta[nodes]-Tb[nodes]).max()),
        deep_temperature_difference_k=float(Ta[-2]-Tb[-2]),
        max_damage_difference=float(abs(np.asarray(a.state['damage'])-b.state['damage']).max()),
        relative_memory_difference=float(np.linalg.norm(memory_a-memory_b)/np.linalg.norm(memory_b)),
        formed_liquid_full_kg=created_a,formed_liquid_halves_kg=created_b,
        liquid_increment_relative_difference=(created_a-created_b)/created_b,
        same_topology=a.state['owner']==b.state['owner'],
        same_born_cell_ids=[r['cell_id'] for r in a.state.get('born_material',[])]==[r['cell_id'] for r in b.state.get('born_material',[])],
        full_water_audit=wa,halves_last_water_audit=wb,
        temporal_accuracy_certified=False,elapsed_seconds=time.monotonic()-start)
    (root/'report.json').write_bytes(encoded(report));print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run(a.source,a.output)
