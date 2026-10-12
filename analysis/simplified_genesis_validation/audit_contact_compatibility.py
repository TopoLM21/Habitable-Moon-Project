"""Inspect existing laws and saved synthetic placement; no evolution or transfer."""
import json,time
from pathlib import Path
from dataclasses import fields
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.genesis_contact_law import ContactLawParameters
from tectonics.spherical_polygons import intersect_convex,polygon_area
from tectonics.genesis_transfer_bridge import digest,encoded

ROOT=Path(__file__).resolve().parents[2]

def main():
    start=time.perf_counter();path=Path('/workspace/simplified-genesis-synthetic-01/final.json');raw=path.read_bytes();model=SimplifiedGenesis.load(path)
    s=model.state;c=model.config;assert len(s['omega'])==2
    vertices=model.mesh.vertices;faces=model.mesh.faces;owner=np.array(s['owner']);R=np.array(s['rotation']);best=None;checked=0
    for a,b,_,_ in model.mesh.shared_edges:
        if owner[a]==owner[b]:continue
        pa=vertices[faces[a]]@R[owner[a]].T;pb=vertices[faces[b]]@R[owner[b]].T
        overlap=intersect_convex(pa,pb);area=polygon_area(overlap)*c['radius_km']**2 if len(overlap) else 0.;checked+=1
        if best is None or area>best['overlap_km2']:
            original=intersect_convex(vertices[faces[a]],vertices[faces[b]])
            best=dict(faces=[a,b],owners=[int(owner[a]),int(owner[b])],overlap_km2=area,original_overlap_km2=polygon_area(original)*c['radius_km']**2 if len(original) else 0.,intersection_vertices=overlap.tolist())
    required=[x.name for x in fields(ContactLawParameters)]
    actual=SimplifiedGenesis.load('/workspace/simplified-genesis-phase-pilot-01/final.json')
    available={**actual.config['shell'],**actual.config['faults']}
    missing=[k for k in required if k not in available]
    result=dict(status='BLOCKED_PHYSICAL_CONTACT_AND_STATE_CONTRACT',new_planetary_steps=0,new_transfers=0,input_sha256=digest(raw),synthetic_domains=2,boundary_neighbor_pairs_checked=checked,largest_observed_pair_overlap=best,scope='saved rotated triangular occupancy; adjacent boundary pairs only; not all-pairs/all-time volume certification',contact_law_fields_missing_by_name=missing,warning='matching names do not establish matching units or physical scope; fault viscosity Pa s is not contact viscosity Pa s/m',real_active_memory_scalars=int(np.array(actual.state['memory']).size),real_thermal_unknowns=len(actual.state['energy']),real_passive_reserve_j=actual.state['passive_reserve_j'],native_mature_compatibility=actual.mature_compatibility(),seconds=time.perf_counter()-start)
    assert digest(path.read_bytes())==result['input_sha256']
    with (ROOT/'analysis/simplified_genesis_validation/contact_compatibility.json').open('xb') as file:file.write(encoded(result))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
