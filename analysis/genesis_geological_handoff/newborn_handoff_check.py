"""Bounded real-inventory, manufactured-boundary handoff; no new Genesis run."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.genesis_geological_thermal import ThermalGeologicalContinuation
from tectonics.genesis_transfer_bridge import encoded

ROOT=Path(__file__).parent

def main():
    start=time.monotonic();p=Path('/workspace/genesis-moving-basal-pilot-01/checkpoint.json');pin=hashlib.sha256(p.read_bytes()).hexdigest()
    assert pin=='7dd22796ca517d8866505b7bf374847f03300117fc5eda0c012f27712035648b'
    output=Path('/workspace/genesis-newborn-handoff-01');assert not output.exists();output.mkdir()
    m=SimplifiedGenesis.load(p);side=m.mesh.centroids[:,2]>0;mask=np.array([side[f]!=side[g] for f,g,_,_ in m.mesh.shared_edges])
    g=ThermalGeologicalContinuation.from_source(m,experimental_nodal_retirement=True,fracture_edges=mask,experimental_parcel_mapping=True)
    np.testing.assert_array_equal(g.record['thermal_inventory']['mass_kg'],m.config['mass']);np.testing.assert_array_equal(g.record['thermal_inventory']['energy_j'],m.state['energy'])
    g.save(output/'import');g.step(.25e-6);g.save(output/'restart');h=ThermalGeologicalContinuation.load(output/'restart')
    g.step(.25e-6);h.step(.25e-6);assert encoded(g.record)==encoded(h.record);g.save(output/'endpoint')
    assert hashlib.sha256(p.read_bytes()).hexdigest()==pin
    result=dict(source_sha256=pin,manufactured_boundary=True,source_unchanged=True,newborn_profiles=len(m.state['born_material']),
        mapped_nodes=g.record['eligibility']['mapped_solid_node_count'],initial_mass_H_exact=True,mapping=g.record['mapping'],
        restart_exact=True,steps=[r['thermal'] for r in g.record['steps']],output=str(output),seconds=time.monotonic()-start)
    (ROOT/'newborn_handoff_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['steps','mapping']}))
if __name__=='__main__':main()
