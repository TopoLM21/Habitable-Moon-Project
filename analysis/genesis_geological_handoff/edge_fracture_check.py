"""Finite verification set; manufactured closure is not natural plate formation."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from tectonics.experimental_edge_fracture import EdgeFracture,Parameters,certify
from tectonics.genesis_starter_topology import _validate_system

ROOT=Path(__file__).parent


def manufactured(level,reverse=False):
    m=EdgeFracture(level,reverse=reverse);old=json.loads(json.dumps(m.state))
    # Exact manufactured displacement jump: tests topology/mechanical accounting only.
    # It is explicitly NOT used in the unseeded loading experiment or real data.
    target=10*np.where(m.mesh.centroids[:,2]>0,1.,-1.)
    dt=100*31557600.;L=m.incidence.T@sparse.diags(m.stiffness)@m.incidence
    load=(L@target+m.params.drag_pa_s_m*m.area/dt*target)/m.area
    first=m.step(100.,load)
    assert first['domains']>=2 and first['reason']=='CLOSED_CONNECTED_BOUNDARIES'
    owner=np.array(m.state['owner']);seam=owner[m.f]!=owner[m.g]
    k=m.stiffness*(1-np.array(m.state['damage']))**2
    assert np.all(k[seam]==0)
    assert m.state['fracture_reservoir_j']>0
    # Cross-domain stiffness must vanish in the mechanically updated operator.
    broken=(m.incidence.T@sparse.diags(k)@m.incidence).tocoo()
    assert all(owner[i]==owner[j] or abs(v)<1e-20 for i,j,v in zip(broken.row,broken.col,broken.data))
    nextrow=m.step(1.,load)
    newu=np.array(m.state['u']);traction=k*(m.incidence@newu)/m.cross
    assert np.max(abs(traction[seam]))==0
    for key in ('mass_kg','area_m2','enthalpy_j'):assert old[key]==m.state[key]
    _validate_system(m.mesh,m.system())
    return m,dict(first=first,post_split=nextrow,zero_transmission_across_boundary=True,
        conserved_inventory=True,plate_system_valid=True,manufactured_not_natural=True)


def main():
    start=time.monotonic();output=dict(controls={},experiments=[])
    zero=EdgeFracture(2)
    for _ in range(5):zero.step(5.,np.zeros(zero.mesh.cell_count))
    assert max(zero.state['damage'])==0 and len(set(zero.state['owner']))==1
    output['controls']['no_load_no_damage']=True
    low=EdgeFracture(2)
    for _ in range(5):low.step(5.,.1*low.params.forcing_pa*low.mesh.centroids[:,2])
    assert max(low.state['damage'])==0
    output['controls']['subthreshold_no_damage']=True
    diffuse=EdgeFracture(2);_,reason=certify(diffuse.mesh,np.ones(len(diffuse.mesh.shared_edges),bool),diffuse.params)
    assert reason=='DIFFUSE_NETWORK_FAILURE';output['controls']['diffuse_rejected']=reason
    for level in (2,3):
        m,result=manufactured(level);other,_=manufactured(level,True)
        owner=np.array(m.state['owner']);cut=np.array([owner[f]!=owner[g] for f,g,_,_ in m.mesh.shared_edges])
        cut[np.flatnonzero(cut)[0]]=False
        _,why=certify(m.mesh,cut,m.params);assert why=='NOT_SEPARATING'
        result['open_boundary_rejected']=True
        assert m.state==other.state;result['reverse_order_exact']=True
        path=ROOT/f'edge_fracture_restart_{level}.json';m.save(path);restart=EdgeFracture.load(path)
        m.step(1.);restart.step(1.);assert m.state==restart.state;result['restart_exact']=True
        output['controls'][f'manufactured_{level}']=result
    # Fixed natural forcing, no premarked weakness; first coarse run is also short cost benchmark.
    for level,dt in ((2,5.),(3,5.),(2,2.5)):
        m=EdgeFracture(level);clock=time.monotonic();rows=[]
        for _ in range(round(500/dt)):
            if time.monotonic()-clock>120:raise TimeoutError('120s per diagnostic')
            rows.append(m.step(dt))
        output['experiments'].append(dict(level=level,faces=m.mesh.cell_count,dt_years=dt,seconds=time.monotonic()-clock,
            endpoint=rows[-1],events=m.state['events'],damage=m.state['damage'],
            max_energy_relative=max(x['energy_relative'] for x in rows),
            initial_max_traction_pa=rows[0]['max_traction_pa'],state_sha256=hashlib.sha256(json.dumps(m.state,sort_keys=True).encode()).hexdigest()))
    output['seconds']=time.monotonic()-start
    output['real_pilot']=dict(status='NOT_STARTED',reason='Shared nodal mechanics requires duplicated seam nodes/contact; real geological handoff lacks active Maxwell-memory/SI energy mapping. No fallback to constrained-column model.')
    from tectonics.simplified_genesis import SimplifiedGenesis
    from tectonics.genesis_geological_handoff import GeologicalContinuation
    source=Path('/workspace/genesis-moving-basal-source-01.json');blob=source.read_bytes();real=SimplifiedGenesis.load(source)
    try:
        GeologicalContinuation.from_synthetic(real,composition='all_solid_nodes_basalt_reference_density_3000',mantle_omega=np.zeros((80,3)),water_volume_km3=0.)
    except ValueError as exc:
        assert str(exc).startswith('REAL_HANDOFF_NOT_READY')
        output['real_pilot'].update(verified_adapter_rejection=str(exc),source_time_myr=real.state['time_myr'],
            source_sha256=hashlib.sha256(blob).hexdigest(),source_mode=real.config['source_mode'],
            mechanical_mode=real.config['mechanical_mode'],source_damage_range=[min(real.state['damage']),max(real.state['damage'])])
    else:raise AssertionError('Real handoff guard unexpectedly absent')
    assert source.read_bytes()==blob
    (ROOT/'edge_fracture_result.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({**output,'experiments':[{k:v for k,v in e.items() if k!='damage'} for e in output['experiments']]},indent=2))


if __name__=='__main__':main()
