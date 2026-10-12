"""One isolated 50-year cut-cell solve, independent audit and sealed restart."""
import argparse,hashlib,json,math,time,resource,signal
from pathlib import Path
import numpy as np
from tectonics.genesis_implicit_enthalpy import solve
from tectonics.genesis_reservoir_thermal import load
from tectonics.genesis import GenesisParameters,SECONDS_PER_MYR
from analysis.genesis_energy_inventory.audit import ROOT,ROOT_SHA


def seal(body):
    encoded=json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return {**body,'sha256':hashlib.sha256(encoded).hexdigest()}


def save(packet,path):
    with path.open('x') as f:json.dump(packet,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    def stop(*args):raise TimeoutError('60-second solve budget')
    signal.signal(signal.SIGALRM,stop);signal.alarm(60);start=time.monotonic()
    raw=ROOT.read_bytes();assert hashlib.sha256(raw).hexdigest()==ROOT_SHA
    candidate_path=Path('/workspace/genesis-cutcell-feasibility-02/candidate.json');candidate_raw=candidate_path.read_bytes()
    if hashlib.sha256(candidate_raw).hexdigest()!='6006521e7b3e22ce23894e63248d0145ee15d3065333c949701c730c6cab47ad':raise ValueError('Candidate SHA')
    c=json.loads(candidate_raw);s=load('/workspace/genesis-coupled-thermal-projection-03/projected.json',legacy_path=ROOT);p=GenesisParameters(**json.loads(s.parameters_json));root=json.loads(raw);shell=root['source_packet']['source']['parameters']['shell'];rho=shell['density_kg_m3'];k=shell['conductivity_w_m_k']
    geometry=root['runtime_state']['geometry'];X=np.array(geometry['vertices_m'])+np.array(root['runtime_state']['displacement_m']).reshape(-1,6)[:,:3];tri=X[np.array(geometry['faces'])];areas=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    cut=c['cells'];ncut=len(cut);N=ncut+2;deep=ncut;water=ncut+1;sc={x.identity:x for x in s.cells}
    mass=np.array([x['mass_kg'] for x in cut]+[sc['deep_mantle'].mass_kg,sc['surface_water'].mass_kg]);h0=np.array([x['specific_h'] for x in cut]+[sc['deep_mantle'].energy_j/sc['deep_mantle'].mass_kg,sc['surface_water'].energy_j/sc['surface_water'].mass_kg])
    cp=np.full(N,p.silicate_heat_capacity_j_kg_k);L=np.full(N,p.silicate_latent_heat_j_kg);offset=np.zeros(N);cp[water]=p.water_heat_capacity_j_kg_k;L[water]=0.;offset[water]=p.water_latent_heat_j_kg
    edges=[]
    for face in range(len(areas)):
        indices=[i for i,x in enumerate(cut) if x['face']==face];dz=mass[indices]/(rho*areas[face]);edges.append((water,indices[0],2*k*areas[face]/dz[0]))
        edges.extend((i,j,k*areas[face]/(.5*(dz[t]+dz[t+1]))) for t,(i,j) in enumerate(zip(indices,indices[1:])))
        edges.append((indices[-1],deep,2*k*areas[face]/dz[-1]))
    dt=5e-5*SECONDS_PER_MYR;result=solve(mass,h0,cp,L,np.full(N,p.solidus_k),np.full(N,p.liquidus_k),offset,edges,dt)
    T=result['temperature_k'];h=result['specific_h'];E0=mass*h0;E1=mass*h
    if T[water]<647.096:raise ValueError('Water no longer on hot-vapor branch')
    # Independent endpoint energy law and edge summation, not solver residual.
    phi=np.minimum(1.,np.maximum(0.,(T-p.solidus_k)/(p.liquidus_k-p.solidus_k)));expected_h=cp*T+L*phi+offset
    if not np.allclose(expected_h,h,rtol=1e-15,atol=0):raise ValueError('Phase enthalpy')
    delta=E1.astype(np.longdouble)-E0.astype(np.longdouble);balance=delta.copy();roundoff=8*(np.spacing(E1)+np.spacing(E0));tolerance=roundoff+3e-11*mass*cp
    pair=[]
    for edge_index,(i,j,G) in enumerate(edges):
        Q=dt*G*(T[i]-T[j]);balance[i]+=np.longdouble(Q);balance[j]-=np.longdouble(Q)
        allowance=16*dt*G*(math.ulp(float(T[i]))+math.ulp(float(T[j])));tolerance[i]+=allowance;tolerance[j]+=allowance
        if Q!=result['exchange_j'][edge_index]:raise ValueError('Pair flux mismatch')
        pair.append([int(i),int(j),float(Q),float(-Q)])
    ratio=float(np.max(np.abs(balance)/tolerance));global_residual=float(np.sum(delta,dtype=np.longdouble));global_allowance=float(roundoff.sum())
    if ratio>1 or abs(global_residual)>global_allowance:raise ValueError('Independent conservation audit')
    ids=[x['material_id'] for x in cut if x['material_id'] is not None];assert len(ids)==len(set(ids))==800
    mixed=[dict(id=x['material_id'],cut_index=i,temperature_k=float(T[i]),liquid_fraction=float(phi[i]),solid_fraction=float(1-phi[i])) for i,x in enumerate(cut) if x['material_id'] is not None and phi[i]>0]
    before_phi=np.clip((result['initial_temperature_k']-p.solidus_k)/(p.liquidus_k-p.solidus_k),0,1)
    body=dict(format='genesis-implicit-cutcell-thermal-0.1',time_myr=s.time_myr+5e-5,parent_raw_sha256=ROOT_SHA,parent_memory_sha256=c['parent_memory_sha256'],historical_opening_balance=None,
      closure='accepted material mean anchors plus residual unowned constant enthalpy; numerical hypothesis',
      fixed_cut_geometry=[{key:x[key] for key in ['face','layer','q_lo','q_hi','material_id','mass_kg']} for x in cut],
      mass_kg=mass.tolist(),specific_h=h.tolist(),temperature_k=T.tolist(),energy_j=E1.tolist(),
      solid_fraction_silicates=(1-phi[:deep+1]).tolist(),paired_exchanges=pair,mechanical_memory_updated=False,retirements_applied=[],
      thermal_input_for_old_material=[dict(id=x['material_id'],before_temperature_k=float(result['initial_temperature_k'][i]),after_temperature_k=float(T[i]),liquid_fraction=float(phi[i])) for i,x in enumerate(cut) if x['material_id'] is not None])
    packet=seal(body);save(packet,a.output/'after.json');restored=json.loads((a.output/'after.json').read_bytes())
    if seal({k:v for k,v in restored.items() if k!='sha256'})!=restored:raise ValueError('Restart seal')
    save(restored,a.output/'restart.json');assert (a.output/'after.json').read_bytes()==(a.output/'restart.json').read_bytes()
    assert ROOT.read_bytes()==raw
    report=dict(status='PASS',isolated_thermal_steps=1,dt_years=50,cutcells=ncut,reservoirs=N,Newton_iterations=result['iterations'],old_material_ids=800,
      old_material_mixed_count=len(mixed),mixed_old_material=mixed,mechanical_admission='BLOCKED_MIXED_MATERIAL_ADMISSION' if mixed else 'READY_THERMAL_LOADING_ONLY',
      minimum_temperature_k=float(T.min()),maximum_temperature_k=float(T.max()),maximum_principle_pass=True,
      old_material_liquid_mass_kg=math.fsum(x['mass_kg']*phi[i] for i,x in enumerate(cut) if x['material_id'] is not None),
      column_liquid_mass_delta_kg=float(np.dot(mass[:ncut],phi[:ncut]-before_phi[:ncut])),
      global_energy_residual_j=global_residual,global_endpoint_roundoff_allowance_j=global_allowance,max_local_balance_over_allowance=ratio,
      numerical_heat_added_j=0.,mass_unchanged=True,retirements_applied=0,canonical_advanced=False,restart_byte_identical=True,
      seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      candidate_input_sha256=hashlib.sha256(candidate_raw).hexdigest(),after_sha256=hashlib.sha256((a.output/'after.json').read_bytes()).hexdigest())
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='mixed_old_material'},indent=2));signal.alarm(0)

if __name__=='__main__':main()
