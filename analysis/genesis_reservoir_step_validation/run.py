"""One actual-geometry isolated thermal step; independent flux/energy audit."""
import argparse,hashlib,json,math,resource,signal,time
from pathlib import Path
import numpy as np
from tectonics.genesis import SECONDS_PER_MYR
from tectonics.genesis_reservoir_thermal import load,save
from tectonics.genesis_reservoir_conduction import step
from analysis.genesis_energy_inventory.audit import ROOT,ROOT_SHA


def silicate_temperature(h,p):
    cp,L,ts,tl=[p[k] for k in ['silicate_heat_capacity_j_kg_k','silicate_latent_heat_j_kg','solidus_k','liquidus_k']]
    return np.where(h<=cp*ts,h/cp,np.where(h>=cp*tl+L,(h-L)/cp,(h+L*ts/(tl-ts))/(cp+L/(tl-ts))))


def independent(before,after,raw,dt,journal):
    root=json.loads(raw);r=root['runtime_state'];p=root['source_packet']['source']['parameters']['thermal'];shell=root['source_packet']['source']['parameters']['shell']
    X=np.asarray(r['geometry']['vertices_m'])+np.asarray(r['displacement_m']).reshape(-1,6)[:,:3]
    tri=X[np.asarray(r['geometry']['faces'])];u=tri[:,1]-tri[:,0];v=tri[:,2]-tri[:,0]
    area=.5*np.sqrt(np.sum(u*u,axis=1)*np.sum(v*v,axis=1)-np.sum(u*v,axis=1)**2)
    mass=np.asarray(r['thermal']['layer_mass_kg']);dz=mass/(shell['density_kg_m3']*area[:,None]);k=shell['conductivity_w_m_k']
    b={c.identity:c for c in before.cells};a={c.identity:c for c in after.cells}
    T=np.array([[float(silicate_temperature(b[f'column:{i}:{j}'].energy_j/m,p)) for j,m in enumerate(row)] for i,row in enumerate(mass)])
    tm=float(silicate_temperature(b['deep_mantle'].energy_j/b['deep_mantle'].mass_kg,p))
    # This real snapshot is above critical: inherited water h=cp*T+latent.
    sw=b['surface_water'];ts=(sw.energy_j/sw.mass_kg-p['water_latent_heat_j_kg'])/p['water_heat_capacity_j_kg_k']
    if ts<647.096:raise ValueError('Independent control requires this hot-water branch')
    flux=np.zeros((len(mass),mass.shape[1]+1))
    flux[:,0]=2*k*(ts-T[:,0])/dz[:,0];flux[:,-1]=2*k*(T[:,-1]-tm)/dz[:,-1]
    flux[:,1:-1]=k*(T[:,:-1]-T[:,1:])/(.5*(dz[:,:-1]+dz[:,1:]))
    expected={f'column:{i}:{j}':float((flux[i,j]-flux[i,j+1])*area[i]*dt) for i,row in enumerate(mass) for j in range(len(row))}
    expected['surface_water']=-float(np.dot(area,flux[:,0])*dt);expected['deep_mantle']=float(np.dot(area,flux[:,-1])*dt)
    ratios=[];actual={};bounds={}
    for name,c in b.items():
        if a[name].mass_kg!=c.mass_kg:raise ValueError('Mass changed')
        actual[name]=float(np.longdouble(a[name].energy_j)-np.longdouble(c.energy_j))
        # Charge saved binary64 endpoint rounding explicitly, not a global-E floor.
        bounds[name]=4*(math.ulp(a[name].energy_j)+math.ulp(c.energy_j))+1e-11*abs(expected[name])
        ratios.append(abs(actual[name]-expected[name])/bounds[name])
    if max(ratios)>1:raise ValueError('Independent local energy/flux audit failed')
    global_residual=math.fsum(actual.values());roundoff=math.fsum(4*(math.ulp(a[n].energy_j)+math.ulp(b[n].energy_j)) for n in b)
    if abs(global_residual)>roundoff:raise ValueError('Independent total energy audit failed')
    if any(c.energy_j<0 for c in after.cells):raise ValueError('Negative enthalpy')
    afterT=np.array([[float(silicate_temperature(a[f'column:{i}:{j}'].energy_j/m,p)) for j,m in enumerate(row)] for i,row in enumerate(mass)])
    phi=lambda t:np.clip((t-p['solidus_k'])/(p['liquidus_k']-p['solidus_k']),0,1)
    old_tm=json.loads(before.origin_json)['legacy_mantle_temperature_k']
    old_bottom=2*k*(T[:,-1]-old_tm)/dz[:,-1]
    return dict(status='PASS',energy_residual_j=global_residual,endpoint_roundoff_allowance_j=roundoff,
      max_local_residual_over_allowance=max(ratios),column_delta_j=math.fsum(actual[n] for n in b if n.startswith('column:')),
      deep_delta_j=actual['deep_mantle'],water_delta_j=actual['surface_water'],
      initial_bottom_into_columns_w=float(-np.dot(area,flux[:,-1])),legacy_bottom_into_columns_w=float(-np.dot(area,old_bottom)),
      bottom_flux_change_w=float(np.dot(area,old_bottom-flux[:,-1])),top_flux_change_w=0.,
      top_flux_comparison='Same retained profile, surface temperature and geometry; only mantle representation changed',
      deep_melt_fraction_before=float(phi(tm)),deep_melt_fraction_after=float(phi(float(silicate_temperature(a['deep_mantle'].energy_j/a['deep_mantle'].mass_kg,p)))),
      column_temperature_min_k=float(afterT.min()),column_temperature_max_k=float(afterT.max()),
      column_melt_mass_change_kg=float(np.sum(mass*(phi(afterT)-phi(T)))),
      all_mass_unchanged=True,latent_heat_added_separately=False,
      local_tolerance='4*(ulp(E_before)+ulp(E_after)) + 1e-11*abs(expected exchange); whole-state energy not used as local scale')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    def stop(*args):raise TimeoutError('60-second budget')
    signal.signal(signal.SIGALRM,stop);signal.alarm(60);start=time.monotonic()
    raw=ROOT.read_bytes();before=load('/workspace/genesis-coupled-thermal-projection-03/projected.json',legacy_path=ROOT)
    if hashlib.sha256(raw).hexdigest()!=ROOT_SHA:raise ValueError('Canonical SHA')
    dt=5e-5*SECONDS_PER_MYR;after,journal=step(before,raw,dt)
    audit=independent(before,after,raw,dt,journal)
    save(after,a.output/'after.json');restarted=load(a.output/'after.json',legacy_path=ROOT);save(restarted,a.output/'restart.json')
    if (a.output/'after.json').read_bytes()!=(a.output/'restart.json').read_bytes():raise ValueError('Restart')
    if ROOT.read_bytes()!=raw:raise ValueError('Canonical changed')
    (a.output/'exchanges.json').write_text(json.dumps(journal,indent=2)+'\n')
    report=dict(status='PASS',isolated_thermal_steps=1,dt_years=50,new_canonical_Genesis_steps=0,
      before_time_myr=before.time_myr,after_time_myr=after.time_myr,exchanges=len(journal['exchanges']),
      audit=audit,restart_byte_identical=True,canonical_sha256=ROOT_SHA,
      historical_opening_balance=after.historical_opening_balance,
      seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in a.output.iterdir()})
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));signal.alarm(0)

if __name__=='__main__':main()
