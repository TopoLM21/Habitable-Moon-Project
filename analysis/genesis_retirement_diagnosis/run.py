"""Saved-profile diagnosis: phase mass is not a connected-solidus interval."""
import argparse,hashlib,json,math,time,resource,signal
from pathlib import Path
import numpy as np
from tectonics.genesis_reservoir_thermal import load,temperature
from tectonics.genesis import GenesisParameters
from analysis.genesis_energy_inventory.audit import ROOT,ROOT_SHA


def integrate_linear(q,T,lo,hi,cp,L,solidus,liquidus):
    """Exact piecewise-linear integral of h(T) and melt fraction; no quadrature."""
    points=[lo,hi]+[float(x) for x in q if lo<x<hi]
    for a,b,ta,tb in zip(q[:-1],q[1:],T[:-1],T[1:]):
        if tb!=ta:
            for boundary in (solidus,liquidus):
                x=a+(boundary-ta)*(b-a)/(tb-ta)
                if max(a,lo)<x<min(b,hi):points.append(float(x))
    points=sorted(set(points));hp=[];ph=[]
    for x in points:
        t=float(np.interp(x,q,T));phi=min(1.,max(0.,(t-solidus)/(liquidus-solidus)))
        hp.append(cp*t+L*phi);ph.append(phi)
    widths=np.diff(points)
    return float(np.dot(widths,.5*(np.array(hp[:-1])+hp[1:]))),float(np.dot(widths,.5*(np.array(ph[:-1])+ph[1:])))


def connected_solid_fraction(q,T,solidus):
    if T[0]>solidus:return 0.
    for a,b,ta,tb in zip(q[:-1],q[1:],T[:-1],T[1:]):
        if tb>solidus:return float(a+(solidus-ta)*(b-a)/(tb-ta))
    return 1.


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);start=time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    def stop(*args):raise TimeoutError('60-second read-only budget')
    signal.signal(signal.SIGALRM,stop);signal.alarm(60)
    raw=ROOT.read_bytes();assert hashlib.sha256(raw).hexdigest()==ROOT_SHA;root=json.loads(raw);r=root['runtime_state'];par=GenesisParameters(**root['source_packet']['source']['parameters']['thermal']);rho=root['source_packet']['source']['parameters']['shell']['density_kg_m3']
    paths=[Path('/workspace/genesis-coupled-thermal-projection-03/projected.json'),Path('/workspace/genesis-reservoir-thermal-step-01/after.json')]
    hashes=['d25fe1fd4e8b56b18ab8ec697b197c326d723e9421148e5d3726fb7affdd45a4','154b7dce4f1dbad58c3cf7ce583c39c8c6a0e55f80bacb895edae8085bb2c236']
    for path,sha in zip(paths,hashes):assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
    states=[load(path,legacy_path=ROOT) for path in paths];cells=[{c.identity:c for c in s.cells} for s in states]
    js=Path('/workspace/genesis-reservoir-thermal-step-01/exchanges.json');journal=json.loads(js.read_bytes());bydonor={x['donor']:x for x in journal['exchanges'] if x['donor'].startswith('surface')};dt=journal['dt_seconds']
    proposed=json.loads(Path('/workspace/genesis-reservoir-birth-02/proposal.json').read_bytes());ranges={x['face']:x for x in proposed['proposed_retirements']}
    masses=np.array(r['thermal']['layer_mass_kg']);rows=[];all_recon=[0.,0.];all_fv=[0.,0.];all_melt=[0.,0.];all_recon_melt=[0.,0.]
    for i,m in enumerate(masses):
        M=float(m.sum());bounds=np.r_[0.,np.cumsum(m)/M];q=np.r_[0.,.5*(bounds[:-1]+bounds[1:]),1.]
        profiles=[];H=[]
        for c in cells:
            h=np.array([c[f'column:{i}:{j}'].energy_j/m[j] for j in range(len(m))]);H.append(h)
            profiles.append(np.r_[temperature(c['surface_water'],par),[temperature(c[f'column:{i}:{j}'],par) for j in range(len(m))],temperature(c['deep_mantle'],par)])
        interval=ranges[i];lo,hi=interval['q_lo'],interval['q_hi'];weight=M*(hi-lo);row=dict(face=i,cohort_id=interval['id'],q_lo=lo,q_hi=hi,interval_mass_kg=weight,volume_m3=weight/rho)
        args=(par.silicate_heat_capacity_j_kg_k,par.silicate_latent_heat_j_kg,par.solidus_k,par.liquidus_k)
        for n,label in enumerate(('before','after')):
            h_int,phi_int=integrate_linear(q,profiles[n],lo,hi,*args)
            row[label]=dict(interpolated_mean_h_j_kg=h_int/(hi-lo),interpolated_solid_fraction=1-phi_int/(hi-lo),interpolated_melt_mass_kg=M*phi_int,
              FV_mean_h_j_kg=float(H[n][0]),FV_solid_fraction=1-np.clip((profiles[n][1]-par.solidus_k)/(par.liquidus_k-par.solidus_k),0,1),
              temperature_lo_k=float(np.interp(lo,q,profiles[n])),temperature_hi_k=float(np.interp(hi,q,profiles[n])))
            full_h,full_phi=integrate_linear(q,profiles[n],0.,1.,*args);all_recon[n]+=M*full_h;all_recon_melt[n]+=M*full_phi;all_fv[n]+=float(m@H[n]);all_melt[n]+=float(m@np.clip((profiles[n][1:-1]-par.solidus_k)/(par.liquidus_k-par.solidus_k),0,1))
        row['interpolated_delta_h_j']=weight*(row['after']['interpolated_mean_h_j_kg']-row['before']['interpolated_mean_h_j_kg'])
        row['FV_subset_delta_h_j']=weight*(H[1][0]-H[0][0])
        row['interpolated_added_liquid_kg']=row['after']['interpolated_melt_mass_kg']-row['before']['interpolated_melt_mass_kg']
        top=next(x for x in journal['exchanges'] if x['donor']=='surface_water' and x['recipient']==f'column:{i}:0')
        row['surface_to_top_energy_j']=top['energy_j'];row['surface_to_top_power_w']=top['energy_j']/dt
        row['top_cell_delta_energy_j']=m[0]*(H[1][0]-H[0][0]);row['surface_still_below_solidus']=bool(profiles[1][0]<par.solidus_k)
        rows.append(row)
    sums={k:math.fsum(x[k] for x in rows) for k in ['interval_mass_kg','volume_m3','interpolated_delta_h_j','FV_subset_delta_h_j','interpolated_added_liquid_kg','surface_to_top_energy_j','top_cell_delta_energy_j']}
    report=dict(status='PASS_DIAGNOSIS',classification='UNVERIFIED_PHASE_RETIREMENT: solidus envelope recession, not demonstrated melting of whole interval',
      sums=sums,interpolated_after_solid_fraction_min=min(x['after']['interpolated_solid_fraction'] for x in rows),
      FV_subset_solid_fraction_change_range=[min(x['after']['FV_solid_fraction']-x['before']['FV_solid_fraction'] for x in rows),max(x['after']['FV_solid_fraction']-x['before']['FV_solid_fraction'] for x in rows)],
      global_FV_melt_mass_delta_kg=all_melt[1]-all_melt[0],
      global_interpolated_melt_mass_delta_kg=all_recon_melt[1]-all_recon_melt[0],
      interpolated_rest_melt_mass_delta_kg=all_recon_melt[1]-all_recon_melt[0]-sums['interpolated_added_liquid_kg'],
      strip_interpolated_latent_energy_delta_j=sums['interpolated_added_liquid_kg']*par.silicate_latent_heat_j_kg,reconstructed_profile_energy_minus_FV_j=[a-b for a,b in zip(all_recon,all_fv)],
      surface_warming_k=temperature(cells[1]['surface_water'],par)-temperature(cells[0]['surface_water'],par),
      surface_cold_and_connected_all_faces=all(x['surface_still_below_solidus'] for x in rows),
      export_candidates=None,export_reason='Phase/enthalpy mapping not conservative; removal not admitted',new_thermal_steps=0,mechanical_state_changed=False,
      seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT,*paths,js,Path('/workspace/genesis-reservoir-birth-02/proposal.json')]},faces=rows)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='faces'},indent=2));signal.alarm(0)

if __name__=='__main__':main()
