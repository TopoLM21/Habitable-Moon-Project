"""One bounded initialization/feasibility check; no thermal evolution."""
import argparse,hashlib,json,math,time,resource,signal
from pathlib import Path
import numpy as np
from tectonics.genesis import GenesisParameters,silicate_specific_enthalpy
from tectonics.genesis_cutcell_initialization import partition_cell,mean_preserving_affine_envelope
from tectonics.genesis_reservoir_thermal import load,temperature
from tectonics.genesis_reservoir_birth import digest
from analysis.genesis_energy_inventory.audit import ROOT,ROOT_SHA


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);start=time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    def stop(*args):raise TimeoutError('60-second budget')
    signal.signal(signal.SIGALRM,stop);signal.alarm(60)
    raw=ROOT.read_bytes();assert hashlib.sha256(raw).hexdigest()==ROOT_SHA;r=json.loads(raw);runtime=r['runtime_state'];params=r['source_packet']['source']['parameters'];p=GenesisParameters(**params['thermal'])
    before=load('/workspace/genesis-coupled-thermal-projection-03/projected.json',legacy_path=ROOT);cells={c.identity:c for c in before.cells}
    memory={c['id']:c for c in runtime['mechanics']['state']['cohorts']};history=runtime['history'];rows=[];max_energy_error=0.;max_mass_error=0.
    bounds_min=[];bounds_max=[]
    for face,masses in enumerate(history['layer_mass_kg']):
        M=math.fsum(masses);bounds=[0.]+[math.fsum(masses[:i])/M for i in range(1,len(masses))]+[1.]
        hs=[cells[f'column:{face}:{i}'].energy_j/m for i,m in enumerate(masses)]
        for layer,(lo,hi,m) in enumerate(zip(bounds[:-1],bounds[1:],masses)):
            anchor=[dict(id=c['id'],q_lo=c['q_lo'],q_hi=c['live_q_hi'],specific_h=silicate_specific_enthalpy(memory[c['id']]['temperature_k'],p)) for c in history['cohorts'] if c['face_index']==face and c['layer_index']==layer and c['live_q_hi']>c['q_lo']]
            lower=min([hs[max(0,layer-1)],hs[layer],hs[min(len(hs)-1,layer+1)],*[x['specific_h'] for x in anchor]])
            upper=max(hs[max(0,layer-1)],hs[layer],hs[min(len(hs)-1,layer+1)])
            E=cells[f'column:{face}:{layer}'].energy_j
            part=partition_cell(face=face,layer=layer,q_lo=lo,q_hi=hi,mass_kg=m,energy_j=E,anchors=anchor,lower_h=lower,upper_h=upper)
            max_energy_error=max(max_energy_error,abs(math.fsum(c['energy_j'] for c in part)-E)/E)
            max_mass_error=max(max_mass_error,abs(math.fsum(c['mass_kg'] for c in part)-m)/m)
            rows.extend(part)
    rho=params['shell']['density_kg_m3'];k=params['shell']['conductivity_w_m_k'];Gdata=json.loads(Path('/workspace/genesis-reservoir-thermal-step-01/exchanges.json').read_bytes())['geometry'];explicit_limits=[];sens=[];ids=[];new_top=[];old_top=[]
    surface_h=silicate_specific_enthalpy(temperature(cells['surface_water'],p),p);deep_h=silicate_specific_enthalpy(temperature(cells['deep_mantle'],p),p)
    for face in range(len(history['layer_mass_kg'])):
        local=[c for c in rows if c['face']==face];area=Gdata['area_m2'][face];dz=np.array([c['mass_kg']/(rho*area) for c in local]);g=k*area/(.5*(dz[:-1]+dz[1:]));rates=np.r_[2*k*area/dz[0],np.zeros(len(dz)-1)]+np.r_[np.zeros(len(dz)-1),2*k*area/dz[-1]];rates[:-1]+=g;rates[1:]+=g
        explicit_limits.extend(np.array([c['mass_kg'] for c in local])*p.silicate_heat_capacity_j_kg_k/rates)
        from tectonics.genesis_reservoir_thermal import Reservoir
        Tfirst=temperature(Reservoir('cut','silicate',local[0]['mass_kg'],local[0]['energy_j']),p)
        Ts=temperature(cells['surface_water'],p);Told=temperature(cells[f'column:{face}:0'],p)
        new_top.append(2*k*area*(Ts-Tfirst)/dz[0]);old_top.append(2*k*area*(Ts-Told)/Gdata['dz_m'][face][0])
        for i,c in enumerate(local):
            c['volume_m3']=c['mass_kg']/rho
            alt=mean_preserving_affine_envelope(c,surface_h if i==0 else local[i-1]['specific_h'],deep_h if i==len(local)-1 else local[i+1]['specific_h'],solidus_h=p.silicate_heat_capacity_j_kg_k*p.solidus_k if c['material_id'] is not None else None)
            sens.append(alt['amplitude_h']);c['alternative_affine_h']=alt
            if c['material_id'] is not None:
                ids.append(c['material_id'])
                if c['specific_h']!=silicate_specific_enthalpy(memory[c['material_id']]['temperature_k'],p):raise ValueError('Accepted temperature changed')
    if len(ids)!=800 or set(ids)!=set(memory):raise ValueError('Material IDs changed')
    if ROOT.read_bytes()!=raw:raise ValueError('Canonical changed')
    packet=dict(format='genesis-cutcell-initialization-0.1',status='FEASIBLE_EXPLICIT_NUMERICAL_CLOSURE',time_myr=before.time_myr,
      cells=rows,parent_raw_sha256=ROOT_SHA,parent_memory_sha256=digest(runtime['mechanics']),historical_opening_balance=None,
      assumptions=['accepted current cohort mean temperatures used, not birth temperatures','constant h per old material interval; residual unowned h fixed by original cell E','not recovered unique historical gradient','no new material IDs or phase threshold'],
      mechanical_memory_changed=False,thermal_steps=0)
    (a.output/'candidate.json').write_text(json.dumps(packet,indent=2)+'\n')
    report=dict(status='PASS_FEASIBILITY',cutcells=len(rows),original_cells=2560,old_ids_preserved=len(ids),
      current_accepted_temperature_range_k=[min(c['temperature_k'] for c in memory.values()),max(c['temperature_k'] for c in memory.values())],
      max_relative_energy_partition_error=max_energy_error,max_relative_mass_partition_error=max_mass_error,
      residual_bounds_pass=True,nonunique_affine_alternatives=sum(x>0 for x in sens),max_affine_amplitude_h_j_kg=max(sens),
      explicit_CFL_limit_seconds=float(min(explicit_limits)),
      initial_top_flux_relative_difference=abs(math.fsum(new_top)-math.fsum(old_top))/abs(math.fsum(old_top)),
      input_sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [ROOT,Path('/workspace/genesis-coupled-thermal-projection-03/projected.json'),Path('/workspace/genesis-reservoir-thermal-step-01/exchanges.json')]},next_solver='conservative implicit enthalpy FV on this fixed material partition; no arbitrary further refinement',
      new_thermal_steps=0,canonical_advanced=False,seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      candidate_sha256=hashlib.sha256((a.output/'candidate.json').read_bytes()).hexdigest())
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));signal.alarm(0)

if __name__=='__main__':main()
