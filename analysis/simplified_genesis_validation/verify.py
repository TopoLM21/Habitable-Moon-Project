"""Saved real pilot audit plus independent synthetic positive cycle; no new real interval."""
import json,time
from pathlib import Path
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis,constrained_force_reaction
from tectonics.genesis_transfer_bridge import encoded,digest
from .synthetic import synthetic
from .import_source import build

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/workspace/simplified-genesis-pilot-01')

def main():
    start=time.perf_counter();initial=SimplifiedGenesis.load(OUT/'initial.json');final=SimplifiedGenesis.load(OUT/'final.json');prev=SimplifiedGenesis.load(OUT/'step_2.json')
    # Exact replay of the same last interval with the added independent local
    # energy identity. The sole new field is an audit metric, not physical state.
    prev.advance(25.);check=dict(prev.state);old=dict(final.state)
    Lnew=check.pop('last_ledger');Lold=old.pop('last_ledger');assert encoded(check)==encoded(old)
    assert all(Lnew[k]==v for k,v in Lold.items())
    imported=build();cfg=imported.config
    reaction=constrained_force_reaction(np.sum(cfg['face_linear_momentum_map'],axis=0),np.sum(cfg['face_force'],axis=0),np.sum(cfg['face_drag_force'],axis=0),np.array(initial.state['omega'][0]),np.array(final.state['omega'][0]),75*365.25*86400)
    # Mean force balance over the full pilot: endpoint reaction diagnostic,
    # not a claim of a resolved center constraint stress distribution.
    syn=synthetic();s0=syn.state['time_myr'];syn.advance(25.);s1=json.loads(encoded(syn.state));syn.advance(25.)
    tmp=Path('/workspace/simplified-genesis-synthetic-01');tmp.mkdir(exist_ok=False);syn.save(tmp/'final.json')
    restored=SimplifiedGenesis.load(tmp/'final.json');syn.advance(25.);restored.advance(25.);assert encoded(syn.state)==encoded(restored.state)
    relative=float(np.linalg.norm(np.array(syn.state['omega'][0])-syn.state['omega'][1]));assert relative>0
    temperature0=initial.temperature(initial.state['energy']);temperature1=final.temperature(final.state['energy'])
    out=dict(real=dict(start_myr=initial.state['time_myr'],end_myr=final.state['time_myr'],new_steps=3,domains=len(final.state['omega']),events=final.state['events'],max_damage_initial=max(initial.state['damage']),max_damage_final=max(final.state['damage']),max_solid_temperature_change_k=float(np.max(abs(temperature1[initial.solid_nodes]-temperature0[initial.solid_nodes]))),solid_mean_temperature_change_k=float(np.mean(temperature1[initial.solid_nodes]-temperature0[initial.solid_nodes])),omega_rad_s=final.state['omega'],active_energy_final_j=final.state['last_ledger']['elastic_after'],passive_reserve_unchanged=initial.state['passive_reserve_j']==final.state['passive_reserve_j'],restart_physical_state_and_existing_ledger_exact=True,moment_balance_relative=Lnew['moment_balance_relative'],endpoint_center_reaction_n=reaction.tolist()),synthetic=dict(initial_damage_zero=True,first_split_time_myr=s1['events'][0]['time_myr'],domains=len(syn.state['omega']),relative_angular_speed_rad_s=relative,relative_surface_speed_m_s=relative*syn.config['radius_km']*1000,rotation_difference_norm=float(np.linalg.norm(np.array(syn.state['rotation'][0])-syn.state['rotation'][1])),split_momentum_relative=s1['events'][0]['angular_momentum_relative'],restart_next_step_exact=True,whole_balance_j=syn.state['last_ledger']['whole_balance_j'],whole_allowance_j=syn.state['last_ledger']['whole_allowance_j']),compatibility=final.mature_compatibility(),seconds=time.perf_counter()-start,new_real_intervals=0,real_replayed_intervals=1,input_hashes={p.name:digest(p.read_bytes()) for p in OUT.glob('*.json')})
    for name,pin in cfg['source_pins'].items():assert digest(Path(name).read_bytes())==pin
    with (ROOT/'analysis/simplified_genesis_validation/result.json').open('xb') as f:f.write(encoded(out))
    (ROOT/'analysis/simplified_genesis_validation/pilot_report.json').write_bytes((OUT/'report.json').read_bytes())
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
