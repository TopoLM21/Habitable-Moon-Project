"""One saved selector/restart transaction. No time integration or actual split."""
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from .adapter import select,eligible,save,load
from analysis.genesis_thermal_moment.adapter import MomentAdapter
from tectonics.genesis_transfer_bridge import digest,encoded
from tectonics.genesis_volume_fault_evolution import volume_fault_history_from_dict,prepare_volume_fault_activation
from tectonics.genesis_starter import StarterParameters

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/workspace/genesis-starter-selector-01')

def main():
    started=time.perf_counter();OUT.mkdir(exist_ok=False)
    report=json.loads((ROOT/'analysis/genesis_thermal_moment/report.json').read_bytes());pins=dict(report['input_sha256'])
    endpoint=Path('/workspace/genesis-thermal-moment-25-01/after.json');pins[str(endpoint)]='18476901044d7c8844a5809725f3100b8f026fa52005d5e276b27040acbd273d'
    for name,pin in pins.items():assert digest(Path(name).read_bytes())==pin
    initial=json.loads(Path('/workspace/genesis-exploratory-100-01/continuation/initial.json').read_bytes())['packet'];source=json.loads((ROOT/'analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json').read_bytes())['source_packet']
    full=json.loads(Path('/workspace/genesis-forced-coupled-25-01/candidate.json').read_bytes());r=json.loads(endpoint.read_bytes())
    a=MomentAdapter(initial,source,full['thermal_packet']['fixed_cut_geometry'],full['inputs'],report['contact_parameters'])
    before=a.history.damage.copy();temps=a.solid_mean_temperature(np.array(r['temperature_k']));dt=r['time_myr']-initial['thermal']['time_myr'];sp=a.shell
    healing=1/sp['cold_healing_timescale_myr']+np.clip((temps-700)/650,0,1)**4/sp['hot_healing_timescale_myr']
    rates=[]
    # Infer the constant rate used over the SAVED step from its exact source-law
    # damage update. A frozen scalar law diagnostic, not an advanced planet.
    for d0,d1,H in zip(before,r['fault_history']['damage'],healing):
        def equation(L):
            rate=L+H;eq=L/rate
            return eq+(d0-eq)*np.exp(-rate*dt)-d1
        rates.append(brentq(equation,0.,1e8,xtol=1e-10))
    equilibrium=np.array(rates)/(np.array(rates)+healing)
    g=initial['state']['geometry'];p=StarterParameters()
    payload=dict(scope='saved_noncanonical_selector_only',state=r,geometry=g,source_pins=pins,
        generalized_momentum_kg_m_s=(a.M@np.array(r['velocity_m_s'])).tolist(),
        initial_material_mass_kg=sum(c['mass_kg'] for c in g['cohorts']),
        selector_parameters=dict(radius_km=a.gp.radius_km,min_child_area_km2=p.min_child_area_km2,min_band_span_km=p.min_band_span_km),
        selector_parameter_origin='existing Starter geometric area/span controls only; no strength/damage/motion defaults',
        topology_motion_view='Euler placeholders only; saved nodal velocities and momentum remain authoritative')
    result=select(payload);assert result['status']=='NO_CANDIDATE' and len(result['owner_ids'])==1
    hist=volume_fault_history_from_dict(r['fault_history'],source);candidate,event=prepare_volume_fault_activation(hist,source,r['stress_pa'])
    assert event is None and np.array_equal(eligible(r['fault_history'],r['source_parameters']['faults']),candidate.fault_active)
    save(OUT/'checkpoint.json',payload);restored=load(OUT/'checkpoint.json');assert encoded(restored)==encoded(payload)
    save(OUT/'restart.json',restored);assert (OUT/'checkpoint.json').read_bytes()==(OUT/'restart.json').read_bytes()
    assert select(restored)==result
    a.restore(restored['state']);np.testing.assert_array_equal(a.M@a.v0,payload['generalized_momentum_kg_m_s'])
    for name,pin in pins.items():assert digest(Path(name).read_bytes())==pin
    result.update(input_sha256=pins,original_source_activation_agrees=True,checkpoint_restart_byte_identical=True,active_state_restored=True,generalized_momentum_exact=True,new_planetary_steps=0,new_physical_splits=0,material_cohorts=len(g['cohorts']),source_strength_pa=sp['tensile_strength_pa'],source_traction_pa=sp['convective_traction_pa'],activation_persistence_myr=r['source_parameters']['faults']['activation_persistence_myr'],frozen_damage_diagnostic=dict(method='inferred loading from saved constant-rate damage update; fixed forcing/temperature only, not a prediction under cooling',loading_per_myr_min_max=[float(min(rates)),float(max(rates))],healing_per_myr_min_max=[float(min(healing)),float(max(healing))],equilibrium_damage_max=float(max(equilibrium)),faces_equilibrium_above_activation=int(sum(equilibrium>=r['source_parameters']['faults']['activation_damage']))),seconds=time.perf_counter()-started,checkpoint_sha256=digest((OUT/'checkpoint.json').read_bytes()),output_directory=str(OUT))
    with (ROOT/'analysis/genesis_starter_selector_bridge/result.json').open('xb') as f:f.write(encoded(result))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
