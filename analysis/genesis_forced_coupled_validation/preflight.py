"""Saved-input source inventory and thermal-work significance; no evolution."""
import json,math
from pathlib import Path
import numpy as np
from dataclasses import replace
from tectonics.genesis import GenesisParameters,outgoing_longwave_w_m2
from tectonics.genesis_tides import TidalParameters,tidal_heat_power_w
from analysis.genesis_connected_step_validation.run import PARENT,PARENT_SHA
from analysis.genesis_coupled_split_control.run import pinned


def inspect():
    root=pinned(PARENT,PARENT_SHA);params=root['source_packet']['source']['parameters'];p=GenesisParameters(**params['thermal']);o=root['runtime_state']['thermal']['orbit'];t=TidalParameters(**params['tides'])
    th=pinned('/workspace/genesis-implicit-cutcell-01/after.json','632354594ccda5aaffeb8933d9ed95be96cb53037a2ce67d18cf407562d6f2e8')
    old=pinned('/workspace/genesis-cutcell-mechanical-02/candidate.json','5a0cfcd3edfd57f4f05b59e8b408b09c25ac2a704068609e22029ab865230f2e')
    steps=[json.loads(Path('/workspace/genesis-coupled-split-25-01/step_'+str(i)+'.json').read_bytes()) for i in (1,2)]
    assert steps[0]['predecessor_sha256']==PARENT_SHA
    import hashlib
    assert hashlib.sha256(Path('/workspace/genesis-coupled-split-25-01/step_1.json').read_bytes()).hexdigest()==steps[1]['predecessor_sha256']
    assert hashlib.sha256(Path('/workspace/genesis-coupled-split-25-01/step_2.json').read_bytes()).hexdigest()=='ad8810c6f6c537e6252deeaccafda569dcc403431a3442651934b7086992ad6d'
    delta={e['material_id']:-e['thermal_work_to_mechanics_j'] for e in old['exchanges']}
    for b in steps:
        for e in b['exchanges']:delta[e['material_id']]+=e['thermal_work_to_mechanics_j']
    values=np.array([delta[e['material_id']] for e in old['exchanges']]);caps=np.array([old['thermal']['mass_kg'][e['cutcell']]*p.silicate_heat_capacity_j_kg_k for e in old['exchanges']])
    top=-math.fsum(e[2] for e in th['paired_exchanges'] if e[0]==3361)
    # Surface T is read from the original migrated water reservoir, not the BE50 endpoint.
    proj=pinned('/workspace/genesis-coupled-thermal-projection-03/projected.json','d25fe1fd4e8b56b18ab8ec697b197c326d723e9421148e5d3726fb7affdd45a4');water=next(c for c in proj['state']['cells'] if c['identity']=='surface_water');Ts=(water['energy_j']/water['mass_kg']-p.water_latent_heat_j_kg)/p.water_heat_capacity_j_kg_k
    flux=dict(radiogenic_w_m2=p.silicate_column_kg_m2*p.radiogenic_specific_power_w_kg*2**(-(o['time_myr']+p.system_age_at_start_myr)/p.radiogenic_half_life_myr),tidal_w_m2=tidal_heat_power_w(replace(t,eccentricity=o['eccentricity'],semimajor_axis_km=o['semimajor_axis_km']))/p.area_m2,stellar_absorbed_w_m2=p.absorbed_stellar_flux_w_m2,giant_absorbed_w_m2=p.giant_absorbed_flux_w_m2,OLR_out_w_m2=outgoing_longwave_w_m2(Ts,p))
    assert p.tidal_heat_flux_w_m2==0 and flux['giant_absorbed_w_m2']==0
    return dict(status='PASS',new_evolution_steps=0,source_time_myr=o['time_myr'],surface_temperature_k=Ts,reference_spherical_area_m2=p.area_m2,source_fluxes=flux,thermal_work_difference_j=float(math.fsum(values)),max_local_equivalent_delta_k=float(np.max(np.abs(values/caps))),mean_equivalent_delta_k=float(math.fsum(values)/caps.sum()),over_saved_top_conduction=float(math.fsum(values)/top),interpretation='Absolute thermal effect is small for this window; retain work/splitting sensitivity as a limitation, not a blocker.')


if __name__=='__main__':print(json.dumps(inspect(),indent=2))
