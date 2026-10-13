"""Active thermal counterpart; preserves the earlier frozen experiment."""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import json
import time
import numpy as np
from tectonics.genesis import GenesisParameters, M_EARTH
from tectonics.genesis_tides import TidalParameters, initial_tidal_orbit
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.genesis_geological_thermal import ThermalGeologicalContinuation, eligibility
from tectonics.genesis_transfer_bridge import encoded
from analysis.simplified_genesis_validation.synthetic import irregular_three_domain_synthetic
from analysis.genesis_geological_handoff.experiment import exact_saved_state


def thermal_source():
    m=irregular_three_domain_synthetic(); c,s=m.config,m.state
    c['face_torque']=np.zeros_like(c['face_torque']).tolist()
    T=m.temperature(s['energy']); c['mass'][-2]=4e24
    # Same existing EOS and source-law constants, inventory parameters derived
    # from this manufactured fixture, not fitted to a successful trajectory.
    base=GenesisParameters()
    p=replace(base,radius_km=c['radius_km'],
        mass_earth=sum(c['mass'][:-1])/(M_EARTH*base.mantle_mass_fraction),
        water_volume_km3=c['mass'][-1]/base.water_density_kg_m3/1e9)
    c['cp'][-1]=p.water_heat_capacity_j_kg_k
    c['latent'][:-1]=[p.silicate_latent_heat_j_kg]*(len(T)-1)
    c['offset'][-1]=p.water_latent_heat_j_kg
    s['energy']=(np.array(c['mass'])*(np.array(c['cp'])*T+np.array(c['offset']))).tolist()
    tides=TidalParameters(enabled=True,spin_state='synchronous_zero_obliquity',
        satellite_mass_kg=p.mass_earth*M_EARTH,satellite_radius_km=p.radius_km,
        surface_gravity_m_s2=p.surface_gravity_m_s2)
    c.update(source_mode='genesis',thermal_parameters=asdict(p),tidal_parameters=asdict(tides))
    s['orbit']=asdict(initial_tidal_orbit(tides)); m=SimplifiedGenesis(c,s)
    counts=[len(np.unique(m.state['owner']))]
    for _ in range(2):
        m.advance(25.); counts.append(len(np.unique(m.state['owner'])))
    return m,counts


def run(root):
    start=time.monotonic(); root=Path(root); root.mkdir(parents=True,exist_ok=False)
    model,counts=thermal_source(); source_bytes=encoded(dict(config=model.config,state=model.state))
    mantle=np.zeros((model.nf,3)); mantle[:,2]=.04*model.mesh.centroids[:,0]
    engine=ThermalGeologicalContinuation.from_source(model,mantle_omega=mantle)
    engine.save(root/'mapped')
    for _ in range(3): engine.step(10.)
    engine.save(root/'split'); resumed=ThermalGeologicalContinuation.load(root/'split')
    engine.step(10.); resumed.step(10.)
    a=engine.save(root/'direct'); b=resumed.save(root/'resumed');exact_saved_state(a,b)
    rows=engine.record['steps']; thermal=[x['thermal'] for x in rows]
    result=dict(selector_owner_counts=counts,mature_steps=len(rows),restart_exact=True,
        source_unchanged=encoded(dict(config=model.config,state=model.state))==source_bytes,
        transported_winning_cells=sum(x['transported_winning_cells'] for x in rows),
        relative_omega_max_rad_per_myr=max(x['relative_omega_rad_per_myr'] for x in rows),
        deep_initial_k=thermal[0]['deep_before_k'],deep_final_k=thermal[-1]['deep_after_k'],
        surface_initial_k=thermal[0]['surface_before_k'],surface_final_k=thermal[-1]['surface_after_k'],
        max_local_H_residual_over_allowance=max(x['max_local_residual_over_allowance'] for x in thermal),
        max_global_H_residual_over_allowance=max(abs(x['global_residual_j'])/x['global_allowance_j'] for x in thermal),
        max_transport_H_residual_over_allowance=max(abs(x['transport_residual_j'])/x['transport_allowance_j'] for x in thermal),
        recycled_H_j=sum(x['recycled_enthalpy_j'] for x in thermal),created_H_j=sum(x['created_enthalpy_j'] for x in thermal),
        mechanical_heat_j=sum(x['mechanical_heat_j'] for x in thermal),
        chemical_fractions_inferred=False,real_planet_steps=0,elapsed_seconds=time.monotonic()-start)
    assert counts==[1,2,3]
    assert result['source_unchanged'] and result['transported_winning_cells']>0
    assert abs(result['deep_final_k']-result['deep_initial_k'])>.01
    assert result['max_global_H_residual_over_allowance']<=1
    assert result['max_local_H_residual_over_allowance']<=1
    (root/'result.json').write_bytes(encoded(result));print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');args=p.parse_args();run(args.output)
