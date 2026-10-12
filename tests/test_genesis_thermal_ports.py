from dataclasses import replace
import math
import pytest
from tectonics.genesis import GenesisParameters,SECONDS_PER_MYR,fluxes,ENERGY_SCALE,mantle_enthalpy,surface_enthalpy
from tectonics.genesis_tides import TidalParameters,initial_tidal_orbit
from tectonics.genesis_thermal_ports import radiogenic_energy,apply_ports


def fixture(p):
    ms=p.silicate_column_kg_m2*p.area_m2;mw=p.water_column_kg_m2*p.area_m2
    return dict(time_myr=0.,mass_kg=[ms,mw],energy_j=[ms*2e6,mw*(p.water_heat_capacity_j_kg_k*1300+p.water_latent_heat_j_kg)])


def test_radiogenic_half_life_integral_and_units():
    p=GenesisParameters();dt=p.radiogenic_half_life_myr*SECONDS_PER_MYR
    expected=p.silicate_column_kg_m2*p.area_m2*p.radiogenic_specific_power_w_kg*dt/(2*math.log(2))
    assert radiogenic_energy(p,0.,dt)==pytest.approx(expected)
    assert radiogenic_energy(p,p.radiogenic_half_life_myr,dt)==pytest.approx(expected/2)


def test_existing_flux_rates_map_once_to_declared_reservoirs():
    p=GenesisParameters();before=fixture(p);after,orbit,e=apply_ports(before,p,TidalParameters(),initial_tidal_orbit(TidalParameters()),1e-6)
    dt=e['dt_seconds'];rows={r['name']:r for r in e['ports']}
    q=fluxes(0.,[mantle_enthalpy(1700,p)/ENERGY_SCALE,surface_enthalpy(1300,p)/ENERGY_SCALE],p)
    assert rows['stellar_absorbed']['energy_into_thermal_j']==pytest.approx(q['absorbed_stellar_flux_w_m2']*p.area_m2*dt)
    assert rows['OLR']['energy_into_thermal_j']==pytest.approx(-q['outgoing_longwave_w_m2']*p.area_m2*dt)
    assert rows['radiogenic']['recipient']==0 and rows['OLR']['recipient']==1
    assert rows['giant_absorbed']['energy_into_thermal_j']==rows['tidal']['energy_into_thermal_j']==0
    assert set(rows)=={'radiogenic','tidal','stellar_absorbed','giant_absorbed','OLR'}
    assert before==fixture(p)


def test_enabled_orbit_heat_has_one_equal_debit():
    p=GenesisParameters();t=TidalParameters(enabled=True,spin_state='synchronous_zero_obliquity');old=initial_tidal_orbit(t)
    _,new,e=apply_ports(fixture(p),p,t,old,1e-6)
    heat=next(x['energy_into_thermal_j'] for x in e['ports'] if x['name']=='tidal')
    assert heat>0 and heat==e['orbital_debit_j']
    assert heat==pytest.approx(new.dissipated_energy_j-old.dissipated_energy_j,rel=2e-12)


def test_duplicate_tidal_source_rejected():
    p=replace(GenesisParameters(),tidal_heat_flux_w_m2=1.)
    with pytest.raises(ValueError,match='duplicate'):apply_ports(fixture(p),p,TidalParameters(),initial_tidal_orbit(TidalParameters()),1e-6)


def test_nonzero_existing_giant_term_no_new_law():
    p=replace(GenesisParameters(),giant_absorbed_flux_w_m2=2.)
    _,_,e=apply_ports(fixture(p),p,TidalParameters(),initial_tidal_orbit(TidalParameters()),1e-6)
    assert next(x['energy_into_thermal_j'] for x in e['ports'] if x['name']=='giant_absorbed')==pytest.approx(2*p.area_m2*e['dt_seconds'])


def test_independent_source_audit_and_duplicate_tamper():
    from analysis.genesis_forced_coupled_validation.audit import audit_ports
    p=GenesisParameters();t=TidalParameters(enabled=True,spin_state='synchronous_zero_obliquity');before=fixture(p)
    after,_,e=apply_ports(before,p,t,initial_tidal_orbit(t),1e-6)
    assert audit_ports(before,after,e,p,t)['status']=='PASS'
    e['ports'].append(dict(e['ports'][0]))
    with pytest.raises(AssertionError):audit_ports(before,after,e,p,t)
