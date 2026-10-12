"""Independent saved energy-source audit; no orbit/conduction/BE evolution."""
import math
import numpy as np
from tectonics.genesis import SECONDS_PER_MYR,SIGMA
from tectonics.genesis_tides import G,TidalOrbitState,validate_tidal_orbit


def audit_ports(before,after,e,p,tides):
    dt=e['dt_seconds'];area=4*math.pi*(1000*p.radius_km)**2
    assert e['area_m2']==area
    a,b=e['orbit_before'],e['orbit_after'];validate_tidal_orbit(TidalOrbitState(**a),tides);validate_tidal_orbit(TidalOrbitState(**b),tides)
    assert dt==(b['time_myr']-a['time_myr'])*SECONDS_PER_MYR and a['time_myr']==before['time_myr']
    ports={x['name']:x for x in e['ports']};assert len(ports)==len(e['ports'])==5
    assert set(ports)=={'radiogenic','tidal','stellar_absorbed','giant_absorbed','OLR'}
    # Independent Gaussian integration of the inherited exponential power law.
    nodes,weights=np.polynomial.legendre.leggauss(12);times=a['time_myr']+.5*(nodes+1)*dt/SECONDS_PER_MYR
    power=p.silicate_column_kg_m2*area*p.radiogenic_specific_power_w_kg*np.exp2(-(times+p.system_age_at_start_myr)/p.radiogenic_half_life_myr)
    radio=float(.5*dt*np.dot(weights,power))
    stellar=dt*area*p.stellar_flux_w_m2*(1-p.bond_albedo)*(1-p.eclipse_fraction)/4;giant=dt*area*p.giant_absorbed_flux_w_m2
    water_mass=before['mass_kg'][-1];T=(after['energy_j'][-1]/water_mass-p.water_latent_heat_j_kg)/p.water_heat_capacity_j_kg_k
    assert T>=647.096
    dry=SIGMA*T**4/(1+.75*p.dry_optical_depth);weight=-math.expm1(-p.steam_opacity_m2_kg*p.water_column_kg_m2)
    OLR=(1-weight)*dry+weight*min(dry,p.steam_olr_limit_w_m2+SIGMA*max(0,T-p.hot_window_temperature_k)**4)
    expected=dict(radiogenic=radio,stellar_absorbed=stellar,giant_absorbed=giant,OLR=-dt*area*OLR)
    for name,val in expected.items():assert math.isclose(ports[name]['energy_into_thermal_j'],val,rel_tol=3e-12,abs_tol=1.)
    heat=ports['tidal']['energy_into_thermal_j'];assert heat==e['orbital_debit_j'] and heat>=0
    # Independent eccentricity reservoir and stored-orbit debit, with local ulps.
    ld=np.longdouble;amin=ld(tides.semimajor_axis_km)*1000*(1-ld(tides.eccentricity)**2);K=ld(G)*tides.primary_mass_kg*tides.satellite_mass_kg/2
    exact=float(K/amin*(ld(a['eccentricity'])**2-ld(b['eccentricity'])**2))
    stored=b['dissipated_energy_j']-a['dissipated_energy_j'];orbit_allow=8*(math.ulp(a['dissipated_energy_j'])+math.ulp(b['dissipated_energy_j']))+3e-9*abs(heat)
    assert abs(stored-heat)<=orbit_allow and abs(exact-heat)<=orbit_allow
    E0=np.array(before['energy_j']);E1=np.array(after['energy_j']);delta=E1.astype(ld)-E0.astype(ld);wanted=np.zeros(len(delta),dtype=ld)
    for name,row in ports.items():
        assert row['recipient']==len(E0)-(2 if name in ('tidal','radiogenic') else 1)
        wanted[row['recipient']]+=ld(row['energy_into_thermal_j'])
    assert before['mass_kg']==after['mass_kg'] and np.array_equal(E0[:-2],E1[:-2])
    allowance=8*(np.spacing(E0)+np.spacing(E1));assert np.all(np.abs(delta-wanted)<=allowance)
    return dict(status='PASS',ports_checked=5,orbital_heat_j=heat,independent_eccentricity_release_j=exact,stored_orbit_debit_j=stored,orbital_debit_allowance_j=orbit_allow,
        net_source_j=math.fsum(row['energy_into_thermal_j'] for row in ports.values()),source_residual_j=float(np.sum(delta-wanted)),source_allowance_j=float(allowance[-2:].sum()),surface_source_temperature_k=T)
