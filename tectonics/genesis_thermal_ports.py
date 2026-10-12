"""Existing Genesis sources as explicit opt-in lumped boundary ports.

Source-first split: deep mantle receives the legacy mantle sources; the
surface-water/atmosphere node receives the legacy surface radiative ports.
No legacy mantle-to-surface transfer, latent heat or secular Maxwell heat is
added here. Planetary area is the inherited spherical reference area.
"""
import math
from dataclasses import asdict
import numpy as np
from scipy.optimize import brentq
from .genesis import SECONDS_PER_MYR,outgoing_longwave_w_m2
from .genesis_tides import advance_tidal_orbit


def radiogenic_energy(p,start_myr,dt_seconds):
    if not math.isfinite(dt_seconds) or dt_seconds<=0:raise ValueError('Positive seconds required')
    rate=math.log(2)/(p.radiogenic_half_life_myr*SECONDS_PER_MYR)
    power=p.silicate_column_kg_m2*p.area_m2*p.radiogenic_specific_power_w_kg*2**(-(start_myr+p.system_age_at_start_myr)/p.radiogenic_half_life_myr)
    return power*(-math.expm1(-rate*dt_seconds))/rate


def apply_ports(before,p,tides,orbit,target_myr):
    if p.tidal_heat_flux_w_m2!=0:raise ValueError('Orbit-derived tides cannot duplicate prescribed tidal flux')
    if before['time_myr']!=orbit.time_myr or target_myr<=orbit.time_myr:raise ValueError('Shared source clocks')
    if not math.isclose(p.radius_km,tides.satellite_radius_km,rel_tol=1e-12):raise ValueError('Shared spherical area')
    E=np.array(before['energy_j'],dtype=float);mass=np.array(before['mass_kg'],dtype=float)
    if E.ndim!=1 or mass.shape!=E.shape or len(E)<2 or not np.isfinite(E).all() or not np.isfinite(mass).all() or np.any(E<0) or np.any(mass<=0):raise ValueError('Positive reservoir fields')
    # Fixed reservoir protocol: all silicate cells, then deep silicate, then water.
    if not math.isclose(float(mass[:-1].sum()),p.silicate_column_kg_m2*p.area_m2,rel_tol=3e-12):raise ValueError('Silicate source mass coverage')
    if not math.isclose(float(mass[-1]),p.water_column_kg_m2*p.area_m2,rel_tol=3e-12):raise ValueError('Water source mass coverage')
    dt=(target_myr-orbit.time_myr)*SECONDS_PER_MYR;area=p.area_m2
    new_orbit,mean_tidal_flux=advance_tidal_orbit(orbit,tides,target_myr)
    tide=mean_tidal_flux*dt*area;radio=radiogenic_energy(p,orbit.time_myr,dt)
    stellar=p.absorbed_stellar_flux_w_m2*area*dt;giant=p.giant_absorbed_flux_w_m2*area*dt
    capacity=mass[-1]*p.water_heat_capacity_j_kg_k;latent=mass[-1]*p.water_latent_heat_j_kg
    # Supported existing hot-vapor branch only; no new water/ice EOS.
    def balance(T):return capacity*T+latent-E[-1]-stellar-giant+dt*area*outgoing_longwave_w_m2(T,p)
    low=647.096;high=max(3000.,(E[-1]+stellar+giant-latent)/capacity+1.)
    if balance(low)>0:raise ValueError('UNSUPPORTED_WATER_PHASE under radiative forcing')
    T=brentq(balance,low,high,xtol=1e-10,rtol=1e-14)
    olr=dt*area*outgoing_longwave_w_m2(T,p)
    rows=[dict(name='radiogenic',recipient=len(E)-2,energy_into_thermal_j=radio,scope='external nuclear energy; no nuclear inventory state'),
          dict(name='tidal',recipient=len(E)-2,energy_into_thermal_j=tide,scope='paired orbital dissipation'),
          dict(name='stellar_absorbed',recipient=len(E)-1,energy_into_thermal_j=stellar,scope='external radiation'),
          dict(name='giant_absorbed',recipient=len(E)-1,energy_into_thermal_j=giant,scope='external radiation; zero if coefficient is zero'),
          dict(name='OLR',recipient=len(E)-1,energy_into_thermal_j=-olr,scope='outgoing radiation')]
    after=E.copy();after[-2]+=radio+tide;after[-1]=capacity*T+latent
    if np.any(after<0):raise ValueError('Source overdraw')
    actual=after.astype(np.longdouble)-E.astype(np.longdouble);requested=np.zeros(len(E),dtype=np.longdouble)
    for row in rows:requested[row['recipient']]+=np.longdouble(row['energy_into_thermal_j'])
    allowance=8*(np.spacing(after)+np.spacing(E))
    if np.any(np.abs(actual-requested)>allowance):raise ValueError('Source endpoint energy audit')
    result={**before,'energy_j':after.tolist()} # Split stage; conduction advances the shared time once.
    evidence=dict(dt_seconds=dt,area_m2=area,ports=rows,net_thermal_input_j=math.fsum(x['energy_into_thermal_j'] for x in rows),orbital_debit_j=tide,
        orbit_before=asdict(orbit),orbit_after=asdict(new_orbit),source_surface_endpoint_k=T,
        source_stage_clock='dose integrated over interval; state clock advanced by subsequent conduction',
        maximum_local_energy_residual_j=float(np.max(np.abs(actual-requested))),endpoint_roundoff_allowance_j=float(allowance[-2:].sum()),
        routing='legacy mantle source -> deep; legacy radiation -> surface water/atmosphere',new_source_coefficients=False)
    return result,new_orbit,evidence
