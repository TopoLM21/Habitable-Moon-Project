"""Existing Genesis sources as explicit opt-in lumped boundary ports.

Source-first split: deep mantle receives the legacy mantle sources; the
surface-water/atmosphere node receives the legacy surface radiative ports.
Optional condensation uses the existing Genesis water EOS.
No legacy mantle-to-surface transfer, latent heat or secular Maxwell heat is
added here. Planetary area is the inherited spherical reference area.
"""
import math
from dataclasses import asdict
import numpy as np
from scipy.optimize import brentq
from .genesis import SECONDS_PER_MYR,outgoing_longwave_w_m2,vapor_column_kg_m2
from .genesis_tides import advance_tidal_orbit


def radiogenic_energy(p,start_myr,dt_seconds):
    if not math.isfinite(dt_seconds) or dt_seconds<=0:raise ValueError('Positive seconds required')
    rate=math.log(2)/(p.radiogenic_half_life_myr*SECONDS_PER_MYR)
    power=p.silicate_column_kg_m2*p.area_m2*p.radiogenic_specific_power_w_kg*2**(-(start_myr+p.system_age_at_start_myr)/p.radiogenic_half_life_myr)
    return power*(-math.expm1(-rate*dt_seconds))/rate


def surface_endpoint(energy,capacity,latent,stellar,giant,dt_area,p):
    """Same radiative equation, compensated residual in kelvin and tight root.

    Temperature convergence alone at 1e-10 K did not meet the existing energy
    ULP audit for large reservoirs. The independent energy bound is unchanged.
    """
    def balance(T):
        return math.fsum((capacity*T,latent,-energy,-stellar,-giant,
                          dt_area*outgoing_longwave_w_m2(T,p)))/capacity
    low=647.096;high=max(3000.,math.fsum((energy,stellar,giant,-latent))/capacity+1.)
    if balance(low)>0:raise ValueError('UNSUPPORTED_WATER_PHASE under radiative forcing')
    T=brentq(balance,low,high,xtol=np.nextafter(0.,1.),rtol=4*np.finfo(float).eps)
    candidates=(np.nextafter(T,-np.inf),T,np.nextafter(T,np.inf))
    return min((x for x in candidates if x>=low),key=lambda x:abs(balance(x)))


def water_specific_enthalpy(T,p):
    """Existing Genesis vapor/condensation EOS, water alone; no new coefficients."""
    return p.water_heat_capacity_j_kg_k*T+p.water_latent_heat_j_kg*vapor_column_kg_m2(T,p)/p.water_column_kg_m2


def water_temperature(h,p):
    low=273.16
    if h<water_specific_enthalpy(low,p):raise ValueError('Ice EOS not implemented in existing Genesis')
    return brentq(lambda T:water_specific_enthalpy(T,p)-h,low,max(3000.,h/p.water_heat_capacity_j_kg_k+1.),
                  xtol=np.nextafter(0.,1.),rtol=4*np.finfo(float).eps)


def water_capacity(T,p):
    # Numerical derivative of the EXISTING smooth vapor law, no physical coefficient.
    d=np.cbrt(np.finfo(float).eps)*max(abs(T),1.)
    return (water_specific_enthalpy(T+d,p)-water_specific_enthalpy(T-d,p))/(2*d)


def condensing_surface_endpoint(energy,mass,stellar,giant,dt_area,p):
    def residual(T):
        return math.fsum((mass*water_specific_enthalpy(T,p),-energy,-stellar,-giant,
                          dt_area*outgoing_longwave_w_m2(T,p)))/(mass*p.water_heat_capacity_j_kg_k)
    low=273.16
    if residual(low)>0:raise ValueError('Ice EOS not implemented in existing Genesis')
    high=max(3000.,(energy+stellar+giant)/(mass*p.water_heat_capacity_j_kg_k)+1.)
    T=brentq(residual,low,high,xtol=np.nextafter(0.,1.),rtol=4*np.finfo(float).eps)
    return min((np.nextafter(T,-np.inf),T,np.nextafter(T,np.inf)),key=lambda x:abs(residual(x)))


def apply_ports(before,p,tides,orbit,target_myr,*,allow_condensation=False):
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
    # Default hot-vapor branch retained; opt-in condensation reuses Genesis.
    T=(condensing_surface_endpoint(E[-1],mass[-1],stellar,giant,dt*area,p) if allow_condensation
       else surface_endpoint(E[-1],capacity,latent,stellar,giant,dt*area,p))
    olr=dt*area*outgoing_longwave_w_m2(T,p)
    rows=[dict(name='radiogenic',recipient=len(E)-2,energy_into_thermal_j=radio,scope='external nuclear energy; no nuclear inventory state'),
          dict(name='tidal',recipient=len(E)-2,energy_into_thermal_j=tide,scope='paired orbital dissipation'),
          dict(name='stellar_absorbed',recipient=len(E)-1,energy_into_thermal_j=stellar,scope='external radiation'),
          dict(name='giant_absorbed',recipient=len(E)-1,energy_into_thermal_j=giant,scope='external radiation; zero if coefficient is zero'),
          dict(name='OLR',recipient=len(E)-1,energy_into_thermal_j=-olr,scope='outgoing radiation')]
    after=E.copy();after[-2]+=radio+tide;after[-1]=(mass[-1]*water_specific_enthalpy(T,p) if allow_condensation else capacity*T+latent)
    if np.any(after<0):raise ValueError('Source overdraw')
    actual=after.astype(np.longdouble)-E.astype(np.longdouble);requested=np.zeros(len(E),dtype=np.longdouble)
    for row in rows:requested[row['recipient']]+=np.longdouble(row['energy_into_thermal_j'])
    allowance=8*(np.spacing(after)+np.spacing(E))
    if allow_condensation:
        # Long geological steps subtract large absorbed/emitted doses to obtain
        # small water storage. Include the arithmetic of the DOSES and the
        # representable root lattice, rather than only the storage ULPs.
        for row in rows:
            allowance[row['recipient']]+=8*abs(np.spacing(row['energy_into_thermal_j']))
        lo=np.nextafter(T,-np.inf);hi=np.nextafter(T,np.inf)
        root_lattice=mass[-1]*abs(water_specific_enthalpy(hi,p)-water_specific_enthalpy(lo,p))
        root_lattice+=dt*area*abs(outgoing_longwave_w_m2(hi,p)-outgoing_longwave_w_m2(lo,p))
        allowance[-1]+=root_lattice
    if np.any(np.abs(actual-requested)>allowance):raise ValueError('Source endpoint energy audit')
    result={**before,'energy_j':after.tolist()} # Split stage; conduction advances the shared time once.
    evidence=dict(dt_seconds=dt,area_m2=area,ports=rows,net_thermal_input_j=math.fsum(x['energy_into_thermal_j'] for x in rows),orbital_debit_j=tide,
        orbit_before=asdict(orbit),orbit_after=asdict(new_orbit),source_surface_endpoint_k=T,
        source_stage_clock='dose integrated over interval; state clock advanced by subsequent conduction',
        maximum_local_energy_residual_j=float(np.max(np.abs(actual-requested))),endpoint_roundoff_allowance_j=float(allowance[-2:].sum()),
        routing='legacy mantle source -> deep; legacy radiation -> surface water/atmosphere',new_source_coefficients=False)
    if allow_condensation:
        evidence.update(water_phase_model="existing Genesis condensation",audit_includes_port_dose_and_root_lattice_roundoff=True)
    return result,new_orbit,evidence
