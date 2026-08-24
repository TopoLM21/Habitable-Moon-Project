"""v0.9 effective long-term thermal evolution of the habitable moon.

The model is deliberately intermediate between a hand-tuned cooling curve and a
full 3-D mantle-convection solver.  It evolves one well-mixed mantle thermal
reservoir with an energy balance

    C_m dT/dt = Q_rad(t) + Q_tide(e) - Q_conv(T, g, R)

and uses boundary-layer/Rayleigh scaling for convective heat loss.  The moon's
canonical mass, radius and surface gravity therefore affect the result directly:

* mass -> mantle heat capacity and radiogenic inventory;
* radius -> cooling surface area and effective mantle depth;
* gravity -> Rayleigh number / convective vigor;
* orbit + eccentricity -> tidal heating.

All rheological and compositional constants remain explicit prototype
parameters.  This is not yet a mineral-physics or core-dynamo model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .tides import EccentricityHistory, G, M_JUPITER, semi_major_axis_from_period

M_EARTH = 5.9722e24
R_GAS = 8.31446261815324
SECONDS_PER_MYR = 1.0e6 * 365.25 * 86400.0


@dataclass(slots=True, frozen=True)
class ThermalParameters:
    system_age_at_start_myr: float = 500.0
    initial_mantle_temperature_k: float = 1850.0
    surface_temperature_k: float = 288.0
    mantle_mass_fraction: float = 0.68
    mantle_depth_fraction_radius: float = 0.45
    mantle_density_kg_m3: float = 4000.0
    mantle_heat_capacity_j_kg_k: float = 1200.0
    thermal_conductivity_w_m_k: float = 4.0
    thermal_expansivity_per_k: float = 3.0e-5
    thermal_diffusivity_m2_s: float = 1.0e-6
    viscosity_reference_pa_s: float = 1.0e21
    viscosity_reference_temperature_k: float = 1600.0
    activation_energy_j_mol: float = 3.0e5
    viscosity_min_pa_s: float = 1.0e18
    viscosity_max_pa_s: float = 1.0e24
    critical_rayleigh: float = 1000.0
    nusselt_prefactor: float = 0.27
    nusselt_exponent: float = 1.0 / 3.0
    radiogenic_specific_power_at_formation_w_kg: float = 5.0e-12
    radiogenic_effective_half_life_myr: float = 2400.0
    tidal_k2_over_q: float = 3.0e-4
    min_tectonic_activity_factor: float = 0.35
    max_tectonic_activity_factor: float = 1.8


@dataclass(slots=True)
class ThermalState:
    time_myr: float
    system_age_myr: float
    mantle_temperature_k: float
    reference_convective_flux_w_m2: float
    tectonic_activity_factor: float
    thermal_lithosphere_thickness_km: float


@dataclass(slots=True)
class ThermalDiagnostics:
    time_myr: float
    system_age_myr: float
    mantle_temperature_k: float
    viscosity_pa_s: float
    rayleigh_number: float
    nusselt_number: float
    convective_heat_flux_w_m2: float
    radiogenic_heat_flux_w_m2: float
    tidal_heat_flux_w_m2: float
    net_heat_flux_w_m2: float
    radiogenic_power_tw: float
    tidal_power_tw: float
    convective_power_tw: float
    thermal_lithosphere_thickness_km: float
    tectonic_activity_factor: float
    eccentricity: float


def _geometry(mass_earth: float, radius_km: float, params: ThermalParameters) -> tuple[float, float, float, float]:
    mass_kg = float(mass_earth) * M_EARTH
    radius_m = float(radius_km) * 1000.0
    mantle_mass_kg = mass_kg * float(params.mantle_mass_fraction)
    surface_area_m2 = 4.0 * math.pi * radius_m**2
    mantle_depth_m = radius_m * float(params.mantle_depth_fraction_radius)
    return mass_kg, mantle_mass_kg, surface_area_m2, mantle_depth_m


def mantle_viscosity_pa_s(temperature_k: float, params: ThermalParameters) -> float:
    t = max(float(temperature_k), 1.0)
    tref = max(float(params.viscosity_reference_temperature_k), 1.0)
    exponent = float(params.activation_energy_j_mol) / R_GAS * (1.0 / t - 1.0 / tref)
    exponent = float(np.clip(exponent, -60.0, 60.0))
    eta = float(params.viscosity_reference_pa_s) * math.exp(exponent)
    return float(np.clip(eta, params.viscosity_min_pa_s, params.viscosity_max_pa_s))


def convective_state(
    temperature_k: float,
    radius_km: float,
    surface_gravity_m_s2: float,
    params: ThermalParameters,
) -> tuple[float, float, float, float]:
    """Return (viscosity, Ra, Nu, surface convective heat flux W/m2)."""
    radius_m = float(radius_km) * 1000.0
    depth = radius_m * float(params.mantle_depth_fraction_radius)
    delta_t = max(float(temperature_k) - float(params.surface_temperature_k), 1.0)
    eta = mantle_viscosity_pa_s(temperature_k, params)
    ra = (
        float(params.mantle_density_kg_m3)
        * float(surface_gravity_m_s2)
        * float(params.thermal_expansivity_per_k)
        * delta_t
        * depth**3
        / (float(params.thermal_diffusivity_m2_s) * eta)
    )
    if ra <= float(params.critical_rayleigh):
        nu = 1.0
    else:
        nu = float(params.nusselt_prefactor) * (
            ra / float(params.critical_rayleigh)
        ) ** float(params.nusselt_exponent)
        nu = max(nu, 1.0)
    conductive = float(params.thermal_conductivity_w_m_k) * delta_t / depth
    flux = conductive * nu
    thermal_lithosphere_km = depth / nu / 1000.0
    return eta, float(ra), float(nu), float(flux), float(thermal_lithosphere_km)


def radiogenic_power_w(
    system_age_myr: float,
    mantle_mass_kg: float,
    params: ThermalParameters,
) -> float:
    half = max(float(params.radiogenic_effective_half_life_myr), 1e-9)
    specific = float(params.radiogenic_specific_power_at_formation_w_kg) * 2.0 ** (-float(system_age_myr) / half)
    return specific * float(mantle_mass_kg)


def tidal_heating_power_w(
    eccentricity: float,
    radius_km: float,
    rotation_period_hours: float,
    primary_mass_jupiter: float,
    params: ThermalParameters,
) -> float:
    """Leading synchronous eccentricity-tide dissipation power.

    Uses P=(21/2)(k2/Q) G Mp^2 R^5 n e^2/a^6.  k2/Q is an effective
    configurable dissipation parameter; no frequency-dependent rheology yet.
    """
    e = max(float(eccentricity), 0.0)
    mp = float(primary_mass_jupiter) * M_JUPITER
    radius_m = float(radius_km) * 1000.0
    period_s = float(rotation_period_hours) * 3600.0
    n = 2.0 * math.pi / period_s
    a = semi_major_axis_from_period(mp, rotation_period_hours)
    return (
        10.5
        * float(params.tidal_k2_over_q)
        * G
        * mp**2
        * radius_m**5
        * n
        * e**2
        / a**6
    )


def initialize_thermal_state(
    mass_earth: float,
    radius_km: float,
    surface_gravity_m_s2: float,
    params: ThermalParameters,
) -> ThermalState:
    _, _, _, _ = _geometry(mass_earth, radius_km, params)
    _, _, nu, qconv, lith_km = convective_state(
        params.initial_mantle_temperature_k,
        radius_km,
        surface_gravity_m_s2,
        params,
    )
    _ = nu
    return ThermalState(
        time_myr=0.0,
        system_age_myr=float(params.system_age_at_start_myr),
        mantle_temperature_k=float(params.initial_mantle_temperature_k),
        reference_convective_flux_w_m2=max(float(qconv), 1e-12),
        tectonic_activity_factor=1.0,
        thermal_lithosphere_thickness_km=float(lith_km),
    )


def diagnose_thermal_state(
    state: ThermalState,
    mass_earth: float,
    radius_km: float,
    surface_gravity_m_s2: float,
    rotation_period_hours: float,
    primary_mass_jupiter: float,
    eccentricity: float,
    params: ThermalParameters,
) -> ThermalDiagnostics:
    _, mantle_mass, area, _ = _geometry(mass_earth, radius_km, params)
    eta, ra, nu, qconv, lith_km = convective_state(
        state.mantle_temperature_k, radius_km, surface_gravity_m_s2, params
    )
    qrad_power = radiogenic_power_w(state.system_age_myr, mantle_mass, params)
    qtide_power = tidal_heating_power_w(
        eccentricity, radius_km, rotation_period_hours, primary_mass_jupiter, params
    )
    qconv_power = qconv * area
    qrad = qrad_power / area
    qtide = qtide_power / area
    net = qrad + qtide - qconv
    activity = float(np.clip(
        qconv / max(state.reference_convective_flux_w_m2, 1e-12),
        params.min_tectonic_activity_factor,
        params.max_tectonic_activity_factor,
    ))
    return ThermalDiagnostics(
        time_myr=float(state.time_myr),
        system_age_myr=float(state.system_age_myr),
        mantle_temperature_k=float(state.mantle_temperature_k),
        viscosity_pa_s=eta,
        rayleigh_number=ra,
        nusselt_number=nu,
        convective_heat_flux_w_m2=qconv,
        radiogenic_heat_flux_w_m2=float(qrad),
        tidal_heat_flux_w_m2=float(qtide),
        net_heat_flux_w_m2=float(net),
        radiogenic_power_tw=float(qrad_power / 1e12),
        tidal_power_tw=float(qtide_power / 1e12),
        convective_power_tw=float(qconv_power / 1e12),
        thermal_lithosphere_thickness_km=lith_km,
        tectonic_activity_factor=activity,
        eccentricity=float(eccentricity),
    )


def advance_thermal_state(
    state: ThermalState,
    dt_myr: float,
    mass_earth: float,
    radius_km: float,
    surface_gravity_m_s2: float,
    rotation_period_hours: float,
    primary_mass_jupiter: float,
    eccentricity_history: EccentricityHistory,
    params: ThermalParameters,
) -> tuple[ThermalState, ThermalDiagnostics]:
    if dt_myr <= 0.0:
        raise ValueError("dt_myr must be positive")
    _, mantle_mass, area, _ = _geometry(mass_earth, radius_km, params)
    # Midpoint eccentricity makes externally supplied e(t) histories behave
    # sensibly even when the geological step is much longer than the orbit.
    mid_time = float(state.time_myr) + 0.5 * float(dt_myr)
    ecc = eccentricity_history.at(mid_time)
    diag0 = diagnose_thermal_state(
        state,
        mass_earth,
        radius_km,
        surface_gravity_m_s2,
        rotation_period_hours,
        primary_mass_jupiter,
        ecc,
        params,
    )
    heat_capacity = mantle_mass * float(params.mantle_heat_capacity_j_kg_k)
    delta_energy = diag0.net_heat_flux_w_m2 * area * float(dt_myr) * SECONDS_PER_MYR
    new_temp = float(state.mantle_temperature_k) + delta_energy / max(heat_capacity, 1e-30)
    new_temp = max(new_temp, float(params.surface_temperature_k) + 100.0)

    provisional = ThermalState(
        time_myr=float(state.time_myr + dt_myr),
        system_age_myr=float(state.system_age_myr + dt_myr),
        mantle_temperature_k=new_temp,
        reference_convective_flux_w_m2=float(state.reference_convective_flux_w_m2),
        tectonic_activity_factor=float(state.tectonic_activity_factor),
        thermal_lithosphere_thickness_km=float(state.thermal_lithosphere_thickness_km),
    )
    ecc_end = eccentricity_history.at(provisional.time_myr)
    diag = diagnose_thermal_state(
        provisional,
        mass_earth,
        radius_km,
        surface_gravity_m_s2,
        rotation_period_hours,
        primary_mass_jupiter,
        ecc_end,
        params,
    )
    new_state = ThermalState(
        time_myr=provisional.time_myr,
        system_age_myr=provisional.system_age_myr,
        mantle_temperature_k=provisional.mantle_temperature_k,
        reference_convective_flux_w_m2=provisional.reference_convective_flux_w_m2,
        tectonic_activity_factor=diag.tectonic_activity_factor,
        thermal_lithosphere_thickness_km=diag.thermal_lithosphere_thickness_km,
    )
    return new_state, diag


__all__ = [
    "ThermalParameters",
    "ThermalState",
    "ThermalDiagnostics",
    "initialize_thermal_state",
    "advance_thermal_state",
    "diagnose_thermal_state",
    "mantle_viscosity_pa_s",
    "convective_state",
    "radiogenic_power_w",
    "tidal_heating_power_w",
]
