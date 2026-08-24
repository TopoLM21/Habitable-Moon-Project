import numpy as np

from tectonics.thermal import (
    ThermalParameters,
    advance_thermal_state,
    convective_state,
    initialize_thermal_state,
    radiogenic_power_w,
    tidal_heating_power_w,
)
from tectonics.tides import constant_eccentricity


def test_canonical_moon_initializes_finite_thermal_state():
    p = ThermalParameters()
    s = initialize_thermal_state(0.5, 5287.0, 7.12, p)
    assert s.mantle_temperature_k == p.initial_mantle_temperature_k
    assert s.reference_convective_flux_w_m2 > 0.0
    assert s.thermal_lithosphere_thickness_km > 0.0


def test_higher_gravity_increases_rayleigh_and_heat_loss():
    p = ThermalParameters()
    _, ra1, _, q1, _ = convective_state(1850.0, 5287.0, 3.56, p)
    _, ra2, _, q2, _ = convective_state(1850.0, 5287.0, 7.12, p)
    assert ra2 > ra1
    assert q2 > q1


def test_radiogenic_power_decays_with_system_age():
    p = ThermalParameters()
    mantle_mass = 2.0e24
    assert radiogenic_power_w(4000.0, mantle_mass, p) < radiogenic_power_w(500.0, mantle_mass, p)


def test_tidal_heating_scales_as_eccentricity_squared():
    p = ThermalParameters()
    a = tidal_heating_power_w(0.00047, 5287.0, 47.0, 5.0, p)
    b = tidal_heating_power_w(0.00094, 5287.0, 47.0, 5.0, p)
    assert np.isclose(b / a, 4.0, rtol=1e-12)


def test_thermal_step_advances_age_and_stays_finite():
    p = ThermalParameters()
    s = initialize_thermal_state(0.5, 5287.0, 7.12, p)
    n, d = advance_thermal_state(s, 4.0, 0.5, 5287.0, 7.12, 47.0, 5.0, constant_eccentricity(0.00047), p)
    assert n.time_myr == 4.0
    assert n.system_age_myr == p.system_age_at_start_myr + 4.0
    assert np.isfinite(n.mantle_temperature_k)
    assert d.convective_heat_flux_w_m2 > 0.0
    assert d.radiogenic_heat_flux_w_m2 > 0.0
    assert d.tidal_heat_flux_w_m2 > 0.0
