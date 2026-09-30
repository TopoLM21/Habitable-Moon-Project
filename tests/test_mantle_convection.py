"""Shared-law checks and exact fixtures captured before the mature extraction."""

from dataclasses import asdict, replace

import pytest

from tectonics.mantle_convection import mantle_convection_state
from tectonics.thermal import (
    ThermalParameters,
    advance_thermal_state,
    convective_state,
    initialize_thermal_state,
)
from tectonics.tides import constant_eccentricity


@pytest.mark.parametrize(
    "temperature,radius,gravity,expected",
    [
        (1850.0, 5287.0, 7.12, (4.748025073272406e19, 378525148.11721265, 19.5311887397848, 0.05129179213003612, 121.8128620688458)),
        (1400.0, 5287.0, 7.12, (2.506771939744813e22, 510407.06218106236, 2.1577576040627258, 0.00403409025192653, 1102.602996518435)),
        (2300.0, 5287.0, 7.12, (1.0453870423558266e18, 22145096439.20284, 75.82102003888531, 0.256481335465586, 31.37850161841449)),
        (288.5, 5287.0, 7.12, (1e24, 11.506062062257852, 1.0, 1.681272723451653e-6, 2379.15)),
        (288.0, 5287.0, 7.12, (1e24, 11.506062062257852, 1.0, 1.681272723451653e-6, 2379.15)),
        (-100.0, 5287.0, 7.12, (1e24, 11.506062062257852, 1.0, 1.681272723451653e-6, 2379.15)),
        (1e8, 5287.0, 7.12, (1e18, 1150602892479911.5, 2829.256037410289, 475673.7303915938, 0.8409101080076564)),
        (1850.0, 5287.0, 0.0, (4.748025073272406e19, 0.0, 1.0, 0.002626147994031482, 2379.15)),
        (1850.0, 5.0, 7.12, (4.748025073272406e19, 0.3201671866809, 1.0, 2.7768888888888887, 2.25)),
    ],
)
def test_mature_convection_matches_pre_extraction_exactly(temperature, radius, gravity, expected):
    # Exact equality deliberately guards arithmetic order, the 1 K floor,
    # viscosity clipping and the legacy public tuple contract.
    assert convective_state(temperature, radius, gravity, ThermalParameters()) == expected


def test_mature_custom_rheology_matches_pre_extraction_exactly():
    params = replace(
        ThermalParameters(),
        mantle_depth_fraction_radius=0.3,
        mantle_density_kg_m3=3500.0,
        thermal_conductivity_w_m_k=3.0,
        thermal_expansivity_per_k=2.5e-5,
        thermal_diffusivity_m2_s=1.2e-6,
        viscosity_reference_pa_s=8e20,
        viscosity_reference_temperature_k=1700.0,
        activation_energy_j_mol=2.5e5,
        viscosity_min_pa_s=1e17,
        viscosity_max_pa_s=1e25,
        critical_rayleigh=1200.0,
        nusselt_prefactor=0.4,
        nusselt_exponent=0.28,
    )
    assert convective_state(1750.0, 6371.0, 9.81, params) == (
        4.826374246128699e20,
        15128949.439879913,
        5.626465995118268,
        0.012911463325793295,
        339.69813407888984,
    )


def test_mature_thermal_evolution_matches_pre_extraction_exactly():
    params = ThermalParameters()
    state = initialize_thermal_state(0.5, 5287.0, 7.12, params)
    assert asdict(state) == {
        "time_myr": 0.0,
        "system_age_myr": 500.0,
        "mantle_temperature_k": 1850.0,
        "reference_convective_flux_w_m2": 0.05129179213003612,
        "tectonic_activity_factor": 1.0,
        "thermal_lithosphere_thickness_km": 121.8128620688458,
    }
    temperatures = []
    for dt in (4.0, 25.0, 100.0):
        state, diagnostics = advance_thermal_state(
            state, dt, 0.5, 5287.0, 7.12, 47.0, 5.0,
            constant_eccentricity(0.00047), params,
        )
        temperatures.append(state.mantle_temperature_k)
    assert temperatures == [1849.6794332983202, 1847.68076973704, 1839.8071245928093]
    assert asdict(state) == {
        "time_myr": 129.0,
        "system_age_myr": 629.0,
        "mantle_temperature_k": 1839.8071245928093,
        "reference_convective_flux_w_m2": 0.05129179213003612,
        "tectonic_activity_factor": 0.9562392585395523,
        "thermal_lithosphere_thickness_km": 126.55616018018453,
    }
    assert asdict(diagnostics) == {
        "time_myr": 129.0,
        "system_age_myr": 629.0,
        "mantle_temperature_k": 1839.8071245928093,
        "viscosity_pa_s": 5.28981083827859e19,
        "rayleigh_number": 337539273.711141,
        "nusselt_number": 18.79916391752627,
        "convective_heat_flux_w_m2": 0.04904722527559058,
        "radiogenic_heat_flux_w_m2": 0.024102374767944534,
        "tidal_heat_flux_w_m2": 0.008657971691135144,
        "net_heat_flux_w_m2": -0.016286878816510898,
        "radiogenic_power_tw": 8.46619602510081,
        "tidal_power_tw": 3.04119765054897,
        "convective_power_tw": 17.228319933963135,
        "thermal_lithosphere_thickness_km": 126.55616018018453,
        "tectonic_activity_factor": 0.9562392585395523,
        "eccentricity": 0.00047,
    }


def test_shared_convection_reports_conductive_and_boundary_layer_scales():
    state = mantle_convection_state(
        1850.0, 5287.0, 7.12, ThermalParameters(), surface_temperature_k=288.0,
    )
    assert state.mantle_depth_m == 2379150.0
    assert state.conductive_heat_flux_w_m2 == 0.002626147994031482
    assert state.convective_heat_flux_w_m2 == 0.05129179213003612
    assert state.thermal_lithosphere_thickness_km == 121.8128620688458
    assert state.effective_conductance_w_m2_k * 1562.0 == pytest.approx(state.convective_heat_flux_w_m2)


@pytest.mark.parametrize("surface_temperature", [1850.0, 2000.0])
def test_zero_floor_has_no_unstable_convection_or_spurious_outward_flux(surface_temperature):
    state = mantle_convection_state(
        1850.0, 5287.0, 7.12, ThermalParameters(),
        surface_temperature_k=surface_temperature, min_delta_temperature_k=0.0,
    )
    assert state.rayleigh_number == 0.0
    assert state.nusselt_number == 1.0
    assert state.conductive_heat_flux_w_m2 == 0.0
    assert state.convective_heat_flux_w_m2 == 0.0
    assert state.thermal_lithosphere_thickness_km == 2379.15
    assert state.effective_conductance_w_m2_k == 1.681272723451653e-6
