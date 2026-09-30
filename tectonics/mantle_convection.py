"""Shared zero-dimensional boundary-layer mantle heat-transfer scaling.

These laws are the mature thermal model's Arrhenius viscosity and Ra/Nu
parameterization. They describe thermal transport only; no plate mechanics or
additional convection calibration is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

R_GAS = 8.31446261815324


class ConvectionParameters(Protocol):
    """Scalar material parameters supplied by either thermal model."""

    mantle_depth_fraction_radius: float
    mantle_density_kg_m3: float
    thermal_conductivity_w_m_k: float
    thermal_expansivity_per_k: float
    thermal_diffusivity_m2_s: float
    viscosity_reference_pa_s: float
    viscosity_reference_temperature_k: float
    activation_energy_j_mol: float
    viscosity_min_pa_s: float
    viscosity_max_pa_s: float
    critical_rayleigh: float
    nusselt_prefactor: float
    nusselt_exponent: float


@dataclass(slots=True, frozen=True)
class ConvectionDiagnostics:
    viscosity_pa_s: float
    rayleigh_number: float
    nusselt_number: float
    conductive_heat_flux_w_m2: float
    convective_heat_flux_w_m2: float
    mantle_depth_m: float
    thermal_lithosphere_thickness_km: float
    effective_conductance_w_m2_k: float


def mantle_viscosity_pa_s(temperature_k: float, params: ConvectionParameters) -> float:
    t = max(float(temperature_k), 1.0)
    tref = max(float(params.viscosity_reference_temperature_k), 1.0)
    exponent = float(params.activation_energy_j_mol) / R_GAS * (1.0 / t - 1.0 / tref)
    exponent = float(np.clip(exponent, -60.0, 60.0))
    eta = float(params.viscosity_reference_pa_s) * math.exp(exponent)
    return float(np.clip(eta, params.viscosity_min_pa_s, params.viscosity_max_pa_s))


def mantle_convection_state(
    temperature_k: float,
    radius_km: float,
    surface_gravity_m_s2: float,
    params: ConvectionParameters,
    *,
    surface_temperature_k: float,
    min_delta_temperature_k: float = 1.0,
) -> ConvectionDiagnostics:
    """Diagnose outward transport from the mantle to an explicit surface.

    The 1 K default temperature-contrast floor preserves the mature model's
    numerical behavior. A two-reservoir model can use a zero floor to turn off
    convection for a stable, inverted gradient, and use the returned finite
    conductance with its signed temperature contrast for heat exchange.

    ``convective_heat_flux_w_m2`` is the total conductive-plus-convective flux
    k * delta_T / depth * Nu, retaining the mature model's arithmetic order.
    The depth/Nu diagnostic is a thermal boundary-layer scale, not a mechanical
    lid or plate thickness.
    """
    radius_m = float(radius_km) * 1000.0
    depth = radius_m * float(params.mantle_depth_fraction_radius)
    delta_t = max(float(temperature_k) - float(surface_temperature_k), float(min_delta_temperature_k))
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
    return ConvectionDiagnostics(
        viscosity_pa_s=eta,
        rayleigh_number=float(ra),
        nusselt_number=float(nu),
        conductive_heat_flux_w_m2=float(conductive),
        convective_heat_flux_w_m2=float(flux),
        mantle_depth_m=float(depth),
        thermal_lithosphere_thickness_km=float(thermal_lithosphere_km),
        effective_conductance_w_m2_k=float(params.thermal_conductivity_w_m_k) * nu / depth,
    )


__all__ = [
    "ConvectionParameters",
    "ConvectionDiagnostics",
    "R_GAS",
    "mantle_viscosity_pa_s",
    "mantle_convection_state",
]
