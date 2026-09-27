"""Rollback-safe thermal, Maxwell and damage loading on a split material shell.

This supplies one shared thermal/orbital clock to the coupled contact solver.
Every existing material face retains its mass and column identity. Conductive
columns remain a one-way boundary-temperature model with their own heat ledger;
their energy is not added to the global mantle energy, nor is tidal heat counted
again in them. These helpers do not solve mechanics or reconnect cut mesh edges.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math
from types import SimpleNamespace

import numpy as np

from .genesis import GenesisState, SECONDS_PER_MYR, temperatures
from .genesis_material import material_column_depth
from .genesis_mobile import _RetryStep
from .genesis_onset import advance_orbit_thermal, update_water_access
from .genesis_onset_support import NonlocalLoading
from .genesis_shell import maxwell_factors, principal_tensile
from .genesis_tides import TidalOrbitState, TidalParameters, tidal_strain_cycle


@dataclass(frozen=True)
class ThermalLoading:
    thermal_state: GenesisState
    orbit: TidalOrbitState
    rows: list[dict]
    dt_myr: float
    column_enthalpy: np.ndarray
    boundary_energy_j: float
    tidal_heat_received_j: float  # Energy received in this trial, not cumulative.
    fraction: np.ndarray
    column_depth_km: np.ndarray
    depth_km: np.ndarray
    mean_temperature_k: np.ndarray
    temperature_k: np.ndarray
    retained: np.ndarray
    viscosity_pa_s: np.ndarray
    maxwell_r: np.ndarray
    maxwell_b: np.ndarray
    effective_b: np.ndarray
    retained_thermal_strain: np.ndarray
    memory: np.ndarray
    damage0: np.ndarray
    water_access: np.ndarray
    active: np.ndarray
    surface_temperature_k: float
    ocean_fraction: float
    tidal_parameters: TidalParameters


def _array(value, shape, name, *, positive=False, fraction=False):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    if positive and np.any(result <= 0):
        raise ValueError(f"{name} must be positive")
    if fraction and np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def advance_thermal_loading(source_model, *, mesh, radius_km, layer_mass_kg,
                            column_enthalpy, elastic_strain, damage, water_access,
                            boundary_energy_j, thermal_state, orbit, target_myr,
                            max_step_myr=.01):
    """Advance an already load-bearing shell, without mutating any input.

    ``mesh`` may have disconnected banks and incomplete spherical coverage.
    It must preserve the ordering of the supplied material face fields. The
    isotropic thermal/Maxwell operation preserves the supplied elastic tensor's
    coordinate basis; it does not rotate memory when ``mesh`` moves. Maxwell
    relaxation and cooling act once per interval, independent of Newton retries.
    Newly frozen material is stress free; remelting removes inherited memory.
    A thermal stopping event may shorten the requested interval; ``dt_myr`` and
    both returned clocks always use the actual end time.
    """
    p = source_model.p
    n = mesh.cell_count
    if (not math.isfinite(radius_km) or radius_km <= 0
            or not math.isfinite(boundary_energy_j)
            or not math.isfinite(target_myr) or target_myr <= thermal_state.time_myr
            or thermal_state.time_myr != orbit.time_myr
            or thermal_state.stopped_reason
            or not math.isfinite(max_step_myr) or max_step_myr <= 0):
        raise ValueError("Coupled heat/orbit clocks must agree and advance from a valid state")
    mass = _array(layer_mass_kg, (n, p.column_layers), "layer_mass_kg", positive=True)
    enthalpy = _array(column_enthalpy, mass.shape, "column_enthalpy", positive=True)
    elastic = _array(elastic_strain, (n, 3), "elastic_strain")
    damage = _array(damage, (n,), "damage", fraction=True)
    water = _array(water_access, (n,), "water_access", fraction=True)

    # The shared integrator updates its returned energy ledger. A private copy
    # of the source state makes rollback independent of its internal aliasing.
    next_global, next_orbit, heating, rows = advance_orbit_thermal(
        deepcopy(thermal_state), orbit, source_model.thermal, source_model.tides_p,
        target_myr, max_step_myr)
    dt = next_global.time_myr-thermal_state.time_myr
    if dt <= 0:
        raise _RetryStep("coupled_thermal_no_progress")
    tm0, ts0 = temperatures(np.asarray(thermal_state.energy), source_model.thermal)
    tm1, ts1 = temperatures(np.asarray(next_global.energy), source_model.thermal)
    columns = SimpleNamespace(radius_km=radius_km, layer_mass_kg=mass,
                              column_enthalpy=enthalpy, boundary_energy_j=boundary_energy_j)
    h, boundary = source_model._conduct(columns, mesh, dt, ts0, ts1, tm0, tm1)
    old_fraction, _, old_temperature = source_model._phase(enthalpy, ts0, tm0)
    fraction, mean, temperature = source_model._phase(h, ts1, tm1)
    column_depth = material_column_depth(mesh, radius_km, mass, p.density_kg_m3)
    depth = fraction*column_depth
    if np.any(depth < .25*p.min_load_bearing_thickness_km):
        raise _RetryStep("coupled_partial_melt_limit")
    active = fraction > 0
    shared = np.minimum(old_fraction, fraction)
    retained = np.divide(shared, fraction, out=np.zeros(n), where=active)*(old_fraction > 0)
    weights = np.clip(shared[:, None]*p.column_layers-np.arange(p.column_layers), 0, 1)
    delta_temperature = np.divide(
        np.sum((temperature-old_temperature)*weights, axis=1), weights.sum(axis=1),
        out=np.zeros(n), where=weights.sum(axis=1) > 0)
    eta = np.clip(p.viscosity_reference_pa_s*np.exp(np.clip(
        p.activation_energy_j_mol/8.314462618*(1/np.maximum(mean, 1)
            -1/p.viscosity_reference_temperature_k), -60, 60)),
        p.viscosity_min_pa_s, p.viscosity_max_pa_s)
    r, b = maxwell_factors(dt*SECONDS_PER_MYR, eta/p.young_modulus_pa)
    effective_b = retained*b+(1-retained)
    thermal_strain = retained*b*p.linear_expansion_per_k*delta_temperature
    memory = (retained*r)[:, None]*elastic
    memory[:, :2] -= thermal_strain[:, None]
    damage0 = damage*retained
    liquid = float(rows[-1]["ocean_fraction"])
    next_water = update_water_access(water, damage0, retained, active, ts1, liquid,
                                    dt, source_model.onset_p)
    tides = replace(source_model.tides_p,
        eccentricity=.5*(orbit.eccentricity+next_orbit.eccentricity),
        semimajor_axis_km=.5*(orbit.semimajor_axis_km+next_orbit.semimajor_axis_km))
    return ThermalLoading(next_global, next_orbit, rows, dt, h, boundary,
        heating*dt*SECONDS_PER_MYR*source_model.thermal.area_m2,
        fraction, column_depth, depth, mean, temperature, retained, eta, r, b,
        effective_b, thermal_strain, memory, damage0, next_water, active, ts1,
        liquid, tides)


def evolve_coupled_damage(source_model, loading, mesh, radius_km, stress_pa):
    """Return bulk damage and peak tidal stress, with no smoothing across cuts.

    Stress components must use ``mesh`` face frames. A mechanics solver using
    reference-frame small strains must supply its reference split mesh here.
    The caller checks the resulting damage increment and re-solves mechanical
    equilibrium. The sampled periodic tide changes tensile damage loading; it
    is never accumulated as an extra secular displacement or eigenstrain.
    """
    p = source_model.p
    stress = _array(stress_pa, (mesh.cell_count, 3), "stress_pa")
    nu = p.poisson_ratio
    elasticity = np.array([[1., nu, 0.], [nu, 1., 0.],
                           [0., 0., .5*(1-nu)]])/(1-nu**2)
    tide = np.einsum("ab,pfb->pfa", elasticity,
                     tidal_strain_cycle(mesh, loading.tidal_parameters))
    degradation = p.residual_stiffness+(1-p.residual_stiffness)*(1-loading.damage0)**2
    tide *= (p.young_modulus_pa*degradation)[None, :, None]
    tide[:, ~loading.active] = 0.
    tensile = np.stack([principal_tensile(stress+part) for part in tide])
    tidal_peak = np.max(np.stack([principal_tensile(part) for part in tide]), axis=0)/1e6
    strength = p.tensile_strength_pa*(1-(1-source_model.onset_p.wet_strength_fraction)
                                     *loading.water_access)
    drive = np.mean(np.maximum(tensile/strength-1, 0)**2, axis=0)/p.damage_timescale_myr
    # An entirely separated mesh has no diffusion path: the identity filter is
    # the exact disconnected limit, avoiding the closed-mesh empty-edge shape.
    if len(mesh.shared_edges):
        drive = NonlocalLoading(mesh, radius_km, source_model.onset_p.regularization_km).apply(
            drive, loading.active)
    else:
        drive = np.where(loading.active, drive, 0.)
    hot = np.clip((loading.mean_temperature_k-700)/650, 0, 1)
    healing = 1/p.cold_healing_timescale_myr+hot**4/p.hot_healing_timescale_myr
    rate = drive+healing
    equilibrium = drive/rate
    result = np.where(loading.active, np.clip(equilibrium+
        (loading.damage0-equilibrium)*np.exp(-rate*loading.dt_myr), 0., 1.), 0.)
    return result, tidal_peak
