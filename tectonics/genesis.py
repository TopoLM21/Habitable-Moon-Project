"""Experimental, zero-dimensional magma/steam cooling before plate tectonics.

Two enthalpy reservoirs distinguish mantle and surface temperatures. The model
closes its energy and water budgets, but its atmosphere and heat-transfer laws
are explicit approximations, not a radiative-convective or petrology solver.
It does not construct plates, continents, or a v0.31 handoff checkpoint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

M_EARTH = 5.9722e24
SIGMA = 5.670374419e-8
SECONDS_PER_MYR = 365.25 * 86400.0 * 1e6
ENERGY_SCALE = 1e12  # J/m2; keeps solver state well scaled
CRITICAL_WATER_K = 647.096
CRITICAL_WATER_PA = 22.064e6
FORMAT = "moon_genesis_thermal_checkpoint"
VERSION = 1
MODEL_VERSION = "genesis-thermal-0.1"


@dataclass(frozen=True)
class GenesisParameters:
    radius_km: float = 5287.0
    mass_earth: float = 0.5
    surface_gravity_m_s2: float = 7.12
    water_volume_km3: float = 1126719620.5563111
    water_density_kg_m3: float = 1000.0
    initial_temperature_k: float = 2300.0
    initial_surface_temperature_k: float = 2300.0
    mantle_mass_fraction: float = 0.68
    silicate_heat_capacity_j_kg_k: float = 1200.0
    silicate_latent_heat_j_kg: float = 4e5
    solidus_k: float = 1400.0
    liquidus_k: float = 2000.0
    surface_layer_depth_m: float = 1000.0
    surface_layer_density_kg_m3: float = 3000.0
    water_heat_capacity_j_kg_k: float = 4200.0
    water_latent_heat_j_kg: float = 2.26e6
    critical_transition_width_k: float = 5.0
    stellar_flux_w_m2: float = 1284.1486427638265
    bond_albedo: float = 0.30
    eclipse_fraction: float = 0.0
    giant_absorbed_flux_w_m2: float = 0.0
    tidal_heat_flux_w_m2: float = 0.0
    radiogenic_specific_power_w_kg: float = 5e-12
    radiogenic_half_life_myr: float = 2400.0
    system_age_at_start_myr: float = 0.0
    dry_optical_depth: float = 0.8
    steam_opacity_m2_kg: float = 0.01
    steam_olr_limit_w_m2: float = 282.0
    hot_window_temperature_k: float = 1600.0
    magma_transfer_w_m2_k: float = 100.0
    solid_transfer_w_m2_k: float = 0.0005
    rheology_transition_low_melt: float = 0.25
    rheology_transition_high_melt: float = 0.55
    lid_melt_threshold: float = 0.4
    thermal_diffusivity_m2_s: float = 1e-6

    def validate(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{item.name} must be a finite number")
        nonnegative = {
            "water_volume_km3", "stellar_flux_w_m2", "giant_absorbed_flux_w_m2",
            "tidal_heat_flux_w_m2", "radiogenic_specific_power_w_kg",
            "system_age_at_start_myr", "dry_optical_depth", "steam_opacity_m2_kg",
            "bond_albedo", "eclipse_fraction", "rheology_transition_low_melt",
        }
        for item in fields(self):
            value = getattr(self, item.name)
            if value < 0 or (value == 0 and item.name not in nonnegative):
                raise ValueError(f"{item.name} must be {'nonnegative' if item.name in nonnegative else 'positive'}")
        if not 0 < self.mantle_mass_fraction <= 1:
            raise ValueError("mantle_mass_fraction must be in (0, 1]")
        if not 0 <= self.bond_albedo <= 1 or not 0 <= self.eclipse_fraction <= 1:
            raise ValueError("albedo and eclipse_fraction must be in [0, 1]")
        if not CRITICAL_WATER_K < self.solidus_k < self.liquidus_k:
            raise ValueError("Require water critical temperature < solidus < liquidus")
        if not 0 <= self.rheology_transition_low_melt < self.rheology_transition_high_melt <= 1:
            raise ValueError("Invalid melt rheology interval")
        if not 0 < self.lid_melt_threshold < 1:
            raise ValueError("lid_melt_threshold must be between 0 and 1")
        if min(self.initial_temperature_k, self.initial_surface_temperature_k) < self.liquidus_k:
            raise ValueError("This experiment requires an initially molten mantle and surface")
        if self.solid_transfer_w_m2_k > self.magma_transfer_w_m2_k:
            raise ValueError("Solid heat transfer must not exceed magma heat transfer")
        if self.surface_column_kg_m2 >= self.silicate_column_kg_m2:
            raise ValueError("Surface layer consumes the entire silicate reservoir")
        if self.critical_transition_width_k >= CRITICAL_WATER_K - 273.16:
            raise ValueError("Critical transition width must stay above the freezing limit")
        if self.water_column_kg_m2 * self.surface_gravity_m_s2 > 1.2 * CRITICAL_WATER_PA:
            raise ValueError("Water inventory above 1.2 critical column pressures needs a supercritical-water EOS")

    @property
    def area_m2(self) -> float:
        return 4 * math.pi * (1000 * self.radius_km) ** 2

    @property
    def water_column_kg_m2(self) -> float:
        return self.water_volume_km3 * 1e9 * self.water_density_kg_m3 / self.area_m2

    @property
    def silicate_column_kg_m2(self) -> float:
        return M_EARTH * self.mass_earth * self.mantle_mass_fraction / self.area_m2

    @property
    def surface_column_kg_m2(self) -> float:
        return self.surface_layer_depth_m * self.surface_layer_density_kg_m3

    @property
    def mantle_column_kg_m2(self) -> float:
        return self.silicate_column_kg_m2 - self.surface_column_kg_m2

    @property
    def absorbed_stellar_flux_w_m2(self) -> float:
        return self.stellar_flux_w_m2 * (1 - self.bond_albedo) * (1 - self.eclipse_fraction) / 4


def parameters_from_config(config: dict) -> GenesisParameters:
    """No implicit relief-based calibration: a genesis water inventory is required."""
    section = config.get("genesis")
    if not isinstance(section, dict) or section.get("schema_version") != 1:
        raise ValueError("Configuration needs genesis.schema_version: 1")
    if section.get("water_volume_km3") is None:
        raise ValueError("genesis.water_volume_km3 must be explicit; null relief calibration is unavailable")
    names = {f.name for f in fields(GenesisParameters)}
    allowed_metadata = {"schema_version", "water_inventory_source", "stellar_flux_source", "description"}
    unknown = set(section) - names - allowed_metadata
    if unknown:
        raise ValueError(f"Unknown genesis parameters: {sorted(unknown)}")
    try:
        if any(isinstance(v, bool) for k, v in section.items() if k in names):
            raise ValueError("Genesis parameters must be numbers, not booleans")
        values = {k: float(v) for k, v in section.items() if k in names}
    except (TypeError, OverflowError) as exc:
        raise ValueError("Genesis parameters must be finite numbers, not null") from exc
    for key in ("radius_km", "mass_earth", "surface_gravity_m_s2"):
        if key not in values and key in config.get("moon", {}):
            try:
                if isinstance(config["moon"][key], bool):
                    raise ValueError(f"moon.{key} must be a number")
                values[key] = float(config["moon"][key])
            except (TypeError, OverflowError) as exc:
                raise ValueError(f"moon.{key} must be a finite number, not null") from exc
    result = GenesisParameters(**values)
    result.validate()
    return result


def saturation_pressure_pa(temperature_k: float) -> float:
    """IAPWS saturation curve over liquid water, 273.16 K to critical.

    Below the triple point this experiment terminates (ice is not modelled).
    A continuous extrapolation is used only during implicit solver iterations.
    """
    t = max(float(temperature_k), 1.0)
    if t >= CRITICAL_WATER_K:
        return CRITICAL_WATER_PA
    if t < 273.16:
        return 611.657 * math.exp(5420.0 * (1 / 273.16 - 1 / t))
    theta = 1 - t / CRITICAL_WATER_K
    terms = ((-7.85951783, 1), (1.84408259, 1.5), (-11.7866497, 3),
             (22.6807411, 3.5), (-15.9618719, 4), (1.80122502, 7.5))
    return CRITICAL_WATER_PA * math.exp(CRITICAL_WATER_K / t * sum(a * theta**b for a, b in terms))


def melt_fraction(temperature_k: float, p: GenesisParameters) -> float:
    return min(1.0, max(0.0, (temperature_k - p.solidus_k) / (p.liquidus_k - p.solidus_k)))


def vapor_column_kg_m2(temperature_k: float, p: GenesisParameters) -> float:
    if temperature_k >= CRITICAL_WATER_K:
        return p.water_column_kg_m2
    saturated = min(p.water_column_kg_m2, saturation_pressure_pa(temperature_k) / p.surface_gravity_m_s2)
    if p.water_column_kg_m2 * p.surface_gravity_m_s2 > CRITICAL_WATER_PA:
        # Near-critical inventories (canonical reference: 228 bar) otherwise
        # produce an enthalpy jump at Tc in this column approximation. Smooth
        # only the last 5 K below Tc, conserving mass/enthalpy; this is NOT a
        # supercritical equation of state or exact saturation in that interval.
        x = min(1.0, max(0.0, (CRITICAL_WATER_K - temperature_k) / p.critical_transition_width_k))
        weight = x*x*(3 - 2*x)
        return p.water_column_kg_m2 + weight * (saturated - p.water_column_kg_m2)
    return saturated


def silicate_specific_enthalpy(temperature_k: float, p: GenesisParameters) -> float:
    return p.silicate_heat_capacity_j_kg_k * temperature_k + p.silicate_latent_heat_j_kg * melt_fraction(temperature_k, p)


def mantle_enthalpy(temperature_k: float, p: GenesisParameters) -> float:
    return p.mantle_column_kg_m2 * silicate_specific_enthalpy(temperature_k, p)


def surface_enthalpy(temperature_k: float, p: GenesisParameters) -> float:
    return (p.surface_column_kg_m2 * silicate_specific_enthalpy(temperature_k, p)
            + p.water_column_kg_m2 * p.water_heat_capacity_j_kg_k * temperature_k
            + p.water_latent_heat_j_kg * vapor_column_kg_m2(temperature_k, p))


def _invert_silicate(enthalpy: float, mass: float, extra_capacity: float, p: GenesisParameters) -> float:
    capacity = mass * p.silicate_heat_capacity_j_kg_k + extra_capacity
    latent = mass * p.silicate_latent_heat_j_kg
    low = capacity * p.solidus_k
    high = capacity * p.liquidus_k + latent
    if enthalpy <= low:
        return enthalpy / capacity
    if enthalpy >= high:
        return (enthalpy - latent) / capacity
    return (enthalpy + latent * p.solidus_k / (p.liquidus_k - p.solidus_k)) / (capacity + latent / (p.liquidus_k - p.solidus_k))


def temperatures(y: np.ndarray, p: GenesisParameters) -> tuple[float, float]:
    tm = _invert_silicate(float(y[0]) * ENERGY_SCALE, p.mantle_column_kg_m2, 0.0, p)
    es = float(y[1]) * ENERGY_SCALE
    if es >= surface_enthalpy(CRITICAL_WATER_K, p):
        ts = _invert_silicate(es - p.water_latent_heat_j_kg * p.water_column_kg_m2,
                              p.surface_column_kg_m2, p.water_column_kg_m2 * p.water_heat_capacity_j_kg_k, p)
    else:
        # Allow small negative stage iterates; accepted states are checked below.
        ts = brentq(lambda t: surface_enthalpy(t, p) - es, min(-100.0, es / (p.surface_column_kg_m2 * p.silicate_heat_capacity_j_kg_k)), CRITICAL_WATER_K, xtol=1e-9)
    return tm, ts


def outgoing_longwave_w_m2(temperature_k: float, p: GenesisParameters) -> float:
    """Explicit grey/steam-limit surrogate with a hot thermal window.

    282 W/m2 is a configurable Earth-like steam-branch approximation, never
    a universal cap. Thin/dry atmospheres and very hot surfaces radiate more.
    """
    blackbody = SIGMA * max(temperature_k, 0.0) ** 4
    dry = blackbody / (1 + 0.75 * p.dry_optical_depth)
    tau = p.steam_opacity_m2_kg * vapor_column_kg_m2(temperature_k, p)
    steam_weight = -math.expm1(-tau)
    hot_window = SIGMA * max(0.0, temperature_k - p.hot_window_temperature_k) ** 4
    steam = min(dry, p.steam_olr_limit_w_m2 + hot_window)
    return (1 - steam_weight) * dry + steam_weight * steam


def fluxes(time_myr: float, y: np.ndarray, p: GenesisParameters) -> dict[str, float]:
    tm, ts = temperatures(y, p)
    phi = melt_fraction(tm, p)
    transition = min(1.0, max(0.0, (phi - p.rheology_transition_low_melt) /
                            (p.rheology_transition_high_melt - p.rheology_transition_low_melt)))
    weight = transition**2 * (3 - 2 * transition)
    transfer = math.exp((1 - weight) * math.log(p.solid_transfer_w_m2_k) + weight * math.log(p.magma_transfer_w_m2_k))
    qmantle = transfer * (tm - ts)
    qrad = p.silicate_column_kg_m2 * p.radiogenic_specific_power_w_kg * 2**(-(time_myr + p.system_age_at_start_myr) / p.radiogenic_half_life_myr)
    olr = outgoing_longwave_w_m2(ts, p)
    return {"mantle_to_surface_flux_w_m2": qmantle, "radiogenic_flux_w_m2": qrad,
            "tidal_flux_w_m2": p.tidal_heat_flux_w_m2,
            "absorbed_stellar_flux_w_m2": p.absorbed_stellar_flux_w_m2,
            "giant_absorbed_flux_w_m2": p.giant_absorbed_flux_w_m2,
            "outgoing_longwave_w_m2": olr,
            "net_cooling_flux_w_m2": olr - p.absorbed_stellar_flux_w_m2 - p.giant_absorbed_flux_w_m2 - qrad - p.tidal_heat_flux_w_m2}


def _rhs(time_myr: float, y: np.ndarray, p: GenesisParameters) -> np.ndarray:
    q = fluxes(time_myr, y, p)
    internal = q["radiogenic_flux_w_m2"] + q["tidal_flux_w_m2"]
    external = p.absorbed_stellar_flux_w_m2 + p.giant_absorbed_flux_w_m2
    exchange = q["mantle_to_surface_flux_w_m2"]
    return np.array([internal - exchange, exchange + external - q["outgoing_longwave_w_m2"],
                     internal + external, q["outgoing_longwave_w_m2"]]) * (SECONDS_PER_MYR / ENERGY_SCALE)


@dataclass
class GenesisState:
    time_myr: float
    energy: list[float]
    initial_total_energy: float
    events: dict[str, float]
    stopped_reason: str | None = None


def initial_state(p: GenesisParameters) -> GenesisState:
    p.validate()
    energy = [mantle_enthalpy(p.initial_temperature_k, p) / ENERGY_SCALE,
              surface_enthalpy(p.initial_surface_temperature_k, p) / ENERGY_SCALE, 0.0, 0.0]
    return GenesisState(0.0, energy, energy[0] + energy[1], {})


def diagnose(state: GenesisState, p: GenesisParameters) -> dict:
    y = np.asarray(state.energy)
    tm, ts = temperatures(y, p)
    vapor = vapor_column_kg_m2(ts, p)
    ocean_column = p.water_column_kg_m2 - vapor
    ocean = ocean_column * p.area_m2 / p.water_density_kg_m3 / 1e9
    phi_s = melt_fraction(ts, p)
    lid_time = state.events.get("surface_lid")
    lid_age_s = max(0.0, state.time_myr - lid_time) * SECONDS_PER_MYR if lid_time is not None else 0.0
    # A diagnostic cooling length, not a differentiated crust or mechanical lid.
    cooling_depth = 2 * math.sqrt(p.thermal_diffusivity_m2_s * lid_age_s) / 1000 if phi_s <= p.lid_melt_threshold else 0.0
    phase = "magma_ocean_cooling" if phi_s > p.lid_melt_threshold else "lid_formation"
    if ocean_column > 1e-6 * max(p.water_column_kg_m2, 1):
        phase = "ocean_condensation"
    residual = y[0] + y[1] - state.initial_total_energy - y[2] + y[3]
    return {"time_myr": state.time_myr, "phase": phase,
            "mantle_temperature_k": tm, "surface_temperature_k": ts,
            "mantle_melt_fraction": melt_fraction(tm, p), "surface_melt_fraction": phi_s,
            "steam_pressure_bar": vapor * p.surface_gravity_m_s2 / 1e5,
            "vapor_mass_kg": vapor * p.area_m2, "ocean_mass_kg": ocean_column * p.area_m2,
            "total_water_mass_kg": p.water_column_kg_m2 * p.area_m2,
            "ocean_volume_km3": ocean, "water_inventory_km3": p.water_volume_km3,
            "ocean_fraction": ocean / p.water_volume_km3 if p.water_volume_km3 else 0.0,
            "cooling_depth_proxy_km": cooling_depth,
            "energy_residual_j_m2": float(residual * ENERGY_SCALE),
            "relative_energy_residual": float(residual / state.initial_total_energy),
            **fluxes(state.time_myr, y, p)}


def advance(state: GenesisState, p: GenesisParameters, target_time_myr: float,
            max_step_myr: float = 0.01, rtol: float = 1e-7) -> tuple[GenesisState, list[dict]]:
    """Advance with implicit adaptive substeps; include event states in history."""
    if not all(math.isfinite(v) for v in (target_time_myr, max_step_myr, rtol)) or max_step_myr <= 0 or rtol <= 0:
        raise ValueError("Finite positive integration controls required")
    if target_time_myr <= state.time_myr or state.stopped_reason:
        raise ValueError("Target must follow a non-stopped state")

    def surface_lid(t, y):
        return temperatures(y, p)[1] - (p.solidus_k + p.lid_melt_threshold * (p.liquidus_k - p.solidus_k))

    def mantle_rheology(t, y):
        return temperatures(y, p)[0] - (p.solidus_k + p.lid_melt_threshold * (p.liquidus_k - p.solidus_k))

    dewpoint = condensation_temperature_k(p)

    def ocean_start(t, y):
        return temperatures(y, p)[1] - dewpoint

    def freezing_limit(t, y):
        return temperatures(y, p)[1] - 273.16

    freezing_limit.terminal = True
    event_names = ["surface_lid", "mantle_rheology", "ocean_start", "freezing_limit"]
    events = [surface_lid, mantle_rheology, ocean_start, freezing_limit]
    if p.water_volume_km3 == 0:
        del events[2]
        del event_names[2]
    for event in events:
        event.direction = -1
    solution = solve_ivp(lambda t, y: _rhs(t, y, p), (state.time_myr, target_time_myr), state.energy,
                         method="Radau", max_step=max_step_myr, rtol=rtol, atol=1e-9, events=events)
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError(f"Genesis integration failed: {solution.message}")
    found = dict(state.events)
    event_points = []
    for name, times, values in zip(event_names, solution.t_events, solution.y_events):
        if len(times) and name not in found:
            event_points.append((float(times[0]), name, values[0].tolist()))
    rows = []
    for time, name, energy in sorted(event_points):
        found[name] = time
        rows.append(diagnose(GenesisState(time, energy, state.initial_total_energy, dict(found)), p))
    reason = "surface_reached_freezing_limit_ice_not_modelled" if solution.status == 1 else None
    result = GenesisState(float(solution.t[-1]), solution.y[:, -1].tolist(), state.initial_total_energy, found, reason)
    if min(temperatures(solution.y[:, -1], p)) <= 0:
        raise RuntimeError("Nonphysical nonpositive temperature")
    rows.append(diagnose(result, p))
    return result, rows


def condensation_temperature_k(p: GenesisParameters) -> float:
    pressure = p.water_column_kg_m2 * p.surface_gravity_m_s2
    if pressure <= 0:
        return 0.0
    if pressure >= CRITICAL_WATER_PA:
        return CRITICAL_WATER_K
    return brentq(lambda t: saturation_pressure_pa(t) - pressure, 1.0, CRITICAL_WATER_K)


def run_genesis(p: GenesisParameters, duration_myr: float = 10.0, sample_interval_myr: float = 0.1,
                max_step_myr: float = 0.01, state: GenesisState | None = None,
                on_sample: Callable[[GenesisState, list[dict]], None] | None = None) -> tuple[GenesisState, list[dict]]:
    p.validate()
    if not all(math.isfinite(v) and v > 0 for v in (duration_myr, sample_interval_myr, max_step_myr)):
        raise ValueError("Duration, sampling interval and max step must be finite and positive")
    state = state or initial_state(p)
    if duration_myr <= state.time_myr:
        raise ValueError("Duration is the absolute end time and must follow the checkpoint")
    rows = [diagnose(state, p)]
    # Anchor output boundaries to t=0 so resume at a sample follows the same calls.
    while state.time_myr < duration_myr - 1e-12 and not state.stopped_reason:
        index = math.floor((state.time_myr + 1e-10 * sample_interval_myr) / sample_interval_myr) + 1
        target = min(duration_myr, index * sample_interval_myr)
        state, new_rows = advance(state, p, target, max_step_myr)
        rows.extend(new_rows)
        if on_sample is not None:
            on_sample(state, new_rows)
    return state, rows


def parameter_hash(p: GenesisParameters) -> str:
    return hashlib.sha256(json.dumps(asdict(p), sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_checkpoint(path: str | Path, state: GenesisState, p: GenesisParameters,
                    controls: dict | None = None, provenance: dict | None = None) -> None:
    path = Path(path)
    payload = {"format": FORMAT, "version": VERSION, "model_version": MODEL_VERSION,
               "parameter_hash": parameter_hash(p), "parameters": asdict(p), "state": asdict(state),
               "controls": controls or {}, "provenance": provenance or {}}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_checkpoint(path: str | Path) -> tuple[GenesisState, GenesisParameters, dict]:
    try:
        return _load_checkpoint(path)
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("Malformed genesis checkpoint") from exc


def _load_checkpoint(path: str | Path) -> tuple[GenesisState, GenesisParameters, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT or payload.get("version") != VERSION or payload.get("model_version") != MODEL_VERSION:
        raise ValueError("Unsupported genesis checkpoint; mature tectonics checkpoints are separate")
    p = GenesisParameters(**payload["parameters"])
    p.validate()
    if payload.get("parameter_hash") != parameter_hash(p):
        raise ValueError("Genesis checkpoint parameter hash mismatch")
    state = GenesisState(**payload["state"])
    if (len(state.energy) != 4 or not all(math.isfinite(v) for v in state.energy)
            or not math.isfinite(state.time_myr) or state.time_myr < 0
            or not math.isfinite(state.initial_total_energy) or state.initial_total_energy <= 0
            or any(not math.isfinite(t) or t < 0 or t > state.time_myr for t in state.events.values())):
        raise ValueError("Invalid genesis checkpoint state")
    if abs(diagnose(state, p)["relative_energy_residual"]) > 1e-5:
        raise ValueError("Genesis checkpoint does not close its energy budget")
    if min(state.energy) < 0 or min(temperatures(np.asarray(state.energy), p)) <= 0:
        raise ValueError("Genesis checkpoint contains nonphysical energy or temperature")
    return state, p, payload
