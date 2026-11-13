"""Opt-in reconstruction: conservative time integration of the existing thermal law.

This is a fixed-mass, well-mixed mantle reservoir, not a closed tectonic energy
model. Radiogenic energy and prescribed orbital heating are explicit inputs;
surface heat loss is an output. No plate work or crust enthalpy is invented.
The v0.31 thermal constitutive functions and parameters are reused unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math

import numpy as np

from .thermal import (
    ThermalParameters, ThermalState, SECONDS_PER_MYR, _geometry,
    convective_state, diagnose_thermal_state, radiogenic_power_w,
    tidal_heating_power_w,
)
from .tides import EccentricityHistory


@dataclass(frozen=True)
class ThermalIntegrationParameters:
    temperature_atol_k: float = 0.1
    temperature_rtol: float = 1e-5
    exchange_rtol: float = 1e-4
    exchange_atol_j_per_myr: float = 1e19
    initial_step_myr: float = 0.25
    max_step_myr: float = 4.0
    max_trials: int = 10000

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not isinstance(self.max_trials, int):
            raise ValueError("max_trials must be an integer")


def _add(pair: list[float], value: float) -> None:
    """Neumaier accumulation, persisted in full at restart."""
    high, low = pair
    total = high + value
    low += (high - total) + value if abs(high) >= abs(value) else (value - total) + high
    pair[:] = [total, low]


class ThermalBudgetIntegrator:
    VERSION = 1

    def __init__(self, state: ThermalState,
                 options: ThermalIntegrationParameters = ThermalIntegrationParameters()):
        self.options = options
        self.data = {
            "version": self.VERSION, "options": asdict(options),
            "origin_time_myr": state.time_myr,
            "origin_system_age_myr": state.system_age_myr,
            "origin_temperature_k": state.mantle_temperature_k,
            "time_myr": state.time_myr,
            "next_step_myr": options.initial_step_myr,
            "accepted_steps": 0, "rejected_steps": 0,
            "max_accepted_error_ratio": 0.0,
            "energy_change_j": [0.0, 0.0],
            "radiogenic_input_j": [0.0, 0.0],
            "orbital_input_j": [0.0, 0.0],
            "surface_output_j": [0.0, 0.0],
            "physical_signature": None, "history": [],
        }

    def checkpoint(self) -> dict:
        return deepcopy(self.data)

    @classmethod
    def restore(cls, data: dict, options: ThermalIntegrationParameters | None = None):
        if data.get("version") != cls.VERSION:
            raise ValueError("Unsupported thermal budget checkpoint")
        required = {"options", "origin_time_myr", "origin_system_age_myr",
                    "origin_temperature_k", "time_myr", "next_step_myr",
                    "accepted_steps", "rejected_steps", "max_accepted_error_ratio",
                    "energy_change_j", "radiogenic_input_j", "orbital_input_j",
                    "surface_output_j", "physical_signature", "history"}
        if required - data.keys():
            raise ValueError(f"Incomplete thermal budget checkpoint: {sorted(required - data.keys())}")
        for key in ("energy_change_j", "radiogenic_input_j", "orbital_input_j", "surface_output_j"):
            pair = data[key]
            if len(pair) != 2 or not all(math.isfinite(value) for value in pair):
                raise ValueError(f"Invalid compensated energy field: {key}")
        for key in ("origin_time_myr", "origin_system_age_myr", "origin_temperature_k",
                    "time_myr", "next_step_myr", "max_accepted_error_ratio"):
            if not math.isfinite(data[key]):
                raise ValueError(f"Invalid thermal controller field: {key}")
        if data["next_step_myr"] <= 0 or data["time_myr"] < data["origin_time_myr"]:
            raise ValueError("Invalid thermal controller chronology")
        saved = ThermalIntegrationParameters(**data["options"])
        if options is not None and options != saved:
            raise ValueError("Thermal controller options changed at restart")
        obj = cls.__new__(cls)
        obj.options, obj.data = saved, deepcopy(data)
        return obj

    def advance(self, state: ThermalState, dt_myr: float, mass_earth: float,
                radius_km: float, gravity: float, period_hours: float,
                primary_mass_jupiter: float, eccentricity: EccentricityHistory,
                params: ThermalParameters):
        """Return state/diagnostics, committing controller only on full success.

        Step doubling estimates thermal truncation error; it does NOT certify
        mechanical/fracture events. Known eccentricity knots are step endpoints.
        Domain violations fail explicitly instead of injecting a temperature floor.
        """
        if not math.isfinite(dt_myr) or dt_myr <= 0 or state.time_myr != self.data["time_myr"]:
            raise ValueError("Invalid time increment or mismatched controller time")
        target = state.time_myr + dt_myr
        if not math.isfinite(target) or target <= state.time_myr:
            raise ValueError("Time increment is not representable")
        _, mass, area, _ = _geometry(mass_earth, radius_km, params)
        capacity = mass * params.mantle_heat_capacity_j_kg_k
        if capacity <= 0 or not math.isfinite(capacity):
            raise ValueError("Invalid mantle heat capacity")
        times = np.asarray(eccentricity.times_myr, dtype=float)
        ecc = np.asarray(eccentricity.eccentricity, dtype=float)
        if (not len(times) or times.shape != ecc.shape or
                np.any(~np.isfinite(times)) or np.any(~np.isfinite(ecc)) or
                np.any(np.diff(times) <= 0) or np.any(ecc < 0) or np.any(ecc >= 1)):
            raise ValueError("Eccentricity history must have ordered finite knots and 0 <= e < 1")
        signature = {"parameters": asdict(params), "mass_earth": mass_earth,
                     "radius_km": radius_km, "gravity": gravity,
                     "period_hours": period_hours, "primary_mass_jupiter": primary_mass_jupiter,
                     "eccentricity_times": times.tolist(), "eccentricity_values": ecc.tolist()}
        data = deepcopy(self.data)
        if data["physical_signature"] is not None and signature != data["physical_signature"]:
            raise ValueError("Physical configuration changed within a thermal history")
        data["physical_signature"] = signature

        def temperature():
            return math.fsum([data["origin_temperature_k"],
                              data["energy_change_j"][0] / capacity,
                              data["energy_change_j"][1] / capacity])

        temp = temperature()
        if abs(temp - state.mantle_temperature_k) > 2 * math.ulp(temp):
            raise ValueError("Thermal state disagrees with saved energy inventory")
        if state.system_age_myr != data["origin_system_age_myr"] + state.time_myr - data["origin_time_myr"]:
            raise ValueError("Thermal system age disagrees with controller")

        def powers(time, temp):
            if not math.isfinite(temp) or temp < params.surface_temperature_k + 1.0:
                raise ValueError("Thermal law outside its domain: mantle must exceed surface by >= 1 K")
            age = data["origin_system_age_myr"] + time - data["origin_time_myr"]
            q = np.array([
                radiogenic_power_w(age, mass, params),
                tidal_heating_power_w(eccentricity.at(time), radius_km, period_hours,
                                      primary_mass_jupiter, params),
                convective_state(temp, radius_km, gravity, params)[3] * area,
            ])
            if not np.all(np.isfinite(q)) or np.any(q < 0):
                raise ValueError("Invalid thermal power")
            return q

        def net(q):
            return math.fsum([float(q[0]), float(q[1]), -float(q[2])])

        def heun(time, temp, h):
            q0 = powers(time, temp)
            predicted = temp + net(q0) * (h * SECONDS_PER_MYR / capacity)
            q1 = powers(time + h, predicted)
            exchanges = (0.5 * h * SECONDS_PER_MYR) * (q0 + q1)
            return temp + net(exchanges) / capacity, exchanges

        powers(state.time_myr, temp)  # validate input before trial rejection
        time = state.time_myr
        trials = 0
        while time < target:
            trials += 1
            if trials > self.options.max_trials:
                raise RuntimeError("Thermal adaptive trial budget exhausted; no state committed")
            future_knots = times[(times > time) & (times < target)]
            endpoint = float(future_knots[0]) if len(future_knots) else target
            h = min(data["next_step_myr"], self.options.max_step_myr, endpoint - time)
            if time + h <= time:
                raise RuntimeError("Thermal step underflow; no state committed")
            try:
                coarse_t, coarse_q = heun(time, temp, h)
                half_t, q1 = heun(time, temp, h / 2)
                fine_t, q2 = heun(time + h / 2, half_t, h / 2)
                q = q1 + q2
                t_scale = self.options.temperature_atol_k + self.options.temperature_rtol * abs(fine_t)
                q_scale = self.options.exchange_atol_j_per_myr * h + self.options.exchange_rtol * np.abs(q)
                # Fine Heun solution's estimated error = difference / (2^2 - 1).
                error = max(abs(fine_t - coarse_t) / (3 * t_scale),
                            float(np.max(np.abs(q - coarse_q) / (3 * q_scale))))
                powers(time + h, fine_t)
            except ValueError:
                error = math.inf
            if not math.isfinite(error) or error > 1:
                data["rejected_steps"] += 1
                factor = 0.5 if not math.isfinite(error) else max(0.1, 0.8 * error ** (-1 / 3))
                data["next_step_myr"] = h * factor
                continue
            _add(data["radiogenic_input_j"], float(q[0]))
            _add(data["orbital_input_j"], float(q[1]))
            _add(data["surface_output_j"], float(q[2]))
            _add(data["energy_change_j"], net(q))
            # h is exactly the remaining interval when endpoint-limited.
            time = endpoint if h == endpoint - time else time + h
            temp = temperature()
            data["accepted_steps"] += 1
            data["max_accepted_error_ratio"] = max(data["max_accepted_error_ratio"], error)
            data["next_step_myr"] = min(self.options.max_step_myr, h * min(2.0, 0.9 * max(error, 1e-12) ** (-1 / 3)))
        new = replace(state, time_myr=target, mantle_temperature_k=temp,
                      system_age_myr=data["origin_system_age_myr"] + target - data["origin_time_myr"])
        diag = diagnose_thermal_state(new, mass_earth, radius_km, gravity,
                                      period_hours, primary_mass_jupiter,
                                      eccentricity.at(target), params)
        new.tectonic_activity_factor = diag.tectonic_activity_factor
        new.thermal_lithosphere_thickness_km = diag.thermal_lithosphere_thickness_km
        data["time_myr"] = target
        self.data = data
        data["history"].append(self.diagnostics(new, capacity))
        return new, diag

    def diagnostics(self, state: ThermalState, capacity: float) -> dict:
        d = self.data
        residual = math.fsum([*d["energy_change_j"],
                             *(-x for x in d["radiogenic_input_j"]),
                             *(-x for x in d["orbital_input_j"]),
                             *d["surface_output_j"]])
        result = {"time_myr": state.time_myr, "temperature_k": state.mantle_temperature_k,
                  "accepted_steps": d["accepted_steps"], "rejected_steps": d["rejected_steps"],
                  "balance_residual_j": residual,
                  # This independently exposes precision lost by a rounded scalar T.
                  "rounded_temperature_energy_residual_j": math.fsum([
                      capacity * (state.mantle_temperature_k - d["origin_temperature_k"]),
                      *(-x for x in d["energy_change_j"])]),
                  "mantle_mass_is_fixed": True, "plate_work_is_accounted": False}
        for key in ("energy_change_j", "radiogenic_input_j", "orbital_input_j", "surface_output_j"):
            result[key] = math.fsum(d[key])
        return result
