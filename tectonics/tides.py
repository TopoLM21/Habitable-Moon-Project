"""Eccentricity-driven tidal deformation for the habitable-moon prototype.

The model intentionally captures only the leading-order *cyclic amplitude*
of the degree-2 tide caused by orbital eccentricity.  It is not a full Love-
number viscoelastic solver and does not include physical libration terms yet.

For a synchronously rotating satellite, the static mean tide does not by itself
cycle every orbit.  The eccentricity-driven variation is approximated as

    ΔU_amp ≈ 3 e U0 |P2(cos ψ)|
    U0 = G M_p R^2 / a^3
    strain_amp ≈ h2 ΔU_amp / (g R)

where ψ is angular distance from the sub-primary axis.  The result is used as a
slow geological weakening/fatigue index after averaging over the fast 47-hour
orbital cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np

Array = np.ndarray

G = 6.67430e-11
M_JUPITER = 1.89813e27


@dataclass(slots=True)
class EccentricityHistory:
    times_myr: Array
    eccentricity: Array

    def at(self, time_myr: float) -> float:
        if len(self.times_myr) == 1:
            return float(self.eccentricity[0])
        return float(np.interp(float(time_myr), self.times_myr, self.eccentricity))


def constant_eccentricity(value: float) -> EccentricityHistory:
    return EccentricityHistory(
        times_myr=np.asarray([0.0], dtype=np.float64),
        eccentricity=np.asarray([float(value)], dtype=np.float64),
    )


def load_eccentricity_csv(
    path: str | Path,
    time_column: str = "time_myr",
    eccentricity_column: str = "eccentricity",
) -> EccentricityHistory:
    times: list[float] = []
    ecc: list[float] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or time_column not in reader.fieldnames or eccentricity_column not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain columns {time_column!r} and {eccentricity_column!r}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            times.append(float(row[time_column]))
            ecc.append(float(row[eccentricity_column]))
    if not times:
        raise ValueError("Eccentricity CSV is empty")
    order = np.argsort(times)
    t = np.asarray(times, dtype=np.float64)[order]
    e = np.asarray(ecc, dtype=np.float64)[order]
    if np.any(e < 0.0) or np.any(e >= 1.0):
        raise ValueError("Eccentricity values must satisfy 0 <= e < 1")
    return EccentricityHistory(times_myr=t, eccentricity=e)


def eccentricity_history_from_config(config: dict, base_dir: str | Path = ".") -> EccentricityHistory:
    tide = config.get("tides", {})
    csv_path = tide.get("eccentricity_history_csv")
    if csv_path:
        path = Path(base_dir) / str(csv_path)
        return load_eccentricity_csv(
            path,
            time_column=str(tide.get("eccentricity_time_column", "time_myr")),
            eccentricity_column=str(tide.get("eccentricity_column", "eccentricity")),
        )
    return constant_eccentricity(float(tide.get("eccentricity_rms", 0.00047)))


def semi_major_axis_from_period(primary_mass_kg: float, period_hours: float) -> float:
    period_s = float(period_hours) * 3600.0
    return (G * float(primary_mass_kg) * (period_s / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)


def tidal_strain_amplitude(
    points: Array,
    eccentricity: float,
    radius_km: float,
    surface_gravity_m_s2: float,
    rotation_period_hours: float,
    primary_mass_jupiter: float = 1.0,
    love_h2: float = 0.6,
    sub_primary_axis: Array | None = None,
) -> Array:
    """Leading-order eccentricity tide strain amplitude at surface points."""
    points = np.asarray(points, dtype=np.float64)
    if sub_primary_axis is None:
        axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = np.asarray(sub_primary_axis, dtype=np.float64)
        axis /= np.linalg.norm(axis)

    mp = float(primary_mass_jupiter) * M_JUPITER
    radius_m = float(radius_km) * 1000.0
    a_m = semi_major_axis_from_period(mp, rotation_period_hours)
    cos_psi = np.clip(points @ axis, -1.0, 1.0)
    p2 = 0.5 * (3.0 * cos_psi**2 - 1.0)
    base = (
        3.0
        * float(eccentricity)
        * float(love_h2)
        * G
        * mp
        * radius_m
        / (float(surface_gravity_m_s2) * a_m**3)
    )
    return np.abs(base * p2)


def tidal_weakening_index(strain_amplitude: Array) -> Array:
    strain = np.asarray(strain_amplitude, dtype=np.float64)
    maximum = float(np.max(strain)) if len(strain) else 0.0
    if maximum <= 0.0:
        return np.zeros_like(strain)
    return np.clip(strain / maximum, 0.0, 1.0)


def radial_displacement_amplitude_m(strain_amplitude: Array, radius_km: float) -> Array:
    return np.asarray(strain_amplitude, dtype=np.float64) * float(radius_km) * 1000.0


__all__ = [
    "EccentricityHistory",
    "constant_eccentricity",
    "load_eccentricity_csv",
    "eccentricity_history_from_config",
    "semi_major_axis_from_period",
    "tidal_strain_amplitude",
    "tidal_weakening_index",
    "radial_displacement_amplitude_m",
]
