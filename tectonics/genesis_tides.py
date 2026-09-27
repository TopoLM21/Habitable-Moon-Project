"""Small-eccentricity, synchronous tides for the genesis experiment.

The Love response is prescribed, elastic, and periodic: it must not be added
to secular eigenstrain. Heat comes from a finite eccentricity-energy reservoir
with conserved orbital angular momentum. Primary spin, resonant pumping,
obliquity, frequency-dependent Love numbers and orbital migration driven by
the primary are outside this approximation.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from numbers import Integral, Real

import numpy as np
from scipy.integrate import solve_ivp

from .genesis import GenesisParameters, M_EARTH, SECONDS_PER_MYR
from .mesh import SphereMesh

G = 6.67430e-11
SYNCHRONOUS_SPIN = "synchronous_zero_obliquity"


@dataclass(frozen=True)
class TidalParameters:
    enabled: bool = False
    spin_state: str | None = None
    primary_mass_kg: float = 9.49065e27
    semimajor_axis_km: float = 773185.98
    eccentricity: float = 0.0002
    satellite_mass_kg: float = 0.5 * M_EARTH
    satellite_radius_km: float = 5287.0
    surface_gravity_m_s2: float = 7.12
    love_h2: float = 0.6
    love_l2: float = 0.08
    k2_over_q: float = 0.003
    phase_samples: int = 16

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("tides.enabled must be boolean")
        if self.spin_state not in (None, SYNCHRONOUS_SPIN):
            raise ValueError("Only synchronous_zero_obliquity tides are supported")
        if self.enabled and self.spin_state != SYNCHRONOUS_SPIN:
            raise ValueError("Enabled tides require explicit synchronous_zero_obliquity spin_state")
        zero_allowed = {"eccentricity", "love_h2", "love_l2", "k2_over_q"}
        for f in fields(self):
            if f.name in {"enabled", "spin_state"}:
                continue
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"tides.{f.name} must be finite")
            if value < 0 or (value == 0 and f.name not in zero_allowed):
                raise ValueError(f"tides.{f.name} is outside its positive range")
        if self.eccentricity > 0.05:
            raise ValueError("The first-order eccentricity tide requires e <= 0.05")
        if not isinstance(self.phase_samples, Integral) or not 4 <= self.phase_samples <= 512:
            raise ValueError("Tidal phase_samples must be an integer in 4..512")
        if self.semimajor_axis_km * (1-self.eccentricity) <= self.satellite_radius_km:
            raise ValueError("Satellite orbit must lie outside the satellite radius")


def tidal_parameters_from_config(config: dict, thermal: GenesisParameters | None = None) -> TidalParameters:
    section = dict(config.get("genesis_tides", {}))
    if section.pop("schema_version", 1) != 1:
        raise ValueError("Unsupported genesis_tides schema")
    section.pop("orbit_source", None)
    unknown = set(section) - {f.name for f in fields(TidalParameters)}
    if unknown:
        raise ValueError(f"Unknown tidal parameters: {sorted(unknown)}")
    # A single canonical satellite is shared by the thermal and orbital models.
    if thermal is not None:
        section.update(satellite_mass_kg=thermal.mass_earth*M_EARTH,
                       satellite_radius_km=thermal.radius_km,
                       surface_gravity_m_s2=thermal.surface_gravity_m_s2)
    p = TidalParameters(**section)
    p.validate()
    return p


def mean_motion_rad_s(p: TidalParameters) -> float:
    return math.sqrt(G*(p.primary_mass_kg+p.satellite_mass_kg)/(p.semimajor_axis_km*1000)**3)


def tidal_heat_power_w(p: TidalParameters) -> float:
    if not p.enabled or p.eccentricity == 0 or p.k2_over_q == 0:
        return 0.0
    return (10.5*p.k2_over_q*G*p.primary_mass_kg**2
            *(p.satellite_radius_km*1000)**5*mean_motion_rad_s(p)*p.eccentricity**2
            /(p.semimajor_axis_km*1000)**6)


def tidal_heat_flux_w_m2(p: TidalParameters) -> float:
    return tidal_heat_power_w(p)/(4*math.pi*(p.satellite_radius_km*1000)**2)


def tidal_strain_coefficients(mesh: SphereMesh, p: TidalParameters) -> tuple[np.ndarray, np.ndarray]:
    """Return cosine/sine coefficients [face, exx, eyy, engineering gamma].

    x points to the primary on the mean orbit and z is the orbit normal.
    U/(gR) = q e [3 cos(M) P2(x) + 6 sin(M) x y]. For a quadratic
    f=n.A.n, Hess_s(f)=2 T.A.T^T-2 f I. The prescribed strain is
    h2 f I + l2 Hess_s(f), collocated at each flat FEM face normal.
    Face bases exactly match genesis_shell.Membrane, including edge ordering.
    """
    p.validate()
    zeros = np.zeros((mesh.cell_count, 3))
    if not p.enabled or p.eccentricity == 0:
        return zeros, zeros.copy()
    xyz = mesh.vertices[mesh.faces]
    e1 = xyz[:, 1]-xyz[:, 0]
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    normal = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    e2 = np.cross(normal, e1)
    basis = np.stack((e1, e2), axis=1)
    q = (G*p.primary_mass_kg*(p.satellite_radius_km*1000)
         /(p.surface_gravity_m_s2*(p.semimajor_axis_km*1000)**3))
    matrices = (np.diag([3., -1.5, -1.5]),
                np.array([[0., 3., 0.], [3., 0., 0.], [0., 0., 0.]]))
    coefficients = []
    for matrix in matrices:
        f = np.einsum("fi,ij,fj->f", normal, matrix, normal)
        projected = np.einsum("fai,ij,fbj->fab", basis, matrix, basis)
        tensor = q*p.eccentricity*(2*p.love_l2*projected
                  +(p.love_h2-2*p.love_l2)*f[:, None, None]*np.eye(2))
        coefficients.append(np.column_stack((tensor[:, 0, 0], tensor[:, 1, 1], 2*tensor[:, 0, 1])))
    return coefficients[0], coefficients[1]


def tidal_strain_cycle(mesh: SphereMesh, p: TidalParameters) -> np.ndarray:
    cosine, sine = tidal_strain_coefficients(mesh, p)
    phase = 2*np.pi*np.arange(p.phase_samples)/p.phase_samples
    return np.cos(phase)[:, None, None]*cosine+np.sin(phase)[:, None, None]*sine


@dataclass(frozen=True)
class TidalOrbitState:
    time_myr: float
    semimajor_axis_km: float
    eccentricity: float
    dissipated_energy_j: float = 0.0


def initial_tidal_orbit(p: TidalParameters) -> TidalOrbitState:
    p.validate()
    return TidalOrbitState(0., p.semimajor_axis_km, p.eccentricity)


def _orbital_constants(p: TidalParameters) -> tuple[float, float]:
    amin_m = p.semimajor_axis_km*1000*(1-p.eccentricity**2)
    k = G*p.primary_mass_kg*p.satellite_mass_kg/2
    return amin_m, k


def validate_tidal_orbit(state: TidalOrbitState, p: TidalParameters) -> None:
    p.validate()
    for f in fields(state):
        value = getattr(state, f.name)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid tidal orbit {f.name}")
    if state.semimajor_axis_km <= 0 or state.eccentricity > p.eccentricity*(1+1e-12):
        raise ValueError("Tidal orbit cannot gain eccentricity in the isolated damping model")
    amin_m, k = _orbital_constants(p)
    if not math.isclose(state.semimajor_axis_km*1000*(1-state.eccentricity**2), amin_m, rel_tol=2e-12):
        raise ValueError("Tidal orbit angular momentum invariant is inconsistent")
    budget = k/amin_m*p.eccentricity**2
    expected = k/amin_m*(p.eccentricity**2-state.eccentricity**2)
    if budget == 0 and state.dissipated_energy_j != 0:
        raise ValueError("Circular orbit has no eccentricity energy to dissipate")
    if not math.isclose(state.dissipated_energy_j, expected, rel_tol=2e-9, abs_tol=max(1., budget*2e-12)):
        raise ValueError("Tidal orbit dissipated energy is inconsistent")
    if (not p.enabled or p.k2_over_q == 0) and (
            state.eccentricity != p.eccentricity or state.semimajor_axis_km != p.semimajor_axis_km
            or state.dissipated_energy_j != 0):
        raise ValueError("Disabled or nondissipative tides must retain the initial orbit")


def advance_tidal_orbit(state: TidalOrbitState, p: TidalParameters,
                       target_myr: float) -> tuple[TidalOrbitState, float]:
    """Advance isolated damping and return its energy-conserving mean heat flux.

    a(1-e^2)=amin, E=-K/a, and d(e^2)/dt=-P amin/K. Integration of
    log(e^2) preserves positivity even over many damping times. expm1 keeps
    the step energy accurate when the eccentricity changes very little.
    """
    validate_tidal_orbit(state, p)
    if (isinstance(target_myr, bool) or not isinstance(target_myr, Real)
            or not math.isfinite(target_myr) or target_myr < state.time_myr):
        raise ValueError("Tidal orbit target must be finite and not precede its state")
    dt = (target_myr-state.time_myr)*SECONDS_PER_MYR
    if dt == 0 or not p.enabled or p.k2_over_q == 0 or state.eccentricity**2 == 0:
        return replace(state, time_myr=float(target_myr)), 0.
    amin_m, k = _orbital_constants(p)
    current = replace(p, semimajor_axis_km=state.semimajor_axis_km, eccentricity=state.eccentricity)
    y0 = state.eccentricity**2
    decay0 = tidal_heat_power_w(current)*amin_m/(k*y0)
    # Integrate the change in log(y), rather than log(y) itself, for short steps.
    def derivative(_time, delta):
        y = y0*math.exp(min(float(delta[0]), 0.))
        return [-decay0*((1-y)/(1-y0))**7.5*dt]
    result = solve_ivp(derivative, (0., 1.), [0.], rtol=2e-11, atol=1e-14)
    if not result.success or not np.isfinite(result.y).all():
        raise RuntimeError("Tidal eccentricity integration failed")
    delta_log = min(float(result.y[0, -1]), 0.)
    eccentricity = state.eccentricity*math.exp(delta_log/2)
    y = eccentricity**2
    released_j = k/amin_m*y0*(-math.expm1(delta_log))
    next_state = TidalOrbitState(float(target_myr), amin_m/(1-y)/1000,
                                eccentricity, state.dissipated_energy_j+released_j)
    area = 4*math.pi*(p.satellite_radius_km*1000)**2
    return next_state, released_j/(dt*area)


def tidal_diagnostics(p: TidalParameters) -> dict:
    p.validate()
    return {"enabled": p.enabled, "spin_state": p.spin_state,
            "orbital_period_days": 2*math.pi/mean_motion_rad_s(p)/86400,
            "period_days": 2*math.pi/mean_motion_rad_s(p)/86400,
            "semimajor_axis_km": p.semimajor_axis_km, "eccentricity": p.eccentricity,
            "tidal_heat_power_w": tidal_heat_power_w(p),
            "tidal_heat_flux_w_m2": tidal_heat_flux_w_m2(p),
            "love_h2": p.love_h2, "love_l2": p.love_l2, "k2_over_q": p.k2_over_q}
