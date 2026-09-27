"""Economical heat/orbit forcing for the coarse, initially molten starter.

The global heat and water inventory belongs to the existing genesis model.
A single passive conductive reference column supplies a *mechanical* lid
estimate from its first surface-connected solidus crossing. Its separate
boundary-flux ledger is never added to global mantle energy. This lid is not
chemical crust, and neither these thermal fields nor the smooth forcing below
construct plates, continents, volcanoes, or a mechanical equilibrium solution.

Large caller intervals contain inexpensive thermal samples; an orbital period
is represented by phase quadrature, never accumulated as secular strain.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math
from numbers import Integral, Real

import numpy as np

from .genesis import (GenesisParameters, GenesisState, M_EARTH,
                      SECONDS_PER_MYR, diagnose, initial_state)
from .genesis_material import face_frames
from .genesis_onset import advance_orbit_thermal
from .genesis_shell import (ShellParameters, lid_geometry, rock_enthalpy,
                           rock_temperature)
from .genesis_tides import (TidalOrbitState, TidalParameters,
                           initial_tidal_orbit, tidal_strain_cycle,
                           validate_tidal_orbit)
from .mesh import SphereMesh


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


@dataclass(frozen=True)
class StarterThermalState:
    thermal: GenesisState
    orbit: TidalOrbitState
    column_enthalpy: np.ndarray
    boundary_energy_j_m2: float
    initial_column_energy_j_m2: float
    last_tidal_heat_flux_w_m2: float = 0.


@dataclass(frozen=True)
class StarterThermalSample:
    thermal: dict
    orbit: TidalOrbitState
    lid_thickness_km: float
    mean_lid_temperature_k: float
    column_energy_residual_j_m2: float
    column_depth_limit_reached: bool
    state: StarterThermalState

    @property
    def time_myr(self):
        return float(self.thermal["time_myr"])


class StarterLoadingModel:
    """Shared heat/orbit clock and one reference conductive column.

    ``advance`` does not mutate its input. Returned samples exclude the initial
    state, include all thermal event points, and stop at the existing thermal
    model's freezing limit. ``max_sample_myr`` is a *forcing quadrature* bound,
    independent of a caller's coarse mechanics/output interval. An additional
    50 K boundary-temperature bound resolves rapid cooling automatically.

    The column has the supplied ShellParameters depth/layers and the existing
    explicit conservative enthalpy discretization. Its thermal stability
    substeps are cheap scalar-vector operations, with no shell FEM solve.
    """

    def __init__(self, thermal: GenesisParameters, tides: TidalParameters,
                 shell: ShellParameters):
        thermal.validate()
        tides.validate()
        shell.validate(thermal, max_subdivisions=8)
        if thermal.tidal_heat_flux_w_m2 != 0:
            raise ValueError("Starter computes tidal heat from orbit; prescribed thermal tide must be zero")
        if not (math.isclose(tides.satellite_radius_km, thermal.radius_km, rel_tol=1e-12)
                and math.isclose(tides.satellite_mass_kg, thermal.mass_earth*M_EARTH, rel_tol=1e-12)
                and math.isclose(tides.surface_gravity_m_s2, thermal.surface_gravity_m_s2, rel_tol=1e-12)):
            raise ValueError("Thermal and tidal satellite parameters must agree")
        self.thermal, self.tides, self.shell = thermal, tides, shell
        self.dz_m = shell.column_depth_km*1000/shell.column_layers
        self.layer_mass_kg_m2 = shell.density_kg_m3*self.dz_m

    def initial(self):
        thermal = initial_state(self.thermal)
        if min(self.thermal.initial_surface_temperature_k,
               self.thermal.initial_temperature_k) < self.thermal.liquidus_k:
            raise ValueError("Starter must begin with a fully molten surface and mantle")
        enthalpy = np.full(self.shell.column_layers,
                           float(rock_enthalpy(self.thermal.initial_temperature_k, self.thermal)))
        return StarterThermalState(thermal, initial_tidal_orbit(self.tides), enthalpy,
                                   0., float(enthalpy.sum()*self.layer_mass_kg_m2))

    def _validate(self, state):
        if not isinstance(state, StarterThermalState):
            raise ValueError("Expected a StarterThermalState")
        validate_tidal_orbit(state.orbit, self.tides)
        h = np.asarray(state.column_enthalpy)
        if (h.shape != (self.shell.column_layers,) or not np.isfinite(h).all()
                or np.any(h <= 0) or state.thermal.time_myr != state.orbit.time_myr
                or not np.isfinite(state.thermal.energy).all()
                or len(state.thermal.energy) != 4
                or not math.isfinite(state.boundary_energy_j_m2)
                or not math.isfinite(state.initial_column_energy_j_m2)
                or state.initial_column_energy_j_m2 <= 0
                or not math.isfinite(state.last_tidal_heat_flux_w_m2)
                or state.last_tidal_heat_flux_w_m2 < 0):
            raise ValueError("Starter thermal state has invalid fields or inconsistent clocks")

    def _sample(self, row, state):
        depth, mean = lid_geometry(rock_temperature(state.column_enthalpy, self.thermal)[None, :],
            row["surface_temperature_k"], row["mantle_temperature_k"], self.shell, self.thermal)
        residual = float(state.column_enthalpy.sum()*self.layer_mass_kg_m2
                         -state.initial_column_energy_j_m2-state.boundary_energy_j_m2)
        return StarterThermalSample(dict(row), state.orbit, float(depth[0]), float(mean[0]),
            residual, bool(depth[0] >= self.shell.column_depth_km*(1-1e-12)), deepcopy(state))

    def sample(self, state):
        self._validate(state)
        row = diagnose(state.thermal, replace(self.thermal,
                       tidal_heat_flux_w_m2=state.last_tidal_heat_flux_w_m2))
        return self._sample(row, state)

    def _conduct(self, enthalpy, boundary, before, after):
        dt = (after["time_myr"]-before["time_myr"])*SECONDS_PER_MYR
        stable = (.20*self.shell.density_kg_m3*self.thermal.silicate_heat_capacity_j_kg_k
                  *self.dz_m**2/self.shell.conductivity_w_m_k)
        count = max(1, math.ceil(dt/stable))
        if count > 100000:
            raise ValueError("Starter column interval exceeds supported thermal substeps")
        ds = dt/count
        h = enthalpy.copy()
        k = self.shell.conductivity_w_m_k
        ts0, tm0 = before["surface_temperature_k"], before["mantle_temperature_k"]
        dts = after["surface_temperature_k"]-ts0
        dtm = after["mantle_temperature_k"]-tm0
        for index in range(count):
            fraction = (index+.5)/count
            temp = rock_temperature(h, self.thermal)
            flux = np.empty(self.shell.column_layers+1)
            flux[0] = 2*k*(ts0+fraction*dts-temp[0])/self.dz_m
            flux[-1] = 2*k*(temp[-1]-tm0-fraction*dtm)/self.dz_m
            flux[1:-1] = k*(temp[:-1]-temp[1:])/self.dz_m
            h += (flux[:-1]-flux[1:])*ds/self.layer_mass_kg_m2
            boundary += float((flux[0]-flux[-1])*ds)
        return h, boundary

    def advance(self, state, target_myr, max_sample_myr=.05,
                max_temperature_change_k=50., max_thermal_step_myr=.01):
        self._validate(state)
        target = _positive(target_myr, "target_myr")
        maximum = _positive(max_sample_myr, "max_sample_myr")
        temperature_bound = _positive(max_temperature_change_k, "max_temperature_change_k")
        thermal_step = _positive(max_thermal_step_myr, "max_thermal_step_myr")
        if target <= state.thermal.time_myr or state.thermal.stopped_reason:
            raise ValueError("Target must follow a non-stopped starter thermal state")
        current = deepcopy(state)
        samples = []
        dt = min(maximum, target-current.thermal.time_myr)
        while current.thermal.time_myr < target and not current.thermal.stopped_reason:
            attempted = min(dt, target-current.thermal.time_myr)
            end = min(target, current.thermal.time_myr+attempted)
            if end <= current.thermal.time_myr:
                raise RuntimeError("Starter thermal sampling cannot advance floating-point time")
            following, orbit, heating, rows = advance_orbit_thermal(
                deepcopy(current.thermal), current.orbit, self.thermal, self.tides,
                end, thermal_step)
            before = self.sample(current).thermal
            previous = before
            largest_change = 0.
            for row in rows:
                largest_change = max(largest_change,
                    abs(row["surface_temperature_k"]-previous["surface_temperature_k"]),
                    abs(row["mantle_temperature_k"]-previous["mantle_temperature_k"]))
                previous = row
            if largest_change > temperature_bound:
                dt = attempted/2
                continue
            h = current.column_enthalpy.copy()
            boundary = current.boundary_energy_j_m2
            for row in rows:
                if row["time_myr"] <= before["time_myr"]:
                    continue
                # Diagnostic event rows do not contain full energy ledgers.
                # Re-evaluate only these rare interior endpoints so a caller
                # stopping at any yielded sample receives its exact heat/orbit
                # checkpoint, never the future state at the macrostep end.
                if row["time_myr"] == following.time_myr:
                    sample_thermal, sample_orbit, sample_heating = following, orbit, heating
                else:
                    sample_thermal, sample_orbit, sample_heating, event_rows = advance_orbit_thermal(
                        deepcopy(current.thermal), current.orbit, self.thermal, self.tides,
                        row["time_myr"], thermal_step)
                    row = event_rows[-1]
                h, boundary = self._conduct(h, boundary, before, row)
                snapshot = StarterThermalState(sample_thermal, sample_orbit, h.copy(), boundary,
                    current.initial_column_energy_j_m2, sample_heating)
                samples.append(self._sample(row, snapshot))
                before = row
            current = StarterThermalState(following, orbit, h, boundary,
                current.initial_column_energy_j_m2, heating)
            dt = min(maximum, attempted*2)
        return current, samples


def smooth_mantle_tensor(mesh: SphereMesh, seed: int):
    """Dimensionless symmetric tangential gradient of a smooth seeded field.

    A trace-free degree-two potential is differentiated on the unit sphere.
    The returned [xx, yy, xy] tensor uses the same face frames as the shell and
    tides, with a mesh-independent spectral bound. Multiplying by tau*L/H is
    a parameterized basal-loading stress estimate, NOT an equilibrium solve.
    No plate count, boundary mask, or mesh-scale random noise is introduced.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    matrix = rng.normal(size=(3, 3))
    matrix = (matrix+matrix.T)/2
    matrix -= np.eye(3)*np.trace(matrix)/3
    frames = face_frames(mesh)
    normal = np.cross(frames[:, :, 0], frames[:, :, 1])
    value = np.einsum("ni,ij,nj->n", normal, matrix, normal)
    projected = np.einsum("nia,ij,njb->nab", frames, matrix, frames)
    tensor = (projected-value[:, None, None]*np.eye(2))/(2*np.linalg.norm(matrix, 2))
    return np.column_stack((tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 0, 1]))


def tidal_stress_cycle(mesh: SphereMesh, orbit: TidalOrbitState,
                       parameters: TidalParameters, young_modulus_pa=6e10,
                       poisson_ratio=.25):
    """Undamaged plane-stress cycle [phase, face, xx/yy/xy], in pascals.

    Caller chooses the sample/midpoint orbit and applies material degradation.
    Cycle averaging and irreversible damage belong to the caller. A zero-mean
    elastic cycle is not itself a secular strain, drift, or fatigue model.
    """
    young = _positive(young_modulus_pa, "young_modulus_pa")
    if not isinstance(poisson_ratio, Real) or isinstance(poisson_ratio, bool) or not 0 < poisson_ratio < .49:
        raise ValueError("poisson_ratio must lie in (0, .49)")
    validate_tidal_orbit(orbit, parameters)
    nu = float(poisson_ratio)
    elasticity = np.array([[1., nu, 0.], [nu, 1., 0.], [0., 0., .5*(1-nu)]])/(1-nu**2)
    current = replace(parameters, eccentricity=orbit.eccentricity,
                      semimajor_axis_km=orbit.semimajor_axis_km)
    return young*np.einsum("ab,pfb->pfa", elasticity, tidal_strain_cycle(mesh, current))
