"""Coarse, load-driven starter on the mature engine's unchanged sphere.

This is an explicit effective stress/damage model, not contact FEM. A smooth
prescribed mantle traction gradient, differential cooling and a sampled tidal
cycle load the first lid. Only already weakened cells can form a separating
band; the existing mature topology operator assigns the daughter domains.
No desired plate count, continental seeds, elevations or artificial velocity
kick are supplied. The run stops at its first partition: chemical crust,
magmatic transport and conservative mature handoff remain separate contracts.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import tempfile

import numpy as np

from .genesis import GenesisParameters, GenesisState, SECONDS_PER_MYR
from .genesis_checkpoint_compat import MODEL_VERSION, require_thermal_model_version
from .genesis_shell import (ShellParameters, mantle_traction, maxwell_factors,
                            smooth_anomaly)
from .genesis_onset_support import NonlocalLoading
from .genesis_starter_loading import (StarterLoadingModel, StarterThermalState,
                                      smooth_mantle_tensor, tidal_stress_cycle,
                                      thermal_budget_fields)
from .genesis_starter_topology import select_starter_cut, split_starter_band
from .genesis_tides import TidalOrbitState, TidalParameters, advance_tidal_orbit
from .mesh import connected_components
from .plates import Plate, PlateSystem
from .basal_coupling import basal_coupling_fraction


VERSION = "genesis-starter-0.1"


@dataclass(frozen=True)
class StarterParameters:
    seed: int = 20260927
    mantle_stress_length_km: float = 1000.
    strength_variation_fraction: float = .15
    cooling_contrast_fraction: float = .10
    rupture_damage: float = .65
    min_child_area_km2: float = 3.1e6
    min_band_span_km: float = 1000.
    max_loading_interval_myr: float = .05
    regularization_km: float = 300.
    water_weakening: bool = True
    tidal_mechanics: bool = True
    wet_strength_fraction: float = .4
    water_access_timescale_myr: float = .1
    hot_strength_fraction: float = .5
    damage_strength_reduction: float = .5
    shear_cohesion_pa: float = 4e6
    friction_dry: float = .6
    friction_wet: float = .2
    basal_drag_pa_s_m: float = 1e14

    def validate(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in {"water_weakening", "tidal_mechanics"}:
                if not isinstance(value, bool):
                    raise ValueError(f"{field.name} must be boolean")
            elif (isinstance(value, bool) or not isinstance(value, Real)
                  or not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid starter parameter {field.name}")
        if not isinstance(self.seed, Integral):
            raise ValueError("Starter seed must be an integer")
        for name in ("mantle_stress_length_km", "min_child_area_km2", "min_band_span_km",
                     "max_loading_interval_myr", "water_access_timescale_myr",
                     "shear_cohesion_pa", "basal_drag_pa_s_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not (0 < self.rupture_damage < 1 and 0 < self.wet_strength_fraction <= 1
                and 0 < self.hot_strength_fraction <= 1
                and 0 <= self.strength_variation_fraction < 1
                and 0 <= self.cooling_contrast_fraction < 1
                and 0 <= self.damage_strength_reduction < 1
                and self.friction_wet <= self.friction_dry):
            raise ValueError("Starter strength/contrast fractions are invalid")


@dataclass
class StarterState:
    thermal_context: StarterThermalState
    system: PlateSystem
    damage: np.ndarray
    cooling_stress_pa: np.ndarray
    water_access: np.ndarray
    yield_ratio: np.ndarray
    strength_pa: np.ndarray
    eligible: np.ndarray
    split_band: np.ndarray
    events: list[dict]
    first_fracture_time_myr: float | None = None
    thermal_samples: int = 0
    tidal_peak_mpa: float = 0.
    event_time_uncertainty_myr: float = 0.
    kinematic_fit_relative_residual: float | None = None
    stopped_reason: str | None = None

    @property
    def time_myr(self):
        return self.thermal_context.thermal.time_myr


def _principal(stress):
    mean = .5*(stress[..., 0]+stress[..., 1])
    shear = np.hypot(.5*(stress[..., 0]-stress[..., 1]), stress[..., 2])
    return np.maximum(mean+shear, 0.), shear, mean


class StarterModel:
    """A fast initial-condition experiment using shared thermal/orbital history.

    Mantle stress scale is tau * L/H * coupling(H); L is an explicit effective
    stress-transfer length. Local thermal stress is a Maxwell-filtered mismatch
    of cooling rates with zero area mean. Neither proxy solves force balance.
    Damage/healing rates reuse the existing shell time scales; phase-averaged
    *overstress* is integrated, not the average signed tidal stress. Subthreshold
    cycle-count fatigue and eruption physics are not represented.
    """

    def __init__(self, mesh, thermal: GenesisParameters, tides: TidalParameters,
                 shell: ShellParameters, parameters: StarterParameters | None = None):
        self.mesh, self.thermal, self.tides, self.shell = mesh, thermal, tides, shell
        self.parameters = parameters or StarterParameters()
        self.parameters.validate()
        self.loading = StarterLoadingModel(thermal, tides, shell)
        if mesh.cell_count != 20*4**shell.subdivisions:
            raise ValueError("Starter mesh and shell subdivisions must agree")
        self.areas = mesh.physical_cell_areas_km2(thermal.radius_km)
        self.anomaly = smooth_anomaly(mesh, self.parameters.seed)
        self.strength_factor = 1+self.parameters.strength_variation_fraction*smooth_anomaly(
            mesh, self.parameters.seed+1)
        self.mantle_tensor = smooth_mantle_tensor(mesh, self.parameters.seed)
        self.smoother = NonlocalLoading(mesh, thermal.radius_km, self.parameters.regularization_km)
        self.configuration = {"thermal_model_version": MODEL_VERSION,
                              "thermal": asdict(thermal), "tides": asdict(tides),
                              "shell": asdict(shell), "starter": asdict(self.parameters)}
        mesh_hash = hashlib.sha256(mesh.vertices.tobytes()+mesh.faces.tobytes()).hexdigest()
        self.fingerprint = hashlib.sha256(json.dumps({**self.configuration, "mesh": mesh_hash},
            sort_keys=True, allow_nan=False).encode()).hexdigest()

    def initial_state(self):
        n = self.mesh.cell_count
        system = PlateSystem(np.zeros(n, dtype=np.int32),
            (Plate(0, 0, np.array([0., 0., 1.]), 0.),))
        return StarterState(self.loading.initial(), system, np.zeros(n), np.zeros(n),
            np.zeros(n), np.zeros(n), self.shell.tensile_strength_pa*self.strength_factor,
            np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), [])

    def _validate(self, state):
        if not isinstance(state, StarterState):
            raise ValueError("Expected a StarterState")
        self.loading.sample(state.thermal_context)
        n = self.mesh.cell_count
        for name in ("damage", "cooling_stress_pa", "water_access", "yield_ratio", "strength_pa"):
            array = np.asarray(getattr(state, name))
            if array.shape != (n,) or not np.isfinite(array).all():
                raise ValueError(f"Invalid starter field {name}")
        if (np.any((state.damage < 0) | (state.damage > 1))
                or np.any((state.water_access < 0) | (state.water_access > 1))
                or np.any(state.yield_ratio < 0) or np.any(state.strength_pa <= 0)):
            raise ValueError("Starter material fields are outside physical ranges")
        for name in ("eligible", "split_band"):
            array = np.asarray(getattr(state, name))
            if array.shape != (n,) or array.dtype != bool:
                raise ValueError(f"Invalid starter mask {name}")
        if not np.array_equal(state.eligible, state.damage >= self.parameters.rupture_damage):
            raise ValueError("Starter eligibility must be derived from the saved damage")
        owner = np.asarray(state.system.cell_plate)
        count = len(state.system.plates)
        if (owner.shape != (n,) or owner.dtype.kind not in "iu" or count < 1
                or not np.array_equal(np.unique(owner), np.arange(count))):
            raise ValueError("Starter domain ownership is invalid")
        for pid, plate in enumerate(state.system.plates):
            cells = np.flatnonzero(owner == pid)
            axis = np.asarray(plate.euler_axis)
            if (plate.plate_id != pid or plate.seed_cell not in cells
                    or len(connected_components(cells, self.mesh.neighbors)) != 1
                    or axis.shape != (3,) or not np.isfinite(axis).all()
                    or not math.isclose(float(np.linalg.norm(axis)), 1., abs_tol=1e-12)
                    or not math.isfinite(plate.angular_speed_rad_per_myr)):
                raise ValueError("Starter domain must be connected with a finite Euler vector")
        if (not isinstance(state.thermal_samples, Integral) or state.thermal_samples < 0
                or not math.isfinite(state.tidal_peak_mpa) or state.tidal_peak_mpa < 0
                or not math.isfinite(state.event_time_uncertainty_myr)
                or state.event_time_uncertainty_myr < 0
                or state.stopped_reason not in (None, "first_partition", "thermal_stop", "column_depth_limit")):
            raise ValueError("Invalid starter progress metadata")
        if state.first_fracture_time_myr is not None and not (
                math.isfinite(state.first_fracture_time_myr)
                and 0 <= state.first_fracture_time_myr <= state.time_myr):
            raise ValueError("Invalid first fracture time")
        if state.kinematic_fit_relative_residual is not None and not (
                math.isfinite(state.kinematic_fit_relative_residual)
                and state.kinematic_fit_relative_residual >= 0):
            raise ValueError("Invalid kinematic fit residual")
        if (len(state.events) != count-1 or len(state.events) > 1
                or (count > 1) != (state.stopped_reason == "first_partition")
                or np.any(state.split_band & ~state.eligible)):
            raise ValueError("Starter partition history is inconsistent")
        for event in state.events:
            if (event.get("kind") != "split" or not 0 <= event.get("time_myr", -1) <= state.time_myr):
                raise ValueError("Starter event is outside its physical history")

    def _material_sample(self, state, before, after):
        p, shell = self.parameters, self.shell
        dt = after.time_myr-before.time_myr
        h = after.lid_thickness_km
        n = self.mesh.cell_count
        active = np.full(n, h >= shell.min_load_bearing_thickness_km)
        retention = min(1., before.lid_thickness_km/max(h, 1e-30))
        if not np.any(active):
            state.damage[:] = 0.
            state.cooling_stress_pa[:] = 0.
            state.water_access[:] = 0.
            state.yield_ratio[:] = 0.
            state.eligible[:] = False
            return
        temperature = .5*(before.mean_lid_temperature_k+after.mean_lid_temperature_k)
        eta = np.clip(shell.viscosity_reference_pa_s*np.exp(np.clip(
            shell.activation_energy_j_mol/8.314462618*(1/max(temperature, 1.)
                -1/shell.viscosity_reference_temperature_k), -60, 60)),
            shell.viscosity_min_pa_s, shell.viscosity_max_pa_s)
        r, b = maxwell_factors(dt*SECONDS_PER_MYR, np.asarray(eta/shell.young_modulus_pa))
        delta = after.mean_lid_temperature_k-before.mean_lid_temperature_k
        mismatch = -shell.young_modulus_pa/(1-shell.poisson_ratio)*shell.linear_expansion_per_k
        # New solid material is born unstressed. The mean-free anomaly prevents
        # homogeneous free cooling from being counted as a tensile load.
        state.cooling_stress_pa = retention*(float(r)*state.cooling_stress_pa
            + float(b)*mismatch*delta*p.cooling_contrast_fraction*self.anomaly)
        wet_available = (p.water_weakening and after.thermal["surface_temperature_k"] < 650
                         and after.thermal["ocean_fraction"] > 1e-6)
        inherited_water = state.water_access*retention
        if wet_available:
            exposure = min(1., after.thermal["ocean_fraction"]/.01)
            state.water_access = 1-(1-inherited_water)*math.exp(-dt*exposure/p.water_access_timescale_myr)
        else:
            state.water_access = inherited_water*math.exp(-dt/.5)
        hot = float(np.clip((temperature-700)/650, 0, 1))
        wet = 1-(1-p.wet_strength_fraction)*state.water_access
        inherited_damage = state.damage*retention
        weakening = (1-p.damage_strength_reduction*inherited_damage)
        warm_strength = 1-(1-p.hot_strength_fraction)*hot
        state.strength_pa = shell.tensile_strength_pa*self.strength_factor*wet*warm_strength*weakening
        coupling = basal_coupling_fraction(h, shell.traction_coupling_depth_km)
        scale = shell.convective_traction_pa*p.mantle_stress_length_km/h*coupling
        stress = self.mantle_tensor*scale
        stress[:, :2] += state.cooling_stress_pa[:, None]
        if p.tidal_mechanics:
            orbit, _ = advance_tidal_orbit(before.orbit, self.tides,
                .5*(before.time_myr+after.time_myr))
            tide = tidal_stress_cycle(self.mesh, orbit, self.tides,
                young_modulus_pa=shell.young_modulus_pa, poisson_ratio=shell.poisson_ratio)
        else:
            tide = np.zeros((1, n, 3))
        state.tidal_peak_mpa = float(np.max(_principal(tide)[0]))/1e6
        tensile, shear, mean = _principal(stress[None, :, :]+tide)
        friction = p.friction_dry+(p.friction_wet-p.friction_dry)*state.water_access
        # In-plane Mohr-circle proxy: this coarse model does not resolve fault
        # dip or pore-pressure/vertical overburden profiles.
        shear_strength = (p.shear_cohesion_pa*self.strength_factor*wet*warm_strength*weakening
                          + friction[None, :]*np.maximum(-mean, 0.))
        ratios = np.maximum(tensile/state.strength_pa[None, :], shear/shear_strength)
        state.yield_ratio = np.max(ratios, axis=0)
        drive = np.mean(np.maximum(ratios-1, 0.)**2, axis=0)/shell.damage_timescale_myr
        drive = self.smoother.apply(drive, active)
        healing = 1/shell.cold_healing_timescale_myr+hot**4/shell.hot_healing_timescale_myr
        rate = drive+healing
        equilibrium = drive/rate
        state.damage = np.clip(equilibrium+(inherited_damage-equilibrium)*np.exp(-rate*dt), 0., 1.)
        state.eligible = state.damage >= p.rupture_damage
        if np.any(state.eligible) and state.first_fracture_time_myr is None:
            state.first_fracture_time_myr = after.time_myr

    def _fit_domain_motion(self, system, lid_thickness_km):
        # This is a candidate Euler fit to an independent prescribed mantle
        # field. It is not a demonstration of dynamically sustained rigid motion.
        p = replace(self.shell, seed=self.parameters.seed)
        traction = mantle_traction(self.mesh, p, np.full(self.mesh.cell_count, lid_thickness_km))
        velocity = traction[self.mesh.faces].mean(axis=1)/self.parameters.basal_drag_pa_s_m
        positions = self.mesh.centroids
        velocity -= positions*np.sum(velocity*positions, axis=1)[:, None]
        angular = velocity*SECONDS_PER_MYR/(self.thermal.radius_km*1000.)
        omega, plates = [], []
        for pid, plate in enumerate(system.plates):
            mask = system.cell_plate == pid
            x, weight = positions[mask], self.areas[mask]
            moment = np.eye(3)*weight.sum()-np.einsum("n,ni,nj->ij", weight, x, x)
            torque = np.sum(weight[:, None]*np.cross(x, angular[mask]), axis=0)
            value = np.linalg.solve(moment, torque)
            omega.append(value)
            speed = float(np.linalg.norm(value))
            plates.append(Plate(pid, plate.seed_cell, value/speed if speed else np.array([0., 0., 1.]), speed))
        fitted = np.cross(np.asarray(omega)[system.cell_plate], positions)
        error = float(np.sum(self.areas[:, None]*(fitted-angular)**2))
        norm = float(np.sum(self.areas[:, None]*angular**2))
        return PlateSystem(system.cell_plate.copy(), tuple(plates)), math.sqrt(error/norm) if norm else 0.

    def advance(self, state, target_myr):
        self._validate(state)
        if (isinstance(target_myr, bool) or not math.isfinite(target_myr)
                or target_myr <= state.time_myr or state.stopped_reason):
            raise ValueError("Target must follow a non-stopped starter state")
        result = deepcopy(state)
        before = self.loading.sample(state.thermal_context)
        _, samples = self.loading.advance(state.thermal_context, target_myr,
            max_sample_myr=self.parameters.max_loading_interval_myr)
        for after in samples:
            self._material_sample(result, before, after)
            result.thermal_context = deepcopy(after.state)
            result.thermal_samples += 1
            if np.any(result.eligible):
                cut = select_starter_cut(self.mesh, result.system, result.eligible,
                    result.damage, self.thermal.radius_km,
                    self.parameters.min_child_area_km2, self.parameters.min_band_span_km)
                if cut is not None:
                    system, event = split_starter_band(self.mesh, result.system, cut,
                        self.thermal.radius_km, self.parameters.min_child_area_km2,
                        self.parameters.min_band_span_km, time_myr=after.time_myr)
                    if system is None:
                        raise RuntimeError("Selected eligible band failed the shared topology split")
                    result.system, result.kinematic_fit_relative_residual = self._fit_domain_motion(
                        system, after.lid_thickness_km)
                    result.split_band[cut] = True
                    result.events.append(asdict(event))
                    result.event_time_uncertainty_myr = after.time_myr-before.time_myr
                    result.stopped_reason = "first_partition"
                    break
            if after.state.thermal.stopped_reason:
                result.stopped_reason = "thermal_stop"
                break
            if after.column_depth_limit_reached:
                result.stopped_reason = "column_depth_limit"
                break
            before = after
        self._validate(result)
        return result

    def diagnose(self, state):
        self._validate(state)
        sample = self.loading.sample(state.thermal_context)
        thermal = sample.thermal
        area = float(self.areas.sum())
        has_lid = sample.lid_thickness_km >= self.shell.min_load_bearing_thickness_km
        return {"time_myr": state.time_myr,
            **thermal_budget_fields(thermal),
            "surface_temperature_k": thermal["surface_temperature_k"],
            "mantle_temperature_k": thermal["mantle_temperature_k"],
            "ocean_fraction": thermal["ocean_fraction"],
            "ocean_volume_km3": thermal["ocean_fraction"]*self.thermal.water_volume_km3,
            "lid_thickness_km": sample.lid_thickness_km,
            "domain_count": len(state.system.plates),
            "plate_count": len(state.system.plates) if has_lid else 0,
            "split_count": len(state.events),
            "damaged_area_fraction": float(self.areas@state.eligible/area),
            "max_damage": float(state.damage.max()),
            "max_yield_ratio": float(state.yield_ratio.max()),
            "mean_strength_mpa": float(self.areas@state.strength_pa/area/1e6),
            "water_access_fraction": float(self.areas@state.water_access/area),
            "tidal_peak_mpa": state.tidal_peak_mpa,
            "first_fracture_time_myr": state.first_fracture_time_myr,
            "thermal_samples": state.thermal_samples,
            "thermal_energy_relative_residual": thermal["relative_energy_residual"],
            "column_energy_relative_residual": sample.column_energy_residual_j_m2/state.thermal_context.initial_column_energy_j_m2,
            "eccentricity": sample.orbit.eccentricity,
            "event_time_uncertainty_myr": state.event_time_uncertainty_myr,
            "kinematic_fit_relative_residual": state.kinematic_fit_relative_residual,
            "stopped_reason": state.stopped_reason,
            "mature_handoff_ready": False,
            "continental_volume_km3": 0.,
            "initial_relief_m": 0.}

    def save_state(self, path, state):
        self._validate(state)
        context = state.thermal_context
        metadata = {"version": VERSION, "thermal_model_version": MODEL_VERSION, "fingerprint": self.fingerprint,
            "configuration": self.configuration,
            "thermal": asdict(context.thermal), "orbit": asdict(context.orbit),
            "boundary_energy_j_m2": context.boundary_energy_j_m2,
            "initial_column_energy_j_m2": context.initial_column_energy_j_m2,
            "last_tidal_heat_flux_w_m2": context.last_tidal_heat_flux_w_m2,
            "plates": [{"plate_id": p.plate_id, "seed_cell": p.seed_cell,
                "euler_axis": p.euler_axis.tolist(), "angular_speed_rad_per_myr": p.angular_speed_rad_per_myr}
                for p in state.system.plates],
            **{name: getattr(state, name) for name in ("events", "first_fracture_time_myr",
                "thermal_samples", "tidal_peak_mpa", "event_time_uncertainty_myr",
                "kinematic_fit_relative_residual", "stopped_reason")}}
        arrays = {name: getattr(state, name) for name in ("damage", "cooling_stress_pa", "water_access",
            "yield_ratio", "strength_pa", "eligible", "split_band")}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, metadata=np.array(json.dumps(metadata, allow_nan=False)),
                    cell_plate=state.system.cell_plate, column_enthalpy=context.column_enthalpy, **arrays)
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def load_state(self, path):
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            require_thermal_model_version(metadata)
            if metadata.get("version") != VERSION or metadata.get("fingerprint") != self.fingerprint:
                raise ValueError("Starter checkpoint belongs to another model/configuration")
            context = StarterThermalState(GenesisState(**metadata["thermal"]),
                TidalOrbitState(**metadata["orbit"]), saved["column_enthalpy"].copy(),
                metadata["boundary_energy_j_m2"], metadata["initial_column_energy_j_m2"],
                metadata["last_tidal_heat_flux_w_m2"])
            plates = tuple(Plate(p["plate_id"], p["seed_cell"], np.asarray(p["euler_axis"], dtype=float),
                p["angular_speed_rad_per_myr"]) for p in metadata["plates"])
            result = StarterState(context, PlateSystem(saved["cell_plate"].copy(), plates),
                **{name: saved[name].copy() for name in ("damage", "cooling_stress_pa", "water_access",
                    "yield_ratio", "strength_pa", "eligible", "split_band")},
                **{name: metadata[name] for name in ("events", "first_fracture_time_myr",
                    "thermal_samples", "tidal_peak_mpa", "event_time_uncertainty_myr",
                    "kinematic_fit_relative_residual", "stopped_reason")})
        self._validate(result)
        return result
