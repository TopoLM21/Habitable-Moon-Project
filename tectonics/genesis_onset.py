"""Coupled orbit forcing, nonlocal weakening and first shell displacements.

The reference shell remains a small-strain Lagrangian membrane. Its solved
tangent displacements move material markers; neither markers nor connected
intact patches are a mobile-plate handoff. Water access is a constitutive
weakening proxy, not an unaccounted bound-water mass reservoir.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np

from .genesis import (GenesisParameters, GenesisState, SECONDS_PER_MYR, ENERGY_SCALE,
                      advance, diagnose, initial_state, temperatures)
from .genesis_checkpoint_compat import MODEL_VERSION, require_thermal_model_version
from .genesis_shell import (Membrane, ShellParameters, ShellState, advance_shell,
    diagnose_shell, initialize_shell, principal_tensile, shell_fields)
from .mesh import build_icosphere
from .genesis_tides import (TidalParameters, TidalOrbitState, initial_tidal_orbit,
    advance_tidal_orbit, tidal_strain_cycle, tidal_heat_flux_w_m2, tidal_diagnostics,
    validate_tidal_orbit)
from .genesis_onset_support import (NonlocalLoading, material_positions,
                                    face_velocities, regional_motion)

ONSET_VERSION = "genesis-onset-0.1"


@dataclass(frozen=True)
class OnsetParameters:
    regularization_km: float = 800.
    water_weakening: bool = True
    wet_strength_fraction: float = 0.4
    access_timescale_myr: float = 0.1
    loss_timescale_myr: float = 0.5
    intact_permeability_fraction: float = 0.05
    water_access_temperature_k: float = 650.
    persistence_time_myr: float = 0.05

    def validate(self):
        if not isinstance(self.water_weakening, bool):
            raise ValueError("water_weakening must be boolean")
        for f in fields(self):
            if f.name == "water_weakening":
                continue
            v = getattr(self, f.name)
            if isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"Invalid onset parameter {f.name}")
        if not 0 < self.wet_strength_fraction <= 1 or not 0 < self.intact_permeability_fraction <= 1:
            raise ValueError("Strength/permeability fractions must be in (0,1]")


def onset_parameters_from_config(config):
    section = dict(config.get("genesis_onset", {}))
    if section.pop("schema_version", 1) != 1:
        raise ValueError("Unsupported genesis_onset schema")
    try:
        p = OnsetParameters(**section)
    except TypeError as exc:
        raise ValueError("Unknown onset parameters") from exc
    p.validate()
    return p


@dataclass
class OnsetState:
    time_myr: float
    water_access: np.ndarray
    displacement_rad: np.ndarray
    material_centroids: np.ndarray
    velocity_km_myr: np.ndarray
    path_length_km: np.ndarray
    weak_duration_myr: np.ndarray
    tidal_stress_mpa: np.ndarray
    tidal_heat_received_j: float = 0.


def initialize_onset(mesh):
    n = mesh.cell_count
    return OnsetState(0., np.zeros(n), np.zeros((mesh.vertex_count, 2)),
                      mesh.centroids.copy(), np.zeros((n, 3)), np.zeros(n),
                      np.zeros(n), np.zeros(n))


def update_water_access(old, damage, retained, active, surface_k, ocean_fraction,
                        dt_myr, p: OnsetParameters):
    """Surface-connected water access, not mineral hydration or water uptake.

    Liquid water permits weakening; hot steam alone does not. Permeability is
    prescribed and enhanced by damage. New solid dilutes inherited alteration.
    No water is removed from or added to the global conserved inventory.
    """
    inherited = np.asarray(old)*retained
    available = p.water_weakening and surface_k < p.water_access_temperature_k and ocean_fraction > 1e-6
    if available:
        wetting = (p.intact_permeability_fraction+(1-p.intact_permeability_fraction)*damage**2) / p.access_timescale_myr
        # Availability saturates at a modest liquid inventory; it is not uptake.
        wetting *= min(1., ocean_fraction/0.01)
        updated = 1-(1-inherited)*np.exp(-wetting*dt_myr)
    else:
        updated = inherited*np.exp(-dt_myr/p.loss_timescale_myr)
    return np.where(active, np.clip(updated, 0, 1), 0.)


def advance_orbit_thermal(state, orbit, thermal, tides, target_myr, max_step_myr=.01):
    """Advance the shared heat/orbit clocks, including an early thermal event."""
    guessed_end = target_myr
    for _ in range(16):
        _, heating = advance_tidal_orbit(orbit, tides, guessed_end)
        step_thermal = replace(thermal, tidal_heat_flux_w_m2=heating)
        result, rows = advance(state, step_thermal, target_myr, max_step_myr)
        actual_end = result.time_myr
        if abs(actual_end-guessed_end) <= 2e-13*max(1., abs(actual_end)):
            break
        guessed_end = actual_end
    else:
        raise RuntimeError("Thermal/orbit event synchronisation needs a smaller shell step")
    next_orbit, exact_heating = advance_tidal_orbit(orbit, tides, actual_end)
    correction = ((exact_heating-heating)*(actual_end-orbit.time_myr)
                  *SECONDS_PER_MYR/ENERGY_SCALE)
    result.energy[0] += correction
    result.energy[2] += correction
    rows[-1] = diagnose(result, replace(thermal, tidal_heat_flux_w_m2=exact_heating))
    return result, next_orbit, exact_heating, rows


class OnsetModel:
    def __init__(self, shell_p, thermal, onset_p, tides_p):
        self.p, self.thermal, self.onset_p, self.tides_p = shell_p, thermal, onset_p, tides_p
        shell_p.validate(thermal); thermal.validate(); onset_p.validate(); tides_p.validate()
        if not math.isclose(tides_p.satellite_radius_km, thermal.radius_km, rel_tol=1e-12):
            raise ValueError("Tidal and thermal satellite radii must agree")
        if thermal.tidal_heat_flux_w_m2 != 0:
            raise ValueError("Onset computes tidal heating from orbit; prescribed thermal tidal flux must be zero")
        self.mesh = build_icosphere(shell_p.subdivisions)
        self.membrane = Membrane(self.mesh, shell_p.poisson_ratio)
        self.smoother = NonlocalLoading(self.mesh, thermal.radius_km, onset_p.regularization_km)
        # Linear eccentricity tide changes only in amplitude as the orbit damps.
        self.reference_tide_cycle = tidal_strain_cycle(self.mesh, tides_p)

    def initial(self):
        return (initialize_shell(self.mesh, self.p, self.thermal), initial_state(self.thermal),
                initialize_onset(self.mesh), initial_tidal_orbit(self.tides_p))

    def current_tides(self, orbit):
        return replace(self.tides_p, semimajor_axis_km=orbit.semimajor_axis_km,
                       eccentricity=orbit.eccentricity)

    def step(self, shell, thermal_state, onset, orbit, target_myr, max_step_myr=.01):
        dt = target_myr-shell.time_myr
        if (not math.isfinite(dt) or dt <= 0 or onset.time_myr != shell.time_myr
                or orbit.time_myr != shell.time_myr or thermal_state.time_myr != shell.time_myr):
            raise ValueError("Onset component clocks must agree and advance")
        before_tm, before_ts = temperatures(np.asarray(thermal_state.energy), self.thermal)
        global_state, new_orbit, heating, rows = advance_orbit_thermal(
            thermal_state, orbit, self.thermal, self.tides_p, target_myr, max_step_myr)
        step_thermal = replace(self.thermal, tidal_heat_flux_w_m2=heating)
        tm, ts = temperatures(np.asarray(global_state.energy), self.thermal)
        dt = global_state.time_myr-shell.time_myr
        liquid_fraction = diagnose(global_state, step_thermal)["ocean_fraction"]
        average_e = (orbit.eccentricity+new_orbit.eccentricity)/2
        average_a = (orbit.semimajor_axis_km+new_orbit.semimajor_axis_km)/2
        tide_scale = ((average_e/self.tides_p.eccentricity)*(self.tides_p.semimajor_axis_km/average_a)**3
                      if self.tides_p.enabled and self.tides_p.eccentricity > 0 else 0.)
        tide_cycle = self.reference_tide_cycle*tide_scale
        water = onset.water_access.copy()
        tidal_peak = np.zeros(self.mesh.cell_count)

        def damage_update(old_damage, secular_stress, mean_temp, depth, retained, step):
            nonlocal water, tidal_peak
            active = depth >= self.p.min_load_bearing_thickness_km
            water = update_water_access(onset.water_access, old_damage, retained, active, ts,
                                        liquid_fraction, step, self.onset_p)
            strength = self.p.tensile_strength_pa*(1-(1-self.onset_p.wet_strength_fraction)*water)
            degradation = self.p.residual_stiffness+(1-self.p.residual_stiffness)*(1-old_damage)**2
            # The orbital period is days. A prescribed elastic Love response
            # is sampled over its cycle; the daily deformation never accumulates
            # as secular material displacement or stress-free eigenstrain.
            tide_stress = np.einsum("ab,pfb->pfa", self.membrane.d, tide_cycle)
            tide_stress *= (self.p.young_modulus_pa*degradation)[None, :, None]
            tide_stress[:, ~active] = 0.
            tensile = np.stack([principal_tensile(secular_stress+t) for t in tide_stress])
            tidal_peak = np.max(np.stack([principal_tensile(t) for t in tide_stress]), axis=0)/1e6
            loading = np.mean(np.maximum(tensile/strength-1, 0)**2, axis=0)/self.p.damage_timescale_myr
            loading = self.smoother.apply(loading, active)
            hot = np.clip((mean_temp-700)/650, 0, 1)
            healing = 1/self.p.cold_healing_timescale_myr+hot**4/self.p.hot_healing_timescale_myr
            rate = loading+healing
            equilibrium = loading/rate
            return np.where(active, np.clip(equilibrium+(old_damage-equilibrium)*np.exp(-rate*step), 0, 1), 0.)

        result = advance_shell(shell, self.mesh, self.membrane, self.p, self.thermal,
            global_state.time_myr, before_ts, ts, before_tm, tm, damage_update=damage_update)
        displacement = self.membrane.last_displacement_rad.copy()
        positions = material_positions(self.mesh, self.membrane.vertex_basis, displacement)
        velocity = face_velocities(onset.material_centroids, positions, self.thermal.radius_km, dt)
        # Markers of still molten material carry no shell motion.
        active = result.lid_thickness_km >= self.p.min_load_bearing_thickness_km
        velocity[~active] = 0.
        increment = np.linalg.norm(velocity, axis=1)*dt
        weak_age = np.where(result.damage >= self.p.damage_threshold, onset.weak_duration_myr+dt, 0.)
        next_onset = OnsetState(result.time_myr, water, displacement, positions, velocity,
            onset.path_length_km+increment, weak_age, tidal_peak,
            onset.tidal_heat_received_j+heating*dt*SECONDS_PER_MYR*self.thermal.area_m2)
        return result, global_state, next_onset, new_orbit, rows

    def diagnostics(self, shell, thermal_state, onset, orbit):
        pnow = self.current_tides(orbit)
        thermal = replace(self.thermal, tidal_heat_flux_w_m2=tidal_heat_flux_w_m2(pnow))
        global_row = diagnose(thermal_state, thermal)
        tm, ts = temperatures(np.asarray(thermal_state.energy), self.thermal)
        shell_row = diagnose_shell(shell, self.mesh, self.p, self.thermal, ts, tm)
        area = self.mesh.areas_unit_sphere
        active = shell.lid_thickness_km >= self.p.min_load_bearing_thickness_km
        speed = np.linalg.norm(onset.velocity_km_myr, axis=1)*.1
        energy_scale = max(abs(orbit.dissipated_energy_j), 1.)
        row = {"time_myr": onset.time_myr,
            "mean_water_access": float(np.average(onset.water_access, weights=area)),
            "max_water_access": float(np.max(onset.water_access)),
            "mean_speed_cm_yr": float(np.average(speed, weights=area)),
            "max_speed_cm_yr": float(np.max(speed)),
            "max_displacement_km": float(np.max(onset.path_length_km)),
            "persistent_weak_area_fraction": float(np.average(onset.weak_duration_myr >= self.onset_p.persistence_time_myr, weights=area)),
            "max_tidal_stress_mpa": float(np.max(onset.tidal_stress_mpa)),
            "tidal_heat_received_j": onset.tidal_heat_received_j,
            "orbit_heat_transfer_relative_residual": float((onset.tidal_heat_received_j-orbit.dissipated_energy_j)/energy_scale),
            "eccentricity": orbit.eccentricity,
            "semimajor_axis_km": orbit.semimajor_axis_km,
            "tidal_heat_flux_w_m2": tidal_heat_flux_w_m2(pnow)}
        motion = regional_motion(self.mesh, onset.material_centroids, onset.velocity_km_myr,
                                 active & (shell.damage < self.p.damage_threshold))
        row.update({k: v for k, v in motion.items() if isinstance(v, (float, int))})
        return global_row, shell_row, row, tidal_diagnostics(pnow)

    def fields(self, shell, thermal_state, onset):
        tm, ts = temperatures(np.asarray(thermal_state.energy), self.thermal)
        result = shell_fields(shell, self.mesh, self.p, self.thermal, ts, tm)
        result.update(water_access=onset.water_access,
            speed_cm_yr=np.linalg.norm(onset.velocity_km_myr, axis=1)*.1,
            displacement_km=onset.path_length_km, tidal_stress_mpa=onset.tidal_stress_mpa,
            material_centroids=onset.material_centroids, velocity_km_myr=onset.velocity_km_myr)
        return result


def _parameter_block(model):
    return {"shell": asdict(model.p), "thermal": asdict(model.thermal),
            "onset": asdict(model.onset_p), "tides": asdict(model.tides_p)}


def save_onset_checkpoint(path, model, shell, thermal_state, onset, orbit, controls, provenance):
    arrays, scalars = {}, {}
    for prefix, state in (("shell", shell), ("onset", onset)):
        scalars[prefix] = {}
        for f in fields(state):
            value = getattr(state, f.name)
            if isinstance(value, np.ndarray):
                arrays[f"{prefix}__{f.name}"] = value
            else:
                scalars[prefix][f.name] = value
    parameters = _parameter_block(model)
    meta = {"format": ONSET_VERSION, "thermal_model_version": MODEL_VERSION, "parameters": parameters,
        "parameter_hash": hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest(),
        "states": scalars, "thermal_state": asdict(thermal_state), "orbit": asdict(orbit),
        "controls": controls, "provenance": provenance}
    target = Path(path)
    temporary = Path(str(target)+".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, metadata=np.array(json.dumps(meta, allow_nan=False)), **arrays)
    temporary.replace(target)


def load_onset_checkpoint(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            parameters = meta["parameters"]
            if meta["format"] != ONSET_VERSION or hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest() != meta["parameter_hash"]:
                raise ValueError("Onset checkpoint version/parameter hash mismatch")
            require_thermal_model_version(meta)
            model = OnsetModel(ShellParameters(**parameters["shell"]), GenesisParameters(**parameters["thermal"]),
                               OnsetParameters(**parameters["onset"]), TidalParameters(**parameters["tides"]))
            states = {}
            for prefix, cls in (("shell", ShellState), ("onset", OnsetState)):
                arrays = {k.split("__", 1)[1]: archive[k].copy() for k in archive.files if k.startswith(prefix+"__")}
                states[prefix] = cls(**meta["states"][prefix], **arrays)
        shell, onset = states["shell"], states["onset"]
        thermal = GenesisState(**meta["thermal_state"])
        orbit = TidalOrbitState(**meta["orbit"])
        _validate_loaded(model, shell, thermal, onset, orbit)
        return model, shell, thermal, onset, orbit, meta
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed onset checkpoint") from exc


def _validate_loaded(model, shell, thermal, onset, orbit):
    validate_tidal_orbit(orbit, model.tides_p)
    n, v = model.mesh.cell_count, model.mesh.vertex_count
    if not all(math.isfinite(t) and t >= 0 and t == shell.time_myr for t in
               (shell.time_myr, thermal.time_myr, onset.time_myr, orbit.time_myr)):
        raise ValueError("Onset checkpoint clocks differ")
    for state, shapes in ((shell, {"column_enthalpy": (n, model.p.column_layers), "lid_thickness_km": (n,),
            "eigenstrain": (n, 3), "strain": (n, 3), "stress_pa": (n, 3), "damage": (n,), "peak_tensile_pa": (n,)}),
        (onset, {"water_access": (n,), "displacement_rad": (v, 2), "material_centroids": (n, 3),
            "velocity_km_myr": (n, 3), "path_length_km": (n,), "weak_duration_myr": (n,), "tidal_stress_mpa": (n,)})):
        for name, shape in shapes.items():
            value = getattr(state, name)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid onset checkpoint field {name}")
    for fraction in (shell.damage, onset.water_access):
        if np.any((fraction < 0) | (fraction > 1)):
            raise ValueError("Invalid checkpoint fraction")
    if (np.any(shell.column_enthalpy <= 0) or np.any(shell.lid_thickness_km < 0)
            or np.any(shell.lid_thickness_km > model.p.column_depth_km)
            or any(np.any(a < 0) for a in (onset.path_length_km, onset.weak_duration_myr, onset.tidal_stress_mpa, shell.peak_tensile_pa))
            or not np.allclose(np.linalg.norm(onset.material_centroids, axis=1), 1, atol=1e-10)):
        raise ValueError("Nonphysical onset checkpoint")
    for state in (shell, onset, orbit):
        for f in fields(state):
            value = getattr(state, f.name)
            if isinstance(value, Real) and not math.isfinite(value):
                raise ValueError(f"Nonfinite onset checkpoint scalar {f.name}")
    if (shell.initial_column_energy_j <= 0 or shell.equilibrium_residual < 0
            or onset.tidal_heat_received_j < 0 or orbit.dissipated_energy_j < 0
            or np.any(onset.weak_duration_myr > onset.time_myr+1e-12)
            or (shell.first_fracture_time_myr is not None
                and not 0 <= shell.first_fracture_time_myr <= shell.time_myr)
            or any(not math.isfinite(t) or not 0 <= t <= thermal.time_myr for t in thermal.events.values())):
        raise ValueError("Invalid onset checkpoint history")
    expected_positions = material_positions(model.mesh, model.membrane.vertex_basis, onset.displacement_rad)
    if not np.allclose(expected_positions, onset.material_centroids, atol=1e-12, rtol=0):
        raise ValueError("Checkpoint material markers disagree with displacement")
    if len(thermal.energy) != 4 or not all(math.isfinite(e) and e >= 0 for e in thermal.energy) or not math.isfinite(thermal.initial_total_energy) or thermal.initial_total_energy <= 0:
        raise ValueError("Invalid thermal checkpoint")
    global_row, shell_row, onset_row, _ = model.diagnostics(shell, thermal, onset, orbit)
    if (abs(global_row["relative_energy_residual"]) > 1e-8
        or abs(shell_row["relative_column_energy_residual"]) > 1e-8
        or abs(onset_row["orbit_heat_transfer_relative_residual"]) > 1e-8):
        raise ValueError("Onset checkpoint energy ledger mismatch")
