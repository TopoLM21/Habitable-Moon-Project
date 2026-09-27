"""Genesis-origin world evolution through the existing v0.31 runner.

Genesis remains the heat/orbit/water owner; the mature solver owns material,
plate velocities, transport and geological histories. This is an explicit
coarse coupling experiment, not certification of rigid plates or petrology.
Run in a dedicated CLI process: the legacy runner installs process-global hooks.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml

from .checkpoint import load_checkpoint, save_checkpoint
from .genesis import GenesisParameters
from .genesis_mature import (build_experimental_genesis_mature_import,
                             save_experimental_genesis_mature_import)
from .genesis_shell import ShellParameters
from .genesis_long_term_support import begin_mechanical_transition, mechanical_sample, refresh_matched_mechanics
from .genesis_starter import StarterModel, StarterParameters
from .genesis_starter_fracture import YoungShellFracture
from .genesis_starter_material import primary_material_inventory, independent_mantle_omega
from .genesis_tides import (TidalParameters, advance_tidal_orbit, mean_motion_rad_s)
from .kinematics import classify_boundaries
from .mesh import build_icosphere
from .simulation import load_config, PrototypeResult
from .subduction_memory import SubductionMemoryParameters
from .thermal import ThermalParameters, ThermalState, ThermalDiagnostics, convective_state
from .topology import PlateTopologyManager, PlateTopologyParameters

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "genesis-starter-continuation-0.2"
_runner_used = False
LIMITATIONS = [
    "Rigid-domain motion is an assumed coarse representation, not a verified fracture-to-rigid transition.",
    "Solid surface-reservoir material is assumed mafic; chemical differentiation is not predicted.",
    "Material bookkeeping uses equivalent-rock volume at the explicit surface reference density.",
    "Genesis heat reservoirs remain authoritative; spatial chemical/enthalpy transport feedback is not resolved.",
    "Slab pull grows with real convergence-integrated slab length; this is an explicit effective activation law.",
    "The shared starter damage law continues on transported material; the duplicate mature tidal damage source is disabled.",
    "Young fractures are evaluated only at mature step endpoints, at most once per step; transient rupture windows may be missed.",
    "Prescribed mantle loading is not recalculated from convection or mechanical stress redistribution after rupture.",
    "The exhausted solidus column joins an effective mature cooling law matched to primordial thickness; younger local crust can change mechanical thickness at that transition.",
    "Long-run execution is tested; geological realism, resolution/time convergence and sustained mobile tectonics are not established.",
]


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def load_starter_source(path):
    path = Path(path)
    metadata = json.loads((path.parent/"parameters.json").read_text(encoding="utf-8"))
    if metadata.get("format") != "genesis-starter-run-0.1":
        raise ValueError("Expected starter checkpoint with its parameters.json")
    shell = ShellParameters(**metadata["shell"])
    model = StarterModel(build_icosphere(shell.subdivisions), GenesisParameters(**metadata["thermal"]),
                         TidalParameters(**metadata["tides"]), shell, StarterParameters(**metadata["starter"]))
    return model, model.load_state(path), metadata


def _period(model, orbit):
    parameters = replace(model.tides, semimajor_axis_km=orbit.semimajor_axis_km,
                         eccentricity=orbit.eccentricity)
    return 2*math.pi/mean_motion_rad_s(parameters)/3600.


def project_thermal(model, state, parameters, reference_flux=None, mechanical_transition=None):
    sample = mechanical_sample(model.loading.sample(state.thermal_context), mechanical_transition)
    row = sample.thermal
    effective = replace(parameters, surface_temperature_k=row["surface_temperature_k"])
    eta, ra, nu, _, _ = convective_state(row["mantle_temperature_k"], model.thermal.radius_km,
                                        model.thermal.surface_gravity_m_s2, effective)
    flux = row["mantle_to_surface_flux_w_m2"]
    reference = max(float(flux if reference_flux is None else reference_flux), 1e-12)
    activity = float(np.clip(flux/reference, parameters.min_tectonic_activity_factor,
                            parameters.max_tectonic_activity_factor))
    time = state.time_myr
    age = time+model.thermal.system_age_at_start_myr
    thermal = ThermalState(time, age, row["mantle_temperature_k"], reference, activity,
                           max(sample.lid_thickness_km, model.shell.min_load_bearing_thickness_km))
    area = model.thermal.area_m2
    qrad, qtide = row["radiogenic_flux_w_m2"], row["tidal_flux_w_m2"]
    diag = ThermalDiagnostics(time, age, thermal.mantle_temperature_k, eta, ra, nu, flux,
        qrad, qtide, qrad+qtide-flux, qrad*area/1e12, qtide*area/1e12, flux*area/1e12,
        thermal.thermal_lithosphere_thickness_km, activity, sample.orbit.eccentricity)
    return thermal, diag


def build_starter_continuation(model, state, mature_config):
    """Import actual starter domains with an explicit primary-material closure."""
    model._validate(state)
    if state.stopped_reason != "first_partition" or len(state.system.plates) < 2:
        raise ValueError("Continuation requires a first-partition starter checkpoint")
    material = primary_material_inventory(model, state)
    if not material.fully_solid_surface:
        raise ValueError("The primary surface reservoir must be fully solid before mature transport")
    cfg = deepcopy(mature_config)
    p = model.thermal
    cfg["mesh"]["subdivisions"] = model.shell.subdivisions
    cfg["moon"].update(radius_km=p.radius_km, mass_earth=p.mass_earth,
        surface_gravity_m_s2=p.surface_gravity_m_s2,
        rotation_period_hours=_period(model, state.thermal_context.orbit))
    cfg["primary"]["mass_jupiter"] = model.tides.primary_mass_kg/1.89813e27
    cfg["plates"].update(count=len(state.system.plates), seed=model.parameters.seed)
    cfg["lithosphere"].update(initial_continental_fraction=0., continental_nuclei=0,
        oceanic_thickness_km=float(material.primary_thickness_km.mean()))
    cfg["thermal"].update(system_age_at_start_myr=p.system_age_at_start_myr,
        initial_mantle_temperature_k=material.mantle_temperature_k,
        surface_temperature_k=material.surface_temperature_k,
        mantle_mass_fraction=p.mantle_mass_fraction,
        mantle_heat_capacity_j_kg_k=p.silicate_heat_capacity_j_kg_k)
    cfg["tides"].update(eccentricity_history_csv=None,
        eccentricity_rms=model.tides.eccentricity, love_h2=model.tides.love_h2)
    # Keep the requested mature slab weight: a runtime length gate below gives
    # zero pull before a real slab exists. Young ridges have no finite-age floor.
    cfg["plate_dynamics"]["ridge_gpe_min_factor"] = 0.
    cfg["plate_topology"]["split_differential_speed_deg_per_myr"] = 0.
    cfg["young_shell"] = {"damage_law": "shared_starter", "fracture_memory_version": 1,
        "slab_activation": "convergence_integrated_length", "max_fractures_per_step": 1}
    cfg["hydrosphere"]["water_volume_km3"] = float(material.liquid_water_mass_kg.sum()/p.water_density_kg_m3/1e9)
    cfg.setdefault("mantle_plumes", {})["initial_plume_count"] = 0
    thermal, _ = project_thermal(model, state, ThermalParameters(**cfg["thermal"]))
    n = model.mesh.cell_count
    h = np.maximum(material.mechanical_lid_thickness_km-material.primary_thickness_km, 0.)
    sample = model.loading.sample(state.thermal_context)
    density = model.shell.density_kg_m3*3e-5*max(material.mantle_temperature_k-sample.mean_lid_temperature_k, 0.)
    bundle = build_experimental_genesis_mature_import(model.mesh, radius_km=p.radius_km,
        system=state.system, crust_age_myr=np.zeros(n), crust_thickness_km=material.primary_thickness_km,
        tidal_damage=state.damage, mantle_lithosphere_thickness_km=h,
        mantle_lithosphere_density_anomaly_kg_m3=np.full(n, density), thermal=thermal,
        elevation_m=np.zeros(n), water_volume_km3=cfg["hydrosphere"]["water_volume_km3"],
        mantle_cell_omega_rad_per_myr=independent_mantle_omega(model, state),
        source_mass_kg=material.source_silicate_mass_kg, source_enthalpy_j=material.source_enthalpy_j,
        next_plume_birth_time_myr=state.time_myr+float(cfg.get("mantle_plumes", {}).get("mean_birth_interval_myr", 160.)),
        source_metadata={"kind": FORMAT, "physical_handoff_certified": False,
            "primary_material": "solid part of the existing global surface reservoir, assumed mafic",
            "chemical_age": "zero elapsed time since entry; solidification-age distribution unavailable",
            "young_thermal_owner": "Genesis; mature ThermalState is a projection only",
            "starter_fingerprint": model.fingerprint},
        topology_parameters=PlateTopologyParameters(**cfg["plate_topology"]))
    bundle.checkpoint.manager.last_split_time_myr = state.time_myr
    bundle.checkpoint.events += deepcopy(state.events)
    report = {"origin_time_myr": state.time_myr,
        "primary_density_kg_m3": material.primary_density_kg_m3,
        "initial_mantle_material_mass_kg": float(material.mantle_mass_kg.sum()),
        "initial_primary_mass_kg": float(material.primary_mass_kg.sum()),
        "initial_primary_thickness_km": float(material.primary_thickness_km.mean()),
        "initial_mechanical_lid_thickness_km": material.mechanical_lid_thickness_km,
        "initial_liquid_water_volume_km3": cfg["hydrosphere"]["water_volume_km3"],
        "initial_starter_fit_residual": state.kinematic_fit_relative_residual,
        "material_closure": "existing surface reservoir assumed primary mafic, not mechanical lid",
        "initial_material_mass_relative_residual": material.silicate_mass_relative_residual,
        "initial_energy_partition_relative_residual": material.thermal_energy_relative_residual}
    return bundle, cfg, report


class YoungWorldCoupling:
    def __init__(self, model, source_state, cfg, initial_checkpoint, history=None, fracture=None):
        self.model, self.source_state, self.cfg = model, deepcopy(source_state), cfg
        self.before_orbit = self.source_state.thermal_context.orbit
        self.initial_checkpoint = initial_checkpoint
        self.history = list(history or [])
        self.reference_flux = initial_checkpoint.thermal.reference_convective_flux_w_m2
        self.thermal_parameters = ThermalParameters(**cfg["thermal"])
        self.fracture = fracture if fracture is not None else YoungShellFracture(model, source_state)
        if abs(self.fracture.time_myr-source_state.time_myr) > 1e-10:
            raise ValueError("Young fracture and thermal checkpoints disagree in time")
        if fracture is not None and not np.array_equal(fracture.damage, initial_checkpoint.state.tidal_damage):
            raise ValueError("Young fracture and mature material checkpoints disagree in damage")
        self.fracture.set_damage(initial_checkpoint.state.tidal_damage)
        self.previous_mechanical_sample = mechanical_sample(model.loading.sample(source_state.thermal_context),
            cfg.get("young_shell", {}).get("mechanical_transition"))

    def at(self, time_myr):
        # Mature tidal damage queries both endpoints and the midpoint.
        if not self.model.tides.enabled:
            return 0.
        if time_myr < self.before_orbit.time_myr-1e-10 or time_myr > self.source_state.time_myr+1e-10:
            raise ValueError("Mature tidal query outside accepted Genesis interval")
        if abs(time_myr-self.before_orbit.time_myr) <= 1e-12:
            return self.before_orbit.eccentricity
        return advance_tidal_orbit(self.before_orbit, self.model.tides, time_myr)[0].eccentricity

    def advance_heat(self, thermal, dt, *unused):
        if not math.isclose(thermal.time_myr, self.source_state.time_myr, rel_tol=0., abs_tol=1e-10):
            raise ValueError("Young and mature clocks diverged")
        proposed_before_orbit = self.source_state.thermal_context.orbit
        transition = deepcopy(self.cfg.get("young_shell", {}).get("mechanical_transition"))
        options = {"max_sample_myr": self.model.parameters.max_loading_interval_myr}
        if transition is not None:
            # The solid, slowly changing regime retains adaptive thermal/error
            # control without restarting the global ODE at every 50 kyr.
            options.update(max_sample_myr=1., max_thermal_step_myr=1.)
        context, samples = self.model.loading.advance(self.source_state.thermal_context,
            thermal.time_myr+dt, **options)
        if context.thermal.stopped_reason:
            raise ValueError(f"Global Genesis thermal model stopped: {context.thermal.stopped_reason}")
        if abs(context.thermal.time_myr-thermal.time_myr-dt) > 1e-10:
            raise ValueError("Young thermal support limit: thermal integration did not reach the requested time")
        if any(s.thermal["surface_melt_fraction"] > 1e-12 for s in samples):
            raise ValueError("Surface remelting is not supported by mature material transport")
        support_samples = []
        for sample in samples:
            if transition is None and sample.column_depth_limit_reached:
                transition = begin_mechanical_transition(sample, self.model,
                    self.cfg.get("mechanical_lithosphere", {}))
            support_samples.append(mechanical_sample(sample, transition))
        # Publish both clocks only after the entire proposed thermal interval
        # and its mechanical samples have succeeded. The source topology stays
        # an archive of the first partition; live fracture memory is separate.
        proposed_fracture = deepcopy(self.fracture)
        proposed_fracture.advance(mechanical_sample(
            self.model.loading.sample(self.source_state.thermal_context), transition), support_samples)
        self.fracture = proposed_fracture
        self.previous_mechanical_sample = mechanical_sample(
            self.model.loading.sample(self.source_state.thermal_context), transition)
        if transition is not None:
            self.cfg.setdefault("young_shell", {})["mechanical_transition"] = transition
        self.before_orbit = proposed_before_orbit
        self.source_state.thermal_context = context
        self.source_state.thermal_samples += len(samples)
        return project_thermal(self.model, self.source_state, self.thermal_parameters,
            self.reference_flux, transition)

    def mechanical_fields(self, state, *args, **kwargs):
        # Retain the depth-resolved young thermal owner instead of replacing it
        # with the mature half-space law based on reset chemical ages.
        transition = self.cfg.get("young_shell", {}).get("mechanical_transition")
        sample = mechanical_sample(self.model.loading.sample(self.source_state.thermal_context), transition)
        crust = np.maximum(state.crust_thickness_km, 0.)
        if transition is not None:
            # Match the primordial shell's accumulated thermal age at entry;
            # younger material still follows its own transported/reset age.
            age_cap = (transition["equivalent_cooling_age_myr"]
                       + max(0., state.time_myr-transition["time_myr"]))
            options = dict(kwargs)
            options["oceanic_crust_reference_km"] = float(self.cfg["lithosphere"]["oceanic_thickness_km"])
            options["mantle_density_kg_m3"] = self.model.shell.density_kg_m3
            options["mantle_temperature_contrast_k"] = max(
                sample.thermal["mantle_temperature_k"]-sample.thermal["surface_temperature_k"], 0.)
            options["oceanic_mean_temperature_deficit_fraction"] = 1.-transition["mean_temperature_fraction"]
            previous = self.previous_mechanical_sample.thermal
            dt = args[0] if args else options.pop("dt_myr", 0.)
            refresh_matched_mechanics(state, dt, age_cap,
                max(previous["mantle_temperature_k"]-previous["surface_temperature_k"], 0.), **options)
            return state
        state.mantle_lithosphere_thickness_km = np.maximum(sample.lid_thickness_km-crust, 0.)
        deficit = max(sample.thermal["mantle_temperature_k"]-sample.mean_lid_temperature_k, 0.)
        state.mantle_lithosphere_density_anomaly_kg_m3 = np.full(len(crust), self.model.shell.density_kg_m3*3e-5*deficit)

    def record(self, state, system, transport):
        from .dynamics import angular_velocity_vectors
        w = angular_velocity_vectors(system)
        radius = self.model.thermal.radius_km
        speed = np.linalg.norm(np.cross(w[state.cell_plate], self.model.mesh.centroids), axis=1)*radius
        sample = mechanical_sample(self.model.loading.sample(self.source_state.thermal_context),
            self.cfg.get("young_shell", {}).get("mechanical_transition"))
        self.fracture.set_damage(state.tidal_damage)
        row = {"time_myr": state.time_myr, "plate_count": len(system.plates),
            "mean_surface_speed_km_myr": float(self.model.areas@speed/self.model.areas.sum()),
            "max_surface_speed_km_myr": float(speed.max()),
            "transport_commits": transport.cumulative_commit_count,
            "surface_temperature_k": sample.thermal["surface_temperature_k"],
            "mantle_temperature_k": sample.thermal["mantle_temperature_k"],
            "ocean_fraction": sample.thermal["ocean_fraction"],
            "eccentricity": sample.orbit.eccentricity,
            "mechanical_lid_thickness_km": sample.lid_thickness_km,
            "thermal_energy_relative_residual": sample.thermal["relative_energy_residual"],
            "oceanic_volume_km3": float(state.oceanic_volume_km3.sum()),
            "continental_volume_km3": float(state.continental_volume_km3.sum()),
            "young_fracture": self.fracture.diagnose()}
        self.history.append(row)

    def install(self, runner):
        """Install only in this dedicated process; ordinary runner files stay intact."""
        base = runner.base
        cfg, model = self.cfg, self.model
        base.advance_thermal_state = self.advance_heat
        base.eccentricity_history_from_config = lambda *args, **kwargs: self
        # No discarded procedural plate map is generated, even by wrapper setup.
        base.build_prototype = lambda config: PrototypeResult(model.mesh,
            self.initial_checkpoint.system, [], 0., config)
        runner.v124._original_refresh_mechanical_lithosphere = self.mechanical_fields
        old_lith = runner.v124._original_advance_lithosphere
        old_hydro = base.advance_hydrosphere
        old_dynamics = base.update_plate_dynamics
        current = {"system": self.initial_checkpoint.system, "transport": self.initial_checkpoint.transport_state}
        old_manager = base.PlateTopologyManager
        old_remap = base.remap_transport_state

        class TrackingManager(old_manager):
            def update(manager, mesh, state, system, boundaries, radius_km, dt_myr):
                updated, diag, events = super().update(mesh, state, system, boundaries, radius_km, dt_myr)
                # Run after the normal connectivity repair, inside its event
                # transaction. The runner will remap transport and slab memory
                # once for the combined topology change.
                if (manager.params.split_enabled
                        and len(events) < manager.params.max_events_per_step
                        and not any(e.kind == "split" for e in events)):
                    trial, event = self.fracture.attempt(updated, time_myr=state.time_myr)
                    if trial is not None:
                        manager._remap_collision_memory(updated, trial)
                        updated = trial
                        state.cell_plate = trial.cell_plate.copy()
                        manager.last_split_time_myr = float(state.time_myr)
                        events = [*events, event]
                        counts = np.bincount(trial.cell_plate, minlength=len(trial.plates))
                        areas = np.bincount(trial.cell_plate, weights=model.areas, minlength=len(trial.plates))
                        diag = replace(diag, plate_count_after=len(trial.plates),
                            split_events=diag.split_events+1, topology_changed=True,
                            min_plate_cells=int(counts.min()), mean_plate_cells=float(counts.mean()),
                            max_plate_cells=int(counts.max()), min_plate_area_km2=float(areas.min()),
                            mean_plate_area_km2=float(areas.mean()), max_plate_area_km2=float(areas.max()))
                current["system"] = updated
                return updated, diag, events

        def remap_transport(*args, **kwargs):
            result = old_remap(*args, **kwargs)
            current["transport"] = result
            return result

        def dynamics(*args, **kwargs):
            from .genesis_starter_slab import young_slab_pull
            with young_slab_pull(SubductionMemoryParameters(**cfg["subduction_memory"])):
                result = old_dynamics(*args, **kwargs)
            current["system"] = result[0]
            return result

        def lithosphere(*args, **kwargs):
            arguments = list(args)
            if len(arguments) > 6:
                arguments[6] = _period(model, self.source_state.thermal_context.orbit)
            else:
                kwargs["rotation_period_hours"] = _period(model, self.source_state.thermal_context.orbit)
            incoming = arguments[2] if len(arguments) > 2 else kwargs["state"]
            if abs(incoming.time_myr-self.before_orbit.time_myr) > 1e-10:
                raise ValueError("Fracture loading must precede the matching material step")
            # Only replace damage on a shallow input copy; preserve the runner's
            # previous_state used for material transport and geological ledgers.
            working = replace(incoming, tidal_damage=self.fracture.damage.copy())
            if len(arguments) > 2:
                arguments[2] = working
            else:
                kwargs["state"] = working
            kwargs["tidal_damage_rate_per_myr"] = 0.
            kwargs["tidal_damage_relaxation_myr"] = math.inf
            result = old_lith(*arguments, **kwargs)
            self.fracture.transport(result[3].material_source_index)
            # The mature loading law is disabled, but real material replacement
            # (for example continental breakup) still resets inherited damage.
            self.fracture.set_damage(result[0].tidal_damage)
            result = (*result[:3], replace(result[3], mean_tidal_damage=float(self.fracture.damage.mean())))
            current["transport"] = kwargs["transport_state"]
            return result

        def hydrosphere(mesh, state, topo, hydro, *args, **kwargs):
            fraction = model.loading.sample(self.source_state.thermal_context).thermal["ocean_fraction"]
            hydro = replace(hydro, water_volume_km3=fraction*model.thermal.water_volume_km3)
            result = old_hydro(mesh, state, topo, hydro, *args, **kwargs)
            if not np.array_equal(state.cell_plate, current["system"].cell_plate):
                raise RuntimeError("Continuation diagnostics lost the post-topology system")
            self.record(state, current["system"], current["transport"])
            return result

        base.update_plate_dynamics = dynamics
        base.PlateTopologyManager = TrackingManager
        base.remap_transport_state = remap_transport
        runner.v124._original_advance_lithosphere = lithosphere
        base.advance_hydrosphere = hydrosphere


def _load_cp(path, cfg):
    return load_checkpoint(path, PlateTopologyManager(PlateTopologyParameters(**cfg["plate_topology"])))


def run_starter_continuation(source, output, duration_myr=10., step_myr=1., resume=False, mature_config=None,
        *, cpu_workers=1, render_workers=1, process_priority="normal", cell_kernels=False,
        numeric_kernels=True, single_source_cells=True, cell_workers=1, arc_kernels=True,
        assignment_columns=False, assignment_optimized=True, boundary_forces=False,
        frame_interval_myr=None, surface_only_frames=False, finalize=False, subdivisions=None):
    """Run one mature segment; duration is total elapsed time since first partition."""
    global _runner_used
    if _runner_used:
        raise RuntimeError("Use a new CLI process for each continuation segment")
    if any(isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in (duration_myr, step_myr)):
        raise ValueError("Continuation duration and step must be finite and positive")
    from .cpu_runtime import CpuExecution
    from visualization.render_runtime import RenderExecution
    from execution_policy import apply_process_priority
    execution = CpuExecution(cpu_workers, cell_kernels=cell_kernels, numeric_kernels=numeric_kernels,
        single_source_cells=single_source_cells, cell_workers=cell_workers, arc_kernels=arc_kernels,
        assignment_columns=assignment_columns, assignment_optimized=assignment_optimized,
        boundary_forces=boundary_forces)
    rendering = RenderExecution(render_workers, process_priority=process_priority)
    if frame_interval_myr is not None and (isinstance(frame_interval_myr, bool)
            or not math.isfinite(frame_interval_myr) or frame_interval_myr <= 0):
        raise ValueError("Frame interval must be finite and positive")
    root, source = Path(output).resolve(), Path(source).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("Continuation output must be new or empty")
    if resume:
        if mature_config is not None:
            raise ValueError("Resume retains its saved mature configuration")
        previous = json.loads((source/"continuation.json").read_text(encoding="utf-8"))
        if previous.get("format") != FORMAT:
            raise ValueError("Expected a continuation with ongoing fracture memory (0.2); older runs must restart from their original starter checkpoint")
        for relative, digest in previous["checkpoint_sha256"].items():
            if hashlib.sha256((source/relative).read_bytes()).hexdigest() != digest:
                raise ValueError("Continuation checkpoint/configuration integrity mismatch")
        model, state, metadata = load_starter_source(source/"young_context"/"starter_checkpoint.npz")
        fracture = YoungShellFracture.load(model, source/"young_context"/"fracture_memory.npz")
        cfg = load_config(source/"mature_config.yaml")
        cp = _load_cp(source/"mature_checkpoint", cfg)
        provenance = previous["import"]
        history = previous["history"]
        previous_transfer = previous["cumulative_mantle_material_transfer_kg"]
        mesh_history = list(previous.get("mesh_history", []))
    else:
        model, state, metadata = load_starter_source(source)
        bundle, cfg, provenance = build_starter_continuation(model, state,
            load_config(mature_config or ROOT/"configs"/"canonical_moon.yaml"))
        cp = bundle.checkpoint
        history, previous_transfer = [], 0.
        fracture = None
        mesh_history = []
    end = provenance["origin_time_myr"]+duration_myr
    span = end-state.time_myr
    if span <= 1e-12 or abs(span/step_myr-round(span/step_myr)) > 1e-8:
        raise ValueError("Remaining elapsed duration must be a positive integer number of mature steps")
    if abs(cp.state.time_myr-state.time_myr) > 1e-10:
        raise ValueError("Continuation mature and young checkpoints disagree in time")
    remeshed = False
    if subdivisions is not None and subdivisions != model.shell.subdivisions:
        from .genesis_continuation_remesh import remesh_continuation_state
        if fracture is None:
            fracture = YoungShellFracture(model, state)
        refined = remesh_continuation_state(model, state, fracture, cp, cfg, subdivisions)
        model, state, fracture = refined.model, refined.starter_state, refined.fracture
        cp, cfg = refined.checkpoint, refined.config
        metadata = {**metadata, **model.configuration}
        mesh_history.append(refined.report)
        remeshed = True
    root.mkdir(parents=True, exist_ok=True)
    young = root/"young_context"
    young.mkdir()
    _json(young/"parameters.json", metadata)
    model.save_state(young/"starter_checkpoint.npz", state)
    cfg["thermal_evolution"]["time_step_myr"] = step_myr
    (root/"mature_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if not resume:
        imported = save_experimental_genesis_mature_import(root/"initial_import", bundle)/"mature_checkpoint"
    if resume or remeshed:
        save_checkpoint(root/"initial_mature_checkpoint", cp)
        incoming = root/"initial_mature_checkpoint"
    else:
        incoming = imported
    # Every legacy module is imported once and executes in this CLI process.
    _runner_used = True
    import run_long_evolution_v131 as runner
    coupling = YoungWorldCoupling(model, state, cfg, cp, history, fracture)
    initial_cp = deepcopy(cp)
    if not history:
        coupling.record(cp.state, cp.system, cp.transport_state)
    coupling.install(runner)
    old_argv = sys.argv
    sys.argv = ["run_long_evolution_v131.py", "--config", str(root/"mature_config.yaml"),
        "--output", str(root/"mature_run"), "--resume", str(incoming),
        "--end-time", str(end), "--dt", str(step_myr), "--checkpoint", str(root/"mature_checkpoint"),
        "--save-frame", "--frame-origin", str(provenance["origin_time_myr"])]
    if frame_interval_myr is not None:
        sys.argv.extend(["--frame-interval", str(frame_interval_myr)])
    if surface_only_frames:
        sys.argv.append("--surface-only-frames")
    if finalize:
        sys.argv.append("--finalize")
    try:
        applied_priority = apply_process_priority(process_priority)
        print(f"Genesis execution: {cpu_workers} CPU worker(s), {render_workers} render worker(s), "
              f"priority={process_priority}", flush=True)
        with execution, rendering:
            rendering.install_runner_hooks()
            runner.main()
    finally:
        sys.argv = old_argv
    # The young mechanical model can finish its depth-resolved phase during
    # this segment; retain that continuous transition on the next resume.
    (root/"mature_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    final = _load_cp(root/"mature_checkpoint", cfg)
    model.save_state(young/"starter_checkpoint.npz", coupling.source_state)
    coupling.fracture.save(young/"fracture_memory.npz")
    from .genesis_starter_ledger import continuation_material_ledger
    ledger = continuation_material_ledger(initial_cp, final, provenance["primary_density_kg_m3"])
    transfer = previous_transfer+ledger["mantle_reference_mass_change_kg"]
    sample = model.loading.sample(coupling.source_state.thermal_context)
    clocks = {name: float(getattr(final, name).time_myr) for name in
              ("state", "thermal", "cycle", "topo", "hydrosphere", "mantle_flow")}
    checks = {"material_ledger": bool(ledger["balanced"]),
        "clocks_agree": all(abs(t-end) < 1e-9 for t in clocks.values()) and abs(coupling.source_state.time_myr-end) < 1e-9
            and abs(coupling.fracture.time_myr-end) < 1e-9,
        "fracture_damage_matches_material": bool(np.array_equal(coupling.fracture.damage, final.state.tidal_damage)),
        "thermal_energy": abs(sample.thermal["relative_energy_residual"]) < 1e-8,
        "liquid_water_matches_condensation": math.isclose(final.hydrosphere.water_volume_km3,
            sample.thermal["ocean_fraction"]*model.thermal.water_volume_km3, rel_tol=1e-12, abs_tol=1e-6),
        "positive_remaining_material": provenance["initial_mantle_material_mass_kg"]+transfer > 0.}
    from .transport import quaternion_angle_deg
    from .dynamics import angular_velocity_vectors
    omega = angular_velocity_vectors(final.system)
    speed = np.linalg.norm(np.cross(omega[final.state.cell_plate], model.mesh.centroids), axis=1)*model.thermal.radius_km
    report = {"format": FORMAT, "status": "completed" if all(checks.values()) else "validation_failed",
        "physical_handoff_certified": False, "mature_engine_executed": True,
        "import": provenance, "duration_myr": duration_myr, "step_myr": step_myr,
        "final_time_myr": end, "clocks": clocks, "checks": checks, "material_ledger": ledger,
        "cumulative_mantle_material_transfer_kg": transfer,
        "remaining_mantle_material_mass_kg": provenance["initial_mantle_material_mass_kg"]+transfer,
        "history": coupling.history, "final_plate_count": len(final.system.plates),
        "young_fracture": coupling.fracture.diagnose(),
        "young_fracture_events": coupling.fracture.events,
        "transport_commits": final.transport_state.cumulative_commit_count,
        "maximum_residual_rotation_deg": max(quaternion_angle_deg(q) for q in final.transport_state.residual_quaternions),
        "final_mean_surface_speed_km_myr": float(model.areas@speed/model.areas.sum()),
        "final_max_surface_speed_km_myr": float(speed.max()),
        "ocean_fraction": sample.thermal["ocean_fraction"], "limitations": LIMITATIONS,
        "mechanical_transition": cfg.get("young_shell", {}).get("mechanical_transition"),
        "mesh_history": mesh_history,
        "execution": {"cpu_workers": cpu_workers, "render_workers": render_workers,
            "process_priority": process_priority, "applied_priority": applied_priority,
            "cell_kernels": cell_kernels, "numeric_kernels": numeric_kernels,
            "single_source_cells": single_source_cells, "cell_workers": cell_workers,
            "arc_kernels": arc_kernels, "assignment_columns": assignment_columns,
            "assignment_optimized": assignment_optimized, "boundary_forces": boundary_forces,
            "frame_interval_myr": frame_interval_myr, "surface_only_frames": surface_only_frames,
            "finalize": finalize},
        "checkpoint_sha256": {}}
    _json(root/"render_timings.json", {**rendering.report(), "cpu_workers": cpu_workers,
                                      "numerical_execution": execution.numerical_report()})
    for relative in ("mature_checkpoint/meta.json", "mature_checkpoint/state.npz", "mature_config.yaml",
                     "young_context/starter_checkpoint.npz", "young_context/parameters.json", "young_context/fracture_memory.npz"):
        report["checkpoint_sha256"][relative] = hashlib.sha256((root/relative).read_bytes()).hexdigest()
    _json(root/"continuation.json", report)
    _save_plot(model, initial_cp, final, report, root/"continuation.png")
    if not all(checks.values()):
        raise RuntimeError(f"Continuation validation failed: {checks}")
    return report


def _save_plot(model, initial, final, report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .dynamics import angular_velocity_vectors
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    xyz = model.mesh.centroids
    lon, lat = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])), np.degrees(np.arcsin(xyz[:, 2]))
    ax = axes[0, 0]
    ax.scatter(lon, lat, c=final.state.cell_plate, s=7, cmap="tab20", vmin=0, vmax=max(1, len(final.system.plates)-1))
    ax.set(title="Области после продолжения зрелым движком", xlabel="Долгота", ylabel="Широта")
    w = angular_velocity_vectors(final.system)
    speed = np.linalg.norm(np.cross(w[final.state.cell_plate], xyz), axis=1)*model.thermal.radius_km
    damage = axes[0, 1].scatter(lon, lat, c=final.state.tidal_damage, s=7, cmap="magma", vmin=0., vmax=1.)
    axes[0, 1].set(title="Сохраняющееся повреждение оболочки", xlabel="Долгота", ylabel="Широта")
    fig.colorbar(damage, ax=axes[0, 1], label="Повреждение, 0–1")
    points = axes[0, 2].scatter(lon, lat, c=speed, s=7, cmap="viridis")
    axes[0, 2].set(title="Скорость поверхности", xlabel="Долгота", ylabel="Широта")
    fig.colorbar(points, ax=axes[0, 2], label="км / млн лет")
    rows = report["history"]
    t = [r["time_myr"] for r in rows]
    axes[1, 0].step(t, [r["plate_count"] for r in rows], where="post", marker=".", color="tab:purple")
    axes[1, 0].set(title="Рождение новых областей", xlabel="Возраст, млн лет", ylabel="Число областей")
    axes[1, 0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[1, 1].plot(t, [r["mean_surface_speed_km_myr"] for r in rows], marker=".")
    axes[1, 1].set(title="Движение по зрелой модели сил", xlabel="Возраст, млн лет", ylabel="Средняя скорость, км / млн лет")
    axes[1, 2].plot(t, [r["ocean_fraction"] for r in rows], marker=".", color="tab:blue")
    axes[1, 2].set(title="Конденсация океана", xlabel="Возраст, млн лет", ylabel="Доля воды в океане", ylim=(-.03, 1.03))
    fig.suptitle("Стартер → зрелая динамика: экспериментальное продолжение\nПервичное вещество принято мафическим; устойчивость плит ещё не доказана", fontsize=13)
    fig.savefig(path, dpi=140)
    plt.close(fig)
