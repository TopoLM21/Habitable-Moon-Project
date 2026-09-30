"""Genesis-origin world evolution through the existing v0.31 runner.

Genesis remains the heat/orbit/water owner; the mature solver owns material,
plate velocities, transport and geological histories. This is an explicit
coarse coupling experiment, not certification of rigid plates or petrology.
Run in a dedicated CLI process: the legacy runner installs process-global hooks.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml

from .checkpoint import load_checkpoint, save_checkpoint
from .genesis import GenesisParameters
from .genesis_checkpoint_compat import require_thermal_model_version
from .genesis_mature import (build_experimental_genesis_mature_import,
                             save_experimental_genesis_mature_import)
from .genesis_shell import ShellParameters
from .genesis_long_term_support import begin_mechanical_transition, mechanical_sample, refresh_matched_mechanics
from .genesis_starter import StarterModel, StarterParameters
from .genesis_starter_fracture import YoungShellFracture
from .genesis_starter_material import (primary_material_inventory, independent_mantle_omega,
    independent_mantle_source_omega, mantle_source_parameters)
from .genesis_young_mechanics import (MECHANICS_MODEL_VERSION, mechanics_version,
    corrected_mechanics, sinking_mechanics, transmitted_mantle_flow, advance_prescribed_source)
from .genesis_starter_loading import thermal_budget_fields
from .genesis_tides import (TidalParameters, advance_tidal_orbit, mean_motion_rad_s)
from .kinematics import classify_boundaries
from .mesh import build_icosphere
from .simulation import load_config, PrototypeResult
from .subduction_memory import SubductionMemoryParameters
from .thermal import ThermalParameters, ThermalState, ThermalDiagnostics
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

NEW_MECHANICS_LIMITATIONS = [
    *LIMITATIONS[:4],
    "Basal torque balance uses prescribed traction and linear drag; convection itself is not solved.",
    "Ridge push uses a passive local cooling profile and a plate-mean flank contrast, not a resolved pressure field.",
    "Accepted slab material and finite-sheet warming are tracked; default slab force is disabled pending a validated transmission/resistance law.",
    "Full-transmission slab forcing is an explicit upper-bound experiment, not a calibrated physical prediction.",
    "Material transport is still raster based; accumulated signed closure does not yet create conservative fractional parcels.",
    "SI collision/transform contact reactions and continental GPE are not implemented in this young force mode.",
    "Live damage still uses the shared prescribed Starter loading; the separate free-edge membrane diagnostic is not a Maxwell evolution solver.",
    "Local cooling and slab warming are passive mechanical diagnostics; Genesis owns global heat without a second heat sink.",
    "Legacy CPU/GPU boundary-force kernels are bypassed by the NumPy SI solve; material kernels remain available.",
    LIMITATIONS[-1],
]


def mechanics_limitations(config):
    """Describe the saved force law without relabeling older model reports."""
    if not sinking_mechanics(config):
        return NEW_MECHANICS_LIMITATIONS if corrected_mechanics(config) else LIMITATIONS
    version = mechanics_version(config).split('-')[-1]
    mode = config.get("plate_dynamics", {}).get("young_slab_force_model")
    force = {
        "viscous_sinking_v1": "Connected accepted slabs exert thermal-buoyancy forces with viscous bending and mantle drag; shape and ambient flow remain prescribed reduced-order approximations.",
        "disabled_pending_closure": f"Slab forces are explicitly disabled in this {version} control configuration; accepted material and finite-sheet warming remain tracked.",
        "full_transmission_upper_bound": f"This {version} configuration explicitly selects full-transmission slab forcing as an upper-bound experiment, not the default viscous-sinking closure.",
    }.get(mode, f"Configured young slab force closure: {mode}.")
    layers = (["Thermal buoyancy follows retained acceptance-age layers, newest at the trench; each cohort remains uniform internally and the dip/shape is prescribed.",
               "Instantaneous neck severing still uses a surface-derived scalar tensile strength; a depth-resolved, conservative viscous neck and pressure-dependent yielding are not yet implemented."]
              if config.get("plate_dynamics", {}).get("young_slab_buoyancy_model") == "ordered_thermal_cohorts_v1" else [])
    return [*NEW_MECHANICS_LIMITATIONS[:6], force, *layers,
        "The attached branch forbids eduction; finite live tensile strength can detach a neck, but gradual neck deformation and return of buried material to the surface are not resolved.",
        "Only thermal mantle buoyancy is included; compositional crust buoyancy, free trench rollback, and dynamically evolved slab dip are not resolved.",
        "Slab connectivity follows local raster-edge transfers; frequent detachment or unresolved connections cannot establish sustained Earth-like subduction.",
        "Mechanical dissipation is diagnosed but is not fed back as heat to the authoritative global Genesis reservoirs.",
        *NEW_MECHANICS_LIMITATIONS[8:]]


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def load_starter_source(path):
    path = Path(path)
    metadata = json.loads((path.parent/"parameters.json").read_text(encoding="utf-8"))
    if metadata.get("format") != "genesis-starter-run-0.1":
        raise ValueError("Expected starter checkpoint with its parameters.json")
    require_thermal_model_version(metadata)
    shell = ShellParameters(**metadata["shell"])
    model = StarterModel(build_icosphere(shell.subdivisions), GenesisParameters(**metadata["thermal"]),
                         TidalParameters(**metadata["tides"]), shell, StarterParameters(**metadata["starter"]))
    return model, model.load_state(path), metadata


def _period(model, orbit):
    parameters = replace(model.tides, semimajor_axis_km=orbit.semimajor_axis_km,
                         eccentricity=orbit.eccentricity)
    return 2*math.pi/mean_motion_rad_s(parameters)/3600.


@dataclass(slots=True)
class GenesisThermalDiagnostics(ThermalDiagnostics):
    """Genesis budget in mature history, without changing ordinary mature output.

    The inherited thermal_lithosphere_thickness_km is a legacy mechanics input
    in this adapter. thermal_boundary_layer_thickness_km is the solid-convection
    D/Nu resistance scale; neither quantity is chemical crust thickness.
    """
    surface_temperature_k: float
    mantle_melt_fraction: float
    solid_convective_heat_flux_w_m2: float
    conductive_heat_flux_w_m2: float
    mantle_to_surface_flux_w_m2: float
    thermal_boundary_layer_thickness_km: float
    effective_heat_transfer_w_m2_k: float
    magma_transport_weight: float
    net_mantle_flux_w_m2: float
    mechanical_lithosphere_thickness_km: float


def project_thermal(model, state, parameters, reference_flux=None, mechanical_transition=None):
    sample = mechanical_sample(model.loading.sample(state.thermal_context), mechanical_transition)
    row = sample.thermal
    # These describe the same transport law that actually advanced enthalpy,
    # even if the supplied mature thermal configuration uses other constants.
    eta, ra, nu = (row[name] for name in ("viscosity_pa_s", "rayleigh_number", "nusselt_number"))
    flux = row["mantle_to_surface_flux_w_m2"]
    reference = max(float(flux if reference_flux is None else reference_flux), 1e-12)
    activity = float(np.clip(flux/reference, parameters.min_tectonic_activity_factor,
                            parameters.max_tectonic_activity_factor))
    time = state.time_myr
    age = time+model.thermal.system_age_at_start_myr
    # The mature runner consumes this legacy slot in force calculations. Keep
    # its existing mechanical-column meaning; never feed it back into heat loss.
    thermal = ThermalState(time, age, row["mantle_temperature_k"], reference, activity,
                           max(sample.lid_thickness_km, model.shell.min_load_bearing_thickness_km))
    area = model.thermal.area_m2
    qrad, qtide = row["radiogenic_flux_w_m2"], row["tidal_flux_w_m2"]
    diag = GenesisThermalDiagnostics(time, age, thermal.mantle_temperature_k, eta, ra, nu, flux,
        qrad, qtide, qrad+qtide-flux, qrad*area/1e12, qtide*area/1e12, flux*area/1e12,
        thermal.thermal_lithosphere_thickness_km, activity, sample.orbit.eccentricity,
        row["surface_temperature_k"], row["mantle_melt_fraction"],
        row["solid_convective_heat_flux_w_m2"], row["conductive_heat_flux_w_m2"], flux,
        row["thermal_boundary_layer_thickness_km"], row["effective_heat_transfer_w_m2_k"],
        row["magma_transport_weight"], row["net_mantle_flux_w_m2"],
        thermal.thermal_lithosphere_thickness_km)
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
    cfg["plate_dynamics"].setdefault("mantle_projection", "velocity_least_squares")
    cfg["plate_topology"]["split_differential_speed_deg_per_myr"] = 0.
    young_options = dict(cfg.get("young_shell", {}))
    cfg["young_shell"] = {"damage_law": "shared_starter", "fracture_memory_version": 1,
        "slab_activation": "convergence_integrated_length", "max_fractures_per_step": 1,
        "mechanics_model_version": MECHANICS_MODEL_VERSION, **young_options,
        "origin_time_myr": float(state.time_myr)}
    new_mechanics = corrected_mechanics(cfg)
    if new_mechanics:
        cfg["young_shell"].update(basal_source=mantle_source_parameters(model).metadata(),
            source_evolution="prescribed_constant", mechanical_activity="explicit_material_fields",
            slab_activation="accepted_material_v1", local_cooling="passive_material_age_v1")
        cfg["plate_dynamics"].update(force_model="young_si_v1",
            basal_drag_pa_s_m=model.parameters.basal_drag_pa_s_m,
            gravity_m_s2=p.surface_gravity_m_s2)
        sinking = sinking_mechanics(cfg)
        buoyancy_model = ("ordered_thermal_cohorts_v1" if mechanics_version(cfg) == MECHANICS_MODEL_VERSION
                          else "uniform_thermal_mass_v1")
        cfg["plate_dynamics"].setdefault("young_slab_force_model",
            "viscous_sinking_v1" if sinking else "disabled_pending_closure")
        cfg["plate_dynamics"].setdefault("young_velocity_response_model",
            "quasistatic" if sinking else "relaxed")
        if sinking:
            cfg["plate_dynamics"].setdefault("young_slab_buoyancy_model", buoyancy_model)
            from .mantle_convection import mantle_viscosity_pa_s
            cfg["plate_dynamics"].setdefault("young_slab_viscosity_contrast", 100.)
            cfg["plate_dynamics"].setdefault("young_slab_bend_radius_thickness_ratio", 3.)
            cfg["plate_dynamics"].setdefault("young_slab_mantle_shear_length_fraction", .5)
            cfg["plate_dynamics"].update(
                young_slab_mantle_viscosity_pa_s=mantle_viscosity_pa_s(material.mantle_temperature_k, p),
                young_slab_mantle_depth_km=p.radius_km*p.mantle_depth_fraction_radius)
        cfg["subduction_memory"]["model"] = "accepted_material_v1"
        cfg["subduction_memory"].update(
            young_slab_thermal_diffusivity_m2_s=p.thermal_diffusivity_m2_s,
            young_slab_mantle_depth_km=p.radius_km*p.mantle_depth_fraction_radius,
            young_slab_connectivity_model=("local_edge_transfer_v1" if sinking else "legacy_fixed_contacts"))
        if sinking:
            cfg["subduction_memory"].setdefault("young_slab_buoyancy_model",
                cfg["plate_dynamics"]["young_slab_buoyancy_model"])
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
        mantle_cell_omega_rad_per_myr=(independent_mantle_source_omega(model) if new_mechanics
                                     else independent_mantle_omega(model, state)),
        source_mass_kg=material.source_silicate_mass_kg, source_enthalpy_j=material.source_enthalpy_j,
        next_plume_birth_time_myr=state.time_myr+float(cfg.get("mantle_plumes", {}).get("mean_birth_interval_myr", 160.)),
        source_metadata={"kind": FORMAT, "physical_handoff_certified": False,
            "primary_material": "solid part of the existing global surface reservoir, assumed mafic",
            "chemical_age": "zero elapsed time since entry; solidification-age distribution unavailable",
            "young_thermal_owner": "Genesis; mature ThermalState is a projection only",
            "starter_fingerprint": model.fingerprint},
        topology_parameters=PlateTopologyParameters(**cfg["plate_topology"]))
    bundle.checkpoint.manager.last_split_time_myr = state.time_myr
    if new_mechanics:
        from .young_boundary import YoungBoundaryState
        bundle.checkpoint.subduction_memory.young_boundary_state = YoungBoundaryState(
            connectivity_model=cfg["subduction_memory"]["young_slab_connectivity_model"],
            buoyancy_geometry_model=cfg["subduction_memory"].get("young_slab_buoyancy_model", "uniform_thermal_mass_v1"))
    bundle.checkpoint.events += deepcopy(state.events)
    report = {"origin_time_myr": state.time_myr,
        "mechanics_model_version": mechanics_version(cfg),
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
        self.new_mechanics = corrected_mechanics(cfg)
        if self.new_mechanics:
            if cfg["young_shell"].get("basal_source") != mantle_source_parameters(model).metadata():
                raise ValueError("Young basal source provenance does not match the Starter")
            if cfg["plate_dynamics"].get("force_model") != "young_si_v1":
                raise ValueError("Versioned young mechanics requires its explicit SI force model")
            if (cfg["plate_dynamics"].get("young_slab_force_model") == "viscous_sinking_v1"
                    and not sinking_mechanics(cfg)):
                raise ValueError("Sinking mechanics requires version 0.4 or later; rebuild from Starter")
            inventory = initial_checkpoint.subduction_memory.young_boundary_state
            expected_connectivity = cfg["subduction_memory"].get(
                "young_slab_connectivity_model", "legacy_fixed_contacts")
            if (cfg["plate_dynamics"].get("young_slab_force_model") == "viscous_sinking_v1"
                    and expected_connectivity != "local_edge_transfer_v1"):
                raise ValueError("Sinking mechanics requires local_edge_transfer_v1 connectivity")
            if inventory is None or inventory.connectivity_model != expected_connectivity:
                raise ValueError("Saved slab connectivity differs from its configured model")
            buoyancy_model = cfg["plate_dynamics"].get("young_slab_buoyancy_model", "uniform_thermal_mass_v1")
            if buoyancy_model not in ("uniform_thermal_mass_v1", "ordered_thermal_cohorts_v1"):
                raise ValueError("Unknown young slab buoyancy geometry")
            if (buoyancy_model == "ordered_thermal_cohorts_v1"
                    and mechanics_version(cfg) != MECHANICS_MODEL_VERSION):
                raise ValueError("Ordered slab buoyancy requires version 0.5; rebuild from Starter")
            if (cfg["subduction_memory"].get("young_slab_buoyancy_model", "uniform_thermal_mass_v1") != buoyancy_model
                    or inventory.buoyancy_geometry_model != buoyancy_model):
                raise ValueError("Saved slab buoyancy geometry differs from its configured model")
            expected_source = independent_mantle_source_omega(model)
            if (initial_checkpoint.mantle_flow is None or not np.allclose(
                    initial_checkpoint.mantle_flow.cell_omega_rad_per_myr,
                    expected_source, rtol=5e-13, atol=1e-18)):
                raise ValueError("Saved prescribed source field differs from its provenance; rebuild from Starter")
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

    def dynamics_parameters(self, params):
        """Use the current thermal owner's rheology without editing its energy.

        The configured initial viscosity is checkpoint provenance; a continuing
        world must not freeze ambient mantle resistance at first partition.
        """
        if params is None or params.young_slab_force_model != "viscous_sinking_v1":
            return params
        from .mantle_convection import mantle_viscosity_pa_s
        sample = self.model.loading.sample(self.source_state.thermal_context)
        return replace(params,
            young_slab_mantle_viscosity_pa_s=mantle_viscosity_pa_s(
                sample.thermal["mantle_temperature_k"], self.model.thermal),
            young_slab_mantle_depth_km=(self.model.thermal.radius_km
                *self.model.thermal.mantle_depth_fraction_radius))

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
            if self.new_mechanics:
                from .lithosphere import oceanic_thermal_lithosphere_total_thickness_km
                ocean_total = oceanic_thermal_lithosphere_total_thickness_km(
                    np.minimum(state.crust_age_myr, age_cap),
                    thermal_diffusivity_m2_s=float(options.get("thermal_diffusivity_m2_s", 1e-6)),
                    cooling_coefficient=float(options.get("cooling_coefficient", 2.)),
                    max_total_thickness_km=float(options.get("oceanic_max_total_thickness_km", 155.)))
                fraction = (np.clip(state.continental_fraction, 0., 1.) if state.continental_fraction is not None
                    else (state.crust_type == 1).astype(float))
                # A chemically present newborn oceanic crust is not already
                # a cold load-bearing lid. Mixed/continental roots retain
                # their existing evolved H plus the chemical crust.
                total = np.where(fraction == 0., ocean_total,
                    np.maximum(state.mantle_lithosphere_thickness_km+crust, 0.))
                self.local_mechanical_diagnostics = {"model_version": "young-mechanics-0.3",
                    "local_total_lid_thickness_km": total,
                    "thermal_owner": "Genesis; matched mature material-age mechanics after column exhaustion"}
            return state
        if self.new_mechanics:
            from .genesis_local_mechanics import refresh_young_material_mechanics
            self.local_mechanical_diagnostics = refresh_young_material_mechanics(
                state, sample, self.model,
                origin_time_myr=self.cfg["young_shell"]["origin_time_myr"],
                thermal_diffusivity_m2_s=float(self.cfg.get("mechanical_lithosphere", {}).get(
                    "thermal_diffusivity_m2_s", 1e-6)))
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
            "mechanics_model_version": mechanics_version(self.cfg),
            "mechanical_activity_factor": 1. if self.new_mechanics else None,
            **thermal_budget_fields(sample.thermal),
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
        if self.new_mechanics:
            # Keep the v0.31 wrapper so plume diagnostics still observe source
            # updates. Only its source evolution is replaced in this process.
            if hasattr(runner, "_original_advance_mantle_flow"):
                runner._original_advance_mantle_flow = advance_prescribed_source
            else:
                base.advance_mantle_flow = advance_prescribed_source
        current = {"system": self.initial_checkpoint.system,
                   "transport": self.initial_checkpoint.transport_state,
                   "subduction_memory": self.initial_checkpoint.subduction_memory}
        if self.new_mechanics:
            from .subduction_memory import advance_subduction_memory
            old_subduction = getattr(base, "advance_subduction_memory", advance_subduction_memory)

            def subduction(*args, **kwargs):
                arguments = list(args)
                if sinking_mechanics(cfg):
                    incoming = arguments[1] if len(arguments) > 1 else kwargs["state"]
                    # Heat and live material strength already represent the
                    # accepted endpoint. Warm retained cohorts and evaluate
                    # their dip/depth at that same time; signed contact-area
                    # integration still receives the original dt unchanged.
                    endpoint = replace(incoming, time_myr=self.source_state.time_myr)
                    if len(arguments) > 1:
                        arguments[1] = endpoint
                    else:
                        kwargs["state"] = endpoint
                result = old_subduction(*arguments, **kwargs)
                current["subduction_memory"] = result[0]
                return result

            base.advance_subduction_memory = subduction
        old_manager = base.PlateTopologyManager
        old_remap = base.remap_transport_state

        class TrackingManager(old_manager):
            def update(manager, mesh, state, system, boundaries, radius_km, dt_myr):
                updated, diag, events = super().update(mesh, state, system, boundaries, radius_km, dt_myr)
                from .genesis_starter_topology import canonicalize_plate_seeds
                updated = canonicalize_plate_seeds(mesh, updated)
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
            if self.new_mechanics:
                arguments = list(args)
                incoming = arguments[1] if len(arguments) > 1 else kwargs["state"]
                # Heat precedes dynamics. Compute current mechanical fields on
                # an independent view without changing the material time step.
                age_step = self.source_state.time_myr - incoming.time_myr
                current_material = replace(incoming, time_myr=self.source_state.time_myr,
                    crust_age_myr=incoming.crust_age_myr + age_step)
                self.local_mechanical_diagnostics = None
                self.mechanical_fields(current_material)
                if len(arguments) > 1:
                    arguments[1] = current_material
                else:
                    kwargs["state"] = current_material
                h = (current_material.mantle_lithosphere_thickness_km
                     + np.maximum(current_material.crust_thickness_km, 0.))
                if self.local_mechanical_diagnostics is not None:
                    h = self.local_mechanical_diagnostics["local_total_lid_thickness_km"]
                kwargs["mantle_flow"] = transmitted_mantle_flow(model, kwargs["mantle_flow"], h)
                if len(arguments) > 8:
                    arguments[8] = self.dynamics_parameters(arguments[8])
                    force_parameters = arguments[8]
                elif "params" in kwargs:
                    kwargs["params"] = self.dynamics_parameters(kwargs["params"])
                    force_parameters = kwargs["params"]
                else:
                    force_parameters = None
                if (force_parameters is not None and
                        force_parameters.young_slab_force_model == "viscous_sinking_v1"):
                    # Neck failure uses the transported, evolved material
                    # strength already owned by the live young fracture law.
                    # A separate array keeps the pure force solve read-only.
                    kwargs["young_slab_strength_pa"] = self.fracture.memory.strength_pa.copy()
                    if kwargs.get("trace") is None:
                        kwargs["trace"] = {}
                result = old_dynamics(*arguments, **kwargs)
                if (force_parameters is not None and
                        force_parameters.young_slab_force_model == "viscous_sinking_v1"):
                    failures = kwargs["trace"].get("slab_neck_failures", ())
                    if failures:
                        from .young_boundary import commit_slab_neck_failures, synchronize_young_zones
                        memory = kwargs.get("subduction_memory")
                        commit_slab_neck_failures(memory, failures, current_material.time_myr)
                        sub_params = kwargs.get("subduction_memory_params")
                        if sub_params is None:
                            sub_params = SubductionMemoryParameters(**cfg["subduction_memory"])
                        synchronize_young_zones(memory, current_material, sub_params)
                current["system"] = result[0]
                return result
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
            if self.new_mechanics:
                from .young_boundary import accept_slab_material, synchronize_young_zones

                def accepted_material(events, new_state):
                    memory = current["subduction_memory"]
                    accept_slab_material(model.mesh, memory.young_boundary_state,
                                         events, new_state.time_myr)
                    synchronize_young_zones(memory, new_state,
                                           SubductionMemoryParameters(**cfg["subduction_memory"]))

                kwargs["young_subduction_sink"] = accepted_material
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
        mechanics_version(cfg)  # Reject unknown semantics; absent means legacy.
        # Older young checkpoints contain tangent local omega fields but did
        # not record a projection method. Upgrade their mapping explicitly in
        # the new output configuration, preserving every physical coefficient
        # and the source archive. An explicit saved legacy mode is respected.
        cfg["plate_dynamics"].setdefault("mantle_projection", "velocity_least_squares")
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
    slab_inventory = None
    if coupling.new_mechanics:
        from .young_boundary import accepted_slab_inventory_diagnostics
        slab_inventory = accepted_slab_inventory_diagnostics(final.subduction_memory)
        initial_slab = accepted_slab_inventory_diagnostics(initial_cp.subduction_memory)
        accepted_delta = (slab_inventory["cumulative_accepted_oceanic_volume_km3"]
                          - initial_slab["cumulative_accepted_oceanic_volume_km3"])
        partitioned = sum(slab_inventory[name] for name in (
            "attached_oceanic_volume_km3", "deep_oceanic_volume_km3",
            "unresolved_or_detached_oceanic_volume_km3"))
        checks["slab_material_inventory"] = (
            math.isclose(accepted_delta, ledger["sink_components_km3"]["oceanic_subduction"],
                         rel_tol=5e-12, abs_tol=1e-6)
            and math.isclose(partitioned, slab_inventory["cumulative_accepted_oceanic_volume_km3"],
                             rel_tol=5e-12, abs_tol=1e-6))
    from .transport import quaternion_angle_deg
    from .dynamics import angular_velocity_vectors
    omega = angular_velocity_vectors(final.system)
    speed = np.linalg.norm(np.cross(omega[final.state.cell_plate], model.mesh.centroids), axis=1)*model.thermal.radius_km
    report = {"format": FORMAT, "status": "completed" if all(checks.values()) else "validation_failed",
        "mechanics_model_version": mechanics_version(cfg),
        "physical_handoff_certified": False, "mature_engine_executed": True,
        "import": provenance, "duration_myr": duration_myr, "step_myr": step_myr,
        "final_time_myr": end, "clocks": clocks, "checks": checks, "material_ledger": ledger,
        "accepted_slab_inventory": slab_inventory,
        "young_slab_force_model": cfg["plate_dynamics"].get("young_slab_force_model"),
        "young_slab_buoyancy_model": cfg["plate_dynamics"].get("young_slab_buoyancy_model", "uniform_thermal_mass_v1"),
        "cumulative_mantle_material_transfer_kg": transfer,
        "remaining_mantle_material_mass_kg": provenance["initial_mantle_material_mass_kg"]+transfer,
        "history": coupling.history, "final_plate_count": len(final.system.plates),
        "young_fracture": coupling.fracture.diagnose(),
        "young_fracture_events": coupling.fracture.events,
        "transport_commits": final.transport_state.cumulative_commit_count,
        "maximum_residual_rotation_deg": max(quaternion_angle_deg(q) for q in final.transport_state.residual_quaternions),
        "final_mean_surface_speed_km_myr": float(model.areas@speed/model.areas.sum()),
        "final_max_surface_speed_km_myr": float(speed.max()),
        "ocean_fraction": sample.thermal["ocean_fraction"],
        "limitations": mechanics_limitations(cfg),
        "mechanical_transition": cfg.get("young_shell", {}).get("mechanical_transition"),
        "mesh_history": mesh_history,
        "execution": {"cpu_workers": cpu_workers, "render_workers": render_workers,
            "plate_force_backend": "numpy_si" if coupling.new_mechanics else "legacy_effective",
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
