"""Experimental age-resolved surface/heat/basal coupling, with explicit ledgers.

Forces and advection use a first-order operator split. Positive time quadrature
resolves birth/removal cohorts inside each transport interval. Genesis owns
heat/orbit; event cooling is a passive evaluation of its thermal trajectory.
Removed surface material is archived, not converted into an invented slab force.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from bisect import bisect_right
import math

import numpy as np

from .fractional_surface import FractionalSurfaceState, SurfaceParcel, surface_totals
from .fractional_surface_io import load_fractional_checkpoint, save_fractional_checkpoint
from .fractional_transport import advance_fractional_transport, FractionalLoss
from .fractional_thermal import refresh_fractional_mechanics, refresh_parcel_mechanics
from .fractional_dynamics import solve_fractional_basal_dynamics
from .genesis import GenesisState
from .genesis_tides import TidalOrbitState
from .genesis_starter_loading import StarterThermalState
from .genesis_starter_material import mantle_source_parameters


LEGACY_VERSION = "fractional-heat-basal-coupling-1"
VERSION = "fractional-heat-basal-coupling-2"
DEFAULT_TIME_SCHEME = "gauss2_v2"
TIME_SCHEMES = (DEFAULT_TIME_SCHEME, "endpoint_v1")
EVENT_QUADRATURE = ((.5-math.sqrt(3.)/6., .5), (.5+math.sqrt(3.)/6., .5))
EXTENSIVE = ("area_km2", "oceanic_volume_km3", "cold_mantle_volume_km3", "density_excess_mass_kg")
LIMITATIONS = (
    "Basal-only oceanic experiment; not a full young-mechanics-0.5 continuation or a GUI model",
    "No ridge force, attached slab force, new fracture, topology change or mechanical work heating",
    "Removed parcels remain a passive unresolved subsurface archive; inherited slabs stay in the source archive",
    "First-order force/advection update; constant within-interval event rates with two positive time cohorts",
    "Diffuse spatial interfaces and growing exact material histories; no certified long-term convergence",
    "Genesis owns heat; local cooling and the mantle reference-material debit do not alter its reservoirs",
    "Only the unexhausted young conductive column is supported; exhaustion and surface remelting stop the step",
)


def time_scheme(provenance):
    scheme = provenance.get("time_scheme", "endpoint_v1" if provenance.get("coupling_version") == LEGACY_VERSION
                            else DEFAULT_TIME_SCHEME)
    if scheme not in TIME_SCHEMES:
        raise ValueError("Unsupported fractional event time scheme")
    return scheme


def coupling_version(provenance):
    return LEGACY_VERSION if time_scheme(provenance) == "endpoint_v1" else VERSION


@dataclass(frozen=True)
class CoupledFractionalState:
    surface: FractionalSurfaceState
    thermal_context: StarterThermalState
    provenance: dict
    plate_count: int
    initial_totals: dict
    cumulative_losses: dict
    cumulative_births: dict
    cumulative_thermal_sources: dict
    removed_material: tuple[FractionalLoss, ...]
    history: tuple[dict, ...]
    initial_diagnostics: dict
    thermal_sample_count: int = 0


def thermal_context_to_json(context):
    value = asdict(context)
    value["column_enthalpy"] = context.column_enthalpy.tolist()
    return value


def thermal_context_from_json(value):
    return StarterThermalState(GenesisState(**value["thermal"]), TidalOrbitState(**value["orbit"]),
        np.asarray(value["column_enthalpy"], dtype=float), value["boundary_energy_j_m2"],
        value["initial_column_energy_j_m2"], value["last_tidal_heat_flux_w_m2"])


def _check_sample(sample, expected_time):
    if not math.isclose(sample.time_myr, expected_time, rel_tol=0.,
                        abs_tol=64.*math.ulp(max(1., abs(expected_time)))):
        raise ValueError("Fractional material and Genesis clocks disagree")
    if sample.state.thermal.stopped_reason:
        raise ValueError("Genesis stopped before the fractional coupling interval completed")
    if sample.thermal["surface_melt_fraction"] > 1e-12:
        raise ValueError("Fractional coupling cannot continue after surface remelting")
    if sample.column_depth_limit_reached:
        raise ValueError("Fractional coupling does not support an exhausted young column")


def _refresh(mesh, surface, sample, model, provenance, plate_count):
    refreshed, cooling = refresh_fractional_mechanics(surface, sample, model,
        origin_time_myr=provenance["origin_time_myr"],
        thermal_diffusivity_m2_s=provenance["local_cooling_diffusivity_m2_s"])
    lid = cooling.pop("local_total_lid_thickness_km")
    dynamics = solve_fractional_basal_dynamics(mesh, refreshed, provenance["radius_km"],
        mantle_source_parameters(model), lid, plate_count=plate_count)
    return refreshed, cooling, dynamics


def dynamics_diagnostics(result):
    return dict(model=result.model, quadrature=result.quadrature, source_frame=result.source_frame,
        omega_rad_per_myr=result.omega_rad_per_myr.tolist(),
        plate_areas_km2=result.plate_areas_km2.tolist(),
        mean_speed_mm_per_year=result.mean_speed_mm_per_year,
        max_speed_mm_per_year=result.max_speed_mm_per_year,
        plate_mean_speed_mm_per_year=result.plate_mean_speed_mm_per_year.tolist(),
        basal_source_power_w=result.basal_source_power_w,
        basal_drag_dissipation_w=result.basal_drag_dissipation_w,
        torque_relative_residual=result.torque_relative_residual,
        power_relative_residual=result.power_relative_residual)


def heat_diagnostics(sample, sample_count):
    return dict(time_myr=sample.time_myr, sample_count=sample_count,
        mantle_temperature_k=sample.thermal["mantle_temperature_k"],
        surface_temperature_k=sample.thermal["surface_temperature_k"],
        reference_lid_thickness_km=sample.lid_thickness_km,
        relative_energy_residual=sample.thermal["relative_energy_residual"],
        column_energy_relative_residual=sample.column_energy_residual_j_m2 / sample.state.initial_column_energy_j_m2,
        additional_global_heat_sink_j=0., thermal_owner="Genesis")


def _totals(parcels):
    values = tuple(parcels)
    return {key: math.fsum(getattr(p, key) for p in values) for key in EXTENSIVE}


def ledger_diagnostics(state):
    remaining = surface_totals(state.surface)
    residuals = {key: math.fsum((remaining[key], state.cumulative_losses[key],
        -state.cumulative_births[key], -state.cumulative_thermal_sources[key], -state.initial_totals[key]))
        / max(abs(state.initial_totals[key]), abs(state.cumulative_thermal_sources[key]),
              abs(state.cumulative_births[key]), abs(state.cumulative_losses[key]), 1.) for key in EXTENSIVE}
    if any(not math.isfinite(v) or abs(v) > 5e-12 for v in residuals.values()):
        raise ValueError("Fractional coupling cumulative material/thermal ledger does not close")
    density = float(state.provenance["reference_density_kg_m3"])
    initial_mantle = float(state.provenance["initial_reference_mantle_mass_kg"])
    if not all(math.isfinite(v) and v > 0 for v in (density, initial_mantle)):
        raise ValueError("Coupling requires positive finite reference density and initial mantle mass")
    # Losses are archived, not immediately mixed into mantle. Births debit a
    # reference material account. Neither account is a new thermal reservoir.
    mantle_debit = -state.cumulative_births["oceanic_volume_km3"]*density*1e9
    remaining_mantle = initial_mantle+mantle_debit
    if not math.isfinite(remaining_mantle) or remaining_mantle <= 0.:
        raise ValueError("Fractional newborn material exhausts the reference mantle reservoir")
    archived_mass = state.cumulative_losses["oceanic_volume_km3"]*density*1e9
    change = (remaining["oceanic_volume_km3"]-state.initial_totals["oceanic_volume_km3"])*density*1e9
    chemical_residual = math.fsum((change, archived_mass, mantle_debit)) / max(
        state.initial_totals["oceanic_volume_km3"]*density*1e9, 1.)
    return dict(retained=remaining, relative_residuals=residuals,
        cumulative_losses=dict(state.cumulative_losses), cumulative_births=dict(state.cumulative_births),
        cumulative_thermal_sources=dict(state.cumulative_thermal_sources),
        reference_mantle_mass_change_kg=mantle_debit,
        initial_reference_mantle_mass_kg=initial_mantle,
        remaining_reference_mantle_mass_kg=remaining_mantle,
        archived_removed_basalt_reference_mass_kg=archived_mass,
        chemical_reference_mass_relative_residual=chemical_residual,
        reference_density_kg_m3=density,
        density_convention="Equivalent rock at one reference density; not a species mass or thermal reservoir")


def initialize_coupling(mesh, surface, model, thermal_context, provenance, plate_count):
    time_scheme(provenance)
    context = deepcopy(thermal_context)
    sample = model.loading.sample(context)
    _check_sample(sample, surface.time_myr)
    prepared, cooling, dynamics = _refresh(mesh, surface, sample, model, provenance, plate_count)
    zero = dict.fromkeys(EXTENSIVE, 0.)
    thermal_sources = dict(zero, **cooling["thermal_source_delta"])
    state = CoupledFractionalState(prepared, context, deepcopy(provenance), plate_count,
        surface_totals(surface), dict(zero), dict(zero), thermal_sources, (), (),
        dict(cooling=cooling, dynamics=dynamics_diagnostics(dynamics), heat=heat_diagnostics(sample, 0)))
    ledger_diagnostics(state)
    return state


def refresh_removed_material(losses, before, accepted_samples, model, provenance):
    """Cool each removed cohort at its event time; return signed source terms.

    The accepted Genesis endpoint is never rewritten. Event states use its
    latest preceding accepted thermal sample, then advance a copy over only
    the unresolved remainder. No interpolation of nonlinear enthalpy/lid
    properties and no second withdrawal from the global energy reservoirs.
    Previously archived material is deliberately not evolved by this helper.
    """
    samples = (before,)+tuple(accepted_samples)
    times = [sample.time_myr for sample in samples]
    if any(a > b for a, b in zip(times, times[1:])):
        raise ValueError("Accepted thermal samples must be ordered")
    groups = {}
    for index, record in enumerate(losses):
        if not times[0] <= record.time_myr <= times[-1]:
            raise ValueError("Material removal time is outside the accepted thermal interval")
        groups.setdefault(record.time_myr, []).append(index)
    refreshed = list(losses)
    source_parts = {key: [] for key in ("cold_mantle_volume_km3", "density_excess_mass_kg")}
    passive_samples = 0
    for event_time, indices in sorted(groups.items()):
        sample = samples[bisect_right(times, event_time)-1]
        if sample.time_myr != event_time:
            context, extra = model.loading.advance(sample.state, event_time,
                max_sample_myr=model.parameters.max_loading_interval_myr)
            sample = model.loading.sample(context)
            passive_samples += len(extra)
        _check_sample(sample, event_time)
        parcels, cooling = refresh_parcel_mechanics(tuple(losses[i].parcel for i in indices),
            event_time, sample, model, origin_time_myr=provenance["origin_time_myr"],
            thermal_diffusivity_m2_s=provenance["local_cooling_diffusivity_m2_s"])
        for index, parcel in zip(indices, parcels):
            refreshed[index] = replace(losses[index], parcel=parcel)
        for key in source_parts:
            source_parts[key].append(cooling["thermal_source_delta"][key])
    return tuple(refreshed), dict(thermal_source_delta={key: math.fsum(parts) for key, parts in source_parts.items()},
        event_time_count=len(groups), passive_thermal_sample_count=passive_samples,
        additional_global_heat_sink_j=0., thermal_owner="Genesis; passive capture-time evaluations only")


def advance_coupling(mesh, state, model, dt_myr, *, birth_factory):
    """Return a complete next transaction; caller state is never mutated."""
    dt = float(dt_myr)
    if not math.isfinite(dt) or dt <= 0 or state.surface.time_myr+dt <= state.surface.time_myr:
        raise ValueError("Fractional coupling time step must advance by a finite positive amount")
    before = model.loading.sample(state.thermal_context)
    _check_sample(before, state.surface.time_myr)
    ledger_diagnostics(state)
    _, start_cooling, driving = _refresh(mesh, state.surface, before, model, state.provenance, state.plate_count)
    if any(value != 0. for value in start_cooling["thermal_source_delta"].values()):
        raise ValueError("Fractional coupling requires mechanics refreshed at the current thermal clock")
    target = state.surface.time_myr+dt
    context, samples = model.loading.advance(state.thermal_context, target,
        max_sample_myr=model.parameters.max_loading_interval_myr)
    after = model.loading.sample(context)
    _check_sample(after, target)
    for sample in samples:
        _check_sample(sample, sample.time_myr)
    scheme = time_scheme(state.provenance)
    transported = advance_fractional_transport(mesh, state.surface, driving.omega_rad_per_myr,
        state.provenance["radius_km"], dt, birth_factory=birth_factory,
        event_time_quadrature=EVENT_QUADRATURE if scheme == DEFAULT_TIME_SCHEME else None)
    if scheme == DEFAULT_TIME_SCHEME:
        removed, removed_cooling = refresh_removed_material(transported.losses, before, samples, model, state.provenance)
    else:
        removed, removed_cooling = transported.losses, {"thermal_source_delta": {}}
    surface, cooling, endpoint = _refresh(mesh, transported.state, after, model,
        state.provenance, state.plate_count)
    step_losses = _totals(loss.parcel for loss in removed)
    step_births = _totals(transported.births)
    losses = {k: math.fsum((state.cumulative_losses[k], step_losses[k])) for k in EXTENSIVE}
    births = {k: math.fsum((state.cumulative_births[k], step_births[k])) for k in EXTENSIVE}
    sources = {k: math.fsum((state.cumulative_thermal_sources[k],
                          cooling["thermal_source_delta"].get(k, 0.),
                          removed_cooling["thermal_source_delta"].get(k, 0.))) for k in EXTENSIVE}
    result = CoupledFractionalState(surface, context, state.provenance, state.plate_count,
        state.initial_totals, losses, births, sources, state.removed_material+removed,
        state.history, state.initial_diagnostics, state.thermal_sample_count+len(samples))
    row = dict(time_myr=surface.time_myr,
        elapsed_since_source_myr=surface.time_myr-state.provenance["source_time_myr"], step_myr=dt,
        parcel_count=len(surface.parcels), removed_record_count=len(result.removed_material),
        heat=heat_diagnostics(after, len(samples)), cooling=cooling,
        transport=transported.diagnostics, driving=dynamics_diagnostics(driving),
        endpoint_dynamics=dynamics_diagnostics(endpoint), **ledger_diagnostics(result))
    if scheme == DEFAULT_TIME_SCHEME:
        row.update(time_scheme=scheme, removed_cooling=removed_cooling)
    return CoupledFractionalState(surface, context, state.provenance, state.plate_count,
        state.initial_totals, losses, births, sources, result.removed_material,
        state.history+(row,), state.initial_diagnostics, result.thermal_sample_count)


def save_coupled_checkpoint(path, mesh, state):
    payload = dict(format=coupling_version(state.provenance), experiment=state.provenance, plate_count=state.plate_count,
        thermal_context=thermal_context_to_json(state.thermal_context), thermal_sample_count=state.thermal_sample_count,
        initial_totals=state.initial_totals, cumulative_losses=state.cumulative_losses,
        cumulative_births=state.cumulative_births, cumulative_thermal_sources=state.cumulative_thermal_sources,
        removed_material=[asdict(loss) for loss in state.removed_material],
        history=list(state.history), initial_diagnostics=state.initial_diagnostics)
    ledger_diagnostics(state)
    return save_fractional_checkpoint(path, mesh, state.surface, state.provenance["radius_km"], provenance=payload)


def load_coupled_checkpoint(path, mesh, model, provenance):
    surface, saved = load_fractional_checkpoint(path, mesh, provenance["radius_km"])
    if saved.get("format") not in (LEGACY_VERSION, VERSION):
        raise ValueError("Expected a fractional heat/basal coupling checkpoint")
    if saved["format"] != coupling_version(provenance):
        raise ValueError("Coupled checkpoint uses another event time scheme; select its saved scheme explicitly")
    if saved.get("experiment") != provenance:
        raise ValueError("Coupled resume requires the identical source and physical parameters")
    context = thermal_context_from_json(saved["thermal_context"])
    _check_sample(model.loading.sample(context), surface.time_myr)
    removed = tuple(FractionalLoss(parcel=SurfaceParcel(**record["parcel"]),
        source_cell=record["source_cell"], source_fraction=record["source_fraction"],
        time_myr=record["time_myr"], receiver_plate_fractions=tuple(map(tuple, record["receiver_plate_fractions"])))
        for record in saved["removed_material"])
    state = CoupledFractionalState(surface, context, deepcopy(provenance), saved["plate_count"],
        saved["initial_totals"], saved["cumulative_losses"], saved["cumulative_births"],
        saved["cumulative_thermal_sources"], removed, tuple(saved["history"]), saved["initial_diagnostics"],
        saved["thermal_sample_count"])
    ledger_diagnostics(state)
    archived_totals = _totals(loss.parcel for loss in removed)
    if any(not math.isclose(archived_totals[k], state.cumulative_losses[k], rel_tol=5e-12, abs_tol=0.)
           for k in EXTENSIVE):
        raise ValueError("Coupled checkpoint removed material archive disagrees with its ledger")
    if surface.time_myr < provenance["source_time_myr"] or (
            state.history and state.history[-1]["time_myr"] != surface.time_myr):
        raise ValueError("Coupled checkpoint history and material clocks disagree")
    return state
