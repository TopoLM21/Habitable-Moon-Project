"""v0.29 plume-head/tail hotspot tracks and post-magmatic evolution.

This module is an effective coarse-grid closure for four processes that were
deliberately absent from v0.28:

* a short, broad plume head followed by a narrow persistent tail;
* advected magmatic heat that temporarily weakens the plate;
* preferential dyke emplacement in already extending continental rifts;
* cooling, eclogitization and possible foundering of old mafic underplate.

The permanent igneous reservoirs remain owned by :mod:`plume_magmatism`.
Underplate foundering is transferred to that module's existing deep-recycled
ledger, so phase evolution cannot create or destroy igneous material.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .mesh import SphereMesh
from .plume_magmatism import (
    PlumeMagmatismDiagnostics,
    PlumeMagmatismParameters,
    PlumeMagmatismState,
    advance_plume_magmatism,
    total_igneous_volume_field,
)
from .plumes import MantlePlumeParameters, MantlePlumeState

Array = np.ndarray


@dataclass(slots=True)
class HotspotTrackParameters:
    enabled: bool = True
    head_tail_separation_enabled: bool = True
    head_duration_fraction: float = 0.18
    head_rise_fraction: float = 0.25
    head_decay_fraction: float = 0.50
    tail_radius_fraction: float = 0.23
    tail_flux_fraction: float = 0.55
    tail_rise_time_fraction: float = 0.08
    tail_decay_time_fraction: float = 0.15
    area_normalize_component_flux: bool = True
    flux_saturation: float = 0.75
    productivity_exponent: float = 1.4
    productivity_threshold: float = 0.025
    magmatic_thermal_weakening_enabled: bool = True
    couple_thermal_to_lithosphere: bool = True
    thermal_relaxation_myr: float = 90.0
    thermal_gain_per_km_emplaced: float = 0.45
    maximum_thermal_extension_forcing: float = 0.32
    dike_localization_enabled: bool = True
    dike_rift_onset: float = 0.18
    dike_rift_saturation: float = 0.70
    maximum_dyke_fraction_gain: float = 0.30
    underplate_evolution_enabled: bool = True
    eclogitization_onset_age_myr: float = 120.0
    eclogitization_transition_myr: float = 120.0
    eclogitization_relaxation_myr: float = 180.0
    maximum_eclogite_fraction: float = 0.65
    eclogite_density_kg_m3: float = 3450.0
    delamination_onset_fraction: float = 0.42
    delamination_timescale_myr: float = 120.0
    mapped_track_volume_km3: float = 1.0e-9


@dataclass(slots=True)
class HotspotTrackState:
    time_myr: float
    thermal_anomaly: Array
    underplate_mean_age_myr: Array
    underplate_eclogite_fraction: Array
    last_dike_localization: Array
    last_head_productivity: Array
    last_tail_productivity: Array
    cumulative_delaminated_underplate_volume_km3: float = 0.0
    cumulative_head_generated_volume_km3: float = 0.0
    cumulative_tail_generated_volume_km3: float = 0.0


@dataclass(slots=True)
class HotspotTrackDiagnostics:
    time_myr: float
    dt_myr: float
    enabled: bool
    head_tail_separation_enabled: bool
    magmatic_thermal_weakening_enabled: bool
    dike_localization_enabled: bool
    underplate_evolution_enabled: bool
    mean_head_productivity: float
    maximum_head_productivity: float
    mean_tail_productivity: float
    maximum_tail_productivity: float
    active_tail_area_fraction: float
    mean_thermal_anomaly: float
    maximum_thermal_anomaly: float
    maximum_thermal_extension_forcing: float
    mean_dike_localization: float
    maximum_dike_localization: float
    mean_underplate_age_myr: float
    maximum_underplate_age_myr: float
    eclogitized_underplate_volume_km3: float
    mean_underplate_eclogite_fraction: float
    maximum_underplate_eclogite_fraction: float
    delaminated_underplate_this_step_km3: float
    cumulative_delaminated_underplate_volume_km3: float
    cumulative_head_generated_volume_km3: float
    cumulative_tail_generated_volume_km3: float
    hotspot_track_age_distance_correlation: float


def _validate(params: HotspotTrackParameters) -> None:
    if params.flux_saturation <= 0.0 or params.productivity_exponent <= 0.0:
        raise ValueError("hotspot productivity scaling must be positive")
    if not (0.0 <= params.productivity_threshold <= 1.0):
        raise ValueError("productivity_threshold must be in [0, 1]")
    if params.thermal_relaxation_myr <= 0.0:
        raise ValueError("thermal_relaxation_myr must be positive")
    if params.thermal_gain_per_km_emplaced < 0.0:
        raise ValueError("thermal_gain_per_km_emplaced must be non-negative")
    if not (0.0 <= params.maximum_thermal_extension_forcing <= 1.0):
        raise ValueError("maximum thermal forcing must be in [0, 1]")
    if not (0.0 <= params.dike_rift_onset < params.dike_rift_saturation):
        raise ValueError("dike rift onset must be below saturation")
    if params.maximum_dyke_fraction_gain < 0.0:
        raise ValueError("maximum_dyke_fraction_gain must be non-negative")
    if params.eclogitization_onset_age_myr < 0.0:
        raise ValueError("eclogitization onset age must be non-negative")
    if params.eclogitization_transition_myr <= 0.0:
        raise ValueError("eclogitization transition must be positive")
    if params.eclogitization_relaxation_myr <= 0.0:
        raise ValueError("eclogitization relaxation must be positive")
    if not (0.0 <= params.maximum_eclogite_fraction <= 1.0):
        raise ValueError("maximum eclogite fraction must be in [0, 1]")
    if params.eclogite_density_kg_m3 <= 0.0:
        raise ValueError("eclogite density must be positive")
    if not (0.0 <= params.delamination_onset_fraction < 1.0):
        raise ValueError("delamination onset must be in [0, 1)")
    if params.delamination_timescale_myr <= 0.0:
        raise ValueError("delamination timescale must be positive")


def plume_parameters_with_head_tail(
    params: MantlePlumeParameters,
    track_params: HotspotTrackParameters,
) -> MantlePlumeParameters:
    """Return plume parameters configured for the v0.29 component fields."""

    _validate(track_params)
    return replace(
        params,
        head_tail_separation_enabled=bool(
            track_params.enabled and track_params.head_tail_separation_enabled
        ),
        head_duration_fraction=float(track_params.head_duration_fraction),
        head_rise_fraction=float(track_params.head_rise_fraction),
        head_decay_fraction=float(track_params.head_decay_fraction),
        tail_radius_fraction=float(track_params.tail_radius_fraction),
        tail_flux_fraction=float(track_params.tail_flux_fraction),
        tail_rise_time_fraction=float(track_params.tail_rise_time_fraction),
        tail_decay_time_fraction=float(track_params.tail_decay_time_fraction),
        component_flux_area_normalization_enabled=bool(
            track_params.area_normalize_component_flux
        ),
    )


def initialize_hotspot_tracks(
    mesh: SphereMesh, time_myr: float = 0.0
) -> HotspotTrackState:
    zero = np.zeros(mesh.cell_count, dtype=np.float64)
    return HotspotTrackState(
        time_myr=float(time_myr),
        thermal_anomaly=zero.copy(),
        underplate_mean_age_myr=zero.copy(),
        underplate_eclogite_fraction=zero.copy(),
        last_dike_localization=zero.copy(),
        last_head_productivity=zero.copy(),
        last_tail_productivity=zero.copy(),
    )


def advect_hotspot_tracks(
    state: HotspotTrackState,
    source_index: Array,
) -> HotspotTrackState:
    """Move intensive thermal/phase memory with the winning surface parcel."""

    n = len(state.thermal_anomaly)
    src = np.asarray(source_index, dtype=np.int64)
    if src.shape != (n,):
        raise ValueError("source_index must match cell count")
    valid = (src >= 0) & (src < n)
    for name in (
        "thermal_anomaly",
        "underplate_mean_age_myr",
        "underplate_eclogite_fraction",
        "last_dike_localization",
        "last_head_productivity",
        "last_tail_productivity",
    ):
        old = np.asarray(getattr(state, name), dtype=np.float64)
        if old.shape != (n,):
            raise ValueError("hotspot-track grid fields must share one cell shape")
        moved = np.zeros(n, dtype=np.float64)
        moved[valid] = old[src[valid]]
        setattr(state, name, moved)
    return state


def _smoothstep(values: Array) -> Array:
    x = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _component_productivity(
    plume_state: MantlePlumeState,
    params: HotspotTrackParameters,
    cell_count: int,
) -> tuple[Array, Array]:
    head_flux = (
        np.zeros(cell_count, dtype=np.float64)
        if plume_state.last_head_flux is None
        else np.asarray(plume_state.last_head_flux, dtype=np.float64)
    )
    tail_flux = (
        np.zeros(cell_count, dtype=np.float64)
        if plume_state.last_tail_flux is None
        else np.asarray(plume_state.last_tail_flux, dtype=np.float64)
    )
    if head_flux.shape != (cell_count,) or tail_flux.shape != (cell_count,):
        raise ValueError("plume component fluxes must match cell count")
    scale = max(float(params.flux_saturation), 1.0e-30)
    exponent = float(params.productivity_exponent)
    head = np.power(np.clip(head_flux / scale, 0.0, 1.0), exponent)
    tail = np.power(np.clip(tail_flux / scale, 0.0, 1.0), exponent)
    head[head < float(params.productivity_threshold)] = 0.0
    tail[tail < float(params.productivity_threshold)] = 0.0
    return head, tail


def dike_localization_field(
    rift_extension: Array,
    params: HotspotTrackParameters,
) -> Array:
    extension = np.asarray(rift_extension, dtype=np.float64)
    scaled = (extension - float(params.dike_rift_onset)) / max(
        float(params.dike_rift_saturation - params.dike_rift_onset), 1.0e-30
    )
    if not params.enabled or not params.dike_localization_enabled:
        return np.zeros_like(extension)
    return _smoothstep(scaled)


def magmatic_extension_forcing(
    state: HotspotTrackState,
    params: HotspotTrackParameters,
) -> Array:
    if (
        not params.enabled
        or not params.magmatic_thermal_weakening_enabled
        or not params.couple_thermal_to_lithosphere
    ):
        return np.zeros_like(state.thermal_anomaly)
    return float(params.maximum_thermal_extension_forcing) * _smoothstep(
        state.thermal_anomaly
    )


def underplate_density_field(
    state: HotspotTrackState,
    magmatic_params: PlumeMagmatismParameters,
    params: HotspotTrackParameters,
) -> Array:
    fraction = np.clip(state.underplate_eclogite_fraction, 0.0, 1.0)
    return (
        (1.0 - fraction) * float(magmatic_params.underplate_density_kg_m3)
        + fraction * float(params.eclogite_density_kg_m3)
    )


def _track_age_distance_correlation(
    mesh: SphereMesh,
    magmatic_state: PlumeMagmatismState,
    plume_state: MantlePlumeState,
    radius_km: float,
    params: HotspotTrackParameters,
) -> float:
    total = total_igneous_volume_field(magmatic_state)
    age = np.asarray(magmatic_state.track_age_myr, dtype=np.float64)
    mapped = (total > float(params.mapped_track_volume_km3)) & (age > 0.0)
    centers = np.asarray(plume_state.centers_unit, dtype=np.float64)
    if np.count_nonzero(mapped) < 3 or centers.size == 0:
        return 0.0
    dots = np.clip(mesh.centroids[mapped] @ centers.T, -1.0, 1.0)
    distance = float(radius_km) * np.min(np.arccos(dots), axis=1)
    mapped_age = age[mapped]
    if float(np.std(distance)) <= 1.0e-12 or float(np.std(mapped_age)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(mapped_age, distance)[0, 1])


def diagnose_hotspot_tracks(
    mesh: SphereMesh,
    state: HotspotTrackState,
    magmatic_state: PlumeMagmatismState,
    plume_state: MantlePlumeState,
    radius_km: float,
    params: HotspotTrackParameters,
    *,
    dt_myr: float = 0.0,
    delaminated_this_step_km3: float = 0.0,
) -> HotspotTrackDiagnostics:
    areas = mesh.physical_cell_areas_km2(float(radius_km))
    total_area = max(float(np.sum(areas)), 1.0e-30)
    underplate = np.maximum(
        np.asarray(magmatic_state.underplate_volume_km3, dtype=np.float64), 0.0
    )
    total_underplate = max(float(np.sum(underplate)), 1.0e-30)
    head = np.asarray(state.last_head_productivity, dtype=np.float64)
    tail = np.asarray(state.last_tail_productivity, dtype=np.float64)
    thermal_forcing = magmatic_extension_forcing(state, params)
    eclogite_fraction = np.clip(state.underplate_eclogite_fraction, 0.0, 1.0)
    eclogite_volume = underplate * eclogite_fraction
    present = underplate > 0.0

    def area_mean(values: Array) -> float:
        return float(np.sum(areas * np.asarray(values, dtype=np.float64)) / total_area)

    return HotspotTrackDiagnostics(
        time_myr=float(state.time_myr),
        dt_myr=float(dt_myr),
        enabled=bool(params.enabled),
        head_tail_separation_enabled=bool(params.head_tail_separation_enabled),
        magmatic_thermal_weakening_enabled=bool(
            params.magmatic_thermal_weakening_enabled
        ),
        dike_localization_enabled=bool(params.dike_localization_enabled),
        underplate_evolution_enabled=bool(params.underplate_evolution_enabled),
        mean_head_productivity=area_mean(head),
        maximum_head_productivity=float(np.max(head)) if len(head) else 0.0,
        mean_tail_productivity=area_mean(tail),
        maximum_tail_productivity=float(np.max(tail)) if len(tail) else 0.0,
        active_tail_area_fraction=float(
            np.sum(areas[tail >= float(params.productivity_threshold)]) / total_area
        ),
        mean_thermal_anomaly=area_mean(state.thermal_anomaly),
        maximum_thermal_anomaly=(
            float(np.max(state.thermal_anomaly)) if len(state.thermal_anomaly) else 0.0
        ),
        maximum_thermal_extension_forcing=(
            float(np.max(thermal_forcing)) if len(thermal_forcing) else 0.0
        ),
        mean_dike_localization=area_mean(state.last_dike_localization),
        maximum_dike_localization=(
            float(np.max(state.last_dike_localization))
            if len(state.last_dike_localization)
            else 0.0
        ),
        mean_underplate_age_myr=(
            float(np.sum(underplate * state.underplate_mean_age_myr) / total_underplate)
            if np.any(present)
            else 0.0
        ),
        maximum_underplate_age_myr=(
            float(np.max(state.underplate_mean_age_myr[present]))
            if np.any(present)
            else 0.0
        ),
        eclogitized_underplate_volume_km3=float(np.sum(eclogite_volume)),
        mean_underplate_eclogite_fraction=(
            float(np.sum(underplate * eclogite_fraction) / total_underplate)
            if np.any(present)
            else 0.0
        ),
        maximum_underplate_eclogite_fraction=(
            float(np.max(eclogite_fraction[present])) if np.any(present) else 0.0
        ),
        delaminated_underplate_this_step_km3=float(delaminated_this_step_km3),
        cumulative_delaminated_underplate_volume_km3=float(
            state.cumulative_delaminated_underplate_volume_km3
        ),
        cumulative_head_generated_volume_km3=float(
            state.cumulative_head_generated_volume_km3
        ),
        cumulative_tail_generated_volume_km3=float(
            state.cumulative_tail_generated_volume_km3
        ),
        hotspot_track_age_distance_correlation=_track_age_distance_correlation(
            mesh, magmatic_state, plume_state, radius_km, params
        ),
    )


def advance_hotspot_tracks(
    mesh: SphereMesh,
    state: HotspotTrackState,
    magmatic_state: PlumeMagmatismState,
    plume_state: MantlePlumeState,
    rift_extension: Array,
    extension_forcing: Array,
    fallback_productivity: Array,
    dt_myr: float,
    radius_km: float,
    magmatic_params: PlumeMagmatismParameters,
    params: HotspotTrackParameters,
) -> tuple[
    HotspotTrackState,
    PlumeMagmatismState,
    HotspotTrackDiagnostics,
    PlumeMagmatismDiagnostics,
]:
    """Advance phase/thermal memory and emplace the next hotspot increment."""

    if dt_myr <= 0.0:
        raise ValueError("dt_myr must be positive")
    _validate(params)
    n = mesh.cell_count
    for field in (
        state.thermal_anomaly,
        state.underplate_mean_age_myr,
        state.underplate_eclogite_fraction,
    ):
        if np.asarray(field).shape != (n,):
            raise ValueError("hotspot-track fields must match cell count")
    rift = np.asarray(rift_extension, dtype=np.float64)
    plume_extension = np.asarray(extension_forcing, dtype=np.float64)
    fallback = np.asarray(fallback_productivity, dtype=np.float64)
    if rift.shape != (n,) or plume_extension.shape != (n,) or fallback.shape != (n,):
        raise ValueError("forcing fields must match cell count")

    # Cooling is represented by decay of the transported thermal anomaly.
    state.thermal_anomaly = np.clip(
        np.asarray(state.thermal_anomaly, dtype=np.float64)
        * np.exp(-float(dt_myr) / float(params.thermal_relaxation_myr)),
        0.0,
        1.0,
    )
    underplate_before = np.maximum(
        np.asarray(magmatic_state.underplate_volume_km3, dtype=np.float64), 0.0
    )
    age = np.asarray(state.underplate_mean_age_myr, dtype=np.float64).copy()
    age[underplate_before > 0.0] += float(dt_myr)
    age[underplate_before <= 0.0] = 0.0
    fraction = np.clip(
        np.asarray(state.underplate_eclogite_fraction, dtype=np.float64), 0.0, 1.0
    )
    delaminated = np.zeros(n, dtype=np.float64)

    if params.enabled and params.underplate_evolution_enabled:
        maturity = _smoothstep(
            (age - float(params.eclogitization_onset_age_myr))
            / float(params.eclogitization_transition_myr)
        )
        target = float(params.maximum_eclogite_fraction) * maturity
        relax = 1.0 - np.exp(
            -float(dt_myr) / float(params.eclogitization_relaxation_myr)
        )
        fraction = fraction + np.maximum(target - fraction, 0.0) * relax
        activation = _smoothstep(
            (fraction - float(params.delamination_onset_fraction))
            / max(1.0 - float(params.delamination_onset_fraction), 1.0e-30)
        )
        removal_fraction = activation * (
            1.0 - np.exp(-float(dt_myr) / float(params.delamination_timescale_myr))
        )
        eclogite_volume = underplate_before * fraction
        delaminated = np.minimum(eclogite_volume * removal_fraction, underplate_before)
        remaining = np.maximum(underplate_before - delaminated, 0.0)
        remaining_eclogite = np.maximum(eclogite_volume - delaminated, 0.0)
        fraction = np.divide(
            remaining_eclogite,
            remaining,
            out=np.zeros_like(remaining),
            where=remaining > 0.0,
        )
        magmatic_state.underplate_volume_km3 = remaining
        removed_total = float(np.sum(delaminated))
        magmatic_state.deep_recycled_underplate_volume_km3 += removed_total
        state.cumulative_delaminated_underplate_volume_km3 += removed_total

    post_evolution_underplate = np.maximum(
        np.asarray(magmatic_state.underplate_volume_km3, dtype=np.float64), 0.0
    ).copy()
    post_evolution_eclogite = post_evolution_underplate * fraction

    if params.enabled and params.head_tail_separation_enabled:
        head, tail = _component_productivity(plume_state, params, n)
        productivity = np.clip(head + tail, 0.0, 1.0)
    else:
        head = np.clip(fallback, 0.0, 1.0)
        tail = np.zeros(n, dtype=np.float64)
        productivity = head.copy()
    localization = dike_localization_field(rift, params)

    before_extrusive = np.asarray(
        magmatic_state.extrusive_volume_km3, dtype=np.float64
    ).copy()
    before_dyke = np.asarray(magmatic_state.dyke_volume_km3, dtype=np.float64).copy()
    before_underplate = post_evolution_underplate.copy()
    magmatic_state, magmatic_diagnostics = advance_plume_magmatism(
        mesh,
        magmatic_state,
        productivity,
        plume_extension,
        dt_myr,
        radius_km,
        magmatic_params,
        dyke_localization=localization,
        maximum_dyke_fraction_gain=(
            float(params.maximum_dyke_fraction_gain)
            if params.enabled and params.dike_localization_enabled
            else 0.0
        ),
    )
    generated_extrusive = np.maximum(
        magmatic_state.extrusive_volume_km3 - before_extrusive, 0.0
    )
    generated_dyke = np.maximum(magmatic_state.dyke_volume_km3 - before_dyke, 0.0)
    generated_underplate = np.maximum(
        magmatic_state.underplate_volume_km3 - before_underplate, 0.0
    )
    generated_total = generated_extrusive + generated_dyke + generated_underplate

    final_underplate = np.maximum(
        np.asarray(magmatic_state.underplate_volume_km3, dtype=np.float64), 0.0
    )
    state.underplate_mean_age_myr = np.divide(
        post_evolution_underplate * age,
        final_underplate,
        out=np.zeros(n, dtype=np.float64),
        where=final_underplate > 0.0,
    )
    state.underplate_eclogite_fraction = np.divide(
        post_evolution_eclogite,
        final_underplate,
        out=np.zeros(n, dtype=np.float64),
        where=final_underplate > 0.0,
    )

    if params.enabled and params.magmatic_thermal_weakening_enabled:
        areas = mesh.physical_cell_areas_km2(float(radius_km))
        generated_thickness_km = generated_total / np.maximum(areas, 1.0e-30)
        heat_increment = 1.0 - np.exp(
            -float(params.thermal_gain_per_km_emplaced) * generated_thickness_km
        )
        state.thermal_anomaly = np.clip(
            state.thermal_anomaly
            + (1.0 - state.thermal_anomaly) * heat_increment,
            0.0,
            1.0,
        )

    source_sum = head + tail
    head_share = np.divide(
        head,
        source_sum,
        out=np.zeros(n, dtype=np.float64),
        where=source_sum > 0.0,
    )
    generated_head = float(np.sum(generated_total * head_share))
    generated_tail = float(np.sum(generated_total)) - generated_head
    state.cumulative_head_generated_volume_km3 += generated_head
    state.cumulative_tail_generated_volume_km3 += generated_tail
    state.last_head_productivity = head
    state.last_tail_productivity = tail
    state.last_dike_localization = localization
    state.time_myr = float(state.time_myr) + float(dt_myr)

    diagnostics = diagnose_hotspot_tracks(
        mesh,
        state,
        magmatic_state,
        plume_state,
        radius_km,
        params,
        dt_myr=dt_myr,
        delaminated_this_step_km3=float(np.sum(delaminated)),
    )
    return state, magmatic_state, diagnostics, magmatic_diagnostics


__all__ = [
    "HotspotTrackParameters",
    "HotspotTrackState",
    "HotspotTrackDiagnostics",
    "plume_parameters_with_head_tail",
    "initialize_hotspot_tracks",
    "advect_hotspot_tracks",
    "dike_localization_field",
    "magmatic_extension_forcing",
    "underplate_density_field",
    "diagnose_hotspot_tracks",
    "advance_hotspot_tracks",
]
