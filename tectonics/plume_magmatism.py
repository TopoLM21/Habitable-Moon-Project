"""v0.28 permanent plume-magmatic crust and transported volcanic tracks.

The mantle-plume and plume-rifting layers provide an Eulerian melt-productivity
field.  This module converts that field into three material reservoirs that
move with the winning surface parcel: extrusive basalt, crustal dykes/sills and
mafic underplate.  Material that loses its surface parcel is recorded as deep
recycling, so generated igneous volume has a closed global ledger.

The model is deliberately an effective coarse-grid closure rather than a
petrological phase-equilibrium or magma-chamber solver.  Partition fractions,
emplacement rate and densities are explicit parameters so factorial controls
can switch magmatism independently from mechanical rifting and dynamic uplift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import SphereMesh

Array = np.ndarray


@dataclass(slots=True)
class PlumeMagmatismParameters:
    """Effective decompression-melt extraction and emplacement parameters."""

    enabled: bool = True
    maximum_emplacement_thickness_km_per_myr: float = 0.018
    minimum_productivity: float = 0.025
    background_extraction_fraction: float = 0.25
    extension_extraction_gain: float = 0.75
    maximum_igneous_thickness_km: float = 25.0
    extrusive_fraction: float = 0.09
    dyke_fraction: float = 0.21
    underplate_fraction: float = 0.70
    extrusive_density_kg_m3: float = 2900.0
    dyke_density_kg_m3: float = 2950.0
    underplate_density_kg_m3: float = 3050.0
    mantle_density_kg_m3: float = 3300.0
    mapped_track_volume_km3: float = 1.0e-9


@dataclass(slots=True)
class PlumeMagmatismState:
    """Checkpointable material reservoirs and global volume ledger."""

    time_myr: float
    extrusive_volume_km3: Array
    dyke_volume_km3: Array
    underplate_volume_km3: Array
    track_age_myr: Array
    last_emplacement_productivity: Array
    cumulative_generated_extrusive_volume_km3: float = 0.0
    cumulative_generated_dyke_volume_km3: float = 0.0
    cumulative_generated_underplate_volume_km3: float = 0.0
    deep_recycled_extrusive_volume_km3: float = 0.0
    deep_recycled_dyke_volume_km3: float = 0.0
    deep_recycled_underplate_volume_km3: float = 0.0


@dataclass(slots=True)
class PlumeMagmatismDiagnostics:
    time_myr: float
    dt_myr: float
    enabled: bool
    mean_emplacement_productivity: float
    maximum_emplacement_productivity: float
    active_emplacement_area_fraction: float
    generated_extrusive_volume_km3: float
    generated_dyke_volume_km3: float
    generated_underplate_volume_km3: float
    cumulative_generated_total_volume_km3: float
    surface_extrusive_volume_km3: float
    surface_dyke_volume_km3: float
    surface_underplate_volume_km3: float
    deep_recycled_total_volume_km3: float
    global_igneous_ledger_error_km3: float
    mean_igneous_thickness_km: float
    maximum_igneous_thickness_km: float
    mapped_track_area_fraction: float
    maximum_track_age_myr: float
    mean_density_aware_support_m: float
    maximum_density_aware_support_m: float


def _validate_parameters(params: PlumeMagmatismParameters) -> None:
    if params.maximum_emplacement_thickness_km_per_myr < 0.0:
        raise ValueError("maximum_emplacement_thickness_km_per_myr must be non-negative")
    if not (0.0 <= params.minimum_productivity <= 1.0):
        raise ValueError("minimum_productivity must be in [0, 1]")
    if not (0.0 <= params.background_extraction_fraction <= 1.0):
        raise ValueError("background_extraction_fraction must be in [0, 1]")
    if params.extension_extraction_gain < 0.0:
        raise ValueError("extension_extraction_gain must be non-negative")
    if params.maximum_igneous_thickness_km <= 0.0:
        raise ValueError("maximum_igneous_thickness_km must be positive")
    fractions = np.asarray(
        [params.extrusive_fraction, params.dyke_fraction, params.underplate_fraction],
        dtype=np.float64,
    )
    if np.any(fractions < 0.0) or not np.isclose(float(np.sum(fractions)), 1.0):
        raise ValueError("igneous reservoir fractions must be non-negative and sum to one")
    densities = np.asarray(
        [
            params.extrusive_density_kg_m3,
            params.dyke_density_kg_m3,
            params.underplate_density_kg_m3,
        ],
        dtype=np.float64,
    )
    if params.mantle_density_kg_m3 <= 0.0 or np.any(densities <= 0.0):
        raise ValueError("igneous and mantle densities must be positive")
    if np.any(densities >= float(params.mantle_density_kg_m3)):
        raise ValueError("igneous reservoir densities must be below mantle density")


def initialize_plume_magmatism(
    mesh: SphereMesh, time_myr: float = 0.0
) -> PlumeMagmatismState:
    zero = np.zeros(mesh.cell_count, dtype=np.float64)
    return PlumeMagmatismState(
        time_myr=float(time_myr),
        extrusive_volume_km3=zero.copy(),
        dyke_volume_km3=zero.copy(),
        underplate_volume_km3=zero.copy(),
        track_age_myr=zero.copy(),
        last_emplacement_productivity=zero.copy(),
    )


def total_igneous_volume_field(state: PlumeMagmatismState) -> Array:
    return (
        np.maximum(np.asarray(state.extrusive_volume_km3, dtype=np.float64), 0.0)
        + np.maximum(np.asarray(state.dyke_volume_km3, dtype=np.float64), 0.0)
        + np.maximum(np.asarray(state.underplate_volume_km3, dtype=np.float64), 0.0)
    )


def igneous_ledger_error_km3(state: PlumeMagmatismState) -> float:
    generated = (
        float(state.cumulative_generated_extrusive_volume_km3)
        + float(state.cumulative_generated_dyke_volume_km3)
        + float(state.cumulative_generated_underplate_volume_km3)
    )
    recycled = (
        float(state.deep_recycled_extrusive_volume_km3)
        + float(state.deep_recycled_dyke_volume_km3)
        + float(state.deep_recycled_underplate_volume_km3)
    )
    return generated - float(np.sum(total_igneous_volume_field(state))) - recycled


def _advect_volume(values: Array, source_index: Array) -> tuple[Array, float, Array]:
    old = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    src = np.asarray(source_index, dtype=np.int64)
    n = len(old)
    if src.shape != (n,):
        raise ValueError("source_index must match cell count")
    valid = (src >= 0) & (src < n)
    counts = np.bincount(src[valid], minlength=n).astype(np.float64)
    out = np.zeros(n, dtype=np.float64)
    if np.any(valid):
        out[valid] = old[src[valid]] / np.maximum(counts[src[valid]], 1.0)
    used = counts > 0.0
    return out, float(np.sum(old[~used])), valid


def advect_plume_magmatism(
    state: PlumeMagmatismState,
    source_index: Array,
    dt_myr: float,
) -> PlumeMagmatismState:
    """Move reservoirs with surface parcels and close losses to deep recycling."""

    if dt_myr <= 0.0:
        raise ValueError("dt_myr must be positive")
    n = len(state.extrusive_volume_km3)
    fields = (
        state.extrusive_volume_km3,
        state.dyke_volume_km3,
        state.underplate_volume_km3,
        state.track_age_myr,
        state.last_emplacement_productivity,
    )
    if any(np.asarray(field).shape != (n,) for field in fields):
        raise ValueError("plume-magmatism grid fields must share one cell shape")

    extrusive, lost_extrusive, valid = _advect_volume(
        state.extrusive_volume_km3, source_index
    )
    dykes, lost_dykes, _ = _advect_volume(state.dyke_volume_km3, source_index)
    underplate, lost_underplate, _ = _advect_volume(
        state.underplate_volume_km3, source_index
    )
    src = np.asarray(source_index, dtype=np.int64)
    age = np.zeros(n, dtype=np.float64)
    productivity = np.zeros(n, dtype=np.float64)
    if np.any(valid):
        age[valid] = np.asarray(state.track_age_myr, dtype=np.float64)[src[valid]]
        productivity[valid] = np.asarray(
            state.last_emplacement_productivity, dtype=np.float64
        )[src[valid]]
    present = (extrusive + dykes + underplate) > 0.0
    age[present] += float(dt_myr)
    age[~present] = 0.0

    state.extrusive_volume_km3 = extrusive
    state.dyke_volume_km3 = dykes
    state.underplate_volume_km3 = underplate
    state.track_age_myr = age
    state.last_emplacement_productivity = productivity
    state.deep_recycled_extrusive_volume_km3 += lost_extrusive
    state.deep_recycled_dyke_volume_km3 += lost_dykes
    state.deep_recycled_underplate_volume_km3 += lost_underplate
    return state


def magmatic_topography_fields(
    mesh: SphereMesh,
    state: PlumeMagmatismState,
    radius_km: float,
    params: PlumeMagmatismParameters,
) -> tuple[Array, Array, Array, Array]:
    """Return extrusive thickness/load, intrusive support and total thickness.

    The extrusive layer adds its geometric thickness to the surface and applies
    a downward load of ``rho_i/rho_m`` times that thickness.  Dykes and
    underplate replace mantle at depth, giving positive Airy support weighted
    by their individual density contrasts.
    """

    _validate_parameters(params)
    areas = mesh.physical_cell_areas_km2(float(radius_km))
    scale = 1000.0 / np.maximum(areas, 1.0e-30)
    extrusive_h = scale * np.maximum(state.extrusive_volume_km3, 0.0)
    dyke_h = scale * np.maximum(state.dyke_volume_km3, 0.0)
    underplate_h = scale * np.maximum(state.underplate_volume_km3, 0.0)
    rho_m = float(params.mantle_density_kg_m3)
    extrusive_load = -float(params.extrusive_density_kg_m3) / rho_m * extrusive_h
    intrusive_support = (
        (rho_m - float(params.dyke_density_kg_m3)) / rho_m * dyke_h
        + (rho_m - float(params.underplate_density_kg_m3))
        / rho_m
        * underplate_h
    )
    total_h = extrusive_h + dyke_h + underplate_h
    return extrusive_h, extrusive_load, intrusive_support, total_h


def diagnose_plume_magmatism(
    mesh: SphereMesh,
    state: PlumeMagmatismState,
    radius_km: float,
    params: PlumeMagmatismParameters,
    *,
    dt_myr: float = 0.0,
    generated: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PlumeMagmatismDiagnostics:
    areas = mesh.physical_cell_areas_km2(float(radius_km))
    total_area = max(float(np.sum(areas)), 1.0e-30)
    productivity = np.asarray(state.last_emplacement_productivity, dtype=np.float64)
    active = productivity >= float(params.minimum_productivity)
    total_volume = total_igneous_volume_field(state)
    mapped = total_volume > float(params.mapped_track_volume_km3)
    _, extrusive_load, intrusive_support, total_h_m = magmatic_topography_fields(
        mesh, state, radius_km, params
    )
    extrusive_h_m = (
        1000.0
        * np.maximum(state.extrusive_volume_km3, 0.0)
        / np.maximum(areas, 1.0e-30)
    )
    net_support = extrusive_h_m + extrusive_load + intrusive_support
    cumulative_generated = (
        state.cumulative_generated_extrusive_volume_km3
        + state.cumulative_generated_dyke_volume_km3
        + state.cumulative_generated_underplate_volume_km3
    )
    deep = (
        state.deep_recycled_extrusive_volume_km3
        + state.deep_recycled_dyke_volume_km3
        + state.deep_recycled_underplate_volume_km3
    )
    return PlumeMagmatismDiagnostics(
        time_myr=float(state.time_myr),
        dt_myr=float(dt_myr),
        enabled=bool(params.enabled),
        mean_emplacement_productivity=float(np.sum(areas * productivity) / total_area),
        maximum_emplacement_productivity=float(np.max(productivity)) if len(productivity) else 0.0,
        active_emplacement_area_fraction=float(np.sum(areas[active]) / total_area),
        generated_extrusive_volume_km3=float(generated[0]),
        generated_dyke_volume_km3=float(generated[1]),
        generated_underplate_volume_km3=float(generated[2]),
        cumulative_generated_total_volume_km3=float(cumulative_generated),
        surface_extrusive_volume_km3=float(np.sum(state.extrusive_volume_km3)),
        surface_dyke_volume_km3=float(np.sum(state.dyke_volume_km3)),
        surface_underplate_volume_km3=float(np.sum(state.underplate_volume_km3)),
        deep_recycled_total_volume_km3=float(deep),
        global_igneous_ledger_error_km3=float(igneous_ledger_error_km3(state)),
        mean_igneous_thickness_km=float(np.sum(areas * total_h_m) / total_area / 1000.0),
        maximum_igneous_thickness_km=float(np.max(total_h_m) / 1000.0) if len(total_h_m) else 0.0,
        mapped_track_area_fraction=float(np.sum(areas[mapped]) / total_area),
        maximum_track_age_myr=float(np.max(state.track_age_myr[mapped])) if np.any(mapped) else 0.0,
        mean_density_aware_support_m=float(np.sum(areas * net_support) / total_area),
        maximum_density_aware_support_m=float(np.max(net_support)) if len(net_support) else 0.0,
    )


def advance_plume_magmatism(
    mesh: SphereMesh,
    state: PlumeMagmatismState,
    melt_productivity: Array,
    extension_forcing: Array,
    dt_myr: float,
    radius_km: float,
    params: PlumeMagmatismParameters,
) -> tuple[PlumeMagmatismState, PlumeMagmatismDiagnostics]:
    """Emplace one permanent magmatic increment into the transported reservoirs."""

    if dt_myr <= 0.0:
        raise ValueError("dt_myr must be positive")
    _validate_parameters(params)
    n = mesh.cell_count
    productivity = np.clip(np.asarray(melt_productivity, dtype=np.float64), 0.0, 1.0)
    extension = np.clip(np.asarray(extension_forcing, dtype=np.float64), 0.0, 1.0)
    if productivity.shape != (n,) or extension.shape != (n,):
        raise ValueError("productivity and extension fields must match cell count")
    if len(state.extrusive_volume_km3) != n:
        raise ValueError("plume-magmatism state must match cell count")

    if not params.enabled:
        state.time_myr = float(state.time_myr) + float(dt_myr)
        state.last_emplacement_productivity = np.zeros(n, dtype=np.float64)
        return state, diagnose_plume_magmatism(
            mesh, state, radius_km, params, dt_myr=dt_myr
        )

    extraction = np.clip(
        float(params.background_extraction_fraction)
        + float(params.extension_extraction_gain) * extension,
        0.0,
        1.0,
    )
    effective = productivity * extraction
    effective[productivity < float(params.minimum_productivity)] = 0.0
    areas = mesh.physical_cell_areas_km2(float(radius_km))
    current_h = total_igneous_volume_field(state) / np.maximum(areas, 1.0e-30)
    capacity = np.clip(
        1.0 - current_h / float(params.maximum_igneous_thickness_km), 0.0, 1.0
    )
    generated_total = (
        areas
        * float(params.maximum_emplacement_thickness_km_per_myr)
        * float(dt_myr)
        * effective
        * capacity
    )
    generated_extrusive = generated_total * float(params.extrusive_fraction)
    generated_dyke = generated_total * float(params.dyke_fraction)
    generated_underplate = generated_total * float(params.underplate_fraction)
    state.extrusive_volume_km3 = (
        np.asarray(state.extrusive_volume_km3, dtype=np.float64) + generated_extrusive
    )
    state.dyke_volume_km3 = (
        np.asarray(state.dyke_volume_km3, dtype=np.float64) + generated_dyke
    )
    state.underplate_volume_km3 = (
        np.asarray(state.underplate_volume_km3, dtype=np.float64)
        + generated_underplate
    )
    state.last_emplacement_productivity = effective
    state.track_age_myr = np.asarray(state.track_age_myr, dtype=np.float64)
    state.track_age_myr[generated_total > 0.0] = 0.0
    increments = (
        float(np.sum(generated_extrusive)),
        float(np.sum(generated_dyke)),
        float(np.sum(generated_underplate)),
    )
    state.cumulative_generated_extrusive_volume_km3 += increments[0]
    state.cumulative_generated_dyke_volume_km3 += increments[1]
    state.cumulative_generated_underplate_volume_km3 += increments[2]
    state.time_myr = float(state.time_myr) + float(dt_myr)
    diagnostics = diagnose_plume_magmatism(
        mesh,
        state,
        radius_km,
        params,
        dt_myr=dt_myr,
        generated=increments,
    )
    return state, diagnostics


__all__ = [
    "PlumeMagmatismParameters",
    "PlumeMagmatismState",
    "PlumeMagmatismDiagnostics",
    "initialize_plume_magmatism",
    "total_igneous_volume_field",
    "igneous_ledger_error_km3",
    "advect_plume_magmatism",
    "magmatic_topography_fields",
    "diagnose_plume_magmatism",
    "advance_plume_magmatism",
]
