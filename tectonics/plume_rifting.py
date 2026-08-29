"""v0.26 mechanical extension driven by mantle-plume heads.

The v0.25 plume layer weakens and erodes continental mantle roots but does not
apply a horizontal tectonic load.  This module adds that missing load as an
effective surface field.  It represents two first-order contributions:

* broad extension above radial plume-head flow and dynamic uplift;
* an annular localization term near the plume-head flank.

The result is passed to the existing continental-extension machinery.  This
module does not directly change crust type, plate topology, or rift memory.
Those transitions remain governed by the already tested lithosphere and
topology solvers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lithosphere import LithosphereState, continental_material_fields
from .mesh import SphereMesh
from .plumes import MantlePlumeState

Array = np.ndarray


@dataclass(slots=True)
class PlumeRiftingParameters:
    """Effective active-rifting parameters for the surface plate model."""

    enabled: bool = True
    couple_to_lithosphere: bool = True
    flux_onset: float = 0.10
    flux_saturation: float = 0.75
    core_weight: float = 0.90
    annular_weight: float = 0.35
    annular_peak_flux: float = 0.35
    annular_width_flux: float = 0.18
    root_contrast_scale_km: float = 35.0
    root_contrast_boost: float = 0.25
    maximum_extension_forcing: float = 1.0
    maximum_dynamic_uplift_m: float = 1800.0
    magmatic_productivity_exponent: float = 1.4
    forced_area_threshold: float = 0.10
    continental_fraction_epsilon: float = 1.0e-9


@dataclass(slots=True)
class PlumeRiftingState:
    """Checkpointable fixed-grid forcing and accumulated impulse."""

    time_myr: float
    last_extension_forcing: Array
    cumulative_extension_impulse_myr: Array
    last_dynamic_uplift_m: Array
    last_magmatic_productivity: Array


@dataclass(slots=True)
class PlumeRiftingDiagnostics:
    time_myr: float
    dt_myr: float
    enabled: bool
    mean_surface_extension_forcing: float
    max_surface_extension_forcing: float
    forced_surface_area_fraction: float
    forced_continental_material_fraction: float
    mean_continental_extension_forcing: float
    max_root_contrast: float
    mean_dynamic_uplift_m: float
    max_dynamic_uplift_m: float
    mean_magmatic_productivity: float
    max_magmatic_productivity: float
    cumulative_mean_extension_impulse_myr: float
    cumulative_max_extension_impulse_myr: float


def _validate_parameters(params: PlumeRiftingParameters) -> None:
    if not (0.0 <= params.flux_onset < params.flux_saturation):
        raise ValueError("flux_onset must be non-negative and below flux_saturation")
    if params.core_weight < 0.0 or params.annular_weight < 0.0:
        raise ValueError("plume forcing weights must be non-negative")
    if params.annular_width_flux <= 0.0:
        raise ValueError("annular_width_flux must be positive")
    if params.root_contrast_scale_km <= 0.0:
        raise ValueError("root_contrast_scale_km must be positive")
    if params.root_contrast_boost < 0.0:
        raise ValueError("root_contrast_boost must be non-negative")
    if not (0.0 <= params.maximum_extension_forcing <= 1.0):
        raise ValueError("maximum_extension_forcing must be in [0, 1]")
    if params.maximum_dynamic_uplift_m < 0.0:
        raise ValueError("maximum_dynamic_uplift_m must be non-negative")
    if params.magmatic_productivity_exponent <= 0.0:
        raise ValueError("magmatic_productivity_exponent must be positive")


def _smoothstep(values: Array) -> Array:
    x = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _neighbor_root_contrast(mesh: SphereMesh, lithosphere: LithosphereState) -> Array:
    n = mesh.cell_count
    if lithosphere.mantle_lithosphere_thickness_km is None:
        return np.zeros(n, dtype=np.float64)
    root = np.asarray(lithosphere.mantle_lithosphere_thickness_km, dtype=np.float64)
    contrast = np.zeros(n, dtype=np.float64)
    for cell, neighbors in enumerate(mesh.neighbors):
        if neighbors:
            contrast[cell] = abs(root[cell] - float(np.mean(root[list(neighbors)])))
    return contrast


def plume_rifting_fields(
    mesh: SphereMesh,
    lithosphere: LithosphereState,
    plume_state: MantlePlumeState,
    radius_km: float,
    params: PlumeRiftingParameters,
) -> tuple[Array, Array, Array, Array]:
    """Return extension, uplift, magmatic productivity, and root contrast."""

    _validate_parameters(params)
    n = mesh.cell_count
    if plume_state.last_flux.shape != (n,):
        raise ValueError("plume last_flux must have shape (cell_count,)")
    if not params.enabled:
        zero = np.zeros(n, dtype=np.float64)
        return zero.copy(), zero.copy(), zero.copy(), zero.copy()

    flux = np.clip(np.asarray(plume_state.last_flux, dtype=np.float64), 0.0, None)
    scaled = (flux - float(params.flux_onset)) / max(
        float(params.flux_saturation - params.flux_onset), 1.0e-30
    )
    core = _smoothstep(scaled)
    annulus = np.exp(
        -0.5
        * (
            (flux - float(params.annular_peak_flux))
            / float(params.annular_width_flux)
        )
        ** 2
    )
    # Suppress the mathematical Gaussian tail where no plume is present.
    annulus *= _smoothstep(flux / max(float(params.flux_onset), 1.0e-30))

    root_contrast_km = _neighbor_root_contrast(mesh, lithosphere)
    normalized_contrast = np.clip(
        root_contrast_km / float(params.root_contrast_scale_km), 0.0, 1.0
    )
    shape = float(params.core_weight) * core + float(params.annular_weight) * annulus
    forcing = shape * (1.0 + float(params.root_contrast_boost) * normalized_contrast)
    forcing = np.clip(
        forcing,
        0.0,
        float(params.maximum_extension_forcing),
    )

    flux_normalized = np.clip(
        flux / max(float(params.flux_saturation), 1.0e-30), 0.0, 1.0
    )
    uplift = float(params.maximum_dynamic_uplift_m) * _smoothstep(flux_normalized)
    magmatic = np.power(flux_normalized, float(params.magmatic_productivity_exponent))
    return forcing, uplift, magmatic, normalized_contrast


def initialize_plume_rifting(
    mesh: SphereMesh,
    time_myr: float,
) -> PlumeRiftingState:
    zero = np.zeros(mesh.cell_count, dtype=np.float64)
    return PlumeRiftingState(
        time_myr=float(time_myr),
        last_extension_forcing=zero.copy(),
        cumulative_extension_impulse_myr=zero.copy(),
        last_dynamic_uplift_m=zero.copy(),
        last_magmatic_productivity=zero.copy(),
    )


def diagnose_plume_rifting(
    mesh: SphereMesh,
    lithosphere: LithosphereState,
    state: PlumeRiftingState,
    radius_km: float,
    params: PlumeRiftingParameters,
    *,
    dt_myr: float = 0.0,
    root_contrast: Array | None = None,
) -> PlumeRiftingDiagnostics:
    areas = mesh.physical_cell_areas_km2(float(radius_km))
    fraction, _ = continental_material_fields(lithosphere, areas)
    continental_weights = areas * fraction
    total_area = max(float(np.sum(areas)), 1.0e-30)
    total_continent = max(float(np.sum(continental_weights)), 1.0e-30)
    forcing = np.asarray(state.last_extension_forcing, dtype=np.float64)
    forced = forcing >= float(params.forced_area_threshold)

    def area_mean(values: Array) -> float:
        return float(np.sum(areas * np.asarray(values, dtype=np.float64)) / total_area)

    def continental_mean(values: Array) -> float:
        return float(
            np.sum(continental_weights * np.asarray(values, dtype=np.float64))
            / total_continent
        )

    contrast = (
        np.zeros(mesh.cell_count, dtype=np.float64)
        if root_contrast is None
        else np.asarray(root_contrast, dtype=np.float64)
    )
    return PlumeRiftingDiagnostics(
        time_myr=float(state.time_myr),
        dt_myr=float(dt_myr),
        enabled=bool(params.enabled),
        mean_surface_extension_forcing=area_mean(forcing),
        max_surface_extension_forcing=float(np.max(forcing)) if len(forcing) else 0.0,
        forced_surface_area_fraction=float(np.sum(areas[forced]) / total_area),
        forced_continental_material_fraction=float(
            np.sum(continental_weights[forced]) / total_continent
        ),
        mean_continental_extension_forcing=continental_mean(forcing),
        max_root_contrast=float(np.max(contrast)) if len(contrast) else 0.0,
        mean_dynamic_uplift_m=area_mean(state.last_dynamic_uplift_m),
        max_dynamic_uplift_m=(
            float(np.max(state.last_dynamic_uplift_m))
            if len(state.last_dynamic_uplift_m)
            else 0.0
        ),
        mean_magmatic_productivity=area_mean(state.last_magmatic_productivity),
        max_magmatic_productivity=(
            float(np.max(state.last_magmatic_productivity))
            if len(state.last_magmatic_productivity)
            else 0.0
        ),
        cumulative_mean_extension_impulse_myr=area_mean(
            state.cumulative_extension_impulse_myr
        ),
        cumulative_max_extension_impulse_myr=(
            float(np.max(state.cumulative_extension_impulse_myr))
            if len(state.cumulative_extension_impulse_myr)
            else 0.0
        ),
    )


def advance_plume_rifting(
    mesh: SphereMesh,
    lithosphere: LithosphereState,
    plume_state: MantlePlumeState,
    state: PlumeRiftingState,
    dt_myr: float,
    radius_km: float,
    params: PlumeRiftingParameters,
) -> tuple[PlumeRiftingState, Array, PlumeRiftingDiagnostics]:
    """Advance forcing diagnostics and return the external extension field."""

    if dt_myr <= 0.0:
        raise ValueError("dt_myr must be positive")
    n = mesh.cell_count
    for field in (
        state.last_extension_forcing,
        state.cumulative_extension_impulse_myr,
        state.last_dynamic_uplift_m,
        state.last_magmatic_productivity,
    ):
        if np.asarray(field).shape != (n,):
            raise ValueError("plume-rifting grid fields must have shape (cell_count,)")

    forcing, uplift, magmatic, contrast = plume_rifting_fields(
        mesh, lithosphere, plume_state, radius_km, params
    )
    state.time_myr = float(lithosphere.time_myr) + float(dt_myr)
    state.last_extension_forcing = forcing
    state.last_dynamic_uplift_m = uplift
    state.last_magmatic_productivity = magmatic
    state.cumulative_extension_impulse_myr = (
        np.asarray(state.cumulative_extension_impulse_myr, dtype=np.float64)
        + forcing * float(dt_myr)
    )
    diagnostics = diagnose_plume_rifting(
        mesh,
        lithosphere,
        state,
        radius_km,
        params,
        dt_myr=dt_myr,
        root_contrast=contrast,
    )
    return state, forcing, diagnostics


__all__ = [
    "PlumeRiftingParameters",
    "PlumeRiftingState",
    "PlumeRiftingDiagnostics",
    "initialize_plume_rifting",
    "plume_rifting_fields",
    "diagnose_plume_rifting",
    "advance_plume_rifting",
]
