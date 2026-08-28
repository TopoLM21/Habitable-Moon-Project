"""Transient, zero-mean dynamic topography above mantle plumes.

The field in this module is convective support, not crustal thickness or a
surface load.  It is therefore added to the non-flexed topographic background
and never enters the continental-material ledger.  Removing the area-weighted
degree-zero component prevents the effective model from changing the planet's
mean radius when a broad plume swell develops.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import SphereMesh
from .plumes import MantlePlumeState

Array = np.ndarray


@dataclass(slots=True)
class PlumeDynamicTopographyParameters:
    enabled: bool = True
    flux_saturation: float = 0.75
    maximum_center_uplift_m: float = 1000.0
    response_time_myr: float = 8.0
    decay_time_myr: float = 20.0
    affected_uplift_threshold_m: float = 50.0
    maximum_absolute_anomaly_m: float = 1200.0
    remove_area_mean: bool = True


@dataclass(slots=True)
class PlumeDynamicTopographyState:
    time_myr: float
    target_dynamic_topography_m: Array
    realized_dynamic_topography_m: Array
    cumulative_positive_support_m_myr: Array


@dataclass(slots=True)
class PlumeDynamicTopographyDiagnostics:
    time_myr: float
    dt_myr: float
    enabled: bool
    area_mean_target_m: float
    area_mean_realized_m: float
    maximum_target_uplift_m: float
    maximum_realized_uplift_m: float
    minimum_realized_subsidence_m: float
    rms_realized_anomaly_m: float
    affected_surface_area_fraction: float
    plume_weighted_mean_uplift_m: float
    maximum_absolute_vertical_rate_m_per_myr: float
    displacement_volume_km3: float
    cumulative_mean_positive_support_m_myr: float
    cumulative_max_positive_support_m_myr: float


def _validate(params: PlumeDynamicTopographyParameters) -> None:
    if params.flux_saturation <= 0.0:
        raise ValueError("flux_saturation must be positive")
    if params.maximum_center_uplift_m < 0.0:
        raise ValueError("maximum_center_uplift_m must be non-negative")
    if params.response_time_myr <= 0.0 or params.decay_time_myr <= 0.0:
        raise ValueError("dynamic-topography response times must be positive")
    if params.affected_uplift_threshold_m < 0.0:
        raise ValueError("affected_uplift_threshold_m must be non-negative")
    if params.maximum_absolute_anomaly_m <= 0.0:
        raise ValueError("maximum_absolute_anomaly_m must be positive")


def _smoothstep(values: Array) -> Array:
    x = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _remove_area_mean(values: Array, areas: Array) -> Array:
    field = np.asarray(values, dtype=np.float64)
    total_area = max(float(np.sum(areas)), 1.0e-30)
    return field - float(np.sum(areas * field) / total_area)


def plume_dynamic_topography_target(
    mesh: SphereMesh,
    plume_state: MantlePlumeState,
    radius_km: float,
    params: PlumeDynamicTopographyParameters,
) -> Array:
    """Return instantaneous convective support relative to the global datum."""

    _validate(params)
    if plume_state.last_flux.shape != (mesh.cell_count,):
        raise ValueError("plume last_flux must have shape (cell_count,)")
    if not params.enabled:
        return np.zeros(mesh.cell_count, dtype=np.float64)
    flux = np.maximum(np.asarray(plume_state.last_flux, dtype=np.float64), 0.0)
    target = float(params.maximum_center_uplift_m) * _smoothstep(
        flux / float(params.flux_saturation)
    )
    if params.remove_area_mean:
        areas = mesh.physical_cell_areas_km2(float(radius_km))
        target = _remove_area_mean(target, areas)
    return np.clip(
        target,
        -float(params.maximum_absolute_anomaly_m),
        float(params.maximum_absolute_anomaly_m),
    )


def initialize_plume_dynamic_topography(
    mesh: SphereMesh, time_myr: float
) -> PlumeDynamicTopographyState:
    zero = np.zeros(mesh.cell_count, dtype=np.float64)
    return PlumeDynamicTopographyState(
        time_myr=float(time_myr),
        target_dynamic_topography_m=zero.copy(),
        realized_dynamic_topography_m=zero.copy(),
        cumulative_positive_support_m_myr=zero.copy(),
    )


def diagnose_plume_dynamic_topography(
    mesh: SphereMesh,
    plume_state: MantlePlumeState,
    state: PlumeDynamicTopographyState,
    radius_km: float,
    params: PlumeDynamicTopographyParameters,
    *,
    dt_myr: float = 0.0,
    vertical_rate_m_per_myr: Array | None = None,
) -> PlumeDynamicTopographyDiagnostics:
    areas = mesh.physical_cell_areas_km2(float(radius_km))
    total_area = max(float(np.sum(areas)), 1.0e-30)
    target = np.asarray(state.target_dynamic_topography_m, dtype=np.float64)
    realized = np.asarray(state.realized_dynamic_topography_m, dtype=np.float64)
    cumulative = np.asarray(
        state.cumulative_positive_support_m_myr, dtype=np.float64
    )
    flux = np.maximum(np.asarray(plume_state.last_flux, dtype=np.float64), 0.0)
    plume_weights = areas * flux
    plume_weight_sum = float(np.sum(plume_weights))
    rate = (
        np.zeros(mesh.cell_count, dtype=np.float64)
        if vertical_rate_m_per_myr is None
        else np.asarray(vertical_rate_m_per_myr, dtype=np.float64)
    )

    def area_mean(values: Array) -> float:
        return float(np.sum(areas * np.asarray(values, dtype=np.float64)) / total_area)

    affected = realized >= float(params.affected_uplift_threshold_m)
    return PlumeDynamicTopographyDiagnostics(
        time_myr=float(state.time_myr),
        dt_myr=float(dt_myr),
        enabled=bool(params.enabled),
        area_mean_target_m=area_mean(target),
        area_mean_realized_m=area_mean(realized),
        maximum_target_uplift_m=float(np.max(target)) if len(target) else 0.0,
        maximum_realized_uplift_m=float(np.max(realized)) if len(realized) else 0.0,
        minimum_realized_subsidence_m=(
            float(np.min(realized)) if len(realized) else 0.0
        ),
        rms_realized_anomaly_m=float(np.sqrt(area_mean(realized * realized))),
        affected_surface_area_fraction=float(np.sum(areas[affected]) / total_area),
        plume_weighted_mean_uplift_m=(
            float(np.sum(plume_weights * realized) / plume_weight_sum)
            if plume_weight_sum > 0.0
            else 0.0
        ),
        maximum_absolute_vertical_rate_m_per_myr=(
            float(np.max(np.abs(rate))) if len(rate) else 0.0
        ),
        displacement_volume_km3=float(np.sum(areas * realized) / 1000.0),
        cumulative_mean_positive_support_m_myr=area_mean(cumulative),
        cumulative_max_positive_support_m_myr=(
            float(np.max(cumulative)) if len(cumulative) else 0.0
        ),
    )


def advance_plume_dynamic_topography(
    mesh: SphereMesh,
    plume_state: MantlePlumeState,
    state: PlumeDynamicTopographyState,
    dt_myr: float,
    radius_km: float,
    params: PlumeDynamicTopographyParameters,
) -> tuple[PlumeDynamicTopographyState, PlumeDynamicTopographyDiagnostics]:
    """Advance the delayed, reversible response to current plume support."""

    if dt_myr <= 0.0:
        raise ValueError("dt_myr must be positive")
    _validate(params)
    n = mesh.cell_count
    for field in (
        state.target_dynamic_topography_m,
        state.realized_dynamic_topography_m,
        state.cumulative_positive_support_m_myr,
    ):
        if np.asarray(field).shape != (n,):
            raise ValueError("dynamic-topography fields must have shape (cell_count,)")

    areas = mesh.physical_cell_areas_km2(float(radius_km))
    target = plume_dynamic_topography_target(
        mesh, plume_state, radius_km, params
    )
    previous = np.asarray(state.realized_dynamic_topography_m, dtype=np.float64)
    growing = np.abs(target) > np.abs(previous)
    tau = np.where(
        growing,
        float(params.response_time_myr),
        float(params.decay_time_myr),
    )
    alpha = 1.0 - np.exp(-float(dt_myr) / tau)
    realized = previous + alpha * (target - previous)
    if params.remove_area_mean:
        realized = _remove_area_mean(realized, areas)
    realized = np.clip(
        realized,
        -float(params.maximum_absolute_anomaly_m),
        float(params.maximum_absolute_anomaly_m),
    )
    if params.remove_area_mean:
        realized = _remove_area_mean(realized, areas)
    rate = (realized - previous) / float(dt_myr)

    state.time_myr = float(state.time_myr) + float(dt_myr)
    state.target_dynamic_topography_m = target
    state.realized_dynamic_topography_m = realized
    state.cumulative_positive_support_m_myr = (
        np.asarray(state.cumulative_positive_support_m_myr, dtype=np.float64)
        + np.maximum(realized, 0.0) * float(dt_myr)
    )
    diagnostics = diagnose_plume_dynamic_topography(
        mesh,
        plume_state,
        state,
        radius_km,
        params,
        dt_myr=dt_myr,
        vertical_rate_m_per_myr=rate,
    )
    return state, diagnostics


__all__ = [
    "PlumeDynamicTopographyParameters",
    "PlumeDynamicTopographyState",
    "PlumeDynamicTopographyDiagnostics",
    "initialize_plume_dynamic_topography",
    "plume_dynamic_topography_target",
    "diagnose_plume_dynamic_topography",
    "advance_plume_dynamic_topography",
]
