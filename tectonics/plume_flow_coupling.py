"""v0.31 coupling between mobile plume sources and Eulerian mantle flow."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .plumes import MantlePlumeParameters, MantlePlumeState


@dataclass(slots=True)
class PlumeFlowCouplingParameters:
    enabled: bool = True
    mantle_flow_velocity_fraction: float = 0.35
    residual_drift_fraction: float = 0.30
    mantle_flow_sampling_radius_km: float = 550.0


@dataclass(slots=True)
class PlumeFlowCouplingDiagnostics:
    time_myr: float
    enabled: bool
    active_source_count: int
    mean_resolved_flow_speed_km_per_myr: float
    maximum_resolved_flow_speed_km_per_myr: float
    mean_residual_speed_km_per_myr: float
    maximum_residual_speed_km_per_myr: float
    mean_effective_source_speed_km_per_myr: float
    maximum_effective_source_speed_km_per_myr: float
    mean_effective_flow_alignment: float
    minimum_effective_flow_alignment: float
    mean_flow_velocity_fraction_of_effective_speed: float


def validate_plume_flow_coupling_parameters(
    params: PlumeFlowCouplingParameters,
) -> None:
    if not (0.0 <= params.mantle_flow_velocity_fraction <= 1.0):
        raise ValueError("mantle_flow_velocity_fraction must be in [0, 1]")
    if not (0.0 <= params.residual_drift_fraction <= 1.0):
        raise ValueError("residual_drift_fraction must be in [0, 1]")
    if params.mantle_flow_sampling_radius_km <= 0.0:
        raise ValueError("mantle_flow_sampling_radius_km must be positive")


def plume_parameters_with_flow_coupling(
    plume_params: MantlePlumeParameters,
    params: PlumeFlowCouplingParameters,
) -> MantlePlumeParameters:
    validate_plume_flow_coupling_parameters(params)
    return replace(
        plume_params,
        source_flow_coupling_enabled=bool(params.enabled),
        source_flow_velocity_fraction=float(params.mantle_flow_velocity_fraction),
        source_residual_drift_fraction=float(params.residual_drift_fraction),
        source_flow_sampling_radius_km=float(params.mantle_flow_sampling_radius_km),
    )


def diagnose_plume_flow_coupling(
    state: MantlePlumeState,
    radius_km: float,
    params: PlumeFlowCouplingParameters,
) -> PlumeFlowCouplingDiagnostics:
    validate_plume_flow_coupling_parameters(params)
    count = len(state.ages_myr)
    centers = np.asarray(state.centers_unit, dtype=np.float64)
    flow_omega = (
        np.zeros((count, 3), dtype=np.float64)
        if state.source_flow_omega_rad_per_myr is None
        else np.asarray(state.source_flow_omega_rad_per_myr, dtype=np.float64)
    )
    residual_axes = (
        np.zeros((count, 3), dtype=np.float64)
        if state.source_drift_axes_unit is None
        else np.asarray(state.source_drift_axes_unit, dtype=np.float64)
    )
    residual_raw_speed = (
        np.zeros(count, dtype=np.float64)
        if state.source_drift_speeds_km_per_myr is None
        else np.asarray(state.source_drift_speeds_km_per_myr, dtype=np.float64)
    )
    effective_axes = (
        np.zeros((count, 3), dtype=np.float64)
        if state.last_effective_source_axes_unit is None
        else np.asarray(state.last_effective_source_axes_unit, dtype=np.float64)
    )
    effective_speed = (
        np.zeros(count, dtype=np.float64)
        if state.last_effective_source_speeds_km_per_myr is None
        else np.asarray(state.last_effective_source_speeds_km_per_myr, dtype=np.float64)
    )
    if any(
        values.shape != shape
        for values, shape in (
            (flow_omega, (count, 3)),
            (residual_axes, (count, 3)),
            (residual_raw_speed, (count,)),
            (effective_axes, (count, 3)),
            (effective_speed, (count,)),
        )
    ):
        raise ValueError("flow-coupled plume arrays must match active source count")
    flow_velocity = np.cross(flow_omega, centers) * float(radius_km) * float(
        params.mantle_flow_velocity_fraction
    )
    residual_velocity = np.cross(residual_axes, centers) * residual_raw_speed[:, None] * float(
        params.residual_drift_fraction
    )
    effective_velocity = np.cross(effective_axes, centers) * effective_speed[:, None]
    flow_speed = np.linalg.norm(flow_velocity, axis=1)
    residual_speed = np.linalg.norm(residual_velocity, axis=1)
    effective_norm = np.linalg.norm(effective_velocity, axis=1)
    alignment = np.zeros(count, dtype=np.float64)
    valid = (flow_speed > 1.0e-12) & (effective_norm > 1.0e-12)
    if np.any(valid):
        alignment[valid] = np.sum(
            flow_velocity[valid] * effective_velocity[valid], axis=1
        ) / (flow_speed[valid] * effective_norm[valid])
    fraction = np.divide(
        flow_speed,
        effective_norm,
        out=np.zeros_like(flow_speed),
        where=effective_norm > 1.0e-12,
    )

    def mean(values) -> float:
        return float(np.mean(values)) if len(values) else 0.0

    def maximum(values) -> float:
        return float(np.max(values)) if len(values) else 0.0

    return PlumeFlowCouplingDiagnostics(
        time_myr=float(state.time_myr),
        enabled=bool(params.enabled),
        active_source_count=int(count),
        mean_resolved_flow_speed_km_per_myr=mean(flow_speed),
        maximum_resolved_flow_speed_km_per_myr=maximum(flow_speed),
        mean_residual_speed_km_per_myr=mean(residual_speed),
        maximum_residual_speed_km_per_myr=maximum(residual_speed),
        mean_effective_source_speed_km_per_myr=mean(effective_norm),
        maximum_effective_source_speed_km_per_myr=maximum(effective_norm),
        mean_effective_flow_alignment=mean(alignment[valid]) if np.any(valid) else 0.0,
        minimum_effective_flow_alignment=(
            float(np.min(alignment[valid])) if np.any(valid) else 0.0
        ),
        mean_flow_velocity_fraction_of_effective_speed=mean(fraction),
    )


__all__ = [
    "PlumeFlowCouplingParameters",
    "PlumeFlowCouplingDiagnostics",
    "validate_plume_flow_coupling_parameters",
    "plume_parameters_with_flow_coupling",
    "diagnose_plume_flow_coupling",
]
