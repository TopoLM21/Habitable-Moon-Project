"""Mantle-plume forcing, head/tail separation and mobile deep sources.

This is an effective surface projection of deep plumes.  Before v0.30 the
plume field is Eulerian and mantle-fixed; v0.30 can opt individual conduits into
slow deterministic drift, while v0.31 can guide most of that motion with the
checkpointed fixed-grid mantle-flow memory. Plates and their material memory
move independently across the sources. While continental lithosphere overlies an active plume, the
model applies three coupled effects supported by thermomechanical studies:

* thermal rejuvenation of effective continental-lithosphere age;
* melt/fluid refertilization of the depleted mantle-root proxy;
* basal erosion of the mechanical mantle-lithosphere root.

The implementation is deliberately not a mantle-convection or petrological
phase-equilibrium solver.  It provides a deterministic, checkpointable forcing
layer with explicit diagnostics so its influence can be tested against the
otherwise unchanged v0.24 integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cratons import CratonParameters, craton_strength_from_memory
from .lithosphere import LithosphereState, continental_material_fields
from .mesh import SphereMesh

Array = np.ndarray


@dataclass(slots=True)
class MantlePlumeParameters:
    """Effective plume-population and lithosphere-coupling parameters."""

    enabled: bool = True
    lithosphere_weakening_enabled: bool = True
    seed: int = 20261055
    initial_plume_count: int = 3
    mean_birth_interval_myr: float = 160.0
    birth_interval_jitter_fraction: float = 0.25
    minimum_lifetime_myr: float = 160.0
    maximum_lifetime_myr: float = 240.0
    head_radius_km: float = 650.0
    head_radius_variation_fraction: float = 0.20
    minimum_peak_flux: float = 0.75
    maximum_peak_flux: float = 1.00
    rise_time_fraction: float = 0.22
    decay_time_fraction: float = 0.38
    affected_flux_threshold: float = 0.10
    age_rejuvenation_rate_per_myr: float = 0.0040
    refertilization_rate_per_myr: float = 0.0035
    root_erosion_km_per_myr: float = 0.35
    minimum_continental_root_thickness_km: float = 60.0
    continental_fraction_epsilon: float = 1.0e-9
    # v0.29 opt-in plume-head/plume-tail separation.  Older runners retain the
    # original single broad Gaussian when this switch is false.
    head_tail_separation_enabled: bool = False
    head_duration_fraction: float = 0.18
    head_rise_fraction: float = 0.25
    head_decay_fraction: float = 0.50
    tail_radius_fraction: float = 0.23
    tail_flux_fraction: float = 0.55
    tail_rise_time_fraction: float = 0.08
    tail_decay_time_fraction: float = 0.15
    component_flux_area_normalization_enabled: bool = False
    component_flux_area_normalization_exponent: float = 1.0
    # v0.30 opt-in motion of the deep source. Each conduit follows piecewise
    # great-circle arcs; its tangent direction and speed change only at the
    # configured persistence interval. Older runners keep exactly fixed sources.
    source_drift_enabled: bool = False
    minimum_source_drift_km_per_myr: float = 8.0
    maximum_source_drift_km_per_myr: float = 30.0
    source_drift_persistence_myr: float = 80.0
    source_drift_direction_memory: float = 0.65
    # v0.31 optional coupling to the fixed-grid mantle-flow memory. The v0.30
    # stochastic arc becomes a smaller unresolved residual when enabled.
    source_flow_coupling_enabled: bool = False
    source_flow_velocity_fraction: float = 0.35
    source_residual_drift_fraction: float = 0.30
    source_flow_sampling_radius_km: float = 550.0


@dataclass(slots=True)
class MantlePlumeState:
    """Checkpointable plume population and fixed-grid exposure diagnostics."""

    time_myr: float
    centers_unit: Array
    ages_myr: Array
    lifetimes_myr: Array
    head_radii_km: Array
    peak_fluxes: Array
    next_plume_id: int
    next_birth_time_myr: float
    last_flux: Array
    cumulative_exposure_myr: Array
    cumulative_root_erosion_km: Array
    last_head_flux: Array | None = None
    last_tail_flux: Array | None = None
    plume_ids: Array | None = None
    source_drift_axes_unit: Array | None = None
    source_drift_speeds_km_per_myr: Array | None = None
    source_drift_segment_index: Array | None = None
    cumulative_source_distance_km: Array | None = None
    cumulative_source_bend_deg: Array | None = None
    source_flow_omega_rad_per_myr: Array | None = None
    last_effective_source_axes_unit: Array | None = None
    last_effective_source_speeds_km_per_myr: Array | None = None
    population_source_distance_km: float = 0.0
    population_source_bend_deg: float = 0.0


@dataclass(slots=True)
class MantlePlumeDiagnostics:
    time_myr: float
    dt_myr: float
    active_plume_count: int
    mean_surface_flux: float
    max_surface_flux: float
    affected_surface_area_fraction: float
    exposed_continental_material_fraction: float
    mean_continental_flux: float
    mean_continental_age_loss_myr: float
    mean_continental_depletion_loss: float
    mean_craton_strength_loss: float
    mean_root_erosion_this_step_km: float
    max_root_erosion_this_step_km: float
    cumulative_mean_surface_exposure_myr: float
    cumulative_max_surface_exposure_myr: float
    mean_head_flux: float = 0.0
    max_head_flux: float = 0.0
    mean_tail_flux: float = 0.0
    max_tail_flux: float = 0.0
    source_drift_enabled: bool = False
    mean_source_drift_speed_km_per_myr: float = 0.0
    max_source_drift_speed_km_per_myr: float = 0.0
    active_source_path_length_km: float = 0.0
    maximum_active_source_path_length_km: float = 0.0
    active_source_bend_angle_deg: float = 0.0
    maximum_active_source_bend_angle_deg: float = 0.0
    population_source_path_length_km: float = 0.0
    population_source_bend_angle_deg: float = 0.0


def _validate_parameters(params: MantlePlumeParameters) -> None:
    if params.initial_plume_count < 0:
        raise ValueError("initial_plume_count must be non-negative")
    if params.mean_birth_interval_myr <= 0.0:
        raise ValueError("mean_birth_interval_myr must be positive")
    if not (0.0 <= params.birth_interval_jitter_fraction < 1.0):
        raise ValueError("birth_interval_jitter_fraction must be in [0, 1)")
    if not (0.0 < params.minimum_lifetime_myr <= params.maximum_lifetime_myr):
        raise ValueError("plume lifetimes must be positive and ordered")
    if params.head_radius_km <= 0.0:
        raise ValueError("head_radius_km must be positive")
    if not (0.0 <= params.head_radius_variation_fraction < 1.0):
        raise ValueError("head_radius_variation_fraction must be in [0, 1)")
    if not (0.0 <= params.minimum_peak_flux <= params.maximum_peak_flux):
        raise ValueError("peak flux bounds must be non-negative and ordered")
    if not (0.0 < params.rise_time_fraction < 1.0):
        raise ValueError("rise_time_fraction must be in (0, 1)")
    if not (0.0 < params.decay_time_fraction < 1.0):
        raise ValueError("decay_time_fraction must be in (0, 1)")
    if params.rise_time_fraction + params.decay_time_fraction > 1.0:
        raise ValueError("rise and decay fractions must sum to at most 1")
    if params.minimum_continental_root_thickness_km < 0.0:
        raise ValueError("minimum_continental_root_thickness_km must be non-negative")
    if not (0.0 < params.head_duration_fraction < 1.0):
        raise ValueError("head_duration_fraction must be in (0, 1)")
    if not (0.0 < params.head_rise_fraction < 1.0):
        raise ValueError("head_rise_fraction must be in (0, 1)")
    if not (0.0 < params.head_decay_fraction < 1.0):
        raise ValueError("head_decay_fraction must be in (0, 1)")
    if params.head_rise_fraction + params.head_decay_fraction > 1.0:
        raise ValueError("head rise and decay fractions must sum to at most 1")
    if not (0.0 < params.tail_radius_fraction < 1.0):
        raise ValueError("tail_radius_fraction must be in (0, 1)")
    if not (0.0 <= params.tail_flux_fraction <= 1.0):
        raise ValueError("tail_flux_fraction must be in [0, 1]")
    if not (0.0 < params.tail_rise_time_fraction < 1.0):
        raise ValueError("tail_rise_time_fraction must be in (0, 1)")
    if not (0.0 < params.tail_decay_time_fraction < 1.0):
        raise ValueError("tail_decay_time_fraction must be in (0, 1)")
    if params.component_flux_area_normalization_exponent <= 0.0:
        raise ValueError("component flux area-normalization exponent must be positive")
    if not (
        0.0
        <= params.minimum_source_drift_km_per_myr
        <= params.maximum_source_drift_km_per_myr
    ):
        raise ValueError("source drift speed bounds must be non-negative and ordered")
    if params.source_drift_persistence_myr <= 0.0:
        raise ValueError("source_drift_persistence_myr must be positive")
    if not (0.0 <= params.source_drift_direction_memory <= 1.0):
        raise ValueError("source_drift_direction_memory must be in [0, 1]")
    if not (0.0 <= params.source_flow_velocity_fraction <= 1.0):
        raise ValueError("source_flow_velocity_fraction must be in [0, 1]")
    if not (0.0 <= params.source_residual_drift_fraction <= 1.0):
        raise ValueError("source_residual_drift_fraction must be in [0, 1]")
    if params.source_flow_sampling_radius_km <= 0.0:
        raise ValueError("source_flow_sampling_radius_km must be positive")


def _rng(params: MantlePlumeParameters, plume_id: int, stream: int = 0) -> np.random.Generator:
    # SeedSequence makes each plume/event independent of call ordering, which
    # is important for exact checkpoint/resume continuation.
    return np.random.default_rng(
        np.random.SeedSequence([int(params.seed), int(plume_id), int(stream)])
    )


def _sample_plume(params: MantlePlumeParameters, plume_id: int) -> tuple[Array, float, float, float]:
    rng = _rng(params, plume_id, 0)
    center = rng.normal(size=3)
    center /= max(float(np.linalg.norm(center)), 1.0e-30)
    lifetime = float(rng.uniform(params.minimum_lifetime_myr, params.maximum_lifetime_myr))
    radius_scale = float(
        rng.uniform(
            1.0 - params.head_radius_variation_fraction,
            1.0 + params.head_radius_variation_fraction,
        )
    )
    peak = float(rng.uniform(params.minimum_peak_flux, params.maximum_peak_flux))
    return center, lifetime, float(params.head_radius_km) * radius_scale, peak


def _birth_interval(params: MantlePlumeParameters, plume_id: int) -> float:
    rng = _rng(params, plume_id, 1)
    jitter = float(
        rng.uniform(
            1.0 - params.birth_interval_jitter_fraction,
            1.0 + params.birth_interval_jitter_fraction,
        )
    )
    return float(params.mean_birth_interval_myr) * jitter


def _unit_tangent(center: Array, vector: Array) -> Array:
    """Project *vector* onto the source tangent plane with a safe fallback."""

    c = np.asarray(center, dtype=np.float64)
    tangent = np.asarray(vector, dtype=np.float64) - float(np.dot(vector, c)) * c
    norm = float(np.linalg.norm(tangent))
    if norm <= 1.0e-14:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(c[0])) > 0.8:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tangent = np.cross(reference, c)
        norm = float(np.linalg.norm(tangent))
    return tangent / max(norm, 1.0e-30)


def _sample_source_drift(
    params: MantlePlumeParameters,
    plume_id: int,
    segment_index: int,
    center: Array,
    previous_axis: Array | None = None,
) -> tuple[Array, float]:
    """Sample a deterministic surface-tangent direction and linear speed."""

    if not params.source_drift_enabled:
        return np.zeros(3, dtype=np.float64), 0.0
    rng = _rng(params, plume_id, 1000 + int(segment_index))
    random_tangent = _unit_tangent(center, rng.normal(size=3))
    if previous_axis is not None and float(np.linalg.norm(previous_axis)) > 0.0:
        previous_tangent = _unit_tangent(
            center, np.cross(np.asarray(previous_axis, dtype=np.float64), center)
        )
        memory = float(params.source_drift_direction_memory)
        tangent = _unit_tangent(
            center,
            memory * previous_tangent
            + np.sqrt(max(1.0 - memory * memory, 0.0)) * random_tangent,
        )
    else:
        tangent = random_tangent
    # cross(axis, center) points along the requested tangent direction.
    axis = np.cross(center, tangent)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-30)
    speed = float(
        rng.uniform(
            params.minimum_source_drift_km_per_myr,
            params.maximum_source_drift_km_per_myr,
        )
    )
    return axis, speed


def _ensure_source_drift_state(
    state: MantlePlumeState, params: MantlePlumeParameters
) -> None:
    """Upgrade an older in-memory/checkpoint plume population in place."""

    count = len(state.ages_myr)
    if state.plume_ids is None or np.asarray(state.plume_ids).shape != (count,):
        first_id = max(int(state.next_plume_id) - count, 0)
        state.plume_ids = np.arange(first_id, first_id + count, dtype=np.int64)
    else:
        state.plume_ids = np.asarray(state.plume_ids, dtype=np.int64)
    if (
        state.source_drift_axes_unit is None
        or np.asarray(state.source_drift_axes_unit).shape != (count, 3)
    ):
        state.source_drift_axes_unit = np.zeros((count, 3), dtype=np.float64)
    else:
        state.source_drift_axes_unit = np.asarray(
            state.source_drift_axes_unit, dtype=np.float64
        )
    if (
        state.source_drift_speeds_km_per_myr is None
        or np.asarray(state.source_drift_speeds_km_per_myr).shape != (count,)
    ):
        state.source_drift_speeds_km_per_myr = np.zeros(count, dtype=np.float64)
    else:
        state.source_drift_speeds_km_per_myr = np.asarray(
            state.source_drift_speeds_km_per_myr, dtype=np.float64
        )
    if (
        state.source_drift_segment_index is None
        or np.asarray(state.source_drift_segment_index).shape != (count,)
    ):
        state.source_drift_segment_index = np.floor(
            np.asarray(state.ages_myr, dtype=np.float64)
            / float(params.source_drift_persistence_myr)
        ).astype(np.int32)
    else:
        state.source_drift_segment_index = np.asarray(
            state.source_drift_segment_index, dtype=np.int32
        )
    for i in range(count):
        if (
            params.source_drift_enabled
            and float(np.linalg.norm(state.source_drift_axes_unit[i])) <= 0.0
        ):
            axis, speed = _sample_source_drift(
                params,
                int(state.plume_ids[i]),
                int(state.source_drift_segment_index[i]),
                state.centers_unit[i],
            )
            state.source_drift_axes_unit[i] = axis
            state.source_drift_speeds_km_per_myr[i] = speed
    if (
        state.cumulative_source_distance_km is None
        or np.asarray(state.cumulative_source_distance_km).shape != (count,)
    ):
        state.cumulative_source_distance_km = np.zeros(count, dtype=np.float64)
    else:
        state.cumulative_source_distance_km = np.asarray(
            state.cumulative_source_distance_km, dtype=np.float64
        )
    if (
        state.cumulative_source_bend_deg is None
        or np.asarray(state.cumulative_source_bend_deg).shape != (count,)
    ):
        state.cumulative_source_bend_deg = np.zeros(count, dtype=np.float64)
    else:
        state.cumulative_source_bend_deg = np.asarray(
            state.cumulative_source_bend_deg, dtype=np.float64
        )
    if (
        state.source_flow_omega_rad_per_myr is None
        or np.asarray(state.source_flow_omega_rad_per_myr).shape != (count, 3)
    ):
        state.source_flow_omega_rad_per_myr = np.zeros((count, 3), dtype=np.float64)
    else:
        state.source_flow_omega_rad_per_myr = np.asarray(
            state.source_flow_omega_rad_per_myr, dtype=np.float64
        )
    if (
        state.last_effective_source_axes_unit is None
        or np.asarray(state.last_effective_source_axes_unit).shape != (count, 3)
    ):
        state.last_effective_source_axes_unit = np.asarray(
            state.source_drift_axes_unit, dtype=np.float64
        ).copy()
    else:
        state.last_effective_source_axes_unit = np.asarray(
            state.last_effective_source_axes_unit, dtype=np.float64
        )
    if (
        state.last_effective_source_speeds_km_per_myr is None
        or np.asarray(state.last_effective_source_speeds_km_per_myr).shape != (count,)
    ):
        state.last_effective_source_speeds_km_per_myr = np.asarray(
            state.source_drift_speeds_km_per_myr, dtype=np.float64
        ).copy()
    else:
        state.last_effective_source_speeds_km_per_myr = np.asarray(
            state.last_effective_source_speeds_km_per_myr, dtype=np.float64
        )


def _effective_source_axis_speed(
    state: MantlePlumeState,
    params: MantlePlumeParameters,
    index: int,
    radius_km: float,
) -> tuple[Array, float]:
    """Combine the resolved mantle-flow velocity and unresolved residual."""

    center = np.asarray(state.centers_unit[index], dtype=np.float64)
    residual = (
        np.cross(state.source_drift_axes_unit[index], center)
        * float(state.source_drift_speeds_km_per_myr[index])
    )
    if params.source_flow_coupling_enabled:
        residual *= float(params.source_residual_drift_fraction)
        flow = (
            np.cross(state.source_flow_omega_rad_per_myr[index], center)
            * float(radius_km)
            * float(params.source_flow_velocity_fraction)
        )
        velocity = flow + residual
    else:
        velocity = residual
    speed = float(np.linalg.norm(velocity))
    if speed <= 1.0e-14:
        return np.zeros(3, dtype=np.float64), 0.0
    tangent = velocity / speed
    axis = np.cross(center, tangent)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-30)
    return axis, speed


def update_plume_source_flow(
    mesh: SphereMesh,
    state: MantlePlumeState,
    mantle_omega_field_rad_per_myr: Array,
    radius_km: float,
    params: MantlePlumeParameters,
    *,
    initialize_effective_velocity: bool = False,
) -> MantlePlumeState:
    """Sample fixed-grid mantle flow at arbitrary active source positions.

    Area-weighted Gaussian interpolation avoids making the source velocity jump
    as it crosses a mesh-cell boundary. The sampled angular-velocity vectors are
    part of the plume checkpoint, while the full Eulerian field remains in the
    existing mantle checkpoint.
    """

    _ensure_source_drift_state(state, params)
    field = np.asarray(mantle_omega_field_rad_per_myr, dtype=np.float64)
    if field.shape != (mesh.cell_count, 3):
        raise ValueError("mantle omega field must have shape (cell_count, 3)")
    count = len(state.ages_myr)
    if count == 0:
        return state
    dots = np.clip(
        np.asarray(mesh.centroids, dtype=np.float64)
        @ np.asarray(state.centers_unit, dtype=np.float64).T,
        -1.0,
        1.0,
    )
    distance_km = float(radius_km) * np.arccos(dots)
    sigma = float(params.source_flow_sampling_radius_km)
    weights = (
        mesh.physical_cell_areas_km2(float(radius_km))[:, None]
        * np.exp(-0.5 * np.square(distance_km / sigma))
    )
    state.source_flow_omega_rad_per_myr = (
        weights.T @ field
    ) / np.maximum(np.sum(weights, axis=0)[:, None], 1.0e-30)
    if initialize_effective_velocity:
        for i in range(count):
            axis, speed = _effective_source_axis_speed(state, params, i, radius_km)
            state.last_effective_source_axes_unit[i] = axis
            state.last_effective_source_speeds_km_per_myr[i] = speed
    return state


def _rotate_about_axis(point: Array, axis: Array, angle_rad: float) -> Array:
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0 or angle_rad == 0.0:
        return np.asarray(point, dtype=np.float64).copy()
    a = np.asarray(axis, dtype=np.float64) / axis_norm
    p = np.asarray(point, dtype=np.float64)
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    rotated = (
        cosine * p
        + sine * np.cross(a, p)
        + (1.0 - cosine) * float(np.dot(a, p)) * a
    )
    return rotated / max(float(np.linalg.norm(rotated)), 1.0e-30)


def _advance_source_drift(
    state: MantlePlumeState,
    params: MantlePlumeParameters,
    start_ages_myr: Array,
    radius_km: float,
) -> None:
    """Move active source projections and update path/bend ledgers."""

    _ensure_source_drift_state(state, params)
    if not params.source_drift_enabled:
        return
    persistence = float(params.source_drift_persistence_myr)
    total_distance = 0.0
    total_bend = 0.0
    end_ages = np.minimum(
        np.asarray(state.ages_myr, dtype=np.float64),
        np.asarray(state.lifetimes_myr, dtype=np.float64),
    )
    starts = np.asarray(start_ages_myr, dtype=np.float64)
    for i in range(len(end_ages)):
        t = float(starts[i])
        end = float(end_ages[i])
        while t < end - 1.0e-12:
            segment = int(state.source_drift_segment_index[i])
            boundary = float(segment + 1) * persistence
            stop = min(end, boundary)
            duration = max(stop - t, 0.0)
            axis, speed = _effective_source_axis_speed(
                state, params, i, radius_km
            )
            old_effective_axis = state.last_effective_source_axes_unit[i]
            if (
                float(np.linalg.norm(old_effective_axis)) > 0.0
                and float(np.linalg.norm(axis)) > 0.0
            ):
                old_tangent = _unit_tangent(
                    state.centers_unit[i],
                    np.cross(old_effective_axis, state.centers_unit[i]),
                )
                new_tangent = _unit_tangent(
                    state.centers_unit[i], np.cross(axis, state.centers_unit[i])
                )
                bend = float(
                    np.rad2deg(
                        np.arccos(
                            np.clip(np.dot(old_tangent, new_tangent), -1.0, 1.0)
                        )
                    )
                )
                if bend > 1.0e-10:
                    state.cumulative_source_bend_deg[i] += bend
                    total_bend += bend
            state.last_effective_source_axes_unit[i] = axis
            state.last_effective_source_speeds_km_per_myr[i] = speed
            distance = speed * duration
            state.centers_unit[i] = _rotate_about_axis(
                state.centers_unit[i],
                axis,
                distance / max(float(radius_km), 1.0e-30),
            )
            state.cumulative_source_distance_km[i] += distance
            total_distance += distance
            t = stop
            if t >= boundary - 1.0e-12:
                old_axis = state.source_drift_axes_unit[i].copy()
                new_segment = segment + 1
                new_axis, new_speed = _sample_source_drift(
                    params,
                    int(state.plume_ids[i]),
                    new_segment,
                    state.centers_unit[i],
                    previous_axis=old_axis,
                )
                state.source_drift_axes_unit[i] = new_axis
                state.source_drift_speeds_km_per_myr[i] = new_speed
                state.source_drift_segment_index[i] = new_segment
    state.population_source_distance_km += total_distance
    state.population_source_bend_deg += total_bend


def _append_plume(state: MantlePlumeState, params: MantlePlumeParameters, plume_id: int, age_myr: float) -> None:
    center, lifetime, radius, peak = _sample_plume(params, plume_id)
    axis, speed = _sample_source_drift(params, plume_id, 0, center)
    state.centers_unit = np.vstack((state.centers_unit, center.reshape(1, 3)))
    state.ages_myr = np.append(state.ages_myr, float(age_myr))
    state.lifetimes_myr = np.append(state.lifetimes_myr, lifetime)
    state.head_radii_km = np.append(state.head_radii_km, radius)
    state.peak_fluxes = np.append(state.peak_fluxes, peak)
    state.plume_ids = np.append(state.plume_ids, int(plume_id)).astype(np.int64)
    state.source_drift_axes_unit = np.vstack(
        (state.source_drift_axes_unit, axis.reshape(1, 3))
    )
    state.source_drift_speeds_km_per_myr = np.append(
        state.source_drift_speeds_km_per_myr, speed
    )
    state.source_drift_segment_index = np.append(
        state.source_drift_segment_index, 0
    ).astype(np.int32)
    state.cumulative_source_distance_km = np.append(
        state.cumulative_source_distance_km, 0.0
    )
    state.cumulative_source_bend_deg = np.append(
        state.cumulative_source_bend_deg, 0.0
    )
    state.source_flow_omega_rad_per_myr = np.vstack(
        (state.source_flow_omega_rad_per_myr, np.zeros((1, 3), dtype=np.float64))
    )
    state.last_effective_source_axes_unit = np.vstack(
        (state.last_effective_source_axes_unit, axis.reshape(1, 3))
    )
    state.last_effective_source_speeds_km_per_myr = np.append(
        state.last_effective_source_speeds_km_per_myr, speed
    )


def initialize_mantle_plumes(
    mesh: SphereMesh,
    time_myr: float,
    params: MantlePlumeParameters | None = None,
) -> MantlePlumeState:
    """Create a deterministic mantle-fixed plume population."""

    p = MantlePlumeParameters() if params is None else params
    _validate_parameters(p)
    n = mesh.cell_count
    state = MantlePlumeState(
        time_myr=float(time_myr),
        centers_unit=np.empty((0, 3), dtype=np.float64),
        ages_myr=np.empty(0, dtype=np.float64),
        lifetimes_myr=np.empty(0, dtype=np.float64),
        head_radii_km=np.empty(0, dtype=np.float64),
        peak_fluxes=np.empty(0, dtype=np.float64),
        next_plume_id=0,
        next_birth_time_myr=float(time_myr),
        last_flux=np.zeros(n, dtype=np.float64),
        cumulative_exposure_myr=np.zeros(n, dtype=np.float64),
        cumulative_root_erosion_km=np.zeros(n, dtype=np.float64),
        last_head_flux=np.zeros(n, dtype=np.float64),
        last_tail_flux=np.zeros(n, dtype=np.float64),
        plume_ids=np.empty(0, dtype=np.int64),
        source_drift_axes_unit=np.empty((0, 3), dtype=np.float64),
        source_drift_speeds_km_per_myr=np.empty(0, dtype=np.float64),
        source_drift_segment_index=np.empty(0, dtype=np.int32),
        cumulative_source_distance_km=np.empty(0, dtype=np.float64),
        cumulative_source_bend_deg=np.empty(0, dtype=np.float64),
        source_flow_omega_rad_per_myr=np.empty((0, 3), dtype=np.float64),
        last_effective_source_axes_unit=np.empty((0, 3), dtype=np.float64),
        last_effective_source_speeds_km_per_myr=np.empty(0, dtype=np.float64),
    )
    if not p.enabled:
        state.next_birth_time_myr = float("inf")
        return state
    for plume_id in range(int(p.initial_plume_count)):
        _append_plume(state, p, plume_id, 0.0)
    state.next_plume_id = int(p.initial_plume_count)
    state.next_birth_time_myr = float(time_myr) + _birth_interval(p, state.next_plume_id)
    return state


def _smoothstep(value: Array) -> Array:
    x = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def plume_component_flux_fields(
    mesh: SphereMesh,
    state: MantlePlumeState,
    radius_km: float,
    params: MantlePlumeParameters,
) -> tuple[Array, Array]:
    """Return broad-head and narrow-tail flux on mantle-fixed cells.

    With head/tail separation disabled, the first array is exactly the legacy
    v0.25 broad-plume field and the second is zero.  The v0.29 branch instead
    gives each plume a short initial head pulse followed by a persistent,
    narrower conduit.  Keeping the centers Eulerian lets plate transport make
    the corresponding material track and age progression.
    """

    if not params.enabled or len(state.ages_myr) == 0:
        zero = np.zeros(mesh.cell_count, dtype=np.float64)
        return zero.copy(), zero.copy()
    age = np.asarray(state.ages_myr, dtype=np.float64)
    life = np.maximum(np.asarray(state.lifetimes_myr, dtype=np.float64), 1.0e-30)
    alive = (age >= 0.0) & (age < life)
    if not np.any(alive):
        zero = np.zeros(mesh.cell_count, dtype=np.float64)
        return zero.copy(), zero.copy()
    age = age[alive]
    life = life[alive]
    centers = np.asarray(state.centers_unit, dtype=np.float64)[alive]
    dots = np.clip(np.asarray(mesh.centroids, dtype=np.float64) @ centers.T, -1.0, 1.0)
    distance_km = float(radius_km) * np.arccos(dots)
    head_sigma = np.maximum(
        np.asarray(state.head_radii_km, dtype=np.float64)[alive], 1.0e-30
    )
    peak = np.asarray(state.peak_fluxes, dtype=np.float64)[alive]

    if not params.head_tail_separation_enabled:
        rise = np.maximum(float(params.rise_time_fraction) * life, 1.0e-30)
        decay = np.maximum(float(params.decay_time_fraction) * life, 1.0e-30)
        envelope = _smoothstep(age / rise) * _smoothstep((life - age) / decay)
        head = np.sum(
            np.exp(-0.5 * (distance_km / head_sigma[None, :]) ** 2)
            * (peak * envelope)[None, :],
            axis=1,
        )
        return np.clip(head, 0.0, 1.5), np.zeros(mesh.cell_count, dtype=np.float64)

    head_duration = np.maximum(
        float(params.head_duration_fraction) * life, 1.0e-30
    )
    head_rise = np.maximum(
        float(params.head_rise_fraction) * head_duration, 1.0e-30
    )
    head_decay = np.maximum(
        float(params.head_decay_fraction) * head_duration, 1.0e-30
    )
    head_envelope = _smoothstep(age / head_rise) * _smoothstep(
        (head_duration - age) / head_decay
    )
    head_kernel = np.exp(-0.5 * (distance_km / head_sigma[None, :]) ** 2)

    tail_rise = np.maximum(
        float(params.tail_rise_time_fraction) * life, 1.0e-30
    )
    tail_decay = np.maximum(
        float(params.tail_decay_time_fraction) * life, 1.0e-30
    )
    tail_envelope = _smoothstep(age / tail_rise) * _smoothstep(
        (life - age) / tail_decay
    )
    tail_sigma = np.maximum(
        float(params.tail_radius_fraction) * head_sigma, 1.0e-30
    )
    tail_kernel = np.exp(-0.5 * (distance_km / tail_sigma[None, :]) ** 2)
    if params.component_flux_area_normalization_enabled:
        # A narrow tail may be smaller than a coarse mesh cell.  Treat sampled
        # values as cell-average quadrature weights. Preserve the continuous
        # small-angle Gaussian integral after the configured nonlinear
        # productivity transform: integral(kernel**q)=2*pi*sigma**2/q. q=1
        # is the v0.29 raw-flux normalization; v0.30 uses the melt-productivity
        # exponent so mobile narrow tails do not regain centroid-alignment bias.
        areas = mesh.physical_cell_areas_km2(float(radius_km))
        exponent = float(params.component_flux_area_normalization_exponent)
        for kernel, sigma in ((head_kernel, head_sigma), (tail_kernel, tail_sigma)):
            sampled = np.sum(areas[:, None] * np.power(kernel, exponent), axis=0)
            target = 2.0 * np.pi * np.square(sigma) / exponent
            kernel *= (
                target / np.maximum(sampled, 1.0e-30)
            )[None, :] ** (1.0 / exponent)
    head = np.sum(head_kernel * (peak * head_envelope)[None, :], axis=1)
    tail = np.sum(
        tail_kernel
        * (peak * float(params.tail_flux_fraction) * tail_envelope)[None, :],
        axis=1,
    )
    return np.clip(head, 0.0, 1.5), np.clip(tail, 0.0, 1.5)


def plume_flux_field(
    mesh: SphereMesh,
    state: MantlePlumeState,
    radius_km: float,
    params: MantlePlumeParameters,
) -> Array:
    """Return normalized superposed total plume flux on fixed surface cells."""

    head, tail = plume_component_flux_fields(mesh, state, radius_km, params)
    return np.clip(head + tail, 0.0, 1.5)


def diagnose_mantle_plumes(
    mesh: SphereMesh,
    lithosphere: LithosphereState,
    plume_state: MantlePlumeState,
    radius_km: float,
    params: MantlePlumeParameters,
    *,
    dt_myr: float = 0.0,
    age_loss: Array | None = None,
    depletion_loss: Array | None = None,
    strength_loss: Array | None = None,
    root_erosion: Array | None = None,
) -> MantlePlumeDiagnostics:
    """Summarize plume activity and this step's material response."""

    areas = mesh.physical_cell_areas_km2(float(radius_km))
    fraction, _ = continental_material_fields(lithosphere, areas)
    weights = areas * fraction
    total_area = max(float(np.sum(areas)), 1.0e-30)
    total_cont = max(float(np.sum(weights)), 1.0e-30)
    flux = np.asarray(plume_state.last_flux, dtype=np.float64)
    head_flux = (
        np.zeros_like(flux)
        if plume_state.last_head_flux is None
        else np.asarray(plume_state.last_head_flux, dtype=np.float64)
    )
    tail_flux = (
        np.zeros_like(flux)
        if plume_state.last_tail_flux is None
        else np.asarray(plume_state.last_tail_flux, dtype=np.float64)
    )
    affected = flux >= float(params.affected_flux_threshold)

    def continental_mean(values: Array | None) -> float:
        if values is None:
            return 0.0
        return float(np.sum(weights * np.asarray(values, dtype=np.float64)) / total_cont)

    return MantlePlumeDiagnostics(
        time_myr=float(plume_state.time_myr),
        dt_myr=float(dt_myr),
        active_plume_count=int(len(plume_state.ages_myr)),
        mean_surface_flux=float(np.sum(areas * flux) / total_area),
        max_surface_flux=float(np.max(flux)) if len(flux) else 0.0,
        affected_surface_area_fraction=float(np.sum(areas[affected]) / total_area),
        exposed_continental_material_fraction=float(np.sum(weights[affected]) / total_cont),
        mean_continental_flux=continental_mean(flux),
        mean_continental_age_loss_myr=continental_mean(age_loss),
        mean_continental_depletion_loss=continental_mean(depletion_loss),
        mean_craton_strength_loss=continental_mean(strength_loss),
        mean_root_erosion_this_step_km=continental_mean(root_erosion),
        max_root_erosion_this_step_km=(
            float(np.max(np.asarray(root_erosion, dtype=np.float64)))
            if root_erosion is not None and len(root_erosion)
            else 0.0
        ),
        cumulative_mean_surface_exposure_myr=float(
            np.sum(areas * plume_state.cumulative_exposure_myr) / total_area
        ),
        cumulative_max_surface_exposure_myr=float(
            np.max(plume_state.cumulative_exposure_myr)
        ) if len(plume_state.cumulative_exposure_myr) else 0.0,
        mean_head_flux=float(np.sum(areas * head_flux) / total_area),
        max_head_flux=float(np.max(head_flux)) if len(head_flux) else 0.0,
        mean_tail_flux=float(np.sum(areas * tail_flux) / total_area),
        max_tail_flux=float(np.max(tail_flux)) if len(tail_flux) else 0.0,
        source_drift_enabled=bool(params.source_drift_enabled),
        mean_source_drift_speed_km_per_myr=(
            float(np.mean(plume_state.last_effective_source_speeds_km_per_myr))
            if plume_state.last_effective_source_speeds_km_per_myr is not None
            and len(plume_state.last_effective_source_speeds_km_per_myr)
            else 0.0
        ),
        max_source_drift_speed_km_per_myr=(
            float(np.max(plume_state.last_effective_source_speeds_km_per_myr))
            if plume_state.last_effective_source_speeds_km_per_myr is not None
            and len(plume_state.last_effective_source_speeds_km_per_myr)
            else 0.0
        ),
        active_source_path_length_km=(
            float(np.sum(plume_state.cumulative_source_distance_km))
            if plume_state.cumulative_source_distance_km is not None
            else 0.0
        ),
        maximum_active_source_path_length_km=(
            float(np.max(plume_state.cumulative_source_distance_km))
            if plume_state.cumulative_source_distance_km is not None
            and len(plume_state.cumulative_source_distance_km)
            else 0.0
        ),
        active_source_bend_angle_deg=(
            float(np.sum(plume_state.cumulative_source_bend_deg))
            if plume_state.cumulative_source_bend_deg is not None
            else 0.0
        ),
        maximum_active_source_bend_angle_deg=(
            float(np.max(plume_state.cumulative_source_bend_deg))
            if plume_state.cumulative_source_bend_deg is not None
            and len(plume_state.cumulative_source_bend_deg)
            else 0.0
        ),
        population_source_path_length_km=float(
            plume_state.population_source_distance_km
        ),
        population_source_bend_angle_deg=float(plume_state.population_source_bend_deg),
    )


def advance_mantle_plumes(
    mesh: SphereMesh,
    lithosphere: LithosphereState,
    plume_state: MantlePlumeState,
    dt_myr: float,
    radius_km: float,
    params: MantlePlumeParameters,
    craton_params: CratonParameters,
    *,
    source_flow_omega_field_rad_per_myr: Array | None = None,
) -> tuple[LithosphereState, MantlePlumeState, MantlePlumeDiagnostics]:
    """Advance plume births/lifecycles and modify overlying continental roots."""

    if dt_myr <= 0.0:
        raise ValueError("dt_myr must be positive")
    _validate_parameters(params)
    if plume_state.last_flux.shape != (mesh.cell_count,):
        raise ValueError("plume grid fields must have shape (cell_count,)")

    end_time = float(plume_state.time_myr) + float(dt_myr)
    _ensure_source_drift_state(plume_state, params)
    start_ages_myr = np.asarray(plume_state.ages_myr, dtype=np.float64).copy()
    plume_state.ages_myr = start_ages_myr + float(dt_myr)
    if params.enabled:
        while plume_state.next_birth_time_myr <= end_time + 1.0e-12:
            birth_time = float(plume_state.next_birth_time_myr)
            plume_id = int(plume_state.next_plume_id)
            birth_age = max(end_time - birth_time, 0.0)
            _append_plume(plume_state, params, plume_id, birth_age)
            start_ages_myr = np.append(start_ages_myr, 0.0)
            plume_state.next_plume_id += 1
            plume_state.next_birth_time_myr = birth_time + _birth_interval(
                params, plume_state.next_plume_id
            )
    if source_flow_omega_field_rad_per_myr is not None:
        update_plume_source_flow(
            mesh,
            plume_state,
            source_flow_omega_field_rad_per_myr,
            radius_km,
            params,
        )
    _advance_source_drift(
        plume_state, params, start_ages_myr, float(radius_km)
    )
    alive = plume_state.ages_myr < plume_state.lifetimes_myr
    plume_state.centers_unit = plume_state.centers_unit[alive]
    plume_state.ages_myr = plume_state.ages_myr[alive]
    plume_state.lifetimes_myr = plume_state.lifetimes_myr[alive]
    plume_state.head_radii_km = plume_state.head_radii_km[alive]
    plume_state.peak_fluxes = plume_state.peak_fluxes[alive]
    plume_state.plume_ids = plume_state.plume_ids[alive]
    plume_state.source_drift_axes_unit = plume_state.source_drift_axes_unit[alive]
    plume_state.source_drift_speeds_km_per_myr = (
        plume_state.source_drift_speeds_km_per_myr[alive]
    )
    plume_state.source_drift_segment_index = (
        plume_state.source_drift_segment_index[alive]
    )
    plume_state.cumulative_source_distance_km = (
        plume_state.cumulative_source_distance_km[alive]
    )
    plume_state.cumulative_source_bend_deg = (
        plume_state.cumulative_source_bend_deg[alive]
    )
    plume_state.source_flow_omega_rad_per_myr = (
        plume_state.source_flow_omega_rad_per_myr[alive]
    )
    plume_state.last_effective_source_axes_unit = (
        plume_state.last_effective_source_axes_unit[alive]
    )
    plume_state.last_effective_source_speeds_km_per_myr = (
        plume_state.last_effective_source_speeds_km_per_myr[alive]
    )
    plume_state.time_myr = end_time
    head_flux, tail_flux = plume_component_flux_fields(
        mesh, plume_state, radius_km, params
    )
    flux = np.clip(head_flux + tail_flux, 0.0, 1.5)
    plume_state.last_flux = flux
    plume_state.last_head_flux = head_flux
    plume_state.last_tail_flux = tail_flux
    plume_state.cumulative_exposure_myr = (
        np.asarray(plume_state.cumulative_exposure_myr, dtype=np.float64)
        + flux * float(dt_myr)
    )

    n = mesh.cell_count
    zero = np.zeros(n, dtype=np.float64)
    if not params.enabled or not params.lithosphere_weakening_enabled:
        return lithosphere, plume_state, diagnose_mantle_plumes(
            mesh, lithosphere, plume_state, radius_km, params, dt_myr=dt_myr
        )

    areas = mesh.physical_cell_areas_km2(float(radius_km))
    fraction, _ = continental_material_fields(lithosphere, areas)
    present = fraction > float(params.continental_fraction_epsilon)
    exposure = flux * np.clip(fraction, 0.0, 1.0)
    age_loss = zero.copy()
    depletion_loss = zero.copy()
    strength_loss = zero.copy()
    root_erosion = zero.copy()

    if (
        lithosphere.continental_lithosphere_age_myr is not None
        and lithosphere.mantle_depletion_fraction is not None
    ):
        age_before = np.asarray(
            lithosphere.continental_lithosphere_age_myr, dtype=np.float64
        ).copy()
        depletion_before = np.asarray(
            lithosphere.mantle_depletion_fraction, dtype=np.float64
        ).copy()
        strength_before = (
            zero.copy()
            if lithosphere.craton_strength is None
            else np.asarray(lithosphere.craton_strength, dtype=np.float64).copy()
        )
        age_retention = np.exp(
            -float(params.age_rejuvenation_rate_per_myr) * exposure * float(dt_myr)
        )
        depletion_retention = np.exp(
            -float(params.refertilization_rate_per_myr) * exposure * float(dt_myr)
        )
        age_after = age_before * age_retention
        depletion_after = depletion_before * depletion_retention
        age_after[~present] = 0.0
        depletion_after[~present] = 0.0
        lithosphere.continental_lithosphere_age_myr = age_after
        lithosphere.mantle_depletion_fraction = depletion_after
        lithosphere.craton_strength = craton_strength_from_memory(
            age_after, depletion_after, fraction, craton_params
        )
        age_loss = np.maximum(age_before - age_after, 0.0)
        depletion_loss = np.maximum(depletion_before - depletion_after, 0.0)
        strength_loss = np.maximum(
            strength_before - np.asarray(lithosphere.craton_strength, dtype=np.float64), 0.0
        )

    if lithosphere.mantle_lithosphere_thickness_km is not None:
        old_h = np.asarray(lithosphere.mantle_lithosphere_thickness_km, dtype=np.float64).copy()
        requested = float(params.root_erosion_km_per_myr) * exposure * float(dt_myr)
        new_h = np.maximum(
            old_h - requested,
            float(params.minimum_continental_root_thickness_km),
        )
        # Oceanic and numerically empty continental cells are never changed.
        new_h[~present] = old_h[~present]
        root_erosion = np.maximum(old_h - new_h, 0.0)
        lithosphere.mantle_lithosphere_thickness_km = new_h
        plume_state.cumulative_root_erosion_km = (
            np.asarray(plume_state.cumulative_root_erosion_km, dtype=np.float64)
            + root_erosion
        )

    diagnostics = diagnose_mantle_plumes(
        mesh,
        lithosphere,
        plume_state,
        radius_km,
        params,
        dt_myr=dt_myr,
        age_loss=age_loss,
        depletion_loss=depletion_loss,
        strength_loss=strength_loss,
        root_erosion=root_erosion,
    )
    return lithosphere, plume_state, diagnostics


__all__ = [
    "MantlePlumeParameters",
    "MantlePlumeState",
    "MantlePlumeDiagnostics",
    "initialize_mantle_plumes",
    "plume_component_flux_fields",
    "plume_flux_field",
    "update_plume_source_flow",
    "diagnose_mantle_plumes",
    "advance_mantle_plumes",
]
