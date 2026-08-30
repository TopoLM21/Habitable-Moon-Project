"""v0.25 mantle-plume forcing and metasomatic craton modification.

This is an effective surface projection of deep, approximately mantle-fixed
plumes.  The plume field itself is Eulerian; plates and their material memory
move across it.  While continental lithosphere overlies an active plume, the
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


def _append_plume(state: MantlePlumeState, params: MantlePlumeParameters, plume_id: int, age_myr: float) -> None:
    center, lifetime, radius, peak = _sample_plume(params, plume_id)
    state.centers_unit = np.vstack((state.centers_unit, center.reshape(1, 3)))
    state.ages_myr = np.append(state.ages_myr, float(age_myr))
    state.lifetimes_myr = np.append(state.lifetimes_myr, lifetime)
    state.head_radii_km = np.append(state.head_radii_km, radius)
    state.peak_fluxes = np.append(state.peak_fluxes, peak)


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
        # values as cell-average quadrature weights and preserve the continuous
        # small-angle Gaussian integral (2*pi*sigma^2) on every resolution.
        # This removes centroid-alignment bias without artificially widening
        # the physical tail.
        areas = mesh.physical_cell_areas_km2(float(radius_km))
        for kernel, sigma in ((head_kernel, head_sigma), (tail_kernel, tail_sigma)):
            sampled = np.sum(areas[:, None] * kernel, axis=0)
            target = 2.0 * np.pi * np.square(sigma)
            kernel *= (
                target / np.maximum(sampled, 1.0e-30)
            )[None, :]
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
    )


def advance_mantle_plumes(
    mesh: SphereMesh,
    lithosphere: LithosphereState,
    plume_state: MantlePlumeState,
    dt_myr: float,
    radius_km: float,
    params: MantlePlumeParameters,
    craton_params: CratonParameters,
) -> tuple[LithosphereState, MantlePlumeState, MantlePlumeDiagnostics]:
    """Advance plume births/lifecycles and modify overlying continental roots."""

    if dt_myr <= 0.0:
        raise ValueError("dt_myr must be positive")
    _validate_parameters(params)
    if plume_state.last_flux.shape != (mesh.cell_count,):
        raise ValueError("plume grid fields must have shape (cell_count,)")

    end_time = float(plume_state.time_myr) + float(dt_myr)
    plume_state.ages_myr = np.asarray(plume_state.ages_myr, dtype=np.float64) + float(dt_myr)
    if params.enabled:
        while plume_state.next_birth_time_myr <= end_time + 1.0e-12:
            birth_time = float(plume_state.next_birth_time_myr)
            plume_id = int(plume_state.next_plume_id)
            _append_plume(plume_state, params, plume_id, max(end_time - birth_time, 0.0))
            plume_state.next_plume_id += 1
            plume_state.next_birth_time_myr = birth_time + _birth_interval(
                params, plume_state.next_plume_id
            )
    alive = plume_state.ages_myr < plume_state.lifetimes_myr
    plume_state.centers_unit = plume_state.centers_unit[alive]
    plume_state.ages_myr = plume_state.ages_myr[alive]
    plume_state.lifetimes_myr = plume_state.lifetimes_myr[alive]
    plume_state.head_radii_km = plume_state.head_radii_km[alive]
    plume_state.peak_fluxes = plume_state.peak_fluxes[alive]
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
    "diagnose_mantle_plumes",
    "advance_mantle_plumes",
]
