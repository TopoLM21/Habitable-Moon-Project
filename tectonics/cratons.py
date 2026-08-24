"""v0.24 continental-lithosphere maturation and cratonic memory.

The surface crust raster alone cannot distinguish a juvenile volcanic arc from
an old, depleted continental nucleus.  This module adds three continuous,
material-following fields to :class:`LithosphereState`:

``continental_lithosphere_age_myr``
    Effective age of the coupled continental crust/mantle-root column.  It is
    chronological while the root remains intact, but is reduced by strong
    rifting or sustained thermal rejuvenation.
``mantle_depletion_fraction``
    A dimensionless 0--1 proxy for melt depletion and compositional buoyancy.
``craton_strength``
    A derived 0--1 maturity measure used by rifting, collision resistance and
    the mechanical mantle-root target.

These are effective geological proxies, not a petrological phase-equilibrium
solver.  They are transported with the same winning material parcel as the
other lithosphere memories and are diluted conservatively when juvenile arc
material increases the tracked continental volume of a cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lithosphere import LithosphereState, continental_material_fields
from .mesh import SphereMesh

Array = np.ndarray


@dataclass(slots=True)
class CratonParameters:
    """Calibrated effective parameters for v0.24 craton maturation."""

    enabled: bool = True
    age_maturity_timescale_myr: float = 700.0
    depletion_maturation_timescale_myr: float = 900.0
    maximum_depletion_fraction: float = 0.72
    maximum_effective_age_myr: float = 4500.0
    rift_rejuvenation_rate_per_myr: float = 0.010
    thermal_rejuvenation_rate_per_myr: float = 0.003
    thermal_rejuvenation_threshold: float = 0.55
    rift_reference_extension: float = 0.75
    cratonic_strength_threshold: float = 0.65
    continental_fraction_epsilon: float = 1.0e-9
    extension_resistance_gain: float = 0.68
    minimum_extension_factor: float = 0.28


@dataclass(slots=True)
class CratonDiagnostics:
    time_myr: float
    dt_myr: float
    mean_continental_lithosphere_age_myr: float
    max_continental_lithosphere_age_myr: float
    mean_mantle_depletion_fraction: float
    max_mantle_depletion_fraction: float
    mean_craton_strength: float
    max_craton_strength: float
    cratonic_area_fraction_of_surface: float
    cratonic_fraction_of_continental_material: float
    juvenile_continental_fraction_of_surface: float
    new_continental_material_volume_km3: float
    mean_extension_factor_continental: float


def _validate_parameters(params: CratonParameters) -> None:
    if params.age_maturity_timescale_myr <= 0.0:
        raise ValueError("age_maturity_timescale_myr must be positive")
    if params.depletion_maturation_timescale_myr <= 0.0:
        raise ValueError("depletion_maturation_timescale_myr must be positive")
    if not (0.0 < params.maximum_depletion_fraction <= 1.0):
        raise ValueError("maximum_depletion_fraction must be in (0, 1]")
    if params.maximum_effective_age_myr <= 0.0:
        raise ValueError("maximum_effective_age_myr must be positive")
    if not (0.0 <= params.thermal_rejuvenation_threshold < 1.0):
        raise ValueError("thermal_rejuvenation_threshold must be in [0, 1)")
    if params.rift_reference_extension <= 0.0:
        raise ValueError("rift_reference_extension must be positive")
    if not (0.0 <= params.cratonic_strength_threshold <= 1.0):
        raise ValueError("cratonic_strength_threshold must be in [0, 1]")
    if not (0.0 <= params.minimum_extension_factor <= 1.0):
        raise ValueError("minimum_extension_factor must be in [0, 1]")


def craton_strength_from_memory(
    continental_lithosphere_age_myr: Array,
    mantle_depletion_fraction: Array,
    continental_fraction: Array,
    params: CratonParameters,
) -> Array:
    """Derive continuous strength from age, depletion and material footprint."""

    age = np.clip(
        np.asarray(continental_lithosphere_age_myr, dtype=np.float64),
        0.0,
        float(params.maximum_effective_age_myr),
    )
    depletion = np.clip(
        np.asarray(mantle_depletion_fraction, dtype=np.float64),
        0.0,
        float(params.maximum_depletion_fraction),
    )
    fraction = np.clip(np.asarray(continental_fraction, dtype=np.float64), 0.0, 1.0)
    age_factor = 1.0 - np.exp(-age / float(params.age_maturity_timescale_myr))
    depletion_factor = depletion / float(params.maximum_depletion_fraction)
    # The geometric mean requires both thermal/chronological maturation and a
    # depleted buoyant root.  Fractional coastline cells blend continuously.
    strength = np.sqrt(np.maximum(age_factor * depletion_factor, 0.0)) * np.sqrt(fraction)
    return np.clip(strength, 0.0, 1.0)


def craton_extension_factor(
    state: LithosphereState,
    *,
    gain: float,
    minimum_factor: float,
) -> Array:
    """Return a multiplicative 0--1 factor for continental extension."""

    n = len(state.crust_age_myr)
    if state.craton_strength is None:
        return np.ones(n, dtype=np.float64)
    strength = np.clip(np.asarray(state.craton_strength, dtype=np.float64), 0.0, 1.0)
    return np.clip(1.0 - float(gain) * strength, float(minimum_factor), 1.0)


def initialize_craton_memory(
    mesh: SphereMesh,
    state: LithosphereState,
    radius_km: float,
    params: CratonParameters | None = None,
) -> LithosphereState:
    """Initialize missing v0.24 fields without altering existing memory."""

    p = CratonParameters() if params is None else params
    _validate_parameters(p)
    areas = mesh.physical_cell_areas_km2(float(radius_km))
    fraction, _ = continental_material_fields(state, areas)
    present = fraction > float(p.continental_fraction_epsilon)

    if state.continental_lithosphere_age_myr is None:
        age = np.zeros(mesh.cell_count, dtype=np.float64)
        age[present] = np.maximum(np.asarray(state.crust_age_myr, dtype=np.float64)[present], 0.0)
        state.continental_lithosphere_age_myr = np.minimum(age, float(p.maximum_effective_age_myr))
    if state.mantle_depletion_fraction is None:
        age = np.asarray(state.continental_lithosphere_age_myr, dtype=np.float64)
        depletion = float(p.maximum_depletion_fraction) * (
            1.0 - np.exp(-age / float(p.depletion_maturation_timescale_myr))
        )
        depletion[~present] = 0.0
        state.mantle_depletion_fraction = depletion

    age = np.asarray(state.continental_lithosphere_age_myr, dtype=np.float64)
    depletion = np.asarray(state.mantle_depletion_fraction, dtype=np.float64)
    age[~present] = 0.0
    depletion[~present] = 0.0
    state.continental_lithosphere_age_myr = age
    state.mantle_depletion_fraction = depletion
    state.craton_strength = craton_strength_from_memory(age, depletion, fraction, p)
    return state


def diagnose_craton_memory(
    mesh: SphereMesh,
    state: LithosphereState,
    radius_km: float,
    params: CratonParameters,
    *,
    dt_myr: float = 0.0,
    new_continental_material_volume_km3: float = 0.0,
) -> CratonDiagnostics:
    """Summarize the transported craton fields using physical area weights."""

    areas = mesh.physical_cell_areas_km2(float(radius_km))
    fraction, _ = continental_material_fields(state, areas)
    weights = areas * fraction
    total_surface = float(np.sum(areas))
    total_continental = float(np.sum(weights))
    age = np.zeros(mesh.cell_count, dtype=np.float64) if state.continental_lithosphere_age_myr is None else np.asarray(state.continental_lithosphere_age_myr, dtype=np.float64)
    depletion = np.zeros(mesh.cell_count, dtype=np.float64) if state.mantle_depletion_fraction is None else np.asarray(state.mantle_depletion_fraction, dtype=np.float64)
    strength = np.zeros(mesh.cell_count, dtype=np.float64) if state.craton_strength is None else np.asarray(state.craton_strength, dtype=np.float64)
    present = weights > 0.0
    cratonic = present & (strength >= float(params.cratonic_strength_threshold))
    juvenile = present & (strength < 0.25)
    ext_factor = craton_extension_factor(
        state,
        gain=float(params.extension_resistance_gain),
        minimum_factor=float(params.minimum_extension_factor),
    )

    def weighted_mean(values: Array) -> float:
        if total_continental <= 0.0:
            return 0.0
        return float(np.sum(weights * np.asarray(values, dtype=np.float64)) / total_continental)

    return CratonDiagnostics(
        time_myr=float(state.time_myr),
        dt_myr=float(dt_myr),
        mean_continental_lithosphere_age_myr=weighted_mean(age),
        max_continental_lithosphere_age_myr=float(np.max(age[present])) if np.any(present) else 0.0,
        mean_mantle_depletion_fraction=weighted_mean(depletion),
        max_mantle_depletion_fraction=float(np.max(depletion[present])) if np.any(present) else 0.0,
        mean_craton_strength=weighted_mean(strength),
        max_craton_strength=float(np.max(strength[present])) if np.any(present) else 0.0,
        cratonic_area_fraction_of_surface=float(np.sum(weights[cratonic]) / max(total_surface, 1e-30)),
        cratonic_fraction_of_continental_material=float(np.sum(weights[cratonic]) / max(total_continental, 1e-30)),
        juvenile_continental_fraction_of_surface=float(np.sum(weights[juvenile]) / max(total_surface, 1e-30)),
        new_continental_material_volume_km3=float(new_continental_material_volume_km3),
        mean_extension_factor_continental=weighted_mean(ext_factor),
    )


def advance_craton_memory(
    mesh: SphereMesh,
    state: LithosphereState,
    dt_myr: float,
    radius_km: float,
    params: CratonParameters,
    *,
    pre_cycle_continental_volume_km3: Array | None = None,
) -> tuple[LithosphereState, CratonDiagnostics]:
    """Advance maturation, juvenile dilution and tectonic rejuvenation in-place."""

    if dt_myr <= 0.0:
        raise ValueError("dt_myr must be positive")
    _validate_parameters(params)
    initialize_craton_memory(mesh, state, radius_km, params)
    areas = mesh.physical_cell_areas_km2(float(radius_km))
    fraction, volume = continental_material_fields(state, areas)
    present = fraction > float(params.continental_fraction_epsilon)

    age = np.asarray(state.continental_lithosphere_age_myr, dtype=np.float64).copy()
    depletion = np.asarray(state.mantle_depletion_fraction, dtype=np.float64).copy()
    new_volume = 0.0
    if pre_cycle_continental_volume_km3 is not None:
        before = np.maximum(np.asarray(pre_cycle_continental_volume_km3, dtype=np.float64), 0.0)
        if before.shape != (mesh.cell_count,):
            raise ValueError("pre_cycle_continental_volume_km3 must have shape (cell_count,)")
        added = np.maximum(volume - before, 0.0)
        new_volume = float(np.sum(added))
        # Existing memory is volume-weighted with newly generated arc material,
        # whose lithospheric age/depletion starts at zero.
        retained = np.ones(mesh.cell_count, dtype=np.float64)
        growing = volume > before + 1e-12
        retained[growing] = np.clip(before[growing] / np.maximum(volume[growing], 1e-30), 0.0, 1.0)
        age *= retained
        depletion *= retained

    if not params.enabled:
        age[~present] = 0.0
        depletion[~present] = 0.0
        state.continental_lithosphere_age_myr = age
        state.mantle_depletion_fraction = depletion
        state.craton_strength = craton_strength_from_memory(age, depletion, fraction, params)
        return state, diagnose_craton_memory(
            mesh, state, radius_km, params, dt_myr=dt_myr,
            new_continental_material_volume_km3=new_volume,
        )

    age[present] += float(dt_myr)
    alpha = 1.0 - np.exp(-float(dt_myr) / float(params.depletion_maturation_timescale_myr))
    depletion[present] += alpha * (float(params.maximum_depletion_fraction) - depletion[present])

    extension = np.zeros(mesh.cell_count, dtype=np.float64) if state.rift_extension is None else np.asarray(state.rift_extension, dtype=np.float64)
    heat = np.zeros(mesh.cell_count, dtype=np.float64) if state.supercontinent_heat is None else np.asarray(state.supercontinent_heat, dtype=np.float64)
    rift_signal = np.clip(extension / float(params.rift_reference_extension), 0.0, 1.0)
    heat_signal = np.clip(
        (heat - float(params.thermal_rejuvenation_threshold))
        / max(1.0 - float(params.thermal_rejuvenation_threshold), 1e-9),
        0.0,
        1.0,
    )
    rejuvenation_rate = (
        float(params.rift_rejuvenation_rate_per_myr) * rift_signal
        + float(params.thermal_rejuvenation_rate_per_myr) * heat_signal
    )
    retention = np.exp(-np.maximum(rejuvenation_rate, 0.0) * float(dt_myr))
    age *= retention
    depletion *= retention

    age = np.clip(age, 0.0, float(params.maximum_effective_age_myr))
    depletion = np.clip(depletion, 0.0, float(params.maximum_depletion_fraction))
    age[~present] = 0.0
    depletion[~present] = 0.0
    state.continental_lithosphere_age_myr = age
    state.mantle_depletion_fraction = depletion
    state.craton_strength = craton_strength_from_memory(age, depletion, fraction, params)
    return state, diagnose_craton_memory(
        mesh, state, radius_km, params, dt_myr=dt_myr,
        new_continental_material_volume_km3=new_volume,
    )


__all__ = [
    "CratonParameters",
    "CratonDiagnostics",
    "craton_strength_from_memory",
    "craton_extension_factor",
    "initialize_craton_memory",
    "diagnose_craton_memory",
    "advance_craton_memory",
]
