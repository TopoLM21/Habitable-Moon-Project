"""Passive global hydrosphere with fixed water inventory and material-aware sea level.

v0.14 introduced one-way conserved water. v0.15 adds sub-grid material hypsometry
for mixed continental/oceanic cells while keeping the hydrosphere one-way coupled: tectonics and
isostatic topography determine basin geometry, while a conserved water volume
sets the global equipotential sea level.  Water does not yet alter erosion,
sedimentation, loading, climate, or plate forces.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .lithosphere import LithosphereState, continental_material_fields
from .mesh import SphereMesh
from .topography import TopographyState, TopographyParameters, material_subgrid_surface_elevations

Array = np.ndarray


@dataclass(slots=True)
class HydrosphereParameters:
    enabled: bool = True
    # If null/None, a fresh run calibrates the conserved inventory so that the
    # initial sea surface is exactly reference_sea_level_m.
    water_volume_km3: float | None = None
    reference_sea_level_m: float = 0.0
    solver_tolerance_m: float = 1.0e-4
    solver_max_iterations: int = 96
    shelf_depth_m: float = 200.0
    deep_ocean_depth_m: float = 2000.0
    # v0.15: mixed continental/oceanic cells contribute two unresolved surface
    # patches to water capacity instead of being flooded/dried as one averaged
    # scalar height.  This avoids turning an 80% continental coastal cell into
    # 100% ocean merely because its area-mean elevation is below sea level.
    subgrid_material_hypsometry: bool = True


@dataclass(slots=True)
class HydrosphereState:
    time_myr: float
    water_volume_km3: float
    reference_sea_level_m: float = 0.0


@dataclass(slots=True)
class HydrosphereDiagnostics:
    time_myr: float
    sea_level_m: float
    water_volume_km3: float
    volume_error_km3: float
    relative_volume_error: float
    equivalent_global_layer_m: float
    land_area_fraction: float
    ocean_area_fraction: float
    shallow_sea_area_fraction: float
    deep_ocean_area_fraction: float
    exposed_continental_material_area_fraction: float
    submerged_continental_material_area_fraction: float
    mean_ocean_depth_m: float
    max_ocean_depth_m: float


def water_volume_at_sea_level_km3(
    mesh: SphereMesh,
    elevation_m: Array,
    radius_km: float,
    sea_level_m: float,
) -> float:
    """Water volume between topography and one spherical sea surface.

    Each triangular cell is treated as a constant-elevation spherical patch.
    The shell-volume expression is exact for that piecewise-constant geometry:
        dV = omega/3 * (r_sea^3 - r_floor^3)
    on cells whose floor lies below sea level.
    """
    elev = np.asarray(elevation_m, dtype=np.float64)
    if elev.shape != (mesh.cell_count,):
        raise ValueError("elevation field must match mesh cell count")
    wet = elev < float(sea_level_m)
    if not np.any(wet):
        return 0.0
    r_sea = float(radius_km) + float(sea_level_m) / 1000.0
    r_floor = float(radius_km) + elev[wet] / 1000.0
    if np.any(r_floor <= 0.0) or r_sea <= 0.0:
        raise ValueError("topography/sea level reaches non-positive radius")
    omega = np.asarray(mesh.areas_unit_sphere, dtype=np.float64)[wet]
    return float(np.sum((omega / 3.0) * (r_sea**3 - r_floor**3)))



def water_volume_at_sea_level_material_km3(
    mesh: SphereMesh,
    continental_fraction: Array,
    ocean_surface_m: Array,
    continental_surface_m: Array,
    radius_km: float,
    sea_level_m: float,
) -> float:
    """Water volume over two unresolved material surface patches per cell."""
    f = np.clip(np.asarray(continental_fraction, dtype=np.float64), 0.0, 1.0)
    ocean = np.asarray(ocean_surface_m, dtype=np.float64)
    cont = np.asarray(continental_surface_m, dtype=np.float64)
    if f.shape != (mesh.cell_count,) or ocean.shape != f.shape or cont.shape != f.shape:
        raise ValueError("material hypsometry fields must match mesh cell count")
    sea = float(sea_level_m)
    r_sea = float(radius_km) + sea / 1000.0
    if r_sea <= 0.0:
        raise ValueError("sea level reaches non-positive radius")
    omega = np.asarray(mesh.areas_unit_sphere, dtype=np.float64)

    def patch_volume(surface: Array, weight: Array) -> float:
        wet = (surface < sea) & (weight > 0.0)
        if not np.any(wet):
            return 0.0
        floor = float(radius_km) + surface[wet] / 1000.0
        if np.any(floor <= 0.0):
            raise ValueError("topography reaches non-positive radius")
        return float(np.sum((omega[wet] * weight[wet] / 3.0) * (r_sea**3 - floor**3)))

    return patch_volume(ocean, 1.0 - f) + patch_volume(cont, f)


def solve_sea_level_material_m(
    mesh: SphereMesh,
    continental_fraction: Array,
    ocean_surface_m: Array,
    continental_surface_m: Array,
    radius_km: float,
    water_volume_km3: float,
    params: HydrosphereParameters,
) -> float:
    target = float(water_volume_km3)
    if target < 0.0:
        raise ValueError("water volume cannot be negative")
    f = np.clip(np.asarray(continental_fraction, dtype=np.float64), 0.0, 1.0)
    ocean = np.asarray(ocean_surface_m, dtype=np.float64)
    cont = np.asarray(continental_surface_m, dtype=np.float64)
    active_surfaces = np.concatenate([ocean[(1.0-f)>0.0], cont[f>0.0]])
    if active_surfaces.size == 0:
        return float(params.reference_sea_level_m)
    if target == 0.0:
        return float(np.min(active_surfaces))
    lo = float(np.min(active_surfaces)) - 1.0
    hi = max(float(np.max(active_surfaces)) + 1.0, float(params.reference_sea_level_m) + 1.0)
    expansion = max(1000.0, hi - lo)
    for _ in range(64):
        cap = water_volume_at_sea_level_material_km3(mesh, f, ocean, cont, radius_km, hi)
        if cap >= target:
            break
        hi += expansion
        expansion *= 2.0
    else:
        raise RuntimeError("failed to bracket material-aware sea level")
    tol = max(float(params.solver_tolerance_m), 1.0e-9)
    for _ in range(max(int(params.solver_max_iterations), 1)):
        mid = 0.5*(lo+hi)
        vol = water_volume_at_sea_level_material_km3(mesh, f, ocean, cont, radius_km, mid)
        if vol < target:
            lo = mid
        else:
            hi = mid
        if hi-lo <= tol:
            break
    return 0.5*(lo+hi)

def solve_sea_level_m(
    mesh: SphereMesh,
    elevation_m: Array,
    radius_km: float,
    water_volume_km3: float,
    params: HydrosphereParameters,
) -> float:
    """Solve the unique sea level matching the conserved water inventory."""
    target = float(water_volume_km3)
    if target < 0.0:
        raise ValueError("water volume cannot be negative")
    elev = np.asarray(elevation_m, dtype=np.float64)
    if target == 0.0:
        return float(np.min(elev))

    lo = float(np.min(elev)) - 1.0
    hi = max(float(np.max(elev)) + 1.0, float(params.reference_sea_level_m) + 1.0)
    cap_hi = water_volume_at_sea_level_km3(mesh, elev, radius_km, hi)
    # Defensive expansion for exotic future topographies or explicit very-deep
    # water inventories.  Normally max(topography) is already more than enough.
    expansion = max(1000.0, hi - lo)
    for _ in range(64):
        if cap_hi >= target:
            break
        hi += expansion
        expansion *= 2.0
        cap_hi = water_volume_at_sea_level_km3(mesh, elev, radius_km, hi)
    else:
        raise RuntimeError("failed to bracket sea level")

    tol = max(float(params.solver_tolerance_m), 1.0e-9)
    for _ in range(max(int(params.solver_max_iterations), 1)):
        mid = 0.5 * (lo + hi)
        vol = water_volume_at_sea_level_km3(mesh, elev, radius_km, mid)
        if vol < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol:
            break
    return 0.5 * (lo + hi)


def initialize_hydrosphere(
    mesh: SphereMesh,
    topography: TopographyState,
    radius_km: float,
    params: HydrosphereParameters,
    lithosphere: LithosphereState | None = None,
    topography_params: TopographyParameters | None = None,
) -> HydrosphereState:
    if not bool(params.enabled):
        return HydrosphereState(float(topography.time_myr), 0.0, float(params.reference_sea_level_m))
    if params.water_volume_km3 is None:
        if bool(params.subgrid_material_hypsometry) and lithosphere is not None and topography_params is not None:
            f, ocean_surface, cont_surface = material_subgrid_surface_elevations(
                mesh, lithosphere, topography, radius_km, topography_params
            )
            volume = water_volume_at_sea_level_material_km3(
                mesh, f, ocean_surface, cont_surface, radius_km, float(params.reference_sea_level_m)
            )
        else:
            volume = water_volume_at_sea_level_km3(
                mesh, topography.elevation_m, radius_km, float(params.reference_sea_level_m)
            )
    else:
        volume = float(params.water_volume_km3)
        if volume < 0.0:
            raise ValueError("hydrosphere.water_volume_km3 cannot be negative")
    return HydrosphereState(
        time_myr=float(topography.time_myr),
        water_volume_km3=float(volume),
        reference_sea_level_m=float(params.reference_sea_level_m),
    )


def diagnose_hydrosphere(
    mesh: SphereMesh,
    lithosphere: LithosphereState,
    topography: TopographyState,
    hydrosphere: HydrosphereState,
    radius_km: float,
    params: HydrosphereParameters,
    topography_params: TopographyParameters | None = None,
) -> HydrosphereDiagnostics:
    areas = mesh.physical_cell_areas_km2(radius_km)
    total_area = float(np.sum(areas))
    elev = np.asarray(topography.elevation_m, dtype=np.float64)
    use_subgrid = bool(params.subgrid_material_hypsometry) and topography_params is not None
    cont_frac, _ = continental_material_fields(lithosphere, areas)

    if use_subgrid:
        cont_frac, ocean_surface, cont_surface = material_subgrid_surface_elevations(
            mesh, lithosphere, topography, radius_km, topography_params
        )
        if not bool(params.enabled) or hydrosphere.water_volume_km3 <= 0.0:
            active = np.concatenate([ocean_surface[(1.0-cont_frac)>0], cont_surface[cont_frac>0]])
            sea = float(np.min(active)) if active.size else float(np.min(elev))
        else:
            sea = solve_sea_level_material_m(
                mesh, cont_frac, ocean_surface, cont_surface, radius_km, hydrosphere.water_volume_km3, params
            )
        ocean_weight = 1.0 - cont_frac
        cont_weight = cont_frac
        ocean_depth = np.maximum(sea - ocean_surface, 0.0)
        cont_depth = np.maximum(sea - cont_surface, 0.0)
        ocean_wet = ocean_depth > 0.0
        cont_wet = cont_depth > 0.0
        wet_area = float(np.sum(areas * (ocean_weight*ocean_wet + cont_weight*cont_wet)))
        land_area = total_area - wet_area
        shallow_area = float(np.sum(areas * (
            ocean_weight * (ocean_wet & (ocean_depth <= float(params.shelf_depth_m)))
            + cont_weight * (cont_wet & (cont_depth <= float(params.shelf_depth_m)))
        )))
        deep_area = float(np.sum(areas * (
            ocean_weight * (ocean_wet & (ocean_depth >= float(params.deep_ocean_depth_m)))
            + cont_weight * (cont_wet & (cont_depth >= float(params.deep_ocean_depth_m)))
        )))
        exposed_cont = float(np.sum(areas * cont_weight * (~cont_wet))) / total_area
        submerged_cont = float(np.sum(areas * cont_weight * cont_wet)) / total_area
        solved_volume = water_volume_at_sea_level_material_km3(
            mesh, cont_frac, ocean_surface, cont_surface, radius_km, sea
        )
        depth_area_sum = float(np.sum(areas * (
            ocean_weight * ocean_depth * ocean_wet + cont_weight * cont_depth * cont_wet
        )))
        mean_depth = depth_area_sum / wet_area if wet_area > 0.0 else 0.0
        max_depth = max(
            float(np.max(ocean_depth[ocean_wet])) if np.any(ocean_wet) else 0.0,
            float(np.max(cont_depth[cont_wet])) if np.any(cont_wet) else 0.0,
        )
    else:
        if not bool(params.enabled) or hydrosphere.water_volume_km3 <= 0.0:
            sea = float(np.min(elev))
        else:
            sea = solve_sea_level_m(mesh, elev, radius_km, hydrosphere.water_volume_km3, params)
        depth = np.maximum(sea - elev, 0.0)
        wet = depth > 0.0
        dry = ~wet
        wet_area = float(np.sum(areas[wet]))
        land_area = float(np.sum(areas[dry]))
        shallow_area = float(np.sum(areas[wet & (depth <= float(params.shelf_depth_m))]))
        deep_area = float(np.sum(areas[wet & (depth >= float(params.deep_ocean_depth_m))]))
        exposed_cont = float(np.sum(areas * cont_frac * dry)) / total_area
        submerged_cont = float(np.sum(areas * cont_frac * wet)) / total_area
        solved_volume = water_volume_at_sea_level_km3(mesh, elev, radius_km, sea)
        mean_depth = float(np.sum(areas[wet] * depth[wet]) / wet_area) if wet_area > 0.0 else 0.0
        max_depth = float(np.max(depth)) if np.any(wet) else 0.0
    global_area = 4.0 * np.pi * float(radius_km) ** 2
    egl_m = 1000.0 * float(hydrosphere.water_volume_km3) / max(global_area, 1e-30)
    return HydrosphereDiagnostics(
        time_myr=float(topography.time_myr),
        sea_level_m=float(sea),
        water_volume_km3=float(hydrosphere.water_volume_km3),
        volume_error_km3=float(solved_volume - hydrosphere.water_volume_km3),
        relative_volume_error=float((solved_volume - hydrosphere.water_volume_km3) / max(abs(hydrosphere.water_volume_km3), 1.0)),
        equivalent_global_layer_m=float(egl_m),
        land_area_fraction=land_area / total_area,
        ocean_area_fraction=wet_area / total_area,
        shallow_sea_area_fraction=shallow_area / total_area,
        deep_ocean_area_fraction=deep_area / total_area,
        exposed_continental_material_area_fraction=exposed_cont,
        submerged_continental_material_area_fraction=submerged_cont,
        mean_ocean_depth_m=mean_depth,
        max_ocean_depth_m=float(max_depth),
    )


def advance_hydrosphere(
    mesh: SphereMesh,
    lithosphere: LithosphereState,
    topography: TopographyState,
    previous: HydrosphereState,
    radius_km: float,
    params: HydrosphereParameters,
    topography_params: TopographyParameters | None = None,
) -> tuple[HydrosphereState, HydrosphereDiagnostics]:
    """Advance only time; water inventory is strictly conserved in v0.14."""
    out = HydrosphereState(
        time_myr=float(topography.time_myr),
        water_volume_km3=float(previous.water_volume_km3),
        reference_sea_level_m=float(previous.reference_sea_level_m),
    )
    return out, diagnose_hydrosphere(mesh, lithosphere, topography, out, radius_km, params, topography_params)
