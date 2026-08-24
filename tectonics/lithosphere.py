"""v0.4 lithosphere: oceanic + continental crust and tidal weakening.

Key numerical change from v0.3
--------------------------------
v0.3 forward-mapped every cell centroid to the nearest fixed cell.  A rigid
rotation of a discrete mesh is not a permutation of that mesh, so this created
spurious gaps/overlaps *inside* plates.

v0.4 uses backward semi-Lagrangian coverage.  For every target surface cell and
for every plate, the target point is inverse-rotated by that plate's Euler
motion and sampled on the previous state.  A plate covers the target only if
that back-traced source cell belonged to the same plate.  Thus interior plate
motion remains filled; gaps and overlaps are concentrated near moving plate
boundaries.

The model is still an effective geological prototype, not a full continuum
mantle/lithosphere solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import heapq

import numpy as np
from scipy.spatial import cKDTree

from .evolution import rotate_points_by_plate
from .kinematics import BoundaryRecord, BoundaryType, classify_boundaries
from .mesh import SphereMesh
from .plates import PlateSystem
from .tides import EccentricityHistory, tidal_strain_amplitude

Array = np.ndarray


class CrustType(IntEnum):
    OCEANIC = 0
    CONTINENTAL = 1


@dataclass(slots=True)
class LithosphereState:
    time_myr: float
    cell_plate: Array
    crust_type: Array
    crust_age_myr: Array
    crust_thickness_km: Array
    tidal_damage: Array
    # v0.9.2: accumulated tectonic extension and duration.  Optional for
    # backwards-compatible construction in older unit tests/checkpoints.
    rift_extension: Array | None = None
    extension_age_myr: Array | None = None
    # v0.9.4: material memory used by late boundary nucleation.
    collision_seam_weakness: Array | None = None
    intraplate_stress: Array | None = None
    supercontinent_heat: Array | None = None
    # v0.11: continental material is tracked independently from the discrete
    # surface/plate raster. ``continental_fraction`` is the fraction of the
    # cell footprint occupied by continental crustal material; the companion
    # volume is the actual felsic crust volume in km^3.  Older states/checkpoints
    # may omit both fields; in that case they are reconstructed losslessly from
    # the legacy binary crust_type + thickness representation at the start of a
    # step.
    continental_fraction: Array | None = None
    continental_volume_km3: Array | None = None
    # v0.16: crust and mechanical lithosphere are distinct layers.  The
    # crustal fields above describe chemical crust; these arrays describe the
    # cold mantle portion of the mechanical plate below it.  Positive density
    # anomaly means denser than the underlying asthenosphere and therefore
    # contributes negative buoyancy / slab pull.
    mantle_lithosphere_thickness_km: Array | None = None
    mantle_lithosphere_density_anomaly_kg_m3: Array | None = None
    # v0.23: unconsolidated/sedimentary cover transported with the winning
    # surface parcel.  Stored as physical bulk volume per cell in km^3 so the
    # surface sediment budget is conservative across unequal-area meshes.
    sediment_volume_km3: Array | None = None
    # v0.24: continuous continental-lithosphere memory.  These fields separate
    # juvenile arc-derived continent from old, depleted, mechanically strong
    # cratonic roots.  They remain optional so historical states/checkpoints and
    # the v0.23 runner preserve their exact behaviour.
    continental_lithosphere_age_myr: Array | None = None
    mantle_depletion_fraction: Array | None = None
    craton_strength: Array | None = None


@dataclass(slots=True)
class LithosphereStepDiagnostics:
    time_myr: float
    dt_myr: float
    eccentricity: float
    max_tidal_strain: float
    max_radial_displacement_m: float
    gap_fraction: float
    overlap_fraction: float
    created_oceanic_area_km2: float
    subducted_oceanic_area_km2: float
    continental_collision_area_km2: float
    tidally_rifted_continental_area_km2: float
    numerical_continental_area_correction_km2: float
    continental_area_fraction: float
    mean_oceanic_age_myr: float
    max_oceanic_age_myr: float
    mean_tidal_damage: float
    visible_continental_area_fraction: float = 0.0
    continental_material_area_error_km2: float = 0.0
    actively_extending_continental_area_km2: float = 0.0
    continental_thinning_volume_km3: float = 0.0
    active_continental_rift_gap_cells: int = 0
    passive_margin_gap_cells_suppressed: int = 0
    mean_rift_extension_continental: float = 0.0
    max_rift_extension: float = 0.0
    max_extension_age_myr: float = 0.0
    # v0.10 reconstructed conservative transport diagnostics.
    conservative_transport_commits: int = 0
    transport_cumulative_commit_count: int = 0
    transport_mean_residual_angle_deg: float = 0.0
    transport_max_residual_angle_deg: float = 0.0
    transport_max_hold_age_myr: float = 0.0
    conservative_transport_volume_error_km3: float = 0.0
    collision_overflow_redistributed_volume_km3: float = 0.0
    collision_raw_max_thickness_km: float = 0.0
    collision_post_redistribution_max_thickness_km: float = 0.0
    collision_overflow_unresolved_volume_km3: float = 0.0
    # Source cell used for the winning surface parcel in this step.  This is
    # ephemeral step metadata used to advect continental-cycle memory with the
    # exact same discrete transport map. -1 denotes newly generated material.
    material_source_index: Array | None = None
    continental_breakup_recycled_volume_km3: float = 0.0
    rift_recycled_volume_km3: float = 0.0


@dataclass(slots=True)
class LithosphereSnapshot:
    state: LithosphereState
    boundaries: list[BoundaryRecord]
    tidal_strain: Array
    tidal_weakening: Array
    diagnostics: LithosphereStepDiagnostics | None


def _plate_counts(system: PlateSystem) -> Array:
    return np.bincount(system.cell_plate, minlength=len(system.plates))


def _grow_patch_within_plate(
    mesh: SphereMesh,
    plate_id: int,
    seed_cell: int,
    quota: int,
    cell_plate: Array,
) -> list[int]:
    if quota <= 0:
        return []
    chosen: set[int] = set()
    seen: set[int] = {int(seed_cell)}
    heap: list[tuple[float, int]] = [(0.0, int(seed_cell))]
    seed_pos = mesh.centroids[int(seed_cell)]
    while heap and len(chosen) < quota:
        _, cell = heapq.heappop(heap)
        if int(cell_plate[cell]) != plate_id:
            continue
        chosen.add(cell)
        for nb in mesh.neighbors[cell]:
            if nb in seen or int(cell_plate[nb]) != plate_id:
                continue
            seen.add(nb)
            angle = float(np.arccos(np.clip(np.dot(seed_pos, mesh.centroids[nb]), -1.0, 1.0)))
            heapq.heappush(heap, (angle, int(nb)))
    return sorted(chosen)



def oceanic_thermal_lithosphere_total_thickness_km(
    age_myr: Array | float,
    *,
    thermal_diffusivity_m2_s: float = 1.0e-6,
    cooling_coefficient: float = 2.0,
    max_total_thickness_km: float = 155.0,
) -> Array:
    """Half-space-cooling proxy for total oceanic thermal lithosphere.

    ``cooling_coefficient * sqrt(kappa*t)`` gives about 112 km at 100 Myr for
    kappa=1e-6 m2/s.  A cap represents the transition toward plate cooling and
    prevents unbounded thickening of very old ocean floor.
    """
    age = np.maximum(np.asarray(age_myr, dtype=np.float64), 0.0)
    seconds = age * 1.0e6 * 365.25 * 24.0 * 3600.0
    total = float(cooling_coefficient) * np.sqrt(max(float(thermal_diffusivity_m2_s), 0.0) * seconds) / 1000.0
    return np.minimum(total, float(max_total_thickness_km))


def target_mantle_lithosphere_fields(
    state: LithosphereState,
    *,
    oceanic_crust_reference_km: float = 7.0,
    thermal_diffusivity_m2_s: float = 1.0e-6,
    cooling_coefficient: float = 2.0,
    oceanic_max_total_thickness_km: float = 155.0,
    continental_mantle_thickness_km: float = 125.0,
    continental_min_rifted_mantle_thickness_km: float = 55.0,
    mantle_density_kg_m3: float = 3300.0,
    thermal_expansion_per_k: float = 3.0e-5,
    mantle_temperature_contrast_k: float = 1300.0,
    oceanic_mean_temperature_deficit_fraction: float = 0.50,
    continental_density_anomaly_kg_m3: float = -8.0,
    craton_root_thickening_km: float = 75.0,
    craton_depletion_density_reduction_kg_m3: float = 28.0,
) -> tuple[Array, Array]:
    """Return local target mantle-lithosphere thickness and density anomaly.

    Oceanic mechanical thickness grows from cooling age independently of the
    ~7-km basaltic crust.  Continental mantle lithosphere is a separate thick
    root; accumulated rift extension thins that root before full breakup.
    Mixed material cells blend the unresolved continental and oceanic mantle
    endmembers by footprint fraction.
    """
    n = len(state.crust_age_myr)
    if state.continental_fraction is None:
        frac = (np.asarray(state.crust_type) == int(CrustType.CONTINENTAL)).astype(np.float64)
    else:
        frac = np.clip(np.asarray(state.continental_fraction, dtype=np.float64), 0.0, 1.0)
    total_ocean = oceanic_thermal_lithosphere_total_thickness_km(
        state.crust_age_myr,
        thermal_diffusivity_m2_s=thermal_diffusivity_m2_s,
        cooling_coefficient=cooling_coefficient,
        max_total_thickness_km=oceanic_max_total_thickness_km,
    )
    ocean_mantle = np.maximum(total_ocean - float(oceanic_crust_reference_km), 0.0)
    extension = np.zeros(n, dtype=np.float64) if state.rift_extension is None else np.clip(np.asarray(state.rift_extension, dtype=np.float64), 0.0, 1.5)
    # Extension of 1 corresponds to a mature rift in the existing model.  The
    # mantle root thins progressively but does not vanish before oceanization.
    craton_strength = (
        np.zeros(n, dtype=np.float64)
        if state.craton_strength is None
        else np.clip(np.asarray(state.craton_strength, dtype=np.float64), 0.0, 1.0)
    )
    unrifted_cont_target = (
        float(continental_mantle_thickness_km)
        + float(craton_root_thickening_km) * craton_strength
    )
    cont_target = unrifted_cont_target * (1.0 - 0.48 * np.clip(extension, 0.0, 1.0))
    cont_target = np.maximum(cont_target, float(continental_min_rifted_mantle_thickness_km))
    thickness = (1.0 - frac) * ocean_mantle + frac * cont_target

    ocean_drho = (
        float(mantle_density_kg_m3)
        * float(thermal_expansion_per_k)
        * float(mantle_temperature_contrast_k)
        * float(oceanic_mean_temperature_deficit_fraction)
    )
    continental_drho = (
        float(continental_density_anomaly_kg_m3)
        - float(craton_depletion_density_reduction_kg_m3) * craton_strength
    )
    density_anomaly = (1.0 - frac) * ocean_drho + frac * continental_drho
    return np.asarray(thickness, dtype=np.float64), np.asarray(density_anomaly, dtype=np.float64)


def refresh_mechanical_lithosphere(
    state: LithosphereState,
    dt_myr: float = 0.0,
    *,
    continental_relaxation_myr: float = 250.0,
    **kwargs,
) -> LithosphereState:
    """Refresh the explicit mantle-lithosphere layer in-place.

    Oceanic structure follows its local cooling age directly.  Continental
    roots evolve more slowly toward their target, avoiding instantaneous
    creation/destruction of a 100-km mantle root when a mixed cell crosses a
    material threshold.  Initial/legacy states (missing arrays) are initialized
    directly to the target.
    """
    target_h, target_drho = target_mantle_lithosphere_fields(state, **kwargs)
    if state.continental_fraction is None:
        frac = (np.asarray(state.crust_type) == int(CrustType.CONTINENTAL)).astype(np.float64)
    else:
        frac = np.clip(np.asarray(state.continental_fraction, dtype=np.float64), 0.0, 1.0)

    if state.mantle_lithosphere_thickness_km is None or state.mantle_lithosphere_density_anomaly_kg_m3 is None:
        state.mantle_lithosphere_thickness_km = target_h.copy()
        state.mantle_lithosphere_density_anomaly_kg_m3 = target_drho.copy()
        return state

    old_h = np.asarray(state.mantle_lithosphere_thickness_km, dtype=np.float64)
    old_drho = np.asarray(state.mantle_lithosphere_density_anomaly_kg_m3, dtype=np.float64)
    # Pure ocean is thermally age-controlled.  Continental/mixed cells keep
    # mechanical memory and relax on a geological timescale.
    alpha_c = 1.0 - np.exp(-max(float(dt_myr), 0.0) / max(float(continental_relaxation_myr), 1e-9))
    alpha = (1.0 - frac) + frac * alpha_c
    state.mantle_lithosphere_thickness_km = old_h + alpha * (target_h - old_h)
    state.mantle_lithosphere_density_anomaly_kg_m3 = old_drho + alpha * (target_drho - old_drho)
    return state


def mantle_lithosphere_negative_buoyancy_proxy(state: LithosphereState) -> Array:
    """Integrated local negative-buoyancy proxy, km*kg/m3."""
    if state.mantle_lithosphere_thickness_km is None or state.mantle_lithosphere_density_anomaly_kg_m3 is None:
        return np.zeros_like(np.asarray(state.crust_age_myr, dtype=np.float64))
    h = np.maximum(np.asarray(state.mantle_lithosphere_thickness_km, dtype=np.float64), 0.0)
    drho = np.maximum(np.asarray(state.mantle_lithosphere_density_anomaly_kg_m3, dtype=np.float64), 0.0)
    return h * drho


def initialize_lithosphere(
    mesh: SphereMesh,
    initial_system: PlateSystem,
    continental_fraction: float = 0.28,
    continental_nuclei: int = 8,
    oceanic_thickness_km: float = 7.0,
    continental_thickness_km: float = 35.0,
    initial_continental_age_myr: float = 500.0,
    radius_km: float = 5287.0,
) -> LithosphereState:
    """Create several proto-continental nuclei embedded in oceanic crust.

    Nuclei are placed on the largest distinct plates and grown as connected
    patches within those plates.  This is an initial-condition generator, not a
    demand on the final continental configuration.
    """
    if not (0.0 <= continental_fraction < 1.0):
        raise ValueError("continental_fraction must be in [0, 1)")
    n = mesh.cell_count
    plate_count = len(initial_system.plates)
    nuclei = min(max(int(continental_nuclei), 1), plate_count)
    counts = _plate_counts(initial_system)
    selected = np.argsort(counts)[::-1][:nuclei]

    target_total = int(round(continental_fraction * n))
    selected_total = int(np.sum(counts[selected]))
    quotas = np.floor(target_total * counts[selected] / max(selected_total, 1)).astype(int)
    # distribute rounding remainder
    for i in range(target_total - int(np.sum(quotas))):
        quotas[i % len(quotas)] += 1

    crust_type = np.full(n, int(CrustType.OCEANIC), dtype=np.int8)
    for plate_id, quota in zip(selected, quotas):
        seed = int(initial_system.plates[int(plate_id)].seed_cell)
        patch = _grow_patch_within_plate(
            mesh,
            int(plate_id),
            seed,
            min(int(quota), int(counts[int(plate_id)])),
            initial_system.cell_plate,
        )
        crust_type[np.asarray(patch, dtype=np.int32)] = int(CrustType.CONTINENTAL)

    age = np.zeros(n, dtype=np.float64)
    age[crust_type == int(CrustType.CONTINENTAL)] = float(initial_continental_age_myr)
    thickness = np.full(n, float(oceanic_thickness_km), dtype=np.float64)
    thickness[crust_type == int(CrustType.CONTINENTAL)] = float(continental_thickness_km)
    areas = mesh.physical_cell_areas_km2(float(radius_km))
    continental_fraction_field = (crust_type == int(CrustType.CONTINENTAL)).astype(np.float64)
    continental_volume = areas * continental_fraction_field * thickness

    return LithosphereState(
        time_myr=0.0,
        cell_plate=np.asarray(initial_system.cell_plate, dtype=np.int32).copy(),
        crust_type=crust_type,
        crust_age_myr=age,
        crust_thickness_km=thickness,
        tidal_damage=np.zeros(n, dtype=np.float64),
        rift_extension=np.zeros(n, dtype=np.float64),
        extension_age_myr=np.zeros(n, dtype=np.float64),
        collision_seam_weakness=np.zeros(n, dtype=np.float64),
        intraplate_stress=np.zeros(n, dtype=np.float64),
        supercontinent_heat=np.zeros(n, dtype=np.float64),
        continental_fraction=continental_fraction_field,
        continental_volume_km3=continental_volume,
        sediment_volume_km3=np.zeros(n, dtype=np.float64),
        mantle_lithosphere_thickness_km=None,
        mantle_lithosphere_density_anomaly_kg_m3=None,
    )


def initialize_oceanic_crust_ages(
    mesh: SphereMesh,
    state: LithosphereState,
    boundaries: list[BoundaryRecord],
    radius_km: float,
    *,
    spreading_rate_km_per_myr: float = 30.0,
    max_age_myr: float = 160.0,
    unseeded_age_myr: float = 120.0,
) -> LithosphereState:
    """Seed a mature initial ocean-age field from active divergent boundaries.

    The historical prototype initialized *all* oceanic crust at age zero.  That
    is a severe transient once bathymetry and sea level are modeled: the whole
    ocean basin cools and subsides in lockstep during the first tens of Myr.

    For a geological t=0 snapshot, oceanic cells adjacent to current divergent
    boundaries are treated as ridge-axis crust (age 0), and age grows with
    along-plate geodesic distance from those ridges using an effective half
    spreading rate.  Oceanic parts of plates with no active ridge receive a
    configurable old-ocean background age.  Continental ages are untouched.
    """
    rate=max(float(spreading_rate_km_per_myr),1.0e-9)
    max_age=max(float(max_age_myr),0.0)
    fallback=float(np.clip(unseeded_age_myr,0.0,max_age))
    ocean=np.asarray(state.crust_type,dtype=np.int8)==int(CrustType.OCEANIC)
    owner=np.asarray(state.cell_plate,dtype=np.int32)
    dist=np.full(mesh.cell_count,np.inf,dtype=np.float64)
    heap:list[tuple[float,int]]=[]
    for b in boundaries:
        if b.boundary_type != BoundaryType.DIVERGENT:
            continue
        for face in (int(b.face_a),int(b.face_b)):
            if ocean[face] and dist[face] > 0.0:
                dist[face]=0.0;heapq.heappush(heap,(0.0,face))
    while heap:
        d,cell=heapq.heappop(heap)
        if d != dist[cell]:
            continue
        pc=int(owner[cell]);c0=mesh.centroids[cell]
        for nb0 in mesh.neighbors[cell]:
            nb=int(nb0)
            if not ocean[nb] or int(owner[nb]) != pc:
                continue
            angle=float(np.arccos(np.clip(np.dot(c0,mesh.centroids[nb]),-1.0,1.0)))
            nd=d+float(radius_km)*angle
            if nd+1.0e-12 < dist[nb]:
                dist[nb]=nd;heapq.heappush(heap,(nd,nb))
    ages=np.asarray(state.crust_age_myr,dtype=np.float64).copy()
    reached=ocean & np.isfinite(dist)
    ages[reached]=np.minimum(dist[reached]/rate,max_age)
    ages[ocean & ~np.isfinite(dist)]=fallback
    state.crust_age_myr=ages
    return state


def _rift_extension_array(state: LithosphereState) -> Array:
    if state.rift_extension is None:
        return np.zeros_like(state.crust_thickness_km, dtype=np.float64)
    return np.asarray(state.rift_extension, dtype=np.float64)


def _extension_age_array(state: LithosphereState) -> Array:
    if state.extension_age_myr is None:
        return np.zeros_like(state.crust_thickness_km, dtype=np.float64)
    return np.asarray(state.extension_age_myr, dtype=np.float64)



def _optional_material_array(value: Array | None, template: Array) -> Array:
    if value is None:
        return np.zeros_like(template, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def continental_material_fields(state: LithosphereState, areas_km2: Array) -> tuple[Array, Array]:
    """Return canonical (area fraction, volume) continental material fields.

    v0.11 keeps these fields independent from discrete plate ownership.  This
    helper also provides backwards compatibility for v0.10/older states.
    """
    areas = np.asarray(areas_km2, dtype=np.float64)
    if state.continental_fraction is None:
        frac = (np.asarray(state.crust_type, dtype=np.int8) == int(CrustType.CONTINENTAL)).astype(np.float64)
    else:
        frac = np.clip(np.asarray(state.continental_fraction, dtype=np.float64), 0.0, 1.0)
    if state.continental_volume_km3 is None:
        volume = areas * frac * np.asarray(state.crust_thickness_km, dtype=np.float64)
    else:
        volume = np.maximum(np.asarray(state.continental_volume_km3, dtype=np.float64), 0.0)
    if frac.shape != areas.shape or volume.shape != areas.shape:
        raise ValueError("continental material fields must match mesh cell count")
    return frac.copy(), volume.copy()


def effective_continental_thickness_km(
    fraction: Array,
    volume_km3: Array,
    areas_km2: Array,
    *,
    eps: float = 1e-12,
) -> Array:
    """Thickness of the continental material itself, independent of coverage."""
    f = np.asarray(fraction, dtype=np.float64)
    v = np.asarray(volume_km3, dtype=np.float64)
    a = np.asarray(areas_km2, dtype=np.float64)
    out = np.zeros_like(f, dtype=np.float64)
    mask = f > float(eps)
    out[mask] = v[mask] / np.maximum(a[mask] * f[mask], float(eps))
    return out

def state_as_plate_system(state: LithosphereState, initial_system: PlateSystem) -> PlateSystem:
    return PlateSystem(cell_plate=np.asarray(state.cell_plate, dtype=np.int32), plates=initial_system.plates)


def boundary_records_for_state(
    mesh: SphereMesh,
    state: LithosphereState,
    initial_system: PlateSystem,
    radius_km: float,
    normal_threshold_km_per_myr: float,
    inactive_speed_km_per_myr: float,
) -> list[BoundaryRecord]:
    return classify_boundaries(
        mesh,
        state_as_plate_system(state, initial_system),
        radius_km,
        normal_threshold_km_per_myr,
        inactive_speed_km_per_myr,
    )


def _backward_coverage(mesh: SphereMesh, system: PlateSystem, state: LithosphereState, dt_myr: float) -> tuple[Array, Array]:
    """Return (covered, source_index) arrays with shape (plate_count, cell_count)."""
    tree = cKDTree(mesh.centroids)
    pcount = len(system.plates)
    n = mesh.cell_count
    covered = np.zeros((pcount, n), dtype=bool)
    source = np.empty((pcount, n), dtype=np.int32)
    for plate_id in range(pcount):
        pids = np.full(n, plate_id, dtype=np.int32)
        back = rotate_points_by_plate(mesh.centroids, pids, system, -float(dt_myr))
        _, src = tree.query(back, k=1, workers=-1)
        src = np.asarray(src, dtype=np.int32)
        source[plate_id] = src
        covered[plate_id] = np.asarray(state.cell_plate, dtype=np.int32)[src] == plate_id
    return covered, source


def _tidal_fields(
    mesh: SphereMesh,
    time_myr: float,
    eccentricity_history: EccentricityHistory,
    radius_km: float,
    surface_gravity_m_s2: float,
    rotation_period_hours: float,
    primary_mass_jupiter: float,
    love_h2: float,
    reference_eccentricity: float,
) -> tuple[float, Array, Array]:
    e = eccentricity_history.at(time_myr)
    strain = tidal_strain_amplitude(
        mesh.centroids,
        e,
        radius_km,
        surface_gravity_m_s2,
        rotation_period_hours,
        primary_mass_jupiter=primary_mass_jupiter,
        love_h2=love_h2,
    )
    ref = tidal_strain_amplitude(
        mesh.centroids,
        reference_eccentricity,
        radius_km,
        surface_gravity_m_s2,
        rotation_period_hours,
        primary_mass_jupiter=primary_mass_jupiter,
        love_h2=love_h2,
    )
    ref_max = max(float(np.max(ref)), 1e-30)
    weakening = np.clip(strain / ref_max, 0.0, 2.0)
    return float(e), strain, weakening



def _conserve_continental_area(
    mesh: SphereMesh,
    areas: Array,
    previous_state: LithosphereState,
    new_plate: Array,
    new_type: Array,
    new_age: Array,
    new_thickness: Array,
    new_damage: Array,
    explicit_rifted_area_km2: float,
    oceanic_thickness_km: float,
) -> float:
    """Numerically conserve continental surface area except explicit rifting.

    Semi-Lagrangian nearest-cell sampling can slowly erode or duplicate a sharp
    material interface.  Continental crust should not appear/disappear because
    of that interpolation.  This boundary-only correction restores the previous
    continental area minus explicitly rifted area.  Returned value is signed
    correction area (positive = promoted back to continent).
    """
    prev_cont = previous_state.crust_type == int(CrustType.CONTINENTAL)
    target_area = max(0.0, float(np.sum(areas[prev_cont])) - float(explicit_rifted_area_km2))
    cont = new_type == int(CrustType.CONTINENTAL)
    current_area = float(np.sum(areas[cont]))
    tolerance = 0.55 * float(np.mean(areas))
    correction = 0.0

    # Add missing continental boundary cells, preferring strong/low-damage cells.
    guard = 0
    while target_area - current_area > tolerance and guard < 12:
        candidates = []
        for cell in np.flatnonzero(new_type == int(CrustType.OCEANIC)):
            same_plate_cont = [nb for nb in mesh.neighbors[int(cell)] if new_type[nb] == int(CrustType.CONTINENTAL) and new_plate[nb] == new_plate[cell]]
            if same_plate_cont:
                candidates.append((float(new_damage[cell]), int(cell), same_plate_cont))
        if not candidates:
            break
        candidates.sort(key=lambda x: (x[0], x[1]))
        changed = False
        for _, cell, neighbors in candidates:
            if target_area - current_area <= tolerance:
                break
            if new_type[cell] == int(CrustType.CONTINENTAL):
                continue
            src = min(neighbors, key=lambda nb: new_damage[nb])
            new_type[cell] = int(CrustType.CONTINENTAL)
            new_age[cell] = float(new_age[src])
            new_thickness[cell] = float(new_thickness[src])
            current_area += float(areas[cell]); correction += float(areas[cell]); changed = True
        if not changed:
            break
        guard += 1

    # Remove numerical excess, preferring weak continental edge cells.
    guard = 0
    while current_area - target_area > tolerance and guard < 12:
        candidates = []
        for cell in np.flatnonzero(new_type == int(CrustType.CONTINENTAL)):
            ocean_nb = [nb for nb in mesh.neighbors[int(cell)] if new_type[nb] == int(CrustType.OCEANIC) and new_plate[nb] == new_plate[cell]]
            if ocean_nb:
                candidates.append((-float(new_damage[cell]), int(cell), ocean_nb))
        if not candidates:
            break
        candidates.sort(key=lambda x: (x[0], x[1]))
        changed = False
        for _, cell, neighbors in candidates:
            if current_area - target_area <= tolerance:
                break
            if new_type[cell] != int(CrustType.CONTINENTAL):
                continue
            src = min(neighbors, key=lambda nb: new_age[nb])
            new_type[cell] = int(CrustType.OCEANIC)
            new_age[cell] = float(new_age[src])
            new_thickness[cell] = float(oceanic_thickness_km)
            current_area -= float(areas[cell]); correction -= float(areas[cell]); changed = True
        if not changed:
            break
        guard += 1
    return float(correction)

def _redistribute_collision_overflow(
    mesh: SphereMesh,
    areas: Array,
    new_plate: Array,
    new_type: Array,
    new_thickness: Array,
    collision_targets: Array,
    cap_km: float,
) -> tuple[float, float, float, float]:
    """Conservatively spread grid-scale continental collision stacks.

    The preserved production-v0.10 diagnostics show that collision columns up
    to ~195 km were reduced to <=~73 km before slow geological delamination.
    Therefore redistribution must search the *connected orogenic belt* until
    the incoming excess is accommodated; an arbitrary small cell-count cap
    would strand numerical one-cell stacks and hand them to delamination.

    Returns (redistributed_volume, raw_max, post_max, unresolved_volume).
    """
    cont = new_type == int(CrustType.CONTINENTAL)
    raw_max = float(np.max(new_thickness[cont])) if np.any(cont) else 0.0
    redistributed = 0.0
    unresolved_total = 0.0
    cap = float(cap_km)
    if cap <= 0.0 or not np.any(cont):
        return 0.0, raw_max, raw_max, 0.0

    overloaded = [int(c) for c in np.asarray(collision_targets, dtype=np.int32)
                  if new_type[int(c)] == int(CrustType.CONTINENTAL)
                  and new_thickness[int(c)] > cap]
    # Process highest stacks first so the result is deterministic and the
    # largest numerical artefacts are relieved before smaller collisions.
    overloaded.sort(key=lambda c: (-float(new_thickness[c]), c))

    for src in overloaded:
        if new_type[src] != int(CrustType.CONTINENTAL):
            continue
        excess = max(float(new_thickness[src] - cap), 0.0) * float(areas[src])
        if excess <= 0.0:
            continue
        new_thickness[src] = cap
        remaining = excess
        seen = {src}
        frontier = [src]

        # Expand ring by ring through the connected continental belt. Within
        # each ring prefer the same visible surface-owner plate, then the
        # adjacent continental material from the colliding belt.
        while frontier and remaining > 1e-9:
            next_frontier: list[int] = []
            same: list[int] = []
            other: list[int] = []
            for cell in frontier:
                for nb in mesh.neighbors[cell]:
                    nb = int(nb)
                    if nb in seen:
                        continue
                    seen.add(nb)
                    if new_type[nb] != int(CrustType.CONTINENTAL):
                        continue
                    next_frontier.append(nb)
                    (same if new_plate[nb] == new_plate[src] else other).append(nb)

            for dst in sorted(same) + sorted(other):
                if remaining <= 1e-9:
                    break
                capacity = max(cap - float(new_thickness[dst]), 0.0) * float(areas[dst])
                if capacity <= 0.0:
                    continue
                moved = min(capacity, remaining)
                new_thickness[dst] += moved / float(areas[dst])
                remaining -= moved
                redistributed += moved
            frontier = next_frontier

        if remaining > 1e-9:
            # True physical/local capacity exhaustion is allowed, but the
            # excess is never deleted. It remains in the source column for the
            # slower geological collapse/delamination physics.
            new_thickness[src] += remaining / float(areas[src])
            unresolved_total += remaining

    post_max = float(np.max(new_thickness[cont])) if np.any(cont) else 0.0
    return float(redistributed), raw_max, post_max, float(unresolved_total)


def _redistribute_continental_footprint_overflow(
    mesh: SphereMesh,
    areas: Array,
    fraction: Array,
    volume: Array,
    preferred_sources: Array,
) -> tuple[float, float, float, Array]:
    """Spread >100% continental footprint locally without losing area/volume.

    The discrete plate raster may map material from two plates onto one target
    during convergence.  Plate ownership still selects one visible owner, but
    continental *material* is not allowed to collapse into an arbitrarily tall
    one-cell stack.  Equivalent continental area above one cell footprint is
    moved ring-by-ring to nearby cells with free material capacity, carrying a
    proportional share of crust volume with it.

    Returns ``(moved_volume, raw_stack_max_km, post_max_km, donor_cell)``.
    ``donor_cell`` records the first overloaded source that materially filled a
    target and is used only to initialise legacy per-cell memory when a cell
    crosses the visible-continent threshold.
    """
    frac = np.asarray(fraction, dtype=np.float64)
    vol = np.asarray(volume, dtype=np.float64)
    area = np.asarray(areas, dtype=np.float64)
    donor = np.full(mesh.cell_count, -1, dtype=np.int32)
    # What the old one-cell representation would have interpreted as crustal
    # thickness before footprint spreading.  This is retained as a diagnostic.
    raw_stack = np.zeros(mesh.cell_count, dtype=np.float64)
    has = vol > 0.0
    raw_stack[has] = vol[has] / np.maximum(area[has], 1e-30)
    raw_max = float(np.max(raw_stack)) if np.any(has) else 0.0

    moved_volume = 0.0
    preferred = {int(x) for x in np.asarray(preferred_sources, dtype=np.int32).ravel()}
    overloaded = [int(i) for i in np.flatnonzero(frac > 1.0 + 1e-12)]
    overloaded.sort(key=lambda i: (0 if i in preferred else 1, -float(frac[i]), i))

    for src in overloaded:
        equiv_area = float(frac[src] * area[src])
        if equiv_area <= area[src] + 1e-9 or vol[src] <= 0.0:
            frac[src] = min(float(frac[src]), 1.0)
            continue
        # Transport the overflow with the mean thickness of the material that
        # arrived in this cell.  Thus both equivalent footprint and volume are
        # conserved exactly, while thickness is not created numerically.
        material_h = float(vol[src] / max(equiv_area, 1e-30))
        overflow_area = equiv_area - float(area[src])
        overflow_volume = overflow_area * material_h
        frac[src] = 1.0
        vol[src] -= overflow_volume

        remaining_area = overflow_area
        remaining_volume = overflow_volume
        seen = {src}
        frontier = [src]
        while frontier and remaining_area > 1e-9:
            next_frontier: list[int] = []
            candidates: list[int] = []
            for cell in frontier:
                for nb in mesh.neighbors[cell]:
                    nb = int(nb)
                    if nb in seen:
                        continue
                    seen.add(nb)
                    next_frontier.append(nb)
                    if frac[nb] < 1.0 - 1e-12:
                        candidates.append(nb)
            # Prefer cells already carrying some continental material so the
            # correction follows the collisional/orogenic belt rather than
            # jumping immediately into unrelated oceanic interiors.
            candidates.sort(key=lambda x: (0 if frac[x] > 1e-12 else 1, -float(frac[x]), x))
            for dst in candidates:
                if remaining_area <= 1e-9:
                    break
                capacity = max((1.0 - float(frac[dst])) * float(area[dst]), 0.0)
                if capacity <= 0.0:
                    continue
                da = min(capacity, remaining_area)
                dv = da * material_h
                frac[dst] += da / float(area[dst])
                vol[dst] += dv
                remaining_area -= da
                remaining_volume -= dv
                moved_volume += dv
                if donor[dst] < 0:
                    donor[dst] = int(src)
            frontier = next_frontier

        # Global continental coverage is far below 100% in intended runs, so
        # local BFS should find capacity.  If a pathological state cannot, do
        # not delete material: restore unresolved material to the source.  The
        # fraction may then remain >1, making the failure explicit in diagnostics.
        if remaining_area > 1e-8:
            frac[src] += remaining_area / float(area[src])
            vol[src] += remaining_volume

    eff = effective_continental_thickness_km(frac, vol, area)
    post_max = float(np.max(eff[frac > 1e-12])) if np.any(frac > 1e-12) else 0.0
    return float(moved_volume), raw_max, post_max, donor


def advance_lithosphere(
    mesh: SphereMesh,
    initial_system: PlateSystem,
    state: LithosphereState,
    dt_myr: float,
    radius_km: float,
    surface_gravity_m_s2: float,
    rotation_period_hours: float,
    eccentricity_history: EccentricityHistory,
    *,
    primary_mass_jupiter: float = 1.0,
    love_h2: float = 0.6,
    reference_eccentricity: float = 0.00047,
    oceanic_thickness_km: float = 7.0,
    continental_thickness_km: float = 35.0,
    max_continental_thickness_km: float = 75.0,
    collision_accretion_fraction: float = 0.18,
    tidal_damage_rate_per_myr: float = 0.006,
    tidal_damage_relaxation_myr: float = 300.0,
    tidal_damage_background_fraction: float = 0.10,
    continental_extension_rate_per_myr: float = 0.018,
    continental_extension_relaxation_myr: float = 90.0,
    continental_extension_min_duration_myr: float = 24.0,
    continental_rift_extension_threshold: float = 0.75,
    continental_thinning_km_per_myr: float = 0.18,
    continental_min_breakup_thickness_km: float = 20.0,
    continental_breakup_min_extension_forcing: float = 0.75,
    continental_extension_requires_two_plate_flanks: bool = True,
    tidal_thinning_boost_max_fraction: float = 0.30,
    continental_extension_suppression: Array | None = None,
    continental_extension_external_forcing: Array | None = None,
    craton_extension_resistance_gain: float = 0.0,
    craton_min_extension_factor: float = 0.28,
    transport_state=None,
    transport_parameters=None,
) -> tuple[LithosphereState, Array, Array, LithosphereStepDiagnostics]:
    if dt_myr <= 0.0:
        raise ValueError("dt_myr must be positive")
    n = mesh.cell_count
    areas = mesh.physical_cell_areas_km2(radius_km)
    transport_diag = None
    conservative_transport = transport_state is not None
    if conservative_transport:
        # Local import avoids a module cycle: transport needs LithosphereState.
        from .transport import build_transport_map
        tmap = build_transport_map(
            mesh, initial_system, state, dt_myr, transport_state, transport_parameters
        )
        covered, source = tmap.covered, tmap.source
        transport_diag = tmap.diagnostics
    else:
        covered, source = _backward_coverage(mesh, initial_system, state, dt_myr)
    multiplicity = np.sum(covered, axis=0)
    gap_mask = multiplicity == 0
    overlap_mask = multiplicity > 1

    new_plate = np.empty(n, dtype=np.int32)
    new_type = np.empty(n, dtype=np.int8)
    new_age = np.empty(n, dtype=np.float64)
    new_thickness = np.empty(n, dtype=np.float64)
    new_damage = np.empty(n, dtype=np.float64)
    new_extension = np.empty(n, dtype=np.float64)
    new_extension_age = np.empty(n, dtype=np.float64)
    new_seam = np.empty(n, dtype=np.float64)
    new_stress = np.empty(n, dtype=np.float64)
    new_superheat = np.empty(n, dtype=np.float64)
    track_craton_memory = (
        state.continental_lithosphere_age_myr is not None
        and state.mantle_depletion_fraction is not None
        and state.craton_strength is not None
    )
    old_cont_lith_age = None if not track_craton_memory else np.asarray(state.continental_lithosphere_age_myr, dtype=np.float64)
    old_mantle_depletion = None if not track_craton_memory else np.asarray(state.mantle_depletion_fraction, dtype=np.float64)
    old_craton_strength = None if not track_craton_memory else np.asarray(state.craton_strength, dtype=np.float64)
    new_cont_lith_age = None if not track_craton_memory else np.empty(n, dtype=np.float64)
    new_mantle_depletion = None if not track_craton_memory else np.empty(n, dtype=np.float64)
    new_craton_strength = None if not track_craton_memory else np.empty(n, dtype=np.float64)
    track_mechanical_lithosphere = (
        state.mantle_lithosphere_thickness_km is not None
        and state.mantle_lithosphere_density_anomaly_kg_m3 is not None
    )
    old_mantle_lith_h = None if not track_mechanical_lithosphere else np.asarray(state.mantle_lithosphere_thickness_km, dtype=np.float64)
    old_mantle_lith_drho = None if not track_mechanical_lithosphere else np.asarray(state.mantle_lithosphere_density_anomaly_kg_m3, dtype=np.float64)
    new_mantle_lith_h = None if not track_mechanical_lithosphere else np.empty(n, dtype=np.float64)
    new_mantle_lith_drho = None if not track_mechanical_lithosphere else np.empty(n, dtype=np.float64)
    material_source_index = np.full(n, -1, dtype=np.int32)
    collision_targets: list[int] = []
    old_extension = _rift_extension_array(state)
    old_extension_age = _extension_age_array(state)
    old_seam = _optional_material_array(state.collision_seam_weakness, state.crust_thickness_km)
    old_stress = _optional_material_array(state.intraplate_stress, state.crust_thickness_km)
    old_superheat = _optional_material_array(state.supercontinent_heat, state.crust_thickness_km)
    old_cont_fraction, old_cont_volume = continental_material_fields(state, areas)
    new_cont_fraction = np.zeros(n, dtype=np.float64)
    new_cont_volume = np.zeros(n, dtype=np.float64)

    created_area = float(np.sum(areas[gap_mask]))
    subducted_ocean = 0.0
    collision_area = 0.0

    # Resolve all targets covered by one or more rigidly moved plates.
    for target in np.flatnonzero(~gap_mask):
        plates = np.flatnonzero(covered[:, target]).astype(np.int32)
        src = source[plates, target]
        ctype = state.crust_type[src]

        # Continental material is transported independently of which parcel
        # wins visible surface ownership.  Before local footprint spreading the
        # fraction may exceed 1 at an inter-plate overlap; that is intentional
        # temporary bookkeeping, not a physical cell state.
        incoming_cont_area = old_cont_fraction[src] * areas[src]
        new_cont_fraction[target] = float(np.sum(incoming_cont_area) / areas[target])
        new_cont_volume[target] = float(np.sum(old_cont_volume[src]))

        continents = np.flatnonzero(ctype == int(CrustType.CONTINENTAL))
        if len(continents):
            # Continental lithosphere remains buoyant relative to oceanic crust.
            continent_src = src[continents]
            continent_plates = plates[continents]
            # Conservative v0.10 parcel columns account for non-equal cell area.
            if conservative_transport:
                incoming_h = state.crust_thickness_km[continent_src] * areas[continent_src] / areas[target]
            else:
                incoming_h = state.crust_thickness_km[continent_src]
            # Stronger (less damaged), then thicker continent wins the visible
            # surface owner. In conservative mode every colliding continental
            # column still contributes its full A*h volume below that surface.
            order = np.lexsort((continent_plates, -incoming_h, state.tidal_damage[continent_src]))
            ci = int(continents[order[0]])
            winner_src = int(src[ci])
            winner_plate = int(plates[ci])

            ocean_losers = src[ctype == int(CrustType.OCEANIC)]
            if len(ocean_losers):
                subducted_ocean += float(areas[target] * len(ocean_losers))

            other_continents = continent_src[order[1:]]
            if len(other_continents):
                collision_area += float(areas[target] * len(other_continents))
                collision_targets.append(int(target))

            new_plate[target] = winner_plate
            new_type[target] = int(CrustType.CONTINENTAL)
            new_age[target] = float(state.crust_age_myr[winner_src] + dt_myr)
            if conservative_transport:
                # Visible thickness is provisional.  Continental material
                # footprint/volume are resolved *after* all surface owners are
                # selected, preventing multiple parcels from becoming one
                # artificial 100-300 km tower.
                wf = max(float(old_cont_fraction[winner_src]), 1e-12)
                new_thickness[target] = float(old_cont_volume[winner_src] / max(areas[winner_src] * wf, 1e-30))
            else:
                extra_thickness = float(
                    collision_accretion_fraction * np.sum(state.crust_thickness_km[other_continents])
                ) if len(other_continents) else 0.0
                new_thickness[target] = min(
                    float(max_continental_thickness_km),
                    float(state.crust_thickness_km[winner_src] + extra_thickness),
                )
            new_damage[target] = float(state.tidal_damage[winner_src])
            new_extension[target] = float(old_extension[winner_src])
            new_extension_age[target] = float(old_extension_age[winner_src])
            new_seam[target] = float(old_seam[winner_src])
            new_stress[target] = float(old_stress[winner_src])
            new_superheat[target] = float(old_superheat[winner_src])
            if track_craton_memory:
                new_cont_lith_age[target] = float(old_cont_lith_age[winner_src])
                new_mantle_depletion[target] = float(old_mantle_depletion[winner_src])
                new_craton_strength[target] = float(old_craton_strength[winner_src])
            if track_mechanical_lithosphere:
                new_mantle_lith_h[target] = float(old_mantle_lith_h[winner_src])
                new_mantle_lith_drho[target] = float(old_mantle_lith_drho[winner_src])
            material_source_index[target] = winner_src
        else:
            # Ocean-ocean overlap: preferentially consume older oceanic crust.
            ages = state.crust_age_myr[src]
            order = np.lexsort((plates, ages))  # youngest first
            winner = int(order[0])
            winner_src = int(src[winner])
            new_plate[target] = int(plates[winner])
            new_type[target] = int(CrustType.OCEANIC)
            new_age[target] = float(state.crust_age_myr[winner_src] + dt_myr)
            new_thickness[target] = (
                float(state.crust_thickness_km[winner_src] * areas[winner_src] / areas[target])
                if conservative_transport else float(oceanic_thickness_km)
            )
            new_damage[target] = float(state.tidal_damage[winner_src])
            new_extension[target] = float(old_extension[winner_src])
            new_extension_age[target] = float(old_extension_age[winner_src])
            new_seam[target] = float(old_seam[winner_src])
            new_stress[target] = float(old_stress[winner_src])
            new_superheat[target] = float(old_superheat[winner_src])
            if track_craton_memory:
                new_cont_lith_age[target] = float(old_cont_lith_age[winner_src])
                new_mantle_depletion[target] = float(old_mantle_depletion[winner_src])
                new_craton_strength[target] = float(old_craton_strength[winner_src])
            if track_mechanical_lithosphere:
                new_mantle_lith_h[target] = float(old_mantle_lith_h[winner_src])
                new_mantle_lith_drho[target] = float(old_mantle_lith_drho[winner_src])
            material_source_index[target] = winner_src
            if len(order) > 1:
                subducted_ocean += float(areas[target] * (len(order) - 1))

    # True divergent gaps receive newborn oceanic crust.
    if np.any(gap_mask):
        tree = cKDTree(mesh.centroids[~gap_mask])
        _, near_local = tree.query(mesh.centroids[gap_mask], k=1, workers=-1)
        survivor_cells = np.flatnonzero(~gap_mask)
        near = survivor_cells[np.asarray(near_local, dtype=np.int32)]
        new_plate[gap_mask] = new_plate[near]
        new_type[gap_mask] = int(CrustType.OCEANIC)
        new_age[gap_mask] = 0.0
        new_thickness[gap_mask] = float(oceanic_thickness_km)
        new_damage[gap_mask] = 0.0
        new_extension[gap_mask] = 0.0
        new_extension_age[gap_mask] = 0.0
        new_seam[gap_mask] = 0.0
        new_stress[gap_mask] = 0.0
        new_superheat[gap_mask] = 0.0
        if track_craton_memory:
            new_cont_lith_age[gap_mask] = 0.0
            new_mantle_depletion[gap_mask] = 0.0
            new_craton_strength[gap_mask] = 0.0
        if track_mechanical_lithosphere:
            # Newly opened ridge-axis ocean has essentially no cold mantle
            # lithosphere yet. Density anomaly is carried as a thermal
            # endmember, but integrated negative buoyancy is zero because h=0.
            new_mantle_lith_h[gap_mask] = 0.0
            new_mantle_lith_drho[gap_mask] = float(3300.0 * 3.0e-5 * 1300.0 * 0.50)
        new_cont_fraction[gap_mask] = 0.0
        new_cont_volume[gap_mask] = 0.0
        material_source_index[gap_mask] = -1

    collision_redistributed = 0.0
    collision_raw_max = float(np.max(new_thickness[new_type == int(CrustType.CONTINENTAL)])) if np.any(new_type == int(CrustType.CONTINENTAL)) else 0.0
    collision_post_max = collision_raw_max
    collision_unresolved = 0.0
    transport_volume_error = 0.0
    material_area_error = 0.0
    if conservative_transport:
        before_transport_area = float(np.sum(areas * old_cont_fraction))
        before_transport_volume = float(np.sum(old_cont_volume))
        collision_redistributed, collision_raw_max, collision_post_max, donor = _redistribute_continental_footprint_overflow(
            mesh, areas, new_cont_fraction, new_cont_volume,
            np.asarray(collision_targets, dtype=np.int32),
        )
        # Rebuild the *visible* crust raster from the independent material
        # layer.  Fractional fringe cells remain material-bearing without
        # forcing the entire plate-ownership raster to become continental.
        was_visible = new_type == int(CrustType.CONTINENTAL)
        visible_cont = new_cont_fraction >= 0.5
        became_visible = visible_cont & ~was_visible
        for dst in np.flatnonzero(became_visible):
            src_meta = int(donor[dst])
            if src_meta >= 0:
                new_age[dst] = float(new_age[src_meta])
                new_damage[dst] = float(new_damage[src_meta])
                new_extension[dst] = float(new_extension[src_meta])
                new_extension_age[dst] = float(new_extension_age[src_meta])
                new_seam[dst] = float(new_seam[src_meta])
                new_stress[dst] = float(new_stress[src_meta])
                new_superheat[dst] = float(new_superheat[src_meta])
                if track_craton_memory:
                    new_cont_lith_age[dst] = float(new_cont_lith_age[src_meta])
                    new_mantle_depletion[dst] = float(new_mantle_depletion[src_meta])
                    new_craton_strength[dst] = float(new_craton_strength[src_meta])
        new_type[:] = int(CrustType.OCEANIC)
        new_type[visible_cont] = int(CrustType.CONTINENTAL)
        effective_h = effective_continental_thickness_km(new_cont_fraction, new_cont_volume, areas)
        new_thickness[~visible_cont] = float(oceanic_thickness_km)
        new_thickness[visible_cont] = effective_h[visible_cont]

        after_transport_area = float(np.sum(areas * new_cont_fraction))
        after_transport_volume = float(np.sum(new_cont_volume))
        transport_volume_error = after_transport_volume - before_transport_volume
        material_area_error = after_transport_area - before_transport_area
        # Any unresolved >1 coverage is reported rather than silently clipped.
        overflow_area = np.maximum(new_cont_fraction - 1.0, 0.0) * areas
        if np.any(overflow_area > 1e-8):
            h = effective_continental_thickness_km(new_cont_fraction, new_cont_volume, areas)
            collision_unresolved = float(np.sum(overflow_area * h))
    else:
        # Legacy path: initialise independent fields from the resulting binary
        # raster.  This keeps old scripts/tests working while v0.11 production
        # runs use conservative material transport.
        new_cont_fraction = (new_type == int(CrustType.CONTINENTAL)).astype(np.float64)
        new_cont_volume = areas * new_cont_fraction * new_thickness

    # v0.9.2: identify *actual tectonic extension* first.  Tides alone are not
    # allowed to break continents.  A continental cell is actively extending
    # only when it lies directly beside a newly opened kinematic gap.
    extension_forcing = np.zeros(n, dtype=np.float64)
    direct_cells: set[int] = set()
    passive_margin_gap_cells = 0
    active_continental_rift_gap_cells = 0
    for gap_cell in np.flatnonzero(gap_mask):
        continental_neighbors = [
            int(nb) for nb in mesh.neighbors[int(gap_cell)]
            if new_type[int(nb)] == int(CrustType.CONTINENTAL)
        ]
        if not continental_neighbors:
            continue
        # v0.9.7: an opening gap is a *continental rift* only while it is
        # actually flanked by continental material belonging to two distinct
        # diverging plates.  Once an oceanic axial strip exists, subsequent
        # spreading gaps are oceanic-spreading gaps and must not march the
        # breakup front cell-by-cell into a passive continental margin.
        if continental_extension_requires_two_plate_flanks:
            flank_plates = {int(new_plate[nb]) for nb in continental_neighbors}
            if len(flank_plates) < 2:
                passive_margin_gap_cells += 1
                continue
        active_continental_rift_gap_cells += 1
        for nb in continental_neighbors:
            direct_cells.add(nb)
            extension_forcing[nb] = 1.0
    # Continental rifts are zones, not mathematical lines.  Propagate a weaker
    # extensional field one additional cell into the same continental plate.
    # This is still causally tied to a real opening gap; tides cannot create it.
    for cell in sorted(direct_cells):
        for nb in mesh.neighbors[cell]:
            if (
                new_type[nb] == int(CrustType.CONTINENTAL)
                and new_plate[nb] == new_plate[cell]
                and extension_forcing[nb] < 0.35
            ):
                extension_forcing[nb] = 0.35
    extension_forcing = np.clip(extension_forcing, 0.0, 1.0)
    if continental_extension_external_forcing is not None:
        external=np.asarray(continental_extension_external_forcing,dtype=np.float64)
        if external.shape != (n,):
            raise ValueError("continental_extension_external_forcing must have shape (cell_count,)")
        # Rollback is genuine tectonic extension, unlike the tidal damage field.
        extension_forcing=np.maximum(extension_forcing,np.clip(external,0.0,1.0))
    if continental_extension_suppression is not None:
        suppression = np.asarray(continental_extension_suppression, dtype=np.float64)
        if suppression.shape != (n,):
            raise ValueError("continental_extension_suppression must have shape (cell_count,)")
        extension_forcing *= 1.0 - np.clip(suppression, 0.0, 1.0)
    if track_craton_memory and float(craton_extension_resistance_gain) > 0.0:
        # Strong depleted roots redirect extension toward juvenile/orogenic
        # belts.  The floor keeps sufficiently large external forcing capable
        # of eventually rejuvenating and rupturing a craton.
        craton_factor = np.clip(
            1.0 - float(craton_extension_resistance_gain) * np.clip(new_craton_strength, 0.0, 1.0),
            float(craton_min_extension_factor),
            1.0,
        )
        extension_forcing *= craton_factor

    # Fast orbital tide is averaged into a slow fatigue/weakening variable.
    new_time = float(state.time_myr + dt_myr)
    e, strain, weakening = _tidal_fields(
        mesh,
        new_time,
        eccentricity_history,
        radius_km,
        surface_gravity_m_s2,
        rotation_period_hours,
        primary_mass_jupiter,
        love_h2,
        reference_eccentricity,
    )
    decay = np.exp(-float(dt_myr) / max(float(tidal_damage_relaxation_myr), 1e-9))
    # Tidal fatigue still exists globally, but most accumulation requires a
    # contemporaneous tectonic stress field.  Thus a saturated tidal map is a
    # weakness map, not a hidden breakup timer.
    stress_coupling = np.clip(
        float(tidal_damage_background_fraction)
        + (1.0 - float(tidal_damage_background_fraction)) * extension_forcing,
        0.0, 1.0,
    )
    new_damage = np.clip(
        new_damage * decay
        + float(tidal_damage_rate_per_myr) * weakening * stress_coupling * float(dt_myr),
        0.0, 1.0,
    )

    # Progressive continental rifting.  Extension has its own memory and crust
    # thins gradually under sustained opening.  Tidal damage can accelerate the
    # thinning/extension rate by at most a configurable fraction; it cannot
    # create extension where kinematics provide none.
    ext_decay = np.exp(-float(dt_myr) / max(float(continental_extension_relaxation_myr), 1e-9))
    active_extension = (extension_forcing > 0.0) & (new_type == int(CrustType.CONTINENTAL))
    inactive = ~active_extension
    new_extension[inactive] *= ext_decay
    new_extension_age[inactive] *= ext_decay

    tide_boost = 1.0 + float(tidal_thinning_boost_max_fraction) * np.clip(new_damage, 0.0, 1.0)
    if np.any(active_extension):
        idx = np.flatnonzero(active_extension)
        forcing = extension_forcing[idx]
        new_extension[idx] += (
            float(continental_extension_rate_per_myr) * forcing * tide_boost[idx] * float(dt_myr)
        )
        new_extension_age[idx] += forcing * float(dt_myr)
        requested_thinning = (
            float(continental_thinning_km_per_myr) * forcing * tide_boost[idx] * float(dt_myr)
        )
        before_h = new_thickness[idx].copy()
        new_thickness[idx] = np.maximum(
            float(continental_min_breakup_thickness_km) * 0.75,
            before_h - requested_thinning,
        )
        thinning = before_h - new_thickness[idx]
        # Keep the independent material volume consistent with the explicit
        # physical thinning rule.  Fractional edge cells lose only the volume
        # corresponding to their continental footprint.
        new_cont_volume[idx] = np.maximum(
            0.0,
            new_cont_volume[idx] - areas[idx] * new_cont_fraction[idx] * thinning,
        )
    else:
        idx = np.empty(0, dtype=np.int32)
        thinning = np.empty(0, dtype=np.float64)

    # Record tracked continental material loss before breakup can zero the
    # footprint fraction/material volume on the same step.
    continental_thinning_volume = (
        float(np.sum(areas[idx] * new_cont_fraction[idx] * thinning)) if len(idx) else 0.0
    )

    # True breakup requires all three conditions: sustained real extension,
    # enough accumulated extensional strain, and crust that has physically
    # thinned close to breakup thickness.  Damage by itself is insufficient.
    breakup = (
        (new_type == int(CrustType.CONTINENTAL))
        & (new_extension_age >= float(continental_extension_min_duration_myr))
        & (new_extension >= float(continental_rift_extension_threshold))
        & (new_thickness <= float(continental_min_breakup_thickness_km))
        # Broad continental rifts can remain thinned/failed rifts.  Full
        # oceanization is restricted to the axial high-extension core rather
        # than converting the entire one-ring damage halo into oceanic crust.
        & (extension_forcing >= float(continental_breakup_min_extension_forcing))
    )
    rifted_area = float(np.sum(areas[breakup] * new_cont_fraction[breakup]))
    breakup_recycled_volume = float(np.sum(new_cont_volume[breakup])) if np.any(breakup) else 0.0
    if np.any(breakup):
        new_type[breakup] = int(CrustType.OCEANIC)
        new_age[breakup] = 0.0
        new_thickness[breakup] = float(oceanic_thickness_km)
        new_cont_fraction[breakup] = 0.0
        new_cont_volume[breakup] = 0.0
        # Preserve a strong post-rift marker for v0.7 topology splitting, while
        # resetting the active continental extension timer.
        new_extension[breakup] = np.maximum(new_extension[breakup], 1.0)
        new_extension_age[breakup] = 0.0
        new_damage[breakup] *= 0.35
        if track_craton_memory:
            new_cont_lith_age[breakup] = 0.0
            new_mantle_depletion[breakup] = 0.0
            new_craton_strength[breakup] = 0.0
        if track_mechanical_lithosphere:
            # Oceanization represents rupture/thermal replacement at the rift
            # axis: the old continental mantle root is no longer retained as a
            # cold coherent plate column at the newborn spreading center.
            new_mantle_lith_h[breakup] = 0.0
            new_mantle_lith_drho[breakup] = float(3300.0 * 3.0e-5 * 1300.0 * 0.50)

    if conservative_transport:
        # The one-to-one remap preserves each discrete parcel footprint by
        # construction. Inter-plate overlaps/gaps are physical convergence /
        # divergence signals, so do not apply the old post-hoc area repair.
        area_correction = 0.0
    else:
        area_correction = _conserve_continental_area(
            mesh, areas, state, new_plate, new_type, new_age, new_thickness, new_damage,
            rifted_area, float(oceanic_thickness_km),
        )
        new_cont_fraction = (new_type == int(CrustType.CONTINENTAL)).astype(np.float64)
        new_cont_volume = areas * new_cont_fraction * new_thickness

    # The visible legacy raster is derived from the independent material layer
    # after all explicit lithosphere processes in this step.  Fractional fringe
    # material is retained even when the visible cell remains oceanic.
    if conservative_transport:
        visible_cont = new_cont_fraction >= 0.5
        new_type[:] = int(CrustType.OCEANIC)
        new_type[visible_cont] = int(CrustType.CONTINENTAL)
        effective_h = effective_continental_thickness_km(new_cont_fraction, new_cont_volume, areas)
        new_thickness[~visible_cont] = float(oceanic_thickness_km)
        new_thickness[visible_cont] = effective_h[visible_cont]

    new_state = LithosphereState(
        time_myr=new_time,
        cell_plate=new_plate,
        crust_type=new_type,
        crust_age_myr=new_age,
        crust_thickness_km=new_thickness,
        tidal_damage=new_damage,
        rift_extension=new_extension,
        extension_age_myr=new_extension_age,
        collision_seam_weakness=new_seam,
        intraplate_stress=new_stress,
        supercontinent_heat=new_superheat,
        continental_fraction=new_cont_fraction,
        continental_volume_km3=new_cont_volume,
        mantle_lithosphere_thickness_km=None if not track_mechanical_lithosphere else new_mantle_lith_h,
        mantle_lithosphere_density_anomaly_kg_m3=None if not track_mechanical_lithosphere else new_mantle_lith_drho,
        sediment_volume_km3=(None if state.sediment_volume_km3 is None else np.zeros(n, dtype=np.float64)),
        continental_lithosphere_age_myr=None if not track_craton_memory else new_cont_lith_age,
        mantle_depletion_fraction=None if not track_craton_memory else new_mantle_depletion,
        craton_strength=None if not track_craton_memory else new_craton_strength,
    )

    ocean_mask = new_type == int(CrustType.OCEANIC)
    cont_mask = ~ocean_mask
    max_disp = float(np.max(strain) * radius_km * 1000.0) if len(strain) else 0.0
    diag = LithosphereStepDiagnostics(
        time_myr=new_time,
        dt_myr=float(dt_myr),
        eccentricity=e,
        max_tidal_strain=float(np.max(strain)) if len(strain) else 0.0,
        max_radial_displacement_m=max_disp,
        gap_fraction=float(np.mean(gap_mask)),
        overlap_fraction=float(np.mean(overlap_mask)),
        created_oceanic_area_km2=created_area,
        subducted_oceanic_area_km2=float(subducted_ocean),
        continental_collision_area_km2=float(collision_area),
        tidally_rifted_continental_area_km2=float(rifted_area),
        numerical_continental_area_correction_km2=float(area_correction),
        continental_area_fraction=float(np.sum(areas * new_cont_fraction) / np.sum(areas)),
        mean_oceanic_age_myr=float(np.mean(new_age[ocean_mask])) if np.any(ocean_mask) else 0.0,
        max_oceanic_age_myr=float(np.max(new_age[ocean_mask])) if np.any(ocean_mask) else 0.0,
        mean_tidal_damage=float(np.mean(new_damage)),
        visible_continental_area_fraction=float(np.sum(areas[cont_mask]) / np.sum(areas)),
        continental_material_area_error_km2=float(material_area_error),
        actively_extending_continental_area_km2=float(np.sum(areas[active_extension] * new_cont_fraction[active_extension])),
        continental_thinning_volume_km3=float(continental_thinning_volume),
        continental_breakup_recycled_volume_km3=float(breakup_recycled_volume),
        rift_recycled_volume_km3=float(continental_thinning_volume + breakup_recycled_volume),
        active_continental_rift_gap_cells=int(active_continental_rift_gap_cells),
        passive_margin_gap_cells_suppressed=int(passive_margin_gap_cells),
        mean_rift_extension_continental=float(np.mean(new_extension[cont_mask])) if np.any(cont_mask) else 0.0,
        max_rift_extension=float(np.max(new_extension)) if len(new_extension) else 0.0,
        max_extension_age_myr=float(np.max(new_extension_age)) if len(new_extension_age) else 0.0,
        conservative_transport_commits=int(transport_diag.committed_plates) if transport_diag is not None else 0,
        transport_cumulative_commit_count=int(transport_diag.cumulative_commit_count) if transport_diag is not None else 0,
        transport_mean_residual_angle_deg=float(transport_diag.mean_residual_angle_deg) if transport_diag is not None else 0.0,
        transport_max_residual_angle_deg=float(transport_diag.max_residual_angle_deg) if transport_diag is not None else 0.0,
        transport_max_hold_age_myr=float(transport_diag.max_hold_age_myr) if transport_diag is not None else 0.0,
        conservative_transport_volume_error_km3=float(transport_volume_error),
        collision_overflow_redistributed_volume_km3=float(collision_redistributed),
        collision_raw_max_thickness_km=float(collision_raw_max),
        collision_post_redistribution_max_thickness_km=float(collision_post_max),
        collision_overflow_unresolved_volume_km3=float(collision_unresolved),
        material_source_index=material_source_index,
    )
    return new_state, strain, weakening, diag


__all__ = [
    "CrustType",
    "LithosphereState",
    "LithosphereStepDiagnostics",
    "LithosphereSnapshot",
    "initialize_lithosphere",
    "oceanic_thermal_lithosphere_total_thickness_km",
    "target_mantle_lithosphere_fields",
    "refresh_mechanical_lithosphere",
    "mantle_lithosphere_negative_buoyancy_proxy",
    "state_as_plate_system",
    "boundary_records_for_state",
    "advance_lithosphere",
]
