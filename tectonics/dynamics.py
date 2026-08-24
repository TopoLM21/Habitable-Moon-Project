"""v0.5 effective force-driven plate kinematics.

This is deliberately an *effective* quasi-static plate-force model, not a
continuum mantle solver.  It replaces the fixed Euler poles of v0.1-v0.4 with
Euler angular-velocity vectors that evolve in response to boundary processes:

- slab pull at convergent oceanic margins;
- ridge push at divergent margins;
- resistance at continent-continent collisions and transforms;
- mantle drag / memory of a slowly varying background flow;
- tidal damage reducing lithospheric boundary resistance.

The force coefficients are calibrated prototype parameters.  They are not
literal SI forces.  Direction, geometry, crust type and relative weighting are
physical; the mapping from those proxies to deg/Myr is intentionally explicit
and configurable.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .kinematics import BoundaryRecord, BoundaryType, classify_boundaries
from .lithosphere import CrustType, LithosphereState, mantle_lithosphere_negative_buoyancy_proxy
from .mesh import SphereMesh
from .plates import Plate, PlateSystem
from .mantle import MantleFlowState, mantle_flow_rms_rad_per_myr, plate_mean_mantle_omega
from .subduction_memory import SubductionMemoryState, SubductionMemoryParameters, residual_pull_by_plate
from .breakoff import slab_pull_multiplier_for_pair

Array = np.ndarray


@dataclass(slots=True)
class DynamicsParameters:
    slab_pull_weight: float = 1.25
    ridge_push_weight: float = 0.70
    # v0.17 thermal/GPE ridge push.  In a half-space cooling model the
    # ridge-driving line force grows approximately linearly with seafloor age.
    # Since mantle-lithosphere thickness H grows ~sqrt(age), the first-moment
    # GPE proxy drho*H^2 has the desired age scaling while reusing v0.16 fields.
    ridge_gpe_reference_mantle_thickness_km: float = 93.5
    ridge_gpe_reference_density_anomaly_kg_m3: float = 64.0
    ridge_gpe_saturation_ratio: float = 0.20
    ridge_gpe_exponent: float = 1.0
    ridge_gpe_calibration_gain: float = 1.014
    ridge_gpe_min_factor: float = 0.20
    ridge_gpe_max_factor: float = 2.40
    slab_age_reference_myr: float = 80.0
    slab_age_gain: float = 0.90
    # v0.16 local mantle-lithosphere slab buoyancy.  When the explicit fields
    # are available these replace crust-age/global-thickness double counting.
    slab_buoyancy_reference_thickness_km: float = 100.0
    slab_buoyancy_reference_density_anomaly_kg_m3: float = 60.0
    slab_buoyancy_exponent: float = 0.75
    slab_buoyancy_calibration_gain: float = 1.85
    continental_collision_resistance: float = 2.5
    transform_resistance: float = 0.55
    tidal_resistance_reduction: float = 0.65
    tidal_ridge_enhancement: float = 0.35
    force_speed_scale_deg_per_myr: float = 0.85
    mantle_memory_fraction: float = 0.22
    velocity_relaxation_myr: float = 18.0
    max_speed_deg_per_myr: float = 1.20
    min_active_speed_deg_per_myr: float = 0.005
    remove_net_rotation: bool = True
    # v0.9.8 reconstruction assumptions. The qualitative couplings are
    # preserved by the surviving reports; these exact numerical coefficients
    # were not recoverable from the lost source archive.
    slab_thermal_reference_km: float = 140.0
    slab_thermal_exponent: float = 0.25
    continental_buoyancy_reference_km: float = 35.0
    continental_buoyancy_resistance_gain: float = 0.65
    craton_collision_resistance_gain: float = 0.60
    gpe_reference_thickness_km: float = 40.0
    gpe_drive_weight: float = 0.30


@dataclass(slots=True)
class DynamicsDiagnostics:
    time_myr: float
    mean_speed_deg_per_myr: float
    max_speed_deg_per_myr: float
    mean_axis_turn_deg: float
    max_axis_turn_deg: float
    ridge_boundary_length_km: float
    slab_boundary_length_km: float
    continental_collision_length_km: float
    transform_boundary_length_km: float
    mean_collision_drag_factor: float
    net_rotation_deg_per_myr_before_removal: float
    mantle_rms_speed_deg_per_myr: float = 0.0
    mean_plate_mantle_slip_deg_per_myr: float = 0.0
    mean_gpe_drive: float = 0.0
    mean_continental_plate_speed_deg_per_myr: float = 0.0
    min_continental_plate_speed_deg_per_myr: float = 0.0
    near_stationary_continental_plates: int = 0
    mean_ridge_push_factor: float = 1.0
    min_ridge_push_factor: float = 1.0
    max_ridge_push_factor: float = 1.0


def angular_velocity_vectors(system: PlateSystem) -> Array:
    return np.asarray(
        [p.euler_axis * p.angular_speed_rad_per_myr for p in system.plates],
        dtype=np.float64,
    )


def system_from_omega(cell_plate: Array, template: PlateSystem, omega: Array) -> PlateSystem:
    plates: list[Plate] = []
    for old, w in zip(template.plates, np.asarray(omega, dtype=np.float64)):
        speed = float(np.linalg.norm(w))
        if speed > 1e-15:
            axis = np.asarray(w / speed, dtype=np.float64)
            signed_speed = speed
        else:
            axis = np.asarray(old.euler_axis, dtype=np.float64).copy()
            signed_speed = 0.0
        plates.append(
            Plate(
                plate_id=old.plate_id,
                seed_cell=old.seed_cell,
                euler_axis=axis,
                angular_speed_rad_per_myr=signed_speed,
            )
        )
    return PlateSystem(cell_plate=np.asarray(cell_plate, dtype=np.int32).copy(), plates=tuple(plates))


def _boundary_length_km(mesh: SphereMesh, b: BoundaryRecord, radius_km: float) -> float:
    u = mesh.vertices[b.vertex_u]
    v = mesh.vertices[b.vertex_v]
    angle = float(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0)))
    return angle * float(radius_km)


def _normal_ab(mesh: SphereMesh, b: BoundaryRecord) -> Array:
    midpoint = np.asarray(b.midpoint, dtype=np.float64)
    direction = mesh.centroids[b.face_b] - mesh.centroids[b.face_a]
    direction -= midpoint * float(np.dot(direction, midpoint))
    norm = float(np.linalg.norm(direction))
    if norm < 1e-14:
        return np.zeros(3, dtype=np.float64)
    return direction / norm


def _choose_subducting_side(state: LithosphereState, b: BoundaryRecord) -> int | None:
    """Return plate id preferred to subduct at this local convergent boundary."""
    ta = int(state.crust_type[b.face_a])
    tb = int(state.crust_type[b.face_b])
    if ta == int(CrustType.OCEANIC) and tb == int(CrustType.CONTINENTAL):
        return int(b.plate_a)
    if tb == int(CrustType.OCEANIC) and ta == int(CrustType.CONTINENTAL):
        return int(b.plate_b)
    if ta == int(CrustType.OCEANIC) and tb == int(CrustType.OCEANIC):
        # v0.16: prefer the side with greater integrated negative buoyancy of
        # mantle lithosphere.  Fall back to crustal age for legacy states.
        if state.mantle_lithosphere_thickness_km is not None and state.mantle_lithosphere_density_anomaly_kg_m3 is not None:
            proxy = mantle_lithosphere_negative_buoyancy_proxy(state)
            aa = float(proxy[b.face_a]); ab = float(proxy[b.face_b])
        else:
            aa = float(state.crust_age_myr[b.face_a]); ab = float(state.crust_age_myr[b.face_b])
        if aa > ab + 1e-9:
            return int(b.plate_a)
        if ab > aa + 1e-9:
            return int(b.plate_b)
        return min(int(b.plate_a), int(b.plate_b))
    return None


def mantle_lithosphere_ridge_gpe_proxy(state: LithosphereState) -> Array:
    """First-moment thermal GPE proxy for oceanic ridge push.

    Slab pull in v0.16 uses integrated negative buoyancy ~drho*H.  Ridge push
    is a gravitational-potential-energy contrast and therefore scales with the
    first moment of that density anomaly, ~drho*H^2.  With H~sqrt(age), this
    gives the expected half-space-cooling ridge-force trend ~age.
    """
    n = len(state.crust_age_myr)
    if state.mantle_lithosphere_thickness_km is None or state.mantle_lithosphere_density_anomaly_kg_m3 is None:
        return np.zeros(n, dtype=np.float64)
    h = np.maximum(np.asarray(state.mantle_lithosphere_thickness_km, dtype=np.float64), 0.0)
    drho = np.maximum(np.asarray(state.mantle_lithosphere_density_anomaly_kg_m3, dtype=np.float64), 0.0)
    return drho * h * h


def plate_ridge_push_factors(
    mesh: SphereMesh,
    state: LithosphereState,
    radius_km: float,
    plate_count: int,
    params: DynamicsParameters,
) -> Array:
    """Return one thermal ridge-push multiplier per plate.

    A ridge line acts on an entire cooling flank, so the cell immediately next
    to the ridge (which is necessarily young) is not a useful force estimate.
    Instead we integrate the v0.16 thermal GPE proxy over each plate's oceanic
    footprint.  For an ideal flank with age increasing roughly linearly away
    from the ridge, twice the area-mean GPE approximates the end-to-end GPE
    contrast.  This is smooth, deterministic and largely mesh independent.

    Purely continental plates retain factor 1 so existing continental-rift
    effective forcing is not silently removed; oceanic/mixed ridge faces blend
    this plate factor locally in update_plate_dynamics().
    """
    if state.mantle_lithosphere_thickness_km is None or state.mantle_lithosphere_density_anomaly_kg_m3 is None:
        return np.ones(int(plate_count), dtype=np.float64)
    areas = mesh.physical_cell_areas_km2(radius_km)
    if state.continental_fraction is None:
        cont = (np.asarray(state.crust_type) == int(CrustType.CONTINENTAL)).astype(np.float64)
    else:
        cont = np.clip(np.asarray(state.continental_fraction, dtype=np.float64), 0.0, 1.0)
    ocean_weight = areas * (1.0 - cont)
    proxy = mantle_lithosphere_ridge_gpe_proxy(state)
    pids = np.asarray(state.cell_plate, dtype=np.int32)
    ocean_area = np.bincount(pids, weights=ocean_weight, minlength=int(plate_count)).astype(np.float64)
    proxy_sum = np.bincount(pids, weights=ocean_weight * proxy, minlength=int(plate_count)).astype(np.float64)
    mean_proxy = np.zeros(int(plate_count), dtype=np.float64)
    nz = ocean_area > 1e-9
    mean_proxy[nz] = proxy_sum[nz] / ocean_area[nz]
    # For a simple ridge-to-old-ocean flank, U grows ~linearly with age, so
    # 2*mean(U) estimates the total ridge-to-flank GPE contrast.
    effective_proxy = 2.0 * mean_proxy
    ref = max(
        float(params.ridge_gpe_reference_density_anomaly_kg_m3)
        * float(params.ridge_gpe_reference_mantle_thickness_km) ** 2,
        1e-12,
    )
    factor = np.ones(int(plate_count), dtype=np.float64)
    raw = np.maximum(effective_proxy[nz] / ref, 0.0)
    # A pure half-space model gives ridge force ~age indefinitely.  Real plate
    # cooling saturates: classic thermal models show only slow additional ridge
    # driving force once oceanic lithosphere is mature.  Apply a normalized
    # exponential saturation to the GPE ratio, preserving raw=1 as the
    # reference scale before the small calibration gain.
    sat = max(float(params.ridge_gpe_saturation_ratio), 1e-9)
    shaped = (1.0 - np.exp(-raw / sat)) / max(1.0 - np.exp(-1.0 / sat), 1e-12)
    factor[nz] = float(params.ridge_gpe_calibration_gain) * shaped ** float(params.ridge_gpe_exponent)
    factor[nz] = np.clip(
        factor[nz],
        float(params.ridge_gpe_min_factor),
        float(params.ridge_gpe_max_factor),
    )
    return factor


def _plate_area_weights(mesh: SphereMesh, state: LithosphereState, radius_km: float, plate_count: int) -> Array:
    areas = mesh.physical_cell_areas_km2(radius_km)
    out = np.bincount(state.cell_plate, weights=areas, minlength=plate_count).astype(np.float64)
    return out



def center_net_rotation(mesh: SphereMesh, state: LithosphereState, system: PlateSystem, radius_km: float) -> PlateSystem:
    """Remove the area-weighted rigid rotation gauge from an entire plate system.

    A common angular-velocity vector added to every plate does not change any
    relative plate boundary motion.  Centering the *initial* system before the
    v0.5 integration prevents an artificial one-step Euler-pole jump when the
    same gauge removal is applied later.
    """
    omega = angular_velocity_vectors(system)
    weights = _plate_area_weights(mesh, state, radius_km, len(system.plates))
    total = max(float(np.sum(weights)), 1e-30)
    mean = np.sum(omega * weights[:, None], axis=0) / total
    return system_from_omega(state.cell_plate, system, omega - mean[None, :])

def update_plate_dynamics(
    mesh: SphereMesh,
    state: LithosphereState,
    current_system: PlateSystem,
    baseline_system: PlateSystem,
    radius_km: float,
    dt_myr: float,
    normal_threshold_km_per_myr: float,
    inactive_speed_km_per_myr: float,
    params: DynamicsParameters,
    *,
    mantle_flow: MantleFlowState | None = None,
    thermal_lithosphere_thickness_km: float | None = None,
    subduction_memory: SubductionMemoryState | None = None,
    subduction_memory_params: SubductionMemoryParameters | None = None,
    rollback_omega_rad_per_myr: Array | None = None,
) -> tuple[PlateSystem, DynamicsDiagnostics, list[BoundaryRecord], Array]:
    """Update Euler vectors from effective boundary-force proxies.

    Returns (new_system, diagnostics, current_boundaries, drive_vectors).
    """
    if dt_myr <= 0.0:
        raise ValueError("dt_myr must be positive")
    pcount = len(current_system.plates)
    current_for_state = PlateSystem(cell_plate=state.cell_plate.copy(), plates=current_system.plates)
    boundaries = classify_boundaries(
        mesh,
        current_for_state,
        radius_km,
        normal_threshold_km_per_myr,
        inactive_speed_km_per_myr,
    )

    drive = np.zeros((pcount, 3), dtype=np.float64)
    gpe_drive = np.zeros((pcount, 3), dtype=np.float64)
    gpe_weight = np.zeros(pcount, dtype=np.float64)
    boundary_weight = np.zeros(pcount, dtype=np.float64)
    collision_length = np.zeros(pcount, dtype=np.float64)
    transform_length = np.zeros(pcount, dtype=np.float64)
    ridge_len = slab_len = coll_len = trans_len = 0.0
    ridge_factors = plate_ridge_push_factors(mesh, state, radius_km, pcount, params)
    ridge_factor_sum = 0.0
    ridge_factor_weight = 0.0
    ridge_factor_min = np.inf
    ridge_factor_max = -np.inf

    for b in boundaries:
        length = _boundary_length_km(mesh, b, radius_km)
        if length <= 0.0:
            continue
        normal = _normal_ab(mesh, b)
        if not np.any(normal):
            continue
        r = np.asarray(b.midpoint, dtype=np.float64)
        pa, pb = int(b.plate_a), int(b.plate_b)
        damage = 0.5 * (float(state.tidal_damage[b.face_a]) + float(state.tidal_damage[b.face_b]))
        resistance_scale = np.clip(1.0 - params.tidal_resistance_reduction * damage, 0.15, 1.0)

        if b.boundary_type == BoundaryType.DIVERGENT:
            ridge_len += length
            # v0.17: ridge push is no longer one global line-force constant.
            # Each plate side is weighted by the GPE of its cooling oceanic
            # lithosphere.  A mixed/continental ridge face blends smoothly back
            # toward the legacy factor 1, preserving continental-rift behaviour.
            if state.continental_fraction is None:
                fa = 1.0 if int(state.crust_type[b.face_a]) == int(CrustType.CONTINENTAL) else 0.0
                fb = 1.0 if int(state.crust_type[b.face_b]) == int(CrustType.CONTINENTAL) else 0.0
            else:
                fa = float(np.clip(state.continental_fraction[b.face_a], 0.0, 1.0))
                fb = float(np.clip(state.continental_fraction[b.face_b], 0.0, 1.0))
            factor_a = fa + (1.0 - fa) * float(ridge_factors[pa])
            factor_b = fb + (1.0 - fb) * float(ridge_factors[pb])
            tidal_gain = 1.0 + params.tidal_ridge_enhancement * damage
            strength_a = params.ridge_push_weight * factor_a * tidal_gain
            strength_b = params.ridge_push_weight * factor_b * tidal_gain
            # A moves away from B, B moves away from A.
            drive[pa] += length * strength_a * np.cross(r, -normal)
            drive[pb] += length * strength_b * np.cross(r, +normal)
            boundary_weight[pa] += length
            boundary_weight[pb] += length
            ridge_factor_sum += length * (factor_a + factor_b)
            ridge_factor_weight += 2.0 * length
            ridge_factor_min = min(ridge_factor_min, factor_a, factor_b)
            ridge_factor_max = max(ridge_factor_max, factor_a, factor_b)

        elif b.boundary_type == BoundaryType.CONVERGENT:
            ta = int(state.crust_type[b.face_a])
            tb = int(state.crust_type[b.face_b])
            if ta == int(CrustType.CONTINENTAL) and tb == int(CrustType.CONTINENTAL):
                coll_len += length
                mean_h = 0.5 * (
                    float(state.crust_thickness_km[b.face_a])
                    + float(state.crust_thickness_km[b.face_b])
                )
                ref_h = max(float(params.continental_buoyancy_reference_km), 1e-9)
                buoyancy = 1.0
                if mantle_flow is not None:
                    buoyancy += float(params.continental_buoyancy_resistance_gain) * max(
                        mean_h / ref_h - 1.0, 0.0
                    )
                craton_drag = 1.0
                if state.craton_strength is not None:
                    mean_craton_strength = 0.5 * (
                        float(state.craton_strength[b.face_a])
                        + float(state.craton_strength[b.face_b])
                    )
                    craton_drag += float(params.craton_collision_resistance_gain) * float(
                        np.clip(mean_craton_strength, 0.0, 1.0)
                    )
                collision_length[pa] += length * resistance_scale * buoyancy * craton_drag
                collision_length[pb] += length * resistance_scale * buoyancy * craton_drag
                continue

            sub = _choose_subducting_side(state, b)
            if sub is not None:
                slab_len += length
                if sub == pa:
                    face = b.face_a
                    toward_trench = +normal
                else:
                    face = b.face_b
                    toward_trench = -normal
                if state.mantle_lithosphere_thickness_km is not None and state.mantle_lithosphere_density_anomaly_kg_m3 is not None:
                    hmantle = max(float(state.mantle_lithosphere_thickness_km[face]), 0.0)
                    drho = max(float(state.mantle_lithosphere_density_anomaly_kg_m3[face]), 0.0)
                    ref = max(
                        float(params.slab_buoyancy_reference_thickness_km)
                        * float(params.slab_buoyancy_reference_density_anomaly_kg_m3),
                        1e-9,
                    )
                    buoyancy_factor = float(np.clip((hmantle * drho) / ref, 0.02, 2.8)) ** float(params.slab_buoyancy_exponent)
                    strength = params.slab_pull_weight * float(params.slab_buoyancy_calibration_gain) * buoyancy_factor
                else:
                    # Backwards-compatible v0.15 path for legacy unit states.
                    age = max(float(state.crust_age_myr[face]), 0.0)
                    age_factor = 1.0 + params.slab_age_gain * min(age / max(params.slab_age_reference_myr, 1e-9), 1.5)
                    thermal_factor = 1.0
                    if thermal_lithosphere_thickness_km is not None:
                        thermal_factor = float(np.clip(
                            max(float(thermal_lithosphere_thickness_km), 1e-9)
                            / max(float(params.slab_thermal_reference_km), 1e-9),
                            0.55, 2.5
                        )) ** float(params.slab_thermal_exponent)
                    strength = params.slab_pull_weight * age_factor * thermal_factor
                over = pb if sub == pa else pa
                breakoff_mult = slab_pull_multiplier_for_pair(subduction_memory, sub, over)
                if breakoff_mult > 0.0:
                    drive[sub] += length * strength * breakoff_mult * np.cross(r, toward_trench)
                    boundary_weight[sub] += length

        elif b.boundary_type == BoundaryType.TRANSFORM:
            trans_len += length
            transform_length[pa] += length * resistance_scale
            transform_length[pb] += length * resistance_scale

    # v0.9.8-style gravitational-potential-energy spreading: thick
    # continental columns push laterally toward thinner same-plate neighbours.
    # This is an effective torque proxy, not an SI stress solver.
    cont_mask = np.asarray(state.crust_type) == int(CrustType.CONTINENTAL)
    href = float(params.gpe_reference_thickness_km)
    for cell in np.flatnonzero(cont_mask):
        pid = int(state.cell_plate[cell])
        h0 = float(state.crust_thickness_km[cell])
        excess = max(h0 - href, 0.0)
        if excess <= 0.0:
            continue
        r0 = np.asarray(mesh.centroids[cell], dtype=np.float64)
        for nb in mesh.neighbors[cell]:
            nb = int(nb)
            if int(state.cell_plate[nb]) != pid or not cont_mask[nb]:
                continue
            dh = max(h0 - float(state.crust_thickness_km[nb]), 0.0)
            if dh <= 0.0:
                continue
            tangent = np.asarray(mesh.centroids[nb] - r0, dtype=np.float64)
            tangent -= r0 * float(np.dot(tangent, r0))
            tn = float(np.linalg.norm(tangent))
            if tn <= 1e-14:
                continue
            tangent /= tn
            w = excess * dh
            gpe_drive[pid] += w * np.cross(r0, tangent)
            gpe_weight[pid] += w
    gnz = gpe_weight > 0.0
    gpe_drive[gnz] /= gpe_weight[gnz, None]

    # Normalize local boundary drives so mesh resolution does not set the speed.
    nonzero = boundary_weight > 0.0
    drive[nonzero] /= boundary_weight[nonzero, None]

    # v0.18: an already submerged slab retains a small, decaying pull for a
    # short interval after the surface contact is reclassified. Active zones
    # receive no extra mature force: v0.17 is already calibrated to that.
    if subduction_memory is not None and subduction_memory_params is not None:
        residual_drive, _residual_frac = residual_pull_by_plate(
            subduction_memory, pcount, subduction_memory_params,
            params.slab_pull_weight, params.slab_buoyancy_calibration_gain,
        )
        drive += residual_drive

    current_omega = angular_velocity_vectors(current_system)
    baseline_omega = angular_velocity_vectors(baseline_system)
    if mantle_flow is not None:
        mantle_omega = plate_mean_mantle_omega(
            mesh, state.cell_plate, pcount, radius_km, mantle_flow
        )
    else:
        mantle_omega = baseline_omega
    drive_scale = np.deg2rad(float(params.force_speed_scale_deg_per_myr))
    gpe_component = float(params.gpe_drive_weight) * gpe_drive if mantle_flow is not None else 0.0
    relative_drive = drive_scale * (drive + gpe_component)
    if rollback_omega_rad_per_myr is not None:
        rb=np.asarray(rollback_omega_rad_per_myr,dtype=np.float64)
        if rb.shape != relative_drive.shape: raise ValueError("rollback_omega_rad_per_myr shape mismatch")
        relative_drive = relative_drive + rb
    common_mantle = float(params.mantle_memory_fraction) * mantle_omega

    # Boundary resistance acts as extra quasi-static drag.  Normalize by each
    # plate's total boundary length so large plates are not penalized merely for size.
    total_boundary = np.maximum(boundary_weight + collision_length + transform_length, 1e-9)
    collision_ratio = collision_length / total_boundary
    transform_ratio = transform_length / total_boundary
    drag_factor = 1.0 / (
        1.0
        + float(params.continental_collision_resistance) * collision_ratio
        + float(params.transform_resistance) * transform_ratio
    )
    # v0.9.8 invariant: collision/transform resistance damps the *relative
    # tectonic drive*, not the common absolute mantle-advection component.
    target = common_mantle + relative_drive * drag_factor[:, None]

    alpha = 1.0 - np.exp(-float(dt_myr) / max(float(params.velocity_relaxation_myr), 1e-9))
    new_omega = current_omega + alpha * (target - current_omega)

    plate_areas = _plate_area_weights(mesh, state, radius_km, pcount)
    area_sum = max(float(np.sum(plate_areas)), 1e-30)
    mean_rotation = np.sum(new_omega * plate_areas[:, None], axis=0) / area_sum
    net_before = float(np.linalg.norm(mean_rotation))
    if params.remove_net_rotation:
        new_omega -= mean_rotation[None, :]

    max_speed = np.deg2rad(float(params.max_speed_deg_per_myr))
    min_speed = np.deg2rad(float(params.min_active_speed_deg_per_myr))
    speeds = np.linalg.norm(new_omega, axis=1)
    too_fast = speeds > max_speed
    if np.any(too_fast):
        new_omega[too_fast] *= (max_speed / speeds[too_fast])[:, None]
    speeds = np.linalg.norm(new_omega, axis=1)
    # Legacy runs keep the old quantisation.  Explicit mantle runs deliberately
    # retain slow non-zero motion so topology/collision cannot ratchet a plate
    # into a permanent zero-speed state.
    if mantle_flow is None:
        tiny = speeds < min_speed
        new_omega[tiny] = 0.0

    # Axis-turn diagnostic between old and new omega vectors.
    old_speed = np.linalg.norm(current_omega, axis=1)
    new_speed = np.linalg.norm(new_omega, axis=1)
    turns = np.zeros(pcount, dtype=np.float64)
    valid = (old_speed > 1e-14) & (new_speed > 1e-14)
    if np.any(valid):
        dots = np.sum(current_omega[valid] * new_omega[valid], axis=1) / (old_speed[valid] * new_speed[valid])
        turns[valid] = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))

    new_system = system_from_omega(state.cell_plate, current_system, new_omega)
    final_speed_deg = np.rad2deg(np.linalg.norm(new_omega, axis=1))
    if mantle_flow is not None:
        mantle_rms_deg = float(np.rad2deg(mantle_flow_rms_rad_per_myr(mantle_flow)))
        slip_deg = np.rad2deg(np.linalg.norm(new_omega - mantle_omega, axis=1))
        mean_slip_deg = float(np.mean(slip_deg)) if len(slip_deg) else 0.0
    else:
        mantle_rms_deg = float(np.rad2deg(np.sqrt(np.mean(np.sum(baseline_omega * baseline_omega, axis=1))))) if len(baseline_omega) else 0.0
        mean_slip_deg = 0.0
    cont_plate_ids = np.unique(np.asarray(state.cell_plate)[cont_mask]).astype(np.int32) if np.any(cont_mask) else np.empty(0, dtype=np.int32)
    cont_speeds = final_speed_deg[cont_plate_ids] if len(cont_plate_ids) else np.empty(0, dtype=float)
    diag = DynamicsDiagnostics(
        time_myr=float(state.time_myr),
        mean_speed_deg_per_myr=float(np.mean(final_speed_deg)),
        max_speed_deg_per_myr=float(np.max(final_speed_deg)),
        mean_axis_turn_deg=float(np.mean(turns)),
        max_axis_turn_deg=float(np.max(turns)),
        ridge_boundary_length_km=float(ridge_len),
        slab_boundary_length_km=float(slab_len),
        continental_collision_length_km=float(coll_len),
        transform_boundary_length_km=float(trans_len),
        mean_collision_drag_factor=float(np.mean(drag_factor)),
        net_rotation_deg_per_myr_before_removal=float(np.rad2deg(net_before)),
        mantle_rms_speed_deg_per_myr=mantle_rms_deg,
        mean_plate_mantle_slip_deg_per_myr=mean_slip_deg,
        mean_gpe_drive=float(np.mean(np.linalg.norm(gpe_drive, axis=1))) if len(gpe_drive) else 0.0,
        mean_continental_plate_speed_deg_per_myr=float(np.mean(cont_speeds)) if len(cont_speeds) else 0.0,
        min_continental_plate_speed_deg_per_myr=float(np.min(cont_speeds)) if len(cont_speeds) else 0.0,
        near_stationary_continental_plates=int(np.sum(cont_speeds < 0.01)) if len(cont_speeds) else 0,
        mean_ridge_push_factor=float(ridge_factor_sum / ridge_factor_weight) if ridge_factor_weight > 0.0 else 1.0,
        min_ridge_push_factor=float(ridge_factor_min) if np.isfinite(ridge_factor_min) else 1.0,
        max_ridge_push_factor=float(ridge_factor_max) if np.isfinite(ridge_factor_max) else 1.0,
    )
    return new_system, diag, boundaries, drive


__all__ = [
    "DynamicsParameters",
    "DynamicsDiagnostics",
    "angular_velocity_vectors",
    "center_net_rotation",
    "system_from_omega",
    "update_plate_dynamics",
]
