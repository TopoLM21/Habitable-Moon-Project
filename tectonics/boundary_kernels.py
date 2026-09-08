"""Exact prepared-CPU boundary-force terms for opt-in production execution.

Static edge geometry is computed with the reference scalar NumPy operations.
Dynamic fields are repacked on every call and reduced in original boundary
order. Python scalar powers are retained for byte-level agreement. This module
does not enable an execution policy or require CUDA.
"""
from __future__ import annotations

import numpy as np

from . import dynamics
from .kinematics import BoundaryType
from .lithosphere import CrustType

NAMES = ("drive", "boundary_weight", "collision_length", "transform_length",
         "ridge_len", "slab_len", "coll_len", "trans_len", "ridge_factor_sum",
         "ridge_factor_weight", "ridge_factor_min", "ridge_factor_max")


class BoundaryGeometry:
    """Lazy, bounded-to-mesh static edge cache; new boundaries pay cold cost.

    Mesh vertices and centroids must remain immutable for this object's lifetime.
    Use a new BoundaryGeometry for a changed mesh. Midpoints are included in keys
    so changed/custom records cannot
    accidentally reuse geometry from a different midpoint. Radius is not cached.
    """
    def __init__(self, mesh):
        self.mesh = mesh
        self.edges = {}
        # Bound even custom midpoint/orientation records; eviction changes only
        # cold geometry work, never arithmetic. Production shared edges fit.
        self.max_cached_edges = max(1, 2 * len(mesh.shared_edges))

    def pack(self, boundaries, radius):
        packed = np.empty((len(boundaries), 7), dtype=np.float64)
        for i, b in enumerate(boundaries):
            key = (b.face_a, b.face_b, b.vertex_u, b.vertex_v,
                   np.asarray(b.midpoint, dtype=np.float64).tobytes())
            value = self.edges.get(key)
            if value is None:
                normal = dynamics._normal_ab(self.mesh, b)
                value = np.concatenate((
                    [dynamics._boundary_length_km(self.mesh, b, 1.0)],
                    np.cross(b.midpoint, -normal), np.cross(b.midpoint, +normal)))
                if not np.any(normal):
                    value[0] = 0.0
                if len(self.edges) >= self.max_cached_edges:
                    self.edges.pop(next(iter(self.edges)))
                self.edges[key] = value
            packed[i] = value
        packed[:, 0] *= float(radius)
        return packed

    @property
    def numeric_bytes(self):
        return sum(value.nbytes for value in self.edges.values())


def _ordered_sum(values):
    # np.sum uses pairwise grouping. cumsum retains the original left fold.
    return float(np.cumsum(values, dtype=np.float64)[-1]) if len(values) else 0.0


def prepare_contributions(geometry, state, boundaries, radius_km, pcount, params,
                          mantle_flow=None, thermal_lithosphere_thickness_km=None,
                          subduction_memory=None):
    """Exact local contributions, still ordered by boundary then A/B side."""
    # The scalar reference promotes each field access to Python float. A float32
    # batch would round earlier, so this optimized path explicitly supports only the
    # finite FP64 fields used by saved production checkpoints. Include checking
    # cost in every inclusive timing; do not silently accept lower precision.
    for name in ('tidal_damage', 'crust_age_myr', 'crust_thickness_km',
                 'continental_fraction', 'craton_strength',
                 'mantle_lithosphere_thickness_km', 'mantle_lithosphere_density_anomaly_kg_m3'):
        value = getattr(state, name)
        if value is not None and (not isinstance(value, np.ndarray) or value.dtype != np.float64
                                  or value.shape != (geometry.mesh.cell_count,)
                                  or not np.all(np.isfinite(value))):
            raise ValueError(f'{name} must be a finite FP64 cell array or None')
    geom = geometry.pack(boundaries, radius_km)
    n = len(boundaries)
    faces = np.asarray([(b.face_a, b.face_b) for b in boundaries], dtype=np.intp).reshape(n, 2)
    owners = np.asarray([(b.plate_a, b.plate_b) for b in boundaries], dtype=np.int32).reshape(n, 2)
    kinds = np.fromiter((int(b.boundary_type) for b in boundaries), np.int8, n)
    length = geom[:, 0]
    valid = length > 0.0
    values = np.zeros((n, 2, 6), dtype=np.float64)
    damage = 0.5 * (state.tidal_damage[faces[:, 0]] + state.tidal_damage[faces[:, 1]])
    resistance = np.clip(1.0 - params.tidal_resistance_reduction * damage, 0.15, 1.0)
    ridge_factors = dynamics.plate_ridge_push_factors(geometry.mesh, state, radius_km, pcount, params)
    crust = np.asarray(state.crust_type)[faces]
    divergent = valid & (kinds == BoundaryType.DIVERGENT)
    convergence = valid & (kinds == BoundaryType.CONVERGENT)
    collision = convergence & np.all(crust == CrustType.CONTINENTAL, axis=1)
    transform = valid & (kinds == BoundaryType.TRANSFORM)

    idx = np.flatnonzero(divergent)
    if state.continental_fraction is None:
        fraction = (crust[idx] == CrustType.CONTINENTAL).astype(np.float64)
    else:
        fraction = np.clip(state.continental_fraction[faces[idx]], 0.0, 1.0)
    factors = fraction + (1.0 - fraction) * ridge_factors[owners[idx]]
    tidal_gain = 1.0 + params.tidal_ridge_enhancement * damage[idx]
    strengths = params.ridge_push_weight * factors * tidal_gain[:, None]
    values[idx, 0, :3] = (length[idx] * strengths[:, 0])[:, None] * geom[idx, 1:4]
    values[idx, 1, :3] = (length[idx] * strengths[:, 1])[:, None] * geom[idx, 4:7]
    values[idx, :, 3] = length[idx, None]
    ridge_sum = _ordered_sum(length[idx] * (factors[:, 0] + factors[:, 1]))
    ridge_weight = _ordered_sum(2.0 * length[idx])
    ridge_min = float(np.min(factors)) if len(idx) else np.inf
    ridge_max = float(np.max(factors)) if len(idx) else -np.inf

    idx = np.flatnonzero(collision)
    mean_h = 0.5 * (state.crust_thickness_km[faces[idx, 0]] + state.crust_thickness_km[faces[idx, 1]])
    buoyancy = np.ones(len(idx), dtype=np.float64)
    if mantle_flow is not None:
        buoyancy += params.continental_buoyancy_resistance_gain * np.maximum(
            mean_h / max(float(params.continental_buoyancy_reference_km), 1e-9) - 1.0, 0.0)
    craton = np.ones(len(idx), dtype=np.float64)
    if state.craton_strength is not None:
        mean_craton = 0.5 * (state.craton_strength[faces[idx, 0]] + state.craton_strength[faces[idx, 1]])
        craton += params.craton_collision_resistance_gain * np.clip(mean_craton, 0.0, 1.0)
    values[idx, :, 4] = (length[idx] * resistance[idx] * buoyancy * craton)[:, None]
    values[transform, :, 5] = (length[transform] * resistance[transform])[:, None]

    idx = np.flatnonzero(convergence & ~collision)
    paired_crust = crust[idx]
    ocean_a = paired_crust[:, 0] == CrustType.OCEANIC
    ocean_b = paired_crust[:, 1] == CrustType.OCEANIC
    mantle = (state.mantle_lithosphere_thickness_km is not None and
              state.mantle_lithosphere_density_anomaly_kg_m3 is not None)
    if mantle:
        pair_h = np.maximum(state.mantle_lithosphere_thickness_km[faces[idx]], 0.0)
        pair_drho = np.maximum(state.mantle_lithosphere_density_anomaly_kg_m3[faces[idx]], 0.0)
        proxy = np.maximum(pair_h * pair_drho, 0.0)
    else:
        proxy = np.asarray(state.crust_age_myr)[faces[idx]]
    choose_a = ((ocean_a & ~ocean_b) | (ocean_a & ocean_b & (
        (proxy[:, 0] > proxy[:, 1] + 1e-9) |
        (~(proxy[:, 1] > proxy[:, 0] + 1e-9) & (owners[idx, 0] <= owners[idx, 1])))))
    # Production crust types are binary. Unknown kinds receive no slab force.
    eligible = ((ocean_a & (ocean_b | (paired_crust[:, 1] == CrustType.CONTINENTAL))) |
                (ocean_b & (ocean_a | (paired_crust[:, 0] == CrustType.CONTINENTAL))))
    idx = idx[eligible]
    side = np.where(choose_a[eligible], 0, 1)
    selected_faces = faces[idx, side]
    sub = owners[idx, side]
    over = owners[idx, 1 - side]
    if mantle:
        h = np.maximum(state.mantle_lithosphere_thickness_km[selected_faces], 0.0)
        rho = np.maximum(state.mantle_lithosphere_density_anomaly_kg_m3[selected_faces], 0.0)
        ref = max(float(params.slab_buoyancy_reference_thickness_km) *
                  float(params.slab_buoyancy_reference_density_anomaly_kg_m3), 1e-9)
        ratio = np.clip((h * rho) / ref, 0.02, 2.8)
        # NumPy/CUDA pow need not equal CPython scalar pow. Preserve this call.
        powered = np.fromiter((float(x) ** float(params.slab_buoyancy_exponent) for x in ratio),
                              dtype=np.float64, count=len(ratio))
        strength = params.slab_pull_weight * float(params.slab_buoyancy_calibration_gain) * powered
    else:
        age = np.maximum(state.crust_age_myr[selected_faces], 0.0)
        age_factor = 1.0 + params.slab_age_gain * np.minimum(age / max(params.slab_age_reference_myr, 1e-9), 1.5)
        thermal_factor = 1.0
        if thermal_lithosphere_thickness_km is not None:
            thermal_factor = float(np.clip(max(float(thermal_lithosphere_thickness_km), 1e-9) /
                max(float(params.slab_thermal_reference_km), 1e-9), 0.55, 2.5)) ** float(params.slab_thermal_exponent)
        strength = params.slab_pull_weight * age_factor * thermal_factor
    multipliers = np.fromiter((dynamics.slab_pull_multiplier_for_pair(subduction_memory, a, b)
                              for a, b in zip(sub, over)), dtype=np.float64, count=len(idx))
    active = multipliers > 0.0
    active_idx, active_side = idx[active], side[active]
    toward = np.where(active_side[:, None] == 0, geom[active_idx, 4:7], geom[active_idx, 1:4])
    values[active_idx, active_side, :3] = (
        length[active_idx] * strength[active] * multipliers[active])[:, None] * toward
    values[active_idx, active_side, 3] = length[active_idx]
    scalars = (_ordered_sum(length[divergent]), _ordered_sum(length[idx]),
               _ordered_sum(length[collision]), _ordered_sum(length[transform]),
               ridge_sum, ridge_weight, ridge_min, ridge_max)
    return owners.ravel(), values.reshape(2 * n, 6), scalars


def _result(output, scalars):
    return (output[:, :3], output[:, 3], output[:, 4], output[:, 5], *scalars)


def prepared_cpu(geometry, *args, **kwargs):
    owners, values, scalars = prepare_contributions(geometry, *args, **kwargs)
    output = np.zeros((args[3], 6), dtype=np.float64)
    np.add.at(output, owners, values)
    return _result(output, scalars)
