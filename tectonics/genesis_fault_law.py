"""Finite-width, non-dilatant weak-plane slip in a damaged membrane.

This material law does not split nodes or solve contact between separate faces.
Strain vectors use engineering shear, while stress vectors store physical xy
stress. Compression contributing to friction is the modeled membrane stress;
water availability changes strength, never an inferred pore pressure.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from numbers import Real

import numpy as np

from .genesis import SECONDS_PER_MYR


@dataclass(frozen=True)
class WeakPlaneParameters:
    enabled: bool = True
    cohesion_pa: float = 4e6
    residual_cohesion_fraction: float = .1
    friction_dry: float = .6
    friction_wet: float = .2
    wet_cohesion_fraction: float = .5
    viscosity_pa_s: float = 1e20
    band_width_km: float = 100.
    activation_damage: float = .65
    activation_persistence_myr: float = .05
    max_shear_increment: float = .002

    def validate(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("faults.enabled must be boolean")
        for field in fields(self):
            if field.name == "enabled":
                continue
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"faults.{field.name} must be a finite number")
            if value < 0:
                raise ValueError(f"faults.{field.name} must be nonnegative")
        if not all(0 <= getattr(self, name) <= 1 for name in (
                "residual_cohesion_fraction", "wet_cohesion_fraction", "activation_damage")):
            raise ValueError("Fault strength fractions and activation damage must lie in [0, 1]")
        if not 0 <= self.friction_wet <= self.friction_dry <= 2:
            raise ValueError("Fault friction must satisfy 0 <= wet <= dry <= 2")
        if self.viscosity_pa_s <= 0 or self.band_width_km <= 0:
            raise ValueError("Fault viscosity and band width must be positive")
        if not 0 < self.max_shear_increment <= .02:
            raise ValueError("Fault maximum shear increment must lie in (0, .02]")


def weak_plane_parameters_from_config(config):
    section = dict(config.get("genesis_faults", {}))
    schema = section.pop("schema_version", 1)
    if isinstance(schema, bool) or schema != 1:
        raise ValueError("Unsupported genesis_faults schema")
    unknown = set(section) - {field.name for field in fields(WeakPlaneParameters)}
    if unknown:
        raise ValueError(f"Unknown fault parameters: {sorted(unknown)}")
    parameters = WeakPlaneParameters(**section)
    parameters.validate()
    return parameters


fault_parameters_from_config = weak_plane_parameters_from_config


def select_plane(stress_pa, friction):
    """Maximize shear minus friction for one conjugate; persist this plane.

    The returned normal is an unoriented material line: n and -n represent the
    same plane. Isotropic stress has no preferred direction and uses angle zero
    for its principal axis. Selection itself never declares a plane active.
    Pressure is clamped exactly as in ``return_map``: candidates comprise the
    compressed Coulomb optimum, maximum shear, and the zero-normal-stress cusp.
    The purely compressive optimum alone is incorrect in tensile stress states.
    """
    stress = np.asarray(stress_pa, dtype=float)
    if stress.ndim < 1 or stress.shape[-1] != 3 or not np.isfinite(stress).all():
        raise ValueError("Plane selection requires finite [..., 3] stress")
    try:
        mu = np.broadcast_to(np.asarray(friction, dtype=float), stress.shape[:-1])
    except ValueError as exc:
        raise ValueError("Plane friction must match stress shape") from exc
    if not np.isfinite(mu).all() or np.any(mu < 0):
        raise ValueError("Plane friction must be finite and nonnegative")
    principal_angle = .5*np.arctan2(2*stress[..., 2], stress[..., 0]-stress[..., 1])
    mean = .5*(stress[..., 0]+stress[..., 1])
    amplitude = np.hypot(.5*(stress[..., 0]-stress[..., 1]), stress[..., 2])
    ratio = np.divide(-mean, amplitude, out=np.zeros_like(mean), where=amplitude > 0)
    candidates = np.stack((np.pi/4-.5*np.arctan(mu), np.full_like(mu, np.pi/4),
                           .5*np.arccos(np.clip(ratio, -1., 1.))), axis=-1)
    normal_stress = mean[..., None]+amplitude[..., None]*np.cos(2*candidates)
    shear = amplitude[..., None]*np.sin(2*candidates)
    scores = shear-mu[..., None]*np.maximum(-normal_stress, 0.)
    valid_cusp = (amplitude > 0) & (np.abs(mean) <= amplitude)
    scores[..., 2] = np.where(valid_cusp, scores[..., 2], -np.inf)
    relative_angle = np.take_along_axis(candidates, np.argmax(scores, axis=-1)[..., None], axis=-1)[..., 0]
    angle = principal_angle+relative_angle
    return np.stack((np.cos(angle), np.sin(angle)), axis=-1)


def return_map(elastic_trial, plane_normal, active, damage, water, effective_b,
               dt_myr, young_pa, poisson_ratio, params, *, residual_stiffness=.02):
    """Implicit viscous slip with the exact fixed-plane material tangent.

    ``elastic_trial`` already includes Maxwell retention and the proposed total
    strain increment. The returned tangent differentiates stress with respect
    to that *total engineering increment*, hence its factor ``effective_b``.
    History, damage, plane orientation, and water are fixed during this local
    update. Neither inputs nor accumulated material state are modified here.

    Work densities refer only to this shear mechanism. They are separate from
    column heat, Maxwell dissipation, and the energy released by damage.
    """
    params.validate()
    for name, value in (("dt_myr", dt_myr), ("young_pa", young_pa),
                        ("poisson_ratio", poisson_ratio), ("residual_stiffness", residual_stiffness)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"Fault {name} must be a finite number")
    if dt_myr <= 0 or young_pa <= 0 or not 0 <= poisson_ratio < .5 or not 0 < residual_stiffness <= 1:
        raise ValueError("Invalid fault time or elastic parameters")
    strain = np.asarray(elastic_trial, dtype=float)
    if strain.ndim != 2 or strain.shape[1] != 3 or not np.isfinite(strain).all():
        raise ValueError("Fault elastic trial must be finite [face, 3]")
    count = strain.shape[0]
    normals = np.asarray(plane_normal, dtype=float)
    enabled = np.asarray(active)
    if normals.shape != (count, 2) or not np.isfinite(normals).all():
        raise ValueError("Fault normals must be finite [face, 2]")
    if enabled.shape != (count,) or enabled.dtype.kind != "b":
        raise ValueError("Fault active mask must be boolean [face]")
    if not np.allclose(np.sum(normals[enabled]**2, axis=1), 1., rtol=0, atol=1e-10):
        raise ValueError("Active fault normals must be unit vectors")
    arrays = []
    for name, value in (("damage", damage), ("water", water), ("effective_b", effective_b)):
        value = np.asarray(value, dtype=float)
        if value.shape != (count,) or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ValueError(f"Fault {name} must be finite [face] within [0, 1]")
        if name == "effective_b" and np.any(value <= 0):
            raise ValueError("Fault effective_b must be positive")
        arrays.append(value)
    damage, water, b = arrays

    d = np.array([[1., poisson_ratio, 0.], [poisson_ratio, 1., 0.],
                  [0., 0., (1-poisson_ratio)/2]])/(1-poisson_ratio**2)
    degradation = residual_stiffness+(1-residual_stiffness)*(1-damage)**2
    physical = young_pa*degradation[:, None, None]*d[None, :, :]
    algorithmic = physical*b[:, None, None]
    trial_stress = np.einsum("fij,fj->fi", physical, strain)
    tangent_direction = np.stack((-normals[:, 1], normals[:, 0]), axis=-1)
    q = np.column_stack((tangent_direction[:, 0]*normals[:, 0],
                         tangent_direction[:, 1]*normals[:, 1],
                         tangent_direction[:, 0]*normals[:, 1]+tangent_direction[:, 1]*normals[:, 0]))
    normal_projector = np.column_stack((normals[:, 0]**2, normals[:, 1]**2,
                                       2*normals[:, 0]*normals[:, 1]))
    tau = np.einsum("fi,fi->f", q, trial_stress)
    normal_stress = np.einsum("fi,fi->f", normal_projector, trial_stress)
    friction = params.friction_dry+(params.friction_wet-params.friction_dry)*water
    cohesion = (params.cohesion_pa*(params.residual_cohesion_fraction
                +(1-params.residual_cohesion_fraction)*(1-damage))
                *(1-(1-params.wet_cohesion_fraction)*water))
    strength = cohesion+friction*np.maximum(-normal_stress, 0.)
    overstress = np.maximum(np.abs(tau)-strength, 0.)
    yielding = enabled & params.enabled & (overstress > 0)
    cq = np.einsum("fij,fj->fi", algorithmic, q)
    viscous_modulus = params.viscosity_pa_s/(dt_myr*SECONDS_PER_MYR)
    denominator = np.einsum("fi,fi->f", q, cq)+viscous_modulus
    shear_increment = np.zeros(count)
    shear_increment[yielding] = np.sign(tau[yielding])*overstress[yielding]/denominator[yielding]
    corrected = strain-b[:, None]*shear_increment[:, None]*q
    stress = np.einsum("fij,fj->fi", physical, corrected)
    tangent = algorithmic.copy()
    flow_derivative = q+((np.sign(tau)*friction*(normal_stress < 0))[:, None]*normal_projector)
    derivative = np.einsum("fi,fij->fj", flow_derivative, algorithmic)
    tangent[yielding] -= (cq[yielding, :, None]*derivative[yielding, None, :]
                          /denominator[yielding, None, None])
    return {
        "elastic_strain": corrected,
        "stress_pa": stress,
        "tangent_pa": tangent,
        "shear_increment": shear_increment,
        "shear_stress_pa": np.einsum("fi,fi->f", q, stress),
        "normal_stress_pa": np.einsum("fi,fi->f", normal_projector, stress),
        "yield_strength_pa": strength,
        "friction_work_density_j_m3": strength*np.abs(shear_increment),
        "viscous_work_density_j_m3": viscous_modulus*shear_increment**2,
    }
