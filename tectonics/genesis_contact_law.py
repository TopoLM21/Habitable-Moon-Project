"""Small-sliding cohesive/contact response of two paired displacement traces.

Positive normal separation opens the interface. Tractions are the internal
forces conjugate to ``[opening, tangential jump]``: compression is negative.
Mode-I softening is irreversible and has an explicit fracture energy; shear
uses a separate cohesive Coulomb return. This is not a mixed-mode fracture
criterion, a contact-search algorithm, or a source of thermal energy.

References for the constitutive building blocks (not the numerical defaults):
https://mooseframework.inl.gov/source/materials/cohesive_zone_model/BiLinearMixedModeTraction.html
https://mooseframework.inl.gov/source/constraints/MechanicalContactConstraint.html
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class ContactLawParameters:
    normal_stiffness_pa_m: float = 1e7
    tangential_stiffness_pa_m: float = 1e7
    tensile_strength_pa: float = 2e6
    fracture_energy_j_m2: float = 2e7
    cohesion_pa: float = 2e6
    friction_dry: float = .6
    friction_wet: float = .2
    wet_cohesion_fraction: float = .5
    viscosity_pa_s_m: float = 1e14

    def validate(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"Contact {field.name} must be a finite number")
            if value < 0:
                raise ValueError(f"Contact {field.name} must be nonnegative")
        if min(self.normal_stiffness_pa_m, self.tangential_stiffness_pa_m,
               self.tensile_strength_pa, self.fracture_energy_j_m2) <= 0:
            raise ValueError("Contact stiffness, tensile strength and fracture energy must be positive")
        if not 0 <= self.friction_wet <= self.friction_dry <= 2:
            raise ValueError("Contact friction must satisfy 0 <= wet <= dry <= 2")
        if not 0 <= self.wet_cohesion_fraction <= 1:
            raise ValueError("Contact wet cohesion fraction must lie in [0, 1]")
        if (not math.isfinite(self.failure_opening_m) or not math.isfinite(self.damage_onset_opening_m)
                or self.damage_onset_opening_m <= 0):
            raise ValueError("Contact cohesive opening scales must be finite and positive")
        if self.failure_opening_m <= self.damage_onset_opening_m:
            raise ValueError("Contact fracture energy must allow a nonzero softening interval")

    @property
    def damage_onset_opening_m(self):
        return self.tensile_strength_pa/self.normal_stiffness_pa_m

    @property
    def failure_opening_m(self):
        return 2*self.fracture_energy_j_m2/self.tensile_strength_pa


def contact_law_parameters_from_config(config):
    """Read the local law independently of mesh/solver contact controls."""
    section = dict(config.get("genesis_contact_law", {}))
    schema = section.pop("schema_version", 1)
    if isinstance(schema, bool) or schema != 1:
        raise ValueError("Unsupported genesis_contact_law schema")
    unknown = set(section)-{field.name for field in fields(ContactLawParameters)}
    if unknown:
        raise ValueError(f"Unknown contact law parameters: {sorted(unknown)}")
    parameters = ContactLawParameters(**section)
    parameters.validate()
    return parameters


def cohesive_damage(max_opening_m, params):
    """Irreversible scalar damage associated with a maximum Mode-I opening."""
    params.validate()
    opening = np.asarray(max_opening_m, dtype=float)
    if not np.isfinite(opening).all() or np.any(opening < 0):
        raise ValueError("Contact maximum opening must be finite and nonnegative")
    onset, failure = params.damage_onset_opening_m, params.failure_opening_m
    safe = np.clip(opening, onset, failure)
    return np.clip((1-onset/safe)/(1-onset/failure), 0., 1.)


def cohesive_dissipation(max_opening_m, params):
    """Mode-I energy already dissipated per reference interface area (J/m²)."""
    params.validate()
    opening = np.asarray(max_opening_m, dtype=float)
    if not np.isfinite(opening).all() or np.any(opening < 0):
        raise ValueError("Contact maximum opening must be finite and nonnegative")
    onset, failure = params.damage_onset_opening_m, params.failure_opening_m
    return params.fracture_energy_j_m2*np.clip((opening-onset)/(failure-onset), 0., 1.)


def contact_return_map(gap_m, jump_m, old_plastic_slip_m, old_cumulative_slip_m,
                       old_max_opening_m, dt_s, water, params, *, old_damage=None):
    """Propose contact history and the exact branchwise local Jacobian.

    All array arguments have shape ``[interface]`` and remain immutable.
    ``old_max_opening_m`` controls damage; an optional stored ``old_damage`` is
    checked against it. Water changes friction and shear cohesion, not pore
    pressure or Mode-I fracture energy. ``dt_s`` is a strictly positive scalar.

    With shear stiffness K, viscosity eta and strength Y, the implicit slip is
    sign(t_trial) max(abs(t_trial)-Y, 0)/(K+eta/dt). Fully ruptured, open traces
    slide freely, including when viscosity is nonzero. Their plastic reference
    follows the current jump, so reclosure starts from the last free position.
    Cumulative slip includes this free relative motion, which dissipates no
    frictional energy. Compression penalty compliance is finite.

    The returned tangent differentiates current traction with respect to the
    current [gap, jump], holding old history and water fixed. At branch corners
    it supplies a one-sided tangent. Coulomb coupling makes it nonsymmetric.
    Dissipation entries are nonnegative constitutive increments, not an exact
    global finite-step work balance or heat deposited into the thermal model.
    ``shear_relaxation_remainder_j_m2`` is the backward-Euler spring-energy
    remainder: trial shear energy minus returned shear energy equals friction
    work plus viscous work plus this remainder, at fixed current jump. It
    includes released spring energy at complete opening and is not labeled
    physical fracture work.
    """
    params.validate()
    if isinstance(dt_s, bool) or not isinstance(dt_s, Real) or not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("Contact dt_s must be finite and positive")
    gap = np.asarray(gap_m, dtype=float)
    if gap.ndim != 1 or not np.isfinite(gap).all():
        raise ValueError("Contact gap_m must be finite [interface]")
    arrays = []
    for name, value in (("jump_m", jump_m), ("old_plastic_slip_m", old_plastic_slip_m),
                        ("old_cumulative_slip_m", old_cumulative_slip_m),
                        ("old_max_opening_m", old_max_opening_m), ("water", water)):
        value = np.asarray(value, dtype=float)
        if value.shape != gap.shape or not np.isfinite(value).all():
            raise ValueError(f"Contact {name} must be finite [interface]")
        arrays.append(value)
    jump, old_slip, old_cumulative, old_opening, wet = arrays
    if np.any(old_cumulative < 0) or np.any(old_opening < 0):
        raise ValueError("Contact cumulative slip and maximum opening must be nonnegative")
    if np.any(np.abs(old_slip) > old_cumulative+1e-10):
        raise ValueError("Contact cumulative slip must bound signed plastic slip")
    if np.any((wet < 0) | (wet > 1)):
        raise ValueError("Contact water must lie in [0, 1]")
    previous_damage = cohesive_damage(old_opening, params)
    if old_damage is not None:
        stored_damage = np.asarray(old_damage, dtype=float)
        if (stored_damage.shape != gap.shape or not np.isfinite(stored_damage).all()
                or np.any((stored_damage < 0) | (stored_damage > 1))
                or not np.allclose(stored_damage, previous_damage, rtol=1e-12, atol=1e-12)):
            raise ValueError("Contact old damage is inconsistent with maximum opening")

    opening = np.maximum(old_opening, np.maximum(gap, 0.))
    damage = cohesive_damage(opening, params)
    kn, kt = params.normal_stiffness_pa_m, params.tangential_stiffness_pa_m
    onset, failure = params.damage_onset_opening_m, params.failure_opening_m
    damage_derivative = np.zeros_like(gap)
    softening = (gap > old_opening) & (gap > onset) & (gap < failure)
    damage_derivative[softening] = failure*onset/((failure-onset)*gap[softening]**2)
    normal = kn*np.where(gap < 0, gap, (1-damage)*gap)
    normal_tangent = kn*np.where(gap < 0, 1., 1-damage-gap*damage_derivative)
    pressure = kn*np.maximum(-gap, 0.)
    friction = params.friction_dry+(params.friction_wet-params.friction_dry)*wet
    cohesion = params.cohesion_pa*(1-(1-params.wet_cohesion_fraction)*wet)
    strength = (1-damage)*cohesion+friction*pressure
    strength_derivative = -cohesion*damage_derivative-friction*kn*(gap < 0)

    trial = kt*(jump-old_slip)
    yielding = np.abs(trial) > strength
    free = (gap >= 0) & (damage >= 1.)
    viscous = np.full_like(gap, params.viscosity_pa_s_m/dt_s)
    viscous[free] = 0.
    increment = np.sign(trial)*np.maximum(np.abs(trial)-strength, 0.)/(kt+viscous)
    proposed_slip = old_slip+increment
    proposed_slip[free] = jump[free]
    increment[free] = jump[free]-old_slip[free]
    shear = kt*(jump-proposed_slip)
    shear[free] = 0.
    tangent = np.zeros((len(gap), 2, 2))
    tangent[:, 0, 0] = normal_tangent
    tangent[:, 1, 1] = np.where(yielding, kt*viscous/(kt+viscous), kt)
    tangent[:, 1, 0] = np.where(yielding, np.sign(trial)*kt/(kt+viscous)*strength_derivative, 0.)
    tangent[free, 1, :] = 0.
    normal_energy = .5*kn*np.where(gap < 0, gap**2, (1-damage)*gap**2)
    shear_energy = .5*kt*(jump-proposed_slip)**2
    result = {
        "traction_pa": np.column_stack((normal, shear)),
        "tangent_pa_m": tangent,
        "plastic_slip_m": proposed_slip,
        "cumulative_slip_m": old_cumulative+np.abs(increment),
        "max_opening_m": opening,
        "damage": damage,
        "slip_increment_m": increment,
        "shear_strength_pa": strength,
        "normal_pressure_pa": pressure,
        "friction_work_j_m2": strength*np.abs(increment),
        "viscous_work_j_m2": viscous*increment**2,
        "shear_relaxation_remainder_j_m2": .5*kt*increment**2,
        "fracture_work_j_m2": cohesive_dissipation(opening, params)-cohesive_dissipation(old_opening, params),
        "recoverable_energy_j_m2": normal_energy+shear_energy,
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Contact response overflowed; check separation and parameter scales")
    return result
